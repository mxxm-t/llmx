#pragma once
// The scheduler's sampling threads (docs/SERVER.md, Sampling): a pass's rows are drawn on them beside the scheduler thread, which hands them the rows and waits.
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace server {

// A fixed set of threads that run one job at a time over its indices, the calling thread taking indices too.
// A job's calls touch only what their index names, so the pool decides nothing and any order gives the same result.
class SamplingPool {
public:
    explicit SamplingPool(size_t threads) {
        try {
            for (size_t i = 0; i < threads; ++i) threads_.emplace_back([this] { work(); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~SamplingPool() { stop(); }
    SamplingPool(const SamplingPool&) = delete;
    SamplingPool& operator=(const SamplingPool&) = delete;

    size_t threads() const { return threads_.size(); }

    // job(i) for every i below n, returning once every call has returned; the first exception a call threw is rethrown then, after the other calls have run.
    // A single index runs on the calling thread alone.
    void run(size_t n, const std::function<void(size_t)>& job) {
        if (n < 2 || threads_.empty()) {
            std::exception_ptr first;
            for (size_t i = 0; i < n; ++i) {
                try {
                    job(i);
                } catch (...) {
                    if (!first) first = std::current_exception();
                }
            }
            if (first) std::rethrow_exception(first);
            return;
        }
        std::unique_lock<std::mutex> lk(m_);
        job_ = &job;
        n_ = n;
        next_ = done_ = 0;
        error_ = nullptr;
        work_.notify_all();
        take(lk);
        idle_.wait(lk, [&] { return done_ == n_; });
        job_ = nullptr;
        if (error_) {
            std::exception_ptr e = error_;
            error_ = nullptr;
            std::rethrow_exception(e);
        }
    }

private:
    // Indices of the current job until none is left, each run without the lock; a thread takes an index only while run waits for it, so the job outlives every call.
    void take(std::unique_lock<std::mutex>& lk) {
        while (job_ && next_ < n_) {
            const size_t i = next_++;
            const std::function<void(size_t)>& job = *job_;
            lk.unlock();
            std::exception_ptr e;
            try {
                job(i);
            } catch (...) {
                e = std::current_exception();
            }
            lk.lock();
            if (e && !error_) error_ = e;
            if (++done_ == n_) idle_.notify_all();
        }
    }

    void work() {
        std::unique_lock<std::mutex> lk(m_);
        for (;;) {
            work_.wait(lk, [&] { return stop_ || (job_ && next_ < n_); });
            if (stop_) return;
            take(lk);
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        work_.notify_all();
        for (auto& t : threads_) t.join();
        threads_.clear();
    }

    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable work_, idle_;
    const std::function<void(size_t)>* job_ = nullptr;   // the job run is waiting on, under m_ like the counts
    size_t n_ = 0, next_ = 0, done_ = 0;
    std::exception_ptr error_;
    bool stop_ = false;
};

} // namespace server
