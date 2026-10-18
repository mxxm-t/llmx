#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "core/host_memory.hpp"

// The one ring a file is streamed through to device storage, which the loader drives blocking and the server steps on its own thread (docs/src/inference-read_ring.md).

namespace infer {

// Reads `count` pieces, in order, on up to `readers` threads into up to `slots` slots of host pages, a piece a slot, and gives each full slot to its one consumer in piece order.
// The readers run only `fill`, a file read and whatever check of its bytes the caller makes; what a slot's bytes become is the consumer's, which gives the slot back once its copies out of it have retired.
class ReadRing {
public:
    // Reads piece `piece` into `into`, a slot, or elsewhere (a direct load's own copy), and returns the bytes it read; a throw fails the stream.
    using Fill = std::function<size_t(size_t piece, uint8_t* into)>;
    struct Full {
        size_t piece = 0, slot = 0, bytes = 0;
    };

    // The slots' pages are made here and the readers only at start(), so a caller makes its views of the pages, which may fail, before any reader runs.
    ReadRing(size_t count, size_t slot_bytes, Fill fill, size_t slots = 4, size_t readers = 2) : count_(count), fill_(std::move(fill)), readers_(readers) {
        for (size_t i = 0; i < std::min(slots, count); ++i) ring_.emplace_back(slot_bytes);
        ready_.assign(ring_.size(), 0);
        got_.assign(ring_.size(), 0);
    }
    // Starts the readers, once.
    void start() {
        const size_t threads = std::min(readers_, ring_.size());
        {
            std::lock_guard<std::mutex> lock(mu_);
            reading_.assign(threads, 0);
            running_ = threads;
        }
        pool_.reserve(threads);
        // A reader that fails to start stops and joins those already started before the throw leaves, since a joinable thread must not be destroyed.
        try {
            for (size_t j = 0; j < threads; ++j) pool_.emplace_back([this, j, threads] { read(j, threads); });
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mu_);
                running_ -= threads - pool_.size();
            }
            stop();
            for (std::thread& t : pool_) t.join();
            pool_.clear();
            throw;
        }
    }
    ReadRing(const ReadRing&) = delete;
    ReadRing& operator=(const ReadRing&) = delete;
    // Stops the readers and joins them: a caller that must not wait stops the ring first and keeps it until idle().
    ~ReadRing() {
        stop();
        for (std::thread& t : pool_) t.join();
    }

    size_t slots() const { return ring_.size(); }
    core::HostPages& pages(size_t slot) { return ring_[slot]; }

    // The next piece once its slot is full, waiting for it when `wait`, else nothing while it is not; a reader's failure is thrown here, and nothing is left after the last piece.
    std::optional<Full> next(bool wait) {
        if (next_ >= count_) return std::nullopt;
        const size_t slot = next_ % ring_.size();
        std::unique_lock<std::mutex> lock(mu_);
        if (wait) cv_.wait(lock, [&] { return failed_ || ready_[slot] == next_ + 1; });
        if (ready_[slot] != next_ + 1) {
            if (failed_) std::rethrow_exception(failed_);
            return std::nullopt;
        }
        return Full{next_++, slot, got_[slot]};
    }
    // Whether next(false) has something to give: the next piece's slot is full, or a reader failed.
    bool ready() const {
        if (next_ >= count_) return false;
        std::lock_guard<std::mutex> lock(mu_);
        return failed_ || ready_[next_ % ring_.size()] == next_ + 1;
    }
    // Every piece up to `piece` has left its slot, which the readers may fill again; pieces are given back in order.
    void free(size_t piece) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            consumed_ = std::max(consumed_, piece + 1);
        }
        cv_.notify_all();
    }
    // Asks the readers to stop after the read each is in, without waiting for them.
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
    }
    // Whether every reader has ended, so the ring and the pages its copies read can go with no wait.
    bool idle() const {
        std::lock_guard<std::mutex> lock(mu_);
        return running_ == 0;
    }
    // The busiest reader's time in reads, in seconds.
    double reading() const {
        std::lock_guard<std::mutex> lock(mu_);
        return reading_.empty() ? 0 : *std::max_element(reading_.begin(), reading_.end());
    }

private:
    void read(size_t j, size_t threads) {
        try {
            for (size_t k = j; k < count_; k += threads) {
                {
                    std::unique_lock<std::mutex> lock(mu_);
                    cv_.wait(lock, [&] { return stop_ || k < consumed_ + ring_.size(); });
                    if (stop_) break;
                }
                const auto t0 = std::chrono::steady_clock::now();
                const size_t n = fill_(k, ring_[k % ring_.size()].data());
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                {
                    std::lock_guard<std::mutex> lock(mu_);
                    got_[k % ring_.size()] = n;
                    ready_[k % ring_.size()] = k + 1;
                    reading_[j] += dt;
                }
                cv_.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu_);
            if (!failed_) failed_ = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            --running_;
        }
        cv_.notify_all();
    }

    const size_t count_;
    const Fill fill_;
    const size_t readers_;
    std::vector<core::HostPages> ring_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    size_t next_ = 0;                 // the consumer's: the next piece it takes
    size_t consumed_ = 0;             // pieces whose slots are free again
    std::vector<size_t> ready_;       // per slot, 1 + the piece read into it, 0 while none is
    std::vector<size_t> got_;         // per slot, the bytes its piece read
    std::vector<double> reading_;
    size_t running_ = 0;
    bool stop_ = false;
    std::exception_ptr failed_;
    std::vector<std::thread> pool_;
};

} // namespace infer
