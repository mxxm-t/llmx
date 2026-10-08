// A set of job threads (core/job_threads.hpp): every index of a job run exactly once, on the pool's threads beside the calling one, an exception rethrown only once every other call has returned, and the pool reused after it.
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/job_threads.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

// n indices on `pool`: each must run exactly once.
void once_each(core::JobThreads& pool, size_t n, const std::string& what) {
    std::unique_ptr<std::atomic<int>[]> runs(new std::atomic<int>[n ? n : 1]);
    for (size_t i = 0; i < n; ++i) runs[i] = 0;
    pool.run(n, [&](size_t i) { ++runs[i]; });
    for (size_t i = 0; i < n; ++i) require(runs[i] == 1, what + ": index " + std::to_string(i) + " of " + std::to_string(n) + " ran " + std::to_string(runs[i]) + " times");
}

// Every thread of the pool and the calling one take an index at once: each of the first threads + 1 calls waits until all of them have begun, which only threads running side by side can meet.
void side_by_side(core::JobThreads& pool) {
    const size_t k = pool.threads() + 1;
    std::atomic<size_t> begun{0};
    std::atomic<bool> met{true};
    std::vector<std::thread::id> ids(k);
    pool.run(k, [&](size_t i) {
        ids[i] = std::this_thread::get_id();
        ++begun;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (begun.load() < k)
            if (std::chrono::steady_clock::now() > until) { met = false; return; }
            else std::this_thread::yield();
    });
    require(met, std::to_string(k) + " calls on a pool of " + std::to_string(pool.threads()) + " threads did not all run at once");
    require(std::set<std::thread::id>(ids.begin(), ids.end()).size() == k, "two calls that ran at once shared a thread");
}

// Calls that throw: run rethrows one of their exceptions, and only once every call has returned, the throwing ones included; the pool then runs the next job whole.
void throwing(core::JobThreads& pool) {
    const size_t n = 64;
    std::atomic<size_t> returned{0}, ran{0};
    bool caught = false;
    try {
        pool.run(n, [&](size_t i) {
            ++ran;
            std::this_thread::sleep_for(std::chrono::microseconds(i % 7 * 50));
            ++returned;
            if (i == 3 || i == 40) throw std::runtime_error("call " + std::to_string(i));
        });
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "call 3" || std::string(e.what()) == "call 40";
        require(returned == n, "run rethrew with " + std::to_string(n - returned) + " calls still running");
    }
    require(caught, "a call's exception was not rethrown");
    require(ran == n, "calls after one that threw did not run: " + std::to_string(ran) + " of " + std::to_string(n));
    once_each(pool, 100, "the job after one that threw");
}

} // namespace

int main() {
    try {
        for (size_t threads : {0, 1, 3, 4}) {
            const std::string what = "a pool of " + std::to_string(threads) + " threads";
            core::JobThreads pool(threads);
            require(pool.threads() == threads, what + " holds " + std::to_string(pool.threads()));
            once_each(pool, 0, what);
            std::thread::id on;
            pool.run(1, [&](size_t) { on = std::this_thread::get_id(); });
            require(on == std::this_thread::get_id(), what + ": a single index ran off the calling thread");
            for (size_t n : {2, 5, 64, 1000}) once_each(pool, n, what);
            side_by_side(pool);
            throwing(pool);
            // Jobs back to back, the size of a pass's rows, so a thread still leaving one job meets the next.
            for (size_t r = 0; r < 2000; ++r) once_each(pool, 1 + r % 17, what + ", job " + std::to_string(r));
        }
        std::cout << "job-threads: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "job-threads: " << e.what() << '\n';
        return 1;
    }
}
