#pragma once
// A thread for each stage whose recording waits on its devices (Model::stage_waits), the one owner of such threads for a pipelined prompt (Model::run_chunks) and the server's rounds alike.
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace infer {

// A lane's thread steps its policy while ready holds and otherwise sleeps under the caller's mutex for as long as idle allows; its policy's state lives under that mutex.
// Destruction stops every lane and joins its thread, so the lanes go before anything they record on.
class StageLanes {
public:
    // Each is called on the lane's thread with the caller's mutex held; step and idle may unlock it and must hold it again when they return.
    struct Policy {
        std::function<bool()> ready;
        std::function<void(std::unique_lock<std::mutex>&)> step;
        std::function<std::optional<std::chrono::milliseconds>(std::unique_lock<std::mutex>&)> idle;   // how long the lane may sleep, none for until woken
    };
    struct Lane {
        std::mutex* m = nullptr;
        std::condition_variable cv;   // the lane's wake, which its policy waits on too
        bool stop = false;            // under *m, set as the lanes go
        Policy policy;
        std::thread thread;
    };

    StageLanes() = default;
    StageLanes(const StageLanes&) = delete;
    StageLanes& operator=(const StageLanes&) = delete;
    ~StageLanes() {
        for (auto& l : lanes_) {
            if (!l) continue;
            {
                std::lock_guard<std::mutex> lk(*l->m);
                l->stop = true;
            }
            l->cv.notify_all();
        }
        for (auto& l : lanes_)
            if (l && l->thread.joinable()) l->thread.join();
    }

    // Stage s's lane under the caller's mutex `m`, its thread not yet started, so the caller can tie its state to the lane's wake and stop flag first.
    Lane& add(size_t s, std::mutex& m) {
        if (lanes_.size() <= s) lanes_.resize(s + 1);
        lanes_[s] = std::make_unique<Lane>();
        lanes_[s]->m = &m;
        return *lanes_[s];
    }
    void start(size_t s, Policy policy) {
        Lane& l = *lanes_.at(s);
        l.policy = std::move(policy);
        l.thread = std::thread([&l] { run(l); });
    }
    void notify(size_t s) { lanes_.at(s)->cv.notify_all(); }

private:
    static void run(Lane& l) {
        std::unique_lock<std::mutex> lk(*l.m);
        for (;;) {
            std::optional<std::chrono::milliseconds> sleep;
            if (l.policy.idle) sleep = l.policy.idle(lk);
            if (l.stop) return;
            if (l.policy.ready()) {
                l.policy.step(lk);
                continue;
            }
            if (sleep) l.cv.wait_for(lk, *sleep);
            else l.cv.wait(lk);
        }
    }

    std::vector<std::unique_ptr<Lane>> lanes_;
};

}  // namespace infer
