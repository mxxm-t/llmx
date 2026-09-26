// Multi-device measurements through the runtime's own Vulkan backend: phase 0 of docs/MULTI-DEVICE.md, what several devices in one process cost each other and what a split pays for moving data between them, and step 0 of the layer split's phase 3 in docs/STATUS.md, how a pipeline's drivers keep its devices fed and what each stage of a split model costs.
// Every mode but `stages` uses the Backend interface as a split would (adopt, matmul, copy into host-visible memory, submit, wait, write), so the numbers include the backend's own submission and waiting.
// `concurrent D...`: each device runs decode-shaped passes on its own thread, first alone, then all at once.
// `pipeline D0 D1 [D...] [options]`: a stage per device with P passes in flight, driven by a thread per stage or by one thread in completion order or in the round's order; each pass leaves a stage through host memory into the next, and the host holds it for a sampling time before it re-enters stage 0.
// `stages MODEL D... [options]`: a model split by layers over the devices as the CLI places it, and each stage's device time a decode pass, and a prefill pass when asked, from GPU timestamps.
// `groupsum D...`: every device's vector summed on the host in device order and written back to each, for a decode row and a 512-row chunk.
// `exchange D... [-- tokens skew]`: a mixture-of-experts layer's dispatch and return between ranks, each rank's entries sent to the ranks holding their experts and the results sent back.
// The synthetic modes use the shapes of a 5120-wide dense model (Qwen3-32B) and Qwen3-235B-A22B's experts (4096 wide, 128 experts of 1536, 8 per token); `stages` reads its model file.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "bench_weights.hpp"
#include "inference/load.hpp"
#include "quant/types.hpp"

namespace {

using backend::Backend;
using backend::BufferPtr;
using backend::Memory;
using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double ms_since(Clock::time_point t0) {
    return ms_between(t0, Clock::now());
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

const size_t kEmbd = 5120, kFF = 25600, kMaxRows = 512;
const uint32_t kQ8 = quant::GGML_TYPE_Q8_0;

// The split nearest `want` among those that cut both projections into equal slices of at least 80 rows.
size_t nearest_split(double want) {
    static const size_t kSplits[] = {1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64};
    size_t best = 1;
    for (size_t c : kSplits)
        if (std::fabs(std::log((double)c / want)) < std::fabs(std::log((double)best / want))) best = c;
    return best;
}

// One device's share of a dense model: a layer's two large projections, repeated to reach a target time per pass, each projection recorded as `split` calls over slices of its rows.
struct Stage {
    backend::BackendPtr b;
    BufferPtr up, down, x, h, y, out;
    std::vector<BufferPtr> outs;   // the pipeline's host-visible output per pass, so a pass recorded behind another leaves the other's output for the host to read
    size_t layers = 1, split = 1;

    explicit Stage(int device) {
        b = backend::make_vulkan_backend(device);
        auto wu = weights(row_bytes(kQ8, kEmbd) * kFF, (uint32_t)device * 7 + 1);
        auto wd = weights(row_bytes(kQ8, kFF) * kEmbd, (uint32_t)device * 7 + 2);
        up = b->adopt(wu.data(), wu.size());
        down = b->adopt(wd.data(), wd.size());
        x = b->alloc(kMaxRows * kEmbd * 4);
        h = b->alloc(kMaxRows * kFF * 4);
        y = b->alloc(kMaxRows * kEmbd * 4);
        out = b->alloc(kMaxRows * kEmbd * 4, Memory::host_visible);
    }

    // A pass over `rows` rows: `layers` times up and down, the residual copied into `dst`, submitted.
    // Each slice writes a region of its own, so the calls of one projection never write the same memory.
    // With `calls_ms`, each call's time on the host is appended to it and `submit_ms` gets the final submission's.
    backend::Ticket run(size_t rows, backend::Buffer& dst, std::vector<double>* calls_ms = nullptr, double* submit_ms = nullptr) {
        auto call = [&](const auto& f) {
            if (!calls_ms) return f();
            const auto t0 = Clock::now();
            f();
            calls_ms->push_back(ms_since(t0));
        };
        const size_t nu = kFF / split, nd = kEmbd / split;
        for (size_t l = 0; l < layers; ++l) {
            for (size_t i = 0; i < split; ++i)
                call([&] { b->matmul(kQ8, {up.get(), i * nu * row_bytes(kQ8, kEmbd) / 4}, {x.get(), 0}, {h.get(), i * nu * rows}, kEmbd, nu, rows); });
            for (size_t i = 0; i < split; ++i)
                call([&] { b->matmul_add(kQ8, {down.get(), i * nd * row_bytes(kQ8, kFF) / 4}, {h.get(), 0}, {x.get(), i * nd * rows}, kFF, nd, rows); });
        }
        call([&] { b->copy(dst, 0, *x, 0, rows * kEmbd * 4); });
        const auto t0 = Clock::now();
        const backend::Ticket t = b->submit();
        if (submit_ms) *submit_ms = ms_since(t0);
        return t;
    }
    backend::Ticket run(size_t rows) { return run(rows, *out); }

    // Backend calls a pass records: the projections' slices and the copy out.
    size_t calls() const { return 2 * split * layers + 1; }

    // Layers per pass so a pass of `rows` rows takes about `ms`; with `calls`, each projection is first split so a pass records about that many calls.
    void calibrate(double ms, size_t rows, size_t calls) {
        auto pass_ms = [&] {
            for (int i = 0; i < 3; ++i) b->wait(run(rows));
            const auto t0 = Clock::now();
            for (int i = 0; i < 10; ++i) b->wait(run(rows));
            return ms_since(t0) / 10;
        };
        split = 1;
        layers = 4;
        double per = pass_ms() / (double)layers;
        // A call takes per / (2 split), and splitting by the ratio of that to ms / calls would bring it there if its time fell as its slice does; small slices lose more, so the split is refined a few times.
        for (int k = 0; calls > 1 && k < 4; ++k) {
            const size_t was = split, next = nearest_split(per / 2 / (ms / (double)calls));
            if (next == was) break;
            split = next;
            const double now = pass_ms() / (double)layers;
            // A larger split whose calls take no less time each only adds dispatch overhead, so the refinement stops before it.
            if (next > was && now / (double)next >= per / (double)was) {
                split = was;
                break;
            }
            per = now;
        }
        layers = std::max<size_t>(1, (size_t)(ms / per + 0.5));
        // One correction at the size chosen, since a short measurement on an idle device or a busy host can misjudge a layer.
        layers = std::max<size_t>(1, (size_t)((double)layers * ms / pass_ms() + 0.5));
    }
};

// The command ring as the running program finds it: probes of several milliseconds each are queued without waiting, and the first whose recording takes over half a submission's time found every slot busy.
struct Ring {
    int depth = 0;             // submissions in flight before recording the next blocks, 0 when not found
    int at_least = 0;          // without a depth, the most submissions that went in without blocking
    bool whole = true;         // false when one matmul alone took more than one submission
    double submission_ms = 0;  // one probe's time on the device
};

Ring ring_depth(Stage& s) {
    Ring r;
    auto probe = [&](int m) {
        for (int i = 0; i < m; ++i) s.b->matmul(kQ8, {s.up.get(), 0}, {s.x.get(), 0}, {s.h.get(), 0}, kEmbd, kFF, kMaxRows);
        return s.b->submit();
    };
    // One probe of m matmuls timed alone, or a negative time when it took more than one submission.
    auto alone = [&](int m) {
        s.b->sync();
        const backend::Ticket a = probe(m);
        s.b->wait(a);
        const auto t0 = Clock::now();
        const backend::Ticket b = probe(m);
        s.b->wait(b);
        const double t = ms_since(t0);
        return b - a == 1 ? t : -1.0;
    };
    int m = 1;
    r.submission_ms = alone(m);
    if (r.submission_ms < 0) {
        r.whole = false;
        return r;
    }
    while (r.submission_ms < 8 && m < 16) {
        const double t = alone(2 * m);
        if (t < 0) break;
        m *= 2;
        r.submission_ms = t;
    }
    // Once the ring is full every later submission blocks too, so a block counts only when the next one blocks as well.
    // Until the first block, every submission must be recorded within a quarter of a submission's time of the first, or the device may have retired one and let one more in.
    // An attempt that runs past that is repeated with probes twice as long, since a deep ring takes more probes than a slow host records in the quarter.
    for (int attempt = 0; attempt < 8; ++attempt) {
        s.b->sync();
        const auto start = Clock::now();
        int blocked_at = -1, queued = 0;
        bool late = false;
        for (; queued < 64; ++queued) {
            late = blocked_at < 0 && ms_since(start) > r.submission_ms / 4;
            if (late) break;
            const auto t0 = Clock::now();
            probe(m);
            const bool blocked = ms_since(t0) > r.submission_ms / 2;
            if (blocked && blocked_at >= 0) {
                s.b->sync();
                r.depth = blocked_at;
                return r;
            }
            blocked_at = blocked ? queued : -1;
        }
        r.at_least = std::max(r.at_least, queued);
        if (!late) break;
        if (r.submission_ms < 64) {
            const double t = alone(2 * m);
            if (t > 0) {
                m *= 2;
                r.submission_ms = t;
            }
        }
    }
    s.b->sync();
    return r;
}

int concurrent(const std::vector<int>& devices) {
    std::vector<std::unique_ptr<Stage>> st;
    for (int d : devices) {
        st.push_back(std::make_unique<Stage>(d));
        st.back()->layers = 16;
    }
    const int passes = 200;
    auto one = [&](size_t i, size_t rows) {
        auto& s = *st[i];
        for (int k = 0; k < 3; ++k) s.b->wait(s.run(rows));
        const auto t0 = Clock::now();
        for (int k = 0; k < passes; ++k) s.b->wait(s.run(rows));
        return ms_since(t0) / passes;
    };
    for (size_t rows : {(size_t)1, (size_t)64}) {
        std::printf("\n%zu rows a pass, 16 layers of up and down (%.0f MB of weights read a pass)\n", rows, 16 * 2.0 * row_bytes(kQ8, kEmbd) * kFF / 1e6);
        std::vector<double> alone(st.size()), together(st.size());
        for (size_t i = 0; i < st.size(); ++i) alone[i] = one(i, rows);
        std::vector<std::thread> th;
        for (size_t i = 0; i < st.size(); ++i) th.emplace_back([&, i] { together[i] = one(i, rows); });
        for (auto& t : th) t.join();
        for (size_t i = 0; i < st.size(); ++i)
            std::printf("  device %d: alone %.3f ms a pass, together %.3f (%+.1f%%)\n", devices[i], alone[i], together[i], 100 * (together[i] / alone[i] - 1));
    }
    return 0;
}

// A queue between threads.
template <class T>
struct Channel {
    std::mutex m;
    std::condition_variable cv;
    std::deque<T> q;
    bool closed = false;
    void put(T v) {
        {
            std::lock_guard<std::mutex> l(m);
            q.push_back(std::move(v));
        }
        cv.notify_one();
    }
    bool get(T& v) {
        std::unique_lock<std::mutex> l(m);
        cv.wait(l, [&] { return !q.empty() || closed; });
        if (q.empty()) return false;
        v = std::move(q.front());
        q.pop_front();
        return true;
    }
    void close() {
        {
            std::lock_guard<std::mutex> l(m);
            closed = true;
        }
        cv.notify_all();
    }
};

struct Pass {
    int id = 0;
    std::vector<float> residual;
    Clock::time_point started;
};

Pass fresh_pass(size_t id, size_t rows) {
    Pass p;
    p.id = (int)id;
    p.residual.assign(rows * kEmbd, 0.01f);
    p.started = Clock::now();
    return p;
}

// The host's sampling of a pass and assembly of the next, as busy time on the calling thread.
void spin(double ms) {
    const auto s = Clock::now();
    while (ms_since(s) < ms) {}
}

// Passes on the devices, recorded and not yet seen complete by the host, as steps in time.
// The host sees a completion only when it waits on one, so a driver that waits late counts a pass on its device for longer than the device holds it.
struct Occupancy {
    std::mutex m;
    std::vector<std::pair<Clock::time_point, int>> steps;

    void add(int change, Clock::time_point at = Clock::now()) {
        std::lock_guard<std::mutex> l(m);
        steps.emplace_back(at, change);
    }

    // The mean over [from, until] weighted by time, and the most at any moment in it.
    std::pair<double, int> over(Clock::time_point from, Clock::time_point until) {
        std::lock_guard<std::mutex> l(m);
        std::sort(steps.begin(), steps.end());
        int n = 0, most = 0;
        double area = 0;
        Clock::time_point at = from;
        for (const auto& s : steps) {
            if (s.first > until) break;
            if (s.first > from) {
                most = std::max(most, n);
                area += n * ms_between(at, s.first);
                at = s.first;
            }
            n += s.second;
        }
        most = std::max(most, n);
        area += n * ms_between(at, until);
        return {area / ms_between(from, until), most};
    }
};

// Where one driving thread's time went over the measured passes, in milliseconds.
struct ThreadTime {
    double waiting = 0;                 // blocked on a ticket, or for a completion
    double write = 0, write_free = 0;   // writing passes in, and what the same writes take when nothing blocks
    double run = 0, run_free = 0;       // recording and submitting passes, and what the same recordings take when nothing blocks
    double copy = 0, sample = 0;        // copying passes out of host-visible memory, and sampling
};

struct Result {
    double rate = 0, latency = 0;   // passes a second, and the median time from entering stage 0 to leaving the last
    size_t circulating = 0;         // distinct passes seen leaving the last stage
    double on_mean = 0;             // passes on the devices (Occupancy), the mean and the most
    int on_most = 0;
    double window = 0;              // the measured passes' span, in milliseconds
    bool one_thread = false;        // whether `thread` holds the driving thread's time
    ThreadTime thread;
    double hop = 0;                 // completion order: the mean delay from a waiter's wait returning to the idle driving thread taking the completion
    size_t hops = 0;
};

// What the host sees of the passes leaving the last stage: from the 3P-th on, each one's time since it entered stage 0 and its id, until 3 seconds have passed and 20 are measured.
struct Tally {
    explicit Tally(size_t passes) : P(passes) {}
    size_t P, left = 0;
    Clock::time_point begin = Clock::now(), from{}, until{};
    std::vector<double> latency;
    std::set<int> ids;

    bool measuring() const { return left >= 3 * P; }

    // Counts a pass leaving the last stage; false once the run has measured enough.
    bool leave(const Pass& p) {
        if (++left == 3 * P) from = Clock::now();
        if (left > 3 * P) {
            latency.push_back(ms_since(p.started));
            ids.insert(p.id);
        }
        if (ms_since(begin) > 3000 && latency.size() >= 20) {
            until = Clock::now();
            return false;
        }
        return true;
    }

    Result result(Occupancy& occ) {
        Result r;
        r.window = ms_between(from, until);
        r.rate = (double)latency.size() / (r.window / 1000);
        r.latency = median(latency);
        r.circulating = ids.size();
        const auto on = occ.over(from, until);
        r.on_mean = on.first;
        r.on_most = on.second;
        return r;
    }
};

// Adds the time `f` takes to `acc` while the tally measures.
template <class F>
void timed(const Tally& tally, double& acc, const F& f) {
    const auto t0 = Clock::now();
    f();
    if (tally.measuring()) acc += ms_since(t0);
}

// What writing a pass in and recording it take on a stage when nothing blocks.
struct FreeCost {
    double write = 0, run = 0;
};

// Both the median of 11 on an idle device: a write alone, and a recording as its calls at their mean and its submissions at the final one's.
// On an idle device only a call that submits or opens a command buffer can wait, for a slot the same pass filled, so each recording's slowest calls, two for each submission, are left out of its mean.
FreeCost free_cost(Stage& s, size_t rows) {
    std::vector<float> r(rows * kEmbd, 0.01f);
    std::vector<double> writes, runs;
    for (int k = 0; k < 11; ++k) {
        s.b->sync();
        const auto t0 = Clock::now();
        s.b->write(*s.x, 0, r.data(), r.size() * 4);
        writes.push_back(ms_since(t0));
        const backend::Ticket before = s.b->submit();
        s.b->wait(before);
        std::vector<double> each;
        double submit = 0;
        const backend::Ticket t = s.run(rows, *s.outs[0], &each, &submit);
        const size_t submissions = (size_t)(t - before);
        s.b->wait(t);
        std::sort(each.begin(), each.end());
        const size_t keep = each.size() > 2 * submissions ? each.size() - 2 * submissions : 1;
        double kept = 0;
        for (size_t i = 0; i < keep; ++i) kept += each[i];
        runs.push_back(kept * (double)each.size() / (double)keep + (double)submissions * submit);
    }
    return {median(writes), median(runs)};
}

// What every driver runs on: the stages, the rows a pass, the host's sampling time a pass and each stage's costs when nothing blocks.
struct Setup {
    std::vector<std::unique_ptr<Stage>>& st;
    size_t rows;
    double sample_ms;
    std::vector<FreeCost> free_cost;
};

// Writes pass `p` into stage s and records it, the time each takes counted in `tt`; returns its ticket.
backend::Ticket record_pass(const Setup& u, size_t s, const Pass& p, const Tally& tally, ThreadTime& tt) {
    Stage& g = *u.st[s];
    timed(tally, tt.write, [&] { g.b->write(*g.x, 0, p.residual.data(), p.residual.size() * 4); });
    backend::Ticket t = 0;
    timed(tally, tt.run, [&] { t = g.run(u.rows, *g.outs[p.id]); });
    if (tally.measuring()) {
        tt.write_free += u.free_cost[s].write;
        tt.run_free += u.free_cost[s].run;
    }
    return t;
}

// Copies pass `p` out of stage s's output once the host has waited on it.
void copy_out(const Setup& u, size_t s, Pass& p, const Tally& tally, ThreadTime& tt) {
    timed(tally, tt.copy, [&] { std::memcpy(p.residual.data(), u.st[s]->outs[p.id]->host_ptr(), p.residual.size() * 4); });
}

// A thread per stage: each takes its next pass, writes it in, runs it, waits for it and hands it to the next stage, so a device gets a pass only once it has finished the last; the calling thread samples what leaves the last stage.
Result per_stage(const Setup& u, size_t P) {
    const size_t S = u.st.size();
    std::vector<Channel<Pass>> ch(S + 1);
    Occupancy occ;
    auto stage = [&](Stage& s, Channel<Pass>& in, Channel<Pass>& out) {
        Pass p;
        while (in.get(p)) {
            s.b->write(*s.x, 0, p.residual.data(), p.residual.size() * 4);
            const backend::Ticket t = s.run(u.rows, *s.outs[p.id]);
            occ.add(+1);
            s.b->wait(t);
            occ.add(-1);
            std::memcpy(p.residual.data(), s.outs[p.id]->host_ptr(), p.residual.size() * 4);
            out.put(std::move(p));
        }
    };
    std::vector<std::thread> th;
    for (size_t i = 0; i < S; ++i) th.emplace_back([&, i] { stage(*u.st[i], ch[i], ch[i + 1]); });
    for (size_t i = 0; i < P; ++i) ch[0].put(fresh_pass(i, u.rows));
    Tally tally(P);
    Pass p;
    while (ch[S].get(p)) {
        if (!tally.leave(p)) break;
        spin(u.sample_ms);
        p.started = Clock::now();
        ch[0].put(std::move(p));
    }
    for (auto& c : ch) c.close();
    for (auto& t : th) t.join();
    return tally.result(occ);
}

// One thread records, relays and samples for every stage, serving the stages in the order their passes complete; a device gets a pass only once it has finished the last, as with a thread per stage.
// A waiter per stage only blocks on that stage's tickets and hands each completion over with the time its wait returned; it touches the backend only while the driving thread leaves it alone.
// So each completion wakes two threads where a poll on the driving thread would wake one, and the result gives the second wake's mean delay.
Result completion(const Setup& u, size_t P) {
    const size_t S = u.st.size();
    struct Done {
        size_t stage = 0;
        Clock::time_point at;
    };
    std::vector<Channel<backend::Ticket>> tickets(S);
    Channel<Done> completed;
    Occupancy occ;
    std::vector<std::thread> waiters;
    for (size_t i = 0; i < S; ++i)
        waiters.emplace_back([&, i] {
            backend::Ticket t = 0;
            while (tickets[i].get(t)) {
                u.st[i]->b->wait(t);
                const auto at = Clock::now();
                occ.add(-1, at);
                completed.put({i, at});
            }
        });
    Tally tally(P);
    ThreadTime tt;
    double hop = 0;
    size_t hops = 0;
    std::vector<std::deque<Pass>> waiting(S);
    std::vector<Pass> on(S);
    std::vector<bool> busy(S, false);
    // A stage whose device is free takes its oldest waiting pass.
    auto launch = [&](size_t s) {
        if (busy[s] || waiting[s].empty()) return;
        on[s] = std::move(waiting[s].front());
        waiting[s].pop_front();
        const backend::Ticket t = record_pass(u, s, on[s], tally, tt);
        busy[s] = true;
        occ.add(+1);
        tickets[s].put(t);
    };
    for (size_t i = 0; i < P; ++i) waiting[0].push_back(fresh_pass(i, u.rows));
    launch(0);
    Done d;
    for (;;) {
        const auto w = Clock::now();
        completed.get(d);
        const auto got = Clock::now();
        if (tally.measuring()) {
            tt.waiting += ms_between(w, got);
            if (d.at > w) {
                hop += ms_between(d.at, got);
                ++hops;
            }
        }
        const size_t s = d.stage;
        busy[s] = false;
        Pass p = std::move(on[s]);
        copy_out(u, s, p, tally, tt);
        // Every device that can take work gets it before the thread samples.
        if (s + 1 < S) {
            waiting[s + 1].push_back(std::move(p));
            launch(s + 1);
            launch(s);
            continue;
        }
        if (!tally.leave(p)) break;
        launch(s);
        timed(tally, tt.sample, [&] { spin(u.sample_ms); });
        p.started = Clock::now();
        waiting[0].push_back(std::move(p));
        launch(0);
    }
    for (size_t n = (size_t)std::count(busy.begin(), busy.end(), true); n; --n) completed.get(d);
    for (auto& t : tickets) t.close();
    for (auto& t : waiters) t.join();
    Result r = tally.result(occ);
    r.one_thread = true;
    r.thread = tt;
    r.hop = hops ? hop / (double)hops : 0;
    r.hops = hops;
    return r;
}

// One thread in the decided round (docs/STATUS.md, Layer split phase 3, The round).
// From the last stage down to stage 1, each stage takes the oldest pass the stage before it recorded in an earlier round, waits on that pass's ticket there, copies it out and records it; then every pass whose last stage was recorded in an earlier round is waited on, sampled and formed again at stage 0.
// With `ahead`, a pass is recorded onto its device whether or not the device is still busy, as the decided design does; without it, the thread first waits for the device to finish its last pass.
Result in_rounds(const Setup& u, size_t P, bool ahead) {
    const size_t S = u.st.size();
    struct Flight {
        Pass p;
        backend::Ticket t = 0;
    };
    std::vector<std::deque<Flight>> recorded(S);          // each stage's passes, oldest first, that the next stage or retirement has not taken
    std::vector<std::deque<backend::Ticket>> running(S);  // each stage's tickets the host has not yet seen complete
    std::vector<backend::Ticket> latest(S, 0);
    std::deque<Pass> ready;
    for (size_t i = 0; i < P; ++i) ready.push_back(fresh_pass(i, u.rows));
    Tally tally(P);
    Occupancy occ;
    ThreadTime tt;
    // A wait on stage s, after which the host knows every pass recorded there up to ticket t complete.
    auto wait = [&](size_t s, backend::Ticket t) {
        timed(tally, tt.waiting, [&] { u.st[s]->b->wait(t); });
        for (auto& q = running[s]; !q.empty() && q.front() <= t; q.pop_front()) occ.add(-1);
    };
    auto record = [&](size_t s, Pass p) {
        if (!ahead) wait(s, latest[s]);
        Flight f{std::move(p), 0};
        f.t = record_pass(u, s, f.p, tally, tt);
        latest[s] = f.t;
        running[s].push_back(f.t);
        occ.add(+1);
        recorded[s].push_back(std::move(f));
    };
    auto take = [&](size_t s) {
        Flight f = std::move(recorded[s].front());
        recorded[s].pop_front();
        wait(s, f.t);
        copy_out(u, s, f.p, tally, tt);
        return std::move(f.p);
    };
    for (bool going = true; going;) {
        const size_t due = recorded[S - 1].size();
        for (size_t s = S - 1; s >= 1; --s)
            if (!recorded[s - 1].empty()) record(s, take(s - 1));
        for (size_t k = 0; k < due && going; ++k) {
            Pass p = take(S - 1);
            going = tally.leave(p);
            if (!going) break;
            timed(tally, tt.sample, [&] { spin(u.sample_ms); });
            p.started = Clock::now();
            ready.push_back(std::move(p));
        }
        // P passes exist, so every one the host holds is formed again.
        for (; going && !ready.empty(); ready.pop_front()) record(0, std::move(ready.front()));
    }
    for (auto& s : u.st) s->b->sync();
    Result r = tally.result(occ);
    r.one_thread = true;
    r.thread = tt;
    return r;
}

enum Driver { kPerStage, kCompletion, kRound, kRoundIdle, kDrivers };
const char* const kDriverNames[kDrivers] = {"per-stage", "completion", "round", "round-idle"};

Result drive(const Setup& u, Driver d, size_t P) {
    if (d == kPerStage) return per_stage(u, P);
    if (d == kCompletion) return completion(u, P);
    return in_rounds(u, P, d == kRound);
}

struct PipelineOptions {
    std::vector<std::vector<double>> ms{{10.0}};   // one list for every row count or one for each, each list one pass's time on every stage or on each
    std::vector<size_t> rows{1, 8};
    std::vector<size_t> passes;                    // P, by default 1, S, S + 1 and 2S
    std::vector<Driver> drivers{kPerStage, kCompletion, kRound, kRoundIdle};
    double sample_ms = 0.3;
    size_t calls = 0;                              // backend calls a stage records a pass, 0 for one call a projection
};

int pipeline(const std::vector<int>& devices, PipelineOptions o) {
    const size_t S = devices.size();
    std::vector<std::unique_ptr<Stage>> st;
    for (int d : devices) st.push_back(std::make_unique<Stage>(d));
    if (o.passes.empty()) o.passes = {1, S, S + 1, 2 * S};
    std::printf("pipeline over %zu stages, devices", S);
    for (int d : devices) std::printf(" %d", d);
    std::printf("; the host spends %.2f ms sampling each pass that leaves the last stage\n", o.sample_ms);
    const Ring ring = ring_depth(*st[0]);
    if (ring.depth > 0)
        std::printf("command ring: %d submissions in flight before recording the next blocks (device %d, submissions of %.1f ms)\n", ring.depth, devices[0], ring.submission_ms);
    else if (!ring.whole)
        std::printf("command ring: not measured, one matmul took more than one submission (device %d)\n", devices[0]);
    else if (ring.at_least > 0)
        std::printf("command ring: no block found, at least %d submissions went in flight without one (device %d, submissions of %.1f ms)\n", ring.at_least, devices[0], ring.submission_ms);
    else
        std::printf("command ring: not measured, the host was too slow to queue the probes (device %d)\n", devices[0]);
    const size_t widest = *std::max_element(o.rows.begin(), o.rows.end()), most = *std::max_element(o.passes.begin(), o.passes.end());
    for (auto& s : st)
        for (size_t i = 0; i < most; ++i) s->outs.push_back(s->b->alloc(widest * kEmbd * 4, Memory::host_visible));
    std::printf("drivers: per-stage, a thread per stage; completion, one thread serving stages as their passes complete; round, one thread in the round's order; round-idle, the round waiting for each device to finish before recording onto it\n");
    std::printf("onto: idle, a device gets a pass only once it has finished the last; busy, a pass is recorded as soon as its input is on the host\n");
    std::printf("passes: distinct passes seen leaving the last stage; on devices: passes recorded and not yet seen complete by the host, the mean over time and the most\n");
    std::printf("of slowest: passes a second against the slowest stage's rate with its passes recorded ahead, the most a filled pipeline can reach\n");
    std::printf("thread: where one driving thread's time went; held is what recording and writes took beyond their median on an idle device: waits for a command slot or for staging, and any delay a busy host adds\n");
    std::printf("each driver runs twice at each P, in the order listed and then reversed\n");
    for (size_t ri = 0; ri < o.rows.size(); ++ri) {
        const size_t rows = o.rows[ri];
        const std::vector<double>& ms = o.ms[o.ms.size() == 1 ? 0 : ri];
        for (size_t s = 0; s < S; ++s) st[s]->calibrate(ms[ms.size() == 1 ? 0 : s], rows, o.calls);
        // One pass through every stage alone, the time a single stream sees per token.
        double serial = 0;
        {
            std::vector<float> r(rows * kEmbd, 0.01f);
            for (int k = 0; k < 13; ++k) {
                const auto t0 = Clock::now();
                for (auto& s : st) {
                    s->b->write(*s->x, 0, r.data(), r.size() * 4);
                    s->b->wait(s->run(rows));
                    std::memcpy(r.data(), s->out->host_ptr(), r.size() * 4);
                }
                if (k >= 3) serial += ms_since(t0) / 10;
            }
        }
        // A stage's passes one after another, each written in, recorded and copied out, either with the next recorded while the last runs or with the host waiting on each before the next.
        // Recorded ahead, the device never idles, which gives the rate a filled pipeline is bounded by; waited on, the device idles while the host records.
        // The tickets count the submissions a pass takes.
        auto back_to_back = [&](Stage& s, bool ahead, double& per_pass) {
            std::vector<float> r(rows * kEmbd, 0.01f);
            backend::Buffer* out[2] = {s.out.get(), s.outs[0].get()};
            for (int k = 0; k < 3; ++k) s.b->wait(s.run(rows));
            backend::Ticket first = 0, last = 0;
            const auto t0 = Clock::now();
            for (int k = 0; k < 20; ++k) {
                s.b->write(*s.x, 0, r.data(), r.size() * 4);
                const backend::Ticket t = s.run(rows, *out[k % 2]);
                if (!k) first = t;
                if (ahead && k) {
                    s.b->wait(last);
                    std::memcpy(r.data(), out[(k - 1) % 2]->host_ptr(), r.size() * 4);
                }
                if (!ahead) {
                    s.b->wait(t);
                    std::memcpy(r.data(), out[k % 2]->host_ptr(), r.size() * 4);
                }
                last = t;
            }
            s.b->wait(last);
            per_pass = ms_since(t0) / 20;
            return double(last - first) / 19;
        };
        std::printf("\n%zu row%s a pass:\n", rows, rows == 1 ? "" : "s");
        Setup u{st, rows, o.sample_ms, {}};
        double slowest = 0;
        for (size_t i = 0; i < S; ++i) {
            Stage& s = *st[i];
            double busy = 0, waited = 0;
            const double submissions = back_to_back(s, true, busy);
            back_to_back(s, false, waited);
            slowest = std::max(slowest, busy);
            u.free_cost.push_back(free_cost(s, rows));
            char asked[48] = "";
            if (o.calls && std::fabs((double)s.calls() - (double)o.calls) > 0.1 * (double)o.calls) std::snprintf(asked, sizeof asked, " of the %zu asked", o.calls);
            std::printf("  stage %zu, device %d: %zu layers, %zu call%s a projection, %zu calls%s in %.1f submissions a pass; %.2f ms a pass recorded ahead and %.2f ms waited on one at a time; written in %.3f ms and recorded in %.3f ms on an idle device\n",
                        i, devices[i], s.layers, s.split, s.split == 1 ? "" : "s", s.calls(), asked, submissions, busy, waited, u.free_cost[i].write, u.free_cost[i].run);
        }
        std::printf("  one pass through every stage alone: %.2f ms\n", serial);
        std::printf("  %-11s %-5s %6s %12s %9s %10s %10s\n", "driver", "onto", "passes", "on devices", "passes/s", "ms a pass", "of slowest");
        auto row = [&](Driver d, const Result& r) {
            std::printf("  %-11s %-5s %6zu %7.2f / %-2d %9.1f %10.2f %9.0f%%\n", kDriverNames[d], d == kRound ? "busy" : "idle", r.circulating, r.on_mean, r.on_most, r.rate, r.latency,
                        100 * r.rate * slowest / 1000);
            if (!r.one_thread) return;
            const ThreadTime& t = r.thread;
            const double run_held = std::max(0.0, t.run - t.run_free), write_held = std::max(0.0, t.write - t.write_free);
            const double other = std::max(0.0, r.window - t.waiting - t.write - t.run - t.copy - t.sample);
            auto pc = [&](double v) { return 100 * v / r.window; };
            std::printf("  %11s thread working %.0f%% (recording %.0f%%, relaying %.0f%%, sampling %.0f%%, other %.0f%%), blocked %.0f%% (waiting %.0f%%, recording held %.0f%%, writes held %.0f%%)", "",
                        pc(t.run - run_held + t.write - write_held + t.copy + t.sample + other), pc(t.run - run_held), pc(t.write - write_held + t.copy), pc(t.sample), pc(other),
                        pc(t.waiting + run_held + write_held), pc(t.waiting), pc(run_held), pc(write_held));
            if (r.hops) std::printf("; a completion reached it %.3f ms after its wait returned", r.hop);
            std::printf("\n");
        };
        for (size_t P : o.passes)
            for (int reading = 0; reading < 2; ++reading)
                for (size_t k = 0; k < o.drivers.size(); ++k) {
                    const Driver d = o.drivers[reading ? o.drivers.size() - 1 - k : k];
                    row(d, drive(u, d, P));
                }
    }
    return 0;
}

struct StagesOptions {
    std::vector<size_t> rows{1, 8, 16, 32, 64};
    std::vector<size_t> prefill;   // prompt rows of the prefill passes timed, none unless asked
    size_t context = 512, steps = 16;
};

// Layers per device from a split's plan, which gives each device a line as LayerSplit::describe writes it; empty when a line does not read so.
std::vector<int> plan_layers(const std::string& plan, size_t devices) {
    std::vector<int> out;
    std::istringstream in(plan);
    for (std::string line; out.size() < devices && std::getline(in, line);) {
        const size_t at = line.find(": layers ");
        if (at == std::string::npos) {
            if (line.find(": no layers") == std::string::npos) return {};
            out.push_back(0);
            continue;
        }
        char* end = nullptr;
        const long first = std::strtol(line.c_str() + at + 9, &end, 10);
        if (*end != '-') return {};
        const long last = std::strtol(end + 1, nullptr, 10);
        out.push_back((int)(last - first + 1));
    }
    return out.size() == devices ? out : std::vector<int>{};
}

// Decode passes of each row count through the model split over `devices`, every sequence first given `context` tokens, and prefill passes of each prompt row count, each the slice that ends a prompt after `context` tokens, so it wants its last row's logits.
// Each device's time a pass is the sum of its dispatches' GPU timestamps.
// The last stage is set against the others, and where the plan gives the layer counts, what it takes beyond its layers at the others' time a layer is the head's.
int stages(const std::string& path, const std::vector<int>& devices, const StagesOptions& o) {
    std::vector<backend::BackendPtr> backends;
    infer::PlacementRequest request;
    for (int d : devices) {
        backends.push_back(backend::make_vulkan_backend(d, true));
        request.names.push_back("vulkan:" + std::to_string(d));
    }
    const size_t most = *std::max_element(o.rows.begin(), o.rows.end()), warm = 3;
    const size_t widest = o.prefill.empty() ? 0 : *std::max_element(o.prefill.begin(), o.prefill.end());
    request.decode_rows = most;
    request.histories = most;
    request.history_tokens = o.context + std::max(warm + o.steps, widest);
    const auto loaded = infer::load_model(path, backends, request);
    infer::Model& model = *loaded->model;
    std::printf("%s, %zu tokens of context, %zu passes timed after %zu:\n%s", path.c_str(), o.context, o.steps, warm, loaded->plan.c_str());
    const std::vector<int> layers = plan_layers(loaded->plan, devices.size());
    if (layers.empty()) std::printf("plan not read: the layer counts and the head's share are left out\n");
    // Fixed pseudo-random ids below 1000, or below a smaller vocabulary's size, the same in every run.
    const uint32_t vocab = (uint32_t)std::min<size_t>(1000, model.n_vocab());
    std::mt19937 rng(12345);
    auto ids = [&](size_t n) {
        std::vector<uint32_t> v(n);
        for (auto& t : v) t = (uint32_t)(rng() % vocab);
        return v;
    };
    const std::vector<uint32_t> prompt = ids(o.context), gen = ids(warm + o.steps), tail = ids(widest);
    const size_t n = devices.size(), ubatch = model.prefill_batch();
    infer::ExecContext ctx;
    // Tokens [from, to) of the prompt `whole` into q, in slices of the ubatch, each slice given the whole prompt's extent; the slice that ends the prompt wants its last row's logits.
    auto feed = [&](infer::Sequence& q, const std::vector<uint32_t>& whole, size_t from, size_t to) {
        for (size_t at = from; at < to; at += ubatch) {
            const size_t len = std::min(ubatch, to - at);
            infer::BatchEntry e{&q, whole.data() + at, len, at + len == whole.size()};
            e.extent = e.fresh = whole.size();
            model.forward(ctx, &e, 1);
        }
    };
    struct Reading {
        std::vector<double> device, dispatches;   // each device's time and dispatches a pass
        double host = 0;                          // a pass on the host, the stages one after another
    };
    // `o.steps` passes timed after `warm`, each after its `setup`, which is not timed.
    auto time_passes = [&](const auto& setup, const auto& pass) {
        Reading r{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0), 0.0};
        for (size_t g = 0; g < warm + o.steps; ++g) {
            setup(g);
            for (auto& b : backends) backend::vulkan_kernel_times(*b);
            const auto t0 = Clock::now();
            pass(g);
            if (g < warm) continue;
            r.host += ms_since(t0) / (double)o.steps;
            for (size_t d = 0; d < n; ++d) {
                for (const auto& k : backend::vulkan_kernel_times(*backends[d])) r.device[d] += k.second / (double)o.steps;
                r.dispatches[d] += (double)backend::vulkan_timed_dispatches(*backends[d]) / (double)o.steps;
            }
        }
        if (std::all_of(r.dispatches.begin(), r.dispatches.end(), [](double v) { return v == 0; }))
            throw std::runtime_error("no device timestamps its dispatches");
        return r;
    };
    auto report = [&](const Reading& r) {
        auto runs_layers = [&](size_t d) { return layers.empty() ? r.dispatches[d] > 0 : layers[d] > 0; };
        size_t last = n;
        for (size_t d = 0; d < n; ++d)
            if (runs_layers(d)) last = d;
        double others = 0;
        size_t n_others = 0;
        int other_layers = 0;
        for (size_t d = 0; d < n; ++d) {
            std::printf("  %s", request.names[d].c_str());
            if (!layers.empty()) std::printf(", %d layers", layers[d]);
            std::printf(": %.3f ms of device time a pass over %.0f dispatches\n", r.device[d], r.dispatches[d]);
            if (d != last && runs_layers(d)) {
                others += r.device[d];
                ++n_others;
                if (!layers.empty()) other_layers += layers[d];
            }
        }
        if (!n_others || last == n) return;
        std::printf("  last stage: %+.1f%% against the other stages' mean", 100 * (r.device[last] / (others / (double)n_others) - 1));
        if (other_layers) {
            const double per = others / other_layers, head = r.device[last] - layers[last] * per;
            std::printf("; beyond its %d layers at their %.3f ms a layer it takes %.3f ms, %.1f%% of it", layers[last], per, head, 100 * head / r.device[last]);
        }
        std::printf("\n");
    };
    for (size_t rows : o.rows) {
        std::vector<infer::Sequence> seqs;
        for (size_t i = 0; i < rows; ++i) seqs.push_back(model.make_sequence());
        for (auto& q : seqs) feed(q, prompt, 0, prompt.size());
        ctx.logits(0);
        std::vector<infer::BatchEntry> batch;
        const Reading r = time_passes([](size_t) {}, [&](size_t g) {
            batch.clear();
            for (auto& q : seqs) batch.push_back(infer::BatchEntry{&q, &gen[g], 1, true});
            model.forward(ctx, batch.data(), batch.size());
            ctx.logits(0);
        });
        for (auto& q : seqs) model.reset(q);
        std::printf("\n%zu row%s a pass: %.2f ms a pass on the host, the stages one after another\n", rows, rows == 1 ? "" : "s", r.host);
        report(r);
    }
    for (size_t rows : o.prefill) {
        std::vector<uint32_t> whole = prompt;
        whole.insert(whole.end(), tail.begin(), tail.begin() + (std::ptrdiff_t)rows);
        infer::Sequence q = model.make_sequence();
        const Reading r = time_passes(
            [&](size_t) {
                model.reset(q);
                feed(q, whole, 0, o.context);
            },
            [&](size_t) {
                feed(q, whole, o.context, whole.size());
                ctx.logits(0);
            });
        model.reset(q);
        std::printf("\nprefill, %zu prompt rows a pass ending a prompt after %zu tokens: %.2f ms a pass on the host, the stages one after another\n", rows, o.context, r.host);
        report(r);
    }
    return 0;
}

// Every device's vector summed on the host in device order, the sum written back to each and consumed by a dependent matmul there, as a tensor group's sum would be.
int groupsum(const std::vector<int>& devices) {
    std::vector<std::unique_ptr<Stage>> st;
    for (int d : devices) st.push_back(std::make_unique<Stage>(d));
    for (size_t rows : {(size_t)1, (size_t)512}) {
        const size_t n = rows * kEmbd;
        std::vector<float> sum(n);
        std::vector<double> t;
        for (int it = 0; it < 60; ++it) {
            const auto t0 = Clock::now();
            std::vector<backend::Ticket> tk(st.size());
            for (size_t i = 0; i < st.size(); ++i) {
                st[i]->b->copy(*st[i]->out, 0, *st[i]->x, 0, n * 4);
                tk[i] = st[i]->b->submit();
            }
            std::fill(sum.begin(), sum.end(), 0.0f);
            for (size_t i = 0; i < st.size(); ++i) {
                st[i]->b->wait(tk[i]);
                const float* p = (const float*)st[i]->out->host_ptr();
                for (size_t k = 0; k < n; ++k) sum[k] += p[k];
            }
            for (size_t i = 0; i < st.size(); ++i) {
                st[i]->b->write(*st[i]->x, 0, sum.data(), n * 4);
                tk[i] = st[i]->b->submit();
            }
            for (size_t i = 0; i < st.size(); ++i) st[i]->b->wait(tk[i]);
            if (it >= 10) t.push_back(ms_since(t0) * 1000);
        }
        std::sort(t.begin(), t.end());
        std::printf("group of %zu, %zu rows (%zu KB each): median %.0f us, p90 %.0f us a sum\n", st.size(), rows, n * 4 / 1024, t[t.size() / 2], t[t.size() * 9 / 10]);
    }
    return 0;
}

// A mixture-of-experts layer's exchange between ranks, the ranks' threads each owning their backend.
// Each rank has `tokens` tokens routed to 8 of 128 experts, expert e held by rank e % ranks; with `skew` a share of the choices go to one expert.
// Dispatch sends each token once to every rank holding any of its experts (4096 floats); return sends back one 4096-float vector per entry.
int exchange(const std::vector<int>& devices, size_t tokens, double skew) {
    const size_t G = devices.size(), E = 4096, K = 8, NE = 128;
    std::vector<std::unique_ptr<Stage>> st;
    for (int d : devices) st.push_back(std::make_unique<Stage>(d));
    std::mt19937 rng(42);
    // counts[src][dst]: tokens rank src sends to dst, and entries dst returns to src.
    std::vector<std::vector<size_t>> tok(G, std::vector<size_t>(G, 0)), ent(G, std::vector<size_t>(G, 0));
    for (size_t s = 0; s < G; ++s)
        for (size_t t = 0; t < tokens; ++t) {
            std::vector<size_t> chosen;
            while (chosen.size() < K) {
                const size_t e = std::uniform_real_distribution<double>(0, 1)(rng) < skew ? 0 : rng() % NE;
                if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
                else if (e == 0) chosen.push_back(1 + rng() % (NE - 1));
            }
            std::vector<bool> to(G, false);
            for (size_t e : chosen) {
                to[e % G] = true;
                ++ent[s][e % G];
            }
            for (size_t d = 0; d < G; ++d) tok[s][d] += to[d];
        }
    size_t most = 0, total = 0;
    for (size_t d = 0; d < G; ++d) {
        size_t in = 0;
        for (size_t s = 0; s < G; ++s) in += ent[s][d];
        most = std::max(most, in);
        total += in;
    }
    std::printf("%zu ranks, %zu tokens each, skew %.2f: entries per rank mean %.0f, most %zu\n", G, tokens, skew, (double)total / G, most);
    // Host staging per rank: what it sends and what it receives, laid out by peer.
    std::vector<BufferPtr> send(G), recv(G);
    std::vector<std::vector<float>> inbox(G);
    size_t cap = 0;
    for (size_t r = 0; r < G; ++r) {
        size_t out = 0, in = 0;
        for (size_t p = 0; p < G; ++p) {
            out += std::max(tok[r][p], ent[p][r]);
            in += std::max(tok[p][r], ent[r][p]);
        }
        cap = std::max({cap, out, in});
    }
    for (size_t r = 0; r < G; ++r) {
        send[r] = st[r]->b->alloc(cap * E * 4, Memory::host_visible);
        recv[r] = st[r]->b->alloc(cap * E * 4);
        inbox[r].resize(cap * E);
    }
    auto phase = [&](const std::vector<std::vector<size_t>>& rows_out) {
        // Every rank copies its outgoing rows into host memory in one submission; the host relays them into each destination's inbox, which the destination writes in and consumes.
        std::vector<backend::Ticket> tk(G);
        for (size_t r = 0; r < G; ++r) {
            size_t n = 0;
            for (size_t p = 0; p < G; ++p) n += rows_out[r][p];
            st[r]->b->copy(*send[r], 0, *st[r]->y, 0, std::min(n, (size_t)512) * E * 4);
            tk[r] = st[r]->b->submit();
        }
        for (size_t r = 0; r < G; ++r) st[r]->b->wait(tk[r]);
        for (size_t d = 0; d < G; ++d) {
            size_t at = 0;
            for (size_t s = 0; s < G; ++s) {
                const size_t n = rows_out[s][d] * E;
                std::memcpy(inbox[d].data() + at, (const float*)send[s]->host_ptr(), std::min(n, cap * E) * 4);
                at += n;
            }
            st[d]->b->write(*recv[d], 0, inbox[d].data(), at * 4);
            tk[d] = st[d]->b->submit();
        }
        for (size_t d = 0; d < G; ++d) st[d]->b->wait(tk[d]);
    };
    std::vector<std::vector<size_t>> back(G, std::vector<size_t>(G));
    for (size_t s = 0; s < G; ++s)
        for (size_t d = 0; d < G; ++d) back[d][s] = ent[s][d];
    std::vector<double> t_dispatch, t_return;
    for (int it = 0; it < 40; ++it) {
        auto t0 = Clock::now();
        phase(tok);
        const double a = ms_since(t0) * 1000;
        t0 = Clock::now();
        phase(back);
        const double b = ms_since(t0) * 1000;
        if (it >= 5) {
            t_dispatch.push_back(a);
            t_return.push_back(b);
        }
    }
    std::sort(t_dispatch.begin(), t_dispatch.end());
    std::sort(t_return.begin(), t_return.end());
    const double mb = total * E * 4 / 1e6;
    std::printf("  dispatch median %.0f us, return median %.0f us, %.1f MB returned; 94 layers would spend %.1f ms a pass exchanging\n",
                t_dispatch[t_dispatch.size() / 2], t_return[t_return.size() / 2], mb, 94 * (t_dispatch[t_dispatch.size() / 2] + t_return[t_return.size() / 2]) / 1000);
    return 0;
}

std::vector<int> parse_devices(int argc, char** argv, int from, int& next) {
    std::vector<int> d;
    int i = from;
    for (; i < argc && std::string(argv[i]) != "--"; ++i) d.push_back(std::atoi(argv[i]));
    next = i + 1;
    return d;
}

const char* const kUsage =
    "usage: llmx-multi-device-bench concurrent D...\n"
    "       llmx-multi-device-bench pipeline D0 D1 [D...] [--ms MS[,MS...][/MS[,MS...]...]] [--rows R[,R...]] [--passes P[,P...]]\n"
    "                                        [--driver all|DRIVER[,DRIVER...]] [--sample MS] [--calls N]\n"
    "         --ms: a stage's time for every stage or for each, in one list for every --rows value or a list for each, separated by /\n"
    "         DRIVER: per-stage, completion, round or round-idle\n"
    "       llmx-multi-device-bench stages MODEL D... [--rows R[,R...]] [--prefill R[,R...]] [--context N] [--steps N]\n"
    "       llmx-multi-device-bench groupsum D...\n"
    "       llmx-multi-device-bench exchange D... [-- tokens skew]\n";

struct Usage : std::runtime_error {
    using std::runtime_error::runtime_error;
};

double number(const std::string& s, const std::string& what) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end || !std::isfinite(v) || v < 0) throw Usage(what + ": " + s + " is not a number of zero or more");
    return v;
}

size_t count(const std::string& s, const std::string& what) {
    const double v = number(s, what);
    if (v != std::floor(v)) throw Usage(what + ": " + s + " is not a whole number");
    return (size_t)v;
}

std::vector<std::string> items(const std::string& s, const std::string& what) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string item; std::getline(in, item, ',');) out.push_back(item);
    if (out.empty()) throw Usage(what + ": an empty list");
    return out;
}

// A mode's arguments: device indices, then `--name value` pairs among the mode's options.
struct Args {
    std::vector<int> devices;
    std::map<std::string, std::string> named;
    bool has(const std::string& k) const { return named.count(k) != 0; }
    const std::string& at(const std::string& k) const { return named.at(k); }
};

Args parse_args(int argc, char** argv, int from, const std::vector<std::string>& known) {
    Args a;
    int i = from;
    for (; i < argc && std::strncmp(argv[i], "--", 2) != 0; ++i) a.devices.push_back((int)count(argv[i], "device"));
    for (; i < argc; i += 2) {
        const std::string k = argv[i];
        if (std::find(known.begin(), known.end(), k) == known.end()) throw Usage(k + ": not an option of this mode");
        if (i + 1 >= argc) throw Usage(k + ": needs a value");
        a.named[k] = argv[i + 1];
    }
    return a;
}

std::vector<size_t> count_list(const Args& a, const std::string& k, std::vector<size_t> fallback) {
    if (!a.has(k)) return fallback;
    std::vector<size_t> out;
    for (const auto& item : items(a.at(k), k)) out.push_back(count(item, k));
    return out;
}

PipelineOptions pipeline_options(const Args& a) {
    PipelineOptions o;
    o.rows = count_list(a, "--rows", o.rows);
    for (size_t r : o.rows)
        if (!r || r > kMaxRows) throw Usage("--rows: 1 to " + std::to_string(kMaxRows) + " rows a pass");
    if (a.has("--ms")) {
        o.ms.clear();
        std::istringstream lists(a.at("--ms"));
        for (std::string list; std::getline(lists, list, '/');) {
            o.ms.emplace_back();
            for (const auto& item : items(list, "--ms")) o.ms.back().push_back(number(item, "--ms"));
        }
    }
    if (o.ms.size() != 1 && o.ms.size() != o.rows.size()) throw Usage("--ms: one list for every --rows value, or one for each");
    for (const auto& list : o.ms) {
        if (list.size() != 1 && list.size() != a.devices.size()) throw Usage("--ms: one time for every stage, or one for each");
        for (double ms : list)
            if (ms <= 0) throw Usage("--ms: a stage takes some time");
    }
    if (o.ms.size() == 1 && o.ms[0].size() > 1 && o.rows.size() > 1)
        throw Usage("--ms: a time for each stage is measured at one row count, so give a list for each --rows value");
    o.passes = count_list(a, "--passes", o.passes);
    for (size_t p : o.passes)
        if (!p) throw Usage("--passes: at least one pass in flight");
    if (a.has("--driver") && a.at("--driver") != "all") {
        o.drivers.clear();
        for (const auto& name : items(a.at("--driver"), "--driver")) {
            const auto at = std::find(std::begin(kDriverNames), std::end(kDriverNames), name);
            if (at == std::end(kDriverNames)) throw Usage("--driver: " + name + " is not one of per-stage, completion, round and round-idle");
            o.drivers.push_back((Driver)(at - std::begin(kDriverNames)));
        }
    }
    if (a.has("--sample")) o.sample_ms = number(a.at("--sample"), "--sample");
    if (a.has("--calls")) o.calls = count(a.at("--calls"), "--calls");
    return o;
}

StagesOptions stages_options(const Args& a) {
    StagesOptions o;
    o.rows = count_list(a, "--rows", o.rows);
    for (size_t r : o.rows)
        if (!r) throw Usage("--rows: at least one row a pass");
    o.prefill = count_list(a, "--prefill", o.prefill);
    for (size_t r : o.prefill)
        if (!r || r > (size_t)infer::kDefaultUbatch) throw Usage("--prefill: 1 to " + std::to_string(infer::kDefaultUbatch) + " prompt rows a pass, the ubatch");
    if (a.has("--context")) o.context = count(a.at("--context"), "--context");
    if (a.has("--steps")) o.steps = count(a.at("--steps"), "--steps");
    if (!o.context || !o.steps) throw Usage("--context and --steps: at least one");
    return o;
}

} // namespace

int main(int argc, char** argv) {
    // Every line reaches a log as it is printed, so a long run shows how far it got.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        int next = 0;
        if (mode == "concurrent") return concurrent(parse_devices(argc, argv, 2, next));
        if (mode == "pipeline") {
            const Args a = parse_args(argc, argv, 2, {"--ms", "--rows", "--passes", "--driver", "--sample", "--calls"});
            if (a.devices.size() < 2) throw Usage("pipeline: two devices or more, a stage on each");
            return pipeline(a.devices, pipeline_options(a));
        }
        if (mode == "stages") {
            if (argc < 3 || !std::strncmp(argv[2], "--", 2)) throw Usage("stages: the model file first");
            const Args a = parse_args(argc, argv, 3, {"--rows", "--prefill", "--context", "--steps"});
            if (a.devices.empty()) throw Usage("stages: one device or more");
            return stages(argv[2], a.devices, stages_options(a));
        }
        if (mode == "groupsum") return groupsum(parse_devices(argc, argv, 2, next));
        if (mode == "exchange") {
            const auto d = parse_devices(argc, argv, 2, next);
            const size_t tokens = next < argc ? (size_t)std::atoi(argv[next]) : 8;
            const double skew = next + 1 < argc ? std::atof(argv[next + 1]) : 0.0;
            return exchange(d, tokens, skew);
        }
        throw Usage(mode.empty() ? "no mode given" : mode + ": not a mode");
    } catch (const Usage& e) {
        std::fprintf(stderr, "llmx-multi-device-bench: %s\n%s", e.what(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-multi-device-bench: %s\n", e.what());
        return 1;
    }
}
