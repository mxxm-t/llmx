// Phase 0 of docs/MULTI-DEVICE.md through the runtime's own Vulkan backend: what several devices in one process cost each other, and what a split pays for moving data between them.
// Every mode but `stages` uses the Backend interface as a split would (adopt, matmul, copy into host-visible memory, submit, wait, write), so the numbers include the backend's own submission and waiting.
// `concurrent D...`: each device runs decode-shaped passes on its own thread, first alone, then all at once.
// `pipeline D0 D1 [D...] [options]`: a stage per device with P passes in flight, driven by a thread per stage or by one thread; each pass leaves a stage through host memory into the next, and the host holds it for a sampling time before it re-enters stage 0.
// `stages MODEL D... [options]`: a model split by layers over the devices as the CLI places it, and each stage's device time a decode pass from GPU timestamps.
// `groupsum D...`: every device's vector summed on the host in device order and written back to each, for a decode row and a 512-row chunk.
// `exchange D... [-- tokens skew]`: a mixture-of-experts layer's dispatch and return between ranks, each rank's entries sent to the ranks holding their experts and the results sent back.
// Shapes are those of a 5120-wide dense model (Qwen3-32B) and Qwen3-235B-A22B's experts (4096 wide, 128 experts of 1536, 8 per token).
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
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "bench_weights.hpp"
#include "format/gguf.hpp"
#include "inference/load.hpp"

namespace {

using backend::Backend;
using backend::BufferPtr;
using backend::Memory;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

const size_t kEmbd = 5120, kFF = 25600, kMaxRows = 512;
const uint32_t kQ8 = gguf::GGML_TYPE_Q8_0;

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

    // A pass over `rows` rows: `layers` times up and down, the residual copied into host-visible memory, submitted.
    // Each slice writes a region of its own, so the calls of one projection never write the same memory.
    backend::Ticket run(size_t rows) {
        const size_t nu = kFF / split, nd = kEmbd / split;
        for (size_t l = 0; l < layers; ++l) {
            for (size_t i = 0; i < split; ++i)
                b->matmul(kQ8, {up.get(), i * nu * row_bytes(kQ8, kEmbd) / 4}, {x.get(), 0}, {h.get(), i * nu * rows}, kEmbd, nu, rows);
            for (size_t i = 0; i < split; ++i)
                b->matmul_add(kQ8, {down.get(), i * nd * row_bytes(kQ8, kFF) / 4}, {h.get(), 0}, {x.get(), i * nd * rows}, kFF, nd, rows);
        }
        b->copy(*out, 0, *x, 0, rows * kEmbd * 4);
        return b->submit();
    }

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
            const size_t next = nearest_split(per / 2 / (ms / (double)calls));
            if (next == split) break;
            split = next;
            per = pass_ms() / (double)layers;
        }
        layers = std::max<size_t>(1, (size_t)(ms / per + 0.5));
        // One correction at the size chosen, since a short measurement on an idle device or a busy host can misjudge a layer.
        layers = std::max<size_t>(1, (size_t)((double)layers * ms / pass_ms() + 0.5));
    }
};

// Submissions the backend keeps in flight before recording the next one blocks, as the running program finds it: submissions of several milliseconds each are queued without waiting, and the first whose recording takes over half a submission's time found every slot busy.
// Returns -1 when none blocked within 64, 0 when one matmul alone takes more than one submission, and -2 when the host never recorded the probes fast enough to tell.
int ring_depth(Stage& s, double& submission_ms) {
    auto probe = [&](int m) {
        for (int i = 0; i < m; ++i) s.b->matmul(kQ8, {s.up.get(), 0}, {s.x.get(), 0}, {s.h.get(), 0}, kEmbd, kFF, kMaxRows);
        return s.b->submit();
    };
    int m = 1;
    for (;;) {
        const backend::Ticket a = probe(m);
        s.b->wait(a);
        const auto t0 = Clock::now();
        const backend::Ticket b = probe(m);
        s.b->wait(b);
        const double t = ms_since(t0);
        if (b - a != 1) {
            if (m == 1) return 0;
            m /= 2;
            break;
        }
        submission_ms = t;
        if (t >= 8 || m >= 16) break;
        m *= 2;
    }
    // Once the ring is full every later submission blocks too, so a block counts only when the next one blocks as well.
    // Until the first block, every submission must be recorded within a quarter of a submission's time of the first, or the device may have retired one and let one more in; an attempt that is slower is repeated.
    for (int attempt = 0; attempt < 8; ++attempt) {
        s.b->sync();
        const auto start = Clock::now();
        int blocked_at = -1;
        bool late = false;
        for (int i = 0; i < 64; ++i) {
            late = blocked_at < 0 && ms_since(start) > submission_ms / 4;
            if (late) break;
            const auto t0 = Clock::now();
            probe(m);
            const bool blocked = ms_since(t0) > submission_ms / 2;
            if (blocked && blocked_at >= 0) {
                s.b->sync();
                return blocked_at;
            }
            blocked_at = blocked ? i : -1;
        }
        if (!late) {
            s.b->sync();
            return -1;
        }
    }
    s.b->sync();
    return -2;
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
    // Takes an item if one is waiting; `closed` reports a channel that will never have one.
    bool try_get(T& v, bool& is_closed) {
        std::lock_guard<std::mutex> l(m);
        is_closed = closed && q.empty();
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

struct Result {
    double rate = 0, latency = 0;   // passes a second, and the median time from entering stage 0 to leaving the last
    size_t in_flight = 0;           // distinct passes seen leaving the last stage
    double busy = -1;               // the driving thread's working share, for the one-thread driver
};

// What the host sees of the passes leaving the last stage: from the 3P-th on, each one's time since it entered stage 0 and its id, until 3 seconds have passed and 20 are measured.
struct Tally {
    size_t P = 0, left = 0;
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

    // `blocked` is the driving thread's time waiting for a completion over the measured passes, negative where no one thread drives.
    Result result(double blocked) {
        Result r;
        const double elapsed = std::chrono::duration<double, std::milli>(until - from).count();
        r.rate = (double)latency.size() / (elapsed / 1000);
        std::sort(latency.begin(), latency.end());
        r.latency = latency[latency.size() / 2];
        r.in_flight = ids.size();
        if (blocked >= 0) r.busy = 1 - blocked / elapsed;
        return r;
    }
};

// A thread per stage: each takes its next pass, writes it in, runs it, waits for it and hands it to the next stage; the calling thread samples what leaves the last stage.
// With `warm_rows`, a stage waiting for its next pass keeps its device working on that many rows of its up projection at a time, so the device does not drop its clock between passes.
Result per_stage(std::vector<std::unique_ptr<Stage>>& st, size_t rows, size_t P, double sample_ms, size_t warm_rows) {
    const size_t S = st.size();
    std::vector<Channel<Pass>> ch(S + 1);
    auto stage = [&](Stage& s, Channel<Pass>& in, Channel<Pass>& out) {
        Pass p;
        for (;;) {
            if (warm_rows) {
                bool closed = false;
                if (!in.try_get(p, closed)) {
                    if (closed) break;
                    s.b->matmul(kQ8, {s.up.get(), 0}, {s.x.get(), 0}, {s.h.get(), 0}, kEmbd, warm_rows, 1);
                    s.b->wait(s.b->submit());
                    continue;
                }
            } else if (!in.get(p)) {
                break;
            }
            s.b->write(*s.x, 0, p.residual.data(), p.residual.size() * 4);
            s.b->wait(s.run(rows));
            std::memcpy(p.residual.data(), s.out->host_ptr(), p.residual.size() * 4);
            out.put(std::move(p));
        }
    };
    std::vector<std::thread> th;
    for (size_t i = 0; i < S; ++i) th.emplace_back([&, i] { stage(*st[i], ch[i], ch[i + 1]); });
    for (size_t i = 0; i < P; ++i) ch[0].put(fresh_pass(i, rows));
    Tally tally;
    tally.P = P;
    Pass p;
    while (ch[S].get(p)) {
        if (!tally.leave(p)) break;
        spin(sample_ms);
        p.started = Clock::now();
        ch[0].put(std::move(p));
    }
    for (auto& c : ch) c.close();
    for (auto& t : th) t.join();
    return tally.result(-1);
}

// One thread records, relays and samples for every stage, serving the stages in the order their passes complete.
// A waiter per stage only blocks on that stage's tickets and reports each completion, as a sync-file or timeline poll would; it touches the backend only while the driving thread leaves it alone.
// A stage has at most one pass on its device, as with a thread per stage, so the two drivers differ only in which threads do the host's work.
Result one_thread(std::vector<std::unique_ptr<Stage>>& st, size_t rows, size_t P, double sample_ms) {
    const size_t S = st.size();
    std::vector<Channel<backend::Ticket>> tickets(S);
    Channel<size_t> completed;
    std::vector<std::thread> waiters;
    for (size_t i = 0; i < S; ++i)
        waiters.emplace_back([&, i] {
            backend::Ticket t = 0;
            while (tickets[i].get(t)) {
                st[i]->b->wait(t);
                completed.put(i);
            }
        });
    std::vector<std::deque<Pass>> waiting(S);
    std::vector<Pass> on(S);
    std::vector<bool> busy(S, false);
    // A stage whose device is free takes its oldest waiting pass.
    auto launch = [&](size_t s) {
        if (busy[s] || waiting[s].empty()) return;
        on[s] = std::move(waiting[s].front());
        waiting[s].pop_front();
        Stage& g = *st[s];
        g.b->write(*g.x, 0, on[s].residual.data(), on[s].residual.size() * 4);
        busy[s] = true;
        tickets[s].put(g.run(rows));
    };
    for (size_t i = 0; i < P; ++i) waiting[0].push_back(fresh_pass(i, rows));
    launch(0);
    Tally tally;
    tally.P = P;
    double blocked = 0;
    for (;;) {
        size_t s = 0;
        const auto w = Clock::now();
        completed.get(s);
        if (tally.measuring()) blocked += ms_since(w);
        busy[s] = false;
        Pass p = std::move(on[s]);
        std::memcpy(p.residual.data(), st[s]->out->host_ptr(), p.residual.size() * 4);
        // Every device that can take work gets it before the thread samples.
        if (s + 1 < S) {
            waiting[s + 1].push_back(std::move(p));
            launch(s + 1);
            launch(s);
            continue;
        }
        if (!tally.leave(p)) break;
        launch(s);
        spin(sample_ms);
        p.started = Clock::now();
        waiting[0].push_back(std::move(p));
        launch(0);
    }
    for (size_t n = (size_t)std::count(busy.begin(), busy.end(), true); n; --n) {
        size_t s = 0;
        completed.get(s);
    }
    for (auto& t : tickets) t.close();
    for (auto& t : waiters) t.join();
    return tally.result(blocked);
}

struct PipelineOptions {
    std::vector<double> ms{10.0};        // one pass's time on each stage, one value for every stage or one per stage
    std::vector<size_t> rows{1, 8};
    std::vector<size_t> passes;          // P, by default 1, S, S + 1 and 2S
    bool per_stage = true, one_thread = true;
    double sample_ms = 0.3;
    size_t calls = 0;                    // backend calls a stage records a pass, 0 for one call a projection
    size_t warm_rows = 0;
};

int pipeline(const std::vector<int>& devices, PipelineOptions o) {
    const size_t S = devices.size();
    std::vector<std::unique_ptr<Stage>> st;
    for (int d : devices) st.push_back(std::make_unique<Stage>(d));
    if (o.passes.empty()) o.passes = {1, S, S + 1, 2 * S};
    std::printf("pipeline over %zu stages, devices", S);
    for (int d : devices) std::printf(" %d", d);
    std::printf("; the host spends %.2f ms sampling each pass that leaves the last stage\n", o.sample_ms);
    double submission_ms = 0;
    const int depth = ring_depth(*st[0], submission_ms);
    if (depth > 0)
        std::printf("command ring: %d submissions in flight before recording the next blocks (device %d, submissions of %.1f ms)\n", depth, devices[0], submission_ms);
    else if (depth == -1)
        std::printf("command ring: more than 64 submissions in flight without blocking (device %d)\n", devices[0]);
    else if (depth == -2)
        std::printf("command ring: not measured, the host was too slow to queue the probes in 8 attempts (device %d)\n", devices[0]);
    else
        std::printf("command ring: not measured, one matmul took more than one submission (device %d)\n", devices[0]);
    for (size_t rows : o.rows) {
        for (size_t s = 0; s < S; ++s) st[s]->calibrate(o.ms.size() == 1 ? o.ms[0] : o.ms[s], rows, o.calls);
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
        // Each stage's pass repeated back to back, so the device never idles: the rate a filled pipeline is bounded by.
        // The tickets count the submissions a pass takes.
        std::printf("\n%zu row%s a pass:\n", rows, rows == 1 ? "" : "s");
        double slowest = 0;
        for (size_t i = 0; i < S; ++i) {
            Stage& s = *st[i];
            std::vector<float> r(rows * kEmbd, 0.01f);
            for (int k = 0; k < 3; ++k) s.b->wait(s.run(rows));
            backend::Ticket first = 0, last = 0;
            const auto t0 = Clock::now();
            for (int k = 0; k < 20; ++k) {
                s.b->write(*s.x, 0, r.data(), r.size() * 4);
                last = s.run(rows);
                if (!k) first = last;
                s.b->wait(last);
                std::memcpy(r.data(), s.out->host_ptr(), r.size() * 4);
            }
            const double busy = ms_since(t0) / 20;
            slowest = std::max(slowest, busy);
            std::printf("  stage %zu, device %d: %zu layers, %zu call%s a projection, %zu calls in %.1f submissions a pass, %.2f ms a pass back to back\n", i, devices[i],
                        s.layers, s.split, s.split == 1 ? "" : "s", s.calls(), double(last - first) / 19, busy);
        }
        std::printf("  one pass through every stage alone: %.2f ms\n", serial);
        std::printf("  %-10s %4s %10s %10s %11s %12s\n", "driver", "P", "passes/s", "ms a pass", "of slowest", "thread busy");
        auto row = [&](const char* driver, const Result& r) {
            std::printf("  %-10s %4zu %10.1f %10.2f %10.0f%%", driver, r.in_flight, r.rate, r.latency, 100 * r.rate * slowest / 1000);
            if (r.busy >= 0) std::printf(" %11.0f%%\n", 100 * r.busy);
            else std::printf(" %12s\n", "-");
        };
        for (size_t P : o.passes) {
            if (o.per_stage) row("per-stage", per_stage(st, rows, P, o.sample_ms, o.warm_rows));
            if (o.one_thread) row("one", one_thread(st, rows, P, o.sample_ms));
        }
    }
    return 0;
}

struct StagesOptions {
    std::vector<size_t> rows{1, 8, 16, 32, 64};
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

// Decode passes of each row count through the model split over `devices`, every sequence first given `context` tokens, and each device's time a pass as the sum of its dispatches' GPU timestamps.
// The last stage is set against the others, and where the plan gives the layer counts, what it takes beyond its layers at the others' time a layer is the head's.
int stages(const std::string& path, const std::vector<int>& devices, const StagesOptions& o) {
    std::vector<backend::BackendPtr> backends;
    infer::PlacementRequest request;
    for (int d : devices) {
        backends.push_back(backend::make_vulkan_backend(d, true));
        request.names.push_back("vulkan:" + std::to_string(d));
    }
    const size_t most = *std::max_element(o.rows.begin(), o.rows.end()), warm = 3;
    request.decode_rows = most;
    request.histories = most;
    request.history_tokens = o.context + warm + o.steps;
    const auto loaded = infer::load_model(path, backends, request);
    infer::Model& model = *loaded->model;
    std::printf("%s, %zu tokens of context, %zu passes timed after %zu:\n%s", path.c_str(), o.context, o.steps, warm, loaded->plan.c_str());
    const std::vector<int> layers = plan_layers(loaded->plan, devices.size());
    // Ids below 1000, or below a smaller vocabulary's size, as the CLI's bench takes them.
    const uint32_t vocab = (uint32_t)std::min<size_t>(1000, model.n_vocab());
    auto ids_from = [vocab](uint32_t seed, size_t n) {
        std::vector<uint32_t> ids(n);
        for (auto& t : ids) {
            seed = seed * 1664525u + 1013904223u;
            t = (seed >> 8) % vocab;
        }
        return ids;
    };
    const std::vector<uint32_t> prompt = ids_from(12345u, o.context), gen = ids_from(777u, warm + o.steps);
    const size_t n = devices.size(), ubatch = model.prefill_batch();
    infer::ExecContext ctx;
    for (size_t rows : o.rows) {
        std::vector<infer::Sequence> seqs;
        for (size_t i = 0; i < rows; ++i) seqs.push_back(model.make_sequence());
        // Each history's prompt in slices of the ubatch, each slice given the whole prompt's extent.
        for (auto& q : seqs)
            for (size_t at = 0; at < prompt.size(); at += ubatch) {
                infer::BatchEntry e{&q, prompt.data() + at, std::min(ubatch, prompt.size() - at), at + ubatch >= prompt.size()};
                e.extent = e.fresh = prompt.size();
                model.forward(ctx, &e, 1);
            }
        ctx.logits(0);
        std::vector<infer::BatchEntry> batch;
        auto pass = [&](size_t g) {
            batch.clear();
            for (auto& q : seqs) batch.push_back(infer::BatchEntry{&q, &gen[g], 1, true});
            model.forward(ctx, batch.data(), batch.size());
            ctx.logits(0);
        };
        for (size_t g = 0; g < warm; ++g) pass(g);
        for (auto& b : backends) backend::vulkan_kernel_times(*b);
        std::vector<double> device(n, 0.0), dispatches(n, 0.0);
        double host = 0;
        for (size_t g = 0; g < o.steps; ++g) {
            const auto t0 = Clock::now();
            pass(warm + g);
            host += ms_since(t0);
            for (size_t d = 0; d < n; ++d) {
                for (const auto& k : backend::vulkan_kernel_times(*backends[d])) device[d] += k.second / (double)o.steps;
                dispatches[d] += (double)backend::vulkan_timed_dispatches(*backends[d]) / (double)o.steps;
            }
        }
        for (auto& q : seqs) model.reset(q);
        if (std::all_of(dispatches.begin(), dispatches.end(), [](double v) { return v == 0; }))
            throw std::runtime_error("no device timestamps its dispatches");

        std::printf("\n%zu row%s a pass: %.2f ms a pass on the host, the stages one after another\n", rows, rows == 1 ? "" : "s", host / (double)o.steps);
        auto runs_layers = [&](size_t d) { return layers.empty() ? dispatches[d] > 0 : layers[d] > 0; };
        size_t last = n;
        for (size_t d = 0; d < n; ++d)
            if (runs_layers(d)) last = d;
        double others = 0;
        size_t n_others = 0;
        int other_layers = 0;
        for (size_t d = 0; d < n; ++d) {
            std::printf("  %s", request.names[d].c_str());
            if (!layers.empty()) std::printf(", %d layers", layers[d]);
            std::printf(": %.3f ms of device time a pass over %.0f dispatches\n", device[d], dispatches[d]);
            if (d != last && runs_layers(d)) {
                others += device[d];
                ++n_others;
                if (!layers.empty()) other_layers += layers[d];
            }
        }
        if (!n_others || last == n) continue;
        std::printf("  last stage: %+.1f%% against the other stages' mean", 100 * (device[last] / (others / (double)n_others) - 1));
        if (other_layers) {
            const double per = others / other_layers, head = device[last] - layers[last] * per;
            std::printf("; beyond its %d layers at their %.3f ms a layer it takes %.3f ms, %.1f%% of it", layers[last], per, head, 100 * head / device[last]);
        }
        std::printf("\n");
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
    "       llmx-multi-device-bench pipeline D0 D1 [D...] [--ms MS[,MS...]] [--rows R[,R...]] [--passes P[,P...]]\n"
    "                                        [--driver per-stage|one|both] [--sample MS] [--calls N] [--warm ROWS]\n"
    "       llmx-multi-device-bench stages MODEL D... [--rows R[,R...]] [--context N] [--steps N]\n"
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
    if (a.has("--ms")) {
        o.ms.clear();
        for (const auto& item : items(a.at("--ms"), "--ms")) o.ms.push_back(number(item, "--ms"));
    }
    if (o.ms.size() != 1 && o.ms.size() != a.devices.size()) throw Usage("--ms: one time for every stage, or one for each");
    for (double ms : o.ms)
        if (ms <= 0) throw Usage("--ms: a stage takes some time");
    o.rows = count_list(a, "--rows", o.rows);
    for (size_t r : o.rows)
        if (!r || r > kMaxRows) throw Usage("--rows: 1 to 512 rows a pass");
    o.passes = count_list(a, "--passes", o.passes);
    for (size_t p : o.passes)
        if (!p) throw Usage("--passes: at least one pass in flight");
    const std::string driver = a.has("--driver") ? a.at("--driver") : "both";
    if (driver != "per-stage" && driver != "one" && driver != "both") throw Usage("--driver: per-stage, one or both");
    o.per_stage = driver != "one";
    o.one_thread = driver != "per-stage";
    if (a.has("--sample")) o.sample_ms = number(a.at("--sample"), "--sample");
    if (a.has("--calls")) o.calls = count(a.at("--calls"), "--calls");
    if (a.has("--warm")) o.warm_rows = count(a.at("--warm"), "--warm");
    if (o.warm_rows > kFF) throw Usage("--warm: at most the up projection's 25600 rows");
    if (o.warm_rows && o.one_thread) throw Usage("--warm: a waiting stage keeps its device busy from its own thread, so it takes --driver per-stage");
    return o;
}

StagesOptions stages_options(const Args& a) {
    StagesOptions o;
    o.rows = count_list(a, "--rows", o.rows);
    for (size_t r : o.rows)
        if (!r) throw Usage("--rows: at least one row a pass");
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
            const Args a = parse_args(argc, argv, 2, {"--ms", "--rows", "--passes", "--driver", "--sample", "--calls", "--warm"});
            if (a.devices.size() < 2) throw Usage("pipeline: two devices or more, a stage on each");
            return pipeline(a.devices, pipeline_options(a));
        }
        if (mode == "stages") {
            if (argc < 3 || !std::strncmp(argv[2], "--", 2)) throw Usage("stages: the model file first");
            const Args a = parse_args(argc, argv, 3, {"--rows", "--context", "--steps"});
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
