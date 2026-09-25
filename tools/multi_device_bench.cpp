// Phase 0 of docs/MULTI-DEVICE.md through the runtime's own Vulkan backend: what several devices in one process cost each other, and what a split pays for moving data between them.
// Every mode uses the Backend interface as a split would (adopt, matmul, copy into host-visible memory, submit, wait, write), so the numbers include the backend's own submission and waiting.
// `concurrent D...`: each device runs decode-shaped passes on its own thread, first alone, then all at once.
// `pipeline D0 D1 [stage_ms] [warm_rows]`: two stages with P passes in flight for P from 1 to 6; each pass leaves stage 0 through host memory into stage 1, and the host holds it for a sampling time before it re-enters stage 0.
// `groupsum D...`: every device's vector summed on the host in device order and written back to each, for a decode row and a 512-row chunk.
// `exchange D... [tokens] [skew]`: a mixture-of-experts layer's dispatch and return between ranks, each rank's entries sent to the ranks holding their experts and the results sent back.
// Shapes are those of a 5120-wide dense model (Qwen3-32B) and Qwen3-235B-A22B's experts (4096 wide, 128 experts of 1536, 8 per token).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "bench_weights.hpp"
#include "format/gguf.hpp"
#include "quant/quant.hpp"

namespace {

using backend::Backend;
using backend::BufferPtr;
using backend::Memory;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

const size_t kEmbd = 5120, kFF = 25600;
const uint32_t kQ8 = gguf::GGML_TYPE_Q8_0;

// One device's share of a dense model: a layer's four large projections, repeated to reach a target time per pass.
struct Stage {
    backend::BackendPtr b;
    BufferPtr up, down, x, h, y, out;
    size_t layers = 1;

    explicit Stage(int device) {
        b = backend::make_vulkan_backend(device);
        auto wu = weights(row_bytes(kQ8, kEmbd) * kFF, (uint32_t)device * 7 + 1);
        auto wd = weights(row_bytes(kQ8, kFF) * kEmbd, (uint32_t)device * 7 + 2);
        up = b->adopt(wu.data(), wu.size());
        down = b->adopt(wd.data(), wd.size());
        x = b->alloc(512 * kEmbd * 4);
        h = b->alloc(512 * kFF * 4);
        y = b->alloc(512 * kEmbd * 4);
        out = b->alloc(512 * kEmbd * 4, Memory::host_visible);
    }

    // A pass over `rows` rows: `layers` times up and down, the residual copied into host-visible memory, submitted.
    backend::Ticket run(size_t rows) {
        for (size_t l = 0; l < layers; ++l) {
            b->matmul(kQ8, {up.get(), 0}, {x.get(), 0}, {h.get(), 0}, kEmbd, kFF, rows);
            b->matmul_add(kQ8, {down.get(), 0}, {h.get(), 0}, {x.get(), 0}, kFF, kEmbd, rows);
        }
        b->copy(*out, 0, *x, 0, rows * kEmbd * 4);
        return b->submit();
    }

    // Layers per pass so one decode pass takes about `ms`.
    void calibrate(double ms) {
        layers = 4;
        for (int i = 0; i < 3; ++i) b->wait(run(1));
        const auto t0 = Clock::now();
        for (int i = 0; i < 10; ++i) b->wait(run(1));
        const double per = ms_since(t0) / 10 / layers;
        layers = std::max<size_t>(1, (size_t)(ms / per + 0.5));
    }
};

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

// A bounded queue between stage threads.
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

// With `warm_rows`, a stage waiting for its next pass keeps its device working on that many rows of its up projection at a time, so the device does not drop its clock between passes.
int pipeline(int d0, int d1, double stage_ms, size_t warm_rows) {
    Stage s0(d0), s1(d1);
    s0.calibrate(stage_ms);
    s1.calibrate(stage_ms);
    std::printf("stages: device %d with %zu layers, device %d with %zu layers, about %.1f ms a decode pass each\n", d0, s0.layers, d1, s1.layers, stage_ms);
    const double sample_ms = 0.3;
    for (size_t rows : {(size_t)1, (size_t)8}) {
        // One pass through both stages alone, the time a single stream sees per token.
        double serial = 0;
        {
            std::vector<float> r(rows * kEmbd, 0.01f);
            for (int k = 0; k < 13; ++k) {
                const auto t0 = Clock::now();
                s0.b->write(*s0.x, 0, r.data(), r.size() * 4);
                s0.b->wait(s0.run(rows));
                std::memcpy(r.data(), s0.out->host_ptr(), r.size() * 4);
                s1.b->write(*s1.x, 0, r.data(), r.size() * 4);
                s1.b->wait(s1.run(rows));
                std::memcpy(r.data(), s1.out->host_ptr(), r.size() * 4);
                if (k >= 3) serial += ms_since(t0) / 10;
            }
        }
        // Each stage's pass repeated back to back, so the device never idles: the rate a filled pipeline is bounded by.
        auto busy = [&](Stage& s) {
            std::vector<float> r(rows * kEmbd, 0.01f);
            for (int k = 0; k < 3; ++k) s.b->wait(s.run(rows));
            const auto t0 = Clock::now();
            for (int k = 0; k < 20; ++k) {
                s.b->write(*s.x, 0, r.data(), r.size() * 4);
                s.b->wait(s.run(rows));
                std::memcpy(r.data(), s.out->host_ptr(), r.size() * 4);
            }
            return ms_since(t0) / 20;
        };
        const double busy0 = busy(s0), busy1 = busy(s1);
        std::printf("\n%zu rows a pass: stage passes back to back %.2f and %.2f ms; one pass through both stages takes %.2f ms\n", rows, busy0, busy1, serial);
        std::printf("  %-4s %12s %12s %12s\n", "P", "passes/s", "ms a pass", "of slowest");
        for (int P : {1, 2, 3, 4, 6}) {
            Channel<Pass> to0, to1, toh;
            std::atomic<bool> stop{false};
            std::atomic<int> done{0};
            std::vector<double> latency;
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
            std::thread t0([&] { stage(s0, to0, to1); });
            std::thread t1([&] { stage(s1, to1, toh); });
            for (int i = 0; i < P; ++i) {
                Pass p;
                p.id = i;
                p.residual.assign(rows * kEmbd, 0.01f);
                p.started = Clock::now();
                to0.put(std::move(p));
            }
            const auto begin = Clock::now();
            Clock::time_point measured_from{};
            int measured = 0;
            Pass p;
            while (toh.get(p)) {
                const int n = ++done;
                if (n == 3 * P) measured_from = Clock::now();
                if (n > 3 * P) {
                    latency.push_back(ms_since(p.started));
                    ++measured;
                }
                if (ms_since(begin) > 3000 && measured >= 20) break;
                // The host samples the pass's rows and assembles the next one before it re-enters stage 0.
                const auto s = Clock::now();
                while (ms_since(s) < sample_ms) {}
                p.started = Clock::now();
                to0.put(std::move(p));
            }
            const double elapsed = ms_since(measured_from);
            to0.close();
            to1.close();
            toh.close();
            t0.join();
            t1.join();
            std::sort(latency.begin(), latency.end());
            const double rate = measured / (elapsed / 1000);
            const double ideal = 1000 / std::max({busy0, busy1, 1e-9});
            std::printf("  %-4d %12.1f %12.2f %11.0f%%\n", P, rate, latency[latency.size() / 2], 100 * rate / ideal);
        }
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

} // namespace

int main(int argc, char** argv) {
    try {
        quant::register_builtins();
        const std::string mode = argc > 1 ? argv[1] : "";
        int next = 0;
        if (mode == "concurrent") return concurrent(parse_devices(argc, argv, 2, next));
        if (mode == "pipeline" && argc >= 4)
            return pipeline(std::atoi(argv[2]), std::atoi(argv[3]), argc > 4 ? std::atof(argv[4]) : 10.0, argc > 5 ? (size_t)std::atoi(argv[5]) : 0);
        if (mode == "groupsum") return groupsum(parse_devices(argc, argv, 2, next));
        if (mode == "exchange") {
            const auto d = parse_devices(argc, argv, 2, next);
            const size_t tokens = next < argc ? (size_t)std::atoi(argv[next]) : 8;
            const double skew = next + 1 < argc ? std::atof(argv[next + 1]) : 0.0;
            return exchange(d, tokens, skew);
        }
        std::fprintf(stderr, "usage: llmx-multi-device-bench concurrent D... | pipeline D0 D1 [stage_ms] [warm_rows] | groupsum D... | exchange D... [-- tokens skew]\n");
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-multi-device-bench: %s\n", e.what());
        return 1;
    }
}
