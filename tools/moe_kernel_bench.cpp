// Times a MoE layer's routed projections against dense matmuls that read the same bytes, on one Vulkan device, so a routed call's cost splits into what its rows cost and what routing them costs.
// Shapes are Qwen3-30B-A3B's: 2048 wide, 128 experts of 768 rows, 8 per token.
// Usage: llmx-moe-kernel-bench [device] [iters] [gate-type] [down-type] [timed]; types are GGUF ids (12 = Q4_K, 14 = Q6_K, 8 = Q8_0).
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "bench_weights.hpp"
#include "format/gguf.hpp"

namespace {

struct Case {
    std::string name;
    double bytes;   // weight bytes one call reads
    std::function<void()> run;
};

} // namespace

int main(int argc, char** argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    const int iters = argc > 2 ? std::atoi(argv[2]) : 200;
    const uint32_t gt = argc > 3 ? (uint32_t)std::atoi(argv[3]) : gguf::GGML_TYPE_Q4_K;
    const uint32_t dt = argc > 4 ? (uint32_t)std::atoi(argv[4]) : gt;
    const size_t E = 2048, F = 768, NE = 128, K = 8, TMAX = 32;

    // Kernel timestamps on unless the fifth argument is 0, which leaves wall times undisturbed by the queries.
    const bool timed = argc > 5 ? std::atoi(argv[5]) != 0 : true;
    auto bp = backend::make_vulkan_backend(device, timed);
    backend::Backend& b = *bp;

    // Every expert's gate, up and down, stacked as a GGUF stacks them.
    const auto gate_w = weights(row_bytes(gt, E) * F * NE, 1), up_w = weights(row_bytes(gt, E) * F * NE, 2);
    const auto down_w = weights(row_bytes(dt, F) * E * NE, 3);
    const auto gate = b.adopt(gate_w.data(), gate_w.size()), up = b.adopt(up_w.data(), up_w.size());
    const auto down = b.adopt(down_w.data(), down_w.size());

    std::vector<float> xs(TMAX * K * F);
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (auto& v : xs) v = u(rng);
    const auto x = b.adopt(xs.data(), E * TMAX * sizeof(float));
    const auto xf = b.adopt(xs.data(), xs.size() * sizeof(float));
    const auto y = b.alloc(TMAX * E * sizeof(float));
    const auto g = b.alloc(TMAX * K * F * sizeof(float)), up_out = b.alloc(TMAX * K * F * sizeof(float));

    // Each token's k distinct experts, and uniform weights.
    std::vector<uint32_t> ids(TMAX * K);
    std::vector<uint32_t> all(NE);
    std::iota(all.begin(), all.end(), 0u);
    for (size_t t = 0; t < TMAX; ++t) {
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + K, ids.begin() + t * K);
    }
    const std::vector<float> wts(TMAX * K, 1.0f / K);
    const auto idb = b.adopt(ids.data(), ids.size() * sizeof(uint32_t));
    const auto wb = b.adopt(wts.data(), wts.size() * sizeof(float));
    const backend::Backend::Routing routing{{idb.get(), 0}, {wb.get(), 0}, K, NE};

    // Dense stand-ins: the same bytes as k experts' gate and up (or down) as one matrix of short rows and one of long rows.
    const auto dense_short = weights(row_bytes(gt, E) * F * K * 2, 4);
    const auto dense_long = weights(row_bytes(gt, 4 * E) * F * K / 2, 5);
    const auto dense_down = weights(row_bytes(dt, F * K) * E, 6);
    const auto ds = b.adopt(dense_short.data(), dense_short.size()), dl = b.adopt(dense_long.data(), dense_long.size());
    const auto dd = b.adopt(dense_down.data(), dense_down.size());

    std::vector<Case> cases;
    // The floor: dependent dispatches that do almost nothing.
    cases.push_back({"floor silu_mul 2048", 0.0, [&] {
        b.silu_mul({g.get(), 0}, {up_out.get(), 0}, {g.get(), 0}, E);
    }});
    cases.push_back({"floor add 2048", 0.0, [&] { b.add({y.get(), 0}, {xf.get(), 0}, E); }});
    // A MoE layer's small ops at one token: its norm, the F32 router, the choice of experts, the SwiGLU.
    const std::vector<float> ones(E, 1.0f), router_w(E * NE, 0.01f);
    const auto nw = b.adopt(ones.data(), ones.size() * sizeof(float));
    const auto rw = b.adopt(router_w.data(), router_w.size() * sizeof(float));
    const size_t PMAX = 512;   // prompt rows the small ops are timed up to
    const std::vector<float> hs(PMAX * (E + 32), 0.25f);
    const auto hx = b.adopt(hs.data(), hs.size() * sizeof(float));
    const auto h = b.alloc(PMAX * (E + 32) * sizeof(float)), scores = b.alloc(PMAX * NE * sizeof(float));
    const auto rid = b.alloc(PMAX * K * sizeof(float)), rwt = b.alloc(PMAX * K * sizeof(float));
    // The float tile on F32 rows of a dense model's height, for its rate apart from the router's few rows.
    const std::vector<float> f32w(4096 * (E + 32), 0.01f);
    const auto fw = b.adopt(f32w.data(), f32w.size() * sizeof(float));
    const auto fy = b.alloc(PMAX * 4096 * sizeof(float));
    for (size_t T : {size_t(64), size_t(512)})
        for (size_t w : {E, E + 32})
            cases.push_back({"f32 tile 4096x" + std::to_string(w) + " T=" + std::to_string(T), 4096.0 * E * 4, [&, T, w] {
                b.matmul(gguf::GGML_TYPE_F32, {fw.get(), 0}, {h.get(), 0}, {fy.get(), 0}, w, 4096, T);
            }});
    for (size_t T : {size_t(1), size_t(32), size_t(128), size_t(512)}) {
        const std::string t = " T=" + std::to_string(T);
        cases.push_back({"rms_norm" + t, 0.0, [&, T] { b.rms_norm_rows({h.get(), 0}, {hx.get(), 0}, {nw.get(), 0}, T, E, E, 1e-6f); }});
        cases.push_back({"router f32" + t, (double)E * NE * 4, [&, T] {
            b.matmul(gguf::GGML_TYPE_F32, {rw.get(), 0}, {h.get(), 0}, {scores.get(), 0}, E, NE, T);
        }});
        if (T > 1) {
            // The same rows through the row kernel: one run of generated tokens, which takes it at any width.
            cases.push_back({"router f32 rows" + t, (double)E * NE * 4, [&, T] {
                const backend::RowRun run{T, 1};
                b.matmul(gguf::GGML_TYPE_F32, {rw.get(), 0}, {h.get(), 0}, {scores.get(), 0}, E, NE, T, {&run, 1});
            }});
        }
        cases.push_back({"route" + t, 0.0, [&, T] {
            b.route_experts({scores.get(), 0}, T, NE, K, true, {rid.get(), 0}, {rwt.get(), 0});
        }});
        if (T <= TMAX)
            cases.push_back({"silu_mul" + t, 0.0, [&, T] { b.silu_mul({g.get(), 0}, {g.get(), 0}, {up_out.get(), 0}, T * K * F); }});
    }
    const double gu_bytes = 2.0 * row_bytes(gt, E) * F * K, dn_bytes = (double)row_bytes(dt, F) * E * K;
    cases.push_back({"dense " + std::to_string(2 * F * K) + "x" + std::to_string(E), gu_bytes, [&] {
        b.matmul(gt, {ds.get(), 0}, {x.get(), 0}, {g.get(), 0}, E, 2 * F * K, 1);
    }});
    cases.push_back({"dense " + std::to_string(F * K / 2) + "x" + std::to_string(4 * E), gu_bytes, [&] {
        b.matmul(gt, {dl.get(), 0}, {xf.get(), 0}, {g.get(), 0}, 4 * E, F * K / 2, 1);
    }});
    cases.push_back({"dense down " + std::to_string(E) + "x" + std::to_string(F * K), dn_bytes, [&] {
        b.matmul_add(dt, {dd.get(), 0}, {xf.get(), 0}, {y.get(), 0}, F * K, E, 1);
    }});
    // A dense model's shapes, where the row kernels are known to read near the card's rate.
    const auto big = weights(row_bytes(gt, 4 * E) * 3 * 4 * E, 8);
    const auto bw = b.adopt(big.data(), big.size());
    const auto by = b.alloc(3 * 4 * E * sizeof(float));
    for (auto [nin, nout] : {std::pair<size_t, size_t>{4 * E, 4 * E}, {4 * E, 12 * E}, {12 * E, 4 * E}, {E, 3 * F}}) {
        cases.push_back({"dense " + std::to_string(nout) + "x" + std::to_string(nin), (double)row_bytes(gt, nin) * nout,
                         [&, nin, nout] { b.matmul(gt, {bw.get(), 0}, {xf.get(), 0}, {by.get(), 0}, nin, nout, 1); }});
    }
    // More columns over the same rows: what a column's activation reads cost beside the weights.
    for (size_t n : {size_t(2), size_t(4), size_t(8)})
        cases.push_back({"dense " + std::to_string(2 * F * K) + "x" + std::to_string(E) + " n=" + std::to_string(n), gu_bytes, [&, n] {
            b.matmul(gt, {ds.get(), 0}, {xf.get(), 0}, {g.get(), 0}, E, 2 * F * K, n);
        }});
    for (size_t T : {size_t(1), size_t(4), size_t(8), size_t(32)}) {
        cases.push_back({"routed gate+up T=" + std::to_string(T), gu_bytes * T, [&, T] {
            b.matmul_experts({{gt, {gate.get(), 0}, {g.get(), 0}, F}, {gt, {up.get(), 0}, {up_out.get(), 0}, F}},
                             {x.get(), 0}, E, T, routing);
        }});
        cases.push_back({"routed down T=" + std::to_string(T), dn_bytes * T, [&, T] {
            b.matmul_experts_add(dt, {down.get(), 0}, {xf.get(), 0}, {y.get(), 0}, F, E, T, routing);
        }});
    }

    std::printf("device %d, gate/up type %u, down type %u, %d iterations\n", device, gt, dt, iters);
    for (const auto& c : cases) {
        for (int i = 0; i < 5; ++i) c.run();
        b.sync();
        backend::vulkan_kernel_times(b);
        c.run();
        b.sync();
        backend::vulkan_kernel_times(b);
        const size_t per_call = backend::vulkan_timed_dispatches(b);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) c.run();
        b.sync();
        const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        const auto times = backend::vulkan_kernel_times(b);
        // The pool samples an interval's first dispatches, so a call's device time is scaled from the calls sampled.
        const double calls = per_call ? (double)backend::vulkan_timed_dispatches(b) / per_call : 1.0;
        std::string kernels;
        double dev = 0.0;
        for (const auto& t : times) {
            dev += t.second;
            char buf[96];
            std::snprintf(buf, sizeof buf, " %s %.1f", t.first.c_str(), t.second * 1e3 / calls);
            kernels += buf;
        }
        dev = dev * 1e3 / calls;
        std::printf("%-26s wall %7.1f us %6.1f GB/s | device %7.1f us %6.1f GB/s, %zu dispatches:%s\n", c.name.c_str(),
                    wall, c.bytes / (wall * 1e3), dev, c.bytes / (dev * 1e3), per_call, kernels.c_str());
    }
    return 0;
}
