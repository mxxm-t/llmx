// Vulkan backend (docs/VULKAN.md). Sub-step 1: storage and submission over
// a real device. Buffers round-trip through adopt, copy, write and read;
// allocations come back zeroed; host-visible memory is readable in place
// after a wait; tickets are monotonic and retire in order. Sub-step 2: the
// elementwise kernels, gather, embed and the norms against the CPU backend
// on random inputs, with the bounds below fixed before the first run: bit
// exact where the arithmetic is the same operation in the same order (add,
// gather, embed), and a stated relative tolerance where a transcendental or
// a reduction order differs. Exits 77, which CTest reports as skipped, when
// there is no loader or no device.
#include <chrono>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "model/kv_cache.hpp"
#include "quant/quant.hpp"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::vector<float> uniform(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// The same op on both backends over the same inputs. Inputs are adopted
// (the CPU keeps the pointer, the device uploads a copy); outputs are
// allocated on each and the device's is read back.
struct Pair {
    backend::CpuBackend cpu;
    backend::Backend& vk;
    explicit Pair(backend::Backend& v) : vk(v) { cpu.set_threads(1); }
    struct In {
        backend::BufferPtr c, v;
        backend::CSlice cs() const { return {c.get(), 0}; }
        backend::CSlice vs() const { return {v.get(), 0}; }
    };
    struct Out {
        backend::BufferPtr c, v;
        size_t n;
        backend::Slice cs() const { return {c.get(), 0}; }
        backend::Slice vs() const { return {v.get(), 0}; }
    };
    In in(const void* data, size_t bytes) {
        return {cpu.adopt(data, bytes), vk.adopt(data, bytes)};
    }
    In in(const std::vector<float>& f) { return in(f.data(), f.size() * sizeof(float)); }
    Out out(size_t floats) {
        return {cpu.alloc(floats * sizeof(float), backend::Memory::device),
                vk.alloc(floats * sizeof(float), backend::Memory::device), floats};
    }
    // Both results, with the device's read back.
    std::pair<std::vector<float>, std::vector<float>> results(const Out& o) {
        std::vector<float> a(o.n), b(o.n);
        cpu.read(*o.c, 0, a.data(), o.n * sizeof(float));
        vk.read(*o.v, 0, b.data(), o.n * sizeof(float));
        return {a, b};
    }
};

size_t exact(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, what);
    for (float v : b) require(std::isfinite(v), "nonfinite device output");
    return a.size();
}

size_t close(const std::vector<float>& a, const std::vector<float>& b, double rel, const char* what) {
    require(a.size() == b.size(), what);
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(b[i]), "nonfinite device output");
        require(std::fabs((double)a[i] - b[i]) <= rel * (1.0 + std::fabs((double)a[i])), what);
    }
    return a.size();
}

size_t check_kernels(backend::Backend& vk) {
    Pair p(vk);
    size_t values = 0;

    // add: same operation in the same order, so exact.
    {
        const size_t n = 100003;
        const auto dst0 = uniform(n, 1), src = uniform(n, 2);
        Pair::In s = p.in(src);
        Pair::Out d = p.out(n);
        p.cpu.write(*d.c, 0, dst0.data(), n * sizeof(float));
        p.vk.write(*d.v, 0, dst0.data(), n * sizeof(float));
        p.cpu.add(d.cs(), s.cs(), n);
        p.vk.add(d.vs(), s.vs(), n);
        auto r = p.results(d);
        values += exact(r.first, r.second, "add differs");
    }
    // silu_mul: exp differs between libm and the device, so a tolerance.
    {
        const size_t n = 50001;
        const auto g = uniform(n, 3, -6.0f, 6.0f), u = uniform(n, 4);
        Pair::In gi = p.in(g), ui = p.in(u);
        Pair::Out d = p.out(n);
        p.cpu.silu_mul(d.cs(), gi.cs(), ui.cs(), n);
        p.vk.silu_mul(d.vs(), gi.vs(), ui.vs(), n);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-6, "silu_mul differs beyond 1e-6");
    }
    // gather_rows: copies, so exact; a row beyond the source is refused.
    {
        const size_t width = 37, rows = 5;
        const auto src = uniform(width * rows, 5);
        const uint32_t pick[4] = {4, 1, 1, 0};
        Pair::In s = p.in(src);
        Pair::Out d = p.out(width * 4);
        p.cpu.gather_rows(d.cs(), s.cs(), width, pick, 4);
        p.vk.gather_rows(d.vs(), s.vs(), width, pick, 4);
        auto r = p.results(d);
        values += exact(r.first, r.second, "gather_rows differs");
        const uint32_t beyond[1] = {5};
        bool rejected = false;
        try { p.vk.gather_rows(d.vs(), s.vs(), width, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "gather beyond the source accepted");
    }
    // rms_norm_rows and rms_norm: the sum of squares is reduced in a
    // different order, so a tolerance; dst aliasing src is exercised.
    {
        const size_t rows = 7, n = 1000, stride = 1013;
        const auto src = uniform(rows * stride, 6), w = uniform(n, 7, 0.5f, 1.5f);
        Pair::In s = p.in(src), wi = p.in(w);
        Pair::Out d = p.out(rows * stride);
        p.cpu.rms_norm_rows(d.cs(), s.cs(), wi.cs(), rows, n, stride, 1e-6f);
        p.vk.rms_norm_rows(d.vs(), s.vs(), wi.vs(), rows, n, stride, 1e-6f);
        auto r = p.results(d);
        for (size_t row = 0; row < rows; ++row) {
            std::vector<float> a(r.first.begin() + row * stride, r.first.begin() + row * stride + n);
            std::vector<float> b(r.second.begin() + row * stride, r.second.begin() + row * stride + n);
            values += close(a, b, 1e-5, "rms_norm_rows differs beyond 1e-5");
        }
        Pair::Out one = p.out(n);
        p.cpu.write(*one.c, 0, src.data(), n * sizeof(float));
        p.vk.write(*one.v, 0, src.data(), n * sizeof(float));
        p.cpu.rms_norm(one.cs(), one.cs(), wi.cs(), n, 1e-6f);
        p.vk.rms_norm(one.vs(), one.vs(), wi.vs(), n, 1e-6f);
        auto q = p.results(one);
        values += close(q.first, q.second, 1e-5, "rms_norm in place differs beyond 1e-5");
    }
    // norm_rope_rows: positions per row out of order, a padded stride.
    {
        const size_t rows = 5, heads = 3, half = 64, dim = 2 * half, stride = heads * dim + 8, table = 12;
        const auto x = uniform(rows * stride, 8), w = uniform(dim, 9, 0.5f, 1.5f);
        std::vector<float> cs(table * half), sn(table * half);
        for (size_t t = 0; t < table; ++t)
            for (size_t i = 0; i < half; ++i) {
                const double f = std::pow(10000.0, -2.0 * double(i) / double(dim));
                cs[t * half + i] = float(std::cos(double(t) * f));
                sn[t * half + i] = float(std::sin(double(t) * f));
            }
        const uint32_t pos[rows] = {5, 2, 9, 0, 11};
        Pair::In wi = p.in(w), ci = p.in(cs), si = p.in(sn);
        Pair::Out d = p.out(rows * stride);
        p.cpu.write(*d.c, 0, x.data(), x.size() * sizeof(float));
        p.vk.write(*d.v, 0, x.data(), x.size() * sizeof(float));
        p.cpu.norm_rope_rows(d.cs(), rows, stride, heads, wi.cs(), 1e-6f, ci.cs(), si.cs(), half, pos);
        p.vk.norm_rope_rows(d.vs(), rows, stride, heads, wi.vs(), 1e-6f, ci.vs(), si.vs(), half, pos);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-5, "norm_rope_rows differs beyond 1e-5");
        const uint32_t beyond[1] = {12};
        bool rejected = false;
        try { p.vk.norm_rope_rows(d.vs(), 1, stride, heads, wi.vs(), 1e-6f, ci.vs(), si.vs(), half, beyond); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "position beyond the table accepted");
    }
    // embed: F32 rows are copies and Q8_0 rows are a half scale times a
    // small integer, exact in float, so both are exact.
    {
        const size_t nin = 96, nrows = 10;
        const auto table = uniform(nin * nrows, 10);
        const uint32_t ids[3] = {3, 9, 0};
        Pair::In t = p.in(table);
        Pair::Out d = p.out(nin * 3);
        p.cpu.embed(d.cs(), gguf::GGML_TYPE_F32, t.cs(), nin, nrows, ids, 3);
        p.vk.embed(d.vs(), gguf::GGML_TYPE_F32, t.vs(), nin, nrows, ids, 3);
        auto r = p.results(d);
        values += exact(r.first, r.second, "embed F32 differs");

        std::vector<uint8_t> q(nrows * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q8_0(table.data() + row * nin,
                                     q.data() + row * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE,
                                     nin / gguf::Q8_0_BLOCK);
        Pair::In tq = p.in(q.data(), q.size());
        Pair::Out dq = p.out(nin * 3);
        p.cpu.embed(dq.cs(), gguf::GGML_TYPE_Q8_0, tq.cs(), nin, nrows, ids, 3);
        p.vk.embed(dq.vs(), gguf::GGML_TYPE_Q8_0, tq.vs(), nin, nrows, ids, 3);
        auto rq = p.results(dq);
        values += exact(rq.first, rq.second, "embed Q8_0 differs");
        const uint32_t beyond[1] = {10};
        bool rejected = false;
        try { p.vk.embed(d.vs(), gguf::GGML_TYPE_F32, t.vs(), nin, nrows, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "embedding row beyond the table accepted");
    }
    // matmul: F32 and Q8_0 over odd sizes and batch widths that fall
    // inside, on and past the eight-column chunk. The reduction order
    // differs from the CPU's, so a tolerance.
    for (size_t nin : {size_t(256), size_t(224)}) {
        // 256 is eight blocks, the word-wide path; 224 is seven, the 16-bit path.
        const size_t nout = 67;
        const auto wf = uniform(nin * nout, 11);
        std::vector<uint8_t> wq(nout * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q8_0(wf.data() + row * nin,
                                     wq.data() + row * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE,
                                     nin / gguf::Q8_0_BLOCK);
        Pair::In wfi = p.in(wf), wqi = p.in(wq.data(), wq.size());
        // 1 to 13 take the row kernel; 16, 64, 100 and 247 the tile kernel,
        // on, inside and past its 64-column tiles.
        for (size_t nbatch : {size_t(1), size_t(3), size_t(8), size_t(13), size_t(16), size_t(64),
                              size_t(100), size_t(247)}) {
            const auto x = uniform(nbatch * nin, 12 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            for (int q = 0; q < 2; ++q) {
                const uint32_t type = q ? gguf::GGML_TYPE_Q8_0 : gguf::GGML_TYPE_F32;
                const Pair::In& wi = q ? wqi : wfi;
                Pair::Out d = p.out(nbatch * nout);
                p.cpu.matmul(type, wi.cs(), xi.cs(), d.cs(), nin, nout, nbatch);
                p.vk.matmul(type, wi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
                auto r = p.results(d);
                values += close(r.first, r.second, 1e-4, q ? "Q8_0 matmul differs beyond 1e-4"
                                                            : "F32 matmul differs beyond 1e-4");
            }
        }
        bool rejected = false;
        try { p.vk.matmul(gguf::GGML_TYPE_Q4_0, wqi.vs(), wqi.vs(), p.out(8).vs(), nin, 1, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported matrix type accepted");
    }
    // The KV cache: the same token-major rows written through each backend's
    // own storage and block size, then attention over each backend's own
    // view. Histories straddle the device's 64-token blocks and the CPU's
    // 128; two views in one call; a block copied with kv_copy attends like
    // the original. The online softmax orders the arithmetic differently
    // from the CPU's global softmax, so a tolerance.
    {
        const int n_head = 4, n_head_kv = 2, head_dim = 40;
        const size_t kvw = (size_t)n_head_kv * head_dim, qw = (size_t)n_head * head_dim, layers = 2;
        for (size_t n_past : {size_t(0), size_t(63), size_t(64), size_t(65), size_t(131)}) {
            for (size_t nq : {size_t(1), size_t(3)}) {
                auto run = [&](backend::Backend& b, const std::vector<float>& K, const std::vector<float>& V,
                               const std::vector<float>& Q, std::vector<float>& out, bool split) {
                    const size_t bt = b.kv_layout().block_tokens;
                    auto st = b.kv_alloc(layers, n_head_kv, head_dim, 8 * 128);
                    infer::BlockPool pool(st->max_blocks());
                    infer::KVSequence seq(&pool, bt), other(&pool, bt);
                    const auto Kb = b.adopt(K.data(), K.size() * sizeof(float));
                    const auto Vb = b.adopt(V.data(), V.size() * sizeof(float));
                    const auto Qb = b.adopt(Q.data(), Q.size() * sizeof(float));
                    const auto ob = b.alloc(out.size() * sizeof(float), backend::Memory::device);
                    // The history, then the queries' own rows, on every layer.
                    seq.prepare(n_past);
                    for (size_t l = 0; l < layers; ++l) {
                        const backend::KVView h = seq.view(st.get());
                        b.kv_write(l, &h, 1, {Kb.get(), l * (n_past + nq) * kvw}, {Vb.get(), l * (n_past + nq) * kvw});
                    }
                    seq.commit();
                    if (split) {
                        // A second sequence with a two-token history shares
                        // the call: its rows come after the first view's.
                        other.prepare(2);
                        for (size_t l = 0; l < layers; ++l) {
                            const backend::KVView h = other.view(st.get());
                            b.kv_write(l, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                        }
                        other.commit();
                    }
                    seq.prepare(nq);
                    const backend::KVView view = seq.view(st.get());
                    for (size_t l = 0; l < layers; ++l) {
                        const size_t tail = (l * (n_past + nq) + n_past) * kvw;
                        b.kv_write(l, &view, 1, {Kb.get(), tail}, {Vb.get(), tail});
                        b.attention({Qb.get(), l * nq * qw}, l, &view, 1, {ob.get(), l * nq * qw},
                                    n_head, n_head_kv, head_dim);
                    }
                    if (split) {
                        other.prepare(1);
                        const backend::KVView views[2] = {seq.view(st.get()), other.view(st.get())};
                        std::vector<float> two((nq + 1) * qw, 0.0f);
                        const auto tb = b.alloc(two.size() * sizeof(float), backend::Memory::device);
                        std::vector<float> qq(Q.begin(), Q.begin() + nq * qw);
                        qq.insert(qq.end(), Q.begin(), Q.begin() + qw);
                        const auto qqb = b.adopt(qq.data(), qq.size() * sizeof(float));
                        b.attention({qqb.get(), 0}, 0, views, 2, {tb.get(), 0}, n_head, n_head_kv, head_dim);
                        b.read(*tb, 0, two.data(), two.size() * sizeof(float));
                        out.insert(out.end(), two.begin(), two.end());
                        other.abort();
                    }
                    std::vector<float> got(layers * nq * qw);
                    b.read(*ob, 0, got.data(), got.size() * sizeof(float));
                    std::copy(got.begin(), got.end(), out.begin());
                    seq.commit();
                    // A copied block attends like the block it came from.
                    if (n_past >= 128) {   // both backends have a full first block
                        const backend::KVView v0 = seq.view(st.get());
                        const int32_t spare = pool.alloc();
                        b.kv_copy(*st, v0.blocks[0], spare);
                        std::vector<int32_t> table(v0.blocks, v0.blocks + v0.n_blocks);
                        table[0] = spare;
                        backend::KVView copied{st.get(), table.data(), table.size(), n_past, nq};
                        std::vector<float> again(nq * qw, 0.0f);
                        const auto ab = b.alloc(again.size() * sizeof(float), backend::Memory::device);
                        b.attention({Qb.get(), 0}, 0, &copied, 1, {ab.get(), 0}, n_head, n_head_kv, head_dim);
                        b.read(*ab, 0, again.data(), again.size() * sizeof(float));
                        out.insert(out.end(), again.begin(), again.end());
                        pool.release(spare);
                    }
                };
                const size_t seq_len = n_past + nq;
                const auto K = uniform(layers * seq_len * kvw, 20 + (uint32_t)n_past);
                const auto V = uniform(layers * seq_len * kvw, 21 + (uint32_t)n_past);
                const auto Q = uniform(layers * nq * qw, 22 + (uint32_t)nq);
                std::vector<float> a(layers * nq * qw), b(layers * nq * qw);
                run(p.cpu, K, V, Q, a, true);
                run(vk, K, V, Q, b, true);
                double worst = 0; size_t at = 0;
                for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
                    const double d = std::fabs((double)a[i] - b[i]) / (1.0 + std::fabs((double)a[i]));
                    if (d > worst) { worst = d; at = i; }
                }
                if (worst > 1e-4 || a.size() != b.size())
                    std::cerr << "attention n_past " << n_past << " nq " << nq << " sizes " << a.size() << "/" << b.size()
                              << " worst " << worst << " at " << at << " cpu " << a[at] << " vk " << b[at]
                              << " (layer region " << at / (nq * qw) << ")\n";
                values += close(a, b, 1e-4, "attention over the device cache differs beyond 1e-4");
            }
        }
    }
    // The cost of a dispatch that does almost nothing, reported and not
    // asserted: a decoded token on Qwen3-0.6B is about four hundred of them.
    {
        const size_t n = 64;
        const auto x = uniform(n, 14);
        const auto a = vk.adopt(x.data(), n * sizeof(float));
        const auto d = vk.alloc(n * sizeof(float), backend::Memory::device);
        vk.add({d.get(), 0}, {a.get(), 0}, n);
        vk.sync();
        const int iters = 2000;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) vk.add({d.get(), 0}, {a.get(), 0}, n);
        vk.sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: tiny dispatch with barrier " << us << " us each over " << iters << "\n";
    }
    // The row kernel at the projection shapes of Qwen3-0.6B and 8B, one
    // column, reported: the small shapes say whether a decoded token is
    // bound by bandwidth or by per-kernel latency.
    for (auto shape : {std::pair<size_t, size_t>{1024, 1024}, {1024, 2048}, {1024, 3072},
                       {3072, 1024}, {4096, 4096}, {4096, 12288}, {12288, 4096}}) {
        const size_t nin = shape.first, nout = shape.second;
        std::vector<uint8_t> wq(nout * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 7 + 3);
        const auto x = uniform(nin, 15);
        const auto w = vk.adopt(wq.data(), wq.size());
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(nout * sizeof(float), backend::Memory::device);
        vk.matmul(gguf::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, 1);
        vk.sync();
        const int iters = 100;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul(gguf::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, 1);
        vk.sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: Q8_0 matvec " << nin << "x" << nout << " " << us << " us, "
                  << (double)wq.size() / us / 1e3 << " GB/s\n";
    }
    // Decode attention and the small kernels at the Qwen3-0.6B shape over a
    // 250-token history, one query, reported.
    {
        const int n_head = 16, n_head_kv = 8, head_dim = 128;
        const size_t hist = 250, kvw = (size_t)n_head_kv * head_dim, qw = (size_t)n_head * head_dim;
        auto st = vk.kv_alloc(1, n_head_kv, head_dim, 1024);
        infer::BlockPool pool(st->max_blocks());
        infer::KVSequence seq(&pool, vk.kv_layout().block_tokens);
        const auto K = uniform(hist * kvw, 16), V = uniform(hist * kvw, 17), Q = uniform(qw, 18);
        const auto Kb = vk.adopt(K.data(), K.size() * sizeof(float));
        const auto Vb = vk.adopt(V.data(), V.size() * sizeof(float));
        const auto Qb = vk.adopt(Q.data(), Q.size() * sizeof(float));
        const auto ob = vk.alloc(qw * sizeof(float), backend::Memory::device);
        seq.prepare(hist);
        const backend::KVView h = seq.view(st.get());
        vk.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
        seq.commit();
        seq.prepare(1);
        const backend::KVView view = seq.view(st.get());
        vk.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        vk.sync();
        auto time = [&](const char* what, const std::function<void()>& op) {
            const int iters = 200;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) op();
            vk.sync();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::cout << "backend-vulkan: " << what << " " << us << " us\n";
        };
        time("attention 16 heads over 250 tokens", [&] {
            vk.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        });
        const auto w = uniform(1024, 19, 0.5f, 1.5f);
        const auto wb = vk.adopt(w.data(), w.size() * sizeof(float));
        const auto xb = vk.alloc(3072 * sizeof(float), backend::Memory::device);
        time("rms_norm_rows 1x1024", [&] { vk.rms_norm_rows({xb.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, 1, 1024, 1024, 1e-6f); });
        const auto cs = uniform(1024 * 64, 20), sn = uniform(1024 * 64, 21);
        const auto cb = vk.adopt(cs.data(), cs.size() * sizeof(float)), sb = vk.adopt(sn.data(), sn.size() * sizeof(float));
        const auto hw = uniform(128, 22, 0.5f, 1.5f);
        const auto hwb = vk.adopt(hw.data(), hw.size() * sizeof(float));
        const uint32_t pos0 = 7;
        time("norm_rope_rows 1 row 16 heads", [&] {
            vk.norm_rope_rows({xb.get(), 0}, 1, 0, 16, {hwb.get(), 0}, 1e-6f, {cb.get(), 0}, {sb.get(), 0}, 64, &pos0);
        });
        time("silu_mul 3072", [&] { vk.silu_mul({xb.get(), 0}, {xb.get(), 0}, {xb.get(), 0}, 3072); });
        time("kv_write 1 row", [&] { vk.kv_write(0, &view, 1, {Kb.get(), 0}, {Vb.get(), 0}); });
    }
    // Decode bandwidth of the row kernel on a Qwen3-8B-sized projection,
    // reported and not asserted: 4096 x 4096 Q8_0 is 17 MiB per column.
    {
        const size_t n = 4096;
        std::vector<uint8_t> wq(n * (n / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 7 + 3);
        const auto x = uniform(n, 13);
        const auto w = vk.adopt(wq.data(), wq.size());
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(n * sizeof(float), backend::Memory::device);
        vk.matmul(gguf::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const int iters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul(gguf::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: Q8_0 matvec 4096x4096 " << ms << " ms, "
                  << (double)wq.size() / ms / 1e6 << " GB/s\n";
    }
    return values;
}

std::vector<uint8_t> pattern(size_t bytes, uint32_t seed) {
    std::vector<uint8_t> v(bytes);
    uint32_t x = seed;
    for (size_t i = 0; i < bytes; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = uint8_t(x >> 24);
    }
    return v;
}
}

int main() {
    backend::BackendPtr b;
    try {
        b = backend::make_vulkan_backend(0);
    } catch (const backend::VulkanUnavailable& e) {
        std::cout << "backend-vulkan: skipped: " << e.what() << "\n";
        return 77;
    }
    try {
        std::cout << "backend-vulkan: " << backend::vulkan_device_name(*b) << "\n";
        size_t checks = 0;

        // Allocations are zeroed, on the device and in host-visible memory.
        const size_t mib = 1u << 20;
        const auto zero = b->alloc(mib);
        std::vector<uint8_t> out(mib, 0xff);
        b->read(*zero, 0, out.data(), mib);
        for (uint8_t v : out) require(v == 0, "device allocation not zeroed");
        require(zero->host_ptr() == nullptr, "device memory reported a host address");
        const auto visible = b->alloc(4096, backend::Memory::host_visible);
        require(visible->host_ptr() != nullptr, "host-visible memory has no host address");
        b->sync();
        for (size_t i = 0; i < 4096; ++i)
            require(((const uint8_t*)visible->host_ptr())[i] == 0, "host-visible allocation not zeroed");
        checks += 2;

        // Adopt uploads a copy; the source may change afterwards.
        std::vector<uint8_t> src = pattern(3 * mib + 12345, 1);
        const std::vector<uint8_t> kept = src;
        const auto adopted = b->adopt(src.data(), src.size());
        std::fill(src.begin(), src.end(), uint8_t(0));
        out.assign(kept.size(), 0);
        b->read(*adopted, 0, out.data(), kept.size());
        require(out == kept, "adopted bytes differ after upload");
        checks += 1;

        // Copy within the device at odd offsets, then read a window.
        const auto dst = b->alloc(kept.size() + 100);
        b->copy(*dst, 100, *adopted, 0, kept.size());
        b->copy(*dst, 0, *adopted, 7, 100);
        out.assign(kept.size() + 100, 0);
        b->read(*dst, 0, out.data(), out.size());
        require(std::memcmp(out.data() + 100, kept.data(), kept.size()) == 0, "device copy differs");
        require(std::memcmp(out.data(), kept.data() + 7, 100) == 0, "offset copy differs");
        std::vector<uint8_t> window(1000);
        b->read(*dst, 100 + 2 * mib + 3, window.data(), window.size());
        require(std::memcmp(window.data(), kept.data() + 2 * mib + 3, window.size()) == 0,
                "windowed read differs");
        checks += 3;

        // Write from the host into device memory, then into host-visible
        // memory, and a copy whose result the host reads in place.
        const std::vector<uint8_t> patch = pattern(777, 2);
        b->write(*dst, 5000, patch.data(), patch.size());
        b->read(*dst, 5000, window.data(), patch.size());
        require(std::memcmp(window.data(), patch.data(), patch.size()) == 0, "device write differs");
        b->write(*visible, 8, patch.data(), 1000);
        b->copy(*visible, 2000, *adopted, 4096, 2000);
        const backend::Ticket t = b->submit();
        b->wait(t);
        require(std::memcmp((const uint8_t*)visible->host_ptr() + 8, patch.data(), 1000) == 0,
                "host-visible write differs");
        require(std::memcmp((const uint8_t*)visible->host_ptr() + 2000, kept.data() + 4096, 2000) == 0,
                "copy into host-visible memory not visible after wait");
        checks += 3;

        // Tickets are monotonic and every one retires.
        const backend::Ticket t1 = b->submit(), t2 = b->submit();
        require(t2 > t1 && t1 > t, "tickets are not monotonic");
        b->wait(t2);
        b->wait(t1);
        b->sync();
        b->copy(*dst, 0, *adopted, 0, 64);
        b->sync();
        b->read(*dst, 0, window.data(), 64);
        require(std::memcmp(window.data(), kept.data(), 64) == 0, "work before sync not retired");
        checks += 2;

        // Empty allocations and ranges outside an allocation.
        const auto empty = b->alloc(0);
        require(empty->size() == 0, "empty allocation has a size");
        b->read(*empty, 0, window.data(), 0);
        bool rejected = false;
        try { b->read(*dst, kept.size() + 100 - 10, window.data(), 11); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "read past the allocation accepted");
        rejected = false;
        try { b->copy(*dst, 0, *adopted, kept.size(), 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "copy past the source accepted");
        checks += 2;

        quant::register_builtins();
        const size_t values = check_kernels(*b);
        std::cout << "backend-vulkan: " << checks << " storage and submission checks; "
                  << values << " kernel outputs against the CPU backend\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
