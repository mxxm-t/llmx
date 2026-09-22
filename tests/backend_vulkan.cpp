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
#include <fstream>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    for (float v : b) require(std::isfinite(v), (std::string("nonfinite device output: ") + what).c_str());
    return a.size();
}

size_t close(const std::vector<float>& a, const std::vector<float>& b, double rel, const char* what) {
    require(a.size() == b.size(), what);
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) throw std::runtime_error(std::string("nonfinite device output: ") + what);
        if (!(std::fabs((double)a[i] - b[i]) <= rel * (1.0 + std::fabs((double)a[i])))) {
            std::fprintf(stderr, "  [%zu] cpu %.9g device %.9g\n", i, a[i], b[i]);
            require(false, what);
        }
    }
    return a.size();
}

// The activations as the device's row kernel sees them: each block of 32
// scaled so its largest magnitude is 32767, rounded half away from zero,
// and back to floats (shaders/quantize_x.comp). The CPU reference of a
// quantized-row matmul on the row kernel takes these, so the comparison
// is about the dot and its reduction order and not about the
// quantization, which is the device's choice and the HF gate's business.
std::vector<float> row_activations(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t b = 0; b + 32 <= x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b + i]));
        const float d = amax / 32767.0f, id = amax > 0.0f ? 32767.0f / amax : 0.0f;
        for (size_t i = 0; i < 32; ++i) {
            const float r = x[b + i] * id;
            int q = (int)(std::copysign(std::floor(std::fabs(r) + 0.5f), r));
            q = std::max(-32767, std::min(32767, q));
            out[b + i] = (float)q * d;
        }
    }
    return out;
}

// The same activations rounded to 8 bits per block of 32 and back, as the integer-dot prefill tile reads them (shaders/quantize_x8.comp). A device that multiplies wide quantized batches that way is compared against a reference fed these, for the same reason as above.
std::vector<float> tile_activations8(const std::vector<float>& x) {
    std::vector<float> out(x.size());
    for (size_t b = 0; b + 32 <= x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b + i]));
        const float d = amax / 127.0f, id = amax > 0.0f ? 127.0f / amax : 0.0f;
        for (size_t i = 0; i < 32; ++i) {
            const float r = x[b + i] * id;
            int q = (int)(std::copysign(std::floor(std::fabs(r) + 0.5f), r));
            q = std::max(-127, std::min(127, q));
            out[b + i] = (float)q * d;
        }
    }
    return out;
}

// The activations as a row family reads them: on a device whose integer dot is native the Q4_K and Q5_K families read the 8-bit twin and the others the 16-bit one; elsewhere every family reads the 16-bit one.
std::vector<float> twin_activations(const std::vector<float>& x, bool twin8) {
    return twin8 ? tile_activations8(x) : row_activations(x);
}

size_t check_kernels(backend::Backend& vk) {
    Pair p(vk);
    // Whether this device's producers and row kernels use the 8-bit twin, and the tolerance a reference fed it needs.
    // Against 8-bit activations one quant can round the other way on the device, whose reciprocal is a few ulps from the host's, and a flip is worth the weight times the block's step, about 0.008 on these inputs whatever the output: an output whose products cancel read 0.912 against 0.906.
    // An indexing error is worth the output itself, so a bound of 1e-2 still separates the two.
    const bool twin8 = backend::vulkan_device_profile(p.vk).prefer_integer_dot;
    const double twin_tol = twin8 ? 1e-2 : 1e-4;
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

        std::vector<uint8_t> q4(nrows * (nin / gguf::Q4_0_BLOCK) * gguf::Q4_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q4_0(table.data() + row * nin,
                                     q4.data() + row * (nin / gguf::Q4_0_BLOCK) * gguf::Q4_0_TYPESIZE,
                                     nin / gguf::Q4_0_BLOCK);
        Pair::In t4 = p.in(q4.data(), q4.size());
        Pair::Out d4 = p.out(nin * 3);
        p.cpu.embed(d4.cs(), gguf::GGML_TYPE_Q4_0, t4.cs(), nin, nrows, ids, 3);
        p.vk.embed(d4.vs(), gguf::GGML_TYPE_Q4_0, t4.vs(), nin, nrows, ids, 3);
        auto r4 = p.results(d4);
        values += exact(r4.first, r4.second, "embed Q4_0 differs");

        // Q6_K rows of 256 from fixed bytes, decoded by both.
        {
            const size_t n6 = 512;
            std::vector<uint8_t> q6(nrows * (n6 / 256) * gguf::Q6_K_TYPESIZE);
            for (size_t i = 0; i < q6.size(); ++i) q6[i] = uint8_t(i * 37 + 11);
            for (size_t b = 0; b < nrows * (n6 / 256); ++b) {
                q6[b * gguf::Q6_K_TYPESIZE + 208] = 0x00;
                q6[b * gguf::Q6_K_TYPESIZE + 209] = 0x30;
            }
            Pair::In t6 = p.in(q6.data(), q6.size());
            Pair::Out d6 = p.out(n6 * 3);
            p.cpu.embed(d6.cs(), gguf::GGML_TYPE_Q6_K, t6.cs(), n6, nrows, ids, 3);
            p.vk.embed(d6.vs(), gguf::GGML_TYPE_Q6_K, t6.vs(), n6, nrows, ids, 3);
            auto r6 = p.results(d6);
            values += exact(r6.first, r6.second, "embed Q6_K differs");
        }
        // Q4_K and Q5_K rows of 256 the same way; d and dmin are the first
        // two halves of a block.
        for (int k = 0; k < 2; ++k) {
            const uint32_t type = k == 0 ? gguf::GGML_TYPE_Q4_K : gguf::GGML_TYPE_Q5_K;
            const size_t bytes = k == 0 ? gguf::Q4_K_TYPESIZE : gguf::Q5_K_TYPESIZE;
            const size_t nk = 512;
            std::vector<uint8_t> qk(nrows * (nk / 256) * bytes);
            for (size_t i = 0; i < qk.size(); ++i) qk[i] = uint8_t(i * 53 + 5 + k);
            for (size_t b = 0; b < nrows * (nk / 256); ++b) {
                qk[b * bytes + 0] = 0x00; qk[b * bytes + 1] = 0x30;
                qk[b * bytes + 2] = 0x00; qk[b * bytes + 3] = 0x2c;
            }
            Pair::In tk = p.in(qk.data(), qk.size());
            Pair::Out dk = p.out(nk * 3);
            p.cpu.embed(dk.cs(), type, tk.cs(), nk, nrows, ids, 3);
            p.vk.embed(dk.vs(), type, tk.vs(), nk, nrows, ids, 3);
            auto rk = p.results(dk);
            values += exact(rk.first, rk.second, k == 0 ? "embed Q4_K differs" : "embed Q5_K differs");
        }
        const uint32_t beyond[1] = {10};
        bool rejected = false;
        try { p.vk.embed(d.vs(), gguf::GGML_TYPE_F32, t.vs(), nin, nrows, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "embedding row beyond the table accepted");
    }
    // matmul: F32 and Q8_0 over odd sizes and batch widths that fall
    // inside, on and past the eight-column chunk. The reduction order
    // differs from the CPU's, so a tolerance.
    for (size_t nin : {size_t(1024), size_t(256), size_t(224)}) {
        // 1024 is sixteen block pairs, the word-wide path; 256 has too few
        // pairs for it and 224 an odd block count, both the 16-bit path.
        const size_t nout = 67;
        const auto wf = uniform(nin * nout, 11);
        std::vector<uint8_t> wq(nout * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q8_0(wf.data() + row * nin,
                                     wq.data() + row * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE,
                                     nin / gguf::Q8_0_BLOCK);
        std::vector<uint8_t> w4(nout * (nin / gguf::Q4_0_BLOCK) * gguf::Q4_0_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q4_0(wf.data() + row * nin,
                                     w4.data() + row * (nin / gguf::Q4_0_BLOCK) * gguf::Q4_0_TYPESIZE,
                                     nin / gguf::Q4_0_BLOCK);
        std::vector<uint8_t> w41(nout * (nin / gguf::Q4_1_BLOCK) * gguf::Q4_1_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q4_1(wf.data() + row * nin,
                                     w41.data() + row * (nin / gguf::Q4_1_BLOCK) * gguf::Q4_1_TYPESIZE,
                                     nin / gguf::Q4_1_BLOCK);
        // No Q6_K quantizer exists here, and none is needed: any bytes are
        // a valid block, and both backends decode the same bytes. The half
        // scale is 2^-10 so the values sit in the range of the other types;
        // at 2^-4 the sub-scales of up to 127 gave 1024-term sums whose
        // reduction-order rounding alone exceeded the tolerance.
        std::vector<uint8_t> w6(nout * (nin / 256) * gguf::Q6_K_TYPESIZE);
        for (size_t i = 0; i < w6.size(); ++i) w6[i] = uint8_t(i * 131 + 7);
        for (size_t row = 0; row < nout * (nin / 256); ++row) {
            // Keep the half scale finite and small.
            w6[row * gguf::Q6_K_TYPESIZE + 208] = 0x00;
            w6[row * gguf::Q6_K_TYPESIZE + 209] = 0x14;
        }
        // Q4_K and Q5_K likewise, d and dmin at 2^-10 and 2^-11.
        std::vector<uint8_t> w4k(nout * (nin / 256) * gguf::Q4_K_TYPESIZE), w5k(nout * (nin / 256) * gguf::Q5_K_TYPESIZE);
        for (size_t i = 0; i < w4k.size(); ++i) w4k[i] = uint8_t(i * 61 + 3);
        for (size_t i = 0; i < w5k.size(); ++i) w5k[i] = uint8_t(i * 67 + 9);
        for (size_t row = 0; row < nout * (nin / 256); ++row) {
            for (uint8_t* blk : {w4k.data() + row * gguf::Q4_K_TYPESIZE, w5k.data() + row * gguf::Q5_K_TYPESIZE}) {
                blk[0] = 0x00; blk[1] = 0x14; blk[2] = 0x00; blk[3] = 0x10;
            }
        }
        Pair::In wfi = p.in(wf), wqi = p.in(wq.data(), wq.size()), w4i = p.in(w4.data(), w4.size());
        Pair::In w41i = p.in(w41.data(), w41.size()), w6i = p.in(w6.data(), w6.size());
        Pair::In w4ki = p.in(w4k.data(), w4k.size()), w5ki = p.in(w5k.data(), w5k.size());
        // The row kernel, whose quantized rows meet 16-bit activations,
        // takes batches below a threshold that depends on the device and on
        // how wide a row is; the tile kernel takes the rest with float
        // activations. The thresholds come from the backend rather than from
        // constants here, because a device that was measured to want other
        // numbers gets them and the reference has to follow.
        const backend::DeviceProfile profile = backend::vulkan_device_profile(p.vk);
        const size_t tile_from_8bit = backend::tile_from_for(profile, true, nin);
        const size_t tile_from_other = backend::tile_from_for(profile, false, nin);
        for (size_t nbatch : {size_t(1), size_t(3), size_t(8), size_t(13), size_t(16), size_t(64),
                              size_t(100), size_t(247)}) {
            const auto x = uniform(nbatch * nin, 12 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            // Which kernel a batch takes is the backend's decision, and it
            // differs by device, so ask rather than assume: below the
            // threshold the row kernel reads quantized activations and the
            // reference must be fed the same, at or above it the tile kernel
            // reads floats.
            // A device whose integer dot is native takes wide quantized batches through the integer-dot tile, which reads 8-bit activations.
            const bool idot = profile.prefer_integer_dot;
            const auto xr8 = nbatch < tile_from_8bit ? row_activations(x) : idot ? tile_activations8(x) : x;   // adopted, so they must outlive the calls
            const auto xr4 = nbatch < tile_from_other ? row_activations(x) : idot ? tile_activations8(x) : x;   // Q4_0, Q4_1 and Q6_K read the 16-bit twin
            const auto xrk = nbatch < tile_from_other ? twin_activations(x, idot) : idot ? tile_activations8(x) : x;
            Pair::In xri8 = p.in(xr8), xri4 = p.in(xr4), xrik = p.in(xrk);
            for (int q = 0; q < 7; ++q) {
                if (q >= 4 && nin % 256) continue;   // K-quant blocks are 256 wide
                const uint32_t type = q == 1 ? gguf::GGML_TYPE_Q8_0 : q == 2 ? gguf::GGML_TYPE_Q4_0
                                    : q == 3 ? gguf::GGML_TYPE_Q4_1 : q == 4 ? gguf::GGML_TYPE_Q6_K
                                    : q == 5 ? gguf::GGML_TYPE_Q4_K : q == 6 ? gguf::GGML_TYPE_Q5_K
                                    : gguf::GGML_TYPE_F32;
                const Pair::In& wi = q == 1 ? wqi : q == 2 ? w4i : q == 3 ? w41i : q == 4 ? w6i
                                   : q == 5 ? w4ki : q == 6 ? w5ki : wfi;
                Pair::Out d = p.out(nbatch * nout);
                p.cpu.matmul(type, wi.cs(), (q == 0 ? xi : q == 1 ? xri8 : q <= 4 ? xri4 : xrik).cs(), d.cs(), nin, nout, nbatch);
                p.vk.matmul(type, wi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
                auto r = p.results(d);
                try {
                    const bool row16 = (q == 1 && nbatch < tile_from_8bit) || (q >= 2 && q <= 4 && nbatch < tile_from_other);
                    const double tol = q == 0 || row16 ? 1e-4 : twin_tol;
                    values += close(r.first, r.second, tol, q == 1 ? "Q8_0 matmul differs beyond 1e-4"
                                                            : q == 2 ? "Q4_0 matmul differs beyond 1e-4"
                                                            : q == 3 ? "Q4_1 matmul differs beyond 1e-4"
                                                            : q == 4 ? "Q6_K matmul differs beyond 1e-4"
                                                            : q == 5 ? "Q4_K matmul differs beyond 1e-4"
                                                            : q == 6 ? "Q5_K matmul differs beyond 1e-4"
                                                                     : "F32 matmul differs beyond 1e-4");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  matmul type %u nin %zu nbatch %zu\n", type, nin, nbatch);
                    throw;
                }
            }
        }
        // norm_rope_kv: the fused attention inputs against the CPU's three
        // separate ops; q compared directly, k and v through attention over
        // the view each backend wrote, after a 70-token history.
        {
            const int n_head = 4, n_head_kv = 2, head_dim = 40;
            const size_t half = head_dim / 2, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
            const size_t rows = 3, hist = 70, table = 128;
            std::vector<float> cs(table * half), sn(table * half);
            for (size_t t = 0; t < table; ++t)
                for (size_t i = 0; i < half; ++i) {
                    const double f = std::pow(10000.0, -2.0 * double(i) / double(head_dim));
                    cs[t * half + i] = float(std::cos(double(t) * f));
                    sn[t * half + i] = float(std::sin(double(t) * f));
                }
            const auto q0 = uniform(rows * qw, 31), k0 = uniform(rows * kvw, 32), v0 = uniform(rows * kvw, 33);
            const auto qn = uniform(head_dim, 34, 0.5f, 1.5f), kn = uniform(head_dim, 35, 0.5f, 1.5f);
            const auto hk = uniform(hist * kvw, 36), hv = uniform(hist * kvw, 37);
            const uint32_t pos[3] = {70, 71, 72};
            auto run = [&](backend::Backend& b, std::vector<float>& qout, std::vector<float>& att) {
                const size_t bt = b.kv_layout().block_tokens;
                auto st = b.kv_alloc(1, n_head_kv, head_dim, 512);
                infer::BlockPool pool(st->max_blocks());
                infer::KVSequence seq(&pool, bt);
                const auto Kh = b.adopt(hk.data(), hk.size() * sizeof(float));
                const auto Vh = b.adopt(hv.data(), hv.size() * sizeof(float));
                seq.prepare(hist);
                {
                    const backend::KVView h = seq.view(st.get());
                    b.kv_write(0, &h, 1, {Kh.get(), 0}, {Vh.get(), 0});
                }
                seq.commit();
                seq.prepare(rows);
                const backend::KVView view = seq.view(st.get());
                const auto Qb = b.alloc(q0.size() * sizeof(float), backend::Memory::device);
                const auto Kb = b.alloc(k0.size() * sizeof(float), backend::Memory::device);
                const auto Vb = b.alloc(v0.size() * sizeof(float), backend::Memory::device);
                b.write(*Qb, 0, q0.data(), q0.size() * sizeof(float));
                b.write(*Kb, 0, k0.data(), k0.size() * sizeof(float));
                b.write(*Vb, 0, v0.data(), v0.size() * sizeof(float));
                const auto qnb = b.adopt(qn.data(), qn.size() * sizeof(float));
                const auto knb = b.adopt(kn.data(), kn.size() * sizeof(float));
                const auto cb = b.adopt(cs.data(), cs.size() * sizeof(float));
                const auto sb = b.adopt(sn.data(), sn.size() * sizeof(float));
                const backend::Backend::RopeArgs rope{{cb.get(), 0}, {sb.get(), 0}, half, pos, 1e-6f};
                b.norm_rope_kv({Qb.get(), 0}, qw, n_head, {qnb.get(), 0}, {Kb.get(), 0}, {Vb.get(), 0},
                               kvw, n_head_kv, {knb.get(), 0}, rope, rows, 0, &view, 1);
                const auto ob = b.alloc(rows * qw * sizeof(float), backend::Memory::device);
                b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                qout.resize(rows * qw);
                att.resize(rows * qw);
                b.read(*Qb, 0, qout.data(), qout.size() * sizeof(float));
                b.read(*ob, 0, att.data(), att.size() * sizeof(float));
            };
            std::vector<float> qc, ac, qv, av;
            run(p.cpu, qc, ac);
            run(p.vk, qv, av);
            values += close(qc, qv, 1e-5, "norm_rope_kv q differs beyond 1e-5");
            values += close(ac, av, 1e-4, "norm_rope_kv attention differs beyond 1e-4");
        }
        // attention over a wide pass of 128-wide heads takes the tiled
        // kernel: 32 and 45 query rows (one full tile, then a partial one
        // whose last rows mask part of a K/V tile) after histories of 0 and
        // 70 tokens, against the CPU at 1e-4.
        for (size_t hist : {size_t(0), size_t(70)}) {
            for (size_t nq : {size_t(32), size_t(45)}) {
                const int n_head = 4, n_head_kv = 2, head_dim = 128;
                const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
                const auto hk = uniform((hist + nq) * kvw, 50 + (uint32_t)nq), hv = uniform((hist + nq) * kvw, 51 + (uint32_t)nq);
                const auto qq = uniform(nq * qw, 52 + (uint32_t)hist);
                auto run = [&](backend::Backend& b, std::vector<float>& att) {
                    const size_t bt = b.kv_layout().block_tokens;
                    auto st = b.kv_alloc(1, n_head_kv, head_dim, 512);
                    infer::BlockPool pool(st->max_blocks());
                    infer::KVSequence seq(&pool, bt);
                    const auto Kb = b.adopt(hk.data(), hk.size() * sizeof(float));
                    const auto Vb = b.adopt(hv.data(), hv.size() * sizeof(float));
                    const auto Qb = b.adopt(qq.data(), qq.size() * sizeof(float));
                    if (hist) {
                        seq.prepare(hist);
                        const backend::KVView h = seq.view(st.get());
                        b.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                        seq.commit();
                    }
                    seq.prepare(nq);
                    const backend::KVView view = seq.view(st.get());
                    b.kv_write(0, &view, 1, {Kb.get(), hist * kvw}, {Vb.get(), hist * kvw});
                    const auto ob = b.alloc(nq * qw * sizeof(float), backend::Memory::device);
                    b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                    att.resize(nq * qw);
                    b.read(*ob, 0, att.data(), att.size() * sizeof(float));
                };
                std::vector<float> ac, av;
                run(p.cpu, ac);
                run(p.vk, av);
                values += close(ac, av, 1e-4, "tiled attention differs beyond 1e-4");
            }
        }
        // The twin the per-row attention kernel, or its merge after a
        // split history, writes beside its output for the row matmul that
        // follows: attention then a matmul from its output on the device,
        // against the CPU's attention, quantized, into the CPU's matmul.
        for (size_t hist : {size_t(0), size_t(70)}) {
            for (size_t nq : {size_t(1), size_t(3)}) {
                const int n_head = 4, n_head_kv = 2, head_dim = 128;
                const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim, nout = 37;
                const auto hk = uniform((hist + nq) * kvw, 60 + (uint32_t)nq), hv = uniform((hist + nq) * kvw, 61 + (uint32_t)nq);
                const auto qq = uniform(nq * qw, 62 + (uint32_t)hist);
                const auto wf = uniform(qw * nout, 63);
                std::vector<uint8_t> wq(nout * (qw / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
                for (size_t row = 0; row < nout; ++row)
                    quant::quantize_row_q8_0(wf.data() + row * qw, wq.data() + row * (qw / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE,
                                             qw / gguf::Q8_0_BLOCK);
                auto run = [&](backend::Backend& b, bool device, std::vector<float>& y) {
                    const size_t bt = b.kv_layout().block_tokens;
                    auto st = b.kv_alloc(1, n_head_kv, head_dim, 512);
                    infer::BlockPool pool(st->max_blocks());
                    infer::KVSequence seq(&pool, bt);
                    const auto Kb = b.adopt(hk.data(), hk.size() * sizeof(float));
                    const auto Vb = b.adopt(hv.data(), hv.size() * sizeof(float));
                    const auto Qb = b.adopt(qq.data(), qq.size() * sizeof(float));
                    const auto Wb = b.adopt(wq.data(), wq.size());
                    if (hist) {
                        seq.prepare(hist);
                        const backend::KVView h = seq.view(st.get());
                        b.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                        seq.commit();
                    }
                    seq.prepare(nq);
                    const backend::KVView view = seq.view(st.get());
                    b.kv_write(0, &view, 1, {Kb.get(), hist * kvw}, {Vb.get(), hist * kvw});
                    const auto ob = b.alloc(nq * qw * sizeof(float), backend::Memory::device);
                    const auto yb = b.alloc(nq * nout * sizeof(float), backend::Memory::device);
                    b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                    if (device) {
                        b.matmul(gguf::GGML_TYPE_Q8_0, {Wb.get(), 0}, {ob.get(), 0}, {yb.get(), 0}, qw, nout, nq);
                    } else {
                        std::vector<float> att(nq * qw);
                        b.read(*ob, 0, att.data(), att.size() * sizeof(float));
                        const auto ar = row_activations(att);
                        const auto Ab = b.adopt(ar.data(), ar.size() * sizeof(float));
                        b.matmul(gguf::GGML_TYPE_Q8_0, {Wb.get(), 0}, {Ab.get(), 0}, {yb.get(), 0}, qw, nout, nq);
                    }
                    y.resize(nq * nout);
                    b.read(*yb, 0, y.data(), y.size() * sizeof(float));
                };
                std::vector<float> yc, yv;
                run(p.cpu, false, yc);
                run(p.vk, true, yv);
                values += close(yc, yv, 1e-4, "matmul from the attention twin differs beyond 1e-4");
            }
        }
        // f16 cache sides: each combination of K and V types on both
        // backends, through kv_write, the fused norm_rope_kv, kv_copy and
        // attention on the per-row and the tiled kernel. The device is
        // compared with the CPU at the same types at 1e-4, and every f16
        // combination with the CPU's f32 result at a looser 2e-2, which is
        // what rounding keys and values to half precision costs here.
        for (int combo = 1; combo < 4; ++combo) {
            const backend::KVType kt = combo & 1 ? backend::KVType::f16 : backend::KVType::f32;
            const backend::KVType vt = combo & 2 ? backend::KVType::f16 : backend::KVType::f32;
            for (size_t nq : {size_t(2), size_t(40)}) {
                const int n_head = 4, n_head_kv = 2, head_dim = 128;
                const size_t half = head_dim / 2, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
                const size_t hist = 70, table = 256;
                std::vector<float> cs(table * half), sn(table * half);
                for (size_t t = 0; t < table; ++t)
                    for (size_t i = 0; i < half; ++i) {
                        const double f = std::pow(10000.0, -2.0 * double(i) / double(head_dim));
                        cs[t * half + i] = float(std::cos(double(t) * f));
                        sn[t * half + i] = float(std::sin(double(t) * f));
                    }
                const auto hk = uniform(hist * kvw, 60 + combo), hv = uniform(hist * kvw, 61 + combo);
                const auto q0 = uniform(nq * qw, 62 + combo), k0 = uniform(nq * kvw, 63 + combo), v0 = uniform(nq * kvw, 64 + combo);
                const auto qn = uniform(head_dim, 65, 0.5f, 1.5f), kn = uniform(head_dim, 66, 0.5f, 1.5f);
                std::vector<uint32_t> pos(nq);
                for (size_t i = 0; i < nq; ++i) pos[i] = (uint32_t)(hist + i);
                auto run = [&](backend::Backend& b, backend::KVType kk, backend::KVType vv, std::vector<float>& att) {
                    const size_t bt = b.kv_layout().block_tokens;
                    auto st = b.kv_alloc(1, n_head_kv, head_dim, 512, kk, vv);
                    infer::BlockPool pool(st->max_blocks());
                    infer::KVSequence seq(&pool, bt);
                    const auto Kh = b.adopt(hk.data(), hk.size() * sizeof(float));
                    const auto Vh = b.adopt(hv.data(), hv.size() * sizeof(float));
                    seq.prepare(hist);
                    {
                        const backend::KVView h = seq.view(st.get());
                        b.kv_write(0, &h, 1, {Kh.get(), 0}, {Vh.get(), 0});
                    }
                    seq.commit();
                    // The history's last block copied over itself through
                    // kv_copy, which must move the stored bytes whatever the type.
                    {
                        const backend::KVView h = seq.view(st.get());
                        const int32_t last = h.blocks[(hist - 1) / bt];
                        b.kv_copy(*st, last, last);
                    }
                    seq.prepare(nq);
                    const backend::KVView view = seq.view(st.get());
                    const auto Qb = b.alloc(q0.size() * sizeof(float), backend::Memory::device);
                    const auto Kb = b.alloc(k0.size() * sizeof(float), backend::Memory::device);
                    const auto Vb = b.alloc(v0.size() * sizeof(float), backend::Memory::device);
                    b.write(*Qb, 0, q0.data(), q0.size() * sizeof(float));
                    b.write(*Kb, 0, k0.data(), k0.size() * sizeof(float));
                    b.write(*Vb, 0, v0.data(), v0.size() * sizeof(float));
                    const auto qnb = b.adopt(qn.data(), qn.size() * sizeof(float));
                    const auto knb = b.adopt(kn.data(), kn.size() * sizeof(float));
                    const auto cb = b.adopt(cs.data(), cs.size() * sizeof(float));
                    const auto sb = b.adopt(sn.data(), sn.size() * sizeof(float));
                    const backend::Backend::RopeArgs rope{{cb.get(), 0}, {sb.get(), 0}, half, pos.data(), 1e-6f};
                    b.norm_rope_kv({Qb.get(), 0}, qw, n_head, {qnb.get(), 0}, {Kb.get(), 0}, {Vb.get(), 0},
                                   kvw, n_head_kv, {knb.get(), 0}, rope, nq, 0, &view, 1);
                    const auto ob = b.alloc(nq * qw * sizeof(float), backend::Memory::device);
                    b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                    att.resize(nq * qw);
                    b.read(*ob, 0, att.data(), att.size() * sizeof(float));
                };
                std::vector<float> ac, av, a32;
                run(p.cpu, kt, vt, ac);
                run(p.vk, kt, vt, av);
                run(p.cpu, backend::KVType::f32, backend::KVType::f32, a32);
                values += close(ac, av, 1e-4, "f16 cache attention differs between backends beyond 1e-4");
                close(a32, av, 2e-2, "f16 cache attention differs from the f32 cache beyond 2e-2");
            }
        }
        // matmul_add: the product joins what Y already holds, on the row
        // kernel and on the tile kernel, against the CPU's scratch-and-add.
        for (size_t nbatch : {size_t(1), size_t(3), size_t(64)}) {
            const auto xa = uniform(nbatch * nin, 20 + (uint32_t)nbatch);
            Pair::In xi = p.in(xa);
            const backend::DeviceProfile prof = backend::vulkan_device_profile(p.vk);
            const auto xr = nbatch < backend::tile_from_for(prof, true, nin) ? row_activations(xa)
                            : prof.prefer_integer_dot                         ? tile_activations8(xa)
                                                                              : xa;
            Pair::In xri = p.in(xr);
            const auto y0 = uniform(nbatch * nout, 21 + (uint32_t)nbatch);
            Pair::Out d = p.out(nbatch * nout);
            p.cpu.write(*d.c, 0, y0.data(), y0.size() * sizeof(float));
            p.vk.write(*d.v, 0, y0.data(), y0.size() * sizeof(float));
            p.cpu.matmul_add(gguf::GGML_TYPE_Q8_0, wqi.cs(), xri.cs(), d.cs(), nin, nout, nbatch);
            p.vk.matmul_add(gguf::GGML_TYPE_Q8_0, wqi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
            auto r = p.results(d);
            values += close(r.first, r.second, nbatch < backend::tile_from_for(prof, true, nin) ? 1e-4 : twin_tol,
                            "matmul_add differs beyond its bound");
        }
        // The twin the norm and SiLU kernels write beside their output for
        // the row kernel: each into a buffer, then a matmul from it, against
        // the CPU's producer followed by the quantized reference.
        {
            const size_t rows = 3;
            const auto src = uniform(rows * nin, 40 + (uint32_t)nin), wn = uniform(nin, 41, 0.5f, 1.5f);
            const auto g = uniform(rows * nin, 42, -6.0f, 6.0f), u = uniform(rows * nin, 43);
            Pair::In si = p.in(src), wni = p.in(wn), gi = p.in(g), ui = p.in(u);
            Pair::Out h = p.out(rows * nin), f = p.out(rows * nin);
            Pair::Out d1 = p.out(rows * nout), d2 = p.out(rows * nout);
            p.vk.rms_norm_rows(h.vs(), si.vs(), wni.vs(), rows, nin, nin, 1e-6f);
            p.vk.matmul(gguf::GGML_TYPE_Q8_0, wqi.vs(), h.vs(), d1.vs(), nin, nout, rows);
            p.vk.silu_mul(f.vs(), gi.vs(), ui.vs(), rows * nin);
            p.vk.matmul(gguf::GGML_TYPE_Q4_0, w4i.vs(), f.vs(), d2.vs(), nin, nout, rows);
            p.cpu.rms_norm_rows(h.cs(), si.cs(), wni.cs(), rows, nin, nin, 1e-6f);
            p.cpu.silu_mul(f.cs(), gi.cs(), ui.cs(), rows * nin);
            std::vector<float> hc(rows * nin), fc(rows * nin);
            p.cpu.read(*h.c, 0, hc.data(), hc.size() * sizeof(float));
            p.cpu.read(*f.c, 0, fc.data(), fc.size() * sizeof(float));
            const auto hr = row_activations(hc), fr = row_activations(fc);   // Q8_0 from the norm, Q4_0 from the SiLU, both on the 16-bit twin
            Pair::In hri = p.in(hr), fri = p.in(fr);
            p.cpu.matmul(gguf::GGML_TYPE_Q8_0, wqi.cs(), hri.cs(), d1.cs(), nin, nout, rows);
            p.cpu.matmul(gguf::GGML_TYPE_Q4_0, w4i.cs(), fri.cs(), d2.cs(), nin, nout, rows);
            auto r1 = p.results(d1), r2 = p.results(d2);
            values += close(r1.first, r1.second, 1e-4, "matmul from the norm's twin differs beyond 1e-4");
            values += close(r2.first, r2.second, 1e-4, "matmul from the SiLU's twin differs beyond 1e-4");
        }
        bool rejected = false;
        try { p.vk.matmul(1u /* F16, no kernel */, wqi.vs(), wqi.vs(), p.out(8).vs(), nin, 1, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported matrix type accepted");
        // Three projections in one dispatch equal the same three one at a
        // time, bit for bit, since each row's work is unchanged; rows are
        // uneven so the workgroup ranges do not line up.
        for (size_t nbatch : {size_t(1), size_t(3)}) {
            const auto x = uniform(nbatch * nin, 30 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            const size_t rows[3] = {nout, 5, 33};
            Pair::Out sep[3] = {p.out(nbatch * rows[0]), p.out(nbatch * rows[1]), p.out(nbatch * rows[2])};
            Pair::Out grp[3] = {p.out(nbatch * rows[0]), p.out(nbatch * rows[1]), p.out(nbatch * rows[2])};
            for (int i = 0; i < 3; ++i)
                p.vk.matmul(gguf::GGML_TYPE_Q8_0, wqi.vs(), xi.vs(), sep[i].vs(), nin, rows[i], nbatch);
            p.vk.matmul_group({{gguf::GGML_TYPE_Q8_0, wqi.vs(), grp[0].vs(), rows[0]},
                               {gguf::GGML_TYPE_Q8_0, wqi.vs(), grp[1].vs(), rows[1]},
                               {gguf::GGML_TYPE_Q8_0, wqi.vs(), grp[2].vs(), rows[2]}},
                              xi.vs(), nin, nbatch);
            for (int i = 0; i < 3; ++i) {
                std::vector<float> a(sep[i].n), b(grp[i].n);
                p.vk.read(*sep[i].v, 0, a.data(), a.size() * sizeof(float));
                p.vk.read(*grp[i].v, 0, b.data(), b.size() * sizeof(float));
                values += exact(a, b, "grouped projections differ from separate ones");
            }
        }
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
                    const double d = std::isfinite(b[i]) ? std::fabs((double)a[i] - b[i]) / (1.0 + std::fabs((double)a[i])) : 1e9;
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
    // bound by bandwidth or by per-kernel latency. Every quantized type
    // the kernel decodes, at the shapes the fixtures use it for; the
    // 151936-row Q6_K is the tied head of the Q4_0 fixture.
    struct Timed { uint32_t type; const char* name; size_t nin, nout; };
    for (const Timed& t : {Timed{gguf::GGML_TYPE_Q8_0, "Q8_0", 1024, 1024}, {gguf::GGML_TYPE_Q8_0, "Q8_0", 1024, 2048},
                           {gguf::GGML_TYPE_Q8_0, "Q8_0", 1024, 3072}, {gguf::GGML_TYPE_Q8_0, "Q8_0", 3072, 1024},
                           {gguf::GGML_TYPE_Q8_0, "Q8_0", 4096, 4096}, {gguf::GGML_TYPE_Q8_0, "Q8_0", 4096, 12288},
                           {gguf::GGML_TYPE_Q8_0, "Q8_0", 12288, 4096},
                           {gguf::GGML_TYPE_Q4_0, "Q4_0", 1024, 3072}, {gguf::GGML_TYPE_Q4_0, "Q4_0", 4096, 12288},
                           {gguf::GGML_TYPE_Q4_1, "Q4_1", 3072, 1024}, {gguf::GGML_TYPE_Q4_1, "Q4_1", 12288, 4096},
                           {gguf::GGML_TYPE_Q6_K, "Q6_K", 1024, 3072}, {gguf::GGML_TYPE_Q6_K, "Q6_K", 4096, 12288},
                           {gguf::GGML_TYPE_Q6_K, "Q6_K", 1024, 151936},
                           {gguf::GGML_TYPE_Q4_K, "Q4_K", 1024, 3072}, {gguf::GGML_TYPE_Q4_K, "Q4_K", 4096, 12288},
                           {gguf::GGML_TYPE_Q5_K, "Q5_K", 1024, 3072}, {gguf::GGML_TYPE_Q5_K, "Q5_K", 4096, 12288},
                           // An 8B feed-forward down projection, the shape per-operation benchmarks of other runtimes report.
                           {gguf::GGML_TYPE_Q8_0, "Q8_0", 14336, 4096}, {gguf::GGML_TYPE_Q4_K, "Q4_K", 14336, 4096},
                           {gguf::GGML_TYPE_Q6_K, "Q6_K", 14336, 4096}}) {
        const size_t nin = t.nin, nout = t.nout;
        const size_t block = t.type >= gguf::GGML_TYPE_Q4_K ? gguf::Q6_K_BLOCK : 32;
        const size_t bytes = t.type == gguf::GGML_TYPE_Q8_0 ? gguf::Q8_0_TYPESIZE
                           : t.type == gguf::GGML_TYPE_Q4_0 ? gguf::Q4_0_TYPESIZE
                           : t.type == gguf::GGML_TYPE_Q4_1 ? gguf::Q4_1_TYPESIZE
                           : t.type == gguf::GGML_TYPE_Q4_K ? gguf::Q4_K_TYPESIZE
                           : t.type == gguf::GGML_TYPE_Q5_K ? gguf::Q5_K_TYPESIZE : gguf::Q6_K_TYPESIZE;
        std::vector<uint8_t> wq(nout * (nin / block) * bytes);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 7 + 3);
        const auto x = uniform(nin, 15);
        const auto w = vk.adopt(wq.data(), wq.size());
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(nout * sizeof(float), backend::Memory::device);
        vk.matmul(t.type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, 1);
        vk.sync();
        const int iters = 100;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul(t.type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, 1);
        vk.sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: " << t.name << " matvec " << nin << "x" << nout << " " << us << " us, "
                  << (double)wq.size() / us / 1e3 << " GB/s\n";
    }
    // The prefill tile at one feed-forward projection of an 8B model over a 512-row pass, reported and not asserted, in operations per second so it reads against a per-operation benchmark of any other runtime at the same shape.
    for (const Timed& t : {Timed{gguf::GGML_TYPE_Q8_0, "Q8_0", 14336, 4096}, {gguf::GGML_TYPE_Q4_K, "Q4_K", 14336, 4096},
                           {gguf::GGML_TYPE_Q6_K, "Q6_K", 14336, 4096}}) {
        const size_t nin = t.nin, nout = t.nout, nbatch = 512;
        const size_t block = t.type == gguf::GGML_TYPE_Q8_0 ? 32 : gguf::Q6_K_BLOCK;
        const size_t bytes = t.type == gguf::GGML_TYPE_Q8_0 ? gguf::Q8_0_TYPESIZE
                           : t.type == gguf::GGML_TYPE_Q4_K ? gguf::Q4_K_TYPESIZE : gguf::Q6_K_TYPESIZE;
        std::vector<uint8_t> wq(nout * (nin / block) * bytes);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 7 + 3);
        const auto x = uniform(nin * nbatch, 23);
        const auto w = vk.adopt(wq.data(), wq.size());
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(nout * nbatch * sizeof(float), backend::Memory::device);
        vk.matmul(t.type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, nbatch);
        vk.sync();
        const int iters = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul(t.type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, nbatch);
        vk.sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: " << t.name << " prefill " << nout << "x" << nin << " over " << nbatch << " rows "
                  << us << " us, " << 2.0 * nin * nout * nbatch / us / 1e6 << " TFLOPS\n";
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

// `--isa DIR` opens the backend for diagnostics and writes the driver's representation of every kernel it compiled, one file per kernel, after the checks.
int main(int argc, char** argv) {
    const std::string isa_dir = argc == 3 && std::strcmp(argv[1], "--isa") == 0 ? argv[2] : "";
    backend::BackendPtr b;
    try {
        b = backend::make_vulkan_backend(0, !isa_dir.empty());
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
        std::cout << backend::vulkan_kernel_statistics(*b);
        if (!isa_dir.empty()) {
            size_t written = 0;
            for (const auto& kr : backend::vulkan_kernel_representations(*b)) {
                std::ofstream f(isa_dir + "/" + kr.first + ".txt");
                f << kr.second;
                written += f.good() ? 1 : 0;
            }
            std::cout << "backend-vulkan: " << written << " kernel representations written to " << isa_dir << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
