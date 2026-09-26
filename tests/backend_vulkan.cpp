// Vulkan backend test (docs/VULKAN.md): storage and submission over a real device, then every kernel against the CPU backend on random inputs.
// Bit exact where the arithmetic is the same operation in the same order, a stated tolerance where a transcendental or a reduction order differs.
// Exits 77, which CTest reports as skipped, when there is no loader or no device.
#include <chrono>
#include <fstream>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
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

// The same op on both backends over the same inputs.
// Inputs are adopted (the CPU keeps the pointer, the device uploads a copy); outputs are allocated on each and the device's is read back.
struct Pair {
    backend::CpuBackend cpu;
    backend::Backend& vk;
    // The reference keeps float activations in decode, so a device's rounding is compared against exact arithmetic.
    explicit Pair(backend::Backend& v) : vk(v) { cpu.set_threads(1); cpu.set_decode_activations8(false); }
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

// The activations as the device's row kernel sees them: each block of 32 scaled so its largest magnitude is 32767, rounded half away from zero, and back to floats (shaders/quantize_x.comp).
// The CPU reference of a quantized-row matmul on the row kernel takes these, so the comparison is about the dot and its reduction order and not about the quantization, which is the device's choice and the HF gate's business.
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

// The activations as a row family reads them: on a device whose integer dot is native every quantized family reads the 8-bit twin, except an output head's Q4_0, Q4_1 or Q6_K rows (matmul_logits); elsewhere every family reads the 16-bit one.
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
    // rms_norm_rows and rms_norm: the sum of squares is reduced in a different order, so a tolerance; dst aliasing src is exercised.
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
    // embed: F32 rows are copies and Q8_0 rows are a half scale times a small integer, exact in float, so both are exact.
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
        // Raw Q4_0 rows, every second block under a negative scale, which the quantizer never writes: nibble 8 then decodes as -0 on both backends.
        std::vector<uint8_t> raw4(q4.size());
        for (size_t b = 0; b < raw4.size() / gguf::Q4_0_TYPESIZE; ++b) {
            uint8_t* blk = raw4.data() + b * gguf::Q4_0_TYPESIZE;
            blk[0] = 0x00;
            blk[1] = b % 2 ? 0xB8 : 0x38;   // -0.5 or 0.5
            for (size_t j = 0; j < 16; ++j) blk[2 + j] = uint8_t(((j + b) & 15) | (((5 * j + 3 + b) & 15) << 4));
        }
        Pair::In traw4 = p.in(raw4.data(), raw4.size());
        Pair::Out draw4 = p.out(nin * 3);
        p.cpu.embed(draw4.cs(), gguf::GGML_TYPE_Q4_0, traw4.cs(), nin, nrows, ids, 3);
        p.vk.embed(draw4.vs(), gguf::GGML_TYPE_Q4_0, traw4.vs(), nin, nrows, ids, 3);
        auto rraw4 = p.results(draw4);
        size_t negative_zeros = 0;
        for (float v : rraw4.first) if (v == 0.0f && std::signbit(v)) ++negative_zeros;
        require(negative_zeros == 10, "raw Q4_0 rows do not decode ten -0 values on the CPU");
        values += exact(rraw4.first, rraw4.second, "embed Q4_0 under a negative scale differs");

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
        // Q4_K and Q5_K rows of 256 the same way; d and dmin are the first two halves of a block.
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
        // A table that holds fewer rows than the call names is refused before any row is read, even when every id is inside the table.
        const uint32_t first[1] = {0};
        auto short_table_refused = [&](uint32_t type, const Pair::In& table_in) {
            try { p.vk.embed(d.vs(), type, table_in.vs(), nin, nrows + 1, first, 1); }
            catch (const std::runtime_error& e) { return std::string(e.what()).find("outside its allocation") != std::string::npos; }
            return false;
        };
        require(short_table_refused(gguf::GGML_TYPE_F32, t), "F32 embedding table shorter than its rows accepted");
        require(short_table_refused(gguf::GGML_TYPE_Q8_0, tq), "Q8_0 embedding table shorter than its rows accepted");
    }
    // The float tile reads an F32 matrix 256 or more floats wide through a padded copy made on first use; after a write into the weights the next call must read what was written.
    {
        const size_t nin = 256, nout = 67, cols = 64;
        const auto w1 = uniform(nin * nout, 21), w2 = uniform(nin * nout, 22), xx = uniform(nin * cols, 23);
        auto wb = p.vk.adopt(w1.data(), w1.size() * sizeof(float));
        auto fresh = p.vk.adopt(w2.data(), w2.size() * sizeof(float));
        auto xb = p.vk.adopt(xx.data(), xx.size() * sizeof(float));
        auto y1 = p.vk.alloc(nout * cols * sizeof(float), backend::Memory::device);
        auto y2 = p.vk.alloc(nout * cols * sizeof(float), backend::Memory::device);
        p.vk.matmul(gguf::GGML_TYPE_F32, {wb.get(), 0}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.write(*wb, 0, w2.data(), w2.size() * sizeof(float));
        p.vk.matmul(gguf::GGML_TYPE_F32, {wb.get(), 0}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.matmul(gguf::GGML_TYPE_F32, {fresh.get(), 0}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
        std::vector<float> a(nout * cols), b(nout * cols);
        p.vk.read(*y1, 0, a.data(), a.size() * sizeof(float));
        p.vk.read(*y2, 0, b.data(), b.size() * sizeof(float));
        values += exact(a, b, "an F32 matmul after a write into its weights read the old weights");
    }
    // matmul: F32 and Q8_0 over odd sizes and batch widths that fall inside, on and past the eight-column chunk.
    // The reduction order differs from the CPU's, so a tolerance.
    for (size_t nin : {size_t(1024), size_t(256), size_t(224)}) {
        // 1024 is sixteen block pairs, the word-wide path; 256 has too few pairs for it and 224 an odd block count, both the 16-bit path.
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
        // No Q6_K quantizer is needed: any bytes are a valid block and both backends decode the same bytes.
        // The half scale is 2^-10 so the values sit in the range of the other types; larger scales made reduction-order rounding alone exceed the tolerance.
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
        // The row kernel, whose quantized rows meet 16-bit activations, takes batches below a threshold that depends on the device and on how wide a row is; the tile kernel takes the rest with float activations.
        // The thresholds come from the backend rather than from constants here, because a device that was measured to want other numbers gets them and the reference has to follow.
        const backend::DeviceProfile profile = backend::vulkan_device_profile(p.vk);
        const size_t tile_from_8bit = backend::tile_from_for(profile, true, nin);
        const size_t tile_from_other = backend::tile_from_for(profile, false, nin);
        for (size_t nbatch : {size_t(1), size_t(3), size_t(8), size_t(13), size_t(16), size_t(64),
                              size_t(100), size_t(247)}) {
            const auto x = uniform(nbatch * nin, 12 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            // Which kernel a batch takes is the backend's decision, and it differs by device, so ask rather than assume: below the threshold the row kernel reads quantized activations and the reference must be fed the same, at or above it the tile kernel reads floats.
            // A device whose integer dot is native takes wide quantized batches through the integer-dot tile, which reads 8-bit activations.
            const bool idot = profile.prefer_integer_dot;
            const auto xr8 = nbatch < tile_from_8bit ? twin_activations(x, idot) : idot ? tile_activations8(x) : x;   // adopted, so they must outlive the calls; Q8_0 rows read the 8-bit twin where the integer dot is native
            const auto xr4 = nbatch < tile_from_other ? twin_activations(x, idot) : idot ? tile_activations8(x) : x;
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
                p.cpu.matmul(type, wi.cs(), (q == 0 ? xi : q == 1 ? xri8 : q <= 3 ? xri4 : xrik).cs(), d.cs(), nin, nout, nbatch);
                p.vk.matmul(type, wi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
                auto r = p.results(d);
                try {
                    const bool row16 = (q == 1 && nbatch < tile_from_8bit && !idot) || (q >= 2 && q <= 4 && nbatch < tile_from_other && !idot);
                    const double tol = q == 0 || row16 ? 1e-4 : twin_tol;
                    values += close(r.first, r.second, tol, q == 1 ? "Q8_0 matmul differs beyond 1e-4"
                                                            : q == 2 ? "Q4_0 matmul differs beyond 1e-4"
                                                            : q == 3 ? "Q4_1 matmul differs beyond 1e-4"
                                                            : q == 4 ? "Q6_K matmul differs beyond 1e-4"
                                                            : q == 5 ? "Q4_K matmul differs beyond 1e-4"
                                                            : q == 6 ? "Q5_K matmul differs beyond 1e-4"
                                                                     : "F32 matmul differs beyond 1e-4");
                    // The output head keeps the 16-bit twin for Q6_K rows.
                    if (q == 4 && nbatch < tile_from_other) {
                        Pair::Out h = p.out(nbatch * nout);
                        const auto x16 = row_activations(x);
                        Pair::In x16i = p.in(x16);
                        p.cpu.matmul(type, wi.cs(), x16i.cs(), h.cs(), nin, nout, nbatch);
                        p.vk.matmul_logits(type, wi.vs(), xi.vs(), h.vs(), nin, nout, nbatch);
                        auto rh = p.results(h);
                        values += close(rh.first, rh.second, 1e-4, "Q6_K output head differs beyond 1e-4");
                    }
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  matmul type %u nin %zu nbatch %zu\n", type, nin, nbatch);
                    throw;
                }
            }
        }
        // norm_rope_kv: the fused attention inputs against the CPU's three separate ops; q compared directly, k and v through attention over the view each backend wrote, after a 70-token history.
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
        // attention over a wide pass of 128-wide heads takes the tiled kernel: 32, 45 and 100 query rows (one full tile, then partial ones whose last rows mask part of a K/V tile) after histories of 0, 70 and 600 tokens, against the CPU at 1e-4.
        // The queries are scaled so a row's scores spread over about ten, peaked as a trained model's are rather than the near-uniform softmax of unit random values, and the cache is taken both as f32 and as f16.
        for (backend::KVType kt : {backend::KVType::f32, backend::KVType::f16})
        for (size_t hist : {size_t(0), size_t(70), size_t(600)}) {
            for (size_t nq : {size_t(32), size_t(45), size_t(100)}) {
                const int n_head = 4, n_head_kv = 2, head_dim = 128;
                const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
                const auto hk = uniform((hist + nq) * kvw, 50 + (uint32_t)nq), hv = uniform((hist + nq) * kvw, 51 + (uint32_t)nq);
                const auto qq = uniform(nq * qw, 52 + (uint32_t)hist, -6.0f, 6.0f);
                auto run = [&](backend::Backend& b, std::vector<float>& att) {
                    const size_t bt = b.kv_layout().block_tokens;
                    auto st = b.kv_alloc(1, n_head_kv, head_dim, 1024, kt, kt);
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
                try {
                    values += close(ac, av, 1e-4, "tiled attention differs beyond 1e-4");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  tiled attention hist %zu rows %zu cache %s\n", hist, nq, backend::kv_type_name(kt));
                    throw;
                }
            }
        }
        // Nonzero attention across vector widths and a non-vector width, including mixed cache sides and grouped heads after a long history.
        // Both rows must compute identically alone and together, even when the longer history selects grouped heads for the short row.
        if (nin == 1024)
        for (int head_dim : {32, 40, 64, 128, 256})
        for (backend::KVType kt : {backend::KVType::f32, backend::KVType::f16})
        for (backend::KVType vt : {backend::KVType::f32, backend::KVType::f16})
        for (int group : {1, 2, 4, 8}) {
            const int n_head_kv = 2, n_head = n_head_kv * group;
            const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
            const size_t longs[2] = {2100, 3000}, shorts = 40;
            const auto hk = uniform(3001 * kvw, 80 + (uint32_t)group), hv = uniform(3001 * kvw, 81 + (uint32_t)group);
            const auto qq = uniform(2 * qw, 82 + (uint32_t)group, -6.0f, 6.0f);
            // Rows of `hists` histories (each followed by one new token) in one call, or each alone; returns every row's output.
            auto run = [&](backend::Backend& b, const std::vector<size_t>& hists, bool together) {
                const size_t bt = b.kv_layout().block_tokens;
                auto st = b.kv_alloc(1, n_head_kv, head_dim, 8192, kt, vt);
                infer::BlockPool pool(st->max_blocks());
                const auto Kb = b.adopt(hk.data(), hk.size() * sizeof(float));
                const auto Vb = b.adopt(hv.data(), hv.size() * sizeof(float));
                const auto Qb = b.adopt(qq.data(), qq.size() * sizeof(float));
                std::vector<infer::KVSequence> seqs;
                for (size_t i = 0; i < hists.size(); ++i) seqs.emplace_back(&pool, bt);
                std::vector<backend::KVView> views;
                for (size_t i = 0; i < hists.size(); ++i) {
                    seqs[i].prepare(hists[i] + 1);
                    const backend::KVView h = seqs[i].view(st.get());
                    b.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                    seqs[i].commit();
                }
                std::vector<float> out(hists.size() * qw);
                const auto ob = b.alloc(out.size() * sizeof(float), backend::Memory::device);
                auto view_of = [&](size_t i) {
                    // The row attends as the last of its history: a view of one query row over hist + 1 tokens.
                    backend::KVView v = seqs[i].view(st.get());
                    v.length -= 1;
                    v.nq = 1;
                    return v;
                };
                if (together) {
                    for (size_t i = 0; i < hists.size(); ++i) views.push_back(view_of(i));
                    b.attention({Qb.get(), 0}, 0, views.data(), views.size(), {ob.get(), 0}, n_head, n_head_kv, head_dim);
                } else {
                    for (size_t i = 0; i < hists.size(); ++i) {
                        const backend::KVView v = view_of(i);
                        b.attention({Qb.get(), i * qw}, 0, &v, 1, {ob.get(), i * qw}, n_head, n_head_kv, head_dim);
                    }
                }
                b.read(*ob, 0, out.data(), out.size() * sizeof(float));
                return out;
            };
            try {
                for (size_t hist : longs) {
                    const auto ac = run(p.cpu, {hist}, false), av = run(p.vk, {hist}, false);
                    values += close(ac, av, 1e-4, "decode attention after a long history differs beyond 1e-4");
                }
                const auto reference = run(p.cpu, {shorts, longs[1]}, false);
                const auto mixed = run(p.vk, {shorts, longs[1]}, true), alone = run(p.vk, {shorts, longs[1]}, false);
                values += close(reference, alone, 1e-4, "mixed-history attention differs beyond 1e-4");
                values += exact(mixed, alone, "attention rows differ alone and beside another history");
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  attention width %d group %d cache K %s V %s\n", head_dim, group,
                             backend::kv_type_name(kt), backend::kv_type_name(vt));
                throw;
            }
        }
        // Batch invariance: a row computes the same, bit for bit, whatever shares its call, given its prompt's RowRun.
        // A prompt's tail alone against those rows of one pass, a generated row alone against it beside others, and a mixed call against both, at shapes that take the tallest tile whole and the shortest for the tail.
        if (nin == 1024)
        for (size_t n_out : {size_t(300), size_t(2048), size_t(6144)}) {
            const size_t n_in = 1024, rows = 249, tail = 9, prompt = 249;
            const auto wi = uniform(n_in * n_out, 70);
            std::vector<uint8_t> wi8(n_out * (n_in / 32) * gguf::Q8_0_TYPESIZE), wi40(n_out * (n_in / 32) * gguf::Q4_0_TYPESIZE);
            for (size_t r = 0; r < n_out; ++r) {
                quant::quantize_row_q8_0(wi.data() + r * n_in, wi8.data() + r * (n_in / 32) * gguf::Q8_0_TYPESIZE, n_in / 32);
                quant::quantize_row_q4_0(wi.data() + r * n_in, wi40.data() + r * (n_in / 32) * gguf::Q4_0_TYPESIZE, n_in / 32);
            }
            std::vector<uint8_t> wi6(n_out * (n_in / 256) * gguf::Q6_K_TYPESIZE), wi4k(n_out * (n_in / 256) * gguf::Q4_K_TYPESIZE);
            for (size_t i = 0; i < wi6.size(); ++i) wi6[i] = uint8_t(i * 131 + 7);
            for (size_t i = 0; i < wi4k.size(); ++i) wi4k[i] = uint8_t(i * 61 + 3);
            for (size_t r = 0; r < n_out * (n_in / 256); ++r) {
                wi6[r * gguf::Q6_K_TYPESIZE + 208] = 0x00;
                wi6[r * gguf::Q6_K_TYPESIZE + 209] = 0x14;
                uint8_t* blk = wi4k.data() + r * gguf::Q4_K_TYPESIZE;
                blk[0] = 0x00; blk[1] = 0x14; blk[2] = 0x00; blk[3] = 0x10;
            }
            const auto x = uniform(rows * n_in, 71);
            const auto y0 = uniform(rows * n_out, 72);
            const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
            struct W { uint32_t type; const void* data; size_t bytes; };
            for (const W& t : {W{gguf::GGML_TYPE_F32, wi.data(), wi.size() * sizeof(float)},
                               W{gguf::GGML_TYPE_Q8_0, wi8.data(), wi8.size()}, W{gguf::GGML_TYPE_Q4_0, wi40.data(), wi40.size()},
                               W{gguf::GGML_TYPE_Q6_K, wi6.data(), wi6.size()}, W{gguf::GGML_TYPE_Q4_K, wi4k.data(), wi4k.size()}}) {
                const auto wb = vk.adopt(t.data, t.bytes);
                for (bool add : {false, true}) {
                    auto run = [&](size_t first, size_t n, const std::vector<backend::RowRun>& runs) {
                        const auto yb = vk.alloc(n * n_out * sizeof(float), backend::Memory::device);
                        vk.write(*yb, 0, y0.data() + first * n_out, n * n_out * sizeof(float));
                        const backend::RowRuns rr{runs.data(), runs.size()};
                        if (add) vk.matmul_add(t.type, {wb.get(), 0}, {xb.get(), first * n_in}, {yb.get(), 0}, n_in, n_out, n, rr);
                        else vk.matmul(t.type, {wb.get(), 0}, {xb.get(), first * n_in}, {yb.get(), 0}, n_in, n_out, n, rr);
                        std::vector<float> y(n * n_out);
                        vk.read(*yb, 0, y.data(), y.size() * sizeof(float));
                        return y;
                    };
                    auto part = [&](const std::vector<float>& v, size_t from, size_t to) {
                        return std::vector<float>(v.begin() + from * n_out, v.begin() + to * n_out);
                    };
                    try {
                        const auto whole = run(0, rows, {{rows, prompt}});
                        values += exact(part(whole, rows - tail, rows), run(rows - tail, tail, {{tail, prompt}}),
                                        "a prompt's last rows differ from the same rows of one pass");
                        std::vector<backend::RowRun> each;
                        for (size_t r = 0; r < 13; ++r) each.push_back({r + 1, 1});
                        const auto gen = run(0, 13, each);
                        values += exact(part(gen, 5, 6), run(5, 1, {{1, 1}}), "a generated row alone differs from it beside others");
                        const auto mixed = run(0, rows, {{3, 1}, {rows, prompt}});
                        values += exact(part(mixed, 0, 3), part(gen, 0, 3), "generated rows beside a prompt differ from them alone");
                        values += exact(part(mixed, 3, rows), part(whole, 3, rows), "a prompt beside generated rows differs from it alone");
                    } catch (const std::runtime_error&) {
                        std::fprintf(stderr, "  batch invariance: matmul type %u, %zu outputs%s\n", t.type, n_out, add ? ", accumulating" : "");
                        throw;
                    }
                }
            }
        }
        // The same for attention: a prompt's tail after its reused history against those rows of one pass over the whole prompt, and a decode row alone against it beside a sequence with a longer history, which splits the dispatch more ways.
        if (nin == 1024)
        for (backend::KVType kt : {backend::KVType::f32, backend::KVType::f16}) {
            const int n_head = 4, n_head_kv = 2, head_dim = 128;
            const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
            const size_t len = 100, pre = 91, ha = 500, hb = 1500;
            const auto K = uniform(2048 * kvw, 80), V = uniform(2048 * kvw, 81), Q = uniform(2048 * qw, 82, -6.0f, 6.0f);
            const auto Kb = vk.adopt(K.data(), K.size() * sizeof(float));
            const auto Vb = vk.adopt(V.data(), V.size() * sizeof(float));
            const auto Qb = vk.adopt(Q.data(), Q.size() * sizeof(float));
            const size_t bt = vk.kv_layout().block_tokens;
            auto st = vk.kv_alloc(1, n_head_kv, head_dim, 4096, kt, kt);
            infer::BlockPool pool(st->max_blocks());
            auto read_rows = [&](const backend::KVView* views, size_t n_views, size_t q_first, size_t nrows) {
                const auto ob = vk.alloc(nrows * qw * sizeof(float), backend::Memory::device);
                vk.attention({Qb.get(), q_first * qw}, 0, views, n_views, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                std::vector<float> o(nrows * qw);
                vk.read(*ob, 0, o.data(), o.size() * sizeof(float));
                return o;
            };
            auto prompt_rows = [&](size_t hist, size_t nq, size_t extent) {
                infer::KVSequence seq(&pool, bt);
                if (hist) {
                    seq.prepare(hist);
                    const backend::KVView h = seq.view(st.get());
                    vk.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                    seq.commit();
                }
                seq.prepare(nq);
                backend::KVView v = seq.view(st.get());
                v.extent = extent;
                vk.kv_write(0, &v, 1, {Kb.get(), hist * kvw}, {Vb.get(), hist * kvw});
                auto o = read_rows(&v, 1, hist, nq);
                seq.abort();
                return o;
            };
            try {
                const auto whole = prompt_rows(0, len, len);
                values += exact(std::vector<float>(whole.end() - (len - pre) * qw, whole.end()), prompt_rows(pre, len - pre, len),
                                "attention over a prompt's tail differs from those rows of one pass");
                infer::KVSequence a(&pool, bt), b(&pool, bt);
                for (auto* s : {&a, &b}) {
                    const size_t hist = s == &a ? ha : hb;
                    s->prepare(hist);
                    const backend::KVView h = s->view(st.get());
                    vk.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                    s->commit();
                    s->prepare(1);
                }
                backend::KVView both[2] = {a.view(st.get()), b.view(st.get())};
                both[0].extent = both[1].extent = 1;
                vk.kv_write(0, &both[0], 1, {Kb.get(), ha * kvw}, {Vb.get(), ha * kvw});
                vk.kv_write(0, &both[1], 1, {Kb.get(), hb * kvw}, {Vb.get(), hb * kvw});
                const auto pair = read_rows(both, 2, 0, 2);
                values += exact(std::vector<float>(pair.begin(), pair.begin() + qw), read_rows(both, 1, 0, 1),
                                "a decode row's attention beside a longer history differs from it alone");
                a.abort();
                b.abort();
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  batch invariance: attention, cache %s\n", backend::kv_type_name(kt));
                throw;
            }
        }
        // The integer-dot tile's 8-bit copy that a norm, a SiLU and a wide attention write when told the tile reads their output next is the one the tile's own pass makes, bit for bit: each producer with runs into a buffer and a matmul with those runs from it, against the producer without them.
        // A write from the host or a kernel between the producer and the matmul drops the copy, so the matmul reads what the buffer holds then.
        if (nin == 1024) {
            const size_t n_in = 1024, n_out = 96, rows = 128;
            const std::vector<backend::RowRun> one{{rows, rows}};
            const backend::RowRuns rr{one.data(), one.size()};
            const auto wf = uniform(n_out * n_in, 90);
            std::vector<uint8_t> w8(n_out * (n_in / 32) * gguf::Q8_0_TYPESIZE);
            for (size_t r = 0; r < n_out; ++r)
                quant::quantize_row_q8_0(wf.data() + r * n_in, w8.data() + r * (n_in / 32) * gguf::Q8_0_TYPESIZE, n_in / 32);
            const auto wb = vk.adopt(w8.data(), w8.size());
            const auto src = uniform(rows * n_in, 91), wn = uniform(n_in, 92, 0.5f, 1.5f);
            const auto g = uniform(rows * n_in, 93, -6.0f, 6.0f), u = uniform(rows * n_in, 94), other = uniform(rows * n_in, 95);
            const size_t bytes = rows * n_in * sizeof(float);
            const auto srcb = vk.adopt(src.data(), bytes), gb = vk.adopt(g.data(), bytes), ub = vk.adopt(u.data(), bytes);
            const auto wnb = vk.adopt(wn.data(), n_in * sizeof(float)), otherb = vk.adopt(other.data(), bytes);
            auto floats = [&](const backend::BufferPtr& b) {
                std::vector<float> v(rows * n_in);
                vk.read(*b, 0, v.data(), bytes);
                return v;
            };
            // One output for every matmul, allocated before any producer runs, since a new buffer drops the producer's copy.
            const auto yb = vk.alloc(rows * n_out * sizeof(float), backend::Memory::device);
            auto matmul_of = [&](const backend::BufferPtr& x) {
                vk.matmul(gguf::GGML_TYPE_Q8_0, {wb.get(), 0}, {x.get(), 0}, {yb.get(), 0}, n_in, n_out, rows, rr);
                std::vector<float> y(rows * n_out);
                vk.read(*yb, 0, y.data(), y.size() * sizeof(float));
                return y;
            };
            auto norm = [&](bool told) {
                auto h = vk.alloc(bytes, backend::Memory::device);
                vk.rms_norm_rows({h.get(), 0}, {srcb.get(), 0}, {wnb.get(), 0}, rows, n_in, n_in, 1e-6f, told ? rr : backend::RowRuns{});
                return h;
            };
            auto silu = [&](bool told) {
                auto f = vk.alloc(bytes, backend::Memory::device);
                vk.silu_mul({f.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, rows * n_in, told ? rr : backend::RowRuns{});
                return f;
            };
            // Eight heads of 128 over two KV heads, n_in wide, one prompt of `rows` rows, which takes the tiled kernel with or without its extent.
            const int n_head = 8, n_head_kv = 2, head_dim = 128;
            const size_t kvw = (size_t)n_head_kv * head_dim;
            const auto K = uniform(rows * kvw, 96), V = uniform(rows * kvw, 97), Q = uniform(rows * n_in, 98, -6.0f, 6.0f);
            const auto Kb = vk.adopt(K.data(), K.size() * sizeof(float)), Vb = vk.adopt(V.data(), V.size() * sizeof(float));
            const auto Qb = vk.adopt(Q.data(), bytes);
            auto st = vk.kv_alloc(1, n_head_kv, head_dim, 512);
            infer::BlockPool pool(st->max_blocks());
            auto attend = [&](bool told) {
                infer::KVSequence seq(&pool, vk.kv_layout().block_tokens);
                seq.prepare(rows);
                backend::KVView view = seq.view(st.get());
                view.extent = told ? rows : 0;
                vk.kv_write(0, &view, 1, {Kb.get(), 0}, {Vb.get(), 0});
                auto o = vk.alloc(bytes, backend::Memory::device);
                vk.attention({Qb.get(), 0}, 0, &view, 1, {o.get(), 0}, n_head, n_head_kv, head_dim);
                vk.sync();
                seq.abort();
                return o;
            };
            try {
                const auto h1 = norm(true), h0 = norm(false);
                values += exact(floats(h1), floats(h0), "the norm's output differs with and without its runs");
                values += exact(matmul_of(norm(true)), matmul_of(norm(false)), "a matmul from the norm's 8-bit copy differs from its own pass");
                const auto f1 = silu(true), f0 = silu(false);
                values += exact(floats(f1), floats(f0), "the SiLU's output differs with and without its runs");
                values += exact(matmul_of(silu(true)), matmul_of(silu(false)), "a matmul from the SiLU's 8-bit copy differs from its own pass");
                // A routed down projection reads the SiLU's output as k entries a token row, each of its token's prompt, so the SiLU given those runs writes the copy the routed tile reads (model/arch_qwen.hpp).
                {
                    const size_t k = 2, n_expert = 4, entries = rows * k, ebytes = entries * n_in * sizeof(float);
                    const std::vector<backend::RowRun> eruns{{entries, rows}};
                    const auto ef = uniform(n_expert * n_out * n_in, 99);
                    std::vector<uint8_t> e8(n_expert * n_out * (n_in / 32) * gguf::Q8_0_TYPESIZE);
                    for (size_t r = 0; r < n_expert * n_out; ++r)
                        quant::quantize_row_q8_0(ef.data() + r * n_in, e8.data() + r * (n_in / 32) * gguf::Q8_0_TYPESIZE, n_in / 32);
                    const auto eb = vk.adopt(e8.data(), e8.size());
                    const auto scores = uniform(rows * n_expert, 100, -3.0f, 3.0f);
                    const auto scoresb = vk.adopt(scores.data(), scores.size() * sizeof(float));
                    const auto idsb = vk.alloc(entries * sizeof(float), backend::Memory::device);
                    const auto wtsb = vk.alloc(entries * sizeof(float), backend::Memory::device);
                    vk.route_experts({scoresb.get(), 0}, rows, n_expert, k, true, {idsb.get(), 0}, {wtsb.get(), 0});
                    const backend::Backend::Routing routing{{idsb.get(), 0}, {wtsb.get(), 0}, k, n_expert};
                    const auto ge = uniform(entries * n_in, 101, -6.0f, 6.0f), ue = uniform(entries * n_in, 102);
                    const auto geb = vk.adopt(ge.data(), ebytes), ueb = vk.adopt(ue.data(), ebytes);
                    auto routed = [&](bool told) {
                        const auto f = vk.alloc(ebytes, backend::Memory::device);
                        const auto yb = vk.alloc(rows * n_out * sizeof(float), backend::Memory::device);
                        vk.silu_mul({f.get(), 0}, {geb.get(), 0}, {ueb.get(), 0}, entries * n_in,
                                    told ? backend::RowRuns{eruns.data(), eruns.size()} : backend::RowRuns{});
                        vk.matmul_experts_add(gguf::GGML_TYPE_Q8_0, {eb.get(), 0}, {f.get(), 0}, {yb.get(), 0}, n_in, n_out, rows, routing, rr);
                        std::vector<float> y(rows * n_out);
                        vk.read(*yb, 0, y.data(), y.size() * sizeof(float));
                        return y;
                    };
                    values += exact(routed(true), routed(false), "a routed down projection from the SiLU's 8-bit copy differs from its own pass");
                }
                const auto o1 = attend(true), o0 = attend(false);
                values += exact(floats(o1), floats(o0), "the attention's output differs with and without its runs");
                const auto y1 = matmul_of(attend(true)), y0 = matmul_of(attend(false));
                values += exact(y1, y0, "a matmul from the attention's 8-bit copy differs from its own pass");
                const auto reference = matmul_of(otherb);
                const auto hw = norm(true);
                vk.write(*hw, 0, other.data(), bytes);
                values += exact(matmul_of(hw), reference, "a matmul read a norm's 8-bit copy after the host wrote its output");
                std::vector<float> sum(rows * n_in);
                const auto plain = floats(norm(false));
                for (size_t i = 0; i < sum.size(); ++i) sum[i] = plain[i] + other[i];
                const auto sumb = vk.adopt(sum.data(), bytes);
                const auto ha = norm(true);
                vk.add({ha.get(), 0}, {otherb.get(), 0}, rows * n_in);
                values += exact(matmul_of(ha), matmul_of(sumb), "a matmul read a norm's 8-bit copy after a kernel wrote its output");
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  producers' 8-bit copy for the tile\n");
                throw;
            }
        }
        // The twin the per-row attention kernel, or its merge after a split history, writes beside its output for the row matmul that follows: attention then a matmul from its output on the device, against the CPU's attention, quantized, into the CPU's matmul.
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
                        const auto ar = twin_activations(att, twin8);
                        const auto Ab = b.adopt(ar.data(), ar.size() * sizeof(float));
                        b.matmul(gguf::GGML_TYPE_Q8_0, {Wb.get(), 0}, {Ab.get(), 0}, {yb.get(), 0}, qw, nout, nq);
                    }
                    y.resize(nq * nout);
                    b.read(*yb, 0, y.data(), y.size() * sizeof(float));
                };
                std::vector<float> yc, yv;
                run(p.cpu, false, yc);
                run(p.vk, true, yv);
                values += close(yc, yv, twin_tol, "matmul from the attention twin differs beyond its bound");
            }
        }
        // f16 cache sides: each combination of K and V types on both backends, through kv_write, the fused norm_rope_kv and attention on the per-row and the tiled kernel.
        // The device is compared with the CPU at the same types at 1e-4, and every f16 combination with the CPU's f32 result at a looser 2e-2, which is what rounding keys and values to half precision costs here.
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
        // matmul_add: the product joins what Y already holds, on the row kernel and on the tile kernel, against the CPU's scratch-and-add.
        for (size_t nbatch : {size_t(1), size_t(3), size_t(64)}) {
            const auto xa = uniform(nbatch * nin, 20 + (uint32_t)nbatch);
            Pair::In xi = p.in(xa);
            const backend::DeviceProfile prof = backend::vulkan_device_profile(p.vk);
            const auto xr = nbatch < backend::tile_from_for(prof, true, nin) ? twin_activations(xa, twin8)
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
            values += close(r.first, r.second, twin_tol, "matmul_add differs beyond its bound");
        }
        // The twin the norm and SiLU kernels write beside their output for the row kernel: each into a buffer, then a matmul from it, against the CPU's producer followed by the quantized reference.
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
            const auto hr = twin_activations(hc, twin8), fr = twin_activations(fc, twin8);   // Q8_0 from the norm and Q4_0 from the SiLU, each on the twin its kernel reads
            Pair::In hri = p.in(hr), fri = p.in(fr);
            p.cpu.matmul(gguf::GGML_TYPE_Q8_0, wqi.cs(), hri.cs(), d1.cs(), nin, nout, rows);
            p.cpu.matmul(gguf::GGML_TYPE_Q4_0, w4i.cs(), fri.cs(), d2.cs(), nin, nout, rows);
            auto r1 = p.results(d1), r2 = p.results(d2);
            values += close(r1.first, r1.second, twin_tol, "matmul from the norm's twin differs beyond its bound");
            values += close(r2.first, r2.second, twin_tol, "matmul from the SiLU's twin differs beyond its bound");
        }
        bool rejected = false;
        try { p.vk.matmul(1u /* F16, no kernel */, wqi.vs(), wqi.vs(), p.out(8).vs(), nin, 1, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported matrix type accepted");
        // Row runs out of order are refused even when every run takes the same kernel, so nothing would have split them.
        {
            const backend::RowRun merged[2] = {{3, 1}, {2, 1}};
            const auto x = uniform(2 * nin, 13);
            Pair::In xi = p.in(x);
            Pair::Out d = p.out(2 * nout);
            rejected = false;
            try { p.vk.matmul(gguf::GGML_TYPE_Q8_0, wqi.vs(), xi.vs(), d.vs(), nin, nout, 2, {merged, 2}); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "row runs out of order accepted");
        }
        // Three projections in one dispatch equal the same three one at a time, bit for bit, since each row's work is unchanged; rows are uneven so the workgroup ranges do not line up.
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
        // At tile widths the integer-dot tile takes the projections of one type in one dispatch, split or not, and a type of its own in another: two Q8_0 projections and a Q4_0 one between them, against the CPU fed the activations the tile reads.
        for (size_t nbatch : {size_t(64), size_t(249)}) {
            const auto x = uniform(nbatch * nin, 40 + (uint32_t)nbatch);
            const backend::DeviceProfile prof = backend::vulkan_device_profile(p.vk);
            const bool tiled = nbatch >= backend::tile_from_for(prof, false, nin);
            const auto xr = !tiled ? x : prof.prefer_integer_dot ? tile_activations8(x) : x;
            if (!tiled) continue;
            Pair::In xi = p.in(x), xri = p.in(xr);
            const size_t rows[3] = {nout, 5, 33};
            const uint32_t types[3] = {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q8_0};
            Pair::Out grp[3] = {p.out(nbatch * rows[0]), p.out(nbatch * rows[1]), p.out(nbatch * rows[2])};
            p.vk.matmul_group({{types[0], wqi.vs(), grp[0].vs(), rows[0]},
                               {types[1], w4i.vs(), grp[1].vs(), rows[1]},
                               {types[2], wqi.vs(), grp[2].vs(), rows[2]}},
                              xi.vs(), nin, nbatch);
            for (int i = 0; i < 3; ++i) {
                p.cpu.matmul(types[i], (i == 1 ? w4i : wqi).cs(), xri.cs(), grp[i].cs(), nin, rows[i], nbatch);
                auto r = p.results(grp[i]);
                try {
                    values += close(r.first, r.second, twin_tol, "grouped tile projections differ beyond their bound");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  grouped tile: projection %d of type %u, %zu rows, %zu columns\n", i, types[i], rows[i], nbatch);
                    throw;
                }
            }
        }
        // A mixed group whose batch reaches the 8-bit crossover but not the other types' takes the row kernel for every partition, its Q8_0 one included, though that partition alone would take the tile (matmul_group_impl).
        // So the group's Q8_0 and Q4_0 outputs equal, bit for bit, each type alone with a row run of extent 1, which forces the row kernel; the batches are the first and the last between the two crossovers, and a device whose crossovers meet at this width has none.
        for (size_t nbatch : {tile_from_8bit, tile_from_other - 1}) {
            if (nbatch < tile_from_8bit || nbatch >= tile_from_other) continue;
            const auto x = uniform(nbatch * nin, 50 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            const uint32_t types[2] = {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0};
            Pair::Out grp[2] = {p.out(nbatch * nout), p.out(nbatch * nout)};
            Pair::Out alone[2] = {p.out(nbatch * nout), p.out(nbatch * nout)};
            p.vk.matmul_group({{types[0], wqi.vs(), grp[0].vs(), nout}, {types[1], w4i.vs(), grp[1].vs(), nout}},
                              xi.vs(), nin, nbatch);
            const backend::RowRun decode{nbatch, 1};
            for (int i = 0; i < 2; ++i)
                p.vk.matmul(types[i], (i == 0 ? wqi : w4i).vs(), xi.vs(), alone[i].vs(), nin, nout, nbatch, {&decode, 1});
            for (int i = 0; i < 2; ++i) {
                std::vector<float> a(alone[i].n), b(grp[i].n);
                p.vk.read(*alone[i].v, 0, a.data(), a.size() * sizeof(float));
                p.vk.read(*grp[i].v, 0, b.data(), b.size() * sizeof(float));
                try {
                    values += exact(a, b, "a mixed group's partition left the row kernel the group chose");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  mixed group on the row kernel: type %u, nin %zu, %zu columns\n", types[i], nin, nbatch);
                    throw;
                }
            }
        }
    }
    // The KV cache: the same token-major rows written through each backend's own storage and block size, then attention over each backend's own view.
    // Histories straddle the device's 64-token blocks and the CPU's 128; two views in one call.
    // The online softmax orders the arithmetic differently from the CPU's global softmax, so a tolerance.
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
                        // A second sequence with a two-token history shares the call: its rows come after the first view's.
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
    // The cost of a dispatch that does almost nothing, reported and not asserted: a decoded token on Qwen3-0.6B is about four hundred of them.
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
    // The row kernel at the projection shapes of Qwen3-0.6B and 8B, one column, reported: the small shapes say whether a decoded token is bound by bandwidth or by per-kernel latency.
    // Every quantized type the kernel decodes, at the shapes the fixtures use it for; the 151936-row Q6_K is the tied head of the Q4_0 fixture.
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
    // Decode attention and the small kernels at the Qwen3-0.6B shape over a 250-token history, one query, reported.
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
    // Decode bandwidth of the row kernel on a Qwen3-8B-sized projection, reported and not asserted: 4096 x 4096 Q8_0 is 17 MiB per column.
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
    // Mixture of experts: routing the same scores gives the same ids and weights, and the routed projections match the CPU over stacked experts of every type, gate and up in one call and the weighted down projection into a residual.
    // Three shapes: a few rows on the row kernel, a prompt past moe_tile_from on the tile kernel over each expert's entries, and a batch mixing decode rows with a prompt, split by its row runs.
    // The reference is fed, row by row, the activations the kernel that row takes reads.
    {
        const size_t n_expert = 6, k = 2, nin = 256, nff = 67, nout = 45;
        const backend::DeviceProfile prof = backend::vulkan_device_profile(p.vk);
        size_t from = prof.moe_tile_from;   // the weight type's, set per type below
        // Stacked expert bytes of a type: F32 and the block quantizers from floats, the K-quants from a byte pattern with small half scales, as the matmul check above builds them.
        auto stacked = [&](uint32_t type, size_t in, size_t out, uint32_t seed) {
            const size_t n_rows = n_expert * out;
            const auto f = uniform(n_rows * in, seed);
            std::vector<uint8_t> bytes;
            if (type == gguf::GGML_TYPE_F32) {
                bytes.resize(f.size() * sizeof(float));
                std::memcpy(bytes.data(), f.data(), bytes.size());
            } else if (type == gguf::GGML_TYPE_Q8_0 || type == gguf::GGML_TYPE_Q4_0 || type == gguf::GGML_TYPE_Q4_1) {
                const size_t ts = type == gguf::GGML_TYPE_Q8_0 ? gguf::Q8_0_TYPESIZE : type == gguf::GGML_TYPE_Q4_0 ? gguf::Q4_0_TYPESIZE : gguf::Q4_1_TYPESIZE;
                bytes.resize(n_rows * (in / 32) * ts);
                for (size_t r = 0; r < n_rows; ++r) {
                    uint8_t* dst = bytes.data() + r * (in / 32) * ts;
                    if (type == gguf::GGML_TYPE_Q8_0) quant::quantize_row_q8_0(f.data() + r * in, dst, in / 32);
                    else if (type == gguf::GGML_TYPE_Q4_0) quant::quantize_row_q4_0(f.data() + r * in, dst, in / 32);
                    else quant::quantize_row_q4_1(f.data() + r * in, dst, in / 32);
                }
            } else {
                const size_t ts = type == gguf::GGML_TYPE_Q6_K ? gguf::Q6_K_TYPESIZE : type == gguf::GGML_TYPE_Q4_K ? gguf::Q4_K_TYPESIZE : gguf::Q5_K_TYPESIZE;
                bytes.resize(n_rows * (in / 256) * ts);
                for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * (61 + seed % 7) + 3);
                for (size_t b = 0; b < n_rows * (in / 256); ++b) {
                    uint8_t* blk = bytes.data() + b * ts;
                    if (type == gguf::GGML_TYPE_Q6_K) { blk[208] = 0x00; blk[209] = 0x14; }
                    else { blk[0] = 0x00; blk[1] = 0x14; blk[2] = 0x00; blk[3] = 0x10; }
                }
            }
            return bytes;
        };
        struct Shape { size_t rows; std::vector<backend::RowRun> runs; };
        const Shape shapes[] = {{5, {}}, {70, {{70, 512}}}, {70, {{3, 1}, {70, 512}}}};
        for (const Shape& sh : shapes) {
            const size_t rows = sh.rows, entries = rows * k;
            const backend::RowRuns runs{sh.runs.data(), sh.runs.size()};
            // Whether token row r takes the tile kernel.
            auto tiled = [&](size_t r) {
                if (sh.runs.empty()) return rows >= from;
                size_t start = 0;
                for (const backend::RowRun& run : sh.runs) {
                    if (r < run.end) return r >= start && run.extent >= from;
                    start = run.end;
                }
                return false;
            };
            const auto scores = uniform(rows * n_expert, 90 + (uint32_t)rows + (uint32_t)sh.runs.size(), -3.0f, 3.0f);
            Pair::In si = p.in(scores);
            Pair::Out ids = p.out(entries), wts = p.out(entries);
            p.cpu.route_experts(si.cs(), rows, n_expert, k, true, ids.cs(), wts.cs());
            p.vk.route_experts(si.vs(), rows, n_expert, k, true, ids.vs(), wts.vs());
            auto ri = p.results(ids);
            values += exact(ri.first, ri.second, "routed expert ids differ");
            auto rw = p.results(wts);
            values += close(rw.first, rw.second, 1e-6, "routing weights differ beyond 1e-6");
            const backend::Backend::Routing rc{ids.cs(), wts.cs(), k, n_expert}, rv{ids.vs(), wts.vs(), k, n_expert};
            const auto x = uniform(rows * nin, 91), x2 = uniform(entries * nin, 92), y0 = uniform(rows * nout, 93);
            Pair::In xi = p.in(x), x2i = p.in(x2);
            for (uint32_t type : {gguf::GGML_TYPE_F32, gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1,
                                  gguf::GGML_TYPE_Q4_K, gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K}) {
                from = backend::moe_tile_from_for(prof, type);
                const bool f32 = type == gguf::GGML_TYPE_F32;
                const bool reads8 = twin8 && !f32;
                // The row kernel reads a twin, 8-bit or 16-bit by family; the tile reads 8-bit activations where the integer dot takes quantized types, else floats.
                const bool tile8 = twin8 && !f32;
                auto fed = [&](const std::vector<float>& v, size_t per_row, size_t per_entry_rows) {
                    std::vector<float> out(v.size());
                    for (size_t i = 0; i < v.size() / per_row; ++i) {
                        const std::vector<float> row(v.begin() + i * per_row, v.begin() + (i + 1) * per_row);
                        const bool t = tiled(i / per_entry_rows);
                        const std::vector<float> got = f32 ? row : t ? (tile8 ? tile_activations8(row) : row) : twin_activations(row, reads8);
                        std::copy(got.begin(), got.end(), out.begin() + i * per_row);
                    }
                    return out;
                };
                // The 8-bit bound wherever any row's kernel reads 8-bit activations.
                bool eight = false;
                for (size_t r = 0; r < rows; ++r) eight = eight || (tiled(r) ? tile8 : reads8);
                const double tol = eight ? twin_tol : 1e-4;
                const auto wg = stacked(type, nin, nff, 94), wu = stacked(type, nin, nff, 95), wd = stacked(type, nin, nout, 96);
                Pair::In wgi = p.in(wg.data(), wg.size()), wui = p.in(wu.data(), wu.size()), wdi = p.in(wd.data(), wd.size());
                const auto xr = fed(x, nin, 1), x2r = fed(x2, nin, k);
                Pair::In xri = p.in(xr), x2ri = p.in(x2r);
                try {
                    Pair::Out g = p.out(entries * nff), u = p.out(entries * nff);
                    p.cpu.matmul_experts({{type, wgi.cs(), g.cs(), nff}, {type, wui.cs(), u.cs(), nff}}, xri.cs(), nin, rows, rc, runs);
                    p.vk.matmul_experts({{type, wgi.vs(), g.vs(), nff}, {type, wui.vs(), u.vs(), nff}}, xi.vs(), nin, rows, rv, runs);
                    auto rg = p.results(g), ru = p.results(u);
                    values += close(rg.first, rg.second, tol, "routed gate projection differs beyond its bound");
                    // Generated tokens beside each other go through the grouped row kernel, one run of an expert's entries a workgroup row; each token alone takes an entry per workgroup row.
                    // The entries must come out bit for bit the same either way.
                    if (!sh.runs.empty() && rows > 1) {
                        std::vector<backend::RowRun> decode(rows);
                        for (size_t r = 0; r < rows; ++r) decode[r] = backend::RowRun{r + 1, 1};
                        Pair::Out gb = p.out(entries * nff);
                        p.vk.matmul_experts({{type, wgi.vs(), gb.vs(), nff}}, xi.vs(), nin, rows, rv, {decode.data(), rows});
                        const std::vector<float> batched = p.results(gb).second;
                        const backend::RowRun one[1] = {{1, 1}};
                        for (size_t r = 0; r < rows; ++r) {
                            Pair::Out ga = p.out(k * nff);
                            const backend::Backend::Routing alone{{ids.vs().buffer, ids.vs().offset + r * k}, {wts.vs().buffer, wts.vs().offset + r * k}, k, n_expert};
                            p.vk.matmul_experts({{type, wgi.vs(), ga.vs(), nff}}, {xi.vs().buffer, xi.vs().offset + r * nin}, nin, 1, alone, {one, 1});
                            const std::vector<float> single = p.results(ga).second;
                            if (std::memcmp(single.data(), batched.data() + r * k * nff, k * nff * sizeof(float)) != 0)
                                throw std::runtime_error("a generated token's routed entries differ beside other tokens");
                            ++values;
                        }
                    }
                    values += close(ru.first, ru.second, tol, "routed up projection differs beyond its bound");
                    Pair::Out y = p.out(rows * nout);
                    p.cpu.write(*y.c, 0, y0.data(), y0.size() * sizeof(float));
                    p.vk.write(*y.v, 0, y0.data(), y0.size() * sizeof(float));
                    p.cpu.matmul_experts_add(type, wdi.cs(), x2ri.cs(), y.cs(), nin, nout, rows, rc, runs);
                    p.vk.matmul_experts_add(type, wdi.vs(), x2i.vs(), y.vs(), nin, nout, rows, rv, runs);
                    auto ry = p.results(y);
                    values += close(ry.first, ry.second, tol, "routed down projection differs beyond its bound");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  routed experts type %u, %zu rows in %zu runs\n", type, rows, sh.runs.size());
                    throw;
                }
            }
        }
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

// Each refusal below records and writes nothing, so the stream works after it as before.
// Each is made twice, in a first pass whose operands no command-buffer slot holds and in a second into an output that must keep what it held.
// alloc and adopt leave a new buffer held by the slot that fills or copies it, so the first pass submits until the slots let go of the operands before it makes the call.
// It drops them when the call throws, so a command the call left naming one names freed memory and fails the next submission.
// After each pass, a valid call must give what it gave before any refusal.
size_t check_refusals(backend::Backend& vk) {
    const backend::DeviceProfile prof = backend::vulkan_device_profile(vk);
    const bool twin8 = prof.prefer_integer_dot;
    const uint32_t q8 = gguf::GGML_TYPE_Q8_0, f32 = gguf::GGML_TYPE_F32, f16 = 1;   // F16 has no kernel
    const size_t nin = 64, nout = 8, rows = 3, n_expert = 4, k = 2, entries = rows * k, nrows = 4, partial = 48;
    const size_t row_bytes = nin / gguf::Q8_0_BLOCK * gguf::Q8_0_TYPESIZE;
    auto quantized = [&](size_t n, uint32_t seed) {
        const auto f = uniform(n * nin, seed);
        std::vector<uint8_t> q(n * row_bytes);
        for (size_t r = 0; r < n; ++r) quant::quantize_row_q8_0(f.data() + r * nin, q.data() + r * row_bytes, nin / gguf::Q8_0_BLOCK);
        return q;
    };
    const auto wq = quantized(nout, 101), stack = quantized(n_expert * nout, 102), tq = quantized(nrows, 103);
    const auto xf = uniform(rows * nin, 104), x2f = uniform(entries * nin, 105), tf = uniform(nrows * nin, 106);
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 1, 2};
    const std::vector<float> weights(entries, 0.5f);

    // The valid call, a Q8_0 matmul on the row kernel, checked against the CPU once.
    const auto w = vk.adopt(wq.data(), wq.size()), x = vk.adopt(xf.data(), xf.size() * sizeof(float));
    const auto y = vk.alloc(rows * nout * sizeof(float));
    auto valid = [&] {
        vk.matmul(q8, {w.get(), 0}, {x.get(), 0}, {y.get(), 0}, nin, nout, rows);
        std::vector<float> out(rows * nout);
        vk.read(*y, 0, out.data(), out.size() * sizeof(float));
        return out;
    };
    const std::vector<float> expected = valid();
    {
        backend::CpuBackend cpu;
        cpu.set_threads(1);
        cpu.set_decode_activations8(false);
        const auto xr = rows < backend::tile_from_for(prof, true, nin) ? twin_activations(xf, twin8)
                        : prof.prefer_integer_dot                     ? tile_activations8(xf)
                                                                      : xf;
        const auto cw = cpu.adopt(wq.data(), wq.size()), cx = cpu.adopt(xr.data(), xr.size() * sizeof(float));
        const auto cy = cpu.alloc(rows * nout * sizeof(float), backend::Memory::device);
        cpu.matmul(q8, {cw.get(), 0}, {cx.get(), 0}, {cy.get(), 0}, nin, nout, rows);
        std::vector<float> ref(rows * nout);
        cpu.read(*cy, 0, ref.data(), ref.size() * sizeof(float));
        close(ref, expected, twin8 ? 1e-2 : 1e-4, "the call run after refusals differs from the CPU beyond its bound");
    }

    const size_t kept = 256;
    const std::vector<float> held(kept, 0.25f);
    const auto keep = vk.alloc(kept * sizeof(float));
    vk.write(*keep, 0, held.data(), kept * sizeof(float));

    // A slot lets go of what it holds when it is next opened, and each read is its own submission.
    // One read more than the backend's four slots therefore reopens every slot, the one open when the reads start included.
    const size_t slots = 4;
    auto let_go = [&] {
        float f = 0;
        for (size_t i = 0; i < slots + 1; ++i) vk.read(*keep, 0, &f, sizeof f);
    };

    // `call` makes its operands, takes its outputs from `out` last, and must be refused.
    // In the first pass `out` lets the slots go after each output it makes, so after the last no slot holds any of the call's operands.
    using Out = std::function<backend::Slice(size_t)>;
    size_t checks = 0;
    auto refused = [&](const char* what, const std::function<void(const Out&)>& call) {
        for (int pass = 0; pass < 2; ++pass) {
            {
                std::vector<backend::BufferPtr> made;
                size_t used = 0;
                const Out out = [&](size_t n) -> backend::Slice {
                    if (pass == 0) {
                        made.push_back(vk.alloc(n * sizeof(float)));
                        let_go();
                        return {made.back().get(), 0};
                    }
                    if (used + n > kept) throw std::logic_error("a refused call's outputs exceed the kept buffer");
                    used += n;
                    return {keep.get(), used - n};
                };
                bool threw = false;
                try { call(out); } catch (const std::runtime_error&) { threw = true; }
                require(threw, what);
            }
            const std::vector<float> again = valid();
            require(std::memcmp(again.data(), expected.data(), again.size() * sizeof(float)) == 0,
                    "a call after a refusal differs from the same call before it");
            std::vector<float> now(kept);
            vk.read(*keep, 0, now.data(), kept * sizeof(float));
            require(now == held, "a refused call wrote into its output");
            ++checks;
        }
    };
    auto in = [&](const void* data, size_t bytes) { return vk.adopt(data, bytes); };
    auto floats = [&](const std::vector<float>& v) { return vk.adopt(v.data(), v.size() * sizeof(float)); };
    auto at = [](const backend::BufferPtr& b) { return backend::CSlice{b.get(), 0}; };
    const backend::RowRun disordered[3] = {{2, 1}, {1, 1}, {3, 1}}, beyond[2] = {{1, 1}, {4, 1}}, short_of[2] = {{1, 1}, {2, 1}};
    // A generated token, then a prompt's rows on the tile: two calls, the second reading rows of X the first does not.
    const backend::RowRun two_calls[2] = {{1, 1}, {3, 512}};

    refused("matmul of a type without a kernel accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul(f16, at(wb), at(xb), out(rows * nout), nin, nout, rows);
    });
    refused("matmul of weights shorter than the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size() - row_bytes), xb = floats(xf);
        vk.matmul(q8, at(wb), at(xb), out(rows * nout), nin, nout, rows);
    });
    refused("matmul of a row ending inside a block accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul(q8, at(wb), at(xb), out(rows * nout), partial, nout, rows);
    });
    refused("matmul_add of weights shorter than the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size() - row_bytes), xb = floats(xf);
        vk.matmul_add(q8, at(wb), at(xb), out(rows * nout), nin, nout, rows);
    });
    refused("matmul_group with a short second projection accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), ws = in(wq.data(), wq.size() - row_bytes), xb = floats(xf);
        vk.matmul_group({{q8, at(wb), out(rows * nout), nout}, {q8, at(ws), out(rows * nout), nout}}, at(xb), nin, rows);
    });
    refused("matmul with X short of its second run accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = in(xf.data(), nin * sizeof(float));
        vk.matmul(q8, at(wb), at(xb), out(rows * nout), nin, nout, rows, {two_calls, 2});
    });
    refused("matmul with row runs out of order accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul(q8, at(wb), at(xb), out(rows * nout), nin, nout, rows, {disordered, 3});
    });
    refused("matmul with row runs past the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul(q8, at(wb), at(xb), out(rows * nout), nin, nout, rows, {beyond, 2});
    });
    refused("matmul with row runs short of the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul(q8, at(wb), at(xb), out(rows * nout), nin, nout, rows, {short_of, 2});
    });
    refused("matmul_group with row runs out of order accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul_group({{q8, at(wb), out(rows * nout), nout}, {q8, at(wb), out(rows * nout), nout}}, at(xb), nin, rows, {disordered, 3});
    });

    // The routed products, over n_expert stacked matrices.
    auto routed = [&](const char* what, uint32_t type, size_t stack_bytes, size_t width, backend::RowRuns runs) {
        refused(what, [&](const Out& out) {
            const auto sb = in(stack.data(), stack_bytes), xb = floats(xf), idb = in(ids.data(), ids.size() * sizeof(uint32_t)), wtb = floats(weights);
            vk.matmul_experts({{type, at(sb), out(entries * nout), nout}}, at(xb), width, rows, {at(idb), at(wtb), k, n_expert}, runs);
        });
        refused(what, [&](const Out& out) {
            const auto sb = in(stack.data(), stack_bytes), xb = floats(x2f), idb = in(ids.data(), ids.size() * sizeof(uint32_t)), wtb = floats(weights);
            vk.matmul_experts_add(type, at(sb), at(xb), out(rows * nout), width, nout, rows, {at(idb), at(wtb), k, n_expert}, runs);
        });
    };
    routed("routed product of a type without a kernel accepted", f16, stack.size(), nin, {});
    routed("routed product of a stack short of its experts accepted", q8, stack.size() - row_bytes, nin, {});
    routed("routed product of a row ending inside a block accepted", q8, stack.size(), partial, {});
    routed("routed product with row runs out of order accepted", q8, stack.size(), nin, {disordered, 3});

    const uint32_t first[1] = {0}, past[1] = {uint32_t(nrows)};
    refused("embedding of a type without a kernel accepted", [&](const Out& out) {
        const auto tb = floats(tf);
        vk.embed(out(nin), f16, at(tb), nin, nrows, first, 1);
    });
    refused("F32 embedding table shorter than its rows accepted", [&](const Out& out) {
        const auto tb = floats(tf);
        vk.embed(out(nin), f32, at(tb), nin, nrows + 1, first, 1);
    });
    refused("Q8_0 embedding table shorter than its rows accepted", [&](const Out& out) {
        const auto tb = in(tq.data(), tq.size());
        vk.embed(out(nin), q8, at(tb), nin, nrows + 1, first, 1);
    });
    refused("embedding row ending inside a block accepted", [&](const Out& out) {
        const auto tb = in(tq.data(), tq.size());
        vk.embed(out(partial), q8, at(tb), partial, nrows, first, 1);
    });
    refused("embedding row beyond the table accepted", [&](const Out& out) {
        const auto tb = floats(tf);
        vk.embed(out(nin), f32, at(tb), nin, nrows, past, 1);
    });
    return checks;
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

        // Write from the host into device memory, then into host-visible memory, and a copy whose result the host reads in place.
        const std::vector<uint8_t> patch = pattern(777, 2);
        b->write(*dst, 5000, patch.data(), patch.size());
        b->read(*dst, 5000, window.data(), patch.size());
        require(std::memcmp(window.data(), patch.data(), patch.size()) == 0, "device write differs");
        b->write(*visible, 8, patch.data(), patch.size());
        b->copy(*visible, 2000, *adopted, 4096, 2000);
        const backend::Ticket t = b->submit();
        b->wait(t);
        require(std::memcmp((const uint8_t*)visible->host_ptr() + 8, patch.data(), patch.size()) == 0,
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

        // A KV budget whose bytes overflow is refused before any block is allocated.
        rejected = false;
        try { b->kv_alloc(1, 1, 1, std::numeric_limits<size_t>::max()); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "an overflowing KV budget accepted");
        checks += 1;

        checks += check_refusals(*b);

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
