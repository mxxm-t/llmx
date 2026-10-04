// Vulkan backend test (docs/VULKAN.md): storage and submission over a real device, then every kernel against the CPU backend on random inputs, and every decode column of every row kernel build against the same column alone.
// Bit exact where the arithmetic is the same operation in the same order, a stated tolerance where a transcendental or a reduction order differs.
// Exits 77, which CTest reports as skipped, when there is no loader or no device.
#include <algorithm>
#include <cctype>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "core/host_memory.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "model/kv_cache.hpp"
#include "quant/quant.hpp"
#include "quantizers.hpp"
#include "matrix_precision.hpp"
#include "row_classes.hpp"

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

// The activations a matmul of `type` reads, on the tile kernel or the row kernel.
std::vector<float> fed(const std::vector<float>& x, uint32_t type, bool idot, bool tile) {
    if (type == quant::GGML_TYPE_F32 || (tile && !idot)) return x;
    return row_activations(x);
}

// `rows` rows of `in` values of a type: F32 and the block quantizers from seeded floats, the K-quants from a byte pattern with small half scales, as the matmul checks build them.
std::vector<uint8_t> matrix(uint32_t type, size_t in, size_t rows, uint32_t seed) {
    const auto f = uniform(rows * in, seed);
    std::vector<uint8_t> bytes;
    if (type == quant::GGML_TYPE_F32) {
        bytes.resize(f.size() * sizeof(float));
        std::memcpy(bytes.data(), f.data(), bytes.size());
    } else if (type == quant::GGML_TYPE_MXFP4) {
        bytes = testq::mxfp4_matrix(rows * in, seed);
    } else if (type == quant::GGML_TYPE_Q8_0 || type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1) {
        const size_t ts = type == quant::GGML_TYPE_Q8_0 ? quant::Q8_0_TYPESIZE : type == quant::GGML_TYPE_Q4_0 ? quant::Q4_0_TYPESIZE : quant::Q4_1_TYPESIZE;
        bytes.resize(rows * (in / 32) * ts);
        for (size_t r = 0; r < rows; ++r) {
            uint8_t* dst = bytes.data() + r * (in / 32) * ts;
            if (type == quant::GGML_TYPE_Q8_0) quant::quantize_row_q8_0(f.data() + r * in, dst, in / 32);
            else if (type == quant::GGML_TYPE_Q4_0) quant::quantize_row_q4_0(f.data() + r * in, dst, in / 32);
            else testq::quantize_row_q4_1(f.data() + r * in, dst, in / 32);
        }
    } else {
        const size_t ts = type == quant::GGML_TYPE_Q6_K ? quant::Q6_K_TYPESIZE : type == quant::GGML_TYPE_Q4_K ? quant::Q4_K_TYPESIZE : quant::Q5_K_TYPESIZE;
        bytes.resize(rows * (in / 256) * ts);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * (61 + seed % 7) + 3);
        for (size_t b = 0; b < rows * (in / 256); ++b) {
            uint8_t* blk = bytes.data() + b * ts;
            if (type == quant::GGML_TYPE_Q6_K) { blk[208] = 0x00; blk[209] = 0x14; }
            else { blk[0] = 0x00; blk[1] = 0x14; blk[2] = 0x00; blk[3] = 0x10; }
        }
    }
    return bytes;
}

// Choose the nearer mathematical BF16 neighbour independently of either converter.
// A non-BF16 weight checks that only inputs are rounded. A sum may turn -0 into +0, so zero sign is not checked here.
size_t check_bf16_rounding(backend::Backend& vk) {
    auto value = [](uint32_t b) {
        const int exponent = int(b >> 7);
        return exponent ? std::ldexp(1.0 + double(b & 127u) / 128.0, exponent - 127) : std::ldexp(double(b), -133);
    };
    std::vector<float> input, rounded;
    for (uint32_t b = 0; b < 0x7f80u; ++b) {
        for (uint32_t tail : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
            const uint32_t bits = (b << 16) | tail;
            float f;
            std::memcpy(&f, &bits, sizeof f);
            const double low = double(f) - value(b), high = value(b + 1) - double(f);
            const uint32_t chosen = high < low || (high == low && (b & 1u)) ? b + 1 : b;
            const float want = chosen == 0x7f80u ? std::numeric_limits<float>::infinity() : float(value(chosen));
            for (bool negative : {false, true}) {
                input.push_back(negative ? -f : f); rounded.push_back(negative ? -want : want);
            }
        }
    }
    for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7f800001u, 0xff800001u, 0x7fc00000u, 0xffc00000u}) {
        float f;
        std::memcpy(&f, &bits, sizeof f);
        input.push_back(f); rounded.push_back(f);
    }
    const float weights[] = {1.0f, 1.0009765625f};
    const auto w = vk.adopt(weights, sizeof weights), x = vk.adopt(input.data(), input.size() * sizeof(float)),
               y = vk.alloc(input.size() * 2 * sizeof(float));
    std::vector<float> output(input.size() * 2);
    size_t checked = 0;
    for (auto dtype : {backend::Dtype::bf16, backend::Dtype::f32}) {
        testq::take_matrix_paths(vk);
        vk.matmul(quant::GGML_TYPE_F32, {w.get(), 0}, {x.get(), 0}, {y.get(), 0}, 1, 2, input.size(), {}, dtype);
        vk.read(*y, 0, output.data(), output.size() * sizeof(float));
        require(testq::take_matrix_paths(vk) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"},
                "BF16 conversion check has the wrong arithmetic witness");
        const auto& reference = dtype == backend::Dtype::bf16 ? rounded : input;
        for (size_t c = 0; c < input.size(); ++c) for (size_t row = 0; row < 2; ++row) {
            const float expected = reference[c] * weights[row], actual = output[2 * c + row];
            if (!(actual == expected || (std::isnan(actual) && std::isnan(expected)))) {
                std::fprintf(stderr, "BF16 boundary input %zu row %zu: got %.9g expected %.9g\n", c, row, actual, expected);
                throw std::runtime_error("BF16 nearest rounding or exact weights differ");
            }
            ++checked;
        }
    }
    return checked;
}

size_t check_matrix_witness(backend::Backend& vk) {
    const bool integer = backend::vulkan_device_profile(vk).prefer_integer_dot;
    testq::take_matrix_paths(vk);
    const auto native = vk.native_dtypes();
    require(vk.dtype_path(backend::Dtype::f32) == "f32", "device F32 catalog claims a narrower path");
    require(vk.dtype_path(backend::Dtype::bf16) == "bf16 (matrix inputs), f32 (routers)", "device BF16 catalog hides retained F32 routers");
    const std::string half_paths = integer
        ? "block-int16 (quantized rows and eligible tiles), f32 (other products, routers)"
        : "block-int16 (quantized rows), f32 (tiles, F32 weights, routers)";
    require(vk.dtype_path(backend::Dtype::f16) == half_paths, "device path catalog ignores the tile profile");
    require(vk.native_dtypes() == native, "device path catalog changed native policy preferences");
    require(testq::take_matrix_paths(vk).empty(), "device path catalog fabricated an execution witness");
    std::vector<uint8_t> weights(18, 0x88);
    weights[0] = 0; weights[1] = 0x3c; weights[3] = 0x89;
    std::vector<float> x(96), out(6), ids(3, 0), gains(3, 1);
    for (size_t r = 0; r < 3; ++r) { x[r * 32] = 2; x[r * 32 + 1] = r == 1 ? -1.005859375f : 1.005859375f; }
    const auto w = vk.adopt(weights.data(), weights.size()), in = vk.adopt(x.data(), x.size() * sizeof(float)),
               y = vk.alloc(out.size() * sizeof(float)), id = vk.adopt(ids.data(), ids.size() * sizeof(float)),
               gain = vk.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing route{{id.get(), 0}, {gain.get(), 0}, 1, 1};
    const uint32_t type = quant::GGML_TYPE_Q4_0;
    size_t checked = 0;
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::f16}) {
        for (bool tile : {false, true}) {
            const backend::RowRun run[] = {{3, tile ? size_t(16384) : size_t(1)}};
            for (int op = 0; op < 6; ++op) {
                std::fill(out.begin(), out.end(), 0.0f);
                vk.write(*y, 0, out.data(), out.size() * sizeof(float));
                testq::take_matrix_paths(vk);
                const backend::CSlice ws{w.get(), 0}, xs{in.get(), 0};
                const backend::Slice ys{y.get(), 0};
                const backend::RowRuns runs{run, 1};
                if (op == 0) vk.matmul(type, ws, xs, ys, 32, 1, 3, runs, dtype);
                if (op == 1) vk.matmul_logits(type, ws, xs, ys, 32, 1, 3, runs, dtype);
                if (op == 2) vk.matmul_add(type, ws, xs, ys, 32, 1, 3, runs, dtype);
                if (op == 3) vk.matmul_group({{type, ws, ys, 1}, {type, ws, {y.get(), 3}, 1}}, xs, 32, 3, runs, dtype);
                if (op == 4) vk.matmul_experts({{type, ws, ys, 1}}, xs, 32, 3, route, runs, dtype);
                if (op == 5) vk.matmul_experts_add(type, ws, xs, ys, 32, 1, 3, route, runs, dtype);
                vk.read(*y, 0, out.data(), out.size() * sizeof(float));
                const bool narrow = dtype == backend::Dtype::f16 && (!tile || integer);
                const auto rounded = !narrow ? x : row_activations(x);
                for (size_t r = 0; r < (op == 3 ? 6u : 3u); ++r) {
                    const float expected = dtype == backend::Dtype::bf16 ? (r % 3 == 1 ? -1.0078125f : 1.0078125f)
                                                                         : rounded[(r % 3) * 32 + 1];
                    require(std::fabs(out[r] - expected) < 2e-6f, "matrix witness arithmetic differs");
                }
                const std::string path = dtype == backend::Dtype::bf16 ? "bf16" : !narrow ? "f32" : "block-int16";
                require(testq::take_matrix_paths(vk) == std::vector<std::string>{path}, "matrix witness differs from dispatched arithmetic");
                require(testq::take_matrix_paths(vk).empty(), "matrix witness crossed measurement boundary");
                ++checked;
            }
        }
    }
    return checked;
}

size_t check_kernels(backend::Backend& vk) {
    Pair p(vk);
    const bool integer_dot = backend::vulkan_device_profile(p.vk).prefer_integer_dot;
    // The integer-dot tile scales each block's exact integer sum by the weight's and the activation's scales and adds the blocks in its own order, which moves an output whose products cancel by up to a few 1e-4 against the CPU's float sums.
    auto tol_of = [&](uint32_t type, bool tile) { return tile && integer_dot && type != quant::GGML_TYPE_F32 ? 1e-3 : 1e-4; };
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
    // norm_rope_partial: positions per row out of order; the whole head rotated in place over contiguous heads, as Qwen3 calls it; q read between its gates at a padded stride with a quarter of each 256-wide head rotated, and k in place so, as qwen35 calls it.
    {
        const size_t rows = 5, table = 12;
        const uint32_t pos[rows] = {5, 2, 9, 0, 11};
        struct Case { size_t heads, dim, rope, src_stride, head_stride; bool in_place; };
        const Case cases[] = {{3, 128, 128, 3 * 128, 128, true}, {3, 256, 64, 3 * 512 + 8, 512, false}, {2, 256, 64, 2 * 256, 256, true},
                              {4, 40, 8, 4 * 80, 80, false}};
        for (const Case& c : cases) {
            const size_t half = c.rope / 2, width = c.heads * c.dim, src_floats = (rows - 1) * c.src_stride + (c.heads - 1) * c.head_stride + c.dim;
            const auto x = uniform(src_floats, 8, -2.0f, 2.0f), w = uniform(c.dim, 9, 0.5f, 1.5f);
            std::vector<float> cs(table * half), sn(table * half);
            for (size_t t = 0; t < table; ++t)
                for (size_t i = 0; i < half; ++i) {
                    const double f = std::pow(10000.0, -2.0 * double(i) / double(c.rope));
                    cs[t * half + i] = float(std::cos(double(t) * f));
                    sn[t * half + i] = float(std::sin(double(t) * f));
                }
            Pair::In xi = p.in(x), wi = p.in(w), ci = p.in(cs), si = p.in(sn);
            Pair::Out d = p.out(c.in_place ? src_floats : rows * width);
            if (c.in_place) {
                p.cpu.write(*d.c, 0, x.data(), x.size() * sizeof(float));
                p.vk.write(*d.v, 0, x.data(), x.size() * sizeof(float));
            }
            const backend::CSlice cs_src = c.in_place ? backend::CSlice(d.cs()) : xi.cs(), vs_src = c.in_place ? backend::CSlice(d.vs()) : xi.vs();
            p.cpu.norm_rope_partial(d.cs(), cs_src, rows, c.src_stride, c.head_stride, c.heads, c.dim, c.rope, wi.cs(), 1e-6f, ci.cs(), si.cs(), pos);
            p.vk.norm_rope_partial(d.vs(), vs_src, rows, c.src_stride, c.head_stride, c.heads, c.dim, c.rope, wi.vs(), 1e-6f, ci.vs(), si.vs(), pos);
            auto r = p.results(d);
            values += close(r.first, r.second, 1e-5, "norm_rope_partial differs beyond 1e-5");
            const uint32_t beyond[1] = {12};
            bool rejected = false;
            try { p.vk.norm_rope_partial(d.vs(), vs_src, 1, c.src_stride, c.head_stride, c.heads, c.dim, c.rope, wi.vs(), 1e-6f, ci.vs(), si.vs(), beyond); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "position beyond the table accepted");
        }
    }
    // embed: F32 rows are copies and Q8_0 rows are a half scale times a small integer, exact in float, so both are exact.
    {
        const size_t nin = 96, nrows = 10;
        const auto table = uniform(nin * nrows, 10);
        const uint32_t ids[3] = {3, 9, 0};
        Pair::In t = p.in(table);
        Pair::Out d = p.out(nin * 3);
        p.cpu.embed(d.cs(), quant::GGML_TYPE_F32, t.cs(), nin, nrows, ids, 3);
        p.vk.embed(d.vs(), quant::GGML_TYPE_F32, t.vs(), nin, nrows, ids, 3);
        auto r = p.results(d);
        values += exact(r.first, r.second, "embed F32 differs");

        std::vector<uint8_t> q(nrows * (nin / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q8_0(table.data() + row * nin,
                                     q.data() + row * (nin / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE,
                                     nin / quant::Q8_0_BLOCK);
        Pair::In tq = p.in(q.data(), q.size());
        Pair::Out dq = p.out(nin * 3);
        p.cpu.embed(dq.cs(), quant::GGML_TYPE_Q8_0, tq.cs(), nin, nrows, ids, 3);
        p.vk.embed(dq.vs(), quant::GGML_TYPE_Q8_0, tq.vs(), nin, nrows, ids, 3);
        auto rq = p.results(dq);
        values += exact(rq.first, rq.second, "embed Q8_0 differs");

        std::vector<uint8_t> q4(nrows * (nin / quant::Q4_0_BLOCK) * quant::Q4_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q4_0(table.data() + row * nin,
                                     q4.data() + row * (nin / quant::Q4_0_BLOCK) * quant::Q4_0_TYPESIZE,
                                     nin / quant::Q4_0_BLOCK);
        Pair::In t4 = p.in(q4.data(), q4.size());
        Pair::Out d4 = p.out(nin * 3);
        p.cpu.embed(d4.cs(), quant::GGML_TYPE_Q4_0, t4.cs(), nin, nrows, ids, 3);
        p.vk.embed(d4.vs(), quant::GGML_TYPE_Q4_0, t4.vs(), nin, nrows, ids, 3);
        auto r4 = p.results(d4);
        values += exact(r4.first, r4.second, "embed Q4_0 differs");
        // Raw Q4_0 rows, every second block under a negative scale, which the quantizer never writes: nibble 8 then decodes as -0 on both backends.
        std::vector<uint8_t> raw4(q4.size());
        for (size_t b = 0; b < raw4.size() / quant::Q4_0_TYPESIZE; ++b) {
            uint8_t* blk = raw4.data() + b * quant::Q4_0_TYPESIZE;
            blk[0] = 0x00;
            blk[1] = b % 2 ? 0xB8 : 0x38;   // -0.5 or 0.5
            for (size_t j = 0; j < 16; ++j) blk[2 + j] = uint8_t(((j + b) & 15) | (((5 * j + 3 + b) & 15) << 4));
        }
        Pair::In traw4 = p.in(raw4.data(), raw4.size());
        Pair::Out draw4 = p.out(nin * 3);
        p.cpu.embed(draw4.cs(), quant::GGML_TYPE_Q4_0, traw4.cs(), nin, nrows, ids, 3);
        p.vk.embed(draw4.vs(), quant::GGML_TYPE_Q4_0, traw4.vs(), nin, nrows, ids, 3);
        auto rraw4 = p.results(draw4);
        size_t negative_zeros = 0;
        for (float v : rraw4.first) if (v == 0.0f && std::signbit(v)) ++negative_zeros;
        require(negative_zeros == 10, "raw Q4_0 rows do not decode ten -0 values on the CPU");
        values += exact(rraw4.first, rraw4.second, "embed Q4_0 under a negative scale differs");

        // Q6_K rows of 256 from fixed bytes, decoded by both.
        {
            const size_t n6 = 512;
            std::vector<uint8_t> q6(nrows * (n6 / 256) * quant::Q6_K_TYPESIZE);
            for (size_t i = 0; i < q6.size(); ++i) q6[i] = uint8_t(i * 37 + 11);
            for (size_t b = 0; b < nrows * (n6 / 256); ++b) {
                q6[b * quant::Q6_K_TYPESIZE + 208] = 0x00;
                q6[b * quant::Q6_K_TYPESIZE + 209] = 0x30;
            }
            Pair::In t6 = p.in(q6.data(), q6.size());
            Pair::Out d6 = p.out(n6 * 3);
            p.cpu.embed(d6.cs(), quant::GGML_TYPE_Q6_K, t6.cs(), n6, nrows, ids, 3);
            p.vk.embed(d6.vs(), quant::GGML_TYPE_Q6_K, t6.vs(), n6, nrows, ids, 3);
            auto r6 = p.results(d6);
            values += exact(r6.first, r6.second, "embed Q6_K differs");
        }
        // Q4_K and Q5_K rows of 256 the same way; d and dmin are the first two halves of a block.
        for (int k = 0; k < 2; ++k) {
            const uint32_t type = k == 0 ? quant::GGML_TYPE_Q4_K : quant::GGML_TYPE_Q5_K;
            const size_t bytes = k == 0 ? quant::Q4_K_TYPESIZE : quant::Q5_K_TYPESIZE;
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
        try { p.vk.embed(d.vs(), quant::GGML_TYPE_F32, t.vs(), nin, nrows, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "embedding row beyond the table accepted");
        // A table that holds fewer rows than the call names is refused before any row is read, even when every id is inside the table.
        const uint32_t first[1] = {0};
        auto short_table_refused = [&](uint32_t type, const Pair::In& table_in) {
            try { p.vk.embed(d.vs(), type, table_in.vs(), nin, nrows + 1, first, 1); }
            catch (const std::runtime_error& e) { return std::string(e.what()).find("outside its allocation") != std::string::npos; }
            return false;
        };
        require(short_table_refused(quant::GGML_TYPE_F32, t), "F32 embedding table shorter than its rows accepted");
        require(short_table_refused(quant::GGML_TYPE_Q8_0, tq), "Q8_0 embedding table shorter than its rows accepted");
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
        p.vk.matmul(quant::GGML_TYPE_F32, {wb.get(), 0}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.write(*wb, 0, w2.data(), w2.size() * sizeof(float));
        p.vk.matmul(quant::GGML_TYPE_F32, {wb.get(), 0}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.matmul(quant::GGML_TYPE_F32, {fresh.get(), 0}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
        std::vector<float> a(nout * cols), b(nout * cols);
        p.vk.read(*y1, 0, a.data(), a.size() * sizeof(float));
        p.vk.read(*y2, 0, b.data(), b.size() * sizeof(float));
        values += exact(a, b, "an F32 matmul after a write into its weights read the old weights");
    }
    // A loader's weight, allocated with alloc_weight and written in pieces that end inside a row, holds the bytes and gives the products of the same weight adopted, whose storage adopt allocates the same way; the float tile reads both through its padded copy.
    {
        const size_t nin = 256, nout = 67, cols = 64;
        const auto w = uniform(nin * nout, 24), xx = uniform(nin * cols, 25);
        const size_t bytes = w.size() * sizeof(float), piece = 4100;
        auto adopted = p.vk.adopt(w.data(), bytes);
        auto filled = p.vk.alloc_weight(bytes);
        for (size_t off = 0; off < bytes; off += piece)
            p.vk.write(*filled, off, (const uint8_t*)w.data() + off, std::min(piece, bytes - off));
        auto xb = p.vk.adopt(xx.data(), xx.size() * sizeof(float));
        auto y1 = p.vk.alloc(nout * cols * sizeof(float), backend::Memory::device);
        auto y2 = p.vk.alloc(nout * cols * sizeof(float), backend::Memory::device);
        p.vk.matmul(quant::GGML_TYPE_F32, {adopted.get(), 0}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.matmul(quant::GGML_TYPE_F32, {filled.get(), 0}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
        std::vector<float> a(nout * cols), b(nout * cols), back(w.size());
        p.vk.read(*y1, 0, a.data(), a.size() * sizeof(float));
        p.vk.read(*y2, 0, b.data(), b.size() * sizeof(float));
        p.vk.read(*filled, 0, back.data(), bytes);
        require(std::memcmp(back.data(), w.data(), bytes) == 0, "a weight written in pieces into alloc_weight storage differs from its source");
        values += exact(a, b, "a weight written in pieces into alloc_weight storage gave other products than the same weight adopted");
    }
    // Host pages the device imports, as a streamed load's read ring: a weight in them copied in pieces that end inside a row into alloc_weight storage holds the bytes and gives the products of the same weight adopted, and memory off the page or of a part of a page is not imported.
    {
        const size_t nin = 256, nout = 67, cols = 64;
        const auto w = uniform(nin * nout, 26), xx = uniform(nin * cols, 27);
        const size_t bytes = w.size() * sizeof(float), piece = 4100;
        core::HostPages pages(bytes);
        std::memcpy(pages.data(), w.data(), bytes);
        require(!p.vk.wrap_host(pages.data() + 64, core::page_size()) && !p.vk.wrap_host(pages.data(), 100),
                "host memory off the page, or a part of a page, was imported");
        auto view = p.vk.wrap_host(pages.data(), pages.size());
        if (view) {
            auto adopted = p.vk.adopt(w.data(), bytes);
            auto filled = p.vk.alloc_weight(bytes);
            for (size_t off = 0; off < bytes; off += piece) p.vk.copy(*filled, off, *view, off, std::min(piece, bytes - off));
            auto xb = p.vk.adopt(xx.data(), xx.size() * sizeof(float));
            auto y1 = p.vk.alloc(nout * cols * sizeof(float), backend::Memory::device);
            auto y2 = p.vk.alloc(nout * cols * sizeof(float), backend::Memory::device);
            p.vk.matmul(quant::GGML_TYPE_F32, {adopted.get(), 0}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
            p.vk.matmul(quant::GGML_TYPE_F32, {filled.get(), 0}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
            std::vector<float> a(nout * cols), b(nout * cols), back(w.size());
            p.vk.read(*y1, 0, a.data(), a.size() * sizeof(float));
            p.vk.read(*y2, 0, b.data(), b.size() * sizeof(float));
            p.vk.read(*filled, 0, back.data(), bytes);
            require(std::memcmp(back.data(), w.data(), bytes) == 0, "a weight copied in pieces out of imported host pages differs from its source");
            values += exact(a, b, "a weight copied in pieces out of imported host pages gave other products than the same weight adopted");
        } else {
            std::cout << "backend-vulkan: this device does not import host memory, so a streamed load writes through staging\n";
        }
    }
    // matmul: F32 and Q8_0 over odd sizes and batch widths that fall inside, on and past the eight-column chunk.
    // The reduction order differs from the CPU's, so a tolerance.
    for (size_t nin : {size_t(1024), size_t(256), size_t(224)}) {
        // 1024 is sixteen block pairs, the word-wide path; 256 has too few pairs for it and 224 an odd block count, both the 16-bit path.
        const size_t nout = 67;
        const auto wf = uniform(nin * nout, 11);
        std::vector<uint8_t> wq(nout * (nin / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q8_0(wf.data() + row * nin,
                                     wq.data() + row * (nin / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE,
                                     nin / quant::Q8_0_BLOCK);
        std::vector<uint8_t> w4(nout * (nin / quant::Q4_0_BLOCK) * quant::Q4_0_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q4_0(wf.data() + row * nin,
                                     w4.data() + row * (nin / quant::Q4_0_BLOCK) * quant::Q4_0_TYPESIZE,
                                     nin / quant::Q4_0_BLOCK);
        std::vector<uint8_t> w41(nout * (nin / quant::Q4_1_BLOCK) * quant::Q4_1_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            testq::quantize_row_q4_1(wf.data() + row * nin,
                                     w41.data() + row * (nin / quant::Q4_1_BLOCK) * quant::Q4_1_TYPESIZE,
                                     nin / quant::Q4_1_BLOCK);
        // No Q6_K quantizer is needed: any bytes are a valid block and both backends decode the same bytes.
        // The half scale is 2^-10 so the values sit in the range of the other types; larger scales made reduction-order rounding alone exceed the tolerance.
        std::vector<uint8_t> w6(nout * (nin / 256) * quant::Q6_K_TYPESIZE);
        for (size_t i = 0; i < w6.size(); ++i) w6[i] = uint8_t(i * 131 + 7);
        for (size_t row = 0; row < nout * (nin / 256); ++row) {
            // Keep the half scale finite and small.
            w6[row * quant::Q6_K_TYPESIZE + 208] = 0x00;
            w6[row * quant::Q6_K_TYPESIZE + 209] = 0x14;
        }
        // Q4_K and Q5_K likewise, d and dmin at 2^-10 and 2^-11.
        std::vector<uint8_t> w4k(nout * (nin / 256) * quant::Q4_K_TYPESIZE), w5k(nout * (nin / 256) * quant::Q5_K_TYPESIZE);
        for (size_t i = 0; i < w4k.size(); ++i) w4k[i] = uint8_t(i * 61 + 3);
        for (size_t i = 0; i < w5k.size(); ++i) w5k[i] = uint8_t(i * 67 + 9);
        for (size_t row = 0; row < nout * (nin / 256); ++row) {
            for (uint8_t* blk : {w4k.data() + row * quant::Q4_K_TYPESIZE, w5k.data() + row * quant::Q5_K_TYPESIZE}) {
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
            // A device whose integer dot is native takes wide quantized batches through the integer-dot tile, which reads the 16-bit twin.
            const bool idot = profile.prefer_integer_dot;
            const bool tile8 = nbatch >= tile_from_8bit, tile_other = nbatch >= tile_from_other;
            const auto xr8 = fed(x, quant::GGML_TYPE_Q8_0, idot, tile8);   // adopted, so they must outlive the calls
            const auto xr4 = fed(x, quant::GGML_TYPE_Q4_0, idot, tile_other);
            const auto xrk = fed(x, quant::GGML_TYPE_Q4_K, idot, tile_other);
            Pair::In xri8 = p.in(xr8), xri4 = p.in(xr4), xrik = p.in(xrk);
            for (int q = 0; q < 7; ++q) {
                if (q >= 4 && nin % 256) continue;   // K-quant blocks are 256 wide
                const uint32_t type = q == 1 ? quant::GGML_TYPE_Q8_0 : q == 2 ? quant::GGML_TYPE_Q4_0
                                    : q == 3 ? quant::GGML_TYPE_Q4_1 : q == 4 ? quant::GGML_TYPE_Q6_K
                                    : q == 5 ? quant::GGML_TYPE_Q4_K : q == 6 ? quant::GGML_TYPE_Q5_K
                                    : quant::GGML_TYPE_F32;
                const Pair::In& wi = q == 1 ? wqi : q == 2 ? w4i : q == 3 ? w41i : q == 4 ? w6i
                                   : q == 5 ? w4ki : q == 6 ? w5ki : wfi;
                Pair::Out d = p.out(nbatch * nout);
                p.cpu.matmul(type, wi.cs(), (q == 0 ? xi : q == 1 ? xri8 : q <= 3 ? xri4 : xrik).cs(), d.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
                p.vk.matmul(type, wi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
                auto r = p.results(d);
                try {
                    const double tol = tol_of(type, q == 1 ? tile8 : tile_other);
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
                        p.cpu.matmul(type, wi.cs(), x16i.cs(), h.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
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
        // attention over a wide pass of 128-wide heads, and of 256-wide heads six to a KV head as qwen35's 27B has them, takes the tiled kernel: 32, 45 and 100 query rows (one full tile, then partial ones whose last rows mask part of a K/V tile) after histories of 0, 70 and 600 tokens, against the CPU at 1e-4.
        // The queries are scaled so a row's scores spread over about ten, peaked as a trained model's are rather than the near-uniform softmax of unit random values, and the cache is taken both as f32 and as f16.
        for (int head_dim : {128, 256})
        for (backend::KVType kt : {backend::KVType::f32, backend::KVType::f16})
        for (size_t hist : {size_t(0), size_t(70), size_t(600)}) {
            for (size_t nq : {size_t(32), size_t(45), size_t(100)}) {
                const int n_head = head_dim == 128 ? 4 : 12, n_head_kv = 2;
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
                    std::fprintf(stderr, "  tiled attention width %d hist %zu rows %zu cache %s\n", head_dim, hist, nq, backend::kv_type_name(kt));
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
        for (int group : {1, 2, 3, 4, 6, 8}) {
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
            std::vector<uint8_t> wi8(n_out * (n_in / 32) * quant::Q8_0_TYPESIZE), wi40(n_out * (n_in / 32) * quant::Q4_0_TYPESIZE);
            for (size_t r = 0; r < n_out; ++r) {
                quant::quantize_row_q8_0(wi.data() + r * n_in, wi8.data() + r * (n_in / 32) * quant::Q8_0_TYPESIZE, n_in / 32);
                quant::quantize_row_q4_0(wi.data() + r * n_in, wi40.data() + r * (n_in / 32) * quant::Q4_0_TYPESIZE, n_in / 32);
            }
            std::vector<uint8_t> wi6(n_out * (n_in / 256) * quant::Q6_K_TYPESIZE), wi4k(n_out * (n_in / 256) * quant::Q4_K_TYPESIZE);
            for (size_t i = 0; i < wi6.size(); ++i) wi6[i] = uint8_t(i * 131 + 7);
            for (size_t i = 0; i < wi4k.size(); ++i) wi4k[i] = uint8_t(i * 61 + 3);
            for (size_t r = 0; r < n_out * (n_in / 256); ++r) {
                wi6[r * quant::Q6_K_TYPESIZE + 208] = 0x00;
                wi6[r * quant::Q6_K_TYPESIZE + 209] = 0x14;
                uint8_t* blk = wi4k.data() + r * quant::Q4_K_TYPESIZE;
                blk[0] = 0x00; blk[1] = 0x14; blk[2] = 0x00; blk[3] = 0x10;
            }
            const auto x = uniform(rows * n_in, 71);
            const auto y0 = uniform(rows * n_out, 72);
            const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
            struct W { uint32_t type; const void* data; size_t bytes; };
            for (const W& t : {W{quant::GGML_TYPE_F32, wi.data(), wi.size() * sizeof(float)},
                               W{quant::GGML_TYPE_Q8_0, wi8.data(), wi8.size()}, W{quant::GGML_TYPE_Q4_0, wi40.data(), wi40.size()},
                               W{quant::GGML_TYPE_Q6_K, wi6.data(), wi6.size()}, W{quant::GGML_TYPE_Q4_K, wi4k.data(), wi4k.size()}}) {
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
                        // What a paused request's resume relies on: one sequence's 40 generated rows in one run of extent 1 beside a prompt compute what each computed decoded in a call of its own.
                        const auto replay = run(0, 100, {{40, 1}, {100, prompt}});
                        for (size_t r : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(15), size_t(31), size_t(39)})
                            values += exact(part(replay, r, r + 1), run(r, 1, {{1, 1}}), "a generated row of a 40-row replay beside a prompt differs from it decoded alone");
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
            // What a paused request's resume relies on: 40 generated rows of one sequence after a 500-token history, attended in one view of extent 1 beside a 60-row prompt's view, against the same rows decoded one call at a time.
            try {
                const size_t gen = 40, prompt = 60, other = 1000;
                infer::KVSequence a(&pool, bt), b(&pool, bt), c(&pool, bt);
                for (auto* s : {&a, &c}) {
                    s->prepare(ha);
                    const backend::KVView h = s->view(st.get());
                    vk.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                    s->commit();
                }
                a.prepare(gen);
                b.prepare(prompt);
                backend::KVView views[2] = {a.view(st.get()), b.view(st.get())};
                views[0].extent = 1;
                views[1].extent = prompt;
                vk.kv_write(0, &views[0], 1, {Kb.get(), ha * kvw}, {Vb.get(), ha * kvw});
                vk.kv_write(0, &views[1], 1, {Kb.get(), other * kvw}, {Vb.get(), other * kvw});
                const auto both = read_rows(views, 2, 0, gen + prompt);
                for (size_t i = 0; i < gen; ++i) {
                    c.prepare(1);
                    backend::KVView v = c.view(st.get());
                    v.extent = 1;
                    vk.kv_write(0, &v, 1, {Kb.get(), (ha + i) * kvw}, {Vb.get(), (ha + i) * kvw});
                    values += exact(std::vector<float>(both.begin() + i * qw, both.begin() + (i + 1) * qw), read_rows(&v, 1, i, 1),
                                    "a generated row of a 40-row replay beside a prompt attends otherwise than decoded alone");
                    c.commit();
                }
                a.abort();
                b.abort();
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  replay by class: attention, cache %s\n", backend::kv_type_name(kt));
                throw;
            }
        }
        // The twin that a norm, a SiLU and a wide attention write four values a lane when told the integer-dot tile reads their output next is the one the tile's own pass makes, bit for bit: each producer with runs into a buffer and a matmul with those runs from it, against the producer without them.
        // A write from the host or a kernel between the producer and the matmul drops the twin, so the matmul reads what the buffer holds then.
        if (nin == 1024) {
            const size_t n_in = 1024, n_out = 96, rows = 128;
            const std::vector<backend::RowRun> one{{rows, rows}};
            const backend::RowRuns rr{one.data(), one.size()};
            const auto wf = uniform(n_out * n_in, 90);
            std::vector<uint8_t> w8(n_out * (n_in / 32) * quant::Q8_0_TYPESIZE);
            for (size_t r = 0; r < n_out; ++r)
                quant::quantize_row_q8_0(wf.data() + r * n_in, w8.data() + r * (n_in / 32) * quant::Q8_0_TYPESIZE, n_in / 32);
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
                vk.matmul(quant::GGML_TYPE_Q8_0, {wb.get(), 0}, {x.get(), 0}, {yb.get(), 0}, n_in, n_out, rows, rr);
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
                values += exact(matmul_of(norm(true)), matmul_of(norm(false)), "a matmul from the norm's twin differs from its own pass");
                const auto f1 = silu(true), f0 = silu(false);
                values += exact(floats(f1), floats(f0), "the SiLU's output differs with and without its runs");
                values += exact(matmul_of(silu(true)), matmul_of(silu(false)), "a matmul from the SiLU's twin differs from its own pass");
                // A routed down projection reads the SiLU's output as k entries a token row, each of its token's prompt, so the SiLU given those runs writes the copy the routed tile reads (model/arch/qwen3.hpp).
                {
                    const size_t k = 2, n_expert = 4, entries = rows * k, ebytes = entries * n_in * sizeof(float);
                    const std::vector<backend::RowRun> eruns{{entries, rows}};
                    const auto ef = uniform(n_expert * n_out * n_in, 99);
                    std::vector<uint8_t> e8(n_expert * n_out * (n_in / 32) * quant::Q8_0_TYPESIZE);
                    for (size_t r = 0; r < n_expert * n_out; ++r)
                        quant::quantize_row_q8_0(ef.data() + r * n_in, e8.data() + r * (n_in / 32) * quant::Q8_0_TYPESIZE, n_in / 32);
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
                        vk.matmul_experts_add(quant::GGML_TYPE_Q8_0, {eb.get(), 0}, {f.get(), 0}, {yb.get(), 0}, n_in, n_out, rows, routing, rr);
                        std::vector<float> y(rows * n_out);
                        vk.read(*yb, 0, y.data(), y.size() * sizeof(float));
                        return y;
                    };
                    values += exact(routed(true), routed(false), "a routed down projection from the SiLU's twin differs from its own pass");
                }
                const auto o1 = attend(true), o0 = attend(false);
                values += exact(floats(o1), floats(o0), "the attention's output differs with and without its runs");
                const auto y1 = matmul_of(attend(true)), y0 = matmul_of(attend(false));
                values += exact(y1, y0, "a matmul from the attention's twin differs from its own pass");
                const auto reference = matmul_of(otherb);
                const auto hw = norm(true);
                vk.write(*hw, 0, other.data(), bytes);
                values += exact(matmul_of(hw), reference, "a matmul read a norm's twin after the host wrote its output");
                std::vector<float> sum(rows * n_in);
                const auto plain = floats(norm(false));
                for (size_t i = 0; i < sum.size(); ++i) sum[i] = plain[i] + other[i];
                const auto sumb = vk.adopt(sum.data(), bytes);
                const auto ha = norm(true);
                vk.add({ha.get(), 0}, {otherb.get(), 0}, rows * n_in);
                values += exact(matmul_of(ha), matmul_of(sumb), "a matmul read a norm's twin after a kernel wrote its output");
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  producers' twin for the tile\n");
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
                std::vector<uint8_t> wq(nout * (qw / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
                for (size_t row = 0; row < nout; ++row)
                    quant::quantize_row_q8_0(wf.data() + row * qw, wq.data() + row * (qw / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE,
                                             qw / quant::Q8_0_BLOCK);
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
                        b.matmul(quant::GGML_TYPE_Q8_0, {Wb.get(), 0}, {ob.get(), 0}, {yb.get(), 0}, qw, nout, nq);
                    } else {
                        std::vector<float> att(nq * qw);
                        b.read(*ob, 0, att.data(), att.size() * sizeof(float));
                        const auto ar = fed(att, quant::GGML_TYPE_Q8_0, integer_dot, false);
                        const auto Ab = b.adopt(ar.data(), ar.size() * sizeof(float));
                        b.matmul(quant::GGML_TYPE_Q8_0, {Wb.get(), 0}, {Ab.get(), 0}, {yb.get(), 0}, qw, nout, nq);
                    }
                    y.resize(nq * nout);
                    b.read(*yb, 0, y.data(), y.size() * sizeof(float));
                };
                std::vector<float> yc, yv;
                run(p.cpu, false, yc);
                run(p.vk, true, yv);
                values += close(yc, yv, 1e-4, "matmul from the attention twin differs beyond its bound");
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
            const bool tile = nbatch >= backend::tile_from_for(prof, true, nin);
            const auto xr = fed(xa, quant::GGML_TYPE_Q8_0, integer_dot, tile);
            Pair::In xri = p.in(xr);
            const auto y0 = uniform(nbatch * nout, 21 + (uint32_t)nbatch);
            Pair::Out d = p.out(nbatch * nout);
            p.cpu.write(*d.c, 0, y0.data(), y0.size() * sizeof(float));
            p.vk.write(*d.v, 0, y0.data(), y0.size() * sizeof(float));
            p.cpu.matmul_add(quant::GGML_TYPE_Q8_0, wqi.cs(), xri.cs(), d.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
            p.vk.matmul_add(quant::GGML_TYPE_Q8_0, wqi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
            auto r = p.results(d);
            values += close(r.first, r.second, tol_of(quant::GGML_TYPE_Q8_0, tile), "matmul_add differs beyond its bound");
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
            p.vk.matmul(quant::GGML_TYPE_Q8_0, wqi.vs(), h.vs(), d1.vs(), nin, nout, rows);
            p.vk.silu_mul(f.vs(), gi.vs(), ui.vs(), rows * nin);
            p.vk.matmul(quant::GGML_TYPE_Q4_0, w4i.vs(), f.vs(), d2.vs(), nin, nout, rows);
            p.cpu.rms_norm_rows(h.cs(), si.cs(), wni.cs(), rows, nin, nin, 1e-6f);
            p.cpu.silu_mul(f.cs(), gi.cs(), ui.cs(), rows * nin);
            std::vector<float> hc(rows * nin), fc(rows * nin);
            p.cpu.read(*h.c, 0, hc.data(), hc.size() * sizeof(float));
            p.cpu.read(*f.c, 0, fc.data(), fc.size() * sizeof(float));
            const auto hr = fed(hc, quant::GGML_TYPE_Q8_0, integer_dot, false), fr = fed(fc, quant::GGML_TYPE_Q4_0, integer_dot, false);   // Q8_0 from the norm and Q4_0 from the SiLU, each on the twin its kernel reads
            Pair::In hri = p.in(hr), fri = p.in(fr);
            p.cpu.matmul(quant::GGML_TYPE_Q8_0, wqi.cs(), hri.cs(), d1.cs(), nin, nout, rows, {}, backend::Dtype::f32);
            p.cpu.matmul(quant::GGML_TYPE_Q4_0, w4i.cs(), fri.cs(), d2.cs(), nin, nout, rows, {}, backend::Dtype::f32);
            auto r1 = p.results(d1), r2 = p.results(d2);
            values += close(r1.first, r1.second, tol_of(quant::GGML_TYPE_Q8_0, false), "matmul from the norm's twin differs beyond its bound");
            values += close(r2.first, r2.second, tol_of(quant::GGML_TYPE_Q4_0, false), "matmul from the SiLU's twin differs beyond its bound");
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
            try { p.vk.matmul(quant::GGML_TYPE_Q8_0, wqi.vs(), xi.vs(), d.vs(), nin, nout, 2, {merged, 2}); }
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
                p.vk.matmul(quant::GGML_TYPE_Q8_0, wqi.vs(), xi.vs(), sep[i].vs(), nin, rows[i], nbatch);
            p.vk.matmul_group({{quant::GGML_TYPE_Q8_0, wqi.vs(), grp[0].vs(), rows[0]},
                               {quant::GGML_TYPE_Q8_0, wqi.vs(), grp[1].vs(), rows[1]},
                               {quant::GGML_TYPE_Q8_0, wqi.vs(), grp[2].vs(), rows[2]}},
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
            const auto xr = fed(x, quant::GGML_TYPE_Q8_0, prof.prefer_integer_dot, true);
            if (!tiled) continue;
            Pair::In xi = p.in(x), xri = p.in(xr);
            const size_t rows[3] = {nout, 5, 33};
            const uint32_t types[3] = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q8_0};
            Pair::Out grp[3] = {p.out(nbatch * rows[0]), p.out(nbatch * rows[1]), p.out(nbatch * rows[2])};
            p.vk.matmul_group({{types[0], wqi.vs(), grp[0].vs(), rows[0]},
                               {types[1], w4i.vs(), grp[1].vs(), rows[1]},
                               {types[2], wqi.vs(), grp[2].vs(), rows[2]}},
                              xi.vs(), nin, nbatch);
            for (int i = 0; i < 3; ++i) {
                p.cpu.matmul(types[i], (i == 1 ? w4i : wqi).cs(), xri.cs(), grp[i].cs(), nin, rows[i], nbatch, {}, backend::Dtype::f32);
                auto r = p.results(grp[i]);
                try {
                    values += close(r.first, r.second, tol_of(types[i], true), "grouped tile projections differ beyond their bound");
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
            const uint32_t types[2] = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0};
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
    for (const Timed& t : {Timed{quant::GGML_TYPE_Q8_0, "Q8_0", 1024, 1024}, {quant::GGML_TYPE_Q8_0, "Q8_0", 1024, 2048},
                           {quant::GGML_TYPE_Q8_0, "Q8_0", 1024, 3072}, {quant::GGML_TYPE_Q8_0, "Q8_0", 3072, 1024},
                           {quant::GGML_TYPE_Q8_0, "Q8_0", 4096, 4096}, {quant::GGML_TYPE_Q8_0, "Q8_0", 4096, 12288},
                           {quant::GGML_TYPE_Q8_0, "Q8_0", 12288, 4096},
                           {quant::GGML_TYPE_Q4_0, "Q4_0", 1024, 3072}, {quant::GGML_TYPE_Q4_0, "Q4_0", 4096, 12288},
                           {quant::GGML_TYPE_Q4_1, "Q4_1", 3072, 1024}, {quant::GGML_TYPE_Q4_1, "Q4_1", 12288, 4096},
                           {quant::GGML_TYPE_Q6_K, "Q6_K", 1024, 3072}, {quant::GGML_TYPE_Q6_K, "Q6_K", 4096, 12288},
                           {quant::GGML_TYPE_Q6_K, "Q6_K", 1024, 151936},
                           {quant::GGML_TYPE_Q4_K, "Q4_K", 1024, 3072}, {quant::GGML_TYPE_Q4_K, "Q4_K", 4096, 12288},
                           {quant::GGML_TYPE_Q5_K, "Q5_K", 1024, 3072}, {quant::GGML_TYPE_Q5_K, "Q5_K", 4096, 12288},
                           // An 8B feed-forward down projection, the shape per-operation benchmarks of other runtimes report.
                           {quant::GGML_TYPE_Q8_0, "Q8_0", 14336, 4096}, {quant::GGML_TYPE_Q4_K, "Q4_K", 14336, 4096},
                           {quant::GGML_TYPE_Q6_K, "Q6_K", 14336, 4096}}) {
        const size_t nin = t.nin, nout = t.nout;
        const size_t block = t.type >= quant::GGML_TYPE_Q4_K ? quant::Q6_K_BLOCK : 32;
        const size_t bytes = t.type == quant::GGML_TYPE_Q8_0 ? quant::Q8_0_TYPESIZE
                           : t.type == quant::GGML_TYPE_Q4_0 ? quant::Q4_0_TYPESIZE
                           : t.type == quant::GGML_TYPE_Q4_1 ? quant::Q4_1_TYPESIZE
                           : t.type == quant::GGML_TYPE_Q4_K ? quant::Q4_K_TYPESIZE
                           : t.type == quant::GGML_TYPE_Q5_K ? quant::Q5_K_TYPESIZE : quant::Q6_K_TYPESIZE;
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
    for (const Timed& t : {Timed{quant::GGML_TYPE_Q8_0, "Q8_0", 14336, 4096}, {quant::GGML_TYPE_Q4_K, "Q4_K", 14336, 4096},
                           {quant::GGML_TYPE_Q6_K, "Q6_K", 14336, 4096}}) {
        const size_t nin = t.nin, nout = t.nout, nbatch = 512;
        const size_t block = t.type == quant::GGML_TYPE_Q8_0 ? 32 : quant::Q6_K_BLOCK;
        const size_t bytes = t.type == quant::GGML_TYPE_Q8_0 ? quant::Q8_0_TYPESIZE
                           : t.type == quant::GGML_TYPE_Q4_K ? quant::Q4_K_TYPESIZE : quant::Q6_K_TYPESIZE;
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
        time("norm_rope_partial 1 row 16 heads", [&] {
            vk.norm_rope_partial({xb.get(), 0}, {xb.get(), 0}, 1, 16 * 128, 128, 16, 128, 128, {hwb.get(), 0}, 1e-6f, {cb.get(), 0}, {sb.get(), 0}, &pos0);
        });
        time("silu_mul 3072", [&] { vk.silu_mul({xb.get(), 0}, {xb.get(), 0}, {xb.get(), 0}, 3072); });
        time("kv_write 1 row", [&] { vk.kv_write(0, &view, 1, {Kb.get(), 0}, {Vb.get(), 0}); });
    }
    // Decode bandwidth of the row kernel on a Qwen3-8B-sized projection, reported and not asserted: 4096 x 4096 Q8_0 is 17 MiB per column.
    {
        const size_t n = 4096;
        std::vector<uint8_t> wq(n * (n / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 7 + 3);
        const auto x = uniform(n, 13);
        const auto w = vk.adopt(wq.data(), wq.size());
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(n * sizeof(float), backend::Memory::device);
        vk.matmul(quant::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const int iters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul(quant::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
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
        // Stacked expert bytes of a type.
        auto stacked = [&](uint32_t type, size_t in, size_t out, uint32_t seed) { return matrix(type, in, n_expert * out, seed); };
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
            for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1,
                                  quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}) {
                from = type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1 ? prof.moe_tile_from_q4
                     : type == quant::GGML_TYPE_Q4_K ? prof.moe_tile_from_q4k
                     : type == quant::GGML_TYPE_Q5_K ? prof.moe_tile_from_q5k : prof.moe_tile_from;
                // The row kernel reads the 16-bit twin; the tile reads the 16-bit twin where the integer dot takes quantized types, else floats.
                auto fed_rows = [&](const std::vector<float>& v, size_t per_row, size_t per_entry_rows) {
                    std::vector<float> out(v.size());
                    for (size_t i = 0; i < v.size() / per_row; ++i) {
                        const std::vector<float> row(v.begin() + i * per_row, v.begin() + (i + 1) * per_row);
                        const bool t = tiled(i / per_entry_rows);
                        const std::vector<float> got = fed(row, type, integer_dot, t);
                        std::copy(got.begin(), got.end(), out.begin() + i * per_row);
                    }
                    return out;
                };
                // The integer tile has its own accumulation bound.
                bool tile16 = false;
                for (size_t r = 0; r < rows; ++r) {
                    tile16 = tile16 || (tiled(r) && integer_dot && type != quant::GGML_TYPE_F32);
                }
                const double tol = tile16 ? 1e-3 : 1e-4;
                const auto wg = stacked(type, nin, nff, 94), wu = stacked(type, nin, nff, 95), wd = stacked(type, nin, nout, 96);
                Pair::In wgi = p.in(wg.data(), wg.size()), wui = p.in(wu.data(), wu.size()), wdi = p.in(wd.data(), wd.size());
                const auto xr = fed_rows(x, nin, 1), x2r = fed_rows(x2, nin, k);
                Pair::In xri = p.in(xr), x2ri = p.in(x2r);
                try {
                    Pair::Out g = p.out(entries * nff), u = p.out(entries * nff);
                    p.cpu.matmul_experts({{type, wgi.cs(), g.cs(), nff}, {type, wui.cs(), u.cs(), nff}}, xri.cs(), nin, rows, rc, runs, backend::Dtype::f32);
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
                    p.cpu.matmul_experts_add(type, wdi.cs(), x2ri.cs(), y.cs(), nin, nout, rows, rc, runs, backend::Dtype::f32);
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

// Decode columns: a row kernel's builds differ only in how many columns and rows share a weight read, so every column of a call must be, bit for bit, that column computed alone, which takes the narrowest build (docs/VULKAN.md, batch invariance).
// Every row is a generated token's, so every width stays on the row kernels: plain calls of 1 to 64 columns, the residual add and a group of three projections at widths that reach every build and chunk, and routed entries of 1 to 32 tokens against each token alone.
// Each type the row kernels decode at rows 4096 and 1280 wide, the block types also at a row of an odd block count, Q8_0 also 2560 wide, and the output head of the types that keep a 16-bit twin for it.
// At 4096 every lane of a 64-lane subgroup takes the same number of steps, and at 1280 some lanes take 3 and every lane of a 32-lane subgroup 5, so a build that takes steps in pairs also takes a single step after them.
// At 2560 every lane of a Q8_0 build that takes steps in pairs takes two pairs and then a single step, and under the half-block order a lane takes 3 steps or 2.
// 300 outputs leave the last workgroup rows past the end, and the grouped projections of 37 and 129 rows a subgroup that holds rows past the end.
size_t check_decode_columns(backend::Backend& vk) {
    const size_t nout = 300, widest = 64;
    // Each width's remainder past the widest Q8_0 decode build, 8, 16 or 32 columns, takes the 1-, 2-, 4-, 8-, 16- or 32-column build, and 18 and 29 a 32-column build's second group in part; past the widest two-row build, 16 columns, the 1-, 2-, 4-, 8- or 16-column build.
    const size_t widths[] = {1, 2, 3, 8, 9, 13, 16, 18, 29, 32, 33, 34, 36, 40, 48, 64};
    const uint32_t f32 = quant::GGML_TYPE_F32, q8 = quant::GGML_TYPE_Q8_0, q40 = quant::GGML_TYPE_Q4_0, q41 = quant::GGML_TYPE_Q4_1;
    const uint32_t q4k = quant::GGML_TYPE_Q4_K, q5k = quant::GGML_TYPE_Q5_K, q6k = quant::GGML_TYPE_Q6_K;
    size_t columns = 0;
    auto floats = [&](const backend::BufferPtr& b, size_t n) {
        std::vector<float> v(n);
        vk.read(*b, 0, v.data(), n * sizeof(float));
        return v;
    };
    // The first n columns of `rows` floats each, against the reference's, bit for bit.
    auto same = [&](const std::vector<float>& got, const std::vector<float>& ref, size_t n, size_t rows, const char* what) {
        for (size_t col = 0; col < n; ++col)
            if (std::memcmp(got.data() + col * rows, ref.data() + col * rows, rows * sizeof(float)) != 0) {
                std::fprintf(stderr, "  decode columns: column %zu of %zu\n", col, n);
                throw std::runtime_error(what);
            }
        columns += n;
    };

    struct Case { uint32_t type; size_t nin; };
    for (const Case& c : {Case{f32, 4096}, Case{f32, 1280}, Case{f32, 224}, Case{q8, 4096}, Case{q8, 2560}, Case{q8, 1280}, Case{q8, 224},
                          Case{q40, 4096}, Case{q40, 1280}, Case{q40, 224}, Case{q41, 4096}, Case{q41, 1280}, Case{q41, 224},
                          Case{q4k, 4096}, Case{q4k, 1280}, Case{q5k, 4096}, Case{q5k, 1280}, Case{q6k, 4096}, Case{q6k, 1280}}) {
        const uint32_t type = c.type;
        const size_t nin = c.nin, rows[3] = {nout, 37, 129};
        const auto x = uniform(widest * nin, 300 + uint32_t(nin)), base = uniform(widest * nout, 301);
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        std::vector<backend::BufferPtr> w;
        for (uint32_t i = 0; i < 3; ++i) {
            const auto bytes = matrix(type, nin, rows[i], 302 + i);
            w.push_back(vk.adopt(bytes.data(), bytes.size()));
        }
        const bool keeps_head = type == q40 || type == q41 || type == q6k;
        for (int head = 0; head <= (keeps_head ? 1 : 0); ++head) {
            try {
                // n columns of X from col0 through projection i into y, as n generated tokens.
                auto product = [&](size_t i, size_t col0, size_t n, backend::Slice y, bool add) {
                    const backend::RowRun decode{n, 1};
                    const backend::CSlice wi{w[i].get(), 0}, xs{xb.get(), col0 * nin};
                    if (add) vk.matmul_add(type, wi, xs, y, nin, rows[i], n, {&decode, 1});
                    else if (head) vk.matmul_logits(type, wi, xs, y, nin, rows[i], n, {&decode, 1});
                    else vk.matmul(type, wi, xs, y, nin, rows[i], n, {&decode, 1});
                };
                // Each column alone, onto its own base column where it adds.
                auto alone = [&](size_t i, bool add) {
                    std::vector<float> out(widest * rows[i]);
                    const auto yb = vk.alloc(rows[i] * sizeof(float));
                    for (size_t col = 0; col < widest; ++col) {
                        if (add) vk.write(*yb, 0, base.data() + col * nout, nout * sizeof(float));
                        product(i, col, 1, {yb.get(), 0}, add);
                        vk.read(*yb, 0, out.data() + col * rows[i], rows[i] * sizeof(float));
                    }
                    return out;
                };
                const auto one = alone(0, false);
                const auto yb = vk.alloc(widest * nout * sizeof(float));
                for (size_t n = 1; n <= widest; ++n) {
                    product(0, 0, n, {yb.get(), 0}, false);
                    same(floats(yb, n * nout), one, n, nout, "a decode column differs from the same column alone");
                }
                if (head) continue;
                const auto one_add = alone(0, true), one1 = alone(1, false), one2 = alone(2, false);
                const std::vector<float>* ones[3] = {&one, &one1, &one2};
                for (size_t n : widths) {
                    vk.write(*yb, 0, base.data(), n * nout * sizeof(float));
                    product(0, 0, n, {yb.get(), 0}, true);
                    same(floats(yb, n * nout), one_add, n, nout, "a decode column's residual add differs from the same column alone");
                    std::vector<backend::BufferPtr> out;
                    for (size_t i = 0; i < 3; ++i) out.push_back(vk.alloc(n * rows[i] * sizeof(float)));
                    const backend::RowRun decode{n, 1};
                    vk.matmul_group({{type, {w[0].get(), 0}, {out[0].get(), 0}, rows[0]}, {type, {w[1].get(), 0}, {out[1].get(), 0}, rows[1]},
                                     {type, {w[2].get(), 0}, {out[2].get(), 0}, rows[2]}},
                                    {xb.get(), 0}, nin, n, {&decode, 1});
                    for (size_t i = 0; i < 3; ++i)
                        same(floats(out[i], n * rows[i]), *ones[i], n, rows[i], "a grouped projection's decode column differs from the same column alone");
                }
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  decode columns: type %u, nin %zu%s\n", type, nin, head ? ", output head" : "");
                throw;
            }
        }
    }

    // Routed: 8 experts, 2 a token, 1 to 32 tokens, so a pass's entries take the one-column path below two an expert and the grouped build from there.
    // Gate and up in one call, and the down projection added into the residual through the combine, each token's against the same token alone.
    // Rows are 4096 and 1280 wide, so a lane adds several blocks of every type and an order that differs shows, as do steps in pairs and a single step after them.
    const size_t n_expert = 8, k = 2, tokens = 32;
    const auto scores = uniform(tokens * n_expert, 310, -3.0f, 3.0f), base = uniform(tokens * nout, 313);
    const auto sb = vk.adopt(scores.data(), scores.size() * sizeof(float));
    const auto ids = vk.alloc(tokens * k * sizeof(float)), wts = vk.alloc(tokens * k * sizeof(float));
    vk.route_experts({sb.get(), 0}, tokens, n_expert, k, true, {ids.get(), 0}, {wts.get(), 0});
    auto routing = [&](size_t first) { return backend::Backend::Routing{{ids.get(), first * k}, {wts.get(), first * k}, k, n_expert}; };
    for (size_t rin : {size_t(4096), size_t(1280)}) {
        const auto x = uniform(tokens * rin, 311), x2 = uniform(tokens * k * rin, 312);
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float)), x2b = vk.adopt(x2.data(), x2.size() * sizeof(float));
        for (uint32_t type : {f32, q8, q40, q41, q4k, q5k, q6k}) {
            try {
                const auto g = matrix(type, rin, n_expert * nout, 314), u = matrix(type, rin, n_expert * nout, 315), d = matrix(type, rin, n_expert * nout, 316);
                const auto gb = vk.adopt(g.data(), g.size()), ub = vk.adopt(u.data(), u.size()), db = vk.adopt(d.data(), d.size());
                // Tokens first .. first + n: gate and up entries, then the down projection's rows added onto their base rows.
                auto routed = [&](size_t first, size_t n) {
                    const backend::RowRun decode{n, 1};
                    const auto go = vk.alloc(n * k * nout * sizeof(float)), uo = vk.alloc(n * k * nout * sizeof(float));
                    const auto yo = vk.alloc(n * nout * sizeof(float));
                    vk.write(*yo, 0, base.data() + first * nout, n * nout * sizeof(float));
                    vk.matmul_experts({{type, {gb.get(), 0}, {go.get(), 0}, nout}, {type, {ub.get(), 0}, {uo.get(), 0}, nout}},
                                      {xb.get(), first * rin}, rin, n, routing(first), {&decode, 1});
                    vk.matmul_experts_add(type, {db.get(), 0}, {x2b.get(), first * k * rin}, {yo.get(), 0}, rin, nout, n, routing(first), {&decode, 1});
                    std::vector<std::vector<float>> out = {floats(go, n * k * nout), floats(uo, n * k * nout), floats(yo, n * nout)};
                    return out;
                };
                std::vector<std::vector<float>> one(3);
                for (size_t t = 0; t < tokens; ++t) {
                    const auto r = routed(t, 1);
                    for (size_t i = 0; i < 3; ++i) one[i].insert(one[i].end(), r[i].begin(), r[i].end());
                }
                for (size_t n = 1; n <= tokens; ++n) {
                    const auto r = routed(0, n);
                    same(r[0], one[0], n * k, nout, "a routed gate entry differs from the same token alone");
                    same(r[1], one[1], n * k, nout, "a routed up entry differs from the same token alone");
                    same(r[2], one[2], n, nout, "a routed down projection's row differs from the same token alone");
                }
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  decode columns: routed type %u, nin %zu\n", type, rin);
                throw;
            }
        }
    }
    return columns;
}

// A kernel's float multiplies and adds, counted over the lines of its disassembly that start with an instruction: multiplies, multiply-adds that round the product first (v_mad_f32, v_mac_f32), fused ones (v_fma, v_fmac, and v_mad_mix_f32, which fuses on gfx906), adds, and adds over lanes shuffled in the same instruction (DPP).
// A multiply by 0x4f7ffffe scales an integer division's reciprocal, which a grouped build divides by more often; neither it nor its recognized denormal normalization is counted.
struct FloatOps {
    size_t mul = 0, mad = 0, fused = 0, add = 0, lane_add = 0;
    unsigned kinds() const {
        return unsigned(mul > 0) | unsigned(mad > 0) << 1 | unsigned(fused > 0) << 2 | unsigned(add > 0) << 3 | unsigned(lane_add > 0) << 4;
    }
    bool operator==(const FloatOps& o) const { return mul == o.mul && mad == o.mad && fused == o.fused && add == o.add && lane_add == o.lane_add; }
};
// Only this captured RADV address-division sequence excludes its two normalizing multiplies.
// Unknown instruction forms or register flows stay counted, so the diagnostic still fails closed.
bool normalized_division(const std::string (&lines)[5]) {
    unsigned a[3], b[2], c[3], d[2], e[2];
    if (std::sscanf(lines[0].c_str(), "v_mul_f32_e32 v%u, v%u, v%u", &a[0], &a[1], &a[2]) != 3 ||
        std::sscanf(lines[1].c_str(), "v_rcp_f32_e32 v%u, v%u", &b[0], &b[1]) != 2 ||
        std::sscanf(lines[2].c_str(), "v_mul_f32_e32 v%u, v%u, v%u", &c[0], &c[1], &c[2]) != 3 ||
        std::sscanf(lines[3].c_str(), "v_mul_f32_e32 v%u, 0x4f7ffffe, v%u", &d[0], &d[1]) != 2 ||
        std::sscanf(lines[4].c_str(), "v_cvt_u32_f32_e32 v%u, v%u", &e[0], &e[1]) != 2) return false;
    return a[0] == a[2] && a[0] != a[1] && b[0] == a[0] && b[1] == a[0] &&
           c[0] == a[1] && c[1] == a[1] && c[2] == a[0] &&
           d[0] == a[1] && d[1] == a[1] && e[0] == a[1] && e[1] == a[1];
}

FloatOps float_ops(const std::string& text) {
    FloatOps n;
    std::string recent[5];
    for (size_t from = 0; from < text.size();) {
        size_t eol = text.find('\n', from);
        if (eol == std::string::npos) eol = text.size();
        const size_t at = text.find_first_not_of(" \t", from);
        for (size_t i = 0; i < 4; ++i) recent[i] = recent[i + 1];
        recent[4] = at < eol ? text.substr(at, eol - at) : "";
        if (normalized_division(recent)) n.mul -= 2;
        if (at < eol && text.compare(at, 2, "v_") == 0) {
            size_t end = at;
            while (end < eol && (std::isalnum((unsigned char)text[end]) || text[end] == '_')) ++end;
            const std::string op = text.substr(at, end - at);
            std::string line = text.substr(at, eol - at);
            for (char& ch : line) ch = char(std::tolower((unsigned char)ch));
            auto starts = [&](const char* p) { return op.rfind(p, 0) == 0; };
            if (((starts("v_fma") || starts("v_fmac")) && op.find("f32") != std::string::npos) || starts("v_mad_mix_f32")) ++n.fused;
            else if (starts("v_mad_f32") || starts("v_mac_f32") || starts("v_mad_legacy_f32") || starts("v_mac_legacy_f32")) ++n.mad;
            else if (starts("v_mul_f32") || starts("v_mul_legacy_f32")) n.mul += line.find("0x4f7ffffe") == std::string::npos ? 1 : 0;
            else if (starts("v_add_f32")) ++(op.find("_dpp") != std::string::npos ? n.lane_add : n.add);
        }
        from = eol + 1;
    }
    return n;
}

// RADV's preserved reciprocal normalizes its operand before the reciprocal and restores it after.
// These multiplies belong to integer address division, not the row's products.
// A backend made to time its work gives in device_ms the time of every dispatch since its last reading, however many: the server's --timing reads each stage every 32 rounds, which on a large model is many times the 4096 dispatches one query pool holds, and a reading of only the first dispatches showed a busy stage as mostly idle.
size_t check_timing_coverage() {
    backend::BackendPtr tb = backend::make_vulkan_backend(0, true);
    const size_t n = 1024;
    const std::vector<float> x = uniform(n, 7);
    const auto xb = tb->adopt(x.data(), n * sizeof(float));
    const auto yb = tb->alloc(n * sizeof(float), backend::Memory::device);
    tb->device_ms();
    tb->silu_mul({yb.get(), 0}, {xb.get(), 0}, {xb.get(), 0}, n);
    tb->device_ms();
    const size_t per = backend::vulkan_timed_dispatches(*tb);
    require(per >= 1, "a timed dispatch was not timed");
    const size_t calls = 6000;
    for (size_t i = 0; i < calls; ++i) tb->silu_mul({yb.get(), 0}, {xb.get(), 0}, {xb.get(), 0}, n);
    const double ms = tb->device_ms();
    require(backend::vulkan_timed_dispatches(*tb) == calls * per, "device timing missed the dispatches past one query pool");
    require(ms > 0.0, "device timing read no time");
    return calls * per;
}

void check_float_ops() {
    const std::string divide =
        "v_mul_f32_e32 v16, v21, v16\n"
        "v_rcp_f32_e32 v16, v16\n"
        "v_mul_f32_e32 v21, v21, v16\n"
        "v_mul_f32_e32 v21, 0x4f7ffffe, v21\n"
        "v_cvt_u32_f32_e32 v21, v21\n";
    const std::string products = "v_mul_f32_e32 v30, v31, v32\nv_fma_f32 v33, v34, v35, v36\n";
    const FloatOps want{1, 0, 1, 0, 0};
    require(float_ops(products + divide + products) == FloatOps{2, 0, 2, 0, 0},
            "integer address division pollutes the matrix instruction counts");
    require(float_ops(divide + products) == want, "address division hides a following matrix product");
    std::string other = divide;
    other.replace(other.find("v_cvt_u32_f32_e32 v21, v21"), std::strlen("v_cvt_u32_f32_e32 v21, v21"), "v_cvt_u32_f32_e32 v22, v21");
    require(float_ops(other + products) == FloatOps{3, 0, 1, 0, 0}, "an unrecognized division sequence hides float products");
    other = divide;
    other.replace(other.find("v_rcp_f32_e32 v16, v16"), std::strlen("v_rcp_f32_e32 v16, v16"), "v_rcp_f32_e32 v16, v17");
    require(float_ops(other + products) == FloatOps{3, 0, 1, 0, 0}, "a mismatched reciprocal hides float products");
    require(float_ops("v_mul_f32_e32 v1, 0x4f7ffffe, v1\n" + products) == want,
            "plain address division changes matrix instruction counts");
}

// The row kernel builds check_contraction checked, by how.
// Where the driver emits native 16-bit dots for K4, Q6 must keep its centered operands at that width too.
size_t check_q6_dots(const std::vector<std::pair<std::string, std::string>>& representations) {
    bool native = false;
    for (const auto& kr : representations)
        if (kr.first.find("matmul_row_k4_dot") == 0 && kr.second.find("v_dot2_i32_i16") != std::string::npos) native = true;
    if (!native) return 0;
    size_t checked = 0;
    for (const auto& kr : representations) {
        if (kr.first.find("matmul_row_k_dot") != 0 || kr.second.find("v_") == std::string::npos) continue;
        if (kr.second.find("v_dot2_i32_i16") == std::string::npos)
            throw std::runtime_error("Q6 centered operands lost the native 16-bit dot: " + kr.first);
        ++checked;
    }
    return checked;
}

struct ContractionChecks {
    size_t same = 0, whole_columns = 0, two_rows = 0, decode = 0, kinds_only = 0, grouped = 0;
};

// A Q8_0 decode build's float operations as its shape and forms give them (matmul_vec_q8.comp), where its one-column build takes `levels` shuffled adds a reduction and `extra` multiplies other than its products'.
// A product is a multiply and a multiply-add, or a fused one in the one-column build's proportion, once for each step in the code: STEPS steps and, past one, a single step after them, again for the columns of a group past the first half of a build of up to 8 columns, whose copy checks each column, and once more in the half-block order's step, which rows of an even block count take.
// Without the transposed reduction each row and column is reduced as the one-column build reduces one, and takes one plain add, the residual add's.
// With it the build's R rows and C columns take the six levels' pairs, R * C - 1 adds and one for each level past log2(R * C), which the driver may take as shuffled or plain adds, and one residual add for each 64 values.
struct DecodeOps {
    size_t products = 0, mul = 0, lane_add = 0, add = 0, reduce = 0;
    bool tree = false;
};
// A Q8_0 decode build as the backend made it, read from the first line of its representation (vulkan_kernel_representations): the columns and rows a subgroup takes, the steps of weights a lane loads before using any, and its forms.
struct DecodeBuild {
    unsigned cols = 0, rows = 0, steps = 0;
    bool tree = false, half = false;
};
// The value after ` key=` on a representation's first line, or false where the key is missing or its value is not a number.
bool header_field(const std::string& line, const char* key, unsigned& v) {
    const std::string at = std::string(" ") + key + "=";
    const size_t p = line.find(at);
    if (p == std::string::npos || p + at.size() >= line.size() || !std::isdigit((unsigned char)line[p + at.size()])) return false;
    v = unsigned(std::stoul(line.substr(p + at.size())));
    return true;
}
bool decode_build(const std::string& text, DecodeBuild& b) {
    const std::string head = "; q8_decode_build";
    const std::string line = text.substr(0, text.find('\n'));
    if (line.compare(0, head.size(), head) != 0) return false;
    auto field = [&](const char* key, unsigned& v) { return header_field(line, key, v); };
    unsigned tree = 0, half = 0;
    if (!field("cols", b.cols) || !field("rows", b.rows) || !field("steps", b.steps) || !field("tree", tree) || !field("half", half)) return false;
    b.tree = tree != 0;
    b.half = half != 0;
    return b.cols && b.rows && b.steps;
}
// A two-row build of the Q4 and K-quant row kernels, read from the first line of its representation: the columns it holds and the rows a cluster takes.
bool row_build(const std::string& text, unsigned& cols, unsigned& rows) {
    const std::string head = "; row_build";
    const std::string line = text.substr(0, text.find('\n'));
    if (line.compare(0, head.size(), head) != 0) return false;
    return header_field(line, "cols", cols) && header_field(line, "rows", rows) && cols && rows;
}
DecodeOps decode_ops(const DecodeBuild& b, size_t levels, size_t extra) {
    DecodeOps n;
    const size_t group = b.cols < 4 ? b.cols : 4, rc = size_t(b.rows) * b.cols;
    size_t cols = b.cols;
    if (b.cols <= 8)
        for (size_t g = group; g < b.cols; g += group)
            if (2 * g >= b.cols) cols += group;
    // The shader keeps scale * activation scale * integer dot, then adds it, without contraction.
    n.products = b.rows * cols * (b.steps + (b.steps > 1 ? 1 : 0) + (b.half ? 1 : 0));
    n.mul = extra + 2 * n.products;
    n.tree = b.tree;
    if (!b.tree) {
        n.lane_add = rc * levels;
        n.add = rc;
        return n;
    }
    size_t log = 0;
    while ((size_t(1) << log) < rc) ++log;
    for (size_t l = 0; l < 6; ++l) n.reduce += l < log ? rc >> (l + 1) : 1;
    n.reduce += rc > 64 ? rc / 64 : 1;
    return n;
}

// Every build of a row kernel holds its one-column build's float multiply and add counts, and a grouped build its wide build's: a screen on how the driver contracts and reduces a column's products and sums, which sees a change only where it changes the counts.
// Reassociation that keeps the counts shows only in the decode-column check, which is what holds batch invariance (docs/VULKAN.md, batch invariance).
// A build of `matmul_row.comp` holds the one-column build's counts exactly where the driver keeps the column loop rolled, and otherwise those counts and N - 1 copies of one column's, N its columns.
// A kernel's two-row builds (the first line of each one's representation gives its columns and rows) differ from each other by whole copies of one row and column's products, since they compute every column they hold.
// The Q8_0 decode kernel's builds differ in rows, steps, copies of their products and forms (as the first line of each one's representation gives them), so where its one-column build reduces over shuffled adds each build holds the counts its shape and forms give (decode_ops), the transposed reduction only where the one-column build's reduction takes six levels.
// A kernel's builds are named after it: the wide build plain, then `_grouped` and `_<N>col`; a kernel whose driver gives no disassembly is not checked.
ContractionChecks check_contraction(const std::vector<std::pair<std::string, std::string>>& representations) {
    ContractionChecks n;
    std::vector<std::pair<std::string, FloatOps>> ops;
    for (const auto& kr : representations) ops.push_back({kr.first, float_ops(kr.second)});
    auto shape_of = [&](const std::string& name, DecodeBuild& b) {
        for (const auto& kr : representations)
            if (kr.first == name) return decode_build(kr.second, b);
        return false;
    };
    auto find = [&](const std::string& name) -> const FloatOps* {
        for (const auto& o : ops)
            if (o.first == name) return &o.second;
        return nullptr;
    };
    auto ends = [](const std::string& s, const std::string& suffix) {
        return s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    const std::string vec = "matmul_vec_q8", one_suffix = "_1col", grouped_suffix = "_grouped";
    struct TwoRow {
        std::string kernel, name;
        size_t cols;
        unsigned rows;
        FloatOps got, ref;
    };
    std::vector<TwoRow> two_row;
    for (const auto& o : ops) {
        if (!ends(o.first, grouped_suffix)) continue;
        const std::string wide = o.first.substr(0, o.first.size() - grouped_suffix.size());
        const FloatOps* w = find(wide);
        if (!w || wide == vec) continue;
        if (!(o.second == *w)) throw std::runtime_error("a grouped build's float multiplies and adds differ from its wide build's: " + o.first);
        ++n.grouped;
    }
    for (const auto& one : ops) {
        if (!ends(one.first, one_suffix)) continue;
        const std::string kernel = one.first.substr(0, one.first.size() - one_suffix.size());
        const FloatOps& ref = one.second;
        if (!ref.kinds()) continue;
        for (const auto& build : ops) {
            const std::string& b = build.first;
            size_t cols = b == kernel || b == kernel + grouped_suffix ? 8 : 0;
            if (!cols && b != one.first && b.size() > kernel.size() + 4 && b.compare(0, kernel.size() + 1, kernel + "_") == 0 && ends(b, "col")) {
                const std::string digits = b.substr(kernel.size() + 1, b.size() - kernel.size() - 4);
                if (!digits.empty() && digits.find_first_not_of("0123456789") == std::string::npos) cols = std::stoul(digits);
            }
            if (cols < 2) continue;
            const FloatOps& got = build.second;
            auto fail = [&](const char* what) { throw std::runtime_error(std::string(what) + ": " + b); };
            if (got.kinds() != ref.kinds()) fail("a build's float multiply and add kinds differ from its one-column build's");
            if (kernel == vec) {
                if (!ref.add || !ref.lane_add) {
                    ++n.kinds_only;
                    continue;
                }
                DecodeBuild shape_b, one_b;
                if (!shape_of(b, shape_b) || !shape_of(one.first, one_b)) fail("a Q8_0 decode build whose representation does not start with its shape");
                const DecodeBuild *shape = &shape_b, *one_shape = &one_b;
                // The one-column build's own counts give its reduction's shuffled adds and the multiplies beside its products.
                const size_t rc1 = size_t(one_shape->rows) * one_shape->cols, levels = ref.lane_add / rc1;
                const DecodeOps want1 = decode_ops(*one_shape, levels, 0);
                if (one_shape->tree || ref.add != rc1 + want1.products || ref.lane_add != rc1 * levels || ref.mad || ref.fused || ref.mul < want1.mul)
                    fail("the Q8_0 decode kernel's one-column build does not hold the counts its shape gives");
                if (shape->half != one_shape->half) fail("a Q8_0 decode build's order differs from its one-column build's");
                const DecodeOps want = decode_ops(*shape, levels, ref.mul - want1.mul);
                char counts[256];
                std::snprintf(counts, sizeof counts, " (multiplies %zu, multiply-adds %zu, fused %zu, adds %zu, shuffled adds %zu; its shape and forms give %zu, %zu products, %zu adds)",
                              got.mul, got.mad, got.fused, got.add, got.lane_add, want.mul, want.products, want.products + (want.tree ? want.reduce : want.add + want.lane_add));
                const bool reduction = want.tree ? levels == 6 && got.add + got.lane_add == want.reduce + want.products
                                                 : got.add == want.add + want.products && got.lane_add == want.lane_add;
                if (!reduction || got.mad || got.fused || got.mul != want.mul)
                    fail((std::string("a Q8_0 decode build's float multiplies and adds differ from those its shape and forms give") + counts).c_str());
                ++n.decode;
                continue;
            }
            unsigned two_cols = 0, rows = 1;
            for (const auto& kr : representations)
                if (kr.first == b && row_build(kr.second, two_cols, rows) && two_cols != cols) fail("a two-row build whose shape gives other columns than its name");
            if (rows > 1) {
                two_row.push_back({kernel, b, cols, rows, got, ref});
                continue;
            }
            if (got == ref) {
                ++n.same;
                continue;
            }
            // Each kind the one-column build's and a whole number of one column's copies, one column's no more than the one-column build holds.
            const size_t g[5] = {got.mul, got.mad, got.fused, got.add, got.lane_add}, r[5] = {ref.mul, ref.mad, ref.fused, ref.add, ref.lane_add};
            for (int i = 0; i < 5; ++i)
                if (g[i] < r[i] || (g[i] - r[i]) % (cols - 1) != 0 || (g[i] - r[i]) / (cols - 1) > r[i])
                    fail("a build's float multiplies and adds are not its one-column build's and whole columns of it");
            ++n.whole_columns;
        }
    }
    // A kernel's two-row builds, which compute every column they hold, differ by whole copies of one row and column's products for each column more, the same in every pair of them, a column's no more than the one-column build holds and in its proportion of fused products.
    std::map<std::string, std::vector<size_t>> column_ops;
    for (size_t i = 0; i < two_row.size(); ++i)
        for (size_t j = 0; j < two_row.size(); ++j) {
            const TwoRow &a = two_row[i], &c = two_row[j];
            if (a.kernel != c.kernel || c.cols <= a.cols) continue;
            auto fail = [&](const char* what) { throw std::runtime_error(std::string(what) + ": " + a.name + " and " + c.name); };
            if (a.rows != c.rows) fail("two-row builds of one kernel with other rows");
            const size_t step = size_t(a.rows) * (c.cols - a.cols);
            const size_t ga[5] = {a.got.mul, a.got.mad, a.got.fused, a.got.add, a.got.lane_add}, gc[5] = {c.got.mul, c.got.mad, c.got.fused, c.got.add, c.got.lane_add};
            const size_t r[5] = {a.ref.mul, a.ref.mad, a.ref.fused, a.ref.add, a.ref.lane_add};
            size_t per[5];
            for (int k = 0; k < 5; ++k) {
                if (gc[k] < ga[k] || (gc[k] - ga[k]) % step != 0 || (gc[k] - ga[k]) / step > r[k])
                    fail("two-row builds' float multiplies and adds differ by other than whole copies of a row and column's products");
                per[k] = (gc[k] - ga[k]) / step;
            }
            if (per[1] * r[2] != per[2] * r[1]) fail("a two-row build fuses a column's products in another proportion than its one-column build");
            auto seen = column_ops.emplace(a.kernel, std::vector<size_t>(per, per + 5));
            if (!std::equal(per, per + 5, seen.first->second.begin())) fail("two-row builds of one kernel differ by other products a column in another pair");
            ++n.two_rows;
        }
    return n;
}

// Captured Q8 one- and two-column builds: two multiplies and one add per product.
// The remaining adds are the lane reduction and output accumulation.
void check_decode_contraction() {
    auto code = [](const char* shape, size_t mul, size_t add, size_t lane_add) {
        std::string out = std::string("; q8_decode_build ") + shape + "\n";
        for (size_t i = 0; i < mul; ++i) out += "v_mul_f32_e32 v0, v1, v2\n";
        for (size_t i = 0; i < add; ++i) out += "v_add_f32_e32 v0, v1, v2\n";
        for (size_t i = 0; i < lane_add; ++i) out += "v_add_f32_dpp v0, v1, v2\n";
        return out;
    };
    const std::string one = code("cols=1 rows=2 steps=1 tree=0 half=1", 8, 6, 12);
    const std::string two = code("cols=2 rows=2 steps=1 tree=1 half=1", 16, 11, 5);
    require(check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_2col", two}}).decode == 1,
            "Q8 separate products fail their captured shape counts");
    for (const std::string& wrong : {code("cols=2 rows=2 steps=1 tree=1 half=1", 15, 11, 5),
                                     code("cols=2 rows=2 steps=1 tree=1 half=1", 16, 10, 5),
                                     two + "v_fma_f32 v0, v1, v2, v3\n"}) {
        bool refused = false;
        try { check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_2col", wrong}}); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "Q8 changed arithmetic escaped its shape counts");
    }
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
// Names come from the expected kernels, not the private dispatch table; numeric and row-class checks run separately.
size_t check_weight_dispatch() {
    const auto owner = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *owner;
    const backend::DeviceProfile p = backend::vulkan_device_profile(vk);
    struct Expected { uint32_t type; const char* row; const char* integer_tile; };
    const Expected cases[] = {{0, "matmul_row_f32", ""}, {8, "matmul_row_q8w", "matmul_tile_q8"},
        {2, "matmul_row_q4", "matmul_tile_q"}, {3, "matmul_row_q4", "matmul_tile_q"},
        {12, "matmul_row_k4", "matmul_tile_q"}, {13, "matmul_row_k5", "matmul_tile_q"},
        {14, "matmul_row_k", "matmul_tile_q6"}, {39, "matmul_row_mxfp4", "matmul_tile_q8mx"}};
    size_t checks = 0;
    for (const Expected& e : cases) {
        if (e.type == 39 && !vk.supports_type(e.type)) continue;
        require(vk.supports_type(e.type), "an implemented weight type lost device support");
        for (size_t nin : {size_t(96), size_t(256), size_t(4096)}) {
            if (nin == 96 && e.type != 2 && e.type != 8) continue;
            const size_t nout = 3;
            const auto bytes = matrix(e.type, nin, nout, 401);
            const auto xf = uniform(nin, 402);
            const uint32_t id = 0;
            const auto w = vk.adopt(bytes.data(), bytes.size()), x = vk.adopt(xf.data(), xf.size() * sizeof(float));
            const auto ids = vk.adopt(&id, sizeof(id)), y = vk.alloc(nout * sizeof(float));
            for (bool routed : {false, true}) {
                const bool fast = e.type == 0 || e.type == 8;
                const size_t dense_from = fast ? (nin < p.tile_narrow_nin ? p.tile_from_8bit_narrow : p.tile_from_8bit)
                                              : (nin < p.tile_narrow_nin ? p.tile_from_other_narrow : p.tile_from_other);
                const size_t from = !routed ? dense_from : e.type == 2 || e.type == 3 ? p.moe_tile_from_q4
                                  : e.type == 12 ? p.moe_tile_from_q4k : e.type == 13 ? p.moe_tile_from_q5k : p.moe_tile_from;
                for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::f32, backend::Dtype::bf16}) {
                    for (size_t extent : {size_t(1), from - 1, from}) {
                        const bool tile = dtype == backend::Dtype::bf16 || extent >= from;
                        const bool integer = dtype == backend::Dtype::f16 && p.prefer_integer_dot && e.type != 0 && !(routed && e.type == 39);
                        std::string expected;
                        if (tile) {
                            expected = integer ? e.integer_tile : e.type == 39 ? "matmul_tile_mxfp4" : "matmul_tile";
                            if (dtype == backend::Dtype::bf16) expected += "_bf16";
                        } else if (dtype == backend::Dtype::f32 && e.type != 0) {
                            expected = e.type == 39 ? "matmul_row_mxfp4_float_x" : "matmul_row_float_x";
                        } else {
                            expected = e.type == 8 && nin == 96 ? "matmul_row" : e.row;
                            if (e.type == 8 && p.prefer_integer_dot) expected = "matmul_vec_q8";
                            else if ((e.type == 39 && p.mxfp4_integer_dot) ||
                                     (p.prefer_integer_dot && e.type != 0 && e.type != 8 && e.type != 39)) expected += "_dot";
                        }
                        (void)backend::vulkan_kernel_times(vk);
                        const backend::RowRun run{1, extent};
                        if (routed) {
                            const backend::Backend::Routing routing{{ids.get(), 0}, {}, 1, 1};
                            vk.matmul_experts({{e.type, {w.get(), 0}, {y.get(), 0}, nout}}, {x.get(), 0}, nin, 1, routing, {&run, 1}, dtype);
                        } else {
                            vk.matmul(e.type, {w.get(), 0}, {x.get(), 0}, {y.get(), 0}, nin, nout, 1, {&run, 1}, dtype);
                        }
                        const auto times = backend::vulkan_kernel_times(vk);
                        if (times.empty() && !checks) {
                            vk.sync();
                            std::cout << "backend-vulkan: dispatch names unavailable without device timestamps\n";
                            return 0;
                        }
                        size_t products = 0;
                        for (const auto& entry : times) {
                            if (entry.first.rfind("matmul_", 0) != 0 || entry.first.rfind("matmul_reduce", 0) == 0) continue;
                            require(entry.first == expected || entry.first == expected + "_1col" || entry.first == expected + "_small",
                                    ("weight dispatch expected " + expected + ", got " + entry.first).c_str());
                            ++products;
                        }
                        require(products == 1, "weight dispatch did not witness exactly one product kernel");
                        ++checks;
                    }
                }
            }
        }
    }
    for (uint32_t type : {1u, 4u, 10u, 30u, 42u, 43u, UINT32_MAX}) {
        require(!vk.supports_type(type), "metadata-only or unknown weight type acquired a kernel");
        const std::string expected = "vulkan: unsupported matrix type " + std::to_string(type) +
                                     " (docs/VULKAN.md lists the types the kernels decode)";
        bool refused = false;
        try { vk.matmul(type, {}, {}, {}, 256, 1, 1); }
        catch (const std::runtime_error& e) { refused = e.what() == expected; }
        require(refused, "unsupported weight type lost its early refusal text");
        ++checks;
    }
    return checks;
}

size_t check_refusals(backend::Backend& vk) {
    require(!vk.implements(backend::Op::mixed_experts), "Vulkan advertises unsupported mixed routed projections");
    const backend::DeviceProfile prof = backend::vulkan_device_profile(vk);
    const bool integer_dot = prof.prefer_integer_dot;
    const uint32_t q8 = quant::GGML_TYPE_Q8_0, f32 = quant::GGML_TYPE_F32, f16 = 1;   // F16 has no kernel
    const size_t nin = 64, nout = 8, rows = 3, n_expert = 4, k = 2, entries = rows * k, nrows = 4, partial = 48;
    const size_t row_bytes = nin / quant::Q8_0_BLOCK * quant::Q8_0_TYPESIZE;
    auto quantized = [&](size_t n, uint32_t seed) {
        const auto f = uniform(n * nin, seed);
        std::vector<uint8_t> q(n * row_bytes);
        for (size_t r = 0; r < n; ++r) quant::quantize_row_q8_0(f.data() + r * nin, q.data() + r * row_bytes, nin / quant::Q8_0_BLOCK);
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
        const bool tile = rows >= backend::tile_from_for(prof, true, nin);
        const auto xr = fed(xf, q8, integer_dot, tile);
        const auto cw = cpu.adopt(wq.data(), wq.size()), cx = cpu.adopt(xr.data(), xr.size() * sizeof(float));
        const auto cy = cpu.alloc(rows * nout * sizeof(float), backend::Memory::device);
        cpu.matmul(q8, {cw.get(), 0}, {cx.get(), 0}, {cy.get(), 0}, nin, nout, rows, {}, backend::Dtype::f32);
        std::vector<float> ref(rows * nout);
        cpu.read(*cy, 0, ref.data(), ref.size() * sizeof(float));
        close(ref, expected, 1e-4, "the call run after refusals differs from the CPU beyond its bound");
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

// The qwen35 layers' ops (docs/QWEN35.md) against the CPU backend, and the device's own bit-for-bit rules for them.
// The recurrence sums a column in another order than the CPU, so its rows and states meet the CPU's within a bound; on the device a sequence gives the same bits alone, beside others, in any view order and cut into passes of any length.
namespace q35 {
using backend::BufferPtr;
using backend::StateShape;
using backend::StateView;

struct Seq {
    size_t length, src, dst, nq;
};
struct Inputs {
    std::vector<float> x, w, alpha, b, a, dt_bias;
};
struct Run {
    std::vector<float> conv, delta;
    std::vector<std::vector<float>> slots;
};

Inputs inputs(uint32_t seed, const StateShape& sh, size_t rows) {
    Inputs in;
    in.x = uniform(rows * sh.channels(), seed, -1.0f, 1.0f);
    in.w = uniform(sh.channels() * backend::kConvTaps, seed + 1, -0.8f, 0.8f);
    in.alpha = uniform(rows * sh.v_heads, seed + 2, -3.0f, 3.0f);
    in.b = uniform(rows * sh.v_heads, seed + 3, -4.0f, 4.0f);
    in.a = uniform(sh.v_heads, seed + 4, -2.0f, -0.05f);
    in.dt_bias = uniform(sh.v_heads, seed + 5, -2.0f, 2.0f);
    return in;
}

std::vector<float> read_floats(backend::Backend& b, const backend::Buffer& buf, size_t n, size_t offset = 0) {
    std::vector<float> v(n);
    if (n) b.read(buf, offset * sizeof(float), v.data(), n * sizeof(float));
    return v;
}

BufferPtr floats_on(backend::Backend& b, const std::vector<float>& v) {
    BufferPtr buf = b.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
    if (!v.empty()) b.write(*buf, 0, v.data(), v.size() * sizeof(float));
    return buf;
}

// One conv and one delta rule call over the views, on layer 1 of a two-layer storage whose slots start as `start`.
Run run(backend::Backend& b, const StateShape& sh, const std::vector<Seq>& seqs, const Inputs& in,
        const std::vector<std::vector<float>>& start) {
    const size_t layer = 1, n = sh.slot_floats();
    auto storage = b.state_alloc(2, start.size(), sh);
    for (size_t s = 0; s < start.size(); ++s) b.write(storage->layer(layer), s * n * sizeof(float), start[s].data(), n * sizeof(float));
    std::vector<StateView> views;
    size_t rows = 0;
    for (const Seq& q : seqs) {
        views.push_back({storage.get(), q.src, q.dst, q.length, q.nq});
        rows += q.nq;
    }
    const size_t C = sh.channels(), Hv = sh.v_heads;
    BufferPtr x = floats_on(b, in.x), w = floats_on(b, in.w), alpha = floats_on(b, in.alpha), bb = floats_on(b, in.b);
    BufferPtr a = floats_on(b, in.a), dt = floats_on(b, in.dt_bias);
    BufferPtr u = b.alloc(rows * C * sizeof(float), backend::Memory::device);
    BufferPtr o = b.alloc(rows * Hv * sh.v_dim * sizeof(float), backend::Memory::device);
    b.causal_conv_silu({u.get(), 0}, {x.get(), 0}, {w.get(), 0}, layer, views.data(), views.size());
    b.gated_delta_rule({o.get(), 0}, {u.get(), 0}, {alpha.get(), 0}, {bb.get(), 0}, {a.get(), 0}, {dt.get(), 0}, layer, views.data(), views.size());
    Run r;
    r.conv = read_floats(b, *u, rows * C);
    r.delta = read_floats(b, *o, rows * Hv * sh.v_dim);
    for (size_t s = 0; s < start.size(); ++s) r.slots.push_back(read_floats(b, storage->layer(layer), n, s * n));
    for (size_t s = 0; s < start.size(); ++s) {
        const auto other = read_floats(b, storage->layer(0), n, s * n);
        require(std::all_of(other.begin(), other.end(), [](float f) { return f == 0.0f; }), "a state op wrote another layer");
    }
    return r;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Every pair of extents of one class gives the same bits through each dtype's matrix products and shared attention (row_classes::check).
size_t check_row_classes(backend::Backend& vk) {
    return row_classes::check(vk,
                              {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q4_K,
                               quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_MXFP4},
                              [](uint32_t type, size_t nin, size_t rows, uint32_t seed) { return matrix(type, nin, rows, seed); });
}

// A slot's matrices within the bound, and its carried rows, which are raw rows, bit for bit.
size_t close_slot(const StateShape& sh, const std::vector<float>& cpu, const std::vector<float>& dev, const char* what) {
    const size_t m = sh.v_heads * sh.matrix_floats();
    const std::vector<float> a(cpu.begin(), cpu.begin() + m), b(dev.begin(), dev.begin() + m);
    size_t values = close(a, b, 1e-4, what);
    require(std::memcmp(cpu.data() + m, dev.data() + m, (cpu.size() - m) * sizeof(float)) == 0, "carried conv rows differ from the CPU's");
    return values + cpu.size() - m;
}

// Sequences of every kind beside each other against the CPU: fresh ones on slots of NaN, histories of 1, 2 and 7 tokens whose carried rows before the sequence's start hold NaN, one-token entries and a verify reading one slot and writing another.
size_t check_mix(backend::Backend& vk, backend::CpuBackend& cpu, const StateShape& sh, uint32_t seed) {
    const std::vector<Seq> seqs = {{0, 0, 0, 5}, {1, 1, 1, 3}, {2, 2, 2, 1}, {7, 3, 3, 6}, {0, 4, 4, 1}, {9, 5, 6, 4}, {3, 7, 7, 2}};
    size_t rows = 0;
    for (const Seq& q : seqs) rows += q.nq;
    const Inputs in = inputs(seed, sh, rows);
    std::vector<std::vector<float>> start;
    for (uint32_t s = 0; s < 8; ++s) start.push_back(uniform(sh.slot_floats(), seed + 10 + s, -0.5f, 0.5f));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const size_t C = sh.channels(), first = sh.v_heads * sh.matrix_floats();
    std::vector<std::vector<float>> clean = start;
    clean[0].assign(sh.slot_floats(), 0.0f);
    clean[4].assign(sh.slot_floats(), 0.0f);
    start[0].assign(sh.slot_floats(), nan);
    start[4].assign(sh.slot_floats(), nan);
    for (size_t c = 0; c < C; ++c) {
        start[1][first + c] = start[1][first + C + c] = nan;
        start[2][first + c] = nan;
        clean[1][first + c] = clean[1][first + C + c] = 0.0f;
        clean[2][first + c] = 0.0f;
    }
    const Run d = run(vk, sh, seqs, in, start), z = run(vk, sh, seqs, in, clean), c = run(cpu, sh, seqs, in, clean);
    for (float f : d.conv) require(std::isfinite(f), "a device conv row read a fresh slot or a carried row before the sequence's start");
    for (float f : d.delta) require(std::isfinite(f), "the device delta rule read the slot of a fresh sequence");
    require(same_bits(d.conv, z.conv) && same_bits(d.delta, z.delta), "a fresh slot's contents changed a device result");
    size_t values = close(c.conv, d.conv, 1e-5, "conv differs from the CPU beyond 1e-5");
    values += close(c.delta, d.delta, 1e-4, "delta rule rows differ from the CPU beyond 1e-4");
    for (const Seq& q : seqs) values += close_slot(sh, c.slots[q.dst], d.slots[q.dst], "a delta rule state differs from the CPU beyond 1e-4");
    // A slot only read keeps its bits.
    require(same_bits(d.slots[5], start[5]), "a verify changed its source slot on the device");
    return values;
}

// On the device a sequence's rows and state are the same bits alone, beside others, in reverse view order and cut into passes of 1, 2 and 3 rows that carry the state in their slot (docs/QWEN35.md, Row classes).
// The calls of all four take the delta rule's 16-token build, as the 40-row view asks for it, and a sequence alone the short build at 8 and 5 rows and in every pass, the 16-token one at 9 and 40, so a column computes the same in both builds, at both sides of the edge.
size_t check_invariance(backend::Backend& vk, const StateShape& sh, uint32_t seed) {
    const std::vector<Seq> seqs = {{0, 0, 0, 9}, {4, 1, 1, 8}, {2, 2, 3, 5}, {6, 4, 4, 40}};
    size_t rows = 0;
    for (const Seq& q : seqs) rows += q.nq;
    const Inputs in = inputs(seed, sh, rows);
    std::vector<std::vector<float>> start;
    for (uint32_t s = 0; s < 5; ++s) start.push_back(uniform(sh.slot_floats(), seed + 20 + s, -0.5f, 0.5f));
    const size_t C = sh.channels(), Hv = sh.v_heads, Dv = sh.v_dim;
    const Run one = run(vk, sh, seqs, in, start);
    std::vector<size_t> first(seqs.size(), 0);
    for (size_t i = 1; i < seqs.size(); ++i) first[i] = first[i - 1] + seqs[i - 1].nq;
    auto slice = [](const std::vector<float>& v, size_t width, size_t r0, size_t n) {
        return std::vector<float>(v.begin() + r0 * width, v.begin() + (r0 + n) * width);
    };
    auto rows_of = [&](size_t vi, size_t r0, size_t n) {
        Inputs part = in;
        part.x = slice(in.x, C, first[vi] + r0, n);
        part.alpha = slice(in.alpha, Hv, first[vi] + r0, n);
        part.b = slice(in.b, Hv, first[vi] + r0, n);
        return part;
    };
    size_t runs = 1;
    {
        std::vector<Seq> rev(seqs.rbegin(), seqs.rend());
        Inputs part = in;
        part.x.clear();
        part.alpha.clear();
        part.b.clear();
        for (size_t k = seqs.size(); k-- > 0;) {
            const Inputs q = rows_of(k, 0, seqs[k].nq);
            part.x.insert(part.x.end(), q.x.begin(), q.x.end());
            part.alpha.insert(part.alpha.end(), q.alpha.begin(), q.alpha.end());
            part.b.insert(part.b.end(), q.b.begin(), q.b.end());
        }
        const Run r = run(vk, sh, rev, part, start);
        size_t at = 0;
        for (size_t k = seqs.size(); k-- > 0;) {
            require(same_bits(slice(r.conv, C, at, seqs[k].nq), slice(one.conv, C, first[k], seqs[k].nq)) &&
                        same_bits(slice(r.delta, Hv * Dv, at, seqs[k].nq), slice(one.delta, Hv * Dv, first[k], seqs[k].nq)),
                    "the view order changed a device result");
            at += seqs[k].nq;
        }
        for (size_t s = 0; s < start.size(); ++s) require(same_bits(r.slots[s], one.slots[s]), "the view order changed a device state");
        ++runs;
    }
    for (size_t vi = 0; vi < seqs.size(); ++vi)
        for (size_t pass : {size_t(0), size_t(1), size_t(2), size_t(3)}) {
            const Seq& q = seqs[vi];
            std::vector<std::vector<float>> slots = start;
            std::vector<float> conv, delta;
            size_t done = 0;
            while (done < q.nq) {
                const size_t n = pass ? std::min(pass, q.nq - done) : q.nq;
                const Seq step = {q.length + done, done ? q.dst : q.src, q.dst, n};
                const Run r = run(vk, sh, {step}, rows_of(vi, done, n), slots);
                conv.insert(conv.end(), r.conv.begin(), r.conv.end());
                delta.insert(delta.end(), r.delta.begin(), r.delta.end());
                slots = r.slots;
                done += n;
                ++runs;
            }
            require(same_bits(conv, slice(one.conv, C, first[vi], q.nq)) && same_bits(delta, slice(one.delta, Hv * Dv, first[vi], q.nq)),
                    "cutting a sequence into device passes changed its rows");
            require(same_bits(slots[q.dst], one.slots[q.dst]), "cutting a sequence into device passes changed its state");
        }
    return runs;
}

// A decay below 2^-126 is 0 and one just above it is kept, on a state whose decayed values stay normal, so the check holds under any denormal handling: with beta 0 the token writes nothing and the state left is the decayed one.
void check_decay_flush(backend::Backend& vk) {
    const StateShape sh = {1, 1, 8, 8};
    const std::vector<Seq> seq = {{5, 0, 0, 1}};
    for (float bias : {88.0f, 87.0f}) {
        Inputs in = inputs(61, sh, 1);
        in.a = {-1.0f};
        in.alpha = {0.0f};
        in.dt_bias = {bias};
        in.b = {-200.0f};
        std::vector<std::vector<float>> start = {uniform(sh.slot_floats(), 62, -0.5f, 0.5f)};
        const auto magnitude = uniform(sh.matrix_floats(), 63, 1.0f, 2.0f);
        for (size_t i = 0; i < sh.matrix_floats(); ++i) start[0][i] = i % 2 ? -magnitude[i] : magnitude[i];
        const Run r = run(vk, sh, seq, in, start);
        const float kept = std::exp(-bias);
        for (size_t i = 0; i < sh.matrix_floats(); ++i) {
            if (bias == 88.0f) {
                require(r.slots[0][i] == 0.0f, "a decay of exp(-88) was not flushed to 0 on the device");
                continue;
            }
            const float want = start[0][i] * kept;
            require(std::fpclassify(r.slots[0][i]) == FP_NORMAL, "a decayed state value below the normal range on the device");
            // The device's exp is exp2 of g log2(e), whose rounded exponent puts about |g| units of 2^-24 into the result.
            require(std::fabs((double)r.slots[0][i] - want) <= 1e-5 * std::fabs((double)want), "a decay of exp(-87) was not kept on the device");
        }
    }
}

// state_alloc zero-fills every slot of every layer on the device, and state_copy copies one slot in every layer and leaves the others.
size_t check_storage(backend::Backend& vk) {
    const StateShape sh = {2, 6, 12, 10};
    const size_t layers = 3, slots = 4, n = sh.slot_floats();
    auto s = vk.state_alloc(layers, slots, sh);
    require(s->layers() == layers && s->slots() == slots, "device state storage counts");
    std::vector<std::vector<std::vector<float>>> held(layers);
    for (size_t l = 0; l < layers; ++l) {
        require(s->layer(l).size() >= slots * n * sizeof(float), "a device state layer's size");
        const auto all = read_floats(vk, s->layer(l), slots * n);
        require(std::all_of(all.begin(), all.end(), [](float f) { return f == 0.0f; }), "state_alloc left a device slot unzeroed");
        for (size_t k = 0; k < slots; ++k) {
            held[l].push_back(uniform(n, 70 + (uint32_t)(l * slots + k)));
            vk.write(s->layer(l), k * n * sizeof(float), held[l][k].data(), n * sizeof(float));
        }
    }
    vk.state_copy(*s, 3, 1);
    vk.state_copy(*s, 2, 2);
    for (size_t l = 0; l < layers; ++l)
        for (size_t k = 0; k < slots; ++k)
            require(same_bits(read_floats(vk, s->layer(l), n, k * n), held[l][k == 3 ? 1 : k]), "state_copy moved the wrong slot on the device");
    bool refused = false;
    try { vk.state_copy(*s, slots, 0); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "state_copy took a slot outside the device storage");
    return layers * slots;
}

// The gated norm against the CPU, in place too, heads near 1e-4 holding eps.
size_t check_gated_norm(Pair& p, size_t heads, size_t dim) {
    const size_t rows = 23, n = rows * heads * dim;
    auto x = uniform(n, 80, -2.0f, 2.0f);
    const auto z = uniform(n, 81, -6.0f, 6.0f), w = uniform(dim, 82, 0.5f, 1.5f);
    for (size_t i = 0; i < dim; ++i) x[(5 * heads + 1) * dim + i] *= 1e-4f;
    Pair::In xi = p.in(x), zi = p.in(z), wi = p.in(w);
    Pair::Out d = p.out(n);
    p.cpu.gated_rms_norm(d.cs(), xi.cs(), zi.cs(), wi.cs(), rows, heads, dim, 1e-6f);
    p.vk.gated_rms_norm(d.vs(), xi.vs(), zi.vs(), wi.vs(), rows, heads, dim, 1e-6f);
    auto r = p.results(d);
    size_t values = close(r.first, r.second, 1e-5, "gated_rms_norm differs beyond 1e-5");
    Pair::Out ip = p.out(n);
    p.vk.write(*ip.v, 0, x.data(), n * sizeof(float));
    p.vk.gated_rms_norm(ip.vs(), ip.vs(), zi.vs(), wi.vs(), rows, heads, dim, 1e-6f);
    std::vector<float> in_place(n);
    p.vk.read(*ip.v, 0, in_place.data(), n * sizeof(float));
    require(same_bits(in_place, r.second), "gated_rms_norm in place differs on the device");
    return values;
}

// sigmoid_mul as the output gate, each head's gate read between its q rows, and as a scale of one value per row, against the CPU; in place too.
size_t check_sigmoid_mul(Pair& p, size_t heads, size_t dim) {
    const size_t rows = 21;
    const auto q = uniform(rows * heads * 2 * dim, 83, -8.0f, 8.0f), x = uniform(rows * heads * dim, 84, -3.0f, 3.0f);
    const auto per_row = uniform(rows, 85, -8.0f, 8.0f);
    Pair::In qi = p.in(q), xi = p.in(x), pi = p.in(per_row);
    size_t values = 0;
    for (int mode = 0; mode < 2; ++mode) {
        const size_t h = mode ? heads * dim : heads, dd = mode ? 1 : dim, gs = mode ? 1 : heads * 2 * dim, ghs = mode ? 0 : 2 * dim;
        const backend::CSlice gc = mode ? pi.cs() : backend::CSlice{qi.c.get(), dim}, gv = mode ? pi.vs() : backend::CSlice{qi.v.get(), dim};
        Pair::Out d = p.out(x.size());
        p.cpu.sigmoid_mul(d.cs(), xi.cs(), gc, rows, h, dd, gs, ghs);
        p.vk.sigmoid_mul(d.vs(), xi.vs(), gv, rows, h, dd, gs, ghs);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-6, "sigmoid_mul differs beyond 1e-6");
        Pair::Out ip = p.out(x.size());
        p.vk.write(*ip.v, 0, x.data(), x.size() * sizeof(float));
        p.vk.sigmoid_mul(ip.vs(), ip.vs(), gv, rows, h, dd, gs, ghs);
        std::vector<float> in_place(x.size());
        p.vk.read(*ip.v, 0, in_place.data(), x.size() * sizeof(float));
        require(same_bits(in_place, r.second), "sigmoid_mul in place differs on the device");
    }
    return values;
}

// Gated attention's tail as the layer runs it at head width 256: attention over a history, the output gated in place by sigmoid_mul, then the output projection added to a residual.
// The attention writes its output's copy for a matmul, which the gate must replace with the gated output's; the projection runs on the row kernel for decode rows and on the tile for a prompt's, and the reference is fed the activations each kernel reads.
size_t check_gated_attention(Pair& p, bool integer_dot) {
    const backend::DeviceProfile prof = backend::vulkan_device_profile(p.vk);
    const int n_head = 12, n_head_kv = 2, head_dim = 256;
    const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim, hist = 90, nout = 72;
    size_t values = 0;
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K}) {
        const bool eight = type == quant::GGML_TYPE_Q8_0;
        const size_t from = backend::tile_from_for(prof, eight, qw);
        // Three decode rows, then a prompt's rows past the tile threshold.
        for (size_t rows : {size_t(3), std::max<size_t>(from, 40)}) {
            const bool tile = rows >= from;
            const std::vector<backend::RowRun> runs = tile ? std::vector<backend::RowRun>{{rows, 512}}
                                                           : std::vector<backend::RowRun>{{1, 1}, {2, 1}, {3, 1}};
            const auto hk = uniform((hist + rows) * kvw, 90), hv = uniform((hist + rows) * kvw, 91);
            const auto qq = uniform(rows * qw, 92, -3.0f, 3.0f), r = uniform(rows * 2 * qw, 93, -4.0f, 4.0f);
            const auto y0 = uniform(rows * nout, 94);
            std::vector<uint8_t> wq;
            const auto wf = uniform(qw * nout, 95);
            if (eight) {
                wq.resize(nout * (qw / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
                for (size_t o = 0; o < nout; ++o)
                    quant::quantize_row_q8_0(wf.data() + o * qw, wq.data() + o * (qw / 32) * quant::Q8_0_TYPESIZE, qw / 32);
            } else {
                wq.resize(nout * (qw / 256) * quant::Q4_K_TYPESIZE);
                for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 61 + 3);
                for (size_t blk = 0; blk < nout * (qw / 256); ++blk) {
                    uint8_t* bb = wq.data() + blk * quant::Q4_K_TYPESIZE;
                    bb[0] = 0x00; bb[1] = 0x14; bb[2] = 0x00; bb[3] = 0x10;
                }
            }
            // The gated output of one backend, and its projection added to y0; `fed` maps the gated output to what the reference's matmul reads.
            auto tail = [&](backend::Backend& b, bool reference, std::vector<float>& gated, std::vector<float>& y) {
                const size_t bt = b.kv_layout().block_tokens;
                auto st = b.kv_alloc(1, n_head_kv, head_dim, 1024);
                infer::BlockPool pool(st->max_blocks());
                infer::KVSequence seq(&pool, bt);
                const auto Kb = b.adopt(hk.data(), hk.size() * sizeof(float));
                const auto Vb = b.adopt(hv.data(), hv.size() * sizeof(float));
                seq.prepare(hist);
                const backend::KVView h = seq.view(st.get());
                b.kv_write(0, &h, 1, {Kb.get(), 0}, {Vb.get(), 0});
                seq.commit();
                seq.prepare(rows);
                backend::KVView view = seq.view(st.get());
                view.extent = tile ? 512 : 1;
                b.kv_write(0, &view, 1, {Kb.get(), hist * kvw}, {Vb.get(), hist * kvw});
                const auto Qb = b.adopt(qq.data(), qq.size() * sizeof(float));
                const auto Rb = b.adopt(r.data(), r.size() * sizeof(float));
                const auto ob = b.alloc(rows * qw * sizeof(float), backend::Memory::device);
                b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                const backend::RowRuns rr{runs.data(), runs.size()};
                b.sigmoid_mul({ob.get(), 0}, {ob.get(), 0}, {Rb.get(), (size_t)head_dim}, rows, (size_t)n_head, (size_t)head_dim, 2 * qw, 2 * (size_t)head_dim, rr);
                gated = read_floats(b, *ob, rows * qw);
                const auto Wb = b.adopt(wq.data(), wq.size());
                const auto Yb = floats_on(b, y0);
                if (reference) {
                    const std::vector<float> xf = fed(gated, type, integer_dot, tile);
                    const auto Xb = floats_on(b, xf);
                    b.matmul_add(type, {Wb.get(), 0}, {Xb.get(), 0}, {Yb.get(), 0}, qw, nout, rows, rr);
                } else {
                    b.matmul_add(type, {Wb.get(), 0}, {ob.get(), 0}, {Yb.get(), 0}, qw, nout, rows, rr);
                }
                y = read_floats(b, *Yb, rows * nout);
            };
            std::vector<float> gc, yc, gv, yv;
            tail(p.cpu, true, gc, yc);
            tail(p.vk, false, gv, yv);
            try {
                values += close(gc, gv, 1e-4, "gated attention at head width 256 differs beyond 1e-4");
                values += close(yc, yv, tile && integer_dot && type != quant::GGML_TYPE_F32 ? 1e-3 : 1e-4, "the output projection of the gated attention differs beyond its bound");
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  gated attention type %u rows %zu %s\n", type, rows, tile ? "tile" : "row kernel");
                throw;
            }
        }
    }
    return values;
}
}  // namespace q35

// The embedded drafter's ops (docs/SPECULATIVE.md, section 7) against the CPU, id for id and bit for bit: argmax_rows over rows of 37 and of 248320 logits, with a tie, an infinity, a NaN first and last, a row of -infinity and a row whose prior id is invalid beside plain rows, with and without prior ids; embed_ids of F32 and Q8_0 tables with valid ids and ids past the table, which write zero rows.
size_t check_drafter_ops(Pair& p) {
    size_t values = 0;
    auto ids_on = [](backend::Backend& b, const std::vector<uint32_t>& ids) {
        backend::BufferPtr buf = b.alloc(ids.size() * sizeof(float), backend::Memory::device);
        b.write(*buf, 0, ids.data(), ids.size() * sizeof(uint32_t));
        return buf;
    };
    auto ids_of = [](backend::Backend& b, const backend::Buffer& buf, size_t n) {
        std::vector<uint32_t> ids(n);
        b.read(buf, 0, ids.data(), n * sizeof(uint32_t));
        return ids;
    };
    for (size_t n : {size_t(37), size_t(248320)}) {
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        const size_t rows = 9;
        std::vector<float> flat = uniform(rows * n, (uint32_t)(300 + n % 7), -6.0f, 6.0f);
        flat[1 * n + 3] = flat[1 * n + n - 2] = 9.5f;   // a tie
        flat[2 * n + n / 2] = inf;
        flat[3 * n] = nan;
        flat[4 * n + n - 1] = nan;
        for (size_t i = 0; i < n; ++i) flat[5 * n + i] = -inf;
        flat[6 * n + 7] = 9.0f;                          // its prior id is invalid
        const std::vector<uint32_t> prior = {0, 0, 0, 0, 0, 0, (uint32_t)n, 0, 0};
        const backend::BufferPtr lc = q35::floats_on(p.cpu, flat), lv = q35::floats_on(p.vk, flat), pc = ids_on(p.cpu, prior), pv = ids_on(p.vk, prior);
        for (bool with_prior : {true, false}) {
            const backend::BufferPtr oc = p.cpu.alloc(rows * sizeof(float), backend::Memory::device), ov = p.vk.alloc(rows * sizeof(float), backend::Memory::device);
            p.cpu.argmax_rows({oc.get(), 0}, {lc.get(), 0}, rows, n, with_prior ? backend::CSlice{pc.get(), 0} : backend::CSlice{});
            p.vk.argmax_rows({ov.get(), 0}, {lv.get(), 0}, rows, n, with_prior ? backend::CSlice{pv.get(), 0} : backend::CSlice{});
            p.vk.sync();
            const std::vector<uint32_t> c = ids_of(p.cpu, *oc, rows), v = ids_of(p.vk, *ov, rows);
            require(c == v, ("argmax_rows differs from the CPU over " + std::to_string(n) + " ids").c_str());
            require(c[1] == 3 && c[2] == n && c[3] == n && c[4] == n && c[5] == n && c[6] == (with_prior ? n : 7), "argmax_rows over ties, infinities and NaN");
            values += rows;
        }
    }
    const size_t width = 64, vocab = 9;
    const std::vector<float> table = uniform(width * vocab, 333, -2.0f, 2.0f);
    std::vector<uint8_t> q8(vocab * (width / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
    for (size_t r = 0; r < vocab; ++r)
        quant::quantize_row_q8_0(table.data() + r * width, q8.data() + r * (width / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE, width / quant::Q8_0_BLOCK);
    const std::vector<uint32_t> ids = {4, (uint32_t)vocab, 0, 0xffffffffu, 8};
    for (bool quantized : {false, true}) {
        const uint32_t type = quantized ? quant::GGML_TYPE_Q8_0 : quant::GGML_TYPE_F32;
        const void* data = quantized ? (const void*)q8.data() : (const void*)table.data();
        const size_t bytes = quantized ? q8.size() : table.size() * sizeof(float);
        const backend::BufferPtr tc = p.cpu.adopt(data, bytes), tv = p.vk.adopt(data, bytes), ic = ids_on(p.cpu, ids), iv = ids_on(p.vk, ids);
        const backend::BufferPtr dc = p.cpu.alloc(ids.size() * width * sizeof(float), backend::Memory::device), dv = p.vk.alloc(ids.size() * width * sizeof(float), backend::Memory::device);
        p.cpu.embed_ids({dc.get(), 0}, type, {tc.get(), 0}, width, vocab, {ic.get(), 0}, ids.size());
        p.vk.embed_ids({dv.get(), 0}, type, {tv.get(), 0}, width, vocab, {iv.get(), 0}, ids.size());
        p.vk.sync();
        const std::vector<float> c = q35::read_floats(p.cpu, *dc, ids.size() * width), v = q35::read_floats(p.vk, *dv, ids.size() * width);
        require(q35::same_bits(c, v), quantized ? "embed_ids differs from the CPU on a Q8_0 table" : "embed_ids differs from the CPU on an F32 table");
        for (size_t i = 0; i < width; ++i) require(v[width + i] == 0.0f && v[3 * width + i] == 0.0f, "embed_ids of an id past the table wrote no zero row");
        values += c.size();
    }
    // Every type the device embeds: embed_ids gives embed's rows for the ids inside the table, through the same decoding, and a zero row past it.
    const size_t wide = 256;
    for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q4_K,
                          quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_MXFP4}) {
        if (!p.vk.supports_type(type)) continue;
        const std::vector<uint8_t> bytes = matrix(type, wide, vocab, 340 + type);
        const backend::BufferPtr tv = p.vk.adopt(bytes.data(), bytes.size()), iv = ids_on(p.vk, ids);
        const backend::BufferPtr dv = p.vk.alloc(ids.size() * wide * sizeof(float), backend::Memory::device), ev = p.vk.alloc(ids.size() * wide * sizeof(float), backend::Memory::device);
        const std::vector<uint32_t> inside = {4, 0, 0, 0, 8};
        p.vk.embed_ids({dv.get(), 0}, type, {tv.get(), 0}, wide, vocab, {iv.get(), 0}, ids.size());
        p.vk.embed({ev.get(), 0}, type, {tv.get(), 0}, wide, vocab, inside.data(), inside.size());
        p.vk.sync();
        const std::vector<float> v = q35::read_floats(p.vk, *dv, ids.size() * wide), e = q35::read_floats(p.vk, *ev, ids.size() * wide);
        for (size_t r : {size_t(0), size_t(2), size_t(4)})
            require(std::memcmp(v.data() + r * wide, e.data() + r * wide, wide * sizeof(float)) == 0, ("embed_ids differs from embed on a table of type " + std::to_string(type)).c_str());
        for (size_t i = 0; i < wide; ++i) require(v[wide + i] == 0.0f && v[3 * wide + i] == 0.0f, "embed_ids of an id past the table wrote no zero row");
        values += v.size();
    }
    return values;
}

size_t check_qwen35(backend::Backend& vk) {
    Pair p(vk);
    const bool integer_dot = backend::vulkan_device_profile(p.vk).prefer_integer_dot;
    size_t values = q35::check_storage(vk);
    // Hv = Hk and Hv = 3 Hk at the tiny fixtures' widths, a shape past one column block, the files' 128 by 128 matrices, and the 0.8B's and the 27B's heads, whose slots are too large to cut into passes of a row here.
    const std::vector<backend::StateShape> shapes = {{2, 2, 12, 10}, {2, 6, 12, 10}, {2, 4, 64, 40}, {2, 4, 128, 128}, {16, 16, 128, 128}, {16, 48, 128, 128}};
    size_t runs = 0;
    uint32_t seed = 100;
    for (const backend::StateShape& sh : shapes) {
        try {
            values += q35::check_mix(vk, p.cpu, sh, seed);
            if (sh.k_heads == 2) runs += q35::check_invariance(vk, sh, seed + 1);
        } catch (const std::runtime_error&) {
            std::fprintf(stderr, "  linear attention K heads %zu V heads %zu, %zu by %zu\n", sh.k_heads, sh.v_heads, sh.k_dim, sh.v_dim);
            throw;
        }
        seed += 50;
    }
    q35::check_decay_flush(vk);
    values += q35::check_gated_norm(p, 2, 10) + q35::check_gated_norm(p, 16, 128);
    values += q35::check_sigmoid_mul(p, 4, 40) + q35::check_sigmoid_mul(p, 3, 256);
    values += q35::check_gated_attention(p, integer_dot);
    values += check_drafter_ops(p);
    std::cout << "backend-vulkan: qwen35 ops: " << runs << " device runs bit for bit across views, orders and passes; decay flush, state storage\n";
    return values;
}
}

// `--isa DIR` opens the backend for diagnostics and writes the driver's representation of every kernel it compiled, one file per kernel, after the checks, then checks each row kernel build's float multiplies and adds against its one-column build's (check_contraction).
// The integer-dot tile and every quantized row kernel hold the precision of 16-bit activations: against a double product of the unquantized inputs, an output is within half a 16-bit step of each block's peak times that block's weights, where 8-bit activations miss by the 8-bit step (docs/STATUS.md, MI50 prompt activations at 16 bits).
// Each block of the inputs holds one value 30 times the others, as a residual stream's outliers do, so a block's step follows its peak.
// Each type takes a wide tile case and a three-column row case. F32 and BF16 also take one and nine columns, permit only accumulation error on their respective inputs and hold each decode column to the same column alone.
size_t check_activation_precision(backend::Backend& vk, backend::Dtype dtype) {
    const size_t nin = 1024, nout = 48;
    const backend::DeviceProfile prof = backend::vulkan_device_profile(vk);
    const size_t tile_cols = std::max<size_t>(64, backend::tile_from_for(prof, false, nin));
    struct Case { uint32_t type; size_t nbatch; };
    std::vector<Case> cases;
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q4_K,
                          quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K})
        for (size_t cols : {size_t(1), size_t(3), size_t(9), tile_cols})
            if (dtype != backend::Dtype::f16 || cols == 3 || cols == tile_cols) cases.push_back({type, cols});
    size_t values = 0;
    for (const Case& c : cases) {
        std::vector<float> x = uniform(c.nbatch * nin, 300 + (uint32_t)c.nbatch);
        for (size_t i = 0; i < x.size(); i += 32) x[i + (i / 32) % 32] *= 30.0f;
        std::vector<float> reference = x;
        if (dtype == backend::Dtype::bf16)
            for (float& v : reference) v = bf16_to_f32(f32_to_bf16(v));
        const std::vector<uint8_t> wb = matrix(c.type, nin, nout, 301 + c.type);
        std::vector<float> w(nout * nin);
        const size_t rb = wb.size() / nout;
        for (size_t o = 0; o < nout; ++o) {
            const uint8_t* src = wb.data() + o * rb;
            float* dst = w.data() + o * nin;
            switch (c.type) {
            case quant::GGML_TYPE_Q8_0: quant::dequantize_row_q8_0(src, dst, nin / 32); break;
            case quant::GGML_TYPE_Q4_0: quant::dequantize_row_q4_0(src, dst, nin / 32); break;
            case quant::GGML_TYPE_Q4_1: quant::dequantize_row_q4_1(src, dst, nin / 32); break;
            case quant::GGML_TYPE_Q4_K: quant::dequantize_row_q4_K(src, dst, nin / 256); break;
            case quant::GGML_TYPE_Q5_K: quant::dequantize_row_q5_K(src, dst, nin / 256); break;
            default: quant::dequantize_row_q6_K(src, dst, nin / 256); break;
            }
        }
        const auto wd = vk.adopt(wb.data(), wb.size()), xd = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto yd = vk.alloc(c.nbatch * nout * sizeof(float));
        const backend::RowRun run{c.nbatch, c.nbatch < tile_cols ? size_t(1) : tile_cols};
        const backend::RowRuns runs = dtype == backend::Dtype::f32 ? backend::RowRuns{&run, 1} : backend::RowRuns{};
        testq::take_matrix_paths(vk);
        vk.matmul(c.type, {wd.get(), 0}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, c.nbatch, runs, dtype);
        if (dtype != backend::Dtype::f16)
            require(testq::take_matrix_paths(vk) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"}, "float precision check dispatched another arithmetic class");
        std::vector<float> y(c.nbatch * nout);
        vk.read(*yd, 0, y.data(), y.size() * sizeof(float));
        for (size_t b = 0; b < c.nbatch; ++b)
            for (size_t o = 0; o < nout; ++o) {
                double exact = 0, size = 0, bound = 0;
                for (size_t k = 0; k < nin; k += 32) {
                    double peak = 0, weights = 0;
                    for (size_t i = k; i < k + 32; ++i) {
                        const double wi = w[o * nin + i], xi = reference[b * nin + i];
                        exact += wi * xi;
                        size += std::fabs(wi * xi);
                        peak = std::max(peak, std::fabs(xi));
                        weights += std::fabs(wi);
                    }
                    if (dtype == backend::Dtype::f16) bound += peak / 32767.0 / 2.0 * weights;
                }
                bound = bound * 1.0001 + 1e-5 * size + 1e-6;
                const double got = y[b * nout + o];
                if (!(std::fabs(got - exact) <= bound)) {
                    std::fprintf(stderr, "  type %u, %zu columns, column %zu row %zu: device %.9g, exact %.9g, bound %.3g\n", c.type, c.nbatch, b, o,
                                 got, exact, bound);
                    throw std::runtime_error("a matmul misses its activation precision");
                }
            }
        if (dtype != backend::Dtype::f16 && c.nbatch < tile_cols) {
            const auto one = vk.alloc(nout * sizeof(float));
            std::vector<float> got(nout);
            for (size_t column = 0; column < c.nbatch; ++column) {
                vk.matmul(c.type, {wd.get(), 0}, {xd.get(), column * nin}, {one.get(), 0}, nin, nout, 1, {}, dtype);
                vk.read(*one, 0, got.data(), got.size() * sizeof(float));
                require(std::memcmp(got.data(), y.data() + column * nout, nout * sizeof(float)) == 0,
                        "float output depends on the decode batch width");
            }
        }
        values += c.nbatch * nout;
    }
    return values;
}

// BF16 uses a tile even for decode.
// Its inner split must follow the logical extent, not the number of independent sequences collected into a physical batch.
size_t check_bf16_batch_split(backend::Backend& vk) {
    const size_t nin = 1024, nout = 256, columns = 129;
    const auto input = uniform(nin, 411);
    std::vector<float> x(columns * nin);
    for (size_t c = 0; c < columns; ++c) std::copy(input.begin(), input.end(), x.begin() + c * nin);
    const auto xd = vk.adopt(x.data(), x.size() * sizeof(float));
    const auto yd = vk.alloc(2 * columns * nout * sizeof(float));
    size_t checked = 0;
    for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1,
                          quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_MXFP4}) {
        if (!vk.supports_type(type)) {
            std::cout << "backend-vulkan: BF16 batch type " << type << " unsupported, skipped\n";
            continue;
        }
        const auto weights = matrix(type, nin, nout, 412 + type);
        const auto wd = vk.adopt(weights.data(), weights.size());
        for (int op = 0; op < 4; ++op) {
            const size_t projections = op == 3 ? 2 : 1;
            auto apply = [&](size_t count, backend::RowRuns runs) {
                const auto dtype = backend::Dtype::bf16;
                std::vector<float> output(projections * count * nout, 0.25f);
                testq::take_matrix_paths(vk);
                if (op == 0) vk.matmul(type, {wd.get(), 0}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, count, runs, dtype);
                else if (op == 1) vk.matmul_logits(type, {wd.get(), 0}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, count, runs, dtype);
                else if (op == 2) {
                    vk.write(*yd, 0, output.data(), output.size() * sizeof(float));
                    vk.matmul_add(type, {wd.get(), 0}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, count, runs, dtype);
                } else vk.matmul_group({{type, {wd.get(), 0}, {yd.get(), 0}, nout},
                                        {type, {wd.get(), 0}, {yd.get(), count * nout}, nout}}, {xd.get(), 0}, nin, count, runs, dtype);
                vk.read(*yd, 0, output.data(), output.size() * sizeof(float));
                require(testq::take_matrix_paths(vk) == std::vector<std::string>{"bf16"}, "BF16 batch check ran another precision");
                return output;
            };
            auto equal_column = [&](const std::vector<float>& expected, const std::vector<float>& actual, size_t count, size_t c, size_t extent) {
                for (size_t p = 0; p < projections; ++p) {
                    if (std::memcmp(expected.data() + p * nout, actual.data() + (p * count + c) * nout, nout * sizeof(float))) {
                        std::fprintf(stderr, "BF16 batch split: type %u, op %d, extent %zu, batch %zu, column %zu\n", type, op, extent, count, c);
                        throw std::runtime_error("BF16 logical row changed with physical batch width");
                    }
                    checked += nout;
                }
            };
            for (size_t extent : {size_t(1), size_t(3), size_t(65), size_t(129)}) {
                const backend::RowRun one{1, extent};
                const auto expected = apply(1, {&one, 1});
                for (size_t count : {size_t(63), size_t(64), size_t(65), size_t(127), size_t(128), size_t(129)}) {
                    const backend::RowRun run{count, extent};
                    const auto actual = apply(count, {&run, 1});
                    for (size_t c = 0; c < count; ++c) equal_column(expected, actual, count, c, extent);
                }
            }
            const backend::RowRun decode{1, 1}, prompt{1, 129}, mixed[] = {{65, 1}, {columns, 129}};
            const auto first = apply(1, {&decode, 1}), last = apply(1, {&prompt, 1}), together = apply(columns, {mixed, 2});
            for (size_t c = 0; c < columns; ++c) equal_column(c < 65 ? first : last, together, columns, c, c < 65 ? 1 : 129);
        }
    }
    return checked;
}

// Four BF16 K parts over this head exceed 256 MiB.
// Odd rows and columns cross a workspace slice and leave a partial tile; sparse, exactly representable products check every result and the untouched output gaps without a second matmul oracle.
size_t check_float_workspace(backend::Backend& vk) {
    const size_t nin = 256, nout = 32769, columns = 513, guard = 64;
    const size_t values = nout * columns, stride = (values + guard - 1) / guard * guard;
    const float untouched = 42.0f;
    auto weight = [](size_t row, size_t k) { return float(int((row * 3 + k * 5) % 127) - 63) / 64.0f; };
    std::vector<float> w(nout * nin), x(guard + columns * nin, 0.0f);
    for (size_t row = 0; row < nout; ++row)
        for (size_t k = 0; k < nin; ++k) w[row * nin + k] = weight(row, k);
    for (size_t c = 0; c < columns; ++c) {
        x[guard + c * nin + c % nin] = c % 2 ? 0.5f : -0.5f;
        x[guard + c * nin + (c + 37) % nin] = 0.25f;
    }
    const auto wd = vk.adopt(w.data(), w.size() * sizeof(float));
    const auto xd = vk.adopt(x.data(), x.size() * sizeof(float));
    const auto yd = vk.alloc((2 * stride + 2 * guard) * sizeof(float));
    const backend::CSlice input{xd.get(), guard};
    const backend::Slice output{yd.get(), guard};
    const backend::RowRun run{columns, 16895};
    const backend::RowRuns runs{&run, 1};
    size_t checked = 0;
    for (int op = 0; op < 4; ++op) {
        const size_t projections = op == 3 ? 2 : 1;
        std::vector<float> actual(2 * stride + 2 * guard, untouched);
        vk.write(*yd, 0, actual.data(), actual.size() * sizeof(float));
        testq::take_matrix_paths(vk);
        if (op == 0) vk.matmul(quant::GGML_TYPE_F32, {wd.get(), 0}, input, output, nin, nout, columns, runs, backend::Dtype::bf16);
        else if (op == 1) vk.matmul_logits(quant::GGML_TYPE_F32, {wd.get(), 0}, input, output, nin, nout, columns, runs, backend::Dtype::bf16);
        else if (op == 2) vk.matmul_add(quant::GGML_TYPE_F32, {wd.get(), 0}, input, output, nin, nout, columns, runs, backend::Dtype::bf16);
        else vk.matmul_group({{quant::GGML_TYPE_F32, {wd.get(), 0}, output, nout},
                              {quant::GGML_TYPE_F32, {wd.get(), 0}, {yd.get(), guard + stride}, nout}}, input, nin, columns, runs, backend::Dtype::bf16);
        vk.read(*yd, 0, actual.data(), actual.size() * sizeof(float));
        require(testq::take_matrix_paths(vk) == std::vector<std::string>{"bf16"}, "workspace check ran another matrix precision");
        for (size_t p = 0; p < projections; ++p) {
            for (size_t c = 0; c < columns; ++c) {
                for (size_t row = 0; row < nout; ++row) {
                    const float dot = weight(row, c % nin) * (c % 2 ? 0.5f : -0.5f) + weight(row, (c + 37) % nin) * 0.25f;
                    require(actual[guard + p * stride + c * nout + row] == dot + (op == 2 ? untouched : 0.0f), "sliced float workspace product differs");
                    ++checked;
                }
            }
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            const bool written = i >= guard && (i - guard) / stride < projections && (i - guard) % stride < values;
            if (!written) require(actual[i] == untouched, "sliced float workspace crossed an output boundary");
        }
    }
    return checked;
}

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
        check_float_ops();
        check_decode_contraction();
        std::cout << "backend-vulkan: " << backend::vulkan_device_name(*b) << "\n";
        size_t checks = 0;
        // The backend's identity names the device and its driver, and a second backend of the same device has the same one.
        require(b->identity().rfind("vulkan ", 0) == 0 && b->identity().find(backend::vulkan_device_name(*b)) != std::string::npos &&
                    backend::make_vulkan_backend(0, false)->identity() == b->identity(),
                "the backend's identity does not name its device or differs between two backends of it");
        std::cout << "backend-vulkan: identity " << b->identity() << "\n";

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

        // A device holding between submissions (Backend::hold_between_submissions): work behind a hold runs once the next submission lets it go, also after an idle gap the watchdog ends, and a backend dropped with a hold pending tears down.
        {
            auto h = backend::make_vulkan_backend(0, false);
            h->hold_between_submissions(true);
            h->hold_between_submissions(true);
            h->hold_between_submissions(false);
            const auto hs = h->adopt(kept.data(), 4096);
            const auto hd = h->alloc(4096, backend::Memory::host_visible);
            h->copy(*hd, 0, *hs, 0, 1024);
            h->wait(h->submit());
            backend::Ticket last = 0;
            for (size_t o = 1024; o < 2048; o += 128) {
                h->copy(*hd, o, *hs, o, 128);
                last = h->submit();
            }
            h->wait(last);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            h->copy(*hd, 2048, *hs, 2048, 2048);
            h->wait(h->submit());
            require(std::memcmp(hd->host_ptr(), kept.data(), 4096) == 0, "work beside holds differs");
            // The last copy retires before its buffers go, and the hold after it is left for the backend's teardown.
            h->copy(*hd, 0, *hs, 0, 64);
            h->wait(h->submit());
        }
        checks += 1;

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
        std::cout << "backend-vulkan: " << check_weight_dispatch() << " expected weight dispatches and type refusals\n";

        std::cout << "backend-vulkan: " << testq::check_matrix_precision(*b) << " shared matrix precision values passed\n";
        std::cout << "backend-vulkan: " << check_matrix_witness(*b) << " matrix-path witnesses match arithmetic\n";
        const size_t values = check_kernels(*b) + check_qwen35(*b);
        std::cout << "backend-vulkan: " << checks << " storage and submission checks; "
                  << values << " kernel outputs against the CPU backend\n";
        std::cout << "backend-vulkan: " << check_float_workspace(*b) << " exact products across bounded float workspace slices\n";
        std::cout << "backend-vulkan: " << check_timing_coverage() << " dispatches timed between two readings, past one query pool\n";
        std::cout << "backend-vulkan: " << check_bf16_batch_split(*b) << " BF16 outputs invariant across physical tile boundaries\n";
        const size_t columns = check_decode_columns(*b);
        std::cout << "backend-vulkan: " << columns << " decode columns equal to the same columns alone\n";
        std::cout << "backend-vulkan: " << check_bf16_rounding(*b) << " BF16 boundary and unchanged-weight products, with F32 control\n";
        const size_t precise = check_activation_precision(*b, backend::Dtype::f16);
        std::cout << "backend-vulkan: " << precise << " outputs within the 16-bit activations' precision\n";
        std::cout << "backend-vulkan: " << check_activation_precision(*b, backend::Dtype::f32) << " F32 outputs against double dots and batch identity\n";
        std::cout << "backend-vulkan: " << check_activation_precision(*b, backend::Dtype::bf16) << " BF16 outputs against rounded-input double dots and batch identity\n";
        std::cout << "backend-vulkan: " << q35::check_row_classes(*b) << " pairs of extents of one class with the same bits\n";
        std::cout << backend::vulkan_kernel_statistics(*b);
        if (!isa_dir.empty()) {
            const auto representations = backend::vulkan_kernel_representations(*b);
            size_t written = 0;
            for (const auto& kr : representations) {
                std::ofstream f(isa_dir + "/" + kr.first + ".txt");
                f << kr.second;
                written += f.good() ? 1 : 0;
            }
            std::cout << "backend-vulkan: " << written << " kernel representations written to " << isa_dir << "\n";
            std::cout << "backend-vulkan: " << check_q6_dots(representations) << " Q6 builds retain native 16-bit dots\n";
            const ContractionChecks c = check_contraction(representations);
            std::cout << "backend-vulkan: row kernel builds against their one-column build's float multiplies and adds: " << c.same << " the same, "
                      << c.whole_columns << " those and whole columns, " << c.two_rows << " pairs of two-row builds whole columns apart, " << c.decode
                      << " Q8_0 decode builds those their shape and forms give, "
                      << c.kinds_only << " their kinds only; " << c.grouped << " grouped builds the same as their wide build\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
