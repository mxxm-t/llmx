// Vulkan storage, submission and kernel checks against CPU or independent references; coverage and bounds are in AGENTS.md, Tests.
// Exits 77 when no supported device opens, which CTest reports as skipped.
#include <algorithm>
#include <cctype>
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
#include "moe_step.hpp"

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

using testq::row_activations;

// The activations a matmul of `type` reads, on the tile kernel or the row kernel.
std::vector<float> fed(const std::vector<float>& x, uint32_t type, bool idot, bool tile) {
    if (type == quant::GGML_TYPE_F32 || (tile && !idot)) return x;
    return row_activations(x);
}

// `rows` rows of `in` values: floats and block quantizers from seeded floats, half weights and K-quants from bounded bit patterns.
std::vector<uint8_t> matrix(uint32_t type, size_t in, size_t rows, uint32_t seed) {
    const auto f = uniform(rows * in, seed);
    std::vector<uint8_t> bytes;
    if (type == quant::GGML_TYPE_F32) {
        bytes.resize(f.size() * sizeof(float));
        std::memcpy(bytes.data(), f.data(), bytes.size());
    } else if (type == quant::GGML_TYPE_F16) {
        bytes.resize(f.size() * 2);
        for (size_t i = 0; i < f.size(); ++i) {
            const uint16_t bits = uint16_t((i % 3 == 0 ? 0x8000u : 0) | ((11 + (i + seed) % 5) << 10) |
                                           ((i * 73 + seed) & 1023u));
            bytes[2 * i] = uint8_t(bits);
            bytes[2 * i + 1] = uint8_t(bits >> 8);
        }
    } else if (type == quant::GGML_TYPE_BF16) {
        bytes.resize(f.size() * sizeof(uint16_t));
        for (size_t i = 0; i < f.size(); ++i) {
            uint32_t bits;
            std::memcpy(&bits, &f[i], sizeof bits);
            const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
            const uint16_t bf = uint16_t(rounded >> 16);
            std::memcpy(bytes.data() + i * sizeof bf, &bf, sizeof bf);
        }
    } else if (type == quant::GGML_TYPE_MXFP4) {
        bytes = testq::mxfp4_matrix(rows * in, seed);
    } else if (type == quant::GGML_TYPE_TQ1_0 || type == quant::GGML_TYPE_TQ2_0) {
        bytes = testq::tq_matrix(type, rows * in, seed);
    } else if (type == quant::GGML_TYPE_Q5_0) {
        bytes.resize(rows * (in / 32) * 22, 0);
        for (size_t b = 0; b < rows * (in / 32); ++b) {
            uint8_t* dst = bytes.data() + b * 22;
            const uint16_t h = uint16_t(0x2400u | ((b * 73 + seed) & 0x83ffu));
            dst[0] = uint8_t(h); dst[1] = uint8_t(h >> 8);
            for (unsigned j = 0; j < 32; ++j) {
                const unsigned code = unsigned(j * 7 + b + seed) & 31;
                dst[2 + j / 8] |= uint8_t((code >> 4) << (j % 8));
                dst[6 + j % 16] |= uint8_t((code & 15) << (4 * (j / 16)));
            }
        }
    } else if (type == quant::GGML_TYPE_Q8_0 || type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1 || type == quant::GGML_TYPE_Q5_1) {
        const size_t ts = type == quant::GGML_TYPE_Q8_0 ? quant::Q8_0_TYPESIZE : type == quant::GGML_TYPE_Q4_0 ? quant::Q4_0_TYPESIZE : type == quant::GGML_TYPE_Q5_1 ? quant::Q5_1_TYPESIZE : quant::Q4_1_TYPESIZE;
        bytes.resize(rows * (in / 32) * ts);
        for (size_t r = 0; r < rows; ++r) {
            uint8_t* dst = bytes.data() + r * (in / 32) * ts;
            if (type == quant::GGML_TYPE_Q8_0) quant::quantize_row_q8_0(f.data() + r * in, dst, in / 32);
            else if (type == quant::GGML_TYPE_Q4_0) quant::quantize_row_q4_0(f.data() + r * in, dst, in / 32);
            else testq::quantizer(type)(f.data() + r * in, dst, in / 32);
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

size_t check_q5_embedding(backend::Backend& vk, uint32_t type) {
    const bool symmetric = type == quant::GGML_TYPE_Q5_0;
    const uint16_t halves[] = {0, 0x8000, 1, 0x8201, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff};
    std::vector<uint8_t> weights;
    std::vector<float> expected;
    std::vector<uint32_t> ids;
    for (uint16_t ds : halves) for (uint16_t ms : halves) for (unsigned shift = 0; shift < 32; ++shift) {
        if (symmetric && ms != 0) continue;
        uint8_t block[24] = {uint8_t(ds), uint8_t(ds >> 8)};
        const unsigned qoff = symmetric ? 6 : 8;
        if (!symmetric) { block[2] = uint8_t(ms); block[3] = uint8_t(ms >> 8); }
        for (unsigned j = 0; j < 32; ++j) {
            const unsigned code = (j + shift) % 32;
            block[qoff - 4 + j / 8] |= uint8_t((code >> 4) << (j % 8));
            block[qoff + j % 16] |= uint8_t((code & 15) << (4 * (j / 16)));
            if (symmetric) {
                const unsigned e = (ds >> 10) & 31, f = ds & 1023;
                const float d = std::copysign(float(std::ldexp(double(f + (e ? 1024 : 0)), int(e ? e : 1) - 25)), ds & 0x8000 ? -1.0f : 1.0f);
                expected.push_back(d * float(int(code) - 16));
            } else expected.push_back(f16_to_f32(ds) * float(code) + f16_to_f32(ms));
        }
        weights.insert(weights.end(), block, block + (symmetric ? 22 : 24));
        ids.push_back(uint32_t(ids.size()));
    }
    const auto w = vk.adopt(weights.data(), weights.size());
    std::vector<float> got(expected.size() + 2, 19.0f);
    const auto y = vk.adopt(got.data(), got.size() * sizeof(float));
    vk.embed({y.get(), 1}, {type, {w.get(), 0}}, 32, ids.size(), ids.data(), ids.size());
    vk.read(*y, 0, got.data(), got.size() * sizeof(float));
    require(got.front() == 19.0f && got.back() == 19.0f, "Q5 embedding changed a guard");
    require(std::memcmp(got.data() + 1, expected.data(), expected.size() * sizeof(float)) == 0,
            "Q5 embedding differs from independent values");
    if (symmetric) {
        std::fill(got.begin() + 1, got.end() - 1, 19.0f);
        vk.write(*y, 0, got.data(), got.size() * sizeof(float));
        const auto tokens = vk.adopt(ids.data(), ids.size() * sizeof(uint32_t));
        vk.embed_ids({y.get(), 1}, {type, {w.get(), 0}}, 32, ids.size(), {tokens.get(), 0}, ids.size());
        vk.read(*y, 0, got.data(), got.size() * sizeof(float));
        require(got.front() == 19 && got.back() == 19 && std::memcmp(got.data() + 1, expected.data(), expected.size() * sizeof(float)) == 0,
                "Q5_0 device-ID embedding differs bitwise");
    }
    return expected.size() * (symmetric ? 2 : 1);
}

// A table past 4 GiB, as Qwen3.8-Flash-Next's per-layer table is (table_views, shaders/embed.comp): F32, Q8_0 and IQ4_NL rows written at the start, either side of 2^32 bytes and at the end of a zeroed table.
// They must read back, bit for bit, as the same rows of a small table on the CPU, through host and device ids, with an id past the table a zero row; a device that cannot hold the table says so and checks nothing.
size_t check_large_table(backend::Backend& vk) {
    struct Kind {
        uint32_t type;
        size_t nin;
        const char* name;
    };
    size_t checked = 0;
    uint32_t seed = 12345;
    const auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (const Kind k : {Kind{quant::GGML_TYPE_F32, 16, "F32"}, Kind{quant::GGML_TYPE_Q8_0, 32, "Q8_0"}, Kind{quant::GGML_TYPE_IQ4_NL, 160, "IQ4_NL"}}) {
        const size_t rb = quant::row_bytes(k.type, k.nin), edge = (size_t(1) << 32) / rb, rows = edge + 4096;
        const std::vector<size_t> at = {0, 1, edge - 1, edge, edge + 1, rows - 1};
        backend::BufferPtr big;
        try {
            big = vk.alloc(rows * rb);
        } catch (const std::exception& e) {
            std::cout << "backend-vulkan: a " << k.name << " table of " << rows * rb << " bytes not checked, the device holds no such buffer (" << e.what() << ")\n";
            continue;
        }
        std::vector<uint8_t> small(at.size() * rb);
        for (size_t i = 0; i < small.size(); ++i) small[i] = uint8_t(next());
        for (size_t r = 0; r < at.size(); ++r) {
            uint8_t* row = small.data() + r * rb;
            if (k.type == quant::GGML_TYPE_F32) {
                for (size_t j = 0; j < k.nin; ++j) {
                    const float v = float(int(next() % 2001) - 1000) / 64.0f;
                    std::memcpy(row + 4 * j, &v, 4);
                }
            } else {
                // Each block's scale a finite half from 1/64 to just under 1, either sign.
                const size_t block = k.type == quant::GGML_TYPE_Q8_0 ? 34 : 18;
                for (size_t b = 0; b < rb / block; ++b) {
                    const uint16_t h = uint16_t(0x2400 + next() % 0x1800) | uint16_t(next() % 2 ? 0x8000 : 0);
                    row[b * block] = uint8_t(h), row[b * block + 1] = uint8_t(h >> 8);
                }
            }
            vk.write(*big, at[r] * rb, row, rb);
        }
        backend::CpuBackend cpu;
        const auto table = cpu.adopt(small.data(), small.size());
        const std::vector<uint32_t> small_ids = {5, 3, 0, 4, 2, 1, 3}, big_ids = {uint32_t(at[5]), uint32_t(at[3]), uint32_t(at[0]), uint32_t(at[4]), uint32_t(at[2]), uint32_t(at[1]), uint32_t(at[3])};
        const size_t n = small_ids.size();
        std::vector<float> want((n + 1) * k.nin, 0.0f), got(want.size(), 19.0f);
        const auto c = cpu.alloc(n * k.nin * sizeof(float), backend::Memory::device);
        cpu.embed({c.get(), 0}, {k.type, {table.get(), 0}}, k.nin, at.size(), small_ids.data(), n);
        cpu.read(*c, 0, want.data(), n * k.nin * sizeof(float));
        const auto y = vk.adopt(got.data(), got.size() * sizeof(float));
        vk.embed({y.get(), 0}, {k.type, {big.get(), 0}}, k.nin, rows, big_ids.data(), n);
        vk.read(*y, 0, got.data(), got.size() * sizeof(float));
        require(std::memcmp(got.data(), want.data(), n * k.nin * sizeof(float)) == 0 && got.back() == 19.0f,
                (std::string("a ") + k.name + " table past 4 GiB gives other rows than the CPU").c_str());
        std::vector<uint32_t> device_ids = big_ids;
        device_ids.push_back(uint32_t(rows));
        const auto ids = vk.adopt(device_ids.data(), device_ids.size() * sizeof(uint32_t));
        std::fill(got.begin(), got.end(), 19.0f);
        vk.write(*y, 0, got.data(), got.size() * sizeof(float));
        vk.embed_ids({y.get(), 0}, {k.type, {big.get(), 0}}, k.nin, rows, {ids.get(), 0}, device_ids.size());
        vk.read(*y, 0, got.data(), got.size() * sizeof(float));
        require(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0,
                (std::string("a ") + k.name + " table past 4 GiB gives other rows than the CPU through device ids").c_str());
        checked += 2 * n * k.nin + k.nin;
    }
    return checked;
}

// Choose the nearest BF16 neighbour independently; non-BF16 weights check that only inputs round.
// Zero sign is excluded because accumulation may turn -0 into +0.
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
        vk.matmul({quant::GGML_TYPE_F32, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 1, 2, input.size(), {}, dtype);
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

// BF16 GGUF weights stay in their original two-byte storage on the device; the reference widens each high half to F32.
size_t check_bf16_weight_decode(backend::Backend& vk) {
    Pair p(vk);
    constexpr size_t width = 2560, qrows = 512, krows = 128;
    const auto raw = matrix(quant::GGML_TYPE_BF16, width, qrows + krows, 1409);
    std::vector<float> wide((qrows + krows) * width);
    for (size_t i = 0; i < wide.size(); ++i) {
        uint16_t bits;
        std::memcpy(&bits, raw.data() + i * sizeof bits, sizeof bits);
        const uint32_t fbits = uint32_t(bits) << 16;
        std::memcpy(&wide[i], &fbits, sizeof fbits);
    }
    const auto wt = p.in(raw.data(), raw.size());
    const auto fw = p.in(wide);

    // The Qwen indexer consumes these 2560->512 and 2560->128 projections together.
    const uint32_t ids[] = {3, 1, 0};
    Pair::Out embedded = p.out(width * 3);
    p.cpu.embed(embedded.cs(), {quant::GGML_TYPE_F32, fw.cs()}, width, qrows + krows, ids, 3);
    p.vk.embed(embedded.vs(), {quant::GGML_TYPE_BF16, wt.vs()}, width, qrows + krows, ids, 3);
    auto er = p.results(embedded);
    size_t checked = exact(er.first, er.second, "BF16 embedding does not widen the stored high half exactly");

    const auto x = uniform(width, 1411, -0.1f, 0.1f);
    Pair::In xi = p.in(x);
    Pair::Out q = p.out(qrows), k = p.out(krows);
    p.cpu.matmul({quant::GGML_TYPE_F32, fw.cs()}, xi.cs(), q.cs(), width, qrows, 1, {}, backend::Dtype::f32);
    p.cpu.matmul({quant::GGML_TYPE_F32, {fw.c.get(), qrows * width}}, xi.cs(), k.cs(), width, krows, 1, {}, backend::Dtype::f32);
    p.vk.matmul_group({{{quant::GGML_TYPE_BF16, wt.vs()}, q.vs(), qrows},
                       {{quant::GGML_TYPE_BF16, {wt.v.get(), qrows * width / 2}}, k.vs(), krows}},
                      xi.vs(), width, 1, {}, backend::Dtype::f32);
    auto qr = p.results(q), kr = p.results(k);
    checked += close(qr.first, qr.second, 2e-5, "BF16 indexer query projection differs from widened F32");
    checked += close(kr.first, kr.second, 2e-5, "BF16 indexer key projection differs from widened F32");

    // BF16 activation requests select the float tile; widen its activations independently for the CPU oracle.
    constexpr size_t cols = 64;
    auto batch = uniform(width * cols, 1413, -0.1f, 0.1f);
    std::vector<float> rounded(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &batch[i], sizeof bits);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        bits &= 0xffff0000u;
        std::memcpy(&rounded[i], &bits, sizeof bits);
    }
    Pair::In bi = p.in(batch), ri = p.in(rounded);
    Pair::Out bq = p.out(qrows * cols), bk = p.out(krows * cols);
    p.cpu.matmul({quant::GGML_TYPE_F32, fw.cs()}, ri.cs(), bq.cs(), width, qrows, cols, {}, backend::Dtype::f32);
    p.cpu.matmul({quant::GGML_TYPE_F32, {fw.c.get(), qrows * width}}, ri.cs(), bk.cs(), width, krows, cols, {}, backend::Dtype::f32);
    p.vk.matmul_group({{{quant::GGML_TYPE_BF16, wt.vs()}, bq.vs(), qrows},
                       {{quant::GGML_TYPE_BF16, {wt.v.get(), qrows * width / 2}}, bk.vs(), krows}},
                      bi.vs(), width, cols, {}, backend::Dtype::bf16);
    auto bqr = p.results(bq), bkr = p.results(bk);
    checked += close(bqr.first, bqr.second, 2e-4, "BF16 indexer tile query projection differs from widened F32");
    checked += close(bkr.first, bkr.second, 2e-4, "BF16 indexer tile key projection differs from widened F32");
    return checked;
}

size_t check_matrix_witness(backend::Backend& vk) {
    const bool integer = backend::vulkan_device_profile(vk).prefer_integer_dot;
    testq::take_matrix_paths(vk);
    const auto native = vk.native_dtypes();
    require(vk.dtype_path(backend::Dtype::f32) == "f32", "device F32 catalog claims a narrower path");
    require(vk.dtype_path(backend::Dtype::bf16) == "bf16 (matrix inputs), f32 (routers)", "device BF16 catalog hides retained F32 routers");
    require(vk.dtype_path(backend::Dtype::int8) == "block-int8 (quantized rows and tiles but Q2_K/Q3_K/MXFP4/TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ4_NL/IQ4_XS/IQ3_S), block-int16 (Q2_K/Q3_K rows on integer-dot profiles and eligible tiles; MXFP4, TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS rows on lookup-dot profiles and eligible tiles), f32 (Q2_K/Q3_K, TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS fallback rows, other products, routers)",
            "device int8 catalog does not name the TQ path");
    const std::string half_paths = integer
        ? "block-int16 (quantized rows and eligible tiles; TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS rows on lookup-dot profiles), f32 (TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS fallback, other products, routers)"
        : "block-int16 (quantized rows except Q2_K/Q3_K; TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS rows on lookup-dot profiles), f32 (Q2_K/Q3_K, TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS fallback rows, tiles, F32 weights, routers)";
    require(vk.dtype_path(backend::Dtype::f16) == half_paths ||
            vk.dtype_path(backend::Dtype::f16) == "f16 (NVFP4 tile inputs), " + half_paths, "device path catalog ignores the tile profile");
    require(vk.native_dtypes() == native, "device path catalog changed native policy preferences");
    require(testq::take_matrix_paths(vk).empty(), "device path catalog fabricated an execution witness");
    size_t checked = 0;
    for (const uint32_t type : {quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q5_0}) {
        std::vector<uint8_t> weights(type == quant::GGML_TYPE_Q5_0 ? 22 : 18, 0x88);
        weights[0] = 0; weights[1] = 0x3c;
        if (type == quant::GGML_TYPE_Q5_0) {
            std::fill(weights.begin() + 2, weights.begin() + 6, 0xff);
            std::fill(weights.begin() + 6, weights.end(), 0);
            weights[7] = 1; // code 17 at column 1; all other codes are 16.
        } else weights[3] = 0x89;
        std::vector<float> x(96), out(6), ids(3, 0), gains(3, 1);
        for (size_t r = 0; r < 3; ++r) { x[r * 32] = 2; x[r * 32 + 1] = r == 1 ? -1.005859375f : 1.005859375f; }
        const auto w = vk.adopt(weights.data(), weights.size()), in = vk.adopt(x.data(), x.size() * sizeof(float)),
                   y = vk.alloc(out.size() * sizeof(float)), id = vk.adopt(ids.data(), ids.size() * sizeof(float)),
                   gain = vk.adopt(gains.data(), gains.size() * sizeof(float));
        const backend::Backend::Routing route{{id.get(), 0}, {gain.get(), 0}, 1, 1};
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::f16, backend::Dtype::int8}) {
            // Native int8 is covered against its independent reconstruction in check_int8.
            if (dtype == backend::Dtype::int8 && integer) continue;
            for (bool tile : {false, true}) {
                const backend::RowRun run[] = {{3, tile ? size_t(16384) : size_t(1)}};
                for (int op = 0; op < 6; ++op) {
                    std::fill(out.begin(), out.end(), 0.0f);
                    vk.write(*y, 0, out.data(), out.size() * sizeof(float));
                    testq::take_matrix_paths(vk);
                    const backend::CSlice ws{w.get(), 0}, xs{in.get(), 0};
                    const backend::Slice ys{y.get(), 0};
                    const backend::RowRuns runs{run, 1};
                    if (op == 0) vk.matmul({type, ws}, xs, ys, 32, 1, 3, runs, dtype);
                    if (op == 1) vk.matmul_logits({type, ws}, xs, ys, 32, 1, 3, runs, dtype);
                    if (op == 2) vk.matmul_add({type, ws}, xs, ys, 32, 1, 3, runs, dtype);
                    if (op == 3) vk.matmul_group({{{type, ws}, ys, 1}, {{type, ws}, {y.get(), 3}, 1}}, xs, 32, 3, runs, dtype);
                    if (op == 4) vk.matmul_experts({{{type, ws}, ys, 1}}, xs, 32, 3, route, runs, dtype);
                    if (op == 5) vk.matmul_experts_add({type, ws}, xs, ys, 32, 1, 3, route, runs, dtype);
                    vk.read(*y, 0, out.data(), out.size() * sizeof(float));
                    const bool narrow = (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8) && (!tile || integer);
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
    }
    return checked;
}

// Each block has its peak and keeps other values at least 0.2 of a quantization step from a tie, so host and device reciprocals choose the same codes.
std::vector<float> tie_free(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> peak(0.5f, 1.5f), off(-0.3f, 0.3f);
    std::uniform_int_distribution<int> code(-126, 126);
    std::vector<float> v(n);
    for (size_t b = 0; b + 32 <= n; b += 32) {
        const float amax = peak(rng);
        v[b] = (b / 32) % 2 ? -amax : amax;
        for (size_t i = 1; i < 32; ++i) v[b + i] = (float(code(rng)) + off(rng)) * amax / 127.0f;
    }
    return v;
}

// Check int8 matrix paths against CPU products of independently rounded inputs, with whole and sliced prompt identity (AGENTS.md, Tests).
// MXFP4 and F32 weights must retain F16 results and report their wider paths.
size_t check_int8(backend::Backend& vk) {
    const auto native = vk.native_dtypes();
    if (std::find(native.begin(), native.end(), backend::Dtype::int8) == native.end()) {
        std::cout << "backend-vulkan: int8 not run here\n";
        return 0;
    }
    Pair p(vk);
    size_t values = 0;
    const uint32_t types[] = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1,
                              quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K};
    for (uint32_t type : types)
        for (size_t nin : {size_t(512), size_t(1280)}) {
            const size_t nout = 80;
            const auto w = matrix(type, nin, nout, 700 + type + uint32_t(nin));
            Pair::In wi = p.in(w.data(), w.size());
            // Generated tokens, each a run of extent 1, at widths that take the one-column, two-row and wide builds, then prompts that take the row kernels and the tile.
            for (size_t nbatch : {size_t(1), size_t(3), size_t(8), size_t(13), size_t(40), size_t(53), size_t(64), size_t(128), size_t(131), size_t(200)}) {
                const bool prompt = nbatch >= 40;
                const auto x = tie_free(nbatch * nin, 800 + uint32_t(nbatch) + type + uint32_t(nin));
                const auto x8 = row_activations(x, 127);
                Pair::In xi = p.in(x), x8i = p.in(x8);
                std::vector<backend::RowRun> runs;
                if (prompt) runs.push_back({nbatch, nbatch});
                else for (size_t r = 0; r < nbatch; ++r) runs.push_back({r + 1, 1});
                Pair::Out d = p.out(nbatch * nout);
                p.cpu.matmul({type, wi.cs()}, x8i.cs(), d.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
                (void)testq::take_matrix_paths(vk);
                p.vk.matmul({type, wi.vs()}, xi.vs(), d.vs(), nin, nout, nbatch, {runs.data(), runs.size()}, backend::Dtype::int8);
                require(testq::take_matrix_paths(vk) == std::vector<std::string>{"block-int8"}, "int8 rows did not witness block-int8");
                auto r = p.results(d);
                try {
                    values += close(r.first, r.second, 1e-3, "int8 matmul differs from the CPU on 8-bit activations beyond 1e-3");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  int8 matmul type %u nin %zu nbatch %zu\n", type, nin, nbatch);
                    throw;
                }
                if (!prompt) continue;
                // The same prompt in two slices, each run naming the prompt's extent.
                const size_t first = nbatch / 2;
                const backend::RowRun head[1] = {{first, nbatch}}, tail[1] = {{nbatch - first, nbatch}};
                Pair::Out sl = p.out(nbatch * nout);
                p.vk.matmul({type, wi.vs()}, xi.vs(), sl.vs(), nin, nout, first, {head, 1}, backend::Dtype::int8);
                p.vk.matmul({type, wi.vs()}, backend::CSlice{xi.v.get(), first * nin}, backend::Slice{sl.v.get(), first * nout}, nin, nout,
                            nbatch - first, {tail, 1}, backend::Dtype::int8);
                std::vector<float> sliced(nbatch * nout);
                vk.read(*sl.v, 0, sliced.data(), sliced.size() * sizeof(float));
                values += exact(r.second, sliced, "an int8 prompt's rows changed with its slicing");
                (void)testq::take_matrix_paths(vk);
            }
        }
    // Types without an 8-bit build: int8 gives f16's bits, decode and prompt.
    for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
        if (!vk.supports_type(type)) continue;
        const size_t nin = 512, nout = 80;
        const auto w = matrix(type, nin, nout, 760 + type);
        Pair::In wi = p.in(w.data(), w.size());
        for (size_t nbatch : {size_t(3), size_t(200)}) {
            const auto x = uniform(nbatch * nin, 900 + type + uint32_t(nbatch));
            Pair::In xi = p.in(x);
            std::vector<backend::RowRun> runs;
            if (nbatch > 3) runs.push_back({nbatch, nbatch});
            else for (size_t r = 0; r < nbatch; ++r) runs.push_back({r + 1, 1});
            Pair::Out a = p.out(nbatch * nout), b = p.out(nbatch * nout);
            p.vk.matmul({type, wi.vs()}, xi.vs(), a.vs(), nin, nout, nbatch, {runs.data(), runs.size()}, backend::Dtype::int8);
            const auto witness = testq::take_matrix_paths(vk);
            require(std::find(witness.begin(), witness.end(), "block-int8") == witness.end(), "a type without an 8-bit build took 8-bit inputs");
            p.vk.matmul({type, wi.vs()}, xi.vs(), b.vs(), nin, nout, nbatch, {runs.data(), runs.size()}, backend::Dtype::f16);
            (void)testq::take_matrix_paths(vk);
            std::vector<float> ra(nbatch * nout), rb(nbatch * nout);
            vk.read(*a.v, 0, ra.data(), ra.size() * sizeof(float));
            vk.read(*b.v, 0, rb.data(), rb.size() * sizeof(float));
            values += exact(rb, ra, "a type without an 8-bit build differs under int8 from f16");
        }
    }
    return values;
}

size_t check_int8_twin_width() {
    const auto owner = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *owner;
    const auto native = vk.native_dtypes();
    if (std::find(native.begin(), native.end(), backend::Dtype::int8) == native.end()) return 0;
    const auto profile = backend::vulkan_device_profile(vk);
    const size_t rows = std::max({size_t(32), backend::tile_from_for(profile, true, 512),
                                 backend::tile_from_for(profile, true, 256)});
    constexpr size_t nout = 80;
    Pair p(vk);
    const auto input = tie_free(512 * rows, 1739);
    const auto wide = matrix(quant::GGML_TYPE_Q8_0, 512, nout, 1741);
    const auto narrow = matrix(quant::GGML_TYPE_Q8_0, 256, nout, 1743);
    const auto x = p.in(input), w0 = p.in(wide.data(), wide.size()), w1 = p.in(narrow.data(), narrow.size());
    const auto first = p.out(nout * rows), reused = p.out(nout * rows * 2), fresh = p.out(nout * rows * 2);
    const backend::RowRun r0{rows, rows}, r1{rows * 2, rows * 2};
    // Keep all allocations and writes before the two calls that must share the cached input.
    vk.matmul({quant::GGML_TYPE_Q8_0, w0.vs()}, x.vs(), first.vs(), 512, nout, rows, {&r0, 1}, backend::Dtype::int8);
    vk.matmul({quant::GGML_TYPE_Q8_0, w1.vs()}, x.vs(), reused.vs(), 256, nout, rows * 2, {&r1, 1}, backend::Dtype::int8);
    vk.write(*x.v, 0, input.data(), input.size() * sizeof(float));
    vk.matmul({quant::GGML_TYPE_Q8_0, w1.vs()}, x.vs(), fresh.vs(), 256, nout, rows * 2, {&r1, 1}, backend::Dtype::int8);
    std::vector<float> actual(reused.n), expected(fresh.n);
    vk.read(*reused.v, 0, actual.data(), actual.size() * sizeof(float));
    vk.read(*fresh.v, 0, expected.data(), expected.size() * sizeof(float));
    const auto kernels = backend::vulkan_kernel_times(vk);
    require(std::any_of(kernels.begin(), kernels.end(), [](const auto& k) {
        return k.first.find("matmul_tile_q8i8") == 0;
    }), "int8 twin width check did not execute an int8 tile");
    return exact(expected, actual, "int8 twin reused across matrix widths changes output");
}

// Twins producers wrote stay with their activations through writes elsewhere in the same buffer: matmuls from them quantize nothing and equal matmuls from fresh copies of their floats (AGENTS.md, Tests).
size_t check_twin_table() {
    const auto owner = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *owner;
    const auto native = vk.native_dtypes();
    const size_t n_in = 1024, n_out = 96, k = 2, n_expert = 4;
    std::vector<uint8_t> w8(n_out * (n_in / 32) * quant::Q8_0_TYPESIZE), e8(n_expert * n_out * (n_in / 32) * quant::Q8_0_TYPESIZE);
    const auto wf = uniform(n_out * n_in, 1801), ef = uniform(n_expert * n_out * n_in, 1802);
    for (size_t r = 0; r < n_out; ++r)
        quant::quantize_row_q8_0(wf.data() + r * n_in, w8.data() + r * (n_in / 32) * quant::Q8_0_TYPESIZE, n_in / 32);
    for (size_t r = 0; r < n_expert * n_out; ++r)
        quant::quantize_row_q8_0(ef.data() + r * n_in, e8.data() + r * (n_in / 32) * quant::Q8_0_TYPESIZE, n_in / 32);
    size_t values = 0;
    bool timed = true;
    for (const backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
        if (std::find(native.begin(), native.end(), dtype) == native.end()) continue;
        for (const size_t rows : {size_t(128), size_t(1)}) {
            const size_t n = rows * n_in, bytes = n * sizeof(float), entries = rows * k;
            const std::vector<backend::RowRun> one{{rows, rows}};
            const backend::RowRuns rr{one.data(), one.size()};
            const auto src = uniform(n, 1803), wn = uniform(n_in, 1804, 0.5f, 1.5f), g = uniform(n, 1805, -6.0f, 6.0f);
            const auto u = uniform(n, 1806), other = uniform(n, 1807), xe = uniform(entries * n_in, 1808);
            const auto scores = uniform(rows * n_expert, 1809, -3.0f, 3.0f);
            // Every adoption and allocation first, since a new buffer drops every twin.
            const auto wb = vk.adopt(w8.data(), w8.size()), eb = vk.adopt(e8.data(), e8.size());
            const auto srcb = vk.adopt(src.data(), bytes), wnb = vk.adopt(wn.data(), n_in * sizeof(float));
            const auto gb = vk.adopt(g.data(), bytes), ub = vk.adopt(u.data(), bytes), otherb = vk.adopt(other.data(), bytes);
            const auto xeb = vk.adopt(xe.data(), entries * n_in * sizeof(float));
            const auto scoresb = vk.adopt(scores.data(), scores.size() * sizeof(float));
            // One buffer of five regions, as activations share one arena: A and D take producers' outputs, B a copy, C a routed sum and E nothing.
            const auto arena = vk.alloc(5 * bytes, backend::Memory::device);
            const auto idsb = vk.alloc(entries * sizeof(float), backend::Memory::device), wtsb = vk.alloc(entries * sizeof(float), backend::Memory::device);
            std::vector<backend::BufferPtr> ys;
            for (int i = 0; i < 8; ++i) ys.push_back(vk.alloc(rows * n_out * sizeof(float), backend::Memory::device));
            const backend::Slice A{arena.get(), 0}, D{arena.get(), 3 * n};
            const backend::Backend::Routing routing{{idsb.get(), 0}, {wtsb.get(), 0}, k, n_expert};
            vk.route_experts({scoresb.get(), 0}, rows, n_expert, k, true, {idsb.get(), 0}, {wtsb.get(), 0});
            // The first 8-bit matmul has the producers after it write the 8-bit twin.
            if (dtype == backend::Dtype::int8)
                vk.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {otherb.get(), 0}, {ys[7].get(), 0}, n_in, n_out, rows, rr, dtype);
            auto matmul = [&](backend::CSlice x, const backend::BufferPtr& y) {
                vk.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, x, {y.get(), 0}, n_in, n_out, rows, rr, dtype);
            };
            auto floats = [&](const backend::BufferPtr& b, size_t at, size_t count) {
                std::vector<float> v(count);
                vk.read(*b, at * sizeof(float), v.data(), count * sizeof(float));
                return v;
            };
            // The matmuls `run` records quantize nothing, where the device times its dispatches.
            auto quantizes_nothing = [&](const char* after, const std::function<void()>& run) {
                (void)backend::vulkan_kernel_times(vk);
                run();
                const auto times = backend::vulkan_kernel_times(vk);
                if (times.empty()) {
                    timed = false;
                    return;
                }
                for (const auto& t : times)
                    if (t.first.rfind("quantize_x", 0) == 0) {
                        std::fprintf(stderr, "  twin table, dtype %s, %zu rows: %s\n", backend::dtype_name(dtype), rows, t.first.c_str());
                        require(false, after);
                    }
            };
            try {
                vk.rms_norm_rows(A, {srcb.get(), 0}, {wnb.get(), 0}, rows, n_in, n_in, 1e-6f, rr);
                quantizes_nothing("a matmul from a norm's output quantized it", [&] { matmul(A, ys[0]); });
                vk.matmul_experts_add({quant::GGML_TYPE_Q8_0, {eb.get(), 0}}, {xeb.get(), 0}, {arena.get(), 2 * n}, n_in, n_out, rows, routing, rr, dtype);
                quantizes_nothing("a routed sum into another region dropped a norm's twin", [&] { matmul(A, ys[1]); });
                vk.copy(*arena, n * sizeof(float), *otherb, 0, bytes);
                quantizes_nothing("a copy into another region dropped a norm's twin", [&] { matmul(A, ys[2]); });
                vk.silu_mul(D, {gb.get(), 0}, {ub.get(), 0}, n, rr);
                quantizes_nothing("a second producer's twin replaced the first", [&] {
                    matmul(A, ys[3]);
                    matmul(D, ys[4]);
                });
                const auto a0 = floats(arena, 0, n), d0 = floats(arena, 3 * n, n);
                // In place over part of a buffer and without a twin, a norm of rows narrower than their stride and a SiLU of a length off the blocks must drop the twins they overwrite.
                vk.rms_norm_rows(D, D, {wnb.get(), 0}, rows, n_in - 24, n_in, 1e-6f, rr);
                matmul(D, ys[5]);
                vk.silu_mul(A, A, {ub.get(), 0}, n - 16, rr);
                matmul(A, ys[6]);
                const auto a1 = floats(arena, 0, n), d1 = floats(arena, 3 * n, n);
                std::vector<std::vector<float>> got;
                for (int i = 0; i < 7; ++i) got.push_back(floats(ys[i], 0, rows * n_out));
                auto fresh = [&](const std::vector<float>& x) {
                    const auto xb = vk.adopt(x.data(), bytes);
                    matmul({xb.get(), 0}, ys[7]);
                    return floats(ys[7], 0, rows * n_out);
                };
                const auto fa0 = fresh(a0), fd0 = fresh(d0), fa1 = fresh(a1), fd1 = fresh(d1);
                values += exact(fa0, got[0], "a matmul from a norm's twin differs from one from a fresh copy");
                values += exact(fa0, got[1], "a matmul from a norm's twin after a routed sum beside it differs from one from a fresh copy");
                values += exact(fa0, got[2], "a matmul from a norm's twin after a copy beside it differs from one from a fresh copy");
                values += exact(fa0, got[3], "a matmul from a norm's twin beside a second twin differs from one from a fresh copy");
                values += exact(fd0, got[4], "a matmul from a SiLU's twin beside a norm's differs from one from a fresh copy");
                values += exact(fd1, got[5], "a matmul read a twin a norm without one overwrote in place");
                values += exact(fa1, got[6], "a matmul read a twin a SiLU without one overwrote in place");
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  twin table, dtype %s, %zu rows\n", backend::dtype_name(dtype), rows);
                throw;
            }
        }
    }
    if (!timed) std::cout << "backend-vulkan: the device times no dispatches, so the twin table's quantizations were not witnessed\n";
    return values;
}

// Each kernel's write set as the backend derives it from its SPIR-V, against its shader's qualifiers: a binding whose every declaration is readonly is read only, and one with no readonly declaration is written (AGENTS.md, Tests).
size_t check_write_sets() {
    std::map<std::string, std::string> sources;
    for (std::string list = LLMX_VK_SHADER_SOURCES; !list.empty();) {
        const size_t comma = list.find(','), eq = list.find('=');
        sources[list.substr(0, eq)] = list.substr(eq + 1, comma == std::string::npos ? std::string::npos : comma - eq - 1);
        list = comma == std::string::npos ? std::string() : list.substr(comma + 1);
    }
    // These kernel entries reuse a compiled row module under their own names.
    sources["matmul_row_f32"] = "matmul_row";
    sources["matmul_row_nvfp4"] = "matmul_row";
    // A shader with its includes, each read once.
    std::function<void(const std::string&, std::string&, std::vector<std::string>&)> read =
        [&](const std::string& file, std::string& text, std::vector<std::string>& seen) {
            if (std::find(seen.begin(), seen.end(), file) != seen.end()) return;
            seen.push_back(file);
            std::ifstream f(std::string(LLMX_VK_SHADER_DIR) + "/" + file);
            require(f.good(), ("shader source missing: " + file).c_str());
            for (std::string line; std::getline(f, line);) {
                text += line + "\n";
                const size_t inc = line.find("#include \"");
                if (inc != std::string::npos) read(line.substr(inc + 10, line.find('"', inc + 10) - inc - 10), text, seen);
            }
        };
    size_t bindings = 0;
    for (const auto& kw : backend::vulkan_kernel_writes()) {
        std::string name = kw.first;
        if (!sources.count(name) && name.size() > 5 && name.compare(name.size() - 5, 5, "_tall") == 0) name.resize(name.size() - 5);
        require(sources.count(name) == 1, ("no shader source for kernel " + kw.first).c_str());
        std::string text;
        std::vector<std::string> seen;
        read(sources[name] + ".comp", text, seen);
        // Per binding, whether some declaration is readonly and whether some is not.
        std::map<size_t, std::pair<bool, bool>> declared;
        for (size_t at = text.find("layout("); at != std::string::npos; at = text.find("layout(", at + 1)) {
            const size_t close = text.find(')', at), binding = text.find("binding", at), end = text.find_first_of("{;", at);
            if (close == std::string::npos || binding > close || end == std::string::npos) continue;
            const std::string qualifiers = text.substr(close + 1, end - close - 1);
            if (qualifiers.find("buffer") == std::string::npos) continue;
            const size_t b = std::stoul(text.substr(text.find('=', binding) + 1));
            (qualifiers.find("readonly") != std::string::npos ? declared[b].first : declared[b].second) = true;
        }
        for (size_t b = 0; b < kw.second.size(); ++b, ++bindings) {
            const auto d = declared[b];
            // No shader declares one binding both ways in different builds; one that does is refused until the check gives it an expectation of its own.
            require(!(d.first && d.second), ("a binding declared both readonly and writable in " + kw.first).c_str());
            if (kw.second[b] != d.second) {
                std::fprintf(stderr, "  %s binding %zu: %s in the shader, %s in the write set\n", kw.first.c_str(), b,
                             d.second ? "written" : "read only", kw.second[b] ? "written" : "read only");
                require(false, "a kernel's write set differs from its shader's qualifiers");
            }
        }
    }
    return bindings;
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
    // Partial RoPE at out-of-order positions, over contiguous full heads and padded gated queries with a quarter of each head rotated.
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
        p.cpu.embed(d.cs(), {quant::GGML_TYPE_F32, t.cs()}, nin, nrows, ids, 3);
        p.vk.embed(d.vs(), {quant::GGML_TYPE_F32, t.vs()}, nin, nrows, ids, 3);
        auto r = p.results(d);
        values += exact(r.first, r.second, "embed F32 differs");

        std::vector<uint8_t> q(nrows * (nin / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q8_0(table.data() + row * nin,
                                     q.data() + row * (nin / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE,
                                     nin / quant::Q8_0_BLOCK);
        Pair::In tq = p.in(q.data(), q.size());
        Pair::Out dq = p.out(nin * 3);
        p.cpu.embed(dq.cs(), {quant::GGML_TYPE_Q8_0, tq.cs()}, nin, nrows, ids, 3);
        p.vk.embed(dq.vs(), {quant::GGML_TYPE_Q8_0, tq.vs()}, nin, nrows, ids, 3);
        auto rq = p.results(dq);
        values += exact(rq.first, rq.second, "embed Q8_0 differs");

        std::vector<uint8_t> q4(nrows * (nin / quant::Q4_0_BLOCK) * quant::Q4_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q4_0(table.data() + row * nin,
                                     q4.data() + row * (nin / quant::Q4_0_BLOCK) * quant::Q4_0_TYPESIZE,
                                     nin / quant::Q4_0_BLOCK);
        Pair::In t4 = p.in(q4.data(), q4.size());
        Pair::Out d4 = p.out(nin * 3);
        p.cpu.embed(d4.cs(), {quant::GGML_TYPE_Q4_0, t4.cs()}, nin, nrows, ids, 3);
        p.vk.embed(d4.vs(), {quant::GGML_TYPE_Q4_0, t4.vs()}, nin, nrows, ids, 3);
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
        p.cpu.embed(draw4.cs(), {quant::GGML_TYPE_Q4_0, traw4.cs()}, nin, nrows, ids, 3);
        p.vk.embed(draw4.vs(), {quant::GGML_TYPE_Q4_0, traw4.vs()}, nin, nrows, ids, 3);
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
            p.cpu.embed(d6.cs(), {quant::GGML_TYPE_Q6_K, t6.cs()}, n6, nrows, ids, 3);
            p.vk.embed(d6.vs(), {quant::GGML_TYPE_Q6_K, t6.vs()}, n6, nrows, ids, 3);
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
            p.cpu.embed(dk.cs(), {type, tk.cs()}, nk, nrows, ids, 3);
            p.vk.embed(dk.vs(), {type, tk.vs()}, nk, nrows, ids, 3);
            auto rk = p.results(dk);
            values += exact(rk.first, rk.second, k == 0 ? "embed Q4_K differs" : "embed Q5_K differs");
        }
        const uint32_t beyond[1] = {10};
        bool rejected = false;
        try { p.vk.embed(d.vs(), {quant::GGML_TYPE_F32, t.vs()}, nin, nrows, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "embedding row beyond the table accepted");
        // A table that holds fewer rows than the call names is refused before any row is read, even when every id is inside the table.
        const uint32_t first[1] = {0};
        auto short_table_refused = [&](uint32_t type, const Pair::In& table_in) {
            try { p.vk.embed(d.vs(), {type, table_in.vs()}, nin, nrows + 1, first, 1); }
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
        p.vk.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.write(*wb, 0, w2.data(), w2.size() * sizeof(float));
        p.vk.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.matmul({quant::GGML_TYPE_F32, {fresh.get(), 0}}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
        std::vector<float> a(nout * cols), b(nout * cols);
        p.vk.read(*y1, 0, a.data(), a.size() * sizeof(float));
        p.vk.read(*y2, 0, b.data(), b.size() * sizeof(float));
        values += exact(a, b, "an F32 matmul after a write into its weights read the old weights");
    }
    // Weights written across row boundaries must retain their bytes and the products of adopted weights, including the float tile padded copy.
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
        p.vk.matmul({quant::GGML_TYPE_F32, {adopted.get(), 0}}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
        p.vk.matmul({quant::GGML_TYPE_F32, {filled.get(), 0}}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
        std::vector<float> a(nout * cols), b(nout * cols), back(w.size());
        p.vk.read(*y1, 0, a.data(), a.size() * sizeof(float));
        p.vk.read(*y2, 0, b.data(), b.size() * sizeof(float));
        p.vk.read(*filled, 0, back.data(), bytes);
        require(std::memcmp(back.data(), w.data(), bytes) == 0, "a weight written in pieces into alloc_weight storage differs from its source");
        values += exact(a, b, "a weight written in pieces into alloc_weight storage gave other products than the same weight adopted");
    }
    // Imported host pages must copy into weights across row boundaries without changing bytes or products.
    // Unaligned addresses and partial pages must be refused.
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
            p.vk.matmul({quant::GGML_TYPE_F32, {adopted.get(), 0}}, {xb.get(), 0}, {y1.get(), 0}, nin, nout, cols);
            p.vk.matmul({quant::GGML_TYPE_F32, {filled.get(), 0}}, {xb.get(), 0}, {y2.get(), 0}, nin, nout, cols);
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
            // Choose reference activations from the device profile: row kernels and integer-dot tiles read the 16-bit twin; float tiles read the original inputs.
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
                p.cpu.matmul({type, wi.cs()}, (q == 0 ? xi : q == 1 ? xri8 : q <= 3 ? xri4 : xrik).cs(), d.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
                p.vk.matmul({type, wi.vs()}, xi.vs(), d.vs(), nin, nout, nbatch);
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
                        p.cpu.matmul({type, wi.cs()}, x16i.cs(), h.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
                        p.vk.matmul_logits({type, wi.vs()}, xi.vs(), h.vs(), nin, nout, nbatch);
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
                infer::BlockPool pool; pool.configure(st->max_blocks());
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
        // Tiled attention covers full and partial query tiles, both head widths and cache types, and several histories against the CPU (AGENTS.md, Tests).
        // Scaled queries produce peaked scores rather than nearly uniform softmax.
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
                    infer::BlockPool pool; pool.configure(st->max_blocks());
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
                infer::BlockPool pool; pool.configure(st->max_blocks());
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
                        if (add) vk.matmul_add({t.type, {wb.get(), 0}}, {xb.get(), first * n_in}, {yb.get(), 0}, n_in, n_out, n, rr);
                        else vk.matmul({t.type, {wb.get(), 0}}, {xb.get(), first * n_in}, {yb.get(), 0}, n_in, n_out, n, rr);
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
            infer::BlockPool pool; pool.configure(st->max_blocks());
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
        // Producer-written and separately quantized twins must give identical products, including both int8 layouts (AGENTS.md, Tests).
        // Intervening host or kernel writes must invalidate the cached twin.
        if (nin == 1024) {
            backend::Dtype dtype = backend::Dtype::f16;
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
                vk.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {x.get(), 0}, {yb.get(), 0}, n_in, n_out, rows, rr, dtype);
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
            infer::BlockPool pool; pool.configure(st->max_blocks());
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
            const auto compare = [&] {
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
                        vk.matmul_experts_add({quant::GGML_TYPE_Q8_0, {eb.get(), 0}}, {f.get(), 0}, {yb.get(), 0}, n_in, n_out, rows, routing, rr, dtype);
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
                // A norm or SiLU that writes no twin, here over part of each row in place, still invalidates the twin of the buffer it writes.
                // Two rows, so the row kernels read the cached twin; the reference's buffer and the outputs are allocated first, since an allocation drops every twin.
                {
                    const size_t r2 = 2, b2 = r2 * n_in * sizeof(float);
                    const std::vector<backend::RowRun> two{{r2, r2}};
                    const auto h2 = vk.alloc(b2, backend::Memory::device), fresh = vk.alloc(b2, backend::Memory::device);
                    const auto y2 = vk.alloc(r2 * n_out * sizeof(float), backend::Memory::device);
                    auto product = [&](const backend::BufferPtr& x) {
                        vk.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {x.get(), 0}, {y2.get(), 0}, n_in, n_out, r2, {two.data(), two.size()}, dtype);
                        std::vector<float> y(r2 * n_out);
                        vk.read(*y2, 0, y.data(), y.size() * sizeof(float));
                        return y;
                    };
                    auto refresh = [&] {
                        std::vector<float> v(r2 * n_in);
                        vk.read(*h2, 0, v.data(), b2);
                        vk.write(*fresh, 0, v.data(), b2);
                        return product(fresh);
                    };
                    vk.rms_norm_rows({h2.get(), 0}, {srcb.get(), 0}, {wnb.get(), 0}, r2, n_in, n_in, 1e-6f);
                    vk.rms_norm_rows({h2.get(), 0}, {h2.get(), 0}, {wnb.get(), 0}, r2, n_in / 2, n_in, 1e-6f);
                    const auto after_norm = product(h2);
                    values += exact(after_norm, refresh(), "a matmul read a norm's twin after a norm without a twin wrote over it");
                    vk.silu_mul({h2.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, r2 * n_in);
                    vk.silu_mul({h2.get(), 0}, {h2.get(), 0}, {ub.get(), 0}, r2 * n_in - 16);
                    const auto after_silu = product(h2);
                    values += exact(after_silu, refresh(), "a matmul read a SiLU's twin after a SiLU without a twin wrote over it");
                }
            };
            try {
                compare();
                const auto native = vk.native_dtypes();
                if (std::find(native.begin(), native.end(), backend::Dtype::int8) != native.end()) {
                    dtype = backend::Dtype::int8;
                    // The first 8-bit matmul has the producers after it write the 8-bit twin.
                    matmul_of(otherb);
                    compare();
                }
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  producers' twin for the tile, dtype %s\n", backend::dtype_name(dtype));
                throw;
            }
        }
        // Check attention and split-history merge twins through a following row matmul, against CPU attention quantized for the same dot.
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
                    infer::BlockPool pool; pool.configure(st->max_blocks());
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
                        b.matmul({quant::GGML_TYPE_Q8_0, {Wb.get(), 0}}, {ob.get(), 0}, {yb.get(), 0}, qw, nout, nq);
                    } else {
                        std::vector<float> att(nq * qw);
                        b.read(*ob, 0, att.data(), att.size() * sizeof(float));
                        const auto ar = fed(att, quant::GGML_TYPE_Q8_0, integer_dot, false);
                        const auto Ab = b.adopt(ar.data(), ar.size() * sizeof(float));
                        b.matmul({quant::GGML_TYPE_Q8_0, {Wb.get(), 0}}, {Ab.get(), 0}, {yb.get(), 0}, qw, nout, nq);
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
                    infer::BlockPool pool; pool.configure(st->max_blocks());
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
            p.cpu.matmul_add({quant::GGML_TYPE_Q8_0, wqi.cs()}, xri.cs(), d.cs(), nin, nout, nbatch, {}, backend::Dtype::f32);
            p.vk.matmul_add({quant::GGML_TYPE_Q8_0, wqi.vs()}, xi.vs(), d.vs(), nin, nout, nbatch);
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
            p.vk.matmul({quant::GGML_TYPE_Q8_0, wqi.vs()}, h.vs(), d1.vs(), nin, nout, rows);
            p.vk.silu_mul(f.vs(), gi.vs(), ui.vs(), rows * nin);
            p.vk.matmul({quant::GGML_TYPE_Q4_0, w4i.vs()}, f.vs(), d2.vs(), nin, nout, rows);
            p.cpu.rms_norm_rows(h.cs(), si.cs(), wni.cs(), rows, nin, nin, 1e-6f);
            p.cpu.silu_mul(f.cs(), gi.cs(), ui.cs(), rows * nin);
            std::vector<float> hc(rows * nin), fc(rows * nin);
            p.cpu.read(*h.c, 0, hc.data(), hc.size() * sizeof(float));
            p.cpu.read(*f.c, 0, fc.data(), fc.size() * sizeof(float));
            const auto hr = fed(hc, quant::GGML_TYPE_Q8_0, integer_dot, false), fr = fed(fc, quant::GGML_TYPE_Q4_0, integer_dot, false);   // Q8_0 from the norm and Q4_0 from the SiLU, each on the twin its kernel reads
            Pair::In hri = p.in(hr), fri = p.in(fr);
            p.cpu.matmul({quant::GGML_TYPE_Q8_0, wqi.cs()}, hri.cs(), d1.cs(), nin, nout, rows, {}, backend::Dtype::f32);
            p.cpu.matmul({quant::GGML_TYPE_Q4_0, w4i.cs()}, fri.cs(), d2.cs(), nin, nout, rows, {}, backend::Dtype::f32);
            auto r1 = p.results(d1), r2 = p.results(d2);
            values += close(r1.first, r1.second, tol_of(quant::GGML_TYPE_Q8_0, false), "matmul from the norm's twin differs beyond its bound");
            values += close(r2.first, r2.second, tol_of(quant::GGML_TYPE_Q4_0, false), "matmul from the SiLU's twin differs beyond its bound");
        }
        bool rejected = false;
        try { p.vk.matmul({25u /* I16, no kernel */, wqi.vs()}, wqi.vs(), p.out(8).vs(), nin, 1, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported matrix type accepted");
        // Row runs out of order are refused even when every run takes the same kernel, so nothing would have split them.
        {
            const backend::RowRun merged[2] = {{3, 1}, {2, 1}};
            const auto x = uniform(2 * nin, 13);
            Pair::In xi = p.in(x);
            Pair::Out d = p.out(2 * nout);
            rejected = false;
            try { p.vk.matmul({quant::GGML_TYPE_Q8_0, wqi.vs()}, xi.vs(), d.vs(), nin, nout, 2, {merged, 2}); }
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
                p.vk.matmul({quant::GGML_TYPE_Q8_0, wqi.vs()}, xi.vs(), sep[i].vs(), nin, rows[i], nbatch);
            p.vk.matmul_group({{{quant::GGML_TYPE_Q8_0, wqi.vs()}, grp[0].vs(), rows[0]},
                               {{quant::GGML_TYPE_Q8_0, wqi.vs()}, grp[1].vs(), rows[1]},
                               {{quant::GGML_TYPE_Q8_0, wqi.vs()}, grp[2].vs(), rows[2]}},
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
            p.vk.matmul_group({{{types[0], wqi.vs()}, grp[0].vs(), rows[0]},
                               {{types[1], w4i.vs()}, grp[1].vs(), rows[1]},
                               {{types[2], wqi.vs()}, grp[2].vs(), rows[2]}},
                              xi.vs(), nin, nbatch);
            for (int i = 0; i < 3; ++i) {
                p.cpu.matmul({types[i], (i == 1 ? w4i : wqi).cs()}, xri.cs(), grp[i].cs(), nin, rows[i], nbatch, {}, backend::Dtype::f32);
                auto r = p.results(grp[i]);
                try {
                    values += close(r.first, r.second, tol_of(types[i], true), "grouped tile projections differ beyond their bound");
                } catch (const std::runtime_error&) {
                    std::fprintf(stderr, "  grouped tile: projection %d of type %u, %zu rows, %zu columns\n", i, types[i], rows[i], nbatch);
                    throw;
                }
            }
        }
        // A mixed group keeps every partition on the whole group's row kernel, even when its Q8_0 partition alone would cross to a tile.
        // Compare both edges of the gap against forced row calls (AGENTS.md, Tests).
        for (size_t nbatch : {tile_from_8bit, tile_from_other - 1}) {
            if (nbatch < tile_from_8bit || nbatch >= tile_from_other) continue;
            const auto x = uniform(nbatch * nin, 50 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            const uint32_t types[2] = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0};
            Pair::Out grp[2] = {p.out(nbatch * nout), p.out(nbatch * nout)};
            Pair::Out alone[2] = {p.out(nbatch * nout), p.out(nbatch * nout)};
            p.vk.matmul_group({{{types[0], wqi.vs()}, grp[0].vs(), nout}, {{types[1], w4i.vs()}, grp[1].vs(), nout}},
                              xi.vs(), nin, nbatch);
            const backend::RowRun decode{nbatch, 1};
            for (int i = 0; i < 2; ++i)
                p.vk.matmul({types[i], (i == 0 ? wqi : w4i).vs()}, xi.vs(), alone[i].vs(), nin, nout, nbatch, {&decode, 1});
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
    // Attention reads each backend's own cache layout across block boundaries, including two views in one call.
    // Its online softmax differs in reduction order from the CPU, so the comparison allows a tolerance.
    {
        const int n_head = 4, n_head_kv = 2, head_dim = 40;
        const size_t kvw = (size_t)n_head_kv * head_dim, qw = (size_t)n_head * head_dim, layers = 2;
        for (size_t n_past : {size_t(0), size_t(63), size_t(64), size_t(65), size_t(131)}) {
            for (size_t nq : {size_t(1), size_t(3)}) {
                auto run = [&](backend::Backend& b, const std::vector<float>& K, const std::vector<float>& V,
                               const std::vector<float>& Q, std::vector<float>& out, bool split) {
                    const size_t bt = b.kv_layout().block_tokens;
                    auto st = b.kv_alloc(layers, n_head_kv, head_dim, 8 * 128);
                    infer::BlockPool pool; pool.configure(st->max_blocks());
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
        vk.matmul({t.type, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, 1);
        vk.sync();
        const int iters = 100;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul({t.type, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, 1);
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
        vk.matmul({t.type, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, nbatch);
        vk.sync();
        const int iters = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul({t.type, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, nbatch);
        vk.sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: " << t.name << " prefill " << nout << "x" << nin << " over " << nbatch << " rows "
                  << us << " us, " << 2.0 * nin * nout * nbatch / us / 1e6 << " TFLOPS\n";
    }
    // The float tile over BF16 weights at a vision encoder's shapes, Qwen3.8-Flash-Next's tower over a 1024 by 1024 image's 4096 patches, reported.
    for (const auto& [nin, nout] : {std::pair<size_t, size_t>{1152, 3456}, {1152, 4304}, {4304, 1152}}) {
        const size_t nbatch = 4096;
        std::vector<uint16_t> wq(nin * nout);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint16_t(0x3c00u + (i * 7 + 3) % 512u);
        const auto x = uniform(nin * nbatch, 24);
        const auto w = vk.adopt(wq.data(), wq.size() * sizeof(uint16_t));
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(nout * nbatch * sizeof(float), backend::Memory::device);
        vk.matmul({quant::GGML_TYPE_BF16, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, nbatch);
        vk.sync();
        const int iters = 10;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) vk.matmul({quant::GGML_TYPE_BF16, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, nbatch);
        vk.sync();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: BF16 prefill " << nout << "x" << nin << " over " << nbatch << " rows " << us << " us, " << 2.0 * nin * nout * nbatch / us / 1e6 << " TFLOPS\n";
    }
    // Decode attention and the small kernels at the Qwen3-0.6B shape over a 250-token history, one query, reported.
    {
        const int n_head = 16, n_head_kv = 8, head_dim = 128;
        const size_t hist = 250, kvw = (size_t)n_head_kv * head_dim, qw = (size_t)n_head * head_dim;
        auto st = vk.kv_alloc(1, n_head_kv, head_dim, 1024);
        infer::BlockPool pool; pool.configure(st->max_blocks());
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
        vk.matmul({quant::GGML_TYPE_Q8_0, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const int iters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul({quant::GGML_TYPE_Q8_0, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: Q8_0 matvec 4096x4096 " << ms << " ms, "
                  << (double)wq.size() / ms / 1e6 << " GB/s\n";
    }
    // Routing and grouped gate/up/down products are checked against the CPU for every type, across row, tile and mixed passes (AGENTS.md, Tests).
    // Each reference row uses the activations its selected kernel reads.
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
            for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1,
                                  quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
        from = type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1 || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_Q5_1 ||
                       type == quant::GGML_TYPE_TQ1_0 || type == quant::GGML_TYPE_TQ2_0 ? prof.moe_tile_from_q4
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
                    p.cpu.matmul_experts({{{type, wgi.cs()}, g.cs(), nff}, {{type, wui.cs()}, u.cs(), nff}}, xri.cs(), nin, rows, rc, runs, backend::Dtype::f32);
                    p.vk.matmul_experts({{{type, wgi.vs()}, g.vs(), nff}, {{type, wui.vs()}, u.vs(), nff}}, xi.vs(), nin, rows, rv, runs);
                    auto rg = p.results(g), ru = p.results(u);
                    values += close(rg.first, rg.second, tol, "routed gate projection differs beyond its bound");
                    // Generated tokens beside each other go through the grouped row kernel, one run of an expert's entries a workgroup row; each token alone takes an entry per workgroup row.
                    // The entries must come out bit for bit the same either way.
                    if (!sh.runs.empty() && rows > 1) {
                        std::vector<backend::RowRun> decode(rows);
                        for (size_t r = 0; r < rows; ++r) decode[r] = backend::RowRun{r + 1, 1};
                        Pair::Out gb = p.out(entries * nff);
                        p.vk.matmul_experts({{{type, wgi.vs()}, gb.vs(), nff}}, xi.vs(), nin, rows, rv, {decode.data(), rows});
                        const std::vector<float> batched = p.results(gb).second;
                        const backend::RowRun one[1] = {{1, 1}};
                        for (size_t r = 0; r < rows; ++r) {
                            Pair::Out ga = p.out(k * nff);
                            const backend::Backend::Routing alone{{ids.vs().buffer, ids.vs().offset + r * k}, {wts.vs().buffer, wts.vs().offset + r * k}, k, n_expert};
                            p.vk.matmul_experts({{{type, wgi.vs()}, ga.vs(), nff}}, {xi.vs().buffer, xi.vs().offset + r * nin}, nin, 1, alone, {one, 1});
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
                    p.cpu.matmul_experts_add({type, wdi.cs()}, x2ri.cs(), y.cs(), nin, nout, rows, rc, runs, backend::Dtype::f32);
                    p.vk.matmul_experts_add({type, wdi.vs()}, x2i.vs(), y.vs(), nin, nout, rows, rv, runs);
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

// A one-column Q8_0 decode call takes two rows and one step for rows narrower than the profile's q8_decode_narrow_below, else one row and four steps below the knee of its activations' width.
// Above it int8 takes two rows and two steps and f16 two rows and one step; a group counts its projections' rows together, on a backend that takes the integer dot (vulkan_prefer_integer_dot).
size_t check_q8_one_column_builds(backend::Backend& vk) {
    const backend::DeviceProfile prof = backend::vulkan_device_profile(vk);
    const size_t wide = prof.q8_decode_narrow_below, from = prof.q8_decode_two_rows_from, from16 = prof.q8_decode_one_row_below_16bit;
    require(wide >= 64 && wide % 32 == 0 && from > 256 && from16 > 256, "the profile's Q8_0 one-column numbers leave no case below them");
    const uint32_t q8 = quant::GGML_TYPE_Q8_0;
    const auto x = uniform(wide, 520);
    const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
    // Rows nin wide, a of one projection and b of a second, one row and one block either side of each number.
    struct Case { size_t nin, a, b; };
    std::vector<Case> cases = {{wide - 32, 256, 0}, {wide, 256, 0}, {wide - 32, from, 0}, {wide - 32, from16, 0}};
    for (size_t knee : {from, from16})
        for (const Case& c : {Case{wide, knee - 1, 0}, Case{wide, knee, 0}, Case{wide, knee / 2, knee - knee / 2 - 1}, Case{wide, knee / 2, knee - knee / 2}}) cases.push_back(c);
    // Short rows take the paired build where the profile has the half-block order and the tree: 16 blocks, 14 in a group of two, 10; the odd 9 and the 18 past a half's lanes keep the others.
    for (const Case& c : {Case{512, 300, 0}, Case{448, 150, 151}, Case{320, from, 0}, Case{288, 300, 0}, Case{576, 300, 0}})
        if (c.nin <= wide) cases.push_back(c);
    const bool pairs = (prof.q8_decode_forms & 3u) == 3u;
    size_t checks = 0;
    for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
        const bool x8 = dtype == backend::Dtype::int8;
        const std::string want_one = x8 ? "matmul_vec_q8_i8_1col" : "matmul_vec_q8_1col";
        // The suffix of the build the rule gives, from the profile's numbers.
        auto build = [&](size_t nin, size_t rows) -> std::string {
            if (pairs && nin / 32 % 2 == 0 && nin / 32 <= 16) return "_paired";
            if (nin < wide) return "";
            if (!x8) return rows < from16 ? "_1row" : "";
            return rows < from ? "_1row" : "_2step";
        };
        for (const Case& c : cases) {
            const auto wa = matrix(q8, c.nin, c.a, 521), wb = matrix(q8, c.nin, c.b ? c.b : 1, 522);
            const auto a = vk.adopt(wa.data(), wa.size()), b = vk.adopt(wb.data(), wb.size());
            const auto ya = vk.alloc(c.a * sizeof(float)), yb = vk.alloc((c.b ? c.b : 1) * sizeof(float));
            const backend::RowRun decode{1, 1};
            (void)backend::vulkan_kernel_times(vk);
            if (c.b) vk.matmul_group({{{q8, {a.get(), 0}}, {ya.get(), 0}, c.a}, {{q8, {b.get(), 0}}, {yb.get(), 0}, c.b}}, {xb.get(), 0}, c.nin, 1, {&decode, 1}, dtype);
            else vk.matmul({q8, {a.get(), 0}}, {xb.get(), 0}, {ya.get(), 0}, c.nin, c.a, 1, {&decode, 1}, dtype);
            const auto times = backend::vulkan_kernel_times(vk);
            if (times.empty() && !checks) {
                vk.sync();
                std::cout << "backend-vulkan: one-column Q8_0 builds unwitnessed without device timestamps\n";
                return 0;
            }
            const std::string want = want_one + build(c.nin, c.a + c.b);
            size_t products = 0;
            for (const auto& entry : times) {
                if (entry.first.rfind("matmul_", 0) != 0) continue;
                require(entry.first == want, ("a one-column Q8_0 call of " + std::to_string(c.a + c.b) + " rows " + std::to_string(c.nin) + " wide, " + backend::dtype_name(dtype) +
                                              ", expected " + want + ", got " + entry.first).c_str());
                ++products;
            }
            require(products == 1, "a one-column Q8_0 call did not witness exactly one product kernel");
            ++checks;
        }
    }
    return checks;
}

// Q8_0 decode calls of 2 to 8 columns over rows halving from 65536 to 32 move to builds of fewer rows a workgroup, from the call's own build to the last (VulkanBackend::filling_build).
// The builds' busy workgroups are a factor of two apart, as the rows are, so the sweep reaches each build of fewer rows than the call's own on any device.
size_t check_q8_filling_builds(backend::Backend& vk) {
    const uint32_t q8 = quant::GGML_TYPE_Q8_0;
    const size_t nin = 256;
    const auto x = uniform(8 * nin, 530);
    const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
    size_t checks = 0;
    for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
        const std::string kernel = dtype == backend::Dtype::int8 ? "matmul_vec_q8_i8" : "matmul_vec_q8";
        // The builds by name, most rows a workgroup first, the call's own first.
        const std::vector<std::string> order = {"", "_8col_2row", "_8col_2row_span2", "_8col_1row_span2"};
        for (size_t n : {size_t(2), size_t(5), size_t(8)}) {
            const std::string own = n == 2 ? "_2col" : "";
            size_t prev = order.size();
            std::vector<bool> seen(order.size());
            for (size_t rows = 65536; rows >= 32; rows /= 2) {
                const auto wb = matrix(q8, nin, rows, 531);
                const auto w = vk.adopt(wb.data(), wb.size());
                const auto y = vk.alloc(n * rows * sizeof(float));
                const backend::RowRun decode{n, 1};
                (void)backend::vulkan_kernel_times(vk);
                vk.matmul({q8, {w.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, rows, n, {&decode, 1}, dtype);
                const auto times = backend::vulkan_kernel_times(vk);
                if (times.empty() && !checks) {
                    vk.sync();
                    std::cout << "backend-vulkan: Q8_0 builds of fewer rows unwitnessed without device timestamps\n";
                    return 0;
                }
                size_t at = order.size(), products = 0;
                for (const auto& entry : times) {
                    if (entry.first.rfind("matmul_", 0) != 0) continue;
                    for (size_t i = 0; i < order.size(); ++i)
                        if (entry.first == kernel + (i ? order[i] : own)) at = i;
                    require(at < order.size(), ("a Q8_0 call of " + std::to_string(n) + " columns took " + entry.first).c_str());
                    ++products;
                }
                require(products == 1, "a Q8_0 decode call did not witness exactly one product kernel");
                require(prev == order.size() || at >= prev, "a smaller Q8_0 decode call took a build of more rows a workgroup");
                require(rows != 65536 || at == 0, "the largest Q8_0 decode call left its own build");
                require(rows != 32 || at == order.size() - 1, "the smallest Q8_0 decode call did not take the build of fewest rows");
                seen[at] = true;
                prev = at;
                ++checks;
            }
            // Two columns keep 8 rows a busy workgroup in their own build, in the build of 8 columns and 2 rows and in the first of two groups, so they reach only the last.
            for (size_t i = n == 2 ? 3 : 1; i < order.size(); ++i) require(seen[i], ("the sweep missed the Q8_0 build " + kernel + order[i]).c_str());
        }
    }
    return checks;
}

// Every decode column must equal the same column computed alone, through plain, grouped, residual and routed calls (docs/VULKAN.md, batch invariance).
// AGENTS.md, Tests describes the widths and tails that distinguish the kernel builds; `only_q8` checks the Q8_0 rows alone.
size_t check_decode_columns(backend::Backend& vk, backend::Dtype dtype, bool only_q8 = false) {
    const size_t widest = 256;
    // Remainders reach every narrower build, 18 and 29 columns leave the widest Q8_0 build's second group partially filled, and past 64 whole chunks run side by side beside a remainder of each width.
    const size_t widths[] = {1, 2, 3, 8, 9, 13, 16, 18, 29, 32, 33, 34, 36, 40, 48, 64, 65, 96, 127, 128, 129, 200, 256};
    const uint32_t f32 = quant::GGML_TYPE_F32, q8 = quant::GGML_TYPE_Q8_0, q40 = quant::GGML_TYPE_Q4_0, q41 = quant::GGML_TYPE_Q4_1;
    const uint32_t q50 = quant::GGML_TYPE_Q5_0, q51 = quant::GGML_TYPE_Q5_1;
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

    // Q8_0 rows also as many as the profile's q8_decode_two_rows_from, from which a one-column chunk of rows 1024 wide or wider takes the build of two rows and two steps; rows 224 wide take the narrow build.
    const size_t two_rows = backend::vulkan_device_profile(vk).q8_decode_two_rows_from;
    struct Case { uint32_t type; size_t nin; size_t nout = 300; };
    // Q8_0 rows 320, 448 and 512 wide take the paired build alone where the profile has the half-block order and the tree, and every wider build beside it.
    // Q8_0 rows 320, 640, 1280 and 5120 tall reach, on a device of 60 compute units, every build a call of 2 to 8 columns takes (VulkanBackend::filling_build).
    for (const Case& c : {Case{f32, 4096}, Case{f32, 1280}, Case{f32, 224}, Case{q8, 4096}, Case{q8, 2560}, Case{q8, 1280}, Case{q8, 224},
                          Case{q8, 2560, 320}, Case{q8, 2560, 640}, Case{q8, 2560, 1280}, Case{q8, 2560, 5120},
                          Case{q8, 320}, Case{q8, 448}, Case{q8, 512}, Case{q8, 2560, two_rows}, Case{q8, 1280, two_rows},
                          Case{q40, 4096}, Case{q40, 1280}, Case{q40, 224}, Case{q41, 4096}, Case{q41, 1280}, Case{q41, 224}, Case{q50, 4096}, Case{q50, 1280}, Case{q50, 224}, Case{q51, 4096}, Case{q51, 1280}, Case{q51, 224},
                          Case{q4k, 4096}, Case{q4k, 1280}, Case{q5k, 4096}, Case{q5k, 1280}, Case{q6k, 4096}, Case{q6k, 1280},
                          Case{quant::GGML_TYPE_TQ1_0, 512}, Case{quant::GGML_TYPE_TQ2_0, 512}}) {
        const uint32_t type = c.type;
        if (only_q8 && type != q8) continue;
        const size_t nin = c.nin, nout = c.nout, rows[3] = {nout, 37, 129};
        const auto x = uniform(widest * nin, 300 + uint32_t(nin)), base = uniform(widest * nout, 301);
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        std::vector<backend::BufferPtr> w;
        for (uint32_t i = 0; i < 3; ++i) {
            const auto bytes = matrix(type, nin, rows[i], 302 + i);
            w.push_back(vk.adopt(bytes.data(), bytes.size()));
        }
        const bool keeps_head = type == q40 || type == q41 || type == q50 || type == q51 || type == q6k;
        for (int head = 0; head <= (keeps_head ? 1 : 0); ++head) {
            try {
                // n columns of X from col0 through projection i into y, as n generated tokens.
                auto product = [&](size_t i, size_t col0, size_t n, backend::Slice y, bool add) {
                    const backend::RowRun decode{n, 1};
                    const backend::CSlice wi{w[i].get(), 0}, xs{xb.get(), col0 * nin};
                    if (add) vk.matmul_add({type, wi}, xs, y, nin, rows[i], n, {&decode, 1}, dtype);
                    else if (head) vk.matmul_logits({type, wi}, xs, y, nin, rows[i], n, {&decode, 1}, dtype);
                    else vk.matmul({type, wi}, xs, y, nin, rows[i], n, {&decode, 1}, dtype);
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
                    vk.matmul_group({{{type, {w[0].get(), 0}}, {out[0].get(), 0}, rows[0]}, {{type, {w[1].get(), 0}}, {out[1].get(), 0}, rows[1]},
                                     {{type, {w[2].get(), 0}}, {out[2].get(), 0}, rows[2]}},
                                    {xb.get(), 0}, nin, n, {&decode, 1}, dtype);
                    for (size_t i = 0; i < 3; ++i)
                        same(floats(out[i], n * rows[i]), *ones[i], n, rows[i], "a grouped projection's decode column differs from the same column alone");
                }
            } catch (const std::runtime_error&) {
                std::fprintf(stderr, "  decode columns: type %u, nin %zu%s, %s\n", type, nin, head ? ", output head" : "", backend::dtype_name(dtype));
                throw;
            }
        }
    }

    // Routed gate/up and residual down products must equal each token alone, on both sides of the grouped-build threshold.
    // The two input widths cover paired steps and their single-step tails.
    const size_t nout = 300, n_expert = 8, k = 2, tokens = 32;
    const auto scores = uniform(tokens * n_expert, 310, -3.0f, 3.0f), base = uniform(tokens * nout, 313);
    const auto sb = vk.adopt(scores.data(), scores.size() * sizeof(float));
    const auto ids = vk.alloc(tokens * k * sizeof(float)), wts = vk.alloc(tokens * k * sizeof(float));
    vk.route_experts({sb.get(), 0}, tokens, n_expert, k, true, {ids.get(), 0}, {wts.get(), 0});
    auto routing = [&](size_t first) { return backend::Backend::Routing{{ids.get(), first * k}, {wts.get(), first * k}, k, n_expert}; };
    for (size_t rin : {size_t(4096), size_t(1280)}) {
        const auto x = uniform(tokens * rin, 311), x2 = uniform(tokens * k * rin, 312);
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float)), x2b = vk.adopt(x2.data(), x2.size() * sizeof(float));
        for (uint32_t type : {f32, q8, q40, q41, q4k, q5k, q6k, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
            if (only_q8 && type != q8) continue;
            try {
                const auto g = matrix(type, rin, n_expert * nout, 314), u = matrix(type, rin, n_expert * nout, 315), d = matrix(type, rin, n_expert * nout, 316);
                const auto gb = vk.adopt(g.data(), g.size()), ub = vk.adopt(u.data(), u.size()), db = vk.adopt(d.data(), d.size());
                // Tokens first .. first + n: gate and up entries, then the down projection's rows added onto their base rows.
                auto routed = [&](size_t first, size_t n) {
                    const backend::RowRun decode{n, 1};
                    const auto go = vk.alloc(n * k * nout * sizeof(float)), uo = vk.alloc(n * k * nout * sizeof(float));
                    const auto yo = vk.alloc(n * nout * sizeof(float));
                    vk.write(*yo, 0, base.data() + first * nout, n * nout * sizeof(float));
                    vk.matmul_experts({{{type, {gb.get(), 0}}, {go.get(), 0}, nout}, {{type, {ub.get(), 0}}, {uo.get(), 0}, nout}},
                                      {xb.get(), first * rin}, rin, n, routing(first), {&decode, 1}, dtype);
                    vk.matmul_experts_add({type, {db.get(), 0}}, {x2b.get(), first * k * rin}, {yo.get(), 0}, rin, nout, n, routing(first), {&decode, 1}, dtype);
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
                std::fprintf(stderr, "  decode columns: routed type %u, nin %zu, %s\n", type, rin, backend::dtype_name(dtype));
                throw;
            }
        }
    }
    return columns;
}

// Count the disassembly's float products and reductions, distinguishing fused operations and shuffled adds.
// Exclude recognized integer-division reciprocal scaling (docs/src/backends-vulkan.md, Instruction-count checks).
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

// Timing must cover every dispatch across multiple query pools, including the work after the first pool fills.
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

// A timed backend's argmax_rows time for one row of 248320 logits, in device and host-visible memory, in slices and in one workgroup, alone and after the head's product that wrote them, as a draft row runs them; printed, held to nothing.
void print_argmax_after_head() {
    backend::BackendPtr tb = backend::make_vulkan_backend(0, true);
    const size_t vocab = 248320, width = 512, blocks = width / quant::Q8_0_BLOCK, calls = 50;
    std::vector<uint8_t> w(vocab * blocks * quant::Q8_0_TYPESIZE);
    uint32_t seed = 2026;
    for (uint8_t& byte : w) byte = (uint8_t)((seed = seed * 1664525u + 1013904223u) >> 24) & 0x3f;
    const std::vector<float> x = uniform(width, 11, -1.0f, 1.0f);
    const auto wb = tb->adopt(w.data(), w.size()), xb = tb->adopt(x.data(), width * sizeof(float));
    const auto ids = tb->alloc(4 * sizeof(float), backend::Memory::host_visible), prior = tb->alloc(4 * sizeof(float), backend::Memory::host_visible);
    const uint32_t zero[4] = {};
    tb->write(*prior, 0, zero, sizeof(zero));
    const backend::RowRun decode{1, 1};
    for (bool host : {false, true}) {
        const auto logits = tb->alloc(vocab * sizeof(float), host ? backend::Memory::host_visible : backend::Memory::device);
        for (bool whole : {true, false})
        for (bool head : {false, true}) {
            tb->matmul_logits({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {logits.get(), 0}, width, vocab, 1, {&decode, 1});
            tb->sync();
            backend::vulkan_kernel_times(*tb);
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < calls; ++i) {
                if (head) tb->matmul_logits({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {logits.get(), 0}, width, vocab, 1, {&decode, 1});
                tb->unordered(whole);
                tb->argmax_rows({ids.get(), 0}, {logits.get(), 0}, 1, vocab, {prior.get(), 0});
                tb->unordered(false);
            }
            tb->sync();
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / double(calls);
            double argmax = 0, product = 0;
            for (const auto& t : backend::vulkan_kernel_times(*tb)) (t.first == "argmax_rows" ? argmax : product) += t.second;
            std::cout << "backend-vulkan: timed argmax_rows over " << vocab << (host ? " host-visible" : " device") << " logits, " << (whole ? "one workgroup a row, " : "in slices, ") << (head ? "after the head's product" : "alone") << ": "
                      << argmax / double(calls) << " ms a row" << (head ? ", the head's product " + std::to_string(product / double(calls)) + " ms" : std::string()) << ", wall " << wall << " ms a call" << std::endl;
        }
    }
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

// Expected Q8_0 float operations for a build's columns, rows, steps and reduction forms.
// The count derivation is in docs/src/backends-vulkan.md, Instruction-count checks.
struct DecodeOps {
    size_t products = 0, mul = 0, lane_add = 0, add = 0, reduce = 0;
    bool tree = false;
};
// A Q8_0 decode build as the backend made it, read from the first line of its representation (vulkan_kernel_representations): the columns and rows a subgroup takes, the steps of weights a lane loads before using any, and its forms.
struct DecodeBuild {
    unsigned cols = 0, rows = 0, steps = 0, halves = 1;
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
    if (line.find(" halves=") != std::string::npos && !field("halves", b.halves)) return false;
    b.tree = tree != 0;
    b.half = half != 0;
    return b.cols && b.rows && b.steps && (b.halves == 1 || (b.halves == 2 && b.tree && b.half && b.cols == 1));
}
// A two-row build of the Q4 and K-quant row kernels, read from the first line of its representation: the columns it holds, the rows a cluster takes and the cluster width its reduction transposes (0 for none).
bool row_build(const std::string& text, unsigned& cols, unsigned& rows, unsigned& tree) {
    const std::string head = "; row_build";
    const std::string line = text.substr(0, text.find('\n'));
    if (line.compare(0, head.size(), head) != 0) return false;
    return header_field(line, "cols", cols) && header_field(line, "rows", rows) && header_field(line, "tree", tree) && cols && rows;
}

// The adds of a transposed reduction of n values over a cluster of w lanes: n / 2, n / 4 .. down to one value a lane, then one add a remaining level.
size_t tree_adds(size_t n, size_t w) {
    size_t adds = 0;
    for (size_t off = w / 2; off > 0; off /= 2) adds += n > 1 ? (n /= 2) : 1;
    return adds;
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
    // A paired build (halves 2) takes the tree's first five levels, its halves holding other rows.
    for (size_t l = 0; l < (b.halves > 1 ? 5u : 6u); ++l) n.reduce += l < log ? rc >> (l + 1) : 1;
    n.reduce += rc > 64 ? rc / 64 : 1;
    return n;
}

// Compare each row-kernel build's operation counts with its one-column baseline and shape (docs/src/backends-vulkan.md, Instruction-count checks).
// Counts detect contraction changes; decode-column identity also catches reassociation with unchanged counts.
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
    const std::string vec = "matmul_vec_q8", vec8 = "matmul_vec_q8_i8", one_suffix = "_1col", grouped_suffix = "_grouped";
    struct TwoRow {
        std::string kernel, name;
        size_t cols;
        unsigned rows;
        FloatOps got, ref;
        unsigned tree;
    };
    std::vector<TwoRow> two_row;
    for (const auto& o : ops) {
        if (!ends(o.first, grouped_suffix)) continue;
        const std::string wide = o.first.substr(0, o.first.size() - grouped_suffix.size());
        const FloatOps* w = find(wide);
        if (!w || wide == vec || wide == vec8) continue;
        // A driver may keep the grouped build's column loop rolled, leaving it one column's operations, the one-column build's.
        const FloatOps* col1 = find(wide + one_suffix);
        if (!(o.second == *w) && !(col1 && o.second == *col1))
            throw std::runtime_error("a grouped build's float multiplies and adds differ from its wide build's and its one-column build's: " + o.first);
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
            // The Q8_0 decode kernel's builds of fewer rows a workgroup are named by their columns, rows and column groups.
            if (!cols && (kernel == vec || kernel == vec8) && b.compare(0, kernel.size() + 1, kernel + "_") == 0) {
                unsigned c = 0, r = 0, s = 0;
                char tail[2];
                const std::string rest = b.substr(kernel.size() + 1);
                if (std::sscanf(rest.c_str(), "%ucol_%urow_span%u%1s", &c, &r, &s, tail) == 3 || std::sscanf(rest.c_str(), "%ucol_%urow%1s", &c, &r, tail) == 2) cols = c;
            }
            // The Q8_0 decode kernel's other one-column builds are held to their shapes like the wider builds.
            const bool two_row_one = (kernel == vec || kernel == vec8) && (b == one.first + "_1row" || b == one.first + "_2step" || b == one.first + "_paired");
            if (cols < 2 && !two_row_one) continue;
            const FloatOps& got = build.second;
            auto fail = [&](const char* what) { throw std::runtime_error(std::string(what) + ": " + b); };
            // A transposed reduction's adds come plain or over shuffled lanes as the driver forms them, so its kinds count both as adds.
            unsigned head_cols = 0, head_rows = 0, head_tree = 0;
            for (const auto& kr : representations)
                if (kr.first == b) row_build(kr.second, head_cols, head_rows, head_tree);
            auto kinds = [&](const FloatOps& o) { const unsigned k = o.kinds(); return head_tree && (k & 16u) ? (k & ~16u) | 8u : k; };
            if (kinds(got) != kinds(ref)) fail("a build's float multiply and add kinds differ from its one-column build's");
            if (kernel == vec || kernel == vec8) {
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
            unsigned two_cols = 0, rows = 1, tree = 0;
            for (const auto& kr : representations)
                if (kr.first == b && row_build(kr.second, two_cols, rows, tree) && two_cols != cols) fail("a two-row build whose shape gives other columns than its name");
            if (rows > 1) {
                two_row.push_back({kernel, b, cols, rows, got, ref, tree});
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
    // Two-row builds differ by whole copies of one row and column's products, retaining the one-column build's proportion of fused products.
    // A transposed reduction's adds, plain or over shuffled lanes as the driver forms them, come off whole with each value a lane stores, leaving the products' adds.
    auto counts = [&](const TwoRow& b, size_t (&g)[5]) {
        const size_t v[5] = {b.got.mul, b.got.mad, b.got.fused, b.got.add, b.got.lane_add};
        std::copy(v, v + 5, g);
        if (!b.tree) return true;
        const size_t nv = size_t(b.rows) * b.cols, reduce = tree_adds(nv, b.tree) + (nv >= b.tree ? nv / b.tree : 1);
        if (g[3] + g[4] < reduce) return false;
        g[3] = g[3] + g[4] - reduce, g[4] = 0;
        return true;
    };
    auto reference = [](const TwoRow& b, size_t (&r)[5]) {
        const size_t v[5] = {b.ref.mul, b.ref.mad, b.ref.fused, b.ref.add + (b.tree ? b.ref.lane_add : 0), b.tree ? 0 : b.ref.lane_add};
        std::copy(v, v + 5, r);
    };
    std::map<std::string, std::vector<size_t>> column_ops, row_ops;
    for (int pass = 0; pass < 2; ++pass)
        for (size_t i = 0; i < two_row.size(); ++i)
            for (size_t j = 0; j < two_row.size(); ++j) {
                const TwoRow &a = two_row[i], &c = two_row[j];
                if (a.kernel != c.kernel || c.cols <= a.cols || (pass == 0) != (a.rows == c.rows)) continue;
                auto fail = [&](const char* what) { throw std::runtime_error(std::string(what) + ": " + a.name + " and " + c.name); };
                if (a.tree != c.tree) fail("two-row builds of one kernel with other reductions");
                size_t ga[5], gc[5], r[5];
                if (!counts(a, ga) || !counts(c, gc)) fail("a transposed two-row build with fewer adds than its reduction takes");
                reference(a, r);
                if (pass == 0) {
                    const size_t step = size_t(a.rows) * (c.cols - a.cols);
                    size_t per[5];
                    for (int k = 0; k < 5; ++k) {
                        if (gc[k] < ga[k] || (gc[k] - ga[k]) % step != 0 || (gc[k] - ga[k]) / step > r[k])
                            fail("two-row builds' float multiplies and adds differ by other than whole copies of a row and column's products");
                        per[k] = (gc[k] - ga[k]) / step;
                    }
                    if (per[1] * r[2] != per[2] * r[1]) fail("a two-row build fuses a column's products in another proportion than its one-column build");
                    auto seen = column_ops.emplace(a.kernel, std::vector<size_t>(per, per + 5));
                    if (!std::equal(per, per + 5, seen.first->second.begin())) fail("two-row builds of one kernel differ by other products a column in another pair");
                } else {
                    // Builds of other rows also differ by whole copies of one row's weight operations, which every column of the row shares, no more than the one-column build holds.
                    const auto per = column_ops.find(a.kernel);
                    if (per == column_ops.end()) fail("two-row builds of other rows without two of the same rows to give a column's products");
                    if (c.rows <= a.rows) fail("a wider two-row build with fewer rows");
                    const size_t products = size_t(c.rows) * c.cols - size_t(a.rows) * a.cols, rows = c.rows - a.rows;
                    size_t w[5];
                    for (int k = 0; k < 5; ++k) {
                        const size_t want = ga[k] + per->second[k] * products;
                        if (gc[k] < want || (gc[k] - want) % rows != 0 || (gc[k] - want) / rows > r[k])
                            fail("two-row builds of other rows differ by other than whole copies of a row and column's products and of a row's weight operations");
                        w[k] = (gc[k] - want) / rows;
                    }
                    auto seen = row_ops.emplace(a.kernel, std::vector<size_t>(w, w + 5));
                    if (!std::equal(w, w + 5, seen.first->second.begin())) fail("two-row builds of one kernel differ by other weight operations a row in another pair");
                }
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
    const std::string one1 = code("cols=1 rows=1 steps=4 tree=0 half=1", 12, 7, 6);
    const std::string one2 = code("cols=1 rows=2 steps=2 tree=0 half=1", 16, 10, 12);
    const std::string two = code("cols=2 rows=2 steps=1 tree=1 half=1", 16, 11, 5);
    // The paired build: two rows a lane as the one-column build, half-block order, and the tree's five levels within a half.
    const std::string paired = code("cols=1 rows=2 steps=1 tree=1 half=1 halves=2", 8, 6, 4);
    // A build of fewer rows a workgroup, held by the name of its columns, rows and column groups.
    const std::string fewer = code("cols=4 rows=1 steps=1 tree=1 half=1", 16, 11, 5);
    require(check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_1col_1row", one1}, {"matmul_vec_q8_1col_2step", one2}, {"matmul_vec_q8_2col", two},
                               {"matmul_vec_q8_1col_paired", paired}, {"matmul_vec_q8_8col_1row_span2", fewer}}).decode == 5,
            "Q8 separate products fail their captured shape counts");
    {
        bool refused = false;
        try { check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_8col_1row_span2", code("cols=4 rows=1 steps=1 tree=1 half=1", 15, 11, 5)}}); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "a Q8 build of fewer rows a workgroup escaped its shape counts");
    }
    // The same build reduced through all six levels, as if its halves were joined, is refused.
    for (const std::string& wrong : {code("cols=1 rows=2 steps=1 tree=1 half=1 halves=2", 8, 6, 5), code("cols=1 rows=2 steps=1 tree=1 half=1 halves=2", 8, 7, 4)}) {
        bool refused = false;
        try { check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_1col_paired", wrong}}); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "a paired Q8 build's changed reduction escaped its shape counts");
    }
    for (const std::string& wrong : {code("cols=2 rows=2 steps=1 tree=1 half=1", 15, 11, 5),
                                     code("cols=2 rows=2 steps=1 tree=1 half=1", 16, 10, 5),
                                     two + "v_fma_f32 v0, v1, v2, v3\n"}) {
        bool refused = false;
        try { check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_2col", wrong}}); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "Q8 changed arithmetic escaped its shape counts");
    }
    for (const std::string& wrong : {code("cols=1 rows=2 steps=2 tree=0 half=1", 14, 10, 12), code("cols=1 rows=2 steps=2 tree=0 half=1", 16, 9, 12),
                                     one2 + "v_fma_f32 v0, v1, v2, v3\n", code("cols=1 rows=1 steps=4 tree=0 half=1", 12, 6, 6)}) {
        bool refused = false;
        try { check_contraction({{"matmul_vec_q8_1col", one}, {"matmul_vec_q8_1col_2step", wrong}}); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "a Q8 one-column build's changed arithmetic escaped its shape counts");
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

// Witness independently named kernels and require each refusal to leave recording and output unchanged, with a valid call afterwards (AGENTS.md, Tests).
// The first pass retires operand slots before refusal; the second retains a previously filled output.
size_t check_weight_dispatch() {
    const auto owner = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *owner;
    const backend::DeviceProfile p = backend::vulkan_device_profile(vk);
    struct Expected { uint32_t type; const char* row; const char* integer_tile; };
    const Expected cases[] = {{0, "matmul_row_f32", ""}, {1, "matmul_row_wf16", ""}, {30, "matmul_row_wbf16", ""},
        {8, "matmul_row_q8w", "matmul_tile_q8"},
        {2, "matmul_row_q4", "matmul_tile_q"}, {3, "matmul_row_q4", "matmul_tile_q"},
        {6, "matmul_row_q50", "matmul_tile_q50"}, {7, "matmul_row_q5", "matmul_tile_q5"},
        {quant::GGML_TYPE_TQ1_0, "matmul_row_tq1_0", "matmul_tile_tq1_0_dot"},
        {quant::GGML_TYPE_TQ2_0, "matmul_row_tq2_0", "matmul_tile_tq2_0_dot"},
        {12, "matmul_row_k4", "matmul_tile_q"}, {13, "matmul_row_k5", "matmul_tile_q"},
        {14, "matmul_row_k", "matmul_tile_q6"}, {39, "matmul_row_mxfp4", "matmul_tile_q8mx"}};
    size_t checks = 0;
    for (const Expected& e : cases) {
        if (e.type == 39 && !vk.supports_type(e.type)) continue;
        require(vk.supports_type(e.type), "an implemented weight type lost device support");
        const bool tq = e.type == quant::GGML_TYPE_TQ1_0 || e.type == quant::GGML_TYPE_TQ2_0;
        for (size_t nin : {size_t(96), size_t(256), size_t(4096)}) {
            if (nin == 96 && e.type != 2 && e.type != 6 && e.type != 7 && e.type != 8) continue;
            const size_t nout = 3;
            const auto bytes = matrix(e.type, nin, nout, 401);
            const auto xf = uniform(nin, 402);
            const uint32_t id = 0;
            const auto w = vk.adopt(bytes.data(), bytes.size()), x = vk.adopt(xf.data(), xf.size() * sizeof(float));
            const auto ids = vk.adopt(&id, sizeof(id)), y = vk.alloc(nout * sizeof(float));
            for (bool routed : {false, true}) {
                const bool floating = e.type == 0 || e.type == 1 || e.type == 30;
                const bool fast = floating || e.type == 8;
                const size_t dense_from = fast ? (nin < p.tile_narrow_nin ? p.tile_from_8bit_narrow : p.tile_from_8bit)
                                              : (nin < p.tile_narrow_nin ? p.tile_from_other_narrow : p.tile_from_other);
                const size_t from = !routed ? dense_from : e.type == 2 || e.type == 3 || e.type == 6 || e.type == 7 || tq ? p.moe_tile_from_q4
                                  : e.type == 12 ? p.moe_tile_from_q4k : e.type == 13 ? p.moe_tile_from_q5k : p.moe_tile_from;
                for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::int8}) {
                    // int8 reaches a device only where it is native; floating weights keep F32 products and MXFP4 keeps its 16-bit dot.
                    if (dtype == backend::Dtype::int8 && !p.prefer_integer_dot) continue;
                    const bool i8 = dtype == backend::Dtype::int8 && !floating && e.type != 39 && !tq;
                    std::vector<size_t> extents = {size_t(1), from - 1, from};
                    if (tq) extents.push_back(from + 1);
                    for (size_t extent : extents) {
                        const bool tile = dtype == backend::Dtype::bf16 || extent >= from;
                        const bool integer = (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8) && p.prefer_integer_dot && !floating &&
                                             !(routed && e.type == 39);
                        std::string expected;
                        if (tile) {
                            if (tq)
                                expected = integer ? e.integer_tile : e.type == quant::GGML_TYPE_TQ1_0 ? "matmul_tile_tq1_0" : "matmul_tile_tq2_0";
                            else
                                expected = integer ? e.integer_tile : e.type == 39 ? "matmul_tile_mxfp4" :
                                           e.type == 1 ? "matmul_tile_wf16" : e.type == 30 ? "matmul_tile_wbf16" : "matmul_tile";
                            if (dtype == backend::Dtype::bf16) expected += "_bf16";
                            if (i8) expected += "i8";
                        } else if (dtype == backend::Dtype::f32 && !floating) {
                            expected = e.type == 39 ? "matmul_row_mxfp4_float_x" :
                                       tq ? e.row : "matmul_row_float_x";
                        } else if (i8) {
                            expected = e.type == 8 ? "matmul_vec_q8_i8" : std::string(e.row) + "_i8";
                        } else {
                            expected = e.type == 8 && nin == 96 ? "matmul_row" : e.row;
                            if (e.type == 8 && p.prefer_integer_dot) expected = "matmul_vec_q8";
                            else if ((tq && p.lookup_integer_dot) || (e.type == 39 && p.lookup_integer_dot) ||
                                     (p.prefer_integer_dot && !floating && e.type != 8 && e.type != 39 && !tq)) expected += "_dot";
                        }
                        (void)backend::vulkan_kernel_times(vk);
                        const backend::RowRun run{1, extent};
                        if (routed) {
                            const backend::Backend::Routing routing{{ids.get(), 0}, {}, 1, 1};
                            vk.matmul_experts({{{e.type, {w.get(), 0}}, {y.get(), 0}, nout}}, {x.get(), 0}, nin, 1, routing, {&run, 1}, dtype);
                        } else {
                            vk.matmul({e.type, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, nin, nout, 1, {&run, 1}, dtype);
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
                            // A routed tile of few entries takes the narrow build of its tall module, or of its one module (kNarrowCols).
                            const bool narrow = routed && tile && integer && (entry.first == expected + "_tall_narrow" || entry.first == expected + "_narrow");
                            require(entry.first == expected || entry.first == expected + "_1col" || entry.first == expected + "_1col_1row" ||
                                        entry.first == expected + "_1col_2step" || entry.first == expected + "_1col_paired" || entry.first == expected + "_small" || narrow,
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
    if (p.prefer_integer_dot) for (uint32_t type : {6u, 7u}) {
        const bool symmetric = type == 6;
        const size_t block_bytes = symmetric ? 22 : 24, qoff = symmetric ? 6 : 8;
        // A tall output with a final partial tile reaches the separate Q5 pipeline
        // specialization on the tuned integer-dot device. Raw affine blocks and a
        // one-hot input keep this oracle independent of the runtime's decoder.
        const size_t nin = 256, nout = 65537, stride = 8 * block_bytes;
        std::vector<uint8_t> packed(nout * stride, 0);
        std::vector<float> input(nin, 0), output(nout + 2, 19.0f);
        input.back() = 0.5f;
        for (size_t r = 0; r < nout; ++r) {
            const uint8_t code = uint8_t(r % 32);
            for (size_t k = 0; k < 8; ++k) {
                auto* b = packed.data() + r * stride + k * block_bytes;
                b[1] = r % 2 ? 0xbc : 0x3c; // scale -1 or +1
                if (!symmetric) b[3] = 0xbc; // Q5_1 minimum -1
                std::fill(b + qoff - 4, b + qoff, code & 16 ? uint8_t(255) : uint8_t(0));
                std::fill(b + qoff, b + block_bytes, uint8_t((code & 15) * 17));
            }
        }
        const uint32_t id = 0;
        const auto w = vk.adopt(packed.data(), packed.size()), x = vk.adopt(input.data(), nin * sizeof(float));
        const auto ids = vk.adopt(&id, sizeof(id)), y = vk.alloc(output.size() * sizeof(float));
        for (auto dtype : {backend::Dtype::f16, backend::Dtype::int8}) for (bool routed : {false, true}) {
            vk.write(*y, 0, output.data(), output.size() * sizeof(float));
            const backend::RowRun run{1, routed ? p.moe_tile_from_q4 : p.tile_from_other_narrow};
            (void)backend::vulkan_kernel_times(vk);
            if (routed) {
                const backend::Backend::Routing routing{{ids.get(), 0}, {}, 1, 1};
                vk.matmul_experts({{{type, {w.get(), 0}}, {y.get(), 1}, nout}}, {x.get(), 0}, nin, 1, routing, {&run, 1}, dtype);
            } else vk.matmul({type, {w.get(), 0}}, {x.get(), 0}, {y.get(), 1}, nin, nout, 1, {&run, 1}, dtype);
            const auto times = backend::vulkan_kernel_times(vk);
            const std::string expected = std::string(symmetric ? "matmul_tile_q50" : "matmul_tile_q5") + (dtype == backend::Dtype::int8 ? "i8" : "");
            size_t products = 0;
            for (const auto& entry : times) {
                if (entry.first.rfind("matmul_", 0) != 0 || entry.first.rfind("matmul_reduce", 0) == 0) continue;
                require(entry.first == expected || entry.first == expected + "_tall" || (routed && entry.first == expected + "_tall_narrow"), "Q5 affine tail did not use its integer tile");
                std::cout << "backend-vulkan: Q5 affine tail " << (routed ? "routed " : "dense ") << entry.first << "\n";
                ++products;
            }
            require(products == 1, "Q5 affine tail has no unique tile witness");
            vk.read(*y, 0, output.data(), output.size() * sizeof(float));
            require(output.front() == 19.0f && output.back() == 19.0f, "Q5 affine tile changed a guard");
            for (size_t r = 0; r < nout; ++r) {
                const float want = 0.5f * ((r % 2 ? -1.0f : 1.0f) * float(int(r % 32) - (symmetric ? 16 : 0)) - (symmetric ? 0.0f : 1.0f));
                require(std::abs(output[r + 1] - want) <= std::abs(want) / 32767.0f + 2e-6f,
                        "Q5 affine tile tail differs from independent signed scale/high-plane values");
            }
            checks += nout;
        }
    }
    for (uint32_t type : {4u, 25u, 42u, 43u, UINT32_MAX}) {
        require(!vk.supports_type(type), "metadata-only or unknown weight type acquired a kernel");
        const std::string expected = "vulkan: unsupported matrix type " + std::to_string(type) +
                                     " (docs/VULKAN.md lists the types the kernels decode)";
        bool refused = false;
        try { vk.matmul({type, {}}, {}, {}, 256, 1, 1); }
        catch (const std::runtime_error& e) { refused = e.what() == expected; }
        require(refused, "unsupported weight type lost its early refusal text");
        ++checks;
    }
    return checks;
}

// The finite value from its integer significand and exponent, independently of the runtime's conversion helpers.
float half_weight_value(uint32_t type, uint16_t bits) {
    const unsigned fraction_bits = type == 1 ? 10 : 7, bias = type == 1 ? 15 : 127;
    const unsigned exponent = (bits & 0x7fffu) >> fraction_bits, fraction = bits & ((1u << fraction_bits) - 1);
    require(exponent != 2 * bias + 1, "half weight oracle received a nonfinite value");
    const double magnitude = std::ldexp(double(fraction + (exponent ? 1u << fraction_bits : 0)),
                                       int(exponent ? exponent : 1) - int(bias) - int(fraction_bits));
    return std::copysign(float(magnitude), bits & 0x8000u ? -1.0f : 1.0f);
}

size_t check_half_weights(backend::Backend& vk) {
    const auto profile = backend::vulkan_device_profile(vk);
    size_t checked = 0;
    for (uint32_t type : {1u, 30u}) {
        require(vk.supports_type(type), "half weight format has no Vulkan kernel");
        std::vector<uint8_t> bytes(1, 0xa5);
        std::vector<float> wide;
        for (unsigned bits = 0; bits < 65536; ++bits) {
            const unsigned mask = type == 1 ? 0x7c00u : 0x7f80u;
            if ((bits & mask) == mask) continue;
            bytes.push_back(uint8_t(bits));
            bytes.push_back(uint8_t(bits >> 8));
            wide.push_back(half_weight_value(type, uint16_t(bits)));
        }
        const size_t n = wide.size();
        const float input = 0.5f;
        const uint32_t id = 0;
        const auto weights = vk.adopt(bytes.data() + 1, n * 2), control = vk.adopt(wide.data(), n * sizeof(float));
        const auto x = vk.adopt(&input, sizeof(input)), y = vk.alloc(n * sizeof(float)), ref = vk.alloc(n * sizeof(float));
        std::vector<float> actual(n), expected(n);
        vk.embed({y.get(), 0}, {type, {weights.get(), 0}}, n, 1, &id, 1);
        vk.read(*y, 0, actual.data(), n * sizeof(float));
        checked += exact(wide, actual, "finite half embedding differs from independent widening bits");
        const uint32_t ids[] = {0, 1, UINT32_MAX, 0};
        const auto device_ids = vk.adopt(ids, sizeof(ids)), gathered = vk.alloc(4 * n * sizeof(float));
        std::vector<float> device_rows(4 * n, 17.0f), expected_rows(4 * n, 0.0f);
        vk.write(*gathered, 0, device_rows.data(), device_rows.size() * sizeof(float));
        std::copy(wide.begin(), wide.end(), expected_rows.begin());
        std::copy(wide.begin(), wide.end(), expected_rows.begin() + 3 * n);
        vk.embed_ids({gathered.get(), 0}, {type, {weights.get(), 0}}, n, 1, {device_ids.get(), 0}, 4);
        vk.read(*gathered, 0, device_rows.data(), device_rows.size() * sizeof(float));
        checked += exact(expected_rows, device_rows, "finite half device-ID embedding or invalid-ID zero row differs");
        vk.matmul({type, {weights.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 1, n, 1);
        vk.matmul({0, {control.get(), 0}}, {x.get(), 0}, {ref.get(), 0}, 1, n, 1);
        vk.read(*y, 0, actual.data(), n * sizeof(float));
        vk.read(*ref, 0, expected.data(), n * sizeof(float));
        checked += exact(expected, actual, "finite half rows differ from the device's widened F32 rows");
        if (type == 1) {
            backend::CpuBackend cpu;
            cpu.set_threads(1);
            const auto cw = cpu.adopt(wide.data(), n * sizeof(float)), cx = cpu.adopt(&input, sizeof(input));
            const auto cy = cpu.alloc(n * sizeof(float), backend::Memory::device);
            cpu.matmul({0, {cw.get(), 0}}, {cx.get(), 0}, {cy.get(), 0}, 1, n, 1);
            cpu.read(*cy, 0, expected.data(), n * sizeof(float));
            checked += exact(expected, actual, "finite F16 weight products, including subnormals, differ from CPU F32");
        }
        for (size_t nin : {size_t(7), size_t(32), size_t(65), size_t(256), size_t(4096)}) {
            const size_t nout = 7, experts = 3, k = 2;
            auto packed = matrix(type, nin, experts * nout, 523);
            std::vector<float> values(packed.size() / 2);
            for (size_t i = 0; i < values.size(); ++i) {
                if (i < nin) {
                    const uint16_t bits = uint16_t((i % 3 ? 0x8000u : 0) | (1 + i % (type == 1 ? 1023 : 127)));
                    packed[2 * i] = uint8_t(bits);
                    packed[2 * i + 1] = uint8_t(bits >> 8);
                }
                values[i] = half_weight_value(type, uint16_t(packed[2 * i] | (unsigned(packed[2 * i + 1]) << 8)));
            }
            const auto hw = vk.adopt(packed.data(), packed.size()), fw = vk.adopt(values.data(), values.size() * sizeof(float));
            for (size_t batch : {size_t(1), size_t(3), size_t(9)}) {
                const size_t entries = batch * k;
                const auto inputs = uniform(entries * nin, 524);
                std::vector<uint32_t> ids(entries);
                std::vector<float> scales(entries);
                for (size_t i = 0; i < entries; ++i) { ids[i] = uint32_t(i % experts); scales[i] = i % k ? 0.75f : 0.25f; }
                const auto xb = vk.adopt(inputs.data(), inputs.size() * sizeof(float));
                const auto ib = vk.adopt(ids.data(), ids.size() * sizeof(uint32_t)), sb = vk.adopt(scales.data(), scales.size() * sizeof(float));
                const backend::Backend::Routing route{{ib.get(), 0}, {sb.get(), 0}, k, experts};
                for (auto dtype : {backend::Dtype::f16, backend::Dtype::f32, backend::Dtype::bf16}) {
                    if (dtype == backend::Dtype::bf16 && !vk.emulates_dtype(dtype)) continue;
                    for (int op = 0; op < 6; ++op) {
                        const size_t columns = op == 4 ? entries : batch;
                        const bool paired = op == 3 || op == 4;
                        const size_t first = 64, second = first + ((columns * nout + 63) & ~size_t(63));
                        const size_t end = paired ? second + columns * (nout - 2) : first + columns * nout;
                        const std::vector<float> initial(end + 64, 0.125f);
                        const auto a = vk.adopt(initial.data(), initial.size() * sizeof(float));
                        const auto b = vk.adopt(initial.data(), initial.size() * sizeof(float));
                        const size_t from = op >= 4 ? profile.moe_tile_from : backend::tile_from_for(profile, true, nin);
                        const backend::RowRun run_rows{batch, batch == 9 ? from : 1};
                        auto run = [&](uint32_t t, const backend::BufferPtr& w, const backend::BufferPtr& out) {
                            const backend::CSlice ws{w.get(), 0}, xs{xb.get(), 0};
                            const backend::Slice ys{out.get(), first};
                            const backend::RowRuns rows{&run_rows, 1};
                            if (op == 0) vk.matmul({t, ws}, xs, ys, nin, nout, batch, rows, dtype);
                            else if (op == 1) vk.matmul_add({t, ws}, xs, ys, nin, nout, batch, rows, dtype);
                            else if (op == 2) vk.matmul_logits({t, ws}, xs, ys, nin, nout, batch, rows, dtype);
                            else if (op == 3) vk.matmul_group({{{t, ws}, ys, nout}, {{0, {fw.get(), 0}}, {out.get(), second}, nout - 2}},
                                                             xs, nin, batch, rows, dtype);
                            else if (op == 4) vk.matmul_experts({{{t, ws}, ys, nout}, {{t, ws}, {out.get(), second}, nout - 2}},
                                                               xs, nin, batch, route, rows, dtype);
                            else vk.matmul_experts_add({t, ws}, xs, ys, nin, nout, batch, route, rows, dtype);
                        };
                        (void)testq::take_matrix_paths(vk);
                        run(type, hw, a);
                        require(testq::take_matrix_paths(vk) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"},
                                "half weights reported quantized activation arithmetic");
                        run(0, fw, b);
                        std::vector<float> result(initial.size()), reference(initial.size());
                        vk.read(*a, 0, result.data(), result.size() * sizeof(float));
                        vk.read(*b, 0, reference.data(), reference.size() * sizeof(float));
                        for (size_t i = 0; i < result.size(); ++i)
                            if (i < first || i >= end || (paired && i >= first + columns * nout && i < second))
                                require(result[i] == initial[i], "half product wrote outside its output");
                        if (std::memcmp(result.data(), reference.data(), result.size() * sizeof(float)) != 0)
                            std::fprintf(stderr, "half weight mismatch: type %u, width %zu, batch %zu, op %d, dtype %d\n", type, nin, batch, op, int(dtype));
                        checked += exact(reference, result, "half product differs from exactly widened F32 storage");
                    }
                }
            }
        }
    }
    return checked;
}

size_t check_refusals(backend::Backend& vk) {
    require(!vk.implements(backend::Op::mixed_experts), "Vulkan advertises unsupported mixed routed projections");
    const backend::DeviceProfile prof = backend::vulkan_device_profile(vk);
    const bool integer_dot = prof.prefer_integer_dot;
    const uint32_t q8 = quant::GGML_TYPE_Q8_0, f32 = quant::GGML_TYPE_F32, i16 = 25;   // Valid storage without an execution kernel.
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
        vk.matmul({q8, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, nin, nout, rows);
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
        cpu.matmul({q8, {cw.get(), 0}}, {cx.get(), 0}, {cy.get(), 0}, nin, nout, rows, {}, backend::Dtype::f32);
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
        vk.matmul({i16, at(wb)}, at(xb), out(rows * nout), nin, nout, rows);
    });
    refused("matmul of weights shorter than the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size() - row_bytes), xb = floats(xf);
        vk.matmul({q8, at(wb)}, at(xb), out(rows * nout), nin, nout, rows);
    });
    refused("matmul of a row ending inside a block accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul({q8, at(wb)}, at(xb), out(rows * nout), partial, nout, rows);
    });
    refused("matmul_add of weights shorter than the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size() - row_bytes), xb = floats(xf);
        vk.matmul_add({q8, at(wb)}, at(xb), out(rows * nout), nin, nout, rows);
    });
    refused("matmul_group with a short second projection accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), ws = in(wq.data(), wq.size() - row_bytes), xb = floats(xf);
        vk.matmul_group({{{q8, at(wb)}, out(rows * nout), nout}, {{q8, at(ws)}, out(rows * nout), nout}}, at(xb), nin, rows);
    });
    refused("matmul with X short of its second run accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = in(xf.data(), nin * sizeof(float));
        vk.matmul({q8, at(wb)}, at(xb), out(rows * nout), nin, nout, rows, {two_calls, 2});
    });
    refused("matmul with row runs out of order accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul({q8, at(wb)}, at(xb), out(rows * nout), nin, nout, rows, {disordered, 3});
    });
    refused("matmul with row runs past the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul({q8, at(wb)}, at(xb), out(rows * nout), nin, nout, rows, {beyond, 2});
    });
    refused("matmul with row runs short of the call accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul({q8, at(wb)}, at(xb), out(rows * nout), nin, nout, rows, {short_of, 2});
    });
    refused("matmul_group with row runs out of order accepted", [&](const Out& out) {
        const auto wb = in(wq.data(), wq.size()), xb = floats(xf);
        vk.matmul_group({{{q8, at(wb)}, out(rows * nout), nout}, {{q8, at(wb)}, out(rows * nout), nout}}, at(xb), nin, rows, {disordered, 3});
    });

    // The routed products, over n_expert stacked matrices.
    auto routed = [&](const char* what, uint32_t type, size_t stack_bytes, size_t width, backend::RowRuns runs) {
        refused(what, [&](const Out& out) {
            const auto sb = in(stack.data(), stack_bytes), xb = floats(xf), idb = in(ids.data(), ids.size() * sizeof(uint32_t)), wtb = floats(weights);
            vk.matmul_experts({{{type, at(sb)}, out(entries * nout), nout}}, at(xb), width, rows, {at(idb), at(wtb), k, n_expert}, runs);
        });
        refused(what, [&](const Out& out) {
            const auto sb = in(stack.data(), stack_bytes), xb = floats(x2f), idb = in(ids.data(), ids.size() * sizeof(uint32_t)), wtb = floats(weights);
            vk.matmul_experts_add({type, at(sb)}, at(xb), out(rows * nout), width, nout, rows, {at(idb), at(wtb), k, n_expert}, runs);
        });
    };
    routed("routed product of a type without a kernel accepted", i16, stack.size(), nin, {});
    routed("routed product of a stack short of its experts accepted", q8, stack.size() - row_bytes, nin, {});
    routed("routed product of a row ending inside a block accepted", q8, stack.size(), partial, {});
    routed("routed product with row runs out of order accepted", q8, stack.size(), nin, {disordered, 3});

    const uint32_t first[1] = {0}, past[1] = {uint32_t(nrows)};
    refused("embedding of a type without a kernel accepted", [&](const Out& out) {
        const auto tb = floats(tf);
        vk.embed(out(nin), {i16, at(tb)}, nin, nrows, first, 1);
    });
    refused("F32 embedding table shorter than its rows accepted", [&](const Out& out) {
        const auto tb = floats(tf);
        vk.embed(out(nin), {f32, at(tb)}, nin, nrows + 1, first, 1);
    });
    refused("Q8_0 embedding table shorter than its rows accepted", [&](const Out& out) {
        const auto tb = in(tq.data(), tq.size());
        vk.embed(out(nin), {q8, at(tb)}, nin, nrows + 1, first, 1);
    });
    refused("embedding row ending inside a block accepted", [&](const Out& out) {
        const auto tb = in(tq.data(), tq.size());
        vk.embed(out(partial), {q8, at(tb)}, partial, nrows, first, 1);
    });
    refused("embedding row beyond the table accepted", [&](const Out& out) {
        const auto tb = floats(tf);
        vk.embed(out(nin), {f32, at(tb)}, nin, nrows, past, 1);
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
        const std::vector<std::vector<float>>& start, size_t tail_floats = 0) {
    const size_t layer = 1, core = sh.slot_floats();
    const float tail_sentinel = -12345.5f;
    const std::vector<size_t> tails = tail_floats ? std::vector<size_t>{0, tail_floats} : std::vector<size_t>{};
    auto storage = b.state_alloc(2, start.size(), sh, tails);
    const size_t n = storage->slot_floats(layer), other_n = storage->slot_floats(0);
    for (size_t s = 0; s < start.size(); ++s) {
        require(start[s].size() == core, "recurrent test slot has the wrong core size");
        std::vector<float> slot(n, tail_sentinel);
        std::copy(start[s].begin(), start[s].end(), slot.begin());
        b.write(storage->layer(layer), s * n * sizeof(float), slot.data(), n * sizeof(float));
        if (tail_floats) {
            std::vector<float> other(other_n, tail_sentinel);
            b.write(storage->layer(0), s * other_n * sizeof(float), other.data(), other_n * sizeof(float));
        }
    }
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
        const auto other = read_floats(b, storage->layer(0), other_n, s * other_n);
        require(std::all_of(other.begin(), other.end(), [&](float f) { return f == (tail_floats ? tail_sentinel : 0.0f); }), "a state op wrote another layer");
    }
    return r;
}

// A layer-specific tail changes slot spacing, but the recurrent ops may only touch their GDN prefix.
size_t check_layer_tail(backend::Backend& vk) {
    const StateShape sh = {1, 2, 8, 8};
    const size_t core = sh.slot_floats(), tail = 11;
    size_t values = 0;
    for (size_t rows : {size_t(3), size_t(33)}) {
        const std::vector<Seq> seqs = {{7, 1, 2, rows}};
        const Inputs in = inputs(171 + (uint32_t)rows, sh, rows);
        std::vector<std::vector<float>> start(4);
        for (size_t s = 0; s < start.size(); ++s) start[s] = uniform(core, 172 + (uint32_t)s, -0.5f, 0.5f);
        const Run base = run(vk, sh, seqs, in, start);
        const Run with_tail = run(vk, sh, seqs, in, start, tail);
        require(base.conv.size() == with_tail.conv.size() && base.delta.size() == with_tail.delta.size() &&
                    std::memcmp(base.conv.data(), with_tail.conv.data(), base.conv.size() * sizeof(float)) == 0 &&
                    std::memcmp(base.delta.data(), with_tail.delta.data(), base.delta.size() * sizeof(float)) == 0,
                "a recurrent tail changed device conv or delta rows");
        for (size_t s = 0; s < start.size(); ++s) {
            require(std::memcmp(base.slots[s].data(), with_tail.slots[s].data(), core * sizeof(float)) == 0,
                    "a recurrent tail changed the device GDN state");
            require(std::all_of(with_tail.slots[s].begin() + core, with_tail.slots[s].end(),
                                [](float f) { return f == -12345.5f; }),
                    "a recurrent op wrote into its layer tail");
        }
        require(std::memcmp(start[1].data(), with_tail.slots[1].data(), core * sizeof(float)) == 0,
                "a recurrent op changed its nonzero source slot");
        values += with_tail.conv.size() + with_tail.delta.size() + start.size() * (core + tail);
    }
    return values;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Every pair of extents of one class gives the same bits through each dtype's matrix products and shared attention (row_classes::check).
size_t check_row_classes(backend::Backend& vk) {
    return row_classes::check(vk,
                              {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q4_K,
                               quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0},
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

// Compare fresh, short-history, one-token and verify sequences beside each other against the CPU, with NaN in unread state and history slots.
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

// Sequence rows and state must be identical alone, reordered, batched and sliced (docs/QWEN35.md, Row classes).
// The chosen lengths also compare both delta-rule builds across their threshold.
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

// state_alloc zero-fills every slot of every layer on the device, and each slot holds what is written to it.
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
    for (size_t l = 0; l < layers; ++l)
        for (size_t k = 0; k < slots; ++k)
            require(same_bits(read_floats(vk, s->layer(l), n, k * n), held[l][k]), "a device state slot does not hold what was written to it");
    return layers * slots;
}

// The gated norm against the CPU, in place too, heads near 1e-4 holding eps.
size_t check_gated_norm(Pair& p, size_t heads, size_t dim, backend::NormGate gate = backend::NormGate::silu) {
    const size_t rows = 23, n = rows * heads * dim;
    auto x = uniform(n, 80, -2.0f, 2.0f);
    auto z = uniform(n, 81, -6.0f, 6.0f);
    z[0] = 0.0f;
    z[1] = -4.0f;
    const auto w = uniform(dim, 82, 0.5f, 1.5f);
    for (size_t i = 0; i < dim; ++i) x[(5 * heads + 1) * dim + i] *= 1e-4f;
    Pair::In xi = p.in(x), zi = p.in(z), wi = p.in(w);
    Pair::Out d = p.out(n);
    p.cpu.gated_rms_norm(d.cs(), xi.cs(), zi.cs(), wi.cs(), rows, heads, dim, 1e-6f, {}, gate);
    p.vk.gated_rms_norm(d.vs(), xi.vs(), zi.vs(), wi.vs(), rows, heads, dim, 1e-6f, {}, gate);
    auto r = p.results(d);
    size_t values = close(r.first, r.second, 1e-5, "gated_rms_norm differs beyond 1e-5");
    Pair::Out ip = p.out(n);
    p.vk.write(*ip.v, 0, x.data(), n * sizeof(float));
    p.vk.gated_rms_norm(ip.vs(), ip.vs(), zi.vs(), wi.vs(), rows, heads, dim, 1e-6f, {}, gate);
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

// The gated-attention tail must replace attention's twin with the gated output before the residual projection.
// Check both row and tile paths against a CPU reference fed their respective activations.
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
                infer::BlockPool pool; pool.configure(st->max_blocks());
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
                    b.matmul_add({type, {Wb.get(), 0}}, {Xb.get(), 0}, {Yb.get(), 0}, qw, nout, rows, rr);
                } else {
                    b.matmul_add({type, {Wb.get(), 0}}, {ob.get(), 0}, {Yb.get(), 0}, qw, nout, rows, rr);
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

// Embedded-drafter argmax and embedding ops must match the CPU, including ties, nonfinite logits, prior ids and out-of-range ids (AGENTS.md, Tests).
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
    // Rows of one slice of the device's reduction, of a slice less one id and plus one, whose last slice is one id, and of a vocabulary's 61 slices; 64 rows a call, as many as a pass drafts.
    for (size_t n : {size_t(37), backend::kVulkanArgmaxSlice - 1, backend::kVulkanArgmaxSlice + 1, size_t(248320)}) {
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        const size_t rows = 64;
        std::vector<float> flat = uniform(rows * n, (uint32_t)(300 + n % 7), -6.0f, 6.0f);
        flat[1 * n + 3] = flat[1 * n + n - 2] = 9.5f;   // a tie
        flat[2 * n + n / 2] = inf;
        flat[3 * n] = nan;                               // a NaN in the first slice, the largest in the last
        flat[3 * n + n - 1] = 9.5f;
        flat[4 * n + n - 1] = nan;                       // and a NaN in the last, the largest in the first
        flat[4 * n] = 9.5f;
        for (size_t i = 0; i < n; ++i) flat[5 * n + i] = -inf;
        flat[6 * n + 7] = 9.0f;                          // its prior id is invalid
        flat[7 * n + n - 1] = 9.25f;                     // the largest in the last slice
        flat[8 * n] = 9.25f;                             // and in the first
        flat[9 * n] = flat[9 * n + n - 1] = 9.75f;       // a tie between the first id and the last
        flat[10 * n + n - 1] = inf;                      // an infinity in the last slice
        std::vector<uint32_t> prior(rows, 0);
        prior[6] = (uint32_t)n;
        const backend::BufferPtr lc = q35::floats_on(p.cpu, flat), lv = q35::floats_on(p.vk, flat), pc = ids_on(p.cpu, prior), pv = ids_on(p.vk, prior);
        for (bool with_prior : {true, false}) {
            const backend::BufferPtr oc = p.cpu.alloc(rows * sizeof(float), backend::Memory::device), ov = p.vk.alloc(rows * sizeof(float), backend::Memory::device);
            p.cpu.argmax_rows({oc.get(), 0}, {lc.get(), 0}, rows, n, with_prior ? backend::CSlice{pc.get(), 0} : backend::CSlice{});
            p.vk.argmax_rows({ov.get(), 0}, {lv.get(), 0}, rows, n, with_prior ? backend::CSlice{pv.get(), 0} : backend::CSlice{});
            p.vk.sync();
            const std::vector<uint32_t> c = ids_of(p.cpu, *oc, rows), v = ids_of(p.vk, *ov, rows);
            require(c == v, ("argmax_rows differs from the CPU over " + std::to_string(n) + " ids").c_str());
            require(c[1] == 3 && c[2] == n && c[3] == n && c[4] == n && c[5] == n && c[6] == (with_prior ? n : 7), "argmax_rows over ties, infinities and NaN");
            require(c[7] == n - 1 && c[8] == 0 && c[9] == 0 && c[10] == n, "argmax_rows over the first and the last slice of a row");
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
        p.cpu.embed_ids({dc.get(), 0}, {type, {tc.get(), 0}}, width, vocab, {ic.get(), 0}, ids.size());
        p.vk.embed_ids({dv.get(), 0}, {type, {tv.get(), 0}}, width, vocab, {iv.get(), 0}, ids.size());
        p.vk.sync();
        const std::vector<float> c = q35::read_floats(p.cpu, *dc, ids.size() * width), v = q35::read_floats(p.vk, *dv, ids.size() * width);
        require(q35::same_bits(c, v), quantized ? "embed_ids differs from the CPU on a Q8_0 table" : "embed_ids differs from the CPU on an F32 table");
        for (size_t i = 0; i < width; ++i) require(v[width + i] == 0.0f && v[3 * width + i] == 0.0f, "embed_ids of an id past the table wrote no zero row");
        values += c.size();
    }
    // Every type the device embeds: embed_ids gives embed's rows for the ids inside the table, through the same decoding, and a zero row past it.
    const size_t wide = 256;
    for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_F16, quant::GGML_TYPE_BF16, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q4_K,
                          quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
        if (!p.vk.supports_type(type)) continue;
        const std::vector<uint8_t> bytes = matrix(type, wide, vocab, 340 + type);
        const backend::BufferPtr tv = p.vk.adopt(bytes.data(), bytes.size()), iv = ids_on(p.vk, ids);
        const backend::BufferPtr dv = p.vk.alloc(ids.size() * wide * sizeof(float), backend::Memory::device), ev = p.vk.alloc(ids.size() * wide * sizeof(float), backend::Memory::device);
        const std::vector<float> sentinel(ids.size() * wide, 17.0f);
        p.vk.write(*dv, 0, sentinel.data(), sentinel.size() * sizeof(float));
        const std::vector<uint32_t> inside = {4, 0, 0, 0, 8};
        p.vk.embed_ids({dv.get(), 0}, {type, {tv.get(), 0}}, wide, vocab, {iv.get(), 0}, ids.size());
        p.vk.embed({ev.get(), 0}, {type, {tv.get(), 0}}, wide, vocab, inside.data(), inside.size());
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
    values += q35::check_layer_tail(vk);
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
    values += q35::check_gated_norm(p, 2, 8, backend::NormGate::sigmoid) + q35::check_gated_norm(p, 16, 128, backend::NormGate::sigmoid);
    values += q35::check_sigmoid_mul(p, 4, 40) + q35::check_sigmoid_mul(p, 3, 256);
    values += q35::check_gated_attention(p, integer_dot);
    values += check_drafter_ops(p);
    std::cout << "backend-vulkan: qwen35 ops: " << runs << " device runs bit for bit across views, orders and passes; decay flush, state storage\n";
    return values;
}
}

// The vision encoder's ops against the CPU (docs/VISION.md, section 3) at the towers' widths: the bias and the row mix exact, the norm, gelu and rope within their bounds.
// Attention over a non-causal view is held to the CPU within 1e-4.
namespace vision {
// `dst` written with `init` on both sides, so an op that adds to it starts from the same rows.
Pair::Out seeded(Pair& p, const std::vector<float>& init) {
    Pair::Out o = p.out(init.size());
    p.cpu.write(*o.c, 0, init.data(), init.size() * sizeof(float));
    p.vk.write(*o.v, 0, init.data(), init.size() * sizeof(float));
    return o;
}

size_t check_ops(backend::Backend& vk) {
    Pair p(vk);
    size_t values = 0;
    for (size_t width : {size_t(48), size_t(72), size_t(1152), size_t(4304)}) {
        const size_t rows = 37;
        const auto x = uniform(rows * width, 400 + uint32_t(width), -3.0f, 5.0f), bias = uniform(width, 401 + uint32_t(width));
        Pair::In bi = p.in(bias);
        Pair::Out d = seeded(p, x);
        p.cpu.add_bias(d.cs(), bi.cs(), rows, width);
        p.vk.add_bias(d.vs(), bi.vs(), rows, width);
        auto r = p.results(d);
        values += exact(r.first, r.second, "add_bias differs from the CPU");
        // The learned positions' four taps of a 48 by 48 table, read in order.
        const size_t side = 48, taps = 4;
        const auto table = uniform(side * side * width, 402 + uint32_t(width)), w = uniform(rows * taps, 403, 0.0f, 1.0f);
        std::vector<uint32_t> idx(rows * taps);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = uint32_t((i * 2654435761u) % (side * side));
        Pair::In ti = p.in(table);
        Pair::Out m = seeded(p, x);
        p.cpu.mix_rows(m.cs(), ti.cs(), width, side * side, idx.data(), w.data(), rows, taps);
        p.vk.mix_rows(m.vs(), ti.vs(), width, side * side, idx.data(), w.data(), rows, taps);
        r = p.results(m);
        // Vulkan does not require fma fused, and on an MI50 under RADV the taps' last bits differ from the CPU's fused chain, so they are held to a bound.
        values += close(r.first, r.second, 1e-6, "mix_rows differs from the CPU beyond 1e-6");
        // LayerNorm over rows of a nonzero mean, packed and with a stride, out of place and in place.
        for (size_t stride : {width, width + 8}) {
            const auto src = uniform(rows * stride, 404 + uint32_t(stride), 1.0f, 7.0f), lw = uniform(width, 405, 0.5f, 1.5f), lb = uniform(width, 406);
            Pair::In si = p.in(src), wi = p.in(lw), bb = p.in(lb);
            Pair::Out n = seeded(p, src);
            p.cpu.layer_norm_rows(n.cs(), si.cs(), wi.cs(), bb.cs(), rows, width, stride, 1e-6f);
            p.vk.layer_norm_rows(n.vs(), si.vs(), wi.vs(), bb.vs(), rows, width, stride, 1e-6f);
            r = p.results(n);
            values += close(r.first, r.second, 1e-5, "layer_norm_rows differs from the CPU beyond 1e-5");
            Pair::Out ip = seeded(p, src);
            p.vk.layer_norm_rows(ip.vs(), {ip.v.get(), 0}, wi.vs(), bb.vs(), rows, width, stride, 1e-6f);
            std::vector<float> in_place(src.size());
            p.vk.read(*ip.v, 0, in_place.data(), in_place.size() * sizeof(float));
            require(q35::same_bits(in_place, r.second), "layer_norm_rows in place differs on the device");
            // A row alone gives the bits it gives beside the others.
            Pair::Out one = p.out(width);
            p.vk.layer_norm_rows(one.vs(), {si.v.get(), 5 * stride}, wi.vs(), bb.vs(), 1, width, width, 1e-6f);
            std::vector<float> alone(width);
            p.vk.read(*one.v, 0, alone.data(), width * sizeof(float));
            require(std::memcmp(alone.data(), r.second.data() + 5 * stride, width * sizeof(float)) == 0, "layer_norm_rows of a row alone differs from the row beside others");
        }
    }
    // GELU in both forms over values from 1e-30 to 1e5 of both signs and a uniform spread where it bends.
    // The bound is absolute where values are small, not the kernel's precision: the erf form's 1.5e-7 is about 1e-3 relative below -3, where gelu is under 4e-4.
    std::vector<float> g = uniform(65536, 407, -12.0f, 12.0f);
    for (int e = -30; e <= 5; ++e)
        for (float s : {1.0f, -1.0f, 3.7f, -3.7f}) g.push_back(s * std::pow(10.0f, float(e)));
    Pair::In gi = p.in(g);
    for (backend::GeluForm form : {backend::GeluForm::tanh, backend::GeluForm::erf}) {
        Pair::Out d = p.out(g.size());
        p.cpu.gelu(d.cs(), gi.cs(), g.size(), form);
        p.vk.gelu(d.vs(), gi.vs(), g.size(), form);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-6, form == backend::GeluForm::erf ? "gelu (erf) differs from the CPU beyond 1e-6" : "gelu (tanh) differs from the CPU beyond 1e-6");
    }
    // The rope over two streams, a head's first quarter of pairs by the patch's row and the rest by its column, with and without a norm, and one stream without a norm.
    for (size_t hd : {size_t(64), size_t(72), size_t(128)}) {
        const size_t rows = 37, heads = 4, half = hd / 2, table = 64, width = heads * hd;
        std::vector<float> cs(table * half), sn(table * half);
        for (size_t t = 0; t < table; ++t)
            for (size_t i = 0; i < half; ++i) {
                const double f = std::pow(10000.0, -2.0 * double(i) / double(half));
                cs[t * half + i] = float(std::cos(double(t) * f));
                sn[t * half + i] = float(std::sin(double(t) * f));
            }
        std::vector<uint8_t> of(half);
        for (size_t j = 0; j < half; ++j) of[j] = j < hd / 4 ? 0 : 1;
        std::vector<uint32_t> pos(rows * 2);
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = uint32_t((i * 7 + 3) % table);
        const auto src = uniform(rows * width, 408 + uint32_t(hd)), nw = uniform(hd, 409, 0.5f, 1.5f);
        Pair::In si = p.in(src), ci = p.in(cs), sni = p.in(sn), wi = p.in(nw);
        for (int mode = 0; mode < 3; ++mode) {
            const backend::RopeStreams streams = mode == 2 ? backend::RopeStreams{} : backend::RopeStreams{2, of.data()};
            const bool norm = mode == 1;
            Pair::Out d = p.out(src.size());
            p.cpu.norm_rope_partial(d.cs(), si.cs(), rows, width, hd, heads, hd, hd, norm ? wi.cs() : backend::CSlice{}, 1e-6f, ci.cs(), sni.cs(), pos.data(), streams);
            p.vk.norm_rope_partial(d.vs(), si.vs(), rows, width, hd, heads, hd, hd, norm ? wi.vs() : backend::CSlice{}, 1e-6f, ci.vs(), sni.vs(), pos.data(), streams);
            auto r = p.results(d);
            values += close(r.first, r.second, 1e-5, "the rope over streams differs from the CPU beyond 1e-5");
            Pair::Out ip = seeded(p, src);
            p.vk.norm_rope_partial(ip.vs(), {ip.v.get(), 0}, rows, width, hd, heads, hd, hd, norm ? wi.vs() : backend::CSlice{}, 1e-6f, ci.vs(), sni.vs(), pos.data(), streams);
            std::vector<float> in_place(src.size());
            p.vk.read(*ip.v, 0, in_place.data(), in_place.size() * sizeof(float));
            require(q35::same_bits(in_place, r.second), "the rope over streams in place differs on the device");
        }
    }
    // Attention over one non-causal view of an image's patches, every row seeing every key: the flash kernel's builds at the towers' head widths, and the per-row kernel's at 128.
    for (int hd : {64, 72, 128})
        for (size_t n : {size_t(37), size_t(300), size_t(4096)}) {
            const int heads = 4;
            const size_t w = size_t(heads) * size_t(hd);
            const auto q = uniform(n * w, 410 + uint32_t(n), -2.0f, 2.0f), k = uniform(n * w, 411 + uint32_t(n), -2.0f, 2.0f), v = uniform(n * w, 412 + uint32_t(n));
            auto run = [&](backend::Backend& b) {
                auto st = b.kv_alloc(1, size_t(heads), size_t(hd), n);
                std::vector<int32_t> blocks(backend::blocks_for(n, b.kv_layout().block_tokens));
                for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = int32_t(i);
                const backend::KVView view{st.get(), blocks.data(), blocks.size(), 0, n, n, false};
                const auto Qb = b.adopt(q.data(), q.size() * sizeof(float)), Kb = b.adopt(k.data(), k.size() * sizeof(float)), Vb = b.adopt(v.data(), v.size() * sizeof(float));
                b.kv_write(0, &view, 1, {Kb.get(), 0}, {Vb.get(), 0});
                const auto ob = b.alloc(n * w * sizeof(float), backend::Memory::device);
                b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, heads, heads, hd);
                std::vector<float> out(n * w);
                b.read(*ob, 0, out.data(), out.size() * sizeof(float));
                return out;
            };
            values += close(run(p.cpu), run(p.vk), 1e-4, "non-causal attention differs from the CPU beyond 1e-4");
        }
    // 8192 patches of 16 heads 64 and 72 wide, which the flash kernel cuts into two dispatches (vulkan-buffer checks the pieces): rows at both ends of each give the bits of the same rows in an uncut call of 32 rows over the same keys.
    // The 4096 rows of heads 128 wide above are cut in two by the per-row build and held to the CPU.
    for (int hd : {64, 72}) {
        const int heads = 16;
        const size_t n = 8192, w = size_t(heads) * size_t(hd), picked[4] = {0, 4088, 4096, 8184};
        const auto q = uniform(n * w, 413 + uint32_t(hd), -2.0f, 2.0f), k = uniform(n * w, 414 + uint32_t(hd), -2.0f, 2.0f), v = uniform(n * w, 415 + uint32_t(hd));
        auto st = vk.kv_alloc(1, size_t(heads), size_t(hd), n);
        std::vector<int32_t> blocks(backend::blocks_for(n, vk.kv_layout().block_tokens));
        for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = int32_t(i);
        const backend::KVView view{st.get(), blocks.data(), blocks.size(), 0, n, n, false};
        const auto Qb = vk.adopt(q.data(), q.size() * sizeof(float)), Kb = vk.adopt(k.data(), k.size() * sizeof(float)), Vb = vk.adopt(v.data(), v.size() * sizeof(float));
        vk.kv_write(0, &view, 1, {Kb.get(), 0}, {Vb.get(), 0});
        const auto ob = vk.alloc(n * w * sizeof(float), backend::Memory::device);
        vk.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, heads, heads, hd);
        std::vector<float> whole(n * w);
        vk.read(*ob, 0, whole.data(), whole.size() * sizeof(float));
        // Eight rows from each place, as the 32 rows of a view whose history is the other 8160 keys.
        std::vector<float> q32(32 * w);
        for (size_t j = 0; j < 32; ++j) std::memcpy(q32.data() + j * w, q.data() + (picked[j / 8] + j % 8) * w, w * sizeof(float));
        const backend::KVView tail{st.get(), blocks.data(), blocks.size(), n - 32, 32, 32, false};
        const auto Q32 = vk.adopt(q32.data(), q32.size() * sizeof(float));
        const auto o32 = vk.alloc(32 * w * sizeof(float), backend::Memory::device);
        vk.attention({Q32.get(), 0}, 0, &tail, 1, {o32.get(), 0}, heads, heads, hd);
        std::vector<float> rows32(32 * w);
        vk.read(*o32, 0, rows32.data(), rows32.size() * sizeof(float));
        for (size_t j = 0; j < 32; ++j)
            require(std::memcmp(rows32.data() + j * w, whole.data() + (picked[j / 8] + j % 8) * w, w * sizeof(float)) == 0,
                    "a row of non-causal attention cut into dispatches differs from the row in an uncut call");
        values += 32 * w;
    }
    // Views of one call are causal or not alike.
    {
        auto st = vk.kv_alloc(1, 2, 64, 256);
        const int32_t blocks[4] = {0, 1, 2, 3};
        const backend::KVView views[2] = {{st.get(), blocks, 1, 0, 8, 8, false}, {st.get(), blocks + 1, 1, 0, 8, 8, true}};
        const auto qb = vk.alloc(16 * 128 * sizeof(float), backend::Memory::device), ob = vk.alloc(16 * 128 * sizeof(float), backend::Memory::device);
        bool refused = false;
        try { vk.attention({qb.get(), 0}, 0, views, 2, {ob.get(), 0}, 2, 2, 64); }
        catch (const std::runtime_error& e) { refused = std::string(e.what()).find("causal and non-causal views") != std::string::npos; }
        require(refused, "an attention call of causal and non-causal views was not refused by name");
    }
    std::cout << "backend-vulkan: vision encoder ops: bias exact, row mix, norm, gelu and rope over streams within their bounds, non-causal attention over 37 to 4096 keys, through the flash kernel at heads 64 and 72, and 8192 keys cut into dispatches at both\n";
    return values;
}
}  // namespace vision

// Matrix precision checks use unquantized double products and the selected activation policy's bound (AGENTS.md, Tests).
// Block outliers expose the quantization step, and F32/BF16 cases also hold decode-column identity.
size_t check_activation_precision(backend::Backend& vk, backend::Dtype dtype) {
    const size_t nin = 1024, nout = 48;
    const backend::DeviceProfile prof = backend::vulkan_device_profile(vk);
    const size_t tile_cols = std::max<size_t>(64, backend::tile_from_for(prof, false, nin));
    struct Case { uint32_t type; size_t nbatch; };
    std::vector<Case> cases;
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q4_K,
                          quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0})
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
            case quant::GGML_TYPE_Q5_0: quant::dequantize_row_q5_0(src, dst, nin / 32); break;
            case quant::GGML_TYPE_Q5_1: quant::dequantize_row_q5_1(src, dst, nin / 32); break;
            case quant::GGML_TYPE_Q4_K: quant::dequantize_row_q4_K(src, dst, nin / 256); break;
            case quant::GGML_TYPE_Q5_K: quant::dequantize_row_q5_K(src, dst, nin / 256); break;
            case quant::GGML_TYPE_Q6_K: quant::dequantize_row_q6_K(src, dst, nin / 256); break;
            case quant::GGML_TYPE_TQ1_0: quant::dequantize_row_tq1_0(src, dst, nin / 256); break;
            default: quant::dequantize_row_tq2_0(src, dst, nin / 256); break;
            }
        }
        const auto wd = vk.adopt(wb.data(), wb.size()), xd = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto yd = vk.alloc(c.nbatch * nout * sizeof(float));
        const backend::RowRun run{c.nbatch, c.nbatch < tile_cols ? size_t(1) : tile_cols};
        const backend::RowRuns runs = dtype == backend::Dtype::f32 ? backend::RowRuns{&run, 1} : backend::RowRuns{};
        testq::take_matrix_paths(vk);
        vk.matmul({c.type, {wd.get(), 0}}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, c.nbatch, runs, dtype);
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
                vk.matmul({c.type, {wd.get(), 0}}, {xd.get(), column * nin}, {one.get(), 0}, nin, nout, 1, {}, dtype);
                vk.read(*one, 0, got.data(), got.size() * sizeof(float));
                require(std::memcmp(got.data(), y.data() + column * nout, nout * sizeof(float)) == 0,
                        "float output depends on the decode batch width");
            }
        }
        values += c.nbatch * nout;
    }
    return values;
}

size_t check_tq_range(backend::Backend& vk) {
    constexpr size_t width = 256, rows = 8, decode_cols = 3;
    const auto profile = backend::vulkan_device_profile(vk);
    const size_t prompt_cols = std::max<size_t>(64, backend::tile_from_for(profile, false, width));
    const uint16_t scales[rows] = {0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    size_t checked = 0;
    for (uint32_t type : {quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
        const size_t block_bytes = type == quant::GGML_TYPE_TQ1_0 ? quant::TQ1_0_TYPESIZE : quant::TQ2_0_TYPESIZE;
        std::vector<uint8_t> packed = testq::tq_matrix(type, width * rows, 0x53);
        for (size_t r = 0; r < rows; ++r) {
            uint8_t* block = packed.data() + r * block_bytes;
            block[block_bytes - 2] = uint8_t(scales[r]);
            block[block_bytes - 1] = uint8_t(scales[r] >> 8);
        }
        std::vector<float> weights(width * rows);
        for (size_t r = 0; r < rows; ++r) {
            const uint8_t* block = packed.data() + r * block_bytes;
            const float d = f16_to_f32(scales[r]);
            for (unsigned i = 0; i < width; ++i) {
                unsigned code;
                if (type == quant::GGML_TYPE_TQ1_0) {
                    const unsigned plane = i < 160 ? i / 32 : i < 240 ? 5 + (i - 160) / 16 : 10 + (i - 240) / 4;
                    const unsigned lane = i < 160 ? i % 32 : i < 240 ? (i - 160) % 16 : (i - 240) % 4;
                    const unsigned byte = plane < 5 ? lane : plane < 10 ? 32 + lane : 48 + lane;
                    const unsigned power = plane < 5 ? plane : plane < 10 ? plane - 5 : plane - 10;
                    constexpr unsigned radix3[] = {1, 3, 9, 27, 81};
                    code = (unsigned(uint8_t(block[byte] * radix3[power])) * 3) >> 8;
                } else {
                    const unsigned half = i / 128, plane = (i % 128) / 32, lane = i % 32;
                    code = (block[half * 32 + lane] >> (2 * plane)) & 3;
                }
                weights[r * width + i] = float(int(code) - 1) * d;
            }
        }
        const auto w = vk.adopt(packed.data(), packed.size());
        for (backend::Dtype dtype : {backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::f16, backend::Dtype::int8}) {
          for (size_t cols : {decode_cols, prompt_cols}) {
            std::vector<float> x(cols * width);
            for (size_t c = 0; c < cols; ++c) {
                const float amplitude = c % 3 == 0 ? 1.0e-30f : c % 3 == 1 ? 1.0f : 1.0e30f;
                for (size_t i = 0; i < width; ++i) {
                    const float fraction = i % 32 == 0 ? 1.0f : i % 32 == 1 ? 0.137f : 0.625f;
                    x[c * width + i] = ((i + c) & 1) ? -amplitude * fraction : amplitude * fraction;
                }
            }
            std::vector<float> rounded = x;
            if (dtype == backend::Dtype::bf16)
                for (float& value : rounded) value = bf16_to_f32(f32_to_bf16(value));
            const auto input = vk.adopt(x.data(), x.size() * sizeof(float));
            std::vector<float> guarded(rows * cols + 2, 19.0f);
            const auto output = vk.alloc(guarded.size() * sizeof(float));
            vk.write(*output, 0, guarded.data(), guarded.size() * sizeof(float));
            std::vector<backend::RowRun> runs;
            if (cols == decode_cols) for (size_t r = 0; r < cols; ++r) runs.push_back({r + 1, 1});
            else runs.push_back({cols, cols});
            (void)testq::take_matrix_paths(vk);
            vk.matmul({type, {w.get(), 0}}, {input.get(), 0}, {output.get(), 1}, width, rows, cols,
                      {runs.data(), runs.size()}, dtype);
            const auto paths = testq::take_matrix_paths(vk);
            const bool tile = dtype == backend::Dtype::bf16 || cols >= backend::tile_from_for(profile, false, width);
            const bool integer = (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8) &&
                                 (tile ? profile.prefer_integer_dot : profile.lookup_integer_dot);
            const std::string expected_path = dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32";
            require(paths == std::vector<std::string>{expected_path}, "TQ range matrix did not witness its selected arithmetic path");
            vk.read(*output, 0, guarded.data(), guarded.size() * sizeof(float));
            require(guarded.front() == 19.0f && guarded.back() == 19.0f, "TQ range matrix overwrote an output guard");
            const auto activations = integer ? testq::row_activations(rounded) : rounded;
            for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
                double want = 0.0, magnitude = 0.0, weight_sum = 0.0, peak = 0.0;
                for (size_t i = 0; i < width; ++i) {
                    const double wi = weights[r * width + i], xi = activations[c * width + i];
                    want += wi * xi;
                    magnitude += std::fabs(wi * xi);
                    weight_sum += std::fabs(wi);
                    peak = std::max(peak, std::fabs(double(rounded[c * width + i])));
                }
                const float got = guarded[1 + c * rows + r];
                require(std::isfinite(got), "TQ range matrix produced a nonfinite output");
                if (scales[r] == 0x0000 || scales[r] == 0x8000)
                    require(got == 0.0f, "TQ zero-scale matrix did not produce exact zero");
                const double u = std::numeric_limits<float>::epsilon() / 2.0;
                const double ops = double(width + 4) * u;
                const double rounding = ops / (1.0 - ops) * magnitude + width * std::numeric_limits<float>::denorm_min();
                const double quantization = integer ? peak / 32767.0 / 2.0 * weight_sum : 0.0;
                require(std::fabs(double(got) - want) <= rounding + quantization,
                        "TQ range matrix differs from the independent code-and-scale oracle");
            }
            ++checked;
          }
        }
    }
    for (uint32_t type : {quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
        const size_t block_bytes = type == quant::GGML_TYPE_TQ1_0 ? quant::TQ1_0_TYPESIZE : quant::TQ2_0_TYPESIZE;
        std::vector<uint8_t> packed(block_bytes, 0);
        packed[block_bytes - 2] = 0x00; packed[block_bytes - 1] = 0x00;
        const auto table = vk.adopt(packed.data(), packed.size());
        const auto output = vk.alloc((width + 2) * sizeof(float));
        const float guard = 19.0f;
        vk.write(*output, 0, &guard, sizeof guard);
        vk.write(*output, (width + 1) * sizeof(float), &guard, sizeof guard);
        const uint32_t id = 0;
        vk.embed({output.get(), 1}, {type, {table.get(), 0}}, width, 1, &id, 1);
        std::vector<float> got(width + 2);
        vk.read(*output, 0, got.data(), got.size() * sizeof(float));
        require(got.front() == guard && got.back() == guard, "TQ embedding overwrote an output guard");
        for (size_t i = 1; i <= width; ++i)
            require(got[i] == 0.0f && std::signbit(got[i]), "TQ embedding lost the negative zero from zero scale and code zero");
        ++checked;
    }
    return checked;
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
    for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_F16, quant::GGML_TYPE_BF16, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1,
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
                if (op == 0) vk.matmul({type, {wd.get(), 0}}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, count, runs, dtype);
                else if (op == 1) vk.matmul_logits({type, {wd.get(), 0}}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, count, runs, dtype);
                else if (op == 2) {
                    vk.write(*yd, 0, output.data(), output.size() * sizeof(float));
                    vk.matmul_add({type, {wd.get(), 0}}, {xd.get(), 0}, {yd.get(), 0}, nin, nout, count, runs, dtype);
                } else vk.matmul_group({{{type, {wd.get(), 0}}, {yd.get(), 0}, nout},
                                        {{type, {wd.get(), 0}}, {yd.get(), count * nout}, nout}}, {xd.get(), 0}, nin, count, runs, dtype);
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
        if (op == 0) vk.matmul({quant::GGML_TYPE_F32, {wd.get(), 0}}, input, output, nin, nout, columns, runs, backend::Dtype::bf16);
        else if (op == 1) vk.matmul_logits({quant::GGML_TYPE_F32, {wd.get(), 0}}, input, output, nin, nout, columns, runs, backend::Dtype::bf16);
        else if (op == 2) vk.matmul_add({quant::GGML_TYPE_F32, {wd.get(), 0}}, input, output, nin, nout, columns, runs, backend::Dtype::bf16);
        else vk.matmul_group({{{quant::GGML_TYPE_F32, {wd.get(), 0}}, output, nout},
                              {{quant::GGML_TYPE_F32, {wd.get(), 0}}, {yd.get(), guard + stride}, nout}}, input, nin, columns, runs, backend::Dtype::bf16);
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

// A tensor group's collective must add partials in member order on every device, with alternating inbox parities (docs/TENSOR-SPLIT.md, section 4.3).
// AGENTS.md, Tests describes the widths, tails and refusals.
size_t check_collective(const backend::BackendPtr& first) {
    backend::BackendPtr second;
    try {
        second = backend::make_vulkan_backend(1, false);
    } catch (const std::exception& e) {
        std::cout << "backend-vulkan: no second device for the collective (" << e.what() << ")\n";
        return 0;
    }
    std::vector<backend::Backend*> members{first.get(), second.get()};
    const size_t reserved = 64, width = 96, offset = 7;
    std::unique_ptr<backend::Collective> sum = first->join(members, reserved, width);
    if (!sum) {
        std::cout << "backend-vulkan: the devices share no memory or semaphores, so no collective\n";
        return 0;
    }
    size_t checks = 0;
    uint32_t seed = 11;
    // Sums of `rows` rows over a collective's members, each member's residual the sum in member order.
    auto sums = [&](const std::vector<backend::Backend*>& group, backend::Collective& c, size_t wide, std::initializer_list<size_t> counts) {
        const size_t W = group.size();
        for (size_t rows : counts) {
            const size_t n = rows * wide;
            std::vector<std::vector<float>> partial(W, std::vector<float>(n));
            std::vector<float> start(n);
            for (size_t i = 0; i < n; ++i) {
                for (size_t m = 0; m < W; ++m) {
                    seed = seed * 1664525u + 1013904223u;
                    partial[m][i] = float(int(seed >> 16) % 2001 - 1000) / 97.0f;
                }
                start[i] = float(int(seed >> 8) % 4001 - 2000) / 13.0f;
            }
            std::vector<backend::BufferPtr> x;
            std::vector<backend::Slice> xs;
            for (size_t m = 0; m < W; ++m) {
                x.push_back(group[m]->alloc((n + offset) * sizeof(float)));
                group[m]->write(*x.back(), offset * sizeof(float), start.data(), n * sizeof(float));
                const backend::Slice p = c.partial(m);
                group[m]->write(*p.buffer, p.offset * sizeof(float), partial[m].data(), n * sizeof(float));
                xs.push_back({x.back().get(), offset});
            }
            c.sum_into(xs, rows, wide);
            for (size_t m = 0; m < W; ++m) {
                group[m]->submit();
                std::vector<float> got(n);
                group[m]->read(*x[m], offset * sizeof(float), got.data(), n * sizeof(float));
                for (size_t i = 0; i < n; ++i) {
                    float all = partial[0][i];
                    for (size_t k = 1; k < W; ++k) all += partial[k][i];
                    const float want = start[i] + all;
                    require(std::memcmp(&got[i], &want, sizeof(float)) == 0, "a member's residual is not the sum in member order");
                }
            }
            ++checks;
        }
    };
    sums(members, *sum, width, {1, 7, 64, 64, 7, 1});
    // Summed in place, each member's partial rows become the sum in member order, at the width made for and at a quarter of it.
    auto in_place = [&](const std::vector<backend::Backend*>& group, backend::Collective& c, size_t wide, std::initializer_list<size_t> counts) {
        const size_t W = group.size();
        for (size_t rows : counts) {
            const size_t n = rows * wide;
            std::vector<std::vector<float>> partial(W, std::vector<float>(n));
            for (size_t i = 0; i < n; ++i)
                for (size_t m = 0; m < W; ++m) {
                    seed = seed * 1664525u + 1013904223u;
                    partial[m][i] = float(int(seed >> 16) % 2001 - 1000) / 97.0f;
                }
            for (size_t m = 0; m < W; ++m) {
                const backend::Slice p = c.partial(m);
                group[m]->write(*p.buffer, p.offset * sizeof(float), partial[m].data(), n * sizeof(float));
            }
            c.sum_in_place(rows, wide);
            for (size_t m = 0; m < W; ++m) {
                group[m]->submit();
                std::vector<float> got(n);
                const backend::Slice p = c.partial(m);
                group[m]->read(*p.buffer, p.offset * sizeof(float), got.data(), n * sizeof(float));
                for (size_t i = 0; i < n; ++i) {
                    float want = partial[0][i];
                    for (size_t k = 1; k < W; ++k) want += partial[k][i];
                    require(std::memcmp(&got[i], &want, sizeof(float)) == 0, "a member's partial rows summed in place are not the sum in member order");
                }
            }
            ++checks;
        }
    };
    in_place(members, *sum, width, {1, 7, 64});
    in_place(members, *sum, width / 4, {1, 64, 256});
    // Two passes' sums interleaved on the one collective, as a pass apart's are between a prompt's pieces: each must end with the bits it ends with alone.
    // The second run clears the partial rows before each sum and writes half of them, as a routed part does.
    for (const bool cleared : {false, true}) {
        const size_t W = members.size(), kSums = 3;
        struct Seq {
            size_t rows = 0;
            std::vector<float> start, want;
            std::vector<backend::BufferPtr> x;                     // per member
            std::vector<std::vector<backend::BufferPtr>> staged;   // per sum, per member
        };
        std::vector<backend::BufferPtr> zeros;
        for (size_t m = 0; m < W; ++m) zeros.push_back(members[m]->alloc(reserved * width * sizeof(float)));
        auto make = [&](size_t rows) {
            Seq q;
            q.rows = rows;
            const size_t n = rows * width, kept = cleared ? rows / 2 * width : n;
            q.start.resize(n);
            for (float& v : q.start) {
                seed = seed * 1664525u + 1013904223u;
                v = float(int(seed >> 8) % 4001 - 2000) / 13.0f;
            }
            q.want = q.start;
            for (size_t m = 0; m < W; ++m) q.x.push_back(members[m]->alloc((n + offset) * sizeof(float)));
            for (size_t k = 0; k < kSums; ++k) {
                std::vector<std::vector<float>> part(W, std::vector<float>(n));
                q.staged.emplace_back();
                for (size_t m = 0; m < W; ++m) {
                    for (float& v : part[m]) {
                        seed = seed * 1664525u + 1013904223u;
                        v = float(int(seed >> 16) % 2001 - 1000) / 97.0f;
                    }
                    q.staged.back().push_back(members[m]->alloc(n * sizeof(float)));
                    members[m]->write(*q.staged.back().back(), 0, part[m].data(), n * sizeof(float));
                }
                for (size_t i = 0; i < n; ++i) {
                    float all = i < kept ? part[0][i] : 0.0f;
                    for (size_t m = 1; m < W; ++m) all += i < kept ? part[m][i] : 0.0f;
                    q.want[i] += all;
                }
            }
            return q;
        };
        auto reset = [&](Seq& q) {
            for (size_t m = 0; m < W; ++m) members[m]->write(*q.x[m], offset * sizeof(float), q.start.data(), q.start.size() * sizeof(float));
        };
        auto step = [&](Seq& q, size_t k) {
            const size_t n = q.rows * width, kept = cleared ? q.rows / 2 * width : n;
            std::vector<backend::Slice> xs;
            for (size_t m = 0; m < W; ++m) {
                const backend::Slice p = sum->partial(m);
                if (cleared) members[m]->copy(*p.buffer, p.offset * sizeof(float), *zeros[m], 0, n * sizeof(float));
                members[m]->copy(*p.buffer, p.offset * sizeof(float), *q.staged[k][m], 0, kept * sizeof(float));
                xs.push_back({q.x[m].get(), offset});
            }
            sum->sum_into(xs, q.rows, width);
        };
        auto read = [&](Seq& q) {
            std::vector<std::vector<float>> got(W, std::vector<float>(q.start.size()));
            for (size_t m = 0; m < W; ++m) {
                members[m]->submit();
                members[m]->read(*q.x[m], offset * sizeof(float), got[m].data(), got[m].size() * sizeof(float));
            }
            return got;
        };
        Seq a = make(7), b = make(64);
        std::vector<std::vector<std::vector<float>>> alone;
        for (Seq* q : {&a, &b}) {
            reset(*q);
            for (size_t k = 0; k < kSums; ++k) step(*q, k);
            alone.push_back(read(*q));
            for (size_t m = 0; m < W; ++m)
                require(std::memcmp(alone.back()[m].data(), q->want.data(), q->want.size() * sizeof(float)) == 0, "a sequence of sums alone is not its sums in member order");
        }
        reset(a);
        reset(b);
        for (size_t k = 0; k < kSums; ++k) {
            step(a, k);
            sum->wait(1);
            step(b, k);
            sum->wait(0);
        }
        const std::vector<std::vector<float>> got_a = read(a), got_b = read(b);
        for (size_t m = 0; m < W; ++m)
            require(std::memcmp(got_a[m].data(), alone[0][m].data(), got_a[m].size() * sizeof(float)) == 0 && std::memcmp(got_b[m].data(), alone[1][m].data(), got_b[m].size() * sizeof(float)) == 0,
                    cleared ? "two sequences of sums interleaved on one collective, their partial rows cleared before each sum, differ from each alone" : "two sequences of sums interleaved on one collective differ from each alone");
        checks += 2;
    }
    std::cout << "backend-vulkan: two sequences of sums interleaved on the collective, each its bits alone, with and without the partial rows cleared\n";
    // Three members, where a third device opens: sums on each side of the size from which the backend sends shares in two shots (327680 floats), at an odd width whose shares differ by a float.
    backend::BackendPtr third;
    try {
        third = backend::make_vulkan_backend(2, false);
    } catch (const std::exception&) {
    }
    if (third) {
        std::vector<backend::Backend*> three{first.get(), second.get(), third.get()};
        const size_t wide = 4097;
        std::unique_ptr<backend::Collective> of_three = first->join(three, 96, wide);
        sums(three, *of_three, wide, {1, 79, 80, 96, 80, 79, 1});
        in_place(three, *of_three, wide, {1, 79, 80, 96});
        std::cout << "backend-vulkan: a collective of three devices, sums below and from the two-shot size on\n";
    } else {
        std::cout << "backend-vulkan: no third device, so no sum in two shots\n";
    }
    // After the exchange each member adds every source and its residual in one kernel (sum_rows), in member order, and records no add of its own.
    // Among three devices the same holds below the two-shot size, and from it a sum_rows per share beside the gather's adds of the other shares.
    {
        auto witness = [&](std::vector<backend::BackendPtr> group, size_t wide, std::initializer_list<size_t> counts, bool gathered) {
            std::vector<backend::Backend*> raw;
            for (const auto& g : group) raw.push_back(g.get());
            std::unique_ptr<backend::Collective> c = raw[0]->join(raw, 96, wide);
            for (backend::Backend* m : raw) (void)backend::vulkan_kernel_times(*m);
            sums(raw, *c, wide, counts);
            for (backend::Backend* m : raw) {
                const auto times = backend::vulkan_kernel_times(*m);
                if (times.empty()) return false;
                bool summed = false;
                for (const auto& t : times) {
                    require(gathered || t.first != "add", "a member added a source of the sum in a kernel of its own");
                    summed = summed || t.first == "sum_rows";
                }
                require(summed, "a member's sum took no sum_rows");
            }
            return true;
        };
        const bool timed = witness({backend::make_vulkan_backend(0, true), backend::make_vulkan_backend(1, true)}, width, {1, 7, 64}, false);
        if (timed && third) {
            witness({backend::make_vulkan_backend(0, true), backend::make_vulkan_backend(1, true), backend::make_vulkan_backend(2, true)}, 4097, {79}, false);
            witness({backend::make_vulkan_backend(0, true), backend::make_vulkan_backend(1, true), backend::make_vulkan_backend(2, true)}, 4097, {80}, true);
        }
        std::cout << "backend-vulkan: " << (timed ? "each member's sum one sum_rows and no add, in one shot and in two" : "the devices time no dispatches, so the sum's kernels are not witnessed") << "\n";
    }
    auto refused = [](const std::function<void()>& f) {
        try { f(); } catch (const std::exception&) { return true; }
        return false;
    };
    std::vector<backend::Backend*> reversed{second.get(), first.get()};
    require(refused([&] { first->join(reversed, 1, 1); }), "a collective joined by a member not first was made");
    require(refused([&] { sum->sum_in_place(reserved, width + 1); }), "a sum in place past the reserved rows was taken");
    auto cpu = backend::make_cpu_backend();
    std::vector<backend::Backend*> mixed{first.get(), cpu.get()};
    require(refused([&] { first->join(mixed, 1, 1); }), "a Vulkan collective over the CPU was made");
    std::vector<backend::Backend*> twice{first.get(), first.get()};
    require(refused([&] { first->join(twice, 1, 1); }), "a Vulkan collective over one device twice was made");
    // Five entries of one device, which the device-twice rule would also refuse, so the text shows which refusal fired.
    std::vector<backend::Backend*> five(5, first.get());
    std::string five_why;
    try { first->join(five, 1, 1); } catch (const std::exception& e) { five_why = e.what(); }
    require(five_why.find("more than 4 devices") != std::string::npos, "a Vulkan collective of five devices was not refused for its size");
    std::cout << "backend-vulkan: the collective of devices 0 and 1, " << checks << " sums in member order and its refusals\n";
    return checks + 3;
}

// A routed integer tile of few entries an expert takes the narrow build of 16 columns, and each entry gives the bits the 64-column build gives it (kNarrowCols).
// Call A routes 200 rows over 64 experts, one of them 40 entries so its runs cross 16; call B adds 400 rows repeating them, which takes the 64-column build, and its first 200 rows must equal A's.
size_t check_routed_narrow() {
    const backend::BackendPtr timed = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *timed;
    const size_t nin = 256, nout = 160, experts = 64, k = 2, few = 200, many = 600;
    size_t checks = 0;
    // The narrow build is the integer tile's: a device whose profile takes the float tile has none.
    if (!backend::vulkan_device_profile(vk).prefer_integer_dot) {
        std::cout << "backend-vulkan: this device takes the float tile, so it has no narrow routed build\n";
        return 0;
    }
    for (const uint32_t type : {quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}) {
        if (!vk.supports_type(type)) continue;
        std::vector<uint8_t> packed(experts * nout * quant::row_bytes(type, nin));
        uint32_t seed = 21 + type;
        for (auto& b : packed) {
            seed = seed * 1664525u + 1013904223u;
            b = uint8_t(seed >> 24);
        }
        std::vector<float> x(many * nin);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int((i * 2654435761u) >> 24) % 33 - 16) / 16.0f;
        std::vector<uint32_t> ids(many * k);
        for (size_t r = 0; r < many; ++r)
            for (size_t j = 0; j < k; ++j) {
                const size_t a = r % few;
                ids[r * k + j] = j == 0 && a < 40 ? 5u : uint32_t((a * 7 + j * 13 + 1) % experts);
            }
        const auto w = vk.adopt(packed.data(), packed.size()), xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto ib = vk.adopt(ids.data(), ids.size() * sizeof(uint32_t));
        for (const auto dtype : {backend::Dtype::int8, backend::Dtype::f16}) {
            std::vector<std::vector<float>> out;
            std::vector<std::string> builds;
            for (const size_t rows : {few, many}) {
                const auto y = vk.alloc(rows * k * nout * sizeof(float));
                const backend::RowRun run{rows, rows};
                const backend::Backend::Routing routing{{ib.get(), 0}, {}, k, experts};
                (void)backend::vulkan_kernel_times(vk);
                vk.matmul_experts({{{type, {w.get(), 0}}, {y.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {&run, 1}, dtype);
                std::string build;
                for (const auto& t : backend::vulkan_kernel_times(vk))
                    if (t.first.rfind("matmul_tile", 0) == 0) build = t.first;
                builds.push_back(build);
                out.emplace_back(rows * k * nout);
                vk.read(*y, 0, out.back().data(), out.back().size() * sizeof(float));
            }
            const bool timed_names = !builds[0].empty();
            require(!timed_names || (builds[0].find("_narrow") != std::string::npos && builds[1].find("_narrow") == std::string::npos),
                    ("routed builds " + builds[0] + " and " + builds[1] + " are not the narrow and the 64-column one").c_str());
            require(std::memcmp(out[0].data(), out[1].data(), out[0].size() * sizeof(float)) == 0, "a narrow routed tile's entries differ from the 64-column build's");
            ++checks;
        }
    }
    return checks;
}

// Printed, not asserted: a routed Q4_K product at Qwen3.8-Flash-Next's expert shape (2560 into 640, 512 experts, 10 a row) over 512 and 2048 rows of int8 activations.
// Picks spread over every expert leave about 10 or 40 entries an expert in tiles of 64 columns; the same picks on 80 experts fill them, so the two per-entry times bound what a narrower routed tile can win.
void report_routed_occupancy() {
    const backend::BackendPtr timed = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *timed;
    const size_t nin = 2560, nout = 640, experts = 512, k = 10, row = quant::row_bytes(quant::GGML_TYPE_Q4_K, nin);
    std::vector<uint8_t> packed(experts * nout * row);
    uint32_t seed = 5;
    for (auto& b : packed) {
        seed = seed * 1664525u + 1013904223u;
        b = uint8_t(seed >> 24);
    }
    // Each block's scales are 1.0, so the products stay finite.
    for (size_t at = 0; at + 4 <= packed.size(); at += 144) {
        packed[at] = 0x00;
        packed[at + 1] = 0x3c;
        packed[at + 2] = 0x00;
        packed[at + 3] = 0x3c;
    }
    const auto w = vk.adopt(packed.data(), packed.size());
    for (const size_t rows : {size_t(512), size_t(2048)}) {
        std::vector<float> x(rows * nin);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 17) - 8) / 8.0f;
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(rows * k * nout * sizeof(float));
        for (const size_t over : {experts, size_t(80)}) {
            std::vector<uint32_t> ids(rows * k);
            for (size_t r = 0; r < rows; ++r)
                for (size_t j = 0; j < k; ++j) ids[r * k + j] = uint32_t((r * 37 + j * (over / k)) % over);
            const auto ib = vk.adopt(ids.data(), ids.size() * sizeof(uint32_t));
            const backend::RowRun run{rows, rows};
            const backend::Backend::Routing routing{{ib.get(), 0}, {}, k, experts};
            const auto call = [&] { vk.matmul_experts({{{quant::GGML_TYPE_Q4_K, {w.get(), 0}}, {y.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {&run, 1}, backend::Dtype::int8); };
            call();
            (void)backend::vulkan_kernel_times(vk);
            for (int i = 0; i < 3; ++i) call();
            double ms = 0;
            for (const auto& t : backend::vulkan_kernel_times(vk))
                if (t.first.rfind("matmul_tile", 0) == 0) ms += t.second;
            if (ms == 0) {
                std::cout << "backend-vulkan: the routed occupancy needs device timestamps\n";
                return;
            }
            ms /= 3;
            std::cout << "backend-vulkan: routed Q4_K 2560 to 640, " << rows << " rows of 10 picks over " << over << " experts, " << ms << " ms, "
                      << 1e6 * ms / double(rows * k) << " ns an entry\n";
        }
    }
}

// A row with no candidate (backend.hpp route_experts) on the CPU and the device: NaN scores give the invalid id with weight 0 at each pick, and one finite score among -inf takes that expert and then the others by lower id.
// The routed products leave an entry without an expert unwritten and add nothing for it, so a row without candidates keeps its output, as decode rows and as a prompt's on the tile.
size_t check_route_edges(backend::Backend& vk) {
    const size_t n_expert = 8, k = 4, nin = 256, nout = 64;
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const uint32_t none = 0xffffffffu;   // backend::Backend::kNoExpert
    auto cpu = backend::make_cpu_backend();
    size_t checks = 0;
    for (backend::Backend* b : {static_cast<backend::Backend*>(cpu.get()), &vk}) {
        const std::string on = b == cpu.get() ? "on the CPU" : "on the device";
        // Routing alone.
        {
            const size_t rows = 3;
            std::vector<float> scores(rows * n_expert, nan);
            for (size_t e = 0; e < n_expert; ++e) scores[n_expert + e] = e == 5 ? 1.0f : -inf;
            for (size_t e = 0; e < n_expert; ++e) scores[2 * n_expert + e] = e == 3 ? nan : 0.25f * (float)e;
            const auto sb = b->adopt(scores.data(), scores.size() * sizeof(float));
            const auto ib = b->alloc(rows * k * sizeof(float), backend::Memory::device), wb = b->alloc(rows * k * sizeof(float), backend::Memory::device);
            b->route_experts({sb.get(), 0}, rows, n_expert, k, true, {ib.get(), 0}, {wb.get(), 0});
            std::vector<uint32_t> ids(rows * k);
            std::vector<float> w(rows * k);
            b->read(*ib, 0, ids.data(), ids.size() * sizeof(uint32_t));
            b->read(*wb, 0, w.data(), w.size() * sizeof(float));
            const std::vector<uint32_t> want_ids = {none, none, none, none, 5, 0, 1, 2, none, none, none, none};
            const std::vector<float> want_w = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t i = 0; i < rows * k; ++i) {
                require(ids[i] == want_ids[i], ("routing a row with no candidate gave another id than the defined one " + on).c_str());
                require(std::memcmp(&w[i], &want_w[i], sizeof(float)) == 0, ("routing a row with no candidate gave another weight than the defined one " + on).c_str());
            }
            ++checks;
        }
        // A probability below FLT_MIN counts as 0 whatever the device's denormal mode, so the experts at gaps of 90, 95 and 100 go by lower id after the one at 87.
        {
            const std::vector<float> scores = {0.0f, -100.0f, -95.0f, -90.0f, -200.0f, -87.0f, -200.0f, -200.0f};
            const auto sb = b->adopt(scores.data(), scores.size() * sizeof(float));
            const auto ib = b->alloc(k * sizeof(float), backend::Memory::device), wb = b->alloc(k * sizeof(float), backend::Memory::device);
            b->route_experts({sb.get(), 0}, 1, n_expert, k, true, {ib.get(), 0}, {wb.get(), 0});
            std::vector<uint32_t> ids(k);
            std::vector<float> w(k);
            b->read(*ib, 0, ids.data(), ids.size() * sizeof(uint32_t));
            b->read(*wb, 0, w.data(), w.size() * sizeof(float));
            const std::vector<uint32_t> want_ids = {0, 5, 1, 2};
            for (size_t j = 0; j < k; ++j)
                require(ids[j] == want_ids[j], ("routing ordered probabilities below FLT_MIN by value rather than counting them as 0 " + on).c_str());
            require(w[0] == 1.0f && w[2] == 0.0f && w[3] == 0.0f && !std::signbit(w[2]) && !std::signbit(w[3]),
                    ("routing gave a probability below FLT_MIN another weight than 0 " + on).c_str());
            require(std::fabs(w[1] - std::exp(-87.0f)) <= 1e-4f * std::exp(-87.0f), ("routing changed the weight of a probability above FLT_MIN " + on).c_str());
            ++checks;
        }
        // The routed products: the gate-up call and the down call with its sum, on 2 decode rows and on a 70-row prompt read at extent 512, every second row without candidates.
        for (const size_t rows : {size_t(2), size_t(70)}) {
            const size_t entries = rows * k;
            const std::vector<backend::RowRun> prompt{{rows, 512}};
            const backend::RowRuns runs = rows > 2 ? backend::RowRuns{prompt.data(), prompt.size()} : backend::RowRuns{};
            std::vector<float> scores = uniform(rows * n_expert, 120 + (uint32_t)rows, -3.0f, 3.0f);
            for (size_t r = 0; r < rows; r += 2) scores[r * n_expert] = nan;
            const auto w8 = matrix(quant::GGML_TYPE_Q8_0, nin, n_expert * nout, 121);
            const auto x = uniform(rows * nin, 122), x2 = uniform(entries * nin, 123);
            const std::vector<float> sentinel(entries * nout, 7.0f), y0 = uniform(rows * nout, 124);
            const auto sb = b->adopt(scores.data(), scores.size() * sizeof(float)), eb = b->adopt(w8.data(), w8.size());
            const auto xb = b->adopt(x.data(), x.size() * sizeof(float)), x2b = b->adopt(x2.data(), x2.size() * sizeof(float));
            const auto ib = b->alloc(entries * sizeof(float), backend::Memory::device), wb = b->alloc(entries * sizeof(float), backend::Memory::device);
            const auto gb = b->alloc(entries * nout * sizeof(float), backend::Memory::device), yb = b->alloc(rows * nout * sizeof(float), backend::Memory::device);
            b->write(*gb, 0, sentinel.data(), sentinel.size() * sizeof(float));
            b->write(*yb, 0, y0.data(), y0.size() * sizeof(float));
            b->route_experts({sb.get(), 0}, rows, n_expert, k, true, {ib.get(), 0}, {wb.get(), 0});
            const backend::Backend::Routing routing{{ib.get(), 0}, {wb.get(), 0}, k, n_expert};
            b->matmul_experts({{{quant::GGML_TYPE_Q8_0, {eb.get(), 0}}, {gb.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, runs);
            b->matmul_experts_add({quant::GGML_TYPE_Q8_0, {eb.get(), 0}}, {x2b.get(), 0}, {yb.get(), 0}, nin, nout, rows, routing, runs);
            std::vector<float> g(entries * nout), y(rows * nout);
            b->read(*gb, 0, g.data(), g.size() * sizeof(float));
            b->read(*yb, 0, y.data(), y.size() * sizeof(float));
            for (size_t r = 0; r < rows; r += 2) {
                for (size_t i = r * k * nout; i < (r + 1) * k * nout; ++i)
                    require(g[i] == 7.0f, ("a routed product wrote the rows of an entry without an expert " + on).c_str());
                for (size_t o = 0; o < nout; ++o)
                    require(std::memcmp(&y[r * nout + o], &y0[r * nout + o], sizeof(float)) == 0, ("a routed sum added to a row without candidates " + on).c_str());
            }
            for (size_t r = 1; r < rows; r += 2)
                for (size_t o = 0; o < nout; ++o) require(std::isfinite(y[r * nout + o]), ("a routed sum beside a row without candidates is not finite " + on).c_str());
            ++checks;
        }
    }
    return checks;
}

// Routing at every expert count a build of moe_route covers, held to the CPU: ids exactly and weights within the routing check's 1e-6, normalized and not.
// Each row kind is a case of the contract: spread scores, ties across and inside lanes, NaN first, middle and last, +inf, all -inf, all NaN, fewer finite experts than k, and gaps either side of FLT_MIN.
size_t check_route_shapes() {
    const auto owner = backend::make_vulkan_backend(0, true);
    backend::Backend& vk = *owner;
    Pair p(vk);
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    // The expert count and the build it reaches, by the experts an invocation holds: moe_route_s1 up to 64, _s2 up to 128, _s4 up to 256, _s8 up to 512, moe_route up to 1024.
    struct Shape { size_t n; const char* build; };
    const Shape shapes[] = {{1, "moe_route_s1"}, {7, "moe_route_s1"}, {64, "moe_route_s1"}, {65, "moe_route_s2"}, {128, "moe_route_s2"}, {200, "moe_route_s4"},
                            {256, "moe_route_s4"}, {257, "moe_route_s8"}, {512, "moe_route_s8"}, {1000, "moe_route"}, {1024, "moe_route"}};
    size_t values = 0, witnessed = 0;
    uint32_t seed = 2400;
    for (const Shape& sh : shapes) {
        const size_t n = sh.n;
        std::vector<size_t> ks;
        for (size_t k : {size_t(1), size_t(8), size_t(10), n})
            if (k <= n && k <= 256 && std::find(ks.begin(), ks.end(), k) == ks.end()) ks.push_back(k);
        for (size_t k : ks)
            for (const bool normalize : {false, true}) {
                const size_t kinds = 11, rows = 3 * kinds;
                std::mt19937 rng(++seed);
                std::uniform_real_distribution<float> spread(-4.0f, 4.0f);
                std::vector<float> scores(rows * n);
                for (size_t r = 0; r < rows; ++r) {
                    float* s = scores.data() + r * n;
                    for (size_t e = 0; e < n; ++e) s[e] = spread(rng);
                    switch (r % kinds) {
                    case 1: for (size_t e = 0; e < n; ++e) s[e] = 0.5f * (float)(rng() % 4); break;
                    case 2: for (size_t e = 0; e < n; ++e) s[e] = e % 64 == r % 64 ? 2.0f : e % 64 == (r + 5) % 64 ? 1.0f : -3.0f; break;
                    case 3: s[0] = nan; break;
                    case 4: s[n / 2] = nan; break;
                    case 5: s[n - 1] = nan; break;
                    case 6: s[rng() % n] = inf; break;
                    case 7: for (size_t e = 0; e < n; ++e) s[e] = -inf; break;
                    case 8: for (size_t e = 0; e < n; ++e) s[e] = nan; break;
                    case 9:
                        for (size_t e = 0; e < n; ++e) s[e] = -inf;
                        for (size_t f = 0; f < (k + 1) / 2; ++f) s[rng() % n] = spread(rng);
                        break;
                    case 10: for (size_t e = 0; e < n; ++e) s[e] = -(float)(rng() % 121); break;
                    default: break;
                    }
                }
                Pair::In si = p.in(scores);
                Pair::Out ids = p.out(rows * k), wts = p.out(rows * k);
                p.cpu.route_experts(si.cs(), rows, n, k, normalize, ids.cs(), wts.cs());
                (void)backend::vulkan_kernel_times(vk);
                p.vk.route_experts(si.vs(), rows, n, k, normalize, ids.vs(), wts.vs());
                const auto ri = p.results(ids), rw = p.results(wts);
                const std::string label = std::to_string(n) + " experts (" + sh.build + "), k " + std::to_string(k) + (normalize ? ", normalized" : "");
                // Where the device times its dispatches, the call took the build the shape names.
                const auto times = backend::vulkan_kernel_times(vk);
                require(times.empty() || std::any_of(times.begin(), times.end(), [&](const auto& kt) { return kt.first == sh.build; }),
                        ("routing did not take its build at " + label).c_str());
                witnessed += !times.empty();
                for (size_t i = 0; i < rows * k; ++i)
                    if (std::memcmp(&ri.first[i], &ri.second[i], sizeof(float)) != 0) {
                        uint32_t c, d;
                        std::memcpy(&c, &ri.first[i], sizeof(c));
                        std::memcpy(&d, &ri.second[i], sizeof(d));
                        std::fprintf(stderr, "  row %zu (kind %zu), pick %zu: cpu %u device %u\n", i / k, i / k % kinds, i % k, c, d);
                        require(false, ("routing ids differ from the CPU at " + label).c_str());
                    }
                values += rows * k;
                values += close(rw.first, rw.second, 1e-6, ("routing weights differ beyond 1e-6 at " + label).c_str());
            }
    }
    // More experts than a build holds are refused.
    {
        const std::vector<float> scores(1025, 0.0f);
        Pair::In si = p.in(scores);
        Pair::Out ids = p.out(8), wts = p.out(8);
        bool refused = false;
        try {
            p.vk.route_experts(si.vs(), 1, 1025, 8, true, ids.vs(), wts.vs());
        } catch (const std::runtime_error&) {
            refused = true;
        }
        require(refused, "routing over 1025 experts was not refused on the device");
    }
    if (!witnessed) std::cout << "backend-vulkan: the device times no dispatches, so the routing builds were not witnessed\n";
    return values;
}

// Each fused routed build's float multiplies and adds less its build without the row products (`_noroles`) must be whole copies of its component one-column builds' (FloatOps).
// A component's residual add on each row it stores, which a fused role does not make, leaves its counts: one for a row family, the build's rows for the Q8_0 decode kernel.
struct MoeIsa { size_t checked = 0, missing = 0; };
MoeIsa check_moe_step_isa(const std::vector<std::pair<std::string, std::string>>& reps) {
    std::map<std::string, FloatOps> ops;
    for (const auto& kr : reps) ops[kr.first] = float_ops(kr.second);
    auto ends = [](const std::string& s, const std::string& t) { return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0; };
    auto starts = [](const std::string& s, const std::string& t) { return s.rfind(t, 0) == 0; };
    MoeIsa result;
    for (const auto& kr : reps) {
        const std::string& name = kr.first;
        if (!starts(name, "moe_step_") || starts(name, "moe_step_act") || ends(name, "_noroles")) continue;
        const auto bare = ops.find(name + "_noroles");
        require(bare != ops.end(), "a fused routed kernel has no build without its row products");
        const bool i8 = name.find("_i8") != std::string::npos;
        // The build suffix, and the Q8_0 decode kernel's name for the same one-column build.
        const std::string build = ends(name, "_1row") ? "_1row" : ends(name, "_2step") ? "_2step" : ends(name, "_1col") ? "_1col" : "";
        const std::string vec = std::string(i8 ? "matmul_vec_q8_i8" : "matmul_vec_q8") + (build == "_1row" ? "_1col_1row" : build == "_2step" ? "_1col_2step" : "_1col");
        const std::string dot = i8 ? "_i8_1col" : "_dot_1col";
        std::vector<std::string> parts;
        if (starts(name, "moe_step_route")) parts = {"matmul_row_f32_only" + build};
        else if (starts(name, "moe_step_up_k4q8")) parts = {"matmul_row_k4" + dot, vec};
        else if (starts(name, "moe_step_up_k4")) parts = {"matmul_row_k4" + dot};
        else if (starts(name, "moe_step_down_q5q8")) parts = {"matmul_row_q5" + dot, vec};
        else if (starts(name, "moe_step_down_q8")) parts = {vec};
        else if (starts(name, "moe_step_down_k")) parts = {"matmul_row_k" + dot};
        std::vector<FloatOps> c;
        for (const std::string& p : parts) {
            const auto it = ops.find(p);
            if (it == ops.end()) break;
            const size_t stores = p.rfind("matmul_vec_q8", 0) == 0 ? (ends(p, "_1row") ? 1 : 2) : 1;
            FloatOps own = it->second;
            own.add = own.add >= stores ? own.add - stores : 0;
            c.push_back(own);
        }
        if (c.size() != parts.size()) {
            std::cout << "backend-vulkan: " << name << "'s component builds were not dispatched, so its counts were not held\n";
            ++result.missing;
            continue;
        }
        const FloatOps full = ops[name], none = bare->second;
        const FloatOps delta{full.mul - none.mul, full.mad - none.mad, full.fused - none.fused, full.add - none.add, full.lane_add - none.lane_add};
        bool held = false;
        for (size_t a = 1; a <= 4 && !held; ++a)
            for (size_t b = c.size() > 1 ? 1 : 0; b <= (c.size() > 1 ? 4u : 0u) && !held; ++b) {
                auto at = [&](size_t i) { return i < c.size() ? c[i] : FloatOps{}; };
                const FloatOps x = at(0), y = at(1);
                held = delta.mul == a * x.mul + b * y.mul && delta.mad == a * x.mad + b * y.mad && delta.fused == a * x.fused + b * y.fused &&
                       delta.lane_add == a * x.lane_add + b * y.lane_add && delta.add == a * x.add + b * y.add;
            }
        std::cout << "backend-vulkan: " << name << " row products: mul " << delta.mul << ", mad " << delta.mad << ", fused " << delta.fused << ", add " << delta.add
                  << ", lane add " << delta.lane_add << (held ? ", whole copies of its components'" : ", NOT copies of its components'");
        for (size_t i = 0; i < c.size(); ++i)
            std::cout << "; " << parts[i] << " mul " << c[i].mul << ", mad " << c[i].mad << ", fused " << c[i].fused << ", add " << c[i].add << ", lane add " << c[i].lane_add;
        std::cout << "\n";
        require(held, "a fused routed kernel's row products are not those of the kernels whose bodies it holds");
        ++result.checked;
    }
    return result;
}

// The fused routed step (Backend::moe_step): a decode step of each type combination in four dispatches, none an unfused routing, combine or gate kernel, and a prompt's step unfused.
size_t check_moe_step_dispatches(backend::Backend& vk) {
    const moe::Shape shapes[] = {{2560, 16, 10, 640, moe::kFlash, true}, {2048, 16, 8, 512, moe::k35b, false}, {2048, 16, 8, 768, moe::k30b, false}};
    size_t steps = 0;
    for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8})
        for (const moe::Shape& s : shapes) {
            const moe::Weights w = moe::weights(vk, s, matrix, 31);
            const auto h = moe::inputs(2, s.E, 32), y0 = moe::residual(2, s.E, 33);
            auto kernels = [&](const std::vector<backend::RowRun>& runs, size_t rows) {
                vk.sync();
                (void)backend::vulkan_kernel_times(vk);
                moe::run(vk, w, h, y0, rows, runs, dtype, moe::How::op);
                std::vector<std::string> names;
                for (const auto& t : backend::vulkan_kernel_times(vk)) names.push_back(t.first);
                return std::make_pair(names, backend::vulkan_timed_dispatches(vk));
            };
            auto has = [](const std::vector<std::string>& names, const char* prefix) {
                for (const std::string& n : names) if (n.rfind(prefix, 0) == 0) return true;
                return false;
            };
            const auto decode = kernels(moe::decode_runs(1), 1);
            const size_t quantize = has(decode.first, "quantize_x") ? 1 : 0;
            require(decode.second - quantize == 4, "a fused routed decode step does not take four dispatches");
            require(has(decode.first, "moe_step_route") && has(decode.first, "moe_step_down"), "a routed decode step did not take the fused kernels");
            require(!has(decode.first, "moe_route") && !has(decode.first, "moe_combine") && !has(decode.first, "sigmoid_mul"),
                    "a fused routed decode step ran an unfused routing, combine or gate kernel");
            if (s.t.sh_gate) require(has(decode.first, "moe_step_up") && has(decode.first, "moe_step_act") && !has(decode.first, "silu_mul"),
                                     "a fused routed decode step with a shared expert ran an unfused SiLU");
            const std::vector<backend::RowRun> prompt = {{2, 2}};
            const auto prefill = kernels(prompt, 2);
            require(has(prefill.first, "moe_route") && !has(prefill.first, "moe_step_route"), "a prompt's routed step did not take the unfused kernels");
            ++steps;
        }
    return steps;
}

// The fused routed step against its base sequence on the same device, every poisoned output bit for bit: each type combination at its file's widths, and at small widths with 60 to 1000 experts.
size_t check_moe_step(backend::Backend& vk) {
    struct Case { moe::Shape s; std::vector<size_t> rows; };
    const Case cases[] = {
        {{2560, 16, 10, 640, moe::kFlash, true}, {1, 2, 3}},
        {{2048, 16, 8, 512, moe::k35b, false}, {1, 2, 3}},
        {{2048, 16, 8, 768, moe::k30b, false}, {1, 2, 3}},
        {{256, 60, 8, 64, moe::k30b, false}, {1, 5, 7}},
        {{256, 100, 8, 64, moe::kFlash, true}, {1, 4, 12}},
        {{256, 200, 8, 256, moe::k35b, false}, {1, 9, 24}},
        {{256, 512, 10, 64, moe::kFlash, false}, {1, 17, 64}},
        {{256, 1000, 8, 64, moe::k30b, true}, {1, 33, 64}},
    };
    size_t values = 0;
    for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8})
        for (const Case& c : cases) {
            const moe::Weights w = moe::weights(vk, c.s, matrix, 41);
            for (size_t rows : c.rows) {
                const auto h = moe::inputs(rows, c.s.E, 42 + uint32_t(rows)), y0 = moe::residual(rows, c.s.E, 43);
                const auto runs = moe::decode_runs(rows);
                std::string what;
                const size_t n = moe::same(moe::run(vk, w, h, y0, rows, runs, dtype, moe::How::op), moe::run(vk, w, h, y0, rows, runs, dtype, moe::How::base), what);
                if (!n) {
                    std::fprintf(stderr, "  moe step: E %zu, %zu experts, ff %zu, %zu rows, %s\n", c.s.E, c.s.n_expert, c.s.ff, rows, backend::dtype_name(dtype));
                    throw std::runtime_error("a fused routed step's " + what + " differ from its unfused calls");
                }
                values += n;
            }
        }
    return values;
}

// A token's fused step at 1 to 64 tokens gives the bits of the token alone, as do 50 steps of random counts with unfused steps between, so the routing's counter is zero after every call.
size_t check_moe_step_columns(backend::Backend& vk) {
    const moe::Shape shapes[] = {{256, 512, 10, 64, moe::kFlash, true}, {256, 512, 8, 256, moe::k35b, false}, {256, 512, 8, 64, moe::k30b, false}};
    const size_t widest = 64;
    size_t columns = 0;
    std::mt19937 rng(57);
    for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8})
        for (const moe::Shape& s : shapes) {
            const moe::Weights w = moe::weights(vk, s, matrix, 51);
            const bool shared = s.t.sh_gate != 0;
            const auto h = moe::inputs(widest, s.E, 52), y0 = moe::residual(widest, s.E, 53);
            // Token t's outputs from a step over `rows` tokens from `first`.
            auto token = [&](const moe::Out& o, size_t rows, size_t t) {
                moe::Out one;
                auto part = [](const std::vector<float>& v, size_t at, size_t n) { return std::vector<float>(v.begin() + long(at), v.begin() + long(at + n)); };
                one.y = part(o.y, t * s.E, s.E);
                one.scores = part(o.scores, t * s.n_expert, s.n_expert);
                one.weights = part(o.weights, t * s.k, s.k);
                one.ids.assign(o.ids.begin() + long(t * s.k), o.ids.begin() + long((t + 1) * s.k));
                one.row_gate = part(o.row_gate, t, 1);
                for (auto* v : {&one.g, &one.u, &one.act}) {
                    const std::vector<float>& src = v == &one.g ? o.g : v == &one.u ? o.u : o.act;
                    *v = part(src, t * s.k * s.ff, s.k * s.ff);
                    if (shared) {
                        const std::vector<float> sh = part(src, rows * s.k * s.ff + t * s.ff, s.ff);
                        v->insert(v->end(), sh.begin(), sh.end());
                    }
                }
                return one;
            };
            auto step = [&](size_t first, size_t rows, bool base) {
                const std::vector<float> hh(h.begin() + long(first * s.E), h.begin() + long((first + rows) * s.E));
                const std::vector<float> yy(y0.begin() + long(first * s.E), y0.begin() + long((first + rows) * s.E));
                return moe::run(vk, w, hh, yy, rows, moe::decode_runs(rows), dtype, base ? moe::How::base : moe::How::op);
            };
            std::vector<moe::Out> alone;
            for (size_t t = 0; t < widest; ++t) alone.push_back(step(t, 1, false));
            auto check = [&](size_t rows, const char* what) {
                const moe::Out o = step(0, rows, false);
                for (size_t t = 0; t < rows; ++t) {
                    std::string which;
                    if (!moe::same(token(o, rows, t), alone[t], which)) {
                        std::fprintf(stderr, "  moe step columns: token %zu of %zu, E %zu, ff %zu, %s\n", t, rows, s.E, s.ff, backend::dtype_name(dtype));
                        throw std::runtime_error(std::string(what) + ": " + which);
                    }
                }
                columns += rows;
            };
            for (size_t rows = 1; rows <= widest; ++rows) check(rows, "a fused routed step's token differs from the same token alone");
            std::uniform_int_distribution<size_t> pick(1, widest);
            for (int i = 0; i < 50; ++i) {
                if (i % 3 == 0) (void)step(0, pick(rng), true);
                check(pick(rng), "a fused routed step after other steps differs from the same token alone");
            }
        }
    return columns;
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
        size_t checks = check_collective(b);
        checks += check_route_edges(*b);
        std::cout << "backend-vulkan: " << check_route_shapes() << " routing picks and weights at every routing build against the CPU\n";
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

        // Reusing the ring after a newer wait must still expose fresh writes; two host waiters retire older tickets too.
        for (size_t round = 0; round < 32; ++round) {
            const size_t offset = 64 * round;
            b->copy(*visible, offset, *adopted, 4096 + offset, 64);
            const backend::Ticket first = b->submit();
            b->copy(*visible, 2048 + offset, *adopted, 8192 + offset, 64);
            const backend::Ticket next = b->submit();
            std::thread newer([&] { b->wait(next); });
            b->wait(first);
            newer.join();
            require(b->done(first) && b->done(next), "a retired ticket is not done");
            b->wait(first);
            require(std::memcmp((const uint8_t*)visible->host_ptr() + offset, kept.data() + 4096 + offset, 64) == 0 &&
                    std::memcmp((const uint8_t*)visible->host_ptr() + 2048 + offset, kept.data() + 8192 + offset, 64) == 0,
                    "a newer wait left a reused ring slot's copy incomplete");
            checks += 2;
        }

        // A command buffer opened with no dispatch running is submitted after its first 8 dispatches, one opened behind running dispatches after the profile's 64.
        {
            const size_t big = size_t(16) << 20;
            const auto x = b->alloc(big * sizeof(float)), y = b->alloc(big * sizeof(float));
            b->sync();
            const backend::Ticket q0 = b->submit();
            b->wait(q0);
            for (int i = 0; i < 8; ++i) b->add(backend::Slice{y.get(), 0}, backend::CSlice{x.get(), 0}, 64);
            const backend::Ticket q1 = b->submit();
            require(q1 == q0 + 2, "8 dispatches on an idle device were not submitted on their own");
            b->wait(q1);
            // 8 adds of 16M floats keep the device busy while the host records the next 64, so those go as one chunk and the last call after them as another.
            for (int i = 0; i < 8; ++i) b->add(backend::Slice{y.get(), 0}, backend::CSlice{x.get(), 0}, big);
            for (int i = 0; i < 65; ++i) b->add(backend::Slice{y.get(), 0}, backend::CSlice{x.get(), 0}, 64);
            const backend::Ticket q2 = b->submit();
            require(q2 == q1 + 3, "dispatches behind running ones were not submitted in a chunk of 64");
            b->wait(q2);
            checks += 2;
        }

        // A device holding between submissions (Backend::hold_between_submissions): work behind a hold runs once the next submission lets it go, also after an idle gap the hold's own bound ends, and a backend dropped with a hold pending tears down.
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
        std::cout << "backend-vulkan: " << check_half_weights(*b) << " finite half-weight values and exact F32-control products\n";

        std::cout << "backend-vulkan: " << testq::check_matrix_precision(*b) << " shared matrix precision values passed\n";
        std::cout << "backend-vulkan: " << check_q5_embedding(*b, quant::GGML_TYPE_Q5_1) << " exact Q5_1 embedding values passed\n";
        std::cout << "backend-vulkan: " << check_q5_embedding(*b, quant::GGML_TYPE_Q5_0) << " exact Q5_0 embedding values passed\n";
        std::cout << "backend-vulkan: " << check_large_table(*b) << " embedding values of tables past 4 GiB passed\n";
        std::cout << "backend-vulkan: " << check_matrix_witness(*b) << " matrix-path witnesses match arithmetic\n";
        const size_t int8_values = check_int8(*b);
        std::cout << "backend-vulkan: " << int8_values << " int8 outputs against the CPU on 8-bit activations, sliced prompts, and types without an 8-bit build as f16\n";
        std::cout << "backend-vulkan: " << check_int8_twin_width() << " int8 outputs unchanged when the input row width changes\n";
        std::cout << "backend-vulkan: " << check_write_sets() << " kernel bindings written or read as their shaders declare them\n";
        std::cout << "backend-vulkan: " << check_twin_table() << " products from kept twins equal to products from fresh copies\n";
        const size_t values = check_kernels(*b) + check_qwen35(*b) + vision::check_ops(*b);
        std::cout << "backend-vulkan: " << checks << " storage and submission checks; "
                  << values << " kernel outputs against the CPU backend\n";
        std::cout << "backend-vulkan: " << check_float_workspace(*b) << " exact products across bounded float workspace slices\n";
        report_routed_occupancy();
        const size_t narrow_tiles = check_routed_narrow();
        std::cout << "backend-vulkan: " << narrow_tiles << " routed narrow tiles equal to the 64-column build\n";
        std::cout << "backend-vulkan: " << check_timing_coverage() << " dispatches timed between two readings, past one query pool\n";
        print_argmax_after_head();
        std::cout << "backend-vulkan: " << check_bf16_batch_split(*b) << " BF16 outputs invariant across physical tile boundaries\n";
        const size_t columns = check_decode_columns(*b, backend::Dtype::f16);
        std::cout << "backend-vulkan: " << columns << " decode columns equal to the same columns alone\n";
        const auto native = b->native_dtypes();
        if (std::find(native.begin(), native.end(), backend::Dtype::int8) != native.end())
            std::cout << "backend-vulkan: " << check_decode_columns(*b, backend::Dtype::int8) << " int8 decode columns equal to the same columns alone\n";
        // The Q8_0 decode kernel on a backend taking the integer dot, which a device whose profile does not prefer it reaches only here.
        backend::BackendPtr q8b = backend::make_vulkan_backend(0, true);
        const bool q8_forced = !backend::vulkan_device_profile(*q8b).prefer_integer_dot;
        if (q8_forced && !backend::vulkan_prefer_integer_dot(*q8b)) {
            std::cout << "backend-vulkan: no integer dot, so no Q8_0 decode kernel to check\n";
            q8b.reset();
        } else {
            std::cout << "backend-vulkan: " << check_q8_one_column_builds(*q8b) << " one-column Q8_0 decode calls in the build their shape gives\n";
            std::cout << "backend-vulkan: " << check_q8_filling_builds(*q8b) << " Q8_0 decode calls of 2 to 8 columns in the build their workgroups give\n";
            std::cout << "backend-vulkan: " << check_moe_step_dispatches(*q8b) << " fused routed decode steps in four dispatches, prompts' steps unfused\n";
            std::cout << "backend-vulkan: " << check_moe_step(*q8b) << " fused routed step outputs equal to its unfused calls'\n";
            std::cout << "backend-vulkan: " << check_moe_step_columns(*q8b) << " fused routed step tokens equal to the same tokens alone, after any step\n";
            std::cout << "backend-vulkan: " << moe::check_row_classes(*q8b, matrix) << " pairs of extents of one class with the same bits through the fused routed step\n";
            if (q8_forced)
                for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8})
                    std::cout << "backend-vulkan: " << check_decode_columns(*q8b, dtype, true) << " Q8_0 decode columns on the integer dot, " << backend::dtype_name(dtype)
                              << ", equal to the same columns alone\n";
        }
        std::cout << "backend-vulkan: " << check_bf16_rounding(*b) << " BF16 boundary and unchanged-weight products, with F32 control\n";
        std::cout << "backend-vulkan: " << check_bf16_weight_decode(*b) << " BF16 weight embedding and indexer projection values\n";
        const size_t precise = check_activation_precision(*b, backend::Dtype::f16);
        std::cout << "backend-vulkan: " << precise << " outputs within the 16-bit activations' precision\n";
        std::cout << "backend-vulkan: " << check_activation_precision(*b, backend::Dtype::f32) << " F32 outputs against double dots and batch identity\n";
        std::cout << "backend-vulkan: " << check_activation_precision(*b, backend::Dtype::bf16) << " BF16 outputs against rounded-input double dots and batch identity\n";
        std::cout << "backend-vulkan: " << check_tq_range(*b) << " TQ edge-scale matrix paths and signed-zero embeddings\n";
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
            // Where only the backend taking the integer dot built the Q8_0 decode kernel, its builds are written and held to their shapes too.
            if (q8b && q8_forced) {
                std::vector<std::pair<std::string, std::string>> q8_builds;
                for (const auto& kr : backend::vulkan_kernel_representations(*q8b))
                    if (kr.first.rfind("matmul_vec_q8", 0) == 0) q8_builds.push_back(kr);
                size_t q8_written = 0;
                for (const auto& kr : q8_builds) {
                    std::ofstream f(isa_dir + "/" + kr.first + ".txt");
                    f << kr.second;
                    q8_written += f.good() ? 1 : 0;
                }
                const ContractionChecks q = check_contraction(q8_builds);
                std::cout << "backend-vulkan: " << q8_written << " Q8_0 decode builds of the integer dot written; against their one-column build: " << q.decode
                          << " those their shape and forms give, " << q.kinds_only << " their kinds only\n";
            }
            // The fused routed kernels the backend taking the integer dot built, with their builds without the row products, held to the kernels whose bodies they hold.
            if (q8b) {
                const auto moe_reps = backend::vulkan_kernel_representations(*q8b);
                size_t moe_written = 0;
                for (const auto& kr : moe_reps)
                    if (kr.first.rfind("moe_step_", 0) == 0) {
                        std::ofstream f(isa_dir + "/" + kr.first + ".txt");
                        f << kr.second;
                        moe_written += f.good() ? 1 : 0;
                    }
                const MoeIsa m = check_moe_step_isa(moe_reps);
                std::cout << "backend-vulkan: " << moe_written << " fused routed kernel builds written; " << m.checked << " held to whole copies of their components' products, "
                          << m.missing << " without their components dispatched\n";
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
