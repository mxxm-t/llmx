#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <string>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "matrix_precision.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif
#include "quant/quant.hpp"

namespace {

void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, sizeof out); return out; }

// Independent base-three encoding of the published eight signed-byte dictionary vectors; see docs/src/quant-iq1-grid.md.
constexpr uint16_t grid[2048] = {
#include "iq1_grid.inc"
};

int ternary(unsigned index, size_t lane) {
    unsigned vector = grid[index];
    for (size_t i = 0; i < lane; ++i) vector /= 3;
    return int(vector % 3) - 1;
}

double half_value(uint16_t h) {
    const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
    double value = exponent ? std::ldexp(double(1024 + fraction), int(exponent) - 25)
                            : std::ldexp(double(fraction), -24);
    return h & 0x8000 ? -value : value;
}

void field(uint8_t* p, size_t bit, unsigned width, unsigned value) {
    for (unsigned i = 0; i < width; ++i) {
        const size_t at = bit + i;
        p[at / 8] = uint8_t((p[at / 8] & ~(1u << (at % 8))) | (((value >> i) & 1u) << (at % 8)));
    }
}

struct Fixture {
    uint16_t half;
    std::array<unsigned, 32> indices{}, scales{}, shifts{};
};

Fixture pack(uint8_t* p, bool medium, uint16_t half, unsigned seed) {
    Fixture fixture{};
    fixture.half = half;
    std::memset(p, 0, medium ? 56 : 50);
    if (medium) {
        for (unsigned nibble = 0; nibble < 4; ++nibble)
            field(p, 48 * 8 + nibble * 16 + 12, 4, (half >> (4 * nibble)) & 15);
    } else field(p, 0, 16, half);
    for (size_t vector = 0; vector < 32; ++vector) {
        fixture.indices[vector] = (seed + unsigned(vector) * 67) % 2048;
        fixture.scales[vector] = (seed + unsigned(vector / (medium ? 2 : 4))) % 8;
        fixture.shifts[vector] = (seed + unsigned(vector / (medium ? 1 : 4))) % 2;
        const unsigned index = fixture.indices[vector];
        if (medium) {
            field(p, vector * 8, 8, index & 255);
            field(p, 32 * 8 + vector * 4, 3, index >> 8);
            field(p, 32 * 8 + vector * 4 + 3, 1, fixture.shifts[vector]);
            field(p, 48 * 8 + (vector / 8) * 16 + ((vector % 8) / 2) * 3, 3, fixture.scales[vector]);
        } else {
            field(p, 16 + vector * 8, 8, index & 255);
            field(p, 34 * 8 + (vector / 4) * 16 + (vector % 4) * 3, 3, index >> 8);
            field(p, 34 * 8 + (vector / 4) * 16 + 12, 3, fixture.scales[vector]);
            field(p, 34 * 8 + (vector / 4) * 16 + 15, 1, fixture.shifts[vector]);
        }
    }
    return fixture;
}

float expected(const Fixture& f, size_t i) {
    return float(half_value(f.half) * double(1 + 2 * f.scales[i / 8]) *
                 (double(ternary(f.indices[i / 8], i % 8)) + (f.shifts[i / 8] ? -0.125 : 0.125)));
}

size_t check_blocks(bool medium, uint16_t half, unsigned seed) {
    const auto* qt = quant::Registry::instance().get(medium ? 29 : 19);
    std::array<uint8_t, 2 * 56 + 2> input{};
    std::array<float, 2 * 256 + 2> output{};
    std::array<Fixture, 2> fixtures{};
    input.fill(0xa5);
    output.fill(123);
    for (size_t b = 0; b < 2; ++b)
        fixtures[b] = pack(input.data() + 1 + b * qt->type_size, medium, uint16_t(half ^ (b ? 0x8000 : 0)), seed + unsigned(b));
    const auto saved = input;
    qt->dequantize(input.data() + 1, output.data() + 1, 2);
    require(input == saved && output.front() == 123 && output.back() == 123, "IQ1 input or output guards changed");
    for (size_t b = 0; b < 2; ++b) {
        const auto& f = fixtures[b];
        for (size_t i = 0; i < 256; ++i) {
            require(bits(output[1 + b * 256 + i]) == bits(expected(f, i)), "IQ1 field decode differs, including signed zero");
        }
    }
    return 512;
}

void packed_reconstruction() {
    size_t checked = 0;
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        const float d = float(half_value(uint16_t(h)));
        for (int odd = 1; odd <= 15; odd += 2)
            for (int shift : {-1, 1})
                for (int value = -1; value <= 1; ++value) {
                    const float scalar = (d * float(odd)) * (float(value) + float(shift) * 0.125f);
                    const float packed = (d * 0.125f) * float(odd * (8 * value + shift));
                    require(bits(scalar) == bits(packed), "IQ1 proposed packed reconstruction differs");
                    ++checked;
                }
    }
    std::cout << "IQ1 exact packed reconstruction: " << checked << " finite-half/odd/shift/ternary cases\n";
}

std::vector<float> read(backend::Backend& b, backend::Buffer& buffer, size_t count) {
    std::vector<float> out(count);
    b.read(buffer, 0, out.data(), count * sizeof(float));
    return out;
}

float bf16_input(float x) {
    uint32_t u = bits(x), high = u >> 16, low = u & 65535;
    if (low > 32768 || (low == 32768 && (high & 1))) ++high;
    u = high << 16;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

size_t crossover(backend::Backend& b, size_t width) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu()) return backend::tile_from_for(backend::vulkan_device_profile(b), false, width);
#else
    (void)b; (void)width;
#endif
    return 64;
}

bool integer_path(backend::Backend& b, backend::Dtype dtype, bool tile) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(b);
        return tile ? profile.prefer_integer_dot : profile.lookup_integer_dot;
    }
#else
    (void)b; (void)dtype; (void)tile;
#endif
    return false;
}

void witness(backend::Backend& b, backend::Dtype dtype, bool integer) {
    require(testq::take_matrix_paths(b) == std::vector<std::string>{
                dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
            "IQ1 requested policy arithmetic witness differs");
}

#if LLMX_HAS_BACKEND_VULKAN
void kernel_witness(backend::Backend& b, bool medium, bool tile, bool integer) {
    const auto times = backend::vulkan_kernel_times(b);
    if (!backend::vulkan_timed_dispatches(b)) {
        std::cout << "IQ1 timestamp witness unavailable\n";
        return;
    }
    const std::string family = std::string(tile ? "matmul_tile_iq1_" : "matmul_row_iq1_") + (medium ? "m" : "s");
    bool found = false;
    for (const auto& entry : times) {
        if (entry.first.find("matmul_") != 0 || entry.first.find("matmul_reduce") == 0) continue;
        require(entry.first.find(family) == 0, "IQ1 selected an unexpected row/tile format kernel");
        require((entry.first.find("_dot") != std::string::npos) == integer, "IQ1 integer shader witness differs");
        found = true;
    }
    require(found, "IQ1 matrix call lacks its actual shader timing witness");
}
#endif

struct MatrixFixture { std::vector<uint8_t> bytes; std::vector<float> values; };
MatrixFixture matrix_fixture(bool medium, size_t rows, size_t width, bool edges = false) {
    const size_t stride = medium ? 56 : 50;
    MatrixFixture out{std::vector<uint8_t>(8 + rows * width / 256 * stride, 0xa5), std::vector<float>(rows * width)};
    const uint16_t halves[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    for (size_t r = 0; r < rows; ++r) for (size_t block = 0; block < width / 256; ++block) {
        const uint16_t half = edges ? halves[(r + block) % 10] : uint16_t((r + block) % 2 ? 0xb555 : 0x3555);
        const auto f = pack(out.bytes.data() + 4 + (r * width / 256 + block) * stride,
                            medium, half, unsigned(r + block * 137));
        for (size_t i = 0; i < 256; ++i) out.values[r * width + block * 256 + i] = expected(f, i);
    }
    return out;
}

size_t embeddings(backend::Backend& b, bool medium) {
    constexpr size_t rows = 2048, width = 512;
    const uint32_t type = medium ? 29 : 19;
    const auto f = matrix_fixture(medium, rows, width, true);
    auto w = b.adopt(f.bytes.data(), f.bytes.size());
    std::vector<uint32_t> ids(rows);
    for (size_t i = 0; i < rows; ++i) ids[i] = uint32_t(rows - 1 - i);
    std::vector<float> guards(rows * width + 2, 99);
    auto y = b.alloc(guards.size() * sizeof(float));
    b.write(*y, 0, guards.data(), guards.size() * sizeof(float));
    b.embed({y.get(), 1}, {type, {w.get(), 1}}, width, rows, ids.data(), rows);
    auto got = read(b, *y, guards.size());
    require(got.front() == 99 && got.back() == 99, "IQ1 embedding output guards");
    for (size_t r = 0; r < rows; ++r) for (size_t i = 0; i < width; ++i)
        require(bits(got[1 + r * width + i]) == bits(f.values[ids[r] * width + i]), "IQ1 embedding fields or signed zero differ");
    const uint32_t picks[] = {0, 1, 3, 2047, 2047, uint32_t(rows), UINT32_MAX};
    auto selected = b.adopt(picks, sizeof picks);
    b.write(*y, 0, guards.data(), guards.size() * sizeof(float));
    b.embed_ids({y.get(), 1}, {type, {w.get(), 1}}, width, rows, {selected.get(), 0}, 7);
    got = read(b, *y, guards.size());
    for (size_t i = 0; i < got.size(); ++i) {
        float want = 99;
        if (i && i <= 7 * width) {
            const size_t row = (i - 1) / width, col = (i - 1) % width;
            want = picks[row] < rows ? f.values[picks[row] * width + col] : 0;
        }
        require(bits(got[i]) == bits(want), "IQ1 embed_ids valid/invalid rows or output guards differ");
    }
    return rows * width + 7 * width;
}

size_t products(backend::Backend& b, bool medium, size_t rows, size_t cols, size_t extent, bool one_hot, bool all_ops) {
    constexpr size_t width = 768;
    const uint32_t type = medium ? 29 : 19;
    const size_t count = rows * cols;
    const auto f = matrix_fixture(medium, 2 * rows, width);
    auto weights = b.adopt(f.bytes.data(), f.bytes.size());
    std::vector<float> input(cols * width);
    for (size_t c = 0; c < cols; ++c) {
        if (one_hot) input[c * width + (c * 137 + 31) % width] = 1;
        else for (size_t i = 0; i < width; ++i)
            input[c * width + i] = float(int((i * 17 + c * 23) % 61) - 30) / 37;
    }
    auto rounded = input;
    for (auto& x : rounded) x = bf16_input(x);
    const auto twin = testq::row_activations(input);
    auto x = b.adopt(input.data(), input.size() * sizeof(float));
    auto output = b.alloc((2 * count + 2) * sizeof(float));
    std::vector<uint32_t> expert_ids(cols);
    for (size_t c = 0; c < cols; ++c) expert_ids[c] = uint32_t(c % 2);
    std::vector<float> gains(cols, 0.5f);
    auto id_buffer = b.adopt(expert_ids.data(), cols * sizeof(uint32_t));
    auto gain_buffer = b.adopt(gains.data(), cols * sizeof(float));
    const backend::Backend::Routing routing{{id_buffer.get(), 0}, {gain_buffer.get(), 0}, 1, 2};
    const backend::RowRun run{cols, extent};
    const backend::RowRuns runs{&run, 1};
    size_t checked = 0;
    for (int op = 0; op < (all_ops ? 6 : 1); ++op)
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::f16, backend::Dtype::int8}) {
            const bool tile = dtype == backend::Dtype::bf16 || extent >= crossover(b, width);
            // Routed controls use two columns at decode extent, below every routed tile crossover.
            const bool integer = integer_path(b, dtype, tile);
            const bool add = op == 3 || op == 5;
            std::vector<float> initial(2 * count + 2, add ? 0.25f : 19);
            initial.front() = initial.back() = 19;
            b.write(*output, 0, initial.data(), initial.size() * sizeof(float));
            testq::take_matrix_paths(b);
#if LLMX_HAS_BACKEND_VULKAN
            if (!b.is_cpu()) backend::vulkan_kernel_times(b);
#endif
            const backend::CSlice w{weights.get(), 1}, in{x.get(), 0};
            const backend::Slice out{output.get(), 1};
            switch (op) {
            case 0: b.matmul({type, w}, in, out, width, rows, cols, runs, dtype); break;
            case 1: b.matmul_logits({type, w}, in, out, width, rows, cols, runs, dtype); break;
            case 2: b.matmul_group({{{type, w}, out, rows}, {{type, w}, {output.get(), count + 1}, rows}}, in, width, cols, runs, dtype); break;
            case 3: b.matmul_add({type, w}, in, out, width, rows, cols, runs, dtype); break;
            case 4: b.matmul_experts({{{type, w}, out, rows}}, in, width, cols, routing, runs, dtype); break;
            case 5: b.matmul_experts_add({type, w}, in, out, width, rows, cols, routing, runs, dtype); break;
            }
            const auto got = read(b, *output, initial.size());
            witness(b, dtype, integer);
#if LLMX_HAS_BACKEND_VULKAN
            if (!b.is_cpu() && !all_ops) kernel_witness(b, medium, tile, integer);
#endif
            const auto& oracle = dtype == backend::Dtype::bf16 ? rounded : integer ? twin : input;
            const size_t copies = op == 2 ? 2 : 1;
            require(got.front() == initial.front(), "IQ1 product leading guard");
            for (size_t i = copies * count + 1; i < got.size(); ++i)
                require(bits(got[i]) == bits(initial[i]), "IQ1 product output tail or guard");
            for (size_t copy = 0; copy < copies; ++copy)
                for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
                    const float* wr = f.values.data() + ((op >= 4 ? expert_ids[c] * rows : 0) + r) * width;
                    const double gain = op == 5 ? 0.5 : 1;
                    double want = add ? 0.25 : 0, magnitude = add ? 0.25 : 0, original = want, reconstruction = 0;
                    for (size_t i = 0; i < width; ++i) {
                        const double term = gain * double(wr[i]) * oracle[c * width + i];
                        want += term; magnitude += std::fabs(term);
                        original += gain * double(wr[i]) * input[c * width + i];
                        reconstruction += gain * std::fabs(double(wr[i]) * (double(input[c * width + i]) - oracle[c * width + i]));
                    }
                    const float have = got[1 + copy * count + c * rows + r];
                    const double nu = 2 * double(width) * 0x1p-24;
                    const double bound = nu / (1 - nu) * magnitude;
                    require(std::isfinite(have) && std::fabs(double(have) - want) <= bound, "IQ1 product exceeds gamma(2n) double bound");
                    if (one_hot && !integer && !add)
                        require(bits(have) == bits(float(want)), "IQ1 exact one-hot selected weight differs");
                    if (integer)
                        require(std::fabs(double(have) - original) <= reconstruction + bound, "IQ1 integer product exceeds original-input reconstruction bound");
                    ++checked;
                }
        }
    require(read(b, *x, input.size()) == input, "IQ1 product changed inputs");
    std::vector<uint8_t> after(f.bytes.size()); b.read(*weights, 0, after.data(), after.size());
    require(after == f.bytes, "IQ1 product changed packed weights or guards");
    return checked;
}

size_t ranges(backend::Backend& b, bool medium) {
    const uint32_t type = medium ? 29 : 19;
    size_t checked = 0;
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x03ff),
                       uint16_t(0x0400), uint16_t(0x3c00), uint16_t(0xbc00), uint16_t(0x7bff), uint16_t(0xfbff)}) {
        std::vector<uint8_t> bytes(4 + (medium ? 56 : 50) + 4, 0xa5);
        const auto f = pack(bytes.data() + 4, medium, h, 7);
        auto w = b.adopt(bytes.data(), bytes.size());
        for (float value : {0.0f, 0x1p-145f, 0x1p-120f, 0x1p-24f, 65504.0f, 0x1p100f})
            for (float sign : {-1.0f, 1.0f}) {
                std::vector<float> input(256, 0); input[0] = sign * value;
                auto x = b.adopt(input.data(), input.size() * sizeof(float));
                auto y = b.alloc(3 * sizeof(float));
                for (size_t extent : {size_t(1), size_t(16384)})
                    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8}) {
                        const float guards[] = {19, 0, 19}; b.write(*y, 0, guards, sizeof guards);
                        const backend::RowRun run{1, extent};
                        testq::take_matrix_paths(b);
                        b.matmul({type, {w.get(), 1}}, {x.get(), 0}, {y.get(), 1}, 256, 1, 1, {&run, 1}, dtype);
                        const auto got = read(b, *y, 3);
                        const bool integer = integer_path(b, dtype, extent >= crossover(b, 256));
                        witness(b, dtype, integer);
                        const double weight = expected(f, 0), want = weight * input[0];
                        const double repair = integer ? std::fabs(double(value)) * 0x1p-22 +
                            32767.0 * std::numeric_limits<float>::denorm_min() : 0;
                        const double bound = std::fabs(weight) * repair + std::fabs(want) * 0x1p-21 +
                            2.0 * std::numeric_limits<float>::denorm_min();
                        require(got.front() == 19 && got.back() == 19 && std::isfinite(got[1]) &&
                                    std::fabs(double(got[1]) - want) <= bound, "IQ1 finite-half/activation range repair differs");
                        ++checked;
                    }
            }
    }
    return checked;
}

void operators(backend::Backend& b) {
    require(bf16_input(1.00390625f) == 1 && bf16_input(1.01171875f) == 1.015625f, "IQ1 independent BF16 tie anchors");
    size_t embedded = 0, multiplied = 0;
    for (bool medium : {false, true}) {
        require(b.supports_type(medium ? 29 : 19), "IQ1 backend support missing");
        embedded += embeddings(b, medium);
        for (bool one_hot : {false, true}) {
            multiplied += products(b, medium, 3, 2, 1, one_hot, true);
            const size_t from = crossover(b, 768);
            for (size_t count : {from - 1, from, from + 1})
                multiplied += products(b, medium, 67, count, count, one_hot, false);
        }
        multiplied += ranges(b, medium);
    }
    std::cout << "IQ1 operators: " << embedded << " exact embed/embed_ids values, " << multiplied
              << " matrix/range values across six calls, F32/BF16/F16/int8 policies and crossover edges [ok]\n";
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        try { auto b = backend::make_vulkan_backend(0, true); operators(*b); return 0; }
        catch (const backend::VulkanUnavailable& e) { std::cout << "IQ1 Vulkan unavailable: " << e.what() << '\n'; return 77; }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
#else
        std::cout << "IQ1 Vulkan backend not built\n"; return 77;
#endif
    }
    if (argc != 1) { std::cerr << "usage: llmx-iq1-test [--vulkan]\n"; return 1; }
    try {
        for (unsigned index = 0; index < 2048; ++index)
            for (size_t lane = 0; lane < 8; ++lane)
                require(quant::iq1_value(index, lane) == ternary(index, lane), "IQ1 dictionary differs from independent base-three data");
        size_t checked = 0;
        for (bool medium : {false, true}) {
            const auto* qt = quant::Registry::instance().get(medium ? 29 : 19);
            require(qt && qt->block_size == 256 && qt->type_size == (medium ? 56u : 50u) && !qt->quantize && qt->dequantize,
                    "IQ1 read-only registry differs");
            require(quant::row_bytes(medium ? 29 : 19, 768, 2) == 6 * qt->type_size, "IQ1 row size differs");
            qt->dequantize(nullptr, nullptr, 0);
            for (unsigned index = 0; index < 2048; ++index)
                checked += check_blocks(medium, 0x3c00, index);
            for (unsigned half = 0; half < 65536; ++half)
                if ((half & 0x7c00) != 0x7c00)
                    checked += check_blocks(medium, uint16_t(half), half);
        }
        packed_reconstruction();
        backend::CpuBackend cpu;
        for (int threads : {1, 3}) { cpu.set_threads(threads); operators(cpu); }
        std::cout << "IQ1 format: " << checked << " exact values, all dictionary indices/lanes, finite halves, fields, unaligned blocks and guards passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
