#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "matrix_precision.hpp"
#include "row_classes.hpp"

namespace {
constexpr int values[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
float scale(uint16_t h) {
    const unsigned e = (h >> 10) & 31, m = h & 1023;
    return std::copysign(float(e ? std::ldexp(double(1024 + m), int(e) - 25) : std::ldexp(double(m), -24)), h & 32768 ? -1.0f : 1.0f);
}
void block(uint8_t* p, uint16_t h, unsigned shift) {
    p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    for (unsigned i = 0; i < 16; ++i) p[2 + i] = uint8_t(((i + shift) & 15) | (((i + shift + 7) & 15) << 4));
}
float weight(const uint8_t* p, size_t i) {
    return scale(uint16_t(p[0] | unsigned(p[1]) << 8)) * float(values[(i < 16 ? p[2 + i] : p[2 + i - 16] >> 4) & 15]);
}
std::vector<float> read(backend::Backend& b, backend::Buffer& buffer, size_t n) {
    std::vector<float> out(n); b.read(buffer, 0, out.data(), n * 4); return out;
}
size_t exact_decode(backend::Backend& b) {
    std::vector<uint8_t> packed;
    std::vector<float> expected;
    std::vector<uint32_t> ids;
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        const size_t at = packed.size(); packed.resize(at + 18);
        block(packed.data() + at, uint16_t(h), h & 15);
        for (size_t i = 0; i < 32; ++i) expected.push_back(weight(packed.data() + at, i));
        ids.push_back(uint32_t(ids.size()));
    }
    const auto w = b.adopt(packed.data(), packed.size()), y = b.alloc(expected.size() * 4);
    b.embed({y.get(), 0}, 20, {w.get(), 0}, 32, ids.size(), ids.data(), ids.size());
    auto got = read(b, *y, expected.size());
    for (size_t i = 0; i < got.size(); ++i) require(bits(got[i]) == bits(expected[i]), "IQ4_NL embedding differs from independent finite-scale decode");
    const uint32_t chosen[] = {17, 17, uint32_t(ids.size()), UINT32_MAX};
    const auto ib = b.adopt(chosen, sizeof(chosen)), gathered = b.alloc(128 * 4);
    b.embed_ids({gathered.get(), 0}, 20, {w.get(), 0}, 32, ids.size(), {ib.get(), 0}, 4);
    got = read(b, *gathered, 128);
    for (size_t i = 0; i < got.size(); ++i) require(bits(got[i]) == (i < 64 ? bits(expected[17 * 32 + i % 32]) : 0), "IQ4_NL device-ID embedding or invalid-ID zeros differ");
    return expected.size() + got.size();
}
size_t ranges(backend::Backend& b) {
    size_t checked = 0;
    const backend::RowRun decode{1, 1}, tile{1, 16384};
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x7bff), uint16_t(0xfbff)}) {
      for (unsigned layout : {0u, 1u, 2u}) {
        std::vector<uint8_t> packed(18); block(packed.data(), h, 0);
        if (layout) std::fill(packed.begin() + 2, packed.end(), layout == 1 ? uint8_t(0) : uint8_t(255));
        const auto w = b.adopt(packed.data(), packed.size()), y = b.alloc(4);
        for (int exponent : {-140, -120, -100, 80, 100, 120, 127}) {
            std::vector<float> x(32, layout ? std::ldexp(1.0f, exponent) : 0); x[0] = std::ldexp(1.0f, exponent);
            const auto xb = b.adopt(x.data(), x.size() * 4);
            double want = 0;
            for (size_t k = 0; k < 32; ++k) want += double(weight(packed.data(), k)) * x[k];
            for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
              // These powers of two are exact BF16 values except below its minimum subnormal, which rounds to zero.
              const double expected = dtype == backend::Dtype::bf16 && exponent < -133 ? 0 : want;
              if (std::fabs(expected) > std::numeric_limits<float>::max()) continue;
              for (const auto* run : {&decode, &tile}) {
                b.matmul(20, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, 32, 1, 1, {run, 1}, dtype);
                const float got = read(b, *y, 1)[0];
                require(std::isfinite(got) && std::fabs(got - expected) <= std::max(1e-44, std::fabs(expected) * 2e-7), "IQ4_NL finite-scale range dot differs");
                ++checked;
              }
            }
        }
      }
    }
    return checked;
}

// Equal and opposite blocks must cancel even with signed scales and very large inputs.
// The small-scale case makes dw*dx subnormal, so the native dot must use its F32 repair.
size_t cancellations(backend::Backend& b) {
    const backend::RowRun decode{1, 1}, tile{1, 16384};
    size_t checked = 0;
    for (uint16_t h : {uint16_t(1), uint16_t(0x3c00), uint16_t(0x7bff)}) {
        std::vector<uint8_t> packed(4 * 18);
        for (size_t j = 0; j < 4; ++j) block(packed.data() + j * 18, uint16_t(h | (j & 1 ? 0x8000 : 0)), 0);
        const auto w = b.adopt(packed.data(), packed.size()), y = b.alloc(4);
        for (int exponent : {-100, 0, 100}) {
            std::vector<float> x(128, std::ldexp(1.0f, exponent));
            const auto xb = b.adopt(x.data(), x.size() * 4);
            for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
              for (const auto* run : {&decode, &tile}) {
                b.matmul(20, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, 128, 1, 1, {run, 1}, dtype);
                require(read(b, *y, 1)[0] == 0, "IQ4_NL signed-block cancellation differs");
                ++checked;
              }
            }
        }
    }
    return checked;
}

size_t products(backend::Backend& b, size_t nin, size_t rows, size_t cols, backend::Dtype dtype) {
    constexpr size_t experts = 3, picks = 2;
    const size_t rb = nin / 32 * 18, n = rows * cols;
    std::vector<uint8_t> packed(experts * rows * rb);
    std::vector<float> x(cols * picks * nin), decoded(experts * rows * nin);
    for (size_t j = 0; j < packed.size() / 18; ++j) {
        block(packed.data() + j * 18, uint16_t((j & 1 ? 0x8000 : 0) | (0x1000 + (j * 13 % 1024))), unsigned(j));
        for (size_t i = 0; i < 32; ++i) decoded[j * 32 + i] = weight(packed.data() + j * 18, i);
    }
    // The selected non-BF16 value rounds to 1.0078125, distinguishing rounded BF16 inputs from the wider F32 path.
    for (size_t i = 0; i < x.size(); ++i) x[i] = i % 7 == 0 ? 1.005859375f : float(int(i * 13 % 31) - 15) / 16;
    const auto rounded = testq::row_activations(x);
    const auto profile = backend::vulkan_device_profile(b);
    const auto w = b.adopt(packed.data(), packed.size()), xb = b.adopt(x.data(), x.size() * 4), y = b.alloc((n * picks + 2) * 4);
    const auto clear = [&] { std::vector<float> guard(n * picks + 2, 17); b.write(*y, 0, guard.data(), guard.size() * 4); };
    const backend::RowRun decode[] = {{cols, 1}}, prompt[] = {{cols, std::max<size_t>(2, cols)}};
    std::vector<uint32_t> ids(cols * picks); std::vector<float> gains(cols * picks);
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = uint32_t((i + 1) % experts); gains[i] = i & 1 ? 0.25f : 0.75f; }
    const auto ib = b.adopt(ids.data(), ids.size() * 4), gb = b.adopt(gains.data(), gains.size() * 4);
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, picks, experts};
    size_t checked = 0;
    for (backend::RowRuns runs : {backend::RowRuns{decode, 1}, backend::RowRuns{prompt, 1}}) {
        for (unsigned op = 0; op < 6; ++op) {
            const size_t crossover = op >= 4 ? profile.moe_tile_from_q4 : backend::tile_from_for(profile, false, nin);
            const bool tile = runs.runs->extent >= crossover;
            const bool integer = dtype != backend::Dtype::f32 && dtype != backend::Dtype::bf16 && (tile ? profile.prefer_integer_dot : profile.lookup_integer_dot);
            clear(); testq::take_matrix_paths(b);
            if (op == 0) b.matmul(20, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
            if (op == 1) b.matmul_logits(20, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
            if (op == 2) b.matmul_add(20, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
            if (op == 3) b.matmul_group({{20, {w.get(), 0}, {y.get(), 1}, rows}, {20, {w.get(), 0}, {y.get(), n + 1}, rows}}, {xb.get(), 0}, nin, cols, runs, dtype);
            if (op == 4) b.matmul_experts({{20, {w.get(), 0}, {y.get(), 1}, rows}}, {xb.get(), 0}, nin, cols, routing, runs, dtype);
            if (op == 5) b.matmul_experts_add(20, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, routing, runs, dtype);
            auto got = read(b, *y, n * picks + 2);
            const auto paths = testq::take_matrix_paths(b);
            require(paths == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"}, "IQ4_NL wider matrix-path witness differs");
            const size_t count = op == 4 || op == 3 ? n * picks : n;
            require(got.front() == 17 && got[count + 1] == 17, "IQ4_NL output guards changed");
            for (size_t at = 0; at < count; ++at) {
                const size_t r = at % rows, c = op == 3 ? (at / rows) % cols : at / rows;
                double want = (op == 2 || op == 5) ? 17 : 0, magnitude = std::fabs(want);
                const size_t count_picks = op == 5 ? picks : 1;
                for (size_t pick = 0; pick < count_picks; ++pick) {
                    const size_t entry = op == 5 ? c * picks + pick : c;
                    const size_t expert = op >= 4 ? ids[entry] : 0;
                    const size_t input = op == 4 ? entry / picks : c;
                    const double gain = op == 5 ? gains[entry] : 1;
                    for (size_t k = 0; k < nin; ++k) {
                        const float original = (integer ? rounded : x)[(op == 5 ? entry : input) * nin + k];
                        const double activation = dtype == backend::Dtype::bf16 && original == 1.005859375f ? 1.0078125 : original;
                        const double term = double(decoded[(expert * rows + r) * nin + k]) * activation * gain;
                        want += term; magnitude += std::fabs(term);
                    }
                }
                require(std::isfinite(got[at + 1]) && std::fabs(got[at + 1] - want) <= 3e-6 * std::max(1.0, magnitude), "IQ4_NL product differs from independent double dot");
                ++checked;
            }
        }
    }
    return checked;
}
}
int main() {
    try {
        backend::BackendPtr b;
        try { b = backend::make_vulkan_backend(0, true); }
        catch (const backend::VulkanUnavailable& e) { std::cout << e.what() << '\n'; return 77; }
        require(b->supports_type(20), "Vulkan refuses IQ4_NL");
        size_t checked = exact_decode(*b) + ranges(*b) + cancellations(*b);
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
            for (size_t nin : {32u, 96u, 256u})
                for (size_t cols : {1u, 3u, 65u, 193u}) checked += products(*b, nin, 7, cols, dtype);
        for (auto dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
            checked += products(*b, 4352, 7, 65, dtype);
            checked += products(*b, 96, 129, 193, dtype);
        }
        for (size_t rows : {2049u, 8193u}) checked += products(*b, 32, rows, 65, backend::Dtype::f16);
        const size_t pairs = row_classes::check(*b, {20}, [](uint32_t, size_t nin, size_t rows, uint32_t seed) {
            std::vector<uint8_t> packed(rows * nin / 32 * 18);
            for (size_t j = 0; j < packed.size() / 18; ++j) block(packed.data() + j * 18, uint16_t((j & 1 ? 0x8000 : 0) | 0x1000 | (j * 13 % 1024)), unsigned(j + seed));
            return packed;
        });
        std::cout << "iq4-nl-vulkan: " << pairs << " row-class bitwise pairs pass\n";
        const auto kernels = backend::vulkan_kernel_times(*b);
        bool row = false, tile = false, rounded = false, integer = false, integer_tile = false;
        for (const auto& kernel : kernels) {
            integer_tile = integer_tile || kernel.first.find("matmul_tile_iq4") != std::string::npos;
            integer = integer || kernel.first.find("matmul_row_iq4_dot") != std::string::npos;
            row = row || kernel.first.find("matmul_row_float_x") != std::string::npos;
            tile = tile || kernel.first.find("matmul_tile") != std::string::npos;
            rounded = rounded || kernel.first.find("matmul_tile_bf16") != std::string::npos;
            std::cout << "iq4-nl-vulkan dispatch: " << kernel.first << '\n';
        }
        if (!kernels.empty()) require(row && tile && rounded, "IQ4_NL row, tile or BF16 dispatch was not witnessed");
        if (!kernels.empty()) {
            const auto profile = backend::vulkan_device_profile(*b);
            if (profile.lookup_integer_dot) require(integer, "IQ4_NL native lookup rows were not witnessed");
            require(integer_tile == profile.prefer_integer_dot, "IQ4_NL tile dispatch ignored its integer-tile preference");
        }
        if (kernels.empty()) std::cout << "iq4-nl-vulkan: dispatch names unavailable without timestamps\n";
        std::cout << "iq4-nl-vulkan: " << checked << " independent decoded/product values and wider-path witnesses pass\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
