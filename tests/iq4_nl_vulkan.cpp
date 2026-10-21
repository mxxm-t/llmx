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
void xs_scale(uint8_t* p, unsigned group, unsigned value) {
    const unsigned high_shift = 2 * (group % 4), low_shift = 4 * (group % 2);
    p[2 + group / 4] = uint8_t((p[2 + group / 4] & ~(3u << high_shift)) | ((value / 16) << high_shift));
    p[4 + group / 2] = uint8_t((p[4 + group / 2] & ~(15u << low_shift)) | ((value % 16) << low_shift));
}
void test_block(uint32_t type, uint8_t* p, uint16_t h, unsigned shift) {
    if (type == 20) { block(p, h, shift); return; }
    p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    std::fill(p + 2, p + 8, uint8_t(0));
    for (unsigned group = 0; group < 8; ++group) {
        xs_scale(p, group, (shift + 9 * group) % 64);
        for (unsigned i = 0; i < 16; ++i)
            p[8 + 16 * group + i] = uint8_t(((i + shift + group) % 16) | (((i + shift + group + 7) % 16) << 4));
    }
}
float test_weight(uint32_t type, const uint8_t* p, size_t i) {
    if (type == 20) return weight(p, i);
    const size_t group = i / 32;
    const unsigned high = unsigned(p[2]) + 256 * unsigned(p[3]);
    const unsigned subscale = ((p[4 + group / 2] >> (4 * (group % 2))) & 15) + 16 * ((high >> (2 * group)) & 3);
    const unsigned code = (p[8 + 16 * group + i % 16] >> (i % 32 < 16 ? 0 : 4)) & 15;
    return float(double(scale(uint16_t(p[0] | unsigned(p[1]) << 8))) * (int(subscale) - 32) * values[code]);
}
std::vector<float> read(backend::Backend& b, backend::Buffer& buffer, size_t n) {
    std::vector<float> out(n); b.read(buffer, 0, out.data(), n * 4); return out;
}
size_t exact_decode(backend::Backend& b, uint32_t type = 20) {
    const size_t bs = type == 20 ? 32 : 256, bytes = type == 20 ? 18 : 136;
    std::vector<uint8_t> packed;
    std::vector<float> expected;
    std::vector<uint32_t> ids;
    const auto append = [&](uint16_t h, unsigned shift) {
        const size_t at = packed.size(); packed.resize(at + bytes);
        test_block(type, packed.data() + at, h, shift);
        for (size_t i = 0; i < bs; ++i) expected.push_back(test_weight(type, packed.data() + at, i));
        ids.push_back(uint32_t(ids.size()));
    };
    for (unsigned h = 0; h < 65536; ++h)
        if ((h & 0x7c00) != 0x7c00) append(uint16_t(h), h & 63);
    if (type == 23)
        for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x03ff),
                           uint16_t(0x0400), uint16_t(0x3c00), uint16_t(0xbc00), uint16_t(0x7bff), uint16_t(0xfbff)})
            for (unsigned shift = 0; shift < 64; ++shift) append(h, shift);
    const auto w = b.adopt(packed.data(), packed.size()), y = b.alloc(expected.size() * 4);
    b.embed({y.get(), 0}, type, {w.get(), 0}, bs, ids.size(), ids.data(), ids.size());
    auto got = read(b, *y, expected.size());
    for (size_t i = 0; i < got.size(); ++i) require(bits(got[i]) == bits(expected[i]), "IQ4_NL/IQ4_XS embedding differs from independent finite-scale decode");
    const uint32_t chosen[] = {17, 17, uint32_t(ids.size()), UINT32_MAX};
    const auto ib = b.adopt(chosen, sizeof(chosen)), gathered = b.alloc(4 * bs * 4);
    b.embed_ids({gathered.get(), 0}, type, {w.get(), 0}, bs, ids.size(), {ib.get(), 0}, 4);
    got = read(b, *gathered, 4 * bs);
    for (size_t i = 0; i < got.size(); ++i) require(bits(got[i]) == (i < 2 * bs ? bits(expected[17 * bs + i % bs]) : 0), "IQ4_NL/IQ4_XS device-ID embedding or invalid-ID zeros differ");
    return expected.size() + got.size();
}
size_t ranges(backend::Backend& b, uint32_t type = 20) {
    const size_t bs = type == 20 ? 32 : 256, bytes = type == 20 ? 18 : 136;
    size_t checked = 0;
    const backend::RowRun decode{1, 1}, tile{1, 16384};
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x7bff), uint16_t(0xfbff)}) {
      for (unsigned layout : {0u, 1u, 2u}) {
        std::vector<uint8_t> packed(bytes); test_block(type, packed.data(), h, 0);
        if (layout) std::fill(packed.begin() + (type == 20 ? 2 : 8), packed.end(), layout == 1 ? uint8_t(0) : uint8_t(255));
        const auto w = b.adopt(packed.data(), packed.size()), y = b.alloc(4);
        for (int exponent : {-140, -120, -100, 80, 100, 120, 127}) {
            std::vector<float> x(bs, layout ? std::ldexp(1.0f, exponent) : 0); x[0] = std::ldexp(1.0f, exponent);
            const auto xb = b.adopt(x.data(), x.size() * 4);
            double want = 0, magnitude = 0;
            std::vector<float> weights(bs);
            for (size_t k = 0; k < bs; ++k) {
                weights[k] = test_weight(type, packed.data(), k);
                const double term = double(weights[k]) * x[k];
                want += term; magnitude += std::fabs(term);
            }
            for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
              // These powers of two are exact BF16 values except below its minimum subnormal, which rounds to zero.
              const double expected = dtype == backend::Dtype::bf16 && exponent < -133 ? 0 : want;
              if (std::fabs(expected) > std::numeric_limits<float>::max()) continue;
              for (const auto* run : {&decode, &tile}) {
                b.matmul(type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, bs, 1, 1, {run, 1}, dtype);
                const float got = read(b, *y, 1)[0];
                // XS sums 256 signed terms; gamma_n bounds FMA rounding against their absolute sum, including cancellation.
                const double nu = double(bs) * 0x1p-24;
                const double bound = type == 23
                    ? nu / (1 - nu) * magnitude + double(bs) * std::numeric_limits<float>::denorm_min()
                    : std::max(1e-44, std::fabs(expected) * 2e-7);
                const bool ok = std::isfinite(got) && std::fabs(got - expected) <= bound;
                if (!ok) std::cerr << "range: type=" << type << " half=" << h << " layout=" << layout
                    << " exponent=" << exponent << " dtype=" << int(dtype) << " tile=" << (run == &tile)
                    << " got=" << got << " expected=" << expected << '\n';
                if (type == 23 && dtype == backend::Dtype::f32 && run == &tile) {
                    float ordered = 0;
                    for (size_t k = 0; k < bs; ++k) ordered = std::fma(weights[k], x[k], ordered);
                    require(bits(got) == bits(ordered), "IQ4_XS range tile differs from independent ordered FMA");
                }
                require(ok, "IQ4_NL/IQ4_XS finite-scale range dot differs");
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
size_t cancellations(backend::Backend& b, uint32_t type = 20) {
    const size_t bs = type == 20 ? 32 : 256, bytes = type == 20 ? 18 : 136;
    const backend::RowRun decode{1, 1}, tile{1, 16384};
    size_t checked = 0;
    for (uint16_t h : {uint16_t(1), uint16_t(0x3c00), uint16_t(0x7bff)}) {
        std::vector<uint8_t> packed(4 * bytes);
        for (size_t j = 0; j < 4; ++j) {
            uint8_t* p = packed.data() + j * bytes;
            test_block(type, p, uint16_t(h | (j & 1 ? 0x8000 : 0)), 0);
            if (type == 23)
                for (unsigned group = 0; group < 8; ++group) xs_scale(p, group, group == 0 ? 33 : 32);
        }
        const auto w = b.adopt(packed.data(), packed.size()), y = b.alloc(4);
        for (int exponent : {-100, 0, 100}) {
            std::vector<float> x(4 * bs, std::ldexp(1.0f, exponent));
            const auto xb = b.adopt(x.data(), x.size() * 4);
            for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
              for (const auto* run : {&decode, &tile}) {
                b.matmul(type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, 4 * bs, 1, 1, {run, 1}, dtype);
                const float got = read(b, *y, 1)[0];
                if (got != 0) std::cerr << "cancel: type=" << type << " half=" << h << " exponent=" << exponent
                    << " dtype=" << int(dtype) << " tile=" << (run == &tile) << " got=" << got << '\n';
                require(got == 0, "IQ4_NL/IQ4_XS signed-block cancellation differs");
                ++checked;
              }
            }
        }
    }
    return checked;
}

size_t products(backend::Backend& b, size_t nin, size_t rows, size_t cols, backend::Dtype dtype, uint32_t type = 20) {
    constexpr size_t experts = 3, picks = 2;
    const size_t bs = type == 20 ? 32 : 256, bytes = type == 20 ? 18 : 136;
    const size_t rb = nin / bs * bytes, n = rows * cols;
    std::vector<uint8_t> packed(experts * rows * rb);
    std::vector<float> x(cols * picks * nin), decoded(experts * rows * nin);
    for (size_t j = 0; j < packed.size() / bytes; ++j) {
        test_block(type, packed.data() + j * bytes, uint16_t((j & 1 ? 0x8000 : 0) | (0x1000 + (j * 13 % 1024))), unsigned(j));
        for (size_t i = 0; i < bs; ++i) decoded[j * bs + i] = test_weight(type, packed.data() + j * bytes, i);
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
            if (op == 0) b.matmul(type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
            if (op == 1) b.matmul_logits(type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
            if (op == 2) b.matmul_add(type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
            if (op == 3) b.matmul_group({{type, {w.get(), 0}, {y.get(), 1}, rows}, {type, {w.get(), 0}, {y.get(), n + 1}, rows}}, {xb.get(), 0}, nin, cols, runs, dtype);
            if (op == 4) b.matmul_experts({{type, {w.get(), 0}, {y.get(), 1}, rows}}, {xb.get(), 0}, nin, cols, routing, runs, dtype);
            if (op == 5) b.matmul_experts_add(type, {w.get(), 0}, {xb.get(), 0}, {y.get(), 1}, nin, rows, cols, routing, runs, dtype);
            auto got = read(b, *y, n * picks + 2);
            const auto paths = testq::take_matrix_paths(b);
            require(paths == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"}, "IQ4_NL/IQ4_XS wider matrix-path witness differs");
            const size_t count = op == 4 || op == 3 ? n * picks : n;
            require(got.front() == 17 && got[count + 1] == 17, "IQ4_NL/IQ4_XS output guards changed");
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
                require(std::isfinite(got[at + 1]) && std::fabs(got[at + 1] - want) <= 3e-6 * std::max(1.0, magnitude), "IQ4_NL/IQ4_XS product differs from independent double dot");
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
        require(b->supports_type(23), "Vulkan refuses IQ4_XS");
        checked += exact_decode(*b, 23) + ranges(*b, 23) + cancellations(*b, 23);
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
            for (size_t nin : {256u, 768u, 4352u})
                for (size_t cols : {1u, 3u, 65u, 193u}) checked += products(*b, nin, 7, cols, dtype, 23);
        for (auto dtype : {backend::Dtype::f16, backend::Dtype::int8})
            checked += products(*b, 768, 129, 193, dtype, 23);
        for (size_t rows : {2049u, 8193u}) checked += products(*b, 256, rows, 65, backend::Dtype::f16, 23);
        const size_t pairs = row_classes::check(*b, {20, 23}, [](uint32_t type, size_t nin, size_t rows, uint32_t seed) {
            const size_t bs = type == 20 ? 32 : 256, bytes = type == 20 ? 18 : 136;
            std::vector<uint8_t> packed(rows * nin / bs * bytes);
            for (size_t j = 0; j < packed.size() / bytes; ++j)
                test_block(type, packed.data() + j * bytes, uint16_t((j & 1 ? 0x8000 : 0) | 0x1000 | (j * 13 % 1024)), unsigned(j + seed));
            return packed;
        });
        std::cout << "iq4-nl-vulkan: " << pairs << " row-class bitwise pairs pass\n";
        const auto kernels = backend::vulkan_kernel_times(*b);
        bool row = false, tile = false, rounded = false, integer = false, integer_tile = false, xs_row = false, xs_tile = false;
        for (const auto& kernel : kernels) {
            integer_tile = integer_tile || kernel.first.find("matmul_tile_iq4") != std::string::npos;
            xs_row = xs_row || kernel.first.find("matmul_row_iq4_xs_dot") != std::string::npos;
            xs_tile = xs_tile || kernel.first.find("matmul_tile_iq4_xs") != std::string::npos;
            integer = integer || kernel.first.find("matmul_row_iq4_dot") != std::string::npos;
            row = row || kernel.first.find("matmul_row_float_x") != std::string::npos;
            tile = tile || kernel.first.find("matmul_tile") != std::string::npos;
            rounded = rounded || kernel.first.find("matmul_tile_bf16") != std::string::npos;
            std::cout << "iq4-nl-vulkan dispatch: " << kernel.first << '\n';
        }
        if (!kernels.empty()) require(row && tile && rounded, "IQ4_NL/IQ4_XS row, tile or BF16 dispatch was not witnessed");
        if (!kernels.empty()) {
            const auto profile = backend::vulkan_device_profile(*b);
            if (profile.lookup_integer_dot) require(integer && xs_row, "IQ4_NL/IQ4_XS native lookup rows were not witnessed");
            require(integer_tile == profile.prefer_integer_dot && xs_tile == profile.prefer_integer_dot, "IQ4_NL/IQ4_XS tile dispatch ignored its integer-tile preference");
        }
        if (kernels.empty()) std::cout << "iq4-nl-vulkan: dispatch names unavailable without timestamps\n";
        std::cout << "iq4-nl-vulkan: " << checked << " independent decoded/product values and wider-path witnesses pass\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
