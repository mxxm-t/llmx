#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif
#include "matrix_precision.hpp"
#include "quant/quant.hpp"

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

// Decoded little-endian byte vectors from ggml-common.h at
// 750127a5173ac6c29c2c159a10404bf184c421ab (same pinned table as spec_decode.py).
constexpr char grid[] =
    "0404040414040404240404040c0c04041c0c04043e0c040404140404141404040c1c0404142404041c3e04042c3e04040c040c041c040c04040c0c04140c0c04"
    "0c140c042c140c04041c0c04141c0c040c240c04242c0c04043e0c040404140414041404240414040c0c140404141404141414040c1c14041c1c14043e1c1404"
    "0c2c14043e2c14042c3e14040c041c043e041c04040c1c04140c1c042c141c04043e1c041c0c24043e1c2404242424043e2c24041c3e24042c3e24040c042c04"
    "3e042c04141c2c04142c2c042c1c340424343404040c3e04240c3e04340c3e041c243e040c343e040c04040c1c04040c040c040c140c040c0c14040c1c14040c"
    "041c040c141c040c241c040c3e24040c042c040c04040c0c14040c0c0c0c0c0c04140c0c14140c0c0c04140c1c04140c040c140c140c140c0c14140c041c140c"
    "143e140c04041c0c14041c0c04141c0c0c1c1c0c34241c0c34341c0c0c04240c2c04240c042c240c04142c0c24142c0c34242c0c0c3e2c0c2c04340c14143e0c"
    "04243e0c04040414140404140c0c04141c0c04140414041414140414341404140c1c0414142404140c040c141c040c142c040c14040c0c14140c0c140c140c14"
    "041c0c141c340c143e340c14043e0c1404041414140414140c0c14143e0c141404141414141414143e1c1414042414142c2c14140c041c14040c1c14240c1c14"
    "043e1c14243e1c142c1c24141c2c24141c042c143e142c140c242c14243e2c140c043e141c043e14340c3e142c243e140c04041c040c041c140c041c0c14041c"
    "1c14041c042c041c2c34041c143e041c04040c1c14040c1c04140c1c0c1c0c1c24240c1c34240c1c0c04141c1c04141c040c141c2c14141c142c141c143e141c"
    "0c0c1c1c1c1c1c1c041c241c3e24241c143e241c04042c1c34042c1c14142c1c2c2c2c1c240c341c341c341c1c34341c1c1c3e1c04343e1c240404243e0c0424"
    "2c1c04243e1c04241c2c04243e2c0424243e0c24041414243e1c14240424142404341424343414243e041c242c241c24240424240c2c2424243424242c142c24"
    "1c242c24043e2c242c043e24040c3e24140c3e24041c3e24140c042c0c24042c043e042c04040c2c34040c2c34140c2c2c2c0c2c240c142c141c142c143e142c"
    "14041c2c1c2c1c2c040c242c1c14242c3e14242c143e242c14042c2c0c1c2c2c042c342c24143e2c14243e2c241404342424043434240434243404340c140c34"
    "0c340c343e0c143424341434041c1c34341c1c34242424342c042c34142c2c341c1c34341c043e340c143e341c04043e2c04043e3e04043e040c043e141c043e"
    "142c043e34140c3e04240c3e140c143e2c24143e142c143e04041c3e2c0c1c3e1c1c1c3e04341c3e0c14243e0c24243e04042c3e14042c3e24142c3e041c343e";

unsigned digit(char c) { return c <= '9' ? unsigned(c - '0') : unsigned(c - 'a' + 10); }
unsigned magnitude(unsigned index, unsigned lane) {
    const size_t at = 2 * (4 * index + lane);
    return 16 * digit(grid[at]) + digit(grid[at + 1]);
}
unsigned sign_byte(unsigned code) {
    unsigned parity = 0;
    for (unsigned bit = 0; bit < 7; ++bit) parity += (code >> bit) & 1u;
    return code | ((parity & 1u) << 7);
}
double half(uint16_t h) {
    const unsigned e = (h >> 10) & 31, f = h & 1023;
    const double v = e ? std::ldexp(double(1024 + f), int(e) - 25) : std::ldexp(double(f), -24);
    return h & 0x8000 ? -v : v;
}
float expected(uint16_t h, unsigned index, unsigned lane, unsigned scale, unsigned code) {
    const double v = half(h) * (1 + 2 * scale) * magnitude(index, lane % 4) * 0.25;
    return float((sign_byte(code) >> lane) & 1u ? -v : v);
}

using Indices = std::array<unsigned, 64>;
using Scales = std::array<unsigned, 8>;
using Codes = std::array<unsigned, 32>;
size_t checked = 0;
void pack(uint8_t* p, uint16_t h, const Indices& ids, const Scales& scales, const Codes& codes) {
    std::memset(p, 0, 98);
    p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    for (size_t i = 0; i < ids.size(); ++i) p[2 + i] = uint8_t(ids[i]);
    for (unsigned g = 0; g < 8; ++g) {
        uint32_t word = scales[g] << 28;
        for (unsigned sub = 0; sub < 4; ++sub) word |= codes[4 * g + sub] << (7 * sub);
        for (unsigned b = 0; b < 4; ++b) p[66 + 4 * g + b] = uint8_t(word >> (8 * b));
    }
}
void check_decode(uint16_t h, const Indices& ids, const Scales& scales, const Codes& codes) {
    std::array<uint8_t, 296> src{};
    std::array<float, 770> dst{};
    src.fill(0xa5); dst.fill(12345.0f);
    for (unsigned b = 0; b < 3; ++b) {
        auto local_ids = ids;
        for (auto& i : local_ids) i = (i + 37 * b) % 256;
        pack(src.data() + 1 + b * 98, uint16_t(h ^ (b == 1 ? 0x8000 : 0)), local_ids, scales, codes);
    }
    const auto before = src;
    quant::dequantize_row_iq3_xxs(src.data() + 1, dst.data() + 1, 3);
    require(src == before, "IQ3_XXS decoder changed packed bytes");
    require(dst.front() == 12345.0f && dst.back() == 12345.0f, "IQ3_XXS decoder crossed guards");
    for (unsigned b = 0; b < 3; ++b)
        for (unsigned g = 0; g < 8; ++g)
            for (unsigned sub = 0; sub < 4; ++sub)
                for (unsigned lane = 0; lane < 8; ++lane) {
                    const unsigned vector = 8 * g + 2 * sub + lane / 4;
                    const unsigned index = (ids[vector] + 37 * b) % 256;
                    const float want = expected(uint16_t(h ^ (b == 1 ? 0x8000 : 0)), index,
                                                lane, scales[g], codes[4 * g + sub]);
                    const size_t out = 1 + b * 256 + 32 * g + 8 * sub + lane;
                    if (bits(dst[out]) != bits(want)) {
                        std::cerr << "IQ3_XXS decode: half=" << h << " block=" << b
                                  << " group=" << g << " sub=" << sub << " lane=" << lane
                                  << " got_bits=" << bits(dst[out]) << " expected_bits=" << bits(want) << '\n';
                        throw std::runtime_error("IQ3_XXS field decode differs from independent oracle");
                    }
                    ++checked;
                }
}
std::vector<float> read(backend::Backend& b, backend::Buffer& buffer, size_t count) {
    std::vector<float> out(count); b.read(buffer, 0, out.data(), count * sizeof(float)); return out;
}
std::vector<std::string> paths(backend::Backend& b) {
    backend::MatrixPaths p; b.swap_matrix_paths(p); return p.take();
}
bool integer_path(backend::Backend& b, backend::Dtype dtype, size_t extent) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(b);
        return extent == 1 ? profile.lookup_integer_dot : profile.prefer_integer_dot;
    }
#else
    (void)b; (void)dtype; (void)extent;
#endif
    return false;
}

struct Fixture { std::vector<uint8_t> packed; std::vector<float> values; };
Fixture fixture(size_t rows, size_t width) {
    Fixture f{std::vector<uint8_t>(rows * width / 256 * 98), std::vector<float>(rows * width)};
    for (size_t r = 0; r < rows; ++r)
        for (size_t block = 0; block < width / 256; ++block) {
            Indices ids{}; Scales scales{}; Codes codes{};
            for (unsigned i = 0; i < 64; ++i) ids[i] = unsigned((r * 43 + block * 71 + i * 13) % 256);
            for (unsigned i = 0; i < 8; ++i) scales[i] = unsigned((r + block + i * 3) % 16);
            for (unsigned i = 0; i < 32; ++i) codes[i] = unsigned((r * 19 + block * 11 + i * 7) % 128);
            const uint16_t h = uint16_t((r + block) % 2 ? 0xb555 : 0x3555);
            auto* p = f.packed.data() + (r * (width / 256) + block) * 98;
            pack(p, h, ids, scales, codes);
            for (unsigned g = 0; g < 8; ++g)
                for (unsigned sub = 0; sub < 4; ++sub)
                    for (unsigned lane = 0; lane < 8; ++lane) {
                        const unsigned vector = 8 * g + 2 * sub + lane / 4;
                        const size_t i = r * width + block * 256 + 32 * g + 8 * sub + lane;
                        f.values[i] = expected(h, ids[vector], lane, scales[g], codes[4 * g + sub]);
                    }
        }
    return f;
}
double dot(const float* w, const float* x, size_t n) {
    double value = 0; for (size_t i = 0; i < n; ++i) value += double(w[i]) * x[i]; return value;
}
double dot_magnitude(const float* w, const float* x, size_t n) {
    double value = 0; for (size_t i = 0; i < n; ++i) value += std::fabs(double(w[i]) * x[i]); return value;
}
void check_close(float got, double want, double mag, size_t n) {
    const double nu = double(2 * n + 8) * 0x1p-24;
    require(std::isfinite(got) && std::fabs(double(got) - want) <= nu / (1 - nu) * mag + n * std::numeric_limits<float>::denorm_min(),
            "IQ3_XXS backend product exceeds independent rounding bound");
}

size_t range_repair_products(backend::Backend& b) {
    constexpr size_t width = 256;
    const uint16_t halves[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    const float inputs[] = {0.0f, 0x1p-145f, 0x1p-120f, 0x1p-24f, 0x1p-14f, 1.0f, 65504.0f, 0x1p90f};
    unsigned max_index = 0;
    for (unsigned i = 1; i < 256; ++i)
        if (magnitude(i, 0) > magnitude(max_index, 0)) max_index = i;
    Indices ids{}; ids[0] = max_index;
    Scales scales; scales.fill(15);
    Codes codes{};
    size_t products = 0;
    for (uint16_t h : halves) {
        std::vector<uint8_t> packed(98);
        pack(packed.data(), h, ids, scales, codes);
        const double weight = expected(h, max_index, 0, 15, 0);
        auto w = b.adopt(packed.data(), packed.size());
        for (float value : inputs) for (float sign : {-1.0f, 1.0f}) {
            std::vector<float> input(width, 0.0f);
            input[0] = sign * value;
            auto x = b.adopt(input.data(), input.size() * sizeof(float));
            float initial[] = {19.0f, 0.0f, 19.0f};
            auto out = b.adopt(initial, sizeof initial);
            for (size_t extent : {size_t(1), size_t(16384)})
                for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8}) {
                    const bool integer = integer_path(b, dtype, extent);
                    const backend::RowRun run{1, extent};
                    paths(b);
                    b.matmul({quant::GGML_TYPE_IQ3_XXS, {w.get(), 0}}, {x.get(), 0}, {out.get(), 1},
                             width, 1, 1, {&run, 1}, dtype);
                    const auto got = read(b, *out, 3);
                    require(paths(b) == std::vector<std::string>{integer ? "block-int16" : "f32"},
                            "IQ3_XXS range arithmetic path witness");
                    const double want = weight * input[0];
                    // Covers activation reconstruction, product rounding and subnormal scale repair.
                    const double repair = integer ? std::fabs(double(value)) * 0x1p-22 +
                        32767.0 * std::numeric_limits<float>::denorm_min() : 0.0;
                    const double bound = std::fabs(weight) * repair + std::fabs(want) * 0x1p-21 +
                        2.0 * std::numeric_limits<float>::denorm_min();
                    require(got.front() == 19.0f && got.back() == 19.0f && std::isfinite(got[1]) &&
                            std::fabs(double(got[1]) - want) <= bound,
                            "IQ3_XXS finite-half or activation-scale repair range control");
                    ++products;
                }
            require(read(b, *x, input.size()) == input, "IQ3_XXS range matmul changed its input");
        }
    }
    return products;
}

void backend_checks(backend::Backend& b) {
    constexpr size_t width = 512, rows = 3, cols = 2, count = rows * cols;
    size_t typed_cases = 0;
    std::set<std::string> witnessed_paths;
    require(b.supports_type(quant::GGML_TYPE_IQ3_XXS), "backend does not expose IQ3_XXS");
    const auto emb = fixture(2, 256);
    auto ew = b.adopt(emb.packed.data(), emb.packed.size());
    auto eo = b.alloc((2 * 256 + 2) * sizeof(float));
    std::vector<float> guards(2 * 256 + 2, 99.0f);
    b.write(*eo, 0, guards.data(), guards.size() * sizeof(float));
    const uint32_t ids[] = {1, 0};
    b.embed({eo.get(), 1}, {quant::GGML_TYPE_IQ3_XXS, {ew.get(), 0}}, 256, 2, ids, 2);
    auto got = read(b, *eo, guards.size());
    require(got.front() == 99.0f && got.back() == 99.0f, "IQ3_XXS embedding guards");
    for (size_t r = 0; r < 2; ++r) for (size_t i = 0; i < 256; ++i)
        require(bits(got[1 + r * 256 + i]) == bits(emb.values[ids[r] * 256 + i]), "IQ3_XXS embedding differs from independent fields");

    uint32_t selected[] = {1, 0, 2, UINT32_MAX};
    auto selected_buffer = b.adopt(selected, sizeof selected);
    auto indexed = b.alloc((4 * 256 + 2) * sizeof(float));
    std::vector<float> indexed_guard(4 * 256 + 2, 99.0f);
    b.write(*indexed, 0, indexed_guard.data(), indexed_guard.size() * sizeof(float));
    b.embed_ids({indexed.get(), 1}, {quant::GGML_TYPE_IQ3_XXS, {ew.get(), 0}}, 256, 2,
                {selected_buffer.get(), 0}, 4);
    got = read(b, *indexed, indexed_guard.size());
    require(got.front() == 99.0f && got.back() == 99.0f, "IQ3_XXS embed_ids guards");
    for (size_t row = 0; row < 4; ++row) for (size_t i = 0; i < 256; ++i) {
        const float want = row < 2 ? emb.values[selected[row] * 256 + i] : 0.0f;
        require(bits(got[1 + row * 256 + i]) == bits(want), "IQ3_XXS embed_ids valid/invalid rows");
    }

    const auto f = fixture(rows, width);
    auto w = b.adopt(f.packed.data(), f.packed.size());
    std::vector<float> input(cols * width), rounded, reconstructed;
    for (size_t c = 0; c < cols; ++c) for (size_t i = 0; i < width; ++i)
        input[c * width + i] = float(int((c * 31 + i * 7) % 59) - 29) / 37.0f;
    rounded = input; for (auto& v : rounded) { uint32_t u; std::memcpy(&u, &v, 4); uint32_t top = u >> 16, low = u & 65535; if (low > 32768 || (low == 32768 && (top & 1))) ++top; u = top << 16; std::memcpy(&v, &u, 4); }
    reconstructed = testq::row_activations(input);
    auto x = b.adopt(input.data(), input.size() * sizeof(float));
    auto out = b.alloc((count + 2) * sizeof(float));
    for (size_t extent : {size_t(1), size_t(16384)})
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            const bool integer = integer_path(b, dtype, extent);
            const auto& oracle = dtype == backend::Dtype::bf16 ? rounded : integer ? reconstructed : input;
            const float initial[] = {77.0f, -0.25f, 77.0f, 77.0f, 77.0f, 77.0f, 77.0f, 77.0f};
            b.write(*out, 0, initial, sizeof initial); paths(b);
            const backend::RowRun run{cols, extent};
            b.matmul({quant::GGML_TYPE_IQ3_XXS, {w.get(), 0}}, {x.get(), 0}, {out.get(), 1}, width, rows, cols, {&run, 1}, dtype);
            got = read(b, *out, count + 2);
            const auto actual_paths = paths(b);
            require(actual_paths == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
                    "IQ3_XXS requested-dtype matrix path witness");
            witnessed_paths.insert(actual_paths.begin(), actual_paths.end());
            ++typed_cases;
            require(got.front() == 77.0f && got.back() == 77.0f, "IQ3_XXS matrix output guards");
            for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
                const double want = dot(f.values.data() + r * width, oracle.data() + c * width, width);
                double mag = 0; for (size_t i = 0; i < width; ++i) mag += std::fabs(double(f.values[r * width + i]) * oracle[c * width + i]);
                check_close(got[1 + c * rows + r], want, mag, width);
            }
        }

    // One-past-threshold rows leave row, column and split-tile tails in the tile dispatch.
    constexpr size_t tile_rows = 65;
    size_t tile_cols = 65;
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu()) {
        const auto profile = backend::vulkan_device_profile(b);
        tile_cols = std::max(tile_cols, backend::tile_from_for(profile, false, width) + 1);
    }
#endif
    const size_t tile_count = tile_rows * tile_cols;
    const auto tile_weights = fixture(tile_rows, width);
    auto tw = b.adopt(tile_weights.packed.data(), tile_weights.packed.size());
    std::vector<float> tile_input(tile_cols * width);
    for (size_t c = 0; c < tile_cols; ++c) for (size_t i = 0; i < width; ++i)
        tile_input[c * width + i] = float(int((c * 17 + i * 11) % 71) - 35) / 37.0f;
    auto tx = b.adopt(tile_input.data(), tile_input.size() * sizeof(float));
    auto tile_out = b.alloc((tile_count + 2) * sizeof(float));
    const backend::RowRun tile_run{tile_cols, tile_cols};
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu()) {
        const auto profile = backend::vulkan_device_profile(b);
        require(tile_cols >= backend::tile_from_for(profile, false, width),
                "IQ3_XXS tile-size test shape is below the selected tile threshold");
    }
#endif
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
        const bool integer = integer_path(b, dtype, tile_run.extent);
        // Reconstruct the tile-specific activation oracle from its unrounded values.
        std::vector<float> tile_oracle = tile_input;
        if (dtype == backend::Dtype::bf16) {
            for (auto& v : tile_oracle) {
                uint32_t u; std::memcpy(&u, &v, 4);
                uint32_t top = u >> 16, low = u & 65535;
                if (low > 32768 || (low == 32768 && (top & 1))) ++top;
                u = top << 16; std::memcpy(&v, &u, 4);
            }
        } else if (integer) tile_oracle = testq::row_activations(tile_input);
        std::vector<float> tile_guard(tile_count + 2, 77.0f);
        b.write(*tile_out, 0, tile_guard.data(), tile_guard.size() * sizeof(float));
        paths(b);
        b.matmul({quant::GGML_TYPE_IQ3_XXS, {tw.get(), 0}}, {tx.get(), 0}, {tile_out.get(), 1},
                 width, tile_rows, tile_cols, {&tile_run, 1}, dtype);
        got = read(b, *tile_out, tile_count + 2);
        const auto actual_paths = paths(b);
        require(actual_paths == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
                "IQ3_XXS tile-size requested-dtype path witness");
        witnessed_paths.insert(actual_paths.begin(), actual_paths.end());
        ++typed_cases;
        require(got.front() == 77.0f && got.back() == 77.0f, "IQ3_XXS tile-size output guards");
        for (size_t c = 0; c < tile_cols; ++c) for (size_t r = 0; r < tile_rows; ++r) {
            const float* wr = tile_weights.values.data() + r * width;
            const float* xr = tile_oracle.data() + c * width;
            check_close(got[1 + c * tile_rows + r], dot(wr, xr, width), dot_magnitude(wr, xr, width), width);
        }
    }

    // The grouped call and residual store share the same independent F32 oracle.
    const backend::RowRun run{cols, 16384};
    auto grouped = b.alloc((2 * count + 2) * sizeof(float));
    std::vector<float> initial_group(2 * count + 2, 0); initial_group.front() = initial_group.back() = 91;
    b.write(*grouped, 0, initial_group.data(), initial_group.size() * sizeof(float));
    b.matmul_group({{{quant::GGML_TYPE_IQ3_XXS, {w.get(), 0}}, {grouped.get(), 1}, rows},
                    {{quant::GGML_TYPE_IQ3_XXS, {w.get(), 0}}, {grouped.get(), count + 1}, rows}},
                   {x.get(), 0}, width, cols, {&run, 1}, backend::Dtype::f32);
    got = read(b, *grouped, 2 * count + 2);
    require(got.front() == 91 && got.back() == 91, "IQ3_XXS grouped guards");
    for (size_t i = 0; i < count; ++i) {
        const double want = dot(f.values.data() + (i % rows) * width, input.data() + (i / rows) * width, width);
        const double mag = dot_magnitude(f.values.data() + (i % rows) * width, input.data() + (i / rows) * width, width);
        check_close(got[1 + i], want, mag, width);
        check_close(got[count + 1 + i], want, mag, width);
    }
    auto residual = b.alloc((count + 2) * sizeof(float));
    std::vector<float> prior(count + 2, 0.25f); prior.front() = prior.back() = 91;
    b.write(*residual, 0, prior.data(), prior.size() * sizeof(float));
    b.matmul_add({quant::GGML_TYPE_IQ3_XXS, {w.get(), 0}}, {x.get(), 0}, {residual.get(), 1}, width, rows, cols, {&run, 1}, backend::Dtype::f32);
    got = read(b, *residual, count + 2);
    require(got.front() == 91 && got.back() == 91, "IQ3_XXS residual guards");
    for (size_t i = 0; i < count; ++i) {
        const double want = dot(f.values.data() + (i % rows) * width, input.data() + (i / rows) * width, width) + 0.25;
        check_close(got[1 + i], want, dot_magnitude(f.values.data() + (i % rows) * width, input.data() + (i / rows) * width, width) + 0.25, width);
    }

    // Two expert matrices exercise routed output and its residual accumulation.
    auto experts = fixture(2 * rows, width);
    auto ew2 = b.adopt(experts.packed.data(), experts.packed.size());
    std::vector<uint32_t> route_ids(cols); std::vector<float> gains(cols, 0.75f);
    for (size_t c = 0; c < cols; ++c) route_ids[c] = uint32_t(c);
    auto ri = b.adopt(route_ids.data(), route_ids.size() * sizeof(uint32_t));
    auto rg = b.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing routing{{ri.get(), 0}, {rg.get(), 0}, 1, 2};
    auto routed = b.alloc((count + 2) * sizeof(float));
    std::vector<float> route_guard(count + 2, 0); route_guard.front() = route_guard.back() = 91;
    b.write(*routed, 0, route_guard.data(), route_guard.size() * sizeof(float));
    b.matmul_experts({{{quant::GGML_TYPE_IQ3_XXS, {ew2.get(), 0}}, {routed.get(), 1}, rows}},
                     {x.get(), 0}, width, cols, routing, {&run, 1}, backend::Dtype::f32);
    got = read(b, *routed, count + 2);
    require(got.front() == 91 && got.back() == 91, "IQ3_XXS routed guards");
    for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
        const double want = dot(experts.values.data() + (route_ids[c] * rows + r) * width, input.data() + c * width, width);
        check_close(got[1 + c * rows + r], want, dot_magnitude(experts.values.data() + (route_ids[c] * rows + r) * width, input.data() + c * width, width), width);
    }
    std::fill(route_guard.begin(), route_guard.end(), 0.25f); route_guard.front() = route_guard.back() = 91;
    b.write(*routed, 0, route_guard.data(), route_guard.size() * sizeof(float));
    b.matmul_experts_add({quant::GGML_TYPE_IQ3_XXS, {ew2.get(), 0}}, {x.get(), 0}, {routed.get(), 1}, width, rows, cols, routing, {&run, 1}, backend::Dtype::f32);
    got = read(b, *routed, count + 2);
    require(got.front() == 91 && got.back() == 91, "IQ3_XXS routed residual guards");
    for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
        const float* wr = experts.values.data() + (route_ids[c] * rows + r) * width;
        const double want = 0.25 + gains[c] * dot(wr, input.data() + c * width, width);
        check_close(got[1 + c * rows + r], want, gains[c] * dot_magnitude(wr, input.data() + c * width, width) + 0.25, width);
    }

    // Two picks per token exercise ordered gain accumulation over distinct experts.
    uint32_t two_ids[] = {0, 1, 1, 0};
    float two_gains[] = {0.25f, 0.75f, 0.5f, 0.125f};
    auto two_id_buffer = b.adopt(two_ids, sizeof two_ids);
    auto two_gain_buffer = b.adopt(two_gains, sizeof two_gains);
    std::vector<float> two_input(cols * 2 * width);
    for (size_t e = 0; e < cols * 2; ++e) for (size_t i = 0; i < width; ++i)
        two_input[e * width + i] = float(int((e * 23 + i * 7) % 67) - 33) / 37.0f;
    auto two_x = b.adopt(two_input.data(), two_input.size() * sizeof(float));
    const backend::Backend::Routing two_routing{{two_id_buffer.get(), 0}, {two_gain_buffer.get(), 0}, 2, 2};
    std::fill(route_guard.begin(), route_guard.end(), 0.25f);
    route_guard.front() = route_guard.back() = 91;
    b.write(*routed, 0, route_guard.data(), route_guard.size() * sizeof(float));
    paths(b);
    b.matmul_experts_add({quant::GGML_TYPE_IQ3_XXS, {ew2.get(), 0}}, {two_x.get(), 0},
                        {routed.get(), 1}, width, rows, cols, two_routing, {&run, 1}, backend::Dtype::f32);
    got = read(b, *routed, count + 2);
    require(got.front() == 91 && got.back() == 91, "IQ3_XXS two-pick routed-add guards");
    for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
        double want = 0.25, mag = 0.25;
        for (size_t pick = 0; pick < 2; ++pick) {
            const size_t entry = 2 * c + pick;
            const float* wr = experts.values.data() + (two_ids[entry] * rows + r) * width;
            const double gain = two_gains[entry];
            want += gain * dot(wr, two_input.data() + entry * width, width);
            mag += gain * dot_magnitude(wr, two_input.data() + entry * width, width);
        }
        check_close(got[1 + c * rows + r], want, mag, 2 * width);
    }
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu()) {
        bool tiled_dispatch = false;
        for (const auto& entry : backend::vulkan_kernel_times(b))
            tiled_dispatch |= entry.first.find("matmul_tile_iq3_xxs") != std::string::npos;
        if (backend::vulkan_timed_dispatches(b))
            require(tiled_dispatch, "IQ3_XXS tile-sized matrix did not dispatch a tile kernel");
        std::cout << "IQ3_XXS device: " << backend::vulkan_device_name(b)
                  << "; tile timestamp witness " << (tiled_dispatch ? "present" : "unavailable") << '\n';
    }
#endif
    const size_t range_products = range_repair_products(b);
    std::cout << "IQ3_XXS backend: embed/embed_ids valid+invalid, " << typed_cases
              << " dtype/extent/tile matmul cases plus grouped, residual, routed and k=2 routed-add controls; matrix paths";
    for (const auto& path : witnessed_paths) std::cout << ' ' << path;
    std::cout << "; range repair " << range_products << " products [ok]\n";
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        try { auto device = backend::make_vulkan_backend(0, true); backend_checks(*device); return 0; }
        catch (const backend::VulkanUnavailable& e) { std::cout << "IQ3_XXS Vulkan unavailable: " << e.what() << '\n'; return 77; }
        catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
#else
        std::cout << "IQ3_XXS Vulkan backend not built\n"; return 77;
#endif
    }
    if (argc != 1) { std::cerr << "usage: llmx-iq3-xxs-test [--vulkan]\n"; return 1; }
    try {
        const auto* qt = quant::Registry::instance().get(quant::GGML_TYPE_IQ3_XXS);
        require(qt && qt->block_size == 256 && qt->type_size == 98 &&
                qt->dequantize == quant::dequantize_row_iq3_xxs && !qt->quantize, "IQ3_XXS read-only registry");
        require(quant::row_bytes(quant::GGML_TYPE_IQ3_XXS, 512, 2) == 392, "IQ3_XXS row bytes");
        bool refused = false;
        try { (void)quant::row_bytes(quant::GGML_TYPE_IQ3_XXS, 257); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "IQ3_XXS partial block accepted");
        quant::dequantize_row_iq3_xxs(nullptr, nullptr, 0);
        Indices ids{}; Scales scales{}; Codes codes{};
        for (unsigned shift = 0; shift < 256; ++shift) {
            for (unsigned i = 0; i < 64; ++i) ids[i] = (shift + i) % 256;
            for (unsigned i = 0; i < 8; ++i) scales[i] = (shift + i) % 16;
            for (unsigned i = 0; i < 32; ++i) codes[i] = (shift + i * 7) % 128;
            check_decode(0x3c00, ids, scales, codes);
        }
        for (unsigned h : {0x0000u, 0x8000u, 0x0001u, 0x8001u, 0x03ffu, 0x0400u, 0x3c00u, 0xbc00u, 0x7bffu, 0xfbffu}) {
            for (unsigned i = 0; i < 64; ++i) ids[i] = (37 * i + h) % 256;
            for (unsigned i = 0; i < 8; ++i) scales[i] = (h + i) % 16;
            for (unsigned i = 0; i < 32; ++i) codes[i] = (h + i * 11) % 128;
            check_decode(uint16_t(h), ids, scales, codes);
        }
        for (unsigned domain = 0; domain < 4; ++domain)
            for (unsigned rotation = 0; rotation < 4; ++rotation) {
                for (unsigned i = 0; i < 32; ++i) codes[i] = domain * 32 + ((i + rotation) % 32);
                check_decode(0x3c00, ids, scales, codes);
            }
        std::cout << "IQ3_XXS: " << checked << " exact decoder values, field coverage, finite half edges, guards [ok]\n";
        backend::CpuBackend cpu; cpu.set_threads(1); backend_checks(cpu);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
