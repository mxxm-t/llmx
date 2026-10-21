#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "matrix_precision.hpp"
#include "row_classes.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

namespace {
constexpr uint32_t type = 11;
constexpr size_t width = 256, block_bytes = 110;
constexpr float sentinel = 17;
size_t decoded = 0, checked = 0, integer_calls = 0;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t bits(float x) { uint32_t n; std::memcpy(&n, &x, 4); return n; }
float half(uint16_t h) {
    const unsigned e = h / 1024 % 32, m = h % 1024;
    return std::copysign(float(std::ldexp(double(m + (e ? 1024 : 0)), int(e ? e : 1) - 25)), h & 32768 ? -1.0f : 1.0f);
}
float rounded(float x, backend::Dtype dtype) {
    if (dtype != backend::Dtype::bf16) return x;
    uint32_t n = bits(x), discarded = n & 65535;
    n = (n & 0xffff0000u) + (discarded > 32768 || (discarded == 32768 && (n & 65536)) ? 65536 : 0);
    std::memcpy(&x, &n, 4); return x;
}
int code(size_t i, unsigned seed) {
    constexpr unsigned planes[] = {0, 3, 6, 1, 5, 2, 7, 4};
    return int((i % 32 + planes[i / 32] + seed) & 7) - 4;
}
int subscale(size_t group, unsigned seed) { return int((11 * group + seed) & 63) - 32; }
float value(uint16_t h, unsigned seed, size_t i) { return (half(h) * float(subscale(i / 16, seed))) * float(code(i, seed)); }
void pack(uint8_t* out, uint16_t h, unsigned seed) {
    std::fill(out, out + block_bytes, uint8_t(0));
    for (size_t g = 0; g < 16; ++g) {
        const unsigned s = unsigned(subscale(g, seed) + 32);
        out[96 + g % 8] |= uint8_t((s & 15) << (4 * (g / 8)));
        out[104 + g % 4] |= uint8_t((s / 16) << (2 * (g / 4)));
    }
    // Low-code planes cover 128 values per 32 bytes; a separate high bit selects nonnegative codes.
    for (size_t i = 0; i < width; ++i) {
        const unsigned q = unsigned(code(i, seed) + 4);
        out[i % 32] |= uint8_t((q / 4) << (i / 32));
        out[32 + (i / 128) * 32 + i % 32] |= uint8_t((q % 4) << (2 * ((i % 128) / 32)));
    }
    out[108] = uint8_t(h); out[109] = uint8_t(h >> 8);
}
std::vector<float> read(backend::Backend& b, const backend::Buffer& src, size_t count) {
    std::vector<float> v(count); b.read(src, 0, v.data(), count * 4); return v;
}
void witness(backend::Backend& b, backend::Dtype dtype, bool integer) {
    require(testq::take_matrix_paths(b) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
            "Q3_K executed matrix path differs from its dtype/profile/extent policy");
    if (integer) ++integer_calls;
}
void decode() {
    const auto* q = quant::Registry::instance().get(type);
    require(q && !q->quantize && q->dequantize && q->block_size == width && q->type_size == block_bytes, "Q3_K registry entry differs");
    std::array<uint8_t, 1 + 2 * block_bytes> storage{};
    std::array<float, 2 * width + 2> output{};
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        pack(storage.data() + 1, uint16_t(h), h);
        pack(storage.data() + 1 + block_bytes, uint16_t(h), h + 19);
        const auto saved = storage;
        output.fill(sentinel); q->dequantize(storage.data() + 1, output.data() + 1, 2);
        require(storage == saved && output.front() == sentinel && output.back() == sentinel, "Q3_K decode crossed guards or changed input");
        for (size_t i = 0; i < 2 * width; ++i) {
            require(bits(output[i + 1]) == bits(value(uint16_t(h), h + (i >= width ? 19 : 0), i % width)),
                    "Q3_K finite-half/high-bit/six-bit-scale decode differs");
            ++decoded;
        }
    }
}
void embeddings(backend::Backend& b) {
    std::array<uint8_t, 3 * block_bytes> storage{};
    std::vector<float> expected(3 * width), initial(5 * width + 2, sentinel);
    const auto table = b.alloc(storage.size()), out = b.alloc(initial.size() * 4);
    const uint32_t host[] = {2, 0, 2}, resident[] = {3, 2, 0, UINT32_MAX, 1};
    const auto ids = b.adopt(resident, sizeof resident);
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001),
                       uint16_t(0x3c00), uint16_t(0xbc00), uint16_t(0x7bff), uint16_t(0xfbff)})
    for (unsigned seed = 0; seed < 64; ++seed) {
        for (size_t row = 0; row < 3; ++row) {
            pack(storage.data() + row * block_bytes, h, seed + unsigned(row));
            for (size_t i = 0; i < width; ++i) expected[row * width + i] = value(h, seed + unsigned(row), i);
        }
        b.write(*table, 0, storage.data(), storage.size());
        for (bool device_ids : {false, true}) {
            b.write(*out, 0, initial.data(), initial.size() * 4);
            if (device_ids) b.embed_ids({out.get(), 1}, type, {table.get(), 0}, width, 3, {ids.get(), 0}, 5);
            else b.embed({out.get(), 1}, type, {table.get(), 0}, width, 3, host, 3);
            const auto got = read(b, *out, initial.size());
            const size_t n = (device_ids ? 5 : 3) * width;
            require(got.front() == sentinel && std::all_of(got.begin() + n + 1, got.end(), [](float x) { return x == sentinel; }), "Q3_K embedding crossed guards");
            const auto check = [&](uint32_t row, size_t i) {
                require(bits(got[i + 1]) == bits(row < 3 ? expected[row * width + i % width] : 0), "Q3_K embedding value or zero sign differs");
                ++checked;
            };
            if (device_ids) {
                for (size_t i = 0; i < n; ++i) check(resident[i / width], i);
            } else {
                for (size_t i = 0; i < n; ++i) check(host[i / width], i);
            }
        }
    }
}
void projections(backend::Backend& b, size_t nin, size_t cols, size_t extent = 0) {
    constexpr size_t rows = 3, experts = 2, picks = 2;
    std::vector<uint8_t> packed(1 + experts * rows * (nin / width) * block_bytes);
    std::vector<float> weights(experts * rows * nin), x(cols * picks * nin);
    for (size_t j = 0; j < weights.size() / width; ++j) {
        const uint16_t h = uint16_t((j & 1 ? 0x8000 : 0) | (j % 13 == 0 ? 1 : 0x1801));
        pack(packed.data() + 1 + j * block_bytes, h, unsigned(j));
        for (size_t i = 0; i < width; ++i) weights[j * width + i] = value(h, unsigned(j), i);
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] = i % 7 == 0 ? 1.005859375f : float(int(i * 19 % 251) - 125) / 127;
    const auto reconstructed = testq::row_activations(x);
    std::vector<uint32_t> ids(cols * picks); std::vector<float> gains(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = uint32_t((i / 3 + i) % experts); gains[i] = i & 1 ? .25f : .75f; }
    const auto w = b.adopt(packed.data() + 1, packed.size() - 1), input = b.adopt(x.data(), x.size() * 4);
    const auto ib = b.adopt(ids.data(), ids.size() * 4), gb = b.adopt(gains.data(), gains.size() * 4);
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, picks, experts};
    const std::vector<float> initial(cols * picks * rows + 2, sentinel);
    const auto y = b.alloc(initial.size() * 4), z = b.alloc(initial.size() * 4);
    const backend::RowRun row{cols, 1}, prompt{cols, extent ? extent : std::max<size_t>(2, cols)};
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
    for (const auto* run : {&row, &prompt})
    for (unsigned op = 0; op < 6; ++op) {
        b.write(*y, 0, initial.data(), initial.size() * 4); b.write(*z, 0, initial.data(), initial.size() * 4); testq::take_matrix_paths(b);
        const backend::RowRuns runs{run, 1};
        if (op == 0) b.matmul(type, {w.get(), 0}, {input.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
        if (op == 1) b.matmul_logits(type, {w.get(), 0}, {input.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
        if (op == 2) b.matmul_add(type, {w.get(), 0}, {input.get(), 0}, {y.get(), 1}, nin, rows, cols, runs, dtype);
        if (op == 3) b.matmul_group({{type, {w.get(), 0}, {y.get(), 1}, rows}, {type, {w.get(), 0}, {z.get(), 1}, rows}}, {input.get(), 0}, nin, cols, runs, dtype);
        if (op == 4) b.matmul_experts({{type, {w.get(), 0}, {y.get(), 1}, rows}}, {input.get(), 0}, nin, cols, routing, runs, dtype);
        if (op == 5) b.matmul_experts_add(type, {w.get(), 0}, {input.get(), 0}, {y.get(), 1}, nin, rows, cols, routing, runs, dtype);
        const bool integer = testq::k23_integer(b, dtype);
        const auto got = read(b, *y, initial.size()), other = read(b, *z, initial.size()); witness(b, dtype, integer);
        const size_t entries = op == 4 ? cols * picks : cols;
        require(got.front() == sentinel && std::all_of(got.begin() + 1 + entries * rows, got.end(), [](float v) { return v == sentinel; }), "Q3_K projection crossed guards");
        require(op == 3 ? got == other : other == initial, "Q3_K grouped output or unused storage differs");
        for (size_t entry = 0; entry < entries; ++entry)
        for (size_t output = 0; output < rows; ++output) {
            double want = op == 2 || op == 5 ? sentinel : 0, magnitude = std::fabs(want);
            for (size_t pick = 0; pick < (op == 5 ? picks : 1); ++pick) {
                const size_t e = op == 5 ? entry * picks + pick : entry;
                const size_t expert = op >= 4 ? ids[e] : 0, col = op == 4 ? entry / picks : op == 5 ? e : entry;
                const double gain = op == 5 ? gains[e] : 1;
                for (size_t k = 0; k < nin; ++k) {
                    const double term = double(weights[(expert * rows + output) * nin + k]) * rounded((integer ? reconstructed : x)[col * nin + k], dtype) * gain;
                    want += term; magnitude += std::fabs(term);
                }
            }
            const double nu = double(nin + 6) * 0x1p-24;
            const double bound = nu / (1 - nu) * magnitude + double(nin * picks) * std::numeric_limits<float>::denorm_min();
            require(std::isfinite(got[1 + entry * rows + output]) && std::fabs(double(got[1 + entry * rows + output]) - want) <= bound,
                    "Q3_K projection differs from independent double magnitude bound");
            ++checked;
        }
    }
    require(read(b, *input, x.size()) == x, "Q3_K projection changed its input");
}
void cancellation(backend::Backend& b) {
    constexpr size_t blocks = 17, nin = blocks * width;
    for (unsigned item = 0; item < 3; ++item) {
        const uint16_t h = item == 0 ? 1 : item == 1 ? 0x7bff : 0x3c00;
        const float peak = std::ldexp(1.0f, item == 0 ? 100 : item == 1 ? -100 : 0);
        std::vector<uint8_t> raw(blocks * block_bytes);
        std::vector<float> weights(nin), x(nin, 0);
        for (size_t block_index = 0; block_index < blocks; ++block_index) {
            pack(raw.data() + block_index * block_bytes, h, 31);
            for (size_t i = 0; i < width; ++i) {
                weights[block_index * width + i] = value(h, 31, i);
                if (block_index < 16) x[block_index * width + i] = (block_index < 8 ? peak : -peak) * (i % 7 ? 1.0f : 1.005859375f);
            }
        }
        x[16 * width] = peak / 1024;
        checked += testq::check_k23_cancellation(b, type, raw, weights, x);
    }
}
void ranges(backend::Backend& b) {
    std::array<uint8_t, block_bytes> packed{}; std::array<float, width> x{};
    const auto w = b.alloc(block_bytes), input = b.alloc(sizeof x), y = b.alloc(12);
    const std::array<float, 3> clear{sentinel, sentinel, sentinel};
    const backend::RowRun row{1, 1}, tile{1, 16384};
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x3c00), uint16_t(0x7bff)}) {
        pack(packed.data(), h, 31); b.write(*w, 0, packed.data(), packed.size());
        for (float peak : {std::ldexp(1.0f, -140), std::ldexp(1.0f, -100), 1.0f, std::ldexp(1.0f, 100), std::ldexp(1.0f, 119),
                           std::ldexp(32767.0f, -81), std::ldexp(32767.0f, -80), std::ldexp(32767.0f, 80), std::ldexp(32767.0f, 81)})
        for (size_t lane : {0u, 15u, 16u, 31u, 32u, 127u, 128u, 255u})
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            x.fill(0); x[lane] = peak;
            const double want = double(value(h, 31, lane)) * rounded(x[lane], dtype);
            if (std::fabs(want) > std::numeric_limits<float>::max()) continue;
            b.write(*input, 0, x.data(), sizeof x);
            for (const auto* run : {&row, &tile}) {
                b.write(*y, 0, clear.data(), sizeof clear); testq::take_matrix_paths(b);
                b.matmul(type, {w.get(), 0}, {input.get(), 0}, {y.get(), 1}, width, 1, 1, {run, 1}, dtype);
                const bool integer = testq::k23_integer(b, dtype);
                const auto got = read(b, *y, 3); witness(b, dtype, integer);
                // Extreme twins may round their scale to a subnormal; one reconstruction step plus F32 rounding bounds the selected input independently.
                const double input_error = integer ? std::fabs(double(x[lane])) / 32767.0 + std::numeric_limits<float>::denorm_min()
                                                   + 2 * std::numeric_limits<float>::epsilon() * std::fabs(double(x[lane])) : 0;
                const double bound = 2 * std::numeric_limits<float>::epsilon() * std::fabs(want) + width * double(std::numeric_limits<float>::denorm_min())
                                   + std::fabs(double(value(h, 31, lane))) * input_error;
                require(got.front() == sentinel && got.back() == sentinel && std::isfinite(got[1]) && std::fabs(double(got[1]) - want) <= bound,
                        "Q3_K finite one-hot product differs"); ++checked;
            }
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--vulkan") == 0), "usage: llmx-q3-k-test [--vulkan]");
        _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040)); decode();
        backend::BackendPtr b;
        if (argc == 2) {
#if LLMX_HAS_BACKEND_VULKAN
            try { b = backend::make_vulkan_backend(0, true); }
            catch (const backend::VulkanUnavailable& e) { std::cout << e.what() << '\n'; return 77; }
#else
            std::cout << "Vulkan backend not built\n"; return 77;
#endif
        } else { auto cpu = std::make_shared<backend::CpuBackend>(); cpu->set_threads(1); b = std::move(cpu); }
        require(b->supports_type(type), "backend refuses Q3_K"); embeddings(*b); ranges(*b); cancellation(*b);
        for (size_t nin : {256u, 768u, 4352u}) for (size_t cols : {1u, 3u, 65u, 193u}) projections(*b, nin, cols);
        // Adjacent valid block widths cross the eight-lane row plan's subgroup-fill boundaries.
        for (size_t nin : {512u, 1024u, 1280u, 2048u, 2304u}) projections(*b, nin, 1);
#if LLMX_HAS_BACKEND_VULKAN
        if (!b->is_cpu()) {
            const auto profile = backend::vulkan_device_profile(*b);
            std::vector<size_t> extents;
            for (size_t from : {backend::tile_from_for(profile, false, width), profile.moe_tile_from_q4k})
                for (size_t extent : {std::max<size_t>(1, from - 1), from, from + 1})
                    if (std::find(extents.begin(), extents.end(), extent) == extents.end()) extents.push_back(extent);
            for (size_t extent : extents) projections(*b, width, 3, extent);
        }
#endif
        const size_t pairs = row_classes::check(*b, {type}, [](uint32_t, size_t nin, size_t rows, uint32_t seed) {
            std::vector<uint8_t> raw(rows * nin / width * block_bytes);
            for (size_t j = 0; j < raw.size() / block_bytes; ++j) {
                seed = seed * 1664525u + 1013904223u;
                pack(raw.data() + j * block_bytes, uint16_t(0x1400 | ((seed >> 16) & 0x83ff)), seed);
            }
            return raw;
        });
#if LLMX_HAS_BACKEND_VULKAN
        if (argc == 2) {
            const auto kernels = backend::vulkan_kernel_times(*b);
            bool row = false, tile = false, bf16_tile = false, integer_tile = false, integer_row = false;
            for (const auto& entry : kernels) {
                const auto& n = entry.first;
                const bool float_row = n.rfind("matmul_row_float_x_q3", 0) == 0;
                const bool packed_row = n.rfind("matmul_row_q3_dot", 0) == 0;
                const bool plain_tile = n == "matmul_tile" || n.rfind("matmul_tile ", 0) == 0 || n.rfind("matmul_tile_small", 0) == 0 || n.rfind("matmul_tile_tall", 0) == 0;
                const bool q3_tile = n == "matmul_tile_q3" || n.rfind("matmul_tile_q3 ", 0) == 0 || n.rfind("matmul_tile_q3_small", 0) == 0;
                const bool rounded_tile = n.rfind("matmul_tile_bf16", 0) == 0;
                if (n.rfind("matmul_row", 0) == 0) require(float_row || packed_row, "Q3_K dispatched an unexpected row");
                if (n.rfind("matmul_tile", 0) == 0) require(plain_tile || rounded_tile || q3_tile, "Q3_K dispatched an unexpected tile");
                require(n.rfind("matmul_vec_q", 0) != 0 && n.rfind("matmul_tile_qi8", 0) != 0, "Q3_K dispatched a native int8 module");
                if (q3_tile || packed_row || n.rfind("quantize_x", 0) == 0) require(integer_calls != 0, "Q3_K integer module lacks a block-int16 witness");
                row |= float_row; tile |= plain_tile; bf16_tile |= rounded_tile; integer_tile |= q3_tile; integer_row |= packed_row;
                std::cout << "Q3_K Vulkan dispatch: " << n << '\n';
            }
            if (!kernels.empty()) {
                require(row && tile && bf16_tile, "Q3_K float row/tile/BF16 tile not witnessed");
                require(integer_tile == (integer_calls != 0), "Q3_K block-int16 tile module and arithmetic witnesses disagree");
                require(integer_row == (integer_calls != 0), "Q3_K block-int16 row module and arithmetic witnesses disagree");
            }
            else std::cout << "Q3_K Vulkan dispatch names unavailable without timestamps\n";
        }
#endif
        std::cout << "Q3_K: " << decoded << " exact decoded values, " << checked << " independent embedding/range/product values, "
                  << pairs << " exact row-class pairs; " << integer_calls << " block-int16 calls; F32/BF16 witnesses checked\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
