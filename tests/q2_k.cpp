#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "matrix_precision.hpp"
#include "row_classes.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

namespace {
constexpr uint32_t type = 10;
constexpr size_t block_size = 256, block_bytes = 84;
constexpr float guard = 19;
size_t decoded = 0, checked = 0, integer_calls = 0;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t bits(float value) { uint32_t out; std::memcpy(&out, &value, 4); return out; }
float half(uint16_t h) {
    const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
    return std::copysign(float(std::ldexp(double(fraction + (exponent ? 1024 : 0)), int(exponent ? exponent : 1) - 25)),
                         h & 0x8000 ? -1.0f : 1.0f);
}
float bf16(float value) {
    uint32_t word = bits(value), low = word & 65535;
    word = (word & 0xffff0000u) + (low > 32768 || (low == 32768 && (word & 65536)) ? 65536 : 0);
    std::memcpy(&value, &word, 4);
    return value;
}
void block(uint8_t* data, uint16_t d, uint16_t m, unsigned shift) {
    std::fill(data, data + block_bytes, uint8_t(0));
    for (unsigned group = 0; group < 16; ++group)
        data[group] = uint8_t(((group + shift) & 15) | (((3 * group + shift) & 15) << 4));
    // Each 32-byte half stores four planes of 32 two-bit codes, while scale/min pairs cover 16 values each.
    for (unsigned i = 0; i < block_size; ++i) {
        const unsigned code = (i / 16 + i % 16 + i / 32 + i / 128 + shift) & 3;
        data[16 + (i / 128) * 32 + i % 32] |= uint8_t(code << (2 * ((i % 128) / 32)));
    }
    data[80] = uint8_t(d); data[81] = uint8_t(d >> 8);
    data[82] = uint8_t(m); data[83] = uint8_t(m >> 8);
}
float value(uint16_t d, uint16_t m, unsigned shift, size_t i) {
    const unsigned group = unsigned(i / 16);
    const unsigned q = unsigned((i / 16 + i % 16 + i / 32 + i / 128 + shift) & 3);
    const float scaled = half(d) * float((group + shift) & 15);
    const float minimum = half(m) * float((3 * group + shift) & 15);
    return scaled * float(q) - minimum;
}
std::vector<float> read(backend::Backend& b, const backend::Buffer& buffer, size_t count) {
    std::vector<float> out(count);
    b.read(buffer, 0, out.data(), out.size() * sizeof(float));
    return out;
}
void witness(backend::Backend& b, backend::Dtype dtype, bool integer) {
    require(testq::take_matrix_paths(b) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
            "Q2_K executed matrix path differs from its dtype/profile/extent policy");
    if (integer) ++integer_calls;
}
void decode() {
    const auto* qt = quant::Registry::instance().get(type);
    require(qt && !qt->quantize && qt->dequantize && qt->block_size == block_size && qt->type_size == block_bytes,
            "Q2_K read-only registry entry differs");
    std::array<uint8_t, 2 * block_bytes + 1> raw{};
    std::array<float, 2 * block_size + 2> out{};
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        for (bool minimum : {false, true}) {
            const uint16_t d = minimum ? uint16_t(0xa401) : uint16_t(h);
            const uint16_t m = minimum ? uint16_t(h) : uint16_t(0x2003);
            block(raw.data() + 1, d, m, h & 15);
            block(raw.data() + 1 + block_bytes, d, m, (h + 9) & 15);
            const auto saved = raw;
            out.fill(guard);
            qt->dequantize(raw.data() + 1, out.data() + 1, 2);
            require(raw == saved && out.front() == guard && out.back() == guard, "Q2_K decoder changed input or output guards");
            for (size_t i = 0; i < 2 * block_size; ++i) {
                const unsigned shift = (h + (i >= block_size ? 9 : 0)) & 15;
                require(bits(out[1 + i]) == bits(value(d, m, shift, i % block_size)), "Q2_K decode differs from independent finite-half/code math");
                ++decoded;
            }
        }
    }
}
void embeddings(backend::Backend& b) {
    constexpr size_t rows = 3, width = block_size;
    std::vector<uint8_t> raw(rows * block_bytes);
    std::vector<float> expected(rows * width), initial(6 * width + 2, guard);
    const auto w = b.alloc(raw.size()), out = b.alloc(initial.size() * 4);
    const uint32_t host_ids[] = {2, 0, 1, 2}, device_ids[] = {2, 3, 0, UINT32_MAX, 1, 2};
    const auto ids = b.adopt(device_ids, sizeof device_ids);
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001),
                       uint16_t(0x3c00), uint16_t(0xbc00), uint16_t(0x7bff), uint16_t(0xfbff)})
    for (unsigned shift = 0; shift < 16; ++shift) {
        for (size_t r = 0; r < rows; ++r) {
            const uint16_t m = r == 0 ? uint16_t(0) : r == 1 ? uint16_t(0x8000) : uint16_t(0x2401);
            block(raw.data() + r * block_bytes, h, m, shift + unsigned(r));
            for (size_t i = 0; i < width; ++i) expected[r * width + i] = value(h, m, shift + unsigned(r), i);
        }
        b.write(*w, 0, raw.data(), raw.size());
        for (bool resident : {false, true}) {
            b.write(*out, 0, initial.data(), initial.size() * 4);
            if (resident) b.embed_ids({out.get(), 1}, {type, {w.get(), 0}}, width, rows, {ids.get(), 0}, 6);
            else b.embed({out.get(), 1}, {type, {w.get(), 0}}, width, rows, host_ids, 4);
            const auto got = read(b, *out, initial.size());
            const size_t count = resident ? 6 : 4;
            require(got.front() == guard && std::all_of(got.begin() + 1 + count * width, got.end(), [](float x) { return x == guard; }),
                    "Q2_K embedding crossed output guards");
            const auto check = [&](uint32_t id, size_t i) {
                const float want = id < rows ? expected[id * width + i % width] : 0;
                require(bits(got[1 + i]) == bits(want), "Q2_K embedding differs from independent values or zero signs");
                ++checked;
            };
            if (resident) {
                for (size_t i = 0; i < 6 * width; ++i) check(device_ids[i / width], i);
            } else {
                for (size_t i = 0; i < 4 * width; ++i) check(host_ids[i / width], i);
            }
        }
    }
}
void products(backend::Backend& b, size_t nin, size_t cols, size_t extent = 0) {
    constexpr size_t rows = 5, experts = 3, picks = 2;
    const size_t blocks = nin / block_size, rb = blocks * block_bytes, count = rows * cols;
    std::vector<uint8_t> raw(1 + experts * rows * rb);
    std::vector<float> weights(experts * rows * nin), x(1 + cols * picks * nin, guard);
    for (size_t j = 0; j < weights.size() / block_size; ++j) {
        const uint16_t d = uint16_t((j & 1 ? 0x8000 : 0) | (j % 7 == 0 ? 1 : 0x2401));
        const uint16_t m = uint16_t((j % 3 == 0 ? 0x8000 : 0) | (j % 11 == 0 ? 0 : 0x2003));
        block(raw.data() + 1 + j * block_bytes, d, m, unsigned(j));
        for (size_t i = 0; i < block_size; ++i) weights[j * block_size + i] = value(d, m, unsigned(j), i);
    }
    for (size_t i = 1; i < x.size(); ++i) x[i] = i % 7 == 0 ? 1.005859375f : float(int(i * 17 % 251) - 125) / 127.0f;
    const auto reconstructed = testq::row_activations(std::vector<float>(x.begin() + 1, x.end()));
    std::vector<uint32_t> ids(cols * picks);
    std::vector<float> gains(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = uint32_t((i + 1) % experts); gains[i] = i & 1 ? 0.25f : 0.75f; }
    std::vector<float> initial(count * picks + 2, guard);
    const auto w = b.adopt(raw.data() + 1, raw.size() - 1), in = b.adopt(x.data(), x.size() * 4);
    const auto a = b.alloc(initial.size() * 4), other = b.alloc(initial.size() * 4);
    const auto ib = b.adopt(ids.data(), ids.size() * 4), gb = b.adopt(gains.data(), gains.size() * 4);
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, picks, experts};
    const backend::RowRun decode[] = {{cols, 1}}, prompt[] = {{cols, extent ? extent : std::max<size_t>(2, cols)}};
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
    for (backend::RowRuns runs : {backend::RowRuns{decode, 1}, backend::RowRuns{prompt, 1}})
    for (unsigned op = 0; op < 6; ++op) {
        b.write(*a, 0, initial.data(), initial.size() * 4); b.write(*other, 0, initial.data(), initial.size() * 4);
        testq::take_matrix_paths(b);
        if (op == 0) b.matmul({type, {w.get(), 0}}, {in.get(), 1}, {a.get(), 1}, nin, rows, cols, runs, dtype);
        if (op == 1) b.matmul_logits({type, {w.get(), 0}}, {in.get(), 1}, {a.get(), 1}, nin, rows, cols, runs, dtype);
        if (op == 2) b.matmul_add({type, {w.get(), 0}}, {in.get(), 1}, {a.get(), 1}, nin, rows, cols, runs, dtype);
        if (op == 3) b.matmul_group({{{type, {w.get(), 0}}, {a.get(), 1}, rows}, {{type, {w.get(), 0}}, {other.get(), 1}, rows}}, {in.get(), 1}, nin, cols, runs, dtype);
        if (op == 4) b.matmul_experts({{{type, {w.get(), 0}}, {a.get(), 1}, rows}}, {in.get(), 1}, nin, cols, routing, runs, dtype);
        if (op == 5) b.matmul_experts_add({type, {w.get(), 0}}, {in.get(), 1}, {a.get(), 1}, nin, rows, cols, routing, runs, dtype);
        const auto got = read(b, *a, initial.size()), second = read(b, *other, initial.size());
        const bool integer = testq::k23_integer(b, dtype);
        witness(b, dtype, integer);
        const size_t entries = op == 4 ? cols * picks : cols, used = entries * rows;
        require(got.front() == guard && std::all_of(got.begin() + used + 1, got.end(), [](float v) { return v == guard; }),
                "Q2_K projection crossed output guards");
        if (op == 3) require(got == second, "Q2_K identical grouped projections differ");
        else require(second == initial, "Q2_K projection changed unused output");
        for (size_t entry = 0; entry < entries; ++entry)
        for (size_t row = 0; row < rows; ++row) {
            double want = op == 2 || op == 5 ? guard : 0, magnitude = std::fabs(want);
            for (size_t pick = 0; pick < (op == 5 ? picks : 1); ++pick) {
                const size_t expert_entry = op == 5 ? entry * picks + pick : entry;
                const size_t expert = op >= 4 ? ids[expert_entry] : 0;
                const size_t input = op == 4 ? entry / picks : op == 5 ? expert_entry : entry;
                const double gain = op == 5 ? gains[expert_entry] : 1;
                for (size_t k = 0; k < nin; ++k) {
                    const float activation = integer ? reconstructed[input * nin + k] : x[1 + input * nin + k];
                    const double term = double(weights[(expert * rows + row) * nin + k])
                                      * (dtype == backend::Dtype::bf16 ? bf16(activation) : activation) * gain;
                    want += term; magnitude += std::fabs(term);
                }
            }
            // A conservative gamma_n bound includes each dot's rounding, gain/reduction and residual add, including cancellation.
            const double nu = double(nin + 2 * picks + 2) * 0x1p-24;
            const double bound = nu / (1 - nu) * magnitude + double(nin * picks) * std::numeric_limits<float>::denorm_min();
            const float actual = got[1 + entry * rows + row];
            require(std::isfinite(actual) && std::fabs(double(actual) - want) <= bound, "Q2_K product differs from independent double magnitude bound");
            ++checked;
        }
    }
    require(read(b, *in, x.size()) == x, "Q2_K product changed activations");
}
void cancellation(backend::Backend& b) {
    constexpr size_t blocks = 17, nin = blocks * block_size;
    for (unsigned item = 0; item < 3; ++item) {
        const uint16_t h = item == 0 ? 1 : item == 1 ? 0x7bff : 0x3c00, m = uint16_t(h ^ 0x8000);
        const float peak = std::ldexp(1.0f, item == 0 ? 100 : item == 1 ? -100 : 0);
        std::vector<uint8_t> raw(blocks * block_bytes);
        std::vector<float> weights(nin), x(nin, 0);
        for (size_t block_index = 0; block_index < blocks; ++block_index) {
            block(raw.data() + block_index * block_bytes, h, m, 3);
            for (size_t i = 0; i < block_size; ++i) {
                weights[block_index * block_size + i] = value(h, m, 3, i);
                if (block_index < 16) x[block_index * block_size + i] = (block_index < 8 ? peak : -peak) * (i % 7 ? 1.0f : 1.005859375f);
            }
        }
        x[16 * block_size] = peak / 1024;
        checked += testq::check_k23_cancellation(b, type, raw, weights, x);
    }
}
void ranges(backend::Backend& b) {
    std::array<uint8_t, block_bytes> raw{};
    std::array<float, block_size> x{};
    const std::array<float, 3> initial{guard, guard, guard};
    const auto w = b.alloc(raw.size()), in = b.alloc(sizeof x), out = b.alloc(sizeof initial);
    const backend::RowRun decode{1, 1}, prompt{1, 16384};
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x3c00), uint16_t(0x7bff)}) {
        block(raw.data(), h, uint16_t(h ^ 0x8000), 3);
        b.write(*w, 0, raw.data(), raw.size());
        for (float peak : {std::ldexp(1.0f, -140), std::ldexp(1.0f, -100), 1.0f, std::ldexp(1.0f, 100), std::ldexp(1.0f, 119),
                           std::ldexp(32767.0f, -81), std::ldexp(32767.0f, -80), std::ldexp(32767.0f, 80), std::ldexp(32767.0f, 81)})
        for (size_t lane : {0u, 15u, 16u, 31u, 32u, 127u, 128u, 255u})
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            x.fill(0); x[lane] = peak;
            const double want = double(value(h, uint16_t(h ^ 0x8000), 3, lane))
                              * (dtype == backend::Dtype::bf16 ? bf16(x[lane]) : x[lane]);
            if (std::fabs(want) > std::numeric_limits<float>::max()) continue;
            b.write(*in, 0, x.data(), sizeof x);
            for (const auto* run : {&decode, &prompt}) {
                b.write(*out, 0, initial.data(), sizeof initial); testq::take_matrix_paths(b);
                b.matmul({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, block_size, 1, 1, {run, 1}, dtype);
                const bool integer = testq::k23_integer(b, dtype);
                const auto got = read(b, *out, 3); witness(b, dtype, integer);
                const double input_error = integer ? std::fabs(double(x[lane])) / 32767.0 + std::numeric_limits<float>::denorm_min()
                                                   + 2 * std::numeric_limits<float>::epsilon() * std::fabs(double(x[lane])) : 0;
                const double bound = 2 * std::numeric_limits<float>::epsilon() * std::fabs(want)
                                   + block_size * double(std::numeric_limits<float>::denorm_min())
                                   + std::fabs(double(value(h, uint16_t(h ^ 0x8000), 3, lane))) * input_error;
                require(got.front() == guard && got.back() == guard && std::isfinite(got[1]) && std::fabs(double(got[1]) - want) <= bound,
                        "Q2_K finite one-hot range differs from independent product");
                ++checked;
            }
        }
    }
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--vulkan") == 0), "usage: llmx-q2-k-test [--vulkan]");
        _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040));
        decode();
        backend::BackendPtr b;
        if (argc == 2) {
#if LLMX_HAS_BACKEND_VULKAN
            try { b = backend::make_vulkan_backend(0, true); }
            catch (const backend::VulkanUnavailable& e) { std::cout << e.what() << '\n'; return 77; }
#else
            std::cout << "Vulkan backend not built\n"; return 77;
#endif
        } else {
            auto cpu = std::make_shared<backend::CpuBackend>(); cpu->set_threads(1); b = std::move(cpu);
        }
        require(b->supports_type(type), "backend refuses Q2_K");
        embeddings(*b); ranges(*b); cancellation(*b);
        for (size_t width : {256u, 768u, 4352u})
            for (size_t cols : {1u, 3u, 65u, 193u}) products(*b, width, cols);
        // Adjacent valid block widths cross the eight-lane row plan's subgroup-fill boundaries.
        for (size_t nin : {512u, 1024u, 1280u, 2048u, 2304u}) products(*b, nin, 1);
#if LLMX_HAS_BACKEND_VULKAN
        if (!b->is_cpu()) {
            const auto profile = backend::vulkan_device_profile(*b);
            std::vector<size_t> extents;
            for (size_t from : {backend::tile_from_for(profile, false, block_size), profile.moe_tile_from_q4k})
                for (size_t extent : {std::max<size_t>(1, from - 1), from, from + 1})
                    if (std::find(extents.begin(), extents.end(), extent) == extents.end()) extents.push_back(extent);
            for (size_t extent : extents) products(*b, block_size, 3, extent);
        }
#endif
        const size_t pairs = row_classes::check(*b, {type}, [](uint32_t, size_t nin, size_t rows, uint32_t seed) {
            std::vector<uint8_t> raw(rows * nin / block_size * block_bytes);
            for (size_t j = 0; j < raw.size() / block_bytes; ++j) {
                seed = seed * 1664525u + 1013904223u;
                block(raw.data() + j * block_bytes, uint16_t(0x1400u | ((seed >> 16) & 0x83ffu)),
                      uint16_t(0x1000u | (seed & 0x83ffu)), seed & 15u);
            }
            return raw;
        });
#if LLMX_HAS_BACKEND_VULKAN
        if (argc == 2) {
            const auto kernels = backend::vulkan_kernel_times(*b);
            bool row = false, tile = false, rounded = false, integer_tile = false, integer_row = false;
            for (const auto& kernel : kernels) {
                const auto& name = kernel.first;
                const bool float_row = name.rfind("matmul_row_float_x_q2", 0) == 0;
                const bool packed_row = name.rfind("matmul_row_q2_dot", 0) == 0;
                const bool float_tile = name == "matmul_tile" || name.rfind("matmul_tile ", 0) == 0
                                     || name.rfind("matmul_tile_small", 0) == 0 || name.rfind("matmul_tile_tall", 0) == 0;
                const bool bf16_tile = name.rfind("matmul_tile_bf16", 0) == 0;
                const bool q2_tile = name == "matmul_tile_q2" || name.rfind("matmul_tile_q2 ", 0) == 0 || name.rfind("matmul_tile_q2_small", 0) == 0 || name == "matmul_tile_q2_narrow";
                if (name.rfind("matmul_row", 0) == 0) require(float_row || packed_row, "Q2_K dispatched an unexpected row module");
                if (name.rfind("matmul_tile", 0) == 0) require(float_tile || bf16_tile || q2_tile, "Q2_K dispatched an unexpected tile module");
                require(name.rfind("matmul_vec_q", 0) != 0 && name.rfind("matmul_tile_qi8", 0) != 0, "Q2_K dispatched a native int8 module");
                if (q2_tile || packed_row || name.rfind("quantize_x", 0) == 0) require(integer_calls != 0, "Q2_K integer module lacks a block-int16 witness");
                integer_tile |= q2_tile; integer_row |= packed_row;
                row = row || float_row; tile = tile || float_tile; rounded = rounded || bf16_tile;
                std::cout << "Q2_K Vulkan dispatch: " << name << '\n';
            }
            if (!kernels.empty()) {
                require(row && tile && rounded, "Q2_K float row, float tile or BF16 tile was not witnessed");
                require(integer_tile == (integer_calls != 0), "Q2_K block-int16 tile and arithmetic witnesses disagree");
                require(integer_row == (integer_calls != 0), "Q2_K block-int16 row module and arithmetic witnesses disagree");
            }
            else std::cout << "Q2_K Vulkan dispatch names unavailable without timestamps\n";
        }
#endif
        std::cout << "Q2_K: " << decoded << " exact decoded values, " << checked
                  << " independent embedding/range/product values, " << pairs
                  << " exact row-class pairs; " << integer_calls << " block-int16 calls; F32/BF16 paths checked\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
