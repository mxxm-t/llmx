#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include "config.hpp"
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "core/fp16.hpp"
#include "quant/quant.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

namespace {
constexpr uint32_t type = 40;
constexpr float guard = 12345.0f;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

// Independent field arithmetic: unsigned E4M3 (bias 7), times signed E2M1.
// GGUF maps the exact scale byte 127 and both E2M1 zero codes to zero.
float expected(const uint8_t* p, size_t i) {
    const unsigned s = p[i / 16], exponent = (s / 8) % 16, fraction = s % 8;
    const double scale = s == 0 || s == 127 ? 0.0 : exponent
        ? std::ldexp(double(8 + fraction), int(exponent) - 10)
        : std::ldexp(double(fraction), -9);
    const size_t group = i / 16, lane = i % 16;
    const unsigned code = (p[4 + group * 8 + lane % 8] >> (4 * (lane / 8))) & 15;
    const unsigned e = (code / 2) % 4, m = code % 2;
    const double magnitude = e ? std::ldexp(double(2 + m), int(e) - 2) : m * 0.5;
    if (!magnitude) return 0.0f;
    return float((code & 8 ? -magnitude : magnitude) * scale);
}

void check_decode() {
    const auto* q = quant::Registry::instance().get(type);
    require(q && !q->quantize && q->dequantize && q->block_size == 64 && q->type_size == 36,
            "NVFP4 read-only registry entry missing or incorrect");
    require(quant::row_bytes(type, 128, 3) == 216, "NVFP4 row bytes differ");
    q->dequantize(nullptr, nullptr, 0);
    std::array<uint8_t, 110> raw{};
    std::array<float, 194> out{};
    // Every scale byte, nibble code and position, plus adjacent unaligned blocks.
    // High-bit scale bytes pin the reader's bit-field behavior, not producer validity.
    for (unsigned s = 0; s < 256; ++s) {
        for (unsigned shift = 0; shift < 16; ++shift) {
            raw.fill(0xa5);
            out.fill(guard);
            for (size_t b = 0; b < 3; ++b) {
                uint8_t* p = raw.data() + 1 + b * 36;
                for (size_t g = 0; g < 4; ++g) p[g] = uint8_t(s + 37 * g + b);
                for (size_t j = 0; j < 32; ++j)
                    p[4 + j] = uint8_t(((j + shift + b) % 16) | (((15 - j + shift + b) % 16) << 4));
            }
            const auto saved = raw;
            q->dequantize(raw.data() + 1, out.data() + 1, 3);
            require(raw == saved && out.front() == guard && out.back() == guard,
                    "NVFP4 decoder crossed guards or changed source");
            for (size_t b = 0; b < 3; ++b)
                for (size_t i = 0; i < 64; ++i)
                    require(bits(out[1 + b * 64 + i]) == bits(expected(raw.data() + 1 + b * 36, i)),
                            "NVFP4 scale, sign or nibble mapping differs");
        }
    }
}

void check_products(backend::Backend& device, size_t rows, size_t columns,
                    backend::Dtype dtype = backend::Dtype::f32, float factor = 1) {
    constexpr size_t width = 128;
    std::vector<uint8_t> raw(1 + rows * 72, 0);
    for (size_t b = 0; b < rows * 2; ++b) {
        uint8_t* p = raw.data() + 1 + b * 36;
        for (size_t g = 0; g < 4; ++g) p[g] = uint8_t((b * 17 + g * 29) % 127);
        for (size_t j = 0; j < 32; ++j) p[4 + j] = uint8_t(j * 73 + b * 11);
    }
    const auto saved = raw;
    auto w = device.adopt(raw.data() + 1, rows * 72);
    auto scales = factor == 1 ? backend::BufferPtr{} : device.adopt(&factor, sizeof factor);
    const backend::WeightView weight{type, {w.get(), 0}, {scales.get(), 0}, scales ? 1u : 0u};
    auto y = device.alloc((rows * columns + 2) * sizeof(float), backend::Memory::host_visible);
    for (size_t start = 0; start < width; start += columns) {
        std::vector<float> inputs(columns * width, 0), out(rows * columns + 2, guard);
        for (size_t c = 0; c < columns; ++c) inputs[c * width + start + c] = 1;
        auto x = device.adopt(inputs.data(), inputs.size() * sizeof(float));
        device.write(*y, 0, out.data(), out.size() * sizeof(float));
        device.matmul(weight, {x.get(), 0}, {y.get(), 1}, width, rows, columns, {}, dtype);
        device.read(*y, 0, out.data(), out.size() * sizeof(float));
        require(out.front() == guard && out.back() == guard && raw == saved, "NVFP4 product changed guards/source");
        for (size_t c = 0; c < columns; ++c)
            for (size_t r = 0; r < rows; ++r) {
                const size_t i = start + c;
                require(out[1 + c * rows + r] == expected(raw.data() + 1 + r * 72 + (i / 64) * 36, i % 64) * factor,
                        "NVFP4 one-hot product differs");
            }
    }
    const uint32_t ids[] = {uint32_t(rows - 1), 0, 1};
    auto embedded = device.alloc((3 * width + 2) * sizeof(float), backend::Memory::host_visible);
    std::vector<float> out(3 * width + 2, guard);
    device.write(*embedded, 0, out.data(), out.size() * sizeof(float));
    device.embed({embedded.get(), 1}, {type, {w.get(), 0}}, width, rows, ids, 3);
    device.read(*embedded, 0, out.data(), out.size() * sizeof(float));
    require(out.front() == guard && out.back() == guard, "NVFP4 embedding crossed guards");
    for (size_t r = 0; r < 3; ++r)
        for (size_t i = 0; i < width; ++i)
            require(bits(out[1 + r * width + i]) == bits(expected(raw.data() + 1 + ids[r] * 72 + (i / 64) * 36, i % 64)),
                    "NVFP4 embedding differs");
}

#if LLMX_HAS_BACKEND_VULKAN
void check_projection_modes(backend::Backend& device, bool scaled, backend::Dtype dtype) {
    constexpr size_t width = 192, rows = 5, experts = 3, selected = 2;
    std::vector<uint8_t> raw(experts * rows * 3 * 36);
    for (size_t b = 0; b < raw.size() / 36; ++b) {
        uint8_t* p = raw.data() + b * 36;
        for (size_t g = 0; g < 4; ++g) p[g] = uint8_t(1 + (b * 19 + g * 23) % 126);
        for (size_t j = 0; j < 32; ++j) p[4 + j] = uint8_t(b * 13 + j * 37);
    }
    auto weights = device.adopt(raw.data(), raw.size());
    const float factors[] = {.25f, -.5f, .125f, guard};
    auto scales = device.adopt(factors, sizeof factors);
    std::vector<float> ones(width * 3, 1.0f);
    auto floats = device.adopt(ones.data(), ones.size() * sizeof(float));
    for (size_t columns : {size_t(1), size_t(65)}) {
        std::vector<uint32_t> ids(columns * selected);
        std::vector<float> probabilities(columns * selected);
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = uint32_t((i / selected + i % selected) % experts);
            probabilities[i] = i % selected ? .75f : .25f;
        }
        auto choices = device.adopt(ids.data(), ids.size() * sizeof(uint32_t));
        auto probs = device.adopt(probabilities.data(), probabilities.size() * sizeof(float));
        const backend::RowRun run{columns, columns};
        for (int mode = 0; mode < 6; ++mode) {
            const bool routed = mode >= 4, add = mode == 2 || mode == 5;
            const size_t input_rows = mode == 5 ? columns * selected : columns;
            const size_t output_rows = mode == 4 ? columns * selected : columns;
            std::vector<float> input(input_rows * width, 0.0f);
            for (size_t c = 0; c < input_rows; ++c) input[c * width + (c * 11 + 17) % width] = 1.0f;
            auto x = device.adopt(input.data(), input.size() * sizeof(float));
            std::vector<float> output(output_rows * rows + 2, guard), second(columns * 3 + 2, guard);
            std::fill(output.begin() + 1, output.end() - 1, add ? .125f : 0.0f);
            auto y = device.adopt(output.data(), output.size() * sizeof(float));
            auto y2 = device.adopt(second.data(), second.size() * sizeof(float));
            const backend::CSlice in{x.get(), 0};
            const backend::WeightView w{type, {weights.get(), 0}, {scaled ? scales.get() : nullptr, 0}, scaled ? (routed ? experts : 1) : 0};
            const backend::Slice out{y.get(), 1};
            if (mode == 0) device.matmul(w, in, out, width, rows, columns, {&run, 1}, dtype);
            if (mode == 1) device.matmul_logits(w, in, out, width, rows, columns, {&run, 1}, dtype);
            if (mode == 2) device.matmul_add(w, in, out, width, rows, columns, {&run, 1}, dtype);
            if (mode == 3) device.matmul_group({{w, out, rows}, {{0, {floats.get(), 0}}, {y2.get(), 1}, 3}}, in, width, columns, {&run, 1}, dtype);
            if (mode == 4) device.matmul_experts({{w, out, rows}}, in, width, columns,
                {{choices.get(), 0}, {probs.get(), 0}, selected, experts}, {&run, 1}, dtype);
            if (mode == 5) device.matmul_experts_add(w, in, out, width, rows, columns,
                {{choices.get(), 0}, {probs.get(), 0}, selected, experts}, {&run, 1}, dtype);
            device.read(*y, 0, output.data(), output.size() * sizeof(float));
            require(output.front() == guard && output.back() == guard, "NVFP4 projection mode crossed guards");
            for (size_t c = 0; c < output_rows; ++c) for (size_t r = 0; r < rows; ++r) {
                float want = 0;
                for (size_t s = 0; s < (mode == 5 ? selected : 1); ++s) {
                    const size_t at = mode == 5 ? c * selected + s : c;
                    const size_t input_row = mode == 4 ? c / selected : at;
                    const size_t column = (input_row * 11 + 17) % width;
                    const size_t weight_row = (routed ? ids[at] * rows : 0) + r;
                    const float value = expected(raw.data() + (weight_row * 3 + column / 64) * 36, column % 64) *
                        (scaled ? factors[routed ? ids[at] : 0] : 1.0f);
                    want += value * (mode == 5 ? probabilities[at] : 1.0f);
                }
                if (add) want += .125f;
                require(output[1 + c * rows + r] == want, "NVFP4 grouped, routed or residual product differs");
            }
            if (mode == 3) {
                device.read(*y2, 0, second.data(), second.size() * sizeof(float));
                require(second.front() == guard && second.back() == guard, "NVFP4 mixed group crossed guards");
                for (size_t i = 1; i + 1 < second.size(); ++i)
                    require(second[i] == 1.0f, "NVFP4 mixed group changed the F32 projection");
            }
        }
    }
    std::cout << "NVFP4 Vulkan " << (scaled ? "scaled" : "unscaled") << " " << backend::dtype_name(dtype) << ": six matrix entry points, mixed group, routed choices and weighted residual [ok]\n";
}

void check_device_decode(backend::Backend& device, float factor = 1) {
    constexpr size_t rows = 256 * 16, width = 64;
    std::vector<uint8_t> raw(rows * 36);
    std::vector<uint32_t> ids(rows);
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* p = raw.data() + r * 36;
        const unsigned scale = unsigned(r / 16), shift = unsigned(r % 16);
        for (size_t g = 0; g < 4; ++g) p[g] = uint8_t(scale + 37 * g);
        for (size_t j = 0; j < 32; ++j)
            p[4 + j] = uint8_t(((j + shift) % 16) | (((15 - j + shift) % 16) << 4));
        ids[r] = uint32_t(rows - 1 - r);
    }
    const auto saved = raw;
    auto weights = device.adopt(raw.data(), raw.size());
    auto scale = device.adopt(&factor, sizeof factor);
    auto output = device.alloc((rows * width + 2) * sizeof(float), backend::Memory::host_visible);
    std::vector<float> out(rows * width + 2, guard);
    device.write(*output, 0, out.data(), out.size() * sizeof(float));
    device.embed({output.get(), 1}, {type, {weights.get(), 0}, {scale.get(), 0}, 1}, width, rows, ids.data(), rows);
    device.read(*output, 0, out.data(), out.size() * sizeof(float));
    require(raw == saved && out.front() == guard && out.back() == guard, "NVFP4 device decode changed guards/source");
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < width; ++i)
            require(bits(out[1 + r * width + i]) == bits(expected(raw.data() + ids[r] * 36, i) * factor),
                    "NVFP4 device scale/code/position or signed zero differs");
    std::cout << "NVFP4 Vulkan scale " << factor << ": 262144 exhaustive embedding values, signed zeros and high-bit scales [ok]\n";
}

void check_scale_range(backend::Backend& device) {
    constexpr size_t width = 128, rows = 3;
    std::vector<uint8_t> raw(rows * 72, 0x77);
    for (size_t b = 0; b < rows * 2; ++b)
        for (size_t g = 0; g < 4; ++g) raw[b * 36 + g] = 0x38;
    const float scale = .0001f;
    auto weights = device.adopt(raw.data(), raw.size()), scales = device.adopt(&scale, sizeof(scale));
    const backend::WeightView weight{type, {weights.get(), 0}, {scales.get(), 0}, 1};
    for (size_t columns : {size_t(1), size_t(65)}) {
        std::vector<float> inputs(width * columns, 0), output(rows * columns + 2, guard);
        for (size_t c = 0; c < columns; ++c) inputs[c * width] = 1e38f;
        std::fill(output.begin() + 1, output.end() - 1, .25f);
        auto x = device.adopt(inputs.data(), inputs.size() * sizeof(float));
        auto y = device.adopt(output.data(), output.size() * sizeof(float));
        device.matmul_add(weight, {x.get(), 0}, {y.get(), 1}, width, rows, columns, {}, backend::Dtype::f32);
        device.read(*y, 0, output.data(), output.size() * sizeof(float));
        const float expected_value = (6.0f * scale) * 1e38f + .25f;
        require(output.front() == guard && output.back() == guard, "scaled range guards changed");
        for (size_t i = 1; i + 1 < output.size(); ++i)
            require(std::isfinite(output[i]) && std::abs(double(output[i]) - expected_value) <= double(expected_value) * 2e-7,
                    "NVFP4 scale was not applied before overflow and residual addition");
        std::fill(output.begin(), output.end(), guard);
        device.write(*y, 0, output.data(), output.size() * sizeof(float));
        auto invalid = weight;
        invalid.scale_count = 2;
        bool refused = false;
        try {
            device.matmul_group({{weight, {y.get(), 1}, rows}, {invalid, {y.get(), 1}, rows}},
                                {x.get(), 0}, width, columns, {}, backend::Dtype::f32);
        } catch (const std::runtime_error&) { refused = true; }
        device.read(*y, 0, output.data(), output.size() * sizeof(float));
        require(refused && std::all_of(output.begin(), output.end(), [](float value) { return value == guard; }),
                "invalid later scale wrote grouped output");
    }
    std::cout << "NVFP4 Vulkan: scale before overflowing product and residual, invalid grouped scale before writes [ok]\n";
}

void check_f16_input_boundaries(backend::Backend& device) {
    constexpr size_t width = 128, rows = 65, columns = 128;
    struct Case { float input; uint16_t half; float widened; };
    const Case cases[] = {
        {1.0f + 0x1p-11f, 0x3c00u, 1.0f},                 // tie to even lower normal
        {1.0f + 3.0f * 0x1p-11f, 0x3c02u, 1.0f + 0x1p-9f}, // tie to even upper normal
        {65504.0f, 0x7bffu, 65504.0f},
        {65520.0f, 0x7c00u, INFINITY},
        {70000.0f, 0x7c00u, INFINITY},
        {-70000.0f, 0xfc00u, -INFINITY},
        {0x1p-24f, 0x0001u, 0x1p-24f},                   // smallest positive subnormal
        {0x1p-25f, 0x0000u, 0.0f},                       // halfway to zero, even
        {-0x1p-25f, 0x8000u, -0.0f},
        {-0.0f, 0x8000u, -0.0f},
    };
    for (const Case& c : cases)
        require(f32_to_f16(c.input) == c.half && bits(f16_to_f32(c.half)) == bits(c.widened),
                "NVFP4 F16 boundary oracle disagrees with literal binary16 rounding");

    // E4M3 byte 56 and E2M1 code 2 both decode to 1. Every weight is nonzero,
    // so overflow exercises inf*1 rather than 0*inf; only K=0 is nonzero in X.
    std::vector<uint8_t> raw(rows * 72);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < 2; ++b) {
            uint8_t* block = raw.data() + r * 72 + b * 36;
            std::fill(block, block + 4, uint8_t(56));
            std::fill(block + 4, block + 36, uint8_t(0x22));
        }
    }
    auto weights = device.adopt(raw.data(), raw.size());
    const backend::WeightView weight{type, {weights.get(), 0}, {}, 0};
    std::vector<float> inputs(columns * width, 0.0f);
    for (size_t c = 0; c < columns; ++c) inputs[c * width] = cases[c % (sizeof(cases) / sizeof(cases[0]))].input;
    auto x = device.adopt(inputs.data(), inputs.size() * sizeof(float));
    auto output = device.alloc((rows * columns + 2) * sizeof(float), backend::Memory::host_visible);
    std::vector<float> result(rows * columns + 2, guard);

    const bool half_inputs = device.dtype_path(backend::Dtype::f16).find("f16 (NVFP4 tile inputs)") == 0;
    backend::MatrixPaths prior;
    device.swap_matrix_paths(prior);
    device.write(*output, 0, result.data(), result.size() * sizeof(float));
    device.matmul(weight, {x.get(), 0}, {output.get(), 1}, width, rows, columns, {}, backend::Dtype::f16);
    device.read(*output, 0, result.data(), result.size() * sizeof(float));
    backend::MatrixPaths f16_paths;
    device.swap_matrix_paths(f16_paths);
    require(f16_paths.take() == std::vector<std::string>{half_inputs ? "f16" : "f32"}, "NVFP4 tile input precision differs from device capabilities");
    require(result.front() == guard && result.back() == guard, "NVFP4 F16 boundary tile crossed output guards");
    for (size_t c = 0; c < columns; ++c) {
        const Case& expected_case = cases[c % (sizeof(cases) / sizeof(cases[0]))];
        float rounded_dot = 0.0f;
        for (size_t k = 0; k < width; ++k)
            rounded_dot = std::fma(1.0f, k == 0 ? (half_inputs ? expected_case.widened : expected_case.input) : 0.0f, rounded_dot);
        for (size_t r = 0; r < rows; ++r) {
            if (bits(result[1 + c * rows + r]) != bits(rounded_dot))
                std::cerr << "F16 boundary column=" << c << " row=" << r << " input=" << expected_case.input
                          << " got_bits=" << bits(result[1 + c * rows + r]) << " expected_bits=" << bits(rounded_dot) << '\n';
            require(bits(result[1 + c * rows + r]) == bits(rounded_dot),
                    "NVFP4 F16 tile input conversion differs at a binary16 boundary");
        }
    }
    bool f16_tile = false;
    for (const auto& entry : backend::vulkan_kernel_times(device))
        f16_tile |= entry.first.find("matmul_tile_nvfp4_f16") == 0;
    if (backend::vulkan_timed_dispatches(device))
        require(f16_tile == half_inputs, "NVFP4 tile kernel differs from native F16 capability");

    std::fill(result.begin(), result.end(), guard);
    device.write(*output, 0, result.data(), result.size() * sizeof(float));
    device.matmul(weight, {x.get(), 0}, {output.get(), 1}, width, rows, columns, {}, backend::Dtype::f32);
    device.read(*output, 0, result.data(), result.size() * sizeof(float));
    backend::MatrixPaths f32_paths;
    device.swap_matrix_paths(f32_paths);
    require(f32_paths.take() == std::vector<std::string>{"f32"}, "explicit-F32 NVFP4 control did not record F32");
    require(result.front() == guard && result.back() == guard, "explicit-F32 NVFP4 control crossed output guards");
    for (size_t c = 0; c < columns; ++c)
        for (size_t r = 0; r < rows; ++r) {
            float f32_dot = 0.0f;
            for (size_t k = 0; k < width; ++k)
                f32_dot = std::fma(1.0f, inputs[c * width + k], f32_dot);
            require(bits(result[1 + c * rows + r]) == bits(f32_dot),
                    "explicit-F32 NVFP4 control changed its input value");
        }
    std::cout << "NVFP4 Vulkan: literal F16 input boundary conversion, F32 accumulation and explicit-F32 control [ok]\n";
}

void check_vulkan(backend::Backend& device) {
    check_products(device, 3, 1);
    check_products(device, 65, 128);
    bool row = false, tile = false;
    for (const auto& entry : backend::vulkan_kernel_times(device)) {
        row |= entry.first.find("matmul_row") != std::string::npos;
        tile |= entry.first.find("matmul_tile") != std::string::npos;
    }
    if (backend::vulkan_timed_dispatches(device))
        require(row && tile, "NVFP4 one-hot checks did not witness both row and tile kernels");
    std::cout << "NVFP4 Vulkan: " << backend::vulkan_device_name(device)
              << "; row/tile timestamp witnesses " << (row && tile ? "present" : "unavailable") << '\n';
    // On a 60-CU device these narrow rows fill the 64-row tile, with one output row past the last whole tile.
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16}) {
        for (float factor : {1.0f, .25f}) check_products(device, 2049, 128, dtype, factor);
        std::cout << "NVFP4 Vulkan 2049-row one-hot " << backend::dtype_name(dtype) << ":";
        bool witnessed = false;
        for (const auto& entry : backend::vulkan_kernel_times(device)) {
            if (entry.first.find("matmul_tile_nvfp4") != 0) continue;
            std::cout << ' ' << entry.first;
            witnessed = true;
        }
        if (backend::vulkan_timed_dispatches(device))
            require(witnessed, "NVFP4 intermediate one-hot checks did not witness a tile kernel");
        std::cout << (witnessed ? " [ok]\n" : " timestamps unavailable [ok]\n");
    }
    // Whole 128-row tiles fill the MI50's narrow tall-tile rule. One extra output row checks the tail while WPR=1 must decode both scale groups.
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16})
        for (float factor : {1.0f, .25f}) check_products(device, 65537, 128, dtype, factor);
    bool tall = false, bf16_tall = false;
    for (const auto& entry : backend::vulkan_kernel_times(device)) {
        tall |= entry.first.find("matmul_tile_nvfp4_tall") != std::string::npos;
        bf16_tall |= entry.first.find("matmul_tile_nvfp4_bf16_tall") != std::string::npos;
    }
    if (backend::vulkan_timed_dispatches(device))
        require(tall && bf16_tall, "NVFP4 one-hot checks did not witness both tall tile policies");
    for (float scale : {1.0f, -.125f, 0.0f, -0.0f}) check_device_decode(device, scale);
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16}) {
        check_projection_modes(device, false, dtype);
        check_projection_modes(device, true, dtype);
    }
    check_scale_range(device);
    check_f16_input_boundaries(device);
}
#endif
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        try { auto device = backend::make_vulkan_backend(0, true); check_vulkan(*device); return 0; }
        catch (const backend::VulkanUnavailable& e) { std::cout << "NVFP4 Vulkan unavailable: " << e.what() << '\n'; return 77; }
        catch (const std::exception& e) { std::cerr << "NVFP4: " << e.what() << '\n'; return 1; }
#else
        std::cout << "NVFP4 Vulkan backend not built\n"; return 77;
#endif
    }
    if (argc != 1) { std::cerr << "usage: llmx-nvfp4-test [--vulkan]\n"; return 1; }
    try {
        check_decode();
        for (int threads : {1, 3}) {
            backend::CpuBackend cpu;
            cpu.set_threads(threads);
            check_products(cpu, 3, 1);
            check_products(cpu, 65, 128);
        }
        std::cout << "NVFP4: all scale bytes/codes/positions, unaligned blocks, guards, CPU one-hot products and embeddings [ok]\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NVFP4: " << e.what() << '\n';
        return 1;
    }
}
