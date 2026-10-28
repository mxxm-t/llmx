#include <algorithm>
#include <array>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "core/fp16.hpp"
#include "quant/quant.hpp"
#include "row_classes.hpp"
#include "matrix_precision.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

namespace {
constexpr float guard = 12345.0f;

void require(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(text);
}

uint32_t bits(float x) {
    uint32_t out;
    std::memcpy(&out, &x, sizeof out);
    return out;
}

float half_value(uint16_t h) {
    const unsigned e = (h >> 10) & 31, f = h & 1023;
    const double magnitude = e ? std::ldexp(double(1024 + f), int(e) - 25)
                               : std::ldexp(double(f), -24);
    return std::copysign(float(magnitude), h & 0x8000 ? -1.0f : 1.0f);
}

uint16_t scale(const uint8_t* block) {
    return uint16_t(block[0] | (uint16_t(block[1]) << 8));
}

void set_q1(uint8_t* block, size_t i, unsigned code) {
    const uint8_t mask = uint8_t(1u << (i % 8));
    if (code & 1) block[2 + i / 8] |= mask;
    else block[2 + i / 8] &= uint8_t(~mask);
}

unsigned q1(const uint8_t* block, size_t i) {
    return (block[2 + i / 8] >> (i % 8)) & 1u;
}

void set_q2(uint8_t* block, size_t i, unsigned code) {
    const unsigned shift = 2 * (i % 4);
    const uint8_t mask = uint8_t(3u << shift);
    block[2 + i / 4] = uint8_t((block[2 + i / 4] & uint8_t(~mask)) | (code << shift));
}

unsigned q2(const uint8_t* block, size_t i) {
    return (block[2 + i / 4] >> (2 * (i % 4))) & 3u;
}

float expected_q1(const uint8_t* block, size_t i) {
    const float d = half_value(scale(block));
    return q1(block, i) ? d : -d;
}

float expected_q2(const uint8_t* block, size_t i) {
    return float(int(q2(block, i)) - 1) * half_value(scale(block));
}

template<size_t N>
void check_bits(const std::array<float, N>& out, const uint8_t* raw, size_t nblocks,
                size_t block_bytes, size_t block_values, float (*expected)(const uint8_t*, size_t)) {
    for (size_t b = 0; b < nblocks; ++b)
        for (size_t i = 0; i < block_values; ++i)
            require(bits(out[1 + b * block_values + i]) == bits(expected(raw + 1 + b * block_bytes, i)),
                    "low-bit block decode differs from independent field oracle");
}

void check_positions() {
    for (uint32_t type : {quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0}) {
        const bool one = type == quant::GGML_TYPE_Q1_0;
        const size_t block_bytes = one ? quant::Q1_0_TYPESIZE : quant::Q2_0_TYPESIZE;
        const size_t block_values = one ? quant::Q1_0_BLOCK : quant::Q2_0_BLOCK;
        const size_t blocks = one ? 2 : 4;
        std::vector<uint8_t> raw(1 + blocks * block_bytes + 1, 0xa5);
        std::array<float, 4 * quant::Q1_0_BLOCK + 2> out{};
        out.fill(guard);
        for (size_t b = 0; b < blocks; ++b) {
            uint8_t* block = raw.data() + 1 + b * block_bytes;
            block[0] = uint8_t(0x01 + 2 * b);
            block[1] = uint8_t(0x3c);
            for (size_t i = 0; i < block_values; ++i) {
                if (one) set_q1(block, i, unsigned((i + b) & 1));
                else set_q2(block, i, unsigned((i + b) & 3));
            }
        }
        const auto saved = raw;
        const auto* entry = quant::Registry::instance().get(type);
        require(entry && !entry->quantize && entry->dequantize, "low-bit read-only registry entry missing");
        require(entry->block_size == block_values && entry->type_size == block_bytes, "low-bit registry geometry differs");
        entry->dequantize(raw.data() + 1, out.data() + 1, blocks);
        require(raw == saved && out.front() == guard && out[1 + blocks * block_values] == guard,
                "unaligned low-bit decoder changed source or crossed guards");
        check_bits(out, raw.data(), blocks, block_bytes, block_values, one ? expected_q1 : expected_q2);
        require(quant::row_bytes(type, block_values * 2, 2) == 4 * block_bytes,
                "low-bit row sizing differs");
    }
}

void check_all_finite_scales() {
    for (uint32_t type : {quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0}) {
        const bool one = type == quant::GGML_TYPE_Q1_0;
        const size_t block_bytes = one ? quant::Q1_0_TYPESIZE : quant::Q2_0_TYPESIZE;
        const size_t block_values = one ? quant::Q1_0_BLOCK : quant::Q2_0_BLOCK;
        std::vector<uint8_t> raw(1 + block_bytes + 1, 0xa5);
        std::array<float, quant::Q1_0_BLOCK + 2> out{};
        out.fill(guard);
        uint8_t* block = raw.data() + 1;
        for (size_t i = 0; i < block_values; ++i) {
            if (one) set_q1(block, i, unsigned(i & 1));
            else set_q2(block, i, unsigned(i & 3));
        }
        const auto* entry = quant::Registry::instance().get(type);
        size_t checked = 0;
        for (unsigned h = 0; h <= 0xffff; ++h) {
            if ((h & 0x7c00) == 0x7c00) continue;
            block[0] = uint8_t(h);
            block[1] = uint8_t(h >> 8);
            entry->dequantize(block, out.data() + 1, 1);
            for (size_t i = 0; i < block_values; ++i) {
                const float want = one ? expected_q1(block, i) : expected_q2(block, i);
                require(bits(out[1 + i]) == bits(want), "low-bit finite half scale differs");
                ++checked;
            }
            require(out.front() == guard && out[1 + block_values] == guard,
                    "low-bit decoder crossed output guards");
        }
        require(checked == 63488 * block_values, "low-bit finite half coverage count differs");
    }
}

void check_zero_blocks() {
    for (uint32_t type : {quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0}) {
        const auto* entry = quant::Registry::instance().get(type);
        entry->dequantize(nullptr, nullptr, 0);
    }
}

struct Format {
    uint32_t type;
    size_t block;
    size_t bytes;
    bool one_bit;
    const char* name;
};

constexpr Format formats[] = {
    {quant::GGML_TYPE_Q1_0, quant::Q1_0_BLOCK, quant::Q1_0_TYPESIZE, true, "Q1_0"},
    {quant::GGML_TYPE_Q2_0, quant::Q2_0_BLOCK, quant::Q2_0_TYPESIZE, false, "Q2_0"},
};

uint8_t* block_at(const Format& f, std::vector<uint8_t>& raw, size_t row, size_t width, size_t i) {
    return raw.data() + row * (width / f.block) * f.bytes + (i / f.block) * f.bytes;
}

const uint8_t* block_at(const Format& f, const std::vector<uint8_t>& raw, size_t row, size_t width, size_t i) {
    return raw.data() + row * (width / f.block) * f.bytes + (i / f.block) * f.bytes;
}

void make_weights(const Format& f, std::vector<uint8_t>& raw, size_t rows, size_t width) {
    constexpr uint16_t scales[] = {0x3c00, 0x3800, 0xbc00, 0x0001, 0x8000};
    for (size_t r = 0; r < rows; ++r) for (size_t b = 0; b < width / f.block; ++b) {
        uint8_t* p = raw.data() + (r * (width / f.block) + b) * f.bytes;
        p[0] = uint8_t(scales[(r + b) % 5]);
        p[1] = uint8_t(scales[(r + b) % 5] >> 8);
        for (size_t i = 0; i < f.block; ++i) {
            const unsigned code = unsigned((i + r + 2 * b) % (f.one_bit ? 2 : 4));
            if (f.one_bit) {
                const uint8_t mask = uint8_t(1u << (i % 8));
                p[2 + i / 8] = uint8_t((p[2 + i / 8] & uint8_t(~mask)) | ((code & 1u) << (i % 8)));
            } else {
                const unsigned shift = 2 * (i % 4);
                const uint8_t mask = uint8_t(3u << shift);
                p[2 + i / 4] = uint8_t((p[2 + i / 4] & uint8_t(~mask)) | (code << shift));
            }
        }
    }
}

double weight_value(const Format& f, const uint8_t* p, size_t i) {
    i %= f.block;
    const double d = half_value(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
    if (f.one_bit) {
        const int code = (p[2 + i / 8] >> (i % 8)) & 1u ? 1 : -1;
        return double(code) * d;
    }
    const unsigned code = (p[2 + i / 4] >> (2 * (i % 4))) & 3u;
    return double(int(code) - 1) * d;
}

double dot(const Format& f, const std::vector<uint8_t>& raw, size_t row, const float* x, size_t width) {
    double sum = 0;
    for (size_t i = 0; i < width; ++i) sum += weight_value(f, block_at(f, raw, row, width, i), i) * double(x[i]);
    return sum;
}

float bf16_oracle(float value) {
    uint32_t word = bits(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) word = (word >> 16) | 0x40u;
    else word = (word + 0x7fffu + ((word >> 16) & 1u)) >> 16;
    const uint32_t rounded = word << 16;
    float out;
    std::memcpy(&out, &rounded, sizeof out);
    return out;
}

bool close(float got, double want) {
    return std::abs(double(got) - want) <= 2e-5 * std::max(1.0, std::abs(want));
}

void check_products(backend::Backend& device, const Format& f, backend::Dtype dtype, size_t columns, size_t blocks = 8, size_t rows = 5, std::array<std::vector<float>, 3>* captured = nullptr) {
    const size_t width = blocks * f.block;
    std::vector<uint8_t> raw(rows * (width / f.block) * f.bytes);
    make_weights(f, raw, rows, width);
    auto weights = device.adopt(raw.data(), raw.size());
    const backend::WeightView w{f.type, {weights.get(), 0}};
    std::vector<float> input(columns * width);
    for (size_t c = 0; c < columns; ++c)
        for (size_t i = 0; i < width; ++i)
            input[c * width + i] = float(int((3 * i + 5 * c) % 9) - 4) * .137f + float((i + c) % 3) * .00091f;
    std::vector<float> oracle_input = input;
    if (dtype == backend::Dtype::bf16 && device.emulates_dtype(dtype))
        for (float& value : oracle_input) value = bf16_oracle(value);
    bool integer = false;
#if LLMX_HAS_BACKEND_VULKAN
    if (!device.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(device);
        integer = profile.prefer_integer_dot;
    }
#endif
    if (integer) oracle_input = testq::row_activations(input);
    const char* expected_path = dtype == backend::Dtype::bf16 && device.emulates_dtype(dtype) ? "bf16"
                              : integer ? "block-int16" : "f32";
    auto check_path = [&] {
        const auto paths = testq::take_matrix_paths(device);
        require(paths == std::vector<std::string>{expected_path}, "low-bit product arithmetic witness differs");
    };
    auto x = device.adopt(input.data(), input.size() * sizeof(float));
    std::vector<float> output(columns * rows + 2, guard);
    auto y = device.adopt(output.data(), output.size() * sizeof(float));
    (void)testq::take_matrix_paths(device);
    device.matmul(w, {x.get(), 0}, {y.get(), 1}, width, rows, columns, {}, dtype);
    check_path();
    device.read(*y, 0, output.data(), output.size() * sizeof(float));
    if (captured) (*captured)[0] = output;
    require(output.front() == guard && output.back() == guard, "low-bit dense output crossed guards");
    for (size_t c = 0; c < columns; ++c)
        for (size_t r = 0; r < rows; ++r)
            require(close(output[1 + c * rows + r], dot(f, raw, r, oracle_input.data() + c * width, width)),
                    "low-bit dense double-dot differs");

    std::fill(output.begin(), output.end(), guard);
    for (size_t i = 1; i + 1 < output.size(); ++i) output[i] = .25f;
    device.write(*y, 0, output.data(), output.size() * sizeof(float));
    (void)testq::take_matrix_paths(device);
    device.matmul_logits(w, {x.get(), 0}, {y.get(), 1}, width, rows, columns, {}, dtype);
    check_path();
    device.read(*y, 0, output.data(), output.size() * sizeof(float));
    if (captured) (*captured)[1] = output;
    require(output.front() == guard && output.back() == guard, "low-bit logits output crossed guards");
    for (size_t c = 0; c < columns; ++c) for (size_t r = 0; r < rows; ++r)
        require(close(output[1 + c * rows + r], dot(f, raw, r, oracle_input.data() + c * width, width)),
                "low-bit logits double-dot differs");

    std::fill(output.begin(), output.end(), guard);
    for (size_t i = 1; i + 1 < output.size(); ++i) output[i] = .25f;
    device.write(*y, 0, output.data(), output.size() * sizeof(float));
    (void)testq::take_matrix_paths(device);
    device.matmul_add(w, {x.get(), 0}, {y.get(), 1}, width, rows, columns, {}, dtype);
    check_path();
    device.read(*y, 0, output.data(), output.size() * sizeof(float));
    if (captured) (*captured)[2] = output;
    require(output.front() == guard && output.back() == guard, "low-bit additive output crossed guards");
    for (size_t c = 0; c < columns; ++c) for (size_t r = 0; r < rows; ++r)
        require(close(output[1 + c * rows + r], dot(f, raw, r, oracle_input.data() + c * width, width) + .25),
                "low-bit additive double-dot differs");
}

#if LLMX_HAS_BACKEND_VULKAN
void check_q1_single_row_bits(backend::Backend& device, const Format& f, size_t rows, size_t blocks,
                              const std::array<std::vector<float>, 3>& combined) {
    require(f.one_bit && rows >= 2, "Q1 single-row control received an invalid row shape");
    const size_t width = blocks * f.block;
    const size_t row_bytes = (width / f.block) * f.bytes;
    std::vector<uint8_t> raw(rows * row_bytes);
    make_weights(f, raw, rows, width);
    std::vector<float> input(width);
    for (size_t i = 0; i < width; ++i)
        input[i] = float(int((3 * i) % 9) - 4) * .137f + float(i % 3) * .00091f;
    auto x = device.adopt(input.data(), input.size() * sizeof(float));
    for (size_t op = 0; op < 3; ++op) for (size_t r = 0; r < rows; ++r) {
        auto one_weights = device.adopt(raw.data() + r * row_bytes, row_bytes);
        const backend::WeightView one{f.type, {one_weights.get(), 0}};
        std::array<float, 3> output{guard, op == 2 ? .25f : guard, guard};
        auto y = device.adopt(output.data(), sizeof output);
        device.write(*y, 0, output.data(), sizeof output);
        (void)testq::take_matrix_paths(device);
        if (op == 0) device.matmul(one, {x.get(), 0}, {y.get(), 1}, width, 1, 1, {}, backend::Dtype::f16);
        else if (op == 1) device.matmul_logits(one, {x.get(), 0}, {y.get(), 1}, width, 1, 1, {}, backend::Dtype::f16);
        else device.matmul_add(one, {x.get(), 0}, {y.get(), 1}, width, 1, 1, {}, backend::Dtype::f16);
        require(testq::take_matrix_paths(device) == std::vector<std::string>{"block-int16"},
                "Q1 single-row control left the integer decode path");
        device.read(*y, 0, output.data(), sizeof output);
        require(output.front() == guard && output.back() == guard, "Q1 single-row control crossed guards");
        require(bits(output[1]) == bits(combined[op][1 + r]), "Q1 batched and single-row outputs differ bitwise");
    }
}

void check_q1_grouped_two_row_bits(backend::Backend& device, const Format& q1, const Format& q2, size_t blocks) {
    require(q1.one_bit && !q2.one_bit, "Q1 grouped two-row fixture received wrong formats");
    const size_t width = blocks * q1.block;
    constexpr size_t rows_a = 3, rows_b = 5, rows_q2 = 3;
    std::vector<uint8_t> raw_a(rows_a * (width / q1.block) * q1.bytes);
    std::vector<uint8_t> raw_b(rows_b * (width / q1.block) * q1.bytes);
    std::vector<uint8_t> raw_c(width / q1.block * q1.bytes);
    std::vector<uint8_t> raw_q2(rows_q2 * (width / q2.block) * q2.bytes);
    make_weights(q1, raw_a, rows_a, width);
    make_weights(q1, raw_b, rows_b, width);
    make_weights(q1, raw_c, 1, width);
    make_weights(q2, raw_q2, rows_q2, width);
    auto weights_a = device.adopt(raw_a.data(), raw_a.size());
    auto weights_b = device.adopt(raw_b.data(), raw_b.size());
    auto weights_c = device.adopt(raw_c.data(), raw_c.size());
    auto weights_q2 = device.adopt(raw_q2.data(), raw_q2.size());
    const backend::WeightView wa{q1.type, {weights_a.get(), 0}};
    const backend::WeightView wb{q1.type, {weights_b.get(), 0}};
    const backend::WeightView wc{q1.type, {weights_c.get(), 0}};
    const backend::WeightView wq2{q2.type, {weights_q2.get(), 0}};
    std::vector<float> input(width);
    for (size_t i = 0; i < width; ++i)
        input[i] = float(int((3 * i) % 9) - 4) * .137f + float(i % 3) * .00091f;
    const std::vector<float> oracle_input = testq::row_activations(input);
    auto x = device.adopt(input.data(), input.size() * sizeof(float));
    std::vector<float> grouped_a(rows_a + 2, guard), grouped_b(rows_b + 2, guard), grouped_q2(rows_q2 + 2, guard);
    auto ya = device.adopt(grouped_a.data(), grouped_a.size() * sizeof(float));
    auto yb = device.adopt(grouped_b.data(), grouped_b.size() * sizeof(float));
    auto yq2 = device.adopt(grouped_q2.data(), grouped_q2.size() * sizeof(float));

    (void)testq::take_matrix_paths(device);
    (void)backend::vulkan_kernel_times(device);
    device.matmul_group({{wa, {ya.get(), 1}, rows_a}, {wq2, {yq2.get(), 1}, rows_q2}, {wb, {yb.get(), 1}, rows_b}},
                        {x.get(), 0}, width, 1, {}, backend::Dtype::f16);
    device.read(*ya, 0, grouped_a.data(), grouped_a.size() * sizeof(float));
    device.read(*yb, 0, grouped_b.data(), grouped_b.size() * sizeof(float));
    device.read(*yq2, 0, grouped_q2.data(), grouped_q2.size() * sizeof(float));
    require(grouped_a.front() == guard && grouped_a.back() == guard &&
            grouped_b.front() == guard && grouped_b.back() == guard &&
            grouped_q2.front() == guard && grouped_q2.back() == guard,
            "Q1 grouped two-row output crossed guards");
    require(testq::take_matrix_paths(device) == std::vector<std::string>{"block-int16"},
            "Q1 mixed grouped call left the block-int16 path");
    for (size_t r = 0; r < rows_a; ++r)
        require(close(grouped_a[1 + r], dot(q1, raw_a, r, oracle_input.data(), width)),
                "grouped Q1 projection differs from the existing double-dot oracle");
    for (size_t r = 0; r < rows_b; ++r)
        require(close(grouped_b[1 + r], dot(q1, raw_b, r, oracle_input.data(), width)),
                "grouped Q1 projection with an odd tail differs from the existing double-dot oracle");
    for (size_t r = 0; r < rows_q2; ++r)
        require(close(grouped_q2[1 + r], dot(q2, raw_q2, r, oracle_input.data(), width)),
                "mixed Q2 projection differs from the existing double-dot oracle");
    const auto grouped_times = backend::vulkan_kernel_times(device);
    if (backend::vulkan_timed_dispatches(device)) {
        const auto saw = [&](const char* name) {
            return std::any_of(grouped_times.begin(), grouped_times.end(), [&](const auto& entry) { return entry.first == name; });
        };
        require(saw("matmul_row_q1_0_dot_1col_2row"), "Q1 grouped projections missed the exact two-row kernel");
        require(saw("matmul_row_q2_0_dot_1col"), "mixed Q2 projection lost its existing one-row kernel");
    } else std::cout << "Q1 grouped two-row kernel timestamp unavailable\n";
    const std::vector<float> mixed_a = grouped_a;
    const std::vector<float> mixed_b = grouped_b;

    std::vector<float> grouped_c(3, guard);
    auto yc = device.adopt(grouped_c.data(), grouped_c.size() * sizeof(float));
    (void)testq::take_matrix_paths(device);
    (void)backend::vulkan_kernel_times(device);
    device.matmul_group({{wa, {ya.get(), 1}, rows_a}, {wb, {yb.get(), 1}, rows_b}, {wc, {yc.get(), 1}, 1}},
                        {x.get(), 0}, width, 1, {}, backend::Dtype::f16);
    device.read(*ya, 0, grouped_a.data(), grouped_a.size() * sizeof(float));
    device.read(*yb, 0, grouped_b.data(), grouped_b.size() * sizeof(float));
    device.read(*yc, 0, grouped_c.data(), grouped_c.size() * sizeof(float));
    require(grouped_a.front() == guard && grouped_a.back() == guard &&
            grouped_b.front() == guard && grouped_b.back() == guard &&
            grouped_c.front() == guard && grouped_c.back() == guard,
            "homogeneous Q1 grouped output crossed guards");
    require(testq::take_matrix_paths(device) == std::vector<std::string>{"block-int16"},
            "homogeneous Q1 grouped call left the block-int16 path");
    for (size_t r = 0; r < rows_a; ++r)
        require(close(grouped_a[1 + r], dot(q1, raw_a, r, oracle_input.data(), width)),
                "homogeneous grouped Q1 projection differs from the existing double-dot oracle");
    for (size_t r = 0; r < rows_b; ++r)
        require(close(grouped_b[1 + r], dot(q1, raw_b, r, oracle_input.data(), width)),
                "homogeneous grouped Q1 projection with an odd tail differs from the existing double-dot oracle");
    require(close(grouped_c[1], dot(q1, raw_c, 0, oracle_input.data(), width)),
            "single-row member of homogeneous Q1 group differs from the existing double-dot oracle");
    const auto homogeneous_times = backend::vulkan_kernel_times(device);
    if (backend::vulkan_timed_dispatches(device)) {
        const bool two_row = std::any_of(homogeneous_times.begin(), homogeneous_times.end(), [](const auto& entry) {
            return entry.first == "matmul_row_q1_0_dot_1col_2row";
        });
        const bool one_row = std::any_of(homogeneous_times.begin(), homogeneous_times.end(), [](const auto& entry) {
            return entry.first == "matmul_row_q1_0_dot_1col";
        });
        require(two_row && !one_row, "homogeneous three-projection Q1 group missed variant 7");
    }

    const auto compare_alone = [&](const backend::WeightView& weight, size_t rows,
                                   const std::vector<float>& grouped, const std::vector<float>* second,
                                   const char* expected_kernel) {
        std::vector<float> alone(rows + 2, guard);
        auto y = device.adopt(alone.data(), alone.size() * sizeof(float));
        (void)testq::take_matrix_paths(device);
        (void)backend::vulkan_kernel_times(device);
        device.matmul(weight, {x.get(), 0}, {y.get(), 1}, width, rows, 1, {}, backend::Dtype::f16);
        device.read(*y, 0, alone.data(), alone.size() * sizeof(float));
        require(alone.front() == guard && alone.back() == guard, "isolated projection output crossed guards");
        require(testq::take_matrix_paths(device) == std::vector<std::string>{"block-int16"},
                "Q1 isolated two-row call left block-int16");
        const auto alone_times = backend::vulkan_kernel_times(device);
        if (backend::vulkan_timed_dispatches(device))
            require(std::any_of(alone_times.begin(), alone_times.end(), [&](const auto& entry) {
                        return entry.first == expected_kernel;
                    }), "isolated projection missed its expected row kernel");
        for (size_t row = 0; row < rows; ++row)
            require(bits(grouped[row + 1]) == bits(alone[row + 1]),
                    "grouped projection differs bitwise from that same projection run alone");
        if (second)
            for (size_t row = 0; row < rows; ++row)
                require(bits((*second)[row + 1]) == bits(alone[row + 1]),
                        "mixed grouped projection differs bitwise from that same projection run alone");
    };
    compare_alone(wa, rows_a, grouped_a, &mixed_a, "matmul_row_q1_0_dot_1col_2row");
    compare_alone(wb, rows_b, grouped_b, &mixed_b, "matmul_row_q1_0_dot_1col_2row");
    compare_alone(wq2, rows_q2, grouped_q2, nullptr, "matmul_row_q2_0_dot_1col");
    compare_alone(wc, 1, grouped_c, nullptr, "matmul_row_q1_0_dot_1col");

    // Compare batched grouped Q1 outputs with each column's variant-7 result.
    constexpr size_t batch = 3;
    std::vector<float> batch_input(batch * width);
    for (size_t c = 0; c < batch; ++c)
        for (size_t i = 0; i < width; ++i)
            batch_input[c * width + i] = input[i] + float(c) * 0.00390625f;
    auto batch_x = device.adopt(batch_input.data(), batch_input.size() * sizeof(float));
    std::vector<float> batch_a(batch * rows_a + 2, guard), batch_b(batch * rows_b + 2, guard), batch_c(batch + 2, guard);
    auto batch_ya = device.adopt(batch_a.data(), batch_a.size() * sizeof(float));
    auto batch_yb = device.adopt(batch_b.data(), batch_b.size() * sizeof(float));
    auto batch_yc = device.adopt(batch_c.data(), batch_c.size() * sizeof(float));
    device.matmul_group({{wa, {batch_ya.get(), 1}, rows_a}, {wb, {batch_yb.get(), 1}, rows_b},
                         {wc, {batch_yc.get(), 1}, 1}},
                        {batch_x.get(), 0}, width, batch, {}, backend::Dtype::f16);
    device.read(*batch_ya, 0, batch_a.data(), batch_a.size() * sizeof(float));
    device.read(*batch_yb, 0, batch_b.data(), batch_b.size() * sizeof(float));
    device.read(*batch_yc, 0, batch_c.data(), batch_c.size() * sizeof(float));
    require(batch_a.front() == guard && batch_a.back() == guard && batch_b.front() == guard &&
            batch_b.back() == guard && batch_c.front() == guard && batch_c.back() == guard,
            "batched Q1 variant-7 control crossed output guards");
    for (size_t c = 0; c < batch; ++c) {
        std::vector<float> one_a(rows_a + 2, guard), one_b(rows_b + 2, guard), one_c(3, guard);
        auto one_ya = device.adopt(one_a.data(), one_a.size() * sizeof(float));
        auto one_yb = device.adopt(one_b.data(), one_b.size() * sizeof(float));
        auto one_yc = device.adopt(one_c.data(), one_c.size() * sizeof(float));
        device.matmul_group({{wa, {one_ya.get(), 1}, rows_a}, {wb, {one_yb.get(), 1}, rows_b},
                             {wc, {one_yc.get(), 1}, 1}},
                            {batch_x.get(), c * width}, width, 1, {}, backend::Dtype::f16);
        device.read(*one_ya, 0, one_a.data(), one_a.size() * sizeof(float));
        device.read(*one_yb, 0, one_b.data(), one_b.size() * sizeof(float));
        device.read(*one_yc, 0, one_c.data(), one_c.size() * sizeof(float));
        require(std::memcmp(batch_a.data() + 1 + c * rows_a, one_a.data() + 1, rows_a * sizeof(float)) == 0 &&
                std::memcmp(batch_b.data() + 1 + c * rows_b, one_b.data() + 1, rows_b * sizeof(float)) == 0 &&
                bits(batch_c[1 + c]) == bits(one_c[1]),
                "batched Q1 grouped column differs from its one-column variant-7 result");
    }
}

void check_integer_witness(backend::Backend& device, const Format& f, bool mixed, bool tile, const char* operation, bool ungrouped = false, bool narrow = false, bool two_rows = false) {
    const std::vector<std::string> expected = mixed ? std::vector<std::string>{"f32", "block-int16"}
                                                   : std::vector<std::string>{"block-int16"};
    require(testq::take_matrix_paths(device) == expected, "low-bit integer matrix-path witness differs");
    const auto times = backend::vulkan_kernel_times(device);
    const std::string kernel = std::string(tile ? "matmul_tile_" : "matmul_row_") + (f.one_bit ? "q1_0_dot" : "q2_0_dot") +
                               (narrow ? "_tall_narrow" : two_rows ? "_1col_2row" : ungrouped ? "_1col" : "");
    const bool executed = std::any_of(times.begin(), times.end(), [&](const auto& entry) {
        return two_rows ? entry.first == kernel : entry.first.find(kernel) == 0;
    });
    if (backend::vulkan_timed_dispatches(device)) {
        require(executed, "low-bit call did not execute its expected integer row/tile kernel");
        for (const auto& entry : times)
            if (entry.first.find(kernel) == 0)
                std::cout << f.name << " " << operation << " block-int16 witness: " << entry.first << '\n';
    } else std::cout << f.name << " " << operation << " block-int16 kernel timestamp unavailable\n";
}
#endif

void check_grouped_routed(backend::Backend& device, const Format& f, size_t columns = 3,
                          size_t rows = 5, backend::Dtype dtype = backend::Dtype::f32, size_t extent = 0, bool all_experts = false) {
    const size_t width = (all_experts ? 16 : 3) * f.block;
    const size_t logical_extent = extent ? extent : columns;
    const size_t experts = all_experts ? 4 : 3, selected = all_experts ? 4 : 2;
    std::vector<uint8_t> raw(experts * rows * (width / f.block) * f.bytes);
    make_weights(f, raw, experts * rows, width);
    auto weights = device.adopt(raw.data(), raw.size());
    const backend::WeightView w{f.type, {weights.get(), 0}};
    std::vector<float> input(columns * width);
    for (size_t c = 0; c < columns; ++c) for (size_t i = 0; i < width; ++i)
        input[c * width + i] = float(int((i + 2 * c) % 7) - 3) * .125f;
    bool integer = false;
#if LLMX_HAS_BACKEND_VULKAN
    if (!device.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(device);
        integer = profile.prefer_integer_dot;
    }
#endif
    const auto oracle_input = integer ? testq::row_activations(input) : input;
    auto begin = [&] {
        if (!integer) return;
        (void)testq::take_matrix_paths(device);
#if LLMX_HAS_BACKEND_VULKAN
        (void)backend::vulkan_kernel_times(device);
#endif
    };
    auto witness = [&](bool mixed, const char* operation) {
        if (!integer) return;
#if LLMX_HAS_BACKEND_VULKAN
        const auto profile = backend::vulkan_device_profile(device);
        const size_t from = mixed ? backend::tile_from_for(profile, false, width) : profile.moe_tile_from_q4;
        const bool tile = logical_extent >= from;
        const bool ungrouped = !mixed && !tile && columns * selected < 2 * experts;
        const size_t entries = columns * selected;
        const bool narrow = !mixed && tile && entries <= 16 * std::min(experts, entries);
        check_integer_witness(device, f, mixed, tile, operation, ungrouped, narrow);
#else
        (void)mixed; (void)operation;
#endif
    };
    auto x = device.adopt(input.data(), input.size() * sizeof(float));
    std::vector<float> out(columns * rows + 2, guard), second(columns * 2 + 2, guard);
    std::fill(out.begin() + 1, out.end() - 1, 0.0f);
    auto y = device.adopt(out.data(), out.size() * sizeof(float));
    auto y2 = device.adopt(second.data(), second.size() * sizeof(float));
    std::vector<float> ones(width * 2, 1.0f);
    auto fw = device.adopt(ones.data(), ones.size() * sizeof(float));
    const backend::RowRun run{columns, logical_extent};
    begin();
    device.matmul_group({{w, {y.get(), 1}, rows}, {{0, {fw.get(), 0}}, {y2.get(), 1}, 2}},
                        {x.get(), 0}, width, columns, {&run, 1}, dtype);
    device.read(*y, 0, out.data(), out.size() * sizeof(float));
    witness(true, "mixed grouped");
    require(out.front() == guard && out.back() == guard, "low-bit grouped output crossed guards");
    for (size_t c = 0; c < columns; ++c) for (size_t r = 0; r < rows; ++r)
        require(close(out[1 + c * rows + r], dot(f, raw, r, oracle_input.data() + c * width, width)),
                "low-bit grouped double-dot differs");
    device.read(*y2, 0, second.data(), second.size() * sizeof(float));
    require(second.front() == guard && second.back() == guard, "low-bit grouped F32 output crossed guards");
    for (size_t c = 0; c < columns; ++c) {
        double want = 0;
        for (size_t i = 0; i < width; ++i) want += input[c * width + i];
        require(close(second[1 + c * 2], want) && close(second[2 + c * 2], want),
                "low-bit grouped F32 double-dot differs");
    }

    constexpr uint32_t id_pattern[] = {0, 2, 1, 0, 2, 1};
    constexpr float probability_pattern[] = {.25f, .75f, .5f, .5f, .75f, .25f};
    constexpr uint32_t all_ids[] = {3, 1, 0, 2};
    constexpr float all_probabilities[] = {.125f, .25f, .5f, .125f};
    std::vector<uint32_t> ids(columns * selected);
    std::vector<float> probs(columns * selected);
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = all_experts ? all_ids[i % 4] : id_pattern[i % 6];
        probs[i] = all_experts ? all_probabilities[i % 4] : probability_pattern[i % 6];
    }
    auto choices = device.adopt(ids.data(), ids.size() * sizeof(uint32_t));
    auto routing = device.adopt(probs.data(), probs.size() * sizeof(float));
    out.assign(columns * selected * rows + 2, guard);
    auto yr = device.adopt(out.data(), out.size() * sizeof(float));
    begin();
    device.matmul_experts({{w, {yr.get(), 1}, rows}}, {x.get(), 0}, width, columns,
        {{choices.get(), 0}, {routing.get(), 0}, selected, experts}, {&run, 1}, dtype);
    device.read(*yr, 0, out.data(), out.size() * sizeof(float));
    witness(false, "routed");
    require(out.front() == guard && out.back() == guard, "low-bit routed output crossed guards");
    for (size_t c = 0; c < columns; ++c) for (size_t s = 0; s < selected; ++s) for (size_t r = 0; r < rows; ++r) {
        const double want = dot(f, raw, ids[c * selected + s] * rows + r, oracle_input.data() + c * width, width);
        require(close(out[1 + (c * selected + s) * rows + r], want), "low-bit routed double-dot differs");
    }

    std::vector<float> expanded(columns * selected * width);
    for (size_t c = 0; c < columns; ++c) for (size_t s = 0; s < selected; ++s)
        std::copy_n(input.data() + c * width, width, expanded.data() + (c * selected + s) * width);
    auto xr = device.adopt(expanded.data(), expanded.size() * sizeof(float));
    out.assign(columns * rows + 2, guard);
    std::fill(out.begin() + 1, out.end() - 1, .25f);
    auto ya = device.adopt(out.data(), out.size() * sizeof(float));
    begin();
    device.matmul_experts_add(w, {xr.get(), 0}, {ya.get(), 1}, width, rows, columns,
        {{choices.get(), 0}, {routing.get(), 0}, selected, experts}, {&run, 1}, dtype);
    device.read(*ya, 0, out.data(), out.size() * sizeof(float));
    witness(false, "routed residual");
    require(out.front() == guard && out.back() == guard, "low-bit routed additive output crossed guards");
    for (size_t c = 0; c < columns; ++c) for (size_t r = 0; r < rows; ++r) {
        double want = .25;
        for (size_t s = 0; s < selected; ++s)
            want += probs[c * selected + s] * dot(f, raw, ids[c * selected + s] * rows + r,
                oracle_input.data() + c * width, width);
        require(close(out[1 + c * rows + r], want), "low-bit routed additive double-dot differs");
    }
}

void check_one_hot(backend::Backend& device, const Format& f) {
    constexpr size_t width = 256, rows = 5;
    std::vector<uint8_t> raw(rows * (width / f.block) * f.bytes);
    make_weights(f, raw, rows, width);
    auto weights = device.adopt(raw.data(), raw.size());
    const backend::WeightView w{f.type, {weights.get(), 0}};
    std::vector<float> input(width * width, 0.0f);
    for (size_t i = 0; i < width; ++i) input[i * width + i] = 1.0f;
    auto x = device.adopt(input.data(), input.size() * sizeof(float));
    std::vector<float> out(width * rows + 2, guard);
    auto y = device.adopt(out.data(), out.size() * sizeof(float));
    device.matmul(w, {x.get(), 0}, {y.get(), 1}, width, rows, width);
    device.read(*y, 0, out.data(), out.size() * sizeof(float));
    require(out.front() == guard && out.back() == guard, "low-bit one-hot output crossed guards");
    for (size_t i = 0; i < width; ++i) for (size_t r = 0; r < rows; ++r)
        require(close(out[1 + i * rows + r], weight_value(f, block_at(f, raw, r, width, i), i)),
                "low-bit one-hot code/position differs");
}

void check_range_and_cancellation(backend::Backend& device, const Format& f) {
    const size_t width = 3 * f.block;
    constexpr size_t rows = 3;
    std::vector<uint8_t> raw(rows * 3 * f.bytes, 0);
    constexpr uint16_t scales[] = {0x7bff, 0xfbff, 0x0001, 0x8001, 0x8000};
    for (size_t r = 0; r < rows; ++r) for (size_t b = 0; b < 3; ++b) {
        uint8_t* p = raw.data() + (r * 3 + b) * f.bytes;
        p[0] = uint8_t(scales[(r + b) % 5]);
        p[1] = uint8_t(scales[(r + b) % 5] >> 8);
        for (size_t i = 0; i < f.block; ++i) {
            const unsigned code = unsigned((i / 2 + r + b) % (f.one_bit ? 2 : 4));
            if (f.one_bit) set_q1(p, i, code);
            else set_q2(p, i, code);
        }
    }
    auto weights = device.adopt(raw.data(), raw.size());
    const backend::WeightView w{f.type, {weights.get(), 0}};
    for (float magnitude : {1e-30f, 1e-8f, 1.0f, 1e8f, 1e28f}) {
        std::vector<float> input(width);
        for (size_t i = 0; i < width; ++i)
            input[i] = (i & 1) ? -magnitude : magnitude;
        auto x = device.adopt(input.data(), input.size() * sizeof(float));
        std::vector<float> output(rows + 2, guard);
        auto y = device.adopt(output.data(), output.size() * sizeof(float));
        device.matmul(w, {x.get(), 0}, {y.get(), 1}, width, rows, 1);
        device.read(*y, 0, output.data(), output.size() * sizeof(float));
        require(output.front() == guard && output.back() == guard,
                "low-bit range output crossed guards");
        for (size_t r = 0; r < rows; ++r) {
            const double want = dot(f, raw, r, input.data(), width);
            if (!std::isfinite(want) || !close(output[1 + r], want))
                std::cerr << f.name << " range: magnitude=" << magnitude << " row=" << r
                          << " observed=" << output[1 + r] << " expected=" << want << '\n';
            require(std::isfinite(want) && close(output[1 + r], want),
                    "low-bit range/cancellation double-dot differs");
        }
    }
}

void check_scale_after_range(backend::Backend& device, const Format& f) {
    // Summing four huge inputs first overflows, but these scaled products fit.
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001)}) {
        std::vector<uint8_t> raw(f.bytes, f.one_bit ? 0xff : 0xaa);
        raw[0] = uint8_t(h);
        raw[1] = uint8_t(h >> 8);
        auto weights = device.adopt(raw.data(), raw.size());
        for (float magnitude : {1e38f, -1e38f}) {
            std::vector<float> input(f.block, magnitude);
            auto x = device.adopt(input.data(), input.size() * sizeof(float));
            std::array<float, 3> out{guard, guard, guard};
            auto y = device.adopt(out.data(), sizeof out);
            device.matmul({f.type, {weights.get(), 0}}, {x.get(), 0}, {y.get(), 1}, f.block, 1, 1);
            device.read(*y, 0, out.data(), sizeof out);
            const double want = dot(f, raw, 0, input.data(), f.block);
            require(out.front() == guard && out.back() == guard && std::isfinite(out[1]) && close(out[1], want),
                    "low-bit scale-after-sum range repair differs");
        }
    }
}

void check_row_classes(backend::Backend& device, const Format& f) {
    constexpr size_t rows = 4;
    const size_t width = 3 * f.block;
    std::vector<uint8_t> raw(rows * 3 * f.bytes);
    make_weights(f, raw, rows, width);
    auto weights = device.adopt(raw.data(), raw.size());
    std::vector<float> input(rows * width);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = float(int((i * 13 + 7) % 29) - 14) * 0.03125f;
    auto x = device.adopt(input.data(), input.size() * sizeof(float));
    auto y = device.alloc(rows * rows * sizeof(float));
    std::vector<std::vector<float>> outputs;
    std::vector<backend::Dtype> dtypes = {backend::Dtype::f32, backend::Dtype::f16};
    const auto native = device.native_dtypes();
    if (std::find(native.begin(), native.end(), backend::Dtype::int8) != native.end())
        dtypes.push_back(backend::Dtype::int8);
    for (backend::Dtype dtype : dtypes) {
        outputs.clear();
        for (size_t extent : row_classes::kExtents) {
            const backend::RowRun run{rows, extent};
            device.matmul({f.type, {weights.get(), 0}}, {x.get(), 0}, {y.get(), 0},
                          width, rows, rows, {&run, 1}, dtype);
            outputs.push_back(row_classes::read(device, *y, rows * rows));
        }
        row_classes::same_within_classes(device, outputs,
            std::string(backend::dtype_name(dtype)) + " " + f.name + " low-bit matmul");
    }
}

void check_embeddings(backend::Backend& device, const Format& f) {
    constexpr size_t width = 256, rows = 5, count = 4;
    std::vector<uint8_t> raw(rows * (width / f.block) * f.bytes);
    make_weights(f, raw, rows, width);
    auto weights = device.adopt(raw.data(), raw.size());
    const backend::WeightView w{f.type, {weights.get(), 0}};
    const uint32_t host_ids[count] = {4, 0, 3, 1};
    auto out = device.alloc((count * width + 2) * sizeof(float), backend::Memory::host_visible);
    std::vector<float> values(count * width + 2, guard);
    device.write(*out, 0, values.data(), values.size() * sizeof(float));
    device.embed({out.get(), 1}, w, width, rows, host_ids, count);
    device.read(*out, 0, values.data(), values.size() * sizeof(float));
    require(values.front() == guard && values.back() == guard, "low-bit host embedding crossed guards");
    for (size_t r = 0; r < count; ++r) for (size_t i = 0; i < width; ++i) {
        const uint8_t* p = block_at(f, raw, host_ids[r], width, i);
        const float want = float(weight_value(f, p, i));
        require(bits(values[1 + r * width + i]) == bits(want), "low-bit host embedding scale/code differs");
    }

    std::array<float, count> packed_ids{};
    for (size_t i = 0; i < count; ++i) {
        const uint32_t id = i == count - 1 ? uint32_t(rows) : host_ids[i];
        std::memcpy(&packed_ids[i], &id, sizeof id);
    }
    auto ids = device.adopt(packed_ids.data(), sizeof packed_ids);
    std::fill(values.begin(), values.end(), guard);
    device.write(*out, 0, values.data(), values.size() * sizeof(float));
    device.embed_ids({out.get(), 1}, w, width, rows, {ids.get(), 0}, count);
    device.read(*out, 0, values.data(), values.size() * sizeof(float));
    require(values.front() == guard && values.back() == guard, "low-bit device-ID embedding crossed guards");
    for (size_t r = 0; r < count; ++r) for (size_t i = 0; i < width; ++i) {
        const uint32_t id = r == count - 1 ? uint32_t(rows) : host_ids[r];
        const float want = id == rows ? 0.0f : float(weight_value(f, block_at(f, raw, id, width, i), i));
        require(bits(values[1 + r * width + i]) == bits(want), "low-bit device-ID embedding differs");
    }
}

#if LLMX_HAS_BACKEND_VULKAN
void check_integer_range(backend::Backend& device, const Format& f, size_t extent,
                         backend::Dtype dtype, size_t rows = 1) {
    const size_t width = 3 * f.block;
    const backend::RowRun run{1, extent};
    const bool tile = extent >= backend::tile_from_for(backend::vulkan_device_profile(device), false, width);
    require(rows == 1 || (rows == 2 && f.one_bit && extent == 1),
            "integer range fixture only supports the Q1 two-row decode case");
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001)}) {
        std::vector<uint8_t> raw(rows * 3 * f.bytes, 0xaa);
        for (size_t r = 0; r < rows; ++r) for (size_t b = 0; b < 3; ++b) {
            const uint16_t row_scale = uint16_t(h ^ ((r & 1) ? 0x8000 : 0));
            const size_t at = r * 3 * f.bytes + b * f.bytes;
            raw[at] = uint8_t(row_scale);
            raw[at + 1] = uint8_t(row_scale >> 8);
        }
        auto weights = device.adopt(raw.data(), raw.size());
        for (float value : {1e38f, -1e38f, 1e-30f, -1e-30f}) {
            std::vector<float> input(width, 0.0f);
            input[0] = value;
            const auto reconstructed = testq::row_activations(input);
            auto x = device.adopt(input.data(), input.size() * sizeof(float));
            std::vector<float> result(rows + 2, guard);
            auto y = device.adopt(result.data(), result.size() * sizeof(float));
            (void)testq::take_matrix_paths(device);
            (void)backend::vulkan_kernel_times(device);
            device.matmul({f.type, {weights.get(), 0}}, {x.get(), 0}, {y.get(), 1}, width, rows, 1,
                          {&run, 1}, dtype);
            device.read(*y, 0, result.data(), result.size() * sizeof(float));
            check_integer_witness(device, f, false, tile, "finite range", false, false, rows == 2);
            for (size_t r = 0; r < rows; ++r) {
                const uint16_t row_scale = uint16_t(h ^ ((r & 1) ? 0x8000 : 0));
                const double weight = half_value(row_scale), want = dot(f, raw, r, reconstructed.data(), width);
                const double repair = std::fabs(double(value)) * 0x1p-22 +
                                      32767.0 * std::numeric_limits<float>::denorm_min();
                const double bound = std::fabs(weight) * repair + std::fabs(want) * 0x1p-21 +
                                     2.0 * std::numeric_limits<float>::denorm_min();
                require(std::isfinite(result[1 + r]) &&
                        std::fabs(double(result[1 + r]) - want) <= bound,
                        "low-bit integer row/tile finite-range repair differs");
            }
            require(result.front() == guard && result.back() == guard,
                    "low-bit integer row/tile finite-range repair crossed guards");

            if (extent == 1) {
                constexpr size_t columns = 3;
                std::vector<float> batch_input(columns * width, 0.0f);
                for (size_t c = 0; c < columns; ++c)
                    batch_input[c * width] = (c & 1) ? -value : value;
                auto batch_x = device.adopt(batch_input.data(), batch_input.size() * sizeof(float));
                std::vector<float> batch_result(columns * rows + 2, guard);
                auto batch_y = device.adopt(batch_result.data(), batch_result.size() * sizeof(float));
                const backend::RowRun batch_run{columns, 1};
                device.matmul({f.type, {weights.get(), 0}}, {batch_x.get(), 0}, {batch_y.get(), 1},
                              width, rows, columns, {&batch_run, 1}, dtype);
                device.read(*batch_y, 0, batch_result.data(), batch_result.size() * sizeof(float));
                require(batch_result.front() == guard && batch_result.back() == guard,
                        "batched low-bit range-repair output crossed guards");
                for (size_t c = 0; c < columns; ++c) {
                    std::vector<float> one_input(width, 0.0f);
                    one_input[0] = batch_input[c * width];
                    auto one_x = device.adopt(one_input.data(), one_input.size() * sizeof(float));
                    std::vector<float> one_result(rows + 2, guard);
                    auto one_y = device.adopt(one_result.data(), one_result.size() * sizeof(float));
                    device.matmul({f.type, {weights.get(), 0}}, {one_x.get(), 0}, {one_y.get(), 1},
                                  width, rows, 1, {&run, 1}, dtype);
                    device.read(*one_y, 0, one_result.data(), one_result.size() * sizeof(float));
                    require(one_result.front() == guard && one_result.back() == guard &&
                            std::memcmp(batch_result.data() + 1 + c * rows, one_result.data() + 1,
                                        rows * sizeof(float)) == 0,
                            "batched low-bit range-repair column differs from the same column alone");
                }
            }
        }
    }
}

void check_vulkan_paths(backend::Backend& device, const Format& f) {
    const auto profile = backend::vulkan_device_profile(device);
    const size_t tile_columns = std::max(size_t(64), backend::tile_from_for(profile, false, 256));
    const char* tile_name = f.type == quant::GGML_TYPE_Q1_0 ? "matmul_tile_q1_0" : "matmul_tile_q2_0";
    bool row = false, tile = false, timed = false;
    auto collect = [&] {
        const auto entries = backend::vulkan_kernel_times(device);
        timed |= backend::vulkan_timed_dispatches(device) != 0;
        for (const auto& entry : entries) {
            row |= entry.first.find("matmul_row") == 0;
            tile |= entry.first.find(tile_name) == 0;
        }
    };
    for (backend::Dtype dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
        check_products(device, f, dtype, 1, 3);
        collect();
        check_products(device, f, dtype, tile_columns, 3);
        collect();
        check_products(device, f, dtype, 1);
        collect();
        check_products(device, f, dtype, tile_columns);
        collect();
    }
    check_grouped_routed(device, f);
    collect();
    if (f.one_bit && profile.prefer_integer_dot) {
        for (size_t blocks : {size_t(3), size_t(16), size_t(48)})
        for (size_t rows : {size_t(1), size_t(2), size_t(3), size_t(5)}) {
            (void)backend::vulkan_kernel_times(device);
            std::array<std::vector<float>, 3> combined;
            check_products(device, f, backend::Dtype::f16, 1, blocks, rows, &combined);
            const auto entries = backend::vulkan_kernel_times(device);
            const std::string expected = std::string("matmul_row_q1_0_dot") +
                                         (rows == 1 ? "_1col" : "_1col_2row");
            if (backend::vulkan_timed_dispatches(device)) {
                const bool expected_seen = std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                    return entry.first == expected;
                });
                const std::string other = std::string("matmul_row_q1_0_dot") +
                                          (rows == 1 ? "_1col_2row" : "_1col");
                const bool other_seen = std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                    return entry.first == other;
                });
                require(expected_seen && !other_seen, "one-column Q1 row-count kernel witness differs");
                std::cout << f.name << " one-column dense row witness: " << expected << '\n';
            } else std::cout << f.name << " one-column dense row witness unavailable (no timestamps)\n";
            if (rows >= 2) check_q1_single_row_bits(device, f, rows, blocks, combined);
        }
        for (size_t blocks : {size_t(3), size_t(16), size_t(48)})
            check_q1_grouped_two_row_bits(device, f, formats[1], blocks);
        std::cout << f.name << " grouped Q1 two-row blocks 3/16/48, odd projections 3/5, mixed and homogeneous groups: exact dispatch and same-projection-alone bits [ok]\n";
        std::cout << f.name << " one-column blocks 3/16/48, row counts 1/2/3/5: baseline, variant 7, odd-tail dispatch [ok]\n";
    }
    if (profile.prefer_integer_dot) {
        const size_t prompt_columns = std::max({size_t(129), tile_columns + 1, profile.moe_tile_from_q4 + 1});
        for (size_t extent : {size_t(1), prompt_columns})
            for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
                check_grouped_routed(device, f, extent == 1 ? 3 : prompt_columns, 65, dtype, extent);
                check_integer_range(device, f, extent, dtype, f.one_bit && extent == 1 ? 2 : 1);
            }
        for (size_t extent : {size_t(1), prompt_columns})
            for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8})
                check_grouped_routed(device, f, 1, 65, dtype, extent, true);
        std::cout << f.name << " routed row/narrow tile: one token, four selected experts, width " << 16 * f.block << " [ok]\n";
        std::cout << f.name << " integer rows/tiles: grouped/routed/residual 65 rows, logical extents 1/" << prompt_columns
                  << ", zero/tiny-scale range repair [ok]\n";
    }
    if (timed) require(row && tile, "low-bit Vulkan row/tile kernels were not both witnessed for format");
    std::cout << f.name << " Vulkan dispatches: timestamps=" << (timed ? "available" : "unavailable")
              << (timed ? (row && tile ? ", row+tile witnessed\n" : ", row/tile witness missing\n") : "\n");
}
#endif

void check_backend(backend::Backend& device) {
    for (const Format& f : formats) {
        check_embeddings(device, f);
        check_grouped_routed(device, f);
        check_range_and_cancellation(device, f);
        check_scale_after_range(device, f);
        check_row_classes(device, f);
        for (backend::Dtype dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            check_products(device, f, dtype, 3);
            check_products(device, f, dtype, 1, 3);
        }
    }
}
}

int main(int argc, char** argv) {
#if LLMX_HAS_BACKEND_VULKAN
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
        try {
            auto device = backend::make_vulkan_backend(0, true);
            std::cout << "low-bits Vulkan device: " << backend::vulkan_device_name(*device) << '\n';
            for (const Format& f : formats) {
                check_embeddings(*device, f);
                check_range_and_cancellation(*device, f);
                check_scale_after_range(*device, f);
                check_row_classes(*device, f);
                check_vulkan_paths(*device, f);
            }
            std::cout << "low-bits Vulkan: Q1_0/Q2_0 embeddings, dense/logits/additive operators, dtype policies and row/tile witnesses [ok]\n";
            return 0;
        } catch (const backend::VulkanUnavailable& e) {
            std::cout << "low-bits Vulkan unavailable: " << e.what() << '\n';
            return 77;
        } catch (const std::exception& e) {
            std::cerr << "low-bits Vulkan: " << e.what() << '\n';
            return 1;
        }
    }
#else
    (void)argv;
#endif
    if (argc != 1) { std::cerr << "usage: llmx-low-bits-test [--vulkan]\n"; return 1; }
    try {
        check_positions();
        check_all_finite_scales();
        check_zero_blocks();
        backend::CpuBackend cpu;
        for (const Format& f : formats) check_one_hot(cpu, f);
        for (int threads : {1, 3}) {
            cpu.set_threads(threads);
            check_backend(cpu);
        }
        std::cout << "low-bits: independent Q1_0/Q2_0 blocks, finite scales, alignment and guards [ok]\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "low-bits: " << e.what() << '\n';
        return 1;
    }
}
