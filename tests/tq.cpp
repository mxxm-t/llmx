#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "quant/quant.hpp"

namespace {
constexpr float guard = 12345.0f;
constexpr unsigned pow3[] = {1, 3, 9, 27, 81};

struct Format {
    uint32_t id;
    size_t bytes;
    bool tq1;
    const char* name;
};
constexpr Format formats[] = {
    {quant::GGML_TYPE_TQ1_0, quant::TQ1_0_TYPESIZE, true, "TQ1_0"},
    {quant::GGML_TYPE_TQ2_0, quant::TQ2_0_TYPESIZE, false, "TQ2_0"},
};

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

unsigned tq1_code(uint8_t packed, unsigned plane) {
    // The independent form retains the low eight product bits, then divides the resulting 0..765 numerator by 256.
    const unsigned remainder = (unsigned(packed) * pow3[plane]) % 256;
    return (remainder * 3) / 256;
}

unsigned code_at(const Format& f, const uint8_t* block, size_t i) {
    if (!f.tq1) {
        const size_t half = i / 128, plane = (i % 128) / 32, lane = i % 32;
        return (block[half * 32 + lane] >> (plane * 2)) & 3;
    }
    if (i < 160) {
        const size_t plane = i / 32, lane = i % 32;
        return tq1_code(block[lane], unsigned(plane));
    }
    if (i < 240) {
        const size_t j = i - 160, plane = j / 16, lane = j % 16;
        return tq1_code(block[32 + lane], unsigned(plane));
    }
    const size_t j = i - 240, plane = j / 4, lane = j % 4;
    return tq1_code(block[48 + lane], unsigned(plane));
}

uint16_t scale_at(const Format& f, const uint8_t* block) {
    const size_t at = f.bytes - 2;
    return uint16_t(block[at] | (uint16_t(block[at + 1]) << 8));
}

float expected(const Format& f, const uint8_t* block, size_t i) {
    return float(int(code_at(f, block, i)) - 1) * half_value(scale_at(f, block));
}

void fill_packed(const Format& f, uint8_t* block, uint8_t seed) {
    for (size_t i = 0; i < f.bytes - 2; ++i)
        block[i] = uint8_t(seed + 73 * i + 19 * (i >> 2));
}

void decode(const Format& f, const uint8_t* src, float* dst, size_t nblocks) {
    if (f.tq1) quant::dequantize_row_tq1_0(src, dst, nblocks);
    else quant::dequantize_row_tq2_0(src, dst, nblocks);
}

void check_registry(const Format& f) {
    const auto* q = quant::Registry::instance().get(f.id);
    require(q && !q->quantize && q->dequantize && q->block_size == 256 && q->type_size == f.bytes,
            "TQ read-only registry entry differs");
    require(quant::row_bytes(f.id, 512, 2) == 4 * f.bytes, "TQ row sizing differs");
    q->dequantize(nullptr, nullptr, 0);
}

void check_all_packed_bytes(const Format& f) {
    std::vector<uint8_t> raw(f.bytes);
    std::array<float, 256> out{};
    const auto* q = quant::Registry::instance().get(f.id);
    for (unsigned packed = 0; packed < 256; ++packed) {
        std::fill(raw.begin(), raw.end() - 2, uint8_t(packed));
        raw[f.bytes - 2] = 0x00;
        raw[f.bytes - 1] = 0x3c;
        q->dequantize(raw.data(), out.data(), 1);
        for (size_t i = 0; i < out.size(); ++i)
            require(bits(out[i]) == bits(expected(f, raw.data(), i)), "TQ packed byte mapping differs");
    }
}

void check_finite_half_scales(const Format& f) {
    std::vector<uint8_t> raw(f.bytes);
    std::array<float, 256> out{};
    fill_packed(f, raw.data(), 91);
    const auto* q = quant::Registry::instance().get(f.id);
    size_t checked = 0;
    for (unsigned h = 0; h <= 0xffff; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        raw[f.bytes - 2] = uint8_t(h);
        raw[f.bytes - 1] = uint8_t(h >> 8);
        q->dequantize(raw.data(), out.data(), 1);
        for (size_t i = 0; i < out.size(); ++i) {
            require(bits(out[i]) == bits(expected(f, raw.data(), i)), "TQ finite-half scale differs");
            ++checked;
        }
    }
    require(checked == 63488 * 256, "TQ finite-half coverage count differs");
}

void check_unaligned_blocks(const Format& f) {
    std::vector<uint8_t> raw(1 + 3 * f.bytes + 1, 0xa5);
    std::vector<float> out(3 * 256 + 2, guard);
    const std::array<uint16_t, 3> scales = {0x8000, 0x0001, 0x3c00};
    for (size_t b = 0; b < scales.size(); ++b) {
        uint8_t* block = raw.data() + 1 + b * f.bytes;
        fill_packed(f, block, uint8_t(31 * b + 7));
        block[f.bytes - 2] = uint8_t(scales[b]);
        block[f.bytes - 1] = uint8_t(scales[b] >> 8);
    }
    const auto saved = raw;
    decode(f, raw.data() + 1, out.data() + 1, 3);
    require(raw == saved && out.front() == guard && out.back() == guard,
            "TQ unaligned decoder changed source or crossed guards");
    for (size_t b = 0; b < 3; ++b)
        for (size_t i = 0; i < 256; ++i)
            require(bits(out[1 + b * 256 + i]) == bits(expected(f, raw.data() + 1 + b * f.bytes, i)),
                    "TQ unaligned multi-block order differs");
}

void check_cpu_product(const Format& f, backend::CpuBackend& cpu) {
    constexpr size_t width = 256, rows = 2, columns = 2;
    std::vector<uint8_t> packed(rows * f.bytes);
    std::vector<float> weights(rows * width), input(columns * width);
    for (size_t r = 0; r < rows; ++r) {
        uint8_t* block = packed.data() + r * f.bytes;
        fill_packed(f, block, uint8_t(37 * r + 11));
        block[f.bytes - 2] = 0x00;
        block[f.bytes - 1] = 0x3c;
        for (size_t i = 0; i < width; ++i) weights[r * width + i] = expected(f, block, i);
    }
    for (size_t c = 0; c < columns; ++c)
        for (size_t i = 0; i < width; ++i)
            input[c * width + i] = float(int((i * 7 + c * 13) % 31) - 15) / 32.0f;

    auto w = cpu.adopt(packed.data(), packed.size());
    auto x = cpu.adopt(input.data(), input.size() * sizeof(float));
    auto y = cpu.alloc((columns * rows + 2) * sizeof(float), backend::Memory::host_visible);
    std::vector<float> initial(columns * rows + 2, guard);
    cpu.write(*y, 0, initial.data(), initial.size() * sizeof(float));
    cpu.matmul({f.id, {w.get(), 0}}, {x.get(), 0}, {y.get(), 1}, width, rows, columns,
               {}, backend::Dtype::f32);
    std::vector<float> got(initial.size());
    cpu.read(*y, 0, got.data(), got.size() * sizeof(float));
    require(got.front() == guard && got.back() == guard, "TQ CPU product crossed output guards");
    for (size_t c = 0; c < columns; ++c)
        for (size_t r = 0; r < rows; ++r) {
            double want = 0, magnitude = 0;
            for (size_t i = 0; i < width; ++i) {
                const double term = double(weights[r * width + i]) * input[c * width + i];
                want += term;
                magnitude += std::fabs(term);
            }
            const double error = std::fabs(double(got[1 + c * rows + r]) - want);
            require(error <= 2e-5 * magnitude + 1e-7, "TQ CPU product differs from decoded-weight oracle");
        }
}

void check_cpu_rows_and_embedding(const Format& f, backend::CpuBackend& cpu) {
    constexpr size_t width = 512, rows = 5;
    std::vector<uint8_t> packed(rows * 2 * f.bytes);
    std::vector<float> weights(rows * width);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t b = 0; b < 2; ++b) {
            uint8_t* block = packed.data() + (r * 2 + b) * f.bytes;
            fill_packed(f, block, uint8_t(41 * r + 17 * b + 3));
            block[f.bytes - 2] = 0x00;
            block[f.bytes - 1] = 0x3c;
            for (size_t i = 0; i < 256; ++i)
                weights[r * width + b * 256 + i] = expected(f, block, i);
        }
    }
    auto w = cpu.adopt(packed.data(), packed.size());

    const std::array<uint32_t, 3> ids = {4, 0, 3};
    auto embedded = cpu.alloc((ids.size() * width + 2) * sizeof(float), backend::Memory::host_visible);
    std::vector<float> initial(ids.size() * width + 2, guard);
    cpu.write(*embedded, 0, initial.data(), initial.size() * sizeof(float));
    cpu.embed({embedded.get(), 1}, {f.id, {w.get(), 0}}, width, rows, ids.data(), ids.size());
    std::vector<float> embedded_values(initial.size());
    cpu.read(*embedded, 0, embedded_values.data(), embedded_values.size() * sizeof(float));
    require(embedded_values.front() == guard && embedded_values.back() == guard,
            "TQ CPU embedding crossed output guards");
    for (size_t n = 0; n < ids.size(); ++n)
        for (size_t i = 0; i < width; ++i)
            require(bits(embedded_values[1 + n * width + i]) == bits(weights[ids[n] * width + i]),
                    "TQ CPU embedding differs from independent decoded rows");

    for (size_t columns : {size_t(1), size_t(3)}) {
        std::vector<float> input(columns * width);
        for (size_t c = 0; c < columns; ++c)
            for (size_t i = 0; i < width; ++i)
                input[c * width + i] = float(int((i * 11 + c * 17) % 37) - 18) / 64.0f;
        auto x = cpu.adopt(input.data(), input.size() * sizeof(float));
        auto output = cpu.alloc((columns * rows + 2) * sizeof(float), backend::Memory::host_visible);
        std::vector<float> output_initial(columns * rows + 2, guard);
        cpu.write(*output, 0, output_initial.data(), output_initial.size() * sizeof(float));
        cpu.matmul({f.id, {w.get(), 0}}, {x.get(), 0}, {output.get(), 1}, width, rows, columns,
                   {}, backend::Dtype::f32);
        std::vector<float> got(output_initial.size());
        cpu.read(*output, 0, got.data(), got.size() * sizeof(float));
        require(got.front() == guard && got.back() == guard, "TQ CPU row product crossed output guards");
        for (size_t c = 0; c < columns; ++c)
            for (size_t r = 0; r < rows; ++r) {
                double want = 0, magnitude = 0;
                for (size_t i = 0; i < width; ++i) {
                    const double term = double(weights[r * width + i]) * input[c * width + i];
                    want += term;
                    magnitude += std::fabs(term);
                }
                const double error = std::fabs(double(got[1 + c * rows + r]) - want);
                require(error <= 2e-5 * magnitude + 1e-7, "TQ CPU wide row product differs from decoded-weight oracle");
            }
    }
}

} // namespace

int main() {
    try {
        backend::CpuBackend cpu;
        cpu.set_threads(1);
        for (const Format& f : formats) {
            check_registry(f);
            require(cpu.supports_type(f.id), "CPU registry does not expose TQ type");
            check_all_packed_bytes(f);
            check_finite_half_scales(f);
            check_unaligned_blocks(f);
            check_cpu_product(f, cpu);
            check_cpu_rows_and_embedding(f, cpu);
            std::cout << f.name << " packed bytes, finite scales and CPU product passed\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
