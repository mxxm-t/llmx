#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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

// Independent fixed dictionary bytes, with provenance and hashes in docs/src/quant-iq2-xxs-grid.md and quant-iq2-xs-grid.md.
constexpr char grid_xxs[] =
"08080808080808082b080808080808081919080808080808082b0808080808082b2b0808080808081908190808080808081919080808080808082b08"
"080808082b082b0808080808082b2b08080808082b2b2b0808080808190808190808080808190819080808080808191908080808082b191908080808"
"19082b190808080808192b19080808080808082b080808082b08082b080808082b2b082b080808082b082b2b08080808190808081908080808190808"
"190808080808190819080808191919081908080808080819190808080819082b19080808082b192b19080808080808082b0808082b0808082b080808"
"2b082b082b0808082b08082b2b08080819080808081908080819080808190808080819080819080819082b080819080808192b080819080808080819"
"081908082b08081908190808082b08190819080808082b19081908081908082b081908080819082b081908080808192b0819080808192b2b08190808"
"08080808191908082b08080819190808082b08081919080808082b08191908082b19081919190808192b2b19191908080808082b191908081908192b"
"19190808192b08082b190808080819082b190808080808192b1908080819082b2b19080808192b2b2b19080808080808082b080819190808082b0808"
"082b0808082b080808191908082b0808082b2b08082b080819080819082b080808190819082b080808081919082b08082b081919082b0808082b082b"
"082b080808190808192b080808080819192b08082b0808082b2b0808081919082b2b0808190808080808190808190808080819080808190808081908"
"19082b0808081908080808190808190808082b19080819080819082b080819080808192b080819081919192b080819080808080819081908082b0808"
"1908190808082b081908190808081919190819082b2b1919190819080808082b1908190808192b082b081908191908192b0819080808080808191908"
"082b08080819190808082b080819190819192b0808191908192b0819081919080808082b08191908082b1908191919082b082b191919190808080808"
"2b1919082b1919082b19190819080808082b190808190808082b190808081908082b190808080819082b19081908082b082b190808080808192b1908"
"19190808192b190808082b2b192b1908190819192b2b19080808080808082b082b08080808082b082b2b080808082b080819081908082b0819082b19"
"08082b080808082b08082b082b08082b08082b08192b2b0819082b08082b081919082b08080808082b082b082b0808082b082b081908080808192b08"
"0819080808192b080808190808192b080808081908192b082b19191908192b080808080819192b081908081919192b0808192b1919192b080808192b"
"2b192b08082b0808082b2b0808082b08082b2b080819192b082b2b08081908192b2b2b08190808080808081908190808080808190808190808080819"
"082b19080808081919082b080808081908192b08080808190808081908080819082b0819080808192b1919190808081908082b19080808191908082b"
"080808190819082b080808190808192b08080819080808081908081908082b081908081919082b19190808190808082b190808191919082b19080819"
"190808082b080819080819082b080819082b08192b0808192b1919192b080819082b2b192b0808190808080808190819082b08080819081908082b08"
"081908190808082b08190819192b192b081908192b0819081919081908192b0819190819080808082b19081919080808082b081908190808082b0819"
"08081908082b081908080819082b081919190819082b081908080808192b0819082b1919192b081919082b19192b08192b08082b192b081919190819"
"2b2b08190808192b2b2b08190808080808081919082b0808080819191908190808081919192b19080808191908082b08080819190808082b08081919"
"082b082b0808191908190808190819192b0808191908191908192b2b190819191908192b2b0819190808192b081919192b08192b081919192b2b0808"
"19191919190808082b191919081919192b19191908080808082b191919081908082b1919192b1908082b191908192b19082b191908080819192b1919"
"082b08082b2b19190819080808082b190808190808082b190808081908082b19082b2b1908082b190808080819082b191919191919082b19082b1908"
"2b082b1908082b192b082b190808080808192b191919080808192b190808190819192b192b08190819192b190819082b19192b192b080819082b2b19"
"080808080808082b2b0808080808082b2b2b08080808082b190808190808082b2b08082b0808082b081908081908082b082b19081908082b08080819"
"1908082b190819082b08082b190808080819082b081908080819082b080819080819082b191919080819082b080808190819082b08082b190819082b"
"080808081919082b2b1908191919082b0819192b1919082b192b08082b19082b080808192b19082b08082b192b19082b2b080808082b082b08190808"
"192b082b190819082b2b082b081908080808192b080819080808192b08192b080808192b080808190808192b19082b2b0808192b2b1919081908192b"
"0808082b1908192b191908192b08192b080808080819192b2b082b080819192b081908190819192b190819191919192b1908082b082b192b08082b08"
"192b192b2b08080808082b2b0808191908082b2b1919082b08082b2b192b080819082b2b080808082b082b2b082b190808192b2b08081919082b2b2b"
"08190808192b2b2b";
constexpr char grid_xs[] =
"08080808080808082b080808080808081919080808080808082b0808080808082b2b080808080808190819080808080808191908080808082b191908"
"08080808192b19080808080808082b08080808082b082b080808080819192b0808080808082b2b080808080819080819080808080819081908080808"
"2b19081908080808192b08190808080808081919080808082b081919080808081919191908080808082b19190808080819082b190808080808192b19"
"080808080808082b080808082b08082b080808081919082b08080808082b082b080808081908192b080808080819192b08080808192b192b08080808"
"08082b2b08080808190808081908080808190808190808082b19080819080808192b08081908080808081908190808082b0819081908080819191908"
"19080808082b1908190808082b2b19081908080819082b081908080808192b081908080808080819190808082b080819190808081919081919080808"
"082b0819190808081908191919080808081919191908080808082b1919080808082b2b19190808081908082b190808080819082b190808080808192b"
"19080808080808082b0808082b0808082b080808191908082b080808082b08082b080808190819082b080808081919082b08080808082b082b080808"
"190808192b080808081908192b080808080819192b080808191919192b0808080808082b2b0808082b2b082b2b080808190808080819080808190808"
"081908082b19080808190808192b08080819080808081908081908082b081908081908081919190808190808082b19080819080819082b0808190808"
"08192b080819080808080819081908082b080819081908081919081908190808082b081908190808190819190819080808191919081908082b191919"
"0819080808082b19081908081908082b081908080819082b081908080808192b0819080808080808191908082b080808191908081919080819190808"
"082b0808191908081908190819190808081919081919080808082b081919080819080819191908080819081919190808080819191919080819082b19"
"191908080808082b19190808190808082b190808081908082b190808080819082b1908082b192b082b190808080808192b1908082b0808192b190808"
"0819082b2b19080808080808082b08082b080808082b080819190808082b0808082b0808082b08082b2b0808082b080819081908082b080808191908"
"082b080808082b08082b080819192b08082b080819080819082b080808190819082b080808081919082b0808082b1919082b08080808082b082b0808"
"08082b2b082b08082b2b2b2b082b080819080808192b080808190808192b080808081908192b080808080819192b08081908082b192b0808192b082b"
"192b0808080808082b2b080808082b082b2b0808082b2b082b2b08082b19192b2b2b080808082b2b2b2b080819080808080819080819080808081908"
"2b19080808081908192b08080808190808081908080819082b081908080819081919190808081908082b19080808190819082b080808190808192b08"
"0808190808080819080819082b080819080819081919081908081908082b0819080819081908191908081908081919190808190808082b1908081908"
"2b2b2b19080819081908082b080819080819082b080819080808192b0808190808080808190819082b080808190819081919080819081908082b0808"
"190819081908190819081908081919081908190808082b08190819081908081919081908081908191908190808081919190819080808082b19081908"
"0819192b190819082b19192b19081908190808082b081908081908082b0819082b1908082b081908080819082b081908080808192b08190808082b19"
"2b08190808080808081919082b080808081919081919080808191908082b0808081919081908190808191908081919080819190808082b0808191908"
"19080819081919080819081908191908192b081908191908080819190819190808192b19081919080808082b08191908190808081919190808190808"
"1919190808081908191919080808081919191908080808082b191908081919082b191908192b08192b19190819080808082b190808190808082b1908"
"08081908082b19082b081908082b190808080819082b190808191919082b19082b19082b082b190808080808192b190819190808192b19082b192b19"
"192b1908190819192b2b1908192b2b2b2b2b19080808080808082b082b08080808082b081919080808082b08082b080808082b082b2b080808082b08"
"1908190808082b080819190808082b0808082b0808082b081908081908082b080819081908082b080808191908082b080808082b08082b0808082b2b"
"08082b081908080819082b080819080819082b080808190819082b080808081919082b08082b081919082b0819192b1919082b08080808082b082b08"
"2b082b082b082b080808082b2b082b08082b2b2b2b082b081908080808192b080819080808192b080808190808192b08192b2b0808192b0808080819"
"08192b080808080819192b081908081919192b082b08191919192b08192b192b19192b08190808082b192b082b2b19082b192b082b192b2b2b192b08"
"08080808082b2b08082b0808082b2b082b2b0808082b2b0808082b08082b2b0819191919082b2b08082b082b082b2b082b082b2b082b2b08082b2b19"
"192b2b080808192b192b2b08082b08082b2b2b0808082b082b2b2b082b08082b2b2b2b08082b082b2b2b2b082b2b082b2b2b2b081908080808080819"
"08190808080808192b19080808080819192b08080808081908081908080808192b081908080808191919190808080819082b19080808081919082b08"
"0808081908192b080808081908080819080808192b080819080808191919081908080819082b0819080808192b2b0819080808191908191908080819"
"081919190808081908082b190808081919192b19080808191908082b080808190819082b080808190808192b0808081908080808190808192b080808"
"190808191919080819080819082b0808190808191908190819080819081919081908081908082b081908081919080819190808190819081919080819"
"08081919190808190808082b190808191919082b190808192b082b2b19080819190808082b080819081908082b080819080819082b0808192b081908"
"2b080819192b2b082b080819080808192b08081908080808081908192b080808081908191919080808190819082b0808081908191908190808190819"
"0819190808190819192b19080819081908082b08081908191908081908190819081908190819081908081919081908190808082b081908190819192b"
"0819081919080808191908190819080819190819080819081919081908192b081919081908080819191908192b2b192b19190819080808082b190819"
"2b2b08082b190819081908192b190819080819192b19081919080808082b081908190808082b081908081908082b081908080819082b081919190819"
"082b081908191919082b08192b082b19082b081908080808192b081919081908192b081908190819192b081908081919192b0819192b2b19192b0819"
"081908082b2b081908080808080819192b080808080819191919080808081919082b0808080819191908190808081919081919080808191908082b08"
"08081919082b2b08080819191908081908081919081908190808191908081919080819190808082b0808191919080808190819190819080819081919"
"0808190819081919191919081908191908080819190819192b08081919081919080808082b081919081908192b0819192b2b2b2b2b08191919080808"
"081919190819080808191919080819080819191919082b0808191919080808190819191908082b19081919191908082b0819191919082b2b08191919"
"0808080819191919082b0808191919190808082b19191919082b082b1919191919082b082b191919082b2b192b19191919082b2b2b19191908080808"
"082b191908191908082b191919080819082b191908081919082b1919192b192b082b19192b2b1908192b191908080819192b19192b080819192b1919"
"1919082b2b2b19191908080808082b190819080808082b190808190808082b190808081908082b190819191908082b192b082b1908082b192b19082b"
"08082b19192b2b2b08082b190808080819082b1908192b082b082b192b2b08192b082b192b08192b2b082b190808080808192b192b19190808192b19"
"0808190819192b190808081919192b191919081919192b1908192b2b19192b1919080808082b2b192b2b2b19082b2b1919192b08192b2b192b190808"
"2b2b2b19081919192b2b2b192b082b192b2b2b19080808080808082b2b0808080808082b191908080808082b082b08080808082b190819080808082b"
"081919080808082b08082b080808082b2b2b2b080808082b190808190808082b081908190808082b080819190808082b0808082b0808082b2b08082b"
"0808082b082b2b2b0808082b2b2b2b2b0808082b190808081908082b081908081908082b2b1908081908082b080819081908082b080808191908082b"
"190819191908082b192b19191908082b080808082b08082b08082b082b08082b0808082b2b08082b2b08082b2b08082b08082b2b2b08082b082b2b2b"
"2b08082b190808080819082b081908080819082b080819080819082b2b0819080819082b191919080819082b080808190819082b08082b190819082b"
"192b082b0819082b080808081919082b081908191919082b19192b2b1919082b082b19082b19082b2b2b2b192b19082b08080808082b082b082b0808"
"082b082b19192b08082b082b2b2b1919082b082b0808082b082b082b2b08082b082b082b082b2b2b082b082b2b190808192b082b2b082b082b2b082b"
"0808082b2b2b082b082b082b2b2b082b2b19192b2b2b082b082b2b2b2b2b082b190808080808192b081908080808192b080819080808192b08080819"
"0808192b2b1919190808192b0819082b0808192b080808081908192b2b082b081908192b08192b191908192b2b1919192b08192b192b082b2b08192b"
"080808080819192b191908080819192b081908190819192b080819190819192b082b19190819192b192b2b081919192b0808192b1919192b2b08192b"
"1919192b190808192b19192b19081919082b192b2b192b2b082b192b192b0819192b192b191919082b2b192b08082b192b2b192b0808080808082b2b"
"2b08080808082b2b082b080808082b2b2b2b080808082b2b08082b0808082b2b2b2b2b0808082b2b08082b2b08082b2b1908191919082b2b192b1919"
"19082b2b2b192b2b19082b2b080808082b082b2b2b0808082b082b2b082b08082b082b2b2b2b2b082b082b2b0808082b2b082b2b08082b2b2b082b2b"
"0808081908192b2b1919192b08192b2b19192b192b192b2b082b192b2b192b2b2b2b0808082b2b2b08082b08082b2b2b2b082b08082b2b2b082b2b08"
"082b2b2b08082b2b082b2b2b082b2b2b082b2b2b08190808192b2b2b0819082b192b2b2b2b19082b192b2b2b082b2b082b2b2b2b2b2b2b082b2b2b2b"
"1908192b2b2b2b2b2b2b2b2b2b2b2b2b";

namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

unsigned hex_digit(char value) {
    return value <= '9' ? unsigned(value - '0') : unsigned(value - 'a' + 10);
}

unsigned magnitude(const char* grid, unsigned index, unsigned lane) {
    const size_t byte = 2 * (8 * index + lane);
    return 16 * hex_digit(grid[byte]) + hex_digit(grid[byte + 1]);
}

unsigned even_parity_signs(unsigned code) {
    unsigned parity = 0;
    for (unsigned bit = 0; bit < 7; ++bit) parity += (code >> bit) & 1u;
    return code | ((parity & 1u) << 7);
}

double half_value(uint16_t bits) {
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned fraction = bits & 1023;
    const double value = exponent ? std::ldexp(double(1024 + fraction), int(exponent) - 25)
                                  : std::ldexp(double(fraction), -24);
    return bits & 0x8000 ? -value : value;
}

struct Format {
    uint32_t id;
    size_t block_bytes;
    unsigned index_count;
    const char* grid;
    bool xs;
    const char* name;
};

const Format formats[] = {
    {quant::GGML_TYPE_IQ2_XXS, 66, 256, grid_xxs, false, "IQ2_XXS"},
    {quant::GGML_TYPE_IQ2_XS, 74, 512, grid_xs, true, "IQ2_XS"},
};

void pack_block(const Format& format, uint8_t* dst, uint16_t scale, unsigned seed) {
    std::memset(dst, 0, format.block_bytes);
    dst[0] = uint8_t(scale);
    dst[1] = uint8_t(scale >> 8);
    for (unsigned group = 0; group < 8; ++group) {
        if (format.xs) {
            for (unsigned vector = 0; vector < 4; ++vector) {
                const unsigned ordinal = 4 * group + vector;
                const unsigned index = (seed + 17 * ordinal) % format.index_count;
                const unsigned code = (seed + 13 * ordinal) % 128;
                const unsigned scale_code = (seed + 3 * (2 * group + vector / 2)) % 16;
                const unsigned word = index | (code << 9);
                const size_t word_at = 2 + 8 * group + 2 * vector;
                dst[word_at] = uint8_t(word);
                dst[word_at + 1] = uint8_t(word >> 8);
                dst[66 + group] |= uint8_t(scale_code << (4 * (vector / 2)));
            }
        } else {
            uint32_t word = ((seed + 3 * group) % 16) << 28;
            for (unsigned vector = 0; vector < 4; ++vector) {
                const unsigned ordinal = 4 * group + vector;
                dst[2 + 8 * group + vector] = uint8_t((seed + 17 * ordinal) % format.index_count);
                word |= ((seed + 13 * ordinal) % 128) << (7 * vector);
            }
            for (unsigned byte = 0; byte < 4; ++byte) dst[6 + 8 * group + byte] = uint8_t(word >> (8 * byte));
        }
    }
}

float expected_value(const Format& format, const uint8_t* block,
                     unsigned group, unsigned vector, unsigned lane) {
    unsigned index, sign_code, scale_code;
    if (format.xs) {
        const size_t at = 2 + 8 * group + 2 * vector;
        const unsigned word = block[at] | (unsigned(block[at + 1]) << 8);
        index = word & 511;
        sign_code = word >> 9;
        scale_code = (block[66 + group] >> (4 * (vector / 2))) & 15;
    } else {
        const size_t at = 2 + 8 * group;
        const uint32_t word = block[at + 4] | (uint32_t(block[at + 5]) << 8) |
                              (uint32_t(block[at + 6]) << 16) | (uint32_t(block[at + 7]) << 24);
        index = block[at + vector];
        sign_code = (word >> (7 * vector)) & 127;
        scale_code = word >> 28;
    }
    const double scale = half_value(uint16_t(block[0] | (unsigned(block[1]) << 8))) *
                         (0.5 + scale_code) * 0.25;
    const double value = scale * magnitude(format.grid, index, lane);
    return float((even_parity_signs(sign_code) >> lane) & 1u ? -value : value);
}

std::vector<float> read(backend::Backend& backend, backend::Buffer& buffer, size_t count) {
    std::vector<float> values(count);
    backend.read(buffer, 0, values.data(), count * sizeof(float));
    return values;
}

void check_close(float actual, double expected, double magnitude, size_t count) {
    const double nu = double(2 * count + 8) * 0x1p-24;
    const double bound = nu / (1 - nu) * magnitude + count * std::numeric_limits<float>::denorm_min();
    require(std::isfinite(actual) && std::fabs(double(actual) - expected) <= bound,
            "F32 backend dot exceeds the existing quantized-dot rounding bound");
}

std::vector<std::string> take_matrix_paths(backend::Backend& backend) {
    backend::MatrixPaths paths;
    backend.swap_matrix_paths(paths);
    return paths.take();
}

bool integer_path(backend::Backend& backend, backend::Dtype dtype, size_t extent) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!backend.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(backend);
        return extent == 1 ? profile.lookup_integer_dot : profile.prefer_integer_dot;
    }
#else
    (void)backend;
    (void)dtype;
    (void)extent;
#endif
    return false;
}

float round_bf16(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    uint32_t high = word >> 16;
    const uint32_t low = word & 0xffff;
    if (low > 0x8000 || (low == 0x8000 && (high & 1))) ++high;
    word = high << 16;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::vector<float> round_bf16(const std::vector<float>& values) {
    std::vector<float> out(values);
    for (float& value : out) value = round_bf16(value);
    return out;
}

void check_decoder(const Format& format) {
    std::vector<uint8_t> source(1 + 3 * format.block_bytes + 1, 0xa5);
    std::vector<float> output(770, 12345.0f);
    for (unsigned block = 0; block < 3; ++block) {
        const uint16_t scale = block == 1 ? 0xb555 : 0x3555;
        pack_block(format, source.data() + 1 + block * format.block_bytes, scale, block * 37 + 19);
    }
    const auto before = source;
    if (format.xs) quant::dequantize_row_iq2_xs(source.data() + 1, output.data() + 1, 3);
    else quant::dequantize_row_iq2_xxs(source.data() + 1, output.data() + 1, 3);
    require(source == before, "decoder changed its packed input");
    require(output.front() == 12345.0f && output.back() == 12345.0f, "decoder crossed output guards");
    for (unsigned block = 0; block < 3; ++block)
        for (unsigned group = 0; group < 8; ++group)
            for (unsigned vector = 0; vector < 4; ++vector)
                for (unsigned lane = 0; lane < 8; ++lane) {
                    const size_t at = 1 + block * 256 + 32 * group + 8 * vector + lane;
                    const float expected = expected_value(format,
                        source.data() + 1 + block * format.block_bytes, group, vector, lane);
                    require(bits(output[at]) == bits(expected), "decoder differs from independent table oracle");
                }
}

void check_backend(backend::Backend& backend, const Format& format) {
    require(backend.supports_type(format.id), "backend does not support test format");
    constexpr size_t width = 512, rows = 2, columns = 2, count = rows * columns;
    std::vector<uint8_t> packed(rows * 2 * format.block_bytes);
    std::vector<float> weights(rows * width);
    for (size_t row = 0; row < rows; ++row)
        for (size_t block = 0; block < 2; ++block) {
            auto* data = packed.data() + (row * 2 + block) * format.block_bytes;
            pack_block(format, data, uint16_t(0x3555 + row * 31 + block), unsigned(row * 23 + block * 11));
            for (unsigned group = 0; group < 8; ++group)
                for (unsigned vector = 0; vector < 4; ++vector)
                    for (unsigned lane = 0; lane < 8; ++lane)
                        weights[row * width + block * 256 + 32 * group + 8 * vector + lane] =
                            expected_value(format, data, group, vector, lane);
        }
    auto weight_buffer = backend.adopt(packed.data(), packed.size());

    uint32_t ids[] = {1, 0, 2, UINT32_MAX};
    auto id_buffer = backend.adopt(ids, sizeof(ids));
    auto embedding = backend.alloc((4 * width + 2) * sizeof(float));
    std::vector<float> embedding_guard(4 * width + 2, 99.0f);
    backend.write(*embedding, 0, embedding_guard.data(), embedding_guard.size() * sizeof(float));
    backend.embed_ids({embedding.get(), 1}, {format.id, {weight_buffer.get(), 0}}, width, rows,
                      {id_buffer.get(), 0}, 4);
    auto actual = read(backend, *embedding, embedding_guard.size());
    require(actual.front() == 99.0f && actual.back() == 99.0f, "embedding crossed output guards");
    for (size_t row = 0; row < 4; ++row)
        for (size_t i = 0; i < width; ++i) {
            const float expected = row < 2 ? weights[ids[row] * width + i] : 0.0f;
            require(bits(actual[1 + row * width + i]) == bits(expected), "embedding differs from oracle");
        }

    std::vector<float> input(width * columns);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 29) - 14) / 23.0f;
    auto input_buffer = backend.adopt(input.data(), input.size() * sizeof(float));
    auto output = backend.alloc((count + 2) * sizeof(float));
    std::vector<float> output_guard(count + 2, 77.0f);
    const auto rounded_bf16 = round_bf16(input);
    const auto reconstructed = testq::row_activations(input);
    std::vector<float> f32_result;
    for (const size_t extent : {size_t(1), size_t(16384)}) {
        const backend::RowRun run{columns, extent};
        for (const auto dtype : {backend::Dtype::f32, backend::Dtype::f16,
                                 backend::Dtype::bf16, backend::Dtype::int8}) {
            const bool integer = integer_path(backend, dtype, extent);
            const auto& oracle_input = dtype == backend::Dtype::bf16 ? rounded_bf16
                                      : integer ? reconstructed : input;
            std::fill(output_guard.begin(), output_guard.end(), 77.0f);
            backend.write(*output, 0, output_guard.data(), output_guard.size() * sizeof(float));
            (void)take_matrix_paths(backend);
            backend.matmul({format.id, {weight_buffer.get(), 0}}, {input_buffer.get(), 0},
                           {output.get(), 1}, width, rows, columns, {&run, 1}, dtype);
            actual = read(backend, *output, output_guard.size());
            const auto paths = take_matrix_paths(backend);
            const std::string expected_path = dtype == backend::Dtype::bf16 ? "bf16"
                                            : integer ? "block-int16" : "f32";
            require(paths == std::vector<std::string>{expected_path},
                    "requested dtype did not select the expected matrix arithmetic path");
            require(actual.front() == 77.0f && actual.back() == 77.0f,
                    "matmul crossed output guards");
            if (dtype == backend::Dtype::f32) f32_result = actual;
            if (!integer && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8))
                for (size_t i = 0; i < actual.size(); ++i)
                    require(bits(actual[i]) == bits(f32_result[i]),
                            "F16/int8 fallback differs from exact F32-input product");
            for (size_t column = 0; column < columns; ++column)
                for (size_t row = 0; row < rows; ++row) {
                    const size_t index = column * rows + row;
                    double expected = 0.0, magnitude_sum = 0.0;
                    for (size_t i = 0; i < width; ++i) {
                        const double product = double(weights[row * width + i]) * oracle_input[column * width + i];
                        expected += product;
                        magnitude_sum += std::fabs(product);
                    }
                    check_close(actual[1 + index], expected, magnitude_sum, width);
                }
        }
    }
    const backend::RowRun run{columns, 16384};
    std::vector<double> expected_dots(count), dot_magnitudes(count);
    for (size_t column = 0; column < columns; ++column)
        for (size_t row = 0; row < rows; ++row)
            for (size_t i = 0; i < width; ++i) {
                const double product = double(weights[row * width + i]) * input[column * width + i];
                const size_t output_index = column * rows + row;
                expected_dots[output_index] += product;
                dot_magnitudes[output_index] += std::fabs(product);
            }
    auto grouped = backend.alloc((2 * count + 2) * sizeof(float));
    std::vector<float> grouped_guard(2 * count + 2, 0.0f);
    grouped_guard.front() = grouped_guard.back() = 81.0f;
    backend.write(*grouped, 0, grouped_guard.data(), grouped_guard.size() * sizeof(float));
    backend.matmul_group({{{format.id, {weight_buffer.get(), 0}}, {grouped.get(), 1}, rows},
                          {{format.id, {weight_buffer.get(), 0}}, {grouped.get(), count + 1}, rows}},
                         {input_buffer.get(), 0}, width, columns, {&run, 1}, backend::Dtype::f32);
    actual = read(backend, *grouped, grouped_guard.size());
    require(actual.front() == 81.0f && actual.back() == 81.0f, "grouped call crossed guards");
    for (size_t i = 0; i < count; ++i) {
        check_close(actual[i + 1], expected_dots[i], dot_magnitudes[i], width);
        check_close(actual[count + i + 1], expected_dots[i], dot_magnitudes[i], width);
    }

    auto residual = backend.alloc((count + 2) * sizeof(float));
    std::vector<float> residual_guard(count + 2, 0.25f);
    residual_guard.front() = residual_guard.back() = 81.0f;
    backend.write(*residual, 0, residual_guard.data(), residual_guard.size() * sizeof(float));
    backend.matmul_add({format.id, {weight_buffer.get(), 0}}, {input_buffer.get(), 0},
                       {residual.get(), 1}, width, rows, columns, {&run, 1}, backend::Dtype::f32);
    actual = read(backend, *residual, residual_guard.size());
    require(actual.front() == 81.0f && actual.back() == 81.0f, "residual call crossed guards");
    for (size_t i = 0; i < count; ++i)
        check_close(actual[i + 1], expected_dots[i] + 0.25, dot_magnitudes[i] + 0.25, width);

    uint32_t route_ids[] = {0, 0};
    float gains[] = {1.0f, 1.0f};
    auto route_buffer = backend.adopt(route_ids, sizeof(route_ids));
    auto gain_buffer = backend.adopt(gains, sizeof(gains));
    const backend::Backend::Routing routing{{route_buffer.get(), 0}, {gain_buffer.get(), 0}, 1, 1};
    auto routed = backend.alloc((count + 2) * sizeof(float));
    std::vector<float> routed_guard(count + 2, 0.0f);
    routed_guard.front() = routed_guard.back() = 83.0f;
    backend.write(*routed, 0, routed_guard.data(), routed_guard.size() * sizeof(float));
    backend.matmul_experts({{{format.id, {weight_buffer.get(), 0}}, {routed.get(), 1}, rows}},
                           {input_buffer.get(), 0}, width, columns, routing, {&run, 1}, backend::Dtype::f32);
    actual = read(backend, *routed, routed_guard.size());
    require(actual.front() == 83.0f && actual.back() == 83.0f, "routed call crossed guards");
    for (size_t i = 0; i < count; ++i)
        check_close(actual[i + 1], expected_dots[i], dot_magnitudes[i], width);

    std::fill(routed_guard.begin(), routed_guard.end(), 0.25f);
    routed_guard.front() = routed_guard.back() = 83.0f;
    backend.write(*routed, 0, routed_guard.data(), routed_guard.size() * sizeof(float));
    backend.matmul_experts_add({format.id, {weight_buffer.get(), 0}}, {input_buffer.get(), 0},
                               {routed.get(), 1}, width, rows, columns, routing, {&run, 1}, backend::Dtype::f32);
    actual = read(backend, *routed, routed_guard.size());
    require(actual.front() == 83.0f && actual.back() == 83.0f, "routed residual crossed guards");
    for (size_t i = 0; i < count; ++i)
        check_close(actual[i + 1], expected_dots[i] + 0.25, dot_magnitudes[i] + 0.25, width);

    std::cout << format.name << " " << (backend.is_cpu() ? "CPU" : "device")
              << ": embed_ids invalid rows, F32 matmul, grouped/residual/routed operators [ok]\n";
}

std::vector<uint8_t> range_block(const Format& format, uint16_t half_scale, unsigned max_index) {
    std::vector<uint8_t> block(format.block_bytes, 0);
    block[0] = uint8_t(half_scale);
    block[1] = uint8_t(half_scale >> 8);
    if (format.xs) {
        for (unsigned group = 0; group < 8; ++group) {
            block[66 + group] = 0xff;
            for (unsigned vector = 0; vector < 4; ++vector) {
                const unsigned word = max_index;
                const size_t at = 2 + 8 * group + 2 * vector;
                block[at] = uint8_t(word);
                block[at + 1] = uint8_t(word >> 8);
            }
        }
    } else {
        for (unsigned group = 0; group < 8; ++group) {
            const size_t at = 2 + 8 * group;
            std::fill(block.begin() + at, block.begin() + at + 4, uint8_t(max_index));
            const uint32_t word = 0xf0000000u;
            for (unsigned byte = 0; byte < 4; ++byte) block[at + 4 + byte] = uint8_t(word >> (8 * byte));
        }
    }
    return block;
}

size_t check_range(backend::Backend& backend, const Format& format) {
    constexpr size_t width = 256;
    const uint16_t half_scales[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400,
                                    0x3c00, 0xbc00, 0x7bff, 0xfbff};
    const float inputs[] = {0.0f, 0x1p-145f, 0x1p-120f, 0x1p-24f,
                            0x1p-14f, 1.0f, 65504.0f, 0x1p90f};
    unsigned max_index = 0;
    for (unsigned index = 1; index < format.index_count; ++index)
        if (magnitude(format.grid, index, 0) > magnitude(format.grid, max_index, 0)) max_index = index;

    size_t products = 0;
    for (const uint16_t half_scale : half_scales) {
        const auto packed = range_block(format, half_scale, max_index);
        const double weight = half_value(half_scale) * 3.875 * magnitude(format.grid, max_index, 0);
        auto weight_buffer = backend.adopt(packed.data(), packed.size());
        for (const float value : inputs)
            for (const float sign : {-1.0f, 1.0f}) {
                std::vector<float> input(width, 0.0f);
                input[0] = sign * value;
                const auto original_input = input;
                auto input_buffer = backend.adopt(input.data(), input.size() * sizeof(float));
                const auto reconstructed = testq::row_activations(input);
                const float initial[] = {19.0f, 0.0f, 19.0f};
                auto output = backend.adopt(initial, sizeof(initial));
                for (const size_t extent : {size_t(1), size_t(16384)})
                    for (const auto dtype : {backend::Dtype::f32, backend::Dtype::f16,
                                             backend::Dtype::int8}) {
                        const bool integer = integer_path(backend, dtype, extent);
                        const backend::RowRun run{1, extent};
                        (void)take_matrix_paths(backend);
                        backend.matmul({format.id, {weight_buffer.get(), 0}}, {input_buffer.get(), 0},
                                       {output.get(), 1}, width, 1, 1, {&run, 1}, dtype);
                        const auto actual = read(backend, *output, 3);
                        const auto paths = take_matrix_paths(backend);
                        const std::string expected_path = integer ? "block-int16" : "f32";
                        require(paths == std::vector<std::string>{expected_path},
                                "range probe selected an unexpected arithmetic path");
                        const double expected = weight * input[0];
                        const double repair = integer ? std::fabs(double(value)) * 0x1p-22 +
                            32767.0 * std::numeric_limits<float>::denorm_min() : 0.0;
                        const double bound = std::fabs(weight) * repair + std::fabs(expected) * 0x1p-21 +
                            2.0 * std::numeric_limits<float>::denorm_min();
                        require(actual.front() == 19.0f && actual.back() == 19.0f &&
                                std::isfinite(actual[1]) && std::fabs(double(actual[1]) - expected) <= bound,
                                "IQ2 finite-half or activation-scale range control failed");
                        ++products;
                    }
                require(read(backend, *input_buffer, width) == original_input,
                        "range probe changed its activation input");
            }
    }
    return products;
}

#if LLMX_HAS_BACKEND_VULKAN
bool has_timed_kernel(backend::Backend& device, const std::string& prefix) {
    const auto times = backend::vulkan_kernel_times(device);
    if (!backend::vulkan_timed_dispatches(device)) return false;
    return std::any_of(times.begin(), times.end(), [&](const auto& entry) {
        return entry.first.rfind(prefix, 0) == 0;
    });
}

size_t check_tile_threshold(backend::Backend& device, const Format& format) {
    if (device.is_cpu()) return 0;
    constexpr size_t width = 512, rows = 65;
    const auto profile = backend::vulkan_device_profile(device);
    const size_t threshold = backend::tile_from_for(profile, false, width);
    require(threshold > 1, "device tile threshold is too small for both edge probes");

    const size_t row_columns = threshold - 1;
    const size_t tile_columns = threshold + 1;
    const size_t packed_row_bytes = 2 * format.block_bytes;
    std::vector<uint8_t> packed(rows * packed_row_bytes);
    std::vector<float> weights(rows * width);
    for (size_t row = 0; row < rows; ++row)
        for (size_t block = 0; block < 2; ++block) {
            auto* data = packed.data() + row * packed_row_bytes + block * format.block_bytes;
            pack_block(format, data, uint16_t(0x3555 + row * 19 + block), unsigned(row * 13 + block * 7));
            for (unsigned group = 0; group < 8; ++group)
                for (unsigned vector = 0; vector < 4; ++vector)
                    for (unsigned lane = 0; lane < 8; ++lane)
                        weights[row * width + block * 256 + 32 * group + 8 * vector + lane] =
                            expected_value(format, data, group, vector, lane);
        }
    auto weight_buffer = device.adopt(packed.data(), packed.size());
    size_t checks = 0;
    bool saw_row_timestamp = false, saw_tile_timestamp = false;

    for (const size_t columns : {row_columns, tile_columns}) {
        std::vector<float> input(columns * width, 0.0f);
        std::vector<size_t> selected(columns);
        for (size_t column = 0; column < columns; ++column) {
            selected[column] = (column * 137 + 31) % width;
            input[column * width + selected[column]] = 1.0f;
        }
        auto input_buffer = device.adopt(input.data(), input.size() * sizeof(float));
        auto output = device.alloc((rows * columns + 2) * sizeof(float));
        std::vector<float> guard(rows * columns + 2, 77.0f);
        const backend::RowRun run{columns, columns};

        const std::vector<backend::Dtype> dtypes = columns == row_columns
            ? std::vector<backend::Dtype>{backend::Dtype::f32}
            : std::vector<backend::Dtype>{backend::Dtype::f32, backend::Dtype::f16,
                                          backend::Dtype::bf16, backend::Dtype::int8};
        for (const auto dtype : dtypes) {
            const bool integer = integer_path(device, dtype, columns);
            const auto rounded = round_bf16(input);
            const auto reconstructed = testq::row_activations(input);
            const auto& oracle_input = dtype == backend::Dtype::bf16 ? rounded
                                     : integer ? reconstructed : input;
            device.write(*output, 0, guard.data(), guard.size() * sizeof(float));
            (void)backend::vulkan_kernel_times(device);
            (void)take_matrix_paths(device);
            device.matmul({format.id, {weight_buffer.get(), 0}}, {input_buffer.get(), 0},
                          {output.get(), 1}, width, rows, columns, {&run, 1}, dtype);
            const auto actual = read(device, *output, guard.size());
            require(actual.front() == 77.0f && actual.back() == 77.0f,
                    "tile threshold probe crossed output guards");
            const std::string type_suffix = format.xs ? "xs" : "xxs";
            const std::string expected_kernel = columns < threshold
                ? "matmul_row_iq2_" + type_suffix
                : dtype == backend::Dtype::bf16 ? "matmul_tile_iq2_" + type_suffix + "_bf16"
                : integer ? "matmul_tile_iq2_" + type_suffix + "_dot"
                          : "matmul_tile_iq2_" + type_suffix;
            const bool timed = has_timed_kernel(device, expected_kernel);
            if (columns < threshold) saw_row_timestamp |= timed;
            else saw_tile_timestamp |= timed;
            if (backend::vulkan_timed_dispatches(device))
                require(timed, "tile threshold edge did not timestamp the expected IQ2 kernel family");

            for (size_t column = 0; column < columns; ++column)
                for (size_t row = 0; row < rows; ++row) {
                    const float weight = weights[row * width + selected[column]];
                    const double expected = double(weight) * oracle_input[column * width + selected[column]];
                    check_close(actual[1 + column * rows + row], expected, std::fabs(expected), width);
                }
            ++checks;
        }
    }
    std::cout << format.name << " tile threshold " << threshold << ": row edge " << row_columns
              << ", tile edge " << tile_columns << ", timestamp row/tile "
              << (saw_row_timestamp ? "yes" : "unavailable") << "/"
              << (saw_tile_timestamp ? "yes" : "unavailable") << '\n';
    return checks;
}
#endif

} // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--vulkan")) {
        std::cerr << "usage: llmx-iq2-small-test [--vulkan]\n";
        return 1;
    }
    if (argc == 2) {
#if LLMX_HAS_BACKEND_VULKAN
        try {
            auto device = backend::make_vulkan_backend(0, true);
            for (const auto& format : formats) {
                check_backend(*device, format);
                const size_t range = check_range(*device, format);
                const size_t tile = check_tile_threshold(*device, format);
                std::cout << format.name << " range products " << range
                          << ", threshold cases " << tile << '\n';
            }
            return 0;
        } catch (const backend::VulkanUnavailable& error) {
            std::cout << "Vulkan unavailable: " << error.what() << '\n';
            return 77;
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            return 1;
        }
#else
        std::cout << "Vulkan backend not built\n";
        return 77;
#endif
    }
    try {
        auto cpu = backend::make_cpu_backend();
        cpu->set_threads(1);
        for (const auto& format : formats) {
            const auto* type = quant::Registry::instance().get(format.id);
            require(type && type->block_size == 256 && type->type_size == format.block_bytes && !type->quantize,
                    "read-only registry metadata differs");
            require(quant::row_bytes(format.id, 512, 2) == 4 * format.block_bytes,
                    "row byte sizing differs");
            check_decoder(format);
            check_backend(*cpu, format);
            const size_t range = check_range(*cpu, format);
            std::cout << format.name << " CPU range products " << range << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
