#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "quant/types.hpp"

namespace quant {

// E2M1 values doubled to signed integers; both zero codes give +0 on decode.
inline constexpr int8_t MXFP4_VALUES[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// At e = 253 the largest doubled code, 12 * 2^(e - 128), is 3 * 2^127, beyond finite f32; at 252 it is finite.
inline constexpr uint8_t MXFP4_FIRST_OVERFLOW_EXPONENT = 253;

// Half the E8M0 scale, for the doubled table: 2^(e - 128), including the GGUF convention for e = 255.
inline float mxfp4_scale(uint8_t e) {
    static constexpr auto table = [] {
        std::array<uint32_t, 256> values{};
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = i <= 1 ? (0x00200000u << i) : (uint32_t(i) - 1) << 23;
        return values;
    }();
    const uint32_t bits = table[e];
    float scale;
    std::memcpy(&scale, &bits, sizeof scale);
    return scale;
}

// A block's low nibbles give values 0..15 and its high nibbles 16..31; weights widen to f32 without an intermediate half.
inline void dequantize_row_mxfp4(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * MXFP4_TYPESIZE;
        const float scale = mxfp4_scale(p[0]);
        for (size_t j = 0; j < MXFP4_BLOCK / 2; ++j) {
            const uint8_t packed = p[j + 1];
            dst[b * MXFP4_BLOCK + j] = scale * MXFP4_VALUES[packed & 15];
            dst[b * MXFP4_BLOCK + j + MXFP4_BLOCK / 2] = scale * MXFP4_VALUES[packed >> 4];
        }
    }
}

} // namespace quant
