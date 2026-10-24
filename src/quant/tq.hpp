#pragma once

#include <cstddef>
#include <cstdint>

#include "core/fp16.hpp"
#include "quant/types.hpp"

namespace quant {

// Codes in both layouts are centered at one and use the block's binary16 scale; TQ1's radix-packed bytes are expanded separately.
inline float tq_value(unsigned code, float d) {
    return float(int(code) - 1) * d;
}

inline unsigned tq1_digit(uint8_t packed, unsigned power) {
    constexpr uint8_t powers[] = {1, 3, 9, 27, 81};
    const uint8_t wrapped = uint8_t(unsigned(packed) * powers[power]);
    return (unsigned(wrapped) * 3) >> 8;
}

inline void dequantize_row_tq1_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* block = src + b * TQ1_0_TYPESIZE;
        const float d = f16_to_f32(uint16_t(block[52] | (uint16_t(block[53]) << 8)));
        float* out = dst + b * TQ1_0_BLOCK;

        // The first 32 packed bytes describe five planes of 32 values.
        for (unsigned n = 0; n < 5; ++n)
            for (unsigned m = 0; m < 32; ++m)
                out[n * 32 + m] = tq_value(tq1_digit(block[m], n), d);
        // The remaining 16 packed bytes describe five planes of 16 values.
        for (unsigned n = 0; n < 5; ++n)
            for (unsigned m = 0; m < 16; ++m)
                out[160 + n * 16 + m] = tq_value(tq1_digit(block[32 + m], n), d);
        // The four qh bytes describe four planes of four values.
        for (unsigned n = 0; n < 4; ++n)
            for (unsigned m = 0; m < 4; ++m)
                out[240 + n * 4 + m] = tq_value(tq1_digit(block[48 + m], n), d);
    }
}

inline void dequantize_row_tq2_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* block = src + b * TQ2_0_TYPESIZE;
        const float d = f16_to_f32(uint16_t(block[64] | (uint16_t(block[65]) << 8)));
        float* out = dst + b * TQ2_0_BLOCK;
        for (unsigned half = 0; half < 2; ++half)
            for (unsigned plane = 0; plane < 4; ++plane)
                for (unsigned lane = 0; lane < 32; ++lane) {
                    const unsigned code = (block[half * 32 + lane] >> (plane * 2)) & 3;
                    out[half * 128 + plane * 32 + lane] = tq_value(code, d);
                }
    }
}

} // namespace quant
