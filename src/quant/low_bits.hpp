#pragma once
#include <cstddef>
#include <cstdint>

#include "core/fp16.hpp"
#include "quant/types.hpp"

namespace quant {

inline void dequantize_row_q1_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* block = src + b * Q1_0_TYPESIZE;
        const float d = f16_to_f32(uint16_t(block[0] | (uint16_t(block[1]) << 8)));
        for (size_t j = 0; j < Q1_0_BLOCK; ++j) {
            const unsigned bit = (block[2 + j / 8] >> (j % 8)) & 1u;
            dst[b * Q1_0_BLOCK + j] = bit ? d : -d;
        }
    }
}


} // namespace quant
