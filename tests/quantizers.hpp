#pragma once
// The block quantizers the tests pack weights with: the registry's for the types llmx writes, and Q4_1's here, which llmx reads but never writes.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/quant.hpp"

namespace testq {

// Each block's range split into 15 steps from its minimum, the block holding the f16 step and minimum and each value's nearest step as a nibble.
inline void quantize_row_q4_1(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * quant::Q4_1_BLOCK;
        uint8_t* y = dst + b * quant::Q4_1_TYPESIZE;
        float mn = x[0], mx = x[0];
        for (size_t j = 1; j < quant::Q4_1_BLOCK; j++) {
            mn = std::min(mn, x[j]);
            mx = std::max(mx, x[j]);
        }
        const float d = (mx - mn) / 15.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        const uint16_t d16 = f32_to_f16(d), m16 = f32_to_f16(mn);
        y[0] = (uint8_t)(d16 & 0xff); y[1] = (uint8_t)(d16 >> 8);
        y[2] = (uint8_t)(m16 & 0xff); y[3] = (uint8_t)(m16 >> 8);
        for (size_t j = 0; j < quant::Q4_1_BLOCK / 2; j++) {
            int lo = (int)std::round((x[j] - mn) * id);
            int hi = (int)std::round((x[j + quant::Q4_1_BLOCK / 2] - mn) * id);
            lo = std::min(15, std::max(0, lo));
            hi = std::min(15, std::max(0, hi));
            y[4 + j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

// The quantizer for `type`: the registry's, Q4_1's above, or none for a type packed some other way.
inline void (*quantizer(uint32_t type))(const float*, uint8_t*, size_t) {
    if (type == quant::GGML_TYPE_Q4_1) return quantize_row_q4_1;
    const quant::QuantType* q = quant::Registry::instance().get(type);
    return q ? q->quantize : nullptr;
}

}  // namespace testq
