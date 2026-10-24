#pragma once
// Test weight packing: the registry's quantizers for writable types, a Q4_1 quantizer and finite MXFP4 raw blocks.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include "core/fp16.hpp"
#include "quant/quant.hpp"

namespace testq {

// Finite MXFP4 blocks spanning all packed codes and seven adjacent power-of-two scales.
inline std::vector<uint8_t> mxfp4_matrix(size_t n, uint32_t seed) {
    std::vector<uint8_t> bytes(n / quant::MXFP4_BLOCK * quant::MXFP4_TYPESIZE);
    for (size_t b = 0; b < bytes.size() / quant::MXFP4_TYPESIZE; ++b) {
        bytes[b * quant::MXFP4_TYPESIZE] = uint8_t(124 + b % 7);
        for (size_t i = 1; i < quant::MXFP4_TYPESIZE; ++i) bytes[b * quant::MXFP4_TYPESIZE + i] = uint8_t(seed + b * 31 + i * 17);
    }
    return bytes;
}

// Deterministic raw TQ blocks for Vulkan's read-only format checks.
inline std::vector<uint8_t> tq_matrix(uint32_t type, size_t values, uint32_t seed) {
    const bool tq1 = type == quant::GGML_TYPE_TQ1_0;
    const size_t block_bytes = tq1 ? quant::TQ1_0_TYPESIZE : quant::TQ2_0_TYPESIZE;
    if ((type != quant::GGML_TYPE_TQ1_0 && type != quant::GGML_TYPE_TQ2_0) || values % 256)
        throw std::runtime_error("testq: invalid TQ matrix shape");
    std::vector<uint8_t> bytes(values / 256 * block_bytes);
    for (size_t b = 0; b < bytes.size() / block_bytes; ++b) {
        uint8_t* block = bytes.data() + b * block_bytes;
        for (size_t i = 0; i < block_bytes - 2; ++i)
            block[i] = uint8_t(seed + 73 * i + 41 * b + 19 * (i >> 2));
        block[block_bytes - 2] = 0x00;
        block[block_bytes - 1] = 0x3c;
    }
    return bytes;
}

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

inline void quantize_row_q5_1(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const float* x = src + b * quant::Q5_1_BLOCK;
        uint8_t* y = dst + b * quant::Q5_1_TYPESIZE;
        const float mn = *std::min_element(x, x + quant::Q5_1_BLOCK);
        const float mx = *std::max_element(x, x + quant::Q5_1_BLOCK);
        const float d = (mx - mn) / 31.0f;
        const uint16_t scale = f32_to_f16(d), offset = f32_to_f16(mn);
        y[0] = (uint8_t)scale; y[1] = (uint8_t)(scale >> 8);
        y[2] = (uint8_t)offset; y[3] = (uint8_t)(offset >> 8);
        std::fill(y + 4, y + quant::Q5_1_TYPESIZE, uint8_t(0));
        for (size_t i = 0; i < quant::Q5_1_BLOCK; ++i) {
            const int rounded = d > 0 ? (int)std::round((x[i] - mn) / d) : 0;
            const unsigned code = (unsigned)std::clamp(rounded, 0, 31);
            y[4 + i / 8] |= (uint8_t)((code >> 4) << (i % 8));
            y[8 + i % 16] |= (uint8_t)((code & 15) << (4 * (i / 16)));
        }
    }
}

// Test-only packers cover read-only types; writable types use the registry's quantizer.
inline void (*quantizer(uint32_t type))(const float*, uint8_t*, size_t) {
    if (type == quant::GGML_TYPE_Q4_1) return quantize_row_q4_1;
    if (type == quant::GGML_TYPE_Q5_1) return quantize_row_q5_1;
    const quant::QuantType* q = quant::Registry::instance().get(type);
    return q ? q->quantize : nullptr;
}

}  // namespace testq
