#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <unordered_map>

#include "core/bf16.hpp"
#include "core/fp16.hpp"
#include "quant/k_quants.hpp"
#include "quant/iq2_xxs.hpp"
#include "quant/iq2_xs.hpp"
#include "quant/iq3_xxs.hpp"
#include "quant/iq3_s.hpp"
#include "quant/iq2_s.hpp"
#include "quant/iq1.hpp"
#include "quant/mxfp4.hpp"
#include "quant/types.hpp"

// Block quantizers, decoders and their registry; types.hpp owns storage metadata.

namespace quant {

inline void dequantize_row_f16(const uint8_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = f16_to_f32((uint16_t)(src[2 * i] | ((uint16_t)src[2 * i + 1] << 8)));
}

inline void dequantize_row_bf16(const uint8_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; ++i)
        dst[i] = bf16_to_f32((uint16_t)(src[2 * i] | ((uint16_t)src[2 * i + 1] << 8)));
}

inline void quantize_row_q8_0(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * Q8_0_BLOCK;
        uint8_t*      y = dst + b * Q8_0_TYPESIZE;

        float amax = 0.0f;
        for (size_t j = 0; j < Q8_0_BLOCK; j++)
            amax = std::max(amax, std::fabs(x[j]));

        const float d = amax / 127.0f;
        const uint16_t d16 = f32_to_f16(d);
        y[0] = (uint8_t)(d16 & 0xff);
        y[1] = (uint8_t)(d16 >> 8);

        for (size_t j = 0; j < Q8_0_BLOCK; j++) {
            float q = (d > 0.0f) ? std::round(x[j] / d) : 0.0f;
            int v = (int)q;
            if (v > 127)  v = 127;
            if (v < -127) v = -127;
            y[2 + j] = (uint8_t)(int8_t)v;
        }
    }
}

inline void dequantize_row_q8_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* y = src + b * Q8_0_TYPESIZE;
        float*         x = dst + b * Q8_0_BLOCK;

        uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
        const float d = f16_to_f32(d16);
        for (size_t j = 0; j < Q8_0_BLOCK; j++)
            x[j] = (float)(int8_t)y[2 + j] * d;
    }
}

// Q4_0 block quantization: 32 floats in a 2-byte f16 scale and 16 bytes of nibbles (Q4_0_TYPESIZE = 18), the low nibble element j and the high element j+16, stored unsigned 0..15 where the true value is nibble - 8.
// The scale is d = amax/7, so codes run -7 to 7 (-8 too for a tiny scale rounded in F32), and decoding as d*(nibble - 8) gives the format's -0 at nibble 8 under a negative scale, exactly in f32.
inline void quantize_row_q4_0(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * Q4_0_BLOCK;
        uint8_t*      y = dst + b * Q4_0_TYPESIZE;

        float amax = 0.0f;
        for (size_t j = 0; j < Q4_0_BLOCK; j++)
            amax = std::max(amax, std::fabs(x[j]));

        const float d = amax / 7.0f;
        const uint16_t d16 = f32_to_f16(d);
        y[0] = (uint8_t)(d16 & 0xff);
        y[1] = (uint8_t)(d16 >> 8);
        float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        float scaled[Q4_0_BLOCK];
        // Normalize only a block whose reciprocal overflows, keeping the ordinary packing loop unchanged.
        if (!std::isfinite(id)) {
            for (size_t j = 0; j < Q4_0_BLOCK; ++j) scaled[j] = x[j] / d;
            x = scaled;
            id = 1.0f;
        }

        for (size_t j = 0; j < Q4_0_BLOCK / 2; j++) {
            int lo = (int)std::round(x[j] * id) + 8;
            int hi = (int)std::round(x[j + Q4_0_BLOCK / 2] * id) + 8;
            lo = std::min(15, std::max(0, lo));
            hi = std::min(15, std::max(0, hi));
            y[2 + j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

inline void dequantize_row_q4_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* y = src + b * Q4_0_TYPESIZE;
        float*         x = dst + b * Q4_0_BLOCK;

        uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
        const float d = f16_to_f32(d16);
        for (size_t j = 0; j < Q4_0_BLOCK / 2; j++) {
            uint8_t byte = y[2 + j];
            x[j] = (float)((int)(byte & 0x0F) - 8) * d;
            x[j + Q4_0_BLOCK / 2] = (float)((int)(byte >> 4) - 8) * d;
        }
    }
}

// Q4_1 block: 2-byte f16 scale d, 2-byte f16 min m, then 16 bytes of nibbles (Q4_1_TYPESIZE = 20).
// Unlike Q4_0 the nibble is unsigned and the block carries its own offset, so the value is d*q + m rather than d*(q-8).
inline void dequantize_row_q4_1(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* y = src + b * Q4_1_TYPESIZE;
        float*         x = dst + b * Q4_1_BLOCK;
        const float d = f16_to_f32((uint16_t)(y[0] | ((uint16_t)y[1] << 8)));
        const float m = f16_to_f32((uint16_t)(y[2] | ((uint16_t)y[3] << 8)));
        for (size_t j = 0; j < Q4_1_BLOCK / 2; j++) {
            const uint8_t byte = y[4 + j];
            x[j]                              = d * (float)(byte & 0x0F) + m;
            x[j + Q4_1_BLOCK / 2]       = d * (float)(byte >> 4)   + m;
        }
    }
}

// Q5_0: half scale, four fifth-bit bytes, then 16 nibble bytes; d * (code - 16).
// Byte loads handle the alternating two-byte alignment of consecutive 22-byte blocks.
inline void dequantize_row_q5_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* y = src + b * Q5_0_TYPESIZE;
        const float d = f16_to_f32(uint16_t(y[0] | (uint16_t(y[1]) << 8)));
        for (size_t j = 0; j < Q5_0_BLOCK; ++j) {
            const unsigned low = (y[6 + j % 16] >> (4 * (j / 16))) & 15;
            const unsigned high = (y[2 + j / 8] >> (j % 8)) & 1;
            dst[b * Q5_0_BLOCK + j] = d * float(int(low | (high << 4)) - 16);
        }
    }
}

// Q5_1 adds one high bit per value to Q4_1's affine codes, before the packed nibbles.
inline void dequantize_row_q5_1(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* y = src + b * Q5_1_TYPESIZE;
        float* x = dst + b * Q5_1_BLOCK;
        const float d = f16_to_f32((uint16_t)(y[0] | ((uint16_t)y[1] << 8)));
        const float m = f16_to_f32((uint16_t)(y[2] | ((uint16_t)y[3] << 8)));
        for (size_t j = 0; j < Q5_1_BLOCK; ++j) {
            const unsigned low = (y[8 + j % 16] >> (4 * (j / 16))) & 15;
            const unsigned high = (y[4 + j / 8] >> (j % 8)) & 1;
            x[j] = d * (float)(low | (high << 4)) + m;
        }
    }
}

inline constexpr int8_t IQ4_NL_VALUES[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// Both IQ4 formats store each 32-value group's low nibbles before its high nibbles.
inline void dequantize_iq4_group(const uint8_t* qs, float d, float* dst) {
    for (size_t j = 0; j < 16; ++j) {
        dst[j] = d * float(IQ4_NL_VALUES[qs[j] & 15]);
        dst[j + 16] = d * float(IQ4_NL_VALUES[qs[j] >> 4]);
    }
}

// IQ4_XS sub-block g: four low bits in a nibble, two high bits in the little-endian word.
inline int iq4_xs_scale(const uint8_t* block, size_t g) {
    const unsigned high = unsigned(block[2]) | (unsigned(block[3]) << 8);
    return int(((block[4 + g / 2] >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4)) - 32;
}

// IQ4_NL keeps the low nibbles in the first half of the block and the high nibbles in the second.
inline void dequantize_row_iq4_nl(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* y = src + b * IQ4_NL_TYPESIZE;
        float* x = dst + b * IQ4_NL_BLOCK;
        const float d = f16_to_f32(uint16_t(y[0] | (uint16_t(y[1]) << 8)));
        dequantize_iq4_group(y + 2, d, x);
    }
}

inline void dequantize_row_iq4_xs(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* y = src + b * IQ4_XS_TYPESIZE;
        const float d = f16_to_f32(uint16_t(y[0] | (uint16_t(y[1]) << 8)));
        for (size_t g = 0; g < 8; ++g)
            dequantize_iq4_group(y + 8 + 16 * g, d * float(iq4_xs_scale(y, g)), dst + b * IQ4_XS_BLOCK + 32 * g);
    }
}

// An implemented type: its storage metadata and block-wise (de)quantize routines; F32 is copied directly.
struct QuantType : StorageType {
    void (*quantize)(const float*, uint8_t*, size_t) = nullptr;
    void (*dequantize)(const uint8_t*, float*, size_t) = nullptr;
};

// Registry of quant types keyed by GGML type id.
class Registry {
public:
    // A function-local static is initialized once even when threads race to it (C++11), and the map is never written after, so any thread may read it without a lock.
    static const Registry& instance() {
        static const Registry r;
        return r;
    }
    const QuantType* get(uint32_t ggml_id) const {
        auto it = types_.find(ggml_id);
        return it == types_.end() ? nullptr : &it->second;
    }

private:
    Registry() : types_{
        { GGML_TYPE_Q8_0,
          { *storage_type(GGML_TYPE_Q8_0), quantize_row_q8_0, dequantize_row_q8_0 } },
        { GGML_TYPE_Q4_0,
          { *storage_type(GGML_TYPE_Q4_0), quantize_row_q4_0, dequantize_row_q4_0 } },
        // Q4_1 and the K-quants are read-only: llmx loads files that carry them, a few Q6_K tensors in an otherwise Q4_0 file included, but produces none, so a quantizer would be unused code; the tests pack Q4_1 themselves (tests/quantizers.hpp).
        { GGML_TYPE_Q4_1,
          { *storage_type(GGML_TYPE_Q4_1), nullptr, dequantize_row_q4_1 } },
        { GGML_TYPE_Q5_0,
          { *storage_type(GGML_TYPE_Q5_0), nullptr, dequantize_row_q5_0 } },
        { GGML_TYPE_Q5_1,
          { *storage_type(GGML_TYPE_Q5_1), nullptr, dequantize_row_q5_1 } },
        { GGML_TYPE_Q2_K,
          { *storage_type(GGML_TYPE_Q2_K), nullptr, dequantize_row_q2_K } },
        { GGML_TYPE_Q3_K,
          { *storage_type(GGML_TYPE_Q3_K), nullptr, dequantize_row_q3_K } },
        { GGML_TYPE_Q4_K,
          { *storage_type(GGML_TYPE_Q4_K), nullptr, dequantize_row_q4_K } },
        { GGML_TYPE_Q5_K,
          { *storage_type(GGML_TYPE_Q5_K), nullptr, dequantize_row_q5_K } },
        { GGML_TYPE_Q6_K,
          { *storage_type(GGML_TYPE_Q6_K), nullptr, dequantize_row_q6_K } },
        { GGML_TYPE_MXFP4,
          { *storage_type(GGML_TYPE_MXFP4), nullptr, dequantize_row_mxfp4 } },
        { GGML_TYPE_F16,
          { *storage_type(GGML_TYPE_F16), nullptr, dequantize_row_f16 } },
        { GGML_TYPE_BF16,
          { *storage_type(GGML_TYPE_BF16), nullptr, dequantize_row_bf16 } },
        { GGML_TYPE_IQ1_S,
          { *storage_type(GGML_TYPE_IQ1_S), nullptr, dequantize_row_iq1_s } },
        { GGML_TYPE_IQ1_M,
          { *storage_type(GGML_TYPE_IQ1_M), nullptr, dequantize_row_iq1_m } },
        { GGML_TYPE_IQ4_NL,
          { *storage_type(GGML_TYPE_IQ4_NL), nullptr, dequantize_row_iq4_nl } },
        { GGML_TYPE_IQ3_S,
          { *storage_type(GGML_TYPE_IQ3_S), nullptr, dequantize_row_iq3_s } },
        { GGML_TYPE_IQ2_XXS,
          { *storage_type(GGML_TYPE_IQ2_XXS), nullptr, dequantize_row_iq2_xxs } },
        { GGML_TYPE_IQ2_XS,
          { *storage_type(GGML_TYPE_IQ2_XS), nullptr, dequantize_row_iq2_xs } },
        { GGML_TYPE_IQ3_XXS,
          { *storage_type(GGML_TYPE_IQ3_XXS), nullptr, dequantize_row_iq3_xxs } },
        { GGML_TYPE_IQ2_S,
          { *storage_type(GGML_TYPE_IQ2_S), nullptr, dequantize_row_iq2_s } },
        { GGML_TYPE_IQ4_XS,
          { *storage_type(GGML_TYPE_IQ4_XS), nullptr, dequantize_row_iq4_xs } },
        { GGML_TYPE_F32,
          { *storage_type(GGML_TYPE_F32), nullptr, nullptr } },
    } {}
    const std::unordered_map<uint32_t, QuantType> types_;
};

} // namespace quant
