#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <unordered_map>

#include "core/fp16.hpp"
#include "quant/k_quants.hpp"
#include "quant/mxfp4.hpp"
#include "quant/types.hpp"

// Block quantizers, decoders and the registry linking each implemented type to its storage metadata and kernels.
// A Q8_0 block holds 32 float values as a 2-byte f16 scale and 32 int8 values (Q8_0_TYPESIZE bytes per block).
// The kernels serve the quantize command (float -> block) and the dequantize command and CPU inference path (block -> float).

namespace quant {

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

// Q4_0 block quantization.
// A block holds 32 floats compressed into a 2-byte f16 scale + 16 bytes of nibbles (Q4_0_TYPESIZE = 18 bytes per block).
// The scale is d = amax/7, so ordinary scaling maps [-amax, amax] onto codes -7 to 7; rounding a tiny scale in F32 can reach the clamp at -8 too.
// Each byte holds two values: the low nibble is element j, the high nibble element j+16; the stored nibble is unsigned 0..15 where the true value = nibble - 8.
// Decoding as d*(nibble - 8) gives the format's -0 at nibble 8 under a negative scale; the product is exact in f32, since |nibble - 8| is at most 8.
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

inline constexpr int8_t IQ4_NL_VALUES[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// IQ4_NL keeps the low nibbles in the first half of the block and the high nibbles in the second.
inline void dequantize_row_iq4_nl(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* y = src + b * IQ4_NL_TYPESIZE;
        float* x = dst + b * IQ4_NL_BLOCK;
        const float d = f16_to_f32(uint16_t(y[0] | (uint16_t(y[1]) << 8)));
        for (size_t j = 0; j < IQ4_NL_BLOCK / 2; ++j) {
            x[j] = d * float(IQ4_NL_VALUES[y[2 + j] & 15]);
            x[j + IQ4_NL_BLOCK / 2] = d * float(IQ4_NL_VALUES[y[2 + j] >> 4]);
        }
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
        // Q4_1 and the K-quants are read-only: llmx loads files that carry them, including a few Q6_K tensors inside an otherwise Q4_0 file, but produces none, so a quantizer would be unused code; the tests pack Q4_1 with their own (tests/quantizers.hpp).
        { GGML_TYPE_Q4_1,
          { *storage_type(GGML_TYPE_Q4_1), nullptr, dequantize_row_q4_1 } },
        { GGML_TYPE_Q4_K,
          { *storage_type(GGML_TYPE_Q4_K), nullptr, dequantize_row_q4_K } },
        { GGML_TYPE_Q5_K,
          { *storage_type(GGML_TYPE_Q5_K), nullptr, dequantize_row_q5_K } },
        { GGML_TYPE_Q6_K,
          { *storage_type(GGML_TYPE_Q6_K), nullptr, dequantize_row_q6_K } },
        { GGML_TYPE_MXFP4,
          { *storage_type(GGML_TYPE_MXFP4), nullptr, dequantize_row_mxfp4 } },
        { GGML_TYPE_IQ4_NL,
          { *storage_type(GGML_TYPE_IQ4_NL), nullptr, dequantize_row_iq4_nl } },
        { GGML_TYPE_F32,
          { *storage_type(GGML_TYPE_F32), nullptr, nullptr } },
    } {}
    const std::unordered_map<uint32_t, QuantType> types_;
};

} // namespace quant
