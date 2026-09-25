#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <unordered_map>

#include "core/fp16.hpp"
#include "format/gguf.hpp"
#include "quant/k_quants.hpp"

// Block quantization: the Q8_0, Q4_0 and Q4_1 row kernels, and the registry naming each GGUF type llmx reads with its block size and kernels (the K-quants' are in k_quants.hpp).
// A Q8_0 block holds 32 float values as a 2-byte f16 scale and 32 int8 values (gguf::Q8_0_TYPESIZE bytes per block).
// The kernels serve the quantize command (float -> block) and the dequantize command and CPU inference path (block -> float).

namespace quant {

inline void quantize_row_q8_0(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * gguf::Q8_0_BLOCK;
        uint8_t*      y = dst + b * gguf::Q8_0_TYPESIZE;

        float amax = 0.0f;
        for (size_t j = 0; j < gguf::Q8_0_BLOCK; j++)
            amax = std::max(amax, std::fabs(x[j]));

        const float d = amax / 127.0f;
        const uint16_t d16 = f32_to_f16(d);
        y[0] = (uint8_t)(d16 & 0xff);
        y[1] = (uint8_t)(d16 >> 8);

        for (size_t j = 0; j < gguf::Q8_0_BLOCK; j++) {
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
        const uint8_t* y = src + b * gguf::Q8_0_TYPESIZE;
        float*         x = dst + b * gguf::Q8_0_BLOCK;

        uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
        const float d = f16_to_f32(d16);
        for (size_t j = 0; j < gguf::Q8_0_BLOCK; j++)
            x[j] = (float)(int8_t)y[2 + j] * d;
    }
}

// Q4_0 block quantization.
// A block holds 32 floats compressed into a 2-byte f16 scale + 16 bytes of nibbles (gguf::Q4_0_TYPESIZE = 18 bytes per block).
// The scale is d = amax/7 so the quantized range [-8, 7] maps to [-amax, amax].
// Each byte holds two values: the low nibble is element j, the high nibble element j+16; the stored nibble is unsigned 0..15 where the true value = nibble - 8.
inline void quantize_row_q4_0(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * gguf::Q4_0_BLOCK;
        uint8_t*      y = dst + b * gguf::Q4_0_TYPESIZE;

        float amax = 0.0f;
        for (size_t j = 0; j < gguf::Q4_0_BLOCK; j++)
            amax = std::max(amax, std::fabs(x[j]));

        const float d = amax / 7.0f;
        const uint16_t d16 = f32_to_f16(d);
        y[0] = (uint8_t)(d16 & 0xff);
        y[1] = (uint8_t)(d16 >> 8);
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;

        for (size_t j = 0; j < gguf::Q4_0_BLOCK / 2; j++) {
            int lo = (int)std::round(x[j] * id) + 8;
            int hi = (int)std::round(x[j + gguf::Q4_0_BLOCK / 2] * id) + 8;
            lo = std::min(15, std::max(0, lo));
            hi = std::min(15, std::max(0, hi));
            y[2 + j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

inline void dequantize_row_q4_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* y = src + b * gguf::Q4_0_TYPESIZE;
        float*         x = dst + b * gguf::Q4_0_BLOCK;

        uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
        const float d = f16_to_f32(d16);
        for (size_t j = 0; j < gguf::Q4_0_BLOCK / 2; j++) {
            uint8_t byte = y[2 + j];
            x[j] = (float)(int)(byte & 0x0F) * d - 8.0f * d;
            x[j + gguf::Q4_0_BLOCK / 2] = (float)(int)(byte >> 4) * d - 8.0f * d;
        }
    }
}

// Q4_1 block: 2-byte f16 scale d, 2-byte f16 min m, then 16 bytes of nibbles (gguf::Q4_1_TYPESIZE = 20).
// Unlike Q4_0 the nibble is unsigned and the block carries its own offset, so the value is d*q + m rather than d*(q-8).
inline void quantize_row_q4_1(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * gguf::Q4_1_BLOCK;
        uint8_t*      y = dst + b * gguf::Q4_1_TYPESIZE;
        float mn = x[0], mx = x[0];
        for (size_t j = 1; j < gguf::Q4_1_BLOCK; j++) {
            mn = std::min(mn, x[j]);
            mx = std::max(mx, x[j]);
        }
        const float d = (mx - mn) / 15.0f;
        const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
        const uint16_t d16 = f32_to_f16(d), m16 = f32_to_f16(mn);
        y[0] = (uint8_t)(d16 & 0xff); y[1] = (uint8_t)(d16 >> 8);
        y[2] = (uint8_t)(m16 & 0xff); y[3] = (uint8_t)(m16 >> 8);
        for (size_t j = 0; j < gguf::Q4_1_BLOCK / 2; j++) {
            int lo = (int)std::round((x[j] - mn) * id);
            int hi = (int)std::round((x[j + gguf::Q4_1_BLOCK / 2] - mn) * id);
            lo = std::min(15, std::max(0, lo));
            hi = std::min(15, std::max(0, hi));
            y[4 + j] = (uint8_t)(lo | (hi << 4));
        }
    }
}

inline void dequantize_row_q4_1(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* y = src + b * gguf::Q4_1_TYPESIZE;
        float*         x = dst + b * gguf::Q4_1_BLOCK;
        const float d = f16_to_f32((uint16_t)(y[0] | ((uint16_t)y[1] << 8)));
        const float m = f16_to_f32((uint16_t)(y[2] | ((uint16_t)y[3] << 8)));
        for (size_t j = 0; j < gguf::Q4_1_BLOCK / 2; j++) {
            const uint8_t byte = y[4 + j];
            x[j]                              = d * (float)(byte & 0x0F) + m;
            x[j + gguf::Q4_1_BLOCK / 2]       = d * (float)(byte >> 4)   + m;
        }
    }
}

// A quantized storage type: block size, bytes per block, and block-wise (de)quantize routines.
// quant::Registry holds one for each type llmx reads, so consumers look a type up by its GGML id.
struct QuantType {
    const char* name = "?";
    size_t block_size = 0;   // values per block
    size_t type_size  = 0;   // bytes per block
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
        { gguf::GGML_TYPE_Q8_0,
          { "Q8_0", gguf::Q8_0_BLOCK, gguf::Q8_0_TYPESIZE, quantize_row_q8_0, dequantize_row_q8_0 } },
        { gguf::GGML_TYPE_Q4_0,
          { "Q4_0", gguf::Q4_0_BLOCK, gguf::Q4_0_TYPESIZE, quantize_row_q4_0, dequantize_row_q4_0 } },
        { gguf::GGML_TYPE_Q4_1,
          { "Q4_1", gguf::Q4_1_BLOCK, gguf::Q4_1_TYPESIZE, quantize_row_q4_1, dequantize_row_q4_1 } },
        // The K-quants are read-only: llmx loads files that carry them, including a few Q6_K tensors inside an otherwise Q4_0 file, but produces none, so a quantizer would be unused code.
        { gguf::GGML_TYPE_Q4_K,
          { "Q4_K", gguf::Q4_K_BLOCK, gguf::Q4_K_TYPESIZE, nullptr, dequantize_row_q4_K } },
        { gguf::GGML_TYPE_Q5_K,
          { "Q5_K", gguf::Q5_K_BLOCK, gguf::Q5_K_TYPESIZE, nullptr, dequantize_row_q5_K } },
        { gguf::GGML_TYPE_Q6_K,
          { "Q6_K", gguf::Q6_K_BLOCK, gguf::Q6_K_TYPESIZE, nullptr, dequantize_row_q6_K } },
        { gguf::GGML_TYPE_F32,
          { "F32", 0, 4, nullptr, nullptr } },
    } {}
    const std::unordered_map<uint32_t, QuantType> types_;
};

} // namespace quant
