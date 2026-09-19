#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <unordered_map>

#include "core/fp16.hpp"
#include "format/gguf.hpp"
#include "quant/k_quants.hpp"

// Q8_0 block quantization kernels, from scratch. A block holds 32 float values
// that are compressed into a 2-byte f16 scale + 32 int8 quantized values
// (gguf::Q8_0_TYPESIZE bytes per block). These kernels are the building blocks
// for both the quantize command (float -> Q8_0) and the dequantize command /
// CPU inference path (Q8_0 -> float).

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

// Q4_0 block quantization. A block holds 32 floats compressed into a 2-byte f16
// scale + 16 bytes of nibbles (gguf::Q4_0_TYPESIZE = 18 bytes per block). The
// scale is d = amax/7 so the quantized range [-8, 7] maps to [-amax, amax]. Each
// byte holds two values: the low nibble is element j, the high nibble element
// j+16; the stored nibble is unsigned 0..15 where the true value = nibble - 8.
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
            if (lo > 15) lo = 15; if (lo < 0) lo = 0;
            if (hi > 15) hi = 15; if (hi < 0) hi = 0;
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

// Q4_1 block: 2-byte f16 scale d, 2-byte f16 min m, then 16 bytes of nibbles
// (gguf::Q4_1_TYPESIZE = 20). Unlike Q4_0 the nibble is unsigned and the block
// carries its own offset, so the value is d*q + m rather than d*(q-8).
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
            if (lo > 15) lo = 15; if (lo < 0) lo = 0;
            if (hi > 15) hi = 15; if (hi < 0) hi = 0;
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

// Description of a quantized storage type: fixed block size, bytes per block,
// and block-wise (de)quantize routines. Register each type with the
// quant::Registry so consumers can look a type up by its GGML id.
struct QuantType {
    const char* name = "?";
    size_t block_size = 0;   // values per block
    size_t type_size  = 0;   // bytes per block
    void (*quantize)(const float*, uint8_t*, size_t) = nullptr;
    void (*dequantize)(const uint8_t*, float*, size_t) = nullptr;
};

// Registry of quant types keyed by GGML type id. Populate at startup with
// quant::register_builtins().
class Registry {
public:
    static Registry& instance() {
        static Registry r;
        return r;
    }
    void add(uint32_t ggml_id, const QuantType& t) { types_[ggml_id] = t; }
    const QuantType* get(uint32_t ggml_id) const {
        auto it = types_.find(ggml_id);
        return it == types_.end() ? nullptr : &it->second;
    }
    void clear() { types_.clear(); }

private:
    Registry() = default;
    std::unordered_map<uint32_t, QuantType> types_;
};

// Register the built-in quant types (Q8_0). Safe to call multiple times.
inline void register_builtins() {
    Registry& r = Registry::instance();
    r.clear();
    r.add(gguf::GGML_TYPE_Q8_0,
          { "Q8_0", gguf::Q8_0_BLOCK, gguf::Q8_0_TYPESIZE,
            quantize_row_q8_0, dequantize_row_q8_0 });
    r.add(gguf::GGML_TYPE_Q4_0,
          { "Q4_0", gguf::Q4_0_BLOCK, gguf::Q4_0_TYPESIZE,
            quantize_row_q4_0, dequantize_row_q4_0 });
    r.add(gguf::GGML_TYPE_Q4_1,
          { "Q4_1", gguf::Q4_1_BLOCK, gguf::Q4_1_TYPESIZE,
            quantize_row_q4_1, dequantize_row_q4_1 });
    // Q6_K is read-only: llama.cpp upgrades a few tensors to it inside an
    // otherwise Q4_0 file, so llmx needs to LOAD it, but nothing here produces
    // it and a quantizer would be unused code.
    r.add(gguf::GGML_TYPE_Q4_K,
          { "Q4_K", gguf::Q4_K_BLOCK, gguf::Q4_K_TYPESIZE,
            nullptr, dequantize_row_q4_K });
    r.add(gguf::GGML_TYPE_Q5_K,
          { "Q5_K", gguf::Q5_K_BLOCK, gguf::Q5_K_TYPESIZE,
            nullptr, dequantize_row_q5_K });
    r.add(gguf::GGML_TYPE_Q6_K,
          { "Q6_K", gguf::Q6_K_BLOCK, gguf::Q6_K_TYPESIZE,
            nullptr, dequantize_row_q6_K });
    r.add(gguf::GGML_TYPE_F32,
          { "F32", 0, 4, nullptr, nullptr });
}

} // namespace quant
