#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <unordered_map>

#include "core/fp16.hpp"
#include "format/gguf.hpp"

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
    r.add(gguf::GGML_TYPE_F32,
          { "F32", 0, 4, nullptr, nullptr });
}

} // namespace quant
