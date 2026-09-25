#pragma once
// Synthetic quantized weights for the Vulkan benches.
#include <cstdint>
#include <random>
#include <vector>
#include "format/gguf.hpp"

// Bytes of one row of `n` values stored as GGUF type `type`.
inline size_t row_bytes(uint32_t type, size_t n) {
    gguf::TensorInfo t;
    t.type = type;
    t.ne = {n};
    return (size_t)t.data_size();
}

// Bytes whose every half is a finite fp16, so whatever a block reads as a scale stays finite.
inline std::vector<uint8_t> weights(size_t bytes, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> v(bytes);
    for (auto& b : v) b = (uint8_t)(rng() % 0x3c);
    return v;
}
