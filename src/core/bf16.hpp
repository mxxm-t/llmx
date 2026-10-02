#pragma once
#include <cstdint>
#include <cstring>

inline uint16_t f32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    // Preserve NaN as NaN even when its only payload bits would be discarded.
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 0x40u);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float bf16_to_f32(uint16_t b) {
    const uint32_t bits = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
