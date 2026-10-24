#pragma once
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/types.hpp"

namespace quant {

inline constexpr uint32_t IQ1_GRID[1024] = {
#include "quant/iq1_grid.inc"
};

// Two eight-lane ternary vectors per word, with each two-bit lane storing value + 1.
inline int iq1_value(unsigned index, size_t lane) {
    return int((IQ1_GRID[index / 2] >> (16 * (index % 2) + 2 * lane)) & 3u) - 1;
}

inline void dequantize_row_iq1_s(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ1_S_TYPESIZE;
        const float d = f16_to_f32(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
        for (size_t g = 0; g < 32; ++g) {
            const size_t h = 34 + 2 * (g / 4);
            const unsigned fields = p[h] | (unsigned(p[h + 1]) << 8);
            const unsigned index = p[2 + g] | (((fields >> (3 * (g % 4))) & 7u) << 8);
            const float scaled = d * float(1 + 2 * ((fields >> 12) & 7u));
            const float shift = (fields & 0x8000) ? -0.125f : 0.125f;
            for (size_t lane = 0; lane < 8; ++lane)
                dst[b * IQ1_S_BLOCK + 8 * g + lane] = scaled * (float(iq1_value(index, lane)) + shift);
        }
    }
}

inline void dequantize_row_iq1_m(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ1_M_TYPESIZE;
        // Each scale word reserves its high nibble for one nibble of the common binary16 scale.
        unsigned half = 0;
        for (size_t k = 0; k < 4; ++k)
            half |= unsigned(p[49 + 2 * k] >> 4) << (4 * k);
        const float d = f16_to_f32(uint16_t(half));
        for (size_t g = 0; g < 32; ++g) {
            const unsigned fields = (p[32 + g / 2] >> (4 * (g % 2))) & 15u;
            const unsigned index = p[g] | ((fields & 7u) << 8);
            const size_t s = 48 + 2 * (g / 8);
            const unsigned scales = p[s] | (unsigned(p[s + 1]) << 8);
            const float scaled = d * float(1 + 2 * ((scales >> (3 * ((g / 2) % 4))) & 7u));
            const float shift = (fields & 8u) ? -0.125f : 0.125f;
            for (size_t lane = 0; lane < 8; ++lane)
                dst[b * IQ1_M_BLOCK + 8 * g + lane] = scaled * (float(iq1_value(index, lane)) + shift);
        }
    }
}

} // namespace quant
