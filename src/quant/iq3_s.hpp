#pragma once
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/types.hpp"

namespace quant {

inline constexpr uint32_t IQ3_S_GRID[512] = {
#include "quant/iq3_s_grid.inc"
};

// Four consecutive values select one dictionary vector; each 32 values share an odd multiplier, with an independent sign bit for every value.
inline void dequantize_row_iq3_s(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ3_S_TYPESIZE;
        const float d = f16_to_f32(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
        for (size_t g = 0; g < 64; ++g) {
            const unsigned index = p[2 + g] | (((p[66 + g / 8] >> (g % 8)) & 1u) << 8);
            const uint32_t vector = IQ3_S_GRID[index];
            const unsigned scale = (p[106 + g / 16] >> (4 * ((g / 8) % 2))) & 15;
            const float scaled = d * float(1 + 2 * scale);
            for (size_t lane = 0; lane < 4; ++lane) {
                const size_t i = 4 * g + lane;
                const float value = scaled * float((vector >> (8 * lane)) & 255);
                dst[b * IQ3_S_BLOCK + i] = ((p[74 + i / 8] >> (i % 8)) & 1) ? -value : value;
            }
        }
    }
}

} // namespace quant
