#pragma once
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/types.hpp"

namespace quant {

inline constexpr uint64_t IQ2_S_GRID[1024] = {
#include "quant/iq2_s_grid.inc"
};

// Eight values select one dictionary vector; each 16 values share an odd multiplier divided by eight.
inline void dequantize_row_iq2_s(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ2_S_TYPESIZE;
        const float d = f16_to_f32(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
        for (size_t g = 0; g < 32; ++g) {
            const unsigned index = p[2 + g] | (((p[66 + g / 4] >> (2 * (g % 4))) & 3u) << 8);
            const uint64_t vector = IQ2_S_GRID[index];
            const unsigned scale = (p[74 + g / 4] >> (4 * ((g / 2) % 2))) & 15;
            const float scaled = d * float(1 + 2 * scale) * 0.125f;
            for (size_t lane = 0; lane < 8; ++lane) {
                const float value = scaled * float((vector >> (8 * lane)) & 255);
                dst[b * IQ2_S_BLOCK + 8 * g + lane] = ((p[34 + g] >> lane) & 1) ? -value : value;
            }
        }
    }
}

} // namespace quant
