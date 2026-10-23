#pragma once
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/iq_signs.hpp"
#include "quant/types.hpp"

namespace quant {

inline constexpr uint32_t IQ3_XXS_GRID[256] = {
#include "quant/iq3_xxs_grid.inc"
};

// Each 32 values share an odd multiplier divided by four and four seven-bit sign codes.
inline void dequantize_row_iq3_xxs(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ3_XXS_TYPESIZE;
        const float d = f16_to_f32(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
        for (size_t g = 0; g < 8; ++g) {
            const uint8_t* fields = p + 66 + 4 * g;
            const uint32_t word = uint32_t(fields[0]) | (uint32_t(fields[1]) << 8) |
                                  (uint32_t(fields[2]) << 16) | (uint32_t(fields[3]) << 24);
            const float scaled = d * float(1 + 2 * (word >> 28)) * 0.25f;
            for (size_t v = 0; v < 8; ++v) {
                const uint32_t vector = IQ3_XXS_GRID[p[2 + 8 * g + v]];
                const uint8_t signs = iq_even_parity_signs((word >> (7 * (v / 2))) & 127u);
                for (size_t lane = 0; lane < 4; ++lane) {
                    const float value = scaled * float((vector >> (8 * lane)) & 255u);
                    dst[b * IQ3_XXS_BLOCK + 32 * g + 4 * v + lane] =
                        ((signs >> (4 * (v % 2) + lane)) & 1u) ? -value : value;
                }
            }
        }
    }
}

} // namespace quant
