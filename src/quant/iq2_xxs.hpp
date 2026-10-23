#pragma once
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/iq_signs.hpp"
#include "quant/types.hpp"

namespace quant {

inline constexpr uint64_t IQ2_XXS_GRID[256] = {
#include "quant/iq2_xxs_grid.inc"
};

// Each 32-value group stores four byte indices and one word of four sign codes plus a scale nibble.
inline void dequantize_row_iq2_xxs(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ2_XXS_TYPESIZE;
        const float d = f16_to_f32(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
        for (size_t group = 0; group < 8; ++group) {
            const uint8_t* fields = p + 2 + 8 * group;
            const uint32_t word = uint32_t(fields[4]) | (uint32_t(fields[5]) << 8) |
                                  (uint32_t(fields[6]) << 16) | (uint32_t(fields[7]) << 24);
            const float scale = d * (0.5f + float(word >> 28)) * 0.25f;
            for (size_t vector = 0; vector < 4; ++vector) {
                const uint64_t magnitudes = IQ2_XXS_GRID[fields[vector]];
                const uint8_t signs = iq_even_parity_signs((word >> (7 * vector)) & 127u);
                for (size_t lane = 0; lane < 8; ++lane) {
                    const float value = scale * float((magnitudes >> (8 * lane)) & 255u);
                    dst[b * IQ2_XXS_BLOCK + 32 * group + 8 * vector + lane] =
                        ((signs >> lane) & 1u) ? -value : value;
                }
            }
        }
    }
}

} // namespace quant
