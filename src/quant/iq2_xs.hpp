#pragma once
#include <cstddef>
#include <cstdint>
#include "core/fp16.hpp"
#include "quant/iq_signs.hpp"
#include "quant/types.hpp"

namespace quant {

inline constexpr uint64_t IQ2_XS_GRID[512] = {
#include "quant/iq2_xs_grid.inc"
};

// Each word selects a 9-bit dictionary index and a 7-bit sign code; each scale nibble covers 16 values.
inline void dequantize_row_iq2_xs(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * IQ2_XS_TYPESIZE;
        const float d = f16_to_f32(uint16_t(p[0] | (uint16_t(p[1]) << 8)));
        for (size_t group = 0; group < 8; ++group) {
            const uint8_t packed_scales = p[66 + group];
            for (size_t vector = 0; vector < 4; ++vector) {
                const size_t word_offset = 2 + 8 * group + 2 * vector;
                const uint16_t word = uint16_t(p[word_offset]) | (uint16_t(p[word_offset + 1]) << 8);
                const unsigned scale_code = (packed_scales >> (4 * (vector / 2))) & 15u;
                const float scale = d * (0.5f + float(scale_code)) * 0.25f;
                const uint64_t magnitudes = IQ2_XS_GRID[word & 511u];
                const uint8_t signs = iq_even_parity_signs(word >> 9);
                for (size_t lane = 0; lane < 8; ++lane) {
                    const float value = scale * float((magnitudes >> (8 * lane)) & 255u);
                    dst[b * IQ2_XS_BLOCK + 32 * group + 8 * vector + lane] =
                        ((signs >> lane) & 1u) ? -value : value;
                }
            }
        }
    }
}

} // namespace quant
