#pragma once
#include <cstddef>
#include <cstdint>
#include <cmath>
#include "quant/fp4.hpp"
#include "quant/types.hpp"

namespace quant {

// Half the unsigned E4M3 scale, paired with the doubled E2M1 codebook.
// GGUF producers emit 0..126; the reserved byte 127 decodes as zero.
inline float nvfp4_scale(uint8_t scale) {
    if (scale == 0 || scale == 127) return 0.0f;
    const unsigned exponent = (scale >> 3) & 15, mantissa = scale & 7;
    return exponent ? std::ldexp(float(8 + mantissa), int(exponent) - 11)
                    : std::ldexp(float(mantissa), -10);
}

// Four scales precede four groups; each group's eight bytes hold low nibbles
// for its first eight values and high nibbles for its last eight values.
inline void dequantize_row_nvfp4(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; ++b) {
        const uint8_t* p = src + b * NVFP4_TYPESIZE;
        for (size_t g = 0; g < 4; ++g) {
            const float scale = nvfp4_scale(p[g]);
            float* out = dst + b * NVFP4_BLOCK + 16 * g;
            for (size_t j = 0; j < 8; ++j) {
                const uint8_t packed = p[4 + 8 * g + j];
                out[j] = scale * FP4_VALUES[packed & 15];
                out[j + 8] = scale * FP4_VALUES[packed >> 4];
            }
        }
    }
}
} // namespace quant
