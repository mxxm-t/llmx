#pragma once
#include <cstdint>
#include <cstddef>

#include "core/fp16.hpp"
#include "format/gguf.hpp"

// K-quant block formats.
// These differ from the simple block types in quant.hpp in two ways: the block is a 256-value SUPER-block, and each super-block carries per-32-value sub-scales that are themselves quantized to 6 bits against a pair of f16 super-block scales.
//
// Read-only: llmx loads these because the Hub hosts them, but produces none.

namespace quant {

// Unpack the 6-bit sub-scale and sub-min of sub-block j (0..7) from the 12 packed bytes, shared by Q4_K and Q5_K.
// The first four pairs sit in the low 6 bits of bytes 0..7; the last four take their low 4 bits from bytes 8..11 and their high 2 bits from the top of the earlier bytes.
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4));
    }
}

// Q4_K super-block, gguf::Q4_K_TYPESIZE = 144 bytes:
//   d      f16 super-block scale for the sub-scales
//   dmin   f16 super-block scale for the sub-mins
//   sc[12] eight 6-bit sub-scales and eight 6-bit sub-mins, packed
//   qs[128] 4-bit quants, low nibble first for each group of 32
// A value is d*sc[sub] * q - dmin*m[sub]; unlike Q4_0 the nibble is unsigned and each sub-block carries its own offset.
inline void dequantize_row_q4_K(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* p = src + b * gguf::Q4_K_TYPESIZE;
        const float d    = f16_to_f32((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
        const float dmin = f16_to_f32((uint16_t)(p[2] | ((uint16_t)p[3] << 8)));
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + 16;
        float* y = dst + b * gguf::Q4_K_BLOCK;
        int is = 0;
        for (int j = 0; j < (int)gguf::Q4_K_BLOCK; j += 64) {
            uint8_t s, mm;
            get_scale_min_k4(is + 0, sc, &s, &mm);
            const float d1 = d * (float)s, m1 = dmin * (float)mm;
            get_scale_min_k4(is + 1, sc, &s, &mm);
            const float d2 = d * (float)s, m2 = dmin * (float)mm;
            for (int l = 0; l < 32; l++) y[l]      = d1 * (float)(qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) y[l + 32] = d2 * (float)(qs[l] >> 4)  - m2;
            y += 64; qs += 32; is += 2;
        }
    }
}

// Q5_K super-block, gguf::Q5_K_TYPESIZE = 176 bytes:
//   d       f16 super-block scale for the sub-scales
//   dmin    f16 super-block scale for the sub-mins
//   sc[12]  eight 6-bit sub-scales and eight 6-bit sub-mins, packed as in Q4_K
//   qh[32]  the FIFTH bit of every quant, one bit per value
//   qs[128] the low 4 bits
// Q4_K with a fifth bit: the value is d*sc*(nibble + 16*bit) - dmin*m.
// A value's bit lives in qh at a position that advances by two per 64-value group, which is what u1/u2 track.
inline void dequantize_row_q5_K(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* p = src + b * gguf::Q5_K_TYPESIZE;
        const float d    = f16_to_f32((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
        const float dmin = f16_to_f32((uint16_t)(p[2] | ((uint16_t)p[3] << 8)));
        const uint8_t* sc = p + 4;
        const uint8_t* qh = p + 16;
        const uint8_t* ql = p + 48;
        float* y = dst + b * gguf::Q5_K_BLOCK;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < (int)gguf::Q5_K_BLOCK; j += 64) {
            uint8_t s, mm;
            get_scale_min_k4(is + 0, sc, &s, &mm);
            const float d1 = d * (float)s, m1 = dmin * (float)mm;
            get_scale_min_k4(is + 1, sc, &s, &mm);
            const float d2 = d * (float)s, m2 = dmin * (float)mm;
            for (int l = 0; l < 32; l++)
                y[l]      = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                y[l + 32] = d2 * (float)((ql[l] >>  4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            y += 64; ql += 32; is += 2;
            u1 = (uint8_t)(u1 << 2);
            u2 = (uint8_t)(u2 << 2);
        }
    }
}

// Q6_K super-block, gguf::Q6_K_TYPESIZE = 210 bytes:
//   ql[128]  low 4 bits of each quant
//   qh[64]   high 2 bits, packed 4 quants per byte
//   sc[16]   int8 per-16-value scale
//   d        f16 super-block scale
// A quant is (low4 | high2 << 4) - 32, scaled by d * sc[group].
// The layout walks the block in two halves of 128, hence strides of 64 for ql, 32 for qh and 8 for sc.
inline void dequantize_row_q6_K(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* p = src + b * gguf::Q6_K_TYPESIZE;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = (const int8_t*)(p + 192);
        const float d = f16_to_f32((uint16_t)(p[208] | ((uint16_t)p[209] << 8)));
        float* y = dst + b * gguf::Q6_K_BLOCK;
        for (int n = 0; n < (int)gguf::Q6_K_BLOCK; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int q1 = (int)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

} // namespace quant
