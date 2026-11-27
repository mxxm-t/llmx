// The GPU kernels' one reader of weight block layouts (docs/src/backends-gpu.md): each type's values per block and bytes per block come from src/quant, and `value` decodes one value exactly as the CPU's dequantizer does.
// The K-quants' and Q5_1's products and sums are written as the CPU's dequantizer rounds them, each product rounded before its sum, the kernels compiling without contraction (CMakeLists.txt, LLMX_HIP_FLAGS).
#pragma once

#include "backends/gpu/kernels/port.h"

#include "quant/fp4.hpp"
#include "quant/iq1.hpp"
#include "quant/iq2_s.hpp"
#include "quant/iq2_xs.hpp"
#include "quant/iq2_xxs.hpp"
#include "quant/iq3_s.hpp"
#include "quant/iq3_xxs.hpp"
#include "quant/types.hpp"

namespace rq {

__device__ inline float half_at(const unsigned char* p) { return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8)))); }

// A K-quant sub-block's 6-bit scale and min, as get_scale_min_k4 in src/quant/k_quants.hpp packs them.
__device__ inline void scale_min_k4(int j, const unsigned char* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct Q8_0 {
    static constexpr unsigned values = quant::Q8_0_BLOCK, bytes = quant::Q8_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) { return half_at(b) * (float)((const signed char*)(b + 2))[j]; }
};

struct Q4_0 {
    static constexpr unsigned values = quant::Q4_0_BLOCK, bytes = quant::Q4_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int q = j < 16 ? b[2 + j] & 0xF : b[2 + j - 16] >> 4;
        return (float)(q - 8) * half_at(b);
    }
};

struct Q4_1 {
    static constexpr unsigned values = quant::Q4_1_BLOCK, bytes = quant::Q4_1_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int q = j < 16 ? b[4 + j] & 0xF : b[4 + j - 16] >> 4;
        return half_at(b) * (float)q + half_at(b + 2);
    }
};

struct Q5_0 {
    static constexpr unsigned values = quant::Q5_0_BLOCK, bytes = quant::Q5_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const unsigned qh = b[2] | (b[3] << 8) | (b[4] << 16) | ((unsigned)b[5] << 24);
        const int lo = j < 16 ? b[6 + j] & 0xF : b[6 + j - 16] >> 4;
        return half_at(b) * (float)((int)(lo | (((qh >> j) & 1) << 4)) - 16);
    }
};

struct Q5_1 {
    static constexpr unsigned values = quant::Q5_1_BLOCK, bytes = quant::Q5_1_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const unsigned qh = b[4] | (b[5] << 8) | (b[6] << 16) | ((unsigned)b[7] << 24);
        const int lo = j < 16 ? b[8 + j] & 0xF : b[8 + j - 16] >> 4;
        return half_at(b) * (float)(lo | (((qh >> j) & 1) << 4)) + half_at(b + 2);
    }
};

struct Q4_K {
    static constexpr unsigned values = quant::Q4_K_BLOCK, bytes = quant::Q4_K_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 6, hi = (j >> 5) & 1, l = j & 31;
        int s, m;
        scale_min_k4(2 * g + hi, b + 4, s, m);
        const int q = hi ? b[16 + 32 * g + l] >> 4 : b[16 + 32 * g + l] & 0xF;
        return half_at(b) * (float)s * (float)q - half_at(b + 2) * (float)m;
    }
};

struct Q5_K {
    static constexpr unsigned values = quant::Q5_K_BLOCK, bytes = quant::Q5_K_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 6, hi = (j >> 5) & 1, l = j & 31;
        int s, m;
        scale_min_k4(2 * g + hi, b + 4, s, m);
        const int lo = hi ? b[48 + 32 * g + l] >> 4 : b[48 + 32 * g + l] & 0xF;
        const int q = lo + (((b[16 + l] >> (2 * g + hi)) & 1) << 4);
        return half_at(b) * (float)s * (float)q - half_at(b + 2) * (float)m;
    }
};

struct Q6_K {
    static constexpr unsigned values = quant::Q6_K_BLOCK, bytes = quant::Q6_K_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int n = j >> 7, r = j & 127, quarter = r >> 5, l = r & 31;
        const int lo = b[64 * n + l + 32 * (quarter & 1)] >> (4 * (quarter >> 1)) & 0xF;
        const int q = (lo | (((b[128 + 32 * n + l] >> (2 * quarter)) & 3) << 4)) - 32;
        const float sc = (float)((const signed char*)(b + 192))[8 * n + l / 16 + 2 * quarter];
        return half_at(b + 208) * sc * (float)q;
    }
};

struct F16 {
    static constexpr unsigned values = 1, bytes = 2;
    __device__ static float value(const unsigned char* b, int) { return half_at(b); }
};

struct BF16 {
    static constexpr unsigned values = quant::BF16_BLOCK, bytes = quant::BF16_TYPESIZE;
    __device__ static float value(const unsigned char* b, int) { return __uint_as_float((unsigned)(b[0] | (b[1] << 8)) << 16); }
};

// IQ4_NL's grid as four words of signed bytes, held in registers by the byte selects below; the assert ties them to quant::IQ4_NL_VALUES.
constexpr unsigned iq4_word(int i) {
    return (unsigned)(unsigned char)quant::IQ4_NL_VALUES[4 * i] | (unsigned)(unsigned char)quant::IQ4_NL_VALUES[4 * i + 1] << 8 |
           (unsigned)(unsigned char)quant::IQ4_NL_VALUES[4 * i + 2] << 16 | (unsigned)(unsigned char)quant::IQ4_NL_VALUES[4 * i + 3] << 24;
}
static_assert(iq4_word(0) == 0xBFAD9881u && iq4_word(1) == 0xF6EADDCFu && iq4_word(2) == 0x26190D01u && iq4_word(3) == 0x71594535u, "IQ4_NL grid words");
// Four nibbles, one a byte, as their IQ4_NL grid values in signed bytes: byte selects of the grid's low and high halves, picked by each nibble's bit 3, so no table is read from memory.
__device__ inline unsigned iq4_grid4(unsigned x) {
    const unsigned idx = x & 0x07070707u, high = ((x >> 3) & 0x01010101u) * 0xffu;
    const unsigned lo = byte_perm(0xF6EADDCFu, 0xBFAD9881u, idx), hi = byte_perm(0x71594535u, 0x26190D01u, idx);
    return (lo & ~high) | (hi & high);
}
// One nibble's IQ4_NL grid value.
__device__ inline int iq4_grid(unsigned q) { return (int)(signed char)(iq4_grid4(q) & 0xffu); }

struct IQ4_NL {
    static constexpr unsigned values = quant::IQ4_NL_BLOCK, bytes = quant::IQ4_NL_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int q = j < 16 ? b[2 + j] & 0xF : b[2 + j - 16] >> 4;
        return half_at(b) * (float)iq4_grid((unsigned)q);
    }
};

// Sixteen affine groups of 16 values: a scale and min nibble a group, four 2-bit planes in two halves, then the f16 scale and min (quant::dequantize_row_q2_K).
struct Q2_K {
    static constexpr unsigned values = quant::Q2_K_BLOCK, bytes = quant::Q2_K_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 4, lane = j & 15;
        const float scale = half_at(b + 80) * (float)(b[g] & 15);
        const float min = half_at(b + 82) * (float)(b[g] >> 4);
        const int code = (b[16 + 32 * (g / 8) + 16 * (g % 2) + lane] >> (2 * ((g % 8) / 2))) & 3;
        return scale * (float)code - min;
    }
};

// Sixteen signed 6-bit scales on signed 3-bit codes, a clear high bit taking four from the stored low code (quant::dequantize_row_q3_K).
struct Q3_K {
    static constexpr unsigned values = quant::Q3_K_BLOCK, bytes = quant::Q3_K_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 4, plane = j >> 5;
        const int scale = ((b[96 + g % 8] >> (4 * (g / 8))) & 15) | (((b[104 + g % 4] >> (2 * (g / 4))) & 3) << 4);
        const float ds = half_at(b + 108) * (float)(scale - 32);
        const int low = (b[32 + 32 * (j >> 7) + (j & 31)] >> (2 * (plane % 4))) & 3;
        return ds * (float)(low - (((b[j & 31] >> plane) & 1) ? 0 : 4));
    }
};

// Eight 32-value groups of IQ4_NL codes under 6-bit signed scales of the f16 scale, low nibbles before high in each group (quant::dequantize_row_iq4_xs).
struct IQ4_XS {
    static constexpr unsigned values = quant::IQ4_XS_BLOCK, bytes = quant::IQ4_XS_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 5, l = j & 31;
        const unsigned high = b[2] | (b[3] << 8);
        const int sc = (int)(((b[4 + g / 2] >> (4 * (g % 2))) & 15) | (((high >> (2 * g)) & 3) << 4)) - 32;
        const unsigned char* qs = b + 8 + 16 * g;
        const int q = l < 16 ? qs[l] & 15 : qs[l - 16] >> 4;
        return half_at(b) * (float)sc * (float)iq4_grid((unsigned)q);
    }
};

// Seven stored sign bits and an eighth that gives the byte even parity, as quant::iq_even_parity_signs sets it.
__device__ inline unsigned parity_signs(unsigned code) { return code | ((__popc(code) & 1u) << 7); }

// Eight 32-value groups of four grid bytes and a word of four sign codes and a scale nibble (quant::dequantize_row_iq2_xxs).
struct IQ2_XXS {
    static constexpr unsigned values = quant::IQ2_XXS_BLOCK, bytes = quant::IQ2_XXS_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int v = (j >> 3) & 3, lane = j & 7;
        const unsigned char* f = b + 2 + 8 * (j >> 5);
        const unsigned word = f[4] | (f[5] << 8) | (f[6] << 16) | ((unsigned)f[7] << 24);
        const float scale = half_at(b) * (0.5f + (float)(word >> 28)) * 0.25f;
        const float value = scale * (float)((GPU_TABLE(quant::IQ2_XXS_GRID)[f[v]] >> (8 * lane)) & 255u);
        return (parity_signs((word >> (7 * v)) & 127u) >> lane) & 1u ? -value : value;
    }
};

// 16-bit words of a 9-bit grid index and a 7-bit sign code, a scale nibble each 16 values (quant::dequantize_row_iq2_xs).
struct IQ2_XS {
    static constexpr unsigned values = quant::IQ2_XS_BLOCK, bytes = quant::IQ2_XS_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 5, v = (j >> 3) & 3, lane = j & 7, at = 2 + 8 * g + 2 * v;
        const unsigned word = b[at] | (b[at + 1] << 8);
        const float scale = half_at(b) * (0.5f + (float)((b[66 + g] >> (4 * (v / 2))) & 15u)) * 0.25f;
        const float value = scale * (float)((GPU_TABLE(quant::IQ2_XS_GRID)[word & 511u] >> (8 * lane)) & 255u);
        return (parity_signs(word >> 9) >> lane) & 1u ? -value : value;
    }
};

// A 10-bit grid index each eight values, a sign bit each value, an odd multiplier over eight each 16 values (quant::dequantize_row_iq2_s).
struct IQ2_S {
    static constexpr unsigned values = quant::IQ2_S_BLOCK, bytes = quant::IQ2_S_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 3, lane = j & 7;
        const unsigned index = b[2 + g] | (((b[66 + g / 4] >> (2 * (g % 4))) & 3u) << 8);
        const unsigned scale = (b[74 + g / 4] >> (4 * ((g / 2) % 2))) & 15u;
        const float scaled = half_at(b) * (float)(1 + 2 * scale) * 0.125f;
        const float value = scaled * (float)((GPU_TABLE(quant::IQ2_S_GRID)[index] >> (8 * lane)) & 255u);
        return (b[34 + g] >> lane) & 1 ? -value : value;
    }
};

// Eight 32-value groups of eight grid bytes, four values each, under a word of four sign codes and an odd multiplier over four (quant::dequantize_row_iq3_xxs).
struct IQ3_XXS {
    static constexpr unsigned values = quant::IQ3_XXS_BLOCK, bytes = quant::IQ3_XXS_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 5, v = (j >> 2) & 7, lane = j & 3;
        const unsigned char* f = b + 66 + 4 * g;
        const unsigned word = f[0] | (f[1] << 8) | (f[2] << 16) | ((unsigned)f[3] << 24);
        const float scaled = half_at(b) * (float)(1 + 2 * (word >> 28)) * 0.25f;
        const float value = scaled * (float)((GPU_TABLE(quant::IQ3_XXS_GRID)[b[2 + 8 * g + v]] >> (8 * lane)) & 255u);
        return (parity_signs((word >> (7 * (v / 2))) & 127u) >> (4 * (v % 2) + lane)) & 1u ? -value : value;
    }
};

// A 9-bit grid index each four values, a sign bit each value, an odd multiplier each 32 values (quant::dequantize_row_iq3_s).
struct IQ3_S {
    static constexpr unsigned values = quant::IQ3_S_BLOCK, bytes = quant::IQ3_S_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 2, lane = j & 3;
        const unsigned index = b[2 + g] | (((b[66 + g / 8] >> (g % 8)) & 1u) << 8);
        const unsigned scale = (b[106 + g / 16] >> (4 * ((g / 8) % 2))) & 15u;
        const float scaled = half_at(b) * (float)(1 + 2 * scale);
        const float value = scaled * (float)((GPU_TABLE(quant::IQ3_S_GRID)[index] >> (8 * lane)) & 255u);
        return (b[74 + j / 8] >> (j % 8)) & 1 ? -value : value;
    }
};

// A ternary lane of the IQ1 grid, stored as value + 1 in two bits, two eight-lane vectors a word (quant::iq1_value).
__device__ inline int iq1_value(unsigned index, int lane) { return (int)((GPU_TABLE(quant::IQ1_GRID)[index / 2] >> (16 * (index % 2) + 2 * lane)) & 3u) - 1; }

// Eight values a grid vector with an 11-bit index, an odd multiplier and a shift of an eighth each 32 values (quant::dequantize_row_iq1_s).
struct IQ1_S {
    static constexpr unsigned values = quant::IQ1_S_BLOCK, bytes = quant::IQ1_S_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 3, h = 34 + 2 * (g / 4);
        const unsigned fields = b[h] | (b[h + 1] << 8);
        const unsigned index = b[2 + g] | (((fields >> (3 * (g % 4))) & 7u) << 8);
        const float scaled = half_at(b) * (float)(1 + 2 * ((fields >> 12) & 7u));
        return scaled * ((float)iq1_value(index, j & 7) + ((fields & 0x8000) ? -0.125f : 0.125f));
    }
};

// IQ1_S's grid under 3-bit multipliers each 16 values and a binary16 scale spread over the scale words' high nibbles (quant::dequantize_row_iq1_m).
struct IQ1_M {
    static constexpr unsigned values = quant::IQ1_M_BLOCK, bytes = quant::IQ1_M_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 3, s = 48 + 2 * (g / 8);
        const unsigned half = (b[49] >> 4) | ((b[51] >> 4) << 4) | ((b[53] >> 4) << 8) | ((unsigned)(b[55] >> 4) << 12);
        const unsigned fields = (b[32 + g / 2] >> (4 * (g % 2))) & 15u;
        const unsigned index = b[g] | ((fields & 7u) << 8);
        const unsigned scales = b[s] | (b[s + 1] << 8);
        const float scaled = __half2float(__ushort_as_half((unsigned short)half)) * (float)(1 + 2 * ((scales >> (3 * ((g / 2) % 4))) & 7u));
        return scaled * ((float)iq1_value(index, j & 7) + ((fields & 8u) ? -0.125f : 0.125f));
    }
};

// Base-3 digits five a byte, read as the CPU's wrapped multiply (quant::tq1_digit), centered at one under the block's binary16 scale.
struct TQ1_0 {
    static constexpr unsigned values = quant::TQ1_0_BLOCK, bytes = quant::TQ1_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int at = j < 160 ? j % 32 : j < 240 ? 32 + (j - 160) % 16 : 48 + (j - 240) % 4;
        const int power = j < 160 ? j / 32 : j < 240 ? (j - 160) / 16 : (j - 240) / 4;
        const unsigned scale = power == 0 ? 1u : power == 1 ? 3u : power == 2 ? 9u : power == 3 ? 27u : 81u;
        const unsigned digit = (((b[at] * scale) & 255u) * 3) >> 8;
        return (float)((int)digit - 1) * half_at(b + 52);
    }
};

// Two-bit codes centered at one, four planes of 32 values in each half (quant::dequantize_row_tq2_0).
struct TQ2_0 {
    static constexpr unsigned values = quant::TQ2_0_BLOCK, bytes = quant::TQ2_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const unsigned code = (b[(j / 128) * 32 + j % 32] >> (2 * ((j % 128) / 32))) & 3u;
        return (float)((int)code - 1) * half_at(b + 64);
    }
};

// One sign bit a value under the block's binary16 scale (quant::dequantize_row_q1_0).
struct Q1_0 {
    static constexpr unsigned values = quant::Q1_0_BLOCK, bytes = quant::Q1_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const float d = half_at(b);
        return (b[2 + j / 8] >> (j % 8)) & 1u ? d : -d;
    }
};

// Two-bit codes centered at one under the block's binary16 scale (quant::dequantize_row_q2_0).
struct Q2_0 {
    static constexpr unsigned values = quant::Q2_0_BLOCK, bytes = quant::Q2_0_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) { return (float)((int)((b[2 + j / 4] >> (2 * (j % 4))) & 3u) - 1) * half_at(b); }
};

// Half an E8M0 scale, 2^(e - 128), for the doubled E2M1 codes, as quant::mxfp4_scale gives it, subnormal below e = 2.
__device__ inline float e8m0_half(unsigned e) { return __uint_as_float(e <= 1 ? 0x00200000u << e : (e - 1) << 23); }

struct MXFP4 {
    static constexpr unsigned values = quant::MXFP4_BLOCK, bytes = quant::MXFP4_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int q = j < 16 ? b[1 + j] & 0xF : b[1 + j - 16] >> 4;
        return e8m0_half(b[0]) * (float)GPU_TABLE(quant::FP4_VALUES)[q];
    }
};

// Half an unsigned E4M3 scale for the doubled E2M1 codes, as quant::nvfp4_scale gives it: subnormal at exponent 0, and the reserved byte 127 zero.
__device__ inline float e4m3_half(unsigned s) {
    const unsigned e = (s >> 3) & 15, m = s & 7;
    if (s == 127) return 0.0f;
    return e ? __uint_as_float(((e + 119) << 23) | (m << 20)) : (float)m * (1.0f / 1024.0f);
}

// Four scales, then four groups of eight bytes, each byte's low nibble a value of the group's first eight and its high nibble one of its last eight (quant::dequantize_row_nvfp4).
struct NVFP4 {
    static constexpr unsigned values = quant::NVFP4_BLOCK, bytes = quant::NVFP4_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int g = j >> 4, packed = b[4 + 8 * g + (j & 7)];
        return e4m3_half(b[g]) * (float)GPU_TABLE(quant::FP4_VALUES)[j & 8 ? packed >> 4 : packed & 15];
    }
};

// An F8_E4M3 code as its unscaled value (quant::dequantize_row_f8_e4m3): subnormal at exponent 0, the codes 0x7F and 0xFF NaN of their sign.
struct F8_E4M3 {
    static constexpr unsigned values = quant::F8_E4M3_BLOCK, bytes = quant::F8_E4M3_TYPESIZE;
    __device__ static float value(const unsigned char* b, int) {
        const unsigned c = b[0], e = (c >> 3) & 15, m = c & 7, sign = (c & 0x80u) << 24;
        if (e == 15 && m == 7) return __uint_as_float(sign | 0x7fc00000u);
        return __uint_as_float(sign | __float_as_uint(e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * (1.0f / 512.0f)));
    }
};

} // namespace rq

// A GLU output from its gate and up values, form 0 SiLU, 1 quick GELU and 2 softcap, which every GLU producer in both modules takes (Backend::silu_mul).
__device__ __forceinline__ float glu_value(float g, float u, float limit, int form, float up_limit) {
    if (limit > 0.0f && form != 2) {
        u = u < -limit ? -limit : u > limit ? limit : u;
        g = g > limit ? limit : g;
    }
    return form == 2 ? gpu_tanhf(g / limit) * limit * (1.0f / (1.0f + expf(-g))) * (gpu_tanhf(u / up_limit) * up_limit)
           : form ? (u + 1.0f) * (g * (1.0f / (1.0f + expf(-(1.702f * g))))) : g / (1.0f + expf(-g)) * u;
}

// The activation twin's rule, which every producer of a twin in both modules takes (docs/src/backends-gpu.md).
// The largest magnitude of the 32 lanes of a twin block, a value a lane.
__device__ inline float block_amax(float v) {
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, shuffle_xor(a, o, 32));
    return a;
}

// Value v of a twin block whose 32 lanes give amax, at L levels: d the block's scale and q the value scaled to L (docs/STATUS.md, the twin's rule), L 32767 for block-int16 and 127 for block-int8; lane 0 writes d, and the code is returned.
template <typename QT, int L>
__device__ inline int twin_value(float v, float amax, QT* q, float* d, int lane) {
    if (amax == 0.0f) {
        *q = 0;
        if (lane == 0) *d = 0.0f;
        return 0;
    }
    int e = 0;
    (void)frexpf(amax, &e);   // amax = m 2^e, m in [0.5, 1)
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = ldexpf(amax, s), xn = ldexpf(v, s);
    const float r = xn * ((float)L / an);
    float t = floorf(fabsf(r) + 0.5f);
    t = fminf(t, (float)L);
    const QT c = (QT)(r < 0.0f ? -t : t);
    *q = c;
    if (lane == 0) *d = ldexpf(an / (float)L, -s);
    return (int)c;
}

// Twin block b's two half sums, of its first 16 codes and its last 16, which the tiles' offset and min terms read in place of summing the codes: two int16 for block-int8 and two ints for block-int16, the first half's first.
// The LANES lanes holding the block call it together, each with the sum c of its codes and `lane` its first value's place, so the lanes of values 0 and 16 write; a null s, a twin no tile reads, takes none.
template <typename QT, int LANES>
__device__ inline void twin_sums(int c, int lane, void* s, unsigned long long b) {
    if (!s) return;
    c = segment_sum<LANES / 2>(c);
    if (lane % 16 == 0) {
        if constexpr (sizeof(QT) == 1) ((short*)s)[2 * b + lane / 16] = (short)c;
        else ((int*)s)[2 * b + lane / 16] = c;
    }
}

// Half h of a twin block's two sums as twin_sums stores them, for block-int8 and block-int16.
__device__ inline int twin_half(int w, int h) { return h ? w >> 16 : (int)(short)(w & 0xffff); }
__device__ inline int twin_half(int2 w, int h) { return h ? w.y : w.x; }
