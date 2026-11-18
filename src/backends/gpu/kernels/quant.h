// The GPU kernels' one reader of weight block layouts (docs/src/backends-gpu.md): each type's values per block and bytes per block come from src/quant, and `value` decodes one value exactly as the CPU's dequantizer does.
// The K-quants' and Q5_1's products and sums are written as the CPU's dequantizer rounds them, each product rounded before its sum, the kernels compiling without contraction (CMakeLists.txt, LLMX_HIP_FLAGS).
#pragma once

#include "backends/gpu/kernels/port.h"

#include "quant/fp4.hpp"
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

struct IQ4_NL {
    static constexpr unsigned values = quant::IQ4_NL_BLOCK, bytes = quant::IQ4_NL_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int q = j < 16 ? b[2 + j] & 0xF : b[2 + j - 16] >> 4;
        return half_at(b) * (float)GPU_TABLE(quant::IQ4_NL_VALUES)[q];
    }
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

} // namespace rq

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
