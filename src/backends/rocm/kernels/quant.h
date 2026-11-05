// The ROCm backend's one reader of weight block layouts (docs/src/backends-rocm.md): each type's values per block and bytes per block come from src/quant, and `value` decodes one value exactly as the CPU's dequantizer does.
// The K-quants' and Q5_1's products and sums are written as the CPU's dequantizer rounds them, each product rounded before its sum, the kernels compiling without contraction (CMakeLists.txt, LLMX_HIP_FLAGS).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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

struct BF16 {
    static constexpr unsigned values = quant::BF16_BLOCK, bytes = quant::BF16_TYPESIZE;
    __device__ static float value(const unsigned char* b, int) { return __uint_as_float((unsigned)(b[0] | (b[1] << 8)) << 16); }
};

struct IQ4_NL {
    static constexpr unsigned values = quant::IQ4_NL_BLOCK, bytes = quant::IQ4_NL_TYPESIZE;
    __device__ static float value(const unsigned char* b, int j) {
        const int q = j < 16 ? b[2 + j] & 0xF : b[2 + j - 16] >> 4;
        return half_at(b) * (float)quant::IQ4_NL_VALUES[q];
    }
};

} // namespace rq
