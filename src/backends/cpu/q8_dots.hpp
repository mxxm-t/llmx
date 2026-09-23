#pragma once
// Decode dots against activations quantized per block of 32 (docs/src/backends-cpu.md).
// A row's weights stay packed and meet the activations in integers, a block's sum then scaled once, where the float dots converted every weight and so were bound by arithmetic rather than by memory.
// Q8_0, Q4_K and Q5_K read 8-bit activations through the unsigned-by-signed byte multiply; Q4_0, Q4_1 and Q6_K read 16-bit ones through the 16-bit multiply-add, since on 8 bits the HF gate's Q4_0 file, whose head is Q6_K, fails its top-5 bound, as the device's row kernels found.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <immintrin.h>

#include "core/fp16.hpp"
#include "format/gguf.hpp"
#include "quant/k_quants.hpp"

namespace backend {
namespace q8 {

// Activation rows as the dots read them: per block of 32 the quants, the scale (the block's largest magnitude over 127) and the quants' sum, which the types with a minimum or an offset need.
struct Rows {
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<int32_t> sum;
    size_t nin = 0;
    const int8_t* qs(size_t r) const { return q.data() + r * nin; }
    const float* ds(size_t r) const { return d.data() + r * (nin / 32); }
    const int32_t* sums(size_t r) const { return sum.data() + r * (nin / 32); }
};

// The same rows on 16 bits, the scale the block's largest magnitude over 32767.
struct Rows16 {
    std::vector<int16_t> q;
    std::vector<float> d;
    std::vector<int32_t> sum;
    size_t nin = 0;
    const int16_t* qs(size_t r) const { return q.data() + r * nin; }
    const float* ds(size_t r) const { return d.data() + r * (nin / 32); }
    const int32_t* sums(size_t r) const { return sum.data() + r * (nin / 32); }
};

inline void quantize16(const float* x, size_t rows, size_t nin, Rows16& out) {
    out.nin = nin;
    out.q.resize(rows * nin);
    out.d.resize(rows * nin / 32);
    out.sum.resize(rows * nin / 32);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (size_t b = 0; b < rows * nin / 32; ++b) {
        const float* p = x + b * 32;
        __m256 v[4];
        __m256 amax = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(p + 8 * i);
            amax = _mm256_max_ps(amax, _mm256_andnot_ps(sign, v[i]));
        }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(amax), _mm256_extractf128_ps(amax, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float top = _mm_cvtss_f32(m);
        const __m256 scale = _mm256_set1_ps(top > 0.0f ? 32767.0f / top : 0.0f);
        __m256i q[4];
        for (int i = 0; i < 4; ++i)
            q[i] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[i], scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        const __m256i s = _mm256_add_epi32(_mm256_add_epi32(q[0], q[1]), _mm256_add_epi32(q[2], q[3]));
        __m128i t = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
        // Packing interleaves the 128-bit lanes, so a permute of 64-bit pieces restores value order.
        _mm256_storeu_si256((__m256i*)(out.q.data() + b * 32), _mm256_permute4x64_epi64(_mm256_packs_epi32(q[0], q[1]), 0xD8));
        _mm256_storeu_si256((__m256i*)(out.q.data() + b * 32 + 16), _mm256_permute4x64_epi64(_mm256_packs_epi32(q[2], q[3]), 0xD8));
        out.d[b] = top / 32767.0f;
        out.sum[b] = _mm_cvtsi128_si32(t);
    }
}

// Quantize `rows` rows of `nin` values, nin a multiple of 32, rounding to nearest with ties to even.
inline void quantize(const float* x, size_t rows, size_t nin, Rows& out) {
    out.nin = nin;
    out.q.resize(rows * nin);
    out.d.resize(rows * nin / 32);
    out.sum.resize(rows * nin / 32);
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (size_t b = 0; b < rows * nin / 32; ++b) {
        const float* p = x + b * 32;
        __m256 v[4];
        __m256 amax = _mm256_setzero_ps();
        for (int i = 0; i < 4; ++i) {
            v[i] = _mm256_loadu_ps(p + 8 * i);
            amax = _mm256_max_ps(amax, _mm256_andnot_ps(sign, v[i]));
        }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(amax), _mm256_extractf128_ps(amax, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_movehdup_ps(m));
        const float top = _mm_cvtss_f32(m);
        const float d = top / 127.0f, id = top > 0.0f ? 127.0f / top : 0.0f;
        const __m256 scale = _mm256_set1_ps(id);
        __m256i q[4];
        for (int i = 0; i < 4; ++i)
            q[i] = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v[i], scale), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        const __m256i s = _mm256_add_epi32(_mm256_add_epi32(q[0], q[1]), _mm256_add_epi32(q[2], q[3]));
        __m128i t = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
        t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
        // Packing interleaves the 128-bit lanes, so a permute restores value order.
        const __m256i w = _mm256_packs_epi32(q[0], q[1]), z = _mm256_packs_epi32(q[2], q[3]);
        __m256i bytes = _mm256_packs_epi16(w, z);
        bytes = _mm256_permutevar8x32_epi32(bytes, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256((__m256i*)(out.q.data() + b * 32), bytes);
        out.d[b] = d;
        out.sum[b] = _mm_cvtsi128_si32(t);
    }
}

// Sums of 32 byte products into eight 32-bit lanes, lane i covering bytes 4i to 4i + 3: unsigned by signed, and signed by signed through the sign trick.
inline __m256i dot_us(__m256i u, __m256i s) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1));
}
inline __m256i dot_ss(__m256i a, __m256i b) {
    return dot_us(_mm256_sign_epi8(a, a), _mm256_sign_epi8(b, a));
}
inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}
inline float half(const uint8_t* p) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)(p[0] | ((uint32_t)p[1] << 8)))));
}
inline __m256i load(const void* p) { return _mm256_loadu_si256((const __m256i*)p); }

// One row of `nin` weights against activation row r.
inline float dot_q8_0(const uint8_t* row, const Rows& x, size_t r) {
    const size_t nb = x.nin / 32;
    const int8_t* q = x.qs(r);
    const float* d = x.ds(r);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    size_t b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t* p = row + b * gguf::Q8_0_TYPESIZE;
        const __m256i s0 = dot_ss(load(p + 2), load(q + b * 32));
        const __m256i s1 = dot_ss(load(p + 2 + gguf::Q8_0_TYPESIZE), load(q + b * 32 + 32));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(half(p) * d[b]), _mm256_cvtepi32_ps(s0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(half(p + gguf::Q8_0_TYPESIZE) * d[b + 1]), _mm256_cvtepi32_ps(s1), acc1);
    }
    for (; b < nb; ++b) {
        const uint8_t* p = row + b * gguf::Q8_0_TYPESIZE;
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(half(p) * d[b]), _mm256_cvtepi32_ps(dot_ss(load(p + 2), load(q + b * 32))), acc0);
    }
    return hsum(_mm256_add_ps(acc0, acc1));
}

// Q4_0 and Q4_1: 16 bytes of nibbles per block, low nibbles values 0 to 15 and high nibbles 16 to 31.
inline __m256i nibbles32(const uint8_t* qs) {
    const __m128i raw = _mm_loadu_si128((const __m128i*)qs);
    const __m128i m = _mm_set1_epi8(0x0F);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), m), _mm_and_si128(raw, m));
}
// 32 signed weight bytes against 32 16-bit activations into eight 32-bit lanes: lanes 0 to 3 hold values 0 to 15 two at a time, lanes 4 to 7 values 16 to 31.
inline __m256i dot16(__m256i w8, const int16_t* x) {
    const __m256i lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(w8)), _mm256_loadu_si256((const __m256i*)x));
    const __m256i hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(w8, 1)), _mm256_loadu_si256((const __m256i*)(x + 16)));
    return _mm256_add_epi32(lo, hi);
}
inline float dot_q4_0(const uint8_t* row, const Rows16& x, size_t r) {
    const size_t nb = x.nin / 32;
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = row + b * gguf::Q4_0_TYPESIZE;
        const __m256i w = _mm256_sub_epi8(nibbles32(p + 2), eight);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(half(p) * d[b]), _mm256_cvtepi32_ps(dot16(w, q + b * 32)), acc);
    }
    return hsum(acc);
}
inline float dot_q4_1(const uint8_t* row, const Rows16& x, size_t r) {
    const size_t nb = x.nin / 32;
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    const int32_t* s = x.sums(r);
    __m256 acc = _mm256_setzero_ps();
    float mins = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = row + b * gguf::Q4_1_TYPESIZE;
        acc = _mm256_fmadd_ps(_mm256_set1_ps(half(p) * d[b]), _mm256_cvtepi32_ps(dot16(nibbles32(p + 4), q + b * 32)), acc);
        mins += half(p + 2) * d[b] * (float)s[b];
    }
    return hsum(acc) + mins;
}
// Q4_K and Q5_K: a 256-value block of eight groups of 32, each with a 6-bit scale and minimum under the block's two half scales; 32 bytes of nibbles hold groups 2j (low nibbles) and 2j + 1 (high).
inline float dot_q45_K(const uint8_t* row, const Rows& x, size_t r, bool five) {
    const size_t nb = x.nin / 256;
    const size_t bytes = five ? gguf::Q5_K_TYPESIZE : gguf::Q4_K_TYPESIZE;
    const int8_t* q = x.qs(r);
    const float* d = x.ds(r);
    const int32_t* s = x.sums(r);
    const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float mins = 0.0f;
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = row + b * bytes;
        const float dd = half(p), dmin = half(p + 2);
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + (five ? 48 : 16);
        const __m256i hb = five ? load(p + 16) : _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i raw = load(qs + 32 * j);
            __m256i lo = _mm256_and_si256(raw, m4), hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4);
            if (five) {
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * j), one), 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * j + 1), one), 4));
            }
            uint8_t s0, m0, s1, m1;
            quant::get_scale_min_k4(2 * j, sc, &s0, &m0);
            quant::get_scale_min_k4(2 * j + 1, sc, &s1, &m1);
            const size_t g = b * 8 + 2 * j;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dd * (float)s0 * d[g]), _mm256_cvtepi32_ps(dot_us(lo, load(q + g * 32))), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dd * (float)s1 * d[g + 1]), _mm256_cvtepi32_ps(dot_us(hi, load(q + g * 32 + 32))), acc);
            mins += dmin * ((float)m0 * d[g] * (float)s[g] + (float)m1 * d[g + 1] * (float)s[g + 1]);
        }
    }
    return hsum(acc) - mins;
}

// Q6_K: 256 values in two halves of 128, each from 64 bytes of low nibbles and 32 bytes of top two bits, minus 32, with a signed 8-bit scale per 16 values.
// Against 16-bit activations a group of 16 is one multiply-add into eight lanes, so each group takes its own scale.
inline float dot_q6_K(const uint8_t* row, const Rows16& x, size_t r) {
    const size_t nb = x.nin / 256;
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3), off = _mm256_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = row + b * gguf::Q6_K_TYPESIZE;
        const float dd = half(p + 208);
        const int8_t* sc = (const int8_t*)(p + 192);
        for (int n = 0; n < 2; ++n) {
            const uint8_t* ql = p + 64 * n;
            const uint8_t* qh = p + 128 + 32 * n;
            const __m256i A = load(ql), B = load(ql + 32), H = load(qh);
            const __m256i w[4] = {
                _mm256_or_si256(_mm256_and_si256(A, m4), _mm256_slli_epi16(_mm256_and_si256(H, m2), 4)),
                _mm256_or_si256(_mm256_and_si256(B, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 2), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 4), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 6), m2), 4)),
            };
            for (int c = 0; c < 4; ++c) {
                const size_t g = b * 8 + n * 4 + c;
                const __m256i v = _mm256_sub_epi8(w[c], off);
                const int16_t* xq = q + g * 32;
                const __m256i lo = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)), _mm256_loadu_si256((const __m256i*)xq));
                const __m256i hi = _mm256_madd_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1)), _mm256_loadu_si256((const __m256i*)(xq + 16)));
                acc = _mm256_fmadd_ps(_mm256_set1_ps(dd * (float)sc[8 * n + 2 * c] * d[g]), _mm256_cvtepi32_ps(lo), acc);
                acc = _mm256_fmadd_ps(_mm256_set1_ps(dd * (float)sc[8 * n + 2 * c + 1] * d[g]), _mm256_cvtepi32_ps(hi), acc);
            }
        }
    }
    return hsum(acc);
}
// Whether a type has a dot here, which activations it reads, and the dot itself.
inline bool has_dot(uint32_t type) {
    return type == gguf::GGML_TYPE_Q8_0 || type == gguf::GGML_TYPE_Q4_0 || type == gguf::GGML_TYPE_Q4_1 ||
           type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K;
}
inline bool reads16(uint32_t type) {
    return type == gguf::GGML_TYPE_Q4_0 || type == gguf::GGML_TYPE_Q4_1 || type == gguf::GGML_TYPE_Q6_K;
}

// The activation rows of a call, each precision quantized once when a type first reads it.
struct Activations {
    Rows x8;
    Rows16 x16;
    const float* src = nullptr;
    size_t rows = 0, nin = 0;
    bool has8 = false, has16 = false;
    void reset(const float* x, size_t n_rows, size_t width) {
        src = x;
        rows = n_rows;
        nin = width;
        has8 = has16 = false;
    }
    void prepare(uint32_t type) {
        if (reads16(type)) {
            if (!has16) { quantize16(src, rows, nin, x16); has16 = true; }
        } else if (!has8) {
            quantize(src, rows, nin, x8);
            has8 = true;
        }
    }
};

inline float dot(uint32_t type, const uint8_t* row, const Activations& x, size_t r) {
    switch (type) {
    case gguf::GGML_TYPE_Q8_0: return dot_q8_0(row, x.x8, r);
    case gguf::GGML_TYPE_Q4_0: return dot_q4_0(row, x.x16, r);
    case gguf::GGML_TYPE_Q4_1: return dot_q4_1(row, x.x16, r);
    case gguf::GGML_TYPE_Q4_K: return dot_q45_K(row, x.x8, r, false);
    case gguf::GGML_TYPE_Q5_K: return dot_q45_K(row, x.x8, r, true);
    default: return dot_q6_K(row, x.x16, r);
    }
}


} // namespace q8
} // namespace backend
