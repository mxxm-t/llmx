#pragma once
// CPU dots over activations quantized per block of 32; weights remain packed and integer sums are scaled once per block (docs/src/backends-cpu.md).
// Every packed dot uses 16-bit activations. Q8_0 keeps original F32 inputs in the float dots.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <immintrin.h>

#include "core/fp16.hpp"
#include "quant/k_quants.hpp"
#include "quant/mxfp4.hpp"
#include "quant/types.hpp"

namespace backend {
namespace q8 {

// Each block stores its 16-bit integers, reconstruction scale and integer sum, which types with a minimum require.
struct Rows16 {
    std::vector<int16_t> q;
    std::vector<float> d;
    std::vector<int32_t> sum;
    size_t nin = 0;
    const int16_t* qs(size_t r) const { return q.data() + r * nin; }
    const float* ds(size_t r) const { return d.data() + r * (nin / 32); }
    const int32_t* sums(size_t r) const { return sum.data() + r * (nin / 32); }
};

// Sizing is apart from quantizing, so blocks b0 .. b1 of a sized buffer can be filled from several threads.
inline void size_rows(size_t rows, size_t nin, Rows16& out) {
    out.nin = nin;
    out.q.resize(rows * nin);
    out.d.resize(rows * nin / 32);
    out.sum.resize(rows * nin / 32);
}

// A tiny block can overflow the float reciprocal; round against a representable scale without losing zeros or signs.
inline void quantize_small(const float* x, size_t b, float top, Rows16& out) {
    float d = top / 32767.0f;
    if (double(d) * 32767 < double(top))
        d = std::nextafter(d, std::numeric_limits<float>::max());
    int32_t sum = 0;
    for (size_t i = 0; i < 32; ++i) {
        const double value = double(x[i]) / double(d);
        const double low = std::floor(value);
        int32_t q = int32_t(low);
        const double tail = value - low;
        if (tail > 0.5 || (tail == 0.5 && q % 2 != 0)) ++q;
        out.q[b * 32 + i] = int16_t(q);
        sum += q;
    }
    out.d[b] = d;
    out.sum[b] = sum;
}

inline void quantize16(const float* x, size_t b0, size_t b1, Rows16& out) {
    const __m256 sign = _mm256_set1_ps(-0.0f);
    for (size_t b = b0; b < b1; ++b) {
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
        const float id = top > 0.0f ? 32767.0f / top : 0.0f;
        if (!std::isfinite(id)) { quantize_small(p, b, top, out); continue; }
        const __m256 scale = _mm256_set1_ps(id);
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
// Each nibble half widens directly into its integer dot, retaining the eight lanes of dot16 without joining and extracting halves.
inline __m256i dot_mxfp4_block(const uint8_t* qs, const int16_t* x) {
    const __m128i packed = _mm_loadu_si128((const __m128i*)qs);
    const __m128i mask = _mm_set1_epi8(15);
    const __m128i table = _mm_loadu_si128((const __m128i*)quant::MXFP4_VALUES);
    const __m128i lo = _mm_shuffle_epi8(table, _mm_and_si128(packed, mask));
    const __m128i hi = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16(packed, 4), mask));
    const __m256i sum0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(lo), _mm256_loadu_si256((const __m256i*)x));
    const __m256i sum1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(hi), _mm256_loadu_si256((const __m256i*)(x + 16)));
    return _mm256_add_epi32(sum0, sum1);
}

// Extreme scales take decoded f32 weights against the same quantized activations, with double products so combining scales cannot underflow or overflow first.
inline float dot_mxfp4_wide(const uint8_t* row, const Rows16& x, size_t r) {
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    double sum = 0.0;
    for (size_t b = 0; b < x.nin / quant::MXFP4_BLOCK; ++b) {
        float weights[quant::MXFP4_BLOCK];
        quant::dequantize_row_mxfp4(row + b * quant::MXFP4_TYPESIZE, weights, 1);
        for (size_t j = 0; j < quant::MXFP4_BLOCK; ++j)
            sum += double(weights[j]) * (double(q[b * quant::MXFP4_BLOCK + j]) * double(d[b]));
    }
    return float(sum);
}

// Whether the packed dot can combine the scales without losing range; the high exponents may already have infinite decoded weights.
inline bool mxfp4_dot_scale(uint8_t e, float activation, float& scale) {
    scale = quant::mxfp4_scale(e) * activation;
    uint32_t bits;
    std::memcpy(&bits, &scale, sizeof bits);
    const uint32_t exponent = bits & 0x7f800000u;
    return e < quant::MXFP4_FIRST_OVERFLOW_EXPONENT && exponent - 0x00800000u < 0x7f000000u;
}

inline float dot_mxfp4(const uint8_t* row, const Rows16& x, size_t r) {
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < x.nin / quant::MXFP4_BLOCK; ++b) {
        const uint8_t* p = row + b * quant::MXFP4_TYPESIZE;
        float scale;
        if (!mxfp4_dot_scale(p[0], d[b], scale)) return dot_mxfp4_wide(row, x, r);
        const __m256i sum = dot_mxfp4_block(p + 1, q + b * quant::MXFP4_BLOCK);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(sum), acc);
    }
    const float result = hsum(acc);
    return std::isfinite(result) ? result : dot_mxfp4_wide(row, x, r);
}

inline float dot_q4_0(const uint8_t* row, const Rows16& x, size_t r) {
    const size_t nb = x.nin / 32;
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 acc = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = row + b * quant::Q4_0_TYPESIZE;
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
        const uint8_t* p = row + b * quant::Q4_1_TYPESIZE;
        acc = _mm256_fmadd_ps(_mm256_set1_ps(half(p) * d[b]), _mm256_cvtepi32_ps(dot16(nibbles32(p + 4), q + b * 32)), acc);
        mins += half(p + 2) * d[b] * (float)s[b];
    }
    return hsum(acc) + mins;
}
// The eight groups' vectors f(0) .. f(7), eight 32-bit lanes each, to their eight sums, one lane each; exact, so the order is free.
// Built as a tree from the calls, so the vectors stay in registers.
template <class F>
inline __m256i sum8(const F& f) {
    const __m256i a = _mm256_hadd_epi32(f(0), f(1)), b = _mm256_hadd_epi32(f(2), f(3));
    const __m256i c = _mm256_hadd_epi32(f(4), f(5)), e = _mm256_hadd_epi32(f(6), f(7));
    const __m256i ab = _mm256_hadd_epi32(a, b), ce = _mm256_hadd_epi32(c, e);
    return _mm256_add_epi32(_mm256_permute2x128_si256(ab, ce, 0x20), _mm256_permute2x128_si256(ab, ce, 0x31));
}
// Q4_K and Q5_K decode use the prompt dots' exact integer reduction per group, then one F32 lane per group across blocks.
// Each integer sum is at most 32 * 31 * 32767, safely inside int32_t before conversion to F32.
inline float dot_q45_K(const uint8_t* row, const Rows16& x, size_t r, bool five) {
    const size_t nb = x.nin / 256;
    const size_t bytes = five ? quant::Q5_K_TYPESIZE : quant::Q4_K_TYPESIZE;
    const int16_t* q = x.qs(r);
    const float* d = x.ds(r);
    const int32_t* s = x.sums(r);
    const __m256i m4 = _mm256_set1_epi8(15), one = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps(), mins = _mm256_setzero_ps();
    for (size_t b = 0; b < nb; ++b) {
        const uint8_t* p = row + b * bytes;
        const uint8_t* qs = p + (five ? 48 : 16);
        const __m256i hb = five ? load(p + 16) : _mm256_setzero_si256();
        const __m256i sums = sum8([&](size_t g) {
            const __m256i raw = load(qs + 32 * (g / 2));
            __m256i w = _mm256_and_si256(g % 2 ? _mm256_srli_epi16(raw, 4) : raw, m4);
            if (five)
                w = _mm256_or_si256(w, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, int(g)), one), 4));
            return dot16(w, q + b * 256 + g * 32);
        });
        const __m128i packed = _mm_loadu_si128((const __m128i*)(p + 4));
        const __m128i low = _mm_and_si128(packed, _mm_set1_epi8(63));
        const __m128i tail = _mm_srli_si128(packed, 8);
        const __m128i upper = _mm_and_si128(_mm_srli_epi16(packed, 2), _mm_set1_epi8(48));
        const __m128i scale_high = _mm_or_si128(_mm_and_si128(tail, _mm_set1_epi8(15)), upper);
        const __m128i min_high = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(tail, 4), _mm_set1_epi8(15)), _mm_srli_si128(upper, 4));
        const __m256 scales = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(low, scale_high)));
        const __m256 minima = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_unpacklo_epi32(_mm_srli_si128(low, 4), min_high)));
        const __m256 dx = _mm256_loadu_ps(d + b * 8);
        const __m256 ws = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(half(p)), scales), dx);
        // Scale the minimum before its activation sum, as the prompt dots do.
        const __m256 ms = _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(half(p + 2)), minima), dx);
        acc = _mm256_fmadd_ps(ws, _mm256_cvtepi32_ps(sums), acc);
        mins = _mm256_fmadd_ps(ms, _mm256_cvtepi32_ps(load(s + b * 8)), mins);
    }
    return hsum(acc) - hsum(mins);
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
        const uint8_t* p = row + b * quant::Q6_K_TYPESIZE;
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
// The prompt dots take a block of consecutive weight rows against n activation rows r[0..n).
// They walk the inner dimension 256 values at a time: each weight row's 256 are unpacked once into a buffer, then met by every column's 256, which stay in the first-level cache across the block's rows.
// Within a group of 32 values the products are summed exactly in integers; eight groups' sums then meet their eight scales in one vector multiply-add.
// Every (row, column) accumulates in the same order whatever the block and the other columns are, so a row computes the same alone or beside others.
constexpr size_t kCols = 32, kRows = 8;   // the columns and rows whose accumulators are held at once

// The first `n` of eight lanes, for a row whose last eight blocks are fewer.
inline __m256i first_lanes(size_t n) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
inline __m256i loada(const void* p) { return _mm256_load_si256((const __m256i*)p); }

inline __m256i widen_lo(__m256i w8) { return _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w8)); }
inline __m256i widen_hi(__m256i w8) { return _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w8, 1)); }
// 32 16-bit weights against 32 16-bit activations into eight 32-bit lanes.
inline __m256i dot16(const int16_t* w, const int16_t* x) {
    return _mm256_add_epi32(_mm256_madd_epi16(loada(w), _mm256_loadu_si256((const __m256i*)x)),
                            _mm256_madd_epi16(loada(w + 16), _mm256_loadu_si256((const __m256i*)(x + 16))));
}

// Q4_0, and with MIN Q4_1, whose nibbles are unsigned under a per-block minimum; eight blocks at a time.
template <bool MIN>
struct KQ4 {
    using X = Rows16;
    struct U { alignas(32) int16_t w[256]; __m256 ws, mw; __m256i lanes; size_t ng; };
    static size_t steps(size_t nin) { return (nin / 32 + 7) / 8; }
    static void unpack(const uint8_t* row, size_t t, size_t nin, U& u) {
        const size_t g0 = t * 8, bytes = MIN ? quant::Q4_1_TYPESIZE : quant::Q4_0_TYPESIZE;
        u.ng = std::min<size_t>(8, nin / 32 - g0);
        const __m256i eight = _mm256_set1_epi8(MIN ? 0 : 8);
        alignas(32) float ws[8] = {0.0f}, mw[8] = {0.0f};
        for (size_t g = 0; g < u.ng; ++g) {
            const uint8_t* p = row + (g0 + g) * bytes;
            const __m256i v = _mm256_sub_epi8(nibbles32(p + (MIN ? 4 : 2)), eight);
            _mm256_store_si256((__m256i*)(u.w + 32 * g), widen_lo(v));
            _mm256_store_si256((__m256i*)(u.w + 32 * g + 16), widen_hi(v));
            ws[g] = half(p);
            if (MIN) mw[g] = half(p + 2);
        }
        u.ws = _mm256_load_ps(ws);
        u.mw = _mm256_load_ps(mw);
        u.lanes = first_lanes(u.ng);
    }
    static void add(const U& u, const X& x, size_t rc, size_t t, __m256& acc, __m256& accm) {
        const int16_t* q = x.qs(rc) + t * 256;
        auto part = [&](size_t g) { return dot16(u.w + 32 * g, q + 32 * g); };
        const size_t ng = u.ng;
        const __m256i s = ng == 8 ? sum8(part) : sum8([&](size_t g) { return g < ng ? part(g) : _mm256_setzero_si256(); });
        const __m256 dx = _mm256_maskload_ps(x.ds(rc) + t * 8, u.lanes);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(u.ws, dx), acc);
        // The minimum meets the scale before the sum, so a zero minimum stays zero however large the activations (fused_dot_overflow).
        if (MIN) accm = _mm256_fmadd_ps(_mm256_mul_ps(u.mw, dx), _mm256_cvtepi32_ps(_mm256_maskload_epi32(x.sums(rc) + t * 8, u.lanes)), accm);
    }
    static float finish(__m256 acc, __m256 accm) { return MIN ? hsum(acc) + hsum(accm) : hsum(acc); }
};

// Q4_K and Q5_K: a 256-value block of eight groups of 32, each with a 6-bit scale and minimum under the block's two half scales; 32 bytes of nibbles hold groups 2j (low nibbles) and 2j + 1 (high).
template <bool FIVE>
struct KQ45_K {
    using X = Rows16;
    struct U { alignas(32) uint8_t w[256]; __m256 ws, mw; };
    static size_t steps(size_t nin) { return nin / 256; }
    static void unpack(const uint8_t* row, size_t t, size_t, U& u) {
        const uint8_t* p = row + t * (FIVE ? quant::Q5_K_TYPESIZE : quant::Q4_K_TYPESIZE);
        const __m256i m4 = _mm256_set1_epi8(0x0F), one = _mm256_set1_epi8(1);
        const float dd = half(p), dmin = half(p + 2);
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + (FIVE ? 48 : 16);
        const __m256i hb = FIVE ? load(p + 16) : _mm256_setzero_si256();
        alignas(32) float ws[8], mw[8];
        for (int j = 0; j < 4; ++j) {
            const __m256i raw = load(qs + 32 * j);
            __m256i lo = _mm256_and_si256(raw, m4), hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4);
            if (FIVE) {
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * j), one), 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hb, 2 * j + 1), one), 4));
            }
            _mm256_store_si256((__m256i*)(u.w + 64 * j), lo);
            _mm256_store_si256((__m256i*)(u.w + 64 * j + 32), hi);
            uint8_t s0, m0, s1, m1;
            quant::get_scale_min_k4(2 * j, sc, &s0, &m0);
            quant::get_scale_min_k4(2 * j + 1, sc, &s1, &m1);
            ws[2 * j] = dd * (float)s0;
            ws[2 * j + 1] = dd * (float)s1;
            mw[2 * j] = dmin * (float)m0;
            mw[2 * j + 1] = dmin * (float)m1;
        }
        u.ws = _mm256_load_ps(ws);
        u.mw = _mm256_load_ps(mw);
    }
    static void add(const U& u, const X& x, size_t rc, size_t t, __m256& acc, __m256& accm) {
        const int16_t* q = x.qs(rc) + t * 256;
        const __m256i s = sum8([&](size_t g) { return dot16(loada(u.w + 32 * g), q + 32 * g); });
        const __m256 dx = _mm256_loadu_ps(x.ds(rc) + t * 8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(u.ws, dx), acc);
        // The minimum meets the scale before the sum, so a zero minimum stays zero however large the activations (fused_dot_overflow).
        accm = _mm256_fmadd_ps(_mm256_mul_ps(u.mw, dx), _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(x.sums(rc) + t * 8))), accm);
    }
    static float finish(__m256 acc, __m256 accm) { return hsum(acc) - hsum(accm); }
};

// A pair of adjacent input features in each lane, for eight prompt columns at once.
struct PromptPairs {
    std::vector<int16_t> q;
    std::vector<float> d;
    std::vector<int32_t> sum;
    size_t cols = 0, groups = 0;
    void reset(const Rows16& x, size_t n) {
        cols = n;
        groups = x.nin / 32;
        const size_t tiles = (n + 7) / 8;
        q.resize(tiles * groups * 256);
        d.resize(tiles * groups * 8);
        sum.resize(tiles * groups * 8);
        for (size_t tile = 0; tile < tiles; ++tile)
            for (size_t g = 0; g < groups; ++g) {
                const size_t at = tile * groups + g;
                for (size_t c = 0; c < 8; ++c) {
                    const size_t r = tile * 8 + c;
                    d[at * 8 + c] = r < n ? x.ds(r)[g] : 0.0f;
                    sum[at * 8 + c] = r < n ? x.sums(r)[g] : 0;
                }
                if (tile * 8 + 8 <= n) {
                    for (size_t half = 0; half < 2; ++half) {
                        __m256i v[8], a[8], b[8];
                        for (size_t c = 0; c < 8; ++c)
                            v[c] = _mm256_loadu_si256((const __m256i*)(x.qs(tile * 8 + c) + g * 32 + half * 16));
                        for (size_t c = 0; c < 8; c += 2) {
                            a[c] = _mm256_unpacklo_epi32(v[c], v[c + 1]);
                            a[c + 1] = _mm256_unpackhi_epi32(v[c], v[c + 1]);
                        }
                        for (size_t c = 0; c < 8; c += 4) {
                            b[c] = _mm256_unpacklo_epi64(a[c], a[c + 2]);
                            b[c + 1] = _mm256_unpackhi_epi64(a[c], a[c + 2]);
                            b[c + 2] = _mm256_unpacklo_epi64(a[c + 1], a[c + 3]);
                            b[c + 3] = _mm256_unpackhi_epi64(a[c + 1], a[c + 3]);
                        }
                        for (size_t j = 0; j < 4; ++j) {
                            _mm256_storeu_si256((__m256i*)(q.data() + at * 256 + (half * 8 + j) * 16),
                                               _mm256_permute2x128_si256(b[j], b[j + 4], 0x20));
                            _mm256_storeu_si256((__m256i*)(q.data() + at * 256 + (half * 8 + j + 4) * 16),
                                               _mm256_permute2x128_si256(b[j], b[j + 4], 0x31));
                        }
                    }
                } else {
                    for (size_t c = 0; c < 8; ++c) {
                        const size_t r = tile * 8 + c;
                        for (size_t j = 0; j < 16; ++j) {
                            q[at * 256 + j * 16 + c * 2] = r < n ? x.qs(r)[g * 32 + j * 2] : 0;
                            q[at * 256 + j * 16 + c * 2 + 1] = r < n ? x.qs(r)[g * 32 + j * 2 + 1] : 0;
                        }
                    }
                }
            }
    }
};

template <bool FIVE, size_t R, size_t C = 1>
inline void prompt_pair_rows(const uint8_t* w, size_t row_bytes, const PromptPairs& x,
                             float* const* out, size_t o0, size_t c0) {
    __m256 acc[C][R][8], accm[C][R][8];
    for (size_t col = 0; col < C; ++col)
        for (size_t r = 0; r < R; ++r)
            for (size_t g = 0; g < 8; ++g) acc[col][r][g] = accm[col][r][g] = _mm256_setzero_ps();
    for (size_t t = 0; t < x.groups / 8; ++t) {
        alignas(32) int16_t expanded[R][256];
        alignas(32) float ws[R][8], mw[R][8];
        for (size_t r = 0; r < R; ++r) {
            typename KQ45_K<FIVE>::U u;
            KQ45_K<FIVE>::unpack(w + r * row_bytes, t, x.groups * 32, u);
            for (size_t g = 0; g < 8; ++g) {
                const __m256i bytes = loada(u.w + g * 32);
                _mm256_store_si256((__m256i*)(expanded[r] + g * 32), widen_lo(bytes));
                _mm256_store_si256((__m256i*)(expanded[r] + g * 32 + 16), widen_hi(bytes));
            }
            _mm256_store_ps(ws[r], u.ws);
            _mm256_store_ps(mw[r], u.mw);
        }
        for (size_t g = 0; g < 8; ++g) {
            const size_t at = (c0 / 8) * x.groups + t * 8 + g;
            const int16_t* q = x.q.data() + at * 256;
            __m256i a[C][R];
            for (size_t col = 0; col < C; ++col)
                for (size_t r = 0; r < R; ++r) a[col][r] = _mm256_setzero_si256();
#if defined(__GNUC__)
#pragma GCC unroll 1
#endif
            for (size_t j = 0; j < 16; j += 2) {
                __m256i x0[C], x1[C];
                for (size_t col = 0; col < C; ++col) {
                    const int16_t* p = q + col * x.groups * 256;
                    x0[col] = _mm256_loadu_si256((const __m256i*)(p + 16 * j));
                    x1[col] = _mm256_loadu_si256((const __m256i*)(p + 16 * (j + 1)));
                }
                for (size_t r = 0; r < R; ++r) {
                    uint32_t w0, w1;
                    std::memcpy(&w0, expanded[r] + g * 32 + 2 * j, sizeof(w0));
                    std::memcpy(&w1, expanded[r] + g * 32 + 2 * (j + 1), sizeof(w1));
                    const __m256i v0 = _mm256_set1_epi32(int(w0));
                    const __m256i v1 = _mm256_set1_epi32(int(w1));
                    for (size_t col = 0; col < C; ++col) {
                        a[col][r] = _mm256_add_epi32(a[col][r], _mm256_madd_epi16(v0, x0[col]));
                        a[col][r] = _mm256_add_epi32(a[col][r], _mm256_madd_epi16(v1, x1[col]));
                    }
                }
            }
            for (size_t col = 0; col < C; ++col) {
                const size_t pos = (at + col * x.groups) * 8;
                const __m256 dx = _mm256_loadu_ps(x.d.data() + pos);
                const __m256 sums = _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(x.sum.data() + pos)));
                for (size_t r = 0; r < R; ++r) {
                    acc[col][r][g] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a[col][r]),
                        _mm256_mul_ps(_mm256_set1_ps(ws[r][g]), dx), acc[col][r][g]);
                    accm[col][r][g] = _mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(mw[r][g]), dx), sums, accm[col][r][g]);
                }
            }
        }
    }
    const auto finish = [] (const __m256* a) {
        return _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a[0], a[4]), _mm256_add_ps(a[2], a[6])),
                             _mm256_add_ps(_mm256_add_ps(a[1], a[5]), _mm256_add_ps(a[3], a[7])));
    };
    for (size_t col = 0; col < C; ++col) {
        for (size_t r = 0; r < R; ++r) {
            alignas(32) float y[8];
            _mm256_store_ps(y, _mm256_sub_ps(finish(acc[col][r]), finish(accm[col][r])));
            const size_t first = c0 + col * 8;
            for (size_t k = 0; k < std::min<size_t>(8, x.cols - first); ++k) out[first + k][o0 + r] = y[k];
        }
    }
}

template <bool FIVE>
inline void prompt_pairs(const uint8_t* w, size_t row_bytes, size_t nrows, const PromptPairs& x,
                         float* const* out, size_t o0) {
    size_t c0 = 0;
    for (; c0 + 16 <= x.cols; c0 += 16) {
        size_t i = 0;
        for (; i + 2 <= nrows; i += 2)
            prompt_pair_rows<FIVE, 2, 2>(w + i * row_bytes, row_bytes, x, out, o0 + i, c0);
        for (; i < nrows; ++i)
            prompt_pair_rows<FIVE, 1, 2>(w + i * row_bytes, row_bytes, x, out, o0 + i, c0);
    }
    for (; c0 < x.cols; c0 += 8) {
        size_t i = 0;
        for (; i + 4 <= nrows; i += 4)
            prompt_pair_rows<FIVE, 4>(w + i * row_bytes, row_bytes, x, out, o0 + i, c0);
        for (; i < nrows; ++i)
            prompt_pair_rows<FIVE, 1>(w + i * row_bytes, row_bytes, x, out, o0 + i, c0);
    }
}

// Q6_K: 256 values in two halves of 128, each from 64 bytes of low nibbles and 32 bytes of top two bits, minus 32, with a signed 8-bit scale per 16 values.
// A group of 32 against 16-bit activations is two sums of 16, one per scale, so the eight groups give two vectors of sums.
struct KQ6_K {
    using X = Rows16;
    struct U { alignas(32) int16_t w[256]; __m256 wl, wh; };
    static size_t steps(size_t nin) { return nin / 256; }
    static void unpack(const uint8_t* row, size_t t, size_t, U& u) {
        const uint8_t* p = row + t * quant::Q6_K_TYPESIZE;
        const __m256i m4 = _mm256_set1_epi8(0x0F), m2 = _mm256_set1_epi8(3), off = _mm256_set1_epi8(32);
        const float dd = half(p + 208);
        const int8_t* sc = (const int8_t*)(p + 192);
        alignas(32) float wl[8], wh[8];
        for (int h = 0; h < 2; ++h) {
            const uint8_t* ql = p + 64 * h;
            const uint8_t* qh = p + 128 + 32 * h;
            const __m256i A = load(ql), B = load(ql + 32), H = load(qh);
            const __m256i v4[4] = {
                _mm256_or_si256(_mm256_and_si256(A, m4), _mm256_slli_epi16(_mm256_and_si256(H, m2), 4)),
                _mm256_or_si256(_mm256_and_si256(B, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 2), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(A, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 4), m2), 4)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(B, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(H, 6), m2), 4)),
            };
            for (int k = 0; k < 4; ++k) {
                const int g = h * 4 + k;
                const __m256i v = _mm256_sub_epi8(v4[k], off);
                _mm256_store_si256((__m256i*)(u.w + 32 * g), widen_lo(v));
                _mm256_store_si256((__m256i*)(u.w + 32 * g + 16), widen_hi(v));
                wl[g] = dd * (float)sc[2 * g];
                wh[g] = dd * (float)sc[2 * g + 1];
            }
        }
        u.wl = _mm256_load_ps(wl);
        u.wh = _mm256_load_ps(wh);
    }
    static void add(const U& u, const X& x, size_t rc, size_t t, __m256& acc, __m256&) {
        const int16_t* q = x.qs(rc) + t * 256;
        const __m256i sl = sum8([&](size_t g) { return _mm256_madd_epi16(loada(u.w + 32 * g), _mm256_loadu_si256((const __m256i*)(q + 32 * g))); });
        const __m256i sh = sum8([&](size_t g) { return _mm256_madd_epi16(loada(u.w + 32 * g + 16), _mm256_loadu_si256((const __m256i*)(q + 32 * g + 16))); });
        const __m256 dx = _mm256_loadu_ps(x.ds(rc) + t * 8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sl), _mm256_mul_ps(u.wl, dx), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sh), _mm256_mul_ps(u.wh, dx), acc);
    }
    static float finish(__m256 acc, __m256) { return hsum(acc); }
};

// `nrows` weight rows from `w`, `row_bytes` apart, against activation rows r[0..n): out[c][o0 + i] is row i against column c.
template <class K>
inline void block_dots(const uint8_t* w, size_t row_bytes, size_t nrows, const typename K::X& x, const size_t* r, size_t n,
                       float* const* out, size_t o0) {
    const size_t steps = K::steps(x.nin);
    __m256 acc[kRows][kCols], accm[kRows][kCols];
    typename K::U u;
    for (size_t c0 = 0; c0 < n; c0 += kCols) {
        const size_t m = std::min(kCols, n - c0);
        for (size_t i0 = 0; i0 < nrows; i0 += kRows) {
            const size_t mr = std::min(kRows, nrows - i0);
            for (size_t i = 0; i < mr; ++i)
                for (size_t c = 0; c < m; ++c) acc[i][c] = accm[i][c] = _mm256_setzero_ps();
            for (size_t t = 0; t < steps; ++t)
                for (size_t i = 0; i < mr; ++i) {
                    K::unpack(w + (i0 + i) * row_bytes, t, x.nin, u);
                    for (size_t c = 0; c < m; ++c) K::add(u, x, r[c0 + c], t, acc[i][c], accm[i][c]);
                }
            for (size_t i = 0; i < mr; ++i)
                for (size_t c = 0; c < m; ++c) out[c0 + c][o0 + i0 + i] = K::finish(acc[i][c], accm[i][c]);
        }
    }
}
// Whether a type has a packed dot here.
inline bool has_dot(uint32_t type) {
    return type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1 ||
           type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q6_K || type == quant::GGML_TYPE_MXFP4;
}
// The activation rows of a call, quantized once; `par(rows, fn)` fills disjoint ranges, from several threads if it likes.
struct Activations {
    Rows16 x16;
    const float* src = nullptr;
    size_t rows = 0, nin = 0;
    bool ready = false;
    void reset(const float* x, size_t n_rows, size_t width) {
        src = x;
        rows = n_rows;
        nin = width;
        ready = false;
    }
    template <class Par>
    void prepare(const Par& par) {
        if (ready) return;
        const size_t per_row = nin / 32;
        size_rows(rows, nin, x16);
        par(rows, [&](size_t r0, size_t r1) { quantize16(src, r0 * per_row, r1 * per_row, x16); });
        ready = true;
    }
};

// `nrows` rows of a type's matrix from `w` against activation rows r[0..n), out[c][o0 + i] for row i and column c; each lands as it would alone.
inline void dot_block(uint32_t type, const uint8_t* w, size_t row_bytes, size_t nrows, const Activations& x, const size_t* r, size_t n,
                      float* const* out, size_t o0) {
    switch (type) {
    case quant::GGML_TYPE_Q4_0: block_dots<KQ4<false>>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    case quant::GGML_TYPE_Q4_1: block_dots<KQ4<true>>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    case quant::GGML_TYPE_Q4_K: block_dots<KQ45_K<false>>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    case quant::GGML_TYPE_Q5_K: block_dots<KQ45_K<true>>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    case quant::GGML_TYPE_Q6_K: block_dots<KQ6_K>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    default: throw std::runtime_error("backend: unsupported packed dot type");
    }
}

// A generated token's row: the decode dots.
inline float dot(uint32_t type, const uint8_t* row, const Activations& x, size_t r) {
    switch (type) {
    case quant::GGML_TYPE_Q4_0: return dot_q4_0(row, x.x16, r);
    case quant::GGML_TYPE_Q4_1: return dot_q4_1(row, x.x16, r);
    case quant::GGML_TYPE_Q4_K: return dot_q45_K(row, x.x16, r, false);
    case quant::GGML_TYPE_Q5_K: return dot_q45_K(row, x.x16, r, true);
    case quant::GGML_TYPE_Q6_K: return dot_q6_K(row, x.x16, r);
    case quant::GGML_TYPE_MXFP4: return dot_mxfp4(row, x.x16, r);
    default: throw std::runtime_error("backend: unsupported packed dot type");
    }
}

} // namespace q8
} // namespace backend
