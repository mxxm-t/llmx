#pragma once
// CPU dots over activations quantized per block of 32; weights remain packed and integer sums are scaled once per block (docs/src/backends-cpu.md).
// Q4_0, Q4_1 and Q6_K use 16-bit activations for ranking precision; Q8_0, Q4_K and Q5_K use 8-bit activations.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <immintrin.h>

#include "core/fp16.hpp"
#include "format/gguf.hpp"
#include "quant/k_quants.hpp"

namespace backend {
namespace q8 {

// Each block stores its integers, reconstruction scale and integer sum, which types with a minimum require.
struct Rows {
    std::vector<int8_t> q;
    std::vector<float> d;
    std::vector<int32_t> sum;
    size_t nin = 0;
    const int8_t* qs(size_t r) const { return q.data() + r * nin; }
    const float* ds(size_t r) const { return d.data() + r * (nin / 32); }
    const int32_t* sums(size_t r) const { return sum.data() + r * (nin / 32); }
};

// The same block representation with 16-bit integers.
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
template <class R>
inline void size_rows(size_t rows, size_t nin, R& out) {
    out.nin = nin;
    out.q.resize(rows * nin);
    out.d.resize(rows * nin / 32);
    out.sum.resize(rows * nin / 32);
}

// A tiny block can overflow the float reciprocal; round against a representable scale without losing zeros or signs.
template <int Levels, class R>
inline void quantize_small(const float* x, size_t b, float top, R& out) {
    float d = top / float(Levels);
    if (double(d) * Levels < double(top))
        d = std::nextafter(d, std::numeric_limits<float>::max());
    int32_t sum = 0;
    using Quant = typename decltype(out.q)::value_type;
    for (size_t i = 0; i < 32; ++i) {
        const double value = double(x[i]) / double(d);
        const double low = std::floor(value);
        int32_t q = int32_t(low);
        const double tail = value - low;
        if (tail > 0.5 || (tail == 0.5 && q % 2 != 0)) ++q;
        out.q[b * 32 + i] = Quant(q);
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
        if (!std::isfinite(id)) { quantize_small<32767>(p, b, top, out); continue; }
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

// Quantize blocks b0 .. b1 of 32 values, rounding to nearest with ties to even.
inline void quantize(const float* x, size_t b0, size_t b1, Rows& out) {
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
        const float d = top / 127.0f, id = top > 0.0f ? 127.0f / top : 0.0f;
        if (!std::isfinite(id)) { quantize_small<127>(p, b, top, out); continue; }
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

// The decode dots: one row of `nin` weights against activation row r, each block unpacked straight into the multiply, which a lone generated token wants since nothing shares the unpacking.
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
// The prompt dots take a block of consecutive weight rows against n activation rows r[0..n).
// They walk the inner dimension 256 values at a time: each weight row's 256 are unpacked once into a buffer, then met by every column's 256, which stay in the first-level cache across the block's rows.
// Within a group of 32 values the products are summed exactly in integers; eight groups' sums then meet their eight scales in one vector multiply-add.
// Every (row, column) accumulates in the same order whatever the block and the other columns are, so a row computes the same alone or beside others.
constexpr size_t kCols = 32, kRows = 8;   // the columns and rows whose accumulators are held at once

// The eight groups' vectors f(0) .. f(7), eight 32-bit lanes each, to their eight sums, one lane each; exact, so the order is free.
// Built as a tree from the calls, so the vectors stay in registers.
template <class F>
inline __m256i sum8(const F& f) {
    const __m256i a = _mm256_hadd_epi32(f(0), f(1)), b = _mm256_hadd_epi32(f(2), f(3));
    const __m256i c = _mm256_hadd_epi32(f(4), f(5)), e = _mm256_hadd_epi32(f(6), f(7));
    const __m256i ab = _mm256_hadd_epi32(a, b), ce = _mm256_hadd_epi32(c, e);
    return _mm256_add_epi32(_mm256_permute2x128_si256(ab, ce, 0x20), _mm256_permute2x128_si256(ab, ce, 0x31));
}
// The first `n` of eight lanes, for a row whose last eight blocks are fewer.
inline __m256i first_lanes(size_t n) {
    return _mm256_cmpgt_epi32(_mm256_set1_epi32((int)n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
}
inline __m256i loada(const void* p) { return _mm256_load_si256((const __m256i*)p); }

// Q8_0: eight blocks of 32 at a time, the weights' signs moved onto the activations, then unsigned by signed bytes.
struct KQ8_0 {
    using X = Rows;
    struct U { alignas(32) int8_t w[256], a[256]; __m256 ws; __m256i lanes; size_t ng; };
    static size_t steps(size_t nin) { return (nin / 32 + 7) / 8; }
    static void unpack(const uint8_t* row, size_t t, size_t nin, U& u) {
        const size_t g0 = t * 8;
        u.ng = std::min<size_t>(8, nin / 32 - g0);
        alignas(32) float ws[8] = {0.0f};
        for (size_t g = 0; g < u.ng; ++g) {
            const uint8_t* p = row + (g0 + g) * gguf::Q8_0_TYPESIZE;
            const __m256i v = load(p + 2);
            _mm256_store_si256((__m256i*)(u.w + 32 * g), v);
            _mm256_store_si256((__m256i*)(u.a + 32 * g), _mm256_sign_epi8(v, v));
            ws[g] = half(p);
        }
        u.ws = _mm256_load_ps(ws);
        u.lanes = first_lanes(u.ng);
    }
    static void add(const U& u, const X& x, size_t rc, size_t t, __m256& acc, __m256&) {
        const int8_t* q = x.qs(rc) + t * 256;
        auto part = [&](size_t g) { return dot_us(loada(u.a + 32 * g), _mm256_sign_epi8(load(q + 32 * g), loada(u.w + 32 * g))); };
        const size_t ng = u.ng;
        const __m256i s = ng == 8 ? sum8(part) : sum8([&](size_t g) { return g < ng ? part(g) : _mm256_setzero_si256(); });
        const __m256 dx = _mm256_maskload_ps(x.ds(rc) + t * 8, u.lanes);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(u.ws, dx), acc);
    }
    static float finish(__m256 acc, __m256) { return hsum(acc); }
};

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
        const size_t g0 = t * 8, bytes = MIN ? gguf::Q4_1_TYPESIZE : gguf::Q4_0_TYPESIZE;
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
    using X = Rows;
    struct U { alignas(32) uint8_t w[256]; __m256 ws, mw; };
    static size_t steps(size_t nin) { return nin / 256; }
    static void unpack(const uint8_t* row, size_t t, size_t, U& u) {
        const uint8_t* p = row + t * (FIVE ? gguf::Q5_K_TYPESIZE : gguf::Q4_K_TYPESIZE);
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
        const int8_t* q = x.qs(rc) + t * 256;
        const __m256i s = sum8([&](size_t g) { return dot_us(loada(u.w + 32 * g), load(q + 32 * g)); });
        const __m256 dx = _mm256_loadu_ps(x.ds(rc) + t * 8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_mul_ps(u.ws, dx), acc);
        // The minimum meets the scale before the sum, so a zero minimum stays zero however large the activations (fused_dot_overflow).
        accm = _mm256_fmadd_ps(_mm256_mul_ps(u.mw, dx), _mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(x.sums(rc) + t * 8))), accm);
    }
    static float finish(__m256 acc, __m256 accm) { return hsum(acc) - hsum(accm); }
};

// Q6_K: 256 values in two halves of 128, each from 64 bytes of low nibbles and 32 bytes of top two bits, minus 32, with a signed 8-bit scale per 16 values.
// A group of 32 against 16-bit activations is two sums of 16, one per scale, so the eight groups give two vectors of sums.
struct KQ6_K {
    using X = Rows16;
    struct U { alignas(32) int16_t w[256]; __m256 wl, wh; };
    static size_t steps(size_t nin) { return nin / 256; }
    static void unpack(const uint8_t* row, size_t t, size_t, U& u) {
        const uint8_t* p = row + t * gguf::Q6_K_TYPESIZE;
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
// Whether a type has a dot here, which activations it reads, and the dot itself.
inline bool has_dot(uint32_t type) {
    return type == gguf::GGML_TYPE_Q8_0 || type == gguf::GGML_TYPE_Q4_0 || type == gguf::GGML_TYPE_Q4_1 ||
           type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K;
}
inline bool reads16(uint32_t type) {
    return type == gguf::GGML_TYPE_Q4_0 || type == gguf::GGML_TYPE_Q4_1 || type == gguf::GGML_TYPE_Q6_K;
}

// The activation rows of a call, each precision quantized once when a type first reads it; `par(rows, fn)` runs fn over ranges of rows, from several threads if it likes.
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
    template <class Par>
    void prepare(uint32_t type, const Par& par) {
        const size_t per_row = nin / 32;
        if (reads16(type)) {
            if (has16) return;
            size_rows(rows, nin, x16);
            par(rows, [&](size_t r0, size_t r1) { quantize16(src, r0 * per_row, r1 * per_row, x16); });
            has16 = true;
        } else if (!has8) {
            size_rows(rows, nin, x8);
            par(rows, [&](size_t r0, size_t r1) { quantize(src, r0 * per_row, r1 * per_row, x8); });
            has8 = true;
        }
    }
    void prepare(uint32_t type) {
        prepare(type, [](size_t n, const auto& fn) { fn(size_t(0), n); });
    }
};

// `nrows` rows of a type's matrix from `w` against activation rows r[0..n), out[c][o0 + i] for row i and column c; each lands as it would alone.
inline void dot_block(uint32_t type, const uint8_t* w, size_t row_bytes, size_t nrows, const Activations& x, const size_t* r, size_t n,
                      float* const* out, size_t o0) {
    switch (type) {
    case gguf::GGML_TYPE_Q8_0: block_dots<KQ8_0>(w, row_bytes, nrows, x.x8, r, n, out, o0); break;
    case gguf::GGML_TYPE_Q4_0: block_dots<KQ4<false>>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    case gguf::GGML_TYPE_Q4_1: block_dots<KQ4<true>>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    case gguf::GGML_TYPE_Q4_K: block_dots<KQ45_K<false>>(w, row_bytes, nrows, x.x8, r, n, out, o0); break;
    case gguf::GGML_TYPE_Q5_K: block_dots<KQ45_K<true>>(w, row_bytes, nrows, x.x8, r, n, out, o0); break;
    default: block_dots<KQ6_K>(w, row_bytes, nrows, x.x16, r, n, out, o0); break;
    }
}

// A generated token's row: the decode dots.
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
