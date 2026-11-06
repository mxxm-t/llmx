#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// DeepSeek-V4.1's storage formats, quantized and dequantized in F32 as its kernels do (docs/DEEPSEEK4.md, section 4.5).
// Codes come from exact comparisons and a value is one F32 multiply of code and scale, so every backend stores the same bytes.
namespace quant::model_formats {

constexpr float FP8_MAX = 448.0f, FP4_MAX = 6.0f;

inline uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

inline float from_bits(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

// The exponent field less 127, plus one where any mantissa bit is set: an exact power of two is its own ceiling, and a subnormal gives -126.
inline int ceil_log2(float x) {
    const uint32_t u = bits(x);
    return int((u >> 23) & 0xFF) - 127 + ((u & 0x7FFFFF) != 0);
}

// 2^e built from its exponent field, for e in [-126, 127].
inline float pow2(int e) { return from_bits(uint32_t(e + 127) << 23); }

// The power-of-two scale of a block whose largest magnitude is `amax`, from one F32 multiply by the format's inverse maximum.
inline float power_scale(float amax, float max_inv) { return pow2(ceil_log2(amax * max_inv)); }

// E4M3 without infinities: bias 7, 0x7E the largest finite value 448, 0x7F and 0xFF NaN.
inline float e4m3_value(uint8_t c) {
    const unsigned e = (c >> 3) & 15, m = c & 7;
    float v;
    if (e == 15 && m == 7) v = NAN;
    else v = e ? std::ldexp(float(8 + m), int(e) - 10) : std::ldexp(float(m), -9);
    return c & 0x80 ? -v : v;
}

// The nearest E4M3 code, ties to even, saturating at 448 as the kernels' conversion does, NaN to 0x7F; the sign of a zero kept.
inline uint8_t e4m3_code(float x) {
    const uint8_t sign = std::signbit(x) ? 0x80 : 0;
    if (std::isnan(x)) return 0x7F;
    const float a = std::fabs(x);
    if (a >= 464.0f) return uint8_t(sign | 0x7E);
    if (a < 0.015625f) return uint8_t(sign | unsigned(std::nearbyint(a * 512.0f)));
    int e;
    std::frexp(a, &e);
    // a = t * 2^(e - 4) with t in [8, 16), rounded to the 3 bits past the leading one; a carry to 16 is the next exponent's code.
    const unsigned t = unsigned(std::nearbyint(std::ldexp(a, 4 - e)));
    return uint8_t(sign | ((unsigned(e + 6) << 3) + (t - 8)));
}

// E2M1: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4 and 6 under codes 0 to 7, the sign in bit 3, each zero keeping its sign.
inline float e2m1_value(uint8_t c) {
    static constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = magnitudes[c & 7];
    return c & 8 ? -v : v;
}

// The nearest E2M1 code of a value already clamped to [-6, 6], ties to the even code, so up at 0.75, 1.75 and 3.5 and down at the other midpoints.
inline uint8_t e2m1_code(float x) {
    const float a = std::fabs(x);
    const unsigned m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return uint8_t((std::signbit(x) ? 8 : 0) | m);
}

inline float clamp(float x, float limit) { return x < -limit ? -limit : (x > limit ? limit : x); }

inline float block_amax(const float* x, size_t n) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    return amax;
}

// The window's FP8 in place, a block of n values under the scale 2^ceil(log2(amax / 448)) with amax at least 1e-4.
inline void qdq_fp8_power(float* x, size_t n) {
    const float s = power_scale(std::fmax(block_amax(x, n), 1e-4f), 1.0f / FP8_MAX);
    for (size_t i = 0; i < n; ++i) x[i] = e4m3_value(e4m3_code(clamp(x[i] / s, FP8_MAX))) * s;
}

// The index keys' and queries' FP4 in place, a block of n values under the scale 2^ceil(log2(amax / 6)) with amax at least 6 * 2^-126.
inline void qdq_fp4_power(float* x, size_t n) {
    const float s = power_scale(std::fmax(block_amax(x, n), 6.0f * 0x1p-126f), 1.0f / FP4_MAX);
    for (size_t i = 0; i < n; ++i) x[i] = e2m1_value(e2m1_code(clamp(x[i] / s, FP4_MAX))) * s;
}

// The entries' FP4 in place, a block of n values under the E4M3 scale nearest amax / 6, with amax at least 6 * 2^-9 so the scale is never zero.
inline void qdq_fp4_e4m3(float* x, size_t n) {
    const float s = e4m3_value(e4m3_code(std::fmax(block_amax(x, n), 6.0f * 0x1p-9f) / FP4_MAX));
    for (size_t i = 0; i < n; ++i) x[i] = e2m1_value(e2m1_code(clamp(x[i] / s, FP4_MAX))) * s;
}

} // namespace quant::model_formats
