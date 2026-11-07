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

// The packed blocks of each format, every block whole bytes and a row of whole blocks a multiple of 4 bytes (docs/DEEPSEEK4.md, section 4.3).
// The window's FP8: an E8M0 scale byte, then 32 E4M3 codes.
constexpr size_t FP8_POWER_VALUES = 32, FP8_POWER_BYTES = 33;
// The index keys' FP4 in the GGUF MXFP4 block: an E8M0 scale byte, then 16 bytes whose low nibbles are values 0 to 15 and high nibbles 16 to 31.
constexpr size_t FP4_POWER_VALUES = 32, FP4_POWER_BYTES = 17;
// The entries' FP4: an E4M3 scale byte, then 8 bytes laid out as the MXFP4 block's, low nibbles values 0 to 7 and high nibbles 8 to 15.
constexpr size_t FP4_E4M3_VALUES = 16, FP4_E4M3_BYTES = 9;

// An E8M0 byte's scale, 2^(byte - 127) built from its exponent field, the power scale it was written from.
inline float e8m0_value(uint8_t e) { return from_bits(uint32_t(e) << 23); }
inline uint8_t e8m0_code(float s) { return uint8_t(bits(s) >> 23); }

// The window's FP8, a block of n values under the scale 2^ceil(log2(amax / 448)) with amax at least 1e-4.
inline void pack_fp8_power(const float* x, uint8_t* out) {
    const float s = power_scale(std::fmax(block_amax(x, FP8_POWER_VALUES), 1e-4f), 1.0f / FP8_MAX);
    out[0] = e8m0_code(s);
    for (size_t i = 0; i < FP8_POWER_VALUES; ++i) out[1 + i] = e4m3_code(clamp(x[i] / s, FP8_MAX));
}
inline float unpack_fp8_power(const uint8_t* block, size_t i) { return e4m3_value(block[1 + i]) * e8m0_value(block[0]); }

// The index keys' and queries' FP4, a block under the scale 2^ceil(log2(amax / 6)) with amax at least 6 * 2^-126.
inline void pack_fp4_power(const float* x, uint8_t* out) {
    const float s = power_scale(std::fmax(block_amax(x, FP4_POWER_VALUES), 6.0f * 0x1p-126f), 1.0f / FP4_MAX);
    out[0] = e8m0_code(s);
    for (size_t i = 0; i < FP4_POWER_VALUES / 2; ++i)
        out[1 + i] = uint8_t(e2m1_code(clamp(x[i] / s, FP4_MAX)) | e2m1_code(clamp(x[i + FP4_POWER_VALUES / 2] / s, FP4_MAX)) << 4);
}
inline float unpack_fp4_power(const uint8_t* block, size_t i) {
    const uint8_t b = block[1 + i % (FP4_POWER_VALUES / 2)];
    return e2m1_value(uint8_t(i < FP4_POWER_VALUES / 2 ? b & 15 : b >> 4)) * e8m0_value(block[0]);
}

// The entries' FP4, a block under the E4M3 scale nearest amax / 6, with amax at least 6 * 2^-9 so the scale is never zero.
inline void pack_fp4_e4m3(const float* x, uint8_t* out) {
    out[0] = e4m3_code(std::fmax(block_amax(x, FP4_E4M3_VALUES), 6.0f * 0x1p-9f) / FP4_MAX);
    const float s = e4m3_value(out[0]);
    for (size_t i = 0; i < FP4_E4M3_VALUES / 2; ++i)
        out[1 + i] = uint8_t(e2m1_code(clamp(x[i] / s, FP4_MAX)) | e2m1_code(clamp(x[i + FP4_E4M3_VALUES / 2] / s, FP4_MAX)) << 4);
}
inline float unpack_fp4_e4m3(const uint8_t* block, size_t i) {
    const uint8_t b = block[1 + i % (FP4_E4M3_VALUES / 2)];
    return e2m1_value(uint8_t(i < FP4_E4M3_VALUES / 2 ? b & 15 : b >> 4)) * e4m3_value(block[0]);
}

// Each format quantized and dequantized in place, a block at a time: its packed block's values, so the stored F32 and the packed bytes decode alike.
inline void qdq_fp8_power(float* x) {
    uint8_t b[FP8_POWER_BYTES];
    pack_fp8_power(x, b);
    for (size_t i = 0; i < FP8_POWER_VALUES; ++i) x[i] = unpack_fp8_power(b, i);
}
inline void qdq_fp4_power(float* x) {
    uint8_t b[FP4_POWER_BYTES];
    pack_fp4_power(x, b);
    for (size_t i = 0; i < FP4_POWER_VALUES; ++i) x[i] = unpack_fp4_power(b, i);
}
inline void qdq_fp4_e4m3(float* x) {
    uint8_t b[FP4_E4M3_BYTES];
    pack_fp4_e4m3(x, b);
    for (size_t i = 0; i < FP4_E4M3_VALUES; ++i) x[i] = unpack_fp4_e4m3(b, i);
}

} // namespace quant::model_formats
