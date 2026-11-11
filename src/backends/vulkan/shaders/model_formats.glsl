// DeepSeek-V4.1's storage formats as quant/model_formats.hpp gives them: the codes, the scales and their packed blocks, each step the CPU's.
// A packed block is its scale byte, then the codes, one a byte for FP8 and two a byte for FP4 with the low nibbles the block's first half; format 3 is F32 values.
#ifndef LLMX_MODEL_FORMATS_GLSL
#define LLMX_MODEL_FORMATS_GLSL
const float FP8_MAX = 448.0, FP4_MAX = 6.0;
// The significand of a nonzero finite magnitude as an integer in [2^23, 2^24) and its exponent, so a = m * 2^e.
void split(float a, out uint m, out int e) {
    uint u = floatBitsToUint(a);
    uint field = u >> 23;
    m = u & 0x7FFFFFu;
    if (field != 0u) { m |= 0x800000u; e = int(field) - 150; return; }
    int lead = findMSB(m);
    m <<= uint(23 - lead);
    e = -149 - (23 - lead);
}
// a / b correctly rounded to nearest even for finite a and b > 0 whose quotient is normal; a quotient below the normal range keeps only its sign and smallness, which no code reads.
float div_rn(float a, float b) {
    float mag = abs(a);
    if (mag == 0.0) return a;
    uint ma, mb;
    int ea, eb;
    split(mag, ma, ea);
    split(b, mb, eb);
    int e = ea - eb;
    if (ma < mb) { ma <<= 1u; e -= 1; }
    // 25 quotient bits, the leading one first, then the remainder as the sticky bit.
    uint q = 0u, rem = ma;
    for (int i = 0; i < 25; ++i) {
        q <<= 1u;
        if (rem >= mb) { rem -= mb; q |= 1u; }
        rem <<= 1u;
    }
    uint kept = q >> 1u;
    if ((q & 1u) != 0u && (rem != 0u || (kept & 1u) != 0u)) kept += 1u;
    float result = ldexp(float(kept), e - 23);
    return a < 0.0 ? -result : result;
}
int ceil_log2(float v) {
    uint u = floatBitsToUint(v);
    return int((u >> 23) & 0xFFu) - 127 + int((u & 0x7FFFFFu) != 0u);
}
float pow2(int e) { return uintBitsToFloat(uint(e + 127) << 23); }
float e4m3_value(uint c) {
    uint e = (c >> 3) & 15u, m = c & 7u;
    float v = e == 15u && m == 7u ? uintBitsToFloat(0x7FC00000u) : e != 0u ? ldexp(float(8u + m), int(e) - 10) : ldexp(float(m), -9);
    return (c & 0x80u) != 0u ? -v : v;
}
uint e4m3_code(float v) {
    uint sign = (floatBitsToUint(v) >> 31) != 0u ? 0x80u : 0u;
    float a = abs(v);
    if (a >= 464.0) return sign | 0x7Eu;
    if (a < 0.015625) return sign | uint(roundEven(a * 512.0));
    int e;
    frexp(a, e);
    uint t = uint(roundEven(ldexp(a, 4 - e)));
    return sign | ((uint(e + 6) << 3) + (t - 8u));
}
float e2m1_value(uint c) {
    const float magnitudes[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    float v = magnitudes[c & 7u];
    return (c & 8u) != 0u ? -v : v;
}
uint e2m1_code(float v) {
    float a = abs(v);
    uint m = uint(a > 0.25) + uint(a >= 0.75) + uint(a > 1.25) + uint(a >= 1.75) + uint(a > 2.5) + uint(a >= 3.5) + uint(a > 5.0);
    return ((floatBitsToUint(v) >> 31) != 0u ? 8u : 0u) | m;
}
float clamp_to(float v, float limit) { return v < -limit ? -limit : (v > limit ? limit : v); }
uint format_values(uint format) { return format == 2u ? 16u : 32u; }
uint format_bytes(uint format) { return format == 0u ? 33u : format == 1u ? 17u : 9u; }
// The byte at `at` of the words from `base`, read through a uint view of a buffer that FORMAT_WORD(i) names.
#define FORMAT_BYTE(base, at) ((FORMAT_WORD((base) + ((at) >> 2u)) >> (((at) & 3u) * 8u)) & 0xFFu)
// Value d of a packed row from word `base`: its code times its block's scale, the one multiply model_format_qdq gives.
#define FORMAT_VALUE(base, d, format, out_value)                                                                           \
    {                                                                                                                       \
        uint blk_ = (d) / format_values(format), i_ = (d) - blk_ * format_values(format), half_ = format_values(format) / 2u; \
        uint at_ = blk_ * format_bytes(format), scale_ = FORMAT_BYTE(base, at_);                                           \
        if ((format) == 0u) out_value = e4m3_value(FORMAT_BYTE(base, at_ + 1u + i_)) * uintBitsToFloat(scale_ << 23);      \
        else {                                                                                                              \
            uint b_ = FORMAT_BYTE(base, at_ + 1u + (i_ < half_ ? i_ : i_ - half_));                                         \
            out_value = e2m1_value(i_ < half_ ? b_ & 15u : b_ >> 4u) * ((format) == 1u ? uintBitsToFloat(scale_ << 23) : e4m3_value(scale_)); \
        }                                                                                                                   \
    }
// q from qb dot the row from word w0 in d order, for a shader holding q, p.dim and the 272-entry code table lut (E4M3 then E2M1 values).
// Four blocks' words load at once and a value is its code's table entry times its block's scale, the product the per-value decode gives.
#define WORD_BYTE(wd, o) ((wd[(o) >> 2u] >> (((o) & 3u) * 8u)) & 0xFFu)
#define FORMAT_DOT(DOT, WORDS, VALUE, F, VB, BB)                                                                                        \
    float DOT(uint qb, uint w0) {                                                                                                      \
        precise float dot = 0.0;                                                                                                       \
        uint d = 0u, w = w0;                                                                                                           \
        if (F != 3u)                                                                                                                   \
            for (; d + 4u * VB <= p.dim; d += 4u * VB, w += BB) {                                                                      \
                uint wd[BB];                                                                                                           \
                [[unroll]] for (uint j = 0u; j < BB; ++j) wd[j] = WORDS[w + j];                                                        \
                [[unroll]] for (uint b = 0u; b < 4u; ++b) {                                                                            \
                    uint sb = WORD_BYTE(wd, b * BB);                                                                                   \
                    float sc = F == 2u ? lut[sb] : uintBitsToFloat(sb << 23);                                                          \
                    [[unroll]] for (uint i = 0u; i < VB; ++i) {                                                                        \
                        uint c = F == 0u ? WORD_BYTE(wd, b * BB + 1u + i)                                                              \
                                         : 256u + (i < VB / 2u ? WORD_BYTE(wd, b * BB + 1u + i) & 15u                                  \
                                                                : WORD_BYTE(wd, b * BB + 1u + i - VB / 2u) >> 4u);                     \
                        precise float v = lut[c] * sc;                                                                                 \
                        dot = fma(q[qb + d + b * VB + i], v, dot);                                                                     \
                    }                                                                                                                  \
                }                                                                                                                      \
            }                                                                                                                          \
        for (; d < p.dim; ++d) dot = fma(q[qb + d], VALUE(w0, d), dot);                                                                \
        return dot;                                                                                                                    \
    }
#endif
