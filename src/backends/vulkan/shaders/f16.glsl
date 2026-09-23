// f32 to f16 with round-to-nearest-even in the bits, so the device stores what the CPU does (core/fp16.hpp); packHalf2x16 leaves the rounding to the driver.
#ifndef LLMX_F16_GLSL
#define LLMX_F16_GLSL
uint f16_bits(float f) {
    uint x = floatBitsToUint(f);
    uint sign = (x >> 16u) & 0x8000u;
    uint exp = (x >> 23u) & 0xFFu;
    uint mant = x & 0x7FFFFFu;
    if (exp == 0xFFu) return sign | 0x7C00u | (mant != 0u ? 0x200u : 0u);
    int e = int(exp) - 127 + 15;
    if (e >= 31) return sign | 0x7C00u;
    if (e <= 0) {
        if (e < -10) return sign;
        mant |= 0x800000u;
        uint shift = uint(14 - e);
        uint h = mant >> shift;
        uint rem = mant & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (h & 1u) != 0u)) h += 1u;
        return sign | h;
    }
    uint h = sign | (uint(e) << 10u) | (mant >> 13u);
    uint rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h += 1u;
    return h;
}
#endif
