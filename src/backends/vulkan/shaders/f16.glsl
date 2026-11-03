// Exact half storage conversions match core/fp16.hpp and core/bf16.hpp without depending on native half arithmetic.
#ifndef LLMX_F16_GLSL
#define LLMX_F16_GLSL
// Widen in integer bits so subnormal half weights and signed zeros do not depend on the device's half arithmetic mode.
float f16_to_f32(uint h) {
    uint sign = (h & 0x8000u) << 16u;
    uint e = (h >> 10u) & 31u, m = h & 1023u;
    uint bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;
        else {
            uint shift = uint(10 - findMSB(m));
            bits = sign | ((113u - shift) << 23u) | (((m << shift) & 1023u) << 13u);
        }
    } else if (e == 31u) bits = sign | 0x7f800000u | (m << 13u);
    else bits = sign | ((e + 112u) << 23u) | (m << 13u);
    return uintBitsToFloat(bits);
}

// Subtracting 0.5 from a float at exponent -1 gives the half subnormal exactly; select the other classes in integer bits.
vec2 f16_pair_to_f32(uint packed) {
    uvec2 h = uvec2(packed & 65535u, packed >> 16u);
    uvec2 magnitude = h & 32767u;
    precise vec2 small = uintBitsToFloat(uvec2(0x3f000000u) | (h & 1023u)) - vec2(0.5);
    uvec2 subnormal = floatBitsToUint(small);
    subnormal = mix(subnormal, uvec2(0u), equal(magnitude, uvec2(0u)));
    uvec2 normal = (magnitude << 13u) + mix(uvec2(0x38000000u), uvec2(0x70000000u), greaterThanEqual(magnitude, uvec2(0x7c00u)));
    uvec2 bits = mix(normal, subnormal, lessThan(magnitude, uvec2(0x400u)));
    return uintBitsToFloat(bits | ((h & 32768u) << 16u));
}

float bf16_to_f32(uint h) { return uintBitsToFloat(h << 16u); }

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
