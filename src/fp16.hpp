#pragma once
#include <cstdint>
#include <cstring>

// IEEE 754 binary16 <-> binary32 conversion, from scratch (no libs).

inline uint16_t f32_to_f16(float f) {
    union { uint32_t u; float f; } x;
    x.f = f;
    uint32_t s = (x.u >> 16) & 0x8000u;   // sign bit (bit 15 of half)
    uint32_t e = (x.u >> 23) & 0xffu;     // f32 exponent
    uint32_t m = x.u & 0x7fffffu;         // f32 mantissa

    if (e == 0xffu) {                     // inf / nan
        return (uint16_t)(s | 0x7c00u | (m ? 0x0200u : 0u));
    }

    int es = (int)e - 127 + 15;           // rebias to f16
    if (es >= 31) {                        // overflow -> inf
        return (uint16_t)(s | 0x7c00u);
    }
    if (es <= 0) {                         // subnormal f16
        if (es < -10) {                    // underflow to zero
            return (uint16_t)s;
        }
        m |= 0x800000u;                    // restore implicit leading 1
        uint32_t shift = (uint32_t)(14 - es);
        uint32_t half = (m >> shift) + 0xfffu + ((m >> (shift + 13)) & 1u);
        return (uint16_t)(s | half);
    }
    uint32_t h = (m >> 13) + ((m & 0x1000u) ? 1u : 0u); // round-to-nearest
    return (uint16_t)(s | ((uint32_t)es << 10) | h);
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m = h & 0x3ffu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;                      // zero
        } else {                           // subnormal: normalize
            e = 1;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            m &= 0x3ffu;
            u = sign | ((e + 112) << 23) | (m << 13);
        }
    } else if (e == 0x1fu) {               // inf / nan
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e + 112) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
