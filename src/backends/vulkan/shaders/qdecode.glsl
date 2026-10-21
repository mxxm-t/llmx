// Per-value decoders over bytes the including shader reads through QBYTE(i), a macro since GLSL functions cannot take an unsized array.
// Each returns value j of the block at byte o in the CPU decoders' arithmetic order (quant/), so embed matches the CPU exactly, except q4_0_at, which only the float tile uses; embed decodes Q4_0 through q4_0_exact.
#ifndef LLMX_QDECODE_GLSL
#define LLMX_QDECODE_GLSL

int mxfp4_value(uint code) {
    int magnitude = int(code & 7u);
    int value = (2 | (magnitude & 1)) << max((magnitude >> 1) - 1, 0);
    value = magnitude < 2 ? magnitude : value;
    return (code & 8u) != 0u ? -value : value;
}
float mxfp4_scale(uint e) {
    return uintBitsToFloat(e <= 1u ? 0x00200000u << e : (e - 1u) << 23u);
}
// MXFP4's doubled E2M1 magnitudes scaled by 2^(e - 128), assembled as bits.
// This keeps subnormals without float arithmetic; both zero codes give +0 and overflow gives infinity.
float mxfp4_at(uint o, uint j) {
    uint packed = QBYTE(o + 1u + (j & 15u));
    uint code = j < 16u ? packed & 15u : packed >> 4u;
    uint magnitude = code & 7u;
    int exponent = int(QBYTE(o)) + int(magnitude >> 1u) - 1;
    uint fraction = magnitude > 1u ? (magnitude & 1u) << 22u : 0u;
    uint bits = (uint(exponent) << 23u) | fraction;
    uint subnormal = (fraction | 0x00800000u) >> uint(max(1 - exponent, 0));
    bits = exponent <= 0 ? subnormal : bits;
    bits = exponent >= 255 ? 0x7f800000u : bits;
    return uintBitsToFloat(magnitude == 0u ? 0u : bits | ((code & 8u) << 28u));
}

// IQ4_NL and IQ4_XS share signed lookup values and 32-value nibble groups.
int iq4_nl_value(uint code) {
    // Four signed bytes per word keep the lane-local lookup in integer registers.
    uint lo = code < 4u ? 0xbfad9881u : 0xf6eaddcfu;
    uint hi = code < 12u ? 0x26190d01u : 0x71594535u;
    return bitfieldExtract(int(code < 8u ? lo : hi), int((code & 3u) * 8u), 8);
}
float iq4_at(uint codes, uint j, float d) {
    uint packed = QBYTE(codes + (j & 15u));
    int value = iq4_nl_value(j < 16u ? packed & 15u : packed >> 4u);
    // Embedding builds lack float-control modes, so construct the format's zero sign as bits.
    uint sign = (floatBitsToUint(d) ^ (value < 0 ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(d * float(value)) & 0x7fffffffu) | sign);
}
float iq4_nl_at(uint o, uint j, float d) {
    return iq4_at(o + 2u, j, d);
}
float iq4_xs_scale(uint o, uint group, float d) {
    uint low = (QBYTE(o + 4u + group / 2u) >> (4u * (group & 1u))) & 15u;
    uint high = (QBYTE(o + 2u + group / 4u) >> (2u * (group & 3u))) & 3u;
    int scale = int(low | (high << 4u)) - 32;
    uint sign = (floatBitsToUint(d) ^ (scale < 0 ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(d * float(scale)) & 0x7fffffffu) | sign);
}
float iq4_xs_at(uint o, uint j, float d) {
    uint group = j / 32u;
    return iq4_at(o + 8u + group * 16u, j % 32u, iq4_xs_scale(o, group, d));
}
// Resolve one 32-value group; row and tile variants make the format a compile-time constant.
uint iq4_group(uint type, uint row, uint width, uint group, out float scale) {
    if (type == TYPE_IQ4_XS) {
        uint o = (row * (width / IQ4_XS_BLOCK) + group / 8u) * IQ4_XS_BYTES;
        scale = iq4_xs_scale(o, group % 8u, half_at(QBYTE(o), QBYTE(o + 1u)));
        return o + 8u + (group % 8u) * 16u;
    }
    uint o = (row * (width / IQ4_NL_BLOCK) + group) * IQ4_NL_BYTES;
    scale = half_at(QBYTE(o), QBYTE(o + 1u));
    return o + 2u;
}

// Q4_0: (nibble - 8) * d as nibble * d - 8 * d, the same value under a finite scale except +0 for the -0 that nibble 8 gives under a negative scale, which the tile's sums never see.
float q4_0_at(uint o, uint j, float d) {
    uint byte = QBYTE(o + 2u + (j & 15u));
    uint nib = j < 16u ? (byte & 15u) : (byte >> 4u);
    return float(nib) * d - 8.0 * d;
}

// Q4_0 as the CPU decodes it, (nibble - 8) * d, exact under a finite scale, with the sign set as bits, since a driver need not keep a zero's sign without SignedZeroInfNanPreserve.
float q4_0_exact(uint o, uint j, float d) {
    uint byte = QBYTE(o + 2u + (j & 15u));
    uint nib = j < 16u ? (byte & 15u) : (byte >> 4u);
    uint sign = (floatBitsToUint(d) ^ (nib < 8u ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(float(int(nib) - 8) * d) & 0x7FFFFFFFu) | sign);
}

// Q4_1: d * nibble + m; d and m are the block's two halves.
float q4_1_at(uint o, uint j, float d, float m) {
    uint byte = QBYTE(o + 4u + (j & 15u));
    uint nib = j < 16u ? (byte & 15u) : (byte >> 4u);
    return d * float(nib) + m;
}

// The common five-bit code, with high bits four bytes before the nibble plane.
uint q5_code(uint o, uint j, uint qoff) {
    uint byte = QBYTE(o + qoff + (j & 15u));
    uint low = j < 16u ? (byte & 15u) : (byte >> 4u);
    uint high = (QBYTE(o + qoff - 4u + j / 8u) >> (j & 7u)) & 1u;
    return low | (high << 4u);
}
// Four five-bit codes as signed bytes centered at16, shared by rows and tiles.
uint q5_signed_bytes(uint codes) {
    uint t = codes ^ 0x10101010u, neg = t & 0x10101010u;
    return t | (neg << 1u) | (neg << 2u) | (neg << 3u);
}
float q5_0_at(uint o, uint j, float d) {
    uint code = q5_code(o, j, 6u);
    uint sign = (floatBitsToUint(d) ^ (code < 16u ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(float(int(code) - 16) * d) & 0x7fffffffu) | sign);
}
float q5_1_at(uint o, uint j, float d, float m) {
    return d * float(q5_code(o, j, 8u)) + m;
}

// Q2_K has a scale/min nibble pair per 16 values and interleaves four code planes within each 128-value half.
struct Q2KScale { float scale; float min; uint zero_sign; };
Q2KScale q2_k_scale(uint o, uint j) {
    float d = half_at(QBYTE(o + 80u), QBYTE(o + 81u));
    float m = half_at(QBYTE(o + 82u), QBYTE(o + 83u));
    uint sc = QBYTE(o + j / 16u);
    precise float scale = d * float(sc & 15u), min = m * float(sc >> 4u);
    return Q2KScale(scale, min, floatBitsToUint(d) & ~floatBitsToUint(m) & 0x80000000u);
}
uint q2_k_code(uint o, uint j) {
    return (QBYTE(o + 16u + 32u * (j / 128u) + j % 32u) >> (2u * ((j % 128u) / 32u))) & 3u;
}
float q2_k_at(uint o, uint j, Q2KScale s) {
    uint code = q2_k_code(o, j);
    precise float value = s.scale * float(code) - s.min;
    // Preserve subtraction's zero sign even when the embedding shader lacks signed-zero float controls.
    return value == 0.0 ? uintBitsToFloat(s.zero_sign) : value;
}
float q2_k_at(uint o, uint j) {
    return q2_k_at(o, j, q2_k_scale(o, j));
}

struct Q3KScale { float scale; uint sign; };
Q3KScale q3_k_scale(uint o, uint j) {
    float d = half_at(QBYTE(o + 108u), QBYTE(o + 109u));
    uint group = j / 16u;
    uint low_scale = (QBYTE(o + 96u + group % 8u) >> (4u * (group / 8u))) & 15u;
    uint high_scale = (QBYTE(o + 104u + group % 4u) >> (2u * (group / 4u))) & 3u;
    int scale = int(low_scale | (high_scale << 4u)) - 32;
    precise float ds = d * float(scale);
    return Q3KScale(ds, (floatBitsToUint(d) ^ (scale < 0 ? 0x80000000u : 0u)) & 0x80000000u);
}
int q3_k_code(uint o, uint j) {
    uint plane = j / 32u;
    return int((QBYTE(o + 32u + 32u * (j / 128u) + j % 32u) >> (2u * (plane % 4u))) & 3u)
             - (((QBYTE(o + j % 32u) >> plane) & 1u) == 0u ? 4 : 0);
}
float q3_k_at(uint o, uint j, Q3KScale s) {
    int code = q3_k_code(o, j);
    precise float value = s.scale * float(code);
    uint sign = s.sign ^ (code < 0 ? 0x80000000u : 0u);
    return uintBitsToFloat((floatBitsToUint(value) & 0x7fffffffu) | sign);
}
float q3_k_at(uint o, uint j) {
    return q3_k_at(o, j, q3_k_scale(o, j));
}

// Q6_K, 256 values in 210 bytes: 128 low nibbles, 64 bytes of high bit pairs, 16 signed sub-scales, a half; value = d * sc * (q - 32).
float q6_k_at(uint o, uint j, float d) {
    uint n = j / 128u, r = j - n * 128u;
    uint which = r / 32u, l = r - which * 32u;
    uint ql = o + n * 64u, qh = o + 128u + n * 32u, sc = o + 192u + n * 8u;
    uint lowb = QBYTE(ql + l + (which == 1u || which == 3u ? 32u : 0u));
    uint low = which < 2u ? (lowb & 15u) : (lowb >> 4u);
    uint high = (QBYTE(qh + l) >> (2u * which)) & 3u;
    int q = int(low | (high << 4u)) - 32;
    int s = (int(QBYTE(sc + l / 16u + 2u * which)) << 24) >> 24;
    return d * float(s) * float(q);
}

// Q4_K and Q5_K sub-scale and sub-min j (0..7) from the twelve packed bytes at s: groups 0..3 are the low six bits of whole bytes, groups 4..7 split nibbles with their top two bits in the first bytes.
void scale_min_k4(uint s, uint j, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = QBYTE(s + j) & 63u;
        mn = QBYTE(s + j + 4u) & 63u;
    } else {
        sc = (QBYTE(s + j + 4u) & 15u) | ((QBYTE(s + j - 4u) >> 6u) << 4u);
        mn = (QBYTE(s + j + 4u) >> 4u) | ((QBYTE(s + j) >> 6u) << 4u);
    }
}

// Q4_K, 256 values in 144 bytes: d, dmin, packed sub-scales, 128 nibble bytes, each 64-value chunk 32 bytes low nibbles first; value = d * sc * q - dmin * mn.
float q4_k_at(uint o, uint j, float d, float dmin) {
    uint c = j / 64u, r = j - c * 64u;
    uint byte = QBYTE(o + 16u + c * 32u + (r & 31u));
    uint q = r < 32u ? (byte & 15u) : (byte >> 4u);
    uint sc, mn;
    scale_min_k4(o + 4u, j / 32u, sc, mn);
    return d * float(sc) * float(q) - dmin * float(mn);
}

// Q5_K, 176 bytes: Q4_K with 32 bytes of fifth bits before the nibbles; for position r of chunk c the bit is 2c (low nibble) or 2c + 1 (high) of byte r & 31.
float q5_k_at(uint o, uint j, float d, float dmin) {
    uint c = j / 64u, r = j - c * 64u;
    uint byte = QBYTE(o + 48u + c * 32u + (r & 31u));
    uint hb = QBYTE(o + 16u + (r & 31u));
    uint q = r < 32u ? ((byte & 15u) + (((hb >> (2u * c)) & 1u) << 4u))
                     : ((byte >> 4u) + (((hb >> (2u * c + 1u)) & 1u) << 4u));
    uint sc, mn;
    scale_min_k4(o + 4u, j / 32u, sc, mn);
    return d * float(sc) * float(q) - dmin * float(mn);
}
// One decoded value of a quantized matrix row, for F32-activation dots.
float quantized_at(uint type, uint row, uint width, uint column) {
    if (type == TYPE_MXFP4) {
        uint o = (row * (width / MXFP4_BLOCK) + column / MXFP4_BLOCK) * MXFP4_BYTES;
        return mxfp4_at(o, column % MXFP4_BLOCK);
    }
    uint block = block_values(type), j = column % block;
    uint o = (row * (width / block) + column / block) * block_bytes(type);
    if (type == TYPE_Q2_K) return q2_k_at(o, j);
    if (type == TYPE_Q3_K) return q3_k_at(o, j);
    uint scale = o + (type == TYPE_Q6_K ? 208u : 0u);
    float d = half_at(QBYTE(scale), QBYTE(scale + 1u));
    if (type == TYPE_Q8_0) return d * float(int(QBYTE(o + 2u + j) << 24u) >> 24);
    if (type == TYPE_Q4_0) return q4_0_exact(o, j, d);
    if (type == TYPE_Q5_0) return q5_0_at(o, j, d);
    if (type == TYPE_IQ4_NL) return iq4_nl_at(o, j, d);
    if (type == TYPE_IQ4_XS) return iq4_xs_at(o, j, d);
    if (type == TYPE_Q6_K) return q6_k_at(o, j, d);
    float m = half_at(QBYTE(o + 2u), QBYTE(o + 3u));
    if (type == TYPE_Q4_1) return q4_1_at(o, j, d, m);
    if (type == TYPE_Q5_1) return q5_1_at(o, j, d, m);
    if (type == TYPE_Q4_K) return q4_k_at(o, j, d, m);
    return q5_k_at(o, j, d, m);
}
#endif
