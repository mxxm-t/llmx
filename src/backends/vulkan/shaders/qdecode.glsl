// Per-value decoders over bytes the including shader reads through QBYTE(i), a macro since GLSL functions cannot take an unsized array.
// Each returns value j of the block at byte o in the CPU decoders' arithmetic order (quant/), so embed matches the CPU exactly, except q4_0_at, which only the float tile uses; embed decodes Q4_0 through q4_0_exact.
#ifndef LLMX_QDECODE_GLSL
#define LLMX_QDECODE_GLSL

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
#endif
