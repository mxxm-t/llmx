// Per-value decoders over a byte buffer the including shader reads through
// QBYTE(i), a macro because GLSL functions cannot take an unsized array and
// because a shader may serve the bytes from a wider view of the same memory:
// this driver issues one load per byte and joins none of them, so the tile
// kernel reads words. Each returns value j of the block starting at byte o,
// in the
// arithmetic order of the CPU decoders in quant/, so embed matches the CPU
// exactly and the matmuls differ only in reduction order.
#ifndef LLMX_QDECODE_GLSL
#define LLMX_QDECODE_GLSL

// Q4_0: (nibble - 8) * d as nibble * d - 8 * d.
float q4_0_at(uint o, uint j, float d) {
    uint byte = QBYTE(o + 2u + (j & 15u));
    uint nib = j < 16u ? (byte & 15u) : (byte >> 4u);
    return float(nib) * d - 8.0 * d;
}

// Q4_1: d * nibble + m; d and m are the block's two halves.
float q4_1_at(uint o, uint j, float d, float m) {
    uint byte = QBYTE(o + 4u + (j & 15u));
    uint nib = j < 16u ? (byte & 15u) : (byte >> 4u);
    return d * float(nib) + m;
}

// Q6_K, 256 values in 210 bytes: 128 low nibbles, 64 bytes of high bit
// pairs, 16 signed sub-scales, a half. Two halves of 128; inside one, value
// r is one of four groups of 32 taking its low nibble and its two high
// bits from a fixed place, and its sub-scale from l / 16 plus the group's
// offset. The value is d * sc * (q - 32).
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

// Q4_K and Q5_K sub-scale and sub-min j (0..7) from the twelve packed
// bytes at s: the first four of each are whole bytes' low six bits, the
// last four are split nibbles with their high two bits in the first
// bytes' top bits.
void scale_min_k4(uint s, uint j, out uint sc, out uint mn) {
    if (j < 4u) {
        sc = QBYTE(s + j) & 63u;
        mn = QBYTE(s + j + 4u) & 63u;
    } else {
        sc = (QBYTE(s + j + 4u) & 15u) | ((QBYTE(s + j - 4u) >> 6u) << 4u);
        mn = (QBYTE(s + j + 4u) >> 4u) | ((QBYTE(s + j) >> 6u) << 4u);
    }
}

// Q4_K, 256 values in 144 bytes: d, dmin, the packed sub-scales, 128
// nibble bytes. Each 64-value chunk is 32 bytes, low nibbles first; the
// value is d * sc * q - dmin * mn with the sub-scale of group j / 32.
float q4_k_at(uint o, uint j, float d, float dmin) {
    uint c = j / 64u, r = j - c * 64u;
    uint byte = QBYTE(o + 16u + c * 32u + (r & 31u));
    uint q = r < 32u ? (byte & 15u) : (byte >> 4u);
    uint sc, mn;
    scale_min_k4(o + 4u, j / 32u, sc, mn);
    return d * float(sc) * float(q) - dmin * float(mn);
}

// Q5_K, 176 bytes: Q4_K with 32 bytes of fifth bits before the nibbles;
// for position r of chunk c the bit is 2c (low nibble) or 2c + 1 (high)
// of byte r & 31.
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
