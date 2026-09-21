// Per-value decoders over a byte buffer the including shader names through
// QBYTES before this file, since GLSL functions cannot take an unsized
// array. Each returns value j of the block starting at byte o, in the
// arithmetic order of the CPU decoders in quant/, so embed matches the CPU
// exactly and the matmuls differ only in reduction order.
#ifndef LLMX_QDECODE_GLSL
#define LLMX_QDECODE_GLSL

// Q4_0: (nibble - 8) * d as nibble * d - 8 * d.
float q4_0_at(uint o, uint j, float d) {
    uint byte = uint(QBYTES[o + 2u + (j & 15u)]);
    uint nib = j < 16u ? (byte & 15u) : (byte >> 4u);
    return float(nib) * d - 8.0 * d;
}

// Q4_1: d * nibble + m; d and m are the block's two halves.
float q4_1_at(uint o, uint j, float d, float m) {
    uint byte = uint(QBYTES[o + 4u + (j & 15u)]);
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
    uint lowb = uint(QBYTES[ql + l + (which == 1u || which == 3u ? 32u : 0u)]);
    uint low = which < 2u ? (lowb & 15u) : (lowb >> 4u);
    uint high = (uint(QBYTES[qh + l]) >> (2u * which)) & 3u;
    int q = int(low | (high << 4u)) - 32;
    int s = (int(uint(QBYTES[sc + l / 16u + 2u * which])) << 24) >> 24;
    return d * float(s) * float(q);
}
#endif
