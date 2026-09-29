// The 16-bit twin of a float activation row for the row kernels (matmul_row.comp) and the integer-dot tile (matmul_tile_q.comp): blocks of 32 scaled so the largest magnitude is 32767, as position pairs (4m, 4m + 2) and (4m + 1, 4m + 3), then per block the scale and scaled sum, then per block the scaled half sums (Q6_K scales halves apart).
// A subgroup writes it: every lane calls with its value at flat position i, and positions i..i+31 aligned to 32 must lie in 32 consecutive lanes.
// The caller declares `xq` as a writable uint buffer.
#ifndef LLMX_XQUANT_GLSL
#define LLMX_XQUANT_GLSL

// Scale an f32 by a power of two as bits, retaining subnormals on devices that flush float arithmetic.
float xq_shift(float v, int shift) {
    uint bits = floatBitsToUint(v), signbit = bits & 0x80000000u;
    uint mantissa = bits & 0x7FFFFFu;
    int exponent = int((bits >> 23u) & 255u);
    if (exponent == 255 || (exponent == 0 && mantissa == 0u)) return v;
    if (exponent == 0) {
        int left = 23 - findMSB(mantissa);
        mantissa <<= uint(left);
        exponent = 1 - left;
    } else mantissa |= 0x800000u;
    exponent += shift;
    if (exponent >= 255) return uintBitsToFloat(signbit | 0x7F800000u);
    if (exponent > 0) return uintBitsToFloat(signbit | (uint(exponent) << 23u) | (mantissa & 0x7FFFFFu));
    int right = 1 - exponent;
    if (right > 24) return uintBitsToFloat(signbit);
    uint low = mantissa & ((1u << uint(right)) - 1u), halfway = 1u << uint(right - 1);
    mantissa >>= uint(right);
    if (low > halfway || (low == halfway && (mantissa & 1u) != 0u)) ++mantissa;
    return uintBitsToFloat(signbit | mantissa);
}

// Extreme blocks are normalized before division; ordinary blocks keep their original operations.
void xq_scale(uint top, float levels, out float d, out float id, out int shift) {
    float amax = uintBitsToFloat(top);
    shift = 0;
    if (top != 0u && (top < 0x08800000u || top >= 0x7E800000u) && top < 0x7F800000u) {
        int exponent = int(top >> 23u);
        if (exponent == 0) exponent = findMSB(top) - 22;
        shift = 127 - exponent;
        float normalized = xq_shift(amax, shift);
        uint scale = max(1u, floatBitsToUint(xq_shift(normalized / levels, -shift)));
        d = xq_shift(uintBitsToFloat(scale), shift);
        if (scale < 0x00800000u && d * levels < normalized) d = xq_shift(uintBitsToFloat(++scale), shift);
        // The largest finite input must not round its reconstruction to infinity.
        if (floatBitsToUint(xq_shift(d * levels, -shift)) == 0x7F800000u) d = xq_shift(uintBitsToFloat(scale - 1u), shift);
        id = 1.0 / d;
    } else {
        d = amax / levels;
        id = amax > 0.0 ? levels / amax : 0.0;
    }
}

// Restore a block's table values under one shared check for the ordinary path.
vec2 xq_shift(vec2 v, int shift) {
    if (shift == 0) return v;
    return vec2(xq_shift(v.x, shift), xq_shift(v.y, shift));
}
vec4 xq_shift(vec4 v, int shift) {
    if (shift == 0) return v;
    return vec4(xq_shift(v.x, shift), xq_shift(v.y, shift), xq_shift(v.z, shift), xq_shift(v.w, shift));
}
float xq_input(float v, int shift) { return shift == 0 ? v : xq_shift(v, shift); }

// Round half away from zero, as the host reference does.
int xq_round(float v, float id) {
    float r = v * id;
    return clamp(int(sign(r) * floor(abs(r) + 0.5)), -32767, 32767);
}

// n is the row's flat length: n / 2 words of pairs, then 2 words per block of scale and scaled sum, then 2 per block of scaled half sums.
void xquant_block(uint i, float v, uint n) {
    uint lane = gl_SubgroupInvocationID;
    uint j = i & 31u;
    uint amax = floatBitsToUint(v) & 0x7FFFFFFFu;
    amax = max(amax, subgroupShuffleXor(amax, 16u));
    amax = max(amax, subgroupShuffleXor(amax, 8u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    float d, id; int shift;
    xq_scale(amax, 32767.0, d, id, shift);
    int q = xq_round(xq_input(v, shift), id);
    // Lanes 0 and 1 of each four write the pairs (4m, 4m + 2) and (4m + 1, 4m + 3).
    int q2 = subgroupShuffle(q, min(lane + 2u, gl_SubgroupSize - 1u));
    if ((j & 3u) < 2u) xq[(i - j) / 2u + 2u * (j >> 2u) + (j & 3u)] = (uint(q) & 0xFFFFu) | (uint(q2) << 16u);
    // The half sums, then the whole; lane 0 of the block writes the table entry.
    int s = q;
    s += subgroupShuffleXor(s, 8u);
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    int other = subgroupShuffleXor(s, 16u);
    if (j == 0u) {
        uint blk = (i - j) / 32u;
        uint t = n / 2u + 2u * blk, th = n / 2u + 2u * (n / 32u) + 2u * blk;
        vec4 values = xq_shift(vec4(d, d * float(s + other), d * float(s), d * float(other)), -shift);
        xq[t] = floatBitsToUint(values.x);
        xq[t + 1u] = floatBitsToUint(values.y);
        xq[th] = floatBitsToUint(values.z);
        xq[th + 1u] = floatBitsToUint(values.w);
    }
}

// The 16-bit twin as above, a lane per four consecutive values: a block is eight consecutive lanes aligned to eight, lane w of them holding values 4w .. 4w + 3 of block blk.
// Every lane calls, those of a block with nothing to write with `live` false; a maximum and an integer sum do not depend on their order, so the twin is the one the lanes above write, bit for bit.
void xquant_word(vec4 v, bool live, uint w, uint blk, uint n) {
    uvec4 a = floatBitsToUint(v) & 0x7FFFFFFFu;
    uint amax = max(max(a.x, a.y), max(a.z, a.w));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    float d, id; int shift;
    xq_scale(amax, 32767.0, d, id, shift);
    ivec4 q = ivec4(xq_round(xq_input(v.x, shift), id), xq_round(xq_input(v.y, shift), id),
                    xq_round(xq_input(v.z, shift), id), xq_round(xq_input(v.w, shift), id));
    // The half sums: lanes 0 to 3 hold values 0 to 15, lanes 4 to 7 values 16 to 31.
    int s = q.x + q.y + q.z + q.w;
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    int other = subgroupShuffleXor(s, 4u);
    if (!live) return;
    xq[blk * 16u + 2u * w] = (uint(q.x) & 0xFFFFu) | (uint(q.z) << 16u);
    xq[blk * 16u + 2u * w + 1u] = (uint(q.y) & 0xFFFFu) | (uint(q.w) << 16u);
    if (w == 0u) {
        uint t = n / 2u + 2u * blk, th = n / 2u + 2u * (n / 32u) + 2u * blk;
        vec4 values = xq_shift(vec4(d, d * float(s + other), d * float(s), d * float(other)), -shift);
        xq[t] = floatBitsToUint(values.x);
        xq[t + 1u] = floatBitsToUint(values.y);
        xq[th] = floatBitsToUint(values.z);
        xq[th + 1u] = floatBitsToUint(values.w);
    }
}

// The 8-bit twin, for the integer-dot row kernels: blocks of 32 scaled so the largest magnitude is 127, four to a word, n / 4 words, then per block the scale and scaled sum, from word `base`.
// The same lane layout as xquant_block; a lane past n calls with a zero.
void xquant8_block(uint i, float v, uint n, uint base) {
    uint lane = gl_SubgroupInvocationID;
    uint j = i & 31u, blk = i / 32u;
    uint amax = floatBitsToUint(v) & 0x7FFFFFFFu;
    amax = max(amax, subgroupShuffleXor(amax, 16u));
    amax = max(amax, subgroupShuffleXor(amax, 8u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    float d, id; int shift;
    xq_scale(amax, 127.0, d, id, shift);
    float r = xq_input(v, shift) * id;
    int q = clamp(int(sign(r) * floor(abs(r) + 0.5)), -127, 127);
    // Lane 4m of each block packs the word from its own byte and the next three lanes'.
    uint b1 = uint(subgroupShuffle(q, min(lane + 1u, gl_SubgroupSize - 1u))) & 255u;
    uint b2 = uint(subgroupShuffle(q, min(lane + 2u, gl_SubgroupSize - 1u))) & 255u;
    uint b3 = uint(subgroupShuffle(q, min(lane + 3u, gl_SubgroupSize - 1u))) & 255u;
    if (i < n && (j & 3u) == 0u) xq[base + blk * 8u + j / 4u] = (uint(q) & 255u) | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    int s = q;
    s += subgroupShuffleXor(s, 16u);
    s += subgroupShuffleXor(s, 8u);
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    if (i < n && j == 0u) {
        uint t = base + n / 4u + 2u * blk;
        vec2 values = xq_shift(vec2(d, d * float(s)), -shift);
        xq[t] = floatBitsToUint(values.x);
        xq[t + 1u] = floatBitsToUint(values.y);
    }
}
// Where the 8-bit twin starts, in words, after the 16-bit twin, rounded up to 256 bytes so it can be bound at its own offset.
uint xquant8_base(uint n) { return (n / 2u + n / 8u + 63u) & ~63u; }

// What a producer writes: the 16-bit twin, and with specialization constant 7 the 8-bit one after it. Constant 7 is a second build, dispatched once a matmul reading the 8-bit twin has run, so other models keep the cheaper producers.
layout(constant_id = 7) const bool TWIN8 = false;
void xquant_twin(uint i, float v, uint n) {
    xquant_block(i, v, n);
    if (TWIN8) xquant8_block(i, v, n, xquant8_base(n));
}
#endif
