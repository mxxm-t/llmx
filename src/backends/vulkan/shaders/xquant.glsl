// The 16-bit twin of a float activation row for the row kernel's integer
// dots (matmul_row.comp): values in blocks of 32 scaled so the block's
// largest magnitude is 32767, written in the order the kernels unpack
// nibble and byte words in, pairs of positions (4m, 4m + 2) and (4m + 1,
// 4m + 3), then per block the scale d and d times the block's sum, and
// after those per block d times each half's sum (Q6_K scales halves of
// 16 apart). A subgroup writes it from its lanes: every lane calls with
// its value at flat position i, positions i..i+31 aligned to 32 lying in
// 32 consecutive lanes, which the producers' thread layouts give. The
// caller declares `xq` as a writable uint buffer and enables the
// subgroup shuffle extension; subgroups are at least 32 lanes wide.
#ifndef LLMX_XQUANT_GLSL
#define LLMX_XQUANT_GLSL

// Round half away from zero, as the host reference does.
int xq_round(float v, float id) {
    float r = v * id;
    return clamp(int(sign(r) * floor(abs(r) + 0.5)), -32767, 32767);
}

// n is the row's flat length in values; the twin occupies n / 2 words,
// then 2 words per block of scale and scaled sum, then 2 words per block
// of scaled half sums.
void xquant_block(uint i, float v, uint n) {
    uint lane = gl_SubgroupInvocationID;
    uint j = i & 31u;
    float amax = abs(v);
    amax = max(amax, subgroupShuffleXor(amax, 16u));
    amax = max(amax, subgroupShuffleXor(amax, 8u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    float d = amax / 32767.0;
    float id = amax > 0.0 ? 32767.0 / amax : 0.0;
    int q = xq_round(v, id);
    // Lanes 0 and 1 of each four write the pairs (4m, 4m + 2) and
    // (4m + 1, 4m + 3) from their own value and the one two lanes on.
    int q2 = subgroupShuffle(q, min(lane + 2u, gl_SubgroupSize - 1u));
    if ((j & 3u) < 2u) xq[(i - j) / 2u + 2u * (j >> 2u) + (j & 3u)] = (uint(q) & 0xFFFFu) | (uint(q2) << 16u);
    // The sums of each half of 16, then the whole; lane 0 of the block
    // writes the table entry.
    int s = q;
    s += subgroupShuffleXor(s, 8u);
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    int other = subgroupShuffleXor(s, 16u);
    if (j == 0u) {
        uint blk = (i - j) / 32u;
        uint t = n / 2u + 2u * blk, th = n / 2u + 2u * (n / 32u) + 2u * blk;
        xq[t] = floatBitsToUint(d);
        xq[t + 1u] = floatBitsToUint(d * float(s + other));
        xq[th] = floatBitsToUint(d * float(s));
        xq[th + 1u] = floatBitsToUint(d * float(other));
    }
}

// The 8-bit twin, for devices whose integer dot is native (the row families built with LLMX_X8, and the prefill tile): values in blocks of 32 scaled so the block's largest magnitude is 127, four to a word in position order, n / 4 words, then per block the scale d and d times the block's integer sum, all from word `base`. A type's offset goes into its weight bytes instead of through half sums, so there are none. The same lane layout as above; a lane past n calls with a zero and writes nothing, so a partial subgroup still reduces correctly.
void xquant8_block(uint i, float v, uint n, uint base) {
    uint lane = gl_SubgroupInvocationID;
    uint j = i & 31u;
    float amax = abs(v);
    amax = max(amax, subgroupShuffleXor(amax, 16u));
    amax = max(amax, subgroupShuffleXor(amax, 8u));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    float d = amax / 127.0;
    float id = amax > 0.0 ? 127.0 / amax : 0.0;
    float r = v * id;
    int q = clamp(int(sign(r) * floor(abs(r) + 0.5)), -127, 127);
    // Lane 4m of each block packs the word from its own byte and the next three lanes'.
    uint b1 = uint(subgroupShuffle(q, min(lane + 1u, gl_SubgroupSize - 1u))) & 255u;
    uint b2 = uint(subgroupShuffle(q, min(lane + 2u, gl_SubgroupSize - 1u))) & 255u;
    uint b3 = uint(subgroupShuffle(q, min(lane + 3u, gl_SubgroupSize - 1u))) & 255u;
    if (i < n && (j & 3u) == 0u) xq[base + i / 4u] = (uint(q) & 255u) | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    int s = q;
    s += subgroupShuffleXor(s, 16u);
    s += subgroupShuffleXor(s, 8u);
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    if (i < n && j == 0u) {
        uint t = base + n / 4u + 2u * (i / 32u);
        xq[t] = floatBitsToUint(d);
        xq[t + 1u] = floatBitsToUint(d * float(s));
    }
}

// Where the 8-bit twin starts, in words, when a producer writes both: after the 16-bit twin's n / 2 words of pairs and n / 8 of tables, rounded up to 256 bytes so the backend can bind it at its own offset.
uint xquant8_base(uint n) { return (n / 2u + n / 8u + 63u) & ~63u; }

// What a producer writes: the 16-bit twin always, and with specialization constant 7 the 8-bit one after it. Constant 7 is a second build of each producer, which the backend dispatches once a matmul that reads the 8-bit twin has run (the Q4_K and Q5_K row families), so a model without them runs exactly the producers it ran before: with the 8-bit writer present behind a runtime branch instead, Qwen3-0.6B-Q4_0 decode lost 2.6 percent on an MI50 without ever taking it.
layout(constant_id = 7) const bool TWIN8 = false;
void xquant_twin(uint i, float v, uint n) {
    xquant_block(i, v, n);
    if (TWIN8) xquant8_block(i, v, n, xquant8_base(n));
}
#endif
