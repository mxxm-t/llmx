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
#endif
