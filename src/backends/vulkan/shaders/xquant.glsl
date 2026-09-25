// The 16-bit twin of a float activation row for the row kernels (matmul_row.comp): blocks of 32 scaled so the largest magnitude is 32767, as position pairs (4m, 4m + 2) and (4m + 1, 4m + 3), then per block the scale and scaled sum, then per block the scaled half sums (Q6_K scales halves apart).
// A subgroup writes it: every lane calls with its value at flat position i, and positions i..i+31 aligned to 32 must lie in 32 consecutive lanes.
// The caller declares `xq` as a writable uint buffer.
#ifndef LLMX_XQUANT_GLSL
#define LLMX_XQUANT_GLSL

// Round half away from zero, as the host reference does.
int xq_round(float v, float id) {
    float r = v * id;
    return clamp(int(sign(r) * floor(abs(r) + 0.5)), -32767, 32767);
}

// n is the row's flat length: n / 2 words of pairs, then 2 words per block of scale and scaled sum, then 2 per block of scaled half sums.
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
        xq[t] = floatBitsToUint(d);
        xq[t + 1u] = floatBitsToUint(d * float(s + other));
        xq[th] = floatBitsToUint(d * float(s));
        xq[th + 1u] = floatBitsToUint(d * float(other));
    }
}

// The 8-bit twin, for integer-dot devices: blocks of 32 scaled so the largest magnitude is 127, four to a word, n / 4 words, then per block the scale and scaled sum, from word `base`.
// The same lane layout as above; a lane past n calls with a zero.
// Block blk of the destination takes value i's block: producers keep position order (blk = i / 32), the prefill tile's copy is block-major.
void xquant8_block_at(uint i, float v, uint n, uint base, uint blk) {
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
    if (i < n && (j & 3u) == 0u) xq[base + blk * 8u + j / 4u] = (uint(q) & 255u) | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    int s = q;
    s += subgroupShuffleXor(s, 16u);
    s += subgroupShuffleXor(s, 8u);
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    if (i < n && j == 0u) {
        uint t = base + n / 4u + 2u * blk;
        xq[t] = floatBitsToUint(d);
        xq[t + 1u] = floatBitsToUint(d * float(s));
    }
}
void xquant8_block(uint i, float v, uint n, uint base) { xquant8_block_at(i, v, n, base, i / 32u); }

// The same 8-bit block, a lane per four consecutive values: a block is eight consecutive lanes aligned to eight, lane w of them holding its word w.
// Every lane calls, those of a block with nothing to write with `live` false; a maximum and an integer sum do not depend on their order, so the block is the one the lanes above write, bit for bit.
// Word w of block blk goes to word blk * 8 + w, and the block's scale and scaled sum to words tab + 2 * blk and the next.
void xquant8_word(vec4 v, bool live, uint w, uint blk, uint tab) {
    vec4 a = abs(v);
    float amax = max(max(a.x, a.y), max(a.z, a.w));
    amax = max(amax, subgroupShuffleXor(amax, 4u));
    amax = max(amax, subgroupShuffleXor(amax, 2u));
    amax = max(amax, subgroupShuffleXor(amax, 1u));
    float d = amax / 127.0;
    float id = amax > 0.0 ? 127.0 / amax : 0.0;
    vec4 r = v * id;
    ivec4 q = clamp(ivec4(sign(r) * floor(abs(r) + 0.5)), -127, 127);
    int s = q.x + q.y + q.z + q.w;
    s += subgroupShuffleXor(s, 4u);
    s += subgroupShuffleXor(s, 2u);
    s += subgroupShuffleXor(s, 1u);
    if (!live) return;
    uvec4 b = uvec4(q) & 255u;
    xq[blk * 8u + w] = b.x | (b.y << 8u) | (b.z << 16u) | (b.w << 24u);
    if (w == 0u) {
        xq[tab + 2u * blk] = floatBitsToUint(d);
        xq[tab + 2u * blk + 1u] = floatBitsToUint(d * float(s));
    }
}

// The prefill tile's copy of a batch nin wide (quantize_x8.comp) orders blocks by block of the inner dimension, then column, a block of the inner dimension being a row of xrow columns: the block of values 4t .. 4t + 3.
// Its scale table starts after all rows' quants, at word xquant8_tile_table.
uint xquant8_tile_block(uint t, uint nin, uint xrow) {
    uint nblk = nin / 32u, run = t / 8u, col = run / nblk;
    return (run - col * nblk) * xrow + col;
}
uint xquant8_tile_table(uint nin, uint xrow) { return nin / 32u * xrow * 8u; }

// Where the 8-bit twin starts, in words, after the 16-bit twin, rounded up to 256 bytes so it can be bound at its own offset.
uint xquant8_base(uint n) { return (n / 2u + n / 8u + 63u) & ~63u; }

// What a producer writes: the 16-bit twin, and with specialization constant 7 the 8-bit one after it. Constant 7 is a second build, dispatched once a matmul reading the 8-bit twin has run, so other models keep the cheaper producers.
layout(constant_id = 7) const bool TWIN8 = false;
void xquant_twin(uint i, float v, uint n) {
    xquant_block(i, v, n);
    if (TWIN8) xquant8_block(i, v, n, xquant8_base(n));
}
#endif
