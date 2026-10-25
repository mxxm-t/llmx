// The Q8_0 decode kernel's helpers (matmul_vec_q8.comp), which a kernel fusing a Q8_0 row product with other work shares beside matmul_row.glsl's declarations.
// A weight word as the dot takes it and the dot itself: four signed bytes in the 8-bit builds, two pairs of 16-bit values otherwise (dot16.glsl).
#ifdef LLMX_I8
#define WQT uint
#define WIDEN(w) (w)
int dot_word(uint w, uint x, int acc) { return acc + dotPacked4x8EXT(int(w), int(x)); }
#else
#define WQT uvec2
#define WIDEN(w) weight16(w)
#define dot_word dot_pairs
#endif

// Columns whose activations a lane loads together: their loads issue back to back, and a group past the pushed count is skipped whole.
// A plain pass gives a build more than half the columns it holds (for_each_column_chunk), so only a group past the first half of a one-group build can be past the count, and the other builds check before every group.
const uint GROUP = MAX_COLS < 4u ? MAX_COLS : 4u;
const bool CHECK_EVERY_GROUP = GROUPED || SPAN > 1u;
// Whether a group past the first half that the count leaves part full skips its surplus columns' products, each behind a uniform check; the 16-column build keeps one copy of its products, which a second copy slowed (docs/VULKAN.md).
const bool PARTIAL = MAX_COLS <= 8u;
// The values TREE reduces, a power of two in every build, and how many a lane holds at the end.
const uint NV = ROWS * MAX_COLS;
const uint LOGNV = NV >= 128u ? 7u : NV >= 64u ? 6u : NV >= 32u ? 5u : NV >= 16u ? 4u : NV >= 8u ? 3u : NV >= 4u ? 2u : NV >= 2u ? 1u : 0u;
const uint NLEFT = NV > 64u ? NV / 64u : 1u;

// Whether a lane keeps the upper half of its values at TREE level l; partners at every level then hold the same values, since each level's choice is unchanged by the later levels' lane masks.
bool keeps_upper(uint l, uint lane) {
    uint v = l == 0u ? (lane ^ (lane >> 2u)) : l == 1u ? ((lane >> 1u) ^ (lane >> 2u)) : l == 2u ? ((lane >> 2u) ^ (lane >> 3u)) : (lane >> l);
    return (v & 1u) != 0u;
}
// The value of lane l ^ 1, 2, 7, 15, 16 or 32 at level 0 to 5: the subgroup reduction's partners, a quad's pairs, its halves, an eight's mirror, a sixteen's mirror, then the sixteens and the halves of the subgroup.
float partner(float v, uint l) {
    if (l == 0u) return subgroupShuffleXor(v, 1u);
    if (l == 1u) return subgroupShuffleXor(v, 2u);
    if (l == 2u) return subgroupShuffleXor(v, 7u);
    if (l == 3u) return subgroupShuffleXor(v, 15u);
    if (l == 4u) return subgroupShuffleXor(v, 16u);
    return subgroupShuffleXor(v, 32u);
}
