// The launch shapes the host and the kernels both decide by, so a grid and the kernel it launches cannot disagree (docs/src/backends-gpu.md).
#pragma once

namespace gpu_shape {

// MXFP4's quad rows (kernels/int16.gpu, mxfp4_quad_rows): kQuadLanes lanes a row, the column's twin staged in shared memory up to kQuadValues values.
constexpr unsigned kQuadLanes = 4;
constexpr unsigned long long kQuadValues = 8192;
// A one-column 8-bit MXFP4 row takes the quad rows at k_rows' 32-lane cluster where it has at least 16 blocks, a multiple of four, and at most kQuadValues values.
constexpr bool quad_rows(unsigned cluster, unsigned long long blocks) { return cluster == 32 && blocks >= 16 && blocks % 4 == 0 && blocks * 32 <= kQuadValues; }

// The passes of 64 lanes, eight values a lane, of the one-column 16-bit Q8_0 walk builds (kernels/int16.gpu, q8_0_walk): 2, 3 or 5 for a row whose width reaches the build's last pass, else 0.
// The builds hold gfx9's dot4 path, and gfx9 is where waves have 64 lanes.
constexpr unsigned walk_passes(unsigned wave, unsigned long long nin) {
    return wave != 64 ? 0 : nin > 512 && nin <= 1024 ? 2 : nin > 1024 && nin <= 1536 ? 3 : nin > 2048 && nin <= 2560 ? 5 : 0;
}
// attention_rel_sink's row kernel (kernels/f32.gpu, llmx_attention_rel_sink): a block takes up to kRelGroup query heads of one kv head, and a row of n keys runs in parts of rel_part_keys(n) keys.
// A part is kRelPartKeys keys, doubled until a row has at most kRelParts parts; the rule reads n alone, so a row's bits do not depend on its batch.
constexpr unsigned kRelGroup = 4, kRelPartKeys = 128, kRelParts = 32;
constexpr unsigned rel_part_keys(unsigned n) {
    unsigned k = kRelPartKeys;
    while ((unsigned long long)k * kRelParts < n) k *= 2;
    return k;
}
constexpr unsigned rel_parts(unsigned n) { return n ? (n + rel_part_keys(n) - 1) / rel_part_keys(n) : 1; }
// The lanes build's parts (kernels/f32.gpu, rel_lanes): kRelLaneKeys keys, doubled until a row has at most kRelLaneParts parts, a rule of n alone.
constexpr unsigned kRelLaneKeys = 64, kRelLaneParts = 16;
constexpr unsigned rel_lane_keys(unsigned n) {
    unsigned k = kRelLaneKeys;
    while ((unsigned long long)k * kRelLaneParts < n) k *= 2;
    return k;
}
constexpr unsigned rel_lane_parts(unsigned n) { return n ? (n + rel_lane_keys(n) - 1) / rel_lane_keys(n) : 1; }
// attention_rel_sink's prompt tile (kernels/f32.gpu, attention_tile): kRelTileRows query rows of one head a block, or with kRelTileGroup query heads a kv head, kRelGroupRows rows of each.
constexpr unsigned kRelTileRows = 32, kRelTileGroup = 3;
// The head-group build's threads, four waves so every SIMD holds one, and its rows: 16 lanes take two row slots, the heads' rows packed over them with at most one left idle.
constexpr unsigned kRelGroupThreads = 256, kRelGroupRows = kRelGroupThreads / 16 * 2 / kRelTileGroup;
// Attention with terms (kernels/f32.gpu, attention_terms): at most kTermsParts key parts a row, each a whole number of the block's key groups and at least two of them.
constexpr unsigned kTermsParts = 32;
constexpr unsigned terms_keys(unsigned n, unsigned parts, unsigned groups) {
    const unsigned k = ((n + parts - 1) / parts + groups - 1) / groups * groups;
    return k < 2 * groups ? 2 * groups : k;
}
constexpr unsigned terms_parts(unsigned n, unsigned parts, unsigned groups) { return n ? (n + terms_keys(n, parts, groups) - 1) / terms_keys(n, parts, groups) : 1; }

}  // namespace gpu_shape
