// Range repair of the float tiles' TILE_FOLD_K-input block fold (docs/VULKAN.md, Kernel notes); the includer defines TILE_REPLAY_WEIGHT(row, k) and TILE_REPLAY_INPUT(col, k).
// The bit test rather than isinf/isnan, since modules without SignedZeroInfNanPreserve may fold those away.
bool tile_nonfinite(float v) { return (floatBitsToUint(v) & 0x7f800000u) == 0x7f800000u; }

// The float tiles' fold block in inputs, a multiple of both tiles' stages.
#define TILE_FOLD_K 256u

#define TILE_REPLAY_UP uintBitsToFloat(0x4f800000u)

// One output's fold again over [kbeg, kend), the larger operand of each product scaled by 2^-32, the total left scaled for tile_replay or a split call's reduce.
// A power of two commutes with rounding, so the total times 2^32 is the fold's own value where no partial left f32, and finite where the sum is; an infinite or NaN operand still gives a nonfinite result.
float tile_replay_scaled(uint row, uint col, uint kbeg, uint kend) {
    const float down = uintBitsToFloat(0x2f800000u);
#ifdef LLMX_PRESERVE_F32
    precise float acc = -0.0, block = 0.0;
#else
    float acc = -0.0, block = 0.0;
#endif
    for (uint k = kbeg; k < kend; ++k) {
        float a = TILE_REPLAY_WEIGHT(row, k), b = TILE_REPLAY_INPUT(col, k);
        if (abs(a) >= abs(b)) a *= down;
        else b *= down;
#ifdef LLMX_PRESERVE_F32
        block = fma(a, b, block);
#else
        block += a * b;
#endif
        if ((k + 1u) % TILE_FOLD_K == 0u || k + 1u == kend) {
            acc += block;
            block = 0.0;
        }
    }
    return acc;
}

float tile_replay(uint row, uint col, uint kbeg, uint kend) {
    precise float s = tile_replay_scaled(row, col, kbeg, kend) * TILE_REPLAY_UP;
    return s;
}
