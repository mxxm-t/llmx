// Range repair of a float-weight split call's parts (docs/VULKAN.md, Kernel notes); the includer defines TILE_REPLAY_WEIGHT(row, k) and TILE_REPLAY_INPUT(col, k).
// The bit test rather than isinf/isnan, since modules without SignedZeroInfNanPreserve may fold those away.
bool tile_nonfinite(float v) { return (floatBitsToUint(v) & 0x7f800000u) == 0x7f800000u; }

#define TILE_REPLAY_UP uintBitsToFloat(0x4f800000u)

// One output's sum again over [kbeg, kend), the larger operand of each product scaled by 2^-32, the total left scaled.
// A power of two commutes with rounding, so the total times 2^32 is finite where the sum is, and an infinite or NaN operand still gives a nonfinite result.
float tile_replay_scaled(uint row, uint col, uint kbeg, uint kend) {
    const float down = uintBitsToFloat(0x2f800000u);
#ifdef LLMX_PRESERVE_F32
    precise float acc = 0.0;
#else
    float acc = 0.0;
#endif
    for (uint k = kbeg; k < kend; ++k) {
        float a = TILE_REPLAY_WEIGHT(row, k), b = TILE_REPLAY_INPUT(col, k);
        if (abs(a) >= abs(b)) a *= down;
        else b *= down;
#ifdef LLMX_PRESERVE_F32
        acc = fma(a, b, acc);
#else
        acc += a * b;
#endif
    }
    return acc;
}
