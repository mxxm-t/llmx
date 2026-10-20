#ifndef LLMX_IQ4_DOT_GLSL
#define LLMX_IQ4_DOT_GLSL
// Recompute block-aligned ranges from exact lookup weights and the reader's reconstructed int16 inputs.
float iq4_nl_float_twin(uint row, uint col, uint nin, uint begin, uint end) {
    precise float sum = 0.0;
    for (uint b = begin; b < end; ++b) {
        uint o = (row * (nin / IQ4_NL_BLOCK) + b) * IQ4_NL_BYTES;
        float dw = half_at(QBYTE(o), QBYTE(o + 1u));
        for (uint j = 0u; j < IQ4_NL_BLOCK; ++j)
            sum = fma(iq4_nl_at(o, j, dw), iq4_nl_input(col, b, j), sum);
    }
    return sum;
}
// A zero operand is exact; otherwise a subnormal, zero or overflowing combined scale needs recomputation.
bool iq4_nl_unsafe_scale(float weight, float activation, float scale) {
    uint exponent = (floatBitsToUint(scale) >> 23u) & 255u;
    return weight != 0.0 && activation != 0.0 && (exponent == 0u || exponent == 255u);
}
#endif
