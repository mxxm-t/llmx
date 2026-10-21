#ifndef LLMX_IQ4_DOT_GLSL
#define LLMX_IQ4_DOT_GLSL
#ifdef LLMX_IQ4_XS
const uint IQ4_TYPE = TYPE_IQ4_XS;
#else
const uint IQ4_TYPE = TYPE_IQ4_NL;
#endif
// Recompute block-aligned ranges from exact lookup weights and the reader's reconstructed int16 inputs.
float iq4_float_twin(uint row, uint col, uint nin, uint begin, uint end) {
    precise float sum = 0.0;
    for (uint b = begin; b < end; ++b) {
        float dw;
        uint codes = iq4_group(IQ4_TYPE, row, nin, b, dw);
        for (uint j = 0u; j < IQ4_NL_BLOCK; ++j)
            sum = fma(iq4_at(codes, j, dw), iq4_input(col, b, j), sum);
    }
    return sum;
}
// A zero operand is exact; otherwise a subnormal, zero or overflowing combined scale needs recomputation.
bool iq4_unsafe_scale(float weight, float activation, float scale) {
    uint exponent = (floatBitsToUint(scale) >> 23u) & 255u;
    return weight != 0.0 && activation != 0.0 && (exponent == 0u || exponent == 255u);
}
#endif
