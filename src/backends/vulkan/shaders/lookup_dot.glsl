#ifndef LLMX_LOOKUP_DOT_GLSL
#define LLMX_LOOKUP_DOT_GLSL
// Both lookup formats repair from exact weights and the same reconstructed int16 inputs.
float lookup_float_twin(uint row, uint col, uint nin, uint begin, uint end) {
    precise float sum = 0.0;
    for (uint b = begin; b < end; ++b) {
        float dw;
        uint group = lookup_group(row, nin, b, dw);
        for (uint j = 0u; j < 32u; ++j)
            sum = fma(lookup_at(group, j, dw), lookup_input(col, b, j), sum);
    }
    return sum;
}
// A zero operand is exact; otherwise a subnormal, zero or overflowing combined scale needs recomputation.
bool lookup_unsafe_scale(float weight, float activation, float scale) {
    uint exponent = (floatBitsToUint(scale) >> 23u) & 255u;
    return weight != 0.0 && activation != 0.0 && (exponent == 0u || exponent == 255u);
}
#endif
