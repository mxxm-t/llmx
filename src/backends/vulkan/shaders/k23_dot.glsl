#ifndef LLMX_K23_DOT_GLSL
#define LLMX_K23_DOT_GLSL
#ifdef LLMX_Q2
const uint K23_TYPE = TYPE_Q2_K;
const uint K23_BYTES = Q2_K_BYTES;
#else
const uint K23_TYPE = TYPE_Q3_K;
const uint K23_BYTES = Q3_K_BYTES;
#endif

// Q3 blocks alternate word and half-word alignment; both words stay inside the tensor.
uint k23_bytes(uint o) {
    uint low = K23_WORD(o >> 2u), shift = (o & 3u) * 8u;
    uint high = K23_WORD((o >> 2u) + 1u);
    return (low >> shift) | ((high << ((32u - shift) & 31u)) & (0u - uint(shift != 0u)));
}

// Callers start on a group of four, so all bytes use the same code plane.
uint k23_word(uint o, uint first) {
    uint plane = first / 32u, offset = 32u * (first / 128u) + first % 32u;
#ifdef LLMX_Q2
    return (k23_bytes(o + 16u + offset) >> (2u * (plane % 4u))) & 0x03030303u;
#else
    uint low = (k23_bytes(o + 32u + offset) >> (2u * (plane % 4u))) & 0x03030303u;
    uint high = (k23_bytes(o + first % 32u) >> plane) & 0x01010101u;
    // Each absent high bit fills that byte's upper six bits without borrowing from its neighbour.
    return low | ((high ^ 0x01010101u) * 252u);
#endif
}

float k23_term(vec4 weights, float activation, int low, int high, int low_sum, int high_sum) {
#ifdef LLMX_Q2
    float correction = weights.z * float(low_sum) + weights.w * float(high_sum);
    return activation * (weights.x * float(low) + weights.y * float(high) - correction);
#else
    return activation * (weights.x * float(low) + weights.y * float(high));
#endif
}

// Repair uses exact weights and the includer's reader of the same reconstructed int16 inputs.
float k23_float_twin(uint row, uint col, uint nin, uint begin, uint end) {
    precise float sum = 0.0;
    for (uint b = begin; b < end; ++b)
        for (uint j = 0u; j < 32u; ++j)
            sum = fma(quantized_at(K23_TYPE, row, nin, b * 32u + j), k23_input(col, b, j), sum);
    return sum;
}

// Finite half scales, K-quant subscales and 16-value integer sums stay normal inside this activation-scale interval.
bool k23_repair_scale(float scale) {
    float a = abs(scale);
    return a != 0.0 && (a < exp2(-80.0) || a > exp2(80.0));
}
#endif
