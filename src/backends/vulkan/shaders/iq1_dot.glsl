#ifndef LLMX_IQ1_DOT_GLSL
#define LLMX_IQ1_DOT_GLSL
// Folding the odd scale into each code gives both formats one d/8 scale per 32 values.
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b;
    dw = iq1_scale(IQ1_TYPE, (group / 8u) * IQ1_BYTES) * 0.125;
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
    uint o = (group / 8u) * IQ1_BYTES;
    int odd, shift;
    uint index = iq1_index(IQ1_TYPE, o, group & 7u, word / 2u, odd, shift);
    uint vector = iq1_vector(index);
    uvec2 packed = uvec2(0u);
    for (uint k = 0u; k < 4u; ++k) {
        int code = odd * (8 * iq1_component(vector, (word & 1u) * 4u + k) + shift);
        packed[k & 1u] |= (uint(code) & 65535u) << ((k / 2u) * 16u);
    }
    return packed;
}
float lookup_at(uint group, uint j, float dw) {
    return iq1_at(IQ1_TYPE, (group / 8u) * IQ1_BYTES, (group & 7u) * 32u + j, dw * 8.0);
}
#include "lookup_dot.glsl"
#endif
