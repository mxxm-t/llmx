#ifndef LLMX_IQ4_DOT_GLSL
#define LLMX_IQ4_DOT_GLSL
#ifdef LLMX_IQ4_XS
const uint IQ4_TYPE = TYPE_IQ4_XS;
#else
const uint IQ4_TYPE = TYPE_IQ4_NL;
#endif
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    return iq4_group(IQ4_TYPE, row, nin, b, dw);
}
float lookup_at(uint group, uint j, float dw) { return iq4_at(group, j, dw); }
uvec2 lookup_pairs(uint group, uint word) {
    uint packed = 0u;
    for (uint k = 0u; k < 4u; ++k) {
        uint pos = word * 4u + k;
        uint byte = QBYTE(group + (pos & 15u));
        uint code = pos < 16u ? byte & 15u : byte >> 4u;
        packed |= (uint(iq4_nl_value(code)) & 255u) << (8u * k);
    }
    return weight16(packed);
}
#include "lookup_dot.glsl"
#endif
