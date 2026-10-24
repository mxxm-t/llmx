#ifndef LLMX_TQ_DOT_GLSL
#define LLMX_TQ_DOT_GLSL
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b;
    dw = tq_scale(TQ_TYPE, (group / 8u) * TQ_BYTES);
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
    uint o = (group / 8u) * TQ_BYTES, j = (group & 7u) * 32u + word * 4u;
    uvec2 packed = uvec2(0u);
    for (uint k = 0u; k < 4u; ++k) {
        int value = tq_code(TQ_TYPE, o, j + k);
        packed[k & 1u] |= (uint(value) & 65535u) << ((k / 2u) * 16u);
    }
    return packed;
}
float lookup_at(uint group, uint j, float dw) {
    return tq_at(TQ_TYPE, (group / 8u) * TQ_BYTES, (group & 7u) * 32u + j, dw);
}
#include "lookup_dot.glsl"
#endif
