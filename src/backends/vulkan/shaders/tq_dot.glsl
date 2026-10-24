#ifndef LLMX_TQ_DOT_GLSL
#define LLMX_TQ_DOT_GLSL
#include "tq_packed.glsl"
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b;
    dw = tq_scale(TQ_TYPE, (group / 8u) * TQ_BYTES);
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
    return tq_pairs(TQ_TYPE, (group / 8u) * TQ_BYTES, group & 7u, word);
}
float lookup_at(uint group, uint j, float dw) {
    return tq_at(TQ_TYPE, (group / 8u) * TQ_BYTES, (group & 7u) * 32u + j, dw);
}
#include "lookup_dot.glsl"
#endif
