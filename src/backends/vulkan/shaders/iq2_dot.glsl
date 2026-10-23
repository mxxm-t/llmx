#ifndef LLMX_IQ2_DOT_GLSL
#define LLMX_IQ2_DOT_GLSL
// Fold the two odd subscales into signed int16 weights, leaving one d/8 scale per group of 32.
#ifdef LLMX_IQ2_SMALL
#define IQ2_DOT_BYTES IQ2_SMALL_BYTES
#else
#define IQ2_DOT_BYTES IQ2_S_BYTES
#endif
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b;
    uint o = (group / 8u) * IQ2_DOT_BYTES;
    dw = half_at(QBYTE(o), QBYTE(o + 1u)) * 0.125;
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
    uint o = (group / 8u) * IQ2_DOT_BYTES, j = (group & 7u) * 32u + word * 4u;
#ifdef LLMX_IQ2_SMALL
    uint signs, subscale;
    uint index = iq2_small_index(IQ2_SMALL_TYPE, o, group & 7u, word / 2u, signs, subscale);
    uint vector = IQ2_GRID_WORD(2u * index + ((j & 7u) / 4u));
    int scale = int(1u + 2u * subscale);
    signs >>= j & 7u;
#else
    uint vector = iq2_s_vector_word(o, j);
    uint scales = QBYTE(o + 74u + (group & 7u));
    int scale = int(1u + 2u * ((scales >> (4u * (word / 4u))) & 15u));
    uint signs = QBYTE(o + 34u + j / 8u) >> (j & 7u);
#endif
    uvec2 packed = uvec2(0u);
    for (uint k = 0u; k < 4u; ++k) {
        int value = int((vector >> (8u * k)) & 255u) * scale;
        if (((signs >> k) & 1u) != 0u) value = -value;
        packed[k & 1u] |= (uint(value) & 65535u) << ((k / 2u) * 16u);
    }
    return packed;
}
float lookup_at(uint group, uint j, float dw) {
#ifdef LLMX_IQ2_SMALL
    return iq2_small_at(IQ2_SMALL_TYPE, (group / 8u) * IQ2_DOT_BYTES, (group & 7u) * 32u + j, dw * 8.0);
#else
    return iq2_s_at((group / 8u) * IQ2_S_BYTES, (group & 7u) * 32u + j, dw * 8.0);
#endif
}
#include "lookup_dot.glsl"
#endif
