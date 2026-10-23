#ifndef LLMX_IQ3_XXS_DOT_GLSL
#define LLMX_IQ3_XXS_DOT_GLSL
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b;
    uint o = (group / 8u) * IQ3_XXS_BYTES;
    dw = half_at(QBYTE(o), QBYTE(o + 1u)) * 0.25;
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
    uint o = (group / 8u) * IQ3_XXS_BYTES;
    uint control = iq3_xxs_control(o, group & 7u);
    uint vector = IQ3_XXS_GRID_WORD(QBYTE(o + 2u + (group & 7u) * 8u + word));
    uint signs = iq3_xxs_signs(control, word / 2u) >> ((word & 1u) * 4u);
    int scale = int(1u + 2u * (control >> 28u));
    uvec2 packed = uvec2(0u);
    for (uint k = 0u; k < 4u; ++k) {
        int value = int((vector >> (8u * k)) & 255u) * scale;
        if (((signs >> k) & 1u) != 0u) value = -value;
        packed[k & 1u] |= (uint(value) & 65535u) << ((k / 2u) * 16u);
    }
    return packed;
}
float lookup_at(uint group, uint j, float dw) {
    return iq3_xxs_at((group / 8u) * IQ3_XXS_BYTES, (group & 7u) * 32u + j, dw * 4.0);
}
#include "lookup_dot.glsl"
#endif
