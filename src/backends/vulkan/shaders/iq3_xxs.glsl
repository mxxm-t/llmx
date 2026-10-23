// Packed sign/scale words may cross an aligned word boundary between blocks.
uint iq3_xxs_control(uint o, uint group) {
    uint at = o + 66u + 4u * group;
    return QBYTE(at) | (QBYTE(at + 1u) << 8u) | (QBYTE(at + 2u) << 16u) | (QBYTE(at + 3u) << 24u);
}
uint iq3_xxs_signs(uint control, uint vector8) {
    uint code = (control >> (7u * vector8)) & 127u;
    return code | ((uint(bitCount(code)) & 1u) << 7u);
}
float iq3_xxs_scale(uint control, float d) {
    precise float scaled = d * float(1u + 2u * (control >> 28u)) * 0.25;
    return scaled;
}
float iq3_xxs_component(uint word, uint lane, uint negative, float scaled) {
    precise float value = scaled * float((word >> (8u * lane)) & 255u);
    return uintBitsToFloat(floatBitsToUint(value) ^ (negative << 31u));
}
float iq3_xxs_at(uint o, uint j, float d) {
    uint control = iq3_xxs_control(o, j / 32u);
    uint signs = iq3_xxs_signs(control, (j & 31u) / 8u);
    uint word = IQ3_XXS_GRID_WORD(QBYTE(o + 2u + j / 4u));
    return iq3_xxs_component(word, j & 3u, (signs >> (j & 7u)) & 1u, iq3_xxs_scale(control, d));
}
