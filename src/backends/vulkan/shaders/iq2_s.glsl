// IQ2_S field decoding over QBYTE and the caller's dictionary word view.
uint iq2_s_vector_word(uint o, uint j) {
    uint g = j / 8u;
    uint index = QBYTE(o + 2u + g) | (((QBYTE(o + 66u + g / 4u) >> (2u * (g & 3u))) & 3u) << 8u);
    return IQ2_GRID_WORD(2u * index + ((j & 7u) / 4u));
}
float iq2_s_scale(uint o, uint group16, float d) {
    uint scale = (QBYTE(o + 74u + group16 / 2u) >> (4u * (group16 & 1u))) & 15u;
    precise float scaled = d * float(1u + 2u * scale) * 0.125;
    return scaled;
}
float iq2_s_component(uint word, uint lane, uint negative, float scaled) {
    precise float value = scaled * float((word >> (8u * lane)) & 255u);
    return uintBitsToFloat(floatBitsToUint(value) ^ (negative << 31u));
}
float iq2_s_at(uint o, uint j, float d) {
    uint word = iq2_s_vector_word(o, j);
    uint negative = (QBYTE(o + 34u + j / 8u) >> (j & 7u)) & 1u;
    return iq2_s_component(word, j & 3u, negative, iq2_s_scale(o, j / 16u, d));
}
