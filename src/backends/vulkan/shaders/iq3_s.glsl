// IQ3_S format decoding over QBYTE and the caller's IQ3_GRID_WORD dictionary view.
uint iq3_s_vector(uint o, uint group, uint high) {
    uint index = QBYTE(o + 2u + group) | (((high >> (group & 7u)) & 1u) << 8u);
    return IQ3_GRID_WORD(index);
}
float iq3_s_scale(uint o, uint group32, float d) {
    uint scale = (QBYTE(o + 106u + group32 / 2u) >> (4u * (group32 & 1u))) & 15u;
    precise float scaled = d * float(1u + 2u * scale);
    return scaled;
}
float iq3_s_component(uint vector, uint lane, uint negative, float scaled, uint sign) {
    uint magnitude = (vector >> (8u * lane)) & 255u;
    precise float value = scaled * float(magnitude);
    return uintBitsToFloat((floatBitsToUint(value) & 0x7fffffffu) | (sign ^ (negative << 31u)));
}
float iq3_s_at(uint o, uint j, float d) {
    uint vector = iq3_s_vector(o, j / 4u, QBYTE(o + 66u + j / 32u));
    uint negative = (QBYTE(o + 74u + j / 8u) >> (j & 7u)) & 1u;
    return iq3_s_component(vector, j & 3u, negative, iq3_s_scale(o, j / 32u, d), floatBitsToUint(d) & 0x80000000u);
}

// One dot group uses signed dictionary magnitudes and a separate d * odd subscale.
uint iq3_s_group(uint row, uint width, uint b, out float dw) {
    uint o = (row * (width / IQ3_S_BLOCK) + b / 8u) * IQ3_S_BYTES;
    dw = iq3_s_scale(o, b & 7u, half_at(QBYTE(o), QBYTE(o + 1u)));
    return row * (width / 32u) + b;
}
uint iq3_s_word(uint group, uint word) {
    uint g = group & 7u, o = (group / 8u) * IQ3_S_BYTES;
    uint vector = iq3_s_vector(o, g * 8u + word, QBYTE(o + 66u + g));
    uint signs = QBYTE(o + 74u + g * 4u + word / 2u) >> (4u * (word & 1u));
    uint packed = 0u;
    for (uint k = 0u; k < 4u; ++k) {
        int value = int((vector >> (8u * k)) & 255u);
        if (((signs >> k) & 1u) != 0u) value = -value;
        packed |= (uint(value) & 255u) << (8u * k);
    }
    return packed;
}
float iq3_s_group_at(uint group, uint j, float dw) {
    uint g = group & 7u, o = (group / 8u) * IQ3_S_BYTES;
    uint vector = iq3_s_vector(o, g * 8u + j / 4u, QBYTE(o + 66u + g));
    uint negative = (QBYTE(o + 74u + g * 4u + j / 8u) >> (j & 7u)) & 1u;
    return iq3_s_component(vector, j & 3u, negative, dw, floatBitsToUint(dw) & 0x80000000u);
}
