// IQ2_XXS packs a control word per 32 values; IQ2_XS attaches signs to each index and scales per 16.
uint iq2_small_index(uint type, uint o, uint group32, uint vector8, out uint signs, out uint scale) {
    uint index, code;
    if (type == TYPE_IQ2_XXS) {
        uint base = o + 2u + group32 * 8u;
        uint at = base + 4u;
        uint control = IQ2_SMALL_HALF(at) | (IQ2_SMALL_HALF(at + 2u) << 16u);
        index = QBYTE(base + vector8);
        code = (control >> (7u * vector8)) & 127u;
        scale = control >> 28u;
    } else {
        uint at = o + 2u + group32 * 8u + vector8 * 2u;
        uint packed = IQ2_SMALL_HALF(at);
        index = packed & 511u;
        code = packed >> 9u;
        scale = (QBYTE(o + 66u + group32) >> (4u * (vector8 / 2u))) & 15u;
    }
    signs = code | ((uint(bitCount(code)) & 1u) << 7u);
    return index;
}
float iq2_small_component(uint word, uint lane, uint negative, float scaled) {
    precise float value = scaled * float((word >> (8u * lane)) & 255u);
    return uintBitsToFloat(floatBitsToUint(value) ^ (negative << 31u));
}
float iq2_small_at(uint type, uint o, uint j, float d) {
    uint signs, scale;
    uint index = iq2_small_index(type, o, j / 32u, (j / 8u) & 3u, signs, scale);
    uint word = IQ2_GRID_WORD(2u * index + ((j & 7u) / 4u));
    precise float scaled = d * float(1u + 2u * scale) * 0.125;
    return iq2_small_component(word, j & 3u, (signs >> (j & 7u)) & 1u, scaled);
}
