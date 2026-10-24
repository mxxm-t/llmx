// Both IQ1 formats share ternary vectors; their scale and index fields stay distinct.
uint iq1_half(uint o) {
    return QBYTE(o) | (QBYTE(o + 1u) << 8u);
}
float iq1_scale(uint type, uint o) {
    uint bits;
    if (type == TYPE_IQ1_S) bits = iq1_half(o);
    else bits = (QBYTE(o + 49u) >> 4u) | (QBYTE(o + 51u) & 240u) |
                ((QBYTE(o + 53u) & 240u) << 4u) | ((QBYTE(o + 55u) & 240u) << 8u);
    return half_at(bits & 255u, bits >> 8u);
}
uint iq1_index(uint type, uint o, uint group32, uint vector8, out int odd, out int shift) {
    if (type == TYPE_IQ1_S) {
        uint control = iq1_half(o + 34u + 2u * group32);
        odd = int(1u + 2u * ((control >> 12u) & 7u));
        shift = (control & 32768u) != 0u ? -1 : 1;
        return QBYTE(o + 2u + 4u * group32 + vector8) |
               (((control >> (3u * vector8)) & 7u) << 8u);
    }
    uint control = QBYTE(o + 32u + 2u * group32 + vector8 / 2u) >> (4u * (vector8 & 1u));
    uint scales = iq1_half(o + 48u + 2u * (group32 / 2u));
    odd = int(1u + 2u * ((scales >> (6u * (group32 & 1u) + 3u * (vector8 / 2u))) & 7u));
    shift = (control & 8u) != 0u ? -1 : 1;
    return QBYTE(o + 4u * group32 + vector8) | ((control & 7u) << 8u);
}
uint iq1_vector(uint index) {
    return IQ1_GRID_WORD(index / 2u) >> (16u * (index & 1u));
}
int iq1_component(uint vector, uint lane) {
    return int((vector >> (2u * lane)) & 3u) - 1;
}
float iq1_at(uint type, uint o, uint j, float d) {
    int odd, shift;
    uint index = iq1_index(type, o, j / 32u, (j / 8u) & 3u, odd, shift);
    precise float scale = d * float(odd);
    precise float value = float(iq1_component(iq1_vector(index), j & 7u)) + float(shift) * 0.125;
    return scale * value;
}
