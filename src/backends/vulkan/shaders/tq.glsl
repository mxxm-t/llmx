#ifndef LLMX_TQ_GLSL
#define LLMX_TQ_GLSL
// Both layouts center their codes at one. TQ1 expands wrapped radix bytes.
int tq_code(uint type, uint o, uint j) {
    if (type == TYPE_TQ2_0) {
        uint packed = QBYTE(o + (j / 128u) * 32u + (j & 31u));
        return int((packed >> (2u * ((j / 32u) & 3u))) & 3u) - 1;
    }
    uint plane, at;
    if (j < 160u) { plane = j / 32u; at = j & 31u; }
    else if (j < 240u) { plane = (j - 160u) / 16u; at = 32u + ((j - 160u) & 15u); }
    else { plane = (j - 240u) / 4u; at = 48u + (j & 3u); }
    const uint powers[5] = uint[5](1u, 3u, 9u, 27u, 81u);
    uint wrapped = (QBYTE(o + at) * powers[plane]) & 255u;
    return int((wrapped * 3u) >> 8u) - 1;
}
float tq_scale(uint type, uint o) {
    uint at = o + (type == TYPE_TQ1_0 ? 52u : 64u);
    return half_at(QBYTE(at), QBYTE(at + 1u));
}
float tq_at(uint type, uint o, uint j, float d) {
    int code = tq_code(type, o, j);
    // Embedding lacks signed-zero execution modes; preserve the CPU product's sign.
    uint sign = (floatBitsToUint(d) ^ (code < 0 ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(float(code) * d) & 0x7fffffffu) | sign);
}
#endif
