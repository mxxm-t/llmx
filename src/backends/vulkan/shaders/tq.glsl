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
uvec2 tq_pairs(uint type, uint o, uint group, uint word) {
    uvec2 packed;
    if (type == TYPE_TQ2_0) {
        uint at = o + (group / 4u) * 32u + word * 4u;
        uint shift = 2u * (group & 3u);
        packed = uvec2(QBYTE(at), QBYTE(at + 1u)) | (uvec2(QBYTE(at + 2u), QBYTE(at + 3u)) << 16u);
        packed = (packed >> shift) & uvec2(0x00030003u);
    } else {
        uint plane, at;
        if (group < 5u) {
            plane = group;
            at = word * 4u;
        } else if (group < 7u) {
            uint pos = (group - 5u) * 32u + word * 4u;
            plane = pos / 16u;
            at = 32u + (pos & 15u);
        } else if (word < 4u) {
            plane = 4u;
            at = 32u + word * 4u;
        } else {
            plane = word - 4u;
            at = 48u;
        }
        const uint powers[5] = uint[5](1u, 3u, 9u, 27u, 81u);
        at += o;
        uvec4 codes = uvec4(QBYTE(at), QBYTE(at + 1u), QBYTE(at + 2u), QBYTE(at + 3u));
        codes = (((codes * powers[plane]) & uvec4(255u)) * 3u) >> 8u;
        packed = codes.xy | (codes.zw << 16u);
    }
    // Each field stays below 65536, so centering cannot carry between the signed-16 lanes.
    return (packed + uvec2(0x7fff7fffu)) ^ uvec2(0x80008000u);
}
float tq_at(uint type, uint o, uint j, float d) {
    int code = tq_code(type, o, j);
    // Embedding lacks signed-zero execution modes; preserve the CPU product's sign.
    uint sign = (floatBitsToUint(d) ^ (code < 0 ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(float(code) * d) & 0x7fffffffu) | sign);
}
#endif
