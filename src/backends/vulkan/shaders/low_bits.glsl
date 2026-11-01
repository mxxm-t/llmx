#ifndef LLMX_LOW_BITS_GLSL
#define LLMX_LOW_BITS_GLSL
float low_bits_value(uint packed, uint lane, float d) {
    return uintBitsToFloat(floatBitsToUint(d) ^ ((((packed >> lane) & 1u) ^ 1u) << 31u));
}
float low_bits_at(uint o, uint j, float d) {
    return low_bits_value(QBYTE(o + 2u + j / 8u), j % 8u, d);
}
vec4 low_bits_four(uint o, uint j, float d) {
    uint packed = QBYTE(o + 2u + j / 8u), lane = j % 8u;
    return vec4(low_bits_value(packed, lane, d),
                low_bits_value(packed, lane + 1u, d),
                low_bits_value(packed, lane + 2u, d),
                low_bits_value(packed, lane + 3u, d));
}
#ifdef LLMX_LOW_BITS_DOT
const uint LOW_BITS_GROUPS = Q1_0_BLOCK / 32u;
const uint LOW_BITS_BYTES = Q1_0_BYTES;
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b, o = (group / LOW_BITS_GROUPS) * LOW_BITS_BYTES;
    dw = unpackHalf2x16(LOW_BITS_HALF(o)).x;
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
    uint raw = LOW_BITS_HALF((group / 4u) * Q1_0_BYTES + 2u + (group & 3u) * 4u + (word / 4u) * 2u);
    raw >>= (word & 3u) * 4u;
    uvec2 bits = uvec2(raw, raw >> 1u);
    uvec2 mask = (bits & uvec2(1u)) | ((bits & uvec2(4u)) << 14u);
    return ~(mask * uvec2(0x0000fffeu));
}
float lookup_at(uint group, uint j, float dw) {
    return low_bits_at((group / LOW_BITS_GROUPS) * LOW_BITS_BYTES,
                       (group % LOW_BITS_GROUPS) * 32u + j, dw);
}
#endif
#endif
