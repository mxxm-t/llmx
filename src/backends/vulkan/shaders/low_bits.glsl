#ifndef LLMX_LOW_BITS_GLSL
#define LLMX_LOW_BITS_GLSL
float low_bits_value(uint type, uint packed, uint lane, float d) {
    // Q1 has only signed unit codes, including when its scale is negative zero.
    if (type == TYPE_Q1_0)
        return uintBitsToFloat(floatBitsToUint(d) ^ ((((packed >> lane) & 1u) ^ 1u) << 31u));
    int code = int((packed >> (2u * lane)) & 3u) - 1;
    // Embedding has no signed-zero execution mode, so retain the product's sign explicitly.
    uint sign = (floatBitsToUint(d) ^ (code < 0 ? 0x80000000u : 0u)) & 0x80000000u;
    return uintBitsToFloat((floatBitsToUint(float(code) * d) & 0x7fffffffu) | sign);
}
float low_bits_at(uint type, uint o, uint j, float d) {
    uint count = type == TYPE_Q1_0 ? 8u : 4u;
    return low_bits_value(type, QBYTE(o + 2u + j / count), j % count, d);
}
// Four adjacent values share one packed byte and one scale in both formats.
vec4 low_bits_four(uint type, uint o, uint j, float d) {
    uint count = type == TYPE_Q1_0 ? 8u : 4u;
    uint packed = QBYTE(o + 2u + j / count), lane = j % count;
    return vec4(low_bits_value(type, packed, lane, d),
                low_bits_value(type, packed, lane + 1u, d),
                low_bits_value(type, packed, lane + 2u, d),
                low_bits_value(type, packed, lane + 3u, d));
}
#ifdef LLMX_LOW_BITS_DOT
// Both formats supply 32-value groups to the shared integer rows and tiles.
#ifdef LLMX_Q1_0
const uint LOW_BITS_GROUPS = Q1_0_BLOCK / 32u;
const uint LOW_BITS_BYTES = Q1_0_BYTES;
const uint LOW_BITS_TYPE = TYPE_Q1_0;
#else
const uint LOW_BITS_GROUPS = Q2_0_BLOCK / 32u;
const uint LOW_BITS_BYTES = Q2_0_BYTES;
const uint LOW_BITS_TYPE = TYPE_Q2_0;
#endif
uint lookup_group(uint row, uint nin, uint b, out float dw) {
    uint group = row * (nin / 32u) + b, o = (group / LOW_BITS_GROUPS) * LOW_BITS_BYTES;
    dw = unpackHalf2x16(LOW_BITS_HALF(o)).x;
    return group;
}
uvec2 lookup_pairs(uint group, uint word) {
#ifdef LLMX_Q1_0
    uint raw = LOW_BITS_HALF((group / 4u) * Q1_0_BYTES + 2u + (group & 3u) * 4u + (word / 4u) * 2u);
    raw >>= (word & 3u) * 4u;
    uvec2 bits = uvec2(raw, raw >> 1u);
    uvec2 mask = (bits & uvec2(1u)) | ((bits & uvec2(4u)) << 14u);
    return ~(mask * uvec2(0x0000fffeu));
#else
    uint raw = LOW_BITS_HALF((group / 2u) * Q2_0_BYTES + 2u + (group & 1u) * 8u + (word / 2u) * 2u);
    raw >>= (word & 1u) * 8u;
    uvec2 packed = uvec2(raw, raw >> 2u);
    packed = (packed & uvec2(3u)) | ((packed & uvec2(48u)) << 12u);
    // Center each unsigned lane independently, retaining Q2's +2 code.
    return (packed + uvec2(0x7fff7fffu)) ^ uvec2(0x80008000u);
#endif
}
float lookup_at(uint group, uint j, float dw) {
    return low_bits_at(LOW_BITS_TYPE, (group / LOW_BITS_GROUPS) * LOW_BITS_BYTES,
                       (group % LOW_BITS_GROUPS) * 32u + j, dw);
}
#endif
#endif
