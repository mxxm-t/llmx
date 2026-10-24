#ifndef LLMX_TQ_PACKED_GLSL
#define LLMX_TQ_PACKED_GLSL
// Packed groups start on an even byte; two word reads cover their four bytes.
uint tq_bytes4(uint at) {
    uint low = (TQ_WORD(at >> 2u) >> ((at & 2u) * 8u)) & 65535u;
    uint high = (TQ_WORD((at + 2u) >> 2u) >> (((at + 2u) & 2u) * 8u)) & 65535u;
    return low | (high << 16u);
}
uvec2 tq_pairs(uint type, uint o, uint group, uint word) {
    uvec2 packed;
    if (type == TYPE_TQ2_0) {
        uint at = o + (group / 4u) * 32u + word * 4u;
        uint shift = 2u * (group & 3u);
        uint raw = tq_bytes4(at);
        packed = uvec2(raw, raw >> 8u) & uvec2(0x00ff00ffu);
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
        uint raw = tq_bytes4(at);
        uvec4 codes = uvec4(raw, raw >> 8u, raw >> 16u, raw >> 24u) & uvec4(255u);
        codes = (((codes * powers[plane]) & uvec4(255u)) * 3u) >> 8u;
        packed = codes.xy | (codes.zw << 16u);
    }
    // Each field stays below 65536, so centering cannot carry between the signed-16 lanes.
    return (packed + uvec2(0x7fff7fffu)) ^ uvec2(0x80008000u);
}
#endif
