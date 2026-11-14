#ifndef LLMX_IQ4_PAIRS_GLSL
#define LLMX_IQ4_PAIRS_GLSL
// Set for the row bodies, which take lookup_pairs8 where it is defined.
#define LLMX_IQ4_PAIRS
// The row kernel's table of two codes' values as one pair word, code a in the low half and code b in the high half at entry a | b << 4.
shared uint iq4_pairs[256];
void iq4_pairs_init() {
    uint i = gl_LocalInvocationIndex;
    iq4_pairs[i] = (uint(iq4_nl_value(i & 15u)) & 0xFFFFu) | (uint(iq4_nl_value(i >> 4u)) << 16u);
    barrier();
}
// A group's 16 code bytes at an even byte offset: four aligned words, the last halfword loaded alone so a block ending halfway through a word stays inside the row.
uvec4 iq4_codes(uint group) {
    uint word = group >> 2u;
    uvec4 lo = uvec4(w32s[pi].w32[word], w32s[pi].w32[word + 1u], w32s[pi].w32[word + 2u], w32s[pi].w32[word + 3u]);
    uint tail = uint(w16s[pi].w16[(group + 14u) >> 1u]);
    uvec4 shifted = uvec4((lo.x >> 16u) | (lo.y << 16u), (lo.y >> 16u) | (lo.z << 16u), (lo.z >> 16u) | (lo.w << 16u), (lo.w >> 16u) | (tail << 16u));
    return (group & 2u) != 0u ? shifted : lo;
}
// The eight pair words lookup_pairs gives, two table entries a word: byte k of word w holds codes 4w + k (low nibble) and 4w + k + 16 (high nibble).
void lookup_pairs8(uint group, out uvec2 weights[8]) {
    uvec4 q = iq4_codes(group);
    [[unroll]] for (uint w = 0u; w < 4u; ++w) {
        uint c = q[w];
        weights[w] = uvec2(iq4_pairs[(c & 15u) | ((c >> 12u) & 240u)], iq4_pairs[((c >> 8u) & 15u) | ((c >> 20u) & 240u)]);
        weights[w + 4u] = uvec2(iq4_pairs[((c >> 4u) & 15u) | ((c >> 16u) & 240u)], iq4_pairs[((c >> 12u) & 15u) | ((c >> 24u) & 240u)]);
    }
}
#endif
