#ifndef LLMX_DOT16_GLSL
#define LLMX_DOT16_GLSL
// Products of signed 8-bit weights and the 16-bit activation twin through two-wide 16-bit integer dots, exact in 32 bits; the includer enables the 16-bit integer types and the integer dot product.
// Four signed quants of a word, as the twin holds four values: values 0 and 2 in one pair of 16-bit halves, 1 and 3 in the other.
uvec2 weight16(uint w) {
    return uvec2(pack32(i16vec2(unpack16(w << 8u)) >> int16_t(8)), pack32(i16vec2(unpack16(w)) >> int16_t(8)));
}
// Four products of 16-bit values added to acc through two two-wide dots; acc goes in first, so each dot takes the sum so far as its accumulator.
int dot_pairs(uvec2 w, uvec2 x, int acc) {
    return acc + dotEXT(i16vec2(unpack16(w.x)), i16vec2(unpack16(x.x))) + dotEXT(i16vec2(unpack16(w.y)), i16vec2(unpack16(x.y)));
}
#endif
