#ifndef LLMX_DOT16_GLSL
#define LLMX_DOT16_GLSL
// Position pairs (0,2), (1,3) become four signed high bytes and four biased low bytes.
uvec2 dot16_parts(uvec2 pairs) {
    uint high = ((pairs.x >> 8u) & 0x00ff00ffu) | (pairs.y & 0xff00ff00u);
    uint low = ((pairs.x & 0x00ff00ffu) | ((pairs.y << 8u) & 0xff00ff00u)) ^ 0x80808080u;
    return uvec2(high, low);
}
// Each signed value is 256 times its high byte plus its biased low byte plus 128.
int dot16(uint weights, uvec2 parts) {
    return 256 * dotPacked4x8EXT(int(weights), int(parts.x)) + dotPacked4x8EXT(int(weights), int(parts.y)) +
           128 * dotPacked4x8EXT(int(weights), 0x01010101);
}
#endif
