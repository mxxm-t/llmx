#ifndef LLMX_IQ3_DOT_GLSL
#define LLMX_IQ3_DOT_GLSL
uint lookup_group(uint row, uint nin, uint b, out float dw) { return iq3_s_group(row, nin, b, dw); }
float lookup_at(uint group, uint j, float dw) { return iq3_s_group_at(group, j, dw); }
uint lookup_word(uint group, uint word) { return iq3_s_word(group, word); }
#include "lookup_dot.glsl"
#endif
