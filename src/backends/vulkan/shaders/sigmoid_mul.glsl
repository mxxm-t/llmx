// x times sigmoid(g), sigmoid(g) = 1 / (1 + exp(-g)), which sigmoid_mul.comp and the fused routed step's kernels both take.
#ifndef LLMX_SIGMOID_MUL_GLSL
#define LLMX_SIGMOID_MUL_GLSL
float sigmoid_scale(float x, float g) { return x * (1.0 / (1.0 + exp(-g))); }
#endif
