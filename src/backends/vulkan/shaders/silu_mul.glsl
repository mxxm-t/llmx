// silu(gate) times up in the form the CPU uses, g / (1 + exp(-g)) * u, which silu_mul.comp and the fused routed step's kernels both take.
#ifndef LLMX_SILU_MUL_GLSL
#define LLMX_SILU_MUL_GLSL
float silu_mul_value(float g, float u) { return g / (1.0 + exp(-g)) * u; }
vec4 silu_mul_value(vec4 g, vec4 u) { return g / (1.0 + exp(-g)) * u; }
#endif
