// silu(gate) times up in the form the CPU uses, g / (1 + exp(-g)) * u, which silu_mul.comp and the fused routed step's kernels both take.
#ifndef LLMX_SILU_MUL_GLSL
#define LLMX_SILU_MUL_GLSL
float silu_mul_value(float g, float u) { return g / (1.0 + exp(-g)) * u; }
vec4 silu_mul_value(vec4 g, vec4 u) { return g / (1.0 + exp(-g)) * u; }
// MiniMax-M3.1's quick_gelu in the CPU's order, (u + 1) (g sigmoid(1.702 g)) (docs/MINIMAX.md, The forward pass).
float quick_gelu_value(float g, float u) { return (u + 1.0) * (g * (1.0 / (1.0 + exp(-1.702 * g)))); }
vec4 quick_gelu_value(vec4 g, vec4 u) { return (u + 1.0) * (g * (1.0 / (1.0 + exp(-1.702 * g)))); }
#endif
