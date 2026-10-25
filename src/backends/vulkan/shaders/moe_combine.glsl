// A routed slot joining its row's sum (backend.hpp matmul_experts_add), for moe_combine.comp and the fused down kernel: slot 0's product is the sum, each later product is rounded, then added.
// It is written out, not left to contraction, which follows the float modes, so preserving and ordinary modules give a normal input the same bits.
#ifndef LLMX_MOE_COMBINE_GLSL
#define LLMX_MOE_COMBINE_GLSL
float combine_add(float s, float w, float part, uint j) {
    precise float product = w * part;
    precise float sum = j == 0u ? product : s + product;
    return sum;
}
#endif
