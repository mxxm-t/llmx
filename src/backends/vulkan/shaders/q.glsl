// Quantized block decoding shared by the kernels. A table is bound twice,
// once as floats for F32 rows and once as bytes for block formats, and the
// caller says which through a push constant. Types are the GGUF ids the
// registry is keyed by (docs/VULKAN.md).
#ifndef LLMX_Q_GLSL
#define LLMX_Q_GLSL
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require

const uint TYPE_F32 = 0u;
const uint TYPE_Q8_0 = 8u;

// Q8_0: 32 values per block, a half scale then 32 signed bytes, 34 bytes.
const uint Q8_0_BLOCK = 32u;
const uint Q8_0_BYTES = 34u;

float q8_0_scale(uint8_t bytes[2]) {
    return unpackHalf2x16(uint(bytes[0]) | (uint(bytes[1]) << 8)).x;
}
#endif
