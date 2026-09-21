// Quantized block decoding shared by the kernels. A table is bound twice,
// once as floats for F32 rows and once as bytes for block formats, and the
// caller says which through a push constant. Types are the GGUF ids the
// registry is keyed by (docs/VULKAN.md).
#ifndef LLMX_Q_GLSL
#define LLMX_Q_GLSL
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require

const uint TYPE_F32 = 0u;
const uint TYPE_Q4_0 = 2u;
const uint TYPE_Q4_1 = 3u;
const uint TYPE_Q8_0 = 8u;
const uint TYPE_Q6_K = 14u;

// Q4_1: 32 values, two halves d and m, 16 bytes of nibbles; 20 bytes.
const uint Q4_1_BYTES = 20u;
// Q6_K: 256 values in 210 bytes; the half scale sits at byte 208.
const uint Q6_K_BLOCK = 256u;
const uint Q6_K_BYTES = 210u;

// Q8_0: 32 values per block, a half scale then 32 signed bytes, 34 bytes.
const uint Q8_0_BLOCK = 32u;
const uint Q8_0_BYTES = 34u;

// Q4_0: 32 values per block, a half scale then 16 bytes of nibbles, 18
// bytes; value j < 16 is the low nibble of byte j, value j + 16 the high
// nibble, and the value is (nibble - 8) * d, computed as nibble * d - 8 * d
// in the CPU's order.
const uint Q4_0_BLOCK = 32u;
const uint Q4_0_BYTES = 18u;

float q8_0_scale(uint8_t bytes[2]) {
    return unpackHalf2x16(uint(bytes[0]) | (uint(bytes[1]) << 8)).x;
}

float half_at(uint8_t lo, uint8_t hi) {
    return unpackHalf2x16(uint(lo) | (uint(hi) << 8)).x;
}

// Bytes per block and values per block of a type, for the callers that
// walk rows generically.
uint block_bytes(uint type) {
    return type == TYPE_Q8_0 ? Q8_0_BYTES : type == TYPE_Q4_0 ? Q4_0_BYTES
         : type == TYPE_Q4_1 ? Q4_1_BYTES : Q6_K_BYTES;
}
uint block_values(uint type) { return type == TYPE_Q6_K ? Q6_K_BLOCK : 32u; }
#endif
