// Quantized block decoding shared by the kernels, types being GGUF ids: a table is bound twice, as floats for F32 rows and as bytes for block formats.
#ifndef LLMX_Q_GLSL
#define LLMX_Q_GLSL
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require

const uint TYPE_MXFP4 = 39u;
const uint MXFP4_FIRST_OVERFLOW_EXPONENT = 253u;
const uint MXFP4_BLOCK = 32u;
const uint MXFP4_BYTES = 17u;

const uint TYPE_F32 = 0u;
const uint TYPE_Q4_0 = 2u;
const uint TYPE_Q4_1 = 3u;
const uint TYPE_Q5_0 = 6u;
const uint TYPE_Q5_1 = 7u;
const uint TYPE_Q8_0 = 8u;
const uint TYPE_Q2_K = 10u;
const uint Q2_K_BYTES = 84u;
const uint TYPE_Q3_K = 11u;
const uint Q3_K_BYTES = 110u;
const uint TYPE_Q4_K = 12u;
const uint TYPE_Q5_K = 13u;
const uint TYPE_Q6_K = 14u;
const uint TYPE_IQ4_NL = 20u;
const uint IQ4_NL_BLOCK = 32u;
const uint IQ4_NL_BYTES = 18u;
const uint TYPE_IQ4_XS = 23u;
const uint IQ4_XS_BLOCK = 256u;
const uint IQ4_XS_BYTES = 136u;

// Q4_K: 256 values in 144 bytes; Q5_K adds 32 bytes of fifth bits.
const uint Q4_K_BYTES = 144u;
const uint Q5_K_BYTES = 176u;

// Q4_1: 32 values, two halves d and m, 16 bytes of nibbles; 20 bytes.
const uint Q4_1_BYTES = 20u;
const uint Q5_0_BYTES = 22u;
const uint Q5_1_BYTES = 24u;
// Q6_K: 256 values in 210 bytes; the half scale sits at byte 208.
const uint Q6_K_BLOCK = 256u;
const uint Q6_K_BYTES = 210u;

// Q8_0: 32 values per block, a half scale then 32 signed bytes, 34 bytes.
const uint Q8_0_BLOCK = 32u;
const uint Q8_0_BYTES = 34u;

// Q4_0: 32 values per block, a half scale then 16 bytes of nibbles, 18 bytes; value j < 16 is byte j's low nibble, j + 16 its high one, and value = (nibble - 8) * d.
const uint Q4_0_BLOCK = 32u;
const uint Q4_0_BYTES = 18u;

float q8_0_scale(uint8_t bytes[2]) {
    return unpackHalf2x16(uint(bytes[0]) | (uint(bytes[1]) << 8)).x;
}

float half_at(uint lo, uint hi) {
    return unpackHalf2x16(lo | (hi << 8)).x;
}

// Bytes and values per block of a type.
uint block_bytes(uint type) {
    return type == TYPE_MXFP4 ? MXFP4_BYTES : type == TYPE_Q8_0 ? Q8_0_BYTES : type == TYPE_Q4_0 ? Q4_0_BYTES
         : type == TYPE_Q4_1 ? Q4_1_BYTES : type == TYPE_Q5_0 ? Q5_0_BYTES : type == TYPE_Q5_1 ? Q5_1_BYTES : type == TYPE_IQ4_NL ? IQ4_NL_BYTES : type == TYPE_IQ4_XS ? IQ4_XS_BYTES : type == TYPE_Q4_K ? Q4_K_BYTES
         : type == TYPE_Q5_K ? Q5_K_BYTES : type == TYPE_Q2_K ? Q2_K_BYTES : type == TYPE_Q3_K ? Q3_K_BYTES : Q6_K_BYTES;
}
bool is_kquant(uint type) { return type == TYPE_Q2_K || type == TYPE_Q3_K || type == TYPE_Q4_K || type == TYPE_Q5_K || type == TYPE_Q6_K; }
uint block_values(uint type) { return type == TYPE_IQ4_XS ? IQ4_XS_BLOCK : is_kquant(type) ? Q6_K_BLOCK : 32u; }
#endif
