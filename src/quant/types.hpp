#pragma once
#include <cstddef>
#include <cstdint>

// The storage types llmx reads, by the GGML type ids that GGUF files and the backends name them by, and the block of each quantized type: the values it holds and its bytes.
// quant::Registry pairs each id with its kernels, and quant::row_bytes sizes rows by it, the format layer's tensors included.
namespace quant {

// A file's dimensions are 64-bit and row_bytes sizes them in size_t, so a narrower size_t would cut them short before its overflow check.
static_assert(sizeof(size_t) >= sizeof(uint64_t), "llmx needs a 64-bit size_t");

constexpr uint32_t GGML_TYPE_F32  = 0;
constexpr uint32_t GGML_TYPE_Q4_0 = 2;
constexpr uint32_t GGML_TYPE_Q4_1 = 3;
constexpr uint32_t GGML_TYPE_Q8_0 = 8;
constexpr uint32_t GGML_TYPE_Q4_K = 12;
constexpr uint32_t GGML_TYPE_Q5_K = 13;
constexpr uint32_t GGML_TYPE_Q6_K = 14;
constexpr uint32_t GGML_TYPE_MXFP4 = 39;

constexpr size_t   Q4_0_BLOCK    = 32;   // values per block
constexpr size_t   Q4_0_TYPESIZE = 18;   // 2-byte f16 scale + 32 nibbles
constexpr size_t   Q4_1_BLOCK    = 32;   // values per block
constexpr size_t   Q4_1_TYPESIZE = 20;   // f16 scale + f16 min + 32 nibbles
constexpr size_t   Q8_0_BLOCK    = 32;   // values per block
constexpr size_t   Q8_0_TYPESIZE = 34;   // 2-byte f16 scale + 32 int8
constexpr size_t   Q4_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q4_K_TYPESIZE = 144;  // 2 f16 + 12 packed 6-bit + 128 nibbles
constexpr size_t   Q5_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q5_K_TYPESIZE = 176;  // Q4_K plus 32 bytes of fifth bits
constexpr size_t   Q6_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q6_K_TYPESIZE = 210;  // 128 low + 64 high + 16 scales + f16

constexpr size_t   MXFP4_BLOCK = 32;
constexpr size_t   MXFP4_TYPESIZE = 17;   // E8M0 scale + 32 E2M1 nibbles

} // namespace quant
