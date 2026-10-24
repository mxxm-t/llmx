#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

// GGUF storage metadata and checked row sizing; execution support belongs to quant::Registry and each backend.
namespace quant {

// A file's dimensions are 64-bit and row_bytes sizes them in size_t, so a narrower size_t would cut them short before its overflow check.
static_assert(sizeof(size_t) >= sizeof(uint64_t), "llmx needs a 64-bit size_t");

constexpr uint32_t GGML_TYPE_F32  = 0;
constexpr uint32_t GGML_TYPE_F16  = 1;
constexpr uint32_t GGML_TYPE_Q4_0 = 2;
constexpr uint32_t GGML_TYPE_Q4_1 = 3;
constexpr uint32_t GGML_TYPE_Q5_0 = 6;
constexpr uint32_t GGML_TYPE_Q5_1 = 7;
constexpr uint32_t GGML_TYPE_Q8_0 = 8;
constexpr uint32_t GGML_TYPE_Q2_K = 10;
constexpr uint32_t GGML_TYPE_Q3_K = 11;
constexpr uint32_t GGML_TYPE_Q4_K = 12;
constexpr uint32_t GGML_TYPE_Q5_K = 13;
constexpr uint32_t GGML_TYPE_Q6_K = 14;
constexpr uint32_t GGML_TYPE_IQ2_XXS = 16;
constexpr uint32_t GGML_TYPE_IQ2_XS = 17;
constexpr uint32_t GGML_TYPE_IQ3_XXS = 18;
constexpr uint32_t GGML_TYPE_IQ1_S = 19;
constexpr uint32_t GGML_TYPE_IQ4_NL = 20;
constexpr uint32_t GGML_TYPE_IQ3_S = 21;
constexpr uint32_t GGML_TYPE_IQ2_S = 22;
constexpr uint32_t GGML_TYPE_IQ4_XS = 23;
constexpr uint32_t GGML_TYPE_IQ1_M = 29;
constexpr uint32_t GGML_TYPE_BF16 = 30;
constexpr uint32_t GGML_TYPE_MXFP4 = 39;

constexpr size_t   Q4_0_BLOCK    = 32;   // values per block
constexpr size_t   Q4_0_TYPESIZE = 18;   // 2-byte f16 scale + 32 nibbles
constexpr size_t   Q4_1_BLOCK    = 32;   // values per block
constexpr size_t   Q4_1_TYPESIZE = 20;   // f16 scale + f16 min + 32 nibbles
constexpr size_t   Q5_0_BLOCK    = 32;
constexpr size_t   Q5_0_TYPESIZE = 22;   // f16 scale + 32 high bits + 32 nibbles
constexpr size_t   Q5_1_BLOCK    = 32;
constexpr size_t   Q5_1_TYPESIZE = 24;   // f16 scale + f16 min + 32 high bits + 32 nibbles
constexpr size_t   Q8_0_BLOCK    = 32;   // values per block
constexpr size_t   Q8_0_TYPESIZE = 34;   // 2-byte f16 scale + 32 int8
constexpr size_t   Q2_K_BLOCK    = 256;
constexpr size_t   Q2_K_TYPESIZE = 84;   // 16 scales/mins + 64 packed codes + 2 f16
constexpr size_t   Q3_K_BLOCK    = 256;
constexpr size_t   Q3_K_TYPESIZE = 110;  // 32 high bits + 64 low codes + 12 scales + f16
constexpr size_t   Q4_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q4_K_TYPESIZE = 144;  // 2 f16 + 12 packed 6-bit + 128 nibbles
constexpr size_t   Q5_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q5_K_TYPESIZE = 176;  // Q4_K plus 32 bytes of fifth bits
constexpr size_t   Q6_K_BLOCK    = 256;  // K-quant super-block
constexpr size_t   Q6_K_TYPESIZE = 210;  // 128 low + 64 high + 16 scales + f16
constexpr size_t   IQ2_XXS_BLOCK = 256;
constexpr size_t   IQ2_XXS_TYPESIZE = 66; // f16 + 32 packed index/sign words
constexpr size_t   IQ2_XS_BLOCK = 256;
constexpr size_t   IQ2_XS_TYPESIZE = 74; // f16 + 32 index/sign words + 8 scale bytes
constexpr size_t   IQ1_S_BLOCK = 256;
constexpr size_t   IQ1_S_TYPESIZE = 50; // f16 + 32 indices + eight scale/shift/high-index words
constexpr size_t   IQ1_M_BLOCK = 256;
constexpr size_t   IQ1_M_TYPESIZE = 56; // 32 indices + 16 high-index/shift bytes + eight scale bytes
constexpr size_t   IQ4_NL_BLOCK = 32;
constexpr size_t   IQ3_XXS_BLOCK = 256;
constexpr size_t   IQ3_XXS_TYPESIZE = 98; // f16 + 64 indices + eight packed scale/sign words
constexpr size_t   IQ3_S_BLOCK = 256;
constexpr size_t   IQ3_S_TYPESIZE = 110; // f16 + 64 indices + 8 high-bit bytes + 32 signs + 4 scales
constexpr size_t   IQ2_S_BLOCK = 256;
constexpr size_t   IQ2_S_TYPESIZE = 82; // f16 + 32 indices + 32 signs + 8 high-bit bytes + 8 scales
constexpr size_t   IQ4_NL_TYPESIZE = 18; // f16 scale + 32 nonuniform lookup indices
constexpr size_t   IQ4_XS_BLOCK = 256;
constexpr size_t   IQ4_XS_TYPESIZE = 136; // f16 scale + eight 6-bit scales + 256 lookup indices

constexpr size_t   MXFP4_BLOCK = 32;
constexpr size_t   MXFP4_TYPESIZE = 17;   // E8M0 scale + 32 E2M1 nibbles

struct StorageType {
    const char* name;
    size_t block_size;   // values per block
    size_t type_size;    // bytes per block
};

// Indexed by the GGML id; removed file layouts have no size and are refused like unknown ids.
inline const StorageType* storage_type(uint32_t id) {
    static constexpr StorageType types[] = {
        {"F32", 1, 4}, {"F16", 1, 2},
        {"Q4_0", Q4_0_BLOCK, Q4_0_TYPESIZE}, {"Q4_1", Q4_1_BLOCK, Q4_1_TYPESIZE},
        {}, {},
        {"Q5_0", Q5_0_BLOCK, Q5_0_TYPESIZE}, {"Q5_1", Q5_1_BLOCK, Q5_1_TYPESIZE},
        {"Q8_0", Q8_0_BLOCK, Q8_0_TYPESIZE}, {"Q8_1", 32, 36},
        {"Q2_K", Q2_K_BLOCK, Q2_K_TYPESIZE}, {"Q3_K", Q3_K_BLOCK, Q3_K_TYPESIZE},
        {"Q4_K", Q4_K_BLOCK, Q4_K_TYPESIZE}, {"Q5_K", Q5_K_BLOCK, Q5_K_TYPESIZE},
        {"Q6_K", Q6_K_BLOCK, Q6_K_TYPESIZE}, {"Q8_K", 256, 292},
        {"IQ2_XXS", IQ2_XXS_BLOCK, IQ2_XXS_TYPESIZE}, {"IQ2_XS", IQ2_XS_BLOCK, IQ2_XS_TYPESIZE}, {"IQ3_XXS", IQ3_XXS_BLOCK, IQ3_XXS_TYPESIZE}, {"IQ1_S", IQ1_S_BLOCK, IQ1_S_TYPESIZE},
        {"IQ4_NL", IQ4_NL_BLOCK, IQ4_NL_TYPESIZE}, {"IQ3_S", IQ3_S_BLOCK, IQ3_S_TYPESIZE}, {"IQ2_S", IQ2_S_BLOCK, IQ2_S_TYPESIZE}, {"IQ4_XS", IQ4_XS_BLOCK, IQ4_XS_TYPESIZE},
        {"I8", 1, 1}, {"I16", 1, 2}, {"I32", 1, 4}, {"I64", 1, 8}, {"F64", 1, 8},
        {"IQ1_M", IQ1_M_BLOCK, IQ1_M_TYPESIZE}, {"BF16", 1, 2},
        {}, {}, {},
        {"TQ1_0", 256, 54}, {"TQ2_0", 256, 66},
        {}, {}, {},
        {"MXFP4", MXFP4_BLOCK, MXFP4_TYPESIZE}, {"NVFP4", 64, 36},
        {"Q1_0", 128, 18}, {"Q2_0", 64, 18},
    };
    return id < sizeof(types) / sizeof(types[0]) && types[id].block_size ? &types[id] : nullptr;
}

// Bytes in `rows` rows of `nin` values of a known storage type, whether or not llmx can execute it.
// Type and whole-block width are checked before overflow; zero rows take zero bytes, however wide a whole-block row.
inline size_t row_bytes(uint32_t type, size_t nin, size_t rows = 1) {
    const StorageType* qt = storage_type(type);
    if (!qt) throw std::runtime_error("quant: unsupported tensor type " + std::to_string(type));
    if (nin % qt->block_size)
        throw std::runtime_error("quant: a row of " + std::to_string(nin) + " values is not whole " + qt->name + " blocks");
    const size_t blocks = nin / qt->block_size;
    if (rows && blocks > std::numeric_limits<size_t>::max() / rows / qt->type_size) throw std::runtime_error("quant: row size overflows");
    return blocks * rows * qt->type_size;
}

} // namespace quant
