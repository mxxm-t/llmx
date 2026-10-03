// Vulkan backend (docs/VULKAN.md). Everything runs on one compute queue: ops record into an open command buffer, submit() ends it and signals a timeline semaphore with the ticket, wait() blocks on it, and a ring of command buffers is reused once their tickets retire.
#include "backends/device_profile.hpp"
#include "backends/kv_storage.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "quant/quant.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace backend {
namespace {

// Every entry point this file uses, fetched through the loader at run time so nothing links against vulkan-1.
#define LLMX_VK_GLOBAL_FUNCTIONS(X) \
    X(vkCreateInstance) \
    X(vkEnumerateInstanceVersion)
#define LLMX_VK_INSTANCE_FUNCTIONS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceMemoryProperties2) \
    X(vkGetPhysicalDeviceFeatures2) \
    X(vkEnumerateDeviceExtensionProperties) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)
#define LLMX_VK_DEVICE_FUNCTIONS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkDeviceWaitIdle) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkFreeCommandBuffers) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkQueueSubmit) \
    X(vkCreateSemaphore) \
    X(vkDestroySemaphore) \
    X(vkWaitSemaphores) \
    X(vkCreateEvent) \
    X(vkDestroyEvent) \
    X(vkSetEvent) \
    X(vkResetEvent) \
    X(vkCmdWaitEvents) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
    X(vkUnmapMemory) \
    X(vkCmdCopyBuffer) \
    X(vkCmdFillBuffer) \
    X(vkCmdPipelineBarrier) \
    X(vkCreateShaderModule) \
    X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) \
    X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) \
    X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) \
    X(vkCmdBindPipeline) \
    X(vkCmdPushConstants) \
    X(vkCmdDispatch) \
    X(vkCreateQueryPool) \
    X(vkDestroyQueryPool) \
    X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) \
    X(vkGetQueryPoolResults)

struct Fn {
#define LLMX_VK_DECLARE(name) PFN_##name name = nullptr;
    LLMX_VK_GLOBAL_FUNCTIONS(LLMX_VK_DECLARE)
    LLMX_VK_INSTANCE_FUNCTIONS(LLMX_VK_DECLARE)
    LLMX_VK_DEVICE_FUNCTIONS(LLMX_VK_DECLARE)
#undef LLMX_VK_DECLARE
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = nullptr;
};

// The kernels, compiled by glslc at build time into the generated include directory as comma-separated words.
const uint32_t kSpvAdd[] = {
#include "vulkan/add.inc"
};
const uint32_t kSpvSiluMul[] = {
#include "vulkan/silu_mul.inc"
};
const uint32_t kSpvGatherRows[] = {
#include "vulkan/gather_rows.inc"
};
const uint32_t kSpvRmsNormRows[] = {
#include "vulkan/rms_norm_rows.inc"
};
const uint32_t kSpvNormRopePartial[] = {
#include "vulkan/norm_rope_partial.inc"
};
const uint32_t kSpvEmbed[] = {
#include "vulkan/embed.inc"
};
const uint32_t kSpvMatmulRow[] = {
#include "vulkan/matmul_row.inc"
};
const uint32_t kSpvMatmulRowQ8W[] = {
#include "vulkan/matmul_row_q8w.inc"
};
const uint32_t kSpvMatmulRowQ4[] = {
#include "vulkan/matmul_row_q4.inc"
};
const uint32_t kSpvMatmulRowK4[] = {
#include "vulkan/matmul_row_k4.inc"
};
const uint32_t kSpvMatmulRowK5[] = {
#include "vulkan/matmul_row_k5.inc"
};
const uint32_t kSpvMatmulRowK[] = {
#include "vulkan/matmul_row_k.inc"
};
const uint32_t kSpvQuantizeX[] = {
#include "vulkan/quantize_x.inc"
};
const uint32_t kSpvKvWrite[] = {
#include "vulkan/kv_write.inc"
};
const uint32_t kSpvNormRopeKv[] = {
#include "vulkan/norm_rope_kv.inc"
};
const uint32_t kSpvAttention[] = {
#include "vulkan/attention.inc"
};
const uint32_t kSpvAttentionMerge[] = {
#include "vulkan/attention_merge.inc"
};
const uint32_t kSpvAttentionTile[] = {
#include "vulkan/attention_tile.inc"
};
// The cache-type variants of every kernel that touches K or V, in kv_variant()'s order: K f16, V f16, both.
const uint32_t kSpvKvWriteK16[] = {
#include "vulkan/kv_write_k16.inc"
};
const uint32_t kSpvKvWriteV16[] = {
#include "vulkan/kv_write_v16.inc"
};
const uint32_t kSpvKvWriteKV16[] = {
#include "vulkan/kv_write_kv16.inc"
};
const uint32_t kSpvAttentionG4[] = {
#include "vulkan/attention_g4.inc"
};
const uint32_t kSpvAttentionK16G4[] = {
#include "vulkan/attention_k16_g4.inc"
};
const uint32_t kSpvAttentionV16G4[] = {
#include "vulkan/attention_v16_g4.inc"
};
const uint32_t kSpvAttentionKV16G4[] = {
#include "vulkan/attention_kv16_g4.inc"
};
const uint32_t kSpvAttentionVec[] = {
#include "vulkan/attention_vec.inc"
};
const uint32_t kSpvAttentionVecK16[] = {
#include "vulkan/attention_vec_k16.inc"
};
const uint32_t kSpvAttentionVecV16[] = {
#include "vulkan/attention_vec_v16.inc"
};
const uint32_t kSpvAttentionVecKV16[] = {
#include "vulkan/attention_vec_kv16.inc"
};
const uint32_t kSpvAttentionVecG4[] = {
#include "vulkan/attention_vec_g4.inc"
};
const uint32_t kSpvAttentionVecK16G4[] = {
#include "vulkan/attention_vec_k16_g4.inc"
};
const uint32_t kSpvAttentionVecV16G4[] = {
#include "vulkan/attention_vec_v16_g4.inc"
};
const uint32_t kSpvAttentionVecKV16G4[] = {
#include "vulkan/attention_vec_kv16_g4.inc"
};
const uint32_t kSpvAttentionK16[] = {
#include "vulkan/attention_k16.inc"
};
const uint32_t kSpvAttentionV16[] = {
#include "vulkan/attention_v16.inc"
};
const uint32_t kSpvAttentionKV16[] = {
#include "vulkan/attention_kv16.inc"
};
const uint32_t kSpvAttentionTileK16[] = {
#include "vulkan/attention_tile_k16.inc"
};
const uint32_t kSpvAttentionTileV16[] = {
#include "vulkan/attention_tile_v16.inc"
};
const uint32_t kSpvAttentionTileKV16[] = {
#include "vulkan/attention_tile_kv16.inc"
};
const uint32_t kSpvNormRopeKvK16[] = {
#include "vulkan/norm_rope_kv_k16.inc"
};
const uint32_t kSpvNormRopeKvV16[] = {
#include "vulkan/norm_rope_kv_v16.inc"
};
const uint32_t kSpvNormRopeKvKV16[] = {
#include "vulkan/norm_rope_kv_kv16.inc"
};

const uint32_t kSpvMatmulRowQ4Dot[] = {
#include "vulkan/matmul_row_q4_dot.inc"
};
const uint32_t kSpvMatmulRowK4Dot[] = {
#include "vulkan/matmul_row_k4_dot.inc"
};
const uint32_t kSpvMatmulRowK5Dot[] = {
#include "vulkan/matmul_row_k5_dot.inc"
};
const uint32_t kSpvMatmulRowKDot[] = {
#include "vulkan/matmul_row_k_dot.inc"
};

const uint32_t kSpvMatmulTile[] = {
#include "vulkan/matmul_tile.inc"
};
const uint32_t kSpvQuantizeXW[] = {
#include "vulkan/quantize_xw.inc"
};
const uint32_t kSpvMatmulTileQ[] = {
#include "vulkan/matmul_tile_q.inc"
};
const uint32_t kSpvMatmulTileQ6[] = {
#include "vulkan/matmul_tile_q6.inc"
};
const uint32_t kSpvMatmulTileQ8[] = {
#include "vulkan/matmul_tile_q8.inc"
};
const uint32_t kSpvMatmulReduce[] = {
#include "vulkan/matmul_reduce.inc"
};
const uint32_t kSpvMatmulVecQ8[] = {
#include "vulkan/matmul_vec_q8.inc"
};
const uint32_t kSpvMatmulTileBf16[] = {
#include "vulkan/matmul_tile_bf16.inc"
};
const uint32_t kSpvMatmulTileBf16Preserve[] = {
#include "vulkan/matmul_tile_bf16_preserve.inc"
};
const uint32_t kSpvMatmulRowFloatX[] = {
#include "vulkan/matmul_row_float_x.inc"
};
const uint32_t kSpvMatmulRowFloatXPreserve[] = {
#include "vulkan/matmul_row_float_x_preserve.inc"
};
const uint32_t kSpvMatmulRowPreserve[] = {
#include "vulkan/matmul_row_preserve.inc"
};
const uint32_t kSpvMatmulRowQ8wPreserve[] = {
#include "vulkan/matmul_row_q8w_preserve.inc"
};
const uint32_t kSpvMatmulVecQ8Preserve[] = {
#include "vulkan/matmul_vec_q8_preserve.inc"
};
const uint32_t kSpvMatmulRowQ4Preserve[] = {
#include "vulkan/matmul_row_q4_preserve.inc"
};
const uint32_t kSpvMatmulRowQ4DotPreserve[] = {
#include "vulkan/matmul_row_q4_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowK4DotPreserve[] = {
#include "vulkan/matmul_row_k4_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowK5Preserve[] = {
#include "vulkan/matmul_row_k5_preserve.inc"
};
const uint32_t kSpvMatmulRowK5DotPreserve[] = {
#include "vulkan/matmul_row_k5_dot_preserve.inc"
};
const uint32_t kSpvMatmulRowKPreserve[] = {
#include "vulkan/matmul_row_k_preserve.inc"
};
const uint32_t kSpvMatmulRowKDotPreserve[] = {
#include "vulkan/matmul_row_k_dot_preserve.inc"
};
const uint32_t kSpvMatmulTilePreserve[] = {
#include "vulkan/matmul_tile_preserve.inc"
};
const uint32_t kSpvMatmulTileQPreserve[] = {
#include "vulkan/matmul_tile_q_preserve.inc"
};
const uint32_t kSpvMatmulTileQ6Preserve[] = {
#include "vulkan/matmul_tile_q6_preserve.inc"
};
const uint32_t kSpvMatmulReducePreserve[] = {
#include "vulkan/matmul_reduce_preserve.inc"
};
const uint32_t kSpvMoeRoute[] = {
#include "vulkan/moe_route.inc"
};
const uint32_t kSpvMoeCombine[] = {
#include "vulkan/moe_combine.inc"
};
const uint32_t kSpvMoeGroup[] = {
#include "vulkan/moe_group.inc"
};
// The attention kernels at head width 256, qwen35's, in the same cache-type order.
const uint32_t kSpvAttentionTileD256[] = {
#include "vulkan/attention_tile_d256.inc"
};
const uint32_t kSpvAttentionTileD256K16[] = {
#include "vulkan/attention_tile_d256_k16.inc"
};
const uint32_t kSpvAttentionTileD256V16[] = {
#include "vulkan/attention_tile_d256_v16.inc"
};
const uint32_t kSpvAttentionTileD256KV16[] = {
#include "vulkan/attention_tile_d256_kv16.inc"
};
const uint32_t kSpvAttentionVecD256[] = {
#include "vulkan/attention_vec_d256.inc"
};
const uint32_t kSpvAttentionVecD256K16[] = {
#include "vulkan/attention_vec_d256_k16.inc"
};
const uint32_t kSpvAttentionVecD256V16[] = {
#include "vulkan/attention_vec_d256_v16.inc"
};
const uint32_t kSpvAttentionVecD256KV16[] = {
#include "vulkan/attention_vec_d256_kv16.inc"
};
const uint32_t kSpvAttentionVecD256G4[] = {
#include "vulkan/attention_vec_d256_g4.inc"
};
const uint32_t kSpvAttentionVecD256K16G4[] = {
#include "vulkan/attention_vec_d256_k16_g4.inc"
};
const uint32_t kSpvAttentionVecD256V16G4[] = {
#include "vulkan/attention_vec_d256_v16_g4.inc"
};
const uint32_t kSpvAttentionVecD256KV16G4[] = {
#include "vulkan/attention_vec_d256_kv16_g4.inc"
};
// The qwen35 layers' ops (docs/QWEN35.md).
const uint32_t kSpvSigmoidMul[] = {
#include "vulkan/sigmoid_mul.inc"
};
const uint32_t kSpvGatedRmsNorm[] = {
#include "vulkan/gated_rms_norm.inc"
};
const uint32_t kSpvCausalConvSilu[] = {
#include "vulkan/causal_conv_silu.inc"
};
const uint32_t kSpvDeltaRule[] = {
#include "vulkan/delta_rule.inc"
};
// The embedded drafter's ops (docs/SPECULATIVE.md, section 7).
const uint32_t kSpvArgmaxRows[] = {
#include "vulkan/argmax_rows.inc"
};
const uint32_t kSpvEmbedIds[] = {
#include "vulkan/embed_ids.inc"
};
const uint32_t kSpvEmbedIdsMxfp4[] = {
#include "vulkan/embed_ids_mxfp4.inc"
};
const uint32_t kSpvDeltaRuleShort[] = {
#include "vulkan/delta_rule_short.inc"
};

const uint32_t kSpvEmbedMxfp4[] = {
#include "vulkan/embed_mxfp4.inc"
};
const uint32_t kSpvMatmulTileMxfp4[] = {
#include "vulkan/matmul_tile_mxfp4.inc"
};
const uint32_t kSpvMatmulTileMxfp4Bf16[] = {
#include "vulkan/matmul_tile_mxfp4_bf16.inc"
};
const uint32_t kSpvMatmulReduceMxfp4[] = {
#include "vulkan/matmul_reduce_mxfp4.inc"
};

const uint32_t kSpvCopyMxfp4[] = {
#include "vulkan/copy_mxfp4.inc"
};
const uint32_t kSpvMatmulRowMxfp4[] = {
#include "vulkan/matmul_row_mxfp4.inc"
};
const uint32_t kSpvMatmulRowMxfp4Dot[] = {
#include "vulkan/matmul_row_mxfp4_dot.inc"
};
const uint32_t kSpvMatmulRowMxfp4FloatX[] = {
#include "vulkan/matmul_row_mxfp4_float_x.inc"
};
const uint32_t kSpvMatmulTileQ8mx[] = {
#include "vulkan/matmul_tile_q8mx.inc"
};

const uint32_t kSpvEmbedWf16[] = {
#include "vulkan/embed_wf16.inc"
};
const uint32_t kSpvEmbedIdsWf16[] = {
#include "vulkan/embed_ids_wf16.inc"
};
const uint32_t kSpvMatmulRowWf16[] = {
#include "vulkan/matmul_row_wf16.inc"
};
const uint32_t kSpvMatmulTileWf16[] = {
#include "vulkan/matmul_tile_wf16.inc"
};
const uint32_t kSpvMatmulTileWf16Preserve[] = {
#include "vulkan/matmul_tile_wf16_preserve.inc"
};
const uint32_t kSpvMatmulTileWf16Bf16[] = {
#include "vulkan/matmul_tile_wf16_bf16.inc"
};
const uint32_t kSpvMatmulTileWf16Bf16Preserve[] = {
#include "vulkan/matmul_tile_wf16_bf16_preserve.inc"
};
const uint32_t kSpvEmbedWbf16[] = {
#include "vulkan/embed_wbf16.inc"
};
const uint32_t kSpvEmbedIdsWbf16[] = {
#include "vulkan/embed_ids_wbf16.inc"
};
const uint32_t kSpvMatmulRowWbf16[] = {
#include "vulkan/matmul_row_wbf16.inc"
};
const uint32_t kSpvMatmulTileWbf16[] = {
#include "vulkan/matmul_tile_wbf16.inc"
};
const uint32_t kSpvMatmulTileWbf16Preserve[] = {
#include "vulkan/matmul_tile_wbf16_preserve.inc"
};
const uint32_t kSpvMatmulTileWbf16Bf16[] = {
#include "vulkan/matmul_tile_wbf16_bf16.inc"
};
const uint32_t kSpvMatmulTileWbf16Bf16Preserve[] = {
#include "vulkan/matmul_tile_wbf16_bf16_preserve.inc"
};

// IDs, diagnostic names and module bindings share this order; cache variants rely on adjacent IDs.
#define LLMX_VULKAN_KERNELS(X) \
    X(K_ADD, "add", kSpvAdd, sizeof(kSpvAdd), 2, nullptr) \
    X(K_SILU_MUL, "silu_mul", kSpvSiluMul, sizeof(kSpvSiluMul), 4, nullptr) \
    X(K_GATHER_ROWS, "gather_rows", kSpvGatherRows, sizeof(kSpvGatherRows), 3, nullptr) \
    X(K_RMS_NORM_ROWS, "rms_norm_rows", kSpvRmsNormRows, sizeof(kSpvRmsNormRows), 4, nullptr) \
    X(K_NORM_ROPE_PARTIAL, "norm_rope_partial", kSpvNormRopePartial, sizeof(kSpvNormRopePartial), 6, nullptr) \
    X(K_EMBED, "embed", kSpvEmbed, sizeof(kSpvEmbed), 4, nullptr) \
    X(K_MATMUL_ROW, "matmul_row", kSpvMatmulRow, sizeof(kSpvMatmulRow), 12, kMatmulRowCounts, kSpvMatmulRowPreserve, sizeof(kSpvMatmulRowPreserve)) \
    X(K_KV_WRITE, "kv_write", kSpvKvWrite, sizeof(kSpvKvWrite), 5, nullptr) \
    X(K_ATTENTION, "attention", kSpvAttention, sizeof(kSpvAttention), 7, nullptr) \
    X(K_ATTENTION_MERGE, "attention_merge", kSpvAttentionMerge, sizeof(kSpvAttentionMerge), 4, nullptr) \
    X(K_MATMUL_TILE, "matmul_tile", kSpvMatmulTile, sizeof(kSpvMatmulTile), 6, nullptr, kSpvMatmulTilePreserve, sizeof(kSpvMatmulTilePreserve)) \
    X(K_MATMUL_ROW_Q4, "matmul_row_q4", kSpvMatmulRowQ4, sizeof(kSpvMatmulRowQ4), 12, kMatmulRowCounts, kSpvMatmulRowQ4Preserve, sizeof(kSpvMatmulRowQ4Preserve)) \
    X(K_MATMUL_ROW_K4, "matmul_row_k4", kSpvMatmulRowK4, sizeof(kSpvMatmulRowK4), 12, kMatmulRowCounts) \
    X(K_MATMUL_ROW_K5, "matmul_row_k5", kSpvMatmulRowK5, sizeof(kSpvMatmulRowK5), 12, kMatmulRowCounts, kSpvMatmulRowK5Preserve, sizeof(kSpvMatmulRowK5Preserve)) \
    X(K_MATMUL_ROW_K, "matmul_row_k", kSpvMatmulRowK, sizeof(kSpvMatmulRowK), 12, kMatmulRowCounts, kSpvMatmulRowKPreserve, sizeof(kSpvMatmulRowKPreserve)) \
    X(K_NORM_ROPE_KV, "norm_rope_kv", kSpvNormRopeKv, sizeof(kSpvNormRopeKv), 11, nullptr) \
    X(K_ATTENTION_TILE, "attention_tile", kSpvAttentionTile, sizeof(kSpvAttentionTile), 6, nullptr) \
    X(K_KV_WRITE_K16, "kv_write_k16", kSpvKvWriteK16, sizeof(kSpvKvWriteK16), 5, nullptr) \
    X(K_KV_WRITE_V16, "kv_write_v16", kSpvKvWriteV16, sizeof(kSpvKvWriteV16), 5, nullptr) \
    X(K_KV_WRITE_KV16, "kv_write_kv16", kSpvKvWriteKV16, sizeof(kSpvKvWriteKV16), 5, nullptr) \
    X(K_ATTENTION_K16, "attention_k16", kSpvAttentionK16, sizeof(kSpvAttentionK16), 7, nullptr) \
    X(K_ATTENTION_V16, "attention_v16", kSpvAttentionV16, sizeof(kSpvAttentionV16), 7, nullptr) \
    X(K_ATTENTION_KV16, "attention_kv16", kSpvAttentionKV16, sizeof(kSpvAttentionKV16), 7, nullptr) \
    X(K_ATTENTION_TILE_K16, "attention_tile_k16", kSpvAttentionTileK16, sizeof(kSpvAttentionTileK16), 6, nullptr) \
    X(K_ATTENTION_TILE_V16, "attention_tile_v16", kSpvAttentionTileV16, sizeof(kSpvAttentionTileV16), 6, nullptr) \
    X(K_ATTENTION_TILE_KV16, "attention_tile_kv16", kSpvAttentionTileKV16, sizeof(kSpvAttentionTileKV16), 6, nullptr) \
    X(K_NORM_ROPE_KV_K16, "norm_rope_kv_k16", kSpvNormRopeKvK16, sizeof(kSpvNormRopeKvK16), 11, nullptr) \
    X(K_NORM_ROPE_KV_V16, "norm_rope_kv_v16", kSpvNormRopeKvV16, sizeof(kSpvNormRopeKvV16), 11, nullptr) \
    X(K_NORM_ROPE_KV_KV16, "norm_rope_kv_kv16", kSpvNormRopeKvKV16, sizeof(kSpvNormRopeKvKV16), 11, nullptr) \
    X(K_QUANTIZE_X, "quantize_x", kSpvQuantizeX, sizeof(kSpvQuantizeX), 2, nullptr) \
    X(K_MATMUL_ROW_Q8W, "matmul_row_q8w", kSpvMatmulRowQ8W, sizeof(kSpvMatmulRowQ8W), 12, kMatmulRowCounts, kSpvMatmulRowQ8wPreserve, sizeof(kSpvMatmulRowQ8wPreserve)) \
    X(K_MATMUL_TILE_TALL, "matmul_tile_tall", kSpvMatmulTile, sizeof(kSpvMatmulTile), 6, nullptr, kSpvMatmulTilePreserve, sizeof(kSpvMatmulTilePreserve)) \
    X(K_MATMUL_ROW_Q4_DOT, "matmul_row_q4_dot", kSpvMatmulRowQ4Dot, sizeof(kSpvMatmulRowQ4Dot), 12, kMatmulRowCounts, kSpvMatmulRowQ4DotPreserve, sizeof(kSpvMatmulRowQ4DotPreserve)) \
    X(K_MATMUL_ROW_K4_DOT, "matmul_row_k4_dot", kSpvMatmulRowK4Dot, sizeof(kSpvMatmulRowK4Dot), 12, kMatmulRowCounts, kSpvMatmulRowK4DotPreserve, sizeof(kSpvMatmulRowK4DotPreserve)) \
    X(K_MATMUL_ROW_K5_DOT, "matmul_row_k5_dot", kSpvMatmulRowK5Dot, sizeof(kSpvMatmulRowK5Dot), 12, kMatmulRowCounts, kSpvMatmulRowK5DotPreserve, sizeof(kSpvMatmulRowK5DotPreserve)) \
    X(K_MATMUL_ROW_K_DOT, "matmul_row_k_dot", kSpvMatmulRowKDot, sizeof(kSpvMatmulRowKDot), 12, kMatmulRowCounts, kSpvMatmulRowKDotPreserve, sizeof(kSpvMatmulRowKDotPreserve)) \
    X(K_QUANTIZE_XW, "quantize_xw", kSpvQuantizeXW, sizeof(kSpvQuantizeXW), 2, nullptr) \
    X(K_MATMUL_TILE_Q, "matmul_tile_q", kSpvMatmulTileQ, sizeof(kSpvMatmulTileQ), 5, kMatmulTileQCounts, kSpvMatmulTileQPreserve, sizeof(kSpvMatmulTileQPreserve)) \
    X(K_MATMUL_TILE_Q_TALL, "matmul_tile_q_tall", kSpvMatmulTileQ, sizeof(kSpvMatmulTileQ), 5, kMatmulTileQCounts, kSpvMatmulTileQPreserve, sizeof(kSpvMatmulTileQPreserve)) \
    X(K_MATMUL_TILE_Q6, "matmul_tile_q6", kSpvMatmulTileQ6, sizeof(kSpvMatmulTileQ6), 5, kMatmulTileQCounts, kSpvMatmulTileQ6Preserve, sizeof(kSpvMatmulTileQ6Preserve)) \
    X(K_MATMUL_TILE_Q8, "matmul_tile_q8", kSpvMatmulTileQ8, sizeof(kSpvMatmulTileQ8), 5, kMatmulTileQCounts) \
    X(K_MATMUL_TILE_Q8_TALL, "matmul_tile_q8_tall", kSpvMatmulTileQ8, sizeof(kSpvMatmulTileQ8), 5, kMatmulTileQCounts) \
    X(K_MATMUL_REDUCE, "matmul_reduce", kSpvMatmulReduce, sizeof(kSpvMatmulReduce), 2, kMatmulReduceCounts, kSpvMatmulReducePreserve, sizeof(kSpvMatmulReducePreserve)) \
    X(K_MATMUL_VEC_Q8, "matmul_vec_q8", kSpvMatmulVecQ8, sizeof(kSpvMatmulVecQ8), 12, kMatmulRowCounts, kSpvMatmulVecQ8Preserve, sizeof(kSpvMatmulVecQ8Preserve)) \
    X(K_MOE_ROUTE, "moe_route", kSpvMoeRoute, sizeof(kSpvMoeRoute), 3, nullptr) \
    X(K_MOE_COMBINE, "moe_combine", kSpvMoeCombine, sizeof(kSpvMoeCombine), 3, nullptr) \
    X(K_MOE_GROUP, "moe_group", kSpvMoeGroup, sizeof(kSpvMoeGroup), 2, nullptr) \
    X(K_ATTENTION_G4, "attention_g4", kSpvAttentionG4, sizeof(kSpvAttentionG4), 7, nullptr) \
    X(K_ATTENTION_K16_G4, "attention_k16_g4", kSpvAttentionK16G4, sizeof(kSpvAttentionK16G4), 7, nullptr) \
    X(K_ATTENTION_V16_G4, "attention_v16_g4", kSpvAttentionV16G4, sizeof(kSpvAttentionV16G4), 7, nullptr) \
    X(K_ATTENTION_KV16_G4, "attention_kv16_g4", kSpvAttentionKV16G4, sizeof(kSpvAttentionKV16G4), 7, nullptr) \
    X(K_ATTENTION_VEC, "attention_vec", kSpvAttentionVec, sizeof(kSpvAttentionVec), 7, nullptr) \
    X(K_ATTENTION_VEC_K16, "attention_vec_k16", kSpvAttentionVecK16, sizeof(kSpvAttentionVecK16), 7, nullptr) \
    X(K_ATTENTION_VEC_V16, "attention_vec_v16", kSpvAttentionVecV16, sizeof(kSpvAttentionVecV16), 7, nullptr) \
    X(K_ATTENTION_VEC_KV16, "attention_vec_kv16", kSpvAttentionVecKV16, sizeof(kSpvAttentionVecKV16), 7, nullptr) \
    X(K_ATTENTION_VEC_G4, "attention_vec_g4", kSpvAttentionVecG4, sizeof(kSpvAttentionVecG4), 7, nullptr) \
    X(K_ATTENTION_VEC_K16_G4, "attention_vec_k16_g4", kSpvAttentionVecK16G4, sizeof(kSpvAttentionVecK16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_V16_G4, "attention_vec_v16_g4", kSpvAttentionVecV16G4, sizeof(kSpvAttentionVecV16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_KV16_G4, "attention_vec_kv16_g4", kSpvAttentionVecKV16G4, sizeof(kSpvAttentionVecKV16G4), 7, nullptr) \
    X(K_MATMUL_ROW_F32, "matmul_row_f32", kSpvMatmulRow, sizeof(kSpvMatmulRow), 12, kMatmulRowCounts) \
    X(K_SIGMOID_MUL, "sigmoid_mul", kSpvSigmoidMul, sizeof(kSpvSigmoidMul), 4, nullptr) \
    X(K_GATED_RMS_NORM, "gated_rms_norm", kSpvGatedRmsNorm, sizeof(kSpvGatedRmsNorm), 5, nullptr) \
    X(K_CAUSAL_CONV_SILU, "causal_conv_silu", kSpvCausalConvSilu, sizeof(kSpvCausalConvSilu), 5, nullptr) \
    X(K_DELTA_RULE, "delta_rule", kSpvDeltaRule, sizeof(kSpvDeltaRule), 8, nullptr) \
    X(K_ATTENTION_TILE_D256, "attention_tile_d256", kSpvAttentionTileD256, sizeof(kSpvAttentionTileD256), 6, nullptr) \
    X(K_ATTENTION_TILE_D256_K16, "attention_tile_d256_k16", kSpvAttentionTileD256K16, sizeof(kSpvAttentionTileD256K16), 6, nullptr) \
    X(K_ATTENTION_TILE_D256_V16, "attention_tile_d256_v16", kSpvAttentionTileD256V16, sizeof(kSpvAttentionTileD256V16), 6, nullptr) \
    X(K_ATTENTION_TILE_D256_KV16, "attention_tile_d256_kv16", kSpvAttentionTileD256KV16, sizeof(kSpvAttentionTileD256KV16), 6, nullptr) \
    X(K_ATTENTION_VEC_D256, "attention_vec_d256", kSpvAttentionVecD256, sizeof(kSpvAttentionVecD256), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_K16, "attention_vec_d256_k16", kSpvAttentionVecD256K16, sizeof(kSpvAttentionVecD256K16), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_V16, "attention_vec_d256_v16", kSpvAttentionVecD256V16, sizeof(kSpvAttentionVecD256V16), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_KV16, "attention_vec_d256_kv16", kSpvAttentionVecD256KV16, sizeof(kSpvAttentionVecD256KV16), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_G4, "attention_vec_d256_g4", kSpvAttentionVecD256G4, sizeof(kSpvAttentionVecD256G4), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_K16_G4, "attention_vec_d256_k16_g4", kSpvAttentionVecD256K16G4, sizeof(kSpvAttentionVecD256K16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_V16_G4, "attention_vec_d256_v16_g4", kSpvAttentionVecD256V16G4, sizeof(kSpvAttentionVecD256V16G4), 7, nullptr) \
    X(K_ATTENTION_VEC_D256_KV16_G4, "attention_vec_d256_kv16_g4", kSpvAttentionVecD256KV16G4, sizeof(kSpvAttentionVecD256KV16G4), 7, nullptr) \
    X(K_MATMUL_ROW_FLOAT_X, "matmul_row_float_x", kSpvMatmulRowFloatX, sizeof(kSpvMatmulRowFloatX), 12, kMatmulRowCounts, kSpvMatmulRowFloatXPreserve, sizeof(kSpvMatmulRowFloatXPreserve)) \
    X(K_MATMUL_TILE_BF16, "matmul_tile_bf16", kSpvMatmulTileBf16, sizeof(kSpvMatmulTileBf16), 6, nullptr, kSpvMatmulTileBf16Preserve, sizeof(kSpvMatmulTileBf16Preserve)) \
    X(K_MATMUL_TILE_BF16_TALL, "matmul_tile_bf16_tall", kSpvMatmulTileBf16, sizeof(kSpvMatmulTileBf16), 6, nullptr, kSpvMatmulTileBf16Preserve, sizeof(kSpvMatmulTileBf16Preserve)) \
    X(K_EMBED_MXFP4, "embed_mxfp4", kSpvEmbedMxfp4, sizeof(kSpvEmbedMxfp4), 4, nullptr) \
    X(K_MATMUL_TILE_MXFP4, "matmul_tile_mxfp4", kSpvMatmulTileMxfp4, sizeof(kSpvMatmulTileMxfp4), 6, nullptr) \
    X(K_MATMUL_TILE_MXFP4_TALL, "matmul_tile_mxfp4_tall", kSpvMatmulTileMxfp4, sizeof(kSpvMatmulTileMxfp4), 6, nullptr) \
    X(K_MATMUL_TILE_MXFP4_BF16, "matmul_tile_mxfp4_bf16", kSpvMatmulTileMxfp4Bf16, sizeof(kSpvMatmulTileMxfp4Bf16), 6, nullptr) \
    X(K_MATMUL_TILE_MXFP4_BF16_TALL, "matmul_tile_mxfp4_bf16_tall", kSpvMatmulTileMxfp4Bf16, sizeof(kSpvMatmulTileMxfp4Bf16), 6, nullptr) \
    X(K_MATMUL_REDUCE_MXFP4, "matmul_reduce_mxfp4", kSpvMatmulReduceMxfp4, sizeof(kSpvMatmulReduceMxfp4), 2, kMatmulReduceCounts) \
    X(K_COPY_MXFP4, "copy_mxfp4", kSpvCopyMxfp4, sizeof(kSpvCopyMxfp4), 2, nullptr) \
    X(K_MATMUL_ROW_MXFP4, "matmul_row_mxfp4", kSpvMatmulRowMxfp4, sizeof(kSpvMatmulRowMxfp4), 12, kMatmulRowCounts) \
    X(K_MATMUL_ROW_MXFP4_DOT, "matmul_row_mxfp4_dot", kSpvMatmulRowMxfp4Dot, sizeof(kSpvMatmulRowMxfp4Dot), 12, kMatmulRowCounts) \
    X(K_MATMUL_ROW_MXFP4_FLOAT_X, "matmul_row_mxfp4_float_x", kSpvMatmulRowMxfp4FloatX, sizeof(kSpvMatmulRowMxfp4FloatX), 12, kMatmulRowCounts) \
    X(K_MATMUL_TILE_Q8MX, "matmul_tile_q8mx", kSpvMatmulTileQ8mx, sizeof(kSpvMatmulTileQ8mx), 6, kMatmulTileQ8mxCounts) \
    X(K_MATMUL_TILE_Q8MX_TALL, "matmul_tile_q8mx_tall", kSpvMatmulTileQ8mx, sizeof(kSpvMatmulTileQ8mx), 6, kMatmulTileQ8mxCounts) \
    X(K_ARGMAX_ROWS, "argmax_rows", kSpvArgmaxRows, sizeof(kSpvArgmaxRows), 3, nullptr) \
    X(K_EMBED_IDS, "embed_ids", kSpvEmbedIds, sizeof(kSpvEmbedIds), 4, nullptr) \
    X(K_DELTA_RULE_SHORT, "delta_rule_short", kSpvDeltaRuleShort, sizeof(kSpvDeltaRuleShort), 8, nullptr) \
    X(K_EMBED_IDS_MXFP4, "embed_ids_mxfp4", kSpvEmbedIdsMxfp4, sizeof(kSpvEmbedIdsMxfp4), 4, nullptr) \
    X(K_EMBED_WF16, "embed_wf16", kSpvEmbedWf16, sizeof(kSpvEmbedWf16), 4, nullptr) \
    X(K_MATMUL_ROW_WF16, "matmul_row_wf16", kSpvMatmulRowWf16, sizeof(kSpvMatmulRowWf16), 12, kMatmulRowCounts) \
    X(K_MATMUL_TILE_WF16, "matmul_tile_wf16", kSpvMatmulTileWf16, sizeof(kSpvMatmulTileWf16), 6, nullptr, kSpvMatmulTileWf16Preserve, sizeof(kSpvMatmulTileWf16Preserve)) \
    X(K_MATMUL_TILE_WF16_TALL, "matmul_tile_wf16_tall", kSpvMatmulTileWf16, sizeof(kSpvMatmulTileWf16), 6, nullptr, kSpvMatmulTileWf16Preserve, sizeof(kSpvMatmulTileWf16Preserve)) \
    X(K_MATMUL_TILE_WF16_BF16, "matmul_tile_wf16_bf16", kSpvMatmulTileWf16Bf16, sizeof(kSpvMatmulTileWf16Bf16), 6, nullptr, kSpvMatmulTileWf16Bf16Preserve, sizeof(kSpvMatmulTileWf16Bf16Preserve)) \
    X(K_MATMUL_TILE_WF16_BF16_TALL, "matmul_tile_wf16_bf16_tall", kSpvMatmulTileWf16Bf16, sizeof(kSpvMatmulTileWf16Bf16), 6, nullptr, kSpvMatmulTileWf16Bf16Preserve, sizeof(kSpvMatmulTileWf16Bf16Preserve)) \
    X(K_EMBED_WBF16, "embed_wbf16", kSpvEmbedWbf16, sizeof(kSpvEmbedWbf16), 4, nullptr) \
    X(K_MATMUL_ROW_WBF16, "matmul_row_wbf16", kSpvMatmulRowWbf16, sizeof(kSpvMatmulRowWbf16), 12, kMatmulRowCounts) \
    X(K_MATMUL_TILE_WBF16, "matmul_tile_wbf16", kSpvMatmulTileWbf16, sizeof(kSpvMatmulTileWbf16), 6, nullptr, kSpvMatmulTileWbf16Preserve, sizeof(kSpvMatmulTileWbf16Preserve)) \
    X(K_MATMUL_TILE_WBF16_TALL, "matmul_tile_wbf16_tall", kSpvMatmulTileWbf16, sizeof(kSpvMatmulTileWbf16), 6, nullptr, kSpvMatmulTileWbf16Preserve, sizeof(kSpvMatmulTileWbf16Preserve)) \
    X(K_MATMUL_TILE_WBF16_BF16, "matmul_tile_wbf16_bf16", kSpvMatmulTileWbf16Bf16, sizeof(kSpvMatmulTileWbf16Bf16), 6, nullptr, kSpvMatmulTileWbf16Bf16Preserve, sizeof(kSpvMatmulTileWbf16Bf16Preserve)) \
    X(K_MATMUL_TILE_WBF16_BF16_TALL, "matmul_tile_wbf16_bf16_tall", kSpvMatmulTileWbf16Bf16, sizeof(kSpvMatmulTileWbf16Bf16), 6, nullptr, kSpvMatmulTileWbf16Bf16Preserve, sizeof(kSpvMatmulTileWbf16Bf16Preserve)) \
    X(K_EMBED_IDS_WF16, "embed_ids_wf16", kSpvEmbedIdsWf16, sizeof(kSpvEmbedIdsWf16), 4, nullptr) \
    X(K_EMBED_IDS_WBF16, "embed_ids_wbf16", kSpvEmbedIdsWbf16, sizeof(kSpvEmbedIdsWbf16), 4, nullptr)

#define LLMX_KERNEL_ID(id, name, ...) id,
enum KernelId { LLMX_VULKAN_KERNELS(LLMX_KERNEL_ID) K_COUNT };
#undef LLMX_KERNEL_ID

// A supported weight format's dispatch choices, independent of the requested activation dtype.
enum class RowLayout { values, blocks, q8_pairs, q4_pairs, k_blocks };
struct WeightKernels {
    uint32_t type;
    RowLayout layout;
    KernelId embed, embed_ids, row, dot_row, float_row;
    KernelId tile, tall_tile, bf16_tile, bf16_tall_tile;
    KernelId integer_tile, integer_tall_tile;
    bool fast_tile, mxfp4;
    size_t DeviceProfile::*moe_from;
};

const WeightKernels* weight_kernels(uint32_t type) {
    static const WeightKernels formats[] = {
        {quant::GGML_TYPE_F32, RowLayout::values, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_F32, K_MATMUL_ROW_F32, K_MATMUL_ROW_F32,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_COUNT, K_COUNT, true, false, &DeviceProfile::moe_tile_from},
        {quant::GGML_TYPE_F16, RowLayout::values, K_EMBED_WF16, K_EMBED_IDS_WF16, K_MATMUL_ROW_WF16, K_MATMUL_ROW_WF16, K_MATMUL_ROW_WF16,
         K_MATMUL_TILE_WF16, K_MATMUL_TILE_WF16_TALL, K_MATMUL_TILE_WF16_BF16, K_MATMUL_TILE_WF16_BF16_TALL,
         K_COUNT, K_COUNT, true, false, &DeviceProfile::moe_tile_from},
        {quant::GGML_TYPE_BF16, RowLayout::values, K_EMBED_WBF16, K_EMBED_IDS_WBF16, K_MATMUL_ROW_WBF16, K_MATMUL_ROW_WBF16, K_MATMUL_ROW_WBF16,
         K_MATMUL_TILE_WBF16, K_MATMUL_TILE_WBF16_TALL, K_MATMUL_TILE_WBF16_BF16, K_MATMUL_TILE_WBF16_BF16_TALL,
         K_COUNT, K_COUNT, true, false, &DeviceProfile::moe_tile_from},
        {quant::GGML_TYPE_Q8_0, RowLayout::q8_pairs, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW, K_MATMUL_ROW, K_MATMUL_ROW_FLOAT_X,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_MATMUL_TILE_Q8, K_MATMUL_TILE_Q8_TALL, true, false, &DeviceProfile::moe_tile_from},
        {quant::GGML_TYPE_Q4_0, RowLayout::q4_pairs, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_Q4, K_MATMUL_ROW_Q4_DOT, K_MATMUL_ROW_FLOAT_X,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, false, false, &DeviceProfile::moe_tile_from_q4},
        {quant::GGML_TYPE_Q4_1, RowLayout::blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_Q4, K_MATMUL_ROW_Q4_DOT, K_MATMUL_ROW_FLOAT_X,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, false, false, &DeviceProfile::moe_tile_from_q4},
        {quant::GGML_TYPE_Q4_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_K4, K_MATMUL_ROW_K4_DOT, K_MATMUL_ROW_FLOAT_X,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, false, false, &DeviceProfile::moe_tile_from_q4k},
        {quant::GGML_TYPE_Q5_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_K5, K_MATMUL_ROW_K5_DOT, K_MATMUL_ROW_FLOAT_X,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, false, false, &DeviceProfile::moe_tile_from_q5k},
        {quant::GGML_TYPE_Q6_K, RowLayout::k_blocks, K_EMBED, K_EMBED_IDS, K_MATMUL_ROW_K, K_MATMUL_ROW_K_DOT, K_MATMUL_ROW_FLOAT_X,
         K_MATMUL_TILE, K_MATMUL_TILE_TALL, K_MATMUL_TILE_BF16, K_MATMUL_TILE_BF16_TALL,
         K_MATMUL_TILE_Q6, K_MATMUL_TILE_Q6, false, false, &DeviceProfile::moe_tile_from},
        {quant::GGML_TYPE_MXFP4, RowLayout::blocks, K_EMBED_MXFP4, K_EMBED_IDS_MXFP4, K_MATMUL_ROW_MXFP4, K_MATMUL_ROW_MXFP4_DOT, K_MATMUL_ROW_MXFP4_FLOAT_X,
         K_MATMUL_TILE_MXFP4, K_MATMUL_TILE_MXFP4_TALL, K_MATMUL_TILE_MXFP4_BF16, K_MATMUL_TILE_MXFP4_BF16_TALL,
         K_MATMUL_TILE_Q8MX, K_MATMUL_TILE_Q8MX_TALL, false, true, &DeviceProfile::moe_tile_from},
    };
    for (const WeightKernels& format : formats) if (format.type == type) return &format;
    return nullptr;
}

// Dense and routed BF16 calls share the float tile's input-rounding variant.
inline KernelId float_tile_kernel(bool tall, Dtype dtype, uint32_t type) {
    const WeightKernels& format = *weight_kernels(type);
    if (dtype == Dtype::bf16) return tall ? format.bf16_tall_tile : format.bf16_tile;
    return tall ? format.tall_tile : format.tile;
}

// Row kernels share dispatch and take the column count as specialization constant 0; the policy selects floats or an activation twin.
inline bool is_row_kernel(KernelId id) {
    if (id == K_MATMUL_ROW_MXFP4 || id == K_MATMUL_ROW_MXFP4_DOT || id == K_MATMUL_ROW_MXFP4_FLOAT_X) return true;
    switch (id) {
    case K_MATMUL_ROW_WF16: case K_MATMUL_ROW_WBF16:
    case K_MATMUL_ROW: case K_MATMUL_ROW_FLOAT_X: case K_MATMUL_ROW_F32: case K_MATMUL_ROW_Q8W: case K_MATMUL_ROW_Q4:
    case K_MATMUL_ROW_K4: case K_MATMUL_ROW_K5: case K_MATMUL_ROW_K:
    case K_MATMUL_ROW_Q4_DOT:
    case K_MATMUL_ROW_K4_DOT: case K_MATMUL_ROW_K5_DOT: case K_MATMUL_ROW_K_DOT: case K_MATMUL_VEC_Q8:
        return true;
    default: return false;
    }
}

// Whether a kernel id is a tile kernel: they take their row count as specialization constant 0 and write only their outputs.
inline bool is_tile_kernel(KernelId id) {
    if (id == K_MATMUL_TILE_Q8MX || id == K_MATMUL_TILE_Q8MX_TALL) return true;
    if (id == K_MATMUL_TILE_MXFP4 || id == K_MATMUL_TILE_MXFP4_TALL ||
        id == K_MATMUL_TILE_MXFP4_BF16 || id == K_MATMUL_TILE_MXFP4_BF16_TALL) return true;
    switch (id) {
    case K_MATMUL_TILE_WF16: case K_MATMUL_TILE_WF16_TALL: case K_MATMUL_TILE_WF16_BF16: case K_MATMUL_TILE_WF16_BF16_TALL:
    case K_MATMUL_TILE_WBF16: case K_MATMUL_TILE_WBF16_TALL: case K_MATMUL_TILE_WBF16_BF16: case K_MATMUL_TILE_WBF16_BF16_TALL:
    case K_MATMUL_TILE: case K_MATMUL_TILE_TALL: case K_MATMUL_TILE_Q: case K_MATMUL_TILE_Q_TALL:
    case K_MATMUL_TILE_Q6: case K_MATMUL_TILE_Q8: case K_MATMUL_TILE_Q8_TALL:
    case K_MATMUL_TILE_BF16: case K_MATMUL_TILE_BF16_TALL:
        return true;
    default: return false;
    }
}

// Whether a row kernel has a one-column build for one-column chunks, which frees the registers of seven unused accumulators.
// The wide Q8_0 row kernel keeps its wide build (docs/VULKAN.md, Kernel notes).
inline bool row_kernel_builds_one_column(KernelId id) {
    return is_row_kernel(id) && id != K_MATMUL_ROW_Q8W;
}

const uint32_t kRowColsWide = 8, kRowColsOne = 1;
// A kernel's pipelines: the wide build, then the one-column, then for the row kernels the wide build grouped by expert, then the Q8_0 decode kernel's further builds.
const int kVariants = 7;

// The Q8_0 decode kernel's forms (shaders/matmul_vec_q8.comp, specialization constants 11 and 12): the reduction through the subgroup reduction's own pairs for every row and column at once, and the half-block order.
// A build asks for the first and takes it where its device's profile allows it (q8_decode_forms), since it gives the subgroup reduction's bits only where that reduction takes the same pairs.
// The half-block order sums a column's products in another order than the quarter layout, so it is the kernel's order on a device rather than a build's form: where the profile allows it every build takes it, and a column computes the same bits in all of them.
const uint32_t kQ8Tree = 1, kQ8Half = 2;

// The Q8_0 decode kernel's builds by pipeline variant: the columns a subgroup keeps, the rows it takes, the steps whose weights a lane loads before using any, the forms it asks for, and how many column groups take the same rows on adjacent workgroups (specialization constants 0, 9 and 10, then 11 and 13).
// Variants 0 to 2 are every row kernel's wide, one-column and grouped builds, and 3 to 6 this kernel's 2-, 4-, 16- and 32-column builds; a build holds its columns times its groups.
// Every build gives a column the same bits, so the builds differ only in time (docs/VULKAN.md).
struct VecBuild {
    uint32_t cols, rows, steps, forms, span;
    uint32_t capacity() const { return cols * span; }
};
const VecBuild kVecBuilds[kVariants] = {{kRowColsWide, 4, 1, kQ8Tree, 1}, {kRowColsOne, 2, 1, 0, 1}, {kRowColsWide, 4, 1, 0, 1},
                                        {2, 2, 1, kQ8Tree, 1}, {4, 4, 1, kQ8Tree, 1}, {16, 4, 2, kQ8Tree, 1}, {16, 4, 2, kQ8Tree, 2}};
// Its builds for plain columns, narrowest first.
const int kVecByWidth[] = {1, 3, 4, 0, 5, 6};

// The forms a Q8_0 decode build takes on a device: those it asks for that the device's profile allows, and the half-block order in every build where the profile allows it.
inline uint32_t vec_forms(int variant, const DeviceProfile& profile) {
    return (kVecBuilds[variant].forms & profile.q8_decode_forms) | (profile.q8_decode_forms & kQ8Half);
}

// Whether a row kernel is one of the Q4 and K-quant families, which have two-row decode builds (shaders/matmul_row.comp, ROWS).
inline bool row_kernel_builds_two_rows(KernelId id) {
    switch (id) {
    case K_MATMUL_ROW_Q4: case K_MATMUL_ROW_K4: case K_MATMUL_ROW_K5: case K_MATMUL_ROW_K:
    case K_MATMUL_ROW_Q4_DOT: case K_MATMUL_ROW_K4_DOT: case K_MATMUL_ROW_K5_DOT: case K_MATMUL_ROW_K_DOT:

        return true;
    default: return false;
    }
}

// Those families' builds by pipeline variant: the columns a lane keeps and the rows a cluster takes (specialization constants 0 and 9).
// Variants 0 to 2 are every row kernel's wide, one-column and grouped builds of one row; 3 to 6 the two-row builds of 2, 4, 8 and 16 columns, which compute every column they hold, so a chunk fills more than half of one.
// Every build gives a column the same bits (docs/VULKAN.md).
struct RowBuild {
    uint32_t cols, rows;
};
const RowBuild kRowBuilds[kVariants] = {{kRowColsWide, 1}, {kRowColsOne, 1}, {kRowColsWide, 1}, {2, 2}, {4, 2}, {8, 2}, {16, 2}};
// The two-row builds, narrowest first.
const int kRowByWidth[] = {3, 4, 5, 6};

// The rows a cluster takes in a row kernel's build, by pipeline variant: one but in the two-row builds.
inline uint32_t build_rows(KernelId id, int variant) { return row_kernel_builds_two_rows(id) ? kRowBuilds[variant].rows : 1; }

// The columns a row kernel's build holds, by pipeline variant.
inline uint32_t build_cols(KernelId id, int variant) {
    if (id == K_MATMUL_VEC_Q8) return kVecBuilds[variant].capacity();
    if (row_kernel_builds_two_rows(id)) return kRowBuilds[variant].cols;
    return variant == 1 ? kRowColsOne : kRowColsWide;
}

const size_t kMxCopyBytes = size_t(256) << 20;
const size_t kFloatPartBytes = size_t(256) << 20;
const size_t kMxCopyBlock = 36; // one F32 scale and 32 signed codes
const size_t kF32Pad = 32;   // floats after each row of a padded F32 matrix (padded_f32)
// The matrix shapes that get a copy with kF32Pad floats after each row, one rule for padded_f32, which makes the copy, and resident_bytes, which counts it.
// F32 rows a multiple of 256 floats wide would otherwise all read the same memory channel (docs/VULKAN.md).
static bool pads_f32(uint32_t type, size_t nin) { return type == quant::GGML_TYPE_F32 && nin && nin % 256 == 0; }

// What the kernels need of a device beyond Vulkan 1.2's core, the first one missing as the refusal names it, or empty.
inline std::string missing_device_need(const VkPhysicalDeviceSubgroupProperties& sg, const VkPhysicalDeviceFeatures2& f2,
                                       const VkPhysicalDeviceVulkan11Features& f11, const VkPhysicalDeviceVulkan12Features& f12) {
    // The row kernel places one subgroup per row group in a 256-lane workgroup, so the subgroup size must divide it.
    if (!sg.subgroupSize || 256 % sg.subgroupSize) return "has an unsupported subgroup size";
    // A block of 32 activations is quantized across 32 consecutive lanes (shaders/xquant.glsl), and every narrower lane group the kernels fix fits such a subgroup, so the kernel choices do not check the width again.
    if (sg.subgroupSize < 32) return "has subgroups narrower than 32 lanes";
    if (!(sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT)) return "has no subgroup arithmetic";
    if (!(sg.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT)) return "has no subgroup shuffles";
    // Timeline semaphores are what submit and wait are built on.
    if (!f12.timelineSemaphore) return "has no timeline semaphores";
    // The row kernels select a projection's buffers per workgroup, dynamic indexing of a storage buffer array.
    if (!f2.features.shaderStorageBufferArrayDynamicIndexing) return "cannot index storage buffer arrays dynamically";
    // The kernels read quantized blocks as 8-bit values and half scales and activations as 16-bit ones (shaders/q.glsl, shaders/quantize_x.comp).
    if (!f2.features.shaderInt16) return "has no 16-bit integer arithmetic";
    if (!f12.shaderInt8) return "has no 8-bit integer arithmetic";
    if (!f11.storageBuffer16BitAccess) return "has no 16-bit storage buffer access";
    if (!f12.storageBuffer8BitAccess) return "has no 8-bit storage buffer access";
    return {};
}

// Heads 128 and 256 wide take the tiled and vector kernels, whose lane groups fit every subgroup the device check accepts (shaders/attention_tile.comp, shaders/attention_vec.comp).
// Other widths take the per-row kernel, which reads at most four elements of a head per lane of a subgroup (shaders/attention.comp); no head is wider than 256.
inline size_t attention_row_width(uint32_t subgroup_size) { return std::min<size_t>(256, 4 * (size_t)subgroup_size); }
inline bool attention_head_fits(size_t head_dim, uint32_t subgroup_size) { return head_dim == 128 || head_dim == 256 || head_dim <= attention_row_width(subgroup_size); }

// The tile kernel's row count, specialization constant 0: the shorter heights fill a device a taller tile would leave idle, the taller reads less shared memory per product.
const uint32_t kTileRowsSmall = 32, kTileRowsShort = 64, kTileRowsTall = 128;   // the small height is variant 1 of the short kernels

// Layouts the shaders fix, which the host must match.
const uint32_t kQ8LanesPerPair = 4;       // matmul_row.comp, wide Q8_0: `group = l / 4u` ("four lanes per pair")
const uint32_t kKQuantLanes = 8;          // matmul_row.comp, Q4_K, Q5_K and Q6_K: `group = l / 8u` ("eight lanes per block")
const size_t kAttentionTileRows = 32;     // attention_tile.comp: `TQ = 32u`, the query rows of one tile

// A kernel's bindings; `counts` gives each one's array length, one for a plain buffer. A dispatch lists its buffers binding by binding, array elements consecutively.
struct KernelSource {
    const uint32_t* words;
    size_t bytes;
    uint32_t bindings;
    const uint32_t* counts;
    const uint32_t* preserved_words = nullptr;
    size_t preserved_bytes = 0;
};

const uint32_t kMatmulRowCounts[12] = {3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1};
// The integer-dot tile's outputs and weights, three of each, and the reduce's outputs.
const uint32_t kMatmulTileQCounts[5] = {3, 3, 1, 1, 1};
const uint32_t kMatmulTileQ8mxCounts[6] = {3, 3, 1, 1, 1, 3};
const uint32_t kMatmulReduceCounts[2] = {3, 1};

#define LLMX_KERNEL_NAME(id, name, ...) name,
const char* const kKernelNames[K_COUNT] = { LLMX_VULKAN_KERNELS(LLMX_KERNEL_NAME) };
#undef LLMX_KERNEL_NAME

#define LLMX_KERNEL_SOURCE(id, name, ...) {__VA_ARGS__},
const KernelSource kKernels[K_COUNT] = { LLMX_VULKAN_KERNELS(LLMX_KERNEL_SOURCE) };
#undef LLMX_KERNEL_SOURCE
#undef LLMX_VULKAN_KERNELS

// The variant of a cache kernel for a storage's K and V types.
class VulkanKVStorage;
inline KernelId kv_variant(KernelId f32, KernelId k16, const VulkanKVStorage& s);

// 64 tokens per KV block, half the CPU's: the attention workgroup reads a block per iteration, and a smaller block wastes less tail per sequence.
const size_t kVkBlockTokens = 64;

class VulkanBackend;

// KV blocks on the device, kept and grown by BlockKVStorage (backends/kv_storage.hpp).
// A growth's copies run on the queue after it returns, so the old buffers are kept until the command buffer that recorded the copies retires.
class VulkanKVStorage final : public BlockKVStorage {
public:
    VulkanKVStorage(VulkanBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens, KVType kt,
                    KVType vt);

private:
    void retire(const BufferPtr& old) override;
};

// A compiled kernel: module, a layout of `bindings` pushed storage buffers and 128 bytes of push constants, and the pipeline.
struct Kernel {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t bindings = 0;
    uint32_t buffers = 0;   // sum of the bindings' array lengths
};

const char* vk_result_name(VkResult r) {
    switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    default: return "VkResult";
    }
}

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS)
        throw std::runtime_error(std::string("vulkan: ") + what + " failed with " + vk_result_name(r));
}

// The loader library and the entry points that do not need an instance.
struct Loader {
    void* lib = nullptr;
    PFN_vkGetInstanceProcAddr gipa = nullptr;

    Loader() {
#if defined(_WIN32)
        HMODULE h = LoadLibraryA("vulkan-1.dll");
        if (!h) throw VulkanUnavailable("vulkan: vulkan-1.dll is not installed");
        lib = (void*)h;
        gipa = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(h, "vkGetInstanceProcAddr");
#else
        lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw VulkanUnavailable("vulkan: libvulkan.so.1 is not installed");
        gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#endif
        if (!gipa) throw VulkanUnavailable("vulkan: the loader has no vkGetInstanceProcAddr");
    }
    ~Loader() {
        if (!lib) return;
#if defined(_WIN32)
        FreeLibrary((HMODULE)lib);
#else
        dlclose(lib);
#endif
    }
    Loader(const Loader&) = delete;
    Loader& operator=(const Loader&) = delete;
};

// The device and everything a buffer needs to free itself; buffers share it, so the device outlives the last buffer.
struct Device {
    Loader loader;
    Fn fn;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memory{};
    DeviceCaps caps{};            // what this device says of itself
    DeviceProfile profile{};      // what measuring its kernels said (backends/device_profile.hpp)
    bool mxfp4 = false; // exact weight range and wide partials require optional float preservation and double
    bool preserve_float32 = false; // optional matrix modules retain subnormal values
    bool push_descriptor = false;
    bool memory_budget = false;   // the device reports what is free of each heap (VK_EXT_memory_budget)
    // Host memory imported as device memory, which a copy reads in place (VK_EXT_external_memory_host), and the alignment of its address and size.
    PFN_vkGetMemoryHostPointerPropertiesEXT host_pointer_props = nullptr;
    size_t host_import_align = 0;
    // The driver's per-kernel statistics (registers, occupancy), when it reports them.
    bool exec_stats = false;
    PFN_vkGetPipelineExecutablePropertiesKHR get_exec_props = nullptr;
    PFN_vkGetPipelineExecutableStatisticsKHR get_exec_stats = nullptr;
    // The driver's internal representations of a kernel, its ISA on AMD, for a backend opened for diagnostics.
    bool exec_ir = false;
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR get_exec_ir = nullptr;
    // Device time per dispatch, for a backend opened for diagnostics: a timestamp either side of every dispatch, read back after the pass.
    bool timestamps = false;
    double timestamp_ns = 0.0;   // nanoseconds per tick, as the device reports

    // Guarded per handle: construction can fail between creating a handle and loading its destroy function.
    ~Device() {
        if (device && fn.vkDestroyDevice) {
            if (fn.vkDeviceWaitIdle) fn.vkDeviceWaitIdle(device);
            fn.vkDestroyDevice(device, nullptr);
        }
        if (instance && fn.vkDestroyInstance) fn.vkDestroyInstance(instance, nullptr);
    }

    // A memory type with every flag in `required`, preferring `preferred` on top, among those the buffer allows.
    uint32_t memory_type(uint32_t allowed, VkMemoryPropertyFlags required,
                         VkMemoryPropertyFlags preferred, VkMemoryPropertyFlags avoid) const {
        int best = -1;
        int best_score = -1;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if (!(allowed & (1u << i))) continue;
            const VkMemoryPropertyFlags f = memory.memoryTypes[i].propertyFlags;
            if ((f & required) != required) continue;
            int score = 0;
            if ((f & preferred) == preferred) score += 2;
            if (!(f & avoid)) score += 1;
            if (score > best_score) { best_score = score; best = (int)i; }
        }
        if (best < 0) throw std::runtime_error("vulkan: no memory type satisfies the request");
        return (uint32_t)best;
    }
};

class VulkanBuffer final : public Buffer {
public:
    VulkanBuffer(std::shared_ptr<Device> dev, size_t bytes, bool host_visible)
        : dev_(std::move(dev)), size_(bytes) {
        if (!bytes) return;   // an empty allocation has no address and no object
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // Whole words, so a 32-bit view can reach a byte-sized tensor's last bytes (a Q8_0 or Q6_K tensor with an odd block count ends two bytes into a word).
        bi.size = (bytes + 3) & ~size_t(3);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(dev_->fn.vkCreateBuffer(dev_->device, &bi, nullptr, &buffer_), "vkCreateBuffer");
        try {
            VkMemoryRequirements req{};
            dev_->fn.vkGetBufferMemoryRequirements(dev_->device, buffer_, &req);
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            // Device memory prefers not to be host visible, so it comes from the device-local heap rather than the BAR window; host-visible memory prefers to be cached.
            ai.memoryTypeIndex = host_visible
                ? dev_->memory_type(req.memoryTypeBits,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0)
                : dev_->memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            VkDeviceMemory memory = VK_NULL_HANDLE;
            const VkResult r = dev_->fn.vkAllocateMemory(dev_->device, &ai, nullptr, &memory);
            if (r != VK_SUCCESS) {
                if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY)
                    throw std::bad_alloc();
                check(r, "vkAllocateMemory");
            }
            memory_ = memory;
            check(dev_->fn.vkBindBufferMemory(dev_->device, buffer_, memory_, 0), "vkBindBufferMemory");
            if (host_visible) {
                void* mapped = nullptr;
                check(dev_->fn.vkMapMemory(dev_->device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
                mapped_ = mapped;
                std::memset(mapped_, 0, bytes);
            }
        } catch (...) {
            release();
            throw;
        }
    }
    // A copy source over host memory the device reads in place: `memory` imported as device memory of a type among `types`, both it and `bytes` on the import alignment.
    VulkanBuffer(std::shared_ptr<Device> dev, void* memory, size_t bytes, uint32_t types)
        : dev_(std::move(dev)), size_(bytes) {
        VkExternalMemoryBufferCreateInfo ei{};
        ei.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ei.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.pNext = &ei;
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(dev_->fn.vkCreateBuffer(dev_->device, &bi, nullptr, &buffer_), "vkCreateBuffer");
        try {
            VkMemoryRequirements req{};
            dev_->fn.vkGetBufferMemoryRequirements(dev_->device, buffer_, &req);
            if (req.size > bytes) throw std::runtime_error("vulkan: an imported buffer needs more than the memory it wraps");
            VkImportMemoryHostPointerInfoEXT hi{};
            hi.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
            hi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            hi.pHostPointer = memory;
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.pNext = &hi;
            ai.allocationSize = bytes;
            ai.memoryTypeIndex = dev_->memory_type(req.memoryTypeBits & types, 0, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
            check(dev_->fn.vkAllocateMemory(dev_->device, &ai, nullptr, &memory_), "vkAllocateMemory");
            check(dev_->fn.vkBindBufferMemory(dev_->device, buffer_, memory_, 0), "vkBindBufferMemory");
        } catch (...) {
            release();
            throw;
        }
    }
    ~VulkanBuffer() override { release(); }
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    size_t size() const override { return size_; }
    // Weights arrive through adopt; only those get padded copies, which a write or copy into the buffer drops (VulkanBackend::padded_f32).
    bool adopted = false;
    struct Padded {
        size_t rows, nin;
        std::shared_ptr<VulkanBuffer> copy;
    };
    std::map<size_t, Padded> padded;   // by byte offset
    const void* host_ptr() const override { return mapped_; }
    void* mapped() const { return mapped_; }
    VkBuffer handle() const { return buffer_; }
    bool host_visible() const { return mapped_ != nullptr; }

private:
    void release() noexcept {
        if (mapped_) dev_->fn.vkUnmapMemory(dev_->device, memory_);
        if (buffer_) dev_->fn.vkDestroyBuffer(dev_->device, buffer_, nullptr);
        if (memory_) dev_->fn.vkFreeMemory(dev_->device, memory_, nullptr);
    }

    std::shared_ptr<Device> dev_;
    size_t size_ = 0;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
};

const VulkanBuffer& as_vulkan(const Buffer& b) {
    const auto* v = dynamic_cast<const VulkanBuffer*>(&b);
    if (!v) throw std::runtime_error("vulkan: buffer of another backend");
    return *v;
}
VulkanBuffer& as_vulkan(Buffer& b) {
    auto* v = dynamic_cast<VulkanBuffer*>(&b);
    if (!v) throw std::runtime_error("vulkan: buffer of another backend");
    return *v;
}

class VulkanBackend final : public Backend {
    friend struct VulkanLifetimeTest;
    friend struct VulkanQuantizationTest;
public:
    explicit VulkanBackend(int index, bool diagnostics = false) : dev_(std::make_shared<Device>()), timed_(diagnostics) {
        Device& d = *dev_;
        Fn& fn = d.fn;
#define LLMX_VK_LOAD_GLOBAL(name) \
        fn.name = (PFN_##name)d.loader.gipa(VK_NULL_HANDLE, #name); \
        if (!fn.name) throw VulkanUnavailable("vulkan: the loader has no " #name);
        LLMX_VK_GLOBAL_FUNCTIONS(LLMX_VK_LOAD_GLOBAL)
#undef LLMX_VK_LOAD_GLOBAL
        uint32_t version = 0;
        check(fn.vkEnumerateInstanceVersion(&version), "vkEnumerateInstanceVersion");
        if (version < VK_API_VERSION_1_2)
            throw VulkanUnavailable("vulkan: the loader is older than Vulkan 1.2");

        VkApplicationInfo app{};
        app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app.pApplicationName = "llmx";
        app.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ii{};
        ii.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ii.pApplicationInfo = &app;
        // A loader with no driver behind it fails here; that is no device.
        const VkResult ir = fn.vkCreateInstance(&ii, nullptr, &d.instance);
        if (ir == VK_ERROR_INCOMPATIBLE_DRIVER) throw VulkanUnavailable("vulkan: the loader has no driver");
        check(ir, "vkCreateInstance");
#define LLMX_VK_LOAD_INSTANCE(name) \
        fn.name = (PFN_##name)d.loader.gipa(d.instance, #name); \
        if (!fn.name) throw std::runtime_error("vulkan: the instance has no " #name);
        LLMX_VK_INSTANCE_FUNCTIONS(LLMX_VK_LOAD_INSTANCE)
#undef LLMX_VK_LOAD_INSTANCE

        uint32_t count = 0;
        check(fn.vkEnumeratePhysicalDevices(d.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
        if (!count) throw VulkanUnavailable("vulkan: no device");
        if (index < 0 || (uint32_t)index >= count)
            throw VulkanUnavailable("vulkan: no device at index " + std::to_string(index) +
                                    " of " + std::to_string(count));
        std::vector<VkPhysicalDevice> devices(count);
        check(fn.vkEnumeratePhysicalDevices(d.instance, &count, devices.data()), "vkEnumeratePhysicalDevices");
        d.physical = devices[(size_t)index];
        fn.vkGetPhysicalDeviceProperties(d.physical, &d.props);
        fn.vkGetPhysicalDeviceMemoryProperties(d.physical, &d.memory);
        VkPhysicalDeviceSubgroupProperties sg{};
        sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 p2{};
        p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p2.pNext = &sg;
        VkPhysicalDeviceFloatControlsProperties floats{};
        floats.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES;
        sg.pNext = &floats;
        // Compute units, which decide the tile height and the split (matmul_group_impl). Core Vulkan does not report them; where the vendor does, ask, else assume a small device.
        uint32_t core_ext_count = 0;
        fn.vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &core_ext_count, nullptr);
        std::vector<VkExtensionProperties> core_exts(core_ext_count);
        if (core_ext_count)
            fn.vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &core_ext_count, core_exts.data());
        bool has_core_props = false;
        for (const auto& e : core_exts)
            if (std::strcmp(e.extensionName, "VK_AMD_shader_core_properties") == 0) has_core_props = true;
        VkPhysicalDeviceShaderCorePropertiesAMD core{};
        core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
        if (has_core_props) {
            core.pNext = p2.pNext;
            p2.pNext = &core;
        }
        fn.vkGetPhysicalDeviceProperties2(d.physical, &p2);
        d.preserve_float32 = floats.shaderDenormPreserveFloat32 && floats.shaderSignedZeroInfNanPreserveFloat32;
        d.caps.subgroup_size = sg.subgroupSize;
        const bool has_units = has_core_props && core.shaderEngineCount && core.shaderArraysPerEngineCount &&
                               core.computeUnitsPerShaderArray;
        d.caps.compute_units = has_units ? core.shaderEngineCount * core.shaderArraysPerEngineCount *
                                               core.computeUnitsPerShaderArray
                                         : 16;
        VkPhysicalDeviceDriverProperties drv{};
        drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 dp{};
        dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        dp.pNext = &drv;
        fn.vkGetPhysicalDeviceProperties2(d.physical, &dp);
        d.caps.device = d.props.deviceName;
        d.caps.driver = drv.driverName;
        if (d.props.apiVersion < VK_API_VERSION_1_2)
            throw VulkanUnavailable("vulkan: " + d.caps.device + " is older than Vulkan 1.2");

        // A compute family without graphics keeps the queue clear of the desktop; any compute family will do.
        uint32_t families = 0;
        fn.vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &families, nullptr);
        std::vector<VkQueueFamilyProperties> qf(families);
        fn.vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &families, qf.data());
        int chosen = -1;
        for (uint32_t i = 0; i < families; ++i) {
            const VkQueueFlags f = qf[i].queueFlags;
            if (!(f & VK_QUEUE_COMPUTE_BIT)) continue;
            if (chosen < 0 || (!(f & VK_QUEUE_GRAPHICS_BIT) && (qf[(size_t)chosen].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
                chosen = (int)i;
        }
        if (chosen < 0) throw VulkanUnavailable("vulkan: " + d.caps.device + " has no compute queue");
        d.queue_family = (uint32_t)chosen;

        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan11Features f11{};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.pNext = &f12;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f11;
        fn.vkGetPhysicalDeviceFeatures2(d.physical, &f2);
        if (const std::string need = missing_device_need(sg, f2, f11, f12); !need.empty())
            throw VulkanUnavailable("vulkan: " + d.caps.device + " " + need);
        VkPhysicalDeviceVulkan12Features e12{};
        e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        e12.timelineSemaphore = VK_TRUE;
        e12.shaderInt8 = VK_TRUE;
        e12.shaderFloat16 = f12.shaderFloat16;
        e12.storageBuffer8BitAccess = VK_TRUE;
        VkPhysicalDeviceVulkan11Features e11{};
        e11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        e11.pNext = &e12;
        e11.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures2 e2{};
        e2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        e2.pNext = &e11;
        e2.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
        e2.features.shaderInt16 = VK_TRUE;
        d.mxfp4 = f2.features.shaderFloat64 && d.preserve_float32 && floats.shaderSignedZeroInfNanPreserveFloat64;
        e2.features.shaderFloat64 = d.mxfp4 ? VK_TRUE : VK_FALSE;

        uint32_t ext_count = 0;
        check(fn.vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &ext_count, nullptr),
              "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> exts(ext_count);
        check(fn.vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &ext_count, exts.data()),
              "vkEnumerateDeviceExtensionProperties");
        std::vector<const char*> enabled;
        for (const auto& e : exts)
            if (std::strcmp(e.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) {
                enabled.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
                d.push_descriptor = true;
            } else if (std::strcmp(e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) {
                // Core in 1.3, an extension on the 1.2 devices this targets; enabled where present so the dot-form kernels can run.
                enabled.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
                d.caps.integer_dot = true;
            } else if (std::strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0) {
                enabled.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
                d.exec_stats = true;
            } else if (std::strcmp(e.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                enabled.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
                d.memory_budget = true;
            } else if (std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
                VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{};
                hp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
                VkPhysicalDeviceProperties2 p{};
                p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                p.pNext = &hp;
                fn.vkGetPhysicalDeviceProperties2(d.physical, &p);
                if (hp.minImportedHostPointerAlignment) {
                    enabled.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
                    d.host_import_align = (size_t)hp.minImportedHostPointerAlignment;
                }
            }
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR estat{};
        estat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        estat.pipelineExecutableInfo = VK_TRUE;
        if (d.exec_stats) {
            estat.pNext = e2.pNext;
            e2.pNext = &estat;
        }
        // A feature struct of an extension the device lacks must not be chained.
        VkPhysicalDeviceShaderIntegerDotProductFeatures edot{};
        edot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        edot.shaderIntegerDotProduct = VK_TRUE;
        if (d.caps.integer_dot) {
            edot.pNext = e2.pNext;
            e2.pNext = &edot;
        }

        // The profile is chosen here because it depends on the extension scan above.
        d.profile = profile_for(d.caps);
        // A queue that timestamps lets a diagnostics backend attribute a pass's time to kernels.
        d.timestamps = diagnostics && d.props.limits.timestampComputeAndGraphics;
        d.timestamp_ns = d.props.limits.timestampPeriod;

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = d.queue_family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        VkDeviceCreateInfo di{};
        di.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        di.pNext = &e2;
        di.queueCreateInfoCount = 1;
        di.pQueueCreateInfos = &qi;
        di.enabledExtensionCount = (uint32_t)enabled.size();
        di.ppEnabledExtensionNames = enabled.data();
        check(fn.vkCreateDevice(d.physical, &di, nullptr, &d.device), "vkCreateDevice");
#define LLMX_VK_LOAD_DEVICE(name) \
        fn.name = (PFN_##name)fn.vkGetDeviceProcAddr(d.device, #name); \
        if (!fn.name) throw std::runtime_error("vulkan: the device has no " #name);
        LLMX_VK_DEVICE_FUNCTIONS(LLMX_VK_LOAD_DEVICE)
#undef LLMX_VK_LOAD_DEVICE
        if (d.exec_stats) {
            d.get_exec_props = (PFN_vkGetPipelineExecutablePropertiesKHR)fn.vkGetDeviceProcAddr(
                d.device, "vkGetPipelineExecutablePropertiesKHR");
            d.get_exec_stats = (PFN_vkGetPipelineExecutableStatisticsKHR)fn.vkGetDeviceProcAddr(
                d.device, "vkGetPipelineExecutableStatisticsKHR");
            d.exec_stats = d.get_exec_props && d.get_exec_stats;
            d.get_exec_ir = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)fn.vkGetDeviceProcAddr(
                d.device, "vkGetPipelineExecutableInternalRepresentationsKHR");
            d.exec_ir = d.exec_stats && d.get_exec_ir && diagnostics;
        }
        if (d.host_import_align)
            d.host_pointer_props = (PFN_vkGetMemoryHostPointerPropertiesEXT)fn.vkGetDeviceProcAddr(d.device, "vkGetMemoryHostPointerPropertiesEXT");
        if (!d.push_descriptor)
            throw VulkanUnavailable("vulkan: " + d.caps.device + " has no VK_KHR_push_descriptor");
        fn.vkCmdPushDescriptorSetKHR =
            (PFN_vkCmdPushDescriptorSetKHR)fn.vkGetDeviceProcAddr(d.device, "vkCmdPushDescriptorSetKHR");
        if (!fn.vkCmdPushDescriptorSetKHR)
            throw std::runtime_error("vulkan: the device has no vkCmdPushDescriptorSetKHR");
        fn.vkGetDeviceQueue(d.device, d.queue_family, 0, &d.queue);

        VkCommandPoolCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = d.queue_family;
        check(fn.vkCreateCommandPool(d.device, &pi, nullptr, &pool_), "vkCreateCommandPool");
        VkCommandBufferAllocateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ci.commandPool = pool_;
        ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ci.commandBufferCount = kRing;
        check(fn.vkAllocateCommandBuffers(d.device, &ci, ring_), "vkAllocateCommandBuffers");

        VkSemaphoreTypeCreateInfo ti{};
        ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        ti.initialValue = 0;
        VkSemaphoreCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        si.pNext = &ti;
        check(fn.vkCreateSemaphore(d.device, &si, nullptr, &timeline_), "vkCreateSemaphore");
    }

    ~VulkanBackend() override {
        Device& d = *dev_;
        if (hold_.watchdog.joinable()) {
            {
                std::lock_guard<std::mutex> lk(hold_.mu);
                hold_release();
                hold_.quit = true;
            }
            hold_.wake.notify_all();
            hold_.watchdog.join();
        }
        d.fn.vkDeviceWaitIdle(d.device);
        for (auto& p : pending_) p.clear();
        for (auto& a : arena_) a.buffer.reset();
        for (auto& variants : kernels_)
            for (Kernel& k : variants) destroy_kernel(k);
        if (queries_) d.fn.vkDestroyQueryPool(d.device, queries_, nullptr);
        staging_.reset();
        if (timeline_) d.fn.vkDestroySemaphore(d.device, timeline_, nullptr);
        for (VkEvent e : hold_.events) if (e) d.fn.vkDestroyEvent(d.device, e, nullptr);
        if (hold_.timeline) d.fn.vkDestroySemaphore(d.device, hold_.timeline, nullptr);
        if (pool_) d.fn.vkDestroyCommandPool(d.device, pool_, nullptr);
    }

    // A compiled kernel's name, a row kernel's builds other than the wide one marked by their columns or as grouped.
    static std::string kernel_variant_name(int id, int variant) {
        const bool tile = is_tile_kernel(static_cast<KernelId>(id));
        if (variant == 2) return std::string(kKernelNames[id]) + "_grouped";
        if (variant && is_row_kernel((KernelId)id)) return std::string(kKernelNames[id]) + "_" + std::to_string(build_cols((KernelId)id, variant)) + "col";
        return std::string(kKernelNames[id]) + (variant && tile ? "_small" : "");
    }

    bool supports_type(uint32_t type) const override {
        const WeightKernels* format = weight_kernels(type);
        return format && (!format->mxfp4 || dev_->mxfp4);
    }
    std::vector<Dtype> native_dtypes() const override { return {Dtype::f16, Dtype::f32}; }
    bool emulates_dtype(Dtype dtype) const override { return dtype == Dtype::bf16 && dev_->preserve_float32; }
    std::string dtype_path(Dtype dtype) const override {
        if (dtype == Dtype::bf16) return "bf16 (matrix inputs), f32 (routers)";
        if (dtype == Dtype::f16)
            return dev_->profile.prefer_integer_dot
                ? "block-int16 (quantized rows and eligible tiles), f32 (other products, routers)"
                : "block-int16 (quantized rows), f32 (tiles, F32 weights, routers)";
        return "f32";
    }
    const std::string& name() const { return dev_->caps.device; }
    const DeviceProfile& profile() const { return dev_->profile; }

    // The driver's statistics for every kernel compiled so far, one line each: on AMD the vector and scalar register counts, scratch, shared memory and occupancy.
    // Empty when the device does not report them.
    std::string kernel_statistics() const {
        std::string out;
        const Device& d = *dev_;
        if (!d.exec_stats) return out;
        for (int slot = 0; slot < K_COUNT * kVariants; ++slot) {
            const int id = slot / kVariants, variant = slot % kVariants;
            const Kernel& k = kernels_[id][variant];
            if (!k.pipeline) continue;
            VkPipelineInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
            pi.pipeline = k.pipeline;
            uint32_t n = 0;
            if (d.get_exec_props(d.device, &pi, &n, nullptr) != VK_SUCCESS) continue;
            for (uint32_t e = 0; e < n; ++e) {
                VkPipelineExecutableInfoKHR ei{};
                ei.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
                ei.pipeline = k.pipeline;
                ei.executableIndex = e;
                uint32_t ns = 0;
                if (d.get_exec_stats(d.device, &ei, &ns, nullptr) != VK_SUCCESS) continue;
                std::vector<VkPipelineExecutableStatisticKHR> st(ns);
                for (auto& s : st) {
                    s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR;
                    s.pNext = nullptr;
                }
                if (d.get_exec_stats(d.device, &ei, &ns, st.data()) != VK_SUCCESS) continue;
                out += kernel_variant_name(id, variant);
                out += ':';
                for (const auto& s : st) {
                    out += ' ';
                    out += s.name;
                    out += '=';
                    switch (s.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR: out += s.value.b32 ? "1" : "0"; break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR: out += std::to_string(s.value.i64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR: out += std::to_string(s.value.u64); break;
                    default: out += std::to_string(s.value.f64); break;
                    }
                }
                out += '\n';
            }
        }
        return out;
    }

    // The driver's internal representations of every kernel compiled so far, under each kernel's name, for a backend opened for diagnostics.
    // A Q8_0 decode build's text starts with a line giving its shape and forms as its pipeline was made, and a two-row build of the Q4 and K-quant families with its columns and rows, which backend-vulkan --isa holds their float operations to.
    std::vector<std::pair<std::string, std::string>> kernel_representations() const {
        std::vector<std::pair<std::string, std::string>> out;
        const Device& d = *dev_;
        if (!d.exec_ir) return out;
        for (int slot = 0; slot < K_COUNT * kVariants; ++slot) {
            const int id = slot / kVariants, variant = slot % kVariants;
            const Kernel& k = kernels_[id][variant];
            if (!k.pipeline) continue;
            VkPipelineInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
            pi.pipeline = k.pipeline;
            uint32_t n = 0;
            if (d.get_exec_props(d.device, &pi, &n, nullptr) != VK_SUCCESS) continue;
            std::string text;
            if (id == K_MATMUL_VEC_Q8) {
                const VecBuild& b = kVecBuilds[variant];
                const uint32_t forms = vec_forms(variant, d.profile);
                text = "; q8_decode_build cols=" + std::to_string(b.cols) + " rows=" + std::to_string(b.rows) + " steps=" + std::to_string(b.steps) +
                       " tree=" + std::to_string((forms & kQ8Tree) ? 1 : 0) + " half=" + std::to_string((forms & kQ8Half) ? 1 : 0) + "\n";
            } else if (build_rows((KernelId)id, variant) > 1) {
                text = "; row_build cols=" + std::to_string(build_cols((KernelId)id, variant)) + " rows=" + std::to_string(build_rows((KernelId)id, variant)) + "\n";
            }
            for (uint32_t e = 0; e < n; ++e) {
                VkPipelineExecutableInfoKHR ei{};
                ei.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
                ei.pipeline = k.pipeline;
                ei.executableIndex = e;
                uint32_t nr = 0;
                if (d.get_exec_ir(d.device, &ei, &nr, nullptr) != VK_SUCCESS) continue;
                std::vector<VkPipelineExecutableInternalRepresentationKHR> reps(nr);
                for (auto& r : reps) {
                    r.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR;
                    r.pNext = nullptr;
                    r.pData = nullptr;
                }
                if (d.get_exec_ir(d.device, &ei, &nr, reps.data()) != VK_SUCCESS) continue;
                std::vector<std::vector<char>> data(nr);
                for (uint32_t r = 0; r < nr; ++r) {
                    data[r].resize(reps[r].dataSize + 1);
                    reps[r].pData = data[r].data();
                }
                if (d.get_exec_ir(d.device, &ei, &nr, reps.data()) != VK_SUCCESS) continue;
                for (uint32_t r = 0; r < nr; ++r) {
                    text += "== ";
                    text += reps[r].name;
                    text += " (";
                    text += reps[r].description;
                    text += ")\n";
                    // The driver's size counts the terminator, so the text is taken up to it.
                    if (reps[r].isText) text.append(data[r].data(), std::strlen(data[r].data()));
                    else text += std::to_string(reps[r].dataSize) + " bytes of binary data\n";
                    text += '\n';
                }
            }
            out.emplace_back(kernel_variant_name(id, variant), std::move(text));
        }
        return out;
    }

    // State and gating ops are supported; grouped expert projections must share one storage type.
    bool implements(Op op) const override { return op != Op::mixed_experts; }

    // Host worker counts mean nothing to a device.
    void set_threads(int) override {}
    int threads_available() const override { return 0; }

    // What the device-local heap can still take: the driver's budget less what is in use, or without the budget extension the heap's size, which overstates.
    std::optional<size_t> memory_available() const override {
        const Fn& fn = dev_->fn;
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp{};
        mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        if (dev_->memory_budget) mp.pNext = &budget;
        fn.vkGetPhysicalDeviceMemoryProperties2(dev_->physical, &mp);
        size_t free = 0;
        for (uint32_t h = 0; h < mp.memoryProperties.memoryHeapCount; ++h) {
            const VkMemoryHeap& heap = mp.memoryProperties.memoryHeaps[h];
            // The small device-local window the host can map (256 MiB on these cards) is not where weights go.
            if (!(heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) || heap.size < (VkDeviceSize(1) << 30)) continue;
            const VkDeviceSize total = dev_->memory_budget ? budget.heapBudget[h] : heap.size;
            const VkDeviceSize used = dev_->memory_budget ? budget.heapUsage[h] : 0;
            free = std::max(free, (size_t)(total > used ? total - used : 0));
        }
        return free;
    }

    // The upload staging and each ring slot's argument arena, host-visible memory held for the backend's life.
    size_t host_resident() const override { return kStagingBytes + kRing * kArenaBytes; }

    // A generated token is a class of its own; a longer extent's class is which of the profile's crossovers it has reached, of every type and row width (matmul_runs, expert_runs, attention), and the tile's split once it can take the tile.
    size_t row_class(size_t extent) const override {
        if (extent <= 1) return 1;
        const DeviceProfile& p = dev_->profile;
        const size_t tile[] = {p.tile_from_8bit, p.tile_from_8bit_narrow, p.tile_from_other, p.tile_from_other_narrow};
        const size_t other[] = {p.moe_tile_from, p.moe_tile_from_q4, p.moe_tile_from_q4k, p.moe_tile_from_q5k, p.attention_tile_rows};
        size_t reached = 0, least = SIZE_MAX;
        for (size_t t : tile) {
            reached = reached * 2 + (extent >= t);
            least = std::min(least, t);
        }
        for (size_t t : other) reached = reached * 2 + (extent >= t);
        return 2 + reached * 16 + (extent >= least ? split_tiles_of(extent) : 0);
    }

    HostTimes host_times() const override { return host_; }

    // The timestamps' sum over every kernel since the last reading.
    double device_ms() override {
        if (!dev_->timestamps) return -1.0;
        double ms = 0.0;
        for (const auto& k : kernel_times()) ms += k.second;
        return ms;
    }
    // Split partials and attention keep their reserve; devices that use the copied integer tile also reserve its capacity.
    size_t scratch_reserve(size_t free) const override {
        return ((size_t)256 << 20) + free / 20 + (dev_->profile.prefer_integer_dot ? kMxCopyBytes : 0);
    }

    // An adopted F32 matrix a product reads, whose rows are a multiple of 256 floats wide, keeps a padded copy beside it once a float tile has read it (padded_f32); routed stacks bind their data as it is.
    size_t resident_bytes(uint32_t type, size_t nin, size_t rows, size_t bytes, bool product) const override {
        return bytes + (product && pads_f32(type, nin) ? rows * (nin + kF32Pad) * sizeof(float) : 0);
    }

    // The tags name a buffer by its handle, which a new buffer can take over once the buffer it named is freed, so a new buffer drops them as a write from the host does.
    void drop_tags() {
        xq_tag_ = XqTag{};
        group_tag_ = GroupTag{};
    }

    BufferPtr alloc(size_t bytes, Memory where) override {
        drop_tags();
        auto b = std::make_shared<VulkanBuffer>(dev_, bytes, where == Memory::host_visible);
        if (bytes && !b->host_visible()) {
            // Zeroed like every other allocation, in stream order.
            // The slot holds the buffer until the fill retires, so a caller may drop it before anything is submitted.
            VkCommandBuffer cmd = open();
            pending_[ring_index_].push_back(b);
            dev_->fn.vkCmdFillBuffer(cmd, b->handle(), 0, VK_WHOLE_SIZE, 0);
            barrier(cmd);
        }
        return b;
    }

    // Device-local and not filled, since the caller writes every byte before an op reads it; adopted, so a float tile keeps a padded copy of it as it does of any adopted weight.
    BufferPtr alloc_weight(size_t bytes) override {
        drop_tags();
        auto b = std::make_shared<VulkanBuffer>(dev_, bytes, false);
        b->adopted = true;
        return b;
    }

    // Host memory the device imports and copies from in place, where the device imports host memory and `memory` and `bytes` are on its alignment; null otherwise, and where the import fails, so the caller writes through staging instead.
    BufferPtr wrap_host(void* memory, size_t bytes) override {
        const Device& d = *dev_;
        const size_t align = d.host_import_align;
        if (!d.host_pointer_props || !memory || !bytes || (uintptr_t)memory % align || bytes % align) return nullptr;
        VkMemoryHostPointerPropertiesEXT hp{};
        hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
        if (d.host_pointer_props(d.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, memory, &hp) != VK_SUCCESS || !hp.memoryTypeBits)
            return nullptr;
        try {
            return std::make_shared<VulkanBuffer>(dev_, memory, bytes, hp.memoryTypeBits);
        } catch (const std::exception&) {
            return nullptr;
        }
    }

    // alloc_weight's storage filled in chunks through staging, for a weight the model adopts as it is built.
    BufferPtr adopt(const void* src, size_t bytes) override {
        if (!src && bytes) throw std::runtime_error("vulkan: adopting null storage");
        auto b = std::static_pointer_cast<VulkanBuffer>(alloc_weight(bytes));
        try {
            upload(*b, 0, src, bytes, b);
        } catch (...) {
            // Earlier chunks may still target this local buffer when a later submission fails.
            sync();
            throw;
        }
        return b;
    }

    Ticket submit() override {
        const Ticket ticket = flush();
        if (hold_.on) hold_queue();
        return ticket;
    }

    // Holding (Backend::hold_between_submissions), while any holder remains: after each submit() the queue waits on an event the host sets at the next submission, or after kHoldMs, so the device stays busy, and its clock up, while another device runs its stage.
    void hold_between_submissions(bool on) override {
        if (!on) {
            if (!hold_.holders || --hold_.holders) return;
            std::lock_guard<std::mutex> lk(hold_.mu);
            hold_release();
            hold_.on = false;
            return;
        }
        if (!hold_.watchdog.joinable()) hold_setup();
        ++hold_.holders;
        hold_.on = true;
    }

    // The hold's command buffers, timeline, events and watchdog, made once; a failure part way destroys what it made, so a later call starts again.
    void hold_setup() {
        Fn& fn = dev_->fn;
        VkCommandBuffer cmds[kHolds] = {};
        VkSemaphore timeline = VK_NULL_HANDLE;
        VkEvent events[kHolds] = {};
        try {
            VkCommandBufferAllocateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ci.commandPool = pool_;
            ci.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ci.commandBufferCount = kHolds;
            check(fn.vkAllocateCommandBuffers(dev_->device, &ci, cmds), "vkAllocateCommandBuffers");
            VkSemaphoreTypeCreateInfo ti{};
            ti.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            VkSemaphoreCreateInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            si.pNext = &ti;
            check(fn.vkCreateSemaphore(dev_->device, &si, nullptr, &timeline), "vkCreateSemaphore");
            VkEventCreateInfo ei{};
            ei.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
            for (VkEvent& e : events) check(fn.vkCreateEvent(dev_->device, &ei, nullptr, &e), "vkCreateEvent");
            // A held queue whose host stopped submitting is let go, so an idle device idles and no wait outlasts the driver's job timeout.
            hold_.watchdog = std::thread([this] {
                std::unique_lock<std::mutex> lk(hold_.mu);
                while (!hold_.quit) {
                    if (!hold_.pending) { hold_.wake.wait(lk); continue; }
                    if (hold_.wake.wait_until(lk, hold_.since + std::chrono::milliseconds(kHoldMs)) == std::cv_status::timeout)
                        hold_release();
                }
            });
        } catch (...) {
            for (VkEvent e : events) if (e) fn.vkDestroyEvent(dev_->device, e, nullptr);
            if (timeline) fn.vkDestroySemaphore(dev_->device, timeline, nullptr);
            if (cmds[0]) fn.vkFreeCommandBuffers(dev_->device, pool_, kHolds, cmds);
            throw;
        }
        std::copy(cmds, cmds + kHolds, hold_.cmds);
        std::copy(events, events + kHolds, hold_.events);
        hold_.timeline = timeline;
    }

    // The queue's submission of the open command buffer, after letting a held queue go.
    Ticket flush() {
        if (hold_.on) {
            std::lock_guard<std::mutex> lk(hold_.mu);
            hold_release();
        }
        VkCommandBuffer cmd = open();
        chunk_ = 0;
        check(dev_->fn.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        const Ticket ticket = ++last_ticket_;
        VkTimelineSemaphoreSubmitInfo tsi{};
        tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &ticket;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.pNext = &tsi;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &timeline_;
        const VkResult r = dev_->fn.vkQueueSubmit(dev_->queue, 1, &si, VK_NULL_HANDLE);
        open_ = false;
        if (r != VK_SUCCESS) {
            --last_ticket_;
            check(r, "vkQueueSubmit");
        }
        ring_ticket_[ring_index_] = ticket;
        ring_index_ = (ring_index_ + 1) % kRing;
        return ticket;
    }

    // Callers hold hold_.mu.
    void hold_release() {
        if (!hold_.pending) return;
        dev_->fn.vkSetEvent(dev_->device, hold_.events[hold_.slot]);
        hold_.pending = false;
    }

    // A submission that waits on the next slot's event; the slot's previous hold has finished once the hold timeline reaches its value.
    void hold_queue() {
        const uint32_t i = (hold_.slot + 1) % kHolds;
        if (hold_.value[i]) {
            VkSemaphoreWaitInfo wi{};
            wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
            wi.semaphoreCount = 1;
            wi.pSemaphores = &hold_.timeline;
            wi.pValues = &hold_.value[i];
            check(dev_->fn.vkWaitSemaphores(dev_->device, &wi, UINT64_MAX), "vkWaitSemaphores");
        }
        check(dev_->fn.vkResetEvent(dev_->device, hold_.events[i]), "vkResetEvent");
        VkCommandBuffer cmd = hold_.cmds[i];
        check(dev_->fn.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(dev_->fn.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        dev_->fn.vkCmdWaitEvents(cmd, 1, &hold_.events[i], VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 0, nullptr, 0, nullptr, 0, nullptr);
        check(dev_->fn.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        const uint64_t value = hold_.last + 1;
        VkTimelineSemaphoreSubmitInfo tsi{};
        tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tsi.signalSemaphoreValueCount = 1;
        tsi.pSignalSemaphoreValues = &value;
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.pNext = &tsi;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &hold_.timeline;
        std::lock_guard<std::mutex> lk(hold_.mu);
        check(dev_->fn.vkQueueSubmit(dev_->queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
        hold_.last = hold_.value[i] = value;
        hold_.slot = i;
        hold_.pending = true;
        hold_.since = std::chrono::steady_clock::now();
        hold_.wake.notify_all();
    }

    void wait(Ticket t) noexcept override { waited(host_.ticket_ms, t); }

    // noexcept by contract: a device that cannot report its work finished has been lost, and nothing here can act on that.
    void wait_for(Ticket t) noexcept {
        if (t == 0 || t > last_ticket_) return;
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timeline_;
        wi.pValues = &t;
        const VkResult r = dev_->fn.vkWaitSemaphores(dev_->device, &wi, UINT64_MAX);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "vulkan: waiting on %s failed with %s; the device is lost\n",
                         dev_->caps.device.c_str(), vk_result_name(r));
            std::abort();
        }
    }

    // Device time per kernel since the last call, in milliseconds, for a diagnostics backend whose queue timestamps; reading them waits for the queue.
    // Dispatches whose time was sampled: the query pool bounds it, so a long run samples its first dispatches.
    size_t timed_dispatches() const { return last_timed_; }

    std::vector<std::pair<std::string, double>> kernel_times() {
        std::vector<std::pair<std::string, double>> out;
        if (!dev_->timestamps || !queries_) return out;
        sync();
        std::vector<uint64_t> stamps(query_next_);
        if (query_next_ &&
            dev_->fn.vkGetQueryPoolResults(dev_->device, queries_, 0, query_next_,
                                           stamps.size() * sizeof(uint64_t), stamps.data(),
                                           sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
                == VK_SUCCESS) {
            for (size_t i = 0; i + 1 < query_kernel_.size() * 2 && i + 1 < stamps.size(); i += 2) {
                const int k = query_kernel_[i / 2];
                kernel_ns_[k] += double(stamps[i + 1] - stamps[i]) * dev_->timestamp_ns;
                ++kernel_calls_[k];
            }
        }
        for (int i = 0; i < K_COUNT * kVariants; ++i)
            if (kernel_calls_[i])
                out.emplace_back(kernel_variant_name(i / kVariants, i % kVariants), kernel_ns_[i] / 1e6);
        // Since the last call: the next reading starts from an empty pool, which the next dispatch resets.
        last_timed_ = query_kernel_.size();
        std::fill(std::begin(kernel_ns_), std::end(kernel_ns_), 0.0);
        std::fill(std::begin(kernel_calls_), std::end(kernel_calls_), size_t(0));
        query_kernel_.clear();
        query_next_ = 0;
        queries_stale_ = true;
        return out;
    }

    void sync() noexcept override {
        if (open_) {
            try { flush(); } catch (const std::exception& e) {
                std::fprintf(stderr, "vulkan: %s; the device is lost\n", e.what());
                std::abort();
            }
        }
        wait(last_ticket_);
    }

    void read(const Buffer& src_b, size_t off, void* dst, size_t bytes) override {
        const VulkanBuffer& src = as_vulkan(src_b);
        span(src, off, bytes);
        if (!bytes) return;
        if (src.host_visible()) {
            sync();
            std::memcpy(dst, (const uint8_t*)src.mapped() + off, bytes);
            return;
        }
        VulkanBuffer& st = staging();
        size_t done = 0;
        while (done < bytes) {
            const size_t n = std::min(bytes - done, st.size());
            VkCommandBuffer cmd = open();
            barrier(cmd);
            VkBufferCopy region{off + done, 0, n};
            dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), st.handle(), 1, &region);
            barrier(cmd);
            wait(flush());
            std::memcpy((uint8_t*)dst + done, st.mapped(), n);
            done += n;
        }
    }

    void write(Buffer& dst_b, size_t off, const void* src, size_t bytes) override {
        if (!timed_) return write_now(dst_b, off, src, bytes);
        const auto t0 = std::chrono::steady_clock::now();
        const HostTimes before = host_;
        write_now(dst_b, off, src, bytes);
        host_.write_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() -
                          (host_.ticket_ms - before.ticket_ms) - (host_.slot_ms - before.slot_ms) - (host_.staging_ms - before.staging_ms);
    }

    void write_now(Buffer& dst_b, size_t off, const void* src, size_t bytes) {
        drop_tags();
        if (!src && bytes) throw std::runtime_error("vulkan: writing from null storage");
        VulkanBuffer& dst = as_vulkan(dst_b);
        drop_padded(dst);
        span(dst, off, bytes);
        if (!bytes) return;
        if (dst.host_visible()) {
            // An enqueued op may still be reading or writing the range.
            sync();
            std::memcpy((uint8_t*)dst.mapped() + off, src, bytes);
            return;
        }
        upload(dst, off, src, bytes);
    }

    void copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off,
              size_t bytes) override {
        drop_tags();
        VulkanBuffer& dst = as_vulkan(dst_b);
        const VulkanBuffer& src = as_vulkan(src_b);
        drop_padded(dst);
        span(dst, dst_off, bytes);
        span(src, src_off, bytes);
        if (!bytes) return;
        VkCommandBuffer cmd = open();
        if (!unordered_) barrier(cmd);
        VkBufferCopy region{src_off, dst_off, bytes};
        dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
        if (!unordered_) barrier(cmd);
    }

    // The narrower of the Q8_0 decode kernel's widest build and the other row kernels', each reading a weight once for its columns.
    size_t decode_columns() const override {
        return std::min<size_t>(dev_->profile.q8_decode_cols, dev_->profile.row_decode_cols ? dev_->profile.row_decode_cols : kRowColsWide);
    }

    // Barriers are left out while unordered, and one orders all of it against what follows once it ends.
    void unordered(bool on) override {
        const bool was = unordered_;
        unordered_ = on;
        if (was && !on) barrier(open());
    }

    // Elementwise kernels: one invocation per element.
    void add(Slice dst, CSlice src, size_t n) override {
        if (!n) return;
        const uint32_t pc[1] = {u32(n)};
        dispatch(K_ADD, {bind(dst), bind(src)}, pc, sizeof(pc), groups(n, 256));
    }

    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, RowRuns runs = {}) override {
        if (!n) return;
        const bool quant = n % 32 == 0;
        // Where the integer-dot tile reads dst next, the twin written four values a lane (shaders/silu_mul.comp).
        const size_t rows = runs.n ? runs.runs[runs.n - 1].end : 0;
        const bool tile = quant && rows && n % rows == 0 && tile_reads(n / rows, rows, runs);
        const uint32_t pc[2] = {u32(n), tile ? 2u : quant ? 1u : 0u};
        dispatch(K_SILU_MUL, {bind(dst), bind(gate), bind(up), quant ? xq_for(n) : bind(dst)}, pc, sizeof(pc),
                 groups(tile ? n / 4 : n, 256), 1);
        if (quant) xq_tag_ = XqTag{bind(dst), n};
    }

    // The output gate, and a scale of one value per row, one invocation per element; like silu_mul it writes the copy the matmul reading dst next takes (shaders/sigmoid_mul.comp).
    void sigmoid_mul(Slice dst, CSlice x, CSlice gate, size_t rows, size_t heads, size_t dim,
                     size_t gate_stride, size_t gate_head_stride, RowRuns runs = {}) override {
        if (!heads || !dim) throw std::runtime_error("vulkan: invalid sigmoid gate dimensions");
        if (!rows) return;
        const size_t width = size_mul(heads, dim), n = size_mul(rows, width);
        const size_t last = size_add(size_add(size_mul(rows - 1, gate_stride), size_mul(heads - 1, gate_head_stride)), dim);
        if (floats_from(dst) < n || floats_from(x) < n || floats_from(gate) < last)
            throw std::runtime_error("vulkan: sigmoid gate operand outside its allocation");
        const bool quant = width % 32 == 0;
        const bool tile = quant && tile_reads(width, rows, runs);
        struct { uint32_t n, quant, width, dim, gate_stride, gate_head_stride; }
            pc{u32(n), tile ? 2u : quant ? 1u : 0u, u32(width), u32(dim), u32(gate_stride), u32(gate_head_stride)};
        dispatch(K_SIGMOID_MUL, {bind(dst), bind(x), bind(gate), quant ? xq_for(n) : bind(dst)},
                 &pc, sizeof(pc), groups(tile ? n / 4 : n, 256), 1);
        if (quant) xq_tag_ = XqTag{bind(dst), n};
    }

    // One workgroup per (row, head); like rms_norm_rows it writes the copy the matmul reading dst next takes (shaders/gated_rms_norm.comp).
    void gated_rms_norm(Slice dst, CSlice x, CSlice z, CSlice w, size_t rows, size_t heads, size_t dim, float eps,
                        RowRuns runs = {}) override {
        if (!heads || !dim) throw std::runtime_error("vulkan: invalid gated norm dimensions");
        const size_t width = size_mul(heads, dim), n = size_mul(rows, width);
        if (floats_from(dst) < n || floats_from(x) < n || floats_from(z) < n || floats_from(w) < dim)
            throw std::runtime_error("vulkan: gated norm operand outside its allocation");
        if (!rows) return;
        const bool quant = dim % 32 == 0;
        const bool tile = quant && tile_reads(width, rows, runs);
        struct { uint32_t heads, dim; float eps; uint32_t quant, n; }
            pc{u32(heads), u32(dim), eps, tile ? 2u : quant ? 1u : 0u, u32(n)};
        dispatch(K_GATED_RMS_NORM, {bind(dst), bind(x), bind(z), bind(w), quant ? xq_for(n) : bind(dst)},
                 &pc, sizeof(pc), u32(rows * heads), 1);
        if (quant) xq_tag_ = XqTag{bind(dst), n};
    }

    // The storage of a call's state views, which must all be one, and the table the state kernels read: the view count, then per view its first batch row, rows, history length, source slot and destination slot; the conv appends its chunks.
    // check_state_views has checked the slots against the storage; the slots' floats must be addressable in 32 bits.
    struct StateTable {
        StateStorage* storage = nullptr;
        std::vector<uint32_t> words;
    };
    static StateTable state_table(const StateView* views, size_t n_views) {
        StateTable t;
        t.words.push_back(u32(n_views));
        size_t row0 = 0;
        for (size_t i = 0; i < n_views; ++i) {
            const StateView& v = views[i];
            if (t.storage && t.storage != v.storage) throw std::runtime_error("vulkan: state views of two storages in one call");
            t.storage = v.storage;
            t.words.insert(t.words.end(), {u32(row0), u32(v.nq), u32(v.length), u32(v.src), u32(v.dst)});
            row0 += v.nq;
        }
        (void)u32(size_mul(t.storage->slots(), t.storage->shape().slot_floats()));
        return t;
    }

    // Rows a conv invocation walks, a chunk of a view; at least the three a window holds, so only a view's first chunk reads the rows it carries in (shaders/causal_conv_silu.comp).
    static constexpr size_t kConvChunk = 16;
    // The most rows a view of a delta rule call takes for the short build to run it.
    static constexpr size_t kShortRows = 8;

    // One dispatch, an invocation per (chunk of a view's rows, channel), whose view's first chunk also leaves its carried rows (shaders/causal_conv_silu.comp).
    void causal_conv_silu(Slice out, CSlice x, CSlice w, size_t layer, const StateView* views, size_t n_views) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        StateTable t = state_table(views, n_views);
        const StateShape& sh = t.storage->shape();
        const size_t C = sh.channels();
        if (floats_from(out) < size_mul(rows, C) || floats_from(x) < size_mul(rows, C) || floats_from(w) < size_mul(C, kConvTaps))
            throw std::runtime_error("vulkan: conv operand outside its allocation");
        size_t chunks = 0;
        for (size_t i = 0; i < n_views; ++i)
            for (size_t b0 = 0; b0 < views[i].nq; b0 += kConvChunk, ++chunks) t.words.insert(t.words.end(), {u32(i), u32(b0)});
        if (chunks > dev_->props.limits.maxComputeWorkGroupCount[1]) throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        const struct { uint32_t C, slot, carried, chunk; }
            pc{u32(C), u32(sh.slot_floats()), u32(sh.v_heads * sh.matrix_floats()), u32(kConvChunk)};
        dispatch(K_CAUSAL_CONV_SILU,
                 {bind(out), bind(x), bind(w), bind(CSlice{&t.storage->layer(layer), 0}), args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 &pc, sizeof(pc), groups(C, 256), u32(chunks));
    }

    // One dispatch, a workgroup per (view, V head, 32 V columns), which norms q and k and makes the gates for each block of its view's tokens before running them (shaders/delta_rule.comp).
    void gated_delta_rule(Slice out, CSlice qkv, CSlice alpha, CSlice b, CSlice a, CSlice dt_bias,
                          size_t layer, const StateView* views, size_t n_views) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        const StateTable t = state_table(views, n_views);
        const StateShape& sh = t.storage->shape();
        const size_t Hk = sh.k_heads, Hv = sh.v_heads, Dk = sh.k_dim, Dv = sh.v_dim, C = sh.channels();
        // The recurrence holds a column's rows in registers, 16 in each of eight lanes, and stages 128 of each q and k row.
        if (Dk > 128) throw std::runtime_error("vulkan: a linear-attention K head wider than 128");
        if (floats_from(out) < size_mul(rows, Hv * Dv) || floats_from(qkv) < size_mul(rows, C) || floats_from(alpha) < size_mul(rows, Hv) ||
            floats_from(b) < size_mul(rows, Hv) || floats_from(a) < Hv || floats_from(dt_bias) < Hv)
            throw std::runtime_error("vulkan: delta rule operand outside its allocation");
        const size_t blocks = (Dv + 31) / 32;
        const struct { uint32_t C, k_heads, v_heads, k_dim, v_dim, slot, blocks; float scale, eps; }
            pc{u32(C), u32(Hk), u32(Hv), u32(Dk), u32(Dv), u32(sh.slot_floats()), u32(blocks), (float)(1.0 / std::sqrt((double)Dk)), kL2NormEps};
        // Views of at most kShortRows rows each take the build that stages 4 tokens a block, which leaves room for more workgroups: a decode step, a verify of a few drafts and its rerun (shaders/delta_rule.comp).
        size_t longest = 0;
        for (size_t i = 0; i < n_views; ++i) longest = std::max(longest, views[i].nq);
        dispatch(longest <= kShortRows ? K_DELTA_RULE_SHORT : K_DELTA_RULE,
                 {bind(out), bind(qkv), bind(alpha), bind(b), bind(a), bind(dt_bias), bind(CSlice{&t.storage->layer(layer), 0}),
                  args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 &pc, sizeof(pc), groups(size_mul(n_views * Hv, blocks), 1));
    }

    void gather_rows(Slice dst, CSlice src, size_t width, const uint32_t* rows,
                     size_t count) override {
        if (count && !rows) throw std::runtime_error("vulkan: gather without rows");
        if (!count || !width) return;
        // Rows are checked here, since a shader cannot refuse them.
        const size_t avail = floats_from(src);
        for (size_t i = 0; i < count; ++i)
            if (rows[i] >= avail / width)
                throw std::runtime_error("vulkan: gather row outside the allocation");
        const uint32_t pc[2] = {u32(width), u32(count)};
        dispatch(K_GATHER_ROWS, {bind(dst), bind(src), args(rows, count * sizeof(uint32_t))},
                 pc, sizeof(pc), groups(width * count, 256));
    }

    // Row kernels: one workgroup per row, or per (row, head).
    void rms_norm_rows(Slice dst, CSlice src, CSlice w, size_t rows, size_t n,
                       size_t stride, float eps, RowRuns runs = {}) override {
        if (!rows || !n) return;
        const bool quant = stride == n && n % 32 == 0;
        // Several workgroups a row when the output does not overlap the input (shaders/rms_norm_rows.comp), up to what fills the device: each reads the whole row for its sum, so a pass of many rows takes one a row.
        const bool overlap = dst.buffer == src.buffer &&
                             dst.offset < src.offset + rows * stride && src.offset < dst.offset + rows * stride;
        // Where the integer-dot tile reads dst next, the twin written four values a lane.
        const bool tile = quant && !overlap && tile_reads(n, rows, runs);
        const size_t fill = (4 * dev_->caps.compute_units + rows - 1) / rows;
        const size_t chunks = overlap ? 1 : std::max<size_t>(1, std::min((tile ? n / 4 + 255 : n + 255) / 256, fill));
        struct { uint32_t rows, n, stride; float eps; uint32_t quant, chunks; }
            pc{u32(rows), u32(n), u32(stride), eps, tile ? 2u : quant ? 1u : 0u, u32(chunks)};
        dispatch(K_RMS_NORM_ROWS, {bind(dst), bind(src), bind(w), quant ? xq_for(rows * n) : bind(dst)},
                 &pc, sizeof(pc), u32(rows * chunks), 1);
        if (quant) xq_tag_ = XqTag{bind(dst), rows * n};
    }

    // One workgroup per (row, head), reading the heads at their strides and writing them contiguously (shaders/norm_rope_partial.comp).
    void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride,
                           size_t heads, size_t head_dim, size_t rope_dim, CSlice w, float eps,
                           CSlice cos, CSlice sin, const uint32_t* pos) override {
        if (!heads || !head_dim || !rope_dim || rope_dim % 2 || rope_dim > head_dim)
            throw std::runtime_error("vulkan: invalid partial rope dimensions");
        if (rows && !pos) throw std::runtime_error("vulkan: rope without positions");
        if (!rows) return;
        const size_t width = size_mul(heads, head_dim), half = rope_dim / 2;
        const size_t last = size_add(size_add(size_mul(rows - 1, src_stride), size_mul(heads - 1, src_head_stride)), head_dim);
        if (floats_from(dst) < size_mul(rows, width) || floats_from(src) < last || floats_from(w) < head_dim)
            throw std::runtime_error("vulkan: partial rope operand outside its allocation");
        if (dst.buffer == src.buffer && dst.offset == src.offset && (src_stride != width || src_head_stride != head_dim))
            throw std::runtime_error("vulkan: partial rope in place over heads that are not contiguous");
        const size_t table = std::min(floats_from(cos), floats_from(sin)) / half;
        for (size_t r = 0; r < rows; ++r)
            if (pos[r] >= table) throw std::runtime_error("vulkan: position outside the RoPE table");
        struct { uint32_t src_stride, src_head_stride, heads, head_dim, half; float eps; }
            pc{u32(src_stride), u32(src_head_stride), u32(heads), u32(head_dim), u32(half), eps};
        dispatch(K_NORM_ROPE_PARTIAL,
                 {bind(dst), bind(src), bind(w), bind(cos), bind(sin), args(pos, rows * sizeof(uint32_t))},
                 &pc, sizeof(pc), u32(rows * heads));
    }

    // One dispatch for q and k norm-rope and the KV write, every view of the batch going through the view table.
    void norm_rope_kv(Slice q, size_t q_stride, size_t n_head, CSlice q_w,
                      Slice k, CSlice v, size_t kv_stride, size_t n_head_kv, CSlice k_w,
                      const RopeArgs& rope, size_t rows, size_t layer,
                      const KVView* views, size_t n_views) override {
        if (!rows || !n_head || !n_head_kv || !rope.half) {
            Backend::norm_rope_kv(q, q_stride, n_head, q_w, k, v, kv_stride, n_head_kv, k_w,
                                  rope, rows, layer, views, n_views);
            return;
        }
        if (!rope.pos) throw std::runtime_error("vulkan: rope without positions");
        const size_t table = floats_from(rope.cos) / rope.half;
        for (size_t r = 0; r < rows; ++r)
            if (rope.pos[r] >= table) throw std::runtime_error("vulkan: position outside the RoPE table");
        std::vector<Placed> placed = place_views(views, n_views);
        ViewTable t = view_table(layer, placed, true);
        if (!t.storage) return;
        VulkanKVStorage& s = *t.storage;
        const size_t dim = 2 * rope.half, hd = s.heads() * s.dim();
        if (s.dim() != dim || s.heads() != n_head_kv || t.rows != rows)
            throw std::runtime_error("vulkan: attention inputs do not match the KV storage");
        if (floats_from(q) < rows * q_stride || floats_from(k) < rows * kv_stride ||
            floats_from(v) < rows * kv_stride || q_stride < n_head * dim || kv_stride < hd)
            throw std::runtime_error("vulkan: attention rows outside their allocation");
        struct { uint32_t q_stride, n_head, kv_stride, n_head_kv, half; float eps; uint32_t bt; }
            pc{u32(q_stride), u32(n_head), u32(kv_stride), u32(n_head_kv), u32(rope.half),
               rope.eps, u32(kVkBlockTokens)};
        dispatch(kv_variant(K_NORM_ROPE_KV, K_NORM_ROPE_KV_K16, s),
                 {bind(q), bind(k), bind(v), bind(q_w), bind(k_w), bind(rope.cos), bind(rope.sin),
                  args(rope.pos, rows * sizeof(uint32_t)),
                  bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                  args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 &pc, sizeof(pc), u32(rows * (n_head + 2 * n_head_kv)));
    }

    void embed(Slice dst, uint32_t type, CSlice table, size_t nin, size_t nrows,
               const uint32_t* ids, size_t count) override {
        if (count && !ids) throw std::runtime_error("vulkan: embed without ids");
        if (!count || !nin) return;
        check_matrix(type, table, nin, nrows, "embedding");
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= nrows) throw std::runtime_error("vulkan: embedding row out of range");
        const uint32_t pc[2] = {u32(nin), type};
        // The table is bound twice, as floats for F32 rows and as bytes for block formats.
        dispatch(weight_kernels(type)->embed, {bind(dst), bind(table), bind(table), args(ids, count * sizeof(uint32_t))},
                 pc, sizeof(pc), u32(count));
    }

    // Ids past the table write zero rows; both embedding calls use the format's decoder (shaders/embed.comp).
    void embed_ids(Slice dst, uint32_t type, CSlice table, size_t nin, size_t nrows, CSlice ids, size_t count) override {
        if (!count || !nin) return;
        check_matrix(type, table, nin, nrows, "embedding");
        if (floats_from(ids) < count || floats_from(dst) < size_mul(count, nin))
            throw std::runtime_error("vulkan: embed_ids operand outside its allocation");
        const uint32_t pc[3] = {u32(nin), type, u32(nrows)};
        dispatch(weight_kernels(type)->embed_ids, {bind(dst), bind(table), bind(table), bind(ids)}, pc, sizeof(pc), u32(count));
    }

    // One workgroup a row (shaders/argmax_rows.comp); without prior ids the logits are bound in their place and not read.
    void argmax_rows(Slice ids, CSlice logits, size_t rows, size_t n, CSlice after = {}) override {
        if (!rows) return;
        if (!n || n >= UINT32_MAX) throw std::runtime_error("vulkan: argmax over no ids or more than 32 bits name");
        if (floats_from(ids) < rows || floats_from(logits) < size_mul(rows, n) || (after.buffer && floats_from(after) < rows))
            throw std::runtime_error("vulkan: argmax_rows operand outside its allocation");
        const uint32_t pc[2] = {u32(n), after.buffer ? 1u : 0u};
        dispatch(K_ARGMAX_ROWS, {bind(ids), bind(logits), after.buffer ? bind(after) : bind(logits)}, pc, sizeof(pc), u32(rows));
    }
    void matmul(uint32_t type, CSlice w, CSlice X, Slice Y, size_t nin, size_t nout,
                size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) override {
        const Projection one{type, w, Y, nout};
        matmul_runs({one}, X, nin, nbatch, false, runs, dtype);
    }
    // The residual add folded into the kernels' store: Y += W X.
    void matmul_add(uint32_t type, CSlice w, CSlice X, Slice Y, size_t nin, size_t nout,
                    size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) override {
        const Projection one{type, w, Y, nout};
        matmul_runs({one}, X, nin, nbatch, true, runs, dtype);
    }
    void matmul_group(std::initializer_list<Projection> projections, CSlice X,
                      size_t nin, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) override {
        matmul_runs(projections, X, nin, nbatch, false, runs, dtype);
    }

    // Batch rows from which a call takes the tile rather than the row kernel, by the types of the projections that have rows.
    // The row kernel costs a weight pass per eight columns and the tile a whole tile however little is filled, so the crossover depends on the row width and the types; measured per device (backends/device_profile.hpp).
    size_t tile_from(const Projection* projections, size_t count, size_t nin) const {
        bool eight_bit_or_float = true;
        for (size_t i = 0; i < count; ++i) {
            const Projection& pr = projections[i];
            if (pr.rows && !weight_kernels(pr.type)->fast_tile) eight_bit_or_float = false;
        }
        return tile_from_for(dev_->profile, eight_bit_or_float, nin);
    }

    // With row runs, a row's kernel follows its prompt's extent rather than the call's width, so a prompt computes the same however its rows are batched (docs/VULKAN.md, batch invariance).
    // Adjacent runs taking the same kernel are one call, and a call of mixed runs becomes one call per kernel over its rows.
    // A tile row's inner-dimension split is the one a pass over its whole prompt would take, up to a microbatch of 512 rows; runs whose splits differ are separate calls.
    static size_t split_tiles_of(size_t extent) { return (std::min<size_t>(extent, 512) + 63) / 64; }
    void matmul_runs(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nbatch,
                     bool accumulate, RowRuns runs, Dtype dtype) {
        if (!nbatch) return;
        check_group(projections.begin(), projections.size(), X, nin, nbatch);
        if (!runs.n) {
            matmul_group_impl(projections.begin(), projections.size(), X, nin, nbatch, accumulate, dtype);
            return;
        }
        const size_t from = tile_from(projections.begin(), projections.size(), nin);
        auto tile_split = [&](const RowRun& r) {
            const bool tile = dtype == Dtype::bf16 || r.extent >= from;
            return std::make_pair(tile, tile ? split_tiles_of(r.extent) : size_t(0));
        };
        for_each_run(nbatch, runs, tile_split, [&](size_t start, size_t rows, std::pair<bool, size_t> ts) {
            const int k = ts.first ? 1 : 0;
            if (rows == nbatch) {
                matmul_group_impl(projections.begin(), projections.size(), X, nin, nbatch, accumulate, dtype, k, ts.second);
            } else {
                std::vector<Projection> at(projections);
                for (Projection& pr : at) pr.out.offset += start * pr.rows;
                const CSlice xs{X.buffer, X.offset + start * nin};
                matmul_group_impl(at.data(), at.size(), xs, nin, rows, accumulate, dtype, k, ts.second);
            }
        });
    }

    // Throws unless the weights, outputs and X hold the call's `nbatch` rows.
    // matmul_runs checks a call whole before recording any of it, so a call its runs split into several is refused before the first.
    void check_group(const Projection* projections, size_t count, CSlice X, size_t nin, size_t nbatch) const {
        for (size_t i = 0; i < count; ++i) {
            const Projection& pr = projections[i];
            check_matrix(pr.type, pr.data, nin, pr.rows);
            if (floats_from(pr.out) < size_mul(nbatch, pr.rows)) throw std::runtime_error("vulkan: matmul operand outside its allocation");
        }
        if (floats_from(X) < size_mul(nbatch, nin)) throw std::runtime_error("vulkan: matmul operand outside its allocation");
    }

    // Up to three projections of one X in one dispatch: the row kernel for narrow batches hands workgroups to projections in order; wide batches take the tile.
    // `kernel_choice` forces the row kernel (0) or the tile (1), below zero the call's width chooses; `split_tiles` is the column tiles the tile's split is taken for, zero for the call's own.
    // The operands are those check_group accepted for the whole call.
    void matmul_group_impl(const Projection* projections, size_t count, CSlice X,
                           size_t nin, size_t nbatch, bool accumulate, Dtype dtype, int kernel_choice = -1, size_t split_tiles = 0) {
        // The kernels' limit; no caller passes more.
        if (count > 3) throw std::logic_error("vulkan: a matmul call of more than three projections");
        if (!nbatch) return;
        std::vector<const Projection*> live;
        for (size_t i = 0; i < count; ++i)
            if (projections[i].rows) live.push_back(&projections[i]);
        if (live.empty()) return;
        if (dtype == Dtype::bf16 || kernel_choice == 1 || (kernel_choice < 0 && nbatch >= tile_from(projections, count, nin))) {
            const size_t gy = (nbatch + 63) / 64;
            std::vector<Projection> copies;
            std::vector<CSlice> packed;
            copies.reserve(live.size());
            auto original = [&](const Projection* pr) -> const CSlice* {
                for (size_t i = 0; i < copies.size(); ++i) if (pr == &copies[i]) return &packed[i];
                return nullptr;
            };
            size_t copy_at = 0;
            if (dtype == Dtype::f16 && dev_->profile.prefer_integer_dot) {
                for (const Projection*& pr : live) {
                    if (pr->type != quant::GGML_TYPE_MXFP4) continue;
                    const size_t bytes = size_mul(pr->rows, size_mul(nin / 32, kMxCopyBlock));
                    if (bytes > kMxCopyBytes - copy_at) continue;
                    copy_mxfp4(pr->data, pr->rows, nin, copy_at);
                    packed.push_back(pr->data);
                    copies.push_back(Projection{pr->type, CSlice(mx_copy_.get(), copy_at / sizeof(float)), pr->out, pr->rows});
                    pr = &copies.back();
                    copy_at += (bytes + 255) & ~size_t(255);
                }
            }
            auto integer = [&](const Projection* pr) {
                return dtype == Dtype::f16 && integer_dot_tile(pr->type) &&
                       (pr->type != quant::GGML_TYPE_MXFP4 || original(pr));
            };
            if (gy > dev_->props.limits.maxComputeWorkGroupCount[1])
                throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
            // On a device whose integer dot is native, quantized types take the integer-dot tile (shaders/matmul_tile_q.comp), which reads X's 16-bit twin, made once for every projection that needs it.
            const size_t nblk = nin / 32;
            // The float tile, one projection a dispatch.
            for (const Projection* pr : live) {
                if (integer(pr)) continue;
                // Split as the integer-dot tile is, by the rows' whole prompt (matmul_runs), so a projection of few rows, such as a router's, still fills the device.
                const size_t st = split_tiles ? split_tiles : gy;
                const uint32_t hs = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, pr->rows, st, nin);
                const size_t steps = (nin + 31) / 32;   // an F32 row's last step may be partial
                // Split only below a quarter of a workgroup per compute unit to limit reduction overhead (docs/VULKAN.md).
                const size_t fwg = groups(pr->rows, hs) * st;
                const bool starved = fwg * 4 < dev_->caps.compute_units;
                const size_t base_kper = starved ? split_blocks(groups(pr->rows, kTileRowsSmall) * st, steps, dev_->profile.float_tile_split_per_cu) : steps;
                // Shorter F32 sums keep the BF16 depth agreement; retain a finer split chosen for occupancy.
                const size_t kper = dtype == Dtype::bf16 ? std::min(base_kper, (steps + 3) / 4) : base_kper;
                const size_t parts = (steps + kper - 1) / kper;
                // A taller tile reads less shared memory per product but halves the workgroups; below one per compute unit the call takes a shorter one, and a starved call the shortest, for the most workgroups.
                const uint32_t selected_height = starved ? kTileRowsSmall
                                                : tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, pr->rows, gy, nin);
                const uint32_t height = pr->type == quant::GGML_TYPE_F16 && selected_height == kTileRowsTall ? kTileRowsShort : selected_height;
                const bool tall = height == kTileRowsTall;
                const VkDescriptorBufferInfo wf = padded_f32(pr->data, pr->type, pr->rows, nin);
                const size_t wstride = wf.buffer == bind(pr->data).buffer ? nin : nin + kF32Pad;
                uint32_t pc[10] = {u32(nin), u32(pr->rows), u32(nbatch), pr->type, accumulate && parts == 1 ? 1u : 0u, 0, 0,
                                         u32(kper * 32), u32(gy), u32(wstride)};
                const KernelId kernel = float_tile_kernel(tall, dtype, pr->type);
                const int small = height == kTileRowsSmall ? 1 : 0;
                if (parts > 1) {
                    const size_t column_bytes = size_mul(size_mul(parts, pr->rows), pr->type == quant::GGML_TYPE_MXFP4 ? sizeof(double) : sizeof(float));
                    const size_t alignment = std::max<size_t>(1, dev_->props.limits.minStorageBufferOffsetAlignment / sizeof(float));
                    const size_t aligned = alignment / std::gcd(alignment, std::gcd(nin, pr->rows));
                    // Keep the same K parts and tile height while bounding their storage by column slices; each binding starts at the device's alignment.
                    const size_t block = column_bytes <= kFloatPartBytes / 64 ? std::max<size_t>(64, aligned) : aligned;
                    const size_t columns = std::max(block, kFloatPartBytes / column_bytes / block * block);
                    const size_t bytes = size_mul(std::min(nbatch, columns), column_bytes);
                    if (!parts_ || parts_->size() < bytes) grow(parts_, bytes);
                    const VkDescriptorBufferInfo pb{parts_->handle(), 0, VK_WHOLE_SIZE};
                    for (size_t first = 0; first < nbatch; first += columns) {
                        const size_t taken = std::min(nbatch - first, columns), tiles = (taken + 63) / 64;
                        const CSlice x{X.buffer, X.offset + first * nin};
                        const Slice y{pr->out.buffer, pr->out.offset + first * pr->rows};
                        pc[2] = u32(taken);
                        pc[8] = u32(tiles);
                        dispatch(kernel, {pb, bind(pr->data), wf, bind(x), bind(pr->data), bind(x)},
                                 pc, sizeof(pc), groups(pr->rows, height), u32(tiles * parts), small);
                        const size_t n = taken * pr->rows;
                        const uint32_t start = u32(groups(n, 256));
                        const uint32_t rc[7] = {u32(n), 0, 0, u32(parts), accumulate ? 1u : 0u, start, start + 1};
                        dispatch(pr->type == quant::GGML_TYPE_MXFP4 ? K_MATMUL_REDUCE_MXFP4 : K_MATMUL_REDUCE, {bind(y), bind(y), bind(y), pb}, rc, sizeof(rc), start);
                    }
                } else {
                    dispatch(kernel, {bind(pr->out), bind(pr->data), wf, bind(X), bind(pr->data), bind(X)},
                             pc, sizeof(pc), groups(pr->rows, height), (uint32_t)gy, small);
                }
                record_matrix_path(dtype == Dtype::bf16 ? MatrixPath::bf16 : MatrixPath::f32);
                if (overlaps_twin(bind(pr->out), nbatch * pr->rows)) xq_tag_ = XqTag{};
            }
            // The integer-dot tile takes the projections of one type in one dispatch, since each alone can be too small to fill the device.
            std::vector<const Projection*> pending;
            for (const Projection* pr : live)
                if (integer(pr)) pending.push_back(pr);
            while (!pending.empty()) {
                std::vector<const Projection*> group, rest;
                for (const Projection* pr : pending)
                    (pr->type == pending[0]->type ? group : rest).push_back(pr);
                pending.swap(rest);
                const VkDescriptorBufferInfo xt = tile_twin(X, nbatch * nin);
                const QTile t = qtile(group, gy, nin);
                // The split is the one the rows' whole prompt would take (matmul_runs).
                const size_t st = split_tiles ? split_tiles : gy;
                const uint32_t hs = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, t.rows, st, nin);
                size_t gxs = 0;
                for (const Projection* pr : group) gxs += groups(pr->rows, hs);
                const size_t kper = split_blocks(gxs * st, nblk);
                const size_t parts = (nblk + kper - 1) / kper;
                const uint32_t pc[15] = {u32(nin), u32(nbatch), group[0]->type, accumulate && parts == 1 ? 1u : 0u, u32(kper),
                                         u32(group.size()), t.nout[0], t.start[0], t.nout[1], t.start[1], t.nout[2], t.start[2], 0, 0, u32(nbatch)};
                const Projection &a = *t.p[0], &b = *t.p[1], &c = *t.p[2];
                const int small = t.height == kTileRowsSmall ? 1 : 0;
                const bool mx = a.type == quant::GGML_TYPE_MXFP4;
                if (parts > 1) {
                    const size_t n = nbatch * t.rows;
                    const size_t bytes = size_mul(size_mul(parts, n), mx ? sizeof(double) : sizeof(float));
                    if (!parts_ || parts_->size() < bytes) grow(parts_, bytes);
                    const VkDescriptorBufferInfo pb{parts_->handle(), 0, VK_WHOLE_SIZE};
                    if (mx)
                        dispatch(t.kernel, {pb, pb, pb, bind(a.data), bind(b.data), bind(c.data), xt, xt, xt,
                                           bind(*original(t.p[0])), bind(*original(t.p[1])), bind(*original(t.p[2]))},
                                 pc, sizeof(pc), u32(t.gx), u32(gy * parts), small);
                    else
                        dispatch(t.kernel, {pb, pb, pb, bind(a.data), bind(b.data), bind(c.data), xt, xt, xt}, pc, sizeof(pc), u32(t.gx),
                                 u32(gy * parts), small);
                    const size_t n0 = nbatch * a.rows, n1 = group.size() > 1 ? nbatch * b.rows : 0,
                                 n2 = group.size() > 2 ? nbatch * c.rows : 0;
                    const size_t start1 = groups(n0, 256), start2 = start1 + groups(n1, 256);
                    // An unused projection's start is past every workgroup.
                    const uint32_t rc[7] = {u32(n0), u32(n1), u32(n2), u32(parts), accumulate ? 1u : 0u,
                                            u32(n1 ? start1 : start2 + groups(n2, 256)), u32(n2 ? start2 : start2 + groups(n2, 256) + 1)};
                    dispatch(mx ? K_MATMUL_REDUCE_MXFP4 : K_MATMUL_REDUCE, {bind(a.out), bind(b.out), bind(c.out), pb}, rc, sizeof(rc),
                             u32(start2 + groups(n2, 256)));
                } else if (mx) {
                    dispatch(t.kernel, {bind(a.out), bind(b.out), bind(c.out), bind(a.data), bind(b.data), bind(c.data), xt, xt, xt,
                                       bind(*original(t.p[0])), bind(*original(t.p[1])), bind(*original(t.p[2]))},
                             pc, sizeof(pc), u32(t.gx), u32(gy), small);
                } else {
                    dispatch(t.kernel, {bind(a.out), bind(b.out), bind(c.out), bind(a.data), bind(b.data), bind(c.data), xt, xt, xt},
                             pc, sizeof(pc), u32(t.gx), (uint32_t)gy, small);
                }
                record_matrix_path(MatrixPath::block_int16);
                for (const Projection* pr : group)
                    if (overlaps_twin(bind(pr->out), nbatch * pr->rows)) xq_tag_ = XqTag{};
            }
            return;
        }
        // One cluster size and one module serve a dispatch, so every projection in it has the same type.
        // A mixed group, such as the Q5_K q and k beside the Q6_K v of a Q5_K_M file, is partitioned by type and each partition is one dispatch: two for that group rather than three.
        // The partitions keep the row kernel the whole group chose, since a partition's own types could move its crossover.
        for (size_t i = 1; i < live.size(); ++i)
            if (live[i]->type != live[0]->type) {
                std::vector<Projection> same, rest;
                for (const Projection* pr : live)
                    (pr->type == live[0]->type ? same : rest).push_back(*pr);
                matmul_group_impl(same.data(), same.size(), X, nin, nbatch, accumulate, dtype, 0);
                matmul_group_impl(rest.data(), rest.size(), X, nin, nbatch, accumulate, dtype, 0);
                return;
            }
        const RowPlan plan = row_plan(live[0]->type, nin, dtype);
        const VkDescriptorBufferInfo xqi = row_twin(X, live[0]->type, nbatch * nin, dtype);
        for_each_column_chunk(plan.kernel, nbatch, [&](size_t col0, size_t ncols, int variant) {
            row_dispatch(plan, live, X, xqi, nin, nbatch, col0, ncols, accumulate, variant);
        });
    }

    // The builds a row kernel's plain columns take on this device, narrowest first, as pipeline variants; returns their count.
    // The Q8_0 decode kernel takes its builds up to the profile's q8_decode_cols, the Q4 and K-quant families their one-column build and their two-row builds up to the profile's row_decode_cols where it sets one, every other row kernel its one-column build where it has one and its wide build.
    size_t column_builds(KernelId id, int (&out)[kVariants]) const {
        size_t n = 0;
        if (id == K_MATMUL_VEC_Q8) {
            for (int v : kVecByWidth)
                if (kVecBuilds[v].capacity() <= dev_->profile.q8_decode_cols) out[n++] = v;
            return n;
        }
        if (row_kernel_builds_one_column(id)) out[n++] = 1;
        if (row_kernel_builds_two_rows(id) && dev_->profile.row_decode_cols >= kRowBuilds[kRowByWidth[0]].cols) {
            for (int v : kRowByWidth)
                if (kRowBuilds[v].cols <= dev_->profile.row_decode_cols) out[n++] = v;
            return n;
        }
        out[n++] = 0;
        return n;
    }

    // Calls `each(col0, ncols, variant)` over a pass's columns: chunks of the widest build while more columns remain than it holds, then the rest in the narrowest build that holds them.
    // Every build gives a column the same bits, so how a pass is chunked changes only its time.
    // Each Q8_0 decode build holds twice the columns of the next narrower, so each of that kernel's chunks fills more than half its build.
    // A build of one column group then checks the count only before the groups past its first half, and the second group of a two-group build, which gets 1 to all of its columns, before each group.
    // So does each two-row build of the Q4 and K-quant families, which computes every column it holds.
    template <typename Fn>
    void for_each_column_chunk(KernelId id, size_t nbatch, const Fn& each) const {
        int builds[kVariants] = {};
        const size_t n = column_builds(id, builds);
        const size_t wide = build_cols(id, builds[n - 1]);
        size_t col0 = 0;
        for (; nbatch - col0 > wide; col0 += wide) each(col0, wide, builds[n - 1]);
        size_t i = 0;
        while (build_cols(id, builds[i]) < nbatch - col0) ++i;
        each(col0, nbatch - col0, builds[i]);
    }

    // The row kernel for a type at a width (matmul_row.comp): its module, whether rows take the wide layout, and how a subgroup's lanes split over rows.
    // Q8_0 pairs go over four lanes and Q4_0 pairs over two when the block count is even, Q4_1 blocks over one, the K-quant blocks over eight, else one unit per block or value.
    // The Q8_0 decode kernel reads no cluster, and its rows follow its build (sg_rows).
    struct RowPlan {
        KernelId kernel;
        uint32_t type, wide, cluster;
    };
    RowPlan row_plan(uint32_t type, size_t nin, Dtype dtype) const {
        const WeightKernels& format = *weight_kernels(type);
        const size_t nblocks = nin / quant::storage_type(type)->block_size;
        uint32_t wide = 0, lanes = 1;
        size_t units = nin;
        const bool dot = format.mxfp4 ? dev_->profile.mxfp4_integer_dot : dev_->profile.prefer_integer_dot;
        KernelId kernel = dot ? format.dot_row : format.row;
        switch (format.layout) {
        case RowLayout::values: break;
        case RowLayout::blocks: units = nblocks; break;
        case RowLayout::q8_pairs:
            wide = nblocks % 2 == 0 && nblocks / 2 >= kQ8LanesPerPair;
            lanes = wide ? kQ8LanesPerPair : 1;
            units = wide ? nblocks / 2 * lanes : nblocks;
            if (wide) kernel = K_MATMUL_ROW_Q8W;
            break;
        case RowLayout::q4_pairs:
            wide = nblocks % 2 == 0;
            lanes = wide ? 2 : 1;
            units = wide ? nblocks / 2 * lanes : nblocks;
            break;
        case RowLayout::k_blocks:
            lanes = kKQuantLanes;
            units = nblocks * lanes;
            break;
        }
        if (dtype == Dtype::f32 && format.layout != RowLayout::values) {
            kernel = format.float_row;
            units = nin;
            lanes = 1;
            wide = 0;
        }
        uint32_t cluster = lanes;
        while (cluster < dev_->caps.subgroup_size && cluster < units) cluster *= 2;
        if (kernel == K_MATMUL_ROW_K4_DOT || kernel == K_MATMUL_ROW_K5_DOT)
            cluster = std::min(cluster, std::max(lanes, dev_->profile.k45_row_lanes));
        // Where the integer dot is native, Q8_0 rows take the integer dot over the 16-bit twin (shaders/matmul_vec_q8.comp).
        if (dtype == Dtype::f16 && type == quant::GGML_TYPE_Q8_0 && dev_->profile.prefer_integer_dot) kernel = K_MATMUL_VEC_Q8;
        return RowPlan{kernel, type, wide, cluster};
    }

    // Rows a subgroup takes in a row kernel's build: in matmul_row.comp the build's rows a cluster of lanes (build_rows), in the Q8_0 decode kernel its build's own (kVecBuilds).
    uint32_t sg_rows(const RowPlan& plan, int variant) const {
        return plan.kernel == K_MATMUL_VEC_Q8 ? kVecBuilds[variant].rows : dev_->caps.subgroup_size / plan.cluster * build_rows(plan.kernel, variant);
    }

    // What a row kernel reads X through: the floats for F32 weights or policy, else the activations' twin (shaders/xquant.glsl), which the norm, SiLU and attention kernels write beside their output and tag.
    // An input without one gets a quantize dispatch here; the scratch is reused stream-ordered.
    VkDescriptorBufferInfo row_twin(CSlice X, uint32_t type, size_t n, Dtype dtype) {
        if (weight_kernels(type)->layout == RowLayout::values || dtype == Dtype::f32) return bind(X);
        const VkDescriptorBufferInfo xf = bind(X);
        VkDescriptorBufferInfo xqi = xq_for(n);
        if (!(xq_tag_.n == n && xq_tag_.x.buffer == xf.buffer && xq_tag_.x.offset == xf.offset)) {
            const uint32_t qpc[1] = {u32(n)};
            dispatch(K_QUANTIZE_X, {xf, xqi}, qpc, sizeof(qpc), groups(n, 256), 1);
            xq_tag_ = XqTag{xf, n};
        }
        return xqi;
    }

    // One row kernel dispatch through the build `variant` over up to three projections of one type and columns col0 .. col0 + ncols of X, which has nbatch columns.
    // A routed dispatch (`per` nonzero) instead runs one entry per workgroup row, `entries` of them, through the expert ids in `ids`.
    void row_dispatch(const RowPlan& plan, const std::vector<const Projection*>& live, CSlice X, VkDescriptorBufferInfo xqi,
                      size_t nin, size_t nbatch, size_t col0, size_t ncols, bool accumulate, int variant,
                      uint32_t per = 0, size_t entries = 1, VkDescriptorBufferInfo ids = {},
                      uint32_t order0 = 0, VkDescriptorBufferInfo tab = {}, size_t routed = 0) {
        // A Q8_0 decode build whose column groups take the same rows gives each workgroup's rows that many adjacent workgroups.
        const uint32_t per_group = (256 / dev_->caps.subgroup_size) * sg_rows(plan, variant);
        const uint32_t span = plan.kernel == K_MATMUL_VEC_Q8 ? kVecBuilds[variant].span : 1;
        uint32_t nout[3] = {0, 0, 0}, start[3] = {0, 0, 0};
        uint32_t total = 0;
        for (size_t i = 0; i < live.size(); ++i) {
            nout[i] = u32(live[i]->rows);
            start[i] = total;
            total += groups(live[i]->rows, per_group) * span;
        }
        if (total > dev_->props.limits.maxComputeWorkGroupCount[0] || entries > dev_->props.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        // Unused projection slots bind the first one's buffers; no workgroup reaches them.
        const Projection& a = *live[0];
        const Projection& b = live.size() > 1 ? *live[1] : a;
        const Projection& c = live.size() > 2 ? *live[2] : a;
        const uint32_t t = plan.type, w = plan.wide;
        // matmul_row.comp counts its rows by cluster, the Q8_0 decode kernel reads neither.
        const uint32_t pc[22] = {u32(nin), u32(nbatch), u32(col0), u32(ncols), plan.cluster, dev_->caps.subgroup_size / plan.cluster,
                                 (uint32_t)live.size(),
                                 nout[0], t, w, start[0],
                                 nout[1], t, w, start[1],
                                 nout[2], t, w, start[2], accumulate ? 1u : 0u, per, order0};
        dispatch(plan.kernel,
                 {bind(a.out), bind(b.out), bind(c.out),
                  bind(a.data), bind(b.data), bind(c.data),
                  bind(a.data), bind(b.data), bind(c.data),
                  bind(a.data), bind(b.data), bind(c.data),
                  bind(X),
                  bind(a.data), bind(b.data), bind(c.data),
                  xqi, xqi, xqi, xqi, ids.buffer ? ids : bind(X), tab.buffer ? tab : bind(X)},
                 pc, sizeof(pc), total, u32(entries), variant);
        record_matrix_path(weight_kernels(a.type)->layout == RowLayout::values || (plan.kernel == K_MATMUL_ROW_FLOAT_X || plan.kernel == K_MATMUL_ROW_MXFP4_FLOAT_X) ? MatrixPath::f32 : MatrixPath::block_int16);
        // The outputs may overlap what the twin describes; a router's scores beside its input do not, so the experts read the same twin.
        for (const Projection* pr : live)
            if (overlaps_twin(bind(pr->out), (routed ? routed : per ? entries : nbatch) * pr->rows)) xq_tag_ = XqTag{};
    }

    // Expert routing and the routed projections (backend.hpp).
    // A row's entries take the row kernel, one workgroup row per entry, or with a prompt extent of at least the weight type's moe_tile_from the tile kernel over each expert's grouped entries.
    // Either way an entry computes the same whatever else is routed beside it, since neither kernel's arithmetic for a column depends on the other columns and a routed tile is never split.
    void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize, Slice ids, Slice weights) override {
        if (!k || k > n_expert || k > 256 || n_expert > 1024)
            throw std::runtime_error("vulkan: routing takes 1 to 256 of at most 1024 experts");
        if (!rows) return;
        if (floats_from(scores) < rows * n_expert || floats_from(ids) < rows * k || floats_from(weights) < rows * k)
            throw std::runtime_error("vulkan: routing operand outside its allocation");
        const uint32_t pc[3] = {u32(n_expert), u32(k), normalize ? 1u : 0u};
        dispatch(K_MOE_ROUTE, {bind(scores), bind(ids), bind(weights)}, pc, sizeof(pc), u32(rows));
        group_tag_ = GroupTag{};
        if (overlaps_twin(bind(ids), rows * k) || overlaps_twin(bind(weights), rows * k)) xq_tag_ = XqTag{};
    }

    // Calls `each(first, count, tile)` over the token rows of a routed call, a call per stretch of rows that take the same kernel.
    template <typename Fn>
    void expert_runs(uint32_t type, size_t nrows, RowRuns runs, const Fn& each) {
        const size_t from = dev_->profile.*(weight_kernels(type)->moe_from);
        if (!runs.n) { each(0, nrows, nrows >= from); return; }
        for_each_run(nrows, runs, [&](const RowRun& r) { return r.extent >= from; }, each);
    }

    static CSlice at_float(CSlice s, size_t floats) { return CSlice{s.buffer, s.offset + floats}; }
    static Slice at_float(Slice s, size_t floats) { return Slice{s.buffer, s.offset + floats}; }

    void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nrows,
                        const Routing& routing, RowRuns runs, Dtype dtype = Dtype::f16) override {
        const size_t k = routing.k, entries = nrows * k;
        if (!entries || !projections.size()) return;
        if (projections.size() > 3) throw std::logic_error("vulkan: more than three routed projections");
        for (const Projection& pr : projections) {
            check_matrix(pr.type, pr.data, nin, size_mul(routing.n_expert, pr.rows));
            if (floats_from(pr.out) < entries * pr.rows) throw std::runtime_error("vulkan: matmul operand outside its allocation");
            if (pr.type != projections.begin()->type) throw std::runtime_error("vulkan: routed projections of different types");
        }
        if (floats_from(X) < nrows * nin || floats_from(routing.ids) < entries)
            throw std::runtime_error("vulkan: matmul operand outside its allocation");
        expert_runs(projections.begin()->type, nrows, runs, [&](size_t first, size_t count, bool tile) {
            std::vector<Projection> at(projections);
            for (Projection& pr : at) pr.out = at_float(pr.out, first * k * pr.rows);
            std::vector<const Projection*> live;
            for (const Projection& pr : at) live.push_back(&pr);
            routed_call(live, at_float(X, first * nin), nin, count, count, k, at_float(routing.ids, first * k), routing.n_expert, tile, dtype);
        });
    }

    void matmul_experts_add(uint32_t type, CSlice data, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows,
                            const Routing& routing, RowRuns runs, Dtype dtype = Dtype::f16) override {
        const size_t k = routing.k, entries = nrows * k;
        if (!entries) return;
        check_matrix(type, data, nin, size_mul(routing.n_expert, nout));
        if (floats_from(X) < entries * nin || floats_from(Y) < nrows * nout || floats_from(routing.ids) < entries ||
            floats_from(routing.weights) < entries)
            throw std::runtime_error("vulkan: matmul operand outside its allocation");
        if (!moe_out_ || moe_out_->size() < entries * nout * sizeof(float)) grow(moe_out_, entries * nout * sizeof(float));
        expert_runs(type, nrows, runs, [&](size_t first, size_t count, bool tile) {
            const Slice part{moe_out_.get(), first * k * nout};
            const Projection one{type, data, part, nout};
            routed_call({&one}, at_float(X, first * k * nin), nin, count * k, count, k, at_float(routing.ids, first * k),
                        routing.n_expert, tile, dtype, 1);
            const uint32_t pc[3] = {u32(count), u32(nout), u32(k)};
            dispatch(K_MOE_COMBINE, {bind(at_float(Y, first * nout)), bind(CSlice(part)), bind(at_float(routing.weights, first * k))},
                     pc, sizeof(pc), groups(count * nout, 256));
        });
    }

    // One routed call over `nrows` token rows, k entries each, whose X has `xcols` columns: entry e reads column e / per, per being k when X is the token rows and 1 when it holds a column per entry.
    void routed_call(const std::vector<const Projection*>& live, CSlice X, size_t nin, size_t xcols, size_t nrows, size_t k,
                     CSlice ids, size_t n_expert, bool tile, Dtype dtype, size_t per = 0) {
        if (dtype == Dtype::bf16) tile = true;
        if (!per) per = k;
        const size_t entries = nrows * k;
        const uint32_t type = live[0]->type;
        if (!tile && entries < 2 * n_expert) {
            // Generated tokens whose entries average fewer than two an expert: each entry its own workgroup row, through the one-column build, since grouping them would save few reads and costs a dispatch.
            const RowPlan plan = row_plan(type, nin, dtype);
            const VkDescriptorBufferInfo xqi = row_twin(X, type, xcols * nin, dtype);
            for_each_column_chunk(plan.kernel, 1, [&](size_t, size_t, int variant) {
                row_dispatch(plan, live, X, xqi, nin, xcols, 0, 1, false, variant, u32(per), entries, bind(ids));
            });
            return;
        }
        // Grouped by expert: the tiles in runs of 64, the row kernels in runs of their column count, so an expert's rows are read once per run rather than once per entry; at most one partial run per expert that has any.
        const size_t chunk = tile ? 64 : kRowColsWide;
        const size_t max_tiles = (entries + chunk - 1) / chunk + std::min(n_expert, entries);
        const size_t tab_bytes = (4 * max_tiles + entries) * sizeof(uint32_t);
        if (!moe_tab_ || moe_tab_->size() < tab_bytes) {
            grow(moe_tab_, tab_bytes);
            group_tag_ = GroupTag{};
        }
        const VkDescriptorBufferInfo tab{moe_tab_->handle(), 0, VK_WHOLE_SIZE};
        // The down projection routes the same ids as gate and up, so their grouping is reused.
        const VkDescriptorBufferInfo idb = bind(ids);
        if (!(group_tag_.ids.buffer == idb.buffer && group_tag_.ids.offset == idb.offset && group_tag_.entries == entries &&
              group_tag_.n_expert == n_expert && group_tag_.chunk == chunk)) {
            const uint32_t gpc[4] = {u32(entries), u32(n_expert), u32(max_tiles), u32(chunk)};
            dispatch(K_MOE_GROUP, {idb, tab}, gpc, sizeof(gpc), u32(n_expert));
            group_tag_ = GroupTag{idb, entries, n_expert, chunk};
        }
        if (max_tiles > dev_->props.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        const uint32_t order0 = u32(4 * max_tiles);
        if (!tile) {
            // Enough generated tokens that experts repeat: each run of one expert's entries a workgroup row of the row kernel's wide build, a column per entry, so the expert's rows are read once per run.
            // A column computes the same in either build and as it would alone, so an entry does not depend on what else is routed beside it.
            const RowPlan plan = row_plan(type, nin, dtype);
            const VkDescriptorBufferInfo xqi = row_twin(X, type, xcols * nin, dtype);
            row_dispatch(plan, live, X, xqi, nin, xcols, 0, kRowColsWide, false, 2, u32(per), max_tiles, bind(ids), order0, tab, entries);
            return;
        }
        if (dtype == Dtype::f16 && integer_dot_tile(type) && type != quant::GGML_TYPE_MXFP4) {
            const VkDescriptorBufferInfo xt = tile_twin(X, xcols * nin);
            const QTile t = qtile(live, max_tiles, nin);
            const Projection &a = *t.p[0], &b = *t.p[1], &c = *t.p[2];
            const uint32_t pc[15] = {u32(nin), u32(xcols), type, 0, u32(nin / 32), u32(live.size()),
                                     t.nout[0], t.start[0], t.nout[1], t.start[1], t.nout[2], t.start[2], u32(per), order0, u32(xcols)};
            dispatch(t.kernel, {bind(a.out), bind(b.out), bind(c.out), bind(a.data), bind(b.data), bind(c.data), xt, xt, tab},
                     pc, sizeof(pc), u32(t.gx), u32(max_tiles), t.height == kTileRowsSmall ? 1 : 0);
            record_matrix_path(MatrixPath::block_int16);
        } else {
            for (const Projection* pr : live) {
                const uint32_t height = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall,
                                                      pr->rows, max_tiles, nin);
                const uint32_t pc[10] = {u32(nin), u32(pr->rows), u32(xcols), type, 0, u32(per), order0, u32(nin), u32(max_tiles), u32(nin)};
                dispatch(float_tile_kernel(height == kTileRowsTall, dtype, type),
                         {bind(pr->out), bind(pr->data), bind(pr->data), bind(X), bind(pr->data), tab},
                         pc, sizeof(pc), groups(pr->rows, height), u32(max_tiles), height == kTileRowsSmall ? 1 : 0);
                record_matrix_path(dtype == Dtype::bf16 ? MatrixPath::bf16 : MatrixPath::f32);
            }
        }
        for (const Projection* pr : live)
            if (overlaps_twin(bind(pr->out), entries * pr->rows)) xq_tag_ = XqTag{};
    }

    // Whether `floats` floats at a binding overlap the input the activations' twin was made from.
    bool overlaps_twin(const VkDescriptorBufferInfo& b, size_t floats) const {
        if (!xq_tag_.n || b.buffer != xq_tag_.x.buffer) return false;
        const VkDeviceSize end = b.offset + floats * sizeof(float), tag_end = xq_tag_.x.offset + xq_tag_.n * sizeof(float);
        return b.offset < tag_end && xq_tag_.x.offset < end;
    }

    // Throws unless the kernels decode `type` and `data` holds `rows` rows of `nin` values of it, each whole blocks.
    // `what` names the operand in the errors; tests/common.py matches the unsupported-type one to skip a model the device has no kernel for.
    void check_matrix(uint32_t type, CSlice data, size_t nin, size_t rows, const char* what = "matrix") const {
        if (!supports_type(type))
            throw std::runtime_error(std::string("vulkan: unsupported ") + what + " type " + std::to_string(type) +
                                     " (docs/VULKAN.md lists the types the kernels decode)");
        if (bytes_from(data) < size_mul(rows, quant::row_bytes(type, nin)))
            throw std::runtime_error(std::string("vulkan: ") + what + " outside its allocation");
    }

    // The scratch the twin of an n-value input lives in.
    VkDescriptorBufferInfo xq_for(size_t n) {
        const size_t bytes = n * 2 + (n / 32) * 8;
        if (!xq_ || xq_->size() < bytes) {
            grow(xq_, bytes);
            xq_tag_ = XqTag{};
        }
        return VkDescriptorBufferInfo{xq_->handle(), 0, VK_WHOLE_SIZE};
    }

    // The 16-bit twin of an n-value batch the integer-dot tile reads: the one a producer wrote, or one made here four values a lane (shaders/quantize_xw.comp).
    VkDescriptorBufferInfo tile_twin(CSlice X, size_t n) {
        const VkDescriptorBufferInfo xf = bind(X), xt = xq_for(n);
        if (!(xq_tag_.n == n && xq_tag_.x.buffer == xf.buffer && xq_tag_.x.offset == xf.offset)) {
            const uint32_t pc[1] = {u32(n)};
            dispatch(K_QUANTIZE_XW, {xf, xt}, pc, sizeof(pc), groups(n / 4, 256));
            xq_tag_ = XqTag{xf, n};
        }
        return xt;
    }

    // Whether a type's wide matmul goes through the integer-dot tile on this device; profile_for prefers the integer dot only where the device has it.
    bool integer_dot_tile(uint32_t type) const {
        return dev_->profile.prefer_integer_dot && weight_kernels(type)->integer_tile != K_COUNT;
    }

    // An integer-dot tile call over up to three projections of one type: the height their rows and `column_groups` call for, its module, and each projection's first workgroup and rows.
    // Unused slots repeat the first projection, which no workgroup reaches.
    struct QTile {
        uint32_t height = 0;
        KernelId kernel = K_MATMUL_TILE_Q;
        size_t rows = 0, gx = 0;
        uint32_t start[3] = {0, 0, 0}, nout[3] = {0, 0, 0};
        const Projection* p[3] = {nullptr, nullptr, nullptr};
    };
    QTile qtile(const std::vector<const Projection*>& ps, size_t column_groups, size_t nin) const {
        QTile t;
        for (const Projection* pr : ps) t.rows += pr->rows;
        t.height = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, t.rows, column_groups, nin);
        const WeightKernels& format = *weight_kernels(ps[0]->type);
        // Q6_K has only the short module: the tall build would hold twice the half sums.
        if (format.integer_tile == format.integer_tall_tile && t.height == kTileRowsTall) t.height = kTileRowsShort;
        t.kernel = t.height == kTileRowsTall ? format.integer_tall_tile : format.integer_tile;
        for (size_t i = 0; i < 3; ++i) {
            t.p[i] = i < ps.size() ? ps[i] : ps[0];
            if (i >= ps.size()) continue;
            t.start[i] = u32(t.gx);
            t.nout[i] = u32(ps[i]->rows);
            t.gx += groups(ps[i]->rows, t.height);
        }
        return t;
    }

    // One bounded stream-ordered scratch holds this call's exact float-scale copies.
    void copy_mxfp4(CSlice data, size_t rows, size_t nin, size_t at) {
        const size_t blocks = size_mul(rows, nin / 32);
        if (!mx_copy_) grow(mx_copy_, kMxCopyBytes);
        const VkDescriptorBufferInfo output{mx_copy_->handle(), at, VK_WHOLE_SIZE};
        const uint32_t pc[] = {u32(blocks)};
        const size_t total = (size_mul(blocks, kMxCopyBlock / 4) + 255) / 256;
        const size_t gx = std::min(total, size_t(dev_->props.limits.maxComputeWorkGroupCount[0]));
        dispatch(K_COPY_MXFP4, {bind(data), output}, pc, sizeof(pc), u32(gx), u32(groups(total, gx)));
    }

    // The binding a float tile reads a matrix through: for an adopted matrix pads_f32 picks, its padded copy, made on first use and kept with the buffer.
    VkDescriptorBufferInfo padded_f32(CSlice data, uint32_t type, size_t rows, size_t nin) {
        if (!pads_f32(type, nin) || !rows) return bind(data);
        VulkanBuffer& src = const_cast<VulkanBuffer&>(as_vulkan(*data.buffer));
        if (!src.adopted) return bind(data);
        const size_t off = data.offset * sizeof(float);
        auto it = src.padded.find(off);
        if (it == src.padded.end() || it->second.rows != rows || it->second.nin != nin) {
            auto copy = std::make_shared<VulkanBuffer>(dev_, rows * (nin + kF32Pad) * sizeof(float), false);
            std::vector<VkBufferCopy> regions(rows);
            for (size_t r = 0; r < rows; ++r)
                regions[r] = VkBufferCopy{off + r * nin * sizeof(float), r * (nin + kF32Pad) * sizeof(float), nin * sizeof(float)};
            const uint32_t count = u32(regions.size());
            VkCommandBuffer cmd = open();
            if (it != src.padded.end()) pending_[ring_index_].push_back(it->second.copy);
            it = src.padded.insert_or_assign(off, VulkanBuffer::Padded{rows, nin, copy}).first;
            barrier(cmd);
            dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), copy->handle(), count, regions.data());
            barrier(cmd);
        }
        return VkDescriptorBufferInfo{it->second.copy->handle(), 0, VK_WHOLE_SIZE};
    }
    // Copies of a buffer about to change, retired in stream order.
    void drop_padded(VulkanBuffer& b) {
        if (b.padded.empty()) return;
        open();
        auto& pending = pending_[ring_index_];
        // Reserve before moving any cache entry so allocation failure leaves the cache intact.
        pending.reserve(pending.size() + b.padded.size());
        for (auto& e : b.padded) pending.push_back(std::move(e.second.copy));
        b.padded.clear();
    }

    // Split below the profile's workgroup target, retaining at least tile_split_min_blocks per part; per_cu overrides the target.
    size_t split_blocks(size_t workgroups, size_t nblk, uint32_t per_cu = 0) const {
        const bool narrow = nblk * 32 < dev_->profile.tile_narrow_nin;
        const size_t target = (size_t)(per_cu ? per_cu : narrow ? dev_->profile.tile_split_per_cu_narrow : dev_->profile.tile_split_per_cu) *
                              dev_->caps.compute_units;
        const size_t floor_blocks = std::max<size_t>(dev_->profile.tile_split_min_blocks, 2);
        if (workgroups >= target) return nblk;
        size_t parts = std::min((target + workgroups - 1) / workgroups, std::max<size_t>(1, nblk / floor_blocks));
        if (parts <= 1) return nblk;
        size_t kper = (nblk + parts - 1) / parts;
        kper = (kper + 1) & ~size_t(1);   // a split call's parts are a multiple of the tile's STEP of two
        return std::min(kper, nblk);
    }

    // Whether the matmul that reads an nbatch x nin batch next, with these runs, takes one integer-dot tile call over all of it whatever its weights' type, dense or routed.
    // Then the batch's producer writes the twin four values a lane (shaders/xquant.glsl, xquant_word), and the call reads it as it is.
    // That holds when every run's prompt reaches every type's tile threshold and all runs take one split, which is when matmul_runs and expert_runs make a single call; any other batch keeps the twin and the pass.
    // The copy's values are the pass's bit for bit, so which of the two makes it changes no result.
    bool tile_reads(size_t nin, size_t nbatch, RowRuns runs) const {
        const DeviceProfile& p = dev_->profile;
        if (!p.prefer_integer_dot || nin % 32 || !runs.n || runs.runs[runs.n - 1].end != nbatch) return false;
        const size_t from = std::max({tile_from_for(p, true, nin), tile_from_for(p, false, nin), p.moe_tile_from,
                                      p.moe_tile_from_q4, p.moe_tile_from_q4k, p.moe_tile_from_q5k});
        const size_t split = split_tiles_of(runs.runs[0].extent);
        for (size_t i = 0; i < runs.n; ++i)
            if (runs.runs[i].extent < from || split_tiles_of(runs.runs[i].extent) != split) return false;
        return true;
    }

    // A stream-ordered scratch outgrown mid-pass: recorded commands still name the old buffer, so it retires with the ring slot.
    void grow(std::shared_ptr<VulkanBuffer>& buffer, size_t bytes) {
        open();
        if (buffer) pending_[ring_index_].push_back(std::move(buffer));
        buffer = std::make_shared<VulkanBuffer>(dev_, bytes, false);
    }
    KVLayout kv_layout() const override { return KVLayout{kVkBlockTokens}; }

    std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim,
                                        size_t max_tokens, KVType k_type = KVType::f32,
                                        KVType v_type = KVType::f32) override {
        if (!layers || !n_head_kv || !head_dim)
            throw std::runtime_error("vulkan: KV storage without layers, heads or width");
        if (!attention_head_fits(head_dim, dev_->caps.subgroup_size))
            throw std::runtime_error("vulkan: a head " + std::to_string(head_dim) + " wide is not supported on this device, whose attention takes heads 128 or 256 wide or at most " +
                                     std::to_string(attention_row_width(dev_->caps.subgroup_size)));
        return std::make_unique<VulkanKVStorage>(*this, layers, n_head_kv, head_dim, max_tokens, k_type, v_type);
    }

    // One dispatch for every view of the batch through the view table, each view's rows scattering into its blocks.
    void kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k,
                  CSlice v) override {
        std::vector<Placed> placed = place_views(views, n_views);
        ViewTable t = view_table(layer, placed, true);
        if (!t.storage || !t.rows) return;
        VulkanKVStorage& s = *t.storage;
        const size_t hd = s.heads() * s.dim();
        if (floats_from(k) < t.rows * hd || floats_from(v) < t.rows * hd)
            throw std::runtime_error("vulkan: KV rows outside their allocation");
        const uint32_t pc[4] = {u32(t.rows), u32(s.heads()), u32(s.dim()), u32(kVkBlockTokens)};
        dispatch(kv_variant(K_KV_WRITE, K_KV_WRITE_K16, s),
                 {bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                  bind(k), bind(v), args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 pc, sizeof(pc), groups(t.rows * hd, 256));
    }

    // Every view of the batch through the view table: one tiled and one per-row dispatch at most, and a merge when the per-row one splits histories.
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out,
                   int n_head, int n_head_kv, int head_dim) override {
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || !attention_head_fits((size_t)head_dim, dev_->caps.subgroup_size))
            throw std::runtime_error("vulkan: invalid attention dimensions");
        std::vector<Placed> placed = place_views(views, n_views);
        if (placed.empty()) return;
        for (const Placed& pv : placed)
            if (!pv.view->nq) throw std::runtime_error("vulkan: invalid attention dimensions");
        const size_t qstride = (size_t)n_head * head_dim;
        const float scale = 1.0f / std::sqrt((float)head_dim);
        const size_t rows = placed.back().row0 + placed.back().view->nq;
        if (floats_from(Q) < rows * qstride || floats_from(out) < rows * qstride)
            throw std::runtime_error("vulkan: attention rows outside their allocation");
        // Views of 128- or 256-wide heads whose prompt reaches the profile's attention_tile_rows take the tiled kernel, the rest the per-row kernel: at most two dispatches per layer.
        // The choice is by the view's extent rather than its row count, so a prompt's rows take the same kernel however they were batched.
        std::vector<Placed> wide, narrow;
        for (const Placed& pv : placed) {
            const size_t extent = pv.view->extent ? pv.view->extent : pv.view->nq;
            (extent >= dev_->profile.attention_tile_rows && (head_dim == 128 || head_dim == 256) ? wide : narrow).push_back(pv);
        }
        if (!wide.empty()) {
            ViewTable t = view_table(layer, wide, false);
            VulkanKVStorage& s = *t.storage;
            check_storage(s, n_head_kv, head_dim);
            size_t tiles = 0;
            for (const Placed& pv : wide) tiles += (pv.view->nq + kAttentionTileRows - 1) / kAttentionTileRows;
            // With every view here and the integer-dot tile reading the output next, the output's twin written beside it (shaders/attention_tile.comp).
            std::vector<RowRun> vruns;
            for (const Placed& pv : placed) vruns.push_back(RowRun{pv.row0 + pv.view->nq, pv.view->extent});
            const bool tile = narrow.empty() && tile_reads(qstride, rows, RowRuns{vruns.data(), vruns.size()});
            struct { uint32_t n_head, n_head_kv, bt; float scale; uint32_t quant, rows; }
                tc{(uint32_t)n_head, (uint32_t)n_head_kv, u32(kVkBlockTokens), scale, tile ? 1u : 0u, u32(rows)};
            dispatch(head_dim == 256 ? kv_variant(K_ATTENTION_TILE_D256, K_ATTENTION_TILE_D256_K16, s) : kv_variant(K_ATTENTION_TILE, K_ATTENTION_TILE_K16, s),
                     {bind(Q), bind(out), bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                      args(t.words.data(), t.words.size() * sizeof(uint32_t)), tile ? xq_for(rows * qstride) : bind(out)},
                     &tc, sizeof(tc), u32(tiles * (size_t)n_head));
            if (tile) xq_tag_ = XqTag{bind(out), rows * qstride};
        }
        if (!narrow.empty()) {
            ViewTable t = view_table(layer, narrow, false);
            VulkanKVStorage& s = *t.storage;
            check_storage(s, n_head_kv, head_dim);
            // The output's 16-bit twin, written by whichever kernel writes the output, when the whole batch is this dispatch and a head is whole blocks.
            const bool quant = wide.empty() && head_dim % 32 == 0;
            const VkDescriptorBufferInfo xq = quant ? xq_for(rows * qstride) : bind(out);
            // A row's history is split in parts of the profile's chunk across workgroups, the part doubling until at most the profile's maximum cover it; the parts depend only on the row's length, so a row computes the same whatever else is in the dispatch.
            // The dispatch has as many splits as its longest row can need; a row's extra splits are empty.
            size_t longest = 0;
            for (const Placed& pv : narrow)
                longest = std::max(longest, size_add(pv.view->length, pv.view->nq));
            const size_t pairs = t.rows * (size_t)n_head;
            const DeviceProfile& prof = dev_->profile;
            const size_t chunk = prof.attention_split_chunk;
            const size_t split_max = head_dim == 256 ? prof.attention_split_max_wide : prof.attention_split_max;
            const size_t nsplit = std::min(split_max, std::max<size_t>(1, (longest + chunk - 1) / chunk));
            const size_t scratch_floats = nsplit > 1 ? pairs * nsplit * ((size_t)head_dim + 2) : 0;
            if (scratch_floats && (!scratch_ || scratch_->size() < scratch_floats * sizeof(float)))
                grow(scratch_, scratch_floats * sizeof(float));
            // The query heads a workgroup takes: once the longest row's history fills every split, up to four of those sharing a KV head, four, three or two as the group divides, so a token's key and value are loaded once for them (shaders/attention.comp).
            // A shorter history leaves few workgroups, and taking heads together would leave the device idle; each head's arithmetic is the same either way.
            const size_t group = (size_t)(n_head / n_head_kv);
            const bool long_history = longest >= chunk * split_max;
            const uint32_t hg = !long_history ? 1u : group % 4 == 0 ? 4u : group % 3 == 0 ? 3u : group % 2 == 0 ? 2u : 1u;
            struct { uint32_t rows, n_head, n_head_kv, dim, bt; float scale; uint32_t nsplit, chunk, quant, max_parts, hg; }
                pc{u32(t.rows), (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)head_dim, u32(kVkBlockTokens),
                   scale, u32(nsplit), u32(chunk), quant ? 1u : 0u, u32(split_max), hg};
            const VkDescriptorBufferInfo scratch = scratch_
                ? VkDescriptorBufferInfo{scratch_->handle(), 0, VK_WHOLE_SIZE} : bind(out);
            const VkDescriptorBufferInfo table = args(t.words.data(), t.words.size() * sizeof(uint32_t));
            // Heads 128 wide take the kernel whose 16 lanes read a token's row in one load each, several tokens a subgroup, and heads 256 wide its build of 32 lanes a token (shaders/attention_vec.comp).
            const KernelId kernel = head_dim == 128 ? (hg > 1 ? kv_variant(K_ATTENTION_VEC_G4, K_ATTENTION_VEC_K16_G4, s) : kv_variant(K_ATTENTION_VEC, K_ATTENTION_VEC_K16, s))
                                  : head_dim == 256 ? (hg > 1 ? kv_variant(K_ATTENTION_VEC_D256_G4, K_ATTENTION_VEC_D256_K16_G4, s)
                                                              : kv_variant(K_ATTENTION_VEC_D256, K_ATTENTION_VEC_D256_K16, s))
                                  : (hg > 1 ? kv_variant(K_ATTENTION_G4, K_ATTENTION_K16_G4, s) : kv_variant(K_ATTENTION, K_ATTENTION_K16, s));
            dispatch(kernel,
                     {bind(Q), bind(out), bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                      table, scratch, xq},
                     &pc, sizeof(pc), groups(pairs / hg * nsplit, 1), 1);
            if (nsplit > 1) {
                const uint32_t mc[5] = {u32(t.rows), (uint32_t)n_head, (uint32_t)head_dim, u32(nsplit), quant ? 1u : 0u};
                dispatch(K_ATTENTION_MERGE, {bind(out), scratch, table, xq}, mc, sizeof(mc), groups(pairs, 1), 1);
            }
            if (quant) xq_tag_ = XqTag{bind(out), rows * qstride};
        }
    }

    // A view with the batch row its rows start at.
    struct Placed {
        const KVView* view;
        size_t row0;
    };
    static std::vector<Placed> place_views(const KVView* views, size_t n_views) {
        if (n_views && !views) throw std::runtime_error("vulkan: cache op without views");
        std::vector<Placed> placed;
        size_t row0 = 0;
        for (size_t i = 0; i < n_views; ++i) {
            placed.push_back(Placed{&views[i], row0});
            row0 += views[i].nq;
        }
        return placed;
    }
    // The table the batched cache kernels read (shaders/views.glsl) for a subset of a batch's views.
    // Every view is checked against the storage by BlockKVStorage::check_view, which backs the blocks a writing op's rows land in and requires a reading op's written.
    struct ViewTable {
        std::vector<uint32_t> words;
        size_t rows = 0;
        VulkanKVStorage* storage = nullptr;
    };
    ViewTable view_table(size_t layer, const std::vector<Placed>& placed, bool writing) {
        ViewTable t;
        t.words.push_back(u32(placed.size()));
        std::vector<uint32_t> blocks;
        size_t local = 0;
        for (const Placed& pv : placed) {
            const KVView& view = *pv.view;
            VulkanKVStorage& s = storage_of(view.storage);
            if (t.storage && t.storage != &s) throw std::runtime_error("vulkan: views of two storages in one call");
            t.storage = &s;
            const size_t used = s.check_view(view, layer, writing);
            t.words.push_back(u32(pv.row0));
            t.words.push_back(u32(local));
            t.words.push_back(u32(view.nq));
            t.words.push_back(u32(view.length));
            t.words.push_back(u32(blocks.size()));
            t.words.push_back(u32(used));
            for (size_t b = 0; b < used; ++b) blocks.push_back((uint32_t)view.blocks[b]);
            local += view.nq;
        }
        t.words.insert(t.words.end(), blocks.begin(), blocks.end());
        t.rows = local;
        return t;
    }
    // The heads and width attention is asked for against the storage's; the view table has checked the layer.
    static void check_storage(const VulkanKVStorage& s, int n_head_kv, int head_dim) {
        if ((size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads())
            throw std::runtime_error("vulkan: attention outside the KV view");
    }

    // Buffers a storage growth copy reads from, kept until the command buffer that recorded the copy retires.
    void keep_until_retired(BufferPtr b) {
        open();
        pending_[ring_index_].push_back(std::dynamic_pointer_cast<VulkanBuffer>(b));
    }

private:
    static constexpr uint32_t kRing = 16;
    uint32_t chunk_ = 0;
    bool unordered_ = false;   // barriers left out (unordered)
    static constexpr size_t kStagingBytes = size_t(64) << 20;
    static constexpr size_t kArenaBytes = size_t(1) << 20;
    static constexpr uint32_t kPushBytes = 128;

    struct Arena {
        std::unique_ptr<VulkanBuffer> buffer;
        size_t used = 0;
    };

    static VulkanKVStorage& storage_of(KVStorage* storage) {
        auto* s = dynamic_cast<VulkanKVStorage*>(storage);
        if (!s) throw std::runtime_error("vulkan: KV storage of another backend");
        return *s;
    }

    static uint32_t u32(size_t v) {
        if (v > 0xffffffffu) throw std::runtime_error("vulkan: dimension exceeds 32 bits");
        return (uint32_t)v;
    }

    uint32_t groups(size_t n, size_t per_group) const {
        const size_t g = (n + per_group - 1) / per_group;
        if (g > dev_->props.limits.maxComputeWorkGroupCount[0])
            throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        return (uint32_t)g;
    }

    // Bytes, and whole floats, from a slice's offset to the end of its buffer.
    static size_t bytes_from(CSlice s) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        const size_t bytes = s.buffer->size();
        if (s.offset * sizeof(float) > bytes) throw std::runtime_error("vulkan: operand outside the allocation");
        return bytes - s.offset * sizeof(float);
    }
    static size_t floats_from(CSlice s) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        const size_t bytes = s.buffer->size();
        if (s.offset * sizeof(float) > bytes) throw std::runtime_error("vulkan: operand outside the allocation");
        return (bytes - s.offset * sizeof(float)) / sizeof(float);
    }

    // A slice as a storage buffer binding: the buffer at a byte offset of four times the float offset, through to the end of the allocation.
    VkDescriptorBufferInfo bind(CSlice s) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        const VulkanBuffer& b = as_vulkan(*s.buffer);
        const VkDeviceSize off = (VkDeviceSize)s.offset * sizeof(float);
        if (off > b.size()) throw std::runtime_error("vulkan: operand outside the allocation");
        if (!b.size()) return VkDescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        return VkDescriptorBufferInfo{b.handle(), off, VK_WHOLE_SIZE};
    }
    VkDescriptorBufferInfo bind(Slice s) { return bind(CSlice(s)); }

    // Small per-call host inputs (ids, positions, row lists) go through a host-visible arena per ring slot, bumped per call and reset when the slot retires; a call larger than the arena gets its own buffer, kept the same way.
    VkDescriptorBufferInfo args(const void* data, size_t bytes) {
        open();
        Arena& a = arena_[ring_index_];
        const size_t need = (bytes + 15) / 16 * 16;
        if (need > kArenaBytes) {
            auto b = std::make_shared<VulkanBuffer>(dev_, bytes, true);
            std::memcpy(b->mapped(), data, bytes);
            pending_[ring_index_].push_back(b);
            return VkDescriptorBufferInfo{b->handle(), 0, VK_WHOLE_SIZE};
        }
        if (!a.buffer) a.buffer = std::make_unique<VulkanBuffer>(dev_, kArenaBytes, true);
        if (a.used + need > kArenaBytes) {
            // The slot's arena is full before its command buffer retired: the overflow gets a second arena kept alongside.
            pending_[ring_index_].emplace_back(std::move(a.buffer));
            a.used = 0;
            a.buffer = std::make_unique<VulkanBuffer>(dev_, kArenaBytes, true);
        }
        std::memcpy((uint8_t*)a.buffer->mapped() + a.used, data, bytes);
        const VkDescriptorBufferInfo info{a.buffer->handle(), a.used, bytes};
        a.used += need;
        return info;
    }

    void destroy_kernel(Kernel& k) noexcept {
        Device& d = *dev_;
        if (k.pipeline) d.fn.vkDestroyPipeline(d.device, k.pipeline, nullptr);
        if (k.layout) d.fn.vkDestroyPipelineLayout(d.device, k.layout, nullptr);
        if (k.set_layout) d.fn.vkDestroyDescriptorSetLayout(d.device, k.set_layout, nullptr);
        if (k.module) d.fn.vkDestroyShaderModule(d.device, k.module, nullptr);
        k = Kernel{};
    }

    Kernel& kernel(KernelId id, int variant = 0) {
        Kernel& cached = kernels_[id][variant];
        if (cached.pipeline) return cached;
        Kernel k;
        try {
            Device& d = *dev_;
            const KernelSource& src = kKernels[id];
            const bool preserve = d.preserve_float32 && src.preserved_words;
            VkShaderModuleCreateInfo mi{};
            mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            mi.codeSize = preserve ? src.preserved_bytes : src.bytes;
            mi.pCode = preserve ? src.preserved_words : src.words;
            VkShaderModule module = VK_NULL_HANDLE;
            check(d.fn.vkCreateShaderModule(d.device, &mi, nullptr, &module), "vkCreateShaderModule");
            k.module = module;
            std::vector<VkDescriptorSetLayoutBinding> bindings(src.bindings);
            k.buffers = 0;
            for (uint32_t i = 0; i < src.bindings; ++i) {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                bindings[i].descriptorCount = src.counts ? src.counts[i] : 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
                k.buffers += bindings[i].descriptorCount;
            }
            VkDescriptorSetLayoutCreateInfo li{};
            li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
            li.bindingCount = src.bindings;
            li.pBindings = bindings.data();
            VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
            check(d.fn.vkCreateDescriptorSetLayout(d.device, &li, nullptr, &set_layout),
                  "vkCreateDescriptorSetLayout");
            k.set_layout = set_layout;
            VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
            VkPipelineLayoutCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pi.setLayoutCount = 1;
            pi.pSetLayouts = &k.set_layout;
            pi.pushConstantRangeCount = 1;
            pi.pPushConstantRanges = &range;
            VkPipelineLayout layout = VK_NULL_HANDLE;
            check(d.fn.vkCreatePipelineLayout(d.device, &pi, nullptr, &layout), "vkCreatePipelineLayout");
            k.layout = layout;
            VkComputePipelineCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = k.module;
            ci.stage.pName = "main";
            // The tile kernels take their row count as specialization constant 0 and the row kernels their column count.
            const bool tile = is_tile_kernel(id);
            const bool tall_tile = id == K_MATMUL_TILE_WF16_TALL || id == K_MATMUL_TILE_WF16_BF16_TALL ||
                                   id == K_MATMUL_TILE_WBF16_TALL || id == K_MATMUL_TILE_WBF16_BF16_TALL || id == K_MATMUL_TILE_Q8MX_TALL || id == K_MATMUL_TILE_MXFP4_TALL || id == K_MATMUL_TILE_MXFP4_BF16_TALL || id == K_MATMUL_TILE_TALL || id == K_MATMUL_TILE_BF16_TALL || id == K_MATMUL_TILE_Q_TALL || id == K_MATMUL_TILE_Q8_TALL;
            const uint32_t spec_value = tile ? (tall_tile ? kTileRowsTall : variant == 1 ? kTileRowsSmall : kTileRowsShort)
                                             : build_cols(id, variant);
            // Constant 8 a row kernel's grouped build, and constant 9 the rows a build takes, a cluster's in matmul_row.comp (build_rows) and a subgroup's in the Q8_0 decode kernel.
            // Constants 10 to 13 are the Q8_0 decode kernel's steps, forms and column groups (kVecBuilds), whose constant 0 is its columns a subgroup rather than the build's.
            // Every pipeline gets all seven entries, and a module that declares none ignores them.
            const VecBuild& vb = kVecBuilds[variant];
            const bool vec = id == K_MATMUL_VEC_Q8;
            const uint32_t forms = vec ? vec_forms(variant, d.profile) : 0;
            const uint32_t spec_data[7] = {vec ? vb.cols : spec_value, variant == 2 ? 1u : 0u, vec ? vb.rows : build_rows(id, variant), vb.steps,
                                           (forms & kQ8Tree) ? 1u : 0u, (forms & kQ8Half) ? 1u : 0u, vb.span};
            const uint32_t spec_ids[7] = {0, 8, 9, 10, 11, 12, 13};
            VkSpecializationMapEntry entries[7];
            for (uint32_t i = 0; i < 7; ++i) entries[i] = {spec_ids[i], i * uint32_t(sizeof(uint32_t)), sizeof(uint32_t)};
            VkSpecializationInfo spec{};
            spec.mapEntryCount = 7;
            spec.pMapEntries = entries;
            spec.dataSize = sizeof(spec_data);
            spec.pData = spec_data;
            ci.stage.pSpecializationInfo = &spec;
            ci.layout = k.layout;
            if (d.exec_stats) ci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
            if (d.exec_ir) ci.flags |= VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
            check(d.fn.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &ci, nullptr, &k.pipeline),
                  "vkCreateComputePipelines");
            k.bindings = src.bindings;
            cached = k;
            return cached;
        } catch (...) {
            destroy_kernel(k);
            throw;
        }
    }

    // One dispatch: bind the pipeline, push the buffers and constants, launch the workgroups, and fence it off from the next command.
    void dispatch(KernelId id, std::initializer_list<VkDescriptorBufferInfo> buffers,
                  const void* push, size_t push_bytes, uint32_t groups_x, uint32_t groups_y = 1,
                  int variant = 0) {
        Kernel& k = kernel(id, variant);
        if (buffers.size() != k.buffers) throw std::logic_error("vulkan: kernel binding count");
        // Routing and grouping write only ids, weights and their own table; route_experts checks those against the twin itself.
        // A tile call, such as a router's before its experts, writes only its outputs, which the call checks against the twin itself.
        if (!is_row_kernel(id) && id != K_QUANTIZE_X && id != K_QUANTIZE_XW && id != K_RMS_NORM_ROWS && id != K_SILU_MUL &&
            id != K_MOE_ROUTE && id != K_MOE_GROUP && id != K_MATMUL_REDUCE && id != K_MATMUL_REDUCE_MXFP4 && id != K_COPY_MXFP4 && !is_tile_kernel(id))
            xq_tag_ = XqTag{};
        if (push_bytes > kPushBytes) throw std::logic_error("vulkan: push constants exceed 128 bytes");
        for (const auto& b : buffers)
            if (!b.buffer) throw std::runtime_error("vulkan: dispatch over an empty allocation");
        VkCommandBuffer cmd = open();
        const KernelSource& src = kKernels[id];
        std::vector<VkWriteDescriptorSet> writes(k.bindings);
        const VkDescriptorBufferInfo* next = buffers.begin();
        for (uint32_t i = 0; i < k.bindings; ++i) {
            VkWriteDescriptorSet& w = writes[i];
            w = VkWriteDescriptorSet{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstBinding = i;
            w.descriptorCount = src.counts ? src.counts[i] : 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = next;
            next += w.descriptorCount;
        }
        dev_->fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
        dev_->fn.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0,
                                           (uint32_t)writes.size(), writes.data());
        dev_->fn.vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    (uint32_t)push_bytes, push);
        if (dev_->timestamps && query_next_ + 2 <= kQueries) {
            if (!queries_) {
                VkQueryPoolCreateInfo qp{};
                qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qp.queryCount = kQueries;
                VkQueryPool queries = VK_NULL_HANDLE;
                check(dev_->fn.vkCreateQueryPool(dev_->device, &qp, nullptr, &queries), "vkCreateQueryPool");
                queries_ = queries;
                queries_stale_ = true;
            }
            if (queries_stale_) {
                dev_->fn.vkCmdResetQueryPool(cmd, queries_, 0, kQueries);
                queries_stale_ = false;
            }
            dev_->fn.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, query_next_);
            dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
            dev_->fn.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, query_next_ + 1);
            query_kernel_.push_back(id * kVariants + variant);
            query_next_ += 2;
        } else {
            dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
        }
        if (!unordered_) barrier(cmd);
        // A pass is submitted in chunks so the device starts on the first while the host records the rest; the ordered timeline makes the last chunk's ticket cover them all.
        if (++chunk_ >= dev_->profile.dispatch_chunk) flush();
    }

    // The open command buffer, beginning the next ring slot once its last submission has retired.
    VkCommandBuffer open() {
        if (open_) return ring_[ring_index_];
        waited(host_.slot_ms, ring_ticket_[ring_index_]);
        pending_[ring_index_].clear();
        arena_[ring_index_].used = 0;
        VkCommandBuffer cmd = ring_[ring_index_];
        check(dev_->fn.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(dev_->fn.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
        open_ = true;
        return cmd;
    }

    // Order compute and transfer writes before subsequent compute and transfer accesses.
    void barrier(VkCommandBuffer cmd) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                           VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        const VkPipelineStageFlags stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        dev_->fn.vkCmdPipelineBarrier(cmd, stages, stages, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    VulkanBuffer& staging() {
        if (!staging_) staging_ = std::make_unique<VulkanBuffer>(dev_, kStagingBytes, true);
        return *staging_;
    }

    // Host to device through the two halves of staging: the host fills one while the device copies from the other, so a long upload runs at the slower of the two rather than at their sum.
    // Consecutive uploads carry on from the half the last one left, so a weight written in pieces overlaps as one long upload does.
    // It returns once the source is consumed; the copies are in stream order, ahead of whatever reads the destination.
    // `keep`, a buffer being adopted, is held by every slot that copies into it, so its caller may drop it before the copies retire.
    void upload(VulkanBuffer& dst, size_t off, const void* src, size_t bytes, const std::shared_ptr<VulkanBuffer>& keep = nullptr) {
        if (!bytes) return;
        VulkanBuffer& st = staging();
        const size_t half = st.size() / 2;
        size_t done = 0;
        for (size_t& i = next_half_; done < bytes; i ^= 1) {
            const size_t n = std::min(bytes - done, half);
            waited(host_.staging_ms, staged_[i]);
            std::memcpy((uint8_t*)st.mapped() + i * half, (const uint8_t*)src + done, n);
            VkCommandBuffer cmd = open();
            if (keep) pending_[ring_index_].push_back(keep);
            barrier(cmd);
            VkBufferCopy region{i * half, off + done, n};
            dev_->fn.vkCmdCopyBuffer(cmd, st.handle(), dst.handle(), 1, &region);
            barrier(cmd);
            staged_[i] = flush();
            done += n;
        }
    }

    // A wait on ticket t, its time added to `into` on a backend made to time its work.
    void waited(double& into, Ticket t) noexcept {
        if (!timed_) return wait_for(t);
        const auto t0 = std::chrono::steady_clock::now();
        wait_for(t);
        into += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }

    std::shared_ptr<Device> dev_;
    const bool timed_;       // made for diagnostics, so its waits and uploads are timed (host_times)
    HostTimes host_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer ring_[kRing] = {};
    Ticket ring_ticket_[kRing] = {};
    uint32_t ring_index_ = 0;
    bool open_ = false;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    Ticket last_ticket_ = 0;
    static constexpr uint32_t kHolds = 4;       // hold submissions in flight; one is pending at a time
    static constexpr int kHoldMs = 100;         // the longest a held queue waits for its host
    struct {
        bool on = false;                        // some holder remains
        size_t holders = 0;
        VkCommandBuffer cmds[kHolds] = {};
        VkEvent events[kHolds] = {};
        VkSemaphore timeline = VK_NULL_HANDLE;  // signalled as each hold ends
        uint64_t value[kHolds] = {};
        uint64_t last = 0;
        uint32_t slot = 0;
        bool pending = false;                   // the slot's event is unset and its hold queued
        bool quit = false;
        std::chrono::steady_clock::time_point since;
        std::mutex mu;                          // the pending hold and its event, shared with the watchdog
        std::condition_variable wake;
        std::thread watchdog;
    } hold_;
    std::unique_ptr<VulkanBuffer> staging_;
    Ticket staged_[2] = {};                   // the last copy out of each half of staging
    size_t next_half_ = 0;                    // the half of staging the next upload fills first
    std::shared_ptr<VulkanBuffer> scratch_;   // attention split states; stream-ordered reuse
    VkQueryPool queries_ = VK_NULL_HANDLE;    // timestamps, only for a diagnostics backend
    static constexpr uint32_t kQueries = 8192;    // two per dispatch; a reading empties the pool, which the next dispatch resets
    uint32_t query_next_ = 0;
    bool queries_stale_ = false;   // the pool holds a read interval's stamps, reset by the next dispatch
    size_t last_timed_ = 0;        // dispatches the last reading covered
    std::vector<int> query_kernel_;   // id * kVariants + variant
    double kernel_ns_[K_COUNT * kVariants] = {0};
    size_t kernel_calls_[K_COUNT * kVariants] = {0};
    std::shared_ptr<VulkanBuffer> mx_copy_; // copies of one call, reused in queue order
    std::shared_ptr<VulkanBuffer> parts_;     // a split integer-dot tile call's partial sums
    std::shared_ptr<VulkanBuffer> xq_;        // the row kernel's quantized activations; likewise
    std::shared_ptr<VulkanBuffer> moe_out_;   // a routed down projection's slots before they are combined
    std::shared_ptr<VulkanBuffer> moe_tab_;   // a routed tile call's grouping (shaders/moe_group.comp)
    // Which ids moe_tab_ groups: their location and count, cleared by every routing, by anything that writes a buffer from the host and by a new buffer (drop_tags).
    struct GroupTag { VkDescriptorBufferInfo ids{}; size_t entries = 0, n_expert = 0, chunk = 0; };
    GroupTag group_tag_;
    // What the twin buffer holds: the float input it was made from, its length; cleared by anything else that writes a buffer, since the input may be what was written, and by a new buffer (drop_tags).
    struct XqTag { VkDescriptorBufferInfo x{}; size_t n = 0; };

    XqTag xq_tag_;
    std::vector<std::shared_ptr<VulkanBuffer>> pending_[kRing];
    Arena arena_[kRing];
    Kernel kernels_[K_COUNT][kVariants];
};

inline KernelId kv_variant(KernelId f32, KernelId k16, const VulkanKVStorage& s) {
    const int i = (s.k_type() == KVType::f16 ? 1 : 0) + (s.v_type() == KVType::f16 ? 2 : 0);
    return i == 0 ? f32 : (KernelId)((int)k16 + i - 1);
}

VulkanKVStorage::VulkanKVStorage(VulkanBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens,
                                 KVType kt, KVType vt)
    : BlockKVStorage(owner, "vulkan", kVkBlockTokens, layers, heads, dim, max_tokens, kt, vt) {}

void VulkanKVStorage::retire(const BufferPtr& old) {
    static_cast<VulkanBackend&>(owner()).keep_until_retired(old);
}

} // namespace

BackendPtr make_vulkan_backend(int device, bool diagnostics) {
    return std::make_shared<VulkanBackend>(device, diagnostics);
}

std::string vulkan_device_name(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->name() : std::string();
}

DeviceProfile vulkan_device_profile(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->profile() : DeviceProfile{};
}

std::string vulkan_kernel_statistics(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->kernel_statistics() : std::string();
}

std::vector<std::pair<std::string, double>> vulkan_kernel_times(Backend& backend) {
    auto* v = dynamic_cast<VulkanBackend*>(&backend);
    return v ? v->kernel_times() : std::vector<std::pair<std::string, double>>();
}

size_t vulkan_timed_dispatches(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->timed_dispatches() : 0;
}

std::vector<std::pair<std::string, std::string>> vulkan_kernel_representations(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->kernel_representations() : std::vector<std::pair<std::string, std::string>>();
}

} // namespace backend
