// Vulkan backend (docs/VULKAN.md). Everything runs on one compute queue: ops record into an open command buffer, submit() ends it and signals a timeline semaphore with the ticket, wait() blocks on it, and a ring of command buffers is reused once their tickets retire.
#include "backends/device_profile.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "format/gguf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkResetCommandBuffer) \
    X(vkQueueSubmit) \
    X(vkCreateSemaphore) \
    X(vkDestroySemaphore) \
    X(vkWaitSemaphores) \
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
const uint32_t kSpvNormRopeRows[] = {
#include "vulkan/norm_rope_rows.inc"
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
const uint32_t kSpvMatmulRowKDot8[] = {
#include "vulkan/matmul_row_k_dot8.inc"
};
const uint32_t kSpvMatmulRowQ4Dot8[] = {
#include "vulkan/matmul_row_q4_dot8.inc"
};

const uint32_t kSpvMatmulTile[] = {
#include "vulkan/matmul_tile.inc"
};
const uint32_t kSpvQuantizeX8[] = {
#include "vulkan/quantize_x8.inc"
};
const uint32_t kSpvMatmulTileQ[] = {
#include "vulkan/matmul_tile_q.inc"
};
const uint32_t kSpvMatmulTileQ6[] = {
#include "vulkan/matmul_tile_q6.inc"
};
const uint32_t kSpvMatmulReduce[] = {
#include "vulkan/matmul_reduce.inc"
};
const uint32_t kSpvMatmulVecQ8[] = {
#include "vulkan/matmul_vec_q8.inc"
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

enum KernelId { K_ADD, K_SILU_MUL, K_GATHER_ROWS, K_RMS_NORM_ROWS, K_NORM_ROPE_ROWS, K_EMBED,
                K_MATMUL_ROW, K_KV_WRITE, K_ATTENTION, K_ATTENTION_MERGE, K_MATMUL_TILE, K_MATMUL_ROW_Q4,
                K_MATMUL_ROW_K4, K_MATMUL_ROW_K5, K_MATMUL_ROW_K, K_NORM_ROPE_KV, K_ATTENTION_TILE,
                K_KV_WRITE_K16, K_KV_WRITE_V16, K_KV_WRITE_KV16,
                K_ATTENTION_K16, K_ATTENTION_V16, K_ATTENTION_KV16,
                K_ATTENTION_TILE_K16, K_ATTENTION_TILE_V16, K_ATTENTION_TILE_KV16,
                K_NORM_ROPE_KV_K16, K_NORM_ROPE_KV_V16, K_NORM_ROPE_KV_KV16,
                K_QUANTIZE_X, K_MATMUL_ROW_Q8W, K_MATMUL_TILE_TALL,
                K_MATMUL_ROW_Q4_DOT,
                K_MATMUL_ROW_K4_DOT, K_MATMUL_ROW_K5_DOT, K_MATMUL_ROW_K_DOT,
                K_QUANTIZE_X8, K_MATMUL_TILE_Q, K_MATMUL_TILE_Q_TALL, K_MATMUL_TILE_Q6, K_MATMUL_TILE_Q6_TALL,
                K_MATMUL_REDUCE, K_MATMUL_VEC_Q8, K_MOE_ROUTE, K_MOE_COMBINE, K_MOE_GROUP, K_MATMUL_ROW_K_DOT8, K_MATMUL_ROW_Q4_DOT8, K_COUNT };

// The same row kernel in its two dot forms; which one a device wants is measured (backends/device_profile.hpp).
// F32 rows have no dot form, and Q8_0 rows take matmul_vec_q8.comp where the dot is preferred.
inline KernelId row_dot_variant(KernelId plain) {
    switch (plain) {
    case K_MATMUL_ROW_Q4: return K_MATMUL_ROW_Q4_DOT;
    case K_MATMUL_ROW_K4: return K_MATMUL_ROW_K4_DOT;
    case K_MATMUL_ROW_K5: return K_MATMUL_ROW_K5_DOT;
    case K_MATMUL_ROW_K: return K_MATMUL_ROW_K_DOT;
    default: return plain;
    }
}

// Whether a kernel id is a row kernel: they share the activation twin and take the column count as specialization constant 0.
inline bool is_row_kernel(KernelId id) {
    switch (id) {
    case K_MATMUL_ROW: case K_MATMUL_ROW_Q8W: case K_MATMUL_ROW_Q4:
    case K_MATMUL_ROW_K4: case K_MATMUL_ROW_K5: case K_MATMUL_ROW_K:
    case K_MATMUL_ROW_Q4_DOT:
    case K_MATMUL_ROW_K4_DOT: case K_MATMUL_ROW_K5_DOT: case K_MATMUL_ROW_K_DOT: case K_MATMUL_VEC_Q8: case K_MATMUL_ROW_K_DOT8: case K_MATMUL_ROW_Q4_DOT8:
        return true;
    default: return false;
    }
}

// Whether a row kernel has a one-column build for one-column chunks, which frees the registers of seven unused accumulators.
// Not the wide Q8_0 path: it already runs five waves per SIMD, and its one-column build measured slower on the 8B file (docs/VULKAN.md).
inline bool row_kernel_builds_one_column(KernelId id) {
    return is_row_kernel(id) && id != K_MATMUL_ROW_Q8W;
}

const uint32_t kRowColsWide = 8, kRowColsOne = 1;
const int kVariants = 2;   // a kernel's pipelines: the wide build, then the one-column

// The tile kernel's row count, specialization constant 0: the shorter heights fill a device a taller tile would leave idle, the taller reads less shared memory per product.
const uint32_t kTileRowsSmall = 32, kTileRowsShort = 64, kTileRowsTall = 128;   // the small height is variant 1 of the short kernels

// A kernel's bindings; `counts` gives each one's array length, one for a plain buffer. A dispatch lists its buffers binding by binding, array elements consecutively.
struct KernelSource {
    const uint32_t* words;
    size_t bytes;
    uint32_t bindings;
    const uint32_t* counts;
};

const uint32_t kMatmulRowCounts[12] = {3, 3, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1};
// The integer-dot tile's outputs and weights, three of each, and the reduce's outputs.
const uint32_t kMatmulTileQCounts[5] = {3, 3, 1, 1, 1};
const uint32_t kMatmulReduceCounts[2] = {3, 1};

const char* const kKernelNames[K_COUNT] = {
    "add", "silu_mul", "gather_rows", "rms_norm_rows", "norm_rope_rows", "embed",
    "matmul_row", "kv_write", "attention", "attention_merge", "matmul_tile", "matmul_row_q4",
    "matmul_row_k4", "matmul_row_k5", "matmul_row_k", "norm_rope_kv", "attention_tile",
    "kv_write_k16", "kv_write_v16", "kv_write_kv16",
    "attention_k16", "attention_v16", "attention_kv16",
    "attention_tile_k16", "attention_tile_v16", "attention_tile_kv16",
    "norm_rope_kv_k16", "norm_rope_kv_v16", "norm_rope_kv_kv16",
    "quantize_x", "matmul_row_q8w", "matmul_tile_tall",
    "matmul_row_q4_dot",
    "matmul_row_k4_dot", "matmul_row_k5_dot", "matmul_row_k_dot",
    "quantize_x8", "matmul_tile_q", "matmul_tile_q_tall", "matmul_tile_q6", "matmul_tile_q6_tall",
    "matmul_reduce", "matmul_vec_q8", "moe_route", "moe_combine", "moe_group", "matmul_row_k_dot8", "matmul_row_q4_dot8",
};

const KernelSource kKernels[K_COUNT] = {
    {kSpvAdd, sizeof(kSpvAdd), 2, nullptr},
    {kSpvSiluMul, sizeof(kSpvSiluMul), 4, nullptr},
    {kSpvGatherRows, sizeof(kSpvGatherRows), 3, nullptr},
    {kSpvRmsNormRows, sizeof(kSpvRmsNormRows), 4, nullptr},
    {kSpvNormRopeRows, sizeof(kSpvNormRopeRows), 5, nullptr},
    {kSpvEmbed, sizeof(kSpvEmbed), 4, nullptr},
    {kSpvMatmulRow, sizeof(kSpvMatmulRow), 12, kMatmulRowCounts},
    {kSpvKvWrite, sizeof(kSpvKvWrite), 5, nullptr},
    {kSpvAttention, sizeof(kSpvAttention), 7, nullptr},
    {kSpvAttentionMerge, sizeof(kSpvAttentionMerge), 4, nullptr},
    {kSpvMatmulTile, sizeof(kSpvMatmulTile), 6, nullptr},
    {kSpvMatmulRowQ4, sizeof(kSpvMatmulRowQ4), 12, kMatmulRowCounts},
    {kSpvMatmulRowK4, sizeof(kSpvMatmulRowK4), 12, kMatmulRowCounts},
    {kSpvMatmulRowK5, sizeof(kSpvMatmulRowK5), 12, kMatmulRowCounts},
    {kSpvMatmulRowK, sizeof(kSpvMatmulRowK), 12, kMatmulRowCounts},
    {kSpvNormRopeKv, sizeof(kSpvNormRopeKv), 11, nullptr},
    {kSpvAttentionTile, sizeof(kSpvAttentionTile), 5, nullptr},
    {kSpvKvWriteK16, sizeof(kSpvKvWriteK16), 5, nullptr},
    {kSpvKvWriteV16, sizeof(kSpvKvWriteV16), 5, nullptr},
    {kSpvKvWriteKV16, sizeof(kSpvKvWriteKV16), 5, nullptr},
    {kSpvAttentionK16, sizeof(kSpvAttentionK16), 7, nullptr},
    {kSpvAttentionV16, sizeof(kSpvAttentionV16), 7, nullptr},
    {kSpvAttentionKV16, sizeof(kSpvAttentionKV16), 7, nullptr},
    {kSpvAttentionTileK16, sizeof(kSpvAttentionTileK16), 5, nullptr},
    {kSpvAttentionTileV16, sizeof(kSpvAttentionTileV16), 5, nullptr},
    {kSpvAttentionTileKV16, sizeof(kSpvAttentionTileKV16), 5, nullptr},
    {kSpvNormRopeKvK16, sizeof(kSpvNormRopeKvK16), 11, nullptr},
    {kSpvNormRopeKvV16, sizeof(kSpvNormRopeKvV16), 11, nullptr},
    {kSpvNormRopeKvKV16, sizeof(kSpvNormRopeKvKV16), 11, nullptr},
    {kSpvQuantizeX, sizeof(kSpvQuantizeX), 2, nullptr},
    {kSpvMatmulRowQ8W, sizeof(kSpvMatmulRowQ8W), 12, kMatmulRowCounts},
    {kSpvMatmulTile, sizeof(kSpvMatmulTile), 6, nullptr},
    {kSpvMatmulRowQ4Dot, sizeof(kSpvMatmulRowQ4Dot), 12, kMatmulRowCounts},
    {kSpvMatmulRowK4Dot, sizeof(kSpvMatmulRowK4Dot), 12, kMatmulRowCounts},
    {kSpvMatmulRowK5Dot, sizeof(kSpvMatmulRowK5Dot), 12, kMatmulRowCounts},
    {kSpvMatmulRowKDot, sizeof(kSpvMatmulRowKDot), 12, kMatmulRowCounts},
    {kSpvQuantizeX8, sizeof(kSpvQuantizeX8), 2, nullptr},
    {kSpvMatmulTileQ, sizeof(kSpvMatmulTileQ), 5, kMatmulTileQCounts},
    {kSpvMatmulTileQ, sizeof(kSpvMatmulTileQ), 5, kMatmulTileQCounts},
    {kSpvMatmulTileQ6, sizeof(kSpvMatmulTileQ6), 5, kMatmulTileQCounts},
    {kSpvMatmulTileQ6, sizeof(kSpvMatmulTileQ6), 5, kMatmulTileQCounts},
    {kSpvMatmulReduce, sizeof(kSpvMatmulReduce), 2, kMatmulReduceCounts},
    {kSpvMatmulVecQ8, sizeof(kSpvMatmulVecQ8), 12, kMatmulRowCounts},
    {kSpvMoeRoute, sizeof(kSpvMoeRoute), 3, nullptr},
    {kSpvMoeCombine, sizeof(kSpvMoeCombine), 3, nullptr},
    {kSpvMoeGroup, sizeof(kSpvMoeGroup), 2, nullptr},
    {kSpvMatmulRowKDot8, sizeof(kSpvMatmulRowKDot8), 12, kMatmulRowCounts},
    {kSpvMatmulRowQ4Dot8, sizeof(kSpvMatmulRowQ4Dot8), 12, kMatmulRowCounts},
};

// The variant of a cache kernel for a storage's K and V types.
class VulkanKVStorage;
inline KernelId kv_variant(KernelId f32, KernelId k16, const VulkanKVStorage& s);

// 64 tokens per KV block, half the CPU's: the attention workgroup reads a block per iteration, and a smaller block wastes less tail per sequence.
const size_t kVkBlockTokens = 64;

class VulkanBackend;

// KV blocks on the device, one K and one V buffer per layer; block b starts at b*block_floats() and holds [kv_head][token][head_dim].
// Blocks are backed in doubling steps as ids are first written; growth allocates and copies on the queue, keeping the old buffers alive until the copy retires.
class VulkanKVStorage final : public KVStorage {
public:
    VulkanKVStorage(VulkanBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_blocks,
                    KVType kt, KVType vt)
        : owner_(&owner), heads_(heads), dim_(dim), max_(max_blocks), kt_(kt), vt_(vt), k_(layers), v_(layers) {
        mul(mul(heads, kVkBlockTokens), dim);
    }
    static size_t mul(size_t a, size_t b) {
        if (a && b > std::numeric_limits<size_t>::max() / a)
            throw std::runtime_error("vulkan: KV storage size overflows");
        return a * b;
    }
    static size_t add(size_t a, size_t b) {
        if (b > std::numeric_limits<size_t>::max() - a)
            throw std::runtime_error("vulkan: KV storage size overflows");
        return a + b;
    }
    static size_t blocks_for(size_t tokens) {
        return tokens / kVkBlockTokens + (tokens % kVkBlockTokens != 0);
    }
    size_t max_blocks() const override { return max_; }
    size_t allocated_bytes() const override {
        size_t bytes = 0;
        for (size_t l = 0; l < k_.size(); ++l)
            if (k_[l]) bytes += k_[l]->size() + v_[l]->size();
        return bytes;
    }
    size_t peak_bytes() const override { return peak_; }
    size_t layers() const { return k_.size(); }
    size_t heads() const { return heads_; }
    size_t dim() const { return dim_; }
    size_t block_floats() const { return heads_ * kVkBlockTokens * dim_; }
    KVType k_type() const { return kt_; }
    KVType v_type() const { return vt_; }
    size_t k_block_bytes() const { return block_floats() * kv_elem_bytes(kt_); }
    size_t v_block_bytes() const { return block_floats() * kv_elem_bytes(vt_); }
    bool backed(size_t id) const { return id < backed_; }
    void ensure(size_t id);
    const BufferPtr& k(size_t layer) const { return k_[layer]; }
    const BufferPtr& v(size_t layer) const { return v_[layer]; }

private:
    VulkanBackend* owner_;
    size_t heads_, dim_, max_, backed_ = 0, peak_ = 0;
    KVType kt_, vt_;
    std::vector<BufferPtr> k_, v_;
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
    uint32_t subgroup_size = 0;
    uint32_t compute_units = 16;  // what the vendor reports, else a small assumption
    DeviceCaps caps{};            // what this device says of itself
    DeviceProfile profile{};      // what measuring its kernels said (backends/device_profile.hpp)
    std::string name;
    bool push_descriptor = false;
    bool int8 = false, float16 = false, storage8 = false, storage16 = false;
    bool integer_dot = false;     // the integer dot product instructions, if the device has them
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
        const VkResult r = dev_->fn.vkAllocateMemory(dev_->device, &ai, nullptr, &memory_);
        if (r != VK_SUCCESS) {
            dev_->fn.vkDestroyBuffer(dev_->device, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
            if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY)
                throw std::bad_alloc();
            check(r, "vkAllocateMemory");
        }
        check(dev_->fn.vkBindBufferMemory(dev_->device, buffer_, memory_, 0), "vkBindBufferMemory");
        if (host_visible) {
            check(dev_->fn.vkMapMemory(dev_->device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped_), "vkMapMemory");
            std::memset(mapped_, 0, bytes);
        }
    }
    ~VulkanBuffer() override {
        if (mapped_) dev_->fn.vkUnmapMemory(dev_->device, memory_);
        if (buffer_) dev_->fn.vkDestroyBuffer(dev_->device, buffer_, nullptr);
        if (memory_) dev_->fn.vkFreeMemory(dev_->device, memory_, nullptr);
    }
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    size_t size() const override { return size_; }
    const void* host_ptr() const override { return mapped_; }
    void* mapped() const { return mapped_; }
    VkBuffer handle() const { return buffer_; }
    bool host_visible() const { return mapped_ != nullptr; }

private:
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

void span(const Buffer& b, size_t off, size_t bytes) {
    if (off > b.size() || bytes > b.size() - off)
        throw std::runtime_error("vulkan: buffer range outside the allocation");
}

class VulkanBackend final : public Backend {
public:
    explicit VulkanBackend(int index, bool diagnostics = false) : dev_(std::make_shared<Device>()) {
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
        d.subgroup_size = sg.subgroupSize;
        if (has_core_props && core.shaderEngineCount && core.shaderArraysPerEngineCount &&
            core.computeUnitsPerShaderArray)
            d.compute_units = core.shaderEngineCount * core.shaderArraysPerEngineCount *
                              core.computeUnitsPerShaderArray;
        VkPhysicalDeviceDriverProperties drv{};
        drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        VkPhysicalDeviceProperties2 dp{};
        dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        dp.pNext = &drv;
        fn.vkGetPhysicalDeviceProperties2(d.physical, &dp);
        d.caps.device = d.props.deviceName;
        d.caps.driver = drv.driverName;
        d.caps.subgroup_size = d.subgroup_size;
        d.caps.compute_units = d.compute_units;
        d.caps.shared_memory_bytes = d.props.limits.maxComputeSharedMemorySize;
        d.caps.matrix_units = false;   // no gfx906 has them; a device that does sets this
        // The row kernel places one subgroup per row group in a 256-lane workgroup, so the subgroup size must divide it.
        if (!d.subgroup_size || 256 % d.subgroup_size ||
            !(sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
            throw VulkanUnavailable("vulkan: " + std::string(d.props.deviceName) +
                                     " has an unsupported subgroup size or no subgroup arithmetic");
        d.name = d.props.deviceName;
        if (d.props.apiVersion < VK_API_VERSION_1_2)
            throw VulkanUnavailable("vulkan: " + d.name + " is older than Vulkan 1.2");
        // A block of 32 activations is quantized across 32 consecutive lanes (shaders/xquant.glsl).
        if (d.subgroup_size < 32)
            throw VulkanUnavailable("vulkan: " + d.name + " has subgroups narrower than 32 lanes");

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
        if (chosen < 0) throw VulkanUnavailable("vulkan: " + d.name + " has no compute queue");
        d.queue_family = (uint32_t)chosen;

        // Timeline semaphores are what submit and wait are built on; the 8- and 16-bit storage and arithmetic features are what the kernels read quantized blocks and half scales with.
        VkPhysicalDeviceVulkan12Features f12{};
        f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceVulkan11Features f11{};
        f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        f11.pNext = &f12;
        VkPhysicalDeviceFeatures2 f2{};
        f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        f2.pNext = &f11;
        fn.vkGetPhysicalDeviceFeatures2(d.physical, &f2);
        if (!f12.timelineSemaphore)
            throw VulkanUnavailable("vulkan: " + d.name + " has no timeline semaphores");
        d.int8 = f12.shaderInt8;
        d.float16 = f12.shaderFloat16;
        d.storage8 = f12.storageBuffer8BitAccess;
        d.storage16 = f11.storageBuffer16BitAccess;
        d.caps.fp16_arithmetic = d.float16;
        d.caps.int8_arithmetic = d.int8;
        d.caps.storage_8bit = d.storage8;
        d.caps.storage_16bit = d.storage16;
        VkPhysicalDeviceVulkan12Features e12{};
        e12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        e12.timelineSemaphore = VK_TRUE;
        e12.shaderInt8 = f12.shaderInt8;
        e12.shaderFloat16 = f12.shaderFloat16;
        e12.storageBuffer8BitAccess = f12.storageBuffer8BitAccess;
        VkPhysicalDeviceVulkan11Features e11{};
        e11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        e11.pNext = &e12;
        e11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
        VkPhysicalDeviceFeatures2 e2{};
        e2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        e2.pNext = &e11;
        // The row kernels select a projection's buffers per workgroup, dynamic indexing of a storage buffer array.
        if (!f2.features.shaderStorageBufferArrayDynamicIndexing)
            throw VulkanUnavailable("vulkan: " + d.name + " cannot index storage buffer arrays dynamically");
        e2.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
        VkPhysicalDeviceShaderIntegerDotProductFeatures edot{};
        edot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
        edot.shaderIntegerDotProduct = VK_TRUE;
        edot.pNext = e2.pNext;
        e2.pNext = &edot;
        // The row kernel's activations are 16-bit integers (shaders/quantize_x.comp).
        if (!f2.features.shaderInt16)
            throw VulkanUnavailable("vulkan: " + d.name + " has no 16-bit integer arithmetic");
        e2.features.shaderInt16 = VK_TRUE;

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
                d.integer_dot = true;
            } else if (std::strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0) {
                enabled.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
                d.exec_stats = true;
            }
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR estat{};
        estat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR;
        estat.pipelineExecutableInfo = VK_TRUE;
        if (d.exec_stats) {
            estat.pNext = e2.pNext;
            e2.pNext = &estat;
        }

        // The profile is chosen here because it depends on the extension scan above.
        d.caps.integer_dot = d.integer_dot;
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
        if (!d.push_descriptor)
            throw VulkanUnavailable("vulkan: " + d.name + " has no VK_KHR_push_descriptor");
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
        d.fn.vkDeviceWaitIdle(d.device);
        for (auto& p : pending_) p.clear();
        for (auto& a : arena_) a.buffer.reset();
        for (auto& variants : kernels_)
            for (Kernel& k : variants) {
                if (k.pipeline) d.fn.vkDestroyPipeline(d.device, k.pipeline, nullptr);
                if (k.layout) d.fn.vkDestroyPipelineLayout(d.device, k.layout, nullptr);
                if (k.set_layout) d.fn.vkDestroyDescriptorSetLayout(d.device, k.set_layout, nullptr);
                if (k.module) d.fn.vkDestroyShaderModule(d.device, k.module, nullptr);
            }
        staging_.reset();
        if (timeline_) d.fn.vkDestroySemaphore(d.device, timeline_, nullptr);
        if (pool_) d.fn.vkDestroyCommandPool(d.device, pool_, nullptr);
    }

    // A compiled kernel's name, the one-column build of a row kernel marked.
    static std::string kernel_variant_name(int id, int variant) {
        const bool tile = id == K_MATMUL_TILE || id == K_MATMUL_TILE_Q || id == K_MATMUL_TILE_Q6;
        return std::string(kKernelNames[id]) + (!variant ? "" : is_row_kernel((KernelId)id) ? "_1col" : tile ? "_small" : "_x8");
    }

    const std::string& name() const { return dev_->name; }
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

    // Host worker counts mean nothing to a device.
    void set_threads(int) override {}
    int threads_available() const override { return 0; }

    BufferPtr alloc(size_t bytes, Memory where) override {
        auto b = std::make_shared<VulkanBuffer>(dev_, bytes, where == Memory::host_visible);
        if (bytes && !b->host_visible()) {
            // Zeroed like every other allocation, in stream order.
            VkCommandBuffer cmd = open();
            dev_->fn.vkCmdFillBuffer(cmd, b->handle(), 0, VK_WHOLE_SIZE, 0);
            barrier(cmd);
        }
        return b;
    }

    // A copy in chunks through staging, each submitted and waited; weights arrive here once at load.
    BufferPtr adopt(const void* src, size_t bytes) override {
        if (!src && bytes) throw std::runtime_error("vulkan: adopting null storage");
        auto b = std::make_shared<VulkanBuffer>(dev_, bytes, false);
        upload(*b, 0, src, bytes);
        return b;
    }

    Ticket submit() override {
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

    // noexcept by contract: a device that cannot report its work finished has been lost, and nothing here can act on that.
    void wait(Ticket t) noexcept override {
        if (t == 0 || t > last_ticket_) return;
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timeline_;
        wi.pValues = &t;
        const VkResult r = dev_->fn.vkWaitSemaphores(dev_->device, &wi, UINT64_MAX);
        if (r != VK_SUCCESS) {
            std::fprintf(stderr, "vulkan: waiting on %s failed with %s; the device is lost\n",
                         dev_->name.c_str(), vk_result_name(r));
            std::abort();
        }
    }

    // Device time per kernel since the last call, in milliseconds, for a diagnostics backend whose queue timestamps; reading them waits for the queue.
    // Dispatches whose time was sampled: the query pool bounds it, so a long run samples its first dispatches.
    size_t timed_dispatches() const {
        size_t n = 0;
        for (size_t i = 0; i < K_COUNT * kVariants; ++i) n += kernel_calls_[i];
        return n;
    }

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
        return out;
    }

    void sync() noexcept override {
        if (open_) {
            try { submit(); } catch (const std::exception& e) {
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
            wait(submit());
            std::memcpy((uint8_t*)dst + done, st.mapped(), n);
            done += n;
        }
    }

    void write(Buffer& dst_b, size_t off, const void* src, size_t bytes) override {
        xq_tag_ = XqTag{};
        group_tag_ = GroupTag{};
        if (!src && bytes) throw std::runtime_error("vulkan: writing from null storage");
        VulkanBuffer& dst = as_vulkan(dst_b);
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
        xq_tag_ = XqTag{};
        group_tag_ = GroupTag{};
        VulkanBuffer& dst = as_vulkan(dst_b);
        const VulkanBuffer& src = as_vulkan(src_b);
        span(dst, dst_off, bytes);
        span(src, src_off, bytes);
        if (!bytes) return;
        VkCommandBuffer cmd = open();
        barrier(cmd);
        VkBufferCopy region{src_off, dst_off, bytes};
        dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
        barrier(cmd);
    }

    // Elementwise kernels: one invocation per element.
    void add(Slice dst, CSlice src, size_t n) override {
        if (!n) return;
        const uint32_t pc[1] = {u32(n)};
        dispatch(K_ADD, {bind(dst), bind(src)}, pc, sizeof(pc), groups(n, 256));
    }

    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n) override {
        if (!n) return;
        const bool quant = n % 32 == 0;
        const uint32_t pc[2] = {u32(n), quant ? 1u : 0u};
        dispatch(K_SILU_MUL, {bind(dst), bind(gate), bind(up), quant ? xq_for(n) : bind(dst)}, pc, sizeof(pc),
                 groups(n, 256), 1, twin_variant());
        if (quant) xq_tag_ = XqTag{bind(dst), n, want_x8_};
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
    void rms_norm(Slice dst, CSlice src, CSlice w, size_t n, float eps) override {
        rms_norm_rows(dst, src, w, 1, n, n, eps);
    }

    void rms_norm_rows(Slice dst, CSlice src, CSlice w, size_t rows, size_t n,
                       size_t stride, float eps) override {
        if (!rows || !n) return;
        const bool quant = stride == n && n % 32 == 0;
        // Several workgroups a row when the output does not overlap the input (shaders/rms_norm_rows.comp).
        const bool overlap = dst.buffer == src.buffer &&
                             dst.offset < src.offset + rows * stride && src.offset < dst.offset + rows * stride;
        const size_t chunks = overlap ? 1 : (n + 255) / 256;
        struct { uint32_t rows, n, stride; float eps; uint32_t quant, chunks; }
            pc{u32(rows), u32(n), u32(stride), eps, quant ? 1u : 0u, u32(chunks)};
        dispatch(K_RMS_NORM_ROWS, {bind(dst), bind(src), bind(w), quant ? xq_for(rows * n) : bind(dst)}, &pc, sizeof(pc),
                 u32(rows * chunks), 1, twin_variant());
        if (quant) xq_tag_ = XqTag{bind(dst), rows * n, want_x8_};
    }

    void norm_rope_rows(Slice x, size_t rows, size_t stride, size_t heads, CSlice w,
                        float eps, CSlice cos, CSlice sin, size_t half,
                        const uint32_t* pos) override {
        if (!rows || !heads || !half) return;
        if (!pos) throw std::runtime_error("vulkan: rope without positions");
        const size_t table = floats_from(cos) / half;
        for (size_t r = 0; r < rows; ++r)
            if (pos[r] >= table) throw std::runtime_error("vulkan: position outside the RoPE table");
        struct { uint32_t rows, stride, heads, half; float eps; }
            pc{u32(rows), u32(stride), u32(heads), u32(half), eps};
        dispatch(K_NORM_ROPE_ROWS,
                 {bind(x), bind(w), bind(cos), bind(sin), args(pos, rows * sizeof(uint32_t))},
                 &pc, sizeof(pc), u32(rows * heads));
    }

    // One kernel for q and k norm-rope and the KV write of one view; a batch over several views takes the three-dispatch default.
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
        struct { uint32_t rows, q_stride, n_head, kv_stride, n_head_kv, half; float eps; uint32_t bt; }
            pc{u32(rows), u32(q_stride), u32(n_head), u32(kv_stride), u32(n_head_kv), u32(rope.half),
               rope.eps, u32(kVkBlockTokens)};
        dispatch(kv_variant(K_NORM_ROPE_KV, K_NORM_ROPE_KV_K16, s),
                 {bind(q), bind(k), bind(v), bind(q_w), bind(k_w), bind(rope.cos), bind(rope.sin),
                  args(rope.pos, rows * sizeof(uint32_t)),
                  bind(CSlice{s.k(layer).get(), 0}), bind(CSlice{s.v(layer).get(), 0}),
                  args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 &pc, sizeof(pc), u32(rows * (n_head + 2 * n_head_kv)));
    }

    void embed(Slice dst, uint32_t type, CSlice table, size_t nin, size_t nrows,
               const uint32_t* ids, size_t count) override {
        if (count && !ids) throw std::runtime_error("vulkan: embed without ids");
        if (!count || !nin) return;
        if (!row_bytes_of(type, nin)) throw std::runtime_error("vulkan: unsupported embedding type");
        if (nin % block_values_of(type))
            throw std::runtime_error("vulkan: embedding width is not whole blocks");
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= nrows) throw std::runtime_error("vulkan: embedding row out of range");
        const uint32_t pc[3] = {u32(nin), u32(count), type};
        // The table is bound twice, as floats for F32 rows and as bytes for block formats.
        dispatch(K_EMBED, {bind(dst), bind(table), bind(table), args(ids, count * sizeof(uint32_t))},
                 pc, sizeof(pc), u32(count));
    }

    void matmul_logits(uint32_t type, CSlice w, CSlice X, Slice Y, size_t nin, size_t nout,
                       size_t nbatch, RowRuns runs = {}) override {
        logits_ = true;
        try { matmul(type, w, X, Y, nin, nout, nbatch, runs); } catch (...) { logits_ = false; throw; }
        logits_ = false;
    }
    void matmul(uint32_t type, CSlice w, CSlice X, Slice Y, size_t nin, size_t nout,
                size_t nbatch, RowRuns runs = {}) override {
        const Projection one{type, w, Y, nout};
        matmul_runs({one}, X, nin, nbatch, false, runs);
    }
    // The residual add folded into the kernels' store: Y += W X.
    void matmul_add(uint32_t type, CSlice w, CSlice X, Slice Y, size_t nin, size_t nout,
                    size_t nbatch, RowRuns runs = {}) override {
        const Projection one{type, w, Y, nout};
        matmul_runs({one}, X, nin, nbatch, true, runs);
    }
    void matmul_group(std::initializer_list<Projection> projections, CSlice X,
                      size_t nin, size_t nbatch, RowRuns runs = {}) override {
        matmul_runs(projections, X, nin, nbatch, false, runs);
    }

    // With row runs, a row's kernel follows its prompt's extent rather than the call's width, so a prompt computes the same however its rows are batched (docs/VULKAN.md, batch invariance).
    // Adjacent runs taking the same kernel are one call, and a call of mixed runs becomes one call per kernel over its rows.
    // A tile row's inner-dimension split is the one a pass over its whole prompt would take, up to a microbatch of 512 rows; runs whose splits differ are separate calls.
    static size_t split_tiles_of(size_t extent) { return (std::min<size_t>(extent, 512) + 63) / 64; }
    void matmul_runs(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nbatch,
                     bool accumulate, RowRuns runs) {
        if (!runs.n || !nbatch) {
            matmul_group_impl(projections, X, nin, nbatch, accumulate);
            return;
        }
        if (runs.runs[runs.n - 1].end != nbatch) throw std::runtime_error("vulkan: row runs do not cover the batch");
        bool eight_bit_or_float = true;
        for (const Projection& pr : projections)
            if (pr.type != gguf::GGML_TYPE_Q8_0 && pr.type != gguf::GGML_TYPE_F32) eight_bit_or_float = false;
        const size_t tile_from = tile_from_for(dev_->profile, eight_bit_or_float, nin);
        size_t start = 0;
        for (size_t i = 0; i < runs.n;) {
            const bool tile = runs.runs[i].extent >= tile_from;
            const size_t split = tile ? split_tiles_of(runs.runs[i].extent) : 0;
            auto same = [&](const RowRun& r) {
                return (r.extent >= tile_from) == tile && (!tile || split_tiles_of(r.extent) == split);
            };
            size_t j = i + 1;
            while (j < runs.n && same(runs.runs[j])) ++j;
            const size_t end = runs.runs[j - 1].end;
            if (end < start) throw std::runtime_error("vulkan: row runs out of order");
            if (end > start) {
                const int k = tile ? 1 : 0;
                if (start == 0 && end == nbatch) {
                    matmul_group_impl(projections, X, nin, nbatch, accumulate, k, split);
                } else {
                    std::vector<Projection> at(projections);
                    for (Projection& pr : at) pr.out.offset += start * pr.rows;
                    const CSlice xs{X.buffer, X.offset + start * nin};
                    if (at.size() == 1) matmul_group_impl({at[0]}, xs, nin, end - start, accumulate, k, split);
                    else if (at.size() == 2) matmul_group_impl({at[0], at[1]}, xs, nin, end - start, accumulate, k, split);
                    else if (at.size() == 3) matmul_group_impl({at[0], at[1], at[2]}, xs, nin, end - start, accumulate, k, split);
                    else throw std::logic_error("vulkan: a row-run call of more than three projections");
                }
            }
            start = end;
            i = j;
        }
    }

    // Up to three projections of one X in one dispatch: the row kernel for narrow batches hands workgroups to projections in order; wide batches take the tile.
    // `kernel_choice` forces the row kernel (0) or the tile (1), below zero the call's width chooses; `split_tiles` is the column tiles the tile's split is taken for, zero for the call's own.
    void matmul_group_impl(std::initializer_list<Projection> projections, CSlice X,
                           size_t nin, size_t nbatch, bool accumulate, int kernel_choice = -1, size_t split_tiles = 0) {
        if (projections.size() > 3) {
            // The kernel's limit, and no caller passes more; split.
            std::vector<Projection> all(projections);
            for (size_t i = 0; i < all.size(); i += 3) {
                std::initializer_list<Projection> part =
                    i + 3 <= all.size() ? std::initializer_list<Projection>{all[i], all[i + 1], all[i + 2]}
                    : (i + 2 == all.size() ? std::initializer_list<Projection>{all[i], all[i + 1]}
                                           : std::initializer_list<Projection>{all[i]});
                matmul_group_impl(part, X, nin, nbatch, accumulate, kernel_choice, split_tiles);
            }
            return;
        }
        if (!nbatch) return;
        std::vector<const Projection*> live;
        for (const Projection& pr : projections) {
            if (!pr.data.buffer) throw std::runtime_error("vulkan: projection without storage");
            const size_t row_bytes = row_bytes_of(pr.type, nin);
            if (!row_bytes)
                throw std::runtime_error("vulkan: unsupported matrix type " + std::to_string(pr.type) +
                                         " (docs/VULKAN.md sub-step 6)");
            if (nin % block_values_of(pr.type))
                throw std::runtime_error("vulkan: matrix width is not whole blocks");
            if (bytes_from(pr.data) < pr.rows * row_bytes || floats_from(pr.out) < nbatch * pr.rows)
                throw std::runtime_error("vulkan: matmul operand outside its allocation");
            if (pr.rows) live.push_back(&pr);
        }
        if (floats_from(X) < nbatch * nin) throw std::runtime_error("vulkan: matmul operand outside its allocation");
        if (live.empty()) return;
        // The row kernel costs a weight pass per eight columns and the tile a whole tile however little is filled, so the crossover depends on the row width and the type; measured per device (backends/device_profile.hpp).
        bool eight_bit_or_float = true;
        for (const Projection* pr : live)
            if (pr->type != gguf::GGML_TYPE_Q8_0 && pr->type != gguf::GGML_TYPE_F32)
                eight_bit_or_float = false;
        const size_t tile_from = tile_from_for(dev_->profile, eight_bit_or_float, nin);
        if (kernel_choice == 1 || (kernel_choice < 0 && nbatch >= tile_from)) {
            const size_t gy = (nbatch + 63) / 64;
            if (gy > dev_->props.limits.maxComputeWorkGroupCount[1])
                throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
            // On a device whose integer dot is native, quantized types take the integer-dot tile (shaders/matmul_tile_q.comp), X quantized to 8 bits once for every projection that needs it.
            VkDescriptorBufferInfo x8{};
            const size_t nblk = nin / 32;
            // The float tile, one projection a dispatch.
            for (const Projection* pr : live) {
                if (integer_dot_tile(pr->type)) continue;
                // A taller tile reads less shared memory per product but halves the workgroups; below one per compute unit the call takes a shorter one.
                const uint32_t height = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall,
                                                      pr->rows, gy, nin);
                const bool tall = height == kTileRowsTall;
                const uint32_t pc[7] = {u32(nin), u32(pr->rows), u32(nbatch), pr->type, accumulate ? 1u : 0u, 0, 0};
                dispatch(tall ? K_MATMUL_TILE_TALL : K_MATMUL_TILE,
                         {bind(pr->out), bind(pr->data), bind(pr->data), bind(X), bind(pr->data), bind(X)},
                         pc, sizeof(pc), groups(pr->rows, height), (uint32_t)gy, height == kTileRowsSmall ? 1 : 0);
            }
            // The integer-dot tile takes the projections of one type in one dispatch, since each alone can be too small to fill the device.
            std::vector<const Projection*> pending;
            for (const Projection* pr : live)
                if (integer_dot_tile(pr->type)) pending.push_back(pr);
            while (!pending.empty()) {
                std::vector<const Projection*> group, rest;
                for (const Projection* pr : pending)
                    (pr->type == pending[0]->type ? group : rest).push_back(pr);
                pending.swap(rest);
                if (!x8.buffer) {
                    x8 = x8_for(nbatch * nin);
                    const uint32_t qpc[3] = {u32(nbatch * nin), u32(nin), u32(nbatch)};
                    dispatch(K_QUANTIZE_X8, {bind(X), x8}, qpc, sizeof(qpc), groups(nbatch * nin, 256));
                }
                size_t rows = 0;
                for (const Projection* pr : group) rows += pr->rows;
                const uint32_t height = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall,
                                                      rows, gy, nin);
                const bool tall = height == kTileRowsTall;
                const bool q6 = group[0]->type == gguf::GGML_TYPE_Q6_K;
                const KernelId kernel = q6 ? (tall ? K_MATMUL_TILE_Q6_TALL : K_MATMUL_TILE_Q6)
                                           : (tall ? K_MATMUL_TILE_Q_TALL : K_MATMUL_TILE_Q);
                uint32_t start[3] = {0, 0, 0}, nout[3] = {0, 0, 0};
                size_t gx = 0;
                for (size_t i = 0; i < group.size(); ++i) {
                    start[i] = u32(gx);
                    nout[i] = u32(group[i]->rows);
                    gx += groups(group[i]->rows, height);
                }
                // The split is the one the rows' whole prompt would take (matmul_runs).
                const size_t st = split_tiles ? split_tiles : gy;
                const uint32_t hs = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, rows, st, nin);
                size_t gxs = 0;
                for (const Projection* pr : group) gxs += groups(pr->rows, hs);
                const size_t kper = split_blocks(gxs * st, nblk);
                const size_t parts = (nblk + kper - 1) / kper;
                const uint32_t pc[14] = {u32(nin), u32(nbatch), group[0]->type, accumulate && parts == 1 ? 1u : 0u, u32(kper),
                                         u32(group.size()), nout[0], start[0], nout[1], start[1], nout[2], start[2], 0, 0};
                const Projection& a = *group[0];
                const Projection& b = group.size() > 1 ? *group[1] : a;
                const Projection& c = group.size() > 2 ? *group[2] : a;
                const int small = height == kTileRowsSmall ? 1 : 0;
                if (parts > 1) {
                    const size_t n = nbatch * rows;
                    if (!parts_ || parts_->size() < parts * n * sizeof(float)) grow(parts_, parts * n * sizeof(float));
                    const VkDescriptorBufferInfo pb{parts_->handle(), 0, VK_WHOLE_SIZE};
                    dispatch(kernel, {pb, pb, pb, bind(a.data), bind(b.data), bind(c.data), x8, x8, x8}, pc, sizeof(pc), u32(gx),
                             u32(gy * parts), small);
                    const size_t n0 = nbatch * a.rows, n1 = group.size() > 1 ? nbatch * b.rows : 0,
                                 n2 = group.size() > 2 ? nbatch * c.rows : 0;
                    const size_t start1 = groups(n0, 256), start2 = start1 + groups(n1, 256);
                    // An unused projection's start is past every workgroup.
                    const uint32_t rc[7] = {u32(n0), u32(n1), u32(n2), u32(parts), accumulate ? 1u : 0u,
                                            u32(n1 ? start1 : start2 + groups(n2, 256)), u32(n2 ? start2 : start2 + groups(n2, 256) + 1)};
                    dispatch(K_MATMUL_REDUCE, {bind(a.out), bind(b.out), bind(c.out), pb}, rc, sizeof(rc),
                             u32(start2 + groups(n2, 256)));
                } else {
                    dispatch(kernel, {bind(a.out), bind(b.out), bind(c.out), bind(a.data), bind(b.data), bind(c.data), x8, x8, x8},
                             pc, sizeof(pc), u32(gx), (uint32_t)gy, small);
                }
            }
            return;
        }
        // One cluster size and one module serve a dispatch, so every projection in it has the same type.
        // A mixed group, such as the Q5_K q and k beside the Q6_K v of a Q5_K_M file, is partitioned by type and each partition is one dispatch: two for that group rather than three.
        for (size_t i = 1; i < live.size(); ++i)
            if (live[i]->type != live[0]->type) {
                std::vector<Projection> same, rest;
                for (const Projection* pr : live)
                    (pr->type == live[0]->type ? same : rest).push_back(*pr);
                auto run = [&](const std::vector<Projection>& v) {
                    if (v.size() == 1) matmul_group_impl({v[0]}, X, nin, nbatch, accumulate, kernel_choice);
                    else if (v.size() == 2) matmul_group_impl({v[0], v[1]}, X, nin, nbatch, accumulate, kernel_choice);
                    else matmul_group_impl({v[0], v[1], v[2]}, X, nin, nbatch, accumulate, kernel_choice);
                };
                run(same);
                run(rest);
                return;
            }
        const RowPlan plan = row_plan(live[0]->type, nin);
        const VkDescriptorBufferInfo xqi = row_twin(X, live[0]->type, plan.kernel, nbatch * nin);
        for (size_t col0 = 0; col0 < nbatch; col0 += 8)
            row_dispatch(plan, live, X, xqi, nin, nbatch, col0, std::min<size_t>(8, nbatch - col0), accumulate);
    }

    // The row kernel for a type at a width (matmul_row.comp): its module, whether rows take the wide layout, and how a subgroup's lanes split over rows.
    // Q8_0 pairs go over four lanes and Q4_0 pairs over two when the block count is even, Q4_1 blocks over one, the K-quant blocks over eight, else one unit per block or value.
    struct RowPlan {
        KernelId kernel;
        uint32_t type, wide, cluster, rows_per_sg, rows_per_group;
    };
    RowPlan row_plan(uint32_t type, size_t nin) const {
        const size_t nblocks = nin / block_values_of(type);
        uint32_t wide = 0, lanes = 1;
        size_t units = nin;
        KernelId kernel = K_MATMUL_ROW;
        switch (type) {
        case gguf::GGML_TYPE_Q8_0: {
            const uint32_t per_pair = dev_->profile.q8_lanes_per_pair;
            wide = nblocks % 2 == 0 && nblocks / 2 >= per_pair && dev_->subgroup_size >= per_pair;
            lanes = wide ? per_pair : 1;
            units = wide ? nblocks / 2 * lanes : nblocks;
            if (wide) kernel = K_MATMUL_ROW_Q8W;
            break;
        }
        case gguf::GGML_TYPE_Q4_0:
            wide = nblocks % 2 == 0;
            lanes = wide ? 2 : 1;
            units = wide ? nblocks / 2 * lanes : nblocks;
            kernel = K_MATMUL_ROW_Q4;
            break;
        case gguf::GGML_TYPE_Q4_1:
            units = nblocks;
            kernel = K_MATMUL_ROW_Q4;
            break;
        case gguf::GGML_TYPE_Q4_K:
        case gguf::GGML_TYPE_Q5_K:
        case gguf::GGML_TYPE_Q6_K:
            lanes = dev_->profile.kquant_lanes;
            if (dev_->subgroup_size < lanes)
                throw std::runtime_error("vulkan: K-quant rows need a subgroup of " + std::to_string(lanes) + " lanes");
            units = nblocks * lanes;
            kernel = type == gguf::GGML_TYPE_Q6_K ? K_MATMUL_ROW_K : type == gguf::GGML_TYPE_Q5_K ? K_MATMUL_ROW_K5 : K_MATMUL_ROW_K4;
            break;
        default: break;
        }
        if (dev_->profile.prefer_integer_dot) kernel = row_dot_variant(kernel);
        // Q6_K, Q4_0 and Q4_1 rows on the 8-bit twin where the integer dot is native, except in the output head, where on 8 bits the HF gate's Q4_0 file fails its top-5 bound.
        if (kernel == K_MATMUL_ROW_K_DOT && !logits_) kernel = K_MATMUL_ROW_K_DOT8;
        if (kernel == K_MATMUL_ROW_Q4_DOT && !logits_) kernel = K_MATMUL_ROW_Q4_DOT8;
        uint32_t cluster = lanes;
        while (cluster < dev_->subgroup_size && cluster < units) cluster *= 2;
        if (kernel == K_MATMUL_ROW_K_DOT8 || kernel == K_MATMUL_ROW_Q4_DOT8)
            cluster = std::min(cluster, std::max(lanes, dev_->profile.q6k_row_lanes));
        // Where the integer dot is native, Q8_0 rows take the four-wide dot over the 8-bit twin (shaders/matmul_vec_q8.comp).
        if (type == gguf::GGML_TYPE_Q8_0 && dev_->profile.prefer_integer_dot && dev_->subgroup_size >= 8) {
            kernel = K_MATMUL_VEC_Q8;
            cluster = dev_->subgroup_size / 2;
        }
        const uint32_t rows_per_sg = dev_->subgroup_size / cluster;
        return RowPlan{kernel, type, wide, cluster, rows_per_sg, (256 / dev_->subgroup_size) * rows_per_sg};
    }

    // What a row kernel reads X through: the floats for F32 rows, else the activations' twin (shaders/xquant.glsl), which the norm, SiLU and attention kernels write beside their output and tag.
    // An input without one gets a quantize dispatch here; the scratch is reused stream-ordered.
    VkDescriptorBufferInfo row_twin(CSlice X, uint32_t type, KernelId kernel, size_t n) {
        if (type == gguf::GGML_TYPE_F32) return bind(X);
        const VkDescriptorBufferInfo xf = bind(X);
        VkDescriptorBufferInfo xqi = xq_for(n);
        const bool x8 = reads_x8(kernel);
        // The first matmul reading the 8-bit twin has it made here; producers after it write both.
        if (x8) want_x8_ = true;
        if (!(xq_tag_.n == n && xq_tag_.x.buffer == xf.buffer && xq_tag_.x.offset == xf.offset && (!x8 || xq_tag_.has8))) {
            const uint32_t qpc[1] = {u32(n)};
            dispatch(K_QUANTIZE_X, {xf, xqi}, qpc, sizeof(qpc), groups(n, 256), 1, twin_variant());
            xq_tag_ = XqTag{xf, n, want_x8_};
        }
        if (x8) xqi.offset = x8_base_bytes(n);
        return xqi;
    }

    // One row kernel dispatch over up to three projections of one type and columns col0 .. col0 + ncols of X, which has nbatch columns.
    // A routed dispatch (`per` nonzero) instead runs one entry per workgroup row, `entries` of them, through the expert ids in `ids`.
    void row_dispatch(const RowPlan& plan, const std::vector<const Projection*>& live, CSlice X, VkDescriptorBufferInfo xqi,
                      size_t nin, size_t nbatch, size_t col0, size_t ncols, bool accumulate,
                      uint32_t per = 0, size_t entries = 1, VkDescriptorBufferInfo ids = {},
                      uint32_t order0 = 0, VkDescriptorBufferInfo tab = {}, size_t routed = 0) {
        uint32_t nout[3] = {0, 0, 0}, start[3] = {0, 0, 0};
        uint32_t total = 0;
        for (size_t i = 0; i < live.size(); ++i) {
            nout[i] = u32(live[i]->rows);
            start[i] = total;
            total += groups(live[i]->rows, plan.rows_per_group);
        }
        if (total > dev_->props.limits.maxComputeWorkGroupCount[0] || entries > dev_->props.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        // Unused projection slots bind the first one's buffers; no workgroup reaches them.
        const Projection& a = *live[0];
        const Projection& b = live.size() > 1 ? *live[1] : a;
        const Projection& c = live.size() > 2 ? *live[2] : a;
        const uint32_t t = plan.type, w = plan.wide;
        const uint32_t pc[23] = {u32(nin), u32(nbatch), u32(col0), u32(ncols), plan.cluster, plan.rows_per_sg,
                                 (uint32_t)live.size(),
                                 nout[0], t, w, start[0],
                                 nout[1], t, w, start[1],
                                 nout[2], t, w, start[2], accumulate ? 1u : 0u, per, tab.buffer ? 1u : 0u, order0};
        dispatch(plan.kernel,
                 {bind(a.out), bind(b.out), bind(c.out),
                  bind(a.data), bind(b.data), bind(c.data),
                  bind(a.data), bind(b.data), bind(c.data),
                  bind(a.data), bind(b.data), bind(c.data),
                  bind(X),
                  bind(a.data), bind(b.data), bind(c.data),
                  xqi, xqi, xqi, xqi, ids.buffer ? ids : bind(X), tab.buffer ? tab : bind(X)},
                 pc, sizeof(pc), total, u32(entries),
                 ncols == 1 && row_kernel_builds_one_column(plan.kernel) ? 1 : 0);
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
        const uint32_t pc[4] = {u32(rows), u32(n_expert), u32(k), normalize ? 1u : 0u};
        dispatch(K_MOE_ROUTE, {bind(scores), bind(ids), bind(weights)}, pc, sizeof(pc), u32(rows));
        group_tag_ = GroupTag{};
        if (overlaps_twin(bind(ids), rows * k) || overlaps_twin(bind(weights), rows * k)) xq_tag_ = XqTag{};
    }

    // Calls `each(first, count, tile)` over the token rows of a routed call, a call per stretch of rows that take the same kernel.
    template <typename Fn>
    void expert_runs(uint32_t type, size_t nrows, RowRuns runs, const Fn& each) {
        const size_t from = moe_tile_from_for(dev_->profile, type);
        if (!runs.n) { each(0, nrows, nrows >= from); return; }
        if (runs.runs[runs.n - 1].end != nrows) throw std::runtime_error("vulkan: row runs do not cover the batch");
        size_t start = 0;
        for (size_t i = 0; i < runs.n;) {
            const bool tile = runs.runs[i].extent >= from;
            size_t j = i + 1;
            while (j < runs.n && (runs.runs[j].extent >= from) == tile) ++j;
            const size_t end = runs.runs[j - 1].end;
            if (end < start) throw std::runtime_error("vulkan: row runs out of order");
            if (end > start) each(start, end - start, tile);
            start = end;
            i = j;
        }
    }

    static CSlice at_float(CSlice s, size_t floats) { return CSlice{s.buffer, s.offset + floats}; }
    static Slice at_float(Slice s, size_t floats) { return Slice{s.buffer, s.offset + floats}; }

    void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nrows,
                        const Routing& routing, RowRuns runs) override {
        const size_t k = routing.k, entries = nrows * k;
        if (!entries || !projections.size()) return;
        if (projections.size() > 3) throw std::logic_error("vulkan: more than three routed projections");
        for (const Projection& pr : projections) {
            check_experts(pr.type, pr.data, nin, pr.rows, routing.n_expert);
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
            routed_call(live, at_float(X, first * nin), nin, count, count, k, at_float(routing.ids, first * k), routing.n_expert, tile);
        });
    }

    void matmul_experts_add(uint32_t type, CSlice data, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows,
                            const Routing& routing, RowRuns runs) override {
        const size_t k = routing.k, entries = nrows * k;
        if (!entries) return;
        check_experts(type, data, nin, nout, routing.n_expert);
        if (floats_from(X) < entries * nin || floats_from(Y) < nrows * nout || floats_from(routing.ids) < entries ||
            floats_from(routing.weights) < entries)
            throw std::runtime_error("vulkan: matmul operand outside its allocation");
        if (!moe_out_ || moe_out_->size() < entries * nout * sizeof(float)) grow(moe_out_, entries * nout * sizeof(float));
        expert_runs(type, nrows, runs, [&](size_t first, size_t count, bool tile) {
            const Slice part{moe_out_.get(), first * k * nout};
            const Projection one{type, data, part, nout};
            routed_call({&one}, at_float(X, first * k * nin), nin, count * k, count, k, at_float(routing.ids, first * k),
                        routing.n_expert, tile, 1);
            const uint32_t pc[3] = {u32(count), u32(nout), u32(k)};
            dispatch(K_MOE_COMBINE, {bind(at_float(Y, first * nout)), bind(CSlice(part)), bind(at_float(routing.weights, first * k))},
                     pc, sizeof(pc), groups(count * nout, 256));
        });
    }

    // One routed call over `nrows` token rows, k entries each, whose X has `xcols` columns: entry e reads column e / per, per being k when X is the token rows and 1 when it holds a column per entry.
    void routed_call(const std::vector<const Projection*>& live, CSlice X, size_t nin, size_t xcols, size_t nrows, size_t k,
                     CSlice ids, size_t n_expert, bool tile, size_t per = 0) {
        if (!per) per = k;
        const size_t entries = nrows * k;
        const uint32_t type = live[0]->type;
        if (!tile && nrows == 1) {
            // A lone generated token: each entry its own workgroup row, through the one-column build.
            const RowPlan plan = row_plan(type, nin);
            const VkDescriptorBufferInfo xqi = row_twin(X, type, plan.kernel, xcols * nin);
            row_dispatch(plan, live, X, xqi, nin, xcols, 0, 1, false, u32(per), entries, bind(ids));
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
            // Generated tokens beside each other: each run of one expert's entries a workgroup row of the row kernel's wide build, a column per entry.
            // A column computes as it would alone, so an entry does not depend on what else is routed beside it.
            const RowPlan plan = row_plan(type, nin);
            const VkDescriptorBufferInfo xqi = row_twin(X, type, plan.kernel, xcols * nin);
            row_dispatch(plan, live, X, xqi, nin, xcols, 0, kRowColsWide, false, u32(per), max_tiles, bind(ids), order0, tab, entries);
            return;
        }
        if (integer_dot_tile(type)) {
            const VkDescriptorBufferInfo x8 = x8_for(xcols * nin);
            const uint32_t qpc[3] = {u32(xcols * nin), u32(nin), u32(xcols)};
            dispatch(K_QUANTIZE_X8, {bind(X), x8}, qpc, sizeof(qpc), groups(xcols * nin, 256));
            size_t rows = 0;
            for (const Projection* pr : live) rows += pr->rows;
            const uint32_t height = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall,
                                                  rows, max_tiles, nin);
            const bool tall = height == kTileRowsTall, q6 = type == gguf::GGML_TYPE_Q6_K;
            const KernelId kernel = q6 ? (tall ? K_MATMUL_TILE_Q6_TALL : K_MATMUL_TILE_Q6) : (tall ? K_MATMUL_TILE_Q_TALL : K_MATMUL_TILE_Q);
            uint32_t start[3] = {0, 0, 0}, nout[3] = {0, 0, 0};
            size_t gx = 0;
            for (size_t i = 0; i < live.size(); ++i) {
                start[i] = u32(gx);
                nout[i] = u32(live[i]->rows);
                gx += groups(live[i]->rows, height);
            }
            const Projection& a = *live[0];
            const Projection& b = live.size() > 1 ? *live[1] : a;
            const Projection& c = live.size() > 2 ? *live[2] : a;
            const uint32_t pc[14] = {u32(nin), u32(xcols), type, 0, u32(nin / 32), u32(live.size()),
                                     nout[0], start[0], nout[1], start[1], nout[2], start[2], u32(per), order0};
            dispatch(kernel, {bind(a.out), bind(b.out), bind(c.out), bind(a.data), bind(b.data), bind(c.data), x8, x8, tab},
                     pc, sizeof(pc), u32(gx), u32(max_tiles), height == kTileRowsSmall ? 1 : 0);
        } else {
            for (const Projection* pr : live) {
                const uint32_t height = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall,
                                                      pr->rows, max_tiles, nin);
                const uint32_t pc[7] = {u32(nin), u32(pr->rows), u32(xcols), type, 0, u32(per), order0};
                dispatch(height == kTileRowsTall ? K_MATMUL_TILE_TALL : K_MATMUL_TILE,
                         {bind(pr->out), bind(pr->data), bind(pr->data), bind(X), bind(pr->data), tab},
                         pc, sizeof(pc), groups(pr->rows, height), u32(max_tiles), height == kTileRowsSmall ? 1 : 0);
            }
        }
        for (const Projection* pr : live)
            if (bind(pr->out).buffer == xq_tag_.x.buffer) xq_tag_ = XqTag{};
    }

    // Whether `floats` floats at a binding overlap the input the activations' twin was made from.
    bool overlaps_twin(const VkDescriptorBufferInfo& b, size_t floats) const {
        if (!xq_tag_.n || b.buffer != xq_tag_.x.buffer) return false;
        const VkDeviceSize end = b.offset + floats * sizeof(float), tag_end = xq_tag_.x.offset + xq_tag_.n * sizeof(float);
        return b.offset < tag_end && xq_tag_.x.offset < end;
    }

    void check_experts(uint32_t type, CSlice data, size_t nin, size_t nout, size_t n_expert) {
        if (!data.buffer) throw std::runtime_error("vulkan: projection without storage");
        const size_t row_bytes = row_bytes_of(type, nin);
        if (!row_bytes) throw std::runtime_error("vulkan: unsupported matrix type " + std::to_string(type));
        if (nin % block_values_of(type)) throw std::runtime_error("vulkan: matrix width is not whole blocks");
        if (bytes_from(data) < n_expert * nout * row_bytes) throw std::runtime_error("vulkan: matmul operand outside its allocation");
    }

    // The scratch the twin of an n-value input lives in.
    VkDescriptorBufferInfo xq_for(size_t n) {
        const size_t bytes = dev_->profile.prefer_integer_dot ? x8_base_bytes(n) + n + (n / 32) * 8 : n * 2 + (n / 32) * 16;
        if (!xq_ || xq_->size() < bytes) {
            grow(xq_, bytes);
            xq_tag_ = XqTag{};
        }
        return VkDescriptorBufferInfo{xq_->handle(), 0, VK_WHOLE_SIZE};
    }

    // Where the 8-bit twin starts after the 16-bit one, in bytes, rounded up to 256 so it is a valid binding offset (shaders/xquant.glsl).
    static size_t x8_base_bytes(size_t n) { return ((n / 2 + n / 8 + 63) & ~size_t(63)) * 4; }
    // The row kernels that read the 8-bit twin: the Q4_0/Q4_1, Q4_K, Q5_K and Q6_K families built with LLMX_X8, and the Q8_0 kernel. An output head of those first three stays on the 16-bit twin, since on the 8-bit one the HF gate's Q4_0 fixture failed its top-5 bound (docs/STATUS.md).
    static bool reads_x8(KernelId id) {
        return id == K_MATMUL_ROW_K4_DOT || id == K_MATMUL_ROW_K5_DOT || id == K_MATMUL_ROW_K_DOT8 || id == K_MATMUL_ROW_Q4_DOT8 ||
               id == K_MATMUL_VEC_Q8;
    }

    // Whether a type's wide matmul goes through the integer-dot tile on this device.
    bool integer_dot_tile(uint32_t type) const {
        return dev_->profile.prefer_integer_dot && dev_->caps.integer_dot &&
               type != gguf::GGML_TYPE_F32;
    }

    // The quant blocks each part of a split integer-dot tile call sums, the whole inner dimension when unsplit: a call of fewer workgroups than tile_split_per_cu (tile_split_per_cu_narrow for narrow rows) per compute unit splits, keeping at least tile_split_min_blocks per part.
    size_t split_blocks(size_t workgroups, size_t nblk) const {
        const bool narrow = nblk * 32 < dev_->profile.tile_narrow_nin;
        const size_t target = (size_t)(narrow ? dev_->profile.tile_split_per_cu_narrow : dev_->profile.tile_split_per_cu) * dev_->caps.compute_units;
        const size_t floor_blocks = std::max<size_t>(dev_->profile.tile_split_min_blocks, 2);
        if (workgroups >= target) return nblk;
        size_t parts = std::min((target + workgroups - 1) / workgroups, std::max<size_t>(1, nblk / floor_blocks));
        if (parts <= 1) return nblk;
        size_t kper = (nblk + parts - 1) / parts;
        kper = (kper + 1) & ~size_t(1);   // a multiple of the tile's STEP
        return std::min(kper, nblk);
    }

    // The scratch the 8-bit twin of an n-value batch lives in (shaders/quantize_x8.comp), reused stream-ordered.
    VkDescriptorBufferInfo x8_for(size_t n) {
        const size_t bytes = n + (n / 32) * 8;
        if (!x8_ || x8_->size() < bytes) grow(x8_, bytes);
        return VkDescriptorBufferInfo{x8_->handle(), 0, VK_WHOLE_SIZE};
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
        if (head_dim > 256) throw std::runtime_error("vulkan: head width above 256 is not supported");
        return std::make_unique<VulkanKVStorage>(*this, layers, n_head_kv, head_dim,
                                                 VulkanKVStorage::blocks_for(max_tokens), k_type, v_type);
    }

    void kv_copy(KVStorage& storage, int32_t src, int32_t dst) override {
        VulkanKVStorage& s = storage_of(storage);
        if (src < 0 || dst < 0 || !s.backed((size_t)src) || (size_t)dst >= s.max_blocks())
            throw std::runtime_error("vulkan: KV copy outside the storage");
        s.ensure((size_t)dst);
        const size_t kb = s.k_block_bytes(), vb = s.v_block_bytes();
        for (size_t l = 0; l < s.layers(); ++l) {
            copy(*s.k(l), (size_t)dst * kb, *s.k(l), (size_t)src * kb, kb);
            copy(*s.v(l), (size_t)dst * vb, *s.v(l), (size_t)src * vb, vb);
        }
    }

    // One dispatch per view: its rows scatter into its blocks.
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
                 {bind(CSlice{s.k(layer).get(), 0}), bind(CSlice{s.v(layer).get(), 0}),
                  bind(k), bind(v), args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 pc, sizeof(pc), groups(t.rows * hd, 256));
    }

    // One dispatch per view, one workgroup per (row, head).
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out,
                   int n_head, int n_head_kv, int head_dim) override {
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > 256)
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
        // Views of 128-wide heads whose prompt reaches 32 tokens take the tiled kernel, the rest the per-row kernel: at most two dispatches per layer.
        // The choice is by the view's extent rather than its row count, so a prompt's rows take the same kernel however they were batched.
        std::vector<Placed> wide, narrow;
        for (const Placed& pv : placed) {
            const size_t extent = pv.view->extent ? pv.view->extent : pv.view->nq;
            (extent >= dev_->profile.attention_tile_rows && head_dim == 128 ? wide : narrow).push_back(pv);
        }
        if (!wide.empty()) {
            ViewTable t = view_table(layer, wide, false);
            VulkanKVStorage& s = *t.storage;
            check_storage(s, layer, n_head_kv, head_dim);
            size_t tiles = 0;
            const size_t atr = dev_->profile.attention_tile_rows;
            for (const Placed& pv : wide) tiles += (pv.view->nq + atr - 1) / atr;
            struct { uint32_t rows, n_head, n_head_kv, bt; float scale; }
                tc{u32(t.rows), (uint32_t)n_head, (uint32_t)n_head_kv, u32(kVkBlockTokens), scale};
            dispatch(kv_variant(K_ATTENTION_TILE, K_ATTENTION_TILE_K16, s),
                     {bind(Q), bind(out), bind(CSlice{s.k(layer).get(), 0}), bind(CSlice{s.v(layer).get(), 0}),
                      args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                     &tc, sizeof(tc), u32(tiles * (size_t)n_head));
        }
        if (!narrow.empty()) {
            ViewTable t = view_table(layer, narrow, false);
            VulkanKVStorage& s = *t.storage;
            check_storage(s, layer, n_head_kv, head_dim);
            // The output's 16-bit twin, written by whichever kernel writes the output, when the whole batch is this dispatch and a head is whole blocks.
            const bool quant = wide.empty() && head_dim % 32 == 0;
            const VkDescriptorBufferInfo xq = quant ? xq_for(rows * qstride) : bind(out);
            // A row's history is split in parts of 32 tokens across workgroups, the part doubling until at most 64 cover it; the parts depend only on the row's length, so a row computes the same whatever else is in the dispatch.
            // The dispatch has as many splits as its longest row can need; a row's extra splits are empty.
            size_t longest = 0;
            for (const Placed& pv : narrow)
                longest = std::max(longest, VulkanKVStorage::add(pv.view->length, pv.view->nq));
            const size_t pairs = t.rows * (size_t)n_head;
            const DeviceProfile& prof = dev_->profile;
            const size_t chunk = prof.attention_split_chunk;
            const size_t nsplit = std::min(prof.attention_split_max, std::max<size_t>(1, (longest + chunk - 1) / chunk));
            const size_t scratch_floats = nsplit > 1 ? pairs * nsplit * ((size_t)head_dim + 2) : 0;
            if (scratch_floats && (!scratch_ || scratch_->size() < scratch_floats * sizeof(float)))
                grow(scratch_, scratch_floats * sizeof(float));
            struct { uint32_t rows, n_head, n_head_kv, dim, bt; float scale; uint32_t nsplit, chunk, quant, max_parts; }
                pc{u32(t.rows), (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)head_dim, u32(kVkBlockTokens),
                   scale, u32(nsplit), u32(chunk), quant ? 1u : 0u, u32(prof.attention_split_max)};
            const VkDescriptorBufferInfo scratch = scratch_
                ? VkDescriptorBufferInfo{scratch_->handle(), 0, VK_WHOLE_SIZE} : bind(out);
            const VkDescriptorBufferInfo table = args(t.words.data(), t.words.size() * sizeof(uint32_t));
            dispatch(kv_variant(K_ATTENTION, K_ATTENTION_K16, s),
                     {bind(Q), bind(out), bind(CSlice{s.k(layer).get(), 0}), bind(CSlice{s.v(layer).get(), 0}),
                      table, scratch, xq},
                     &pc, sizeof(pc), groups(pairs * nsplit, 1), 1, twin_variant());
            if (nsplit > 1) {
                const uint32_t mc[5] = {u32(t.rows), (uint32_t)n_head, (uint32_t)head_dim, u32(nsplit), quant ? 1u : 0u};
                dispatch(K_ATTENTION_MERGE, {bind(out), scratch, table, xq}, mc, sizeof(mc), groups(pairs, 1), 1,
                         twin_variant());
            }
            if (quant) xq_tag_ = XqTag{bind(out), rows * qstride, want_x8_};
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
    // The table the batched cache kernels read (shaders/views.glsl) for a subset of a batch's views. Every view is checked against the storage; a writing op backs the blocks its rows land in, a reading op requires them written.
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
            VulkanKVStorage& s = storage_of(*view.storage);
            if (t.storage && t.storage != &s) throw std::runtime_error("vulkan: views of two storages in one call");
            t.storage = &s;
            const size_t sequence = VulkanKVStorage::add(view.length, view.nq);
            const size_t used = VulkanKVStorage::blocks_for(sequence);
            if (layer >= s.layers() || used > view.n_blocks)
                throw std::runtime_error(writing ? "vulkan: KV write outside the view" : "vulkan: attention outside the KV view");
            if (writing) {
                for (size_t tk = view.length; tk < sequence; tk += kVkBlockTokens - tk % kVkBlockTokens)
                    s.ensure((size_t)view.blocks[tk / kVkBlockTokens]);
                if (view.nq) s.ensure((size_t)view.blocks[(sequence - 1) / kVkBlockTokens]);
            } else {
                for (size_t b = 0; b < used; ++b)
                    if (!s.backed((size_t)view.blocks[b]))
                        throw std::runtime_error("vulkan: attention over unwritten KV blocks");
            }
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
    static void check_storage(const VulkanKVStorage& s, size_t layer, int n_head_kv, int head_dim) {
        if (layer >= s.layers() || (size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads())
            throw std::runtime_error("vulkan: attention outside the KV view");
    }

    // Buffers a storage growth copy reads from, kept until the command buffer that recorded the copy retires.
    void keep_until_retired(BufferPtr b) {
        open();
        pending_[ring_index_].push_back(std::dynamic_pointer_cast<VulkanBuffer>(b));
    }

private:
    static const uint32_t kRing = 4;
    uint32_t chunk_ = 0;
    static const size_t kStagingBytes = size_t(64) << 20;
    static const size_t kArenaBytes = size_t(1) << 20;
    static const uint32_t kPushBytes = 128;

    struct Arena {
        std::unique_ptr<VulkanBuffer> buffer;
        size_t used = 0;
    };

    static VulkanKVStorage& storage_of(KVStorage& storage) {
        auto* s = dynamic_cast<VulkanKVStorage*>(&storage);
        if (!s) throw std::runtime_error("vulkan: KV storage of another backend");
        return *s;
    }

    // Bytes per row of a matrix type the kernels decode, zero for one they do not; and the values per block.
    static size_t block_values_of(uint32_t type) {
        return type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K
                   ? gguf::Q6_K_BLOCK : type == gguf::GGML_TYPE_F32 ? 1 : 32;
    }
    static size_t row_bytes_of(uint32_t type, size_t nin) {
        switch (type) {
        case gguf::GGML_TYPE_F32: return nin * sizeof(float);
        case gguf::GGML_TYPE_Q8_0: return (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE;
        case gguf::GGML_TYPE_Q4_0: return (nin / gguf::Q4_0_BLOCK) * gguf::Q4_0_TYPESIZE;
        case gguf::GGML_TYPE_Q4_1: return (nin / gguf::Q4_1_BLOCK) * gguf::Q4_1_TYPESIZE;
        case gguf::GGML_TYPE_Q4_K: return (nin / gguf::Q4_K_BLOCK) * gguf::Q4_K_TYPESIZE;
        case gguf::GGML_TYPE_Q5_K: return (nin / gguf::Q5_K_BLOCK) * gguf::Q5_K_TYPESIZE;
        case gguf::GGML_TYPE_Q6_K: return (nin / gguf::Q6_K_BLOCK) * gguf::Q6_K_TYPESIZE;
        default: return 0;
        }
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
            pending_[ring_index_].push_back(std::shared_ptr<VulkanBuffer>(std::move(a.buffer)));
            a.buffer = std::make_unique<VulkanBuffer>(dev_, kArenaBytes, true);
            a.used = 0;
        }
        std::memcpy((uint8_t*)a.buffer->mapped() + a.used, data, bytes);
        const VkDescriptorBufferInfo info{a.buffer->handle(), a.used, bytes};
        a.used += need;
        return info;
    }

    Kernel& kernel(KernelId id, int variant = 0) {
        Kernel& k = kernels_[id][variant];
        if (k.pipeline) return k;
        Device& d = *dev_;
        const KernelSource& src = kKernels[id];
        VkShaderModuleCreateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = src.bytes;
        mi.pCode = src.words;
        check(d.fn.vkCreateShaderModule(d.device, &mi, nullptr, &k.module), "vkCreateShaderModule");
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
        check(d.fn.vkCreateDescriptorSetLayout(d.device, &li, nullptr, &k.set_layout),
              "vkCreateDescriptorSetLayout");
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
        VkPipelineLayoutCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pi.setLayoutCount = 1;
        pi.pSetLayouts = &k.set_layout;
        pi.pushConstantRangeCount = 1;
        pi.pPushConstantRanges = &range;
        check(d.fn.vkCreatePipelineLayout(d.device, &pi, nullptr, &k.layout), "vkCreatePipelineLayout");
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = k.module;
        ci.stage.pName = "main";
        // The tile kernels take their row count as specialization constant 0 and the row kernels their column count.
        const bool tile = id == K_MATMUL_TILE || id == K_MATMUL_TILE_TALL || id == K_MATMUL_TILE_Q ||
                          id == K_MATMUL_TILE_Q_TALL || id == K_MATMUL_TILE_Q6 || id == K_MATMUL_TILE_Q6_TALL;
        const bool tall_tile = id == K_MATMUL_TILE_TALL || id == K_MATMUL_TILE_Q_TALL || id == K_MATMUL_TILE_Q6_TALL;
        const uint32_t spec_value = tile ? (tall_tile ? kTileRowsTall : variant ? kTileRowsSmall : kTileRowsShort)
                                         : (variant ? kRowColsOne : kRowColsWide);
        // Constant 7 selects a producer's build that also writes the 8-bit twin; every pipeline gets both entries, and a module that declares neither ignores them.
        const uint32_t spec_data[2] = {spec_value, variant ? 1u : 0u};
        const VkSpecializationMapEntry entries[2] = {{0, 0, sizeof(uint32_t)}, {7, sizeof(uint32_t), sizeof(uint32_t)}};
        VkSpecializationInfo spec{};
        spec.mapEntryCount = 2;
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
        return k;
    }

    // One dispatch: bind the pipeline, push the buffers and constants, launch the workgroups, and fence it off from the next command.
    void dispatch(KernelId id, std::initializer_list<VkDescriptorBufferInfo> buffers,
                  const void* push, size_t push_bytes, uint32_t groups_x, uint32_t groups_y = 1,
                  int variant = 0) {
        Kernel& k = kernel(id, variant);
        if (buffers.size() != k.buffers) throw std::logic_error("vulkan: kernel binding count");
        // Routing and grouping write only ids, weights and their own table; route_experts checks those against the twin itself.
        if (!is_row_kernel(id) && id != K_QUANTIZE_X && id != K_RMS_NORM_ROWS && id != K_SILU_MUL &&
            id != K_MOE_ROUTE && id != K_MOE_GROUP)
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
                check(dev_->fn.vkCreateQueryPool(dev_->device, &qp, nullptr, &queries_), "vkCreateQueryPool");
                dev_->fn.vkCmdResetQueryPool(cmd, queries_, 0, kQueries);
            }
            dev_->fn.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, query_next_);
            dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
            dev_->fn.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, query_next_ + 1);
            query_kernel_.push_back(id * kVariants + variant);
            query_next_ += 2;
        } else {
            dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
        }
        barrier(cmd);
        // A pass is submitted in chunks so the device starts on the first while the host records the rest; the ordered timeline makes the last chunk's ticket cover them all.
        if (++chunk_ >= dev_->profile.dispatch_chunk) submit();
    }

    [[noreturn]] static void todo(const char* op, int substep) {
        throw std::runtime_error(std::string("vulkan: ") + op +
                                 " is not implemented yet (docs/VULKAN.md sub-step " +
                                 std::to_string(substep) + ")");
    }

    // The open command buffer, beginning the next ring slot once its last submission has retired.
    VkCommandBuffer open() {
        if (open_) return ring_[ring_index_];
        wait(ring_ticket_[ring_index_]);
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

    // A pass is a chain, so one barrier between consecutive commands is correct. It names only the compute and transfer stages, since on this driver a barrier over all commands is a full flush.
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
    // It returns once the source is consumed; the copies are in stream order, ahead of whatever reads the destination.
    void upload(VulkanBuffer& dst, size_t off, const void* src, size_t bytes) {
        if (!bytes) return;
        VulkanBuffer& st = staging();
        const size_t half = st.size() / 2;
        size_t done = 0;
        for (size_t i = 0; done < bytes; i ^= 1) {
            const size_t n = std::min(bytes - done, half);
            wait(staged_[i]);
            std::memcpy((uint8_t*)st.mapped() + i * half, (const uint8_t*)src + done, n);
            VkCommandBuffer cmd = open();
            barrier(cmd);
            VkBufferCopy region{i * half, off + done, n};
            dev_->fn.vkCmdCopyBuffer(cmd, st.handle(), dst.handle(), 1, &region);
            barrier(cmd);
            staged_[i] = submit();
            done += n;
        }
    }

    std::shared_ptr<Device> dev_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer ring_[kRing] = {};
    Ticket ring_ticket_[kRing] = {};
    uint32_t ring_index_ = 0;
    bool open_ = false;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    Ticket last_ticket_ = 0;
    std::unique_ptr<VulkanBuffer> staging_;
    Ticket staged_[2] = {};                   // the last copy out of each half of staging
    std::shared_ptr<VulkanBuffer> scratch_;   // attention split states; stream-ordered reuse
    VkQueryPool queries_ = VK_NULL_HANDLE;    // timestamps, only for a diagnostics backend
    static const uint32_t kQueries = 8192;    // two per dispatch, reset each submission
    uint32_t query_next_ = 0;
    std::vector<int> query_kernel_;   // id * kVariants + variant
    double kernel_ns_[K_COUNT * kVariants] = {0};
    size_t kernel_calls_[K_COUNT * kVariants] = {0};
    std::shared_ptr<VulkanBuffer> x8_;        // the integer-dot tile's 8-bit activations
    std::shared_ptr<VulkanBuffer> parts_;     // a split integer-dot tile call's partial sums
    std::shared_ptr<VulkanBuffer> xq_;        // the row kernel's quantized activations; likewise
    bool logits_ = false;                     // inside matmul_logits
    std::shared_ptr<VulkanBuffer> moe_out_;   // a routed down projection's slots before they are combined
    std::shared_ptr<VulkanBuffer> moe_tab_;   // a routed tile call's grouping (shaders/moe_group.comp)
    // Which ids moe_tab_ groups: their location and count, cleared by every routing and by anything that writes a buffer from the host.
    struct GroupTag { VkDescriptorBufferInfo ids{}; size_t entries = 0, n_expert = 0, chunk = 0; };
    GroupTag group_tag_;
    // What the twin buffer holds: the float input it was made from, its length, and whether the 8-bit twin was written; cleared by anything else that writes a buffer, since the input may be what was written.
    struct XqTag { VkDescriptorBufferInfo x{}; size_t n = 0; bool has8 = false; };
    // Set once a matmul that reads the 8-bit twin has run, so producers take their build that writes it from then on.
    bool want_x8_ = false;
    int twin_variant() const { return want_x8_ ? 1 : 0; }
    XqTag xq_tag_;
    std::vector<std::shared_ptr<VulkanBuffer>> pending_[kRing];
    Arena arena_[kRing];
    Kernel kernels_[K_COUNT][kVariants];
};

inline KernelId kv_variant(KernelId f32, KernelId k16, const VulkanKVStorage& s) {
    const int i = (s.k_type() == KVType::f16 ? 1 : 0) + (s.v_type() == KVType::f16 ? 2 : 0);
    return i == 0 ? f32 : (KernelId)((int)k16 + i - 1);
}

void VulkanKVStorage::ensure(size_t id) {
    if (id < backed_) return;
    if (id >= max_) throw std::runtime_error("vulkan: KV block outside the budget");
    const size_t want = std::max(id + 1, std::min(max_, backed_ * 2));
    const size_t kbytes = mul(want, k_block_bytes()), vbytes = mul(want, v_block_bytes());
    const size_t held = mul(add(kbytes, vbytes), k_.size());
    std::vector<BufferPtr> nk(k_.size()), nv(v_.size());
    for (size_t l = 0; l < k_.size(); ++l) {
        nk[l] = owner_->alloc(kbytes, Memory::device);
        nv[l] = owner_->alloc(vbytes, Memory::device);
        if (k_[l]) {
            owner_->copy(*nk[l], 0, *k_[l], 0, k_[l]->size());
            owner_->copy(*nv[l], 0, *v_[l], 0, v_[l]->size());
        }
    }
    peak_ = std::max(peak_, add(allocated_bytes(), held));
    for (size_t l = 0; l < k_.size(); ++l) {
        if (k_[l]) { owner_->keep_until_retired(k_[l]); owner_->keep_until_retired(v_[l]); }
    }
    k_.swap(nk);
    v_.swap(nv);
    backed_ = want;
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
