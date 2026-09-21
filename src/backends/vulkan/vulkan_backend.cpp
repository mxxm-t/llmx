// Vulkan backend, sub-step 1 of docs/VULKAN.md: storage and submission.
// Everything runs on one compute queue. Ops record into an open command
// buffer; submit() ends it and signals a timeline semaphore with the ticket
// value, wait() blocks on that value, and a ring of command buffers is
// reused once their tickets have retired.
#include "backends/vulkan/vulkan_backend.hpp"
#include "format/gguf.hpp"

#include <algorithm>
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

// Every entry point this file uses, fetched through the loader at run time
// so nothing links against vulkan-1.
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
    X(vkCmdDispatch)

struct Fn {
#define LLMX_VK_DECLARE(name) PFN_##name name = nullptr;
    LLMX_VK_GLOBAL_FUNCTIONS(LLMX_VK_DECLARE)
    LLMX_VK_INSTANCE_FUNCTIONS(LLMX_VK_DECLARE)
    LLMX_VK_DEVICE_FUNCTIONS(LLMX_VK_DECLARE)
#undef LLMX_VK_DECLARE
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR = nullptr;
};

// The kernels, compiled by glslc at build time (CMakeLists.txt) into the
// generated include directory as comma-separated words.
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
const uint32_t kSpvKvWrite[] = {
#include "vulkan/kv_write.inc"
};
const uint32_t kSpvAttention[] = {
#include "vulkan/attention.inc"
};
const uint32_t kSpvAttentionMerge[] = {
#include "vulkan/attention_merge.inc"
};
const uint32_t kSpvMatmulTile[] = {
#include "vulkan/matmul_tile.inc"
};

enum KernelId { K_ADD, K_SILU_MUL, K_GATHER_ROWS, K_RMS_NORM_ROWS, K_NORM_ROPE_ROWS, K_EMBED,
                K_MATMUL_ROW, K_KV_WRITE, K_ATTENTION, K_ATTENTION_MERGE, K_MATMUL_TILE, K_COUNT };

struct KernelSource {
    const uint32_t* words;
    size_t bytes;
    uint32_t bindings;
};

const KernelSource kKernels[K_COUNT] = {
    {kSpvAdd, sizeof(kSpvAdd), 2},
    {kSpvSiluMul, sizeof(kSpvSiluMul), 3},
    {kSpvGatherRows, sizeof(kSpvGatherRows), 3},
    {kSpvRmsNormRows, sizeof(kSpvRmsNormRows), 3},
    {kSpvNormRopeRows, sizeof(kSpvNormRopeRows), 5},
    {kSpvEmbed, sizeof(kSpvEmbed), 4},
    {kSpvMatmulRow, sizeof(kSpvMatmulRow), 6},
    {kSpvKvWrite, sizeof(kSpvKvWrite), 5},
    {kSpvAttention, sizeof(kSpvAttention), 6},
    {kSpvAttentionMerge, sizeof(kSpvAttentionMerge), 2},
    {kSpvMatmulTile, sizeof(kSpvMatmulTile), 4},
};

// 64 tokens per KV block: half the CPU's, since the attention workgroup
// reads a block per iteration and a smaller block wastes less tail per
// sequence on the device that bounds concurrency. Screened on the real
// models before it is fixed (docs/VULKAN.md).
const size_t kVkBlockTokens = 64;

class VulkanBackend;

// KV blocks on the device, one K buffer and one V buffer per layer. Block
// id b starts at b*block_floats() and holds [kv_head][token][head_dim].
// Blocks are backed in doubling steps as ids are first written. Growth is
// allocate and copy on the queue, and the buffers the copy reads from are
// kept alive until it has retired.
class VulkanKVStorage final : public KVStorage {
public:
    VulkanKVStorage(VulkanBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_blocks)
        : owner_(&owner), heads_(heads), dim_(dim), max_(max_blocks), k_(layers), v_(layers) {
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
    bool backed(size_t id) const { return id < backed_; }
    void ensure(size_t id);
    const BufferPtr& k(size_t layer) const { return k_[layer]; }
    const BufferPtr& v(size_t layer) const { return v_[layer]; }

private:
    VulkanBackend* owner_;
    size_t heads_, dim_, max_, backed_ = 0, peak_ = 0;
    std::vector<BufferPtr> k_, v_;
};

// A compiled kernel: module, layout with `bindings` storage buffers pushed
// per dispatch and 128 bytes of push constants, and the pipeline.
struct Kernel {
    VkShaderModule module = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    uint32_t bindings = 0;
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

// The device and everything a buffer needs to free itself. Buffers hold a
// shared handle to it, so a buffer that outlives its backend still frees
// correctly and the device is destroyed after the last buffer.
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
    std::string name;
    bool push_descriptor = false;
    bool int8 = false, float16 = false, storage8 = false, storage16 = false;

    // Guarded per handle: construction can fail between creating a handle
    // and loading the function that destroys it.
    ~Device() {
        if (device && fn.vkDestroyDevice) {
            if (fn.vkDeviceWaitIdle) fn.vkDeviceWaitIdle(device);
            fn.vkDestroyDevice(device, nullptr);
        }
        if (instance && fn.vkDestroyInstance) fn.vkDestroyInstance(instance, nullptr);
    }

    // A memory type with every flag in `required`, preferring `preferred`
    // on top, among those the buffer allows.
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
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(dev_->fn.vkCreateBuffer(dev_->device, &bi, nullptr, &buffer_), "vkCreateBuffer");
        VkMemoryRequirements req{};
        dev_->fn.vkGetBufferMemoryRequirements(dev_->device, buffer_, &req);
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        // Device memory prefers not to be host visible, so it comes from the
        // large device-local heap rather than the BAR window. Host-visible
        // memory prefers to be cached, because the host reads it in bulk.
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
    explicit VulkanBackend(int index) : dev_(std::make_shared<Device>()) {
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
        check(fn.vkCreateInstance(&ii, nullptr, &d.instance), "vkCreateInstance");
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
        fn.vkGetPhysicalDeviceProperties2(d.physical, &p2);
        d.subgroup_size = sg.subgroupSize;
        // The row kernel places one subgroup per row inside a workgroup of
        // 256, which needs the subgroup size to divide it.
        if (!d.subgroup_size || 256 % d.subgroup_size ||
            !(sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT))
            throw std::runtime_error("vulkan: " + std::string(d.props.deviceName) +
                                     " has an unsupported subgroup size or no subgroup arithmetic");
        d.name = d.props.deviceName;
        if (d.props.apiVersion < VK_API_VERSION_1_2)
            throw std::runtime_error("vulkan: " + d.name + " is older than Vulkan 1.2");

        // A compute family without graphics keeps the queue clear of the
        // desktop; any compute family will do.
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
        if (chosen < 0) throw std::runtime_error("vulkan: " + d.name + " has no compute queue");
        d.queue_family = (uint32_t)chosen;

        // Timeline semaphores are what submit and wait are built on. The
        // 8- and 16-bit storage and arithmetic features are what the kernels
        // will read quantized blocks and half scales with; they are enabled
        // now where present so device creation does not change per sub-step.
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
            throw std::runtime_error("vulkan: " + d.name + " has no timeline semaphores");
        d.int8 = f12.shaderInt8;
        d.float16 = f12.shaderFloat16;
        d.storage8 = f12.storageBuffer8BitAccess;
        d.storage16 = f11.storageBuffer16BitAccess;
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
            }

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
        if (!d.push_descriptor)
            throw std::runtime_error("vulkan: " + d.name + " has no VK_KHR_push_descriptor");
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
        for (Kernel& k : kernels_) {
            if (k.pipeline) d.fn.vkDestroyPipeline(d.device, k.pipeline, nullptr);
            if (k.layout) d.fn.vkDestroyPipelineLayout(d.device, k.layout, nullptr);
            if (k.set_layout) d.fn.vkDestroyDescriptorSetLayout(d.device, k.set_layout, nullptr);
            if (k.module) d.fn.vkDestroyShaderModule(d.device, k.module, nullptr);
        }
        staging_.reset();
        if (timeline_) d.fn.vkDestroySemaphore(d.device, timeline_, nullptr);
        if (pool_) d.fn.vkDestroyCommandPool(d.device, pool_, nullptr);
    }

    const std::string& name() const { return dev_->name; }

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

    // A copy, in chunks through staging, each submitted and waited. Weights
    // arrive here once at load; nothing can run before they are there.
    BufferPtr adopt(const void* src, size_t bytes) override {
        if (!src && bytes) throw std::runtime_error("vulkan: adopting null storage");
        auto b = std::make_shared<VulkanBuffer>(dev_, bytes, false);
        upload(*b, 0, src, bytes);
        return b;
    }

    Ticket submit() override {
        VkCommandBuffer cmd = open();
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

    // noexcept by contract: a device that cannot report that its work has
    // finished has been lost, and nothing at this layer can act on that.
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
        const uint32_t pc[1] = {u32(n)};
        dispatch(K_SILU_MUL, {bind(dst), bind(gate), bind(up)}, pc, sizeof(pc), groups(n, 256));
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
        struct { uint32_t rows, n, stride; float eps; } pc{u32(rows), u32(n), u32(stride), eps};
        dispatch(K_RMS_NORM_ROWS, {bind(dst), bind(src), bind(w)}, &pc, sizeof(pc), u32(rows));
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

    void embed(Slice dst, uint32_t type, CSlice table, size_t nin, size_t nrows,
               const uint32_t* ids, size_t count) override {
        if (count && !ids) throw std::runtime_error("vulkan: embed without ids");
        if (!count || !nin) return;
        if (type != gguf::GGML_TYPE_F32 && type != gguf::GGML_TYPE_Q8_0)
            throw std::runtime_error("vulkan: unsupported embedding type");
        if (type == gguf::GGML_TYPE_Q8_0 && nin % gguf::Q8_0_BLOCK)
            throw std::runtime_error("vulkan: embedding width is not whole blocks");
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= nrows) throw std::runtime_error("vulkan: embedding row out of range");
        const uint32_t pc[3] = {u32(nin), u32(count), type};
        // The table is bound twice: as floats for F32 rows, as bytes for
        // block formats. The shader reads the one the type selects.
        dispatch(K_EMBED, {bind(dst), bind(table), bind(table), args(ids, count * sizeof(uint32_t))},
                 pc, sizeof(pc), u32(count));
    }

    // One subgroup per output row, columns in chunks of eight so a weight
    // is read once per chunk. Every column beyond the first chunk re-reads
    // the weights; the tile kernel for wide batches is what removes that.
    void matmul(uint32_t type, CSlice w, CSlice X, Slice Y, size_t nin, size_t nout,
                size_t nbatch) override {
        if (!nout || !nbatch) return;
        if (type != gguf::GGML_TYPE_F32 && type != gguf::GGML_TYPE_Q8_0)
            throw std::runtime_error("vulkan: unsupported matrix type " + std::to_string(type) +
                                     " (docs/VULKAN.md sub-step 6)");
        if (type == gguf::GGML_TYPE_Q8_0 && nin % gguf::Q8_0_BLOCK)
            throw std::runtime_error("vulkan: matrix width is not whole blocks");
        const size_t row_bytes = type == gguf::GGML_TYPE_Q8_0
            ? (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE : nin * sizeof(float);
        if (bytes_from(w) < nout * row_bytes ||
            floats_from(X) < nbatch * nin || floats_from(Y) < nbatch * nout)
            throw std::runtime_error("vulkan: matmul operand outside its allocation");
        // Wide batches go to the tile kernel, which reads a weight once per
        // pass; the row kernel below reads it once per eight columns.
        if (nbatch >= 16) {
            const uint32_t pc[4] = {u32(nin), u32(nout), u32(nbatch), type};
            const uint32_t gx = groups(nout, 64);
            const size_t gy = (nbatch + 63) / 64;
            if (gy > dev_->props.limits.maxComputeWorkGroupCount[1])
                throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
            dispatch(K_MATMUL_TILE, {bind(Y), bind(w), bind(w), bind(X)}, pc, sizeof(pc), gx, (uint32_t)gy);
            return;
        }
        // The word-wide path needs every row to start on a word boundary,
        // which an even block count gives, and X columns on 16 bytes, which
        // a column width that is whole blocks gives.
        const uint32_t wide = type == gguf::GGML_TYPE_Q8_0 && (nin / gguf::Q8_0_BLOCK) % 2 == 0 &&
                              X.offset % 4 == 0 ? 1u : 0u;
        // A row's work units: block pairs, blocks, or floats. A cluster of
        // lanes takes one row, sized to the units so a narrow row does not
        // idle most of a subgroup, and a subgroup takes several rows.
        const size_t units = type == gguf::GGML_TYPE_Q8_0
            ? (wide ? nin / gguf::Q8_0_BLOCK / 2 : nin / gguf::Q8_0_BLOCK) : nin;
        uint32_t cluster = 1;
        while (cluster < dev_->subgroup_size && cluster < units) cluster *= 2;
        const uint32_t rows_per_sg = dev_->subgroup_size / cluster;
        const uint32_t rows_per_group = (256 / dev_->subgroup_size) * rows_per_sg;
        const uint32_t g = groups(nout, rows_per_group);
        for (size_t col0 = 0; col0 < nbatch; col0 += 8) {
            const size_t ncols = std::min<size_t>(8, nbatch - col0);
            const uint32_t pc[9] = {u32(nin), u32(nout), u32(nbatch), type, u32(col0), u32(ncols), wide,
                                    cluster, rows_per_sg};
            dispatch(K_MATMUL_ROW, {bind(Y), bind(w), bind(w), bind(w), bind(X), bind(X)},
                     pc, sizeof(pc), g);
        }
    }
    KVLayout kv_layout() const override { return KVLayout{kVkBlockTokens}; }

    std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim,
                                        size_t max_tokens) override {
        if (!layers || !n_head_kv || !head_dim)
            throw std::runtime_error("vulkan: KV storage without layers, heads or width");
        if (head_dim > 256) throw std::runtime_error("vulkan: head width above 256 is not supported");
        return std::make_unique<VulkanKVStorage>(*this, layers, n_head_kv, head_dim,
                                                 VulkanKVStorage::blocks_for(max_tokens));
    }

    void kv_copy(KVStorage& storage, int32_t src, int32_t dst) override {
        VulkanKVStorage& s = storage_of(storage);
        if (src < 0 || dst < 0 || !s.backed((size_t)src) || (size_t)dst >= s.max_blocks())
            throw std::runtime_error("vulkan: KV copy outside the storage");
        s.ensure((size_t)dst);
        const size_t bytes = s.block_floats() * sizeof(float);
        for (size_t l = 0; l < s.layers(); ++l) {
            copy(*s.k(l), (size_t)dst * bytes, *s.k(l), (size_t)src * bytes, bytes);
            copy(*s.v(l), (size_t)dst * bytes, *s.v(l), (size_t)src * bytes, bytes);
        }
    }

    // One dispatch per view: its rows scatter into its blocks.
    void kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k,
                  CSlice v) override {
        if (n_views && !views) throw std::runtime_error("vulkan: KV write without views");
        size_t row0 = 0;
        for (size_t i = 0; i < n_views; ++i) {
            const KVView& view = views[i];
            VulkanKVStorage& s = storage_of(*view.storage);
            const size_t hd = s.heads() * s.dim();
            if (layer >= s.layers() ||
                VulkanKVStorage::blocks_for(VulkanKVStorage::add(view.length, view.nq)) > view.n_blocks)
                throw std::runtime_error("vulkan: KV write outside the view");
            if (view.nq) {
                if (floats_from(k) < (row0 + view.nq) * hd || floats_from(v) < (row0 + view.nq) * hd)
                    throw std::runtime_error("vulkan: KV rows outside their allocation");
                for (size_t t = view.length; t < view.length + view.nq; t += kVkBlockTokens - t % kVkBlockTokens)
                    s.ensure((size_t)view.blocks[t / kVkBlockTokens]);
                s.ensure((size_t)view.blocks[(view.length + view.nq - 1) / kVkBlockTokens]);
                const uint32_t pc[6] = {u32(view.length), u32(view.nq), u32(s.heads()), u32(s.dim()),
                                        u32(kVkBlockTokens), u32(row0)};
                dispatch(K_KV_WRITE,
                         {bind(CSlice{s.k(layer).get(), 0}), bind(CSlice{s.v(layer).get(), 0}),
                          bind(k), bind(v), args(view.blocks, view.n_blocks * sizeof(int32_t))},
                         pc, sizeof(pc), groups(view.nq * hd, 256));
            }
            row0 += view.nq;
        }
    }

    // One dispatch per view, one workgroup per (row, head).
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out,
                   int n_head, int n_head_kv, int head_dim) override {
        if (n_views && !views) throw std::runtime_error("vulkan: attention without views");
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > 256)
            throw std::runtime_error("vulkan: invalid attention dimensions");
        const size_t qstride = (size_t)n_head * head_dim;
        const float scale = 1.0f / std::sqrt((float)head_dim);
        size_t row0 = 0;
        for (size_t i = 0; i < n_views; ++i) {
            const KVView& view = views[i];
            if (!view.nq) throw std::runtime_error("vulkan: invalid attention dimensions");
            VulkanKVStorage& s = storage_of(*view.storage);
            const size_t sequence = VulkanKVStorage::add(view.length, view.nq);
            const size_t blocks = VulkanKVStorage::blocks_for(sequence);
            if (layer >= s.layers() || (size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads() ||
                blocks > view.n_blocks)
                throw std::runtime_error("vulkan: attention outside the KV view");
            for (size_t b = 0; b < blocks; ++b)
                if (!s.backed((size_t)view.blocks[b]))
                    throw std::runtime_error("vulkan: attention over unwritten KV blocks");
            if (floats_from(Q) < (row0 + view.nq) * qstride || floats_from(out) < (row0 + view.nq) * qstride)
                throw std::runtime_error("vulkan: attention rows outside their allocation");
            // A decode token has few (row, head) pairs, so the history is
            // split into chunks of 32 tokens across workgroups, enough to
            // fill the device, capped at 64 splits; a wide pass already
            // has the workgroups and takes one split.
            const size_t pairs = view.nq * (size_t)n_head;
            size_t nsplit = 1;
            if (pairs < 256) nsplit = std::min<size_t>(64, std::max<size_t>(1, (sequence + 31) / 32));
            const size_t chunk = (sequence + nsplit - 1) / nsplit;
            nsplit = (sequence + chunk - 1) / chunk;
            const size_t scratch_floats = nsplit > 1 ? pairs * nsplit * ((size_t)head_dim + 2) : 0;
            if (scratch_floats && (!scratch_ || scratch_->size() < scratch_floats * sizeof(float)))
                scratch_ = std::make_shared<VulkanBuffer>(dev_, scratch_floats * sizeof(float), false);
            struct { uint32_t length, nq, n_head, n_head_kv, dim, bt, row0; float scale; uint32_t nsplit, chunk; }
                pc{u32(view.length), u32(view.nq), (uint32_t)n_head, (uint32_t)n_head_kv,
                   (uint32_t)head_dim, u32(kVkBlockTokens), u32(row0), scale, u32(nsplit), u32(chunk)};
            const VkDescriptorBufferInfo scratch = scratch_
                ? VkDescriptorBufferInfo{scratch_->handle(), 0, VK_WHOLE_SIZE} : bind(out);
            dispatch(K_ATTENTION,
                     {bind(Q), bind(out), bind(CSlice{s.k(layer).get(), 0}), bind(CSlice{s.v(layer).get(), 0}),
                      args(view.blocks, blocks * sizeof(int32_t)), scratch},
                     &pc, sizeof(pc), groups(pairs * nsplit, 1));
            if (nsplit > 1) {
                const uint32_t mc[5] = {u32(view.nq), (uint32_t)n_head, (uint32_t)head_dim, u32(row0), u32(nsplit)};
                dispatch(K_ATTENTION_MERGE, {bind(out), scratch}, mc, sizeof(mc), groups(pairs, 1));
            }
            row0 += view.nq;
        }
    }

    // Storage growth hands the buffers a copy reads from here, so they live
    // until the command buffer that recorded the copy has retired.
    void keep_until_retired(BufferPtr b) {
        open();
        pending_[ring_index_].push_back(std::dynamic_pointer_cast<VulkanBuffer>(b));
    }

private:
    static const uint32_t kRing = 4;
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

    // A slice as a storage buffer binding: the buffer at a byte offset of
    // four times the float offset, which the device's 4-byte alignment
    // allows, through to the end of the allocation. An empty allocation
    // binds nothing and nothing reads it.
    VkDescriptorBufferInfo bind(CSlice s) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        const VulkanBuffer& b = as_vulkan(*s.buffer);
        const VkDeviceSize off = (VkDeviceSize)s.offset * sizeof(float);
        if (off > b.size()) throw std::runtime_error("vulkan: operand outside the allocation");
        if (!b.size()) return VkDescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
        return VkDescriptorBufferInfo{b.handle(), off, VK_WHOLE_SIZE};
    }
    VkDescriptorBufferInfo bind(Slice s) { return bind(CSlice(s)); }

    // Small per-call inputs the host holds, ids and positions and row lists,
    // go to the device through a host-visible arena per ring slot, bumped
    // per call and reset when the slot's command buffer has retired. An
    // allocation per call was over a hundred vkAllocateMemory calls per
    // decoded token, which was most of the token on Qwen3-0.6B. A call
    // larger than the arena gets its own buffer, kept the same way.
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
            // The slot's arena is full before its command buffer retired;
            // the overflow gets a second arena kept alongside, and the
            // slot starts a fresh one.
            pending_[ring_index_].push_back(std::shared_ptr<VulkanBuffer>(std::move(a.buffer)));
            a.buffer = std::make_unique<VulkanBuffer>(dev_, kArenaBytes, true);
            a.used = 0;
        }
        std::memcpy((uint8_t*)a.buffer->mapped() + a.used, data, bytes);
        const VkDescriptorBufferInfo info{a.buffer->handle(), a.used, bytes};
        a.used += need;
        return info;
    }

    Kernel& kernel(KernelId id) {
        Kernel& k = kernels_[id];
        if (k.pipeline) return k;
        Device& d = *dev_;
        const KernelSource& src = kKernels[id];
        VkShaderModuleCreateInfo mi{};
        mi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        mi.codeSize = src.bytes;
        mi.pCode = src.words;
        check(d.fn.vkCreateShaderModule(d.device, &mi, nullptr, &k.module), "vkCreateShaderModule");
        std::vector<VkDescriptorSetLayoutBinding> bindings(src.bindings);
        for (uint32_t i = 0; i < src.bindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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
        ci.layout = k.layout;
        check(d.fn.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &ci, nullptr, &k.pipeline),
              "vkCreateComputePipelines");
        k.bindings = src.bindings;
        return k;
    }

    // One dispatch: bind the pipeline, push the buffers and the constants,
    // launch `groups` workgroups, and fence it off from the next command.
    void dispatch(KernelId id, std::initializer_list<VkDescriptorBufferInfo> buffers,
                  const void* push, size_t push_bytes, uint32_t groups_x, uint32_t groups_y = 1) {
        Kernel& k = kernel(id);
        if (buffers.size() != k.bindings) throw std::logic_error("vulkan: kernel binding count");
        if (push_bytes > kPushBytes) throw std::logic_error("vulkan: push constants exceed 128 bytes");
        for (const auto& b : buffers)
            if (!b.buffer) throw std::runtime_error("vulkan: dispatch over an empty allocation");
        VkCommandBuffer cmd = open();
        std::vector<VkWriteDescriptorSet> writes(buffers.size());
        uint32_t i = 0;
        for (const auto& b : buffers) {
            VkWriteDescriptorSet& w = writes[i];
            w = VkWriteDescriptorSet{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstBinding = i;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &b;
            ++i;
        }
        dev_->fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
        dev_->fn.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0,
                                           (uint32_t)writes.size(), writes.data());
        dev_->fn.vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                    (uint32_t)push_bytes, push);
        dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
        barrier(cmd);
    }

    [[noreturn]] static void todo(const char* op, int substep) {
        throw std::runtime_error(std::string("vulkan: ") + op +
                                 " is not implemented yet (docs/VULKAN.md sub-step " +
                                 std::to_string(substep) + ")");
    }

    // The open command buffer, beginning the next ring slot once its last
    // submission has retired.
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

    // A pass is a chain, so every op reads what the one before wrote: one
    // full barrier between consecutive commands is correct, and tracking
    // which buffers an op touches is an optimization for later.
    // Only the compute and transfer stages ever touch a buffer here, so the
    // barrier names those rather than every stage: on this driver a barrier
    // over all commands is a full flush and idle between every two kernels.
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

    // Host to device through staging, each chunk submitted and waited so the
    // staging buffer can take the next one.
    void upload(VulkanBuffer& dst, size_t off, const void* src, size_t bytes) {
        if (!bytes) return;
        VulkanBuffer& st = staging();
        size_t done = 0;
        while (done < bytes) {
            const size_t n = std::min(bytes - done, st.size());
            std::memcpy(st.mapped(), (const uint8_t*)src + done, n);
            VkCommandBuffer cmd = open();
            barrier(cmd);
            VkBufferCopy region{0, off + done, n};
            dev_->fn.vkCmdCopyBuffer(cmd, st.handle(), dst.handle(), 1, &region);
            barrier(cmd);
            wait(submit());
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
    std::shared_ptr<VulkanBuffer> scratch_;   // attention split states; stream-ordered reuse
    std::vector<std::shared_ptr<VulkanBuffer>> pending_[kRing];
    Arena arena_[kRing];
    Kernel kernels_[K_COUNT];
};

void VulkanKVStorage::ensure(size_t id) {
    if (id < backed_) return;
    if (id >= max_) throw std::runtime_error("vulkan: KV block outside the budget");
    const size_t want = std::max(id + 1, std::min(max_, backed_ * 2));
    const size_t bytes = mul(mul(want, block_floats()), sizeof(float));
    const size_t held = mul(mul(bytes, 2), k_.size());
    std::vector<BufferPtr> nk(k_.size()), nv(v_.size());
    for (size_t l = 0; l < k_.size(); ++l) {
        nk[l] = owner_->alloc(bytes, Memory::device);
        nv[l] = owner_->alloc(bytes, Memory::device);
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

BackendPtr make_vulkan_backend(int device) {
    return std::make_shared<VulkanBackend>(device);
}

std::string vulkan_device_name(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->name() : std::string();
}

} // namespace backend
