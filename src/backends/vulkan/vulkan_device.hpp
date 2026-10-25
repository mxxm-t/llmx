#pragma once
// The Vulkan loader and device: the entry points fetched at run time, the device a backend opens, and what the kernels need of a device.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include "backends/device_profile.hpp"
#include "backends/vulkan/vulkan_backend.hpp"

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace backend {
namespace {

// Every entry point the backend uses, fetched through the loader at run time so nothing links against vulkan-1.
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

// Whether a device of this type allocates from the host's memory, which the host tier shares: an integrated GPU or a CPU device; a discrete GPU's memory is its own, whatever window the host can map.
inline bool host_memory_device(VkPhysicalDeviceType type) { return type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || type == VK_PHYSICAL_DEVICE_TYPE_CPU; }

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

// What the fused routed step needs beyond the kernels' needs, its refusal's text, or empty: its last workgroup routes four rows at once, a row a quarter of 64 lanes, one subgroup or two (shaders/moe_step_route.comp).
// A device without it takes the unfused step.
inline std::string missing_counter_need(uint32_t subgroup_size) {
    if (subgroup_size != 32 && subgroup_size != 64) return "has subgroups other than 32 or 64 lanes, which the fused routing's rows do not fit";
    return {};
}

// Heads 128 and 256 wide take the tiled and vector kernels, whose lane groups fit every subgroup the device check accepts (shaders/attention_tile.comp, shaders/attention_vec.comp).
// Other widths take the per-row kernel, which reads at most four elements of a head per lane of a subgroup (shaders/attention.comp); no head is wider than 256.
inline size_t attention_row_width(uint32_t subgroup_size) { return std::min<size_t>(256, 4 * (size_t)subgroup_size); }
inline bool attention_head_fits(size_t head_dim, uint32_t subgroup_size) { return head_dim == 128 || head_dim == 256 || head_dim <= attention_row_width(subgroup_size); }

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
    bool workgroup_counter = false; // the fused routed step's routing by the last workgroup (missing_counter_need)
    std::string identity;          // the device and the driver that compiles its kernels (Backend::identity)
    bool push_descriptor = false;
    bool memory_budget = false;   // the device reports what is free of each heap (VK_EXT_memory_budget)
    // Host memory imported as device memory, which a copy reads in place (VK_EXT_external_memory_host), and the alignment of its address and size.
    PFN_vkGetMemoryHostPointerPropertiesEXT host_pointer_props = nullptr;
    size_t host_import_align = 0;
    // A tensor group's exchange (docs/TENSOR-SPLIT.md, section 4.3): device memory shared as dma-buf and binary semaphores shared as sync files, which only Linux offers.
    bool exchange = false;
    std::string pci_root;   // the PCI root complex above the device (pci_root_of), empty where the system does not say
    PFN_vkGetMemoryFdKHR get_memory_fd = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR memory_fd_props = nullptr;
    PFN_vkGetSemaphoreFdKHR get_semaphore_fd = nullptr;
    PFN_vkImportSemaphoreFdKHR import_semaphore_fd = nullptr;
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

// The PCI root complex above the device at this address, as Linux's sysfs names it (pci0000:80), or empty where the system does not say, which a tensor group's startup report gives (docs/TENSOR-SPLIT.md, section 4.5).
inline std::string pci_root_of(uint32_t domain, uint32_t bus, uint32_t device, uint32_t function) {
#if defined(__linux__)
    char address[64];
    std::snprintf(address, sizeof address, "/sys/bus/pci/devices/%04x:%02x:%02x.%x", domain, bus, device, function);
    char resolved[4096];
    if (!realpath(address, resolved)) return {};
    const std::string path(resolved);
    const size_t at = path.find("/pci");
    if (at == std::string::npos) return {};
    return path.substr(at + 1, path.find('/', at + 1) - at - 1);
#else
    (void)domain, (void)bus, (void)device, (void)function;
    return {};
#endif
}

} // namespace
} // namespace backend
