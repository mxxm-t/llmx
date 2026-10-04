// Measures what a layer split's handoff can use between two Vulkan devices, before any split is written (docs/MULTI-DEVICE.md, phase 0).
// `probe` lists, per device, the external semaphore and memory handle types, host-pointer import, identifiers and memory budget, and the device groups.
// `time A B` moves a buffer from device A's memory to device B's in each way below and times it, after checking that the way delivers the bytes.
// read-write: A copies to its staging, the host waits, copies into B's staging, and B copies in while the host waits; this is today's crossing.
// imported: one host allocation imported into both devices, A copying into it and B out of it, with the host waiting on each.
// relay: as imported, but B's copy is queued first behind its own timeline, which the host signals once A's has passed.
// sync-fd: as imported, but B waits on a binary semaphore imported from A's as a sync file, with no host wait between them.
// dma-buf: A copies into its own memory exported as a dma-buf and B copies out of its import of it, with the host waiting on each.
// peer-read: only B's copy out of A's exported memory, which is the handoff when A's output buffer is itself the exported one.
// dma+sync-fd: as dma-buf, with B waiting on A's sync file rather than the host waiting between them.
// `pingpong A B` bounces bytes between the two devices through dma-buf, with the host waiting on every hop and with the chain queued ahead through sync files.
// `exchange A,B[,C,D]` is a tensor group's all-reduce on 2 to 4 devices: sync files, the members' arrival spread and a flag wait under the Vulkan memory model (docs/TENSOR-SPLIT.md, step 0).
// Usage: llmx-vk-handoff probe | llmx-vk-handoff time A B [iterations] | llmx-vk-handoff pingpong A B [hops] | llmx-vk-handoff exchange A,B[,C,D] [epochs] [device|host]
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

#define GLOBAL_FNS(X) X(vkCreateInstance)
#define INSTANCE_FNS(X) \
    X(vkDestroyInstance) \
    X(vkEnumeratePhysicalDevices) \
    X(vkEnumeratePhysicalDeviceGroups) \
    X(vkGetPhysicalDeviceProperties) \
    X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkGetPhysicalDeviceMemoryProperties2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkEnumerateDeviceExtensionProperties) \
    X(vkGetPhysicalDeviceExternalSemaphoreProperties) \
    X(vkGetPhysicalDeviceExternalBufferProperties) \
    X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)
#define DEVICE_FNS(X) \
    X(vkDestroyDevice) \
    X(vkGetDeviceQueue) \
    X(vkDeviceWaitIdle) \
    X(vkCreateCommandPool) \
    X(vkDestroyCommandPool) \
    X(vkAllocateCommandBuffers) \
    X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) \
    X(vkQueueSubmit) \
    X(vkCreateSemaphore) \
    X(vkDestroySemaphore) \
    X(vkWaitSemaphores) \
    X(vkSignalSemaphore) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
    X(vkCmdCopyBuffer) \
    X(vkCmdPipelineBarrier) \
    X(vkGetDeviceGroupPeerMemoryFeatures) \
    X(vkCreateShaderModule) \
    X(vkCreateDescriptorSetLayout) \
    X(vkCreatePipelineLayout) \
    X(vkCreateComputePipelines) \
    X(vkCreateDescriptorPool) \
    X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets) \
    X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) \
    X(vkCmdPushConstants) \
    X(vkCmdDispatch) \
    X(vkCreateQueryPool) \
    X(vkCmdResetQueryPool) \
    X(vkCmdWriteTimestamp) \
    X(vkGetQueryPoolResults) \
    X(vkCreateEvent) \
    X(vkCmdSetEvent) \
    X(vkCmdWaitEvents) \
    X(vkGetEventStatus) \
    X(vkSetEvent)

#define DECLARE(name) PFN_##name name = nullptr;
PFN_vkGetInstanceProcAddr get_instance_proc = nullptr;
GLOBAL_FNS(DECLARE)
INSTANCE_FNS(DECLARE)
#undef DECLARE

void check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) throw std::runtime_error(std::string(what) + " failed: VkResult " + std::to_string((int)r));
}

void open_loader() {
#if defined(_WIN32)
    HMODULE lib = LoadLibraryA("vulkan-1.dll");
    if (!lib) throw std::runtime_error("no Vulkan loader");
    get_instance_proc = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(lib, "vkGetInstanceProcAddr");
#else
    void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) throw std::runtime_error("no Vulkan loader");
    get_instance_proc = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#endif
    if (!get_instance_proc) throw std::runtime_error("the loader has no vkGetInstanceProcAddr");
#define LOAD(name) name = (PFN_##name)get_instance_proc(nullptr, #name);
    GLOBAL_FNS(LOAD)
#undef LOAD
}

VkInstance make_instance() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "llmx-vk-handoff";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    check(vkCreateInstance(&ci, nullptr, &inst), "vkCreateInstance");
#define LOAD(name) name = (PFN_##name)get_instance_proc(inst, #name);
    INSTANCE_FNS(LOAD)
#undef LOAD
    return inst;
}

std::vector<VkPhysicalDevice> physical_devices(VkInstance inst) {
    uint32_t n = 0;
    check(vkEnumeratePhysicalDevices(inst, &n, nullptr), "vkEnumeratePhysicalDevices");
    std::vector<VkPhysicalDevice> v(n);
    check(vkEnumeratePhysicalDevices(inst, &n, v.data()), "vkEnumeratePhysicalDevices");
    return v;
}

std::vector<std::string> extensions(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, e.data());
    std::vector<std::string> names;
    for (auto& x : e) names.push_back(x.extensionName);
    return names;
}

bool has(const std::vector<std::string>& exts, const char* name) {
    return std::find(exts.begin(), exts.end(), name) != exts.end();
}

std::string hex(const uint8_t* p, size_t n) {
    std::string s;
    char b[3];
    for (size_t i = 0; i < n; ++i) {
        std::snprintf(b, sizeof b, "%02x", p[i]);
        s += b;
    }
    return s;
}

struct Identity {
    std::string name, driver, device_uuid, driver_uuid, pci;
};

Identity identity(VkPhysicalDevice pd, const std::vector<std::string>& exts) {
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceDriverProperties drv{};
    drv.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDevicePCIBusInfoPropertiesEXT pci{};
    pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    id.pNext = &drv;
    if (has(exts, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) drv.pNext = &pci;
    VkPhysicalDeviceProperties2 p{};
    p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p.pNext = &id;
    vkGetPhysicalDeviceProperties2(pd, &p);
    Identity r;
    r.name = p.properties.deviceName;
    r.driver = std::string(drv.driverName) + " " + drv.driverInfo;
    r.device_uuid = hex(id.deviceUUID, VK_UUID_SIZE);
    r.driver_uuid = hex(id.driverUUID, VK_UUID_SIZE);
    if (has(exts, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)) {
        char b[32];
        std::snprintf(b, sizeof b, "%04x:%02x:%02x.%x", pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        r.pci = b;
    }
    return r;
}

std::string semaphore_features(VkPhysicalDevice pd, VkExternalSemaphoreHandleTypeFlagBits type, bool timeline) {
    VkSemaphoreTypeCreateInfo st{};
    st.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    st.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    VkPhysicalDeviceExternalSemaphoreInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    info.pNext = &st;
    info.handleType = type;
    VkExternalSemaphoreProperties props{};
    props.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(pd, &info, &props);
    std::string s;
    s += (props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) ? "export " : "";
    s += (props.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) ? "import " : "";
    return s.empty() ? "none" : s;
}

std::string buffer_features(VkPhysicalDevice pd, VkExternalMemoryHandleTypeFlagBits type) {
    VkPhysicalDeviceExternalBufferInfo info{};
    info.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.handleType = type;
    VkExternalBufferProperties props{};
    props.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES;
    vkGetPhysicalDeviceExternalBufferProperties(pd, &info, &props);
    const auto f = props.externalMemoryProperties.externalMemoryFeatures;
    std::string s;
    s += (f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) ? "dedicated-only " : "";
    s += (f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) ? "export " : "";
    s += (f & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) ? "import " : "";
    return s.empty() ? "none" : s;
}

std::string memory_flags(VkMemoryPropertyFlags f) {
    std::string s;
    s += (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "device-local " : "";
    s += (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "host-visible " : "";
    s += (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "coherent " : "";
    s += (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "cached " : "";
    return s.empty() ? "none" : s;
}

uint32_t queue_family(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, nullptr);
    std::vector<VkQueueFamilyProperties> q(n);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, q.data());
    // The compute family, as the backend takes it, so a copy here queues where the backend's copies would.
    for (uint32_t i = 0; i < n; ++i)
        if ((q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) return i;
    for (uint32_t i = 0; i < n; ++i)
        if (q[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
    throw std::runtime_error("no compute queue");
}

// A logical device with a timeline, the external semaphore and host-memory extensions where offered, and one compute queue.
struct Device {
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    bool host_import = false, sync_fd = false, dma_buf = false, uncached = false, calibrated = false;
    VkDeviceSize host_alignment = 0;
    VkPhysicalDeviceMemoryProperties mem{};
#define DECLARE(name) PFN_##name name = nullptr;
    DEVICE_FNS(DECLARE)
#undef DECLARE
    PFN_vkGetMemoryHostPointerPropertiesEXT vkGetMemoryHostPointerPropertiesEXT = nullptr;
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR = nullptr;
    PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR = nullptr;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR vkGetMemoryFdPropertiesKHR = nullptr;
    PFN_vkGetCalibratedTimestampsEXT vkGetCalibratedTimestampsEXT = nullptr;
};

// `memory_model` enables the Vulkan memory model, which the exchange's flag wait needs and the other modes do not ask of a device.
Device open_device(VkPhysicalDevice pd, bool memory_model = false) {
    Device d;
    d.pd = pd;
    const auto exts = extensions(pd);
    std::vector<const char*> enable;
    if (has(exts, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) {
        enable.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
        d.host_import = true;
    }
#if !defined(_WIN32)
    if (has(exts, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) &&
        semaphore_features(pd, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, false) == "export import ") {
        enable.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
        d.sync_fd = true;
    }
    if (has(exts, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) && has(exts, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) &&
        buffer_features(pd, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT) == "export import ") {
        enable.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        enable.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
        d.dma_buf = true;
    }
#endif
    // Uncached device memory for the exchange's inboxes, and timestamps the host can place on its own clock for the members' arrival.
    if (has(exts, "VK_AMD_device_coherent_memory")) {
        enable.push_back("VK_AMD_device_coherent_memory");
        d.uncached = true;
    }
    if (has(exts, VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME)) {
        enable.push_back(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
        d.calibrated = true;
    }
    if (d.host_import) {
        VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{};
        hp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT;
        VkPhysicalDeviceProperties2 p{};
        p.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        p.pNext = &hp;
        vkGetPhysicalDeviceProperties2(pd, &p);
        d.host_alignment = hp.minImportedHostPointerAlignment;
    }
    vkGetPhysicalDeviceMemoryProperties(pd, &d.mem);
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qi{};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = queue_family(pd);
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.timelineSemaphore = VK_TRUE;
    f12.vulkanMemoryModel = memory_model ? VK_TRUE : VK_FALSE;
    f12.vulkanMemoryModelDeviceScope = memory_model ? VK_TRUE : VK_FALSE;
    VkPhysicalDeviceCoherentMemoryFeaturesAMD coherent{};
    coherent.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;
    coherent.deviceCoherentMemory = VK_TRUE;
    if (d.uncached) f12.pNext = &coherent;
    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &f12;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    ci.enabledExtensionCount = (uint32_t)enable.size();
    ci.ppEnabledExtensionNames = enable.data();
    check(vkCreateDevice(pd, &ci, nullptr, &d.dev), "vkCreateDevice");
#define LOAD(name) d.name = (PFN_##name)vkGetDeviceProcAddr(d.dev, #name);
    DEVICE_FNS(LOAD)
    LOAD(vkGetMemoryHostPointerPropertiesEXT)
    LOAD(vkGetSemaphoreFdKHR)
    LOAD(vkImportSemaphoreFdKHR)
    LOAD(vkGetMemoryFdKHR)
    LOAD(vkGetMemoryFdPropertiesKHR)
    LOAD(vkGetCalibratedTimestampsEXT)
#undef LOAD
    d.vkGetDeviceQueue(d.dev, qi.queueFamilyIndex, 0, &d.queue);
    VkCommandPoolCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = qi.queueFamilyIndex;
    check(d.vkCreateCommandPool(d.dev, &pi, nullptr, &d.pool), "vkCreateCommandPool");
    return d;
}

uint32_t memory_type(const Device& d, uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags prefer = 0) {
    int found = -1;
    for (uint32_t i = 0; i < d.mem.memoryTypeCount; ++i) {
        const auto f = d.mem.memoryTypes[i].propertyFlags;
        if (!(bits & (1u << i)) || (f & want) != want) continue;
        if ((f & prefer) == prefer) return i;
        if (found < 0) found = (int)i;
    }
    if (found < 0) throw std::runtime_error("no memory type with the wanted properties");
    return (uint32_t)found;
}

struct Buffer {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
};

Buffer make_buffer(Device& d, VkDeviceSize bytes, VkMemoryPropertyFlags want, VkMemoryPropertyFlags prefer = 0) {
    Buffer b;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check(d.vkCreateBuffer(d.dev, &bi, nullptr, &b.buf), "vkCreateBuffer");
    VkMemoryRequirements req;
    d.vkGetBufferMemoryRequirements(d.dev, b.buf, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(d, req.memoryTypeBits, want, prefer);
    check(d.vkAllocateMemory(d.dev, &ai, nullptr, &b.mem), "vkAllocateMemory");
    check(d.vkBindBufferMemory(d.dev, b.buf, b.mem, 0), "vkBindBufferMemory");
    if (want & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) check(d.vkMapMemory(d.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
    return b;
}

// A buffer over host memory the caller allocated, imported rather than copied, so two devices can address the same bytes.
// `prefer` picks among the memory types the import allows, such as the device-uncached one the exchange's inboxes take.
Buffer import_host(Device& d, void* host, VkDeviceSize bytes, VkMemoryPropertyFlags prefer = 0) {
    Buffer b;
    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.pNext = &ext;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    check(d.vkCreateBuffer(d.dev, &bi, nullptr, &b.buf), "vkCreateBuffer (imported)");
    VkMemoryHostPointerPropertiesEXT hp{};
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    check(d.vkGetMemoryHostPointerPropertiesEXT(d.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hp), "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryRequirements req;
    d.vkGetBufferMemoryRequirements(d.dev, b.buf, &req);
    VkImportMemoryHostPointerInfoEXT imp{};
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = host;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = bytes;
    ai.memoryTypeIndex = memory_type(d, hp.memoryTypeBits & req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, prefer);
    if (prefer) std::printf("  host import: memory type %u (flags 0x%x)\n", ai.memoryTypeIndex, d.mem.memoryTypes[ai.memoryTypeIndex].propertyFlags);
    check(d.vkAllocateMemory(d.dev, &ai, nullptr, &b.mem), "vkAllocateMemory (host pointer import)");
    check(d.vkBindBufferMemory(d.dev, b.buf, b.mem, 0), "vkBindBufferMemory (imported)");
    b.map = host;
    return b;
}

#if !defined(_WIN32)
VkBuffer external_buffer(Device& d, VkDeviceSize bytes, VkExternalMemoryHandleTypeFlagBits type) {
    VkExternalMemoryBufferCreateInfo ext{};
    ext.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    ext.handleTypes = type;
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.pNext = &ext;
    bi.size = bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    VkBuffer buf;
    check(d.vkCreateBuffer(d.dev, &bi, nullptr, &buf), "vkCreateBuffer (external)");
    return buf;
}

// Device-local memory on `d` exported as a dma-buf, and the file descriptor another device imports it from.
Buffer export_dma_buf(Device& d, VkDeviceSize bytes, int& fd) {
    Buffer b;
    b.buf = external_buffer(d, bytes, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    VkMemoryRequirements req;
    d.vkGetBufferMemoryRequirements(d.dev, b.buf, &req);
    VkExportMemoryAllocateInfo ex{};
    ex.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &ex;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(d.vkAllocateMemory(d.dev, &ai, nullptr, &b.mem), "vkAllocateMemory (dma-buf export)");
    check(d.vkBindBufferMemory(d.dev, b.buf, b.mem, 0), "vkBindBufferMemory (dma-buf export)");
    VkMemoryGetFdInfoKHR gi{};
    gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gi.memory = b.mem;
    gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    check(d.vkGetMemoryFdKHR(d.dev, &gi, &fd), "vkGetMemoryFdKHR");
    return b;
}

// The memory another device exported, addressed from `d`; the descriptor passes to the driver.
Buffer import_dma_buf(Device& d, int fd, VkDeviceSize bytes, VkMemoryPropertyFlags prefer = 0) {
    Buffer b;
    b.buf = external_buffer(d, bytes, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    VkMemoryFdPropertiesKHR fp{};
    fp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
    check(d.vkGetMemoryFdPropertiesKHR(d.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp), "vkGetMemoryFdPropertiesKHR");
    VkMemoryRequirements req;
    d.vkGetBufferMemoryRequirements(d.dev, b.buf, &req);
    VkImportMemoryFdInfoKHR imp{};
    imp.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    imp.fd = fd;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &imp;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(d, fp.memoryTypeBits & req.memoryTypeBits, 0, prefer);
    std::printf("  dma-buf import on B: memory type %u (%s)\n", ai.memoryTypeIndex, memory_flags(d.mem.memoryTypes[ai.memoryTypeIndex].propertyFlags).c_str());
    check(d.vkAllocateMemory(d.dev, &ai, nullptr, &b.mem), "vkAllocateMemory (dma-buf import)");
    check(d.vkBindBufferMemory(d.dev, b.buf, b.mem, 0), "vkBindBufferMemory (dma-buf import)");
    return b;
}
#endif

VkSemaphore make_semaphore(Device& d, bool timeline, bool exportable_sync_fd = false) {
    VkSemaphoreTypeCreateInfo st{};
    st.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    st.semaphoreType = timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    VkExportSemaphoreCreateInfo ex{};
    ex.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    ex.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    if (exportable_sync_fd) st.pNext = &ex;
    VkSemaphoreCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    ci.pNext = &st;
    VkSemaphore s;
    check(d.vkCreateSemaphore(d.dev, &ci, nullptr, &s), "vkCreateSemaphore");
    return s;
}

// A command buffer holding one copy, recorded once and submitted every iteration.
// A copy into host memory ends with a barrier to the host, so its bytes are available to whoever reads that memory next.
VkCommandBuffer record_copy(Device& d, VkBuffer src, VkBuffer dst, VkDeviceSize bytes, bool to_host) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = d.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    check(d.vkAllocateCommandBuffers(d.dev, &ai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    check(d.vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    VkBufferCopy region{0, 0, bytes};
    d.vkCmdCopyBuffer(cb, src, dst, 1, &region);
    if (to_host) {
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
        d.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    check(d.vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    return cb;
}

// A command buffer holding `n` dependent copies of the same bytes, so a copy's cost inside a submission separates from the submission's own.
VkCommandBuffer record_copies(Device& d, VkBuffer src, VkBuffer dst, VkDeviceSize bytes, int n) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = d.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    check(d.vkAllocateCommandBuffers(d.dev, &ai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    check(d.vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    VkBufferCopy region{0, 0, bytes};
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    for (int i = 0; i < n; ++i) {
        d.vkCmdCopyBuffer(cb, src, dst, 1, &region);
        d.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    check(d.vkEndCommandBuffer(cb), "vkEndCommandBuffer");
    return cb;
}

// One submission of `cb`, waiting on `wait` (a timeline at `wait_value`, or a binary semaphore when `wait_value` is 0) and signalling `timeline` to `value` and optionally a binary semaphore.
void submit(Device& d, VkCommandBuffer cb, VkSemaphore timeline, uint64_t value, VkSemaphore wait = VK_NULL_HANDLE, uint64_t wait_value = 0, VkSemaphore binary = VK_NULL_HANDLE) {
    VkSemaphore signals[2] = {timeline, binary};
    uint64_t signal_values[2] = {value, 0};
    const uint64_t wait_values[1] = {wait_value};
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkTimelineSemaphoreSubmitInfo ti{};
    ti.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    ti.signalSemaphoreValueCount = binary ? 2 : 1;
    ti.pSignalSemaphoreValues = signal_values;
    ti.waitSemaphoreValueCount = wait ? 1 : 0;
    ti.pWaitSemaphoreValues = wait_values;
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &ti;
    si.commandBufferCount = cb ? 1 : 0;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = binary ? 2 : 1;
    si.pSignalSemaphores = signals;
    si.waitSemaphoreCount = wait ? 1 : 0;
    si.pWaitSemaphores = &wait;
    si.pWaitDstStageMask = &stage;
    check(d.vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
}

void wait_value(Device& d, VkSemaphore timeline, uint64_t value) {
    VkSemaphoreWaitInfo wi{};
    wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline;
    wi.pValues = &value;
    check(d.vkWaitSemaphores(d.dev, &wi, UINT64_MAX), "vkWaitSemaphores");
}

void* host_alloc(size_t bytes, size_t alignment) {
#if defined(_WIN32)
    return _aligned_malloc(bytes, alignment);
#else
    void* p = nullptr;
    return posix_memalign(&p, alignment, bytes) == 0 ? p : nullptr;
#endif
}

void host_free(void* p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

int probe() {
    VkInstance inst = make_instance();
    const auto pds = physical_devices(inst);
    std::printf("%zu devices\n", pds.size());
    for (size_t i = 0; i < pds.size(); ++i) {
        const auto exts = extensions(pds[i]);
        const auto id = identity(pds[i], exts);
        std::printf("\ndevice %zu: %s | %s | pci %s\n", i, id.name.c_str(), id.driver.c_str(), id.pci.empty() ? "?" : id.pci.c_str());
        std::printf("  device uuid %s\n  driver uuid %s\n", id.device_uuid.c_str(), id.driver_uuid.c_str());
        for (const char* e : {VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
                              VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME,
                              "VK_KHR_external_semaphore_win32", "VK_KHR_external_memory_win32"})
            std::printf("  %-36s %s\n", e, has(exts, e) ? "yes" : "no");
        for (bool timeline : {false, true}) {
            std::printf("  %s semaphore: opaque-fd %s| sync-fd %s| opaque-win32 %s\n", timeline ? "timeline" : "binary  ",
                        semaphore_features(pds[i], VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, timeline).c_str(),
                        semaphore_features(pds[i], VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, timeline).c_str(),
                        semaphore_features(pds[i], VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT, timeline).c_str());
        }
        std::printf("  buffer: host-allocation %s| opaque-fd %s| dma-buf %s\n",
                    buffer_features(pds[i], VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT).c_str(),
                    buffer_features(pds[i], VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT).c_str(),
                    buffer_features(pds[i], VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT).c_str());
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
        budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
        VkPhysicalDeviceMemoryProperties2 mp{};
        mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
        const bool has_budget = has(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        if (has_budget) mp.pNext = &budget;
        vkGetPhysicalDeviceMemoryProperties2(pds[i], &mp);
        for (uint32_t h = 0; h < mp.memoryProperties.memoryHeapCount; ++h)
            std::printf("  heap %u: %.2f GiB%s, budget %.2f GiB, used %.2f GiB\n", h, mp.memoryProperties.memoryHeaps[h].size / 1073741824.0,
                        (mp.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " device-local" : "",
                        has_budget ? budget.heapBudget[h] / 1073741824.0 : 0.0, has_budget ? budget.heapUsage[h] / 1073741824.0 : 0.0);
        Device d = open_device(pds[i]);
        if (d.host_import) {
            const size_t bytes = std::max<size_t>((size_t)d.host_alignment, 1 << 20);
            void* host = host_alloc(bytes, (size_t)d.host_alignment);
            VkMemoryHostPointerPropertiesEXT hp{};
            hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
            check(d.vkGetMemoryHostPointerPropertiesEXT(d.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, host, &hp), "vkGetMemoryHostPointerPropertiesEXT");
            std::printf("  host-pointer import: alignment %llu, memory types", (unsigned long long)d.host_alignment);
            for (uint32_t t = 0; t < d.mem.memoryTypeCount; ++t)
                if (hp.memoryTypeBits & (1u << t)) std::printf(" [%u: heap %u, %s]", t, d.mem.memoryTypes[t].heapIndex, memory_flags(d.mem.memoryTypes[t].propertyFlags).c_str());
            Buffer b = import_host(d, host, bytes);
            std::printf("\n  host-pointer import of %zu bytes into a buffer: ok\n", bytes);
            d.vkDestroyBuffer(d.dev, b.buf, nullptr);
            d.vkFreeMemory(d.dev, b.mem, nullptr);
            host_free(host);
        }
        d.vkDestroyCommandPool(d.dev, d.pool, nullptr);
        d.vkDestroyDevice(d.dev, nullptr);
    }
    uint32_t ng = 0;
    check(vkEnumeratePhysicalDeviceGroups(inst, &ng, nullptr), "vkEnumeratePhysicalDeviceGroups");
    std::vector<VkPhysicalDeviceGroupProperties> groups(ng);
    for (auto& g : groups) g.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
    check(vkEnumeratePhysicalDeviceGroups(inst, &ng, groups.data()), "vkEnumeratePhysicalDeviceGroups");
    size_t widest = 0;
    for (auto& g : groups) widest = std::max<size_t>(widest, g.physicalDeviceCount);
    std::printf("\n%u device groups, the widest of %zu device%s\n", ng, widest, widest == 1 ? " (no peer memory between devices)" : "s");
    for (uint32_t gi = 0; gi < ng; ++gi) {
        if (groups[gi].physicalDeviceCount < 2) continue;
        std::printf("group %u: %u devices, subset allocation %s\n", gi, groups[gi].physicalDeviceCount, groups[gi].subsetAllocation ? "yes" : "no");
        Device d = open_device(groups[gi].physicalDevices[0]);
        for (uint32_t h = 0; h < d.mem.memoryHeapCount; ++h) {
            VkPeerMemoryFeatureFlags f = 0;
            d.vkGetDeviceGroupPeerMemoryFeatures(d.dev, h, 0, 1, &f);
            std::printf("  heap %u, device 0 reading device 1: flags 0x%x\n", h, f);
        }
    }
    if (pds.size() > 1) {
        const auto a = identity(pds[0], extensions(pds[0])), b = identity(pds[1], extensions(pds[1]));
        std::printf("devices 0 and 1: driver uuid %s, device uuid %s\n", a.driver_uuid == b.driver_uuid ? "same" : "different", a.device_uuid == b.device_uuid ? "same" : "different");
    }
    vkDestroyInstance(inst, nullptr);
    return 0;
}

struct Stats {
    double min, median, p90;
};

Stats stats(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return {v.front(), v[v.size() / 2], v[v.size() * 9 / 10]};
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int time_handoff(int ia, int ib, int iters) {
    VkInstance inst = make_instance();
    const auto pds = physical_devices(inst);
    if (ia < 0 || ib < 0 || ia >= (int)pds.size() || ib >= (int)pds.size() || ia == ib) throw std::runtime_error("need two distinct device indices");
    std::printf("A: device %d %s (pci %s)\nB: device %d %s (pci %s)\n", ia, identity(pds[ia], extensions(pds[ia])).name.c_str(), identity(pds[ia], extensions(pds[ia])).pci.c_str(),
                ib, identity(pds[ib], extensions(pds[ib])).name.c_str(), identity(pds[ib], extensions(pds[ib])).pci.c_str());
    Device A = open_device(pds[ia]), B = open_device(pds[ib]);
    const bool imported = A.host_import && B.host_import;
    const bool sync_fd = imported && A.sync_fd && B.sync_fd;
    const bool dma_buf = A.dma_buf && B.dma_buf;
    std::printf("host-pointer import: %s; sync-fd semaphores: %s; dma-buf: %s\n", imported ? "yes" : "no", sync_fd ? "yes" : "no", dma_buf ? "yes" : "no");
    VkSemaphore tla = make_semaphore(A, true), tlb = make_semaphore(B, true), relay = make_semaphore(B, true);
#if !defined(_WIN32)
    VkSemaphore bina = sync_fd ? make_semaphore(A, false, true) : VK_NULL_HANDLE;
    VkSemaphore binb = sync_fd ? make_semaphore(B, false) : VK_NULL_HANDLE;
#endif
    uint64_t va = 0, vb = 0, vr = 0;
    const size_t align = (size_t)std::max<VkDeviceSize>({A.host_alignment, B.host_alignment, 4096});

    // A submission that only signals, waited on by the host: the floor under every way below.
    {
        std::vector<double> t;
        for (int i = 0; i < iters; ++i) {
            const double t0 = now_us();
            submit(A, VK_NULL_HANDLE, tla, ++va);
            wait_value(A, tla, va);
            t.push_back(now_us() - t0);
        }
        const auto s = stats(t);
        std::printf("\nempty submission and host wait on A: min %.1f us, median %.1f, p90 %.1f\n", s.min, s.median, s.p90);
    }

    // Bytes a handoff carries: a decode token of a 5120-wide model, eight of them, a 64-row chunk and a 512-row chunk.
    const size_t sizes[] = {20480, 163840, 1310720, 10485760};
    std::printf("\n%-10s %-12s %10s %10s %10s %10s\n", "bytes", "way", "min us", "median us", "p90 us", "GB/s");
    for (size_t bytes : sizes) {
        const size_t padded = (bytes + align - 1) / align * align;
        Buffer src = make_buffer(A, bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer dst = make_buffer(B, bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Buffer stage_a = make_buffer(A, bytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        Buffer stage_b = make_buffer(B, bytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        void* host = imported ? host_alloc(padded, align) : nullptr;
        Buffer shared_a, shared_b;
        if (imported) {
            shared_a = import_host(A, host, padded);
            shared_b = import_host(B, host, padded);
        }
        VkCommandBuffer a_out = record_copy(A, src.buf, stage_a.buf, bytes, true);
        VkCommandBuffer b_in = record_copy(B, stage_b.buf, dst.buf, bytes, false);
        VkCommandBuffer a_shared = imported ? record_copy(A, src.buf, shared_a.buf, bytes, true) : VK_NULL_HANDLE;
        VkCommandBuffer b_shared = imported ? record_copy(B, shared_b.buf, dst.buf, bytes, false) : VK_NULL_HANDLE;
        VkCommandBuffer b_check = record_copy(B, dst.buf, stage_b.buf, bytes, true);
        VkCommandBuffer a_fill = record_copy(A, stage_a.buf, src.buf, bytes, false);

        // Each way is checked once for delivering a fresh pattern before it is timed.
        uint32_t pattern = 0x9e3779b9u * (uint32_t)bytes;
        auto seed_source = [&]() {
            ++pattern;
            auto* w = (uint32_t*)stage_a.map;
            for (size_t k = 0; k < bytes / 4; ++k) w[k] = pattern + (uint32_t)k;
            submit(A, a_fill, tla, ++va);
            wait_value(A, tla, va);
            // Every host-side copy of the bytes is cleared, so only a way that moves them from A's memory can pass.
            std::memset(stage_a.map, 0, bytes);
            std::memset(stage_b.map, 0, bytes);
            if (host) std::memset(host, 0, padded);
        };
        auto delivered = [&]() {
            submit(B, b_check, tlb, ++vb);
            wait_value(B, tlb, vb);
            const auto* w = (const uint32_t*)stage_b.map;
            for (size_t k = 0; k < bytes / 4; ++k)
                if (w[k] != pattern + (uint32_t)k) return false;
            return true;
        };

        auto run_prepared = [&](const char* name, auto&& prepare, auto&& once) {
            seed_source();
            prepare();
            once();
            if (!delivered()) {
                std::printf("%-10zu %-12s delivered wrong bytes\n", bytes, name);
                return;
            }
            std::vector<double> t;
            for (int i = 0; i < iters; ++i) {
                const double t0 = now_us();
                once();
                t.push_back(now_us() - t0);
            }
            const auto s = stats(t);
            std::printf("%-10zu %-12s %10.1f %10.1f %10.1f %10.2f\n", bytes, name, s.min, s.median, s.p90, bytes / (s.median * 1e3));
        };
        auto run = [&](const char* name, auto&& once) { run_prepared(name, []() {}, once); };

        // The single copies the ways are built from, each with its own host wait, to split a way's time into its parts.
        auto single = [&](const char* name, Device& d, VkCommandBuffer cb, VkSemaphore tl, uint64_t& v) {
            std::vector<double> t;
            for (int i = 0; i < iters; ++i) {
                const double t0 = now_us();
                submit(d, cb, tl, ++v);
                wait_value(d, tl, v);
                t.push_back(now_us() - t0);
            }
            const auto s = stats(t);
            std::printf("%-10zu %-12s %10.1f %10.1f %10.1f %10.2f\n", bytes, name, s.min, s.median, s.p90, bytes / (s.median * 1e3));
        };
        {
            Buffer local = make_buffer(A, bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (bytes == sizes[0]) {
                VkCommandBuffer empty = record_copies(A, src.buf, local.buf, bytes, 0);
                single("A empty cb", A, empty, tla, va);
                // Eight empty command buffers submitted back to back and one wait: eight times one if the cost is the device's, about one if it is latency.
                std::vector<VkCommandBuffer> empties;
                for (int k = 0; k < 8; ++k) empties.push_back(record_copies(A, src.buf, local.buf, bytes, 0));
                std::vector<double> t;
                for (int i = 0; i < iters; ++i) {
                    const double t0 = now_us();
                    for (int k = 0; k < 8; ++k) submit(A, empties[k], tla, ++va);
                    wait_value(A, tla, va);
                    t.push_back(now_us() - t0);
                }
                const auto s = stats(t);
                std::printf("%-10zu %-12s %10.1f %10.1f %10.1f\n", bytes, "A 8 empty", s.min, s.median, s.p90);
            }
            single("A local", A, record_copy(A, src.buf, local.buf, bytes, false), tla, va);
            single("A local x8", A, record_copies(A, src.buf, local.buf, bytes, 8), tla, va);
            single("A host x8", A, record_copies(A, src.buf, stage_a.buf, bytes, 8), tla, va);
            single("A to host", A, a_out, tla, va);
            single("B from host", B, b_in, tlb, vb);
            A.vkDeviceWaitIdle(A.dev);
            A.vkDestroyBuffer(A.dev, local.buf, nullptr);
            A.vkFreeMemory(A.dev, local.mem, nullptr);
        }

        run("read-write", [&]() {
            submit(A, a_out, tla, ++va);
            wait_value(A, tla, va);
            std::memcpy(stage_b.map, stage_a.map, bytes);
            submit(B, b_in, tlb, ++vb);
            wait_value(B, tlb, vb);
        });
        if (imported) {
            run("imported", [&]() {
                submit(A, a_shared, tla, ++va);
                wait_value(A, tla, va);
                submit(B, b_shared, tlb, ++vb);
                wait_value(B, tlb, vb);
            });
            run("relay", [&]() {
                // B's copy is queued before A's starts, behind a timeline only the host advances.
                submit(B, b_shared, tlb, ++vb, relay, ++vr);
                submit(A, a_shared, tla, ++va);
                wait_value(A, tla, va);
                VkSemaphoreSignalInfo si{};
                si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
                si.semaphore = relay;
                si.value = vr;
                check(B.vkSignalSemaphore(B.dev, &si), "vkSignalSemaphore");
                wait_value(B, tlb, vb);
            });
        }
        Buffer exp_a, imp_b;
#if !defined(_WIN32)
        if (sync_fd) {
            run("sync-fd", [&]() {
                submit(A, a_shared, tla, ++va, VK_NULL_HANDLE, 0, bina);
                VkSemaphoreGetFdInfoKHR gi{};
                gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
                gi.semaphore = bina;
                gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                int fd = -1;
                check(A.vkGetSemaphoreFdKHR(A.dev, &gi, &fd), "vkGetSemaphoreFdKHR");
                VkImportSemaphoreFdInfoKHR ii{};
                ii.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                ii.semaphore = binb;
                ii.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                ii.fd = fd;
                check(B.vkImportSemaphoreFdKHR(B.dev, &ii), "vkImportSemaphoreFdKHR");
                submit(B, b_shared, tlb, ++vb, binb, 0);
                wait_value(B, tlb, vb);
            });
        }
        if (dma_buf) {
            int fd = -1;
            exp_a = export_dma_buf(A, bytes, fd);
            imp_b = import_dma_buf(B, fd, bytes);
            VkCommandBuffer a_exp = record_copy(A, src.buf, exp_a.buf, bytes, true);
            VkCommandBuffer b_imp = record_copy(B, imp_b.buf, dst.buf, bytes, false);
            run("dma-buf", [&]() {
                submit(A, a_exp, tla, ++va);
                wait_value(A, tla, va);
                submit(B, b_imp, tlb, ++vb);
                wait_value(B, tlb, vb);
            });
            // B reading what A left in its own memory: the handoff when A's output buffer is itself the exported one.
            run_prepared("peer-read", [&]() {
                submit(A, a_exp, tla, ++va);
                wait_value(A, tla, va);
            }, [&]() {
                submit(B, b_imp, tlb, ++vb);
                wait_value(B, tlb, vb);
            });
            if (sync_fd) {
                run("dma+sync-fd", [&]() {
                    submit(A, a_exp, tla, ++va, VK_NULL_HANDLE, 0, bina);
                    VkSemaphoreGetFdInfoKHR gi{};
                    gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
                    gi.semaphore = bina;
                    gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                    int sfd = -1;
                    check(A.vkGetSemaphoreFdKHR(A.dev, &gi, &sfd), "vkGetSemaphoreFdKHR");
                    VkImportSemaphoreFdInfoKHR ii{};
                    ii.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                    ii.semaphore = binb;
                    ii.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                    ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                    ii.fd = sfd;
                    check(B.vkImportSemaphoreFdKHR(B.dev, &ii), "vkImportSemaphoreFdKHR");
                    submit(B, b_imp, tlb, ++vb, binb, 0);
                    wait_value(B, tlb, vb);
                });
            }
        }
#endif
        A.vkDeviceWaitIdle(A.dev);
        B.vkDeviceWaitIdle(B.dev);
        for (auto* p : {&imp_b}) {
            if (p->buf) B.vkDestroyBuffer(B.dev, p->buf, nullptr);
            if (p->mem) B.vkFreeMemory(B.dev, p->mem, nullptr);
        }
        for (auto* p : {&exp_a}) {
            if (p->buf) A.vkDestroyBuffer(A.dev, p->buf, nullptr);
            if (p->mem) A.vkFreeMemory(A.dev, p->mem, nullptr);
        }
        for (auto* p : {&src, &stage_a, &shared_a}) {
            if (p->buf) A.vkDestroyBuffer(A.dev, p->buf, nullptr);
            if (p->mem) A.vkFreeMemory(A.dev, p->mem, nullptr);
        }
        for (auto* p : {&dst, &stage_b, &shared_b}) {
            if (p->buf) B.vkDestroyBuffer(B.dev, p->buf, nullptr);
            if (p->mem) B.vkFreeMemory(B.dev, p->mem, nullptr);
        }
        host_free(host);
    }
    return 0;
}

#if !defined(_WIN32)
// Bytes bounced between two devices `hops` times, each hop reading the other device's memory through dma-buf and writing its own for the next hop, as a tensor group's sums would chain.
// With the host waiting on every hop, and with the whole chain queued ahead through sync files so no hop waits for the host: the difference is what a device-side wait saves per sum.
int pingpong(int ia, int ib, int hops) {
    VkInstance inst = make_instance();
    const auto pds = physical_devices(inst);
    if (ia < 0 || ib < 0 || ia >= (int)pds.size() || ib >= (int)pds.size() || ia == ib) throw std::runtime_error("need two distinct device indices");
    Device dev[2] = {open_device(pds[ia]), open_device(pds[ib])};
    if (!dev[0].dma_buf || !dev[1].dma_buf || !dev[0].sync_fd || !dev[1].sync_fd) throw std::runtime_error("needs dma-buf memory and sync-file semaphores on both devices");
    std::printf("devices %d (pci %s) and %d (pci %s), %d hops\n", ia, identity(pds[ia], extensions(pds[ia])).pci.c_str(), ib, identity(pds[ib], extensions(pds[ib])).pci.c_str(), hops);
    for (size_t bytes : {(size_t)20480, (size_t)163840, (size_t)1310720}) {
        Buffer local[2], exported[2], peer[2];
        for (int d = 0; d < 2; ++d) {
            local[d] = make_buffer(dev[d], bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            int fd = -1;
            exported[d] = export_dma_buf(dev[d], bytes, fd);
            peer[1 - d] = import_dma_buf(dev[1 - d], fd, bytes);
        }
        // A hop on device d: the peer's bytes into local memory, then into its own exported memory for the next hop.
        auto record_hop = [&](int d) {
            VkCommandBufferAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ai.commandPool = dev[d].pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb;
            check(dev[d].vkAllocateCommandBuffers(dev[d].dev, &ai, &cb), "vkAllocateCommandBuffers");
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            check(dev[d].vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
            VkBufferCopy region{0, 0, bytes};
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
            dev[d].vkCmdCopyBuffer(cb, peer[d].buf, local[d].buf, 1, &region);
            dev[d].vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            dev[d].vkCmdCopyBuffer(cb, local[d].buf, exported[d].buf, 1, &region);
            dev[d].vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            check(dev[d].vkEndCommandBuffer(cb), "vkEndCommandBuffer");
            return cb;
        };
        std::vector<VkCommandBuffer> cbs(hops);
        std::vector<VkSemaphore> sig(hops), wait(hops);
        for (int i = 0; i < hops; ++i) {
            cbs[i] = record_hop(i % 2);
            sig[i] = make_semaphore(dev[i % 2], false, true);
            wait[i] = make_semaphore(dev[1 - i % 2], false);
        }
        VkSemaphore tl[2] = {make_semaphore(dev[0], true), make_semaphore(dev[1], true)};
        uint64_t v[2] = {0, 0};
        // Every hop on one device, back to back with no waits: what a submission costs the device itself.
        auto one_device = [&]() {
            const double t0 = now_us();
            for (int i = 0; i < hops; i += 2) submit(dev[0], cbs[i], tl[0], ++v[0]);
            wait_value(dev[0], tl[0], v[0]);
            return (now_us() - t0) / (hops / 2);
        };
        auto host_relay = [&]() {
            const double t0 = now_us();
            for (int i = 0; i < hops; ++i) {
                const int d = i % 2;
                submit(dev[d], cbs[i], tl[d], ++v[d]);
                wait_value(dev[d], tl[d], v[d]);
            }
            return (now_us() - t0) / hops;
        };
        auto chained = [&]() {
            const double t0 = now_us();
            for (int i = 0; i < hops; ++i) {
                const int d = i % 2;
                submit(dev[d], cbs[i], tl[d], ++v[d], i ? wait[i - 1] : VK_NULL_HANDLE, 0, sig[i]);
                VkSemaphoreGetFdInfoKHR gi{};
                gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
                gi.semaphore = sig[i];
                gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                int fd = -1;
                check(dev[d].vkGetSemaphoreFdKHR(dev[d].dev, &gi, &fd), "vkGetSemaphoreFdKHR");
                VkImportSemaphoreFdInfoKHR ii{};
                ii.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                ii.semaphore = wait[i];
                ii.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                ii.fd = fd;
                check(dev[1 - d].vkImportSemaphoreFdKHR(dev[1 - d].dev, &ii), "vkImportSemaphoreFdKHR");
            }
            const int last = (hops - 1) % 2;
            wait_value(dev[last], tl[last], v[last]);
            // The last hop's semaphore was imported but never waited on; a wait-only submission consumes it before the next round reuses it.
            submit(dev[1 - last], VK_NULL_HANDLE, tl[1 - last], ++v[1 - last], wait[hops - 1], 0);
            wait_value(dev[1 - last], tl[1 - last], v[1 - last]);
            return (now_us() - t0) / hops;
        };
        std::vector<double> a, b, c;
        for (int r = 0; r < 5; ++r) {
            a.push_back(one_device());
            b.push_back(host_relay());
            c.push_back(chained());
        }
        std::printf("%8zu bytes: us a hop, median of 5 chains: one device back to back %.1f, host waits every hop %.1f, sync files queued ahead %.1f\n", bytes,
                    stats(a).median, stats(b).median, stats(c).median);
        for (int d = 0; d < 2; ++d) dev[d].vkDeviceWaitIdle(dev[d].dev);
    }
    return 0;
}
#endif

#if !defined(_WIN32)
// The exchange's kernels (tools/shaders), compiled by CMake: plain for the sync-file exchange, and with the Vulkan memory model at device and at queue-family scope for the flag wait.
const uint32_t kSend[] = {
#include "exchange_send.inc"
};
const uint32_t kSum[] = {
#include "exchange_sum.inc"
};
const uint32_t kSendDev[] = {
#include "exchange_send_dev.inc"
};
const uint32_t kFlagDev[] = {
#include "exchange_flag_dev.inc"
};
const uint32_t kSpinDev[] = {
#include "exchange_spin_dev.inc"
};
const uint32_t kSumDev[] = {
#include "exchange_sum_dev.inc"
};
const uint32_t kSendQf[] = {
#include "exchange_send_qf.inc"
};
const uint32_t kFlagQf[] = {
#include "exchange_flag_qf.inc"
};
const uint32_t kSpinQf[] = {
#include "exchange_spin_qf.inc"
};
const uint32_t kSumQf[] = {
#include "exchange_sum_qf.inc"
};

// The exchange's pipelines on one device, over one layout: the target inbox, the member's own inbox, the result counters and the sums.
struct ExchangePipes {
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline send = VK_NULL_HANDLE, sum = VK_NULL_HANDLE;
    VkPipeline mm_send[2] = {}, mm_flag[2] = {}, mm_spin[2] = {}, mm_sum[2] = {};   // [0] device scope, [1] queue-family scope
};

ExchangePipes exchange_pipes(Device& d) {
    ExchangePipes p;
    VkDescriptorSetLayoutBinding b[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
        b[i].binding = i;
        b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1;
        b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dci.bindingCount = 4;
    dci.pBindings = b;
    check(d.vkCreateDescriptorSetLayout(d.dev, &dci, nullptr, &p.set_layout), "vkCreateDescriptorSetLayout");
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &p.set_layout;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges = &pr;
    check(d.vkCreatePipelineLayout(d.dev, &lci, nullptr, &p.layout), "vkCreatePipelineLayout");
    auto pipe = [&](const uint32_t* code, size_t bytes) {
        VkShaderModuleCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sci.codeSize = bytes;
        sci.pCode = code;
        VkShaderModule m;
        check(d.vkCreateShaderModule(d.dev, &sci, nullptr, &m), "vkCreateShaderModule");
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = m;
        ci.stage.pName = "main";
        ci.layout = p.layout;
        VkPipeline pl;
        check(d.vkCreateComputePipelines(d.dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pl), "vkCreateComputePipelines");
        return pl;
    };
    p.send = pipe(kSend, sizeof kSend);
    p.sum = pipe(kSum, sizeof kSum);
    p.mm_send[0] = pipe(kSendDev, sizeof kSendDev);
    p.mm_flag[0] = pipe(kFlagDev, sizeof kFlagDev);
    p.mm_spin[0] = pipe(kSpinDev, sizeof kSpinDev);
    p.mm_sum[0] = pipe(kSumDev, sizeof kSumDev);
    p.mm_send[1] = pipe(kSendQf, sizeof kSendQf);
    p.mm_flag[1] = pipe(kFlagQf, sizeof kFlagQf);
    p.mm_spin[1] = pipe(kSpinQf, sizeof kSpinQf);
    p.mm_sum[1] = pipe(kSumQf, sizeof kSumQf);
    return p;
}

// Exportable device memory of the uncached, device-coherent type, which bypasses the L2 of the card that owns it.
Buffer export_uncached(Device& d, VkDeviceSize bytes, int& fd) {
    Buffer b;
    b.buf = external_buffer(d, bytes, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT);
    VkMemoryRequirements req;
    d.vkGetBufferMemoryRequirements(d.dev, b.buf, &req);
    VkExportMemoryAllocateInfo ex{};
    ex.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.pNext = &ex;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = memory_type(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD);
    check(d.vkAllocateMemory(d.dev, &ai, nullptr, &b.mem), "vkAllocateMemory (uncached export)");
    check(d.vkBindBufferMemory(d.dev, b.buf, b.mem, 0), "vkBindBufferMemory (uncached export)");
    VkMemoryGetFdInfoKHR gi{};
    gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    gi.memory = b.mem;
    gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
    check(d.vkGetMemoryFdKHR(d.dev, &gi, &fd), "vkGetMemoryFdKHR");
    return b;
}

// One submission waiting on every semaphore of `waits` and signalling the timeline to `value` and every semaphore of `signals`.
void submit_all(Device& d, VkCommandBuffer cb, VkSemaphore timeline, uint64_t value, const std::vector<VkSemaphore>& waits, const std::vector<VkSemaphore>& signals) {
    std::vector<VkSemaphore> sig(signals);
    sig.push_back(timeline);
    std::vector<uint64_t> sig_values(sig.size(), 0), wait_values(waits.size(), 0);
    sig_values.back() = value;
    const std::vector<VkPipelineStageFlags> stages(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkTimelineSemaphoreSubmitInfo ti{};
    ti.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    ti.signalSemaphoreValueCount = (uint32_t)sig_values.size();
    ti.pSignalSemaphoreValues = sig_values.data();
    ti.waitSemaphoreValueCount = (uint32_t)wait_values.size();
    ti.pWaitSemaphoreValues = wait_values.data();
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &ti;
    si.commandBufferCount = cb ? 1 : 0;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = (uint32_t)sig.size();
    si.pSignalSemaphores = sig.data();
    si.waitSemaphoreCount = (uint32_t)waits.size();
    si.pWaitSemaphores = waits.data();
    si.pWaitDstStageMask = stages.data();
    check(d.vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
}

VkCommandBuffer begin_commands(Device& d) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = d.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    check(d.vkAllocateCommandBuffers(d.dev, &ai, &cb), "vkAllocateCommandBuffers");
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    check(d.vkBeginCommandBuffer(cb, &bi), "vkBeginCommandBuffer");
    return cb;
}

void compute_barrier(Device& d, VkCommandBuffer cb) {
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    d.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

// The all-reduce of a tensor group, measured on 2 to 4 devices (docs/TENSOR-SPLIT.md, step 0): in each epoch every member writes its F32 partial into slot `member` of every member's inbox, waits for the others and adds the slots in member order, every sum checked.
// Inboxes are uncached device memory exported as dma-buf (`device`) or host memory imported into every member (`host`).
// It times the dispatch floor with no peer, the exchange through sync files with one submission an epoch a member, and the members' arrival at each epoch on the host's clock, then tries a wait inside one submission on a flag written with the Vulkan memory model at device and at queue-family scope, which on RADV and gfx906 never sees a peer's writes (docs/TENSOR-SPLIT.md, section 2.6), so each spin is bounded at 2^16 reads and a timeout ends the chain's waits.
// With host inboxes it also times host-relayed events (docs/TENSOR-SPLIT.md, section 8): every epoch in one command buffer a member, its partials written into the inboxes, a barrier to the host, which the command processor's cache writeback serves, and an event the host polls; once every member's event of the epoch is set, a host thread sets each member's event that its sum waits on, so no member's submission waits on another's.
int exchange(const std::vector<int>& ids, int epochs, bool host_inboxes) {
    VkInstance inst = make_instance();
    const auto pds = physical_devices(inst);
    const uint32_t W = (uint32_t)ids.size();
    if (W < 2 || W > 4) throw std::runtime_error("the exchange takes 2 to 4 devices");
    std::vector<Device> dev;
    for (size_t a = 0; a < ids.size(); ++a) {
        if (ids[a] < 0 || ids[a] >= (int)pds.size()) throw std::runtime_error("no device " + std::to_string(ids[a]));
        for (size_t b = 0; b < a; ++b)
            if (ids[a] == ids[b]) throw std::runtime_error("a device is listed twice");
        dev.push_back(open_device(pds[ids[a]], true));
        const Device& d = dev.back();
        if (!d.dma_buf || !d.sync_fd || !d.uncached || (host_inboxes && !d.host_import)) throw std::runtime_error("needs dma-buf, sync files and the inbox memory on every device");
    }
    std::printf("width %u:", W);
    for (uint32_t m = 0; m < W; ++m) std::printf(" device %d (pci %s)", ids[m], identity(pds[ids[m]], extensions(pds[ids[m]])).pci.c_str());
    std::printf(", %d epochs a chain, inboxes in %s\n", epochs, host_inboxes ? "host memory imported into every member" : "uncached device memory exported as dma-buf");
    std::vector<ExchangePipes> pipes;
    for (auto& d : dev) pipes.push_back(exchange_pipes(d));
    const bool timestamps = std::all_of(dev.begin(), dev.end(), [](const Device& d) { return d.calibrated; });
    uint64_t wrong_total = 0;   // wrong sums of the floor and the sync-file exchange, which fail the run; a flag wait's timeouts are its result
    for (uint32_t n : {5120u, 40960u, 327680u, 2621440u}) {
        // Whole 64 KiB, so host memory meets every device's import alignment.
        const VkDeviceSize bytes = (((VkDeviceSize)2 * W * n + 2 * W * 64) * 4 + 65535) / 65536 * 65536;
        // inbox[t][d]: member t's inbox as device d addresses it; result[d], out[d]: d's counters and sums.
        std::vector<std::vector<Buffer>> inbox(W, std::vector<Buffer>(W));
        std::vector<Buffer> result(W), out(W);
        std::vector<void*> host(W, nullptr);
        for (uint32_t t = 0; t < W; ++t) {
            if (host_inboxes) {
                host[t] = host_alloc((size_t)bytes, 1 << 16);
                if (!host[t]) throw std::runtime_error("host allocation failed");
                std::memset(host[t], 0, (size_t)bytes);
                for (uint32_t d = 0; d < W; ++d)
                    inbox[t][d] = import_host(dev[d], host[t], bytes, dev[d].uncached ? VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD : 0);
                continue;
            }
            int fd = -1;
            inbox[t][t] = export_uncached(dev[t], bytes, fd);
            for (uint32_t d = 0; d < W; ++d) {
                if (d == t) continue;
                inbox[t][d] = import_dma_buf(dev[d], dup(fd), bytes, VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD);
            }
            close(fd);
        }
        for (uint32_t d = 0; d < W; ++d) {
            result[d] = make_buffer(dev[d], 64, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            out[d] = make_buffer(dev[d], (VkDeviceSize)n * 4, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        }
        // sets[d][t]: device d with member t's inbox as its target.
        std::vector<std::vector<VkDescriptorSet>> sets(W, std::vector<VkDescriptorSet>(W));
        for (uint32_t d = 0; d < W; ++d) {
            const VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * W};
            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = W;
            pci.poolSizeCount = 1;
            pci.pPoolSizes = &ps;
            VkDescriptorPool pool;
            check(dev[d].vkCreateDescriptorPool(dev[d].dev, &pci, nullptr, &pool), "vkCreateDescriptorPool");
            const std::vector<VkDescriptorSetLayout> layouts(W, pipes[d].set_layout);
            VkDescriptorSetAllocateInfo dai{};
            dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            dai.descriptorPool = pool;
            dai.descriptorSetCount = W;
            dai.pSetLayouts = layouts.data();
            check(dev[d].vkAllocateDescriptorSets(dev[d].dev, &dai, sets[d].data()), "vkAllocateDescriptorSets");
            for (uint32_t t = 0; t < W; ++t) {
                const VkDescriptorBufferInfo bi[4] = {{inbox[t][d].buf, 0, VK_WHOLE_SIZE}, {inbox[d][d].buf, 0, VK_WHOLE_SIZE},
                                                      {result[d].buf, 0, VK_WHOLE_SIZE}, {out[d].buf, 0, VK_WHOLE_SIZE}};
                VkWriteDescriptorSet w[4]{};
                for (uint32_t i = 0; i < 4; ++i) {
                    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w[i].dstSet = sets[d][t];
                    w[i].dstBinding = i;
                    w[i].descriptorCount = 1;
                    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    w[i].pBufferInfo = &bi[i];
                }
                dev[d].vkUpdateDescriptorSets(dev[d].dev, 4, w, 0, nullptr);
            }
        }
        const uint32_t groups = (n + 255) / 256;
        auto dispatch = [&](VkCommandBuffer cb, uint32_t d, uint32_t target, VkPipeline pl, uint32_t member, uint32_t e, uint32_t x) {
            const uint32_t pc[4] = {n, e, member, W};
            dev[d].vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl);
            dev[d].vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[d].layout, 0, 1, &sets[d][target], 0, nullptr);
            dev[d].vkCmdPushConstants(cb, pipes[d].layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
            dev[d].vkCmdDispatch(cb, x, 1, 1);
        };
        // Member d's partial into every inbox, or with `local` every member's partial into d's own, which is the floor with no peer.
        auto sends = [&](VkCommandBuffer cb, uint32_t d, uint32_t e, VkPipeline pl, bool local) {
            for (uint32_t t = 0; t < W; ++t) dispatch(cb, d, local ? d : t, pl, local ? t : d, e, groups);
            compute_barrier(dev[d], cb);
        };
        auto sum = [&](VkCommandBuffer cb, uint32_t d, uint32_t e, VkPipeline pl) {
            dispatch(cb, d, d, pl, d, e, groups);
            compute_barrier(dev[d], cb);
        };
        std::vector<VkSemaphore> tl(W);
        std::vector<uint64_t> v(W, 0);
        for (uint32_t d = 0; d < W; ++d) tl[d] = make_semaphore(dev[d], true);
        uint32_t epoch = 1;
        auto reset_results = [&]() { for (uint32_t d = 0; d < W; ++d) std::memset(result[d].map, 0, 64); };
        auto counter = [&](uint32_t d, int i) { return ((const uint32_t*)result[d].map)[i]; };
        auto mismatches = [&]() { uint64_t m = 0; for (uint32_t d = 0; d < W; ++d) m += counter(d, 3); return m; };
        auto timeouts = [&]() { uint64_t m = 0; for (uint32_t d = 0; d < W; ++d) m += counter(d, 0); return m; };
        auto run_one = [&](std::vector<VkCommandBuffer>& cb) {
            const double t0 = now_us();
            for (uint32_t d = 0; d < W; ++d) submit(dev[d], cb[d], tl[d], ++v[d]);
            for (uint32_t d = 0; d < W; ++d) wait_value(dev[d], tl[d], v[d]);
            return now_us() - t0;
        };
        // The floor: every epoch in one command buffer a member, no peer.
        auto local_chain = [&]() {
            std::vector<VkCommandBuffer> cb(W);
            for (uint32_t d = 0; d < W; ++d) {
                cb[d] = begin_commands(dev[d]);
                for (int i = 0; i < epochs; ++i) {
                    sends(cb[d], d, epoch + (uint32_t)i, pipes[d].send, true);
                    sum(cb[d], d, epoch + (uint32_t)i, pipes[d].sum);
                }
                check(dev[d].vkEndCommandBuffer(cb[d]), "vkEndCommandBuffer");
            }
            epoch += (uint32_t)epochs;
            return run_one(cb) / epochs;
        };
        // The flag wait: every epoch in one command buffer a member, its partial and flags released into the peers' inboxes and an acquiring spin on its own.
        auto flag_chain = [&](int scope, int count) {
            std::vector<VkCommandBuffer> cb(W);
            for (uint32_t d = 0; d < W; ++d) {
                cb[d] = begin_commands(dev[d]);
                for (int i = 0; i < count; ++i) {
                    const uint32_t e = epoch + (uint32_t)i;
                    sends(cb[d], d, e, pipes[d].mm_send[scope], false);
                    for (uint32_t t = 0; t < W; ++t)
                        if (t != d) dispatch(cb[d], d, t, pipes[d].mm_flag[scope], d, e, 1);
                    compute_barrier(dev[d], cb[d]);
                    dispatch(cb[d], d, d, pipes[d].mm_spin[scope], d, e, 1);
                    compute_barrier(dev[d], cb[d]);
                    sum(cb[d], d, e, pipes[d].mm_sum[scope]);
                }
                check(dev[d].vkEndCommandBuffer(cb[d]), "vkEndCommandBuffer");
            }
            epoch += (uint32_t)count;
            return run_one(cb) / count;
        };
        // Sync files: one submission an epoch a member, [sum of the epoch before, partial into every inbox, arrival timestamp], waiting on each peer's semaphore of the epoch before and signalling one to each peer.
        std::vector<VkQueryPool> pools(W, VK_NULL_HANDLE);
        if (timestamps)
            for (uint32_t d = 0; d < W; ++d) {
                VkQueryPoolCreateInfo qi{};
                qi.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
                qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
                qi.queryCount = (uint32_t)epochs;
                check(dev[d].vkCreateQueryPool(dev[d].dev, &qi, nullptr, &pools[d]), "vkCreateQueryPool");
            }
        std::vector<double> spread;
        double deviation_us = 0;   // the calibrations' largest uncertainty, beside the spread it bounds
        auto sync_chain = [&]() {
            std::vector<std::vector<VkCommandBuffer>> cbs(W);
            // sig[d][i][t]: d's semaphore for peer t after epoch i; wait[t][i][d]: t's import of it.
            std::vector<std::vector<std::vector<VkSemaphore>>> sig(W), wt(W);
            const uint32_t e0 = epoch;
            for (uint32_t d = 0; d < W; ++d) {
                sig[d].assign((size_t)epochs, std::vector<VkSemaphore>(W, VK_NULL_HANDLE));
                wt[d].assign((size_t)epochs, std::vector<VkSemaphore>(W, VK_NULL_HANDLE));
            }
            for (uint32_t d = 0; d < W; ++d) {
                for (int i = 0; i <= epochs; ++i) {
                    VkCommandBuffer cb = begin_commands(dev[d]);
                    if (i == 0 && timestamps) dev[d].vkCmdResetQueryPool(cb, pools[d], 0, (uint32_t)epochs);
                    if (i > 0) sum(cb, d, e0 + (uint32_t)i - 1, pipes[d].sum);
                    if (i < epochs) {
                        sends(cb, d, e0 + (uint32_t)i, pipes[d].send, false);
                        if (timestamps) dev[d].vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, pools[d], (uint32_t)i);
                        for (uint32_t t = 0; t < W; ++t)
                            if (t != d) {
                                sig[d][(size_t)i][t] = make_semaphore(dev[d], false, true);
                                wt[t][(size_t)i][d] = make_semaphore(dev[t], false);
                            }
                    }
                    check(dev[d].vkEndCommandBuffer(cb), "vkEndCommandBuffer");
                    cbs[d].push_back(cb);
                }
            }
            epoch += (uint32_t)epochs;
            const double t0 = now_us();
            for (int i = 0; i <= epochs; ++i)
                for (uint32_t d = 0; d < W; ++d) {
                    std::vector<VkSemaphore> waits, signals;
                    for (uint32_t t = 0; t < W; ++t) {
                        if (t == d) continue;
                        if (i > 0) waits.push_back(wt[d][(size_t)i - 1][t]);
                        if (i < epochs) signals.push_back(sig[d][(size_t)i][t]);
                    }
                    submit_all(dev[d], cbs[d][(size_t)i], tl[d], ++v[d], waits, signals);
                    for (uint32_t t = 0; i < epochs && t < W; ++t) {
                        if (t == d) continue;
                        VkSemaphoreGetFdInfoKHR gi{};
                        gi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
                        gi.semaphore = sig[d][(size_t)i][t];
                        gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                        int fd = -1;
                        check(dev[d].vkGetSemaphoreFdKHR(dev[d].dev, &gi, &fd), "vkGetSemaphoreFdKHR");
                        VkImportSemaphoreFdInfoKHR ii{};
                        ii.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
                        ii.semaphore = wt[t][(size_t)i][d];
                        ii.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
                        ii.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
                        ii.fd = fd;
                        check(dev[t].vkImportSemaphoreFdKHR(dev[t].dev, &ii), "vkImportSemaphoreFdKHR");
                    }
                }
            for (uint32_t d = 0; d < W; ++d) wait_value(dev[d], tl[d], v[d]);
            const double us = (now_us() - t0) / epochs;
            if (!timestamps) return us;
            // Each member's arrival placed on the host's clock through a calibration taken now, and the spread of the members' arrivals at every epoch.
            std::vector<std::vector<double>> at(W, std::vector<double>((size_t)epochs));
            for (uint32_t d = 0; d < W; ++d) {
                std::vector<uint64_t> ts((size_t)epochs);
                check(dev[d].vkGetQueryPoolResults(dev[d].dev, pools[d], 0, (uint32_t)epochs, ts.size() * 8, ts.data(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT), "vkGetQueryPoolResults");
                VkCalibratedTimestampInfoEXT ci[2]{};
                ci[0].sType = ci[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
                ci[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
                ci[1].timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT;
                uint64_t cal[2], deviation = 0;
                check(dev[d].vkGetCalibratedTimestampsEXT(dev[d].dev, 2, ci, cal, &deviation), "vkGetCalibratedTimestampsEXT");
                deviation_us = std::max(deviation_us, (double)deviation / 1000.0);
                VkPhysicalDeviceProperties props;
                vkGetPhysicalDeviceProperties(dev[d].pd, &props);
                for (int i = 0; i < epochs; ++i)
                    at[d][(size_t)i] = (double)cal[1] / 1000.0 + ((double)ts[(size_t)i] - (double)cal[0]) * props.limits.timestampPeriod / 1000.0;
            }
            for (int i = 0; i < epochs; ++i) {
                double lo = at[0][(size_t)i], hi = lo;
                for (uint32_t d = 1; d < W; ++d) lo = std::min(lo, at[d][(size_t)i]), hi = std::max(hi, at[d][(size_t)i]);
                spread.push_back(hi - lo);
            }
            return us;
        };
        // Host-relayed events: done[d][i] set by member d after its partials of epoch i reach host memory, go[d][i] set by the host once every member's done[i] is, which d's sum of epoch i waits on.
        // With `self` each member sets its own go event behind its done event, no host between them, which isolates what the barrier to the host and the events cost; such a chain does not order the members, so its sums are not counted.
        // Without `to_host` the barrier to the host is left out, which tells its cost from the events'.
        auto relay_chain = [&](bool self, bool to_host = true) {
            std::vector<std::vector<VkEvent>> done(W, std::vector<VkEvent>((size_t)epochs)), go(W, std::vector<VkEvent>((size_t)epochs));
            VkEventCreateInfo eci{};
            eci.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
            std::vector<VkCommandBuffer> cb(W);
            for (uint32_t d = 0; d < W; ++d) {
                for (int i = 0; i < epochs; ++i) {
                    check(dev[d].vkCreateEvent(dev[d].dev, &eci, nullptr, &done[d][(size_t)i]), "vkCreateEvent");
                    check(dev[d].vkCreateEvent(dev[d].dev, &eci, nullptr, &go[d][(size_t)i]), "vkCreateEvent");
                }
                cb[d] = begin_commands(dev[d]);
                for (int i = 0; i < epochs; ++i) {
                    const uint32_t e = epoch + (uint32_t)i;
                    sends(cb[d], d, e, pipes[d].send, false);
                    VkMemoryBarrier out{};
                    out.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    out.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    out.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    if (to_host) dev[d].vkCmdPipelineBarrier(cb[d], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &out, 0, nullptr, 0, nullptr);
                    dev[d].vkCmdSetEvent(cb[d], done[d][(size_t)i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                    if (self) dev[d].vkCmdSetEvent(cb[d], go[d][(size_t)i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                    VkMemoryBarrier in{};
                    in.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                    in.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                    in.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    dev[d].vkCmdWaitEvents(cb[d], 1, &go[d][(size_t)i], VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 1, &in, 0, nullptr, 0, nullptr);
                    sum(cb[d], d, e, pipes[d].sum);
                }
                check(dev[d].vkEndCommandBuffer(cb[d]), "vkEndCommandBuffer");
            }
            epoch += (uint32_t)epochs;
            const double t0 = now_us();
            for (uint32_t d = 0; d < W; ++d) submit(dev[d], cb[d], tl[d], ++v[d]);
            for (int i = 0; !self && i < epochs; ++i) {
                for (uint32_t d = 0; d < W; ++d)
                    while (dev[d].vkGetEventStatus(dev[d].dev, done[d][(size_t)i]) != VK_EVENT_SET) {}
                for (uint32_t d = 0; d < W; ++d) check(dev[d].vkSetEvent(dev[d].dev, go[d][(size_t)i]), "vkSetEvent");
            }
            for (uint32_t d = 0; d < W; ++d) wait_value(dev[d], tl[d], v[d]);
            return (now_us() - t0) / epochs;
        };
        std::vector<double> floor_us, sync_us, relay_us, self_us, bare_us;
        reset_results();
        local_chain();
        sync_chain();
        if (host_inboxes) relay_chain(false);
        spread.clear();
        for (int r = 0; r < 5; ++r) {
            floor_us.push_back(local_chain());
            sync_us.push_back(sync_chain());
            if (host_inboxes) relay_us.push_back(relay_chain(false));
        }
        const uint64_t wrong = mismatches();
        wrong_total += wrong;
        // The chains without the host do not order the members, so their sums go uncounted.
        for (int r = 0; host_inboxes && r < 5; ++r) {
            self_us.push_back(relay_chain(true));
            bare_us.push_back(relay_chain(true, false));
        }
        reset_results();
        std::printf("%9u floats (%7.1f KB): us an epoch, median of 5 chains: floor %.1f, sync files %.1f", n, n * 4 / 1024.0, stats(floor_us).median, stats(sync_us).median);
        if (host_inboxes) std::printf(", host-relayed events %.1f, the same barriers and events with no host between them %.1f, those events without the barrier to the host %.1f",
                                      stats(relay_us).median, stats(self_us).median, stats(bare_us).median);
        if (timestamps) std::printf("; members' arrival spread median %.1f, p90 %.1f, calibration uncertainty up to %.1f", stats(spread).median, stats(spread).p90, deviation_us);
        std::printf("; wrong sums %llu\n", (unsigned long long)wrong);
        for (int scope = 0; scope < 2; ++scope) {
            reset_results();
            const double us = flag_chain(scope, 8);
            const uint64_t missed = timeouts();
            std::printf("    flag wait, %s scope: %s", scope ? "queue-family" : "device", missed ? "the peers' flags never arrived inside the submission" : "the peers' flags arrived");
            if (!missed) {
                std::vector<double> t;
                reset_results();
                for (int r = 0; r < 5; ++r) t.push_back(flag_chain(scope, epochs));
                std::printf(", %.1f us an epoch, median of 5 chains, %llu timeouts", stats(t).median, (unsigned long long)timeouts());
            } else {
                std::printf(" (8 epochs, %.0f us an epoch with the bounded spin)", us);
            }
            std::printf(", wrong sums %llu\n", (unsigned long long)mismatches());
        }
        // The buffers stay alive, imported host memory included, until the process ends.
        for (auto& d : dev) d.vkDeviceWaitIdle(d.dev);
    }
    if (wrong_total) std::fprintf(stderr, "llmx-vk-handoff: %llu sums were wrong\n", (unsigned long long)wrong_total);
    return wrong_total ? 1 : 0;
}
#endif

} // namespace

int main(int argc, char** argv) {
    try {
        open_loader();
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "probe") return probe();
        if (mode == "time" && argc >= 4) return time_handoff(std::atoi(argv[2]), std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 200);
#if !defined(_WIN32)
        if (mode == "pingpong" && argc >= 4) return pingpong(std::atoi(argv[2]), std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 200);
        if (mode == "exchange" && argc >= 3 && argc <= 5) {
            // A whole decimal number in [lo, hi], or a usage error.
            auto number = [](const std::string& t, long lo, long hi) {
                char* end = nullptr;
                const long v = t.empty() ? -1 : std::strtol(t.c_str(), &end, 10);
                if (t.empty() || *end || v < lo || v > hi) throw std::runtime_error("not a number from " + std::to_string(lo) + " to " + std::to_string(hi) + ": " + t);
                return (int)v;
            };
            std::vector<int> ids;
            std::string list = argv[2];
            for (size_t at = 0;;) {
                const size_t comma = list.find(',', at);
                ids.push_back(number(list.substr(at, comma - at), 0, 63));
                if (comma == std::string::npos) break;
                at = comma + 1;
            }
            const int epochs = argc > 3 ? number(argv[3], 1, 100000) : 200;
            const std::string where = argc > 4 ? argv[4] : "device";
            if (where != "device" && where != "host") throw std::runtime_error("inboxes are device or host");
            return exchange(ids, epochs, where == "host");
        }
#endif
        std::fprintf(stderr, "usage: llmx-vk-handoff probe | llmx-vk-handoff time A B [iterations] | llmx-vk-handoff pingpong A B [hops] | llmx-vk-handoff exchange A,B[,C,D] [epochs] [device|host]\n");
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-vk-handoff: %s\n", e.what());
        return 1;
    }
}
