// What a new Vulkan driver must show before devices share work through it (docs/TENSOR-SPLIT.md, step 0).
// `probe` lists per device the external handle types, host-pointer import, identifiers, memory budget and device groups; `exchange A,B[,C,D] [epochs]` is a tensor group's all-reduce on 2 to 4 devices.
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
    X(vkWaitSemaphores) \
    X(vkCreateBuffer) \
    X(vkDestroyBuffer) \
    X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) \
    X(vkFreeMemory) \
    X(vkBindBufferMemory) \
    X(vkMapMemory) \
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
    X(vkGetQueryPoolResults)

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

// The all-reduce of a tensor group on 2 to 4 devices (docs/TENSOR-SPLIT.md, step 0): each epoch every member writes its F32 partial into slot `member` of every member's inbox (uncached dma-buf memory) and adds the slots in member order.
// It times the dispatch floor with no peer, the exchange through sync files and the members' arrival, then a flag wait inside one submission, which on RADV and gfx906 never sees a peer's writes (docs/TENSOR-SPLIT.md, 2.6).
int exchange(const std::vector<int>& ids, int epochs) {
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
        if (!d.dma_buf || !d.sync_fd || !d.uncached) throw std::runtime_error("needs dma-buf, sync files and the inbox memory on every device");
    }
    std::printf("width %u:", W);
    for (uint32_t m = 0; m < W; ++m) std::printf(" device %d (pci %s)", ids[m], identity(pds[ids[m]], extensions(pds[ids[m]])).pci.c_str());
    std::printf(", %d epochs a chain, inboxes in uncached device memory exported as dma-buf\n", epochs);
    std::vector<ExchangePipes> pipes;
    for (auto& d : dev) pipes.push_back(exchange_pipes(d));
    const bool timestamps = std::all_of(dev.begin(), dev.end(), [](const Device& d) { return d.calibrated; });
    uint64_t wrong_total = 0;   // wrong sums of the floor and the sync-file exchange, which fail the run; a flag wait's timeouts are its result
    for (uint32_t n : {5120u, 40960u, 327680u, 2621440u}) {
        const VkDeviceSize bytes = (((VkDeviceSize)2 * W * n + 2 * W * 64) * 4 + 65535) / 65536 * 65536;
        // inbox[t][d]: member t's inbox as device d addresses it; result[d], out[d]: d's counters and sums.
        std::vector<std::vector<Buffer>> inbox(W, std::vector<Buffer>(W));
        std::vector<Buffer> result(W), out(W);
        for (uint32_t t = 0; t < W; ++t) {
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
        std::vector<double> floor_us, sync_us;
        reset_results();
        local_chain();
        sync_chain();
        spread.clear();
        for (int r = 0; r < 5; ++r) {
            floor_us.push_back(local_chain());
            sync_us.push_back(sync_chain());
        }
        const uint64_t wrong = mismatches();
        wrong_total += wrong;
        reset_results();
        std::printf("%9u floats (%7.1f KB): us an epoch, median of 5 chains: floor %.1f, sync files %.1f", n, n * 4 / 1024.0, stats(floor_us).median, stats(sync_us).median);
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
        // The buffers stay alive until the process ends.
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
#if !defined(_WIN32)
        if (mode == "exchange" && argc >= 3 && argc <= 4) {
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
            return exchange(ids, epochs);
        }
#endif
        std::fprintf(stderr, "usage: llmx-vk-handoff probe | llmx-vk-handoff exchange A,B[,C,D] [epochs]\n");
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-vk-handoff: %s\n", e.what());
        return 1;
    }
}
