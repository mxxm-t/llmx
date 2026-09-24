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
// Usage: llmx-vk-handoff probe | llmx-vk-handoff time A B [iterations]
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
    X(vkGetDeviceGroupPeerMemoryFeatures)

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
    bool host_import = false, sync_fd = false, dma_buf = false;
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
};

Device open_device(VkPhysicalDevice pd) {
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
Buffer import_host(Device& d, void* host, VkDeviceSize bytes) {
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
    ai.memoryTypeIndex = memory_type(d, hp.memoryTypeBits & req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
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
Buffer import_dma_buf(Device& d, int fd, VkDeviceSize bytes) {
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
    ai.memoryTypeIndex = memory_type(d, fp.memoryTypeBits & req.memoryTypeBits, 0);
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
    VkSemaphore bina = sync_fd ? make_semaphore(A, false, true) : VK_NULL_HANDLE;
    VkSemaphore binb = sync_fd ? make_semaphore(B, false) : VK_NULL_HANDLE;
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

} // namespace

int main(int argc, char** argv) {
    try {
        open_loader();
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "probe") return probe();
        if (mode == "time" && argc >= 4) return time_handoff(std::atoi(argv[2]), std::atoi(argv[3]), argc > 4 ? std::atoi(argv[4]) : 200);
        std::fprintf(stderr, "usage: llmx-vk-handoff probe | llmx-vk-handoff time A B [iterations]\n");
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-vk-handoff: %s\n", e.what());
        return 1;
    }
}
