// Vulkan backend over one compute queue (docs/VULKAN.md).
// Operations record into a command-buffer ring whose slots retire by timeline tickets.
#include "backends/device_profile.hpp"
#include "backends/kv_storage.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_device.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"
#include "quant/quant.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace backend {
namespace {

struct MoeBuild;   // the fused routed step's builds (vulkan_moe.hpp)

// The one thread that makes a tensor group's queue calls, in the order they are handed over, so the recording thread hands a submission over and records on (docs/TENSOR-SPLIT.md, section 4.3).
// A call that fails settles its own tickets and every later job's without calling, so no waiter hangs; the next hand-over runs the owners' recovery and throws the failure, and the jobs after it run again.
class VulkanSubmitter {
public:
    // `call` makes the queue calls; `settle` marks the job's tickets made without a call, for a job skipped after a failure or one whose call threw.
    struct Job {
        std::function<void()> call, settle;
    };

    VulkanSubmitter() : thread_([this] { loop(); }) {}
    ~VulkanSubmitter() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        thread_.join();
    }
    VulkanSubmitter(const VulkanSubmitter&) = delete;
    VulkanSubmitter& operator=(const VulkanSubmitter&) = delete;

    // On a recording thread; after a failure the job is settled uncalled, and the failure given out (raise) unless `report` is false.
    void run(Job job, bool report = true) {
        std::unique_lock<std::mutex> lk(mu_);
        if (failed_) {
            lk.unlock();
            job.settle();
            if (report) raise();
            return;
        }
        jobs_.push_back(std::move(job));
        ++posted_;
        cv_.notify_all();
    }
    // On a recording thread: a failure not yet given out is given here, once the jobs handed over before it are settled and the owners recovered.
    void raise() {
        std::unique_lock<std::mutex> lk(mu_);
        if (!failed_) return;
        const std::exception_ptr e = failed_;
        const uint64_t target = posted_;
        cv_.wait(lk, [&] { return ran_ >= target; });
        failed_ = nullptr;
        std::vector<std::function<void()>> recover;
        for (const auto& o : owners_) recover.push_back(o.second);
        lk.unlock();
        for (const auto& r : recover) r();
        std::rethrow_exception(e);
    }
    // Until every job handed over so far has been called or settled.
    void drain() noexcept {
        std::unique_lock<std::mutex> lk(mu_);
        const uint64_t target = posted_;
        cv_.wait(lk, [&] { return ran_ >= target; });
    }
    // Until `pred` holds; `pred` reads what the jobs set, which every finished job notifies.
    template <class Pred> void until(Pred pred) noexcept {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return pred(); });
    }
    // An owner's recovery, run on the recording thread that is given a failure, before any later job; removed with the owner.
    void own(const void* owner, std::function<void()> recover) {
        std::lock_guard<std::mutex> lk(mu_);
        owners_.emplace_back(owner, std::move(recover));
    }
    void disown(const void* owner) {
        std::lock_guard<std::mutex> lk(mu_);
        owners_.erase(std::remove_if(owners_.begin(), owners_.end(), [&](const auto& o) { return o.first == owner; }), owners_.end());
    }

private:
    void loop() {
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || !jobs_.empty(); });
            if (jobs_.empty()) return;
            Job job = std::move(jobs_.front());
            jobs_.pop_front();
            // Jobs handed over before a failure was given out depend on what failed, so they are settled uncalled.
            const bool skip = failed_ != nullptr;
            lk.unlock();
            std::exception_ptr e;
            if (skip) {
                job.settle();
            } else {
                try { job.call(); } catch (...) {
                    e = std::current_exception();
                }
                // The failure is recorded before the tickets are settled, so a waiter that wakes on them finds it at its next hand-over.
                if (e) {
                    lk.lock();
                    if (!failed_) failed_ = e;
                    lk.unlock();
                    job.settle();
                }
            }
            lk.lock();
            ++ran_;
            cv_.notify_all();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    uint64_t posted_ = 0, ran_ = 0;
    std::exception_ptr failed_;   // a failure not yet given out
    std::vector<std::pair<const void*, std::function<void()>>> owners_;
    bool stop_ = false;
    std::thread thread_;
};

class VulkanBackend final : public Backend {
    friend struct VulkanLifetimeTest;
    friend struct VulkanQuantizationTest;
    friend struct VulkanIndexTest;
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
        bool has_pci = false;
        for (const auto& e : core_exts) {
            if (std::strcmp(e.extensionName, "VK_AMD_shader_core_properties") == 0) has_core_props = true;
            if (std::strcmp(e.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0) has_pci = true;
        }
        VkPhysicalDeviceShaderCorePropertiesAMD core{};
        core.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
        if (has_core_props) {
            core.pNext = p2.pNext;
            p2.pNext = &core;
        }
        VkPhysicalDevicePCIBusInfoPropertiesEXT pci{};
        pci.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
        if (has_pci) {
            pci.pNext = p2.pNext;
            p2.pNext = &pci;
        }
        fn.vkGetPhysicalDeviceProperties2(d.physical, &p2);
        if (has_pci) d.pci_root = pci_root_of(pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
        d.float_controls = float_control_support(floats);
        d.preserve_float32 = (d.float_controls & kPreserve32) == kPreserve32;
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
        // The pipeline cache id changes with every build of the driver, also one that keeps its version strings, so a history kept on disk is not read back under a driver that may compile other kernels.
        std::string cache;
        for (const uint8_t byte : d.props.pipelineCacheUUID) {
            cache += "0123456789abcdef"[byte >> 4];
            cache += "0123456789abcdef"[byte & 15];
        }
        d.identity = "vulkan " + std::to_string(d.props.vendorID) + ":" + std::to_string(d.props.deviceID) + " " + d.caps.device + "; " + d.caps.driver + " " +
                     drv.driverInfo + " " + std::to_string(d.props.driverVersion) + " " + cache;
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
        d.f16_inputs = f12.shaderFloat16 && (d.float_controls & kF16InputControls) == kF16InputControls;
        e12.storageBuffer8BitAccess = VK_TRUE;
        d.workgroup_counter = missing_counter_need(sg.subgroupSize).empty();
        VkPhysicalDeviceVulkan11Features e11{};
        e11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        e11.pNext = &e12;
        e11.storageBuffer16BitAccess = VK_TRUE;
        VkPhysicalDeviceFeatures2 e2{};
        e2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        e2.pNext = &e11;
        e2.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
        e2.features.shaderInt16 = VK_TRUE;
        d.mxfp4 = f2.features.shaderFloat64 && (d.float_controls & kMxfp4Controls) == kMxfp4Controls;
        e2.features.shaderFloat64 = d.mxfp4 ? VK_TRUE : VK_FALSE;

        uint32_t ext_count = 0;
        check(fn.vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &ext_count, nullptr),
              "vkEnumerateDeviceExtensionProperties");
        std::vector<VkExtensionProperties> exts(ext_count);
        check(fn.vkEnumerateDeviceExtensionProperties(d.physical, nullptr, &ext_count, exts.data()),
              "vkEnumerateDeviceExtensionProperties");
        std::vector<const char*> enabled;
        int exchange_exts = 0;
        for (const auto& e : exts)
            if (std::strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) == 0 ||
                std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME) == 0 ||
                std::strcmp(e.extensionName, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) == 0 ||
                std::strcmp(e.extensionName, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME) == 0) {
                ++exchange_exts;
            } else if (std::strcmp(e.extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) {
                enabled.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
                d.push_descriptor = true;
            } else if (std::strcmp(e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) {
                // Core in 1.3, an extension on the 1.2 devices this targets; enabled where present so the dot-form kernels can run.
                enabled.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
                d.caps.integer_dot = true;
            } else if (std::strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0) {
                enabled.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
                d.exec_stats = true;
            } else if (std::strcmp(e.extensionName, VK_KHR_SHADER_CLOCK_EXTENSION_NAME) == 0) {
                VkPhysicalDeviceShaderClockFeaturesKHR clock{};
                clock.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
                VkPhysicalDeviceFeatures2 q{};
                q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                q.pNext = &clock;
                fn.vkGetPhysicalDeviceFeatures2(d.physical, &q);
                if (clock.shaderDeviceClock) {
                    enabled.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
                    d.device_clock = true;
                }
            } else if (std::strcmp(e.extensionName, VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME) == 0) {
                VkPhysicalDeviceCoherentMemoryFeaturesAMD coherent{};
                coherent.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;
                VkPhysicalDeviceFeatures2 q{};
                q.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                q.pNext = &coherent;
                fn.vkGetPhysicalDeviceFeatures2(d.physical, &q);
                if (coherent.deviceCoherentMemory) {
                    enabled.push_back(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
                    d.uncached_memory = true;
                }
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
#if !defined(_WIN32)
        if (exchange_exts == 4) {
            enabled.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            enabled.push_back(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
            enabled.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
            enabled.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
            d.exchange = true;
        }
#endif
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
        VkPhysicalDeviceShaderClockFeaturesKHR eclock{};
        eclock.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR;
        eclock.shaderDeviceClock = VK_TRUE;
        if (d.device_clock) {
            eclock.pNext = e2.pNext;
            e2.pNext = &eclock;
        }
        VkPhysicalDeviceCoherentMemoryFeaturesAMD ecoherent{};
        ecoherent.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COHERENT_MEMORY_FEATURES_AMD;
        ecoherent.deviceCoherentMemory = VK_TRUE;
        if (d.uncached_memory) {
            ecoherent.pNext = e2.pNext;
            e2.pNext = &ecoherent;
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
        if (d.exchange) {
            d.get_memory_fd = (PFN_vkGetMemoryFdKHR)fn.vkGetDeviceProcAddr(d.device, "vkGetMemoryFdKHR");
            d.memory_fd_props = (PFN_vkGetMemoryFdPropertiesKHR)fn.vkGetDeviceProcAddr(d.device, "vkGetMemoryFdPropertiesKHR");
            d.get_semaphore_fd = (PFN_vkGetSemaphoreFdKHR)fn.vkGetDeviceProcAddr(d.device, "vkGetSemaphoreFdKHR");
            d.import_semaphore_fd = (PFN_vkImportSemaphoreFdKHR)fn.vkGetDeviceProcAddr(d.device, "vkImportSemaphoreFdKHR");
            d.exchange = d.get_memory_fd && d.memory_fd_props && d.get_semaphore_fd && d.import_semaphore_fd;
        }
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
        if (submitter_) submitter_->drain();
        hold_release();
        d.fn.vkDeviceWaitIdle(d.device);
        for (auto& p : pending_) p.clear();
        for (auto& a : arena_) a.buffer.reset();
        for (auto& variants : role_free_)
            for (VkPipeline p : variants) if (p) d.fn.vkDestroyPipeline(d.device, p, nullptr);
        for (auto& variants : kernels_)
            for (Kernel& k : variants) destroy_kernel(k);
        for (VkQueryPool q : queries_) d.fn.vkDestroyQueryPool(d.device, q, nullptr);
        staging_.reset();
        if (timeline_) d.fn.vkDestroySemaphore(d.device, timeline_, nullptr);
        hold_.word.reset();
        if (hold_.timeline) d.fn.vkDestroySemaphore(d.device, hold_.timeline, nullptr);
        if (hold_.pool) d.fn.vkDestroyCommandPool(d.device, hold_.pool, nullptr);
        if (pool_) d.fn.vkDestroyCommandPool(d.device, pool_, nullptr);
    }

    // A compiled kernel's name, a row kernel's builds other than the wide one marked by their columns or as grouped.
    static std::string kernel_variant_name(int id, int variant) {
        const bool tile = is_tile_kernel(static_cast<KernelId>(id));
        if (variant == 2) return std::string(kKernelNames[id]) + (tile ? "_narrow" : "_grouped");
        if (is_vec_kernel((KernelId)id) && variant == kVecOneRow) return std::string(kKernelNames[id]) + "_1col_1row";
        if (is_vec_kernel((KernelId)id) && variant == kVecTwoSteps) return std::string(kKernelNames[id]) + "_1col_2step";
        if (is_vec_kernel((KernelId)id) && variant == kVecPaired) return std::string(kKernelNames[id]) + "_1col_paired";
        if (is_vec_kernel((KernelId)id) && std::find(std::begin(kVecFewerRows), std::end(kVecFewerRows), variant) != std::end(kVecFewerRows)) {
            const VecBuild& b = kVecBuilds[variant];
            return std::string(kKernelNames[id]) + "_" + std::to_string(b.capacity()) + "col_" + std::to_string(b.rows) + "row" + (b.span > 1 ? "_span" + std::to_string(b.span) : "");
        }
        if (variant && is_row_kernel((KernelId)id)) return std::string(kKernelNames[id]) + "_" + std::to_string(build_cols((KernelId)id, variant)) + "col";
        if (moe_step_kernel((KernelId)id)) return std::string(kKernelNames[id]) + (variant == kVecOneRow ? "_1row" : variant == kVecTwoSteps ? "_2step" : variant == 1 ? "_1col" : "");
        return std::string(kKernelNames[id]) + (!variant ? "" : tile ? "_small" : "_x8");
    }

    bool supports_type(uint32_t type) const override {
        const WeightKernels* format = weight_kernels(type);
        return format && (!format->mxfp4 || dev_->mxfp4);
    }
    // Native dtype support is device-wide; each weight descriptor selects its build or a wider fallback.
    std::vector<Dtype> native_dtypes() const override {
        if (dev_->profile.prefer_integer_dot) return {Dtype::f16, Dtype::f32, Dtype::int8};
        return {Dtype::f16, Dtype::f32};
    }
    bool emulates_dtype(Dtype dtype) const override { return dtype == Dtype::bf16 && dev_->preserve_float32; }
    std::string dtype_path(Dtype dtype) const override {
        if (dtype == Dtype::bf16) return "bf16 (matrix inputs), f32 (routers)";
        if (dtype == Dtype::int8) return "block-int8 (quantized rows and tiles but Q2_K/Q3_K/MXFP4/TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ4_NL/IQ4_XS/IQ3_S), block-int16 (Q2_K/Q3_K rows on integer-dot profiles and eligible tiles; MXFP4, TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS rows on lookup-dot profiles and eligible tiles), f32 (Q2_K/Q3_K, TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS fallback rows, other products, routers)";
        if (dtype == Dtype::f16)
            return std::string(dev_->f16_inputs ? "f16 (NVFP4 tile inputs), " : "") + (dev_->profile.prefer_integer_dot
                ? "block-int16 (quantized rows and eligible tiles; TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS rows on lookup-dot profiles), f32 (TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS fallback, other products, routers)"
                : "block-int16 (quantized rows except Q2_K/Q3_K; TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS rows on lookup-dot profiles), f32 (Q2_K/Q3_K, TQ1_0/TQ2_0/IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ3_XXS/IQ2_S/IQ3_S/IQ4_NL/IQ4_XS fallback rows, tiles, F32 weights, routers)");
        return "f32";
    }
    const std::string& name() const { return dev_->caps.device; }
    std::string identity() const override { return dev_->identity; }
    const DeviceProfile& profile() const { return dev_->profile; }
    bool prefer_integer_dot() {
        if (!dev_->caps.integer_dot) return false;
        dev_->profile.prefer_integer_dot = true;
        return true;
    }

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
        // A pipeline's internal representations as text, appended to `text`; false where the driver reports none.
        const auto pipeline_text = [&d](VkPipeline pipeline, std::string& text) {
            VkPipelineInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR;
            pi.pipeline = pipeline;
            uint32_t n = 0;
            if (d.get_exec_props(d.device, &pi, &n, nullptr) != VK_SUCCESS) return false;
            for (uint32_t e = 0; e < n; ++e) {
                VkPipelineExecutableInfoKHR ei{};
                ei.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR;
                ei.pipeline = pipeline;
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
            return true;
        };
        for (int slot = 0; slot < K_COUNT * kVariants; ++slot) {
            const int id = slot / kVariants, variant = slot % kVariants;
            const Kernel& k = kernels_[id][variant];
            if (!k.pipeline) continue;
            std::string text;
            if (is_vec_kernel((KernelId)id)) {
                const VecBuild& b = kVecBuilds[variant];
                const uint32_t forms = vec_forms(variant, d.profile);
                text = "; q8_decode_build cols=" + std::to_string(b.cols) + " rows=" + std::to_string(b.rows) + " steps=" + std::to_string(b.steps) +
                       " tree=" + std::to_string((forms & kQ8Tree) ? 1 : 0) + " half=" + std::to_string((forms & kQ8Half) ? 1 : 0) +
                       " halves=" + std::to_string(b.halves) + "\n";
            } else if (build_rows((KernelId)id, variant) > 1) {
                text = "; row_build cols=" + std::to_string(build_cols((KernelId)id, variant)) + " rows=" + std::to_string(build_rows((KernelId)id, variant)) +
                       " tree=" + std::to_string(row_tree_cluster((KernelId)id, variant, d.profile, d.caps.subgroup_size)) + "\n";
            }
            if (!pipeline_text(k.pipeline, text)) continue;
            out.emplace_back(kernel_variant_name(id, variant), std::move(text));
            std::string bare;
            if (role_free_[id][variant] && pipeline_text(role_free_[id][variant], bare)) out.emplace_back(kernel_variant_name(id, variant) + "_noroles", std::move(bare));
        }
        return out;
    }

    // State, gating and the vision encoder's ops are supported; grouped expert projections must share one storage type.
    bool implements(Op op) const override { return op != Op::mixed_experts; }

    // Host worker counts mean nothing to a device.
    void set_threads(int) override {}
    int threads_available() const override { return 0; }

    std::optional<size_t> memory_available() const override;
    bool memory_is_host() const override;
    size_t host_resident() const override;

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
    size_t scratch_reserve(size_t free) const override;
    size_t resident_bytes(uint32_t type, size_t nin, size_t rows, size_t bytes, bool product) const override;
    void drop_tags();
    void forget_written(VkBuffer buffer, VkDeviceSize begin, VkDeviceSize end);
    BufferPtr alloc(size_t bytes, Memory where) override;
    BufferPtr alloc_weight(size_t bytes) override;
    BufferPtr wrap_host(void* memory, size_t bytes) override;
    BufferPtr adopt(const void* src, size_t bytes) override;

    Ticket submit() override {
        const Ticket ticket = flush();
        if (hold_.on) hold_queue();
        return ticket;
    }

    // The collective splits flush into prepare, queue and queued so members can submit concurrently, signalling peer semaphores beside their timelines.
    // It adds no hold after a sum; the stage's own submission holds between stages.
    void wait_on(VkSemaphore s) { waits_.push_back(s); }

    // The open command buffer ended and its submission filled, with the ticket it will carry.
    // From here to queued() the queue is lent: only Submission::queue may be called for this backend, from any one thread, and open() and prepare() refuse.
    struct Submission {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        Ticket ticket = 0;
        bool dispatches = false;   // the command buffer holds a dispatch
        std::vector<VkSemaphore> signal, waits;
        std::atomic<uint32_t>* release = nullptr;   // on a submitter, the hold this submission lets go, released just before its call
        uint32_t release_seq = 0;

        // The queue's call alone; it touches nothing of the backend but the device's queue and a held queue's release word.
        VkResult queue(const Device& d) const {
            if (release) release->store(release_seq);
            std::vector<uint64_t> values(signal.size(), 0), wait_values(waits.size(), 0);
            values[0] = ticket;
            const std::vector<VkPipelineStageFlags> stages(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            VkTimelineSemaphoreSubmitInfo tsi{};
            tsi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
            tsi.signalSemaphoreValueCount = (uint32_t)values.size();
            tsi.pSignalSemaphoreValues = values.data();
            tsi.waitSemaphoreValueCount = (uint32_t)wait_values.size();
            tsi.pWaitSemaphoreValues = wait_values.data();
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.pNext = &tsi;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            si.signalSemaphoreCount = (uint32_t)signal.size();
            si.pSignalSemaphores = signal.data();
            si.waitSemaphoreCount = (uint32_t)waits.size();
            si.pWaitSemaphores = waits.data();
            si.pWaitDstStageMask = stages.data();
            return d.fn.vkQueueSubmit(d.queue, 1, &si, VK_NULL_HANDLE);
        }
    };

    Submission prepare(const std::vector<VkSemaphore>& signals) {
        Submission s;
        if (hold_.on) {
            // On a submitter the release travels with the submission, so the hold spins until the next work reaches the queue.
            if (submitter_ && hold_.pending) {
                s.release = reinterpret_cast<std::atomic<uint32_t>*>(hold_.word->mapped());
                s.release_seq = hold_.seq;
                hold_.pending = false;
            } else {
                hold_release();
            }
        }
        s.cmd = open();
        s.dispatches = chunk_ > 0;
        chunk_ = 0;
        check(dev_->fn.vkEndCommandBuffer(s.cmd), "vkEndCommandBuffer");
        s.signal.reserve(1 + signals.size());
        s.signal.push_back(timeline_);
        s.signal.insert(s.signal.end(), signals.begin(), signals.end());
        s.waits.swap(waits_);
        open_ = false;
        s.ticket = tickets_.take();
        lent_.store(true, std::memory_order_relaxed);
        return s;
    }

    // The queue taken back with what its call returned: the ticket, or the call's failure with the ticket given up.
    Ticket queued(const Submission& s, VkResult r) {
        lent_.store(false, std::memory_order_relaxed);
        if (r != VK_SUCCESS) {
            tickets_.give_back();
            check(r, "vkQueueSubmit");
        }
        ring_ticket_[ring_index_] = s.ticket;
        if (s.dispatches) dispatched_ = s.ticket;
        ring_index_ = (ring_index_ + 1) % kRing;
        return s.ticket;
    }
    const std::shared_ptr<Device>& device() const { return dev_; }
    std::unique_ptr<Collective> join(const std::vector<Backend*>& members, size_t rows, size_t width) override;
    std::string pci_root() const override { return dev_->pci_root; }
    size_t collective_bytes_per_row(size_t members, size_t width) const override;
    static size_t two_shot_crossover(size_t members);

    // While held, each submission is followed by a wait on the device that the next submission or kHoldMs of the device's clock ends, keeping the device busy between stages (Backend::hold_between_submissions).
    // A device whose shaders cannot read its clock keeps no hold, since a wait it cannot bound outlives a killed process.
    void hold_between_submissions(bool on) override {
        if (!on) {
            if (!hold_.holders || --hold_.holders) return;
            // A submission follows the release, whose start makes it visible at once, where the atomic alone took up to 3 ms on an MI50.
            if (hold_.pending) flush();
            hold_.on = false;
            return;
        }
        if (hold_.refused) return;
        if (!dev_->device_clock) {
            std::fprintf(stderr, "vulkan: %s keeps no hold between submissions, its shaders reading no device clock\n", dev_->caps.device.c_str());
            hold_.refused = true;
            return;
        }
        if (!hold_.timeline) {
            hold_setup();
            if (!hold_seen()) {
                std::fprintf(stderr, "vulkan: %s keeps no hold between submissions, a hold not seeing its release\n", dev_->caps.device.c_str());
                hold_.refused = true;
                return;
            }
        }
        ++hold_.holders;
        hold_.on = true;
    }

    // The hold's command buffers, timeline, release word and kernel, made once; a failure part way destroys what it made, so a later call starts again.
    void hold_setup() {
        Fn& fn = dev_->fn;
        VkCommandBuffer cmds[kHolds] = {};
        VkCommandPool pool = VK_NULL_HANDLE;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::unique_ptr<VulkanBuffer> word;
        try {
            // A pool of their own, since a submitter records them on its thread while this one records the ring's.
            VkCommandPoolCreateInfo pi{};
            pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            pi.queueFamilyIndex = dev_->queue_family;
            check(fn.vkCreateCommandPool(dev_->device, &pi, nullptr, &pool), "vkCreateCommandPool");
            VkCommandBufferAllocateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ci.commandPool = pool;
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
            // The release counter, in host memory the device reads past its caches where it offers such a type.
            word = std::make_unique<VulkanBuffer>(dev_, 2 * sizeof(uint32_t), true, true);
            kernel(K_HOLD, 0);
        } catch (...) {
            word.reset();
            if (timeline) fn.vkDestroySemaphore(dev_->device, timeline, nullptr);
            if (cmds[0]) fn.vkFreeCommandBuffers(dev_->device, pool, kHolds, cmds);
            if (pool) fn.vkDestroyCommandPool(dev_->device, pool, nullptr);
            throw;
        }
        std::copy(cmds, cmds + kHolds, hold_.cmds);
        hold_.pool = pool;
        hold_.timeline = timeline;
        hold_.word = std::move(word);
        // The device's clock is taken to tick at the timestamp period, as the drivers feed both from one reference clock; backend-vulkan holds an unreleased hold's end to 50-400 ms.
        hold_.ticks = (uint32_t)std::min<double>(4.0e9, kHoldMs * 1.0e6 / std::max(dev_->timestamp_ns, 1e-3));
    }

    // Whether one hold, released once it is spinning and with no submission after it, ends by the release rather than its bound, which a device that cannot show a running shader the host's write would not; checked once, at setup.
    // The queue's earlier work is waited for and the release comes a tenth of the bound after the hold's submission, so no submission's start shows the hold the write.
    bool hold_seen() {
        if (const Ticket t = tickets_.last()) wait(t);
        hold_queue();
        std::this_thread::sleep_for(std::chrono::milliseconds(kHoldMs / 10));
        hold_release();
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &hold_.timeline;
        wi.pValues = &hold_.last;
        check(dev_->fn.vkWaitSemaphores(dev_->device, &wi, UINT64_MAX), "vkWaitSemaphores");
        return reinterpret_cast<const std::atomic<uint32_t>*>(hold_.word->mapped())[1].load() != hold_.seq;
    }

    // The queue's submission of the open command buffer, after letting a held queue go, waiting on the semaphores a group's exchange left (wait_on).
    // On a submitter, a failure not yet given out is thrown here unless `report` is false, when this submission is given up and the failure kept.
    Ticket flush(bool report = true) {
        const Submission s = prepare({});
        if (!submitter_) return queued(s, s.queue(*dev_));
        const Ticket t = queued(s, VK_SUCCESS);
        submitter_->run({[this, s] {
                             check(s.queue(*dev_), "vkQueueSubmit");
                             made(s.ticket, true);
                         },
                         [this, t] { made(t, false); }},
                        report);
        return t;
    }

    // A submitter's group: every queue call of this backend goes to it from now on, in the order handed over.
    void submit_through(const std::shared_ptr<VulkanSubmitter>& s) {
        if (submitter_ == s) return;
        if (submitter_) throw std::logic_error("vulkan: a device in two tensor groups' submitters");
        sync();
        // Every ticket so far was queued here.
        made_ticket_.store(tickets_.last(), std::memory_order_release);
        good_ticket_.store(tickets_.last(), std::memory_order_release);
        submitter_ = s;
    }
    const std::shared_ptr<VulkanSubmitter>& submitter() const { return submitter_; }
    // Its queue calls its own again, once the submitter has made every call handed over.
    void submit_alone() noexcept {
        if (!submitter_) return;
        submitter_->drain();
        submitter_.reset();
    }
    // After a failed queue call: drops the waits left without a signal and, once earlier work has finished, moves the timeline past the tickets given up.
    void give_up() noexcept {
        waits_.clear();
        if (open_) {
            try { flush(); } catch (const std::exception& e) {
                std::fprintf(stderr, "vulkan: %s; the device is lost\n", e.what());
                std::abort();
            }
        }
        const Ticket last = tickets_.last();
        wait(last);
        if (good_ticket_.load(std::memory_order_acquire) >= last) return;
        VkSemaphoreSignalInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        si.semaphore = timeline_;
        si.value = last;
        if (dev_->fn.vkSignalSemaphore(dev_->device, &si) != VK_SUCCESS) return;
        good_ticket_.store(last, std::memory_order_release);
        tickets_.retired(last);
    }
    // On the submitter's thread: the call carrying `t` has been made, or given up (not `queued`), so no wait for it waits on its timeline value.
    void made(Ticket t, bool queued) noexcept {
        if (queued) {
            good_ticket_.store(t, std::memory_order_release);
        } else {
            std::lock_guard<std::mutex> lk(lost_mu_);
            if (!lost_.empty() && lost_.back().second + 1 == t) lost_.back().second = t;
            else lost_.push_back({t, t});
        }
        made_ticket_.store(t, std::memory_order_release);
    }
    // Whether the call carrying `t` was given up after a failure.
    bool lost(Ticket t) {
        std::lock_guard<std::mutex> lk(lost_mu_);
        for (const auto& r : lost_)
            if (r.first <= t && t <= r.second) return true;
        return false;
    }
    void raise(Ticket t) override {
        if (!submitter_ || !lost(t)) return;
        submitter_->raise();
        throw std::runtime_error("vulkan: the submission of these results failed");
    }

    // The dispatches recorded so far submitted, so a long dispatch recorded next shares its submission with none, whose time the GPU's job timeout bounds.
    void isolate() {
        if (open_ && chunk_) flush();
    }

    // Every hold queued so far ends: the release counter takes the last hold's sequence, which the spinning hold reads with an atomic.
    void hold_release() {
        if (!hold_.pending) return;
        reinterpret_cast<std::atomic<uint32_t>*>(hold_.word->mapped())->store(hold_.seq);
        hold_.pending = false;
    }

    // A submission of the hold's kernel in the next slot; the slot's previous hold has finished once the hold timeline reaches its value.
    // On a submitter the slot's wait, its recording and its call are the submitter's, after the work handed over before it.
    void hold_queue() {
        const uint32_t i = (hold_.slot + 1) % kHolds;
        const Kernel& k = kernel(K_HOLD, 0);
        const uint64_t value = hold_.last + 1;
        const uint32_t seq = hold_.seq + 1;
        auto queue_hold = [this, i, value, seq, pipeline = k.pipeline, layout = k.layout] {
            // The slot's last hold queued, read where the calls are made, so a hold given up after a failure is not waited for.
            const uint64_t before = hold_.queued[i];
            if (before) {
                VkSemaphoreWaitInfo wi{};
                wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
                wi.semaphoreCount = 1;
                wi.pSemaphores = &hold_.timeline;
                wi.pValues = &before;
                check(dev_->fn.vkWaitSemaphores(dev_->device, &wi, UINT64_MAX), "vkWaitSemaphores");
            }
            VkCommandBuffer cmd = hold_.cmds[i];
            check(dev_->fn.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check(dev_->fn.vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
            const VkDescriptorBufferInfo released{hold_.word->handle(), 0, VK_WHOLE_SIZE};
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &released;
            const uint32_t pc[2] = {seq, hold_.ticks};
            dev_->fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            dev_->fn.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &w);
            dev_->fn.vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            dev_->fn.vkCmdDispatch(cmd, 1, 1, 1);
            // What the hold records reaches the host once its timeline value does (hold_seen).
            VkMemoryBarrier mb{};
            mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            dev_->fn.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
            check(dev_->fn.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
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
            hold_.queued[i] = 0;
            check(dev_->fn.vkQueueSubmit(dev_->queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit");
            hold_.queued[i] = value;
        };
        if (submitter_) submitter_->run({queue_hold, [] {}});
        else queue_hold();
        hold_.last = value;
        hold_.slot = i;
        hold_.seq = seq;
        hold_.pending = true;
    }

    void wait(Ticket t) noexcept override { waited(host_.ticket_ms, t); }

    // Only a successful Vulkan wait advances this timeline's known completion; another waiting thread may advance it too.

    // A wait of no time; anything but a timeout is left to the wait that follows.
    bool done(Ticket t) noexcept override {
        if (tickets_.settled(t)) return true;
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timeline_;
        wi.pValues = &t;
        const VkResult r = dev_->fn.vkWaitSemaphores(dev_->device, &wi, 0);
        if (r == VK_SUCCESS) tickets_.retired(t);
        return r != VK_TIMEOUT;
    }

    // noexcept by contract: a device that cannot report its work finished has been lost, and nothing here can act on that.
    void wait_for(Ticket t) noexcept {
        if (tickets_.settled(t)) return;
        // A ticket the submitter has not yet queued is waited for there first; one given up is never signalled, so the wait is for the last ticket queued before it.
        if (submitter_) {
            if (t > made_ticket_.load(std::memory_order_acquire)) submitter_->until([&] { return made_ticket_.load(std::memory_order_acquire) >= t; });
            t = std::min(t, good_ticket_.load(std::memory_order_acquire));
            if (tickets_.settled(t)) return;
        }
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
        tickets_.retired(t);
    }

    // Device time per kernel since the last call, in milliseconds, for a diagnostics backend whose queue timestamps; reading them waits for the queue.
    // Dispatches the last reading timed: every one since the reading before it.
    size_t timed_dispatches() const { return last_timed_; }

    std::vector<std::pair<std::string, double>> kernel_times() {
        std::vector<std::pair<std::string, double>> out;
        if (!dev_->timestamps || queries_.empty()) return out;
        sync();
        std::vector<uint64_t> stamps(query_next_);
        bool read = true;
        for (size_t at = 0; at < query_next_ && read; at += kQueries) {
            const uint32_t count = (uint32_t)std::min<size_t>(kQueries, query_next_ - at);
            read = dev_->fn.vkGetQueryPoolResults(dev_->device, queries_[at / kQueries], 0, count, count * sizeof(uint64_t), stamps.data() + at,
                                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS;
        }
        if (query_next_ && read) {
            for (size_t i = 0; i + 1 < query_kernel_.size() * 2 && i + 1 < stamps.size(); i += 2) {
                const int k = query_kernel_[i / 2];
                kernel_ns_[k] += double(stamps[i + 1] - stamps[i]) * dev_->timestamp_ns;
                ++kernel_calls_[k];
            }
        }
        for (int i = 0; i < 2 * kTimedKernels; ++i)
            if (kernel_calls_[i])
                out.emplace_back(kernel_variant_name(i % kTimedKernels / kVariants, i % kVariants) + (i < kTimedKernels ? "" : " head"), kernel_ns_[i] / 1e6);
        // Since the last call: the next reading starts from an empty pool, which the next dispatch resets.
        last_timed_ = query_kernel_.size();
        std::fill(std::begin(kernel_ns_), std::end(kernel_ns_), 0.0);
        std::fill(std::begin(kernel_calls_), std::end(kernel_calls_), size_t(0));
        query_kernel_.clear();
        query_next_ = 0;
        pools_reset_ = 0;
        return out;
    }

    // Open work and waits a group's exchange left (wait_on) are submitted first, so no semaphore a wait names outlives its use.
    void sync() noexcept override {
        if (open_ || !waits_.empty()) {
            // A submitter's failure is not given out here: the work is given up and the failure kept for the next hand-over.
            try { flush(false); } catch (const std::exception& e) {
                std::fprintf(stderr, "vulkan: %s; the device is lost\n", e.what());
                std::abort();
            }
        }
        wait(tickets_.last());
    }

    void read(const Buffer& src_b, size_t off, void* dst, size_t bytes) override;
    void write(Buffer& dst_b, size_t off, const void* src, size_t bytes) override;
    void write_now(Buffer& dst_b, size_t off, const void* src, size_t bytes);
    void copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off,
              size_t bytes) override;
    VkBufferMemoryBarrier external_ownership_barrier(Buffer& buffer, size_t offset, size_t bytes, bool acquire);
    void external_ownership(const VkBufferMemoryBarrier* barriers, size_t count, bool acquire);
    void external_ownership(Buffer& buffer, size_t offset, size_t bytes, bool acquire);
    size_t decode_columns() const override;

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
        dispatch(K_ADD, {bind(dst), bind(src)}, pc, sizeof(pc), groups(n, 256), 1, 0, {n});
    }

    void sum_rows(Slice residual, const CSlice* sources, size_t members, size_t n, const Slice* kept, bool replace = false);
    static constexpr size_t kSumSources = 4;

    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, RowRuns runs = {}) override {
        if (!n) return;
        const bool quant = n % 32 == 0;
        // Where the integer-dot tile reads dst next, the twin written four values a lane (shaders/silu_mul.comp).
        const size_t rows = runs.n ? runs.runs[runs.n - 1].end : 0;
        const bool tile = quant && rows && n % rows == 0 && tile_reads(n / rows, rows, runs);
        const uint32_t major = tile ? x8_major(n / rows, rows) : 0;
        const uint32_t pc[3] = {u32(n), tile ? 2u : quant ? 1u : 0u, major};
        const size_t slot = quant ? take_twin(n) : 0;
        dispatch(K_SILU_MUL, {bind(dst), bind(gate), bind(up), quant ? twin_binding(slot) : bind(dst)}, pc, sizeof(pc),
                 groups(tile ? n / 4 : n, 256), 1, quant ? twin_variant() : 0, {n, 0, 0, n});
        if (quant) keep_twin(slot, bind(dst), n, major);
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
        const uint32_t major = tile ? x8_major(width, rows) : 0;
        struct { uint32_t n, quant, width, dim, gate_stride, gate_head_stride, major; }
            pc{u32(n), tile ? 2u : quant ? 1u : 0u, u32(width), u32(dim), u32(gate_stride), u32(gate_head_stride), major};
        const size_t slot = quant ? take_twin(n) : 0;
        dispatch(K_SIGMOID_MUL, {bind(dst), bind(x), bind(gate), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), groups(tile ? n / 4 : n, 256), 1, quant ? twin_variant() : 0, {n, 0, 0, n});
        if (quant) keep_twin(slot, bind(dst), n, major);
    }

    // The vision encoder's ops (docs/VISION.md, section 3); the elementwise ones stride over their elements from at most a full x dimension of workgroups.
    uint32_t striding_groups(size_t n) const { return u32(std::min<size_t>((n + 255) / 256, dev_->props.limits.maxComputeWorkGroupCount[0])); }

    void layer_norm_rows(Slice dst, CSlice src, CSlice w, CSlice b, size_t rows, size_t n, size_t stride, float eps) override {
        if (!n || stride < n || !std::isfinite(eps) || eps < 0.0f) throw std::runtime_error("vulkan: invalid layer norm dimensions");
        if (!rows) return;
        const size_t last = size_add(size_mul(rows - 1, stride), n);
        if (floats_from(dst) < last || floats_from(src) < last || floats_from(w) < n || floats_from(b) < n)
            throw std::runtime_error("vulkan: layer norm operand outside its allocation");
        struct { uint32_t n, stride; float eps; } pc{u32(n), u32(stride), eps};
        dispatch(K_LAYER_NORM_ROWS, {bind(dst), bind(src), bind(w), bind(b)}, &pc, sizeof(pc), groups(rows, 1));
    }

    void gelu(Slice dst, CSlice src, size_t n, GeluForm form) override {
        if (!n) return;
        if (floats_from(dst) < n || floats_from(src) < n) throw std::runtime_error("vulkan: gelu operand outside its allocation");
        const uint32_t pc[2] = {u32(n), form == GeluForm::erf ? 1u : 0u};
        dispatch(K_GELU, {bind(dst), bind(src)}, pc, sizeof(pc), striding_groups(n));
    }

    void add_bias(Slice dst, CSlice bias, size_t rows, size_t width) override {
        if (!width) throw std::runtime_error("vulkan: invalid bias width");
        const size_t n = size_mul(rows, width);
        if (!n) return;
        if (floats_from(dst) < n || floats_from(bias) < width) throw std::runtime_error("vulkan: bias operand outside its allocation");
        const uint32_t pc[2] = {u32(n), u32(width)};
        dispatch(K_ADD_BIAS, {bind(dst), bind(bias)}, pc, sizeof(pc), striding_groups(n));
    }

    void mix_rows(Slice dst, CSlice table, size_t width, size_t table_rows, const uint32_t* idx, const float* weight, size_t rows, size_t taps) override {
        if (!width || !taps) throw std::runtime_error("vulkan: invalid row mix dimensions");
        const size_t n = size_mul(rows, width), picks = size_mul(rows, taps);
        if (!n) return;
        if (!idx || !weight) throw std::runtime_error("vulkan: a row mix without its taps");
        for (size_t i = 0; i < picks; ++i)
            if (idx[i] >= table_rows) throw std::runtime_error("vulkan: a row mix names a row past its table");
        if (floats_from(dst) < n || floats_from(table) < size_mul(table_rows, width)) throw std::runtime_error("vulkan: row mix operand outside its allocation");
        const uint32_t pc[3] = {u32(n), u32(width), u32(taps)};
        dispatch(K_MIX_ROWS, {bind(dst), bind(table), args(idx, picks * sizeof(uint32_t)), args(weight, picks * sizeof(float))}, pc, sizeof(pc),
                 striding_groups(n));
    }

    // One workgroup per (row, head); like rms_norm_rows it writes the copy the matmul reading dst next takes (shaders/gated_rms_norm.comp).
    void gated_rms_norm(Slice dst, CSlice x, CSlice z, CSlice w, size_t rows, size_t heads, size_t dim, float eps,
                        RowRuns runs = {}, NormGate gate = NormGate::silu) override {
        if (!heads || !dim) throw std::runtime_error("vulkan: invalid gated norm dimensions");
        const size_t width = size_mul(heads, dim), n = size_mul(rows, width);
        if (floats_from(dst) < n || floats_from(x) < n || floats_from(z) < n || floats_from(w) < dim)
            throw std::runtime_error("vulkan: gated norm operand outside its allocation");
        if (!rows) return;
        const bool quant = dim % 32 == 0;
        const bool tile = quant && tile_reads(width, rows, runs);
        const uint32_t major = tile ? x8_major(width, rows) : 0;
        struct { uint32_t heads, dim; float eps; uint32_t quant, n, major, gate; }
            pc{u32(heads), u32(dim), eps, tile ? 2u : quant ? 1u : 0u, u32(n), major, gate == NormGate::sigmoid ? 1u : 0u};
        const size_t slot = quant ? take_twin(n) : 0;
        dispatch(K_GATED_RMS_NORM, {bind(dst), bind(x), bind(z), bind(w), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), u32(rows * heads), 1, quant ? twin_variant() : 0, {n, 0, 0, 0, n});
        if (quant) keep_twin(slot, bind(dst), n, major);
    }

    void group_rms_norm(Slice dst, CSlice src, CSlice w, size_t rows, size_t group_count, size_t dim, float eps,
                        RowRuns runs = {}) override {
        if (!group_count || !dim || !std::isfinite(eps) || eps <= 0.0f)
            throw std::runtime_error("vulkan: invalid grouped norm dimensions or epsilon");
        if (!rows) return;
        validate_runs(rows, runs);
        const size_t width = size_mul(group_count, dim), n = size_mul(rows, width);
        if (floats_from(dst) < n || floats_from(src) < n || floats_from(w) < width)
            throw std::runtime_error("vulkan: grouped norm operand outside its allocation");
        if (overlaps(CSlice(dst), n, src, n) && !(dst.buffer == src.buffer && dst.offset == src.offset))
            throw std::runtime_error("vulkan: grouped norm output must be disjoint or exactly in place");
        if (overlaps(CSlice(dst), n, w, width))
            throw std::runtime_error("vulkan: grouped norm output overlaps its weights");
        // The activation twin is row-major. A group can produce it independently only at a 32-value boundary.
        const bool quant = dim % 32 == 0;
        const bool tile = quant && tile_reads(width, rows, runs);
        const uint32_t major = tile ? x8_major(width, rows) : 0;
        // Slices of a group each repeat its reduction and write a share of its chunks (shaders/group_rms_norm.comp), up to what fills the device.
        // One in place, where a slice's writes would meet another's reads.
        const size_t fill = (4 * dev_->caps.compute_units + rows * group_count - 1) / (rows * group_count);
        const bool in_place = dst.buffer == src.buffer && dst.offset == src.offset;
        const size_t slices = in_place ? 1 : std::max<size_t>(1, std::min((dim + 1023) / 1024, fill));
        struct { uint32_t rows, groups, dim, quant, n, major; float eps; }
            pc{u32(rows), u32(group_count), u32(dim), quant ? 1u : 0u, u32(n), major, eps};
        const size_t slot = quant ? take_twin(n) : 0;
        dispatch(K_GROUP_RMS_NORM, {bind(dst), bind(src), bind(w), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), groups(size_mul(rows, group_count), 1), u32(slices), quant ? twin_variant() : 0, {n, 0, 0, n});
        if (quant) keep_twin(slot, bind(dst), n, major);
    }

    void silu_scaled(Slice dst, CSlice src, size_t rows, size_t width, float scale, RowRuns runs = {}) override {
        if (!width || !std::isfinite(scale)) throw std::runtime_error("vulkan: invalid scaled SiLU dimensions or scale");
        if (!rows) return;
        validate_runs(rows, runs);
        const size_t n = size_mul(rows, width);
        if (floats_from(dst) < n || floats_from(src) < n)
            throw std::runtime_error("vulkan: scaled SiLU operand outside its allocation");
        if (overlaps(CSlice(dst), n, src, n) && !(dst.buffer == src.buffer && dst.offset == src.offset))
            throw std::runtime_error("vulkan: scaled SiLU output must be disjoint or exactly in place");
        const bool quant = width % 32 == 0;
        const bool tile = quant && tile_reads(width, rows, runs);
        const uint32_t major = tile ? x8_major(width, rows) : 0;
        struct { uint32_t n, quant, major; float scale; } pc{u32(n), tile ? 2u : quant ? 1u : 0u, major, scale};
        const size_t slot = quant ? take_twin(n) : 0;
        dispatch(K_SILU_SCALED, {bind(dst), bind(src), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), groups(tile ? n / 4 : n, 256), 1, quant ? twin_variant() : 0, {n, 0, n});
        if (quant) keep_twin(slot, bind(dst), n, major);
    }

    void mix_streams(Slice dst, CSlice src, CSlice gate, size_t rows, size_t streams, size_t dim,
                     RowRuns runs = {}) override {
        if (!streams || !dim) throw std::runtime_error("vulkan: invalid stream mix dimensions");
        if (!rows) return;
        validate_runs(rows, runs);
        const size_t width = size_mul(streams, dim), n = size_mul(rows, width), out_n = size_mul(rows, dim);
        if (floats_from(dst) < out_n || floats_from(src) < n || floats_from(gate) < n)
            throw std::runtime_error("vulkan: stream mix operand outside its allocation");
        (void)u32(n); // The shader indexes the full [row, stream, dim] source in uint arithmetic.
        if (overlaps(CSlice(dst), out_n, src, n) || overlaps(CSlice(dst), out_n, gate, n))
            throw std::runtime_error("vulkan: stream mix output must be disjoint from its inputs");
        const bool quant = dim % 32 == 0;
        const bool tile = quant && tile_reads(dim, rows, runs);
        const uint32_t major = tile ? x8_major(dim, rows) : 0;
        struct { uint32_t rows, streams, dim, quant, n, major; }
            pc{u32(rows), u32(streams), u32(dim), tile ? 2u : quant ? 1u : 0u, u32(out_n), major};
        const size_t slot = quant ? take_twin(out_n) : 0;
        dispatch(K_MIX_STREAMS, {bind(dst), bind(src), bind(gate), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), groups(tile ? out_n / 4 : out_n, 256), 1, quant ? twin_variant() : 0, {out_n, 0, 0, out_n});
        if (quant) keep_twin(slot, bind(dst), out_n, major);
    }

    void inject_streams(Slice dst, CSlice block, CSlice gate, size_t rows, size_t streams, size_t dim) override {
        if (!streams || !dim) throw std::runtime_error("vulkan: invalid stream injection dimensions");
        if (!rows) return;
        const size_t width = size_mul(streams, dim), n = size_mul(rows, width);
        if (floats_from(dst) < n || floats_from(block) < size_mul(rows, dim) || floats_from(gate) < size_mul(rows, streams))
            throw std::runtime_error("vulkan: stream injection operand outside its allocation");
        if (overlaps(CSlice(dst), n, block, size_mul(rows, dim)) || overlaps(CSlice(dst), n, gate, size_mul(rows, streams)) ||
            overlaps(block, size_mul(rows, dim), gate, size_mul(rows, streams)))
            throw std::runtime_error("vulkan: stream injection operands must be disjoint");
        const struct { uint32_t rows, streams, dim, n; } pc{u32(rows), u32(streams), u32(dim), u32(n)};
        dispatch(K_INJECT_STREAMS, {bind(dst), bind(block), bind(gate)}, &pc, sizeof(pc), groups(n, 256), 1, 0, {n});
    }

    void signed_sqrt_gate(Slice dst, CSlice key, CSlice query, CSlice value, size_t rows, size_t groups_count, size_t dim,
                          RowRuns runs = {}) override {
        if (!groups_count || !dim) throw std::runtime_error("vulkan: invalid signed square-root gate dimensions");
        if (!rows) return;
        validate_runs(rows, runs);
        const size_t width = size_mul(groups_count, dim), n = size_mul(rows, width);
        if (floats_from(dst) < n || floats_from(key) < n || floats_from(query) < n || floats_from(value) < size_mul(rows, dim))
            throw std::runtime_error("vulkan: signed square-root gate operand outside its allocation");
        if (overlaps(CSlice(dst), n, key, n) || overlaps(CSlice(dst), n, query, n) ||
            overlaps(CSlice(dst), n, value, size_mul(rows, dim)))
            throw std::runtime_error("vulkan: signed square-root gate output must be disjoint from its inputs");
        const bool quant = dim % 32 == 0;
        const bool tile = quant && tile_reads(width, rows, runs);
        const uint32_t major = tile ? x8_major(width, rows) : 0;
        struct { uint32_t rows, groups, dim, quant, n, major; float scale; }
            pc{u32(rows), u32(groups_count), u32(dim), tile ? 2u : quant ? 1u : 0u, u32(n), major,
               1.0f / std::sqrt(float(dim))};
        const size_t slot = quant ? take_twin(n) : 0;
        dispatch(K_SIGNED_SQRT_GATE, {bind(dst), bind(key), bind(query), bind(value), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), groups(size_mul(rows, groups_count), 1), 1, quant ? twin_variant() : 0, {n, 0, 0, 0, n});
        if (quant) keep_twin(slot, bind(dst), n, major);
    }

    // The storage of a call's state views, which must all be one, and the table the state kernels read: the view count, then per view its first batch row, rows, history length, source slot and destination slot; the conv appends its chunks.
    // check_state_views has checked the slots against the storage; the slots' floats must be addressable in 32 bits.
    struct StateTable {
        StateStorage* storage = nullptr;
        std::vector<uint32_t> words;
    };
    static StateTable state_table(const StateView* views, size_t n_views, size_t layer) {
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
        // This layer may carry an additional recurrent tail after the shared GDN prefix.
        (void)u32(size_mul(t.storage->slots(), t.storage->slot_floats(layer)));
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
        StateTable t = state_table(views, n_views, layer);
        const StateShape& sh = t.storage->shape();
        const size_t C = sh.channels();
        if (floats_from(out) < size_mul(rows, C) || floats_from(x) < size_mul(rows, C) || floats_from(w) < size_mul(C, kConvTaps))
            throw std::runtime_error("vulkan: conv operand outside its allocation");
        size_t chunks = 0;
        for (size_t i = 0; i < n_views; ++i)
            for (size_t b0 = 0; b0 < views[i].nq; b0 += kConvChunk, ++chunks) t.words.insert(t.words.end(), {u32(i), u32(b0)});
        if (chunks > dev_->props.limits.maxComputeWorkGroupCount[1]) throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        const struct { uint32_t C, slot, carried, chunk; }
            pc{u32(C), u32(t.storage->slot_floats(layer)), u32(sh.v_heads * sh.matrix_floats()), u32(kConvChunk)};
        dispatch(K_CAUSAL_CONV_SILU,
                 {bind(out), bind(x), bind(w), bind(CSlice{&t.storage->layer(layer), 0}), args(t.words.data(), t.words.size() * sizeof(uint32_t))},
                 &pc, sizeof(pc), groups(C, 256), u32(chunks));
    }

    // A dilated causal conv reads recurrent raw-row tails only in its first chunk; all later chunks begin past the full history window.
    void dilated_conv_silu_add(Slice out, CSlice x, CSlice residual, CSlice w, size_t channels, size_t taps, size_t dilation,
                               size_t layer, const StateView* views, size_t n_views) override {
        if (!channels || !taps || !dilation) throw std::runtime_error("vulkan: invalid dilated conv dimensions");
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        StateTable t = state_table(views, n_views, layer);
        const size_t history = size_mul(taps - 1, dilation);
        const size_t row_floats = size_mul(rows, channels), weight_floats = size_mul(channels, taps);
        const size_t tail_floats = size_mul(history, channels);
        const size_t prefix_floats = t.storage->shape().slot_floats();
        if (t.storage->slot_floats(layer) < size_add(prefix_floats, tail_floats))
            throw std::runtime_error("vulkan: recurrent slot lacks dilated conv history");
        if (floats_from(out) < row_floats || floats_from(x) < row_floats || floats_from(residual) < row_floats ||
            floats_from(w) < weight_floats)
            throw std::runtime_error("vulkan: dilated conv operand outside its allocation");

        // Shader indexing and each slot's entire address range use u32 arithmetic.
        (void)u32(rows);
        (void)u32(channels);
        (void)u32(taps);
        (void)u32(dilation);
        (void)u32(history);
        (void)u32(row_floats);
        (void)u32(weight_floats);
        (void)u32(tail_floats);
        (void)u32(prefix_floats);
        (void)u32(t.storage->slot_floats(layer));
        const CSlice outc(out);
        const bool exact_residual = out.buffer == residual.buffer && out.offset == residual.offset;
        if ((overlaps(outc, row_floats, residual, row_floats) && !exact_residual) ||
            overlaps(outc, row_floats, x, row_floats) || overlaps(outc, row_floats, w, weight_floats))
            throw std::runtime_error("vulkan: dilated conv output may only alias the residual exactly");
        const CSlice state{&t.storage->layer(layer), 0};
        const size_t state_floats = t.storage->layer(layer).size() / sizeof(float);
        if (overlaps(outc, row_floats, state, state_floats) || overlaps(x, row_floats, state, state_floats) ||
            overlaps(residual, row_floats, state, state_floats) || overlaps(w, weight_floats, state, state_floats))
            throw std::runtime_error("vulkan: dilated conv operand overlaps recurrent state storage");

        const size_t chunk = std::max(kConvChunk, history);
        (void)u32(chunk);
        size_t chunks = 0;
        for (size_t i = 0; i < n_views; ++i) {
            for (size_t b0 = 0; b0 < views[i].nq; b0 += chunk) {
                t.words.insert(t.words.end(), {u32(i), u32(b0)});
                ++chunks;
            }
        }
        (void)u32(chunks);
        if (chunks > dev_->props.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        (void)u32(t.words.size());
        const struct { uint32_t C, taps, dilation, history, slot, prefix, chunk; }
            pc{u32(channels), u32(taps), u32(dilation), u32(history), u32(t.storage->slot_floats(layer)),
               u32(prefix_floats), u32(chunk)};
        dispatch(K_DILATED_CONV_SILU_ADD,
                 {bind(out), bind(x), bind(residual), bind(w), bind(state), args(t.words.data(), size_mul(t.words.size(), sizeof(uint32_t)))},
                 &pc, sizeof(pc), groups(channels, 256), u32(chunks));
    }

    // One dispatch, a workgroup per (view, V head, 32 V columns), which norms q and k and makes the gates for each block of its view's tokens before running them (shaders/delta_rule.comp).
    void gated_delta_rule(Slice out, CSlice qkv, CSlice alpha, CSlice b, CSlice a, CSlice dt_bias,
                          size_t layer, const StateView* views, size_t n_views) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        const StateTable t = state_table(views, n_views, layer);
        const StateShape& sh = t.storage->shape();
        const size_t Hk = sh.k_heads, Hv = sh.v_heads, Dk = sh.k_dim, Dv = sh.v_dim, C = sh.channels();
        // The recurrence holds a column's rows in registers, 16 in each of eight lanes, and stages 128 of each q and k row.
        if (Dk > 128) throw std::runtime_error("vulkan: a linear-attention K head wider than 128");
        if (floats_from(out) < size_mul(rows, Hv * Dv) || floats_from(qkv) < size_mul(rows, C) || floats_from(alpha) < size_mul(rows, Hv) ||
            floats_from(b) < size_mul(rows, Hv) || floats_from(a) < Hv || floats_from(dt_bias) < Hv)
            throw std::runtime_error("vulkan: delta rule operand outside its allocation");
        const size_t blocks = (Dv + 31) / 32;
        const struct { uint32_t C, k_heads, v_heads, k_dim, v_dim, slot, blocks; float scale, eps; }
            pc{u32(C), u32(Hk), u32(Hv), u32(Dk), u32(Dv), u32(t.storage->slot_floats(layer)), u32(blocks), (float)(1.0 / std::sqrt((double)Dk)), kL2NormEps};
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
        const uint32_t major = tile ? x8_major(n, rows) : 0;
        struct { uint32_t rows, n, stride; float eps; uint32_t quant, chunks, major; }
            pc{u32(rows), u32(n), u32(stride), eps, tile ? 2u : quant ? 1u : 0u, u32(chunks), major};
        const size_t slot = quant ? take_twin(rows * n) : 0, reach = (rows - 1) * stride + n;
        dispatch(K_RMS_NORM_ROWS, {bind(dst), bind(src), bind(w), quant ? twin_binding(slot) : bind(dst)},
                 &pc, sizeof(pc), u32(rows * chunks), 1, quant ? twin_variant() : 0, {reach, 0, 0, reach});
        if (quant) keep_twin(slot, bind(dst), rows * n, major);
    }

    // One workgroup per (row, head), reading the heads at their strides and writing them contiguously (shaders/norm_rope_partial.comp).
    void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride,
                           size_t heads, size_t head_dim, size_t rope_dim, CSlice w, float eps,
                           CSlice cos, CSlice sin, const uint32_t* pos, RopeStreams streams = {}) override {
        if (!heads || !head_dim || !rope_dim || rope_dim % 2 || rope_dim > head_dim)
            throw std::runtime_error("vulkan: invalid partial rope dimensions");
        const size_t half = rope_dim / 2;
        // The plain path is one stream with a norm; several streams, or no norm, take the streams build (shaders/norm_rope_partial.comp).
        const bool gathered = streams.count != 1 || !w.buffer;
        if (!streams.count || (streams.count > 1 && !streams.of)) throw std::runtime_error("vulkan: invalid rope streams");
        if (rows && !pos) throw std::runtime_error("vulkan: rope without positions");
        std::vector<uint32_t> of;
        if (gathered) {
            of.resize(half, 0);
            for (size_t j = 0; streams.count > 1 && j < half; ++j) {
                if (streams.of[j] >= streams.count) throw std::runtime_error("vulkan: a rope frequency names a stream past its row's streams");
                of[j] = streams.of[j];
            }
        }
        if (!rows) return;
        const size_t width = size_mul(heads, head_dim);
        const size_t last = size_add(size_add(size_mul(rows - 1, src_stride), size_mul(heads - 1, src_head_stride)), head_dim);
        if (floats_from(dst) < size_mul(rows, width) || floats_from(src) < last || (w.buffer && floats_from(w) < head_dim))
            throw std::runtime_error("vulkan: partial rope operand outside its allocation");
        if (dst.buffer == src.buffer && dst.offset == src.offset && (src_stride != width || src_head_stride != head_dim))
            throw std::runtime_error("vulkan: partial rope in place over heads that are not contiguous");
        const size_t table = std::min(floats_from(cos), floats_from(sin)) / half, npos = size_mul(rows, streams.count);
        for (size_t r = 0; r < npos; ++r)
            if (pos[r] >= table) throw std::runtime_error("vulkan: position outside the RoPE table");
        if (gathered) {
            struct { uint32_t src_stride, src_head_stride, heads, head_dim, half; float eps; uint32_t count, norm; }
                pc{u32(src_stride), u32(src_head_stride), u32(heads), u32(head_dim), u32(half), eps, u32(streams.count), w.buffer ? 1u : 0u};
            dispatch(K_NORM_ROPE_PARTIAL_STREAMS,
                     {bind(dst), bind(src), w.buffer ? bind(w) : bind(src), bind(cos), bind(sin), args(pos, npos * sizeof(uint32_t)), args(of.data(), half * sizeof(uint32_t))},
                     &pc, sizeof(pc), u32(size_mul(rows, heads)));
            return;
        }
        struct { uint32_t src_stride, src_head_stride, heads, head_dim, half; float eps; }
            pc{u32(src_stride), u32(src_head_stride), u32(heads), u32(head_dim), u32(half), eps};
        dispatch(K_NORM_ROPE_PARTIAL,
                 {bind(dst), bind(src), bind(w), bind(cos), bind(sin), args(pos, rows * sizeof(uint32_t))},
                 &pc, sizeof(pc), u32(rows * heads));
    }

    void norm_rope_kv(Slice q, size_t q_stride, size_t n_head, CSlice q_w,
                      Slice k, CSlice v, size_t kv_stride, size_t n_head_kv, CSlice k_w,
                      const RopeArgs& rope, size_t rows, size_t layer,
                      const KVView* views, size_t n_views) override;

    void embed(Slice dst, WeightView weight, size_t nin, size_t nrows,
               const uint32_t* ids, size_t count) override {
        const uint32_t type = weight.type;
        const CSlice table = weight.data;
        if (count && !ids) throw std::runtime_error("vulkan: embed without ids");
        if (!count || !nin) return;
        check_matrix(type, table, nin, nrows, "embedding");
        check_weight_scales(weight);
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= nrows) throw std::runtime_error("vulkan: embedding row out of range");
        gather_views(weight_kernels(type)->embed, dst, weight, nin, nrows, args(ids, count * sizeof(uint32_t)), count, false);
    }

    // Ids past the table write zero rows; both embedding calls use the format's decoder (shaders/embed.comp).
    void embed_ids(Slice dst, WeightView weight, size_t nin, size_t nrows, CSlice ids, size_t count) override {
        const uint32_t type = weight.type;
        const CSlice table = weight.data;
        if (!count || !nin) return;
        check_matrix(type, table, nin, nrows, "embedding");
        check_weight_scales(weight);
        if (floats_from(ids) < count || floats_from(dst) < size_mul(count, nin))
            throw std::runtime_error("vulkan: embed_ids operand outside its allocation");
        gather_views(weight_kernels(type)->embed_ids, dst, weight, nin, nrows, bind(ids), count, true);
    }
    // One dispatch a view of the table (table_views), each workgroup writing its row where that view holds it; with device ids the last view also writes the zero rows of ids past the table.
    void gather_views(KernelId kernel, Slice dst, WeightView weight, size_t nin, size_t nrows, VkDescriptorBufferInfo ids, size_t count, bool device_ids) {
        const uint32_t type = weight.type;
        const CSlice table = weight.data;
        const VkBuffer buffer = as_vulkan(*table.buffer).handle();
        for (const TableView& v : table_views(bind(table).offset, nrows, quant::row_bytes(type, nin), dev_->props.limits.maxStorageBufferRange,
                                              dev_->props.limits.minStorageBufferOffsetAlignment)) {
            const VkDescriptorBufferInfo view{buffer, v.offset, v.range};
            const uint32_t pc[6] = {u32(nin), type, v.skew, v.first, v.end, u32(nrows)};
            // Dictionary formats use the spare F32 table view for immutable format data.
            dispatch(kernel, {bind(dst), weight_floats(weight, lookup_grid(type, view)), view, ids}, pc, device_ids ? sizeof(pc) : sizeof(pc) - sizeof(uint32_t), u32(count));
        }
    }

    // A row past one slice is reduced by a workgroup a slice into partials, then a workgroup a row over them (shaders/argmax_rows.comp); a shorter row, or work recorded without barriers, takes one workgroup a row.
    // Without prior ids the logits are bound in their place and not read.
    void argmax_rows(Slice ids, CSlice logits, size_t rows, size_t n, CSlice after = {}) override {
        if (!rows) return;
        if (!n || n >= UINT32_MAX) throw std::runtime_error("vulkan: argmax over no ids or more than 32 bits name");
        if (floats_from(ids) < rows || floats_from(logits) < size_mul(rows, n) || (after.buffer && floats_from(after) < rows))
            throw std::runtime_error("vulkan: argmax_rows operand outside its allocation");
        const VkDescriptorBufferInfo prior = after.buffer ? bind(after) : bind(logits);
        const size_t slices = (n + kVulkanArgmaxSlice - 1) / kVulkanArgmaxSlice;
        uint32_t pc[5] = {u32(n), after.buffer ? 1u : 0u, 0u, u32(slices), u32(kVulkanArgmaxSlice)};
        if (slices == 1 || unordered_) {
            dispatch(K_ARGMAX_ROWS, {bind(ids), bind(logits), prior, bind(ids)}, pc, sizeof(pc), u32(rows));
            return;
        }
        const size_t bytes = size_mul(size_mul(rows, slices), 3 * sizeof(uint32_t));
        if (!argmax_parts_ || argmax_parts_->size() < bytes) grow(argmax_parts_, bytes);
        const VkDescriptorBufferInfo parts{argmax_parts_->handle(), 0, VK_WHOLE_SIZE};
        pc[2] = 1u;
        dispatch(K_ARGMAX_ROWS, {bind(ids), bind(logits), prior, parts}, pc, sizeof(pc), u32(slices), u32(rows));
        pc[2] = 2u;
        dispatch(K_ARGMAX_ROWS, {bind(ids), bind(logits), prior, parts}, pc, sizeof(pc), u32(rows));
    }
    void matmul(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout,
                size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) override;
    void matmul_logits(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {},
                       Dtype dtype = Dtype::f16) override;
    void matmul_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout,
                    size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) override;
    void matmul_group(std::initializer_list<Projection> projections, CSlice X,
                      size_t nin, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) override;
    size_t tile_from(const Projection* projections, size_t count, size_t nin) const;
    static size_t split_tiles_of(size_t extent);
    void matmul_runs(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nbatch,
                     bool accumulate, RowRuns runs, Dtype dtype);
    void check_group(const Projection* projections, size_t count, CSlice X, size_t nin, size_t nbatch) const;
    void matmul_group_impl(const Projection* projections, size_t count, CSlice X,
                           size_t nin, size_t nbatch, bool accumulate, Dtype dtype, int kernel_choice = -1, size_t split_tiles = 0);
    size_t column_builds(KernelId id, size_t nin, size_t rows, uint32_t cluster, int (&out)[kVariants]) const;
    int one_column_build(size_t nin, size_t rows, bool x8) const;
    bool paired_build(KernelId id, size_t nin) const;
    int filling_build(KernelId id, int variant, size_t ncols, size_t rows) const;
    template <typename Fn>
    void for_each_column_chunk(KernelId id, size_t nbatch, size_t nin, size_t rows, uint32_t cluster, const Fn& each) const;

    // Choose a row module, layout and lanes per row from the weight format and input width.
    // The Q8_0 decode module instead takes its rows from the build (sg_rows).
    struct RowPlan {
        KernelId kernel;
        uint32_t type, wide, cluster;
    };
    RowPlan row_plan(uint32_t type, size_t nin, Dtype dtype) const;
    uint32_t sg_rows(const RowPlan& plan, int variant) const;
    VkDescriptorBufferInfo row_twin(CSlice X, const RowPlan& plan, size_t n, Dtype dtype);
    VkDescriptorBufferInfo twin(CSlice X, size_t n, bool x8, size_t tile_nin, uint32_t* kept_major = nullptr);
    void row_dispatch(const RowPlan& plan, const std::vector<const Projection*>& live, CSlice X, VkDescriptorBufferInfo xqi,
                      size_t nin, size_t nbatch, size_t col0, size_t ncols, bool accumulate, int variant,
                      uint32_t per = 0, size_t entries = 1, VkDescriptorBufferInfo ids = {},
                      uint32_t order0 = 0, VkDescriptorBufferInfo tab = {}, size_t routed = 0);
    void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize, Slice ids, Slice weights) override;
    template <typename Fn>
    void expert_runs(uint32_t type, size_t nrows, RowRuns runs, const Fn& each);
    static CSlice at_float(CSlice s, size_t floats);
    static Slice at_float(Slice s, size_t floats);
    void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nrows,
                        const Routing& routing, RowRuns runs, Dtype dtype = Dtype::f16) override;
    void matmul_experts_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows,
                            const Routing& routing, RowRuns runs, Dtype dtype = Dtype::f16) override;
    void routed_call(const std::vector<const Projection*>& live, CSlice X, size_t nin, size_t xcols, size_t nrows, size_t k,
                     CSlice ids, size_t n_expert, bool tile, Dtype dtype, size_t per = 0);
    void moe_step(const MoeStep& m, CSlice h, Slice y, size_t nin, size_t rows, RowRuns runs = {}, Dtype dtype = Dtype::f16) override;
    const MoeBuild* moe_build(const MoeStep& m, size_t nin, size_t rows, RowRuns runs, Dtype dtype) const;
    // The decode rows a fused routed step takes at most, the columns the decode-column checks reach.
    static constexpr size_t kMoeStepRows = 64;
    void check_matrix(uint32_t type, CSlice data, size_t nin, size_t rows, const char* what = "matrix") const;
    size_t take_twin(size_t n, VkDescriptorBufferInfo x = {});
    VkDescriptorBufferInfo twin_binding(size_t slot) const;
    void keep_twin(size_t slot, VkDescriptorBufferInfo x, size_t n, uint32_t major);
    VkDescriptorBufferInfo tile_twin(CSlice X, size_t n, size_t nin, bool x8, uint32_t& xpad);
    uint32_t x8_major(size_t nin, size_t rows) const;
    static uint32_t x8_columns(uint32_t cols);
    static constexpr size_t kX8MajorRows = 32;

    static size_t x8_base_bytes(size_t n);
    int twin_variant() const;
    bool integer_dot_tile(uint32_t type) const;

    // An integer-dot tile call over up to three projections of one type: the height their rows and `column_groups` call for, its module, and each projection's first workgroup and rows.
    // Unused slots repeat the first projection, which no workgroup reaches.
    struct QTile {
        uint32_t height = 0;
        int variant = 0;   // 1 the small height, 2 the narrow build of 16 columns
        KernelId kernel = K_MATMUL_TILE_Q;
        size_t rows = 0, gx = 0;
        uint32_t start[3] = {0, 0, 0}, nout[3] = {0, 0, 0};
        const Projection* p[3] = {nullptr, nullptr, nullptr};
    };
    // A `narrow` call takes the build of 16 columns at the tall height, or the short one for a type with one integer module.
    QTile qtile(const std::vector<const Projection*>& ps, size_t column_groups, size_t nin, bool i8 = false, bool narrow = false) const;
    void dispatch_qtile(const QTile& tile, std::array<VkDescriptorBufferInfo, 3> output,
                        VkDescriptorBufferInfo activation, VkDescriptorBufferInfo grouping,
                        const uint32_t (&pc)[16], uint32_t groups_y, std::array<size_t, 3> reach = {});
    void copy_mxfp4(CSlice data, size_t rows, size_t nin, size_t at);
    VkDescriptorBufferInfo padded_f32(CSlice data, uint32_t type, size_t rows, size_t nin);
    void drop_padded(VulkanBuffer& b);
    size_t split_blocks(size_t workgroups, size_t nblk, uint32_t per_cu = 0) const;
    bool tile_reads(size_t nin, size_t nbatch, RowRuns runs) const;
    void grow(std::shared_ptr<VulkanBuffer>& buffer, size_t bytes);
    KVLayout kv_layout() const override;
    std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim,
                                        size_t max_tokens, KVType k_type = KVType::f32,
                                        KVType v_type = KVType::f32,
                                        const std::vector<size_t>& auxiliary_widths = {}, const std::vector<size_t>& auxiliary_pools = {}) override;
    void kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k,
                  CSlice v, CSlice auxiliary = {}) override;
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out,
                   int n_head, int n_head_kv, int head_dim) override;
    void sparse_attention(CSlice query, size_t layer, const KVView* views, size_t n_views,
                          CSlice ids, size_t id_stride, Slice out, int n_head, int n_head_kv, int head_dim) override;
    // An auxiliary plane's blocks as the indexer's kernels address them (views.glsl, pooled_row): their tokens, their floats and the floats of their tokens' rows before the pooled rows.
    struct PlaneShape {
        uint32_t bt, block_floats, rows_floats;
    };
    void select_kv(CSlice query, size_t layer, const KVView* views, size_t n_views, Slice ids,
                   const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin, const GroupPositions& positions = {}) override;
    void pool_kv(size_t layer, const KVView* views, size_t n_views, const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin,
                 const GroupPositions& positions = {}) override;

    // A view with the batch row its rows start at.
    struct Placed {
        const KVView* view;
        size_t row0;
    };
    static std::vector<Placed> place_views(const KVView* views, size_t n_views);
    // The complete groups a view's rows complete, pooled into their rows of the auxiliary plane (shaders/kv_index_pool.comp); the earlier groups were pooled by the call whose rows completed them.
    // With group positions, the view's start at `at` of them: each group turns by the positions its streams name, else at its first token's index.
    void pool_view(VkDescriptorBufferInfo aux, PlaneShape plane, VkDescriptorBufferInfo key_norm, VkDescriptorBufferInfo cos, VkDescriptorBufferInfo sin,
                   const std::vector<uint32_t>& table, uint32_t view, const KVView& v, const IndexArgs& index, const GroupPositions& positions = {}, size_t at = 0);
    // The scores of `rows` rows from `row` on against every complete group they see, a row of `groups` apart, kScoreTile groups a workgroup (shaders/kv_index_scores_tile.comp).
    void score_rows(VkDescriptorBufferInfo query, VkDescriptorBufferInfo pooled, PlaneShape plane, VkDescriptorBufferInfo scores, const std::vector<uint32_t>& table,
                    uint32_t view, size_t row, size_t rows, size_t groups, uint32_t heads, uint32_t dim, uint32_t compression, float scale);
    // The groups a selection takes for `rows` rows (at most kSelectRows) from `row` on, their scores a row of `groups` apart (shaders/kv_index_select.comp).
    // `wide` spreads it over slices of kSelectSlice groups (shaders/kv_index_select_wide.comp): three histograms of 11, 11 and 10 bits, a count and an emission, five dispatches giving the same ids.
    void select_rows(VkDescriptorBufferInfo scores, VkDescriptorBufferInfo ids, const std::vector<uint32_t>& table, uint32_t view, size_t row, size_t rows,
                     size_t groups, uint32_t compression, uint32_t budget, uint32_t stride, bool wide);
    // The table the batched cache kernels read (shaders/views.glsl) for a subset of a batch's views.
    // Every view is checked against the storage by BlockKVStorage::check_view, which backs the blocks a writing op's rows land in and requires a reading op's written.
    struct ViewTable {
        std::vector<uint32_t> words;
        size_t rows = 0;
        VulkanKVStorage* storage = nullptr;
    };
    ViewTable view_table(size_t layer, const std::vector<Placed>& placed, bool writing);
    static void check_storage(const VulkanKVStorage& s, int n_head_kv, int head_dim);
    void keep_until_retired(BufferPtr b);

private:
    static constexpr uint32_t kRing = 16;
    uint32_t chunk_ = 0;
    // The dispatches the open command buffer takes before it is submitted, and the last submission that held any (docs/VULKAN.md, Structure).
    static constexpr uint32_t kFirstChunk = 8;
    uint32_t chunk_limit_ = kFirstChunk;
    Ticket dispatched_ = 0;
    bool unordered_ = false;   // barriers left out (unordered)
    static constexpr size_t kStagingBytes = size_t(64) << 20;
    static constexpr size_t kArenaBytes = size_t(1) << 20;
    static constexpr uint32_t kPushBytes = 128;

    struct Arena {
        std::unique_ptr<VulkanBuffer> buffer;
        size_t used = 0;
    };

    static VulkanKVStorage& storage_of(KVStorage* storage);

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

    static void validate_runs(size_t rows, RowRuns runs) {
        if (!runs.n) return;
        if (!runs.runs) throw std::runtime_error("vulkan: row runs without storage");
        for_each_run(rows, runs, [](const RowRun&) { return 0; }, [](size_t, size_t, int) {});
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
    static bool overlaps(CSlice a, size_t an, CSlice b, size_t bn) {
        return an && bn && a.buffer == b.buffer && a.offset < b.offset + bn && b.offset < a.offset + an;
    }

    VkDescriptorBufferInfo bind(CSlice s);
    VkDescriptorBufferInfo bind(Slice s);
    static void check_weight_scales(WeightView weight, size_t experts = 1);
    VkDescriptorBufferInfo weight_floats(WeightView weight, VkDescriptorBufferInfo otherwise);
    VkDescriptorBufferInfo lookup_grid(uint32_t type, VkDescriptorBufferInfo otherwise);
    VkDescriptorBufferInfo args(const void* data, size_t bytes);
    void destroy_kernel(Kernel& k) noexcept;
    Kernel& kernel(KernelId id, int variant = 0);
    void dispatch(KernelId id, std::initializer_list<VkDescriptorBufferInfo> buffers,
                  const void* push, size_t push_bytes, uint32_t groups_x, uint32_t groups_y = 1,
                  int variant = 0, std::initializer_list<size_t> extents = {});

    // The open command buffer, beginning the next ring slot once its last submission has retired.
    VkCommandBuffer open() {
        if (lent_.load(std::memory_order_relaxed)) throw std::logic_error("vulkan: a call on a backend whose queue a submission holds");
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
        // A device with no dispatch left running starts on the first few, rather than waiting while the host records a whole chunk.
        chunk_limit_ = done(dispatched_) ? kFirstChunk : dev_->profile.dispatch_chunk;
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

    VulkanBuffer& staging();
    void upload(VulkanBuffer& dst, size_t off, const void* src, size_t bytes, const std::shared_ptr<VulkanBuffer>& keep = nullptr);

    // A wait on ticket t, its time added to `into` on a backend made to time its work.
    void waited(double& into, Ticket t) noexcept {
        if (!timed_) return wait_for(t);
        const auto t0 = std::chrono::steady_clock::now();
        wait_for(t);
        // A wait from another thread adds its time beside the recording thread's.
        std::lock_guard<std::mutex> lk(times_);
        into += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    std::mutex times_;

    std::shared_ptr<Device> dev_;
    const bool timed_;       // made for diagnostics, so its waits and uploads are timed (host_times)
    HostTimes host_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer ring_[kRing] = {};
    Ticket ring_ticket_[kRing] = {};
    uint32_t ring_index_ = 0;
    bool open_ = false;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    std::shared_ptr<VulkanSubmitter> submitter_;   // a tensor group's queue thread, which makes this backend's queue calls once set
    std::atomic<Ticket> made_ticket_{0};          // on a submitter, the last ticket whose queue call was made or given up
    std::atomic<Ticket> good_ticket_{0};          // of those, the last whose call was made
    std::mutex lost_mu_;
    std::vector<std::pair<Ticket, Ticket>> lost_;   // the runs of tickets whose calls were given up, which raise refuses to read
    TicketLedger tickets_;   // what a successful Vulkan wait confirmed is its retired ticket
    std::vector<VkSemaphore> waits_;          // what the next submission waits on (wait_on)
    std::atomic<bool> lent_{false};           // between prepare() and queued(), while another thread may be making the queue's call
    static constexpr uint32_t kHolds = 4;       // hold submissions in flight; one is pending at a time
    static constexpr int kHoldMs = 100;         // the longest a held queue waits for its host, on the device's clock
    struct {
        bool on = false;                        // some holder remains
        bool refused = false;                   // the device keeps no hold, without a device clock or a hold seeing its release, and said so
        size_t holders = 0;
        VkCommandBuffer cmds[kHolds] = {};
        VkCommandPool pool = VK_NULL_HANDLE;    // the holds' own, apart from the ring's
        VkSemaphore timeline = VK_NULL_HANDLE;  // signalled as each hold ends
        uint64_t queued[kHolds] = {};           // per slot, the value of its last hold whose call was made, read and written only where the calls are made
        uint64_t last = 0;
        uint32_t slot = 0;
        uint32_t seq = 0;                       // the last hold queued; a hold ends once the release counter reaches its sequence
        uint32_t ticks = 0;                     // kHoldMs in the device clock's ticks
        bool pending = false;                   // the last hold is queued and not yet released
        std::unique_ptr<VulkanBuffer> word;     // the release counter, which the host writes and the holds read, then the sequence of the last hold its bound ended
    } hold_;
    std::unique_ptr<VulkanBuffer> staging_;
    Ticket staged_[2] = {};                   // the last copy out of each half of staging
    size_t next_half_ = 0;                    // the half of staging the next upload fills first
    std::array<BufferPtr, sizeof(weight_formats) / sizeof(weight_formats[0])> lookup_grids_;
    BufferPtr unit_scale_;
    std::shared_ptr<VulkanBuffer> scratch_;   // attention split states; stream-ordered reuse
    std::shared_ptr<VulkanBuffer> select_state_;   // a wide selection's per-row histograms, prefixes and slice counts (select_rows)
    std::vector<VkQueryPool> queries_;        // timestamps, only for a diagnostics backend, as many pools as a reading interval has needed
    static constexpr uint32_t kQueries = 8192;    // a pool's queries, two per dispatch
    size_t query_next_ = 0;        // the next query of the reading interval, across its pools
    size_t pools_reset_ = 0;       // the pools this interval has reset; a reading empties them all
    size_t last_timed_ = 0;        // dispatches the last reading covered
    static constexpr int kTimedKernels = K_COUNT * kVariants;
    bool head_ = false;               // on a backend that times its dispatches, those being recorded are the output head's (matmul_logits)
    std::vector<int> query_kernel_;   // id * kVariants + variant, past kTimedKernels for the output head's
    double kernel_ns_[2 * kTimedKernels] = {0};
    size_t kernel_calls_[2 * kTimedKernels] = {0};
    std::shared_ptr<VulkanBuffer> mx_copy_; // copies of one call, reused in queue order
    std::shared_ptr<VulkanBuffer> parts_;     // a split integer-dot tile call's partial sums
    std::shared_ptr<VulkanBuffer> argmax_parts_;   // an argmax's partials, a slice of a row each; likewise
    std::shared_ptr<VulkanBuffer> moe_out_;   // a routed down projection's slots before they are combined
    std::shared_ptr<VulkanBuffer> moe_tab_;   // a routed tile call's grouping (shaders/moe_group.comp)
    std::shared_ptr<VulkanBuffer> moe_count_; // the fused routed step's finished workgroups, zero between calls (shaders/moe_step_route.comp)
    // Which ids moe_tab_ groups: their location and count, cleared by every routing, by a new buffer (drop_tags) and by any write, copy or dispatch over its ids (forget_written).
    struct GroupTag { VkDescriptorBufferInfo ids{}; size_t entries = 0, n_expert = 0, chunk = 0; };
    GroupTag group_tag_;
    // A twin of a float input in its own stream-ordered scratch: the input's binding, its length and whether the 8-bit twin follows; empty once a write reaches the input (forget_written) and after a new buffer (drop_tags).
    // `major`: the 8-bit twin's blocks a column where it lies block-major (x8_major), else zero; `used` orders the slots; a mapped buffer is written from the host only through `write`.
    struct Twin {
        VkDescriptorBufferInfo x{};
        size_t n = 0;
        bool has8 = false;
        uint32_t major = 0;
        uint64_t used = 0;
        std::shared_ptr<VulkanBuffer> buffer;
    };
    // A few activations' twins stay alive at once, such as a layer's input beside its sublayers' outputs; the least recently used goes first.
    static constexpr size_t kTwins = 4;
    Twin twins_[kTwins];
    uint64_t twin_clock_ = 0;
    // Whether a matmul has read the 8-bit twin, after which producers write it too (twin_variant); it changes which copies exist, never a result.
    bool want_x8_ = false;

    std::vector<std::shared_ptr<VulkanBuffer>> pending_[kRing];
    Arena arena_[kRing];
    Kernel kernels_[K_COUNT][kVariants];
    // A diagnostic backend's fused routed kernels without their row products (kernel, constant 14), for the instruction counts of backend-vulkan --isa.
    VkPipeline role_free_[K_COUNT][kVariants] = {};
};

} // namespace
} // namespace backend

#include "backends/vulkan/vulkan_memory.hpp"
#include "backends/vulkan/vulkan_dispatch.hpp"
#include "backends/vulkan/vulkan_twins.hpp"
#include "backends/vulkan/vulkan_matmul.hpp"
#include "backends/vulkan/vulkan_moe.hpp"
#include "backends/vulkan/vulkan_attention.hpp"
#include "backends/vulkan/vulkan_collective.hpp"

namespace backend {

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

bool vulkan_prefer_integer_dot(Backend& backend) {
    auto* v = dynamic_cast<VulkanBackend*>(&backend);
    return v && v->prefer_integer_dot();
}

std::string vulkan_kernel_statistics(const Backend& backend) {
    const auto* v = dynamic_cast<const VulkanBackend*>(&backend);
    return v ? v->kernel_statistics() : std::string();
}

std::vector<std::pair<std::string, double>> vulkan_kernel_times(Backend& backend) {
    auto* v = dynamic_cast<VulkanBackend*>(&backend);
    return v ? v->kernel_times() : std::vector<std::pair<std::string, double>>();
}

std::vector<std::pair<std::string, std::vector<bool>>> vulkan_kernel_writes() {
    std::vector<std::pair<std::string, std::vector<bool>>> out;
    for (int i = 0; i < K_COUNT; ++i) {
        std::vector<bool> writes(kKernels[i].bindings);
        for (uint32_t b = 0; b < kKernels[i].bindings; ++b) writes[b] = kernel_writes()[i] >> b & 1;
        out.emplace_back(kKernelNames[i], writes);
    }
    return out;
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
