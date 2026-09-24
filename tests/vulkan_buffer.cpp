// Include the implementation to exercise private resource ownership with fake Vulkan calls, without adding runtime test hooks.
#include "backends/vulkan/vulkan_backend.cpp"
#include <iostream>
#include <stdexcept>
#include <new>

namespace {
thread_local int allocation_countdown = -1;
}
void* operator new(std::size_t size) {
    if (allocation_countdown > 0 && --allocation_countdown == 0) {
        allocation_countdown = -1;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace backend {
namespace {
struct VulkanLifetimeTest {
    static std::shared_ptr<Device> device(VulkanBackend& b) { return b.dev_; }
    static VkDescriptorBufferInfo padded(VulkanBackend& b, CSlice data, size_t rows) {
        return b.padded_f32(data, rows, 256);
    }
    static VkDescriptorBufferInfo args(VulkanBackend& b, const void* data, size_t bytes) {
        return b.args(data, bytes);
    }
};
}
}

namespace {
struct QueueCalls;
QueueCalls* queue_calls = nullptr;
struct QueueCalls {
    backend::Fn original;
    std::shared_ptr<backend::Device> device;
    VkBuffer pending[128]{};
    size_t count = 0;
    int premature = 0, allocations = 0, fail_allocation = 0, copies = 0, fail_copy = 0;
    bool fail_after_copy = false, copied = false;
    explicit QueueCalls(backend::VulkanBackend& b) : device(backend::VulkanLifetimeTest::device(b)) {
        original = device->fn;
        queue_calls = this;
        device->fn.vkAllocateMemory = allocate;
        device->fn.vkDestroyBuffer = destroy;
        device->fn.vkCmdFillBuffer = fill;
        device->fn.vkCmdCopyBuffer = copy;
        device->fn.vkWaitSemaphores = wait;
        device->fn.vkDeviceWaitIdle = idle;
    }
    ~QueueCalls() {
        allocation_countdown = -1;
        original.vkDeviceWaitIdle(device->device);
        device->fn = original;
        queue_calls = nullptr;
    }
    void mark(VkBuffer b) {
        for (size_t i = 0; i < count; ++i) if (pending[i] == b) return;
        if (count == 128) std::abort();
        pending[count++] = b;
    }
    // Replaced transfers record lifetimes without submitting references that a broken path could destroy.
    static VKAPI_ATTR void VKAPI_CALL fill(VkCommandBuffer, VkBuffer b, VkDeviceSize, VkDeviceSize, uint32_t) {
        queue_calls->mark(b);
    }
    static VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer, VkBuffer src, VkBuffer dst, uint32_t, const VkBufferCopy*) {
        auto& q = *queue_calls;
        q.mark(src); q.mark(dst); q.copied = true;
        if (q.fail_after_copy || ++q.copies == q.fail_copy) allocation_countdown = 1;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice d, const VkMemoryAllocateInfo* info,
                                                  const VkAllocationCallbacks* alloc, VkDeviceMemory* out) {
        auto& q = *queue_calls;
        if (++q.allocations == q.fail_allocation) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return q.original.vkAllocateMemory(d, info, alloc, out);
    }
    static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice d, VkBuffer b, const VkAllocationCallbacks* alloc) {
        auto& q = *queue_calls;
        for (size_t i = 0; i < q.count; ++i) if (q.pending[i] == b) ++q.premature;
        q.original.vkDestroyBuffer(d, b, alloc);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL wait(VkDevice d, const VkSemaphoreWaitInfo* info, uint64_t timeout) {
        auto& q = *queue_calls;
        const VkResult result = q.original.vkWaitSemaphores(d, info, timeout);
        if (result == VK_SUCCESS) q.count = 0;
        return result;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL idle(VkDevice d) {
        auto& q = *queue_calls;
        const VkResult result = q.original.vkDeviceWaitIdle(d);
        if (result == VK_SUCCESS) q.count = 0;
        return result;
    }
};

int queue_checks() {
    int failures = 0;
    for (int kind = 0; kind < 7; ++kind) {
        auto base = backend::make_vulkan_backend(0);
        auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
        QueueCalls q(b);
        bool threw = false, exercised = false;
        if (kind < 2 || kind == 4) {
            auto kv = b.kv_alloc(2, 1, 4, 512, backend::KVType::f32, backend::KVType::f32);
            auto& storage = dynamic_cast<backend::VulkanKVStorage&>(*kv);
            if (kind != 0) { storage.ensure(0); b.sync(); }
            const size_t held = kv->allocated_bytes(), peak = kv->peak_bytes();
            if (kind == 4) q.fail_copy = 4;
            else q.fail_allocation = q.allocations + 2;
            try { storage.ensure(kind == 0 ? 0 : 1); }
            catch (const std::bad_alloc&) { threw = true; }
            exercised = threw && kv->allocated_bytes() == held && kv->peak_bytes() == peak;
            q.fail_copy = q.fail_allocation = 0;
            storage.ensure(kind == 0 ? 0 : 1);
            exercised = exercised && kv->allocated_bytes() > held;
            b.sync();
        } else if (kind == 2 || kind == 5) {
            auto weights = b.alloc(2 * 256 * sizeof(float), backend::Memory::device);
            backend::as_vulkan(*weights).adopted = true;
            b.sync();
            if (kind == 5) {
                auto old = backend::VulkanLifetimeTest::padded(b, {weights.get(), 0}, 1);
                b.sync(); q.mark(old.buffer); q.copied = false;
            }
            q.fail_after_copy = true;
            try { backend::VulkanLifetimeTest::padded(b, {weights.get(), 0}, 2); }
            catch (const std::bad_alloc&) { threw = true; }
            allocation_countdown = -1;
            exercised = q.copied;
            b.sync();
        } else {
            std::vector<unsigned char> bytes((1 << 20), 0);
            const auto first = backend::VulkanLifetimeTest::args(b, bytes.data(), bytes.size() - 16);
            q.mark(first.buffer);
            if (kind == 6) q.fail_allocation = q.allocations + 1;
            else allocation_countdown = 2;
            try { backend::VulkanLifetimeTest::args(b, bytes.data(), 32); }
            catch (const std::bad_alloc&) { threw = true; }
            allocation_countdown = -1;
            exercised = threw;
            q.fail_allocation = 0;
            const int before_retry = q.allocations;
            const auto retry = backend::VulkanLifetimeTest::args(b, bytes.data(), 32);
            exercised = exercised && retry.buffer != VK_NULL_HANDLE;
            if (kind == 6) exercised = exercised && q.allocations == before_retry + 1;
            q.mark(retry.buffer);
            b.sync();
        }
        const bool ok = exercised && q.premature == 0;
        std::cout << "queue_case=" << kind << " threw=" << threw << " premature=" << q.premature
                  << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    std::cout << "vulkan queue ownership: 7 cases, " << failures << " failures (transfers intercepted)\n";
    return failures ? 1 : 0;
}
}

namespace {
enum class Failure { none, create, memory_type, allocate_device, allocate_host, bind, map };
struct Calls {
    Failure failure = Failure::none;
    int buffers = 0, memory = 0, maps = 0;
    bool bad_release = false;
    unsigned char bytes[64]{};
} calls;

VKAPI_ATTR VkResult VKAPI_CALL create_buffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer* out) {
    if (calls.failure == Failure::create) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out = (VkBuffer)(uintptr_t)1;
    ++calls.buffers;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {
    if (--calls.buffers != 0) calls.bad_release = true;
}
VKAPI_ATTR void VKAPI_CALL memory_requirements(VkDevice, VkBuffer, VkMemoryRequirements* out) {
    *out = VkMemoryRequirements{64, 4, 1};
}
VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    if (calls.failure == Failure::allocate_device) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (calls.failure == Failure::allocate_host) return VK_ERROR_OUT_OF_HOST_MEMORY;
    *out = (VkDeviceMemory)(uintptr_t)2;
    ++calls.memory;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {
    if (calls.buffers || calls.maps || --calls.memory != 0) calls.bad_release = true;
}
VKAPI_ATTR VkResult VKAPI_CALL bind_memory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize) {
    return calls.failure == Failure::bind ? VK_ERROR_OUT_OF_DEVICE_MEMORY : VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL map_memory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void** out) {
    if (calls.failure == Failure::map) return VK_ERROR_MEMORY_MAP_FAILED;
    ++calls.maps; *out = calls.bytes;
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL unmap_memory(VkDevice, VkDeviceMemory) {
    if (--calls.maps != 0) calls.bad_release = true;
}

std::shared_ptr<backend::Device> fake_device(Failure failure) {
    auto dev = std::make_shared<backend::Device>();
    dev->memory.memoryTypeCount = failure == Failure::memory_type ? 0 : 1;
    dev->memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto& fn = dev->fn;
    fn.vkCreateBuffer = create_buffer;
    fn.vkDestroyBuffer = destroy_buffer;
    fn.vkGetBufferMemoryRequirements = memory_requirements;
    fn.vkAllocateMemory = allocate_memory;
    fn.vkFreeMemory = free_memory;
    fn.vkBindBufferMemory = bind_memory;
    fn.vkMapMemory = map_memory;
    fn.vkUnmapMemory = unmap_memory;
    return dev;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--queue") return queue_checks();
        if (argc != 1) return 2;
        int failures = 0, cases = 0;
        for (const Failure failure : {Failure::none, Failure::create, Failure::memory_type,
                Failure::allocate_device, Failure::allocate_host, Failure::bind, Failure::map}) {
            calls = Calls{}; calls.failure = failure;
            auto dev = fake_device(failure);
            bool threw = false;
            try {
                backend::VulkanBuffer buffer(dev, 64, true);
                if (failure == Failure::none && (!buffer.host_ptr() || buffer.size() != 64))
                    throw std::runtime_error("successful buffer has no mapped storage");
            } catch (const std::exception&) { threw = true; }
            const bool ok = threw == (failure != Failure::none) && !calls.buffers && !calls.memory &&
                            !calls.maps && !calls.bad_release;
            std::cout << "case=" << int(failure) << " threw=" << threw << " live_buffers=" << calls.buffers
                      << " live_memory=" << calls.memory << " maps=" << calls.maps << (ok ? " PASS\n" : " FAIL\n");
            ++cases; if (!ok) ++failures;
        }
        calls = Calls{};
        {
            backend::VulkanBuffer empty(fake_device(Failure::none), 0, true);
            if (empty.size() || empty.host_ptr() || empty.handle()) ++failures;
        }
        ++cases;
        if (calls.buffers || calls.memory || calls.maps || calls.bad_release) ++failures;
        std::cout << "vulkan-buffer: " << cases << " cases, " << failures << " failures (fake API; no device)\n";
        return failures ? 1 : 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
