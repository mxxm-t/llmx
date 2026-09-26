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
#if defined(__GNUC__) && !defined(__clang__)
// GCC takes the free below for a mismatch with operator new, though the replaced new above allocates with malloc.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace backend {
namespace {
struct VulkanLifetimeTest {
    static std::shared_ptr<Device> device(VulkanBackend& b) { return b.dev_; }
    static VkDescriptorBufferInfo padded(VulkanBackend& b, CSlice data, size_t rows) {
        return b.padded_f32(data, gguf::GGML_TYPE_F32, rows, 256);
    }
    static VkDescriptorBufferInfo args(VulkanBackend& b, const void* data, size_t bytes) {
        return b.args(data, bytes);
    }
    static VkPipeline kernel(VulkanBackend& b) { return b.kernel(K_ADD).pipeline; }
    static bool query_empty(VulkanBackend& b) { return b.queries_ == VK_NULL_HANDLE; }
    static void clear_query(VulkanBackend& b) { b.queries_ = VK_NULL_HANDLE; }
    static void prepare_drop(VulkanBackend& b) {
        b.open();
        std::vector<std::shared_ptr<VulkanBuffer>> pending;
        pending.reserve(1);
        b.pending_[b.ring_index_].swap(pending);
    }
    static size_t retained(VulkanBackend& b) { return b.pending_[b.ring_index_].size(); }
    static void drop(VulkanBackend& b, VulkanBuffer& buffer) { b.drop_padded(buffer); }
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

struct KernelCalls;
KernelCalls* kernel_calls = nullptr;
struct KernelCalls {
    backend::Fn original;
    std::shared_ptr<backend::Device> device;
    int failure, creates = 0, modules = 0, sets = 0, layouts = 0, pipelines = 0;
    uintptr_t next = 1024;
    explicit KernelCalls(backend::VulkanBackend& b, int fail)
        : device(backend::VulkanLifetimeTest::device(b)), failure(fail) {
        original = device->fn;
        kernel_calls = this;
        auto& fn = device->fn;
        fn.vkCreateShaderModule = shader; fn.vkDestroyShaderModule = destroy_shader;
        fn.vkCreateDescriptorSetLayout = set; fn.vkDestroyDescriptorSetLayout = destroy_set;
        fn.vkCreatePipelineLayout = layout; fn.vkDestroyPipelineLayout = destroy_layout;
        fn.vkCreateComputePipelines = pipeline; fn.vkDestroyPipeline = destroy_pipeline;
    }
    ~KernelCalls() { allocation_countdown = -1; device->fn = original; kernel_calls = nullptr; }
    bool empty() const { return !modules && !sets && !layouts && !pipelines; }
    template <typename T> static VkResult create(T* out, int stage, int& live) {
        auto& q = *kernel_calls;
        ++q.creates;
        if (q.failure == stage) {
            *out = stage == 4 ? VK_NULL_HANDLE : (T)(uintptr_t)0xdead;
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        *out = (T)++q.next;
        ++live;
        if (stage == 1 && q.failure == 5) allocation_countdown = 1;
        return VK_SUCCESS;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL shader(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*, VkShaderModule* out) {
        return create(out, 1, kernel_calls->modules);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL set(VkDevice, const VkDescriptorSetLayoutCreateInfo*, const VkAllocationCallbacks*, VkDescriptorSetLayout* out) {
        return create(out, 2, kernel_calls->sets);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL layout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*, VkPipelineLayout* out) {
        return create(out, 3, kernel_calls->layouts);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL pipeline(VkDevice, VkPipelineCache, uint32_t, const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline* out) {
        return create(out, 4, kernel_calls->pipelines);
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_shader(VkDevice, VkShaderModule, const VkAllocationCallbacks*) { --kernel_calls->modules; }
    static VKAPI_ATTR void VKAPI_CALL destroy_set(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) { --kernel_calls->sets; }
    static VKAPI_ATTR void VKAPI_CALL destroy_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) { --kernel_calls->layouts; }
    static VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) { --kernel_calls->pipelines; }
};

int kernel_checks() {
    int failures = 0;
    for (int kind = 1; kind <= 5; ++kind) {
        auto base = backend::make_vulkan_backend(0);
        auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
        KernelCalls q(b, kind);
        bool threw = false;
        try { backend::VulkanLifetimeTest::kernel(b); }
        catch (const std::exception&) { threw = true; }
        allocation_countdown = -1;
        const bool cleaned = q.empty();
        q.failure = 0;
        const VkPipeline first = backend::VulkanLifetimeTest::kernel(b);
        const int creates = q.creates;
        const VkPipeline reused = backend::VulkanLifetimeTest::kernel(b);
        const bool cached = first && first == reused && creates == q.creates;
        base.reset();
        const bool ok = threw && cleaned && cached && q.empty();
        std::cout << "kernel_case=" << kind << " cleaned=" << cleaned << " cached=" << cached
                  << " empty_after_teardown=" << q.empty() << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    return failures;
}

struct QueryCalls;
QueryCalls* query_calls = nullptr;
struct QueryCalls {
    backend::Fn original;
    std::shared_ptr<backend::Device> device;
    VkQueryPool pool = VK_NULL_HANDLE;
    int creates = 0, destroys = 0;
    bool idle = false, ordered = false, fail = false, bad_destroy = false;
    explicit QueryCalls(backend::VulkanBackend& b) : device(backend::VulkanLifetimeTest::device(b)) {
        original = device->fn;
        query_calls = this;
        device->fn.vkCreateQueryPool = create;
        device->fn.vkDestroyQueryPool = destroy;
        device->fn.vkDeviceWaitIdle = wait_idle;
    }
    ~QueryCalls() {
        original.vkDeviceWaitIdle(device->device);
        if (pool && !destroys) original.vkDestroyQueryPool(device->device, pool, nullptr);
        device->fn = original;
        query_calls = nullptr;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice d, const VkQueryPoolCreateInfo* info, const VkAllocationCallbacks* alloc, VkQueryPool* out) {
        auto& q = *query_calls;
        if (q.fail) { *out = (VkQueryPool)(uintptr_t)0xdead; return VK_ERROR_OUT_OF_HOST_MEMORY; }
        const VkResult r = q.original.vkCreateQueryPool(d, info, alloc, out);
        if (r == VK_SUCCESS) { ++q.creates; q.pool = *out; }
        return r;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice d, VkQueryPool pool, const VkAllocationCallbacks* alloc) {
        auto& q = *query_calls;
        if (pool != q.pool) { q.bad_destroy = true; return; }
        ++q.destroys; q.ordered = q.idle;
        q.original.vkDestroyQueryPool(d, pool, alloc);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkDevice d) {
        auto& q = *query_calls;
        const VkResult r = q.original.vkDeviceWaitIdle(d);
        q.idle = r == VK_SUCCESS;
        return r;
    }
};

int query_checks(bool fail) {
    auto base = backend::make_vulkan_backend(0, true);
    auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
    QueryCalls q(b);
    if (!q.device->timestamps) {
        std::cout << "query teardown: SKIP (device has no diagnostic timestamps)\n";
        base.reset();
        return 0;
    }
    auto dst = b.alloc(sizeof(float), backend::Memory::host_visible);
    bool threw = false, clean_failure = true;
    if (fail) {
        q.fail = true;
        try { b.add({dst.get(), 0}, {dst.get(), 0}, 1); }
        catch (const std::runtime_error&) { threw = true; }
        clean_failure = threw && backend::VulkanLifetimeTest::query_empty(b);
        // An old implementation may have kept the poisoned output; do not let the control submit it.
        if (!clean_failure) backend::VulkanLifetimeTest::clear_query(b);
        q.fail = false;
    }
    b.add({dst.get(), 0}, {dst.get(), 0}, 1);
    b.sync();
    q.idle = false;
    base.reset();
    const bool ok = clean_failure && !q.bad_destroy && q.creates == 1 && q.destroys == 1 && q.ordered;
    std::cout << "query_case=" << fail << " clean_failure=" << clean_failure
              << " created=" << q.creates << " destroyed=" << q.destroys
              << " after_idle=" << q.ordered << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;
}

int padded_drop_checks() {
    auto base = backend::make_vulkan_backend(0);
    auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
    QueueCalls q(b);
    auto weights = b.alloc(512 * sizeof(float), backend::Memory::host_visible);
    auto& w = backend::as_vulkan(*weights);
    w.adopted = true;
    const auto first = backend::VulkanLifetimeTest::padded(b, {weights.get(), 0}, 1);
    const auto second = backend::VulkanLifetimeTest::padded(b, {weights.get(), 256}, 1);
    b.sync();
    backend::VulkanLifetimeTest::prepare_drop(b);
    q.mark(first.buffer); q.mark(second.buffer);
    allocation_countdown = 1;
    bool threw = false;
    try { backend::VulkanLifetimeTest::drop(b, w); }
    catch (const std::bad_alloc&) { threw = true; }
    allocation_countdown = -1;
    bool intact = w.padded.size() == 2;
    for (const auto& entry : w.padded) intact = intact && bool(entry.second.copy);
    if (intact) {
        intact = backend::VulkanLifetimeTest::padded(b, {weights.get(), 0}, 1).buffer == first.buffer &&
                 backend::VulkanLifetimeTest::padded(b, {weights.get(), 256}, 1).buffer == second.buffer;
    }
    backend::VulkanLifetimeTest::drop(b, w);
    const bool retained = backend::VulkanLifetimeTest::retained(b) == 2 && w.padded.empty();
    b.sync();
    const bool ok = threw && intact && retained && q.premature == 0;
    std::cout << "padded_drop threw=" << threw << " intact=" << intact << " retained=" << retained
              << " premature=" << q.premature << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;
}

// A loader's weight: alloc_weight storage is device-local and adopted, so written in pieces that end inside a row it holds the bytes and still gets the float tile's padded copy, which storage from alloc does not.
int loader_weight_checks() {
    auto base = backend::make_vulkan_backend(0);
    auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
    std::vector<float> w(2 * 256);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(i % 97) - 48.0f;
    const size_t bytes = w.size() * sizeof(float);
    auto filled = b.alloc_weight(bytes);
    auto& fv = backend::as_vulkan(*filled);
    for (size_t off = 0; off < bytes; off += 1000) b.write(*filled, off, (const uint8_t*)w.data() + off, std::min<size_t>(1000, bytes - off));
    const auto binding = backend::VulkanLifetimeTest::padded(b, {filled.get(), 0}, 2);
    auto plain = b.alloc(bytes, backend::Memory::device);
    const auto plain_binding = backend::VulkanLifetimeTest::padded(b, {plain.get(), 0}, 2);
    std::vector<float> back(w.size());
    b.read(*filled, 0, back.data(), bytes);
    b.sync();
    const bool held = std::memcmp(back.data(), w.data(), bytes) == 0;
    const bool padded = fv.adopted && !filled->host_ptr() && binding.buffer != fv.handle() && fv.padded.size() == 1;
    const bool unpadded = plain_binding.buffer == backend::as_vulkan(*plain).handle();
    const bool ok = held && padded && unpadded;
    std::cout << "loader_weight held=" << held << " padded=" << padded << " plain_unpadded=" << unpadded << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;
}

int queue_checks() {
    int failures = 0;
    for (int kind = 0; kind < 8; ++kind) {
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
        } else if (kind == 7) {
            // A buffer dropped before anything is submitted outlives its zero fill, and an adopted one its upload.
            { auto dropped = b.alloc(256, backend::Memory::device); }
            const std::vector<unsigned char> bytes(256, 1);
            { auto dropped = b.adopt(bytes.data(), bytes.size()); }
            exercised = q.count >= 2;
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
    std::cout << "vulkan queue ownership: 8 cases, " << failures << " failures (transfers intercepted)\n";
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
        if (argc == 2 && std::string(argv[1]) == "--queue") {
            const int failures = queue_checks() + kernel_checks() + query_checks(false) + query_checks(true) + padded_drop_checks() +
                                 loader_weight_checks();
            return failures ? 1 : 0;
        }
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
