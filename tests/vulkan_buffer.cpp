// Include the implementation to exercise private resource ownership with fake Vulkan calls, without adding runtime test hooks.
#include "backends/vulkan/vulkan_backend.cpp"
#include <atomic>
#include <iostream>
#include <thread>
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
        return b.padded_f32(data, quant::GGML_TYPE_F32, rows, 256);
    }
    static VkDescriptorBufferInfo args(VulkanBackend& b, const void* data, size_t bytes) {
        return b.args(data, bytes);
    }
    static VkPipeline kernel(VulkanBackend& b) { return b.kernel(K_ADD).pipeline; }
    static bool query_empty(VulkanBackend& b) { return b.queries_.empty(); }
    static void clear_query(VulkanBackend& b) { b.queries_.clear(); }
    static void prepare_drop(VulkanBackend& b) {
        b.open();
        std::vector<std::shared_ptr<VulkanBuffer>> pending;
        pending.reserve(1);
        b.pending_[b.ring_index_].swap(pending);
    }
    static size_t retained(VulkanBackend& b) { return b.pending_[b.ring_index_].size(); }
    static void drop(VulkanBackend& b, VulkanBuffer& buffer) { b.drop_padded(buffer); }
    // Every submission retired, nothing open and no wait left for a next submission.
    static bool idle(VulkanBackend& b) {
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &b.timeline_;
        const uint64_t last = b.last_ticket_;
        wi.pValues = &last;
        return !b.open_ && b.waits_.empty() && b.dev_->fn.vkWaitSemaphores(b.dev_->device, &wi, 0) == VK_SUCCESS;
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

// A row's MXFP4 decoder can use integer dots without ever creating a tile copy.
// Fitting must reserve that copy only when the device's tile path can reach it.
int scratch_reserve_checks() {
    backend::VulkanBackend b(0);
    const auto dev = backend::VulkanLifetimeTest::device(b);
    const auto saved = dev->profile;
    int failures = 0;
    for (const size_t free : {size_t(0), size_t(1) << 30, size_t(32) << 30}) {
        for (const bool row_dot : {false, true}) {
            dev->profile.mxfp4_integer_dot = row_dot;
            dev->profile.prefer_integer_dot = true;
            const size_t copied = b.scratch_reserve(free);
            dev->profile.prefer_integer_dot = false;
            const size_t direct = b.scratch_reserve(free);
            const bool ok = copied >= backend::kMxCopyBytes && direct == copied - backend::kMxCopyBytes;
            std::cout << "scratch free=" << free << " row_dot=" << row_dot
                      << " copy=" << copied << " no_copy=" << direct << (ok ? " PASS\n" : " FAIL\n");
            if (!ok) ++failures;
        }
    }
    dev->profile = saved;
    return failures;
}

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

// Hold setup (Backend::hold_between_submissions) failing at its timeline (1) or at one of its four events (2 to 5): what it made is destroyed, and a later call makes the hold whole.
struct HoldCalls;
HoldCalls* hold_calls = nullptr;
struct HoldCalls {
    backend::Fn original;
    std::shared_ptr<backend::Device> device;
    int failure, creates = 0, freed = 0;
    std::vector<uint64_t> live;   // the semaphores and events made since these calls were set
    HoldCalls(backend::VulkanBackend& b, int fail) : device(backend::VulkanLifetimeTest::device(b)), failure(fail) {
        original = device->fn;
        hold_calls = this;
        auto& fn = device->fn;
        fn.vkCreateSemaphore = semaphore; fn.vkDestroySemaphore = destroy_semaphore;
        fn.vkCreateEvent = event; fn.vkDestroyEvent = destroy_event;
        fn.vkFreeCommandBuffers = free_cmds;
    }
    ~HoldCalls() { device->fn = original; hold_calls = nullptr; }
    bool fail_next() { return ++creates == failure; }
    void gone(uint64_t h) {
        for (size_t i = 0; i < live.size(); ++i)
            if (live[i] == h) { live.erase(live.begin() + (std::ptrdiff_t)i); return; }
    }
    static VKAPI_ATTR VkResult VKAPI_CALL semaphore(VkDevice d, const VkSemaphoreCreateInfo* ci, const VkAllocationCallbacks* a, VkSemaphore* out) {
        auto& q = *hold_calls;
        if (q.fail_next()) return VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult r = q.original.vkCreateSemaphore(d, ci, a, out);
        if (r == VK_SUCCESS) q.live.push_back((uint64_t)*out);
        return r;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL event(VkDevice d, const VkEventCreateInfo* ci, const VkAllocationCallbacks* a, VkEvent* out) {
        auto& q = *hold_calls;
        if (q.fail_next()) return VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult r = q.original.vkCreateEvent(d, ci, a, out);
        if (r == VK_SUCCESS) q.live.push_back((uint64_t)*out);
        return r;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a) {
        hold_calls->gone((uint64_t)s);
        hold_calls->original.vkDestroySemaphore(d, s, a);
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_event(VkDevice d, VkEvent e, const VkAllocationCallbacks* a) {
        hold_calls->gone((uint64_t)e);
        hold_calls->original.vkDestroyEvent(d, e, a);
    }
    static VKAPI_ATTR void VKAPI_CALL free_cmds(VkDevice d, VkCommandPool p, uint32_t n, const VkCommandBuffer* c) {
        ++hold_calls->freed;
        hold_calls->original.vkFreeCommandBuffers(d, p, n, c);
    }
};

int hold_checks() {
    int failures = 0;
    const std::vector<uint8_t> src = [] { std::vector<uint8_t> v(4096); for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t)(i * 7 + 1); return v; }();
    for (int kind = 1; kind <= 5; ++kind) {
        auto base = backend::make_vulkan_backend(0);
        auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
        bool threw = false, cleaned = false, copied = false;
        {
            HoldCalls q(b, kind);
            try { b.hold_between_submissions(true); }
            catch (const std::exception&) { threw = true; }
            cleaned = q.live.empty() && q.freed == 1;
            q.failure = 0;
            b.hold_between_submissions(true);
            const auto s = b.adopt(src.data(), src.size());
            const auto d = b.alloc(src.size(), backend::Memory::host_visible);
            b.copy(*d, 0, *s, 0, 2048);
            b.wait(b.submit());
            b.copy(*d, 2048, *s, 2048, 2048);
            b.wait(b.submit());
            copied = std::memcmp(d->host_ptr(), src.data(), src.size()) == 0;
            b.hold_between_submissions(false);
            base.reset();
            cleaned = cleaned && q.live.empty();
        }
        const bool ok = threw && cleaned && copied;
        std::cout << "hold_case=" << kind << " threw=" << threw << " cleaned=" << cleaned << " copied=" << copied
                  << (ok ? " PASS\n" : " FAIL\n");
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
// A tensor group's collective over devices 0 and 1 (VulkanCollective): its semaphores freed when a join fails part way, its members drained before any goes, and a sum that fails part way leaving it as new.
struct CollectiveCalls;
CollectiveCalls* collective_calls = nullptr;
struct CollectiveCalls {
    std::vector<backend::VulkanBackend*> members;
    std::vector<backend::Fn> original;
    std::vector<PFN_vkGetSemaphoreFdKHR> get_fd;
    std::vector<PFN_vkImportSemaphoreFdKHR> import_fd;
    int creates = 0, live = 0, fail_create = 0, fail_get = 0, fail_import = 0, gets = 0, imports = 0;
    bool premature = false;
    explicit CollectiveCalls(const std::vector<backend::VulkanBackend*>& m) : members(m) {
        collective_calls = this;
        for (auto* b : members) {
            auto dev = backend::VulkanLifetimeTest::device(*b);
            original.push_back(dev->fn);
            get_fd.push_back(dev->get_semaphore_fd);
            import_fd.push_back(dev->import_semaphore_fd);
            dev->fn.vkCreateSemaphore = create;
            dev->fn.vkDestroySemaphore = destroy;
            dev->get_semaphore_fd = get;
            dev->import_semaphore_fd = import;
        }
    }
    ~CollectiveCalls() {
        for (size_t i = 0; i < members.size(); ++i) {
            auto dev = backend::VulkanLifetimeTest::device(*members[i]);
            dev->fn = original[i];
            dev->get_semaphore_fd = get_fd[i];
            dev->import_semaphore_fd = import_fd[i];
        }
        collective_calls = nullptr;
    }
    size_t index(VkDevice d) const {
        for (size_t i = 0; i < members.size(); ++i)
            if (backend::VulkanLifetimeTest::device(*members[i])->device == d) return i;
        std::abort();
    }
    static VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice d, const VkSemaphoreCreateInfo* info, const VkAllocationCallbacks* alloc, VkSemaphore* out) {
        auto& q = *collective_calls;
        if (q.fail_create && ++q.creates == q.fail_create) return VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult r = q.original[q.index(d)].vkCreateSemaphore(d, info, alloc, out);
        if (r == VK_SUCCESS) ++q.live;
        return r;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy(VkDevice d, VkSemaphore s, const VkAllocationCallbacks* alloc) {
        auto& q = *collective_calls;
        for (auto* b : q.members)
            if (!backend::VulkanLifetimeTest::idle(*b)) q.premature = true;
        --q.live;
        q.original[q.index(d)].vkDestroySemaphore(d, s, alloc);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL get(VkDevice d, const VkSemaphoreGetFdInfoKHR* info, int* fd) {
        auto& q = *collective_calls;
        if (++q.gets == q.fail_get) return VK_ERROR_TOO_MANY_OBJECTS;
        return q.get_fd[q.index(d)](d, info, fd);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL import(VkDevice d, const VkImportSemaphoreFdInfoKHR* info) {
        auto& q = *collective_calls;
        if (++q.imports == q.fail_import) return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        return q.import_fd[q.index(d)](d, info);
    }
};

// The one call another thread may make on a backend while its own thread records (Backend::wait): a thread copies and submits without pause while this one waits on each ticket it has been handed, and the bytes must arrive as written.
int wait_beside_recording() {
    auto base = backend::make_vulkan_backend(0);
    backend::Backend& b = *base;
    const size_t n = 4096, rounds = 400;
    std::vector<uint8_t> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = (uint8_t)(i * 13 + 5);
    const auto a = b.adopt(src.data(), n);
    const auto out = b.alloc(n, backend::Memory::host_visible);
    b.sync();
    std::atomic<uint64_t> handed{0};
    std::atomic<bool> done{false};
    std::thread recorder([&] {
        for (size_t r = 0; r < rounds; ++r) {
            b.copy(*out, 0, *a, 0, n);
            handed.store(b.submit());
        }
        done.store(true);
    });
    size_t waits = 0;
    while (!done.load()) {
        if (const uint64_t t = handed.load()) {
            b.wait(t);
            ++waits;
        }
    }
    recorder.join();
    b.sync();
    const bool ok = waits > 0 && std::memcmp(out->host_ptr(), src.data(), n) == 0;
    std::cout << "wait_beside_recording waits=" << waits << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;
}

int collective_checks() {
    backend::BackendPtr first = backend::make_vulkan_backend(0), second;
    try {
        second = backend::make_vulkan_backend(1);
    } catch (const std::exception& e) {
        std::cout << "collective: no second device (" << e.what() << "), skipped\n";
        return 0;
    }
    std::vector<backend::Backend*> members{first.get(), second.get()};
    if (!first->join(members, 1, 1)) {
        std::cout << "collective: the devices share no memory or semaphores, skipped\n";
        return 0;
    }
    std::vector<backend::VulkanBackend*> vk{&dynamic_cast<backend::VulkanBackend&>(*first), &dynamic_cast<backend::VulkanBackend&>(*second)};
    const size_t rows = 5, width = 64, n = rows * width;
    int failures = 0;
    uint32_t seed = 3;
    auto value = [&] { seed = seed * 1664525u + 1013904223u; return float(int(seed >> 16) % 2001 - 1000) / 97.0f; };
    // Sums back to back on fresh residuals, each member's partial written before its sum and nothing read between, the collective destroyed before the residuals are read; true when every member's residual is the chain of sums in member order.
    auto chain = [&](std::unique_ptr<backend::Collective>& sum, size_t sums) {
        std::vector<float> want(n);
        for (float& v : want) v = value();
        std::vector<backend::BufferPtr> x;
        std::vector<backend::Slice> xs;
        for (size_t m = 0; m < 2; ++m) {
            x.push_back(members[m]->alloc(n * sizeof(float)));
            members[m]->write(*x.back(), 0, want.data(), n * sizeof(float));
            xs.push_back({x.back().get(), 0});
        }
        for (size_t k = 0; k < sums; ++k) {
            std::vector<std::vector<float>> partial(2, std::vector<float>(n));
            for (size_t m = 0; m < 2; ++m) {
                for (float& v : partial[m]) v = value();
                const backend::Slice p = sum->partial(m);
                members[m]->write(*p.buffer, p.offset * sizeof(float), partial[m].data(), n * sizeof(float));
            }
            sum->sum_into(xs, rows, width);
            for (size_t i = 0; i < n; ++i) want[i] = want[i] + (partial[0][i] + partial[1][i]);
        }
        sum.reset();
        bool same = true;
        for (size_t m = 0; m < 2; ++m) {
            std::vector<float> got(n);
            members[m]->read(*x[m], 0, got.data(), n * sizeof(float));
            same = same && std::memcmp(got.data(), want.data(), n * sizeof(float)) == 0;
        }
        return same;
    };
    for (int kind = 1; kind <= 4; ++kind) {
        bool threw = false, clean = false, summed = false;
        {
            CollectiveCalls q(vk);
            q.fail_create = kind;
            try { first->join(members, rows, width); }
            catch (const std::exception&) { threw = true; }
            clean = q.live == 0 && !q.premature;
            q.fail_create = 0;
            auto sum = first->join(members, rows, width);
            summed = chain(sum, 3) && q.live == 0 && !q.premature;
        }
        const bool ok = threw && clean && summed;
        std::cout << "collective_join_case=" << kind << " threw=" << threw << " clean=" << clean << " summed=" << summed << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    {
        CollectiveCalls q(vk);
        auto sum = first->join(members, rows, width);
        const bool ok = chain(sum, 7) && q.live == 0 && !q.premature;
        std::cout << "collective_in_flight sums=7" << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    {
        // A member that cannot import its peer's inbox refuses the group, naming its devices, and leaves nothing behind.
        auto dev = backend::VulkanLifetimeTest::device(*vk[1]);
        const auto original = dev->memory_fd_props;
        dev->memory_fd_props = [](VkDevice, VkExternalMemoryHandleTypeFlagBits, int, VkMemoryFdPropertiesKHR*) -> VkResult { return VK_ERROR_INVALID_EXTERNAL_HANDLE; };
        std::string why;
        {
            CollectiveCalls q(vk);
            try { first->join(members, rows, width); }
            catch (const std::exception& e) { why = e.what(); }
            dev->memory_fd_props = original;
            const bool named = why.find("cannot share its memory and semaphores") != std::string::npos && why.find(backend::vulkan_device_name(*second)) != std::string::npos;
            auto sum = first->join(members, rows, width);
            const bool ok = named && chain(sum, 3) && q.live == 0 && !q.premature;
            std::cout << "collective_import_refused named=" << named << (ok ? " PASS\n" : " FAIL\n");
            if (!ok) ++failures;
        }
    }
    for (int kind = 1; kind <= 4; ++kind) {
        bool threw = false, drained = false, summed = false;
        {
            CollectiveCalls q(vk);
            auto sum = first->join(members, rows, width);
            const int live = q.live;
            if (kind <= 2) q.fail_get = kind; else q.fail_import = kind - 2;
            std::vector<backend::BufferPtr> x;
            std::vector<backend::Slice> xs;
            for (size_t m = 0; m < 2; ++m) {
                x.push_back(members[m]->alloc(n * sizeof(float)));
                xs.push_back({x.back().get(), 0});
            }
            try { sum->sum_into(xs, rows, width); }
            catch (const std::exception&) { threw = true; }
            drained = backend::VulkanLifetimeTest::idle(*vk[0]) && backend::VulkanLifetimeTest::idle(*vk[1]) && q.live == live && !q.premature;
            q.fail_get = q.fail_import = 0;
            summed = chain(sum, 3) && q.live == 0 && !q.premature;
        }
        const bool ok = threw && drained && summed;
        std::cout << "collective_sum_case=" << kind << " threw=" << threw << " drained=" << drained << " summed=" << summed << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    return failures;
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

namespace {
// The device needs the kernels declare, on property and feature values rather than a device: all present, then each taken away in turn, which must be named.
int device_need_checks() {
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.subgroupSize = 64;
    sg.supportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    VkPhysicalDeviceFeatures2 f2{};
    f2.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    f2.features.shaderInt16 = VK_TRUE;
    VkPhysicalDeviceVulkan11Features f11{};
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{};
    f12.timelineSemaphore = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_TRUE;
    int failures = 0;
    const auto expect = [&](const char* what, const std::string& need) {
        const std::string got = backend::missing_device_need(sg, f2, f11, f12);
        const bool ok = need.empty() ? got.empty() : got.find(need) != std::string::npos;
        std::cout << "device_need " << what << ": \"" << got << "\"" << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    };
    expect("all present", "");
    sg.subgroupSize = 32; expect("32 lanes", ""); sg.subgroupSize = 16; expect("16 lanes", "narrower than 32"); sg.subgroupSize = 96; expect("96 lanes", "subgroup size"); sg.subgroupSize = 64;
    sg.supportedOperations &= ~VK_SUBGROUP_FEATURE_ARITHMETIC_BIT; expect("no subgroup arithmetic", "subgroup arithmetic"); sg.supportedOperations |= VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
    sg.supportedOperations &= ~VK_SUBGROUP_FEATURE_SHUFFLE_BIT; expect("no subgroup shuffle", "subgroup shuffle"); sg.supportedOperations |= VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    f12.timelineSemaphore = VK_FALSE; expect("no timeline semaphores", "timeline semaphores"); f12.timelineSemaphore = VK_TRUE;
    f2.features.shaderStorageBufferArrayDynamicIndexing = VK_FALSE; expect("no dynamic indexing", "storage buffer arrays"); f2.features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    f2.features.shaderInt16 = VK_FALSE; expect("no 16-bit integers", "16-bit integer arithmetic"); f2.features.shaderInt16 = VK_TRUE;
    f12.shaderInt8 = VK_FALSE; expect("no 8-bit integers", "8-bit integer arithmetic"); f12.shaderInt8 = VK_TRUE;
    f11.storageBuffer16BitAccess = VK_FALSE; expect("no 16-bit storage", "16-bit storage"); f11.storageBuffer16BitAccess = VK_TRUE;
    f12.storageBuffer8BitAccess = VK_FALSE; expect("no 8-bit storage", "8-bit storage"); f12.storageBuffer8BitAccess = VK_TRUE;
    // Heads 128 and 256 wide take kernels whose lane groups fit any subgroup the device check accepts, 256 on a 32-lane device included.
    // Other widths take the per-row kernel, which reads four elements of a head per lane of a subgroup, so such a head wider than four subgroups does not fit.
    const bool widths = backend::attention_head_fits(128, 32) && backend::attention_head_fits(256, 32) && !backend::attention_head_fits(192, 32) &&
                        backend::attention_head_fits(192, 64) && backend::attention_head_fits(256, 64) && !backend::attention_head_fits(260, 64) &&
                        !backend::attention_head_fits(512, 64) && backend::attention_head_fits(64, 64) && backend::attention_head_fits(80, 32);
    std::cout << "attention head widths against 32- and 64-lane subgroups" << (widths ? " PASS\n" : " FAIL\n");
    // The integer-dot kernels (the Q8_0 decode kernel and the integer-dot tile, whose 16-bit and packed 8-bit dots need the integer dot product) run only under a profile that prefers the integer dot, which a device without the extension never gets, even where its row asks for it.
    backend::DeviceCaps caps;
    caps.device = "AMD Radeon Instinct MI60 / MI50 (RADV VEGA20)";
    caps.driver = "radv Mesa";
    caps.integer_dot = true;
    const backend::DeviceProfile with = backend::profile_for(caps);
    caps.integer_dot = false;
    const backend::DeviceProfile without = backend::profile_for(caps);
    const bool gate = with.prefer_integer_dot && with.mxfp4_integer_dot && !without.prefer_integer_dot && !without.mxfp4_integer_dot;
    std::cout << "integer-dot kernels only with the integer dot product" << (gate ? " PASS\n" : " FAIL\n");
    return failures + !widths + !gate;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--queue") {
            const int failures = queue_checks() + kernel_checks() + query_checks(false) + query_checks(true) + padded_drop_checks() +
                                 loader_weight_checks() + scratch_reserve_checks() + hold_checks() + collective_checks() + wait_beside_recording();
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
        failures += device_need_checks();
        cases += 14;
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
