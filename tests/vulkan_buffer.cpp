// Include the implementation to exercise private resource ownership with fake Vulkan calls, without adding runtime test hooks.
#include "backends/vulkan/vulkan_backend.cpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
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
    static bool owns(VulkanBackend& b, VkCommandBuffer cmd) {
        for (VkCommandBuffer c : b.ring_) if (c == cmd) return true;
        return false;
    }
    static VkDescriptorBufferInfo padded(VulkanBackend& b, CSlice data, size_t rows) {
        return b.padded_f32(data, quant::GGML_TYPE_F32, rows, 256);
    }
    static VkDescriptorBufferInfo args(VulkanBackend& b, const void* data, size_t bytes) {
        return b.args(data, bytes);
    }
    static uint32_t slot(VulkanBackend& b) { return b.ring_index_; }
    // Whether a binding names the argument arena of a ring slot other than the one recording.
    static bool foreign_arena(VulkanBackend& b, VkBuffer h) {
        for (uint32_t i = 0; i < VulkanBackend::kRing; ++i)
            if (i != b.ring_index_ && b.arena_[i].buffer && b.arena_[i].buffer->handle() == h) return true;
        return false;
    }
    static void select_wide(VulkanBackend& b, CSlice scores, CSlice ids, const std::vector<uint32_t>& table, size_t groups, uint32_t compression,
                            uint32_t budget, uint32_t stride) {
        b.select_rows(b.bind(scores), b.bind(ids), table, 0, 0, 1, groups, compression, budget, stride, true);
    }
    static BufferPtr dictionary(VulkanBackend& b, uint32_t type) {
        const auto view = b.lookup_grid(type, {});
        const auto& owner = b.lookup_grids_[size_t(weight_kernels(type) - weight_formats)];
        if (!owner || as_vulkan(*owner).handle() != view.buffer)
            throw std::runtime_error("dictionary descriptor differs from its owner");
        return owner;
    }
    static VkPipeline kernel(VulkanBackend& b, KernelId id = K_ADD) { return b.kernel(id).pipeline; }
    static bool query_empty(VulkanBackend& b) { return b.queries_.empty(); }
    static void clear_query(VulkanBackend& b) { b.queries_.clear(); }
    static void prepare_drop(VulkanBackend& b) {
        b.open();
        std::vector<std::shared_ptr<VulkanBuffer>> pending;
        pending.reserve(1);
        b.pending_[b.ring_index_].swap(pending);
    }
    static size_t retained(VulkanBackend& b) { return b.pending_[b.ring_index_].size(); }
    static bool device_clock(VulkanBackend& b) { return b.dev_->device_clock; }
    // Whether the last hold queued ends within `ns`.
    static bool hold_ended(VulkanBackend& b, uint64_t ns, uint64_t value = 0) {
        VkSemaphoreWaitInfo wi{};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &b.hold_.timeline;
        if (!value) value = b.hold_.last;
        wi.pValues = &value;
        return b.dev_->fn.vkWaitSemaphores(b.dev_->device, &wi, ns) == VK_SUCCESS;
    }
    static uint64_t hold_last(VulkanBackend& b) { return b.hold_.last; }
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
    bool capture_module = false;
    std::vector<uint32_t> module_words;
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
    static VKAPI_ATTR VkResult VKAPI_CALL shader(VkDevice, const VkShaderModuleCreateInfo* info, const VkAllocationCallbacks*, VkShaderModule* out) {
        if (kernel_calls->capture_module)
            kernel_calls->module_words.assign(info->pCode, info->pCode + info->codeSize / sizeof(uint32_t));
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
            dev->profile.lookup_integer_dot = row_dot;
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

int dictionary_checks() {
    int failures = 0;
    for (const bool medium_first : {false, true}) {
        backend::VulkanBackend b(0);
        const uint32_t first_type = medium_first ? quant::GGML_TYPE_IQ1_M : quant::GGML_TYPE_IQ1_S;
        const uint32_t second_type = medium_first ? quant::GGML_TYPE_IQ1_S : quant::GGML_TYPE_IQ1_M;
        const auto first = backend::VulkanLifetimeTest::dictionary(b, first_type);
        const auto repeated = backend::VulkanLifetimeTest::dictionary(b, first_type);
        const auto second = backend::VulkanLifetimeTest::dictionary(b, second_type);
        const auto other = backend::VulkanLifetimeTest::dictionary(b, quant::GGML_TYPE_IQ2_XXS);
        const bool reused = first == repeated && first == second &&
                            first == backend::VulkanLifetimeTest::dictionary(b, first_type) &&
                            second == backend::VulkanLifetimeTest::dictionary(b, second_type);
        const bool distinct = other != first && other == backend::VulkanLifetimeTest::dictionary(b, quant::GGML_TYPE_IQ2_XXS);
        std::array<uint32_t, 1024> copied{};
        b.read(*first, 0, copied.data(), sizeof copied);
        const bool intact = first->size() == sizeof(quant::IQ1_GRID) &&
                            std::memcmp(copied.data(), quant::IQ1_GRID, sizeof copied) == 0;
        const bool ok = reused && distinct && intact;
        std::cout << "dictionary first=" << (medium_first ? "IQ1_M" : "IQ1_S")
                  << " shared=" << reused << " distinct_IQ2=" << distinct << " bytes=" << intact
                  << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    return failures;
}

// A hold (Backend::hold_between_submissions) ends within 1 ms at the 99th percentile once the next submission releases it or it is given back, and by the device's clock within 50 to 400 ms if never.
// A release the spinning hold does not see shows as a hold that runs to its bound, and a misread clock rate as one that ends at 10 ms or at a second.
int hold_timing_checks() {
    auto base = backend::make_vulkan_backend(0);
    auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
    if (!backend::VulkanLifetimeTest::device_clock(b)) {
        std::cout << "hold timing skipped, the device's shaders read no device clock\n";
        return 0;
    }
    using Clock = std::chrono::steady_clock;
    const auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    const auto at = [](const std::vector<double>& v, size_t i) { return v.empty() ? 0.0 : v[std::min(i, v.size() - 1)]; };
    b.hold_between_submissions(true);
    std::cout << "hold timing: uncached host memory " << backend::VulkanLifetimeTest::device(b)->uncached_memory << "\n";
    int failures = 0;
    std::vector<double> release;
    for (int i = 0; i < 1000; ++i) {
        b.wait(b.submit());
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        const uint64_t held = backend::VulkanLifetimeTest::hold_last(b);
        const auto t0 = Clock::now();
        b.submit();
        if (!backend::VulkanLifetimeTest::hold_ended(b, 2000000000ull, held)) break;
        release.push_back(ms_since(t0));
    }
    std::sort(release.begin(), release.end());
    const bool released = release.size() == 1000 && at(release, 990) < 1.0;
    std::cout << "hold released by the next submission: " << release.size() << " holds, p50 " << at(release, 500) << " ms, p99 " << at(release, 990) << " ms, max "
              << at(release, 999) << " ms" << (released ? " PASS\n" : " FAIL\n");
    if (!released) ++failures;
    // The hold given back with no work after it.
    std::vector<double> alone;
    for (int i = 0; i < 300; ++i) {
        b.wait(b.submit());
        std::this_thread::sleep_for(std::chrono::microseconds(500));
        const uint64_t held = backend::VulkanLifetimeTest::hold_last(b);
        const auto t0 = Clock::now();
        b.hold_between_submissions(false);
        if (!backend::VulkanLifetimeTest::hold_ended(b, 2000000000ull, held)) break;
        alone.push_back(ms_since(t0));
        b.hold_between_submissions(true);
    }
    std::sort(alone.begin(), alone.end());
    const bool given_back = alone.size() == 300 && at(alone, 296) < 1.0;
    std::cout << "hold given back: " << alone.size() << " holds, p50 " << at(alone, 150) << " ms, p99 " << at(alone, 296) << " ms, max " << at(alone, 299) << " ms"
              << (given_back ? " PASS\n" : " FAIL\n");
    if (!given_back) ++failures;
    for (int i = 0; i < 3; ++i) {
        b.wait(b.submit());
        const auto t1 = Clock::now();
        const bool ended = backend::VulkanLifetimeTest::hold_ended(b, 2000000000ull);
        const double took = ms_since(t1);
        const bool ok = ended && took >= 50 && took <= 400;
        std::cout << "hold never released: ended " << ended << " after " << took << " ms" << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    b.hold_between_submissions(false);
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

// Hold setup (Backend::hold_between_submissions) failing at its timeline (1) or at its release counter's memory (2): what it made is destroyed, and a later call makes the hold whole.
struct HoldCalls;
HoldCalls* hold_calls = nullptr;
struct HoldCalls {
    backend::Fn original;
    std::shared_ptr<backend::Device> device;
    int failure, creates = 0, freed = 0;
    std::vector<uint64_t> live;   // the semaphores and memory made since these calls were set
    HoldCalls(backend::VulkanBackend& b, int fail) : device(backend::VulkanLifetimeTest::device(b)), failure(fail) {
        original = device->fn;
        hold_calls = this;
        auto& fn = device->fn;
        fn.vkCreateSemaphore = semaphore; fn.vkDestroySemaphore = destroy_semaphore;
        fn.vkAllocateMemory = memory; fn.vkFreeMemory = free_memory;
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
    static VKAPI_ATTR VkResult VKAPI_CALL memory(VkDevice d, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks* a, VkDeviceMemory* out) {
        auto& q = *hold_calls;
        if (q.fail_next()) return VK_ERROR_OUT_OF_HOST_MEMORY;
        const VkResult r = q.original.vkAllocateMemory(d, ai, a, out);
        if (r == VK_SUCCESS) q.live.push_back((uint64_t)*out);
        return r;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice d, VkSemaphore s, const VkAllocationCallbacks* a) {
        hold_calls->gone((uint64_t)s);
        hold_calls->original.vkDestroySemaphore(d, s, a);
    }
    static VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice d, VkDeviceMemory m, const VkAllocationCallbacks* a) {
        hold_calls->gone((uint64_t)m);
        hold_calls->original.vkFreeMemory(d, m, a);
    }
    static VKAPI_ATTR void VKAPI_CALL free_cmds(VkDevice d, VkCommandPool p, uint32_t n, const VkCommandBuffer* c) {
        ++hold_calls->freed;
        hold_calls->original.vkFreeCommandBuffers(d, p, n, c);
    }
};

int hold_checks() {
    int failures = 0;
    const std::vector<uint8_t> src = [] { std::vector<uint8_t> v(4096); for (size_t i = 0; i < v.size(); ++i) v[i] = (uint8_t)(i * 7 + 1); return v; }();
    for (int kind = 1; kind <= 2; ++kind) {
        auto base = backend::make_vulkan_backend(0);
        auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
        if (!backend::VulkanLifetimeTest::device_clock(b)) {
            std::cout << "hold_case=" << kind << " skipped, the device's shaders read no device clock\n";
            continue;
        }
        bool threw = false, cleaned = false, copied = false;
        {
            HoldCalls q(b, kind);
            try { b.hold_between_submissions(true); }
            catch (const std::exception&) { threw = true; }
            cleaned = q.live.empty() && q.freed == 1;
            q.failure = 0;
            b.hold_between_submissions(true);
            {
                const auto s = b.adopt(src.data(), src.size());
                const auto d = b.alloc(src.size(), backend::Memory::host_visible);
                b.copy(*d, 0, *s, 0, 2048);
                b.wait(b.submit());
                b.copy(*d, 2048, *s, 2048, 2048);
                b.wait(b.submit());
                copied = std::memcmp(d->host_ptr(), src.data(), src.size()) == 0;
            }
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
// A tensor group's collective (VulkanCollective): its semaphores are freed when a join fails part way, members drain before any goes, and a partial sum failure leaves it reusable without open files.
struct CollectiveCalls;
CollectiveCalls* collective_calls = nullptr;
struct CollectiveCalls {
    std::vector<backend::VulkanBackend*> members;
    std::vector<backend::Fn> original;
    std::vector<PFN_vkGetSemaphoreFdKHR> get_fd;
    std::vector<PFN_vkImportSemaphoreFdKHR> import_fd;
    std::vector<PFN_vkGetMemoryFdKHR> memory_fd;
    int creates = 0, live = 0, fail_create = 0, fail_get = 0, fail_import = 0, imports = 0;
    std::atomic<int> gets{0}, fail_submit{0};   // reached from the members' threads; fail_submit is the member whose next queue call fails, from 1
    std::atomic<int> submit_calls{0};
    int fail_submit_call = 0;
    bool premature = false;
    // Trace only shared inbox/gather payloads. These are actual Vulkan calls, not
    // an assumed barrier count: aliases are paired through their exported fd.
    enum class Op { write, barrier, read };
    struct Event {
        Op op; size_t payload; VkDeviceSize offset, bytes;
        VkPipelineStageFlags src_stage = 0, dst_stage = 0;
        VkAccessFlags src_access = 0, dst_access = 0;
        uint32_t src_family = 0, dst_family = 0;
    };
    struct Allocation { size_t member, payload; VkDeviceMemory memory; };
    struct Shared { size_t member, payload; VkBuffer buffer; };
    struct Command {
        VkCommandBuffer cmd; size_t member; std::vector<Event> events;
        std::vector<VkDescriptorBufferInfo> sources; uint32_t count = 0;   // a dispatch's bindings past the first, which a sum's kernel reads
    };
    struct Submitted { size_t member; std::vector<Event> events; std::vector<size_t> dependencies; };
    struct Signal { size_t member; VkSemaphore semaphore; size_t submission; };
    struct FenceFd { int fd; size_t submission; };
    struct Imported { size_t member; VkSemaphore semaphore; std::vector<size_t> dependencies; };
    bool trace = false;
    size_t payloads = 0;
    std::vector<Allocation> allocations;
    std::vector<Shared> shared;
    std::vector<std::pair<int, size_t>> memory_exports;
    std::vector<Command> commands;
    std::vector<Submitted> submitted;
    std::vector<Signal> signals;
    std::vector<FenceFd> fences;
    std::vector<Imported> imported;
    std::mutex trace_mutex;
    bool bad_wait_stage = false;
    explicit CollectiveCalls(const std::vector<backend::VulkanBackend*>& m, bool ownership = false) : members(m), trace(ownership) {
        if (trace) for (auto* b : members)
            if (!backend::VulkanLifetimeTest::device(*b)->push_descriptor)
                throw std::runtime_error("collective ownership trace requires push descriptors to observe shader peer reads");
        collective_calls = this;
        for (auto* b : members) {
            auto dev = backend::VulkanLifetimeTest::device(*b);
            original.push_back(dev->fn);
            get_fd.push_back(dev->get_semaphore_fd);
            import_fd.push_back(dev->import_semaphore_fd);
            memory_fd.push_back(dev->get_memory_fd);
            dev->fn.vkCreateSemaphore = create;
            dev->fn.vkDestroySemaphore = destroy;
            dev->fn.vkQueueSubmit = submit;
            dev->get_semaphore_fd = get;
            dev->import_semaphore_fd = import;
            if (trace) {
                dev->fn.vkAllocateMemory = allocate;
                dev->fn.vkBindBufferMemory = bind;
                dev->fn.vkDestroyBuffer = destroy_buffer;
                dev->fn.vkFreeMemory = free_memory;
                dev->fn.vkBeginCommandBuffer = begin;
                dev->fn.vkCmdCopyBuffer = copy;
                dev->fn.vkCmdPipelineBarrier = barrier;
                dev->fn.vkCmdPushDescriptorSetKHR = descriptors;
                dev->fn.vkCmdPushConstants = constants;
                dev->fn.vkCmdDispatch = dispatch;
                dev->get_memory_fd = export_memory;
            }
        }
    }
    ~CollectiveCalls() {
        for (size_t i = 0; i < members.size(); ++i) {
            auto dev = backend::VulkanLifetimeTest::device(*members[i]);
            dev->fn = original[i];
            dev->get_semaphore_fd = get_fd[i];
            dev->import_semaphore_fd = import_fd[i];
            dev->get_memory_fd = memory_fd[i];
        }
        collective_calls = nullptr;
    }
    size_t index(VkDevice d) const {
        for (size_t i = 0; i < members.size(); ++i)
            if (backend::VulkanLifetimeTest::device(*members[i])->device == d) return i;
        std::abort();
    }
    Command& command(VkCommandBuffer cmd) {
        for (auto& c : commands) if (c.cmd == cmd) return c;
        for (size_t m = 0; m < members.size(); ++m)
            if (backend::VulkanLifetimeTest::owns(*members[m], cmd)) {
                commands.push_back({cmd, m, {}});
                return commands.back();
            }
        std::abort();
    }
    size_t payload(size_t member, VkBuffer buffer) const {
        for (const auto& b : shared) if (b.member == member && b.buffer == buffer) return b.payload;
        return 0;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL allocate(VkDevice d, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks* a, VkDeviceMemory* out) {
        auto& q = *collective_calls;
        const size_t m = q.index(d);
        size_t id = 0;
        for (auto p = static_cast<const VkBaseInStructure*>(info->pNext); p; p = p->pNext) {
            if (p->sType == VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO) id = ++q.payloads;
            if (p->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR) {
                const int fd = reinterpret_cast<const VkImportMemoryFdInfoKHR*>(p)->fd;
                for (const auto& e : q.memory_exports) if (e.first == fd) id = e.second;
            }
        }
        const VkResult r = q.original[m].vkAllocateMemory(d, info, a, out);
        if (r == VK_SUCCESS && id) q.allocations.push_back({m, id, *out});
        return r;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL bind(VkDevice d, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
        auto& q = *collective_calls;
        const size_t m = q.index(d);
        const VkResult r = q.original[m].vkBindBufferMemory(d, buffer, memory, offset);
        if (r == VK_SUCCESS) for (const auto& a : q.allocations)
            if (a.member == m && a.memory == memory) q.shared.push_back({m, a.payload, buffer});
        return r;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL export_memory(VkDevice d, const VkMemoryGetFdInfoKHR* info, int* fd) {
        auto& q = *collective_calls;
        const size_t m = q.index(d);
        const VkResult r = q.memory_fd[m](d, info, fd);
        if (r == VK_SUCCESS) for (const auto& a : q.allocations)
            if (a.member == m && a.memory == info->memory) q.memory_exports.push_back({*fd, a.payload});
        return r;
    }
    static VKAPI_ATTR void VKAPI_CALL destroy_buffer(VkDevice d, VkBuffer buffer, const VkAllocationCallbacks* a) {
        auto& q = *collective_calls;
        const size_t m = q.index(d);
        q.shared.erase(std::remove_if(q.shared.begin(), q.shared.end(), [&](const Shared& b) { return b.member == m && b.buffer == buffer; }), q.shared.end());
        q.original[m].vkDestroyBuffer(d, buffer, a);
    }
    static VKAPI_ATTR void VKAPI_CALL free_memory(VkDevice d, VkDeviceMemory memory, const VkAllocationCallbacks* a) {
        auto& q = *collective_calls;
        const size_t m = q.index(d);
        q.allocations.erase(std::remove_if(q.allocations.begin(), q.allocations.end(), [&](const Allocation& b) { return b.member == m && b.memory == memory; }), q.allocations.end());
        q.original[m].vkFreeMemory(d, memory, a);
    }
    static VKAPI_ATTR VkResult VKAPI_CALL begin(VkCommandBuffer cmd, const VkCommandBufferBeginInfo* info) {
        auto& q = *collective_calls;
        auto& c = q.command(cmd);
        c.events.clear(); c.sources.clear(); c.count = 0;
        return q.original[c.member].vkBeginCommandBuffer(cmd, info);
    }
    static VKAPI_ATTR void VKAPI_CALL copy(VkCommandBuffer cmd, VkBuffer src, VkBuffer dst, uint32_t n, const VkBufferCopy* regions) {
        auto& q = *collective_calls;
        auto& c = q.command(cmd);
        for (uint32_t i = 0; i < n; ++i) {
            if (const size_t p = q.payload(c.member, src)) c.events.push_back({Op::read, p, regions[i].srcOffset, regions[i].size, 0, VK_PIPELINE_STAGE_TRANSFER_BIT});
            if (const size_t p = q.payload(c.member, dst)) c.events.push_back({Op::write, p, regions[i].dstOffset, regions[i].size});
        }
        q.original[c.member].vkCmdCopyBuffer(cmd, src, dst, n, regions);
    }
    static VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags flags,
            uint32_t nm, const VkMemoryBarrier* mb, uint32_t nb, const VkBufferMemoryBarrier* bb, uint32_t ni, const VkImageMemoryBarrier* ib) {
        auto& q = *collective_calls;
        auto& c = q.command(cmd);
        for (uint32_t i = 0; i < nb; ++i) if (const size_t p = q.payload(c.member, bb[i].buffer))
            c.events.push_back({Op::barrier, p, bb[i].offset, bb[i].size, src, dst,
                               bb[i].srcAccessMask, bb[i].dstAccessMask, bb[i].srcQueueFamilyIndex, bb[i].dstQueueFamilyIndex});
        q.original[c.member].vkCmdPipelineBarrier(cmd, src, dst, flags, nm, mb, nb, bb, ni, ib);
    }
    static VKAPI_ATTR void VKAPI_CALL descriptors(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout,
            uint32_t set, uint32_t n, const VkWriteDescriptorSet* writes) {
        auto& q = *collective_calls;
        auto& c = q.command(cmd);
        c.sources.clear();
        for (uint32_t i = 0; i < n; ++i) if (writes[i].dstBinding >= 1 && writes[i].pBufferInfo) c.sources.push_back(*writes[i].pBufferInfo);
        q.original[c.member].vkCmdPushDescriptorSetKHR(cmd, point, layout, set, n, writes);
    }
    static VKAPI_ATTR void VKAPI_CALL constants(VkCommandBuffer cmd, VkPipelineLayout layout, VkShaderStageFlags stages,
            uint32_t offset, uint32_t bytes, const void* data) {
        auto& q = *collective_calls;
        auto& c = q.command(cmd);
        if (!offset && bytes >= sizeof(c.count)) std::memcpy(&c.count, data, sizeof(c.count));
        q.original[c.member].vkCmdPushConstants(cmd, layout, stages, offset, bytes, data);
    }
    static VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t z) {
        auto& q = *collective_calls;
        auto& c = q.command(cmd);
        for (const VkDescriptorBufferInfo& source : c.sources)
            if (const size_t p = q.payload(c.member, source.buffer))
                c.events.push_back({Op::read, p, source.offset, VkDeviceSize(c.count) * sizeof(float), 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT});
        q.original[c.member].vkCmdDispatch(cmd, x, y, z);
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
    static VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, uint32_t n, const VkSubmitInfo* info, VkFence fence) {
        auto& q = *collective_calls;
        size_t i = 0;
        while (backend::VulkanLifetimeTest::device(*q.members[i])->queue != queue) ++i;
        if (++q.submit_calls == q.fail_submit_call) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        int armed = (int)i + 1;
        if (q.fail_submit.compare_exchange_strong(armed, 0)) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        const VkResult r = q.original[i].vkQueueSubmit(queue, n, info, fence);
        if (q.trace && r == VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(q.trace_mutex);
            for (uint32_t j = 0; j < n; ++j) {
                Submitted s{i, {}, {}};
                for (size_t k = q.submitted.size(); k-- > 0;) if (q.submitted[k].member == i) { s.dependencies.push_back(k); break; }
                for (uint32_t k = 0; k < info[j].waitSemaphoreCount; ++k)
                    for (size_t w = q.imported.size(); w-- > 0;) if (q.imported[w].member == i && q.imported[w].semaphore == info[j].pWaitSemaphores[k]) {
                        if (!(info[j].pWaitDstStageMask[k] & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)) q.bad_wait_stage = true;
                        s.dependencies.insert(s.dependencies.end(), q.imported[w].dependencies.begin(), q.imported[w].dependencies.end());
                        break;
                    }
                for (uint32_t k = 0; k < info[j].commandBufferCount; ++k) {
                    const auto& c = q.command(info[j].pCommandBuffers[k]);
                    s.events.insert(s.events.end(), c.events.begin(), c.events.end());
                }
                const size_t id = q.submitted.size();
                q.submitted.push_back(std::move(s));
                for (uint32_t k = 0; k < info[j].signalSemaphoreCount; ++k) q.signals.push_back({i, info[j].pSignalSemaphores[k], id});
            }
        }
        return r;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL get(VkDevice d, const VkSemaphoreGetFdInfoKHR* info, int* fd) {
        auto& q = *collective_calls;
        if (++q.gets == q.fail_get) return VK_ERROR_TOO_MANY_OBJECTS;
        const VkResult r = q.get_fd[q.index(d)](d, info, fd);
        if (q.trace && r == VK_SUCCESS) {
            std::lock_guard<std::mutex> lock(q.trace_mutex);
            for (size_t i = q.signals.size(); i-- > 0;) if (q.signals[i].member == q.index(d) && q.signals[i].semaphore == info->semaphore) {
                q.fences.push_back({*fd, q.signals[i].submission});
                break;
            }
        }
        return r;
    }
    static VKAPI_ATTR VkResult VKAPI_CALL import(VkDevice d, const VkImportSemaphoreFdInfoKHR* info) {
        auto& q = *collective_calls;
        if (++q.imports == q.fail_import) return VK_ERROR_INVALID_EXTERNAL_HANDLE;
        const VkResult r = q.import_fd[q.index(d)](d, info);
        if (q.trace && r == VK_SUCCESS) {
            Imported w{q.index(d), info->semaphore, {}};
            // -1 denotes already-completed work, so every captured -1 source
            // really has completed before this import; positive fds are unique
            // until import consumes them, and may then be reused by the OS.
            if (info->fd == -1) {
                for (const auto& f : q.fences) if (f.fd == -1) w.dependencies.push_back(f.submission);
            } else {
                for (size_t i = q.fences.size(); i-- > 0;) if (q.fences[i].fd == info->fd) {
                    w.dependencies.push_back(q.fences[i].submission); break;
                }
            }
            q.imported.push_back(std::move(w));
        }
        return r;
    }
    bool ordered(size_t before, size_t after) const {
        if (before == after) return true;
        std::vector<size_t> todo{after};
        std::vector<bool> seen(submitted.size(), false);
        while (!todo.empty()) {
            const size_t at = todo.back(); todo.pop_back();
            if (seen[at]) continue;
            seen[at] = true;
            for (size_t p : submitted[at].dependencies) {
                if (p == before) return true;
                todo.push_back(p);
            }
        }
        return false;
    }
    bool ownership_ok(bool quiet = false) const {
        struct State {
            Event write{Op::write, 0, 0, 0}, acquire{Op::barrier, 0, 0, 0};
            size_t writer = SIZE_MAX, release = SIZE_MAX, reader = SIZE_MAX, acquired = SIZE_MAX;
        };
        std::vector<State> state(payloads + 1);
        size_t reads = 0;
        auto fail = [quiet](const char* why, size_t payload, size_t submission) {
            if (!quiet) std::cout << "collective_ownership payload=" << payload << " submission=" << submission << " " << why << " FAIL\n";
            return false;
        };
        if (bad_wait_stage) return fail("peer wait does not cover ownership acquire", 0, 0);
        auto stages_cover = [](VkPipelineStageFlags mask, VkPipelineStageFlags stage) {
            return (mask & VK_PIPELINE_STAGE_ALL_COMMANDS_BIT) || (mask & stage);
        };
        for (size_t s = 0; s < submitted.size(); ++s) for (const auto& e : submitted[s].events) {
            auto& p = state[e.payload];
            const uint32_t local = backend::VulkanLifetimeTest::device(*members[submitted[s].member])->queue_family;
            if (e.op == Op::write) {
                if (p.reader != SIZE_MAX && !ordered(p.reader, s)) return fail("overwrite precedes prior consumer", e.payload, s);
                p.write = e; p.writer = s; p.release = p.acquired = SIZE_MAX;
            } else if (e.op == Op::barrier && e.src_family == local && e.dst_family == VK_QUEUE_FAMILY_FOREIGN_EXT) {
                if (p.writer == SIZE_MAX || !ordered(p.writer, s) || e.offset != p.write.offset || e.bytes != p.write.bytes ||
                    submitted[p.writer].member != submitted[s].member || !stages_cover(e.src_stage, VK_PIPELINE_STAGE_TRANSFER_BIT) ||
                    !(e.src_access & (VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT)))
                    return fail("release does not cover preceding producer copy", e.payload, s);
                p.release = s;
            } else if (e.op == Op::barrier && e.src_family == VK_QUEUE_FAMILY_FOREIGN_EXT && e.dst_family == local) {
                if (p.release == SIZE_MAX || !ordered(p.release, s) || e.offset != p.write.offset || e.bytes != p.write.bytes ||
                    submitted[p.release].member == submitted[s].member)
                    return fail("acquire lacks matching release and semaphore dependency", e.payload, s);
                p.acquire = e; p.acquired = s;
                p.release = SIZE_MAX; // One preserving release permits one acquire, followed by any reads.
            } else if (e.op == Op::read) {
                const VkAccessFlags access = e.dst_stage == VK_PIPELINE_STAGE_TRANSFER_BIT ? VK_ACCESS_TRANSFER_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
                if (p.acquired == SIZE_MAX || !ordered(p.acquired, s) || e.offset != p.acquire.offset || e.bytes != p.acquire.bytes ||
                    submitted[p.acquired].member != submitted[s].member || !stages_cover(p.acquire.dst_stage, e.dst_stage) ||
                    !(p.acquire.dst_access & (access | VK_ACCESS_MEMORY_READ_BIT)))
                    return fail("consumer read lacks preceding matching acquire", e.payload, s);
                p.reader = s; ++reads;
            } else return fail("wrong ownership barrier direction or family", e.payload, s);
        }
        if (!quiet) std::cout << "collective_ownership payloads=" << payloads << " peer_reads=" << reads << (reads ? " PASS\n" : " FAIL\n");
        return reads != 0;
    }
    // Mutate the actual successful trace in memory; no additional Vulkan calls.
    bool ownership_mutations() {
        size_t rs = SIZE_MAX, re = 0, as = SIZE_MAX, ae = 0, read = SIZE_MAX;
        for (size_t s = 0; s < submitted.size(); ++s) for (size_t e = 0; e < submitted[s].events.size(); ++e) {
            const auto& event = submitted[s].events[e];
            if (rs == SIZE_MAX && event.op == Op::barrier && event.dst_family == VK_QUEUE_FAMILY_FOREIGN_EXT) { rs = s; re = e; }
            if (rs != SIZE_MAX && as == SIZE_MAX && event.op == Op::barrier && event.src_family == VK_QUEUE_FAMILY_FOREIGN_EXT &&
                event.payload == submitted[rs].events[re].payload) { as = s; ae = e; }
            if (as == s && e > ae && event.op == Op::read && event.payload == submitted[as].events[ae].payload && read == SIZE_MAX) read = e;
        }
        if (rs == SIZE_MAX || as == SIZE_MAX || read == SIZE_MAX) {
            std::cout << "collective_ownership_negative missing handoff/read witness FAIL\n";
            return false;
        }
        size_t rejected = 0;
        auto check = [&](const char* name, auto mutate) {
            const auto saved = submitted;
            mutate();
            if (!ownership_ok(true)) ++rejected;
            else std::cout << "collective_ownership_negative unexpectedly accepted " << name << '\n';
            submitted = saved;
        };
        check("missing release", [&] { submitted[rs].events.erase(submitted[rs].events.begin() + re); });
        check("reversed direction", [&] { auto& e = submitted[rs].events[re]; std::swap(e.src_family, e.dst_family); });
        check("wrong size", [&] { submitted[as].events[ae].bytes += sizeof(float); });
        check("acquire after read", [&] {
            auto& events = submitted[as].events;
            const auto acquire = events[ae];
            events.erase(events.begin() + ae);
            events.insert(events.begin() + read, acquire); // The original read shifted left by one.
        });
        check("duplicate acquire", [&] {
            auto& events = submitted[as].events;
            const auto acquire = events[ae];
            events.insert(events.begin() + ae + 1, acquire);
        });
        const bool ok = rejected == 5;
        std::cout << "collective_ownership_negative cases=5 rejected=" << rejected << (ok ? " PASS\n" : " FAIL\n");
        return ok;
    }
};

// The files this process holds open, where the system lists them.
size_t open_files() {
    size_t n = 0;
#if defined(__linux__)
    for (const auto& e : std::filesystem::directory_iterator("/proc/self/fd")) { (void)e; ++n; }
#endif
    return n;
}

// A backend whose queue a submission holds (VulkanBackend::prepare) refuses to record or submit until the queue is taken back, and works as before after.
int lent_checks() {
    auto base = backend::make_vulkan_backend(0);
    auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
    const size_t n = 256;
    std::vector<uint8_t> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = (uint8_t)(i * 7 + 3);
    const auto a = b.adopt(src.data(), n);
    const auto out = b.alloc(n, backend::Memory::host_visible);
    b.sync();
    b.copy(*out, 0, *a, 0, n);
    const auto held = b.prepare({});
    bool recorded = true, submitted = true;
    try { b.copy(*out, 0, *a, 0, n); recorded = false; } catch (const std::logic_error&) {}
    try { b.submit(); submitted = false; } catch (const std::logic_error&) {}
    b.wait(b.queued(held, held.queue(*b.device())));
    const bool arrived = std::memcmp(out->host_ptr(), src.data(), n) == 0;
    std::vector<uint8_t> next(n);
    for (size_t i = 0; i < n; ++i) next[i] = (uint8_t)~src[i];
    const auto after = b.adopt(next.data(), n);
    b.copy(*out, 0, *after, 0, n);
    b.sync();
    const bool ok = recorded && submitted && arrived && std::memcmp(out->host_ptr(), next.data(), n) == 0;
    std::cout << "lent_queue refused_record=" << recorded << " refused_submit=" << submitted << " arrived=" << arrived << (ok ? " PASS\n" : " FAIL\n");
    return ok ? 0 : 1;
}

// A dispatch reads only its own slot's argument arena, which the slot resets when it reopens after waiting for its own submission alone.
// Pushed sets are counted by the slot recording them, and every binding naming another slot's arena counts as foreign.
struct ArgsCalls;
ArgsCalls* args_calls = nullptr;
struct ArgsCalls {
    backend::VulkanBackend& b;
    backend::Fn original;
    std::shared_ptr<backend::Device> device;
    std::vector<uint32_t> slots;
    size_t foreign = 0;
    explicit ArgsCalls(backend::VulkanBackend& be) : b(be), device(backend::VulkanLifetimeTest::device(be)) {
        original = device->fn;
        args_calls = this;
        device->fn.vkCmdPushDescriptorSetKHR = push;
    }
    ~ArgsCalls() {
        device->fn = original;
        args_calls = nullptr;
    }
    static VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer cmd, VkPipelineBindPoint point, VkPipelineLayout layout, uint32_t set, uint32_t count,
                                           const VkWriteDescriptorSet* writes) {
        auto& a = *args_calls;
        a.slots.push_back(backend::VulkanLifetimeTest::slot(a.b));
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t j = 0; j < writes[i].descriptorCount; ++j)
                a.foreign += backend::VulkanLifetimeTest::foreign_arena(a.b, writes[i].pBufferInfo[j].buffer);
        a.original.vkCmdPushDescriptorSetKHR(cmd, point, layout, set, count, writes);
    }
};

// An op whose dispatches read one table: after a sync, k single adds put the chunk flush between two of its dispatches for some k, and no dispatch may then bind the table in the slot before it.
// Attention's merge after its main dispatch (a decode row whose 600-token history splits) and select_rows' five wide stages, each output the bits of the run without adds.
int args_slot_checks() {
    backend::VulkanBackend b(0);
    backend::Backend& base = b;
    const auto zero = std::vector<float>(256, 0.0f);
    const auto filler = b.adopt(zero.data(), zero.size() * sizeof(float));
    auto pattern = [](size_t n, uint32_t seed) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = float(std::sin(double(i * 2654435761u % 1000003u + seed)));
        return v;
    };
    // Runs `op` after k adds for k from 0 to 15, the first run without adds giving the reference.
    auto forced = [&](const char* what, size_t dispatches, const std::function<std::vector<uint8_t>()>& op) {
        std::vector<uint8_t> want;
        bool straddled = false, same = true;
        size_t foreign = 0;
        for (int k = -1; k < 16; ++k) {
            b.sync();
            std::vector<uint8_t> got;
            {
                ArgsCalls calls(b);
                for (int i = 0; i < k; ++i) base.add({filler.get(), 0}, {filler.get(), 0}, zero.size());
                const size_t first = calls.slots.size();
                got = op();
                same = same && calls.slots.size() - first == dispatches;
                for (size_t i = first + 1; i < calls.slots.size(); ++i) straddled = straddled || calls.slots[i] != calls.slots[first];
                foreign += calls.foreign;
            }
            if (k < 0) want = got;
            else same = same && got == want;
        }
        const bool ok = straddled && same && !foreign;
        std::cout << "args_slot " << what << " straddled=" << straddled << " same=" << same << " foreign=" << foreign << (ok ? " PASS\n" : " FAIL\n");
        return ok ? 0 : 1;
    };
    int failures = 0;
    {
        const int n_head = 4, n_head_kv = 2, head_dim = 128;
        const size_t hist = 600, qw = size_t(n_head) * head_dim, kvw = size_t(n_head_kv) * head_dim;
        const size_t bt = b.kv_layout().block_tokens, n_blocks = (hist + 1 + bt - 1) / bt;
        auto st = b.kv_alloc(1, n_head_kv, head_dim, n_blocks * bt);
        std::vector<int32_t> blocks(n_blocks);
        for (size_t i = 0; i < n_blocks; ++i) blocks[i] = int32_t(i);
        const auto kh = pattern((hist + 1) * kvw, 1), vh = pattern((hist + 1) * kvw, 2), q = pattern(qw, 3);
        const auto Kb = b.adopt(kh.data(), kh.size() * sizeof(float));
        const auto Vb = b.adopt(vh.data(), vh.size() * sizeof(float));
        const auto Qb = b.adopt(q.data(), q.size() * sizeof(float));
        const auto ob = b.alloc(qw * sizeof(float), backend::Memory::device);
        const backend::KVView all{st.get(), blocks.data(), n_blocks, 0, hist + 1};
        base.kv_write(0, &all, 1, {Kb.get(), 0}, {Vb.get(), 0});
        const backend::KVView row{st.get(), blocks.data(), n_blocks, hist, 1};
        failures += forced("attention_merge", 2, [&] {
            base.attention({Qb.get(), 0}, 0, &row, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
            std::vector<uint8_t> out(qw * sizeof(float));
            base.read(*ob, 0, out.data(), out.size());
            return out;
        });
    }
    {
        const size_t groups = 4096;
        const uint32_t compression = 4, budget = 2048, stride = budget + compression - 1;
        const auto scores = pattern(groups, 4);
        const auto sb = b.adopt(scores.data(), scores.size() * sizeof(float));
        const auto ib = b.alloc(stride * sizeof(uint32_t), backend::Memory::device);
        const std::vector<uint32_t> table = {1u, 0u, 0u, 1u, uint32_t(groups * compression - 1), 0u, 0u};
        failures += forced("select_wide", 5, [&] {
            backend::VulkanLifetimeTest::select_wide(b, {sb.get(), 0}, {ib.get(), 0}, table, groups, compression, budget, stride);
            std::vector<uint8_t> out(stride * sizeof(uint32_t));
            base.read(*ib, 0, out.data(), out.size());
            return out;
        });
    }
    return failures;
}

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
    auto chain = [&](const std::vector<backend::Backend*>& active_members,
                     std::unique_ptr<backend::Collective>& sum, size_t sums, size_t active_rows = 0, size_t active_width = 0,
                     std::initializer_list<size_t> row_counts = {}) {
        if (!active_rows) active_rows = rows;
        if (!active_width) active_width = width;
        if (row_counts.size() && row_counts.size() != sums) throw std::logic_error("collective chain row counts differ from sum count");
        for (size_t count : row_counts) if (count > active_rows) throw std::logic_error("collective chain exceeds reserved rows");
        const size_t n = active_rows * active_width;
        std::vector<float> want(n);
        for (float& v : want) v = value();
        std::vector<backend::BufferPtr> x;
        std::vector<backend::Slice> xs;
        for (size_t m = 0; m < active_members.size(); ++m) {
            x.push_back(active_members[m]->alloc(n * sizeof(float)));
            active_members[m]->write(*x.back(), 0, want.data(), n * sizeof(float));
            xs.push_back({x.back().get(), 0});
        }
        for (size_t k = 0; k < sums; ++k) {
            const size_t sum_rows = row_counts.size() ? *(row_counts.begin() + k) : active_rows;
            std::vector<std::vector<float>> partial(active_members.size(), std::vector<float>(n));
            for (size_t m = 0; m < active_members.size(); ++m) {
                for (float& v : partial[m]) v = value();
                const backend::Slice p = sum->partial(m);
                active_members[m]->write(*p.buffer, p.offset * sizeof(float), partial[m].data(), n * sizeof(float));
            }
            sum->sum_into(xs, sum_rows, active_width);
            for (size_t i = 0; i < sum_rows * active_width; ++i) {
                float p = partial[0][i];
                for (size_t m = 1; m < active_members.size(); ++m) p = p + partial[m][i];
                want[i] = want[i] + p;
            }
        }
        sum.reset();
        bool same = true;
        for (size_t m = 0; m < active_members.size(); ++m) {
            std::vector<float> got(n);
            active_members[m]->read(*x[m], 0, got.data(), n * sizeof(float));
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
            summed = chain(members, sum, 3) && q.live == 0 && !q.premature;
        }
        const bool ok = threw && clean && summed;
        std::cout << "collective_join_case=" << kind << " threw=" << threw << " clean=" << clean << " summed=" << summed << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    {
        CollectiveCalls q(vk, true);
        auto sum = first->join(members, rows, width);
        const bool same = chain(members, sum, 7);
        const bool ownership = q.ownership_ok();
        const bool negatives = ownership && q.ownership_mutations();
        const bool ok = same && ownership && negatives && q.live == 0 && !q.premature;
        std::cout << "collective_in_flight sums=7" << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    }
    {
        // This checks refusal at the existing capability boundary, not the
        // extension scan or vkCreateDevice's enabled-extension list.
        auto dev = backend::VulkanLifetimeTest::device(*vk[1]);
        const bool exchange = dev->exchange;
        dev->exchange = false;
        CollectiveCalls q(vk, true);
        const bool refused = !first->join(members, rows, width);
        dev->exchange = exchange;
        const bool ok = refused && q.payloads == 0 && q.live == 0;
        std::cout << "collective_capability_refused before_shared_allocation=" << ok << (ok ? " PASS\n" : " FAIL\n");
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
            const bool ok = named && chain(members, sum, 3) && q.live == 0 && !q.premature;
            std::cout << "collective_import_refused named=" << named << (ok ? " PASS\n" : " FAIL\n");
            if (!ok) ++failures;
        }
    }
    // Probe each additional width once, then reuse the same checks for all shareable groups.
    backend::BackendPtr third, fourth;
    bool width_three = false, width_four = false;
    try {
        third = backend::make_vulkan_backend(2);
        auto candidate = members;
        candidate.push_back(third.get());
        if (first->join(candidate, 1, 1)) {
            members = std::move(candidate);
            vk.push_back(&dynamic_cast<backend::VulkanBackend&>(*third));
            width_three = true;
        }
    } catch (const std::exception&) {}
    if (width_three) {
        try {
            fourth = backend::make_vulkan_backend(3);
            auto candidate = members;
            candidate.push_back(fourth.get());
            if (first->join(candidate, 1, 1)) {
                members = std::move(candidate);
                vk.push_back(&dynamic_cast<backend::VulkanBackend&>(*fourth));
                width_four = true;
            }
        } catch (const std::exception&) {}
    }
    if (!width_three) std::cout << "collective_width3 skipped: no third shareable device\n";
    if (!width_four) std::cout << "collective_width4 skipped: no fourth shareable device\n";
    if (!width_three) std::cout << "collective: two-shot ownership and second-exchange failure checks skipped\n";

    auto member_width_checks = [&](size_t count) {
        std::vector<backend::Backend*> group(members.begin(), members.begin() + count);
        std::vector<backend::VulkanBackend*> group_vk(vk.begin(), vk.begin() + count);
        const int W = (int)group.size();
        if (W >= 3) {
            // Both inbox and gather handoffs, with each parity reused. 4097*80 is
            // just above the production two-shot crossover (327680 floats).
            const size_t large_rows = 80, large_width = 4097;
            {
                CollectiveCalls q(group_vk, true);
                auto sum = first->join(group, large_rows, large_width);
                const bool same = chain(group, sum, 5, large_rows, large_width);
                const bool ownership = q.ownership_ok();
                const bool negatives = ownership && q.ownership_mutations();
                const bool ok = same && ownership && negatives && q.live == 0 && !q.premature;
                std::cout << "collective_two_shot_ownership members=" << W << " sums=5" << (ok ? " PASS\n" : " FAIL\n");
                if (!ok) ++failures;
            }
            {
                // Reuse the same collective while both parities change between
                // one and two shots. Full residual tails must remain unchanged.
                CollectiveCalls q(group_vk, true);
                auto sum = first->join(group, large_rows, large_width);
                const bool same = chain(group, sum, 7, large_rows, large_width, {80, 1, 1, 80, 1, 1, 80});
                const bool ownership = q.ownership_ok();
                const bool negatives = ownership && q.ownership_mutations();
                const bool ok = same && ownership && negatives && q.live == 0 && !q.premature;
                std::cout << "collective_mixed_shots_ownership members=" << W << " rows=80,1,1,80,1,1,80 unchanged_tails=" << same << (ok ? " PASS\n" : " FAIL\n");
                if (!ok) ++failures;
            }
            for (int kind = 0; kind < 3; ++kind) {
                const size_t files = open_files();
                CollectiveCalls q(group_vk);
                auto sum = first->join(group, large_rows, large_width);
                const int live = q.live;
                if (kind == 0) q.fail_get = W * (W - 1) + 1;
                if (kind == 1) q.fail_import = W * (W - 1) + 1;
                if (kind == 2) q.fail_submit_call = W + 1;
                std::vector<backend::BufferPtr> x;
                std::vector<backend::Slice> xs;
                for (auto* b : group) {
                    x.push_back(b->alloc(large_rows * large_width * sizeof(float)));
                    xs.push_back({x.back().get(), 0});
                }
                bool threw = false;
                try { sum->sum_into(xs, large_rows, large_width); }
                catch (const std::exception&) { threw = true; }
                bool drained = q.live == live && !q.premature;
                for (auto* b : group_vk) drained = drained && backend::VulkanLifetimeTest::idle(*b);
                q.fail_get = q.fail_import = q.fail_submit_call = 0;
                const bool same = chain(group, sum, 3, large_rows, large_width);
                const bool ok = threw && drained && same && q.live == 0 && !q.premature && open_files() == files;
                std::cout << "collective_second_exchange_failure members=" << W << " kind=" << kind << " drained=" << drained << " recovered=" << same << (ok ? " PASS\n" : " FAIL\n");
                if (!ok) ++failures;
            }
        }
        for (int kind = 1; kind <= 7; ++kind) {
            bool threw = false, drained = false, summed = false;
            const size_t files = open_files();
            {
                CollectiveCalls q(group_vk);
                auto sum = first->join(group, rows, width);
                const int live = q.live;
                if (kind <= 2) q.fail_get = kind; else if (kind <= 4) q.fail_import = kind - 2; else q.fail_submit = kind == 5 ? 1 : W;
                std::vector<backend::BufferPtr> x;
                std::vector<backend::Slice> xs;
                for (size_t m = 0; m < group.size(); ++m) {
                    x.push_back(group[m]->alloc(n * sizeof(float)));
                    xs.push_back({x.back().get(), 0});
                }
                try { sum->sum_into(xs, rows, width); }
                catch (const std::exception&) { threw = true; }
                drained = q.live == live && !q.premature && q.fail_submit == 0;
                for (auto* b : group_vk) drained = drained && backend::VulkanLifetimeTest::idle(*b);
                q.fail_get = q.fail_import = 0;
                if (kind == 7) sum.reset();
                summed = (kind == 7 || chain(group, sum, 3)) && q.live == 0 && !q.premature;
            }
            const bool closed = open_files() == files;
            const bool ok = threw && drained && summed && closed;
            std::cout << "collective_sum_case=" << kind << " members=" << W << " threw=" << threw << " drained=" << drained << " summed=" << summed << " closed=" << closed << (ok ? " PASS\n" : " FAIL\n");
            if (!ok) ++failures;
        }
    };
    member_width_checks(width_three ? 3 : 2);
    if (width_four) member_width_checks(4);
    return failures;
}
}

namespace {
enum class Failure { none, create, memory_type, allocate_device, allocate_host, bind, map };
struct Calls {
    Failure failure = Failure::none;
    int buffers = 0, memory = 0, maps = 0;
    bool bad_release = false;
    VkDeviceSize allocation_bytes = 64;
    uint32_t memory_types = 1;
    VkDeviceSize requested_bytes = 0;
    uint32_t requested_type = 0;
    bool fail_reporting_allocation = false;
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
    *out = VkMemoryRequirements{calls.allocation_bytes, 4, calls.memory_types};
}
VKAPI_ATTR VkResult VKAPI_CALL allocate_memory(VkDevice, const VkMemoryAllocateInfo* info, const VkAllocationCallbacks*, VkDeviceMemory* out) {
    calls.requested_bytes = info->allocationSize;
    calls.requested_type = info->memoryTypeIndex;
    if (calls.fail_reporting_allocation) allocation_countdown = 1;
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

int allocation_error_checks() {
    int failures = 0;
    for (const Failure failure : {Failure::allocate_device, Failure::allocate_host}) {
        for (const bool host_visible : {false, true}) {
            calls = Calls{};
            calls.failure = failure;
            calls.fail_reporting_allocation = true;
            calls.allocation_bytes = host_visible ? 4294967360ull : 64;
            calls.memory_types = host_visible ? 4 : 1;
            const uint32_t memory_type = host_visible ? 2 : 0;
            auto dev = fake_device(failure);
            dev->caps.device = "test allocation GPU";
            dev->memory.memoryTypes[memory_type] = dev->memory.memoryTypes[0];
            dev->memory.memoryTypeCount = memory_type + 1;
            const char* result = failure == Failure::allocate_device ? "VK_ERROR_OUT_OF_DEVICE_MEMORY" : "VK_ERROR_OUT_OF_HOST_MEMORY";
            const std::string expected = std::string("vulkan: test allocation GPU: vkAllocateMemory failed with ") + result +
                " for " + std::to_string(calls.allocation_bytes) + " bytes (memory type " + std::to_string(memory_type) + ")";
            bool caught = false, no_allocation = false, matches = false;
            std::string message;
            try {
                backend::VulkanBuffer buffer(dev, 17, host_visible);
            } catch (const std::bad_alloc& error) {
                caught = true;
                matches = std::strcmp(error.what(), expected.c_str()) == 0;
                no_allocation = allocation_countdown == 1;
                allocation_countdown = -1;
                message = error.what();
            } catch (...) {
                allocation_countdown = -1;
            }
            allocation_countdown = -1;
            const bool ok = caught && no_allocation && matches && calls.requested_bytes == calls.allocation_bytes &&
                calls.requested_type == memory_type && !calls.buffers && !calls.memory && !calls.maps && !calls.bad_release;
            std::cout << "allocation error " << result << " host_visible=" << host_visible << ": " << message << (ok ? " PASS\n" : " FAIL\n");
            failures += !ok;
        }
    }
    return failures;
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
    // The fused routed step's routing is its own need: a device whose subgroups are not 32 or 64 lanes opens and takes the unfused step, the need named.
    const auto counter = [&](const char* what, const std::string& need) {
        const std::string got = backend::missing_counter_need(sg.subgroupSize);
        const bool ok = (need.empty() ? got.empty() : got.find(need) != std::string::npos) && backend::missing_device_need(sg, f2, f11, f12).empty();
        std::cout << "counter_need " << what << ": \"" << got << "\"" << (ok ? " PASS\n" : " FAIL\n");
        if (!ok) ++failures;
    };
    sg.subgroupSize = 64; counter("64 lanes", "");
    sg.subgroupSize = 32; counter("32 lanes", "");
    sg.subgroupSize = 128; counter("128 lanes", "32 or 64 lanes");
    sg.subgroupSize = 64;
    // Heads 128 and 256 wide take kernels whose lane groups fit any subgroup the device check accepts, 256 on a 32-lane device included.
    // Other widths take the per-row kernel, which reads four elements of a head per lane of a subgroup, so such a head wider than four subgroups does not fit.
    const bool widths = backend::attention_head_fits(128, 32) && backend::attention_head_fits(256, 32) && !backend::attention_head_fits(192, 32) &&
                        backend::attention_head_fits(192, 64) && backend::attention_head_fits(256, 64) && !backend::attention_head_fits(260, 64) &&
                        !backend::attention_head_fits(512, 64) && backend::attention_head_fits(64, 64) && backend::attention_head_fits(80, 32);
    std::cout << "attention head widths against 32- and 64-lane subgroups" << (widths ? " PASS\n" : " FAIL\n");
    // The integer-dot kernels (the Q8_0 decode kernel and the integer-dot tile) run only under a profile that prefers the integer dot, which a device without the extension never gets, even where its row asks for it.
    backend::DeviceCaps caps;
    caps.device = "AMD Radeon Instinct MI60 / MI50 (RADV VEGA20)";
    caps.driver = "radv Mesa";
    caps.integer_dot = true;
    const backend::DeviceProfile with = backend::profile_for(caps);
    caps.integer_dot = false;
    const backend::DeviceProfile without = backend::profile_for(caps);
    const bool gate = with.prefer_integer_dot && with.lookup_integer_dot && !without.prefer_integer_dot && !without.lookup_integer_dot;
    std::cout << "integer-dot kernels only with the integer dot product" << (gate ? " PASS\n" : " FAIL\n");
    return failures + !widths + !gate;
}
}

// A two-shot sum's shares (VulkanCollective): they cover the floats once and in order, and each that holds a float begins on the alignment, where a device takes a storage-buffer offset.
// At an alignment of one float they are the equal shares, as on a device whose alignment is four bytes.
int share_checks() {
    int failures = 0;
    for (const size_t members : {size_t(3), size_t(4), size_t(5)})
        for (const size_t n : {size_t(327680), size_t(327681), size_t(4097) * 96, size_t(2), size_t(1) << 20})
            for (const size_t align : {size_t(1), size_t(4), size_t(16), size_t(64)}) {
                const size_t even = (n + members - 1) / members, most = (even + align - 1) / align * align;
                bool ok = backend::share_begin(0, n, members, align) == 0 && backend::share_begin(members, n, members, align) == n;
                for (size_t k = 0; k < members; ++k) {
                    const size_t lo = backend::share_begin(k, n, members, align), hi = backend::share_begin(k + 1, n, members, align);
                    ok = ok && lo <= hi && hi - lo <= most && (lo == hi || lo % align == 0) && (align > 1 || lo == std::min(n, k * even));
                }
                if (!ok) {
                    std::cout << "a two-shot sum's shares of " << n << " floats over " << members << " members at an alignment of " << align << " floats FAIL" << std::endl;
                    ++failures;
                }
            }
    return failures;
}

// A table's views (table_views): they cover its rows once and in order, each within one binding's range and 32-bit indices, beginning on the alignment with its first row at the skew.
// Among them the 26.8 GiB per-layer table of Qwen3.8-Flash-Next, IQ4_NL rows 160 wide, whose row offsets pass 2^32 bytes; a table that fits one binding keeps one view.
int table_view_checks() {
    struct Case {
        VkDeviceSize base;
        size_t rows, row_bytes;
        VkDeviceSize max_range, align;
        bool one;
    };
    const Case cases[] = {{0, 320001536, 90, 0xffffffffull, 16, false},  {4096, 320001536, 90, 0x7fffffffull, 256, false},
                          {0, 70000000, 64, 0xffffffffull, 4, false},    {0, 248320, 2720, 0xffffffffull, 4, true},
                          {12, 1000, 64, 0xffffffffull, 64, true},       {0, 1, 4096, 0xffffffffull, 256, true}};
    int failures = 0;
    for (const Case& c : cases) {
        const std::vector<backend::TableView> views = backend::table_views(c.base, c.rows, c.row_bytes, c.max_range, c.align);
        bool ok = !views.empty() && views.front().first == 0 && views.back().end == c.rows && (views.size() == 1) == c.one;
        uint32_t next = 0;
        for (const backend::TableView& v : views) {
            const VkDeviceSize at = c.base + VkDeviceSize(v.first) * c.row_bytes;
            ok = ok && v.first == next && v.first < v.end && v.offset % c.align == 0 && v.skew < c.align && (c.row_bytes % 4 || v.skew % 4 == 0) && v.offset + v.skew == at &&
                 v.range == v.skew + VkDeviceSize(v.end - v.first) * c.row_bytes && v.range <= c.max_range && v.range <= 0xffffffffull;
            next = v.end;
        }
        std::cout << "table views of " << c.rows << " rows of " << c.row_bytes << " bytes from " << c.base << " at a range of " << c.max_range << ": "
                  << views.size() << (ok ? " PASS\n" : " FAIL\n");
        failures += !ok;
    }
    return failures;
}

bool check_float_modes(const char* name, const uint32_t* words, size_t count, uint32_t wanted) {
    unsigned denorm32 = 0, special32 = 0, special64 = 0, denorm16 = 0, special16 = 0, rte16 = 0, rte32 = 0, other = 0;
    bool denorm_cap = false, special_cap = false, rte_cap = false;
    if (count < 5 || words[0] != 0x07230203u) throw std::runtime_error("float controls: invalid SPIR-V header");
    for (size_t at = 5; at < count;) {
        const uint32_t length = words[at] >> 16, opcode = words[at] & 0xffffu;
        if (!length || length > count - at) throw std::runtime_error("float controls: invalid SPIR-V instruction");
        if (opcode == 17u && length == 2u) {
            denorm_cap |= words[at + 1] == 4464u;
            special_cap |= words[at + 1] == 4466u;
            rte_cap |= words[at + 1] == 4467u;
        }
        if (opcode == 16u && length >= 3u && words[at + 2] >= 4459u && words[at + 2] <= 4463u) {
            const uint32_t mode = words[at + 2], width = length == 4 ? words[at + 3] : 0;
            if (mode == 4459u && width == 32u) ++denorm32;
            else if (mode == 4461u && width == 32u) ++special32;
            else if (mode == 4461u && width == 64u) ++special64;
            else if (mode == 4459u && width == 16u) ++denorm16;
            else if (mode == 4461u && width == 16u) ++special16;
            else if (mode == 4462u && width == 16u) ++rte16;
            else if (mode == 4462u && width == 32u) ++rte32;
            else ++other;
        }
        at += length;
    }
    const bool ok = denorm32 == unsigned(bool(wanted & backend::kDenorm32)) &&
        special32 == unsigned(bool(wanted & backend::kSpecial32)) &&
        special64 == unsigned(bool(wanted & backend::kSpecial64)) && !other &&
        denorm16 == unsigned(bool(wanted & backend::kDenorm16)) &&
        special16 == unsigned(bool(wanted & backend::kSpecial16)) &&
        rte16 == unsigned(bool(wanted & backend::kRte16)) && rte32 == unsigned(bool(wanted & backend::kRte32)) &&
        (!(denorm32 || denorm16) || denorm_cap) && (!(special32 || special64 || special16) || special_cap) &&
        (!(rte16 || rte32) || rte_cap);
    if (!ok) std::cout << name << " wanted=" << wanted << " DenormPreserve32=" << denorm32
                      << " SignedZeroInfNanPreserve32=" << special32 << " SignedZeroInfNanPreserve64=" << special64
                      << " DenormPreserve16=" << denorm16 << " SignedZeroInfNanPreserve16=" << special16
                      << " RoundingModeRTE16=" << rte16 << " RoundingModeRTE32=" << rte32
                      << " other=" << other << " FAIL\n";
    return ok;
}

int float_control_checks(const char* output_dir = nullptr) {
    constexpr uint32_t supported = backend::kMxfp4Controls | backend::kF16InputControls;
    const backend::KernelId ids[] = {backend::K_MATMUL_ROW_MXFP4, backend::K_MATMUL_ROW_MXFP4_DOT,
        backend::K_MATMUL_ROW_MXFP4_FLOAT_X, backend::K_MATMUL_TILE_MXFP4,
        backend::K_MATMUL_TILE_MXFP4_BF16, backend::K_MATMUL_REDUCE_MXFP4, backend::K_MATMUL_TILE_Q8MX};
    int failures = 0;
    for (const auto id : ids) {
        std::vector<uint32_t> storage;
        const auto info = backend::kernel_module(id, false, backend::kMxfp4Controls, storage);
        if (!check_float_modes(backend::kKernelNames[id], info.pCode, info.codeSize / sizeof(uint32_t),
                              backend::kDenorm32 | backend::kSpecial32 | backend::kSpecial64)) ++failures;
    }
    for (const auto id : {backend::K_MATMUL_TILE_NVFP4_F16, backend::K_MATMUL_TILE_NVFP4_F16_TALL}) {
        std::vector<uint32_t> storage;
        // No 64-bit property is supplied: the native-half tile uses only F16/F32.
        const auto info = backend::kernel_module(id, false, backend::kF16InputControls, storage);
        if (!check_float_modes(backend::kKernelNames[id], info.pCode, info.codeSize / sizeof(uint32_t),
                              backend::kDenorm32 | backend::kSpecial32 | backend::kDenorm16 | backend::kSpecial16 |
                              backend::kRte16 | backend::kRte32)) ++failures;
        bool half = false, single = false, wide = false;
        for (size_t at = 5; at < info.codeSize / sizeof(uint32_t); at += info.pCode[at] >> 16) {
            const uint32_t opcode = info.pCode[at] & 0xffffu, length = info.pCode[at] >> 16;
            if (opcode == 22u && length == 3u) {
                half |= info.pCode[at + 2] == 16u;
                single |= info.pCode[at + 2] == 32u;
                wide |= info.pCode[at + 2] == 64u;
            }
            if (opcode == 17u && length == 2u && info.pCode[at + 1] == 10u) wide = true;
        }
        if (!half || !single || wide) ++failures;
    }
    size_t checked = 0, refusals = 0;
    for (int id = 0; id < backend::K_COUNT; ++id) {
        const auto& source = backend::kKernels[id];
        for (const bool preserve : {false, true}) {
            const uint32_t wanted = preserve && source.preserved_words ? source.preserved_controls : source.controls;
            std::vector<uint32_t> storage;
            const auto info = backend::kernel_module(static_cast<backend::KernelId>(id), preserve, supported, storage);
            if (!check_float_modes(backend::kKernelNames[id], info.pCode, info.codeSize / sizeof(uint32_t), wanted)) ++failures;
            if (!(wanted & (backend::kSpecial64 | backend::kDenorm16)) && !storage.empty()) ++failures;
            if (output_dir && !storage.empty()) {
                const auto file = std::filesystem::path(output_dir) / (std::string(backend::kKernelNames[id]) + (preserve ? "-preserved.spv" : ".spv"));
                std::filesystem::create_directories(file.parent_path());
                std::ofstream out(file, std::ios::binary);
                out.write(reinterpret_cast<const char*>(info.pCode), std::streamsize(info.codeSize));
                out.close();
                if (!out) throw std::runtime_error("float controls: cannot write prepared module");
            }
            ++checked;
            for (const uint32_t missing : {backend::kDenorm32, backend::kSpecial32, backend::kSpecial64,
                                          backend::kDenorm16, backend::kSpecial16, backend::kRte16, backend::kRte32}) {
                if (!(wanted & missing)) continue;
                bool refused = false;
                try { (void)backend::kernel_module(static_cast<backend::KernelId>(id), preserve, supported & ~missing, storage); }
                catch (const std::runtime_error& e) { refused = std::string(e.what()) == std::string("vulkan: device lacks float controls for ") + backend::kKernelNames[id]; }
                if (!refused) ++failures;
                ++refusals;
            }
        }
    }
    for (unsigned mask = 0; mask <= supported; ++mask) {
        VkPhysicalDeviceFloatControlsProperties p{};
        p.shaderDenormPreserveFloat32 = mask & 1 ? VK_TRUE : VK_FALSE;
        p.shaderSignedZeroInfNanPreserveFloat32 = mask & 2 ? VK_TRUE : VK_FALSE;
        p.shaderSignedZeroInfNanPreserveFloat64 = mask & 4 ? VK_TRUE : VK_FALSE;
        p.shaderDenormPreserveFloat16 = mask & 8 ? VK_TRUE : VK_FALSE;
        p.shaderSignedZeroInfNanPreserveFloat16 = mask & 16 ? VK_TRUE : VK_FALSE;
        p.shaderRoundingModeRTEFloat16 = mask & 32 ? VK_TRUE : VK_FALSE;
        p.shaderRoundingModeRTEFloat32 = mask & 64 ? VK_TRUE : VK_FALSE;
        if (backend::float_control_support(p) != mask) ++failures;
    }
    std::cout << "float controls: " << checked << " prepared modules, " << refusals
              << " missing-property refusals, " << failures << " failures\n";
    return failures;
}

int created_float_control_checks() {
    auto base = backend::make_vulkan_backend(0);
    auto& b = dynamic_cast<backend::VulkanBackend&>(*base);
    KernelCalls calls_for_modules(b, 0);
    calls_for_modules.capture_module = true;
    const auto& device = *calls_for_modules.device;
    int failures = 0;
    size_t checked = 0;
    for (int id = 0; id < backend::K_COUNT; ++id) {
        const auto& source = backend::kKernels[id];
        const uint32_t wanted = device.preserve_float32 && source.preserved_words ? source.preserved_controls : source.controls;
        if (wanted & ~device.float_controls) continue;
        calls_for_modules.module_words.clear();
        backend::VulkanLifetimeTest::kernel(b, static_cast<backend::KernelId>(id));
        const auto& words = calls_for_modules.module_words;
        if (!check_float_modes(backend::kKernelNames[id], words.data(), words.size(), wanted)) ++failures;
        VkShaderModuleCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.codeSize = words.size() * sizeof(uint32_t);
        info.pCode = words.data();
        VkShaderModule module = VK_NULL_HANDLE;
        const auto result = calls_for_modules.original.vkCreateShaderModule(device.device, &info, nullptr, &module);
        if (result != VK_SUCCESS) {
            std::cout << backend::kKernelNames[id] << " driver module creation failed: " << result << '\n';
            ++failures;
        }
        if (module) calls_for_modules.original.vkDestroyShaderModule(device.device, module, nullptr);
        ++checked;
    }
    base.reset();
    if (!calls_for_modules.empty()) ++failures;
    std::cout << "float controls: " << checked << " intercepted modules and driver creation checks, " << failures << " failures\n";
    return failures;
}

// A non-causal attention call's pieces (attention_pieces): they cover its workgroups once and in order, none over the budget unless it is one workgroup, a cut view's pieces within one workgroup of each other.
// Among them Flash-Next's tower at 2048 by 2048, 16384 patches of 16 heads 72 wide, in five pieces of the tiled build, and its 1024 by 1024 and the backend test's 8192 patches in one and two.
int attention_piece_checks() {
    using Runs = std::vector<std::pair<size_t, size_t>>;
    const size_t tile72 = 2 * backend::kFullTileRows * 72;
    struct Case { const char* name; Runs runs; size_t budget, pieces; };
    const Case cases[] = {
        {"Flash-Next at 2048 by 2048", {{256 * 16, tile72 * 16384}}, backend::kFullTileProducts, 5},
        {"Flash-Next at 1024 by 1024", {{64 * 16, tile72 * 4096}}, backend::kFullTileProducts, 1},
        {"8192 patches of 16 heads 72 wide", {{128 * 16, tile72 * 8192}}, backend::kFullTileProducts, 2},
        {"4096 rows of 4 heads 128 wide, per row", {{4096 * 4, 2 * 128 * 4096}}, backend::kFullRowProducts, 2},
        {"a view beside the open piece", {{10, 10}, {7, 10}, {40, 10}}, 100, 6},
        {"a workgroup over the budget", {{3, 500}, {2, 10}}, 100, 4},
        {"nothing", {}, 100, 0},
        {"an empty view", {{0, 10}, {5, 10}}, 100, 1},
    };
    int failures = 0;
    for (const Case& c : cases) {
        const std::vector<size_t> cuts = backend::attention_pieces(c.runs, c.budget);
        size_t total = 0;
        for (const auto& r : c.runs) total += r.first;
        bool ok = !cuts.empty() && cuts.front() == 0 && cuts.back() == total && cuts.size() == c.pieces + 1;
        for (size_t i = 0; ok && i + 1 < cuts.size(); ++i) {
            size_t products = 0, at = 0;
            for (const auto& r : c.runs) {
                const size_t lo = std::max(at, cuts[i]), hi = std::min(at + r.first, cuts[i + 1]);
                if (lo < hi) products += (hi - lo) * r.second;
                at += r.first;
            }
            ok = cuts[i] < cuts[i + 1] && (products <= c.budget || cuts[i + 1] - cuts[i] == 1);
        }
        if (c.runs.size() == 1)
            for (size_t i = 0; ok && i + 2 < cuts.size(); ++i) {
                const size_t a = cuts[i + 1] - cuts[i], b = cuts[i + 2] - cuts[i + 1];
                ok = std::max(a, b) - std::min(a, b) <= 1;
            }
        std::cout << "attention pieces, " << c.name << ": " << (cuts.empty() ? 0 : cuts.size() - 1) << (ok ? " PASS\n" : " FAIL\n");
        failures += !ok;
    }
    return failures;
}

int main(int argc, char** argv) {
    try {
        if ((argc == 2 || argc == 3) && std::string(argv[1]) == "--float-controls") return float_control_checks(argc == 3 ? argv[2] : nullptr) ? 1 : 0;
        if (argc == 2 && std::string(argv[1]) == "--float-controls-device") return created_float_control_checks() ? 1 : 0;
        if (argc == 2 && std::string(argv[1]) == "--collective") return collective_checks() ? 1 : 0;
        if (argc == 2 && std::string(argv[1]) == "--queue") {
            const int failures = queue_checks() + dictionary_checks() + kernel_checks() + query_checks(false) + query_checks(true) + padded_drop_checks() +
                                 loader_weight_checks() + scratch_reserve_checks() + hold_checks() + hold_timing_checks() + collective_checks() + lent_checks() + wait_beside_recording() + args_slot_checks();
            return failures ? 1 : 0;
        }
        if (argc != 1) return 2;
        int failures = allocation_error_checks(), cases = 0;
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
        // A device whose memory is the host's (Backend::memory_is_host) by its type alone: an integrated GPU and a CPU device are, a discrete and a virtual GPU are not.
        if (!backend::host_memory_device(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) || !backend::host_memory_device(VK_PHYSICAL_DEVICE_TYPE_CPU) ||
            backend::host_memory_device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) || backend::host_memory_device(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU))
            ++failures;
        ++cases;
        failures += share_checks();
        ++cases;
        failures += table_view_checks();
        failures += attention_piece_checks();
        cases += 6;
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
