// backend-rocm: the ROCm backend's buffers, tickets, refusals and dense F32 ops against the CPU backend, and each op's batch invariance; exits 77 without a runtime or a device.
#include "backends/cpu/cpu_backend.hpp"
#include "backends/kv_storage.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_device.hpp"
#include "backends/rocm/rocm_runtime.hpp"
#include "core/fp16.hpp"
#include "core/host_memory.hpp"
#include "deepseek4_checks.hpp"
#include "model/kv_cache.hpp"
#include "quant/fp4.hpp"
#include "quant/types.hpp"
#include "group_sums.hpp"
#include "qwen35_checks.hpp"
#include "row_classes.hpp"
#include "scratch_growth.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>

using namespace backend;

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cout << "FAIL " << what << "\n";
        ++failures;
    }
}
template <class F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
std::vector<float> read(Backend& b, const Buffer& buf, size_t n, size_t off = 0) {
    std::vector<float> v(n);
    b.read(buf, off * sizeof(float), v.data(), n * sizeof(float));
    return v;
}
// Every value within 1e-4 * (1 + |reference|) of the CPU's, the bound the device tests hold F32 ops to.
void close(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
    bool ok = got.size() == want.size();
    size_t bad = 0;
    for (size_t i = 0; ok && i < got.size(); ++i) ok = std::fabs(got[i] - want[i]) <= 1e-4f * (1.0f + std::fabs(want[i])), bad = i;
    if (!ok && got.size() == want.size()) std::fprintf(stderr, "  value %zu: %.9g against %.9g\n", bad, got[bad], want[bad]);
    require(ok, what);
}
bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; }

void buffers(Backend& g) {
    const std::vector<float> x = row_classes::values(1000, 1);
    auto d = g.alloc(4000), h = g.alloc(4000, Memory::host_visible);
    require(read(g, *d, 1000) == std::vector<float>(1000, 0.0f), "device memory reads as zeros");
    require(read(g, *h, 1000) == std::vector<float>(1000, 0.0f), "host-visible memory reads as zeros");
    g.write(*d, 40, x.data() + 10, 400);
    g.copy(*h, 0, *d, 0, 4000);
    g.sync();
    std::vector<float> want(1000, 0.0f);
    std::copy(x.begin() + 10, x.begin() + 110, want.begin() + 10);
    require(read(g, *d, 1000) == want, "a write at an offset reads back");
    require(std::memcmp(h->host_ptr(), want.data(), 4000) == 0, "host-visible memory holds a copy in place after sync");
    auto a = g.adopt(x.data(), 4000);
    require(read(g, *a, 1000) == x, "an adopted buffer holds its bytes");
    // A buffer dropped right after the work that reads it: the work is ordered on the stream before the free.
    {
        auto tmp = g.adopt(x.data(), 4000);
        g.copy(*d, 0, *tmp, 0, 4000);
    }
    require(read(g, *d, 1000) == x, "a buffer dropped after its copy was enqueued gave its bytes");
    // A large buffer dropped while many copies out of it are still queued: hipFree unmaps it at once, so its memory must wait for them.
    {
        const size_t n = size_t(256) << 20;
        auto dst = g.alloc(n);
        {
            auto src = g.alloc(n);
            g.write(*src, 0, x.data(), 4000);
            for (int i = 0; i < 64; ++i) g.copy(*dst, 0, *src, 0, n);
        }
        require(read(g, *dst, 1000) == x, "a large buffer dropped with copies out of it queued gave its bytes");
    }
    require(throws([&] { g.read(*d, 3999, nullptr, 2); }), "a read past the buffer is refused");
    require(throws([&] { g.write(*d, 0, x.data(), 4001); }), "a write past the buffer is refused");
    auto big = [&] { g.alloc(size_t(1) << 50); };
    require(throws(big), "an allocation past the device's memory is refused");
    const Ticket t1 = g.submit(), t2 = g.submit();
    require(t2 == t1 + 1, "tickets increase by one");
    g.wait(t2);
    require(g.done(t1) && g.done(t2), "waited tickets read as done");
    // More tickets than events, so slots are reused.
    for (int i = 0; i < 2100; ++i) g.submit();
    g.sync();
    require(g.done(t2 + 2100), "tickets past the event ring retire");
    // Another thread waits on each ticket while this one keeps launching and submitting, as a scheduler waits on a pass while the next is recorded.
    {
        auto buf = g.alloc(1 << 20);
        std::atomic<Ticket> handed{0};
        std::atomic<bool> stop{false}, ok{true};
        std::thread waiter([&] {
            Ticket seen = 0;
            while (!stop || seen < handed) {
                const Ticket t = handed;
                if (t == seen) continue;
                g.wait(t);
                if (!g.done(t)) ok = false;
                seen = t;
            }
        });
        for (int i = 0; i < 3000; ++i) {
            g.copy(*buf, 0, *buf, 1 << 19, 1 << 19);
            handed = g.submit();
        }
        stop = true;
        waiter.join();
        g.sync();
        require(ok && g.done(handed), "tickets waited on from another thread while this one submits read as done");
    }
    // A ticket another thread is waiting on reads as not done at once: HIP would hold the query until the work ends, stalling a recording thread that only asks.
    {
        const size_t n = size_t(256) << 20;
        auto src = g.alloc(n), dst = g.alloc(n);
        for (int i = 0; i < 256; ++i) g.copy(*dst, 0, *src, 0, n);
        const Ticket t = g.submit();
        std::atomic<bool> entering{false};
        std::thread waiter([&] {
            entering.store(true);
            g.wait(t);
        });
        while (!entering.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto from = std::chrono::steady_clock::now();
        const bool early = g.done(t);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
        waiter.join();
        if (ms >= 50.0) std::fprintf(stderr, "  done took %.1f ms\n", ms);
        require(!early && ms < 50.0, "a ticket another thread waits on reads as not done without waiting");
        require(g.done(t), "the waited ticket reads as done once its waiter returns");
    }
    // A dropped buffer's memory goes back without draining the work queued after the drop: hipFree waits for every stream of the device, so a submitting or waiting thread that freed would sit behind all of that work.
    {
        const size_t n = size_t(256) << 20, dropped = size_t(512) << 20;
        auto src = g.alloc(n), dst = g.alloc(n);
        g.sync();
        const size_t before = g.memory_available().value_or(0);
        auto heavy = [&] {
            for (int i = 0; i < 256; ++i) g.copy(*dst, 0, *src, 0, n);
        };
        auto since = [](std::chrono::steady_clock::time_point from) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
        };
        // Through a submission: the drop's ticket is seen done, then the next submission follows a long queue.
        {
            auto victim = g.alloc(dropped);
            g.copy(*victim, 0, *src, 0, n);
        }
        const Ticket t1 = g.submit();
        while (!g.done(t1)) {}
        heavy();
        auto from = std::chrono::steady_clock::now();
        const Ticket t2 = g.submit();
        double ms = since(from);
        const bool queued = !g.done(t2);
        if (ms >= 50.0 || !queued) std::fprintf(stderr, "  submit after a drop took %.1f ms\n", ms);
        require(queued && ms < 50.0, "a submission leaves a dropped buffer's free to run beside the work queued before it");
        g.sync();
        // Through a wait: the drop's ticket is waited on while a long queue follows it.
        {
            auto victim = g.alloc(dropped);
            g.copy(*victim, 0, *src, 0, n);
        }
        const Ticket t3 = g.submit();
        heavy();
        const Ticket t4 = g.submit();
        from = std::chrono::steady_clock::now();
        g.wait(t3);
        ms = since(from);
        const bool waiting = !g.done(t4);
        if (ms >= 50.0 || !waiting) std::fprintf(stderr, "  wait after a drop took %.1f ms\n", ms);
        require(waiting && ms < 50.0, "a wait leaves a dropped buffer's free to run beside the work queued after it");
        g.sync();
        // The fit reads free memory right after sync, so the drops' memory must be back then, with no polling.
        require(g.memory_available().value_or(0) + (size_t(64) << 20) >= before, "dropped buffers' memory is free again at once after sync");
        // A drop after the last ticket, with work queued before it, goes back at sync too.
        {
            auto victim = g.alloc(dropped);
            g.copy(*victim, 0, *src, 0, n);
            heavy();
            (void)g.submit();
        }
        g.sync();
        require(g.memory_available().value_or(0) + (size_t(64) << 20) >= before, "a buffer dropped after the last submission is free again at once after sync");
        // An allocation the device refuses waits for dropped memory whose work is still queued, and is still refused with nothing dropped.
        const size_t want = size_t(1) << 30, chunk = size_t(256) << 20;
        auto victim = g.alloc(want);
        std::vector<BufferPtr> fill;
        auto fill_up = [&] {
            while (g.memory_available().value_or(0) >= chunk + want / 2) {
                BufferPtr f;
                if (throws([&] { f = g.alloc(chunk); })) break;
                fill.push_back(f);
            }
        };
        // Filled again after a sync, so memory a sync gives back late still leaves less than one allocation.
        fill_up();
        g.sync();
        fill_up();
        require(throws([&] { g.alloc(want); }), "an allocation past the free memory is refused with nothing dropped");
        g.copy(*victim, 0, *src, 0, n);
        heavy();
        victim.reset();
        const Ticket t5 = g.submit();
        const bool pending = !g.done(t5);
        BufferPtr again;
        const bool refused = throws([&] { again = g.alloc(want); });
        if (refused || !pending) std::fprintf(stderr, "  retry: work queued %d, refused %d, free %zu\n", int(pending), int(refused), g.memory_available().value_or(0));
        require(pending && !refused && again && again->size() == want, "an allocation the device refuses takes the memory of a buffer dropped under a ticket with work queued");
        again.reset();
        fill.clear();
        g.sync();
    }
}

// Host memory a device copies from in place (Backend::wrap_host): the caller frees it once its views are gone, so a dropped view must leave it unregistered and the next memory at its address must read as itself.
void host_views(Backend& g) {
    HipFn& f = hip();
    const size_t n = 1 << 16, bytes = n * sizeof(float);
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = float(i) * 0.5f;
    {
        // A view dropped after its copy ran, with no backend wait since, must unregister at once: a raw device synchronize does not run the backend's deferred cleanup.
        void* lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
        using SyncFn = hipError_t (*)();
        const SyncFn device_sync = lib ? reinterpret_cast<SyncFn>(dlsym(lib, "hipDeviceSynchronize")) : nullptr;
        require(device_sync != nullptr, "hipDeviceSynchronize was not found");
        if (device_sync) {
            core::HostPages pages(bytes);
            std::memcpy(pages.data(), x.data(), bytes);
            BufferPtr view = g.wrap_host(pages.data(), bytes), d = g.alloc(bytes);
            g.copy(*d, 0, *view, 0, bytes);
            g.submit();
            require(device_sync() == hipSuccess, "hipDeviceSynchronize failed");
            view.reset();
            const hipError_t u = f.hipHostUnregister(pages.data());
            require(u == hipErrorHostMemoryNotRegistered, std::string("host pages whose view dropped after a raw device synchronize stayed registered: ") + f.hipGetErrorName(u));
            g.wait(g.submit());
        }
    }
    void* first = nullptr;
    {
        core::HostPages pages(bytes);
        first = pages.data();
        std::memcpy(pages.data(), x.data(), bytes);
        BufferPtr view = g.wrap_host(pages.data(), bytes), again = g.wrap_host(pages.data(), bytes);
        require(view && again && view->host_ptr() == pages.data(), "host pages were not wrapped twice");
        require(!g.wrap_host(pages.data(), bytes / 2), "a second span at a wrapped address was wrapped");
        BufferPtr d = g.alloc(bytes), e = g.alloc(bytes);
        g.copy(*d, 0, *view, 0, bytes);
        g.wait(g.submit());
        // One view dropped, and its stream moved on past the drop, leaves the other its registration.
        view.reset();
        g.wait(g.submit());
        g.copy(*e, 0, *again, 0, bytes);
        require(read(g, *e, n) == x, "a copy through a view after another view of its memory dropped is not its bytes");
        require(read(g, *d, n) == x, "a copy out of wrapped host pages is not their bytes");
        again.reset();
        // HIP takes a second registration of a span and drops it at one unregister, so the probe is the unregister: it must find nothing left.
        const hipError_t u = f.hipHostUnregister(pages.data());
        require(u == hipErrorHostMemoryNotRegistered, std::string("host pages whose views were dropped stayed registered: ") + f.hipGetErrorName(u));
        if (u != hipErrorHostMemoryNotRegistered) return;
    }
    core::HostPages pages(bytes);
    for (size_t i = 0; i < n; ++i) x[i] = -float(i);
    std::memcpy(pages.data(), x.data(), bytes);
    BufferPtr view = g.wrap_host(pages.data(), bytes), d = g.alloc(bytes);
    require(view != nullptr, "host pages mapped after others were freed were not wrapped");
    g.copy(*d, 0, *view, 0, bytes);
    require(read(g, *d, n) == x, "a copy out of host pages mapped after others were freed is not their bytes");
    std::cout << "backend-rocm: wrapped host pages copy in place and drop their registration with their last view"
              << (pages.data() == first ? ", the second at the first's address\n" : "\n");
    int count = 0;
    hip_check(f.hipGetDeviceCount(&count), "hipGetDeviceCount");
    if (count < 2) return;
    // Two members of a head group: member 0's view dropped, member 1 still copies into the memory, and once its view drops nothing is registered.
    BackendPtr h = make_rocm_backend(1);
    core::HostPages rows(bytes);
    std::vector<BufferPtr> views = {g.wrap_host(rows.data(), bytes), h->wrap_host(rows.data(), bytes)};
    require(views[0] && views[1], "two members did not wrap one host memory");
    views[0].reset();
    g.wait(g.submit());
    BufferPtr src = h->alloc(bytes);
    h->write(*src, 0, x.data(), bytes);
    h->copy(*views[1], 0, *src, 0, bytes);
    h->wait(h->submit());
    require(std::memcmp(rows.data(), x.data(), bytes) == 0, "a member's copy into host memory after the other member's view dropped is not its rows");
    views[1].reset();
    h->wait(h->submit());
    const hipError_t u = f.hipHostUnregister(rows.data());
    require(u == hipErrorHostMemoryNotRegistered, std::string("host memory both members' views dropped stayed registered: ") + f.hipGetErrorName(u));
    std::cout << "backend-rocm: a member copies into host memory after the other member's view dropped, and the last drop unregisters it\n";
}

void refusals(Backend& g) {
    std::vector<uint8_t> q4k(144 * 4, 0);
    auto w = g.adopt(q4k.data(), q4k.size());
    auto x = g.alloc(256 * 4), y = g.alloc(16);
    require(g.supports_type(quant::GGML_TYPE_F32) && g.supports_type(quant::GGML_TYPE_Q8_0) && g.supports_type(quant::GGML_TYPE_Q4_K) && !g.supports_type(quant::GGML_TYPE_Q2_K),
            "the backend supports the types of its descriptor table and no other");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q2_K, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 256, 4, 1); }), "a Q2_K matmul is refused");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q4_K, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 128, 4, 1); }), "a Q4_K row ending inside a block is refused");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q8_0, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 48, 4, 1); }), "a Q8_0 row ending inside a block is refused");
    // The ops ROCm implements are its `implements`, which the kernel checks below reach one by one; the vision tower's layer norm is one it lacks.
    require(!g.implements(Op::layer_norm), "the backend does not report the layer norm it lacks");
    for (Op op : {Op::sqrt_softplus_routing, Op::silu_mul_clamp, Op::rope_forms, Op::hc_coeffs, Op::pool_write, Op::model_format_qdq})
        require(g.implements(op), std::string("the backend reports the ") + backend::op_name(op) + " it implements");
    require(throws([&] { g.layer_norm_rows({y.get(), 0}, {x.get(), 0}, {x.get(), 0}, {x.get(), 0}, 1, 4, 4, 1e-6f); }), "an op the backend lacks throws");
}

// A Q8_0 matrix of `rows` rows `nin` wide: each block's scale and values from the seed, every value in range.
std::vector<uint8_t> q8_matrix(size_t nin, size_t rows, uint32_t seed) {
    std::vector<uint8_t> m;
    const std::vector<float> scale = row_classes::values(rows * nin / 32, seed, 0.05f), v = row_classes::values(rows * nin, seed + 1, 127.0f);
    for (size_t b = 0; b < rows * nin / 32; ++b) {
        const uint16_t d = f32_to_f16(scale[b]);
        m.push_back(uint8_t(d));
        m.push_back(uint8_t(d >> 8));
        for (size_t j = 0; j < 32; ++j) m.push_back(uint8_t(int8_t(std::max(-127.0f, std::min(127.0f, std::round(v[b * 32 + j]))))));
    }
    return m;
}

// A product into pinned host memory, as the logits are, gives the bits of the same product into device memory, both written and added, for a generated row and a prompt's rows.
// A copy between two host memories, as a tensor group's head copies its member logits into the rows every member wraps: enqueued behind queued work, it returns at once and holds the source's bytes once that work retires.
void host_copies(Backend& g) {
    const size_t nin = 4096, nout = 4096, cols = 64, n = 124160;
    const std::vector<float> x = row_classes::values(nin * cols, 61), v = row_classes::values(nin * nout, 62), want = row_classes::values(n + 3, 63);
    auto wb = g.adopt(v.data(), v.size() * 4), xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(nout * cols * 4);
    const auto work = [&](int calls) {
        for (int i = 0; i < calls; ++i) g.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, cols, {}, Dtype::f32);
    };
    // Enough queued work to take 200 ms at least on this device.
    int calls = 1;
    for (;; calls *= 2) {
        g.sync();
        const auto t0 = std::chrono::steady_clock::now();
        work(calls);
        g.sync();
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() >= 200) break;
    }
    core::HostPages pages((n + 3) * sizeof(float));
    BufferPtr view = g.wrap_host(pages.data(), pages.size()), src = g.alloc((n + 3) * sizeof(float), Memory::host_visible);
    require(view && src, "the host memories of a host copy were not made");
    g.write(*src, 0, want.data(), (n + 3) * sizeof(float));
    std::memset(pages.data(), 0, pages.size());
    g.sync();
    // A whole-word copy and one of three floats at an offset of one, which takes the byte loop.
    const auto t0 = std::chrono::steady_clock::now();
    work(calls);
    const auto t1 = std::chrono::steady_clock::now();
    g.copy(*view, 0, *src, 0, n * sizeof(float));
    g.copy(*view, (n + 1) * sizeof(float), *src, n * sizeof(float), 2 * sizeof(float));
    const auto t2 = std::chrono::steady_clock::now();
    g.wait(g.submit());
    const auto t3 = std::chrono::steady_clock::now();
    const double queued = std::chrono::duration<double, std::milli>(t3 - t1).count(), call = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::printf("backend-rocm: two host-to-host copies behind %d products: %.3f ms in the calls, %.1f ms until the work retired (%.1f ms to queue it)\n", calls, call,
                queued, std::chrono::duration<double, std::milli>(t1 - t0).count());
    require(call < queued / 4, "a copy between host memories waited for the work queued before it");
    const float* got = (const float*)pages.data();
    require(std::memcmp(got, want.data(), n * sizeof(float)) == 0, "a host-to-host copy is not the source's bytes");
    require(got[n] == 0.0f && std::memcmp(got + n + 1, want.data() + n, 2 * sizeof(float)) == 0, "an unaligned host-to-host copy is not the source's bytes or wrote past them");
    view.reset();
    g.sync();
}

void host_output(Backend& g) {
    const size_t nin = 256, nout = 300;
    const std::vector<uint8_t> w = q8_matrix(nin, nout, 51);
    const std::vector<float> x = row_classes::values(nin * 3, 52), y0 = row_classes::values(nout * 3, 53);
    auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4);
    const WeightView wv{quant::GGML_TYPE_Q8_0, {wb.get(), 0}};
    for (size_t rows : {size_t(1), size_t(3)})
        for (bool add : {false, true}) {
            auto dev = g.adopt(y0.data(), nout * rows * 4), host = g.alloc(nout * rows * 4, Memory::host_visible);
            g.write(*host, 0, y0.data(), nout * rows * 4);
            if (add) {
                g.matmul_add(wv, {xb.get(), 0}, {dev.get(), 0}, nin, nout, rows);
                g.matmul_add(wv, {xb.get(), 0}, {host.get(), 0}, nin, nout, rows);
            } else {
                g.matmul(wv, {xb.get(), 0}, {dev.get(), 0}, nin, nout, rows);
                g.matmul(wv, {xb.get(), 0}, {host.get(), 0}, nin, nout, rows);
            }
            g.sync();
            require(same(read(g, *dev, nout * rows), read(g, *host, nout * rows)),
                    std::string("a product into host memory is the product into device memory (") + (add ? "added" : "written") + ", " + std::to_string(rows) + " rows)");
        }
}

// Rows of `type` from seeded bytes, every f16 or bf16 field set to a finite scale, so each value is defined and finite.
std::vector<uint8_t> typed_matrix(uint32_t type, size_t nin, size_t rows, uint32_t seed) {
    std::vector<uint8_t> m(quant::row_bytes(type, nin, rows));
    uint32_t r = seed * 2654435761u + 1;
    for (uint8_t& v : m) {
        r = r * 1664525u + 1013904223u;
        v = uint8_t(r >> 24);
    }
    const quant::StorageType& st = *quant::storage_type(type);
    const std::vector<float> scale = row_classes::values(m.size() / st.type_size * 2, seed + 7, 0.02f);
    const auto put = [&](size_t at, uint16_t h) {
        m[at] = uint8_t(h);
        m[at + 1] = uint8_t(h >> 8);
    };
    std::vector<size_t> halves;
    if (type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q5_1) halves = {0, 2};
    if (type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_IQ4_NL) halves = {0};
    if (type == quant::GGML_TYPE_Q6_K) halves = {208};
    for (size_t b = 0, i = 0; b < m.size() / st.type_size; ++b)
        for (size_t h : halves) put(b * st.type_size + h, f32_to_f16(scale[i++]));
    // MXFP4's E8M0 scale byte at 2^-10 to 2^-6, as the raw-block fixtures' ordinary blocks take it.
    if (type == quant::GGML_TYPE_MXFP4)
        for (size_t b = 0; b < m.size() / st.type_size; ++b) m[b * st.type_size] = uint8_t(118 + m[b * st.type_size] % 5);
    if (type == quant::GGML_TYPE_BF16) {
        const std::vector<float> v = row_classes::values(nin * rows, seed + 9);
        for (size_t i = 0; i < v.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &v[i], 4);
            put(2 * i, uint16_t(u >> 16));
        }
    }
    return m;
}

// Flash-Next's projection shapes, input by output, that a type's rows fill whole: the K-quants' 256-value blocks the 2560- and 6144-wide rows, the 32-value types the 640-wide expert down rows too.
std::vector<std::array<size_t, 2>> flash_shapes(uint32_t type) {
    const size_t all[][2] = {{2560, 640}, {640, 2560}, {2560, 320}, {2560, 512}, {6144, 2560}};
    const uint64_t block = quant::storage_type(type)->block_size;
    std::vector<std::array<size_t, 2>> out;
    for (const auto& sh : all)
        if (sh[0] % block == 0) out.push_back({sh[0], sh[1]});
    return out;
}

// Double sums of the CPU's decoded weights wd over x, columns of nout, and beside each the sum of |w*x| that bounds an F32 reduction's rounding.
struct DoubleSums {
    std::vector<double> sum, mag;
};
DoubleSums double_sums(const std::vector<float>& wd, const std::vector<float>& x, size_t nin, size_t nout) {
    const size_t n = x.size() / nin * nout;
    DoubleSums d{std::vector<double>(n), std::vector<double>(n)};
    for (size_t i = 0; i < n; ++i) {
        const size_t r = i / nout, o = i % nout;
        for (size_t j = 0; j < nin; ++j) {
            const double p = double(wd[o * nin + j]) * x[r * nin + j];
            d.sum[i] += p;
            d.mag[i] += std::fabs(p);
        }
    }
    return d;
}

// An exact product against the double sums, within 16 F32 epsilons of the sum of |w*x|: each backend reduces in its own order, and the bound covers that rounding however the terms cancel.
void exact_close(const std::vector<float>& got, const DoubleSums& d, const std::string& what) {
    bool ok = got.size() == d.sum.size();
    for (size_t i = 0; ok && i < got.size(); ++i) {
        const double bound = 16 * double(FLT_EPSILON) * d.mag[i];
        ok = std::fabs(got[i] - d.sum[i]) <= bound + 1e-30;
        if (!ok) std::fprintf(stderr, "  value %zu: %.9g against %.9g, bound %.3g\n", i, got[i], d.sum[i], bound);
    }
    require(ok, what);
}

// The descriptor table's types at Flash-Next's shapes: embeddings the CPU's rows bit for bit, exact products of the CPU and the device within the F32 bound of a double sum, and a row of a call the row alone.
void types(Backend& g, Backend& c) {
    const uint32_t kinds[] = {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q5_0,
                              quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
    for (uint32_t type : kinds) {
        const std::string name = quant::storage_type(type)->name;
        require(g.supports_type(type), name + " is supported");
        for (const auto& sh : flash_shapes(type)) {
            const size_t nin = sh[0], nout = sh[1];
            const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(nout);
            const std::vector<uint8_t> w = typed_matrix(type, nin, nout, uint32_t(type * 31 + nin + nout));
            const std::vector<float> x = row_classes::values(nin * 33, uint32_t(nin + 5));
            // The last id is past the table, which writes a zero row (Backend::embed).
            const uint32_t ids[5] = {uint32_t(nout - 1), 0, 7, uint32_t(nout / 2), uint32_t(nout)};
            std::vector<float> emb[2];
            Backend* bs[2] = {&g, &c};
            for (int k = 0; k < 2; ++k) {
                Backend& b = *bs[k];
                auto wb = b.adopt(w.data(), w.size()), eb = b.alloc(5 * nin * 4);
                b.embed({eb.get(), 0}, {type, {wb.get(), 0}}, nin, nout, ids, 5);
                emb[k] = read(b, *eb, 5 * nin);
            }
            require(std::all_of(emb[0].end() - std::ptrdiff_t(nin), emb[0].end(), [](float v) { return v == 0.0f && !std::signbit(v); }),
                    "a " + at + " embedding writes a zero row for an id past its table");
            for (size_t i = 0; i < emb[0].size(); ++i)
                if (std::memcmp(&emb[0][i], &emb[1][i], 4)) {
                    std::fprintf(stderr, "  value %zu: %.9g against %.9g\n", i, emb[0][i], emb[1][i]);
                    break;
                }
            require(same(emb[0], emb[1]), "a " + at + " embedding gives the CPU's rows, bit for bit");
            auto cw = c.adopt(w.data(), w.size()), cx = c.adopt(x.data(), x.size() * 4), cy = c.alloc(nout * 33 * 4);
            std::vector<uint32_t> every(nout);
            for (size_t i = 0; i < nout; ++i) every[i] = uint32_t(i);
            auto ce = c.alloc(nout * nin * 4);
            c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, nout, every.data(), nout);
            const DoubleSums want = double_sums(read(c, *ce, nout * nin), x, nin, nout);
            c.matmul({type, {cw.get(), 0}}, {cx.get(), 0}, {cy.get(), 0}, nin, nout, 33, {}, Dtype::f32);
            exact_close(read(c, *cy, nout * 33), want, "a " + at + " CPU matmul of 33 rows against the double sum");
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * 33 * 4);
            // A type with twins takes its exact kernel in the F32 class alone (twin_types).
            const bool twins = type != quant::GGML_TYPE_BF16;
            for (Dtype dt : {Dtype::int8, Dtype::f16, Dtype::f32}) {
                if (twins && dt != Dtype::f32) continue;
                MatrixPaths paths;
                g.swap_matrix_paths(paths);
                g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, 33, {}, dt);
                g.swap_matrix_paths(paths);
                exact_close(read(g, *yb, nout * 33), want, "a " + at + " matmul of 33 rows against the double sum, " + dtype_name(dt));
                require(paths.take() == std::vector<std::string>{"f32"},
                        "a " + at + " matmul runs the exact kernel, " + dtype_name(dt));
            }
            const std::vector<float> all = read(g, *yb, nout * 33);
            auto y1 = g.alloc(nout * 4);
            g.matmul({type, {wb.get(), 0}}, {xb.get(), 32 * nin}, {y1.get(), 0}, nin, nout, 1, {}, Dtype::f32);
            require(std::memcmp(read(g, *y1, nout).data(), all.data() + 32 * nout, nout * 4) == 0, "a row of a " + at + " matmul is the row alone, bit for bit");
        }
    }
}

// Every exact type's product over several rows must give each output the bits of its row alone through the one-row build, plain and added, at each side of a block's outputs and rows.
// The widths give lanes chains of unequal length (2561, 2080), lanes without one (100), the hyper-connection maps' 10240 and F32 rows off the vector loads' alignment (2561).
void exact_rows(Backend& g) {
    const uint32_t kinds[] = {quant::GGML_TYPE_F32,  quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K,
                              quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
    const size_t nouts[] = {1, 13, 512}, nrows[] = {2, 3, 4, 5, 7, 8, 9, 33};
    for (uint32_t type : kinds) {
        const std::string name = quant::storage_type(type)->name;
        const size_t block = quant::storage_type(type)->block_size;
        std::string tname = name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        const auto build = [&](size_t n) { return type == quant::GGML_TYPE_F32 && n <= 4 ? std::string(n <= 2 ? "llmx_matmul_cols2_f32" : "llmx_matmul_cols4_f32") : "llmx_matmul_rows_" + tname; };
        const std::vector<size_t> widths = block == 1 ? std::vector<size_t>{2560, 2561, 100, 10240} : block == 32 ? std::vector<size_t>{2560, 2080} : std::vector<size_t>{2560};
        for (size_t nin : widths)
        for (size_t nout : nouts) {
            std::vector<uint8_t> w;
            if (type == quant::GGML_TYPE_F32) {
                const std::vector<float> v = row_classes::values(nin * nout, uint32_t(nout + 3));
                w.resize(v.size() * 4);
                std::memcpy(w.data(), v.data(), w.size());
            } else {
                w = typed_matrix(type, nin, nout, uint32_t(type * 13 + nout));
                if (type == quant::GGML_TYPE_Q8_0)
                    for (size_t b = 0; b < w.size() / 34; ++b) {
                        const uint16_t h = f32_to_f16(0.01f + 0.001f * float(b % 7));
                        w[b * 34] = uint8_t(h), w[b * 34 + 1] = uint8_t(h >> 8);
                    }
            }
            const std::vector<float> x = row_classes::values(nin * 33, uint32_t(nout + 11)), y0 = row_classes::values(nout * 33, uint32_t(nout + 17));
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * 33 * 4), y1 = g.alloc(nout * 4);
            for (size_t n : nrows)
                for (int add = 0; add < 2; ++add) {
                    const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(nout) + " over " + std::to_string(n) + " rows" + (add ? ", added" : "");
                    g.write(*yb, 0, y0.data(), y0.size() * 4);
                    rocm_take_launches(g);
                    if (add) g.matmul_add({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {}, Dtype::f32);
                    else g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {}, Dtype::f32);
                    require(rocm_take_launches(g).count(build(n)) == 1, "a " + at + " exact product takes " + build(n));
                    const std::vector<float> all = read(g, *yb, nout * 33);
                    bool ok = std::memcmp(all.data() + n * nout, y0.data() + n * nout, (33 - n) * nout * 4) == 0;
                    for (size_t r = 0; r < n && ok; ++r) {
                        g.write(*y1, 0, y0.data() + r * nout, nout * 4);
                        if (add) g.matmul_add({type, {wb.get(), 0}}, {xb.get(), r * nin}, {y1.get(), 0}, nin, nout, 1, {}, Dtype::f32);
                        else g.matmul({type, {wb.get(), 0}}, {xb.get(), r * nin}, {y1.get(), 0}, nin, nout, 1, {}, Dtype::f32);
                        ok = std::memcmp(read(g, *y1, nout).data(), all.data() + r * nout, nout * 4) == 0;
                    }
                    require(ok, "every row of a " + at + " exact product is the row alone, bit for bit, and nothing past it is written");
                }
        }
    }
}

void twin_rule(const float* x, float& d, int16_t* q, float L);

// A block's min for its sub-block: Q4_K's and Q5_K's dmin times the 6-bit min as the format packs them, Q5_1's min; Q6_K, Q5_0, IQ4_NL and MXFP4 have none.
float k_min(uint32_t type, const uint8_t* block, int sub) {
    if (type == quant::GGML_TYPE_Q6_K || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_MXFP4) return 0.0f;
    if (type == quant::GGML_TYPE_Q5_1) return f16_to_f32(uint16_t(block[2] | (block[3] << 8)));
    const uint8_t* q = block + 4;
    const int m = sub < 4 ? q[sub + 4] & 63 : (q[sub + 4] >> 4) | ((q[sub] >> 6) << 4);
    return f16_to_f32(uint16_t(block[2] | (block[3] << 8))) * float(m);
}

// A twin product's output (o, r) against a double sum of the CPU's decoded weights over the twin of x by its rule, within 1e-6 of its magnitude, each min counted apart since a sub-block's term subtracts it.
bool twin_close(uint32_t type, const std::vector<uint8_t>& w, const std::vector<float>& wd, const std::vector<float>& x, size_t nin, size_t o, size_t r, float L, float got) {
    const quant::StorageType& st = *quant::storage_type(type);
    double sum = 0, mag = 0;
    float d;
    int16_t q[32];
    for (size_t b = 0; b < nin / 32; ++b) {
        twin_rule(x.data() + r * nin + b * 32, d, q, L);
        const float dm = k_min(type, w.data() + (o * nin + b * 32) / st.block_size * st.type_size, int(b % (st.block_size / 32)));
        for (size_t j = 0; j < 32; ++j) {
            const double xv = double(d) * q[j], wv = wd[o * nin + b * 32 + j];
            sum += wv * xv;
            mag += (std::fabs(wv) + 2 * std::fabs(double(dm))) * std::fabs(xv);
        }
    }
    return std::fabs(got - sum) <= 1e-6 * mag + 1e-30;
}

// Whether the backend's row_form takes the quarter form's builds for a row of nin values, restated from its rule: for Q5_0 and Q5_1, where four lanes a block fill the cluster's steps fuller than one lane a block, one lane on a tie.
bool quarter_form(uint32_t type, size_t nin) {
    if (type != quant::GGML_TYPE_Q5_0 && type != quant::GGML_TYPE_Q5_1) return false;
    const size_t W = rocm_device(0).caps.subgroup_size, blocks = nin / 32;
    const auto span = [W](size_t units) {
        size_t cl = 1;
        while (cl < W && cl < units) cl *= 2;
        return (units + cl - 1) / cl * cl;
    };
    return 4 * blocks * span(blocks) > blocks * span(4 * blocks);
}

// Widths whose rows take every lane cluster a type's rule gives (kernels/int16.hip, k_rows): lane units below the wave, at the Q4_K and Q5_K cap, above it and not a multiple of the cluster.
// Q5_0's and Q5_1's also lie each side of the lane form's switches on a 64-lane wave (one lane at 1024 and 2048, quarter at 1056 and 2080); 37 outputs, which no block's rows divide.
std::vector<std::array<size_t, 2>> cluster_shapes(uint32_t type) {
    const bool k = quant::storage_type(type)->block_size == 256;
    const std::vector<size_t> widths = k ? std::vector<size_t>{256, 512, 768, 1024, 4096} : std::vector<size_t>{32, 64, 96, 640, 1024, 1056, 2048, 2080, 4096};
    std::vector<std::array<size_t, 2>> out;
    for (size_t nin : widths) out.push_back({nin, 37});
    return out;
}

// The twin products of the K-quants, Q5_0, Q5_1, IQ4_NL and MXFP4 at Flash-Next's shapes in the F16 and int8 classes, extents on both sides of each crossover, each output against a double sum over the twin.
// The backend's launches name the build or tile that ran, and a row of a call is the row alone or in another call of its class, every generated column of up to 64; the cluster widths (cluster_shapes) take the generated rows alone.
void twin_types(Backend& g, Backend& c) {
    const uint32_t kinds[] = {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
    const size_t cols[] = {1, 2, 3, 5, 8, 9, 16, 17, 33}, extents[] = {2, 33, 64, 65, 320};
    for (uint32_t type : kinds) {
        std::string tname = quant::storage_type(type)->name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        std::vector<std::array<size_t, 2>> shapes = flash_shapes(type);
        const size_t flash = shapes.size();
        for (const auto& sh : cluster_shapes(type)) shapes.push_back(sh);
        if (type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_Q5_1) {
            size_t quarter = 0;
            for (const auto& sh : shapes) quarter += quarter_form(type, sh[0]);
            require(quarter > 0 && quarter < shapes.size(), tname + "'s widths take both lane forms");
        }
        for (size_t si = 0; si < shapes.size(); ++si) {
            const auto& sh = shapes[si];
            const size_t nin = sh[0], nout = sh[1], nmax = 320;
            const std::string at = tname + " " + std::to_string(nin) + "->" + std::to_string(nout);
            const std::vector<uint8_t> w = typed_matrix(type, nin, nout, uint32_t(type * 17 + nin + nout));
            const std::vector<float> x = row_classes::values(nin * nmax, uint32_t(nin + 11));
            // The CPU's decoded rows, the embedding of every row.
            std::vector<uint32_t> ids(nout);
            for (size_t i = 0; i < nout; ++i) ids[i] = uint32_t(i);
            auto cw = c.adopt(w.data(), w.size()), ce = c.alloc(nout * nin * 4);
            c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, nout, ids.data(), nout);
            const std::vector<float> wd = read(c, *ce, nout * nin);
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * nmax * 4), y1 = g.alloc(nout * nmax * 4);
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                const std::string form = dt == Dtype::int8 ? "8" : "16", cls = at + ", " + dtype_name(dt);
                const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
                const auto run = [&](size_t n, size_t extent, Buffer& y, size_t x0) {
                    const RowRun rr{n, extent};
                    rocm_take_launches(g);
                    MatrixPaths paths;
                    g.swap_matrix_paths(paths);
                    g.matmul({type, {wb.get(), 0}}, {xb.get(), x0 * nin}, {&y, 0}, nin, nout, n, {&rr, 1}, dt);
                    g.swap_matrix_paths(paths);
                    require(paths.take() == std::vector<std::string>{dt == Dtype::int8 ? "block-int8" : "block-int16"}, "a " + cls + " product records its twin's path");
                    std::map<std::string, size_t> mine;
                    for (const auto& [k, v] : rocm_take_launches(g))
                        if (k.rfind("llmx_" + tname, 0) == 0 || k.rfind("llmx_matmul", 0) == 0 || k.rfind("llmx_q8_0", 0) == 0) mine[k] = v;
                    return mine;
                };
                const auto check = [&](const std::vector<float>& y, size_t n, const std::string& what) {
                    bool ok = true;
                    for (size_t r : {size_t(0), n / 2, n - 1})
                        for (size_t o = 0; o < nout && ok; ++o) ok = twin_close(type, w, wd, x, nin, o, r, L, y[r * nout + o]);
                    require(ok, what + " within 1e-6 of a double sum over the twin");
                };
                // Generated rows: each build by the columns a launch takes, sixteen at most, and each column the column alone.
                for (size_t n : cols) {
                    std::map<std::string, size_t> want;
                    for (size_t left = n; left; left -= std::min<size_t>(left, 16)) {
                        const size_t m = std::min<size_t>(left, 16), b = m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : m <= 8 ? 8 : 16;
                        ++want["llmx_" + tname + (quarter_form(type, nin) ? "_quarter" : "") + "_rows" + form + "_" + std::to_string(b)];
                    }
                    const std::string what = "a " + cls + " call of " + std::to_string(n) + " generated rows";
                    require(run(n, 1, *yb, 0) == want, what + " runs its row builds alone");
                    const std::vector<float> all = read(g, *yb, nout * n);
                    check(all, n, what);
                }
                // Every column of 1 to 64 generated rows, as one run and as one run a row as a decode pass of n sequences gives them, is the column alone, bit for bit.
                {
                    const size_t most = 64;
                    std::vector<float> one(most * nout);
                    for (size_t r = 0; r < most; ++r) {
                        run(1, 1, *y1, r);
                        g.read(*y1, 0, one.data() + r * nout, nout * 4);
                    }
                    bool alone = true;
                    for (size_t n = 1; n <= most; ++n) {
                        run(n, 1, *yb, 0);
                        alone = alone && std::memcmp(read(g, *yb, nout * n).data(), one.data(), nout * n * 4) == 0;
                        std::vector<RowRun> each(n);
                        for (size_t r = 0; r < n; ++r) each[r] = {r + 1, 1};
                        g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {each.data(), n}, dt);
                        alone = alone && std::memcmp(read(g, *yb, nout * n).data(), one.data(), nout * n * 4) == 0;
                    }
                    require(alone, "a " + cls + " call of 1 to 64 generated rows: every column is the column alone, bit for bit, as one run and as one run a row");
                }
                // Prompt rows: the tile, and the first rows of a prompt the same bits in a call of fewer of its rows.
                for (size_t e : extents) {
                    if (si >= flash) break;
                    const std::string what = "a " + cls + " prompt of " + std::to_string(e) + " rows";
                    const std::map<std::string, size_t> ran = run(e, e, *yb, 0);
                    require(ran.size() == 1 && ran.begin()->first == "llmx_" + tname + "_tile" + form, what + " runs the tile alone");
                    const std::vector<float> all = read(g, *yb, nout * e);
                    check(all, e, what);
                    const size_t part = (e + 1) / 2;
                    run(part, e, *y1, 0);
                    require(std::memcmp(read(g, *y1, nout * part).data(), all.data(), nout * part * 4) == 0, what + ": its first " + std::to_string(part) + " rows in a call of their own, bit for bit");
                }
            }
        }
    }
}

// Printed, never asserted: the exact products at Qwen3.8-Flash-Next's F32 and BF16 shapes (the router, the hyper-connection maps), over 1, 2, 3, 4, 8, 9 and 293 rows, in wall time a call.
void exact_rates(Backend& g) {
    struct Shape { const char* name; uint32_t type; size_t nin, nout; };
    const Shape shapes[] = {{"flash router", quant::GGML_TYPE_F32, 2560, 512}, {"flash hc_pre", quant::GGML_TYPE_F32, 10240, 24}, {"flash hc_post", quant::GGML_TYPE_F32, 10240, 4},
                            {"flash ssm_alpha", quant::GGML_TYPE_F32, 2560, 24}, {"flash shexp gate", quant::GGML_TYPE_F32, 2560, 1},
                            {"flash bf16 512", quant::GGML_TYPE_BF16, 2560, 512}, {"flash bf16 128", quant::GGML_TYPE_BF16, 2560, 128}};
    std::cout << "backend-rocm: exact rates; type, shape (in x out), rows: us a call, kernels\n";
    for (const Shape& s : shapes) {
        const std::vector<float> x = row_classes::values(s.nin * 293, 901), v = row_classes::values(s.nin * s.nout, 902);
        std::vector<uint8_t> w(quant::row_bytes(s.type, s.nin, s.nout));
        for (size_t i = 0; i < v.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &v[i], 4);
            if (s.type == quant::GGML_TYPE_F32) std::memcpy(&w[4 * i], &u, 4);
            else w[2 * i] = uint8_t(u >> 16), w[2 * i + 1] = uint8_t(u >> 24);
        }
        auto xb = g.adopt(x.data(), x.size() * 4), wb = g.adopt(w.data(), w.size()), y = g.alloc(s.nout * 293 * 4);
        for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(8), size_t(9), size_t(293)}) {
            const auto call = [&] { g.matmul({s.type, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, s.nin, s.nout, n, {}, Dtype::f32); };
            for (int i = 0; i < 3; ++i) call();
            g.sync();
            rocm_take_launches(g);
            const int iters = 20;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) call();
            g.sync();
            const double us = 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::string ran;
            for (const auto& [k, n2] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + k;
            std::printf("backend-rocm: exact rate %s %s (%zu x %zu), %zu: %.1f us, %s\n", quant::storage_type(s.type)->name, s.name, s.nin, s.nout, n, us, ran.c_str());
        }
    }
}

// Printed, never asserted: the twin types' products at Qwen3.8-Flash-Next's projection shapes, generated rows of 1, 16 and 64 columns and a 320-row prompt, in wall time a call over a synchronized loop, with the kernels that ran.
void rates(Backend& g) {
    struct Shape { const char* name; size_t nin, nout; };
    const Shape shapes[] = {{"flash attn_qkv", 2560, 10240}, {"flash ssm_out", 6144, 2560}, {"flash hc_down", 10240, 320}, {"flash shexp_gate", 2560, 640}};
    const uint32_t kinds[] = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
    const size_t widths[] = {1, 16, 64, 320};
    std::cout << "backend-rocm: rates; type, dtype, shape (in x out), rows (320 a prompt): us a call, TMAC/s, kernels\n";
    for (const Shape& s : shapes) {
        const std::vector<float> x = row_classes::values(s.nin * 320, 900);
        auto xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(s.nout * 320 * 4);
        for (uint32_t type : kinds) {
            const std::vector<uint8_t> w = typed_matrix(type, s.nin, s.nout, 7);
            auto wb = g.adopt(w.data(), w.size());
            for (Dtype dt : {Dtype::f16, Dtype::int8})
                for (size_t n : widths) {
                    const RowRun rr{n, n == 320 ? n : 1};
                    const auto call = [&] { g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, s.nin, s.nout, n, {&rr, 1}, dt); };
                    for (int i = 0; i < 3; ++i) call();
                    g.sync();
                    rocm_take_launches(g);
                    const int iters = 20;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int i = 0; i < iters; ++i) call();
                    g.sync();
                    const double us = 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
                    std::string ran;
                    for (const auto& [k, v] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + k;
                    std::printf("backend-rocm: rate %s %s %s (%zu x %zu), %zu: %.1f us, %.3f TMAC/s, %s\n", quant::storage_type(type)->name, dtype_name(dt), s.name, s.nin, s.nout,
                                n, us, double(s.nin) * s.nout * n / us / 1e6, ran.c_str());
                }
        }
    }
    std::fflush(stdout);
}

// The median wall time of `call` in us over seven rounds of 100 calls after a warm-up, which the decode rates print; a round's spread is printed beside it so a busy machine shows.
std::pair<double, double> median_us(Backend& g, const std::function<void()>& call) {
    for (int i = 0; i < 5; ++i) call();
    g.sync();
    std::vector<double> t;
    for (int round = 0; round < 7; ++round) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) call();
        g.sync();
        t.push_back(1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 100);
    }
    std::sort(t.begin(), t.end());
    return {t[3], t[6] - t[0]};
}

// Decode calls at Flash-Next's shapes, printed, never asserted: one generated row's dense Q8_0 products in both twin classes, from weights outside the L2 cache, the gated delta rule and the router.
// Each with its median us a call, the kernels taken and, for the products, an FNV-1a hash of the outputs, which two builds compare.
void decode_rates(Backend& g) {
    struct Shape { const char* name; size_t nin, nout; };
    const Shape shapes[] = {{"hc_up", 512, 10240}, {"hc_up_320", 320, 10240}, {"hc_down", 10240, 320}, {"shexp_down", 768, 2560}, {"shexp_gate_up", 2560, 1280},
                            {"attn_gate", 2560, 6144}, {"attn_qkv", 2560, 10240}, {"ssm_out", 6144, 2560}, {"narrow_odd", 576, 333}};
    const auto kernels = [&] {
        std::string ran;
        for (const auto& [k, v] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + k;
        return ran;
    };
    for (const Shape& s : shapes) {
        const std::vector<float> x = row_classes::values(s.nin, 901);
        const std::vector<uint8_t> w = typed_matrix(quant::GGML_TYPE_Q8_0, s.nin, s.nout, 8);
        // Copies of the weights past the L2 cache, taken in turn, so each call reads its weights from memory as a decode step does.
        std::vector<BufferPtr> wb;
        for (size_t i = 0; i < std::max<size_t>(1, (size_t(32) << 20) / w.size()); ++i) wb.push_back(g.adopt(w.data(), w.size()));
        auto xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(s.nout * 4);
        for (Dtype dt : {Dtype::f16, Dtype::int8}) {
            const RowRun rr{1, 1};
            rocm_take_launches(g);
            size_t turn = 0;
            const auto [us, spread] = median_us(g, [&] {
                g.matmul({quant::GGML_TYPE_Q8_0, {wb[turn++ % wb.size()].get(), 0}}, {xb.get(), 0}, {y.get(), 0}, s.nin, s.nout, 1, {&rr, 1}, dt);
            });
            const std::string ran = kernels();
            const std::vector<float> out = read(g, *y, s.nout);
            uint64_t h = 1469598103934665603ull;
            for (const float f : out) {
                uint32_t u;
                std::memcpy(&u, &f, 4);
                for (int i = 0; i < 4; ++i) h = (h ^ ((u >> (8 * i)) & 0xff)) * 1099511628211ull;
            }
            std::printf("backend-rocm: decode rate Q8_0 %s %s (%zu x %zu): %.1f us (spread %.1f), %s, hash %016llx\n", dtype_name(dt), s.name, s.nin, s.nout, us, spread, ran.c_str(),
                        (unsigned long long)h);
        }
    }
    // The gated delta rule of one sequence at Flash-Next's linear-attention heads, 1 and 4 rows a call and a 9-row prompt slice, on a state with a history.
    const StateShape sh{16, 48, 128, 128};
    const size_t C = sh.channels();
    auto storage = g.state_alloc(1, 1, sh);
    for (size_t nq : {size_t(1), size_t(4), size_t(9)}) {
        const std::vector<float> u = row_classes::values(nq * C, 911, 0.5f), ab = row_classes::values(2 * nq * sh.v_heads, 912), adt = row_classes::values(2 * sh.v_heads, 913, 0.1f);
        auto ub = g.adopt(u.data(), u.size() * 4), abb = g.adopt(ab.data(), ab.size() * 4), adb = g.adopt(adt.data(), adt.size() * 4), ob = g.alloc(nq * sh.v_heads * sh.v_dim * 4);
        const StateView view{storage.get(), 0, 0, 5, nq};
        rocm_take_launches(g);
        const auto [us, spread] = median_us(g, [&] {
            g.gated_delta_rule({ob.get(), 0}, {ub.get(), 0}, {abb.get(), 0}, {abb.get(), nq * sh.v_heads}, {adb.get(), 0}, {adb.get(), sh.v_heads}, 0, &view, 1);
        });
        std::printf("backend-rocm: decode rate delta rule %zu rows (%zu V heads of %zu x %zu): %.1f us (spread %.1f), %s\n", nq, sh.v_heads, sh.k_dim, sh.v_dim, us, spread, kernels().c_str());
    }
    // The router of one generated row over 256 and 512 experts.
    for (const auto& [n_expert, k] : {std::pair<size_t, size_t>{256, 8}, {512, 10}}) {
        const std::vector<float> sc = row_classes::values(n_expert, 921, 4.0f);
        auto sb = g.adopt(sc.data(), sc.size() * 4), ib = g.alloc(k * 4), wb = g.alloc(k * 4);
        rocm_take_launches(g);
        const auto [us, spread] = median_us(g, [&] { g.route_experts({sb.get(), 0}, 1, n_expert, k, backend::RouteRule::softmax(true), {ib.get(), 0}, {wb.get(), 0}); });
        // A hash of 256 rows' picks and weights, which two builds of the router compare bit for bit.
        const size_t hr = 256;
        const std::vector<float> hs = row_classes::values(hr * n_expert, 922, 4.0f);
        auto hsb = g.adopt(hs.data(), hs.size() * 4), hib = g.alloc(hr * k * 4), hwb = g.alloc(hr * k * 4);
        g.route_experts({hsb.get(), 0}, hr, n_expert, k, backend::RouteRule::softmax(true), {hib.get(), 0}, {hwb.get(), 0});
        std::vector<float> out = read(g, *hib, hr * k), wout = read(g, *hwb, hr * k);
        out.insert(out.end(), wout.begin(), wout.end());
        uint64_t h = 1469598103934665603ull;
        for (const float f : out) {
            uint32_t u;
            std::memcpy(&u, &f, 4);
            for (int i = 0; i < 4; ++i) h = (h ^ ((u >> (8 * i)) & 0xff)) * 1099511628211ull;
        }
        std::printf("backend-rocm: decode rate route %zu of %zu experts: %.1f us (spread %.1f), %s, hash %016llx\n", k, n_expert, us, spread, kernels().c_str(), (unsigned long long)h);
    }
    std::fflush(stdout);
}

void ops(Backend& g, Backend& c) {
    // Q8_0 rows decoded exactly: a matmul and an embedding against the CPU.
    {
        const size_t nin = 4096, nout = 37, rows = 3;
        const std::vector<uint8_t> w = q8_matrix(nin, nout, 21);
        const std::vector<float> x = row_classes::values(nin * rows, 22);
        const uint32_t ids[3] = {36, 0, 5};
        std::vector<std::vector<float>> prod(2), emb(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * 4), yb = b.alloc(nout * rows * 4), eb = b.alloc(3 * nin * 4);
            b.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, {}, Dtype::f32);
            prod[k] = read(b, *yb, nout * rows);
            b.embed({eb.get(), 0}, {quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, nin, nout, ids, 3);
            emb[k] = read(b, *eb, 3 * nin);
        }
        close(prod[0], prod[1], "a Q8_0 matmul against the CPU");
        require(same(emb[0], emb[1]), "a Q8_0 embedding gives the CPU's rows, bit for bit");
    }
    // Products: odd widths, several rows, and each row of a call equal to the row alone, bit for bit.
    for (size_t nin : {size_t(1000), size_t(4096)}) {
        const size_t nout = 37, rows = 5;
        const std::vector<float> w = row_classes::values(nin * nout, 2), x = row_classes::values(nin * rows, 3), base = row_classes::values(nout * rows, 4);
        std::vector<std::vector<float>> out(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto wb = b.adopt(w.data(), w.size() * 4), xb = b.adopt(x.data(), x.size() * 4), yb = b.adopt(base.data(), base.size() * 4);
            b.matmul_add({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows);
            out[k] = read(b, *yb, nout * rows);
            if (k == 0) {
                auto ya = b.alloc(nout * rows * 4);
                b.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {ya.get(), 0}, nin, nout, rows);
                const std::vector<float> all = read(b, *ya, nout * rows);
                bool alone = true;
                for (size_t r = 0; r < rows; ++r) {
                    auto y1 = b.alloc(nout * 4);
                    b.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), r * nin}, {y1.get(), 0}, nin, nout, 1);
                    alone = alone && std::memcmp(read(b, *y1, nout).data(), all.data() + r * nout, nout * 4) == 0;
                }
                require(alone, "each row of a matmul " + std::to_string(nin) + " wide is the row alone, bit for bit");
            }
        }
        close(out[0], out[1], "matmul_add " + std::to_string(nin) + " wide against the CPU");
    }
    // Norms and elementwise ops.
    {
        const size_t rows = 3, n = 1000, stride = 1100;
        const std::vector<float> x = row_classes::values(rows * stride, 5), w = row_classes::values(n, 6), y = row_classes::values(rows * stride, 7);
        std::vector<std::vector<float>> norm(2), sm(2), ad(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto xb = b.adopt(x.data(), x.size() * 4), wb = b.adopt(w.data(), w.size() * 4), yb = b.adopt(y.data(), y.size() * 4), ob = b.alloc(x.size() * 4);
            b.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
            norm[k] = read(b, *ob, rows * stride);
            b.silu_mul({ob.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, rows * stride, 0.0f);
            sm[k] = read(b, *ob, rows * stride);
            b.add({ob.get(), 0}, {yb.get(), 0}, rows * stride);
            ad[k] = read(b, *ob, rows * stride);
        }
        close(norm[0], norm[1], "rms_norm_rows against the CPU");
        close(sm[0], sm[1], "silu_mul against the CPU");
        close(ad[0], ad[1], "add against the CPU");
    }
    // The rope with and without a norm, positions out of order.
    for (bool weighted : {true, false}) {
        const size_t rows = 4, heads = 3, head_dim = 128, half = 64, positions = 50;
        const std::vector<float> x = row_classes::values(rows * heads * head_dim, 8), w = row_classes::values(head_dim, 9);
        std::vector<float> cos(positions * half), sin(positions * half);
        for (size_t p = 0; p < positions; ++p)
            for (size_t j = 0; j < half; ++j) {
                const float a = (float)p * std::pow(10000.0f, -2.0f * (float)j / (float)head_dim);
                cos[p * half + j] = std::cos(a);
                sin[p * half + j] = std::sin(a);
            }
        const uint32_t pos[rows] = {7, 0, 49, 3};
        std::vector<std::vector<float>> out(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto xb = b.adopt(x.data(), x.size() * 4), wb = b.adopt(w.data(), w.size() * 4);
            auto cb = b.adopt(cos.data(), cos.size() * 4), sb = b.adopt(sin.data(), sin.size() * 4);
            b.norm_rope_partial({xb.get(), 0}, {xb.get(), 0}, rows, heads * head_dim, head_dim, heads, head_dim, 2 * half,
                                weighted ? CSlice{wb.get(), 0} : CSlice{}, 1e-6f, {cb.get(), 0}, {sb.get(), 0}, pos, backend::RopeForm{});
            out[k] = read(b, *xb, x.size());
        }
        close(out[0], out[1], std::string("norm_rope_partial ") + (weighted ? "with" : "without") + " a norm against the CPU");
    }
    // Rows of a table: embed and gather_rows.
    {
        const size_t width = 300, nrows = 20;
        const std::vector<float> t = row_classes::values(width * nrows, 10);
        const uint32_t ids[4] = {19, 0, 7, 7};
        auto tb = g.adopt(t.data(), t.size() * 4), ob = g.alloc(4 * width * 4);
        g.embed({ob.get(), 0}, {quant::GGML_TYPE_F32, {tb.get(), 0}}, width, nrows, ids, 4);
        std::vector<float> want;
        for (uint32_t id : ids) want.insert(want.end(), t.begin() + id * width, t.begin() + (id + 1) * width);
        require(same(read(g, *ob, 4 * width), want), "embed gives the table's rows");
        g.gather_rows({ob.get(), 0}, {tb.get(), 0}, width, ids, 4);
        require(same(read(g, *ob, 4 * width), want), "gather_rows gives the source's rows");
        const uint32_t past[2] = {20, 3};
        g.embed({ob.get(), 0}, {quant::GGML_TYPE_F32, {tb.get(), 0}}, width, nrows, past, 2);
        want.assign(width, 0.0f);
        want.insert(want.end(), t.begin() + 3 * width, t.begin() + 4 * width);
        require(same(read(g, *ob, 2 * width), want), "an embedding id past the table writes a zero row beside the table's rows");
    }
}

// KV writes and attention after histories across block edges, both cache types and two sequences in one call, against the CPU; then attention's rows, alone and beside others, bit for bit.
// Heads 128 and 256 wide take the per-row body, its only check at 256 (Flash-Next's MTP block, qwen35's full attention), split from 33 tokens on and at the profile's limit by 3000.
void attention(Backend& g, Backend& c) {
    for (const int head_dim : {128, 256})
    for (KVType kt : {KVType::f32, KVType::f16})
        for (size_t hist : {size_t(0), size_t(70), size_t(130), size_t(3000)}) {
            const int n_head = 4, n_head_kv = 2;
            const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
            const size_t rows = 3;
            const std::vector<float> keys = row_classes::values((hist + rows) * kvw * 2, 11), vals = row_classes::values((hist + rows) * kvw * 2, 12);
            const std::vector<float> q = row_classes::values(rows * 2 * qw, 13, 4.0f);
            std::vector<std::vector<float>> out(2);
            Backend* bs[2] = {&g, &c};
            for (int k = 0; k < 2; ++k) {
                Backend& b = *bs[k];
                auto kb = b.adopt(keys.data(), keys.size() * 4), vb = b.adopt(vals.data(), vals.size() * 4);
                auto qb = b.adopt(q.data(), q.size() * 4), ob = b.alloc(rows * 2 * qw * 4);
                auto st = b.kv_alloc(1, n_head_kv, head_dim, 6400, kt, kt);
                infer::BlockPool pool;
                pool.configure(st->max_blocks());
                infer::KVSequence s1(&pool, b.kv_layout().block_tokens), s2(&pool, b.kv_layout().block_tokens);
                for (infer::KVSequence* s : {&s1, &s2}) {
                    if (!hist) continue;
                    s->prepare(hist);
                    const KVView h = s->view(st.get());
                    const size_t at = s == &s1 ? 0 : (hist + rows) * kvw;
                    b.kv_write(0, &h, 1, {kb.get(), at}, {vb.get(), at});
                    s->commit();
                }
                s1.prepare(rows);
                s2.prepare(rows);
                const KVView v[2] = {s1.view(st.get()), s2.view(st.get())};
                std::vector<float> kr, vr;
                for (size_t sq = 0; sq < 2; ++sq) {
                    const size_t at = sq * (hist + rows) * kvw + hist * kvw;
                    kr.insert(kr.end(), keys.begin() + at, keys.begin() + at + rows * kvw);
                    vr.insert(vr.end(), vals.begin() + at, vals.begin() + at + rows * kvw);
                }
                auto krb = b.adopt(kr.data(), kr.size() * 4), vrb = b.adopt(vr.data(), vr.size() * 4);
                b.kv_write(0, v, 2, {krb.get(), 0}, {vrb.get(), 0});
                if (k == 0) rocm_take_launches(b);
                b.attention({qb.get(), 0}, 0, v, 2, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                out[k] = read(b, *ob, rows * 2 * qw);
                if (k == 0) {
                    const auto ran = rocm_take_launches(b);
                    const std::string body = head_dim == 256 ? "llmx_attention_vec256" : "llmx_attention_vec";
                    require(ran.count(body) || ran.count(body + "_major"), "attention of heads " + std::to_string(head_dim) + " wide runs the per-row body");
                    // The first sequence alone gives its rows' bits.
                    auto o1 = b.alloc(rows * qw * 4);
                    b.attention({qb.get(), 0}, 0, v, 1, {o1.get(), 0}, n_head, n_head_kv, head_dim);
                    require(std::memcmp(read(b, *o1, rows * qw).data(), out[0].data(), rows * qw * 4) == 0,
                            "attention of a sequence alone gives its rows beside another, bit for bit");
                }
                s1.abort();
                s2.abort();
            }
            close(out[0], out[1], std::string("attention, heads ") + std::to_string(head_dim) + " wide, " + kv_type_name(kt) + " cache, history " + std::to_string(hist) + ", two sequences, against the CPU");
        }
}

// The twin's rule as docs/STATUS.md gives it, on the host, at L levels: a block's scale and its values, 32767 for block-int16 and 127 for block-int8.
void twin_rule(const float* x, float& d, int16_t* q, float L) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
    if (amax == 0.0f) {
        d = 0.0f;
        for (int j = 0; j < 32; ++j) q[j] = 0;
        return;
    }
    int e = 0;
    std::frexp(amax, &e);
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = std::ldexp(amax, s);
    const float k = L / an;
    for (int j = 0; j < 32; ++j) {
        const float r = std::ldexp(x[j], s) * k;
        const float t = std::min(std::floor(std::fabs(r) + 0.5f), L);
        q[j] = (int16_t)(r < 0.0f ? -t : t);
    }
    d = std::ldexp(an / L, -s);
}

// A Q8_0 row of `nin` values whose block b holds value 1 at position j and scale 1, zero elsewhere: its product reads the twin's value d q at (b, j) alone.
std::vector<uint8_t> one_hot_q8(size_t nin, size_t b, size_t j) {
    std::vector<uint8_t> row(nin / 32 * 34, 0);
    const uint16_t one = 0x3c00;
    row[b * 34] = uint8_t(one);
    row[b * 34 + 1] = uint8_t(one >> 8);
    row[b * 34 + 2 + j] = 1;
    return row;
}

// The one-column Q8_0 decode row of one wave of W lanes, on the host, in the kernels' order, then the xor tree from W / 2.
// The half-block layout (int8 twin, an even block count) takes two lanes a block, each lane's half-block dots scaled in block order, and the quarter layout four lanes a block.
float q8_row_host(const uint8_t* row, const std::vector<float>& d, const std::vector<int16_t>& q, size_t blocks, bool half, unsigned W) {
    std::vector<float> acc(W, 0.0f);
    const size_t parts = half ? 2 : 4, per = 32 / parts;
    for (size_t u = 0; u < blocks * parts; ++u) {
        const size_t b = u / parts, p = u % parts;
        const uint8_t* block = row + b * 34;
        int s = 0;
        for (size_t j = p * per; j < (p + 1) * per; ++j) s += int(int8_t(block[2 + j])) * q[b * 32 + j];
        const float dw = f16_to_f32(uint16_t(block[0] | block[1] << 8)), dd = dw * d[b];
        float& a = acc[half ? (b % (W / 2)) * 2 + p : u % W];
        a = std::fmaf(dd, float(s), a);
    }
    for (unsigned m = W / 2; m; m /= 2) {
        std::vector<float> next(W);
        for (unsigned l = 0; l < W; ++l) next[l] = acc[l] + acc[l ^ m];
        acc = next;
    }
    return acc[0];
}

// Each one-column Q8_0 decode build against the host's one-row build bit for bit, the paired build (two rows a wave, rows of at most 16 blocks) among them, whose idle half the one-row tree adds as +0.
// The rows pin the zero's sign: all-zero weights, a negative scale against a zero integer dot (dw * xd = -0), and lanes whose partials cancel to an exact zero from nonzero terms.
void q8_decode_bits(Backend& g) {
    const unsigned W = rocm_device(0).caps.subgroup_size;
    const size_t nout = 33;
    DeviceProfile& profile = rocm_profile(g);
    const uint32_t keep = profile.q8_decode_two_rows_from;
    const size_t four = profile.q8_decode_four_steps_from;
    // Row widths reach both edges of the step rule (63 to 65 and 254 to 257 blocks) and step tails, and output counts both edges of the two-row rule at both profiles' values.
    for (size_t blocks0 : {size_t(8), size_t(16), size_t(18), size_t(20), size_t(24), size_t(33), size_t(40), size_t(63), size_t(64), size_t(65), size_t(80), size_t(191), size_t(192),
                           size_t(193), size_t(254), size_t(255), size_t(256), size_t(257)}) {
        const size_t nin = blocks0 * 32;
        const size_t blocks = nin / 32;
        std::vector<uint8_t> w = q8_matrix(nin, nout, uint32_t(nin + 5));
        const auto set_scale = [&](size_t o, size_t b, float s) {
            const uint16_t h = f32_to_f16(s);
            w[(o * blocks + b) * 34] = uint8_t(h), w[(o * blocks + b) * 34 + 1] = uint8_t(h >> 8);
        };
        for (size_t b = 0; b < blocks; ++b) {
            std::memset(&w[(0 * blocks + b) * 34], 0, 34);          // row 0: all zero
            std::memset(&w[(1 * blocks + b) * 34 + 2], 0, 32);      // row 1: values zero under negative scales
            set_scale(1, b, -0.25f - 0.01f * float(b));
            if (b % 2) {
                // Row 2: each odd block the even block before it negated, so on a block-periodic twin a lane's partial cancels its partner's exactly.
                std::memcpy(&w[(2 * blocks + b) * 34], &w[(2 * blocks + b - 1) * 34], 34);
                for (size_t j = 0; j < 32; ++j) w[(2 * blocks + b) * 34 + 2 + j] = uint8_t(-int8_t(w[(2 * blocks + b) * 34 + 2 + j]));
            }
        }
        const std::vector<float> rnd = row_classes::values(nin, uint32_t(nin + 6));
        std::vector<float> periodic(nin);
        for (size_t i = 0; i < nin; ++i) periodic[i] = rnd[i % 32];
        // Rows past the special ones repeat the first 33, so a call of many outputs holds them too.
        const size_t nmax = 4097;
        const std::vector<uint8_t> first = w;
        for (size_t o = nout; o < nmax; ++o) w.insert(w.end(), first.begin() + (o % nout) * blocks * 34, first.begin() + (o % nout + 1) * blocks * 34);
        auto wb = g.adopt(w.data(), w.size()), yb = g.alloc(nmax * 4);
        const std::vector<float>* inputs[] = {&rnd, &periodic};
        for (const std::vector<float>* xv : inputs) {
            auto xb = g.adopt(xv->data(), nin * 4);
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                const bool half = dt == Dtype::int8 && blocks % 2 == 0;
                const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
                std::vector<float> d(blocks);
                std::vector<int16_t> q(nin);
                for (size_t b = 0; b < blocks; ++b) twin_rule(xv->data() + b * 32, d[b], q.data() + b * 32, L);
                for (const auto& [from, n] : std::vector<std::pair<uint32_t, size_t>>{{2560, 1}, {2560, 2}, {2560, nout}, {2560, 2559}, {2560, 2560}, {2560, 4097}, {4096, 4095}, {4096, 4096}}) {
                    if (dt == Dtype::f16 && from == 4096) continue;
                    const RowRun rr{1, 1};
                    profile.q8_decode_two_rows_from = from;
                    rocm_take_launches(g);
                    g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, n, 1, {&rr, 1}, dt);
                    profile.q8_decode_two_rows_from = keep;
                    const std::vector<float> y = read(g, *yb, n);
                    std::string ran;
                    for (const auto& [k, v] : rocm_take_launches(g))
                        if (k.rfind("llmx_q8_0", 0) == 0) ran += k;
                    // The half-block rows take one row a wave for few outputs and two for many, paired where a row has at most 16 blocks.
                    // A row under two spans of blocks takes one step a load, and at two rows a wave from the profile's four-step width four.
                    const bool narrow = blocks < W;
                    const std::string want = !half                   ? "llmx_q8_0_rows" + std::string(dt == Dtype::int8 ? "8" : "16") + "_1"
                                             : W == 64 && blocks <= 16 ? "llmx_q8_0_half8_1p"
                                             : n < from              ? (narrow ? "llmx_q8_0_half8_1s1" : "llmx_q8_0_half8_1")
                                             : narrow                ? "llmx_q8_0_half8_1r2s1"
                                             : blocks >= four        ? "llmx_q8_0_half8_1r2s4"
                                                                     : "llmx_q8_0_half8_1r2";
                    const std::string what = "a Q8_0 decode row " + std::to_string(nin) + " wide, " + std::to_string(n) + " outputs from " + std::to_string(from) + ", " + dtype_name(dt) +
                                             (xv == &periodic ? ", periodic twin" : "");
                    require(ran == want, what + " runs " + want + " (ran " + ran + ")");
                    bool ok = true;
                    for (size_t o = 0; o < n; ++o) {
                        const float h = q8_row_host(w.data() + o * blocks * 34, d, q, blocks, half, W);
                        ok = ok && std::memcmp(&h, &y[o], 4) == 0;
                    }
                    require(ok, what + ": every row the one-row build's bits");
                    bool zero = true;
                    for (size_t o = 0; o < std::min<size_t>(n, 2); ++o) zero = zero && y[o] == 0.0f && !std::signbit(y[o]);
                    require(zero, what + ": the zero rows are +0");
                    if (xv == &periodic && n > 2 && blocks % 2 == 0) require(y[2] == 0.0f && !std::signbit(y[2]), what + ": the cancelling row is +0");
                }
            }
        }
    }
}

// One MXFP4 decode row on the host in the order every MXFP4 row kernel keeps: four units a block of eight values, unit u's terms chained in accumulator u % W.
// W is the power of two covering the units, at most the wave, and the accumulators are summed by the xor tree from W / 2.
float mxfp4_row_host(const uint8_t* row, const std::vector<float>& d, const int16_t* q, size_t blocks, unsigned wave) {
    unsigned W = 1;
    while (W < wave && W < blocks * 4) W *= 2;
    std::vector<float> acc(W, 0.0f);
    for (size_t u = 0; u < blocks * 4; ++u) {
        const size_t b = u / 4, t = u % 4;
        const uint8_t* block = row + b * 17;
        int s = 0;
        for (size_t j = 8 * t; j < 8 * t + 8; ++j) s += quant::FP4_VALUES[j < 16 ? block[1 + j] & 15 : block[1 + j - 16] >> 4] * q[b * 32 + j];
        const uint32_t bits = block[0] <= 1 ? 0x00200000u << block[0] : uint32_t(block[0] - 1) << 23;
        float ds;
        std::memcpy(&ds, &bits, 4);
        acc[u % W] = std::fmaf(d[b], ds * float(s), acc[u % W]);
    }
    for (unsigned m = W / 2; m; m /= 2) {
        std::vector<float> next(W);
        for (unsigned l = 0; l < W; ++l) next[l] = acc[l] + acc[l ^ m];
        acc = next;
    }
    return acc[0];
}

// Every MXFP4 decode build against the host's row bit for bit in both twins: dense rows of 1 to 16 columns, routed entries and the grouped builds of 2, 4 and 8.
// The widths take 4 to 640 units a row, on both sides of 32 and 64, with DeepSeek-V4.1-Flash's 2304 and 5120 and a 65-block row.
void mxfp4_decode_bits(Backend& g) {
    const unsigned wave = rocm_device(0).caps.subgroup_size;
    const uint32_t T = quant::GGML_TYPE_MXFP4;
    const size_t nout = 40, ne = 4, cols = 16, K = 2;
    for (size_t nin : {size_t(32), size_t(224), size_t(256), size_t(288), size_t(480), size_t(512), size_t(544), size_t(2080), size_t(2304), size_t(5120)}) {
        const size_t blocks = nin / 32, rowb = blocks * 17;
        // 64 experts of rows, so the routed calls over 64 have them all; the entries pick among the first 4.
        const std::vector<uint8_t> w = typed_matrix(T, nin, nout * 64, uint32_t(nin + 3));
        const std::vector<float> x = row_classes::values(nin * cols, uint32_t(nin + 4));
        std::vector<uint32_t> ids(cols * K);
        for (size_t c = 0; c < cols; ++c)
            for (size_t k = 0; k < K; ++k) ids[c * K + k] = uint32_t((c + 3 * k) % ne);
        const std::vector<float> wts(cols * K, 0.5f);
        auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(ids.data(), ids.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
        auto yb = g.alloc(nout * ne * cols * 4);
        for (Dtype dt : {Dtype::f16, Dtype::int8}) {
            const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
            std::vector<std::vector<float>> d(cols, std::vector<float>(blocks));
            std::vector<std::vector<int16_t>> q(cols, std::vector<int16_t>(nin));
            for (size_t c = 0; c < cols; ++c)
                for (size_t b = 0; b < blocks; ++b) twin_rule(x.data() + c * nin + b * 32, d[c][b], q[c].data() + b * 32, L);
            const std::string tw = dt == Dtype::int8 ? "8" : "16", at = std::to_string(nin) + " wide, " + dtype_name(dt);
            const auto bits = [&](const std::vector<float>& y, const std::vector<std::pair<size_t, size_t>>& rc, const std::string& what) {
                bool ok = true;
                for (size_t i = 0; ok && i < rc.size(); ++i) {
                    const float h = mxfp4_row_host(w.data() + rc[i].first * rowb, d[rc[i].second], q[rc[i].second].data(), blocks, wave);
                    ok = std::memcmp(&h, &y[i], 4) == 0;
                    if (!ok) std::fprintf(stderr, "  output %zu: %.9g against %.9g\n", i, y[i], h);
                }
                require(ok, what + ": every output the host row's bits");
            };
            const auto ran = [&](const std::string& build, const std::string& what) {
                const auto l = rocm_take_launches(g);
                require(l.count(build) > 0, what + " runs " + build);
            };
            for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(5), size_t(9), size_t(16)}) {
                const RowRun rr{n, 1};
                rocm_take_launches(g);
                g.matmul({T, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout * ne, n, {&rr, 1}, dt);
                const std::string what = "MXFP4 decode rows " + at + ", " + std::to_string(n) + " columns";
                ran("llmx_mxfp4_rows" + tw + "_" + std::to_string(n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : n <= 8 ? 8 : 16), what);
                std::vector<std::pair<size_t, size_t>> rc;
                for (size_t c = 0; c < n; ++c)
                    for (size_t o = 0; o < nout * ne; ++o) rc.push_back({o, c});
                bits(read(g, *yb, nout * ne * n), rc, what);
            }
            // 64 experts keep the entries below the grouped builds' break-even; over 4 the average entries an expert pick g2, g4 and g8.
            const std::pair<size_t, size_t> cases[] = {{64, 1}, {64, 6}, {4, 1}, {4, 6}, {4, 16}};
            for (const auto& [n_expert, n] : cases) {
                const RowRun rr{n, 1};
                const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, K, n_expert};
                rocm_take_launches(g);
                g.matmul_experts({Projection{{T, {wb.get(), 0}}, {yb.get(), 0}, nout}}, {xb.get(), 0}, nin, n, routing, {&rr, 1}, dt);
                const size_t avg = n * K / std::min(n_expert, n * K);
                const std::string build = "llmx_mxfp4_routed" + tw + (n_expert == 64 ? "" : avg <= 2 ? "_g2" : avg <= 4 ? "_g4" : "_g8");
                const std::string what = "MXFP4 routed decode " + at + ", " + std::to_string(n) + " columns over " + std::to_string(n_expert) + " experts";
                ran(build, what);
                std::vector<std::pair<size_t, size_t>> rc;
                for (size_t e = 0; e < n * K; ++e)
                    for (size_t o = 0; o < nout; ++o) rc.push_back({ids[e] * nout + o, e / K});
                bits(read(g, *yb, nout * n * K), rc, what);
            }
        }
    }
}

// A class's Q8_0 products, f16 on the block-int16 twin and int8 on the block-int8 one, against the twin's rule and a double oracle on the twin.
// Every decode column must be the column alone, and a prompt row the same bits in any call.
void twin_products(Backend& g, Dtype dt) {
    const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    // Twin: blocks of ordinary, tiny, huge, tied and zero values, read through one-hot rows on both kernels.
    const size_t nin = 32 * 6;
    std::vector<float> x = row_classes::values(nin * 2, 31);
    for (size_t j = 0; j < 32; ++j) {
        x[32 + j] *= 1e-35f;              // a block whose scale needs the power of two
        x[64 + j] *= 1e35f;               // and one whose largest value would overflow its reciprocal's product
        x[96 + j] = 0.0f;                 // a block of zeros
        x[128 + j] = j == 0 ? L : (float)j + 0.5f;   // exact halves: ties away from zero
    }
    x[160] = -L;
    x[161] = -0.5f;
    std::vector<uint8_t> w;
    for (size_t o = 0; o < nin; ++o) {
        const std::vector<uint8_t> row = one_hot_q8(nin, o / 32, o % 32);
        w.insert(w.end(), row.begin(), row.end());
    }
    auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nin * 2 * 4);
    bool rule = true;
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{2, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nin, 2, {run, 1}, dt);
        const std::vector<float> y = read(g, *yb, nin * 2);
        for (size_t r = 0; r < 2; ++r)
            for (size_t b = 0; b < nin / 32; ++b) {
                float d;
                int16_t q[32];
                twin_rule(x.data() + r * nin + b * 32, d, q, L);
                for (size_t j = 0; j < 32; ++j) {
                    const float want = d * (float)q[j];
                    rule = rule && std::memcmp(&y[r * nin + b * 32 + j], &want, 4) == 0;
                }
            }
    }
    require(rule, "the " + form + " twin is its rule, through the row kernel and the tile");
    // Products against a double oracle on the twin, rows 4096 wide, generated and prompt rows.
    const size_t width = 4096, nout = 40, rows = 21;
    const std::vector<uint8_t> m = q8_matrix(width, nout, 33);
    const std::vector<float> xs = row_classes::values(width * rows, 34);
    auto mb = g.adopt(m.data(), m.size()), xsb = g.adopt(xs.data(), xs.size() * 4), ys = g.alloc(nout * rows * 4);
    std::vector<std::vector<float>> twin_d(rows, std::vector<float>(width / 32));
    std::vector<std::vector<int16_t>> twin_q(rows, std::vector<int16_t>(width));
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < width / 32; ++b) twin_rule(xs.data() + r * width + b * 32, twin_d[r][b], twin_q[r].data() + b * 32, L);
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{rows, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, rows, {run, 1}, dt);
        const std::vector<float> y = read(g, *ys, nout * rows);
        bool close_ok = true;
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < nout; ++o) {
                double sum = 0.0, mag = 0.0;
                for (size_t b = 0; b < width / 32; ++b) {
                    const uint8_t* block = m.data() + (o * (width / 32) + b) * 34;
                    const double dw = f16_to_f32((uint16_t)(block[0] | block[1] << 8));
                    long isum = 0;
                    for (size_t j = 0; j < 32; ++j) isum += (long)(int8_t)block[2 + j] * twin_q[r][b * 32 + j];
                    sum += dw * twin_d[r][b] * (double)isum;
                    mag += std::fabs(dw * twin_d[r][b] * (double)isum);
                }
                close_ok = close_ok && std::fabs(y[r * nout + o] - sum) <= 1e-6 * mag + 1e-30;
            }
        require(close_ok, std::string("the ") + (extent == 1 ? "row kernel" : "tile") + " within 1e-6 of a double sum over the " + form + " twin");
    }
    // A group of three projections of one X in one tile call, its inner dimension split for a prompt of 21 rows, each projection within 1e-6 of the double sum over the twin.
    {
        const size_t outs[3] = {40, 104, 64};
        std::vector<std::vector<uint8_t>> gm;
        std::vector<BufferPtr> gw, gy;
        for (size_t p = 0; p < 3; ++p) {
            gm.push_back(q8_matrix(width, outs[p], 81 + (uint32_t)p));
            gw.push_back(g.adopt(gm[p].data(), gm[p].size()));
            gy.push_back(g.alloc(outs[p] * rows * 4));
        }
        const RowRun run[1] = {{rows, size_t(21)}};
        g.matmul_group({Projection({quant::GGML_TYPE_Q8_0, {gw[0].get(), 0}}, {gy[0].get(), 0}, outs[0]),
                        Projection({quant::GGML_TYPE_Q8_0, {gw[1].get(), 0}}, {gy[1].get(), 0}, outs[1]),
                        Projection({quant::GGML_TYPE_Q8_0, {gw[2].get(), 0}}, {gy[2].get(), 0}, outs[2])},
                       {xsb.get(), 0}, width, rows, {run, 1}, dt);
        bool group_ok = true;
        for (size_t p = 0; p < 3; ++p) {
            const std::vector<float> y = read(g, *gy[p], outs[p] * rows);
            for (size_t r = 0; r < rows; ++r)
                for (size_t o = 0; o < outs[p]; ++o) {
                    double sum = 0.0, mag = 0.0;
                    for (size_t b = 0; b < width / 32; ++b) {
                        const uint8_t* block = gm[p].data() + (o * (width / 32) + b) * 34;
                        const double dw = f16_to_f32((uint16_t)(block[0] | block[1] << 8));
                        long isum = 0;
                        for (size_t j = 0; j < 32; ++j) isum += (long)(int8_t)block[2 + j] * twin_q[r][b * 32 + j];
                        sum += dw * twin_d[r][b] * (double)isum;
                        mag += std::fabs(dw * twin_d[r][b] * (double)isum);
                    }
                    group_ok = group_ok && std::fabs(y[r * outs[p] + o] - sum) <= 1e-6 * mag + 1e-30;
                }
        }
        require(group_ok, form + ": a group of three projections in one split tile call within 1e-6 of a double sum over the twin");
    }
    auto y1 = g.alloc(nout * 4);
    // Every decode column of 1 to 64 generated rows, up to four launches of 16, is the column alone, bit for bit.
    // The residual add and a group of three projections are held so at counts either side of each launch's edge.
    {
        // 37 outputs, which neither two nor four divides, so each build's last wave or block holds rows past the end, which it must not store.
        const size_t cols = 64, dn = 37;
        bool alone = true, added = true, grouped = true;
        for (size_t w : {size_t(4096), size_t(1280), size_t(1312)}) {
            const std::vector<uint8_t> mw = q8_matrix(w, dn, 61);
            const std::vector<float> xw = row_classes::values(w * cols, 62), base = row_classes::values(dn * cols, 63);
            auto mwb = g.adopt(mw.data(), mw.size()), xwb = g.adopt(xw.data(), xw.size() * 4);
            auto yw = g.alloc(dn * cols * 4), ya = g.alloc(dn * cols * 4), yb = g.alloc(dn * cols * 4), yc = g.alloc(dn * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {mwb.get(), 0}};
            std::vector<std::vector<float>> one(cols), one_added(cols);
            for (size_t r = 0; r < cols; ++r) {
                const RowRun run[1] = {{1, 1}};
                g.matmul(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, dn, 1, {run, 1}, dt);
                one[r] = read(g, *y1, dn);
                g.write(*y1, 0, base.data() + r * dn, dn * 4);
                g.matmul_add(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, dn, 1, {run, 1}, dt);
                one_added[r] = read(g, *y1, dn);
            }
            for (size_t n = 1; n <= cols; ++n) {
                const RowRun run[1] = {{n, 1}};
                g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, dn, n, {run, 1}, dt);
                const std::vector<float> y = read(g, *yw, dn * n);
                for (size_t r = 0; r < n; ++r) alone = alone && std::memcmp(y.data() + r * dn, one[r].data(), dn * 4) == 0;
                // The same rows as n sequences' runs of one generated row each, as a decode pass of n sequences gives them.
                std::vector<RowRun> each(n);
                for (size_t r = 0; r < n; ++r) each[r] = {r + 1, 1};
                g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, dn, n, {each.data(), n}, dt);
                const std::vector<float> ys = read(g, *yw, dn * n);
                for (size_t r = 0; r < n; ++r) alone = alone && std::memcmp(ys.data() + r * dn, one[r].data(), dn * 4) == 0;
            }
            for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(48), size_t(49), size_t(63), size_t(64)}) {
                const RowRun run[1] = {{n, 1}};
                g.write(*ya, 0, base.data(), dn * n * 4);
                g.matmul_add(wv, {xwb.get(), 0}, {ya.get(), 0}, w, dn, n, {run, 1}, dt);
                const std::vector<float> y = read(g, *ya, dn * n);
                for (size_t r = 0; r < n; ++r) added = added && std::memcmp(y.data() + r * dn, one_added[r].data(), dn * 4) == 0;
                g.matmul_group({Projection(wv, {yw.get(), 0}, dn), Projection(wv, {yb.get(), 0}, dn), Projection(wv, {yc.get(), 0}, dn)},
                               {xwb.get(), 0}, w, n, {run, 1}, dt);
                for (BufferPtr* out : {&yw, &yb, &yc}) {
                    const std::vector<float> yg = read(g, **out, dn * n);
                    for (size_t r = 0; r < n; ++r) grouped = grouped && std::memcmp(yg.data() + r * dn, one[r].data(), dn * 4) == 0;
                }
            }
        }
        require(alone, form + ": every decode column of 1 to 64 generated rows is the column alone, bit for bit, as one run and as one run a row, rows 4096, 1280 and 1312 wide (41 blocks, the int8 class's quarter layout)");
        require(added, form + ": every decode column of the residual add at 1 to 64 rows is the column alone, bit for bit");
        require(grouped, form + ": every decode column of a group of three projections at 1 to 64 rows is the column alone, bit for bit");
    }
    // A row gives the same bits whatever rows a wave the call's output count takes: the first 37 rows of one matrix in calls of 37 to 16384 outputs, at 1 to 16 columns in one launch, 1280 wide and 1312 (the int8 quarter layout).
    {
        const size_t big = 16384, head = 37, cols = 16;
        bool same = true;
        for (size_t w : {size_t(1280), size_t(1312)}) {
            const std::vector<uint8_t> mw = q8_matrix(w, big, 67);
            const std::vector<float> xw = row_classes::values(w * cols, 68);
            auto mwb = g.adopt(mw.data(), mw.size()), xwb = g.adopt(xw.data(), xw.size() * 4), yw = g.alloc(big * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {mwb.get(), 0}};
            for (size_t n = 1; n <= cols; ++n) {
                const RowRun run[1] = {{n, 1}};
                std::vector<float> first;
                for (size_t nout : {head, size_t(1024), size_t(2048), size_t(3072), size_t(4096), size_t(6144), size_t(8192), size_t(12288), big}) {
                    g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, nout, n, {run, 1}, dt);
                    const std::vector<float> y = read(g, *yw, nout * n);
                    std::vector<float> rows;
                    for (size_t r = 0; r < n; ++r) rows.insert(rows.end(), y.begin() + r * nout, y.begin() + r * nout + head);
                    if (first.empty()) first = rows;
                    same = same && std::memcmp(rows.data(), first.data(), rows.size() * 4) == 0;
                }
            }
        }
        require(same, form + ": a decode row gives the same bits in calls of 37 to 16384 outputs, whatever rows a wave they take, at 1 to 16 columns");
    }
    // A decode group of three different projections in one launch gives each projection's own call, bit for bit; 1248 wide takes the int8 quarter layout.
    {
        const size_t outs[3] = {41, 104, 63};
        bool group_ok = true;
        for (size_t w : {size_t(1280), size_t(1248)}) {
            const std::vector<float> xw = row_classes::values(w * 17, 91);
            auto xwb = g.adopt(xw.data(), xw.size() * 4);
            std::vector<std::vector<uint8_t>> gm;
            std::vector<BufferPtr> gw, gy, ys;
            for (size_t p = 0; p < 3; ++p) {
                gm.push_back(q8_matrix(w, outs[p], 92 + (uint32_t)p));
                gw.push_back(g.adopt(gm[p].data(), gm[p].size()));
                gy.push_back(g.alloc(outs[p] * 17 * 4));
                ys.push_back(g.alloc(outs[p] * 17 * 4));
            }
            for (size_t n : {size_t(1), size_t(2), size_t(17)}) {
                const RowRun run[1] = {{n, 1}};
                for (size_t p = 0; p < 3; ++p) g.matmul({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {xwb.get(), 0}, {ys[p].get(), 0}, w, outs[p], n, {run, 1}, dt);
                g.matmul_group({Projection({quant::GGML_TYPE_Q8_0, {gw[0].get(), 0}}, {gy[0].get(), 0}, outs[0]),
                                Projection({quant::GGML_TYPE_Q8_0, {gw[1].get(), 0}}, {gy[1].get(), 0}, outs[1]),
                                Projection({quant::GGML_TYPE_Q8_0, {gw[2].get(), 0}}, {gy[2].get(), 0}, outs[2])},
                               {xwb.get(), 0}, w, n, {run, 1}, dt);
                for (size_t p = 0; p < 3; ++p) {
                    const std::vector<float> a = read(g, *gy[p], outs[p] * n), b = read(g, *ys[p], outs[p] * n);
                    group_ok = group_ok && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                }
            }
        }
        require(group_ok, form + ": a decode group of three different projections in one launch is each projection's own call, bit for bit, at 1, 2 and 17 rows");
    }
    // DeepSeek-V4.1's query and window projections, 1280 and 512 rows, and a third of 1024, as two and three projections of one decode call on rows across the step rule's edges.
    // At one column the three take two rows a wave where the two take one, and each projection is its own call at 1 and 2 columns; a prompt's tile splits by the call's outputs, so a prompt group is not held to it.
    {
        const size_t outs[3] = {1280, 512, 1024};
        bool group_ok = true;
        for (size_t w : {size_t(1280), size_t(5120), size_t(6144)}) {
            const std::vector<float> xw = row_classes::values(w * 17, 93);
            auto xwb = g.adopt(xw.data(), xw.size() * 4);
            std::vector<BufferPtr> gw, gy, ys;
            for (size_t p = 0; p < 3; ++p) {
                const std::vector<uint8_t> m = q8_matrix(w, outs[p], 94 + (uint32_t)p);
                gw.push_back(g.adopt(m.data(), m.size()));
                gy.push_back(g.alloc(outs[p] * 17 * 4));
                ys.push_back(g.alloc(outs[p] * 17 * 4));
            }
            const auto proj = [&](size_t p, const BufferPtr& y) { return Projection({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {y.get(), 0}, outs[p]); };
            for (size_t nproj : {size_t(2), size_t(3)})
                for (const auto& [n, extent] : std::vector<std::pair<size_t, size_t>>{{1, 1}, {2, 1}}) {
                    const RowRun run[1] = {{n, extent}};
                    for (size_t p = 0; p < nproj; ++p) g.matmul({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {xwb.get(), 0}, {ys[p].get(), 0}, w, outs[p], n, {run, 1}, dt);
                    if (nproj == 2) g.matmul_group({proj(0, gy[0]), proj(1, gy[1])}, {xwb.get(), 0}, w, n, {run, 1}, dt);
                    else g.matmul_group({proj(0, gy[0]), proj(1, gy[1]), proj(2, gy[2])}, {xwb.get(), 0}, w, n, {run, 1}, dt);
                    for (size_t p = 0; p < nproj; ++p) {
                        const std::vector<float> a = read(g, *gy[p], outs[p] * n), b = read(g, *ys[p], outs[p] * n);
                        group_ok = group_ok && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                    }
                }
        }
        require(group_ok, form + ": groups of 1280 and 512 rows, and of a third of 1024, are each projection's own call, bit for bit, at 1 and 2 columns");
    }
    // A prompt row gives the same bits in calls of 1 to 21 rows and of 63 to 65, 128 and 130, which cross the tile's 64 rows, over 100 outputs, which end inside its 64.
    {
        const size_t pn = 130, po = 100;
        const std::vector<uint8_t> mp = q8_matrix(width, po, 71);
        const std::vector<float> xp = row_classes::values(width * pn, 72);
        auto mpb = g.adopt(mp.data(), mp.size()), xpb = g.adopt(xp.data(), xp.size() * 4), yp = g.alloc(po * pn * 4), yp1 = g.alloc(po * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mpb.get(), 0}};
        std::vector<std::vector<float>> alone(pn);
        for (size_t r = 0; r < pn; ++r) {
            const RowRun run[1] = {{1, 2}};
            g.matmul(wv, {xpb.get(), r * width}, {yp1.get(), 0}, width, po, 1, {run, 1}, dt);
            alone[r] = read(g, *yp1, po);
        }
        bool tiled = true;
        std::vector<size_t> counts;
        for (size_t n = 1; n <= 21; ++n) counts.push_back(n);
        for (size_t n : {size_t(63), size_t(64), size_t(65), size_t(128), size_t(130)}) counts.push_back(n);
        for (size_t n : counts) {
            const RowRun run[1] = {{n, 2}};
            g.matmul(wv, {xpb.get(), 0}, {yp.get(), 0}, width, po, n, {run, 1}, dt);
            const std::vector<float> y = read(g, *yp, po * n);
            for (size_t r = 0; r < n; ++r) tiled = tiled && std::memcmp(y.data() + r * po, alone[r].data(), po * 4) == 0;
        }
        require(tiled, form + ": a prompt row gives the same bits in calls of 1 to 130 rows across the tile's edges");
    }
    // A matrix of an odd number of blocks, 65 outputs of 3 blocks adopted alone, so its last block starts on a word and the tile's last word load of it runs past the matrix: each prompt row the bits of that row alone.
    {
        const size_t sw = 96, so = 65, sn = 70;
        const std::vector<uint8_t> ms = q8_matrix(sw, so, 73);
        const std::vector<float> xs = row_classes::values(sw * sn, 74);
        auto msb = g.adopt(ms.data(), ms.size()), xsb2 = g.adopt(xs.data(), xs.size() * 4), ysn = g.alloc(so * sn * 4), ys1 = g.alloc(so * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {msb.get(), 0}};
        bool edge = true;
        const RowRun all[1] = {{sn, 2}};
        g.matmul(wv, {xsb2.get(), 0}, {ysn.get(), 0}, sw, so, sn, {all, 1}, dt);
        const std::vector<float> y = read(g, *ysn, so * sn);
        for (size_t r = 0; r < sn; ++r) {
            const RowRun run[1] = {{1, 2}};
            g.matmul(wv, {xsb2.get(), r * sw}, {ys1.get(), 0}, sw, so, 1, {run, 1}, dt);
            edge = edge && std::memcmp(y.data() + r * so, read(g, *ys1, so).data(), so * 4) == 0;
        }
        // The same matrix with room after it in its buffer gives the same bits, so the load past its end reads nothing that reaches a sum.
        std::vector<uint8_t> roomy = ms;
        roomy.resize(ms.size() + 64, 0xa5);
        auto rb = g.adopt(roomy.data(), roomy.size());
        g.matmul({quant::GGML_TYPE_Q8_0, {rb.get(), 0}}, {xsb2.get(), 0}, {ysn.get(), 0}, sw, so, sn, {all, 1}, dt);
        edge = edge && std::memcmp(read(g, *ysn, so * sn).data(), y.data(), so * sn * 4) == 0;
        require(edge, form + ": a prompt row of a matrix whose last block ends inside a word gives the bits of that row alone and of the matrix with room after it");
    }
    // The tall build gives the 64-output build's bits, a whole matrix against slices of the same inner split (docs/src/backends-rocm.md).
    for (const size_t tw : {size_t(512), size_t(4096), size_t(4128)}) {
        const size_t tr = tw == 512 ? 70 : 520, tn = tw == 512 ? 3940 : 5860, slice = tw == 512 ? 100 : 3840;
        const std::vector<uint8_t> mt = q8_matrix(tw, tn, 91);
        const std::vector<float> xt = row_classes::values(tw * tr, 92);
        const size_t row_bytes = tw / 32 * 34;
        auto mtb = g.adopt(mt.data(), mt.size()), xtb = g.adopt(xt.data(), xt.size() * 4), yt = g.alloc(tn * tr * 4), ys = g.alloc(slice * tr * 4);
        const RowRun run[1] = {{tr, tr}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mtb.get(), 0}}, {xtb.get(), 0}, {yt.get(), 0}, tw, tn, tr, {run, 1}, dt);
        const std::vector<float> whole = read(g, *yt, tn * tr);
        bool same = true;
        for (size_t first : {size_t(0), tn - slice}) {
            const std::vector<uint8_t> part(mt.begin() + first * row_bytes, mt.begin() + (first + slice) * row_bytes);
            auto pb = g.adopt(part.data(), part.size());
            g.matmul({quant::GGML_TYPE_Q8_0, {pb.get(), 0}}, {xtb.get(), 0}, {ys.get(), 0}, tw, slice, tr, {run, 1}, dt);
            const std::vector<float> y = read(g, *ys, slice * tr);
            for (size_t r = 0; r < tr; ++r) same = same && std::memcmp(y.data() + r * slice, whole.data() + r * tn + first, slice * 4) == 0;
        }
        require(same, form + ": the tall tile's outputs are the 64-output tile's bits, at the matrix's start and at its end inside a partial tile");
    }
    // The twin's layout: a prompt's rows block-major, padded to an odd multiple of 4 rows, give the bits of the same rows laid row after row, which a call of more regions than the layout holds takes.
    // Row counts either side of each padding step (4, 12, 20) and of a 64-row tile, final tiles cut short, the tall build at 261 and 520 rows, and runs of both kernels side by side, a prompt's region after a row kernel's.
    for (const size_t lw : {size_t(512), size_t(4096)}) {
        const size_t lo = lw == 512 ? 200 : 5860;
        const std::vector<uint8_t> ml = q8_matrix(lw, lo, 95);
        auto mlb = g.adopt(ml.data(), ml.size());
        const WeightView wl{quant::GGML_TYPE_Q8_0, {mlb.get(), 0}};
        const std::vector<size_t> counts = lw == 512 ? std::vector<size_t>{1, 4, 5, 12, 13, 20, 21, 63, 64, 65, 70, 129} : std::vector<size_t>{261, 520};
        bool flat = true, mixed = true;
        for (const size_t n : counts) {
            // xa: the prompt's n rows and 8 more; xo: 3 generated rows, 2 more and a prompt of 5; xm: xo's first 3, the prompt's n, then xo's other 7.
            const std::vector<float> xa = row_classes::values(lw * (n + 8), 96 + (uint32_t)n), xo = row_classes::values(lw * 10, 97);
            std::vector<float> xm(xo.begin(), xo.begin() + 3 * lw);
            xm.insert(xm.end(), xa.begin(), xa.begin() + n * lw);
            xm.insert(xm.end(), xo.begin() + 3 * lw, xo.end());
            auto xab = g.adopt(xa.data(), xa.size() * 4), xob = g.adopt(xo.data(), xo.size() * 4), xmb = g.adopt(xm.data(), xm.size() * 4);
            auto ya = g.alloc(lo * (n + 8) * 4), ym = g.alloc(lo * (n + 10) * 4), yo = g.alloc(lo * 5 * 4);
            const RowRun alone[1] = {{n, n}};
            g.matmul(wl, {xab.get(), 0}, {ya.get(), 0}, lw, lo, n, {alone, 1}, dt);
            const std::vector<float> a = read(g, *ya, lo * n);
            // Nine regions: the prompt, then single rows of the row kernel and the tile in turn.
            std::vector<RowRun> nine = {{n, n}};
            for (size_t i = 1; i <= 8; ++i) nine.push_back({n + i, i % 2 ? size_t(1) : size_t(3)});
            g.matmul(wl, {xab.get(), 0}, {ya.get(), 0}, lw, lo, n + 8, {nine.data(), nine.size()}, dt);
            flat = flat && std::memcmp(read(g, *ya, lo * n).data(), a.data(), lo * n * 4) == 0;
            const RowRun four_runs[4] = {{3, 1}, {3 + n, n}, {5 + n, 1}, {10 + n, 7}};
            g.matmul(wl, {xmb.get(), 0}, {ym.get(), 0}, lw, lo, n + 10, {four_runs, 4}, dt);
            const std::vector<float> m4 = read(g, *ym, lo * (n + 10));
            mixed = mixed && std::memcmp(m4.data() + 3 * lo, a.data(), lo * n * 4) == 0;
            // Two prompts side by side, which share a tile call where their splits match, give each prompt's bits alone.
            const RowRun two[2] = {{n, n}, {n + 8, 8}};
            g.matmul(wl, {xab.get(), 0}, {ya.get(), 0}, lw, lo, n + 8, {two, 2}, dt);
            const std::vector<float> a2 = read(g, *ya, lo * (n + 8));
            const RowRun eight[1] = {{8, 8}};
            g.matmul(wl, {xab.get(), n * lw}, {ym.get(), 0}, lw, lo, 8, {eight, 1}, dt);
            mixed = mixed && std::memcmp(a2.data(), a.data(), lo * n * 4) == 0 && std::memcmp(a2.data() + n * lo, read(g, *ym, lo * 8).data(), lo * 8 * 4) == 0;
            // Each other run alone gives its rows' bits.
            struct Part { size_t row, rows, extent, at; };
            for (const Part p : {Part{0, 3, 1, 0}, Part{3, 2, 1, 3 + n}, Part{5, 5, 7, 5 + n}}) {
                const RowRun run[1] = {{p.rows, p.extent}};
                g.matmul(wl, {xob.get(), p.row * lw}, {yo.get(), 0}, lw, lo, p.rows, {run, 1}, dt);
                mixed = mixed && std::memcmp(read(g, *yo, lo * p.rows).data(), m4.data() + p.at * lo, lo * p.rows * 4) == 0;
            }
        }
        require(flat, form + ": a prompt's rows read block-major give the bits of the same rows read row after row, rows " + std::to_string(lw) + " wide");
        require(mixed, form + ": runs of the row kernel and the tile side by side give each run's bits alone, rows " + std::to_string(lw) + " wide");
    }
    // Every pair of extents of one class: the same bits through a Q8_0 product in this class.
    std::vector<std::vector<float>> out;
    const size_t four = 4;
    for (size_t e : row_classes::kExtents) {
        const RowRun run[1] = {{four, e}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, four, {run, 1}, dt);
        out.push_back(read(g, *ys, nout * four));
    }
    require(row_classes::same_within_classes(g, out, "a Q8_0 product, " + form) > 0, "the row-class check compared pairs, " + form);
}

// The twins producers write beside their outputs, once a product has read one in the class: a product of a norm's or silu_mul's output gives the bits of the same product of a copy of it, whose twin the product makes itself.
// A write into the output between them, or a product into it, drops the producer's twin.
void producer_twins(Backend& g, Dtype dt) {
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    const size_t nout = 40;
    bool norms = true, silus = true, dropped = true;
    // Rows of 1024, 4096 and 5120 take the held builds of 16 and 32 values a thread, 9216 the plain one, and a stride past n writes no twin.
    struct Case { size_t n, rows, stride, extent; };
    for (const Case cs : {Case{4096, 1, 4096, 1}, Case{4096, 3, 4096, 3}, Case{5120, 2, 5120, 1}, Case{9216, 2, 9216, 2}, Case{1024, 2, 1024, 2}, Case{4096, 2, 4200, 1}}) {
        const size_t n = cs.n, rows = cs.rows, stride = cs.stride;
        const std::vector<float> x = row_classes::values(rows * stride, 91), w = row_classes::values(n, 92), up = row_classes::values(rows * stride, 93);
        const std::vector<uint8_t> m = q8_matrix(n, nout, 94);
        auto xb = g.adopt(x.data(), x.size() * 4), wb = g.adopt(w.data(), w.size() * 4), ub = g.adopt(up.data(), up.size() * 4), mb = g.adopt(m.data(), m.size());
        auto ob = g.alloc(rows * stride * 4), cb = g.alloc(rows * n * 4), y = g.alloc(nout * rows * 4), yc = g.alloc(nout * rows * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
        const RowRun run[1] = {{rows, cs.extent}};
        // The copy is dense, so the product of a strided output reads row 0 alone.
        const size_t prows = stride == n ? rows : 1;
        const RowRun prun[1] = {{prows, prows == 1 ? size_t(1) : cs.extent}};
        g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f, {run, 1});
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, prows, {prun, 1}, dt);
        g.copy(*cb, 0, *ob, 0, prows * n * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, prows, {prun, 1}, dt);
        norms = norms && same(read(g, *y, nout * prows), read(g, *yc, nout * prows));
        if (stride != n) continue;
        g.silu_mul({ob.get(), 0}, {xb.get(), 0}, {ub.get(), 0}, rows * n, 0.0f, {run, 1});
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, {run, 1}, dt);
        g.copy(*cb, 0, *ob, 0, rows * n * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, rows, {run, 1}, dt);
        silus = silus && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        // A write into the norm's output: the product reads the written values.
        g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
        g.write(*ob, 0, up.data(), rows * n * 4);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, {run, 1}, dt);
        g.matmul(wv, {ub.get(), 0}, {yc.get(), 0}, n, nout, rows, {run, 1}, dt);
        dropped = dropped && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        // A product into the norm's output, then a product of it.
        if (n >= nout * rows) {
            g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
            g.matmul(wv, {ob.get(), 0}, {ob.get(), 0}, n, nout, 1, {}, dt);
            g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, {run, 1}, dt);
            g.copy(*cb, 0, *ob, 0, rows * n * 4);
            g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, rows, {run, 1}, dt);
            dropped = dropped && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        }
    }
    // The producers write the twin as the product's runs lay it out (TwinMap): runs of both kernels side by side, and the same output read by a product of other runs, which makes its own.
    {
        const size_t n = 4096, rows = 13;
        const RowRun mixed[3] = {{3, 1}, {10, 7}, {13, 1}}, prompt[1] = {{13, 13}};
        const std::vector<float> x = row_classes::values(rows * n, 95), w = row_classes::values(n, 96), up = row_classes::values(rows * n, 97);
        const std::vector<uint8_t> m = q8_matrix(n, nout, 98);
        auto xb = g.adopt(x.data(), x.size() * 4), wb = g.adopt(w.data(), w.size() * 4), ub = g.adopt(up.data(), up.size() * 4), mb = g.adopt(m.data(), m.size());
        auto ob = g.alloc(rows * n * 4), cb = g.alloc(rows * n * 4), y = g.alloc(nout * rows * 4), yc = g.alloc(nout * rows * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
        for (const RowRun* pr : {mixed, prompt}) {
            const RowRuns produced{mixed, 3}, read_as{pr, pr == mixed ? size_t(3) : size_t(1)};
            g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, n, 1e-6f, produced);
            g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, read_as, dt);
            g.copy(*cb, 0, *ob, 0, rows * n * 4);
            g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, rows, read_as, dt);
            norms = norms && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
            g.silu_mul({ob.get(), 0}, {xb.get(), 0}, {ub.get(), 0}, rows * n, 0.0f, produced);
            g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, read_as, dt);
            g.copy(*cb, 0, *ob, 0, rows * n * 4);
            g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, rows, read_as, dt);
            silus = silus && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        }
    }
    require(norms, form + ": a product of rms_norm_rows' output gives the bits of a product of its copy");
    require(silus, form + ": a product of silu_mul's output gives the bits of a product of its copy");
    require(dropped, form + ": a write or a product into a producer's output drops its twin");
}

// moe_step of one generated row in the int8 class: on a 64-lane device with the dot module, three launches (llmx_moe_step_*) whose every output is the bits of the base's ops (Backend::moe_step).
// Flash-Next's shape and a narrower h, other expert and pick counts, at both routed widths and all four type pairs; the router sends every pick to the first `few` experts, so only theirs are filled.
void moe_step_bits(Backend& g) {
    const size_t few = 24;
    // The build compiles the dot module for gfx906 alone (CMakeLists.txt, int16dot).
    const bool fused = rocm_device(0).arch.rfind("gfx906", 0) == 0 && rocm_device(0).caps.subgroup_size == 64;
    struct Shape {
        size_t nin, ne, k;
    };
    const Shape shapes[] = {{2560, 512, 10}, {1280, 512, 10}, {2560, 128, 10}, {2560, 512, 6}, {1280, 64, 4}};
    for (const Shape& shape : shapes) {
        const size_t nin = shape.nin, ne = shape.ne, k = shape.k;
        const std::vector<float> h = row_classes::values(nin, 171), rgw = row_classes::values(nin, 172, 0.01f);
        double hh = 0;
        for (float v : h) hh += double(v) * v;
        std::vector<float> router(ne * nin);
        for (size_t e = 0; e < ne; ++e) {
            const double score = e < few ? 1.0 + 0.17 * double((e * 7) % few) : -10.0;
            for (size_t i = 0; i < nin; ++i) router[e * nin + i] = float(h[i] * score / hh);
        }
        const std::vector<float> zero(nin, 0.0f);
        auto hb = g.adopt(h.data(), nin * 4), rb = g.adopt(router.data(), router.size() * 4), gb = g.adopt(rgw.data(), nin * 4), zb = g.adopt(zero.data(), nin * 4);
        // h in host memory, which the fused step refuses, since its twin would take the slot the routed act's twin takes.
        BufferPtr hhost = g.alloc(nin * 4, Memory::host_visible);
        g.write(*hhost, 0, h.data(), nin * 4);
        const auto stack = [&](uint32_t type, size_t width, size_t rows, uint32_t seed) {
            const std::vector<uint8_t> m = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(width, few * rows, seed) : typed_matrix(type, width, few * rows, seed);
            BufferPtr b = g.alloc(quant::row_bytes(type, width, ne * rows));
            g.write(*b, 0, m.data(), m.size());
            return b;
        };
        // At Flash-Next's shape also a width of 17 blocks, an odd count the fused step refuses for either down type.
        const bool flash = nin == 2560 && ne == 512 && k == 10;
        std::vector<size_t> widths = {320, 640};
        if (flash) widths.push_back(544);
        for (const size_t ff : widths)
            for (const uint32_t gt : {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K})
                for (const uint32_t dt : {quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q8_0}) {
                    if (ff == 544 && gt != quant::GGML_TYPE_Q4_K) continue;
                    const std::string at = std::to_string(nin) + " wide, " + std::to_string(k) + " of " + std::to_string(ne) + " experts, " + quant::storage_type(gt)->name +
                                           " gate and up, " + quant::storage_type(dt)->name + " down, ff " + std::to_string(ff);
                    const BufferPtr wg = stack(gt, nin, ff, 173), wu = stack(gt, nin, ff, 174), wd = stack(dt, ff, nin, 175);
                    const std::vector<uint8_t> sg = q8_matrix(nin, ff, 176), su = q8_matrix(nin, ff, 177), sd = q8_matrix(ff, nin, 178);
                    auto sgb = g.adopt(sg.data(), sg.size()), sub = g.adopt(su.data(), su.size()), sdb = g.adopt(sd.data(), sd.size());
                    struct Out {
                        BufferPtr scores, ids, weights, gate, up, act, sgate, sup, sact, rg, y;
                        std::vector<RowRun> scratch;
                    };
                    const auto outs = [&] {
                        return Out{g.alloc(ne * 4), g.alloc(k * 4), g.alloc(k * 4), g.alloc(k * ff * 4), g.alloc(k * ff * 4), g.alloc(k * ff * 4), g.alloc(ff * 4),
                                   g.alloc(ff * 4), g.alloc(ff * 4), g.alloc(4), g.alloc(nin * 4), {}};
                    };
                    // The base's outputs, then the step's with h's twin kept, with it not kept, and with h in host memory.
                    Out o[4] = {outs(), outs(), outs(), outs()};
                    MoeShared sh[4];
                    MoeStep st[4];
                    for (int a = 0; a < 4; ++a) {
                        sh[a] = MoeShared{Projection{{quant::GGML_TYPE_Q8_0, {sgb.get(), 0}}, {o[a].sgate.get(), 0}, ff},
                                          Projection{{quant::GGML_TYPE_Q8_0, {sub.get(), 0}}, {o[a].sup.get(), 0}, ff}, {quant::GGML_TYPE_Q8_0, {sdb.get(), 0}}, {o[a].sact.get(), 0},
                                          Projection{{quant::GGML_TYPE_F32, {gb.get(), 0}}, {o[a].rg.get(), 0}, 1}};
                        st[a] = MoeStep{Projection{{quant::GGML_TYPE_F32, {rb.get(), 0}}, {o[a].scores.get(), 0}, ne}, k, backend::RouteRule::softmax(true), {o[a].ids.get(), 0},
                                        {o[a].weights.get(), 0}, Projection{{gt, {wg.get(), 0}}, {o[a].gate.get(), 0}, ff}, Projection{{gt, {wu.get(), 0}}, {o[a].up.get(), 0}, ff},
                                        {dt, {wd.get(), 0}}, {o[a].act.get(), 0}, &sh[a], zb.get(), &o[a].scratch};
                    }
                    const RowRun decode[1] = {{1, 1}};
                    rocm_take_launches(g);
                    g.Backend::moe_step(st[0], {hb.get(), 0}, {o[0].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    const auto base = rocm_take_launches(g);
                    g.moe_step(st[1], {hb.get(), 0}, {o[1].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    const auto step = rocm_take_launches(g);
                    // A write forgets every kept twin, so the route launch makes h's.
                    g.write(*hb, 0, h.data(), nin * 4);
                    rocm_take_launches(g);
                    g.moe_step(st[2], {hb.get(), 0}, {o[2].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    const auto unkept = rocm_take_launches(g);
                    g.moe_step(st[3], {hhost.get(), 0}, {o[3].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    const auto host = rocm_take_launches(g);
                    size_t base_steps = 0, twins = 0;
                    for (const auto& [name, n] : base) base_steps += name.rfind("llmx_moe_step", 0) == 0 ? n : 0;
                    for (const auto& [name, n] : unkept) twins += name.rfind("llmx_twin", 0) == 0 ? n : 0;
                    require(base_steps == 0, "the base's moe_step launches no fused kernel, " + at);
                    const std::string up = gt == quant::GGML_TYPE_Q5_K ? "llmx_moe_step_up_q5_k" : "llmx_moe_step_up_q4_k",
                                      down = dt == quant::GGML_TYPE_Q8_0 ? "llmx_moe_step_down_q8_0" : "llmx_moe_step_down_q5_1";
                    if (fused && ff != 544) {
                        require(step.size() == 3 && step.count("llmx_moe_step_route") && (step.count(up) || step.count(up + "_w")) && step.count(down), "moe_step is the three fused launches, " + at);
                        require(unkept.size() == 3 && unkept.count("llmx_moe_step_route") && twins == 0, "with h's twin not kept the route launch makes it, " + at);
                    } else {
                        require(!step.count("llmx_moe_step_route") && !unkept.count("llmx_moe_step_route"), "a step the fused kernels do not hold runs the base's ops, " + at);
                    }
                    require(!host.count("llmx_moe_step_route"), "a step of h in host memory runs the base's ops, " + at);
                    const std::vector<float> ids = read(g, *o[0].ids, k);
                    bool picked = true;
                    for (float v : ids) {
                        uint32_t u;
                        std::memcpy(&u, &v, 4);
                        picked = picked && u < few;
                    }
                    require(picked, "the router picks among the filled experts, " + at);
                    for (int a = 1; a < 4; ++a) {
                        const std::string of = std::string(a == 1 ? "" : a == 2 ? "h's twin not kept, " : "h in host memory, ") + at;
                        require(same(ids, read(g, *o[a].ids, k)), "the step's picks are the base's, " + of);
                        require(same(read(g, *o[0].weights, k), read(g, *o[a].weights, k)), "the step's weights are the base's, " + of);
                        require(same(read(g, *o[0].scores, ne), read(g, *o[a].scores, ne)), "the step's router scores are the base's, " + of);
                        require(same(read(g, *o[0].rg, 1), read(g, *o[a].rg, 1)), "the step's row gate is the base's, " + of);
                        require(same(read(g, *o[0].gate, k * ff), read(g, *o[a].gate, k * ff)) && same(read(g, *o[0].up, k * ff), read(g, *o[a].up, k * ff)),
                                "the step's routed gate and up are the base's, " + of);
                        require(same(read(g, *o[0].act, k * ff), read(g, *o[a].act, k * ff)), "the step's routed act is the base's, " + of);
                        require(same(read(g, *o[0].sact, ff), read(g, *o[a].sact, ff)), "the step's shared act is the base's, " + of);
                        require(same(read(g, *o[0].y, nin), read(g, *o[a].y, nin)), "the step's output is the base's, " + of);
                    }
                    // A step the fused kernels do not compute, a clamped SiLU or another score, runs the base's ops, which ROCm implements, and launches no fused kernel.
                    MoeStep clamped = st[1], scored = st[1];
                    clamped.limit = 7.0f;
                    scored.rule = backend::RouteRule::sqrt_softplus({}, 1.0f, true);
                    rocm_take_launches(g);
                    g.moe_step(clamped, {hb.get(), 0}, {o[1].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    require(!rocm_take_launches(g).count("llmx_moe_step_route"), "a clamped moe_step runs the base's ops, " + at);
                    g.moe_step(scored, {hb.get(), 0}, {o[1].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    require(!rocm_take_launches(g).count("llmx_moe_step_route"), "a sqrt_softplus moe_step runs the base's ops, " + at);
                }
    }
}


// delta_step of a linear-attention layer against the base calls (Backend::delta_step), bit for bit in every slot both write but the conv, both state slots and the output product.
// It covers the tensor-width shard shapes and other head widths, 1 to 8 rows a view over histories of 0, 1, 2 and 100, one view and three, f16 and int8, and which launches ran.
void delta_step_bits(Backend& g) {
    struct Shape {
        size_t hk, hv, kd, vd, nin;
    };
    const Shape shapes[] = {{16, 48, 128, 128, 2560}, {8, 24, 128, 128, 2560}, {4, 12, 128, 128, 640}, {2, 6, 128, 128, 640}, {4, 4, 64, 64, 512},
                            {2, 6, 128, 64, 512},     {2, 4, 96, 128, 512},     {3, 6, 4, 4, 256},    {1, 2, 36, 8, 256},   {2, 4, 132, 64, 256},
                            {2, 4, 64, 130, 256},     {2, 4, 64, 66, 256}};
    // Each case is its views' (rows, history) pairs; the last view of a case of several reads slot 0 and writes slot 2, as a verify does.
    const std::vector<std::vector<std::pair<size_t, size_t>>> cases = {{{1, 100}}, {{1, 0}}, {{2, 1}}, {{3, 2}}, {{4, 100}}, {{5, 0}}, {{8, 100}}, {{9, 100}},
                                                                     {{1, 100}, {8, 2}, {3, 0}}, {{6, 1}, {7, 100}, {2, 2}}, {{1, 0}, {1, 0}, {1, 0}}};
    for (const Shape& sp : shapes) {
        const StateShape sh{sp.hk, sp.hv, sp.kd, sp.vd};
        const size_t C = sh.channels(), V = sp.hv * sp.vd, E = sp.nin, slots = 6;
        const bool width = sp.kd <= 128 && sp.kd % 4 == 0 && sp.vd <= 128 && sp.vd % 4 == 0;
        // The base refuses K heads past 128 itself, so a step at that width is refused either way.
        const bool base_runs = sp.kd <= 128;
        const std::vector<uint8_t> wqkv = q8_matrix(E, C, 301), wgate = q8_matrix(E, V, 302), wout = q8_matrix(V % 32 ? 32 : V, 64, 303);
        const std::vector<float> wa = row_classes::values(sp.hv * E, 304, 0.02f), wb = row_classes::values(sp.hv * E, 305, 0.02f);
        std::vector<float> a = row_classes::values(sp.hv, 306, 1.0f), dt = row_classes::values(sp.hv, 307, 0.5f), nw = row_classes::values(sp.vd, 308, 1.0f),
                           cw = row_classes::values(C * kConvTaps, 309, 0.5f), state = row_classes::values(slots * sh.slot_floats(), 310, 0.1f);
        for (float& v : a) v = -std::fabs(v) - 0.05f;
        auto qb = g.adopt(wqkv.data(), wqkv.size()), gb = g.adopt(wgate.data(), wgate.size()), ob = g.adopt(wout.data(), wout.size());
        auto ab = g.adopt(wa.data(), wa.size() * 4), bb = g.adopt(wb.data(), wb.size() * 4), ap = g.adopt(a.data(), a.size() * 4), db = g.adopt(dt.data(), dt.size() * 4),
             nb = g.adopt(nw.data(), nw.size() * 4), cb = g.adopt(cw.data(), cw.size() * 4);
        const std::string shape = std::to_string(sp.hv) + "/" + std::to_string(sp.hk) + " heads " + std::to_string(sp.kd) + "x" + std::to_string(sp.vd) + ", " + std::to_string(E) + " wide";
        for (const auto& views_at : cases)
            for (Dtype dt_ : {Dtype::f16, Dtype::int8}) {
                size_t rows = 0;
                for (const auto& [nq, len] : views_at) rows += nq;
                std::string at = shape + ", rows";
                for (const auto& [nq, len] : views_at) at += " " + std::to_string(nq) + "@" + std::to_string(len);
                at += std::string(", ") + dtype_name(dt_);
                const std::vector<float> mixed = row_classes::values(rows * E, 311 + (uint32_t)rows, 1.0f);
                auto mb = g.adopt(mixed.data(), mixed.size() * 4);
                struct Arm {
                    std::unique_ptr<StateStorage> st;
                    BufferPtr raw, conv, gate, ab, out, y;
                };
                Arm arm[2];
                std::vector<StateView> views[2];
                DeltaStep step[2];
                for (int k = 0; k < 2; ++k) {
                    arm[k] = Arm{g.state_alloc(1, slots, sh), g.alloc(rows * C * 4), g.alloc(rows * C * 4), g.alloc(rows * V * 4), g.alloc(2 * rows * sp.hv * 4),
                                 g.alloc(rows * V * 4), g.alloc(rows * 64 * 4)};
                    g.write(arm[k].st->layer(0), 0, state.data(), state.size() * 4);
                    size_t v = 0;
                    for (const auto& [nq, len] : views_at) {
                        const bool verify = views_at.size() > 1 && v + 1 == views_at.size();
                        views[k].push_back(StateView{arm[k].st.get(), verify ? 0 : v + 1, verify ? 5 : v + 1, len, nq});
                        ++v;
                    }
                    const Slice alpha{arm[k].ab.get(), 0}, beta{arm[k].ab.get(), rows * sp.hv};
                    step[k] = DeltaStep{Projection{{quant::GGML_TYPE_Q8_0, {qb.get(), 0}}, {arm[k].raw.get(), 0}, C},
                                        Projection{{quant::GGML_TYPE_Q8_0, {gb.get(), 0}}, {arm[k].gate.get(), 0}, V},
                                        Projection{{quant::GGML_TYPE_F32, {ab.get(), 0}}, alpha, sp.hv},
                                        Projection{{quant::GGML_TYPE_F32, {bb.get(), 0}}, beta, sp.hv},
                                        {cb.get(), 0}, {ap.get(), 0}, {db.get(), 0}, {nb.get(), 0}, {arm[k].conv.get(), 0}, {arm[k].out.get(), 0}, 1e-6f};
                }
                // A run a view, its extent the view's history and rows, as a decode step or a verify gives them.
                std::vector<RowRun> runs;
                for (const auto& [nq, len] : views_at) runs.push_back(RowRun{(runs.empty() ? 0 : runs.back().end) + nq, len + nq});
                const RowRuns rr{runs.data(), runs.size()};
                const auto call = [&](int k) {
                    if (k == 0) g.Backend::delta_step(step[0], {mb.get(), 0}, E, rows, 0, views[0].data(), views[0].size(), rr, dt_);
                    else g.delta_step(step[1], {mb.get(), 0}, E, rows, 0, views[1].data(), views[1].size(), rr, dt_);
                    if (V % 32 == 0) g.matmul({quant::GGML_TYPE_Q8_0, {ob.get(), 0}}, {arm[k].out.get(), 0}, {arm[k].y.get(), 0}, V, 64, rows, rr, dt_);
                };
                rocm_take_launches(g);
                if (!base_runs) {
                    require(throws([&] { call(1); }), "a step past the base's K heads is refused, " + at);
                    rocm_take_launches(g);
                    continue;
                }
                call(0);
                const auto base = rocm_take_launches(g);
                call(1);
                const auto fused = rocm_take_launches(g);
                bool holds = width && views_at.size() <= 64;
                for (const auto& [nq, len] : views_at) holds = holds && nq <= 8;
                require(!base.count("llmx_delta_step"), "the base's delta_step launches no fused kernel, " + at);
                require(fused.count("llmx_delta_step") == (holds ? 1u : 0u), std::string(holds ? "delta_step is one fused launch, " : "a step outside the limits runs the base's ops, ") + at);
                if (holds) require(!fused.count("llmx_gated_delta_rule_short") && !fused.count("llmx_causal_conv_silu") && !fused.count("llmx_gated_rms_norm"), "the fused step runs no chain op, " + at);
                require(same(read(g, *arm[0].raw, rows * C), read(g, *arm[1].raw, rows * C)), "the qkv rows are the base's, " + at);
                require(same(read(g, *arm[0].gate, rows * V), read(g, *arm[1].gate, rows * V)), "the gate rows are the base's, " + at);
                require(same(read(g, *arm[0].ab, 2 * rows * sp.hv), read(g, *arm[1].ab, 2 * rows * sp.hv)), "alpha and beta are the base's, " + at);
                require(same(read(g, *arm[0].out, rows * V), read(g, *arm[1].out, rows * V)), "the output is the base's, " + at);
                require(same(read(g, *arm[0].y, rows * 64), read(g, *arm[1].y, rows * 64)), "the product of the output is the base's, " + at);
                require(same(read(g, arm[0].st->layer(0), state.size()), read(g, arm[1].st->layer(0), state.size())), "every state slot is the base's, " + at);
            }
        // A step of BF16 alpha and beta matrices, or of 65 views, runs the base's ops.
        if (!width) continue;
        const size_t rows = 65;
        const std::vector<float> mixed = row_classes::values(rows * E, 399, 1.0f);
        auto mb = g.adopt(mixed.data(), mixed.size() * 4);
        auto st = g.state_alloc(1, 2 * rows, sh);
        auto raw = g.alloc(rows * C * 4), conv = g.alloc(rows * C * 4), gate = g.alloc(rows * V * 4), abo = g.alloc(2 * rows * sp.hv * 4), out = g.alloc(rows * V * 4);
        std::vector<uint16_t> wa16(wa.size());
        for (size_t i = 0; i < wa.size(); ++i) wa16[i] = f32_to_bf16(wa[i]);
        auto ab16 = g.adopt(wa16.data(), wa16.size() * 2);
        std::vector<StateView> many;
        std::vector<RowRun> each;
        for (size_t v = 0; v < rows; ++v) many.push_back(StateView{st.get(), v, v, 3, 1}), each.push_back(RowRun{v + 1, 4});
        for (const bool f16 : {false, true}) {
            const size_t n = f16 ? 1 : rows;
            const DeltaStep d{Projection{{quant::GGML_TYPE_Q8_0, {qb.get(), 0}}, {raw.get(), 0}, C}, Projection{{quant::GGML_TYPE_Q8_0, {gb.get(), 0}}, {gate.get(), 0}, V},
                              Projection{{f16 ? quant::GGML_TYPE_BF16 : quant::GGML_TYPE_F32, {f16 ? ab16.get() : ab.get(), 0}}, {abo.get(), 0}, sp.hv},
                              Projection{{quant::GGML_TYPE_F32, {bb.get(), 0}}, {abo.get(), n * sp.hv}, sp.hv},
                              {cb.get(), 0}, {ap.get(), 0}, {db.get(), 0}, {nb.get(), 0}, {conv.get(), 0}, {out.get(), 0}, 1e-6f};
            rocm_take_launches(g);
            g.delta_step(d, {mb.get(), 0}, E, n, 0, many.data(), n, RowRuns{each.data(), n}, Dtype::int8);
            require(!rocm_take_launches(g).count("llmx_delta_step"), std::string(f16 ? "a step of BF16 alpha runs the base's ops, " : "a step of 65 views runs the base's ops, ") + shape);
        }
    }
}

// moe_step of several generated rows in the int8 class against the base's ops (Backend::moe_step) bit for bit, with the launches that show which ran.
// The fused step takes the rows while the base's routed products would not group their entries, rows * k below the gate type's int8 break-even of 1.25 entries an expert (rocm_backend.cpp, kQ4KTwins and kQ5KTwins).
void moe_rows_bits(Backend& g) {
    const size_t nin = 2560, few = 24;
    const double grouped_from = 1.25;
    const bool fused = rocm_device(0).arch.rfind("gfx906", 0) == 0 && rocm_device(0).caps.subgroup_size == 64;
    size_t wide_runs = 0, lean_runs = 0;
    struct Shape {
        size_t ne, k, ff;
        uint32_t gt, dt;
    };
    const Shape shapes[] = {{512, 10, 320, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_1}, {512, 10, 640, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q8_0},
                            {128, 6, 320, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q5_1}};
    for (const Shape& sh : shapes) {
        const size_t ne = sh.ne, k = sh.k, ff = sh.ff;
        // The last row count the fused step takes, and one more, which the base takes.
        const size_t last = (size_t)std::ceil(grouped_from * double(ne) / double(k)) - 1, max_rows = last + 1;
        // The rows share most of h, so each routes among the first `few` experts, and the router's own part of those reorders the picks a row.
        const std::vector<float> h0 = row_classes::values(nin, 181), noise = row_classes::values(max_rows * nin, 182, 0.25f), rgw = row_classes::values(nin, 183, 0.01f);
        const std::vector<float> rnoise = row_classes::values(few * nin, 184, 0.02f);
        std::vector<float> h(max_rows * nin);
        for (size_t i = 0; i < h.size(); ++i) h[i] = h0[i % nin] + noise[i];
        double hh = 0;
        for (float v : h0) hh += double(v) * v;
        std::vector<float> router(ne * nin);
        for (size_t e = 0; e < ne; ++e) {
            const double score = e < few ? 1.0 + 0.17 * double((e * 7) % few) : -10.0;
            for (size_t i = 0; i < nin; ++i) router[e * nin + i] = float(h0[i] * score / hh) + (e < few ? rnoise[e * nin + i] : 0.0f);
        }
        const std::vector<float> zero(max_rows * nin, 0.0f);
        auto hb = g.adopt(h.data(), h.size() * 4), rb = g.adopt(router.data(), router.size() * 4), gb = g.adopt(rgw.data(), nin * 4), zb = g.adopt(zero.data(), zero.size() * 4);
        const auto stack = [&](uint32_t type, size_t width, size_t rows, uint32_t seed) {
            const std::vector<uint8_t> m = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(width, few * rows, seed) : typed_matrix(type, width, few * rows, seed);
            BufferPtr b = g.alloc(quant::row_bytes(type, width, ne * rows));
            g.write(*b, 0, m.data(), m.size());
            return b;
        };
        const BufferPtr wg = stack(sh.gt, nin, ff, 185), wu = stack(sh.gt, nin, ff, 186), wd = stack(sh.dt, ff, nin, 187);
        const std::vector<uint8_t> sg = q8_matrix(nin, ff, 188), su = q8_matrix(nin, ff, 189), sd = q8_matrix(ff, nin, 190);
        auto sgb = g.adopt(sg.data(), sg.size()), sub = g.adopt(su.data(), su.size()), sdb = g.adopt(sd.data(), sd.size());
        const std::string up = sh.gt == quant::GGML_TYPE_Q5_K ? "llmx_moe_step_up_q5_k" : "llmx_moe_step_up_q4_k";
        for (const size_t rows : {size_t(1), size_t(2), size_t(3), size_t(4), last - 1, last, last + 1}) {
            const std::string at = std::to_string(rows) + " rows, " + std::to_string(k) + " of " + std::to_string(ne) + " experts, " + quant::storage_type(sh.gt)->name +
                                   " gate and up, " + quant::storage_type(sh.dt)->name + " down, ff " + std::to_string(ff);
            const size_t entries = rows * k;
            struct Out {
                BufferPtr scores, ids, weights, gate, up, act, sgate, sup, sact, rg, y;
                std::vector<RowRun> scratch;
            };
            const auto outs = [&] {
                return Out{g.alloc(rows * ne * 4), g.alloc(entries * 4), g.alloc(entries * 4), g.alloc(entries * ff * 4), g.alloc(entries * ff * 4), g.alloc(entries * ff * 4),
                           g.alloc(rows * ff * 4), g.alloc(rows * ff * 4), g.alloc(rows * ff * 4), g.alloc(rows * 4), g.alloc(rows * nin * 4), {}};
            };
            // The base's outputs, then the step's with h's twin kept and, at one row and at the last, with it not kept.
            Out o[3] = {outs(), outs(), outs()};
            MoeShared shr[3];
            MoeStep st[3];
            for (int a = 0; a < 3; ++a) {
                shr[a] = MoeShared{Projection{{quant::GGML_TYPE_Q8_0, {sgb.get(), 0}}, {o[a].sgate.get(), 0}, ff},
                                   Projection{{quant::GGML_TYPE_Q8_0, {sub.get(), 0}}, {o[a].sup.get(), 0}, ff}, {quant::GGML_TYPE_Q8_0, {sdb.get(), 0}}, {o[a].sact.get(), 0},
                                   Projection{{quant::GGML_TYPE_F32, {gb.get(), 0}}, {o[a].rg.get(), 0}, 1}};
                st[a] = MoeStep{Projection{{quant::GGML_TYPE_F32, {rb.get(), 0}}, {o[a].scores.get(), 0}, ne}, k, backend::RouteRule::softmax(true), {o[a].ids.get(), 0},
                                {o[a].weights.get(), 0}, Projection{{sh.gt, {wg.get(), 0}}, {o[a].gate.get(), 0}, ff}, Projection{{sh.gt, {wu.get(), 0}}, {o[a].up.get(), 0}, ff},
                                {sh.dt, {wd.get(), 0}}, {o[a].act.get(), 0}, &shr[a], zb.get(), &o[a].scratch};
            }
            const RowRun generated[1] = {{rows, 1}};
            rocm_take_launches(g);
            g.Backend::moe_step(st[0], {hb.get(), 0}, {o[0].y.get(), 0}, nin, rows, {generated, 1}, Dtype::int8);
            const auto base = rocm_take_launches(g);
            g.moe_step(st[1], {hb.get(), 0}, {o[1].y.get(), 0}, nin, rows, {generated, 1}, Dtype::int8);
            const auto step = rocm_take_launches(g);
            // A write forgets every kept twin, so the route launch makes h's over all the rows, through its strided twin loop.
            const bool unkept_arm = rows == 1 || rows == last;
            std::map<std::string, size_t> unkept;
            if (unkept_arm) {
                g.write(*hb, 0, h.data(), rows * nin * 4);
                rocm_take_launches(g);
                g.moe_step(st[2], {hb.get(), 0}, {o[2].y.get(), 0}, nin, rows, {generated, 1}, Dtype::int8);
                unkept = rocm_take_launches(g);
            }
            size_t base_steps = 0, step_twins = 0, unkept_twins = 0;
            for (const auto& [name, n] : base) base_steps += name.rfind("llmx_moe_step", 0) == 0 ? n : 0;
            for (const auto& [name, n] : step) step_twins += name.rfind("llmx_twin", 0) == 0 ? n : 0;
            for (const auto& [name, n] : unkept) unkept_twins += name.rfind("llmx_twin", 0) == 0 ? n : 0;
            require(base_steps == 0, "the base's moe_step launches no fused kernel, " + at);
            const std::string down = sh.dt == quant::GGML_TYPE_Q8_0 ? "llmx_moe_step_down_q8_0" : "llmx_moe_step_down_q5_1";
            const size_t wide = step.count(up + "_w") ? step.at(up + "_w") : 0, lean = step.count(up) ? step.at(up) : 0;
            if (fused && rows <= last) {
                require(step.size() == 3 && step.count("llmx_moe_step_route") && step.at("llmx_moe_step_route") == 1 && wide + lean == 1 && step.count(down) && step.at(down) == 1,
                        "moe_step is the three fused launches below the grouped break-even, " + at);
                require(step_twins == 0, "with h's twin kept the fused step launches no twin, " + at);
                if (unkept_arm)
                    require(unkept.size() == 3 && unkept.count("llmx_moe_step_route") && unkept_twins == 0, "with h's twin not kept the route launch makes it, " + at);
                wide_runs += wide, lean_runs += lean;
            } else {
                require(!step.count("llmx_moe_step_route") && wide + lean == 0, "moe_step from the grouped break-even on runs the base's ops, " + at);
                if (unkept_arm) require(!unkept.count("llmx_moe_step_route"), "a step the fused kernels do not take runs the base's ops, " + at);
            }
            for (int a = 1; a < (unkept_arm ? 3 : 2); ++a) {
                const std::string of = std::string(a == 2 ? "h's twin not kept, " : "") + at;
                require(same(read(g, *o[0].ids, entries), read(g, *o[a].ids, entries)), "the step's picks are the base's, " + of);
                require(same(read(g, *o[0].weights, entries), read(g, *o[a].weights, entries)), "the step's weights are the base's, " + of);
                require(same(read(g, *o[0].scores, rows * ne), read(g, *o[a].scores, rows * ne)), "the step's router scores are the base's, " + of);
                require(same(read(g, *o[0].rg, rows), read(g, *o[a].rg, rows)), "the step's row gates are the base's, " + of);
                require(same(read(g, *o[0].act, entries * ff), read(g, *o[a].act, entries * ff)), "the step's routed act is the base's, " + of);
                require(same(read(g, *o[0].sact, rows * ff), read(g, *o[a].sact, rows * ff)), "the step's shared act is the base's, " + of);
                require(same(read(g, *o[0].y, rows * nin), read(g, *o[a].y, rows * nin)), "the step's output is the base's, " + of);
            }
        }
    }
    // One row at ff 320 is 110 workgroups, which the wide build holds resident on a 60-unit gfx906; the rows near the break-even are far past it.
    if (fused) require(wide_runs > 0 && lean_runs > 0, "the fused gate+up takes the wide build where its grid is resident and the lean one elsewhere");
}

// The hyper-connection mix's launches (llmx_hc_mix and its column builds) against the chain of ops they replace (Backend::hyper_mix's own steps), bit for bit, over 1 to 16 generated rows (backend-rocm, `twins`).
// Each column of a launch of several is also the one-column launch's, so verify rows mix as decode rows; with `other`, another device's backend is made current before each fused call, as a layer split's earlier stage leaves it.
void hyper_mix_bits(Backend& g, Backend* other = nullptr) {
    struct Shape {
        size_t streams, dim, low;
        bool inject;
    };
    const Shape shapes[] = {{4, 2560, 320, true}, {4, 2560, 320, false}, {2, 512, 64, true}, {2, 256, 64, true}};
    for (const Shape& sh : shapes)
        for (size_t rows : {1, 2, 3, 4, 5, 8, 9, 13, 16}) {
            const size_t width = sh.streams * sh.dim, nout = 64;
            const std::vector<uint8_t> down = q8_matrix(width, sh.low, 151), up = q8_matrix(sh.low, width, 152), head = q8_matrix(sh.dim, nout, 153);
            const std::vector<float> x = row_classes::values(rows * width, 154), w = row_classes::values(width, 155), inj = row_classes::values(sh.streams * width, 156);
            auto db = g.adopt(down.data(), down.size()), ub = g.adopt(up.data(), up.size()), hb = g.adopt(head.data(), head.size());
            auto xb = g.adopt(x.data(), x.size() * 4), wb = g.adopt(w.data(), w.size() * 4), ib = g.adopt(inj.data(), inj.size() * 4);
            const WeightView hv{quant::GGML_TYPE_Q8_0, {hb.get(), 0}};
            struct Slots {
                BufferPtr out, normed, low, gates, injg, y;
            };
            const auto slots = [&] {
                return Slots{g.alloc(rows * sh.dim * 4), g.alloc(rows * width * 4), g.alloc(rows * sh.low * 4), g.alloc(rows * width * 4), g.alloc(rows * sh.streams * 4),
                             g.alloc(rows * nout * 4)};
            };
            Slots s[3] = {slots(), slots(), slots()};
            const RowRun generated[1] = {{rows, 1}}, decode[1] = {{1, 1}};
            const auto mix = [&](Slots& o, size_t r) {
                return Backend::HyperMix{{wb.get(), 0}, {quant::GGML_TYPE_Q8_0, {db.get(), 0}}, {quant::GGML_TYPE_Q8_0, {ub.get(), 0}},
                                         sh.inject ? WeightView{quant::GGML_TYPE_F32, {ib.get(), 0}} : WeightView{}, {o.normed.get(), r * width}, {o.low.get(), r * sh.low},
                                         {o.gates.get(), r * width}, {o.injg.get(), r * sh.streams}};
            };
            // A product in the 8-bit class first, so the producers write their twins.
            g.matmul(hv, {xb.get(), 0}, {s[0].y.get(), 0}, sh.dim, nout, 1, {}, Dtype::int8);
            rocm_take_launches(g);
            g.Backend::hyper_mix({s[0].out.get(), 0}, {xb.get(), 0}, mix(s[0], 0), rows, sh.streams, sh.dim, sh.low, 1e-6f, {generated, 1}, Dtype::int8);
            g.matmul(hv, {s[0].out.get(), 0}, {s[0].y.get(), 0}, sh.dim, nout, rows, {generated, 1}, Dtype::int8);
            const size_t chain_fused = rocm_take_launches(g).count("llmx_hc_mix");
            if (other) (void)other->alloc(4);
            g.hyper_mix({s[1].out.get(), 0}, {xb.get(), 0}, mix(s[1], 0), rows, sh.streams, sh.dim, sh.low, 1e-6f, {generated, 1}, Dtype::int8);
            const auto fused = rocm_take_launches(g);
            g.matmul(hv, {s[1].out.get(), 0}, {s[1].y.get(), 0}, sh.dim, nout, rows, {generated, 1}, Dtype::int8);
            size_t twins = 0;
            for (const auto& [k, v] : rocm_take_launches(g))
                if (k.rfind("llmx_twin", 0) == 0) twins += v;
            // Each column alone, as a decode row.
            for (size_t r = 0; r < rows; ++r) g.hyper_mix({s[2].out.get(), r * sh.dim}, {xb.get(), r * width}, mix(s[2], r), 1, sh.streams, sh.dim, sh.low, 1e-6f, {decode, 1}, Dtype::int8);
            const auto alone = rocm_take_launches(g);
            size_t launches = 0, wanted = 0;
            for (const auto& [k, v] : fused) launches += k.rfind("llmx_hc_mix", 0) == 0 ? v : 0;
            for (size_t left = rows; left; left -= std::min<size_t>(left, 8)) ++wanted;
            const std::string at = std::to_string(sh.streams) + "x" + std::to_string(sh.dim) + " low " + std::to_string(sh.low) + (sh.inject ? " with inject gates" : "") + ", " +
                                   std::to_string(rows) + " rows";
            require(chain_fused == 0, "the hyper mix's chain launches no llmx_hc_mix, " + at);
            const size_t wide = fused.count("llmx_hc_mix8") ? fused.at("llmx_hc_mix8") : 0;
            require(launches == wanted && wide == rows / 8 + (rows % 8 > 1) && (rows > 1 || fused.count("llmx_hc_mix")), "the hyper mix is llmx_hc_mix8 launches of 8 columns at most and llmx_hc_mix for one, " + at);
            for (const auto& [k, v] : fused) require(k.rfind("llmx_hc_mix", 0) == 0, "the fused hyper mix launches " + k + ", " + at);
            require(alone.size() == 1 && alone.count("llmx_hc_mix") && alone.at("llmx_hc_mix") == rows, "a column alone is one llmx_hc_mix launch, " + at);
            require(twins == 0, "a product of the fused mix's output reads its kept twin, " + at);
            for (int k = 1; k < 3; ++k) {
                const std::string of = (k == 1 ? "the fused mix's " : "a column alone's ") + std::string("%s are the chain's, ") + at;
                const auto says = [&](const char* what) {
                    std::string m = of;
                    m.replace(m.find("%s"), 2, what);
                    return m;
                };
                require(same(read(g, *s[0].out, rows * sh.dim), read(g, *s[k].out, rows * sh.dim)), says("outputs"));
                require(same(read(g, *s[0].normed, rows * width), read(g, *s[k].normed, rows * width)), says("normed streams"));
                require(same(read(g, *s[0].low, rows * sh.low), read(g, *s[k].low, rows * sh.low)), says("low rows"));
                require(same(read(g, *s[0].gates, rows * width), read(g, *s[k].gates, rows * width)), says("gates"));
                if (sh.inject) require(same(read(g, *s[0].injg, rows * sh.streams), read(g, *s[k].injg, rows * sh.streams)), says("inject gates"));
            }
            require(same(read(g, *s[0].y, rows * nout), read(g, *s[1].y, rows * nout)), "a product of the fused mix's output and twin is the chain's, " + at);
        }
}

// group_rms_norm takes four values a thread where its operands lie on 16 bytes and one a thread at an offset of one float: both give the same output and twin, at dims 256, 1024 and 2560, generated rows and runs of both kernels.
void group_norm_four(Backend& g, Dtype dt) {
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    const size_t rows = 13, nout = 24;
    const RowRun decode[1] = {{rows, 1}}, mixed[3] = {{3, 1}, {10, 7}, {13, 1}};
    for (const auto& [groups, dim] : {std::pair<size_t, size_t>{4, 256}, {1, 1024}, {1, 2560}}) {
        const size_t width = groups * dim, n = rows * width;
        const std::vector<uint8_t> m = q8_matrix(width, nout, 141);
        std::vector<float> a = row_classes::values(n + 1, 142), w = row_classes::values(width + 1, 143);
        a[3] = 1e30f, a[40] = 1e-30f;   // a huge and a tiny value beside ordinary ones
        for (size_t i = 64; i < 96; ++i) a[i + 1] = 0.0f;   // a zero block at the offset form's input
        auto mb = g.adopt(m.data(), m.size()), ab = g.adopt(a.data(), a.size() * 4), wb = g.adopt(w.data(), w.size() * 4);
        auto o0 = g.alloc(n * 4), o1 = g.alloc((n + 1) * 4), y0 = g.alloc(nout * rows * 4), y1 = g.alloc(nout * rows * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
        auto a0 = g.alloc(n * 4), w0 = g.alloc(width * 4);
        g.copy(*a0, 0, *ab, 4, n * 4);
        g.copy(*w0, 0, *wb, 4, width * 4);
        g.matmul(wv, {a0.get(), 0}, {y0.get(), 0}, width, nout, 1, {}, dt);
        for (const RowRuns runs : {RowRuns{decode, 1}, RowRuns{mixed, 3}}) {
            g.group_rms_norm({o0.get(), 0}, {a0.get(), 0}, {w0.get(), 0}, rows, groups, dim, 1e-6f, runs);
            g.matmul(wv, {o0.get(), 0}, {y0.get(), 0}, width, nout, rows, runs, dt);
            g.group_rms_norm({o1.get(), 1}, {ab.get(), 1}, {wb.get(), 1}, rows, groups, dim, 1e-6f, runs);
            g.matmul(wv, {o1.get(), 1}, {y1.get(), 0}, width, nout, rows, runs, dt);
            const std::string at = std::to_string(groups) + "x" + std::to_string(dim) + (runs.n == 1 ? ", generated rows" : ", runs of both kernels");
            require(same(read(g, *o0, n), read(g, *o1, n, 1)), form + ": group_rms_norm on 16 bytes is not its output at an offset of one float, " + at);
            require(same(read(g, *y0, nout * rows), read(g, *y1, nout * rows)), form + ": a product of group_rms_norm's output on 16 bytes is not its product at an offset of one float, " + at);
        }
    }
}

// The stream and gate ops write their outputs' twins as the other producers do: a product of the output launches no twin build and gives the bits of the same product of a copy, over generated rows and over runs of both kernels.
// The kept twin of a product's input survives an exact product beside it and goes once anything writes that input or the twin's buffer grows.
void stream_twins(Backend& g, Dtype dt) {
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    const size_t nin = 1024, nout = 40, rows = 13;
    const std::vector<uint8_t> m = q8_matrix(nin, nout, 131);
    const std::vector<float> a = row_classes::values(rows * 4 * nin, 132), b = row_classes::values(rows * 4 * nin, 133), w = row_classes::values(nin, 134);
    auto mb = g.adopt(m.data(), m.size()), ab = g.adopt(a.data(), a.size() * 4), bb = g.adopt(b.data(), b.size() * 4), wb = g.adopt(w.data(), w.size() * 4);
    auto ob = g.alloc(rows * nin * 4), cb = g.alloc(rows * nin * 4), y = g.alloc(nout * rows * 4), yc = g.alloc(nout * rows * 4);
    const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
    const auto twins = [&] {
        size_t n = 0;
        for (const auto& [k, v] : rocm_take_launches(g))
            if (k.rfind("llmx_twin", 0) == 0) n += v;
        return n;
    };
    // The product of ob, then of its copy: the same bits, and the first builds no twin when `kept`.
    const auto product_of = [&](RowRuns runs, size_t n, bool kept, const std::string& what) {
        rocm_take_launches(g);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, n, runs, dt);
        const size_t built = twins();
        g.copy(*cb, 0, *ob, 0, n * nin * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, nin, nout, n, runs, dt);
        require(same(read(g, *y, nout * n), read(g, *yc, nout * n)), form + ": " + what + " gives the bits of a product of its copy");
        require((built == 0) == kept, form + ": " + what + (kept ? " reads the kept twin" : " builds its twin"));
    };
    const RowRun decode[1] = {{rows, 1}}, mixed[3] = {{3, 1}, {10, 7}, {13, 1}};
    // A product in the class first, so the producers write its twins.
    g.matmul(wv, {ab.get(), 0}, {y.get(), 0}, nin, nout, 1, {}, dt);
    for (const RowRuns runs : {RowRuns{decode, 1}, RowRuns{mixed, 3}}) {
        const std::string at = runs.n == 1 ? "generated rows" : "runs of both kernels";
        g.sigmoid_mul({ob.get(), 0}, {ab.get(), 0}, {bb.get(), 0}, rows, 8, 128, nin, 128, runs);
        product_of(runs, rows, true, "a product of sigmoid_mul's output, " + at);
        g.gated_rms_norm({ob.get(), 0}, {ab.get(), 0}, {bb.get(), 0}, {wb.get(), 0}, rows, 8, 128, 1e-6f, runs, NormGate::silu);
        product_of(runs, rows, true, "a product of gated_rms_norm's output, " + at);
        g.group_rms_norm({ob.get(), 0}, {ab.get(), 0}, {wb.get(), 0}, rows, 4, 256, 1e-6f, runs);
        product_of(runs, rows, true, "a product of group_rms_norm's output, " + at);
        g.silu_scaled({ob.get(), 0}, {ab.get(), 0}, rows, nin, 0.75f, runs);
        product_of(runs, rows, true, "a product of silu_scaled's output, " + at);
        g.mix_streams({ob.get(), 0}, {ab.get(), 0}, {bb.get(), 0}, rows, 4, nin, runs);
        product_of(runs, rows, true, "a product of mix_streams' output, " + at);
    }
    // A product's own twin is kept beside an exact product of the same input, and dropped by an exact product into it, a write into it or a grown twin buffer.
    const std::vector<float> wf = row_classes::values(nin * 8, 135);
    auto fb = g.adopt(wf.data(), wf.size() * 4), fy = g.alloc(8 * rows * 4);
    const WeightView fv{quant::GGML_TYPE_F32, {fb.get(), 0}};
    const RowRuns one{decode, 1};
    g.silu_scaled({ob.get(), 0}, {ab.get(), 0}, rows, nin, 1.0f, {});
    g.copy(*cb, 0, *ob, 0, rows * nin * 4);
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.matmul(fv, {ob.get(), 0}, {fy.get(), 0}, nin, 8, rows, one, dt);
    product_of(one, rows, true, "a product after an exact product of its input");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.matmul(fv, {ab.get(), 0}, {ob.get(), 0}, nin, 8, 1, {}, dt);
    product_of(one, rows, false, "a product after an exact product into its input");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.write(*ob, 0, b.data(), nin * 4);
    product_of(one, rows, false, "a product after a write into its input");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    // Two twins are kept, the newer first: another twin, grown or a producer's, leaves this one, a third drops the older, and a copy or an exact product forgets one only where it writes its input.
    auto big = g.adopt(a.data(), a.size() * 4), sb = g.alloc(rows * nin * 4);
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.matmul(wv, {big.get(), 0}, {fy.get(), 0}, nin, 1, rows * 4, {}, dt);
    product_of(one, rows, true, "a product after a larger twin grew the other slot");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.silu_mul({sb.get(), 0}, {ab.get(), 0}, {bb.get(), 0}, rows * nin, 0.0f, one);
    product_of(one, rows, true, "a product after another producer's twin");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.matmul(wv, {ab.get(), 0}, {fy.get(), 0}, nin, 1, 1, {}, dt);
    g.matmul(wv, {big.get(), 0}, {fy.get(), 0}, nin, 1, rows * 4, {}, dt);
    product_of(one, rows, false, "a product after two other twins");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.copy(*sb, 0, *ab, 0, nin * 4);
    product_of(one, rows, true, "a product after a copy elsewhere");
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.copy(*ob, 4, *ab, 0, nin * 4);
    product_of(one, rows, false, "a product after a copy into its input");
    g.matmul(wv, {ab.get(), 0}, {fy.get(), 0}, nin, 1, 1, {}, dt);
    g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, rows, one, dt);
    g.silu_mul({sb.get(), 0}, {ab.get(), 0}, {bb.get(), 0}, rows * nin, 0.0f, one);
    g.matmul(fv, {ab.get(), 0}, {ob.get(), 0}, nin, 8, 1, {}, dt);
    product_of(one, rows, false, "a product after an exact product into the older twin's input");
    group_norm_four(g, dt);
}

// Attention times at heads 128 and 256 wide over an f16 cache, generated rows after a history and prompts on the tile; checks nothing.
void attention_rates(Backend& g) {
    struct Case { int n_head, head_dim; size_t hist, rows, extent; };
    std::cout << "backend-rocm: attention rates; heads, width, history, rows, extent: us a call, kernels\n";
    for (const Case cs : {Case{16, 128, 4096, 1, 1}, Case{16, 128, 4096, 16, 1}, Case{16, 128, 16000, 1, 1}, Case{16, 128, 0, 512, 512}, Case{16, 128, 4096, 512, 512},
                          Case{24, 256, 16384, 1, 1}, Case{24, 256, 16384, 4, 1}, Case{24, 256, 16384, 16, 16}, Case{24, 256, 16384, 64, 64}, Case{24, 256, 16384, 512, 512}}) {
        const int n_head = cs.n_head, n_head_kv = 2, head_dim = cs.head_dim;
        const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
        const size_t all = cs.hist + cs.rows;
        const std::vector<float> keys = row_classes::values(all * kvw, 81), vals = row_classes::values(all * kvw, 82), q = row_classes::values(cs.rows * qw, 83, 4.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
        auto ob = g.alloc(cs.rows * qw * 4);
        auto st = g.kv_alloc(1, n_head_kv, head_dim, 16896, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, g.kv_layout().block_tokens);
        s.prepare(all);
        KVView v = s.view(st.get());
        g.kv_write(0, &v, 1, {kb.get(), 0}, {vb.get(), 0});
        v.length = cs.hist;
        v.nq = cs.rows;
        v.extent = cs.extent;
        const auto call = [&] { g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim); };
        for (int i = 0; i < 3; ++i) call();
        g.sync();
        rocm_take_launches(g);
        const int iters = 20;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) call();
        g.sync();
        const double us = 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::string ran;
        for (const auto& [k, n] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + k;
        std::printf("backend-rocm: attention rate %d, %d, %zu, %zu, %zu: %.1f us, %s\n", n_head, head_dim, cs.hist, cs.rows, cs.extent, us, ran.c_str());
        s.abort();
    }
    std::fflush(stdout);
}

// argmax_rows over Flash-Next's 248320 logits at 1 and 16 rows, at each slice the profile may take: printed, not checked.
void argmax_rates(Backend& g) {
    const size_t n = 248320;
    const size_t keep = rocm_profile(g).argmax_slice;
    for (const size_t rows : {size_t(1), size_t(16)}) {
        const std::vector<float> logits = row_classes::values(rows * n, 91, 8.0f);
        auto lb = g.adopt(logits.data(), logits.size() * 4), ib = g.alloc(rows * 4);
        for (const size_t slice : {size_t(1024), size_t(4096), size_t(16384), size_t(65536), n}) {
            rocm_profile(g).argmax_slice = slice;
            const auto call = [&] { g.argmax_rows({ib.get(), 0}, {lb.get(), 0}, rows, n); };
            for (int i = 0; i < 3; ++i) call();
            g.sync();
            const int iters = 50;
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) call();
            g.sync();
            const double us = 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
            std::printf("backend-rocm: argmax rate %zu rows of %zu, slice %zu: %.1f us\n", rows, n, slice, us);
        }
    }
    rocm_profile(g).argmax_slice = keep;
    std::fflush(stdout);
}

// The per-row attention's bits on gfx906 after 100 and 3000 tokens, one split and several, pinned so a build that sums in another order fails here (docs/STATUS.md, ROCm block-major twin).
void vec_bits(Backend& g) {
    const int n_head = 8, n_head_kv = 2, head_dim = 128;
    const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim, rows = 3;
    const bool pinned = g.identity().find(" gfx906") != std::string::npos;
    struct Case { size_t hist; uint64_t want; };
    for (const Case cs : {Case{100, 0x6dd90927b3b484a2ull}, Case{3000, 0x2abdc799a2845129ull}}) {
        const size_t all = cs.hist + rows;
        const std::vector<float> keys = row_classes::values(all * kvw, 71), vals = row_classes::values(all * kvw, 72), q = row_classes::values(rows * qw, 73, 4.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
        auto ob = g.alloc(rows * qw * 4);
        auto st = g.kv_alloc(1, n_head_kv, head_dim, 3200, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, g.kv_layout().block_tokens);
        s.prepare(all);
        KVView v = s.view(st.get());
        g.kv_write(0, &v, 1, {kb.get(), 0}, {vb.get(), 0});
        v.length = cs.hist;
        v.nq = rows;
        v.extent = 1;
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        const std::vector<float> out = read(g, *ob, rows * qw);
        uint64_t h = 1469598103934665603ull;
        for (float x : out) {
            uint32_t u;
            std::memcpy(&u, &x, 4);
            for (int k = 0; k < 4; ++k) h = (h ^ ((u >> (8 * k)) & 0xff)) * 1099511628211ull;
        }
        std::cout << "per-row attention after " << cs.hist << " tokens: bits 0x" << std::hex << h << std::dec << (pinned ? "" : " (not pinned on this device)") << "\n";
        if (pinned) require(h == cs.want, "the per-row attention after " + std::to_string(cs.hist) + " tokens gives its pinned bits on gfx906");
        s.abort();
    }
}

// The scratch a fit prices holds every pass of its widths (Backend::reserve_scratch, docs/src/backends-rocm.md, Scratch), on a synthetic Qwen3 and on a GGUF `file` when given.
// HIP's free memory, polled through a server's passes, falls below its reading by no more than the device keeps; a model whose scratch was made for one row must show the growth.
void scratch_reserved(const std::string& file, int index) {
    const auto used = [index] {
        static thread_local bool set = false;
        if (!set) set = hip().hipSetDevice(index) == hipSuccess;
        size_t f = 0, t = 0;
        return hip().hipMemGetInfo(&f, &t) == hipSuccess ? t - f : 0;
    };
    const auto make = [index] { return make_rocm_backend(index); };
    const std::string name = "rocm:" + std::to_string(index);
    std::string why;
    const gguf::GGUFModel qwen3 = infer::synthetic_model({2, 1024, 1024, 8, 2, 128, 8192, 20261111u});
    require(scratch_growth::held(infer::gguf_weights(qwen3), make, name, used, true, why), "Qwen3: " + why);
    if (file.empty()) return;
    gguf::GGUFModel m = gguf::read_gguf(file);
    gguf::map_payload(m);
    require(scratch_growth::held(infer::gguf_weights(m), make, name, used, false, why), file + ": " + why);
}

// A prompt's rows whose histories cross a change in the attention's chunks, at heads 128 and 256 wide and at 16K on the tile: each row the bits of that row alone in its class.
void attention_chunks(Backend& g) {
    const int n_head = 4, n_head_kv = 2;
    for (int head_dim : {128, 256})
    for (size_t hist : {size_t(250), size_t(4090), size_t(16300)}) {
        if (head_dim == 256 && hist > 8192) continue;
        const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
        const size_t rows = hist > 8192 ? 40 : 13;
        const std::vector<float> keys = row_classes::values((hist + rows) * kvw, 41), vals = row_classes::values((hist + rows) * kvw, 42);
        const std::vector<float> q = row_classes::values(rows * qw, 43, 4.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4);
        auto qb = g.adopt(q.data(), q.size() * 4), ob = g.alloc(rows * qw * 4), o1 = g.alloc(qw * 4);
        auto st = g.kv_alloc(1, n_head_kv, head_dim, 16448, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, g.kv_layout().block_tokens);
        s.prepare(hist + rows);
        KVView all = s.view(st.get());
        g.kv_write(0, &all, 1, {kb.get(), 0}, {vb.get(), 0});
        KVView v = all;
        v.length = hist;
        v.nq = rows;
        v.extent = rows;   // the prompt's extent, which a row alone below keeps, so both are one row class
        rocm_take_launches(g);
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        const std::vector<float> together = read(g, *ob, rows * qw);
        if (head_dim == 256) {
            const auto ran = rocm_take_launches(g);
            require(ran.count("llmx_attention_vec256") || ran.count("llmx_attention_vec256_major"), "a 256-wide prompt's rows across a chunk change run the per-row body");
        }
        bool alone = true;
        for (size_t b = 0; b < rows; ++b) {
            KVView one = all;
            one.length = hist + b;
            one.nq = 1;
            one.extent = rows;
            g.attention({qb.get(), b * qw}, 0, &one, 1, {o1.get(), 0}, n_head, n_head_kv, head_dim);
            alone = alone && std::memcmp(read(g, *o1, qw).data(), together.data() + b * qw, qw * 4) == 0;
        }
        require(alone, "attention rows of heads " + std::to_string(head_dim) + " wide across a chunk change after a history of " + std::to_string(hist) + " are each the row alone, bit for bit");
        s.abort();
    }
}

// The fused norm_rope_kv against the base's ops one at a time on the same backend: q, k and both KV sides bit for bit, two sequences after histories, f32 and f16 caches, with and without the norms' weights.
void fused_norm_rope_kv(Backend& g) {
    const size_t n_head = 8, n_head_kv = 2, dim = 128, half = dim / 2, qw = n_head * dim, kvw = n_head_kv * dim, ctx = 256;
    const std::vector<float> cosv = row_classes::values(ctx * half, 51), sinv = row_classes::values(ctx * half, 52);
    const std::vector<float> qwt = row_classes::values(dim, 53), kwt = row_classes::values(dim, 54);
    bool ok = true;
    for (KVType kt : {KVType::f32, KVType::f16})
        for (bool weighted : {true, false})
            for (size_t hist : {size_t(0), size_t(61)}) {
                const size_t rows = 3;
                const std::vector<float> q = row_classes::values(rows * 2 * qw, 55, 3.0f), k = row_classes::values(rows * 2 * kvw, 56, 3.0f),
                                         v = row_classes::values(rows * 2 * kvw, 57);
                auto cb = g.adopt(cosv.data(), cosv.size() * 4), sb = g.adopt(sinv.data(), sinv.size() * 4);
                auto qwb = g.adopt(qwt.data(), qwt.size() * 4), kwb = g.adopt(kwt.data(), kwt.size() * 4);
                std::vector<float> outs[2][2];
                std::vector<uint8_t> caches[2][2];
                for (int fused = 0; fused < 2; ++fused) {
                    auto qb = g.adopt(q.data(), q.size() * 4), kb = g.adopt(k.data(), k.size() * 4), vb = g.adopt(v.data(), v.size() * 4);
                    auto st = g.kv_alloc(1, n_head_kv, dim, 512, kt, kt);
                    infer::BlockPool pool;
                    pool.configure(st->max_blocks());
                    infer::KVSequence s1(&pool, g.kv_layout().block_tokens), s2(&pool, g.kv_layout().block_tokens);
                    if (hist) {
                        for (infer::KVSequence* s : {&s1, &s2}) {
                            s->prepare(hist);
                            s->commit();
                        }
                    }
                    s1.prepare(rows);
                    s2.prepare(rows);
                    const KVView views[2] = {s1.view(st.get()), s2.view(st.get())};
                    std::vector<uint32_t> pos(rows * 2);
                    for (size_t r = 0; r < pos.size(); ++r) pos[r] = (uint32_t)(hist + r % rows + 7 * (r / rows));
                    const Backend::RopeArgs rope{{cb.get(), 0}, {sb.get(), 0}, half, pos.data(), 1e-6f};
                    const CSlice qn = weighted ? CSlice{qwb.get(), 0} : CSlice{}, kn = weighted ? CSlice{kwb.get(), 0} : CSlice{};
                    if (fused) g.norm_rope_kv({qb.get(), 0}, qw, n_head, qn, {kb.get(), 0}, {vb.get(), 0}, kvw, n_head_kv, kn, rope, rows * 2, 0, views, 2);
                    else g.Backend::norm_rope_kv({qb.get(), 0}, qw, n_head, qn, {kb.get(), 0}, {vb.get(), 0}, kvw, n_head_kv, kn, rope, rows * 2, 0, views, 2);
                    outs[fused][0] = read(g, *qb, q.size());
                    outs[fused][1] = read(g, *kb, k.size());
                    auto* bs = dynamic_cast<BlockKVStorage*>(st.get());
                    for (int side = 0; side < 2; ++side) {
                        const Buffer& buf = side ? *bs->v_buffer(0) : *bs->k_buffer(0);
                        caches[fused][side].resize(buf.size());
                        g.read(buf, 0, caches[fused][side].data(), buf.size());
                    }
                    s1.abort();
                    s2.abort();
                }
                ok = ok && same(outs[0][0], outs[1][0]) && same(outs[0][1], outs[1][1]) && caches[0][0] == caches[1][0] && caches[0][1] == caches[1][1];
            }
    require(ok, "the fused norm_rope_kv gives q, k and both KV sides of the ops one at a time, bit for bit");
}

// Attention's output twin: a product of the output gives the bits of a product of its copy, whose twin llmx_twin8 or llmx_twin16 makes, at one split and at several, and with a row on the tile kernel, which writes none.
void attention_twins(Backend& g, Dtype dt) {
    const int n_head = 8, n_head_kv = 2, head_dim = 128;
    const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim, nout = 40;
    const std::vector<uint8_t> m = q8_matrix(qw, nout, 61);
    auto mb = g.adopt(m.data(), m.size());
    const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
    bool ok = true;
    struct Case { size_t hist, rows, extent; };
    for (const Case cs : {Case{10, 1, 1}, Case{200, 1, 1}, Case{200, 3, 1}, Case{70, 5, 5}, Case{70, 40, 40}}) {
        const size_t rows = cs.rows, all = cs.hist + rows;
        const std::vector<float> keys = row_classes::values(all * kvw, 62), vals = row_classes::values(all * kvw, 63), q = row_classes::values(rows * qw, 64, 4.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
        auto ob = g.alloc(rows * qw * 4), cb = g.alloc(rows * qw * 4), y = g.alloc(nout * rows * 4), yc = g.alloc(nout * rows * 4);
        auto st = g.kv_alloc(1, n_head_kv, head_dim, 512, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, g.kv_layout().block_tokens);
        s.prepare(all);
        KVView v = s.view(st.get());
        g.kv_write(0, &v, 1, {kb.get(), 0}, {vb.get(), 0});
        v.length = cs.hist;
        v.nq = rows;
        v.extent = cs.extent;
        const RowRun run[1] = {{rows, cs.extent}};
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, qw, nout, rows, {run, 1}, dt);
        g.copy(*cb, 0, *ob, 0, rows * qw * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, qw, nout, rows, {run, 1}, dt);
        ok = ok && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        s.abort();
    }
    require(ok, std::string(dt == Dtype::int8 ? "8-bit" : "16-bit") + ": a product of attention's output gives the bits of a product of its copy");
}

// Every pair of extents of one class gives the same bits, through a matmul and through attention after a history (row_classes.hpp's shapes).
void classes(Backend& g) {
    require(g.row_class(1) != g.row_class(2), "a generated token is a class of its own");
    size_t pairs = 0;
    const size_t rows = 4, nout = 64;
    for (size_t nin : {size_t(256), size_t(4096)}) {
        const std::vector<float> w = row_classes::values(nin * nout, 71), x = row_classes::values(rows * nin, 72);
        auto wb = g.adopt(w.data(), w.size() * 4), xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(rows * nout * 4);
        std::vector<std::vector<float>> out;
        for (size_t e : row_classes::kExtents) {
            const RowRun run[1] = {{rows, e}};
            g.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, rows, {run, 1}, Dtype::f32);
            out.push_back(read(g, *y, rows * nout));
        }
        pairs += row_classes::same_within_classes(g, out, "an F32 matmul, rows " + std::to_string(nin) + " wide");
    }
    // Attention of four query rows after a 70-token history at each extent, on both sides of the attention tile's threshold: rows of one class give the same bits.
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t hist = 70, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    const std::vector<float> keys = row_classes::values((hist + rows) * kvw, 79), vals = row_classes::values((hist + rows) * kvw, 80);
    const std::vector<float> q = row_classes::values(rows * qw, 81, 4.0f);
    auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
    auto ob = g.alloc(rows * qw * 4);
    auto st = g.kv_alloc(1, n_head_kv, head_dim, 512, KVType::f16, KVType::f16);
    infer::BlockPool pool;
    pool.configure(st->max_blocks());
    std::vector<std::vector<float>> att;
    for (size_t e : row_classes::kExtents) {
        infer::KVSequence seq(&pool, g.kv_layout().block_tokens);
        seq.prepare(hist);
        const KVView h = seq.view(st.get());
        g.kv_write(0, &h, 1, {kb.get(), 0}, {vb.get(), 0});
        seq.commit();
        seq.prepare(rows);
        KVView v = seq.view(st.get());
        v.extent = e;
        g.kv_write(0, &v, 1, {kb.get(), hist * kvw}, {vb.get(), hist * kvw});
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        att.push_back(read(g, *ob, rows * qw));
        seq.abort();
    }
    const size_t att_pairs = row_classes::same_within_classes(g, att, "attention");
    require(att_pairs > 0, "the attention row-class check compared pairs");
    pairs += att_pairs;
    require(pairs > 0, "the row-class check compared pairs");
}

// pass_arg: 9000 decode steps of two layers take the device ring round, and the last step gives a fresh backend's bits.
void pass_args(Backend& g) {
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t steps = 9000, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    const std::vector<float> keys = row_classes::values(steps * kvw, 91), vals = row_classes::values(steps * kvw, 92), q = row_classes::values(qw, 93, 4.0f);
    std::vector<std::vector<float>> out;
    for (int fresh = 0; fresh < 2; ++fresh) {
        BackendPtr own = fresh ? make_rocm_backend(0) : nullptr;
        Backend& b = fresh ? *own : g;
        auto kb = b.adopt(keys.data(), keys.size() * 4), vb = b.adopt(vals.data(), vals.size() * 4), qb = b.adopt(q.data(), q.size() * 4);
        auto ob = b.alloc(2 * qw * 4);
        auto st = b.kv_alloc(2, n_head_kv, head_dim, steps, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, b.kv_layout().block_tokens);
        if (!fresh) {
            for (size_t t = 0; t < steps; ++t) {
                s.prepare(1);
                const KVView v = s.view(st.get());
                for (size_t l = 0; l < 2; ++l) {
                    b.kv_write(l, &v, 1, {kb.get(), t * kvw}, {vb.get(), t * kvw});
                    b.attention({qb.get(), 0}, l, &v, 1, {ob.get(), l * qw}, n_head, n_head_kv, head_dim);
                }
                s.commit();
                (void)b.submit();
            }
        } else {
            s.prepare(steps);
            KVView v = s.view(st.get());
            for (size_t l = 0; l < 2; ++l) b.kv_write(l, &v, 1, {kb.get(), 0}, {vb.get(), 0});
            v.length = steps - 1;
            v.nq = 1;
            for (size_t l = 0; l < 2; ++l) b.attention({qb.get(), 0}, l, &v, 1, {ob.get(), l * qw}, n_head, n_head_kv, head_dim);
        }
        out.push_back(read(b, *ob, 2 * qw));
    }
    require(same(out[0], out[1]), "attention after 9000 decode steps of two layers, its pass values round the device ring, gives a fresh backend's bits");
}

// Launches HIP was asked to make, and a refusal to give, read by the records check through the HIP call table.
int issued = 0;
bool refuse_launch = false;
decltype(HipFn::hipModuleLaunchKernel) issued_real = nullptr;
hipError_t issued_launch(hipFunction_t k, unsigned gx, unsigned gy, unsigned gz, unsigned tx, unsigned ty, unsigned tz, unsigned shared, hipStream_t s, void** args, void** extra) {
    ++issued;
    if (refuse_launch) return hipErrorInvalidValue;
    return issued_real(k, gx, gy, gz, tx, ty, tz, shared, s, args, extra);
}

// Held launches reach HIP at the profile's chunk or at a submission, and a decode of two layers through KV growth, copies and pass values gives the same bits at a chunk of 1 (every launch at once), 64 and 100000 (docs/TESTS.md, backend-rocm).
void records() {
    HipFn& f = hip();
    issued_real = f.hipModuleLaunchKernel;
    {
        BackendPtr b = make_rocm_backend(0);
        auto x = b->alloc(4096);
        b->sync();
        rocm_profile(*b).dispatch_chunk = 8;
        f.hipModuleLaunchKernel = issued_launch;
        issued = 0;
        for (int i = 0; i < 7; ++i) b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int held = issued;
        b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int chunk = issued;
        for (int i = 0; i < 3; ++i) b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int before = issued;
        b->wait(b->submit());
        f.hipModuleLaunchKernel = issued_real;
        require(held == 0 && chunk == 8 && before == 8 && issued == 11, "held launches reached HIP other than at the chunk of 8 and the submission: " + std::to_string(held) + ", " + std::to_string(chunk) + ", " +
                                                                              std::to_string(before) + ", " + std::to_string(issued));
        // With nothing running, the first 8 held launches go at once and the rest at the profile's chunk.
        rocm_profile(*b).dispatch_chunk = 64;
        f.hipModuleLaunchKernel = issued_launch;
        issued = 0;
        for (int i = 0; i < 7; ++i) b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int idle_held = issued;
        b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int first = issued;
        for (int i = 0; i < 63; ++i) b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int rest = issued;
        b->add({x.get(), 0}, {x.get(), 0}, 1024);
        const int full = issued;
        b->wait(b->submit());
        f.hipModuleLaunchKernel = issued_real;
        require(idle_held == 0 && first == 8 && rest == 8 && full == 72, "an idle device's held launches reached HIP other than at its first 8 and then the chunk of 64: " + std::to_string(idle_held) + ", " +
                                                                              std::to_string(first) + ", " + std::to_string(rest) + ", " + std::to_string(full));
        // A launch HIP refuses as it is issued throws naming its kernel, and the backend goes on.
        rocm_profile(*b).dispatch_chunk = 64;
        b->add({x.get(), 0}, {x.get(), 0}, 1024);
        f.hipModuleLaunchKernel = issued_launch;
        refuse_launch = true;
        std::string what;
        try {
            b->submit();
        } catch (const std::exception& e) {
            what = e.what();
        }
        refuse_launch = false;
        f.hipModuleLaunchKernel = issued_real;
        require(what.find("llmx_add") != std::string::npos, "a held launch HIP refused threw without its kernel's name: " + what);
        b->add({x.get(), 0}, {x.get(), 0}, 1024);
        require(!throws([&] { b->wait(b->submit()); }), "a backend whose held launch HIP refused did not go on");
    }
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t steps = 300, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    const std::vector<float> keys = row_classes::values(steps * kvw, 94), vals = row_classes::values(steps * kvw, 95), q = row_classes::values(steps * qw, 96, 4.0f);
    std::vector<std::vector<float>> out;
    for (const uint32_t chunk : {1u, 64u, 100000u}) {
        BackendPtr b = make_rocm_backend(0);
        rocm_profile(*b).dispatch_chunk = chunk;
        auto kb = b->adopt(keys.data(), keys.size() * 4), vb = b->adopt(vals.data(), vals.size() * 4), qb = b->adopt(q.data(), q.size() * 4);
        auto ob = b->alloc(2 * qw * 4), all = b->alloc(steps * 2 * qw * 4);
        // The storage backs blocks as the history reaches them, so the decode crosses KV growth.
        auto st = b->kv_alloc(2, n_head_kv, head_dim, steps, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, b->kv_layout().block_tokens);
        for (size_t t = 0; t < steps; ++t) {
            s.prepare(1);
            const KVView v = s.view(st.get());
            for (size_t l = 0; l < 2; ++l) {
                b->kv_write(l, &v, 1, {kb.get(), t * kvw}, {vb.get(), t * kvw});
                b->attention({qb.get(), t * qw}, l, &v, 1, {ob.get(), l * qw}, n_head, n_head_kv, head_dim);
            }
            b->copy(*all, t * 2 * qw * 4, *ob, 0, 2 * qw * 4);
            s.commit();
            if (t % 7 == 0) (void)b->submit();
        }
        out.push_back(read(*b, *all, steps * 2 * qw));
    }
    require(same(out[0], out[1]) && same(out[0], out[2]), "a decode of two layers held in chunks of 64 or 100000 is not its bits launched at once");
}

// A failure the device reports at a ticket's event is kept as the first cause: raise(t) before results are read in place, a read and the next launch throw it.
void failure() {
    BackendPtr g = make_rocm_backend(0);
    HipFn& f = hip();
    const auto synchronize = f.hipEventSynchronize;
    auto buf = g->alloc(4096);
    const Ticket t = g->submit();
    f.hipEventSynchronize = [](hipEvent_t) { return hipErrorLaunchFailure; };
    g->wait(t);
    f.hipEventSynchronize = synchronize;
    std::vector<float> got(1024);
    require(throws([&] { g->raise(t); }), "raise throws a failure the device reported at the ticket's event");
    require(throws([&] { g->read(*buf, 0, got.data(), 4096); }), "a read after the failure throws it");
    require(throws([&] { g->add({buf.get(), 0}, {buf.get(), 0}, 1024); }), "a launch after the failure throws it");
}

// Counts of the group sum's launches and copies, and a launch to drop, read by the collective checks through the HIP call table.
struct GroupCalls {
    int launches = 0, copies = 0, fail = 0;   // fail: the launch, counted from 1, that HIP refuses
    int late = 0;                             // late: the launch, counted from 1, that the host makes only after a stall
    std::vector<void*> areas;                 // every uncached allocation, in order
};
GroupCalls group_calls;
decltype(HipFn::hipModuleLaunchKernel) real_launch = nullptr;
decltype(HipFn::hipMemcpyAsync) real_copy = nullptr;
decltype(HipFn::hipExtMallocWithFlags) real_uncached = nullptr;
hipError_t counted_launch(hipFunction_t k, unsigned gx, unsigned gy, unsigned gz, unsigned tx, unsigned ty, unsigned tz, unsigned shared, hipStream_t s, void** args, void** extra) {
    if (++group_calls.launches == group_calls.fail) return hipErrorLaunchFailure;
    if (group_calls.launches == group_calls.late) std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    return real_launch(k, gx, gy, gz, tx, ty, tz, shared, s, args, extra);
}
hipError_t counted_copy(void* d, const void* s, size_t n, hipMemcpyKind kind, hipStream_t st) {
    ++group_calls.copies;
    return real_copy(d, s, n, kind, st);
}
hipError_t kept_uncached(void** p, size_t n, unsigned flags) {
    const hipError_t e = real_uncached(p, n, flags);
    if (e == hipSuccess) group_calls.areas.push_back(*p);
    return e;
}
hipError_t no_peer(int* can, int, int) {
    *can = 0;
    return hipSuccess;
}

std::vector<BackendPtr> rocm_devices(size_t most) {
    std::vector<BackendPtr> d;
    for (size_t i = 0; i < most; ++i) {
        try {
            d.push_back(make_rocm_backend((int)i));
        } catch (const RocmUnavailable&) {
            break;
        }
    }
    return d;
}
std::vector<Backend*> raw(const std::vector<BackendPtr>& d, size_t W) {
    std::vector<Backend*> r;
    for (size_t m = 0; m < W; ++m) r.push_back(d[m].get());
    return r;
}

// A sum's launches and copies: one launch a member and no copy on the fast path, a copy for each peer on the conservative one.
void sum_calls(Collective& c, size_t W, bool fast) {
    HipFn& f = hip();
    group_calls.launches = group_calls.copies = 0;
    f.hipModuleLaunchKernel = counted_launch;
    f.hipMemcpyAsync = counted_copy;
    c.sum_in_place(1, 97);
    f.hipModuleLaunchKernel = real_launch;
    f.hipMemcpyAsync = real_copy;
    require(group_calls.launches == (int)W && group_calls.copies == (fast ? 0 : (int)(W * (W - 1))),
            fast ? "a fast group sum made other launches or copies than one launch a member" : "a conservative group sum made other launches or copies than a launch a member and a copy a peer");
}

// Two sequences of three sums, alone and then interleaved with and without waits between them, each ending with its bits alone (docs/TESTS.md).
void sums_in_flight(const std::vector<Backend*>& group, Collective& c, size_t width, uint32_t seed) {
    const size_t W = group.size(), kSums = 3;
    struct Seq {
        size_t rows = 0;
        std::vector<float> start, want;
        std::vector<BufferPtr> x;
        std::vector<std::vector<BufferPtr>> staged;   // per sum, per member
    };
    auto make = [&](size_t rows) {
        Seq q;
        q.rows = rows;
        const size_t n = rows * width;
        group_sums::Case k = group_sums::make(W, n, seed++);
        q.start = k.start;
        q.want = k.start;
        for (size_t m = 0; m < W; ++m) q.x.push_back(group[m]->alloc(n * sizeof(float)));
        for (size_t s = 0; s < kSums; ++s) {
            if (s) {
                k = group_sums::make(W, n, seed++);
                k.start = q.want;
            }
            q.want = group_sums::oracle(k, false);
            q.staged.emplace_back();
            for (size_t m = 0; m < W; ++m) {
                q.staged.back().push_back(group[m]->alloc(n * sizeof(float)));
                group[m]->write(*q.staged.back().back(), 0, k.partial[m].data(), n * sizeof(float));
            }
        }
        return q;
    };
    auto reset = [&](Seq& q) {
        for (size_t m = 0; m < W; ++m) group[m]->write(*q.x[m], 0, q.start.data(), q.start.size() * sizeof(float));
    };
    auto step = [&](Seq& q, size_t s) {
        std::vector<Slice> xs;
        for (size_t m = 0; m < W; ++m) {
            const Slice p = c.partial(m);
            group[m]->copy(*p.buffer, 0, *q.staged[s][m], 0, q.rows * width * sizeof(float));
            xs.push_back({q.x[m].get(), 0});
        }
        c.sum_into(xs, q.rows, width);
    };
    auto got = [&](Seq& q, size_t m) {
        group[m]->submit();
        return read(*group[m], *q.x[m], q.start.size());
    };
    Seq a = make(7), b = make(64);
    for (Seq* q : {&a, &b}) {
        reset(*q);
        for (size_t s = 0; s < kSums; ++s) step(*q, s);
        for (size_t m = 0; m < W; ++m) require(same(got(*q, m), q->want), "a sequence of group sums alone is not the oracle's");
    }
    for (const bool waits : {false, true}) {
        reset(a);
        reset(b);
        for (size_t s = 0; s < kSums; ++s) {
            step(a, s);
            if (waits) c.wait(1);
            step(b, s);
            if (waits) c.wait(0);
        }
        for (size_t m = 0; m < W; ++m)
            require(same(got(a, m), a.want) && same(got(b, m), b.want), "two sequences of group sums in flight on one collective differ from each alone");
    }
}

// Backend::busy_ms: a known run of copies holds the device a positive time below the host's wall time around it, and a reader whose take waits behind the sender's 50 ms of copies is held for its copy alone.
void busy_times() {
    std::vector<BackendPtr> d = rocm_devices(2);
    const size_t big = (size_t)64 << 20, n = 4099, bytes = n * sizeof(float);
    const BufferPtr from = d[0]->alloc(big), to = d[0]->alloc(big);
    const Ticket before = d[0]->submit();
    d[0]->wait(before);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 32; ++i) d[0]->copy(*to, 0, *from, 0, big);
    const Ticket ran = d[0]->submit();
    d[0]->wait(ran);
    const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), held = d[0]->busy_ms(before, ran);
    std::printf("backend-rocm: busy %.3f ms of %.3f ms wall for 32 copies\n", held, wall);
    require(held > 0 && held < wall, "a run of copies was not held a positive time below its wall time");
    require(d[0]->busy_ms(ran, before) < 0 && d[0]->busy_ms(ran, ran + 1) < 0, "busy_ms answered for tickets out of order or not yet submitted");
    if (d.size() < 2) {
        std::cout << "backend-rocm: one device, so no busy time across a wait on another\n";
        return;
    }
    const std::unique_ptr<Handoff> h = d[0]->handoff({d[1].get()});
    require(h != nullptr, "a ROCm handoff to a ROCm reader is made");
    if (!h) return;
    const BufferPtr box = d[0]->alloc(bytes, Memory::host_visible), rows = d[0]->alloc(bytes), dst = d[1]->alloc(bytes);
    const Ticket last = d[1]->submit(), idle = d[0]->submit();
    d[1]->wait(last);
    d[0]->wait(idle);
    // Copies queued on the sender until they hold it 50 ms by its own busy time, ahead of the rows the reader waits for.
    int count = 32;
    for (double one = 0; count < 4096; count *= 2) {
        for (int i = 0; i < count; ++i) d[0]->copy(*to, 0, *from, 0, big);
        const Ticket t = d[0]->submit();
        d[0]->wait(t);
        one = d[0]->busy_ms(t - 1, t) / count;
        if (one > 0 && one * count * 2 >= 50) {
            count = std::max(1, (int)std::ceil(50 / one));
            break;
        }
    }
    const Ticket ahead = d[0]->submit();
    d[0]->wait(ahead);
    for (int i = 0; i < count; ++i) d[0]->copy(*to, 0, *from, 0, big);
    h->begin();
    d[0]->copy(*box, 0, *rows, 0, bytes);
    h->sent();
    const auto t1 = std::chrono::steady_clock::now();
    h->take(0, *dst, 0, *box, 0, bytes);
    h->taken(0);
    const Ticket took = d[1]->submit(), sent = d[0]->submit();
    d[1]->wait(took);
    d[0]->wait(sent);
    const double waited = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count(), sender = d[0]->busy_ms(ahead, sent), took_ms = d[1]->busy_ms(last, took);
    std::printf("backend-rocm: a reader held %.3f ms of %.3f ms behind %.3f ms of the sender's copies\n", took_ms, waited, sender);
    require(sender >= 45 && waited >= 0.9 * sender && took_ms >= 0 && took_ms < 10, "a reader's busy time counted its wait for the sender's copies");
}

// Device-side crossings (Backend::handoff): device 0 sends rows through a host-visible buffer to every other device, each reader's queue waiting on the sender's, with every device's backend alive.
// A slow sender proves a reader's copy waits on the device and that take returns before the sender's work ends; a slow reader proves begin waits for its last take before the buffer is written again.
void handoffs() {
    std::vector<BackendPtr> d = rocm_devices(kMaxTensorWidth);
    BackendPtr cpu = make_cpu_backend();
    require(d[0]->handoff({cpu.get()}) == nullptr, "a ROCm handoff to the CPU alone is null");
    if (d.size() < 2) {
        std::cout << "backend-rocm: one device, so no handoff between devices\n";
        return;
    }
    const size_t n = 4099, bytes = n * sizeof(float), big = (size_t)64 << 20;
    std::vector<Backend*> readers;
    for (size_t r = 1; r < d.size(); ++r) readers.push_back(d[r].get());
    readers.push_back(cpu.get());
    const std::unique_ptr<Handoff> h = d[0]->handoff(readers);
    require(h != nullptr, "a ROCm handoff to ROCm readers is made");
    if (!h) return;
    for (size_t r = 0; r + 1 < readers.size(); ++r) require(h->serves(r), "a ROCm handoff serves a ROCm reader");
    require(!h->serves(readers.size() - 1), "a ROCm handoff does not serve the CPU");
    const BufferPtr box = d[0]->alloc(bytes, Memory::host_visible);
    const BufferPtr rows = d[0]->alloc(bytes);
    // Copies of a large buffer, about a millisecond each, queued ahead of a side's part.
    const auto slow = [&](Backend& b, const BufferPtr& from, const BufferPtr& to) {
        for (int i = 0; i < 32; ++i) b.copy(*to, 0, *from, 0, big);
    };
    std::vector<BufferPtr> s_from{d[0]->alloc(big)}, s_to{d[0]->alloc(big)};
    std::vector<BufferPtr> dst, r_from, r_to;
    for (size_t r = 1; r < d.size(); ++r) {
        dst.push_back(d[r]->alloc(bytes));
        r_from.push_back(d[r]->alloc(big));
        r_to.push_back(d[r]->alloc(big));
    }
    const auto values = [&](int round) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) v[i] = float((int)((i * 7 + (size_t)round * 13) % 61) - 30) * 0.25f + (float)round;
        return v;
    };
    for (int round = 1; round <= 3; ++round) {
        const std::vector<float> want = values(round);
        d[0]->write(*rows, 0, want.data(), bytes);
        d[0]->wait(d[0]->submit());
        // Round 1 maps the buffer for every reader, round 2 slows the sender, round 3 every reader.
        const auto t0 = std::chrono::steady_clock::now();
        if (round == 2) slow(*d[0], s_from[0], s_to[0]);
        const auto ts = std::chrono::steady_clock::now();
        h->begin();
        d[0]->copy(*box, 0, *rows, 0, bytes);
        h->sent();
        const Ticket sent = d[0]->submit();
        std::vector<Ticket> took(d.size());
        const auto tt = std::chrono::steady_clock::now();
        for (size_t r = 1; r < d.size(); ++r) {
            if (round == 3) slow(*d[r], r_from[r - 1], r_to[r - 1]);
            h->take(r - 1, *dst[r - 1], 0, *box, 0, bytes);
            h->taken(r - 1);
            took[r] = d[r]->submit();
        }
        if (round == 2) {
            const bool early = !d[0]->done(sent);
            const auto t1 = std::chrono::steady_clock::now();
            d[0]->wait(sent);
            const double rest = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
            std::printf("backend-rocm: handoff round 2, slow %.3f send %.3f take %.3f; the host's calls %.3f ms, the sender's work ran %.3f ms past the readers' takes\n", std::chrono::duration<double, std::milli>(ts - t0).count(), std::chrono::duration<double, std::milli>(tt - ts).count(), std::chrono::duration<double, std::milli>(t1 - tt).count(), std::chrono::duration<double, std::milli>(t1 - t0).count(), rest);
            require(early, "a reader's take returns before the sender's slow work ends");
        }
        if (round == 3) {
            // The next round's rows go in behind the slow readers' takes, or they read them in this round.
            const std::vector<float> next = values(4);
            d[0]->write(*rows, 0, next.data(), bytes);
            h->begin();
            d[0]->copy(*box, 0, *rows, 0, bytes);
            h->sent();
            d[0]->submit();
        }
        for (size_t r = 1; r < d.size(); ++r) {
            d[r]->wait(took[r]);
            std::vector<float> got(n);
            d[r]->read(*dst[r - 1], 0, got.data(), bytes);
            require(same(got, want), "handoff round " + std::to_string(round) + " reader " + std::to_string(r) + " reads the rows sent to it");
        }
        d[0]->sync();
        if (round == 3) {
            // The rows sent behind the slow readers reach them too.
            for (size_t r = 1; r < d.size(); ++r) {
                h->take(r - 1, *dst[r - 1], 0, *box, 0, bytes);
                h->taken(r - 1);
                std::vector<float> got(n);
                d[r]->read(*dst[r - 1], 0, got.data(), bytes);
                require(same(got, values(4)), "handoff reader " + std::to_string(r) + " reads the rows sent behind its slow take");
            }
        }
    }
    {
        // A count or mark whose write HIP refuses: taken and sent throw, and neither the sender's next begin nor any sync waits for the dropped write.
        HipFn& f = hip();
        const auto real_write = f.hipStreamWriteValue64;
        const auto refuse = [](hipStream_t, void*, uint64_t, unsigned) { return hipErrorLaunchFailure; };
        const auto from = std::chrono::steady_clock::now();
        h->begin();
        d[0]->copy(*box, 0, *rows, 0, bytes);
        h->sent();
        d[0]->submit();
        for (size_t r = 1; r < d.size(); ++r) {
            h->take(r - 1, *dst[r - 1], 0, *box, 0, bytes);
            if (r == 1) f.hipStreamWriteValue64 = refuse;
            require(throws([&] { h->taken(r - 1); }) == (r == 1), "a taken throws exactly where HIP refuses its count's write");
            f.hipStreamWriteValue64 = real_write;
            d[r]->submit();
        }
        h->begin();
        d[0]->copy(*box, 0, *rows, 0, bytes);
        f.hipStreamWriteValue64 = refuse;
        require(throws([&] { h->sent(); }), "a sent throws where HIP refuses its mark's write");
        f.hipStreamWriteValue64 = real_write;
        d[0]->submit();
        for (size_t r = 0; r < d.size(); ++r) d[r]->sync();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
        require(s < 2.0, "a handoff whose writes HIP refused waited " + std::to_string(s) + " s, past 2 s");
    }
    {
        // A reader's first copy from a buffer: a second host-visible buffer of the same sender after the first is warm tells a cost per buffer from one per device pair.
        const BufferPtr box2 = d[0]->alloc(bytes, Memory::host_visible);
        const std::unique_ptr<Handoff> h2 = d[0]->handoff(readers);
        double first = 0, again = 0;
        for (int i = 0; i < 2; ++i) {
            h2->begin();
            d[0]->copy(*box2, 0, *rows, 0, bytes);
            h2->sent();
            d[0]->wait(d[0]->submit());
            const auto t0 = std::chrono::steady_clock::now();
            h2->take(0, *dst[0], 0, *box2, 0, bytes);
            h2->taken(0);
            d[1]->submit();
            (i ? again : first) = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            d[1]->sync();
        }
        std::printf("backend-rocm: a reader's take from a new buffer of a warm pair holds its host %.3f ms the first time, %.3f ms the next\n", first, again);
    }
    std::cout << "backend-rocm: a handoff from device 0 reaches " << d.size() - 1 << " readers behind device-side waits, " << d.size() << " ROCm backends alive\n";
}

// The collective over 2 up to kMaxTensorWidth devices where they open, fast and conservative, a sum that gives up and the epochs' wrap (docs/TESTS.md, backend-rocm).
void collective() {
    HipFn& f = hip();
    real_launch = f.hipModuleLaunchKernel;
    real_copy = f.hipMemcpyAsync;
    real_uncached = f.hipExtMallocWithFlags;
    std::vector<BackendPtr> d = rocm_devices(kMaxTensorWidth);
    if (d.size() < 2) {
        std::cout << "backend-rocm: one device, so no collective\n";
        return;
    }
    uint32_t seed = 101;
    auto print = [](bool ok, const std::string& what) { require(ok, what); };
    for (size_t W = 2; W <= d.size(); ++W) {
        const std::vector<Backend*> group = raw(d, W);
        std::unique_ptr<Collective> c = d[0]->join(group, 64, 97);
        require(c != nullptr, "ROCm devices joined no collective");
        if (!c) return;
        sum_calls(*c, W, true);
        group_sums::check(group, *c, 97, {1, 3, 64}, seed, print);
        seed += 100;
        sums_in_flight(group, *c, 97, seed);
        seed += 100;
        // A collective for fewer rows of the same group takes the cached path and its own signals.
        std::unique_ptr<Collective> small = d[0]->join(group, 2, 97);
        group_sums::check(group, *small, 97, {1, 2}, seed, print);
        seed += 100;
        // Each form forced across the devices, about a chain's chunk and at 17 chunks, peer access on from the join above.
        for (const RocmSumForm form : {RocmSumForm::one_shot, RocmSumForm::two_shots, RocmSumForm::chain, RocmSumForm::wrap}) {
            const size_t cap = 17 * 4 * 4096 + 5;
            std::unique_ptr<Collective> forced = rocm_collective(group, cap, 1, form);
            group_sums::check(group, *forced, 1, {1, 4 * W + 1, 4 * 4096 + 1, cap}, seed, print);
            seed += 100;
        }
        // The member order timed for the group is an order of its members, asked again the same, and the group in that order sums in its member order.
        const std::vector<size_t> order = d[0]->member_order(group, 64, 97);
        std::vector<size_t> sorted = order;
        std::sort(sorted.begin(), sorted.end());
        bool permutes = sorted.size() == W;
        for (size_t k = 0; permutes && k < W; ++k) permutes = sorted[k] == k;
        require(permutes && d[0]->member_order(group, 64, 97) == order, "a ROCm group's member order is not one order of its members, kept");
        std::vector<Backend*> ordered;
        for (size_t k : order) ordered.push_back(group[k]);
        std::unique_ptr<Collective> moved = ordered[0]->join(ordered, 64, 97);
        group_sums::check(ordered, *moved, 97, {1, 64}, seed, print);
        seed += 100;
        std::cout << "backend-rocm: a collective of " << W << " devices sums in member order by peer writes, two sums in flight, in each form forced and in its timed member order\n";
    }
    {
        // Devices that cannot reach each other's memory sum by pulls, the same bytes.
        std::vector<BackendPtr> e = rocm_devices(d.size());
        const auto can = f.hipDeviceCanAccessPeer;
        f.hipDeviceCanAccessPeer = no_peer;
        std::unique_ptr<Collective> c = e[0]->join(raw(e, e.size()), 64, 97);
        f.hipDeviceCanAccessPeer = can;
        sum_calls(*c, e.size(), false);
        group_sums::check(raw(e, e.size()), *c, 97, {1, 3, 64}, seed, print);
        seed += 100;
        sums_in_flight(raw(e, e.size()), *c, 97, seed);
        seed += 100;
        std::cout << "backend-rocm: a collective of " << e.size() << " devices without peer access sums by pulls, the same bytes\n";
    }
    {
        // A member whose launch HIP refuses: the sum throws, the host cancels it so its launched peer stops waiting and leaves NaN, which throws once on that backend, and every later sum is conservative and exact.
        std::vector<BackendPtr> e = rocm_devices(2);
        std::unique_ptr<Collective> c = e[0]->join(raw(e, 2), 64, 97);
        const std::vector<float> rows(97, 1.0f);
        for (size_t m = 0; m < 2; ++m) e[m]->write(*c->partial(m).buffer, 0, rows.data(), rows.size() * sizeof(float));
        group_calls.launches = 0;
        group_calls.fail = 2;
        f.hipModuleLaunchKernel = counted_launch;
        const auto from = std::chrono::steady_clock::now();
        require(throws([&] { c->sum_in_place(1, 97); }), "a group sum whose peer's launch failed did not throw");
        f.hipModuleLaunchKernel = real_launch;
        group_calls.fail = 0;
        c->wait(0);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count();
        require(s < 0.5, "a group sum whose peer's launch failed waited " + std::to_string(s) + " s, past 0.5 s");
        require(throws([&] { e[0]->raise(0); }), "a group sum that gave up did not throw on its backend");
        require(!throws([&] { e[0]->raise(0); }), "a group sum that gave up threw on its backend twice");
        bool poisoned = true;
        for (float v : read(*e[0], *c->partial(0).buffer, 97)) poisoned = poisoned && std::isnan(v);
        require(poisoned, "a group sum that gave up left values other than NaN");
        sum_calls(*c, 2, false);
        group_sums::check(raw(e, 2), *c, 97, {1, 64}, seed, print);
        seed += 100;
        std::cout << "backend-rocm: a group sum whose peer's launch failed was cancelled in " << s << " s; later sums conservative and exact\n";
    }
    {
        // A launch that fails while join times the forms: join does not throw, the group is conservative from then on and its sums are exact.
        std::vector<BackendPtr> e = rocm_devices(2);
        group_calls.launches = 0;
        group_calls.fail = 3;   // the probe's two launches come first
        f.hipModuleLaunchKernel = counted_launch;
        std::unique_ptr<Collective> c;
        require(!throws([&] { c = e[0]->join(raw(e, 2), 64, 97); }), "a launch that failed while join timed the forms threw out of join");
        f.hipModuleLaunchKernel = real_launch;
        group_calls.fail = 0;
        require(c != nullptr, "a join whose timing failed made no collective");
        if (c) {
            for (BackendPtr& b : e) throws([&] { b->raise(0); });   // the cancelled sum throws once on the backend whose peer waited
            sum_calls(*c, 2, false);
            group_sums::check(raw(e, 2), *c, 97, {1, 64}, seed, print);
            seed += 100;
        }
        std::cout << "backend-rocm: a launch that failed while join timed the forms left the group conservative and exact\n";
    }
    {
        // A member whose launch comes after a host stall longer than any wait: its peer is alive but late, so the sum stays fast and exact and nothing throws.
        std::vector<BackendPtr> e = rocm_devices(2);
        std::unique_ptr<Collective> c = e[0]->join(raw(e, 2), 64, 97);
        std::vector<float> rows(97), want(97);
        for (size_t i = 0; i < 97; ++i) {
            rows[i] = float((int)(i % 13) - 6) * 0.25f;
            want[i] = rows[i] + rows[i];
        }
        for (size_t m = 0; m < 2; ++m) e[m]->write(*c->partial(m).buffer, 0, rows.data(), rows.size() * sizeof(float));
        group_calls.launches = 0;
        group_calls.late = 2;
        f.hipModuleLaunchKernel = counted_launch;
        c->sum_in_place(1, 97);
        f.hipModuleLaunchKernel = real_launch;
        group_calls.late = 0;
        c->wait(0);
        require(!throws([&] { e[0]->raise(0); }) && !throws([&] { e[1]->raise(0); }), "a group sum whose peer launched late threw");
        for (size_t m = 0; m < 2; ++m)
            require(read(*e[m], *c->partial(m).buffer, 97) == want, "a group sum whose peer launched late did not give the members' sum on member " + std::to_string(m));
        sum_calls(*c, 2, true);
        std::cout << "backend-rocm: a group sum whose peer launched 2.5 s late stays fast and exact\n";
    }
    {
        // Epochs seeded below the 32-bit wrap: twelve sums cross it, skipping 0, each exact.
        std::vector<BackendPtr> e = rocm_devices(2);
        group_calls.areas.clear();
        f.hipExtMallocWithFlags = kept_uncached;
        std::unique_ptr<Collective> c = e[0]->join(raw(e, 2), 64, 97);
        f.hipExtMallocWithFlags = real_uncached;
        require(group_calls.areas.size() >= 2, "a fast collective made no uncached areas");
        if (group_calls.areas.size() < 2) return;
        const std::vector<void*> areas(group_calls.areas.end() - 2, group_calls.areas.end());
        const std::vector<unsigned> seeded(16, 0xfffffffcu);
        for (size_t m = 0; m < 2; ++m) {
            e[m]->sync();
            hip_check(f.hipSetDevice((int)m), "hipSetDevice");
            hip_check(f.hipMemcpyAsync(areas[m], seeded.data(), seeded.size() * sizeof(unsigned), hipMemcpyHostToDevice, nullptr), "hipMemcpyAsync");
            hip_check(f.hipStreamSynchronize(nullptr), "hipStreamSynchronize");
        }
        group_sums::check(raw(e, 2), *c, 97, {1, 1, 1}, seed, print);
        for (size_t m = 0; m < 2; ++m) {
            e[m]->sync();
            unsigned epoch = 0;
            hip_check(f.hipSetDevice((int)m), "hipSetDevice");
            hip_check(f.hipMemcpyAsync(&epoch, areas[m], sizeof(epoch), hipMemcpyDeviceToHost, nullptr), "hipMemcpyAsync");
            hip_check(f.hipStreamSynchronize(nullptr), "hipStreamSynchronize");
            require(epoch == 9, "twelve group sums from epoch 0xfffffffc ended at epoch " + std::to_string(epoch) + ", not 9");
        }
        std::cout << "backend-rocm: twelve group sums across the epochs' wrap, skipping 0, each exact\n";
    }
    {
        // A head group's logits rows: each member copies its slice into one host memory that every member wraps, the first registering it.
        const size_t n = 4096, bytes = 2 * n * sizeof(float);
        void* pages = std::aligned_alloc(4096, bytes);
        std::vector<BufferPtr> views;
        for (size_t m = 0; m < 2; ++m) views.push_back(d[m]->wrap_host(pages, bytes));
        require(views[0] && views[1] && views[0]->host_ptr() == pages && views[1]->host_ptr() == pages, "ROCm members did not wrap one host memory");
        std::vector<std::vector<float>> want(2, std::vector<float>(n));
        for (size_t m = 0; m < 2; ++m) {
            for (size_t i = 0; i < n; ++i) want[m][i] = float(m * n + i) * 0.25f;
            BufferPtr src = d[m]->alloc(n * sizeof(float));
            d[m]->write(*src, 0, want[m].data(), n * sizeof(float));
            d[m]->copy(*views[m], m * n * sizeof(float), *src, 0, n * sizeof(float));
            d[m]->wait(d[m]->submit());
        }
        for (size_t m = 0; m < 2; ++m)
            require(std::memcmp((const float*)pages + m * n, want[m].data(), n * sizeof(float)) == 0, "a member's copy into wrapped host memory is not its rows");
        views.clear();
        for (size_t m = 0; m < 2; ++m) d[m]->sync();
        std::free(pages);
        std::cout << "backend-rocm: two members copy into one wrapped host memory\n";
    }
    auto cpu = make_cpu_backend();
    require(throws([&] { d[0]->join({d[1].get(), d[0].get()}, 1, 1); }), "a ROCm collective joined by a member not first was made");
    require(throws([&] { d[0]->join({d[0].get(), cpu.get()}, 1, 1); }), "a ROCm collective over the CPU was made");
    require(throws([&] { d[0]->join({d[0].get(), d[0].get()}, 1, 1); }), "a ROCm collective over one device twice was made");
    std::string why;
    try { d[0]->join(std::vector<Backend*>(kMaxTensorWidth + 1, d[0].get()), 1, 1); } catch (const std::exception& e) { why = e.what(); }
    require(why.find("more than " + std::to_string(kMaxTensorWidth) + " devices") != std::string::npos, "a ROCm collective wider than the widest group was not refused for its size");
}

} // namespace

// The qwen35 layers' ops every device takes (q35::check_ops).

// Routing on the host in the kernel's order (kernels/f32.hip, llmx_moe_route): lane t of 64 takes experts t, t + 64, ..., the softmax's sum by the 64-pair tree, then k picks, ties to the lowest id.
// Its expf is the host's, so its weights may differ from the device's in the last bits.
void route_host(const float* s, size_t n_expert, size_t k, bool normalize, uint32_t* ids, float* weights) {
    constexpr size_t L = 64;
    float top = -INFINITY;
    for (size_t e = 0; e < n_expert; ++e) top = std::fmax(top, s[e]);
    std::vector<float> p(n_expert), part(L, 0.0f);
    for (size_t e = 0; e < n_expert; ++e) {
        p[e] = std::exp(s[e] - top);
        part[e % L] += p[e];
    }
    for (size_t h = L / 2; h; h /= 2)
        for (size_t t = 0; t < h; ++t) part[t] = part[t] + part[t + h];
    const float sum = part[0];
    if (std::isnan(sum)) {
        for (size_t j = 0; j < k; ++j) ids[j] = Backend::kNoExpert, weights[j] = 0.0f;
        return;
    }
    for (float& v : p) v = v / sum < FLT_MIN ? 0.0f : v / sum;
    float total = 0.0f;
    std::vector<float> chosen(k);
    for (size_t j = 0; j < k; ++j) {
        float m = -2.0f;
        uint32_t id = Backend::kNoExpert;
        for (size_t e = 0; e < n_expert; ++e)
            if (p[e] > m) m = p[e], id = uint32_t(e);
        chosen[j] = id < n_expert ? m : 0.0f;
        total += chosen[j];
        if (id < n_expert) p[id] = -1.0f;
        ids[j] = id;
    }
    for (size_t j = 0; j < k; ++j) weights[j] = ids[j] >= n_expert ? 0.0f : normalize ? chosen[j] / total : chosen[j];
}

// Routing against the CPU: the same picks, ties to the lower id, rows without a candidate giving kNoExpert, and weights within the F32 bound.
void routing(Backend& g, Backend& c) {
    for (size_t n_expert : {16, 128, 1024})
        for (size_t k : {1, 4, 8})
            for (bool normalize : {false, true}) {
                const size_t rows = 37;
                std::vector<float> s = row_classes::values(rows * n_expert, uint32_t(n_expert + k), 4.0f);
                s[1 * n_expert + 3] = s[1 * n_expert + 9] = 30.0f;   // a tie at the top
                s[2 * n_expert + 5] = NAN;
                for (size_t e = 0; e < n_expert; ++e) s[3 * n_expert + e] = -INFINITY;
                s[4 * n_expert] = INFINITY;
                std::vector<float> ids[2], ws[2];
                Backend* bs[2] = {&g, &c};
                for (int b = 0; b < 2; ++b) {
                    auto sb = bs[b]->adopt(s.data(), s.size() * 4), ib = bs[b]->alloc(rows * k * 4), wb = bs[b]->alloc(rows * k * 4);
                    bs[b]->route_experts({sb.get(), 0}, rows, n_expert, k, backend::RouteRule::softmax(normalize), {ib.get(), 0}, {wb.get(), 0});
                    ids[b] = read(*bs[b], *ib, rows * k);
                    ws[b] = read(*bs[b], *wb, rows * k);
                }
                const std::string at = std::to_string(k) + " of " + std::to_string(n_expert) + (normalize ? " normalized" : "");
                require(same(ids[0], ids[1]), "routing " + at + " picks the CPU's experts");
                close(ws[0], ws[1], "routing " + at + " weighs as the CPU does");
                uint32_t none;
                std::memcpy(&none, &ids[0][2 * k], 4);
                require(none == Backend::kNoExpert, "a row with a NaN score routes to no expert");
                // The device's picks are the host's in the kernel's order, ties to the lower id, whatever the device's wave width; the weights hold the CPU's bound, since the host's expf is not the device's.
                std::vector<uint32_t> hid(rows * k);
                std::vector<float> hw(rows * k);
                for (size_t r = 0; r < rows; ++r) route_host(s.data() + r * n_expert, n_expert, k, normalize, hid.data() + r * k, hw.data() + r * k);
                require(std::memcmp(hid.data(), ids[0].data(), rows * k * 4) == 0, "routing " + at + " picks the host's experts in the kernel's order");
            }
}

// Whether a twin class's routed call groups its entries by expert, from each build's break-even (rocm_backend.cpp, TwinKernels::group_from).
bool groups_at(uint32_t type, Dtype dt, size_t entries, size_t n_expert, size_t nin) {
    const bool i8 = dt == Dtype::int8, quarter = quarter_form(type, nin);
    const double from = type == quant::GGML_TYPE_Q8_0 ? (i8 ? 0 : 1.25) : type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K ? (i8 ? 1.25 : 2.5)
                      : type == quant::GGML_TYPE_Q6_K || type == quant::GGML_TYPE_Q5_0 ? 1.25 : type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_MXFP4 ? 0.3125
                      : quarter ? (i8 ? 2.5 : 1.25) : (i8 ? 1.25 : 5.0);
    // The half-block form's calls of at most one entry an expert take the one-column build an entry where it fills its steps.
    const size_t blocks = nin / 32, sub = rocm_device(0).caps.subgroup_size;
    const bool half = type == quant::GGML_TYPE_Q8_0 && i8 && blocks % 2 == 0 && (blocks < sub || blocks >= 2 * sub);
    return entries >= 1 && double(entries) >= from * double(n_expert) && !(half && entries <= n_expert);
}

// The prompt extent from which a routed build takes the routed tile for gate and up or for the down, the backend's `TwinKernels::tile_from`: 32, but gate and up in block-int16 from 128 for the types other than Q8_0, IQ4_NL and MXFP4.
size_t tile_from_at(uint32_t type, Dtype dt, bool down) {
    return down || dt == Dtype::int8 || type == quant::GGML_TYPE_Q8_0 || type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_MXFP4 ? 32 : 128;
}

// Routed products at Flash-Next's expert shapes, every type and class, each entry its expert's one-column product bit for bit (docs/src/backends-rocm.md).
void experts(Backend& g, Backend& c) {
    const uint32_t kinds[] = {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K,
                              quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
    const size_t n_expert = 8, k = 4;
    for (uint32_t type : kinds) {
        const std::string name = quant::storage_type(type)->name;
        for (const auto& sh : std::vector<std::array<size_t, 2>>{{2560, 640}, {640, 2560}}) {
            const size_t nin = sh[0], ff = sh[1];
            if (nin % quant::storage_type(type)->block_size) continue;
            const size_t ebytes = quant::row_bytes(type, nin, ff);
            require(ebytes % 4 == 0, "an expert's matrix ends on a float");
            std::vector<uint8_t> w = type == quant::GGML_TYPE_F32  ? std::vector<uint8_t>()
                                     : type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * ff, uint32_t(nin))
                                                                     : typed_matrix(type, nin, n_expert * ff, uint32_t(type * 7 + nin));
            if (type == quant::GGML_TYPE_F32) {
                const std::vector<float> v = row_classes::values(nin * n_expert * ff, uint32_t(nin + 3), 0.05f);
                w.assign((const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 4);
            }
            auto wa = g.adopt(w.data(), w.size()), wb = g.adopt(w.data(), w.size());
            // The row counts take the one-column build and, past each build's break-even, a grouped one in the twin classes (groups_at).
            for (size_t rows : {1, 3, 4, 5, 33}) {
                const std::vector<float> x = row_classes::values(rows * nin, uint32_t(rows + nin));
                std::vector<uint32_t> idv(rows * k);
                for (size_t e = 0; e < idv.size(); ++e) idv[e] = uint32_t((e * 5 + rows) % n_expert);
                idv[k + 1] = Backend::kNoExpert;
                const std::vector<float> wts = row_classes::values(rows * k, uint32_t(rows + 11), 0.5f);
                auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, k, n_expert};
                std::vector<RowRun> decode(rows);
                for (size_t r = 0; r < rows; ++r) decode[r] = {r + 1, 1};
                const RowRun prompt[1] = {{rows, rows}};
                for (Dtype dt : {Dtype::f32, Dtype::f16, Dtype::int8}) {
                    const bool twins = type != quant::GGML_TYPE_F32 && type != quant::GGML_TYPE_BF16;
                    if (!twins && dt != Dtype::f32) continue;
                    for (const RowRuns runs : {RowRuns{decode.data(), rows}, RowRuns{prompt, 1}}) {
                        const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(ff) + ", " + std::to_string(rows) + (runs.n == 1 && rows > 1 ? " prompt rows, " : " rows, ") + dtype_name(dt);
                        const std::vector<float> fill(rows * k * ff, 7.0f);
                        auto ga = g.adopt(fill.data(), fill.size() * 4), gb = g.adopt(fill.data(), fill.size() * 4);
                        rocm_take_launches(g);
                        g.matmul_experts({Projection{{type, {wa.get(), 0}}, {ga.get(), 0}, ff}, Projection{{type, {wb.get(), 0}}, {gb.get(), 0}, ff}}, {xb.get(), 0}, nin, rows, routing, runs, dt);
                        // The kernels each class's routed call runs, and nothing else (docs/src/backends-rocm.md).
                        const bool tiled = twins && dt != Dtype::f32 && runs.n == 1 && rows >= tile_from_at(type, dt, false);
                        const bool grouped = twins && dt != Dtype::f32 && groups_at(type, dt, rows * k, n_expert, nin);
                        std::string ran;
                        bool routed_only = true, routed_ran = false, lane_form = true;
                        for (const auto& [kn, v] : rocm_take_launches(g)) {
                            ran += (ran.empty() ? "" : "+") + kn;
                            const bool exact = kn.rfind("llmx_routed_", 0) == 0, twin = kn.find("_routed") != std::string::npos && !exact;
                            const bool group_build = twin && kn.find("_g", kn.find("_routed")) != std::string::npos;
                            const bool tile = kn.find("_routed_tile") != std::string::npos || kn.find("_routed_narrow") != std::string::npos || kn.find("_routed_stream") != std::string::npos ||
                                              kn.find("_routed_thin") != std::string::npos || kn.find("_routed_mid") != std::string::npos;
                            if (twin && !tile) lane_form = lane_form && (kn.find("_quarter_") != std::string::npos) == quarter_form(type, nin);
                            routed_ran = routed_ran || (tiled ? tile : twins && dt != Dtype::f32 ? twin && !tile && group_build == grouped : exact);
                            routed_only = routed_only && (tiled ? tile || kn.rfind("llmx_twin", 0) == 0 || kn == "llmx_moe_group"
                                                                : exact || (twin && !tile && group_build == grouped) || kn.rfind("llmx_twin", 0) == 0 || (grouped && kn == "llmx_moe_group"));
                        }
                        require(routed_ran && routed_only, at + " runs its routed build alone, ran " + ran);
                        require(lane_form, at + " takes its row's lane form, ran " + ran);
                        const std::vector<float> got[2] = {read(g, *ga, rows * k * ff), read(g, *gb, rows * k * ff)};
                        auto one = g.alloc(ff * 4);
                        bool alone = true, unwritten = true;
                        for (size_t e = 0; e < rows * k; ++e) {
                            if (idv[e] == Backend::kNoExpert) {
                                unwritten = std::all_of(got[0].begin() + e * ff, got[0].begin() + (e + 1) * ff, [](float v) { return v == 7.0f; });
                                continue;
                            }
                            if (tiled) {
                                alone = alone && !std::memcmp(got[0].data() + e * ff, got[1].data() + e * ff, ff * 4);
                                continue;
                            }
                            g.matmul({type, {wa.get(), idv[e] * ebytes / 4}}, {xb.get(), e / k * nin}, {one.get(), 0}, nin, ff, 1, {}, dt);
                            const std::vector<float> o = read(g, *one, ff);
                            alone = alone && !std::memcmp(o.data(), got[0].data() + e * ff, ff * 4) && !std::memcmp(o.data(), got[1].data() + e * ff, ff * 4);
                        }
                        require(alone, at + (tiled ? ": each entry has the same bits in both projections of one weight" : ": each entry is its expert's one-column product, bit for bit"));
                        require(unwritten, at + ": an entry without an expert is not written");
                        if (dt == Dtype::f32 && rows == 5 && runs.n > 1) {
                            // Every expert's decoded rows by the CPU's embedding, and each entry against the double sum.
                            auto cw = c.adopt(w.data(), w.size()), ce = c.alloc(n_expert * ff * nin * 4);
                            std::vector<uint32_t> every(n_expert * ff);
                            for (size_t i = 0; i < every.size(); ++i) every[i] = uint32_t(i);
                            c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, n_expert * ff, every.data(), every.size());
                            const std::vector<float> wd = read(c, *ce, n_expert * ff * nin);
                            for (size_t e = 0; e < rows * k; ++e) {
                                if (idv[e] == Backend::kNoExpert) continue;
                                const std::vector<float> we(wd.begin() + idv[e] * ff * nin, wd.begin() + (idv[e] + 1) * ff * nin);
                                const std::vector<float> xe(x.begin() + e / k * nin, x.begin() + (e / k + 1) * nin);
                                exact_close(std::vector<float>(got[0].begin() + e * ff, got[0].begin() + (e + 1) * ff), double_sums(we, xe, nin, ff), at + ": entry " + std::to_string(e) + " against the double sum");
                            }
                        }
                    }
                }
            }
        }
    }
}

// The grouping's own branches, 300 experts 8 a token over 80 rows and skewed ids, every entry its expert's one-column product bit for bit (docs/src/backends-rocm.md).
// Every twin type takes its grouped builds and its one-column build here, each entry against the same one-column product, so the builds give the same bits whichever the break-evens choose.
void experts_grouped(Backend& g) {
    struct Case { size_t n_expert, k, rows; bool skew; };
    // Past each build's break-even the calls below take the 4-, 8-, 4- and 2-column builds, below it the one-column build, and 4 rows of 300 the one-column build below every break-even.
    const Case cases[] = {{300, 8, 80, false}, {16, 8, 33, true}, {16, 8, 5, false}, {64, 2, 64, true}, {300, 8, 4, false}};
    const size_t ff = 256;
    for (const auto& [type, nin] : std::vector<std::pair<uint32_t, size_t>>{{quant::GGML_TYPE_Q8_0, 2080}, {quant::GGML_TYPE_Q8_0, 2560}, {quant::GGML_TYPE_Q8_0, 1280}, {quant::GGML_TYPE_Q8_0, 4096}, {quant::GGML_TYPE_Q4_K, 2560}, {quant::GGML_TYPE_Q5_K, 2560},
                                                                             {quant::GGML_TYPE_Q6_K, 2560}, {quant::GGML_TYPE_Q5_0, 2560}, {quant::GGML_TYPE_Q5_1, 2560}, {quant::GGML_TYPE_IQ4_NL, 2560}, {quant::GGML_TYPE_MXFP4, 2560},
                                                                             {quant::GGML_TYPE_Q5_0, 640}, {quant::GGML_TYPE_Q5_1, 640}}) {
        const std::string name = quant::storage_type(type)->name;
        const size_t ebytes = quant::row_bytes(type, nin, ff);
        const std::vector<uint8_t> w = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, 300 * ff, uint32_t(nin)) : typed_matrix(type, nin, 300 * ff, uint32_t(type * 7 + nin));
        auto wb = g.adopt(w.data(), w.size());
        for (const Case& cs : cases) {
            const size_t entries = cs.rows * cs.k;
            const std::vector<float> x = row_classes::values(cs.rows * nin, uint32_t(cs.rows + nin));
            std::vector<uint32_t> idv(entries);
            for (size_t e = 0; e < entries; ++e) idv[e] = uint32_t((e * 7 + e / cs.k) % cs.n_expert);
            if (cs.skew) {
                // Expert 0 one entry, expert 1 every other entry (several groups, the last partial), the top quarter unused.
                const uint32_t spread = uint32_t(cs.n_expert * 3 / 4 - 2);
                for (size_t e = 0; e < entries; ++e) idv[e] = e == 4 ? 0u : e % 2 ? 1u : uint32_t(2 + (e * 5) % spread);
                idv[entries - 2] = Backend::kNoExpert;
            }
            const std::vector<float> wts = row_classes::values(entries, uint32_t(cs.rows + 11), 0.5f);
            auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
            const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, cs.k, cs.n_expert};
            std::vector<RowRun> decode(cs.rows);
            for (size_t r = 0; r < cs.rows; ++r) decode[r] = {r + 1, 1};
            const RowRun prompt[1] = {{cs.rows, cs.rows}};
            for (Dtype dt : {Dtype::f16, Dtype::int8})
                for (const RowRuns runs : {RowRuns{decode.data(), cs.rows}, RowRuns{prompt, 1}}) {
                    if (runs.n == 1 && cs.rows >= tile_from_at(type, dt, false)) continue;   // the routed tile's (experts_tile)
                    const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(ff) + ", " + std::to_string(cs.n_expert) + " experts, " + std::to_string(cs.rows) + (runs.n == 1 ? " prompt rows, " : " rows, ") + dtype_name(dt);
                    const std::vector<float> fill(entries * ff, 7.0f);
                    auto gb = g.adopt(fill.data(), fill.size() * 4);
                    rocm_take_launches(g);
                    g.matmul_experts({Projection{{type, {wb.get(), 0}}, {gb.get(), 0}, ff}}, {xb.get(), 0}, nin, cs.rows, routing, runs, dt);
                    const bool grouped = groups_at(type, dt, entries, cs.n_expert, nin);
                    const size_t touched = std::min(cs.n_expert, entries), average = (entries + touched - 1) / touched;
                    const std::string build = !grouped ? "" : average <= 2 ? "_g2" : average <= 4 ? "_g4" : "_g8";
                    // The half-block one-column build takes one step a load on a row under two spans of blocks, as the dense one-column build does.
                    const bool half = type == quant::GGML_TYPE_Q8_0 && dt == Dtype::int8 && nin / 32 % 2 == 0;
                    const std::string one_build = !half ? "" : nin / 32 < rocm_device(0).caps.subgroup_size ? "llmx_q8_0_routed_half8s1" : "llmx_q8_0_routed_half8";
                    std::string ran;
                    bool built = false, grouping = false;
                    for (const auto& [kn, v] : rocm_take_launches(g)) {
                        ran += (ran.empty() ? "" : "+") + kn;
                        const size_t r = kn.find("_routed");
                        built = built || (r != std::string::npos && (grouped ? kn.compare(kn.size() - 3, 3, build) == 0 : kn.find("_g", r) == std::string::npos && (one_build.empty() || kn == one_build)));
                        grouping = grouping || kn == "llmx_moe_group";
                    }
                    require(built && grouping == grouped, at + " takes its " + (grouped ? build.substr(1) + " grouped" : "one-column") + " build, ran " + ran);
                    const std::vector<float> got = read(g, *gb, entries * ff);
                    auto one = g.alloc(ff * 4);
                    bool alone = true, unwritten = true;
                    for (size_t e = 0; e < entries; ++e) {
                        if (idv[e] == Backend::kNoExpert) {
                            unwritten = unwritten && std::all_of(got.begin() + e * ff, got.begin() + (e + 1) * ff, [](float v) { return v == 7.0f; });
                            continue;
                        }
                        g.matmul({type, {wb.get(), idv[e] * ebytes / 4}}, {xb.get(), e / cs.k * nin}, {one.get(), 0}, nin, ff, 1, {}, dt);
                        alone = alone && !std::memcmp(read(g, *one, ff).data(), got.data() + e * ff, ff * 4);
                    }
                    require(alone, at + ": each entry is its expert's one-column product, bit for bit");
                    require(unwritten, at + ": an entry without an expert is not written");
                }
        }
    }
}

// Routed products at Flash-Next's expert shapes for every twin type: the one-column build against the build the call takes, medians of five interleaved.
// Each build's break-even, the figure TwinKernels::group_from holds, follows (docs/src/backends-rocm.md).
void routed_rates(Backend& g) {
    const size_t n_expert = 512, past = 1025, k = 10, kCounts = 8;
    const size_t counts[kCounts] = {1, 2, 4, 8, 16, 32, 64, 128};
    std::cout << "backend-rocm: routed rates; type, dtype, shape x projections, generated rows: us a call one-column, taken, ratio, build taken\n";
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4}) {
        const std::string name = quant::storage_type(type)->name;
        // Sums and the grouped builds' reach per lane form (quarter_form) and class, since each form has its own break-evens.
        double one_sum[2][2][kCounts] = {}, taken_sum[2][2][kCounts] = {};
        bool grouped_at[2][2][kCounts], measured[2] = {};
        for (auto& f : grouped_at)
            for (auto& row : f) std::fill(row, row + kCounts, true);
        for (const auto& sh : std::vector<std::array<size_t, 3>>{{2560, 640, 2}, {640, 2560, 1}}) {
            const size_t nin = sh[0], ff = sh[1], nproj = sh[2];
            if (nin % quant::storage_type(type)->block_size) continue;
            const int lf = quarter_form(type, nin);
            measured[lf] = true;
            const std::vector<uint8_t> w = typed_matrix(type, nin, n_expert * ff, 7);
            const size_t wide = quant::row_bytes(type, nin, past * ff);
            auto wsrc = g.adopt(w.data(), w.size()), wa = g.alloc(wide), wb = g.alloc(wide);
            g.copy(*wa, 0, *wsrc, 0, w.size());
            g.copy(*wb, 0, *wsrc, 0, w.size());
            for (size_t ci = 0; ci < kCounts; ++ci) {
                const size_t rows = counts[ci], entries = rows * k;
                const std::vector<float> x = row_classes::values(rows * nin, 900);
                std::vector<uint32_t> idv(entries);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t j = 0; j < k; ++j) idv[r * k + j] = uint32_t((r * 131 + j * 47 + (r * j) % 7) % n_expert);
                const std::vector<float> wts(entries, 0.1f);
                auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                auto ya = g.alloc(entries * ff * 4), yb = g.alloc(entries * ff * 4);
                const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, k, n_expert}, one_column{{ib.get(), 0}, {tb.get(), 0}, k, past};
                std::vector<RowRun> decode(rows);
                for (size_t r = 0; r < rows; ++r) decode[r] = {r + 1, 1};
                const RowRuns runs{decode.data(), rows};
                for (int di = 0; di < 2; ++di) {
                    const Dtype dt = di ? Dtype::int8 : Dtype::f16;
                    const Projection pa{{type, {wa.get(), 0}}, {ya.get(), 0}, ff}, pb{{type, {wb.get(), 0}}, {yb.get(), 0}, ff};
                    const auto time = [&](const Backend::Routing& rt) {
                        const auto call = [&] {
                            if (nproj == 2) g.matmul_experts({pa, pb}, {xb.get(), 0}, nin, rows, rt, runs, dt);
                            else g.matmul_experts({pa}, {xb.get(), 0}, nin, rows, rt, runs, dt);
                        };
                        for (int i = 0; i < 2; ++i) call();
                        g.sync();
                        const int iters = 50;
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int i = 0; i < iters; ++i) call();
                        g.sync();
                        return 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
                    };
                    std::vector<double> one, taken;
                    std::string build;
                    for (int rep = 0; rep < 5; ++rep) {
                        one.push_back(time(one_column));
                        rocm_take_launches(g);
                        taken.push_back(time(routing));
                        for (const auto& [kn, v] : rocm_take_launches(g))
                            if (kn.find("_routed") != std::string::npos) build = kn;
                    }
                    std::sort(one.begin(), one.end());
                    std::sort(taken.begin(), taken.end());
                    const bool grouped = build.find("_g", build.find("_routed")) != std::string::npos;
                    grouped_at[lf][di][ci] = grouped_at[lf][di][ci] && grouped;
                    one_sum[lf][di][ci] += one[2];
                    taken_sum[lf][di][ci] += taken[2];
                    std::printf("backend-rocm: routed rate %s, %s, %zux%zu x%zu, %zu: %.1f us, %.1f us, %.3f, %s\n", name.c_str(), dtype_name(dt), nin, ff, nproj, rows, one[2], taken[2], taken[2] / one[2], build.c_str());
                }
            }
        }
        for (int lf = 0; lf < 2; ++lf)
            for (int di = 0; di < 2; ++di) {
                if (!measured[lf]) continue;
                const char* dn = dtype_name(di ? Dtype::int8 : Dtype::f16);
                const std::string at = name + (lf ? " quarter form" : "") + ", " + dn;
                size_t from = kCounts;
                while (from > 0 && grouped_at[lf][di][from - 1] && taken_sum[lf][di][from - 1] <= 1.01 * one_sum[lf][di][from - 1]) --from;
                if (from == kCounts) std::printf("backend-rocm: routed break-even %s: the grouped build wins at no count measured\n", at.c_str());
                else std::printf("backend-rocm: routed break-even %s: grouped wins or is level from %.4g entries an expert (%zu rows)%s\n", at.c_str(), double(counts[from] * k) / double(n_expert), counts[from],
                                 from > 0 && !grouped_at[lf][di][from - 1] ? "; below it the table takes the one-column build, unmeasured here" : "");
            }
    }
    std::fflush(stdout);
}

// The routed tile's group sizes over rows [first, first + count) of `ids` (llmx_moe_group), each expert's entries in groups of 64, the last holding the rest; per_expert takes each expert's entries.
std::vector<size_t> tile_groups(const std::vector<uint32_t>& ids, size_t first, size_t count, size_t k, size_t n_expert, std::vector<size_t>& per_expert) {
    std::vector<size_t> n(n_expert, 0), sizes;
    for (size_t e = first * k; e < (first + count) * k; ++e)
        if (ids[e] < n_expert) ++n[ids[e]];
    for (size_t c : n)
        for (size_t at = 0; at < c; at += 64) sizes.push_back(std::min<size_t>(64, c - at));
    per_expert = n;
    return sizes;
}

// experts_tile's ids for half extent H over 2H rows: experts 11 to 15 hold 1, 7, 15, 16 and 17 entries in the half call and one group of 17 to 33 each in the whole call.
// Expert 0's five first-half entries make the whole call launch a narrow group too, and two entries have no expert.
std::vector<uint32_t> tile_ids(size_t H, size_t k) {
    std::vector<uint32_t> v(2 * H * k);
    for (size_t e = 0; e < v.size(); ++e) v[e] = e % 2 ? 1u : uint32_t(2 + (e * 5 + e / k) % 9);
    const size_t sizes[5] = {1, 7, 15, 16, 17};
    size_t at = H * k;
    for (size_t s = 0; s < 5; ++s)
        for (size_t j = 0; j < sizes[s]; ++j, at += 2) v[at] = uint32_t(11 + s);
    require(at <= 2 * H * k, "experts_tile's second half holds the sparse experts' entries");
    at = 0;
    for (size_t s = 0; s < 5; ++s)
        for (size_t j = 0; j < 16; ++j, at += 2) v[at] = uint32_t(11 + s);
    require(at <= H * k, "experts_tile's first half holds the sparse experts' entries");
    for (size_t e : {1, 5, 13, 17, 21}) v[e] = 0;
    v[9] = v[(H + 5) * k + 1] = Backend::kNoExpert;
    return v;
}

// Whether the calls over rows [0, 2H) and [H, 2H) each hold groups of both builds (narrow up to 16 entries), and the half call tile_ids' five sizes as wide groups of the whole call.
bool tile_groups_cover(const std::vector<uint32_t>& ids, size_t H, size_t k, size_t n_expert) {
    std::vector<size_t> whole_n, half_n;
    const auto whole = tile_groups(ids, 0, 2 * H, k, n_expert, whole_n), half = tile_groups(ids, H, H, k, n_expert, half_n);
    const auto both = [](const std::vector<size_t>& g) {
        return std::any_of(g.begin(), g.end(), [](size_t n) { return n <= 16; }) && std::any_of(g.begin(), g.end(), [](size_t n) { return n > 16; });
    };
    bool sizes = true;
    for (size_t want : {1, 7, 15, 16, 17}) {
        bool found = false;
        for (size_t x = 0; x < n_expert; ++x) found = found || (half_n[x] == want && whole_n[x] > 16 && whole_n[x] <= 64);
        sizes = sizes && found;
    }
    return both(whole) && both(half) && sizes;
}

// The routed prompt tile from its build's tile extent T for the call's role, every type, gate and up and the down, held to the same bits at 2T, T and beside generated rows (docs/src/backends-rocm.md).
// T - 1 rows take the row builds, and the two extents are different row classes.
void experts_tile(Backend& g, Backend& c) {
    const size_t n_expert = 16, k = 4, ff[2] = {256, 200}, most = 2 * 128;   // twice the widest tile extent of tile_from_at
    const std::vector<std::pair<uint32_t, size_t>> kinds = {{quant::GGML_TYPE_Q8_0, 2560}, {quant::GGML_TYPE_Q8_0, 2080}, {quant::GGML_TYPE_Q4_K, 2560}, {quant::GGML_TYPE_Q5_K, 2560},
                                                            {quant::GGML_TYPE_Q6_K, 2560}, {quant::GGML_TYPE_Q5_0, 2560}, {quant::GGML_TYPE_Q5_1, 2560}, {quant::GGML_TYPE_IQ4_NL, 2560}, {quant::GGML_TYPE_MXFP4, 2560}};
    for (const auto& [type, nin] : kinds) {
        const std::string name = quant::storage_type(type)->name, tname = std::string(name == "Q8_0" ? "q8_0" : name == "Q4_K" ? "q4_k" : name == "Q5_K" ? "q5_k" : name == "Q6_K" ? "q6_k" : name == "Q5_0" ? "q5_0" : name == "Q5_1" ? "q5_1" : name == "MXFP4" ? "mxfp4" : "iq4_nl");
        const size_t nblk = nin / 32;
        std::vector<uint8_t> w[2];
        BufferPtr wb[2];
        std::vector<float> wd[2];
        for (int p = 0; p < 2; ++p) {
            w[p] = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * ff[p], uint32_t(nin + p)) : typed_matrix(type, nin, n_expert * ff[p], uint32_t(type * 13 + nin + p));
            wb[p] = g.adopt(w[p].data(), w[p].size());
            auto cw = c.adopt(w[p].data(), w[p].size()), ce = c.alloc(n_expert * ff[p] * nin * 4);
            std::vector<uint32_t> every(n_expert * ff[p]);
            for (size_t i = 0; i < every.size(); ++i) every[i] = uint32_t(i);
            c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, n_expert * ff[p], every.data(), every.size());
            wd[p] = read(c, *ce, n_expert * ff[p] * nin);
        }
        const std::vector<float> x = row_classes::values(most * nin, uint32_t(nin + 41));
        const std::vector<float> wts(most * k, 0.25f);
        auto xb = g.adopt(x.data(), x.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
        for (Dtype dt : {Dtype::f16, Dtype::int8}) {
            const std::string at = name + " " + std::to_string(nin) + ", " + dtype_name(dt);
            const std::string tile = "llmx_" + tname + "_routed_tile" + (dt == Dtype::int8 ? "8" : "16"), narrow = "llmx_" + tname + "_routed_narrow" + (dt == Dtype::int8 ? "8" : "16");
            for (bool down : {false, true}) {
                const size_t from = tile_from_at(type, dt, down);
                require(g.row_class(from - 1) != g.row_class(from), at + (down ? ", down" : ", gate and up") + ": extents " + std::to_string(from - 1) + " and " + std::to_string(from) + " are different row classes");
            }
            // The grouping and the routed tile's two builds launched, no row build; which build a group ran in is witnessed by the bits across extents.
            const auto tiled_alone = [&](const std::string& ran) {
                return ran.find(tile) != std::string::npos && ran.find(narrow) != std::string::npos && ran.find("llmx_moe_group") != std::string::npos && ran.find("_rows") == std::string::npos && ran.find("_routed1") == std::string::npos &&
                       ran.find("_routed8") == std::string::npos && ran.find("_routed_half") == std::string::npos;
            };
            // Gate and up.
            {
                const size_t from = tile_from_at(type, dt, false), half_rows = std::max<size_t>(from, 40), rows = 2 * half_rows;
                require(rows <= most, "experts_tile holds the rows of every build's tile extent");
                const std::vector<uint32_t> idv = tile_ids(half_rows, k);
                require(tile_groups_cover(idv, half_rows, k, n_expert), at + ": the calls at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " hold groups of both builds, of 1, 7, 15, 16 and 17 entries wide at twice the extent");
                auto ib = g.adopt(idv.data(), idv.size() * 4);
                // A call over rows [first, first + count) as one prompt of that extent, or with `decode` generated rows ahead of the prompt, into outputs filled with 7.
                const auto call = [&](size_t first, size_t count, size_t decode, std::string& ran) {
                    std::vector<RowRun> runs;
                    for (size_t r = 0; r < decode; ++r) runs.push_back({r + 1, 1});
                    if (count > decode) runs.push_back({count, count - decode});
                    const std::vector<float> fill(count * k * 256, 7.0f);
                    auto ga = g.adopt(fill.data(), count * k * ff[0] * 4), gu = g.adopt(fill.data(), count * k * ff[1] * 4);
                    const Backend::Routing routing{{ib.get(), first * k}, {tb.get(), first * k}, k, n_expert};
                    rocm_take_launches(g);
                    g.matmul_experts({Projection{{type, {wb[0].get(), 0}}, {ga.get(), 0}, ff[0]}, Projection{{type, {wb[1].get(), 0}}, {gu.get(), 0}, ff[1]}}, {xb.get(), first * nin}, nin, count, routing,
                                     RowRuns{runs.data(), runs.size()}, dt);
                    ran.clear();
                    for (const auto& [kn, v] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + kn;
                    return std::array<std::vector<float>, 2>{read(g, *ga, count * k * ff[0]), read(g, *gu, count * k * ff[1])};
                };
                std::string ran;
                const auto whole = call(0, rows, 0, ran);
                require(tiled_alone(ran), at + ": a " + std::to_string(rows) + "-row prompt takes the grouping, " + tile + " and " + narrow + " alone, ran " + ran);
                call(0, from - 1, 0, ran);
                require(ran.find("routed_tile") == std::string::npos, at + ": a " + std::to_string(from - 1) + "-row prompt takes the row builds, ran " + ran);
                call(0, from, 0, ran);
                require(ran.find(tile) != std::string::npos, at + ": a " + std::to_string(from) + "-row prompt takes " + tile + ", ran " + ran);
                const auto half = call(half_rows, half_rows, 0, ran), mixed = call(0, rows, 3, ran);
                bool same_bits = true, unwritten = true, decode_alone = true, near = true;
                auto one = g.alloc(256 * 4);
                for (int p = 0; p < 2; ++p)
                    for (size_t e = 0; e < rows * k; ++e) {
                        const float* got = whole[p].data() + e * ff[p];
                        if (idv[e] == Backend::kNoExpert) {
                            unwritten = unwritten && std::all_of(got, got + ff[p], [](float v) { return v == 7.0f; }) &&
                                        std::all_of(mixed[p].begin() + e * ff[p], mixed[p].begin() + (e + 1) * ff[p], [](float v) { return v == 7.0f; });
                            continue;
                        }
                        if (e >= half_rows * k) same_bits = same_bits && !std::memcmp(got, half[p].data() + (e - half_rows * k) * ff[p], ff[p] * 4);
                        if (e >= 3 * k) same_bits = same_bits && !std::memcmp(got, mixed[p].data() + e * ff[p], ff[p] * 4);
                        if (e >= 3 * k && e % 7) continue;
                        g.matmul({type, {wb[p].get(), idv[e] * quant::row_bytes(type, nin, ff[p]) / 4}}, {xb.get(), e / k * nin}, {one.get(), 0}, nin, ff[p], 1, {}, dt);
                        const std::vector<float> o = read(g, *one, ff[p]);
                        if (e < 3 * k) decode_alone = decode_alone && !std::memcmp(mixed[p].data() + e * ff[p], o.data(), ff[p] * 4);
                        for (size_t j = 0; j < ff[p]; ++j) {
                            double mag = 0.0;
                            const float* wr = wd[p].data() + (idv[e] * ff[p] + j) * nin;
                            for (size_t i = 0; i < nin; ++i) mag += std::fabs(double(wr[i]) * x[e / k * nin + i]);
                            const double bound = double(nblk + 8) * double(FLT_EPSILON) * mag;
                            if (near && std::fabs(double(got[j]) - o[j]) > bound) std::fprintf(stderr, "  entry %zu output %zu: %.9g against %.9g, bound %.3g\n", e, j, got[j], o[j], bound);
                            near = near && std::fabs(double(got[j]) - o[j]) <= bound;
                        }
                    }
                require(same_bits, at + ": each prompt entry has the same bits at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " and beside generated rows");
                require(decode_alone, at + ": generated rows beside a prompt give their one-column bits");
                require(unwritten, at + ": an entry without an expert is not written");
                require(near, at + ": each sampled tile entry is within (blocks + 8) F32 epsilons of the sum of |w*x| of its one-column product");
            }
            // The down, the gate's matrix over a row an entry.
            {
                const size_t from = tile_from_at(type, dt, true), half_rows = std::max<size_t>(from, 40), rows = 2 * half_rows, nout = ff[0];
                require(rows <= most, "experts_tile holds the rows of every build's down extent");
                const std::vector<uint32_t> idv = tile_ids(half_rows, k);
                require(tile_groups_cover(idv, half_rows, k, n_expert), at + ", down: the calls at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " hold groups of both builds, of 1, 7, 15, 16 and 17 entries wide at twice the extent");
                auto ib = g.adopt(idv.data(), idv.size() * 4);
                const std::vector<float> xd = row_classes::values(rows * k * nin, uint32_t(nin + 43));
                auto xdb = g.adopt(xd.data(), xd.size() * 4);
                // A down over rows [first, first + count) onto zeros, as one prompt of that extent or with `decode` generated rows ahead of it.
                const auto call = [&](size_t first, size_t count, size_t decode, std::string& ran) {
                    std::vector<RowRun> runs;
                    for (size_t r = 0; r < decode; ++r) runs.push_back({r + 1, 1});
                    if (count > decode) runs.push_back({count, count - decode});
                    const std::vector<float> zeros(count * nout, 0.0f);
                    auto yb = g.adopt(zeros.data(), zeros.size() * 4);
                    const Backend::Routing routing{{ib.get(), first * k}, {tb.get(), first * k}, k, n_expert};
                    rocm_take_launches(g);
                    g.matmul_experts_add({type, {wb[0].get(), 0}}, {xdb.get(), first * k * nin}, {yb.get(), 0}, nin, nout, count, routing, RowRuns{runs.data(), runs.size()}, dt);
                    ran.clear();
                    for (const auto& [kn, v] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + kn;
                    return read(g, *yb, count * nout);
                };
                std::string ran;
                const auto whole = call(0, rows, 0, ran);
                require(tiled_alone(ran), at + ", down: a " + std::to_string(rows) + "-row prompt takes the grouping, " + tile + " and " + narrow + " alone, ran " + ran);
                call(0, from - 1, 0, ran);
                require(ran.find("routed_tile") == std::string::npos, at + ", down: a " + std::to_string(from - 1) + "-row prompt takes the row builds, ran " + ran);
                call(0, from, 0, ran);
                require(ran.find(tile) != std::string::npos, at + ", down: a " + std::to_string(from) + "-row prompt takes " + tile + ", ran " + ran);
                const auto half = call(half_rows, half_rows, 0, ran), mixed = call(0, rows, 3, ran), by_rows = call(0, rows, rows, ran);
                bool same_bits = true, decode_rows = true, near = true;
                for (size_t r = 0; r < rows; ++r) {
                    const float* got = whole.data() + r * nout;
                    if (r >= half_rows) same_bits = same_bits && !std::memcmp(got, half.data() + (r - half_rows) * nout, nout * 4);
                    if (r >= 3) same_bits = same_bits && !std::memcmp(got, mixed.data() + r * nout, nout * 4);
                    else decode_rows = decode_rows && !std::memcmp(mixed.data() + r * nout, by_rows.data() + r * nout, nout * 4);
                    if (r >= 3 && r % 7) continue;
                    for (size_t j = 0; j < nout; ++j) {
                        double mag = 0.0;
                        for (size_t s = 0; s < k; ++s) {
                            const uint32_t id = idv[r * k + s];
                            if (id == Backend::kNoExpert) continue;
                            const float* wr = wd[0].data() + (id * nout + j) * nin;
                            for (size_t i = 0; i < nin; ++i) mag += std::fabs(double(wts[r * k + s]) * double(wr[i]) * xd[(r * k + s) * nin + i]);
                        }
                        const float o = by_rows[r * nout + j];
                        const double bound = double(nblk + 8 + k) * double(FLT_EPSILON) * mag;
                        if (near && std::fabs(double(got[j]) - o) > bound) std::fprintf(stderr, "  down row %zu output %zu: %.9g against %.9g, bound %.3g\n", r, j, got[j], o, bound);
                        near = near && std::fabs(double(got[j]) - o) <= bound;
                    }
                }
                require(same_bits, at + ", down: each prompt row has the same bits at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " and beside generated rows");
                require(decode_rows, at + ", down: generated rows beside a prompt give the row builds' bits");
                require(near, at + ", down: each sampled tile row is within (blocks + 8 + k) F32 epsilons of the weighted sum of |w*x| of the row builds' row");
            }
        }
    }
}

// The four-head sparse build against the one-head build: 12 heads over one KV head against the same queries over six copies of it, every head the same bits.
// Over 64 rows (a row alone keeps the one-head build), heads 128 and 256 wide, f16 and f32 caches, 85 ids a row, some past a row's visible tokens and some invalid.
void sparse_grouped(Backend& g) {
    const size_t heads = 12, bt = g.kv_layout().block_tokens, length = 3 * bt + 5, stride = 85;
    uint32_t seed = 977;
    const auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    for (size_t dim : {size_t(128), size_t(256)})
        for (KVType kt : {KVType::f16, KVType::f32})
            for (size_t rows : {size_t(1), size_t(64)}) {
                const std::string at = "sparse attention " + std::to_string(dim) + " wide, " + kv_type_name(kt) + " cache, " + std::to_string(rows) + " rows";
                const size_t total = length + rows;
                std::vector<int32_t> map(blocks_for(total, bt));
                for (size_t i = 0; i < map.size(); ++i) map[i] = int32_t(map.size() - 1 - i);
                std::vector<float> k1(total * dim), v1(k1.size()), q(rows * heads * dim);
                for (size_t i = 0; i < k1.size(); ++i) k1[i] = float(int(next() % 2001u) - 1000) / 1600.0f, v1[i] = float(int(next() % 2001u) - 1000) / 900.0f;
                for (float& x : q) x = float(int(next() % 2001u) - 1000) / 700.0f;
                std::vector<uint32_t> ids(rows * stride);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t t = 0; t < stride; ++t) {
                        const uint32_t pick = next();
                        ids[r * stride + t] = t % 9 == 0 ? (pick % 2 ? UINT32_MAX : uint32_t(length + r + 1)) : uint32_t(pick % (length + r + 1));
                    }
                std::vector<float> outs[2];
                std::string ran[2];
                for (int side = 0; side < 2; ++side) {
                    const size_t kv_heads = side ? 6 : 1;
                    std::vector<float> k(total * kv_heads * dim), v(k.size());
                    for (size_t t = 0; t < total; ++t)
                        for (size_t h = 0; h < kv_heads; ++h) {
                            std::memcpy(&k[(t * kv_heads + h) * dim], &k1[t * dim], dim * 4);
                            std::memcpy(&v[(t * kv_heads + h) * dim], &v1[t * dim], dim * 4);
                        }
                    auto storage = g.kv_alloc(1, kv_heads, dim, map.size() * bt, kt, kt);
                    KVView w{storage.get(), map.data(), map.size(), 0, total};
                    auto kb = g.adopt(k.data(), k.size() * 4), vb = g.adopt(v.data(), v.size() * 4), qb = g.adopt(q.data(), q.size() * 4), ib = g.adopt(ids.data(), ids.size() * 4);
                    g.kv_write(0, &w, 1, {kb.get(), 0}, {vb.get(), 0});
                    auto ob = g.alloc(rows * heads * dim * 4);
                    KVView view{storage.get(), map.data(), map.size(), length, rows};
                    rocm_take_launches(g);
                    g.sparse_attention({qb.get(), 0}, 0, &view, 1, {ib.get(), 0}, stride, {ob.get(), 0}, int(heads), int(kv_heads), int(dim));
                    for (const auto& [kn, n] : rocm_take_launches(g))
                        if (kn.find("sparse_attention_vec") != std::string::npos && kn.find("merge") == std::string::npos) ran[side] = kn;
                    outs[side] = read(g, *ob, rows * heads * dim);
                }
                const std::string want = std::string("llmx_sparse_attention_vec") + (dim == 256 ? "256" : "");
                require(ran[0] == (rows > 1 ? want + "_g4" : want) && ran[1] == want, at + ": one KV head takes " + (rows > 1 ? want + "_g4" : want) + " and six take " + want + ", ran " + ran[0] + " and " + ran[1]);
                require(same(outs[0], outs[1]), at + ": every head of the four-head build is the one-head build's, bit for bit");
            }
}

// The routed builds by group: experts of 1 to 33 entries each side of every build's range and every narrow quad, an empty expert between, every type in f16 and int8.
// Every entry is the entry routed alone, bit for bit, at inner sizes ending the stream's last chunk full, half full, short and with a one-block tail, and MXFP4 at 65 blocks a row (docs/TESTS.md).
void experts_quads(Backend& g) {
    const size_t n_expert = 24, ff = 256, sizes[] = {1, 3, 4, 5, 8, 9, 12, 13, 16, 17, 32, 33};
    std::vector<std::pair<uint32_t, size_t>> kinds;
    for (uint32_t type : {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K})
        for (size_t nin : {size_t(2560), size_t(768)}) kinds.push_back({type, nin});
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4})
        for (size_t nin : {size_t(2560), size_t(640), size_t(800)}) kinds.push_back({type, nin});
    kinds.push_back({quant::GGML_TYPE_MXFP4, 2080});
    for (const auto& [type, nin] : kinds) {
        const std::string name = quant::storage_type(type)->name;
        const std::vector<uint8_t> w = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * ff, uint32_t(nin + 3)) : typed_matrix(type, nin, n_expert * ff, uint32_t(type * 19 + nin));
        auto wb = g.adopt(w.data(), w.size());
        for (Dtype dt : {Dtype::f16, Dtype::int8}) {
            const std::string at = name + " " + std::to_string(nin) + ", " + dtype_name(dt);
            const size_t rows = std::max<size_t>(tile_from_at(type, dt, false), 160);
            const std::vector<float> x = row_classes::values(rows * nin, uint32_t(nin + 47)), wts(rows, 1.0f);
            auto xb = g.adopt(x.data(), x.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
            // Expert 2i + 1 takes sizes[i] entries, interleaved with the others' over the rows; the rest have no expert.
            std::vector<uint32_t> ids(rows, Backend::kNoExpert);
            std::vector<std::vector<size_t>> of(std::size(sizes));
            for (size_t r = 0, left = 1; left;) {
                left = 0;
                for (size_t i = 0; i < std::size(sizes); ++i)
                    if (of[i].size() < sizes[i]) {
                        ids[r] = uint32_t(2 * i + 1);
                        of[i].push_back(r++);
                        left = 1;
                    }
            }
            const auto call = [&](const std::vector<uint32_t>& v, std::string& ran) {
                auto ib = g.adopt(v.data(), v.size() * 4);
                const std::vector<float> fill(rows * ff, 7.0f);
                auto yb = g.adopt(fill.data(), fill.size() * 4);
                const RowRun run{rows, rows};
                rocm_take_launches(g);
                g.matmul_experts({Projection{{type, {wb.get(), 0}}, {yb.get(), 0}, ff}}, {xb.get(), 0}, nin, rows, {{ib.get(), 0}, {tb.get(), 0}, 1, n_expert}, RowRuns{&run, 1}, dt);
                ran.clear();
                for (const auto& [kn, n] : rocm_take_launches(g)) ran += (ran.empty() ? "" : "+") + kn;
                return read(g, *yb, rows * ff);
            };
            std::string ran;
            const std::vector<float> all = call(ids, ran);
            require(ran.find("_routed_narrow") != std::string::npos && ran.find("_routed_tile") != std::string::npos && (ran.find("_routed_stream") != std::string::npos) == (dt == Dtype::int8),
                    at + ": a " + std::to_string(rows) + "-row prompt takes both routed tile builds, and the stream in int8 only, ran " + ran);
            require(type == quant::GGML_TYPE_Q8_0 || (ran.find("_routed_mid") != std::string::npos && (ran.find("_routed_thin") != std::string::npos) == (dt == Dtype::f16 && type == quant::GGML_TYPE_MXFP4)),
                    at + ": a K-type's groups of 17 to 32 entries take its mid tile, and MXFP4's of up to 8 its thin build in f16, ran " + ran);
            bool same = true;
            for (size_t i = 0; i < std::size(sizes); ++i)
                for (size_t r : of[i]) {
                    std::vector<uint32_t> one(rows, Backend::kNoExpert);
                    one[r] = ids[r];
                    const std::vector<float> alone = call(one, ran);
                    same = same && !std::memcmp(all.data() + r * ff, alone.data() + r * ff, ff * 4);
                }
            require(same, at + ": every entry of experts of 1 to 33 entries is the entry routed alone, bit for bit");
        }
    }
}

// Which routed build computes a group: each build launched alone over a sentinel output writes the groups its range holds and no other, each the bits of the call with every build.
// A group of up to 8 entries takes the stream in int8 and MXFP4's thin build in f16, then the narrow tile up to 16, a K-type's mid tile up to 32 and the 64-row tile past.
void routed_builds_witness(Backend& g) {
    const size_t n_expert = 20, ff = 128, sizes[] = {4, 5, 8, 9, 16, 17, 32, 33}, rows = 256;
    const char* builds[] = {"thin", "stream", "narrow", "mid", "tile"};
    const uint32_t sentinel = 0x7fc0deadu;
    for (const auto& [type, nin] : std::vector<std::pair<uint32_t, size_t>>{{quant::GGML_TYPE_MXFP4, 2048}, {quant::GGML_TYPE_Q4_K, 2560}, {quant::GGML_TYPE_Q8_0, 1024}}) {
        const bool ktype = type != quant::GGML_TYPE_Q8_0;
        const std::string name = quant::storage_type(type)->name;
        const std::vector<uint8_t> w = ktype ? typed_matrix(type, nin, n_expert * ff, uint32_t(type * 7 + 3)) : q8_matrix(nin, n_expert * ff, 5);
        auto wb = g.adopt(w.data(), w.size());
        const std::vector<float> x = row_classes::values(rows * nin, 61), wts(rows, 1.0f);
        auto xb = g.adopt(x.data(), x.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
        // Expert 2i + 1 takes sizes[i] consecutive rows.
        std::vector<uint32_t> ids(rows, Backend::kNoExpert);
        std::vector<std::pair<size_t, size_t>> span(std::size(sizes));
        for (size_t i = 0, r = 0; i < std::size(sizes); r += sizes[i], ++i) {
            span[i] = {r, sizes[i]};
            for (size_t j = 0; j < sizes[i]; ++j) ids[r + j] = uint32_t(2 * i + 1);
        }
        auto ib = g.adopt(ids.data(), ids.size() * 4);
        for (Dtype dt : {Dtype::f16, Dtype::int8}) {
            require(tile_from_at(type, dt, false) <= rows, name + " " + dtype_name(dt) + ": a prompt of " + std::to_string(rows) + " rows takes the routed tile");
            const auto expect = [&](size_t n) -> std::string {
                return n <= 8 ? (dt == Dtype::int8 ? "stream" : type == quant::GGML_TYPE_MXFP4 ? "thin" : "narrow") : n <= 16 ? "narrow" : n <= 32 && ktype ? "mid" : "tile";
            };
            const auto call = [&](const std::string& only) {
                const std::vector<uint32_t> fill(rows * ff, sentinel);
                auto yb = g.adopt(fill.data(), fill.size() * 4);
                const RowRun run{rows, rows};
                rocm_route_only(g, only.empty() ? only : "_routed_" + only);
                g.matmul_experts({Projection{{type, {wb.get(), 0}}, {yb.get(), 0}, ff}}, {xb.get(), 0}, nin, rows, {{ib.get(), 0}, {tb.get(), 0}, 1, n_expert}, RowRuns{&run, 1}, dt);
                rocm_route_only(g, "");
                return read(g, *yb, rows * ff);
            };
            const std::vector<float> all = call("");
            for (const std::string build : builds) {
                const std::vector<float> one = call(build);
                for (size_t i = 0; i < std::size(sizes); ++i) {
                    const auto [r0, n] = span[i];
                    bool written = true, kept = true;
                    for (size_t v = r0 * ff; v < (r0 + n) * ff; ++v) {
                        uint32_t bits;
                        std::memcpy(&bits, &one[v], 4);
                        written = written && bits != sentinel && !std::memcmp(&one[v], &all[v], 4);
                        kept = kept && bits == sentinel;
                    }
                    const bool mine = expect(n) == build;
                    require(mine ? written : kept, name + " " + dtype_name(dt) + ": a group of " + std::to_string(n) + " entries is " + (mine ? "" : "not ") + "computed by the " + build +
                                                       " build launched alone" + (mine ? ", with the bits of the call of every build" : ""));
                }
            }
        }
    }
}

// The routed tile's time against the grouped row builds over prompts of 32 to 512 rows at Flash-Next's expert shapes, medians of five interleaved; checks nothing.
// The tile arm times the tile only from the role's `tile_from`, so an entry is derived lower by building with it at 32.
void routed_tile_rates(Backend& g) {
    const size_t n_expert = 512, k = 10;
    std::cout << "backend-rocm: routed tile rates; type, dtype, shape x projections, prompt rows: us a call grouped rows, tile, ratio\n";
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4})
        for (const auto& sh : std::vector<std::array<size_t, 3>>{{2560, 640, 2}, {640, 2560, 1}, {768, 2560, 1}}) {
            const size_t nin = sh[0], ff = sh[1], nproj = sh[2], block = quant::storage_type(type)->block_size;
            if (nin % block || (nin == 768 && 640 % block == 0)) continue;
            const std::vector<uint8_t> w = typed_matrix(type, nin, n_expert * ff, 7);
            auto wa = g.adopt(w.data(), w.size()), wb = g.adopt(w.data(), w.size());
            for (size_t rows : {32, 128, 192, 256, 320, 384, 512}) {
                const size_t entries = rows * k;
                const std::vector<float> x = row_classes::values((nproj == 2 ? rows : entries) * nin, 900);
                std::vector<uint32_t> idv(entries);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t j = 0; j < k; ++j) idv[r * k + j] = uint32_t((r * 131 + j * 47 + (r * j) % 7) % n_expert);
                const std::vector<float> wts(entries, 0.1f);
                auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                auto ya = g.alloc(entries * ff * 4), yb = g.alloc(entries * ff * 4), yd = g.alloc(rows * ff * 4);
                const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, k, n_expert};
                std::vector<RowRun> short_runs(rows);
                for (size_t r = 0; r < rows; ++r) short_runs[r] = {r + 1, 1};
                // Runs of extent 2 keep the rows a prompt's, below the tile's extent, in one stretch.
                for (RowRun& r : short_runs) r.extent = 2;
                const RowRun prompt[1] = {{rows, rows}};
                for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                    const Projection pa{{type, {wa.get(), 0}}, {ya.get(), 0}, ff}, pb{{type, {wb.get(), 0}}, {yb.get(), 0}, ff};
                    const auto time = [&](RowRuns runs) {
                        const auto call = [&] {
                            if (nproj == 2) g.matmul_experts({pa, pb}, {xb.get(), 0}, nin, rows, routing, runs, dt);
                            else g.matmul_experts_add(pa.weight, {xb.get(), 0}, {yd.get(), 0}, nin, ff, rows, routing, runs, dt);
                        };
                        for (int i = 0; i < 2; ++i) call();
                        g.sync();
                        const int iters = 10;
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int i = 0; i < iters; ++i) call();
                        g.sync();
                        return 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
                    };
                    std::vector<double> rowb, tile;
                    for (int rep = 0; rep < 5; ++rep) {
                        rowb.push_back(time({short_runs.data(), rows}));
                        tile.push_back(time({prompt, 1}));
                    }
                    std::sort(rowb.begin(), rowb.end());
                    std::sort(tile.begin(), tile.end());
                    std::printf("backend-rocm: routed tile rate %s, %s, %zux%zu x%zu, %zu: %.1f us, %.1f us, %.3f\n", quant::storage_type(type)->name, dtype_name(dt), nin, ff, nproj, rows, rowb[2], tile[2], tile[2] / rowb[2]);
                }
            }
        }
    std::fflush(stdout);
}

// matmul_experts_add: Y plus each row's slots, product by product in slot order, the products those of matmul_experts over a row an entry.
void experts_add(Backend& g) {
    const size_t n_expert = 8, k = 4, nin = 640, nout = 2560;
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_F32}) {
        if (nin % quant::storage_type(type)->block_size) continue;
        std::vector<uint8_t> w;
        if (type == quant::GGML_TYPE_F32) {
            const std::vector<float> v = row_classes::values(nin * n_expert * nout, 21, 0.05f);
            w.assign((const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 4);
        } else {
            w = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * nout, 5) : typed_matrix(type, nin, n_expert * nout, uint32_t(type + 3));
        }
        auto wb = g.adopt(w.data(), w.size());
        for (size_t rows : {1, 7}) {
            const std::vector<float> x = row_classes::values(rows * k * nin, uint32_t(rows + 2)), y0 = row_classes::values(rows * nout, uint32_t(rows + 4));
            std::vector<uint32_t> idv(rows * k);
            for (size_t e = 0; e < idv.size(); ++e) idv[e] = uint32_t((e * 3 + 1) % n_expert);
            idv[k - 1] = Backend::kNoExpert;
            idv[0] = rows > 1 ? Backend::kNoExpert : idv[0];
            const std::vector<float> wts = row_classes::values(rows * k, uint32_t(rows + 6), 0.5f);
            auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
            const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, k, n_expert};
            for (Dtype dt : {Dtype::f32, Dtype::f16, Dtype::int8}) {
                if (type == quant::GGML_TYPE_F32 && dt != Dtype::f32) continue;
                auto yb = g.adopt(y0.data(), y0.size() * 4), parts = g.alloc(rows * k * nout * 4);
                g.matmul_experts({Projection{{type, {wb.get(), 0}}, {parts.get(), 0}, nout}}, {xb.get(), 0}, nin, rows * k, Backend::Routing{routing.ids, routing.weights, 1, n_expert}, {}, dt);
                g.matmul_experts_add({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, routing, {}, dt);
                const std::vector<float> p = read(g, *parts, rows * k * nout), got = read(g, *yb, rows * nout);
                bool ok = true;
                for (size_t r = 0; r < rows; ++r)
                    for (size_t o = 0; o < nout; ++o) {
                        float s = 0.0f;
                        for (size_t j = 0; j < k; ++j) {
                            if (idv[r * k + j] == Backend::kNoExpert) continue;
                            volatile float product = wts[r * k + j] * p[(r * k + j) * nout + o];
                            s = j == 0 ? product : s + product;
                        }
                        volatile float sum = y0[r * nout + o] + s;
                        if (ok && std::memcmp((const void*)&sum, &got[r * nout + o], 4))
                            std::fprintf(stderr, "  row %zu output %zu: %.9g against %.9g, y %.9g, first slot %.9g\n", r, o, got[r * nout + o], (float)sum, y0[r * nout + o], p[r * k * nout + o]);
                        ok = ok && !std::memcmp((const void*)&sum, &got[r * nout + o], 4);
                    }
                require(ok, std::string(quant::storage_type(type)->name) + " routed down of " + std::to_string(rows) + " rows, " + dtype_name(dt) + ": Y plus the slots in order, bit for bit");
            }
        }
    }
}

void qwen35(Backend& g) {
    q35::Pair p(g);
    size_t runs = 0;
    rocm_take_launches(g);
    size_t values = q35::check_ops(p, runs);
    // The invariance check's lengths cross the short build's rule, so a sequence's bits are held across both delta rule builds.
    const std::map<std::string, size_t> ran = rocm_take_launches(g);
    require(ran.count("llmx_gated_delta_rule") && ran.count("llmx_gated_delta_rule_short"), "the qwen35 checks run both delta rule builds");
    values += q35::check_drafter_ops(p, rocm_profile(g).argmax_slice, typed_matrix);
    std::cout << "backend-rocm: qwen35 ops: " << values << " values against the CPU, " << runs << " device runs bit for bit across views, orders and passes; decay flush, state storage\n";
}

// DeepSeek-V4.1's ops against the CPU (deepseek4_checks.hpp), those whose kernels form each value in the CPU's order held to its bits.
void deepseek4(Backend& g) {
    q35::Pair p(g);
    const size_t values = ds4::check_ops(p, {"hc_collapse_norm", "hc_expand", "indexer_scores", "rope_forms"});
    std::cout << "backend-rocm: deepseek4 ops: " << values << " values against the CPU, the formats, selections, routing picks, collapse, expansion, index scores and rope forms bit for bit\n";
}

const unsigned char kF32Module[] = {
#include "rocm/f32.inc"
};

// A collective whose sums first wait for every member's queued work, so no copy waits in a hardware queue the members share behind a peer's sum that waits for this member.
struct Settled final : Collective {
    Settled(std::vector<Backend*> members, std::unique_ptr<Collective> c) : members(std::move(members)), c(std::move(c)) {}
    Slice partial(size_t member) override { return c->partial(member); }
    void sum_into(const std::vector<Slice>& residual, size_t rows, size_t width) override {
        settle();
        c->sum_into(residual, rows, width);
    }
    void sum_in_place(size_t rows, size_t width) override {
        settle();
        c->sum_in_place(rows, width);
    }
    void settle() {
        for (Backend* b : members) b->wait(b->submit());
    }
    std::vector<Backend*> members;
    std::unique_ptr<Collective> c;
};

// Each fast sum's form forced over ranks that are backends of device 0, at widths 2 to 5, 8 and 12, sizes about the shares' and chunks' edges, and a give-up in each form (docs/TESTS.md, backend-rocm-sixteen).
// The ranks may share the device's hardware queues, so every sum first waits for the writes before it, and two sums in flight, which queue copies between sums, run only up to width 4.
void sum_forms() {
    HipFn& f = hip();
    real_launch = f.hipModuleLaunchKernel;
    real_copy = f.hipMemcpyAsync;
    const RocmSumForm forms[] = {RocmSumForm::one_shot, RocmSumForm::two_shots, RocmSumForm::chain, RocmSumForm::wrap};
    const char* const names[] = {"one-shot", "two shots", "chain", "wrap"};
    const size_t chunk = 4 * 4096;   // a chain's chunk in floats (rocm_collective.hpp, kChainUnits)
    auto print = [](bool ok, const std::string& what) { require(ok, what); };
    uint32_t seed = 1601;
    for (const size_t W : {2, 3, 4, 5, 8, 12}) {
        std::vector<BackendPtr> d;
        for (size_t m = 0; m < W; ++m) d.push_back(make_rocm_backend(0));
        const std::vector<Backend*> group = raw(d, W);
        const size_t cap = W > 5 ? 3 * chunk + 5 : 17 * chunk + 5;
        std::vector<size_t> sizes = {1, 4 * W - 1, 4 * W + 1, 20 * W + 3, chunk - 1, chunk + 1, cap};
        if (W <= 5) sizes.push_back(16 * chunk + 7);
        for (size_t k = 0; k < 4; ++k) {
            Settled c(group, rocm_collective(group, cap, 1, forms[k]));
            sum_calls(*c.c, W, true);
            group_sums::check(group, c, 1, sizes, seed, print);
            seed += 100;
            if (W <= 4) {
                std::unique_ptr<Collective> rows = rocm_collective(group, 64, 97, forms[k]);
                sums_in_flight(group, *rows, 97, seed);
                seed += 100;
            }
            std::cout << "backend-rocm: " << W << " ranks on one device sum in " << names[k] << ", into and in place at " << sizes.size() << " sizes up to " << cap << " floats"
                      << (W <= 4 ? ", two sequences in flight," : ",") << " each the oracle's bits\n";
        }
    }
    std::vector<BackendPtr> e = {make_rocm_backend(0), make_rocm_backend(0)};
    for (size_t k = 0; k < 4; ++k) {
        std::unique_ptr<Collective> c = rocm_collective(raw(e, 2), 64, 97, forms[k]);
        const std::vector<float> rows(97, 1.0f);
        for (size_t m = 0; m < 2; ++m) e[m]->write(*c->partial(m).buffer, 0, rows.data(), rows.size() * sizeof(float));
        group_calls.launches = 0;
        group_calls.fail = 2;
        f.hipModuleLaunchKernel = counted_launch;
        require(throws([&] { c->sum_in_place(1, 97); }), std::string(names[k]) + ": a group sum whose peer's launch failed did not throw");
        f.hipModuleLaunchKernel = real_launch;
        group_calls.fail = 0;
        c->wait(0);
        require(throws([&] { e[0]->raise(0); }), std::string(names[k]) + ": a group sum that gave up did not throw on its backend");
        bool poisoned = true;
        for (float v : read(*e[0], *c->partial(0).buffer, 97)) poisoned = poisoned && std::isnan(v);
        require(poisoned, std::string(names[k]) + ": a group sum that gave up left values other than NaN");
        sum_calls(*c, 2, false);
        group_sums::check(raw(e, 2), *c, 97, {1, 64}, seed, print);
        seed += 100;
        std::cout << "backend-rocm: a sum in " << names[k] << " whose peer's launch failed turned the collective conservative, later sums exact\n";
    }
}

// The fast path's kernel at the widest group on one device: a launch of llmx_group_sum a rank in each form, each on its own stream and area, sums each the oracle's bits and no rank giving up (docs/TESTS.md).
// Only kernels go to the streams while sums run, since HIP gives this many streams on one device shared hardware queues, where a rank's queued copy would wait behind a peer's sum.
struct SixteenPeers {
    float* stage[kMaxTensorWidth];
    unsigned* signal[kMaxTensorWidth];
    unsigned* status;
    unsigned* fault;
    unsigned* cancel;
};
static_assert(sizeof(SixteenPeers) == 280 && offsetof(SixteenPeers, status) == 256 && offsetof(SixteenPeers, cancel) == 272, "kernels/f32.hip, GroupPeers");

void sixteen_ranks() {
    HipFn& f = hip();
    const size_t W = kMaxTensorWidth, n = 3 * 4 * 4096 + 5, cap = (n + 3) / 4 * 4, signal = 8192, units = (n + 3) / 4;
    hip_check(f.hipSetDevice(0), "hipSetDevice");
    hipModule_t module = nullptr;
    hipFunction_t sums[3] = {};   // the wrap is the chain's kernel with flag 8
    hip_check(f.hipModuleLoadData(&module, kF32Module), "hipModuleLoadData");
    const char* const kernels[] = {"llmx_group_sum", "llmx_group_two_shots", "llmx_group_chain"};
    for (int k = 0; k < 3; ++k) hip_check(f.hipModuleGetFunction(&sums[k], module, kernels[k]), "hipModuleGetFunction");
    int khz = 0;
    hip_check(f.hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0), "hipDeviceGetAttribute");
    unsigned long long timeout = (unsigned long long)khz * 1000, tick = (unsigned long long)khz;
    std::vector<void*> area(W), own(W), out(W);
    std::vector<hipStream_t> streams(W);
    unsigned *status = nullptr, *fault = nullptr, *cancel = nullptr;
    hip_check(f.hipHostMalloc((void**)&status, W * sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    hip_check(f.hipHostMalloc((void**)&fault, sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    hip_check(f.hipHostMalloc((void**)&cancel, sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    std::memset(status, 0, W * sizeof(unsigned));
    *fault = 0;
    *cancel = 0;
    for (size_t k = 0; k < W; ++k) {
        hip_check(f.hipExtMallocWithFlags(&area[k], signal + W * cap * sizeof(float), hipDeviceMallocUncached), "hipExtMallocWithFlags");
        hip_check(f.hipMalloc(&own[k], n * sizeof(float)), "hipMalloc");
        hip_check(f.hipMalloc(&out[k], n * sizeof(float)), "hipMalloc");
        hip_check(f.hipStreamCreateWithFlags(&streams[k], 1), "hipStreamCreateWithFlags");
        hip_check(f.hipMemsetAsync(area[k], 0, signal, streams[k]), "hipMemsetAsync");
    }
    for (hipStream_t s : streams) hip_check(f.hipStreamSynchronize(s), "hipStreamSynchronize");
    SixteenPeers peers{};
    for (size_t k = 0; k < W; ++k) {
        peers.stage[k] = (float*)((uint8_t*)area[k] + signal);
        peers.signal[k] = (unsigned*)area[k];
    }
    peers.status = status;
    peers.fault = fault;
    peers.cancel = cancel;
    // The kernel reads the table from device memory, as the collective writes it at join.
    void* table = nullptr;
    hip_check(f.hipMalloc(&table, sizeof(SixteenPeers)), "hipMalloc");
    hip_check(f.hipMemcpyAsync(table, &peers, sizeof(SixteenPeers), hipMemcpyHostToDevice, streams[0]), "hipMemcpyAsync");
    hip_check(f.hipStreamSynchronize(streams[0]), "hipStreamSynchronize");
    uint32_t seed = 1601;
    // Each form's blocks as the collective launches them (rocm_collective.hpp): the one-shot's 16, two shots a block a 256 units of a share, the chain two a chunk of 4096 units.
    const unsigned blocks[] = {16, (unsigned)(((units + W - 1) / W + 255) / 256), (unsigned)(2 * ((units + 4095) / 4096)), (unsigned)(2 * ((units + 4095) / 4096))};
    for (int form = 0; form < 4; ++form)
    for (const bool replace : {false, true}) {
        const group_sums::Case c = group_sums::make(W, n, seed++);
        for (size_t k = 0; k < W; ++k) {
            hip_check(f.hipMemcpyAsync(own[k], c.partial[k].data(), n * sizeof(float), hipMemcpyHostToDevice, streams[k]), "hipMemcpyAsync");
            hip_check(f.hipMemcpyAsync(out[k], replace ? c.partial[k].data() : c.start.data(), n * sizeof(float), hipMemcpyHostToDevice, streams[k]), "hipMemcpyAsync");
            hip_check(f.hipStreamSynchronize(streams[k]), "hipStreamSynchronize");
        }
        for (size_t k = 0; k < W; ++k) {
            const float* o = (const float*)own[k];
            float* x = (float*)(replace ? own[k] : out[k]);
            unsigned long long nn = n, cc = cap;
            int members = (int)W, rank = (int)k, flags = 2 | (replace ? 1 : 0) | (form == 3 ? 8 : 0);
            void* args[] = {&table, &o, &x, &nn, &cc, &members, &rank, &flags, &timeout, &tick};
            hip_check(f.hipModuleLaunchKernel(sums[form < 3 ? form : 2], blocks[form], 1, 1, 256, 1, 1, 0, streams[k], args, nullptr), "hipModuleLaunchKernel");
        }
        for (hipStream_t s : streams) hip_check(f.hipStreamSynchronize(s), "hipStreamSynchronize");
        bool met = !*fault;
        for (size_t k = 0; k < W; ++k) met = met && !status[k];
        require(met, "a rank of the sixteen gave up waiting for its peers in form " + std::to_string(form) + (replace ? " in place" : " into"));
        const std::vector<float> want = group_sums::oracle(c, replace);
        for (size_t k = 0; k < W; ++k) {
            std::vector<float> got(n);
            hip_check(f.hipMemcpyAsync(got.data(), replace ? own[k] : out[k], n * sizeof(float), hipMemcpyDeviceToHost, streams[k]), "hipMemcpyAsync");
            hip_check(f.hipStreamSynchronize(streams[k]), "hipStreamSynchronize");
            require(same(got, want), "rank " + std::to_string(k) + " of sixteen in form " + std::to_string(form) + " is not the oracle's sum in member order");
        }
    }
    for (size_t k = 0; k < W; ++k) {
        (void)f.hipStreamDestroy(streams[k]);
        (void)f.hipFree(area[k]);
        (void)f.hipFree(own[k]);
        (void)f.hipFree(out[k]);
    }
    (void)f.hipFree(table);
    (void)f.hipHostFree(status);
    (void)f.hipHostFree(fault);
    (void)f.hipHostFree(cancel);
    (void)f.hipModuleUnload(module);
    std::cout << "backend-rocm: " << W << " ranks of the fast sum on one device in each form, into and in place, each the oracle's bits\n";
}

// The grouped sparse softmax's head width: it runs while its four heads' queries fit the shared memory its own arrays leave, read here from the kernel itself, and the one-head build past that.
// At the widest head that fits and one block wider, a 16-row call takes its build and gives every row the bits of the same row called alone, which the one-head build runs.
void sparse_head_budget(Backend& g) {
    HipFn& f = hip();
    hipModule_t module = nullptr;
    hipFunction_t softmax = nullptr;
    hip_check(f.hipModuleLoadData(&module, kF32Module), "hipModuleLoadData");
    hip_check(f.hipModuleGetFunction(&softmax, module, "llmx_sparse_softmax_g"), "hipModuleGetFunction");
    int fixed = 0;
    hip_check(f.hipFuncGetAttribute(&fixed, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, softmax), "hipFuncGetAttribute");
    hip_check(f.hipModuleUnload(module), "hipModuleUnload");
    const size_t widest = (rocm_device(0).caps.shared_bytes - size_t(fixed)) / 4 / 4 / 32 * 32, heads = 2, rows = 16, k = 24, kv_rows = 64;
    std::cout << "backend-rocm: the four-head sparse softmax holds heads up to " << widest << " values wide beside its " << fixed << " bytes of shared memory\n";
    const auto fmt = ModelFormat::fp8_power;
    for (const size_t dim : {widest, widest + 32}) {
        const size_t words = format_words(fmt, dim);
        const std::vector<float> x = row_classes::values(kv_rows * dim, 71, 2.0f), q = row_classes::values(rows * heads * dim, 72), sink = row_classes::values(heads, 73);
        std::vector<uint32_t> ids(rows * k);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t((i * 37 + i / k) % kv_rows);
        auto xb = g.adopt(x.data(), x.size() * 4), qb = g.adopt(q.data(), q.size() * 4), sb = g.adopt(sink.data(), sink.size() * 4), ib = g.adopt(ids.data(), ids.size() * 4);
        auto kv = g.alloc(kv_rows * words * 4), all = g.alloc(rows * heads * dim * 4), one = g.alloc(rows * heads * dim * 4);
        g.model_format_pack({kv.get(), 0}, words, {xb.get(), 0}, kv_rows, dim, fmt);
        const float scale = 1.0f / std::sqrt(float(dim));
        rocm_take_launches(g);
        g.sparse_attention_sink({all.get(), 0}, {qb.get(), 0}, {kv.get(), 0}, {ib.get(), 0}, {sb.get(), 0}, rows, heads, dim, kv_rows, k, scale, 0, nullptr, 0, {}, 0, fmt, fmt);
        const bool fits = dim == widest, grouped = rocm_take_launches(g).count("llmx_sparse_softmax_g") == 1;
        require(grouped == fits, std::to_string(rows) + " rows of heads " + std::to_string(dim) + " wide take the " + (fits ? "grouped" : "one-head") + " sparse softmax, the widest that fits " +
                                     std::to_string(widest));
        for (size_t r = 0; r < rows; ++r)
            g.sparse_attention_sink({one.get(), r * heads * dim}, {qb.get(), r * heads * dim}, {kv.get(), 0}, {ib.get(), r * k}, {sb.get(), 0}, 1, heads, dim, kv_rows, k, scale, 0, nullptr, 0, {},
                                    0, fmt, fmt);
        require(same(read(g, *all, rows * heads * dim), read(g, *one, rows * heads * dim)), "every row of a 16-row sparse attention call of heads " + std::to_string(dim) +
                                                                                             " wide is the row called alone, bit for bit");
    }
}

int main(int argc, char** argv) {
    // `sixteen` runs groups of up to 16 ranks on one device, which with the backend's own stream need more hardware queues than HIP's default of 4; a real group runs one rank per device and needs none.
    const bool sixteen = argc > 1 && std::string(argv[1]) == "sixteen";
    if (sixteen) setenv("GPU_MAX_HW_QUEUES", "32", 1);
    BackendPtr g;
    try {
        g = make_rocm_backend(0);
    } catch (const RocmUnavailable& e) {
        std::cout << "backend-rocm: skipped, " << e.what() << "\n";
        return 77;
    }
    BackendPtr c = make_cpu_backend();
    c->set_threads(4);
    try {
        if (sixteen) {
            sixteen_ranks();
            sum_forms();
            std::cout << (failures ? "backend-rocm: FAIL\n" : "backend-rocm: PASS\n");
            return failures ? 1 : 0;
        }
        // `rates` prints the twin types' product times at Flash-Next's shapes, the exact products' times, attention's times and the routed products' times and break-evens, and checks nothing.
        // `rates attention`, `rates argmax`, `rates routed`, `rates tile`, `rates exact` or `rates decode` prints one part.
        if (argc > 1 && std::string(argv[1]) == "rates") {
            const std::string part = argc > 2 ? argv[2] : "";
            if (part.empty()) rates(*g);
            if (part.empty() || part == "attention") attention_rates(*g);
            if (part.empty() || part == "argmax") argmax_rates(*g);
            if (part.empty() || part == "routed") routed_rates(*g);
            if (part.empty() || part == "tile") routed_tile_rates(*g);
            if (part.empty() || part == "exact") exact_rates(*g);
            if (part.empty() || part == "decode") decode_rates(*g);
            return 0;
        }
        // `handoffs` runs only the device-side crossings over every visible device, and the busy times across them.
        if (argc > 1 && std::string(argv[1]) == "handoffs") {
            handoffs();
            busy_times();
            std::cout << (failures ? "backend-rocm handoffs: FAIL\n" : "backend-rocm handoffs: PASS\n");
            return failures ? 1 : 0;
        }
        // `qwen35` runs only the qwen35 layers' ops against the CPU and the device's bit-for-bit rules for them (qwen35_checks.hpp).
        if (argc > 1 && std::string(argv[1]) == "qwen35") {
            qwen35(*g);
            std::cout << (failures ? "backend-rocm qwen35: FAIL\n" : "backend-rocm qwen35: PASS\n");
            return failures ? 1 : 0;
        }
        // `deepseek4` runs only DeepSeek-V4.1's ops against the CPU.
        if (argc > 1 && std::string(argv[1]) == "deepseek4") {
            deepseek4(*g);
            sparse_head_budget(*g);
            std::cout << (failures ? "backend-rocm deepseek4: FAIL\n" : "backend-rocm deepseek4: PASS\n");
            return failures ? 1 : 0;
        }
        // `moe` runs only the routed experts' checks, the focused test of their kernels.
        if (argc > 1 && std::string(argv[1]) == "moe") {
            routing(*g, *c);
            experts(*g, *c);
            experts_grouped(*g);
            experts_tile(*g, *c);
            experts_quads(*g);
            routed_builds_witness(*g);
            experts_add(*g);
            std::cout << (failures ? "backend-rocm moe: FAIL\n" : "backend-rocm moe: PASS\n");
            return failures ? 1 : 0;
        }
        // `scratch [FILE [DEVICE]]` runs only the check that a fitted model's passes grow no device memory past what its fit priced, on FILE too when given.
        if (argc > 1 && std::string(argv[1]) == "scratch") {
            scratch_reserved(argc > 2 ? argv[2] : "", argc > 3 ? std::atoi(argv[3]) : 0);
            std::cout << (failures ? "backend-rocm scratch: FAIL\n" : "backend-rocm scratch: PASS\n");
            return failures ? 1 : 0;
        }
        // `attention` runs only the attention checks, the focused test of its kernels.
        if (argc > 1 && std::string(argv[1]) == "attention") {
            attention(*g, *c);
            classes(*g);
            attention_twins(*g, Dtype::f16);
            attention_twins(*g, Dtype::int8);
            attention_chunks(*g);
            vec_bits(*g);
            sparse_grouped(*g);
            std::cout << (failures ? "backend-rocm attention: FAIL\n" : "backend-rocm attention: PASS\n");
            return failures ? 1 : 0;
        }
        // `twins` runs only the producers' twins and the kept twin's rules, the hyper-connection mix's one launch against its chain and the fused moe_step against the base's.
        if (argc > 1 && std::string(argv[1]) == "twins") {
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                producer_twins(*g, dt);
                stream_twins(*g, dt);
            }
            hyper_mix_bits(*g);
            if (rocm_devices(2).size() == 2) hyper_mix_bits(*make_rocm_backend(1), g.get());
            moe_step_bits(*g);
            moe_rows_bits(*g);
            std::cout << (failures ? "backend-rocm twins: FAIL\n" : "backend-rocm twins: PASS\n");
            return failures ? 1 : 0;
        }
        // `delta` runs only the fused linear-attention step against the base's calls.
        if (argc > 1 && std::string(argv[1]) == "delta") {
            delta_step_bits(*g);
            std::cout << (failures ? "backend-rocm delta: FAIL\n" : "backend-rocm delta: PASS\n");
            return failures ? 1 : 0;
        }
        // `types` runs only the weight types' check, the focused test of their kernels.
        if (argc > 1 && std::string(argv[1]) == "types") {
            types(*g, *c);
            exact_rows(*g);
            twin_types(*g, *c);
            q8_decode_bits(*g);
        mxfp4_decode_bits(*g);
            mxfp4_decode_bits(*g);
            std::cout << (failures ? "backend-rocm types: FAIL\n" : "backend-rocm types: PASS\n");
            return failures ? 1 : 0;
        }
        buffers(*g);
        host_views(*g);
        refusals(*g);
        ops(*g, *c);
        types(*g, *c);
        exact_rows(*g);
        twin_types(*g, *c);
        q8_decode_bits(*g);
        mxfp4_decode_bits(*g);
        attention(*g, *c);
        classes(*g);
        twin_products(*g, Dtype::f16);
        producer_twins(*g, Dtype::f16);
        stream_twins(*g, Dtype::f16);
        twin_products(*g, Dtype::int8);
        producer_twins(*g, Dtype::int8);
        stream_twins(*g, Dtype::int8);
        hyper_mix_bits(*g);
        if (rocm_devices(2).size() == 2) hyper_mix_bits(*make_rocm_backend(1), g.get());
        moe_step_bits(*g);
        delta_step_bits(*g);
        moe_rows_bits(*g);
        fused_norm_rope_kv(*g);
        attention_twins(*g, Dtype::f16);
        attention_twins(*g, Dtype::int8);
        host_output(*g);
        host_copies(*g);
        attention_chunks(*g);
        vec_bits(*g);
        sparse_grouped(*g);
        pass_args(*g);
        records();
        failure();
        collective();
        handoffs();
        busy_times();
        qwen35(*g);
        deepseek4(*g);
        sparse_head_budget(*g);
        routing(*g, *c);
        experts(*g, *c);
        experts_add(*g);
        experts_quads(*g);
        routed_builds_witness(*g);
    } catch (const std::exception& e) {
        std::cout << "FAIL threw: " << e.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "backend-rocm: FAIL\n" : "backend-rocm: PASS\n");
    return failures ? 1 : 0;
}
