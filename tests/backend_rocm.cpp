// backend-rocm: the ROCm backend's buffers, tickets, refusals and dense F32 ops against the CPU backend, and each op's batch invariance; exits 77 without a runtime or a device.
#include "backends/cpu/cpu_backend.hpp"
#include "backends/gpu/backend_core.hpp"
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
#include "moe_checks.hpp"
#include "qwen35_checks.hpp"
#include "row_classes.hpp"
#include "scratch_growth.hpp"
#include "twin_checks.hpp"
#include "attention_tiles.hpp"
#include "dspine_checks.hpp"
#include "vision_checks.hpp"

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
#include <set>
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
using twin_checks::flash_shapes;
using twin_checks::q8_matrix;
using twin_checks::read;
using twin_checks::twin_rule;
using twin_checks::typed_matrix;
void close(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) { require(twin_checks::f32_close(got, want), what); }
using twin_checks::same;

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
    require(g.supports_type(quant::GGML_TYPE_F32) && g.supports_type(quant::GGML_TYPE_Q8_0) && g.supports_type(quant::GGML_TYPE_Q4_K) && !g.supports_type(quant::GGML_TYPE_I8),
            "the backend supports the types of its descriptor table and no other");
    require(throws([&] { g.matmul({quant::GGML_TYPE_I8, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 256, 4, 1); }), "an I8 matmul is refused");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q4_K, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 128, 4, 1); }), "a Q4_K row ending inside a block is refused");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q8_0, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 48, 4, 1); }), "a Q8_0 row ending inside a block is refused");
    // The ops ROCm implements are its `implements`, which the kernel checks below reach one by one: every op but the ones it lacks.
    // The loop ends at Op's last enumerator (backend.hpp), so an Op appended there moves this end with it.
    for (int i = 0; i <= int(Op::chain_pick); ++i) {
        const Op op = Op(i);
        const bool lacked = op == Op::mixed_experts || op == Op::sigmoid_routing;
        require(g.implements(op) != lacked, std::string("the backend reports ") + backend::op_name(op) + (lacked ? ", which it lacks" : " as it lacks it"));
    }
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

using twin_checks::DoubleSums;
using twin_checks::double_sums;
void exact_close(const std::vector<float>& got, const DoubleSums& d, const std::string& what) { require(twin_checks::exact_close(got, d), what); }

// The descriptor table's types at Flash-Next's shapes: embeddings the CPU's rows bit for bit, exact products of the CPU and the device within the F32 bound of a double sum, and a row of a call the row alone.
void types(Backend& g, Backend& c) {
    const uint32_t kinds[] = {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0,
                              quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_F16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_NVFP4,
                              quant::GGML_TYPE_Q2_K, quant::GGML_TYPE_Q3_K, quant::GGML_TYPE_IQ4_XS,
                              quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS, quant::GGML_TYPE_IQ2_S, quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ3_S,
                              quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0, quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0};
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
            // A type with twins takes its exact kernel in the F32 class alone (twin_types), and F16 weights' prompt rows outside it the half tile on binary16 inputs (half_tile).
            const bool twins = !twin_checks::exact_only(type), halves = type == quant::GGML_TYPE_F16;
            std::vector<float> xh(x.size());
            for (size_t i = 0; i < x.size(); ++i) xh[i] = f16_to_f32(f32_to_f16(x[i]));
            const DoubleSums want_half = halves ? double_sums(read(c, *ce, nout * nin), xh, nin, nout) : DoubleSums{};
            for (Dtype dt : {Dtype::int8, Dtype::f16, Dtype::f32}) {
                if (twins && dt != Dtype::f32) continue;
                const bool tiled = halves && dt != Dtype::f32;
                MatrixPaths paths;
                g.swap_matrix_paths(paths);
                g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, 33, {}, dt);
                g.swap_matrix_paths(paths);
                exact_close(read(g, *yb, nout * 33), tiled ? want_half : want, "a " + at + " matmul of 33 rows against the double sum, " + dtype_name(dt));
                require(paths.take() == std::vector<std::string>{tiled ? "f16" : "f32"},
                        "a " + at + " matmul runs the " + (tiled ? "half tile, " : "exact kernel, ") + dtype_name(dt));
            }
            const std::vector<float> all = read(g, *yb, nout * 33);
            auto y1 = g.alloc(nout * 4);
            g.matmul({type, {wb.get(), 0}}, {xb.get(), 32 * nin}, {y1.get(), 0}, nin, nout, 1, {}, Dtype::f32);
            require(std::memcmp(read(g, *y1, nout).data(), all.data() + 32 * nout, nout * 4) == 0, "a row of a " + at + " matmul is the row alone, bit for bit");
        }
    }
}

// The shared twin checks (twin_checks.hpp) at this device's wave, over the types and classes ROCm runs.
bool quarter_form(uint32_t type, size_t nin, bool int8) { return twin_checks::quarter_form(type, nin, rocm_device(0).caps.subgroup_size, int8); }
// The type witnesses that read the device's caps and profile (twin_checks::Bits), with ROCm's launches.
twin_checks::Bits& bits() {
    static twin_checks::Bits b{rocm_device(0).caps, rocm_profile, rocm_take_launches, require};
    return b;
}
// The routed experts' checks (moe_checks.hpp) at this device's wave, with ROCm's launches and routed-build filter.
moe_checks::Moe& moe() {
    static moe_checks::Moe m{rocm_device(0).caps.subgroup_size, rocm_take_launches, rocm_route_only, require};
    return m;
}
void exact_rows(Backend& g) {
    twin_checks::exact_rows(g, {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0,
                               quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_F16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_NVFP4,
                               quant::GGML_TYPE_Q2_K, quant::GGML_TYPE_Q3_K, quant::GGML_TYPE_IQ4_XS,
                              quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS, quant::GGML_TYPE_IQ2_S, quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ3_S,
                              quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0, quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0},
                            rocm_take_launches, require);
}
// At prompt_rows_below 2, so a prompt of 2 rows takes the tile it takes on devices without the crossover; prompt_crossover holds the rows below it.
void twin_types(Backend& g, Backend& c) {
    DeviceProfile& profile = rocm_profile(g);
    const size_t keep = profile.prompt_rows_below;
    profile.prompt_rows_below = 2;
    twin_checks::twin_types(g, c, {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL,
                                   quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_Q3_K, quant::GGML_TYPE_Q2_K, quant::GGML_TYPE_IQ4_XS, quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS, quant::GGML_TYPE_IQ3_S,
                                   quant::GGML_TYPE_IQ2_S, quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M,
                                   quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0},
                            {Dtype::f16, Dtype::int8}, rocm_device(0).caps.subgroup_size, rocm_take_launches, [](uint32_t, Dtype, size_t, size_t, size_t) { return std::string(); }, require);
    profile.prompt_rows_below = keep;
}

// At prompt_rows_below 2, so its products of up to 5 rows reach the tile kernels the check requires to have run.
void nvfp4_scales(Backend& g, Backend& c) {
    DeviceProfile& profile = rocm_profile(g);
    const size_t keep = profile.prompt_rows_below;
    profile.prompt_rows_below = 2;
    twin_checks::nvfp4_scales(g, c, rocm_take_launches, require);
    profile.prompt_rows_below = keep;
}

// The 8-bit tile's offset types, IQ4_NL and the types whose decode sets no min take their term in one fma against the scale times 1.5 * 2^23 (docs/PRECISION.md, Biased sums in the 8-bit HIP tile).
// An infinite weight scale gives NaN at each output it reaches, where a multiply gave an infinity, and leaves the other outputs' bits.
void biased_edges(Backend& g) {
    for (uint32_t type : {quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_IQ4_XS, quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS,
                          quant::GGML_TYPE_IQ2_S, quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ3_S, quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M, quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0}) {
        const quant::StorageType& st = *quant::storage_type(type);
        const std::string name = st.name;
        std::string tname = name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        const size_t nin = 512, nout = 64, rows = 64, blocks = nin / st.block_size;
        std::vector<uint8_t> w = typed_matrix(type, nin, nout, uint32_t(type * 13 + 5));
        const std::vector<float> x = row_classes::values(nin * rows, 17);
        auto xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * rows * 4);
        const RowRun rr{rows, rows};
        const auto run = [&]() {
            auto wb = g.adopt(w.data(), w.size());
            rocm_take_launches(g);
            g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, {&rr, 1}, Dtype::int8);
            std::map<std::string, size_t> ran;
            for (const auto& [k, v] : rocm_take_launches(g))
                if (k.rfind("llmx_" + tname, 0) == 0) ran[k] = v;
            require(ran.size() == 1 && ran.begin()->first == "llmx_" + tname + "_tile8", "a " + name + " int8 prompt of 64 rows runs its 8-bit tile alone");
            return read(g, *yb, nout * rows);
        };
        const std::vector<float> base = run();
        // Output 5's second block: its half scale ends a Q6_K block, sits in the high nibbles of an IQ1_M block's bytes 49, 51, 53 and 55, and leads every other one here.
        const size_t at = (5 * blocks + 1) * st.type_size + (type == quant::GGML_TYPE_Q6_K ? 208 : 0);
        if (type == quant::GGML_TYPE_IQ1_M) w[at + 49] &= 0x0f, w[at + 51] &= 0x0f, w[at + 53] = uint8_t((w[at + 53] & 0x0f) | 0xc0), w[at + 55] = uint8_t((w[at + 55] & 0x0f) | 0x70);
        else w[at] = 0x00, w[at + 1] = 0x7c;
        const std::vector<float> got = run();
        bool nan = true, kept = true;
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < nout; ++o) {
                const size_t i = r * nout + o;
                if (o == 5) nan = nan && std::isnan(got[i]);
                else kept = kept && std::memcmp(&got[i], &base[i], 4) == 0;
            }
        require(nan, "a " + name + " int8 tile gives NaN at an output whose weight scale is infinite");
        require(kept, "a " + name + " int8 tile keeps the other outputs' bits beside an infinite weight scale");
    }
}

// Printed, never asserted: the exact products at Qwen3.8-Flash-Next's F32 and BF16 shapes and Qwen3's F16 ones over 1 to 293 rows, and the F16 half tile over 293 and 512, in wall time a call.
void exact_rates(Backend& g) {
    struct Shape { const char* name; uint32_t type; size_t nin, nout; };
    const Shape shapes[] = {{"flash router", quant::GGML_TYPE_F32, 2560, 512}, {"flash hc_pre", quant::GGML_TYPE_F32, 10240, 24}, {"flash hc_post", quant::GGML_TYPE_F32, 10240, 4},
                            {"flash ssm_alpha", quant::GGML_TYPE_F32, 2560, 24}, {"flash shexp gate", quant::GGML_TYPE_F32, 2560, 1},
                            {"flash bf16 512", quant::GGML_TYPE_BF16, 2560, 512}, {"flash bf16 128", quant::GGML_TYPE_BF16, 2560, 128},
                            {"qwen3-0.6b f16 ffn_up", quant::GGML_TYPE_F16, 1024, 3072}, {"qwen3-4b f16 ffn_up", quant::GGML_TYPE_F16, 2560, 9728},
                            {"qwen3-4b f16 ffn_down", quant::GGML_TYPE_F16, 9728, 2560}};
    std::cout << "backend-rocm: exact rates; type, shape (in x out), rows, and F16 weights' half tile in the F16 class: us a call, TMAC/s, kernels\n";
    for (const Shape& s : shapes) {
        const std::vector<float> x = row_classes::values(s.nin * 512, 901), v = row_classes::values(s.nin * s.nout, 902);
        std::vector<uint8_t> w(quant::row_bytes(s.type, s.nin, s.nout));
        for (size_t i = 0; i < v.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &v[i], 4);
            if (s.type == quant::GGML_TYPE_F32) std::memcpy(&w[4 * i], &u, 4);
            else if (s.type == quant::GGML_TYPE_F16) w[2 * i] = uint8_t(f32_to_f16(v[i])), w[2 * i + 1] = uint8_t(f32_to_f16(v[i]) >> 8);
            else w[2 * i] = uint8_t(u >> 16), w[2 * i + 1] = uint8_t(u >> 24);
        }
        auto xb = g.adopt(x.data(), x.size() * 4), wb = g.adopt(w.data(), w.size()), y = g.alloc(s.nout * 512 * 4);
        for (const auto& [n, dt] : std::vector<std::pair<size_t, Dtype>>{{1, Dtype::f32}, {2, Dtype::f32}, {3, Dtype::f32}, {4, Dtype::f32}, {8, Dtype::f32}, {9, Dtype::f32},
                                                                         {293, Dtype::f32}, {293, Dtype::f16}, {512, Dtype::f16}}) {
            if (dt != Dtype::f32 && s.type != quant::GGML_TYPE_F16) continue;
            const auto call = [&] { g.matmul({s.type, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, s.nin, s.nout, n, {}, dt); };
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
            std::printf("backend-rocm: exact rate %s %s (%zu x %zu), %zu, %s: %.1f us, %.3f TMAC/s, %s\n", quant::storage_type(s.type)->name, s.name, s.nin, s.nout, n, dtype_name(dt), us,
                        double(s.nin) * s.nout * n / us / 1e6, ran.c_str());
        }
    }
}

// Printed, never asserted: the twin types' products at Qwen3.8-Flash-Next's projection shapes, generated rows of 1, 16 and 64 columns and a 320-row prompt, in wall time a call over a synchronized loop, with the kernels that ran.
void rates(Backend& g) {
    struct Shape { const char* name; size_t nin, nout; };
    const Shape shapes[] = {{"flash attn_qkv", 2560, 10240}, {"flash ssm_out", 6144, 2560}, {"flash hc_down", 10240, 320}, {"flash shexp_gate", 2560, 640}};
    const uint32_t kinds[] = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
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

// Decode calls, printed, never asserted: one generated row's dense Q8_0 products at Flash-Next's shapes and Q4_0 to Q5_1 at Qwen3's in both twin classes, from weights outside the L2 cache, the gated delta rule and the router.
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
    // One generated row of Qwen3-0.6B's, 4B's and 14B's dense products in the 32-value twin types, by which row_form's lane form is chosen; GB/s counts the weight bytes read.
    const Shape dense[] = {{"0.6b ffn_up", 1024, 3072}, {"0.6b ffn_down", 3072, 1024}, {"4b ffn_up", 2560, 9728}, {"4b ffn_down", 9728, 2560}, {"14b attn_out", 5120, 5120},
                           {"14b attn_qkv", 5120, 7168}, {"14b ffn_up", 5120, 17408}, {"14b ffn_down", 17408, 5120}};
    for (uint32_t type : {quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1})
        for (const Shape& s : dense) {
            const std::vector<float> x = row_classes::values(s.nin, 902);
            const std::vector<uint8_t> w = typed_matrix(type, s.nin, s.nout, 9);
            std::vector<BufferPtr> wb;
            for (size_t i = 0; i < std::max<size_t>(1, (size_t(64) << 20) / w.size()); ++i) wb.push_back(g.adopt(w.data(), w.size()));
            auto xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(s.nout * 4);
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                const RowRun rr{1, 1};
                rocm_take_launches(g);
                size_t turn = 0;
                const auto [us, spread] = median_us(g, [&] { g.matmul({type, {wb[turn++ % wb.size()].get(), 0}}, {xb.get(), 0}, {y.get(), 0}, s.nin, s.nout, 1, {&rr, 1}, dt); });
                std::printf("backend-rocm: decode rate %s %s %s (%zu x %zu): %.1f us (spread %.1f), %.0f GB/s, %s\n", quant::storage_type(type)->name, dtype_name(dt), s.name, s.nin, s.nout, us,
                            spread, w.size() / us / 1e3, kernels().c_str());
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
    // quick_gelu with and without the limit, beside the calibration it is read against: the device's silu at quick_gelu's sigmoid arguments, gate 1.702 g and up 1, on the path the form leaves as it was.
    {
        const size_t n = 50001;
        const std::vector<float> gv = row_classes::values(n, 8, 12.0f), uv = row_classes::values(n, 9, 12.0f), one(n, 1.0f);
        std::vector<float> scaled(n);
        for (size_t i = 0; i < n; ++i) scaled[i] = 1.702f * gv[i];
        auto worst = [](const std::vector<float>& a, const std::vector<float>& b) {
            double m = 0;
            for (size_t i = 0; i < a.size(); ++i) m = std::max(m, double(std::fabs(a[i] - b[i])) / (1.0 + std::fabs(b[i])));
            return m;
        };
        std::vector<float> cal[2], qg[2][2];
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto gb = b.adopt(gv.data(), n * 4), ub = b.adopt(uv.data(), n * 4), sb = b.adopt(scaled.data(), n * 4), ob = b.adopt(one.data(), n * 4), out = b.alloc(n * 4);
            b.silu_mul({out.get(), 0}, {sb.get(), 0}, {ob.get(), 0}, n, 0.0f);
            cal[k] = read(b, *out, n);
            for (int l = 0; l < 2; ++l) {
                b.silu_mul({out.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, l ? 7.0f : 0.0f, {}, backend::GluForm::quick_gelu);
                qg[l][k] = read(b, *out, n);
            }
        }
        std::printf("  silu at quick_gelu's arguments %.3g, quick_gelu %.3g and at limit 7 %.3g from the CPU, relative to 1 + |CPU|\n", worst(cal[0], cal[1]), worst(qg[0][0], qg[0][1]),
                    worst(qg[1][0], qg[1][1]));
        close(qg[0][0], qg[0][1], "silu_mul's quick_gelu against the CPU");
        close(qg[1][0], qg[1][1], "silu_mul's quick_gelu at limit 7 against the CPU");
    }
    // The rope with and without a norm, positions out of order, on unit tables and on tables YaRN's attention factor scales (1.2).
    for (const float scale : {1.0f, 1.2f})
    for (bool weighted : {true, false}) {
        const size_t rows = 4, heads = 3, head_dim = 128, half = 64, positions = 50;
        const std::vector<float> x = row_classes::values(rows * heads * head_dim, 8), w = row_classes::values(head_dim, 9);
        std::vector<float> cos(positions * half), sin(positions * half);
        for (size_t p = 0; p < positions; ++p)
            for (size_t j = 0; j < half; ++j) {
                const float a = (float)p * std::pow(10000.0f, -2.0f * (float)j / (float)head_dim);
                cos[p * half + j] = std::cos(a) * scale;
                sin[p * half + j] = std::sin(a) * scale;
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
                    // A per-row bias (a -vl file's image rows) runs the base's ops, so no fused route reads one bias for every row.
                    std::vector<float> bias(ne);
                    for (size_t e = 0; e < ne; ++e) bias[e] = 0.01f * float(e % 7);
                    const BufferPtr bb = g.alloc(ne * 4);
                    g.write(*bb, 0, bias.data(), ne * 4);
                    MoeStep strided = st[1];
                    strided.rule = backend::RouteRule::sqrt_softplus({bb.get(), 0}, 1.0f, true);
                    strided.rule.bias_stride = ne;
                    g.moe_step(strided, {hb.get(), 0}, {o[1].y.get(), 0}, nin, 1, {decode, 1}, Dtype::int8);
                    require(!rocm_take_launches(g).count("llmx_moe_step_route"), "a moe_step with a per-row bias runs the base's ops, " + at);
                }
    }
}


// delta_step's times, printed, never asserted: the base's ops against the fused step at Qwen3.8-27B's linear layer shards, Q8_0 throughout, one view a generated row as streams decoding together give them, from weights outside the L2 cache.
// Each arm prints its median us a step, its launches and whether every slot it writes matches the base's bits.
void delta_rates(Backend& g) {
    struct Shape { const char* name; size_t hk, hv; };
    const Shape shapes[] = {{"1x4", 4, 12}, {"2x2", 8, 24}, {"1x1", 16, 48}};
    const size_t E = 5120, slots = 8;
    for (const Shape& sp : shapes) {
        const StateShape sh{sp.hk, sp.hv, 128, 128};
        const size_t C = sh.channels(), V = sp.hv * 128;
        struct Set { BufferPtr qkv, gate, wa, wb; };
        std::vector<Set> sets;
        for (uint32_t k = 0; k < 2; ++k) {
            const auto mk = [&](size_t rows, uint32_t seed) { const std::vector<uint8_t> m = q8_matrix(E, rows, seed + 10 * k); return g.adopt(m.data(), m.size()); };
            sets.push_back({mk(C, 401), mk(V, 402), mk(sp.hv, 403), mk(sp.hv, 404)});
        }
        std::vector<float> a = row_classes::values(sp.hv, 406, 1.0f), dt = row_classes::values(sp.hv, 407, 0.5f), nw = row_classes::values(128, 408, 1.0f),
                           cw = row_classes::values(C * kConvTaps, 409, 0.5f), state = row_classes::values(slots * sh.slot_floats(), 410, 0.1f);
        for (float& v : a) v = -std::fabs(v) - 0.05f;
        auto ap = g.adopt(a.data(), a.size() * 4), db = g.adopt(dt.data(), dt.size() * 4), nb = g.adopt(nw.data(), nw.size() * 4), cb = g.adopt(cw.data(), cw.size() * 4);
        // Rows 1 and 4 as one-row views of streams decoding together, and rows 4 as one view, a verify of three drafts.
        for (int lay = 0; lay < 3; ++lay)
            for (Dtype dt_ : {Dtype::int8, Dtype::f16}) {
                const size_t rows = lay ? 4 : 1;
                const bool one_view = lay == 2;
                const std::vector<float> mixed = row_classes::values(rows * E, 411 + (uint32_t)rows, 1.0f);
                auto mb = g.adopt(mixed.data(), mixed.size() * 4);
                std::vector<RowRun> runs;
                for (size_t r = 0; r < rows; ++r) runs.push_back(RowRun{r + 1, 1});
                struct Arm { std::unique_ptr<StateStorage> st; BufferPtr raw, conv, gate, ab, out; };
                Arm arm[2];
                std::vector<float> got[2];
                std::string ran[2];
                double us[2], spread[2];
                for (int k = 0; k < 2; ++k) {
                    arm[k] = Arm{g.state_alloc(1, slots, sh), g.alloc(rows * C * 4), g.alloc(rows * C * 4), g.alloc(rows * V * 4), g.alloc(2 * rows * sp.hv * 4), g.alloc(rows * V * 4)};
                    g.write(arm[k].st->layer(0), 0, state.data(), state.size() * 4);
                    std::vector<StateView> views;
                    if (one_view) views.push_back(StateView{arm[k].st.get(), 0, 1, 100, rows});
                    else
                        for (size_t r = 0; r < rows; ++r) views.push_back(StateView{arm[k].st.get(), 2 * r, 2 * r + 1, 100, 1});
                    const auto step = [&](const Set& w) {
                        return DeltaStep{Projection{{quant::GGML_TYPE_Q8_0, {w.qkv.get(), 0}}, {arm[k].raw.get(), 0}, C}, Projection{{quant::GGML_TYPE_Q8_0, {w.gate.get(), 0}}, {arm[k].gate.get(), 0}, V},
                                         Projection{{quant::GGML_TYPE_Q8_0, {w.wa.get(), 0}}, {arm[k].ab.get(), 0}, sp.hv},
                                         Projection{{quant::GGML_TYPE_Q8_0, {w.wb.get(), 0}}, {arm[k].ab.get(), rows * sp.hv}, sp.hv},
                                         {cb.get(), 0}, {ap.get(), 0}, {db.get(), 0}, {nb.get(), 0}, {arm[k].conv.get(), 0}, {arm[k].out.get(), 0}, 1e-6f, NormGate::silu};
                    };
                    const RowRuns rr{runs.data(), runs.size()};
                    size_t turn = 0;
                    const auto call = [&] {
                        const DeltaStep d = step(sets[turn++ % sets.size()]);
                        if (k == 0) g.Backend::delta_step(d, {mb.get(), 0}, E, rows, 0, views.data(), views.size(), rr, dt_);
                        else g.delta_step(d, {mb.get(), 0}, E, rows, 0, views.data(), views.size(), rr, dt_);
                    };
                    rocm_take_launches(g);
                    call();
                    g.sync();
                    for (const auto& [kn, n] : rocm_take_launches(g)) ran[k] += (ran[k].empty() ? "" : "+") + kn + (n > 1 ? "x" + std::to_string(n) : "");
                    for (const BufferPtr* b : {&arm[k].raw, &arm[k].gate, &arm[k].out}) {
                        const std::vector<float> v = read(g, **b, (*b == arm[k].raw ? C : V) * rows);
                        got[k].insert(got[k].end(), v.begin(), v.end());
                    }
                    const std::vector<float> abv = read(g, *arm[k].ab, 2 * rows * sp.hv), stv = read(g, arm[k].st->layer(0), state.size());
                    got[k].insert(got[k].end(), abv.begin(), abv.end());
                    got[k].insert(got[k].end(), stv.begin(), stv.end());
                    const auto timed = median_us(g, call);
                    us[k] = timed.first, spread[k] = timed.second;
                    rocm_take_launches(g);
                }
                size_t differ = 0;
                for (size_t i = 0; i < got[0].size(); ++i) differ += std::memcmp(&got[0][i], &got[1][i], 4) != 0;
                std::printf("backend-rocm: delta rate Qwen3.8-27B %s Q8_0 %s rows %zu%s: base %.1f us (spread %.1f) [%s], fused %.1f us (spread %.1f) [%s], %zu of %zu floats differ\n", sp.name,
                            dtype_name(dt_), rows, one_view ? " one view" : "", us[0], spread[0], ran[0].c_str(), us[1], spread[1], ran[1].c_str(), differ, got[0].size());
            }
    }
}

// moe_step of several generated rows in the int8 class against the base's ops (Backend::moe_step) bit for bit, with the launches that show which ran.
// The fused step takes the rows while the base's routed products would not group their entries, rows * k below the gate type's int8 break-even of 1.25 entries an expert (backend_core.hpp, kQ4KTwins and kQ5KTwins).
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
    // Widths of a multiple of 1280 mix under the entry rule (hc_entry_rows), so the widest shapes here sit just beside it.
    const Shape shapes[] = {{4, 2304, 320, true}, {4, 2304, 320, false}, {2, 512, 64, true}, {2, 256, 64, true}};
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

// Attention times at heads 128 and 256 wide over an f16 cache, generated rows after a history and prompts on the tile; checks nothing.
void attention_rates(Backend& g) {
    struct Case { int n_head, n_head_kv, head_dim; size_t hist, rows, extent; };
    std::cout << "backend-rocm: attention rates; heads, KV heads, width, history, rows, extent: us a call, kernels\n";
    for (const Case cs : {Case{16, 2, 128, 4096, 1, 1}, Case{16, 2, 128, 4096, 16, 1}, Case{16, 2, 128, 16000, 1, 1}, Case{16, 2, 128, 0, 512, 512}, Case{16, 2, 128, 4096, 512, 512},
                          Case{16, 8, 128, 0, 512, 512}, Case{16, 8, 128, 16384, 512, 512}, Case{16, 16, 128, 16384, 512, 512}, Case{24, 2, 256, 16384, 1, 1},
                          Case{24, 2, 256, 16384, 4, 1}, Case{24, 2, 256, 16384, 16, 16}, Case{24, 2, 256, 16384, 64, 64}, Case{24, 2, 256, 16384, 512, 512},
                          Case{24, 4, 256, 0, 512, 512}, Case{24, 4, 256, 16384, 512, 512}, Case{24, 8, 256, 16384, 512, 512}}) {
        const int n_head = cs.n_head, n_head_kv = cs.n_head_kv, head_dim = cs.head_dim;
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
        std::printf("backend-rocm: attention rate %d, %d, %d, %zu, %zu, %zu: %.1f us, %s\n", n_head, n_head_kv, head_dim, cs.hist, cs.rows, cs.extent, us, ran.c_str());
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

// The scratch a fit prices holds every pass of its widths (Backend::reserve_scratch, docs/src/backends-gpu.md, Scratch), on a synthetic Qwen3 and on a GGUF `file` when given.
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

// Gated attention, which writes no twin of the ungated output, gives the output and the product of attention then sigmoid_mul bit for bit, launching the same kernels, the gate read in place from a q-and-gate row as qwen35 lays it.
void gated_attention_bits(Backend& g, Dtype dt) {
    const int n_head = 8, n_head_kv = 2, head_dim = 256;
    const size_t D = head_dim, qw = (size_t)n_head * D, kvw = (size_t)n_head_kv * D, nout = 40;
    const std::vector<uint8_t> m = q8_matrix(qw, nout, 71);
    auto mb = g.adopt(m.data(), m.size());
    const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
    // Producers write the twin class the last product read, so one product in dt first starts both arms of the first case in that class.
    {
        const std::vector<float> x = row_classes::values(qw, 70);
        const RowRun one[1] = {{1, 1}};
        auto xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * 4);
        g.matmul(wv, {xb.get(), 0}, {yb.get(), 0}, qw, nout, 1, {one, 1}, dt);
        read(g, *yb, nout);
    }
    struct Case { size_t hist, rows, extent; };
    for (const Case cs : {Case{10, 1, 1}, Case{200, 1, 1}, Case{200, 3, 1}, Case{70, 5, 5}, Case{70, 40, 40}}) {
        const size_t rows = cs.rows, all = cs.hist + rows;
        const std::vector<float> keys = row_classes::values(all * kvw, 72), vals = row_classes::values(all * kvw, 73), q = row_classes::values(rows * qw, 74, 4.0f),
                                 r = row_classes::values(rows * 2 * qw, 75, 6.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4), rb = g.adopt(r.data(), r.size() * 4);
        auto st = g.kv_alloc(1, n_head_kv, D, 512, KVType::f16, KVType::f16);
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
        const CSlice gate{rb.get(), D};
        std::vector<float> outs[2], ys[2];
        std::map<std::string, size_t> ran[2];
        for (int arm = 0; arm < 2; ++arm) {
            auto ob = g.alloc(rows * qw * 4), y = g.alloc(nout * rows * 4);
            rocm_take_launches(g);
            if (arm == 0) {
                g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                g.sigmoid_mul({ob.get(), 0}, {ob.get(), 0}, gate, rows, n_head, D, 2 * qw, 2 * D, {run, 1});
            } else {
                g.gated_attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim, {gate, rows, 2 * qw, 2 * D, {run, 1}});
            }
            g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, qw, nout, rows, {run, 1}, dt);
            ran[arm] = rocm_take_launches(g);
            outs[arm] = read(g, *ob, rows * qw);
            ys[arm] = read(g, *y, nout * rows);
        }
        const std::string at = std::string(dt == Dtype::int8 ? "8-bit" : "16-bit") + ", history " + std::to_string(cs.hist) + ", rows " + std::to_string(rows);
        require(same(outs[0], outs[1]) && same(ys[0], ys[1]), at + ": gated attention gives the output and product of attention then sigmoid_mul");
        require(ran[0] == ran[1], at + ": gated attention launches the kernels of attention then sigmoid_mul");
        s.abort();
    }
}

// Every pair of extents of one class gives the same bits, through a matmul and through attention after a history (row_classes.hpp's shapes).
// Prompts shorter than the profile's prompt_rows_below take the decode row builds with the one-column call's bits, and from it the tile, a run-less call by its rows' count (cases in docs/TESTS.md, backend-rocm).
void prompt_crossover(Backend& g) {
    DeviceProfile& profile = rocm_profile(g);
    const size_t keep = profile.prompt_rows_below, from = 9, nout = 300;
    profile.prompt_rows_below = 2;
    require(g.row_class(2) != g.row_class(1) && g.row_class(8) == g.row_class(9), "at prompt_rows_below 2 a prompt of 2 to 9 rows is one tile class");
    profile.prompt_rows_below = from;
    require(g.row_class(from - 1) != g.row_class(from) && g.row_class(from - 1) == g.row_class(2) && g.row_class(from - 1) != g.row_class(1), "the crossover's two sides are classes of their own, never a generated token's");
    const auto tiles = [](const std::map<std::string, size_t>& ran) {
        size_t n = 0;
        for (const auto& [k, v] : ran) n += k.rfind("llmx_q8_0_tile", 0) == 0 ? v : 0;
        return n;
    };
    for (Dtype dt : {Dtype::int8, Dtype::f16})
        for (size_t nin : {size_t(1024), size_t(1056)}) {
            const std::string what = std::string(dt == Dtype::int8 ? "int8" : "f16") + ", rows " + std::to_string(nin) + " wide";
            const size_t cols = 14 + (from - 1) + from;
            const std::vector<uint8_t> w = q8_matrix(nin, nout, 61);
            const std::vector<float> x = row_classes::values(cols * nin, 62);
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(nout * cols * 4), y1 = g.alloc(nout * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {wb.get(), 0}};
            std::vector<float> one(nout * cols);
            for (size_t c = 0; c < cols; ++c) {
                const RowRun rr{1, 1};
                g.matmul(wv, {xb.get(), c * nin}, {y1.get(), 0}, nin, nout, 1, {&rr, 1}, dt);
                const std::vector<float> r = read(g, *y1, nout);
                std::copy(r.begin(), r.end(), one.begin() + std::ptrdiff_t(c * nout));
            }
            // 14 decode columns, a prompt of from - 1 rows across the chunk edge, then a prompt of `from` rows on the tile.
            const RowRun runs[3] = {{14, 1}, {14 + from - 1, from - 1}, {cols, from}};
            rocm_take_launches(g);
            g.matmul(wv, {xb.get(), 0}, {y.get(), 0}, nin, nout, cols, {runs, 3}, dt);
            const auto ran = rocm_take_launches(g);
            const std::vector<float> mixed = read(g, *y, nout * cols);
            const size_t short_end = 14 + from - 1;
            require(std::memcmp(mixed.data(), one.data(), nout * short_end * 4) == 0, what + ": decode columns and a short prompt across a chunk edge have the one-column bits");
            const RowRun alone{from, from};
            g.matmul(wv, {xb.get(), short_end * nin}, {y1.get(), 0}, nin, nout, from, {&alone, 1}, dt);
            require(std::memcmp(read(g, *y1, nout * from).data(), mixed.data() + short_end * nout, nout * from * 4) == 0, what + ": a prompt at the crossover in a mixed call has its bits alone");
            require(tiles(ran) == 1, what + ": a mixed call launches the tile once, for the prompt at the crossover");
            // Run-less calls: below the crossover each row a column, from it the tile with the bits it gives at prompt_rows_below 2.
            for (size_t rows = 2; rows <= 16; ++rows) {
                rocm_take_launches(g);
                g.matmul(wv, {xb.get(), 0}, {y.get(), 0}, nin, nout, rows, {}, dt);
                const auto r = rocm_take_launches(g);
                const std::string at = what + ", a run-less call of " + std::to_string(rows) + " rows";
                if (rows < from) {
                    require(tiles(r) == 0 && std::memcmp(read(g, *y, nout * rows).data(), one.data(), nout * rows * 4) == 0, at + " takes the row builds with the one-column bits");
                } else {
                    const std::vector<float> now = read(g, *y, nout * rows);
                    profile.prompt_rows_below = 2;
                    g.matmul(wv, {xb.get(), 0}, {y.get(), 0}, nin, nout, rows, {}, dt);
                    profile.prompt_rows_below = from;
                    require(tiles(r) == 1 && std::memcmp(now.data(), read(g, *y, nout * rows).data(), nout * rows * 4) == 0, at + " takes the tile with the bits it has at prompt_rows_below 2");
                }
            }
            // A norm's twin read by a short prompt's product: once a product has read one, the norm writes it and the product makes none.
            const std::vector<float> nw = row_classes::values(nin, 63);
            auto nb = g.adopt(nw.data(), nw.size() * 4), ob = g.alloc(5 * nin * 4);
            const RowRun five{5, 5};
            size_t twins = 0;
            for (int pass = 0; pass < 2; ++pass) {
                g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {nb.get(), 0}, 5, nin, nin, 1e-6f, {&five, 1});
                rocm_take_launches(g);
                g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, nin, nout, 5, {&five, 1}, dt);
                twins = 0;
                for (const auto& [k, v] : rocm_take_launches(g)) twins += k.rfind("llmx_twin", 0) == 0 ? v : 0;
            }
            require(twins == 0, what + ": a short prompt's product reads its norm's twin without making one");
        }
    // Products of two rows at the extents about the crossover: one class, one set of bits.
    std::vector<std::vector<float>> out;
    std::vector<size_t> extents;
    const std::vector<uint8_t> w = q8_matrix(1024, 64, 64);
    const std::vector<float> x = row_classes::values(2 * 1024, 65);
    auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(64 * 2 * 4);
    for (size_t e = 2; e <= from + 2; ++e) {
        const RowRun rr{2, e};
        g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, 1024, 64, 2, {&rr, 1}, Dtype::int8);
        out.push_back(read(g, *y, 64 * 2));
    }
    bool ok = true;
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = i + 1; j < out.size(); ++j)
            if (g.row_class(i + 2) == g.row_class(j + 2)) ok = ok && same(out[i], out[j]);
    require(ok, "extents of one class about the crossover give one set of bits");
    profile.prompt_rows_below = keep;
}

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

// The fast collective against the conservative one on the same cards, over partials of the armed bits, their quiet twin, NaN payloads, infinities, zeros and a denormal, every bit but a NaN's payload (docs/TESTS.md, backend-rocm).
// The forms follow one another by size, at tails and an unaligned residual, two sums in flight, and 1800 sums arm each parity half again hundreds of times.
void sentinel_sums() {
    HipFn& f = hip();
    std::vector<BackendPtr> d = rocm_devices(4);
    if (d.size() < 2) return;
    static const uint32_t bits[] = {0x7f800001u, 0x7fc00001u, 0xff800001u, 0xffc00001u, 0x7fc00000u, 0x7f812345u, 0x7fd12345u, 0x7f800000u, 0xff800000u, 0x80000000u, 0x0u, 0x00000001u};
    const size_t cap = 8 * 1031, sizes[] = {1, 3, 5, 63, 64, 97, 255, 256, 1023, 1024, 1027, cap};
    // The fast arm's picks switch transport at each size, so a join that keeps both follows them with sums in flight across a switch.
    const std::map<size_t, GpuSumPick> forms = {{64, {GroupTransport::host_writes, GpuSumForm::shares}}, {256, {GroupTransport::peer_writes, GpuSumForm::chain}},
                                                {512, {GroupTransport::host_writes, GpuSumForm::chain}}, {1024, {GroupTransport::peer_writes, GpuSumForm::one_shot}},
                                                {1027, {GroupTransport::host_writes, GpuSumForm::pulls}}, {1028, {GroupTransport::host_writes, GpuSumForm::one_shot}}};
    for (size_t W = 2; W <= d.size(); ++W) {
        const std::vector<Backend*> group = raw(d, W);
        std::unique_ptr<Collective> fast = rocm_collective(group, cap, 1, forms);
        std::vector<BackendPtr> e = rocm_devices(W), h = rocm_devices(W);
        const auto can = f.hipDeviceCanAccessPeer;
        f.hipDeviceCanAccessPeer = no_peer;
        std::unique_ptr<Collective> host = h[0]->join(raw(h, W), cap, 1);
        f.hipDeviceCanAccessPeer = can;
        sum_calls(*host, W, true);
        std::unique_ptr<Collective> slow = rocm_collective(raw(e, W), cap, 1, GpuSumForm::one_shot, GroupTransport::pulls);
        sum_calls(*slow, W, false);
        constexpr size_t kArms = 3;   // peer writes, host writes, pulls; the last is the reference
        const std::vector<Backend*> sides[kArms] = {group, raw(h, W), raw(e, W)};
        Collective* arms[kArms] = {fast.get(), host.get(), slow.get()};
        std::vector<BufferPtr> x[kArms][2];   // per arm, per sum in flight, per member
        for (size_t a = 0; a < kArms; ++a)
            for (size_t s = 0; s < 2; ++s)
                for (size_t m = 0; m < W; ++m) x[a][s].push_back(sides[a][m]->alloc((cap + 7) * sizeof(float)));
        uint32_t seed = 7001 + (uint32_t)W;
        auto value = [&] {
            seed = seed * 1664525u + 1013904223u;
            const uint32_t r = seed >> 8;
            float v;
            if (r % 3) {
                std::memcpy(&v, &bits[r % (sizeof(bits) / sizeof(bits[0]))], sizeof(v));
            } else {
                v = float(int(r % 4001) - 2000) / 16.0f;
            }
            return v;
        };
        size_t one_shots = 0;
        for (size_t it = 0; it < 1800; ++it) {
            const size_t n = sizes[(it * 7 + it / 12) % (sizeof(sizes) / sizeof(sizes[0]))], offset = it % 2 ? 7 : 0;
            const bool in_place = it % 3 == 0, twice = !in_place && it % 10 == 1;
            auto it_form = forms.upper_bound(n);
            one_shots += (it_form == forms.begin() || std::prev(it_form)->second.form == GpuSumForm::one_shot) ? (twice ? 2 : 1) : 0;
            std::vector<std::vector<float>> partial[2];
            std::vector<float> start(n);
            for (size_t s = 0; s < (twice ? 2u : 1u); ++s) {
                partial[s].assign(W, std::vector<float>(n));
                for (size_t m = 0; m < W; ++m)
                    for (size_t i = 0; i < n; ++i) partial[s][m][i] = value();
            }
            for (size_t i = 0; i < n; ++i) start[i] = value();
            for (size_t a = 0; a < kArms; ++a) {
                Collective& c = *arms[a];
                const std::vector<Backend*>& g = sides[a];
                for (size_t s = 0; s < (twice ? 2u : 1u); ++s) {
                    std::vector<Slice> xs;
                    for (size_t m = 0; m < W; ++m) {
                        const Slice p = c.partial(m);
                        g[m]->write(*p.buffer, p.offset * sizeof(float), partial[s][m].data(), n * sizeof(float));
                        if (!in_place) g[m]->write(*x[a][s][m], offset * sizeof(float), start.data(), n * sizeof(float));
                        xs.push_back({x[a][s][m].get(), offset});
                    }
                    if (in_place) c.sum_in_place(n, 1);
                    else c.sum_into(xs, n, 1);
                }
            }
            for (size_t s = 0; s < (twice ? 2u : 1u); ++s)
                for (size_t m = 0; m < W; ++m) {
                    std::vector<float> got[kArms];
                    for (size_t a = 0; a < kArms; ++a) {
                        sides[a][m]->submit();
                        got[a] = in_place ? read(*sides[a][m], *arms[a]->partial(m).buffer, n) : read(*sides[a][m], *x[a][s][m], n, offset);
                    }
                    for (size_t a = 0; a + 1 < kArms; ++a) {
                        size_t bad = n;
                        for (size_t i = 0; i < n && bad == n; ++i) {
                            uint32_t u[2];
                            std::memcpy(&u[0], &got[a][i], 4);
                            std::memcpy(&u[1], &got[kArms - 1][i], 4);
                            if (u[0] != u[1] && !(std::isnan(got[a][i]) && std::isnan(got[kArms - 1][i]))) {
                                bad = i;
                                std::fprintf(stderr, "  float %zu of %zu: %s %08x, conservative %08x\n", i, n, a ? "host writes" : "peer writes", u[0], u[1]);
                            }
                        }
                        require(bad == n, "sum " + std::to_string(it) + " of " + std::to_string(n) + " floats over " + std::to_string(W) + " devices: member " + std::to_string(m) + "'s " +
                                              (a ? "host-written" : "peer-written") + " sum is not the conservative path's");
                    }
                }
        }
        sum_calls(*fast, W, true);
        sums_in_flight(group, *fast, 1, seed);
        std::cout << "backend-rocm: " << W << " devices, 1800 sums of armed bits, NaN payloads, infinities and zeros, " << one_shots << " in the one-shot, give the conservative path's bits by peer and by host writes\n";
    }
}

// The host golden of the hyper mix's entry rule (Backend::hyper_mix, docs/PRECISION.md) for one generated row, from the injected streams x.
// Its sums follow the rule's orders exactly; low and out pass through expf, whose last bit the host and the device may round apart, so the up product starts from the device's low.
struct EntryGolden {
    std::vector<float> rr, low, injg, out;
    std::vector<double> mag;   // per output, the sum of its stream terms' magnitudes
};
void entry_twin(const float* v, int8_t* q, float* d) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    if (amax == 0.0f) {
        std::fill(q, q + 32, int8_t(0));
        *d = 0.0f;
        return;
    }
    int e = 0;
    (void)std::frexp(amax, &e);
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = std::ldexp(amax, s);
    for (int i = 0; i < 32; ++i) {
        const float r = std::ldexp(v[i], s) * (127.0f / an);
        const float t = std::fmin(std::floor(std::fabs(r) + 0.5f), 127.0f);
        q[i] = (int8_t)(r < 0.0f ? -t : t);
    }
    *d = std::ldexp(an / 127.0f, -s);
}
// Blocks 2p and 2p + 1 of a Q8_0 row against the twin's: fmaf(d1 xd1, i1, (d0 xd0) i0).
float entry_pair(const uint8_t* w, const int8_t* q, const float* qd) {
    int s[2] = {0, 0};
    float d[2];
    for (int k = 0; k < 2; ++k) {
        uint16_t h;
        std::memcpy(&h, w + 34 * k, 2);
        d[k] = f16_to_f32(h);
        for (int i = 0; i < 32; ++i) s[k] += (int)(int8_t)w[34 * k + 2 + i] * (int)q[32 * k + i];
    }
    return std::fma(d[1] * qd[1], (float)s[1], (d[0] * qd[0]) * (float)s[0]);
}
// The pairwise tree of n values, n a power of two: a[i] += a[i + s] for s = n / 2 down to 1.
float entry_tree(std::vector<float> a) {
    for (size_t s = a.size() / 2; s; s >>= 1)
        for (size_t i = 0; i < s; ++i) a[i] += a[i + s];
    return a[0];
}
EntryGolden entry_golden(const std::vector<float>& x, const std::vector<float>& nw, const std::vector<uint8_t>& down, const std::vector<uint8_t>& up, const std::vector<float>* inj,
                         const std::vector<float>& dev_low, size_t H, size_t E, size_t L, float eps) {
    const size_t G = 1280, KG = E / G, R = H * E, drow = R / 32 * 34, urow = L / 32 * 34;
    EntryGolden o;
    std::vector<float> pss(H * KG), pdown(H * KG * L), pinj(H * KG * H);
    for (size_t kg = 0; kg < H * KG; ++kg) {
        const size_t gb = kg * G;
        std::vector<float> y(G), qd(G / 32), ss(256, 0.0f);
        std::vector<int8_t> q(G);
        for (size_t i = 0; i < G; ++i) y[i] = x[gb + i] * nw[gb + i];
        for (size_t b = 0; b < G / 32; ++b) entry_twin(&y[b * 32], &q[b * 32], &qd[b]);
        for (size_t k = 0; k < G / 256; ++k)
            for (size_t t = 0; t < 256; ++t) ss[t] = std::fma(x[gb + t + k * 256], x[gb + t + k * 256], ss[t]);
        pss[kg] = entry_tree(ss);
        for (size_t r = 0; inj && r < H; ++r) {
            std::vector<float> ip(256, 0.0f);
            for (size_t k = 0; k < G / 256; ++k)
                for (size_t t = 0; t < 256; ++t) ip[t] = std::fma((*inj)[r * R + gb + t + k * 256], y[t + k * 256], ip[t]);
            pinj[kg * H + r] = entry_tree(ip);
        }
        for (size_t row = 0; row < L; ++row) {
            std::vector<float> v(32, 0.0f);
            for (size_t p = 0; p < G / 64; ++p) v[p] = entry_pair(&down[row * drow + (gb / 32 + 2 * p) * 34], &q[64 * p], &qd[2 * p]);
            pdown[kg * L + row] = entry_tree(v);
        }
    }
    const auto across = [&](const float* part, size_t stride, size_t at) {
        float v = 0.0f;
        for (size_t h = 0; h < H; ++h) {
            float ts = part[(h * KG) * stride + at];
            for (size_t g = 1; g < KG; ++g) ts = ts + part[(h * KG + g) * stride + at];
            v = h ? std::fma(o.rr[h], ts, v) : o.rr[h] * ts;
        }
        return v;
    };
    for (size_t h = 0; h < H; ++h) {
        float sum = pss[h * KG];
        for (size_t g = 1; g < KG; ++g) sum = sum + pss[h * KG + g];
        o.rr.push_back(1.0f / std::sqrt(sum / (float)E + eps));
    }
    for (size_t r = 0; inj && r < H; ++r) o.injg.push_back(across(pinj.data(), H, r));
    for (size_t row = 0; row < L; ++row) {
        const float v = across(pdown.data(), L, row) * (1.0f / float(H));
        o.low.push_back(v / (1.0f + std::exp(-v)));
    }
    std::vector<int8_t> uq(L);
    std::vector<float> ud(L / 32);
    for (size_t b = 0; b < L / 32; ++b) entry_twin(&dev_low[b * 32], &uq[b * 32], &ud[b]);
    for (size_t d = 0; d < E; ++d) {
        float sum = 0.0f;
        double mag = 0.0;
        for (size_t h = 0; h < H; ++h) {
            const uint8_t* w = &up[(h * E + d) * urow];
            float g = entry_pair(w, &uq[0], &ud[0]);
            for (size_t p = 1; p < L / 64; ++p) g += entry_pair(w + p * 68, &uq[64 * p], &ud[2 * p]);
            const float c = x[h * E + d] * (o.rr[h] * nw[h * E + d]), s = 1.0f / (1.0f + std::exp(-g));
            sum = std::fma(c, s, sum);
            mag += std::fabs((double)c * s);
        }
        o.out.push_back(sum / (float)H);
        o.mag.push_back(mag);
    }
    return o;
}
// The largest distance in units in the last place between two runs of floats of one sign.
uint32_t max_ulp(const std::vector<float>& a, const std::vector<float>& b) {
    uint32_t m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        int32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        m = std::max<uint32_t>(m, (uint32_t)std::abs((long long)x - (long long)y));
    }
    return m;
}

// Batch invariance of the entry rule: 1 to 9 generated rows injected and mixed in one call give each row the bytes it gets alone as a decode row, one llmx_hc_entry launch.
// Rows past one run llmx_hc_entry8, up to 8 rows a launch, so 9 rows take two launches.
void hc_entry_rows(Backend& g, bool with_inj) {
    const size_t H = 4, E = 2560, L = 320, R = H * E, kMax = 9;
    const float eps = 1e-6f;
    const std::vector<uint8_t> down = q8_matrix(R, L, 171), up = q8_matrix(L, R, 172);
    const std::vector<float> nw = row_classes::values(R, 173), inj = row_classes::values(H * R, 174), xs = row_classes::values(kMax * R, 175), blks = row_classes::values(kMax * E, 176),
                             gins = row_classes::values(kMax * H, 177);
    auto db = g.adopt(down.data(), down.size()), ub = g.adopt(up.data(), up.size()), wb = g.adopt(nw.data(), nw.size() * 4), ib = g.adopt(inj.data(), inj.size() * 4);
    struct Run {
        std::vector<float> x, out, low, injg;
        std::map<std::string, size_t> ran;
    };
    // Rows first to first + n, injected and mixed as one call, n generated rows unless runs are given.
    const auto run = [&](size_t first, size_t n, std::vector<RowRun> runs = {}) {
        auto xb = g.alloc(n * R * 4), bb = g.alloc(n * E * 4), gb = g.alloc(n * H * 4), out = g.alloc(n * E * 4), normed = g.alloc(n * R * 4), low = g.alloc(n * L * 4),
             gates = g.alloc(n * R * 4), injg = g.alloc(n * H * 4);
        g.write(*xb, 0, xs.data() + first * R, n * R * 4);
        g.write(*bb, 0, blks.data() + first * E, n * E * 4);
        g.write(*gb, 0, gins.data() + first * H, n * H * 4);
        g.sync();
        const Backend::HyperMix m{{wb.get(), 0}, {quant::GGML_TYPE_Q8_0, {db.get(), 0}}, {quant::GGML_TYPE_Q8_0, {ub.get(), 0}},
                                  with_inj ? WeightView{quant::GGML_TYPE_F32, {ib.get(), 0}} : WeightView{}, {normed.get(), 0}, {low.get(), 0}, {gates.get(), 0}, {injg.get(), 0}};
        std::vector<RowRun> gen(n);
        for (size_t r = 0; r < n; ++r) gen[r] = {r + 1, 1};
        if (!runs.empty()) gen = runs;
        rocm_take_launches(g);
        g.inject_streams({xb.get(), 0}, {bb.get(), 0}, {gb.get(), 0}, n, H, E);
        g.hyper_mix({out.get(), 0}, {xb.get(), 0}, m, n, H, E, L, eps, {gen.data(), gen.size()}, Dtype::int8);
        Run r{read(g, *xb, n * R), read(g, *out, n * E), read(g, *low, n * L), read(g, *injg, n * H), {}};
        r.ran = rocm_take_launches(g);
        return r;
    };
    const auto part = [](const std::vector<float>& v, size_t at, size_t n) { return std::vector<float>(v.begin() + at * n, v.begin() + (at + 1) * n); };
    std::vector<Run> alone;
    for (size_t r = 0; r < kMax; ++r) alone.push_back(run(r, 1));
    for (size_t r = 0; r < kMax; ++r)
        require(alone[r].ran.count("llmx_hc_entry") && alone[r].ran.at("llmx_hc_entry") == 1, "a decode row's inject and mix are one llmx_hc_entry launch, row " + std::to_string(r));
    for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(4), size_t(8), size_t(9)}) {
        const Run b = run(0, n);
        bool exact = true;
        for (size_t r = 0; r < n; ++r)
            exact = exact && same(part(b.x, r, R), alone[r].x) && same(part(b.out, r, E), alone[r].out) && same(part(b.low, r, L), alone[r].low) &&
                    same(part(b.injg, r, H), alone[r].injg);
        const std::string at = std::to_string(n) + " generated rows" + (with_inj ? " with inject gates" : "");
        require(exact, at + " in one call give each row its decode bytes");
        const char* k = n == 1 ? "llmx_hc_entry" : "llmx_hc_entry8";
        require(b.ran.count(k) && b.ran.at(k) == (n + 7) / 8 && !b.ran.count("llmx_hc_mix") && !b.ran.count("llmx_hc_mix8"), at + " mix as one llmx_hc_entry for a row, else llmx_hc_entry8 launches of 8 rows at most");
    }
    // Generated rows beside another request's prompt run of 3 rows, as a server pass batches them, keep their bytes.
    const Run mixed = run(0, 5, {{1, 1}, {4, 3}, {5, 1}});
    require(same(part(mixed.out, 0, E), alone[0].out) && same(part(mixed.low, 0, L), alone[0].low) && same(part(mixed.out, 4, E), alone[4].out) &&
                same(part(mixed.low, 4, L), alone[4].low) && same(part(mixed.injg, 4, H), alone[4].injg),
            std::string("generated rows beside a prompt run keep their decode bytes") + (with_inj ? " with inject gates" : ""));
    std::cout << "backend-rocm: 1, 2, 3, 4, 8 and 9 generated rows in one call" << (with_inj ? " with inject gates" : "") << " give each row the bytes llmx_hc_entry gives it alone, and beside a prompt run too\n";
}

// A launch that issues a held inject at the chunk keeps its own arguments: the inject's flush must not leave it reading the bytes the launches after it write.
// Seven adds and the inject fill an idle device's first chunk of 8, so the inject goes from inside the next add's hold and that add is the first of the next chunk.
void inject_at_chunk() {
    BackendPtr b = make_rocm_backend(0);
    rocm_profile(*b).dispatch_chunk = 64;
    const size_t n = 1024, E = 1280;
    const std::vector<float> one(n, 1.0f), zero(E, 0.0f);
    auto ones = b->alloc(n * 4), pad = b->alloc(n * 4), x = b->alloc(E * 4), blk = b->alloc(E * 4), gate = b->alloc(64);
    std::vector<BufferPtr> ys;
    b->write(*ones, 0, one.data(), n * 4);
    b->write(*pad, 0, one.data(), n * 4);
    for (int i = 0; i < 9; ++i) {
        ys.push_back(b->alloc(n * 4));
        const std::vector<float> v(n, (float)(i + 1));
        b->write(*ys.back(), 0, v.data(), n * 4);
    }
    b->write(*x, 0, zero.data(), E * 4);
    b->write(*blk, 0, zero.data(), E * 4);
    b->write(*gate, 0, zero.data(), 4);
    b->sync();
    for (int i = 0; i < 7; ++i) b->add({pad.get(), 0}, {ones.get(), 0}, n);
    b->inject_streams({x.get(), 0}, {blk.get(), 0}, {gate.get(), 0}, 1, 1, E);
    for (int i = 0; i < 9; ++i) b->add({ys[i].get(), 0}, {ones.get(), 0}, n);
    b->wait(b->submit());
    bool exact = read(*b, *pad, n) == std::vector<float>(n, 8.0f);
    std::string got;
    for (int i = 0; i < 9; ++i) {
        const std::vector<float> y = read(*b, *ys[i], n);
        exact = exact && y == std::vector<float>(n, (float)(i + 2));
        got += " " + std::to_string((int)y[0]);
    }
    require(exact, "the adds around a held inject issued at the chunk gave other sums (want 2 to 10, got" + got + ")");
    std::cout << "backend-rocm: a launch that issues a held inject at the chunk keeps its own arguments\n";
}

// llmx_hc_entry, a generated row's inject and mix as one launch, at widths 1 and 2 against the host golden of the entry rule, with the flush paths between them giving the entry's bytes.
// A refused entry launch throws and turns its group conservative, and a sum on members no entry mix reached stays one llmx_group_sum launch.
void hc_entry() {
    inject_at_chunk();
    real_launch = hip().hipModuleLaunchKernel;
    std::vector<BackendPtr> d = rocm_devices(2);
    const size_t H = 4, E = 2560, L = 320, R = H * E;
    const float eps = 1e-6f;
    const std::vector<uint8_t> down = q8_matrix(R, L, 161), up = q8_matrix(L, R, 162);
    const std::vector<float> x0 = row_classes::values(R, 163), nw = row_classes::values(R, 164), inj = row_classes::values(H * R, 165), p0 = row_classes::values(E, 166),
                             p1 = row_classes::values(E, 167), gin = row_classes::values(H, 168);
    std::vector<float> blk(E);
    for (size_t i = 0; i < E; ++i) blk[i] = p0[i] + p1[i];
    struct Run {
        std::vector<float> x, out, low, injg;
        std::map<std::string, size_t> ran;
    };
    // A member's operands, made and written before any sum, since a write issues a pending join.
    struct Ops {
        BufferPtr db, ub, wb, ib, xb, gb, out, normed, low, gates, injg;
    };
    const auto prepare = [&](Backend& g) {
        auto db = g.adopt(down.data(), down.size()), ub = g.adopt(up.data(), up.size()), wb = g.adopt(nw.data(), nw.size() * 4), ib = g.adopt(inj.data(), inj.size() * 4);
        auto xb = g.alloc(R * 4), gb = g.alloc(H * 4), out = g.alloc(E * 4), normed = g.alloc(R * 4), low = g.alloc(L * 4), gates = g.alloc(R * 4), injg = g.alloc(H * 4);
        g.write(*xb, 0, x0.data(), R * 4);
        g.write(*gb, 0, gin.data(), H * 4);
        g.sync();
        return Ops{db, ub, wb, ib, xb, gb, out, normed, low, gates, injg};
    };
    // One member's inject and mix; `between` runs on the backend after the inject.
    const auto go = [&](Backend& g, Ops& o, CSlice block, const std::function<void(Backend&)>& between, bool chain) {
        const Backend::HyperMix m{{o.wb.get(), 0}, {quant::GGML_TYPE_Q8_0, {o.db.get(), 0}}, {quant::GGML_TYPE_Q8_0, {o.ub.get(), 0}},
                                  WeightView{quant::GGML_TYPE_F32, {o.ib.get(), 0}}, {o.normed.get(), 0}, {o.low.get(), 0}, {o.gates.get(), 0}, {o.injg.get(), 0}};
        const RowRun gen[1] = {{1, 1}};
        rocm_take_launches(g);
        g.inject_streams({o.xb.get(), 0}, block, {o.gb.get(), 0}, 1, H, E);
        if (between) between(g);
        if (chain) g.Backend::hyper_mix({o.out.get(), 0}, {o.xb.get(), 0}, m, 1, H, E, L, eps, {gen, 1}, Dtype::int8);
        else g.hyper_mix({o.out.get(), 0}, {o.xb.get(), 0}, m, 1, H, E, L, eps, {gen, 1}, Dtype::int8);
        Run r{read(g, *o.xb, R), read(g, *o.out, E), read(g, *o.low, L), read(g, *o.injg, H), {}};
        r.ran = rocm_take_launches(g);
        return r;
    };
    const auto run = [&](Backend& g, CSlice block, const std::function<void(Backend&)>& between, bool chain) {
        Ops o = prepare(g);
        return go(g, o, block, between, chain);
    };
    const auto launched = [](const Run& r, const char* k) { return r.ran.count(k) ? r.ran.at(k) : size_t(0); };
    Backend& g = *d[0];
    auto bb = g.adopt(blk.data(), E * 4);
    const Run chain = run(g, {bb.get(), 0}, nullptr, true), one = run(g, {bb.get(), 0}, nullptr, false);
    require(launched(one, "llmx_hc_entry") == 1 && one.ran.size() == 1, "a generated row's inject and mix are one llmx_hc_entry launch at width 1");
    require(same(one.x, chain.x), "llmx_hc_entry's injected streams are inject_streams' bits");
    const EntryGolden want = entry_golden(one.x, nw, down, up, &inj, one.low, H, E, L, eps);
    require(same(one.injg, want.injg), "llmx_hc_entry's inject gates are the entry rule's bits");
    // The device's expf is within 1 ulp (HIP's math table) and the host's within 0.502, so e = expf(-v) differs by at most 1 ulp, 1 + e by 1 ulp and each rounded quotient by 2^-22 of itself.
    // The output adds that per stream over H fmaf steps, each a rounding of at most 2^-23 of the terms' magnitudes apart, then divides by H (exact for a power of two, another 2^-23 otherwise).
    const double kLow = 0x1p-22 * (1 + 0x1p-20), kOut = (2.0 + H) * 0x1p-23 * (1 + 0x1p-20) / H, tiny = 0x1p-148;
    double hl = 0.0, ho = 0.0;
    bool inside = one.low.size() == L && one.out.size() == E;
    for (size_t i = 0; inside && i < L; ++i) {
        const double bound = kLow * std::fabs(want.low[i]) + tiny, e = std::fabs((double)one.low[i] - want.low[i]);
        inside = e <= bound;
        hl = std::max(hl, e / bound);
    }
    for (size_t i = 0; inside && i < E; ++i) {
        const double bound = kOut * want.mag[i] + ((H & (H - 1)) ? 0x1p-23 * std::fabs(want.out[i]) : 0.0) + tiny, e = std::fabs((double)one.out[i] - want.out[i]);
        inside = e <= bound;
        ho = std::max(ho, e / bound);
    }
    const uint32_t ul = max_ulp(one.low, want.low), uo = max_ulp(one.out, want.out);
    require(inside, "llmx_hc_entry's low and output are the entry rule's within the bounds expf's error gives");
    std::cout << "backend-rocm: llmx_hc_entry at width 1 is the entry rule: streams and inject gates exact, low within " << ul << " ulp (" << hl << " of its bound) and output within " << uo
              << " ulp (" << ho << " of its bound) of the host golden\n";
    hc_entry_rows(g, true);
    hc_entry_rows(g, false);
    // Any operation between the inject and the mix issues the inject first, and the mix of the injected row is llmx_hc_entry8's, with the entry's bytes.
    auto other = g.alloc(64 * 4), other2 = g.alloc(64 * 4), x2 = g.alloc(R * 4);
    const std::pair<const char*, std::function<void(Backend&)>> paths[] = {
        {"a launch", [&](Backend& b) { b.inject_streams({x2.get(), 0}, {bb.get(), 0}, {other.get(), 0}, 1, 1, 640); }},
        {"a read", [&](Backend& b) { (void)read(b, *other, 64); }},
        {"a copy", [&](Backend& b) { b.copy(*other2, 0, *other, 0, 64 * 4); }},
        {"a second inject", [&](Backend& b) { b.inject_streams({x2.get(), 0}, {bb.get(), 0}, {other.get(), 0}, 1, 4, 640); }}};
    for (const auto& [what, between] : paths) {
        const Run r = run(g, {bb.get(), 0}, between, false);
        require(launched(r, "llmx_hc_entry") == 0 && launched(r, "llmx_hc_entry8") == 1, std::string("llmx_hc_entry8 and no llmx_hc_entry with ") + what + " between the inject and the mix");
        require(same(r.x, one.x) && same(r.out, one.out) && same(r.low, one.low) && same(r.injg, one.injg), std::string("with ") + what + " between the inject and the mix, the entry's bytes");
    }
    std::cout << "backend-rocm: a launch, a read, a copy or a second inject between the inject and the mix gives llmx_hc_entry8 with the entry's bytes\n";
    if (d.size() < 2) {
        std::cout << "backend-rocm: one device, so no llmx_hc_entry at width 2\n";
        return;
    }
    // Width 2: the sum in place pushes, each member's entry takes its peer's partial; the bytes are width 1's with the summed block.
    std::unique_ptr<Collective> c = d[0]->join(raw(d, 2), 1, E);
    require(c != nullptr, "two ROCm devices joined no collective");
    if (!c) return;
    // Member 1 has run no mix, so the group's sum is main's one launch; member 1's first mix then takes the summed block as a width-1 entry.
    for (size_t m = 0; m < 2; ++m) d[m]->write(*c->partial(m).buffer, c->partial(m).offset * 4, (m ? p1 : p0).data(), E * 4);
    (void)rocm_take_launches(*d[0]);
    (void)rocm_take_launches(*d[1]);
    c->sum_in_place(1, E);
    for (size_t m = 0; m < 2; ++m) {
        const std::map<std::string, size_t> k = rocm_take_launches(*d[m]);
        require(k.count("llmx_group_sum") && k.at("llmx_group_sum") == 1 && !k.count("llmx_group_push"), "a sum on members no entry mix reached is one llmx_group_sum launch on member " + std::to_string(m));
    }
    const Run first = run(*d[1], CSlice(c->partial(1)), nullptr, false);
    require(launched(first, "llmx_hc_entry") == 1 && same(first.out, one.out), "member 1's first mix is a width-1 entry with width 1's bytes");
    const auto summed = [&](const std::function<void(Backend&)>& between, bool fail0) {
        for (size_t m = 0; m < 2; ++m) d[m]->write(*c->partial(m).buffer, c->partial(m).offset * 4, (m ? p1 : p0).data(), E * 4);
        Ops ops[2] = {prepare(*d[0]), prepare(*d[1])};
        c->sum_in_place(1, E);
        std::vector<Run> rs;
        for (size_t m = 0; m < 2; ++m) {
            HipFn& f = hip();
            if (fail0 && !m) {
                group_calls.launches = 0;
                group_calls.fail = 1;
                f.hipModuleLaunchKernel = counted_launch;
            }
            bool threw = false;
            try {
                rs.push_back(go(*d[m], ops[m], CSlice(c->partial(m)), between, false));
            } catch (const std::exception&) {
                threw = true;
                rs.push_back({});
            }
            f.hipModuleLaunchKernel = real_launch;
            group_calls.fail = 0;
            require(threw == (fail0 && !m), "a member's refused entry launch throws there alone");
        }
        return rs;
    };
    const std::vector<Run> two = summed(nullptr, false);
    for (size_t m = 0; m < 2; ++m) {
        const std::string at = " on member " + std::to_string(m);
        require(launched(two[m], "llmx_hc_entry") == 1 && launched(two[m], "llmx_group_take") == 0 && launched(two[m], "llmx_group_sum") == 0, "the sum's take and the mix are one llmx_hc_entry launch" + at);
        require(same(two[m].x, one.x) && same(two[m].out, one.out) && same(two[m].low, one.low) && same(two[m].injg, one.injg), "llmx_hc_entry at width 2 gives width 1's bytes" + at);
    }
    const std::vector<Run> took = summed([&](Backend& b) { (void)read(b, *other, 64); }, false);
    for (size_t m = 0; m < 2; ++m)
        require(launched(took[m], "llmx_group_take") == 1 && launched(took[m], "llmx_hc_entry") == 0 && same(took[m].x, one.x) && same(took[m].out, one.out),
                "a read between the sum and the mix takes the sum alone and gives the entry's bytes on member " + std::to_string(m));
    std::cout << "backend-rocm: llmx_hc_entry at width 2 takes the peer's partial and gives width 1's bytes; a read before the mix takes the sum alone, the entry's bytes\n";
    // A refused entry launch: it throws on its member, the peer's entry is exact, and the group's next sums are conservative and exact.
    const std::vector<Run> failed = summed(nullptr, true);
    require(same(failed[1].out, one.out), "the peer of a member whose entry launch was refused gives width 1's bytes");
    const std::vector<Run> after = summed(nullptr, false);
    for (size_t m = 0; m < 2; ++m)
        require(launched(after[m], "llmx_hc_entry") == 1 && launched(after[m], "llmx_group_take") == 0 && same(after[m].out, one.out),
                "after a refused entry launch the group sums conservatively and the inject and mix still run as one launch, width 1's bytes, on member " + std::to_string(m));
    const Run again = run(g, {bb.get(), 0}, nullptr, false);
    require(launched(again, "llmx_hc_entry") == 1 && same(again.out, one.out), "a member whose entry launch was refused runs llmx_hc_entry again with the same bytes");
    std::cout << "backend-rocm: a refused llmx_hc_entry launch throws on its member, its peer's entry is exact, and the group's later sums are conservative and exact\n";
}

// The held rows' push (GpuCollective, push_rows): where the device's held builds push, a sum whose rows a held 8-bit product wrote last launches its take alone, with the bits of the same sum after a copy keeps the push from it.
// Both parity halves are taken, at the widths of 3 and 4, at a tile-exact and a ragged row count, and on the two- and five-span held builds.
void held_push() {
    std::vector<BackendPtr> d = rocm_devices(kMaxTensorWidth);
    if (d.size() < 3 || rocm_profile(*d[0]).held_push_members == 0 || d.size() < rocm_profile(*d[0]).held_push_members) {
        std::cout << "backend-rocm: no group the held builds push, so no held push\n";
        return;
    }
    struct Shape {
        size_t nin, nout;
        const char* kernel;
    };
    const Shape shapes[] = {{1536, 5120, "llmx_q8_0_half8_1h2p"}, {4352, 5120, "llmx_q8_0_half8_1h5p"}, {1536, 204, "llmx_q8_0_half8_1h2p"}};
    for (size_t W = std::max<size_t>(3, rocm_profile(*d[0]).held_push_members); W <= std::min<size_t>(d.size(), 4); ++W)
        for (const Shape& s : shapes) {
            const std::vector<Backend*> group = raw(d, W);
            std::unique_ptr<Collective> c = d[0]->join(group, 1, s.nout);
            require(c != nullptr, "ROCm devices joined no collective");
            if (!c) return;
            std::vector<BufferPtr> wb, xb, other;
            for (size_t m = 0; m < W; ++m) {
                const std::vector<uint8_t> w = q8_matrix(s.nin, s.nout, 71 + (uint32_t)m);
                const std::vector<float> x = row_classes::values(s.nin, 81 + (uint32_t)m);
                wb.push_back(d[m]->adopt(w.data(), w.size()));
                xb.push_back(d[m]->adopt(x.data(), x.size() * 4));
                other.push_back(d[m]->alloc(8, Memory::device));
            }
            // A copy after the product leaves the sum its push, so the sum is the one llmx_group_sum launch the product cannot reach.
            const auto summed = [&](bool copied) {
                for (size_t m = 0; m < W; ++m) {
                    (void)rocm_take_launches(*d[m]);
                    d[m]->matmul({quant::GGML_TYPE_Q8_0, {wb[m].get(), 0}}, {xb[m].get(), 0}, c->partial(m), s.nin, s.nout, 1, {}, Dtype::int8);
                    if (copied) d[m]->copy(*other[m], 0, *other[m], 4, 4);
                }
                c->sum_in_place(1, s.nout);
                std::vector<std::vector<float>> out;
                for (size_t m = 0; m < W; ++m) {
                    d[m]->sync();
                    out.push_back(read(*d[m], *c->partial(m).buffer, s.nout));
                    const std::map<std::string, size_t> k = rocm_take_launches(*d[m]);
                    const auto count = [&](const char* name) { return k.count(name) ? k.at(name) : size_t(0); };
                    const std::string at = " at width " + std::to_string(W) + ", " + std::to_string(s.nin) + " x " + std::to_string(s.nout) + ", member " + std::to_string(m);
                    if (copied) require(count("llmx_group_sum") == 1 && !count("llmx_group_take") && !count(s.kernel), "a sum after a copy is one llmx_group_sum launch" + at);
                    else require(count(s.kernel) == 1 && count("llmx_group_take") == 1 && !count("llmx_group_sum"), std::string("a sum after a held product is ") + s.kernel + " and llmx_group_take" + at);
                }
                return out;
            };
            const std::vector<std::vector<float>> want = summed(true);
            for (size_t m = 1; m < W; ++m) require(same(want[m], want[0]), "every member's sum has member 0's bits");
            for (int r = 0; r < 3; ++r) {
                const std::vector<std::vector<float>> got = summed(false);
                for (size_t m = 0; m < W; ++m)
                    require(same(got[m], want[0]), "a pushed sum has the bits of the sum llmx_group_sum gives, run " + std::to_string(r) + " at width " + std::to_string(W) + ", " + std::to_string(s.nout) + " rows, member " + std::to_string(m));
            }
        }
    std::cout << "backend-rocm: a sum after a held 8-bit product launches its take alone, with llmx_group_sum's bits, at both halves\n";
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
        for (const GpuSumForm form : {GpuSumForm::one_shot, GpuSumForm::chain, GpuSumForm::wrap, GpuSumForm::shares}) {
            const size_t cap = 17 * 4 * 4096 + 5;
            std::unique_ptr<Collective> forced = rocm_collective(group, cap, 1, form);
            group_sums::check(group, *forced, 1, {1, 4 * W + 1, 4 * 4096 + 1, cap}, seed, print);
            seed += 100;
        }
        // The member order timed for the group is an order of its members, asked again the same, and the group in that order sums in its member order.
        const std::vector<size_t> order = d[0]->member_order(group, W, 64, 97);
        std::vector<size_t> sorted = order;
        std::sort(sorted.begin(), sorted.end());
        bool permutes = sorted.size() == W;
        for (size_t k = 0; permutes && k < W; ++k) permutes = sorted[k] == k;
        require(permutes && d[0]->member_order(group, W, 64, 97) == order, "a ROCm group's member order is not one order of its members, kept");
        std::vector<Backend*> ordered;
        for (size_t k : order) ordered.push_back(group[k]);
        std::unique_ptr<Collective> moved = ordered[0]->join(ordered, 64, 97);
        group_sums::check(ordered, *moved, 97, {1, 64}, seed, print);
        seed += 100;
        std::cout << "backend-rocm: a collective of " << W << " devices sums in member order by peer writes, two sums in flight, in each form forced and in its timed member order\n";
    }
    if (d.size() >= 4) {
        // The pick over every device in groups of two is a permutation of them, each device once, asked again the same, and each group it forms sums in its member order.
        const size_t N = d.size() / 2 * 2;
        const std::vector<Backend*> listing = raw(d, N);
        const std::vector<size_t> pick = d[0]->member_order(listing, 2, 64, 97);
        std::vector<size_t> sorted = pick;
        std::sort(sorted.begin(), sorted.end());
        bool permutes = sorted.size() == N;
        for (size_t k = 0; permutes && k < N; ++k) permutes = sorted[k] == k;
        require(permutes && d[0]->member_order(listing, 2, 64, 97) == pick, "a ROCm pick of groups is not one permutation of the devices, kept");
        // Each group in listing order and the groups by their first member, so timing noise between near ties cannot pick another order of the same groups.
        bool canonical = pick[0] == 0;
        for (size_t g = 0; g < N; g += 2) canonical = canonical && pick[g] < pick[g + 1] && (g == 0 || pick[g - 2] < pick[g]);
        require(canonical, "a ROCm pick of groups does not keep each group and the groups in listing order");
        for (size_t g = 0; permutes && g < N; g += 2) {
            const std::vector<Backend*> pair{listing[pick[g]], listing[pick[g + 1]]};
            std::unique_ptr<Collective> c = pair[0]->join(pair, 64, 97);
            group_sums::check(pair, *c, 97, {1, 64}, seed, print);
            seed += 100;
        }
        std::cout << "backend-rocm: a pick of " << N / 2 << " groups of two over " << N << " devices is a permutation of them, kept, each group summing in its member order\n";
    }
    {
        // Devices that cannot reach each other's memory sum by host writes, and by pulls where those are forced, the same bytes.
        std::vector<BackendPtr> e = rocm_devices(d.size());
        const auto can = f.hipDeviceCanAccessPeer;
        f.hipDeviceCanAccessPeer = no_peer;
        std::unique_ptr<Collective> c = e[0]->join(raw(e, e.size()), 64, 97);
        f.hipDeviceCanAccessPeer = can;
        sum_calls(*c, e.size(), true);
        group_sums::check(raw(e, e.size()), *c, 97, {1, 3, 64}, seed, print);
        seed += 100;
        sums_in_flight(raw(e, e.size()), *c, 97, seed);
        seed += 100;
        std::unique_ptr<Collective> pulls = rocm_collective(raw(e, e.size()), 64, 97, GpuSumForm::one_shot, GroupTransport::pulls);
        sum_calls(*pulls, e.size(), false);
        group_sums::check(raw(e, e.size()), *pulls, 97, {1, 3, 64}, seed, print);
        seed += 100;
        sums_in_flight(raw(e, e.size()), *pulls, 97, seed);
        seed += 100;
        std::cout << "backend-rocm: a collective of " << e.size() << " devices without peer access sums by host writes, and forced to pulls, the same bytes\n";
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
        group_calls.fail = 15;   // the add checks, each transport's probe armings and sum, then the collective's armings for both transports, two launches each, come first
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
        // Epochs seeded below the 32-bit wrap: twelve sums in a chain, whose meetings count them, cross it, skipping 0, each exact.
        std::vector<BackendPtr> e = rocm_devices(2);
        group_calls.areas.clear();
        f.hipExtMallocWithFlags = kept_uncached;
        std::unique_ptr<Collective> c = rocm_collective(raw(e, 2), 64, 97, GpuSumForm::chain);
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

// Routed products at Flash-Next's expert shapes for every twin type: the one-column build against the build the call takes, medians of five interleaved.
// Each build's break-even, the figure TwinKernels::group_from holds, follows (docs/src/backends-gpu.md).
void routed_rates(Backend& g) {
    const size_t n_expert = 512, past = 1025, k = 10, kCounts = 8;
    const size_t counts[kCounts] = {1, 2, 4, 8, 16, 32, 64, 128};
    std::cout << "backend-rocm: routed rates; type, dtype, shape x projections, generated rows: us a call one-column, taken, ratio, build taken\n";
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4}) {
        const std::string name = quant::storage_type(type)->name;
        // Sums and the grouped builds' reach per lane form (quarter_form) and class, since each form has its own break-evens.
        double one_sum[2][2][kCounts] = {}, taken_sum[2][2][kCounts] = {};
        bool grouped_at[2][2][kCounts], measured[2][2] = {};
        for (auto& f : grouped_at)
            for (auto& row : f) std::fill(row, row + kCounts, true);
        for (const auto& sh : std::vector<std::array<size_t, 3>>{{2560, 640, 2}, {640, 2560, 1}}) {
            const size_t nin = sh[0], ff = sh[1], nproj = sh[2];
            if (nin % quant::storage_type(type)->block_size) continue;
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
                    const int lf = quarter_form(type, nin, di == 1);
                    measured[lf][di] = true;
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
                if (!measured[lf][di]) continue;
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

// The routed tile's time against the grouped row builds over prompts of 32 to 512 rows at Flash-Next's expert shapes, medians of five interleaved; checks nothing.
// The tile arm times the tile only from the role's `tile_from`, so an entry is derived lower by building with it at 32.
void routed_tile_rates(Backend& g) {
    const size_t n_expert = 512, k = 10;
    std::cout << "backend-rocm: routed tile rates; type, dtype, shape x projections, prompt rows: us a call grouped rows, tile, ratio\n";
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4})
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
    rocm_take_launches(g);
    const size_t values = ds4::check_ops(p, {"hc_collapse_norm", "hc_expand", "indexer_scores", "rope_forms", "mix_rows"});
    // The FP4 keys 128 wide reach the integer index tile wherever the 8-bit dot is built.
    if (rocm_device(0).caps.integer_dot) require(rocm_take_launches(g).count("llmx_indexer_scores_int") == 1, "deepseek4's index checks take llmx_indexer_scores_int");
    std::cout << "backend-rocm: deepseek4 ops: " << values << " values against the CPU, the formats, selections, routing picks, collapse, expansion, index scores, rope forms and taps bit for bit\n";
}

// The vision encoder's ops against the CPU (vision_checks.hpp), each kernel the core adds for it reached.
void vision_ops(Backend& g) {
    q35::Pair p(g);
    rocm_take_launches(g);
    const size_t values = vision::check_ops(p);
    const std::map<std::string, size_t> ran = rocm_take_launches(g);
    for (const char* k : {"llmx_layer_norm_rows", "llmx_gelu", "llmx_add_bias", "llmx_mix_rows", "llmx_norm_rope_streams", "llmx_attention_chunk"})
        require(ran.count(k), std::string("the vision checks take ") + k);
    // The per-row body writes a block-major twin once a product has read one (twin_build), so either build counts.
    require(ran.count("llmx_attention_vec") || ran.count("llmx_attention_vec_major"), "the vision checks take llmx_attention_vec");
    std::cout << "backend-rocm: vision encoder ops: " << values << " values against the CPU, bias exact, row mix, norm, gelu, rope over streams and non-causal attention within their bounds\n";
}

// The dense F32 kernels the backend links in (cmake/embed_modules.cmake).
extern "C" const unsigned char llmx_rocm_f32_module[];
constexpr const unsigned char* kF32Module = llmx_rocm_f32_module;

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
    const GpuSumForm forms[] = {GpuSumForm::one_shot, GpuSumForm::chain, GpuSumForm::wrap, GpuSumForm::shares};
    const size_t chunk = 4 * 4096;   // a chain's largest chunk in floats (kernels/group.h, kChainUnits)
    auto print = [](bool ok, const std::string& what) { require(ok, what); };
    uint32_t seed = 1601;
    for (const size_t W : {2, 3, 4, 5, 8, 12}) {
        std::vector<BackendPtr> d;
        for (size_t m = 0; m < W; ++m) d.push_back(make_rocm_backend(0));
        const std::vector<Backend*> group = raw(d, W);
        const size_t cap = W > 5 ? 3 * chunk + 5 : 17 * chunk + 5;
        std::vector<size_t> sizes = {1, 4 * W - 1, 4 * W + 1, 20 * W + 3, chunk - 1, chunk + 1, cap};
        if (W <= 5) sizes.push_back(16 * chunk + 7);
        for (size_t k = 0; k < std::size(forms); ++k) {
            Settled c(group, rocm_collective(group, cap, 1, forms[k]));
            sum_calls(*c.c, W, true);
            group_sums::check(group, c, 1, sizes, seed, print);
            seed += 100;
            if (W <= 4) {
                std::unique_ptr<Collective> rows = rocm_collective(group, 64, 97, forms[k]);
                sums_in_flight(group, *rows, 97, seed);
                seed += 100;
            }
            std::cout << "backend-rocm: " << W << " ranks on one device sum in " << sum_form_name(forms[k]) << ", into and in place at " << sizes.size() << " sizes up to " << cap << " floats"
                      << (W <= 4 ? ", two sequences in flight," : ",") << " each the oracle's bits\n";
        }
    }
    std::vector<BackendPtr> e = {make_rocm_backend(0), make_rocm_backend(0)};
    for (size_t k = 0; k < std::size(forms); ++k) {
        std::unique_ptr<Collective> c = rocm_collective(raw(e, 2), 64, 97, forms[k]);
        const std::vector<float> rows(97, 1.0f);
        for (size_t m = 0; m < 2; ++m) e[m]->write(*c->partial(m).buffer, 0, rows.data(), rows.size() * sizeof(float));
        group_calls.launches = 0;
        group_calls.fail = 2;
        f.hipModuleLaunchKernel = counted_launch;
        require(throws([&] { c->sum_in_place(1, 97); }), std::string(sum_form_name(forms[k])) + ": a group sum whose peer's launch failed did not throw");
        f.hipModuleLaunchKernel = real_launch;
        group_calls.fail = 0;
        c->wait(0);
        require(throws([&] { e[0]->raise(0); }), std::string(sum_form_name(forms[k])) + ": a group sum that gave up did not throw on its backend");
        bool poisoned = true;
        for (float v : read(*e[0], *c->partial(0).buffer, 97)) poisoned = poisoned && std::isnan(v);
        require(poisoned, std::string(sum_form_name(forms[k])) + ": a group sum that gave up left values other than NaN");
        sum_calls(*c, 2, false);
        group_sums::check(raw(e, 2), *c, 97, {1, 64}, seed, print);
        seed += 100;
        std::cout << "backend-rocm: a sum in " << sum_form_name(forms[k]) << " whose peer's launch failed turned the collective conservative, later sums exact\n";
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
static_assert(sizeof(SixteenPeers) == 280 && offsetof(SixteenPeers, status) == 256 && offsetof(SixteenPeers, cancel) == 272, "kernels/f32.gpu, GroupPeers");

void sixteen_ranks() {
    HipFn& f = hip();
    const size_t W = kMaxTensorWidth, n = 3 * 4 * 4096 + 5, cap = (n + 3) / 4 * 4, signal = 8192, units = (n + 3) / 4;
    hip_check(f.hipSetDevice(0), "hipSetDevice");
    hipModule_t module = nullptr;
    hipFunction_t sums[3] = {}, arm = nullptr;
    hip_check(f.hipModuleLoadData(&module, kF32Module), "hipModuleLoadData");
    const char* const kernels[] = {"llmx_group_sum", "llmx_group_shares", "llmx_group_chain"};
    for (int k = 0; k < 3; ++k) hip_check(f.hipModuleGetFunction(&sums[k], module, kernels[k]), "hipModuleGetFunction");
    hip_check(f.hipModuleGetFunction(&arm, module, "llmx_group_arm"), "hipModuleGetFunction");
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
        // Two parity halves of slots, armed, as the collective makes them at join.
        hip_check(f.hipExtMallocWithFlags(&area[k], signal + 2 * W * cap * sizeof(float), hipDeviceMallocUncached), "hipExtMallocWithFlags");
        hip_check(f.hipMalloc(&own[k], n * sizeof(float)), "hipMalloc");
        hip_check(f.hipMalloc(&out[k], n * sizeof(float)), "hipMalloc");
        hip_check(f.hipStreamCreateWithFlags(&streams[k], 1), "hipStreamCreateWithFlags");
        hip_check(f.hipMemsetAsync(area[k], 0, signal, streams[k]), "hipMemsetAsync");
        void* slots = (uint8_t*)area[k] + signal;
        unsigned long long words = 2 * W * cap;
        void* arm_args[] = {&slots, &words};
        hip_check(f.hipModuleLaunchKernel(arm, 256, 1, 1, 256, 1, 1, 0, streams[k], arm_args, nullptr), "hipModuleLaunchKernel");
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
    // Each form's blocks for ranks sharing one device: the one-shot's 16, shares a block a 256 units of a share up to 16, the chain two a chunk of 4096 units.
    const unsigned blocks[] = {16, std::min(16u, (unsigned)(((units + W - 1) / W + 255) / 256)), (unsigned)(2 * ((units + 4095) / 4096))};
    int half = 0;   // the parity half, flipped by each one-shot and shares sum as the collective flips it
    // Form 3 is the wrap, the chain's kernel with flag 8.
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
            int members = (int)W, rank = (int)k, flags = 2 | (replace ? 1 : 0) | half << 2 | (form == 3 ? 8 : 0);
            void* args[] = {&table, &o, &x, &nn, &cc, &members, &rank, &flags, &timeout, &tick};
            hip_check(f.hipModuleLaunchKernel(sums[std::min(form, 2)], blocks[std::min(form, 2)], 1, 1, 256, 1, 1, 0, streams[k], args, nullptr), "hipModuleLaunchKernel");
        }
        if (form < 2) half ^= 1;
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

// A chain with a peer whose totals land after its count: rank 0 is the kernel and the host plays rank 1, its second sum raising rank 0's back count 50 ms before the totals it orders.
// A read must wait for its data, so rank 0 sums the late totals and not what the first sum's take left in the slot, which a write passing another on the way to a peer would give (docs/TESTS.md).
void late_totals() {
    HipFn& f = hip();
    const size_t W = 2, n = 2048, cap = n, signal = 8192;
    hip_check(f.hipSetDevice(0), "hipSetDevice");
    hipModule_t module = nullptr;
    hipFunction_t sum = nullptr, arm = nullptr;
    hip_check(f.hipModuleLoadData(&module, kF32Module), "hipModuleLoadData");
    hip_check(f.hipModuleGetFunction(&sum, module, "llmx_group_chain"), "hipModuleGetFunction");
    hip_check(f.hipModuleGetFunction(&arm, module, "llmx_group_arm"), "hipModuleGetFunction");
    int khz = 0;
    hip_check(f.hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0), "hipDeviceGetAttribute");
    unsigned long long timeout = (unsigned long long)khz * 2000, tick = (unsigned long long)khz;
    std::vector<void*> area(W);
    void *own = nullptr, *out = nullptr, *table = nullptr;
    hipStream_t run = nullptr, peer = nullptr;
    hip_check(f.hipStreamCreateWithFlags(&run, 1), "hipStreamCreateWithFlags");
    hip_check(f.hipStreamCreateWithFlags(&peer, 1), "hipStreamCreateWithFlags");
    unsigned *status = nullptr, *fault = nullptr, *cancel = nullptr, *word = nullptr;
    float* staged = nullptr;
    hip_check(f.hipHostMalloc((void**)&status, W * sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    hip_check(f.hipHostMalloc((void**)&fault, sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    hip_check(f.hipHostMalloc((void**)&cancel, sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    hip_check(f.hipHostMalloc((void**)&word, 2 * sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
    hip_check(f.hipHostMalloc((void**)&staged, n * sizeof(float), hipHostMallocCoherent), "hipHostMalloc");
    std::memset(status, 0, W * sizeof(unsigned));
    *fault = 0;
    *cancel = 0;
    for (size_t k = 0; k < W; ++k) {
        hip_check(f.hipExtMallocWithFlags(&area[k], signal + 2 * W * cap * sizeof(float), hipDeviceMallocUncached), "hipExtMallocWithFlags");
        hip_check(f.hipMemsetAsync(area[k], 0, signal, run), "hipMemsetAsync");
        void* slots = (uint8_t*)area[k] + signal;
        unsigned long long words = 2 * W * cap;
        void* arm_args[] = {&slots, &words};
        hip_check(f.hipModuleLaunchKernel(arm, 256, 1, 1, 256, 1, 1, 0, run, arm_args, nullptr), "hipModuleLaunchKernel");
    }
    hip_check(f.hipMalloc(&own, n * sizeof(float)), "hipMalloc");
    hip_check(f.hipMalloc(&out, n * sizeof(float)), "hipMalloc");
    SixteenPeers peers{};
    for (size_t k = 0; k < W; ++k) {
        peers.stage[k] = (float*)((uint8_t*)area[k] + signal);
        peers.signal[k] = (unsigned*)area[k];
    }
    peers.status = status;
    peers.fault = fault;
    peers.cancel = cancel;
    hip_check(f.hipMalloc(&table, sizeof(SixteenPeers)), "hipMalloc");
    hip_check(f.hipMemcpyAsync(table, &peers, sizeof(SixteenPeers), hipMemcpyHostToDevice, run), "hipMemcpyAsync");
    hip_check(f.hipStreamSynchronize(run), "hipStreamSynchronize");
    // Rank 0's GroupSignal words (kernels/group.h) after 32 epochs and the abort word: done[block][sender 1] for the chain's two blocks, then back[0] past the 32 x 16 done words and 16 ahead.
    const auto at = [&](size_t index) { return (uint8_t*)area[0] + index * sizeof(unsigned); };
    const auto put = [&](void* to, const void* from, size_t bytes) {
        hip_check(f.hipMemcpyAsync(to, from, bytes, hipMemcpyHostToDevice, peer), "hipMemcpyAsync");
        hip_check(f.hipStreamSynchronize(peer), "hipStreamSynchronize");
    };
    const unsigned chunks = 8;   // n floats are 512 units, which a chain's chunk halves from 4096 to 64 for kChainDepth chunks (kernels/group.h)
    unsigned* back_word = word + 1;
    float* totals = peers.stage[0] + cap;   // rank 0's slot from rank 1 in parity half 0, where the chain's total comes back
    for (unsigned epoch = 1; epoch <= 2; ++epoch) {
        const group_sums::Case c = group_sums::make(W, n, 1700 + epoch);
        const std::vector<float> want = group_sums::oracle(c, true);
        hip_check(f.hipMemcpyAsync(own, c.partial[0].data(), n * sizeof(float), hipMemcpyHostToDevice, run), "hipMemcpyAsync");
        hip_check(f.hipStreamSynchronize(run), "hipStreamSynchronize");
        const float* o = (const float*)own;
        float* x = (float*)out;
        unsigned long long nn = n, cc = cap;
        int members = (int)W, rank = 0, flags = 2 | 1;
        void* args[] = {&table, &o, &x, &nn, &cc, &members, &rank, &flags, &timeout, &tick};
        hip_check(f.hipModuleLaunchKernel(sum, 2, 1, 1, 256, 1, 1, 0, run, args, nullptr), "hipModuleLaunchKernel");
        std::memcpy(staged, want.data(), n * sizeof(float));
        *word = epoch;
        *back_word = chunks * epoch;
        if (epoch == 1) put(totals, staged, n * sizeof(float));
        put(at(32 + 1 + 32 * 16 + 16), back_word, sizeof(unsigned));
        if (epoch == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            put(totals, staged, n * sizeof(float));
        }
        for (size_t b = 0; b < 2; ++b) put(at(32 + 1 + b * 16 + 1), word, sizeof(unsigned));
        hip_check(f.hipStreamSynchronize(run), "hipStreamSynchronize");
        require(!*fault && !status[0], "the chain gave up on a peer whose totals came late in sum " + std::to_string(epoch));
        std::vector<float> got(n);
        hip_check(f.hipMemcpyAsync(got.data(), out, n * sizeof(float), hipMemcpyDeviceToHost, run), "hipMemcpyAsync");
        hip_check(f.hipStreamSynchronize(run), "hipStreamSynchronize");
        require(same(got, want), "the chain summed what its slot held before a peer's totals that landed after its count, sum " + std::to_string(epoch));
    }
    (void)f.hipStreamDestroy(run);
    (void)f.hipStreamDestroy(peer);
    for (void* a : area) (void)f.hipFree(a);
    (void)f.hipFree(own);
    (void)f.hipFree(out);
    (void)f.hipFree(table);
    for (void* h : {(void*)status, (void*)fault, (void*)cancel, (void*)word, (void*)staged}) (void)f.hipHostFree(h);
    (void)f.hipModuleUnload(module);
    std::cout << "backend-rocm: the chain waits for a peer's totals that land after its count\n";
}

// Measurement for tools/sums_vs_reference.py: the joined group's sums per size in one member order, timed as `sums` times them.
void vsnccl(bool reversed, size_t W, size_t model) {
    std::vector<BackendPtr> d = rocm_devices(W);
    if (d.size() < W) return;
    std::vector<Backend*> group = raw(d, W);
    if (reversed) std::reverse(group.begin(), group.end());
    // A model width of 5120 times the 27B's rows and the odd row its picks cover; another width joins as a model of that width and times its one decode row.
    const std::unique_ptr<Collective> rows = group[0]->join(group, 512, model), odd = model == 5120 ? group[0]->join(group, 1, 65537) : nullptr;
    std::vector<size_t> rows_of = {1, 2, 4, 6, 8, 1, 13, 32, 128, 512}, widths = {5120, 5120, 5120, 5120, 5120, 65537, 5120, 5120, 5120, 5120};
    if (model != 5120) rows_of = {1}, widths = {model};
    for (size_t s = 0; s < rows_of.size(); ++s) {
        Collective& c = odd && widths[s] == 65537 ? *odd : *rows;
        const size_t n = rows_of[s] * widths[s];
        const int N = n >= 100000 ? 10 : 32;
        // A pass submits each device once for many sums, as all_reduce_perf queues its calls: the judged arm submits once per timed window, the llmx-each arm after every sum.
        c.sum_in_place(rows_of[s], widths[s]);
        for (Backend* m : group) m->submit();
        for (Backend* m : group) m->sync();
        for (const bool each : {false, true}) {
            std::vector<double> ms;
            for (int rep = 0; rep < 15; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) {
                    c.sum_in_place(rows_of[s], widths[s]);
                    if (each)
                        for (Backend* m : group) m->submit();
                }
                if (!each)
                    for (Backend* m : group) m->submit();
                for (Backend* m : group) m->sync();
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("ARM %s order %s n %zu us %.2f GBs %.3f min %.2f max %.2f width %zu\n", each ? "llmx-each" : "llmx", reversed ? "10" : "01", n, ms[7] * 1e3,
                        n * 4.0 / ms[7] / 1e6, ms[0] * 1e3, ms[14] * 1e3, W);
        }
    }
    std::fflush(stdout);
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
            late_totals();
            sum_forms();
            std::cout << (failures ? "backend-rocm: FAIL\n" : "backend-rocm: PASS\n");
            return failures ? 1 : 0;
        }
        // `rates` prints the twin types' product times at Flash-Next's shapes, the exact products' times, attention's times and the routed products' times and break-evens, and checks nothing.
        // `rates attention`, `rates argmax`, `rates routed`, `rates tile`, `rates exact` or `rates decode` prints one part, and `rates delta`, which the whole run leaves out, the fused linear-attention step against the base's calls.
        if (argc > 1 && std::string(argv[1]) == "vsnccl") {
            vsnccl(argc > 2 && std::string(argv[2]) == "10", argc > 3 ? (size_t)std::atoi(argv[3]) : 2, argc > 4 ? (size_t)std::atoi(argv[4]) : 5120);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "rates") {
            const std::string part = argc > 2 ? argv[2] : "";
            if (part.empty()) rates(*g);
            if (part.empty() || part == "attention") attention_rates(*g);
            if (part.empty() || part == "argmax") argmax_rates(*g);
            if (part.empty() || part == "routed") routed_rates(*g);
            if (part.empty() || part == "tile") routed_tile_rates(*g);
            if (part.empty() || part == "exact") exact_rates(*g);
            if (part.empty() || part == "decode") decode_rates(*g);
            if (part == "delta") delta_rates(*g);
            return 0;
        }
        // `handoffs` runs only the device-side crossings over every visible device, and the busy times across them.
        if (argc > 1 && std::string(argv[1]) == "handoffs") {
            handoffs();
            busy_times();
            std::cout << (failures ? "backend-rocm handoffs: FAIL\n" : "backend-rocm handoffs: PASS\n");
            return failures ? 1 : 0;
        }
        // `collective` runs only the tensor group's collective checks over every visible device.
        // `hc_entry` runs only llmx_hc_entry's checks, on one device and on a pair where two are visible.
        if (argc > 1 && std::string(argv[1]) == "hc_entry") {
            hc_entry();
            std::cout << (failures ? "backend-rocm hc_entry: FAIL\n" : "backend-rocm hc_entry: PASS\n");
            return failures ? 1 : 0;
        }
        if (argc > 1 && std::string(argv[1]) == "collective") {
            collective();
            held_push();
            sentinel_sums();
            std::cout << (failures ? "backend-rocm collective: FAIL\n" : "backend-rocm collective: PASS\n");
            return failures ? 1 : 0;
        }
        // `qwen35` runs only the qwen35 layers' ops against the CPU and the device's bit-for-bit rules for them (qwen35_checks.hpp).
        if (argc > 1 && std::string(argv[1]) == "qwen35") {
            qwen35(*g);
            std::cout << (failures ? "backend-rocm qwen35: FAIL\n" : "backend-rocm qwen35: PASS\n");
            return failures ? 1 : 0;
        }
        // `vision` runs only the vision encoder's ops against the CPU (vision_checks.hpp).
        if (argc > 1 && std::string(argv[1]) == "vision") {
            vision_ops(*g);
            std::cout << (failures ? "backend-rocm vision: FAIL\n" : "backend-rocm vision: PASS\n");
            return failures ? 1 : 0;
        }
        // `deepseek4` runs only DeepSeek-V4.1's ops against the CPU.
        if (argc > 1 && std::string(argv[1]) == "deepseek4") {
            deepseek4(*g);
            ds4::sparse_tile_pair(*g, rocm_caps, rocm_take_launches, require);
            ds4::router(*g, rocm_caps, rocm_take_launches, require);
            std::cout << (failures ? "backend-rocm deepseek4: FAIL\n" : "backend-rocm deepseek4: PASS\n");
            return failures ? 1 : 0;
        }
        // `moe` runs only the routed experts' checks, the focused test of their kernels.
        if (argc > 1 && std::string(argv[1]) == "moe") {
            moe().routing(*g, *c);
            moe().experts(*g, *c);
            moe().experts_grouped(*g);
            moe().experts_tile(*g, *c);
            moe().experts_quads(*g);
            moe().routed_builds_witness(*g);
            moe().experts_add(*g);
            std::cout << (failures ? "backend-rocm moe: FAIL\n" : "backend-rocm moe: PASS\n");
            return failures ? 1 : 0;
        }
        // `scratch [FILE [DEVICE]]` runs only the check that a fitted model's passes grow no device memory past what its fit priced, on FILE too when given.
        if (argc > 1 && std::string(argv[1]) == "scratch") {
            scratch_reserved(argc > 2 ? argv[2] : "", argc > 3 ? std::atoi(argv[3]) : 0);
            std::cout << (failures ? "backend-rocm scratch: FAIL\n" : "backend-rocm scratch: PASS\n");
            return failures ? 1 : 0;
        }
        // `dspine` runs only a DSpine drafter's ops, windowed attention and the four of its block.
        if (argc > 1 && std::string(argv[1]) == "dspine") {
            dspine_checks::window(*g, *c, rocm_take_launches, require);
            dspine_checks::ops(*g, *c, require, {1, 4, 1});
            std::cout << (failures ? "backend-rocm dspine: FAIL\n" : "backend-rocm dspine: PASS\n");
            return failures ? 1 : 0;
        }
        // `attention` runs only the attention checks, the focused test of its kernels.
        if (argc > 1 && std::string(argv[1]) == "attention") {
            attention(*g, *c);
            classes(*g);
            attention_twins(*g, Dtype::f16);
            attention_twins(*g, Dtype::int8);
            gated_attention_bits(*g, Dtype::f16);
            gated_attention_bits(*g, Dtype::int8);
            attention_chunks(*g);
            attention_tiles::check(*g, rocm_take_launches, require);
            attention_tiles::check_long(*g, rocm_take_launches, require);
            dspine_checks::window(*g, *c, rocm_take_launches, require);
            vec_bits(*g);
            ds4::sparse_grouped(*g, rocm_caps, rocm_take_launches, require);
            std::cout << (failures ? "backend-rocm attention: FAIL\n" : "backend-rocm attention: PASS\n");
            return failures ? 1 : 0;
        }
        // `twins` runs only the producers' twins and the kept twin's rules, the hyper-connection mix's one launch against its chain and the fused moe_step against the base's.
        if (argc > 1 && std::string(argv[1]) == "twins") {
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                twin_checks::producer_twins(*g, dt, require);
                twin_checks::stream_twins(*g, dt, rocm_take_launches, require);
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
            twin_checks::delta_step_bits(*g, rocm_device(0).caps, rocm_take_launches, require);
            std::cout << (failures ? "backend-rocm delta: FAIL\n" : "backend-rocm delta: PASS\n");
            return failures ? 1 : 0;
        }
        // `types` runs only the weight types' check, the focused test of their kernels.
        if (argc > 1 && std::string(argv[1]) == "types") {
            types(*g, *c);
            exact_rows(*g);
            bits().half_tile(*g);
            twin_types(*g, *c);
            biased_edges(*g);
            bits().q8_decode_bits(*g);
            bits().q8_columns_bits(*g);
            prompt_crossover(*g);
            bits().mxfp4_decode_bits(*g);
            nvfp4_scales(*g, *c);
            twin_checks::fp8_grids(*g, *c, rocm_take_launches, require);
            std::cout << (failures ? "backend-rocm types: FAIL\n" : "backend-rocm types: PASS\n");
            return failures ? 1 : 0;
        }
        buffers(*g);
        host_views(*g);
        refusals(*g);
        ops(*g, *c);
        types(*g, *c);
        exact_rows(*g);
        bits().half_tile(*g);
        twin_types(*g, *c);
        biased_edges(*g);
        bits().q8_decode_bits(*g);
        bits().q8_columns_bits(*g);
        prompt_crossover(*g);
        bits().mxfp4_decode_bits(*g);
        nvfp4_scales(*g, *c);
        twin_checks::fp8_grids(*g, *c, rocm_take_launches, require);
        attention(*g, *c);
        classes(*g);
        twin_checks::twin_products(*g, rocm_device(0).caps, Dtype::f16, rocm_take_launches, require);
        twin_checks::producer_twins(*g, Dtype::f16, require);
        twin_checks::stream_twins(*g, Dtype::f16, rocm_take_launches, require);
        twin_checks::twin_products(*g, rocm_device(0).caps, Dtype::int8, rocm_take_launches, require);
        twin_checks::producer_twins(*g, Dtype::int8, require);
        twin_checks::stream_twins(*g, Dtype::int8, rocm_take_launches, require);
        hyper_mix_bits(*g);
        if (rocm_devices(2).size() == 2) hyper_mix_bits(*make_rocm_backend(1), g.get());
        moe_step_bits(*g);
        twin_checks::delta_step_bits(*g, rocm_device(0).caps, rocm_take_launches, require);
        moe_rows_bits(*g);
        fused_norm_rope_kv(*g);
        attention_twins(*g, Dtype::f16);
        attention_twins(*g, Dtype::int8);
        gated_attention_bits(*g, Dtype::f16);
        gated_attention_bits(*g, Dtype::int8);
        host_output(*g);
        host_copies(*g);
        attention_chunks(*g);
        attention_tiles::check(*g, rocm_take_launches, require);
        attention_tiles::check_long(*g, rocm_take_launches, require);
        dspine_checks::window(*g, *c, rocm_take_launches, require);
        dspine_checks::ops(*g, *c, require, {1, 4, 1});
        vec_bits(*g);
        ds4::sparse_grouped(*g, rocm_caps, rocm_take_launches, require);
        pass_args(*g);
        twin_checks::kernel_times(*make_rocm_backend(0, true), *g, rocm_take_launches, require);
        records();
        failure();
        collective();
        held_push();
        hc_entry();
        sentinel_sums();
        handoffs();
        busy_times();
        qwen35(*g);
        vision_ops(*g);
        deepseek4(*g);
        ds4::sparse_tile_pair(*g, rocm_caps, rocm_take_launches, require);
        ds4::router(*g, rocm_caps, rocm_take_launches, require);
        moe().routing(*g, *c);
        moe().experts(*g, *c);
        moe().experts_add(*g);
        moe().experts_quads(*g);
        moe().routed_builds_witness(*g);
    } catch (const std::exception& e) {
        std::cout << "FAIL threw: " << e.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "backend-rocm: FAIL\n" : "backend-rocm: PASS\n");
    return failures ? 1 : 0;
}
