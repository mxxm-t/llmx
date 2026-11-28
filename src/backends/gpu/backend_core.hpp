#pragma once
// The GPU backend's host core (docs/src/backends-gpu.md), a class template over a vendor's Api: one stream a device, every op an asynchronous launch on it, a ticket an event recorded on the stream.
// The host never waits inside a pass: it waits only where a caller asks for a ticket, a read, or a ring of host memory it reuses; a vendor's translation unit instantiates it once (rocm/rocm_backend.cpp).
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "backends/expert_ring.hpp"
#include "backends/gpu/gpu_backend.hpp"
#include "backends/gpu/kernels/shapes.h"
#include "backends/kv_storage.hpp"
#include "format/weight_scales.hpp"
#include "quant/model_formats.hpp"
#include "quant/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace backend {
inline int diag_arm = 0;   // DIAG dev only: 1 leak graves, 2 settle in record, 3 stream sync in read, 4 coherent stage, 5 sync after write, 6 malloc excludes free, 7 issue excludes free
namespace {

constexpr unsigned kThreads = 256;         // the kernels' block size (kernels/f32.gpu)
constexpr size_t kBlockTokens = 64;        // tokens a KV block holds
constexpr size_t kRawIndexRows = 16, kRawIndexThreads = 256;   // llmx_raw_index_scores' rows a block and its threads (kernels/f32.gpu)
constexpr size_t kIndexTileRows = 16, kIndexTilePositions = 32, kIndexTileThreads = 128, kIndexTileFixed = 16 * 32 * 4;   // llmx_indexer_scores_tile's tile and its static shared bytes (kernels/f32.gpu)
constexpr size_t kIndexIntRows = 32, kIndexIntPositions = 128, kIndexIntDim = 128;   // llmx_indexer_scores_int's tile and the one dim it reads (kernels/f32.gpu)
constexpr size_t kSparseWideRows = 8, kSparseTileWide = 128;   // the rows from which the sparse tile pair runs, and its tile width (kernels/f32.gpu, kSparseTileWide)
constexpr size_t kSparseTileHeads32 = 32, kSparseTileWide32 = 256;   // the heads up to which the pair's 32-head build runs, and its tile width (kernels/f32.gpu, kSparseTileWide32)
// The few-row sparse pair's tile width and its heads a block, 16 at most and 2 at least: 2 heads of 16 lanes keep a block a wave of the narrowest devices (kernels/f32.gpu, LLMX_SPARSE_FEW).
constexpr size_t kSparseFewWide = 16, kSparseFewMost = 16, kSparseFewLeast = 2;
constexpr size_t kSparseFewTwoPerUnit = 3;   // one-column blocks a compute unit from which the few-row values kernel takes two columns a lane (sparse_few_two)
constexpr size_t kEvents = 1024;           // tickets in flight before a submission waits for the oldest
constexpr size_t kRingSlots = 4;           // a host ring's slots, each reused once the submission that last read it retires
constexpr size_t kArgBytes = size_t(1) << 20, kStageBytes = size_t(16) << 20;
constexpr size_t kGroupSignalBytes = 8192;  // a group member's signal block before its staging (kernels/f32.gpu, GroupSignal)

// The host's side of a word of registered host memory a kernel sets: an acquire load, an acquire-release exchange and a release store; MSVC's interlocked calls are full barriers, at least as strong.
#if defined(_MSC_VER)
inline unsigned host_load(unsigned* p) { return unsigned(_InterlockedCompareExchange(reinterpret_cast<volatile long*>(p), 0, 0)); }
inline unsigned host_exchange(unsigned* p, unsigned v) { return unsigned(_InterlockedExchange(reinterpret_cast<volatile long*>(p), long(v))); }
inline void host_store(unsigned* p, unsigned v) { (void)_InterlockedExchange(reinterpret_cast<volatile long*>(p), long(v)); }
inline void host_store(uint64_t* p, uint64_t v) { (void)_InterlockedExchange64(reinterpret_cast<volatile long long*>(p), (long long)v); }
#else
inline unsigned host_load(unsigned* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
inline unsigned host_exchange(unsigned* p, unsigned v) { return __atomic_exchange_n(p, v, __ATOMIC_ACQ_REL); }
inline void host_store(unsigned* p, unsigned v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
inline void host_store(uint64_t* p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
#endif

// What a tensor group's first member keeps of the group's probes: the fast transports whose probes passed, why the others failed, and each fast sum's transport and form by its size, timed up to `timed` floats.
struct GroupPath {
    std::vector<GroupTransport> transports;   // peer writes before host writes; none for the conservative path
    std::string why;
    bool copies = false;   // host writes also time copies, whose own probe passed
    size_t timed = 0;
    std::map<size_t, GpuSumPick> forms;   // from each timed size up, the pick that won there; below the least, the one-shot of that pick's transport
};

// A transport as one bit of a set of them.
inline unsigned transport_bit(GroupTransport t) { return 1u << (unsigned)t; }

template <class Api> class GpuBackend;
template <class Api> class GpuCollective;
template <class Api> class GpuHandoff;

// A kernel's arguments by value, packed at their natural alignments as the kernel's argument segment lays them out, so a launch can be held and issued later.
struct Args {
    static constexpr size_t kBytes = 1024;
    alignas(16) uint8_t bytes[kBytes];
    size_t size = 0;
    template <class... A>
    explicit Args(const A&... a) { (put(a), ...); }
    template <class T>
    void put(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>, "a kernel argument is copied by its bytes");
        size = (size + alignof(T) - 1) / alignof(T) * alignof(T);
        if (size + sizeof(T) > kBytes) throw std::logic_error("gpu: kernel arguments past their buffer");
        std::memcpy(bytes + size, &v, sizeof(T));
        size += sizeof(T);
    }
};

// Keeps timing flushes with a closed gate apart from runtime calls that wait on a stream or on every stream of a device, process wide.
// Such a call can hold a runtime lock while it waits behind the gate, and the flushing thread would then block on that lock before it opens the gate (docs/src/backends-gpu.md).
class GateFence {
public:
    void gate_enter() {
        std::unique_lock<std::mutex> lock(m_);
        wake_.wait(lock, [&] { return !waiting_ && !syncing_; });
        ++gated_;
    }
    void gate_leave() {
        std::lock_guard<std::mutex> lock(m_);
        if (!--gated_) wake_.notify_all();
    }
    void sync_enter() {
        std::unique_lock<std::mutex> lock(m_);
        ++waiting_;
        wake_.wait(lock, [&] { return !gated_; });
        --waiting_;
        ++syncing_;
    }
    void sync_leave() {
        std::lock_guard<std::mutex> lock(m_);
        if (!--syncing_) wake_.notify_all();
    }
private:
    std::mutex m_;
    std::condition_variable wake_;
    size_t gated_ = 0, waiting_ = 0, syncing_ = 0;   // flushes inside their gate, and waiting calls queued and running
};
inline GateFence& gate_fence() {
    static GateFence f;
    return f;
}
// Held around a runtime call that may wait on a stream: a synchronise, a free, an unregister or a teardown.
struct Fenced {
    Fenced() { gate_fence().sync_enter(); }
    ~Fenced() { gate_fence().sync_leave(); }
    Fenced(const Fenced&) = delete;
    Fenced& operator=(const Fenced&) = delete;
};

// A free that fails has nothing left to act on.
template <class Api>
void free_memory(void* p, bool host) {
    const Fenced fenced;
    (void)(host ? Api::host_free(p) : Api::free(p));
}
template <class Api>
void sync_stream(typename Api::Stream s) {
    const Fenced fenced;
    (void)Api::stream_sync(s);
}

// Host memory wrap_host registered with HIP, which registers an address once for every device and drops it at one unregister however often it was registered: each view counts, and the last one dropped unregisters it.
// HIP keys a registration by address, so one that outlived the caller's memory would be taken for the next memory mapped there, its pages long gone.
struct Registrations {
    std::mutex m;
    std::map<void*, std::pair<size_t, size_t>> held;   // bytes and views of each address
};
Registrations& registrations() {
    static Registrations r;
    return r;
}

// Registers `bytes` at `memory` or counts one more view of a registration of the same span; false where HIP refuses it or another span holds the address.
template <class Api>
bool register_host(void* memory, size_t bytes) {
    Registrations& r = registrations();
    std::lock_guard<std::mutex> lock(r.m);
    const auto it = r.held.find(memory);
    if (it != r.held.end()) {
        if (it->second.first != bytes) return false;
        ++it->second.second;
        return true;
    }
    if (Api::host_register(memory, bytes) != Api::kSuccess) return false;
    r.held[memory] = {bytes, 1};
    return true;
}

template <class Api>
void unregister_host(void* memory) {
    Registrations& r = registrations();
    std::lock_guard<std::mutex> lock(r.m);
    const auto it = r.held.find(memory);
    if (it == r.held.end() || --it->second.second) return;
    const Fenced fenced;
    (void)Api::host_unregister(memory);
    r.held.erase(it);
}

// Memory of dropped buffers, freed once the work recorded before the drop is done, since hipFree does not wait for a launch or copy still queued that names it.
// The backend's own thread frees it: hipFree waits for every stream of the device, so a recording or waiting thread that freed would stop feeding its device until all the work queued after the drop had run (docs/src/backends-gpu.md).
template <class Api>
struct Graves {
    std::mutex m;
    std::condition_variable wake;
    std::vector<std::pair<Ticket, std::pair<void*, bool>>> held;   // the ticket that covers the drop, and the memory
    size_t freeing = 0;                                             // graves the thread has taken and not yet freed
    std::mutex issue;                                               // DIAG arms 6 and 7
    const TicketLedger* tickets = nullptr;                          // null once the backend has drained its stream, when a drop frees at once
    typename Api::Stream stream = nullptr;                                   // the backend's stream, null once it is drained
    std::function<void()> flush;                                    // issues the backend's held stream operations, before a drop waits for its stream
    bool stop = false;                                              // set as the backend goes, which frees what is left and ends the thread
    std::thread thread;

    // The thread: a grave whose covering ticket is recorded has its work on the stream, which a stream synchronise ends without holding another thread's query or launch.
    void run(int device) {
        (void)Api::set_device(device);
        std::unique_lock<std::mutex> lock(m);
        for (;;) {
            const Ticket last = tickets ? tickets->last() : ~Ticket(0);
            std::vector<std::pair<void*, bool>> ready;
            const auto kept = std::remove_if(held.begin(), held.end(), [&](const auto& g) {
                if (g.first > last) return false;
                ready.push_back(g.second);
                return true;
            });
            held.erase(kept, held.end());
            if (ready.empty()) {
                if (stop) return;
                wake.wait(lock);
                continue;
            }
            freeing = ready.size();
            const typename Api::Stream s = stream;
            lock.unlock();
            if (s) sync_stream<Api>(s);
            if (diag_arm != 1) {
                std::unique_lock<std::mutex> il(issue, std::defer_lock);
                if (diag_arm == 6 || diag_arm == 7) il.lock();
                for (const auto& r : ready) free_memory<Api>(r.first, r.second);
            }
            lock.lock();
            freeing = 0;
            wake.notify_all();
        }
    }
    // A ticket was recorded: the graves it covers are ready for the thread.
    void recorded() {
        std::lock_guard<std::mutex> lock(m);
        if (!held.empty()) wake.notify_all();
    }
    // Waits until every grave whose ticket is recorded is freed, and says whether there was any.
    bool settle() {
        std::unique_lock<std::mutex> lock(m);
        const Ticket last = tickets ? tickets->last() : ~Ticket(0);
        auto pending = [&] { return freeing || std::any_of(held.begin(), held.end(), [&](const auto& g) { return g.first <= last; }); };
        if (!pending()) return false;
        wake.notify_all();
        wake.wait(lock, [&] { return !pending(); });
        return true;
    }
    // Whether a drop waits for a ticket not yet recorded.
    bool unrecorded() {
        std::lock_guard<std::mutex> lock(m);
        const Ticket last = tickets ? tickets->last() : ~Ticket(0);
        return std::any_of(held.begin(), held.end(), [&](const auto& g) { return g.first > last; });
    }
};

// A device allocation refused for want of memory: a bad_alloc, as the fit's recut and the passes' retries catch it, carrying the runtime's reason.
class DeviceOutOfMemory final : public std::bad_alloc {
    std::string what_;
public:
    explicit DeviceOutOfMemory(std::string what) : what_(std::move(what)) {}
    const char* what() const noexcept override { return what_.c_str(); }
};

// Device memory from hipMalloc, pinned host memory the device reads in place (Memory::host_visible, coherent for Memory::host_coherent), or the caller's host memory registered with HIP (wrap_host).
// Dropped, HIP's memory goes to the backend's graves, while the caller's is unregistered at once, once the stream's copies naming it retire, since the caller may free it as soon as the view is gone (Backend::wrap_host).
template <class Api>
class GpuBuffer final : public Buffer {
    static void taken(typename Api::Error e, const char* what) {
        if (e != Api::kOutOfMemory) return Api::check(e, what);
        try {
            Api::check(e, what);
        } catch (const std::runtime_error& r) {
            throw DeviceOutOfMemory(r.what());
        }
    }
public:
    GpuBuffer(size_t bytes, bool host, std::shared_ptr<Graves<Api>> graves, bool coherent = false) : size_(bytes), host_(host), graves_(std::move(graves)) {
        if (!bytes) return;
        // Whole words: the prompt tile reads a Q8_0 quarter as aligned words, the last of which may run past a buffer's final block.
        const size_t words = (bytes + 3) & ~size_t(3);
        std::unique_lock<std::mutex> il(graves_->issue, std::defer_lock);
        if (diag_arm == 6) il.lock();
        taken(host ? Api::host_malloc(&p_, words, coherent) : Api::malloc(&p_, words), host ? "host_malloc" : "malloc");
        if (il.owns_lock()) il.unlock();
        device_ = p_;
        if (host) Api::check(Api::host_device_pointer(&device_, p_), "host_device_pointer");
    }
    // A view of memory register_host holds a count of.
    GpuBuffer(void* memory, size_t bytes, void* device, std::shared_ptr<Graves<Api>> graves)
        : p_(memory), device_(device), size_(bytes), host_(true), registered_(true), graves_(std::move(graves)) {}
    ~GpuBuffer() override {
        if (!p_) return;
        if (registered_) {
            {
                std::lock_guard<std::mutex> lock(graves_->m);
                if (graves_->stream) {
                    graves_->flush();
                    sync_stream<Api>(graves_->stream);
                }
            }
            unregister_host<Api>(p_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(graves_->m);
            if (graves_->tickets) {
                graves_->held.push_back({graves_->tickets->last() + 1, {p_, host_}});
                return;
            }
        }
        free_memory<Api>(p_, host_);
    }
    GpuBuffer(const GpuBuffer<Api>&) = delete;
    GpuBuffer<Api>& operator=(const GpuBuffer<Api>&) = delete;
    size_t size() const override { return size_; }
    const void* host_ptr() const override { return host_ ? p_ : nullptr; }
    void* data() const { return p_; }
    // Where a kernel reaches the memory, which for host memory HIP may map at another address.
    void* device_ptr() const { return device_; }
private:
    void* p_ = nullptr;
    void* device_ = nullptr;
    size_t size_;
    bool host_, registered_ = false;
    std::shared_ptr<Graves<Api>> graves_;
};

template <class Api>
const GpuBuffer<Api>& as_gpu(const Buffer& b) {
    const auto* r = dynamic_cast<const GpuBuffer<Api>*>(&b);
    if (!r) throw std::runtime_error(Api::prefix + "a buffer of another backend");
    return *r;
}

// KV blocks on the device, kept and grown by BlockKVStorage; an old buffer a growth copied from is dropped, its memory kept until the copies retire (Graves).
template <class Api>
class GpuKVStorage final : public BlockKVStorage {
public:
    GpuKVStorage(GpuBackend<Api>& owner, const std::vector<KVShape>& shapes, size_t max_tokens, KVType kt, KVType vt,
                  const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools);
private:
    void retire(const BufferPtr& old) override;
};

template <class Api>
class GpuBackend final : public Backend {
public:
    // With `timing` (make_backend's diagnostics), flush times each kernel, copy and fill for kernel_times.
    GpuBackend(int index, bool timing) : dev_(Api::device(index)), timing_(timing) {
        graves_->tickets = &tickets_;
        Api::check(Api::set_device(index), "set_device");
        Api::check(Api::stream_create(&stream_), "stream_create");
        graves_->stream = stream_;
        graves_->flush = [this]() noexcept {
            try { flush(); } catch (const std::exception&) {}
        };
        Api::check(Api::host_malloc((void**)&fault_, sizeof(unsigned), true), "host_malloc");
        *fault_ = 0;
        for (auto* set : {&events_, &starts_})
            for (auto& e : *set) Api::check(Api::event_create(&e, true), "event_create");
        // A device whose architecture the build left out finds no code object in the modules.
        const typename Api::Error loaded = Api::module_load(&module_, Api::f32_module);
        if (Api::no_binary(loaded))
            throw std::runtime_error(Api::prefix + "this build has no kernels for " + dev_.arch + ": rebuild with it in " + Api::arch_option + " (docs/BUILD.md)");
        Api::check(loaded, "module_load");
        Api::check(Api::module_load(&int16_module_, Api::int16_module), "module_load");
        // The dot module loads only where the build compiled it for this device: that is the capability flag, read from the code object rather than from the device's name.
        dev_.caps.integer_dot = Api::module_load(&dot_module_, Api::int16_dot_module) == Api::kSuccess;
        if (!dev_.caps.integer_dot) dot_module_ = nullptr;
        dev_.profile = profile_for(dev_.caps);
        // The prompt tiles and the row builds of four columns and more take the dot where there is one; the narrower quarter-layout builds measured faster on the plain module (docs/STATUS.md, ROCm wide decode rows).
        for (size_t k = 0; k < kKernels; ++k) {
            const bool plain = k == kRows16x1 || k == kRows16x2 || k == kRows16x2r4 || k == kRows8x1 || k == kRows8x2 || k == kRows8x2r4 || k == kTileReduce ||
                               k == kRouted16 || k == kRouted8 || k == kRows16x1Conv || k == kRows8x1Conv;
            const typename Api::Module m = k < kRows16x1 ? module_ : !plain && dot_module_ ? dot_module_ : int16_module_;
            // The fused moe_step's products are in the dot module alone; without it every step runs the base's ops.
            if (k >= kMoeStepUpQ4K && k <= kMoeStepDownQ51C4 && !dot_module_) continue;
            // The integer index tile is built for the targets with the 8-bit dot alone (GPU_DOT4 in kernels/port.h): under HIP the dot module marks them, and the API says where every target has it.
            if (k == kIndexerScoresInt && !dev_.caps.integer_dot && !Api::dot4_everywhere) continue;
            if (k >= kTallQ4K8 && k <= kSquareQ6K8 && !Api::int8_tensor_tiles) continue;
            const typename Api::Error found = Api::module_function(&kernels_[k], m, kKernelNames[k]);
            // BF16's tensor-core tiles are in the modules of targets that have the instructions alone (GPU_FLOAT_MMA in kernels/port.h): CUDA from sm_80, never HIP.
            if (found != Api::kSuccess && (k == kBF16Tile || k == kBF16TileBF16)) {
                kernels_[k] = {};
                continue;
            }
            if (found != Api::kSuccess) Api::check(found, (std::string("module_function ") + kKernelNames[k]).c_str());
        }
        for (size_t s = 0; s < kRingSlots; ++s) {
            args_.slots[s] = std::make_shared<GpuBuffer<Api>>(kArgBytes, true, graves_);
            stage_.slots[s] = std::make_shared<GpuBuffer<Api>>(kStageBytes, true, graves_, diag_arm == 4);
            pass_.slots[s] = std::make_shared<GpuBuffer<Api>>(kArgBytes, false, graves_);
        }
        if (timing_) {
            gate_ = std::make_shared<GpuBuffer<Api>>(sizeof(uint64_t), true, graves_, true);
            host_store(static_cast<uint64_t*>(as_gpu<Api>(*gate_).data()), uint64_t{0});
        }
        graves_->thread = std::thread([g = graves_.get(), index] { g->run(index); });
        probe_memory();
    }
    ~GpuBackend() override {
        try { flush(); } catch (const std::exception&) {}
        (void)Api::set_device(dev_.index);
        sync_stream<Api>(stream_);
        for (auto s : {beside_stream_, copy_stream_})
            if (s) sync_stream<Api>(s), (void)Api::stream_destroy(s);
        ring_.reset();
        {
            std::lock_guard<std::mutex> lock(graves_->m);
            graves_->tickets = nullptr;
            graves_->stream = nullptr;
            graves_->flush = nullptr;
            graves_->stop = true;
            graves_->wake.notify_all();
        }
        if (graves_->thread.joinable()) graves_->thread.join();
        parts_ = {};
        sparse_ = {};
        moe_out_ = {};
        step_twin_ = {};
        moe_tab_ = {};
        hc_scratch_ = {};
        twin_ = {};
        staged_ = {};
        partials_ = {};
        scores_ = {};
        topk_work_ = {};
        select_state_ = {};
        argmax_parts_ = {};
        args_ = {};
        stage_ = {};
        pass_ = {};
        uploads_ = {};
        const Fenced fenced;
        if (module_) (void)Api::module_unload(module_);
        if (int16_module_) (void)Api::module_unload(int16_module_);
        if (dot_module_) (void)Api::module_unload(dot_module_);
        for (auto e : events_) if (e) (void)Api::event_destroy(e);
        for (auto e : starts_) if (e) (void)Api::event_destroy(e);
        for (auto e : timed_events_) (void)Api::event_destroy(e);
        if (stream_) (void)Api::stream_destroy(stream_);
        if (fault_) free_memory<Api>(fault_, true);
    }

    // The device and its driver, not the index or PCI address: cards of one device and driver give the same bits, so a history continues on whichever of them holds its member.
    std::string identity() const override { return Api::name + std::string(" ") + dev_.caps.device + "; " + dev_.caps.driver; }
    size_t prompt_rows() const override { return dev_.profile.prompt_rows; }
    // The card as the device list names it, for messages.
    std::string ordinal() const { return Api::name + std::string(":") + std::to_string(dev_.index); }
    std::string pci_root() const override { return dev_.pci_root; }
    void set_threads(int) override {}
    int threads_available() const override { return 0; }
    // The recorded graves are freed first, so a reading after sync() counts dropped memory as free.
    std::optional<size_t> memory_available() const override {
        graves_->settle();
        size_t free = 0, total = 0;
        if (Api::set_device(dev_.index) != Api::kSuccess || Api::mem_info(&free, &total) != Api::kSuccess) return std::nullopt;
        return free;
    }
    // Each buffer as hipMalloc takes it (alloc_bytes), so a weight's fit counts its pages.
    size_t resident_bytes(uint32_t, size_t, size_t, size_t bytes, bool) const override { return allocation_bytes(bytes); }
    size_t allocation_bytes(size_t bytes) const override { return alloc_bytes(bytes); }
    size_t scratch_bytes(const PassWidths& w) const override {
        size_t n = 0;
        for (size_t b : scratch_sizes(w)) n = size_add(n, alloc_bytes(b));
        return n;
    }
    // Each scratch buffer made at its priced size before the first pass, so no pass grows one and none waits in the graves beside its successor.
    void reserve_scratch(const PassWidths& w) override {
        const ScratchSizes z = scratch_sizes(w);
        for (size_t i = 0; i < z.size(); ++i) {
            BufferPtr& b = *scratch_buffer(i);
            if (z[i] && (!b || b->size() < z[i])) b = alloc(z[i], Memory::device);
        }
    }
    // The runtime's private scratch for the kernels that spill, which it takes at their first launch, and a group's signal block beside the staging collective_bytes_per_row counts.
    size_t memory_kept(size_t) const override { return size_add(driver_scratch_, alloc_bytes(kGroupSignalBytes)); }
    std::unique_ptr<Collective> join(const std::vector<Backend*>& members, size_t rows, size_t width) override;
    std::vector<size_t> member_order(const std::vector<Backend*>& devices, size_t width, size_t rows, size_t summed) override;
    bool times_groups() const override { return true; }
    std::unique_ptr<Handoff> handoff(const std::vector<Backend*>& readers) override;
    // A member's partial rows beside two parity halves of a device slot a member for each transport its timed groups picked (GpuCollective<Api>), the staging or the slots pulls copy into, and two draw slots under peer writes.
    // Before any group of this device is timed, peer writes' alone.
    size_t collective_bytes_per_row(size_t members, size_t width) const override {
        if (!members) return 0;
        const unsigned by = summed_by_ ? summed_by_ : transport_bit(GroupTransport::peer_writes);
        size_t slots = 1;
        for (const GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes, GroupTransport::pulls})
            if (by & transport_bit(t)) slots += 2 * members + 2 * (t == GroupTransport::peer_writes);
        return size_mul(slots, size_mul(width, sizeof(float)));
    }
    bool supports_type(uint32_t type) const override { return type_kernels(type) != nullptr; }
    bool implements(Op op) const override {
        return op == Op::norm_rope_partial || op == Op::sigmoid_mul || op == Op::gated_rms_norm || op == Op::causal_conv_silu ||
               op == Op::gated_delta_rule || op == Op::routed_experts || op == Op::group_rms_norm || op == Op::silu_scaled || op == Op::mix_streams ||
               op == Op::inject_streams || op == Op::signed_sqrt_gate || op == Op::depthwise_conv || op == Op::select_kv || op == Op::sparse_attention || op == Op::argmax_rows || op == Op::embed_ids ||
               op == Op::hc_coeffs || op == Op::hc_collapse_norm || op == Op::hc_expand || op == Op::sparse_attention_sink || op == Op::topk_rows ||
               op == Op::candidate_blocks || op == Op::compress_pool || op == Op::indexer_scores || op == Op::model_format_qdq || op == Op::model_format_pack || op == Op::pool_write ||
               op == Op::sqrt_softplus_routing || op == Op::sigmoid_routing || op == Op::silu_mul_clamp || op == Op::rope_forms || op == Op::mix_rows || op == Op::layer_norm || op == Op::gelu || op == Op::add_bias || op == Op::rope_streams ||
               op == Op::attention_full || op == Op::glu_quick_gelu || op == Op::raw_index_scores || op == Op::sparse_attention_blocks || op == Op::attention_window ||
               op == Op::unit_rows || op == Op::sigmoid_gate_rows || op == Op::matmul_at_ids || op == Op::chain_pick ||
               op == Op::copy_rows || op == Op::softcap || op == Op::glu_softcap || op == Op::expert_window || op == Op::l2_clip ||
               op == Op::query_mix || op == Op::tanh_gate_mix || op == Op::depthwise_conv_ring || op == Op::attention_rel_sink || op == Op::tap_weighted_rows;
    }
    // A generated token is a class; a prompt's extent is classed by the tiles it reaches, product, routed and attention (docs/src/backends-gpu.md, batch invariance).
    size_t row_class(size_t extent) const override {
        if (extent == 1) return 1;
        constexpr size_t entries = sizeof(TwinKernels::tile_from) / sizeof(size_t);   // a type's tile extents, by role and width
        size_t routed = 0;   // the routed builds whose tile the extent reaches; their entries are ordered, so the count names the set
        for (const TypeKernels& t : kTypes)
            if (t.twin)
                for (const auto& role : t.twin->tile_from)
                    for (size_t from : role) routed += extent >= from;
        // A prompt on the row builds takes the split digit 0, which no extent on the tile has.
        return 2 + (extent >= dev_.profile.attention_tile_rows ? 1 : 0) + 2 * (routed + (entries * std::size(kTypes) + 1) * (on_rows(extent) ? 0 : split_tiles_of(extent)));
    }
    // Whether a run of this extent takes the decode row builds, a column each: a generated row, or a prompt shorter than the profile's crossover.
    bool on_rows(size_t extent) const { return extent == 1 || extent < dev_.profile.prompt_rows_below; }
    // The column tiles the tile's inner split is chosen for: the prompt's own, counted up to 512 rows, as the Vulkan tile counts them.
    static size_t split_tiles_of(size_t extent) { return (std::min<size_t>(extent, 512) + kTileRows - 1) / kTileRows; }

    // HIP allocates on the calling thread's current device, and a backend may be called from several threads.
    BufferPtr alloc(size_t bytes, Memory where) override {
        Api::check(Api::set_device(dev_.index), "set_device");
        std::shared_ptr<GpuBuffer<Api>> b;
        try {
            b = std::make_shared<GpuBuffer<Api>>(bytes, where != Memory::device, graves_, where == Memory::host_coherent);
        } catch (const std::exception&) {
            // Dropped memory whose covering ticket is recorded goes back first, which waits for the whole stream; a drop since the last ticket waits for the next.
            // Nothing is recorded here, since alloc may run beside the thread that records.
            if (!graves_->settle()) throw;
            b = std::make_shared<GpuBuffer<Api>>(bytes, where != Memory::device, graves_, where == Memory::host_coherent);
        }
        forget_twins();
        if (!bytes) return b;
        // KV growth and the model read a fresh buffer as zeros.
        if (where != Memory::device) std::memset(b->data(), 0, bytes);
        else hold_fill(b->data(), bytes);
        return b;
    }
    // One registration serves every device's view of the memory; null where HIP refuses it, so the caller writes through staging.
    BufferPtr wrap_host(void* memory, size_t bytes) override {
        if (!memory || !bytes) return nullptr;
        Api::check(Api::set_device(dev_.index), "set_device");
        if (!register_host<Api>(memory, bytes)) return nullptr;
        void* device = nullptr;
        if (Api::host_device_pointer(&device, memory) != Api::kSuccess) {
            unregister_host<Api>(memory);
            return nullptr;
        }
        return std::make_shared<GpuBuffer<Api>>(memory, bytes, device, graves_);
    }
    BufferPtr adopt(const void* src, size_t bytes) override {
        Api::check(Api::set_device(dev_.index), "set_device");
        auto b = std::make_shared<GpuBuffer<Api>>(bytes, false, graves_);
        write(*b, 0, src, bytes);
        return b;
    }

    // The work launched since the last ticket gets the next one; a slot is reused once its earlier ticket retires, which waits only when kEvents tickets are in flight.
    Ticket submit() override {
        raise_failed();
        return record();
    }
    // submit without the failure check, for the paths that cannot throw it to anyone: sync and a collective's wait.
    // Every operation held so far is issued first, so a ticket covers only work on the stream.
    Ticket record() {
        flush();
        // The event is recorded before the ticket is handed out, so a waiter on another thread never reads a slot not yet recorded for it.
        const Ticket t = tickets_.last() + 1;
        const size_t i = t % kEvents;
        if (t > kEvents) {
            wait(t - kEvents);
            // A thread still in wait() on the slot would synchronise on the ticket recorded next, behind work that may need it (the expert ring's reader), so the slot is closed and its waiters, on an event already done, leave first.
            slot_ticket_[i].store(0, std::memory_order_seq_cst);
            while (syncing_[i].load(std::memory_order_seq_cst)) std::this_thread::yield();
        }
        if (!started_) start();
        {
            std::unique_lock<std::mutex> il(graves_->issue, std::defer_lock);
            if (diag_arm == 7) il.lock();
            Api::check(Api::event_record(events_[i], stream_), "event_record");
        }
        slot_ticket_[i].store(t, std::memory_order_release);
        tickets_.take();
        graves_->recorded();
        if (diag_arm == 2) graves_->settle();
        opened_ = started_ = asked_ = waited_ = false;
        for (Ring* r : {&args_, &stage_, &pass_}) r->tag(t);
        return t;
    }
    // The open submission's start: with no wait before it and its device still running the previous ticket, it starts as that one ends, so that one's event times it (docs/src/backends-gpu.md).
    void start() {
        const Ticket t = tickets_.last() + 1;
        chained_[t % kEvents] = !waited_ && t > 1 && !tickets_.settled(t - 1) && !syncing_[(t - 1) % kEvents].load(std::memory_order_acquire) &&
                                Api::event_query(events_[(t - 1) % kEvents]) == Api::kNotReady;
        if (!chained_[t % kEvents]) Api::check(Api::event_record(starts_[t % kEvents], stream_), "event_record");
    }
    double busy_ms(Ticket a, Ticket b) override {
        if (a > b || b > tickets_.last() || a + kEvents <= tickets_.last()) return -1.0;
        double sum = 0;
        for (Ticket t = a + 1; t <= b; ++t) {
            float ms = 0;
            const typename Api::Event from = chained_[t % kEvents] ? events_[(t - 1) % kEvents] : starts_[t % kEvents];
            if (Api::event_elapsed(&ms, from, events_[t % kEvents]) != Api::kSuccess) return -1.0;
            sum += ms;
        }
        return sum;
    }
    Ticket last_ticket() const noexcept override { return tickets_.last(); }
    HostTimes host_times() const override {
        HostTimes t;
        t.launch_ms = launch_ms_;
        return t;
    }
    // The ops timed since the last reading, the first kTimedOps of them, read after the stream drains.
    KernelTimes kernel_times() override {
        KernelTimes out;
        if (!timing_) return out;
        sync();
        (void)Api::set_device(dev_.index);
        std::map<std::string, double> ms;
        for (size_t i = 0; i < timed_.size(); ++i) {
            float t = 0, gap = 0;
            if (Api::event_elapsed(&t, timed_events_[2 * i], timed_events_[2 * i + 1]) != Api::kSuccess) {
                ++out.unresolved;
                continue;
            }
            ms[std::string(timed_[i].name) + (timed_[i].head ? " head" : "")] += t;
            out.dispatches += timed_[i].kernel;
            if (i && Api::event_elapsed(&gap, timed_events_[2 * i - 1], timed_events_[2 * i]) == Api::kSuccess) out.gaps.push_back(std::max(gap, 0.0f));
        }
        out.kernels.assign(ms.begin(), ms.end());
        out.ops = timed_.size();
        out.untimed = untimed_;
        untimed_ = 0;
        timed_.clear();
        return out;
    }
    void wait(Ticket t) noexcept override {
        if (tickets_.settled(t)) return;
        std::atomic<uint32_t>& held = syncing_[t % kEvents];
        held.fetch_add(1, std::memory_order_seq_cst);
        // A slot holds another ticket only once t retired (record).
        if (slot_ticket_[t % kEvents].load(std::memory_order_seq_cst) != t) {
            held.fetch_sub(1, std::memory_order_acq_rel);
            return tickets_.retired(t);
        }
        const typename Api::Error e = Api::event_sync(events_[t % kEvents]);
        held.fetch_sub(1, std::memory_order_acq_rel);
        if (e != Api::kSuccess) remember(e, "a submission");
        tickets_.retired(t);
    }
    bool done(Ticket t) noexcept override {
        if (tickets_.settled(t)) return true;
        // A slot submitted over again held an earlier ticket that submit waited for; a slot taken while this reads it only reads as later.
        if (t + kEvents <= tickets_.last()) {
            tickets_.retired(t);
            return true;
        }
        // HIP holds an event while a thread synchronises on it and a query of it waits that long, so an event some thread waits on reads as not done.
        if (syncing_[t % kEvents].load(std::memory_order_acquire)) return false;
        const typename Api::Error e = Api::event_query(events_[t % kEvents]);
        if (e == Api::kNotReady) return false;
        if (e != Api::kSuccess) remember(e, "a submission");
        tickets_.retired(t);
        return true;
    }
    void sync() noexcept override {
        if (opened_ || graves_->unrecorded()) {
            try { record(); } catch (const std::exception&) {}
        }
        wait(tickets_.last());
        if (ring_) ring_->drain();
    }
    // Results read in place after wait(t): the first failure the device reported throws, then an expert cache fault.
    void raise(Ticket t) override {
        (void)t;
        raise_failed();
        ExpertRing::raise(ring_, *this, Api::prefix);
    }

    void read(const Buffer& src_b, size_t off, void* dst, size_t bytes) override {
        const GpuBuffer<Api>& src = as_gpu<Api>(src_b);
        span(src, off, bytes);
        if (!bytes) return;
        if (src.host_ptr()) {
            sync();
            raise_failed();
            ExpertRing::raise(ring_, *this, Api::prefix);
            std::memcpy(dst, (const uint8_t*)src.data() + off, bytes);
            return;
        }
        for (size_t done = 0; done < bytes;) {
            const size_t n = std::min(bytes - done, kStageBytes);
            void* st = stage_.take(*this, n);
            hold_copy(st, (const uint8_t*)src.data() + off + done, n, Api::kToHost, "readback");
            wait(submit());
            if (diag_arm == 3) sync_stream<Api>(stream_);
            raise_failed();
            ExpertRing::raise(ring_, *this, Api::prefix);
            std::memcpy((uint8_t*)dst + done, st, n);
            done += n;
        }
    }
    void write(Buffer& dst_b, size_t off, const void* src, size_t bytes) override {
        const GpuBuffer<Api>& dst = as_gpu<Api>(dst_b);
        span(dst, off, bytes);
        forget_twins();
        for (size_t done = 0; done < bytes;) {
            const size_t n = std::min(bytes - done, kStageBytes);
            void* st = stage_.take(*this, n);
            std::memcpy(st, (const uint8_t*)src + done, n);
            // HIP copies from the stage into host memory on the host once the stream has drained, as copy() notes, and behind a timing backend's closed gate never.
            if (dst.host_ptr()) {
                const unsigned long long m = n;
                launch(kCopy, (unsigned)std::min<size_t>(1024, (n + 16 * kThreads - 1) / (16 * kThreads)), 1, Args((uint8_t*)dst.device_ptr() + off + done, (const uint8_t*)st, m));
            } else {
                hold_copy((uint8_t*)dst.data() + off + done, st, n, Api::kToDevice, "upload");
            }
            done += n;
        }
        if (diag_arm == 5) sync();
    }
    void copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off, size_t bytes) override {
        const GpuBuffer<Api>& dst = as_gpu<Api>(dst_b);
        const GpuBuffer<Api>& src = as_gpu<Api>(src_b);
        span(dst, dst_off, bytes);
        span(src, src_off, bytes);
        if (!bytes) return;
        // A copy forgets a kept twin only where it writes the twin's input, so a zero row written elsewhere keeps it.
        const float* written = (const float*)((const uint8_t*)dst.data() + dst_off);
        const size_t floats = (bytes + sizeof(float) - 1) / sizeof(float);
        // HIP copies between two host memories on the host once the stream has drained, which blocks the caller, so the device copies them in the stream instead.
        if (src.host_ptr() && dst.host_ptr()) {
            uint8_t* d = (uint8_t*)dst.device_ptr() + dst_off;
            const uint8_t* s = (const uint8_t*)src.device_ptr() + src_off;
            unsigned long long n = bytes;
            const Args args(d, s, n);
            return launch_writing({{written, floats}}, kCopy, (unsigned)std::min<size_t>(1024, (bytes + 16 * kThreads - 1) / (16 * kThreads)), 1, args);
        }
        hold_copy((uint8_t*)dst.data() + dst_off, (const uint8_t*)src.data() + src_off, bytes, Api::kAnyCopy, src.host_ptr() ? "upload" : dst.host_ptr() ? "readback" : "copy");
        forget_written(written, floats);
    }

    void matmul(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs, Dtype dtype) override {
        product(weight, X, Y, nin, nout, nbatch, false, runs, dtype, true);
    }
    // The output head's products, which kernel_times names with " head" after their kernel's.
    void matmul_logits(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs, Dtype dtype, float cap) override {
        head_ = timing_;
        try {
            product(weight, X, Y, nin, nout, nbatch, false, runs, dtype, true, cap);
        } catch (...) {
            head_ = false;
            throw;
        }
        head_ = false;
    }
    void matmul_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs, Dtype dtype) override {
        product(weight, X, Y, nin, nout, nbatch, true, runs, dtype, true);
    }
    // A group's projections as a range, so a run of them is a group too.
    struct Items {
        const Projection* first;
        size_t n;
        const Projection* begin() const { return first; }
        const Projection* end() const { return first + n; }
        size_t size() const { return n; }
    };
    // F32 projections of one row take one exact launch whatever their inputs; other projections of own inputs (Projection::in) take one launch where inputs_list holds them.
    // Else each run of one input is a group of that input, until the rest of the group is one launch.
    void matmul_group(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nbatch, RowRuns runs, Dtype dtype) override {
        const Items all{projections.begin(), projections.size()};
        if (exact_group(all, X, nin, nbatch)) return;
        if (!own_inputs(projections)) return project_group(all, X, nin, nbatch, runs, dtype);
        for (size_t i = 0; i < all.n;) {
            if (inputs_list({all.first + i, all.n - i}, X, nin, nbatch, runs, dtype)) return;
            const CSlice in = input_of(all.first[i], X);
            size_t j = i + 1;
            while (j < all.n && same_input(input_of(all.first[j], X), in)) ++j;
            project_group({all.first + i, j - i}, in, nin, nbatch, runs, dtype);
            i = j;
        }
    }
    static bool same_input(CSlice a, CSlice b) { return a.buffer == b.buffer && a.offset == b.offset; }
    // Two to six F32 projections of one row, each of X or its own input, take one launch of the one-row product, each output the bits of its own call.
    bool exact_group(Items projections, CSlice X, size_t nin, size_t nbatch) {
        bool exact = projections.size() > 1 && projections.size() <= 6 && nin && nbatch == 1;
        for (const Projection& p : projections)
            exact = exact && p.weight.data.buffer && p.weight.type == quant::GGML_TYPE_F32 && p.rows && !as_gpu<Api>(*p.out.buffer).host_ptr() && input_of(p, X).buffer;
        if (!exact) return false;
        ExactGroup g{};
        size_t n = 0, most = 0;
        for (const Projection& p : projections) {
            const CSlice x = input_of(p, X);
            check_weight(p.weight, nin, "matrix", 1);
            check_grid(p.weight, nin, p.rows, 1);
            span(*x.buffer, x.offset * sizeof(float), size_mul(nin, sizeof(float)));
            span(*p.weight.data.buffer, p.weight.data.offset * sizeof(float), quant::row_bytes(p.weight.type, nin, p.rows));
            span(*p.out.buffer, p.out.offset * sizeof(float), size_mul(p.rows, sizeof(float)));
            g.w[n] = bytes(p.weight.data), g.y[n] = f32(p.out), g.x[n] = f32(x), g.nout[n++] = p.rows;
            most = std::max(most, p.rows);
        }
        unsigned long long ni = nin;
        const Args args(g, ni);
        // A row of at most 16 (32) values a chain's lane takes 16 (32) lanes an output, as the kernel's exact_lanes gives them (kernels/f32.gpu).
        const size_t wave = dev_.caps.subgroup_size, chains = kThreads / wave, lanes = nin <= 16 * chains ? 16 : nin <= 32 * chains && wave > 32 ? 32 : wave;
        const size_t per = kThreads / lanes;
        const auto kept = kept_;
        launch(kMatmulGroupF32, (unsigned)((most + per - 1) / per), (unsigned)n, args);
        kept_ = kept;
        for (size_t i = 0; i < n; ++i) forget_written(g.y[i], g.nout[i]);
        record_matrix_path(MatrixPath::f32);
        return true;
    }
    // One twin of X serves every projection of the group.
    void project_group(Items projections, CSlice X, size_t nin, size_t nbatch, RowRuns runs, Dtype dtype) {
        // Integer projections into device memory take one twin and up to six a product_twin call in the order given, which launches each three, or six where a pair build holds them.
        // Past three, a prompt tile gives each projection its own call's split, so a row keeps the bits of its own matmul; up to three share the split of their summed tiles.
        bool together = projections.size() > 1 && nin && nbatch;
        for (const Projection& p : projections)
            together = together && p.weight.data.buffer && integer(p.weight, dtype) && p.weight.type == projections.begin()->weight.type && p.rows &&
                       !as_gpu<Api>(*p.out.buffer).host_ptr();
        if (together) {
            Proj proj[6];
            size_t n = 0, left = projections.size();
            span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nin, nbatch), sizeof(float)));
            for (const Projection& p : projections) {
                check_weight(p.weight, nin, "matrix", 1);
                span(*p.weight.data.buffer, p.weight.data.offset * sizeof(float), quant::row_bytes(p.weight.type, nin, p.rows));
                span(*p.out.buffer, p.out.offset * sizeof(float), size_mul(size_mul(p.rows, nbatch), sizeof(float)));
                proj[n++] = {bytes(p.weight.data), f32(p.out), p.rows, scales(p.weight)};
                --left;
                if (n < 6 && left) continue;
                product_twin(*type_kernels(projections.begin()->weight.type)->twin, proj, n, X, nin, nbatch, false, runs, true, dtype == Dtype::int8, projections.size() > 3);
                n = 0;
            }
            return;
        }
        // Up to three exact projections of one type into disjoint device memory, every row on the column builds, take one column launch.
        const TypeKernels* kinds = projections.size() ? type_kernels(projections.begin()->weight.type) : nullptr;
        bool columns = kinds && projections.size() > 1 && projections.size() <= 3 && nin && nbatch && X.buffer;
        for (auto p = projections.begin(); columns && p != projections.end(); ++p) {
            columns = p->weight.data.buffer && p->weight.type == projections.begin()->weight.type && !integer(p->weight, dtype) && !p->weight.scales.buffer && p->rows && p->out.buffer &&
                      !as_gpu<Api>(*p->out.buffer).host_ptr() && all_columns(*kinds, p->weight, f32(X), nin, nbatch, runs, dtype) &&
                      !overlaps(CSlice(p->out), size_mul(p->rows, nbatch), X, size_mul(nin, nbatch));
            for (auto q = projections.begin(); columns && q != p; ++q) columns = !overlaps(CSlice(p->out), size_mul(p->rows, nbatch), CSlice(q->out), size_mul(q->rows, nbatch));
        }
        if (columns) {
            Proj proj[3];
            size_t n = 0;
            span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nin, nbatch), sizeof(float)));
            for (const Projection& p : projections) {
                check_weight(p.weight, nin, "matrix", 1);
                check_grid(p.weight, nin, p.rows, 1);
                span(*p.weight.data.buffer, p.weight.data.offset * sizeof(float), quant::row_bytes(p.weight.type, nin, p.rows));
                span(*p.out.buffer, p.out.offset * sizeof(float), size_mul(size_mul(p.rows, nbatch), sizeof(float)));
                proj[n++] = {bytes(p.weight.data), f32(p.out), p.rows};
            }
            exact_columns(*kinds, proj, n, f32(X), nin, nbatch, 0);
            record_matrix_path(MatrixPath::f32);
            return;
        }
        bool twinned = false;
        for (const Projection& p : projections) {
            if (!p.weight.data.buffer) throw std::runtime_error(Api::prefix + "projection without storage");
            product(p.weight, X, p.out, nin, p.rows, nbatch, false, runs, dtype, !twinned);
            twinned = twinned || integer(p.weight, dtype);
        }
    }
    // A group of two or three inputs at one generated column, each run of one input's projections cut into calls of at most three: two calls as one pair launch, three or four as one list launch, each call on its input's twin.
    // Two inputs' twins are found or built in the two slots, three must lie in kept twins, and each row keeps the bits of its own matmul; false where no pair or list build holds the group.
    bool inputs_list(Items projections, CSlice X, size_t nin, size_t nbatch, RowRuns runs, Dtype dtype) {
        const Projection* p = projections.first;
        const size_t n = projections.n;
        if (nbatch != 1 || !nin || n < 2) return false;
        // The calls' first projections and their inputs, and the group's distinct inputs.
        size_t first[kListCalls + 1], calls = 0, inputs = 0;
        CSlice in[kListCalls], distinct[kListCalls];
        for (size_t i = 0; i < n;) {
            const CSlice x = input_of(p[i], X);
            size_t j = i + 1;
            while (j < n && j - i < 3 && same_input(input_of(p[j], X), x)) ++j;
            if (calls == kListCalls || as_gpu<Api>(*x.buffer).host_ptr()) return false;
            if (!inputs || !same_input(distinct[inputs - 1], x)) {
                for (size_t d = 0; d < inputs; ++d)
                    if (same_input(distinct[d], x)) return false;
                distinct[inputs++] = x;
            }
            first[calls] = i, in[calls++] = x;
            i = j;
        }
        first[calls] = n;
        if (inputs < 2 || inputs > 3) return false;
        for (size_t i = 0; i < n; ++i)
            if (!p[i].weight.data.buffer || !integer(p[i].weight, dtype) || p[i].weight.type != p[0].weight.type || !p[i].rows || as_gpu<Api>(*p[i].out.buffer).host_ptr()) return false;
        const RowRun whole[1] = {{1, 1}};
        const RowRuns all = runs.n ? runs : RowRuns{whole, 1};
        for (size_t r = 0; r < all.n; ++r)
            if (!on_rows(all.runs[r].extent)) return false;
        const TwinKernels& tk = *type_kernels(p[0].weight.type)->twin;
        const bool int8 = dtype == Dtype::int8;
        const int form = !int8 ? 0 : nin / 32 % 2 ? 1 : 2;
        const TwinKernels& rk = row_form(tk, nin, int8);
        size_t total = 0;
        for (size_t i = 0; i < n; ++i) total += p[i].rows;
        const auto [build, per] = calls == 2 ? pair_build(rk, form, nin, total) : list_build(rk, form, nin, total);
        if (build == kKernels) return false;
        const TwinMap map = twin_map(all, 1, nin);
        if (inputs == 3)
            for (size_t d = 0; d < inputs; ++d)
                if (!twin_kept(f32(distinct[d]), nin / 32, int8, map)) return false;
        Proj proj[3 * kListCalls];
        for (size_t i = 0; i < n; ++i) {
            check_weight(p[i].weight, nin, "matrix", 1);
            span(*p[i].weight.data.buffer, p[i].weight.data.offset * sizeof(float), quant::row_bytes(p[i].weight.type, nin, p[i].rows));
            span(*p[i].out.buffer, p[i].out.offset * sizeof(float), size_mul(p[i].rows, sizeof(float)));
            proj[i] = {bytes(p[i].weight.data), f32(p[i].out), p[i].rows, scales(p[i].weight)};
        }
        half_span(form, 1, nin);
        size_t at = 0;
        unsigned xrow = 0, xblock = 0;
        map.locate(0, at, xrow, xblock);
        const size_t qb = int8 ? 32 : 64;
        // Each distinct input's twin in group order, so the first is found or built and kept before the second takes the other slot.
        TwinPlanes twin[kListCalls];
        for (size_t d = 0; d < inputs; ++d) {
            span(*distinct[d].buffer, distinct[d].offset * sizeof(float), size_mul(nin, sizeof(float)));
            twin[d] = twin_of(distinct[d], 1, nin, int8, map, true, false);
        }
        const auto twin_at = [&](size_t c) {
            size_t d = 0;
            while (!same_input(distinct[d], in[c])) ++d;
            return twin[d];
        };
        unsigned long long ni = nin;
        int add = 0;
        if (calls == 2) {
            const TwinPlanes ta = twin_at(0), tb = twin_at(1);
            rows_pair(rk, form, proj, n, first[1], 0, nin, 1, ta.q + at * qb, ta.d + at, tb.q + at * qb, tb.d + at, ni, add);
        } else {
            RowList l{};
            unsigned blocks = 0;
            for (size_t c = 0; c < calls; ++c) {
                const TwinPlanes t = twin_at(c);
                l.c[c] = row_call(proj + first[c], first[c + 1] - first[c], 0, rk, nin);
                l.q[c] = t.q + at * qb, l.dx[c] = t.d + at;
                if (c) l.start[c - 1] = blocks;
                blocks += (unsigned)((l.c[c].total + per - 1) / per);
            }
            for (size_t c = calls; c < kListCalls; ++c) l.start[c - 1] = blocks;
            launch(build, blocks, 1, Args(l, ni, add));
        }
        record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
        for (size_t i = 0; i < n; ++i) forget_written(proj[i].y, proj[i].nout);
        return true;
    }
    std::vector<Dtype> native_dtypes() const override { return {Dtype::f16, Dtype::f32, Dtype::int8}; }
    // An explicit bf16 takes BF16 weights' BF16 prompt tile where the module has it and F32 everywhere else, so auto never selects it (docs/PRECISION.md, section 2).
    bool emulates_dtype(Dtype dtype) const override { return dtype == Dtype::bf16 && kernels_[kBF16TileBF16]; }
    std::string dtype_path(Dtype dtype) const override {
        if (dtype == Dtype::f32) return "f32";
        if (dtype == Dtype::bf16) return "bf16 (BF16 tile inputs), f32 (other products, decode columns, routers)";
        // A half tile reads binary16 inputs; the types without twins take their exact products in every class but there, wider than asked, and say so.
        std::string exact = "F32 weights";
        for (const TypeKernels& t : kTypes)
            if (!t.twin && t.type != quant::GGML_TYPE_F32) exact += std::string(exact == "F32 weights" ? ", " : "/") + t.name;
        exact += exact == "F32 weights" ? "" : " rows";
        std::string twins;
        for (const TypeKernels& t : kTypes)
            if (t.twin) twins += std::string(twins.empty() ? "" : "/") + t.name;
        std::string halves;
        for (const TypeKernels& t : kTypes)
            if (t.half != kKernels && kernels_[t.half]) halves += std::string(halves.empty() ? "f16 (" : "/") + t.name;
        halves += halves.empty() ? "" : " tile inputs), ";
        const std::string form = dtype == Dtype::int8 ? "block-int8 (" : "block-int16 (";
        return halves + form + twins + (dev_.caps.integer_dot ? std::string(" rows, and tiles on the ") + (dtype == Dtype::int8 ? "8" : "16") + "-bit dot), f32 (" : " rows and tiles), f32 (") + exact + ")";
    }
    void embed(Slice dst, WeightView weight, size_t nin, size_t nrows, const uint32_t* ids, size_t count) override {
        const TypeKernels& kinds = check_weight(weight, nin, "embedding", 1);
        check_grid(weight, nin, nrows, 1);
        if (!count) return;
        if (nrows > UINT32_MAX) throw std::runtime_error(Api::prefix + "an embedding of more rows than 32-bit ids name");
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nrows));
        gather(kinds.embed, dst, bytes(weight.data), nin, (const unsigned*)arg(ids, count * sizeof(uint32_t)), count, nrows, &kinds, &weight);
    }
    // embed's gather with the ids read where the device holds them; the kernel writes a zero row for an id of `nrows` or more.
    void embed_ids(Slice dst, WeightView weight, size_t nin, size_t nrows, CSlice ids, size_t count) override {
        const TypeKernels& kinds = check_weight(weight, nin, "embedding", 1);
        check_grid(weight, nin, nrows, 1);
        if (!count) return;
        if (nrows > UINT32_MAX) throw std::runtime_error(Api::prefix + "an embedding of more rows than 32-bit ids name");
        if (!ids.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nrows));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(count, sizeof(float)));
        gather(kinds.embed, dst, bytes(weight.data), nin, (const unsigned*)f32(ids), count, nrows, &kinds, &weight);
    }
    // For each row of `n` logits the id of the largest (Backend::argmax_rows), in slices of the profile's argmax_slice ids merged a row; no ids is refused, as Vulkan refuses it.
    void argmax_rows(Slice ids, CSlice logits, size_t rows, size_t n, CSlice after = {}, size_t skip = SIZE_MAX) override {
        if (!rows) return;
        if (!n || n > UINT32_MAX) throw std::runtime_error(Api::prefix + "argmax over no ids or more than 32 bits name");
        if (!ids.buffer || !logits.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*logits.buffer, logits.offset * sizeof(float), size_mul(size_mul(rows, n), sizeof(float)));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(rows, sizeof(float)));
        if (after.buffer) span(*after.buffer, after.offset * sizeof(float), size_mul(rows, sizeof(float)));
        const size_t slice = std::max<size_t>(1, std::min(dev_.profile.argmax_slice, n)), nslice = (n + slice - 1) / slice;
        if (size_mul(rows, nslice) > UINT32_MAX) throw std::runtime_error(Api::prefix + "argmax over too many slices");
        const float* lp = f32(logits);
        unsigned* ip = (unsigned*)f32(ids);
        const unsigned* ap = after.buffer ? (const unsigned*)f32(after) : nullptr;
        const size_t need = argmax_parts_bytes(rows, n);
        if (nslice > 1 && (!argmax_parts_ || argmax_parts_->size() < need)) argmax_parts_ = alloc(need, Memory::device);
        float* pp = nslice > 1 ? (float*)as_gpu<Api>(*argmax_parts_).data() : nullptr;
        unsigned nn = (unsigned)n, sl = (unsigned)slice, ns = (unsigned)nslice, sk = (unsigned)std::min(skip, n);
        const Args args(lp, ip, pp, nn, sl, ns, ap, sk);
        launch(kArgmaxSlices, (unsigned)(rows * nslice), 1, args);
        if (nslice > 1) {
            const Args margs(pp, ip, nn, ns, ap);
            launch(kArgmaxMerge, (unsigned)rows, 1, margs, 1, 64);
        }
    }
    // argmax_rows' slices into partials, then a row's partials into its candidate (Backend::argmax_part).
    void argmax_part(Slice cands, CSlice logits, size_t rows, size_t n, size_t first, size_t slot, size_t slots, size_t skip = SIZE_MAX) override {
        if (!rows) return;
        const unsigned own = (unsigned)part_skip(skip, first, n);
        if (!n || slot >= slots || first + n > kExactIds || (n == 1 && own == 0))
            throw std::runtime_error(Api::prefix + "argmax_part over no ids, ids a float does not hold, or a slot past its row");
        if (!cands.buffer || !logits.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*logits.buffer, logits.offset * sizeof(float), size_mul(size_mul(rows, n), sizeof(float)));
        span(*cands.buffer, cands.offset * sizeof(float), size_mul(size_mul(rows, slots), 3 * sizeof(float)));
        const size_t slice = std::max<size_t>(1, std::min(dev_.profile.argmax_slice, n)), nslice = (n + slice - 1) / slice;
        if (size_mul(rows, nslice) > UINT32_MAX) throw std::runtime_error(Api::prefix + "argmax over too many slices");
        const size_t need = argmax_parts_bytes(rows, n);
        if (!argmax_parts_ || argmax_parts_->size() < need) argmax_parts_ = alloc(need, Memory::device);
        float* pp = (float*)as_gpu<Api>(*argmax_parts_).data();
        unsigned* none = nullptr;
        const unsigned* no_after = nullptr;
        unsigned nn = (unsigned)n, sl = (unsigned)slice, ns = (unsigned)nslice, f = (unsigned)first, at = (unsigned)slot, w = (unsigned)slots;
        const Args args(f32(logits), none, pp, nn, sl, ns, no_after, own);
        launch(kArgmaxSlices, (unsigned)(rows * nslice), 1, args);
        const Args pargs((const float*)pp, f32(cands), ns, f, at, w);
        launch(kArgmaxPart, (unsigned)rows, 1, pargs, 1, 64);
    }
    void argmax_pick(Slice ids, CSlice cands, size_t rows, size_t slots, size_t n, CSlice after = {}) override {
        if (!rows) return;
        if (!slots || n > kExactIds) throw std::runtime_error(Api::prefix + "argmax_pick over no candidates or ids a float does not hold");
        if (!ids.buffer || !cands.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*cands.buffer, cands.offset * sizeof(float), size_mul(size_mul(rows, slots), 3 * sizeof(float)));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(rows, sizeof(float)));
        if (after.buffer) span(*after.buffer, after.offset * sizeof(float), size_mul(rows, sizeof(float)));
        const unsigned* ap = after.buffer ? (const unsigned*)f32(after) : nullptr;
        unsigned nn = (unsigned)n, w = (unsigned)slots;
        const Args args(f32(cands), (unsigned*)f32(ids), nn, w, ap);
        launch(kArgmaxPick, (unsigned)rows, 1, args, 1, 64);
    }
    // A block a row's run of columns, its taps summed in order by fused multiply-adds as the CPU sums them, then added to the row.
    void mix_rows(Slice dst, CSlice table, size_t width, size_t table_rows, const uint32_t* idx, const float* weight, size_t rows, size_t taps) override {
        if (!width || !taps) throw std::runtime_error(Api::prefix + "invalid row mix dimensions");
        if (!rows) return;
        const size_t picks = size_mul(rows, taps);
        if (!idx || !weight) throw std::runtime_error(Api::prefix + "a row mix without its taps");
        for (size_t i = 0; i < picks; ++i)
            if (idx[i] >= table_rows) throw std::runtime_error(Api::prefix + "a row mix names a row past its table");
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(rows, width), sizeof(float)));
        span(*table.buffer, table.offset * sizeof(float), size_mul(size_mul(table_rows, width), sizeof(float)));
        const Args args(f32(dst), f32(table), (const unsigned*)arg(idx, picks * sizeof(uint32_t)), (const float*)arg(weight, picks * sizeof(float)),
                        (unsigned long long)width, (unsigned)taps);
        launch(kMixRows, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)rows, args);
    }
    // The vision encoder's ops (docs/VISION.md, section 3): a block a row for the norm and the bias, a thread an element for gelu.
    void layer_norm_rows(Slice dst, CSlice src, CSlice w, CSlice b, size_t rows, size_t n, size_t stride, float eps) override {
        if (!n || stride < n || n > INT32_MAX || !std::isfinite(eps) || eps < 0.0f) throw std::runtime_error(Api::prefix + "invalid layer norm dimensions");
        if (!rows) return;
        const size_t last = size_add(size_mul(rows - 1, stride), n);
        for (const CSlice& x : {CSlice(dst), src}) span(*x.buffer, x.offset * sizeof(float), size_mul(last, sizeof(float)));
        for (const CSlice& x : {w, b}) span(*x.buffer, x.offset * sizeof(float), size_mul(n, sizeof(float)));
        int nn = (int)n;
        unsigned long long st = stride;
        const Args args(f32(dst), f32(src), f32(w), f32(b), nn, st, eps);
        launch(kLayerNormRows, (unsigned)rows, 1, args);
    }
    void gelu(Slice dst, CSlice src, size_t n, GeluForm form) override {
        if (!n) return;
        for (const CSlice& x : {CSlice(dst), src}) span(*x.buffer, x.offset * sizeof(float), size_mul(n, sizeof(float)));
        unsigned long long nn = n;
        int erf_form = form == GeluForm::erf;
        const Args args(f32(dst), f32(src), nn, erf_form);
        launch(kGelu, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
    }
    void add_bias(Slice dst, CSlice bias, size_t rows, size_t width, CSlice scale = {}) override {
        if (!width || width > UINT32_MAX || rows > UINT32_MAX) throw std::runtime_error(Api::prefix + "invalid bias dimensions");
        if (!rows) return;
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(rows, width), sizeof(float)));
        span(*bias.buffer, bias.offset * sizeof(float), size_mul(width, sizeof(float)));
        if (scale.buffer) span(*scale.buffer, scale.offset * sizeof(float), size_mul(width, sizeof(float)));
        unsigned wd = (unsigned)width;
        const float* sc = scale.buffer ? f32(scale) : nullptr;
        const Args args(f32(dst), f32(bias), sc, wd);
        launch(kAddBias, (unsigned)rows, 1, args);
    }
    // The elementwise row ops: a thread a float of a row, the grid's height stepping over rows (kernels/f32.gpu).
    void copy_rows(Slice dst, size_t dst_stride, CSlice src, size_t src_stride, size_t rows, size_t width, bool add) override {
        if (!width || width > UINT32_MAX || dst_stride < width || src_stride < width) throw std::runtime_error(Api::prefix + "invalid row copy dimensions");
        if (!rows) return;
        const size_t dn = size_add(size_mul(rows - 1, dst_stride), width), sn = size_add(size_mul(rows - 1, src_stride), width);
        floats(dst, dn);
        floats(src, sn);
        float* d = f32(dst);
        unsigned long long ds = dst_stride, ss = src_stride, nr = rows;
        unsigned wd = (unsigned)width;
        int a = add;
        const Args args(d, f32(src), ds, ss, nr, wd, a);
        launch_writing({{d, dn}}, kCopyRows, u32((width + kThreads - 1) / kThreads), (unsigned)std::min<size_t>(rows, 65535), args);
    }
    void softcap(Slice x, size_t n, float cap) override {
        if (!(cap > 0.0f) || !std::isfinite(cap)) throw std::runtime_error(Api::prefix + "a softcap that is not positive and finite");
        if (!n) return;
        floats(x, n);
        float* d = f32(x);
        unsigned long long nn = n;
        const Args args(d, nn, cap);
        launch_writing({{d, n}}, kSoftcap, u32((n + kThreads - 1) / kThreads), 1, args);
    }
    // The host's copies of llmx_normed_gate_mix's item and call (kernels/f32.gpu, GateStep and GateSteps), field for field.
    struct GateStep {
        const float *h, *m, *d0, *d1;
        const unsigned char *down, *forget, *gain;
        float *low, *mix, *out;
        const float* norm;
        void* q;
        float* d;
        void* s;
    };
    struct GateSteps {
        GateStep it[3];
        const float* down_w;
        unsigned* bar;
        unsigned width, cols, split;
        float eps;
        int normed;
    };
    static_assert(sizeof(GateStep) == 112 && sizeof(GateSteps) == 376 && offsetof(GateSteps, normed) == 368, "GateSteps is the kernel's layout");
    static constexpr size_t kGateRank = 64, kGateCols = 144;   // kernels/f32.gpu, kGateRank and 16 groups of kGatePasses
    // Items with down steps at one row as one launch of llmx_normed_gate_mix (kernels/f32.gpu): F32 down, forget and gain weights of rank kGateRank and widths of 256 to 2304 in steps of 256.
    // Blocks of an item meet at grid barriers, so the grid is as many blocks as the compute units hold at one each; with norms the outputs lie one after another and keep one twin (docs/src/backends-gpu.md).
    bool gate_steps(std::initializer_list<GateMixItem> items, size_t rows, size_t width, RowRuns runs, float eps) {
        const size_t n = items.size(), packed = width;
        const GateMixItem* it = items.begin();
        const size_t cu = std::max(1u, dev_.caps.compute_units), B = 32 * n <= cu ? 32 : 16;
        if (rows != 1 || !n || n > 3 || dev_.caps.subgroup_size != 64 || width % kThreads || width / kThreads > kNormHeldSteps || width / 16 > kGateCols || B * n > cu ||
            !std::isfinite(eps) || eps <= 0.0f)
            return false;
        const bool normed = it[0].norm.buffer != nullptr;
        NormPlan p[3];
        uintptr_t bits = 0;
        for (size_t i = 0; i < n; ++i) {
            const GateMixItem& g = it[i];
            const NormProduct& d = g.down.into;
            if (!d.weight.data.buffer || d.weight.type != quant::GGML_TYPE_F32 || d.nout != kGateRank || !d.y.buffer || as_gpu<Api>(*d.y.buffer).host_ptr() ||
                !same_input(g.down_w, it[0].down_w) || !g.forget.weight.data.buffer || !g.gain.weight.data.buffer || g.forget.weight.type != quant::GGML_TYPE_F32 ||
                g.gain.weight.type != quant::GGML_TYPE_F32 || g.rank != kGateRank || g.forget.rows != width || g.gain.rows != width || !same_input(g.forget.in, d.y) ||
                !same_input(g.gain.in, d.y) || !g.norm.buffer != !normed || as_gpu<Api>(*g.out.buffer).host_ptr())
                return false;
            p[i] = norm_plan(g.down, it[0].down_w, 1, 2, width);
            if (p[i].split != p[0].split) return false;
            bits |= (uintptr_t)p[i].s | (uintptr_t)p[i].s2 | (uintptr_t)f32(g.down.into.y) | (uintptr_t)bytes(d.weight.data) | (uintptr_t)bytes(g.forget.weight.data) |
                    (uintptr_t)bytes(g.gain.weight.data);
        }
        // What an item writes meets nothing another item reads or writes, nor its own inputs but h at exactly its out where there is no norm.
        for (size_t i = 0; i < n; ++i) {
            const GateMixItem& g = it[i];
            const CSlice low = g.down.into.y, mix = normed ? g.f : CSlice(g.out);
            for (size_t j = 0; j < n; ++j) {
                const GateMixItem& o = it[j];
                const CSlice reads[6] = {o.h, o.m, CSlice(o.down.into.y), p[j].an ? o.down.src : CSlice{}, p[j].bn ? p[j].rest : CSlice{}, o.norm};
                const size_t extents[6] = {packed, packed, kGateRank, p[j].an, p[j].bn, o.norm.buffer ? packed : 0};
                for (size_t r = 0; r < 6; ++r) {
                    if (!reads[r].buffer || !extents[r]) continue;
                    if (r != 2 && overlaps(low, kGateRank, reads[r], extents[r])) return false;
                    if (r == 2 && j != i && overlaps(low, kGateRank, reads[r], extents[r])) return false;
                    // An item's own down sources are read before its first barrier, so its mix may meet them; its low row is read by every block after it.
                    const bool own = j == i && (r == 3 || r == 4 || (r == 0 && !normed && same_input(mix, o.h)));
                    if (!own && overlaps(mix, packed, reads[r], extents[r])) return false;
                    if (normed && overlaps(g.out, packed, reads[r], extents[r])) return false;
                }
                if (j != i && (overlaps(mix, packed, normed ? o.f : CSlice(o.out), packed) || overlaps(g.out, packed, o.out, packed))) return false;
            }
            if (normed && overlaps(g.out, packed, mix, packed)) return false;
        }
        TwinOut t{};
        if (normed) {
            for (size_t i = 0; i < n; ++i)
                if (it[i].out.buffer != it[0].out.buffer || it[i].out.offset != it[0].out.offset + i * packed) return false;
            const TwinMap all = twin_map(runs, 1, size_mul(n, width));
            if (all.regions && !all.flat()) return false;
            t = twin_out(it[0].out, size_mul(n, packed), all);
        }
        if (bits % 16) return false;
        GateSteps a{};
        const size_t qsize = t.int8 ? 1 : 2, blocks = packed / 32;
        for (size_t i = 0; i < n; ++i) {
            const GateMixItem& g = it[i];
            check_weight(g.down.into.weight, 2 * width, "matrix", 1);
            check_grid(g.down.into.weight, 2 * width, kGateRank, 1);
            span(*g.down.into.weight.data.buffer, g.down.into.weight.data.offset * sizeof(float), quant::row_bytes(quant::GGML_TYPE_F32, 2 * width, kGateRank));
            span(*g.down.into.y.buffer, g.down.into.y.offset * sizeof(float), kGateRank * sizeof(float));
            for (const Projection* pr : {&g.forget, &g.gain}) {
                check_weight(pr->weight, kGateRank, "matrix", 1);
                check_grid(pr->weight, kGateRank, width, 1);
                span(*pr->weight.data.buffer, pr->weight.data.offset * sizeof(float), quant::row_bytes(quant::GGML_TYPE_F32, kGateRank, width));
            }
            a.it[i] = {f32(g.h), f32(g.m), p[i].s, p[i].s2, bytes(g.down.into.weight.data), bytes(g.forget.weight.data), bytes(g.gain.weight.data), f32(g.down.into.y),
                       normed ? (float*)f32(g.f) : f32(g.out), f32(g.out), normed ? f32(g.norm) : nullptr, !t.q ? nullptr : (uint8_t*)t.q + i * packed * qsize,
                       !t.q ? nullptr : t.d + i * blocks, !t.s ? nullptr : (uint8_t*)t.s + i * blocks * 4 * qsize};
        }
        a.down_w = f32(it[0].down_w), a.bar = step_counter() + 16, a.width = unsigned(width), a.cols = unsigned(width / B), a.split = p[0].split, a.eps = eps, a.normed = normed;
        const Kernel k = !t.q ? kNormedGateMix : t.int8 ? kNormedGateMixTwin8 : kNormedGateMixTwin16;
        const Args args(a);
        launch_writing({{a.it[0].out, normed ? n * packed : 0}, {a.it[0].mix, packed}, {a.it[0].low, kGateRank}, {a.it[1].mix, n > 1 ? packed : 0}, {a.it[1].low, n > 1 ? kGateRank : 0},
                        {a.it[2].mix, n > 2 ? packed : 0}, {a.it[2].low, n > 2 ? kGateRank : 0}},
                       k, unsigned(B), unsigned(n), args);
        if (t.q) keep_twin({a.it[0].out, size_mul(n, packed), t.int8, t.map, t.slot, t.s != nullptr, true});
        for (size_t i = 0; i < 2 * n; ++i) record_matrix_path(MatrixPath::f32);
        return true;
    }
    // With norms, rows of up to 4096 values lying row after row and apart from every item's inputs take the held norm with the mix as its values, up to three items whose outputs lie one after another a launch (kernels/f32.gpu, llmx_tanh_norm_16).
    void tanh_gate_mix(std::initializer_list<GateMixItem> items, size_t rows, size_t width, size_t out_stride, size_t h_stride, size_t m_stride, RowRuns runs, float eps) override {
        if (!width || width > UINT32_MAX || out_stride < width || h_stride < width || m_stride < width) throw std::runtime_error(Api::prefix + "invalid gate mix dimensions");
        if (!rows) return;
        const auto strided = [&](size_t stride) { return size_add(size_mul(rows - 1, stride), width); };
        const size_t packed = size_mul(rows, width);
        for (const GateMixItem& it : items) {
            floats(it.out, strided(out_stride)), floats(it.h, strided(h_stride)), floats(it.m, strided(m_stride));
            for (const CSlice& v : {it.f, it.g}) floats(v, packed);
            if (it.norm.buffer) floats(it.norm, width);
        }
        if (gate_steps(items, rows, width, runs, eps)) return;
        gate_downs(items, rows, width, eps, runs);
        if (mix_tile(items, rows, width, out_stride, h_stride, m_stride, runs, eps)) return;
        gate_products(items, rows, runs);
        const TwinMap map = twin_map(runs, rows, width);
        bool held = out_stride == width && width % 32 == 0 && width <= 16 * kThreads && (!map.regions || map.flat());
        for (const GateMixItem& it : items) {
            held = held && it.norm.buffer;
            for (const GateMixItem& o : items)
                held = held && !overlaps(it.out, packed, o.h, strided(h_stride)) && !overlaps(it.out, packed, o.m, strided(m_stride)) && !overlaps(it.out, packed, o.f, packed) &&
                       !overlaps(it.out, packed, o.g, packed) && (&o == &it || !overlaps(it.out, packed, o.out, packed));
        }
        if (held) {
            // Up to three items whose outputs lie one after another in device memory share a launch and one twin of all their rows.
            const TwinMap all = twin_map(runs, rows, size_mul(items.size(), width));
            bool along = items.size() <= kHeldMixItems && (!all.regions || all.flat());
            for (const GateMixItem& it : items)
                along = along && it.out.buffer == items.begin()->out.buffer && it.out.offset == items.begin()->out.offset + size_t(&it - items.begin()) * packed &&
                        !as_gpu<Api>(*it.out.buffer).host_ptr();
            if (along) return held_mix(items.begin(), items.size(), rows, width, h_stride, m_stride, runs, eps);
            for (const GateMixItem& it : items) held_mix(&it, 1, rows, width, h_stride, m_stride, runs, eps);
            return;
        }
        for (const GateMixItem& it : items) gate_mix(it.out, it.h, it.m, it.f, it.g, rows, width, out_stride, h_stride, m_stride, runs, it.norm, eps);
    }
    // Items whose forget and gain are wide F32 weights of one input take f32_tile's mix build on prompt runs the tile cannot split, without writing f or g.
    // Their norms then run in place in item order after every mix, so the k and v twins take the two slots as the held pair's do (kernels/int16.gpu, llmx_f32_tile_mix).
    bool mix_tile(std::initializer_list<GateMixItem> items, size_t rows, size_t width, size_t out_stride, size_t h_stride, size_t m_stride, RowRuns runs, float eps) {
        const RowRun whole[1] = {{rows, 2}};
        const RowRuns all = runs.n ? runs : RowRuns{whole, 1};
        bool tile = out_stride == width && items.size();
        for (size_t r = 0; r < all.n; ++r) tile = tile && !on_rows(all.runs[r].extent);
        const size_t packed = size_mul(rows, width);
        for (const GateMixItem& it : items) {
            const Projection &f = it.forget, &g = it.gain;
            const size_t nblk = it.rank / 32;
            tile = tile && f.weight.data.buffer && g.weight.data.buffer && f.weight.type == quant::GGML_TYPE_F32 && g.weight.type == quant::GGML_TYPE_F32 && f.weight.wide && g.weight.wide && f.rows == width &&
                   g.rows == width && nblk && it.rank % 32 == 0 && f.in.buffer && same_input(f.in, g.in) && !as_gpu<Api>(*it.out.buffer).host_ptr() &&
                   split_blocks(dev_.profile, dev_.caps, 1, nblk, dev_.profile.float_tile_split_per_cu) == nblk && !overlaps(it.out, packed, f.in, size_mul(rows, it.rank));
        }
        if (!tile) return false;
        for (const GateMixItem& it : items) {
            for (const Projection* p : {&it.forget, &it.gain}) {
                check_weight(p->weight, it.rank, "matrix", 1);
                check_grid(p->weight, it.rank, width, 1);
                span(*p->weight.data.buffer, p->weight.data.offset * sizeof(float), quant::row_bytes(quant::GGML_TYPE_F32, it.rank, width));
            }
            span(*it.forget.in.buffer, it.forget.in.offset * sizeof(float), size_mul(size_mul(rows, it.rank), sizeof(float)));
            TileCall c{};
            c.w[0] = bytes(it.forget.weight.data), c.w[1] = bytes(it.gain.weight.data), c.y[0] = f32(it.out), c.nout[0] = c.nout[1] = width, c.nproj = 2, c.nin = it.rank, c.nrows = rows;
            c.kper = (unsigned)(it.rank / 32), c.parts = 1;
            const TileMix mx{f32(it.h), f32(it.m), h_stride, m_stride};
            const Args args(c, (const float*)f32(it.forget.in), mx);
            launch_writing({{c.y[0], packed}}, kF32TileMix, u32((width + kTileOut - 1) / kTileOut), u32((rows + kTileRows - 1) / kTileRows), args);
        }
        for (const GateMixItem& it : items)
            if (it.norm.buffer) rms_norm_rows(it.out, it.out, it.norm, rows, width, out_stride, eps, runs);
        record_matrix_path(MatrixPath::f32);
        return true;
    }
    // The host's copy of f32_tile's mix operands (kernels/int16.gpu, TileMix), field for field.
    struct TileMix {
        const float* h;
        const float* m;
        unsigned long long h_stride, m_stride;
    };
    static_assert(sizeof(TileMix) == 32 && offsetof(TileMix, m_stride) == 24, "TileMix is the kernel's layout");
    static constexpr size_t kHeldMixItems = 3;   // kernels/f32.gpu, LLMX_NORM_MIX's items
    // One to three items of the held mix norm, their outputs one after another, as one launch, blockIdx.y the item, and one twin of all their rows kept, which a product of any one item's rows reads (find_twin).
    void held_mix(const GateMixItem* items, size_t n, size_t rows, size_t width, size_t h_stride, size_t m_stride, RowRuns runs, float eps) {
        const size_t packed = size_mul(rows, width);
        const TwinOut t = twin_out(items[0].out, size_mul(n, packed), twin_map(runs, rows, size_mul(n, width)));
        const size_t qsize = t.int8 ? 1 : 2, blocks = packed / 32;
        GateNorm g[kHeldMixItems]{};
        for (size_t i = 0; i < n; ++i) {
            const GateMixItem& it = items[i];
            g[i] = {f32(it.out), {f32(it.h), f32(it.m), f32(it.f), f32(it.g), h_stride, m_stride}, f32(it.norm), !t.q ? nullptr : (uint8_t*)t.q + i * packed * qsize,
                    !t.q ? nullptr : t.d + i * blocks, !t.s ? nullptr : (uint8_t*)t.s + i * blocks * 4 * qsize};
        }
        const Kernel k = !t.q ? kTanhNorm16 : t.int8 ? kTanhNorm16Twin8 : kTanhNorm16Twin16;
        int in = (int)width;
        const Args args(g[0], g[1], g[2], in, eps);
        // One part a row, since each part would recompute the whole row's mix for its sum (docs/src/backends-gpu.md).
        launch_writing({{g[0].dst, size_mul(n, packed)}}, k, (unsigned)rows, (unsigned)n, args, 1, kNormThreads);
        if (t.q) keep_twin({g[0].dst, size_mul(n, packed), t.int8, t.map, t.slot, t.s != nullptr, true});
    }
    void gate_mix(Slice out, CSlice h, CSlice m, CSlice f, CSlice g, size_t rows, size_t width, size_t out_stride, size_t h_stride, size_t m_stride, RowRuns runs, CSlice norm,
                  float eps) {
        const auto strided = [&](size_t stride) { return size_add(size_mul(rows - 1, stride), width); };
        float* o = f32(out);
        unsigned long long nr = rows, os = out_stride, hs = h_stride, ms = m_stride;
        unsigned wd = (unsigned)width;
        const Args args(o, f32(h), f32(m), f32(f), f32(g), nr, wd, os, hs, ms);
        launch_writing({{o, strided(out_stride)}}, kTanhGateMix, u32((width + kThreads - 1) / kThreads), (unsigned)std::min<size_t>(rows, 65535), args);
        if (norm.buffer) rms_norm_rows(out, out, norm, rows, width, out_stride, eps, runs);
    }
    // The host's copy of llmx_tanh_norm_16's mix argument (kernels/f32.gpu, GateMix), field for field.
    struct GateMix {
        const float* h;
        const float* m;
        const float* f;
        const float* g;
        unsigned long long h_stride, m_stride;
    };
    static_assert(sizeof(GateMix) == 48 && offsetof(GateMix, m_stride) == 40, "GateMix is the kernel's layout");
    // The host's copy of llmx_tanh_norm_16's item (kernels/f32.gpu, GateNorm), field for field.
    struct GateNorm {
        float* dst;
        GateMix mx;
        const float* w;
        void* q;
        float* d;
        void* s;
    };
    static_assert(sizeof(GateNorm) == 88 && offsetof(GateNorm, s) == 80, "GateNorm is the kernel's layout");
    // Every item in one launch, a block a (row, head) of all the items' heads in order, or a segment of lanes a (row, head) where every head fits the lanes build and its blocks fill every compute unit (docs/src/backends-gpu.md).
    void l2_clip(std::initializer_list<ClipRows> items, size_t rows, RowRuns) override {
        if (items.size() > kClipItems) throw std::runtime_error(Api::prefix + "invalid l2 clip dimensions");
        ClipArgs a{};
        std::pair<const void*, size_t> written[kClipItems] = {};
        size_t heads = 0;
        for (const ClipRows& it : items) {
            const size_t width = size_mul(it.heads, it.dim);
            if (!it.heads || !it.dim || it.dim > UINT32_MAX || it.stride < width || !std::isfinite(it.threshold)) throw std::runtime_error(Api::prefix + "invalid l2 clip dimensions");
            heads += it.heads;
            if (heads > 65535) throw std::runtime_error(Api::prefix + "invalid l2 clip dimensions");
            if (!rows) continue;
            const size_t n = size_add(size_mul(rows - 1, it.stride), width);
            floats(it.x, n);
            if (it.scale.buffer) floats(it.scale, width);
            if (it.row_scale.buffer) floats(it.row_scale, rows);
            ClipItem& c = a.item[a.n];
            c = {f32(it.x), it.scale.buffer ? f32(it.scale) : nullptr, it.row_scale.buffer ? f32(it.row_scale) : nullptr, it.stride, (unsigned)it.dim, (unsigned)it.heads,
                 it.threshold, 0};
            written[a.n++] = {c.x, n};
        }
        if (!rows || !a.n) return;
        // As launch_writing, for a count of writes known only here.
        const auto kept = kept_;
        size_t pow2 = 1;
        for (unsigned i = 0; i < a.n; ++i)
            while (pow2 < a.item[i].dim) pow2 *= 2;
        const size_t lanes = std::min<size_t>(pow2, dev_.caps.subgroup_size), units = size_mul(rows, heads), per = kThreads / lanes, blocks = (units + per - 1) / per;
        if (pow2 <= 128 && units <= UINT32_MAX - kThreads && blocks >= std::max(1u, dev_.caps.compute_units))
            launch(kL2ClipLanes, u32(blocks), 1, Args(a, (unsigned)heads, (unsigned)lanes, (unsigned)(pow2 / lanes), (unsigned)units));
        else
            launch(kL2Clip, u32(rows), (unsigned)heads, Args(a));
        kept_ = kept;
        for (unsigned i = 0; i < a.n; ++i) forget_written((const float*)written[i].first, written[i].second);
    }
    // The host's copy of llmx_query_mix's argument (kernels/f32.gpu, QueryMixArgs), field for field.
    struct QueryMixItem {
        float* out;
        const float* src;
        const float* queries;
        unsigned n, nq;
    };
    struct QueryMixArgs {
        QueryMixItem item[kQueryMixItems];
        unsigned long long stride;
        unsigned dim, count;
        float eps, norm_eps;
        void* tq;
        float* td;
        void* ts;
        int level;
        unsigned parts;
        const float* w[kQueryMixNorms];
    };
    static_assert(sizeof(QueryMixItem) == 32 && offsetof(QueryMixArgs, stride) == 32 * kQueryMixItems && offsetof(QueryMixArgs, eps) == 32 * kQueryMixItems + 16 &&
                      offsetof(QueryMixArgs, tq) == 32 * kQueryMixItems + 24 && offsetof(QueryMixArgs, parts) == 32 * kQueryMixItems + 52 && offsetof(QueryMixArgs, w) == 32 * kQueryMixItems + 56,
                  "QueryMixArgs is the kernel's layout");
    // Every item in one launch, a block a (row, query) of all the items' queries in order (kernels/f32.gpu).
    // A lone normed item of a held row is normed in its block with the twin a product reads next (llmx_query_mix_norm), the planes of several queries one twin of parts; other normed items take rms_norm_rows after the mixes.
    void query_mix(std::initializer_list<QueryMixRows> items, size_t rows, size_t dim, size_t stride, float eps, RowRuns runs) override {
        std::pair<const void*, size_t> written[kQueryMixItems] = {};
        size_t queries = 0;
        QueryMixArgs a = query_mix_args(items, rows, dim, stride, eps, written, queries);
        if (!rows) return;
        const QueryMixRows& first = *items.begin();
        const TwinMap map = twin_map(runs, rows, dim);
        const size_t nq = first.nq;
        if (items.size() == 1 && first.norm[0].buffer && first.n <= kQueryMixNormItems && dim <= kQueryMixHeld && (!map.regions || map.flat())) {
            // Several queries' planes are one twin of parts whose rows lie query after query, so only a single row has it serve a product of one plane.
            const bool plane_twin = nq == 1 || rows == 1;
            const TwinOut t = dim % 32 == 0 && plane_twin ? twin_out(first.out, size_mul(rows, size_mul(nq, dim)), nq == 1 ? map : twin_map(runs, rows, size_mul(nq, dim))) : TwinOut{};
            for (size_t q = 0; q < nq; ++q) a.w[q] = f32(first.norm[q]);
            a.norm_eps = first.norm_eps;
            a.tq = t.q, a.td = t.d, a.ts = t.s, a.level = !t.q ? 0 : t.int8 ? 2 : 1;
            // A row runs in parts of kThreads values, at most one block a compute unit, as rms_norm_rows' parts.
            a.parts = (unsigned)std::max<size_t>(1, std::min((dim + kThreads - 1) / kThreads, std::max(1u, dev_.caps.compute_units) / size_mul(rows, nq)));
            launch_writing({written[0]}, kQueryMixNorm, u32(rows), (unsigned)(nq * a.parts), Args(a));
            if (t.q) keep_twin({f32(first.out), size_mul(rows, size_mul(nq, dim)), t.int8, t.map, t.slot, t.s != nullptr, nq > 1});
            return;
        }
        // As launch_writing, for a count of writes known only here.
        const auto kept = kept_;
        const unsigned count = a.count;
        // With a row a compute unit or more, a held mix of at most kQueryMixRowsItems items takes the row form, a launch an item whose block loads a row once for all its queries.
        if (dim <= kQueryMixHeld && rows >= std::max(1u, dev_.caps.compute_units)) {
            static constexpr Kernel row_form[kQueryMixRowsItems] = {kQueryMixRows1, kQueryMixRows2, kQueryMixRows3, kQueryMixRows4, kQueryMixRows5};
            unsigned rest = 0;
            for (unsigned i = 0; i < a.count; ++i) {
                if (a.item[i].n > kQueryMixRowsItems) {
                    a.item[rest++] = a.item[i];
                    continue;
                }
                QueryMixArgs one = a;
                one.item[0] = a.item[i], one.count = 1;
                launch(row_form[a.item[i].n - 1], u32(rows), 1, Args(one));
                queries -= a.item[i].nq;
            }
            a.count = rest;
        }
        // Mixes of at most kQueryMixFew items each take the build of that many, whose registers allow four waves a SIMD where the full build holds one.
        bool few = true;
        for (unsigned i = 0; i < a.count; ++i) few = few && a.item[i].n <= kQueryMixFew;
        if (a.count) launch(few ? kQueryMix4 : kQueryMix, u32(rows), (unsigned)queries, Args(a));
        kept_ = kept;
        for (unsigned i = 0; i < count; ++i) forget_written((const float*)written[i].first, written[i].second);
        for (const QueryMixRows& it : items)
            for (size_t q = 0; it.norm[0].buffer && q < it.nq; ++q) rms_norm_rows(Slice{it.out.buffer, it.out.offset + q * dim}, Slice{it.out.buffer, it.out.offset + q * dim}, it.norm[q], rows, dim, it.nq * dim, it.norm_eps, runs);
    }
    // query_mix's checks and its kernel argument, each item's output and its floats in `written`, and the items' queries counted in `queries`.
    QueryMixArgs query_mix_args(std::initializer_list<QueryMixRows> items, size_t rows, size_t dim, size_t stride, float eps, std::pair<const void*, size_t>* written, size_t& queries) {
        if (!items.size() || items.size() > kQueryMixItems || !dim || dim > UINT32_MAX || !(eps >= 0.0f)) throw std::runtime_error(Api::prefix + "invalid query mix dimensions");
        QueryMixArgs a{};
        a.stride = stride;
        a.dim = (unsigned)dim;
        a.eps = eps;
        for (const QueryMixRows& it : items) {
            if (!it.n || it.n > 16 || !it.nq || stride < size_mul(it.n, dim)) throw std::runtime_error(Api::prefix + "invalid query mix dimensions");
            if (it.norm[0].buffer && (it.nq > kQueryMixNorms || !it.norm[it.nq - 1].buffer)) throw std::runtime_error(Api::prefix + "a normed query mix without a norm for each query");
            queries += it.nq;
            if (queries > 65535) throw std::runtime_error(Api::prefix + "invalid query mix dimensions");
            if (!rows) continue;
            const size_t on = size_mul(rows, size_mul(it.nq, dim));
            floats(it.out, on);
            floats(it.src, size_add(size_mul(rows - 1, stride), size_mul(it.n, dim)));
            floats(it.queries, size_mul(it.nq, dim));
            for (size_t q = 0; it.norm[0].buffer && q < it.nq; ++q) floats(it.norm[q], dim);
            for (const QueryMixRows& o : items) {
                bool hits = overlaps(it.out, on, o.src, size_add(size_mul(rows - 1, stride), size_mul(o.n, dim))) || overlaps(it.out, on, o.queries, size_mul(o.nq, dim)) ||
                            (&o != &it && overlaps(it.out, on, o.out, size_mul(rows, size_mul(o.nq, dim))));
                for (size_t q = 0; o.norm[0].buffer && q < o.nq; ++q) hits = hits || overlaps(it.out, on, o.norm[q], dim);
                if (hits) throw std::runtime_error(Api::prefix + "a query mix whose output overlaps its items");
            }
            a.item[a.count] = {f32(it.out), f32(it.src), f32(it.queries), (unsigned)it.n, (unsigned)it.nq};
            written[a.count] = {a.item[a.count].out, on};
            ++a.count;
        }
        return a;
    }
    void gather_rows(Slice dst, CSlice src, size_t width, const uint32_t* rows, size_t count) override {
        if (count && !rows) throw std::runtime_error(Api::prefix + "gather without rows");
        if (!count) return;
        span(*src.buffer, src.offset * sizeof(float), size_mul(size_mul(size_add(*std::max_element(rows, rows + count), 1), width), sizeof(float)));
        gather(kGather, dst, f32(src), width, (const unsigned*)arg(rows, count * sizeof(uint32_t)), count, UINT32_MAX);
    }
    void rms_norm_rows(Slice dst, CSlice src, CSlice w, size_t rows, size_t n, size_t stride, float eps, RowRuns runs) override {
        if (!rows) return;
        const size_t last = size_add(size_mul(rows - 1, stride), n);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(last, sizeof(float)));
        span(*src.buffer, src.offset * sizeof(float), size_mul(last, sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(n, sizeof(float)));
        float* d = f32(dst);
        const float* s = f32(src);
        const float* wp = f32(w);
        int in = (int)n;
        unsigned long long st = stride;
        if (n > 32 * kThreads) {
            const Args args(d, s, wp, in, st, eps);
            return launch(kRmsNorm, (unsigned)rows, 1, args);
        }
        // Rows of up to 8192 values take a held build, which writes the twin beside them where a product reads one next.
        const TwinOut t = stride == n && n % 32 == 0 ? twin_out(dst, rows * n, twin_map(runs, rows, n)) : TwinOut{};
        const bool wide = n > 16 * kThreads;
        const Kernel k = !t.q ? (wide ? kRmsNorm32 : kRmsNorm16) : t.int8 ? (wide ? kRmsNorm32Twin8 : kRmsNorm16Twin8) : (wide ? kRmsNorm32Twin16 : kRmsNorm16Twin16);
        void* q = t.q;
        float* td = t.d;
        TwinMap map = t.map;
        const Kernel build = twin_build(k, map);
        // A row runs in parts of about kNormPart values, at most one block a compute unit, each repeating the row's sum and writing its own share in whole twin blocks (docs/src/backends-gpu.md, kernels/f32.gpu).
        // In place it runs as one part, since a part's writes would reach another's sum.
        const size_t units = std::max(1u, dev_.caps.compute_units) / rows;
        const unsigned parts = overlaps(CSlice(dst), rows * stride, src, rows * stride) ? 1 : (unsigned)std::max<size_t>(1, std::min((n + kNormPart - 1) / kNormPart, units));
        Args args(d, s, wp, in, st, eps, parts, q, td, t.s);
        if (build != k) args.put(map);
        launch_writing({{d, size_mul(rows, std::max<size_t>(n, stride))}}, build, (unsigned)rows, parts, args, 1, parts > 1 && build == k ? kThreads : kNormThreads);
        keep(t, d, rows * n);
    }
    void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride, size_t heads, size_t head_dim,
                           size_t rope_dim, CSlice w, float eps, CSlice cos, CSlice sin, const uint32_t* pos, const RopeForm& form, RopeStreams streams) override {
        if (!heads || !head_dim || !rope_dim || rope_dim % 2 || rope_dim > head_dim || form.offset > head_dim - rope_dim)
            throw std::runtime_error(Api::prefix + "invalid partial rope dimensions");
        if (!form.plain() && head_dim > 8 * kThreads) throw std::runtime_error(Api::prefix + "a rope form over a head wider than " + std::to_string(8 * kThreads));
        if (!streams.count || (streams.count > 1 && !streams.of)) throw std::runtime_error(Api::prefix + "a rope of several streams without their pairs");
        if (rows && !pos) throw std::runtime_error(Api::prefix + "partial rope without positions");
        if (!rows) return;
        const size_t width = size_mul(heads, head_dim), half = rope_dim / 2;
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(rows, width), sizeof(float)));
        const size_t last = size_add(size_add(size_mul(rows - 1, src_stride), size_mul(heads - 1, src_head_stride)), head_dim);
        span(*src.buffer, src.offset * sizeof(float), size_mul(last, sizeof(float)));
        if (w.buffer) span(*w.buffer, w.offset * sizeof(float), size_mul(head_dim, sizeof(float)));
        const size_t npos = size_mul(rows, streams.count);
        const size_t table = size_mul(size_add(*std::max_element(pos, pos + npos), 1), half * sizeof(float));
        span(*cos.buffer, cos.offset * sizeof(float), table);
        span(*sin.buffer, sin.offset * sizeof(float), table);
        float* d = f32(dst);
        const float* s = f32(src);
        if ((const float*)d == s && (src_stride != width || src_head_stride != head_dim))
            throw std::runtime_error(Api::prefix + "partial rope in place over heads that are not contiguous");
        const unsigned* p = (const unsigned*)pass_arg(pos, npos * sizeof(uint32_t));
        unsigned long long dst_stride = width, ss = src_stride, shs = src_head_stride;
        int hd = (int)head_dim, hf = (int)half;
        const float* wp = w.buffer ? f32(w) : nullptr;
        const float* c = f32(cos);
        const float* sn = f32(sin);
        if (streams.count == 1 && form.plain()) {
            const Args args(d, s, dst_stride, ss, shs, hd, hf, wp, eps, c, sn, p);
            return launch(kNormRope, (unsigned)rows, (unsigned)heads, args);
        }
        for (size_t i = 0; i < half && streams.count > 1; ++i)
            if (streams.of[i] >= streams.count) throw std::runtime_error(Api::prefix + "a rope pair of a stream past the row's");
        const unsigned char* of = streams.count > 1 ? (const unsigned char*)pass_arg(streams.of, half) : nullptr;
        int count = (int)streams.count;
        // The adjacent, offset and inverse forms take their own kernel, which forms the norm and the pairs as the CPU does (kernels/f32.gpu, llmx_norm_rope_form).
        if (!form.plain()) {
            unsigned hdu = (unsigned)head_dim, hfu = (unsigned)half, cu = (unsigned)count, adj = form.adjacent, off = (unsigned)form.offset, inv = form.inverse, hu = (unsigned)heads;
            unsigned long long items = size_mul(rows, heads);
            const Args args(d, s, dst_stride, ss, shs, hdu, hfu, wp, eps, c, sn, p, of, cu, adj, off, inv, hu, items);
            const size_t waves = kThreads / dev_.caps.subgroup_size;   // a wave a (row, head)
            return launch(kNormRopeForm, (unsigned)((items + waves - 1) / waves), 1, args);
        }
        const Args args(d, s, dst_stride, ss, shs, hd, hf, wp, eps, c, sn, p, of, count);
        launch(kNormRopeStreams, (unsigned)rows, (unsigned)heads, args);
    }
    void sigmoid_mul(Slice dst, CSlice x, CSlice gate, size_t rows, size_t heads, size_t dim, size_t gate_stride, size_t gate_head_stride, RowRuns runs) override {
        if (!heads || !dim) throw std::runtime_error(Api::prefix + "invalid sigmoid gate dimensions");
        if (!rows) return;
        const size_t width = size_mul(heads, dim), n = size_mul(rows, width);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*x.buffer, x.offset * sizeof(float), size_mul(n, sizeof(float)));
        const size_t last = size_add(size_add(size_mul(rows - 1, gate_stride), size_mul(heads - 1, gate_head_stride)), dim);
        span(*gate.buffer, gate.offset * sizeof(float), size_mul(last, sizeof(float)));
        float* d = f32(dst);
        const float* xp = f32(x);
        const float* g = f32(gate);
        unsigned long long nn = n, wd = width, dd = dim, gs = gate_stride, ghs = gate_head_stride;
        StreamTwin t = stream_twin(dst, rows, width, runs);
        const Args args(d, xp, g, nn, wd, dd, gs, ghs, t.q, t.d, t.s, t.level, t.map, t.major);
        launch_writing({{d, n}}, kSigmoidMul, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, n);
    }
    void gated_rms_norm(Slice dst, CSlice x, CSlice z, CSlice w, size_t rows, size_t heads, size_t dim, float eps, RowRuns runs, NormGate gate) override {
        if (!heads || !dim) throw std::runtime_error(Api::prefix + "invalid gated norm dimensions");
        const size_t n = size_mul(size_mul(rows, heads), dim);
        for (const CSlice& v : {CSlice(dst), x, z}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(dim, sizeof(float)));
        if (!rows) return;
        float* d = f32(dst);
        const float* xp = f32(x);
        const float* zp = f32(z);
        const float* wp = f32(w);
        int dm = (int)dim, sig = gate == NormGate::sigmoid;
        StreamTwin t = stream_twin(dst, rows, size_mul(heads, dim), runs, dim % 32 == 0);
        const Args args(d, xp, zp, wp, dm, eps, sig, t.q, t.d, t.s, t.level, t.map, t.major);
        launch_writing({{d, n}}, kGatedRmsNorm, (unsigned)size_mul(rows, heads), 1, args);
        keep(t.t, d, n);
    }
    // The view count, then five words a view (first row, rows, history length, source slot, destination slot), as the state kernels read them.
    static std::vector<unsigned> state_table(const StateView* views, size_t n_views) {
        std::vector<unsigned> tab{(unsigned)n_views};
        size_t row0 = 0;
        for (size_t i = 0; i < n_views; ++i) {
            const StateView& v = views[i];
            if (v.storage != views[0].storage) throw std::runtime_error(Api::prefix + "state views of two storages in one call");
            if (v.length > UINT32_MAX || row0 + v.nq > UINT32_MAX) throw std::runtime_error(Api::prefix + "a state view past 32-bit rows");
            tab.insert(tab.end(), {(unsigned)row0, (unsigned)v.nq, (unsigned)v.length, (unsigned)v.src, (unsigned)v.dst});
            row0 += v.nq;
        }
        return tab;
    }
    // A block a (view, V head, 32 V columns), in the Vulkan shader's order (kernels/f32.gpu, llmx_gated_delta_rule).
    void gated_delta_rule(Slice out, CSlice qkv, CSlice alpha, CSlice b, CSlice a, CSlice dt_bias, size_t layer, const StateView* views, size_t n_views) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        const std::vector<unsigned> tab = state_table(views, n_views);
        const StateStorage& ss = *views[0].storage;
        const StateShape& sh = ss.shape();
        const size_t Hk = sh.k_heads, Hv = sh.v_heads, Dk = sh.k_dim, Dv = sh.v_dim, C = sh.channels();
        // A column's K rows sit 16 in each of eight lanes, and a block stages 128 of each q and k row.
        if (Dk > 128) throw std::runtime_error(Api::prefix + "a linear-attention K head wider than 128");
        span(*out.buffer, out.offset * sizeof(float), size_mul(size_mul(rows, Hv * Dv), sizeof(float)));
        span(*qkv.buffer, qkv.offset * sizeof(float), size_mul(size_mul(rows, C), sizeof(float)));
        for (const CSlice& g : {alpha, b}) span(*g.buffer, g.offset * sizeof(float), size_mul(size_mul(rows, Hv), sizeof(float)));
        for (const CSlice& g : {a, dt_bias}) span(*g.buffer, g.offset * sizeof(float), size_mul(Hv, sizeof(float)));
        const size_t blocks = (Dv + 31) / 32, grid = size_mul(size_mul(n_views, Hv), blocks);
        if (grid > INT32_MAX || C > UINT32_MAX) throw std::runtime_error(Api::prefix + "a delta rule past the launch's grid");
        float* o = f32(out);
        const float* u = f32(qkv);
        const float* al = f32(alpha);
        const float* bp = f32(b);
        const float* ap = f32(a);
        const float* dt = f32(dt_bias);
        float* st = f32(Slice{&ss.layer(layer), 0});
        const unsigned* tp = (const unsigned*)arg(tab.data(), tab.size() * sizeof(unsigned));
        unsigned cc = (unsigned)C, hk = (unsigned)Hk, hv = (unsigned)Hv, dk = (unsigned)Dk, dv = (unsigned)Dv, nb = (unsigned)blocks;
        unsigned long long slot = ss.slot_floats(layer);
        float scale = (float)(1.0 / std::sqrt((double)Dk)), eps = kL2NormEps;
        const Args args(o, u, al, bp, ap, dt, st, tp, cc, hk, hv, dk, dv, slot, nb, scale, eps);
        // The short build where every view holds at most 8 rows, as the Vulkan backend chooses it; the staged count changes no bits (kernels/f32.gpu).
        bool short_build = true;
        for (size_t i = 0; i < n_views; ++i) short_build = short_build && views[i].nq <= 8;
        launch(short_build ? kGatedDeltaRuleShort : kGatedDeltaRule, (unsigned)grid, 1, args);
    }
    // Rows a conv thread walks, a chunk of a view; at least the three a window holds, so only a view's first chunk reads the rows it carries in.
    static constexpr size_t kConvChunk = 16;
    void causal_conv_silu(Slice out, CSlice x, CSlice w, size_t layer, const StateView* views, size_t n_views) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        std::vector<unsigned> tab = state_table(views, n_views);
        size_t chunks = 0;
        for (size_t i = 0; i < n_views; ++i)
            for (size_t b0 = 0; b0 < views[i].nq; b0 += kConvChunk, ++chunks) tab.insert(tab.end(), {(unsigned)i, (unsigned)b0});
        const StateStorage& ss = *views[0].storage;
        const StateShape& sh = ss.shape();
        const size_t C = sh.channels();
        for (const CSlice& v : {CSlice(out), x}) span(*v.buffer, v.offset * sizeof(float), size_mul(size_mul(rows, C), sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(size_mul(C, kConvTaps), sizeof(float)));
        if (chunks > 65535 || C > UINT32_MAX) throw std::runtime_error(Api::prefix + "a conv past the launch's grid");
        float* o = f32(out);
        const float* xp = f32(x);
        const float* wp = f32(w);
        float* st = f32(Slice{&ss.layer(layer), 0});
        const unsigned* tp = (const unsigned*)arg(tab.data(), tab.size() * sizeof(unsigned));
        unsigned cc = (unsigned)C, chunk = (unsigned)kConvChunk;
        unsigned long long slot = ss.slot_floats(layer), carried = size_mul(sh.v_heads, sh.matrix_floats());
        const Args args(o, xp, wp, st, tp, cc, slot, carried, chunk);
        launch(kCausalConvSilu, (unsigned)((C + kThreads - 1) / kThreads), (unsigned)chunks, args);
    }
    // Where delta_holds, the products and then one launch of llmx_delta_step (docs/src/backends-gpu.md, delta_step), every slot but conv the base's bits.
    void delta_step(const DeltaStep& d, CSlice mixed, size_t nin, size_t rows, size_t layer, const StateView* views, size_t n_views, RowRuns runs = {},
                    Dtype dtype = Dtype::f16) override {
        if (!delta_holds(d, nin, rows, layer, views, n_views, runs, dtype)) return Backend::delta_step(d, mixed, nin, rows, layer, views, n_views, runs, dtype);
        check_runs(rows, runs);
        const StateStorage& ss = *views[0].storage;
        const StateShape& sh = ss.shape();
        const size_t Hv = sh.v_heads, C = sh.channels(), V = size_mul(Hv, sh.v_dim), n = size_mul(rows, V);
        span(*mixed.buffer, mixed.offset * sizeof(float), size_mul(size_mul(rows, nin), sizeof(float)));
        for (const Projection* p : {&d.alpha, &d.beta}) {
            check_weight(p->weight, nin, "delta step matrix", 1);
            span(*p->weight.data.buffer, p->weight.data.offset * sizeof(float), quant::row_bytes(p->weight.type, nin, Hv));
            span(*p->out.buffer, p->out.offset * sizeof(float), size_mul(size_mul(rows, Hv), sizeof(float)));
        }
        span(*d.conv_w.buffer, d.conv_w.offset * sizeof(float), size_mul(size_mul(C, kConvTaps), sizeof(float)));
        for (const CSlice& g : {d.a, d.dt_bias}) span(*g.buffer, g.offset * sizeof(float), size_mul(Hv, sizeof(float)));
        span(*d.norm.buffer, d.norm.offset * sizeof(float), size_mul(sh.v_dim, sizeof(float)));
        // Form 3 reads alpha and beta from the base's own products, of any type and class.
        // Otherwise generated rows give the same bits in any group, so they take qkv and gate as one; a prompt, a short one on the rows included, takes the two products, since the tile splits by the group's outputs.
        const int form = delta_form(d, nin, rows, runs, dtype);
        if (form == 3) delta_products(d, mixed, nin, rows, runs, dtype);
        else {
            unordered(true);
            if (generated_rows(rows, runs)) matmul_group({d.qkv, d.gate}, mixed, nin, rows, runs, dtype);
            else {
                matmul(d.qkv.weight, mixed, d.qkv.out, nin, d.qkv.rows, rows, runs, dtype);
                matmul(d.gate.weight, mixed, d.gate.out, nin, d.gate.rows, rows, runs, dtype);
            }
            unordered(false);
        }
        span(*d.out.buffer, d.out.offset * sizeof(float), size_mul(n, sizeof(float)));
        block_counters();
        const std::vector<unsigned> tab = state_table(views, n_views);
        // The half-block form reads the int8 twin of the rows the products just took, the kept twin, generated rows lying row after row.
        const uint8_t* xq = nullptr;
        const float* xd = nullptr;
        if (form == 2) {
            const RowRun whole[1] = {{rows, 1}};
            const TwinPlanes twin = twin_of(mixed, rows, nin, true, twin_map(runs.n ? runs : RowRuns{whole, 1}, rows, nin), true, false);
            xq = twin.q, xd = twin.d;
        }
        StreamTwin t = stream_twin(d.out, rows, V, runs, sh.v_dim % 32 == 0);
        const DeltaStepArgs A{f32(d.out), f32(d.qkv.out), f32(d.conv_w), f32(mixed), bytes(d.alpha.weight.data), bytes(d.beta.weight.data), f32(d.alpha.out),
                              f32(d.beta.out), f32(d.a), f32(d.dt_bias), f32(Slice{&ss.layer(layer), 0}), (const unsigned*)arg(tab.data(), tab.size() * sizeof(unsigned)),
                              f32(d.gate.out), f32(d.norm), (unsigned*)as_gpu<Api>(*delta_counter_).data(), (unsigned)C, (unsigned)sh.k_heads, (unsigned)Hv, (unsigned)nin,
                              ss.slot_floats(layer), size_mul(Hv, sh.matrix_floats()), (float)(1.0 / std::sqrt((double)sh.k_dim)), kL2NormEps, d.eps, t.q, t.d, t.s, t.level,
                              t.major, t.map, (unsigned)sh.k_dim, (unsigned)sh.v_dim, (const int8_t*)xq, xd, (unsigned)form, d.norm_gate == NormGate::sigmoid ? 1u : 0u};
        if (form == 3) launch_writing({{A.o, n}}, kDeltaStep, (unsigned)size_mul(n_views, Hv), 1, Args(A));
        else launch_writing({{A.o, n}, {A.ao, size_mul(rows, Hv)}, {A.bo, size_mul(rows, Hv)}}, kDeltaStep, (unsigned)size_mul(n_views, Hv), 1, Args(A));
        keep(t.t, A.o, n);
    }
    // llmx_delta_step's limits: K and V heads up to kDeltaHead wide in multiples of 4, rows a view, views a launch and its (view, K head) counters.
    static constexpr size_t kDeltaHead = 128, kDeltaRows = 8, kDeltaViews = 64, kDeltaCounters = 4096;
    static_assert(kConvTaps == 4, "llmx_delta_step's conv takes four taps");
    // Whether every run is a generated row, as a call without runs is when it holds one row.
    static bool generated_rows(size_t rows, RowRuns runs) {
        bool generated = runs.n ? true : rows == 1;
        for (size_t r = 0; r < runs.n; ++r) generated = generated && runs.runs[r].extent == 1;
        return generated;
    }
    // How llmx_delta_step takes alpha and beta (kernels/f32.gpu, DeltaStepArgs form): summed in-kernel in the order the base's products take, or, form 3, read from those products.
    // F32 rows that ask for no wide tile and the F32 class's Q8_0 rows take exact_sum, the int8 class's generated Q8_0 rows of an even block count the row kernel's half-block order, and every other type and class form 3.
    static int delta_form(const DeltaStep& d, size_t nin, size_t rows, RowRuns runs, Dtype dtype) {
        const uint32_t t = d.alpha.weight.type;
        if (d.beta.weight.type != t) return 3;
        if (t == quant::GGML_TYPE_F32) return d.alpha.weight.wide || d.beta.weight.wide ? 3 : 0;
        if (t != quant::GGML_TYPE_Q8_0) return 3;
        if (dtype == Dtype::f32 || dtype == Dtype::bf16) return 1;
        return dtype == Dtype::int8 && nin / 32 % 2 == 0 && generated_rows(rows, runs) ? 2 : 3;
    }
    // Whether a step has the shape llmx_delta_step holds (docs/src/backends-gpu.md, delta_step), with state rows on four-float boundaries.
    bool delta_holds(const DeltaStep& d, size_t nin, size_t rows, size_t layer, const StateView* views, size_t n_views, RowRuns runs, Dtype dtype) const {
        if (!nin || nin > UINT32_MAX || !n_views || n_views > kDeltaViews || !views[0].storage) return false;
        const StateStorage& ss = *views[0].storage;
        const StateShape& sh = ss.shape();
        const auto width = [](size_t w) { return w && w <= kDeltaHead && w % 4 == 0; };
        if (!width(sh.k_dim) || !width(sh.v_dim) || !sh.k_heads || sh.v_heads % sh.k_heads || size_mul(n_views, sh.k_heads) > kDeltaCounters) return false;
        size_t total = 0;
        for (size_t i = 0; i < n_views; ++i) {
            if (!views[i].nq || views[i].nq > kDeltaRows || views[i].storage != &ss) return false;
            total += views[i].nq;
        }
        if (total != rows || layer >= ss.layers() || ss.slot_floats(layer) % 4 || size_mul(sh.v_heads, sh.matrix_floats()) % 4) return false;
        // A block runs a view's rows in turn over a whole V head, where the base's rule spreads a head over Dv / 32 blocks, so views of several rows need blocks for half the compute units.
        if (total > n_views && 2 * size_mul(n_views, sh.v_heads) < dev_.caps.compute_units) return false;
        return d.alpha.rows == sh.v_heads && d.beta.rows == sh.v_heads && d.gate.rows == size_mul(sh.v_heads, sh.v_dim) && d.qkv.rows == sh.channels();
    }
    // The Flash-Next stream ops (kernels/f32.gpu), plain F32 outputs whose products make their own twins; every refusal comes before any launch.
    // Items whose products all run the norm inside them go three a launch of llmx_matmul_normed_f32; the rest run as single calls, in order.
    void group_rms_norm_items(const NormRows* items, size_t n, CSlice w, size_t rows, size_t groups, size_t dim, float eps, RowRuns runs) override {
        if (!groups || !dim || !std::isfinite(eps) || eps <= 0.0f) throw std::runtime_error(Api::prefix + "invalid grouped norm dimensions or epsilon");
        if (!rows) return;
        for (size_t i = 0; i < n;) {
            const size_t k = std::min<size_t>(3, n - i);
            NormPlan p[3];
            for (size_t j = 0; j < k; ++j) p[j] = norm_plan(items[i + j], w, rows, groups, dim);
            size_t m = 0;
            while (m < k && norm_inside(items + i, p, m, w, rows, groups * dim)) ++m;
            if (m) {
                normed_products(items + i, p, m, w, groups, dim, eps);
            } else {
                norm_rows(items[i], p[0], w, rows, groups, dim, eps, runs);
                m = 1;
            }
            i += m;
        }
    }
    // A grouped norm item checked against its shape: its pointers, source strides and extents.
    struct NormPlan {
        float* d = nullptr;
        const float *s = nullptr, *s2 = nullptr;
        CSlice rest;
        unsigned long long st = 0, st2 = 0;
        unsigned split = 0;
        size_t an = 0, bn = 0;
        bool in_place = false;
    };
    NormPlan norm_plan(const NormRows& it, CSlice w, size_t rows, size_t groups, size_t dim) {
        const Slice dst = it.dst;
        const CSlice src = it.src;
        const size_t width = size_mul(groups, dim), n = size_mul(rows, width), blocks = size_mul(rows, groups);
        const NormSource f = it.from.resolve(src, groups, width);
        if (f.split > groups || f.stride < size_mul(f.split, dim) || f.rest_stride < size_mul(groups - f.split, dim)) throw std::runtime_error(Api::prefix + "invalid grouped norm sources");
        NormPlan p;
        p.in_place = f.contiguous(src, groups, width) && dst.buffer == src.buffer && dst.offset == src.offset;
        // Each source's extent, rows stride floats apart.
        p.an = f.split ? size_add(size_mul(rows - 1, f.stride), f.split * dim) : 0;
        p.bn = f.split < groups ? size_add(size_mul(rows - 1, f.rest_stride), (groups - f.split) * dim) : 0;
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(n, sizeof(float)));
        if (p.an) span(*src.buffer, src.offset * sizeof(float), size_mul(p.an, sizeof(float)));
        if (p.bn) span(*f.rest.buffer, f.rest.offset * sizeof(float), size_mul(p.bn, sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(width, sizeof(float)));
        if (!p.in_place && ((p.an && overlaps(dst, n, src, p.an)) || (p.bn && overlaps(dst, n, f.rest, p.bn))))
            throw std::runtime_error(Api::prefix + "grouped norm output must be disjoint or exactly in place");
        if (overlaps(dst, n, w, width)) throw std::runtime_error(Api::prefix + "grouped norm output overlaps its weights");
        if (blocks > INT32_MAX || dim > UINT32_MAX || f.split > UINT32_MAX) throw std::runtime_error(Api::prefix + "a grouped norm past the launch's grid");
        p.d = f32(dst);
        p.s = p.an ? f32(src) : f32(f.rest);
        p.s2 = p.bn ? f32(f.rest) : p.s;
        p.rest = f.rest, p.st = f.stride, p.st2 = f.rest_stride, p.split = unsigned(f.split);
        return p;
    }
    // Whether item m's product runs the norm inside it beside items 0 to m - 1: one row, an F32 product the shared memory fits, item 0's nout and split.
    // Its y lies on the device apart from w, every other y and every source, since the launch reads them while it writes.
    bool norm_inside(const NormRows* items, const NormPlan* p, size_t m, CSlice w, size_t rows, size_t width) {
        const NormProduct& into = items[m].into;
        if (!into.weight.data.buffer || rows != 1 || into.weight.type != quant::GGML_TYPE_F32 || width > kNormWidth || !into.y.buffer || as_gpu<Api>(*into.y.buffer).host_ptr() ||
            overlaps(into.y, into.nout, w, width) || (m && (into.nout != items[0].into.nout || p[m].split != p[0].split)))
            return false;
        for (size_t j = 0; j <= m; ++j) {
            const NormProduct& o = items[j].into;
            if ((p[j].an && overlaps(into.y, into.nout, items[j].src, p[j].an)) || (p[j].bn && overlaps(into.y, into.nout, p[j].rest, p[j].bn))) return false;
            if (j < m && ((overlaps(o.y, o.nout, into.y, into.nout)) || (p[m].an && overlaps(o.y, o.nout, items[m].src, p[m].an)) || (p[m].bn && overlaps(o.y, o.nout, p[m].rest, p[m].bn))))
                return false;
        }
        return true;
    }
    // Up to three items' products in one launch of llmx_matmul_normed_f32, blockIdx.y the item: every block forms its item's normed row in shared memory and leaves dst unwritten (kernels/f32.gpu).
    void normed_products(const NormRows* items, const NormPlan* p, size_t n, CSlice w, size_t groups, size_t dim, float eps) {
        const size_t width = groups * dim, nout = items[0].into.nout;
        for (size_t i = 0; i < n; ++i) {
            check_weight(items[i].into.weight, width, "matrix", 1);
            check_grid(items[i].into.weight, width, nout, 1);
        }
        if (!nout) return;
        NormItems a{};
        uintptr_t bits = (uintptr_t)f32(w);
        for (size_t i = 0; i < n; ++i) {
            const NormProduct& into = items[i].into;
            span(*into.weight.data.buffer, into.weight.data.offset * sizeof(float), quant::row_bytes(into.weight.type, width, nout));
            span(*into.y.buffer, into.y.offset * sizeof(float), size_mul(nout, sizeof(float)));
            a.W[i] = bytes(into.weight.data), a.Y[i] = f32(into.y), a.src0[i] = p[i].s, a.src1[i] = p[i].s2;
            bits |= (uintptr_t)p[i].s | (uintptr_t)p[i].s2 | (uintptr_t)a.W[i];
        }
        a.w = f32(w), a.groups = unsigned(groups), a.dim = unsigned(dim), a.split = p[0].split, a.eps = eps;
        unsigned long long no = nout;
        const Args args(no, a);
        // At most 2 groups of up to kNormHeldSteps * 256 floats on aligned rows take the held form, two waves a block (kernels/f32.gpu, normed_held).
        const bool held = dev_.caps.subgroup_size == 64 && groups <= 2 && dim % kThreads == 0 && dim / kThreads <= kNormHeldSteps && bits % 16 == 0;
        const size_t waves = held ? 2 : kThreads / dev_.caps.subgroup_size;
        launch_writing({{a.Y[0], nout}, {a.Y[1], n > 1 ? nout : 0}, {a.Y[2], n > 2 ? nout : 0}}, kMatmulNormedF32, (unsigned)((nout + waves - 1) / waves), unsigned(n), args, 1,
                       held ? unsigned(2 * 64) : kThreads, held ? 0u : u32(size_mul(width, sizeof(float))));
        for (size_t i = 0; i < n; ++i) record_matrix_path(MatrixPath::f32);
    }
    // One item as the norm's own launch, then its product.
    void norm_rows(const NormRows& it, const NormPlan& p, CSlice w, size_t rows, size_t groups, size_t dim, float eps, RowRuns runs) {
        const size_t width = size_mul(groups, dim), n = size_mul(rows, width), blocks = size_mul(rows, groups);
        const float* wp = f32(w);
        StreamTwin t = stream_twin(it.dst, rows, width, runs, dim % 32 == 0);
        // Four values a thread only where every operand lies on 16 bytes and every source row stays there, since a slice's offset may be any float.
        int four = dim % 4 == 0 && p.st % 4 == 0 && p.st2 % 4 == 0 && ((uintptr_t)p.d | (uintptr_t)p.s | (uintptr_t)p.s2 | (uintptr_t)wp) % 16 == 0;
        const Args args(p.d, p.s, p.s2, p.st, p.st2, p.split, wp, unsigned(groups), unsigned(dim), eps, t.q, t.d, t.s, t.level, t.map, t.major, four);
        // Slices of a group, each repeating its reduction, up to what gives four blocks a compute unit (Vulkan's split, fc6f53415); one in place, where a slice's writes would meet another's reads.
        const size_t fill = (4 * size_t(std::max(1u, dev_.caps.compute_units)) + blocks - 1) / blocks;
        const size_t slices = p.in_place ? 1 : std::max<size_t>(1, std::min((dim + 1023) / 1024, fill));
        launch_writing({{p.d, n}}, kGroupRmsNorm, (unsigned)blocks, (unsigned)slices, args);
        keep(t.t, p.d, n);
        norm_product(it.dst, it.into, rows, width, runs);
    }
    void silu_scaled(Slice dst, CSlice src, size_t rows, size_t width, float scale, RowRuns runs) override {
        if (!width || !std::isfinite(scale)) throw std::runtime_error(Api::prefix + "invalid scaled SiLU dimensions or scale");
        if (!rows) return;
        const size_t n = size_mul(rows, width);
        for (const CSlice& v : {CSlice(dst), src}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        if (overlaps(dst, n, src, n) && !(dst.buffer == src.buffer && dst.offset == src.offset)) throw std::runtime_error(Api::prefix + "scaled SiLU output must be disjoint or exactly in place");
        float* d = f32(dst);
        const float* s = f32(src);
        unsigned long long nn = n;
        StreamTwin t = stream_twin(dst, rows, width, runs);
        const Args args(d, s, nn, scale, t.q, t.d, t.s, t.level, t.map, t.major);
        launch_writing({{d, n}}, kSiluScaled, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, n);
    }
    void mix_streams(Slice dst, CSlice src, CSlice gate, size_t rows, size_t streams, size_t dim, RowRuns runs) override {
        if (!streams || !dim) throw std::runtime_error(Api::prefix + "invalid stream mix dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(streams, dim)), out_n = size_mul(rows, dim);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(out_n, sizeof(float)));
        for (const CSlice& v : {src, gate}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        if (overlaps(dst, out_n, src, n) || overlaps(dst, out_n, gate, n)) throw std::runtime_error(Api::prefix + "stream mix output must be disjoint from its inputs");
        if (streams > UINT32_MAX || dim > UINT32_MAX) throw std::runtime_error(Api::prefix + "a stream mix past 32-bit streams or widths");
        float* d = f32(dst);
        const float* s = f32(src);
        const float* g = f32(gate);
        unsigned long long nn = out_n;
        unsigned st = (unsigned)streams, dm = (unsigned)dim;
        StreamTwin t = stream_twin(dst, rows, dim, runs);
        const Args args(d, s, g, nn, st, dm, t.q, t.d, t.s, t.level, t.map, t.major);
        launch_writing({{d, out_n}}, kMixStreams, (unsigned)((out_n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, out_n);
    }
    // The hyper-connection mix as persistent launches (kernels/int16.gpu, llmx_hc_mix) for generated rows over 8-bit activations on a 64-lane wave, up to 8 columns a launch; the chain of ops otherwise.
    // A column gives the one-column launch's bits in any build, so verify rows mix as the decode row would; on a shape the entry fits, every generated row follows the entry rule (entry, entry_rows).
    void hyper_mix(Slice out, CSlice x, const HyperMix& m, size_t rows, size_t streams, size_t dim, size_t low_rank, float eps, RowRuns runs, Dtype dtype) override {
        const bool inj = m.inject.data.buffer != nullptr;
        const size_t width = streams * dim;
        bool generated = runs.n ? rows > 0 : rows == 1;
        for (size_t r = 0; r < runs.n; ++r) generated = generated && runs.runs[r].extent == 1;
        bool fits = dtype == Dtype::int8 && dev_.caps.subgroup_size == 64 && streams && dim % 32 == 0 && low_rank && low_rank % 32 == 0 &&
                    low_rank <= 512 && size_mul(width, kHcColumns) <= INT32_MAX && std::isfinite(eps) && eps > 0.0f && m.down.type == quant::GGML_TYPE_Q8_0 &&
                    m.up.type == quant::GGML_TYPE_Q8_0 && (!inj || m.inject.type == quant::GGML_TYPE_F32) && m.down.data.buffer && m.up.data.buffer;
        if (fits)
            for (const CSlice& v : {CSlice(out), x, m.norm, CSlice(m.normed), CSlice(m.low), CSlice(m.gates), inj ? CSlice(m.inject_gates) : x})
                if (!v.buffer || as_gpu<Api>(*v.buffer).host_ptr()) fits = false;
        // A launch of n columns is ceil(low rank / 4) down blocks and a block an inject output of a column.
        const auto grid = [&](size_t n) { return (unsigned)((low_rank + 3) / 4 + (inj ? streams * n : 0)); };
        // A shape the entry fits mixes every generated row under the entry rule, whatever the rows in the call, so a row's bits never depend on them.
        const bool entry_rule = fits && entry_fits(streams, dim, low_rank, inj);
        if (entry_rule && !generated && std::any_of(runs.runs, runs.runs + runs.n, [](const RowRun& r) { return r.extent == 1; })) {
            mix_by_class(out, x, m, streams, dim, low_rank, eps, runs, dtype);
            return;
        }
        fits = fits && generated;
        if (fits && !entry_rule)
            for (size_t left = rows; left && fits; left -= std::min(left, kHcColumns))
                fits = hc_resident(hc_build(std::min(left, kHcColumns)), grid(std::min(left, kHcColumns)));
        if (!fits) {
            Backend::hyper_mix(out, x, m, rows, streams, dim, low_rank, eps, runs, dtype);
            return;
        }
        const size_t q8 = 34;
        span(*out.buffer, out.offset * sizeof(float), size_mul(rows, dim * sizeof(float)));
        for (const CSlice& v : {x, CSlice(m.normed), CSlice(m.gates)}) span(*v.buffer, v.offset * sizeof(float), size_mul(rows, width * sizeof(float)));
        span(*m.norm.buffer, m.norm.offset * sizeof(float), width * sizeof(float));
        span(*m.low.buffer, m.low.offset * sizeof(float), size_mul(rows, low_rank * sizeof(float)));
        span(*m.down.data.buffer, m.down.data.offset * sizeof(float), size_mul(low_rank, width / 32 * q8));
        span(*m.up.data.buffer, m.up.data.offset * sizeof(float), size_mul(width, low_rank / 32 * q8));
        if (inj) {
            span(*m.inject.data.buffer, m.inject.data.offset * sizeof(float), size_mul(streams, width) * sizeof(float));
            span(*m.inject_gates.buffer, m.inject_gates.offset * sizeof(float), size_mul(rows, streams * sizeof(float)));
        }
        if (entry_rule) {
            entry_mixes_ = true;
            if (rows == 1 && pending_ && pending_->inject && pending_->dst == f32(x) && (!pending_->take || pending_->take->members == 2)) entry(out, x, m, streams, dim, low_rank, eps, runs);
            else entry_rows(out, x, m, rows, streams, dim, low_rank, eps, runs);
            return;
        }
        // The launch's scratch: the two barrier counters first, which the allocation zeroes and every launch leaves at zero, then each stream's root, the twin's scales and the twin, for a launch's columns.
        // The counters keep their offset whatever the shape, so a launch on scratch a wider mix used reads zeros there, never that mix's twin bytes.
        const HcLayout hl = hc_layout(rows, streams, width);
        const size_t rr_at = hl.rr_at, nd_at = hl.nd_at, nq_at = hl.nq_at, need = hl.need;
        if (!hc_scratch_ || hc_scratch_->size() < need) hc_scratch_ = alloc(need, Memory::device);
        uint8_t* sc = (uint8_t*)as_gpu<Api>(*hc_scratch_).data();
        forget_twins();
        StreamTwin t = stream_twin(out, rows, dim, runs);
        if (t.major || t.level == 1) t = {};
        for (size_t at = 0; at < rows;) {
            const size_t n = std::min(rows - at, kHcColumns);
            const HcMix a{f32(x) + at * width, f32(m.norm), bytes(m.down.data), inj ? f32(m.inject.data) : nullptr, bytes(m.up.data), f32(m.normed) + at * width,
                          (signed char*)(sc + nq_at), (float*)(sc + nd_at), f32(m.low) + at * low_rank, inj ? f32(m.inject_gates) + at * streams : nullptr,
                          f32(m.gates) + at * width, f32(out) + at * dim, t.q ? (signed char*)t.q + at * dim : nullptr, t.q ? t.d + at * dim / 32 : nullptr,
                          (float*)(sc + rr_at), (unsigned*)sc, (unsigned)streams, (unsigned)dim, (unsigned)low_rank, (unsigned)n, eps, 1.0f / float(streams)};
            launch(hc_build(n), grid(n), 1, Args(a));
            at += n;
        }
        keep(t.t, f32(out), rows * dim);
    }
    // The kernel's argument, laid out as kernels/int16.gpu's HcMix.
    struct HcMix {
        const float* x;
        const float* norm_w;
        const uint8_t* down;
        const float* inj;
        const uint8_t* up;
        float* normed;
        signed char* nq;
        float* nd;
        float* low;
        float* injg;
        float* gates;
        float* out;
        signed char* oq;
        float* od;
        float* rr;
        unsigned* bar;
        unsigned streams, dim, lowr, cols;
        float eps, scale;
    };
    static_assert(offsetof(HcMix, streams) == 16 * sizeof(void*) && sizeof(HcMix) == 16 * sizeof(void*) + 6 * 4, "HcMix mirrors the kernel's argument");
    // The workgroups of build k the device holds at once: its occupancy a compute unit, read once into `per_cu`, times the units.
    size_t resident(int& per_cu, int k) {
        if (per_cu < 0) {
            // HIP finds the module's function through the calling thread's device, which another backend's call may have changed.
            Api::check(Api::set_device(dev_.index), "set_device");
            Api::check(Api::occupancy(&per_cu, kernels_[k], kThreads, 0), "occupancy");
        }
        return size_t(per_cu) * dev_.caps.compute_units;
    }
    // Whether a grid of the build is resident at once, which its barriers need; a grid past it is said once and runs the chain.
    bool hc_resident(int k, unsigned grid) {
        const size_t i = k == kHcMix ? 0 : k == kHcMix8 ? 1 : k == kHcEntry ? 2 : 3;
        const bool ok = size_t(grid) <= resident(hc_per_cu_[i], k);
        if (!ok && !hc_refused_) {
            hc_refused_ = true;
            std::fprintf(stderr, "%sthe hyper-connection mix's %u workgroups exceed the %d a compute unit times %u units resident on %s; it runs as separate launches\n", Api::prefix.c_str(), grid,
                         hc_per_cu_[i], dev_.caps.compute_units, identity().c_str());
        }
        return ok;
    }
    // The pending inject and the mix after it as one launch (kernels/int16.gpu, llmx_hc_entry), on a shape whose grid of streams * dim / 1280 * low / 32 blocks is resident.
    // The scratch holds the two barrier counters first, zero between launches, then the down, square and inject partials.
    static constexpr size_t kEntryGroup = 1280;   // kernels/int16.gpu, kEntGroup
    bool entry_fits(size_t streams, size_t dim, size_t low, bool inj) {
        if (streams > 8 || dim % kEntryGroup || low % 64 || low > 512 || (inj && streams > low / 32 - 1)) return false;
        const size_t nb = streams * (dim / 1280) * (low / 32);
        return nb <= UINT32_MAX && (dim / 32 + nb - 1) / nb * streams * 32 <= 512 && hc_resident(kHcEntry, (unsigned)nb) && hc_resident(kHcEntry8, (unsigned)nb);
    }
    // A call of generated rows beside prompt runs, on a shape the entry fits: each stretch of one class mixes alone, the generated rows under the entry rule and the prompt runs by the chain.
    void mix_by_class(Slice out, CSlice x, const HyperMix& m, size_t streams, size_t dim, size_t low, float eps, RowRuns runs, Dtype dtype) {
        const size_t width = streams * dim;
        const bool inj = m.inject.data.buffer != nullptr;
        size_t at = 0;
        for (size_t r = 0; r < runs.n;) {
            const bool gen = runs.runs[r].extent == 1;
            size_t e = r;
            while (e < runs.n && (runs.runs[e].extent == 1) == gen) ++e;
            std::vector<RowRun> part(runs.runs + r, runs.runs + e);
            for (RowRun& p : part) p.end -= at;
            const size_t n = part.back().end;
            HyperMix s = m;
            s.normed.offset += at * width, s.low.offset += at * low, s.gates.offset += at * width;
            if (inj) s.inject_gates.offset += at * streams;
            hyper_mix({out.buffer, out.offset + at * dim}, {x.buffer, x.offset + at * width}, s, n, streams, dim, low, eps, {part.data(), part.size()}, dtype);
            at += n;
            r = e;
        }
    }
    static constexpr size_t kEntryColumns = 8;   // llmx_hc_entry8's rows, kEntCols in kernels/int16.gpu
    // The scratch of a launch of n rows: the two barrier counters, then each row's down, square and inject partials; both builds take kEntryColumns' size, so the buffer is made once.
    static size_t entry_need(size_t n, size_t streams, size_t dim, size_t low) {
        const size_t ng = streams * (dim / kEntryGroup);
        return 64 + n * (ng * low + ng + ng * streams) * sizeof(float);
    }
    void entry_scratch(size_t need) {
        if (!hc_entry_scratch_ || hc_entry_scratch_->size() < need) hc_entry_scratch_ = alloc(need, Memory::device);
    }
    // Rows the inject already ran on, up to kEntryColumns a launch of llmx_hc_entry8, each row with the one-row entry's bits.
    void entry_rows(Slice out, CSlice x, const HyperMix& m, size_t rows, size_t streams, size_t dim, size_t low, float eps, RowRuns runs) {
        const bool inj = m.inject.data.buffer != nullptr;
        const size_t width = streams * dim, nb = streams * (dim / kEntryGroup) * (low / 32);
        entry_scratch(entry_need(kEntryColumns, streams, dim, low));
        uint8_t* sc = (uint8_t*)as_gpu<Api>(*hc_entry_scratch_).data();
        forget_twins();
        StreamTwin t = stream_twin(out, rows, dim, runs);
        if (t.major || t.level == 1) t = {};
        for (size_t at = 0; at < rows;) {
            const size_t n = std::min(rows - at, kEntryColumns);
            const HcEntry a{const_cast<float*>(f32(x)) + at * width, nullptr, nullptr, f32(m.norm), bytes(m.down.data), bytes(m.up.data), inj ? f32(m.inject.data) : nullptr, (float*)(sc + 64),
                            f32(m.low) + at * low, inj ? f32(m.inject_gates) + at * streams : nullptr, f32(out) + at * dim, t.q ? (signed char*)t.q + at * dim : nullptr,
                            t.q ? t.d + at * dim / 32 : nullptr, (unsigned*)sc, nullptr, 0, 0, 0, 0, (unsigned)streams, (unsigned)dim, (unsigned)low, 1u, 0u, (unsigned)n, eps,
                            1.0f / float(streams)};
            launch(kHcEntry8, (unsigned)nb, 1, Args(a));
            at += n;
        }
        keep(t.t, f32(out), rows * dim);
    }
    void entry(Slice out, CSlice x, const HyperMix& m, size_t streams, size_t dim, size_t low, float eps, RowRuns runs) {
        const bool inj = m.inject.data.buffer != nullptr;
        const size_t nb = streams * (dim / kEntryGroup) * (low / 32), need = entry_need(kEntryColumns, streams, dim, low);
        // The pending join leaves before the allocation, whose held fill would issue it first and leave the launch nothing to take.
        const Pending p = *pending_;
        pending_.reset();
        if (!hc_entry_scratch_ || hc_entry_scratch_->size() < need) {
            try {
                entry_scratch(need);
            } catch (...) {
                pending_ = p;
                throw;
            }
        }
        uint8_t* sc = (uint8_t*)as_gpu<Api>(*hc_entry_scratch_).data();
        forget_twins();
        StreamTwin t = stream_twin(out, 1, dim, runs);
        if (t.major || t.level == 1) t = {};
        // With a pending take, the block is this member's own partial and the kernel adds its peer's from the staging half the push filled.
        const TakeDesc* k = p.take ? &*p.take : nullptr;
        const HcEntry a{p.dst, k ? k->own : p.block, p.gate, f32(m.norm), bytes(m.down.data), bytes(m.up.data), inj ? f32(m.inject.data) : nullptr, (float*)(sc + 64), f32(m.low),
                        inj ? f32(m.inject_gates) : nullptr, f32(out), t.q ? (signed char*)t.q : nullptr, t.q ? t.d : nullptr, (unsigned*)sc, k ? k->peers : nullptr,
                        k ? k->half_at : 0, k ? k->cap : 0, k ? k->timeout : 0, k ? k->tick : 0, (unsigned)streams, (unsigned)dim, (unsigned)low, k ? 2u : 1u, k ? (unsigned)k->rank : 0u,
                        1u, eps, 1.0f / float(streams)};
        launch(kHcEntry, (unsigned)nb, 1, Args(a));
        if (k) take_status_ = k->status;
        keep(t.t, f32(out), dim);
    }
    // The kernel's argument, laid out as kernels/int16.gpu's HcEntry.
    struct HcEntry {
        float* x;
        const float* blk;
        const float* gin;
        const float* nw;
        const uint8_t* down;
        const uint8_t* up;
        const float* inj;
        float* part;
        float* low;
        float* injg;
        float* out;
        signed char* oq;
        float* od;
        unsigned* bar;
        const void* peers;
        unsigned long long half_at, cap, timeout, tick;
        unsigned streams, dim, lowr, members, rank, cols;
        float eps, scale;
    };
    static_assert(offsetof(HcEntry, half_at) == 15 * sizeof(void*) && offsetof(HcEntry, streams) == 15 * sizeof(void*) + 32 && offsetof(HcEntry, scale) == 15 * sizeof(void*) + 60,
                  "HcEntry mirrors the kernel's argument");
    // A one-shot sum's take, recorded by its collective after the push (collective_core.hpp, pushed): llmx_group_sum's arguments and the launch's blocks.
    struct TakeDesc {
        const void* peers;
        const float* own;
        float* out;
        unsigned long long n, cap, half_at, timeout, tick;
        int members, rank, flags;
        unsigned blocks;
        unsigned* status;   // the member's status word in the collective's host memory, set where the take's launch fails so its next sum turns conservative
        bool whole = false;   // the push is held too: the next ring convolution of `out` runs both (conv_ring), else the whole one-shot launches as llmx_group_sum
    };
    // The join held until the next operation, which issues it first, unless the mix after it takes it (entry): a sum's take, a generated row's inject, or the take and the inject of its sum.
    struct Pending {
        std::optional<TakeDesc> take;
        bool inject;
        float* dst;
        const float* block;
        const float* gate;
        unsigned streams, dim;
    };
    std::optional<Pending> pending_;
    // A take's slots are armed again only by the take, so a flush that drops a held take turns its collective conservative (flush).
    unsigned* take_status_ = nullptr;
    void materialize() {
        if (!pending_) return;
        const Pending p = *pending_;
        pending_.reset();
        if (p.take) {
            const TakeDesc& k = *p.take;
            // A whole sum goes to the stream as it is held, as its peers' do (GpuCollective::summed); one whose launch throws stays pending, so it is not lost to a peer that waits for it.
            flush_held_ = k.whole;
            take_status_ = k.status;
            try {
                launch(k.whole ? kGroupSum : kGroupTake, k.blocks, 1, Args(k.peers, k.own, k.out, k.n, k.cap, k.members, k.rank, k.flags, k.timeout, k.tick));
            } catch (...) {
                flush_held_ = false;
                if (k.whole) pending_ = p;
                throw;
            }
        }
        if (p.inject) launch(kInjectStreams, p.streams * ((p.dim + kThreads - 1) / kThreads), 1, Args(p.dst, p.block, p.gate, p.streams, p.dim));
    }
    // The collective's take, held as the pending join after every member's push went to its stream.
    void pend_take(const TakeDesc& k) {
        materialize();
        // A whole sum writes its rows wherever it runs, so no twin of them is kept past it.
        if (k.whole) forget_twins();
        pending_ = Pending{k, false, nullptr, nullptr, nullptr, 0, 0};
        opened_ = true;
    }
    BufferPtr hc_scratch_, hc_entry_scratch_, topk_work_;
    int hc_per_cu_[4] = {-1, -1, -1, -1};
    bool hc_refused_ = false;
    // Whether a generated row's mix here fits the entry, so a row's sum in place holds its take for one (GpuCollective::sum_in_place); a backend no mix reaches keeps one launch a sum.
    bool entry_mixes_ = false;
    bool flush_held_ = false;   // the operation just held is a whole sum, flushed by hold at once
    int conv_sum_per_cu_ = -1, conv_mix_per_cu_ = -1;
    // The rows the last one-shot sum in place here wrote, and whether a ring convolution has read such rows next, so a member's later sums hold for one (GpuCollective::sum_in_place); a backend no such convolution reaches keeps one launch a sum.
    const float* summed_out_ = nullptr;
    bool conv_sums_ = false;
    void inject_streams(Slice dst, CSlice block, CSlice gate, size_t rows, size_t streams, size_t dim) override {
        if (!streams || !dim) throw std::runtime_error(Api::prefix + "invalid stream injection dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(streams, dim)), bn = size_mul(rows, dim), gn = size_mul(rows, streams);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*block.buffer, block.offset * sizeof(float), size_mul(bn, sizeof(float)));
        span(*gate.buffer, gate.offset * sizeof(float), size_mul(gn, sizeof(float)));
        if (overlaps(dst, n, block, bn) || overlaps(dst, n, gate, gn) || overlaps(block, bn, gate, gn)) throw std::runtime_error(Api::prefix + "stream injection operands must be disjoint");
        const size_t chunks = (dim + kThreads - 1) / kThreads;
        if (gn > INT32_MAX / chunks || dim > UINT32_MAX) throw std::runtime_error(Api::prefix + "a stream injection past the launch's grid");
        float* d = f32(dst);
        const float* b = f32(block);
        const float* g = f32(gate);
        unsigned st = (unsigned)streams, dm = (unsigned)dim;
        // A generated row's inject waits for the mix after it, which may take it into one launch (entry); any other operation issues it first.
        if (rows == 1 && streams <= 8 && dim % 1280 == 0 && !as_gpu<Api>(*dst.buffer).host_ptr() && !as_gpu<Api>(*block.buffer).host_ptr() && !as_gpu<Api>(*gate.buffer).host_ptr()) {
            // The take of the sum this inject reads joins it, so the mix after may take both.
            std::optional<TakeDesc> take;
            if (pending_ && pending_->take && !pending_->take->whole && !pending_->inject && pending_->take->out == b && pending_->take->n == dm) {
                take = pending_->take;
                pending_.reset();
            }
            materialize();
            forget_twins();
            pending_ = Pending{take, true, d, b, g, st, dm};
            opened_ = true;
            return;
        }
        const Args args(d, b, g, st, dm);
        launch(kInjectStreams, (unsigned)(gn * chunks), 1, args);
    }
    void signed_sqrt_gate(Slice dst, CSlice key, CSlice query, CSlice value, size_t rows, size_t groups, size_t dim, RowRuns, ZeroSign zero) override {
        if (!groups || !dim) throw std::runtime_error(Api::prefix + "invalid signed square-root gate dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(groups, dim)), vn = size_mul(rows, dim), blocks = size_mul(rows, groups);
        for (const CSlice& v : {CSlice(dst), key, query}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*value.buffer, value.offset * sizeof(float), size_mul(vn, sizeof(float)));
        if (overlaps(dst, n, key, n) || overlaps(dst, n, query, n) || overlaps(dst, n, value, vn)) throw std::runtime_error(Api::prefix + "signed square-root gate output must be disjoint from its inputs");
        if (blocks > INT32_MAX || dim > UINT32_MAX) throw std::runtime_error(Api::prefix + "a signed square-root gate past the launch's grid");
        float* d = f32(dst);
        const float* k = f32(key);
        const float* q = f32(query);
        const float* v = f32(value);
        unsigned g = (unsigned)groups, dm = (unsigned)dim;
        float scale = 1.0f / std::sqrt(float(dim));
        unsigned kept = zero == ZeroSign::kept ? 1u : 0u;
        const Args args(d, k, q, v, g, dm, scale, kept);
        launch(kSignedSqrtGate, (unsigned)blocks, 1, args);
    }
    // The depthwise conv's tail, SiLU and given-residual subset: a thread a channel of a chunk of a view, chunks at least the history long so only a view's first chunk reads or leaves its carried rows (kernels/f32.gpu).
    void conv_one(Slice out, CSlice x, const DepthwiseConv& a, size_t layer, const StateView* views, size_t n_views) {
        if (a.history == ConvHistory::ring) {
            const ConvRows one{out, x, a};
            conv_ring(&one, 1, layer, views, n_views, false);
            return;
        }
        if (a.history != ConvHistory::tail || a.act != ConvAct::silu || a.residual_from != ConvResidual::given || a.divisor != 1.0f || a.dyn.buffer || a.out_stride || a.accumulate ||
            (out.buffer == x.buffer && out.offset == x.offset))
            lacks(Op::depthwise_conv_ring);
        if (!a.residual.buffer || !a.w.buffer) throw std::runtime_error("backend: depthwise_conv without its residual or weights");
        const CSlice residual = a.residual, w = a.w;
        const size_t channels = a.channels, taps = a.taps, dilation = a.dilation;
        if (!channels || !taps || !dilation) throw std::runtime_error(Api::prefix + "invalid dilated conv dimensions");
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        std::vector<unsigned> tab = state_table(views, n_views);
        const StateStorage& ss = *views[0].storage;
        const size_t history = size_mul(taps - 1, dilation), n = size_mul(rows, channels), wn = size_mul(channels, taps);
        const size_t prefix = ss.shape().slot_floats(), slot = ss.slot_floats(layer);
        if (slot < size_add(prefix, size_mul(history, channels))) throw std::runtime_error(Api::prefix + "recurrent slot lacks dilated conv history");
        for (const CSlice& v : {CSlice(out), x, residual}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(wn, sizeof(float)));
        if ((overlaps(out, n, residual, n) && !(out.buffer == residual.buffer && out.offset == residual.offset)) || overlaps(out, n, x, n) || overlaps(out, n, w, wn))
            throw std::runtime_error(Api::prefix + "dilated conv output may only alias the residual exactly");
        const CSlice state{&ss.layer(layer), 0};
        const size_t sn = ss.layer(layer).size() / sizeof(float);
        if (overlaps(out, n, state, sn) || overlaps(x, n, state, sn) || overlaps(residual, n, state, sn) || overlaps(w, wn, state, sn))
            throw std::runtime_error(Api::prefix + "dilated conv operand overlaps recurrent state storage");
        const size_t chunk = std::max(kConvChunk, history);
        size_t chunks = 0;
        for (size_t i = 0; i < n_views; ++i)
            for (size_t b0 = 0; b0 < views[i].nq; b0 += chunk, ++chunks) tab.insert(tab.end(), {(unsigned)i, (unsigned)b0});
        if (chunks > 65535 || channels > UINT32_MAX || taps > UINT32_MAX || history > UINT32_MAX) throw std::runtime_error(Api::prefix + "a dilated conv past the launch's grid");
        float* o = f32(out);
        const float* xp = f32(x);
        const float* rp = f32(residual);
        const float* wp = f32(w);
        float* st = f32(Slice{&ss.layer(layer), 0});
        const unsigned* tp = (const unsigned*)arg(tab.data(), tab.size() * sizeof(unsigned));
        unsigned cc = (unsigned)channels, tt = (unsigned)taps, dl = (unsigned)dilation, hs = (unsigned)history, ch = (unsigned)chunk;
        unsigned long long sl = slot, pf = prefix;
        const Args args(o, xp, rp, wp, st, tp, cc, tt, dl, hs, sl, pf, ch);
        launch(kDilatedConvSiluAdd, (unsigned)((channels + kThreads - 1) / kThreads), (unsigned)chunks, args);
    }
    // The host's copy of llmx_conv_ring's argument (kernels/f32.gpu, ConvArgs), field for field.
    static constexpr size_t kConvItems = 2;
    static constexpr size_t kConvRaw = (256 + 3) * 16;   // as kernels/f32.gpu
    static constexpr size_t kConvPart = 8;                 // as kernels/f32.gpu
    static constexpr size_t kConvPartsFrom = 128;          // the longest view from which parts beat the staged build (docs/src/backends-gpu.md)
    struct ConvItem {
        float* o;
        const float* x;
        const float* w;
        const float* dyn;
        const float* res;
        unsigned long long ring, dyn_stride, out_stride;
        unsigned C, T, R, group, blocks;
        int silu, from, accumulate;
        float divisor;
        int kv_side;   // 0 none, 1 the KV write's k, 2 its v
        const float* post_w;
        unsigned post, post_dim;
        float post_limit;
        int kv_half;
    };
    struct ConvArgs {
        ConvItem item[kConvItems];
        float* st;
        const unsigned* tab;
        unsigned long long slot;
        unsigned n, cw;
        void* kc;
        void* vc;
        const int* kv_blocks;   // a view's KV block, then after kv_views of them its slot (kv_targets)
        unsigned bt, kv_views;
        unsigned parts;   // nonzero: llmx_conv_ring_parts, a view's rows in parts of kConvPart on grid.z
    };
    static_assert(sizeof(ConvItem) == 128 && offsetof(ConvItem, C) == 64 && offsetof(ConvItem, divisor) == 96 && offsetof(ConvItem, kv_side) == 100 && offsetof(ConvItem, post) == 112 &&
                      offsetof(ConvItem, kv_half) == 124 && offsetof(ConvArgs, st) == 128 * kConvItems && offsetof(ConvArgs, kc) == 128 * kConvItems + 32 &&
                      offsetof(ConvArgs, parts) == 128 * kConvItems + 64 && sizeof(ConvArgs) == 128 * kConvItems + 72,
                  "ConvArgs is the kernel's layout");
    // Ring items that touch none of each other's writes, up to kConvItems, share one launch; any other list runs as the single calls in order.
    // A launch of one row a view takes its items' post steps and their KV writes where the heads allow (docs/src/backends-gpu.md, ring convolution items).
    ConvDone depthwise_conv_items(const ConvRows* items, size_t n, size_t layer, const StateView* views, size_t n_views, const ConvKv* kv) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows || !n) return ConvDone::none;
        if (conv_in_store(items, n, rows, layer, views, n_views)) return ConvDone::posts;
        const auto out_n = [&](const DepthwiseConv& a) { return size_add(size_mul(rows - 1, a.out_stride ? a.out_stride : a.channels), a.channels); };
        // Whether p's output and ring meet nothing q reads or writes.
        const auto apart = [&](const ConvRows& p, const ConvRows& q) {
            const DepthwiseConv &a = p.conv, &b = q.conv;
            const size_t on = out_n(a), xn = size_mul(rows, b.channels);
            const size_t dn = b.dyn.buffer ? size_add(size_mul(rows - 1, b.dyn_stride), size_mul(b.dyn_heads, b.taps)) : 0;
            return !overlaps(p.out, on, q.out, out_n(b)) && !overlaps(p.out, on, q.x, xn) && !overlaps(p.out, on, b.w, size_mul(b.channels, b.taps)) &&
                   !(b.residual.buffer && overlaps(p.out, on, b.residual, xn)) && !(dn && overlaps(p.out, on, b.dyn, dn)) &&
                   (size_add(a.ring_offset, size_mul(a.ring_rows, a.channels)) <= b.ring_offset || size_add(b.ring_offset, size_mul(b.ring_rows, b.channels)) <= a.ring_offset);
        };
        bool shared = n <= kConvItems;
        for (size_t i = 0; shared && i < n; ++i) {
            shared = items[i].conv.history == ConvHistory::ring;
            for (size_t j = 0; shared && j < i; ++j) shared = apart(items[i], items[j]) && apart(items[j], items[i]);
        }
        bool posts = shared && rows == n_views;
        for (size_t i = 0; posts && i < n; ++i) {
            const DepthwiseConv& a = items[i].conv;
            const size_t D = a.post_heads ? a.channels / a.post_heads : 0;
            posts = a.post == ConvPost::none || (D && a.channels % a.post_heads == 0 && !(D & (D - 1)) && D % dev_.caps.subgroup_size == 0 && D <= kThreads && !a.accumulate &&
                                                 (a.post == ConvPost::l2_clip || (a.post_w.buffer && (!a.out_stride || a.out_stride == a.channels))));
        }
        bool stores = posts && kv && kv->n_views == n_views && kv->views;
        for (size_t v = 0; stores && v < n_views; ++v) stores = views[v].nq == 1 && kv->views[v].nq == 1;
        if (stores) {
            GpuKVStorage<Api>& s = storage(kv->views[0]);
            const size_t heads = s.heads(kv->layer), dim = s.dim(kv->layer);
            for (size_t i : {kv->k, kv->v}) {
                const DepthwiseConv& a = items[i].conv;
                stores = stores && a.post != ConvPost::none && a.post_heads == heads && a.channels == heads * dim && (!a.out_stride || a.out_stride == a.channels);
            }
        }
        if (shared) {
            conv_ring(items, n, layer, views, n_views, posts, stores ? kv : nullptr);
            return !posts ? ConvDone::none : stores ? ConvDone::posts_kv : ConvDone::posts;
        }
        for (size_t i = 0; i < n; ++i) conv_one(items[i].out, items[i].x, items[i].conv, layer, views, n_views);
        return ConvDone::none;
    }
    // One ring convolution of a row with no product or post step and the mix after it, in one launch where every mix item takes llmx_query_mix4 (conv_ring); otherwise the two calls.
    bool depthwise_conv_mix(const ConvRows* items, size_t n, size_t layer, const StateView* views, size_t n_views, const RowMix& then) override {
        const size_t rows = check_state_views(views, n_views, layer);
        const DepthwiseConv& c = items[0].conv;
        if (n != 1 || rows != 1 || n_views != 1 || items[0].from.weight.data.buffer || items[0].from.step || c.history != ConvHistory::ring || c.post != ConvPost::none ||
            then.dim > kQueryMixHeld || then.mix.norm[0].buffer || (then.before && then.before->norm[0].buffer))
            return false;
        std::pair<const void*, size_t> written[kQueryMixItems] = {};
        size_t queries = 0;
        const QueryMixArgs m = then.before ? query_mix_args({*then.before, then.mix}, rows, then.dim, then.stride, then.eps, written, queries)
                                           : query_mix_args({then.mix}, rows, then.dim, then.stride, then.eps, written, queries);
        for (unsigned i = 0; i < m.count; ++i)
            if (m.item[i].n > kQueryMixFew) return false;
        if (conv_ring(items, 1, layer, views, n_views, true, nullptr, &m)) return true;
        if (then.before) query_mix({*then.before, then.mix}, rows, then.dim, then.stride, then.eps, {});
        else query_mix({then.mix}, rows, then.dim, then.stride, then.eps, {});
        return true;
    }
    // A one-row ring convolution with its input as residual and no other step, over one view that keeps its slot, runs in its product's store where that product takes a one-column Q8_0 row kernel.
    // Otherwise the list's products run as their own calls and the convolutions after them, as llmx_conv_ring computes them (kernels/int16.gpu, ConvCols); false then.
    bool conv_in_store(const ConvRows* items, size_t n, size_t rows, size_t layer, const StateView* views, size_t n_views) {
        const ConvProduct& p = items[0].from;
        const DepthwiseConv& a = items[0].conv;
        const StateStorage& ss = *views[0].storage;
        const std::vector<unsigned> tab = state_table(views, n_views);
        const bool fits = n == 1 && (p.weight.data.buffer || p.step) && rows == 1 && n_views == 1 && tab[2] == 1 && tab[4] == tab[5] && a.history == ConvHistory::ring && a.dilation == 1 &&
                          a.taps && a.taps <= 4 && a.ring_rows >= a.taps - 1 && a.ring_rows <= UINT32_MAX && a.channels && a.channels <= UINT32_MAX && !a.dyn.buffer &&
                          a.act == ConvAct::none && a.residual_from == ConvResidual::input && a.post == ConvPost::none && a.divisor > 0.0f && std::isfinite(a.divisor) &&
                          items[0].x.buffer == p.y.buffer && items[0].x.offset == p.y.offset &&
                          (p.step ? !overlaps(items[0].out, a.channels, p.h, p.nin) && !overlaps(items[0].out, a.channels, p.step->act, size_mul(p.step->k, p.step->gate.rows))
                                  : !overlaps(items[0].out, a.channels, p.x, p.nin)) &&
                          ss.slot_floats(layer) >= size_add(a.ring_offset, size_mul(a.ring_rows, a.channels));
        if (!fits) {
            conv_products(items, n, rows);
            return false;
        }
        floats(a.w, size_mul(a.channels, a.taps));
        const ConvStore m{f32(a.w), f32(Slice{&ss.layer(layer), 0}) + size_t(tab[5]) * ss.slot_floats(layer) + a.ring_offset, f32(items[0].out) + size_t(tab[1]) * (a.out_stride ? a.out_stride : a.channels),
                          (unsigned)a.channels, (unsigned)a.taps, (unsigned)a.ring_rows, tab[3], a.divisor, a.accumulate ? 1 : 0};
        conv_stored_ = false;
        if (p.step) {
            // Only the step's routed down with its combine takes it (walk_combine), never a product before it.
            combine_store_ = &m;
            moe_step(*p.step, p.h, p.y, p.nin, rows, p.runs, p.dtype);
            combine_store_ = nullptr;
        } else {
            conv_store_ = &m;
            matmul(p.weight, p.x, p.y, p.nin, a.channels, rows, p.runs, p.dtype);
            conv_store_ = nullptr;
        }
        return conv_stored_;
    }
    // The host's copy of llmx_conv_ring_sum's sum (kernels/f32.gpu, ConvSum).
    struct ConvSum {
        const void* p;
        unsigned long long n, cap, half_at, timeout, tick;
        int members, rank;
    };
    static_assert(sizeof(ConvSum) == 56 && offsetof(ConvSum, members) == 48, "ConvSum is the kernel's layout");
    // depthwise_conv over a ring, a block cw channels of an item (16 to kThreads by the longest view) and its lanes the view's rows (kernels/f32.gpu, llmx_conv_ring).
    // With `mix`, the query mix after it runs in the launch where its blocks fit the convolution's resident grid of kThreads-channel blocks (llmx_conv_ring_mix4); true then.
    bool conv_ring(const ConvRows* items, size_t n_items, size_t layer, const StateView* views, size_t n_views, bool posts, const ConvKv* kv = nullptr, const QueryMixArgs* mix = nullptr) {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return false;
        if (n_views > 65535) throw std::runtime_error(Api::prefix + "a ring convolution past the launch's grid");
        const StateStorage& ss = *views[0].storage;
        const size_t slot = ss.slot_floats(layer);
        ConvArgs args{};
        size_t blocks = 0;
        for (size_t i = 0; i < n_items; ++i) {
            args.item[i] = ring_item(items[i], rows, ss, layer);
            const DepthwiseConv& a = items[i].conv;
            if (posts && a.post != ConvPost::none) {
                if (a.post == ConvPost::rms_norm) floats(a.post_w, a.channels);
                args.item[i].post_w = a.post == ConvPost::rms_norm ? f32(a.post_w) : nullptr;
                args.item[i].post = a.post == ConvPost::l2_clip ? 1u : 2u;
                args.item[i].post_dim = (unsigned)(a.channels / a.post_heads);
                args.item[i].post_limit = a.post_limit;
            }
        }
        if (kv) {
            const KvTargets t = kv_targets(kv->layer, kv->views, kv->n_views);
            t.s->validate_auxiliary(kv->layer, CSlice{}, t.rows);
            args.kc = t.kc, args.vc = t.vc, args.kv_blocks = t.blocks, args.bt = (unsigned)kBlockTokens, args.kv_views = (unsigned)t.rows;
            args.item[kv->k].kv_side = 1, args.item[kv->k].kv_half = t.k_half;
            args.item[kv->v].kv_side = 2, args.item[kv->v].kv_half = t.v_half;
        }
        // The rows of a view take lanes up to 16, the next power of two of the longest view, so a one-row call, which alone takes a post step, keeps a thread a channel.
        size_t longest = 0, lanes = 1;
        for (size_t v = 0; v < n_views; ++v) longest = std::max(longest, views[v].nq);
        while (!posts && lanes < 16 && lanes < longest) lanes *= 2;
        static_assert(kConvRaw / (kThreads / 2) > 6, "the widest staged block, kThreads / 2 channels, stages more rows than its three lags");
        args.cw = (unsigned)(kThreads / lanes);
        for (size_t i = 0; i < n_items; ++i) blocks += args.item[i].blocks = (args.item[i].C + args.cw - 1) / args.cw;
        // A pending whole sum of the one item's input, which the blocks push and take before they read it (kernels/f32.gpu, llmx_conv_ring_sum) where the whole grid is resident, as its waits need.
        std::optional<TakeDesc> sum;
        if (n_items == 1 && summed_out_ && summed_out_ == args.item[0].x) conv_sums_ = true;
        if (pending_ && pending_->take && pending_->take->whole && !pending_->inject && n_items == 1) {
            const TakeDesc& k = *pending_->take;
            const size_t C = args.item[0].C;
            size_t covered = 0;
            for (size_t v = 0; v < n_views; ++v) covered += views[v].nq;
            if (k.out == args.item[0].x && k.own == k.out && k.n == size_mul(rows, C) && covered == rows && C % 4 == 0 && args.cw % 4 == 0 && ((uintptr_t)k.out & 15) == 0 &&
                size_mul(blocks, n_views) <= resident(conv_sum_per_cu_, kConvRingSum)) {
                sum = k;
                pending_.reset();
            }
        }
        const bool mixes = mix && n_items == 1 && n_views == 1 && args.cw == kThreads && args.item[0].C % kThreads == 0 &&
                           mix->item[0].nq + (mix->count > 1 ? mix->item[1].nq : 0) <= blocks && blocks <= resident(conv_mix_per_cu_, kConvRingMix4);
        // A taken sum whose launch throws is pending again, so it is not lost to a peer that waits for it.
        try {
            const std::vector<unsigned> tab = state_table(views, n_views);
            args.st = f32(Slice{&ss.layer(layer), 0});
            args.tab = (const unsigned*)arg(tab.data(), tab.size() * sizeof(unsigned));
            args.slot = slot;
            args.n = (unsigned)n_items;
            if (sum) {
                const ConvSum cs{sum->peers, sum->n, sum->cap, sum->half_at, sum->timeout, sum->tick, sum->members, sum->rank};
                take_status_ = sum->status;
                if (mixes) launch(kConvRingMix4, u32(blocks), 1, Args(args, cs, *mix, conv_mix_bar()));
                else launch(kConvRingSum, u32(blocks), (unsigned)n_views, Args(args, cs));
                sum.reset();
                // Every member's convolution goes to its stream as it is recorded, as the sum it replaces did (GpuCollective::summed).
                flush();
                return mixes;
            }
        } catch (...) {
            if (sum) pending_ = Pending{sum, false, nullptr, nullptr, nullptr, 0, 0};
            throw;
        }
        if (mixes) {
            launch(kConvRingMix4, u32(blocks), 1, Args(args, ConvSum{}, *mix, conv_mix_bar()));
            return true;
        }
        // A view of kConvPartsFrom rows or more whose items write no input of their own runs in parts of kConvPart rows, a thread a channel (llmx_conv_ring_parts).
        bool own_input = false;
        for (size_t i = 0; i < n_items; ++i) own_input = own_input || args.item[i].o == args.item[i].x;
        if (!posts && !kv && !own_input && longest >= kConvPartsFrom) {
            args.parts = 1, args.cw = (unsigned)kThreads, blocks = 0;
            for (size_t i = 0; i < n_items; ++i) blocks += args.item[i].blocks = (args.item[i].C + kThreads - 1) / kThreads;
            launch(kConvRingParts, u32(blocks), (unsigned)n_views, Args(args), u32((longest + kConvPart - 1) / kConvPart));
            return false;
        }
        launch(kConvRing, u32(blocks), (unsigned)n_views, Args(args));
        return false;
    }
    // One ring convolution's checks and its launch item.
    ConvItem ring_item(const ConvRows& item, size_t rows, const StateStorage& ss, size_t layer) {
        const Slice out = item.out;
        const CSlice x = item.x;
        const DepthwiseConv& a = item.conv;
        if (a.dilation != 1 || !a.taps || a.taps > 4) throw std::runtime_error(Api::prefix + "a ring convolution takes dilation 1 and one to four taps");
        const size_t C = a.channels, T = a.taps, R = a.ring_rows;
        const bool dynamic = a.dyn.buffer != nullptr, given = a.residual_from == ConvResidual::given;
        if (!C || C > UINT32_MAX || !(a.divisor > 0.0f) || !std::isfinite(a.divisor) || given != (a.residual.buffer != nullptr) ||
            (dynamic && (!a.dyn_heads || C % a.dyn_heads || a.dyn_stride < size_mul(a.dyn_heads, T))))
            throw std::runtime_error(Api::prefix + "invalid depthwise convolution dimensions");
        if (!R || R < T - 1 || R > UINT32_MAX) throw std::runtime_error(Api::prefix + "depthwise convolution ring shorter than its history");
        const size_t n = size_mul(rows, C), wn = size_mul(C, T), slot = ss.slot_floats(layer);
        const size_t dn = dynamic ? size_add(size_mul(rows - 1, a.dyn_stride), size_mul(a.dyn_heads, T)) : 0;
        if (slot < size_add(a.ring_offset, size_mul(R, C))) throw std::runtime_error(Api::prefix + "depthwise convolution history outside the state slot");
        const size_t os = a.out_stride ? a.out_stride : C, on = size_add(size_mul(rows - 1, os), C);
        const bool strided = os != C || a.accumulate;
        if (os < C) throw std::runtime_error(Api::prefix + "depthwise convolution output rows overlap");
        floats(out, on);
        floats(x, n);
        floats(a.w, wn);
        if (given) floats(a.residual, n);
        if (dynamic) floats(a.dyn, dn);
        const auto exact = [](CSlice p, CSlice q) { return p.buffer == q.buffer && p.offset == q.offset; };
        if ((overlaps(out, on, x, n) && (strided || !exact(out, x))) || (given && overlaps(out, on, a.residual, n) && (strided || !exact(out, a.residual))))
            throw std::runtime_error(Api::prefix + "depthwise convolution output must be disjoint or exactly its input or residual");
        if (overlaps(out, on, a.w, wn) || (dynamic && overlaps(out, on, a.dyn, dn))) throw std::runtime_error(Api::prefix + "depthwise convolution output overlaps its taps");
        const CSlice state{&ss.layer(layer), 0};
        const size_t sn = ss.layer(layer).size() / sizeof(float);
        if (overlaps(out, on, state, sn) || overlaps(x, n, state, sn) || overlaps(a.w, wn, state, sn) || (given && overlaps(a.residual, n, state, sn)) || (dynamic && overlaps(a.dyn, dn, state, sn)))
            throw std::runtime_error(Api::prefix + "depthwise convolution operands overlap the state layer");
        return {f32(out), f32(x), f32(a.w), dynamic ? f32(a.dyn) : nullptr, given ? f32(a.residual) : nullptr, a.ring_offset, a.dyn_stride, os, (unsigned)C, (unsigned)T, (unsigned)R,
                dynamic ? (unsigned)(C / a.dyn_heads) : 1u, u32((C + kThreads - 1) / kThreads), a.act == ConvAct::silu, a.residual_from == ConvResidual::input ? 1 : given ? 2 : 0,
                a.accumulate, a.divisor, 0};
    }
    // The host's copy of attention_rel_sink's key operands (kernels/f32.gpu, RelKeys), field for field.
    struct RelKeys {
        const float* kn;
        const float* vn;
        const unsigned* tab;
        const float* st;
        const float* aux;
        unsigned long long slot, ring, block_floats;
        unsigned R, rows_floats, width, kd;
    };
    static_assert(sizeof(RelKeys) == 80 && offsetof(RelKeys, slot) == 40 && offsetof(RelKeys, kd) == 76, "RelKeys is the kernel's layout");
    // The host's copy of the tile's relative form (kernels/f32.gpu, RelTile), field for field.
    struct RelTile {
        RelKeys a;
        const float* rel;
        const float* table;
        const float* sink;
        unsigned NR, extent, window, tile_rows;
    };
    static_assert(sizeof(RelTile) == 120 && offsetof(RelTile, rel) == 80 && offsetof(RelTile, NR) == 104 && offsetof(RelTile, tile_rows) == 116, "RelTile is the kernel's layout");
    static constexpr size_t kRelTileValues = 16;   // kernels/f32.gpu, the most relative values a row the tile stages
    static constexpr size_t kRelScratchBytes = size_t(16) << 20;   // the most scratch one launch of the row kernel's key parts takes
    // The rows a launch of the row kernel takes at `parts` key parts a row: all of them in one part, else as many as kRelScratchBytes and the block counters hold.
    static size_t rel_slice_rows(size_t heads, size_t parts, size_t D) {
        if (parts <= 1) return SIZE_MAX;
        return std::max<size_t>(1, std::min(kDeltaCounters / heads, kRelScratchBytes / attention_parts_bytes(1, heads, parts, D)));
    }
    // A block a (row, query head) for the scores and values, then a launch storing the new [k|v] rows (kernels/f32.gpu, llmx_attention_rel_sink and llmx_rel_sink_store).
    // Without tile rows the row kernel also gates its output and writes the gated output's twin, and stores the new rows itself where no block reads a slot it writes.
    void attention_rel_sink(Slice out, Slice q, CSlice k_new, CSlice v_new, Slice rel, CSlice table, CSlice sink, size_t heads, size_t kv_heads, size_t D, size_t R,
                            size_t extent, size_t window, float scale, size_t layer, const StateView* rings, const KVView* pool, size_t n_views, size_t ring_offset,
                            size_t ring_rows, RowRuns runs = {}, const OutputGate& gate = {}, const RelClip& clip = {}) override {
        if (!heads || !kv_heads || heads % kv_heads || !D || !R || !extent || !std::isfinite(scale) || bool(rings) == bool(pool) ||
            (rings && (!ring_rows || (window && ring_rows + 1 < window) || !window)))
            throw std::runtime_error(Api::prefix + "invalid relative attention dimensions");
        if (D > kThreads || kThreads % D || R > 32 || heads > 65535 || extent > UINT32_MAX || window > UINT32_MAX || ring_rows > UINT32_MAX)
            throw std::runtime_error(Api::prefix + "relative attention takes heads of at most 256 floats dividing 256 and at most 32 relative values");
        const size_t KD = size_mul(kv_heads, D), HD = size_mul(heads, D), row_w = 2 * KD;
        std::vector<unsigned> tab(size_mul(n_views, 7));
        size_t rows = 0;
        RelKeys a{};
        if (rings) {
            rows = check_state_views(rings, n_views, layer);
            if (rows && rings[0].storage->slot_floats(layer) < size_add(ring_offset, size_mul(ring_rows, row_w)))
                throw std::runtime_error(Api::prefix + "relative attention ring outside the state slot");
            for (size_t i = 0; i < n_views; ++i) tab[7 * i + 3] = u32(rings[i].src), tab[7 * i + 4] = u32(rings[i].dst);
            if (rows) {
                a.st = f32(CSlice{&rings[0].storage->layer(layer), 0});
                a.slot = rings[0].storage->slot_floats(layer);
            }
            a.ring = ring_offset;
            a.R = u32(ring_rows);
        } else {
            GpuKVStorage<Api>& s = storage(pool[0]);
            for (size_t i = 0; i < n_views; ++i) {
                if (&storage(pool[i]) != &s) throw std::runtime_error(Api::prefix + "relative attention over views of different storages");
                const size_t used = s.check_view(pool[i], layer, true);
                if (s.auxiliary_pool(layer) != 1 || s.auxiliary_pool_width(layer) != row_w || !s.auxiliary_buffer(layer))
                    throw std::runtime_error(Api::prefix + "relative attention over a pool that is not one [k|v] row a position");
                tab[7 * i + 5] = u32(tab.size());
                tab.insert(tab.end(), pool[i].blocks, pool[i].blocks + used);
                rows = size_add(rows, pool[i].nq);
            }
            a.aux = f32(CSlice{s.auxiliary_buffer(layer).get(), 0});
            a.block_floats = s.auxiliary_block_bytes(layer) / sizeof(float);
            a.rows_floats = u32(s.auxiliary_rows_bytes(layer) / sizeof(float));
            a.width = u32(row_w);
        }
        for (size_t i = 0, first = 0; i < n_views; ++i) {
            const size_t nq = rings ? rings[i].nq : pool[i].nq, length = rings ? rings[i].length : pool[i].length;
            tab[7 * i] = u32(first), tab[7 * i + 1] = u32(nq), tab[7 * i + 2] = u32(length);
            if (size_add(length, nq) > UINT32_MAX) throw std::runtime_error(Api::prefix + "relative attention past 32-bit positions");
            first += nq;
        }
        if (!rows) return;
        for (size_t i = 0; i < n_views; ++i) {
            const size_t nq = tab[7 * i + 1], own = rings ? nq : (pool[i].extent ? pool[i].extent : nq);
            tab[7 * i + 6] = u32(std::min<size_t>(nq ? run_extent(runs, rows, tab[7 * i], own) : own, UINT32_MAX));
        }
        if (n_views > 65535 || rows > UINT32_MAX) throw std::runtime_error(Api::prefix + "relative attention past the launch's grid");
        floats(out, size_mul(rows, HD));
        floats(q, size_mul(rows, HD));
        floats(k_new, size_mul(rows, KD));
        floats(v_new, size_mul(rows, KD));
        floats(rel, size_mul(rows, size_mul(heads, R)));
        floats(table, size_mul(R, extent));
        floats(sink, size_mul(rows, heads));
        for (const auto& [in, n] : std::initializer_list<std::pair<CSlice, size_t>>{{q, HD}, {k_new, KD}, {v_new, KD}, {rel, heads * R}, {sink, heads}})
            if (overlaps(out, size_mul(rows, HD), in, size_mul(rows, n))) throw std::runtime_error(Api::prefix + "relative attention output overlaps an input");
        a.kn = f32(k_new);
        a.vn = f32(v_new);
        a.kd = u32(KD);
        // The views and block table in device memory, as the other attention ops take theirs: a pooled row reads a block index a key, which over PCIe cost 11 us a global layer (docs/DEVICE-TUNING.md).
        a.tab = (const unsigned*)pass_arg(tab.data(), tab.size() * sizeof(unsigned));
        float* o = f32(out);
        unsigned nv = u32(n_views), hh = u32(heads), kvh = u32(kv_heads), dd = u32(D), nr = u32(R), ex = u32(extent), wn = u32(window);
        // Views of a prompt whose extent reaches the profile's attention_tile_rows take the tile, the rest the row kernel: by the prompt's extent, its row class (row_class), never by the call's.
        const size_t tile_rows = D == 128 && R <= kRelTileValues ? dev_.profile.attention_tile_rows : 0;
        // Where a kv head serves gpu_shape::kRelTileGroup query heads, a tile block takes all of them and stages each key once (llmx_attention_rel_tile_g3).
        const bool tile_group = heads == size_mul(gpu_shape::kRelTileGroup, kv_heads);
        const size_t tq = tile_group ? gpu_shape::kRelGroupRows : gpu_shape::kRelTileRows;
        size_t tiles = 0, narrow = 0;
        for (size_t i = 0; i < n_views; ++i) {
            const size_t nq = tab[7 * i + 1];
            if (tile_rows && tab[7 * i + 6] >= tile_rows) tiles += (nq + tq - 1) / tq;
            else narrow += nq;
        }
        const unsigned tr = u32(tile_rows);
        // A block takes gq query heads of one kv head, 3 or 4 dividing the kv head's group, in the lanes build where heads are 128 wide with at most 16 relative values; other shapes take the one-head row kernel.
        size_t gq = 1;
        for (size_t d = 3; d <= gpu_shape::kRelGroup; ++d)
            if ((heads / kv_heads) % d == 0) gq = d;
        if (D != 128 || R > 16) gq = 1;
        // The grid depth is the most key parts a narrow row takes (gpu_shape::rel_lane_parts in the lanes build, else rel_parts).
        size_t most = 1;
        for (size_t i = 0; i < n_views; ++i) {
            const size_t nq = tab[7 * i + 1], end = size_add(tab[7 * i + 2], nq);
            const unsigned keys = u32(window ? std::min(window, end) : end);
            if (nq && !(tile_rows && tab[7 * i + 6] >= tile_rows)) most = std::max<size_t>(most, gq > 1 ? gpu_shape::rel_lane_parts(keys) : gpu_shape::rel_parts(keys));
        }
        // The row kernel clips q and rel as it stages them where a block holds one head and the row one key part; otherwise, and with tile rows, which read q from memory, the clip is its own launch first.
        // Every part block and every head of a group would repeat the clip's barriers: 4 us a call at three heads, more than the launch (docs/src/backends-gpu.md).
        const bool clip_in = clip.q_limit > 0.0f && !tiles && gq == 1 && most == 1;
        if (!clip_in) clip_rel(q, rel, heads, D, R, clip);
        const float *cq = nullptr, *cr = nullptr, *cs = nullptr;
        if (clip_in) {
            if (clip.q_scale.buffer) span(*clip.q_scale.buffer, clip.q_scale.offset * sizeof(float), size_mul(HD, sizeof(float))), cq = f32(clip.q_scale);
            if (clip.rel_scale.buffer) span(*clip.rel_scale.buffer, clip.rel_scale.offset * sizeof(float), size_mul(size_mul(heads, R), sizeof(float))), cr = f32(clip.rel_scale);
            if (clip.row_scale.buffer) span(*clip.row_scale.buffer, clip.row_scale.offset * sizeof(float), size_mul(rows, sizeof(float))), cs = f32(clip.row_scale);
        }
        if (tiles) {
            const size_t blocks = size_mul(tiles, tile_group ? kv_heads : heads);
            if (blocks > UINT32_MAX) throw std::runtime_error(Api::prefix + "relative attention past the launch's grid");
            const RelTile rt{a, f32(rel), f32(table), f32(sink), nr, ex, wn, u32(tile_rows)};
            if (tile_group) launch(kAttentionRelTileG3, u32(blocks), 1, Args(f32(q), o, rt, int(nv), int(hh), int(kvh), scale), 1, gpu_shape::kRelGroupThreads);
            else launch(kAttentionRelTile, u32(blocks), 1, Args(f32(q), o, rt, int(nv), int(hh), int(kvh), scale));
        }
        // A ring slot a row stores into holds a position no row reads once nq <= ring_rows + 1 - window; a pooled row past the length is never read from the pool.
        bool in_place = !tiles && heads / gq * kThreads >= row_w;
        for (size_t i = 0; rings && i < n_views; ++i) in_place = in_place && rings[i].src == rings[i].dst && rings[i].nq + window <= ring_rows + 1;
        const bool gated = gate.gate.buffer && !tiles && gate.rows == rows;
        const float* gp = nullptr;
        unsigned long long gs = 0, ghs = 0;
        if (gated) {
            const size_t last = size_add(size_add(size_mul(rows - 1, gate.stride), size_mul(heads - 1, gate.head_stride)), D);
            span(*gate.gate.buffer, gate.gate.offset * sizeof(float), size_mul(last, sizeof(float)));
            gp = f32(gate.gate), gs = gate.stride, ghs = gate.head_stride;
        }
        const StreamTwin t = gated ? stream_twin(out, rows, HD, gate.runs, D % 32 == 0) : StreamTwin{};
        const int store = in_place ? 1 : 0;
        // Rows run in slices whose parts fit the scratch and the counters (rel_slice_rows).
        const size_t slice = std::min(rows, rel_slice_rows(heads, most, D));
        float* pp = nullptr;
        unsigned* cp = nullptr;
        if (narrow && most > 1) {
            const size_t need = attention_parts_bytes(slice, heads, most, D);
            if (!parts_ || parts_->size() < need) parts_ = alloc(need, Memory::device);
            pp = (float*)as_gpu<Api>(*parts_).data();
            cp = block_counters();
        }
        for (size_t g0 = 0; narrow && g0 < rows; g0 += slice) {
            Args args(o, f32(q), f32(rel), f32(table), f32(sink), a, nv, hh, kvh, dd, nr, ex, wn, scale, tr, gp, gs, ghs, store, t.q, t.d, t.s, t.level, t.map, t.major, pp, cp, u32(g0));
            // Only the one-head instance takes the clip (the lanes builds never clip); a group of three or four heads takes its lanes build, ring or pooled keys alike.
            if (gq == 1) args.put(cq), args.put(cr), args.put(cs), args.put(clip_in ? clip.q_limit : 0.0f), args.put(clip.rel_limit);
            launch(gq == 1 ? kAttentionRelSink : gq == 3 ? kAttentionRelLanesG3 : kAttentionRelLanesG4, u32(std::min(slice, rows - g0)), u32(heads / gq), args, u32(most));
        }
        // A row chunk a workgroup beside the row's column blocks, at most kStoreChunks, so a prompt's rows store in parallel.
        constexpr size_t kStoreChunks = 64;
        if (!in_place) {
            size_t longest = 1;
            for (size_t i = 0; i < n_views; ++i) longest = std::max<size_t>(longest, tab[7 * i + 1]);
            const unsigned cols = u32((row_w + kThreads - 1) / kThreads), chunks = u32(std::min(longest, kStoreChunks));
            launch(kRelSinkStore, cols * chunks, u32(n_views), Args(a, cols, chunks));
        }
        if (gated) keep(t.t, o, size_mul(rows, HD));
        else if (gate.gate.buffer) sigmoid_mul(out, out, gate.gate, gate.rows, heads, D, gate.stride, gate.head_stride, gate.runs);
    }
    // DeepSeek-V4.1's layer ops (docs/DEEPSEEK4.md, section 4.2), a kernel each (kernels/f32.gpu); the pooled-row ops read the auxiliary plane through pooled_table.
    void hc_coeffs(Slice pre, Slice post, Slice comb, CSlice mixes, CSlice stream, CSlice scale, CSlice base, size_t rows, size_t hc, size_t dim, float norm_eps, float eps,
                   size_t iters) override {
        hc_launch(&pre, post, comb, mixes, stream, scale, base, eps, iters, nullptr, {}, {}, rows, hc, dim, norm_eps);
    }
    void hc_collapse_norm(Slice dst, CSlice stream, CSlice pre, CSlice w, size_t rows, size_t hc, size_t dim, float eps) override {
        hc_launch(nullptr, {}, {}, {}, stream, {}, {}, 0.0f, 0, &dst, pre, w, rows, hc, dim, eps);
    }
    void hc_coeffs_collapse(Slice pre, Slice post, Slice comb, CSlice mixes, CSlice stream, CSlice scale, CSlice base, float hc_eps, size_t iters, Slice dst, CSlice pre_in,
                            CSlice w, size_t rows, size_t hc, size_t dim, float eps) override {
        hc_launch(&pre, post, comb, mixes, stream, scale, base, hc_eps, iters, &dst, pre_in, w, rows, hc, dim, eps);
    }
    void hc_expand(Slice dst, CSlice f, CSlice stream, CSlice post, CSlice comb, size_t rows, size_t hc, size_t dim) override {
        if (!hc || hc > 16 || !dim) throw std::runtime_error(Api::prefix + "invalid mHC expansion dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(hc, dim)), fn = size_mul(rows, dim), pn = size_mul(rows, hc), cn = size_mul(pn, hc);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{dst, n}, {f, fn}, {stream, n}, {post, pn}, {comb, cn}}) floats(s, count);
        const bool in_place = dst.buffer == stream.buffer && dst.offset == stream.offset;
        if ((!in_place && overlaps(dst, n, stream, n)) || overlaps(dst, n, f, fn) || overlaps(dst, n, post, pn) || overlaps(dst, n, comb, cn))
            throw std::runtime_error(Api::prefix + "mHC expansion output must be disjoint or exactly the stream");
        float* o = f32(dst);
        const float *fp = f32(f), *x = f32(stream), *p = f32(post), *c = f32(comb);
        unsigned h = (unsigned)hc, d = u32(dim);
        unsigned long long lanes = fn;
        const Args args(o, fp, x, p, c, h, d, lanes);
        launch(kHcExpand, u32((fn + kThreads - 1) / kThreads), 1, args);
    }
    // qdq and pack are one kernel, a lane a value (kernels/f32.gpu, llmx_model_format_qdq), writing values in place or each row's packed blocks to `out`.
    void model_format(float* out, size_t stride, CSlice x, size_t count, size_t dim, ModelFormat format) {
        if (format == ModelFormat::f32) throw std::runtime_error(Api::prefix + "f32 is no model format");
        const size_t block = format_block_values(format);
        if (count % block || dim % block) throw std::runtime_error(Api::prefix + "a model format's values are not whole blocks");
        if (!count) return;
        floats(x, count);
        float* p = const_cast<float*>(f32(x));
        unsigned long long blocks = count / block;
        unsigned fm = unsigned(format), row_blocks = u32(dim / block), st = u32(stride), words = u32(format_words(format, dim));
        float inv8 = 1.0f / quant::model_formats::FP8_MAX, inv4 = 1.0f / quant::model_formats::FP4_MAX;
        const Args args(p, out, blocks, fm, inv8, inv4, row_blocks, st, words);
        launch(kModelFormatQdq, u32((count + kThreads - 1) / kThreads), 1, args);
    }
    void model_format_qdq(Slice x, size_t count, ModelFormat format) override { model_format(nullptr, 0, x, count, format_block_values(format), format); }
    void model_format_pack(Slice out, size_t stride, CSlice x, size_t rows, size_t dim, ModelFormat format) override {
        const size_t words = format_words(format, dim);
        if (stride < words) throw std::runtime_error(Api::prefix + "packed rows closer than a row's words");
        if (!rows) return;
        const size_t on = size_add(size_mul(rows - 1, stride), words), xn = size_mul(rows, dim);
        floats(out, on);
        if (overlaps(out, on, x, xn)) throw std::runtime_error(Api::prefix + "packed model format output must be disjoint from its input");
        model_format(f32(out), stride, x, xn, dim, format);
    }
    // topk_rows and candidate_blocks are one kernel, a row a block (kernels/f32.gpu, llmx_topk_rows).
    void topk_rows(Slice ids, CSlice scores, CSlice lens, size_t rows, size_t n, size_t k, uint32_t offset) override {
        if (k > n || n >= UINT32_MAX || size_t(offset) + n >= UINT32_MAX) throw std::runtime_error(Api::prefix + "invalid top-k rows dimensions");
        if (!rows || !k) return;
        topk_select(ids, scores, lens, rows, n, k, k, offset, 0, 1, n);
    }
    // A DSpine drafter's ops, each a kernel of kernels/f32.gpu doing the CPU's arithmetic (Backend::unit_rows and the three after it).
    void unit_rows(Slice dst, Slice norms, CSlice src, size_t rows, size_t n, float gain, float eps, RowScale mode) override {
        if (!n) throw std::runtime_error(Api::prefix + "unit rows of no width");
        if (!rows) return;
        if (rows > UINT32_MAX) throw std::runtime_error(Api::prefix + "unit rows of more rows than a grid holds");
        floats(dst, size_mul(rows, n));
        floats(src, size_mul(rows, n));
        if (norms.buffer) floats(norms, rows);
        unsigned long long nn = n;
        int m = mode == RowScale::rms ? 0 : mode == RowScale::l2 ? 1 : 2;
        const Args args(f32(dst), norms.buffer ? f32(norms) : (float*)nullptr, f32(src), nn, gain, eps, m);
        launch(kUnitRows, (unsigned)rows, 1, args);
    }
    void sigmoid_gate_rows(Slice msg, CSlice q, CSlice s, CSlice v, CSlice bias, CSlice scale, size_t rows, size_t width, const uint32_t* q_rows, const uint32_t* from) override {
        if (!width || width > UINT32_MAX) throw std::runtime_error(Api::prefix + "code messages of no width");
        if (!rows) return;
        if (!q_rows || !from) throw std::runtime_error(Api::prefix + "code messages without their rows");
        size_t qn = 0, fn = 0;
        for (size_t r = 0; r < rows; ++r) {
            if (from[r] == UINT32_MAX) continue;
            qn = std::max(qn, size_t(q_rows[r]) + 1);
            fn = std::max(fn, size_t(from[r]) + 1);
        }
        floats(msg, size_mul(rows, width));
        floats(bias, width);
        if (qn) {
            floats(q, size_mul(qn, width));
            floats(scale, qn);
            floats(s, size_mul(fn, width));
            floats(v, size_mul(fn, width));
        }
        unsigned wd = (unsigned)width;
        const Args args(f32(msg), f32(q), f32(s), f32(v), f32(bias), f32(scale), wd, (const unsigned*)arg(q_rows, rows * sizeof(uint32_t)),
                        (const unsigned*)arg(from, rows * sizeof(uint32_t)));
        launch(kSigmoidGateRows, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)rows, args);
    }
    // The head rows gathered by the embedding's kernel into scratch, then a block a (candidate, group) sums each of the group's rows against its candidate.
    void matmul_at_ids(Slice out, CSlice x, CSlice w, size_t w_stride, WeightView head, size_t width, size_t nrows, CSlice ids, size_t groups, size_t k, size_t per,
                       CSlice add, size_t add_stride) override {
        if (!width || !k || !per || per > k || k > UINT32_MAX || nrows > UINT32_MAX || (w_stride && w_stride < width))
            throw std::runtime_error(Api::prefix + "invalid grouped row dot dimensions");
        if (!groups) return;
        const size_t picks = size_mul(groups, k);
        floats(out, size_mul(picks, k));
        floats(x, size_mul(size_mul(groups, per), width));
        floats(w, size_add(size_mul(groups - 1, w_stride), width));
        floats(ids, picks);
        if (add.buffer) floats(add, size_add(size_mul(groups - 1, add_stride), nrows));
        const size_t need = size_mul(size_mul(picks, width), sizeof(float));
        if (!dot_rows_ || dot_rows_->size() < need) dot_rows_ = alloc(need, Memory::device);
        embed_ids(Slice{dot_rows_.get(), 0}, head, width, nrows, ids, picks);
        unsigned long long wd = width;
        unsigned nr = (unsigned)nrows, kk = (unsigned)k, pp = (unsigned)per;
        unsigned long long ws = w_stride, as = add_stride;
        const Args args(f32(out), f32(x), f32(w), (const float*)as_gpu<Api>(*dot_rows_).data(), (const unsigned*)f32(ids), wd, nr, kk, pp, ws,
                        add.buffer ? f32(add) : (const float*)nullptr, as);
        launch(kMatmulAtIds, kk, (unsigned)groups, args);
    }
    void tap_weighted_rows(Slice dst, CSlice src, size_t width, const uint32_t* idx, size_t rows, size_t taps, CSlice base, CSlice coeff, size_t coeff_stride, size_t group,
                           bool accumulate) override {
        if (!width || !taps || !group || width % group || width > UINT32_MAX || taps > UINT32_MAX || coeff_stride < size_mul(taps, width / group))
            throw std::runtime_error(Api::prefix + "invalid tap weighted rows dimensions");
        if (!rows) return;
        if (!idx) throw std::runtime_error(Api::prefix + "tap weighted rows without their taps' rows");
        if (rows > UINT32_MAX) throw std::runtime_error(Api::prefix + "tap weighted rows of more rows than a grid holds");
        size_t sn = 0;
        for (size_t i = 0; i < size_mul(rows, taps); ++i)
            if (idx[i] != UINT32_MAX) sn = std::max(sn, size_t(idx[i]) + 1);
        const size_t dn = size_mul(rows, width);
        floats(dst, dn);
        floats(base, size_mul(taps, width));
        floats(coeff, size_add(size_mul(rows - 1, coeff_stride), size_mul(taps, width / group)));
        if (sn) floats(src, size_mul(sn, width));
        if (sn && overlaps(dst, dn, src, size_mul(sn, width))) throw std::runtime_error(Api::prefix + "tap weighted rows into their own source");
        unsigned wd = (unsigned)width, tp = (unsigned)taps, gp = (unsigned)group;
        unsigned long long cs = coeff_stride;
        int acc = accumulate ? 1 : 0;
        const Args args(f32(dst), sn ? f32(src) : (const float*)nullptr, f32(base), f32(coeff), (const unsigned*)arg(idx, size_mul(rows, taps) * sizeof(uint32_t)), wd, tp, cs, gp, acc);
        launch(kTapWeightedRows, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)rows, args);
    }
    void chain_pick(Slice next, CSlice score, CSlice cand, size_t seqs, size_t steps, size_t k) override {
        if (!k || k > UINT32_MAX) throw std::runtime_error(Api::prefix + "a chain over no candidates");
        if (!seqs || !steps) return;
        if (seqs > UINT32_MAX || steps > UINT32_MAX) throw std::runtime_error(Api::prefix + "a chain of more sequences or steps than 32 bits name");
        floats(next, size_mul(steps, seqs));
        floats(score, size_mul(size_mul(steps, seqs), size_mul(k, k)));
        floats(cand, size_mul(size_mul(steps, seqs), k));
        unsigned sq = (unsigned)seqs, st = (unsigned)steps, kk = (unsigned)k;
        const Args args((unsigned*)f32(next), f32(score), (const unsigned*)f32(cand), sq, st, kk);
        launch(kChainPick, (unsigned)((seqs + kThreads - 1) / kThreads), 1, args);
    }
    void candidate_blocks(Slice blocks, CSlice scores, CSlice lens, size_t rows, size_t n, size_t block_size, size_t k) override {
        if (!block_size || !n || n >= UINT32_MAX) throw std::runtime_error(Api::prefix + "invalid candidate block dimensions");
        if (!rows || !k) return;
        const size_t items = (n + block_size - 1) / block_size;
        topk_select(blocks, scores, lens, rows, n, k, std::min(k, items), 0, 1, block_size, items);
    }
    void compress_pool(Slice out, CSlice kv, CSlice score, CSlice w, size_t groups, size_t ratio, size_t dim, float eps) override {
        if (!ratio || !dim || !std::isfinite(eps) || eps <= 0.0f) throw std::runtime_error(Api::prefix + "invalid compressor pooling dimensions or epsilon");
        if (!groups) return;
        const size_t on = size_mul(groups, dim), in = size_mul(size_mul(groups, ratio), dim);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{out, on}, {kv, in}, {score, in}, {w, dim}}) floats(s, count);
        if (overlaps(out, on, kv, in) || overlaps(out, on, score, in) || overlaps(out, on, w, dim))
            throw std::runtime_error(Api::prefix + "compressor pooling output must be disjoint from its inputs");
        float* o = f32(out);
        const float *k = f32(kv), *s = f32(score), *wp = f32(w);
        unsigned r = u32(ratio), d = u32(dim);
        const Args args(o, k, s, wp, r, d, eps);
        launch(kCompressPool, u32(groups), 1, args);
    }
    void sparse_attention_sink(Slice out, CSlice q, CSlice kv, CSlice ids, CSlice sink, size_t rows, size_t heads, size_t dim, size_t kv_rows, size_t k, float scale,
                               size_t layer, const KVView* views, size_t n_views, CSlice entry_ids, size_t entry_k, ModelFormat kv_format,
                               ModelFormat entry_format, const SinkTerms& terms = {}) override {
        if (!heads || !dim || !std::isfinite(scale)) throw std::runtime_error(Api::prefix + "invalid sparse attention dimensions or scale");
        if (!n_views && (entry_k || entry_ids.buffer)) throw std::runtime_error(Api::prefix + "sparse attention entries without views");
        if (size_add(k, entry_k) > 1024) throw std::runtime_error(Api::prefix + "sparse attention names at most 1024 rows a row");
        if (!rows) return;
        const size_t kw = format_words(kv_format, dim), ew = format_words(entry_format, dim);
        const size_t qn = size_mul(rows, size_mul(heads, dim)), kvn = size_mul(kv_rows, kw), idn = size_mul(rows, k), en = entry_ids.buffer ? size_mul(rows, entry_k) : 0;
        const size_t bn = terms.place_bias.buffer ? size_mul(heads, k) : 0, gn = terms.gate.buffer ? size_mul(rows, heads) : 0;
        const size_t tail_rows = terms.tail.buffer ? terms.tail_rows : 0, tn = size_mul(tail_rows, dim);
        if (terms.tail_rows && !terms.tail.buffer) throw std::runtime_error(Api::prefix + "sparse attention tail rows without their buffer");
        if (terms.gate_form != GateForm::identity && !terms.gate.buffer) throw std::runtime_error(Api::prefix + "sparse attention gate form without its gate");
        if (size_add(kv_rows, tail_rows) >= UINT32_MAX) throw std::runtime_error(Api::prefix + "sparse attention names more rows than U32 ids hold");
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{out, qn}, {q, qn}, {kv, kvn}, {ids, idn}, {sink, heads}}) floats(s, count);
        if (en) floats(entry_ids, en);
        if (bn) floats(terms.place_bias, bn);
        if (gn) floats(terms.gate, gn);
        if (tn) floats(terms.tail, tn);
        if (overlaps(out, qn, q, qn) || overlaps(out, qn, kv, kvn) || overlaps(out, qn, ids, idn) || overlaps(out, qn, sink, heads) || (en && overlaps(out, qn, entry_ids, en)) ||
            (bn && overlaps(out, qn, terms.place_bias, bn)) || (gn && overlaps(out, qn, terms.gate, gn)) || (tn && overlaps(out, qn, terms.tail, tn)))
            throw std::runtime_error(Api::prefix + "sparse attention output must be disjoint from its inputs");
        const PooledTable t = n_views ? pooled_table(layer, views, n_views, rows, ew, "sparse attention", false) : PooledTable{};
        float* o = f32(out);
        const float *qp = f32(q), *kp = f32(kv), *sk = f32(sink);
        const unsigned* ip = (const unsigned*)f32(ids);
        const unsigned* ep = en ? (const unsigned*)f32(entry_ids) : nullptr;
        const float* aux = t.aux;
        const unsigned* tab = t.rows;
        const float *bias = bn ? f32(terms.place_bias) : nullptr, *gate = gn ? f32(terms.gate) : nullptr, *tail = tn ? f32(terms.tail) : nullptr;
        const unsigned gt = terms.gate_form == GateForm::tanh ? 1u : 0u;
        const float* tail_end = tail ? tail + tn : nullptr;
        const unsigned trows = u32(tail_rows);
        // A call passing any term takes the pairs' _terms builds, which follow the plain builds in the same order (kernels/f32.gpu).
        constexpr int terms_at = kSparseScoresFew16Terms - kSparseScoresFew16;
        static_assert(kSparseValuesTile32Terms - kSparseValuesTile32 == terms_at && kSparseScoresFew16Terms == kSparseValuesTile32 + 1, "the _terms builds mirror the plain ones");
        const auto built = [&](Kernel kn) { return bias || gate || tail ? Kernel(kn + terms_at) : kn; };
        unsigned h = u32(heads), d = u32(dim), kk = u32(k), ek = u32(entry_k), kr = u32(kv_rows), tokens = t.tokens, bt = (unsigned)kBlockTokens, rf = t.rows_floats, width = t.width;
        unsigned long long bf = t.block_floats;
        unsigned kf = unsigned(kv_format), ef = unsigned(entry_format), kws = u32(kw);
        // The scratch holds a chunk of rows of at most 32 MiB: the scores by kept row and head, each tile's maxima, the kept rows and their two counts (kernels/f32.gpu).
        const size_t named = size_add(k, entry_k);
        const SinkLayout sl = sink_layout(rows, named, heads);
        const size_t chunk = sl.chunk, den_at = sl.den_at, kept_at = sl.kept_at, counts_at = sl.counts_at, need = sl.need;
        if (!sparse_ || sparse_->size() < need) sparse_ = alloc(need, Memory::device);
        uint8_t* sc = (uint8_t*)as_gpu<Api>(*sparse_).data();
        float *pt = (float*)sc, *den = (float*)(sc + den_at);
        const float** kept = (const float**)(sc + kept_at);
        unsigned* counts = (unsigned*)(sc + counts_at);
        const unsigned nm = u32(named);
        for (size_t first = 0; first < rows; first += chunk) {
            const size_t n = std::min(chunk, rows - first);
            const unsigned fr = u32(first);
            // From kSparseWideRows rows the tile pair runs, its 32-head build at up to 32 heads; below them the few-row pair, its heads a block from the device's units, so that its blocks fill them.
            if (n >= kSparseWideRows) {
                const bool h32 = heads <= kSparseTileHeads32;
                const size_t tw = h32 ? kSparseTileWide32 : kSparseTileWide;
                launch(built(h32 ? kSparseScoresTile32 : kSparseScoresTile), u32((named + tw - 1) / tw), u32(n),
                       Args(pt, den, kept, counts, qp, kp, ip, aux, tab, ep, h, d, kk, ek, scale, kr, tokens, bt, bf, rf, width, kf, ef, kws, fr, bias, tail, trows));
                launch(built(h32 ? kSparseValuesTile32 : kSparseValuesTile), u32((dim + tw - 1) / tw), u32(n),
                       Args(o, (const float*)pt, (const float*)den, (const float* const*)kept, (const unsigned*)counts, sk, h, d, nm, kf, ef, fr, gate, gt, tail, tail_end));
                continue;
            }
            // Scores at twice kSparseFewMost heads take the register-tiled build, tiles twice as wide; the values kernel reads its tiles' maxima at that width.
            const size_t th = sparse_few_heads(dev_.caps, n, heads, named, dim, kSparseFewWide, kSparseFewMost, kSparseFewLeast), tv = std::min(th, kSparseFewMost);
            static constexpr Kernel few[5][2] = {{kSparseScoresFew32, kSparseValuesFew16}, {kSparseScoresFew16, kSparseValuesFew16}, {kSparseScoresFew8, kSparseValuesFew8},
                                                 {kSparseScoresFew4, kSparseValuesFew4}, {kSparseScoresFew2, kSparseValuesFew2}};
            const size_t f = th == 32 ? 0 : th == 16 ? 1 : th == 8 ? 2 : th == 4 ? 3 : 4, sw = th == 32 ? 2 * kSparseFewWide : kSparseFewWide;
            const bool two = sparse_few_two(dev_.caps, n, heads, dim, kSparseFewWide, tv, kSparseFewMost, kSparseFewTwoPerUnit);
            const unsigned tile = u32(sw), vw = u32((two ? 2 : 1) * kSparseFewWide);
            launch(built(few[f][0]), u32((named + sw - 1) / sw), u32(n),
                   Args(pt, den, kept, counts, qp, kp, ip, aux, tab, ep, h, d, kk, ek, scale, kr, tokens, bt, bf, rf, width, kf, ef, kws, fr, bias, tail, trows),
                   u32((heads + th - 1) / th), u32(th == 32 ? 256 : th * kSparseFewWide));
            launch(built(two ? kSparseValuesFew16x2 : few[f][1]), u32((dim + vw - 1) / vw), u32(n),
                   Args(o, (const float*)pt, (const float*)den, (const float* const*)kept, (const unsigned*)counts, sk, h, d, nm, tile, kf, ef, fr, gate, gt, tail, tail_end),
                   u32((heads + tv - 1) / tv),
                   u32(tv * kSparseFewWide));
        }
    }
    void indexer_scores(Slice out, CSlice q, size_t layer, const KVView* views, size_t n_views, CSlice w, float w_scale, size_t heads, size_t dim, ModelFormat key_format,
                        size_t n, CSlice blocks, size_t block_k, size_t block_size, IndexForm form = IndexForm::weighted_relu) override {
        if (form == IndexForm::raw_blocks) {
            if (w.buffer || blocks.buffer || block_k || key_format != ModelFormat::f32)
                throw std::runtime_error(Api::prefix + "raw index scores take no weights, candidate blocks or key format but f32");
            return raw_index_scores(out, q, layer, views, n_views, heads, dim, n, block_size);
        }
        if (!heads || heads > 1024 || !dim || n >= UINT32_MAX || (blocks.buffer && !block_size) || !std::isfinite(w_scale))
            throw std::runtime_error(Api::prefix + "invalid index score dimensions");
        size_t rows = 0;
        for (size_t vi = 0; vi < n_views && views; ++vi) rows = size_add(rows, views[vi].nq);
        if (!rows || !n) return;
        const size_t key_words = format_words(key_format, dim);
        const PooledTable t = pooled_table(layer, views, n_views, rows, key_words, "index scores", false);
        const size_t on = size_mul(rows, n), qn = size_mul(rows, size_mul(heads, dim)), wn = size_mul(rows, heads), bn = blocks.buffer ? size_mul(rows, block_k) : 0;
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{out, on}, {q, qn}, {w, wn}}) floats(s, count);
        if (bn) floats(blocks, bn);
        if (overlaps(out, on, q, qn) || overlaps(out, on, w, wn) || (bn && overlaps(out, on, blocks, bn))) throw std::runtime_error(Api::prefix + "index scores output must be disjoint from its inputs");
        for (size_t vi = 0; vi < n_views; ++vi)
            if (views[vi].nq && size_add(views[vi].length, views[vi].nq) / t.tokens > n) throw std::runtime_error(Api::prefix + "index scores narrower than a row's groups");
        if (rows > 65535) throw std::runtime_error(Api::prefix + "index scores of more rows than a launch holds");
        float* o = f32(out);
        const float *qp = f32(q), *wp = f32(w);
        const float* aux = t.aux;
        const unsigned* tab = t.rows;
        const unsigned* bp = bn ? (const unsigned*)f32(blocks) : nullptr;
        unsigned nn = u32(n), h = u32(heads), d = u32(dim), tokens = t.tokens, bt = (unsigned)kBlockTokens, rf = t.rows_floats, width = t.width, bk = bn ? u32(block_k) : 0u,
                 bs = u32(block_size);
        unsigned long long bf = t.block_floats;
        unsigned kf = unsigned(key_format), kws = u32(key_words);
        // Two tile blocks fit a unit: half its shared memory under HIP, where a block may take it all, and a block's cap under CUDA, whose units hold two such blocks (Api::shared_block_cap).
        // Rows of a view in runs of kIndexTileRows share each decoded key in the tile kernel, taken where its work fills the device (tile_fills_twice); rows a view each, as decode's, keep a block a row.
        const size_t tile_shared = size_mul(size_add(kIndexTileRows, kIndexTilePositions), size_mul(dim + 4, sizeof(float))), tile_bytes = size_add(tile_shared, kIndexTileFixed);
        if (dim % 4 == 0 && rows >= size_mul(n_views, kIndexTileRows / 4) && tile_bytes <= (Api::shared_block_cap ? dev_.caps.shared_bytes : dev_.caps.shared_bytes / 2) &&
            tile_fills_twice(dev_.caps, tile_bytes, kIndexTileRows, kIndexTilePositions, rows, n)) {
            const Args args(o, qp, aux, tab, wp, bp, nn, u32(rows), h, d, w_scale, tokens, bt, bf, rf, width, bk, bs, kf, kws);
            // FP4 power keys of the one dim the integer tile reads take it: the model rounds the query to FP4 power values first (deepseek4's model_format_qdq).
            if (key_format == ModelFormat::fp4_power && dim == kIndexIntDim && kernels_[kIndexerScoresInt])
                return launch(kIndexerScoresInt, u32((n + kIndexIntPositions - 1) / kIndexIntPositions), u32((rows + kIndexIntRows - 1) / kIndexIntRows), args);
            return launch(kIndexerScoresTile, u32((n + kIndexTilePositions - 1) / kIndexTilePositions), u32((rows + kIndexTileRows - 1) / kIndexTileRows), args, 1, kIndexTileThreads,
                          u32(tile_shared));
        }
        unsigned per = std::max(1u, (unsigned)kThreads / h);
        const Args args(o, qp, aux, tab, wp, bp, nn, h, d, w_scale, tokens, bt, bf, rf, width, bk, bs, kf, kws, per);
        launch(kIndexerScores, u32((n + per - 1) / per), (unsigned)rows, args);
    }
    // indexer_scores' raw form over the token rows of the layer's auxiliary plane, kRawIndexRows rows by 4 heads and one block a block (kernels/f32.gpu, llmx_raw_index_scores).
    void raw_index_scores(Slice out, CSlice q, size_t layer, const KVView* views, size_t n_views, size_t heads, size_t dim, size_t n, size_t block_size) {
        if (!heads || !dim || !block_size || n >= UINT32_MAX || size_mul(n, block_size) >= UINT32_MAX) throw std::runtime_error(Api::prefix + "invalid index score dimensions");
        if (!views && n_views) throw std::runtime_error(Api::prefix + "raw index scores without views");
        size_t rows = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            rows = size_add(rows, views[vi].nq);
            if (views[vi].nq && (size_add(views[vi].length, views[vi].nq) + block_size - 1) / block_size > n) throw std::runtime_error(Api::prefix + "raw index scores narrower than a row's blocks");
        }
        if (!rows) return;
        const PooledTable t = pooled_table(layer, views, n_views, rows, dim, "raw index scores", true);
        const size_t on = size_mul(rows, size_mul(heads, n)), qn = size_mul(rows, size_mul(heads, dim));
        floats(out, on);
        floats(q, qn);
        if (overlaps(out, on, q, qn)) throw std::runtime_error(Api::prefix + "index scores output must be disjoint from its inputs");
        if (!n) return;
        if ((rows + kRawIndexRows - 1) / kRawIndexRows > 65535) throw std::runtime_error(Api::prefix + "raw index scores of more rows than a launch holds");
        const Args args(f32(out), f32(q), t.aux, t.rows, u32(rows), u32(heads), u32(dim), u32(n), u32(block_size), (unsigned)kBlockTokens, t.block_floats);
        launch(kRawIndexScores, u32(n), u32((rows + kRawIndexRows - 1) / kRawIndexRows), args, 1, kRawIndexThreads);
    }
    void pool_write(size_t layer, const KVView* views, size_t n_views, CSlice rows) override {
        if (!n_views) return;
        if (!views) throw std::runtime_error(Api::prefix + "pooled rows written without views");
        GpuKVStorage<Api>& s = storage(views[0]);
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error(Api::prefix + "pooled rows written to views of different storages");
            s.check_view(views[vi], layer, true);
        }
        const size_t tokens = s.auxiliary_pool(layer), width = s.auxiliary_pool_width(layer);
        if (!tokens || !s.auxiliary_buffer(layer)) throw std::runtime_error(Api::prefix + "pooled rows written to a layer that keeps none of their width");
        // Each group's first float in the plane, in view order and each view's groups in order.
        const size_t block_floats = s.auxiliary_block_bytes(layer) / sizeof(float), rows_floats = s.auxiliary_rows_bytes(layer) / sizeof(float);
        std::vector<unsigned long long> at;
        for (size_t vi = 0; vi < n_views; ++vi)
            for (size_t g = views[vi].length / tokens; g < size_add(views[vi].length, views[vi].nq) / tokens; ++g) {
                const size_t t = g * tokens;
                at.push_back((unsigned long long)views[vi].blocks[t / kBlockTokens] * block_floats + rows_floats + (t % kBlockTokens) / tokens * width);
            }
        if (at.empty()) return;
        const size_t n = size_mul(at.size(), width);
        floats(rows, n);
        float* aux = f32(Slice{s.auxiliary_buffer(layer).get(), 0});
        const float* src = f32(rows);
        const unsigned long long* ap = (const unsigned long long*)pass_arg(at.data(), at.size() * sizeof(unsigned long long));
        unsigned wd = u32(width);
        unsigned long long nn = n;
        const Args args(aux, src, ap, wd, nn);
        launch(kPoolWrite, u32((n + kThreads - 1) / kThreads), 1, args);
    }
    // A GLU's kernel form (kernels/quant.h, glu_value), its caps checked.
    static int glu_form(const Glu& glu) {
        const float limit = glu.limit, up_limit = glu.up_limit;
        const GluForm form = glu.form;
        if (!(limit >= 0.0f) || !std::isfinite(limit)) throw std::runtime_error(Api::prefix + "a SwiGLU limit that is negative or not finite");
        if (form == GluForm::softcap ? !(limit > 0.0f) || !(up_limit > 0.0f) || !std::isfinite(up_limit) : up_limit != 0.0f)
            throw std::runtime_error(Api::prefix + "a softcap GLU needs positive finite caps, and only it takes an up limit");
        return form == GluForm::softcap ? 2 : form == GluForm::quick_gelu;
    }
    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, Glu glu, RowRuns runs) override {
        const float limit = glu.limit, up_limit = glu.up_limit;
        const int fm = glu_form(glu);
        if (!n) return;
        for (const CSlice& s : {CSlice(dst), gate, up}) span(*s.buffer, s.offset * sizeof(float), size_mul(n, sizeof(float)));
        float* d = f32(dst);
        const float* g = f32(gate);
        const float* u = f32(up);
        unsigned long long nn = n;
        const size_t rows = runs.n ? runs.runs[runs.n - 1].end : 0;
        const TwinOut t = twin_out(dst, n, rows && n % rows == 0 && n / rows % 32 == 0 ? twin_map(runs, rows, n / rows) : twin_map({}, 1, n));
        void* q = t.q;
        float* td = t.d;
        TwinMap map = t.map;
        const Kernel k = !t.q ? kSiluMul : t.int8 ? kSiluMulTwin8 : kSiluMulTwin16, build = twin_build(k, map);
        Args args(d, g, u, nn, limit, fm, up_limit, q, td, t.s);
        if (build != k) args.put(map);
        launch_writing({{d, n}}, build, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t, d, n);
    }
    // A row a block of 64, its reductions in one tree of 64 pairs (kernels/f32.gpu, llmx_moe_route).
    void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, const RouteRule& rule, Slice ids, Slice weights) override {
        if (!k || k > n_expert || k > 256 || n_expert > 1024) throw std::runtime_error(Api::prefix + "routing takes 1 to 256 of at most 1024 experts");
        if (rule.score != RouteRule::Score::softmax && !std::isfinite(rule.scale)) throw std::runtime_error(Api::prefix + "a routing scale that is not finite");
        if (!rows) return;
        span(*scores.buffer, scores.offset * sizeof(float), size_mul(size_mul(rows, n_expert), sizeof(float)));
        for (const Slice& s : {ids, weights}) span(*s.buffer, s.offset * sizeof(float), size_mul(size_mul(rows, k), sizeof(float)));
        const float* sc = f32(scores);
        float* id = f32(ids);
        float* w = f32(weights);
        unsigned ne = (unsigned)n_expert, kk = (unsigned)k;
        int norm = rule.normalize;
        // The sqrt_softplus and sigmoid rules take a wave a row in the CPU's order (kernels/f32.gpu, llmx_route_biased).
        if (rule.score != RouteRule::Score::softmax) {
            const int sig = rule.score == RouteRule::Score::sigmoid;
            if (!sig) rule.factory_eps(Api::prefix);
            if (!(rule.eps >= 0.0f) || !std::isfinite(rule.eps)) throw std::runtime_error(Api::prefix + "a routing eps that is negative or not finite");
            float eps = rule.eps;
            if (rule.bias.buffer) floats(rule.bias, size_mul(rows - 1, rule.bias_stride) + n_expert);
            const float* bias = rule.bias.buffer ? f32(rule.bias) : nullptr;
            unsigned long long nr = rows, stride = rule.bias_stride;
            float scale = rule.scale;
            const Args args(sc, bias, id, w, nr, ne, kk, norm, scale, stride, sig, eps);
            return launch_writing({{id, size_mul(rows, k)}, {w, size_mul(rows, k)}}, kRouteBiased, u32(rows), 1, args, 1, 64);
        }
        rule.softmax_only(Api::name);
        const Args args(sc, id, w, ne, kk, norm);
        launch_writing({{id, size_mul(rows, k)}, {w, size_mul(rows, k)}}, kMoeRoute, (unsigned)rows, 1, args, 1, 64);
    }
    void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nrows, const Routing& routing, RowRuns runs, Dtype dtype) override {
        const size_t entries = size_mul(nrows, routing.k);
        if (projections.size() > 3) throw std::logic_error(Api::prefix + "more than three routed projections");
        if (!entries || !projections.size()) return;
        check_runs(nrows, runs);
        span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nrows, nin), sizeof(float)));
        routed_runs(projections.begin(), projections.size(), X, nin, nrows, routing, false, runs, dtype);
    }
    // The slots' products land in scratch, then each row's weighted sum in slot order joins Y (kernels/f32.gpu, llmx_moe_combine).
    void matmul_experts_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows, const Routing& routing, RowRuns runs, Dtype dtype,
                            const Buffer* from) override {
        const size_t entries = size_mul(nrows, routing.k);
        if (from && (!entries || !nout)) repeat_row(Y, *from, nrows, nout);
        if (!entries || !nout) return;
        if (from) span(*from, 0, size_mul(nout, sizeof(float)));
        if (walk_combine(weight, X, Y, nin, nout, nrows, routing, runs, dtype, from)) return;
        routed_parts(weight, X, nin, nout, nrows, routing, runs, dtype);
        combine_parts(Y, nout, nrows, routing, from);
    }
    // Each entry's product into moe_out_, an entry of no expert left as it was.
    void routed_parts(WeightView weight, CSlice X, size_t nin, size_t nout, size_t nrows, const Routing& routing, RowRuns runs, Dtype dtype) {
        const size_t entries = size_mul(nrows, routing.k);
        check_runs(nrows, runs);
        span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(entries, nin), sizeof(float)));
        const size_t out_bytes = moe_out_bytes(entries, nout);
        if (!moe_out_ || moe_out_->size() < out_bytes) moe_out_ = alloc(out_bytes, Memory::device);
        const Projection one{weight, Slice{moe_out_.get(), 0}, nout};
        routed_runs(&one, 1, X, nin, nrows, routing, true, runs, dtype);
    }
    // moe_out_'s parts, each row's weighted sum in j order, joining Y.
    void combine_parts(Slice Y, size_t nout, size_t nrows, const Routing& routing, const Buffer* from = nullptr) {
        const size_t k = routing.k, entries = size_mul(nrows, k);
        span(*Y.buffer, Y.offset * sizeof(float), size_mul(size_mul(nrows, nout), sizeof(float)));
        span(*routing.weights.buffer, routing.weights.offset * sizeof(float), size_mul(entries, sizeof(float)));
        float* y = f32(Y);
        const float* part = (const float*)as_gpu<Api>(*moe_out_).data();
        const float* w = f32(routing.weights);
        const float* id = f32(routing.ids);
        unsigned long long nr = nrows, no = nout;
        unsigned kk = (unsigned)k;
        // The rows start from `from`'s row, read in this launch (stride 0), or from Y itself.
        const float* y0 = from ? (const float*)as_gpu<Api>(*from).data() : y;
        unsigned long long y0s = from ? 0 : nout;
        const Args args(y, part, w, id, nr, no, kk, y0, y0s);
        launch_writing({{y, size_mul(nrows, nout)}}, kMoeCombine, (unsigned)((nrows * nout + kThreads - 1) / kThreads), 1, args);
    }
    // A routed down whose product the walk build would take adds its entries' weighted sum in the product's launch, up to kWalkEntries entries a row (kernels/int16.gpu, q8_0_walk_combine).
    // Each entry keeps its one-column bits and the sum moe_combine's order, so the launch gives the two launches' bits; false leaves the call to them.
    bool walk_combine(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows, const Routing& routing, RowRuns runs, Dtype dtype, const Buffer* from) {
        constexpr size_t kWalkEntries = 8, kWalkOut = 8, kGridY = 65535, kGroupExperts = 1024;   // kernels/int16.gpu; HIP's grid height; llmx_moe_group's experts
        const size_t k = routing.k, entries = size_mul(nrows, k);
        if (k > kWalkEntries || nrows > kGridY || weight.type != quant::GGML_TYPE_Q8_0 || !integer(weight, dtype)) return false;
        const bool int8 = dtype == Dtype::int8;
        const TypeKernels& kinds = check_weight(weight, nin, "expert matrix", routing.n_expert);
        const TwinKernels& rk = row_form(*kinds.twin, nin, int8);
        for (size_t i = 0; i < runs.n; ++i)
            if (runs.runs[i].extent >= kinds.twin->tile_from[1][int8]) return false;
        const unsigned* ids = (const unsigned*)f32(routing.ids);
        const Projection one{weight, Y, nout};
        RoutedCall c = routed_call(&one, 1, nin, rk, ids, 1, routing.n_expert);
        const Kernel adds[4] = {kRouted16Walk2Add, kRouted16Walk3Add, kRouted16Walk5Add, kRoutedHalf8WalkAdd},
                     conv_adds[4] = {kRouted16Walk2AddConv, kRouted16Walk3AddConv, kRouted16Walk5AddConv, kRoutedHalf8WalkAddConv};
        int h = 3;
        if (int8) {
            // The int8 build holds one half-block step a lane: the calls routed_runs gives llmx_q8_0_routed_half8s1, an entry an expert, at an even block count of at most half a wave.
            const size_t nb = nin / 32;
            if (rk.routed[2] != kRoutedHalf8 || nb % 2 || dev_.caps.subgroup_size != 64 || 2 * nb > dev_.caps.subgroup_size || entries > routing.n_expert) return false;
        } else {
            if (rk.routed[0] != kRouted16 || (rk.groups(entries, routing.n_expert, false) && routing.n_expert <= kGroupExperts)) return false;
            Kernel kernel = kRouted16;
            (void)block_rows(kernel, rk.two_rows[0][0], c.c.cluster, nin, size_mul(nout, entries));
            h = kernel == kRouted16Walk2 ? 0 : kernel == kRouted16Walk3 ? 1 : kernel == kRouted16Walk5 ? 2 : -1;
            if (h < 0) return false;
        }
        check_grid(weight, nin, nout, routing.n_expert);
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, size_mul(routing.n_expert, nout)));
        span(*routing.ids.buffer, routing.ids.offset * sizeof(float), size_mul(entries, sizeof(float)));
        half_span(int8 ? 2 : 0, entries, nin);
        auto [q, d, xs] = twin_of(X, entries, nin, int8, twin_map({}, entries, nin), true, false);
        (void)xs;
        float* y = f32(Y);
        const float* w = f32(routing.weights);
        const float* y0 = from ? (const float*)as_gpu<Api>(*from).data() : y;
        unsigned long long ni = nin, y0s = from ? 0 : nout;
        unsigned kk = (unsigned)k;
        const Args args(c, q, d, ni, y, w, y0, y0s, kk);
        const unsigned blocks = (unsigned)((nout + kWalkOut - 1) / kWalkOut), threads = unsigned(64 * kWalkEntries);
        if (combine_store_ && nrows == 1 && nout == combine_store_->C) {
            launch_writing({{y, nout}, {combine_store_->out, nout}}, conv_adds[h], blocks, 1, Args(c, q, d, ni, y, w, y0, y0s, kk, *combine_store_), 1,
                           threads);
            conv_stored_ = true;
        } else
            launch_writing({{y, size_mul(nrows, nout)}}, adds[h], blocks, (unsigned)nrows, args, 1, threads);
        record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
        return true;
    }
    void moe_act(const MoeStep& m, CSlice h, size_t nin, size_t rows, const Routing& routing, RowRuns runs, Dtype dtype) override {
        if (!walk_glu(m.gate, m.up, m.act, m.glu, h, nin, rows, &routing, runs, dtype)) Backend::moe_act(m, h, nin, rows, routing, runs, dtype);
    }
    void glu_products(const Projection& gate, const Projection& up, Slice act, Glu glu, CSlice X, size_t nin, size_t rows, RowRuns runs, Dtype dtype) override {
        if (rows != 1 || gate.in.buffer || up.in.buffer || !walk_glu(gate, up, act, glu, X, nin, rows, nullptr, runs, dtype)) Backend::glu_products(gate, up, act, glu, X, nin, rows, runs, dtype);
    }
    // Gate and up products that both take the walk build at eight rows a wave, or in the int8 class the half-block build at most three steps a lane, form their GLU and flat twin in their launch (kernels/int16.gpu, q8_0_walk_glu).
    // Routed, an entry is a row's pick, and dense the one row is the one entry of expert 0; each output keeps the bits of the products and silu_mul, and false leaves the act to them.
    bool walk_glu(const Projection& gate, const Projection& up, Slice act_s, const Glu& glu, CSlice h, size_t nin, size_t nrows, const Routing* routing, RowRuns runs, Dtype dtype) {
        constexpr size_t kGluOut = 32, kGluWaves = 8, kGluWalk = 8, kGridY = 65535, kGroupExperts = 1024;   // kernels/int16.gpu; HIP's grid height; llmx_moe_group's experts
        const size_t k = routing ? routing->k : 1, n_expert = routing ? routing->n_expert : 1, entries = size_mul(nrows, k), ff = gate.rows;
        const char* what = routing ? "expert matrix" : "matrix";
        const Projection gu[2] = {gate, up};
        const bool int8 = dtype == Dtype::int8;
        if (!entries || entries > kGridY || !ff || ff % kGluOut || up.rows != ff || want_twin_ != (int8 ? 2 : 1) || !integer(gu, 2, dtype)) return false;
        if (gate.weight.type != quant::GGML_TYPE_Q8_0 || up.weight.type != quant::GGML_TYPE_Q8_0) return false;
        for (const Buffer* b : {h.buffer, (const Buffer*)act_s.buffer, (const Buffer*)gate.out.buffer, (const Buffer*)up.out.buffer})
            if (as_gpu<Api>(*b).host_ptr()) return false;
        const TypeKernels& kinds = check_weight(gate.weight, nin, what, n_expert);
        const TwinKernels& rk = row_form(*kinds.twin, nin, int8);
        for (size_t i = 0; i < runs.n; ++i)
            if (runs.runs[i].extent >= kinds.twin->tile_from[0][int8]) return false;
        const unsigned* ids = routing ? (const unsigned*)f32(routing->ids) : nullptr;
        RoutedCall c = routed_call(gu, 2, nin, rk, ids, k, n_expert);
        Kernel kernel = kRouted16;
        if (int8) {
            // The int8 build holds three half-block steps a lane: a routed call of an even block count of at most 96, an entry an expert, which reads each expert's rows once as the grouped build would.
            const size_t nb = nin / 32;
            if (!routing || rk.routed[2] != kRoutedHalf8 || nb % 2 || dev_.caps.subgroup_size != 64 || nb > 3 * dev_.caps.subgroup_size / 2 || entries > n_expert) return false;
            kernel = kRoutedHalf8Walk3Glu;
        } else {
            if (rk.routed[0] != kRouted16 || (routing && rk.groups(entries, n_expert, false) && n_expert <= kGroupExperts)) return false;
            // The build holds eight rows a wave, so it takes only a call the walk takes at eight.
            if (block_rows(kernel, rk.two_rows[0][0], c.c.cluster, nin, size_mul(c.c.total, entries)) != kGluWalk * rows_per_block_) return false;
            if (kernel != kRouted16Walk3 && kernel != kRouted16Walk5) return false;
            kernel = kernel == kRouted16Walk3 ? kRouted16Walk3Glu : kRouted16Walk5Glu;
        }
        const float limit = glu.limit, up_limit = glu.up_limit;
        const int form = glu_form(glu);
        check_runs(nrows, runs);
        span(*h.buffer, h.offset * sizeof(float), size_mul(size_mul(nrows, nin), sizeof(float)));
        if (routing) span(*routing->ids.buffer, routing->ids.offset * sizeof(float), size_mul(entries, sizeof(float)));
        span(*act_s.buffer, act_s.offset * sizeof(float), size_mul(size_mul(entries, ff), sizeof(float)));
        for (const Projection& p : gu) {
            check_weight(p.weight, nin, what, n_expert);
            check_grid(p.weight, nin, p.rows, n_expert);
            span(*p.weight.data.buffer, p.weight.data.offset * sizeof(float), quant::row_bytes(p.weight.type, nin, size_mul(n_expert, p.pitch ? p.pitch : p.rows)));
        }
        half_span(int8 ? 2 : 0, nrows, nin);
        auto [q, d, xs] = twin_of(h, nrows, nin, int8, twin_map({}, nrows, nin), true, false);
        (void)xs;
        // The act's twin goes to the slot h's twin does not hold, as silu_mul's would.
        float* act = f32(act_s);
        const size_t n = entries * ff;
        const TwinOut t = twin_out(act_s, n, twin_map({}, 1, n));
        if (!t.q || t.int8 != int8) throw std::logic_error(Api::prefix + "walk_glu without a twin of its act in the call's class");
        unsigned long long ni = nin;
        const Args args(c, q, d, ni, act, t.q, t.d, t.s, limit, form, up_limit);
        launch_writing({{act, n}}, kernel, (unsigned)(ff / kGluOut), (unsigned)entries, args, 1, unsigned(64 * kGluWaves));
        keep(t, act, n);
        for (const Projection& p : gu) forget_written(f32(p.out), n);
        record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
        return true;
    }
    void add(Slice dst, CSlice src, size_t n) override {
        if (!n) return;
        for (const CSlice& s : {CSlice(dst), src}) span(*s.buffer, s.offset * sizeof(float), size_mul(n, sizeof(float)));
        float* d = f32(dst);
        const float* s = f32(src);
        unsigned long long nn = n;
        const Args args(d, s, nn);
        launch(kAdd, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
    }

    KVLayout kv_layout() const override { return {kBlockTokens}; }
    std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim, size_t max_tokens, KVType k_type, KVType v_type,
                                        const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools,
                                        const std::vector<KVShape>& layer_shapes) override {
        const std::vector<KVShape> shapes = BlockKVStorage::shapes(Api::name, layers, n_head_kv, head_dim, layer_shapes);
        if (shapes.empty()) throw std::runtime_error(Api::prefix + "invalid KV storage shape");
        for (const KVShape& shape : shapes) {
            if (shape.heads && shape.dim == 0) throw std::runtime_error(Api::prefix + "invalid KV storage shape");
            if (shape.heads && shape.dim > kThreads) throw std::runtime_error(Api::prefix + "a KV head wider than " + std::to_string(kThreads) + " is not implemented");
        }
        return std::make_unique<GpuKVStorage<Api>>(*this, shapes, max_tokens, k_type, v_type, auxiliary_widths, auxiliary_pools);
    }
    void kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k, CSlice v, CSlice auxiliary) override {
        if (n_views && !views) throw std::runtime_error(Api::prefix + "KV write without views");
        if (!n_views) {
            if (auxiliary.buffer) throw std::runtime_error(Api::prefix + "auxiliary KV rows supplied without views");
            return;
        }
        const KvTargets t = kv_targets(layer, views, n_views);
        GpuKVStorage<Api>& s = *t.s;
        const size_t floats = size_mul(size_mul(t.rows, s.heads(layer)), s.dim(layer));
        span(*k.buffer, k.offset * sizeof(float), size_mul(floats, sizeof(float)));
        span(*v.buffer, v.offset * sizeof(float), size_mul(floats, sizeof(float)));
        // A layer's auxiliary rows go into its own plane of the same blocks, token-major, through the shared storage's copies (BlockKVStorage::write_aux).
        s.validate_auxiliary(layer, auxiliary, t.rows);
        if (!t.rows) return;
        s.write_aux(layer, views, n_views, auxiliary);
        void* kc = t.kc;
        void* vc = t.vc;
        const float* kp = f32(k);
        const float* vp = f32(v);
        const int* bp = t.blocks;
        const int* sp = t.slots;
        int heads = (int)s.heads(layer), dim = (int)s.dim(layer), bt = (int)kBlockTokens;
        int kh = t.k_half, vh = t.v_half;
        const Args args(kc, vc, kp, vp, bp, sp, heads, dim, bt, kh, vh);
        launch(kKvWrite, (unsigned)t.rows, (unsigned)heads, args);
    }
    // q and k normed and rotated and the KV write in one launch, where every head fits a block and k's and v's rows are dense; otherwise the ops one at a time.
    void norm_rope_kv(Slice q, size_t q_stride, size_t n_head, CSlice q_w, Slice k, CSlice v, size_t kv_stride, size_t n_head_kv, CSlice k_w,
                      const RopeArgs& rope, size_t rows, size_t layer, const KVView* views, size_t n_views) override {
        const size_t dim = 2 * rope.half;
        if (!rows || !n_views || !n_head || !n_head_kv || !rope.half || dim > kThreads || kv_stride != size_mul(n_head_kv, dim))
            return Backend::norm_rope_kv(q, q_stride, n_head, q_w, k, v, kv_stride, n_head_kv, k_w, rope, rows, layer, views, n_views);
        if (!views) throw std::runtime_error(Api::prefix + "KV write without views");
        if (!rope.pos) throw std::runtime_error(Api::prefix + "partial rope without positions");
        const KvTargets t = kv_targets(layer, views, n_views);
        GpuKVStorage<Api>& s = *t.s;
        if (t.rows != rows || !s.holds(layer, n_head_kv, dim)) throw std::runtime_error(Api::prefix + "norm_rope_kv inputs do not match the KV storage");
        span(*q.buffer, q.offset * sizeof(float), size_mul(size_add(size_mul(rows - 1, q_stride), size_mul(n_head, dim)), sizeof(float)));
        for (const CSlice& x : {CSlice(k), v}) span(*x.buffer, x.offset * sizeof(float), size_mul(size_mul(rows, kv_stride), sizeof(float)));
        for (const CSlice& w : {q_w, k_w})
            if (w.buffer) span(*w.buffer, w.offset * sizeof(float), size_mul(dim, sizeof(float)));
        const size_t table = size_mul(size_add(*std::max_element(rope.pos, rope.pos + rows), 1), rope.half * sizeof(float));
        span(*rope.cos.buffer, rope.cos.offset * sizeof(float), table);
        span(*rope.sin.buffer, rope.sin.offset * sizeof(float), table);
        float* qp = f32(q);
        float* kp = f32(k);
        const float* vp = f32(v);
        const float* qw = q_w.buffer ? f32(q_w) : nullptr;
        const float* kw = k_w.buffer ? f32(k_w) : nullptr;
        const float* c = f32(rope.cos);
        const float* sn = f32(rope.sin);
        const unsigned* p = (const unsigned*)pass_arg(rope.pos, rows * sizeof(uint32_t));
        unsigned long long qs = q_stride;
        int nh = (int)n_head, nkv = (int)n_head_kv, hf = (int)rope.half, bt = (int)kBlockTokens, kh = t.k_half, vh = t.v_half;
        float eps = rope.eps;
        void* kc = t.kc;
        void* vc = t.vc;
        const int* bp = t.blocks;
        const int* sp = t.slots;
        const Args args(qp, qs, nh, qw, kp, vp, nkv, kw, hf, eps, c, sn, p, kc, vc, bp, sp, bt, kh, vh);
        launch(kNormRopeKv, (unsigned)rows, (unsigned)(n_head + n_head_kv), args);
    }
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out, int n_head, int n_head_kv, int head_dim,
                   const AttentionTerms* terms = nullptr) override {
        attend(Q, layer, views, n_views, out, n_head, n_head_kv, head_dim, true, terms);
    }
    // The gate rewrites the output and writes the twin a product reads, so the attention writes none of its own; attention with terms applies it in its final store.
    void gated_attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out, int n_head, int n_head_kv, int head_dim, const OutputGate& g,
                         const AttentionTerms* terms = nullptr) override {
        if (attend(Q, layer, views, n_views, out, n_head, n_head_kv, head_dim, false, terms, &g)) return;
        sigmoid_mul(out, out, g.gate, g.rows, (size_t)n_head, (size_t)head_dim, g.stride, g.head_stride, g.runs);
    }
    // One decode row of one causal view on the chunk kernel, heads 256 wide, one rope stream and each gate beside its q head in raw, runs in llmx_attention_gated256 (and its merge past one split); anything else takes the default calls.
    void gated_attention_layer(const GatedLayer& in, size_t layer, const KVView* views, size_t n_views, Slice out, int n_head, int n_head_kv, int head_dim, const OutputGate& g) override {
        const bool fits = g.rows == 1 && n_views == 1 && views && views[0].nq == 1 && views[0].causal && head_dim == 256 && in.streams.count == 1 && in.rope_dim &&
                          in.rope_dim % 2 == 0 && in.rope_dim <= (size_t)head_dim && in.pos && in.q_w.buffer && in.k_w.buffer && n_head > 0 && n_head_kv > 0 && n_head % n_head_kv == 0 &&
                          g.gate.buffer == in.raw.buffer && g.gate.offset == in.raw.offset + (size_t)head_dim && g.stride == 2 * (size_t)n_head * (size_t)head_dim && g.head_stride == 2 * (size_t)head_dim &&
                          (views[0].extent ? views[0].extent : 1) < dev_.profile.attention_tile_rows;
        if (!fits) return Backend::gated_attention_layer(in, layer, views, n_views, out, n_head, n_head_kv, head_dim, g);
        GpuKVStorage<Api>& s = storage(views[0]);
        s.validate_auxiliary(layer, CSlice{}, 1);
        if (!s.holds(layer, (size_t)n_head_kv, (size_t)head_dim)) throw std::runtime_error(Api::prefix + "attention outside the KV view");
        const KvTargets t = kv_targets(layer, views, n_views);
        const KVView& view = views[0];
        const size_t used = s.check_view(view, layer, false);
        const size_t D = (size_t)head_dim, Hq = (size_t)n_head, Hkv = (size_t)n_head_kv, half = in.rope_dim / 2, floats = Hq * D;
        span(*in.raw.buffer, in.raw.offset * sizeof(float), size_mul(2 * Hq * D, sizeof(float)));
        span(*in.k.buffer, in.k.offset * sizeof(float), size_mul(Hkv * D, sizeof(float)));
        span(*in.v.buffer, in.v.offset * sizeof(float), size_mul(Hkv * D, sizeof(float)));
        span(*out.buffer, out.offset * sizeof(float), size_mul(floats, sizeof(float)));
        for (const CSlice& w : {in.q_w, in.k_w}) span(*w.buffer, w.offset * sizeof(float), size_mul(D, sizeof(float)));
        const size_t table_bytes = size_mul(size_add(in.pos[0], 1), half * sizeof(float));
        span(*in.cos.buffer, in.cos.offset * sizeof(float), table_bytes);
        span(*in.sin.buffer, in.sin.offset * sizeof(float), table_bytes);
        // The view's blocks, its end and the new token's (block, slot) in one pass argument.
        std::vector<int32_t> table(view.blocks, view.blocks + used);
        const int32_t len = (int32_t)(view.length + 1);
        table.push_back(len);
        const size_t nt = used;
        const int* tp = (const int*)pass_arg(table.data(), table.size() * sizeof(int32_t));
        const int* ep = tp + nt;
        const unsigned* pp = (const unsigned*)pass_arg(in.pos, sizeof(uint32_t));
        const float* raw = f32(in.raw);
        const float* qw = f32(in.q_w);
        const float* kw = f32(in.k_w);
        const float* kr = f32(in.k);
        const float* vp = f32(in.v);
        const float* c = f32(in.cos);
        const float* sn = f32(in.sin);
        float* o = f32(out);
        float eps = in.eps, scale = 1.0f / std::sqrt((float)head_dim);
        int hf = (int)half, hq = n_head, hkv = n_head_kv, bt = (int)kBlockTokens, kh = t.k_half, vh = t.v_half;
        int chunk = (int)dev_.profile.attention_split_chunk, nsplit = (int)attention_splits(D, (size_t)len), max_parts = (int)dev_.profile.attention_split_max_wide;
        const size_t vneed = attention_parts_bytes(1, Hq, (size_t)nsplit, D);
        if (nsplit > 1 && (!parts_ || parts_->size() < vneed)) parts_ = alloc(vneed, Memory::device);
        float* parts = nsplit > 1 ? (float*)as_gpu<Api>(*parts_).data() : nullptr;
        StreamTwin tw = stream_twin(out, 1, floats, g.runs);
        const int tl = nsplit > 1 ? 0 : tw.level;
        const Args args(raw, qw, kr, vp, kw, eps, c, sn, pp, hf, o, t.kc, t.vc, tp, ep, t.blocks, hq, hkv, bt, kh, vh, scale, nsplit, chunk, max_parts, parts, tw.q, tw.d, tw.s, tl, tw.map, tw.major);
        launch_writing({{o, floats}}, kAttentionGated256, (unsigned)(Hq * nsplit), 1, args);
        if (nsplit > 1) {
            const Args margs(parts, o, nsplit, raw, tw.q, tw.d, tw.s, tw.level, tw.map, tw.major);
            launch_writing({{o, floats}}, kAttentionGatedMerge256, (unsigned)Hq, 1, margs);
        }
        keep(tw.t, o, floats);
    }
    // Returns whether it applied `gate`, which only attention with terms takes.
    bool attend(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out, int n_head, int n_head_kv, int head_dim, bool twin, const AttentionTerms* terms = nullptr,
                const OutputGate* gate = nullptr) {
        if (n_views && !views) throw std::runtime_error(Api::prefix + "attention without views");
        // Terms of neither part are no terms, with the bits of the call without them.
        if (terms && !terms->rank && !terms->rel.buffer && !terms->table.buffer && !terms->sink.buffer) terms = nullptr;
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > (int)kThreads)
            throw std::runtime_error(Api::prefix + "invalid attention dimensions");
        if (!n_views) return false;
        GpuKVStorage<Api>& s = storage(views[0]);
        if (!s.holds(layer, (size_t)n_head_kv, (size_t)head_dim)) throw std::runtime_error(Api::prefix + "attention outside the KV view");
        // Views of a prompt whose extent reaches the profile's attention_tile_rows, causal and of 128- or 256-wide heads, take the tile kernel, the rest the chunk kernel: by the view's extent, its row class, never by how many rows the call holds.
        const size_t tile_rows = dev_.profile.attention_tile_rows;
        std::vector<int32_t> table, firsts, ends, wide, starts, narrow_rows;
        std::vector<RowRun> runs;   // the views as the rows' runs, for the layout of the twin written beside the output
        int32_t row0 = 0, tiles = 0;
        bool windowed = false;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            if (&storage(view) != &s) throw std::runtime_error(Api::prefix + "attention over views of different storages");
            const size_t used = s.check_view(view, layer, false);
            const int32_t first = (int32_t)table.size();
            table.insert(table.end(), view.blocks, view.blocks + used);
            const size_t extent = view.extent ? view.extent : view.nq;
            const bool tiled = !terms && view.causal && !view.window && extent >= tile_rows && (head_dim == 128 || head_dim == 256) && view.nq;
            if (tiled) {
                wide.insert(wide.end(), {row0, (int32_t)view.nq, (int32_t)view.length, first, tiles});
                tiles += (int32_t)((view.nq + 31) / 32);
            }
            // A windowed row's end is its span, the positions from its start, set by its own, through its own or a non-causal view's last (CpuBackend::attention).
            for (size_t b = 0; b < view.nq; ++b) {
                if (!tiled) narrow_rows.push_back((int32_t)firsts.size());
                const size_t own = view.length + b + 1, end = view.causal ? own : view.length + view.nq, begin = view.window && own > view.window ? own - view.window : 0;
                firsts.push_back(first);
                ends.push_back(tiled ? 0 : (int32_t)(end - begin));
                starts.push_back((int32_t)begin);
                windowed = windowed || begin;
            }
            row0 += (int32_t)view.nq;
            if (view.nq) runs.push_back({(size_t)row0, extent});
        }
        const size_t rows = firsts.size(), nr = narrow_rows.size();
        if (!rows) return false;
        const bool narrow = nr != 0;
        const size_t floats = size_mul(size_mul(rows, (size_t)n_head), (size_t)head_dim);
        span(*Q.buffer, Q.offset * sizeof(float), size_mul(floats, sizeof(float)));
        span(*out.buffer, out.offset * sizeof(float), size_mul(floats, sizeof(float)));
        const float* q = f32(Q);
        float* o = f32(out);
        const void* kc = as_gpu<Api>(*s.k_buffer(layer)).data();
        const void* vc = as_gpu<Api>(*s.v_buffer(layer)).data();
        // The table, firsts, ends, tile views, under a window the starts, and the rows of the per-row kernels in one set, which every layer of the pass reads alike.
        const size_t nt = table.size();
        table.insert(table.end(), firsts.begin(), firsts.end());
        table.insert(table.end(), ends.begin(), ends.end());
        table.insert(table.end(), wide.begin(), wide.end());
        if (windowed) table.insert(table.end(), starts.begin(), starts.end());
        table.insert(table.end(), narrow_rows.begin(), narrow_rows.end());
        const int* tp = (const int*)pass_arg(table.data(), table.size() * sizeof(int32_t));
        const int* fp = tp + nt;
        const int* ep = fp + rows;
        const int* sp = ep + rows + wide.size();
        const int* np = sp + (windowed ? rows : 0);
        int ratio = n_head / n_head_kv, bt = (int)kBlockTokens;
        int kh = s.k_type() == KVType::f16, vh = s.v_type() == KVType::f16;
        float scale = 1.0f / std::sqrt((float)head_dim);
        if (terms) {
            attend_terms(*terms, views, n_views, ends, out, q, o, kc, vc, tp, fp, ep, windowed ? sp : nullptr, rows, n_head, n_head_kv, head_dim, kh, vh, scale, gate);
            return true;
        }
        if (!wide.empty()) {
            const int* wv = ep + rows;
            int nviews = (int)(wide.size() / 5);
            const Args args(q, o, kc, vc, tp, wv, nviews, n_head, n_head_kv, bt, kh, vh, scale);
            launch(head_dim == 256 ? kAttentionTile256 : kAttentionTile, (unsigned)(tiles * n_head), 1, args);
        }
        // Each row's history in chunks of its own, merged in order (kernels/f32.gpu), so its bits do not depend on the call.
        const size_t need = attention_parts_bytes(rows, (size_t)n_head, kChunks, (size_t)head_dim);
        if (!parts_ || parts_->size() < need) parts_ = alloc(need, Memory::device);
        float* pp = (float*)as_gpu<Api>(*parts_).data();
        // Rows of the tile kernel's views were answered above: the per-row kernel and its merge launch over the other rows alone (np), the chunk kernel skips them by their end of 0.
        // Heads 128 and 256 wide take the per-row kernel, its splits chunks of each row's length, as many as the longest row needs (a shorter row's extra splits are empty), within the profile's limit for the width.
        if (narrow && (head_dim == 128 || head_dim == 256)) {
            int longest = 0;
            for (int32_t e : ends) longest = std::max(longest, e);
            int chunk = (int)dev_.profile.attention_split_chunk, nsplit = (int)attention_splits((size_t)head_dim, (size_t)longest);
            int max_parts = (int)(head_dim == 256 ? dev_.profile.attention_split_max_wide : dev_.profile.attention_split_max);
            const size_t vneed = attention_parts_bytes(nr, (size_t)n_head, (size_t)nsplit, (size_t)head_dim);
            if (nsplit > 1 && (!parts_ || parts_->size() < vneed)) parts_ = alloc(vneed, Memory::device);
            float* vp = nsplit > 1 ? (float*)as_gpu<Api>(*parts_).data() : nullptr;
            // With no row on the tile kernel, the kernel that writes the output writes the twin the output projection reads beside it.
            const TwinOut t = twin && wide.empty() && !windowed ? twin_out(out, floats, twin_map({runs.data(), runs.size()}, rows, (size_t)n_head * head_dim)) : TwinOut{};
            void* tq = t.q;
            float* td = t.d;
            void* ts = t.s;
            TwinMap map = t.map;
            int tl = !t.q ? 0 : t.int8 ? 2 : 1, none = 0;
            if (windowed) {
                const Args args(q, o, kc, vc, tp, fp, ep, np, n_head, n_head_kv, bt, kh, vh, scale, nsplit, chunk, max_parts, vp, sp);
                launch(head_dim == 256 ? kAttentionVec256Window : kAttentionVecWindow, (unsigned)(nr * n_head * nsplit), 1, args);
            } else {
                const Kernel k = head_dim == 256 ? kAttentionVec256 : kAttentionVec, build = twin_build(k, map);
                Args args(q, o, kc, vc, tp, fp, ep, np, n_head, n_head_kv, bt, kh, vh, scale, nsplit, chunk, max_parts, vp, tq, td, ts, nsplit > 1 ? none : tl);
                if (build != k) args.put(map);
                launch(build, (unsigned)(nr * n_head * nsplit), 1, args);
            }
            if (nsplit > 1) {
                const Kernel mk = head_dim == 256 ? kAttentionVecMerge256 : kAttentionVecMerge, mbuild = twin_build(mk, map);
                Args margs(vp, o, np, n_head, nsplit, tq, td, ts, tl);
                if (mbuild != mk) margs.put(map);
                launch(mbuild, (unsigned)(nr * n_head), 1, margs);
            }
            keep(t, o, floats);
            return false;
        }
        if (narrow && windowed) {
            const Args args(q, pp, kc, vc, tp, fp, ep, n_head, ratio, n_head_kv, head_dim, bt, kh, vh, scale, sp);
            launch(kAttentionChunkWindow, (unsigned)rows, (unsigned)n_head, args, kChunks);
        } else if (narrow) {
            const Args args(q, pp, kc, vc, tp, fp, ep, n_head, ratio, n_head_kv, head_dim, bt, kh, vh, scale);
            launch(kAttentionChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        } else {
            return false;
        }
        const Args merge(pp, o, ep, n_head, head_dim);
        launch(kAttentionMerge, (unsigned)rows, (unsigned)n_head, merge);
        return false;
    }

    // Attention with terms: every row on attention_terms (kernels/f32.gpu), three query heads a block where a KV head's group divides by three, else one.
    // The gate and the twin and the shape by head groups and device are described in docs/src/backends-gpu.md (attention with terms).
    void attend_terms(const AttentionTerms& terms, const KVView* views, size_t n_views, const std::vector<int32_t>& ends, Slice out, const float* q, float* o, const void* kc, const void* vc,
                      const int* tp, const int* fp, const int* ep, const int* sp, size_t rows, int n_head, int n_head_kv, int head_dim, int kh, int vh, float scale, const OutputGate* gate) {
        attention_terms_rows(terms, views, n_views);
        if (head_dim != 128 || terms.rank > 16) throw std::runtime_error(Api::prefix + "attention terms need heads 128 wide and at most 16 relative values a head");
        if (terms.rank) {
            span(*terms.rel.buffer, terms.rel.offset * sizeof(float), size_mul(size_mul(size_mul(rows, (size_t)n_head), terms.rank), sizeof(float)));
            span(*terms.table.buffer, terms.table.offset * sizeof(float), size_mul(size_mul(terms.rank, terms.extent), sizeof(float)));
        }
        if (terms.sink.buffer) span(*terms.sink.buffer, terms.sink.offset * sizeof(float), size_mul(size_mul(rows, (size_t)n_head), sizeof(float)));
        const unsigned g = (n_head / n_head_kv) % 3 == 0 ? 3u : 1u, groups = u32(rows * (size_t)n_head / g), cus = std::max(1u, dev_.caps.compute_units);
        const unsigned threads = g == 3 && size_t(groups) * 8 <= cus ? 512u : 256u;
        const unsigned parts = std::clamp(cus * 8 / (groups * (threads / 64)), 1u, 16u);
        // Each row's parts follow its own count and do not grow with it (shapes.h terms_parts), so the call takes the most any row needs, from the host's ends (ep is device memory).
        unsigned nsplit = 1;
        for (int32_t e : ends) nsplit = std::max(nsplit, gpu_shape::terms_parts(u32(e), parts, threads / 16));
        const size_t need = attention_parts_bytes(rows, (size_t)n_head, nsplit, 128);
        if (nsplit > 1 && (!parts_ || parts_->size() < need)) parts_ = alloc(need, Memory::device);
        float* pp = nsplit > 1 ? (float*)as_gpu<Api>(*parts_).data() : nullptr;
        if (!terms_counter_ || terms_counter_->size() < size_t(groups) * sizeof(unsigned)) {
            terms_counter_ = alloc(size_t(groups) * sizeof(unsigned), Memory::device);
            hold_fill(as_gpu<Api>(*terms_counter_).data(), size_t(groups) * sizeof(unsigned));
        }
        unsigned* counter = (unsigned*)as_gpu<Api>(*terms_counter_).data();
        const float* rel = terms.rank ? f32(terms.rel) : nullptr;
        const float* table = terms.rank ? f32(terms.table) : nullptr;
        const float* sink = terms.sink.buffer ? f32(terms.sink) : nullptr;
        const Kernel k = g == 1 ? kAttentionTermsG1 : threads == 512 ? kAttentionTermsG3x512 : kAttentionTermsG3;
        const size_t HD = size_mul((size_t)n_head, 128);
        const bool gated = gate && gate->gate.buffer && gate->rows == rows;
        const float* gp = nullptr;
        unsigned long long gs = 0, ghs = 0;
        if (gated) {
            const size_t last = size_add(size_add(size_mul(rows - 1, gate->stride), size_mul((size_t)n_head - 1, gate->head_stride)), 128);
            span(*gate->gate.buffer, gate->gate.offset * sizeof(float), size_mul(last, sizeof(float)));
            gp = f32(gate->gate), gs = gate->stride, ghs = gate->head_stride;
        }
        const StreamTwin t = gated ? stream_twin(out, rows, HD, gate->runs) : StreamTwin{};
        const Args args(q, o, kc, vc, tp, fp, ep, sp, u32(n_head), u32(n_head_kv), u32(kBlockTokens), kh, vh, scale, parts, pp, counter, rel, table, u32(terms.rank), u32(terms.extent), sink,
                        gp, gs, ghs, t.q, t.d, t.s, t.level, t.map, t.major);
        launch(k, u32(rows), u32(n_head) / g, args, nsplit, threads);
        if (gated) keep(t.t, o, size_mul(rows, HD));
        else if (gate && gate->gate.buffer) sigmoid_mul(out, out, gate->gate, gate->rows, (size_t)n_head, 128, gate->stride, gate->head_stride, gate->runs);
    }

    // Attention over each row's selection slots through the dense bodies' sparse builds, per-row for heads 128 and 256 wide and chunked otherwise.
    // A row's slots are its group's ids, each a power of two of positions (kernels/f32.gpu, slot_token), and its splits follow the slots alone, never the call.
    void sparse_attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, CSlice ids, size_t id_stride, Slice out, int n_head, int n_head_kv,
                          int head_dim, size_t groups, size_t block_size) override {
        if (!block_size || (block_size & (block_size - 1))) throw std::runtime_error(Api::prefix + "sparse attention blocks of a size that is not a power of two");
        unsigned shift = 0;
        while ((size_t(1) << shift) < block_size) ++shift;
        if (!n_views) return;
        if (!views) throw std::runtime_error(Api::prefix + "sparse attention without views");
        if (!id_stride || shift > 16 || id_stride > (size_t)INT32_MAX >> shift || n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv || head_dim <= 0 ||
            head_dim > (int)kThreads || !groups || size_t(n_head_kv) % groups)
            throw std::runtime_error(Api::prefix + "invalid sparse attention dimensions");
        GpuKVStorage<Api>& s = storage(views[0]);
        if (!s.holds(layer, (size_t)n_head_kv, (size_t)head_dim)) throw std::runtime_error(Api::prefix + "sparse attention outside the KV view");
        std::vector<int32_t> table, firsts, ends;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            if (!view.nq) throw std::runtime_error(Api::prefix + "invalid sparse attention dimensions");
            if (&storage(view) != &s) throw std::runtime_error(Api::prefix + "sparse attention over views of different storages");
            const size_t used = s.check_view(view, layer, false);
            const int32_t first = (int32_t)table.size();
            table.insert(table.end(), view.blocks, view.blocks + used);
            for (size_t b = 0; b < view.nq; ++b) {
                firsts.push_back(first);
                ends.push_back((int32_t)(view.length + b + 1));
            }
        }
        const size_t rows = firsts.size(), floats = size_mul(size_mul(rows, (size_t)n_head), (size_t)head_dim), selected = size_mul(size_mul(rows, groups), id_stride);
        const size_t slots = id_stride << shift;
        if (!Q.buffer || !ids.buffer || !out.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*Q.buffer, Q.offset * sizeof(float), size_mul(floats, sizeof(float)));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(selected, sizeof(uint32_t)));
        span(*out.buffer, out.offset * sizeof(float), size_mul(floats, sizeof(float)));
        if (overlaps(out, floats, Q, floats) || overlaps(out, floats, ids, selected) || out.buffer == s.k_buffer(layer).get() ||
            out.buffer == s.v_buffer(layer).get() || out.buffer == s.auxiliary_buffer(layer).get())
            throw std::runtime_error(Api::prefix + "sparse attention output overlaps an input");
        const float* q = f32(Q);
        float* o = f32(out);
        const unsigned* idp = (const unsigned*)f32(ids);
        const void* kc = as_gpu<Api>(*s.k_buffer(layer)).data();
        const void* vc = as_gpu<Api>(*s.v_buffer(layer)).data();
        const size_t nt = table.size();
        table.insert(table.end(), firsts.begin(), firsts.end());
        table.insert(table.end(), ends.begin(), ends.end());
        const int* tp = (const int*)pass_arg(table.data(), table.size() * sizeof(int32_t));
        const int* fp = tp + nt;
        const int* ep = fp + rows;
        int ratio = n_head / n_head_kv, bt = (int)kBlockTokens, kh = s.k_type() == KVType::f16, vh = s.v_type() == KVType::f16, stride = (int)slots, ng = (int)groups;
        float scale = 1.0f / std::sqrt((float)head_dim);
        if (head_dim == 128 || head_dim == 256) {
            const size_t ids_per = sparse_split_ids(slots);
            int chunk = (int)ids_per, nsplit = (int)((slots + ids_per - 1) / ids_per);
            const size_t need = attention_parts_bytes(rows, (size_t)n_head, (size_t)nsplit, (size_t)head_dim);
            if (nsplit > 1 && (!parts_ || parts_->size() < need)) parts_ = alloc(need, Memory::device);
            float* vp = nsplit > 1 ? (float*)as_gpu<Api>(*parts_).data() : nullptr;
            const Args args(q, o, kc, vc, tp, fp, ep, n_head, n_head_kv, bt, kh, vh, scale, nsplit, chunk, vp, idp, stride, shift, ng);
            // Four heads of a KV head a block, each head its one-head bits (kernels/f32.gpu, sparse_vec_grouped), where the groups fill the compute units, as the Vulkan backend chooses.
            if (ratio % 4 == 0 && rows * size_t(n_head) / 4 >= size_t(dev_.caps.compute_units)) launch(head_dim == 256 ? kSparseVec256G4 : kSparseVecG4, (unsigned)(rows * (n_head / 4) * nsplit), 1, args);
            else launch(head_dim == 256 ? kSparseVec256 : kSparseVec, (unsigned)(rows * n_head * nsplit), 1, args);
            if (nsplit > 1) {
                const Args margs(vp, o, n_head, nsplit);
                launch(head_dim == 256 ? kSparseVecMerge256 : kSparseVecMerge, (unsigned)(rows * n_head), 1, margs);
            }
            return;
        }
        const size_t need = attention_parts_bytes(rows, (size_t)n_head, kChunks, (size_t)head_dim);
        if (!parts_ || parts_->size() < need) parts_ = alloc(need, Memory::device);
        float* pp = (float*)as_gpu<Api>(*parts_).data();
        const Args args(q, pp, kc, vc, tp, fp, ep, n_head, ratio, n_head_kv, head_dim, bt, kh, vh, scale, idp, stride, shift, ng);
        launch(kSparseChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        const Args margs(pp, o, ep, n_head, head_dim, stride);
        launch(kSparseMerge, (unsigned)rows, (unsigned)n_head, margs);
    }

    // The indexer in Vulkan's design: each view's groups pooled, then at most 16 rows at once scored and selected (docs/src/backends-gpu.md).
    void pool_kv(size_t layer, const KVView* views, size_t n_views, const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin, const GroupPositions& positions) override {
        if (!n_views) return;
        const IndexCall c = index_call(layer, views, n_views, index, key_norm, cos, sin, positions, false);
        for (size_t vi = 0; vi < n_views; ++vi) pool_view(c, views[vi], vi, index);
    }
    void select_kv(CSlice query, size_t layer, const KVView* views, size_t n_views, Slice ids, const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin,
                   const GroupPositions& positions) override {
        if (!n_views) return;
        const IndexCall c = index_call(layer, views, n_views, index, key_norm, cos, sin, positions, true);
        if (!c.rows) return;
        const size_t width = size_mul(index.heads, index.dim), stride = size_add(index.budget, index.compression - 1);
        const size_t queries = size_mul(c.rows, width), outputs = size_mul(c.rows, stride);
        span(*query.buffer, query.offset * sizeof(float), size_mul(queries, sizeof(float)));
        if (!ids.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(outputs, sizeof(uint32_t)));
        if (overlaps(ids, outputs, query, queries) || overlaps(ids, outputs, key_norm, index.dim) || overlaps(ids, outputs, cos, c.rope_values) ||
            overlaps(ids, outputs, sin, c.rope_values))
            throw std::runtime_error(Api::prefix + "index selection output overlaps an input");
        if (ids.buffer == c.s->auxiliary_buffer(layer).get() || ids.buffer == c.s->k_buffer(layer).get() || ids.buffer == c.s->v_buffer(layer).get())
            throw std::runtime_error(Api::prefix + "index selection output overlaps KV storage");
        // The scores of 16 rows at most, bounded by the call's longest view and never by its rows.
        const size_t score_bytes = select_score_bytes(c.max_rows, c.longest, index.compression);
        if (score_bytes && (!scores_ || scores_->size() < score_bytes)) scores_ = alloc(score_bytes, Memory::device);
        const float* q = f32(query);
        unsigned* out = (unsigned*)f32(ids);
        float* sc = score_bytes ? f32(Slice{scores_.get(), 0}) : nullptr;
        unsigned heads = u32(index.heads), dim = u32(index.dim), compression = u32(index.compression), budget = u32(index.budget), ostride = u32(stride), bt = (unsigned)kBlockTokens;
        float* aux = c.aux;
        unsigned long long block_floats = c.block_floats;
        unsigned rows_floats = c.rows_floats;
        float scale = 1.0f / std::sqrt((float)index.dim);
        // The wide form's state for the call's most slices, made before the first launch and zeroed as it is made (alloc); each call's emission clears its histograms.
        size_t most_slices = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const size_t count = (views[vi].length + views[vi].nq) / index.compression;
            if (views[vi].nq && count >= dev_.profile.select_wide_from) most_slices = std::max(most_slices, (count + kSelectSlice - 1) / kSelectSlice);
        }
        const size_t state_bytes = select_state_bytes(most_slices);
        if (most_slices && (!select_state_ || select_state_->size() < state_bytes)) select_state_ = alloc(state_bytes, Memory::device);
        unsigned* state = most_slices ? (unsigned*)f32(Slice{select_state_.get(), 0}) : nullptr;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& v = views[vi];
            if (!v.nq) continue;
            pool_view(c, v, vi, index);
            const size_t count = (v.length + v.nq) / index.compression;
            const bool wide = count >= dev_.profile.select_wide_from;
            unsigned groups = u32(count), slices = u32((count + kSelectSlice - 1) / kSelectSlice);
            const int* blocks = c.blocks + c.first[vi];
            // A row that sees at most `whole` tokens takes every group, so its ids are its visible positions in order, all such rows from one launch.
            const size_t whole = size_mul(index.budget / index.compression + 1, index.compression) - 1;
            const size_t taken = whole > v.length ? std::min(v.nq, whole - v.length) : 0;
            if (taken) {
                const Args args(out + c.row0[vi] * stride, u32(v.length), ostride);
                launch(kKvIndexAll, u32(taken), 1, args);
            }
            for (size_t row = taken; row < v.nq; row += kSelectRows) {
                const size_t rows = std::min<size_t>(kSelectRows, v.nq - row);
                const float* qr = q + (c.row0[vi] + row) * width;
                unsigned* o = out + (c.row0[vi] + row) * stride;
                unsigned hist = u32(v.length + row);
                if (count) {
                    const Args sargs(qr, aux, sc, blocks, hist, groups, heads, dim, compression, bt, block_floats, rows_floats, scale);
                    launch(kKvIndexScores, u32((count + 7) / 8), (unsigned)rows, sargs);
                }
                if (!wide) {
                    const Args args(sc, o, hist, groups, compression, budget, ostride);
                    launch(kKvIndexSelect, (unsigned)rows, 1, args);
                    continue;
                }
                for (unsigned stage = 0; stage < 5; ++stage) {
                    const Args args(sc, o, state, hist, groups, compression, budget, ostride, stage, slices);
                    launch(kKvIndexSelectWide, slices, (unsigned)rows, args);
                }
            }
        }
    }

    DeviceProfile& profile() { return dev_.profile; }
    DeviceCaps& caps() { return dev_.caps; }

    // The gate+up build for a grid: the wide shared build, two rows a wave and four steps a load, where the grid is resident at that build's occupancy, else the lean one.
    // A type's lean build is step_build of its twin's step_up and the wide build the next kernel (kernels/int16.gpu, LLMX_STEP_UP_CN).
    auto step_up_kernel(int lean, unsigned c, size_t grid) {
        const auto l = step_build(static_cast<Kernel>(lean), c), wide = static_cast<Kernel>(l + 1);
        return grid <= resident(step_up_per_cu_[c][(lean - kMoeStepUpQ4K) / 2], wide) ? wide : l;
    }
    // The device expert cache (ExpertSlots): the f32 module's mark, wave, pull and round kernels, and a ring a device whose reader fills the waves' misses (backends/expert_ring.hpp).
    bool caches_experts() const override { return kernels_[kExpertWave] != nullptr; }
    // A cached step: route, mark the picks, then rounds of waves and products, each round's down products kept as parts until one combine in j order (docs/src/backends-gpu.md, expert cache).
    // The waves, pulls, rounds and products run in ExpertRing::run's order, the two ring halves alternating.
    void cached_step(const MoeStep& m, Slice h, Slice y, size_t nin, size_t rows, RowRuns runs, Dtype dtype) {
        const ExpertSlots& c = *m.slots;
        const size_t n_expert = m.router.rows, k = m.k, ff = m.gate.rows, entries = size_mul(rows, k);
        if (n_expert != c.n_expert || c.slots > 0xffff || n_expert > 0xffff || m.gate.weight.scales.buffer || m.up.weight.scales.buffer || m.down.scales.buffer)
            throw std::runtime_error(Api::prefix + "an expert cache step of another expert count, over 65535 slots or experts, or of scaled stacks");
        ExpertRing& ring = ExpertRing::of(ring_, *this, c, Api::prefix);
        constexpr size_t chunk = 65536;
        ExpertStacks st{};
        ExpertRing::Source src{};
        size_t chunks = 0;
        for (int i = 0; i < 3; ++i) {
            const ExpertSlots::Stack& s = c.stack[i];
            if (s.stride % 16 || !s.host || s.dst->size() < c.slots * s.stride || s.stride > ring.entry(i))
                throw std::runtime_error(Api::prefix + "an expert cache stack off its alignment, outside its allocation or wider than the ring");
            st.ring[i] = (const void*)as_gpu<Api>(ring.entries(i)).device_ptr();
            st.dst[i] = (void*)as_gpu<Api>(*s.dst).data();
            st.stride[i] = s.stride / 16;
            src.at[i] = s.host;
            src.stride[i] = s.stride;
            src.file[i] = s.file;
            src.file_at[i] = s.file_at;
            chunks = std::max<size_t>(chunks, (s.stride + chunk - 1) / chunk);
        }
        ExpertRing::count_step(src, c, rows);
        if (m.zero) repeat_row(y, *m.zero, rows, nin);
        moe_router(m, h, nin, rows, runs);
        if (!expert_ids_ || expert_ids_->size() < entries * sizeof(unsigned)) expert_ids_ = alloc(entries * sizeof(unsigned), Memory::device);
        unsigned* state = (unsigned*)f32(c.state);
        unsigned* ids = (unsigned*)f32(m.ids);
        unsigned* round_ids = (unsigned*)as_gpu<Api>(*expert_ids_).data();
        uint32_t* xchg = (uint32_t*)as_gpu<Api>(ring.exchange_buffer()).device_ptr();
        const unsigned ne = u32(n_expert), en = u32(entries), sl = u32(c.slots), hw = u32(ring.half()), ch = u32(chunks), fl = u32(c.flags_at());
        const unsigned long long units = chunk / 16;
        launch_writing({{state, c.words()}}, kExpertMark, 1, 1, Args((const unsigned*)ids, state, ne, en, u32(c.age_at()), u32(c.counts_at()), fl));
        const auto wave = [&] {
            const uint64_t seq = ring.next();
            launch_writing({{state, c.words()}}, kExpertWave, 1, 1,
                           Args(state, xchg + ring.half_at(seq), ne, sl, hw, (unsigned long long)seq, u32(c.owner_at()), u32(c.age_at()), u32(c.counts_at()), fl), 1, 64);
            ring.request(submit(), seq, src);
            return seq;
        };
        const auto pull = [&](uint64_t seq) {
            uint32_t* half = xchg + ring.half_at(seq);
            hold_wait((uint64_t*)half, seq);
            launch(kExpertPull, u32(ring.half() * chunks), 3, Args((const unsigned*)half, (unsigned*)xchg + ExpertRing::kFault, st, ch, units, (unsigned long long)seq, (const unsigned*)state + c.counts_at() + 2 + (seq & 1)));
        };
        const size_t out_bytes = moe_out_bytes(entries, nin);
        if (!moe_out_ || moe_out_->size() < out_bytes) moe_out_ = alloc(out_bytes, Memory::device);
        std::vector<RowRun>& entry_runs = *m.scratch;
        entry_runs.clear();
        for (size_t i = 0; i < runs.n; ++i) entry_runs.push_back(RowRun{runs.runs[i].end * k, runs.runs[i].extent});
        const auto round = [&](size_t, bool last) {
            launch_writing({{round_ids, entries}, {state, c.words()}}, kExpertRound, 1, 1,
                           Args((const unsigned*)ids, round_ids, state, (unsigned*)xchg + ExpertRing::kUnplaced, ne, en, fl, u32(last)));
        };
        // A round's products: entries outside it read no expert, so their parts in moe_out_ keep the rounds before, which nothing else writes until the combine.
        const auto products = [&] {
            const Routing routing{Slice{expert_ids_.get(), 0}, m.weights, k, c.slots};
            matmul_experts({m.gate, m.up}, h, nin, rows, routing, runs, dtype);
            silu_mul(m.act, m.gate.out, m.up.out, entries * ff, m.glu, {entry_runs.data(), entry_runs.size()});
            routed_parts(m.down, m.act, ff, nin, rows, routing, runs, dtype);
        };
        ExpertRing::run(ring.plan(entries, n_expert, c.slots), wave, pull, round, products);
        combine_parts(y, nin, rows, Routing{m.ids, m.weights, k, n_expert});
        moe_shared(m, h, y, nin, rows, runs, dtype);
    }
    // Where fused_step holds, a step is three launches (docs/src/backends-gpu.md): route with row gate and twin of h, every gate and up with their SiLU, the down with combine and shared down.
    // Each output keeps the bits of the base's ops (Backend::moe_step), which run every other step.
    void moe_step(const MoeStep& m, Slice h, Slice y, size_t nin, size_t rows, RowRuns runs = {}, Dtype dtype = Dtype::f16) override {
        if (m.slots) return cached_step(m, h, y, nin, rows, runs, dtype);
        if (!fused_step(m, h, nin, rows, runs, dtype)) return Backend::moe_step(m, h, y, nin, rows, runs, dtype);
        moe_input(m, h, nin, rows, runs);
        check_runs(rows, runs);
        const MoeShared& s = *m.shared;
        const size_t k = m.k, ff = m.gate.rows, ne = m.router.rows, entries = size_mul(rows, k);
        const uint32_t gt = m.gate.weight.type;
        span(*h.buffer, h.offset * sizeof(float), size_mul(size_mul(rows, nin), sizeof(float)));
        span(*y.buffer, y.offset * sizeof(float), size_mul(size_mul(rows, nin), sizeof(float)));
        for (const Projection* p : {&m.router, &m.gate, &m.up, &s.gate, &s.up, &s.row_gate}) {
            const bool routed = p == &m.gate || p == &m.up;
            span(*p->out.buffer, p->out.offset * sizeof(float), size_mul(routed ? size_mul(entries, ff) : size_mul(rows, p->rows), sizeof(float)));
            check_weight(p->weight, nin, "moe step matrix");
            span(*p->weight.data.buffer, p->weight.data.offset * sizeof(float), quant::row_bytes(p->weight.type, nin, routed ? size_mul(ne, ff) : p->rows));
        }
        check_weight(m.down, ff, "expert matrix");
        check_weight(s.down, ff, "moe step matrix");
        span(*m.down.data.buffer, m.down.data.offset * sizeof(float), quant::row_bytes(m.down.type, ff, size_mul(ne, nin)));
        span(*s.down.data.buffer, s.down.data.offset * sizeof(float), quant::row_bytes(s.down.type, ff, nin));
        span(*m.act.buffer, m.act.offset * sizeof(float), size_mul(size_mul(entries, ff), sizeof(float)));
        span(*s.act.buffer, s.act.offset * sizeof(float), size_mul(size_mul(rows, ff), sizeof(float)));
        for (const Slice& r : {m.ids, m.weights}) span(*r.buffer, r.offset * sizeof(float), size_mul(entries, sizeof(float)));
        // The shared act's twin: rows ff-value rows of int8, then their block scales from the half.
        const size_t sact_half = step_twin_bytes(rows, ff) / 2;
        if (!step_twin_ || step_twin_->size() < 2 * sact_half) step_twin_ = alloc(2 * sact_half, Memory::device);
        const size_t out_bytes = moe_out_bytes(entries, nin);
        if (!moe_out_ || moe_out_->size() < out_bytes) moe_out_ = alloc(out_bytes, Memory::device);
        const float* x = f32(h);
        // The twin of h unless kept, and the routed act's twin in the other slot, both kept as the base's products keep them.
        const TwinMap hm = twin_map({}, rows, nin), am = twin_map({}, entries, ff);
        const bool h_kept = find_twin(x, rows * nin / 32, true, hm, false, twin_bytes(hm.total, true));
        const TwinPlanes ht = twin_planes((uint8_t*)as_gpu<Api>(*twin_[twin_slot_]).data(), hm.total, true);
        uint8_t* hq = ht.q;
        float* hd = ht.d;
        if (!h_kept) keep_twin({x, rows * nin, true, hm, twin_slot_});
        float* act = f32(m.act);
        forget_written(act, entries * ff);
        (void)find_twin(act, entries * ff / 32, true, am, false, twin_bytes(am.total, true));
        const TwinPlanes at = twin_planes((uint8_t*)as_gpu<Api>(*twin_[twin_slot_]).data(), am.total, true);
        uint8_t* aq = at.q;
        float* ad = at.d;
        const size_t act_slot = twin_slot_;
        // The shared act's twin, which only the fused down reads, in a buffer of its own: no twin slot is written while the launch writing it reads them.
        uint8_t* sq = (uint8_t*)as_gpu<Api>(*step_twin_).data();
        float* sd = (float*)(sq + sact_half);
        unsigned* ids = (unsigned*)f32(m.ids);
        const StepRoute r{bytes(m.router.weight.data), x, f32(m.router.out), nin, ne, bytes(s.row_gate.weight.data), f32(s.row_gate.out), hq, hd, h_kept ? 0 : rows * nin / 32,
                          ids, f32(m.weights), (unsigned)ne, (unsigned)k, m.rule.normalize ? 1 : 0, step_counter(), (unsigned)rows};
        // The router's and the row gate's types pick the build of their two exact decoders (TypeKernels::step_route), so each build holds its own decoders' registers alone.
        launch_step(step_route_kernel(m.router.weight.type, s.row_gate.weight.type), (unsigned)(ne / 4 + 1), 1, Args(r));
        const TypeKernels& gk = *type_kernels(gt);
        const Projection gu[2] = {m.gate, m.up};
        // The shared expert's rows of h take the first block rows, so the routed entries start rows block rows on.
        RoutedCall gc = routed_call(gu, 2, nin, *gk.twin, ids, k, ne);
        gc.first = 0ull - rows;
        const Proj sp[2] = {{bytes(s.gate.weight.data), f32(s.gate.out), ff}, {bytes(s.up.weight.data), f32(s.up.out), ff}};
        const RowCall sc = row_call(sp, 2, 0, row_form(kQ8Twins, nin, true), nin);
        const StepUp u{gc, hq, hd, nin, act, aq, ad, sc, hq, hd, nin, f32(s.act), f32(s.row_gate.out), sq, sd, rows};
        launch_step(step_up_kernel(gk.twin->step_up, step_chains(*gk.twin, nin), ff / 32 * (entries + rows)), (unsigned)(entries + rows), (unsigned)(ff / 32), Args(u));
        keep_twin({act, entries * ff, true, am, act_slot});
        const TypeKernels& dk = *type_kernels(m.down.type);
        const Projection dn{m.down, Slice{moe_out_.get(), 0}, nin};
        const RoutedCall dc = routed_call(&dn, 1, ff, *dk.twin, ids, 1, ne);
        const Proj sdn{bytes(s.down.data), f32(y), nin};
        const RowCall dsc = row_call(&sdn, 1, 0, row_form(kQ8Twins, ff, true), ff);
        const float* y0 = m.zero ? (const float*)as_gpu<Api>(*m.zero).data() : f32(y);
        const StepDown d{dc, aq, ad, ff, f32(m.weights), f32(y), (unsigned)k, dsc, sq, sd, ff, y0, m.zero ? 0 : nin};
        launch_step(step_build(dk.twin->step_down, step_chains(*dk.twin, ff)), (unsigned)rows, (unsigned)(nin / 8), Args(d));
        record_matrix_path(MatrixPath::block_int8);
        const std::pair<const float*, size_t> written[] = {{f32(m.router.out), rows * ne}, {f32(s.row_gate.out), rows}, {(const float*)ids, entries}, {f32(m.weights), entries},
                                                           {f32(m.gate.out), entries * ff}, {f32(m.up.out), entries * ff}, {f32(s.gate.out), rows * ff}, {f32(s.up.out), rows * ff},
                                                           {f32(s.act), rows * ff}, {f32(y), rows * nin}};
        for (const auto& [p, n] : written) forget_written(p, n);
    }
    // The fused step kernels' own limits (kernels/f32.gpu mf_route, kernels/int16.gpu gate_up and down): experts, picks, the width of h and the routed width ff.
    static constexpr size_t kStepExperts = 512, kStepPicks = 10, kStepIn = 2560, kStepFf = 640;
    static constexpr size_t kRouteStepExperts = 512;   // kernels/f32.gpu, route_step's keys of eight a lane of 64 or sixteen a lane of 32
    static constexpr size_t kRouteNormIn = 16 * kThreads;   // kernels/f32.gpu, the row route_step's input norm holds in shared memory
    // Whether a step has the shape the fused kernels hold (docs/src/backends-gpu.md, moe_step), h and every output in device memory.
    // One token row's sqrt_softplus or sigmoid router on a device of 64 or 32 lanes runs as one launch, its rows as exact_row gives them, a wave each, and the last block out picking (kernels/f32.gpu, llmx_route_step).
    void moe_router(const MoeStep& m, Slice h, size_t nin, size_t rows, RowRuns runs) override {
        const size_t ne = m.router.rows, k = m.k;
        const uint32_t t = m.router.weight.type;
        const bool sig = m.rule.score == RouteRule::Score::sigmoid;
        if (rows != 1 || m.rule.bias_stride || (!sig && m.rule.score != RouteRule::Score::sqrt_softplus) || (sig && !(m.rule.eps >= 0.0f && std::isfinite(m.rule.eps))) || (dev_.caps.subgroup_size != 64 && dev_.caps.subgroup_size != 32) || !ne || ne > kRouteStepExperts || !k || k > ne || k > 256 ||
            (t != quant::GGML_TYPE_F32 && t != quant::GGML_TYPE_BF16) || !std::isfinite(m.rule.scale))
            return Backend::moe_router(m, h, nin, rows, runs);
        if (!sig) m.rule.factory_eps(Api::prefix);
        check_weight(m.router.weight, nin, "router matrix");
        span(*m.router.weight.data.buffer, m.router.weight.data.offset * sizeof(float), quant::row_bytes(t, nin, ne));
        floats(h, nin);
        floats(m.router.out, ne);
        for (const Slice& r : {m.ids, m.weights}) floats(r, k);
        if (m.rule.bias.buffer) floats(m.rule.bias, ne);
        float* sc = f32(m.router.out);
        unsigned* id = (unsigned*)f32(m.ids);
        float* w = f32(m.weights);
        const unsigned char* wt = bytes(m.router.weight.data);
        // The input's norm runs in the launch where its row fits the blocks' shared row and h is not its source: every block sums the row and writes its share of h and h's twin.
        const MoeInput* in = m.input;
        const bool held = in && nin <= kRouteNormIn && !overlaps(CSlice(h), nin, in->src, nin);
        if (in && !held) moe_input(m, h, nin, rows, runs);
        float* hd = f32(h);
        const float *x = held ? f32(in->src) : hd, *bias = m.rule.bias.buffer ? f32(m.rule.bias) : nullptr;
        if (held) floats(in->src, nin), floats(in->norm, nin);
        const TwinOut tw = held && nin % 32 == 0 ? twin_out(h, nin, twin_map(runs, 1, nin)) : TwinOut{};
        const float* nw = held ? f32(in->norm) : nullptr;
        float neps = held ? in->eps : 0.0f;
        void* tq = tw.q;
        float* td = tw.d;
        int level = !tw.q ? 0 : tw.int8 ? 2 : 1;
        unsigned long long n = nin;
        unsigned o = (unsigned)ne, kk = (unsigned)k;
        int norm = m.rule.normalize ? 1 : 0;
        float scale = m.rule.scale, eps = m.rule.eps;
        const int score = sig;
        const Args args(wt, x, sc, n, o, bias, id, w, kk, norm, scale, score, eps, step_counter(), nw, neps, hd, tq, td, level);
        const size_t blocks = (ne + rows_per_block_ - 1) / rows_per_block_;
        launch_writing({{sc, ne}, {id, k}, {w, k}, {hd, held ? nin : 0}}, t == quant::GGML_TYPE_F32 ? kRouteStepF32 : kRouteStepBF16, u32(blocks), 1, args);
        if (held) keep(tw, hd, nin);
    }

    bool fused_step(const MoeStep& m, CSlice h, size_t nin, size_t rows, RowRuns runs, Dtype dtype) const {
        if (!rows) return false;
        for (size_t i = 0; rows > 1 && i < runs.n; ++i)
            if (runs.runs[i].extent != 1) return false;
        if (dtype != Dtype::int8 || dev_.caps.subgroup_size != 64 || !m.shared || !m.shared->row_gate.weight.data.buffer) return false;
        const MoeShared& s = *m.shared;
        const size_t ff = m.gate.rows;
        const uint32_t gt = m.gate.weight.type, dt = m.down.type;
        if (m.rule.score != RouteRule::Score::softmax || m.rule.bias.buffer || m.rule.scale != 1.0f || m.glu.limit != 0.0f || m.glu.form != GluForm::silu || s.glu.form != GluForm::silu) return false;
        const size_t ne = m.router.rows;
        if (!ne || ne % 64 || ne > kStepExperts || !m.k || m.k > kStepPicks || !nin || nin % 256 || nin > kStepIn) return false;
        // An even block count, since the shared Q8_0 down takes the half-block order at any width where the base's quarter layout holds an odd one.
        if (!ff || ff % 64 || ff > kStepFf || m.up.rows != ff || s.gate.rows != ff || s.up.rows != ff) return false;
        // The router and the row gate in a type with a route decoder (TypeKernels::step_route), each row as the one-row product gives it.
        // Several rows of a type with a half tile take it in the base (product), whose bits exact_row does not give.
        for (const uint32_t t : {m.router.weight.type, s.row_gate.weight.type}) {
            const TypeKernels* tk = type_kernels(t);
            if (!tk || tk->step_route < 0 || (rows > 1 && tk->half != kKernels)) return false;
        }
        if (s.row_gate.rows != 1) return false;
        if (m.up.weight.type != gt) return false;
        if (s.gate.weight.type != quant::GGML_TYPE_Q8_0 || s.up.weight.type != quant::GGML_TYPE_Q8_0 || s.down.type != quant::GGML_TYPE_Q8_0) return false;
        const TypeKernels* gk = type_kernels(gt);
        const TypeKernels* dk = type_kernels(dt);
        if (!gk || !gk->twin || !dk || !dk->twin || &row_form(*gk->twin, nin, true) != gk->twin || &row_form(*dk->twin, ff, true) != dk->twin) return false;
        // The type's fused builds (TwinKernels step_up, step_down), whose rows take whole blocks on the routed product's cluster, at most the chains a lane of 16 holds (kernels/int16.gpu, lane_row).
        if (gk->twin->step_up == kKernels || !kernels_[step_build(gk->twin->step_up, step_chains(*gk->twin, nin))] || (gk->twin->row_lanes && !step_lanes(*gk->twin, nin, kStepIn))) return false;
        if (dk->twin->step_down == kKernels || !kernels_[step_build(dk->twin->step_down, step_chains(*dk->twin, ff))] || (dk->twin->row_lanes && !step_lanes(*dk->twin, ff, kStepFf))) return false;
        // Where the base's routed products group the entries and the entries reach the experts, each expert's gate and up rows are read once for several entries, which outweighs the fused step's fewer launches.
        if (const size_t entries = size_mul(rows, m.k); gk->twin->groups(entries, ne, true) && entries >= ne) return false;
        if (as_gpu<Api>(*h.buffer).host_ptr()) return false;
        for (const Buffer* b : {m.router.out.buffer, m.gate.out.buffer, m.up.out.buffer, m.act.buffer, m.ids.buffer, m.weights.buffer, s.gate.out.buffer, s.up.out.buffer, s.act.buffer,
                                s.row_gate.out.buffer})
            if (as_gpu<Api>(*b).host_ptr()) return false;
        return true;
    }

    // The kernels launched since the last reading, by name, and their counts.
    std::map<std::string, size_t> take_launches() {
        std::map<std::string, size_t> m;
        for (size_t k = 0; k < kKernels; ++k)
            if (launched_[k]) m[kKernelNames[k]] = launched_[k];
        std::fill(std::begin(launched_), std::end(launched_), size_t(0));
        return m;
    }
    void route_only(std::string build) { route_only_ = std::move(build); }
    // A timing backend sleeps this long between a timed op's start event and its launch: the tests' host that falls behind its device.
    void delay_launches(std::chrono::microseconds delay) { delay_ = delay; }

private:
    // The F32 module's kernels, then from kRows16x1 the int16 module's, which only read twins.
    enum Kernel { kAdd, kCopy, kSiluMul, kSiluMulTwin16, kSiluMulTwin8, kRmsNorm, kRmsNorm16, kRmsNorm16Twin16, kRmsNorm16Twin8, kRmsNorm32, kRmsNorm32Twin16,
                  kRmsNorm32Twin8, kNormRope, kNormRopeKv, kGather, kGatherQ8, kMatmul, kMatmulQ8, kMatmulQ4K, kGatherQ4K, kMatmulQ5K, kGatherQ5K, kMatmulQ6K,
                  kGatherQ6K, kMatmulQ50, kGatherQ50, kMatmulQ40, kGatherQ40, kMatmulQ41, kGatherQ41, kMatmulQ51, kGatherQ51, kMatmulBF16, kGatherBF16, kMatmulF16, kGatherF16, kMatmulIQ4NL, kGatherIQ4NL, kMatmulMXFP4, kGatherMXFP4, kMatmulNVFP4, kGatherNVFP4, kMatmulQ2K, kGatherQ2K, kMatmulQ3K, kGatherQ3K, kMatmulIQ4XS, kGatherIQ4XS, kMatmulIQ2XXS, kGatherIQ2XXS, kMatmulIQ2XS, kGatherIQ2XS, kMatmulIQ2S, kGatherIQ2S, kMatmulIQ3XXS, kGatherIQ3XXS, kMatmulIQ3S, kGatherIQ3S, kMatmulIQ1S, kGatherIQ1S, kMatmulIQ1M, kGatherIQ1M, kMatmulTQ10, kGatherTQ10, kMatmulTQ20, kGatherTQ20, kMatmulQ10, kGatherQ10, kMatmulQ20, kGatherQ20, kMatmulF8, kGatherF8, kKvWrite, kAttentionChunk, kAttentionTile, kAttentionTile256, kAttentionMerge, kAttentionVec,
                  kAttentionVecMerge, kTwin16, kTwin8, kSiluMulTwin16Major, kSiluMulTwin8Major, kRmsNorm16Twin16Major, kRmsNorm16Twin8Major,
                  kRmsNorm32Twin16Major, kRmsNorm32Twin8Major, kAttentionVecMajor, kAttentionVecMergeMajor, kTwin16Major, kTwin8Major, kGroupSum, kGroupChain, kGroupShares, kGroupCopies, kGroupArm, kGroupAddCheck, kGroupPush, kGroupTake, kSigmoidMul, kGatedRmsNorm, kNormRopeStreams, kCausalConvSilu, kGatedDeltaRule, kGatedDeltaRuleShort, kGroupRmsNorm, kSiluScaled, kMixStreams, kInjectStreams, kSignedSqrtGate, kDilatedConvSiluAdd, kKvIndexPool, kKvIndexScores, kKvIndexSelect, kKvIndexSelectWide, kKvIndexAll, kSparseChunk, kSparseMerge, kSparseVec, kSparseVec256, kSparseVecMerge, kSparseVecMerge256,
                  kAttentionVec256, kAttentionVec256Major, kAttentionVecMerge256, kAttentionVecMerge256Major, kAttentionGated256, kAttentionGatedMerge256, kArgmaxSlices, kArgmaxMerge, kArgmaxPart, kArgmaxPick,
                  kMoeRoute, kMoeCombine, kMoeGroup, kRoutedExactF32, kRoutedExactQ8, kRoutedExactQ4K, kRoutedExactQ5K, kRoutedExactQ6K, kRoutedExactQ50, kRoutedExactQ40, kRoutedExactQ41, kRoutedExactQ51, kRoutedExactBF16, kRoutedExactF16, kRoutedExactIQ4NL, kRoutedExactMXFP4, kRoutedExactNVFP4, kRoutedExactQ2K, kRoutedExactQ3K, kRoutedExactIQ4XS, kRoutedExactIQ2XXS, kRoutedExactIQ2XS, kRoutedExactIQ2S, kRoutedExactIQ3XXS, kRoutedExactIQ3S, kRoutedExactIQ1S, kRoutedExactIQ1M, kRoutedExactTQ10, kRoutedExactTQ20, kRoutedExactQ10, kRoutedExactQ20, kRoutedExactF8,
                  kRowsExactF32, kRowsExactQ8, kRowsExactQ4K, kRowsExactQ5K, kRowsExactQ6K, kRowsExactQ50, kRowsExactQ40, kRowsExactQ41, kRowsExactQ51, kRowsExactBF16, kRowsExactF16, kRowsExactIQ4NL, kRowsExactMXFP4, kRowsExactNVFP4, kRowsExactQ2K, kRowsExactQ3K, kRowsExactIQ4XS, kRowsExactIQ2XXS, kRowsExactIQ2XS, kRowsExactIQ2S, kRowsExactIQ3XXS, kRowsExactIQ3S, kRowsExactIQ1S, kRowsExactIQ1M, kRowsExactTQ10, kRowsExactTQ20, kRowsExactQ10, kRowsExactQ20, kRowsExactF8, kSparseVecG4, kSparseVec256G4, kColsF32, kColsF32Many, kColsBF16, kColsBF16Many, kColsF16, kColsF16Many, kMoeStepRoute, kMoeStepRouteF32F16, kMoeStepRouteF32BF16, kMoeStepRouteF16F32, kMoeStepRouteF16F16, kMoeStepRouteF16BF16, kMoeStepRouteBF16F32, kMoeStepRouteBF16F16, kMoeStepRouteBF16BF16,
                  kHcCoeffsCollapse, kHcExpand, kModelFormatQdq, kTopkRows, kCompressPool, kIndexerScores, kIndexerScoresTile, kIndexerScoresInt, kRawIndexScores, kPoolWrite, kSparseScoresFew16, kSparseValuesFew16, kSparseScoresFew8, kSparseValuesFew8, kSparseScoresFew4, kSparseValuesFew4, kSparseScoresFew2, kSparseValuesFew2, kSparseScoresFew32, kSparseValuesFew16x2, kSparseScoresTile, kSparseValuesTile, kSparseScoresTile32, kSparseValuesTile32, kSparseScoresFew16Terms, kSparseValuesFew16Terms, kSparseScoresFew8Terms, kSparseValuesFew8Terms, kSparseScoresFew4Terms, kSparseValuesFew4Terms, kSparseScoresFew2Terms, kSparseValuesFew2Terms, kSparseScoresFew32Terms, kSparseValuesFew16x2Terms, kSparseScoresTileTerms, kSparseValuesTileTerms, kSparseScoresTile32Terms, kSparseValuesTile32Terms, kRouteBiased, kRouteStepF32, kRouteStepBF16,

                  kNormRopeForm, kDeltaStep, kMixRows, kLayerNormRows, kGelu, kAddBias, kCopyRows, kSoftcap, kL2Clip, kL2ClipLanes, kQueryMix, kQueryMix4, kQueryMixRows1, kQueryMixRows2, kQueryMixRows3, kQueryMixRows4, kQueryMixRows5, kQueryMixNorm, kTanhGateMix, kTanhNorm16, kTanhNorm16Twin16, kTanhNorm16Twin8, kNormedGateMix, kNormedGateMixTwin16, kNormedGateMixTwin8, kMatmulGroupF32, kMatmulNormedF32, kConvRing, kConvRingParts, kConvRingSum, kConvRingMix4, kAttentionRelSink, kAttentionRelLanesG3, kAttentionRelLanesG4, kRelSinkStore, kAttentionRelTile, kAttentionRelTileG3, kAttentionTermsG1, kAttentionTermsG3, kAttentionTermsG3x512, kAttentionChunkWindow, kAttentionVecWindow, kAttentionVec256Window, kUnitRows, kSigmoidGateRows, kMatmulAtIds, kChainPick, kTapWeightedRows, kExpertMark, kExpertWave, kExpertPull, kExpertRound,
                  kRows16x1, kRows16x2, kRows16x2r4, kRows16x4, kRows16x4r4, kRows16x8, kRows16x8r4, kRows16x16, kTile16,
                  kRows8x1, kRows8x2, kRows8x2r4, kRows8x4, kRows8x4r4, kRows8x8, kRows8x8r4, kRows8x16, kTile8, kTileReduce,
kHalf8x1, kHalf8x1p, kHalf8x1r2, kHalf8x1s1, kHalf8x1r2s1, kHalf8x1r2s4, kHalf8x1h2, kHalf8x1h3, kHalf8x1h4, kHalf8x1h5, kHalf8x1h2p, kHalf8x1h3p, kHalf8x1h4p, kHalf8x1h5p, kHalf8x1h2Pair, kHalf8x1h3Pair, kHalf8x1h4Pair, kHalf8x1h5Pair, kHalf8x1h2List, kHalf8x1h3List, kHalf8x1h4List, kHalf8x1h5List, kHalf8x2, kHalf8x2r4, kHalf8x4, kHalf8x4r4, kHalf8x8, kHalf8x8r4, kHalf8x16, kHalf8x16r4, kHalf8x11, kHalf8x12, kHalf8x13, kHalf8x14, kHalf8x15, kTile16Tall, kTile8Tall, kRouted16, kRouted8, kRoutedHalf8, kRoutedHalf8s1, kGrouped16x2, kGrouped16x4, kGrouped16x8, kGrouped8x2, kGrouped8x4, kGrouped8x8, kGroupedHalf8x2, kGroupedHalf8x4, kGroupedHalf8x8, kRoutedTile16, kRoutedTile8, kRoutedThin16, kRoutedMid16, kRoutedStream16, kRoutedThinMXFP416,
                  kRowsQ4K16x1, kRowsQ4K16x2, kRowsQ4K16x4, kRowsQ4K16x8, kRowsQ4K16x16, kRowsQ4K8x1, kRowsQ4K8x2, kRowsQ4K8x4, kRowsQ4K8x8, kRowsQ4K8x16, kTileQ4K16, kTileQ4K8, kRoutedQ4K16, kRoutedQ4K8, kGroupedQ4K16x2, kGroupedQ4K16x4, kGroupedQ4K16x8, kGroupedQ4K8x2, kGroupedQ4K8x4, kGroupedQ4K8x8, kRoutedTileQ4K16, kRoutedTileQ4K8, kRoutedNarrowQ4K16, kRoutedStream16Q4K8, kRoutedStreamQ4K8, kRoutedMidQ4K16, kRoutedMidQ4K8, kRowsQ5K16x1, kRowsQ5K16x2, kRowsQ5K16x4, kRowsQ5K16x8, kRowsQ5K16x16, kRowsQ5K8x1, kRowsQ5K8x2, kRowsQ5K8x4, kRowsQ5K8x8, kRowsQ5K8x16, kTileQ5K16, kTileQ5K8, kRoutedQ5K16, kRoutedQ5K8, kGroupedQ5K16x2, kGroupedQ5K16x4, kGroupedQ5K16x8, kGroupedQ5K8x2, kGroupedQ5K8x4, kGroupedQ5K8x8, kRoutedTileQ5K16, kRoutedTileQ5K8, kRoutedNarrowQ5K16, kRoutedStream16Q5K8, kRoutedStreamQ5K8, kRoutedMidQ5K16, kRoutedMidQ5K8, kRowsQ6K16x1, kRowsQ6K16x2, kRowsQ6K16x4, kRowsQ6K16x8, kRowsQ6K16x16, kRowsQ6K8x1, kRowsQ6K8x2, kRowsQ6K8x4, kRowsQ6K8x8, kRowsQ6K8x16, kTileQ6K16, kTileQ6K8, kRoutedQ6K16, kRoutedQ6K8, kGroupedQ6K16x2, kGroupedQ6K16x4, kGroupedQ6K16x8, kGroupedQ6K8x2, kGroupedQ6K8x4, kGroupedQ6K8x8, kRoutedTileQ6K16, kRoutedTileQ6K8, kRoutedNarrowQ6K16, kRoutedStream16Q6K8, kRoutedStreamQ6K8, kRoutedMidQ6K16, kRoutedMidQ6K8,
                  kRowsQ5016x1, kRowsQ5016x2, kRowsQ5016x4, kRowsQ5016x8, kRowsQ5016x16, kRowsQ508x1, kRowsQ508x2, kRowsQ508x4, kRowsQ508x8, kRowsQ508x16, kTileQ5016, kTileQ508, kRoutedQ5016, kRoutedQ508, kGroupedQ5016x2, kGroupedQ5016x4, kGroupedQ5016x8, kGroupedQ508x2, kGroupedQ508x4, kGroupedQ508x8, kRoutedTileQ5016, kRoutedTileQ508, kRoutedNarrowQ5016, kRoutedStream16Q508, kRoutedStreamQ508, kRoutedMidQ5016, kRoutedMidQ508, kRowsQ4016x1, kRowsQ4016x2, kRowsQ4016x4, kRowsQ4016x8, kRowsQ4016x16, kRowsQ408x1, kRowsQ408x2, kRowsQ408x4, kRowsQ408x8, kRowsQ408x16, kTileQ4016, kTileQ408, kRoutedQ4016, kRoutedQ408, kGroupedQ4016x2, kGroupedQ4016x4, kGroupedQ4016x8, kGroupedQ408x2, kGroupedQ408x4, kGroupedQ408x8, kRoutedTileQ4016, kRoutedTileQ408, kRoutedNarrowQ4016, kRoutedStream16Q408, kRoutedStreamQ408, kRoutedMidQ4016, kRoutedMidQ408, kRowsQ4116x1, kRowsQ4116x2, kRowsQ4116x4, kRowsQ4116x8, kRowsQ4116x16, kRowsQ418x1, kRowsQ418x2, kRowsQ418x4, kRowsQ418x8, kRowsQ418x16, kTileQ4116, kTileQ418, kRoutedQ4116, kRoutedQ418, kGroupedQ4116x2, kGroupedQ4116x4, kGroupedQ4116x8, kGroupedQ418x2, kGroupedQ418x4, kGroupedQ418x8, kRoutedTileQ4116, kRoutedTileQ418, kRoutedNarrowQ4116, kRoutedStream16Q418, kRoutedStreamQ418, kRoutedMidQ4116, kRoutedMidQ418, kRowsQ5116x1, kRowsQ5116x2, kRowsQ5116x4, kRowsQ5116x8, kRowsQ5116x16, kRowsQ518x1, kRowsQ518x2, kRowsQ518x4, kRowsQ518x8, kRowsQ518x16, kTileQ5116, kTileQ518, kRoutedQ5116, kRoutedQ518, kGroupedQ5116x2, kGroupedQ5116x4, kGroupedQ5116x8, kGroupedQ518x2, kGroupedQ518x4, kGroupedQ518x8, kRoutedTileQ5116, kRoutedTileQ518, kRoutedNarrowQ5116, kRoutedStream16Q518, kRoutedStreamQ518, kRoutedMidQ5116, kRoutedMidQ518, kRowsIQ4NL16x1, kRowsIQ4NL16x2, kRowsIQ4NL16x4, kRowsIQ4NL16x8, kRowsIQ4NL16x16, kRowsIQ4NL8x1, kRowsIQ4NL8x2, kRowsIQ4NL8x4, kRowsIQ4NL8x8, kRowsIQ4NL8x16, kTileIQ4NL16, kTileIQ4NL8, kRoutedIQ4NL16, kRoutedIQ4NL8, kGroupedIQ4NL16x2, kGroupedIQ4NL16x4, kGroupedIQ4NL16x8, kGroupedIQ4NL8x2, kGroupedIQ4NL8x4, kGroupedIQ4NL8x8, kRoutedTileIQ4NL16, kRoutedTileIQ4NL8, kRoutedNarrowIQ4NL16, kRoutedStream16IQ4NL8, kRoutedStreamIQ4NL8, kRoutedMidIQ4NL16, kRoutedMidIQ4NL8, kRowsMXFP416x1, kRowsMXFP416x2, kRowsMXFP416x4, kRowsMXFP416x8, kRowsMXFP416x16, kRowsMXFP48x1, kRowsMXFP48x2, kRowsMXFP48x4, kRowsMXFP48x8, kRowsMXFP48x16, kTileMXFP416, kTileMXFP48, kRoutedMXFP416, kRoutedMXFP48, kGroupedMXFP416x2, kGroupedMXFP416x4, kGroupedMXFP416x8, kGroupedMXFP48x2, kGroupedMXFP48x4, kGroupedMXFP48x8, kRoutedTileMXFP416, kRoutedTileMXFP48, kRoutedNarrowMXFP416, kRoutedStream16MXFP48, kRoutedStreamMXFP48, kRoutedMidMXFP416, kRoutedStream32MXFP48, kQuadRowsMXFP48x1, kQuadRoutedMXFP48,
                  kRowsQ3K16x1, kRowsQ3K16x2, kRowsQ3K16x4, kRowsQ3K16x8, kRowsQ3K16x16, kRowsQ3K8x1, kRowsQ3K8x2, kRowsQ3K8x4, kRowsQ3K8x8, kRowsQ3K8x16, kTileQ3K16, kTileQ3K8, kRoutedQ3K16, kRoutedQ3K8, kGroupedQ3K16x2, kGroupedQ3K16x4, kGroupedQ3K16x8, kGroupedQ3K8x2, kGroupedQ3K8x4, kGroupedQ3K8x8, kRoutedTileQ3K16, kRoutedTileQ3K8, kRoutedNarrowQ3K16, kRoutedStream16Q3K8, kRoutedStreamQ3K8, kRoutedMidQ3K16, kRoutedMidQ3K8,
                  kRowsQ2K16x1, kRowsQ2K16x2, kRowsQ2K16x4, kRowsQ2K16x8, kRowsQ2K16x16, kRowsQ2K8x1, kRowsQ2K8x2, kRowsQ2K8x4, kRowsQ2K8x8, kRowsQ2K8x16, kTileQ2K16, kTileQ2K8, kRoutedQ2K16, kRoutedQ2K8, kGroupedQ2K16x2, kGroupedQ2K16x4, kGroupedQ2K16x8, kGroupedQ2K8x2, kGroupedQ2K8x4, kGroupedQ2K8x8, kRoutedTileQ2K16, kRoutedTileQ2K8, kRoutedNarrowQ2K16, kRoutedStream16Q2K8, kRoutedStreamQ2K8, kRoutedMidQ2K16, kRoutedMidQ2K8,
                  kRowsIQ4XS16x1, kRowsIQ4XS16x2, kRowsIQ4XS16x4, kRowsIQ4XS16x8, kRowsIQ4XS16x16, kRowsIQ4XS8x1, kRowsIQ4XS8x2, kRowsIQ4XS8x4, kRowsIQ4XS8x8, kRowsIQ4XS8x16, kTileIQ4XS16, kTileIQ4XS8, kRoutedIQ4XS16, kRoutedIQ4XS8, kGroupedIQ4XS16x2, kGroupedIQ4XS16x4, kGroupedIQ4XS16x8, kGroupedIQ4XS8x2, kGroupedIQ4XS8x4, kGroupedIQ4XS8x8, kRoutedTileIQ4XS16, kRoutedTileIQ4XS8, kRoutedNarrowIQ4XS16, kRoutedStream16IQ4XS8, kRoutedStreamIQ4XS8, kRoutedMidIQ4XS16, kRoutedMidIQ4XS8,
                  kRowsIQ2XXS16x1, kRowsIQ2XXS16x2, kRowsIQ2XXS16x4, kRowsIQ2XXS16x8, kRowsIQ2XXS16x16, kRowsIQ2XXS8x1, kRowsIQ2XXS8x2, kRowsIQ2XXS8x4, kRowsIQ2XXS8x8, kRowsIQ2XXS8x16, kTileIQ2XXS16, kTileIQ2XXS8, kRoutedIQ2XXS16, kRoutedIQ2XXS8, kGroupedIQ2XXS16x2, kGroupedIQ2XXS16x4, kGroupedIQ2XXS16x8, kGroupedIQ2XXS8x2, kGroupedIQ2XXS8x4, kGroupedIQ2XXS8x8, kRoutedTileIQ2XXS16, kRoutedTileIQ2XXS8, kRoutedNarrowIQ2XXS16, kRoutedStream16IQ2XXS8, kRoutedStreamIQ2XXS8, kRoutedMidIQ2XXS16, kRoutedMidIQ2XXS8,
                  kRowsIQ2XS16x1, kRowsIQ2XS16x2, kRowsIQ2XS16x4, kRowsIQ2XS16x8, kRowsIQ2XS16x16, kRowsIQ2XS8x1, kRowsIQ2XS8x2, kRowsIQ2XS8x4, kRowsIQ2XS8x8, kRowsIQ2XS8x16, kTileIQ2XS16, kTileIQ2XS8, kRoutedIQ2XS16, kRoutedIQ2XS8, kGroupedIQ2XS16x2, kGroupedIQ2XS16x4, kGroupedIQ2XS16x8, kGroupedIQ2XS8x2, kGroupedIQ2XS8x4, kGroupedIQ2XS8x8, kRoutedTileIQ2XS16, kRoutedTileIQ2XS8, kRoutedNarrowIQ2XS16, kRoutedStream16IQ2XS8, kRoutedStreamIQ2XS8, kRoutedMidIQ2XS16, kRoutedMidIQ2XS8,
                  kRowsIQ3S16x1, kRowsIQ3S16x2, kRowsIQ3S16x4, kRowsIQ3S16x8, kRowsIQ3S16x16, kRowsIQ3S8x1, kRowsIQ3S8x2, kRowsIQ3S8x4, kRowsIQ3S8x8, kRowsIQ3S8x16, kTileIQ3S16, kTileIQ3S8, kRoutedIQ3S16, kRoutedIQ3S8, kGroupedIQ3S16x2, kGroupedIQ3S16x4, kGroupedIQ3S16x8, kGroupedIQ3S8x2, kGroupedIQ3S8x4, kGroupedIQ3S8x8, kRoutedTileIQ3S16, kRoutedTileIQ3S8, kRoutedNarrowIQ3S16, kRoutedStream16IQ3S8, kRoutedStreamIQ3S8, kRoutedMidIQ3S16, kRoutedMidIQ3S8,
                  kRowsIQ2S16x1, kRowsIQ2S16x2, kRowsIQ2S16x4, kRowsIQ2S16x8, kRowsIQ2S16x16, kRowsIQ2S8x1, kRowsIQ2S8x2, kRowsIQ2S8x4, kRowsIQ2S8x8, kRowsIQ2S8x16, kTileIQ2S16, kTileIQ2S8, kRoutedIQ2S16, kRoutedIQ2S8, kGroupedIQ2S16x2, kGroupedIQ2S16x4, kGroupedIQ2S16x8, kGroupedIQ2S8x2, kGroupedIQ2S8x4, kGroupedIQ2S8x8, kRoutedTileIQ2S16, kRoutedTileIQ2S8, kRoutedNarrowIQ2S16, kRoutedStream16IQ2S8, kRoutedStreamIQ2S8, kRoutedMidIQ2S16, kRoutedMidIQ2S8,
                  kRowsIQ3XXS16x1, kRowsIQ3XXS16x2, kRowsIQ3XXS16x4, kRowsIQ3XXS16x8, kRowsIQ3XXS16x16, kRowsIQ3XXS8x1, kRowsIQ3XXS8x2, kRowsIQ3XXS8x4, kRowsIQ3XXS8x8, kRowsIQ3XXS8x16, kTileIQ3XXS16, kTileIQ3XXS8, kRoutedIQ3XXS16, kRoutedIQ3XXS8, kGroupedIQ3XXS16x2, kGroupedIQ3XXS16x4, kGroupedIQ3XXS16x8, kGroupedIQ3XXS8x2, kGroupedIQ3XXS8x4, kGroupedIQ3XXS8x8, kRoutedTileIQ3XXS16, kRoutedTileIQ3XXS8, kRoutedNarrowIQ3XXS16, kRoutedStream16IQ3XXS8, kRoutedStreamIQ3XXS8, kRoutedMidIQ3XXS16, kRoutedMidIQ3XXS8,
                  kRowsIQ1S16x1, kRowsIQ1S16x2, kRowsIQ1S16x4, kRowsIQ1S16x8, kRowsIQ1S16x16, kRowsIQ1S8x1, kRowsIQ1S8x2, kRowsIQ1S8x4, kRowsIQ1S8x8, kRowsIQ1S8x16, kTileIQ1S16, kTileIQ1S8, kRoutedIQ1S16, kRoutedIQ1S8, kGroupedIQ1S16x2, kGroupedIQ1S16x4, kGroupedIQ1S16x8, kGroupedIQ1S8x2, kGroupedIQ1S8x4, kGroupedIQ1S8x8, kRoutedTileIQ1S16, kRoutedTileIQ1S8, kRoutedNarrowIQ1S16, kRoutedStream16IQ1S8, kRoutedStreamIQ1S8, kRoutedMidIQ1S16, kRoutedMidIQ1S8,
                  kRowsIQ1M16x1, kRowsIQ1M16x2, kRowsIQ1M16x4, kRowsIQ1M16x8, kRowsIQ1M16x16, kRowsIQ1M8x1, kRowsIQ1M8x2, kRowsIQ1M8x4, kRowsIQ1M8x8, kRowsIQ1M8x16, kTileIQ1M16, kTileIQ1M8, kRoutedIQ1M16, kRoutedIQ1M8, kGroupedIQ1M16x2, kGroupedIQ1M16x4, kGroupedIQ1M16x8, kGroupedIQ1M8x2, kGroupedIQ1M8x4, kGroupedIQ1M8x8, kRoutedTileIQ1M16, kRoutedTileIQ1M8, kRoutedNarrowIQ1M16, kRoutedStream16IQ1M8, kRoutedStreamIQ1M8, kRoutedMidIQ1M16, kRoutedMidIQ1M8,
                  kRowsQ1016x1, kRowsQ1016x2, kRowsQ1016x4, kRowsQ1016x8, kRowsQ1016x16, kRowsQ108x1, kRowsQ108x2, kRowsQ108x4, kRowsQ108x8, kRowsQ108x16, kTileQ1016, kTileQ108, kRoutedQ1016, kRoutedQ108, kGroupedQ1016x2, kGroupedQ1016x4, kGroupedQ1016x8, kGroupedQ108x2, kGroupedQ108x4, kGroupedQ108x8, kRoutedTileQ1016, kRoutedTileQ108, kRoutedNarrowQ1016, kRoutedStream16Q108, kRoutedStreamQ108, kRoutedMidQ1016, kRoutedMidQ108,
                  kRowsQ2016x1, kRowsQ2016x2, kRowsQ2016x4, kRowsQ2016x8, kRowsQ2016x16, kRowsQ208x1, kRowsQ208x2, kRowsQ208x4, kRowsQ208x8, kRowsQ208x16, kTileQ2016, kTileQ208, kRoutedQ2016, kRoutedQ208, kGroupedQ2016x2, kGroupedQ2016x4, kGroupedQ2016x8, kGroupedQ208x2, kGroupedQ208x4, kGroupedQ208x8, kRoutedTileQ2016, kRoutedTileQ208, kRoutedNarrowQ2016, kRoutedStream16Q208, kRoutedStreamQ208, kRoutedMidQ2016, kRoutedMidQ208,
                  kRowsNVFP416x1, kRowsNVFP416x2, kRowsNVFP416x4, kRowsNVFP416x8, kRowsNVFP416x16, kRowsNVFP48x1, kRowsNVFP48x2, kRowsNVFP48x4, kRowsNVFP48x8, kRowsNVFP48x16, kTileNVFP416, kTileNVFP48, kRoutedNVFP416, kRoutedNVFP48, kGroupedNVFP416x2, kGroupedNVFP416x4, kGroupedNVFP416x8, kGroupedNVFP48x2, kGroupedNVFP48x4, kGroupedNVFP48x8, kRoutedTileNVFP416, kRoutedTileNVFP48, kRoutedNarrowNVFP416, kRoutedStream16NVFP48, kRoutedStreamNVFP48, kRoutedMidNVFP416, kRoutedMidNVFP48,
                  kQuarterQ5016x1, kQuarterQ5016x2, kQuarterQ5016x4, kQuarterQ5016x8, kQuarterQ5016x16, kQuarterRoutedQ5016, kQuarterGroupedQ5016x2, kQuarterGroupedQ5016x4, kQuarterGroupedQ5016x8, kQuarterQ4016x1, kQuarterQ4016x2, kQuarterQ4016x4, kQuarterQ4016x8, kQuarterQ4016x16, kQuarterRoutedQ4016, kQuarterGroupedQ4016x2, kQuarterGroupedQ4016x4, kQuarterGroupedQ4016x8, kQuarterQ4116x1, kQuarterQ4116x2, kQuarterQ4116x4, kQuarterQ4116x8, kQuarterQ4116x16, kQuarterRoutedQ4116, kQuarterGroupedQ4116x2, kQuarterGroupedQ4116x4, kQuarterGroupedQ4116x8, kQuarterQ5116x1, kQuarterQ5116x2, kQuarterQ5116x4, kQuarterQ5116x8, kQuarterQ5116x16, kQuarterRoutedQ5116, kQuarterGroupedQ5116x2, kQuarterGroupedQ5116x4, kQuarterGroupedQ5116x8, kHcMix, kHcMix8, kHcEntry, kHcEntry8, kMoeStepUpQ4K, kMoeStepUpQ4KW, kMoeStepUpQ5K, kMoeStepUpQ5KW, kMoeStepUpQ6K, kMoeStepUpQ6KW, kMoeStepUpQ40, kMoeStepUpQ40W, kMoeStepUpQ41, kMoeStepUpQ41W, kMoeStepUpQ50, kMoeStepUpQ50W, kMoeStepUpQ51, kMoeStepUpQ51W, kMoeStepUpQ8, kMoeStepUpQ8W, kMoeStepDownQ8, kMoeStepDownQ4K, kMoeStepDownQ5K, kMoeStepDownQ6K, kMoeStepDownQ40, kMoeStepDownQ41, kMoeStepDownQ50, kMoeStepDownQ51, kMoeStepUpQ4KC2, kMoeStepUpQ4KWC2, kMoeStepUpQ5KC2, kMoeStepUpQ5KWC2, kMoeStepUpQ6KC2, kMoeStepUpQ6KWC2, kMoeStepUpQ40C2, kMoeStepUpQ40WC2, kMoeStepUpQ41C2, kMoeStepUpQ41WC2, kMoeStepUpQ50C2, kMoeStepUpQ50WC2, kMoeStepUpQ51C2, kMoeStepUpQ51WC2, kMoeStepUpQ4KC4, kMoeStepUpQ4KWC4, kMoeStepUpQ5KC4, kMoeStepUpQ5KWC4, kMoeStepUpQ6KC4, kMoeStepUpQ6KWC4, kMoeStepUpQ40C4, kMoeStepUpQ40WC4, kMoeStepUpQ41C4, kMoeStepUpQ41WC4, kMoeStepUpQ50C4, kMoeStepUpQ50WC4, kMoeStepUpQ51C4, kMoeStepUpQ51WC4, kMoeStepDownQ4KC2, kMoeStepDownQ5KC2, kMoeStepDownQ6KC2, kMoeStepDownQ40C2, kMoeStepDownQ41C2, kMoeStepDownQ50C2, kMoeStepDownQ51C2, kMoeStepDownQ4KC4, kMoeStepDownQ5KC4, kMoeStepDownQ6KC4, kMoeStepDownQ40C4, kMoeStepDownQ41C4, kMoeStepDownQ50C4, kMoeStepDownQ51C4, kF16Tile, kF8Tile, kF32Tile, kTallQ4K8, kTallQ5K8, kTallQ6K8, kSquareQ4K8, kSquareQ5K8, kSquareQ6K8, kRows16x1Walk3, kRows16x1Walk5, kRouted16Walk3, kRouted16Walk5, kRouted16Walk3Add, kRouted16Walk5Add, kRows16x1Pair, kRows16x1Walk3Pair, kRows16x1Walk5Pair, kRows16x1Conv, kRows8x1Conv, kRouted16Walk3AddConv, kRouted16Walk5AddConv, kRows16x1Walk2, kRouted16Walk2, kRouted16Walk2Add, kRows16x1Walk2Pair, kRouted16Walk2AddConv, kRouted16Walk3Glu, kRouted16Walk5Glu, kF32TileMix, kBF16Tile, kBF16TileBF16, kRoutedHalf8WalkAdd, kRoutedHalf8WalkAddConv, kRoutedHalf8Walk3Glu, kKernels };
    static constexpr size_t kHcColumns = 8;   // llmx_hc_mix8's columns, hc_mix<8> in kernels/int16.gpu, which backend-rocm twins holds at 2 to 16 rows
    // The build for n columns: one column's own, else the one of up to kHcColumns.
    static Kernel hc_build(size_t n) { return n == 1 ? kHcMix : kHcMix8; }
    static constexpr const char* kKernelNames[kKernels] = {"llmx_add", "llmx_copy", "llmx_silu_mul", "llmx_silu_mul_twin16", "llmx_silu_mul_twin8", "llmx_rms_norm_rows",
                                                           "llmx_rms_norm_16", "llmx_rms_norm_16_twin16", "llmx_rms_norm_16_twin8", "llmx_rms_norm_32",
                                                           "llmx_rms_norm_32_twin16", "llmx_rms_norm_32_twin8", "llmx_norm_rope", "llmx_norm_rope_kv", "llmx_gather",
                                                           "llmx_gather_q8_0", "llmx_matmul_f32", "llmx_matmul_q8_0", "llmx_matmul_q4_k", "llmx_gather_q4_k",
                                                           "llmx_matmul_q5_k", "llmx_gather_q5_k", "llmx_matmul_q6_k", "llmx_gather_q6_k", "llmx_matmul_q5_0",
                                                           "llmx_gather_q5_0", "llmx_matmul_q4_0",
                                                           "llmx_gather_q4_0", "llmx_matmul_q4_1",
                                                           "llmx_gather_q4_1", "llmx_matmul_q5_1", "llmx_gather_q5_1", "llmx_matmul_bf16", "llmx_gather_bf16",
                                                           "llmx_matmul_f16", "llmx_gather_f16",
                                                           "llmx_matmul_iq4_nl", "llmx_gather_iq4_nl", "llmx_matmul_mxfp4", "llmx_gather_mxfp4", "llmx_matmul_nvfp4", "llmx_gather_nvfp4", "llmx_matmul_q2_k", "llmx_gather_q2_k", "llmx_matmul_q3_k", "llmx_gather_q3_k", "llmx_matmul_iq4_xs", "llmx_gather_iq4_xs", "llmx_matmul_iq2_xxs", "llmx_gather_iq2_xxs", "llmx_matmul_iq2_xs", "llmx_gather_iq2_xs", "llmx_matmul_iq2_s", "llmx_gather_iq2_s", "llmx_matmul_iq3_xxs", "llmx_gather_iq3_xxs", "llmx_matmul_iq3_s", "llmx_gather_iq3_s", "llmx_matmul_iq1_s", "llmx_gather_iq1_s", "llmx_matmul_iq1_m", "llmx_gather_iq1_m", "llmx_matmul_tq1_0", "llmx_gather_tq1_0", "llmx_matmul_tq2_0", "llmx_gather_tq2_0", "llmx_matmul_q1_0", "llmx_gather_q1_0", "llmx_matmul_q2_0", "llmx_gather_q2_0", "llmx_matmul_f8_e4m3", "llmx_gather_f8_e4m3", "llmx_kv_write",
                                                           "llmx_attention_chunk", "llmx_attention_tile", "llmx_attention_tile256", "llmx_attention_merge", "llmx_attention_vec", "llmx_attention_vec_merge",
                                                           "llmx_twin16", "llmx_twin8", "llmx_silu_mul_twin16_major", "llmx_silu_mul_twin8_major",
                                                           "llmx_rms_norm_16_twin16_major", "llmx_rms_norm_16_twin8_major", "llmx_rms_norm_32_twin16_major",
                                                           "llmx_rms_norm_32_twin8_major", "llmx_attention_vec_major", "llmx_attention_vec_merge_major",
                                                           "llmx_twin16_major", "llmx_twin8_major", "llmx_group_sum", "llmx_group_chain", "llmx_group_shares", "llmx_group_copies", "llmx_group_arm", "llmx_group_add_check", "llmx_group_push", "llmx_group_take", "llmx_sigmoid_mul", "llmx_gated_rms_norm", "llmx_norm_rope_streams", "llmx_causal_conv_silu", "llmx_gated_delta_rule", "llmx_gated_delta_rule_short", "llmx_group_rms_norm", "llmx_silu_scaled", "llmx_mix_streams", "llmx_inject_streams", "llmx_signed_sqrt_gate", "llmx_dilated_conv_silu_add", "llmx_kv_index_pool", "llmx_kv_index_scores", "llmx_kv_index_select", "llmx_kv_index_select_wide", "llmx_kv_index_all", "llmx_sparse_attention_chunk", "llmx_sparse_attention_merge", "llmx_sparse_attention_vec", "llmx_sparse_attention_vec256", "llmx_sparse_attention_vec_merge", "llmx_sparse_attention_vec_merge256", "llmx_attention_vec256", "llmx_attention_vec256_major", "llmx_attention_vec_merge256", "llmx_attention_vec_merge256_major", "llmx_attention_gated256", "llmx_attention_gated_merge256", "llmx_argmax_slices", "llmx_argmax_merge", "llmx_argmax_part", "llmx_argmax_pick", "llmx_moe_route", "llmx_moe_combine", "llmx_moe_group", "llmx_routed_f32", "llmx_routed_q8_0", "llmx_routed_q4_k", "llmx_routed_q5_k", "llmx_routed_q6_k", "llmx_routed_q5_0", "llmx_routed_q4_0", "llmx_routed_q4_1", "llmx_routed_q5_1", "llmx_routed_bf16", "llmx_routed_f16", "llmx_routed_iq4_nl", "llmx_routed_mxfp4", "llmx_routed_nvfp4", "llmx_routed_q2_k", "llmx_routed_q3_k", "llmx_routed_iq4_xs", "llmx_routed_iq2_xxs", "llmx_routed_iq2_xs", "llmx_routed_iq2_s", "llmx_routed_iq3_xxs", "llmx_routed_iq3_s", "llmx_routed_iq1_s", "llmx_routed_iq1_m", "llmx_routed_tq1_0", "llmx_routed_tq2_0", "llmx_routed_q1_0", "llmx_routed_q2_0", "llmx_routed_f8_e4m3", "llmx_matmul_rows_f32", "llmx_matmul_rows_q8_0", "llmx_matmul_rows_q4_k", "llmx_matmul_rows_q5_k", "llmx_matmul_rows_q6_k", "llmx_matmul_rows_q5_0", "llmx_matmul_rows_q4_0", "llmx_matmul_rows_q4_1", "llmx_matmul_rows_q5_1", "llmx_matmul_rows_bf16", "llmx_matmul_rows_f16", "llmx_matmul_rows_iq4_nl", "llmx_matmul_rows_mxfp4", "llmx_matmul_rows_nvfp4", "llmx_matmul_rows_q2_k", "llmx_matmul_rows_q3_k", "llmx_matmul_rows_iq4_xs", "llmx_matmul_rows_iq2_xxs", "llmx_matmul_rows_iq2_xs", "llmx_matmul_rows_iq2_s", "llmx_matmul_rows_iq3_xxs", "llmx_matmul_rows_iq3_s", "llmx_matmul_rows_iq1_s", "llmx_matmul_rows_iq1_m", "llmx_matmul_rows_tq1_0", "llmx_matmul_rows_tq2_0", "llmx_matmul_rows_q1_0", "llmx_matmul_rows_q2_0", "llmx_matmul_rows_f8_e4m3", "llmx_sparse_attention_vec_g4", "llmx_sparse_attention_vec256_g4", "llmx_matmul_cols_few_f32", "llmx_matmul_cols_many_f32", "llmx_matmul_cols_few_bf16", "llmx_matmul_cols_many_bf16", "llmx_matmul_cols_few_f16", "llmx_matmul_cols_many_f16", "llmx_moe_step_route", "llmx_moe_step_route_f32_f16", "llmx_moe_step_route_f32_bf16", "llmx_moe_step_route_f16_f32", "llmx_moe_step_route_f16_f16", "llmx_moe_step_route_f16_bf16", "llmx_moe_step_route_bf16_f32", "llmx_moe_step_route_bf16_f16", "llmx_moe_step_route_bf16_bf16", "llmx_hc_coeffs_collapse", "llmx_hc_expand", "llmx_model_format_qdq", "llmx_topk_rows", "llmx_compress_pool", "llmx_indexer_scores", "llmx_indexer_scores_tile", "llmx_indexer_scores_int", "llmx_raw_index_scores", "llmx_pool_write", "llmx_sparse_scores_few16", "llmx_sparse_values_few16", "llmx_sparse_scores_few8", "llmx_sparse_values_few8", "llmx_sparse_scores_few4", "llmx_sparse_values_few4", "llmx_sparse_scores_few2", "llmx_sparse_values_few2", "llmx_sparse_scores_few32", "llmx_sparse_values_few16x2", "llmx_sparse_scores_tile", "llmx_sparse_values_tile", "llmx_sparse_scores_tile32", "llmx_sparse_values_tile32", "llmx_sparse_scores_few16_terms", "llmx_sparse_values_few16_terms", "llmx_sparse_scores_few8_terms", "llmx_sparse_values_few8_terms", "llmx_sparse_scores_few4_terms", "llmx_sparse_values_few4_terms", "llmx_sparse_scores_few2_terms", "llmx_sparse_values_few2_terms", "llmx_sparse_scores_few32_terms", "llmx_sparse_values_few16x2_terms", "llmx_sparse_scores_tile_terms", "llmx_sparse_values_tile_terms", "llmx_sparse_scores_tile32_terms", "llmx_sparse_values_tile32_terms", "llmx_route_biased", "llmx_route_step_f32", "llmx_route_step_bf16", "llmx_norm_rope_form", "llmx_delta_step", "llmx_mix_rows", "llmx_layer_norm_rows", "llmx_gelu", "llmx_add_bias", "llmx_copy_rows", "llmx_softcap", "llmx_l2_clip", "llmx_l2_clip_lanes", "llmx_query_mix", "llmx_query_mix4", "llmx_query_mix_rows1", "llmx_query_mix_rows2", "llmx_query_mix_rows3", "llmx_query_mix_rows4", "llmx_query_mix_rows5", "llmx_query_mix_norm", "llmx_tanh_gate_mix", "llmx_tanh_norm_16", "llmx_tanh_norm_16_twin16", "llmx_tanh_norm_16_twin8", "llmx_normed_gate_mix", "llmx_normed_gate_mix_twin16", "llmx_normed_gate_mix_twin8", "llmx_matmul_group_f32", "llmx_matmul_normed_f32", "llmx_conv_ring", "llmx_conv_ring_parts", "llmx_conv_ring_sum", "llmx_conv_ring_mix4", "llmx_attention_rel_sink", "llmx_attention_rel_lanes_g3", "llmx_attention_rel_lanes_g4", "llmx_rel_sink_store", "llmx_attention_rel_tile", "llmx_attention_rel_tile_g3", "llmx_attention_terms_g1", "llmx_attention_terms_g3", "llmx_attention_terms_g3_512", "llmx_attention_chunk_window", "llmx_attention_vec_window", "llmx_attention_vec256_window", "llmx_unit_rows", "llmx_sigmoid_gate_rows", "llmx_matmul_at_ids", "llmx_chain_pick", "llmx_tap_weighted_rows", "llmx_expert_mark", "llmx_expert_wave", "llmx_expert_pull", "llmx_expert_round", "llmx_q8_0_rows16_1",

                                                           "llmx_q8_0_rows16_2", "llmx_q8_0_rows16_2_r4", "llmx_q8_0_rows16_4", "llmx_q8_0_rows16_4_r4",
                                                           "llmx_q8_0_rows16_8", "llmx_q8_0_rows16_8_r4", "llmx_q8_0_rows16_16", "llmx_q8_0_tile16", "llmx_q8_0_rows8_1",
                                                           "llmx_q8_0_rows8_2", "llmx_q8_0_rows8_2_r4", "llmx_q8_0_rows8_4", "llmx_q8_0_rows8_4_r4", "llmx_q8_0_rows8_8",
                                                           "llmx_q8_0_rows8_8_r4", "llmx_q8_0_rows8_16", "llmx_q8_0_tile8", "llmx_tile_reduce", "llmx_q8_0_half8_1", "llmx_q8_0_half8_1p", "llmx_q8_0_half8_1r2", "llmx_q8_0_half8_1s1", "llmx_q8_0_half8_1r2s1", "llmx_q8_0_half8_1r2s4", "llmx_q8_0_half8_1h2", "llmx_q8_0_half8_1h3", "llmx_q8_0_half8_1h4", "llmx_q8_0_half8_1h5", "llmx_q8_0_half8_1h2p", "llmx_q8_0_half8_1h3p", "llmx_q8_0_half8_1h4p", "llmx_q8_0_half8_1h5p", "llmx_q8_0_half8_1h2_pair", "llmx_q8_0_half8_1h3_pair", "llmx_q8_0_half8_1h4_pair", "llmx_q8_0_half8_1h5_pair", "llmx_q8_0_half8_1h2_list", "llmx_q8_0_half8_1h3_list", "llmx_q8_0_half8_1h4_list", "llmx_q8_0_half8_1h5_list",
                                                           "llmx_q8_0_half8_2", "llmx_q8_0_half8_2_r4", "llmx_q8_0_half8_4", "llmx_q8_0_half8_4_r4", "llmx_q8_0_half8_8",
"llmx_q8_0_half8_8_r4", "llmx_q8_0_half8_16", "llmx_q8_0_half8_16_r4", "llmx_q8_0_half8_11", "llmx_q8_0_half8_12", "llmx_q8_0_half8_13", "llmx_q8_0_half8_14", "llmx_q8_0_half8_15", "llmx_q8_0_tile16_tall", "llmx_q8_0_tile8_tall", "llmx_q8_0_routed16", "llmx_q8_0_routed8", "llmx_q8_0_routed_half8", "llmx_q8_0_routed_half8s1", "llmx_q8_0_routed16_g2", "llmx_q8_0_routed16_g4", "llmx_q8_0_routed16_g8", "llmx_q8_0_routed8_g2", "llmx_q8_0_routed8_g4", "llmx_q8_0_routed8_g8", "llmx_q8_0_routed_half8_g2", "llmx_q8_0_routed_half8_g4", "llmx_q8_0_routed_half8_g8", "llmx_q8_0_routed_tile16", "llmx_q8_0_routed_tile8", "llmx_q8_0_routed_thin16", "llmx_q8_0_routed_mid16", "llmx_q8_0_routed_stream16", "llmx_mxfp4_routed_thin16",
                                                           "llmx_q4_k_rows16_1", "llmx_q4_k_rows16_2", "llmx_q4_k_rows16_4", "llmx_q4_k_rows16_8", "llmx_q4_k_rows16_16", "llmx_q4_k_rows8_1", "llmx_q4_k_rows8_2", "llmx_q4_k_rows8_4", "llmx_q4_k_rows8_8", "llmx_q4_k_rows8_16", "llmx_q4_k_tile16", "llmx_q4_k_tile8", "llmx_q4_k_routed16", "llmx_q4_k_routed8", "llmx_q4_k_routed16_g2", "llmx_q4_k_routed16_g4", "llmx_q4_k_routed16_g8", "llmx_q4_k_routed8_g2", "llmx_q4_k_routed8_g4", "llmx_q4_k_routed8_g8", "llmx_q4_k_routed_tile16", "llmx_q4_k_routed_tile8", "llmx_q4_k_routed_narrow16", "llmx_q4_k_routed_stream16_8", "llmx_q4_k_routed_stream8", "llmx_q4_k_routed_mid16", "llmx_q4_k_routed_mid8", "llmx_q5_k_rows16_1", "llmx_q5_k_rows16_2", "llmx_q5_k_rows16_4", "llmx_q5_k_rows16_8", "llmx_q5_k_rows16_16", "llmx_q5_k_rows8_1", "llmx_q5_k_rows8_2", "llmx_q5_k_rows8_4", "llmx_q5_k_rows8_8", "llmx_q5_k_rows8_16", "llmx_q5_k_tile16", "llmx_q5_k_tile8", "llmx_q5_k_routed16", "llmx_q5_k_routed8", "llmx_q5_k_routed16_g2", "llmx_q5_k_routed16_g4", "llmx_q5_k_routed16_g8", "llmx_q5_k_routed8_g2", "llmx_q5_k_routed8_g4", "llmx_q5_k_routed8_g8", "llmx_q5_k_routed_tile16", "llmx_q5_k_routed_tile8", "llmx_q5_k_routed_narrow16", "llmx_q5_k_routed_stream16_8", "llmx_q5_k_routed_stream8", "llmx_q5_k_routed_mid16", "llmx_q5_k_routed_mid8", "llmx_q6_k_rows16_1", "llmx_q6_k_rows16_2", "llmx_q6_k_rows16_4", "llmx_q6_k_rows16_8", "llmx_q6_k_rows16_16", "llmx_q6_k_rows8_1", "llmx_q6_k_rows8_2", "llmx_q6_k_rows8_4", "llmx_q6_k_rows8_8", "llmx_q6_k_rows8_16", "llmx_q6_k_tile16", "llmx_q6_k_tile8", "llmx_q6_k_routed16", "llmx_q6_k_routed8", "llmx_q6_k_routed16_g2", "llmx_q6_k_routed16_g4", "llmx_q6_k_routed16_g8", "llmx_q6_k_routed8_g2", "llmx_q6_k_routed8_g4", "llmx_q6_k_routed8_g8", "llmx_q6_k_routed_tile16", "llmx_q6_k_routed_tile8", "llmx_q6_k_routed_narrow16", "llmx_q6_k_routed_stream16_8", "llmx_q6_k_routed_stream8", "llmx_q6_k_routed_mid16", "llmx_q6_k_routed_mid8",
                                                           "llmx_q5_0_rows16_1", "llmx_q5_0_rows16_2", "llmx_q5_0_rows16_4", "llmx_q5_0_rows16_8", "llmx_q5_0_rows16_16", "llmx_q5_0_rows8_1", "llmx_q5_0_rows8_2", "llmx_q5_0_rows8_4", "llmx_q5_0_rows8_8", "llmx_q5_0_rows8_16", "llmx_q5_0_tile16", "llmx_q5_0_tile8", "llmx_q5_0_routed16", "llmx_q5_0_routed8", "llmx_q5_0_routed16_g2", "llmx_q5_0_routed16_g4", "llmx_q5_0_routed16_g8", "llmx_q5_0_routed8_g2", "llmx_q5_0_routed8_g4", "llmx_q5_0_routed8_g8", "llmx_q5_0_routed_tile16", "llmx_q5_0_routed_tile8", "llmx_q5_0_routed_narrow16", "llmx_q5_0_routed_stream16_8", "llmx_q5_0_routed_stream8", "llmx_q5_0_routed_mid16", "llmx_q5_0_routed_mid8", "llmx_q4_0_rows16_1", "llmx_q4_0_rows16_2", "llmx_q4_0_rows16_4", "llmx_q4_0_rows16_8", "llmx_q4_0_rows16_16", "llmx_q4_0_rows8_1", "llmx_q4_0_rows8_2", "llmx_q4_0_rows8_4", "llmx_q4_0_rows8_8", "llmx_q4_0_rows8_16", "llmx_q4_0_tile16", "llmx_q4_0_tile8", "llmx_q4_0_routed16", "llmx_q4_0_routed8", "llmx_q4_0_routed16_g2", "llmx_q4_0_routed16_g4", "llmx_q4_0_routed16_g8", "llmx_q4_0_routed8_g2", "llmx_q4_0_routed8_g4", "llmx_q4_0_routed8_g8", "llmx_q4_0_routed_tile16", "llmx_q4_0_routed_tile8", "llmx_q4_0_routed_narrow16", "llmx_q4_0_routed_stream16_8", "llmx_q4_0_routed_stream8", "llmx_q4_0_routed_mid16", "llmx_q4_0_routed_mid8", "llmx_q4_1_rows16_1", "llmx_q4_1_rows16_2", "llmx_q4_1_rows16_4", "llmx_q4_1_rows16_8", "llmx_q4_1_rows16_16", "llmx_q4_1_rows8_1", "llmx_q4_1_rows8_2", "llmx_q4_1_rows8_4", "llmx_q4_1_rows8_8", "llmx_q4_1_rows8_16", "llmx_q4_1_tile16", "llmx_q4_1_tile8", "llmx_q4_1_routed16", "llmx_q4_1_routed8", "llmx_q4_1_routed16_g2", "llmx_q4_1_routed16_g4", "llmx_q4_1_routed16_g8", "llmx_q4_1_routed8_g2", "llmx_q4_1_routed8_g4", "llmx_q4_1_routed8_g8", "llmx_q4_1_routed_tile16", "llmx_q4_1_routed_tile8", "llmx_q4_1_routed_narrow16", "llmx_q4_1_routed_stream16_8", "llmx_q4_1_routed_stream8", "llmx_q4_1_routed_mid16", "llmx_q4_1_routed_mid8", "llmx_q5_1_rows16_1", "llmx_q5_1_rows16_2", "llmx_q5_1_rows16_4", "llmx_q5_1_rows16_8", "llmx_q5_1_rows16_16", "llmx_q5_1_rows8_1", "llmx_q5_1_rows8_2", "llmx_q5_1_rows8_4", "llmx_q5_1_rows8_8", "llmx_q5_1_rows8_16", "llmx_q5_1_tile16", "llmx_q5_1_tile8", "llmx_q5_1_routed16", "llmx_q5_1_routed8", "llmx_q5_1_routed16_g2", "llmx_q5_1_routed16_g4", "llmx_q5_1_routed16_g8", "llmx_q5_1_routed8_g2", "llmx_q5_1_routed8_g4", "llmx_q5_1_routed8_g8", "llmx_q5_1_routed_tile16", "llmx_q5_1_routed_tile8", "llmx_q5_1_routed_narrow16", "llmx_q5_1_routed_stream16_8", "llmx_q5_1_routed_stream8", "llmx_q5_1_routed_mid16", "llmx_q5_1_routed_mid8",
                                                           "llmx_iq4_nl_rows16_1", "llmx_iq4_nl_rows16_2", "llmx_iq4_nl_rows16_4", "llmx_iq4_nl_rows16_8", "llmx_iq4_nl_rows16_16", "llmx_iq4_nl_rows8_1", "llmx_iq4_nl_rows8_2", "llmx_iq4_nl_rows8_4", "llmx_iq4_nl_rows8_8", "llmx_iq4_nl_rows8_16", "llmx_iq4_nl_tile16", "llmx_iq4_nl_tile8", "llmx_iq4_nl_routed16", "llmx_iq4_nl_routed8", "llmx_iq4_nl_routed16_g2", "llmx_iq4_nl_routed16_g4", "llmx_iq4_nl_routed16_g8", "llmx_iq4_nl_routed8_g2", "llmx_iq4_nl_routed8_g4", "llmx_iq4_nl_routed8_g8", "llmx_iq4_nl_routed_tile16", "llmx_iq4_nl_routed_tile8", "llmx_iq4_nl_routed_narrow16", "llmx_iq4_nl_routed_stream16_8", "llmx_iq4_nl_routed_stream8", "llmx_iq4_nl_routed_mid16", "llmx_iq4_nl_routed_mid8", "llmx_mxfp4_rows16_1", "llmx_mxfp4_rows16_2", "llmx_mxfp4_rows16_4", "llmx_mxfp4_rows16_8", "llmx_mxfp4_rows16_16", "llmx_mxfp4_rows8_1", "llmx_mxfp4_rows8_2", "llmx_mxfp4_rows8_4", "llmx_mxfp4_rows8_8", "llmx_mxfp4_rows8_16", "llmx_mxfp4_tile16", "llmx_mxfp4_tile8", "llmx_mxfp4_routed16", "llmx_mxfp4_routed8", "llmx_mxfp4_routed16_g2", "llmx_mxfp4_routed16_g4", "llmx_mxfp4_routed16_g8", "llmx_mxfp4_routed8_g2", "llmx_mxfp4_routed8_g4", "llmx_mxfp4_routed8_g8", "llmx_mxfp4_routed_tile16", "llmx_mxfp4_routed_tile8", "llmx_mxfp4_routed_narrow16", "llmx_mxfp4_routed_stream16_8", "llmx_mxfp4_routed_stream8", "llmx_mxfp4_routed_mid16", "llmx_mxfp4_routed_stream32_8", "llmx_mxfp4_quad_rows8_1", "llmx_mxfp4_quad_routed8",
                                                           "llmx_q3_k_rows16_1", "llmx_q3_k_rows16_2", "llmx_q3_k_rows16_4", "llmx_q3_k_rows16_8", "llmx_q3_k_rows16_16", "llmx_q3_k_rows8_1", "llmx_q3_k_rows8_2", "llmx_q3_k_rows8_4", "llmx_q3_k_rows8_8", "llmx_q3_k_rows8_16", "llmx_q3_k_tile16", "llmx_q3_k_tile8", "llmx_q3_k_routed16", "llmx_q3_k_routed8", "llmx_q3_k_routed16_g2", "llmx_q3_k_routed16_g4", "llmx_q3_k_routed16_g8", "llmx_q3_k_routed8_g2", "llmx_q3_k_routed8_g4", "llmx_q3_k_routed8_g8", "llmx_q3_k_routed_tile16", "llmx_q3_k_routed_tile8", "llmx_q3_k_routed_narrow16", "llmx_q3_k_routed_stream16_8", "llmx_q3_k_routed_stream8", "llmx_q3_k_routed_mid16", "llmx_q3_k_routed_mid8",
                                                           "llmx_q2_k_rows16_1", "llmx_q2_k_rows16_2", "llmx_q2_k_rows16_4", "llmx_q2_k_rows16_8", "llmx_q2_k_rows16_16", "llmx_q2_k_rows8_1", "llmx_q2_k_rows8_2", "llmx_q2_k_rows8_4", "llmx_q2_k_rows8_8", "llmx_q2_k_rows8_16", "llmx_q2_k_tile16", "llmx_q2_k_tile8", "llmx_q2_k_routed16", "llmx_q2_k_routed8", "llmx_q2_k_routed16_g2", "llmx_q2_k_routed16_g4", "llmx_q2_k_routed16_g8", "llmx_q2_k_routed8_g2", "llmx_q2_k_routed8_g4", "llmx_q2_k_routed8_g8", "llmx_q2_k_routed_tile16", "llmx_q2_k_routed_tile8", "llmx_q2_k_routed_narrow16", "llmx_q2_k_routed_stream16_8", "llmx_q2_k_routed_stream8", "llmx_q2_k_routed_mid16", "llmx_q2_k_routed_mid8",
                                                           "llmx_iq4_xs_rows16_1", "llmx_iq4_xs_rows16_2", "llmx_iq4_xs_rows16_4", "llmx_iq4_xs_rows16_8", "llmx_iq4_xs_rows16_16", "llmx_iq4_xs_rows8_1", "llmx_iq4_xs_rows8_2", "llmx_iq4_xs_rows8_4", "llmx_iq4_xs_rows8_8", "llmx_iq4_xs_rows8_16", "llmx_iq4_xs_tile16", "llmx_iq4_xs_tile8", "llmx_iq4_xs_routed16", "llmx_iq4_xs_routed8", "llmx_iq4_xs_routed16_g2", "llmx_iq4_xs_routed16_g4", "llmx_iq4_xs_routed16_g8", "llmx_iq4_xs_routed8_g2", "llmx_iq4_xs_routed8_g4", "llmx_iq4_xs_routed8_g8", "llmx_iq4_xs_routed_tile16", "llmx_iq4_xs_routed_tile8", "llmx_iq4_xs_routed_narrow16", "llmx_iq4_xs_routed_stream16_8", "llmx_iq4_xs_routed_stream8", "llmx_iq4_xs_routed_mid16", "llmx_iq4_xs_routed_mid8",
                                                           "llmx_iq2_xxs_rows16_1", "llmx_iq2_xxs_rows16_2", "llmx_iq2_xxs_rows16_4", "llmx_iq2_xxs_rows16_8", "llmx_iq2_xxs_rows16_16", "llmx_iq2_xxs_rows8_1", "llmx_iq2_xxs_rows8_2", "llmx_iq2_xxs_rows8_4", "llmx_iq2_xxs_rows8_8", "llmx_iq2_xxs_rows8_16", "llmx_iq2_xxs_tile16", "llmx_iq2_xxs_tile8", "llmx_iq2_xxs_routed16", "llmx_iq2_xxs_routed8", "llmx_iq2_xxs_routed16_g2", "llmx_iq2_xxs_routed16_g4", "llmx_iq2_xxs_routed16_g8", "llmx_iq2_xxs_routed8_g2", "llmx_iq2_xxs_routed8_g4", "llmx_iq2_xxs_routed8_g8", "llmx_iq2_xxs_routed_tile16", "llmx_iq2_xxs_routed_tile8", "llmx_iq2_xxs_routed_narrow16", "llmx_iq2_xxs_routed_stream16_8", "llmx_iq2_xxs_routed_stream8", "llmx_iq2_xxs_routed_mid16", "llmx_iq2_xxs_routed_mid8",
                                                           "llmx_iq2_xs_rows16_1", "llmx_iq2_xs_rows16_2", "llmx_iq2_xs_rows16_4", "llmx_iq2_xs_rows16_8", "llmx_iq2_xs_rows16_16", "llmx_iq2_xs_rows8_1", "llmx_iq2_xs_rows8_2", "llmx_iq2_xs_rows8_4", "llmx_iq2_xs_rows8_8", "llmx_iq2_xs_rows8_16", "llmx_iq2_xs_tile16", "llmx_iq2_xs_tile8", "llmx_iq2_xs_routed16", "llmx_iq2_xs_routed8", "llmx_iq2_xs_routed16_g2", "llmx_iq2_xs_routed16_g4", "llmx_iq2_xs_routed16_g8", "llmx_iq2_xs_routed8_g2", "llmx_iq2_xs_routed8_g4", "llmx_iq2_xs_routed8_g8", "llmx_iq2_xs_routed_tile16", "llmx_iq2_xs_routed_tile8", "llmx_iq2_xs_routed_narrow16", "llmx_iq2_xs_routed_stream16_8", "llmx_iq2_xs_routed_stream8", "llmx_iq2_xs_routed_mid16", "llmx_iq2_xs_routed_mid8",
                                                           "llmx_iq3_s_rows16_1", "llmx_iq3_s_rows16_2", "llmx_iq3_s_rows16_4", "llmx_iq3_s_rows16_8", "llmx_iq3_s_rows16_16", "llmx_iq3_s_rows8_1", "llmx_iq3_s_rows8_2", "llmx_iq3_s_rows8_4", "llmx_iq3_s_rows8_8", "llmx_iq3_s_rows8_16", "llmx_iq3_s_tile16", "llmx_iq3_s_tile8", "llmx_iq3_s_routed16", "llmx_iq3_s_routed8", "llmx_iq3_s_routed16_g2", "llmx_iq3_s_routed16_g4", "llmx_iq3_s_routed16_g8", "llmx_iq3_s_routed8_g2", "llmx_iq3_s_routed8_g4", "llmx_iq3_s_routed8_g8", "llmx_iq3_s_routed_tile16", "llmx_iq3_s_routed_tile8", "llmx_iq3_s_routed_narrow16", "llmx_iq3_s_routed_stream16_8", "llmx_iq3_s_routed_stream8", "llmx_iq3_s_routed_mid16", "llmx_iq3_s_routed_mid8",
                                                           "llmx_iq2_s_rows16_1", "llmx_iq2_s_rows16_2", "llmx_iq2_s_rows16_4", "llmx_iq2_s_rows16_8", "llmx_iq2_s_rows16_16", "llmx_iq2_s_rows8_1", "llmx_iq2_s_rows8_2", "llmx_iq2_s_rows8_4", "llmx_iq2_s_rows8_8", "llmx_iq2_s_rows8_16", "llmx_iq2_s_tile16", "llmx_iq2_s_tile8", "llmx_iq2_s_routed16", "llmx_iq2_s_routed8", "llmx_iq2_s_routed16_g2", "llmx_iq2_s_routed16_g4", "llmx_iq2_s_routed16_g8", "llmx_iq2_s_routed8_g2", "llmx_iq2_s_routed8_g4", "llmx_iq2_s_routed8_g8", "llmx_iq2_s_routed_tile16", "llmx_iq2_s_routed_tile8", "llmx_iq2_s_routed_narrow16", "llmx_iq2_s_routed_stream16_8", "llmx_iq2_s_routed_stream8", "llmx_iq2_s_routed_mid16", "llmx_iq2_s_routed_mid8",
                                                           "llmx_iq3_xxs_rows16_1", "llmx_iq3_xxs_rows16_2", "llmx_iq3_xxs_rows16_4", "llmx_iq3_xxs_rows16_8", "llmx_iq3_xxs_rows16_16", "llmx_iq3_xxs_rows8_1", "llmx_iq3_xxs_rows8_2", "llmx_iq3_xxs_rows8_4", "llmx_iq3_xxs_rows8_8", "llmx_iq3_xxs_rows8_16", "llmx_iq3_xxs_tile16", "llmx_iq3_xxs_tile8", "llmx_iq3_xxs_routed16", "llmx_iq3_xxs_routed8", "llmx_iq3_xxs_routed16_g2", "llmx_iq3_xxs_routed16_g4", "llmx_iq3_xxs_routed16_g8", "llmx_iq3_xxs_routed8_g2", "llmx_iq3_xxs_routed8_g4", "llmx_iq3_xxs_routed8_g8", "llmx_iq3_xxs_routed_tile16", "llmx_iq3_xxs_routed_tile8", "llmx_iq3_xxs_routed_narrow16", "llmx_iq3_xxs_routed_stream16_8", "llmx_iq3_xxs_routed_stream8", "llmx_iq3_xxs_routed_mid16", "llmx_iq3_xxs_routed_mid8",
                                                           "llmx_iq1_s_rows16_1", "llmx_iq1_s_rows16_2", "llmx_iq1_s_rows16_4", "llmx_iq1_s_rows16_8", "llmx_iq1_s_rows16_16", "llmx_iq1_s_rows8_1", "llmx_iq1_s_rows8_2", "llmx_iq1_s_rows8_4", "llmx_iq1_s_rows8_8", "llmx_iq1_s_rows8_16", "llmx_iq1_s_tile16", "llmx_iq1_s_tile8", "llmx_iq1_s_routed16", "llmx_iq1_s_routed8", "llmx_iq1_s_routed16_g2", "llmx_iq1_s_routed16_g4", "llmx_iq1_s_routed16_g8", "llmx_iq1_s_routed8_g2", "llmx_iq1_s_routed8_g4", "llmx_iq1_s_routed8_g8", "llmx_iq1_s_routed_tile16", "llmx_iq1_s_routed_tile8", "llmx_iq1_s_routed_narrow16", "llmx_iq1_s_routed_stream16_8", "llmx_iq1_s_routed_stream8", "llmx_iq1_s_routed_mid16", "llmx_iq1_s_routed_mid8",
                                                           "llmx_iq1_m_rows16_1", "llmx_iq1_m_rows16_2", "llmx_iq1_m_rows16_4", "llmx_iq1_m_rows16_8", "llmx_iq1_m_rows16_16", "llmx_iq1_m_rows8_1", "llmx_iq1_m_rows8_2", "llmx_iq1_m_rows8_4", "llmx_iq1_m_rows8_8", "llmx_iq1_m_rows8_16", "llmx_iq1_m_tile16", "llmx_iq1_m_tile8", "llmx_iq1_m_routed16", "llmx_iq1_m_routed8", "llmx_iq1_m_routed16_g2", "llmx_iq1_m_routed16_g4", "llmx_iq1_m_routed16_g8", "llmx_iq1_m_routed8_g2", "llmx_iq1_m_routed8_g4", "llmx_iq1_m_routed8_g8", "llmx_iq1_m_routed_tile16", "llmx_iq1_m_routed_tile8", "llmx_iq1_m_routed_narrow16", "llmx_iq1_m_routed_stream16_8", "llmx_iq1_m_routed_stream8", "llmx_iq1_m_routed_mid16", "llmx_iq1_m_routed_mid8",
                                                           "llmx_q1_0_rows16_1", "llmx_q1_0_rows16_2", "llmx_q1_0_rows16_4", "llmx_q1_0_rows16_8", "llmx_q1_0_rows16_16", "llmx_q1_0_rows8_1", "llmx_q1_0_rows8_2", "llmx_q1_0_rows8_4", "llmx_q1_0_rows8_8", "llmx_q1_0_rows8_16", "llmx_q1_0_tile16", "llmx_q1_0_tile8", "llmx_q1_0_routed16", "llmx_q1_0_routed8", "llmx_q1_0_routed16_g2", "llmx_q1_0_routed16_g4", "llmx_q1_0_routed16_g8", "llmx_q1_0_routed8_g2", "llmx_q1_0_routed8_g4", "llmx_q1_0_routed8_g8", "llmx_q1_0_routed_tile16", "llmx_q1_0_routed_tile8", "llmx_q1_0_routed_narrow16", "llmx_q1_0_routed_stream16_8", "llmx_q1_0_routed_stream8", "llmx_q1_0_routed_mid16", "llmx_q1_0_routed_mid8",
                                                           "llmx_q2_0_rows16_1", "llmx_q2_0_rows16_2", "llmx_q2_0_rows16_4", "llmx_q2_0_rows16_8", "llmx_q2_0_rows16_16", "llmx_q2_0_rows8_1", "llmx_q2_0_rows8_2", "llmx_q2_0_rows8_4", "llmx_q2_0_rows8_8", "llmx_q2_0_rows8_16", "llmx_q2_0_tile16", "llmx_q2_0_tile8", "llmx_q2_0_routed16", "llmx_q2_0_routed8", "llmx_q2_0_routed16_g2", "llmx_q2_0_routed16_g4", "llmx_q2_0_routed16_g8", "llmx_q2_0_routed8_g2", "llmx_q2_0_routed8_g4", "llmx_q2_0_routed8_g8", "llmx_q2_0_routed_tile16", "llmx_q2_0_routed_tile8", "llmx_q2_0_routed_narrow16", "llmx_q2_0_routed_stream16_8", "llmx_q2_0_routed_stream8", "llmx_q2_0_routed_mid16", "llmx_q2_0_routed_mid8",
                                                           "llmx_nvfp4_rows16_1", "llmx_nvfp4_rows16_2", "llmx_nvfp4_rows16_4", "llmx_nvfp4_rows16_8", "llmx_nvfp4_rows16_16", "llmx_nvfp4_rows8_1", "llmx_nvfp4_rows8_2", "llmx_nvfp4_rows8_4", "llmx_nvfp4_rows8_8", "llmx_nvfp4_rows8_16", "llmx_nvfp4_tile16", "llmx_nvfp4_tile8", "llmx_nvfp4_routed16", "llmx_nvfp4_routed8", "llmx_nvfp4_routed16_g2", "llmx_nvfp4_routed16_g4", "llmx_nvfp4_routed16_g8", "llmx_nvfp4_routed8_g2", "llmx_nvfp4_routed8_g4", "llmx_nvfp4_routed8_g8", "llmx_nvfp4_routed_tile16", "llmx_nvfp4_routed_tile8", "llmx_nvfp4_routed_narrow16", "llmx_nvfp4_routed_stream16_8", "llmx_nvfp4_routed_stream8", "llmx_nvfp4_routed_mid16", "llmx_nvfp4_routed_mid8",
                                                           "llmx_q5_0_quarter_rows16_1", "llmx_q5_0_quarter_rows16_2", "llmx_q5_0_quarter_rows16_4", "llmx_q5_0_quarter_rows16_8", "llmx_q5_0_quarter_rows16_16", "llmx_q5_0_quarter_routed16", "llmx_q5_0_quarter_routed16_g2", "llmx_q5_0_quarter_routed16_g4", "llmx_q5_0_quarter_routed16_g8", "llmx_q4_0_quarter_rows16_1", "llmx_q4_0_quarter_rows16_2", "llmx_q4_0_quarter_rows16_4", "llmx_q4_0_quarter_rows16_8", "llmx_q4_0_quarter_rows16_16", "llmx_q4_0_quarter_routed16", "llmx_q4_0_quarter_routed16_g2", "llmx_q4_0_quarter_routed16_g4", "llmx_q4_0_quarter_routed16_g8", "llmx_q4_1_quarter_rows16_1", "llmx_q4_1_quarter_rows16_2", "llmx_q4_1_quarter_rows16_4", "llmx_q4_1_quarter_rows16_8", "llmx_q4_1_quarter_rows16_16", "llmx_q4_1_quarter_routed16", "llmx_q4_1_quarter_routed16_g2", "llmx_q4_1_quarter_routed16_g4", "llmx_q4_1_quarter_routed16_g8", "llmx_q5_1_quarter_rows16_1", "llmx_q5_1_quarter_rows16_2", "llmx_q5_1_quarter_rows16_4", "llmx_q5_1_quarter_rows16_8", "llmx_q5_1_quarter_rows16_16", "llmx_q5_1_quarter_routed16", "llmx_q5_1_quarter_routed16_g2", "llmx_q5_1_quarter_routed16_g4", "llmx_q5_1_quarter_routed16_g8", "llmx_hc_mix", "llmx_hc_mix8", "llmx_hc_entry", "llmx_hc_entry8", "llmx_moe_step_up_q4_k", "llmx_moe_step_up_q4_k_w", "llmx_moe_step_up_q5_k", "llmx_moe_step_up_q5_k_w", "llmx_moe_step_up_q6_k", "llmx_moe_step_up_q6_k_w", "llmx_moe_step_up_q4_0", "llmx_moe_step_up_q4_0_w", "llmx_moe_step_up_q4_1", "llmx_moe_step_up_q4_1_w", "llmx_moe_step_up_q5_0", "llmx_moe_step_up_q5_0_w", "llmx_moe_step_up_q5_1", "llmx_moe_step_up_q5_1_w", "llmx_moe_step_up_q8_0", "llmx_moe_step_up_q8_0_w", "llmx_moe_step_down_q8_0", "llmx_moe_step_down_q4_k", "llmx_moe_step_down_q5_k", "llmx_moe_step_down_q6_k", "llmx_moe_step_down_q4_0", "llmx_moe_step_down_q4_1", "llmx_moe_step_down_q5_0", "llmx_moe_step_down_q5_1", "llmx_moe_step_up_q4_k_c2", "llmx_moe_step_up_q4_k_c2_w", "llmx_moe_step_up_q5_k_c2", "llmx_moe_step_up_q5_k_c2_w", "llmx_moe_step_up_q6_k_c2", "llmx_moe_step_up_q6_k_c2_w", "llmx_moe_step_up_q4_0_c2", "llmx_moe_step_up_q4_0_c2_w", "llmx_moe_step_up_q4_1_c2", "llmx_moe_step_up_q4_1_c2_w", "llmx_moe_step_up_q5_0_c2", "llmx_moe_step_up_q5_0_c2_w", "llmx_moe_step_up_q5_1_c2", "llmx_moe_step_up_q5_1_c2_w", "llmx_moe_step_up_q4_k_c4", "llmx_moe_step_up_q4_k_c4_w", "llmx_moe_step_up_q5_k_c4", "llmx_moe_step_up_q5_k_c4_w", "llmx_moe_step_up_q6_k_c4", "llmx_moe_step_up_q6_k_c4_w", "llmx_moe_step_up_q4_0_c4", "llmx_moe_step_up_q4_0_c4_w", "llmx_moe_step_up_q4_1_c4", "llmx_moe_step_up_q4_1_c4_w", "llmx_moe_step_up_q5_0_c4", "llmx_moe_step_up_q5_0_c4_w", "llmx_moe_step_up_q5_1_c4", "llmx_moe_step_up_q5_1_c4_w", "llmx_moe_step_down_q4_k_c2", "llmx_moe_step_down_q5_k_c2", "llmx_moe_step_down_q6_k_c2", "llmx_moe_step_down_q4_0_c2", "llmx_moe_step_down_q4_1_c2", "llmx_moe_step_down_q5_0_c2", "llmx_moe_step_down_q5_1_c2", "llmx_moe_step_down_q4_k_c4", "llmx_moe_step_down_q5_k_c4", "llmx_moe_step_down_q6_k_c4", "llmx_moe_step_down_q4_0_c4", "llmx_moe_step_down_q4_1_c4", "llmx_moe_step_down_q5_0_c4", "llmx_moe_step_down_q5_1_c4", "llmx_f16_tile", "llmx_f8_e4m3_tile", "llmx_f32_tile", "llmx_q4_k_tile8_tall", "llmx_q5_k_tile8_tall", "llmx_q6_k_tile8_tall", "llmx_q4_k_tile8_square", "llmx_q5_k_tile8_square", "llmx_q6_k_tile8_square", "llmx_q8_0_rows16_1_walk3", "llmx_q8_0_rows16_1_walk5", "llmx_q8_0_routed16_walk3", "llmx_q8_0_routed16_walk5", "llmx_q8_0_routed16_walk3_add", "llmx_q8_0_routed16_walk5_add", "llmx_q8_0_rows16_1_pair", "llmx_q8_0_rows16_1_walk3_pair", "llmx_q8_0_rows16_1_walk5_pair", "llmx_q8_0_rows16_1_conv", "llmx_q8_0_rows8_1_conv", "llmx_q8_0_routed16_walk3_add_conv", "llmx_q8_0_routed16_walk5_add_conv", "llmx_q8_0_rows16_1_walk2", "llmx_q8_0_routed16_walk2", "llmx_q8_0_routed16_walk2_add", "llmx_q8_0_rows16_1_walk2_pair", "llmx_q8_0_routed16_walk2_add_conv", "llmx_q8_0_routed16_walk3_glu", "llmx_q8_0_routed16_walk5_glu", "llmx_f32_tile_mix", "llmx_bf16_tile", "llmx_bf16_tile_bf16", "llmx_q8_0_routed_half8_walk_add", "llmx_q8_0_routed_half8_walk_add_conv", "llmx_q8_0_routed_half8_walk3_glu"};
    // A type's twin kernels by form and columns, with its routed builds' break-evens and tile extents (docs/src/backends-gpu.md, weight types).
    // kKernels marks a build the type lacks.
    struct TwinKernels {
        Kernel two[3][5];
        size_t two_rows[3][5];
        Kernel four[3][5];
        Kernel tile[2], tall[2];
        Kernel routed[3];
        Kernel grouped[3][3];
        Kernel routed_tile[2], routed_narrow[2], routed_mid[2];
        Kernel routed_thin;     // the 16-bit twin's thin build (kernels/int16.gpu): MXFP4's groups of up to kThinRows entries (k_thin), Q8_0's of up to kNarrowRows (q8_0_thin)
        Kernel routed_stream;   // the 8-bit twin's routed stream, groups of up to kStreamRows entries, Q8_0's up to kNarrowRows (kernels/int16.gpu, routed_stream)
        float group_from[2];
        size_t tile_from[2][2];
        size_t block_values;   // a weight block's values
        unsigned row_lanes;    // the row kernel's lane units a weight block (kernels/int16.gpu, KRaw's Lane), 0 where a wave takes a row
        unsigned row_chains;   // the lane units a lane holds, each in an accumulator of its own: 2 for MXFP4 (kernels/int16.gpu, MXFP4Lane), else 1
        bool k45;              // Q4_K or Q5_K, whose rows take at most the profile's k45_row_lanes
        const TwinKernels* quarter;   // the quarter form's row, routed and grouped builds (kernels/int16.gpu, QuarterLane), which row_form may take instead
        bool tile_sums;   // whether the prompt tiles read the twin's half sums: every type's but Q8_0's, whose tile has no min or offset term
        // Which of the 8-bit twin's routed_narrow and routed_mid are streams at 16 and 32 rows, a wave a block (kernels/int16.gpu, routed_stream), rather than tiles.
        enum class Ranges : uint8_t { tiles, narrow, both } ranges = Ranges::tiles;
        Kernel square = kKernels;   // the 8-bit tall build at 128 rows a block, which a call of more than kTileRows rows takes (kernels/int16.gpu, LLMX_KSQUARE8)
        Kernel step_up = kKernels, step_down = kKernels;   // the fused moe_step's gate+up, its wide build the next kernel, and its down (kernels/int16.gpu, LLMX_STEP_UP_CN and LLMX_STEP_DOWN_CN, and step_build)
        // Whether routed products of entries over n_expert take the grouped builds, whose break-even is group_from.
        constexpr bool groups(size_t entries, size_t n_expert, bool int8) const { return entries && double(entries) >= double(group_from[int8]) * double(n_expert); }
    };
    // Whether a row of n values takes lane_row: one chain a lane unit, on a cluster at most the chains a lane of 16 holds at the widest row max (step_chains).
    bool step_lanes(const TwinKernels& tk, size_t n, size_t max) const {
        const size_t units = max / tk.block_values * tk.row_lanes, chains = units > 32 ? 4 : units > 16 ? 2 : 1;
        return tk.row_lanes && tk.row_chains == 1 && n % tk.block_values == 0 && row_cluster(tk, n) <= 16 * chains;
    }
    // The chain count's build a lane_row row of n values takes, 0, 1 or 2 for one, two or four chains a lane: its routed cluster over 16 lanes (kernels/int16.gpu, LLMX_STEP).
    unsigned step_chains(const TwinKernels& tk, size_t n) const {
        const unsigned cl = tk.row_lanes ? row_cluster(tk, n) : 0;
        return cl > 32 ? 2 : cl > 16 ? 1 : 0;
    }
    // The fused step route's build for a router of type rt and a row gate of type gt, both with a route decoder (fused_step): three builds a router type, one a row gate type.
    static Kernel step_route_kernel(uint32_t rt, uint32_t gt) {
        static_assert(kMoeStepRouteBF16BF16 - kMoeStepRoute == 8, "kernels/f32.gpu has a route build for each pair of the three decoders");
        return static_cast<Kernel>(kMoeStepRoute + 3 * type_kernels(rt)->step_route + type_kernels(gt)->step_route);
    }
    // A one-chain build's build of chain count c: the _c2 and _c4 builds follow the one-chain builds of the lane_row types in their order.
    static Kernel step_build(Kernel k, unsigned c) {
        static_assert(kMoeStepUpQ51WC2 - kMoeStepUpQ4KC2 == kMoeStepUpQ51W - kMoeStepUpQ4K && kMoeStepUpQ51WC4 - kMoeStepUpQ4KC4 == kMoeStepUpQ51W - kMoeStepUpQ4K);
        static_assert(kMoeStepDownQ51C2 - kMoeStepDownQ4KC2 == kMoeStepDownQ51 - kMoeStepDownQ4K && kMoeStepDownQ51C4 - kMoeStepDownQ4KC4 == kMoeStepDownQ51 - kMoeStepDownQ4K);
        if (c == 0) return k;
        if (k < kMoeStepDownQ8) return static_cast<Kernel>((c == 1 ? kMoeStepUpQ4KC2 : kMoeStepUpQ4KC4) + (k - kMoeStepUpQ4K));
        return static_cast<Kernel>((c == 1 ? kMoeStepDownQ4KC2 : kMoeStepDownQ4KC4) + (k - kMoeStepDownQ4K));
    }
    static constexpr auto with_step = [](TwinKernels t, Kernel up, Kernel down) { t.step_up = up, t.step_down = down; return t; };
    static constexpr TwinKernels kQ8Twins = with_step(TwinKernels{{{kRows16x1, kRows16x2, kRows16x4, kRows16x8, kRows16x16},
                                          {kRows8x1, kRows8x2, kRows8x4, kRows8x8, kRows8x16},
                                          {kHalf8x1, kHalf8x2, kHalf8x4, kHalf8x8, kHalf8x16}},
                                         {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 4}, {1, 2, 2, 2, 2}},
                                         {{kKernels, kRows16x2r4, kRows16x4r4, kRows16x8r4, kKernels},
                                          {kKernels, kRows8x2r4, kRows8x4r4, kRows8x8r4, kKernels},
                                          {kKernels, kHalf8x2r4, kHalf8x4r4, kHalf8x8r4, kHalf8x16r4}},
                                         {kTile16, kTile8},
                                         {kTile16Tall, kTile8Tall},
                                         {kRouted16, kRouted8, kRoutedHalf8},
                                         {{kGrouped16x2, kGrouped16x4, kGrouped16x8}, {kGrouped8x2, kGrouped8x4, kGrouped8x8}, {kGroupedHalf8x2, kGroupedHalf8x4, kGroupedHalf8x8}},
                                         {kRoutedTile16, kRoutedTile8},
                                         {kKernels, kKernels},
                                         {kRoutedMid16, kKernels},
                                         kRoutedThin16,
                                         kRoutedStream16,
                                         {1.25f, 0.0f},
                                         {{32, 32}, {32, 32}},
                                         32,
                                         0,
                                         1,
                                         false,
                                         nullptr,
                                         false}, kMoeStepUpQ8, kMoeStepDownQ8);
    // A K-type's twins from its first kernel, its set in LLMX_KTYPE's order (kernels/int16.gpu), which the static_assert below holds.
    static constexpr auto k_twins = [](Kernel first, float group_from16, float group_from8, size_t up_from16, size_t up_from8, size_t down_from16, size_t down_from8, size_t block_values, unsigned row_lanes, bool k45) {
        const auto k = [first](int i) { return static_cast<Kernel>(first + i); };
        return TwinKernels{{{k(0), k(1), k(2), k(3), k(4)}, {k(5), k(6), k(7), k(8), k(9)}, {k(5), k(6), k(7), k(8), k(9)}},
                {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}, {2, 2, 2, 2, 2}},
                {{kKernels, kKernels, kKernels, kKernels, kKernels}, {kKernels, kKernels, kKernels, kKernels, kKernels}, {kKernels, kKernels, kKernels, kKernels, kKernels}},
                {k(10), k(11)},
                {kKernels, kKernels},
                {k(12), k(13), k(13)},
                {{k(14), k(15), k(16)}, {k(17), k(18), k(19)}, {k(17), k(18), k(19)}},
                {k(20), k(21)},
                {k(22), k(23)},
                {k(25), k(26)},
                kKernels,
                k(24),
                {group_from16, group_from8},
                {{up_from16, up_from8}, {down_from16, down_from8}},
                block_values,
                row_lanes,
                1,
                k45,
                nullptr,
                true,
                TwinKernels::Ranges::narrow};
    };
    // The quarter form of a 32-value type from its first quarter build, in LLMX_KQUARTER's order (kernels/int16.gpu): four lanes a block in the F16 class alone, its tiles the type's own.
    // row_form never takes it in the int8 classes, so those keep the base's builds.
    static constexpr auto quarter_twins = [](const TwinKernels& base, Kernel first, float group_from16) {
        const auto k = [first](int i) { return static_cast<Kernel>(first + i); };
        TwinKernels q = base;
        for (int b = 0; b < 5; ++b) q.two[0][b] = k(b);
        q.routed[0] = k(5);
        for (int b = 0; b < 3; ++b) q.grouped[0][b] = k(6 + b);
        q.group_from[0] = group_from16;
        q.row_lanes = 4;
        q.quarter = nullptr;
        return q;
    };
    // The tall 8-bit builds of the K-quants at 64 and 128 rows, the tensor-core tile that only the CUDA module has (kernels/int16.gpu, k_mma at 128 outputs).
    static constexpr auto with_tall8 = [](TwinKernels t, Kernel k, Kernel square) { t.tall[1] = k, t.square = square; return t; };
    static constexpr TwinKernels kQ4KTwins = with_step(with_tall8(k_twins(kRowsQ4K16x1, 2.5f, 1.25f, 128, 32, 32, 32, 256, 8, true), kTallQ4K8, kSquareQ4K8), kMoeStepUpQ4K, kMoeStepDownQ4K),
                                 kQ5KTwins = with_step(with_tall8(k_twins(kRowsQ5K16x1, 2.5f, 1.25f, 128, 32, 32, 32, 256, 8, true), kTallQ5K8, kSquareQ5K8), kMoeStepUpQ5K, kMoeStepDownQ5K),
                                 kQ6KTwins = with_step(with_tall8(k_twins(kRowsQ6K16x1, 1.25f, 1.25f, 128, 32, 32, 32, 256, 16, false), kTallQ6K8, kSquareQ6K8), kMoeStepUpQ6K, kMoeStepDownQ6K);
    static constexpr TwinKernels kQ50Base = k_twins(kRowsQ5016x1, 1.25f, 1.25f, 128, 32, 32, 32, 32, 1, false), kQ51Base = k_twins(kRowsQ5116x1, 5.0f, 1.25f, 128, 32, 32, 32, 32, 1, false), kIQ4NLTwins = k_twins(kRowsIQ4NL16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 32, 4, false), kMXFP4Base = k_twins(kRowsMXFP416x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 32, 4, false);
    static constexpr TwinKernels kQ3KTwins = k_twins(kRowsQ3K16x1, 1.25f, 1.25f, 128, 32, 32, 32, 256, 16, false), kQ2KTwins = k_twins(kRowsQ2K16x1, 1.25f, 1.25f, 128, 32, 32, 32, 256, 16, false), kIQ4XSTwins = k_twins(kRowsIQ4XS16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false);
    static constexpr TwinKernels kIQ2XXSTwins = k_twins(kRowsIQ2XXS16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false), kIQ2XSTwins = k_twins(kRowsIQ2XS16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false), kIQ3STwins = k_twins(kRowsIQ3S16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false);
    static constexpr TwinKernels kIQ2STwins = k_twins(kRowsIQ2S16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false), kIQ3XXSTwins = k_twins(kRowsIQ3XXS16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false);
    static_assert(kIQ2STwins.routed_mid[1] == kRoutedMidIQ2S8 && kIQ3XXSTwins.routed_mid[1] == kRoutedMidIQ3XXS8 && kIQ3XXSTwins.routed_narrow[1] == kRoutedStream16IQ3XXS8, "IQ2_S's and IQ3_XXS's kernels follow LLMX_KTYPE's order");
    static constexpr TwinKernels kIQ1STwins = k_twins(kRowsIQ1S16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false), kIQ1MTwins = k_twins(kRowsIQ1M16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 256, 32, false);
    static_assert(kIQ1STwins.routed_mid[1] == kRoutedMidIQ1S8 && kIQ1MTwins.routed_mid[1] == kRoutedMidIQ1M8 && kIQ1MTwins.routed_narrow[1] == kRoutedStream16IQ1M8, "IQ1_S's and IQ1_M's kernels follow LLMX_KTYPE's order");
    static constexpr TwinKernels kQ10Twins = k_twins(kRowsQ1016x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 128, 16, false), kQ20Twins = k_twins(kRowsQ2016x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 64, 8, false);
    static_assert(kQ10Twins.routed_mid[1] == kRoutedMidQ108 && kQ20Twins.routed_mid[1] == kRoutedMidQ208 && kQ20Twins.routed_narrow[1] == kRoutedStream16Q208, "Q1_0's and Q2_0's kernels follow LLMX_KTYPE's order");
    static constexpr TwinKernels kNVFP4Twins = k_twins(kRowsNVFP416x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 64, 8, false);
    static_assert(kNVFP4Twins.routed_mid[1] == kRoutedMidNVFP48 && kNVFP4Twins.routed_narrow[1] == kRoutedStream16NVFP48, "NVFP4's kernels follow LLMX_KTYPE's order");
    static_assert(kIQ2XXSTwins.routed_mid[1] == kRoutedMidIQ2XXS8 && kIQ2XSTwins.routed_mid[1] == kRoutedMidIQ2XS8 && kIQ3STwins.routed_mid[1] == kRoutedMidIQ3S8 && kIQ3STwins.routed_narrow[1] == kRoutedStream16IQ3S8, "IQ2_XXS's, IQ2_XS's and IQ3_S's kernels follow LLMX_KTYPE's order");
    static_assert(kQ3KTwins.routed_mid[1] == kRoutedMidQ3K8 && kQ2KTwins.routed_mid[1] == kRoutedMidQ2K8 && kIQ4XSTwins.routed_mid[1] == kRoutedMidIQ4XS8 && kIQ4XSTwins.routed_narrow[1] == kRoutedStream16IQ4XS8, "Q3_K's, Q2_K's and IQ4_XS's kernels follow LLMX_KTYPE's order");
    static constexpr TwinKernels kQ40Base = k_twins(kRowsQ4016x1, 5.0f, 5.0f, 128, 32, 128, 32, 32, 1, false), kQ41Base = k_twins(kRowsQ4116x1, 5.0f, 5.0f, 128, 32, 128, 32, 32, 1, false);
    static constexpr TwinKernels kQ40Quarter = quarter_twins(kQ40Base, kQuarterQ4016x1, 5.0f), kQ41Quarter = quarter_twins(kQ41Base, kQuarterQ4116x1, 2.5f);
    static constexpr TwinKernels kQ50Quarter = quarter_twins(kQ50Base, kQuarterQ5016x1, 1.25f), kQ51Quarter = quarter_twins(kQ51Base, kQuarterQ5116x1, 1.25f);
    static constexpr auto with_quarter = [](TwinKernels t, const TwinKernels* q) { t.quarter = q; return t; };
    static constexpr TwinKernels kQ50Twins = with_step(with_quarter(kQ50Base, &kQ50Quarter), kMoeStepUpQ50, kMoeStepDownQ50), kQ51Twins = with_step(with_quarter(kQ51Base, &kQ51Quarter), kMoeStepUpQ51, kMoeStepDownQ51);
    static constexpr TwinKernels kQ40Twins = with_step(with_quarter(kQ40Base, &kQ40Quarter), kMoeStepUpQ40, kMoeStepDownQ40), kQ41Twins = with_step(with_quarter(kQ41Base, &kQ41Quarter), kMoeStepUpQ41, kMoeStepDownQ41);
    // MXFP4 alone has the thin build, which the narrow tile beat on Q4_K (docs/STATUS.md, DeepSeek-V4.1 ROCm kernels).
    static constexpr TwinKernels kMXFP4Twins = [] { TwinKernels t = kMXFP4Base; t.routed_thin = kRoutedThinMXFP416; t.row_chains = 2; t.ranges = TwinKernels::Ranges::both; return t; }();
    static_assert(kQ6KTwins.tile[1] == kTileQ6K8 && kIQ4NLTwins.routed[1] == kRoutedIQ4NL8 && kIQ4NLTwins.grouped[1][2] == kGroupedIQ4NL8x8 && kIQ4NLTwins.routed_tile[1] == kRoutedTileIQ4NL8 && kIQ4NLTwins.routed_narrow[1] == kRoutedStream16IQ4NL8 && kIQ4NLTwins.routed_mid[1] == kRoutedMidIQ4NL8 && kQ4KTwins.routed_narrow[1] == kRoutedStream16Q4K8 && kQ4KTwins.routed_mid[1] == kRoutedMidQ4K8 && kQ5KTwins.routed_narrow[1] == kRoutedStream16Q5K8 && kQ5KTwins.routed_mid[1] == kRoutedMidQ5K8 && kQ6KTwins.routed_narrow[1] == kRoutedStream16Q6K8 && kQ6KTwins.routed_mid[1] == kRoutedMidQ6K8 && kIQ4NLTwins.routed_stream == kRoutedStreamIQ4NL8 && kMXFP4Twins.routed_stream == kRoutedStreamMXFP48 && kMXFP4Twins.routed_narrow[1] == kRoutedStream16MXFP48 && kMXFP4Twins.routed_mid[1] == kRoutedStream32MXFP48, "a K-type's kernels follow LLMX_KTYPE's order");
    static_assert(kQ41Quarter.two[0][4] == kQuarterQ4116x16 && kQ40Quarter.grouped[0][2] == kQuarterGroupedQ4016x8 && kQ41Twins.tile[1] == kTileQ418 && kQ40Twins.routed_mid[1] == kRoutedMidQ408 && kQ40Twins.routed_narrow[1] == kRoutedStream16Q408 && kQ41Twins.routed_narrow[1] == kRoutedStream16Q418 && kQ41Twins.routed_mid[1] == kRoutedMidQ418 && kQ50Twins.routed_narrow[1] == kRoutedStream16Q508 && kQ50Twins.routed_mid[1] == kRoutedMidQ508 && kQ51Twins.routed_narrow[1] == kRoutedStream16Q518 && kQ51Twins.routed_mid[1] == kRoutedMidQ518, "Q4_0's and Q4_1's kernels follow LLMX_KTYPE's and LLMX_KQUARTER's order");
    static_assert(kQ51Quarter.two[0][4] == kQuarterQ5116x16 && kQ51Quarter.routed[0] == kQuarterRoutedQ5116 && kQ51Quarter.grouped[0][2] == kQuarterGroupedQ5116x8 && kQ51Quarter.two[1][4] == kQ51Base.two[1][4] && kQ51Quarter.tile[1] == kTileQ518, "a quarter form's kernels follow LLMX_KQUARTER's order");
    // Each weight type's kernels, one row a type for every caller (docs/ROCM.md, Owners and boundaries): its exact product, the same over several rows, its embedding, its routed exact product and its twin kernels.
    // A type without twins takes its exact product in every class.
    enum class Scaled : uint8_t { no, factor, grid };
    struct TypeKernels {
        uint32_t type;
        const char* name;
        Kernel exact, rows, embed, routed;
        const TwinKernels* twin;
        Kernel cols;   // the column builds of the exact product, the few-row build and the many-row build after it (exact_columns); kKernels where the type has none
        Kernel half = kKernels;   // the F16 class's prompt tile, which reads the activations rounded to binary16 (half_product); kKernels where the type has none
        Scaled scaled = Scaled::no;   // the scales the exact family takes after its other arguments (kernels/f32.gpu): NVFP4's factors, F8_E4M3's grid and row shift
        size_t outputs = 1;   // the outputs a wave takes in the one-row and routed exact products (kernels/f32.gpu): F8_E4M3's f8_wave takes four
        Kernel wide = kKernels;   // the prompt tile on F32 inputs and sums outside the F32 class, split by the run's extent (half_product); kKernels where the type has none
        Kernel brain = kKernels;   // the BF16 class's prompt tile, which reads the activations rounded to BF16 (half_product at --dtype bf16); kKernels where the type has none
        int step_route = -1;   // the fused step route's index of this type's exact decoder, the builds' order in kernels/f32.gpu (step_route_kernel); -1 where the route has none
    };
    static constexpr TypeKernels kTypes[] = {
        {quant::GGML_TYPE_F32, "F32", kMatmul, kRowsExactF32, kGather, kRoutedExactF32, nullptr, kColsF32, kKernels, Scaled::no, 1, kF32Tile, kKernels, 0},          {quant::GGML_TYPE_Q8_0, "Q8_0", kMatmulQ8, kRowsExactQ8, kGatherQ8, kRoutedExactQ8, &kQ8Twins, kKernels},
        {quant::GGML_TYPE_Q4_K, "Q4_K", kMatmulQ4K, kRowsExactQ4K, kGatherQ4K, kRoutedExactQ4K, &kQ4KTwins, kKernels}, {quant::GGML_TYPE_Q5_K, "Q5_K", kMatmulQ5K, kRowsExactQ5K, kGatherQ5K, kRoutedExactQ5K, &kQ5KTwins, kKernels},
        {quant::GGML_TYPE_Q6_K, "Q6_K", kMatmulQ6K, kRowsExactQ6K, kGatherQ6K, kRoutedExactQ6K, &kQ6KTwins, kKernels}, {quant::GGML_TYPE_Q5_0, "Q5_0", kMatmulQ50, kRowsExactQ50, kGatherQ50, kRoutedExactQ50, &kQ50Twins, kKernels},
        {quant::GGML_TYPE_Q5_1, "Q5_1", kMatmulQ51, kRowsExactQ51, kGatherQ51, kRoutedExactQ51, &kQ51Twins, kKernels},    {quant::GGML_TYPE_BF16, "BF16", kMatmulBF16, kRowsExactBF16, kGatherBF16, kRoutedExactBF16, nullptr, kColsBF16, kBF16Tile, Scaled::no, 1, kKernels, kBF16TileBF16, 2},
        {quant::GGML_TYPE_F16, "F16", kMatmulF16, kRowsExactF16, kGatherF16, kRoutedExactF16, nullptr, kColsF16, kF16Tile, Scaled::no, 1, kKernels, kKernels, 1},
        {quant::GGML_TYPE_Q4_0, "Q4_0", kMatmulQ40, kRowsExactQ40, kGatherQ40, kRoutedExactQ40, &kQ40Twins, kKernels},     {quant::GGML_TYPE_Q4_1, "Q4_1", kMatmulQ41, kRowsExactQ41, kGatherQ41, kRoutedExactQ41, &kQ41Twins, kKernels},
        {quant::GGML_TYPE_IQ4_NL, "IQ4_NL", kMatmulIQ4NL, kRowsExactIQ4NL, kGatherIQ4NL, kRoutedExactIQ4NL, &kIQ4NLTwins, kKernels},
        {quant::GGML_TYPE_MXFP4, "MXFP4", kMatmulMXFP4, kRowsExactMXFP4, kGatherMXFP4, kRoutedExactMXFP4, &kMXFP4Twins, kKernels},
        {quant::GGML_TYPE_NVFP4, "NVFP4", kMatmulNVFP4, kRowsExactNVFP4, kGatherNVFP4, kRoutedExactNVFP4, &kNVFP4Twins, kKernels, kKernels, Scaled::factor},
        {quant::GGML_TYPE_Q2_K, "Q2_K", kMatmulQ2K, kRowsExactQ2K, kGatherQ2K, kRoutedExactQ2K, &kQ2KTwins, kKernels},
        {quant::GGML_TYPE_Q3_K, "Q3_K", kMatmulQ3K, kRowsExactQ3K, kGatherQ3K, kRoutedExactQ3K, &kQ3KTwins, kKernels},
        {quant::GGML_TYPE_IQ4_XS, "IQ4_XS", kMatmulIQ4XS, kRowsExactIQ4XS, kGatherIQ4XS, kRoutedExactIQ4XS, &kIQ4XSTwins, kKernels},
        {quant::GGML_TYPE_IQ2_XXS, "IQ2_XXS", kMatmulIQ2XXS, kRowsExactIQ2XXS, kGatherIQ2XXS, kRoutedExactIQ2XXS, &kIQ2XXSTwins, kKernels},
        {quant::GGML_TYPE_IQ2_XS, "IQ2_XS", kMatmulIQ2XS, kRowsExactIQ2XS, kGatherIQ2XS, kRoutedExactIQ2XS, &kIQ2XSTwins, kKernels},
        {quant::GGML_TYPE_IQ2_S, "IQ2_S", kMatmulIQ2S, kRowsExactIQ2S, kGatherIQ2S, kRoutedExactIQ2S, &kIQ2STwins, kKernels},
        {quant::GGML_TYPE_IQ3_XXS, "IQ3_XXS", kMatmulIQ3XXS, kRowsExactIQ3XXS, kGatherIQ3XXS, kRoutedExactIQ3XXS, &kIQ3XXSTwins, kKernels},
        {quant::GGML_TYPE_IQ3_S, "IQ3_S", kMatmulIQ3S, kRowsExactIQ3S, kGatherIQ3S, kRoutedExactIQ3S, &kIQ3STwins, kKernels},
        {quant::GGML_TYPE_IQ1_S, "IQ1_S", kMatmulIQ1S, kRowsExactIQ1S, kGatherIQ1S, kRoutedExactIQ1S, &kIQ1STwins, kKernels},
        {quant::GGML_TYPE_IQ1_M, "IQ1_M", kMatmulIQ1M, kRowsExactIQ1M, kGatherIQ1M, kRoutedExactIQ1M, &kIQ1MTwins, kKernels},
        {quant::GGML_TYPE_TQ1_0, "TQ1_0", kMatmulTQ10, kRowsExactTQ10, kGatherTQ10, kRoutedExactTQ10, nullptr, kKernels},
        {quant::GGML_TYPE_TQ2_0, "TQ2_0", kMatmulTQ20, kRowsExactTQ20, kGatherTQ20, kRoutedExactTQ20, nullptr, kKernels},
        {quant::GGML_TYPE_Q1_0, "Q1_0", kMatmulQ10, kRowsExactQ10, kGatherQ10, kRoutedExactQ10, &kQ10Twins, kKernels},
        {quant::GGML_TYPE_Q2_0, "Q2_0", kMatmulQ20, kRowsExactQ20, kGatherQ20, kRoutedExactQ20, &kQ20Twins, kKernels},
        {quant::GGML_TYPE_F8_E4M3, "F8_E4M3", kMatmulF8, kRowsExactF8, kGatherF8, kRoutedExactF8, nullptr, kKernels, kF8Tile, Scaled::grid, 4},
    };
    static_assert(kQ8Twins.routed_stream != kKernels && kQ4KTwins.routed_stream != kKernels && kQ5KTwins.routed_stream != kKernels && kQ6KTwins.routed_stream != kKernels && kQ50Twins.routed_stream != kKernels && kQ40Twins.routed_stream != kKernels && kQ41Twins.routed_stream != kKernels && kQ51Twins.routed_stream != kKernels && kIQ4NLTwins.routed_stream != kKernels && kMXFP4Twins.routed_stream != kKernels, "every twin type has its 8-bit routed stream, since its 8-bit routed tiles leave it the groups of up to kStreamRows entries");
    static_assert(kColsF32Many < kRows16x1 && kColsBF16Many < kRows16x1 && kColsF16Many < kRows16x1 && kAttentionChunkWindow < kRows16x1 && kAttentionVecWindow < kRows16x1 && kAttentionVec256Window < kRows16x1 && kUnitRows < kRows16x1 && kSigmoidGateRows < kRows16x1 && kMatmulAtIds < kRows16x1 && kChainPick < kRows16x1, "the column builds, the window builds and the unit_rows, sigmoid_gate_rows, matmul_at_ids and chain_pick kernels are in kernels/f32.gpu, which the constructor loads for the kernels below kRows16x1");
    static const TypeKernels* type_kernels(uint32_t type) {
        for (const TypeKernels& t : kTypes)
            if (t.type == type) return &t;
        return nullptr;
    }
    static constexpr unsigned kChunks = 16;   // a row's attention chunks at most (kernels/f32.gpu)
    static constexpr unsigned kNormThreads = 1024;   // the held norms' block (kernels/f32.gpu)
    static constexpr size_t kQueryMixHeld = 9 * kThreads;     // the widest row llmx_query_mix_norm holds (kernels/f32.gpu, kQueryMixSteps)
    static constexpr size_t kQueryMixNormItems = 6;          // the most items it mixes
    static constexpr size_t kQueryMixFew = 4;                // the most items a call's llmx_query_mix4 mixes (kernels/f32.gpu)
    static constexpr size_t kQueryMixRowsItems = 5;          // the most items the row form's builds mix (kernels/f32.gpu, llmx_query_mix_rows1 to 5)
    static constexpr size_t kNormPart = 256;        // the values a held norm's part writes, where the measured gain levels off
    static constexpr size_t kColumns = 16;   // the row kernel's columns (kernels/int16.gpu)
    static constexpr size_t kRowsOut = 8, kRowsIn = 8;   // the blocked exact product's outputs and rows a block (kernels/f32.gpu)
    static constexpr size_t kTileOut = 64, kTileOutTall = 128, kTileRows = 64, kTileRowsTall = 128;   // the prompt tile's outputs, its tall build's, its rows and the K-quants' square build's (kernels/int16.gpu)
    static constexpr unsigned kStreamLanes = 64;   // a routed stream block's lanes, an output each, so its grid is the tile's (kernels/int16.gpu, routed_stream)

    // Memory in slots, pinned host memory the device reads in place or copies from, or device memory: a slot is taken again once the ticket that last read it retires.
    struct Ring {
        std::array<BufferPtr, kRingSlots> slots{};
        std::array<Ticket, kRingSlots> tags{};
        std::array<bool, kRingSlots> dirty{};   // read by work launched since the last submission
        std::array<size_t, kRingSlots> laps{};   // times each slot was taken again, so what it held before is known gone
        size_t slot = 0, used = 0;
        void* take(GpuBackend<Api>& b, size_t bytes) {
            const size_t cap = slots[0]->size();
            if (bytes > cap) throw std::logic_error(Api::prefix + "a ring take wider than a slot");
            bytes = (bytes + 255) & ~size_t(255);
            if (used + bytes > cap) {
                // The next slot waits for the ticket that last covered its readers, and a slot read since the last submission has none yet, so only then does a wrap submit.
                // A stage never fills every slot, so a wrap is not a flush point (docs/src/backends-gpu.md, records).
                slot = (slot + 1) % kRingSlots;
                if (dirty[slot]) b.submit();
                b.wait(tags[slot]);
                ++laps[slot];
                used = 0;
            }
            void* p = (uint8_t*)as_gpu<Api>(*slots[slot]).data() + used;
            used += bytes;
            dirty[slot] = true;
            return p;
        }
        void tag(Ticket t) {
            for (size_t i = 0; i < kRingSlots; ++i)
                if (dirty[i]) tags[i] = t;
            dirty = {};
        }
    };
    // A pass's values in device memory (pass_arg): their bytes, and where they sit while their slot's lap lasts.
    struct Upload {
        std::vector<uint8_t> bytes;
        const void* at = nullptr;
        size_t slot = 0, lap = 0;
    };
    static constexpr size_t kUploads = 8;   // sets kept for a later layer to find: attention's, the KV write's and the positions, of a pass and the one before

    // A 32-bit kernel argument, refused past it.
    static unsigned u32(size_t v) {
        if (v > UINT32_MAX) throw std::runtime_error(Api::prefix + "a value past 32 bits");
        return (unsigned)v;
    }
    // An indexer call's views on one storage, checked as the CPU and Vulkan check them before any launch: their blocks in device memory, each view's first block and first row, and the group positions.
    struct IndexCall {
        GpuKVStorage<Api>* s = nullptr;
        float* aux = nullptr;
        const float *w = nullptr, *cs = nullptr, *sn = nullptr;
        const int* blocks = nullptr;
        const unsigned* gpos = nullptr;
        const unsigned char* of = nullptr;
        unsigned streams = 0;
        std::vector<size_t> first, row0, gpos_at;
        size_t rows = 0, longest = 0, max_rows = 0, rope_values = 0;
        unsigned long long block_floats = 0;
        unsigned rows_floats = 0;
    };
    IndexCall index_call(size_t layer, const KVView* views, size_t n_views, const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin, const GroupPositions& positions,
                         bool selecting) {
        if (!views) throw std::runtime_error(Api::prefix + "index op without views");
        if (!index.dim || index.dim > kThreads || !index.rope_dim || index.rope_dim % 2 || index.rope_dim > index.dim || !index.compression || !std::isfinite(index.eps) ||
            index.eps <= 0 || (selecting && (!index.heads || !index.budget || index.budget % index.compression)))
            throw std::runtime_error(Api::prefix + "invalid index dimensions");
        if (positions.pos && (!positions.streams.count || (positions.streams.count > 1 && !positions.streams.of)))
            throw std::runtime_error(Api::prefix + "invalid group position streams");
        IndexCall c;
        c.s = &storage(views[0]);
        std::vector<int32_t> table;
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != c.s) throw std::runtime_error(Api::prefix + "index op over views of different storages");
            const size_t used = c.s->check_view(views[vi], layer, false), visible = size_add(views[vi].length, views[vi].nq);
            u32(visible);
            c.first.push_back(table.size());
            c.row0.push_back(c.rows);
            c.gpos_at.push_back(vi ? c.gpos_at.back() + ((views[vi - 1].length + views[vi - 1].nq) / index.compression - views[vi - 1].length / index.compression) * positions.streams.count : 0);
            table.insert(table.end(), views[vi].blocks, views[vi].blocks + used);
            c.rows = size_add(c.rows, views[vi].nq);
            c.longest = std::max(c.longest, visible);
            c.max_rows = std::max(c.max_rows, views[vi].nq);
        }
        if (c.s->auxiliary_width(layer) != index.dim || c.s->auxiliary_pool(layer) != index.compression)
            throw std::runtime_error(Api::prefix + "index op differs from the rows and groups its auxiliary blocks keep");
        const auto& auxiliary = c.s->auxiliary_buffer(layer);
        if (!auxiliary) throw std::runtime_error(Api::prefix + "index op without auxiliary key storage");
        c.rope_values = size_mul(c.longest, index.rope_dim / 2);
        for (const auto& [sl, count] : {std::pair<CSlice, size_t>{key_norm, index.dim}, {cos, c.rope_values}, {sin, c.rope_values}}) {
            if (!sl.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
            span(*sl.buffer, sl.offset * sizeof(float), size_mul(count, sizeof(float)));
        }
        c.aux = f32(Slice{auxiliary.get(), 0});
        c.w = f32(key_norm);
        c.cs = f32(cos);
        c.sn = f32(sin);
        c.block_floats = c.s->auxiliary_block_bytes(layer) / sizeof(float);
        c.rows_floats = u32(c.s->auxiliary_rows_bytes(layer) / sizeof(float));
        c.blocks = (const int*)pass_arg(table.data(), table.size() * sizeof(int32_t));
        if (positions.pos) {
            const size_t n = c.gpos_at.back() + ((views[n_views - 1].length + views[n_views - 1].nq) / index.compression - views[n_views - 1].length / index.compression) *
                                                    positions.streams.count;
            c.streams = u32(positions.streams.count);
            c.gpos = (const unsigned*)pass_arg(positions.pos, n * sizeof(uint32_t));
            if (c.streams > 1) c.of = (const unsigned char*)pass_arg(positions.streams.of, index.rope_dim / 2);
        }
        return c;
    }
    // An operand of `count` floats, inside its allocation.
    static void floats(const CSlice& s, size_t count) {
        if (!s.buffer) throw std::runtime_error(Api::prefix + "operand without storage");
        span(*s.buffer, s.offset * sizeof(float), size_mul(count, sizeof(float)));
    }
    void topk_select(Slice ids, CSlice scores, CSlice lens, size_t rows, size_t n, size_t stride, size_t picks, uint32_t offset, unsigned mode, size_t block_size, size_t items) {
        const size_t sn = size_mul(rows, n), idn = size_mul(rows, stride);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{ids, idn}, {scores, sn}, {lens, rows}}) floats(s, count);
        if (overlaps(ids, idn, scores, sn) || overlaps(ids, idn, lens, rows)) throw std::runtime_error(Api::prefix + "selection output must be disjoint from its inputs");
        unsigned* o = (unsigned*)f32(ids);
        const float* s = f32(scores);
        const unsigned* l = (const unsigned*)f32(lens);
        unsigned nn = u32(n), st = u32(stride), pk = u32(picks), of = offset, md = mode, bs = u32(block_size), it = u32(items);
        // A long row of mode 0 runs as chunks a block and a merge over their picks, the same ids (kernels/f32.gpu, modes 2 and 3).
        const size_t chunk = dev_.profile.topk_chunk, need = mode == 0 ? topk_work_bytes(rows, items, picks) : 0;
        if (need) {
            const size_t chunks = items / chunk, cands = chunks * picks;
            // The fit prices this buffer (scratch_sizes, slot 16); only a selection wider than its widths grows it here.
            if (!topk_work_ || topk_work_->size() < need) topk_work_ = alloc(need, Memory::device);
            unsigned* w = (unsigned*)as_gpu<Api>(*topk_work_).data();
            unsigned two = 2, three = 3, ch = u32(chunk), cn = u32(cands);
            launch(kTopkRows, u32(rows), u32(chunks), Args(w, s, l, nn, pk, pk, of, two, bs, it, (const unsigned*)nullptr, ch));
            launch(kTopkRows, u32(rows), 1, Args(o, s, l, nn, st, pk, of, three, bs, cn, (const unsigned*)w, ch));
            return;
        }
        const Args args(o, s, l, nn, st, pk, of, md, bs, it, (const unsigned*)nullptr, 0u);
        launch(kTopkRows, u32(rows), 1, args);
    }
    // The pooled rows a call's rows read (kernels/f32.gpu, pooled_at): for each row the offset of its view's block ids in the table and its position plus one, then the ids.
    struct PooledTable {
        const float* aux = nullptr;
        const unsigned* rows = nullptr;
        unsigned tokens = 1, width = 0, rows_floats = 0;
        unsigned long long block_floats = 0;
    };
    // With `tokens` the table reads the layer's token rows, which are dim wide, in place of its pooled rows.
    PooledTable pooled_table(size_t layer, const KVView* views, size_t n_views, size_t rows, size_t dim, const char* op, bool tokens) {
        if (!views) throw std::runtime_error(Api::prefix + op + " without views");
        GpuKVStorage<Api>& s = storage(views[0]);
        std::vector<unsigned> tab(size_mul(rows, 2)), blocks;
        size_t r = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error(Api::prefix + op + " over views of different storages");
            const size_t used = s.check_view(views[vi], layer, false);
            const unsigned first = u32(size_add(tab.size(), blocks.size()));
            blocks.insert(blocks.end(), views[vi].blocks, views[vi].blocks + used);
            for (size_t i = 0; i < views[vi].nq; ++i, ++r) {
                if (r == rows) throw std::runtime_error(Api::prefix + op + " views do not cover its rows");
                tab[2 * r] = first;
                tab[2 * r + 1] = u32(size_add(views[vi].length, i + 1));
            }
        }
        if (r != rows) throw std::runtime_error(Api::prefix + op + " views do not cover its rows");
        if (tokens ? s.auxiliary_width(layer) != dim : !s.auxiliary_pool(layer) || s.auxiliary_pool_width(layer) < dim)
            throw std::runtime_error(Api::prefix + op + (tokens ? " reads auxiliary rows its layer does not keep at their width" : " reads pooled rows its layer does not keep"));
        if (!s.auxiliary_buffer(layer)) throw std::runtime_error(Api::prefix + op + " reads an auxiliary plane its layer does not keep");
        tab.insert(tab.end(), blocks.begin(), blocks.end());
        PooledTable t;
        t.aux = f32(CSlice{s.auxiliary_buffer(layer).get(), 0});
        t.rows = (const unsigned*)pass_arg(tab.data(), tab.size() * sizeof(unsigned));
        t.tokens = tokens ? 1u : u32(s.auxiliary_pool(layer));
        t.width = u32(tokens ? dim : s.auxiliary_pool_width(layer));
        t.rows_floats = tokens ? 0u : u32(s.auxiliary_rows_bytes(layer) / sizeof(float));
        t.block_floats = s.auxiliary_block_bytes(layer) / sizeof(float);
        return t;
    }
    // The groups view vi's rows complete, pooled into their rows of the auxiliary blocks (kernels/f32.gpu, llmx_kv_index_pool).
    void pool_view(const IndexCall& c, const KVView& v, size_t vi, const IndexArgs& index) {
        const size_t done = v.length / index.compression, count = (v.length + v.nq) / index.compression;
        if (count <= done) return;
        float* aux = c.aux;
        const float *w = c.w, *cs = c.cs, *sn = c.sn;
        const int* blocks = c.blocks + c.first[vi];
        const unsigned* gpos = c.gpos ? c.gpos + c.gpos_at[vi] : nullptr;
        const unsigned char* of = c.of;
        unsigned first = u32(done), base = u32(done), dim = u32(index.dim), half = u32(index.rope_dim / 2), compression = u32(index.compression), bt = (unsigned)kBlockTokens;
        unsigned rows_floats = c.rows_floats, streams = c.streams;
        unsigned long long block_floats = c.block_floats;
        float eps = index.eps;
        const Args args(aux, w, cs, sn, blocks, gpos, of, first, base, dim, half, compression, bt, block_floats, rows_floats, eps, streams);
        launch(kKvIndexPool, u32(count - done), 1, args);
    }
    static constexpr size_t kSelectRows = 16, kSelectSlice = 1024, kSelectRowWords = 5122;   // a selection's rows at most and the wide form's slice and state a row (kernels/f32.gpu)

    // Each scratch buffer's bytes for a call, the one formula its grow site and scratch_bytes share (docs/src/backends-gpu.md, Scratch).
    static size_t attention_parts_bytes(size_t rows, size_t heads, size_t parts, size_t dim) {
        return size_mul(size_mul(size_mul(rows, heads), parts), size_mul(dim + 2, sizeof(float)));
    }
    // The per-row kernel's splits of a call whose longest row is `longest` positions, within the profile's limit for the head width.
    size_t attention_splits(size_t dim, size_t longest) const {
        const DeviceProfile& p = dev_.profile;
        const size_t chunk = std::max<size_t>(1, p.attention_split_chunk), most = dim == 256 ? p.attention_split_max_wide : p.attention_split_max;
        return std::min(most, std::max<size_t>(1, (longest + chunk - 1) / chunk));
    }
    // A sparse row's slots a split, doubled from the profile's until the splits fit a block.
    size_t sparse_split_ids(size_t slots) const {
        size_t ids_per = std::max<size_t>(1, dev_.profile.sparse_split_ids);
        while ((slots + ids_per - 1) / ids_per > kThreads) ids_per *= 2;
        return ids_per;
    }
    // The sink's scratch for a chunk of rows of at most 32 MiB: the scores by kept row and head, each tile's maxima, the kept rows and their two counts (kernels/f32.gpu).
    struct SinkLayout {
        size_t chunk, den_at, kept_at, counts_at, need;
    };
    static SinkLayout sink_layout(size_t rows, size_t named, size_t heads) {
        // A row's maxima a head of each tile of kSparseFewWide kept rows, the narrower pair's tile, since a call's last chunk may take that pair.
        const size_t dens = size_mul(heads, std::max<size_t>(1, (named + kSparseFewWide - 1) / kSparseFewWide));
        const size_t row_bytes = size_add(size_mul(size_add(size_mul(named, heads), dens), 4), size_mul(size_add(named, 1), 8));
        const size_t chunk = std::min(rows, std::max<size_t>(1, ((size_t)32 << 20) / row_bytes));
        const size_t pt_bytes = size_mul(size_mul(chunk, named), heads * 4), den_bytes = size_mul(chunk, dens * 4), kept_bytes = size_mul(size_mul(chunk, named), 8);
        const size_t den_at = pt_bytes, kept_at = (den_at + den_bytes + 7) / 8 * 8, counts_at = kept_at + kept_bytes;
        return {chunk, den_at, kept_at, counts_at, counts_at + size_mul(chunk, 8)};
    }
    // The hyper-connection mix's scratch: two barrier counters, each stream's root, the twin's scales and the twin, for a launch's columns.
    struct HcLayout {
        size_t cols, rr_at, nd_at, nq_at, need;
    };
    static HcLayout hc_layout(size_t rows, size_t streams, size_t width) {
        const size_t cols = std::min(rows, kHcColumns), rr_at = 64, nd_at = rr_at + (cols * streams * sizeof(float) + 15) / 16 * 16,
                     nq_at = nd_at + (cols * width / 32 * sizeof(float) + 15) / 16 * 16;
        return {cols, rr_at, nd_at, nq_at, nq_at + cols * width};
    }
    static size_t moe_out_bytes(size_t entries, size_t width) { return size_mul(size_mul(entries, width), sizeof(float)); }
    // The fused moe_step's shared act twin: `rows` ff-value rows of int8, then their block scales from the half.
    static size_t step_twin_bytes(size_t rows, size_t ff) { return 2 * std::max<size_t>(2048, size_mul(rows, ff)); }
    static size_t moe_tab_bytes(size_t groups, size_t entries) { return size_mul(size_add(size_mul(4, groups), entries), sizeof(unsigned)); }
    static size_t tile_partials_bytes(size_t parts, size_t rows, size_t outputs) { return size_mul(size_mul(size_mul(parts, rows), outputs), sizeof(float)); }
    // The indexer's scores of 16 rows at most, bounded by the call's longest view and never by its rows.
    static size_t select_score_bytes(size_t rows, size_t longest, size_t compression) {
        return size_mul(size_mul(std::min(rows, kSelectRows), longest / std::max<size_t>(1, compression)), sizeof(float));
    }
    static size_t select_state_bytes(size_t slices) { return size_mul(size_add(kSelectRows * kSelectRowWords, size_mul(size_mul(kSelectRows, slices), 2)), sizeof(uint32_t)); }
    size_t argmax_parts_bytes(size_t rows, size_t n) const {
        const size_t slice = std::max<size_t>(1, std::min(dev_.profile.argmax_slice, n));
        return size_mul(size_mul(rows, (n + slice - 1) / slice), 2 * sizeof(float));
    }
    // A twin of `blocks` blocks of 32 values: its codes, then a scale a block, then a block's two half sums (TwinPlanes).
    static size_t twin_bytes(size_t blocks, bool int8) { return size_add(size_add(size_mul(blocks, int8 ? 32 : 64), size_mul(blocks, sizeof(float))), size_mul(blocks, int8 ? 4 : 8)); }
    // Where a twin of `total` blocks lies from `base`, each plane indexed by twin block (TwinMap): the codes, the scales, and each block's two half sums as kernels/quant.h's twin_sums writes them.
    struct TwinPlanes {
        uint8_t* q;
        float* d;
        void* s;
    };
    static TwinPlanes twin_planes(uint8_t* base, size_t total, bool int8) {
        float* d = (float*)(base + size_mul(total, int8 ? 32 : 64));
        return {base, d, d + total};
    }
    // The scratch buffers in one order, and what passes of widths `w` grow each to.
    static constexpr size_t kScratchBuffers = 17;
    using ScratchSizes = std::array<size_t, kScratchBuffers>;
    BufferPtr* scratch_buffer(size_t i) {
        BufferPtr* const held[kScratchBuffers] = {&parts_, &sparse_, &moe_out_, &step_counter_, &step_twin_, &moe_tab_, &delta_counter_, &staged_,
                                                  &partials_, &scores_, &argmax_parts_, &select_state_, &hc_scratch_, &twin_[0], &twin_[1], &expert_ids_, &topk_work_};
        return held[i];
    }
    // The candidates a mode-0 selection of `rows` rows keeps between its chunks and their merge (topk_select), 0 where it runs a block a row.
    size_t topk_work_bytes(size_t rows, size_t items, size_t picks) const {
        const size_t chunk = dev_.profile.topk_chunk, chunks = chunk && picks && picks <= chunk ? items / chunk : 0;
        return chunks >= 2 ? size_mul(size_mul(rows, size_mul(chunks, picks)), sizeof(unsigned)) : 0;
    }
    ScratchSizes scratch_sizes(const PassWidths& w) const {
        ScratchSizes z{};
        const size_t R = w.rows, H = w.heads, D = w.head_dim;
        const bool vec = D == 128 || D == 256;
        if (!R) return z;
        size_t parts = 0;
        if (w.dense_attention && H) {
            parts = attention_parts_bytes(R, H, kChunks, D);
            if (vec) parts = std::max(parts, attention_parts_bytes(R, H, attention_splits(D, w.context), D));
        }
        if (w.sparse_ids && H) {
            const size_t ids_per = sparse_split_ids(w.sparse_ids);
            parts = std::max(parts, attention_parts_bytes(R, H, vec ? (w.sparse_ids + ids_per - 1) / ids_per : kChunks, D));
        }
        if (w.relative_attention && H) {
            const size_t most = gpu_shape::rel_parts(u32(std::min<size_t>(w.context, UINT32_MAX)));
            if (most > 1) parts = std::max(parts, attention_parts_bytes(std::min(R, rel_slice_rows(H, most, D)), H, most, D));
        }
        z[0] = parts;
        if (w.sparse_named && H) z[1] = sink_layout(R, w.sparse_named, H).need;
        const size_t entries = size_mul(R, w.routed);
        if (entries) {
            z[2] = moe_out_bytes(entries, w.expert_out);
            // The grouped builds take groups of 2 entries at the least, and every entry may be the one of its expert.
            z[5] = moe_tab_bytes(size_add((entries + 1) / 2, entries), entries);
            if (w.expert_cache) z[15] = size_mul(entries, sizeof(unsigned));
        }
        if (w.shared_ff && dot_module_) {
            z[3] = 64;
            z[4] = step_twin_bytes(R, w.shared_ff);
        }
        z[6] = kDeltaCounters * sizeof(unsigned);
        z[7] = size_mul(size_mul(w.logit_rows, w.vocab), sizeof(float));
        z[8] = tile_partials_bytes(1, 1, split_partials_most(dev_.profile, dev_.caps, R, w.product_in, w.product_out, kTileOut));
        if (w.index_compression) {
            const size_t scored = w.context / w.index_compression;
            z[9] = select_score_bytes(R, w.context, w.index_compression);
            if (scored >= dev_.profile.select_wide_from) z[11] = select_state_bytes((scored + kSelectSlice - 1) / kSelectSlice);
        }
        if (w.logit_rows && w.vocab > std::max<size_t>(1, dev_.profile.argmax_slice)) z[10] = argmax_parts_bytes(w.logit_rows, w.vocab);
        if (w.streams) z[12] = hc_layout(R, w.streams, w.mix_width).need;
        const size_t topk_items = w.topk_compression ? (w.context + w.topk_compression - 1) / w.topk_compression : w.topk_items;
        z[16] = topk_work_bytes(R, topk_items, w.topk_picks);
        // A twin slot holds a product's rows, a prompt's padded to an odd multiple of 4 in each of the map's regions (twin_map), or a routed call's entries, at the 16-bit class's bytes.
        const size_t pad = TwinMap::kRegions * 7;
        z[13] = z[14] = std::max(twin_bytes(size_mul(size_add(R, pad), w.product_in / 32), false), twin_bytes(size_mul(size_add(entries, pad), w.expert_in / 32), false));
        return z;
    }
    // The bytes hipMalloc takes for `bytes`, at the units probe_memory read: whole words, then its small pages up to kLargeFrom and its large pages past it.
    static constexpr size_t kLargeFrom = (size_t)1 << 20, kFallbackUnit = (size_t)2 << 20;
    size_t alloc_bytes(size_t bytes) const {
        if (!bytes) return 0;
        const size_t words = (bytes + 3) & ~size_t(3), unit = words > kLargeFrom ? large_unit_ : small_unit_;
        return size_mul((words + unit - 1) / unit, unit);
    }
    // The allocation units from free-memory readings around fresh allocations, and the runtime's private scratch from the kernels' private bytes a lane, read once at construction.
    void probe_memory() {
        const auto taken = [&](size_t bytes, size_t n) -> size_t {
            size_t before = 0, after = 0, total = 0, made = 0;
            std::vector<void*> p(n, nullptr);
            if (Api::mem_info(&before, &total) != Api::kSuccess) return 0;
            while (made < n && Api::malloc(&p[made], bytes) == Api::kSuccess) ++made;
            const bool read = Api::mem_info(&after, &total) == Api::kSuccess;
            for (size_t i = 0; i < made; ++i) free_memory<Api>(p[i], false);
            return read && made == n && before > after ? (before - after) / n : 0;
        };
        small_unit_ = allocation_unit(4, taken(4, 64));
        large_unit_ = allocation_unit(kLargeFrom + 4, taken(kLargeFrom + 4, 4));
        int most = 0;
        for (typename Api::Function k : kernels_) {
            int v = 0;
            if (k && Api::function_local_bytes(&v, k) == Api::kSuccess) most = std::max(most, v);
        }
        driver_scratch_ = most > 0 ? alloc_bytes(size_mul(size_mul((size_t)most, dev_.lanes_resident), std::max(1u, dev_.caps.compute_units))) : 0;
    }

    // Where a write's rows land: each row's KV block and token slot, read by the kernels from the argument ring.
    struct KvTargets {
        GpuKVStorage<Api>* s = nullptr;
        size_t rows = 0;
        void* kc = nullptr;
        void* vc = nullptr;
        const int* blocks = nullptr;
        const int* slots = nullptr;
        int k_half = 0, v_half = 0;
    };
    KvTargets kv_targets(size_t layer, const KVView* views, size_t n_views) {
        KvTargets t;
        GpuKVStorage<Api>& s = storage(views[0]);
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error(Api::prefix + "KV views of different storages in one write");
            s.check_view(views[vi], layer, true);
            t.rows = size_add(t.rows, views[vi].nq);
        }
        t.s = &s;
        if (!t.rows) return t;
        // Each row's block, then each row's slot.
        std::vector<int32_t> targets(2 * t.rows);
        size_t r = 0;
        for (size_t vi = 0; vi < n_views; ++vi)
            for (size_t b = 0; b < views[vi].nq; ++b, ++r) {
                const size_t at = views[vi].length + b;
                targets[r] = views[vi].blocks[at / kBlockTokens];
                targets[t.rows + r] = (int32_t)(at % kBlockTokens);
            }
        t.kc = as_gpu<Api>(*s.k_buffer(layer)).data();
        t.vc = as_gpu<Api>(*s.v_buffer(layer)).data();
        t.blocks = (const int*)pass_arg(targets.data(), targets.size() * sizeof(int32_t));
        t.slots = t.blocks + t.rows;
        t.k_half = s.k_type() == KVType::f16;
        t.v_half = s.v_type() == KVType::f16;
        return t;
    }
    // Host values a kernel reads, copied into the argument ring.
    const void* arg(const void* src, size_t bytes) {
        void* p = args_.take(*this, std::max<size_t>(bytes, 1));
        std::memcpy(p, src, bytes);
        return p;
    }
    // Values every layer of a pass reads alike, copied once into device memory on the stream and found again by their bytes, so kernels do not read them over PCIe (docs/src/backends-gpu.md).
    const void* pass_arg(const void* src, size_t bytes) {
        bytes = std::max<size_t>(bytes, 1);
        for (Upload& u : uploads_)
            if (u.at && u.lap == pass_.laps[u.slot] && u.bytes.size() == bytes && !std::memcmp(u.bytes.data(), src, bytes)) {
                pass_.dirty[u.slot] = true;
                return u.at;
            }
        void* d = pass_.take(*this, bytes);
        hold_copy(d, arg(src, bytes), bytes, Api::kToDevice, "upload");
        Upload& u = uploads_[next_upload_++ % kUploads];
        u.bytes.assign((const uint8_t*)src, (const uint8_t*)src + bytes);
        u.at = d;
        u.slot = pass_.slot;
        u.lap = pass_.laps[pass_.slot];
        return d;
    }
    static float* f32(Slice s) { return (float*)as_gpu<Api>(*s.buffer).data() + s.offset; }
    static const float* f32(CSlice s) { return (const float*)as_gpu<Api>(*s.buffer).data() + s.offset; }
    // A weight's scales on the device, null without them (check_weight admits them).
    static const float* scales(const WeightView& w) { return w.scales.buffer ? f32(w.scales) : nullptr; }
    static const uint8_t* bytes(CSlice s) { return (const uint8_t*)as_gpu<Api>(*s.buffer).data() + s.offset * sizeof(float); }
    // The weights the products and the embedding read: the types of kTypes, decoded exactly into F32 sums, in whole blocks.
    // A caller passing `experts` takes a scaled type's scales as the CPU's weight_scales admits them, one or one an expert, or a grid at row shift 0 or 7 (check_grid counts it); every other caller refuses scales.
    static const TypeKernels& check_weight(const WeightView& w, size_t nin, const char* what, size_t experts = 0) {
        const TypeKernels* t = type_kernels(w.type);
        if (!t) throw std::runtime_error(std::string(Api::prefix + "an ") + what + " of type " + std::to_string(w.type) + " is not implemented");
        if (!w.scales.buffer && (w.scale_count || w.scales.offset)) throw std::runtime_error(Api::prefix + "a weight scale count without storage");
        if (!w.scales.buffer && t->scaled == Scaled::grid) throw std::runtime_error(std::string(Api::prefix + "an ") + t->name + " " + what + " needs its scale grid");
        if (w.scales.buffer) {
            if (t->scaled == Scaled::no || !experts) throw std::runtime_error(std::string(Api::prefix + "a scaled ") + what + " is not implemented");
            if (t->scaled == Scaled::factor && w.scale_count != 1 && w.scale_count != experts) throw std::runtime_error(Api::prefix + "weight scales neither one nor one an expert");
            if (t->scaled == Scaled::grid && ((w.scale_shift != 0 && w.scale_shift != 7) || !w.scale_count || w.scale_count % experts))
                throw std::runtime_error(Api::prefix + "a scale grid of another row shift or count");
            span(*w.scales.buffer, w.scales.offset * sizeof(float), size_mul(w.scale_count, sizeof(float)));
        }
        if (!w.data.buffer) throw std::runtime_error(std::string(Api::prefix + "an ") + what + " without storage");
        if (nin % quant::storage_type(w.type)->block_size) throw std::runtime_error(std::string(Api::prefix + "a ") + t->name + " " + what + " row ends inside a block");
        return *t;
    }
    // A grid type's scales against `experts` matrices of `nout` rows, one scale per 128 inputs of every 2^shift rows, as the CPU's fp8_scales counts them.
    static void check_grid(const WeightView& w, size_t nin, size_t nout, size_t experts) {
        if (type_kernels(w.type)->scaled == Scaled::grid && w.scale_count != gguf::fp8_grid_count(nin, nout, w.scale_shift, experts))
            throw std::runtime_error(Api::prefix + "a scale grid that does not match its matrices");
    }
    // The scale arguments a scaled type's exact family takes after its others: a factor's pointer, or a grid's pointer and row shift; routed calls add the step between experts' scales.
    static void put_scales(Args& args, const TypeKernels& kinds, const WeightView& w, size_t experts = 0) {
        if (kinds.scaled == Scaled::no) return;
        args.put(scales(w));
        if (experts) args.put((unsigned long long)(kinds.scaled == Scaled::grid ? w.scale_count / experts : w.scale_count == 1 ? 0 : 1));
        if (kinds.scaled == Scaled::grid) args.put((unsigned long long)w.scale_shift);
    }
    GpuKVStorage<Api>& storage(const KVView& v) {
        auto* s = dynamic_cast<GpuKVStorage<Api>*>(v.storage);
        if (!s) throw std::runtime_error(Api::prefix + "a KV view of another backend");
        return *s;
    }
    // A product of a type with twins outside the F32 class reads the activations' twin; every other product reads the activations as they are.
    static bool integer(const WeightView& weight, Dtype dtype) {
        const TypeKernels* t = type_kernels(weight.type);
        // A factor an expert stays on the exact family: the twins take one factor a projection.
        return t && t->twin && (dtype == Dtype::f16 || dtype == Dtype::int8) && !(t->scaled == Scaled::factor && weight.scale_count > 1);
    }
    // A routed call's projections share one kernel, so it takes the twins only when each of them would.
    static bool integer(const Projection* proj, size_t nproj, Dtype dtype) {
        for (size_t p = 0; p < nproj; ++p)
            if (!integer(proj[p].weight, dtype)) return false;
        return true;
    }
    // A cap above 0 softcaps the outputs after the product, a host output in device memory before it is copied over (matmul_logits).
    void product(const WeightView& weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, bool accumulate, RowRuns runs, Dtype dtype,
                 bool twin, float cap = 0.0f) {
        const TypeKernels& kinds = check_weight(weight, nin, "matrix", 1);
        check_grid(weight, nin, nout, 1);
        if (!nin || !nout || !nbatch) return;
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nout));
        span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nin, nbatch), sizeof(float)));
        span(*Y.buffer, Y.offset * sizeof(float), size_mul(size_mul(nout, nbatch), sizeof(float)));
        // Kernels write a product a float a row, which over PCIe into pinned host memory took the 0.6B's head 3.3 ms of a token: a host output is formed in device memory and copied over whole.
        if (as_gpu<Api>(*Y.buffer).host_ptr()) {
            const size_t bytes_out = size_mul(size_mul(nout, nbatch), sizeof(float));
            if (!staged_ || staged_->size() < bytes_out) staged_ = alloc(bytes_out, Memory::device);
            if (accumulate) copy(*staged_, 0, *Y.buffer, Y.offset * sizeof(float), bytes_out);
            product(weight, X, {staged_.get(), 0}, nin, nout, nbatch, accumulate, runs, dtype, twin, cap);
            copy(*Y.buffer, Y.offset * sizeof(float), *staged_, 0, bytes_out);
            return;
        }
        if (cap != 0.0f) {
            product(weight, X, Y, nin, nout, nbatch, accumulate, runs, dtype, twin);
            return softcap(Y, size_mul(nout, nbatch), cap);
        }
        if (integer(weight, dtype)) {
            const Proj one[1] = {{bytes(weight.data), f32(Y), nout, scales(weight)}};
            return product_twin(*kinds.twin, one, 1, X, nin, nbatch, accumulate, runs, twin, dtype == Dtype::int8, false);
        }
        const uint8_t* w = bytes(weight.data);
        const float* x = f32(X);
        float* y = f32(Y);
        unsigned long long ni = nin, no = nout;
        int add = accumulate;
        const Kernel tile = prompt_tile(kinds, weight, w, x, nin, nbatch, dtype);
        if (tile != kKernels) return half_product(kinds, tile, weight, w, x, y, nin, nout, nbatch, add, runs);
        if (on_columns(kinds, nbatch, nin)) {
            const Proj one[1] = {{w, y, nout}};
            exact_columns(kinds, one, 1, x, nin, nbatch, add);
        } else if (nbatch > 1) {
            // Several rows take the blocked build, each output the bits of its one-row product (kernels/f32.gpu, exact_rows).
            unsigned long long nr = nbatch;
            Args args(w, x, y, ni, no, nr, add);
            put_scales(args, kinds, weight);
            launch_writing({{y, size_mul(nout, nbatch)}}, kinds.rows, (unsigned)((nout + kRowsOut - 1) / kRowsOut), (unsigned)((nbatch + kRowsIn - 1) / kRowsIn), args);
        } else {
            // One row takes a wave an output (kernels/f32.gpu, exact_wave).
            Args args(w, x, y, ni, no, add);
            put_scales(args, kinds, weight);
            const size_t waves = kThreads / dev_.caps.subgroup_size;
            const size_t per = waves * kinds.outputs;
            launch_writing({{y, nout}}, kinds.exact, (unsigned)((nout + per - 1) / per), 1, args);
        }
        // Every product here reads F32 activations, whatever dtype was asked for.
        record_matrix_path(MatrixPath::f32);
    }
    // The tile a call's prompt rows take outside the F32 class, on whole steps of 32 values (half_product), or kKernels where they take none.
    // That is the BF16 tile at bf16, the half tile at f16 and int8, else the type's wide tile on F32 inputs whose weight asks for it (WeightView::wide).
    Kernel prompt_tile(const TypeKernels& kinds, const WeightView& weight, const uint8_t* w, const float* x, size_t nin, size_t nbatch, Dtype dtype) const {
        if (nbatch < 2 || dtype == Dtype::f32 || nin % 32 || ((uintptr_t)w | (uintptr_t)x) % 16) return kKernels;
        const Kernel tile = dtype == Dtype::bf16 ? kinds.brain : kinds.half;
        if (tile != kKernels && kernels_[tile]) return tile;
        return weight.wide && kinds.wide != kKernels && kernels_[kinds.wide] ? kinds.wide : kKernels;
    }
    // Whether rows of an exact product take the type's column builds: from two up to the profile's exact_cols_upto, past which a call, a prompt's among them, takes the 8-by-8 row block; both give the same bits.
    // A row's units stay below 2^31, which the builds count in 32 bits (kernels/f32.gpu, exact_cols).
    bool on_columns(const TypeKernels& kinds, size_t rows, size_t nin) const { return kinds.cols != kKernels && dev_.profile.exact_on_columns(rows) && nin < (size_t(1) << 31); }
    // Whether every row of an exact call takes the column builds, as product and half_product choose: decode columns past any tile, or rows of a call that takes none.
    bool all_columns(const TypeKernels& kinds, const WeightView& weight, const float* x, size_t nin, size_t nbatch, RowRuns runs, Dtype dtype) const {
        return on_columns(kinds, nbatch, nin) && (generated_rows(nbatch, runs) || prompt_tile(kinds, weight, bytes(weight.data), x, nin, nbatch, dtype) == kKernels);
    }
    // A call's prompt runs on the type's half tile, which rounds the activations to binary16 and sums in F32, or its wide tile on F32 inputs, and its decode columns on the exact product, so a column keeps its bits in any call.
    // The wide tile splits the inner dimension by the run's extent, as the twins' tile does (tile), so adjacent runs share a launch only where their splits agree.
    void half_product(const TypeKernels& kinds, Kernel tile, const WeightView& weight, const uint8_t* w, const float* x, float* y, size_t nin, size_t nout, size_t nbatch, int add, RowRuns runs) {
        const RowRun whole[1] = {{nbatch, 2}};
        const RowRuns all = runs.n ? runs : RowRuns{whole, 1};
        unsigned long long ni = nin, no = nout;
        bool tiled = false, exact = false, wide = false;
        size_t begin = 0;
        for (size_t r = 0; r < all.n; ++r) {
            size_t end = all.runs[r].end;
            if (end <= begin || end > nbatch) throw std::runtime_error(Api::prefix + "row runs out of order or past the call");
            const bool column = all.runs[r].extent == 1;
            const size_t tiles = split_tiles_of(all.runs[r].extent);
            const auto same = [&](const RowRun& n) { return (n.extent == 1) == column && (column || kinds.wide == kKernels || split_tiles_of(n.extent) == tiles); };
            while (r + 1 < all.n && same(all.runs[r + 1]) && all.runs[r + 1].end > end && all.runs[r + 1].end <= nbatch) end = all.runs[++r].end;
            const size_t rows = end - begin;
            float* yr = y + begin * nout;
            if (column && on_columns(kinds, rows, nin)) {
                const Proj one[1] = {{w, yr, nout}};
                exact_columns(kinds, one, 1, x + begin * nin, nin, rows, add);
                exact = true;
            } else if (column) {
                unsigned long long nr = rows;
                Args args(w, x + begin * nin, yr, ni, no, nr, add);
                put_scales(args, kinds, weight);
                launch_writing({{yr, size_mul(nout, rows)}}, kinds.rows, (unsigned)((nout + kRowsOut - 1) / kRowsOut), (unsigned)((rows + kRowsIn - 1) / kRowsIn), args);
                exact = true;
            } else if (tile == kinds.wide) {
                TileCall c{};
                const size_t nblk = nin / 32, gx = (nout + kTileOut - 1) / kTileOut;
                const size_t kper = split_blocks(dev_.profile, dev_.caps, gx * tiles, nblk, dev_.profile.float_tile_split_per_cu);
                c.w[0] = w, c.y[0] = yr, c.nout[0] = nout, c.nproj = 1, c.nin = nin, c.nrows = rows, c.add = add, c.kper = (unsigned)kper, c.parts = (unsigned)((nblk + kper - 1) / kper);
                if (c.parts > 1) {
                    const size_t need = tile_partials_bytes(c.parts, rows, nout);
                    if (!partials_ || partials_->size() < need) partials_ = alloc(need, Memory::device);
                    c.partials = (float*)as_gpu<Api>(*partials_).data();
                }
                const Args args(c, x + begin * nin);
                launch_writing({{yr, size_mul(nout, rows)}}, kinds.wide, (unsigned)gx, (unsigned)((rows + kTileRows - 1) / kTileRows), args, c.parts);
                if (c.parts > 1) {
                    const size_t g = (nout * rows + kThreads - 1) / kThreads;
                    const Args rargs(c, (unsigned)g);
                    launch_writing({{yr, size_mul(nout, rows)}}, kTileReduce, (unsigned)std::min(g, std::max<size_t>(1, resident(reduce_per_cu_, kTileReduce))), 1, rargs);
                }
                wide = true;
            } else {
                TileCall c{};
                c.w[0] = w, c.y[0] = yr, c.nout[0] = nout, c.nproj = 1, c.nin = nin, c.nrows = rows, c.add = add, c.kper = (unsigned)(nin / 32), c.parts = 1;
                const float* xr = x + begin * nin;
                Args args(c, xr);
                put_scales(args, kinds, weight);
                launch_writing({{yr, size_mul(nout, rows)}}, tile, (unsigned)((nout + kTileOut - 1) / kTileOut), (unsigned)((rows + kTileRows - 1) / kTileRows), args);
                tiled = true;
            }
            begin = end;
        }
        if (begin != nbatch) throw std::runtime_error(Api::prefix + "row runs short of the call");
        // The tensor-core tile sums each group of 8 inside the instruction, so it is witnessed apart from the F16 class's IEEE sums and held to its own budget (docs/CUDA.md, BF16 prompt tile).
        if (tiled) record_matrix_path(tile == kBF16TileBF16 ? MatrixPath::bf16_mma : tile == kBF16Tile ? MatrixPath::f16_mma : MatrixPath::f16);
        if (exact || wide) record_matrix_path(MatrixPath::f32);
    }
    // A projection of a call: its weight rows, its output and their count.
    struct Proj {
        const uint8_t* w;
        float* y;
        size_t nout;
        const float* f = nullptr;   // the weight's F32 factor, NVFP4's
    };
    // The host's copy of the grouped one-row F32 product's call (kernels/f32.gpu, ExactGroup), field for field.
    struct ExactGroup {
        const uint8_t* w[6];
        float* y[6];
        const float* x[6];
        unsigned long long nout[6];
    };
    static_assert(sizeof(ExactGroup) == 192 && offsetof(ExactGroup, x) == 96 && offsetof(ExactGroup, nout) == 144, "ExactGroup is the kernel's layout");
    // The host's copy of the normed products' items (kernels/f32.gpu, NormItems), field for field, and the widest row its shared memory holds.
    struct NormItems {
        const unsigned char* W[3];
        float* Y[3];
        const float *src0[3], *src1[3], *w;
        unsigned groups, dim, split;
        float eps;
    };
    static_assert(sizeof(NormItems) == 120 && offsetof(NormItems, eps) == 116, "NormItems is the kernel's layout");
    static constexpr size_t kNormWidth = 8192;
    static constexpr size_t kNormHeldSteps = 9;   // kernels/f32.gpu, llmx_matmul_normed_f32's held forms
    // The host's copy of the tile kernel's call (kernels/int16.gpu, TileCall), field for field.
    static constexpr size_t kTileProj = 6;   // kernels/int16.gpu, kTileProj
    struct TileCall {
        const uint8_t* w[kTileProj];
        float* y[kTileProj];
        unsigned long long nout[kTileProj];
        unsigned start[kTileProj];
        unsigned nproj;
        unsigned long long nin, nrows;
        int add;
        unsigned kper, parts;
        float* partials;
        unsigned xrow, xblock;
        const void* xs;
        const float* f[kTileProj];
    };
    static_assert(sizeof(TileCall) == 280 && offsetof(TileCall, nin) == 176 && offsetof(TileCall, partials) == 208 && offsetof(TileCall, xblock) == 220 && offsetof(TileCall, xs) == 224 && offsetof(TileCall, f) == 232,
                  "TileCall is the kernel's layout");
    // The host's copy of a column call (kernels/f32.gpu, ColsCall), field for field.
    struct ColsCall {
        const uint8_t* w[3];
        float* y[3];
        unsigned long long nout[3];
        unsigned start[4];
    };
    static_assert(sizeof(ColsCall) == 88 && offsetof(ColsCall, nout) == 48 && offsetof(ColsCall, start) == 72, "ColsCall is the kernel's layout");
    // The column builds' shapes (kernels/f32.gpu, LLMX_COLS), the few-row build's then the many-row build's: outputs and rows a wave, and waves a block; wave32's, then wave64's.
    struct ColsShape {
        unsigned outputs, rows, waves;
    };
    static constexpr ColsShape kColsShapes[2][2] = {{{2, 2, 4}, {4, 2, 8}}, {{2, 4, 4}, {1, 8, 4}}};
    // Rows of X on a type's column builds for up to three projections of one input in one launch, each output the bits of its one-row product (kernels/f32.gpu, exact_cols).
    // The few-row build takes calls up to its block's rows, the many-row build the rest; a block's waves split the rows by the fewest powers of two that hold them, the others taking more outputs.
    void exact_columns(const TypeKernels& kinds, const Proj* proj, size_t n, const float* x, size_t nin, size_t nbatch, int add) {
        const unsigned wave = dev_.caps.subgroup_size;
        const ColsShape* shapes = kColsShapes[wave == 32 ? 0 : 1];
        const bool many = nbatch > size_t(shapes[0].rows) * shapes[0].waves;
        const ColsShape& c = shapes[many];
        unsigned ls = 0;
        while ((1u << ls) < c.waves && (size_t(c.rows) << ls) < nbatch) ++ls;
        const size_t per = size_t(c.waves >> ls) * c.outputs, block_rows = size_t(c.rows) << ls;
        ColsCall call{};
        for (size_t m = 0; m < n; ++m) {
            call.w[m] = proj[m].w, call.y[m] = proj[m].y, call.nout[m] = proj[m].nout;
            call.start[m + 1] = call.start[m] + (unsigned)((proj[m].nout + per - 1) / per);
        }
        for (size_t m = n; m < 3; ++m) call.start[m + 1] = call.start[n];
        unsigned long long ni = nin, nr = nbatch;
        const Args args(call, x, ni, nr, ls, add);
        const Proj& p1 = proj[n > 1 ? 1 : 0];
        const Proj& p2 = proj[n > 2 ? 2 : 0];
        launch_writing({{proj[0].y, size_mul(proj[0].nout, nbatch)}, {p1.y, size_mul(p1.nout, nbatch)}, {p2.y, size_mul(p2.nout, nbatch)}}, static_cast<Kernel>(kinds.cols + (many ? 1 : 0)),
                       call.start[3], (unsigned)((nbatch + block_rows - 1) / block_rows), args, 1, c.waves * wave);
    }
    // The host's copy of a routed tile call's groups (kernels/int16.gpu, TileGroups), field for field.
    struct TileGroups {
        const unsigned* tab;
        unsigned long long order0, per, first;
        unsigned long long stride[3];
    };
    static_assert(sizeof(TileGroups) == 56 && offsetof(TileGroups, stride) == 32, "TileGroups is the kernel's layout");
    // The host's copy of the twin's layout (kernels/f32.gpu, TwinMap), field for field.
    struct TwinMap {
        static constexpr unsigned kRegions = 8;
        unsigned regions, blocks, shift, magic_lo, magic_hi;
        unsigned first[kRegions], base[kRegions], pad[kRegions];
        unsigned total;
        bool flat() const { return regions == 1 && !pad[0]; }
        // Two maps of every row row after row lie alike however their values divide into rows.
        bool operator==(const TwinMap& o) const { return flat() && o.flat() ? total == o.total : std::memcmp(this, &o, sizeof(TwinMap)) == 0; }
        // Where row b's block 0 lies, and the blocks from one row to the next and from one block to the next.
        void locate(size_t b, size_t& at, unsigned& row, unsigned& block) const {
            unsigned g = 0;
            for (unsigned i = 1; i < regions; ++i) g = b >= first[i] ? i : g;
            row = pad[g] ? 1 : blocks;
            block = pad[g] ? pad[g] : 1;
            at = base[g] + (b - first[g]) * row;
        }
    };
    static_assert(sizeof(TwinMap) == 120 && offsetof(TwinMap, pad) == 84 && offsetof(TwinMap, total) == 116, "TwinMap is the kernels' layout");
    // The twin's layout from each run's kernel: rows on the row builds (generated rows and short prompts) row after row, a tile prompt's rows block-major and padded to an odd multiple of 4 (docs/src/backends-gpu.md).
    // It moves where a value lies, never the value; without a prompt's run, or past the map's regions, every row lies row after row.
    TwinMap twin_map(RowRuns runs, size_t rows, size_t nin) const {
        TwinMap m{};
        m.blocks = (unsigned)(nin / 32);
        size_t total = 0, begin = 0;
        unsigned n = 0;
        bool major = false;
        for (size_t r = 0; r < runs.n && begin < rows; ++r) {
            const size_t end = std::min(runs.runs[r].end, rows);
            const bool tile = !on_rows(runs.runs[r].extent);
            if (end <= begin) continue;
            if (n && (m.pad[n - 1] != 0) == tile) {
                if (tile) total -= size_t(m.pad[n - 1]) * m.blocks;
            } else {
                if (n == TwinMap::kRegions) {
                    major = false;
                    break;
                }
                m.first[n] = (unsigned)begin;
                m.base[n] = (unsigned)total;
                ++n;
            }
            const size_t height = end - m.first[n - 1];
            m.pad[n - 1] = tile ? (unsigned)((((height + 3) >> 2) | 1) << 2) : 0;
            total += tile ? size_t(m.pad[n - 1]) * m.blocks : (end - begin) * m.blocks;
            major = major || tile;
            begin = end;
        }
        m.regions = n;
        if (!major || begin != rows) {
            m = TwinMap{};
            m.regions = 1;
            m.blocks = (unsigned)(nin / 32);
            total = rows * (nin / 32);
        }
        if (total >= 0x80000000ull || rows * (nin / 32) >= 0x80000000ull) throw std::runtime_error(Api::prefix + "a twin of 2^31 blocks or more");
        m.total = (unsigned)total;
        // A block's row from its number, by a multiply and a shift: with 2^(L-1) < blocks <= 2^L, magic = ceil(2^(32 + L) / blocks) is exact for every block number below 2^32, and a number below 2^31 times it stays below 2^64.
        if (!m.flat() && m.blocks) {
            unsigned l = 0;
            while ((1ull << l) < m.blocks) ++l;
            m.shift = 32 + l;
            const unsigned long long magic = ((1ull << m.shift) + m.blocks - 1) / m.blocks;
            m.magic_lo = (unsigned)magic;
            m.magic_hi = (unsigned)(magic >> 32);
        }
        return m;
    }
    // The host's copy of the row kernel's call (kernels/int16.gpu, RowCall), field for field.
    struct RowCall {
        const uint8_t* w[3];
        float* y[3];
        unsigned long long nout[3];
        unsigned long long start[2];   // projections 1 and 2; projection 0 starts at row 0
        unsigned long long total;
        unsigned nproj;
        unsigned cluster;
        const float* f[3];
    };
    // 128 bytes at most: at 136 ptxas held llmx_q6_k_rows16_8 to 128 registers on sm_86 and spilled (docs/src/backends-gpu.md).
    static_assert(sizeof(RowCall) == 128 && offsetof(RowCall, start) == 72 && offsetof(RowCall, nproj) == 96 && offsetof(RowCall, cluster) == 100 && offsetof(RowCall, f) == 104, "RowCall is the kernel's layout");
    // The host's copy of a list launch's calls (kernels/int16.gpu, RowList): call c from block start[c - 1], an unused call starting at the grid's end.
    static constexpr size_t kListCalls = 4;
    struct RowList {
        RowCall c[kListCalls];
        const void* q[kListCalls];
        const float* dx[kListCalls];
        unsigned start[kListCalls - 1];
    };
    static_assert(sizeof(RowList) == 592 && offsetof(RowList, q) == 512 && offsetof(RowList, dx) == 544 && offsetof(RowList, start) == 576, "RowList is the kernel's layout");
    // The host's copy of the convolution a one-column row kernel's store runs (kernels/int16.gpu, ConvCols): its weights, the view's ring, the output row and the view's length.
    struct ConvStore {
        const float* w;
        float* ring;
        float* out;
        unsigned C, T, R, len;
        float divisor;
        int accumulate;
    };
    static_assert(sizeof(ConvStore) == 48 && offsetof(ConvStore, C) == 24 && offsetof(ConvStore, divisor) == 40, "ConvStore is the kernel's layout");
    // The convolution conv_in_store hands the next product, and whether a launch ran it.
    const ConvStore *conv_store_ = nullptr, *combine_store_ = nullptr;
    bool conv_stored_ = false;
    // The lanes a row takes in the row kernel: a wave for Q8_0, else the power of two covering the row's lane units (kernels/int16.gpu, KRaw's Lane), at most the wave.
    // Q4_K and Q5_K take at most the profile's k45_row_lanes, as the Vulkan row plan has it; the width and type alone decide, so a row keeps its bits in every build.
    unsigned row_cluster(const TwinKernels& tk, size_t nin) const {
        const unsigned wave = dev_.caps.subgroup_size;
        if (!tk.row_lanes) return wave;
        const size_t units = nin / tk.block_values * tk.row_lanes;
        unsigned cl = 1;
        while (cl < wave && cl < units) cl *= 2;
        if (tk.k45) cl = std::min(cl, std::max(8u, (unsigned)dev_.profile.k45_row_lanes));
        // A lane holding several units takes their share of the lanes: MXFP4's row covers at least four units, so its cluster is at least two lanes.
        return cl / tk.row_chains;
    }
    // A row's lane form: in the F16 class the quarter form where its lane units fill the cluster's steps fuller than one lane a block, else one lane (docs/src/backends-gpu.md).
    // The int8 class takes one lane a block at every width, measured faster there; like row_cluster the form reads the row's width, type and class alone, so a dense row and a routed entry keep one order.
    const TwinKernels& row_form(const TwinKernels& tk, size_t nin, bool int8) const {
        if (!tk.quarter || int8) return tk;
        const auto fill = [&](const TwinKernels& f, size_t& span) {
            const size_t units = nin / f.block_values * f.row_lanes, cl = row_cluster(f, nin);
            span = (units + cl - 1) / cl * cl;
            return units;
        };
        size_t one_span, quarter_span;
        const size_t one_units = fill(tk, one_span), quarter_units = fill(*tk.quarter, quarter_span);
        return quarter_units * one_span > one_units * quarter_span ? *tk.quarter : tk;
    }
    // The host's copies of the routed calls (kernels/int16.gpu, RoutedCall, and kernels/f32.gpu, RoutedExact), field for field.
    struct RoutedCall {
        RowCall c;
        unsigned long long stride[3];
        const unsigned* ids;
        unsigned long long first, per, n_expert;
    };
    static_assert(sizeof(RoutedCall) == 184 && offsetof(RoutedCall, ids) == 152 && offsetof(RoutedCall, n_expert) == 176, "RoutedCall is the kernel's layout");
    // The host's copy of llmx_l2_clip's argument (kernels/f32.gpu, ClipArgs), field for field.
    struct ClipItem {
        float* x;
        const float* scale;
        const float* row_scale;
        unsigned long long stride;
        unsigned dim, heads;
        float threshold;
        unsigned pad;
    };
    struct ClipArgs {
        ClipItem item[kClipItems];
        unsigned n;
    };
    static_assert(sizeof(ClipItem) == 48 && offsetof(ClipItem, threshold) == 40 && offsetof(ClipArgs, n) == 48 * kClipItems, "ClipArgs is the kernel's layout");
    // The host's copy of kernels/f32.gpu's ExpertStacks, field for field: each stack's ring entries and slots, and its expert's 16-byte units.
    struct ExpertStacks {
        const void* ring[3];
        void* dst[3];
        unsigned long long stride[3];
    };
    static_assert(sizeof(ExpertStacks) == 72 && offsetof(ExpertStacks, dst) == 24 && offsetof(ExpertStacks, stride) == 48, "kernels/f32.gpu's ExpertStacks");
    // The exchange words the kernels and the reader share (ExpertRing): a half's 64-bit ticket first, so hold_wait reads it in place.
    static_assert(ExpertRing::kAlign % 2 == 0 && ExpertRing::kPairs % 2 == 0 && ExpertRing::kTicket == 0 && ExpertRing::kPulls == 2 && ExpertRing::kTag == 3,
                  "kernels/f32.gpu's llmx_expert_wave and llmx_expert_pull exchange words");
    // The host's copies of the fused moe_step's calls (kernels/f32.gpu, MfRoute, and kernels/int16.gpu, MfGateUpS and MfDown), field for field.
    // The host's copy of llmx_delta_step's argument (kernels/f32.gpu, DeltaStepArgs), field for field.
    struct DeltaStepArgs {
        float* o;
        const float* x;
        const float* cw;
        const float* mixed;
        const uint8_t* wa;
        const uint8_t* wb;
        float* ao;
        float* bo;
        const float* a;
        const float* dt_bias;
        float* st;
        const unsigned* tab;
        const float* z;
        const float* nw;
        unsigned* counter;
        unsigned C, k_heads, v_heads, E;
        unsigned long long slot, carried;
        float scale, eps, neps;
        void* tq;
        float* td;
        void* ts;
        int tlevel, major;
        TwinMap tm;
        unsigned kd, vd;
        const int8_t* xq;
        const float* xd;
        unsigned form, sig;
    };
    static_assert(offsetof(DeltaStepArgs, slot) == 136 && offsetof(DeltaStepArgs, tq) == 168 && offsetof(DeltaStepArgs, tm) == 200 && offsetof(DeltaStepArgs, kd) == 320 &&
                      offsetof(DeltaStepArgs, xq) == 328 && offsetof(DeltaStepArgs, form) == 344 && offsetof(DeltaStepArgs, sig) == 348 && sizeof(DeltaStepArgs) == 352,
                  "DeltaStepArgs is the kernel's layout");
    struct StepRoute {
        const uint8_t* w;
        const float* x;
        float* scores;
        unsigned long long nin, nout;
        const uint8_t* gw;
        float* rg;
        uint8_t* hq;
        float* hd;
        unsigned long long hblocks;
        unsigned* ids;
        float* weights;
        unsigned n_expert, k;
        int normalize;
        unsigned* counter;
        unsigned rows;
    };
    static_assert(offsetof(StepRoute, rows) == 120 && offsetof(StepRoute, gw) == 40 && offsetof(StepRoute, ids) == 80 && offsetof(StepRoute, n_expert) == 96 && offsetof(StepRoute, counter) == 112 && sizeof(StepRoute) == 128,
                  "StepRoute is the kernel's layout");
    struct StepUp {
        RoutedCall rc;
        const uint8_t* q;
        const float* dx;
        unsigned long long nin;
        float* act;
        uint8_t* aq;
        float* ad;
        RowCall sc;
        const uint8_t* sq;
        const float* sdx;
        unsigned long long snin;
        float* sact;
        const float* rg;
        uint8_t* stq;
        float* std_;
        unsigned long long rows;
    };
    static_assert(offsetof(StepUp, q) == 184 && offsetof(StepUp, act) == 208 && offsetof(StepUp, sc) == 232 && offsetof(StepUp, sq) == 360 && offsetof(StepUp, sact) == 384 &&
                      offsetof(StepUp, std_) == 408 && sizeof(StepUp) == 424,
                  "StepUp is the kernel's layout");
    struct StepDown {
        RoutedCall rc;
        const uint8_t* aq;
        const float* ad;
        unsigned long long nin;
        const float* weights;
        float* y;
        unsigned k;
        RowCall sc;
        const uint8_t* sq;
        const float* sdx;
        unsigned long long snin;
        const float* y0;
        unsigned long long y0s;
    };
    static_assert(offsetof(StepDown, aq) == 184 && offsetof(StepDown, weights) == 208 && offsetof(StepDown, k) == 224 && offsetof(StepDown, sc) == 232 && offsetof(StepDown, sq) == 360 &&
                      offsetof(StepDown, y0) == 384 && sizeof(StepDown) == 400,
                  "StepDown is the kernel's layout");
    struct RoutedExact {
        const unsigned* ids;
        unsigned long long first, per, n_expert, expert_bytes;
    };
    static_assert(sizeof(RoutedExact) == 40 && offsetof(RoutedExact, expert_bytes) == 32, "RoutedExact is the kernel's layout");
    // A call's runs must be in order and end at its rows, as matmul's are, though the routed products read no extent.
    static void check_runs(size_t rows, RowRuns runs) {
        if (runs.n) for_each_run(rows, runs, [](const RowRun&) { return 0; }, [](size_t, size_t, int) {});
    }
    // A routed call's `nrows` token rows, k entries each, in stretches by kernel: prompt rows from the role's `tile_from` on the routed tile, the rest on the row builds.
    // X holds a row a token, or for the down a row an entry.
    void routed_runs(const Projection* proj, size_t nproj, CSlice X, size_t nin, size_t nrows, const Routing& routing, bool down, RowRuns runs, Dtype dtype) {
        const size_t k = routing.k, per = down ? 1 : k;
        const auto stretch = [&](size_t first, size_t count, bool tile) {
            Projection at[3];
            for (size_t p = 0; p < nproj; ++p) at[p] = proj[p], at[p].out = {proj[p].out.buffer, proj[p].out.offset + first * k * proj[p].rows};
            const Routing r{{routing.ids.buffer, routing.ids.offset + first * k}, routing.weights, k, routing.n_expert};
            const size_t xfirst = per == 1 ? first * k : first, xrows = per == 1 ? count * k : count;
            routed(at, nproj, {X.buffer, X.offset + xfirst * nin}, nin, xrows, count * k, r, per, dtype, tile);
        };
        if (!runs.n) return stretch(0, nrows, false);
        const TypeKernels* kinds = type_kernels(proj[0].weight.type);
        const size_t from = kinds && kinds->twin && integer(proj, nproj, dtype) ? kinds->twin->tile_from[down][dtype == Dtype::int8] : SIZE_MAX;
        for_each_run(nrows, runs, [from](const RowRun& r) { return r.extent >= from; }, stretch);
    }
    // A routed projection's first row's byte offset in its expert's matrix and the matrices' stride, a window of `rows` rows from `first` in matrices of `pitch` rows when given.
    std::pair<size_t, size_t> window(const Projection& p, size_t nin) const {
        const uint32_t t = p.weight.type;
        if (!p.first && !p.pitch) return {0, quant::row_bytes(t, nin, p.rows)};
        if (t == quant::GGML_TYPE_F8_E4M3 || p.weight.scales.buffer || p.first > p.pitch || p.rows > p.pitch - p.first)
            throw std::runtime_error(Api::prefix + "an expert row window outside its matrix, or over an F8_E4M3 grid");
        return {quant::row_bytes(t, nin, p.first), quant::row_bytes(t, nin, p.pitch)};
    }
    // The routed row builds' call over the stacks of `proj`, `per` entries a twin row, from entry 0.
    RoutedCall routed_call(const Projection* proj, size_t nproj, size_t nin, const TwinKernels& rk, const unsigned* ids, size_t per, size_t n_expert) const {
        RoutedCall c{};
        for (size_t p = 0; p < nproj; ++p) {
            c.c.w[p] = bytes(proj[p].weight.data) + window(proj[p], nin).first;
            c.c.y[p] = f32(proj[p].out);
            c.c.nout[p] = proj[p].rows;
            c.c.f[p] = scales(proj[p].weight);
            if (p) c.c.start[p - 1] = c.c.total;
            c.c.total += proj[p].rows;
            c.stride[p] = window(proj[p], nin).second;
        }
        c.c.nproj = (unsigned)nproj;
        c.c.cluster = row_cluster(rk, nin);
        c.ids = ids;
        c.per = per;
        c.n_expert = n_expert;
        return c;
    }
    void routed(const Projection* proj, size_t nproj, CSlice X, size_t nin, size_t xrows, size_t entries, const Routing& routing, size_t per, Dtype dtype, bool tile) {
        const uint32_t type = proj[0].weight.type;
        const TypeKernels& kinds = check_weight(proj[0].weight, nin, "expert matrix", routing.n_expert);
        span(*routing.ids.buffer, routing.ids.offset * sizeof(float), size_mul(entries, sizeof(float)));
        for (size_t p = 0; p < nproj; ++p) {
            if (proj[p].weight.type != type) throw std::runtime_error(Api::prefix + "routed projections of different types");
            check_weight(proj[p].weight, nin, "expert matrix", routing.n_expert);
            check_grid(proj[p].weight, nin, proj[p].rows, routing.n_expert);
            const size_t pitch = proj[p].pitch ? proj[p].pitch : proj[p].rows;
            span(*proj[p].weight.data.buffer, proj[p].weight.data.offset * sizeof(float), quant::row_bytes(type, nin, size_mul(routing.n_expert, pitch)));
            span(*proj[p].out.buffer, proj[p].out.offset * sizeof(float), size_mul(size_mul(entries, proj[p].rows), sizeof(float)));
            if (as_gpu<Api>(*proj[p].out.buffer).host_ptr()) throw std::runtime_error(Api::prefix + "a routed product into host memory");
        }
        const unsigned* ids = (const unsigned*)f32(routing.ids);
        constexpr size_t kGridY = 65535, kGroupExperts = 1024;   // HIP's grid height; the experts llmx_moe_group counts in shared memory
        unsigned long long ni = nin;
        if (integer(proj, nproj, dtype)) {
            const bool int8 = dtype == Dtype::int8;
            const int form = !int8 ? 0 : nin / 32 % 2 ? 1 : 2;
            const TwinKernels& rk = row_form(*kinds.twin, nin, int8);
            half_span(form, xrows, nin);
            auto [q, d, xs] = twin_of(X, xrows, nin, int8, twin_map({}, xrows, nin), true, tile && rk.tile_sums);
            RoutedCall c = routed_call(proj, nproj, nin, rk, ids, per, routing.n_expert);
            const size_t clusters = dev_.caps.subgroup_size / c.c.cluster;
            // A half-block call of at most one entry an expert takes the one-column build an entry, which needs no grouping, where that build fills its steps: one under two spans of blocks, kSteps from four (kernels/int16.gpu).
            const size_t blocks = nin / 32, sub = dev_.caps.subgroup_size;
            const bool one_entry = rk.routed[form] == kRoutedHalf8 && entries <= routing.n_expert && (blocks < sub || blocks >= 2 * sub);
            if (tile && routing.n_expert <= kGroupExperts) {
                // A prompt's entries in groups of up to 64 of one expert, each group a block row of the routed tile over its expert's rows (kernels/int16.gpu, TileGroups); the tile never splits, so each entry's sums are its own.
                const unsigned chunk = kTileRows, groups = (unsigned)((entries + chunk - 1) / chunk + std::min(routing.n_expert, entries));
                const size_t tab_bytes = moe_tab_bytes(groups, entries);
                if (!moe_tab_ || moe_tab_->size() < tab_bytes) moe_tab_ = alloc(tab_bytes, Memory::device);
                unsigned* tab = (unsigned*)as_gpu<Api>(*moe_tab_).data();
                unsigned ne = (unsigned)entries, nx = (unsigned)routing.n_expert, ng = groups, ch = chunk;
                const Args gargs(ids, tab, ne, nx, ng, ch);
                launch_writing({{tab, tab_bytes / sizeof(float)}}, kMoeGroup, nx, 1, gargs);
                const size_t nblk = nin / 32;
                if (nblk * kTileOut * 34 + 64 > 0xffffffffull || (xrows * nblk) * 32 * (int8 ? 1 : 2) > 0xffffffffull)
                    throw std::runtime_error(Api::prefix + "a routed tile's rows are too wide for its 32-bit offsets");
                TileCall t{};
                TileGroups tg{tab, 4ull * groups, per, 0, {}};
                size_t gx = 0;
                for (size_t p = 0; p < nproj; ++p) {
                    t.w[p] = c.c.w[p];
                    t.y[p] = c.c.y[p];
                    t.nout[p] = proj[p].rows;
                    t.f[p] = c.c.f[p];
                    t.start[p] = (unsigned)gx;
                    gx += (proj[p].rows + kTileOut - 1) / kTileOut;
                    tg.stride[p] = c.stride[p];
                }
                t.nproj = (unsigned)nproj;
                t.nin = nin;
                t.kper = (unsigned)nblk;
                t.parts = 1;
                t.xrow = (unsigned)nblk;
                t.xblock = 1;
                t.xs = xs;
                // A group runs in one build by its entries: thin or stream up to 8 (Q8_0's to 16), narrow up to 16, a K-type's mid up to 32, the 64-row tile past; a type's 8-bit narrow, or narrow and mid, builds are streams.
                // The other builds' workgroups return at once; the thin build takes four output tiles a workgroup, a thread an output.
                const Kernel thin = int8 ? kKernels : kinds.twin->routed_thin;
                using Ranges = typename TwinKernels::Ranges;
                const unsigned narrow = int8 && kinds.twin->ranges != Ranges::tiles ? kStreamLanes : kThreads, mid = int8 && kinds.twin->ranges == Ranges::both ? kStreamLanes : kThreads;
                const std::pair<Kernel, unsigned> builds[5] = {{kinds.twin->routed_tile[int8], kThreads}, {kinds.twin->routed_mid[int8], mid}, {kinds.twin->routed_narrow[int8], narrow},
                                                               {thin, kThreads}, {int8 ? kinds.twin->routed_stream : kKernels, kStreamLanes}};
                for (const auto& [kernel, threads] : builds)
                    for (size_t first = 0; kernel != kKernels && first < groups && (route_only_.empty() || std::strstr(kKernelNames[kernel], route_only_.c_str())); first += kGridY) {
                        tg.first = first;
                        const Args args(t, tg, q, d);
                        launch(kernel, (unsigned)(kernel == thin ? (gx + 3) / 4 : gx), (unsigned)std::min<size_t>(kGridY, groups - first), args, 1, threads);
                    }
            } else if (rk.groups(entries, routing.n_expert, int8) && routing.n_expert <= kGroupExperts && !one_entry) {
                // One expert's entries a group of the narrowest grouped build that holds their average, up to 8, each column the bits of its one-column product.
                const size_t touched = std::min(routing.n_expert, entries), average = (entries + touched - 1) / touched;
                const int width = average <= 2 ? 0 : average <= 4 ? 1 : 2;
                unsigned chunk = 2u << width, groups = (unsigned)((entries + chunk - 1) / chunk + touched);
                const size_t tab_bytes = moe_tab_bytes(groups, entries);
                if (!moe_tab_ || moe_tab_->size() < tab_bytes) moe_tab_ = alloc(tab_bytes, Memory::device);
                unsigned* tab = (unsigned*)as_gpu<Api>(*moe_tab_).data();
                unsigned ne = (unsigned)entries, nx = (unsigned)routing.n_expert;
                const Args gargs(ids, tab, ne, nx, groups, chunk);
                launch_writing({{tab, tab_bytes / sizeof(float)}}, kMoeGroup, nx, 1, gargs);
                unsigned long long order0 = 4ull * groups;
                const size_t wave_rows = 2 * rows_per_block_ * clusters;
                for (size_t first = 0; first < groups; first += kGridY) {
                    c.first = first;
                    const Args args(c, tab, order0, q, d, ni);
                    launch(rk.grouped[form][width], (unsigned)((c.c.total + wave_rows - 1) / wave_rows), (unsigned)std::min<size_t>(kGridY, groups - first), args);
                }
            } else {
                // Calls below the grouped builds' break-even take the one-column build an entry, as do one_entry's, the half-block build taking the dense one-column build's steps by row width (product_twin).
                Kernel kernel = rk.routed[form] == kRoutedHalf8 && blocks < sub ? kRoutedHalf8s1 : rk.routed[form];
                const size_t wave_rows = block_rows(kernel, rk.two_rows[form][0], c.c.cluster, ni, size_mul(c.c.total, entries));
                for (size_t first = 0; first < entries; first += kGridY) {
                    c.first = first;
                    const Args args(c, q, d, ni);
                    launch(kernel, (unsigned)((c.c.total + wave_rows - 1) / wave_rows), (unsigned)std::min(kGridY, entries - first), args);
                }
            }
            record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
            for (size_t p = 0; p < nproj; ++p) forget_written(c.c.y[p], proj[p].rows * entries);
            return;
        }
        const float* x = f32(X);
        for (size_t p = 0; p < nproj; ++p) {
            const auto [offset, stride] = window(proj[p], nin);
            const uint8_t* w = bytes(proj[p].weight.data) + offset;
            float* y = f32(proj[p].out);
            unsigned long long no = proj[p].rows;
            RoutedExact r{ids, 0, per, routing.n_expert, stride};
            // An entry's outputs a wave each, as the one-row product takes them (kernels/f32.gpu, exact_wave).
            const size_t per = kThreads / dev_.caps.subgroup_size * kinds.outputs;
            for (; r.first < entries; r.first += kGridY) {
                Args args(w, x, y, ni, no, r);
                put_scales(args, kinds, proj[p].weight, routing.n_expert);
                launch_writing({{y, size_mul(proj[p].rows, entries)}}, kinds.routed, (unsigned)((proj[p].rows + per - 1) / per), (unsigned)std::min<size_t>(kGridY, entries - r.first), args);
            }
        }
        record_matrix_path(MatrixPath::f32);
    }
    // Points twin_slot_ at the twin of x's `blocks` blocks under `map`, with its half sums where `sums` asks: true where a kept twin holds it, which becomes the newer, false where a fresh slot waits for its build.
    bool find_twin(const float* x, size_t blocks, bool int8, const TwinMap& map, bool sums, size_t need) {
        want_twin_ = int8 ? 2 : 1;
        for (int i = 0; i < 2; ++i)
            if (serves_whole(kept_[i], x, blocks, int8, map, sums)) {
                if (i == 1) std::swap(kept_[0], kept_[1]);
                twin_slot_ = kept_[0].slot;
                return true;
            }
        twin_slot_ = take_twin_slot(need);
        return false;
    }
    // The twin of X's `rows` rows laid out by `map` (block-int8 for the int8 class), with its half sums where a tile reads it (`sums`), built unless the kept twin holds it or `build` is false.
    TwinPlanes twin_of(CSlice X, size_t rows, size_t nin, bool int8, TwinMap map, bool build, bool sums) {
        const size_t blocks = size_mul(rows, nin / 32), need = twin_bytes(map.total, int8);
        const float* x = f32(X);
        // A kept twin of rows lying row after row serves any whole blocks of them, as held_mix keeps one twin of several items' rows.
        for (int i = 0; build && i < 2; ++i) {
            const Kept& k = kept_[i];
            if (!serves_part(k, x, blocks, int8, map, sums)) continue;
            const size_t at = size_t(x - k.x) / 32;
            if (i == 1) std::swap(kept_[0], kept_[1]);
            twin_slot_ = kept_[0].slot;
            const TwinPlanes t = twin_planes((uint8_t*)as_gpu<Api>(*twin_[twin_slot_]).data(), kept_[0].map.total, int8);
            return {t.q + at * (int8 ? 32 : 64), t.d + at, (float*)((uint8_t*)t.s + at * (int8 ? 4 : 8))};
        }
        if (build && !find_twin(x, blocks, int8, map, sums, need)) {
            const TwinPlanes t = twin_planes((uint8_t*)as_gpu<Api>(*twin_[twin_slot_]).data(), map.total, int8);
            unsigned long long nb = blocks;
            const Kernel k = int8 ? kTwin8 : kTwin16, build = twin_build(k, map);
            Args args(x, t.q, t.d, sums ? t.s : nullptr, nb);
            if (build != k) args.put(map);
            launch_writing({}, build, (unsigned)((blocks + kThreads / 32 - 1) / (kThreads / 32)), 1, args);
            // Kept for the next product of the same activations, unless the host may write them in place.
            if (!as_gpu<Api>(*X.buffer).host_ptr()) keep_twin({x, blocks * 32, int8, map, twin_slot_, sums});
        }
        return twin_planes((uint8_t*)as_gpu<Api>(*twin_[twin_slot_]).data(), map.total, int8);
    }
    // The half-block builds of up to 8 columns address a column's twin by a 32-bit byte offset from the call's first column (kernels/int16.gpu, q8_0_rows_half); checked before any launch, for every int8 call of that form.
    // The check is conservative: it also refuses calls of that form that keep 64-bit offsets, whose twin would pass 4 GiB only at shapes no model has.
    static void half_span(int form, size_t columns, size_t nin) {
        if (form == 2 && size_mul(columns, nin) > 0xffffffffull) throw std::runtime_error(Api::prefix + "a half-block row call's twin is too wide for its 32-bit offsets");
    }
    // The row kernel's call for the group's projections from column `at`, their output rows one after another.
    RowCall row_call(const Proj* proj, size_t nproj, size_t at, const TwinKernels& rk, size_t nin) const {
        RowCall c{};
        for (size_t pi = 0; pi < nproj; ++pi) {
            c.w[pi] = proj[pi].w;
            c.y[pi] = proj[pi].y + at * proj[pi].nout;
            c.nout[pi] = proj[pi].nout;
            c.f[pi] = proj[pi].f;
            if (pi) c.start[pi - 1] = c.total;
            c.total += proj[pi].nout;
        }
        c.nproj = (unsigned)nproj;
        c.cluster = row_cluster(rk, nin);
        return c;
    }
    // The twin of X, then each run on its kernel: generated rows and prompts below the crossover on the row kernel, kColumns at a time, longer prompts' rows on the tile.
    // Without runs a call is one run, on the row kernel below the crossover and from it on the tile at a split of one column tile, an extent of the crossover itself.
    void product_twin(const TwinKernels& tk, const Proj* proj, size_t nproj, CSlice X, size_t nin, size_t nbatch, bool accumulate, RowRuns runs, bool twin, bool int8, bool own) {
        const size_t qsize = int8 ? 1 : 2;
        const RowRun whole[1] = {{nbatch, on_rows(nbatch) ? size_t(1) : std::max<size_t>(2, dev_.profile.prompt_rows_below)}};
        const RowRuns all = runs.n ? runs : RowRuns{whole, 1};
        TwinMap map = twin_map(all, nbatch, nin);
        bool tiles = false;
        for (size_t r = 0; r < all.n; ++r) tiles = tiles || !on_rows(all.runs[r].extent);
        if (tiles) want_sums_ = tk.tile_sums;
        half_span(!int8 ? 0 : nin / 32 % 2 ? 1 : 2, std::min(nbatch, kColumns), nin);
        const auto [q0, d0, s0] = twin_of(X, nbatch, nin, int8, map, twin, tiles && tk.tile_sums);
        unsigned long long ni = nin;
        int add = accumulate;
        size_t begin = 0;
        for (size_t r = 0; r < all.n; ++r) {
            size_t end = all.runs[r].end;
            if (end <= begin || end > nbatch) throw std::runtime_error(Api::prefix + "row runs out of order or past the call");
            // Adjacent runs share launches where both take the same kernel and split, one per sequence otherwise: a decode column, and a tile row, gives the same bits in any call.
            const auto same = [&](const RowRun& a, const RowRun& b) {
                return on_rows(a.extent) == on_rows(b.extent) && (on_rows(a.extent) || split_tiles_of(a.extent) == split_tiles_of(b.extent));
            };
            while (r + 1 < all.n && same(all.runs[r], all.runs[r + 1]) && all.runs[r + 1].end > end && all.runs[r + 1].end <= nbatch) end = all.runs[++r].end;
            for (size_t at = begin; at < end;) {
                // int8 rows of an even block count take the half-block order, on word loads; the F16 class keeps the quarter layout, measured faster for its widened weights (docs/STATUS.md).
                const int form = !int8 ? 0 : nin / 32 % 2 ? 1 : 2;
                const bool columns = on_rows(all.runs[r].extent);
                const size_t rows = columns ? std::min(end - at, kColumns) : end - at;
                size_t first = 0;
                unsigned xrow = 0, xblock = 0;
                map.locate(at, first, xrow, xblock);
                const uint8_t* q = q0 + first * 32 * qsize;
                const float* d = d0 + first;
                const void* xs = (const uint8_t*)s0 + first * 4 * qsize;
                if (columns) {
                    // One launch for each three of the group's projections, their output rows one after another, or one for six where a pair build holds them.
                    const TwinKernels& rk = row_form(tk, nin, int8);
                    if (nproj > 3 && !rows_pair(rk, form, proj, nproj, 3, at, nin, rows, q, d, q, d, ni, add)) {
                        for (size_t p0 = 0; p0 < nproj; p0 += 3) rows_launch(rk, form, row_call(proj + p0, std::min<size_t>(3, nproj - p0), at, rk, nin), nin, rows, q, d, ni, add);
                    } else if (nproj <= 3) {
                        rows_launch(rk, form, row_call(proj, nproj, at, rk, nin), nin, rows, q, d, ni, add, nproj == 1);
                    }
                } else {
                    // The split follows the run's extent, its row class, never the rows this call holds; a tile call takes up to kTileProj projections.
                    tile(tk, proj, nproj, q, d, xs, xrow, xblock, at, rows, split_tiles_of(all.runs[r].extent), nin, add, int8, own);
                }
                at += rows;
            }
            begin = end;
        }
        if (begin != nbatch) throw std::runtime_error(Api::prefix + "row runs short of the call");
        record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
        for (size_t pi = 0; pi < nproj; ++pi) forget_written(proj[pi].y, proj[pi].nout * nbatch);
    }
    // A row kernel build and the output rows a wave of it takes, R, so a twin load serves that many rows (kernels/int16.gpu).
    struct RowBuild {
        Kernel kernel;
        size_t rows;
    };
    // The row build of a call of `rows` columns and `total` output rows, and the rows a block of it takes.
    std::pair<RowBuild, size_t> rows_build(const TwinKernels& rk, int form, size_t nin, size_t rows, size_t total) {
        const int build = rows <= 1 ? 0 : rows <= 2 ? 1 : rows <= 4 ? 2 : rows <= 8 ? 3 : 4;
        // A one-column half-block row of at most 16 blocks takes a 64-lane wave's halves as two rows, and from q8_decode_two_rows_from output rows the others take two rows a wave.
        // Every build keeps the one-row build's bits and reads the row width, the output count and the type, never the call's columns (kernels/int16.gpu).
        const bool half1 = rk.two[form][0] == kHalf8x1 && build == 0;
        const size_t blocks = nin / 32;
        // A row under two spans of blocks loads one step at a time, and at two rows a wave a row of q8_decode_four_steps_from blocks four (kernels/int16.gpu, kNarrowSteps).
        const bool paired = half1 && blocks <= 16 && dev_.caps.subgroup_size == 64, narrow = blocks < dev_.caps.subgroup_size;
        const bool two = half1 && total >= dev_.profile.q8_decode_two_rows_from;
        // A row of two to five spans of blocks takes one row a wave, all of its steps' loads issued at once (kernels/int16.gpu, q8_0_half_held).
        static constexpr Kernel kHeld[4] = {kHalf8x1h2, kHalf8x1h3, kHalf8x1h4, kHalf8x1h5};
        const size_t span = dev_.caps.subgroup_size / 2, spans = (blocks + span - 1) / span;
        const bool held = half1 && !paired && spans >= 2 && spans <= 5;
        RowBuild rb = held   ? RowBuild{kHeld[spans - 2], 1}
                      : paired ? RowBuild{kHalf8x1p, 2}
                      : two  ? RowBuild{narrow ? kHalf8x1r2s1 : blocks >= dev_.profile.q8_decode_four_steps_from ? kHalf8x1r2s4 : kHalf8x1r2, 2}
                      : half1 && narrow ? RowBuild{kHalf8x1s1, 1}
                                        : row_build(rk, form, build, total);
        // The half form's builds of 11 to 15 columns compute every column they hold at two rows a wave, so a call of 9 to 15 columns takes the narrowest that holds it.
        // A build of 9 or 10 columns ends on a group of one or two, whose twin loads the compiler issued one at a time, each behind the previous one's wait, so those calls take the 11-column build (docs/STATUS.md, ROCm wide decode columns).
        static constexpr Kernel kHalfWide[5] = {kHalf8x11, kHalf8x12, kHalf8x13, kHalf8x14, kHalf8x15};
        if ((rb.kernel == kHalf8x16 || rb.kernel == kHalf8x16r4) && rows < 16) rb = {kHalfWide[std::max<size_t>(rows, 11) - 11], kQ8Twins.two_rows[2][4]};
        const size_t per = block_rows(rb.kernel, rb.rows, row_cluster(rk, nin), nin, total);
        return {rb, per};
    }
    // One launch of a call's rows on the build its columns and rows take; push, a call of one projection, lets a held build's launch be held for its push (HeldPush).
    void rows_launch(const TwinKernels& rk, int form, const RowCall& c, size_t nin, size_t rows, const void* q, const float* d, unsigned long long ni, int add, bool push = false) {
        const auto [rb, per] = rows_build(rk, form, nin, rows, c.total);
        int ncols = (int)rows;
        if (conv_store_ && c.nproj == 1 && !add && rows == 1 && (rb.kernel == kRows16x1 || rb.kernel == kRows8x1)) {
            launch(rb.kernel == kRows16x1 ? kRows16x1Conv : kRows8x1Conv, (unsigned)((c.total + per - 1) / per), 1, Args(c, q, d, ni, ncols, add, *conv_store_));
            conv_stored_ = true;
        } else
            launch(rb.kernel, (unsigned)((c.total + per - 1) / per), 1, Args(c, q, d, ni, ncols, add));
        static constexpr Kernel kHeld[4] = {kHalf8x1h2, kHalf8x1h3, kHalf8x1h4, kHalf8x1h5}, kHeldPush[4] = {kHalf8x1h2p, kHalf8x1h3p, kHalf8x1h4p, kHalf8x1h5p};
        for (size_t i = 0; i < 4; ++i)
            if (push && rb.kernel == kHeld[i] && !add && !ops_.empty()) push_ = HeldPush{ops_.size() - 1, kHeldPush[i], c.y[0], c.total};
    }
    // The pair build of one column's `total` summed rows and the rows a block of it takes, kKernels where no pair build holds them.
    std::pair<Kernel, size_t> pair_build(const TwinKernels& rk, int form, size_t nin, size_t total) {
        const auto [rb, per] = rows_build(rk, form, nin, 1, total);
        static constexpr Kernel kHeld[4] = {kHalf8x1h2, kHalf8x1h3, kHalf8x1h4, kHalf8x1h5}, kHeldPair[4] = {kHalf8x1h2Pair, kHalf8x1h3Pair, kHalf8x1h4Pair, kHalf8x1h5Pair};
        for (size_t i = 0; i < 4; ++i)
            if (rb.kernel == kHeld[i]) return {kHeldPair[i], per};
        return {rb.kernel == kRows16x1 ? kRows16x1Pair : rb.kernel == kRows16x1Walk2 ? kRows16x1Walk2Pair : rb.kernel == kRows16x1Walk3 ? kRows16x1Walk3Pair : rb.kernel == kRows16x1Walk5 ? kRows16x1Walk5Pair : kKernels, per};
    }
    // The list build of one column's `total` summed rows and the rows a block of it takes: the held builds' (int16.gpu, LLMX_HELD_LIST), kKernels for any other build.
    std::pair<Kernel, size_t> list_build(const TwinKernels& rk, int form, size_t nin, size_t total) {
        const auto [rb, per] = rows_build(rk, form, nin, 1, total);
        static constexpr Kernel kHeld[4] = {kHalf8x1h2, kHalf8x1h3, kHalf8x1h4, kHalf8x1h5}, kHeldList[4] = {kHalf8x1h2List, kHalf8x1h3List, kHalf8x1h4List, kHalf8x1h5List};
        for (size_t i = 0; i < 4; ++i)
            if (rb.kernel == kHeld[i]) return {kHeldList[i], per};
        return {kKernels, per};
    }
    // Two to six projections of one column in one launch where their summed rows' build has a pair build (int16.gpu, LLMX_PAIR, LLMX_HELD_PAIR): the first `split` and the rest as two calls over the grid's y, the second on twin (qb, db).
    // Every row build keeps the one-row build's bits, so the summed rows choosing the build change no bits; false where no pair build holds the call.
    bool rows_pair(const TwinKernels& rk, int form, const Proj* proj, size_t nproj, size_t split, size_t at, size_t nin, size_t rows, const void* q, const float* d, const void* qb, const float* db,
                   unsigned long long ni, int add) {
        if (rows != 1 || split > 3 || nproj - split > 3) return false;
        const RowCall a = row_call(proj, split, at, rk, nin), b = row_call(proj + split, nproj - split, at, rk, nin);
        const auto [pair, per] = pair_build(rk, form, nin, a.total + b.total);
        if (pair == kKernels) return false;
        int ncols = 1;
        launch(pair, (unsigned)((std::max(a.total, b.total) + per - 1) / per), 2, Args(a, b, q, d, ni, ncols, add, qb, db));
        return true;
    }
    // The output rows a block of a one-column or dense row launch takes, R rows a cluster, where MXFP4's one-column 8-bit builds become the quad rows' builds and gpu_shape::kQuadLanes lanes a row by gpu_shape::quad_rows.
    size_t block_rows(Kernel& k, size_t R, unsigned cluster, size_t nin, size_t rows) {
        if ((k == kRowsMXFP48x1 || k == kRoutedMXFP48) && gpu_shape::quad_rows(cluster, nin / 32)) {
            k = k == kRowsMXFP48x1 ? kQuadRowsMXFP48x1 : kQuadRoutedMXFP48;
            return kThreads / gpu_shape::kQuadLanes;
        }
        // A one-column 16-bit Q8_0 row of a width a walk build holds takes it at eight or four rows a wave where the call's `rows` still give each compute unit row_walk_waves_per_cu waves (kernels/int16.gpu, q8_0_walk).
        if (const unsigned h = gpu_shape::walk_passes(dev_.caps.subgroup_size, nin); h && (k == kRows16x1 || k == kRouted16))
            for (size_t walk = 8; walk >= 4; walk /= 2)
                if (rows / walk >= size_t(dev_.profile.row_walk_waves_per_cu) * dev_.caps.compute_units) {
                    k = k == kRows16x1 ? (h == 2 ? kRows16x1Walk2 : h == 3 ? kRows16x1Walk3 : kRows16x1Walk5) : (h == 2 ? kRouted16Walk2 : h == 3 ? kRouted16Walk3 : kRouted16Walk5);
                    return walk * rows_per_block_;
                }
        return R * rows_per_block_ * (dev_.caps.subgroup_size / cluster);
    }
    // The build of a form and column count for a call of `total` rows: the four-row build once its waves give each compute unit row_four_waves_per_cu, else the two-row one.
    // A row keeps its lanes and order in every build, so the choice changes no bits; the sweep behind it is in docs/STATUS.md (ROCm many users).
    RowBuild row_build(const TwinKernels& tk, int form, int build, size_t total) const {
        if (tk.four[form][build] != kKernels && rows_a_cluster(dev_.caps, total, dev_.caps.subgroup_size, 4, dev_.profile.row_four_waves_per_cu) == 4) return {tk.four[form][build], 4};
        return {tk.two[form][build], tk.two_rows[form][build]};
    }
    // One tile call over the projections' rows [at, at + rows): their output tiles side by side, the inner dimension split for `tiles` column tiles (a prompt's own), the parts added by the reduce kernel.
    // The twin's rows lie `xrow` blocks apart and a row's blocks `xblock` apart (TwinMap); with `own` each projection takes its own call's split, so its rows keep their bits in any group (docs/src/backends-gpu.md).
    void tile(const TwinKernels& tk, const Proj* proj, size_t nproj, const uint8_t* q, const float* d, const void* xs, unsigned xrow, unsigned xblock, size_t at, size_t rows, size_t tiles, size_t nin,
              int add, bool int8, bool own) {
        const size_t nblk = nin / 32;
        size_t own_kper[kTileProj] = {};
        for (size_t pi = 0; own && pi < nproj; ++pi) own_kper[pi] = split_blocks(dev_.profile, dev_.caps, (proj[pi].nout + kTileOut - 1) / kTileOut * tiles, nblk);
        if (own && std::any_of(own_kper, own_kper + nproj, [&](size_t k) { return k != own_kper[0]; })) {
            bool taken[kTileProj] = {};
            for (size_t a = 0; a < nproj; ++a) {
                Proj same[kTileProj];
                size_t n = 0;
                for (size_t b = a; b < nproj && !taken[a]; ++b)
                    if (!taken[b] && own_kper[b] == own_kper[a]) same[n++] = proj[b], taken[b] = b != a;
                if (n) tile(tk, same, n, q, d, xs, xrow, xblock, at, rows, tiles, nin, add, int8, true);
                taken[a] = true;
            }
            return;
        }
        // The tall build where the profile's tile_rows_for takes the tallest height; the split stays the 64-output build's, so both builds give the same bits.
        size_t outputs = 0, gx64 = 0;
        for (size_t pi = 0; pi < nproj; ++pi) {
            outputs += proj[pi].nout;
            gx64 += (proj[pi].nout + kTileOut - 1) / kTileOut;
        }
        const size_t tall_from = int8 ? dev_.profile.qtile_tall_from_8bit : dev_.profile.qtile_tall_from;
        const bool tall = tk.tall[int8] != kKernels && (tk.tall[int8] < kTallQ4K8 || Api::int8_tensor_tiles) && tile_rows_for(dev_.caps, dev_.profile, kTileOut, kTileOut, kTileOutTall, outputs, tiles, nin, tall_from, int8) == kTileOutTall;
        // A call of more than kTileRows rows takes the K-quants' square build, whose 128 rows read each weight half as often; the 64-row build keeps shorter calls from running half-empty blocks.
        const bool square = tall && int8 && tk.square != kKernels && rows > kTileRows;
        const size_t per = tall ? kTileOutTall : kTileOut, trows = square ? kTileRowsTall : kTileRows;
        // The tile addresses a workgroup's weights and twin by 32-bit offsets off its first output and row, a Q8_0 block's 34 bytes bounding every twin type's bytes for 32 values.
        if ((nin / 32) * per * 34 + 64 > 0xffffffffull || (trows * xrow + (nin / 32) * xblock) * 32 * (int8 ? 1 : 2) > 0xffffffffull)
            throw std::runtime_error(Api::prefix + "a prompt tile's rows are too wide for its 32-bit offsets");
        TileCall c{};
        size_t gx = 0;
        for (size_t pi = 0; pi < nproj; ++pi) {
            c.w[pi] = proj[pi].w;
            c.y[pi] = proj[pi].y + at * proj[pi].nout;
            c.nout[pi] = proj[pi].nout;
            c.f[pi] = proj[pi].f;
            c.start[pi] = (unsigned)gx;
            gx += (proj[pi].nout + per - 1) / per;
        }
        c.nproj = (unsigned)nproj;
        c.nin = nin;
        c.nrows = rows;
        c.add = add;
        c.xrow = xrow;
        c.xblock = xblock;
        c.xs = xs;
        const size_t kper = own ? own_kper[0] : split_blocks(dev_.profile, dev_.caps, gx64 * tiles, nblk);
        c.kper = (unsigned)kper;
        c.parts = (unsigned)((nblk + kper - 1) / kper);
        if (c.parts > 1) {
            const size_t need = tile_partials_bytes(c.parts, rows, outputs);
            if (!partials_ || partials_->size() < need) partials_ = alloc(need, Memory::device);
            c.partials = (float*)as_gpu<Api>(*partials_).data();
        }
        const Args args(c, q, d);
        launch(square ? tk.square : tall ? tk.tall[int8] : tk.tile[int8], (unsigned)gx, (unsigned)((rows + trows - 1) / trows), args, c.parts);
        if (c.parts > 1) {
            size_t g = 0;
            for (size_t pi = 0; pi < nproj; ++pi) {
                c.start[pi] = (unsigned)g;
                g += (proj[pi].nout * rows + kThreads - 1) / kThreads;
            }
            const Args rargs(c, (unsigned)g);
            launch(kTileReduce, (unsigned)std::min(g, std::max<size_t>(1, resident(reduce_per_cu_, kTileReduce))), 1, rargs);
        }
    }
    // Rows `ids` of a table through the type's gather kernel, ids on the device; an id of `nrows` or more gives a zero row.
    void gather(Kernel k, Slice dst, const void* table, size_t width, const unsigned* ids, size_t count, size_t nrows, const TypeKernels* kinds = nullptr, const WeightView* weight = nullptr) {
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(count, width), sizeof(float)));
        float* d = f32(dst);
        unsigned long long w = width;
        unsigned nr = (unsigned)std::min<size_t>(nrows, UINT32_MAX);
        Args args(d, table, ids, w, nr);
        if (kinds) put_scales(args, *kinds, *weight);
        launch(k, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)count, args);
    }
    // Where a producer writes the twin of its output of `values` floats: once a product has read a twin, in that class, for an output of whole blocks in device memory.
    struct TwinOut {
        void* q = nullptr;
        float* d = nullptr;
        void* s = nullptr;
        bool int8 = false;
        TwinMap map{};
        size_t slot = 0;
    };
    // Its rows lie as `map` lays them out (TwinMap), the layout the product's own runs give; the twin takes the half sums too, flat or mapped, where the last tiled product read them.
    TwinOut twin_out(Slice dst, size_t values, const TwinMap& map) {
        if (!want_twin_ || values % 32 || as_gpu<Api>(*dst.buffer).host_ptr()) return {};
        const bool int8 = want_twin_ == 2;
        const size_t slot = take_twin_slot(twin_bytes(map.total, int8));
        const TwinPlanes t = twin_planes((uint8_t*)as_gpu<Api>(*twin_[slot]).data(), map.total, int8);
        return {t.q, t.d, !want_sums_ ? nullptr : t.s, int8, map, slot};
    }
    // A twin producer's build for `m`: the map-free one where every row lies row after row or no twin is written, the one reading the map otherwise (kernels/f32.gpu, _major).
    // The callers pass the map as the last argument only to the build that declares it.
    static Kernel twin_build(Kernel k, const TwinMap& m) {
        if (!m.regions || m.flat()) return k;
        switch (k) {
        case kSiluMulTwin16: return kSiluMulTwin16Major;
        case kSiluMulTwin8: return kSiluMulTwin8Major;
        case kRmsNorm16Twin16: return kRmsNorm16Twin16Major;
        case kRmsNorm16Twin8: return kRmsNorm16Twin8Major;
        case kRmsNorm32Twin16: return kRmsNorm32Twin16Major;
        case kRmsNorm32Twin8: return kRmsNorm32Twin8Major;
        case kAttentionVec: return kAttentionVecMajor;
        case kAttentionVecMerge: return kAttentionVecMergeMajor;
        case kAttentionVec256: return kAttentionVec256Major;
        case kAttentionVecMerge256: return kAttentionVecMerge256Major;
        case kTwin16: return kTwin16Major;
        case kTwin8: return kTwin8Major;
        default: return k;
        }
    }
    // A row's mHC coefficients where pre is given and its collapse and norm with pre_in into dst where dst is given, in one launch whose roles run side by side (kernels/f32.gpu, llmx_hc_coeffs_collapse).
    void hc_launch(const Slice* pre, Slice post, Slice comb, CSlice mixes, CSlice stream, CSlice scale, CSlice base, float hc_eps, size_t iters, const Slice* dst, CSlice pre_in,
                   CSlice w, size_t rows, size_t hc, size_t dim, float eps) {
        if (pre && (!hc || hc > 16 || !dim || !iters || !std::isfinite(eps) || eps < 0.0f || !std::isfinite(hc_eps) || hc_eps < 0.0f))
            throw std::runtime_error(Api::prefix + "invalid mHC coefficient dimensions or epsilon");
        if (dst && (!hc || !dim || !std::isfinite(eps) || eps <= 0.0f)) throw std::runtime_error(Api::prefix + "invalid mHC collapse dimensions or epsilon");
        if (!rows) return;
        const size_t mix = size_mul(hc + 2, hc), n = size_mul(rows, hc), nc = size_mul(n, hc), mn = size_mul(rows, mix), xn = size_mul(rows, size_mul(hc, dim)), dn = size_mul(rows, dim);
        floats(stream, xn);
        float *a = nullptr, *b = nullptr, *c = nullptr, *o = nullptr;
        const float *m = nullptr, *sc = nullptr, *bs = nullptr, *p = nullptr, *wp = nullptr;
        if (pre) {
            for (const auto& [s, count] : {std::pair<CSlice, size_t>{*pre, n}, {post, n}, {comb, nc}, {mixes, mn}, {scale, 3}, {base, mix}}) floats(s, count);
            for (const auto& [o, on] : {std::pair<CSlice, size_t>{*pre, n}, {post, n}, {comb, nc}})
                if (overlaps(o, on, mixes, mn) || overlaps(o, on, stream, xn) || overlaps(o, on, scale, 3) || overlaps(o, on, base, mix) || (dst && overlaps(o, on, pre_in, n)))
                    throw std::runtime_error(Api::prefix + "mHC coefficient outputs overlap their inputs");
            if (overlaps(*pre, n, post, n) || overlaps(*pre, n, comb, nc) || overlaps(post, n, comb, nc)) throw std::runtime_error(Api::prefix + "mHC coefficient outputs must be disjoint");
            a = f32(*pre), b = f32(post), c = f32(comb), m = f32(mixes), sc = f32(scale), bs = f32(base);
            u32(size_mul(hc, dim));
        }
        if (dst) {
            for (const auto& [s, count] : {std::pair<CSlice, size_t>{*dst, dn}, {pre_in, n}, {w, dim}}) floats(s, count);
            if (overlaps(*dst, dn, stream, xn) || overlaps(*dst, dn, pre_in, n) || overlaps(*dst, dn, w, dim) ||
                (pre && (overlaps(*dst, dn, *pre, n) || overlaps(*dst, dn, post, n) || overlaps(*dst, dn, comb, nc) || overlaps(*dst, dn, mixes, mn) || overlaps(*dst, dn, scale, 3) || overlaps(*dst, dn, base, mix))))
                throw std::runtime_error(Api::prefix + "mHC collapse output must be disjoint from its inputs");
            o = f32(*dst), p = f32(pre_in), wp = f32(w);
        }
        const float* x = f32(stream);
        unsigned h = u32(hc), d = u32(dim), it = u32(iters), collapses = dst ? u32(rows) : 0u, blocks = u32((pre ? rows : 0) + collapses);
        // The output's twin beside it where a product reads one next, as rms_norm_rows writes it.
        StreamTwin t = dst ? stream_twin(*dst, rows, dim, {}, dim % 32 == 0) : StreamTwin{};
        const Args args(a, b, c, m, x, sc, bs, it, hc_eps, o, p, wp, h, d, eps, t.q, t.d, t.s, t.level, t.map, t.major, collapses);
        if (dst) launch_writing({{o, dn}}, kHcCoeffsCollapse, blocks, 1, args, 1, 1024);   // kernels/f32.gpu, kCollapseThreads
        else launch(kHcCoeffsCollapse, blocks, 1, args);
        if (dst) keep(t.t, o, dn);
    }
    void keep(const TwinOut& t, const float* x, size_t values) {
        if (t.q) keep_twin({x, values, t.int8, t.map, t.slot, t.s != nullptr});
    }
    // A stream or gate op's twin of its `rows` rows `width` wide, written where a product reads one next and each row is whole blocks (kernels/f32.gpu, twin_at): where it goes, its level, its map and whether the map is read.
    struct StreamTwin {
        TwinOut t;
        void* q;
        float* d;
        void* s;
        int level, major;
        TwinMap map;
    };
    StreamTwin stream_twin(Slice dst, size_t rows, size_t width, RowRuns runs, bool whole = true) {
        const TwinOut t = whole && width % 32 == 0 ? twin_out(dst, size_mul(rows, width), twin_map(runs, rows, width)) : TwinOut{};
        return {t, t.q, t.d, t.s, !t.q ? 0 : t.int8 ? 2 : 1, t.q && t.map.regions && !t.map.flat() ? 1 : 0, t.map};
    }
    // A twin kept in its slot, which a product of the same activations reads instead of making its own; a launch or copy writing its input, any other launch, a write or an allocation forgets it.
    struct Kept {
        const float* x = nullptr;
        size_t values = 0;
        bool int8 = false;
        TwinMap map{};
        size_t slot = 0;
        bool sums = false;   // whether it holds the half sums a tile reads
        bool parts = false;  // whether whole blocks of it serve a product, as held_mix's row-after-row twin of several items does
    };
    // Whether kept twin k is the twin of x's `blocks` blocks under `map`, or holds them as whole blocks of a parts twin (held_mix's).
    static bool serves_whole(const Kept& k, const float* x, size_t blocks, bool int8, const TwinMap& map, bool sums) {
        return k.x == x && k.values == blocks * 32 && k.int8 == int8 && k.map == map && (k.sums || !sums);
    }
    static bool serves_part(const Kept& k, const float* x, size_t blocks, bool int8, const TwinMap& map, bool sums) {
        const bool linear = (!map.regions || map.flat()) && (!k.map.regions || k.map.flat());
        return k.x && k.parts && linear && k.int8 == int8 && (!sums || k.sums) && x >= k.x && x + blocks * 32 <= k.x + k.values && (x - k.x) % 32 == 0 && !(x == k.x && k.values == blocks * 32);
    }
    // Whether a kept twin serves x's `blocks` blocks, so twin_of builds none.
    bool twin_kept(const float* x, size_t blocks, bool int8, const TwinMap& map) const {
        for (const Kept& k : kept_)
            if (serves_whole(k, x, blocks, int8, map, false) || serves_part(k, x, blocks, int8, map, false)) return true;
        return false;
    }
    // The one owner of forgetting kept twins: forget_written drops each whose input the `n` floats written at y meet, keeping the newer first, and forget_twins drops both.
    void forget_written(const float* y, size_t n) {
        for (Kept& k : kept_)
            if (k.x && y < k.x + k.values && k.x < y + n) k = {};
        if (!kept_[0].x) std::swap(kept_[0], kept_[1]);
    }
    void forget_twins() { kept_ = {}; }
    // The slot a new twin goes into, the one not holding the newer kept twin, grown to `need` bytes; a twin kept there is dropped, and the choice follows the op sequence alone.
    size_t take_twin_slot(size_t need) {
        const size_t slot = kept_[0].x ? 1 - kept_[0].slot : 0;
        if (kept_[1].x && kept_[1].slot == slot) kept_[1] = {};
        if (!twin_[slot] || twin_[slot]->size() < need) {
            // A fresh twin buffer is no kept twin's input, so the twin kept in the other slot stays.
            const auto kept = kept_;
            twin_[slot] = alloc(need, Memory::device);
            kept_ = kept;
        }
        return slot;
    }
    void keep_twin(const Kept& k) {
        kept_[1] = kept_[0];
        kept_[0] = k;
    }
    // A launch that writes only `writes`, an exact product's or a router's outputs: it keeps the twin a later product may read unless one of them meets its input.
    void launch_writing(std::initializer_list<std::pair<const void*, size_t>> writes, Kernel k, unsigned gx, unsigned gy, const Args& args, unsigned gz = 1, unsigned threads = kThreads,
                        unsigned shared = 0) {
        const auto kept = kept_;
        launch(k, gx, gy, args, gz, threads, shared);
        kept_ = kept;
        for (const auto& [y, n] : writes) forget_written((const float*)y, n);
    }
    void launch(Kernel k, unsigned gx, unsigned gy, const Args& args, unsigned gz = 1, unsigned threads = kThreads, unsigned shared = 0) {
        raise_failed();
        // Any kernel but a product may write a kept twin's input.
        if (k < kRows16x1) forget_twins();
        issue(kernels_[k], gx, gy, gz, threads, args, kKernelNames[k], shared);
        ++launched_[k];
    }
    // A fused step's launch, which writes only the outputs moe_step forgets after it.
    void launch_step(Kernel k, unsigned gx, unsigned gy, const Args& args) {
        raise_failed();
        issue(kernels_[k], gx, gy, 1, kThreads, args, kKernelNames[k]);
        ++launched_[k];
    }
    // A launch or copy refused as it is enqueued throws at once; the stream is still the caller's.
    void launch_check(typename Api::Error e, const char* what) {
        Api::check(e, what);
        opened_ = true;
    }
    // The first asynchronous failure, kept as the cause every later read or launch throws.
    void remember(typename Api::Error e, const char* where) noexcept {
        std::lock_guard<std::mutex> lk(failed_mu_);
        if (failed_.empty()) failed_ = Api::prefix + where + " failed on the device with " + Api::error_name(e);
    }
    // Whether the device reported an asynchronous failure, which a collective's wait cancels its sum for.
    bool device_failed() noexcept {
        std::lock_guard<std::mutex> lk(failed_mu_);
        return !failed_.empty();
    }
    // A group sum that gave up on this device (fault_) throws once, at the next call that checks; the collective's later sums take the conservative path, so the backend goes on.
    void raise_failed() {
        {
            std::lock_guard<std::mutex> lk(failed_mu_);
            if (!failed_.empty()) throw std::runtime_error(failed_);
        }
        if (host_load(fault_) && host_exchange(fault_, 0u))
            throw std::runtime_error(Api::prefix + "a tensor group's sum on " + ordinal() + " gave up waiting for its peers; the group's later sums take the conservative path");
    }
    // A group sum's kernel, launched without raise_failed: the collective checks every member before it launches any, so no member waits for a peer whose launch threw.
    // A one-shot whose rows the held build already pushed (push_rows) launches its take alone; a stream `beside` the member's takes the launch at once rather than held with the member's operations.
    void group_launch(GpuSumForm form, unsigned blocks, const Args& args, bool pushed = false, typename Api::Stream beside = nullptr) {
        forget_twins();
        const Kernel k = pushed ? kGroupTake : form == GpuSumForm::chain || form == GpuSumForm::wrap ? kGroupChain : form == GpuSumForm::shares || form == GpuSumForm::draws ? kGroupShares : form == GpuSumForm::copies ? kGroupCopies : kGroupSum;
        ++launched_[k];
        if (!beside) return issue(kernels_[k], blocks, 1, 1, kThreads, args, kKernelNames[k]);
        Args a = args;
        launch_check(Api::launch(kernels_[k], blocks, 1, 1, kThreads, 0, beside, a.bytes, a.size), kKernelNames[k]);
    }
    // A one-shot's push alone, whose take the collective then hands the member as its pending join (pend_take).
    void group_push(unsigned blocks, const Args& args) {
        issue(kernels_[kGroupPush], blocks, 1, 1, kThreads, args, kKernelNames[kGroupPush]);
        ++launched_[kGroupPush];
    }
    // The tail a pushing held build takes after the held build's arguments (kernels/int16.gpu, GroupTail).
    struct GroupTail {
        const void* peers;
        unsigned* tiles;
        unsigned long long half_at, cap;
        int members, rank;
        unsigned tile;
    };
    static_assert(offsetof(GroupTail, half_at) == 16 && offsetof(GroupTail, tile) == 40 && sizeof(GroupTail) == 48, "GroupTail mirrors the kernel's argument");
    // The last held Q8_0 row launch of one projection that replaces its rows, which a sum of exactly those rows may turn into its pushing build (push_rows).
    struct HeldPush {
        size_t op;
        Kernel kernel;
        const float* y;
        unsigned long long n;
    };
    std::optional<HeldPush> push_;
    // Whether the last held operation is the held row launch writing own's n rows and nothing else, which then becomes its pushing build sending them as the tail says; the collective then launches only the take.
    bool push_rows(const float* own, size_t n, const GroupTail& t) {
        if (!push_ || pending_ || push_->op + 1 != ops_.size() || push_->y != own || push_->n != n) return false;
        Held& op = ops_.back();
        if (op.at + op.size != op_bytes_.size()) return false;
        Args a;
        std::memcpy(a.bytes, op_bytes_.data() + op.at, op.size);
        a.size = op.size;
        a.put(t);
        op_bytes_.resize(op.at + a.size);
        std::memcpy(op_bytes_.data() + op.at, a.bytes, a.size);
        op.size = a.size;
        op.f = kernels_[push_->kernel];
        op.name = kKernelNames[push_->kernel];
        ++launched_[push_->kernel];
        push_.reset();
        return true;
    }
    // A fast collective's staging armed at join (kernels/f32.gpu, llmx_group_arm).
    void group_arm(void* stage, size_t words) {
        const unsigned long long w = words;
        issue(kernels_[kGroupArm], (unsigned)std::min<size_t>(1024, (words + kThreads - 1) / kThreads), 1, 1, kThreads, Args(stage, w), "llmx_group_arm");
    }
    // The probe's check of this device's add against sanitized operands, np * np threads in one block (kernels/f32.gpu, llmx_group_add_check).
    void group_add_check(const Buffer& patterns, int np, Buffer& differ) {
        issue(kernels_[kGroupAddCheck], 1, 1, 1, (unsigned)(np * np), Args(f32(CSlice{&patterns, 0}), np, f32(Slice{&differ, 0})), "llmx_group_add_check");
    }

    // A stream operation held until the next flush, which issues them in order: a kernel with its packed arguments, a copy, a zero fill, or a wait for or write of a 64-bit word, `bytes` its value (docs/src/backends-gpu.md, records).
    struct Held {
        enum Kind : uint8_t { kernel, copy, fill, wait_word, write_word, wait_event } kind;
        typename Api::Function f;
        unsigned gx, gy, gz, threads;
        const char* name;
        size_t at, size;   // a kernel's argument bytes in op_bytes_
        void* dst;
        const void* src;
        size_t bytes;   // a copy's or fill's bytes, a wait's or write's value, a kernel's dynamic shared memory
        typename Api::CopyKind dir;
        bool head = false;   // a kernel of the output head's products (matmul_logits)
    };
    void hold(const Held& op) {
        materialize();
        ops_.push_back(op);
        opened_ = true;
        if (flush_held_) {
            flush_held_ = false;
            return flush();
        }
        // A device with nothing left running starts on the first few, rather than waiting while the host records a whole chunk; it is asked once a submission, at the first few, so a short one asks nothing.
        if (ops_.size() == kFirstChunk && !asked_) {
            asked_ = true;
            if (done(tickets_.last())) return flush();
        }
        if (ops_.size() >= dev_.profile.dispatch_chunk) flush();
    }
    void issue(typename Api::Function f, unsigned gx, unsigned gy, unsigned gz, unsigned threads, const Args& args, const char* name, unsigned shared = 0) {
        // A held inject goes first, since its hold may flush and empty op_bytes_ under an offset taken before it.
        materialize();
        const size_t at = (op_bytes_.size() + 15) & ~size_t(15);
        op_bytes_.resize(at + args.size);
        std::memcpy(op_bytes_.data() + at, args.bytes, args.size);
        hold({Held::kernel, f, gx, gy, gz, threads, name, at, args.size, nullptr, nullptr, shared, Api::kAnyCopy, head_});
    }
    // A copy named by what kernel_times calls it: an upload from host memory, a readback to it, or a copy on the device.
    void hold_copy(void* dst, const void* src, size_t bytes, typename Api::CopyKind dir, const char* name) { hold({Held::copy, nullptr, 0, 0, 0, 0, name, 0, 0, dst, src, bytes, dir}); }
    void hold_fill(void* dst, size_t bytes) { hold({Held::fill, nullptr, 0, 0, 0, 0, "zero_async", 0, 0, dst, nullptr, bytes, Api::kAnyCopy}); }
    // The stream waits until *word is at least `value`, or writes `value` there once the work before it is done.
    void hold_wait(uint64_t* word, uint64_t value) { hold({Held::wait_word, nullptr, 0, 0, 0, 0, "wait_word", 0, 0, word, nullptr, value, Api::kAnyCopy}); }
    void hold_write(uint64_t* word, uint64_t value) { hold({Held::write_word, nullptr, 0, 0, 0, 0, "write_word", 0, 0, word, nullptr, value, Api::kAnyCopy}); }
    // The stream waits for the work before the event's last record when this is flushed, on another stream of this device.
    void hold_wait_event(typename Api::Event event) { hold({Held::wait_event, nullptr, 0, 0, 0, 0, "stream_wait_event", 0, 0, (void*)event, nullptr, 0, Api::kAnyCopy}); }
    // Issues the held operations in order; one HIP refuses throws naming it, and those after it are dropped with it.
    void flush() {
        materialize();
        push_.reset();
        if (ops_.empty()) return;
        std::unique_lock<std::mutex> il(graves_->issue, std::defer_lock);
        if (diag_arm == 7) il.lock();
        std::vector<Held> ops;
        std::vector<uint8_t> bytes;
        ops.swap(ops_);
        bytes.swap(op_bytes_);
        // HIP finds a module's function through the calling thread's device, which another backend's call may have changed.
        Api::check(Api::set_device(dev_.index), "set_device");
        unsigned* const take = take_status_;
        take_status_ = nullptr;
        // The host's time in the launches, also where one throws.
        struct Timed {
            double& ms;
            std::chrono::steady_clock::time_point from;
            ~Timed() { ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count(); }
        } timed_launches{launch_ms_, std::chrono::steady_clock::now()};
        // A timing backend's stream waits behind its gate word until the host has issued the whole chunk, so a host behind its device shows as idle before the chunk's first op rather than inside one (docs/src/backends-gpu.md).
        uint64_t* const gate = timing_ ? (uint64_t*)as_gpu<Api>(*gate_).data() : nullptr;
        if (gate) gate_fence().gate_enter();
        try {
            if (gate) {
                if (!started_) waited_ = true;
                Api::check(Api::wait_word(stream_, as_gpu<Api>(*gate_).device_ptr(), ++gated_), "wait_word");
            }
            for (const Held& op : ops) {
                // A submission's busy time starts at its first operation after its waits on other devices (busy_ms).
                if (!started_ && op.kind == Held::wait_word) waited_ = true;
                if (!started_ && op.kind != Held::wait_word && op.kind != Held::wait_event) {
                    start();
                    started_ = true;
                }
                const bool timed = timing_ && op.kind <= Held::fill && timed_.size() < kTimedOps;
                if (timing_ && op.kind <= Held::fill && !timed) ++untimed_;
                if (timed) Api::check(Api::event_record(timed_event(2 * timed_.size()), stream_), "event_record");
                if (timed && delay_.count()) std::this_thread::sleep_for(delay_);
                if (op.kind == Held::kernel) {
                    launch_check(Api::launch(op.f, op.gx, op.gy, op.gz, op.threads, (unsigned)op.bytes, stream_, bytes.data() + op.at, op.size), op.name);
                } else if (op.kind == Held::copy) {
                    launch_check(Api::copy_async(op.dst, op.src, op.bytes, op.dir, stream_), "copy_async");
                } else if (op.kind == Held::wait_word) {
                    launch_check(Api::wait_word(stream_, op.dst, op.bytes), op.name);
                } else if (op.kind == Held::write_word) {
                    launch_check(Api::write_word(stream_, op.dst, op.bytes), op.name);
                } else if (op.kind == Held::wait_event) {
                    launch_check(Api::stream_wait_event(stream_, (typename Api::Event)op.dst), op.name);
                } else {
                    launch_check(Api::zero_async(op.dst, op.bytes, stream_), op.name);
                }
                if (timed) {
                    Api::check(Api::event_record(timed_event(2 * timed_.size() + 1), stream_), "event_record");
                    timed_.push_back({op.kind == Held::fill ? "fill" : op.name, op.head, op.kind == Held::kernel});
                }
            }
        } catch (...) {
            if (gate) host_store(gate, gated_), gate_fence().gate_leave();
            if (take) host_store(take, 1u);
            throw;
        }
        if (gate) host_store(gate, gated_), gate_fence().gate_leave();
        // The vectors keep their capacity for the next records.
        ops.clear();
        bytes.clear();
        ops_.swap(ops);
        bytes.swap(op_bytes_);
    }
    // A timing backend's i-th timing event of the reading interval, made the first time an interval reaches it and kept for the next.
    typename Api::Event timed_event(size_t i) {
        while (timed_events_.size() <= i) {
            typename Api::Event e{};
            Api::check(Api::event_create(&e, true), "event_create");
            timed_events_.push_back(e);
        }
        return timed_events_[i];
    }
    // The stream every collective of this backend sends its lanes' sums on (`beside`) or its copies' copies on, made the first time one asks.
    // One of each a backend, however many collectives it joins, keeps a device at four hardware queues with HIP's null stream, one to each compute pipe, since two sums waiting on one pipe can starve each other (docs/src/backends-gpu.md).
    typename Api::Stream sum_stream(bool copies) {
        typename Api::Stream& s = copies ? copy_stream_ : beside_stream_;
        if (!s) {
            Api::check(Api::set_device(dev_.index), "set_device");
            Api::check(Api::stream_create(&s), "stream_create");
        }
        return s;
    }
    template <class> friend class GpuCollective;
    template <class> friend class GpuHandoff;

    typename Api::Device dev_;
    std::shared_ptr<Graves<Api>> graves_ = std::make_shared<Graves<Api>>();   // before every buffer member, so it outlives them
    typename Api::Stream stream_ = nullptr;
    typename Api::Stream beside_stream_ = nullptr, copy_stream_ = nullptr;   // sum_stream's, null until a collective asks
    double launch_ms_ = 0;   // the host's time in flush's launches (host_times)
    typename Api::Module module_ = nullptr, int16_module_ = nullptr, dot_module_ = nullptr;
    BufferPtr parts_;  // attention's chunk partials, grown as calls need
    BufferPtr sparse_;  // sparse attention's scratch rows, grown as calls need
    // Two twin slots, each the twin of a product's activations or of a producer's output, grown as calls need, so the routed act's twin leaves the mixed streams' for the shared expert (Vulkan's twins, at the count that pattern needs).
    std::array<BufferPtr, 2> twin_{};
    std::array<Kept, 2> kept_{};   // the newer first
    size_t twin_slot_ = 0;   // the slot of the twin the last product read
    BufferPtr moe_out_;   // a routed down projection's slots before they join the residual
    BufferPtr expert_ids_;  // a cached step's round: each entry's slot, or none
    // The device's expert ring (cached_step, ExpertRing::of).
    std::unique_ptr<ExpertRing> ring_;
    BufferPtr delta_counter_;   // llmx_delta_step's (view, K head) and llmx_attention_rel_sink's (row, head group) counters, zero between launches
    unsigned* block_counters() {
        if (!delta_counter_) {
            delta_counter_ = alloc(kDeltaCounters * sizeof(unsigned), Memory::device);
            hold_fill(as_gpu<Api>(*delta_counter_).data(), kDeltaCounters * sizeof(unsigned));
        }
        return (unsigned*)as_gpu<Api>(*delta_counter_).data();
    }
    BufferPtr terms_counter_;   // attention_terms' finished parts a (row, head group), zero between launches
    BufferPtr step_counter_, step_twin_;   // the fused launches' finished blocks (moe_step's route, moe_router), zero between launches, and the fused moe_step's shared act's twin
    unsigned* step_counter() {
        if (!step_counter_) {
            step_counter_ = alloc(4 * 64, Memory::device);
            hold_fill(as_gpu<Api>(*step_counter_).data(), 4 * 64);
        }
        return (unsigned*)as_gpu<Api>(*step_counter_).data();
    }
    // llmx_conv_ring_mix4's grid barrier, the last word of step_counter's, which no other launch takes.
    unsigned* conv_mix_bar() { return step_counter() + 15; }
    int step_up_per_cu_[3][(kMoeStepDownQ8 - kMoeStepUpQ4K) / 2] = {{-1, -1, -1, -1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1, -1, -1, -1}};   // the wide gate+up builds' resident workgroups a compute unit, by chain count and one-chain lean build, found at their first step
    int reduce_per_cu_ = -1;   // the tile reduce's resident workgroups a compute unit, found at its first launch, which caps its grid
    BufferPtr moe_tab_;   // a grouped routed call's groups and entries (kernels/f32.gpu, llmx_moe_group)
    int want_twin_ = 0;   // the class of the last twin a product read, 1 block-int16 and 2 block-int8, which producers write; 0 before any
    bool want_sums_ = true;   // whether the last product with prompt tiles read the half sums (TwinKernels::tile_sums), which producers then write
    BufferPtr staged_;   // a product bound for host memory, formed here first
    BufferPtr partials_;   // the parts of a split tile call, grown as calls need
    BufferPtr scores_;   // the indexer's scores of 16 rows at most, grown as calls need
    BufferPtr dot_rows_;   // matmul_at_ids's gathered head rows, grown as calls need
    BufferPtr argmax_parts_;   // argmax_rows' slice maxima and ids, grown as calls need
    BufferPtr select_state_;   // the wide selection's histograms and slice counts, zero between calls
    size_t small_unit_ = kFallbackUnit, large_unit_ = kFallbackUnit;   // hipMalloc's pages at and past kLargeFrom (probe_memory)
    size_t driver_scratch_ = 0;   // the runtime's private scratch for the kernels that spill, taken at their first launch
    size_t rows_per_block_ = kThreads / dev_.caps.subgroup_size;   // the row kernel's output rows a block, a wave each
    std::array<typename Api::Function, kKernels> kernels_{};
    std::array<typename Api::Event, kEvents> events_{};
    std::array<std::atomic<uint32_t>, kEvents> syncing_{};   // per event slot, the threads in its synchronise (done, record)
    std::array<std::atomic<Ticket>, kEvents> slot_ticket_{};   // per event slot, the ticket recorded in it, 0 while record reuses it
    std::array<typename Api::Event, kEvents> starts_{};   // each ticket's first operation after its cross-device waits, timed against events_ (busy_ms)
    Ring args_, stage_, pass_;   // the argument ring, the staging ring, and pass_arg's ring in device memory
    std::array<Upload, kUploads> uploads_{};
    size_t next_upload_ = 0;
    TicketLedger tickets_;
    bool opened_ = false;
    bool started_ = false;   // the open ticket's start event is recorded
    bool waited_ = false;    // the open ticket waits on another device before its first operation
    std::array<bool, kEvents> chained_{};   // per event slot, its ticket starts at the previous ticket's end event rather than its own start (start)
    static constexpr size_t kFirstChunk = 8;
    bool asked_ = false;   // the open submission asked at kFirstChunk held whether its device had nothing running
    std::vector<Held> ops_;   // the stream operations held since the last flush
    std::vector<uint8_t> op_bytes_;   // their kernels' arguments
    const bool timing_;   // made for diagnostics, so flush times its kernels, copies and fills (kernel_times)
    bool head_ = false;   // on a timing backend, the products being issued are the output head's
    struct TimedOp {
        const char* name;
        bool head, kernel;
    };
    // A reading times the first kTimedOps ops of its interval, each between a start and an end event, which bounds the events a long interval takes.
    static constexpr size_t kTimedOps = 4096;
    std::vector<TimedOp> timed_;   // the ops timed since the last reading
    size_t untimed_ = 0;   // the ops past kTimedOps since the last reading
    BufferPtr gate_;   // a timing backend's gate word in host memory, which each flush's stream waits on until the host sets it to gated_
    uint64_t gated_ = 0;
    std::vector<typename Api::Event> timed_events_;   // their start and end events, two an op
    size_t launched_[kKernels] = {};   // launches of each kernel since rocm_take_launches last read them
    std::string route_only_;   // rocm_route_only's filter of the routed tile builds, empty for all of them
    std::chrono::microseconds delay_{0};   // delay_launches' sleep before each timed launch
    std::mutex failed_mu_;
    std::string failed_;
    unsigned* fault_ = nullptr;   // host memory a group sum's kernel sets where it gave up waiting (kernels/f32.gpu, llmx_group_sum)
    std::map<std::vector<int>, GroupPath> group_paths_;   // per group of device indices led by this backend
    std::map<std::pair<std::vector<const Backend*>, size_t>, std::vector<size_t>> picks_;   // member_order's permutation by listing and width, asked once
    unsigned summed_by_ = 0;   // transport_bit of each transport a timed group of this device picked, which collective_bytes_per_row counts
    // The backends of this kind `members` names, this one first, with their device indices, refused as join refuses them.
    std::vector<GpuBackend<Api>*> group_of(const std::vector<Backend*>& members, std::vector<int>& ids);
    // This backend's path for the group of `ids`, probed and reported on stderr the first time the group is named.
    GroupPath& group_path(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids);
    // Records the transports `path` picks on every member of `group`, and reports the times its timing gave on stderr, where it gave any.
    static void timed_group(const std::vector<GpuBackend<Api>*>& group, const std::vector<int>& ids, const GroupPath& path, const std::string& times);
};

template <class Api>
GpuKVStorage<Api>::GpuKVStorage(GpuBackend<Api>& owner, const std::vector<KVShape>& shapes, size_t max_tokens, KVType kt, KVType vt,
                             const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools)
    : BlockKVStorage(owner, Api::name, kBlockTokens, shapes, max_tokens, kt, vt, auxiliary_widths, auxiliary_pools) {}

// Dropping the old buffer is enough: its memory waits in the graves for the copies recorded before.
template <class Api>
void GpuKVStorage<Api>::retire(const BufferPtr&) {}

} // namespace
} // namespace backend

#include "backends/gpu/collective_core.hpp"
