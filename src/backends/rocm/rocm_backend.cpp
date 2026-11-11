// The ROCm backend (docs/ROCM.md): one HIP stream a device, every op an asynchronous launch on it, a ticket an event recorded on the stream.
// The host never waits inside a pass: it waits only where a caller asks for a ticket, a read, or a ring of host memory it reuses (docs/src/backends-rocm.md).
#include "backends/rocm/rocm_backend.hpp"

#include "backends/kv_storage.hpp"
#include "backends/rocm/rocm_device.hpp"
#include "quant/model_formats.hpp"
#include "quant/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace backend {
namespace {

// The dense F32 kernels, a code object for each target (kernels/f32.hip).
const unsigned char kF32Module[] = {
#include "rocm/f32.inc"
};
// The F16 class's quantized products on a block-int16 twin (kernels/int16.hip).
const unsigned char kInt16Module[] = {
#include "rocm/int16.inc"
};
// The same kernels on the 16-bit integer dot, compiled only for the targets with the instruction; a device it holds no code object for runs the module above.
const unsigned char kInt16DotModule[] = {
#include "rocm/int16dot.inc"
};

constexpr unsigned kThreads = 256;         // the kernels' block size (kernels/f32.hip)
constexpr size_t kBlockTokens = 64;        // tokens a KV block holds
constexpr size_t kSparseColumnsNarrow = 8, kSparseColumnsWide = 64, kSparseWideRows = 8, kSparseSoftmaxHeads = 4;  // columns of the narrow and wide column blocks (kernels/f32.hip, kSparseColumns and COLS of llmx_sparse_columns64), the rows from which the wide one runs, heads a grouped softmax block (kSparseHeads)
constexpr size_t kEvents = 1024;           // tickets in flight before a submission waits for the oldest
constexpr size_t kRingSlots = 4;           // a host ring's slots, each reused once the submission that last read it retires
constexpr size_t kArgBytes = size_t(1) << 20, kStageBytes = size_t(16) << 20;
constexpr size_t kGroupSignalBytes = 4096;  // a group member's signal block before its staging (kernels/f32.hip, GroupSignal)

class RocmBackend;
class RocmCollective;
class RocmHandoff;

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
        if (size + sizeof(T) > kBytes) throw std::logic_error("rocm: kernel arguments past their buffer");
        std::memcpy(bytes + size, &v, sizeof(T));
        size += sizeof(T);
    }
};

// A free that fails has nothing left to act on.
void free_memory(void* p, bool host) {
    const HipFn& f = hip();
    (void)(host ? f.hipHostFree(p) : f.hipFree(p));
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
bool register_host(void* memory, size_t bytes) {
    Registrations& r = registrations();
    std::lock_guard<std::mutex> lock(r.m);
    const auto it = r.held.find(memory);
    if (it != r.held.end()) {
        if (it->second.first != bytes) return false;
        ++it->second.second;
        return true;
    }
    if (hip().hipHostRegister(memory, bytes, hipHostRegisterDefault) != hipSuccess) return false;
    r.held[memory] = {bytes, 1};
    return true;
}

void unregister_host(void* memory) {
    Registrations& r = registrations();
    std::lock_guard<std::mutex> lock(r.m);
    const auto it = r.held.find(memory);
    if (it == r.held.end() || --it->second.second) return;
    (void)hip().hipHostUnregister(memory);
    r.held.erase(it);
}

// Memory of dropped buffers, freed once the work recorded before the drop is done, since hipFree does not wait for a launch or copy still queued that names it.
// The backend's own thread frees it: hipFree waits for every stream of the device, so a recording or waiting thread that freed would stop feeding its device until all the work queued after the drop had run (docs/src/backends-rocm.md).
struct Graves {
    std::mutex m;
    std::condition_variable wake;
    std::vector<std::pair<Ticket, std::pair<void*, bool>>> held;   // the ticket that covers the drop, and the memory
    size_t freeing = 0;                                             // graves the thread has taken and not yet freed
    const TicketLedger* tickets = nullptr;                          // null once the backend has drained its stream, when a drop frees at once
    hipStream_t stream = nullptr;                                   // the backend's stream, null once it is drained
    std::function<void()> flush;                                    // issues the backend's held stream operations, before a drop waits for its stream
    bool stop = false;                                              // set as the backend goes, which frees what is left and ends the thread
    std::thread thread;

    // The thread: a grave whose covering ticket is recorded has its work on the stream, which a stream synchronise ends without holding another thread's query or launch.
    void run(int device) {
        const HipFn& f = hip();
        (void)f.hipSetDevice(device);
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
            const hipStream_t s = stream;
            lock.unlock();
            if (s) (void)f.hipStreamSynchronize(s);
            for (const auto& r : ready) free_memory(r.first, r.second);
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

// Device memory from hipMalloc, pinned host memory the device reads in place (Memory::host_visible), or the caller's host memory registered with HIP (wrap_host).
// Dropped, HIP's memory goes to the backend's graves, while the caller's is unregistered at once, once the stream's copies naming it retire, since the caller may free it as soon as the view is gone (Backend::wrap_host).
class RocmBuffer final : public Buffer {
public:
    RocmBuffer(size_t bytes, bool host, std::shared_ptr<Graves> graves) : size_(bytes), host_(host), graves_(std::move(graves)) {
        if (!bytes) return;
        const HipFn& f = hip();
        // Whole words: the prompt tile reads a Q8_0 quarter as aligned words, the last of which may run past a buffer's final block.
        const size_t words = (bytes + 3) & ~size_t(3);
        if (host) hip_check(f.hipHostMalloc(&p_, words, hipHostMallocDefault), "hipHostMalloc");
        else hip_check(f.hipMalloc(&p_, words), "hipMalloc");
        device_ = p_;
        if (host) hip_check(f.hipHostGetDevicePointer(&device_, p_, 0), "hipHostGetDevicePointer");
    }
    // A view of memory register_host holds a count of.
    RocmBuffer(void* memory, size_t bytes, void* device, std::shared_ptr<Graves> graves)
        : p_(memory), device_(device), size_(bytes), host_(true), registered_(true), graves_(std::move(graves)) {}
    ~RocmBuffer() override {
        if (!p_) return;
        if (registered_) {
            {
                std::lock_guard<std::mutex> lock(graves_->m);
                if (graves_->stream) {
                    graves_->flush();
                    (void)hip().hipStreamSynchronize(graves_->stream);
                }
            }
            unregister_host(p_);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(graves_->m);
            if (graves_->tickets) {
                graves_->held.push_back({graves_->tickets->last() + 1, {p_, host_}});
                return;
            }
        }
        free_memory(p_, host_);
    }
    RocmBuffer(const RocmBuffer&) = delete;
    RocmBuffer& operator=(const RocmBuffer&) = delete;
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
    std::shared_ptr<Graves> graves_;
};

const RocmBuffer& as_rocm(const Buffer& b) {
    const auto* r = dynamic_cast<const RocmBuffer*>(&b);
    if (!r) throw std::runtime_error("rocm: a buffer of another backend");
    return *r;
}

// KV blocks on the device, kept and grown by BlockKVStorage; an old buffer a growth copied from is dropped, its memory kept until the copies retire (Graves).
class RocmKVStorage final : public BlockKVStorage {
public:
    RocmKVStorage(RocmBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens, KVType kt, KVType vt,
                  const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools);
private:
    void retire(const BufferPtr& old) override;
};

class RocmBackend final : public Backend {
public:
    explicit RocmBackend(int index) : dev_(rocm_device(index)) {
        graves_->tickets = &tickets_;
        const HipFn& f = hip();
        hip_check(f.hipSetDevice(index), "hipSetDevice");
        hip_check(f.hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        graves_->stream = stream_;
        graves_->flush = [this]() noexcept {
            try { flush(); } catch (const std::exception&) {}
        };
        hip_check(f.hipHostMalloc((void**)&fault_, sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
        *fault_ = 0;
        for (auto* set : {&events_, &starts_})
            for (auto& e : *set) hip_check(f.hipEventCreateWithFlags(&e, hipEventDefault), "hipEventCreateWithFlags");
        hip_check(f.hipModuleLoadData(&module_, kF32Module), "hipModuleLoadData");
        hip_check(f.hipModuleLoadData(&int16_module_, kInt16Module), "hipModuleLoadData");
        // The dot module loads only where the build compiled it for this device: that is the capability flag, read from the code object rather than from the device's name.
        dev_.caps.integer_dot = f.hipModuleLoadData(&dot_module_, kInt16DotModule) == hipSuccess;
        if (!dev_.caps.integer_dot) dot_module_ = nullptr;
        dev_.profile = profile_for(dev_.caps);
        // The prompt tiles and the row builds of four columns and more take the dot where there is one; the narrower quarter-layout builds measured faster on the plain module (docs/STATUS.md, ROCm wide decode rows).
        for (size_t k = 0; k < kKernels; ++k) {
            const bool plain = k == kRows16x1 || k == kRows16x2 || k == kRows16x2r4 || k == kRows8x1 || k == kRows8x2 || k == kRows8x2r4 || k == kTileReduce ||
                               k == kRouted16 || k == kRouted8;
            const hipModule_t m = k < kRows16x1 ? module_ : !plain && dot_module_ ? dot_module_ : int16_module_;
            // The fused moe_step's products are in the dot module alone; without it every step runs the base's ops.
            if (k >= kMoeStepUpQ4K && !dot_module_) continue;
            hip_check(f.hipModuleGetFunction(&kernels_[k], m, kKernelNames[k]), "hipModuleGetFunction");
        }
        int fixed = 0;
        hip_check(f.hipFuncGetAttribute(&fixed, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernels_[kSparseSoftmaxG]), "hipFuncGetAttribute");
        sparse_q_floats_ = dev_.caps.shared_bytes > (uint32_t)fixed ? (dev_.caps.shared_bytes - (uint32_t)fixed) / 4 : 0;
        for (size_t s = 0; s < kRingSlots; ++s) {
            args_.slots[s] = std::make_shared<RocmBuffer>(kArgBytes, true, graves_);
            stage_.slots[s] = std::make_shared<RocmBuffer>(kStageBytes, true, graves_);
            pass_.slots[s] = std::make_shared<RocmBuffer>(kArgBytes, false, graves_);
        }
        graves_->thread = std::thread([g = graves_.get(), index] { g->run(index); });
    }
    ~RocmBackend() override {
        const HipFn& f = hip();
        try { flush(); } catch (const std::exception&) {}
        (void)f.hipSetDevice(dev_.index);
        (void)f.hipStreamSynchronize(stream_);
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
        twin_ = {};
        staged_ = {};
        partials_ = {};
        scores_ = {};
        select_state_ = {};
        argmax_parts_ = {};
        args_ = {};
        stage_ = {};
        pass_ = {};
        uploads_ = {};
        if (module_) (void)f.hipModuleUnload(module_);
        if (int16_module_) (void)f.hipModuleUnload(int16_module_);
        if (dot_module_) (void)f.hipModuleUnload(dot_module_);
        for (auto e : events_) if (e) (void)f.hipEventDestroy(e);
        for (auto e : starts_) if (e) (void)f.hipEventDestroy(e);
        if (stream_) (void)f.hipStreamDestroy(stream_);
        if (fault_) (void)f.hipHostFree(fault_);
    }

    std::string identity() const override { return "rocm:" + std::to_string(dev_.index) + " " + dev_.caps.device + " " + dev_.pci_bus; }
    std::string pci_root() const override { return dev_.pci_root; }
    void set_threads(int) override {}
    int threads_available() const override { return 0; }
    // The recorded graves are freed first, so a reading after sync() counts dropped memory as free.
    std::optional<size_t> memory_available() const override {
        graves_->settle();
        size_t free = 0, total = 0;
        if (hip().hipSetDevice(dev_.index) != hipSuccess || hip().hipMemGetInfo(&free, &total) != hipSuccess) return std::nullopt;
        return free;
    }
    // HIP rounds each device allocation to its pages (2 MiB past 1 MiB) and keeps memory of its own, and the kernels' scratch grows with a pass, so the fit leaves the share Vulkan leaves.
    // A group's signal block is a fixed size beside the staging collective_bytes_per_row counts.
    size_t scratch_reserve(size_t free) const override { return ((size_t)256 << 20) + kGroupSignalBytes + free / 20; }
    std::unique_ptr<Collective> join(const std::vector<Backend*>& members, size_t rows, size_t width) override;
    std::unique_ptr<Handoff> handoff(const std::vector<Backend*>& readers) override;
    // A member's partial rows and a staging slot for each member's (RocmCollective).
    size_t collective_bytes_per_row(size_t members, size_t width) const override {
        return members ? size_mul(members + 1, size_mul(width, sizeof(float))) : 0;
    }
    bool supports_type(uint32_t type) const override { return type_kernels(type) != nullptr; }
    bool implements(Op op) const override {
        return op == Op::norm_rope_partial || op == Op::sigmoid_mul || op == Op::gated_rms_norm || op == Op::causal_conv_silu ||
               op == Op::gated_delta_rule || op == Op::routed_experts || op == Op::group_rms_norm || op == Op::silu_scaled || op == Op::mix_streams ||
               op == Op::inject_streams || op == Op::signed_sqrt_gate || op == Op::dilated_conv_silu_add || op == Op::select_kv || op == Op::sparse_attention || op == Op::argmax_rows || op == Op::embed_ids ||
               op == Op::hc_coeffs || op == Op::hc_collapse_norm || op == Op::hc_expand || op == Op::sparse_attention_sink || op == Op::topk_rows ||
               op == Op::candidate_blocks || op == Op::compress_pool || op == Op::indexer_scores || op == Op::model_format_qdq || op == Op::model_format_pack || op == Op::pool_write ||
               op == Op::sqrt_softplus_routing || op == Op::silu_mul_clamp || op == Op::rope_forms;
    }
    // A generated token is a class; a prompt's extent is classed by the tiles it reaches, product, routed and attention (docs/src/backends-rocm.md, batch invariance).
    size_t row_class(size_t extent) const override {
        if (extent == 1) return 1;
        constexpr size_t entries = sizeof(TwinKernels::tile_from) / sizeof(size_t);   // a type's tile extents, by role and width
        size_t routed = 0;   // the routed builds whose tile the extent reaches; their entries are ordered, so the count names the set
        for (const TypeKernels& t : kTypes)
            if (t.twin)
                for (const auto& role : t.twin->tile_from)
                    for (size_t from : role) routed += extent >= from;
        return 2 + (extent >= dev_.profile.attention_tile_rows ? 1 : 0) + 2 * (routed + (entries * std::size(kTypes) + 1) * split_tiles_of(extent));
    }
    // The column tiles the tile's inner split is chosen for: the prompt's own, counted up to 512 rows, as the Vulkan tile counts them.
    static size_t split_tiles_of(size_t extent) { return (std::min<size_t>(extent, 512) + kTileRows - 1) / kTileRows; }

    // HIP allocates on the calling thread's current device, and a backend may be called from several threads.
    BufferPtr alloc(size_t bytes, Memory where) override {
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        std::shared_ptr<RocmBuffer> b;
        try {
            b = std::make_shared<RocmBuffer>(bytes, where == Memory::host_visible, graves_);
        } catch (const std::runtime_error&) {
            // Dropped memory whose covering ticket is recorded goes back first, which waits for the whole stream; a drop since the last ticket waits for the next.
            // Nothing is recorded here, since alloc may run beside the thread that records.
            if (!graves_->settle()) throw;
            b = std::make_shared<RocmBuffer>(bytes, where == Memory::host_visible, graves_);
        }
        forget_twins();
        if (!bytes) return b;
        // KV growth and the model read a fresh buffer as zeros.
        if (where == Memory::host_visible) std::memset(b->data(), 0, bytes);
        else hold_fill(b->data(), bytes);
        return b;
    }
    // One registration serves every device's view of the memory; null where HIP refuses it, so the caller writes through staging.
    BufferPtr wrap_host(void* memory, size_t bytes) override {
        if (!memory || !bytes) return nullptr;
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        if (!register_host(memory, bytes)) return nullptr;
        void* device = nullptr;
        if (hip().hipHostGetDevicePointer(&device, memory, 0) != hipSuccess) {
            unregister_host(memory);
            return nullptr;
        }
        return std::make_shared<RocmBuffer>(memory, bytes, device, graves_);
    }
    BufferPtr adopt(const void* src, size_t bytes) override {
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        auto b = std::make_shared<RocmBuffer>(bytes, false, graves_);
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
        if (t > kEvents) wait(t - kEvents);
        if (!started_) hip_check(hip().hipEventRecord(starts_[i], stream_), "hipEventRecord");
        hip_check(hip().hipEventRecord(events_[i], stream_), "hipEventRecord");
        tickets_.take();
        graves_->recorded();
        opened_ = started_ = false;
        for (Ring* r : {&args_, &stage_, &pass_}) r->tag(t);
        return t;
    }
    double busy_ms(Ticket a, Ticket b) override {
        if (a > b || b > tickets_.last() || a + kEvents <= tickets_.last()) return -1.0;
        double sum = 0;
        for (Ticket t = a + 1; t <= b; ++t) {
            float ms = 0;
            if (hip().hipEventElapsedTime(&ms, starts_[t % kEvents], events_[t % kEvents]) != hipSuccess) return -1.0;
            sum += ms;
        }
        return sum;
    }
    Ticket last_ticket() const noexcept override { return tickets_.last(); }
    void wait(Ticket t) noexcept override {
        if (tickets_.settled(t)) return;
        if (t + kEvents <= tickets_.last()) return tickets_.retired(t);
        std::atomic<uint32_t>& held = syncing_[t % kEvents];
        held.fetch_add(1, std::memory_order_acq_rel);
        const hipError_t e = hip().hipEventSynchronize(events_[t % kEvents]);
        held.fetch_sub(1, std::memory_order_acq_rel);
        if (e != hipSuccess) remember(e, "a submission");
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
        const hipError_t e = hip().hipEventQuery(events_[t % kEvents]);
        if (e == hipErrorNotReady) return false;
        if (e != hipSuccess) remember(e, "a submission");
        tickets_.retired(t);
        return true;
    }
    void sync() noexcept override {
        if (opened_ || graves_->unrecorded()) {
            try { record(); } catch (const std::exception&) {}
        }
        wait(tickets_.last());
    }
    // Results read in place after wait(t): the first failure the device reported throws.
    void raise(Ticket t) override {
        (void)t;
        raise_failed();
    }

    void read(const Buffer& src_b, size_t off, void* dst, size_t bytes) override {
        const RocmBuffer& src = as_rocm(src_b);
        span(src, off, bytes);
        if (!bytes) return;
        if (src.host_ptr()) {
            sync();
            raise_failed();
            std::memcpy(dst, (const uint8_t*)src.data() + off, bytes);
            return;
        }
        for (size_t done = 0; done < bytes;) {
            const size_t n = std::min(bytes - done, kStageBytes);
            void* st = stage_.take(*this, n);
            hold_copy(st, (const uint8_t*)src.data() + off + done, n, hipMemcpyDeviceToHost);
            wait(submit());
            raise_failed();
            std::memcpy((uint8_t*)dst + done, st, n);
            done += n;
        }
    }
    void write(Buffer& dst_b, size_t off, const void* src, size_t bytes) override {
        const RocmBuffer& dst = as_rocm(dst_b);
        span(dst, off, bytes);
        forget_twins();
        for (size_t done = 0; done < bytes;) {
            const size_t n = std::min(bytes - done, kStageBytes);
            void* st = stage_.take(*this, n);
            std::memcpy(st, (const uint8_t*)src + done, n);
            hold_copy((uint8_t*)dst.data() + off + done, st, n, hipMemcpyHostToDevice);
            done += n;
        }
    }
    void copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off, size_t bytes) override {
        const RocmBuffer& dst = as_rocm(dst_b);
        const RocmBuffer& src = as_rocm(src_b);
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
        hold_copy((uint8_t*)dst.data() + dst_off, (const uint8_t*)src.data() + src_off, bytes, hipMemcpyDefault);
        forget_written(written, floats);
    }

    void matmul(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs, Dtype dtype) override {
        product(weight, X, Y, nin, nout, nbatch, false, runs, dtype, true);
    }
    void matmul_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs, Dtype dtype) override {
        product(weight, X, Y, nin, nout, nbatch, true, runs, dtype, true);
    }
    // One twin of X serves every projection of the group.
    void matmul_group(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nbatch, RowRuns runs, Dtype dtype) override {
        // Up to three integer projections into device memory take one twin and one tile call, as the Vulkan tile takes them.
        bool together = projections.size() > 1 && projections.size() <= 3 && nin && nbatch;
        for (const Projection& p : projections)
            together = together && p.weight.data.buffer && integer(p.weight, dtype) && p.weight.type == projections.begin()->weight.type && p.rows &&
                       !as_rocm(*p.out.buffer).host_ptr();
        if (together) {
            Proj proj[3];
            size_t n = 0;
            span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nin, nbatch), sizeof(float)));
            for (const Projection& p : projections) {
                check_weight(p.weight, nin, "matrix");
                span(*p.weight.data.buffer, p.weight.data.offset * sizeof(float), quant::row_bytes(p.weight.type, nin, p.rows));
                span(*p.out.buffer, p.out.offset * sizeof(float), size_mul(size_mul(p.rows, nbatch), sizeof(float)));
                proj[n++] = {bytes(p.weight.data), f32(p.out), p.rows};
            }
            return product_twin(*type_kernels(projections.begin()->weight.type)->twin, proj, n, X, nin, nbatch, false, runs, true, dtype == Dtype::int8);
        }
        bool twinned = false;
        for (const Projection& p : projections) {
            if (!p.weight.data.buffer) throw std::runtime_error("rocm: projection without storage");
            product(p.weight, X, p.out, nin, p.rows, nbatch, false, runs, dtype, !twinned);
            twinned = twinned || integer(p.weight, dtype);
        }
    }
    std::vector<Dtype> native_dtypes() const override { return {Dtype::f16, Dtype::f32, Dtype::int8}; }
    std::string dtype_path(Dtype dtype) const override {
        if (dtype == Dtype::f32) return "f32";
        // The types without twins take their exact products in every class, wider than asked, and say so.
        std::string exact = "F32 weights";
        for (const TypeKernels& t : kTypes)
            if (!t.twin && t.type != quant::GGML_TYPE_F32) exact += std::string(exact == "F32 weights" ? ", " : "/") + t.name;
        exact += exact == "F32 weights" ? "" : " rows";
        std::string twins;
        for (const TypeKernels& t : kTypes)
            if (t.twin) twins += std::string(twins.empty() ? "" : "/") + t.name;
        const std::string form = dtype == Dtype::int8 ? "block-int8 (" : "block-int16 (";
        return form + twins + (dev_.caps.integer_dot ? std::string(" rows, and tiles on the ") + (dtype == Dtype::int8 ? "8" : "16") + "-bit dot), f32 (" : " rows and tiles), f32 (") + exact + ")";
    }
    void embed(Slice dst, WeightView weight, size_t nin, size_t nrows, const uint32_t* ids, size_t count) override {
        const TypeKernels& kinds = check_weight(weight, nin, "embedding");
        if (!count) return;
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= nrows) throw std::runtime_error("rocm: embedding row out of range");
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nrows));
        gather(kinds.embed, dst, bytes(weight.data), nin, (const unsigned*)arg(ids, count * sizeof(uint32_t)), count, nrows);
    }
    // embed's gather with the ids read where the device holds them; the kernel writes a zero row for an id of `nrows` or more.
    void embed_ids(Slice dst, WeightView weight, size_t nin, size_t nrows, CSlice ids, size_t count) override {
        const TypeKernels& kinds = check_weight(weight, nin, "embedding");
        if (!count) return;
        if (nrows > UINT32_MAX) throw std::runtime_error("rocm: an embedding of more rows than 32-bit ids name");
        if (!ids.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nrows));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(count, sizeof(float)));
        gather(kinds.embed, dst, bytes(weight.data), nin, (const unsigned*)f32(ids), count, nrows);
    }
    // For each row of `n` logits the id of the largest (Backend::argmax_rows), in slices of the profile's argmax_slice ids merged a row; no ids is refused, as Vulkan refuses it.
    void argmax_rows(Slice ids, CSlice logits, size_t rows, size_t n, CSlice after = {}) override {
        if (!rows) return;
        if (!n || n > UINT32_MAX) throw std::runtime_error("rocm: argmax over no ids or more than 32 bits name");
        if (!ids.buffer || !logits.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*logits.buffer, logits.offset * sizeof(float), size_mul(size_mul(rows, n), sizeof(float)));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(rows, sizeof(float)));
        if (after.buffer) span(*after.buffer, after.offset * sizeof(float), size_mul(rows, sizeof(float)));
        const size_t slice = std::max<size_t>(1, std::min(dev_.profile.argmax_slice, n)), nslice = (n + slice - 1) / slice;
        if (size_mul(rows, nslice) > UINT32_MAX) throw std::runtime_error("rocm: argmax over too many slices");
        const float* lp = f32(logits);
        unsigned* ip = (unsigned*)f32(ids);
        const unsigned* ap = after.buffer ? (const unsigned*)f32(after) : nullptr;
        const size_t need = size_mul(size_mul(rows, nslice), 2 * sizeof(float));
        if (nslice > 1 && (!argmax_parts_ || argmax_parts_->size() < need)) argmax_parts_ = alloc(need, Memory::device);
        float* pp = nslice > 1 ? (float*)as_rocm(*argmax_parts_).data() : nullptr;
        unsigned nn = (unsigned)n, sl = (unsigned)slice, ns = (unsigned)nslice;
        const Args args(lp, ip, pp, nn, sl, ns, ap);
        launch(kArgmaxSlices, (unsigned)(rows * nslice), 1, args);
        if (nslice > 1) {
            const Args margs(pp, ip, nn, ns, ap);
            launch(kArgmaxMerge, (unsigned)rows, 1, margs, 1, 64);
        }
    }
    // argmax_rows' slices into partials, then a row's partials into its candidate (Backend::argmax_part).
    void argmax_part(Slice cands, CSlice logits, size_t rows, size_t n, size_t first, size_t slot, size_t slots) override {
        if (!rows) return;
        if (!n || slot >= slots || first + n > kExactIds) throw std::runtime_error("rocm: argmax_part over no ids, ids a float does not hold, or a slot past its row");
        if (!cands.buffer || !logits.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*logits.buffer, logits.offset * sizeof(float), size_mul(size_mul(rows, n), sizeof(float)));
        span(*cands.buffer, cands.offset * sizeof(float), size_mul(size_mul(rows, slots), 3 * sizeof(float)));
        const size_t slice = std::max<size_t>(1, std::min(dev_.profile.argmax_slice, n)), nslice = (n + slice - 1) / slice;
        if (size_mul(rows, nslice) > UINT32_MAX) throw std::runtime_error("rocm: argmax over too many slices");
        const size_t need = size_mul(size_mul(rows, nslice), 2 * sizeof(float));
        if (!argmax_parts_ || argmax_parts_->size() < need) argmax_parts_ = alloc(need, Memory::device);
        float* pp = (float*)as_rocm(*argmax_parts_).data();
        unsigned* none = nullptr;
        const unsigned* no_after = nullptr;
        unsigned nn = (unsigned)n, sl = (unsigned)slice, ns = (unsigned)nslice, f = (unsigned)first, at = (unsigned)slot, w = (unsigned)slots;
        const Args args(f32(logits), none, pp, nn, sl, ns, no_after);
        launch(kArgmaxSlices, (unsigned)(rows * nslice), 1, args);
        const Args pargs((const float*)pp, f32(cands), ns, f, at, w);
        launch(kArgmaxPart, (unsigned)rows, 1, pargs, 1, 64);
    }
    void argmax_pick(Slice ids, CSlice cands, size_t rows, size_t slots, size_t n, CSlice after = {}) override {
        if (!rows) return;
        if (!slots || n > kExactIds) throw std::runtime_error("rocm: argmax_pick over no candidates or ids a float does not hold");
        if (!ids.buffer || !cands.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*cands.buffer, cands.offset * sizeof(float), size_mul(size_mul(rows, slots), 3 * sizeof(float)));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(rows, sizeof(float)));
        if (after.buffer) span(*after.buffer, after.offset * sizeof(float), size_mul(rows, sizeof(float)));
        const unsigned* ap = after.buffer ? (const unsigned*)f32(after) : nullptr;
        unsigned nn = (unsigned)n, w = (unsigned)slots;
        const Args args(f32(cands), (unsigned*)f32(ids), nn, w, ap);
        launch(kArgmaxPick, (unsigned)rows, 1, args, 1, 64);
    }
    void gather_rows(Slice dst, CSlice src, size_t width, const uint32_t* rows, size_t count) override {
        if (count && !rows) throw std::runtime_error("rocm: gather without rows");
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
        const Args args(d, s, wp, in, st, eps, q, td, map);
        launch_writing({{d, size_mul(rows, std::max<size_t>(n, stride))}}, twin_build(k, map), (unsigned)rows, 1, args, 1, kNormThreads);
        keep(t, d, rows * n);
    }
    void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride, size_t heads, size_t head_dim,
                           size_t rope_dim, CSlice w, float eps, CSlice cos, CSlice sin, const uint32_t* pos, const RopeForm& form, RopeStreams streams) override {
        if (!heads || !head_dim || !rope_dim || rope_dim % 2 || rope_dim > head_dim || form.offset > head_dim - rope_dim)
            throw std::runtime_error("rocm: invalid partial rope dimensions");
        if (!form.plain() && head_dim > 8 * kThreads) throw std::runtime_error("rocm: a rope form over a head wider than " + std::to_string(8 * kThreads));
        if (!streams.count || (streams.count > 1 && !streams.of)) throw std::runtime_error("rocm: a rope of several streams without their pairs");
        if (rows && !pos) throw std::runtime_error("rocm: partial rope without positions");
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
            throw std::runtime_error("rocm: partial rope in place over heads that are not contiguous");
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
            if (streams.of[i] >= streams.count) throw std::runtime_error("rocm: a rope pair of a stream past the row's");
        const unsigned char* of = streams.count > 1 ? (const unsigned char*)pass_arg(streams.of, half) : nullptr;
        int count = (int)streams.count;
        // The adjacent, offset and inverse forms take their own kernel, which forms the norm and the pairs as the CPU does (kernels/f32.hip, llmx_norm_rope_form).
        if (!form.plain()) {
            unsigned hdu = (unsigned)head_dim, hfu = (unsigned)half, cu = (unsigned)count, adj = form.adjacent, off = (unsigned)form.offset, inv = form.inverse;
            const Args args(d, s, dst_stride, ss, shs, hdu, hfu, wp, eps, c, sn, p, of, cu, adj, off, inv);
            return launch(kNormRopeForm, (unsigned)rows, (unsigned)heads, args);
        }
        const Args args(d, s, dst_stride, ss, shs, hd, hf, wp, eps, c, sn, p, of, count);
        launch(kNormRopeStreams, (unsigned)rows, (unsigned)heads, args);
    }
    void sigmoid_mul(Slice dst, CSlice x, CSlice gate, size_t rows, size_t heads, size_t dim, size_t gate_stride, size_t gate_head_stride, RowRuns runs) override {
        if (!heads || !dim) throw std::runtime_error("rocm: invalid sigmoid gate dimensions");
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
        const Args args(d, xp, g, nn, wd, dd, gs, ghs, t.q, t.d, t.level, t.map, t.major);
        launch_writing({{d, n}}, kSigmoidMul, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, n);
    }
    void gated_rms_norm(Slice dst, CSlice x, CSlice z, CSlice w, size_t rows, size_t heads, size_t dim, float eps, RowRuns runs, NormGate gate) override {
        if (!heads || !dim) throw std::runtime_error("rocm: invalid gated norm dimensions");
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
        const Args args(d, xp, zp, wp, dm, eps, sig, t.q, t.d, t.level, t.map, t.major);
        launch_writing({{d, n}}, kGatedRmsNorm, (unsigned)size_mul(rows, heads), 1, args);
        keep(t.t, d, n);
    }
    // The view count, then five words a view (first row, rows, history length, source slot, destination slot), as the state kernels read them.
    static std::vector<unsigned> state_table(const StateView* views, size_t n_views) {
        std::vector<unsigned> tab{(unsigned)n_views};
        size_t row0 = 0;
        for (size_t i = 0; i < n_views; ++i) {
            const StateView& v = views[i];
            if (v.storage != views[0].storage) throw std::runtime_error("rocm: state views of two storages in one call");
            if (v.length > UINT32_MAX || row0 + v.nq > UINT32_MAX) throw std::runtime_error("rocm: a state view past 32-bit rows");
            tab.insert(tab.end(), {(unsigned)row0, (unsigned)v.nq, (unsigned)v.length, (unsigned)v.src, (unsigned)v.dst});
            row0 += v.nq;
        }
        return tab;
    }
    // A block a (view, V head, 32 V columns), in the Vulkan shader's order (kernels/f32.hip, llmx_gated_delta_rule).
    void gated_delta_rule(Slice out, CSlice qkv, CSlice alpha, CSlice b, CSlice a, CSlice dt_bias, size_t layer, const StateView* views, size_t n_views) override {
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        const std::vector<unsigned> tab = state_table(views, n_views);
        const StateStorage& ss = *views[0].storage;
        const StateShape& sh = ss.shape();
        const size_t Hk = sh.k_heads, Hv = sh.v_heads, Dk = sh.k_dim, Dv = sh.v_dim, C = sh.channels();
        // A column's K rows sit 16 in each of eight lanes, and a block stages 128 of each q and k row.
        if (Dk > 128) throw std::runtime_error("rocm: a linear-attention K head wider than 128");
        span(*out.buffer, out.offset * sizeof(float), size_mul(size_mul(rows, Hv * Dv), sizeof(float)));
        span(*qkv.buffer, qkv.offset * sizeof(float), size_mul(size_mul(rows, C), sizeof(float)));
        for (const CSlice& g : {alpha, b}) span(*g.buffer, g.offset * sizeof(float), size_mul(size_mul(rows, Hv), sizeof(float)));
        for (const CSlice& g : {a, dt_bias}) span(*g.buffer, g.offset * sizeof(float), size_mul(Hv, sizeof(float)));
        const size_t blocks = (Dv + 31) / 32, grid = size_mul(size_mul(n_views, Hv), blocks);
        if (grid > INT32_MAX || C > UINT32_MAX) throw std::runtime_error("rocm: a delta rule past the launch's grid");
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
        // The short build where every view holds at most 8 rows, as the Vulkan backend chooses it; the staged count changes no bits (kernels/f32.hip).
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
        if (chunks > 65535 || C > UINT32_MAX) throw std::runtime_error("rocm: a conv past the launch's grid");
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
    // Where delta_holds, the qkv and gate products and then one launch of llmx_delta_step (docs/src/backends-rocm.md, delta_step), every slot but conv the base's bits.
    void delta_step(const DeltaStep& d, CSlice mixed, size_t nin, size_t rows, size_t layer, const StateView* views, size_t n_views, RowRuns runs = {},
                    Dtype dtype = Dtype::f16) override {
        if (!delta_holds(d, nin, rows, layer, views, n_views)) return Backend::delta_step(d, mixed, nin, rows, layer, views, n_views, runs, dtype);
        check_runs(rows, runs);
        const StateStorage& ss = *views[0].storage;
        const StateShape& sh = ss.shape();
        const size_t Hv = sh.v_heads, C = sh.channels(), V = size_mul(Hv, sh.v_dim), n = size_mul(rows, V);
        span(*mixed.buffer, mixed.offset * sizeof(float), size_mul(size_mul(rows, nin), sizeof(float)));
        for (const Projection* p : {&d.alpha, &d.beta}) {
            check_weight(p->weight, nin, "delta step matrix");
            span(*p->weight.data.buffer, p->weight.data.offset * sizeof(float), quant::row_bytes(p->weight.type, nin, Hv));
            span(*p->out.buffer, p->out.offset * sizeof(float), size_mul(size_mul(rows, Hv), sizeof(float)));
        }
        span(*d.conv_w.buffer, d.conv_w.offset * sizeof(float), size_mul(size_mul(C, kConvTaps), sizeof(float)));
        for (const CSlice& g : {d.a, d.dt_bias}) span(*g.buffer, g.offset * sizeof(float), size_mul(Hv, sizeof(float)));
        span(*d.norm.buffer, d.norm.offset * sizeof(float), size_mul(sh.v_dim, sizeof(float)));
        // Generated rows give the same bits in any group, so they take qkv and gate as one; a prompt's tile splits by the group's outputs, so its rows take the base's two products.
        bool generated = runs.n ? true : rows == 1;
        for (size_t r = 0; r < runs.n; ++r) generated = generated && runs.runs[r].extent == 1;
        unordered(true);
        if (generated) matmul_group({d.qkv, d.gate}, mixed, nin, rows, runs, dtype);
        else {
            matmul(d.qkv.weight, mixed, d.qkv.out, nin, d.qkv.rows, rows, runs, dtype);
            matmul(d.gate.weight, mixed, d.gate.out, nin, d.gate.rows, rows, runs, dtype);
        }
        unordered(false);
        span(*d.out.buffer, d.out.offset * sizeof(float), size_mul(n, sizeof(float)));
        if (!delta_counter_) {
            delta_counter_ = alloc(kDeltaCounters * sizeof(unsigned), Memory::device);
            hold_fill(as_rocm(*delta_counter_).data(), kDeltaCounters * sizeof(unsigned));
        }
        const std::vector<unsigned> tab = state_table(views, n_views);
        StreamTwin t = stream_twin(d.out, rows, V, runs, sh.v_dim % 32 == 0);
        const DeltaStepArgs A{f32(d.out), f32(d.qkv.out), f32(d.conv_w), f32(mixed), bytes(d.alpha.weight.data), bytes(d.beta.weight.data), f32(d.alpha.out),
                              f32(d.beta.out), f32(d.a), f32(d.dt_bias), f32(Slice{&ss.layer(layer), 0}), (const unsigned*)arg(tab.data(), tab.size() * sizeof(unsigned)),
                              f32(d.gate.out), f32(d.norm), (unsigned*)as_rocm(*delta_counter_).data(), (unsigned)C, (unsigned)sh.k_heads, (unsigned)Hv, (unsigned)nin,
                              ss.slot_floats(layer), size_mul(Hv, sh.matrix_floats()), (float)(1.0 / std::sqrt((double)sh.k_dim)), kL2NormEps, d.eps, t.q, t.d, t.level,
                              t.major, t.map, (unsigned)sh.k_dim, (unsigned)sh.v_dim};
        launch_writing({{A.o, n}, {A.ao, size_mul(rows, Hv)}, {A.bo, size_mul(rows, Hv)}}, kDeltaStep, (unsigned)size_mul(n_views, Hv), 1, Args(A));
        keep(t.t, A.o, n);
    }
    // llmx_delta_step's limits: K and V heads up to kDeltaHead wide in multiples of 4, rows a view, views a launch and its (view, K head) counters.
    static constexpr size_t kDeltaHead = 128, kDeltaRows = 8, kDeltaViews = 64, kDeltaCounters = 4096;
    static_assert(kConvTaps == 4, "llmx_delta_step's conv takes four taps");
    // Whether a step has the shape llmx_delta_step holds (docs/src/backends-rocm.md, delta_step), with F32 alpha and beta matrices and state rows on four-float boundaries.
    bool delta_holds(const DeltaStep& d, size_t nin, size_t rows, size_t layer, const StateView* views, size_t n_views) const {
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
        if (d.alpha.weight.type != quant::GGML_TYPE_F32 || d.beta.weight.type != quant::GGML_TYPE_F32) return false;
        return d.alpha.rows == sh.v_heads && d.beta.rows == sh.v_heads && d.gate.rows == size_mul(sh.v_heads, sh.v_dim) && d.qkv.rows == sh.channels();
    }
    // The Flash-Next stream ops (kernels/f32.hip), plain F32 outputs whose products make their own twins; every refusal comes before any launch.
    void group_rms_norm(Slice dst, CSlice src, CSlice w, size_t rows, size_t groups, size_t dim, float eps, RowRuns runs) override {
        if (!groups || !dim || !std::isfinite(eps) || eps <= 0.0f) throw std::runtime_error("rocm: invalid grouped norm dimensions or epsilon");
        if (!rows) return;
        const size_t width = size_mul(groups, dim), n = size_mul(rows, width), blocks = size_mul(rows, groups);
        for (const CSlice& v : {CSlice(dst), src}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(width, sizeof(float)));
        if (overlaps(dst, n, src, n) && !(dst.buffer == src.buffer && dst.offset == src.offset)) throw std::runtime_error("rocm: grouped norm output must be disjoint or exactly in place");
        if (overlaps(dst, n, w, width)) throw std::runtime_error("rocm: grouped norm output overlaps its weights");
        if (blocks > INT32_MAX || dim > UINT32_MAX) throw std::runtime_error("rocm: a grouped norm past the launch's grid");
        float* d = f32(dst);
        const float* s = f32(src);
        const float* wp = f32(w);
        unsigned g = (unsigned)groups, dm = (unsigned)dim;
        StreamTwin t = stream_twin(dst, rows, width, runs, dim % 32 == 0);
        // Four values a thread only where every operand lies on 16 bytes, since a slice's offset may be any float.
        int four = dim % 4 == 0 && ((uintptr_t)d | (uintptr_t)s | (uintptr_t)wp) % 16 == 0;
        const Args args(d, s, wp, g, dm, eps, t.q, t.d, t.level, t.map, t.major, four);
        // Slices of a group, each repeating its reduction, up to what gives four blocks a compute unit (Vulkan's split, fc6f53415); one in place, where a slice's writes would meet another's reads.
        const size_t fill = (4 * size_t(std::max(1u, dev_.caps.compute_units)) + blocks - 1) / blocks;
        const size_t slices = dst.buffer == src.buffer && dst.offset == src.offset ? 1 : std::max<size_t>(1, std::min((dim + 1023) / 1024, fill));
        launch_writing({{d, n}}, kGroupRmsNorm, (unsigned)blocks, (unsigned)slices, args);
        keep(t.t, d, n);
    }
    void silu_scaled(Slice dst, CSlice src, size_t rows, size_t width, float scale, RowRuns runs) override {
        if (!width || !std::isfinite(scale)) throw std::runtime_error("rocm: invalid scaled SiLU dimensions or scale");
        if (!rows) return;
        const size_t n = size_mul(rows, width);
        for (const CSlice& v : {CSlice(dst), src}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        if (overlaps(dst, n, src, n) && !(dst.buffer == src.buffer && dst.offset == src.offset)) throw std::runtime_error("rocm: scaled SiLU output must be disjoint or exactly in place");
        float* d = f32(dst);
        const float* s = f32(src);
        unsigned long long nn = n;
        StreamTwin t = stream_twin(dst, rows, width, runs);
        const Args args(d, s, nn, scale, t.q, t.d, t.level, t.map, t.major);
        launch_writing({{d, n}}, kSiluScaled, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, n);
    }
    void mix_streams(Slice dst, CSlice src, CSlice gate, size_t rows, size_t streams, size_t dim, RowRuns runs) override {
        if (!streams || !dim) throw std::runtime_error("rocm: invalid stream mix dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(streams, dim)), out_n = size_mul(rows, dim);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(out_n, sizeof(float)));
        for (const CSlice& v : {src, gate}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        if (overlaps(dst, out_n, src, n) || overlaps(dst, out_n, gate, n)) throw std::runtime_error("rocm: stream mix output must be disjoint from its inputs");
        if (streams > UINT32_MAX || dim > UINT32_MAX) throw std::runtime_error("rocm: a stream mix past 32-bit streams or widths");
        float* d = f32(dst);
        const float* s = f32(src);
        const float* g = f32(gate);
        unsigned long long nn = out_n;
        unsigned st = (unsigned)streams, dm = (unsigned)dim;
        StreamTwin t = stream_twin(dst, rows, dim, runs);
        const Args args(d, s, g, nn, st, dm, t.q, t.d, t.level, t.map, t.major);
        launch_writing({{d, out_n}}, kMixStreams, (unsigned)((out_n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, out_n);
    }
    // The hyper-connection mix as persistent launches (kernels/int16.hip, llmx_hc_mix) for generated rows over 8-bit activations on a 64-lane wave, up to 8 columns a launch; the chain of ops otherwise.
    // A column gives the one-column launch's bits in any build, so verify rows mix as the decode row would.
    void hyper_mix(Slice out, CSlice x, const HyperMix& m, size_t rows, size_t streams, size_t dim, size_t low_rank, float eps, RowRuns runs, Dtype dtype) override {
        const bool inj = m.inject.data.buffer != nullptr;
        const size_t width = streams * dim;
        bool generated = runs.n ? rows > 0 : rows == 1;
        for (size_t r = 0; r < runs.n; ++r) generated = generated && runs.runs[r].extent == 1;
        bool fits = generated && dtype == Dtype::int8 && dev_.caps.subgroup_size == 64 && streams && dim % 32 == 0 && low_rank && low_rank % 32 == 0 &&
                    low_rank <= 512 && size_mul(width, kHcColumns) <= INT32_MAX && std::isfinite(eps) && eps > 0.0f && m.down.type == quant::GGML_TYPE_Q8_0 &&
                    m.up.type == quant::GGML_TYPE_Q8_0 && (!inj || m.inject.type == quant::GGML_TYPE_F32) && m.down.data.buffer && m.up.data.buffer;
        if (fits)
            for (const CSlice& v : {CSlice(out), x, m.norm, CSlice(m.normed), CSlice(m.low), CSlice(m.gates), inj ? CSlice(m.inject_gates) : x})
                if (!v.buffer || as_rocm(*v.buffer).host_ptr()) fits = false;
        // A launch of n columns is ceil(low rank / 4) down blocks and a block an inject output of a column.
        const auto grid = [&](size_t n) { return (unsigned)((low_rank + 3) / 4 + (inj ? streams * n : 0)); };
        if (fits)
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
        // The launch's scratch: the two barrier counters first, which the allocation zeroes and every launch leaves at zero, then each stream's root, the twin's scales and the twin, for a launch's columns.
        // The counters keep their offset whatever the shape, so a launch on scratch a wider mix used reads zeros there, never that mix's twin bytes.
        const size_t cols = std::min(rows, kHcColumns), rr_at = 64, nd_at = rr_at + (cols * streams * sizeof(float) + 15) / 16 * 16,
                     nq_at = nd_at + (cols * width / 32 * sizeof(float) + 15) / 16 * 16, need = nq_at + cols * width;
        if (!hc_scratch_ || hc_scratch_->size() < need) hc_scratch_ = alloc(need, Memory::device);
        uint8_t* sc = (uint8_t*)as_rocm(*hc_scratch_).data();
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
    // The kernel's argument, laid out as kernels/int16.hip's HcMix.
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
    // Whether a grid of the build is resident at once, which its barriers need: the blocks a compute unit holds times the units, read once a build; a grid past it is said once and runs the chain.
    bool hc_resident(int k, unsigned grid) {
        const size_t i = k == kHcMix ? 0 : 1;
        if (hc_per_cu_[i] < 0) {
            int n = 0;
            // HIP finds the module's function through the calling thread's device, which another backend's call may have changed.
            hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
            hip_check(hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernels_[k], kThreads, 0), "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
            hc_per_cu_[i] = n;
        }
        const bool ok = size_t(grid) <= size_t(hc_per_cu_[i]) * dev_.caps.compute_units;
        if (!ok && !hc_refused_) {
            hc_refused_ = true;
            std::fprintf(stderr, "rocm: the hyper-connection mix's %u workgroups exceed the %d a compute unit times %u units resident on %s; it runs as separate launches\n", grid,
                         hc_per_cu_[i], dev_.caps.compute_units, identity().c_str());
        }
        return ok;
    }
    BufferPtr hc_scratch_;
    int hc_per_cu_[2] = {-1, -1};
    bool hc_refused_ = false;
    void inject_streams(Slice dst, CSlice block, CSlice gate, size_t rows, size_t streams, size_t dim) override {
        if (!streams || !dim) throw std::runtime_error("rocm: invalid stream injection dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(streams, dim)), bn = size_mul(rows, dim), gn = size_mul(rows, streams);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*block.buffer, block.offset * sizeof(float), size_mul(bn, sizeof(float)));
        span(*gate.buffer, gate.offset * sizeof(float), size_mul(gn, sizeof(float)));
        if (overlaps(dst, n, block, bn) || overlaps(dst, n, gate, gn) || overlaps(block, bn, gate, gn)) throw std::runtime_error("rocm: stream injection operands must be disjoint");
        const size_t chunks = (dim + kThreads - 1) / kThreads;
        if (gn > INT32_MAX / chunks || dim > UINT32_MAX) throw std::runtime_error("rocm: a stream injection past the launch's grid");
        float* d = f32(dst);
        const float* b = f32(block);
        const float* g = f32(gate);
        unsigned st = (unsigned)streams, dm = (unsigned)dim;
        const Args args(d, b, g, st, dm);
        launch(kInjectStreams, (unsigned)(gn * chunks), 1, args);
    }
    void signed_sqrt_gate(Slice dst, CSlice key, CSlice query, CSlice value, size_t rows, size_t groups, size_t dim, RowRuns, ZeroSign zero) override {
        if (!groups || !dim) throw std::runtime_error("rocm: invalid signed square-root gate dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(groups, dim)), vn = size_mul(rows, dim), blocks = size_mul(rows, groups);
        for (const CSlice& v : {CSlice(dst), key, query}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*value.buffer, value.offset * sizeof(float), size_mul(vn, sizeof(float)));
        if (overlaps(dst, n, key, n) || overlaps(dst, n, query, n) || overlaps(dst, n, value, vn)) throw std::runtime_error("rocm: signed square-root gate output must be disjoint from its inputs");
        if (blocks > INT32_MAX || dim > UINT32_MAX) throw std::runtime_error("rocm: a signed square-root gate past the launch's grid");
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
    // A thread a channel of a chunk of a view, chunks at least the history long so only a view's first chunk reads or leaves its carried rows (kernels/f32.hip).
    void dilated_conv_silu_add(Slice out, CSlice x, CSlice residual, CSlice w, size_t channels, size_t taps, size_t dilation, size_t layer, const StateView* views,
                               size_t n_views) override {
        if (!channels || !taps || !dilation) throw std::runtime_error("rocm: invalid dilated conv dimensions");
        const size_t rows = check_state_views(views, n_views, layer);
        if (!rows) return;
        std::vector<unsigned> tab = state_table(views, n_views);
        const StateStorage& ss = *views[0].storage;
        const size_t history = size_mul(taps - 1, dilation), n = size_mul(rows, channels), wn = size_mul(channels, taps);
        const size_t prefix = ss.shape().slot_floats(), slot = ss.slot_floats(layer);
        if (slot < size_add(prefix, size_mul(history, channels))) throw std::runtime_error("rocm: recurrent slot lacks dilated conv history");
        for (const CSlice& v : {CSlice(out), x, residual}) span(*v.buffer, v.offset * sizeof(float), size_mul(n, sizeof(float)));
        span(*w.buffer, w.offset * sizeof(float), size_mul(wn, sizeof(float)));
        if ((overlaps(out, n, residual, n) && !(out.buffer == residual.buffer && out.offset == residual.offset)) || overlaps(out, n, x, n) || overlaps(out, n, w, wn))
            throw std::runtime_error("rocm: dilated conv output may only alias the residual exactly");
        const CSlice state{&ss.layer(layer), 0};
        const size_t sn = ss.layer(layer).size() / sizeof(float);
        if (overlaps(out, n, state, sn) || overlaps(x, n, state, sn) || overlaps(residual, n, state, sn) || overlaps(w, wn, state, sn))
            throw std::runtime_error("rocm: dilated conv operand overlaps recurrent state storage");
        const size_t chunk = std::max(kConvChunk, history);
        size_t chunks = 0;
        for (size_t i = 0; i < n_views; ++i)
            for (size_t b0 = 0; b0 < views[i].nq; b0 += chunk, ++chunks) tab.insert(tab.end(), {(unsigned)i, (unsigned)b0});
        if (chunks > 65535 || channels > UINT32_MAX || taps > UINT32_MAX || history > UINT32_MAX) throw std::runtime_error("rocm: a dilated conv past the launch's grid");
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
    // DeepSeek-V4.1's layer ops (docs/DEEPSEEK4.md, section 4.2), a kernel each (kernels/f32.hip); the pooled-row ops read the auxiliary plane through pooled_table.
    void hc_coeffs(Slice pre, Slice post, Slice comb, CSlice mixes, CSlice stream, CSlice scale, CSlice base, size_t rows, size_t hc, size_t dim, float norm_eps, float eps,
                   size_t iters) override {
        if (!hc || hc > 16 || !dim || !iters || !std::isfinite(norm_eps) || norm_eps < 0.0f || !std::isfinite(eps) || eps < 0.0f)
            throw std::runtime_error("rocm: invalid mHC coefficient dimensions or epsilon");
        if (!rows) return;
        const size_t mix = size_mul(hc + 2, hc), n = size_mul(rows, hc), nc = size_mul(n, hc), mn = size_mul(rows, mix), xn = size_mul(rows, size_mul(hc, dim));
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{pre, n}, {post, n}, {comb, nc}, {mixes, mn}, {stream, xn}, {scale, 3}, {base, mix}}) floats(s, count);
        for (const auto& [o, on] : {std::pair<CSlice, size_t>{pre, n}, {post, n}, {comb, nc}})
            if (overlaps(o, on, mixes, mn) || overlaps(o, on, stream, xn) || overlaps(o, on, scale, 3) || overlaps(o, on, base, mix))
                throw std::runtime_error("rocm: mHC coefficient outputs overlap their inputs");
        if (overlaps(pre, n, post, n) || overlaps(pre, n, comb, nc) || overlaps(post, n, comb, nc)) throw std::runtime_error("rocm: mHC coefficient outputs must be disjoint");
        float *a = f32(pre), *b = f32(post), *c = f32(comb);
        const float *m = f32(mixes), *x = f32(stream), *sc = f32(scale), *bs = f32(base);
        unsigned h = (unsigned)hc, d = u32(dim), it = u32(iters);
        u32(size_mul(hc, dim));
        const Args args(a, b, c, m, x, sc, bs, h, d, norm_eps, eps, it);
        launch(kHcCoeffs, u32(rows), 1, args);
    }
    void hc_collapse_norm(Slice dst, CSlice stream, CSlice pre, CSlice w, size_t rows, size_t hc, size_t dim, float eps) override {
        if (!hc || !dim || !std::isfinite(eps) || eps <= 0.0f) throw std::runtime_error("rocm: invalid mHC collapse dimensions or epsilon");
        if (!rows) return;
        const size_t dn = size_mul(rows, dim), xn = size_mul(rows, size_mul(hc, dim)), pn = size_mul(rows, hc);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{dst, dn}, {stream, xn}, {pre, pn}, {w, dim}}) floats(s, count);
        if (overlaps(dst, dn, stream, xn) || overlaps(dst, dn, pre, pn) || overlaps(dst, dn, w, dim)) throw std::runtime_error("rocm: mHC collapse output must be disjoint from its inputs");
        float* o = f32(dst);
        const float *x = f32(stream), *p = f32(pre), *wp = f32(w);
        unsigned h = u32(hc), d = u32(dim);
        // The output's twin beside it where a product reads one next, as rms_norm_rows writes it.
        StreamTwin t = stream_twin(dst, rows, dim, {}, dim % 32 == 0);
        const Args args(o, x, p, wp, h, d, eps, t.q, t.d, t.level, t.map, t.major);
        launch_writing({{o, dn}}, kHcCollapseNorm, u32(rows), 1, args, 1, 1024);   // kernels/f32.hip, kCollapseThreads
        keep(t.t, o, dn);
    }
    void hc_expand(Slice dst, CSlice f, CSlice stream, CSlice post, CSlice comb, size_t rows, size_t hc, size_t dim) override {
        if (!hc || hc > 16 || !dim) throw std::runtime_error("rocm: invalid mHC expansion dimensions");
        if (!rows) return;
        const size_t n = size_mul(rows, size_mul(hc, dim)), fn = size_mul(rows, dim), pn = size_mul(rows, hc), cn = size_mul(pn, hc);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{dst, n}, {f, fn}, {stream, n}, {post, pn}, {comb, cn}}) floats(s, count);
        const bool in_place = dst.buffer == stream.buffer && dst.offset == stream.offset;
        if ((!in_place && overlaps(dst, n, stream, n)) || overlaps(dst, n, f, fn) || overlaps(dst, n, post, pn) || overlaps(dst, n, comb, cn))
            throw std::runtime_error("rocm: mHC expansion output must be disjoint or exactly the stream");
        float* o = f32(dst);
        const float *fp = f32(f), *x = f32(stream), *p = f32(post), *c = f32(comb);
        unsigned h = (unsigned)hc, d = u32(dim);
        unsigned long long lanes = fn;
        const Args args(o, fp, x, p, c, h, d, lanes);
        launch(kHcExpand, u32((fn + kThreads - 1) / kThreads), 1, args);
    }
    // qdq and pack are one kernel, a lane a value (kernels/f32.hip, llmx_model_format_qdq), writing values in place or each row's packed blocks to `out`.
    void model_format(float* out, size_t stride, CSlice x, size_t count, size_t dim, ModelFormat format) {
        if (format == ModelFormat::f32) throw std::runtime_error("rocm: f32 is no model format");
        const size_t block = format_block_values(format);
        if (count % block || dim % block) throw std::runtime_error("rocm: a model format's values are not whole blocks");
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
        if (stride < words) throw std::runtime_error("rocm: packed rows closer than a row's words");
        if (!rows) return;
        const size_t on = size_add(size_mul(rows - 1, stride), words), xn = size_mul(rows, dim);
        floats(out, on);
        if (overlaps(out, on, x, xn)) throw std::runtime_error("rocm: packed model format output must be disjoint from its input");
        model_format(f32(out), stride, x, xn, dim, format);
    }
    // topk_rows and candidate_blocks are one kernel, a row a block (kernels/f32.hip, llmx_topk_rows).
    void topk_rows(Slice ids, CSlice scores, CSlice lens, size_t rows, size_t n, size_t k, uint32_t offset) override {
        if (k > n || n >= UINT32_MAX || size_t(offset) + n >= UINT32_MAX) throw std::runtime_error("rocm: invalid top-k rows dimensions");
        if (!rows || !k) return;
        topk_select(ids, scores, lens, rows, n, k, k, offset, 0, 1, n);
    }
    void candidate_blocks(Slice blocks, CSlice scores, CSlice lens, size_t rows, size_t n, size_t block_size, size_t k) override {
        if (!block_size || !n || n >= UINT32_MAX) throw std::runtime_error("rocm: invalid candidate block dimensions");
        if (!rows || !k) return;
        const size_t items = (n + block_size - 1) / block_size;
        topk_select(blocks, scores, lens, rows, n, k, std::min(k, items), 0, 1, block_size, items);
    }
    void compress_pool(Slice out, CSlice kv, CSlice score, CSlice w, size_t groups, size_t ratio, size_t dim, float eps) override {
        if (!ratio || !dim || !std::isfinite(eps) || eps <= 0.0f) throw std::runtime_error("rocm: invalid compressor pooling dimensions or epsilon");
        if (!groups) return;
        const size_t on = size_mul(groups, dim), in = size_mul(size_mul(groups, ratio), dim);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{out, on}, {kv, in}, {score, in}, {w, dim}}) floats(s, count);
        if (overlaps(out, on, kv, in) || overlaps(out, on, score, in) || overlaps(out, on, w, dim))
            throw std::runtime_error("rocm: compressor pooling output must be disjoint from its inputs");
        float* o = f32(out);
        const float *k = f32(kv), *s = f32(score), *wp = f32(w);
        unsigned r = u32(ratio), d = u32(dim);
        const Args args(o, k, s, wp, r, d, eps);
        launch(kCompressPool, u32(groups), 1, args);
    }
    void sparse_attention_sink(Slice out, CSlice q, CSlice kv, CSlice ids, CSlice sink, size_t rows, size_t heads, size_t dim, size_t kv_rows, size_t k, float scale,
                               size_t layer, const KVView* views, size_t n_views, CSlice entry_ids, size_t entry_k, ModelFormat kv_format,
                               ModelFormat entry_format) override {
        if (!heads || !dim || !std::isfinite(scale)) throw std::runtime_error("rocm: invalid sparse attention dimensions or scale");
        if (!n_views && (entry_k || entry_ids.buffer)) throw std::runtime_error("rocm: sparse attention entries without views");
        if (size_add(k, entry_k) > 1024) throw std::runtime_error("rocm: sparse attention names at most 1024 rows a row");
        if (!rows) return;
        const size_t kw = format_words(kv_format, dim), ew = format_words(entry_format, dim);
        const size_t qn = size_mul(rows, size_mul(heads, dim)), kvn = size_mul(kv_rows, kw), idn = size_mul(rows, k), en = size_mul(rows, entry_k);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{out, qn}, {q, qn}, {kv, kvn}, {ids, idn}, {sink, heads}}) floats(s, count);
        if (en) floats(entry_ids, en);
        if (overlaps(out, qn, q, qn) || overlaps(out, qn, kv, kvn) || overlaps(out, qn, ids, idn) || overlaps(out, qn, sink, heads) || (en && overlaps(out, qn, entry_ids, en)))
            throw std::runtime_error("rocm: sparse attention output must be disjoint from its inputs");
        const PooledTable t = n_views ? pooled_table(layer, views, n_views, rows, ew, "sparse attention") : PooledTable{};
        float* o = f32(out);
        const float *qp = f32(q), *kp = f32(kv), *sk = f32(sink);
        const unsigned* ip = (const unsigned*)f32(ids);
        const unsigned* ep = en ? (const unsigned*)f32(entry_ids) : nullptr;
        const float* aux = t.aux;
        const unsigned* tab = t.rows;
        unsigned h = u32(heads), d = u32(dim), kk = u32(k), ek = u32(entry_k), kr = u32(kv_rows), tokens = t.tokens, bt = (unsigned)kBlockTokens, rf = t.rows_floats, width = t.width;
        unsigned long long bf = t.block_floats;
        unsigned kf = unsigned(kv_format), ef = unsigned(entry_format), kws = u32(kw);
        // The scratch holds a chunk of rows of at most 32 MiB: p by kept row and head, the denominators, the kept rows and their two counts (kernels/f32.hip, llmx_sparse_softmax).
        const size_t named = size_add(k, entry_k), row_bytes = size_add(size_mul(size_add(size_mul(named, heads), heads), 4), size_mul(size_add(named, 1), 8));
        const size_t chunk = std::min(rows, std::max<size_t>(1, ((size_t)32 << 20) / row_bytes));
        const size_t pt_bytes = size_mul(size_mul(chunk, named), heads * 4), den_bytes = size_mul(chunk, heads * 4), kept_bytes = size_mul(size_mul(chunk, named), 8);
        const size_t den_at = pt_bytes, kept_at = (den_at + den_bytes + 7) / 8 * 8, counts_at = kept_at + kept_bytes, need = counts_at + size_mul(chunk, 8);
        if (!sparse_ || sparse_->size() < need) sparse_ = alloc(need, Memory::device);
        uint8_t* sc = (uint8_t*)as_rocm(*sparse_).data();
        float *pt = (float*)sc, *den = (float*)(sc + den_at);
        const float** kept = (const float**)(sc + kept_at);
        unsigned* counts = (unsigned*)(sc + counts_at);
        const unsigned nm = u32(named);
        for (size_t first = 0; first < rows; first += chunk) {
            const size_t n = std::min(chunk, rows - first);
            const unsigned fr = u32(first);
            // From kSparseWideRows rows whose kSparseSoftmaxHeads queries fit in shared memory the wide pair runs, below them main's one-head pair.
            const bool wide = n >= kSparseWideRows && kSparseSoftmaxHeads * dim <= sparse_q_floats_;
            launch(wide ? kSparseSoftmaxG : kSparseSoftmax, u32(size_mul(n, wide ? (heads + kSparseSoftmaxHeads - 1) / kSparseSoftmaxHeads : heads)), 1, Args(pt, den, kept, counts, qp, kp, ip, sk, aux, tab, ep, h, d, kk, ek, scale, kr, tokens, bt, bf, rf, width, kf, ef, kws, fr), 1, kThreads, wide ? u32(kSparseSoftmaxHeads * dim * 4) : 0u);
            const size_t cols = wide ? kSparseColumnsWide : kSparseColumnsNarrow;
            launch(wide ? kSparseColumns64 : kSparseColumns, u32((dim + cols - 1) / cols), u32(n), Args(o, (const float*)pt, (const float*)den, (const float* const*)kept,
                                                                                                          (const unsigned*)counts, h, d, nm, kf, ef, fr));
        }
    }
    void indexer_scores(Slice out, CSlice q, size_t layer, const KVView* views, size_t n_views, CSlice w, float w_scale, size_t heads, size_t dim, ModelFormat key_format,
                        size_t n, CSlice blocks, size_t block_k, size_t block_size) override {
        if (!heads || heads > 1024 || !dim || n >= UINT32_MAX || (blocks.buffer && !block_size) || !std::isfinite(w_scale))
            throw std::runtime_error("rocm: invalid index score dimensions");
        size_t rows = 0;
        for (size_t vi = 0; vi < n_views && views; ++vi) rows = size_add(rows, views[vi].nq);
        if (!rows || !n) return;
        const size_t key_words = format_words(key_format, dim);
        const PooledTable t = pooled_table(layer, views, n_views, rows, key_words, "index scores");
        const size_t on = size_mul(rows, n), qn = size_mul(rows, size_mul(heads, dim)), wn = size_mul(rows, heads), bn = blocks.buffer ? size_mul(rows, block_k) : 0;
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{out, on}, {q, qn}, {w, wn}}) floats(s, count);
        if (bn) floats(blocks, bn);
        if (overlaps(out, on, q, qn) || overlaps(out, on, w, wn) || (bn && overlaps(out, on, blocks, bn))) throw std::runtime_error("rocm: index scores output must be disjoint from its inputs");
        for (size_t vi = 0; vi < n_views; ++vi)
            if (views[vi].nq && size_add(views[vi].length, views[vi].nq) / t.tokens > n) throw std::runtime_error("rocm: index scores narrower than a row's groups");
        if (rows > 65535) throw std::runtime_error("rocm: index scores of more rows than a launch holds");
        float* o = f32(out);
        const float *qp = f32(q), *wp = f32(w);
        const float* aux = t.aux;
        const unsigned* tab = t.rows;
        const unsigned* bp = bn ? (const unsigned*)f32(blocks) : nullptr;
        unsigned nn = u32(n), h = u32(heads), d = u32(dim), tokens = t.tokens, bt = (unsigned)kBlockTokens, rf = t.rows_floats, width = t.width, bk = bn ? u32(block_k) : 0u,
                 bs = u32(block_size);
        unsigned long long bf = t.block_floats;
        unsigned kf = unsigned(key_format), kws = u32(key_words);
        unsigned per = std::max(1u, (unsigned)kThreads / h);
        const Args args(o, qp, aux, tab, wp, bp, nn, h, d, w_scale, tokens, bt, bf, rf, width, bk, bs, kf, kws, per);
        launch(kIndexerScores, u32((n + per - 1) / per), (unsigned)rows, args);
    }
    void pool_write(size_t layer, const KVView* views, size_t n_views, CSlice rows) override {
        if (!n_views) return;
        if (!views) throw std::runtime_error("rocm: pooled rows written without views");
        RocmKVStorage& s = storage(views[0]);
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error("rocm: pooled rows written to views of different storages");
            s.check_view(views[vi], layer, true);
        }
        const size_t tokens = s.auxiliary_pool(layer), width = s.auxiliary_pool_width(layer);
        if (!tokens || !s.auxiliary_buffer(layer)) throw std::runtime_error("rocm: pooled rows written to a layer that keeps none of their width");
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
    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, float limit, RowRuns runs) override {
        if (!(limit >= 0.0f) || !std::isfinite(limit)) throw std::runtime_error("rocm: a SwiGLU limit that is negative or not finite");
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
        const Args args(d, g, u, nn, limit, q, td, map);
        launch_writing({{d, n}}, twin_build(!t.q ? kSiluMul : t.int8 ? kSiluMulTwin8 : kSiluMulTwin16, map), (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t, d, n);
    }
    // A row a block of 64, its reductions in one tree of 64 pairs (kernels/f32.hip, llmx_moe_route).
    void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, const RouteRule& rule, Slice ids, Slice weights) override {
        if (!k || k > n_expert || k > 256 || n_expert > 1024) throw std::runtime_error("rocm: routing takes 1 to 256 of at most 1024 experts");
        if (rule.score == RouteRule::Score::sqrt_softplus && !std::isfinite(rule.scale)) throw std::runtime_error("rocm: a routing scale that is not finite");
        if (!rows) return;
        span(*scores.buffer, scores.offset * sizeof(float), size_mul(size_mul(rows, n_expert), sizeof(float)));
        for (const Slice& s : {ids, weights}) span(*s.buffer, s.offset * sizeof(float), size_mul(size_mul(rows, k), sizeof(float)));
        const float* sc = f32(scores);
        float* id = f32(ids);
        float* w = f32(weights);
        unsigned ne = (unsigned)n_expert, kk = (unsigned)k;
        int norm = rule.normalize;
        // The sqrt_softplus rule takes a wave a row in the CPU's order (kernels/f32.hip, llmx_route_sqrt_softplus).
        if (rule.score == RouteRule::Score::sqrt_softplus) {
            if (rule.bias.buffer) floats(rule.bias, n_expert);
            const float* bias = rule.bias.buffer ? f32(rule.bias) : nullptr;
            unsigned long long nr = rows;
            float scale = rule.scale;
            const Args args(sc, bias, id, w, nr, ne, kk, norm, scale);
            return launch_writing({{id, size_mul(rows, k)}, {w, size_mul(rows, k)}}, kRouteSqrtSoftplus, u32(rows), 1, args, 1, 64);
        }
        rule.softmax_only("rocm");
        const Args args(sc, id, w, ne, kk, norm);
        launch_writing({{id, size_mul(rows, k)}, {w, size_mul(rows, k)}}, kMoeRoute, (unsigned)rows, 1, args, 1, 64);
    }
    void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nrows, const Routing& routing, RowRuns runs, Dtype dtype) override {
        const size_t entries = size_mul(nrows, routing.k);
        if (projections.size() > 3) throw std::logic_error("rocm: more than three routed projections");
        if (!entries || !projections.size()) return;
        check_runs(nrows, runs);
        span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nrows, nin), sizeof(float)));
        routed_runs(projections.begin(), projections.size(), X, nin, nrows, routing, false, runs, dtype);
    }
    // The slots' products land in scratch, then each row's weighted sum in slot order joins Y (kernels/f32.hip, llmx_moe_combine).
    void matmul_experts_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows, const Routing& routing, RowRuns runs, Dtype dtype) override {
        const size_t k = routing.k, entries = size_mul(nrows, k);
        if (!entries || !nout) return;
        check_runs(nrows, runs);
        span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(entries, nin), sizeof(float)));
        span(*Y.buffer, Y.offset * sizeof(float), size_mul(size_mul(nrows, nout), sizeof(float)));
        span(*routing.weights.buffer, routing.weights.offset * sizeof(float), size_mul(entries, sizeof(float)));
        const size_t out_bytes = size_mul(size_mul(entries, nout), sizeof(float));
        if (!moe_out_ || moe_out_->size() < out_bytes) moe_out_ = alloc(out_bytes, Memory::device);
        const Projection one{weight, Slice{moe_out_.get(), 0}, nout};
        routed_runs(&one, 1, X, nin, nrows, routing, true, runs, dtype);
        float* y = f32(Y);
        const float* part = (const float*)as_rocm(*moe_out_).data();
        const float* w = f32(routing.weights);
        const float* id = f32(routing.ids);
        unsigned long long nr = nrows, no = nout;
        unsigned kk = (unsigned)k;
        const Args args(y, part, w, id, nr, no, kk);
        launch_writing({{y, size_mul(nrows, nout)}}, kMoeCombine, (unsigned)((nrows * nout + kThreads - 1) / kThreads), 1, args);
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
                                        const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools) override {
        if (layers == 0 || (n_head_kv && head_dim == 0)) throw std::runtime_error("rocm: invalid KV storage shape");
        if (n_head_kv && head_dim > kThreads) throw std::runtime_error("rocm: a KV head wider than " + std::to_string(kThreads) + " is not implemented");
        return std::make_unique<RocmKVStorage>(*this, layers, n_head_kv, head_dim, max_tokens, k_type, v_type, auxiliary_widths, auxiliary_pools);
    }
    void kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k, CSlice v, CSlice auxiliary) override {
        if (n_views && !views) throw std::runtime_error("rocm: KV write without views");
        if (!n_views) {
            if (auxiliary.buffer) throw std::runtime_error("rocm: auxiliary KV rows supplied without views");
            return;
        }
        const KvTargets t = kv_targets(layer, views, n_views);
        RocmKVStorage& s = *t.s;
        const size_t floats = size_mul(size_mul(t.rows, s.heads()), s.dim());
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
        int heads = (int)s.heads(), dim = (int)s.dim(), bt = (int)kBlockTokens;
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
        if (!views) throw std::runtime_error("rocm: KV write without views");
        if (!rope.pos) throw std::runtime_error("rocm: partial rope without positions");
        const KvTargets t = kv_targets(layer, views, n_views);
        RocmKVStorage& s = *t.s;
        if (t.rows != rows || s.dim() != dim || s.heads() != n_head_kv) throw std::runtime_error("rocm: norm_rope_kv inputs do not match the KV storage");
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
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out, int n_head, int n_head_kv, int head_dim) override {
        if (n_views && !views) throw std::runtime_error("rocm: attention without views");
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > (int)kThreads)
            throw std::runtime_error("rocm: invalid attention dimensions");
        if (!n_views) return;
        RocmKVStorage& s = storage(views[0]);
        if ((size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads()) throw std::runtime_error("rocm: attention outside the KV view");
        // Views of a prompt whose extent reaches the profile's attention_tile_rows, causal and of 128-wide heads, take the tile kernel, the rest the chunk kernel: by the view's extent, its row class, never by how many rows the call holds.
        const size_t tile_rows = dev_.profile.attention_tile_rows;
        std::vector<int32_t> table, firsts, ends, wide;
        std::vector<RowRun> runs;   // the views as the rows' runs, for the layout of the twin written beside the output
        int32_t row0 = 0, tiles = 0;
        bool narrow = false;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            if (&storage(view) != &s) throw std::runtime_error("rocm: attention over views of different storages");
            const size_t used = s.check_view(view, layer, false);
            const int32_t first = (int32_t)table.size();
            table.insert(table.end(), view.blocks, view.blocks + used);
            const size_t extent = view.extent ? view.extent : view.nq;
            const bool tiled = view.causal && extent >= tile_rows && head_dim == 128 && view.nq;
            if (tiled) {
                wide.insert(wide.end(), {row0, (int32_t)view.nq, (int32_t)view.length, first, tiles});
                tiles += (int32_t)((view.nq + 31) / 32);
            }
            narrow = narrow || (!tiled && view.nq);
            for (size_t b = 0; b < view.nq; ++b) {
                firsts.push_back(first);
                ends.push_back(tiled ? 0 : (int32_t)(view.causal ? view.length + b + 1 : view.length + view.nq));
            }
            row0 += (int32_t)view.nq;
            if (view.nq) runs.push_back({(size_t)row0, extent});
        }
        const size_t rows = firsts.size();
        if (!rows) return;
        const size_t floats = size_mul(size_mul(rows, (size_t)n_head), (size_t)head_dim);
        span(*Q.buffer, Q.offset * sizeof(float), size_mul(floats, sizeof(float)));
        span(*out.buffer, out.offset * sizeof(float), size_mul(floats, sizeof(float)));
        const float* q = f32(Q);
        float* o = f32(out);
        const void* kc = as_rocm(*s.k_buffer(layer)).data();
        const void* vc = as_rocm(*s.v_buffer(layer)).data();
        // The table, firsts, ends and tile views in one set, which every layer of the pass reads alike.
        const size_t nt = table.size();
        table.insert(table.end(), firsts.begin(), firsts.end());
        table.insert(table.end(), ends.begin(), ends.end());
        table.insert(table.end(), wide.begin(), wide.end());
        const int* tp = (const int*)pass_arg(table.data(), table.size() * sizeof(int32_t));
        const int* fp = tp + nt;
        const int* ep = fp + rows;
        int ratio = n_head / n_head_kv, bt = (int)kBlockTokens;
        int kh = s.k_type() == KVType::f16, vh = s.v_type() == KVType::f16;
        float scale = 1.0f / std::sqrt((float)head_dim);
        if (!wide.empty()) {
            const int* wv = ep + rows;
            int nviews = (int)(wide.size() / 5);
            const Args args(q, o, kc, vc, tp, wv, nviews, n_head, n_head_kv, bt, kh, vh, scale);
            launch(kAttentionTile, (unsigned)(tiles * n_head), 1, args);
        }
        // Each row's history in chunks of its own, merged in order (kernels/f32.hip), so its bits do not depend on the call.
        const size_t need = size_mul(size_mul(size_mul(rows, (size_t)n_head), kChunks), size_mul((size_t)head_dim + 2, sizeof(float)));
        if (!parts_ || parts_->size() < need) {
            parts_ = alloc(need, Memory::device);
        }
        float* pp = (float*)as_rocm(*parts_).data();
        // Rows of the tile kernel's views were answered above, and every kernel below skips them by their end of 0.
        // Heads 128 and 256 wide take the per-row kernel, its splits chunks of each row's length, as many as the longest row needs (a shorter row's extra splits are empty), within the profile's limit for the width.
        if (narrow && (head_dim == 128 || head_dim == 256)) {
            const DeviceProfile& prof = dev_.profile;
            int longest = 0;
            for (int32_t e : ends) longest = std::max(longest, e);
            int chunk = (int)prof.attention_split_chunk, max_parts = (int)(head_dim == 256 ? prof.attention_split_max_wide : prof.attention_split_max);
            int nsplit = std::min(max_parts, std::max(1, (longest + chunk - 1) / chunk));
            const size_t vneed = size_mul(size_mul(size_mul(rows, (size_t)n_head), (size_t)nsplit), size_mul((size_t)head_dim + 2, sizeof(float)));
            if (nsplit > 1 && (!parts_ || parts_->size() < vneed)) parts_ = alloc(vneed, Memory::device);
            float* vp = nsplit > 1 ? (float*)as_rocm(*parts_).data() : nullptr;
            // With no row on the tile kernel, the kernel that writes the output writes the twin the output projection reads beside it.
            const TwinOut t = wide.empty() ? twin_out(out, floats, twin_map({runs.data(), runs.size()}, rows, (size_t)n_head * head_dim)) : TwinOut{};
            void* tq = t.q;
            float* td = t.d;
            TwinMap map = t.map;
            int tl = !t.q ? 0 : t.int8 ? 2 : 1, none = 0;
            const Args args(q, o, kc, vc, tp, fp, ep, n_head, n_head_kv, bt, kh, vh, scale, nsplit, chunk, max_parts, vp, tq, td, nsplit > 1 ? none : tl, map);
            launch(twin_build(head_dim == 256 ? kAttentionVec256 : kAttentionVec, map), (unsigned)(rows * n_head * nsplit), 1, args);
            if (nsplit > 1) {
                const Args margs(vp, o, ep, n_head, nsplit, tq, td, tl, map);
                launch(twin_build(head_dim == 256 ? kAttentionVecMerge256 : kAttentionVecMerge, map), (unsigned)(rows * n_head), 1, margs);
            }
            keep(t, o, floats);
            return;
        }
        if (narrow) {
            const Args args(q, pp, kc, vc, tp, fp, ep, n_head, ratio, n_head_kv, head_dim, bt, kh, vh, scale);
            launch(kAttentionChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        } else {
            return;
        }
        const Args merge(pp, o, ep, n_head, head_dim);
        launch(kAttentionMerge, (unsigned)rows, (unsigned)n_head, merge);
    }

    // Attention over each row's selected ids through the dense bodies' sparse builds, per-row for heads 128 and 256 wide and chunked otherwise.
    // A row's splits follow the id stride alone, never the call.
    void sparse_attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, CSlice ids, size_t id_stride, Slice out, int n_head, int n_head_kv,
                          int head_dim) override {
        if (!n_views) return;
        if (!views) throw std::runtime_error("rocm: sparse attention without views");
        if (!id_stride || id_stride > (size_t)INT32_MAX || n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv || head_dim <= 0 || head_dim > (int)kThreads)
            throw std::runtime_error("rocm: invalid sparse attention dimensions");
        RocmKVStorage& s = storage(views[0]);
        if ((size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads()) throw std::runtime_error("rocm: sparse attention outside the KV view");
        std::vector<int32_t> table, firsts, ends;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            if (!view.nq) throw std::runtime_error("rocm: invalid sparse attention dimensions");
            if (&storage(view) != &s) throw std::runtime_error("rocm: sparse attention over views of different storages");
            const size_t used = s.check_view(view, layer, false);
            const int32_t first = (int32_t)table.size();
            table.insert(table.end(), view.blocks, view.blocks + used);
            for (size_t b = 0; b < view.nq; ++b) {
                firsts.push_back(first);
                ends.push_back((int32_t)(view.length + b + 1));
            }
        }
        const size_t rows = firsts.size(), floats = size_mul(size_mul(rows, (size_t)n_head), (size_t)head_dim), selected = size_mul(rows, id_stride);
        if (!Q.buffer || !ids.buffer || !out.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*Q.buffer, Q.offset * sizeof(float), size_mul(floats, sizeof(float)));
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(selected, sizeof(uint32_t)));
        span(*out.buffer, out.offset * sizeof(float), size_mul(floats, sizeof(float)));
        if (overlaps(out, floats, Q, floats) || overlaps(out, floats, ids, selected) || out.buffer == s.k_buffer(layer).get() ||
            out.buffer == s.v_buffer(layer).get() || out.buffer == s.auxiliary_buffer(layer).get())
            throw std::runtime_error("rocm: sparse attention output overlaps an input");
        const float* q = f32(Q);
        float* o = f32(out);
        const unsigned* idp = (const unsigned*)f32(ids);
        const void* kc = as_rocm(*s.k_buffer(layer)).data();
        const void* vc = as_rocm(*s.v_buffer(layer)).data();
        const size_t nt = table.size();
        table.insert(table.end(), firsts.begin(), firsts.end());
        table.insert(table.end(), ends.begin(), ends.end());
        const int* tp = (const int*)pass_arg(table.data(), table.size() * sizeof(int32_t));
        const int* fp = tp + nt;
        const int* ep = fp + rows;
        int ratio = n_head / n_head_kv, bt = (int)kBlockTokens, kh = s.k_type() == KVType::f16, vh = s.v_type() == KVType::f16, stride = (int)id_stride;
        float scale = 1.0f / std::sqrt((float)head_dim);
        if (head_dim == 128 || head_dim == 256) {
            size_t ids_per = dev_.profile.sparse_split_ids;
            while ((id_stride + ids_per - 1) / ids_per > kThreads) ids_per *= 2;
            int chunk = (int)ids_per, nsplit = (int)((id_stride + ids_per - 1) / ids_per);
            const size_t need = size_mul(size_mul(size_mul(rows, (size_t)n_head), (size_t)nsplit), size_mul((size_t)head_dim + 2, sizeof(float)));
            if (nsplit > 1 && (!parts_ || parts_->size() < need)) parts_ = alloc(need, Memory::device);
            float* vp = nsplit > 1 ? (float*)as_rocm(*parts_).data() : nullptr;
            const Args args(q, o, kc, vc, tp, fp, ep, n_head, n_head_kv, bt, kh, vh, scale, nsplit, chunk, vp, idp, stride);
            // Four heads of a KV head a block, each head its one-head bits (kernels/f32.hip, sparse_vec_grouped), where the groups fill the compute units, as the Vulkan backend chooses.
            if (ratio % 4 == 0 && rows * size_t(n_head) / 4 >= size_t(dev_.caps.compute_units)) launch(head_dim == 256 ? kSparseVec256G4 : kSparseVecG4, (unsigned)(rows * (n_head / 4) * nsplit), 1, args);
            else launch(head_dim == 256 ? kSparseVec256 : kSparseVec, (unsigned)(rows * n_head * nsplit), 1, args);
            if (nsplit > 1) {
                const Args margs(vp, o, ep, n_head, nsplit);
                launch(head_dim == 256 ? kSparseVecMerge256 : kSparseVecMerge, (unsigned)(rows * n_head), 1, margs);
            }
            return;
        }
        const size_t need = size_mul(size_mul(size_mul(rows, (size_t)n_head), kChunks), size_mul((size_t)head_dim + 2, sizeof(float)));
        if (!parts_ || parts_->size() < need) parts_ = alloc(need, Memory::device);
        float* pp = (float*)as_rocm(*parts_).data();
        const Args args(q, pp, kc, vc, tp, fp, ep, n_head, ratio, n_head_kv, head_dim, bt, kh, vh, scale, idp, stride);
        launch(kSparseChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        const Args margs(pp, o, ep, n_head, head_dim, stride);
        launch(kSparseMerge, (unsigned)rows, (unsigned)n_head, margs);
    }

    // The indexer in Vulkan's design: each view's groups pooled, then at most 16 rows at once scored and selected (docs/src/backends-rocm.md).
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
        if (!ids.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*ids.buffer, ids.offset * sizeof(float), size_mul(outputs, sizeof(uint32_t)));
        if (overlaps(ids, outputs, query, queries) || overlaps(ids, outputs, key_norm, index.dim) || overlaps(ids, outputs, cos, c.rope_values) ||
            overlaps(ids, outputs, sin, c.rope_values))
            throw std::runtime_error("rocm: index selection output overlaps an input");
        if (ids.buffer == c.s->auxiliary_buffer(layer).get() || ids.buffer == c.s->k_buffer(layer).get() || ids.buffer == c.s->v_buffer(layer).get())
            throw std::runtime_error("rocm: index selection output overlaps KV storage");
        // The scores of 16 rows at most, bounded by the call's longest view and never by its rows.
        const size_t score_bytes = size_mul(size_mul(std::min<size_t>(c.max_rows, kSelectRows), c.longest / index.compression), sizeof(float));
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
        const size_t words = size_add(kSelectRows * kSelectRowWords, size_mul(size_mul(kSelectRows, most_slices), 2));
        if (most_slices && (!select_state_ || select_state_->size() < words * sizeof(uint32_t))) select_state_ = alloc(words * sizeof(uint32_t), Memory::device);
        unsigned* state = most_slices ? (unsigned*)f32(Slice{select_state_.get(), 0}) : nullptr;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& v = views[vi];
            if (!v.nq) continue;
            pool_view(c, v, vi, index);
            const size_t count = (v.length + v.nq) / index.compression;
            const bool wide = count >= dev_.profile.select_wide_from;
            unsigned groups = u32(count), slices = u32((count + kSelectSlice - 1) / kSelectSlice);
            const int* blocks = c.blocks + c.first[vi];
            for (size_t row = 0; row < v.nq; row += kSelectRows) {
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

    // The gate+up build for a grid: the wide shared build, two rows a wave and four steps a load, where the grid is resident at that build's occupancy, else the lean one.
    auto step_up_kernel(bool q5, size_t grid) {
        const auto lean = q5 ? kMoeStepUpQ5K : kMoeStepUpQ4K, wide = q5 ? kMoeStepUpQ5KW : kMoeStepUpQ4KW;
        int& per_cu = step_up_per_cu_[q5];
        if (per_cu < 0) {
            hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
            hip_check(hip().hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernels_[wide], kThreads, 0), "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor");
        }
        return grid <= size_t(per_cu) * dev_.caps.compute_units ? wide : lean;
    }
    // Where fused_step holds, a step is three launches (docs/src/backends-rocm.md): route with row gate and twin of h, every gate and up with their SiLU, the down with combine and shared down.
    // Each output keeps the bits of the base's ops (Backend::moe_step), which run every other step.
    void moe_step(const MoeStep& m, CSlice h, Slice y, size_t nin, size_t rows, RowRuns runs = {}, Dtype dtype = Dtype::f16) override {
        if (!fused_step(m, h, nin, rows, runs, dtype)) return Backend::moe_step(m, h, y, nin, rows, runs, dtype);
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
        if (!step_counter_) {
            step_counter_ = alloc(64, Memory::device);
            hold_fill(as_rocm(*step_counter_).data(), 64);
        }
        // The shared act's twin: rows ff-value rows of int8, then their block scales from the half.
        const size_t sact_half = std::max<size_t>(2048, size_mul(rows, ff));
        if (!step_twin_ || step_twin_->size() < 2 * sact_half) step_twin_ = alloc(2 * sact_half, Memory::device);
        const size_t out_bytes = size_mul(size_mul(entries, nin), sizeof(float));
        if (!moe_out_ || moe_out_->size() < out_bytes) moe_out_ = alloc(out_bytes, Memory::device);
        const float* x = f32(h);
        // The twin of h unless kept, and the routed act's twin in the other slot, both kept as the base's products keep them.
        const TwinMap hm = twin_map({}, rows, nin), am = twin_map({}, entries, ff);
        const bool h_kept = find_twin(x, rows * nin / 32, true, hm, size_mul(hm.total, 36));
        uint8_t* hq = (uint8_t*)as_rocm(*twin_[twin_slot_]).data();
        float* hd = (float*)(hq + size_mul(hm.total, 32));
        if (!h_kept) keep_twin({x, rows * nin, true, hm, twin_slot_});
        float* act = f32(m.act);
        forget_written(act, entries * ff);
        (void)find_twin(act, entries * ff / 32, true, am, size_mul(am.total, 36));
        uint8_t* aq = (uint8_t*)as_rocm(*twin_[twin_slot_]).data();
        float* ad = (float*)(aq + size_mul(am.total, 32));
        const size_t act_slot = twin_slot_;
        // The shared act's twin, which only the fused down reads, in a buffer of its own: no twin slot is written while the launch writing it reads them.
        uint8_t* sq = (uint8_t*)as_rocm(*step_twin_).data();
        float* sd = (float*)(sq + sact_half);
        unsigned* ids = (unsigned*)f32(m.ids);
        const StepRoute r{bytes(m.router.weight.data), x, f32(m.router.out), nin, ne, bytes(s.row_gate.weight.data), f32(s.row_gate.out), hq, hd, h_kept ? 0 : rows * nin / 32,
                          ids, f32(m.weights), (unsigned)ne, (unsigned)k, m.rule.normalize ? 1 : 0, (unsigned*)as_rocm(*step_counter_).data(), (unsigned)rows, 0};
        launch_step(kMoeStepRoute, (unsigned)(ne / 4 + 1), 1, Args(r));
        const TypeKernels& gk = *type_kernels(gt);
        const Projection gu[2] = {m.gate, m.up};
        // The shared expert's rows of h take the first block rows, so the routed entries start rows block rows on.
        RoutedCall gc = routed_call(gu, 2, nin, *gk.twin, ids, k, ne);
        gc.first = 0ull - rows;
        const Proj sp[2] = {{bytes(s.gate.weight.data), f32(s.gate.out), ff}, {bytes(s.up.weight.data), f32(s.up.out), ff}};
        const RowCall sc = row_call(sp, 2, 0, row_form(kQ8Twins, nin), nin);
        const StepUp u{gc, hq, hd, nin, act, aq, ad, sc, hq, hd, nin, f32(s.act), f32(s.row_gate.out), sq, sd, rows};
        launch_step(step_up_kernel(gt == quant::GGML_TYPE_Q5_K, ff / 32 * (entries + rows)), (unsigned)(ff / 32), (unsigned)(entries + rows), Args(u));
        keep_twin({act, entries * ff, true, am, act_slot});
        const TypeKernels& dk = *type_kernels(m.down.type);
        const Projection dn{m.down, Slice{moe_out_.get(), 0}, nin};
        const RoutedCall dc = routed_call(&dn, 1, ff, *dk.twin, ids, 1, ne);
        const Proj sdn{bytes(s.down.data), f32(y), nin};
        const RowCall dsc = row_call(&sdn, 1, 0, row_form(kQ8Twins, ff), ff);
        const float* y0 = m.zero ? (const float*)as_rocm(*m.zero).data() : f32(y);
        const StepDown d{dc, aq, ad, ff, f32(m.weights), f32(y), (unsigned)k, dsc, sq, sd, ff, y0, m.zero ? 0 : nin};
        launch_step(m.down.type == quant::GGML_TYPE_Q8_0 ? kMoeStepDownQ8 : kMoeStepDownQ51, (unsigned)(nin / 8), (unsigned)rows, Args(d));
        record_matrix_path(MatrixPath::block_int8);
        const std::pair<const float*, size_t> written[] = {{f32(m.router.out), rows * ne}, {f32(s.row_gate.out), rows}, {(const float*)ids, entries}, {f32(m.weights), entries},
                                                           {f32(m.gate.out), entries * ff}, {f32(m.up.out), entries * ff}, {f32(s.gate.out), rows * ff}, {f32(s.up.out), rows * ff},
                                                           {f32(s.act), rows * ff}, {f32(y), rows * nin}};
        for (const auto& [p, n] : written) forget_written(p, n);
    }
    // The fused step kernels' own limits (kernels/f32.hip mf_route, kernels/int16.hip gate_up and down): experts, picks, the width of h and the routed width ff.
    static constexpr size_t kStepExperts = 512, kStepPicks = 10, kStepIn = 2560, kStepFf = 640;
    // Whether a step has the shape the fused kernels hold (docs/src/backends-rocm.md, moe_step), h and every output in device memory.
    bool fused_step(const MoeStep& m, CSlice h, size_t nin, size_t rows, RowRuns runs, Dtype dtype) const {
        if (!rows) return false;
        for (size_t i = 0; rows > 1 && i < runs.n; ++i)
            if (runs.runs[i].extent != 1) return false;
        if (dtype != Dtype::int8 || !kernels_[kMoeStepUpQ4K] || dev_.caps.subgroup_size != 64 || !m.shared || !m.shared->row_gate.weight.data.buffer) return false;
        const MoeShared& s = *m.shared;
        const size_t ff = m.gate.rows;
        const uint32_t gt = m.gate.weight.type, dt = m.down.type;
        if (m.rule.score != RouteRule::Score::softmax || m.rule.bias.buffer || m.rule.scale != 1.0f || m.limit != 0.0f) return false;
        const size_t ne = m.router.rows;
        if (!ne || ne % 64 || ne > kStepExperts || !m.k || m.k > kStepPicks || !nin || nin % 256 || nin > kStepIn) return false;
        // An even block count, since the shared Q8_0 down takes the half-block order at any width where the base's quarter layout holds an odd one.
        if (!ff || ff % 64 || ff > kStepFf || m.up.rows != ff || s.gate.rows != ff || s.up.rows != ff) return false;
        if (m.router.weight.type != quant::GGML_TYPE_F32 || s.row_gate.weight.type != quant::GGML_TYPE_F32 || s.row_gate.rows != 1) return false;
        if ((gt != quant::GGML_TYPE_Q4_K && gt != quant::GGML_TYPE_Q5_K) || m.up.weight.type != gt) return false;
        if (dt != quant::GGML_TYPE_Q5_1 && dt != quant::GGML_TYPE_Q8_0) return false;
        if (s.gate.weight.type != quant::GGML_TYPE_Q8_0 || s.up.weight.type != quant::GGML_TYPE_Q8_0 || s.down.type != quant::GGML_TYPE_Q8_0) return false;
        const TypeKernels* gk = type_kernels(gt);
        const TypeKernels* dk = type_kernels(dt);
        if (!gk || !gk->twin || !dk || !dk->twin || &row_form(*gk->twin, nin) != gk->twin || row_cluster(*gk->twin, nin) != 32 || &row_form(*dk->twin, ff) != dk->twin) return false;
        // Where the base's routed products group the entries, each expert's gate and up rows are read once for all of them, which outweighs the fused step's fewer launches.
        if (gk->twin->groups(size_mul(rows, m.k), ne, true)) return false;
        if (as_rocm(*h.buffer).host_ptr()) return false;
        for (const Buffer* b : {m.router.out.buffer, m.gate.out.buffer, m.up.out.buffer, m.act.buffer, m.ids.buffer, m.weights.buffer, s.gate.out.buffer, s.up.out.buffer, s.act.buffer,
                                s.row_gate.out.buffer})
            if (as_rocm(*b).host_ptr()) return false;
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

private:
    // The F32 module's kernels, then from kRows16x1 the int16 module's, which only read twins.
    enum Kernel { kAdd, kCopy, kSiluMul, kSiluMulTwin16, kSiluMulTwin8, kRmsNorm, kRmsNorm16, kRmsNorm16Twin16, kRmsNorm16Twin8, kRmsNorm32, kRmsNorm32Twin16,
                  kRmsNorm32Twin8, kNormRope, kNormRopeKv, kGather, kGatherQ8, kMatmul, kMatmulQ8, kMatmulQ4K, kGatherQ4K, kMatmulQ5K, kGatherQ5K, kMatmulQ6K,
                  kGatherQ6K, kMatmulQ50, kGatherQ50, kMatmulQ51, kGatherQ51, kMatmulBF16, kGatherBF16, kMatmulIQ4NL, kGatherIQ4NL, kMatmulMXFP4, kGatherMXFP4, kKvWrite, kAttentionChunk, kAttentionTile, kAttentionMerge, kAttentionVec,
                  kAttentionVecMerge, kTwin16, kTwin8, kSiluMulTwin16Major, kSiluMulTwin8Major, kRmsNorm16Twin16Major, kRmsNorm16Twin8Major,
                  kRmsNorm32Twin16Major, kRmsNorm32Twin8Major, kAttentionVecMajor, kAttentionVecMergeMajor, kTwin16Major, kTwin8Major, kGroupSum, kSigmoidMul, kGatedRmsNorm, kNormRopeStreams, kCausalConvSilu, kGatedDeltaRule, kGatedDeltaRuleShort, kGroupRmsNorm, kSiluScaled, kMixStreams, kInjectStreams, kSignedSqrtGate, kDilatedConvSiluAdd, kKvIndexPool, kKvIndexScores, kKvIndexSelect, kKvIndexSelectWide, kSparseChunk, kSparseMerge, kSparseVec, kSparseVec256, kSparseVecMerge, kSparseVecMerge256,
                  kAttentionVec256, kAttentionVec256Major, kAttentionVecMerge256, kAttentionVecMerge256Major, kArgmaxSlices, kArgmaxMerge, kArgmaxPart, kArgmaxPick,
                  kMoeRoute, kMoeCombine, kMoeGroup, kRoutedExactF32, kRoutedExactQ8, kRoutedExactQ4K, kRoutedExactQ5K, kRoutedExactQ6K, kRoutedExactQ50, kRoutedExactQ51, kRoutedExactBF16, kRoutedExactIQ4NL, kRoutedExactMXFP4,
                  kRowsExactF32, kRowsExactQ8, kRowsExactQ4K, kRowsExactQ5K, kRowsExactQ6K, kRowsExactQ50, kRowsExactQ51, kRowsExactBF16, kRowsExactIQ4NL, kRowsExactMXFP4, kSparseVecG4, kSparseVec256G4, kColsF32x2, kColsF32x4, kMoeStepRoute,
                  kHcCoeffs, kHcCollapseNorm, kHcExpand, kModelFormatQdq, kTopkRows, kCompressPool, kIndexerScores, kPoolWrite, kSparseSoftmax, kSparseSoftmaxG, kSparseColumns, kSparseColumns64, kRouteSqrtSoftplus,
                  kNormRopeForm, kDeltaStep,
                  kRows16x1, kRows16x2, kRows16x2r4, kRows16x4, kRows16x4r4, kRows16x8, kRows16x8r4, kRows16x16, kTile16,
                  kRows8x1, kRows8x2, kRows8x2r4, kRows8x4, kRows8x4r4, kRows8x8, kRows8x8r4, kRows8x16, kTile8, kTileReduce,
                  kHalf8x1, kHalf8x1p, kHalf8x1r2, kHalf8x2, kHalf8x2r4, kHalf8x4, kHalf8x4r4, kHalf8x8, kHalf8x8r4, kHalf8x16, kHalf8x9, kHalf8x10, kHalf8x11, kHalf8x12, kHalf8x13, kHalf8x14, kHalf8x15, kTile16Tall, kTile8Tall, kRouted16, kRouted8, kRoutedHalf8, kGrouped16x2, kGrouped16x4, kGrouped16x8, kGrouped8x2, kGrouped8x4, kGrouped8x8, kGroupedHalf8x2, kGroupedHalf8x4, kGroupedHalf8x8, kRoutedTile16, kRoutedTile8, kRoutedNarrow16, kRoutedNarrow8, kRoutedStream8, kRoutedThinMXFP416,
                  kRowsQ4K16x1, kRowsQ4K16x2, kRowsQ4K16x4, kRowsQ4K16x8, kRowsQ4K16x16, kRowsQ4K8x1, kRowsQ4K8x2, kRowsQ4K8x4, kRowsQ4K8x8, kRowsQ4K8x16, kTileQ4K16, kTileQ4K8, kRoutedQ4K16, kRoutedQ4K8, kGroupedQ4K16x2, kGroupedQ4K16x4, kGroupedQ4K16x8, kGroupedQ4K8x2, kGroupedQ4K8x4, kGroupedQ4K8x8, kRoutedTileQ4K16, kRoutedTileQ4K8, kRoutedNarrowQ4K16, kRoutedNarrowQ4K8, kRoutedStreamQ4K8, kRoutedMidQ4K16, kRoutedMidQ4K8, kRowsQ5K16x1, kRowsQ5K16x2, kRowsQ5K16x4, kRowsQ5K16x8, kRowsQ5K16x16, kRowsQ5K8x1, kRowsQ5K8x2, kRowsQ5K8x4, kRowsQ5K8x8, kRowsQ5K8x16, kTileQ5K16, kTileQ5K8, kRoutedQ5K16, kRoutedQ5K8, kGroupedQ5K16x2, kGroupedQ5K16x4, kGroupedQ5K16x8, kGroupedQ5K8x2, kGroupedQ5K8x4, kGroupedQ5K8x8, kRoutedTileQ5K16, kRoutedTileQ5K8, kRoutedNarrowQ5K16, kRoutedNarrowQ5K8, kRoutedStreamQ5K8, kRoutedMidQ5K16, kRoutedMidQ5K8, kRowsQ6K16x1, kRowsQ6K16x2, kRowsQ6K16x4, kRowsQ6K16x8, kRowsQ6K16x16, kRowsQ6K8x1, kRowsQ6K8x2, kRowsQ6K8x4, kRowsQ6K8x8, kRowsQ6K8x16, kTileQ6K16, kTileQ6K8, kRoutedQ6K16, kRoutedQ6K8, kGroupedQ6K16x2, kGroupedQ6K16x4, kGroupedQ6K16x8, kGroupedQ6K8x2, kGroupedQ6K8x4, kGroupedQ6K8x8, kRoutedTileQ6K16, kRoutedTileQ6K8, kRoutedNarrowQ6K16, kRoutedNarrowQ6K8, kRoutedStreamQ6K8, kRoutedMidQ6K16, kRoutedMidQ6K8,
                  kRowsQ5016x1, kRowsQ5016x2, kRowsQ5016x4, kRowsQ5016x8, kRowsQ5016x16, kRowsQ508x1, kRowsQ508x2, kRowsQ508x4, kRowsQ508x8, kRowsQ508x16, kTileQ5016, kTileQ508, kRoutedQ5016, kRoutedQ508, kGroupedQ5016x2, kGroupedQ5016x4, kGroupedQ5016x8, kGroupedQ508x2, kGroupedQ508x4, kGroupedQ508x8, kRoutedTileQ5016, kRoutedTileQ508, kRoutedNarrowQ5016, kRoutedNarrowQ508, kRoutedStreamQ508, kRoutedMidQ5016, kRoutedMidQ508, kRowsQ5116x1, kRowsQ5116x2, kRowsQ5116x4, kRowsQ5116x8, kRowsQ5116x16, kRowsQ518x1, kRowsQ518x2, kRowsQ518x4, kRowsQ518x8, kRowsQ518x16, kTileQ5116, kTileQ518, kRoutedQ5116, kRoutedQ518, kGroupedQ5116x2, kGroupedQ5116x4, kGroupedQ5116x8, kGroupedQ518x2, kGroupedQ518x4, kGroupedQ518x8, kRoutedTileQ5116, kRoutedTileQ518, kRoutedNarrowQ5116, kRoutedNarrowQ518, kRoutedStreamQ518, kRoutedMidQ5116, kRoutedMidQ518, kRowsIQ4NL16x1, kRowsIQ4NL16x2, kRowsIQ4NL16x4, kRowsIQ4NL16x8, kRowsIQ4NL16x16, kRowsIQ4NL8x1, kRowsIQ4NL8x2, kRowsIQ4NL8x4, kRowsIQ4NL8x8, kRowsIQ4NL8x16, kTileIQ4NL16, kTileIQ4NL8, kRoutedIQ4NL16, kRoutedIQ4NL8, kGroupedIQ4NL16x2, kGroupedIQ4NL16x4, kGroupedIQ4NL16x8, kGroupedIQ4NL8x2, kGroupedIQ4NL8x4, kGroupedIQ4NL8x8, kRoutedTileIQ4NL16, kRoutedTileIQ4NL8, kRoutedNarrowIQ4NL16, kRoutedNarrowIQ4NL8, kRoutedStreamIQ4NL8, kRoutedMidIQ4NL16, kRoutedMidIQ4NL8, kRowsMXFP416x1, kRowsMXFP416x2, kRowsMXFP416x4, kRowsMXFP416x8, kRowsMXFP416x16, kRowsMXFP48x1, kRowsMXFP48x2, kRowsMXFP48x4, kRowsMXFP48x8, kRowsMXFP48x16, kTileMXFP416, kTileMXFP48, kRoutedMXFP416, kRoutedMXFP48, kGroupedMXFP416x2, kGroupedMXFP416x4, kGroupedMXFP416x8, kGroupedMXFP48x2, kGroupedMXFP48x4, kGroupedMXFP48x8, kRoutedTileMXFP416, kRoutedTileMXFP48, kRoutedNarrowMXFP416, kRoutedNarrowMXFP48, kRoutedStreamMXFP48, kRoutedMidMXFP416, kRoutedMidMXFP48,
                  kQuarterQ5016x1, kQuarterQ5016x2, kQuarterQ5016x4, kQuarterQ5016x8, kQuarterQ5016x16, kQuarterQ508x1, kQuarterQ508x2, kQuarterQ508x4, kQuarterQ508x8, kQuarterQ508x16, kQuarterRoutedQ5016, kQuarterRoutedQ508, kQuarterGroupedQ5016x2, kQuarterGroupedQ5016x4, kQuarterGroupedQ5016x8, kQuarterGroupedQ508x2, kQuarterGroupedQ508x4, kQuarterGroupedQ508x8, kQuarterQ5116x1, kQuarterQ5116x2, kQuarterQ5116x4, kQuarterQ5116x8, kQuarterQ5116x16, kQuarterQ518x1, kQuarterQ518x2, kQuarterQ518x4, kQuarterQ518x8, kQuarterQ518x16, kQuarterRoutedQ5116, kQuarterRoutedQ518, kQuarterGroupedQ5116x2, kQuarterGroupedQ5116x4, kQuarterGroupedQ5116x8, kQuarterGroupedQ518x2, kQuarterGroupedQ518x4, kQuarterGroupedQ518x8, kHcMix, kHcMix8, kMoeStepUpQ4K, kMoeStepUpQ5K, kMoeStepDownQ51, kMoeStepDownQ8, kMoeStepUpQ4KW, kMoeStepUpQ5KW, kKernels };
    static constexpr size_t kHcColumns = 8;   // llmx_hc_mix8's columns, hc_mix<8> in kernels/int16.hip, which backend-rocm twins holds at 2 to 16 rows
    // The build for n columns: one column's own, else the one of up to kHcColumns.
    static Kernel hc_build(size_t n) { return n == 1 ? kHcMix : kHcMix8; }
    static constexpr const char* kKernelNames[kKernels] = {"llmx_add", "llmx_copy", "llmx_silu_mul", "llmx_silu_mul_twin16", "llmx_silu_mul_twin8", "llmx_rms_norm_rows",
                                                           "llmx_rms_norm_16", "llmx_rms_norm_16_twin16", "llmx_rms_norm_16_twin8", "llmx_rms_norm_32",
                                                           "llmx_rms_norm_32_twin16", "llmx_rms_norm_32_twin8", "llmx_norm_rope", "llmx_norm_rope_kv", "llmx_gather",
                                                           "llmx_gather_q8_0", "llmx_matmul_f32", "llmx_matmul_q8_0", "llmx_matmul_q4_k", "llmx_gather_q4_k",
                                                           "llmx_matmul_q5_k", "llmx_gather_q5_k", "llmx_matmul_q6_k", "llmx_gather_q6_k", "llmx_matmul_q5_0",
                                                           "llmx_gather_q5_0", "llmx_matmul_q5_1", "llmx_gather_q5_1", "llmx_matmul_bf16", "llmx_gather_bf16",
                                                           "llmx_matmul_iq4_nl", "llmx_gather_iq4_nl", "llmx_matmul_mxfp4", "llmx_gather_mxfp4", "llmx_kv_write",
                                                           "llmx_attention_chunk", "llmx_attention_tile", "llmx_attention_merge", "llmx_attention_vec", "llmx_attention_vec_merge",
                                                           "llmx_twin16", "llmx_twin8", "llmx_silu_mul_twin16_major", "llmx_silu_mul_twin8_major",
                                                           "llmx_rms_norm_16_twin16_major", "llmx_rms_norm_16_twin8_major", "llmx_rms_norm_32_twin16_major",
                                                           "llmx_rms_norm_32_twin8_major", "llmx_attention_vec_major", "llmx_attention_vec_merge_major",
                                                           "llmx_twin16_major", "llmx_twin8_major", "llmx_group_sum", "llmx_sigmoid_mul", "llmx_gated_rms_norm", "llmx_norm_rope_streams", "llmx_causal_conv_silu", "llmx_gated_delta_rule", "llmx_gated_delta_rule_short", "llmx_group_rms_norm", "llmx_silu_scaled", "llmx_mix_streams", "llmx_inject_streams", "llmx_signed_sqrt_gate", "llmx_dilated_conv_silu_add", "llmx_kv_index_pool", "llmx_kv_index_scores", "llmx_kv_index_select", "llmx_kv_index_select_wide", "llmx_sparse_attention_chunk", "llmx_sparse_attention_merge", "llmx_sparse_attention_vec", "llmx_sparse_attention_vec256", "llmx_sparse_attention_vec_merge", "llmx_sparse_attention_vec_merge256", "llmx_attention_vec256", "llmx_attention_vec256_major", "llmx_attention_vec_merge256", "llmx_attention_vec_merge256_major", "llmx_argmax_slices", "llmx_argmax_merge", "llmx_argmax_part", "llmx_argmax_pick", "llmx_moe_route", "llmx_moe_combine", "llmx_moe_group", "llmx_routed_f32", "llmx_routed_q8_0", "llmx_routed_q4_k", "llmx_routed_q5_k", "llmx_routed_q6_k", "llmx_routed_q5_0", "llmx_routed_q5_1", "llmx_routed_bf16", "llmx_routed_iq4_nl", "llmx_routed_mxfp4", "llmx_matmul_rows_f32", "llmx_matmul_rows_q8_0", "llmx_matmul_rows_q4_k", "llmx_matmul_rows_q5_k", "llmx_matmul_rows_q6_k", "llmx_matmul_rows_q5_0", "llmx_matmul_rows_q5_1", "llmx_matmul_rows_bf16", "llmx_matmul_rows_iq4_nl", "llmx_matmul_rows_mxfp4", "llmx_sparse_attention_vec_g4", "llmx_sparse_attention_vec256_g4", "llmx_matmul_cols2_f32", "llmx_matmul_cols4_f32", "llmx_moe_step_route", "llmx_hc_coeffs", "llmx_hc_collapse_norm", "llmx_hc_expand", "llmx_model_format_qdq", "llmx_topk_rows", "llmx_compress_pool", "llmx_indexer_scores", "llmx_pool_write", "llmx_sparse_softmax", "llmx_sparse_softmax_g", "llmx_sparse_columns", "llmx_sparse_columns64", "llmx_route_sqrt_softplus", "llmx_norm_rope_form", "llmx_delta_step", "llmx_q8_0_rows16_1",
                                                           "llmx_q8_0_rows16_2", "llmx_q8_0_rows16_2_r4", "llmx_q8_0_rows16_4", "llmx_q8_0_rows16_4_r4",
                                                           "llmx_q8_0_rows16_8", "llmx_q8_0_rows16_8_r4", "llmx_q8_0_rows16_16", "llmx_q8_0_tile16", "llmx_q8_0_rows8_1",
                                                           "llmx_q8_0_rows8_2", "llmx_q8_0_rows8_2_r4", "llmx_q8_0_rows8_4", "llmx_q8_0_rows8_4_r4", "llmx_q8_0_rows8_8",
                                                           "llmx_q8_0_rows8_8_r4", "llmx_q8_0_rows8_16", "llmx_q8_0_tile8", "llmx_tile_reduce", "llmx_q8_0_half8_1", "llmx_q8_0_half8_1p", "llmx_q8_0_half8_1r2",
                                                           "llmx_q8_0_half8_2", "llmx_q8_0_half8_2_r4", "llmx_q8_0_half8_4", "llmx_q8_0_half8_4_r4", "llmx_q8_0_half8_8",
                                                           "llmx_q8_0_half8_8_r4", "llmx_q8_0_half8_16", "llmx_q8_0_half8_9", "llmx_q8_0_half8_10", "llmx_q8_0_half8_11", "llmx_q8_0_half8_12", "llmx_q8_0_half8_13", "llmx_q8_0_half8_14", "llmx_q8_0_half8_15", "llmx_q8_0_tile16_tall", "llmx_q8_0_tile8_tall", "llmx_q8_0_routed16", "llmx_q8_0_routed8", "llmx_q8_0_routed_half8", "llmx_q8_0_routed16_g2", "llmx_q8_0_routed16_g4", "llmx_q8_0_routed16_g8", "llmx_q8_0_routed8_g2", "llmx_q8_0_routed8_g4", "llmx_q8_0_routed8_g8", "llmx_q8_0_routed_half8_g2", "llmx_q8_0_routed_half8_g4", "llmx_q8_0_routed_half8_g8", "llmx_q8_0_routed_tile16", "llmx_q8_0_routed_tile8", "llmx_q8_0_routed_narrow16", "llmx_q8_0_routed_narrow8", "llmx_q8_0_routed_stream8", "llmx_mxfp4_routed_thin16",
                                                           "llmx_q4_k_rows16_1", "llmx_q4_k_rows16_2", "llmx_q4_k_rows16_4", "llmx_q4_k_rows16_8", "llmx_q4_k_rows16_16", "llmx_q4_k_rows8_1", "llmx_q4_k_rows8_2", "llmx_q4_k_rows8_4", "llmx_q4_k_rows8_8", "llmx_q4_k_rows8_16", "llmx_q4_k_tile16", "llmx_q4_k_tile8", "llmx_q4_k_routed16", "llmx_q4_k_routed8", "llmx_q4_k_routed16_g2", "llmx_q4_k_routed16_g4", "llmx_q4_k_routed16_g8", "llmx_q4_k_routed8_g2", "llmx_q4_k_routed8_g4", "llmx_q4_k_routed8_g8", "llmx_q4_k_routed_tile16", "llmx_q4_k_routed_tile8", "llmx_q4_k_routed_narrow16", "llmx_q4_k_routed_narrow8", "llmx_q4_k_routed_stream8", "llmx_q4_k_routed_mid16", "llmx_q4_k_routed_mid8", "llmx_q5_k_rows16_1", "llmx_q5_k_rows16_2", "llmx_q5_k_rows16_4", "llmx_q5_k_rows16_8", "llmx_q5_k_rows16_16", "llmx_q5_k_rows8_1", "llmx_q5_k_rows8_2", "llmx_q5_k_rows8_4", "llmx_q5_k_rows8_8", "llmx_q5_k_rows8_16", "llmx_q5_k_tile16", "llmx_q5_k_tile8", "llmx_q5_k_routed16", "llmx_q5_k_routed8", "llmx_q5_k_routed16_g2", "llmx_q5_k_routed16_g4", "llmx_q5_k_routed16_g8", "llmx_q5_k_routed8_g2", "llmx_q5_k_routed8_g4", "llmx_q5_k_routed8_g8", "llmx_q5_k_routed_tile16", "llmx_q5_k_routed_tile8", "llmx_q5_k_routed_narrow16", "llmx_q5_k_routed_narrow8", "llmx_q5_k_routed_stream8", "llmx_q5_k_routed_mid16", "llmx_q5_k_routed_mid8", "llmx_q6_k_rows16_1", "llmx_q6_k_rows16_2", "llmx_q6_k_rows16_4", "llmx_q6_k_rows16_8", "llmx_q6_k_rows16_16", "llmx_q6_k_rows8_1", "llmx_q6_k_rows8_2", "llmx_q6_k_rows8_4", "llmx_q6_k_rows8_8", "llmx_q6_k_rows8_16", "llmx_q6_k_tile16", "llmx_q6_k_tile8", "llmx_q6_k_routed16", "llmx_q6_k_routed8", "llmx_q6_k_routed16_g2", "llmx_q6_k_routed16_g4", "llmx_q6_k_routed16_g8", "llmx_q6_k_routed8_g2", "llmx_q6_k_routed8_g4", "llmx_q6_k_routed8_g8", "llmx_q6_k_routed_tile16", "llmx_q6_k_routed_tile8", "llmx_q6_k_routed_narrow16", "llmx_q6_k_routed_narrow8", "llmx_q6_k_routed_stream8", "llmx_q6_k_routed_mid16", "llmx_q6_k_routed_mid8",
                                                           "llmx_q5_0_rows16_1", "llmx_q5_0_rows16_2", "llmx_q5_0_rows16_4", "llmx_q5_0_rows16_8", "llmx_q5_0_rows16_16", "llmx_q5_0_rows8_1", "llmx_q5_0_rows8_2", "llmx_q5_0_rows8_4", "llmx_q5_0_rows8_8", "llmx_q5_0_rows8_16", "llmx_q5_0_tile16", "llmx_q5_0_tile8", "llmx_q5_0_routed16", "llmx_q5_0_routed8", "llmx_q5_0_routed16_g2", "llmx_q5_0_routed16_g4", "llmx_q5_0_routed16_g8", "llmx_q5_0_routed8_g2", "llmx_q5_0_routed8_g4", "llmx_q5_0_routed8_g8", "llmx_q5_0_routed_tile16", "llmx_q5_0_routed_tile8", "llmx_q5_0_routed_narrow16", "llmx_q5_0_routed_narrow8", "llmx_q5_0_routed_stream8", "llmx_q5_0_routed_mid16", "llmx_q5_0_routed_mid8", "llmx_q5_1_rows16_1", "llmx_q5_1_rows16_2", "llmx_q5_1_rows16_4", "llmx_q5_1_rows16_8", "llmx_q5_1_rows16_16", "llmx_q5_1_rows8_1", "llmx_q5_1_rows8_2", "llmx_q5_1_rows8_4", "llmx_q5_1_rows8_8", "llmx_q5_1_rows8_16", "llmx_q5_1_tile16", "llmx_q5_1_tile8", "llmx_q5_1_routed16", "llmx_q5_1_routed8", "llmx_q5_1_routed16_g2", "llmx_q5_1_routed16_g4", "llmx_q5_1_routed16_g8", "llmx_q5_1_routed8_g2", "llmx_q5_1_routed8_g4", "llmx_q5_1_routed8_g8", "llmx_q5_1_routed_tile16", "llmx_q5_1_routed_tile8", "llmx_q5_1_routed_narrow16", "llmx_q5_1_routed_narrow8", "llmx_q5_1_routed_stream8", "llmx_q5_1_routed_mid16", "llmx_q5_1_routed_mid8",
                                                           "llmx_iq4_nl_rows16_1", "llmx_iq4_nl_rows16_2", "llmx_iq4_nl_rows16_4", "llmx_iq4_nl_rows16_8", "llmx_iq4_nl_rows16_16", "llmx_iq4_nl_rows8_1", "llmx_iq4_nl_rows8_2", "llmx_iq4_nl_rows8_4", "llmx_iq4_nl_rows8_8", "llmx_iq4_nl_rows8_16", "llmx_iq4_nl_tile16", "llmx_iq4_nl_tile8", "llmx_iq4_nl_routed16", "llmx_iq4_nl_routed8", "llmx_iq4_nl_routed16_g2", "llmx_iq4_nl_routed16_g4", "llmx_iq4_nl_routed16_g8", "llmx_iq4_nl_routed8_g2", "llmx_iq4_nl_routed8_g4", "llmx_iq4_nl_routed8_g8", "llmx_iq4_nl_routed_tile16", "llmx_iq4_nl_routed_tile8", "llmx_iq4_nl_routed_narrow16", "llmx_iq4_nl_routed_narrow8", "llmx_iq4_nl_routed_stream8", "llmx_iq4_nl_routed_mid16", "llmx_iq4_nl_routed_mid8", "llmx_mxfp4_rows16_1", "llmx_mxfp4_rows16_2", "llmx_mxfp4_rows16_4", "llmx_mxfp4_rows16_8", "llmx_mxfp4_rows16_16", "llmx_mxfp4_rows8_1", "llmx_mxfp4_rows8_2", "llmx_mxfp4_rows8_4", "llmx_mxfp4_rows8_8", "llmx_mxfp4_rows8_16", "llmx_mxfp4_tile16", "llmx_mxfp4_tile8", "llmx_mxfp4_routed16", "llmx_mxfp4_routed8", "llmx_mxfp4_routed16_g2", "llmx_mxfp4_routed16_g4", "llmx_mxfp4_routed16_g8", "llmx_mxfp4_routed8_g2", "llmx_mxfp4_routed8_g4", "llmx_mxfp4_routed8_g8", "llmx_mxfp4_routed_tile16", "llmx_mxfp4_routed_tile8", "llmx_mxfp4_routed_narrow16", "llmx_mxfp4_routed_narrow8", "llmx_mxfp4_routed_stream8", "llmx_mxfp4_routed_mid16", "llmx_mxfp4_routed_mid8",
                                                           "llmx_q5_0_quarter_rows16_1", "llmx_q5_0_quarter_rows16_2", "llmx_q5_0_quarter_rows16_4", "llmx_q5_0_quarter_rows16_8", "llmx_q5_0_quarter_rows16_16", "llmx_q5_0_quarter_rows8_1", "llmx_q5_0_quarter_rows8_2", "llmx_q5_0_quarter_rows8_4", "llmx_q5_0_quarter_rows8_8", "llmx_q5_0_quarter_rows8_16", "llmx_q5_0_quarter_routed16", "llmx_q5_0_quarter_routed8", "llmx_q5_0_quarter_routed16_g2", "llmx_q5_0_quarter_routed16_g4", "llmx_q5_0_quarter_routed16_g8", "llmx_q5_0_quarter_routed8_g2", "llmx_q5_0_quarter_routed8_g4", "llmx_q5_0_quarter_routed8_g8", "llmx_q5_1_quarter_rows16_1", "llmx_q5_1_quarter_rows16_2", "llmx_q5_1_quarter_rows16_4", "llmx_q5_1_quarter_rows16_8", "llmx_q5_1_quarter_rows16_16", "llmx_q5_1_quarter_rows8_1", "llmx_q5_1_quarter_rows8_2", "llmx_q5_1_quarter_rows8_4", "llmx_q5_1_quarter_rows8_8", "llmx_q5_1_quarter_rows8_16", "llmx_q5_1_quarter_routed16", "llmx_q5_1_quarter_routed8", "llmx_q5_1_quarter_routed16_g2", "llmx_q5_1_quarter_routed16_g4", "llmx_q5_1_quarter_routed16_g8", "llmx_q5_1_quarter_routed8_g2", "llmx_q5_1_quarter_routed8_g4", "llmx_q5_1_quarter_routed8_g8", "llmx_hc_mix", "llmx_hc_mix8", "llmx_moe_step_up_q4_k", "llmx_moe_step_up_q5_k", "llmx_moe_step_down_q5_1", "llmx_moe_step_down_q8_0", "llmx_moe_step_up_q4_k_w", "llmx_moe_step_up_q5_k_w"};
    // A type's twin kernels by form and columns, with its routed builds' break-evens and tile extents (docs/src/backends-rocm.md, weight types).
    // kKernels marks a build the type lacks.
    struct TwinKernels {
        Kernel two[3][5];
        size_t two_rows[3][5];
        Kernel four[3][5];
        Kernel tile[2], tall[2];
        Kernel routed[3];
        Kernel grouped[3][3];
        Kernel routed_tile[2], routed_narrow[2], routed_mid[2];
        Kernel routed_thin;     // the 16-bit twin's thin build, groups of up to kThinRows entries (kernels/int16.hip, k_thin), MXFP4's alone
        Kernel routed_stream;   // the 8-bit twin's routed stream, groups of up to its kStreamRows entries (kernels/int16.hip, routed_stream)
        float group_from[2];
        size_t tile_from[2][2];
        size_t block_values;   // a weight block's values
        unsigned row_lanes;    // the row kernel's lane units a weight block (kernels/int16.hip, KRaw's Lane), 0 where a wave takes a row
        unsigned row_chains;   // the lane units a lane holds, each in an accumulator of its own: 2 for MXFP4 (kernels/int16.hip, MXFP4Lane), else 1
        bool k45;              // Q4_K or Q5_K, whose rows take at most the profile's k45_row_lanes
        const TwinKernels* quarter;   // the quarter form's row, routed and grouped builds (kernels/int16.hip, QuarterLane), which row_form may take instead
        // Whether routed products of entries over n_expert take the grouped builds, whose break-even is group_from.
        constexpr bool groups(size_t entries, size_t n_expert, bool int8) const { return entries && double(entries) >= double(group_from[int8]) * double(n_expert); }
    };
    static constexpr TwinKernels kQ8Twins = {{{kRows16x1, kRows16x2, kRows16x4, kRows16x8, kRows16x16},
                                          {kRows8x1, kRows8x2, kRows8x4, kRows8x8, kRows8x16},
                                          {kHalf8x1, kHalf8x2, kHalf8x4, kHalf8x8, kHalf8x16}},
                                         {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 4}, {1, 2, 2, 2, 2}},
                                         {{kKernels, kRows16x2r4, kRows16x4r4, kRows16x8r4, kKernels},
                                          {kKernels, kRows8x2r4, kRows8x4r4, kRows8x8r4, kKernels},
                                          {kKernels, kHalf8x2r4, kHalf8x4r4, kHalf8x8r4, kKernels}},
                                         {kTile16, kTile8},
                                         {kTile16Tall, kTile8Tall},
                                         {kRouted16, kRouted8, kRoutedHalf8},
                                         {{kGrouped16x2, kGrouped16x4, kGrouped16x8}, {kGrouped8x2, kGrouped8x4, kGrouped8x8}, {kGroupedHalf8x2, kGroupedHalf8x4, kGroupedHalf8x8}},
                                         {kRoutedTile16, kRoutedTile8},
                                         {kRoutedNarrow16, kRoutedNarrow8},
                                         {kKernels, kKernels},
                                         kKernels,
                                         kRoutedStream8,
                                         {1.25f, 0.0f},
                                         {{32, 32}, {32, 32}},
                                         32,
                                         0,
                                         1,
                                         false,
                                         nullptr};
    // A K-type's twins from its first kernel, its set in LLMX_KTYPE's order (kernels/int16.hip), which the static_assert below holds.
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
                nullptr};
    };
    // The quarter form of a 32-value type from its first quarter build, in LLMX_KQUARTER's order (kernels/int16.hip): four lanes a block, its tiles the type's own.
    static constexpr auto quarter_twins = [](const TwinKernels& base, Kernel first, float group_from16, float group_from8) {
        const auto k = [first](int i) { return static_cast<Kernel>(first + i); };
        TwinKernels q = base;
        for (int f = 0; f < 3; ++f)
            for (int b = 0; b < 5; ++b) q.two[f][b] = k((f ? 5 : 0) + b);
        q.routed[0] = k(10), q.routed[1] = q.routed[2] = k(11);
        for (int f = 0; f < 3; ++f)
            for (int b = 0; b < 3; ++b) q.grouped[f][b] = k((f ? 15 : 12) + b);
        q.group_from[0] = group_from16, q.group_from[1] = group_from8;
        q.row_lanes = 4;
        q.quarter = nullptr;
        return q;
    };
    static constexpr TwinKernels kQ4KTwins = k_twins(kRowsQ4K16x1, 2.5f, 1.25f, 128, 32, 32, 32, 256, 8, true), kQ5KTwins = k_twins(kRowsQ5K16x1, 2.5f, 1.25f, 128, 32, 32, 32, 256, 8, true), kQ6KTwins = k_twins(kRowsQ6K16x1, 1.25f, 1.25f, 128, 32, 32, 32, 256, 16, false);
    static constexpr TwinKernels kQ50Base = k_twins(kRowsQ5016x1, 1.25f, 1.25f, 128, 32, 32, 32, 32, 1, false), kQ51Base = k_twins(kRowsQ5116x1, 5.0f, 1.25f, 128, 32, 32, 32, 32, 1, false), kIQ4NLTwins = k_twins(kRowsIQ4NL16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 32, 4, false), kMXFP4Base = k_twins(kRowsMXFP416x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 32, 4, false);
    static constexpr TwinKernels kQ50Quarter = quarter_twins(kQ50Base, kQuarterQ5016x1, 1.25f, 1.25f), kQ51Quarter = quarter_twins(kQ51Base, kQuarterQ5116x1, 1.25f, 2.5f);
    static constexpr auto with_quarter = [](TwinKernels t, const TwinKernels* q) { t.quarter = q; return t; };
    static constexpr TwinKernels kQ50Twins = with_quarter(kQ50Base, &kQ50Quarter), kQ51Twins = with_quarter(kQ51Base, &kQ51Quarter);
    // MXFP4 alone has the thin build, which the narrow tile beat on Q4_K (docs/STATUS.md, DeepSeek-V4.1 ROCm kernels).
    static constexpr TwinKernels kMXFP4Twins = [] { TwinKernels t = kMXFP4Base; t.routed_thin = kRoutedThinMXFP416; t.row_chains = 2; return t; }();
    static_assert(kQ6KTwins.tile[1] == kTileQ6K8 && kIQ4NLTwins.routed[1] == kRoutedIQ4NL8 && kIQ4NLTwins.grouped[1][2] == kGroupedIQ4NL8x8 && kIQ4NLTwins.routed_tile[1] == kRoutedTileIQ4NL8 && kIQ4NLTwins.routed_narrow[1] == kRoutedNarrowIQ4NL8 && kIQ4NLTwins.routed_stream == kRoutedStreamIQ4NL8 && kMXFP4Twins.routed_stream == kRoutedStreamMXFP48 && kMXFP4Twins.routed_narrow[1] == kRoutedNarrowMXFP48 && kMXFP4Twins.routed_mid[1] == kRoutedMidMXFP48, "a K-type's kernels follow LLMX_KTYPE's order");
    static_assert(kQ51Quarter.two[1][4] == kQuarterQ518x16 && kQ51Quarter.routed[1] == kQuarterRoutedQ518 && kQ51Quarter.grouped[1][2] == kQuarterGroupedQ518x8 && kQ51Quarter.tile[1] == kTileQ518, "a quarter form's kernels follow LLMX_KQUARTER's order");
    // Each weight type's kernels, one row a type for every caller (docs/ROCM.md, Owners and boundaries): its exact product, the same over several rows, its embedding, its routed exact product and its twin kernels.
    // A type without twins takes its exact product in every class.
    struct TypeKernels {
        uint32_t type;
        const char* name;
        Kernel exact, rows, embed, routed;
        const TwinKernels* twin;
        Kernel cols;   // the few-row builds of the exact product, 2 and 4 rows from here on; kKernels where the type has none
    };
    static constexpr TypeKernels kTypes[] = {
        {quant::GGML_TYPE_F32, "F32", kMatmul, kRowsExactF32, kGather, kRoutedExactF32, nullptr, kColsF32x2},          {quant::GGML_TYPE_Q8_0, "Q8_0", kMatmulQ8, kRowsExactQ8, kGatherQ8, kRoutedExactQ8, &kQ8Twins, kKernels},
        {quant::GGML_TYPE_Q4_K, "Q4_K", kMatmulQ4K, kRowsExactQ4K, kGatherQ4K, kRoutedExactQ4K, &kQ4KTwins, kKernels}, {quant::GGML_TYPE_Q5_K, "Q5_K", kMatmulQ5K, kRowsExactQ5K, kGatherQ5K, kRoutedExactQ5K, &kQ5KTwins, kKernels},
        {quant::GGML_TYPE_Q6_K, "Q6_K", kMatmulQ6K, kRowsExactQ6K, kGatherQ6K, kRoutedExactQ6K, &kQ6KTwins, kKernels}, {quant::GGML_TYPE_Q5_0, "Q5_0", kMatmulQ50, kRowsExactQ50, kGatherQ50, kRoutedExactQ50, &kQ50Twins, kKernels},
        {quant::GGML_TYPE_Q5_1, "Q5_1", kMatmulQ51, kRowsExactQ51, kGatherQ51, kRoutedExactQ51, &kQ51Twins, kKernels},    {quant::GGML_TYPE_BF16, "BF16", kMatmulBF16, kRowsExactBF16, kGatherBF16, kRoutedExactBF16, nullptr, kKernels},
        {quant::GGML_TYPE_IQ4_NL, "IQ4_NL", kMatmulIQ4NL, kRowsExactIQ4NL, kGatherIQ4NL, kRoutedExactIQ4NL, &kIQ4NLTwins, kKernels},
        {quant::GGML_TYPE_MXFP4, "MXFP4", kMatmulMXFP4, kRowsExactMXFP4, kGatherMXFP4, kRoutedExactMXFP4, &kMXFP4Twins, kKernels},
    };
    static_assert([] { for (const TypeKernels& t : kTypes) if (t.twin && t.twin->routed_stream == kKernels) return false; return true; }(), "every twin type has its 8-bit routed stream, since its 8-bit routed tiles leave it the groups of up to kStreamRows entries");
    static const TypeKernels* type_kernels(uint32_t type) {
        for (const TypeKernels& t : kTypes)
            if (t.type == type) return &t;
        return nullptr;
    }
    static constexpr unsigned kChunks = 16;   // a row's attention chunks at most (kernels/f32.hip)
    static constexpr unsigned kNormThreads = 1024;   // the held norms' block (kernels/f32.hip)
    static constexpr size_t kColumns = 16;   // the row kernel's columns (kernels/int16.hip)
    static constexpr size_t kRowsOut = 8, kRowsIn = 8;   // the blocked exact product's outputs and rows a block (kernels/f32.hip)
    static constexpr size_t kTileOut = 64, kTileOutTall = 128, kTileRows = 64;   // the prompt tile's outputs, its tall build's, and its rows (kernels/int16.hip)
    static constexpr unsigned kStreamLanes = 64;   // a routed stream block's lanes, an output each, so its grid is the tile's (kernels/int16.hip, routed_stream)

    // Memory in slots, pinned host memory the device reads in place or copies from, or device memory: a slot is taken again once the ticket that last read it retires.
    struct Ring {
        std::array<BufferPtr, kRingSlots> slots{};
        std::array<Ticket, kRingSlots> tags{};
        std::array<bool, kRingSlots> dirty{};   // read by work launched since the last submission
        std::array<size_t, kRingSlots> laps{};   // times each slot was taken again, so what it held before is known gone
        size_t slot = 0, used = 0;
        void* take(RocmBackend& b, size_t bytes) {
            const size_t cap = slots[0]->size();
            if (bytes > cap) throw std::logic_error("rocm: a ring take wider than a slot");
            bytes = (bytes + 255) & ~size_t(255);
            if (used + bytes > cap) {
                // The next slot waits for the ticket that last covered its readers, and a slot read since the last submission has none yet, so only then does a wrap submit.
                // A stage never fills every slot, so a wrap is not a flush point (docs/src/backends-rocm.md, records).
                slot = (slot + 1) % kRingSlots;
                if (dirty[slot]) b.submit();
                b.wait(tags[slot]);
                ++laps[slot];
                used = 0;
            }
            void* p = (uint8_t*)as_rocm(*slots[slot]).data() + used;
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
        if (v > UINT32_MAX) throw std::runtime_error("rocm: a value past 32 bits");
        return (unsigned)v;
    }
    // An indexer call's views on one storage, checked as the CPU and Vulkan check them before any launch: their blocks in device memory, each view's first block and first row, and the group positions.
    struct IndexCall {
        RocmKVStorage* s = nullptr;
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
        if (!views) throw std::runtime_error("rocm: index op without views");
        if (!index.dim || index.dim > kThreads || !index.rope_dim || index.rope_dim % 2 || index.rope_dim > index.dim || !index.compression || !std::isfinite(index.eps) ||
            index.eps <= 0 || (selecting && (!index.heads || !index.budget || index.budget % index.compression)))
            throw std::runtime_error("rocm: invalid index dimensions");
        if (positions.pos && (!positions.streams.count || (positions.streams.count > 1 && !positions.streams.of)))
            throw std::runtime_error("rocm: invalid group position streams");
        IndexCall c;
        c.s = &storage(views[0]);
        std::vector<int32_t> table;
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != c.s) throw std::runtime_error("rocm: index op over views of different storages");
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
            throw std::runtime_error("rocm: index op differs from the rows and groups its auxiliary blocks keep");
        const auto& auxiliary = c.s->auxiliary_buffer(layer);
        if (!auxiliary) throw std::runtime_error("rocm: index op without auxiliary key storage");
        c.rope_values = size_mul(c.longest, index.rope_dim / 2);
        for (const auto& [sl, count] : {std::pair<CSlice, size_t>{key_norm, index.dim}, {cos, c.rope_values}, {sin, c.rope_values}}) {
            if (!sl.buffer) throw std::runtime_error("rocm: operand without storage");
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
        if (!s.buffer) throw std::runtime_error("rocm: operand without storage");
        span(*s.buffer, s.offset * sizeof(float), size_mul(count, sizeof(float)));
    }
    void topk_select(Slice ids, CSlice scores, CSlice lens, size_t rows, size_t n, size_t stride, size_t picks, uint32_t offset, unsigned mode, size_t block_size, size_t items) {
        const size_t sn = size_mul(rows, n), idn = size_mul(rows, stride);
        for (const auto& [s, count] : {std::pair<CSlice, size_t>{ids, idn}, {scores, sn}, {lens, rows}}) floats(s, count);
        if (overlaps(ids, idn, scores, sn) || overlaps(ids, idn, lens, rows)) throw std::runtime_error("rocm: selection output must be disjoint from its inputs");
        unsigned* o = (unsigned*)f32(ids);
        const float* s = f32(scores);
        const unsigned* l = (const unsigned*)f32(lens);
        unsigned nn = u32(n), st = u32(stride), pk = u32(picks), of = offset, md = mode, bs = u32(block_size), it = u32(items);
        const Args args(o, s, l, nn, st, pk, of, md, bs, it);
        launch(kTopkRows, u32(rows), 1, args);
    }
    // The pooled rows a call's rows read (kernels/f32.hip, pooled_at): for each row the offset of its view's block ids in the table and its position plus one, then the ids.
    struct PooledTable {
        const float* aux = nullptr;
        const unsigned* rows = nullptr;
        unsigned tokens = 1, width = 0, rows_floats = 0;
        unsigned long long block_floats = 0;
    };
    PooledTable pooled_table(size_t layer, const KVView* views, size_t n_views, size_t rows, size_t dim, const char* op) {
        if (!views) throw std::runtime_error(std::string("rocm: ") + op + " without views");
        RocmKVStorage& s = storage(views[0]);
        std::vector<unsigned> tab(size_mul(rows, 2)), blocks;
        size_t r = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error(std::string("rocm: ") + op + " over views of different storages");
            const size_t used = s.check_view(views[vi], layer, false);
            const unsigned first = u32(size_add(tab.size(), blocks.size()));
            blocks.insert(blocks.end(), views[vi].blocks, views[vi].blocks + used);
            for (size_t i = 0; i < views[vi].nq; ++i, ++r) {
                if (r == rows) throw std::runtime_error(std::string("rocm: ") + op + " views do not cover its rows");
                tab[2 * r] = first;
                tab[2 * r + 1] = u32(size_add(views[vi].length, i + 1));
            }
        }
        if (r != rows) throw std::runtime_error(std::string("rocm: ") + op + " views do not cover its rows");
        if (!s.auxiliary_pool(layer) || s.auxiliary_pool_width(layer) < dim || !s.auxiliary_buffer(layer))
            throw std::runtime_error(std::string("rocm: ") + op + " reads pooled rows its layer does not keep");
        tab.insert(tab.end(), blocks.begin(), blocks.end());
        PooledTable t;
        t.aux = f32(CSlice{s.auxiliary_buffer(layer).get(), 0});
        t.rows = (const unsigned*)pass_arg(tab.data(), tab.size() * sizeof(unsigned));
        t.tokens = u32(s.auxiliary_pool(layer));
        t.width = u32(s.auxiliary_pool_width(layer));
        t.rows_floats = u32(s.auxiliary_rows_bytes(layer) / sizeof(float));
        t.block_floats = s.auxiliary_block_bytes(layer) / sizeof(float);
        return t;
    }
    // The groups view vi's rows complete, pooled into their rows of the auxiliary blocks (kernels/f32.hip, llmx_kv_index_pool).
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
    static constexpr size_t kSelectRows = 16, kSelectSlice = 1024, kSelectRowWords = 5122;   // a selection's rows at most and the wide form's slice and state a row (kernels/f32.hip)

    // Where a write's rows land: each row's KV block and token slot, read by the kernels from the argument ring.
    struct KvTargets {
        RocmKVStorage* s = nullptr;
        size_t rows = 0;
        void* kc = nullptr;
        void* vc = nullptr;
        const int* blocks = nullptr;
        const int* slots = nullptr;
        int k_half = 0, v_half = 0;
    };
    KvTargets kv_targets(size_t layer, const KVView* views, size_t n_views) {
        KvTargets t;
        RocmKVStorage& s = storage(views[0]);
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error("rocm: KV views of different storages in one write");
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
        t.kc = as_rocm(*s.k_buffer(layer)).data();
        t.vc = as_rocm(*s.v_buffer(layer)).data();
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
    // Values every layer of a pass reads alike, copied once into device memory on the stream and found again by their bytes, so kernels do not read them over PCIe (docs/src/backends-rocm.md).
    const void* pass_arg(const void* src, size_t bytes) {
        bytes = std::max<size_t>(bytes, 1);
        for (Upload& u : uploads_)
            if (u.at && u.lap == pass_.laps[u.slot] && u.bytes.size() == bytes && !std::memcmp(u.bytes.data(), src, bytes)) {
                pass_.dirty[u.slot] = true;
                return u.at;
            }
        void* d = pass_.take(*this, bytes);
        hold_copy(d, arg(src, bytes), bytes, hipMemcpyHostToDevice);
        Upload& u = uploads_[next_upload_++ % kUploads];
        u.bytes.assign((const uint8_t*)src, (const uint8_t*)src + bytes);
        u.at = d;
        u.slot = pass_.slot;
        u.lap = pass_.laps[pass_.slot];
        return d;
    }
    static float* f32(Slice s) { return (float*)as_rocm(*s.buffer).data() + s.offset; }
    static const float* f32(CSlice s) { return (const float*)as_rocm(*s.buffer).data() + s.offset; }
    static const uint8_t* bytes(CSlice s) { return (const uint8_t*)as_rocm(*s.buffer).data() + s.offset * sizeof(float); }
    // The weights the products and the embedding read: the types of kTypes, decoded exactly into F32 sums, in whole blocks and without scales.
    static const TypeKernels& check_weight(const WeightView& w, size_t nin, const char* what) {
        const TypeKernels* t = type_kernels(w.type);
        if (!t) throw std::runtime_error(std::string("rocm: an ") + what + " of type " + std::to_string(w.type) + " is not implemented");
        if (w.scales.buffer) throw std::runtime_error(std::string("rocm: a scaled ") + what + " is not implemented");
        if (!w.data.buffer) throw std::runtime_error(std::string("rocm: an ") + what + " without storage");
        if (nin % quant::storage_type(w.type)->block_size) throw std::runtime_error(std::string("rocm: a ") + t->name + " " + what + " row ends inside a block");
        return *t;
    }
    RocmKVStorage& storage(const KVView& v) {
        auto* s = dynamic_cast<RocmKVStorage*>(v.storage);
        if (!s) throw std::runtime_error("rocm: a KV view of another backend");
        return *s;
    }
    // A product of a type with twins outside the F32 class reads the activations' twin; every other product reads the activations as they are.
    static bool integer(const WeightView& weight, Dtype dtype) {
        const TypeKernels* t = type_kernels(weight.type);
        return t && t->twin && dtype != Dtype::f32;
    }
    void product(const WeightView& weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, bool accumulate, RowRuns runs, Dtype dtype,
                 bool twin) {
        const TypeKernels& kinds = check_weight(weight, nin, "matrix");
        if (!nin || !nout || !nbatch) return;
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nout));
        span(*X.buffer, X.offset * sizeof(float), size_mul(size_mul(nin, nbatch), sizeof(float)));
        span(*Y.buffer, Y.offset * sizeof(float), size_mul(size_mul(nout, nbatch), sizeof(float)));
        // Kernels write a product a float a row, which over PCIe into pinned host memory took the 0.6B's head 3.3 ms of a token: a host output is formed in device memory and copied over whole.
        if (as_rocm(*Y.buffer).host_ptr()) {
            const size_t bytes_out = size_mul(size_mul(nout, nbatch), sizeof(float));
            if (!staged_ || staged_->size() < bytes_out) staged_ = alloc(bytes_out, Memory::device);
            if (accumulate) copy(*staged_, 0, *Y.buffer, Y.offset * sizeof(float), bytes_out);
            product(weight, X, {staged_.get(), 0}, nin, nout, nbatch, accumulate, runs, dtype, twin);
            copy(*Y.buffer, Y.offset * sizeof(float), *staged_, 0, bytes_out);
            return;
        }
        if (integer(weight, dtype)) {
            const Proj one[1] = {{bytes(weight.data), f32(Y), nout}};
            return product_twin(*kinds.twin, one, 1, X, nin, nbatch, accumulate, runs, twin, dtype == Dtype::int8);
        }
        const uint8_t* w = bytes(weight.data);
        const float* x = f32(X);
        float* y = f32(Y);
        unsigned long long ni = nin, no = nout;
        int add = accumulate;
        if (nbatch > 1 && nbatch <= 4 && kinds.cols != kKernels) {
            // Up to four rows take a wave an output over all of them, each output the bits of its one-row product (kernels/f32.hip, exact_cols); at eight the blocked build measured faster on the router (docs/STATUS.md).
            unsigned long long nr = nbatch;
            const Args args(w, x, y, ni, no, nr, add);
            const size_t waves = kThreads / dev_.caps.subgroup_size;
            launch_writing({{y, size_mul(nout, nbatch)}}, static_cast<Kernel>(kinds.cols + (nbatch <= 2 ? 0 : 1)), (unsigned)((nout + waves - 1) / waves), 1, args);
        } else if (nbatch > 1) {
            // Several rows take the blocked build, each output the bits of its one-row product (kernels/f32.hip, exact_rows).
            unsigned long long nr = nbatch;
            const Args args(w, x, y, ni, no, nr, add);
            launch_writing({{y, size_mul(nout, nbatch)}}, kinds.rows, (unsigned)((nout + kRowsOut - 1) / kRowsOut), (unsigned)((nbatch + kRowsIn - 1) / kRowsIn), args);
        } else {
            // One row takes a wave an output (kernels/f32.hip, exact_wave).
            const Args args(w, x, y, ni, no, add);
            const size_t waves = kThreads / dev_.caps.subgroup_size;
            launch_writing({{y, nout}}, kinds.exact, (unsigned)((nout + waves - 1) / waves), 1, args);
        }
        // Every product here reads F32 activations, whatever dtype was asked for.
        record_matrix_path(MatrixPath::f32);
    }
    // A projection of a call: its weight rows, its output and their count.
    struct Proj {
        const uint8_t* w;
        float* y;
        size_t nout;
    };
    // The host's copy of the tile kernel's call (kernels/int16.hip, TileCall), field for field.
    struct TileCall {
        const uint8_t* w[3];
        float* y[3];
        unsigned long long nout[3];
        unsigned start[3];
        unsigned nproj;
        unsigned long long nin, nrows;
        int add;
        unsigned kper, parts;
        float* partials;
        unsigned xrow, xblock;
    };
    static_assert(sizeof(TileCall) == 136 && offsetof(TileCall, nin) == 88 && offsetof(TileCall, partials) == 120 && offsetof(TileCall, xblock) == 132,
                  "TileCall is the kernel's layout");
    // The host's copy of a routed tile call's groups (kernels/int16.hip, TileGroups), field for field.
    struct TileGroups {
        const unsigned* tab;
        unsigned long long order0, per, first;
        unsigned long long stride[3];
    };
    static_assert(sizeof(TileGroups) == 56 && offsetof(TileGroups, stride) == 32, "TileGroups is the kernel's layout");
    // The host's copy of the twin's layout (kernels/f32.hip, TwinMap), field for field.
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
    // The twin's layout from each run's kernel: generated rows row after row, a prompt's rows block-major and padded to an odd multiple of 4 (docs/src/backends-rocm.md).
    // It moves where a value lies, never the value; without a prompt's run, or past the map's regions, every row lies row after row.
    static TwinMap twin_map(RowRuns runs, size_t rows, size_t nin) {
        TwinMap m{};
        m.blocks = (unsigned)(nin / 32);
        size_t total = 0, begin = 0;
        unsigned n = 0;
        bool major = false;
        for (size_t r = 0; r < runs.n && begin < rows; ++r) {
            const size_t end = std::min(runs.runs[r].end, rows);
            const bool tile = runs.runs[r].extent != 1;
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
        if (total >= 0x80000000ull || rows * (nin / 32) >= 0x80000000ull) throw std::runtime_error("rocm: a twin of 2^31 blocks or more");
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
    // The host's copy of the row kernel's call (kernels/int16.hip, RowCall), field for field.
    struct RowCall {
        const uint8_t* w[3];
        float* y[3];
        unsigned long long nout[3];
        unsigned long long start[3];
        unsigned long long total;
        unsigned nproj;
        unsigned cluster;
    };
    static_assert(sizeof(RowCall) == 112 && offsetof(RowCall, start) == 72 && offsetof(RowCall, nproj) == 104 && offsetof(RowCall, cluster) == 108, "RowCall is the kernel's layout");
    // The lanes a row takes in the row kernel: a wave for Q8_0, else the power of two covering the row's lane units (kernels/int16.hip, KRaw's Lane), at most the wave.
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
    // The lane form of a row of nin values: for Q5_0 and Q5_1 the quarter form where its lane units fill the cluster's steps fuller than one lane a block does, one lane a block on a tie; every other type its one form.
    // Like row_cluster it reads the row's width and type alone, so a dense row and a routed entry of one weight row take one form and one order.
    const TwinKernels& row_form(const TwinKernels& tk, size_t nin) const {
        if (!tk.quarter) return tk;
        const auto fill = [&](const TwinKernels& f, size_t& span) {
            const size_t units = nin / f.block_values * f.row_lanes, cl = row_cluster(f, nin);
            span = (units + cl - 1) / cl * cl;
            return units;
        };
        size_t one_span, quarter_span;
        const size_t one_units = fill(tk, one_span), quarter_units = fill(*tk.quarter, quarter_span);
        return quarter_units * one_span > one_units * quarter_span ? *tk.quarter : tk;
    }
    // The host's copies of the routed calls (kernels/int16.hip, RoutedCall, and kernels/f32.hip, RoutedExact), field for field.
    struct RoutedCall {
        RowCall c;
        unsigned long long stride[3];
        const unsigned* ids;
        unsigned long long first, per, n_expert;
    };
    static_assert(sizeof(RoutedCall) == 168 && offsetof(RoutedCall, ids) == 136 && offsetof(RoutedCall, n_expert) == 160, "RoutedCall is the kernel's layout");
    // The host's copies of the fused moe_step's calls (kernels/f32.hip, MfRoute, and kernels/int16.hip, MfGateUpS and MfDown), field for field.
    // The host's copy of llmx_delta_step's argument (kernels/f32.hip, DeltaStepArgs), field for field.
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
        int tlevel, major;
        TwinMap tm;
        unsigned kd, vd;
    };
    static_assert(offsetof(DeltaStepArgs, slot) == 136 && offsetof(DeltaStepArgs, tq) == 168 && offsetof(DeltaStepArgs, tm) == 192 && offsetof(DeltaStepArgs, kd) == 312 &&
                      sizeof(DeltaStepArgs) == 320,
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
        unsigned rows, pad;
    };
    static_assert(offsetof(StepRoute, gw) == 40 && offsetof(StepRoute, ids) == 80 && offsetof(StepRoute, n_expert) == 96 && offsetof(StepRoute, counter) == 112 && sizeof(StepRoute) == 128,
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
    static_assert(offsetof(StepUp, q) == 168 && offsetof(StepUp, act) == 192 && offsetof(StepUp, sc) == 216 && offsetof(StepUp, sq) == 328 && offsetof(StepUp, sact) == 352 &&
                      offsetof(StepUp, std_) == 376 && sizeof(StepUp) == 392,
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
    static_assert(offsetof(StepDown, aq) == 168 && offsetof(StepDown, weights) == 192 && offsetof(StepDown, k) == 208 && offsetof(StepDown, sc) == 216 && offsetof(StepDown, sq) == 328 &&
                      offsetof(StepDown, y0) == 352 && sizeof(StepDown) == 368,
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
            for (size_t p = 0; p < nproj; ++p) at[p] = {proj[p].weight, {proj[p].out.buffer, proj[p].out.offset + first * k * proj[p].rows}, proj[p].rows};
            const Routing r{{routing.ids.buffer, routing.ids.offset + first * k}, routing.weights, k, routing.n_expert};
            const size_t xfirst = per == 1 ? first * k : first, xrows = per == 1 ? count * k : count;
            routed(at, nproj, {X.buffer, X.offset + xfirst * nin}, nin, xrows, count * k, r, per, dtype, tile);
        };
        if (!runs.n) return stretch(0, nrows, false);
        const TypeKernels* kinds = type_kernels(proj[0].weight.type);
        const size_t from = kinds && kinds->twin && integer(proj[0].weight, dtype) ? kinds->twin->tile_from[down][dtype == Dtype::int8] : SIZE_MAX;
        for_each_run(nrows, runs, [from](const RowRun& r) { return r.extent >= from; }, stretch);
    }
    // The routed row builds' call over the stacks of `proj`, `per` entries a twin row, from entry 0.
    RoutedCall routed_call(const Projection* proj, size_t nproj, size_t nin, const TwinKernels& rk, const unsigned* ids, size_t per, size_t n_expert) const {
        RoutedCall c{};
        for (size_t p = 0; p < nproj; ++p) {
            c.c.w[p] = bytes(proj[p].weight.data);
            c.c.y[p] = f32(proj[p].out);
            c.c.nout[p] = proj[p].rows;
            c.c.start[p] = c.c.total;
            c.c.total += proj[p].rows;
            c.stride[p] = quant::row_bytes(proj[0].weight.type, nin, proj[p].rows);
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
        const TypeKernels& kinds = check_weight(proj[0].weight, nin, "expert matrix");
        span(*routing.ids.buffer, routing.ids.offset * sizeof(float), size_mul(entries, sizeof(float)));
        for (size_t p = 0; p < nproj; ++p) {
            if (proj[p].weight.type != type) throw std::runtime_error("rocm: routed projections of different types");
            check_weight(proj[p].weight, nin, "expert matrix");
            span(*proj[p].weight.data.buffer, proj[p].weight.data.offset * sizeof(float), quant::row_bytes(type, nin, size_mul(routing.n_expert, proj[p].rows)));
            span(*proj[p].out.buffer, proj[p].out.offset * sizeof(float), size_mul(size_mul(entries, proj[p].rows), sizeof(float)));
            if (as_rocm(*proj[p].out.buffer).host_ptr()) throw std::runtime_error("rocm: a routed product into host memory");
        }
        const unsigned* ids = (const unsigned*)f32(routing.ids);
        constexpr size_t kGridY = 65535, kGroupExperts = 1024;   // HIP's grid height; the experts llmx_moe_group counts in shared memory
        unsigned long long ni = nin;
        if (integer(proj[0].weight, dtype)) {
            const bool int8 = dtype == Dtype::int8;
            const int form = !int8 ? 0 : nin / 32 % 2 ? 1 : 2;
            const TwinKernels& rk = row_form(*kinds.twin, nin);
            auto [q, d] = twin_of(X, xrows, nin, int8, twin_map({}, xrows, nin), true);
            RoutedCall c = routed_call(proj, nproj, nin, rk, ids, per, routing.n_expert);
            const size_t clusters = dev_.caps.subgroup_size / c.c.cluster;
            if (tile && routing.n_expert <= kGroupExperts) {
                // A prompt's entries in groups of up to 64 of one expert, each group a block row of the routed tile over its expert's rows (kernels/int16.hip, TileGroups); the tile never splits, so each entry's sums are its own.
                const unsigned chunk = kTileRows, groups = (unsigned)((entries + chunk - 1) / chunk + std::min(routing.n_expert, entries));
                const size_t tab_bytes = (4 * size_t(groups) + entries) * sizeof(unsigned);
                if (!moe_tab_ || moe_tab_->size() < tab_bytes) moe_tab_ = alloc(tab_bytes, Memory::device);
                unsigned* tab = (unsigned*)as_rocm(*moe_tab_).data();
                unsigned ne = (unsigned)entries, nx = (unsigned)routing.n_expert, ng = groups, ch = chunk;
                const Args gargs(ids, tab, ne, nx, ng, ch);
                launch_writing({{tab, tab_bytes / sizeof(float)}}, kMoeGroup, nx, 1, gargs);
                const size_t nblk = nin / 32;
                if (nblk * kTileOut * 34 + 64 > 0xffffffffull || (xrows * nblk) * 32 * (int8 ? 1 : 2) > 0xffffffffull)
                    throw std::runtime_error("rocm: a routed tile's rows are too wide for its 32-bit offsets");
                TileCall t{};
                TileGroups tg{tab, 4ull * groups, per, 0, {}};
                size_t gx = 0;
                for (size_t p = 0; p < nproj; ++p) {
                    t.w[p] = c.c.w[p];
                    t.y[p] = c.c.y[p];
                    t.nout[p] = proj[p].rows;
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
                // Each group runs in one build by its entries: thin (16-bit) or stream (8-bit) up to 8, narrow up to 16, a K-type's mid up to 32, the 64-row tile past.
                // The other builds' workgroups return at once; the thin build takes four output tiles a workgroup, a thread an output.
                const Kernel thin = int8 ? kKernels : kinds.twin->routed_thin;
                const std::pair<Kernel, unsigned> builds[5] = {{kinds.twin->routed_tile[int8], kThreads}, {kinds.twin->routed_mid[int8], kThreads}, {kinds.twin->routed_narrow[int8], kThreads},
                                                               {thin, kThreads}, {int8 ? kinds.twin->routed_stream : kKernels, kStreamLanes}};
                for (const auto& [kernel, threads] : builds)
                    for (size_t first = 0; kernel != kKernels && first < groups && (route_only_.empty() || std::strstr(kKernelNames[kernel], route_only_.c_str())); first += kGridY) {
                        tg.first = first;
                        const Args args(t, tg, q, d);
                        launch(kernel, (unsigned)(kernel == thin ? (gx + 3) / 4 : gx), (unsigned)std::min<size_t>(kGridY, groups - first), args, 1, threads);
                    }
            } else if (rk.groups(entries, routing.n_expert, int8) && routing.n_expert <= kGroupExperts) {
                // One expert's entries a group of the narrowest grouped build that holds their average, up to 8, each column the bits of its one-column product.
                const size_t touched = std::min(routing.n_expert, entries), average = (entries + touched - 1) / touched;
                const int width = average <= 2 ? 0 : average <= 4 ? 1 : 2;
                unsigned chunk = 2u << width, groups = (unsigned)((entries + chunk - 1) / chunk + touched);
                const size_t tab_bytes = (4 * size_t(groups) + entries) * sizeof(unsigned);
                if (!moe_tab_ || moe_tab_->size() < tab_bytes) moe_tab_ = alloc(tab_bytes, Memory::device);
                unsigned* tab = (unsigned*)as_rocm(*moe_tab_).data();
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
                const size_t wave_rows = rk.two_rows[form][0] * rows_per_block_ * clusters;
                for (size_t first = 0; first < entries; first += kGridY) {
                    c.first = first;
                    const Args args(c, q, d, ni);
                    launch(rk.routed[form], (unsigned)((c.c.total + wave_rows - 1) / wave_rows), (unsigned)std::min(kGridY, entries - first), args);
                }
            }
            record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
            for (size_t p = 0; p < nproj; ++p) forget_written(c.c.y[p], proj[p].rows * entries);
            return;
        }
        const float* x = f32(X);
        for (size_t p = 0; p < nproj; ++p) {
            const uint8_t* w = bytes(proj[p].weight.data);
            float* y = f32(proj[p].out);
            unsigned long long no = proj[p].rows;
            RoutedExact r{ids, 0, per, routing.n_expert, quant::row_bytes(type, nin, proj[p].rows)};
            // An entry's outputs a wave each, as the one-row product takes them (kernels/f32.hip, exact_wave).
            const size_t waves = kThreads / dev_.caps.subgroup_size;
            for (; r.first < entries; r.first += kGridY) {
                const Args args(w, x, y, ni, no, r);
                launch_writing({{y, size_mul(proj[p].rows, entries)}}, kinds.routed, (unsigned)((proj[p].rows + waves - 1) / waves), (unsigned)std::min<size_t>(kGridY, entries - r.first), args);
            }
        }
        record_matrix_path(MatrixPath::f32);
    }
    // The blocks a part of a split tile call sums, by the Vulkan tile's rule (split_blocks): a split below the profile's workgroups per compute unit, keeping at least its fewest blocks a part.
    size_t split_blocks(size_t workgroups, size_t nblk) const {
        const DeviceProfile& p = dev_.profile;
        const size_t per_cu = nblk * 32 < p.tile_narrow_nin ? p.tile_split_per_cu_narrow : p.tile_split_per_cu;
        const size_t target = per_cu * std::max(1u, dev_.caps.compute_units);
        if (workgroups >= target) return nblk;
        const size_t parts = std::min((target + workgroups - 1) / workgroups, std::max<size_t>(1, nblk / std::max<size_t>(p.tile_split_min_blocks, 2)));
        if (parts <= 1) return nblk;
        const size_t kper = (((nblk + parts - 1) / parts) + 1) & ~size_t(1);
        return std::min(kper, nblk);
    }
    // Points twin_slot_ at the twin of x's `blocks` blocks under `map`: true where a kept twin holds it, which becomes the newer, false where a fresh slot waits for its build.
    bool find_twin(const float* x, size_t blocks, bool int8, const TwinMap& map, size_t need) {
        want_twin_ = int8 ? 2 : 1;
        for (int i = 0; i < 2; ++i)
            if (kept_[i].x == x && kept_[i].values == blocks * 32 && kept_[i].int8 == int8 && kept_[i].map == map) {
                if (i == 1) std::swap(kept_[0], kept_[1]);
                twin_slot_ = kept_[0].slot;
                return true;
            }
        twin_slot_ = take_twin_slot(need);
        return false;
    }
    // The twin of X's `rows` rows laid out by `map` (block-int8 for the int8 class), built unless the kept twin holds it or `build` is false.
    std::pair<const uint8_t*, const float*> twin_of(CSlice X, size_t rows, size_t nin, bool int8, TwinMap map, bool build) {
        const size_t blocks = size_mul(rows, nin / 32), qbytes = size_mul(map.total, 32 * (int8 ? 1 : 2)), need = size_add(qbytes, size_mul(map.total, sizeof(float)));
        const float* x = f32(X);
        if (build && !find_twin(x, blocks, int8, map, need)) {
            uint8_t* q = (uint8_t*)as_rocm(*twin_[twin_slot_]).data();
            float* d = (float*)(q + qbytes);
            unsigned long long nb = blocks;
            const Args args(x, q, d, nb, map);
            launch_writing({}, twin_build(int8 ? kTwin8 : kTwin16, map), (unsigned)((blocks + kThreads / 32 - 1) / (kThreads / 32)), 1, args);
            // Kept for the next product of the same activations, unless the host may write them in place.
            if (!as_rocm(*X.buffer).host_ptr()) keep_twin({x, blocks * 32, int8, map, twin_slot_});
        }
        const uint8_t* q0 = (const uint8_t*)as_rocm(*twin_[twin_slot_]).data();
        return {q0, (const float*)(q0 + qbytes)};
    }
    // The row kernel's call for the group's projections from column `at`, their output rows one after another.
    RowCall row_call(const Proj* proj, size_t nproj, size_t at, const TwinKernels& rk, size_t nin) const {
        RowCall c{};
        for (size_t pi = 0; pi < nproj; ++pi) {
            c.w[pi] = proj[pi].w;
            c.y[pi] = proj[pi].y + at * proj[pi].nout;
            c.nout[pi] = proj[pi].nout;
            c.start[pi] = c.total;
            c.total += proj[pi].nout;
        }
        c.nproj = (unsigned)nproj;
        c.cluster = row_cluster(rk, nin);
        return c;
    }
    // The twin of X, then each run on its kernel: generated rows on the row kernel, kColumns at a time, a prompt's rows on the tile.
    // Without runs a call of one row is a generated row and any other a prompt's.
    void product_twin(const TwinKernels& tk, const Proj* proj, size_t nproj, CSlice X, size_t nin, size_t nbatch, bool accumulate, RowRuns runs, bool twin, bool int8) {
        const size_t qsize = int8 ? 1 : 2;
        const RowRun whole[1] = {{nbatch, nbatch == 1 ? size_t(1) : size_t(2)}};
        const RowRuns all = runs.n ? runs : RowRuns{whole, 1};
        TwinMap map = twin_map(all, nbatch, nin);
        const auto [q0, d0] = twin_of(X, nbatch, nin, int8, map, twin);
        unsigned long long ni = nin;
        int add = accumulate;
        size_t begin = 0;
        for (size_t r = 0; r < all.n; ++r) {
            size_t end = all.runs[r].end;
            if (end <= begin || end > nbatch) throw std::runtime_error("rocm: row runs out of order or past the call");
            // Adjacent runs share launches where both take the same kernel and split, one per sequence otherwise: a decode column, and a tile row, gives the same bits in any call.
            const auto same = [&](const RowRun& a, const RowRun& b) {
                return (a.extent == 1) == (b.extent == 1) && (a.extent == 1 || split_tiles_of(a.extent) == split_tiles_of(b.extent));
            };
            while (r + 1 < all.n && same(all.runs[r], all.runs[r + 1]) && all.runs[r + 1].end > end && all.runs[r + 1].end <= nbatch) end = all.runs[++r].end;
            for (size_t at = begin; at < end;) {
                // int8 rows of an even block count take the half-block order, on word loads; the F16 class keeps the quarter layout, measured faster for its widened weights (docs/STATUS.md).
                const int form = !int8 ? 0 : nin / 32 % 2 ? 1 : 2;
                const size_t rows = all.runs[r].extent == 1 ? std::min(end - at, kColumns) : end - at;
                size_t first = 0;
                unsigned xrow = 0, xblock = 0;
                map.locate(at, first, xrow, xblock);
                const uint8_t* q = q0 + first * 32 * qsize;
                const float* d = d0 + first;
                if (all.runs[r].extent == 1) {
                    // One launch for the group's projections, their output rows one after another.
                    const TwinKernels& rk = row_form(tk, nin);
                    RowCall c = row_call(proj, nproj, at, rk, nin);
                    int ncols = (int)rows;
                    const Args args(c, q, d, ni, ncols, add);
                    const int build = rows <= 1 ? 0 : rows <= 2 ? 1 : rows <= 4 ? 2 : rows <= 8 ? 3 : 4;
                    // A one-column half-block row of at most 16 blocks takes a 64-lane wave's halves as two rows, and from q8_decode_two_rows_from output rows the others take two rows a wave.
                    // Both keep the one-row build's bits and read the row width, the output count and the type, never the call's columns (kernels/int16.hip).
                    const bool half1 = rk.two[form][0] == kHalf8x1 && build == 0;
                    const bool paired = half1 && nin / 32 <= 16 && dev_.caps.subgroup_size == 64;
                    RowBuild rb = paired ? RowBuild{kHalf8x1p, 2} : half1 && c.total >= dev_.profile.q8_decode_two_rows_from ? RowBuild{kHalf8x1r2, 2} : row_build(rk, form, build, c.total);
                    // The half form's builds over 8 columns compute every column they hold at two rows a wave, so 9 to 15 columns take the narrowest that holds them (kernels/int16.hip).
                    static constexpr Kernel kHalfWide[7] = {kHalf8x9, kHalf8x10, kHalf8x11, kHalf8x12, kHalf8x13, kHalf8x14, kHalf8x15};
                    if (rb.kernel == kHalf8x16 && rows < 16) rb.kernel = kHalfWide[rows - 9];
                    const size_t per = rb.rows * rows_per_block_ * (dev_.caps.subgroup_size / c.cluster);
                    launch(rb.kernel, (unsigned)((c.total + per - 1) / per), 1, args);
                } else {
                    // The split follows the run's extent, its row class, never the rows this call holds.
                    tile(tk, proj, nproj, q, d, xrow, xblock, at, rows, split_tiles_of(all.runs[r].extent), nin, add, int8);
                }
                at += rows;
            }
            begin = end;
        }
        if (begin != nbatch) throw std::runtime_error("rocm: row runs short of the call");
        record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
        for (size_t pi = 0; pi < nproj; ++pi) forget_written(proj[pi].y, proj[pi].nout * nbatch);
    }
    // A row kernel build and the output rows a wave of it takes, R, so a twin load serves that many rows (kernels/int16.hip).
    struct RowBuild {
        Kernel kernel;
        size_t rows;
    };
    // The build of a form and column count for a call of `total` rows: the four-row build once its waves give each compute unit row_four_waves_per_cu, else the two-row one.
    // A row keeps its lanes and order in every build, so the choice changes no bits; the sweep behind it is in docs/STATUS.md (ROCm many users).
    RowBuild row_build(const TwinKernels& tk, int form, int build, size_t total) const {
        if (tk.four[form][build] != kKernels && total / 4 >= size_t(dev_.profile.row_four_waves_per_cu) * dev_.caps.compute_units) return {tk.four[form][build], 4};
        return {tk.two[form][build], tk.two_rows[form][build]};
    }
    // One tile call over the projections' rows [at, at + rows): their output tiles side by side, the inner dimension split for `tiles` column tiles (a prompt's own), the parts added by the reduce kernel.
    // The twin's rows lie `xrow` blocks apart and a row's blocks `xblock` apart (TwinMap).
    void tile(const TwinKernels& tk, const Proj* proj, size_t nproj, const uint8_t* q, const float* d, unsigned xrow, unsigned xblock, size_t at, size_t rows, size_t tiles, size_t nin,
              int add, bool int8) {
        // The tall build where the profile's tile_rows_for takes the tallest height; the split stays the 64-output build's, so both builds give the same bits.
        size_t outputs = 0, gx64 = 0;
        for (size_t pi = 0; pi < nproj; ++pi) {
            outputs += proj[pi].nout;
            gx64 += (proj[pi].nout + kTileOut - 1) / kTileOut;
        }
        const size_t tall_from = int8 ? dev_.profile.qtile_tall_from_8bit : dev_.profile.qtile_tall_from;
        const bool tall = tk.tall[int8] != kKernels && tile_rows_for(dev_.caps, dev_.profile, kTileOut, kTileOut, kTileOutTall, outputs, tiles, nin, tall_from) == kTileOutTall;
        const size_t per = tall ? kTileOutTall : kTileOut;
        // The tile addresses a workgroup's weights and twin by 32-bit offsets off its first output and row, a Q8_0 block's 34 bytes bounding every twin type's bytes for 32 values.
        if ((nin / 32) * per * 34 + 64 > 0xffffffffull || (size_t(kTileRows) * xrow + (nin / 32) * xblock) * 32 * (int8 ? 1 : 2) > 0xffffffffull)
            throw std::runtime_error("rocm: a prompt tile's rows are too wide for its 32-bit offsets");
        TileCall c{};
        size_t gx = 0;
        for (size_t pi = 0; pi < nproj; ++pi) {
            c.w[pi] = proj[pi].w;
            c.y[pi] = proj[pi].y + at * proj[pi].nout;
            c.nout[pi] = proj[pi].nout;
            c.start[pi] = (unsigned)gx;
            gx += (proj[pi].nout + per - 1) / per;
        }
        c.nproj = (unsigned)nproj;
        c.nin = nin;
        c.nrows = rows;
        c.add = add;
        c.xrow = xrow;
        c.xblock = xblock;
        const size_t nblk = nin / 32, kper = split_blocks(gx64 * tiles, nblk);
        c.kper = (unsigned)kper;
        c.parts = (unsigned)((nblk + kper - 1) / kper);
        if (c.parts > 1) {
            const size_t need = size_mul(size_mul(size_mul(c.parts, rows), outputs), sizeof(float));
            if (!partials_ || partials_->size() < need) partials_ = alloc(need, Memory::device);
            c.partials = (float*)as_rocm(*partials_).data();
        }
        const Args args(c, q, d);
        launch(tall ? tk.tall[int8] : tk.tile[int8], (unsigned)gx, (unsigned)((rows + kTileRows - 1) / kTileRows), args, c.parts);
        if (c.parts > 1) {
            size_t g = 0;
            for (size_t pi = 0; pi < nproj; ++pi) {
                c.start[pi] = (unsigned)g;
                g += (proj[pi].nout * rows + kThreads - 1) / kThreads;
            }
            const Args rargs(c);
            launch(kTileReduce, (unsigned)g, 1, rargs);
        }
    }
    // Rows `ids` of a table through the type's gather kernel, ids on the device; an id of `nrows` or more gives a zero row.
    void gather(Kernel k, Slice dst, const void* table, size_t width, const unsigned* ids, size_t count, size_t nrows) {
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(count, width), sizeof(float)));
        float* d = f32(dst);
        unsigned long long w = width;
        unsigned nr = (unsigned)std::min<size_t>(nrows, UINT32_MAX);
        const Args args(d, table, ids, w, nr);
        launch(k, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)count, args);
    }
    // Where a producer writes the twin of its output of `values` floats: once a product has read a twin, in that class, for an output of whole blocks in device memory.
    struct TwinOut {
        void* q = nullptr;
        float* d = nullptr;
        bool int8 = false;
        TwinMap map{};
        size_t slot = 0;
    };
    // Its rows lie as `map` lays them out (TwinMap), the layout the product's own runs give.
    TwinOut twin_out(Slice dst, size_t values, const TwinMap& map) {
        if (!want_twin_ || values % 32 || as_rocm(*dst.buffer).host_ptr()) return {};
        const bool int8 = want_twin_ == 2;
        const size_t qbytes = size_mul(map.total, int8 ? 32 : 64), need = size_add(qbytes, size_mul(map.total, sizeof(float)));
        const size_t slot = take_twin_slot(need);
        uint8_t* q = (uint8_t*)as_rocm(*twin_[slot]).data();
        return {q, (float*)(q + qbytes), int8, map, slot};
    }
    // A twin producer's build for `m`: the map-free one where every row lies row after row or no twin is written, the one reading the map otherwise (kernels/f32.hip, _major).
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
    void keep(const TwinOut& t, const float* x, size_t values) {
        if (t.q) keep_twin({x, values, t.int8, t.map, t.slot});
    }
    // A stream or gate op's twin of its `rows` rows `width` wide, written where a product reads one next and each row is whole blocks (kernels/f32.hip, twin_at): where it goes, its level, its map and whether the map is read.
    struct StreamTwin {
        TwinOut t;
        void* q;
        float* d;
        int level, major;
        TwinMap map;
    };
    StreamTwin stream_twin(Slice dst, size_t rows, size_t width, RowRuns runs, bool whole = true) {
        const TwinOut t = whole && width % 32 == 0 ? twin_out(dst, size_mul(rows, width), twin_map(runs, rows, width)) : TwinOut{};
        return {t, t.q, t.d, !t.q ? 0 : t.int8 ? 2 : 1, t.q && t.map.regions && !t.map.flat() ? 1 : 0, t.map};
    }
    // A twin kept in its slot, which a product of the same activations reads instead of making its own; a launch or copy writing its input, any other launch, a write or an allocation forgets it.
    struct Kept {
        const float* x = nullptr;
        size_t values = 0;
        bool int8 = false;
        TwinMap map{};
        size_t slot = 0;
    };
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
    void launch_writing(std::initializer_list<std::pair<const void*, size_t>> writes, Kernel k, unsigned gx, unsigned gy, const Args& args, unsigned gz = 1, unsigned threads = kThreads) {
        const auto kept = kept_;
        launch(k, gx, gy, args, gz, threads);
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
    void launch_check(hipError_t e, const char* what) {
        hip_check(e, what);
        opened_ = true;
    }
    // The first asynchronous failure, kept as the cause every later read or launch throws.
    void remember(hipError_t e, const char* where) noexcept {
        std::lock_guard<std::mutex> lk(failed_mu_);
        if (failed_.empty()) failed_ = std::string("rocm: ") + where + " failed on the device with " + hip().hipGetErrorName(e);
    }
    // A group sum that gave up on this device (fault_) throws once, at the next call that checks; the collective's later sums take the conservative path, so the backend goes on.
    void raise_failed() {
        {
            std::lock_guard<std::mutex> lk(failed_mu_);
            if (!failed_.empty()) throw std::runtime_error(failed_);
        }
        if (__atomic_load_n(fault_, __ATOMIC_ACQUIRE) && __atomic_exchange_n(fault_, 0u, __ATOMIC_ACQ_REL))
            throw std::runtime_error("rocm: a tensor group's sum on " + identity() + " gave up waiting for its peers; the group's later sums take the conservative path");
    }
    // A group sum's kernel, launched without raise_failed: the collective checks every member before it launches any, so no member waits for a peer whose launch threw.
    void group_launch(unsigned blocks, const Args& args) {
        forget_twins();
        issue(kernels_[kGroupSum], blocks, 1, 1, kThreads, args, "llmx_group_sum");
    }

    // A stream operation held until the next flush, which issues them in order: a kernel with its packed arguments, a copy, a zero fill, or a wait for or write of a 64-bit word, `bytes` its value (docs/src/backends-rocm.md, records).
    struct Held {
        enum Kind : uint8_t { kernel, copy, fill, wait_word, write_word } kind;
        hipFunction_t f;
        unsigned gx, gy, gz, threads;
        const char* name;
        size_t at, size;   // a kernel's argument bytes in op_bytes_
        void* dst;
        const void* src;
        size_t bytes;   // a copy's or fill's bytes, a wait's or write's value, a kernel's dynamic shared memory
        hipMemcpyKind dir;
    };
    void hold(const Held& op) {
        // A device with nothing left running starts on the first few, rather than waiting while the host records a whole chunk.
        if (!opened_) chunk_ = done(tickets_.last()) ? kFirstChunk : dev_.profile.dispatch_chunk;
        ops_.push_back(op);
        opened_ = true;
        if (ops_.size() >= chunk_) {
            flush();
            chunk_ = dev_.profile.dispatch_chunk;
        }
    }
    void issue(hipFunction_t f, unsigned gx, unsigned gy, unsigned gz, unsigned threads, const Args& args, const char* name, unsigned shared = 0) {
        const size_t at = (op_bytes_.size() + 15) & ~size_t(15);
        op_bytes_.resize(at + args.size);
        std::memcpy(op_bytes_.data() + at, args.bytes, args.size);
        hold({Held::kernel, f, gx, gy, gz, threads, name, at, args.size, nullptr, nullptr, shared, hipMemcpyDefault});
    }
    void hold_copy(void* dst, const void* src, size_t bytes, hipMemcpyKind dir) { hold({Held::copy, nullptr, 0, 0, 0, 0, "hipMemcpyAsync", 0, 0, dst, src, bytes, dir}); }
    void hold_fill(void* dst, size_t bytes) { hold({Held::fill, nullptr, 0, 0, 0, 0, "hipMemsetAsync", 0, 0, dst, nullptr, bytes, hipMemcpyDefault}); }
    // The stream waits until *word is at least `value`, or writes `value` there once the work before it is done.
    void hold_wait(uint64_t* word, uint64_t value) { hold({Held::wait_word, nullptr, 0, 0, 0, 0, "hipStreamWaitValue64", 0, 0, word, nullptr, value, hipMemcpyDefault}); }
    void hold_write(uint64_t* word, uint64_t value) { hold({Held::write_word, nullptr, 0, 0, 0, 0, "hipStreamWriteValue64", 0, 0, word, nullptr, value, hipMemcpyDefault}); }
    // Issues the held operations in order; one HIP refuses throws naming it, and those after it are dropped with it.
    void flush() {
        if (ops_.empty()) return;
        const HipFn& f = hip();
        std::vector<Held> ops;
        std::vector<uint8_t> bytes;
        ops.swap(ops_);
        bytes.swap(op_bytes_);
        // HIP finds a module's function through the calling thread's device, which another backend's call may have changed.
        hip_check(f.hipSetDevice(dev_.index), "hipSetDevice");
        for (const Held& op : ops) {
            // A submission's busy time starts at its first operation after its waits on other devices (busy_ms).
            if (!started_ && op.kind != Held::wait_word) {
                hip_check(f.hipEventRecord(starts_[(tickets_.last() + 1) % kEvents], stream_), "hipEventRecord");
                started_ = true;
            }
            if (op.kind == Held::kernel) {
                size_t size = op.size;
                void* extra[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, bytes.data() + op.at, HIP_LAUNCH_PARAM_BUFFER_SIZE, &size, HIP_LAUNCH_PARAM_END};
                launch_check(f.hipModuleLaunchKernel(op.f, op.gx, op.gy, op.gz, op.threads, 1, 1, (unsigned)op.bytes, stream_, nullptr, extra), op.name);
            } else if (op.kind == Held::copy) {
                launch_check(f.hipMemcpyAsync(op.dst, op.src, op.bytes, op.dir, stream_), op.name);
            } else if (op.kind == Held::wait_word) {
                launch_check(f.hipStreamWaitValue64(stream_, op.dst, op.bytes, hipStreamWaitValueGte, ~0ull), op.name);
            } else if (op.kind == Held::write_word) {
                launch_check(f.hipStreamWriteValue64(stream_, op.dst, op.bytes, 0), op.name);
            } else {
                launch_check(f.hipMemsetAsync(op.dst, 0, op.bytes, stream_), op.name);
            }
        }
        // The vectors keep their capacity for the next records.
        ops.clear();
        bytes.clear();
        ops_.swap(ops);
        bytes.swap(op_bytes_);
    }
    friend class RocmCollective;
    friend class RocmHandoff;

    RocmDevice dev_;
    std::shared_ptr<Graves> graves_ = std::make_shared<Graves>();   // before every buffer member, so it outlives them
    hipStream_t stream_ = nullptr;
    hipModule_t module_ = nullptr, int16_module_ = nullptr, dot_module_ = nullptr;
    BufferPtr parts_;  // attention's chunk partials, grown as calls need
    BufferPtr sparse_;  // sparse attention's scratch rows, grown as calls need
    // Two twin slots, each the twin of a product's activations or of a producer's output, grown as calls need, so the routed act's twin leaves the mixed streams' for the shared expert (Vulkan's twins, at the count that pattern needs).
    std::array<BufferPtr, 2> twin_{};
    std::array<Kept, 2> kept_{};   // the newer first
    size_t twin_slot_ = 0;   // the slot of the twin the last product read
    BufferPtr moe_out_;   // a routed down projection's slots before they join the residual
    BufferPtr delta_counter_;   // llmx_delta_step's (view, K head) counters, zero between launches
    BufferPtr step_counter_, step_twin_;   // the fused moe_step's finished route blocks, zero between launches, and its shared act's twin
    int step_up_per_cu_[2] = {-1, -1};      // the wide gate+up builds' resident workgroups a compute unit, Q4_K then Q5_K, found at their first step
    BufferPtr moe_tab_;   // a grouped routed call's groups and entries (kernels/f32.hip, llmx_moe_group)
    int want_twin_ = 0;   // the class of the last twin a product read, 1 block-int16 and 2 block-int8, which producers write; 0 before any
    BufferPtr staged_;   // a product bound for host memory, formed here first
    BufferPtr partials_;   // the parts of a split tile call, grown as calls need
    BufferPtr scores_;   // the indexer's scores of 16 rows at most, grown as calls need
    BufferPtr argmax_parts_;   // argmax_rows' slice maxima and ids, grown as calls need
    BufferPtr select_state_;   // the wide selection's histograms and slice counts, zero between calls
    size_t rows_per_block_ = kThreads / dev_.caps.subgroup_size;   // the row kernel's output rows a block, a wave each
    std::array<hipFunction_t, kKernels> kernels_{};
    std::array<hipEvent_t, kEvents> events_{};
    std::array<std::atomic<uint32_t>, kEvents> syncing_{};   // per event slot, the threads in its synchronise (done)
    std::array<hipEvent_t, kEvents> starts_{};   // each ticket's first operation after its cross-device waits, timed against events_ (busy_ms)
    Ring args_, stage_, pass_;   // the argument ring, the staging ring, and pass_arg's ring in device memory
    std::array<Upload, kUploads> uploads_{};
    size_t next_upload_ = 0;
    TicketLedger tickets_;
    bool opened_ = false;
    bool started_ = false;   // the open ticket's start event is recorded
    static constexpr size_t kFirstChunk = 8;
    size_t chunk_ = kFirstChunk;   // the held operations the open submission issues at, kFirstChunk while its device had nothing running
    std::vector<Held> ops_;   // the stream operations held since the last flush
    std::vector<uint8_t> op_bytes_;   // their kernels' arguments
    size_t launched_[kKernels] = {};   // launches of each kernel since rocm_take_launches last read them
    size_t sparse_q_floats_ = 0;   // the query floats llmx_sparse_softmax_g stages in the shared memory its own arrays leave, its kSparseSoftmaxHeads heads' at most
    std::string route_only_;   // rocm_route_only's filter of the routed tile builds, empty for all of them
    std::mutex failed_mu_;
    std::string failed_;
    unsigned* fault_ = nullptr;   // host memory a group sum's kernel sets where it gave up waiting (kernels/f32.hip, llmx_group_sum)
    std::map<std::vector<int>, std::string> group_paths_;   // per group of device indices led by this backend: empty for fast sums, else why sums are conservative
};

RocmKVStorage::RocmKVStorage(RocmBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens, KVType kt, KVType vt,
                             const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools)
    : BlockKVStorage(owner, "rocm", kBlockTokens, layers, heads, dim, max_tokens, kt, vt, auxiliary_widths, auxiliary_pools) {}

// Dropping the old buffer is enough: its memory waits in the graves for the copies recorded before.
void RocmKVStorage::retire(const BufferPtr&) {}

} // namespace
} // namespace backend

#include "backends/rocm/rocm_collective.hpp"

namespace backend {

BackendPtr make_rocm_backend(int device) { return std::make_shared<RocmBackend>(device); }

std::map<std::string, size_t> rocm_take_launches(Backend& backend) { return dynamic_cast<RocmBackend&>(backend).take_launches(); }
void rocm_route_only(Backend& backend, std::string build) { dynamic_cast<RocmBackend&>(backend).route_only(std::move(build)); }

DeviceProfile& rocm_profile(Backend& backend) { return dynamic_cast<RocmBackend&>(backend).profile(); }

} // namespace backend
