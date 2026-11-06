// The ROCm backend (docs/ROCM.md): one HIP stream a device, every op an asynchronous launch on it, a ticket an event recorded on the stream.
// The host never waits inside a pass: it waits only where a caller asks for a ticket, a read, or a ring of host memory it reuses (docs/src/backends-rocm.md).
#include "backends/rocm/rocm_backend.hpp"

#include "backends/kv_storage.hpp"
#include "backends/rocm/rocm_device.hpp"
#include "quant/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <string>
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
constexpr size_t kEvents = 1024;           // tickets in flight before a submission waits for the oldest
constexpr size_t kRingSlots = 4;           // a host ring's slots, each reused once the submission that last read it retires
constexpr size_t kArgBytes = size_t(1) << 20, kStageBytes = size_t(16) << 20;
constexpr size_t kGroupSignalBytes = 4096;  // a group member's signal block before its staging (kernels/f32.hip, GroupSignal)

class RocmBackend;
class RocmCollective;

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

// Memory of dropped buffers, freed once the work recorded before the drop retires, since hipFree does not wait for a launch or copy still queued that names it.
struct Graves {
    std::mutex m;
    std::vector<std::pair<Ticket, std::pair<void*, bool>>> held;   // the ticket that covers the drop, and the memory
    const TicketLedger* tickets = nullptr;                          // null once the backend has drained its stream, when a drop frees at once
    hipStream_t stream = nullptr;                                   // the backend's stream, null once it is drained
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
    }
    // A view of memory register_host holds a count of.
    RocmBuffer(void* memory, size_t bytes, std::shared_ptr<Graves> graves) : p_(memory), size_(bytes), host_(true), registered_(true), graves_(std::move(graves)) {}
    ~RocmBuffer() override {
        if (!p_) return;
        if (registered_) {
            {
                std::lock_guard<std::mutex> lock(graves_->m);
                if (graves_->stream) (void)hip().hipStreamSynchronize(graves_->stream);
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
private:
    void* p_ = nullptr;
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
                  const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools);
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
        hip_check(f.hipHostMalloc((void**)&fault_, sizeof(unsigned), hipHostMallocCoherent), "hipHostMalloc");
        *fault_ = 0;
        for (auto& e : events_) hip_check(f.hipEventCreateWithFlags(&e, hipEventDisableTiming), "hipEventCreateWithFlags");
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
            hip_check(f.hipModuleGetFunction(&kernels_[k], m, kKernelNames[k]), "hipModuleGetFunction");
        }
        for (size_t s = 0; s < kRingSlots; ++s) {
            args_.slots[s] = std::make_shared<RocmBuffer>(kArgBytes, true, graves_);
            stage_.slots[s] = std::make_shared<RocmBuffer>(kStageBytes, true, graves_);
            pass_.slots[s] = std::make_shared<RocmBuffer>(kArgBytes, false, graves_);
        }
    }
    ~RocmBackend() override {
        const HipFn& f = hip();
        (void)f.hipSetDevice(dev_.index);
        (void)f.hipStreamSynchronize(stream_);
        {
            std::lock_guard<std::mutex> lock(graves_->m);
            graves_->tickets = nullptr;
            graves_->stream = nullptr;
            for (const auto& g : graves_->held) free_memory(g.second.first, g.second.second);
            graves_->held.clear();
        }
        parts_ = {};
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
        if (stream_) (void)f.hipStreamDestroy(stream_);
        if (fault_) (void)f.hipHostFree(fault_);
    }

    std::string identity() const override { return "rocm:" + std::to_string(dev_.index) + " " + dev_.caps.device + " " + dev_.pci_bus; }
    std::string pci_root() const override { return dev_.pci_root; }
    void set_threads(int) override {}
    int threads_available() const override { return 0; }
    std::optional<size_t> memory_available() const override {
        size_t free = 0, total = 0;
        if (hip().hipSetDevice(dev_.index) != hipSuccess || hip().hipMemGetInfo(&free, &total) != hipSuccess) return std::nullopt;
        return free;
    }
    // HIP rounds each device allocation to its pages (2 MiB past 1 MiB) and keeps memory of its own, and the kernels' scratch grows with a pass, so the fit leaves the share Vulkan leaves.
    // A group's signal block is a fixed size beside the staging collective_bytes_per_row counts.
    size_t scratch_reserve(size_t free) const override { return ((size_t)256 << 20) + kGroupSignalBytes + free / 20; }
    std::unique_ptr<Collective> join(const std::vector<Backend*>& members, size_t rows, size_t width) override;
    // A member's partial rows and a staging slot for each member's (RocmCollective).
    size_t collective_bytes_per_row(size_t members, size_t width) const override {
        return members ? size_mul(members + 1, size_mul(width, sizeof(float))) : 0;
    }
    bool supports_type(uint32_t type) const override { return type_kernels(type) != nullptr; }
    bool implements(Op op) const override {
        return op == Op::norm_rope_partial || op == Op::sigmoid_mul || op == Op::gated_rms_norm || op == Op::causal_conv_silu ||
               op == Op::gated_delta_rule || op == Op::routed_experts || op == Op::group_rms_norm || op == Op::silu_scaled || op == Op::mix_streams ||
               op == Op::inject_streams || op == Op::signed_sqrt_gate || op == Op::dilated_conv_silu_add || op == Op::select_kv || op == Op::sparse_attention || op == Op::argmax_rows || op == Op::embed_ids;
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
        auto b = std::make_shared<RocmBuffer>(bytes, where == Memory::host_visible, graves_);
        kept_ = {};
        if (!bytes) return b;
        // KV growth and the model read a fresh buffer as zeros.
        if (where == Memory::host_visible) std::memset(b->data(), 0, bytes);
        else launch_check(hip().hipMemsetAsync(b->data(), 0, bytes, stream_), "hipMemsetAsync");
        return b;
    }
    // One registration serves every device's view of the memory; null where HIP refuses it, so the caller writes through staging.
    BufferPtr wrap_host(void* memory, size_t bytes) override {
        if (!memory || !bytes) return nullptr;
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        if (!register_host(memory, bytes)) return nullptr;
        return std::make_shared<RocmBuffer>(memory, bytes, graves_);
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
    Ticket record() {
        // The event is recorded before the ticket is handed out, so a waiter on another thread never reads a slot not yet recorded for it.
        const Ticket t = tickets_.last() + 1;
        const size_t i = t % kEvents;
        if (t > kEvents) wait(t - kEvents);
        bury();
        hip_check(hip().hipEventRecord(events_[i], stream_), "hipEventRecord");
        tickets_.take();
        opened_ = false;
        for (Ring* r : {&args_, &stage_, &pass_}) r->tag(t);
        return t;
    }
    void wait(Ticket t) noexcept override {
        if (tickets_.settled(t)) return;
        if (t + kEvents <= tickets_.last()) return tickets_.retired(t);
        const hipError_t e = hip().hipEventSynchronize(events_[t % kEvents]);
        if (e != hipSuccess) remember(e, "a submission");
        tickets_.retired(t);
        bury();
    }
    bool done(Ticket t) noexcept override {
        if (tickets_.settled(t)) return true;
        // A slot submitted over again held an earlier ticket that submit waited for; a slot taken while this reads it only reads as later.
        if (t + kEvents <= tickets_.last()) {
            tickets_.retired(t);
            return true;
        }
        const hipError_t e = hip().hipEventQuery(events_[t % kEvents]);
        if (e == hipErrorNotReady) return false;
        if (e != hipSuccess) remember(e, "a submission");
        tickets_.retired(t);
        return true;
    }
    void sync() noexcept override {
        if (opened_) {
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
            launch_check(hip().hipMemcpyAsync(st, (const uint8_t*)src.data() + off + done, n, hipMemcpyDeviceToHost, stream_), "hipMemcpyAsync");
            wait(submit());
            raise_failed();
            std::memcpy((uint8_t*)dst + done, st, n);
            done += n;
        }
    }
    void write(Buffer& dst_b, size_t off, const void* src, size_t bytes) override {
        const RocmBuffer& dst = as_rocm(dst_b);
        span(dst, off, bytes);
        kept_ = {};
        for (size_t done = 0; done < bytes;) {
            const size_t n = std::min(bytes - done, kStageBytes);
            void* st = stage_.take(*this, n);
            std::memcpy(st, (const uint8_t*)src + done, n);
            launch_check(hip().hipMemcpyAsync((uint8_t*)dst.data() + off + done, st, n, hipMemcpyHostToDevice, stream_), "hipMemcpyAsync");
            done += n;
        }
    }
    void copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off, size_t bytes) override {
        const RocmBuffer& dst = as_rocm(dst_b);
        const RocmBuffer& src = as_rocm(src_b);
        span(dst, dst_off, bytes);
        span(src, src_off, bytes);
        if (!bytes) return;
        kept_ = {};
        launch_check(hip().hipMemcpyAsync((uint8_t*)dst.data() + dst_off, (const uint8_t*)src.data() + src_off, bytes, hipMemcpyDefault, stream_),
                     "hipMemcpyAsync");
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
        void* args[] = {&lp, &ip, &pp, &nn, &sl, &ns, &ap};
        launch(kArgmaxSlices, (unsigned)(rows * nslice), 1, args);
        if (nslice > 1) {
            void* margs[] = {&pp, &ip, &nn, &ns, &ap};
            launch(kArgmaxMerge, (unsigned)rows, 1, margs, 1, 64);
        }
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
            void* args[] = {&d, &s, &wp, &in, &st, &eps};
            return launch(kRmsNorm, (unsigned)rows, 1, args);
        }
        // Rows of up to 8192 values take a held build, which writes the twin beside them where a product reads one next.
        const TwinOut t = stride == n && n % 32 == 0 ? twin_out(dst, rows * n, twin_map(runs, rows, n)) : TwinOut{};
        const bool wide = n > 16 * kThreads;
        const Kernel k = !t.q ? (wide ? kRmsNorm32 : kRmsNorm16) : t.int8 ? (wide ? kRmsNorm32Twin8 : kRmsNorm16Twin8) : (wide ? kRmsNorm32Twin16 : kRmsNorm16Twin16);
        void* q = t.q;
        float* td = t.d;
        TwinMap map = t.map;
        void* args[] = {&d, &s, &wp, &in, &st, &eps, &q, &td, &map};
        launch(twin_build(k, map), (unsigned)rows, 1, args, 1, kNormThreads);
        keep(t, d, rows * n);
    }
    void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride, size_t heads, size_t head_dim,
                           size_t rope_dim, CSlice w, float eps, CSlice cos, CSlice sin, const uint32_t* pos, RopeStreams streams) override {
        if (!heads || !head_dim || !rope_dim || rope_dim % 2 || rope_dim > head_dim)
            throw std::runtime_error("rocm: invalid partial rope dimensions");
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
        if (streams.count == 1) {
            void* args[] = {&d, &s, &dst_stride, &ss, &shs, &hd, &hf, &wp, &eps, &c, &sn, &p};
            return launch(kNormRope, (unsigned)rows, (unsigned)heads, args);
        }
        for (size_t i = 0; i < half; ++i)
            if (streams.of[i] >= streams.count) throw std::runtime_error("rocm: a rope pair of a stream past the row's");
        const unsigned char* of = (const unsigned char*)pass_arg(streams.of, half);
        int count = (int)streams.count;
        void* args[] = {&d, &s, &dst_stride, &ss, &shs, &hd, &hf, &wp, &eps, &c, &sn, &p, &of, &count};
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
        void* args[] = {&d, &xp, &g, &nn, &wd, &dd, &gs, &ghs, &t.q, &t.d, &t.level, &t.map, &t.major};
        launch(kSigmoidMul, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
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
        void* args[] = {&d, &xp, &zp, &wp, &dm, &eps, &sig, &t.q, &t.d, &t.level, &t.map, &t.major};
        launch(kGatedRmsNorm, (unsigned)size_mul(rows, heads), 1, args);
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
        void* args[] = {&o, &u, &al, &bp, &ap, &dt, &st, &tp, &cc, &hk, &hv, &dk, &dv, &slot, &nb, &scale, &eps};
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
        void* args[] = {&o, &xp, &wp, &st, &tp, &cc, &slot, &carried, &chunk};
        launch(kCausalConvSilu, (unsigned)((C + kThreads - 1) / kThreads), (unsigned)chunks, args);
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
        void* args[] = {&d, &s, &wp, &g, &dm, &eps, &t.q, &t.d, &t.level, &t.map, &t.major};
        // Slices of a group, each repeating its reduction, up to what gives four blocks a compute unit (Vulkan's split, fc6f53415); one in place, where a slice's writes would meet another's reads.
        const size_t fill = (4 * size_t(std::max(1u, dev_.caps.compute_units)) + blocks - 1) / blocks;
        const size_t slices = dst.buffer == src.buffer && dst.offset == src.offset ? 1 : std::max<size_t>(1, std::min((dim + 1023) / 1024, fill));
        launch(kGroupRmsNorm, (unsigned)blocks, (unsigned)slices, args);
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
        void* args[] = {&d, &s, &nn, &scale, &t.q, &t.d, &t.level, &t.map, &t.major};
        launch(kSiluScaled, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
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
        void* args[] = {&d, &s, &g, &nn, &st, &dm, &t.q, &t.d, &t.level, &t.map, &t.major};
        launch(kMixStreams, (unsigned)((out_n + kThreads - 1) / kThreads), 1, args);
        keep(t.t, d, out_n);
    }
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
        void* args[] = {&d, &b, &g, &st, &dm};
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
        void* args[] = {&d, &k, &q, &v, &g, &dm, &scale, &kept};
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
        void* args[] = {&o, &xp, &rp, &wp, &st, &tp, &cc, &tt, &dl, &hs, &sl, &pf, &ch};
        launch(kDilatedConvSiluAdd, (unsigned)((channels + kThreads - 1) / kThreads), (unsigned)chunks, args);
    }
    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, RowRuns runs) override {
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
        void* args[] = {&d, &g, &u, &nn, &q, &td, &map};
        launch(twin_build(!t.q ? kSiluMul : t.int8 ? kSiluMulTwin8 : kSiluMulTwin16, map), (unsigned)((n + kThreads - 1) / kThreads), 1, args);
        keep(t, d, n);
    }
    // A row a block of 64, its reductions in one tree of 64 pairs (kernels/f32.hip, llmx_moe_route).
    void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize, Slice ids, Slice weights) override {
        if (!k || k > n_expert || k > 256 || n_expert > 1024) throw std::runtime_error("rocm: routing takes 1 to 256 of at most 1024 experts");
        if (!rows) return;
        span(*scores.buffer, scores.offset * sizeof(float), size_mul(size_mul(rows, n_expert), sizeof(float)));
        for (const Slice& s : {ids, weights}) span(*s.buffer, s.offset * sizeof(float), size_mul(size_mul(rows, k), sizeof(float)));
        const float* sc = f32(scores);
        float* id = f32(ids);
        float* w = f32(weights);
        unsigned ne = (unsigned)n_expert, kk = (unsigned)k;
        int norm = normalize;
        void* args[] = {&sc, &id, &w, &ne, &kk, &norm};
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
        void* args[] = {&y, &part, &w, &id, &nr, &no, &kk};
        launch_writing({{y, size_mul(nrows, nout)}}, kMoeCombine, (unsigned)((nrows * nout + kThreads - 1) / kThreads), 1, args);
    }
    void add(Slice dst, CSlice src, size_t n) override {
        if (!n) return;
        for (const CSlice& s : {CSlice(dst), src}) span(*s.buffer, s.offset * sizeof(float), size_mul(n, sizeof(float)));
        float* d = f32(dst);
        const float* s = f32(src);
        unsigned long long nn = n;
        void* args[] = {&d, &s, &nn};
        launch(kAdd, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
    }

    KVLayout kv_layout() const override { return {kBlockTokens}; }
    std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim, size_t max_tokens, KVType k_type, KVType v_type,
                                        const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools) override {
        if (layers == 0 || n_head_kv == 0 || head_dim == 0) throw std::runtime_error("rocm: invalid KV storage shape");
        if (head_dim > kThreads) throw std::runtime_error("rocm: a KV head wider than " + std::to_string(kThreads) + " is not implemented");
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
        void* args[] = {&kc, &vc, &kp, &vp, &bp, &sp, &heads, &dim, &bt, &kh, &vh};
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
        void* args[] = {&qp, &qs, &nh, &qw, &kp, &vp, &nkv, &kw, &hf, &eps, &c, &sn, &p, &kc, &vc, &bp, &sp, &bt, &kh, &vh};
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
            void* args[] = {&q, &o, &kc, &vc, &tp, &wv, &nviews, &n_head, &n_head_kv, &bt, &kh, &vh, &scale};
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
            void* args[] = {&q, &o, &kc, &vc, &tp, &fp, &ep, &n_head, &n_head_kv, &bt, &kh, &vh, &scale, &nsplit, &chunk, &max_parts, &vp, &tq, &td,
                            nsplit > 1 ? &none : &tl, &map};
            launch(twin_build(head_dim == 256 ? kAttentionVec256 : kAttentionVec, map), (unsigned)(rows * n_head * nsplit), 1, args);
            if (nsplit > 1) {
                void* margs[] = {&vp, &o, &ep, &n_head, &nsplit, &tq, &td, &tl, &map};
                launch(twin_build(head_dim == 256 ? kAttentionVecMerge256 : kAttentionVecMerge, map), (unsigned)(rows * n_head), 1, margs);
            }
            keep(t, o, floats);
            return;
        }
        if (narrow) {
            void* args[] = {&q, &pp, &kc, &vc, &tp, &fp, &ep, &n_head, &ratio, &n_head_kv, &head_dim, &bt, &kh, &vh, &scale};
            launch(kAttentionChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        } else {
            return;
        }
        void* merge[] = {&pp, &o, &ep, &n_head, &head_dim};
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
            void* args[] = {&q, &o, &kc, &vc, &tp, &fp, &ep, &n_head, &n_head_kv, &bt, &kh, &vh, &scale, &nsplit, &chunk, &vp, &idp, &stride};
            // Four heads of a KV head a block, each head its one-head bits (kernels/f32.hip, sparse_vec_grouped), where the groups fill the compute units, as the Vulkan backend chooses.
            if (ratio % 4 == 0 && rows * size_t(n_head) / 4 >= size_t(dev_.caps.compute_units)) launch(head_dim == 256 ? kSparseVec256G4 : kSparseVecG4, (unsigned)(rows * (n_head / 4) * nsplit), 1, args);
            else launch(head_dim == 256 ? kSparseVec256 : kSparseVec, (unsigned)(rows * n_head * nsplit), 1, args);
            if (nsplit > 1) {
                void* margs[] = {&vp, &o, &ep, &n_head, &nsplit};
                launch(head_dim == 256 ? kSparseVecMerge256 : kSparseVecMerge, (unsigned)(rows * n_head), 1, margs);
            }
            return;
        }
        const size_t need = size_mul(size_mul(size_mul(rows, (size_t)n_head), kChunks), size_mul((size_t)head_dim + 2, sizeof(float)));
        if (!parts_ || parts_->size() < need) parts_ = alloc(need, Memory::device);
        float* pp = (float*)as_rocm(*parts_).data();
        void* args[] = {&q, &pp, &kc, &vc, &tp, &fp, &ep, &n_head, &ratio, &n_head_kv, &head_dim, &bt, &kh, &vh, &scale, &idp, &stride};
        launch(kSparseChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        void* margs[] = {&pp, &o, &ep, &n_head, &head_dim, &stride};
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
                    void* sargs[] = {&qr, &aux, &sc, &blocks, &hist, &groups, &heads, &dim, &compression, &bt, &block_floats, &rows_floats, &scale};
                    launch(kKvIndexScores, u32((count + 7) / 8), (unsigned)rows, sargs);
                }
                if (!wide) {
                    void* args[] = {&sc, &o, &hist, &groups, &compression, &budget, &ostride};
                    launch(kKvIndexSelect, (unsigned)rows, 1, args);
                    continue;
                }
                for (unsigned stage = 0; stage < 5; ++stage) {
                    void* args[] = {&sc, &o, &state, &hist, &groups, &compression, &budget, &ostride, &stage, &slices};
                    launch(kKvIndexSelectWide, slices, (unsigned)rows, args);
                }
            }
        }
    }

    DeviceProfile& profile() { return dev_.profile; }

    // The kernels launched since the last reading, by name, and their counts.
    std::map<std::string, size_t> take_launches() {
        std::map<std::string, size_t> m;
        for (size_t k = 0; k < kKernels; ++k)
            if (launched_[k]) m[kKernelNames[k]] = launched_[k];
        std::fill(std::begin(launched_), std::end(launched_), size_t(0));
        return m;
    }

private:
    // The F32 module's kernels, then from kRows16x1 the int16 module's, which only read twins.
    enum Kernel { kAdd, kSiluMul, kSiluMulTwin16, kSiluMulTwin8, kRmsNorm, kRmsNorm16, kRmsNorm16Twin16, kRmsNorm16Twin8, kRmsNorm32, kRmsNorm32Twin16,
                  kRmsNorm32Twin8, kNormRope, kNormRopeKv, kGather, kGatherQ8, kMatmul, kMatmulQ8, kMatmulQ4K, kGatherQ4K, kMatmulQ5K, kGatherQ5K, kMatmulQ6K,
                  kGatherQ6K, kMatmulQ50, kGatherQ50, kMatmulQ51, kGatherQ51, kMatmulBF16, kGatherBF16, kMatmulIQ4NL, kGatherIQ4NL, kKvWrite, kAttentionChunk, kAttentionTile, kAttentionMerge, kAttentionVec,
                  kAttentionVecMerge, kTwin16, kTwin8, kSiluMulTwin16Major, kSiluMulTwin8Major, kRmsNorm16Twin16Major, kRmsNorm16Twin8Major,
                  kRmsNorm32Twin16Major, kRmsNorm32Twin8Major, kAttentionVecMajor, kAttentionVecMergeMajor, kTwin16Major, kTwin8Major, kGroupSum, kSigmoidMul, kGatedRmsNorm, kNormRopeStreams, kCausalConvSilu, kGatedDeltaRule, kGatedDeltaRuleShort, kGroupRmsNorm, kSiluScaled, kMixStreams, kInjectStreams, kSignedSqrtGate, kDilatedConvSiluAdd, kKvIndexPool, kKvIndexScores, kKvIndexSelect, kKvIndexSelectWide, kSparseChunk, kSparseMerge, kSparseVec, kSparseVec256, kSparseVecMerge, kSparseVecMerge256,
                  kAttentionVec256, kAttentionVec256Major, kAttentionVecMerge256, kAttentionVecMerge256Major, kArgmaxSlices, kArgmaxMerge,
                  kMoeRoute, kMoeCombine, kMoeGroup, kRoutedExactF32, kRoutedExactQ8, kRoutedExactQ4K, kRoutedExactQ5K, kRoutedExactQ6K, kRoutedExactQ50, kRoutedExactQ51, kRoutedExactBF16, kRoutedExactIQ4NL,
                  kRowsExactF32, kRowsExactQ8, kRowsExactQ4K, kRowsExactQ5K, kRowsExactQ6K, kRowsExactQ50, kRowsExactQ51, kRowsExactBF16, kRowsExactIQ4NL, kSparseVecG4, kSparseVec256G4,
                  kRows16x1, kRows16x2, kRows16x2r4, kRows16x4, kRows16x4r4, kRows16x8, kRows16x8r4, kRows16x16, kTile16,
                  kRows8x1, kRows8x2, kRows8x2r4, kRows8x4, kRows8x4r4, kRows8x8, kRows8x8r4, kRows8x16, kTile8, kTileReduce,
                  kHalf8x1, kHalf8x1p, kHalf8x1r2, kHalf8x2, kHalf8x2r4, kHalf8x4, kHalf8x4r4, kHalf8x8, kHalf8x8r4, kHalf8x16, kTile16Tall, kTile8Tall, kRouted16, kRouted8, kRoutedHalf8, kGrouped16x2, kGrouped16x4, kGrouped16x8, kGrouped8x2, kGrouped8x4, kGrouped8x8, kGroupedHalf8x2, kGroupedHalf8x4, kGroupedHalf8x8, kRoutedTile16, kRoutedTile8, kRoutedNarrow16, kRoutedNarrow8,
                  kRowsQ4K16x1, kRowsQ4K16x2, kRowsQ4K16x4, kRowsQ4K16x8, kRowsQ4K16x16, kRowsQ4K8x1, kRowsQ4K8x2, kRowsQ4K8x4, kRowsQ4K8x8, kRowsQ4K8x16, kTileQ4K16, kTileQ4K8, kRoutedQ4K16, kRoutedQ4K8, kGroupedQ4K16x2, kGroupedQ4K16x4, kGroupedQ4K16x8, kGroupedQ4K8x2, kGroupedQ4K8x4, kGroupedQ4K8x8, kRoutedTileQ4K16, kRoutedTileQ4K8, kRoutedNarrowQ4K16, kRoutedNarrowQ4K8, kRowsQ5K16x1, kRowsQ5K16x2, kRowsQ5K16x4, kRowsQ5K16x8, kRowsQ5K16x16, kRowsQ5K8x1, kRowsQ5K8x2, kRowsQ5K8x4, kRowsQ5K8x8, kRowsQ5K8x16, kTileQ5K16, kTileQ5K8, kRoutedQ5K16, kRoutedQ5K8, kGroupedQ5K16x2, kGroupedQ5K16x4, kGroupedQ5K16x8, kGroupedQ5K8x2, kGroupedQ5K8x4, kGroupedQ5K8x8, kRoutedTileQ5K16, kRoutedTileQ5K8, kRoutedNarrowQ5K16, kRoutedNarrowQ5K8, kRowsQ6K16x1, kRowsQ6K16x2, kRowsQ6K16x4, kRowsQ6K16x8, kRowsQ6K16x16, kRowsQ6K8x1, kRowsQ6K8x2, kRowsQ6K8x4, kRowsQ6K8x8, kRowsQ6K8x16, kTileQ6K16, kTileQ6K8, kRoutedQ6K16, kRoutedQ6K8, kGroupedQ6K16x2, kGroupedQ6K16x4, kGroupedQ6K16x8, kGroupedQ6K8x2, kGroupedQ6K8x4, kGroupedQ6K8x8, kRoutedTileQ6K16, kRoutedTileQ6K8, kRoutedNarrowQ6K16, kRoutedNarrowQ6K8,
                  kRowsQ5016x1, kRowsQ5016x2, kRowsQ5016x4, kRowsQ5016x8, kRowsQ5016x16, kRowsQ508x1, kRowsQ508x2, kRowsQ508x4, kRowsQ508x8, kRowsQ508x16, kTileQ5016, kTileQ508, kRoutedQ5016, kRoutedQ508, kGroupedQ5016x2, kGroupedQ5016x4, kGroupedQ5016x8, kGroupedQ508x2, kGroupedQ508x4, kGroupedQ508x8, kRoutedTileQ5016, kRoutedTileQ508, kRoutedNarrowQ5016, kRoutedNarrowQ508, kRowsQ5116x1, kRowsQ5116x2, kRowsQ5116x4, kRowsQ5116x8, kRowsQ5116x16, kRowsQ518x1, kRowsQ518x2, kRowsQ518x4, kRowsQ518x8, kRowsQ518x16, kTileQ5116, kTileQ518, kRoutedQ5116, kRoutedQ518, kGroupedQ5116x2, kGroupedQ5116x4, kGroupedQ5116x8, kGroupedQ518x2, kGroupedQ518x4, kGroupedQ518x8, kRoutedTileQ5116, kRoutedTileQ518, kRoutedNarrowQ5116, kRoutedNarrowQ518, kRowsIQ4NL16x1, kRowsIQ4NL16x2, kRowsIQ4NL16x4, kRowsIQ4NL16x8, kRowsIQ4NL16x16, kRowsIQ4NL8x1, kRowsIQ4NL8x2, kRowsIQ4NL8x4, kRowsIQ4NL8x8, kRowsIQ4NL8x16, kTileIQ4NL16, kTileIQ4NL8, kRoutedIQ4NL16, kRoutedIQ4NL8, kGroupedIQ4NL16x2, kGroupedIQ4NL16x4, kGroupedIQ4NL16x8, kGroupedIQ4NL8x2, kGroupedIQ4NL8x4, kGroupedIQ4NL8x8, kRoutedTileIQ4NL16, kRoutedTileIQ4NL8, kRoutedNarrowIQ4NL16, kRoutedNarrowIQ4NL8,
                  kQuarterQ5016x1, kQuarterQ5016x2, kQuarterQ5016x4, kQuarterQ5016x8, kQuarterQ5016x16, kQuarterQ508x1, kQuarterQ508x2, kQuarterQ508x4, kQuarterQ508x8, kQuarterQ508x16, kQuarterRoutedQ5016, kQuarterRoutedQ508, kQuarterGroupedQ5016x2, kQuarterGroupedQ5016x4, kQuarterGroupedQ5016x8, kQuarterGroupedQ508x2, kQuarterGroupedQ508x4, kQuarterGroupedQ508x8, kQuarterQ5116x1, kQuarterQ5116x2, kQuarterQ5116x4, kQuarterQ5116x8, kQuarterQ5116x16, kQuarterQ518x1, kQuarterQ518x2, kQuarterQ518x4, kQuarterQ518x8, kQuarterQ518x16, kQuarterRoutedQ5116, kQuarterRoutedQ518, kQuarterGroupedQ5116x2, kQuarterGroupedQ5116x4, kQuarterGroupedQ5116x8, kQuarterGroupedQ518x2, kQuarterGroupedQ518x4, kQuarterGroupedQ518x8, kKernels };
    static constexpr const char* kKernelNames[kKernels] = {"llmx_add", "llmx_silu_mul", "llmx_silu_mul_twin16", "llmx_silu_mul_twin8", "llmx_rms_norm_rows",
                                                           "llmx_rms_norm_16", "llmx_rms_norm_16_twin16", "llmx_rms_norm_16_twin8", "llmx_rms_norm_32",
                                                           "llmx_rms_norm_32_twin16", "llmx_rms_norm_32_twin8", "llmx_norm_rope", "llmx_norm_rope_kv", "llmx_gather",
                                                           "llmx_gather_q8_0", "llmx_matmul_f32", "llmx_matmul_q8_0", "llmx_matmul_q4_k", "llmx_gather_q4_k",
                                                           "llmx_matmul_q5_k", "llmx_gather_q5_k", "llmx_matmul_q6_k", "llmx_gather_q6_k", "llmx_matmul_q5_0",
                                                           "llmx_gather_q5_0", "llmx_matmul_q5_1", "llmx_gather_q5_1", "llmx_matmul_bf16", "llmx_gather_bf16",
                                                           "llmx_matmul_iq4_nl", "llmx_gather_iq4_nl", "llmx_kv_write",
                                                           "llmx_attention_chunk", "llmx_attention_tile", "llmx_attention_merge", "llmx_attention_vec", "llmx_attention_vec_merge",
                                                           "llmx_twin16", "llmx_twin8", "llmx_silu_mul_twin16_major", "llmx_silu_mul_twin8_major",
                                                           "llmx_rms_norm_16_twin16_major", "llmx_rms_norm_16_twin8_major", "llmx_rms_norm_32_twin16_major",
                                                           "llmx_rms_norm_32_twin8_major", "llmx_attention_vec_major", "llmx_attention_vec_merge_major",
                                                           "llmx_twin16_major", "llmx_twin8_major", "llmx_group_sum", "llmx_sigmoid_mul", "llmx_gated_rms_norm", "llmx_norm_rope_streams", "llmx_causal_conv_silu", "llmx_gated_delta_rule", "llmx_gated_delta_rule_short", "llmx_group_rms_norm", "llmx_silu_scaled", "llmx_mix_streams", "llmx_inject_streams", "llmx_signed_sqrt_gate", "llmx_dilated_conv_silu_add", "llmx_kv_index_pool", "llmx_kv_index_scores", "llmx_kv_index_select", "llmx_kv_index_select_wide", "llmx_sparse_attention_chunk", "llmx_sparse_attention_merge", "llmx_sparse_attention_vec", "llmx_sparse_attention_vec256", "llmx_sparse_attention_vec_merge", "llmx_sparse_attention_vec_merge256", "llmx_attention_vec256", "llmx_attention_vec256_major", "llmx_attention_vec_merge256", "llmx_attention_vec_merge256_major", "llmx_argmax_slices", "llmx_argmax_merge", "llmx_moe_route", "llmx_moe_combine", "llmx_moe_group", "llmx_routed_f32", "llmx_routed_q8_0", "llmx_routed_q4_k", "llmx_routed_q5_k", "llmx_routed_q6_k", "llmx_routed_q5_0", "llmx_routed_q5_1", "llmx_routed_bf16", "llmx_routed_iq4_nl", "llmx_matmul_rows_f32", "llmx_matmul_rows_q8_0", "llmx_matmul_rows_q4_k", "llmx_matmul_rows_q5_k", "llmx_matmul_rows_q6_k", "llmx_matmul_rows_q5_0", "llmx_matmul_rows_q5_1", "llmx_matmul_rows_bf16", "llmx_matmul_rows_iq4_nl", "llmx_sparse_attention_vec_g4", "llmx_sparse_attention_vec256_g4", "llmx_q8_0_rows16_1",
                                                           "llmx_q8_0_rows16_2", "llmx_q8_0_rows16_2_r4", "llmx_q8_0_rows16_4", "llmx_q8_0_rows16_4_r4",
                                                           "llmx_q8_0_rows16_8", "llmx_q8_0_rows16_8_r4", "llmx_q8_0_rows16_16", "llmx_q8_0_tile16", "llmx_q8_0_rows8_1",
                                                           "llmx_q8_0_rows8_2", "llmx_q8_0_rows8_2_r4", "llmx_q8_0_rows8_4", "llmx_q8_0_rows8_4_r4", "llmx_q8_0_rows8_8",
                                                           "llmx_q8_0_rows8_8_r4", "llmx_q8_0_rows8_16", "llmx_q8_0_tile8", "llmx_tile_reduce", "llmx_q8_0_half8_1", "llmx_q8_0_half8_1p", "llmx_q8_0_half8_1r2",
                                                           "llmx_q8_0_half8_2", "llmx_q8_0_half8_2_r4", "llmx_q8_0_half8_4", "llmx_q8_0_half8_4_r4", "llmx_q8_0_half8_8",
                                                           "llmx_q8_0_half8_8_r4", "llmx_q8_0_half8_16", "llmx_q8_0_tile16_tall", "llmx_q8_0_tile8_tall", "llmx_q8_0_routed16", "llmx_q8_0_routed8", "llmx_q8_0_routed_half8", "llmx_q8_0_routed16_g2", "llmx_q8_0_routed16_g4", "llmx_q8_0_routed16_g8", "llmx_q8_0_routed8_g2", "llmx_q8_0_routed8_g4", "llmx_q8_0_routed8_g8", "llmx_q8_0_routed_half8_g2", "llmx_q8_0_routed_half8_g4", "llmx_q8_0_routed_half8_g8", "llmx_q8_0_routed_tile16", "llmx_q8_0_routed_tile8", "llmx_q8_0_routed_narrow16", "llmx_q8_0_routed_narrow8",
                                                           "llmx_q4_k_rows16_1", "llmx_q4_k_rows16_2", "llmx_q4_k_rows16_4", "llmx_q4_k_rows16_8", "llmx_q4_k_rows16_16", "llmx_q4_k_rows8_1", "llmx_q4_k_rows8_2", "llmx_q4_k_rows8_4", "llmx_q4_k_rows8_8", "llmx_q4_k_rows8_16", "llmx_q4_k_tile16", "llmx_q4_k_tile8", "llmx_q4_k_routed16", "llmx_q4_k_routed8", "llmx_q4_k_routed16_g2", "llmx_q4_k_routed16_g4", "llmx_q4_k_routed16_g8", "llmx_q4_k_routed8_g2", "llmx_q4_k_routed8_g4", "llmx_q4_k_routed8_g8", "llmx_q4_k_routed_tile16", "llmx_q4_k_routed_tile8", "llmx_q4_k_routed_narrow16", "llmx_q4_k_routed_narrow8", "llmx_q5_k_rows16_1", "llmx_q5_k_rows16_2", "llmx_q5_k_rows16_4", "llmx_q5_k_rows16_8", "llmx_q5_k_rows16_16", "llmx_q5_k_rows8_1", "llmx_q5_k_rows8_2", "llmx_q5_k_rows8_4", "llmx_q5_k_rows8_8", "llmx_q5_k_rows8_16", "llmx_q5_k_tile16", "llmx_q5_k_tile8", "llmx_q5_k_routed16", "llmx_q5_k_routed8", "llmx_q5_k_routed16_g2", "llmx_q5_k_routed16_g4", "llmx_q5_k_routed16_g8", "llmx_q5_k_routed8_g2", "llmx_q5_k_routed8_g4", "llmx_q5_k_routed8_g8", "llmx_q5_k_routed_tile16", "llmx_q5_k_routed_tile8", "llmx_q5_k_routed_narrow16", "llmx_q5_k_routed_narrow8", "llmx_q6_k_rows16_1", "llmx_q6_k_rows16_2", "llmx_q6_k_rows16_4", "llmx_q6_k_rows16_8", "llmx_q6_k_rows16_16", "llmx_q6_k_rows8_1", "llmx_q6_k_rows8_2", "llmx_q6_k_rows8_4", "llmx_q6_k_rows8_8", "llmx_q6_k_rows8_16", "llmx_q6_k_tile16", "llmx_q6_k_tile8", "llmx_q6_k_routed16", "llmx_q6_k_routed8", "llmx_q6_k_routed16_g2", "llmx_q6_k_routed16_g4", "llmx_q6_k_routed16_g8", "llmx_q6_k_routed8_g2", "llmx_q6_k_routed8_g4", "llmx_q6_k_routed8_g8", "llmx_q6_k_routed_tile16", "llmx_q6_k_routed_tile8", "llmx_q6_k_routed_narrow16", "llmx_q6_k_routed_narrow8",
                                                           "llmx_q5_0_rows16_1", "llmx_q5_0_rows16_2", "llmx_q5_0_rows16_4", "llmx_q5_0_rows16_8", "llmx_q5_0_rows16_16", "llmx_q5_0_rows8_1", "llmx_q5_0_rows8_2", "llmx_q5_0_rows8_4", "llmx_q5_0_rows8_8", "llmx_q5_0_rows8_16", "llmx_q5_0_tile16", "llmx_q5_0_tile8", "llmx_q5_0_routed16", "llmx_q5_0_routed8", "llmx_q5_0_routed16_g2", "llmx_q5_0_routed16_g4", "llmx_q5_0_routed16_g8", "llmx_q5_0_routed8_g2", "llmx_q5_0_routed8_g4", "llmx_q5_0_routed8_g8", "llmx_q5_0_routed_tile16", "llmx_q5_0_routed_tile8", "llmx_q5_0_routed_narrow16", "llmx_q5_0_routed_narrow8", "llmx_q5_1_rows16_1", "llmx_q5_1_rows16_2", "llmx_q5_1_rows16_4", "llmx_q5_1_rows16_8", "llmx_q5_1_rows16_16", "llmx_q5_1_rows8_1", "llmx_q5_1_rows8_2", "llmx_q5_1_rows8_4", "llmx_q5_1_rows8_8", "llmx_q5_1_rows8_16", "llmx_q5_1_tile16", "llmx_q5_1_tile8", "llmx_q5_1_routed16", "llmx_q5_1_routed8", "llmx_q5_1_routed16_g2", "llmx_q5_1_routed16_g4", "llmx_q5_1_routed16_g8", "llmx_q5_1_routed8_g2", "llmx_q5_1_routed8_g4", "llmx_q5_1_routed8_g8", "llmx_q5_1_routed_tile16", "llmx_q5_1_routed_tile8", "llmx_q5_1_routed_narrow16", "llmx_q5_1_routed_narrow8",
                                                           "llmx_iq4_nl_rows16_1", "llmx_iq4_nl_rows16_2", "llmx_iq4_nl_rows16_4", "llmx_iq4_nl_rows16_8", "llmx_iq4_nl_rows16_16", "llmx_iq4_nl_rows8_1", "llmx_iq4_nl_rows8_2", "llmx_iq4_nl_rows8_4", "llmx_iq4_nl_rows8_8", "llmx_iq4_nl_rows8_16", "llmx_iq4_nl_tile16", "llmx_iq4_nl_tile8", "llmx_iq4_nl_routed16", "llmx_iq4_nl_routed8", "llmx_iq4_nl_routed16_g2", "llmx_iq4_nl_routed16_g4", "llmx_iq4_nl_routed16_g8", "llmx_iq4_nl_routed8_g2", "llmx_iq4_nl_routed8_g4", "llmx_iq4_nl_routed8_g8", "llmx_iq4_nl_routed_tile16", "llmx_iq4_nl_routed_tile8", "llmx_iq4_nl_routed_narrow16", "llmx_iq4_nl_routed_narrow8",
                                                           "llmx_q5_0_quarter_rows16_1", "llmx_q5_0_quarter_rows16_2", "llmx_q5_0_quarter_rows16_4", "llmx_q5_0_quarter_rows16_8", "llmx_q5_0_quarter_rows16_16", "llmx_q5_0_quarter_rows8_1", "llmx_q5_0_quarter_rows8_2", "llmx_q5_0_quarter_rows8_4", "llmx_q5_0_quarter_rows8_8", "llmx_q5_0_quarter_rows8_16", "llmx_q5_0_quarter_routed16", "llmx_q5_0_quarter_routed8", "llmx_q5_0_quarter_routed16_g2", "llmx_q5_0_quarter_routed16_g4", "llmx_q5_0_quarter_routed16_g8", "llmx_q5_0_quarter_routed8_g2", "llmx_q5_0_quarter_routed8_g4", "llmx_q5_0_quarter_routed8_g8", "llmx_q5_1_quarter_rows16_1", "llmx_q5_1_quarter_rows16_2", "llmx_q5_1_quarter_rows16_4", "llmx_q5_1_quarter_rows16_8", "llmx_q5_1_quarter_rows16_16", "llmx_q5_1_quarter_rows8_1", "llmx_q5_1_quarter_rows8_2", "llmx_q5_1_quarter_rows8_4", "llmx_q5_1_quarter_rows8_8", "llmx_q5_1_quarter_rows8_16", "llmx_q5_1_quarter_routed16", "llmx_q5_1_quarter_routed8", "llmx_q5_1_quarter_routed16_g2", "llmx_q5_1_quarter_routed16_g4", "llmx_q5_1_quarter_routed16_g8", "llmx_q5_1_quarter_routed8_g2", "llmx_q5_1_quarter_routed8_g4", "llmx_q5_1_quarter_routed8_g8"};
    // A type's twin kernels by form and columns, with its routed builds' break-evens and tile extents (docs/src/backends-rocm.md, weight types).
    // kKernels marks a build the type lacks.
    struct TwinKernels {
        Kernel two[3][5];
        size_t two_rows[3][5];
        Kernel four[3][5];
        Kernel tile[2], tall[2];
        Kernel routed[3];
        Kernel grouped[3][3];
        Kernel routed_tile[2], routed_narrow[2];
        float group_from[2];
        size_t tile_from[2][2];
        size_t block_values;   // a weight block's values
        unsigned row_lanes;    // the row kernel's lanes a weight block (kernels/int16.hip, KRaw's Lane), 0 where a wave takes a row
        bool k45;              // Q4_K or Q5_K, whose rows take at most the profile's k45_row_lanes
        const TwinKernels* quarter;   // the quarter form's row, routed and grouped builds (kernels/int16.hip, QuarterLane), which row_form may take instead
    };
    static constexpr TwinKernels kQ8Twins = {{{kRows16x1, kRows16x2, kRows16x4, kRows16x8, kRows16x16},
                                          {kRows8x1, kRows8x2, kRows8x4, kRows8x8, kRows8x16},
                                          {kHalf8x1, kHalf8x2, kHalf8x4, kHalf8x8, kHalf8x16}},
                                         {{2, 2, 2, 2, 2}, {2, 2, 2, 2, 4}, {1, 2, 2, 2, 4}},
                                         {{kKernels, kRows16x2r4, kRows16x4r4, kRows16x8r4, kKernels},
                                          {kKernels, kRows8x2r4, kRows8x4r4, kRows8x8r4, kKernels},
                                          {kKernels, kHalf8x2r4, kHalf8x4r4, kHalf8x8r4, kKernels}},
                                         {kTile16, kTile8},
                                         {kTile16Tall, kTile8Tall},
                                         {kRouted16, kRouted8, kRoutedHalf8},
                                         {{kGrouped16x2, kGrouped16x4, kGrouped16x8}, {kGrouped8x2, kGrouped8x4, kGrouped8x8}, {kGroupedHalf8x2, kGroupedHalf8x4, kGroupedHalf8x8}},
                                         {kRoutedTile16, kRoutedTile8},
                                         {kRoutedNarrow16, kRoutedNarrow8},
                                         {1.25f, 0.0f},
                                         {{32, 32}, {32, 32}},
                                         32,
                                         0,
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
                {group_from16, group_from8},
                {{up_from16, up_from8}, {down_from16, down_from8}},
                block_values,
                row_lanes,
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
    static constexpr TwinKernels kQ50Base = k_twins(kRowsQ5016x1, 1.25f, 1.25f, 128, 32, 32, 32, 32, 1, false), kQ51Base = k_twins(kRowsQ5116x1, 5.0f, 1.25f, 128, 32, 32, 32, 32, 1, false), kIQ4NLTwins = k_twins(kRowsIQ4NL16x1, 0.3125f, 0.3125f, 32, 32, 32, 32, 32, 4, false);
    static constexpr TwinKernels kQ50Quarter = quarter_twins(kQ50Base, kQuarterQ5016x1, 1.25f, 1.25f), kQ51Quarter = quarter_twins(kQ51Base, kQuarterQ5116x1, 1.25f, 2.5f);
    static constexpr auto with_quarter = [](TwinKernels t, const TwinKernels* q) { t.quarter = q; return t; };
    static constexpr TwinKernels kQ50Twins = with_quarter(kQ50Base, &kQ50Quarter), kQ51Twins = with_quarter(kQ51Base, &kQ51Quarter);
    static_assert(kQ6KTwins.tile[1] == kTileQ6K8 && kIQ4NLTwins.routed[1] == kRoutedIQ4NL8 && kIQ4NLTwins.grouped[1][2] == kGroupedIQ4NL8x8 && kIQ4NLTwins.routed_tile[1] == kRoutedTileIQ4NL8 && kIQ4NLTwins.routed_narrow[1] == kRoutedNarrowIQ4NL8, "a K-type's kernels follow LLMX_KTYPE's order");
    static_assert(kQ51Quarter.two[1][4] == kQuarterQ518x16 && kQ51Quarter.routed[1] == kQuarterRoutedQ518 && kQ51Quarter.grouped[1][2] == kQuarterGroupedQ518x8 && kQ51Quarter.tile[1] == kTileQ518, "a quarter form's kernels follow LLMX_KQUARTER's order");
    // Each weight type's kernels, one row a type for every caller (docs/ROCM.md, Owners and boundaries): its exact product, the same over several rows, its embedding, its routed exact product and its twin kernels.
    // A type without twins takes its exact product in every class.
    struct TypeKernels {
        uint32_t type;
        const char* name;
        Kernel exact, rows, embed, routed;
        const TwinKernels* twin;
    };
    static constexpr TypeKernels kTypes[] = {
        {quant::GGML_TYPE_F32, "F32", kMatmul, kRowsExactF32, kGather, kRoutedExactF32, nullptr},          {quant::GGML_TYPE_Q8_0, "Q8_0", kMatmulQ8, kRowsExactQ8, kGatherQ8, kRoutedExactQ8, &kQ8Twins},
        {quant::GGML_TYPE_Q4_K, "Q4_K", kMatmulQ4K, kRowsExactQ4K, kGatherQ4K, kRoutedExactQ4K, &kQ4KTwins}, {quant::GGML_TYPE_Q5_K, "Q5_K", kMatmulQ5K, kRowsExactQ5K, kGatherQ5K, kRoutedExactQ5K, &kQ5KTwins},
        {quant::GGML_TYPE_Q6_K, "Q6_K", kMatmulQ6K, kRowsExactQ6K, kGatherQ6K, kRoutedExactQ6K, &kQ6KTwins}, {quant::GGML_TYPE_Q5_0, "Q5_0", kMatmulQ50, kRowsExactQ50, kGatherQ50, kRoutedExactQ50, &kQ50Twins},
        {quant::GGML_TYPE_Q5_1, "Q5_1", kMatmulQ51, kRowsExactQ51, kGatherQ51, kRoutedExactQ51, &kQ51Twins},    {quant::GGML_TYPE_BF16, "BF16", kMatmulBF16, kRowsExactBF16, kGatherBF16, kRoutedExactBF16, nullptr},
        {quant::GGML_TYPE_IQ4_NL, "IQ4_NL", kMatmulIQ4NL, kRowsExactIQ4NL, kGatherIQ4NL, kRoutedExactIQ4NL, &kIQ4NLTwins},
    };
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
                // The slots read since the last submission are tagged by a ticket covering every launch that read them, and the next slot waits for its own.
                if (std::find(dirty.begin(), dirty.end(), true) != dirty.end()) b.submit();
                slot = (slot + 1) % kRingSlots;
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
        void* args[] = {&aux, &w, &cs, &sn, &blocks, &gpos, &of, &first, &base, &dim, &half, &compression, &bt, &block_floats, &rows_floats, &eps, &streams};
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
        launch_check(hip().hipMemcpyAsync(d, arg(src, bytes), bytes, hipMemcpyHostToDevice, stream_), "hipMemcpyAsync");
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
        if (nbatch > 1) {
            // Several rows take the blocked build, each output the bits of its one-row product (kernels/f32.hip, exact_rows).
            unsigned long long nr = nbatch;
            void* args[] = {&w, &x, &y, &ni, &no, &nr, &add};
            launch_writing({{y, size_mul(nout, nbatch)}}, kinds.rows, (unsigned)((nout + kRowsOut - 1) / kRowsOut), (unsigned)((nbatch + kRowsIn - 1) / kRowsIn), args);
        } else {
            // One row takes a wave an output (kernels/f32.hip, exact_wave).
            void* args[] = {&w, &x, &y, &ni, &no, &add};
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
        return cl;
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
            RoutedCall c{};
            for (size_t p = 0; p < nproj; ++p) {
                c.c.w[p] = bytes(proj[p].weight.data);
                c.c.y[p] = f32(proj[p].out);
                c.c.nout[p] = proj[p].rows;
                c.c.start[p] = c.c.total;
                c.c.total += proj[p].rows;
                c.stride[p] = quant::row_bytes(type, nin, proj[p].rows);
            }
            c.c.nproj = (unsigned)nproj;
            c.c.cluster = row_cluster(rk, nin);
            const size_t clusters = dev_.caps.subgroup_size / c.c.cluster;
            c.ids = ids;
            c.per = per;
            c.n_expert = routing.n_expert;
            if (tile && routing.n_expert <= kGroupExperts) {
                // A prompt's entries in groups of up to 64 of one expert, each group a block row of the routed tile over its expert's rows (kernels/int16.hip, TileGroups); the tile never splits, so each entry's sums are its own.
                const unsigned chunk = kTileRows, groups = (unsigned)((entries + chunk - 1) / chunk + std::min(routing.n_expert, entries));
                const size_t tab_bytes = (4 * size_t(groups) + entries) * sizeof(unsigned);
                if (!moe_tab_ || moe_tab_->size() < tab_bytes) moe_tab_ = alloc(tab_bytes, Memory::device);
                unsigned* tab = (unsigned*)as_rocm(*moe_tab_).data();
                unsigned ne = (unsigned)entries, nx = (unsigned)routing.n_expert, ng = groups, ch = chunk;
                void* gargs[] = {(void*)&ids, &tab, &ne, &nx, &ng, &ch};
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
                // Each group runs in one of the two builds, the narrow one where its entries fit its 16 rows; the other build's workgroups return at once.
                for (const Kernel kernel : {kinds.twin->routed_tile[int8], kinds.twin->routed_narrow[int8]})
                    for (size_t first = 0; first < groups; first += kGridY) {
                        tg.first = first;
                        void* args[] = {&t, &tg, &q, &d};
                        launch(kernel, (unsigned)gx, (unsigned)std::min<size_t>(kGridY, groups - first), args);
                    }
            } else if (entries && double(entries) >= double(rk.group_from[int8]) * double(routing.n_expert) && routing.n_expert <= kGroupExperts) {
                // One expert's entries a group of the narrowest grouped build that holds their average, up to 8, each column the bits of its one-column product.
                const size_t touched = std::min(routing.n_expert, entries), average = (entries + touched - 1) / touched;
                const int width = average <= 2 ? 0 : average <= 4 ? 1 : 2;
                unsigned chunk = 2u << width, groups = (unsigned)((entries + chunk - 1) / chunk + touched);
                const size_t tab_bytes = (4 * size_t(groups) + entries) * sizeof(unsigned);
                if (!moe_tab_ || moe_tab_->size() < tab_bytes) moe_tab_ = alloc(tab_bytes, Memory::device);
                unsigned* tab = (unsigned*)as_rocm(*moe_tab_).data();
                unsigned ne = (unsigned)entries, nx = (unsigned)routing.n_expert;
                void* gargs[] = {(void*)&ids, &tab, &ne, &nx, &groups, &chunk};
                launch_writing({{tab, tab_bytes / sizeof(float)}}, kMoeGroup, nx, 1, gargs);
                unsigned long long order0 = 4ull * groups;
                const size_t wave_rows = 2 * rows_per_block_ * clusters;
                for (size_t first = 0; first < groups; first += kGridY) {
                    c.first = first;
                    void* args[] = {&c, &tab, &order0, &q, &d, &ni};
                    launch(rk.grouped[form][width], (unsigned)((c.c.total + wave_rows - 1) / wave_rows), (unsigned)std::min<size_t>(kGridY, groups - first), args);
                }
            } else {
                const size_t wave_rows = rk.two_rows[form][0] * rows_per_block_ * clusters;
                for (size_t first = 0; first < entries; first += kGridY) {
                    c.first = first;
                    void* args[] = {&c, &q, &d, &ni};
                    launch(rk.routed[form], (unsigned)((c.c.total + wave_rows - 1) / wave_rows), (unsigned)std::min(kGridY, entries - first), args);
                }
            }
            record_matrix_path(int8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
            for (size_t p = 0; p < nproj; ++p)
                if (kept_.x && c.c.y[p] < kept_.x + kept_.values && kept_.x < c.c.y[p] + proj[p].rows * entries) kept_ = {};
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
                void* args[] = {&w, &x, &y, &ni, &no, &r};
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
    // The twin of X's `rows` rows laid out by `map` (block-int8 for the int8 class), built unless the kept twin holds it or `build` is false.
    std::pair<const uint8_t*, const float*> twin_of(CSlice X, size_t rows, size_t nin, bool int8, TwinMap map, bool build) {
        const size_t blocks = size_mul(rows, nin / 32), qbytes = size_mul(map.total, 32 * (int8 ? 1 : 2)), need = size_add(qbytes, size_mul(map.total, sizeof(float)));
        const float* x = f32(X);
        if (build) want_twin_ = int8 ? 2 : 1;
        if (build && !(kept_.x == x && kept_.values == blocks * 32 && kept_.int8 == int8 && kept_.map == map)) {
            if (!twin_ || twin_->size() < need) twin_ = alloc(need, Memory::device);
            uint8_t* q = (uint8_t*)as_rocm(*twin_).data();
            float* d = (float*)(q + qbytes);
            unsigned long long nb = blocks;
            void* args[] = {&x, &q, &d, &nb, &map};
            launch(twin_build(int8 ? kTwin8 : kTwin16, map), (unsigned)((blocks + kThreads / 32 - 1) / (kThreads / 32)), 1, args);
            // Kept for the next product of the same activations, unless the host may write them in place.
            if (!as_rocm(*X.buffer).host_ptr()) kept_ = {x, blocks * 32, int8, map};
        }
        const uint8_t* q0 = (const uint8_t*)as_rocm(*twin_).data();
        return {q0, (const float*)(q0 + qbytes)};
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
                    RowCall c{};
                    for (size_t pi = 0; pi < nproj; ++pi) {
                        c.w[pi] = proj[pi].w;
                        c.y[pi] = proj[pi].y + at * proj[pi].nout;
                        c.nout[pi] = proj[pi].nout;
                        c.start[pi] = c.total;
                        c.total += proj[pi].nout;
                    }
                    c.nproj = (unsigned)nproj;
                    const TwinKernels& rk = row_form(tk, nin);
                    c.cluster = row_cluster(rk, nin);
                    int ncols = (int)rows;
                    void* args[] = {&c, &q, &d, &ni, &ncols, &add};
                    const int build = rows <= 1 ? 0 : rows <= 2 ? 1 : rows <= 4 ? 2 : rows <= 8 ? 3 : 4;
                    // A one-column half-block row of at most 16 blocks takes a 64-lane wave's halves as two rows, and from q8_decode_two_rows_from output rows the others take two rows a wave.
                    // Both keep the one-row build's bits and read the row width, the output count and the type, never the call's columns (kernels/int16.hip).
                    const bool half1 = rk.two[form][0] == kHalf8x1 && build == 0;
                    const bool paired = half1 && nin / 32 <= 16 && dev_.caps.subgroup_size == 64;
                    const RowBuild rb = paired ? RowBuild{kHalf8x1p, 2} : half1 && c.total >= dev_.profile.q8_decode_two_rows_from ? RowBuild{kHalf8x1r2, 2} : row_build(rk, form, build, c.total);
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
        for (size_t pi = 0; pi < nproj; ++pi)
            if (kept_.x && proj[pi].y < kept_.x + kept_.values && kept_.x < proj[pi].y + proj[pi].nout * nbatch) kept_ = {};
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
        void* args[] = {&c, &q, &d};
        launch(tall ? tk.tall[int8] : tk.tile[int8], (unsigned)gx, (unsigned)((rows + kTileRows - 1) / kTileRows), args, c.parts);
        if (c.parts > 1) {
            size_t g = 0;
            for (size_t pi = 0; pi < nproj; ++pi) {
                c.start[pi] = (unsigned)g;
                g += (proj[pi].nout * rows + kThreads - 1) / kThreads;
            }
            void* rargs[] = {&c};
            launch(kTileReduce, (unsigned)g, 1, rargs);
        }
    }
    // Rows `ids` of a table through the type's gather kernel, ids on the device; an id of `nrows` or more gives a zero row.
    void gather(Kernel k, Slice dst, const void* table, size_t width, const unsigned* ids, size_t count, size_t nrows) {
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(count, width), sizeof(float)));
        float* d = f32(dst);
        unsigned long long w = width;
        unsigned nr = (unsigned)std::min<size_t>(nrows, UINT32_MAX);
        void* args[] = {&d, &table, &ids, &w, &nr};
        launch(k, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)count, args);
    }
    // Where a producer writes the twin of its output of `values` floats: once a product has read a twin, in that class, for an output of whole blocks in device memory.
    struct TwinOut {
        void* q = nullptr;
        float* d = nullptr;
        bool int8 = false;
        TwinMap map{};
    };
    // Its rows lie as `map` lays them out (TwinMap), the layout the product's own runs give.
    TwinOut twin_out(Slice dst, size_t values, const TwinMap& map) {
        if (!want_twin_ || values % 32 || as_rocm(*dst.buffer).host_ptr()) return {};
        const bool int8 = want_twin_ == 2;
        const size_t qbytes = size_mul(map.total, int8 ? 32 : 64), need = size_add(qbytes, size_mul(map.total, sizeof(float)));
        if (!twin_ || twin_->size() < need) twin_ = alloc(need, Memory::device);
        uint8_t* q = (uint8_t*)as_rocm(*twin_).data();
        return {q, (float*)(q + qbytes), int8, map};
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
        if (t.q) kept_ = {x, values, t.int8, t.map};
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
    // Forgets the kept twin where `n` floats written at y meet its input.
    void forget_written(const float* y, size_t n) {
        if (kept_.x && y < kept_.x + kept_.values && kept_.x < y + n) kept_ = {};
    }
    // A launch that writes only `writes`, an exact product's or a router's outputs: it keeps the twin a later product may read unless one of them meets its input.
    void launch_writing(std::initializer_list<std::pair<const void*, size_t>> writes, Kernel k, unsigned gx, unsigned gy, void** args, unsigned gz = 1, unsigned threads = kThreads) {
        const auto kept = kept_;
        launch(k, gx, gy, args, gz, threads);
        kept_ = kept;
        for (const auto& [y, n] : writes) forget_written((const float*)y, n);
    }
    void launch(Kernel k, unsigned gx, unsigned gy, void** args, unsigned gz = 1, unsigned threads = kThreads) {
        raise_failed();
        // Any kernel but a product may write the kept twin's input.
        if (k < kRows16x1) kept_ = {};
        // HIP finds a module's function through the calling thread's device, which another backend's call may have changed.
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        launch_check(hip().hipModuleLaunchKernel(kernels_[k], gx, gy, gz, threads, 1, 1, 0, stream_, args, nullptr), kKernelNames[k]);
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
    // Frees the graves whose covering ticket has retired.
    void bury() noexcept {
        std::lock_guard<std::mutex> lock(graves_->m);
        auto& held = graves_->held;
        const Ticket last = tickets_.last();
        const auto kept = std::remove_if(held.begin(), held.end(), [&](const auto& g) {
            if (g.first > last || !tickets_.settled(g.first)) return false;
            free_memory(g.second.first, g.second.second);
            return true;
        });
        held.erase(kept, held.end());
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
    void group_launch(unsigned blocks, void** args) {
        kept_ = {};
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        launch_check(hip().hipModuleLaunchKernel(kernels_[kGroupSum], blocks, 1, 1, kThreads, 1, 1, 0, stream_, args, nullptr), "llmx_group_sum");
    }
    friend class RocmCollective;

    RocmDevice dev_;
    std::shared_ptr<Graves> graves_ = std::make_shared<Graves>();   // before every buffer member, so it outlives them
    hipStream_t stream_ = nullptr;
    hipModule_t module_ = nullptr, int16_module_ = nullptr, dot_module_ = nullptr;
    BufferPtr parts_;  // attention's chunk partials, grown as calls need
    BufferPtr twin_;   // the twin of the last product's activations, or of a producer's output, grown as calls need
    // The twin a producer wrote into twin_ beside its output (the Vulkan producers' design), which a product of that output reads instead of making its own; any other launch, copy, write or allocation forgets it.
    struct {
        const float* x = nullptr;
        size_t values = 0;
        bool int8 = false;
        TwinMap map{};
    } kept_;
    BufferPtr moe_out_;   // a routed down projection's slots before they join the residual
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
    Ring args_, stage_, pass_;   // the argument ring, the staging ring, and pass_arg's ring in device memory
    std::array<Upload, kUploads> uploads_{};
    size_t next_upload_ = 0;
    TicketLedger tickets_;
    bool opened_ = false;
    size_t launched_[kKernels] = {};   // launches of each kernel since rocm_take_launches last read them
    std::mutex failed_mu_;
    std::string failed_;
    unsigned* fault_ = nullptr;   // host memory a group sum's kernel sets where it gave up waiting (kernels/f32.hip, llmx_group_sum)
    std::map<std::vector<int>, std::string> group_paths_;   // per group of device indices led by this backend: empty for fast sums, else why sums are conservative
};

RocmKVStorage::RocmKVStorage(RocmBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens, KVType kt, KVType vt,
                             const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools)
    : BlockKVStorage(owner, "rocm", kBlockTokens, layers, heads, dim, max_tokens, kt, vt, auxiliary_widths, auxiliary_pools) {}

// Dropping the old buffer is enough: its memory waits in the graves for the copies recorded before.
void RocmKVStorage::retire(const BufferPtr&) {}

} // namespace
} // namespace backend

#include "backends/rocm/rocm_collective.hpp"

namespace backend {

BackendPtr make_rocm_backend(int device) { return std::make_shared<RocmBackend>(device); }

std::map<std::string, size_t> rocm_take_launches(Backend& backend) { return dynamic_cast<RocmBackend&>(backend).take_launches(); }

DeviceProfile& rocm_profile(Backend& backend) { return dynamic_cast<RocmBackend&>(backend).profile(); }

} // namespace backend
