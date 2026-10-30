// The ROCm backend (docs/ROCM.md): one HIP stream a device, every op an asynchronous launch on it, a ticket an event recorded on the stream.
// The host never waits inside a pass: it waits only where a caller asks for a ticket, a read, or a ring of host memory it reuses (docs/src/backends-rocm.md).
#include "backends/rocm/rocm_backend.hpp"

#include "backends/kv_storage.hpp"
#include "backends/rocm/rocm_device.hpp"
#include "quant/types.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <exception>
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

class RocmBackend;

// A free that fails has nothing left to act on.
void free_memory(void* p, bool host) {
    const HipFn& f = hip();
    (void)(host ? f.hipHostFree(p) : f.hipFree(p));
}

// Memory of dropped buffers, freed once the work recorded before the drop retires, since hipFree does not wait for a launch or copy still queued that names it.
struct Graves {
    std::mutex m;
    std::vector<std::pair<Ticket, std::pair<void*, bool>>> held;   // the ticket that covers the drop, and the memory
    const TicketLedger* tickets = nullptr;                          // null once the backend has drained its stream, when a drop frees at once
};

// Device memory from hipMalloc, or pinned host memory the device reads in place (Memory::host_visible); dropped, its memory goes to the backend's graves.
class RocmBuffer final : public Buffer {
public:
    RocmBuffer(size_t bytes, bool host, std::shared_ptr<Graves> graves) : size_(bytes), host_(host), graves_(std::move(graves)) {
        if (!bytes) return;
        const HipFn& f = hip();
        if (host) hip_check(f.hipHostMalloc(&p_, bytes, hipHostMallocDefault), "hipHostMalloc");
        else hip_check(f.hipMalloc(&p_, bytes), "hipMalloc");
    }
    ~RocmBuffer() override {
        if (!p_) return;
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
    bool host_;
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
        for (auto& e : events_) hip_check(f.hipEventCreateWithFlags(&e, hipEventDisableTiming), "hipEventCreateWithFlags");
        hip_check(f.hipModuleLoadData(&module_, kF32Module), "hipModuleLoadData");
        hip_check(f.hipModuleLoadData(&int16_module_, kInt16Module), "hipModuleLoadData");
        // The dot module loads only where the build compiled it for this device: that is the capability flag, read from the code object rather than from the device's name.
        dev_.caps.integer_dot = f.hipModuleLoadData(&dot_module_, kInt16DotModule) == hipSuccess;
        if (!dev_.caps.integer_dot) dot_module_ = nullptr;
        // The prompt tile takes the dot where there is one; the row kernels keep the plain module, measured faster there (docs/STATUS.md, ROCm step 3).
        for (size_t k = 0; k < kKernels; ++k) {
            const hipModule_t m = k < kTwin16 ? module_ : k == kTile16 && dot_module_ ? dot_module_ : int16_module_;
            hip_check(f.hipModuleGetFunction(&kernels_[k], m, kKernelNames[k]), "hipModuleGetFunction");
        }
        for (size_t s = 0; s < kRingSlots; ++s) {
            args_.slots[s] = std::make_shared<RocmBuffer>(kArgBytes, true, graves_);
            stage_.slots[s] = std::make_shared<RocmBuffer>(kStageBytes, true, graves_);
        }
    }
    ~RocmBackend() override {
        const HipFn& f = hip();
        (void)f.hipSetDevice(dev_.index);
        (void)f.hipStreamSynchronize(stream_);
        {
            std::lock_guard<std::mutex> lock(graves_->m);
            graves_->tickets = nullptr;
            for (const auto& g : graves_->held) free_memory(g.second.first, g.second.second);
            graves_->held.clear();
        }
        parts_ = {};
        twin_ = {};
        staged_ = {};
        args_ = {};
        stage_ = {};
        if (module_) (void)f.hipModuleUnload(module_);
        if (int16_module_) (void)f.hipModuleUnload(int16_module_);
        if (dot_module_) (void)f.hipModuleUnload(dot_module_);
        for (auto e : events_) if (e) (void)f.hipEventDestroy(e);
        if (stream_) (void)f.hipStreamDestroy(stream_);
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
    size_t scratch_reserve(size_t free) const override { return ((size_t)256 << 20) + free / 20; }
    bool supports_type(uint32_t type) const override { return type == quant::GGML_TYPE_F32 || type == quant::GGML_TYPE_Q8_0; }
    bool implements(Op op) const override { return op == Op::norm_rope_partial; }
    // No kernel here chooses anything by a row's extent, so a generated token is a class and every prompt row another.
    size_t row_class(size_t extent) const override { return extent == 1 ? 1 : 2; }

    // HIP allocates on the calling thread's current device, and a backend may be called from several threads.
    BufferPtr alloc(size_t bytes, Memory where) override {
        hip_check(hip().hipSetDevice(dev_.index), "hipSetDevice");
        auto b = std::make_shared<RocmBuffer>(bytes, where == Memory::host_visible, graves_);
        if (!bytes) return b;
        // KV growth and the model read a fresh buffer as zeros.
        if (where == Memory::host_visible) std::memset(b->data(), 0, bytes);
        else launch_check(hip().hipMemsetAsync(b->data(), 0, bytes, stream_), "hipMemsetAsync");
        return b;
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
        // The event is recorded before the ticket is handed out, so a waiter on another thread never reads a slot not yet recorded for it.
        const Ticket t = tickets_.last() + 1;
        const size_t i = t % kEvents;
        if (t > kEvents) wait(t - kEvents);
        bury();
        hip_check(hip().hipEventRecord(events_[i], stream_), "hipEventRecord");
        tickets_.take();
        opened_ = false;
        for (Ring* r : {&args_, &stage_}) r->tag(t);
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
            try { submit(); } catch (const std::exception&) {}
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
        bool twinned = false;
        for (const Projection& p : projections) {
            if (!p.weight.data.buffer) throw std::runtime_error("rocm: projection without storage");
            product(p.weight, X, p.out, nin, p.rows, nbatch, false, runs, dtype, !twinned);
            twinned = twinned || int16(p.weight, dtype);
        }
    }
    std::vector<Dtype> native_dtypes() const override { return {Dtype::f16, Dtype::f32}; }
    std::string dtype_path(Dtype dtype) const override {
        if (dtype == Dtype::f32) return "f32";
        return dev_.caps.integer_dot ? "block-int16 (Q8_0 rows, and tiles on the 16-bit dot), f32 (F32 weights)" : "block-int16 (Q8_0 rows and tiles), f32 (F32 weights)";
    }
    void embed(Slice dst, WeightView weight, size_t nin, size_t nrows, const uint32_t* ids, size_t count) override {
        check_weight(weight, nin, "embedding");
        if (!count) return;
        for (size_t i = 0; i < count; ++i)
            if (ids[i] >= nrows) throw std::runtime_error("rocm: embedding row out of range");
        span(*weight.data.buffer, weight.data.offset * sizeof(float), quant::row_bytes(weight.type, nin, nrows));
        if (weight.type == quant::GGML_TYPE_F32) return gather(dst, weight.data, nin, ids, count);
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(count, nin), sizeof(float)));
        float* d = f32(dst);
        const uint8_t* t = bytes(weight.data);
        const unsigned* ip = (const unsigned*)arg(ids, count * sizeof(uint32_t));
        unsigned long long w = nin;
        void* args[] = {&d, &t, &ip, &w};
        launch(kGatherQ8, (unsigned)((nin + kThreads - 1) / kThreads), (unsigned)count, args);
    }
    void gather_rows(Slice dst, CSlice src, size_t width, const uint32_t* rows, size_t count) override {
        if (count && !rows) throw std::runtime_error("rocm: gather without rows");
        if (!count) return;
        span(*src.buffer, src.offset * sizeof(float), size_mul(size_mul(size_add(*std::max_element(rows, rows + count), 1), width), sizeof(float)));
        gather(dst, src, width, rows, count);
    }
    void rms_norm_rows(Slice dst, CSlice src, CSlice w, size_t rows, size_t n, size_t stride, float eps, RowRuns) override {
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
        void* args[] = {&d, &s, &wp, &in, &st, &eps};
        launch(kRmsNorm, (unsigned)rows, 1, args);
    }
    void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride, size_t heads, size_t head_dim,
                           size_t rope_dim, CSlice w, float eps, CSlice cos, CSlice sin, const uint32_t* pos, RopeStreams streams) override {
        if (!heads || !head_dim || !rope_dim || rope_dim % 2 || rope_dim > head_dim)
            throw std::runtime_error("rocm: invalid partial rope dimensions");
        if (streams.count != 1) throw std::runtime_error("rocm: a rope of several streams is not implemented");
        if (rows && !pos) throw std::runtime_error("rocm: partial rope without positions");
        if (!rows) return;
        const size_t width = size_mul(heads, head_dim), half = rope_dim / 2;
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(rows, width), sizeof(float)));
        const size_t last = size_add(size_add(size_mul(rows - 1, src_stride), size_mul(heads - 1, src_head_stride)), head_dim);
        span(*src.buffer, src.offset * sizeof(float), size_mul(last, sizeof(float)));
        if (w.buffer) span(*w.buffer, w.offset * sizeof(float), size_mul(head_dim, sizeof(float)));
        const size_t table = size_mul(size_add(*std::max_element(pos, pos + rows), 1), half * sizeof(float));
        span(*cos.buffer, cos.offset * sizeof(float), table);
        span(*sin.buffer, sin.offset * sizeof(float), table);
        float* d = f32(dst);
        const float* s = f32(src);
        if ((const float*)d == s && (src_stride != width || src_head_stride != head_dim))
            throw std::runtime_error("rocm: partial rope in place over heads that are not contiguous");
        const unsigned* p = (const unsigned*)arg(pos, rows * sizeof(uint32_t));
        unsigned long long dst_stride = width, ss = src_stride, shs = src_head_stride;
        int hd = (int)head_dim, hf = (int)half;
        const float* wp = w.buffer ? f32(w) : nullptr;
        const float* c = f32(cos);
        const float* sn = f32(sin);
        void* args[] = {&d, &s, &dst_stride, &ss, &shs, &hd, &hf, &wp, &eps, &c, &sn, &p};
        launch(kNormRope, (unsigned)rows, (unsigned)heads, args);
    }
    void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, RowRuns) override {
        if (!n) return;
        for (const CSlice& s : {CSlice(dst), gate, up}) span(*s.buffer, s.offset * sizeof(float), size_mul(n, sizeof(float)));
        float* d = f32(dst);
        const float* g = f32(gate);
        const float* u = f32(up);
        unsigned long long nn = n;
        void* args[] = {&d, &g, &u, &nn};
        launch(kSiluMul, (unsigned)((n + kThreads - 1) / kThreads), 1, args);
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
        for (size_t w : auxiliary_widths)
            if (w) throw std::runtime_error("rocm: auxiliary KV rows are not implemented");
        return std::make_unique<RocmKVStorage>(*this, layers, n_head_kv, head_dim, max_tokens, k_type, v_type, auxiliary_widths, auxiliary_pools);
    }
    void kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k, CSlice v, CSlice auxiliary) override {
        if (n_views && !views) throw std::runtime_error("rocm: KV write without views");
        if (auxiliary.buffer) throw std::runtime_error("rocm: auxiliary KV rows are not implemented");
        if (!n_views) return;
        RocmKVStorage& s = storage(views[0]);
        size_t rows = 0;
        for (size_t vi = 0; vi < n_views; ++vi) {
            if (&storage(views[vi]) != &s) throw std::runtime_error("rocm: KV views of different storages in one write");
            s.check_view(views[vi], layer, true);
            rows = size_add(rows, views[vi].nq);
        }
        if (!rows) return;
        const size_t floats = size_mul(size_mul(rows, s.heads()), s.dim());
        span(*k.buffer, k.offset * sizeof(float), size_mul(floats, sizeof(float)));
        span(*v.buffer, v.offset * sizeof(float), size_mul(floats, sizeof(float)));
        std::vector<int32_t> blocks(rows), slots(rows);
        size_t r = 0;
        for (size_t vi = 0; vi < n_views; ++vi)
            for (size_t b = 0; b < views[vi].nq; ++b, ++r) {
                const size_t t = views[vi].length + b;
                blocks[r] = views[vi].blocks[t / kBlockTokens];
                slots[r] = (int32_t)(t % kBlockTokens);
            }
        void* kc = as_rocm(*s.k_buffer(layer)).data();
        void* vc = as_rocm(*s.v_buffer(layer)).data();
        const float* kp = f32(k);
        const float* vp = f32(v);
        const int* bp = (const int*)arg(blocks.data(), rows * sizeof(int32_t));
        const int* sp = (const int*)arg(slots.data(), rows * sizeof(int32_t));
        int heads = (int)s.heads(), dim = (int)s.dim(), bt = (int)kBlockTokens;
        int kh = s.k_type() == KVType::f16, vh = s.v_type() == KVType::f16;
        void* args[] = {&kc, &vc, &kp, &vp, &bp, &sp, &heads, &dim, &bt, &kh, &vh};
        launch(kKvWrite, (unsigned)rows, (unsigned)heads, args);
    }
    void attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out, int n_head, int n_head_kv, int head_dim) override {
        if (n_views && !views) throw std::runtime_error("rocm: attention without views");
        if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || head_dim > (int)kThreads)
            throw std::runtime_error("rocm: invalid attention dimensions");
        if (!n_views) return;
        RocmKVStorage& s = storage(views[0]);
        if ((size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads()) throw std::runtime_error("rocm: attention outside the KV view");
        std::vector<int32_t> table, firsts, ends;
        for (size_t vi = 0; vi < n_views; ++vi) {
            const KVView& view = views[vi];
            if (&storage(view) != &s) throw std::runtime_error("rocm: attention over views of different storages");
            const size_t used = s.check_view(view, layer, false);
            const int32_t first = (int32_t)table.size();
            table.insert(table.end(), view.blocks, view.blocks + used);
            for (size_t b = 0; b < view.nq; ++b) {
                firsts.push_back(first);
                ends.push_back((int32_t)(view.causal ? view.length + b + 1 : view.length + view.nq));
            }
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
        const int* tp = (const int*)arg(table.data(), table.size() * sizeof(int32_t));
        const int* fp = (const int*)arg(firsts.data(), rows * sizeof(int32_t));
        const int* ep = (const int*)arg(ends.data(), rows * sizeof(int32_t));
        int ratio = n_head / n_head_kv, bt = (int)kBlockTokens;
        int kh = s.k_type() == KVType::f16, vh = s.v_type() == KVType::f16;
        float scale = 1.0f / std::sqrt((float)head_dim);
        // Each row's history in chunks of its own, merged in order (kernels/f32.hip), so its bits do not depend on the call.
        const size_t need = size_mul(size_mul(size_mul(rows, (size_t)n_head), kChunks), size_mul((size_t)head_dim + 2, sizeof(float)));
        if (!parts_ || parts_->size() < need) {
            parts_ = alloc(need, Memory::device);
        }
        float* pp = (float*)as_rocm(*parts_).data();
        // A call with a prompt's rows on F16 sides of heads up to 128 wide takes the tile kernel, a view's rows of one chunk split sharing each K and V tile; it gives the chunk kernel's bits.
        // Generated rows alone keep the chunk kernel, whose blocks of one row and head each fill the device better.
        const int per_tile = std::max(1, std::min(16, 32 / ratio));
        const bool prompt = std::any_of(views, views + n_views, [](const KVView& v) { return v.nq > 1; });
        if (prompt && kh && vh && head_dim <= 128 && ratio <= 32) {
            std::vector<int32_t> tiles;
            int last_count = -1, last_length = -1;
            for (size_t r = 0; r < rows; ++r) {
                const int count = std::min(16, (ends[r] + 255) / 256), length = ((ends[r] + count - 1) / count + 63) / 64 * 64;
                const bool joins = !tiles.empty() && firsts[r] == firsts[(size_t)tiles[tiles.size() - 2]] && count == last_count &&
                                   length == last_length && tiles.back() < per_tile;
                if (joins) ++tiles.back();
                else tiles.insert(tiles.end(), {(int32_t)r, 1});
                last_count = count;
                last_length = length;
            }
            const int* tl = (const int*)arg(tiles.data(), tiles.size() * sizeof(int32_t));
            void* args[] = {&q, &pp, &kc, &vc, &tp, &fp, &ep, &tl, &n_head, &ratio, &n_head_kv, &head_dim, &bt, &scale};
            launch(kAttentionTile, (unsigned)(tiles.size() / 2), (unsigned)n_head_kv, args, kChunks);
        } else {
            void* args[] = {&q, &pp, &kc, &vc, &tp, &fp, &ep, &n_head, &ratio, &n_head_kv, &head_dim, &bt, &kh, &vh, &scale};
            launch(kAttentionChunk, (unsigned)rows, (unsigned)n_head, args, kChunks);
        }
        void* merge[] = {&pp, &o, &ep, &n_head, &head_dim};
        launch(kAttentionMerge, (unsigned)rows, (unsigned)n_head, merge);
    }


private:
    // The F32 module's kernels, then from kTwin16 the int16 module's.
    enum Kernel { kAdd, kSiluMul, kRmsNorm, kNormRope, kGather, kGatherQ8, kMatmul, kMatmulQ8, kKvWrite, kAttentionChunk, kAttentionTile, kAttentionMerge,
                  kTwin16, kRows16x1, kRows16x2, kRows16x4, kRows16x8, kRows16x16, kTile16, kKernels };
    static constexpr const char* kKernelNames[kKernels] = {"llmx_add", "llmx_silu_mul", "llmx_rms_norm_rows", "llmx_norm_rope", "llmx_gather",
                                                           "llmx_gather_q8_0", "llmx_matmul_f32", "llmx_matmul_q8_0", "llmx_kv_write",
                                                           "llmx_attention_chunk", "llmx_attention_tile", "llmx_attention_merge", "llmx_twin16", "llmx_q8_0_rows16_1",
                                                           "llmx_q8_0_rows16_2", "llmx_q8_0_rows16_4", "llmx_q8_0_rows16_8", "llmx_q8_0_rows16_16",
                                                           "llmx_q8_0_tile16"};
    static constexpr unsigned kChunks = 16;   // a row's attention chunks at most (kernels/f32.hip)
    static constexpr size_t kColumns = 16;   // the row kernel's columns (kernels/int16.hip)

    // Pinned host memory in slots the device reads in place or copies from: a slot is taken again once the ticket that last read it retires.
    struct Ring {
        std::array<BufferPtr, kRingSlots> slots{};
        std::array<Ticket, kRingSlots> tags{};
        size_t slot = 0, used = 0;
        bool dirty = false;
        void* take(RocmBackend& b, size_t bytes) {
            const size_t cap = slots[0]->size();
            if (bytes > cap) throw std::logic_error("rocm: a ring take wider than a slot");
            bytes = (bytes + 255) & ~size_t(255);
            if (used + bytes > cap) {
                // The slot moved off is tagged by a ticket covering every launch that read it, and the next slot waits for its own.
                if (dirty) b.submit();
                slot = (slot + 1) % kRingSlots;
                b.wait(tags[slot]);
                used = 0;
            }
            void* p = (uint8_t*)const_cast<void*>(slots[slot]->host_ptr()) + used;
            used += bytes;
            dirty = true;
            return p;
        }
        void tag(Ticket t) {
            if (dirty) tags[slot] = t;
            dirty = false;
        }
    };

    // Host values a kernel reads, copied into the argument ring.
    const void* arg(const void* src, size_t bytes) {
        void* p = args_.take(*this, std::max<size_t>(bytes, 1));
        std::memcpy(p, src, bytes);
        return p;
    }
    static float* f32(Slice s) { return (float*)as_rocm(*s.buffer).data() + s.offset; }
    static const float* f32(CSlice s) { return (const float*)as_rocm(*s.buffer).data() + s.offset; }
    static const uint8_t* bytes(CSlice s) { return (const uint8_t*)as_rocm(*s.buffer).data() + s.offset * sizeof(float); }
    // The weights the products and the embedding read: F32, and Q8_0 decoded exactly into F32 sums, in whole blocks and without scales.
    static void check_weight(const WeightView& w, size_t nin, const char* what) {
        if (w.type != quant::GGML_TYPE_F32 && w.type != quant::GGML_TYPE_Q8_0)
            throw std::runtime_error(std::string("rocm: an ") + what + " of type " + std::to_string(w.type) + " is not implemented");
        if (w.scales.buffer) throw std::runtime_error(std::string("rocm: a scaled ") + what + " is not implemented");
        if (!w.data.buffer) throw std::runtime_error(std::string("rocm: an ") + what + " without storage");
        if (w.type == quant::GGML_TYPE_Q8_0 && nin % 32) throw std::runtime_error(std::string("rocm: a Q8_0 ") + what + " row ends inside a block");
    }
    RocmKVStorage& storage(const KVView& v) {
        auto* s = dynamic_cast<RocmKVStorage*>(v.storage);
        if (!s) throw std::runtime_error("rocm: a KV view of another backend");
        return *s;
    }
    // A Q8_0 product outside the F32 class reads the block-int16 twin; F32 weights and the F32 class read the activations as they are.
    static bool int16(const WeightView& weight, Dtype dtype) { return weight.type == quant::GGML_TYPE_Q8_0 && dtype != Dtype::f32; }
    void product(const WeightView& weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, bool accumulate, RowRuns runs, Dtype dtype,
                 bool twin) {
        check_weight(weight, nin, "matrix");
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
        if (int16(weight, dtype)) return product16(weight, X, Y, nin, nout, nbatch, accumulate, runs, twin);
        const uint8_t* w = bytes(weight.data);
        const float* x = f32(X);
        float* y = f32(Y);
        unsigned long long ni = nin, no = nout;
        int add = accumulate;
        void* args[] = {&w, &x, &y, &ni, &no, &add};
        launch(weight.type == quant::GGML_TYPE_Q8_0 ? kMatmulQ8 : kMatmul, (unsigned)nout, (unsigned)nbatch, args);
        // Every product here reads F32 activations, whatever dtype was asked for.
        record_matrix_path(MatrixPath::f32);
    }
    // The twin of nbatch rows of X, then each run on its kernel: generated rows (extent 1) on the row kernel, kColumns at a time, prompt rows on the tile.
    // Without runs a call of one row is a generated row and any other a prompt's.
    void product16(const WeightView& weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, bool accumulate, RowRuns runs, bool twin) {
        const size_t blocks = size_mul(nbatch, nin / 32);
        const size_t qbytes = size_mul(blocks, 32 * sizeof(int16_t)), need = size_add(qbytes, size_mul(blocks, sizeof(float)));
        if (twin) {
            if (!twin_ || twin_->size() < need) {
                twin_ = alloc(need, Memory::device);
            }
            const float* x = f32(X);
            int16_t* q = (int16_t*)as_rocm(*twin_).data();
            float* d = (float*)((uint8_t*)q + qbytes);
            unsigned long long nb = blocks;
            void* args[] = {&x, &q, &d, &nb};
            launch(kTwin16, (unsigned)((blocks + kThreads / 32 - 1) / (kThreads / 32)), 1, args);
        }
        const uint8_t* w = bytes(weight.data);
        const int16_t* q0 = (const int16_t*)as_rocm(*twin_).data();
        const float* d0 = (const float*)((const uint8_t*)q0 + qbytes);
        float* y0 = f32(Y);
        unsigned long long ni = nin, no = nout;
        int add = accumulate;
        const RowRun whole[1] = {{nbatch, nbatch == 1 ? size_t(1) : size_t(2)}};
        const RowRuns all = runs.n ? runs : RowRuns{whole, 1};
        size_t begin = 0;
        for (size_t r = 0; r < all.n; ++r) {
            const size_t end = all.runs[r].end;
            if (end <= begin || end > nbatch) throw std::runtime_error("rocm: row runs out of order or past the call");
            for (size_t at = begin; at < end;) {
                const size_t rows = all.runs[r].extent == 1 ? std::min(end - at, kColumns) : end - at;
                const int16_t* q = q0 + at * nin;
                const float* d = d0 + at * (nin / 32);
                float* y = y0 + at * nout;
                if (all.runs[r].extent == 1) {
                    int ncols = (int)rows;
                    void* args[] = {&w, &q, &d, &y, &ni, &no, &ncols, &add};
                    const Kernel build = rows <= 1 ? kRows16x1 : rows <= 2 ? kRows16x2 : rows <= 4 ? kRows16x4 : rows <= 8 ? kRows16x8 : kRows16x16;
                    launch(build, (unsigned)((nout + rows_per_block_ - 1) / rows_per_block_), 1, args);
                } else {
                    unsigned long long nr = rows;
                    void* args[] = {&w, &q, &d, &y, &ni, &no, &nr, &add};
                    launch(kTile16, (unsigned)((nout + 15) / 16), (unsigned)((rows + 15) / 16), args);
                }
                at += rows;
            }
            begin = end;
        }
        if (begin != nbatch) throw std::runtime_error("rocm: row runs short of the call");
        record_matrix_path(MatrixPath::block_int16);
    }
    void gather(Slice dst, CSlice src, size_t width, const uint32_t* ids, size_t count) {
        span(*dst.buffer, dst.offset * sizeof(float), size_mul(size_mul(count, width), sizeof(float)));
        float* d = f32(dst);
        const float* s = f32(src);
        const unsigned* ip = (const unsigned*)arg(ids, count * sizeof(uint32_t));
        unsigned long long w = width;
        void* args[] = {&d, &s, &ip, &w};
        launch(kGather, (unsigned)((width + kThreads - 1) / kThreads), (unsigned)count, args);
    }
    void launch(Kernel k, unsigned gx, unsigned gy, void** args, unsigned gz = 1) {
        raise_failed();
        launch_check(hip().hipModuleLaunchKernel(kernels_[k], gx, gy, gz, kThreads, 1, 1, 0, stream_, args, nullptr), kKernelNames[k]);
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
    void raise_failed() {
        std::lock_guard<std::mutex> lk(failed_mu_);
        if (!failed_.empty()) throw std::runtime_error(failed_);
    }

    RocmDevice dev_;
    std::shared_ptr<Graves> graves_ = std::make_shared<Graves>();   // before every buffer member, so it outlives them
    hipStream_t stream_ = nullptr;
    hipModule_t module_ = nullptr, int16_module_ = nullptr, dot_module_ = nullptr;
    BufferPtr parts_;  // attention's chunk partials, grown as calls need
    BufferPtr twin_;   // the block-int16 twin of the last product's activations, grown as calls need
    BufferPtr staged_;   // a product bound for host memory, formed here first
    size_t rows_per_block_ = kThreads / dev_.caps.subgroup_size;   // the row kernel's output rows a block, a wave each
    std::array<hipFunction_t, kKernels> kernels_{};
    std::array<hipEvent_t, kEvents> events_{};
    Ring args_, stage_;
    TicketLedger tickets_;
    bool opened_ = false;
    std::mutex failed_mu_;
    std::string failed_;
};

RocmKVStorage::RocmKVStorage(RocmBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens, KVType kt, KVType vt,
                             const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools)
    : BlockKVStorage(owner, "rocm", kBlockTokens, layers, heads, dim, max_tokens, kt, vt, auxiliary_widths, auxiliary_pools) {}

// Dropping the old buffer is enough: its memory waits in the graves for the copies recorded before.
void RocmKVStorage::retire(const BufferPtr&) {}

} // namespace

BackendPtr make_rocm_backend(int device) { return std::make_shared<RocmBackend>(device); }

} // namespace backend
