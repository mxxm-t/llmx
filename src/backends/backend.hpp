#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <functional>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Backends own storage and parallelize primitive ops; models use buffer handles and own multi-device placement.
// The execution and ownership contracts are in docs/DEVICE-EXECUTION.md.

namespace backend {

// Storage owned by the backend that allocated it.
// Models pass buffer handles to compute ops; host-visible results can be read after retirement.
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    // Non-null for nonempty host-addressable allocations: Memory::host_visible and all host-backend storage.
    // What it points at is current only after wait() or sync().
    virtual const void* host_ptr() const = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

// Throws unless `bytes` bytes from byte `off` lie inside the buffer.
inline void span(const Buffer& b, size_t off, size_t bytes) {
    if (off > b.size() || bytes > b.size() - off)
        throw std::runtime_error("backend: buffer range outside the allocation");
}

// Size arithmetic for storage and operands: a product or sum that would wrap throws instead.
inline size_t size_mul(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error("backend: size overflows");
    return a * b;
}
inline size_t size_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error("backend: size overflows");
    return a + b;
}

// device memory may be unreachable from the host; host_visible permits host_ptr() access after retirement.
// Both use the same storage on a host backend.
enum class Memory { device, host_visible };

// A submission.
// `submit()` hands everything enqueued so far to the device and returns one; `wait()` blocks until that submission has retired.
using Ticket = uint64_t;

// An operand identifies backend storage with a buffer and a float offset.
struct Slice {
    Buffer* buffer = nullptr;
    size_t offset = 0;
};
struct CSlice {
    const Buffer* buffer = nullptr;
    size_t offset = 0;
    CSlice() = default;
    CSlice(const Buffer* b, size_t o) : buffer(b), offset(o) {}
    CSlice(const Slice& s) : buffer(s.buffer), offset(s.offset) {}
};

enum class Dtype { f32, f16, bf16 };
enum class MatrixPath { f32, f16, bf16, block_int16 };

// The dispatched matrix forms, accumulated without allocating; read after the work has completed.
class MatrixPaths {
public:
    void record(MatrixPath path) { bits_ |= 1u << unsigned(path); }
    std::vector<std::string> take() {
        static const char* const names[] = {"f32", "f16", "bf16", "block-int16"};
        std::vector<std::string> paths;
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i)
            if (bits_ & (1u << i)) paths.emplace_back(names[i]);
        bits_ = 0;
        return paths;
    }
private:
    uint32_t bits_ = 0;
};
inline const char* dtype_name(Dtype dtype) {
    switch (dtype) {
        case Dtype::f32: return "f32";
        case Dtype::f16: return "f16";
        case Dtype::bf16: return "bf16";
    }
    return "unknown";
}

struct Projection {
    uint32_t type;
    CSlice data;
    Slice out;
    size_t rows;
};

// The rows of a matmul grouped by the prompt they belong to, so a device picks a row's kernel by `extent` rather than by the call's width and a prompt computes the same however its rows are batched (docs/VULKAN.md, batch invariance).
// `end` is one past the run's last row, runs in row order; `extent` is the position one past the prompt's last token for prompt rows, 1 for a generated token.
// Without runs a backend chooses by the call's width.
struct RowRun {
    size_t end;
    size_t extent;
};
struct RowRuns {
    const RowRun* runs = nullptr;
    size_t n = 0;
};

// Calls `each(first, count, key)` over the stretches of adjacent runs whose `key(run)` is equal, once the runs are known to be in row order and to end at row `n`.
// Every run is checked before the first call, so a malformed list reaches no rows; a caller handles a call without runs itself.
template <typename Key, typename Each>
void for_each_run(size_t n, RowRuns runs, const Key& key, const Each& each) {
    size_t end = 0;
    for (size_t i = 0; i < runs.n; ++i) {
        if (runs.runs[i].end < end) throw std::runtime_error("backend: row runs out of order");
        end = runs.runs[i].end;
    }
    if (end != n) throw std::runtime_error("backend: row runs do not cover the batch");
    size_t start = 0;
    for (size_t i = 0; i < runs.n;) {
        const auto k = key(runs.runs[i]);
        size_t j = i + 1;
        while (j < runs.n && key(runs.runs[j]) == k) ++j;
        const size_t stop = runs.runs[j - 1].end;
        if (stop > start) each(start, stop - start, k);
        start = stop;
        i = j;
    }
}

// The backend owns KV storage and its block layout; models keep logical views (docs/KV-CACHE.md).
// A block id indexes the same block in every layer of one KVStorage.
struct KVLayout {
    size_t block_tokens;
};

// Whole blocks of `block_tokens` for `tokens` positions, without the usual +bt-1 overflow.
inline size_t blocks_for(size_t tokens, size_t block_tokens) {
    return tokens / block_tokens + (tokens % block_tokens != 0);
}

// How a cache side is stored; the CLI's --cache-type-k and --cache-type-v.
enum class KVType { f32, f16 };
inline size_t kv_elem_bytes(KVType t) { return t == KVType::f16 ? 2 : 4; }
// The same two names on every backend.
inline KVType kv_type_of(const std::string& name) {
    if (name == "f32") return KVType::f32;
    if (name == "f16") return KVType::f16;
    throw std::runtime_error("unknown cache type '" + name + "' (f32 or f16)");
}
inline const char* kv_type_name(KVType t) { return t == KVType::f16 ? "f16" : "f32"; }

class KVStorage {
public:
    virtual ~KVStorage() = default;
    virtual size_t max_blocks() const = 0;
    // Bytes retained for blocks now, and the most held across successful growths (a growth copies, so old plus new is held for a moment).
    // A growth that failed part way is not counted.
    virtual size_t allocated_bytes() const = 0;
    virtual size_t peak_bytes() const = 0;
    // Backs every block of the budget at once, so no later write grows the storage.
    virtual void back_all() = 0;
};

// One sequence's history in one storage: logical block i is physical block blocks[i], `length` entries are committed, and `nq` rows of this pass belong to the sequence.
// Row b is at position length + b and attends through it, so the table must cover length + nq.
// `extent` is the rows' RowRun extent, 0 when unknown.
struct KVView {
    KVStorage* storage;
    const int32_t* blocks;
    size_t n_blocks;
    size_t length;
    size_t nq;
    size_t extent = 0;
};

// The width of the linear-attention layers' causal conv, 4 in every qwen35 file; a slot carries the kConvTaps - 1 raw rows before a sequence's next token.
inline constexpr size_t kConvTaps = 4;

// The epsilon of the linear-attention layers' L2 norms of q and k, which no file carries (docs/QWEN35.md, What the converter folds).
inline constexpr float kL2NormEps = 1e-6f;

// One linear-attention layer's recurrent state for one sequence (docs/QWEN35.md, The recurrent state): K heads of k_dim and V heads of v_dim, V head j reading K head j mod k_heads.
struct StateShape {
    size_t k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0;
    // The conv's channels, those of a raw projection row [q: k_heads k_dim | k: k_heads k_dim | v: v_heads v_dim].
    size_t channels() const { return size_add(size_mul(2 * k_heads, k_dim), size_mul(v_heads, v_dim)); }
    // Floats of each V head's matrix, laid out [K row][V column].
    size_t matrix_floats() const { return size_mul(k_dim, v_dim); }
    // Floats of a slot: every V head's matrix, V head by V head, then the conv's kConvTaps - 1 carried rows of channels(), oldest first.
    size_t slot_floats() const {
        return size_add(size_mul(v_heads, matrix_floats()), size_mul(kConvTaps - 1, channels()));
    }
    // Bytes of one layer's buffer of `slots` slots.
    size_t layer_bytes(size_t slots) const { return size_mul(size_mul(slots, slot_floats()), sizeof(float)); }
};

// The recurrent state of `layers` linear-attention layers, made whole by Backend::state_alloc and never grown: one buffer per layer, holding every slot back to back, all F32.
// It refuses a buffer that cannot hold its slots, so an op resolving a slot stays inside its buffer.
class StateStorage {
public:
    StateStorage(std::vector<BufferPtr> buffers, size_t slots, const StateShape& shape)
        : buffers_(std::move(buffers)), slots_(slots), shape_(shape) {
        const size_t bytes = shape_.layer_bytes(slots_);
        for (const BufferPtr& b : buffers_)
            if (!b || b->size() < bytes) throw std::runtime_error("backend: state buffer smaller than its slots");
    }
    size_t layers() const { return buffers_.size(); }
    size_t slots() const { return slots_; }
    const StateShape& shape() const { return shape_; }
    Buffer& layer(size_t l) const {
        if (l >= buffers_.size()) throw std::runtime_error("backend: state layer outside the storage");
        return *buffers_[l];
    }

private:
    std::vector<BufferPtr> buffers_;
    size_t slots_;
    StateShape shape_;
};

// One sequence's rows of a pass in one StateStorage: `nq` rows, in view order, continue a history of `length` tokens.
// The state is read from slot `src` and written to slot `dst`, the same slot except in a verify, and length 0 reads a zero state whatever `src` holds, so a recycled slot needs no clearing.
struct StateView {
    StateStorage* storage = nullptr;
    size_t src = 0, dst = 0;
    size_t length = 0;
    size_t nq = 0;
};

// Throws unless every view has rows, names a storage holding `layer` with slots inside it and the first view's shape, and writes a slot no other view reads or writes, so the views do not depend on their order.
// Returns the rows of all the views.
inline size_t check_state_views(const StateView* views, size_t n_views, size_t layer) {
    if (n_views && !views) throw std::runtime_error("backend: state op without views");
    size_t rows = 0;
    for (size_t i = 0; i < n_views; ++i) {
        const StateView& v = views[i];
        if (!v.storage) throw std::runtime_error("backend: state view without storage");
        if (layer >= v.storage->layers()) throw std::runtime_error("backend: state layer outside the storage");
        if (v.src >= v.storage->slots() || v.dst >= v.storage->slots())
            throw std::runtime_error("backend: state slot outside the storage");
        if (!v.nq) throw std::runtime_error("backend: state view without rows");
        const StateShape& a = v.storage->shape();
        const StateShape& b = views[0].storage->shape();
        if (a.k_heads != b.k_heads || a.v_heads != b.v_heads || a.k_dim != b.k_dim || a.v_dim != b.v_dim)
            throw std::runtime_error("backend: state views of different shapes");
        for (size_t j = 0; j < i; ++j) {
            const StateView& u = views[j];
            if (u.storage != v.storage) continue;
            if (u.dst == v.dst || u.dst == v.src || u.src == v.dst)
                throw std::runtime_error("backend: a state slot written by one view is used by another");
        }
        rows = size_add(rows, v.nq);
    }
    return rows;
}

// The ops some backends lack, which a model's plan names for each part that issues one, so a backend without one refuses the model at load rather than in a pass (Backend::implements).
enum class Op : uint8_t { causal_conv_silu, gated_delta_rule, gated_rms_norm, norm_rope_partial, sigmoid_mul };
inline const char* op_name(Op op) {
    switch (op) {
        case Op::causal_conv_silu: return "causal_conv_silu";
        case Op::gated_delta_rule: return "gated_delta_rule";
        case Op::gated_rms_norm: return "gated_rms_norm";
        case Op::norm_rope_partial: return "norm_rope_partial";
        case Op::sigmoid_mul: return "sigmoid_mul";
    }
    return "an unknown op";
}

class Backend {
public:
    virtual ~Backend() = default;

    // Complete native policies, preferred first; emulation alone does not add a policy to auto's choices.
    virtual std::vector<Dtype> native_dtypes() const { return {Dtype::f32}; }
    virtual bool emulates_dtype(Dtype) const { return false; }
    virtual std::string dtype_path(Dtype dtype) const { return dtype_name(dtype); }

    // A model's stage swaps in its own evidence, then restores the caller's on every exit.
    void swap_matrix_paths(MatrixPaths& paths) noexcept { std::swap(matrix_paths_, paths); }


    // Whether the implemented weight-reading ops support this storage type on this device, checked before a model adopts its weights.
    virtual bool supports_type(uint32_t type) const { (void)type; return false; }
    // Whether this backend runs `op`; one that does not refuses it by name when it is called anyway.
    virtual bool implements(Op op) const {
        (void)op;
        return false;
    }

    // Set the worker thread count hint; 0 leaves the current count unchanged.
    virtual void set_threads(int n) = 0;

    // Number of worker threads this backend will actually use (after the last set_threads, or auto-detected).
    // Reported by the CLI; a device backend returns whatever is meaningful for it, or 0.
    virtual int threads_available() const = 0;

    // Bytes this backend can still allocate for weights, caches and activations, as the device or the operating system reports them now; nothing when it cannot tell.
    // A placement across several devices is fitted against it (docs/MULTI-DEVICE.md).
    virtual std::optional<size_t> memory_available() const { return std::nullopt; }

    // What adopting a matrix of this quant type and shape keeps resident on this backend: its bytes, and any copy the backend makes of it, so a fit counts it.
    // `product` is a matrix a matrix product reads as its weights, as opposed to a gathered table, a norm or a routed expert stack.
    virtual size_t resident_bytes(uint32_t type, size_t nin, size_t rows, size_t bytes, bool product) const {
        (void)type; (void)nin; (void)rows; (void)product;
        return bytes;
    }

    // Host memory the backend holds for its own use, such as staging for uploads, which counts against the host rather than against memory_available().
    virtual size_t host_resident() const { return 0; }

    // Whether adopt() reads the caller's memory in place rather than copying it into the backend's own, so weights placed here cost no memory of this backend.
    virtual bool reads_in_place() const { return false; }

    // Whether this backend is the CPU itself, so experts placed on the CPU beside it are already where they run; a device may read in place and still not be the CPU.
    virtual bool is_cpu() const { return false; }

    // The class of rows of this RowRun extent: rows of two extents of one class take the same arithmetic in every op of this backend, so they give the same bits.
    // A backend claims only what it proves, so by default every extent is a class of its own (docs/SPECULATIVE.md, section 1).
    virtual size_t row_class(size_t extent) const { return extent; }

    // Where a backend made to time its work (make_backend's diagnostics) held its caller since it was made, in milliseconds: waiting on tickets, for a free command slot and for staging, and in uploads (write) apart from those waits.
    // A backend that computes as it records never waits, and one not made to time its work reports nothing.
    struct HostTimes {
        double ticket_ms = 0, slot_ms = 0, staging_ms = 0, write_ms = 0;
    };
    virtual HostTimes host_times() const { return {}; }
    // The device time of the work recorded since the last call, in milliseconds, by timestamps on a backend made to time its work, which waits for its queue to read them; negative where the backend keeps none.
    virtual double device_ms() { return -1.0; }

    // Memory the backend's kernels take for themselves beside weights, caches and activations, such as split partials and merge state, out of `free` bytes; a fit keeps it back.
    virtual size_t scratch_reserve(size_t free) const { (void)free; return 0; }

    // Invoke once on the caller and complete all cleanup before returning.
    virtual void run_prefill(const std::function<void()>& work) { work(); }

    // Zero-filled backend storage.
    virtual BufferPtr alloc(size_t bytes, Memory where = Memory::device) = 0;

    // Make src reachable by this backend's ops.
    // A backend that reads in place (reads_in_place) borrows src for the returned buffer's life and does not read it inside adopt; one that copies has consumed src when adopt returns, so the caller may release it then.
    virtual BufferPtr adopt(const void* src, size_t bytes) = 0;

    // Storage for a weight the caller fills with write before any op reads it; it need not be zeroed, and the backend keeps it as it keeps an adopted weight.
    virtual BufferPtr alloc_weight(size_t bytes) { return alloc(bytes); }

    // A buffer over the caller's page-aligned memory that copy reads from in place, or null where this backend cannot read it so.
    // The caller keeps the memory for the buffer's life and leaves it unchanged until the copies out of it retire.
    virtual BufferPtr wrap_host(void* memory, size_t bytes) {
        (void)memory;
        (void)bytes;
        return nullptr;
    }

    // Ops enqueue on one stream; submit() flushes and returns a monotonic ticket, and wait(t) retires that submission and everything before it.
    // Results require wait(), sync() or read(); CPU ops complete eagerly (docs/DEVICE-EXECUTION.md).
    virtual Ticket submit() = 0;
    // Retirement cannot throw: callers release storage afterward, so failure to establish completion must terminate.
    virtual void wait(Ticket t) noexcept = 0;
    // Also retires work behind no ticket, including on failure paths.
    virtual void sync() noexcept = 0;

    // Copy to host memory after retiring earlier ops; logits can instead use host_visible storage after a wait.
    virtual void read(const Buffer& src, size_t off, void* dst, size_t bytes) = 0;

    // Storage to storage, within this backend. The KV cache grows with it.
    virtual void copy(Buffer& dst, size_t dst_off,
                      const Buffer& src, size_t src_off, size_t bytes) = 0;

    // Enqueue a host-to-storage copy, consuming the source before returning so callers can reuse staging immediately.
    virtual void write(Buffer& dst, size_t off, const void* src, size_t bytes) = 0;

    // Y[b*nout + o] = dot(row_o, X + b*nin) for all b in [0,nbatch) and o in [0,nout); X and Y are row-major with nbatch rows.
    // `type` is the quant type the registry is keyed by (the GGUF id), so every block format gets the batched path.
    virtual void matmul(uint32_t type, CSlice data, CSlice X,
                        Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // The output head: its results are the logits a caller reads directly, so a backend may keep more precise activations for it than for the projections inside the layers.
    virtual void matmul_logits(uint32_t type, CSlice data, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch,
                               RowRuns runs = {}, Dtype dtype = Dtype::f16) {
        matmul(type, data, X, Y, nin, nout, nbatch, runs, dtype);
    }

    // Y += W X, the projection whose output joins the residual stream: the model asks for the sum and each backend produces it its own way.
    // The CPU computes the product into scratch and adds; a device folds the add into the matmul's store, one dispatch fewer per projection.
    virtual void matmul_add(uint32_t type, CSlice data, CSlice X,
                            Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // Gather `count` rows of an embedding table into `dst`, row-major, `nin` floats each.
    // This is an op rather than a model-side read because the table is a Buffer: a device backend holds it in its own memory and the model cannot address it.
    virtual void embed(Slice dst, uint32_t type, CSlice table,
                       size_t nin, size_t nrows, const uint32_t* ids,
                       size_t count) = 0;

    // Independent projections of the same X; outputs must not overlap each other, X, or any weights.
    // Outputs are observable after wait(), sync() or read(), as for matmul.
    virtual void matmul_group(std::initializer_list<Projection> projections,
                              CSlice X, size_t nin, size_t nbatch, RowRuns runs = {}, Dtype dtype = Dtype::f16) {
        for (const auto& p : projections) {
            if (!p.data.buffer) throw std::runtime_error("backend: projection without storage");
            matmul(p.type, p.data, X, p.out, nin, p.rows, nbatch, runs, dtype);
        }
    }

    virtual KVLayout kv_layout() const = 0;

    // Keys and values for `layers` layers of n_head_kv x head_dim, whole blocks for max_tokens positions, each side stored as f32 or f16 (round-to-nearest, read back exactly), so only the stored precision differs between backends.
    // Nothing is backed until a block is written.
    // A backend without a type throws rather than substituting.
    virtual std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv,
                                                size_t head_dim, size_t max_tokens,
                                                KVType k_type = KVType::f32,
                                                KVType v_type = KVType::f32) = 0;

    // Store token-major [rows, n_head_kv, head_dim] rows, laid out in view order: view v owns the next views[v].nq rows and they go to positions length .. length + nq of its sequence.
    // Several views carry rows from several sequences in one call.
    virtual void kv_write(size_t layer, const KVView* views, size_t n_views,
                          CSlice k, CSlice v) = 0;

    // Causal GQA: Q/out are [rows, n_head, head_dim] in the same view order, and row b of view v attends through views[v].length + b.
    // A device backend gets every sequence of the pass in one launch.
    virtual void attention(CSlice Q, size_t layer, const KVView* views,
                           size_t n_views, Slice out,
                           int n_head, int n_head_kv, int head_dim) = 0;

    // dst[i] = src[i] * rsqrt(mean(src^2) + eps) * w[i]  (RMS norm), the one-row case of rms_norm_rows.
    void rms_norm(Slice dst, CSlice src, CSlice w, size_t n, float eps) {
        rms_norm_rows(dst, src, w, 1, n, n, eps);
    }

    // The batched forms below exist so the model layer holds no elementwise loops and needs no host parallelism of its own.
    // Each is one call per layer instead of one per row (or per head, per row), which is what makes the graph expressible on a device: see docs/ROADMAP.md #4a.

    // RMS norm of `rows` rows of `n` floats against a shared weight.
    // Row r is at src/dst + r*stride. src and dst may alias only if identical.
    // `runs`, when given, are those of the matmul that reads dst next, so a device can write dst's activations in the form that matmul's kernel reads.
    virtual void rms_norm_rows(Slice dst, CSlice src, CSlice w,
                               size_t rows, size_t n, size_t stride, float eps, RowRuns runs = {}) = 0;

    // Per-head RMS norm over `head_dim` floats, then RoPE on the first `rope_dim` of them only, pairs (i, i + rope_dim / 2) at pos[r] in `cos`/`sin` tables of rope_dim / 2 entries a position.
    // Source head h of row r starts at src + r * src_stride + h * src_head_stride, and the heads are written contiguously, heads * head_dim floats a row of dst.
    // Positions are per row because a batch may carry several sequences; norm and RoPE are one op so a device gets one launch per layer.
    // With rope_dim equal to head_dim it is the whole head's rope; for text the qwen35 rope sections give every frequency the same position, so they reduce to this (docs/QWEN35.md, Gated attention).
    // dst may alias src only if identical and src's heads are contiguous.
    virtual void norm_rope_partial(Slice dst, CSlice src, size_t rows, size_t src_stride, size_t src_head_stride,
                                   size_t heads, size_t head_dim, size_t rope_dim, CSlice w, float eps,
                                   CSlice cos, CSlice sin, const uint32_t* pos) = 0;

    // Normalize and rotate q and k in place over the whole head, then write k and v into the views' KV blocks using the primitive ops' row layouts.
    // A device may fuse these consecutive ops; k must still hold its normalized, rotated rows afterward.
    struct RopeArgs {
        CSlice cos, sin;
        size_t half;
        const uint32_t* pos;
        float eps;
    };
    virtual void norm_rope_kv(Slice q, size_t q_stride, size_t n_head, CSlice q_w,
                              Slice k, CSlice v, size_t kv_stride, size_t n_head_kv, CSlice k_w,
                              const RopeArgs& rope, size_t rows, size_t layer,
                              const KVView* views, size_t n_views) {
        const size_t dim = 2 * rope.half;
        norm_rope_partial(q, q, rows, q_stride, dim, n_head, dim, dim, q_w, rope.eps, rope.cos, rope.sin, rope.pos);
        norm_rope_partial(k, k, rows, kv_stride, dim, n_head_kv, dim, dim, k_w, rope.eps, rope.cos, rope.sin, rope.pos);
        kv_write(layer, views, n_views, k, v);
    }

    // dst[i] = silu(gate[i]) * up[i], the SwiGLU elementwise stage.
    // `runs`, when given, group dst's rows of n / rows floats, `rows` being the last run's end, by prompt as matmul's runs do, for the matmul that reads dst next, as for rms_norm_rows.
    virtual void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n, RowRuns runs = {}) = 0;

    // dst[i] += src[i], the residual add.
    virtual void add(Slice dst, CSlice src, size_t n) = 0;

    // dst row i = src row rows[i], `width` floats each.
    // Compacts the rows of a pass that want logits, which a batch mixing prefill and decode entries leaves non-contiguous, so the output head runs once over exactly those rows.
    virtual void gather_rows(Slice dst, CSlice src, size_t width,
                             const uint32_t* rows, size_t count) = 0;

    // Mixture of experts (docs/EXECUTION.md). A layer's router scores pick `k` of `n_expert` experts per token row; slot j of row r is entry r*k + j.
    // `ids` holds the chosen experts as 32-bit integers in float-sized slots, so they live in the activation arena beside everything else and never leave the device.
    struct Routing {
        CSlice ids, weights;
        size_t k, n_expert;
    };

    // For each of `rows` rows of `scores` (n_expert floats each), the k experts of highest softmax probability, the most probable first and ties to the lower id, with their probabilities, divided by the k probabilities' sum when `normalize`.
    // The softmax is over all n_expert scores.
    virtual void route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize,
                               Slice ids, Slice weights) = 0;

    // Routed projections of one X: entry e = r*k + j of projection p is out[e*rows_p + o] = dot(row o of expert ids[e], X row r), for up to three projections.
    // A projection's data holds its n_expert matrices of `rows` rows each back to back; `runs` are matmul's, over the `nrows` token rows.
    virtual void matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin,
                                size_t nrows, const Routing& routing, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // The routed projection whose output joins the residual stream: Y row r += sum over j in order of weights[e] * dot(expert ids[e], X row e), e = r*k + j.
    // The weighted sum is formed first and then added, so a row's result does not depend on how its slots were computed.
    virtual void matmul_experts_add(uint32_t type, CSlice data, CSlice X, Slice Y, size_t nin, size_t nout,
                                    size_t nrows, const Routing& routing, RowRuns runs = {}, Dtype dtype = Dtype::f16) = 0;

    // The ops of the qwen35 layers (docs/QWEN35.md, The forward pass), with norm_rope_partial above.

    // The recurrent state of `layers` linear-attention layers with `slots` slots each, allocated now and zero-filled, so no pass allocates state.
    std::unique_ptr<StateStorage> state_alloc(size_t layers, size_t slots, const StateShape& shape) {
        if (!layers || !slots || !shape.k_heads || !shape.k_dim || !shape.v_dim || !shape.v_heads || shape.v_heads % shape.k_heads)
            throw std::runtime_error("backend: invalid state shape");
        const size_t bytes = shape.layer_bytes(slots);
        std::vector<BufferPtr> buffers;
        for (size_t l = 0; l < layers; ++l) buffers.push_back(alloc(bytes));
        return std::make_unique<StateStorage>(std::move(buffers), slots, shape);
    }

    // Slot `src` into slot `dst` in every layer of `s`, enqueued as copy is: a checkpoint restored, or a state taken back into a live slot.
    void state_copy(StateStorage& s, size_t dst, size_t src) {
        if (dst >= s.slots() || src >= s.slots()) throw std::runtime_error("backend: state slot outside the storage");
        if (dst == src) return;
        const size_t bytes = s.shape().slot_floats() * sizeof(float);
        for (size_t l = 0; l < s.layers(); ++l) copy(s.layer(l), dst * bytes, s.layer(l), src * bytes, bytes);
    }

    // The linear-attention layers' causal conv, then SiLU: out[t][c] = silu(sum over tap i of w[c * kConvTaps + i] * x[t - kConvTaps + 1 + i][c]), w being `ssm_conv1d` as stored, so tap kConvTaps - 1 multiplies row t.
    // x and out are the views' rows of the storage's channels(), in view order; a view reads the rows before its first from slot src, rows before its sequence's start being zero, and leaves its last kConvTaps - 1 raw rows in slot dst.
    virtual void causal_conv_silu(Slice out, CSlice x, CSlice w, size_t layer, const StateView* views, size_t n_views) = 0;

    // The gated delta rule of the linear-attention layers (docs/QWEN35.md, Linear attention, steps 3 to 5), token by token for every (view, V head) from slot src's matrices into slot dst's.
    // qkv holds the conv's output rows [q | k | v] of channels(); alpha and b are the rows of `ssm_alpha` and `ssm_beta`, and a and dt_bias `ssm_a` and `ssm_dt.bias`, v_heads floats each; out is v_heads * v_dim floats a row.
    // q and k are L2-normed with kL2NormEps and q scaled by 1 / sqrt(k_dim), beta is sigmoid(b), and the decay exp(a * softplus(alpha + dt_bias)) is 0 below 2^-126.
    virtual void gated_delta_rule(Slice out, CSlice qkv, CSlice alpha, CSlice b, CSlice a, CSlice dt_bias,
                                  size_t layer, const StateView* views, size_t n_views) = 0;

    // dst = RMSNorm(x; w) * silu(z) over each of `heads` heads of `dim` floats in each of `rows` rows, w being one dim-wide weight every head shares.
    // dst may alias x only if identical, and never z; `runs` as for rms_norm_rows.
    virtual void gated_rms_norm(Slice dst, CSlice x, CSlice z, CSlice w, size_t rows, size_t heads, size_t dim, float eps,
                                RowRuns runs = {}) = 0;

    // dst[r][h][d] = x[r][h][d] * sigmoid(gate[r * gate_stride + h * gate_head_stride + d]), x and dst being `rows` rows of heads * dim floats.
    // The output gate reads each query head's gate in place from `attn_q`'s rows; a scale of one value per row is heads = the row's width, dim = 1 and gate_head_stride = 0.
    // dst may alias x only if identical; `runs` as for silu_mul.
    virtual void sigmoid_mul(Slice dst, CSlice x, CSlice gate, size_t rows, size_t heads, size_t dim,
                             size_t gate_stride, size_t gate_head_stride, RowRuns runs = {}) = 0;
protected:
    void record_matrix_path(MatrixPath path) { matrix_paths_.record(path); }

private:
    MatrixPaths matrix_paths_;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
