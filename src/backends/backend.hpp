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

class Backend {
public:
    virtual ~Backend() = default;

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
                        Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}) = 0;

    // The output head: its results are the logits a caller reads directly, so a backend may keep more precise activations for it than for the projections inside the layers.
    virtual void matmul_logits(uint32_t type, CSlice data, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch,
                               RowRuns runs = {}) {
        matmul(type, data, X, Y, nin, nout, nbatch, runs);
    }

    // Y += W X, the projection whose output joins the residual stream: the model asks for the sum and each backend produces it its own way.
    // The CPU computes the product into scratch and adds; a device folds the add into the matmul's store, one dispatch fewer per projection.
    virtual void matmul_add(uint32_t type, CSlice data, CSlice X,
                            Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}) = 0;

    // Gather `count` rows of an embedding table into `dst`, row-major, `nin` floats each.
    // This is an op rather than a model-side read because the table is a Buffer: a device backend holds it in its own memory and the model cannot address it.
    virtual void embed(Slice dst, uint32_t type, CSlice table,
                       size_t nin, size_t nrows, const uint32_t* ids,
                       size_t count) = 0;

    // Independent projections of the same X; outputs must not overlap each other, X, or any weights.
    // Outputs are observable after wait(), sync() or read(), as for matmul.
    virtual void matmul_group(std::initializer_list<Projection> projections,
                              CSlice X, size_t nin, size_t nbatch, RowRuns runs = {}) {
        for (const auto& p : projections) {
            if (!p.data.buffer) throw std::runtime_error("backend: projection without storage");
            matmul(p.type, p.data, X, p.out, nin, p.rows, nbatch, runs);
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

    // Per-head RMS norm followed by RoPE over a batch of rows: row r starts at x + r*stride with `heads` contiguous heads of `2*half` floats, at position pos[r] in the `cos`/`sin` tables.
    // Positions are per row because a batch may carry several sequences; norm and RoPE are one op so a device gets one launch per layer.
    virtual void norm_rope_rows(Slice x, size_t rows, size_t stride,
                                size_t heads, CSlice w, float eps,
                                CSlice cos, CSlice sin, size_t half,
                                const uint32_t* pos) = 0;

    // Normalize and rotate q and k in place, then write k and v into the views' KV blocks using the primitive ops' row layouts.
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
        norm_rope_rows(q, rows, q_stride, n_head, q_w, rope.eps, rope.cos, rope.sin, rope.half, rope.pos);
        norm_rope_rows(k, rows, kv_stride, n_head_kv, k_w, rope.eps, rope.cos, rope.sin, rope.half, rope.pos);
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
                                size_t nrows, const Routing& routing, RowRuns runs = {}) = 0;

    // The routed projection whose output joins the residual stream: Y row r += sum over j in order of weights[e] * dot(expert ids[e], X row e), e = r*k + j.
    // The weighted sum is formed first and then added, so a row's result does not depend on how its slots were computed.
    virtual void matmul_experts_add(uint32_t type, CSlice data, CSlice X, Slice Y, size_t nin, size_t nout,
                                    size_t nrows, const Routing& routing, RowRuns runs = {}) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
