#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <memory>
#include <functional>
#include <initializer_list>

// Compute backend abstraction. The inference graph runs its primitive ops
// (matmul, attention, RMSNorm, RoPE) through a Backend so the same model code
// can target CPU now and ROCm / Vulkan later.
//
// Operands are a Buffer and an offset rather than host pointers, so the
// backend owns its storage and the model layer never dereferences it. That is
// what lets a device backend keep weights and activations resident. Two things
// remain before a vendor backend is writable: KV blocks on buffers, and the
// enqueue/sync contract that makes ops asynchronous. They are steps 5 and 6 of
// docs/DEVICE-EXECUTION.md, which designs the whole migration, and
// docs/ROADMAP.md #4a.
//
// Host parallelism is deliberately NOT on this interface: a callback run
// across host threads has no device implementation. Backends parallelize
// inside their own ops. `threads_available` remains so the CLI can report what
// a backend is using.
//
// Multi-device / multi-node: the split strategies (per-layer, per-tensor,
// per-row) live at the model layer and are documented in docs/ROADMAP.md; they
// are NOT implemented yet.

namespace backend {

// Storage owned by the backend that allocated it. The model layer holds
// handles and never dereferences them, so a device backend can keep weights
// resident instead of receiving a host pointer on every call.
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    // Non-null only where the host can address the allocation directly. A
    // device backend returns nullptr and the caller must use read/write.
    virtual const void* host_ptr() const = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

// Where an operand lives: a buffer and a float offset into it. Ops take these
// rather than pointers so a device backend never receives a host address.
// Offsets are in floats, because every activation is float and a byte offset
// at each call site would be noise.
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

// KV cache storage belongs to the backend; the model layer keeps only the
// logical view (docs/KV-CACHE.md). Block ids index one KVStorage and the same
// id addresses every layer of it. Block size and the layout inside a block
// are the backend's choice, which is why nothing here exposes an offset.
struct KVLayout {
    size_t block_tokens;
};

class KVStorage {
public:
    virtual ~KVStorage() = default;
    virtual size_t max_blocks() const = 0;
    // Bytes retained for blocks now, and the most held across successful
    // growths (a growth copies, so old plus new is held for a moment). A
    // growth that failed part way is not counted.
    virtual size_t allocated_bytes() const = 0;
    virtual size_t peak_bytes() const = 0;
};

// One sequence's history in one storage: logical block i is physical block
// blocks[i], and `length` tokens are committed. A batch of nbatch queries
// attends through length + b, so the table must cover length + nbatch.
struct KVView {
    KVStorage* storage;
    const int32_t* blocks;
    size_t n_blocks;
    size_t length;
};

class Backend {
public:
    virtual ~Backend() = default;

    // Set the worker thread count hint (0 = auto / leave as-is).
    virtual void set_threads(int n) = 0;

    // Number of worker threads this backend will actually use (after the last
    // set_threads, or auto-detected). Reported by the CLI; a device backend
    // returns whatever is meaningful for it, or 0.
    virtual int threads_available() const = 0;

    // Invoke once on the caller and complete all cleanup before returning.
    virtual void run_prefill(const std::function<void()>& work) { work(); }

    // Zero-filled backend storage.
    virtual BufferPtr alloc(size_t bytes) = 0;

    // Make `src` reachable by this backend, by whatever means it needs. The
    // name is not "upload": a host backend must not copy, or adopting an 8 GB
    // model would double peak memory for nothing. The contract that allows
    // that is the caller's: **src must outlive the returned buffer**. Model
    // already requires the GGUF model to outlive it, so this is free on CPU,
    // and a backend that copies simply never relies on the guarantee.
    virtual BufferPtr adopt(const void* src, size_t bytes) = 0;

    // Every op below enqueues on this backend's single implicit stream and
    // returns. Results are observable only after sync() or read(), and the
    // model needs host-side data at exactly one point per forward pass, so
    // that is one sync per pass rather than one per op.
    //
    // noexcept by contract, because a caller frees storage on the strength of
    // it: a backend that cannot establish that its outstanding work has
    // finished must fail hard rather than report something nobody at this
    // layer can act on. The CPU backend runs each op to completion as it is
    // called, so this returns immediately.
    virtual void sync() noexcept = 0;

    // Host-visible copy out, for the logits. Syncs first: what it returns has
    // to include every op enqueued before it.
    virtual void read(const Buffer& src, size_t off, void* dst, size_t bytes) = 0;

    // Storage to storage, within this backend. The KV cache grows with it.
    // There is deliberately no host-to-device write: weights arrive through
    // adopt and everything else is produced by an op, so nothing needs one.
    // The first backend that does should add it back with its caller.
    virtual void copy(Buffer& dst, size_t dst_off,
                      const Buffer& src, size_t src_off, size_t bytes) = 0;

    // Y[b*nout + o] = dot(row_o, X + b*nin), for all b in [0,nbatch) and o in
    // [0,nout). X and Y are row-major with nbatch rows.
    // Type-generic: the quant type is looked up in quant::Registry, so every
    // block format gets the batched path, not just Q8_0. `type` is whatever
    // the registry is keyed by, which today is the id GGUF stores; that the
    // numbering is GGML's is a fact about the container format and stays in
    // format/gguf.hpp, where the constants are. Rows are iterated
    // outer and the batch inner so each weight row is read once per block.
    virtual void matmul(uint32_t type, CSlice data, CSlice X,
                        Slice Y, size_t nin, size_t nout, size_t nbatch) = 0;

    // Gather `count` rows of an embedding table into `dst`, row-major, `nin`
    // floats each. This is an op rather than a model-side read because the
    // table is a Buffer: a device backend holds it in its own memory and the
    // model cannot address it.
    virtual void embed(Slice dst, uint32_t type, CSlice table,
                       size_t nin, size_t nrows, const uint32_t* ids,
                       size_t count) = 0;

    // Independent projections of the same X; outputs must not overlap each
    // other, X, or any weights. All outputs are complete on return.
    virtual void matmul_group(std::initializer_list<Projection> projections,
                              CSlice X, size_t nin, size_t nbatch) {
        for (const auto& p : projections) {
            if (!p.data.buffer) throw std::runtime_error("backend: projection without storage");
            matmul(p.type, p.data, X, p.out, nin, p.rows, nbatch);
        }
    }

    virtual KVLayout kv_layout() const = 0;

    // F32 keys and values for `layers` layers of n_head_kv x head_dim, enough
    // whole blocks for max_tokens positions. The backend alone knows what a
    // block costs in bytes; nothing is backed until a block is written.
    virtual std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv,
                                                size_t head_dim,
                                                size_t max_tokens) = 0;

    // Store `batch` token-major [batch, n_head_kv, head_dim] rows at
    // positions pos .. pos+batch of the view's sequence.
    virtual void kv_write(size_t layer, const KVView& view, size_t pos,
                          CSlice k, CSlice v, size_t batch) = 0;

    // Causal GQA over the view: Q/out are [nbatch, n_head, head_dim] and
    // query b attends through view.length + b.
    virtual void attention(CSlice Q, size_t layer, const KVView& view,
                           Slice out, int n_head, int n_head_kv, int head_dim,
                           int nbatch) = 0;

    // dst[i] = src[i] * rsqrt(mean(src^2) + eps) * w[i]  (RMS norm).
    virtual void rms_norm(Slice dst, CSlice src, CSlice w,
                          size_t n, float eps) = 0;

    // The batched forms below exist so the model layer holds no elementwise
    // loops and needs no host parallelism of its own. Each is one call per
    // layer instead of one per row (or per head, per row), which is what makes
    // the graph expressible on a device: see docs/ROADMAP.md #4a.

    // RMS norm of `rows` rows of `n` floats against a shared weight. Row r is
    // at src/dst + r*stride. src and dst may alias only if identical.
    virtual void rms_norm_rows(Slice dst, CSlice src, CSlice w,
                               size_t rows, size_t n, size_t stride, float eps) = 0;

    // Per-head RMS norm followed by RoPE, over a batch of rows. Row r starts
    // at x + r*stride and holds `heads` contiguous heads of `2*half` floats.
    // `cos`/`sin` are the per-position tables, `half` floats per position;
    // row r is at position pos[r] and reads entry pos[r] of each. Positions
    // are per row rather than a base plus r because a batch may carry rows
    // from several sequences (docs/EXECUTION.md). The model always applies
    // norm and RoPE together and per head, so they are one op: the head
    // stays in registers between the two passes, and a device backend gets
    // one kernel launch per layer rather than rows*heads of them.
    virtual void norm_rope_rows(Slice x, size_t rows, size_t stride,
                                size_t heads, CSlice w, float eps,
                                CSlice cos, CSlice sin, size_t half,
                                const uint32_t* pos) = 0;

    // dst[i] = silu(gate[i]) * up[i], the SwiGLU elementwise stage.
    virtual void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n) = 0;

    // dst[i] += src[i], the residual add.
    virtual void add(Slice dst, CSlice src, size_t n) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
