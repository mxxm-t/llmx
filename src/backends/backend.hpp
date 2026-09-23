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
    // Non-null where the host can address the allocation directly: always
    // for Memory::host_visible, and for everything on a host backend. What
    // it points at is current only after wait() or sync().
    virtual const void* host_ptr() const = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

// Where an allocation lives. `device` is the default and may be unreachable
// from the host; `host_visible` is memory an op can write and the host can
// read through host_ptr() after a wait, which is how the logits leave the
// backend without a copy op. On a host backend the two are the same memory.
enum class Memory { device, host_visible };

// A submission. `submit()` hands everything enqueued so far to the device
// and returns one; `wait()` blocks until that submission has retired.
using Ticket = uint64_t;

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

// The rows of a matmul grouped by the prompt they belong to. A device whose kernels differ with the width of a batch picks a row's kernel by `extent` rather than by the call's width, so a prompt computes the same whether its rows arrive in one pass or several, alone or beside other sequences' rows, and a server reusing a cached prefix produces what one pass over the whole prompt does.
// `end` is one past the run's last row, runs in row order; `extent` is the position one past the prompt's last token for prompt rows, and 1 for a generated token. Without runs a backend chooses by the call's width.
struct RowRun {
    size_t end;
    size_t extent;
};
struct RowRuns {
    const RowRun* runs = nullptr;
    size_t n = 0;
};

// KV cache storage belongs to the backend; the model layer keeps only the
// logical view (docs/KV-CACHE.md). Block ids index one KVStorage and the same
// id addresses every layer of it. Block size and the layout inside a block
// are the backend's choice, which is why nothing here exposes an offset.
struct KVLayout {
    size_t block_tokens;
};

// How a cache side is stored; the CLI's --cache-type-k and --cache-type-v.
enum class KVType { f32, f16 };
inline size_t kv_elem_bytes(KVType t) { return t == KVType::f16 ? 2 : 4; }

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
// blocks[i], `length` entries are committed, and `nq` rows of the current
// pass belong to this sequence. Row b of those is at position length + b
// and attends through it, so the table must cover length + nq. Entries are
// whatever the storage was allocated to hold, tokens for the dense cache
// (docs/EXECUTION.md).
// `extent` is what RowRun's is for these rows, 0 when unknown, and a device choosing an attention kernel by it computes a row the same way whatever shares its pass.
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

    // Set the worker thread count hint (0 = auto / leave as-is).
    virtual void set_threads(int n) = 0;

    // Number of worker threads this backend will actually use (after the last
    // set_threads, or auto-detected). Reported by the CLI; a device backend
    // returns whatever is meaningful for it, or 0.
    virtual int threads_available() const = 0;

    // Invoke once on the caller and complete all cleanup before returning.
    virtual void run_prefill(const std::function<void()>& work) { work(); }

    // Zero-filled backend storage.
    virtual BufferPtr alloc(size_t bytes, Memory where = Memory::device) = 0;

    // Make `src` reachable by this backend, by whatever means it needs. The
    // name is not "upload": a host backend must not copy, or adopting an 8 GB
    // model would double peak memory for nothing. The contract that allows
    // that is the caller's: **src must outlive the returned buffer**. Model
    // already requires the GGUF model to outlive it, so this is free on CPU,
    // and a backend that copies simply never relies on the guarantee.
    virtual BufferPtr adopt(const void* src, size_t bytes) = 0;

    // Every op below enqueues on this backend's single implicit stream and
    // returns. submit() flushes what has been enqueued and returns a ticket
    // that is monotonic within this backend; wait(t) blocks until that
    // submission and everything before it has retired. The model submits
    // once per forward pass and waits on that ticket for the logits, so a
    // pass is one submission rather than one per op, and a caller with two
    // things in flight (docs/EXECUTION.md) waits for exactly the one it
    // needs. Results are observable only after wait(), sync() or read().
    //
    // sync() waits for everything, including work enqueued after the last
    // submit, which is what an error path needs: a failed pass has ops
    // queued behind no ticket. wait() and sync() are noexcept by contract,
    // because a caller frees storage on the strength of them: a backend
    // that cannot establish that its work has finished must fail hard
    // rather than report something nobody at this layer can act on. The
    // CPU backend runs each op to completion as it is called, so both
    // return immediately and submit only counts.
    virtual Ticket submit() = 0;
    virtual void wait(Ticket t) noexcept = 0;
    virtual void sync() noexcept = 0;

    // Copy out to host memory. Syncs first: what it returns has to include
    // every op enqueued before it. This is the transfer and test path; the
    // logits leave through a host_visible buffer and a wait instead.
    virtual void read(const Buffer& src, size_t off, void* dst, size_t bytes) = 0;

    // Storage to storage, within this backend. The KV cache grows with it.
    virtual void copy(Buffer& dst, size_t dst_off,
                      const Buffer& src, size_t src_off, size_t bytes) = 0;

    // Host to storage. Enqueued like every op; the caller's bytes are
    // consumed before this returns, so a staging buffer can be reused at
    // once. Its caller is the residual stream crossing to another device at
    // a placement boundary (docs/EXECUTION.md): weights arrive through
    // adopt and every other value is produced by an op, so nothing else
    // needs one.
    virtual void write(Buffer& dst, size_t off, const void* src, size_t bytes) = 0;

    // Y[b*nout + o] = dot(row_o, X + b*nin), for all b in [0,nbatch) and o in
    // [0,nout). X and Y are row-major with nbatch rows.
    // Type-generic: the quant type is looked up in quant::Registry, so every
    // block format gets the batched path, not just Q8_0. `type` is whatever
    // the registry is keyed by, which today is the id GGUF stores; that the
    // numbering is GGML's is a fact about the container format and stays in
    // format/gguf.hpp, where the constants are. Rows are iterated
    // outer and the batch inner so each weight row is read once per block.
    virtual void matmul(uint32_t type, CSlice data, CSlice X,
                        Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}) = 0;

    // Y += W X, the projection whose output joins the residual stream: the
    // model asks for the sum and each backend produces it its own way. The
    // CPU computes the product into scratch and adds; a device folds the
    // add into the matmul's store, one dispatch fewer per projection.
    virtual void matmul_add(uint32_t type, CSlice data, CSlice X,
                            Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs = {}) = 0;

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
                              CSlice X, size_t nin, size_t nbatch, RowRuns runs = {}) {
        for (const auto& p : projections) {
            if (!p.data.buffer) throw std::runtime_error("backend: projection without storage");
            matmul(p.type, p.data, X, p.out, nin, p.rows, nbatch, runs);
        }
    }

    virtual KVLayout kv_layout() const = 0;

    // Keys and values for `layers` layers of n_head_kv x head_dim, enough
    // whole blocks for max_tokens positions, each side stored as k_type or
    // v_type: f32, or f16 written with round-to-nearest and read back
    // exactly as stored, so the arithmetic against the cache is the same on
    // every backend and only the stored precision differs. The two are
    // separate because K feeds every score and V is averaged under the
    // softmax, so V tolerates less precision first. The backend alone knows
    // what a block costs in bytes; nothing is backed until a block is
    // written. A backend without a type throws rather than substituting.
    virtual std::unique_ptr<KVStorage> kv_alloc(size_t layers, size_t n_head_kv,
                                                size_t head_dim, size_t max_tokens,
                                                KVType k_type = KVType::f32,
                                                KVType v_type = KVType::f32) = 0;

    // Every layer's K and V of block `src` into block `dst` of the same
    // storage, enqueued. A fork's private tail is filled this way from the
    // block it shares up to; the backend owns the layout, so only it can
    // copy a block.
    virtual void kv_copy(KVStorage& storage, int32_t src, int32_t dst) = 0;

    // Store token-major [rows, n_head_kv, head_dim] rows, laid out in view
    // order: view v owns the next views[v].nq rows and they go to positions
    // length .. length + nq of its sequence. Several views in one call is
    // what a batch carrying rows from several sequences needs, and one view
    // is the case the model passes today.
    virtual void kv_write(size_t layer, const KVView* views, size_t n_views,
                          CSlice k, CSlice v) = 0;

    // Causal GQA: Q/out are [rows, n_head, head_dim] in the same view order,
    // and row b of view v attends through views[v].length + b. A device
    // backend gets every sequence of the pass in one launch.
    virtual void attention(CSlice Q, size_t layer, const KVView* views,
                           size_t n_views, Slice out,
                           int n_head, int n_head_kv, int head_dim) = 0;

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

    // A layer's attention inputs together: q normed and rotated in place,
    // k normed and rotated and written with v into the views' KV blocks,
    // rows laid out as norm_rope_rows and kv_write take them. The model
    // always does these three things back to back, so it asks for them as
    // one op; this default is the three, and a device backend makes one
    // kernel of them. After it k holds its normed and rotated rows too.
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
    virtual void silu_mul(Slice dst, CSlice gate, CSlice up, size_t n) = 0;

    // dst[i] += src[i], the residual add.
    virtual void add(Slice dst, CSlice src, size_t n) = 0;

    // dst row i = src row rows[i], `width` floats each. Compacts the rows of
    // a pass that want logits, which a batch mixing prefill and decode
    // entries leaves non-contiguous, so the output head runs once over
    // exactly those rows.
    virtual void gather_rows(Slice dst, CSlice src, size_t width,
                             const uint32_t* rows, size_t count) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
