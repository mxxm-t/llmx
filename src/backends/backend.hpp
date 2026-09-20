#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
#include <functional>
#include <initializer_list>

// Compute backend abstraction. The inference graph runs its primitive ops
// (matmul, attention, RMSNorm, RoPE) through a Backend so the same model code
// can target CPU now and ROCm / Vulkan later.
//
// This interface is device-agnostic in shape, but host-pointer based: every
// call takes raw host pointers and returns synchronously, so a backend cannot
// own device memory, keep activations resident, or run async. Adding that is
// the prerequisite for any GPU backend -- see docs/DEVICE-EXECUTION.md, which
// designs the replacement, and docs/ROADMAP.md #4a.
//
// Host parallelism is deliberately NOT on this interface: a callback run
// across host threads has no device implementation. Backends parallelize
// inside their own ops. `threads_available` remains only so the CLI can
// report what a backend is using.
//
// Multi-device / multi-node: the split strategies (per-layer, per-tensor,
// per-row) live at the model layer and are documented in docs/ROADMAP.md; they
// are NOT implemented yet.

namespace backend {

struct Projection {
    uint32_t type;
    const uint8_t* data;
    float* out;
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
    // Bytes retained for blocks now, and the most held at any instant
    // (growth may copy, so the peak can exceed the retained figure).
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
    // set_threads, or auto-detected). Used to size parallelism that happens in
    // the model layer (e.g. attention heads).
    virtual int threads_available() const = 0;

    // Invoke once on the caller and complete all cleanup before returning.
    virtual void run_prefill(const std::function<void()>& work) { work(); }

    // Dot product of one Q8_0 block row (nblocks*32 values) against `x`.
    virtual float dot_q8_0(const uint8_t* row, const float* x, size_t nblocks) = 0;

    // out[o] = dot(row_o, x) for all rows o in [0, nout). Row o starts at
    // data + o * nblocks * Q8_0_TYPESIZE.
    virtual void matvec_q8_0(const uint8_t* data, const float* x, float* out,
                             size_t nblocks, size_t nout) = 0;

    // Y[b*nout + o] = dot(row_o, X + b*nin), for all b in [0,nbatch) and o in
    // [0,nout). X and Y are row-major with nbatch rows.
    // Type-generic: the quant type is looked up in quant::Registry, so every
    // block format gets the batched path, not just Q8_0. Rows are iterated
    // outer and the batch inner so each weight row is read once per block.
    virtual void matmul(uint32_t ggml_type, const uint8_t* data, const float* X,
                        float* Y, size_t nin, size_t nout, size_t nbatch) = 0;

    // Independent projections of the same X; outputs must not overlap each
    // other, X, or any weights. All outputs are complete on return.
    virtual void matmul_group(std::initializer_list<Projection> projections,
                              const float* X, size_t nin, size_t nbatch) {
        for (const auto& p : projections)
            matmul(p.type, p.data, X, p.out, nin, p.rows, nbatch);
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
                          const float* k, const float* v, size_t batch) = 0;

    // Causal GQA over the view: Q/out are [nbatch, n_head, head_dim] and
    // query b attends through view.length + b.
    virtual void attention(const float* Q, size_t layer, const KVView& view,
                           float* out, int n_head, int n_head_kv, int head_dim,
                           int nbatch) = 0;

    // dst[i] = src[i] * rsqrt(mean(src^2) + eps) * w[i]  (RMS norm).
    virtual void rms_norm(float* dst, const float* src, const float* w,
                          size_t n, float eps) = 0;

    // Rotary position embedding on head_dim floats. `cos`/`sin` point at the
    // per-position table (half entries each); the halves of x are interleaved.
    virtual void rope(float* x, const float* cos, const float* sin, int half) = 0;

    // The batched forms below exist so the model layer holds no elementwise
    // loops and needs no host parallelism of its own. Each is one call per
    // layer instead of one per row (or per head, per row), which is what makes
    // the graph expressible on a device: see docs/ROADMAP.md #4a.

    // RMS norm of `rows` rows of `n` floats against a shared weight. Row r is
    // at src/dst + r*stride. src and dst may alias only if identical.
    virtual void rms_norm_rows(float* dst, const float* src, const float* w,
                               size_t rows, size_t n, size_t stride, float eps) = 0;

    // Per-head RMS norm followed by RoPE, over a batch of rows. Row r starts
    // at x + r*stride and holds `heads` contiguous heads of `2*half` floats;
    // it is at position pos0 + r, so it reads cos/sin + r*half. The model
    // always applies these together and per head, so they are one op: the head
    // stays in registers between the two passes, and a device backend gets one
    // kernel launch per layer rather than rows*heads of them.
    virtual void norm_rope_rows(float* x, size_t rows, size_t stride,
                                size_t heads, const float* w, float eps,
                                const float* cos, const float* sin,
                                size_t half) = 0;

    // dst[i] = silu(gate[i]) * up[i], the SwiGLU elementwise stage.
    virtual void silu_mul(float* dst, const float* gate, const float* up,
                          size_t n) = 0;

    // dst[i] += src[i], the residual add.
    virtual void add(float* dst, const float* src, size_t n) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
