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
// the prerequisite for any GPU backend -- see docs/ROADMAP.md #4a.
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

class Backend {
public:
    virtual ~Backend() = default;

    // Set the worker thread count hint (0 = auto / leave as-is).
    virtual void set_threads(int n) = 0;

    // Number of worker threads this backend will actually use (after the last
    // set_threads, or auto-detected). Used to size parallelism that happens in
    // the model layer (e.g. attention heads).
    virtual int threads_available() const = 0;

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

    // Causal GQA: Q/out are [nbatch, n_head, head_dim]. K/V histories
    // are contiguous per head, separated by kv_head_stride floats.
    // Query b attends through n_past+b.
    virtual void attention(const float* Q, const float* K, const float* V, float* out,
                           int n_head, int n_head_kv, int head_dim,
                           int n_past, int nbatch, size_t kv_head_stride) = 0;

    // Run fn(i) for i in [0, n) across the backend's workers. On failure,
    // finish active participants before propagating an exception. Outputs may
    // be partial; this does not roll back an operation or make submission concurrent.
    virtual void parallel_for(int n, const std::function<void(int)>& fn) = 0;

    // dst[i] = src[i] * rsqrt(mean(src^2) + eps) * w[i]  (RMS norm).
    virtual void rms_norm(float* dst, const float* src, const float* w,
                          size_t n, float eps) = 0;

    // Rotary position embedding on head_dim floats. `cos`/`sin` point at the
    // per-position table (half entries each); the halves of x are interleaved.
    virtual void rope(float* x, const float* cos, const float* sin, int half) = 0;
};

using BackendPtr = std::shared_ptr<Backend>;

} // namespace backend
