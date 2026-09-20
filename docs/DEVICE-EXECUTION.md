# Device execution model

Design for ROADMAP #4a. This is the prerequisite for every vendor backend in
#4b (ROCm, CUDA, SYCL, Vulkan): until it lands, a GPU backend would re-upload
weights and round-trip activations through host memory on every call, which
costs more than it saves. Nothing here is implemented yet.

The constraint that shapes the whole design: **the CPU backend must stay the
correctness and performance reference throughout.** It is the A/B baseline for
every GPU claim (#8), so no step may make it slower. Each step below is
independently benchmarkable, and several are CPU wins on their own.

## The problem

`Backend` today is device-agnostic in *shape* but host-pointer based in
*substance*. Three properties make it unimplementable on a device:

1. **Weights are passed as host pointers, per call.** `Model::matvec` and
   `Model::matmul` call `m_->tensor_data(tindex_.at(t.name))` on every
   invocation - every projection, every layer, every token. A device backend
   receiving a host pointer has no way to know it is the same tensor it saw
   last token, so it must copy. Residency is impossible to express.

2. **Activations round-trip through host memory.** The elementwise work in
   `Model::step` - SiLU, the two residual adds, the per-head q/k norms - runs
   in model code on `std::vector<float>`. Between two backend ops, the data is
   on the host by construction. On a device that is a download and an upload
   per layer.

3. **Every op is synchronous and scalar-grained.** `dot_q8_0` returns a `float`
   by value: as a GPU kernel that is one launch and one full pipeline drain per
   row. `rope` is called once per head per token; at `n_head` 32 and a 512-wide
   ubatch that is ~20k virtual calls per layer.

There is a fourth, subtler problem: **two methods on the interface are CPU
worker-pool details that leaked out of the CPU backend.** `parallel_for` takes
a host callback and runs it across host threads - a GPU backend cannot
implement it at all. `threads_available` exists so the model can decide whether
dispatching is worth it. Both are consequences of (2): because elementwise work
lives in model code, the model needs host parallelism to run it. Fix (2) and
both leaks close.

Verified caller counts on the current tree:

| Method | Callers outside `cpu_backend.hpp` |
|---|---|
| `parallel_for` | 1 - the `for_rows` helper in `arch_qwen.hpp` |
| `threads_available` | 1 compute use (`for_rows`); the rest are CLI reporting |
| `dot_q8_0` | 1 test |
| `matvec_q8_0` | 1 - the `bench` CLI command |

`dot_q8_0` / `matvec_q8_0` are pre-`matmul` leftovers: Q8_0-specific, single
row, scalar return. The type-generic `matmul` superseded them everywhere except
a micro-benchmark. They should be removed rather than ported.

## The design

### Buffers

An opaque handle owned by the backend that allocated it.

```cpp
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    // Non-null only if the allocation is directly addressable by the host.
    // Device backends return nullptr; callers must use read()/write().
    virtual void* host_ptr() = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;
```

On `Backend`:

```cpp
virtual BufferPtr alloc(size_t bytes) = 0;
virtual BufferPtr adopt(const void* src, size_t bytes) = 0;
virtual void write(Buffer&, size_t off, const void* src, size_t bytes) = 0;
virtual void read(const Buffer&, size_t off, void* dst, size_t bytes) = 0;
virtual void copy(Buffer& dst, size_t dst_off,
                  const Buffer& src, size_t src_off, size_t bytes) = 0;
```

`adopt` is the load-time entry point for weights, and its name is deliberate.
"Upload" would imply a copy, and the CPU backend must not pay one: model
weights are already resident in the GGUF blob, and copying an 8B model to make
it a `Buffer` would double peak memory for nothing. `adopt` means *the backend
takes responsibility for making this data reachable by its device, by whatever
means it needs*. The CPU backend stores the pointer. A GPU backend does a
staging copy to VRAM.

The contract that makes this work without a capability query: **`src` must
remain valid for the lifetime of the returned `Buffer`.** `Model` already
requires the `GGUFModel` to outlive it, so this is free on CPU; a GPU backend
that copied simply never relies on the guarantee.

`copy` exists for the KV cache, whose writes are device-to-device once
activations are resident.

### Tensor residency

`Model` resolves every tensor **once, at construction**, into per-layer structs:

```cpp
struct DeviceTensor {
    uint32_t type = 0;
    BufferPtr data;
    size_t nin = 0, nout = 0;
};
struct LayerWeights {
    DeviceTensor attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm,
                 attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down;
};
std::vector<LayerWeights> layers_;
```

The current forward pass builds `"blk." + std::to_string(l) + "."` and then
does ~10 string concatenations and ~10 `unordered_map` lookups **per layer, per
token** - ~280 of each per decoded token on a 28-layer model. Pre-resolution
deletes all of it.

**Measured effect on CPU decode: none.** Interleaved A/B on Qwen3-0.6B-Q8_0,
10 pairs, gave mean 25.32 tok/s both before and after (best +0.8%, inside
noise). The lookups are real but they are a few thousand per second against
matmuls streaming hundreds of MB per second; they were never the bottleneck.
This step is justified as a *prerequisite*, not an optimization: it is the only
way to express residency, since the backend must receive the same handle for
`blk.7.ffn_up.weight` on every token to keep it on the device. It also removes
a per-token heap allocation (the string build) and is less code than what it
replaces.

Do not expect the rest of the migration to pay for itself on CPU either. The
honest claim is that it is CPU-neutral and GPU-enabling.

### Activation arena

The nine activation vectors (`xb_`, `hb_`, `qb_`, `kb_`, `vb_`, `attnb_`,
`gateb_`, `upb_`, `ffnb_`) become offsets into **one** `BufferPtr` sized in
`ensure_batch_buffers`. One allocation rather than nine; device allocators
handle a few large allocations far better than many small ones, and it lets
`ubatch` resizing be a single realloc.

Ops therefore take buffer + offset, not a bare pointer:

```cpp
virtual void matmul(uint32_t type,
                    const Buffer& W, size_t w_off,
                    const Buffer& X, size_t x_off,
                    Buffer& Y, size_t y_off,
                    size_t nin, size_t nout, size_t nbatch) = 0;
```

### Elementwise ops move down

For activations to stay resident, everything between two matmuls must be a
backend op. Added, all batched over rows:

```cpp
virtual void embed(Buffer& dst, const DeviceTensor& emb,
                   const uint32_t* ids, size_t n) = 0;
virtual void rms_norm_rows(Buffer& dst, size_t dst_off,
                           const Buffer& src, size_t src_off,
                           const Buffer& w, size_t w_off,
                           size_t rows, size_t n, float eps) = 0;
virtual void rope_rows(Buffer& x, size_t off,
                       const Buffer& cos, const Buffer& sin,
                       size_t rows, size_t heads, size_t half,
                       size_t pos0) = 0;
virtual void silu_mul(Buffer& dst, size_t dst_off,
                      const Buffer& gate, size_t gate_off,
                      const Buffer& up, size_t up_off, size_t n) = 0;
virtual void add(Buffer& dst, size_t dst_off,
                 const Buffer& src, size_t src_off, size_t n) = 0;
```

The `_rows` suffixes matter more than they look. `rope_rows` replaces
`B * (n_head + n_head_kv)` virtual calls per layer with one. `silu_mul` fuses
the SwiGLU loop that currently reads `gate` and `up` and writes `ffn` as three
separate streams. `embed` replaces the per-token host-side `dequant_row`.

Once these exist, `Model` has no elementwise loops left, so it no longer needs
`for_rows`, so `parallel_for` and the compute use of `threads_available` come
off the interface and become private details of `CpuBackend`. The CLI's
reporting use of `threads_available` stays - a GPU backend reports whatever is
meaningful for it, or 0.

### KV cache

`HostKVCache`'s layout arithmetic (head-major, capacity-strided) is
backend-independent and correct; only its storage is host-specific. It becomes
`KVCache` holding a `BufferPtr` per layer, with `write` implemented as
`Backend::copy` from the activation arena. On CPU that is a `memcpy` - the same
work it does today. On a device the K/V never touch host memory.

### Async

The minimal model, and deliberately no more:

- Every op **enqueues** on the backend's single implicit stream and returns.
- `Backend::sync()` blocks until all prior work completes.
- `read()` syncs implicitly.
- Results are only guaranteed observable after a `sync()` or a `read()`.

The CPU backend executes eagerly and `sync()` is a no-op, so the CPU path is
unchanged by construction.

This is sufficient because `Model` needs host-side data at exactly one point
per forward pass: the logits at the end. One sync per forward pass, not one per
op. Explicit streams, events and cross-stream dependencies are deferred to #5
(multi-device), where they are actually needed - building them now would be
speculative machinery with no consumer.

## Migration order

Six steps, each mergeable on its own, each keeping the CPU backend correct and
benchmarkable against the floor.

| # | Step | CPU effect |
|---|---|---|
| 1 | Pre-resolve tensors into `LayerWeights` | **Measured neutral** (mean 25.32 -> 25.32 tok/s, 10 interleaved pairs) |
| 2 | Batched elementwise ops (`rms_norm_rows`, `rope_rows`, `silu_mul`, `add`, `embed`); drop `parallel_for` / `for_rows` | Expected neutral on decode; `silu_mul` may help prefill, which reads and writes three `n_ff * B` streams |
| 3 | `Buffer`, `alloc`/`adopt`/`read`/`write`/`copy`; weights become buffers; delete `dot_q8_0` / `matvec_q8_0` | Neutral - CPU buffers wrap host memory, zero-copy |
| 4 | Activation arena; op signatures take buffer + offset | Neutral to slight win (one allocation, better locality) |
| 5 | KV cache on buffers via `copy` | Neutral - same `memcpy` |
| 6 | `sync()` and the enqueue contract | Neutral - no-op on CPU |

The bar for each step is therefore **no measured regression**, not a win. Each
must be A/B'd against the previous binary interleaved, never sequentially: on a
loaded workstation, sequential sampling drifts enough to invent a 7% change in
either direction. The CPU backend is the floor for every GPU claim, so a step
that costs throughput is not acceptable even though the destination is a device.

A vendor backend (#4b) is only writable after step 6.

## Scope boundary

Explicitly **not** part of #4a:

- Multi-device placement, per-layer/per-tensor/per-row splits - #5.
- Explicit streams, events, cross-device dependencies - #5.
- Graph capture / fusion beyond `silu_mul`.
- A kernel registry - gated on this document landing, per ROADMAP.
- Any vendor SDK code - #4b.

## Open questions

- **`bench` after step 3.** Removing `matvec_q8_0` removes what the `bench`
  command measures. It should be reframed onto `matmul` with a type argument,
  which also makes it generic across quants instead of Q8_0-only - consistent
  with the type-generic direction the rest of the code already took.
- **`adopt` alignment.** GGUF tensor offsets are float-aligned. Device
  backends may want stricter alignment for coalesced access; whether `adopt`
  may re-align (and therefore must copy on every backend) is unresolved.
  Deferring: the first vendor backend has the information to decide.
- **Quant dispatch on device.** `quant::Registry` holds host function pointers.
  A device backend needs its own per-type kernel table keyed by the same GGML
  type ids. The registry stays the source of truth for *which* types exist;
  each backend owns *how* it decodes them.
