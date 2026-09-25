# Device execution model

Design for ROADMAP #4a. This is the prerequisite for every vendor backend in
#4b (ROCm, CUDA, SYCL, Vulkan): until it lands, a GPU backend would re-upload
weights and round-trip activations through host memory on every call, which
costs more than it saves. All six steps of the migration order below are done
and gated. What ROADMAP #5 and #7 need on top of this interface is designed in
[EXECUTION](EXECUTION.md), which supersedes the deferrals in the async
section below.

The constraint that shapes the whole design: **the CPU backend must stay the
correctness and performance reference throughout.** It is the A/B baseline for
every GPU claim (#8), so no step may make it slower. Each step below is
independently benchmarkable, and several are CPU wins on their own.

## The problem

This section describes the interface as it stood when the design was written,
and is kept because the migration order below is answering it point by point.
Points 1 and 2 are addressed as of step 4; point 3 is what step 6 is for.

`Backend` was device-agnostic in *shape* but host-pointer based in
*substance*. Three properties made it unimplementable on a device:

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

An opaque handle owned by the backend that allocated it. The signatures below
are the original migration design. The current interface returns
`const void* host_ptr() const` and supports host-visible device allocations;
see [EXECUTION](EXECUTION.md) and `src/backends/backend.hpp`.

```cpp
class Buffer {
public:
    virtual ~Buffer() = default;
    virtual size_t size() const = 0;
    // Non-null only if the allocation is directly addressable by the host.
    // Device backends return nullptr; callers must use read().
    virtual void* host_ptr() = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;
```

On `Backend`:

```cpp
virtual BufferPtr alloc(size_t bytes) = 0;
virtual BufferPtr adopt(const void* src, size_t bytes) = 0;
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
activations are resident. A host-to-device `write` was part of this design,
was dropped when step 5 gave it no caller, and returned with one at
[EXECUTION](EXECUTION.md) step 6: the residual stream crossing to another
device at a placement boundary.

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

The current forward pass builds `"blk." + std::to_string(l) + "."` and does ~10
hash lookups **per layer, per token** - ~280 per decoded token on 28 layers.
Pre-resolution deletes them.

**No admissible measurement yet.** An interleaved A/B showed 25.32 tok/s before
and after, but it ran off-protocol and on a machine another agent's benchmark
owned, so it establishes nothing. Neutrality is expected rather than shown: a
few thousand lookups per second do not compete with matmuls streaming hundreds
of MB per second. This step is a prerequisite, not an optimization - the
backend must receive the same handle for a given weight on every token to keep
it resident. Expect the same of the rest of the migration: CPU-neutral and
GPU-enabling.

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

From writing this step:

- Fuse the per-head norm with RoPE. The model never applies one without the
  other, so `norm_rope_rows` matches the only caller; two orthogonal ops would
  not.
- Dispatch thresholds stay backend-private. The CPU pool declines under two
  rows per worker; a GPU wants every row in one launch at any count. Keeping
  the rule in the backend rather than the model is what allows both.
- Only prefill can benefit on CPU. Decode runs `B == 1`, so there is no row
  dimension to amortize. Treat a decode gain here as an artifact until proven.

Once these exist, `Model` has no elementwise loops left, so it no longer needs
`for_rows`, so `parallel_for` and the compute use of `threads_available` come
off the interface and become private details of `CpuBackend`. The CLI's
reporting use of `threads_available` stays - a GPU backend reports whatever is
meaningful for it, or 0.

### KV cache

Done, and designed separately in [KV-CACHE](KV-CACHE.md). The cache is paged:
`model/kv_cache.hpp` owns the logical side, a block pool and a per-sequence
block table, and the backend owns the physical blocks along with their size
and layout. `Backend::attention` takes a `KVView` rather than raw pointers,
so the model layer never computes an offset into KV storage.

Storage placement followed without changing the interface: the backend's
blocks became a `Buffer` at step 5 below, without changing the view
contract. The release rule already anticipates it - a block returns to the
free list only when its refcount is zero **and** the backend has retired
every submission that read it.

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

**Why deferring them costs no performance today, and where it would.** The
project is performance first, so "minimal" has to be justified on throughput
rather than on taste. In a single-device forward pass there is nothing to
overlap: layer N+1 consumes layer N's output, so the graph is a chain and a
deeper async model has no second thing to run. The one win actually available
is not draining the pipeline once per op, and a single implicit stream takes
all of it. Host work that could overlap - sampling, detokenizing - is
microseconds against matmuls that stream hundreds of MB.

Overlap appears where there are two things to run at once, and both are
roadmap items rather than this one:

- **#5, multi-device.** Pipeline parallelism runs layer N on one device while
  layer N-1 of the next microbatch runs on another; tensor parallelism
  overlaps a reduction with compute. Both need events and cross-stream
  dependencies, and both are worth real throughput.
- **#7, the multi-user server.** Continuous batching overlaps one request's
  prefill with another's decode. That needs a submission queue and completion
  signalling, not just a drain.

The reason to wait is not that the work is small, it is that the abstraction
has to sit over vendor models that differ in structure. HIP streams with
events and Vulkan queues with timeline semaphores and command buffers do not
share a shape, and choosing one before writing either backend is a guess. One
vendor backend tells us what the abstraction must support; #5 then builds it
against two consumers instead of none.

Since this was written, per-row split was dropped from the roadmap, which
closed the requirements enough to design the async extension before the
first vendor backend rather than after: tickets from `submit`/`wait`, with
no events, in [EXECUTION](EXECUTION.md).

What this step must get right is only that it does not foreclose that. Ops
enqueue rather than complete, which is the property a richer model extends
rather than replaces; adding explicit streams later adds a parameter, it does
not undo anything here. Leaving ops synchronous is the choice that would have
to be unpicked, which is why this step exists at all.

## Migration order

Six steps, each mergeable on its own, each keeping the CPU backend correct and
benchmarkable against the floor.

| # | Step | CPU effect |
|---|---|---|
| 1 | Pre-resolve tensors into `LayerWeights` (**done**) | Expected neutral; no admissible measurement |
| 2 | Batched elementwise ops (`rms_norm_rows`, `norm_rope_rows`, `silu_mul`, `add`, `embed`); drop `parallel_for` / `for_rows` (**done**) | Measured; `silu_mul` helps prefill, which reads and writes three `n_ff * B` streams |
| 3 | `Buffer`, `alloc`/`adopt`/`read`/`copy`; weights become buffers; `dot_q8_0` / `matvec_q8_0` leave the interface, `CpuBackend` keeping `matvec_q8_0` as its float Q8_0 decode path (**done**) | Measured neutral over 15 and 9 pairs |
| 4 | Activation arena; op signatures take buffer + offset (**done**) | Measured neutral over 15 and 9 pairs, twice |
| 5 | KV blocks on buffers (the view contract is already in place) (**done**) | Measured neutral over three 15-pair runs and one of 9 |
| 6 | `sync()` and the enqueue contract (**done**) | Neutral - no-op on CPU |

Where to follow along: this table is the plan and `docs/STATUS.md` is the
state. STATUS carries one block per step, newest first, each with what is
done, what is left, the gate numbers and the gotchas found on the way. A
step's block is deleted and its row marked here when it ships.

The bar for each step is therefore **no measured regression**, not a win. Each
must be A/B'd against the previous binary interleaved, never sequentially: on a
loaded workstation, sequential sampling drifts enough to invent a 7% change in
either direction. The CPU backend is the floor for every GPU claim, so a step
that costs throughput is not acceptable even though the destination is a device.

Interleaving is necessary but not sufficient. A step also needs what the
placement study does: `monitor_windows.py` telemetry in every arm, an advance
rule and contamination criteria frozen before measuring, the matched
mx-llama.cpp column, all slow samples retained, and per-pair reporting. A mean
hides two contaminated rounds carrying the whole effect - how step 2's first
decode number went wrong.

A vendor backend (#4b) is only writable after step 6. All six are done, so it
is writable now. What a vendor backend has to supply is the whole `Backend`
interface over its own allocator: `alloc`, `adopt`, `read` and `copy` against
device memory, `kv_alloc` and a `KVStorage` it owns, the ops themselves, and a
`sync()` that means it. What it does not have to do is invent a place for
activations or a cache layout, because the model layer holds no address and
computes no offset into KV storage.
Nor does it write the cache's bookkeeping again: its storage derives from `BlockKVStorage` (`src/backends/kv_storage.hpp`), which grows the blocks through its `alloc` and `copy` and keeps the accounting and the view checks, and it supplies the block size, the kernels, its error prefix and a `retire` that keeps the buffers a growth copied from until its copies retire.

**No step lands without its consumer.** Adding `Buffer` and its five methods
while the model still passes host pointers would be a seam with no caller and
no test, which `AGENTS.md` forbids. Step 3 therefore means buffers *and* the
weights converted to use them, in one change. This was attempted as
plumbing-only and reverted for exactly that reason. The same applies to steps
4-6: if a step cannot be landed with something calling it, it is not ready to
be split off.

## Scope boundary

Explicitly **not** part of #4a:

- Multi-device placement, per-layer/per-tensor/per-row splits - #5.
- Explicit streams, events, cross-device dependencies - #5.
- Graph capture / fusion beyond `silu_mul`.
- A kernel registry - gated on this document landing, per ROADMAP.
- Any vendor SDK code - #4b.

## Open questions

- **`bench` after step 3 (settled).** `bench` now calls `matmul` on an
  adopted buffer, so removing the per-row entries cost it nothing. It is
  still Q8_0-only in what it constructs; making it generic across quants is a
  separate, smaller change.
- **`adopt` alignment (settled).** GGUF tensor offsets are float-aligned.
  The Vulkan backend's `adopt` copies into its own buffers and the device
  accepts 4-byte storage offsets, so `adopt` never has to re-align on any
  backend (`docs/VULKAN.md`, "Adopt copies").
- **Quant dispatch on device.** `quant::Registry` holds host function pointers.
  A device backend needs its own per-type kernel table keyed by the same GGML
  type ids. The registry stays the source of truth for *which* types exist;
  each backend owns *how* it decodes them.
