# Execution model: batches, tickets and placement

Design for what ROADMAP #5 (multi-device) and #7 (multi-user server) need
from the backend interface and the model layer, written once both were
scoped. It succeeds [DEVICE-EXECUTION](DEVICE-EXECUTION.md), which is
complete and stays as the record of how the interface got to where it is.
Nothing here undoes that work; every change is an extension of a signature
that already exists.

## Requirements

Fixed on 2026-09-21, and the reason this can be designed now rather than
guessed:

1. **Vendor backends.** Vulkan on this workstation's Radeon VII (gfx906) and
   ROCm on the Linux machine's MI50 (the same silicon). Both run the same
   graph; neither shares memory with the other.
2. **Per-layer split.** Consecutive layers on different devices, so a model
   that does not fit one device runs across two, or CPU takes the layers a
   GPU has no room for.
3. **Per-tensor placement.** The embedding table and the output head placed
   independently of the layers. The practical case is llama.cpp's default:
   a large embedding table stays in host memory because a gather is cheap
   and the table is not.
4. **Continuous batching.** One forward pass carrying tokens from several
   sequences at different positions, with prefill and decode mixed.
5. **Overlap where it pays.** The device must not idle while the host
   samples, and a second device must not wait for the first at every
   microbatch.
6. **Architectures this project does not run yet.** Hybrid compressed
   attention, mixture of experts, lookup-table memory and residual mixing
   are current designs (DeepSeek V4 and V4.1 carry all four), and the
   interface must not bake in the dense-Qwen assumptions that would have to
   be undone for them. The section "Beyond dense Qwen" lists which
   assumptions those are; the steps below are written to avoid them.

**Per-row (tensor-parallel) split is dropped from the roadmap.** It moves
data between devices at every projection of every layer, so it only pays
with peer-to-peer copies and cross-device events, and it was the one item
forcing that machinery into the interface. The workloads here are one or two
devices, where a layer split under continuous batching keeps both busy. If
it ever returns it is additive: a reduction op plus an event type, on top of
everything below.

## What holds and what changes

The single implicit stream per backend holds. A GPU saturates through batch
size, not through concurrent callers, so continuous batching on one stream
is the throughput design, and every requirement above fits it. A `Backend`
is driven by **one thread at a time**; the server's scheduler is the single
submitter per device and request threads only queue work. This is a
contract, not a lock.

Eager per-op enqueue holds. A Vulkan backend records every op between two
submit points into one command buffer, so a forward pass is one submission;
replaying an identical decode pass is a backend-private optimization the
interface neither needs nor prevents.

`sync()` stays `noexcept` and a lost device still fails the process. A
server that loses its device cannot complete any request on it; the
supervisor restarts the process. Recovery inside the process is not
designed.

Five things change. Each is small on its own and they are ordered below so
each lands with its consumer.

### 1. Tickets

```cpp
using Ticket = uint64_t;
virtual Ticket submit() = 0;            // flush everything enqueued; monotonic
virtual void wait(Ticket t) noexcept = 0;
virtual void sync() noexcept = 0;       // wait for everything, as today
```

`submit` hands the enqueued work to the device and returns a ticket that
`wait` blocks on. `sync` is unchanged and remains what the error paths call.
`read` stays synchronous: it is the transfer and test path, not the hot one.

Tickets are what make overlap expressible without events. Two things are
in flight at once only ever on **different devices or different execution
contexts**, and each has its own ticket, so the scheduler waits for exactly
the one it needs. The KV release rule in [KV-CACHE](KV-CACHE.md) gains
precision at the same time: a block returns to the pool after the ticket of
the last pass that touched it has retired, which is a `wait`, not a drain.

On the CPU backend `submit` returns a counter and `wait` returns
immediately, so the CPU path is unchanged by construction.

### 2. Host-visible memory and `write`

```cpp
enum class Memory { device, host_visible };
virtual BufferPtr alloc(size_t bytes, Memory where = Memory::device) = 0;
virtual void write(Buffer& dst, size_t off, const void* src, size_t bytes) = 0;
```

The logits buffer becomes host-visible: the output projection writes into
memory the host can read after `wait`, through `host_ptr()`, with no copy
op. On CPU everything is host-visible already. On Vulkan it is a
host-coherent allocation the last kernel writes to, which for a logits row
is the right trade; on HIP it is pinned host memory.

`write` returns to the interface with its first caller, the transfer at a
placement boundary (below). It is enqueued like every op and consumes
`src` before returning, so the caller's staging vector may be reused.

### 3. Rows carry positions

```cpp
virtual void norm_rope_rows(Slice x, size_t rows, size_t stride, size_t heads,
                            CSlice w, float eps, CSlice cos, CSlice sin,
                            size_t half, const uint32_t* pos) = 0;
```

Today row `r` is at position `pos0 + r`, which is true only while every row
belongs to one sequence. With sequences mixed in a batch the position is
per row, so the op takes an array of `rows` positions, like `embed` takes
its ids. The cos/sin table becomes a buffer the model adopts from its host
table, which is what lets a device read it. `Backend::rope`, whose only
caller is `bench`, is deleted; `bench` measures `norm_rope_rows`, the op
the runtime runs.

### 4. Batched views

```cpp
struct KVView {
    KVStorage* storage;
    const int32_t* blocks;
    size_t n_blocks;
    size_t length;   // committed history
    size_t nq;       // rows of this pass that belong to this sequence
};
virtual void kv_write(size_t layer, const KVView* views, size_t n_views,
                      CSlice k, CSlice v) = 0;
virtual void attention(CSlice Q, size_t layer, const KVView* views,
                       size_t n_views, Slice out,
                       int n_head, int n_head_kv, int head_dim) = 0;
```

Rows are laid out in view order: view `v` owns the next `nq` rows, its
row `b` is at position `length + b` and attends through `length + b`. The
current signatures are the case `n_views == 1`, and `pos` and `batch` are
gone because both follow from the view. The CPU implementation is a loop
over views around the code it has now; a device backend gets all sequences
in one launch, which is what continuous batching needs from it.

`length` and the block table are counted in the **entries of that
storage**, which for the dense cache is tokens. A storage whose entry
stands for several tokens (compressed attention) has a shorter table with
the same contract. Nothing in the view says what an entry contains; the
storage that was allocated does, and `kv_alloc` describes an entry by its
key and value widths rather than by a head count and a head dimension, so
a latent-attention row (one wide key, a narrower value, no heads) is the
same call with different numbers.

Logits are wanted for every decode row but only the last row of a prefill
entry. A batch mixing both selects rows that are not contiguous, so one op
compacts them before the output norm and head:

```cpp
virtual void gather_rows(Slice dst, CSlice src, size_t width,
                         const uint32_t* rows, size_t count) = 0;
```

It lands with step 4 rather than with the views: until a `Batch` can carry
two entries, the only caller would be a prefill selecting its last row,
which a slice offset already does for free, and an op with a contrived
caller is the seam AGENTS.md forbids. The `kv-cache` test is what gives
the views their second consumer today, two sequences in one call against
the same two taken separately.

### 5. The model layer splits into three

`Model` today is weights, one sequence, activation scratch and a backend.
The server needs the first shared and the rest per request and per pass:

```
Model         config, placement, weights per device, RoPE tables per device,
              the backends. Read-only after construction; shared.
Sequence      one request's history: a KVSequence per storage, the
              committed length, the ticket of its last pass. A device may
              host several storages (one per attention kind), which is why
              the table is per storage and not per device.
ExecContext   one pass in flight: an activation arena per device, the
              host-visible logits buffer, a host staging vector for
              transfers, and its tickets.
Batch         entries of (Sequence*, token ids, want_logits).
```

`Model::forward(ExecContext&, const BatchEntry*, n)` enqueues the whole pass on
every device in layer order, submits, and returns. `ExecContext::logits()`
waits on the tickets and returns the rows in entry order. A prefill
microbatch is one entry with many tokens; a decode batch is many entries
with one token each; the two mix freely.

The CLI keeps one `Sequence` and one `ExecContext`, and `step` and
`prefill` become wrappers over `forward`, so `generate`, `chat` and
`perplexity` do not change. Two contexts are what the server uses to keep
a device busy: a prefill of new requests in one while the host samples the
decode batch of the other. The CPU cost per step in this runtime is small
against a device pass, so that gain is measured before it is claimed.

### Placement

Placement is a device per **tensor role**, not per layer: a layer's
attention, its feed-forward block, the embedding table, the output head.
The roles a dense model has are few, and the struct starts with those:

```cpp
struct Placement {
    std::vector<int> attn_device;   // per layer
    std::vector<int> ffn_device;    // per layer; the routed experts of an MoE layer
    int embed_device;
    int output_device;
};
```

Splitting attention from feed-forward at the outset is what expert offload
needs later: on a mixture-of-experts model the routed experts are most of
the parameters while few are active per token, so attention stays on the
device and the experts run on the CPU from the mapped file, and the
placement that expresses it is the same struct with a different value.

Each device hosts the storages for the layers placed on it, with its own
block size and its own pool; a `Sequence` therefore holds one block table
per storage and prepares, commits and aborts them together. Weights are
adopted by the backend that hosts them, which on the CPU is the mapped
GGUF bytes and costs no RAM.

Wherever two consecutive graph nodes sit on different devices the residual
stream crosses: `read` from the source into the context's staging vector,
`write` into the destination. With a layer split that is one crossing per
pass, `n_embd * rows` floats: 16 KiB for a decode token on Qwen3-8B, 8 MiB
for a 512-token microbatch, against matmuls that stream gigabytes. A direct
device-to-device copy is a faster implementation of the same two calls and
is not needed to make the split correct.

The pipeline overlap that makes a two-device split worth its transfer is a
loop in the caller, not a feature of the interface: submit microbatch `i+1`
on the first device before waiting for microbatch `i` on the second. The
tickets express it and the CLI's prefill loop can use it once two devices
exist.

Flags follow llama.cpp's names so the vocabulary carries over: `--device`
selects the backend (`cpu`, `vulkan:0`, `rocm:0`), `--n-gpu-layers N` puts
the last `N` layers on it and the rest on CPU, and the embedding table
stays on CPU unless every layer is on the device. `--tensor-split` waits
for a second device to exist. `--n-cpu-moe N` and `--cpu-moe` exist: the
experts of the first `N` routed layers, or all, on the CPU beside a device
(`docs/USAGE.md`); `--n-gpu-layers` and `--tensor-split` do not yet. Choosing a fit automatically needs each backend to
report its free memory; that query is added with the first device backend
that can answer it.

Two implementation notes for the crossing itself. A `read` whose
destination is host-addressable lands directly in it, so a device-to-CPU
crossing is one copy. And what the design does not do is stream weights
into the device per token: moving an expert's bytes across the bus every
token is slower than running it where it is.

## Order of work

Each step is mergeable on its own, keeps the CPU backend as the reference,
and is gated the same way as every device execution step: interleaved A/B
against the previous binary with a same-file layout control, HF fixtures
unchanged. Steps 1 to 4 change the interface and go **before** the first
vendor backend, so Vulkan implements each signature once.

| # | Step | Consumer that lands with it | CPU effect |
|---|---|---|---|
| 1 | RoPE table as a buffer; per-row positions; delete `rope` (**done**) | `Model` adopts its table and passes positions | Measured neutral over five 0.6B cells and one 8B, with a same-file layout control |
| 2 | `submit`/`wait`; `Memory::host_visible`; logits read through `host_ptr` (**done**) | `Model::step` waits a ticket instead of `read`; reset waits the last ticket | Measured neutral over three 0.6B cells and one 8B, with a same-file layout control |
| 3 | Batched views (**done**) | `Model` passes one view; `kv-cache` batches two sequences in one call | Measured neutral over three 0.6B cells and one 8B, with a same-file layout control |
| 4 | `Model` / `Sequence` / `ExecContext` / `Batch`; `gather_rows` (**done**) | The CLI as one sequence and one context; `kv-cache` runs `forward` with two entries | Measured neutral over three 0.6B cells and one 8B, decode positive in all four |
| 5 | Vulkan backend (#4b) (**done** on the Radeon VII: every CPU quant type, tiled prefill attention, f16 cache sides, decode at or above the reference on all four measured files; on the MI50 the one-card gap is the open item, `docs/STATUS.md`) | [VULKAN](VULKAN.md) | CPU-vs-Vulkan A/B on identical inputs; the HF gate with `--device vulkan:0`; the matched floor under the reference bench tool's protocol, `llmx bench --model` |
| 6 | `Placement`, `write`, transfers (**done**, taken before 5); the placement flags are not added yet | Two `CpuBackend` instances splitting roles inside layers, bit-identical to one; then CPU plus Vulkan | Measured neutral at a single-device placement over three 0.6B cells and one 8B |
| 7 | Scheduler, per-request sequences, prefix reuse through donors (#7) (**done**, `docs/SERVER.md`) | The server | Measured against single-sequence decode |

Step 6's first test needs no GPU: two CPU backends with different thread
counts, one holding layers 0 to k and the other the rest, must produce the
same bytes as one backend, because per-layer arithmetic is unchanged and
only the residual stream crosses. That checks every piece of the plumbing
before a device is involved.

## Beyond dense Qwen

The models this project will be asked to run next do four things dense
Qwen does not, and each is an addition on top of the interface above
rather than a change to it, provided the steps do not assume otherwise.
The assumptions to avoid are marked.

- **Hybrid compressed attention** (DeepSeek V4: layers that pool every 4
  or 128 tokens into one entry, a lightning indexer choosing the top
  entries per query, a 128-token sliding window over raw tokens; V4.1
  shares one layer's entries and indices across following layers). Needs:
  several storages per model with different entries per token, which the
  per-storage table above allows; a per-sequence **state** for the window
  and for tokens not yet forming a full group, which is fixed-size,
  private and overwritten, so it is not paged history and is copied on
  fork rather than shared; a compress op, an indexer plus top-k op, and an
  attention variant taking a selected index list. Cross-layer reuse is a
  view naming another layer's entries, which the view can already do. Do
  not assume one entry per token, one storage per device, or that a
  sequence holds only refcounted blocks.
- **Mixture of experts.** A routed matmul over the experts each token
  selected, and expert placement, which the per-role placement above
  carries. Do not assume the feed-forward block is on the layer's device.
  Implemented for `qwen3moe`: `route_experts`, `matmul_experts` and
  `matmul_experts_add` in `backend.hpp`, and `--n-cpu-moe` for placement.
- **Lookup-table memory** (Engram: hashed n-gram tables, 196B parameters
  on V4.1-Flash). An embed-like gather over a table that on any hardware
  here lives in host memory; per-tensor placement is what puts it there.
- **Residual mixing** (mHC: the residual add becomes a per-token mix over
  several parallel streams). The residual stream is wider and the `add` op
  becomes a mixing op. Do not assume the arena holds one residual row of
  `n_embd` per token.
- **Low-precision cache rows.** The storage contract already leaves dtype
  to the backend; the CPU cache implements F32 only, and FP8 or FP4 rows
  with per-group scales are a kernel addition with its own HF gate.

The correctness gate for any of these is the HF reference, and the models
themselves exceed this hardware by an order of magnitude. The gate is
therefore a tiny random-weight model of the real architecture generated
through HF modeling code, which is how `tests/f32.py` already works, and
a small released member of the family when one exists.

## Not chosen

- **A graph.** The model describing the pass as a graph the backend fuses,
  plans and replays is what ggml and the compilers do, and it buys automatic
  replay on decode. It also brings a graph compiler into a dependency-free
  project with one architecture. Eager enqueue with hand-fused ops is kept;
  a backend that wants replay can hash the op sequence between two submits.
  If the architecture count grows, a graph sits on top of this interface,
  it does not replace it.
- **Per-request streams.** Concurrency across callers instead of across
  rows. Batching beats it on every device this project targets.
- **Events and cross-device dependencies.** Not needed once per-row split
  is gone; tickets plus a scheduler loop express every overlap above.
- **Recovery from device loss.** Out of scope; the process exits.

## Open questions

- **`adopt` alignment (carried from DEVICE-EXECUTION).** GGUF tensor
  offsets are float-aligned. Whether a device backend re-aligns on adopt is
  the Vulkan backend's decision and is recorded on its page.
- **Quant dispatch on device (carried).** `quant::Registry` names the types;
  each backend owns its per-type kernel table keyed by the same ids.
- **Block size per device.** The CPU block is 128 tokens by measurement. A
  device backend chooses its own, and `Sequence` holding one table per
  device is what allows them to differ; whether a shared prefix index can
  span two block sizes is a #7 question.
