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

Logits are wanted for every decode row but only the last row of a prefill
entry. A batch mixing both selects rows that are not contiguous, so one op
compacts them before the output norm and head:

```cpp
virtual void gather_rows(Slice dst, CSlice src, size_t width,
                         const uint32_t* rows, size_t count) = 0;
```

### 5. The model layer splits into three

`Model` today is weights, one sequence, activation scratch and a backend.
The server needs the first shared and the rest per request and per pass:

```
Model         config, placement, weights per device, RoPE tables per device,
              the backends. Read-only after construction; shared.
Sequence      one request's history: a KVSequence per device that holds
              layers, the committed length, the ticket of its last pass.
ExecContext   one pass in flight: an activation arena per device, the
              host-visible logits buffer, a host staging vector for
              transfers, and its tickets.
Batch         entries of (Sequence*, token ids, want_logits).
```

`Model::forward(ExecContext&, const Batch&)` enqueues the whole pass on
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

```cpp
struct Placement {
    std::vector<int> layer_device;   // device index per layer
    int embed_device;
    int output_device;
};
```

Each device hosts the layers placed on it, in its own `KVStorage`, with its
own block size and its own pool; a `Sequence` therefore holds one block
table per device and prepares, commits and aborts them together. Weights
are adopted by the backend that hosts them.

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
for a second device to exist. The flags land with the placement, in
`docs/USAGE.md` and `print_usage` together.

## Order of work

Each step is mergeable on its own, keeps the CPU backend as the reference,
and is gated the same way as every device execution step: interleaved A/B
against the previous binary with a same-file layout control, HF fixtures
unchanged. Steps 1 to 4 change the interface and go **before** the first
vendor backend, so Vulkan implements each signature once.

| # | Step | Consumer that lands with it | CPU effect |
|---|---|---|---|
| 1 | RoPE table as a buffer; per-row positions; delete `rope` | `Model` adopts its table and passes positions | Gated; touches the decode path |
| 2 | `submit`/`wait`; `Memory::host_visible`; logits read through `host_ptr` | `Model::step` waits a ticket instead of `read`; KV release waits the sequence's ticket | Neutral by construction |
| 3 | Batched views; `gather_rows` | `Model` passes one view; prefill selects its last row through the gather | Gated |
| 4 | `Model` / `Sequence` / `ExecContext` / `Batch` | The CLI as one sequence and one context; `kv-cache` and `prefill-scope` tests over `forward` | Gated |
| 5 | Vulkan backend (#4b) | Its own design page | CPU-vs-Vulkan A/B on identical inputs |
| 6 | `Placement`, `write`, transfers, flags | Two `CpuBackend` instances splitting layers, bit-identical to one; then CPU plus Vulkan | Gated at a single-device placement |
| 7 | Scheduler, per-request sequences, prefix index (#7) | The server | Measured against single-sequence decode |

Step 6's first test needs no GPU: two CPU backends with different thread
counts, one holding layers 0 to k and the other the rest, must produce the
same bytes as one backend, because per-layer arithmetic is unchanged and
only the residual stream crosses. That checks every piece of the plumbing
before a device is involved.

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
