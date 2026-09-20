# Paged KV cache

Design for the KV cache that the multi-user server (ROADMAP #7) and the device
execution model (ROADMAP #4a) both need. Status: direction agreed by both
developers on 2026-09-20; the contract conditions XDEV set are recorded in
their sections below. Step 1 is implemented on the design branch and the
block-size screening below fixed the CPU block at 128 tokens.

## Why change

Before this design, `HostKVCache` held one sequence per model as a
contiguous, head-major, capacity-strided F32 array per layer, growing by
doubling and copying. That is correct and fast for one chat. Serving many
sequences from it would cost more than it should:

- Each sequence needs its own capacity reservation and its own growth copies,
  and the reservations fragment as sequences come and go.
- A shared prompt prefix is stored once per sequence, or copied on fork.
- The layout is a CPU decision that the attention op receives as raw pointers
  plus a stride, so a GPU backend would inherit it.

Contiguous caches can serve; the point is that paging makes the sharing and
reuse cases cheap by construction, at an indirection cost that is measured
below rather than assumed.

vLLM answers all three with paging: uniform blocks, a per-sequence block
table, refcounts for sharing. llama.cpp keeps a flat cell array with a
sequence set per cell and defragments it. Paging is the structure chosen
here because it makes sharing and reuse a refcount. The open question was its
cost on CPU, where decode attention is memory bound and about a fifth of a
token.

## Measured before deciding

`tools/paged_attn_bench.cpp` (build: `cl /O2 /arch:AVX2 /EHsc`) runs decode
attention for one layer over a pool of KV sets larger than cache and with
shuffled block tables, so it measures cold memory and real indirection rather
than a sequentially allocated best case. Seven paired repeats per cell, one
thread, 2026-09-20 on the Ryzen 7 5800X; each cell is the median paired
ratio of paged over contiguous ms/step. Raw output, commands, source and
binary hashes and the activity monitor are in
`docs/benchmarks/kv-paging-20260920/`. System CPU during the run averaged
12.7% of 16 logical CPUs (one busy thread is 6.25%), so other activity was
present and every sample is kept.

| Block tokens | 0.6B shape (16/8 heads) n_past 256 / 840 / 2048 | 8B shape (32/8 heads) 256 / 840 / 2048 |
|---|---|---|
| 16 | +29.2% / +24.6% / +32.6% | +25.5% / +25.9% / +29.4% |
| 32 | +15.3% / +15.4% / +17.2% | +16.4% / +13.3% / +13.4% |
| 64 | +9.0% / +6.1% / +11.5% | +7.6% / +8.3% / +10.2% |
| 128 | +9.8% / +6.0% / +6.5% | +4.7% / +5.2% / +6.2% |
| 256 | +0.9% / +3.3% / +1.5% | +3.2% / +2.6% / +2.5% |

Contiguous baselines run 0.11 to 1.9 ms/step across the cells. Per-repeat
spread at 64 and 128 reaches 20 points, so those two are not separated from
each other; 16 and 32 are separated from everything, and 256 is the cheapest
in every cell.

vLLM's default of 16 is a GPU answer. On CPU the block must be large enough
that per-block setup and the loss of hardware prefetch across block edges are
amortized over the vectorized inner loop. 64, 128 and 256 go to the
real-model screening; 16 and 32 do not. This is isolated attention with a
cold cache, not an end-to-end decode cost; the evaluation section covers what
still has to be measured.

Memory waste pulls the other way. On 0.6B (28 layers, 8 KV heads, head_dim
128, F32 K and V) one token costs 224 KiB. A sequence's partially filled last
block is private, so the worst-case tail waste per sequence is 13.8 MiB at 64
tokens, 27.8 MiB at 128 and 55.8 MiB at 256. With many short sequences
resident, that is the number that bounds concurrency, not indirection cost.

## Decision: paged, with a logical/physical split

The cache is two things with different owners.

**Logical view, owned by `model/`, identical on every backend.** Which
positions a sequence holds and in what order, the block table that maps
logical block index to physical block id, refcounts, and the free list. This
is bookkeeping and has no layout in it.

**Physical storage, owned by the backend.** Block size in tokens, byte layout
inside a block, dtype, and where the memory lives. The CPU wants large blocks
with each KV head contiguous inside the block. A GPU wants something
else: vLLM interleaves for coalesced loads and uses small blocks because its
attention is massively parallel. Neither number belongs in the model layer.

The rule that keeps the split honest: the model layer never computes a byte
offset into KV storage. It hands the backend a block table and a length, and
the backend maps those to bytes however it likes.

### Logical side

```
BlockPool     free list (O(1) alloc/release), refcount per block,
              byte budget as the constructor limit; physical storage grows
              on demand up to it, so a short chat does not allocate the budget
KVSequence    ordered physical block ids, valid length;
              append allocates a block when length % block_tokens == 0;
              fork shares full blocks (refcount+1) and copies the partial tail
```

Both own what they hold: neither is copyable, the pool is not movable either
because sequences hold its address and it is configured in place while idle,
a sequence returns its blocks when destroyed or moved from, and every
bookkeeping vector is reserved to the budget so alloc, release, abort and
reset never allocate. The CPU storage grows by copying the history into
exact-size buffers, all layers before any is published, and reports the
capacity it retains. A failed step restores history and length; capacity
the backend grew for the attempt may stay retained, within the budget.

`KVSequence` replaces the per-model position bookkeeping; `Model` keeps one
today and the server keeps one per request later. The budget covers every
layer, K and V, and layout and alignment overhead. A CLI flag for it waits
for a concrete consumer with a default and a defined exhaustion behaviour;
no flag is added in this design.

`length` is the committed history: tokens whose K and V are written and
retired. A forward pass appends `batch` tokens with `kv_write` after
allocating the blocks they need; query `b` of the batch attends through
`length + b`. If allocation or the write fails, the sequence's block list and
`length` are unchanged and any blocks allocated for the attempt are released,
so a failed step leaves the previous history valid.

### Backend contract

```
KVLayout   { block_tokens }                      queried once, backend-chosen
KVStorage  handle from kv_alloc; owns the physical blocks of one cache
KVView     { storage, blocks, n_blocks, length } one sequence, one layer
kv_alloc(layers, max_tokens) -> KVStorage        grows on demand
kv_write(layer, view, pos, k, v, batch)          model -> storage
attention(Q, layer, view, out, n_head, n_head_kv, head_dim, nbatch)
```

A view names its storage: block ids are only meaningful inside one
`KVStorage`, and a process may hold several caches (two models, or two
pools). The budget crosses the seam in tokens: only the backend knows what
a block costs in bytes, so a byte budget, when the server needs one,
converts inside the backend. Storage reports retained bytes and the peak
held during a growth copy separately. Storage is host memory now and
becomes a `Buffer` at step 5 of
[DEVICE-EXECUTION](DEVICE-EXECUTION.md) without changing this contract. The
public raw-pointer `attention` overload is deleted once every model, test and
benchmark caller uses the view form; a contiguous implementation may survive
as a private detail behind the view.

### CPU physical layout

Per layer, K and V pools of `n_blocks * block_tokens * n_head_kv * head_dim`
floats. Inside a block the order is `[kv_head][token][head_dim]`, so each
head's history within a block is contiguous and the existing dot and
weighted-value loops run unchanged inside a block. Attention walks blocks in
table order, which is the form the benchmark measured. The softmax stays one
global pass over the scores and the value accumulation stays token-ordered
across block boundaries; there are no per-block partial reductions, so the
arithmetic and its reduction order are those of the contiguous path.
`block_tokens` is 128, from the screening below.

### Lifetime and reuse

A physical block returns to the free list when its refcount reaches zero
**and** the backend has retired every submission that read it. On the eager
CPU backend the second condition is always already true. Under the async
contract of DEVICE-EXECUTION it is the `sync()` point; refcounts alone must
never free device memory a kernel may still be reading. The same rule holds
the view's block table and its `KVStorage` alive until retirement, not only
the blocks.

A fork copies only the partial tail block; full blocks are shared read-only.
A write to a shared full block is a design error and is checked, not handled.

### Prefix sharing

Sharing is by full immutable blocks only. A block's lookup key is the hash
of (model identity including revision and the effective RoPE and position
configuration, dtype and layout id, block position range, chain hash of all
token ids up to and including this block, adapter state). A hash hit is a
candidate, not a match: the actual token prefix and the identity fields are
compared before two sequences alias a block, because a chain hash inside the
key is still a hash. The index and its eviction policy land with the server,
which is their first consumer, not with this design.

## Out of scope

- F16 or quantized KV. A separate change with its own HF gate.
- Scheduler, admission, continuous batching across sequences.
- Prefix index and eviction.
- Any change to attention arithmetic or reduction order.

## Evaluation before implementation

The microbenchmark chose the candidates. The default is chosen on real
models, each candidate against the contiguous baseline, following AGENTS
"Measuring a change":

- Candidates: 64, 128 and 256, each against the contiguous baseline.
- Workloads: decode, prefill, and follow-up turns; Qwen3-0.6B and Qwen3-8B
  Q8_0; lengths that straddle block boundaries (63, 64, 65, 127, 128, 129,
  255, 256, 257 and a long prompt) so the partial-tail path is exercised.
- Growth: allocation and growth cost per token against the doubling copy.
- Memory: allocated versus used bytes per sequence, at each length.
- Method: `tools/ab_runner.py`, A/A first, interleaved arms, paired ratios,
  all samples kept, activity monitoring in every arm, raw commands and binary
  identities recorded under `docs/benchmarks/`.

The default is the complete tradeoff of latency, allocated versus used
bytes and growth cost, not the attention column alone. A contiguous path
survives only if the paged form loses beyond noise on the primary decode
workload.

### Screening result (2026-09-20)

Eight frozen plans in `docs/benchmarks/kv-block-screen-20260920/` (plan,
samples, report and a per-run monitor summary each; raw monitors archived
locally). Base is contiguous main b46994c; candidates are ec0747d built
with 64, 128 and 256. Qwen3-0.6B Q8_0 with 7 pairs and Qwen3-8B Q8_0 with 5
pairs, 6 threads, 32 decoded tokens, prompts of 50, 105, 247 and 841 tokens
so decode crosses the 64, 128 and 256 boundaries. Cells are the paired
median change against base, prefill / decode; F is a cell that failed the
runner's frozen rule (mean or median below -3%, or baseline wins at the
binomial threshold). System CPU during runs averaged 32% to 59% against the
benchmark's own 37.5%, so background activity was present; every sample is
kept.

| Plan | tokens | 64 pp / tg | 128 pp / tg | 256 pp / tg |
|---|---|---|---|---|
| 0.6B | 50 | -7.8% F / +0.7% | +0.5% / +2.3% | -11.0% F / -0.3% |
| 0.6B | 105 | +5.0% / +0.3% | -1.3% / +0.3% | +1.3% / +3.5% |
| 0.6B | 247 | -6.5% F / -2.0% | -2.7% / -0.8% | +2.4% / -2.6% |
| 0.6B | 841 | -3.3% F / -2.2% | +0.1% / +0.2% | -0.6% / -1.4% |
| 8B | 50 | -1.0% F / -0.5% | -2.0% / -1.2% | -6.2% F / -1.4% |
| 8B | 105 | +0.2% / -0.8% | -1.2% / -0.8% | -0.5% / -1.8% |
| 8B | 247 | -4.0% F / +2.4% | -1.9% / +0.6% | -0.1% / +1.6% |
| 8B | 841 | -2.8% / +2.2% | -3.9% F / -3.6% F | -2.8% / +0.9% |

Memory at the end of a run, from `generate --verbose`:

| Plan | 64 alloc / peak / used | 128 alloc / peak / used | 256 alloc / peak / used |
|---|---|---|---|
| 0.6B, 82 tokens | 28 / 42 / 18 MiB | 28 / 28 / 18 MiB | 56 / 56 / 18 MiB |
| 8B, 82 tokens | 36 / 54 / 23 MiB | 36 / 36 / 23 MiB | 72 / 72 / 23 MiB |
| 0.6B, 873 tokens | 224 / 336 / 191 MiB | same | same |
| 8B, 873 tokens | 288 / 432 / 246 MiB | same | same |

Decision: **128**. 64 fails prefill in five of eight plans and never wins.
128 and 256 are not separable on decode; 256 fails prefill on both
50-token plans and backs twice the memory for a short sequence, which is the
figure that bounds server concurrency. 128's one failing cell (8B, 841
tokens, 4 of 5 pairs) is noted and not explained; it is the first thing to
re-measure when the contiguous path is deleted or the copying growth is
replaced. The `LLMX_KV_BLOCK` knob is deleted with this decision, per
AGENTS: a temporary A/B knob goes once it has answered its question.

## Order of work

| # | Step | Gate |
|---|---|---|
| 1 | `BlockPool`, `KVSequence`, paged host storage, view-form `attention`; one sequence, same outputs | HF gate unchanged; A/B vs contiguous picks `block_tokens` |
| 2 | Fork with tail copy, refcount release, `kv-cache` CTest extended | Distinct values across shared and private blocks |
| 3 | Storage on `Buffer`, completion-gated release | DEVICE-EXECUTION step 5, no CPU regression |
| 4 | Prefix index, per-request sequences | With the server, ROADMAP #7 |

## Agreement record

XDEV agreed the direction on 2026-09-20 with six conditions, all accepted
and folded into the sections above: 64 stays in the real-model screening;
the raw-pointer overload goes only after every caller migrates; the byte
budget is a limit that storage grows toward, with no CLI flag yet; views
name their storage and `length` has a defined meaning and failure rule;
softmax and value accumulation keep the contiguous arithmetic across blocks
and async retirement holds tables and storage as well as blocks; a hash hit
is verified against the actual prefix and model identity.
