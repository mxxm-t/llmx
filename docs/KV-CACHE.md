# Paged KV cache

Design for the KV cache that the multi-user server (ROADMAP #7) and the device
execution model (ROADMAP #4a) both need. Status: proposal, awaiting agreement
between both developers before implementation. Nothing here is built yet.

## Why change

`HostKVCache` holds one sequence per model as a contiguous, head-major,
capacity-strided F32 array per layer, growing by doubling and copying. That is
correct and fast for one chat, and it cannot become a server:

- Two sequences need two full capacity reservations, or one grows into the
  other. Contiguous growth fragments memory and copies the whole history.
- A shared prompt prefix would have to be stored once per sequence.
- The layout is a CPU decision that the attention op receives as raw pointers
  plus a stride, so a GPU backend would inherit it.

vLLM answers all three with paging: uniform blocks, a per-sequence block
table, refcounts for sharing. llama.cpp's flat cell array with per-cell
sequence sets is simpler but needs a defragmentation pass and cannot share a
prefix without copying. Paging is the right structure. The open question was
its cost on CPU, where decode attention is memory bound and about a fifth of a
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
amortized over the vectorized inner loop. 128 and 256 are the candidates.
This is isolated attention with a cold cache, not an end-to-end decode cost;
the evaluation section covers what still has to be measured.

Memory waste pulls the other way. On 0.6B (28 layers, 8 KV heads, head_dim
128, F32 K and V) one token costs 224 KiB. A sequence's partially filled last
block is private, so the worst-case tail waste per sequence is 27.8 MiB at 128
tokens and 55.8 MiB at 256. With many short sequences resident, that is the
number that bounds concurrency, not indirection cost.

## Decision: paged, with a logical/physical split

The cache is two things with different owners.

**Logical view, owned by `model/`, identical on every backend.** Which
positions a sequence holds and in what order, the block table that maps
logical block index to physical block id, refcounts, and the free list. This
is bookkeeping and has no layout in it.

**Physical storage, owned by the backend.** Block size in tokens, byte layout
inside a block, dtype, and where the memory lives. The CPU wants 128 to 256
tokens with each KV head contiguous inside the block. A GPU wants something
else: vLLM interleaves for coalesced loads and uses small blocks because its
attention is massively parallel. Neither number belongs in the model layer.

The rule that keeps the split honest: the model layer never computes a byte
offset into KV storage. It hands the backend a block table and a length, and
the backend maps those to bytes however it likes.

### Logical side

```
BlockPool     free list (O(1) alloc/release), refcount per block,
              capacity fixed at construction from a byte budget
KVSequence    ordered physical block ids, valid length;
              append allocates a block when length % block_tokens == 0;
              fork shares full blocks (refcount+1) and copies the partial tail
```

`KVSequence` replaces the per-model position bookkeeping; `Model` keeps one
today and the server keeps one per request later.

### Backend contract

```
KVLayout   { block_tokens }                      queried once, backend-chosen
KVView     { blocks, n_blocks, length }          one sequence, one layer
kv_alloc(layers, n_blocks)                       physical storage
kv_write(layer, view, pos, k, v, batch)          model -> storage
attention(Q, layer, view, out, n_head, n_head_kv, head_dim, nbatch)
```

Storage is host memory now and becomes a `Buffer` at step 5 of
[DEVICE-EXECUTION](DEVICE-EXECUTION.md) without changing this contract. The
current raw-pointer `attention` overload is deleted when the view form lands;
there is no reason to keep two.

### CPU physical layout

Per layer, K and V pools of `n_blocks * block_tokens * n_head_kv * head_dim`
floats. Inside a block the order is `[kv_head][token][head_dim]`, so each
head's history within a block is contiguous and the existing dot and
weighted-value loops run unchanged inside a block. Attention walks blocks in
table order, which is the form the benchmark measured. `block_tokens` is
decided by the evaluation, not here.

### Lifetime and reuse

A physical block returns to the free list when its refcount reaches zero
**and** the backend has retired every submission that read it. On the eager
CPU backend the second condition is always already true. Under the async
contract of DEVICE-EXECUTION it is the `sync()` point; refcounts alone must
never free device memory a kernel may still be reading.

A fork copies only the partial tail block; full blocks are shared read-only.
A write to a shared full block is a design error and is checked, not handled.

### Prefix sharing

Sharing is by full immutable blocks only. A block's identity is the hash of
(model revision, dtype and layout id, block position range, chain hash of all
token ids up to and including this block, adapter state). Hash equality is a
lookup key; the full key is compared before two sequences alias a block.
The index and its eviction policy land with the server, which is their first
consumer, not with this design.

## Out of scope

- F16 or quantized KV. A separate change with its own HF gate.
- Scheduler, admission, continuous batching across sequences.
- Prefix index and eviction.
- Any change to attention arithmetic or reduction order.

## Evaluation before implementation

The microbenchmark chose the candidates. The default is chosen on real
models, with both candidates against the contiguous baseline, following
AGENTS "Measuring a change":

- Workloads: decode, prefill, and follow-up turns; Qwen3-0.6B and Qwen3-8B
  Q8_0; lengths that straddle block boundaries (127, 128, 129, 255, 256, 257
  and a long prompt) so the partial-tail path is exercised.
- Growth: allocation and growth cost per token against the doubling copy.
- Memory: allocated versus used bytes per sequence, at each length.
- Method: `tools/ab_runner.py`, A/A first, interleaved arms, paired ratios,
  all samples kept, activity monitoring in every arm, raw commands and binary
  identities recorded under `docs/benchmarks/`.

A contiguous fast path survives only if the paged form loses beyond noise on
the primary decode workload.

## Order of work

| # | Step | Gate |
|---|---|---|
| 1 | `BlockPool`, `KVSequence`, paged host storage, view-form `attention`; one sequence, same outputs | HF gate unchanged; A/B vs contiguous picks `block_tokens` |
| 2 | Fork with tail copy, refcount release, `kv-cache` CTest extended | Distinct values across shared and private blocks |
| 3 | Storage on `Buffer`, completion-gated release | DEVICE-EXECUTION step 5, no CPU regression |
| 4 | Prefix index, per-request sequences | With the server, ROADMAP #7 |

## Open for agreement

1. Block size decided by measurement between 128 and 256, with memory waste
   reported next to latency. Any objection to excluding 64?
2. Deleting the raw-pointer `attention` overload once the view form exists.
3. Byte budget as the `BlockPool` constructor input, with the CLI flag named
   after llama.cpp's when one exists.
