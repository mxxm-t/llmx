# `src/backends/cpu/cpu_backend.hpp` - CPU backend (AVX2)

CPU implementation of the `Backend` interface, in namespace `backend`. The
current build requires x86 AVX2/FMA/F16C; retained scalar branches do not make
the compiled binary portable to older CPUs.

- Detects AVX2 **once** in the constructor (via `__cpuid` on MSVC, `__get_cpuid`
  on GCC/Clang) and caches it - not per row.
- Persistent worker pool, started once. `matvec_q8_0` and attention both
  run through it; previously each created and joined `std::thread`s per call,
  which on Qwen3-8B was thousands of thread creations per token.
- Dispatch catches exceptions from callers and workers, waits for all
  participants, clears the borrowed job and rethrows on the caller. Failure
  leaves outputs potentially partial but the pool reusable. Partial startup
  joins created threads; failed reconfiguration falls back to serial execution.
  Concurrent or recursive submissions remain unsupported.
- `matvec_q8_0`: fused dequant+FMA AVX2 row dot, kept for the single-column
  (decode) case. It streams weight blocks; native sampled instruction locations
  alone do not establish DRAM bandwidth saturation or memory-stall causes.
- Q4_K, Q5_K and Q6_K decode also have fused row dots. F16C availability is cached and used
  for half conversion where supported.
- `matmul`: type-generic batched matmul. Dequantizes `DOT_ROWS` weight rows
  through the registry, then walks the batch. `dot_f32_x4` loads each
  activation vector once and reuses it across those 4 rows, because the naive
  kernel needs 2 loads per FMA while Zen3 sustains only ~2 loads/cycle against
  2 FMAs/cycle, so it was load bound at half of FMA peak. `dot_f32` uses four
  independent accumulators; with one, every FMA depends on the previous and the
  loop runs at FMA latency rather than throughput.
- `dot_f32_x4x3` reuses four weight rows across three activation columns in
  batched prefill; remaining columns/rows use the smaller kernels. Explicit
  ordered lane reductions avoid making all 12 accumulators addressable after
  the FMA loop. Per-lane accumulation, final addition order and tails remain
  unchanged; the larger epilogue trades code size for less stack traffic.
- F32 matrices use those same float dot kernels directly on resident host
  weights, without a dequantization buffer or row copy. Quantized inputs retain
  the existing row staging and fused decode paths.
  Single-column F32 uses `dot_f32` one row at a time for contiguous weight
  access; this has a different reduction order from the fused four-row dot.
- `DOT_ROWS` is the fused kernel's width, not a tuning constant. A cache-byte
  budget was measured instead and was worse at every size (see
  `docs/STATUS.md`).
- `matmul_group`: one pool dispatch for multiple native F32/Q8_0/Q4_K decode
  projections, preserving each matrix's row partition and dot kernel. Other
  formats, batches, single projections and small jobs use sequential matmul.
- `dot_row_impl`: AVX2 fused dequant + FMA accumulation over int8 blocks.
  The stored half scale is broadcast directly from memory before F16C
  conversion; signed byte groups load directly into the widening operations.
  Float activations and per-lane accumulation order are unchanged. Software
  half conversion and scalar dot fallbacks remain available.
- **Order of operations decides whether a kernel can overflow, and the two
  families differ.** `dot_row_impl` folds the scale into each weight before it
  meets the activation, `(q*d)*x`, avoiding the demonstrated scale-after-sum
  overflow. This does not prevent accumulation overflow under cancellation.
  The fused K-quant dots accumulate
  `sum(q*x)` and apply the scale afterwards, which is what makes them fast and
  what lets a large activation reach infinity before a small scale could bound
  it. Those rows fall back to dequantizing first when the fused result is not
  finite. Do not "simplify" the Q8_0 kernel into the same shape: measured at
  q=127 and x=2^123 the accumulate-first form reaches 4.3e40 where the exact
  answer is finite. `tests/fused_dot_overflow.cpp` pins both families.
- `rms_norm`, and the `rope_raw` helper under `norm_rope_rows`: AVX2
  vectorized with scalar tails for non-multiples of 8.
- `rms_norm_rows`, `norm_rope_rows`, `silu_mul`, `add`: the batched forms the
  model calls. Two private helpers decide dispatch. `spread` keeps a stage on
  the calling thread below two rows per worker; `chunk` keeps elementwise spans
  under 32K elements there. Both thresholds are properties of a host thread
  pool - waking it costs more than the work - and a device backend must not
  inherit them. `silu_mul` keeps `std::exp` per element: a vectorized
  approximation would shift logits and needs its own correctness gate.
- `parallel_for` stays public here but is deliberately off the `Backend`
  interface; the batched ops above are how the model gets parallelism.
- `CpuKVStorage`, `kv_layout`, `kv_alloc`, `kv_write`: the physical half of
  the paged KV cache. Per layer, block `b` of K or V holds
  `[kv_head][token][head_dim]`, so a head's history is contiguous inside a
  block. Blocks are backed in doubling steps as ids are first written, up to
  the budget; growth copies the history into exact-size buffers for every
  layer before publishing any, and `allocated_bytes` is the retained
  capacity. `KV_BLOCK_TOKENS` is 128, fixed by the screening in
  `docs/KV-CACHE.md`; it is a property of this backend, not a knob.
- `attention`: causal GQA over a `KVView`. Heads use the persistent worker
  pool and separate score rows, reused across queries. Blocks are walked in
  table order with one global softmax and token-ordered value accumulation
  across block edges, so the arithmetic is that of a contiguous history.
  The backend grows scratch to the sequence being processed, rather than
  reserving the model's full context. AVX2 dots and weighted value accumulation have scalar
  tails; a scalar branch is retained for the runtime AVX2 check.
  Vectorized dot reductions change summation order and require the HF gate.
  Value coefficients are normalized once, then output accumulators stay in
  registers across the KV sequence: 32-lane tiles, eight-lane remainders, and
  scalar tails. This avoids repeatedly loading/storing output rows while
  retaining each value lane's sequence order.
- Decode rows and prompt rows: with row runs a generated token (extent 1)
  takes the decode dots and a prompt's rows the batched float path (the
  prompt dots for routed experts, below), so a row computes the same alone
  or beside others; without runs a one-column call is decode. The decode dots (`q8_dots.hpp`) quantize a call's activations once
  per block of 32, 8-bit for Q8_0, Q4_K and Q5_K and 16-bit for Q4_0, Q4_1
  and Q6_K, the same split as the device's row kernels, and meet the packed
  weights in integers (`maddubs` and `madd`), one scale per block; the float
  dots they replaced converted every weight and were bound by arithmetic.
  `set_decode_activations8(false)` keeps the float dots, which the device
  comparison test's reference and the float-kernel checks use.
- `route_experts`, `matmul_experts`, `matmul_experts_add`: routing in
  float, then the entries grouped by expert. A generated token's entries take
  the decode dots, every such entry's rows of a call in one pool dispatch. A
  prompt's entries take the prompt dots (`q8_dots.hpp` `dot_block`): stretches
  of 16 of an expert's rows handed to workers as they free up, each against
  all of that expert's entries. The block walks the inner dimension 256
  values at a time, unpacks each weight row's 256 once into a buffer every
  entry then reads, keeps each group of 32's products exact in integers and
  meets eight groups' scales in one vector multiply-add, so a (row, entry)
  pair accumulates in the same order whatever else is in the block. The
  activations are quantized once per call, split across the pool, and gate
  and up share them. Where a type has no quantized dots, or with
  `set_decode_activations8(false)`, a prompt's entries take one batched float
  matmul per expert over its gathered rows (`matmul_raw`, the matmul on host
  addresses, reaches an expert's matrix inside the stacked tensor).
- `make_cpu_backend()` factory.

The AVX-512 path is deferred (no dev hardware to benchmark/prove lossless); a
runtime-dispatched AVX512F/VNNI kernel can be added later without touching the
`Backend` seam.

`run_prefill` scopes automatic Windows six-worker placement across the complete
prefill body. See [placement](backends-cpu-placement.md) for topology, fallback
and restoration rules. Nested scopes and effective thread-count changes inside
a scope are rejected. Same-count configuration remains a no-op; the guard is
for synchronous reentrancy and does not make concurrent calls safe.
