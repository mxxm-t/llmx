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
- Q4_K decode also has a fused row dot. F16C availability is cached and used
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
  meets the activation, `(q*d)*x`, so nothing intermediate is larger than the
  result and Q8_0 has no overflow window. The fused K-quant dots accumulate
  `sum(q*x)` and apply the scale afterwards, which is what makes them fast and
  what lets a large activation reach infinity before a small scale could bound
  it. Those rows fall back to dequantizing first when the fused result is not
  finite. Do not "simplify" the Q8_0 kernel into the same shape: measured at
  q=127 and x=2^123 the accumulate-first form reaches 4.3e40 where the exact
  answer is finite. `tests/fused_dot_overflow.cpp` pins both families.
- `rms_norm`, `rope`: AVX2 vectorized with scalar tails for non-multiples of 8.
- `rms_norm_rows`, `norm_rope_rows`, `silu_mul`, `add`: the batched forms the
  model calls. Two private helpers decide dispatch. `spread` keeps a stage on
  the calling thread below two rows per worker; `chunk` keeps elementwise spans
  under 32K elements there. Both thresholds are properties of a host thread
  pool - waking it costs more than the work - and a device backend must not
  inherit them. `silu_mul` keeps `std::exp` per element: a vectorized
  approximation would shift logits and needs its own correctness gate.
- `parallel_for` stays public here but is deliberately off the `Backend`
  interface; the batched ops above are how the model gets parallelism.
- `attention`: causal GQA over the host KV cache. Heads use the persistent
  worker pool and separate score rows, reused across queries. The backend
  reads contiguous per-head histories with an explicit head stride in floats;
  physical capacity does not extend the causal sequence. The layout change
  preserves dot and value-reduction arithmetic order.
  The backend
  grows scratch to the sequence being processed, rather than reserving the
  model's full context. AVX2 dots and weighted value accumulation have scalar
  tails; a scalar branch is retained for the runtime AVX2 check.
  Vectorized dot reductions change summation order and require the HF gate.
  Value coefficients are normalized once, then output accumulators stay in
  registers across the KV sequence: 32-lane tiles, eight-lane remainders, and
  scalar tails. This avoids repeatedly loading/storing output rows while
  retaining each value lane's sequence order.
- `make_cpu_backend()` factory.

The AVX-512 path is deferred (no dev hardware to benchmark/prove lossless); a
runtime-dispatched AVX512F/VNNI kernel can be added later without touching the
`Backend` seam.

`run_prefill` scopes automatic Windows six-worker placement across the complete
prefill body. See [placement](backends-cpu-placement.md) for topology, fallback
and restoration rules. Nested scopes and effective thread-count changes inside
a scope are rejected. Same-count configuration remains a no-op; the guard is
for synchronous reentrancy and does not make concurrent calls safe.
