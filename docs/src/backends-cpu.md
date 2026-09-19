# `src/backends/cpu/cpu_backend.hpp` — CPU backend (AVX2)

CPU implementation of the `Backend` interface, in namespace `backend`.

- Detects AVX2 **once** in the constructor (via `__cpuid` on MSVC, `__get_cpuid`
  on GCC/Clang) and caches it — not per row.
- Persistent worker pool, started once. `matvec_q8_0` and attention both
  run through it; previously each created and joined `std::thread`s per call,
  which on Qwen3-8B was thousands of thread creations per token.
- `matvec_q8_0`: fused dequant+FMA AVX2 row dot, kept for the single-column
  (decode) case, which is bandwidth bound.
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
  batched prefill; remaining columns/rows use the smaller kernels.
- F32 matrices use those same float dot kernels directly on resident host
  weights, without a dequantization buffer or row copy. Quantized inputs retain
  the existing row staging and fused decode paths.
  Single-column F32 uses `dot_f32` one row at a time for contiguous weight
  access; this has a different reduction order from the fused four-row dot.
- `DOT_ROWS` is the fused kernel's width, not a tuning constant. A cache-byte
  budget was measured instead and was worse at every size (see
  `docs/STATUS.md`).
- `dot_row_impl`: AVX2 fused dequant (f16 scale broadcast) + FMA accumulation
  over int8 blocks, with a scalar fallback.
- `rms_norm`, `rope`: AVX2 vectorized with scalar tails for non-multiples of 8.
- `attention`: causal GQA over the host KV cache. Heads use the persistent
  worker pool and separate score rows, reused across queries. The backend
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
