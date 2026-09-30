# `src/backends/cpu/cpu_backend.hpp` - CPU backend (AVX2)

CPU implementation of the `Backend` interface, in namespace `backend`.
`supports_type` reads the quant registry, including its F32 entry, and `implements(op)` is true for every `Op`.
The build requires x86-64 AVX2, FMA and F16C (`docs/BUILD.md`), and the kernels use them with no runtime check and no scalar fallback; their scalar loops cover the tails of lengths that are not a multiple of 8.
Every multiply-add in those tails is an explicit FMA (`std::fma`), never `a * b + c`: a compiler that contracts fuses such an expression in one inlined copy and not in another by the code around it, which gave a prompt row different bits by its place in the batch (`tests/backend_group.cpp`, docs/STATUS.md).
A compile without them stops at one `#error` at the top of the header.

- Persistent worker pool. The decode row dots and attention both run through it; previously each created and joined `std::thread`s per call, which on Qwen3-8B was thousands of thread creations per token.
- The pool starts on the first dispatch that needs it (`run_parallel`), at the thread count in force then, and runs until the count changes.
  Construction and `set_threads` start no threads, so a loader that sets its count before any work starts one pool of the size it uses, and a one-thread backend never starts one.
  The first parallel pass after a load therefore includes the pool's start, once; the synthetic `bench` runs one untimed matmul first so that its timed loop leaves the start out.
  A count changed after work stops the running workers, and the next dispatch starts a pool of the new size.
  `workers_started()` counts the threads started over the backend's life; `tests/backend_errors.cpp` and `tests/model_validation.cpp` pin it.
- Dispatch catches exceptions from callers and workers, waits for all
  participants, clears the borrowed job and rethrows on the caller. Failure
  leaves outputs potentially partial but the pool reusable.
  A failed start joins the threads it created and fails the dispatch that asked for it before any participant runs; the count stays and the next dispatch retries, so a lasting failure fails every dispatch rather than leaving the backend serial.
  A count change allocates its scratch before stopping anything, so a failed change keeps the previous count and pool.
  Concurrent or recursive submissions remain unsupported.
- `set_threads(0)` leaves the current pool unchanged, including inside a prefill scope.
  The constructor takes the initial automatic count from `core::automatic_threads()` ([cpus](core-cpus.md)).
  The CLI resolves its automatic thread flags before requesting a change.
- `read`, `write` and `copy` accept valid zero-byte ranges, including empty
  buffers and an offset exactly at the end. Range checks still reject offsets
  past the end, and a nonempty write still requires a non-null source.
- `row_dot`: one weight row against one activation row in float, the one float decode row dot.
  F32 takes `dot_f32`, Q8_0 `dot_row_impl`, and Q4_K, Q5_K and Q6_K their fused dequant+FMA dots, falling back to `dot_row_dequant` when a fused sum overflows.
  A decode run of those types calls it in one pooled loop over weight rows, walking all activation columns against each row before moving on; a routed decode entry calls it for every type.
  Q8_0 always takes this float path; the other supported quantized types take `q8_dots.hpp` unless `set_decode_activations8(false)` selects their float reference path.
  Other types, routed Q4_0 and Q4_1 decode among them, take `dot_row_dequant`, which dequantizes and sums in double, while a dense Q4_0 or Q4_1 decode keeps the batched float path, so the two differ in rounding.
  The Q6_K dot rounds each scaled group sum before accumulating it, through separate intrinsics, because compilers fused the two into one FMA or not by the code around them and by their contraction rules.
  GCC and MSVC builds round this dot as they did before the intrinsics, and a Clang build, which fused the plain expression at its default contraction, now rounds as they do; a Clang build with `-ffp-contract=fast` would still fuse the intrinsics.
  `dot_row_impl` streams weight blocks with no software prefetch, which the hardware prefetcher makes unnecessary on these sequential streams; native sampled instruction locations alone do not establish DRAM bandwidth saturation or memory-stall causes.
- `matmul`: type-generic batched matmul. Dequantizes `DOT_ROWS` weight rows
  through the registry, then walks the batch. `dot_f32_x4` loads each
  activation vector once and reuses it across those 4 rows, because the naive
  kernel needs 2 loads per FMA while Zen3 sustains only ~2 loads/cycle against
  2 FMAs/cycle, so it was load bound at half of FMA peak. `dot_f32` uses four
  independent accumulators; with one, every FMA depends on the previous and the
  loop runs at FMA latency rather than throughput.
  Its paths, `embed` and the expert products all size a row by `quant::row_bytes`, so a row that ends inside a block is refused rather than truncated.
- `dot_f32_x4x3` reuses four weight rows across three activation columns in
  batched prefill; remaining columns/rows use the smaller kernels. Explicit
  ordered lane reductions avoid making all 12 accumulators addressable after
  the FMA loop. Per-lane accumulation, final addition order and tails remain
  unchanged; the larger epilogue trades code size for less stack traffic.
  A prompt row takes the three-, two- or one-column dot by its place in the batch, so the three give a column the same bits: FMAs per lane, the lanes added in order, then the tail's FMAs in order, which `backend-group` holds each column of all three to.
- F32 matrices use those same float dot kernels directly on resident host
  weights, without a dequantization buffer or row copy. Quantized inputs retain
  the existing row staging and fused decode paths.
  F32 decode goes through `row_dot`, `dot_f32` one row at a time for contiguous weight access; this has a different reduction order from the fused four-row dot.
- `DOT_ROWS` is the fused kernel's width, not a tuning constant. A cache-byte
  budget was measured instead and was worse at every size (see
  `docs/STATUS.md`).
- `matmul_group`: shares quantized activations across multiple Q4_0,
  Q4_1, Q4_K, Q5_K or Q6_K decode projections in one pool dispatch. `RowRuns`
  also admits batches of generated rows. F32, Q8_0, prompt rows, single projections
  and unsupported integer-dot configurations fall back to separate matmul
  calls; small grouped jobs run on the caller (`split_rows`, below).
  Whether every row is a generated token's is read through `for_each_run`, so malformed runs are refused here as in `matmul`.
- `dot_row_impl`: AVX2 fused dequant + FMA accumulation over int8 blocks.
  The stored half scale is broadcast directly from memory before F16C
  conversion; signed byte groups load directly into the widening operations.
  Float activations and per-lane accumulation order are unchanged.
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
- The `rms_norm_raw` helper under `rms_norm_rows`, `gated_rms_norm` and `norm_rope_raw`, and
  `rope_raw` under `norm_rope_raw`: AVX2 vectorized with scalar tails for
  non-multiples of 8.
  The tails' multiply-adds are explicit FMAs, `rope_raw`'s the same ones as its vector body, so a build that contracts expressions and one that does not give the same bits (docs/QWEN35.md, Row classes), and `rms_norm_raw`'s tail scales by the row's factor times the weight, as its body does.
  `norm_rope_raw` norms each head and rotates its first dims from heads read at a stride of their own into contiguous heads, the one norm and rope of `norm_rope_partial`, which Qwen3 reaches in place at the full rotary width through `Backend::norm_rope_kv`.
- `silu_of`, `sigmoid_of`, `softplus_of` and `decay_of`: the elementwise ops' transcendental steps, each computed in one place with `std::exp` per element.
  `softplus_of` takes its argument as it is above 20, and its `exp` in float and its log in double, rounded once, so its value does not follow how the C library rounds the float `log1p`, and `decay_of` gives 0 for a decay factor below 2^-126, as every backend does, so the factor does not depend on the host's denormal handling, though a state value it scales below 2^-126 does.
- The qwen35 layers' ops (`backends-backend.md`), each on the pool with every value computed by one routine whatever the thread count, so the thread count and the grouping of rows into calls change no bit.
  - `causal_conv_silu`: one task per (view, channel), which reads the channel's carried rows from slot `src`, zero before the sequence's start, walks the view's rows through a window of the last raw values and leaves that window in slot `dst`; on the calling thread below 32K row values.
  - `gated_delta_rule`: a prologue over the rows writes the L2-normed q and k of every K head and each V head's beta and decay into scratch, then one task per (view, V head) copies its matrix from slot `src` into slot `dst`, or zeroes it at length 0, and runs the recurrence there token by token (`delta_rule_head`).
    Each V column's arithmetic is its own, with every multiply-add an explicit FMA and the sums over K in row order, so the columns run 32, 8 or 1 at a time with the same bits, and a block of 32 columns of a 128-row matrix stays in the first-level cache over a whole view's tokens.
  - `gated_rms_norm`, `norm_rope_partial` and `sigmoid_mul`: row by row through `spread`; `gated_rms_norm` and `sigmoid_mul` do not read their row runs, since the CPU keeps no activation copies.
  - `state_slot` resolves a slot of a state storage's layer to host floats and refuses storage of another backend; the conv and the recurrence resolve every view's slots before they write anything.
- `rms_norm_rows`, `silu_mul`, `add`: the batched forms the model calls.
  Private helpers decide dispatch.
  `spread` keeps a stage on the calling thread below two rows per worker; `chunk` keeps elementwise spans under 32K elements there; `split_rows` hands the row dots of a decode matmul, `matvec_q8x`, `matmul_group` and the routed decode entries to the pool in one contiguous range per worker, and keeps them on the caller below eight rows per worker, except an F32 decode matmul, which splits as the batched float path does, from `DOT_ROWS` rows per worker in whole `DOT_ROWS` chunks.
  The thresholds are properties of a host thread pool - waking it costs more than the work - and a device backend must not inherit them.
  `silu_mul` keeps `std::exp` per element: a vectorized approximation would shift logits and needs its own correctness gate.
- `parallel_for` stays public here but is deliberately off the `Backend`
  interface; the batched ops above are how the model gets parallelism.
- `CpuKVStorage`, `kv_layout`, `kv_alloc`, `kv_write`: the physical half of the paged KV cache.
  Per layer, block `b` of K or V holds `[kv_head][token][head_dim]`, so a head's history is contiguous inside a block.
  The storage derives from `BlockKVStorage` (`backends-kv_storage.md`), which backs blocks in doubling steps as ids are first written, up to the budget, copies the history into exact-size buffers for every layer before publishing any, and checks the views.
  Its copies are eager, so its `retire` keeps nothing, and its `after_growth` resolves each layer's host pointers once per growth, which `kraw`, `k`, `kh` and their V forms read.
  `KV_BLOCK_TOKENS` is 128, fixed by the screening in `docs/KV-CACHE.md`; it is a property of this backend, not a knob.
- `attention`: causal GQA over a `KVView`. Heads use the persistent worker
  pool and separate score rows, reused across queries. Blocks are walked in
  table order with one global softmax and token-ordered value accumulation
  across block edges, so the arithmetic is that of a contiguous history.
  The backend grows scratch to the sequence being processed, rather than
  reserving the model's full context.
  AVX2 dots and weighted value accumulation have scalar tails, of explicit FMAs.
  Vectorized dot reductions change summation order and require the HF gate.
  Value coefficients are normalized once, then output accumulators stay in
  registers across the KV sequence: 32-lane tiles, eight-lane remainders, and
  scalar tails. This avoids repeatedly loading/storing output rows while
  retaining each value lane's sequence order.
- Decode rows and prompt rows: with row runs a generated token (extent 1)
  takes the decode dots and a prompt's rows the batched float path, or the
  prompt dots for routed experts (below) and for K-quant rows at least
  `kPromptDotsFrom` (4096) wide, where the float path's dequantized row
  blocks no longer stay in the first-level cache, so a row computes the same alone
  or beside others; without runs a one-column call is decode. The decode dots (`q8_dots.hpp`) quantize a call's activations once
  per block of 32, 8-bit for Q4_K and Q5_K and 16-bit for Q4_0, Q4_1
  and Q6_K, plus MXFP4, and meet the packed
  weights in integers (`maddubs` and `madd`), one scale per block; the float
  dots they replaced converted every weight and were bound by arithmetic.
  If a tiny finite block overflows the float reciprocal, an exceptional scalar
  path uses the smallest positive representable scale covering its magnitude
  range, then rounds double-precision ratios to nearest, ties to even. This
  preserves zeros, signs and the sum of the packed integers even when the
  ordinary scale would round to zero; tiny endpoints need not reach the integer
  limit when a finer covering scale cannot be represented. Normal-range SIMD
  arithmetic is unchanged. The fallback is checked strictly for nearest
  reconstruction; ordinary controls allow the existing float-rounding error.
  The range tests compare reconstructed values against the original inputs and
  neighbouring integer representations; they do not reuse the encoder formula.
  Gradual underflow is assumed; flush-to-zero and nonfinite input handling are
  not established by this check.
  Q8_0 always reads the original F32 activations through the existing float dots, including grouped and routed calls; its integer consumers are removed.
  `set_decode_activations8(false)` keeps the float dots for the other types, which the device
  comparison test's reference and the float-kernel checks use.
  `each_run` reads the runs through `backend.hpp` `for_each_run`, keyed by whether a run is a generated token's.
- `route_experts`, `matmul_experts`, `matmul_experts_add`: routing in
  float, then the entries grouped by expert. For types with integer dots, a generated token's entries take
  the decode dots, every such entry's rows of a call in one pool dispatch. A
  prompt's entries take the prompt dots (`q8_dots.hpp` `dot_block`): stretches
  of 16 of an expert's rows handed to workers as they free up, each against
  all of that expert's entries. The block walks the inner dimension 256
  values at a time, unpacks each weight row's 256 once into a buffer every
  entry then reads, keeps each group of 32's products exact in integers and
  meets eight groups' scales in one vector multiply-add, so a (row, entry)
  pair accumulates in the same order whatever else is in the block. The
  activations are quantized once per call, split across the pool, and gate
  and up share them. For MXFP4, whose prompt retains original F32 activations, or where a type has no quantized dots, including Q8_0, or with
  `set_decode_activations8(false)`, a prompt's entries take one batched float
  matmul per expert over its gathered rows (`matmul_raw`, the matmul on host
  addresses, reaches an expert's matrix inside the stacked tensor).
- `memory_available()`: the host memory the process can still take, the host's available physical memory or less where a cgroup or job object memory limit leaves less (`core/host_memory.hpp`). Weights on the CPU read the mapped file in place, so what counts against it is caches, activations and what a loader materializes. `reads_in_place()` is true: `adopt` aliases the caller's bytes. `is_cpu()` is true, so experts on the CPU beside it stay on it.
- `make_cpu_backend()` factory.

The AVX-512 path is deferred (no dev hardware to benchmark/prove lossless); a
runtime-dispatched AVX512F/VNNI kernel can be added later without touching the
`Backend` seam.

Historical comments at `83cca18` recorded 14.1 us for an empty condition-variable
dispatch and estimated 196 matvec plus 28 attention dispatches per decode token.
They also used approximate FMA latency/throughput of 4/0.5 cycles to motivate
independent accumulators. These are preserved design notes, not measurements
repeated by the CPU interface checkpoint; the original benchmark setup and raw
samples are not identified by those comments.

`run_prefill` scopes automatic Windows six-worker placement across the complete
prefill body. See [placement](backends-cpu-placement.md) for topology, fallback
and restoration rules. Nested scopes and effective thread-count changes inside
a scope are rejected. Same-count configuration remains a no-op; the guard is
for synchronous reentrancy and does not make concurrent calls safe.

MXFP4 decode reuses this activation owner with 16-bit packed inputs. Its nibble lookup widens weights directly into integer products; exceptional scale products or nonfinite fast sums use decoded F32 weights and double products over those same activations. The format interpretation and boundary coverage live in [quant-mxfp4](quant-mxfp4.md).
