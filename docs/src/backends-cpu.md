# `src/backends/cpu/cpu_backend.hpp` - CPU backend (AVX2)

`dtype_path` describes possible matrix families for the placement record. F16 names packed Q4/Q5/Q6/MXFP4 decode, K-quant dense prompts at widths of at least 4096 and routed Q4/Q5/Q6 prompts, with other products retaining F32 inputs. BF16 names rounded matrix inputs and retained F32 routers. This catalog neither advertises a native policy nor records execution; the dispatch witnesses below remain the evidence for numerical bounds. MXFP4 range repair keeps the reconstructed packed inputs, so wider accumulation does not turn its activation class into F32.

Matrix dispatch records the activation form where it enters float or packed dots. Every packed CPU dot now reads the shared 16-bit activation rows, so the dispatch records `block-int16` directly without another type catalog. Dense, grouped and routed paths all contribute to the shared `MatrixPaths` evidence, which the model retains per stage.

Matrix calls consume their activation dtype in the existing dispatch owner. `native_dtypes` prefers F16, then F32. On AVX2, the F16 policy uses block-int16 activations in eligible Q4/Q5/Q6/MXFP4 products and wider F32 inputs elsewhere; the weights retain their stored interpretation. Explicit F32 bypasses the block-integer dots. BF16 inputs are rounded once per dense or routed call by `dtype_input`, then widened for the same F32 dot kernels. Grouped fallbacks forward the policy to each projection. BF16 is reported as emulated by `emulates_dtype` and is selected only by an explicit request; it is not an auto preference.

Q4_K/Q5_K decode reduces each group of 32 integer products exactly before converting to F32, using the same `sum8` reduction as the prompt dots. It accumulates one float lane per group across blocks, with separate weight and minimum terms; the minimum meets its activation scale before its integer sum. Scale/minimum metadata is unpacked into the eight lanes together.

Dense Q4_K/Q5_K packed prompts at widths of at least 4096 and eight or more columns use `PromptPairs` in the existing CPU dot owner. It repacks the already quantized 16-bit activations, pairing adjacent features across eight columns. The dot takes two such groups across two output rows, reusing each weight broadcast for sixteen columns; a final group uses four output rows. A compiler-guarded loop hint keeps GCC-compatible builds from expanding the inner pair loop eightfold. Complete tiles transpose eight-by-eight 32-bit lanes with AVX2, each lane holding two adjacent 16-bit features; a partial tile keeps the scalar copy and zero padding. Each integer lane sums one group of 32 products; its magnitude is bounded by 32 * 31 * 32767, so the reassociation is exact in int32. Integer group sums, the order of the per-group float FMAs and the final reduction are unchanged. Smaller prompts, routed products and decode keep their existing dots. The reusable packing scratch retains about two bytes per input value plus eight bytes per group of 32 values, with columns rounded up to eight, alongside the original activation rows; packing is included in measured matmul time.

CPU implementation of the `Backend` interface, in namespace `backend`.
`supports_type` reads the quant registry, including its F32, F16 and BF16 entries, and `implements(op)` is true for every `Op`.
F16 and BF16 weight storage is read-only. Embedding and prompt products widen through the registry; decode uses the shared `dot_float` below. The weights always retain their stored values, independently of activation dtype. F16 execution retains F32 activation inputs for these float-storage products; BF16 execution rounds inputs before the same float math. Model roles that require F32 vectors keep that requirement.
The build requires x86-64 AVX2, FMA and F16C (`docs/BUILD.md`), and the kernels use them with no runtime check and no scalar fallback; their scalar loops cover the tails of lengths that are not a multiple of 8.
Every multiply-add in those tails is an explicit FMA (`std::fma`), never `a * b + c`: a compiler that contracts fuses such an expression in one inlined copy and not in another by the code around it, which gave a prompt row different bits by its place in the batch (`tests/backend_group.cpp`, docs/STATUS-2026-09.md).
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
  F32, F16, BF16, IQ4_NL and IQ4_XS take `dot_float`, Q8_0 takes `dot_row_impl`, and Q4_K, Q5_K and Q6_K take their fused dequant+FMA dots, falling back to `dot_row_dequant` when a fused sum overflows.
  `dot_float` has one accumulation body with compile-time weight loaders: direct F32 vectors, eight unaligned F16 words widened through F16C, BF16 words shifted into binary32, or IQ4 signed lookup values scaled into F32. All loaders use the same four accumulators, FMA order, final reduction and scalar tails. `dot_f32` is its F32 entry point. Dense, grouped, routed and output-head decode share this owner.
  A decode run of those types calls it in one pooled loop over weight rows, walking all activation columns against each row before moving on; a routed decode entry calls it for every type.
  Q8_0, IQ4_NL and IQ4_XS always take this float path. Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4 take `q8_dots.hpp` under F16; Q5_0 and Q5_1 keep the dequantized products. An explicit F32 request selects the original float path, and BF16 rounds its inputs before the float dots.
  IQ4_NL looks up the signed nibble values and multiplies each weight by its binary16 scale before the activation FMA, preserving the four accumulators and final reduction of `dot_f32`. It does not form a potentially overflowing unscaled dot. Prompts keep the existing registry dequantization and F32 batch kernels; grouped, routed, head and embedding calls reuse those owners. `iq4-nl` holds each path to independently widened F32 weights, across row runs, threads, dtype policies, output guards and unaligned storage. The separate Python `iq4-nl` component checks tiny dense/MoE models against independent HF. The pinned 0.6B file also passes its file-exact CPU check. Original-weight quality approval remains pending.
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
  F32/F16/BF16 weight decode goes through `row_dot`, using `dot_float` one row at a time for contiguous weight access; this has a different reduction order from the fused four-row dot.
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
  - `argmax_rows` and `embed_ids`, the embedded drafter's (`backends-backend.md`): on the calling thread, a row at a time, `embed_ids` through `embed` for each valid id.
  - `gated_rms_norm`, `norm_rope_partial` and `sigmoid_mul`: row by row through `spread`; `gated_rms_norm` and `sigmoid_mul` do not read their row runs, since the CPU keeps no activation copies. The gated norm selects the existing `silu_of` or `sigmoid_of` helper through `NormGate`.
  - `state_slot` resolves a slot using that layer's full `StateStorage::slot_floats(layer)` stride and refuses storage of another backend. The conv and delta recurrence address only the common `StateShape` prefix, leaving any per-layer F32 tail unchanged; both resolve every view's slots before they write anything.
- `dilated_conv_silu_add` computes a causal depthwise convolution one channel at a time, applies SiLU and adds to the residual. Tap weights are `[channel, tap]`, in oldest-to-current order; history rows are spaced by `dilation`, with zero before sequence start. It reads the preceding raw `x` rows from the active layer's F32 state tail and writes the newest rows to `dst`; only exact in-place residual output is allowed.
- The Qwen4Exp stream ops: `group_rms_norm` applies the shared RMSNorm routine to each group's own gamma slice, and `silu_scaled` applies SiLU after its scale. `mix_streams` takes the per-element sigmoid gate, multiplies the normalized stream values, and averages over streams. `inject_streams` adds the block result to each residual stream with `2 * sigmoid(raw_gate / streams)`. `signed_sqrt_gate` reduces each key/query group, divides by `sqrt(dim)`, clamps the absolute score to `1e-6` before its signed square root, and gates the shared value. Row-wise work uses `spread`; `silu_scaled` uses `chunk`. `tests/stream_ops.cpp` holds their F32 equations, thread invariance, guards and alias contracts.
- `select_kv` pools each complete compression group of raw auxiliary F32 keys once per view, then applies F32 RMSNorm with the stored GGUF multiplier and split-half RoPE at the first token position. The caller supplies queries already normalized and RoPE-rotated. It scores each causal query in F32 by summing the rectified per-head dot divided by the square root of its width across heads, selects the highest-scoring complete groups with earlier groups winning exact ties, sorts selected token IDs chronologically, appends the causal partial tail, and pads to `budget + compression - 1` IDs with `UINT32_MAX`. Preflight checks cover geometry, views, CPU storage, spans and output/input overlap before the ID buffer is changed. This all-F32 operator does not reproduce mixed-precision HF casts or perform sparse attention; HF `topk` leaves tied-index order unspecified.
- `sparse_attention` reads the original K/V rows named by each query's chronological U32 ID row. It ignores `UINT32_MAX` and future IDs before cache access, maps query heads to KV heads by the GQA ratio, computes a max-subtracted softmax over selected keys, and accumulates values in F32. K and V independently support F32 or F16 cache storage; an empty selection writes zeros. It preflights every view, operand span and output alias before writing. `tests/sparse_attention.cpp` holds it to an independent double-precision oracle across widths 8, 128 and 256 and all four K/V type pairs, with batched and per-row bit equality, refusal guards and the selector-to-attention chain.
- `rms_norm_rows`, `silu_mul`, `add`: the batched forms the model calls.
  Private helpers decide dispatch.
  `spread` keeps a stage on the calling thread below two rows per worker; `chunk` keeps elementwise spans under 32K elements there; `split_rows` hands the row dots of a decode matmul, `matvec_q8x`, `matmul_group` and the routed decode entries to the pool in one contiguous range per worker, and keeps them on the caller below eight rows per worker, except an F32/F16/BF16 weight decode matmul, which splits as the batched float path does, from `DOT_ROWS` rows per worker in whole `DOT_ROWS` chunks.
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
  per block of 32, 16-bit for Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4, and meet the packed
  weights in integers (`madd`), one scale per block; the float
  dots they replaced converted every weight and were bound by arithmetic.
  Routed and smaller dense Q4_K/Q5_K prompt scratch keeps weight codes as bytes
  until the shared 16-bit dot widens them. Dense prompts with at least eight
  columns expand two weight rows for sixteen columns, or four rows for the
  eight-column remainder; activations remain 16-bit in both paths.
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
  The device comparison references and float-kernel checks request F32 on each matrix call; no mutable backend precision switch is needed.
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
  and up share them. For MXFP4, whose prompt uses float products over the selected policy's inputs, or where a type has no quantized dots, including Q8_0, or with an F32 or BF16 request, a prompt's entries take one batched float
  matmul per expert over its gathered rows (`matmul_raw`, the matmul on host
  addresses, reaches an expert's matrix inside the stacked tensor).
- `memory_available()`: the host memory the process can still take, the host's available physical memory or less where a cgroup or job object memory limit leaves less (`core/host_memory.hpp`). Weights on the CPU read the mapped file in place, so what counts against it is caches, activations and what a loader materializes. `scratch_reserve(free)` keeps a twentieth of that reported room for page tables, allocator overhead and kernel workspaces; the fit counts this beside its explicit buffers. `reads_in_place()` is true: `adopt` aliases the caller's bytes. `is_cpu()` is true, so experts on the CPU beside it stay on it, and `memory_is_host()` is true, so a server's automatic KV budget does not take the host's memory.
- `identity()` is `cpu`: the same code on every CPU of the baseline, whatever its worker count.
- `row_class(extent)`: two classes, a generated token (extent 1), which takes the decode dots, and every longer extent, which takes the prompt path whose rows compute the same however they are batched.
- `join(members, rows, width)`: a tensor group of CPU backends in this process, joined by its first member and refused over a backend that is not the CPU; each member's partial rows sit in host memory allocated on that member, and `sum_into` reads them in member order on the first member's workers, adding the one sum to every member's residual, eight floats a step and the tail one at a time in the same order; `collective_bytes_per_row` is the member's partial row.
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

Under F16 execution, MXFP4 decode reuses this activation owner with 16-bit packed inputs. Its nibble lookup widens weights directly into integer products; exceptional scale products or nonfinite fast sums use decoded F32 weights and double products over those same activations. The format interpretation and boundary coverage live in [quant-mxfp4](quant-mxfp4.md).

The prompt dots (`q8_dots.hpp`, `dot_block`) sum the products of a group of 32 values exactly in integers, and eight groups' sums then meet their eight scales in one vector multiply-add.

Read-only Q5_0, Q5_1, Q2_K and Q3_K weights use the registry decoder and existing float products: batched dense calls reuse dequantized rows, and routed decode uses the double-dot fallback. Q2_K/Q3_K F16/int8 requests use wider F32 products; BF16 rounds matrix inputs through the existing CPU precision owner. The native Q2_K and Q3_K tests check matmul and routed-product bit identity across extents assigned to the same CPU row class; these operator checks are complete. Q3_K uses the same generic product owner. IQ4_XS CPU support reuses the IQ4_NL lookup loader in `dot_float`, adding eight signed six-bit subscales and 256-value block addressing. Its accumulation body and activation arithmetic are unchanged: F32/retained F16 inputs use float products, BF16 rounds inputs, and int8 requests use the existing wider F32 path. Prompts and embedding/product roles reuse the registry and shared dispatch. The native `iq4-nl` test independently checks both formats, including finite ranges and exact widened-F32 controls. Q2_K/Q3_K read-only CPU/Vulkan support and release qualification are complete; the landing evidence is in STATUS. IQ4_XS read-only support and local qualification are also complete. The [README support matrix](../../README.md#quantization-types) summarizes current CPU/Vulkan inference and writer support.
