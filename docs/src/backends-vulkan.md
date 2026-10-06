# `src/backends/vulkan/` - the Vulkan backend

`dtype_path` describes possible matrix families for the placement record. F16 names packed quantized rows and, where the device profile prefers integer dots, eligible integer tiles; the other products and routers retain F32 inputs. Without that preference the tiles retain F32 inputs. BF16 names rounded matrix inputs and retained F32 routers. The catalog neither changes native capabilities nor records a dispatch. MXFP4 range repair retains the same reconstructed integer inputs, so its wider accumulation remains part of the packed activation class.

Dense and routed tile dispatches and the shared row-dispatch owner record their actual activation forms. The row witness uses the selected kernel's twin, not merely the requested dtype. The BF16 float tile records BF16 after rounding its inputs; the ordinary float tile records F32. The shared `MatrixPaths` evidence is retained by each model stage and consumed through `Model::take_matrix_paths`, without enabling timestamps or shader diagnostics.

The matrix-call dtype is forwarded through dense, grouped, routed, additive and recursive projection dispatch. `native_dtypes` prefers F16, then F32; the F16 policy uses the packed or wider F32 families described above. An explicit F32 request uses the shared row shader with F32 inputs for narrow batches and the float tile for wide batches, bypassing quantized activation twins. A BF16 request takes the float tile's `LLMX_BF16` build, rounding each input once as it enters shared memory with ties to even and NaNs preserved. Weights are unchanged; ordinary products accumulate in F32, while MXFP4 retains its double range repair and split partials; no conversion buffer or extra dispatch is needed. Dense and routed calls choose that build through `float_tile_kernel`, with the existing tile heights and optional float preservation. `emulates_dtype` reports BF16 only when the device preserves F32 denormals, signed zero, infinities and NaNs; placement otherwise selects and reports F32 fallback for that device. BF16 emulation does not add an auto preference.

MXFP4 uses exact packed-weight decoding in the embedding and float tile. Separate shader builds keep other quantized types on their existing modules. The type requires F64 arithmetic, F32 denormal and signed-zero/infinity/NaN preservation, and F64 signed-zero/infinity/NaN preservation before adoption; other types keep their existing device requirements. Unsafe products are recomputed from decoded weights in double, and split partials stay double through reduction. BF16 rounds inputs through the same `matrix_input` function in the tile and recomputation. F32 short calls use the dedicated packed-weight row build. F16 calls use block-int16 rows; on integer-tile profiles, dense prompts copy each eligible projection into a queue-ordered 256 MiB scratch buffer as an exact F32 scale and 32 signed codes per block. Grouped copies retain their original type and packed weight binding. Projections beyond the copy budget and routed prompts take the F32 tile. Both row and tile range repair read exact weights and the same reconstructed int16 inputs as their normal paths; integer split partials remain double through reduction. MXFP4 row dot selection has its own measured profile preference.

The `Backend` of `backend.hpp` over a Vulkan 1.2 compute queue, in one
translation unit (`vulkan_backend.cpp`, built only with
`LLMX_HAS_BACKEND_VULKAN=ON`) and the GLSL kernels under `shaders/`, which
`glslc` compiles at build time into arrays the unit includes. The loader is
opened at run time, so a build carries no link dependency; the design,
kernel notes and measurements are `docs/VULKAN.md`.

The lifetime and packed-quantization tests include the implementation and use test-only friends to inspect private storage and dispatch kernels; there is no runtime probe API. The kernel registry keeps F32 rows on the ordinary row module while selecting optional float-preserving modules for Q8 row consumers. The Q8 integer tile uses its ordinary module: F16-range inputs and binary16 weight scales keep its products above the F32 denormal range. The separate MXFP4 integer tile retains preservation for its wider weight scales. Both entries reuse the row shader source. The ordinary K4 row uses the unpreserved module: its selected F16 input range and binary16 weight scales do not need F32 denormal preservation. Its unused preserving build is removed; the integer-dot K4 variant keeps its separate selection. Quantized weights with an explicit F32 policy use its `LLMX_FLOAT_X` build, optionally float-preserving on a supporting device. It reads one exact decoded weight through `quantized_at` in `shaders/qdecode.glsl`, then uses F32 FMA and the existing row reduction. Dense, grouped and routed calls share this path; the F32 policy binds the original input instead of preparing an integer twin.

- `weight_kernels(type)` is the private descriptor of each implemented storage type: row layout and modules, ordinary and BF16 float tiles, integer tiles, and its dense and routed crossover families. An absent type has no descriptor. `supports_type(type)` reads this owner, with MXFP4 additionally requiring optional double arithmetic and float preservation;
  the model's pre-adoption check and the backend's matrix checks use this
  same query, so they cannot disagree about a weight type.
  `implements` supports the qwen35 layer ops but refuses `Op::mixed_experts`. A grouped gate/up pair with differing storage types is therefore refused at model load; an otherwise eligible streamed layer stays on its capable host. Its separate down projection may use another type.

- `LLMX_VULKAN_KERNELS` is the single ordered list of kernel IDs, diagnostic names and source bindings, including optional preservation modules. It generates `KernelId`, `kKernelNames` and `kKernels`; cache variants depend on the adjacent IDs it preserves. The source dead-code check reads this list and holds its reachability with planted missing-module, orphan-ID and missing-name faults.
  `backend-vulkan` witnesses dense and routed products through the existing timestamp diagnostics, with expected kernel names independent of the descriptor, at each crossover's two sides under F16, F32 and BF16. It also pins unsupported-type refusal text; the existing numerical and row-class checks remain separate. A device without timestamps reports that dispatch-name coverage is unavailable.

- `make_vulkan_backend(index, diagnostics)`, `vulkan_device_name`: open
  the loader, pick the device, require what the kernels need (Vulkan 1.2,
  a compute queue, subgroups of 32 lanes or more whose size divides 256,
  subgroup arithmetic and shuffles, 8- and 16-bit integers, 8- and 16-bit storage buffer access, timeline semaphores, dynamically
  indexed storage buffer arrays, push descriptors); anything missing throws `VulkanUnavailable`, which
  the test skips on and the CLI reports, as does a loader with no driver
  behind it.
  `missing_device_need` holds the features and subgroup properties of that list in one place and names the first one missing.
  `attention_head_fits` takes a head 128 or 256 wide, which the tiled and vector kernels read on any subgroup the device check accepts, and a head of another width up to four elements per subgroup lane and 256 in all (`attention_row_width`), which the per-row kernel reads; `kv_alloc` and `attention` check it, so a 32-lane device refuses a head of another width above 128. The device's `DeviceCaps` choose its `DeviceProfile`
  (`backends/device_profile.hpp`), which `vulkan_device_profile` returns so
  the test predicts the kernel the backend picks.
- `vulkan_kernel_statistics` returns the driver's per-kernel registers,
  shared memory and scratch when the device serves them, which the test
  prints after its checks. With `diagnostics` the backend also captures
  the driver's disassembly of each kernel, which
  `vulkan_kernel_representations` returns and `backend-vulkan --isa DIR`
  writes one file per kernel, then holds each row kernel build to its
  one-column build's counts of float multiplies and adds, each Q8_0
  decode build to the counts its shape and forms give, and a kernel's
  two-row builds to whole copies of one row and column's products apart,
  as the first line of each build's representation states its shape (AGENTS.md, Tests). On a queue that
  timestamps it the backend also times the
  dispatches: `vulkan_kernel_times` returns device milliseconds per kernel
  since the last reading, waiting for the queue, and
  `vulkan_timed_dispatches` how many dispatches that reading covered, every
  one since the reading before, in as many query pools as the interval needs
  (`bench --profile`, `serve --timing`).
  `wait(t)` of a returned ticket may come from another thread while the
  owning one records: the last ticket is atomic and the wait times it
  adds are under a lock, and nothing else of the backend is shared.
  Such a backend also times where it holds its caller (`host_times`): its
  ticket waits, its wait for a free ring slot in `open`, its wait for a
  half of staging in `upload`, and its writes apart from those waits;
  and `device_ms` sums the kernel times since the last reading, which
  `serve --timing` reads each stage's device time from. All of these are
  read-only reporting: nothing in the runtime path depends on them, and a
  backend not opened for diagnostics times nothing.
- Buffers are `VulkanBuffer`, device-local or host-visible, sized in whole
  32-bit words; every op is
  recorded into a ring of 16 command buffers and submitted in chunks of 64
  dispatches, so the device starts a pass while the host records the rest,
  and a host recording several stages ahead onto a busy device waits for a
  free slot only once 16 submissions are in flight.
  A backend asked to hold between submissions (a device of a placement over
  several), while any request to hold remains, follows each `submit()` with a
  submission that waits on an event:
  the next submission sets it, or a watchdog thread, started then, after
  100 ms without one, so the card stays busy, and its clock up, while
  another device runs its stage, and an idle card still idles. The
  backend's own flushes (a chunk, an upload, a read, `sync`) set a pending
  event and add none, and neither does a tensor group's sum, whose
  submission the member's next work follows at once: a group's member is
  held once a stage, after the stage's own `submit()`. The hold's command buffers, timeline, events and
  watchdog are made at the first request; a failure part way destroys what
  it made, so a later request starts again.
  Small per-call inputs go through a host-visible arena per ring slot; a
  scratch outgrown mid-pass retires with the slot rather than being freed
  while recorded commands still name it. Buffer construction cleans up handles
  on failed memory selection, allocation, binding or mapping. Arena overflow
  retains the old buffer before replacing it and leaves its offset reset if
  the replacement allocation fails.
  Padded-cache invalidation reserves retirement capacity before moving entries,
  preserving the cache if allocation fails. Kernel construction owns only
  successful outputs and cleans partial resources before retry; its cached
  entry is published after completion. Diagnostic query pools are destroyed
  after device idle. The [backend audit](../benchmarks/backend-audit-20260925/README.md)
  retains the original failures and probe scope.
- `adopt` copies weights through two staging halves and returns after consuming
  the source, with device copies still ordered on the queue. If a later upload
  chunk fails, it drains the queue before releasing the local destination.
  Its storage is `alloc_weight`'s: device-local, not zero-filled, and marked
  adopted, so a weight a loader allocates that way and writes in pieces keeps
  its eligibility for padded F32 copies as an adopted one does.
  An upload carries on from the staging half the last one left, so consecutive
  writes of one weight overlap host and device copies as one long upload does.
- `join(members, rows, width)` and `VulkanCollective`: a tensor group's sum over Vulkan devices of one kind, each opened once, on a platform that shares device memory as dma-buf and binary semaphores as sync files (`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_semaphore_fd`, enabled where all three are offered, never on Windows), null without them ([TENSOR-SPLIT](../TENSOR-SPLIT.md), section 4.3). Members may sit under different PCI root complexes, whose writes into a peer's memory cost the same on the machines measured; a group whose memory or semaphores cannot be shared is refused, naming each device and its root, and `pci_root()` gives a device's root for the startup report.
  Each member owns an inbox for each peer in each of two parities, exported as a dma-buf and imported into that peer alone (`VulkanBuffer`'s shared constructor, `export_fd`): the kernel orders an importer's submissions behind the other importers' that listed the same dma-buf, so an inbox two peers imported ran those peers in turn at widths of three and more.
  A sum copies each member's partial rows into its slot of every peer's inbox, submits each member's work signalling a binary semaphore a peer (`submit_signalling`), and once every member has submitted imports each as a sync file into the peer, whose next submission waits on it, so no member's work toward a sum waits for another's (`wait_on`, consumed by `flush`), and adds the slots into every member's residual in member order through a copy and `add`, the CPU's order; a parity is written again two sums later, behind the chain of waits.
  It refuses a member that is not Vulkan, a device twice and devices of different kinds, and drains every member before it frees its semaphores and buffers.
- `collective_bytes_per_row(members, width)`: where the device has the exchange, a member's partial and scratch rows and its inbox for each peer in each of two parities, `2 + 2 * (members - 1)` rows of `width` floats, and two rows more among three members or more, its gather rows; a peer's import of an inbox takes none of the importer's memory.
- Two shots: among three members or more a sum of `two_shot_crossover` floats or more (327680, 1.25 MB, where the exchange timed on MI50s first beats broadcast) sends member k only share k of each partial, which k sums in member order and copies into its gather rows on every member, each gather buffer with one importer as an inbox has; every float is the same sum in the same order as broadcast gives, and a smaller sum, a decode step's among them, is broadcast.
- `wrap_host` imports the caller's host memory as device memory
  (`VK_EXT_external_memory_host`, enabled where the device offers it with an
  import alignment) into a buffer that is only a copy source. Memory whose
  address or size is off that alignment, and an import the driver refuses,
  give null, so the caller writes through staging instead. The Radeon VII
  and the MI50s take it at 4096 bytes, one page, so a streamed load's ring
  slots import as they are.
  Padded copies enter their owning cache before their copy command is recorded;
  a replaced copy is retained by the command-buffer slot first.
  A buffer `alloc` zero-fills or `adopt` copies into is held by each command-buffer slot that names it until the slot retires, so a caller may drop it before anything is submitted.
- `check_matrix(type, data, nin, rows, what)`: the one check of a weight operand, made by `check_group` for a matmul and by the routed products and `embed` before they record a dispatch.
  `check_group` checks a matmul call whole, weights, outputs and X, before `matmul_runs` records any of the calls its runs split it into, so a call it refuses records and writes nothing.
  `supports_type`, through the weight descriptor, decides which types have kernels, and a type without one is refused with the unsupported-type error that `tests/common.py` matches, naming the operand as `what`, a matrix or an embedding.
  The operand must then hold `rows` rows sized by `quant::row_bytes`, which refuses a row that ends inside a block.
- `matmul` and `matmul_group` (up to three projections a call, the
  kernels' limit, with a type's block sizes from `quant::Registry`): narrow
  batches take the row kernel, one
  module per family of types, reading quantized rows against an integer
  twin of the activations (`shaders/xquant.glsl`) that the producing
  kernel, the norm, the SiLU or the attention, writes beside its output
  and tags. Under the F16 policy every quantized row and the integer-dot tile read the same
  16-bit twin; explicit F32 rows read the original floats. Each row kernel but the
  Q8_0 decode kernel is built for eight columns and for one
  (specialization constant 0), the one-column build taken when a chunk is
  one wide, except the wide Q8_0 kernel, the Q4 and K-quant families also
  for two rows (below), and a third pipeline is the
  eight-column build grouped by expert (specialization constant 8,
  Mixture of experts below). Integer-dot devices take the Q4 and K-quant
  families' `LLMX_DOT` builds over the same 16-bit twin.
  Q6_K centers its weights with a 16-bit offset so the dot operands stay 16-bit.
  Q4_K and Q5_K rows take at most `k45_row_lanes`. There Q8_0 rows
  take `shaders/matmul_vec_q8.comp`, two-wide 16-bit dots of the weights widened to signed 16-bit pairs against the 16-bit twin (`shaders/dot16.glsl`, which the integer-dot tile shares),
  and F32 rows the plain build.
  The Q8_0 decode kernel is built for 1, 2, 4, 8, 16 and 32 columns (`kVecBuilds`), with the rows a subgroup takes, the steps of weights a lane loads ahead, its two forms and its column groups as specialization constants 9 to 13; `sg_rows` gives the rows a subgroup takes in any row kernel build, which a dispatch's rows per workgroup follow.
  The 32-column build is two 16-column groups over the same rows, and a dispatch gives each workgroup's rows two adjacent workgroups.
  A build takes the transposed reduction where it asks for it and the profile's `q8_decode_forms` allows it, and the half-block order (`kQ8Half`) wherever the profile allows it, in every build alike, since that order sums a column's products in another order than the quarter layout (`vec_forms`); `kernel_representations` starts each build's text with its shape and forms for `backend-vulkan`.
  `for_each_column_chunk` splits a pass's columns: chunks of the widest build the kernel has on the device (the profile's `q8_decode_cols` for this kernel) while more columns remain than it holds, then the rest in the narrowest build that holds them.
  Each Q8_0 decode build holds twice the next narrower's columns, so a chunk fills more than half its build, and a build of one column group checks the column count only before the groups of columns past its first half, while the 32-column build's second group, which gets 1 to 16 columns, checks it before each, and the grouped build, whose runs hold any count, checks it before each group; a build of up to 8 columns also skips the products of the columns past the count in a group the pass fills in part.
  Every build computes a column as the one-column build does, so the split changes only the time; `backend-vulkan` checks each column against the same column alone.
  The Q4 and K-quant families (`row_kernel_builds_two_rows`) also have two-row builds of 2, 4, 8 and 16 columns (`kRowBuilds`, specialization constants 0 and 9, the latter `build_rows`), which a device takes in place of the eight-column build up to its profile's `row_decode_cols` (16 on the MI50 under RADV, none by default): a cluster takes two adjacent rows and computes every column the build holds, so its chunks too fill more than half a build, and `sg_rows` counts the rows a subgroup's clusters take while the push constant keeps the clusters.
  `kernel_representations` starts such a build's text with `; row_build cols=C rows=R`.
  The lanes that share a wide Q8_0 block pair (four) and a K-quant block (eight) are fixed by `matmul_row.comp`, and the host mirrors them in constants beside the tile heights rather than in the profile.
- Wide batches take a tile kernel. Where the profile sets
  `prefer_integer_dot`, every quantized type goes through the
  integer-dot tile (`shaders/matmul_tile_q.comp`, Q6_K in its own module
  `matmul_tile_q6` and Q8_0 in `matmul_tile_q8`) over the row kernels'
  16-bit twin, its quants widened to signed 16-bit pairs (`shaders/dot16.glsl`). When one tile call reads a whole batch next (`tile_reads`),
  the norm, the SiLU or the wide attention that wrote the batch writes
  the twin four values a lane (`xquant_word` in `shaders/xquant.glsl`);
  otherwise `tile_twin` makes it with
  `shaders/quantize_xw.comp` before the call. A layer's projections of one type share one
  dispatch, and a call too small to fill the device splits its inner
  dimension into parts that `shaders/matmul_reduce.comp` adds in order.
  F32 weights, explicit F32 requests, BF16 round/widen requests, and every type on other devices take the float tile
  (`shaders/matmul_tile.comp`).
  Under `--dtype int8` every row of a quantized type but MXFP4 takes the 8-bit build of the kernel f16 gives it (`LLMX_I8`, `WeightKernels::int8_row`, `int8_tile` and `int8_tall_tile`): the row kernels `matmul_row_q4_i8`, `matmul_row_k4_i8`, `matmul_row_k5_i8` and `matmul_row_k_i8`, the Q8_0 decode kernel `matmul_vec_q8_i8` (`is_vec_kernel`, which shares `kVecBuilds`), and the tiles `matmul_tile_qi8`, `matmul_tile_q6i8` and `matmul_tile_q8i8`, each with a preserving variant but the Q8_0 tile, in the same lanes, blocks and order as the 16-bit build, its quants as signed bytes through the four-wide 8-bit dot.
  They read the 8-bit twin (`reads_x8`), which lives after the 16-bit twin in the same scratch at `x8_base_bytes`; `row_twin` and `tile_twin` bind it there, quantizing the input themselves where the tag does not say a producer wrote it (`XqTag::has8`).
  Where the tile reads a producer's output next in a pass of 32 rows or more, the 8-bit twin lies block-major (`x8_major`, `XqTag::major`), which the tile reads through its `xpad` constant and `row_twin` has made again column after column.
  The first such read sets `want_x8_`, after which every producer and quantizer is dispatched in its second build (`twin_variant`, specialization constant 7), which writes both twins; this changes which copies exist, never a result.
  MXFP4 and F32 weights keep their f16 kernels, and `native_dtypes` lists int8 where the profile prefers the integer dot.
  Dense BF16 caps the 32-value stages per K part at `ceil(stages / 4)`, retaining a finer occupancy split, so its F32 sums remain shorter. The float tile's partial buffer has a `kFloatPartBytes` target of 256 MiB: `matmul_group_impl` slices columns into aligned spans, keeping full 64-column tiles where the target holds them, and runs each span's reduction before reusing the buffer. A minimum aligned span can exceed the target. Each output keeps its K parts, order and tile height, and MXFP4 partials stay double. This is a bound on these float partials, not on every temporary buffer or total device memory.
  The row count where the tile starts winning is one of four measured thresholds in `backends/device_profile.hpp` (8-bit or other types, narrower or at least 4096 wide), which `tile_from` takes once per call from the types of the projections that have rows, Q8_0 grouped with F32 and the width split at the profile's `tile_narrow_nin` (4096).
  A mixed-type group on the row kernel becomes a dispatch per type, each kept on the row kernel.
  `tile_rows_for` picks a height of 128, 64 or 32 rows from the workgroups each height gives, the output rows over the height times the column groups, against the profile's workgroups per compute unit and the device's compute units, by the projection's width.
- Batch invariance: with row runs (`backend.hpp` `RowRuns`) a row's
  matmul kernel and split follow its prompt's extent rather than the
  call's width. Attention uses that extent for tiled versus row dispatch
  and each row's length for its history splits. Grouped-head variants
  also depend on the dispatch's longest history while preserving each
  row's arithmetic, so batching must not change a sequence's output.
  `matmul_runs` and `expert_runs` read the runs through `for_each_run`, keyed by kernel and split and by kernel alone. BF16 forces a tile even below the ordinary crossover, so `matmul_runs` also gives those rows the split of their logical extent; collecting more than 64 decode rows cannot change their reduction.
- `rms_norm_rows` spreads a row over several workgroups when the output
  does not overlap the input, with the same tree reduction as one, up to
  four workgroups per compute unit over the pass: each reads the whole row
  for its sum, so a decode row takes one workgroup per 256 values of its width, sixteen at 4096, and a pass of many rows one.
- Mixture of experts: `shaders/moe_route.comp` routes a row per workgroup
  through subgroup reductions. The row kernels and both tile kernels take a
  routed mode (push constant `per`): a row kernel runs one entry per
  workgroup row with its expert's offset on the weight rows, and a tile
  runs one tile of up to 64 entries of one expert from the grouping
  `shaders/moe_group.comp` writes, a workgroup per expert in a stable
  order. Generated tokens with at least twice as many entries as experts
  take the row kernels' grouped pipeline over the same grouping, a run of
  up to eight entries of one expert per workgroup row, so the expert's rows
  are read once per run. A row's entries take the tile when its prompt's extent reaches the
  weight descriptor's selected crossover in `DeviceProfile`; a routed tile is never split, so an entry
  computes the same whatever else is routed beside it. The down
  projection's slots land in scratch and `shaders/moe_combine.comp` adds
  their weighted sum to the residual; it reuses the grouping made for gate and up and reads the twin the SiLU writes for its input, while the router and gate and up share their input's twin.
- The KV cache is `VulkanKVStorage`, blocks of 64 tokens in f32 or f16, written and read through view tables, so every view of a batch goes through one dispatch of each cache kernel.
  It derives from `BlockKVStorage` (`backends-kv_storage.md`), which grows it, keeps its accounting and checks each view as the table is built; its `retire` hands the buffers a growth copied from to `keep_until_retired`, and `kv_alloc` refuses a head `attention_head_fits` does not take.
  Attention can dispatch tiled and row kernels plus a history-split merge in one layer.
  It gives views of 128- or 256-wide heads whose prompt reaches the profile's `attention_tile_rows` to the tiled kernel (`shaders/attention_tile.comp`, 32 query rows a tile as the shader fixes them, its `_d256` builds staging 8 keys a tile where the 128-wide ones stage 16) and the rest to the per-row kernel, which splits a row's history into parts from the row's own length and merges them (`shaders/attention_merge.comp`); once the longest row fills every split, a workgroup takes up to four query heads of one KV head (the `_g4` builds), loading the history once for them with each head's arithmetic unchanged.
  The merge reads each part's maximum and sum once into shared memory, forms its weight once, and loads eight parts' column values before summing them in part order. The profile keeps the number of parts at most 256, the merge's workgroup width.
  Heads 128 wide take `shaders/attention_vec.comp`, which reads a token's row in 16 lanes, one 16-byte load a lane for an f16 side and two for f32, and several tokens a subgroup, and heads 256 wide its `_d256` builds, 32 lanes a token; other widths keep `attention.comp`.
- `kv_variant` picks the shader module for a storage's K and V types.
- `norm_rope_partial`: one workgroup per (row, head) (`shaders/norm_rope_partial.comp`), the head's sum of squares a tree through shared memory, reading the heads at their strides and writing them contiguously; Qwen3 does not reach it on this backend, whose fused `norm_rope_kv` has the same arithmetic per head at the full width.
- The qwen35 layers' ops (docs/QWEN35.md), each checked against the CPU and for its own bit-for-bit rules by `backend-vulkan`:
  - `sigmoid_mul` (`shaders/sigmoid_mul.comp`) and `gated_rms_norm` (`shaders/gated_rms_norm.comp`, one workgroup per (row, head)) write the copy of their output that the matmul reading it next takes, by its row runs, as `silu_mul` and `rms_norm_rows` do: the 16-bit twin, four values a lane where `tile_reads` says one tile call reads the batch.
    So the output gate replaces the copy attention wrote of the ungated output, which its dispatch dropped the tag of.
  - `causal_conv_silu` (`shaders/causal_conv_silu.comp`): one dispatch, an invocation per (chunk of `kConvChunk` rows of a view, channel) walking its rows through a window of the last three raw values; a view's first chunk alone reads the rows it carries in and writes the rows it leaves, after reading them, so no other invocation touches them.
    Each output is the CPU's multiply-add chain in tap order, so it is the same however the view is cut into chunks.
  - `gated_delta_rule` (`shaders/delta_rule.comp`): one dispatch, a workgroup per (view, V head, 32 V columns), eight lanes a column, each holding 16 of its rows in registers.
    For each block of 16 of its view's tokens, or of 4 in its short build (`delta_rule_short`), which a call whose views hold at most 8 rows each takes so that more workgroups fit a compute unit, the workgroup stages the L2-normed q and k of the V head's K head, each V head's beta and decay, 0 below 2^-126, and its columns' v into shared memory, then every column runs the block's tokens.
    Every sum is a lane's multiply-adds in row order, then one butterfly over eight lanes, the same for every token whatever else is in the call, so a sequence gives the same bits alone, beside others and cut into passes; o of a token and m of the next share their butterfly.
    Rows past the K head's width stage as zeros, so the recurrence tests no row.
    It refuses K heads wider than 128.
  - `state_table` lays a call's state views out for the kernels, the conv appending its chunks, and refuses views of two storages in one call; the slots' floats must be addressable in 32 bits.
  - `state_alloc` is `Backend`'s own, built on this backend's `alloc`; `backend-vulkan` checks its zeroed slots.
  - `implements` answers true for these state and gating ops and the drafter's two below; mixed routed projection types remain unsupported as described above.
- The embedded drafter's ops (docs/SPECULATIVE.md, section 7), held to the CPU id for id and bit for bit by `backend-vulkan`:
  - `argmax_rows` (`shaders/argmax_rows.comp`): a workgroup a row, each lane its first largest over its strided ids, then a tree to the lowest id of the largest, with a NaN anywhere, a largest that is not finite or an invalid prior id giving the row the invalid id.
  - `embed_ids` (`shaders/embed.comp` built with `LLMX_DEVICE_IDS`, and with `LLMX_MXFP4` too for an MXFP4 table, as `embed` takes its own build for one): `embed` reading its ids from a device buffer, a zero row for an id past the table.
- `memory_available()`: the device-local heap's budget less its usage from `VK_EXT_memory_budget`, enabled where the device offers it, or the heap's size without it; the small host-mappable device window is skipped. `resident_bytes` adds the padded copy an F32 product matrix whose rows are a multiple of 256 floats gets once a float tile reads it (`padded_f32`), both reading the shape from one rule, `pads_f32`; routed stacks and gathered tables are bound as they are. `host_resident()`: the upload staging buffer and the ring of host-visible arenas, which live in host memory. `scratch_reserve(free)`: 256 MiB plus a twentieth of what is free, with another 256 MiB for the bounded MXFP4 copy on profiles that prefer integer dots. That extra reserve is profile-based even when the model has no MXFP4 weights; float partials and attention merge state use the existing base reserve.
- `decode_columns()`: the narrower of the Q8_0 decode kernel's widest build (`DeviceProfile::q8_decode_cols`) and the other row kernels' (`row_decode_cols`, else the eight-column build), each reading a weight once for its columns.
- `identity()`: the vendor and device ids and name, and the driver's name, information and version, which compiles the kernels.
- `row_class(extent)`: a generated token is a class of its own; a longer extent's class is which of the profile's crossovers it has reached (the matmul tile of both type families at both row widths, the routed tile of every family and the attention tile) and, once it can take the tile, its split (`split_tiles_of`), so every extent from 449 on is one class.

## Finite activation range repair

`xquant.glsl` keeps the activation packing and its scale arithmetic together.
`xquant.glsl` normalizes extreme finite blocks before division, uses a representable scale and restores the scale and scaled sums as bits so gradual underflow does not depend on the producer shader's floating-point mode.
Ordinary blocks keep their existing scale, reciprocal and rounding expressions. A packed table's values and the word-wise input share one ordinary-block check before their vector overload calls the scalar bit-shift routine; the scalar range arithmetic has one implementation.
The packed-twin probe passes on the MI50. `shaders/float_controls.glsl` gives the Q8 row and integer-dot consumer modules 32-bit denormal and signed-zero/infinity/NaN preservation. Device creation queries both properties; only devices reporting both select these modules, and other devices keep their existing modules. The selection changes no kernel layout, column build or decode-order specialization. The preserved Q8 row variants request their fused products explicitly into a precise accumulator, since enabling the modes can disable the driver's implicit contraction and change ordinary logits. The preserved Q8 integer-dot vector variant keeps a precise accumulator and separate scale product, integer-dot product and addition; explicitly fusing those operations failed ordinary-input identity on the MI50. The native packed-activation test also holds these consumers to the ordinary modules' bits on normal inputs across row and column tails; any disagreement fails the gate on a tested driver. The integrated native regression passes on Radeon, including fallback execution with unsupported modes explicitly refused. Radeon and MI50 whole-model identity pass on the pinned four small quants in both cache types, and STATUS records the timing gate. Preserving the packed producer does not imply that every consumer on every device preserves tiny results.
Native tests pass on Windows and Linux, and identity against main holds on the CPU, the Radeon VII and the MI50s for the four small quants and for 8B Q8_0 in both cache types, and for 30B-A3B Q4_K_M once F32 rows kept the ordinary module; STATUS records the gates and the cost.

### Dtype range checks

The F16 implementation is held to binary16's finite input range, including subnormal and normal boundaries and the largest finite input, following PRECISION section 9. Its row and tile kernels carry no branches, extra integer sums or scale-order choices solely for preserving original F32 input range. Q4_0's scaled dot and offset subtraction use an explicit rounding order so zero weights cancel and decode batch widths agree. Q6_K still centers integer weights before its dot. The 16-bit twin holds two bytes per activation and eight bytes per block of 32 for its scale and whole scaled sum, 2.25 bytes per activation in total.

Explicit F32 retains the original extreme-input checks on both rows and prompts, independent unit-weight and zero-row fixtures, all six matrix calls, negative scales and cancelling minima, three half weight scales and dispatch witnesses. The optional float-preservation modules remain selected through the existing owner; unsupported devices take their ordinary modules. Tests of the activation representation itself still cover its wider input range. These unit checks complement the independent HF comparisons and matched performance gates described in [PRECISION](../PRECISION.md).
