# `src/backends/vulkan/` - the Vulkan backend

`dtype_path` describes possible matrix families for the placement record. F16 names packed quantized rows and, where the device profile prefers integer dots, eligible integer tiles; the other products and routers retain F32 inputs. Each weight descriptor owns its row profile preference and kernel choices; integer tiles use the device's existing `prefer_integer_dot` preference, with F32 tiles otherwise. BF16 names rounded matrix inputs and retained F32 routers. The catalog neither changes native capabilities nor records a dispatch. MXFP4 range repair retains the same reconstructed integer inputs, so its wider accumulation remains part of the packed activation class.

Dense and routed tile dispatches and the shared row-dispatch owner record their actual activation forms. The row witness uses the selected kernel's twin, not merely the requested dtype. The BF16 float tile records BF16 after rounding its inputs; the ordinary float tile records F32. The shared `MatrixPaths` evidence is retained by each model stage and consumed through `Model::take_matrix_paths`, without enabling timestamps or shader diagnostics.

The matrix-call dtype is forwarded through dense, grouped, routed, additive and recursive projection dispatch. `native_dtypes` prefers F16, then F32; the F16 policy uses the packed or wider F32 families described above. An explicit F32 request uses the shared row shader with F32 inputs for narrow batches and the float tile for wide batches, bypassing quantized activation twins. A BF16 request takes the float tile's `LLMX_BF16` build, rounding each input once as it enters shared memory with ties to even and NaNs preserved. Weights are unchanged; ordinary products accumulate in F32, while MXFP4 retains its double range repair and split partials; no conversion buffer or extra dispatch is needed. Dense and routed calls choose that build through `float_tile_kernel`, with the existing tile heights and optional float preservation. `emulates_dtype` reports BF16 only when the device preserves F32 denormals, signed zero, infinities and NaNs; placement otherwise selects and reports F32 fallback for that device. BF16 emulation does not add an auto preference.

IQ4_NL uses `iq4_nl_at` in `shaders/qdecode.glsl` for its signed lookup and binary16 scale, keeping signed zero explicitly. The embedding and float tile use that decoder. The weight descriptor selects `LLMX_IQ4` builds of the row family under `lookup_integer_dot` and signed-byte Q8 tiles under the existing `prefer_integer_dot` tile policy: signed lookup bytes are widened through `dot16.glsl` and dotted with the block-int16 activation twin. One block sums at most 32 * 127 * 32767 = 133165088 in magnitude, within int32; conversion and scaling accumulate in F32. F16 and int8 use these 16-bit row and tile paths, with a `block-int16` witness; int8 has no narrower IQ4 build. F32 retains original inputs, BF16 its input-rounding tile, and other row or tile profiles use wider F32. Radeon VII retains faster F32 tiles after the integer-tile screen. Unsafe combined scales and nonfinite row sums or tile partials trigger ordered F32 recomputation from exact weights and the same reconstructed int16 inputs, shared in `shaders/iq4_dot.glsl`. Split parts retain their F32 reduction. The tile specializes lookup staging and range checks; its widened signed-byte storage, dot loop, heights, grouping and split dispatch remain the Q8 family's. Separate compilation leaves the existing Q8 modules unchanged. This retains F32 range limits: cancellation with overflowing individual F32 terms is not rescued by double arithmetic. The native `iq4-nl-vulkan` checks independent storage, activation/product and cancellation oracles, actual row/grouped/head/small/short/tall tile witnesses, split and tail cases, and shared row-class identity. IQ4_NL row and tile functional checks pass on MI50 and Radeon VII. Full-model MI50 timings include IQ4_NL prefill and decode, but no isolated MI50 tile benchmark; remaining speed followups are recorded in STATUS.

MXFP4 uses exact packed-weight decoding in the embedding and float tile. Separate shader builds keep other quantized types on their existing modules. The type requires F64 arithmetic, F32 denormal and signed-zero/infinity/NaN preservation, and F64 signed-zero/infinity/NaN preservation before adoption; other types keep their existing device requirements. Unsafe products are recomputed from decoded weights in double, and split partials stay double through reduction. BF16 rounds inputs through the same `matrix_input` function in the tile and recomputation. F32 short calls use the dedicated packed-weight row build. F16 calls use block-int16 rows; on integer-tile profiles, dense prompts copy each eligible projection into a queue-ordered 256 MiB scratch buffer as an exact F32 scale and 32 signed codes per block. Grouped copies retain their original type and packed weight binding. Projections beyond the copy budget and routed prompts take the F32 tile. Both row and tile range repair read exact weights and the same reconstructed int16 inputs as their normal paths; integer split partials remain double through reduction. MXFP4 row dot selection has its own measured profile preference.

F16/BF16 weights keep their 16-bit storage through adoption. Their separate builds of the existing embedding, row and float tile shaders widen with the integer-bit helpers in `shaders/f16.glsl`; no new model or loader representation is needed. Host-ID and device-ID embedding share these conversions, and an invalid device ID writes a zero row before reading any weight. The float row kernels share F32's loads-in-fours, accumulation order, row layout and dispatch thresholds. Their plain row modules inherit F32's float controls; tiles use the existing optional preservation modules. Exact storage widening does not add a stronger guarantee for subnormal products than the corresponding F32 path. Explicit BF16 activation requests use the same tile input-rounding owner as F32 weights. Dense, grouped, routed, residual and output-head products select these modules through the weight descriptor. Half rows remain compact; power-of-two padding is a separate measured decision.

The `Backend` of `backend.hpp` over a Vulkan 1.2 compute queue, in one
translation unit (`vulkan_backend.cpp`, built only with
`LLMX_HAS_BACKEND_VULKAN=ON`) and the GLSL kernels under `shaders/`, which
`glslc` compiles at build time into arrays the unit includes. The loader is
opened at run time, so a build carries no link dependency; the design,
kernel notes and measurements are `docs/VULKAN.md`.

The lifetime and packed-quantization tests include the implementation and use test-only friends to inspect private storage and dispatch kernels; there is no runtime probe API. The kernel registry keeps F32 rows on the ordinary row module while selecting optional float-preserving modules for Q8 row consumers. The Q8 integer tile uses its ordinary module: F16-range inputs and binary16 weight scales keep its products above the F32 denormal range. The separate MXFP4 integer tile retains preservation for its wider weight scales. Both entries reuse the row shader source. The ordinary K4 row uses the unpreserved module: its selected F16 input range and binary16 weight scales do not need F32 denormal preservation. Its unused preserving build is removed; the integer-dot K4 variant keeps its separate selection. Quantized weights with an explicit F32 policy use its `LLMX_FLOAT_X` build, optionally float-preserving on a supporting device. It reads one exact decoded weight through `quantized_at` in `shaders/qdecode.glsl`, then uses F32 FMA and the existing row reduction. Dense, grouped and routed calls share this path; the F32 policy binds the original input instead of preparing an integer twin.

- `weight_kernels(type)` is the private descriptor of each implemented storage type: host-ID and device-ID embedding modules, row layout and modules, ordinary and BF16 float tiles, integer tiles, row profile preferences, and its dense and routed crossover families. An absent type has no descriptor. `supports_type(type)` reads this owner, with MXFP4 additionally requiring optional double arithmetic and float preservation;
  the model's pre-adoption check and the backend's matrix checks use this
  same query, so they cannot disagree about a weight type.
  `implements` supports the qwen35 layer ops but refuses `Op::mixed_experts`. A grouped gate/up pair with differing storage types is therefore refused at model load; an otherwise eligible streamed layer stays on its capable host. Its separate down projection may use another type.

- `LLMX_VULKAN_KERNELS` is the single ordered list of kernel IDs, diagnostic names and source bindings, including optional preservation modules. It generates `KernelId`, `kKernelNames` and `kKernels`; cache variants depend on the adjacent IDs it preserves. The source dead-code check reads this list and holds its reachability with planted missing-module, orphan-ID and missing-name faults.
  `backend-vulkan` checks dense and routed dispatches through the existing timestamp diagnostics, with expected kernel names independent of the descriptor, at each crossover's two sides under F16, F32 and BF16. It also pins unsupported-type refusal text; the existing numerical and row-class checks remain separate. A device without timestamps reports that dispatch-name coverage is unavailable.

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
  `matmul_logits` is `matmul`; on a backend that times its dispatches it names that call's apart, under the kernel's name with ` head` after it, so a profile shows the output head's read on its own, and without timestamps it calls `matmul` and keeps nothing.
  `wait(t)` of a returned ticket may come from another thread while the
  owning one records: the last ticket is atomic and the wait times it
  adds are under a lock, and nothing else of the backend is shared.
  `done(t)` reads the timeline semaphore with a wait of no time, so it never blocks, and `VulkanCollective::wait(back)` waits each member's ticket of the last sum or of the one before it, which the collective keeps as it submits them.
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
- `join(members, rows, width)` and `VulkanCollective`: a tensor group's sum over Vulkan devices of one kind, each opened once, on a platform that shares device memory as dma-buf and binary semaphores as sync files (`VK_KHR_external_memory_fd`, `VK_EXT_external_memory_dma_buf`, `VK_KHR_external_semaphore_fd`, and `VK_EXT_queue_family_foreign`, enabled where all four are offered, never on Windows), null without them ([TENSOR-SPLIT](../TENSOR-SPLIT.md), section 4.3). Members may sit under different PCI root complexes, whose writes into a peer's memory cost the same on the machines measured; a group whose memory or semaphores cannot be shared is refused, naming each device and its root, and `pci_root()` gives a device's root for the startup report.
  Each member owns an inbox for each peer in each of two parities, exported as a dma-buf and imported into that peer alone (`VulkanBuffer`'s shared constructor, `export_fd`): the kernel orders an importer's submissions behind the other importers' that listed the same dma-buf, so an inbox two peers imported ran those peers in turn at widths of three and more.
  For each exchange, the producer releases the copied inbox range to `VK_QUEUE_FAMILY_FOREIGN_EXT` before signaling; after its semaphore wait, the consumer acquires the matching range before reading. The semaphore orders submissions but does not itself transfer ownership between separate Vulkan instances on different physical devices (TENSOR-SPLIT, section 4.3).
  Adjacent inbox acquires share one pipeline-barrier command per member in a one-shot sum and the first exchange of a two-shot sum. Their buffer barriers keep peer order, exact ranges and masks, using recording-thread storage sized once with the collective; all spans are checked before recording the batch. Releases and the final gather acquires remain separate commands. The [six-arm timing record](../benchmarks/vulkan-ownership-sixarm-20261018.json) retains the matched original and alignment-control results; the accepted modest cost and its limits are recorded in STATUS.
  A sum copies each member's partial rows into its slot of every peer's inbox, submits each member's work signalling a binary semaphore a peer, and once every member has submitted imports each as a sync file into the peer, whose next submission waits on it, so no member's work toward a sum waits for another's (`wait_on`, consumed by the next submission), and adds the slots into every member's residual in member order through a copy and `add`, the CPU's order; a parity is written again two sums later, behind the chain of waits.
  The members' submissions of a sum run side by side: `flush` is `prepare`, `Submission::queue` and `queued`, and the recording thread ends every member's command buffer and takes its ticket (`prepare`), the queue calls and the exports of the sync files then run on a `core::JobThreads` of one thread a member but the first, the first member's on the recording thread, in a group of three or four (between two members the calls side by side gained nothing for the cores they took, so a group of two has no thread and makes them in turn), and the recording thread takes every queue back (`queued`), imports the files and only then throws what a member's call failed with, closing the files no import took.
  A member's thread calls its device's queue and its semaphores' exports and nothing of the backend's own state; between `prepare` and `queued` the backend refuses to record or submit (`lent_`), which is what holds the recording rule (`backends/backend.hpp`) there.
  The threads are made with the collective, sleep between sums and use no CPU while a group is idle; on fewer free cores than members the recording thread takes the calls no thread has woken for, so a sum runs toward the serial order and never waits for a sleeping thread.
  It refuses a member that is not Vulkan, a device twice and devices of different kinds, and stops its threads, then drains every member, before it frees its semaphores and buffers; a context holds its collectives and goes before the model's backends.
- `collective_bytes_per_row(members, width)`: where the device has the exchange, a member's partial and scratch rows and its inbox for each peer in each of two parities, `2 + 2 * (members - 1)` rows of `width` floats, and two rows more among three members or more, its gather rows; a peer's import of an inbox takes none of the importer's memory.
- Two shots, whose shares are the equal ones rounded up to the members' storage-buffer offset alignment (`share_begin`, `share_size`), since each share is bound at its own offset: among three members or more a sum of `two_shot_crossover` floats or more (327680, 1.25 MB, where the exchange timed on MI50s first beats broadcast) sends member k only share k of each partial, which k sums in member order and copies into its gather rows on every member, each gather buffer with one importer as an inbox has; every float is the same sum in the same order as broadcast gives, and a smaller sum, a decode step's among them, is broadcast.
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
  `twin` reuses a block-major 8-bit copy only when its blocks per input row match the next reader's layout; changing the row width repacks it. A column-major copy remains reusable at any compatible row width, and a 16-bit reader does not depend on the 8-bit layout.
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
  - `state_table` lays a call's state views out for the kernels, the conv appending its chunks, and refuses views of two storages in one call; the addressed slot count uses the active layer's full stride, including an optional F32 tail, and must fit 32-bit shader indexing. The conv and delta shaders index only the common GDN prefix and leave such a tail unchanged.
  - `state_alloc` is `Backend`'s own, built on this backend's `alloc`; it allocates each layer at its own stride and zeroes every slot, including any tail.
  - `implements` answers true for these state and gating ops and the drafter's two below; mixed routed projection types remain unsupported as described above.
- The PLE convolution primitive `dilated_conv_silu_add` (`shaders/dilated_conv_silu_add.comp`) dispatches one invocation per channel and view chunk. The first chunk reads and updates the layer's `(taps - 1) * dilation` raw-row F32 tail after the common prefix; later chunks use rows produced earlier in the same call. It applies the causal convolution, SiLU and residual add in tap order, and checks shader-index and alias limits before dispatch.
- `select_kv` uses `kv_index_pool`, `kv_index_scores` and `kv_index_select`. It pools one view at a time from raw auxiliary F32 keys, applies the stored GGUF RMSNorm multiplier and first-position split-half RoPE, and scores at most 16 query rows at once. Eight radix passes locate the score cutoff; the chronological scan selects earlier groups first on exact ties, emits selected token IDs in ascending order, appends a visible partial group and pads with `UINT32_MAX`. The selector accepts index dimensions through 256 and preflights geometry, views, storage, spans and aliases before dispatch. At the released geometry (262144 positions, index width 128, compression 4) its pooled-key and score buffers, 32 MiB and 4 MiB, fit within the 256 MiB base scratch reserve. It computes in F32 throughout.
- `sparse_attention` reuses `attention.comp` and `attention_vec.comp` built with `LLMX_SPARSE`, selecting the F16/F32 K/V variant for each cache side. It launches one workgroup per query row and head, reads the supplied IDs, and performs online softmax directly over selected original KV rows. The 128-wide vector path and 256-wide `LLMX_D256` vector path are available; this sparse dispatch uses one head per workgroup and one unsplit pass, without the dense path's history split/merge, tiled prompt or four-head G4 dispatch. Empty selections write zero. `tests/sparse_attention.cpp` holds it to the same double-precision oracle as the CPU's.
- The embedded drafter's ops (docs/SPECULATIVE.md, section 7), held to the CPU id for id and bit for bit by `backend-vulkan`:
  - `argmax_rows` (`shaders/argmax_rows.comp`): each lane its first largest over its strided ids, then a tree to the lowest id of the largest, with a NaN anywhere, a largest that is not finite or an invalid prior id giving the row the invalid id. A row of more than `kVulkanArgmaxSlice` (4096) ids is reduced in two dispatches: a workgroup a slice of ids writes a partial (the value, the id, whether a NaN was seen) into a scratch of the call's, then a workgroup a row reduces the row's partials in slice order by the same comparison and applies the row's rules once, so the id is the one a single workgroup gives; a shorter row, or a call recorded without barriers, whose second dispatch could not follow its first, takes one workgroup a row.
  - `embed_ids` (`shaders/embed.comp` built with `LLMX_DEVICE_IDS` and the weight-format define selected through the weight descriptor): `embed` reading its ids from a device buffer, a zero row for an id past the table.
- `memory_available()`: the device-local heap's budget less its usage from `VK_EXT_memory_budget`, enabled where the device offers it, or the heap's size without it; the small host-mappable device window is skipped. `resident_bytes` adds the padded copy an F32 product matrix whose rows are a multiple of 256 floats gets once a float tile reads it (`padded_f32`), both reading the shape from one rule, `pads_f32`; routed stacks and gathered tables are bound as they are. `host_resident()`: the upload staging buffer and the ring of host-visible arenas, which live in host memory. `memory_is_host()`: true for an integrated GPU or a CPU device by its type (`host_memory_device`), whose device-local heap is the host's memory, so a server's automatic KV budget keeps the model's context there; a discrete GPU's memory is its own, whatever window the host can map. `scratch_reserve(free)`: 256 MiB plus a twentieth of what is free, with another 256 MiB for the bounded MXFP4 copy on profiles that prefer integer dots. That extra reserve is profile-based even when the model has no MXFP4 weights; float partials and attention merge state use the existing base reserve.
- `decode_columns()`: the narrower of the Q8_0 decode kernel's widest build (`DeviceProfile::q8_decode_cols`) and the other row kernels' (`row_decode_cols`, else the eight-column build), each reading a weight once for its columns.
- `identity()`: the vendor and device ids and name, and the driver's name, information and version, which compiles the kernels.
- `row_class(extent)`: a generated token is a class of its own; a longer extent's class is which of the profile's crossovers it has reached (the matmul tile of both type families at both row widths, the routed tile of every family and the attention tile) and, once it can take the tile, its split (`split_tiles_of`), so every extent from 449 on is one class.

## Decode builds

The Q8_0 row module describes each pipeline with `VecBuild`: columns per
subgroup, rows, load steps, reduction forms and adjacent column groups.
Variants 0, 1 and 2 are the wide, one-column and grouped builds; variants
3 through 6 hold 2, 4, 16 and 32 columns respectively, the last using two
groups of 16. Specialization constants 0, 9, 10, 11 and 13 carry these
properties. Constant 12 selects half-block order, fixed by the device profile
for every build; the transposed reduction is allowed only where the subgroup
uses its expected reduction pairs. These choices preserve a column's bits
across build widths.

The Q4 and K-quant row families use constants 0 and 9 for columns per lane
and rows per cluster. Their first three variants are the one-row wide,
one-column and grouped builds, followed by two-row builds with 2, 4, 8 and
16 columns. Constant 8 marks grouped execution. All pipelines receive the
shared specialization entries; a module ignores those it does not declare.

`for_each_column_chunk` fills the widest supported build while more columns remain
than it holds, then chooses the narrowest build that holds the remainder.
Successive Q8_0 capacities double, so a final chunk fills more than half its
build. A single-group build checks columns beyond that first half; a
two-group build checks each group's count. Two-row Q4 and K-quant builds
compute every column they hold and use the same half-fill rule.

## Instruction-count checks

`tests/backend_vulkan.cpp` checks the driver disassembly when `--isa DIR`
is requested. It counts ordinary multiplies, rounded multiply-adds, fused
products, ordinary adds and shuffled adds. RADV's integer address division
uses a reciprocal scaled by `0x4f7ffffe`; its recognized normalization and
restore multiplies are excluded from matrix product counts. Planted
instruction sequences hold this exclusion to the matching register chain.

The Q8_0 expected counts follow each build's recorded shape. Products repeat
for its load steps, the single-step tail after paired steps, partially filled
column groups, and the half-block-order path for even block counts. Each
copy retains the one-column build's ratio of fused to unfused products.
Without transposed reduction, each row and column uses the baseline
reduction and one residual add. With it, R rows and C columns use
`R * C - 1` pairwise adds and one add for each reduction level above
the base-two logarithm of `R * C`, plus one residual add per 64 values; the driver may express
those reduction adds as shuffled or ordinary instructions.

Other row builds must retain the baseline counts or whole additional copies
of one column's products, depending on whether the compiler unrolls the
column loop. Two-row builds keep whole copies and the baseline fusion ratio.
These checks detect count changes, not every reassociation: the separate
decode-column identity checks hold results across widths and tails, as
described in AGENTS.md, Tests.

## Finite activation range repair

`xquant.glsl` keeps the activation packing and its scale arithmetic together.
`xquant.glsl` normalizes extreme finite blocks before division, uses a representable scale and restores the scale and scaled sums as bits so gradual underflow does not depend on the producer shader's floating-point mode.
Ordinary blocks keep their existing scale, reciprocal and rounding expressions. A packed table's values and the word-wise input share one ordinary-block check before their vector overload calls the scalar bit-shift routine; the scalar range arithmetic has one implementation.
The packed-twin probe passes on the MI50. `shaders/float_controls.glsl` gives the Q8 row and integer-dot consumer modules 32-bit denormal and signed-zero/infinity/NaN preservation. Device creation queries both properties; only devices reporting both select these modules, and other devices keep their existing modules. The selection changes no kernel layout, column build or decode-order specialization. The preserved Q8 row variants request their fused products explicitly into a precise accumulator, since enabling the modes can disable the driver's implicit contraction and change ordinary logits. The preserved Q8 integer-dot vector variant keeps a precise accumulator and separate scale product, integer-dot product and addition; explicitly fusing those operations failed ordinary-input identity on the MI50. The native packed-activation test also holds these consumers to the ordinary modules' bits on normal inputs across row and column tails; any disagreement fails the gate on a tested driver. The integrated native regression passes on Radeon, including fallback execution with unsupported modes explicitly refused. Radeon and MI50 whole-model identity pass on the pinned four small quants in both cache types, and STATUS records the timing gate. Preserving the packed producer does not imply that every consumer on every device preserves tiny results.
Native tests pass on Windows and Linux, and identity against main holds on the CPU, the Radeon VII and the MI50s for the four small quants and for 8B Q8_0 in both cache types, and for 30B-A3B Q4_K_M once F32 rows kept the ordinary module; STATUS records the gates and the cost.

### Dtype range checks

The F16 implementation is held to binary16's finite input range, including subnormal and normal boundaries and the largest finite input, following PRECISION section 9. Its row and tile kernels carry no branches, extra integer sums or scale-order choices solely for preserving original F32 input range. Q4_0's scaled dot and offset subtraction use an explicit rounding order so zero weights cancel and decode batch widths agree. Q6_K still centers integer weights before its dot. The 16-bit twin holds two bytes per activation and eight bytes per block of 32 for its scale and whole scaled sum, 2.25 bytes per activation in total.

Explicit F32 retains the original extreme-input checks on both rows and prompts, independent unit-weight and zero-row fixtures, all six matrix calls, negative scales and cancelling minima, three half weight scales and dispatch witnesses. The optional float-preservation modules remain selected through the existing owner; unsupported devices take their ordinary modules. Tests of the activation representation itself still cover its wider input range. These unit checks complement the independent HF comparisons and matched performance gates described in [PRECISION](../PRECISION.md).

## Shader notes

The design notes the shaders' own comments held before they were cut to a sentence or two, kept as the kernels' authors wrote them; each shader's comment names the part that matters at the code and points here.
The kernels' measurements are in `docs/VULKAN.md`.

### `src/backends/vulkan/shaders/attention.comp`

From the comment at line 2:

- Causal GQA over every view of a batch: one workgroup per (batch row, block of heads, split), a split being a range of the row's history and a block up to four query heads that read one KV head.
- The workgroup loads each token's key and value once for its block, since a long history's cache outweighs the weights a decode step reads.
- Each head's arithmetic is what it would be alone: the split's tokens go round robin to the subgroups with an online softmax per head, and the subgroups merge through shared memory one head at a time; with several splits each state goes to scratch for attention_merge.
- Row b of a view attends through its hist + b.

### `src/backends/vulkan/shaders/attention_merge.comp`

From the comment at line 2:

- Merge the split attention states of one (row, head): the maximum and sum across splits, then the accumulation rescaled to them; a split with no tokens carries -inf and zero.
- Each split's maximum and sum are read once into shared memory and its weight formed once, so a lane's column sum reads only independent loads; the sums run in split order, as one lane reading every split would take them.
- The row is found through the view table, since a dispatch's rows need not be a prefix of the batch.
- With `quant` set the output's 16-bit twin is written beside it.

### `src/backends/vulkan/shaders/attention_tile.comp`

From the comment at line 2:

- Causal GQA for a wide pass over every view of a batch: one workgroup per (tile of 32 query rows of one view, head), the head's K and V streamed through shared memory in tiles of TK tokens.
- Row b of a view attends through its hist + b; tokens past that are masked and later tiles are not read.
- The head width is fixed at 128, or at 256 in the LLMX_D256 build, whose K and V tiles hold 8 tokens so both fit 16 KiB of shared memory; other widths take the per-row kernel.
- Workgroups are laid out view by view, so a tile never straddles two views.
- Eight lanes share a query row, lane l owning dimensions 32k + 4l .. 32k + 4l + 3, so a row reads a staged token as contiguous 128-byte runs.
- A staged token is DIM / 4 words and lane l's word of run k is 8k + l; with the lane taken by a mask the driver folds each word's address into its read.
- TK sets where the online softmax rescales, so it is a constant of the head width, never of the batch.
- Per K tile: partial dots eight tokens at a time, a butterfly that finishes the sums and deals one to each lane, then one online-softmax step per tile, and the probabilities through shared memory for the weighted sum of V.

### `src/backends/vulkan/shaders/attention_vec.comp`

From the comment at line 2:

- Causal GQA for rows that attend one at a time (decode and short prompts), as attention.comp computes it, for heads 128 wide, or 256 wide in the LLMX_D256 build.
- One workgroup per (batch row, block of heads, split), a block being up to four query heads that read one KV head (LLMX_G4) or one head.
- A token's key and value are read by 16 lanes, or 32 at width 256, eight values each in one load, so a subgroup reads several tokens at once; each such group of lanes keeps its own online softmax over the tokens it takes, and the groups merge in a fixed order at the end.
- The parts a row's history splits into follow from its length alone, as in attention.comp, so a row computes the same whatever else is in the dispatch.

### `src/backends/vulkan/shaders/causal_conv_silu.comp`

From the comment at line 2:

- The linear attention's causal conv of four taps, then SiLU, over every view of a batch (backend.hpp, causal_conv_silu), in one dispatch.
- An invocation takes one channel of a chunk of up to `chunk` rows of one view, the workgroups' second index naming the chunk, and walks the rows through a window of the last three raw values: out[row][c] = silu(sum over tap j of w[c][j] raw[t - 3 + j][c]).
- A view's first chunk reads the rows before it from slot src's carried rows, rows before the sequence's start being zero, and leaves the view's last three raw rows in slot dst once it has read them, so no other invocation touches a view's carried rows.
- The sum is one fused multiply-add chain in tap order, the CPU's, so a row computes the same however its view is cut into chunks and whatever else is in the call.
- The table (vulkan_backend.cpp, state_table) holds the view count, then per view its first batch row, rows, history length, source slot and destination slot, then per chunk its view and first row within it.

### `src/backends/vulkan/shaders/delta_rule.comp`

From the comment at line 2:

- The gated delta rule (backend.hpp, gated_delta_rule; docs/QWEN35.md, Linear attention): the L2 norms of q and k, the gates, and the per-token recurrence.
- One workgroup per (view, V head j, block of 32 V columns): eight consecutive lanes hold one column, each of them rows part, part + 8, ... of the head's k_dim x v_dim matrix in registers, read from slot src (zero at length 0) and written to slot dst.
- The view's tokens go in blocks of TB: the whole workgroup reads a block's q and k rows of K head j mod k_heads, norms them, and makes each token's beta and decay and stages its v values into shared memory, then every column runs the block's tokens.
- Per token, in HF's order: the matrix decayed, m from the decayed matrix, d = beta (v - m), the rank-1 write, then o from the updated matrix; o of a token and m of the next are both sums over the matrix after the token, so their butterflies run together, each the sum it would be alone.
- Every sum is a lane's fused multiply-adds in row order, then one butterfly over eight lanes, the same arithmetic for every row whatever else is in the call, so a sequence gives the same bits alone, batched or cut into passes; the decay is 0 below 2^-126, so no driver's denormal handling decides it.
- The view table (vulkan_backend.cpp, state_table) holds the view count, then per view its first batch row, rows, history length, source slot and destination slot.

### `src/backends/vulkan/shaders/gated_rms_norm.comp`

From the comment at line 2:

- One workgroup per (row, head): dst = x * rsqrt(mean(x^2) + eps) * w * gate(z) over the head's dim values, w shared by every head. The uniform `NormGate` selector chooses SiLU (default) or sigmoid in the same dispatch and activation-copy path.
- The sum of squares is a lane's fused multiply-adds, then a tree through shared memory in a fixed order, so a head computes the same in every build and beside any other rows.
- With `quant` 1, dim is a multiple of 32 and the output's 16-bit twin is written beside it (xquant.glsl); with `quant` 2 the integer-dot tile reads the rows next, and a lane writes four values of the twin at a time.
- dst may be x: every value is read for the sum before the barrier, and after it each lane reads and writes only its own values.

### `src/backends/vulkan/shaders/matmul_row.comp`

From the comment at line 2:

- Y[b*nout + o] = dot(row o, X column b), the decode kernel, for up to three projections of one X in one dispatch (a layer's q, k and v, or gate and up).
- Workgroups go to projections in order, so a workgroup's projection and buffers are uniform.
- A subgroup takes `rows_per_sg` clusters of `cluster` lanes, each a row (ROWS rows in a two-row build) whose work units its lanes stride over, reducing with xor shuffles; a lane keeps up to MAX_COLS batch columns in registers.
- Quantized rows meet the activations as a signed 16-bit twin in blocks of 32 (xquant.glsl) with the block's scale and scaled sums; Q4 offsets and K-quant minima use the block sum; Q6_K's -32 is applied before its integer dot. F32 rows keep float activations. LLMX_FLOAT_X also reads quantized weights against the original F32 activations, using the shared qdecode.glsl values and this kernel's reduction.
- The 8-bit builds (LLMX_I8, with LLMX_DOT), --dtype int8 of docs/PRECISION.md, read the 8-bit twin instead (xquant8_block) through the four-wide 8-bit dot, Q6_K's -32 taken into each weight byte; blocks, lanes, terms and reduction are the 16-bit builds'.
- One module per family of types, since one module for all of them raised every path's registers: as is for F32 and narrow Q8_0, LLMX_Q8W wide Q8_0, LLMX_Q4 Q4_0 and Q4_1, LLMX_Q4+LLMX_Q5 Q5_1, LLMX_K4 Q4_K, LLMX_K5 Q5_K, LLMX_KQUANT Q6_K.
- Wide Q8_0: four lanes share a 68-byte block pair, lane t loading words t, t + 4, t + 8, t + 12 and lane 0 word 16; block 0's quants start two bytes into its words, so each lane shifts them by a half word with the next lane's word.
- Q4_0 with an even block count: two lanes share a 36-byte pair.
- Q4_1/Q5_1: one lane per 20/24-byte affine block; Q5_1 adds a contiguous high-bit plane.
- Q6_K: 210-byte blocks, eight lanes a block, every other block two bytes off word alignment.
- Measurements behind these layouts are in docs/VULKAN.md.

From the comment at line 81:

- A grouped routed call, the wide build's third pipeline (GROUPED, specialization constant 8), runs one chunk of the grouping (moe_group.comp) per workgroup row y: up to MAX_COLS entries of one expert, whose rows are then read once for all of them.
- As a specialization it costs the plain and routed pipelines nothing.

From the comment at line 85:

- Rows a cluster takes, specialization constant 9: the Q4 and K-quant families' two-row decode builds (the backend's kRowBuilds) give a lane two rows' weights, so each column's activation loads serve both, and compute every column they hold, one past the pass's count on its last column and never stored.
- A row's lanes, blocks, products and reduction are those of the one-row builds, and each block's weights and float terms come from the same macros below, so a column gets the one-column build's bits (docs/VULKAN.md, batch invariance).

From the comment at line 208:

- The sub-scales and sub-mins of groups 2c and 2c + 1 from the twelve packed bytes in three words (quant/k_quants.hpp): groups 0..3 are the low six bits of bytes 0..3 (scales) and 4..7 (mins), groups 4..7 the nibbles of bytes 8..11 with the earlier bytes' top two bits above.
- Branch-free, since the lanes of a block hold different c.

### `src/backends/vulkan/shaders/matmul_tile.comp`

From the comment at line 2:

- Y[b*nout + o] = dot(row o, X column b) for wide batches: a workgroup computes a TILE_ROWS x TILE_COLS output tile, walking the inner dimension one stage at a time.
- Each step stages the dequantized W tile and the X tile in shared memory, and every thread accumulates a MROWS x 4 micro-tile in registers.
- TILE_ROWS is a specialization constant: a taller tile reads less shared memory per product but yields fewer workgroups, so the backend picks it per call.
- Rows past nout and columns past nbatch read as zero and are not stored.
- A call of few workgroups splits the inner dimension: workgroup row y is part y / ctiles of column tile y % ctiles, summing `kper` values from the part's start into its own slice of Y, which matmul_reduce.comp then adds in part order.

From the comment at line 32:

- A routed call (per != 0, backend.hpp matmul_experts) runs workgroup row y as tile y of the grouping (moe_group.comp): up to TILE_COLS entries of one expert, column c being entry tab[order0 + first + c], which reads X column entry / per through the expert's rows and writes output row entry.
- A routed call is never split.

### `src/backends/vulkan/shaders/matmul_tile_q.comp`

From the comment at line 2:

- Y[b*nout + o] = dot(row o, X column b) for wide batches through the two-wide 16-bit integer dot, on devices where the integer dot is native, against the activations' 16-bit twin (xquant.glsl), which the row kernels read too.
- A workgroup computes a TILE_ROWS x 64 tile, walking the inner dimension one quant block of 32 at a time, over all of it or over one part when the call is split; matmul_reduce.comp adds the parts in order.
- Each step stages, per row, the block's quants widened to the twin's signed 16-bit pairs with its scale and minimum, and per column the twin's 16 words of the block with its scale and scaled sum.
- A thread forms (TILE_ROWS / 16) x 4 dots of 32 products, as two of 16 since Q6_K scales the halves apart.
- Types: Q4_0, Q4_1, Q4_K and Q5_K share a build; Q5_1 (LLMX_Q5), Q6_K (LLMX_Q6) and Q8_0 (LLMX_Q8) have separate builds of the same tile algorithm.
- The 8-bit builds (LLMX_I8), --dtype int8 of docs/PRECISION.md, read the activations' 8-bit twin (xquant8_word) instead, its words four signed bytes in position order, against the quants as signed bytes through the four-wide 8-bit dot; the Q8_0 8-bit build stages four blocks a step in the shared memory the 16-bit build's two take.
- The Q8_0 build stages one float scale per row and per column, since Q8_0 has one scale per block and no minimum, and takes no type branch.
- Rows past nout and columns past nbatch read as zero and are not stored.

From the comment at line 43:

- Up to three projections of one X and one type per dispatch; workgroups start0, start1 and start2 on begin each projection's row tiles. kper is the quant blocks a workgroup sums, a multiple of two when the call is split; blocks of a step past its part read as zero.
- When the call is split, projection q's part z goes to y at parts * nbatch * (rows of the projections before q) + z * nbatch * nout_q, all projections sharing one scratch buffer.
- xcols is the columns of X the twin holds, which a routed call's entries read by entry / per; the scale table follows their pairs.

From the comment at line 60:

- A routed call (per != 0, backend.hpp matmul_experts) runs workgroup row y as tile y of the grouping (moe_group.comp): up to 64 entries of one expert, column c being entry tab[order0 + first + c], which reads X column entry / per through the expert's rows and writes output row entry.
- A routed call is never split.

From the comment at line 364:

- Q4_0, 18-byte blocks: a half scale, then 16 bytes whose low nibbles are values 0 to 15 and high nibbles 16 to 31; the offset of 8 goes into each byte, and the words are funnelled when they straddle.
- Q4_1, 20-byte blocks, always word aligned: a half scale and a half minimum, then Q4_0's nibbles without the offset; the minimum is stored negated since it is added.
- Staged words w and w + 4 are the low and high nibbles of one source word, so a thread staging two or more takes source words whole and stages both halves.
- A Q4_0 word that does not straddle loads itself twice, so no load leaves the matrix.

### `src/backends/vulkan/shaders/matmul_vec_q8.comp`

From the comment at line 2:

- Y[b*nout + o] = dot(row o, X column b) for Q8_0 rows through two-wide 16-bit integer dots over 16-bit activations, a weight word widened once for every column: the Q8_0 decode kernel on integer-dot devices, with the row kernel's bindings and push constants (matmul_row.comp).
- A subgroup takes ROWS rows; lane l covers quarter l % 4 of blocks l / 4, l / 4 + S / 4 and on, so a step reads S / 4 consecutive blocks of a row, or with HALF half l % 2 of blocks l / 2, l / 2 + 32 and on.
- A lane keeps MAX_COLS columns and loads each column's eight 16-bit activation values, sixteen with HALF, once for all the subgroup's rows.
- Every other block's quants start two bytes into a word, so a lane loads the words its bytes touch and shifts.
- The lanes' sums meet in one subgroup reduction per row and column, or with TREE in the same reduction's pairs taken for every row and column at once.
- A build changes only how many columns and rows share a weight read and how the same operations are scheduled: a column's lanes, their block order, its products and sums and its reduction's pairs are the same in every build, so it computes the same bits in all of them (docs/VULKAN.md).
- The 8-bit builds (LLMX_I8), --dtype int8 of docs/PRECISION.md, read the 8-bit twin (xquant8_block) against the raw weight words through the four-wide 8-bit dot, in the same lanes, blocks and order.

From the comment at line 51:

- A grouped routed call, the wide build's third pipeline (GROUPED, specialization constant 8), runs one chunk of the grouping (moe_group.comp) per workgroup row y: up to MAX_COLS entries of one expert, whose rows are then read once for all of them.
- As a specialization it costs the plain and routed pipelines nothing.

From the comment at line 67:

- Columns whose activations a lane loads together: their loads issue back to back, and a group past the pushed count is skipped whole.
- A plain pass gives a build more than half the columns it holds (for_each_column_chunk), so in a build of one column group only a group past the first half can be past the count; the second group of a two-group build gets 1 to MAX_COLS columns and a grouped build any count, so those check before every group.
- No other group ends a block before its loads.

From the comment at line 143:

- Each row's first word, a multiple of 17 words since the row's first block is even; per lane its four quant words, a fifth where the bytes straddle (the fourth again where they do not), its scale word and its activation offsets, which a step moves by 272 words, 64 runs of 16 bytes and 32 blocks.

### `src/backends/vulkan/shaders/moe_route.comp`

From the comment at line 2:

- Expert routing (backend.hpp route_experts): a workgroup per row takes the k experts of highest softmax probability, the most probable first and ties to the lower id, with their probabilities, divided by the k probabilities' sum when `normalize`.
- 64 invocations, one subgroup on a 64-wide device and two on a 32-wide one; the reductions are subgroup operations, joined through shared memory when there are two.

### `src/backends/vulkan/shaders/norm_rope_kv.comp`

From the comment at line 2:

- A layer's attention inputs in one dispatch for every view of a batch: one workgroup per (batch row, head) over the q, then k, then v heads. q is normed and rotated in place, k normed and rotated into its KV block at position hist + b, v copied into its block; the arithmetic per head is norm_rope_partial's at the full width exactly.

### `src/backends/vulkan/shaders/rms_norm_rows.comp`

From the comment at line 2:

- dst[i] = src[i] * rsqrt(mean(src^2) + eps) * w[i], per row, over `chunks` workgroups a row.
- The sum of squares is a tree, through shared memory down to one subgroup's width and then through that subgroup's shuffles in the same order; a subgroup reduction adds in another order and moved an HF gate result (docs/STATUS-2026-09.md, Vulkan backend, sub-step 1), so the tree stays.
- With `quant` set the rows are dense and a multiple of 32 wide and the output's twin is written beside it (xquant.glsl); with `quant` 2 the integer-dot tile reads the rows next, and a lane writes four values of the twin at a time.
- With one chunk dst and src may be the same buffer: every element is read before the barrier and written after it by its own lane.

### `src/backends/vulkan/shaders/sigmoid_mul.comp`

From the comment at line 2:

- dst[i] = x[i] * sigmoid(g), sigmoid(g) = 1 / (1 + exp(-g)), g being element d of head h of row r's gate at r * gate_stride + h * gate_head_stride + d, where i = (r * heads + h) * dim + d.
- With `quant` 1, a row is a multiple of 32 wide and the output's 16-bit twin is written beside it (xquant.glsl).
- With `quant` 2 the integer-dot tile reads dst next, and a lane takes four values and writes the same twin a word at a time (xquant_word).
- dst may be x: each element is read and written by one lane.

### `src/backends/vulkan/shaders/views.glsl`

From the comment at line 1:

- The view table a batched cache kernel reads (vulkan_backend.cpp, view_table): tab[0] is the view count, then six words per view (batch row, dispatch-local row, row count, history length, block-table offset and length), then every view's block ids.
- A workgroup finds its view by walking the entries, which are few.

### `src/backends/vulkan/shaders/xquant.glsl`

From the comment at line 1:

- The 16-bit twin of a float activation row for the row kernels (matmul_row.comp) and the integer-dot tile (matmul_tile_q.comp): blocks of 32 scaled so the largest magnitude is 32767, as position pairs (4m, 4m + 2) and (4m + 1, 4m + 3), then per block the scale and scaled sum.
- A subgroup writes it: every lane calls with its value at flat position i, and positions i..i+31 aligned to 32 must lie in 32 consecutive lanes.
- With specialization constant 7 a producer also writes the 8-bit twin, which --dtype int8's kernels read, after the 16-bit one (xquant8_base).
- The caller declares `xq` as a writable uint buffer.

From the comment at line 164:

- The 8-bit twin as xquant16_word writes the 16-bit one, a lane per four consecutive values; the same bits as xquant8_block.
- With `major` zero the blocks lie as xquant8_block lays them, column after column.
- Otherwise `major` is the blocks of a column and the twin is block-major, as the integer-dot tile reads a block of every column together: block k of column c at k * xquant8_columns(columns) + c, the table after the padded blocks in the same order.


Q5_1 uses the Q4 affine row and tile families with a separate fifth-bit plane. Its type descriptor selects compile-time Q5 row and tile builds for F16 and int8, and the existing decoded-weight float kernels for F32/BF16. `qdecode.glsl` owns the per-value decode used by embeddings and float products.

### `src/backends/vulkan/shaders/group_rms_norm.comp`

One workgroup reduces each row's group, using that group's weight slice.
Groups aligned to 32 values also produce the activation twin for a later matrix consumer; other widths leave packing to the consumer.

### `src/backends/vulkan/shaders/silu_scaled.comp`

Elementwise SiLU after a scalar multiplier, with exact in-place output allowed.
Aligned rows also produce the activation twin.

### `src/backends/vulkan/shaders/mix_streams.comp`

Elementwise sigmoid gates weight the streams, averaged into one output row.
Each stream and element has its own gate.
Aligned output rows also produce the activation twin.

### `src/backends/vulkan/shaders/inject_streams.comp`

Adds one block output into each residual stream using twice the sigmoid of the raw stream gate divided by the stream count.
The residual is updated in place; the block and gate inputs must be disjoint from it and each other.

### `src/backends/vulkan/shaders/signed_sqrt_gate.comp`

One workgroup reduces each normalized key/query group's dot, scales by the inverse square root of its width, and applies the signed square-root sigmoid gate to the shared value row.
Zero retains zero sign before sigmoid.
Aligned groups also produce the activation twin, with every lane participating in the packing rounds even when a final round has inactive elements.
