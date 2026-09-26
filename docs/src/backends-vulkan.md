# `src/backends/vulkan/` - the Vulkan backend

The `Backend` of `backend.hpp` over a Vulkan 1.2 compute queue, in one
translation unit (`vulkan_backend.cpp`, built only with
`LLMX_HAS_BACKEND_VULKAN=ON`) and the GLSL kernels under `shaders/`, which
`glslc` compiles at build time into arrays the unit includes. The loader is
opened at run time, so a build carries no link dependency; the design,
kernel notes and measurements are `docs/VULKAN.md`.

- `make_vulkan_backend(index, diagnostics)`, `vulkan_device_name`: open
  the loader, pick the device, require what the kernels need (Vulkan 1.2,
  subgroups of 32 lanes or more, 16-bit integers, timeline semaphores,
  push descriptors); anything missing throws `VulkanUnavailable`, which
  the test skips on and the CLI reports, as does a loader with no driver
  behind it. The device's `DeviceCaps` choose its `DeviceProfile`
  (`backends/device_profile.hpp`), which `vulkan_device_profile` returns so
  the test predicts the kernel the backend picks.
- `vulkan_kernel_statistics` returns the driver's per-kernel registers,
  shared memory and scratch when the device serves them, which the test
  prints after its checks. With `diagnostics` the backend also captures
  the driver's disassembly of each kernel, which
  `vulkan_kernel_representations` returns and `backend-vulkan --isa DIR`
  writes one file per kernel, and on a queue that timestamps it times the
  dispatches: `vulkan_kernel_times` returns device milliseconds per kernel
  since the last reading, waiting for the queue, and
  `vulkan_timed_dispatches` how many dispatches that reading covered, the
  query pool sampling a long interval's first ones (`bench --profile`). All
  of these are read-only reporting: nothing in the runtime path depends on
  them.
- Buffers are `VulkanBuffer`, device-local or host-visible, sized in whole
  32-bit words; every op is
  recorded into a ring of command buffers and submitted in chunks of 64
  dispatches, so the device starts a pass while the host records the rest.
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
  Padded copies enter their owning cache before their copy command is recorded;
  a replaced copy is retained by the command-buffer slot first.
  A buffer `alloc` zero-fills or `adopt` copies into is held by each command-buffer slot that names it until the slot retires, so a caller may drop it before anything is submitted.
- `check_matrix(type, data, nin, rows, what)`: the one check of a weight operand, made by `check_group` for a matmul and by the routed products and `embed` before they record a dispatch.
  `check_group` checks a matmul call whole, weights, outputs and X, before `matmul_runs` records any of the calls its runs split it into, so a call it refuses records and writes nothing.
  `decoded_blocks` decides which types have kernels, and a type without one is refused with the unsupported-type error that `tests/common.py` matches, naming the operand as `what`, a matrix or an embedding.
  The operand must then hold `rows` rows sized by `quant::row_bytes`, which refuses a row that ends inside a block.
- `matmul` and `matmul_group` (up to three projections a call, the
  kernels' limit, with a type's block sizes from `quant::Registry`): narrow
  batches take the row kernel, one
  module per family of types, reading quantized rows against an integer
  twin of the activations (`shaders/xquant.glsl`) that the producing
  kernel, the norm, the SiLU or the attention, writes beside its output
  and tags. The twin is 16-bit, or on a device whose profile prefers the
  integer dot 8-bit for every quantized family, except the Q4_0, Q4_1 and
  Q6_K rows of the output head (`matmul_logits`). Each row kernel is built
  for eight columns and for one (specialization constant 0), the
  one-column build taken when a chunk is one wide, except the wide Q8_0
  kernel, and a third pipeline is the eight-column build grouped by expert
  (specialization constant 8, Mixture of experts below). For integer-dot
  devices the Q4 (Q4_0 and Q4_1) and Q6_K families are built again with
  `LLMX_DOT` over the 16-bit twin, which only their output head takes, and
  with `LLMX_DOT` and `LLMX_X8` over the 8-bit twin, whose rows take at
  most `q6k_row_lanes` lanes; the Q4_K and Q5_K families have only the
  8-bit dot build, whose rows take at most `k45_row_lanes`. There Q8_0 rows
  take `shaders/matmul_vec_q8.comp`, the four-wide dot over the 8-bit twin,
  and F32 rows the plain build.
  The lanes that share a wide Q8_0 block pair (four) and a K-quant block (eight) are fixed by `matmul_row.comp`, and the host mirrors them in constants beside the tile heights rather than in the profile.
- Wide batches take a tile kernel. Where the profile sets
  `prefer_integer_dot`, every quantized type goes through the 8-bit
  integer-dot tile (`shaders/matmul_tile_q.comp`, Q6_K in its own module
  `matmul_tile_q6` and Q8_0 in `matmul_tile_q8`) over block-major 8-bit
  activations. When one tile call reads a whole batch next (`tile_reads`),
  the norm, the SiLU or the wide attention that wrote the batch writes
  that copy in place of the row kernels' twin (`xquant8_word` in
  `shaders/xquant.glsl`); otherwise `shaders/quantize_x8.comp` makes it
  before the call. A layer's projections of one type share one
  dispatch, and a call too small to fill the device splits its inner
  dimension into parts that `shaders/matmul_reduce.comp` adds in order.
  F32, and every type on other devices, take the float tile
  (`shaders/matmul_tile.comp`).
  The row count where the tile starts winning is one of four measured thresholds in `backends/device_profile.hpp` (8-bit or other types, narrower or at least 4096 wide), which `tile_from` takes once per call from the types of the projections that have rows.
  A mixed-type group on the row kernel becomes a dispatch per type, each kept on the row kernel.
  `tile_rows_for` picks a height of 128, 64 or 32 rows from the device's compute units and the projection's width.
- Batch invariance: with row runs (`backend.hpp` `RowRuns`) a row's
  matmul kernel and split follow its prompt's extent rather than the
  call's width. Attention uses that extent for tiled versus row dispatch
  and each row's length for its history splits. Grouped-head variants
  also depend on the dispatch's longest history while preserving each
  row's arithmetic, so batching must not change a sequence's output.
  `matmul_runs` and `expert_runs` read the runs through `for_each_run`, keyed by kernel and split and by kernel alone.
- `rms_norm_rows` spreads a row over several workgroups when the output
  does not overlap the input, with the same tree reduction as one, up to
  four workgroups per compute unit over the pass: each reads the whole row
  for its sum, so a decode row takes sixteen and a pass of many rows one.
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
  weight type's `moe_tile_from_for` (`device_profile.hpp`); a routed tile is never split, so an entry
  computes the same whatever else is routed beside it. The down
  projection's slots land in scratch and `shaders/moe_combine.comp` adds
  their weighted sum to the residual; the grouping and the activation
  twin made for gate and up are reused by it.
- The KV cache is `VulkanKVStorage`, blocks of 64 tokens in f32 or f16, written and read through view tables, so every view of a batch goes through one dispatch of each cache kernel.
  It derives from `BlockKVStorage` (`backends-kv_storage.md`), which grows it, keeps its accounting and checks each view as the table is built; its `retire` hands the buffers a growth copied from to `keep_until_retired`, and `kv_alloc` refuses heads wider than 256.
  Attention can dispatch tiled and row kernels plus a history-split merge in one layer.
  It gives views of 128-wide heads whose prompt reaches the profile's `attention_tile_rows` to the tiled kernel (`shaders/attention_tile.comp`, 32 query rows a tile as the shader fixes them) and the rest to the per-row kernel, which splits a row's history into parts from the row's own length and merges them (`shaders/attention_merge.comp`); once the longest row fills every split, a workgroup takes up to four query heads of one KV head (the `_g4` builds), loading the history once for them with each head's arithmetic unchanged.
  Heads 128 wide take `shaders/attention_vec.comp`, which reads a token's row in 16 lanes, one load a lane, and several tokens a subgroup; other widths keep `attention.comp`.
- `kv_variant` picks the shader module for a storage's K and V types.
- `memory_available()`: the device-local heap's budget less its usage from `VK_EXT_memory_budget`, enabled where the device offers it, or the heap's size without it; the small host-mappable device window is skipped. `resident_bytes` adds the padded copy an F32 product matrix whose rows are a multiple of 256 floats gets once a float tile reads it (`padded_f32`), both reading the shape from one rule, `pads_f32`; routed stacks and gathered tables are bound as they are. `host_resident()`: the upload staging buffer and the ring of host-visible arenas, which live in host memory. `scratch_reserve(free)`: 256 MiB plus a twentieth of what is free, for tile split partials and attention merge state.
