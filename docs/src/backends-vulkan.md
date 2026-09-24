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
  behind it.
- `vulkan_kernel_statistics` returns the driver's per-kernel registers,
  shared memory and scratch when the device serves them, which the test
  prints after its checks. With `diagnostics` the backend also captures
  the driver's disassembly of each kernel, which
  `vulkan_kernel_representations` returns and `backend-vulkan --isa DIR`
  writes one file per kernel. Both are read-only reporting: nothing in
  the runtime path depends on them.
- Buffers are `VulkanBuffer`, device-local or host-visible, sized in whole
  32-bit words; every op is
  recorded into a ring of command buffers and submitted in chunks of 64
  dispatches, so the device starts a pass while the host records the rest.
  Small per-call inputs go through a host-visible arena per ring slot; a
  scratch outgrown mid-pass retires with the slot rather than being freed
  while recorded commands still name it.
- `matmul` and `matmul_group`: narrow batches take the row kernel, one
  module per family of types, reading quantized rows against an integer
  twin of the activations (`shaders/xquant.glsl`) that the producing
  kernel, the norm, the SiLU or the attention, writes beside its output
  and tags. The twin is 16-bit, or on a device whose profile prefers the
  integer dot 8-bit for every quantized family, except the Q4_0, Q4_1 and
  Q6_K rows of the output head (`matmul_logits`). Each row kernel is built
  for eight columns and for one (specialization constant 0), the
  one-column build taken when a chunk is one wide, except the wide Q8_0
  kernel. The Q4, Q4_K, Q5_K and Q6_K families are built a second time with
  `LLMX_DOT` for integer-dot devices, and the Q4 and Q6_K families a third
  with `LLMX_X8` for the 8-bit twin, whose Q6_K rows take at most 32 lanes; there Q8_0 rows take
  `shaders/matmul_vec_q8.comp`, the four-wide dot over the 8-bit twin, and
  F32 rows the plain build.
- Wide batches take a tile kernel. Where the profile sets
  `prefer_integer_dot`, every quantized type goes through the 8-bit
  integer-dot tile (`shaders/matmul_tile_q.comp`, Q6_K in its own module
  `matmul_tile_q6`) over block-major activations from
  `shaders/quantize_x8.comp`. A layer's projections of one type share one
  dispatch, and a call too small to fill the device splits its inner
  dimension into parts that `shaders/matmul_reduce.comp` adds in order.
  F32, and every type on other devices, take the float tile
  (`shaders/matmul_tile.comp`). The row count where the tile starts
  winning is one of four measured thresholds in
  `backends/device_profile.hpp` (8-bit or other types, narrower or at
  least 4096 wide), and `tile_rows_for` picks a height of 128, 64 or 32
  rows from the device's compute units and the projection's width.
- Batch invariance: with row runs (`backend.hpp` `RowRuns`) a row's
  matmul kernel and split, and its attention kernel and history split,
  follow its prompt's extent rather than the call's width, so a prompt
  computes the same whether it arrives alone, in slices or beside other
  sequences, and the server gives the CLI's text.
- `rms_norm_rows` spreads a row over several workgroups when the output
  does not overlap the input, with the same tree reduction as one.
- Mixture of experts: `shaders/moe_route.comp` routes a row per workgroup
  through subgroup reductions. The row kernels and both tile kernels take a
  routed mode (push constant `per`): a row kernel runs one entry per
  workgroup row with its expert's offset on the weight rows, and a tile
  runs one tile of up to 64 entries of one expert from the grouping
  `shaders/moe_group.comp` writes, a workgroup per expert in a stable
  order. A row's entries take the tile when its prompt's extent reaches the
  weight type's `moe_tile_from_for` (`device_profile.hpp`); a routed tile is never split, so an entry
  computes the same whatever else is routed beside it. The down
  projection's slots land in scratch and `shaders/moe_combine.comp` adds
  their weighted sum to the residual; the grouping and the activation
  twin made for gate and up are reused by it.
- The KV cache is `VulkanKVStorage`, blocks of 64 tokens in f32 or f16,
  written and read through a view table so every cache kernel runs once
  per layer over every view of a batch. Attention gives views of 128-wide
  heads whose prompt reaches 32 tokens to the tiled kernel
  (`shaders/attention_tile.comp`) and the rest to the per-row kernel, which
  splits a row's history into parts from the row's own length and merges
  them (`shaders/attention_merge.comp`).
- `kv_variant` picks the shader module for a storage's K and V types.
- `memory_available()`: the device-local heap's budget less its usage from `VK_EXT_memory_budget`, enabled where the device offers it, or the heap's size without it; the small host-mappable device window is skipped. `resident_bytes` adds the padded copy an F32 matrix whose rows are a multiple of 256 floats gets once a float tile reads it (`padded_f32`).
