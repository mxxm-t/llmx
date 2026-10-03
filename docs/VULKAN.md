# Vulkan backend

Design for the first vendor backend, ROADMAP #4b, and step 5 of
[EXECUTION](EXECUTION.md). It implements the `Backend` interface over
a Vulkan device, the qwen35 layers' ops included, and nothing else:
the model layer holds no address,
computes no offset into KV storage, and submits stages through tickets.
A scheduler can keep several passes in flight over the stages; a backend
supplies an allocator, an ordered queue, and kernels.

Vulkan goes first because it is the one GPU path both machines run. The
Windows workstation's Radeon VII and the Linux machine's MI50 are the same
gfx906 silicon, the Windows HIP SDK does not support it, and the Vulkan
runtime is already present on the workstation. ROCm stays the first-class
target on Linux; it comes after this and reuses the structure.

## F16 and BF16 weight storage

The backend retains these read-only weights in their file storage and widens them in separate builds of the existing F32 embedding, row and tile kernels. Storage precision is independent of activation `--dtype`; the weight descriptor selects the modules and existing crossovers. The CPU and device qualification status is in [STATUS](STATUS.md), and ownership and float-control details are in [the backend page](src/backends-vulkan.md).

The current F16 prompt experiment uses the existing 64-row float tile where dense or grouped dispatch would select 128 rows. It leaves K partitioning and arithmetic unchanged; BF16 weights and routed experts keep their tile selections. Focused MI50 range and full-vocabulary identity checks and a fresh Windows Radeon backend check pass. A bounded MI50 performance repeat with a behavior-preserving dispatch control is complete; its tradeoff and the remaining release qualification are recorded in STATUS.

## MXFP4 and activation precision

The Vulkan backend reads MXFP4 through exact embedding, row and tile decoders. F32 uses original activations; BF16 tiles round inputs explicitly. F16 calls use block-int16 rows and, on integer-tile profiles, dense prompts with an exact float-scale copy. Routed prompts and projections beyond the bounded 256 MiB copy scratch take the wider F32 tile. Unsafe sums are recomputed in double from exact weights and the selected path's inputs; split partials stay double until reduction. Optional float preservation and double properties gate the type without raising other types' requirements. The backend advertises native F16 and resolves auto to it on the qualified MI50 and Radeon VII paths.

Focused tests cover actual row/copy/tile paths and mixed prompt/decode microbatches. BF16 tiles use the logical row extent for their inner split even below ordinary tile thresholds, and the native batch regression crosses physical widths 64 and 128 with short, long and mixed logical runs. Isolated range fixtures require the integer tile and its reduction together. Mutations forcing intermediate F32 storage fail on both overflow cancellation and subnormal sums; restored wide storage passes. A same-binary MI50 F32/F16 screen shows 2.99x/3.68x decode and 1.74x/1.64x prompt matrix-call speedups for 4096/14336 output rows at inner width 4096. These are kernel screens, not main/mx or end-to-end gates; STATUS records retained evidence and limitations. The released dtype qualification is recorded in STATUS, together with remaining per-model and reference-speed work.

The dense BF16 float dispatch now caps stages per K part at `ceil(stages / 4)` while preserving any finer occupancy split. Column slices bound its partial-buffer target to 256 MiB without changing an output's arithmetic; details and the minimum-aligned-span exception belong to [the Vulkan owner](src/backends-vulkan.md). The native workspace regression checks exact sparse F32-weight products under BF16 through dense, head, residual and grouped calls across that boundary, with odd rows, a partial column tile and guarded output gaps. The independent HF/depth and measured before/mx comparisons are recorded in STATUS; these dispatches are part of released dtype support.

## The device this is designed against

From `vulkaninfo` on the workstation, 2026-09-21: AMD Radeon VII, vendor
`0x1002` device `0x66af`, API 1.3.260, AMD driver 2.0.279, loader 1.4.321.
The facts the design depends on:

| Property | Value | Consequence |
|---|---|---|
| Subgroup size | 64 | Wave64: a row's lanes are the subgroup or a cluster of it, and a short row shares a subgroup with others |
| `shaderInt8`, `storageBuffer8BitAccess` | yes | Quantized blocks are read as 32-bit words and unpacked, and `embed` reads them as bytes |
| `shaderFloat16`, `storageBuffer16BitAccess` | yes | Block scales are unpacked from 16-bit halves of their words |
| `timelineSemaphore` | yes | `submit`/`wait` map onto one timeline value per ticket |
| `VK_KHR_push_descriptor` | yes | No descriptor pools; each dispatch pushes its buffers |
| `maxPushConstantsSize` | 128 bytes | Sizes and offsets of every op fit in push constants |
| `minStorageBufferOffsetAlignment` | 4 bytes | A float offset into a buffer is a legal binding offset |
| `maxComputeSharedMemorySize` | 32 KiB | Prefill tiles stage dequantized weights through shared memory |
| Cooperative matrix | absent | Matmul is subgroup dot products, not matrix cores |
| `maxMemoryAllocationCount` | 4096 | Separate allocation per buffer; allocation failures are reported by Vulkan |
| `VK_EXT_memory_budget` | yes | The free-memory query `memory_available` reports |

Memory heaps and the types this backend uses:

| Heap | Size | Type used | For |
|---|---|---|---|
| 0, device local | 15.73 GiB | `DEVICE_LOCAL` | `Memory::device`: weights, activations, KV blocks |
| 1, host | 15.71 GiB | `HOST_VISIBLE, COHERENT, CACHED` | `Memory::host_visible`: the logits the host reads in place |
| 1, host | | `HOST_VISIBLE, COHERENT, CACHED` | Staging for `adopt` uploads and `read`, and each ring slot's argument arena |
| 2, device local and host visible | 256 MiB | not used | The BAR window; too small to matter and uncached on the host |

Queue families: one compute-only family with transfer (two queues), one
graphics family, one transfer-only. The backend takes one queue from the
compute family. Everything, including copies, goes through that queue, so
the single implicit stream of the interface is literally one `VkQueue`.

The Linux MI50 was validated after the workstation, through RADV. Nothing
above is Windows-specific; the driver differs and the numbers are re-read
there.

## What the backend needs from the machine

The Vulkan SDK is the dependency exception `config.hpp` carves out for GPU
backends, and it is needed at **build time only**: `vulkan.h` for the
declarations, and `glslc` to compile the shaders. The runtime needs the
loader, `vulkan-1.dll` or `libvulkan.so.1`, which the driver installs.

The workstation has the LunarG SDK 1.4.357 at `C:\VulkanSDK`, installed for this work; CMake finds it through `VULKAN_SDK`.
[BUILD](BUILD.md) gives the packages and commands that build the backend on Windows and Linux, and the Docker image that carries them.

- **The loader is loaded at run time**, `LoadLibrary` or `dlopen`, and
  entry points are fetched through `vkGetInstanceProcAddr`. No import
  library, so one `llmx` binary runs on a machine with no Vulkan and
  `--device vulkan:0` fails with a message instead of the process failing
  to start. `LLMX_HAS_BACKEND_VULKAN` still gates the code, because the
  headers and the shader compiler are the SDK.
- **Shaders are GLSL source in the tree**, `src/backends/vulkan/shaders/`,
  compiled to SPIR-V by `glslc` at build time, one generated `.inc` of
  comma-separated `uint32_t` words per kernel, which `vulkan_backend.cpp`
  includes as an array. The binary carries its kernels; there are
  no files to find at run time and no runtime compiler. `build.bat` stays
  CPU-only.

### Activation range repair

The shared producer keeps extreme finite activation blocks representable by normalizing before division and restoring scale and sum bits after quantization. Ordinary blocks retain their arithmetic. Q8 row consumers optionally request 32-bit denormal and signed-zero/infinity/NaN preservation; the backend selects those modules only when both device properties are reported. Devices without them retain the original modules and may flush subnormal scales. The preserved Q8 row variants request their fused products explicitly into a precise accumulator, since enabling the modes can disable the driver's implicit contraction and change ordinary logits. The preserved Q8 integer-dot vector variant keeps a precise accumulator and separate scale product, integer-dot product and addition; explicitly fusing those operations failed ordinary-input identity on the MI50. The native packed-activation test also holds these consumers to the ordinary modules' bits on normal inputs across row and column tails; any disagreement fails the gate on a tested driver. F32 rows have a separate registry entry for the same ordinary row module, so the Q8 modes and precise accumulator cannot change their arithmetic, including MoE router scores. This is a device capability choice, with no new CLI flag or model-layer policy.

The producer and the conditional Q8 consumers pass the native range checks on the Radeon VII and the MI50, and the whole-model identity, HF and timing gates are recorded in [STATUS](STATUS.md), which also records the exact-head hosted run.
This does not establish every quant type's extreme-value arithmetic.

## Structure

`src/backends/vulkan/vulkan_backend.hpp` and, because the device code is
not header-only material, `vulkan_backend.cpp` compiled only when the
option is on. The layering rule holds: it depends on `backends/backend.hpp`,
`quant/` for the type registry, and nothing above.

- **Instance, device, queue.** One instance, the physical device selected
  by the index in `--device vulkan:N`, one logical device with the features
  above enabled, one compute queue, one command pool.
- **Buffers.** `VulkanBuffer` is one `VkBuffer` bound to its own
  `VkDeviceMemory`. `host_ptr()` is the mapped pointer for host-visible
  memory and null for device memory. Each buffer uses a separate device
  allocation; failures are reported by Vulkan. Construction releases acquired
  handles if memory selection, allocation, binding or mapping fails. Zero-fill on `alloc` is a `vkCmdFillBuffer` in the current command
  buffer, so it is ordered like every other op, and that slot holds the new buffer until it retires, as each slot an adoption copies through does, so a buffer dropped at once leaves no command naming freed memory. A buffer's size is
  rounded up to whole 32-bit words, since a tensor with an odd block
  count can end two bytes into a word its 32-bit view reads.
- **Adopt copies.** The contract lets it: a backend that copies has consumed
  `src` when `adopt` returns. Weights arrive through `adopt` in a mapped load
  and through `alloc_weight` storage filled in a streamed one, by a copy
  straight out of the loader's read ring, which the device imports as host
  memory (`VK_EXT_external_memory_host`), or by `write` where it cannot import
  it. `adopt` and `write` upload through two halves of staging, each upload
  carrying on from the half the last one left. Device copies may remain queued ahead of later work
  on the same backend.
  If an upload fails, adoption drains before releasing its local destination.
  This answers the alignment question left open
  in [DEVICE-EXECUTION](DEVICE-EXECUTION.md): the device accepts 4-byte
  storage offsets, and an upload lays the bytes out however it likes, so
  `adopt` never has to re-align anything on any backend.
- **Command recording.** Every op appends a dispatch, or a copy, to the
  open command buffer. `submit()` ends it, submits it with a timeline
  semaphore signal of the next ticket value, advances to the next ring slot, and
  returns the ticket. `wait(t)` is `vkWaitSemaphores` on that value;
  `sync()` submits any open commands and waits on the latest ticket;
  retirement failure aborts because callers rely on it before freeing storage.
  Command buffers are a ring of 16; one is reused once its ticket has retired.
  Sixteen, not four, since a server with a pass in flight per stage records a stage onto a device still running the stages before it, and an 8B stage on two or three cards is 3 to 7 submissions plus its handoff's upload: with four a host recording ahead waited in `open` for a slot, and with 16 it does not (`docs/STATUS.md`, layer split phase 3).
  A pass of several hundred dispatches is submitted in chunks of 64 as
  it is recorded, so the device starts on the first chunk while the host
  records the rest; the timeline is ordered, so the last chunk's ticket
  covers them all. Recording a Qwen3-0.6B decode token costs the host
  about 0.5 ms against 6 ms on the device, and chunks of 64 measured
  best of 16, 32, 64, 128 and 256.
  `read` records a copy into staging, submits, waits, and copies out.
- **Barriers.** A pass is a chain, so every op reads what the previous op
  wrote. The memory barrier orders compute and transfer writes before
  subsequent compute and transfer reads or writes.
  Tracking which buffers an op touches, to let independent dispatches
  overlap, is an optimization with its own measurement.
- **Descriptors.** Every kernel takes a handful of storage buffer
  bindings, several of them views of one buffer, and a block of push
  constants. With push descriptors there is no pool and no
  set allocation per op; a `Slice` becomes a buffer binding with a byte
  offset of four times its float offset. Small per-call inputs the host
  holds, ids, positions and row lists, go through a host-visible arena
  per ring slot, bumped per call and reused once the command buffer it
  was recorded into has retired. Rows and positions are
  range-checked on the host before the dispatch, since a shader cannot
  refuse them.
- **Threads.** `set_threads` is accepted and ignored; `threads_available`
  reports 0. `run_prefill` is the default.

## Kernels

All in GLSL, compute stage, subgroup operations enabled, one workgroup
size per kernel chosen for wave64. Activations are F32 everywhere except
where a quantized matmul reads them.
The decode row kernels and integer-dot prefill tile read signed 16-bit integers in blocks of 32 (`xquant.glsl`, below).
Their reference inputs use the same rounding; independent HF gates measure the resulting model error.

- **Dequantization** is one GLSL include (`qdecode.glsl`) with a per-value
  decoder for each block type but Q8_0, which `embed` uses.
  The float tile takes its Q4_0 and Q4_1 decoders and the K-quant sub-scale reader, and decodes Q8_0 and the K-quant runs itself; the row kernels decode words in place. Types are
  keyed by the storage IDs in `quant/types.hpp`; `quant::Registry` names
  implemented C++ decoders, and the shader include supplies device decoders.
  Recognized storage metadata alone does not enable a Vulkan kernel.
  Q4_0 has a second decoder for `embed`, `q4_0_exact`, which computes the CPU's `(nibble - 8) * d` and sets a zero's sign as bits: `embed` matches the CPU bit for bit under every finite scale, and a driver need not keep a zero's sign.
  Every other type's `embed` decode takes a zero's sign from the driver's arithmetic, which the drivers tested keep today; of backend-vulkan's exact embed checks only the Q6_K rows decode a -0, so for Q8_0, Q4_1, Q4_K and Q5_K nothing checks it.
  A model holding a type without a kernel is refused before any weight is adopted, naming the tensor and its type (`Backend::supports_type`).
  A direct matmul, routed product or embed of such a type is refused before it records a dispatch, naming the type by its numeric id.
  Today: F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K, plus MXFP4 on devices with its required properties. Other known storage layouts remain metadata-only.
- **matmul, decode** (`nbatch` small): each row takes a cluster of lanes, the subgroup's width or fewer for a short row and at most `k45_row_lanes` in the Q4_K/Q5_K integer-dot families, each lane accumulating a stride of blocks and the cluster meeting in an xor-shuffle reduction at the end. Rows
  are the outer loop and the batch the inner, as on the CPU, so a weight
  block is read once per chunk of a build's columns, eight, or on the MI50
  up to 32 in the Q8_0 decode kernel and 16 in the two-row builds below. Q8_0 rows are read as
  32-bit words over pairs of blocks, since a pair is 68 bytes and a row
  with an even block count starts every pair on a word boundary, with each
  weight word meeting four 16-bit activations in one 8-byte load; the first version read 16-bit words and
  managed 32 GB/s, this one 201 GB/s on the same 4096-square matvec.
  Rows with an odd block count, or fewer than eight blocks, keep the 16-bit path. The kernel is one module
  per family of types, built from one source with a define: F32 and
  Q8_0, Q4_0 and Q4_1, Q4_K, Q5_K, Q6_K. With every family in one module
  the register demand of the whole set the occupancy of every path and
  Q8_0 decode lost 40 percent without any of its instructions changing;
  Q4_K and Q5_K beside Q6_K cost Q6_K the same 40 percent, and Q4_K
  alone runs 30 percent faster than beside Q5_K; the wide Q8_0 path is
  a module of its own too, since the narrow path beside it cost it a
  third at the 8B shape. The Q4_0 path gives
  each lane a block of the 9-word pair, the first block's nibble words
  assembled from two loads since its scale is two bytes; Q4_1 is a lane
  per 5-word block; Q4_K and Q5_K are eight lanes per block, each the
  sixteen nibble bytes of one half of a 64-value chunk as one 16-byte
  load, with the scale and the three packed sub-scale words another,
  and its two groups' sub-scales and sub-mins decoded branch-free with
  selects, since the eight lanes hold different groups and the earlier
  byte-select form on divergent branches cost a tenth of Q4_K and a
  sixth of Q5_K at the 8B shape (121 to 112 us and 165 to 138); Q6_K is eight lanes per
  block, each eight consecutive positions of one half, six words of
  quants, two of sub-scales and the scale, with every other block's
  words assembled from three loads per consecutive pair since 210 bytes
  is not a multiple of four. Sixteen lanes of four positions had been
  the layout; eight of eight halves the loads per weight, and reading
  the four group scales as two 16-byte loads rather than four took the
  8B shape from 189 to 236 GB/s and the 0.6B files' 151,936-row head
  from 686 to 578 us. The final xor-shuffle
  reduction runs over the live columns only: reducing all eight slots
  for one column was 48 shuffles per lane after five loads and cost the
  1024-square matvec a quarter of its time (17.1 to 13.2 us) and the 8B
  shapes 313 to 373 GB/s.

  **Integer activations.** On a device whose profile does not prefer the integer dot, every quantized row meets the activations as
  signed 16-bit values in blocks of 32, each block scaled so its largest
  magnitude is 32767, with the block's scale `d` and `d` times its sum
  in a table: a weight word's values pair off with activation words,
  a block's integer sum is scaled once, and the Q4 offsets and K-quant
  minima use the block sum. Q6_K subtracts 32 from each weight before
  its dot, keeping the offset and operands 16-bit in the native-dot
  build; it needs no half-sum table. Values are stored in the order nibble and byte words unpack in,
  pairs of positions (4m, 4m + 2) and (4m + 1, 4m + 3), and read 8 or 16
  bytes at a time. Q8_0's first block starts two bytes into its words,
  so each lane shifts its two first-block words by a half word with the
  word after, fetched from the next lane of the pair by a shuffle. The
  twin is written by the kernel that produces the input, where the
  producer holds whole blocks in consecutive lanes: `rms_norm_rows`,
  `silu_mul`, and the per-row `attention` or its merge for heads that
  are whole blocks, each tagging the buffer it describes for the next
  row matmul on that buffer; an input without one gets a `quantize_x`
  dispatch, which a decode token never takes. Why 16 bits and not 8: the CPU
  experiment in [ASSETS](ASSETS.md) put 8-bit activations at 0.0093 of
  NLL against a 0.010 bound on the 8B excerpt and 16-bit at 0.00003,
  and on the device 16-bit dots were the faster of the two on Q8_0
  besides. At the 8B shapes on the Radeon VII, float activations to
  integer: Q8_0 413 to 398 GB/s at 4096 x 12288 and 350 to 400 at
  12288 x 4096, Q4_0 182 to 257, Q4_1 189 to 319, Q4_K 173 to 234,
  Q5_K 158 to 216, Q6_K 167 to 186, the figures including the
  standalone quantize dispatch the test's matmul takes. What bounds
  these paths is the load count, not the arithmetic: the same dots over
  8-byte activation loads throughout ran Q4_0 at 150 us against 105
  with 16-byte loads. The driver's per-kernel statistics
  (`VK_KHR_pipeline_executable_properties`, captured when the device
  has the extension and printed by `backend-vulkan` after its checks)
  put registers behind the ranking of the paths: the wide Q8_0 path
  holds 41 vector registers, five waves per SIMD, and the K-quant paths
  held 75 (Q4_K), 93 (Q5_K) and 78 (Q6_K), three, two and three waves.
  Folding a block's scales once and unpacking Q5_K's fifth bits per
  nibble word took Q5_K to 84 registers; Q4_K stayed at 74, since the
  compiler hoists the activation loads whatever the source order. Three
  layouts measured worse and are not in the tree: sixteen lanes per
  block (55 and 59 registers, four waves, but 245 and 272 GB/s, since a
  lane then keeps half the weight bytes in flight per load), the next
  block's words loaded before this block's dots (81 and 92 registers,
  251 and 257 GB/s: the hardware's load counter is in order, so a wait
  for this block's loads waits for the prefetch too), and both at once.

  History, on the Radeon VII under the AMD proprietary driver; the
  current rule is the paragraph after this one. The same statistics
  carry the driver's disassembly, which
  `backend-vulkan --isa DIR` writes out, and reading it ended the
  integer dot product extension's use there. The extension's 16-bit dot
  lowered to exactly the multiply-add pairs a plain expression gives,
  with the operands sign-extended first, so the dots are now written as
  multiplies of sign-extended halves and bytes. Interleaved, two passes
  each, that is worth about 5 percent on Q4_K at 4096 x 12288 (261 and
  267 GB/s against 271 and 282) and 4 on the 0.6B files' Q6_K head (223
  and 224 against 232 and 233), and level on Q8_0, Q4_0, Q4_1 and Q5_K.
  On the models it is 2.5 percent of 8B Q4_K_M decode and 0.8 of 8B
  Q8_0; 0.6B decode did not move outside its spread, those shapes being
  bound by dispatch latency. At the time no shader used the extension,
  so the backend stopped asking a device for
  `VK_KHR_shader_integer_dot_product`.

  The current rule. The backend enables the extension wherever the
  device offers it, and the Q4_0 and Q4_1, Q4_K, Q5_K and Q6_K row families are built again with `LLMX_DOT`, taking their dots through the integer dot instructions, while Q8_0 rows take the Q8_0 decode kernel below and F32 rows the plain build.
  That build, and the integer-dot prefill tile below, run only where the
  device's measured profile sets `prefer_integer_dot`, which
  `profile_for` honours only when the device has the integer dot
  (`backends/device_profile.hpp`). The same gfx906 silicon gains 15
  percent of 8B decode that way on the MI50 under Mesa, which lowers the
  instructions to the chip's native dot, and loses 2 percent on the
  Radeon VII under the AMD proprietary driver, which lowers them to the
  multiplies with the operands widened first; so the Radeon VII keeps
  the plain multiplies below and the float tile.

  Reading the same disassembly again showed what the multiply itself
  costs. The driver spends one `v_mad_u64_u32` per product, a 32-bit
  integer multiply-add this chip runs at a quarter rate, 32 of them in
  the Q4_K kernel's 249 vector instructions: a third of the kernel's
  issue slots for an eighth of its instructions. Removing the weight
  side's sign extension, which the compiler emits even on a value it has
  just masked to four bits, changed nothing, because the compiler
  replaced each one with another instruction rather than a cheaper
  multiply, and it will not narrow the multiply to the full-rate 24-bit
  form on its own.

  So the nibble and K-quant dots multiply as floats in the default build
  of the row kernels. Each product is a
  non-negative quant of at most five bits, or a centered Q6_K value from
  -32 through 31, against a 16-bit activation. The accumulators stay
  inside the 16,777,216 a float counts exactly, so
  the float dot returns the same integer:

  | Accumulator | Products | Largest quant magnitude | Largest partial sum |
  |-------------|---------:|--------------:|--------------------:|
  | Q4_0, Q4_1, Q4_K | 16 | 15 | 7,864,080 |
  | Q5_K | 16 | 31 | 16,252,432 |
  | Q6_K | 8 | 32 | 8,388,352 |
  | Q8_0, narrow | 32 | 128 | 134,213,632 |
  | Q8_0, wide | 16 | 128 | 67,106,816 |

  Q8_0's blocks sum past the exact float range, so that path keeps the
  integer multiply. The bounds include its valid -128 weight code.
  For the rest every `v_mad_u64_u32` is gone; the conversions do not fold
  into the operand select, so the Q4_K kernel is 265 vector instructions
  rather than 249, but all of them issue at full rate against 345
  quarter-rate-weighted slots before.

  Cutting a third of the issue slots was worth 3.4 percent of 8B Q4_K_M
  decode and 1.2 of 0.6B Q5_K_M, with Q8_0 and every prefill cell flat,
  which is the control the change wants: only the paths it touches
  moved. The size of it says these kernels are not issue bound. They are
  not bandwidth bound either, the Q4_K matmul reading 390 GB/s of a
  thousand. They are latency bound, and the register file is what limits
  how much latency the chip can hide: the K-quant kernels ran three
  waves per SIMD where the Q8_0 wide kernel runs five.

  Eight of those registers are the batch columns a lane keeps, and a
  single-sequence decode uses one. So the column count is specialization
  constant 0 and the backend builds each row kernel twice, dispatching
  the one-column build whenever a chunk of columns is one wide:

  | Kernel | Registers, 8 columns | Registers, 1 column | Waves per SIMD |
  |--------|---------------------:|--------------------:|----------------|
  | matmul_row_q4 | 69 | 57 | 3 -> 4 |
  | matmul_row_k4 | 73 | 63 | 3 -> 4 |
  | matmul_row_k5 | 83 | 69 | 3 |
  | matmul_row_k  | 72 | 62 | 3 -> 4 |

  Q5_K does not clear the 64 registers a fourth wave needs, which is why
  it gains least. The wide Q8_0 kernel is the exception and does not get
  a one-column build: it is the one row kernel whose eight-column build
  is not register starved, running five waves per SIMD, and the narrow
  build takes it to eight. On 8B Q8_0, already reading at the memory
  system's limit, that cost 9 percent of decode and took its Q8_0 matmul
  from 338 to 367 ms of device time. On the 0.6B files the same kernel
  gained 12 percent, one work unit per lane there against four, so the
  direction follows the shape as well as the path; the larger model's
  loss is the one that decides it, that cell clearing the reference by 6
  percent where the smaller clears it by 13.

  Prompt processing on a K-quant file was a separate and larger gap: on
  the MI50 a Qwen3-8B-Q4_K_M file read 98 tok/s at 247 rows against the
  reference's 530 (that reference ran split across ten cards; against one it reads 634, docs/STATUS.md thirty-fourth paragraph), and prefilled slower in absolute terms than the Q8_0
  file of the same model on the same card, 98 against 265, while reading
  a little over half the bytes. That is not bandwidth and not the tile
  shape. The tile kernel staged K-quant weights through the per-value
  decoders in `qdecode.glsl`, which re-read and re-unpack the block's
  packed sub-scale and sub-min for every value, three to five byte loads
  and the unpacking each time.

  A thread's run of values is inside one group of 32 whichever tile
  height is built, since it stages 4, 8 or 16 values starting at a multiple
  of that, so the sub-scale, the sub-min, which nibble half the run
  takes and the byte the run starts at are all invariant across it. The
  tile kernel now reads them once per run. Nothing about the arithmetic
  or the staged values changed. Interleaved, two passes each, on the
  Radeon VII:

  | model | rows | before | after |
  |---|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | 64 | 95.98 tok/s | 150.14 |
  | Qwen3-8B-Q4_K_M | 247 | 96.51 | 191.87 |
  | Qwen3-8B-Q4_K_M | 512 | 109.21 | 230.89 |
  | Qwen3-0.6B-Q5_K_M | 64 | 723.55 | 781.26 |
  | Qwen3-0.6B-Q5_K_M | 247 | 1158.02 | 1572.42 |
  | Qwen3-0.6B-Q5_K_M | 512 | 1072.31 | 1892.35 |

  Qwen3-8B-Q8_0 is the control, its branch untouched, and reads 210.93,
  283.27 and 339.15 tok/s against 204.94, 281.10 and 339.33: flat at 247
  and 512 rows and 2.8 percent down at 64, which is inside the layout
  band this file records for an unrelated edit.

  Then the loads themselves. The driver issues one `buffer_load_ubyte`
  per byte and joins none of them, 71 of them in this kernel, so the
  tile kernel reads the weights as words: the K-quant runs whose bytes
  are word aligned take one word per four values, and every scattered
  read, the scales and the odd-aligned Q6_K block, extracts from the
  word that holds it. `qdecode.glsl` reaches its bytes through a
  `QBYTE(i)` macro rather than naming an array, so a shader serves them
  from whatever view it binds, and the tile kernel's byte binding is
  gone rather than a word binding added. The kernel now issues 86 dword
  loads and no byte loads, against 32 dword, 72 byte and 5 short before.
  On top of the table above:

  | model | rows | scales hoisted | words |
  |---|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | 64 | 152.04 tok/s | 170.75 |
  | Qwen3-8B-Q4_K_M | 247 | 193.08 | 240.31 |
  | Qwen3-8B-Q4_K_M | 512 | 232.89 | 281.73 |
  | Qwen3-0.6B-Q5_K_M | 64 | 791.91 | 829.14 |
  | Qwen3-0.6B-Q5_K_M | 247 | 1578.05 | 2381.28 |
  | Qwen3-0.6B-Q5_K_M | 512 | 1924.43 | 2565.02 |

  The Q8_0 control reads 212.95, 278.65 and 337.61 against 206.13,
  283.22 and 342.28, up 3.3 percent at 64 rows and down 1.6 and 1.4 at
  247 and 512. Its staging source is unchanged, and the two directions
  in one run are the layout band rather than a result. Over both
  changes, prompt processing on the Qwen3-8B-Q4_K_M file went 95.98,
  96.51 and 109.21 tok/s to 170.75, 240.31 and 281.73.

  Half precision in the tile, tried four ways and not kept. The card
  runs half-precision arithmetic at twice the float rate, and the tile
  kernel issued 512 scalar `v_mac_f32` and no packed instruction at all,
  which is what a fully occupied card drawing half its power looks like.
  On the 8B file at 247 rows, against 240.31 tok/s:

  | form | what the driver emitted | tok/s |
  |---|---|---:|
  | halves in shared memory, float math | 512 `v_mac_f32`, a convert per read | 179.77 |
  | halves, two-wide accumulators | 256 `v_pk_mul_f16` and 248 `v_pk_add_f16` | 196.97 |
  | halves, four-wide vectors and accumulators | the same, on wide reads | 227.41 |
  | the same through `fma()` | 256 `v_pk_fma_f16` | 243.26 |

  The last form is the one that reaches the hardware properly: exactly
  half the arithmetic instructions of the float kernel, on half the
  shared memory, 8192 bytes against 16384. It is worth 1.2 percent. So
  the tile kernel is not arithmetic bound either, and halving its
  multiplies buys about what halving the row kernels' did.

  Two things the attempt did establish. The driver will not contract a
  half multiply and add on its own, so a packed multiply-add has to be
  written as `fma()`; asking for `a * b + c` gives two instructions and
  no gain. And the kernel runs three waves per SIMD on 67 registers,
  four above the 64 a fourth wave needs, which is the same limit the row
  kernels were under and where 15 to 18 percent came from there. The
  half-precision form raised registers to 77 rather than lowering them,
  so it did not help that either. Prompt processing wants the register
  count, not the precision, and a half tile would also need a second
  module to keep F32 weights in float, so none of this is kept.

  What the device does instead is the 8-bit integer dot. Measured on one MI50 under Mesa at a 4096 x 14336 projection over 512 rows, the float tile reads 4.87 TFLOPS on Q8_0 and 4.65 on Q4_K, level with another runtime's float tile at 4.77 on the same card, while that runtime's integer-dot tile reads 13.30 and 11.42. So where the profile records `prefer_integer_dot`, every quantized type's wide calls take `matmul_tile_q.comp`, Q6_K through its own module of it, `matmul_tile_q6` (`LLMX_Q6`), and later Q8_0 through `matmul_tile_q8` (`LLMX_Q8`, below); F32 keeps the float tile `matmul_tile.comp`, as does every type on a device without the preference. A pass first wrote each activation column as 8-bit values per block of 32, 8 words of signed bytes in position order, followed by a table of each block's scale and scale times integer sum, which the tile has since replaced by the 16-bit twin (below). The tile then walks the inner dimension one quant block at a time. It stages, for each of its rows, that block's quants as 8 words with the block's scale and minimum, and for each of its 64 columns the activation block's words and scales. Each output gets eight four-wide dots per block, one float multiply-add for the scales and one more for a minimum. Q8_0 quants go in as they are, and Q4_K nibbles become bytes with a shift and a mask, since values 0 to 15 are valid signed bytes.

  | type | float tile | integer-dot tile |
  |---|---:|---:|
  | Q8_0 | 4.87 TFLOPS | 7.72 |
  | Q4_K | 4.65 | 11.48 |

  The dtype implementation removes the former 8-bit activation twin and its producer specialization. Every quantized row now uses the 16-bit twin, including the output head. The earlier 8-bit measurements remain in STATUS (GPU activation history, 2026-09-30); they are not performance evidence for this implementation.

  Q8_0 was slower on the 8-bit twin in the row kernel's wide path, 167.0 against 176.8 us, since that path spreads a load instruction over 16-byte pieces of every block pair. So on such a device Q8_0 takes a kernel of its own (`matmul_vec_q8.comp`). A subgroup takes two rows; lane l covers quarter l % 4 of every (S / 4)-th block, so a step reads a contiguous run of a row; its eight 8-bit activation values are loaded once per column and serve both rows, through two four-wide dots per quarter. On an MI50 the 14336 x 4096 Q8_0 matvec goes from 167 to 135 us, and decode from 45.5 to 60.6 tok/s on 8B Q8_0 and from 259 to 311 on 0.6B Q8_0. One row per subgroup read 322 tok/s on the 0.6B file and 53 on the 8B, four rows 283 and 53.
  Since 2026-09-29 it reads the 16-bit twin instead, each value split into a signed high byte and a biased low byte and taken through two four-wide dots with a correction by the weights' sum (`dot16.glsl`), which gives the 16-bit products exactly (The 16-bit twin in the tile, below).

  A pass of several generated tokens reads each weight once per chunk of columns, so the kernel is built for 1, 2, 4, 8, 16 and 32 columns (`kVecBuilds`). The 4-, 8- and 16-column builds and the grouped build take 4 rows a subgroup, so one activation load feeds 4 rows, and the 16-column build loads two steps of every row's weights before using either in the quarter layout; the 1- and 2-column builds keep 2 rows. The 32-column build is the 16-column build twice, on adjacent workgroups over the same rows, so the second read of a row meets the first in the L2 cache; a subgroup that kept all 32 columns ran slower than two 16-column chunks. A lane loads its columns' activations in groups of 4. Each build holds twice the next narrower's columns, so a chunk fills more than half its build: a grouped build, the groups past the first half of a build of one column group, and every group of the 32-column build's second half, which gets 1 to 16 columns, check the pass's count and skip a group past it. In builds of up to 8 columns a group the pass fills in part also skips its surplus columns' products, each column behind a uniform check, which took 4 percent off a pass of 5 rows; the second copy of the group's products that takes cost the 16-column build about 4 percent from 16 rows, so it keeps one. Rows past the matrix's end are read as its last row rather than branched around. A pass takes chunks of the widest build the device's profile allows while more columns remain, then the narrowest build that holds the rest. The profile's `q8_decode_cols` is 32 on the MI50 under RADV and 8 by default, and only a device whose profile prefers the integer dot takes this kernel. On the MI50 one 16-column chunk cost 34.8 ms against 41.8 for two 8-column chunks in a screening model of 8B's matmuls (docs/STATUS.md, Layer split phase 3, step 5). Every build keeps a column's lanes, its blocks in order, the product `(dw * dx) * float(s)` added to its accumulator unfused, and its reduction's pairs, so a column computes the same bits in every build (Batch invariance). Routed chunks at Qwen3-30B-A3B's shapes take 1.6 to 1.8 times less than main's from 2 entries an expert, since a chunk of 2 to 4 entries loads one group of activations where main's loaded 8.

  On the MI50 under RADV the plain builds of 2 to 32 columns also take the transposed reduction, the one-column and grouped builds keeping the subgroup reduction, as the profile's `q8_decode_forms` allows, since each build was bound by its instructions rather than its weight reads. It pairs a lane's values i and i + h at each of the subgroup reduction's six levels, lanes l and l ^ 1, 2, 7, 15, 16 and 32, keeping one and sending the other, so after six levels every lane holds whole sums and one store writes the build's rows and columns: the 16-column build's 64 reductions of six dependent shuffled adds, a lane read and a branched store each become 63 adds. It takes the same pairs as the subgroup reduction only where that reduction does, as RADV's on a 64-lane GCN subgroup does in its disassembly, which is why a profile allows it, and a column's operations and their order are unchanged, so it computes the same bits (Batch invariance).

  The profile also gives the MI50 under RADV the half-block order (`kQ8Half`), which is not a build's form but the kernel's order on the device: every build takes it where a row holds an even block count, the one-column, grouped and routed builds included. Lane l covers half l % 2 of blocks l / 2, l / 2 + 32 and on, so a lane's 16 bytes of a block meet in one integer sum of four dots and one scaled product, where the quarter layout forms a scaled product for every 8 bytes: 14 instructions for 32 products against the quarter layout's 17 with its best forms. A column then adds other partial sums in another order than the quarter layout, so the device's decode results changed once when the order came in (2026-09-27, docs/STATUS.md, The half-block order), and since every build takes it a column computes the same bits in all of them (Batch invariance). Rows of an odd block count, and every device whose profile does not allow the order, keep the quarter layout. The quarter layout had two more forms on the MI50, offsets computed once per step and a block's scale products shared over its quad, which took only rows of an even block count; the order left them unreached and they went with it. On one MI50, 8B Q8_0 `bench --seqs` takes 13.5 ms a pass at 1 row against 14.5 in the quarter layout with its forms (main before the order), 19.4 against 23.2 at 8, 31.4 against 36.4 at 16, 57.2 against 67.5 at 32 and 115.6 against 135.8 at 64 (docs/STATUS.md, The half-block order).

  The Q4 and K-quant families' decode rows paid each column past the first 4.0 to 5.5 ms of an 8B pass, against the Q8_0 decode kernel's 1.4 (the [server investigation](benchmarks/server-investigation-20260929/README.md)). Their memory was not what bound them: with every column's activations read from one address, so each load hit the cache, the eight-column Q4_K build took the same 22.5 ms of an 8B Q4_K_M pass's eight columns as with each column's own, and staging a workgroup's activations in shared memory changed nothing. Each column's own loads, their wait and the branch before the next column were. So on the MI50 under RADV these families have two-row builds of 2, 4, 8 and 16 columns (`row_decode_cols`, 16 there): a cluster takes two adjacent rows, so each column's activation loads serve both, and computes every column the build holds, one past the pass's count on its last column and never stored, so the column loop has no branch; a pass takes 16-column chunks and the narrowest build that holds the rest. Three and four rows a cluster ran two waves a SIMD and were slower than two, and so were column loads issued in groups. A row's lanes, blocks, dots and reduction are the one-row builds', and each block's weights and float terms come from the same macros, so a column computes the one-column build's bits (Batch invariance); the one-column, eight-column and grouped builds compile to the instructions they compiled to before. Under the AMD proprietary driver a first form of these builds changed every build's bits, the one-column build's included, and made one column 25 percent slower on the Radeon VII, so a device takes them only where its profile sets the number. On one MI50, an 8B pass's decode matmuls, projections and head, in ms at 1, 2, 4, 8, 16, 32 and 64 columns, main against the two-row builds (docs/STATUS.md, Two-row decode builds for the Q4 and K-quant rows):

  | model | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
  |---|---|---|---|---|---|---|---|
  | Qwen3-8B Q4_K_M | 9.1 / 9.2 | 12.8 / 11.7 | 19.6 / 15.7 | 33.8 / 24.7 | 67.2 / 49.0 | 134.4 / 97.8 | 268.3 / 195.3 |
  | Qwen3-8B Q6_K | 15.6 / 15.6 | 19.7 / 19.9 | 27.4 / 25.1 | 44.1 / 35.9 | 87.9 / 65.5 | 175.7 / 130.6 | 351.3 / 260.9 |
  | Qwen3-8B Q4_0 | 7.8 / 7.8 | 11.0 / 9.4 | 17.9 / 12.9 | 31.9 / 21.7 | 63.5 / 39.4 | 126.8 / 78.4 | 253.6 / 156.4 |

  Qwen3-8B-Q4_K_M prompt processing at 512 rows goes from 297.8 to 488.7 tok/s. The HF perplexity cells pass in both scoring modes, and on the 8B Q4_K_M file 40 wikitext windows score mean NLL 2.47005 against the float tile's 2.47023. Q8_0 lagged its neighbour in that first version because a 34-byte block is not word aligned, so each quant word was assembled from two 16-bit loads. The AMD proprietary driver lowers the integer dot extension to widened multiplies, so the Radeon VII keeps the float tile.

  Then the other types. Q8_0 now loads a word at a time, one extra word and a funnel shift where a block straddles a word boundary; Q6_K scales each half of a 32-value group apart, so its own module sums the halves separately and folds its offset of 32 into each staged byte (one module for all of them cost Q8_0 and Q4_K 4 percent); Q5_K is Q4_K plus a fifth bit; Q4_0 folds its offset of 8 into each byte and Q4_1 adds its minimum. At the same shape on one MI50 (docs/STATUS.md, thirty-fifth paragraph):

  | type | float tile | integer-dot tile | reference's integer-dot tile |
  |---|---:|---:|---:|
  | Q8_0 | 4.87 TFLOPS | 12.07 | 13.30 |
  | Q4_K | 4.65 | 11.44 | 11.42 |
  | Q6_K | 3.62 | 9.60 | 7.03 |

  Short prompts were the weak end. A 64-row prompt is one column tile, so a 4096-row projection of an 8B model is 64 workgroups on sixty compute units. The call that suffered most was the down projection: at 64 columns it read 2.2 TFLOPS, where the gate projection read 6.2. Two changes fixed it.

  First, the pass that wrote the tile's 8-bit copy then ordered its blocks by block of the inner dimension, then by column. Before, it wrote them column after column, which put the tile's 64 columns of one block a row width apart, and one step of the tile then read its activations as a single 2 KB run. The tile has since read the 16-bit twin (below).

  Second, a call with fewer workgroups than `tile_split_per_cu` per compute unit (eight, measured; four for rows narrower than `tile_narrow_nin`, where on the MI50 Qwen3-0.6B prefill at 512 rows went from about 7300 to 8100 tok/s on Q4_0 and Q8_0 and from 6700 to 7300 on Q5_K_M, while Qwen3-8B Q4_K_M lost 3 percent with four) splits its inner dimension into parts of at least `tile_split_min_blocks` quant blocks (8; 16 capped a 1024-wide projection at two parts, and 8 raised Qwen3-0.6B prefill at 64 rows by 4 percent with no change on 8B or 30B-A3B). Each part writes its partial sums to a scratch buffer. `matmul_reduce.comp` then adds the parts in order, so the result does not depend on which workgroup finishes first.

  The workgroups counted are those of a pass over the row's whole prompt, up to a microbatch of 512 rows, rather than those of the call, so a row sums its inner dimension in the same parts however its prompt was batched (Batch invariance, below). Taking the split from the projection's shape alone, as if every call were one column tile, did that too, but split 512-row passes as finely as 64-row ones: 8B Q8_0 at 512 rows fell from 866 to 802 tok/s.

  A layer's projections of one type go through one tile dispatch, as the row kernel's do: q, k and v, and gate and up. Workgroups from start_q on take projection q's row tiles, and a split call's reduce adds every projection's parts in one dispatch, each projection on whole workgroups of its own so the output buffer a workgroup indexes is uniform across it. On an MI50 at 64 columns a 0.6B layer's q, k and v took 232 us as three dispatches and 89 us as one, and gate and up 177 against 120, since each projection alone was too small to fill the device. Qwen3-0.6B-Q8_0 prompt processing at 64 rows goes from 3761 to 5043 tok/s.

  Same card, Q8_0 at 64 columns:

  | projection | before | block-major | block-major and split |
  |---|---:|---:|---:|
  | 4096 x 12288 (down) | 2970 us | 1299 | 731 |
  | 4096 x 4096 | 617 | 499 | 354 |
  | 12288 x 4096 (gate) | 1032 | 677 | 662 |

  The block order alone takes the 512-column shape from 9.56 to 11.0 TFLOPS. Splitting on its own, without the new block order, made most shapes slower (4096 x 4096 went from 617 to 896 us), so the split target was measured with the new order in place.

  The 128-row tile's disassembly on one MI50 then showed where its time went. Every staging load sat under a branch the compiler could not prove uniform and was waited for before the next went out, so a step of two blocks waited for ten memory round trips one after another. Timed apart in an 8B Q8_0 pass of 512 rows, the math alone took 302 ms and the staging alone 362, against 490 for both. Now every load of a step goes out before the first wait: a load out of range reads a valid address and its zero is selected at the store, the K-quants unpack their sub-scales from the 16-byte head of the 256-value block held in registers (`scale_min_k4_words`), and each type's staging ends by storing the activations it loaded first, since the compiler moves a load down to its first use. The math takes word w of all of a thread's rows against word w of its four columns, so the block's dot sums stay live and each column word is read once: 91 registers before scheduling where holding four columns' 32 words took 115 (99 once every type stages its step up front), and the 64-row and 32-row tiles drop to 64 and 48 registers, four and five waves per SIMD. Q6_K, whose halves are summed apart, takes the same form four rows at a time. Each sum still adds its words in order, so no result changes.

  Q8_0 then took a module of its own, `matmul_tile_q8` (`LLMX_Q8`). A Q8_0 block has one scale and no minimum, so the module stages one float per row and one per column where the general module stages four and two, drops the minimum's multiply-add and takes no type branch. A row's first thread finds the half scale in words it already loads: the extra word of a thread's run is the one after its share where the block straddles a word boundary and the one before where it does not. The product of the scales with the sum is `precise`; in the general module the minimum term, zero for Q8_0, stands between that product and the add, so both round it before the add and the logits are the same. The smaller arrays leave room for four blocks a step, 27,648 bytes on the 128-row tile at 128 registers and two waves per SIMD, once the four blocks' math is unrolled: the compiler keeps four blocks a loop, and left that way the step paid nothing.

  The 8-bit copy, until the tile took the 16-bit twin, was a pass of its own before every tile call, four a layer, each reading back a batch its producer had just written, while that producer wrote the row kernels' twin, which a tile call never reads. The pass took four values a lane, so a block's maximum and sum take three shuffles each and a lane packs its own word. Where one tile call over the whole batch read it next, the producer wrote the copy in the twin's place and the pass does not run. The norm and the SiLU are given the row runs of the matmul that reads their output (`Backend::rms_norm_rows` and `silu_mul`, `runs`; a routed layer's SiLU gets its entries' runs, k a token row), and the wide attention has them in its views' extents. `tile_reads` accepts runs whose prompts all reach every type's tile threshold, dense or routed, with one split, which is when `matmul_runs` and `expert_runs` make a single call. The call finds the copy by a tag of the batch it was made from, cleared by every dispatch that could write that batch, by every write from the host and by every new buffer, which can take over a freed buffer's handle; any other batch keeps the twin and the pass. The arithmetic is the same in all of them, so the logits are the same bit for bit whichever writes the copy.

  A row of the copy held its batch's columns padded to an odd number of 128-byte runs. Unpadded, 512 columns put rows 16 KB apart and every writer of the copy took twice as long: the pass read 8.3 ms of an 8B Q8_0 prompt of 512 rows against 4.0 at 500 columns, and 4.05 padded to 516, where padding by 1, 8, 16, 32 and 64 blocks gave 4.50, 4.21, 4.19, 4.25 and 4.51. The RMS norm, which reads its whole row once per workgroup, splits a row only until the pass has four workgroups a compute unit, one a row at 512 rows where it took sixteen.

  **The 16-bit twin in the tile** (2026-09-29). Against HF run on a file's own weights, the 8-bit activations parted from HF by up to 8.2 logits at 16k tokens on Qwen3.5-9B Q4_K_M and missed the Qwen3.5-0.8B Q8_0's top-5 bound on one prompt, where float activations met HF within 0.08. So the tile reads the row kernels' 16-bit twin: each staged word of four quants is widened to two pairs of signed 16-bit values and multiplied by two two-wide 16-bit dots, whose sum per block of 32, at most 32 * 128 * 32767 in magnitude, is the exact integer, and the scaling after it is the 8-bit tile's. Where one tile call reads a whole batch next (`tile_reads`), the norm, the SiLU and the wide attention write the twin four values a lane (`xquant_word`), the same bits as the lane-per-value writer, without the 8-bit twin; otherwise `quantize_xw.comp` writes it before the call. The 8-bit copy, its pass and its padding are gone. The dots double, and the tile is bound by them: on one MI50 at default clocks prompt processing at 512 rows fell from 1301 to 726 tok/s on Qwen3-8B Q8_0 and from 1605 to 820 on Qwen3-30B-A3B Q4_K_M, below the reference's 810 and 1067, a speed gate left open for the branch that follows (docs/STATUS.md, MI50 prompt activations at 16 bits). Splitting each 16-bit value into two bytes for the four-wide dot gave the same bits and no speed. Each pair of dots takes the running sum as its accumulator (`dot_pairs`): summed apart, the pair cost an add after it, about 500 adds per 1024 dots in the Q8_0 tall build, and removing them gave up to 37 percent more prompt processing with the same bits (docs/STATUS.md, MI50 prompt speed at 16 bits).

  One MI50, a prompt of 512 rows, device time of a pass (`bench --profile`) and prompt processing in tok/s, the mean of three interleaved rounds (docs/STATUS.md, Prefill kernels on the MI50):

  | | 8B Q8_0 | 8B Q4_K_M | 30B-A3B Q4_K_M | 0.6B Q8_0 |
  |---|---:|---:|---:|---:|
  | before | 547.6 ms, 921 | 583.3 ms, 862 | 411.6 ms, 1207 | 55.3 ms, 9366 |
  | staging and word-major math | 452.2 ms, 1121 | 431.7 ms, 1158 | 324.7 ms, 1561 | 45.1 ms, 11494 |
  | Q8_0 module, four blocks a step | 402.5 ms, 1252 | 430.6 ms, 1164 | 323.1 ms, 1548 | 41.6 ms, 12443 |
  | the copy from its producers, padded | 386.0 ms, 1318 | 414.4 ms, 1222 | 314.1 ms, 1603 | 38.7 ms, 13380 |
  | mx-llama.cpp (ROCm), same card | 1323 | 813 | 1155 | 7200 |

  The integer-dot tile itself went from 489.6 to 344.1 ms of the 8B Q8_0 pass and from 525.1 to 372.9 of the 8B Q4_K_M one.
- **matmul, prefill** (the row counts below): a workgroup computes a
  TILE_ROWS x 64 output tile, walking the inner dimension 32 at a time;
  each step stages the dequantized W tile and the X tile in shared
  memory and every thread accumulates a (TILE_ROWS/16) x 4 micro-tile in
  registers, so a weight is read from memory once per pass. TILE_ROWS is
  a specialization constant, 32, 64 or 128, chosen per dispatch (below). Rows past `nout` and columns past
  `nbatch` read as zero and are not stored. On the Radeon VII this took
  prefill from 449 to 1025 tok/s on Qwen3-0.6B-Q8_0 and from 40 to 220
  on Qwen3-8B-Q8_0, past the upstream llama.cpp Vulkan build's 660 and
  99; the CPU-versus-device A/B checks it at batch widths 16, 64, 100 and
  247. The tile is TILE_ROWS by 64, and TILE_ROWS is a specialization
  constant, so each tile module builds a 128-row pipeline and a 64-row
  one, whose second variant is the 32-row tile, and the backend picks
  per dispatch through `tile_rows_for` in `backends/device_profile.hpp`.
  The taller tile reads three quarters of the shared memory per product, so
  128 rows are taken while they still give `tile_tall_per_cu` workgroups
  per compute unit (`tile_tall_per_cu_narrow` under 4096 values to a row),
  the unit count coming from `VK_AMD_shader_core_properties` where that
  exists and assumed small otherwise. On the Radeon VII, whose float tile
  this is, one workgroup per unit is the measured fill: four and eight
  took Qwen3-0.6B Q8_0 at 512 rows from 3060 to 1855 tok/s and Qwen3-8B
  Q8_0 at 64 rows from 217 to 187. On the MI50 under RADV the integer-dot tile
  wants four for wide rows and eight for narrow ones: with one, a
  Qwen3-0.6B prompt of 247 rows made about two workgroups per unit, two
  waves per SIMD, and prefilled 6923 tok/s on Q4_0 against 8277 with
  four; 512 rows took 9108 with eight against 8406 with four, and
  Qwen3-8B Q4_K_M 64 rows 735 with four against 678 with one, while eight
  cost 8B 2 percent at 512 rows and sixteen gained nothing over eight. Below that the choice follows
  the projection's width: under 4096 values to a row the 64-row tile is
  taken while it fills every compute unit and the 32-row one otherwise,
  and at 4096 or more the 32-row tile only below half fill, since it does
  half the arithmetic per barrier, which a narrow projection's few inner
  steps absorb and a wide one's do not (an 8B model's 4096-wide k and v
  at 128 prompt rows took 72.5 ms on the small tile against 51.5 on the
  middle one on the MI50). The 32-row tile gave the 0.6B files 11 to 23
  percent at 48 to 64 prompt rows there. Q6_K stops at 64 rows: its tile sums each block's two halves apart, and at 128 rows those sums took the build to 208 registers and one subgroup a SIMD, where 64 rows keeps two, which gave Qwen3.5-9B Q4_K_M 3 percent, Qwen3.6-27B Q4_K_M 4 and Qwen3-30B-A3B Q4_K_M 6 at 512 prompt rows on the MI50 with the same bits (docs/STATUS.md, MI50 prompt speed at 16 bits). This is the tile kernel's
  equivalent of the row kernel's lanes-per-row: the shape follows the
  device and the call rather than the source.
  The row count where this kernel starts beating the per-row one is
  `tile_from_for` in `backends/device_profile.hpp`, and it depends on
  the width of a projection and on the driver: 40 rows on a 1024-wide
  8-bit projection under the AMD proprietary driver against 96 under
  Mesa, and 26 against 30 on a 4096-wide one, both against the float
  tile. There are four thresholds, 8-bit (and F32) projections and the
  other types, each split at 4096 values to a row: `tile_from_8bit`,
  `tile_from_8bit_narrow`, `tile_from_other` and
  `tile_from_other_narrow`. The defaults are 32, 64, 64 and 64; a device
  and driver that have been measured take their own row from
  `tuned_devices`: the Radeon VII under the AMD proprietary driver
  32, 48, 64 and 64, and the MI50 under Mesa, measured against the
  integer-dot tile, 16, 32, 24 and 40. A row measured with the
  integer-dot tile applies only when the device has the integer dot. To
  measure a new one, force each kernel in turn by setting the
  thresholds to 1 and to a large number, sweep the prompt sizes, and
  take the crossing.
  The crossover from the row kernel was measured as prompt
  processing at 8 to 256 rows with the tile kernel at its old threshold
  of 16 and with the row kernel taking every width: a tile costs a
  64-row tile whatever its fill and the row kernel a weight pass per
  eight columns, so its tok/s is flat in the width (0.6B Q8_0 727 to
  838, 8B Q8_0 73 to 75, 8B Q4_K_M 98 to 104) while the tile's climbs
  (0.6B 333, 640, 831, 1810 at 16, 32, 64, 128; 8B Q8_0 52, 101, 173,
  224; 8B Q4_K_M 34, 63, 102, 143). The tile loses at 16 on every file
  and wins from about 24 rows on 8B Q8_0 and 64 on the other two, which
  the thresholds of 32 and 64 follow. What made the 16 visible was the
  server: a decode pass that a new prompt's chunk joined at 16 rows and
  up took the tile kernel and stalled every decoder for one tile.
- **attention**: one workgroup per (query row, head block, history split).
  In the non-vector kernel, subgroups take the history's tokens round robin;
  inside a subgroup each
  lane owns `head_dim / subgroup_size` elements, a token's score is one
  `subgroupAdd`, and the softmax is online, a running maximum and sum with
  the value accumulation rescaled as the maximum moves, so a 40k-token
  history needs no score array. The subgroups' partial states merge
  through shared memory at the end. A row's history is split across
  workgroups in parts of 32 tokens, the part doubling until at most 64
  cover the row (32 on an MI50 for heads 128 wide and narrower, whose
  lane groups then take twice the tokens each), each writing its
  unnormalized state to a scratch buffer that `attention_merge` combines; splitting took a 250-token decode from
  134 to 36 us per layer on the Radeon VII. The parts follow from the
  row's own length, so a row computes the same in every dispatch (Batch
  invariance, below). Keys are walked
  through the block table, a small buffer uploaded per call. GQA maps
  `n_head / n_head_kv` query heads to one KV head. Each attention kernel
  processes its selected views through a view table. Head widths up to
  256: heads 128 and 256 wide take the vector and tiled kernels below on
  any device, and a head of another width takes this kernel, whose lanes
  read four of its values each, so on a 32-lane device it is at most 128
  wide (`attention_head_fits`).
  Once the dispatch's longest row fills every split (2048 tokens at the
  defaults), a workgroup takes up to four query heads of one KV head and
  loads each token's key and value once for them, in the `_g4` builds;
  each head's arithmetic is unchanged, so the output is bit-identical to
  one head a workgroup. On a long history the cache outweighs the
  weights a decode step reads, and one head a workgroup read it
  `n_head / n_head_kv` times: Qwen3-8B Q8_0 at a 16384-token history
  decoded at 20.8 tok/s on an MI50 against the reference's 44.5, and 27.1
  grouped. A shorter history keeps one head a workgroup, since grouping
  there leaves too few workgroups, and its build holds 32 registers where
  the first grouped kernel's single build held 61.
  Heads 128 wide, every Qwen3's, take `attention_vec.comp` instead: a
  token's key and value row is read by 16 lanes in one 16-byte load each
  for f16 (two for f32), so a 64-lane subgroup reads four tokens at once,
  and each such group of lanes keeps its own online softmax, the groups
  merging in a fixed order. One head a lane group read a 256-byte row two
  bytes a lane and crossed a 64-lane reduction and an exp between tokens,
  which kept few bytes in flight: on an MI50 Qwen3-8B Q8_0 at a
  16384-token history decoded at 27.9 tok/s grouped and 29.4 with this
  kernel (0.6B 64.4 to 86.3, 30B-A3B 29.1 to 31.4), and level or better at
  every shorter history. It sums in another order than the per-lane
  kernel, so its logits are not bit-identical to it; against the CPU they
  are closer.
  Three changes that leave its arithmetic as it was then took it further.
  The width is known when the kernel is compiled, so a score's four
  exchanges across its 16 lanes compile to fixed cross-lane moves rather
  than permutes through shared memory; that was most of the difference.
  The four-head build keeps the queries in shared memory, read per token,
  which took it from 116 to about 80 registers and three subgroups a SIMD
  where it had two; and it takes a group's tokens two at a time, both
  loads issued before either is used. On one MI50, tg at a 16384-token
  history: Qwen3-8B Q8_0 30.0 to 42.8 tok/s (the reference 44.5), 0.6B
  86.1 to 129.4, 30B-A3B 31.7 to 59.1; from an empty history level or
  better (0.6B 352 to 363).
  qwen35's heads, 256 wide, take its `_d256` builds, which read a token's
  row with 32 lanes, 8 streams a workgroup, with the arithmetic of each
  lane unchanged.
- **attention_tile**, for a wide pass of 128-wide heads: a workgroup
  per 32 query rows and head, the head's K and V streamed through shared
  memory in 16-token tiles so a tile is read once per 32 rows rather
  than once per row. It took a 16384-token prompt on Qwen3-0.6B from
  155 to 513 tok/s, level with the reference's 514. Its `_d256` builds
  take qwen35's 256-wide heads with 8-key tiles, which keep K and V at
  16 KiB of shared memory, inside the 32 KiB some drivers give a
  workgroup; the tile's key count sets where the online softmax rescales,
  so it is a constant of the head width. Other head widths and narrow
  passes take the per-row kernel.

  Eight lanes share a query row. Lane l owns dimensions 32k + 4l to 32k + 4l + 3, so a row's lanes read a staged token as one contiguous 128-byte run per k, and K and V are staged eight values per load.

  A tile of 16 keys goes through the online softmax as a unit. The scores come eight tokens at a time: every lane forms its partial dots, then a butterfly over the row's lanes finishes the sums and deals them out one to a lane, which is 7 shuffles per 8 tokens. Then comes one maximum, one exponential per score, and one rescale of the accumulator per tile. The probabilities reach the row's lanes through shared memory for the weighted sum of V.

  The first version finished every score in every lane with three shuffles, and took two exponentials and a rescale per token. At 512 rows it was 28 percent of Qwen3-0.6B-Q8_0's device time on an MI50.

  | | pp512, 0.6B Q8_0 | pp2048, 0.6B Q8_0 | pp2048, 8B Q8_0 |
  |---|---:|---:|---:|
  | MI50, before | 6029 tok/s | 3267 | 668 |
  | MI50, after | 7513 | 5566 | 809 |
  | Radeon VII, before | 3052 | 2161 | - |
  | Radeon VII, after | 3165 | 2416 | - |

  Finishing each tile's scores all at once rather than eight at a time held 208 registers and one wave per SIMD, where the eight-token chunks hold 128 and two waves. A 32-token tile ran slower than a 16-token one: 4577 against 5576 tok/s at 2048 rows, since its shared memory leaves one workgroup per compute unit.
- **The view table** (`views.glsl`): every cache kernel takes one
  table over the views it processes. The host writes that table
  into the args arena, per view its batch row, dispatch-local row, row
  count, history length, block-table offset and length, then every
  view's block ids, and a workgroup or thread finds its view by walking
  the entries. Attention splits a batch into the views the tiled kernel
  takes and the rest for the per-row kernel, with an additional merge
  dispatch where a long history is split, so
  a decode row never sits in a tile staging its history for one live
  row; the merge kernel reads the same table since a dispatch's rows
  need not be a prefix of the batch. This is what took the server from
  81 to 109 percent of the reference's server at eight concurrent
  requests.
- **kv_write**: a scatter of `[rows, n_head_kv, head_dim]` into blocks,
  one lane per float.
- **norm_rope_partial**: one workgroup per (row, head): the head's sum of
  squares in a tree through shared memory, then the rotation of its first
  `rope_dim` values reading the table at the row's position, the heads read
  at their strides and written contiguously. It is the partial rope of the
  qwen35 layers; Qwen3's rope, the op at the full width in place, runs in
  the fused kernel below.
- **norm_rope_kv**: the layer's attention inputs in one dispatch, a
  workgroup per (row, head) over the q heads, the k heads and the v
  heads: q normed and rotated in place with norm_rope_partial's arithmetic at the full width,
  k normed and rotated straight into its KV block, v copied into its
  block.
  The model asks for the three together (`Backend::norm_rope_kv`, whose default is `norm_rope_partial` on q and on k, then `kv_write`, and is what the CPU runs), and every view of a batch goes through the view table in that one dispatch.
  Three dispatches fewer per layer, 0.6B Q8_0 decode 202 to 221 tok/s under the matched protocol.
- **silu_mul, add, gather_rows**: elementwise or gather kernels, one invocation per output float, or four a lane where the integer-dot tile reads `silu_mul`'s output next.
- **rms_norm_rows**: a row over one or more workgroups, each summing the whole row's squares in a tree and writing its own chunks.
- **embed**: a workgroup per gathered row, dequantizing it on the way.
- **The qwen35 layers** (docs/QWEN35.md): `sigmoid_mul` and `gated_rms_norm`,
  which write the copy of their output the next matmul reads as `silu_mul`
  and `rms_norm_rows` do; `causal_conv_silu`, one dispatch of an invocation
  per (chunk of 16 rows, channel), whose view's first chunk alone reads and
  leaves the carried rows; and the gated delta rule, one dispatch, a
  workgroup per (view, V head, 32 V columns) that stages each block of 16
  tokens' normed q and k, gates and v in shared memory and then runs the
  per-token recurrence, eight lanes a column with 16 of its rows each in
  registers, summed in row order within a lane and through one butterfly
  across the eight, so a sequence gives the same bits however its rows are
  batched or cut into passes. The state is read from slot `src` and written
  to slot `dst` through a view table like the KV cache's. How these kernels
  measure is in STATUS, the qwen35 plan's step 5.

### KV layout on the device

The backend chooses its block size and the layout inside a block, per [KV-CACHE](KV-CACHE.md).
Blocks are per-layer device buffers holding K and V.
The storage derives from `BlockKVStorage` (`src/backends/kv_storage.hpp`), the one the CPU storage derives from, so both grow by the same allocate-and-copy, the same doubling rule and the same accounting, and check views the same way.
Here the copy is enqueued on the queue, and the buffers it reads are kept until the command buffer that recorded it retires.
Failed growth drains those copies while the new buffers are still alive, preserves the prior backing and peak accounting, and permits a retry.
Successful growth stays asynchronous.
The layout inside a block is `[kv_head][token][head_dim]` so a head's keys within a block are contiguous for the attention lanes.
The block size starts at 64 tokens, half the CPU's, because the attention workgroup reads a block per iteration and smaller blocks waste less tail per sequence on the device that bounds concurrency; screened on the real models the same way the CPU's 128 was, 32, 64 and 128 measured within noise and 64 kept (sub-step 7).

Each side is stored as f32 or f16 (`--cache-type-k`, `--cache-type-v`,
the same flags and meaning on the CPU). An f16 side is written by the
kernels with an explicit round-to-nearest-even in the bits (`f16.glsl`),
because `packHalf2x16` leaves the rounding to the driver and a driver
that truncates makes the device cache differ from the CPU's by an f16
ulp; read back it is exact, so the two backends hold identical bytes and
their attention differs only by reduction order. Every kernel that
touches the cache (`kv_write`, `norm_rope_kv`, `attention`, `attention_vec`,
`attention_tile`, and the `_g4` builds of the two per-row kernels) is built in four variants, one per combination of the
two sides' types, and the storage picks the variant, so a kernel carries
no type branch. Halving the cache is what lets Qwen3-8B run a 16k
context on the 16 GB card.

## Kernel notes

Measurements behind choices in the kernels, kept here rather than in the code.

- **Row kernel accumulators.** The wide Q8_0 path is unrolled by hand with scalar accumulators; a generic loop over words with accumulator arrays indexed by column spilled to scratch and ran ten times slower. Handling block 0's two-byte straddle with per-word selects instead of a half-word shift cost a quarter of the kernel.
- **One-column builds.** Eight column accumulators are eight of the Q4_K row kernel's 73 registers, the difference between three waves per SIMD and four, which was worth 15 percent of 8B Q4_K_M decode. The wide Q8_0 path already runs five waves and keeps its eight-column build: its one-column build cost 8B Q8_0 decode 9 percent.
- **Float dots.** The nibble and K-quant row dots multiply as floats because the chip's 32-bit integer multiply is quarter rate. They stay exact: the largest partial sum any of them reaches is 16.5 million, below the 16.8 million (2^24) a float counts exactly.
- **Dot forms.** The integer dot product extension, which Mesa lowers to the chip's native 16-bit dot, was worth 15 percent of 8B decode and 6 of 0.6B on the MI50; the AMD proprietary driver lowers it to widened multiplies and prefers the plain form by 2 percent on the same silicon.
- **Tile crossover.** Forcing each kernel and sweeping, a 1024-wide 8-bit projection crosses from the row kernel to the float tile near 40 rows under the AMD proprietary driver and near 96 under Mesa, a 4096-wide one near 26 and 30. Against the integer-dot tile the crossings fell to 24 to 32 rows on Qwen3-0.6B-Q8_0, 8 to 16 on 8B-Q8_0, 32 to 48 on 0.6B-Q5_K_M and 16 to 24 on 8B-Q4_K_M; at 64 rows on the 0.6B file the tile read 2160 tok/s where the row kernel read 1098.
- **Per-call arena.** An allocation per call for ids, positions and row lists was over a hundred `vkAllocateMemory` calls per decoded token, most of the token on Qwen3-0.6B; the ring slot's arena replaced them.
- **Loads up front in small dispatches.** The compiler does not overlap one loop iteration's loads with the next, and a dispatch of a few waves per compute unit has nothing else to hide them behind, so a loop of eight loads a lane waits eight times. The RMS norm, the routing, the combine and the F32 row path now load a lane's first values into registers before summing any, in the same order, so their results are unchanged bit for bit; the norm's tree also finishes its last six steps inside one subgroup through shuffles, the same additions without barriers. Timed alone on an MI50 without timestamps, where a dependent dispatch that does almost nothing takes 4.1 us: the 2048-wide norm went from 11.4 to 7.4 us, the F32 router's 128 x 2048 matvec from 13.4 to 7.8, the routing from 10.9 to 9.2. Qwen3-30B-A3B Q4_K_M decode went from 120.0 to 129.0 tok/s and Qwen3-8B Q4_K_M from 93.1 to 96.4. Two adjacent Q4_K rows a cluster, sharing each activation load, did not help (branch `research/k45-two-rows-rejected`).
- **Timestamps.** Kernel timestamps inflate small dispatches: a 12288 x 2048 Q4_K matvec reads 26 us without them and 39.5 with, so `--profile` shares of the small kernels are upper bounds.
- **F32 rows a power of two wide.** Such rows sit a multiple of the memory's channel interleave apart, so every row of a tile step reads the same channel: on either card a 4096 x 2048 F32 tile took 7 to 10 times as long as a 4096 x 2080 one. The float tile reads an adopted F32 matrix whose rows are a multiple of 256 floats wide through a copy with 32 floats after each row, made on its first tile call and kept with the buffer until a write or copy into it (`padded_f32`); the Radeon VII's 4096 x 2048 case at 512 columns went from 26.3 ms to 3.2, against 2.5 at 2080 wide. It replaced starting each block of rows at its own inner step, which reached 7.1 ms, and leaves the sums in their plain order. A call split because it was starved takes the shortest tile, for the most workgroups. On the MI50 Qwen3-30B-A3B's prefill at 512 tokens went from 1179-1195 to 1204-1221 tok/s; its router's 512 activation rows are 2048 floats apart too, and it still takes about 400 us a layer there.
- **Float tile split.** A call with fewer workgroups than a quarter of the compute units splits the float tile's inner dimension as the integer-dot tile does, the parts added in order by `matmul_reduce.comp`. Qwen3-30B-A3B's F32 router at 32 prompt rows was four workgroups; split, the MI50's prefill went from 357 to 410-412 tok/s at 32 tokens and 628 to 663 at 128. Splitting every call below eight workgroups per compute unit, the integer-dot tile's rule, cost the Radeon VII 2.7 percent of Qwen3-0.6B Q4_0 prefill at 64 tokens, where its extra reduce dispatches outweighed the fill.

## Batch invariance

A row computes the same, bit for bit, whatever else shares its pass. The row kernels and the tiles round differently, so a kernel chosen by a call's width made a prompt's result depend on how its rows were batched. A server that reused a cached prefix prefilled the prompt's tail through the row kernel, while one pass over the whole prompt took the tile for the same rows, and on a near-tie the two gave different greedy text.

- **Kernel by prompt, not by batch.** The model passes each call its rows as runs (`backend::RowRun`). A run's extent is the position one past the last token of the prompt its rows belong to, or 1 for a generated token. A matmul row takes the tile when its extent reaches the tile threshold and the row kernel below it, and a call that mixes the two becomes one call per kernel over its rows. Attention takes the tile for views whose extent reaches 32. Short prompts keep the row kernels they took before.
- **Split by prompt.** The integer-dot tile's inner-dimension split, and the float tile's, is the one a pass over the row's whole prompt takes (above), and a call whose rows belong to prompts with different splits becomes one call per split.
- **Decode columns by build.** A row kernel's builds differ only in how many columns and rows share a weight read, and the Q8_0 decode kernel's transposed reduction only in how the same operations are scheduled, while its half-block order is every build's on a device that takes it, so a decode column is the same bits in every build and every chunking of its pass (the Q8_0 decode kernel above). `backend-vulkan` checks each column against the same column alone: plain calls of 1 to 64 columns, the residual add and a group of three projections at sixteen widths from 1 to 64 that reach every build, and routed calls of 1 to 32 tokens (AGENTS.md, Tests). Under `--isa` it also holds each build to its one-column build's counts of float multiplies and adds, the same or with whole columns added, and each Q8_0 decode build to the counts its shape and forms give; that screens how a driver contracted a column's sums, and only the column comparison shows a reassociation that keeps the counts.
- **Sums written in order.** The ordinary row kernels' arithmetic carries no `precise` qualifier, only the Q8_0 builds taken for float preservation keeping a `precise` accumulator, so a driver may fuse a multiply into the add that takes it, and the AMD proprietary driver also reassociates sums; each build compiles on its own, so the choice can differ between a row kernel's one-column and wide builds. The Q4_1 row kernel without the integer dot writes a block's two terms into the sum as two additions, the minimum's first, which that driver fuses into two multiply-adds in every build. Written as one addition of both terms, the Radeon VII's one-column build folded them into the sum one at a time and its wide build added them together first, so a generated token's Q4_1 products depended on how many tokens shared its pass.
  On a driver that keeps source order the new order changes the one-column Q4_1 result once as well: on an MI50 under RADV with its profile off the integer dot, 3319 of 4800 outputs of rows 4096 wide moved, in the last bits (docs/STATUS.md, One order for a Q4_1 block's two terms). The old order under `precise`, which no driver may reassociate or fuse, moved 3100 of them there too, so neither form keeps both drivers' one-column bits. The integer-dot Q4_1 builds keep the one addition, since RADV compiles it the same way in every build, which `--isa` and the column comparison check, and the Radeon VII takes no integer dot.
- **Attention history by row.** The per-row kernel splits a row's history into parts of 32 tokens, the part doubling until at most the profile's maximum cover the row (`attention_split_max`, `attention_split_max_wide`), so the parts follow from the row's length alone. Every dispatch splits by that rule, and a row's empty parts merge as exact zeros.

`backend-vulkan` checks this bitwise at the model's shapes, 2048 and 6144 outputs over 249 rows, where the whole prompt takes the tallest tile and its 9-row tail the shortest. It compares a prompt's last rows alone against the same rows of one pass, a generated row alone against it beside others, a call mixing both against each, attention over a prompt's tail after a reused history against one pass, and a decode row beside a longer history against it alone. At the model API, Qwen3-0.6B-Q8_0 on an MI50 gives bit-identical logits through one pass, a forked reused prefix and a prefix-then-tail pass.

## Mixture of experts

A routed layer (`qwen3moe`) applies the router matmul, routing, gate and up projections, SiLU over every slot, the down projection and a separate combine. Grouping, activation quantization and split reductions can add dispatches, so these operations do not imply a fixed dispatch count.

- **Routing.** `moe_route.comp` takes a row per 64-invocation workgroup: the softmax's maximum and sum, then k rounds of the largest untaken probability and the lowest id holding it, through subgroup reductions joined in shared memory when the workgroup is two subgroups. With shared-memory trees it cost about seventy barriers a token and 8 percent of a Qwen3-30B-A3B decode pass; with subgroups 4.
- **Decode.** The row kernels and `matmul_vec_q8.comp` take a routed mode: workgroup row y is entry y, which reads X column y / per through its expert's rows, found by offsetting the weight row by the expert times the rows per expert, since a GGUF stacks the experts back to back. Nothing else in the kernels changes, so every type's decode path serves routed rows.
- **Generated tokens beside each other.** An expert chosen by several tokens of one pass had its rows read once per entry, and a server's throughput on Qwen3-30B-A3B Q4_K_M stopped growing past eight concurrent requests (about 225 tok/s on one MI50). When a pass's entries average at least two an expert, `moe_group.comp` groups them in runs of the row kernel's column count and each run is a workgroup row of the wide build with a column per entry (the grouped mode, specialization constant 8), so the rows are read once a run; below that, each entry keeps its own workgroup row, since grouping saves few reads and costs a dispatch. Splitting runs of one entry off to the one-column build did not help: the loss below two entries an expert was the grouping and its dispatches. A column computes the same in either build, and `backend-vulkan` checks each token's entries beside others against the token alone, bit for bit. The grouped mode is the wide build's third pipeline, specialization constant 8, so the plain and routed pipelines carry none of it: computed per column inside every row kernel it had cost the MI50's integer-dot builds up to 7.5 percent of plain decode. On the MI50 at 64 tokens a request the server went from 225-226 to 262-263 tok/s at 32 concurrent, time to first token from 848 to 606-614 ms, and stayed within 2 percent below that. Timed alone, a routed gate and up pair reads its rows at 580 GB/s for one token and 660-680 for four to eight, near a dense 8B projection's 650-685; what a decode token spends beyond its rows is its dependent dispatches and small kernels (Kernel notes, loads up front).
- **Prompts.** A row whose prompt extent reaches its weight type's `moe_tile_from` takes the tile kernels over each expert's entries. On the MI50 under RADV that is 32 for Q8_0 and Q6_K, 48 for Q5_K, 64 for Q4_K and 96 for Q4_0 and Q4_1, where the tiles first overtook the rows on Qwen3-30B-A3B (rows against tiles, tok/s: Q4_K_M 299 and 234 at 32, 340 and 326 at 48, 361 and 402 at 64; Q5_K_M 251 and 218 at 32, 291 and 321 at 48; Q6_K 206 and 178 at 24, 204 and 211 at 32; Q8_0 236 and 215 at 24, 239 and 263 at 32; Q4_0 459 and 405 at 64, 471 and 496 at 96; Q4_1 447 and 431 at 64, 463 and 541 at 96). A 4-bit row is cheap to unpack once per entry, so the row kernels stay ahead of the tile's grouping longer. The Radeon VII under the AMD proprietary driver gains from the same values with experts on the CPU (tok/s, 32 for every type against these: Q4_0 70.1 and 78.3 at 48, 91.7 and 94.3 at 80; Q4_K_M 54.9 and 63.0 at 32, 68.9 and 74.1 at 48; Q5_K_M 33.7 and 35.1 at 32), and they are the defaults. `moe_group.comp` groups a call's entries, a workgroup per expert: a shared histogram of every id, the expert's first entry and first tile from the lower experts' counts, and its entries placed in order by a prefix sum over chunks of 256. The tile's workgroup row y is tile y, up to 64 entries of one expert, gathered as its columns and scattered back as output rows; a routed tile is never split. Through the row kernels a prompt read an expert's weights once per entry, and Qwen3-30B-A3B prefilled 383 tok/s at 512 rows on an MI50; through the tiles 1048, and with the grouping a workgroup per expert and reused by the down projection 1214.
- **Invariance.** Neither kernel's arithmetic for a column depends on the other columns, and the kernel follows the row's extent, so an entry computes the same whatever else is routed beside it, as the dense rows do.
- **The twin.** A row kernel's output dropped the activation twin whenever it shared a buffer with the input, which the arena always does; only an overlapping output drops it now, so the experts read the twin the router's input has, and routing keeps it the same way.

## Selection and reporting

`--device cpu` is the default and `--device vulkan:N` selects a device, as `docs/USAGE.md` and `print_usage` give it.
`llmx info` gains nothing, and there is no `llmx devices` listing.
`--threads` keeps its CPU meaning and does nothing for a Vulkan device unless experts run on the CPU beside it, which the usage text says.

## Gates

The two project gates apply unchanged (ROADMAP #8):

- **Correctness is the HF reference.** `tests/baseline.py` and the F32 and
  shard fixtures run with `--device vulkan:0` and must pass the same
  bounds as on CPU. Before that, each kernel is compared against the CPU
  backend on identical random inputs with a bound frozen before the run:
  bit-exact for the elementwise and gather kernels and for `kv_write`,
  and a stated float tolerance for the reductions, matmul, norms and
  attention, whose order differs.
- **Performance floor is mx-llama.cpp's Vulkan backend on the same card**,
  same model, quant and prompt, prefill and decode tokens per second, the
  matched comparison `tools/compare_cpu.py` already makes for the CPU. A
  Vulkan backend slower than the thing it replaces has no claim to being
  a runtime, exactly as for the CPU. The measurement is `llmx bench
  --model F --device vulkan:0 --p N --n N --r R` beside the reference's
  bench tool with the same `-p N -n N -r R`: both warm up, both time
  model work only, both process the prompt into an empty history and
  generate from an empty history, and both report the mean of R repeats.
  `generate --verbose` is not that measurement: its decode figure is one
  cold run after the prompt with sampling inside the timer, and on
  Qwen3-0.6B-Q8_0 it read 138 tok/s where the matched protocol reads 202
  against the reference's 195, so the floor tables below name which
  protocol each number came from.

There is a third number worth recording but not gating on: the CPU
backend on the same machine. The Ryzen 7 5800X does 0.6B decode at about
46 tokens per second and 8B at about 4; the card's 1 TB/s of memory
bandwidth against the CPU's 50 GB/s says where decode should land.

## Order of work

Each sub-step is mergeable, builds with the option off on every CI runner,
and adds a `backend-vulkan` CTest that skips cleanly with a message when no
device is present, so the tree stays green without a GPU.

The table preserves the sub-step validation at the time it landed, including
the then-open floor in step 5. The later two-platform gate is recorded in
[STATUS](STATUS.md), in the Vulkan block's forty-seventh paragraph; those
measured cells do not establish a universal performance floor.

| # | Sub-step | Test |
|---|---|---|
| 1 | Build gate, loader, device and queue, buffers, `adopt`/`read`/`write`/`copy`, `submit`/`wait`/`sync` (**done**) | `backend-vulkan`: zeroed allocations, adopt and copy round trips at odd offsets, writes into device and host-visible memory, a copy read in place after a wait, monotonic tickets, empty and out-of-range buffers; skips without a device |
| 2 | Elementwise kernels, `gather_rows`, `embed` (F32 and Q8_0), the norms, `norm_rope_rows`; the shader build step (**done**) | CPU-vs-Vulkan on random inputs, bounds fixed in the test before the first run: exact for add, gather and embed, 1e-6 relative for SiLU, 1e-5 for the norms and RoPE; 160,688 outputs on the Radeon VII |
| 3 | `matmul` for F32 and Q8_0: the row kernel and the tile kernel (**done**) | Against the CPU over batch widths 1, 3, 8 and 13 on the row kernel and 16, 64, 100 and 247 on the tile kernel, both block-count parities, 1e-4 relative; the 4096-square Q8_0 matvec reads at about 200 GB/s on the Radeon VII, reported and not gated |
| 4 | KV storage, `kv_write`, `kv_copy`, `attention` over views (**done**) | Against the CPU backend through each backend's own storage and block size: histories of 0, 63, 64, 65 and 131 tokens with 1 and 3 queries, two views in one call, a copied block attending like its source; 1e-4 relative |
| 5 | `--device`; the models end to end (**done** except the floor) | HF baselines with `--device vulkan:0`: Q8_0 logits and all four perplexity cases match the CPU's numbers to the digit; the whole Python suite runs on the device; the matched mx Vulkan floor is the open item |
| 6 | Q4_0, Q4_1, Q4_K, Q5_K, Q6_K shaders (**done**) | HF baselines on the Q4_0 and Q5_K_M fixtures pass on the device with the CPU's numbers; `backend-vulkan` checks each type against the CPU in `embed`, the tile and the row kernel and reports the matvec bandwidth per type |
| 7 | Block-size screening (**done**: 32, 64 and 128 tokens measured within noise on 0.6B and 8B decode over 512 tokens, 64 kept); barrier tracking if a profile says so | The KV screening method, on the device |
| 8 | Integer activations for the decode row kernel (**done**: 16-bit values in blocks of 32 through integer multiplies, the twin written by the norm, SiLU and attention kernels) | `backend-vulkan` against the CPU fed the same quantized activations, 1e-4 relative, every type and both block-count parities, plus a norm, a SiLU and an attention into a buffer and a matmul from it; the HF gate on the device through the tile path and, with `--ubatch 8`, through the row kernel; the matched decode floor |

The CPU backend is untouched throughout and remains the reference.

## Open questions

- **Wave64 tuned.** The kernels are tuned for a 64-wide subgroup. The
  backend requires subgroups of at least 32 lanes whose size divides the
  row kernel's 256-invocation workgroup, and reads the size rather than
  forcing it; RDNA and other vendors default to 32, and whether the
  kernels are re-tuned for that is not decided until there is such a
  device to decide against.
- **Sub-allocation.** One allocation per buffer is within the device limit
  for the models here and is simpler to get right. If a model or a KV
  budget approaches the limit, a chunked allocator goes behind the same
  `alloc`.
