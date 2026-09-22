# Vulkan backend

Design for the first vendor backend, ROADMAP #4b, and step 5 of
[EXECUTION](EXECUTION.md). It implements the whole `Backend` interface over
a Vulkan device, and nothing else: the model layer holds no address,
computes no offset into KV storage, and submits one pass at a time through
tickets, so what a backend supplies is an allocator, a queue, and kernels.

Vulkan goes first because it is the one GPU path both machines run. The
Windows workstation's Radeon VII and the Linux machine's MI50 are the same
gfx906 silicon, the Windows HIP SDK does not support it, and the Vulkan
runtime is already present on the workstation. ROCm stays the first-class
target on Linux; it comes after this and reuses the structure.

## The device this is designed against

From `vulkaninfo` on the workstation, 2026-09-21: AMD Radeon VII, vendor
`0x1002` device `0x66af`, API 1.3.260, AMD driver 2.0.279, loader 1.4.321.
The facts the design depends on:

| Property | Value | Consequence |
|---|---|---|
| Subgroup size | 64 | Wave64: one subgroup per output row in decode dots |
| `shaderInt8`, `storageBuffer8BitAccess` | yes | Quantized blocks are read as bytes, no unpacking through uints |
| `shaderFloat16`, `storageBuffer16BitAccess` | yes | Block scales are read as half directly |
| `timelineSemaphore` | yes | `submit`/`wait` map onto one timeline value per ticket |
| `VK_KHR_push_descriptor` | yes | No descriptor pools; each dispatch pushes its buffers |
| `maxPushConstantsSize` | 128 bytes | Sizes and offsets of every op fit in push constants |
| `minStorageBufferOffsetAlignment` | 4 bytes | A float offset into a buffer is a legal binding offset |
| `maxComputeSharedMemorySize` | 32 KiB | Prefill tiles stage dequantized weights through shared memory |
| Cooperative matrix | absent | Matmul is subgroup dot products, not matrix cores |
| `maxMemoryAllocationCount` | 4096 | One allocation per buffer is enough for a model; counted, not sub-allocated yet |
| `VK_EXT_memory_budget` | yes | The free-memory query the placement flags will want |

Memory heaps and the types this backend uses:

| Heap | Size | Type used | For |
|---|---|---|---|
| 0, device local | 15.73 GiB | `DEVICE_LOCAL` | `Memory::device`: weights, activations, KV blocks |
| 1, host | 15.71 GiB | `HOST_VISIBLE, COHERENT, CACHED` | `Memory::host_visible`: the logits the host reads in place |
| 1, host | | `HOST_VISIBLE, COHERENT` | Staging for `adopt` uploads and `read` |
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

The workstation has the LunarG SDK 1.4.357 at `C:\VulkanSDK`, installed
for this work; CMake finds it through `VULKAN_SDK`.

- **The loader is loaded at run time**, `LoadLibrary` or `dlopen`, and
  entry points are fetched through `vkGetInstanceProcAddr`. No import
  library, so one `llmx` binary runs on a machine with no Vulkan and
  `--device vulkan:0` fails with a message instead of the process failing
  to start. `LLMX_HAS_BACKEND_VULKAN` still gates the code, because the
  headers and the shader compiler are the SDK.
- **Shaders are GLSL source in the tree**, `src/backends/vulkan/shaders/`,
  compiled to SPIR-V by `glslc` at build time and embedded in a generated
  header as `uint32_t` arrays. The binary carries its kernels; there are
  no files to find at run time and no runtime compiler. `build.bat` stays
  CPU-only.
- **Validation layers are a CMake option**, `LLMX_VULKAN_VALIDATION`, off
  by default. Not an environment variable: it changes behaviour and must
  be visible in how the binary was built.

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
  memory and null for device memory. Allocation count is checked against
  `maxMemoryAllocationCount`; sub-allocation waits for a model that needs
  it. Zero-fill on `alloc` is a `vkCmdFillBuffer` in the current command
  buffer, so it is ordered like every other op. A buffer's size is
  rounded up to whole 32-bit words, since a tensor with an odd block
  count can end two bytes into a word its 32-bit view reads.
- **Adopt copies.** The contract lets it: `src` outlives the handle, and a
  backend that copies never relies on that. Weights are uploaded through
  the staging buffer in chunks at load, synchronously, because nothing can
  run before they are there. This answers the alignment question left open
  in [DEVICE-EXECUTION](DEVICE-EXECUTION.md): the device accepts 4-byte
  storage offsets, and an upload lays the bytes out however it likes, so
  `adopt` never has to re-align anything on any backend.
- **Command recording.** Every op appends a dispatch, or a copy, to the
  open command buffer. `submit()` ends it, submits it with a timeline
  semaphore signal of the next ticket value, opens the next one, and
  returns the ticket. `wait(t)` is `vkWaitSemaphores` on that value;
  `sync()` waits on the latest and, per contract, aborts on device loss.
  Command buffers are a ring; one is reused once its ticket has retired.
  A pass of several hundred dispatches is submitted in chunks of 64 as
  it is recorded, so the device starts on the first chunk while the host
  records the rest; the timeline is ordered, so the last chunk's ticket
  covers them all. Recording a Qwen3-0.6B decode token costs the host
  about 0.5 ms against 6 ms on the device, and chunks of 64 measured
  best of 16, 32, 64, 128 and 256.
  `read` records a copy into staging, submits, waits, and copies out.
- **Barriers.** A pass is a chain, so every op reads what the previous op
  wrote. One memory barrier, compute write to compute read, between
  consecutive dispatches is correct and is what the first version does.
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
at the decode row kernel's quantized rows, which read them as signed
16-bit integers in blocks of 32 (`xquant.glsl`, below), and at the
integer-dot prefill tile, which reads them as signed 8-bit integers in
blocks of 32 (`quantize_x8.comp`, below): there the arithmetic differs
from the CPU by that quantization, elsewhere only in reduction order. The
HF gate measures the cost of it.

- **Dequantization** is one GLSL include (`qdecode.glsl`) with a function
  per quant type returning the float at (block, index), which `embed` and
  the tile kernel use; the row kernel decodes words in place. Types are
  keyed by the same ids `quant::Registry` uses; the registry says which
  types exist, the shader include says how the device decodes them. A type
  with no shader is rejected at load with its name, not at the first op.
  Today: F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K, every type the CPU
  reads.
- **matmul, decode** (`nbatch` small): one subgroup per output row, each
  lane accumulating a stride of blocks, one `subgroupAdd` at the end. Rows
  are the outer loop and the batch the inner, as on the CPU, so a weight
  block is read once per chunk of eight columns. Q8_0 rows are read as
  32-bit words over pairs of blocks, since a pair is 68 bytes and a row
  with an even block count starts every pair on a word boundary, with the
  activations as 16-byte vectors; the first version read 16-bit words and
  managed 32 GB/s, this one 201 GB/s on the same 4096-square matvec. Rows
  with an odd block count keep the 16-bit path. The kernel is one module
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

  **Integer activations.** Every quantized row meets the activations as
  signed 16-bit values in blocks of 32, each block scaled so its largest
  magnitude is 32767, with the block's scale `d` and `d` times its sum
  (whole and per half of 16) in a table: a weight word's values pair off
  with activation words through 16-bit integer multiplies, a
  block's integer sum is scaled once, and a type's offset (Q4_0's -8,
  Q4_1's min, the K-quant mins, Q6_K's -32) is folded through the block
  sum. Values are stored in the order nibble and byte words unpack in,
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
  device offers it, and every row kernel family is built a second time
  with `LLMX_DOT`, taking its dots through the integer dot instructions.
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
  non-negative quant of at most six bits against a 16-bit activation, and
  the accumulators stay inside the 16,777,216 a float counts exactly, so
  the float dot returns the same integer:

  | Accumulator | Products | Largest quant | Largest partial sum |
  |-------------|---------:|--------------:|--------------------:|
  | Q4_0, Q4_1, Q4_K | 16 | 15 | 7,864,080 |
  | Q5_K | 16 | 31 | 16,252,432 |
  | Q6_K | 8 | 63 | 16,514,568 |
  | Q8_0, narrow | 32 | 127 | 133,165,088 |
  | Q8_0, wide | 16 | 127 | 66,582,544 |

  Q8_0 is the exception on both counts, its weights signed and its blocks
  summing past the exact range, so that path keeps the integer multiply.
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
  height is built, since it stages 8 or 16 values starting at a multiple
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

  What the device does instead is the 8-bit integer dot. Measured on one MI50 under Mesa at a 4096 x 14336 projection over 512 rows, the float tile reads 4.87 TFLOPS on Q8_0 and 4.65 on Q4_K, level with another runtime's float tile at 4.77 on the same card, while that runtime's integer-dot tile reads 13.30 and 11.42. So where the profile records `prefer_integer_dot`, every quantized type's wide calls take `matmul_tile_q.comp`, Q6_K through its own module of it, `matmul_tile_q6` (`LLMX_Q6`); F32 keeps the float tile `matmul_tile.comp`, as does every type on a device without the preference. `quantize_x8.comp` first writes each activation column as 8-bit values per block of 32, 8 words of signed bytes in position order, followed by a table of each block's scale and scale times integer sum. The tile then walks the inner dimension one quant block at a time. It stages, for each of its rows, that block's quants as 8 words with the block's scale and minimum, and for each of its 64 columns the activation block's words and scales. Each output gets eight four-wide dots per block, one float multiply-add for the scales and one more for a minimum. Q8_0 quants go in as they are, and Q4_K nibbles become bytes with a shift and a mask, since values 0 to 15 are valid signed bytes.

  | type | float tile | integer-dot tile |
  |---|---:|---:|
  | Q8_0 | 4.87 TFLOPS | 7.72 |
  | Q4_K | 4.65 | 11.48 |

  Qwen3-8B-Q4_K_M prompt processing at 512 rows goes from 297.8 to 488.7 tok/s. The HF perplexity cells pass in both scoring modes, and on the 8B Q4_K_M file 40 wikitext windows score mean NLL 2.47005 against the float tile's 2.47023. Q8_0 lagged its neighbour in that first version because a 34-byte block is not word aligned, so each quant word was assembled from two 16-bit loads. The AMD proprietary driver lowers the integer dot extension to widened multiplies, so the Radeon VII keeps the float tile.

  Then the other types. Q8_0 now loads a word at a time, one extra word and a funnel shift where a block straddles a word boundary; Q6_K scales each half of a 32-value group apart, so its own module sums the halves separately and folds its offset of 32 into each staged byte (one module for all of them cost Q8_0 and Q4_K 4 percent); Q5_K is Q4_K plus a fifth bit; Q4_0 folds its offset of 8 into each byte and Q4_1 adds its minimum. At the same shape on one MI50 (docs/STATUS.md, thirty-fifth paragraph):

  | type | float tile | integer-dot tile | reference's integer-dot tile |
  |---|---:|---:|---:|
  | Q8_0 | 4.87 TFLOPS | 12.07 | 13.30 |
  | Q4_K | 4.65 | 11.44 | 11.42 |
  | Q6_K | 3.62 | 9.60 | 7.03 |
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
  The taller tile reads two thirds of the shared memory per product, so
  128 rows are taken while they still give a workgroup per compute unit,
  which the backend learns from `VK_AMD_shader_core_properties` where
  that exists and assumes small otherwise. Below that the choice follows
  the projection's width: under 4096 values to a row the 64-row tile is
  taken while it fills every compute unit and the 32-row one otherwise,
  and at 4096 or more the 32-row tile only below half fill, since it does
  half the arithmetic per barrier, which a narrow projection's few inner
  steps absorb and a wide one's do not (an 8B model's 4096-wide k and v
  at 128 prompt rows took 72.5 ms on the small tile against 51.5 on the
  middle one on the MI50). The 32-row tile gave the 0.6B files 11 to 23
  percent at 48 to 64 prompt rows there. This is the tile kernel's
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
  `measured_profiles`: the Radeon VII under the AMD proprietary driver
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
- **attention**: one workgroup per (query row, head).
  subgroups take the history's tokens round robin; inside a subgroup each
  lane owns `head_dim / subgroup_size` elements, a token's score is one
  `subgroupAdd`, and the softmax is online, a running maximum and sum with
  the value accumulation rescaled as the maximum moves, so a 40k-token
  history needs no score array. The subgroups' partial states merge
  through shared memory at the end. When a pass has few (row, head)
  pairs, a decode token, the history is split into 32-token chunks across
  workgroups, capped at 64 splits, each writing its unnormalized state to
  a scratch buffer that `attention_merge` combines; that took a 250-token
  decode from 134 to 36 us per layer on the Radeon VII. Keys are walked
  through the block table, a small buffer uploaded per call. GQA maps
  `n_head / n_head_kv` query heads to one KV head. Several views in one
  call are one dispatch through the view table below. Head widths up to
  256.
- **attention_tile**, for a wide pass of 128-wide heads: a workgroup
  per 32 query rows and head, the head's K and V streamed through shared
  memory in 16-token tiles so a tile is read once per 32 rows rather
  than once per row; eight lanes share a row, a score is three xor
  shuffles, the softmax is online per row. It took a 16384-token prompt
  on Qwen3-0.6B from 155 to 513 tok/s, level with the reference's 514.
  Other head widths and narrow passes take the per-row kernel.
- **The view table** (`views.glsl`): every cache kernel takes one
  dispatch per layer over every view of a batch. The host writes a table
  into the args arena, per view its batch row, dispatch-local row, row
  count, history length, block-table offset and length, then every
  view's block ids, and a workgroup or thread finds its view by walking
  the entries. Attention splits a batch into the views the tiled kernel
  takes and the rest for the per-row kernel, two dispatches at most, so
  a decode row never sits in a tile staging its history for one live
  row; the merge kernel reads the same table since a dispatch's rows
  need not be a prefix of the batch. This is what took the server from
  81 to 109 percent of the reference's server at eight concurrent
  requests.
- **kv_write**: a scatter of `[rows, n_head_kv, head_dim]` into blocks,
  one lane per float.
- **norm_rope_rows**: one workgroup per (row, head): the head's sum of
  squares in a subgroup reduction, then the rotation reading the table at
  the row's position.
- **norm_rope_kv**: the layer's attention inputs in one dispatch, a
  workgroup per (row, head) over the q heads, the k heads and the v
  heads: q normed and rotated in place with norm_rope_rows' arithmetic,
  k normed and rotated straight into its KV block, v copied into its
  block. The model asks for the three together (`Backend::norm_rope_kv`,
  whose default is the three ops and is what the CPU runs); a batch over
  several views takes that default. Three dispatches fewer per layer,
  0.6B Q8_0 decode 202 to 221 tok/s under the matched protocol.
- **rms_norm_rows, silu_mul, add, gather_rows, embed**: elementwise or
  gather kernels, one invocation per output float, `embed` dequantizing
  its row on the way.

### KV layout on the device

The backend chooses its block size and the layout inside a block, per
[KV-CACHE](KV-CACHE.md). Blocks are per-layer device buffers holding K and
V, grown by allocate-and-copy exactly as the CPU storage does, the copy
enqueued on the queue. The layout inside a block is
`[kv_head][token][head_dim]` so a head's keys within a block are contiguous
for the attention lanes. The block size starts at 64 tokens, half the
CPU's, because the attention workgroup reads a block per iteration and
smaller blocks waste less tail per sequence on the device that bounds
concurrency; screened on the real models the same way the CPU's 128
was, 32, 64 and 128 measured within noise and 64 kept (sub-step 7).

Each side is stored as f32 or f16 (`--cache-type-k`, `--cache-type-v`,
the same flags and meaning on the CPU). An f16 side is written by the
kernels with an explicit round-to-nearest-even in the bits (`f16.glsl`),
because `packHalf2x16` leaves the rounding to the driver and a driver
that truncates makes the device cache differ from the CPU's by an f16
ulp; read back it is exact, so the two backends hold identical bytes and
their attention differs only by reduction order. Every kernel that
touches the cache (`kv_write`, `norm_rope_kv`, `attention`,
`attention_tile`) is built in four variants, one per combination of the
two sides' types, and the storage picks the variant, so a kernel carries
no type branch. Halving the cache is what lets Qwen3-8B run a 16k
context on the 16 GB card.

## Selection and reporting

`--device cpu` is the default and `--device vulkan:N` selects a device;
the flag lands in `docs/USAGE.md` and `print_usage` with the backend.
`llmx info` gains nothing; a `llmx devices` listing waits until there is a
second backend to list. `--threads` keeps its CPU meaning and does nothing
for a Vulkan device, which the usage text says.

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
