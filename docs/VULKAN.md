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

The Linux MI50 is validated after the workstation, through RADV. Nothing
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
  buffer, so it is ordered like every other op.
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
- **Descriptors.** Every kernel takes at most five storage buffers and a
  block of push constants. With push descriptors there is no pool and no
  set allocation per op; a `Slice` becomes a buffer binding with a byte
  offset of four times its float offset. Small per-call inputs the host
  holds, ids, positions and row lists, go through a host-visible buffer
  allocated per call and kept until the command buffer it was recorded
  into has retired, which the ring slot tracks. Rows and positions are
  range-checked on the host before the dispatch, since a shader cannot
  refuse them.
- **Threads.** `set_threads` is accepted and ignored; `threads_available`
  reports 0. `run_prefill` is the default.

## Kernels

All in GLSL, compute stage, subgroup operations enabled, one workgroup
size per kernel chosen for wave64. Activations are F32 everywhere except
at the decode row kernel's quantized rows, which read them as signed
16-bit integers in blocks of 32 (`xquant.glsl`, below): there the
arithmetic differs from the CPU by that quantization, elsewhere only in
reduction order. The HF gate measures the cost of it.

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
  with activation words through the device's 16-bit integer dots, a
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
  with 16-byte loads.
- **matmul, prefill** (`nbatch` of 32 and up for F32 and Q8_0 rows, 64
  and up for the others): a workgroup computes a
  64 x 64 output tile, walking the inner dimension 32 at a time; each
  step stages the dequantized W tile and the X tile in shared memory and
  every thread accumulates a 4 x 4 micro-tile in registers, so a weight
  is read from memory once per pass. Rows past `nout` and columns past
  `nbatch` read as zero and are not stored. On the Radeon VII this took
  prefill from 449 to 1025 tok/s on Qwen3-0.6B-Q8_0 and from 40 to 220
  on Qwen3-8B-Q8_0, past the upstream llama.cpp Vulkan build's 660 and
  99; the CPU-versus-device A/B checks it at batch widths 16, 64, 100 and
  247. The crossover from the row kernel was measured as prompt
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
  call are one dispatch per view today; one launch over all of them is an
  optimization with its own measurement. Head widths up to 256.
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
concurrency; it is screened on the real models before the number is
fixed, the same way the CPU's 128 was.

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
| 8 | Integer activations for the decode row kernel (**done**: 16-bit values in blocks of 32 through the device's integer dots, the twin written by the norm, SiLU and attention kernels) | `backend-vulkan` against the CPU fed the same quantized activations, 1e-4 relative, every type and both block-count parities, plus a norm, a SiLU and an attention into a buffer and a matmul from it; the HF gate on the device through the tile path and, with `--ubatch 8`, through the row kernel; the matched decode floor |

The CPU backend is untouched throughout and remains the reference.

## Open questions

- **Wave64 assumed.** The kernels are written for a 64-wide subgroup and
  use `VK_EXT_subgroup_size_control` to require it. RDNA and other vendors
  default to 32; when a second device exists the kernels either take the
  size as a specialization constant or are re-tuned. Not decided until
  there is a second device to decide against.
- **F16 KV or activations.** Not in scope; the cache and activations are
  F32 as on the CPU. A change of precision is a separate step with its own
  HF gate, as [KV-CACHE](KV-CACHE.md) already says.
- **Sub-allocation.** One allocation per buffer is within the device limit
  for the models here and is simpler to get right. If a model or a KV
  budget approaches the limit, a chunked allocator goes behind the same
  `alloc`.
