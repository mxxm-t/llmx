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
size per kernel chosen for wave64. Activations are F32 throughout, as on
the CPU, so the arithmetic differs from the CPU only in reduction order.

- **Dequantization** is one GLSL include with a function per quant type
  returning the float at (block, index), plus a block-scale accessor. Types
  are keyed by the same ids `quant::Registry` uses; the registry says which
  types exist, the shader include says how the device decodes them. A type
  with no shader is rejected at load with its name, not at the first op.
- **matmul, decode** (`nbatch` small): one subgroup per output row, each
  lane accumulating a stride of blocks, one `subgroupAdd` at the end. Rows
  are the outer loop and the batch the inner, as on the CPU, so a weight
  block is read once per chunk of eight columns. Q8_0 rows are read as
  32-bit words over pairs of blocks, since a pair is 68 bytes and a row
  with an even block count starts every pair on a word boundary, with the
  activations as 16-byte vectors; the first version read 16-bit words and
  managed 32 GB/s, this one 201 GB/s on the same 4096-square matvec. Rows
  with an odd block count keep the 16-bit path.
- **matmul, prefill** (`nbatch` of 16 and up): a workgroup computes a
  64 x 64 output tile, walking the inner dimension 32 at a time; each
  step stages the dequantized W tile and the X tile in shared memory and
  every thread accumulates a 4 x 4 micro-tile in registers, so a weight
  is read from memory once per pass. Rows past `nout` and columns past
  `nbatch` read as zero and are not stored. On the Radeon VII this took
  prefill from 449 to 1025 tok/s on Qwen3-0.6B-Q8_0 and from 40 to 220
  on Qwen3-8B-Q8_0, past the upstream llama.cpp Vulkan build's 660 and
  99; the CPU-versus-device A/B checks it at batch widths 16, 64, 100 and
  247.
- **attention**: one workgroup per (query row, head). The workgroup's
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
- **kv_write**: a scatter of `[rows, n_head_kv, head_dim]` into blocks,
  one lane per float.
- **norm_rope_rows**: one workgroup per (row, head): the head's sum of
  squares in a subgroup reduction, then the rotation reading the table at
  the row's position.
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
  a runtime, exactly as for the CPU.

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
| 6 | Q4_0, Q4_1, Q4_K, Q5_K, Q6_K shaders | HF baselines on the Q4_0 and K-quant models |
| 7 | Block-size screening; barrier tracking if the profile says so | The KV screening method, on the device |

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
