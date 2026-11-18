# CUDA backend: design and qualification

The CUDA backend is the second vendor backend over the shared GPU core in `src/backends/gpu/` ([backends-gpu](src/backends-gpu.md)), after ROCm ([ROCM](ROCM.md)).
This page is its design and plan; steps 1 to 5 of the order of work below are built, step 6 is written, and step 8's weight types and DeepSeek-V4.1 operations: dense qwen3 and qwen35 models ran on an RTX 3060 and an RTX 3090 at `--dtype f32`, `f16` and `int8`, and the MoE models and the DeepSeek operations on an RTX 3090, while step 6's layer split and tensor groups ran once, on an RTX 3090 pair without peer access, so only host writes and pulls; step 8's card sessions are recorded in Qualification so far, below.
The design was reviewed and approved on 2026-11-13.
[STATUS](STATUS.md) records each step as it lands, and a result on a card nobody has run stays marked untested ([ROADMAP](ROADMAP.md), 4b).

## Outcome and scope (planned)

Implement the existing `Backend`, `Collective` and `Handoff` contracts on NVIDIA GPUs with llmx's own kernels and ownership rules.
Reuse the loader, model architectures, placement, pass scheduler, server, dtype resolution and reference tests.
There is no second model runtime, scheduler, loader or precision system.

The first delivery target is Linux on one Ampere-or-newer card (sm_80 and later).
The completion target is what the ROCm backend covers: the text architectures, the active weight types, `--dtype` f32, f16 and int8, layer splits, tensor groups, the server and the embedded drafter.
Unimplemented operations and weight types are refused at load through `implements` and `supports_type`, before any upload.

Windows is a later scope line only: the driver library loads the same way there, but no Windows build or run is claimed until a card has run it.
Cards older than Ampere are not a target; nothing in the design rules them out, and nothing claims them.

The backend links against nothing from the toolkit.
It opens the driver library (`libcuda.so.1`) at run time into one call table, as the ROCm backend does for HIP, so a CPU-only run starts where no driver is installed and a missing driver, call or device is reported as a missing device.
Kernels are compiled ahead of time to a fatbin, SASS for each architecture in `LLMX_CUDA_ARCHS` (by default sm_70, sm_75, sm_80, sm_86, sm_89, sm_90, sm_100 and sm_120, without sm_70 from CUDA 13) plus PTX for the lowest from sm_80 on, embedded in the binary and loaded through the driver (docs/BUILD.md, GPU architectures).
There is no runtime library, no runtime compiler and no vendor BLAS in the first delivery; a vendor GEMM stays admissible only under the per-kernel rule of [ROADMAP](ROADMAP.md), 4b.
The toolkit is pinned at 12.8 or newer, the first with sm_120 SASS, in the build image and the hosted job.

## Owners and boundaries (planned)

| Concern | Owner and boundary |
| --- | --- |
| Host core: records, tickets, buffers, scratch pricing, KV layout, op dispatch | `gpu/backend_core.hpp`, the class template `GpuBackend<Api>` ROCm already instantiates; CUDA is its second `Api` |
| Collective forms, member order, timing at join, failure words | `gpu/collective_core.hpp`, shared with ROCm |
| Kernel bodies | `gpu/kernels/f32.gpu`, `gpu/kernels/int16.gpu` and `gpu/kernels/quant.h`, compiled by ROCm clang as HIP and by nvcc as CUDA |
| Vendor intrinsics | `gpu/kernels/port.h`, the only kernel file that names one |
| Driver call table, device facts, peer and host staging | `src/backends/cuda/`, CUDA's alone, as `src/backends/rocm/` is HIP's |
| Weight types, dtype resolution, placement, loading, flags | existing owners, unchanged; the backend advertises only what it implements |

The backend is compiled only with `LLMX_HAS_BACKEND_CUDA`, off by default, added with the implementation to CMake, `cmake/llmx-config.hpp.in` and the fallback `src/config.hpp`.
Without it, `--device cuda:N` reports that the build lacks the backend, as `rocm:N` does.
`cuda:N` is the driver's ordinal, as `rocm:N` is the HIP index; llmx does not reorder devices, and `identity()` carries the name, the SM version and the driver version, not the ordinal or the PCI address, since cards of one device and driver compute the same bits.
Vendor environment variables that select devices are the vendor's, not llmx knobs.

Every existing flag is honoured through the same seams or refused at load, and no flag is CUDA's alone: `--dtype` through the dtype resolver, `--cache-type-k` and `--cache-type-v` through the shared KV storage, `--load-mode` through buffer adoption, `--layer-shares` and `--tensor-width` through the collective and handoff, and `--ubatch` through the fit's scratch pricing.

## Kernel port (planned)

The kernels are the ROCm kernels, compiled once more, with each AMD construct mapped in `port.h`.
Wave width is already a per-target constant and every block reduction is a tree independent of it, so NVIDIA takes the wave32 path that gfx1151 compile coverage keeps today.

| AMD construct | CUDA form | Bits |
| --- | --- | --- |
| lane shuffles, swizzle, DPP moves | `__shfl_xor_sync` and `__shfl_down_sync` over the lanes of the exchange's width | same pairs, same bits |
| first-lane read | `__shfl_sync` from lane 0 | same |
| byte and bit align | funnel shift | same |
| byte permute | `__byte_perm`, the selector translated per call, since the AMD form has zero and sign-replicate selectors | checked per call |
| 64-bit ballot | `__ballot_sync`, 32 lanes | mask width only |
| four-way int8 dot | `__dp4a` | exact int32 |
| two-way int16 dot | `__dp2a_lo` and `__dp2a_hi` where weights are 8-bit codes, else two integer multiply-adds | exact int32 |
| scheduling barrier | dropped | none |
| sleep, wall clock | `__nanosleep`, the global timer | none |
| scoped atomics and fences | PTX `ld.acquire` and `st.release` at `.gpu` and `.sys` scope | none |
| peer stream waits and writes | the driver's 64-bit stream wait and write value calls, where the device reports stream memory operations | none |

Two hazards are audited once, before any device run.
NVIDIA allows 48 KiB of static shared memory a block against AMD's 64 KiB, so all 84 static `__shared__` uses are checked against that limit, not only the two dynamic ones.
Threads of a warp schedule independently, so any shared-memory exchange that relies on lanes moving together gets an explicit warp barrier.

The kernels compile without contraction (nvcc `--fmad=false`, IEEE division and square root, no flush to zero), mirroring ROCm's `-ffp-contract=off`, and every fused multiply-add is already written as `fmaf`.
CUDA's bits may differ from ROCm's where the math library differs; CUDA is held to the CPU bound and the HF reference, as ROCm is, not to ROCm's bits.

## Precision (planned)

The rule of [PRECISION](PRECISION.md) holds unchanged: 16-bit matmul inputs, F32 sums, and `--dtype auto` never picks int8.

| Class | CUDA form | Bound |
| --- | --- | --- |
| F32 | the exact F32 rows and tiles, ported first | the F32 bounds |
| F16 | block-int16 twins: rows on `__dp2a_lo` and `__dp2a_hi` (int16 activations against 8-bit weight codes, exact int32), prompt tiles on integer multiply-adds | the existing frozen F16 budget, with a path witness |
| int8 | block-int8 twins: rows on `__dp4a`, prompt tiles on integer tensor cores (`mma.sync` m16n8k32 s8, exact int32 sums) | the existing frozen int8 budget |

The int8 class is the same arithmetic as ROCm's four-way dot, and it is the same-precision cell against the reference runtime's 8-bit activations.
An F16 prompt tile on FP16 tensor cores is a later candidate only.
Its inputs are 16-bit and its sums F32, but the alignment inside the instruction makes it a different arithmetic, so it gets its own budget, calibrated and frozen before the candidate runs and never widened after a failure.
It may become the NVIDIA F16 prompt tile only within that budget, with headroom, at 16k depth.
FP8 and FP4 inputs violate the rule and stay out.

Weight types follow this order: Q8_0, then Q4_K with Q5_K and Q6_K (a Q4_K_M file mixes them, and a model is refused whole when one type is missing), then Q5_0, Q5_1 and IQ4_NL, then MXFP4.

## Collective design (planned)

Three transports, kept at join by the probes (peer access, then a sum of known rows) and picked per size by timing, with no rule written for any topology:

1. Peer writes, over NVLink or over PCIe where the driver allows peer access: the shared one-shot, shares and chain forms unchanged, staging in device memory, signals released and acquired at system scope.
2. Host-staged writes, where the driver refuses peer access: each member writes its partial rows and an epoch into pinned host memory mapped to every member, and peers read them over PCIe, with device-side signals and no host round trip.
3. Conservative: event-ordered host-staged copies and the same sum kernel, where the probe fails.

Every form sums in member order in F32 and writes the same bytes, so a group's result does not depend on which transport or form timing chose.
As built in step 6, transports 1 and 2 are one collective in the shared core (`gpu/collective_core.hpp`, `GroupTransport`): host writes run the peer-write kernels and forms with each member's signals and slots in mapped pinned host memory placed on the member's NUMA node, and also time transport 3's pulls and copies, the shares sent through host memory by copy engines, which store 28 GB/s down in all where four RTX 3080s' kernels stored 4.8 to 11 (docs/src/backends-gpu.md, the collective).
A group keeps every transport whose probe passes, and `join` times every form of each at every size and keeps the fastest transport and form per size (`GpuSumPick`), so on a PCIe 3 pair where peer stores run slower than host writes a prompt's sums take host writes and a token's peer writes; the staging of a transport no size picked is freed, and a group where both probes fail takes pulls throughout.
Every kernel form within 8 times the best pick's time is also timed at halved block caps (`GpuSumPick::blocks`), then at 3/4 and 3/2 of the cap the halving kept, since peer stores that cross a CPU's host bridges run faster from fewer writers: on two RTX A4000 on PCIe 3 x8 (topology NODE) the peer one-shot of two blocks sums a 5120-float row in 16.5 us against 26.8 for the pulls the join took before, and pulls keep 32 rows and more at 94% of the copy engine's both-ways rate (docs/STATUS.md).
NVLink and NVSwitch take transport 1 unchanged and the form timing picks per size; a context reaches at most eight peers, so an all-peer group is at most nine cards on every NVIDIA card, and host writes are the fast transport past that.
Hopper's asynchronous copies do not enter the collective.
A bridgeless GeForce pair refuses peer access, so a session on one proves transports 2 and 3 only, and peer writes on CUDA wait for the NVLink session.
On two RTX 3090s without a bridge (2026-11-16, gitea cuda/step6-split 9f254dd5) the join took pulls from 81920 floats a sum and the host-written wrap or chain below, and held 80 MiB of pinned host memory for Qwen3.6-27B's largest sums.
There host writes did not make a group of two beat the layer split: on Qwen3.6-27B Q4_K_M at int8 the group read 485.9 against 535.6 tok/s at pp512 and 14.0 against 21.9 at tg128, about 128 sums a token at 0.17 ms each.
Layer-split handoffs use the shared counters in coherent host memory with the driver's stream wait and write calls, and need no peer access.

NCCL is an optional byte transport behind its own off-by-default build option, valid only with the CUDA backend, and opened at run time like the driver.
It would be a fourth `GroupTransport` and carries bytes only, send and receive or gather inside the shares and chain forms, followed by the ordered local sums; its all-reduce is never used, since its sum order follows size and topology and would break member-order identity and batch invariance.
This is the rule ROCM.md gives RCCL (ROCM.md, Collective design and proof), and it is adopted only if timing at join shows it beating transports 1 and 2 at some size.
Graph capture is not part of the first delivery; it is added later only with a measured launch-gap cost.

## Order of work

Each step runs its build, its focused test and short microbenchmarks on real shapes; the landing gates run once, on the head that lands ([AGENTS](../AGENTS.md), Merge gates).
Steps 1 to 5 may share a branch; steps 6 and 8 are branches of their own.

| Step | Adds | Focused test | Card |
| --- | --- | --- | --- |
| 0 | The ROCm host core and kernels moved to `src/backends/gpu/`, with `Api` and `port.h`; ROCm behaviour unchanged | ROCm code objects disassemble identical to main, byte identity on ROCm, a timing round against main and a perturbed control | none; lands first and alone |
| 1 | Build option, fatbin embedding, driver call table, device facts, `cuda:N`, buffers, tickets, errors, the F32 kernels and F32 matmul | port-mapping test, driver table test, backend CTest, F32 matmul against the CPU | Ampere or newer, 12 GB or more |
| 2 | Dense qwen3 at F32: attention, f32 and f16 caches, norms, rope, sampling, server passes | tiny HF fixtures, suite synthetic components on `cuda:0` at `--dtype f32` | same |
| 3 | F16 class: Q8_0, Q4_K, Q5_K, Q6_K rows and integer tiles | per-type product witnesses, row-class identity, Qwen3-0.6B Q8_0 HF short and 16k raw and chat depth | same |
| 4 | int8 class: four-way dot rows and the integer tensor-core prompt tiles of every int8 tile type; first tuning on real shapes | product microbenchmarks at Qwen3-0.6B and 27B shapes, the int8 budget, a short screen against the reference runtime | same |
| 5 | qwen35 operations for Qwen3.5-27B Q4_K_M; Qwen3.5-0.8B HF | the qwen35 baseline on 0.8B, 27B at 16k depth, drafts on and off | 24 GB or more |
| 6 | Layer split and tensor groups: the three transports, handoffs, member order | split check, tensor-split suite components, group identity across forms and widths | a pair; NVLink in a later session |
| 7 | First landing's final gates on one card | below | the card each model needs |
| 8 | MXFP4 and the deepseek4 operations, the NCCL option, Q4_0, Q4_1, Q5_0, Q5_1 and IQ4_NL, graphs if measured | per-family witnesses, the DeepSeek fixture | per item |

In step 8 `backend-cuda` runs `ds4::check_ops({"mix_rows"})`: `mix_rows`, which the DeepSeek-V4.1 DSpark drafter takes, is claimed on CUDA through the shared core and untested there until then.

Every card session is one scripted batch with a deadline; the first runs step 1 as a smoke tier and continues to step 2 only if it passes.
A session reads where a prompt's device time goes with `llmx bench --model FILE --device cuda:0 --profile`, which the shared core times by events around each kernel ([USAGE](USAGE.md)), so it needs no system profiler on the card's image.
The checks that need no card run on every change: nvcc builds of every kernel for every target, SASS witnesses (the integer dot and tensor-core instructions present where declared, no local-memory spills), host-core tests against a fake `Api`, and a host model of each `port.h` mapping.
The shared kernels keep running on ROCm, so most kernel logic is exercised on every ROCm change.
A hosted job installs the toolkit without a GPU, builds with the CUDA option on and runs the host-only tests and the SASS witnesses.

## Final gate (planned)

The first landing needs, on the cards each model needs: HF on Qwen3-0.6B Q8_0 and Qwen3.5-0.8B with the headroom recorded per path, the full suite and CTest on `cuda:0`, every dtype, and a timing round against the reference runtime's CUDA build on the same card in the same session.
Both arms are built the same way and interleaved; each run records the driver, clocks, power limit, host CPU activity and both binaries' `--version` and hash, at default clocks and power.
A figure from another runtime build or another machine is reported separately, never subtracted.
SASS for an SM version no card has run is compile coverage, marked untested.

## Qualification so far (step 8, RTX 3090, sm_86, 2026-11-16)

Two scripted sessions on 24 GB RTX 3090s ran the checks of step 8 at `--dtype f16` and `int8`: S8 on the branch before its port fixes (driver 610.57, 280 W power limit), and S9 with them (driver 580.196, 180 W power limit).
A result below is what the logs show; a cell that says fail is open work, not a pass.

| Type or model | Check | f16 | int8 |
| --- | --- | --- | --- |
| Q8_0, Q4_K, Q5_K, Q6_K, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 | `backend-cuda types`: product witnesses, row classes, path records (S9) | pass | pass |
| MXFP4 | suite `mxfp4`, dense tied, untied and MoE files against HF (S9) | pass, max logit error 0.00032 against 0.0033 | pass, max logit error 0.054 against 0.079 |
| F16 weights | `backend-cuda types` half tile; suite `half-weights` (S9) | pass | half tile pass; `half-weights` skips at int8 by design |
| qwen3moe, qwen35moe | suite `moe` and `qwen35`, tiny HF fixtures (S8; `qwen35` again in S9) | pass | pass |
| qwen3moe, qwen35moe | `backend-cuda moe`, routed builds against the CPU (S9) | pass | pass |
| deepseek41 | suite `deepseek4` (S8) | pass | pass |
| deepseek41 | `backend-cuda deepseek4`, `ds4::check_ops` and the sparse tile pair (S9) | pass | pass |
| qwen35 | `backend-cuda qwen35` (S8) | pass | pass |
| drafters | suite `drafters` (S9) | pass | pass |
| Qwen3-30B-A3B Q4_K_M | greedy text, 96 tokens (S8 and S9) | runs | the same text as f16, in both sessions |
| any | `--dtype bf16` (S8) | runs f32 with the fallback warning | not applicable |

The synthetic suite components `f32`, `moe`, `int8`, `q5-0`, `q5-1`, `iq4-nl`, `qwen35` and `deepseek4` passed on `cuda:0` at both dtypes in S8, on the binary before the fixes; the fixes change only the CUDA byte permute, the dot-module skip of the F16 tile and the NaN of `model_format_qdq`, and S9's types check reran every type's rows on them.
The suite's `moe` and `deepseek4` pass again on the fixed binary at f16 and int8 in batch S10, since `model_format_qdq` is on the DeepSeek path, and on one MI50 on the fixed ROCm build.
S10 timed the fixes, 4591a038 (A) against eec57368 (B) on Qwen3-30B-A3B Q4_K_M, `bench --p 512 --n 128 --r 3`, ABAB twice on that one card, medians of A against B in tok/s: int8 pp512 1226 against 1197 and tg128 124.7 against 124.0, f16 pp512 645 against 646 and tg128 98.4 against 97.2.
B ran after A in each pair and 0.1 to 2.8 percent below it, while each arm's second round ran 0.7 to 4.2 percent below its first, so the difference is inside the drift and the move to `prmt.b32` shows no cost these runs resolve.
Qwen3-30B-A3B Q4_K_M timing, `bench --p 512 --n 128 --r 3` and the reference runtime's CUDA `llama-bench -ngl 99 -fa 1 -r 3` on the same card, ABAB twice in S9, tok/s:

| Arm | pp512 | tg128 |
| --- | --- | --- |
| llmx int8 | 1175.65, 1159.86 | 122.51, 121.54 |
| llmx f16 | 649.76, 637.10 | 98.46, 96.03 |
| reference | 1867.79, 1828.00 | 85.16, 83.82 |

Decode leads the reference by 44 percent at int8 and prompts trail it at 63 percent, an open gate step 2b's routed tensor-core tiles own.
S8 measured 2112 and 177 tok/s at int8 and 1200 and 150 at f16 on its 280 W card; the two sessions differ in card, driver and power limit, so their figures are separate results and no regression or gain is read between them.

DeepSeek-V4.1's FP4 power keys 128 wide take the integer index tile `llmx_indexer_scores_int` on every CUDA target, as on gfx9 (docs/src/backends-gpu.md), since each one builds the 8-bit dot.
Before, CUDA ran the row kernel `llmx_indexer_scores` on every index call: the tiles' gate took half of `caps.shared_bytes`, a compute unit's whole LDS under HIP but a block's 48 KiB cap under CUDA, and the float tile's 27,392 bytes at dim 128 exceed 24 KiB; the API's `shared_block_cap` now gives CUDA the whole cap, since an SM of every target holds two such blocks, so CUDA takes the float tile at other dims as well.
Its block sums are exact integer dots, so a score keeps the row kernel's bits wherever its float sums are exact and is closer to exact elsewhere, and CUDA's DeepSeek greedy hashes recorded on the float tile are recorded again on a card.
`backend-cuda deepseek4` requires the integer tile to run once on the index checks.

## The int8 prompt tile on other cards

Every int8 K tile runs one tensor-core body (`k_mma`, docs/src/backends-gpu.md) at 64 outputs, or for Q4_K, Q5_K and Q6_K at 128 where `tile_rows_for` picks it from the device's units.
A 128-output block holds 33,792 bytes of shared memory and a 64-output block 22,528; with two blocks an SM (`LLMX_TILE8_BOUNDS`, 128 registers) that is 67,584 bytes, inside sm_86's and sm_89's 100 KB and sm_80's and sm_90's 164 and 228 KB, so registers and not shared memory set the blocks an SM on every target; Q6_K's builds keep one block an SM.
On one RTX A6000 the change took int8 pp512 from 702 to 935 tok/s on Qwen3.6-27B Q4_K_M and from 3436 to 4035 on Qwen3-30B-A3B Q4_K_M, against the reference's 1281 and 4123 on the same card, with decode level (docs/STATUS.md, CUDA prompt gap, change 3, second part).
A call of more than 64 rows takes those types' square build of 128 outputs by 128 rows (`llmx_q4_k_tile8_square`, `_q5_k_`, `_q6_k_`), one block an SM at 224 to 255 registers and 45,056 bytes of shared memory, which reads and decodes each weight half as often a prompt; calls of up to 64 rows keep the 64-row build, since a square block would run half empty.
On one RTX 3090 the square build took int8 pp512 from 901 to 1024 tok/s on Qwen3.6-27B Q4_K_M, against the reference's 1213 on the same card, with Qwen3-30B-A3B level at 3736 and 3752 and decode and greedy ids unchanged (docs/STATUS.md, CUDA square K tile).
At 128 outputs the dense guard bounds the 32-bit weight offsets to about 31.5 million inputs, half the 64-output tile's.
Each IMMA's per-32 fold costs about 16 FP32 instructions, fixed by the formats' scales and the CPU's bits; Ada and Hopper double the IMMA rate an SM, so there the fold takes as many issue slots as the IMMA and the tile caps near 50 to 60 percent of the integer peak rather than Ampere's 35 to 45, and fewer issues a fold, not a wider integer sum, is the next lever there.
ROCm does not build the tensor-core body (`GPU_IMMA8`), so MI300X's int8 tile stays on four-way dots, with MFMA int8 its own lever.

## Open questions

- How far the F16 class trails the reference runtime's 8-bit tensor-core prefill on NVIDIA; int8 is the matched cell, and the FP16 tensor-core candidate is the answer if the gap is wide.
- Whether a vendor GEMM is ever needed for F32-class prefill on cards with matrix cores; decided per kernel with the numbers in hand.
