# CUDA backend: design and qualification

The CUDA backend is the second vendor backend over the shared GPU core in `src/backends/gpu/` ([backends-gpu](src/backends-gpu.md)), after ROCm ([ROCM](ROCM.md)).
This page is its design and plan; steps 1 and 2 of the order of work below are built, and the suite's synthetic dense qwen3 models ran on an RTX 3060 at `--dtype f32` within the HF bounds, while no released model, other dtype or qwen35 model has run on CUDA yet.
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
Kernels are compiled ahead of time to a fatbin, SASS for sm_80, sm_86, sm_89, sm_90 and sm_120 plus PTX for compute_80, embedded in the binary and loaded through the driver.
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
`cuda:N` is the driver's ordinal, as `rocm:N` is the HIP index; llmx does not reorder devices, and `identity()` carries the name, the SM version and the PCI address.
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

Three transports, chosen at join by the existing probe (peer access, then a sum of known rows) and the existing per-size form timing, with no rule written for any topology:

1. Peer writes, over NVLink or over PCIe where the driver allows peer access: the shared one-shot, two-shot, chain and ring forms unchanged, staging in device memory, signals released and acquired at system scope.
2. Host-staged writes, where the driver refuses peer access: each member writes its partial rows and an epoch into pinned host memory mapped to every member, and peers read them over PCIe, with device-side signals and no host round trip.
3. Conservative: event-ordered host-staged copies and the same sum kernel, where the probe fails.

Every form sums in member order in F32 and writes the same bytes, so a group's result does not depend on which transport or form timing chose.
Layer-split handoffs use the shared counters in coherent host memory with the driver's stream wait and write calls, and need no peer access.

NCCL is an optional byte transport behind its own off-by-default build option, valid only with the CUDA backend, and opened at run time like the driver.
It carries bytes only, send and receive or gather inside the two-shot and chain forms, followed by the ordered local sums; its all-reduce is never used, since its sum order follows size and topology and would break member-order identity and batch invariance.
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
| 4 | int8 class: four-way dot rows and the integer tensor-core prompt tile; first tuning on real shapes | product microbenchmarks at Qwen3-0.6B and 27B shapes, the int8 budget, a short screen against the reference runtime | same |
| 5 | qwen35 operations for Qwen3.5-27B Q4_K_M; Qwen3.5-0.8B HF | the qwen35 baseline on 0.8B, 27B at 16k depth, drafts on and off | 24 GB or more |
| 6 | Layer split and tensor groups: the three transports, handoffs, member order | split check, tensor-split suite components, group identity across forms and widths | a pair; NVLink in a later session |
| 7 | First landing's final gates on one card | below | the card each model needs |
| 8 | MXFP4 and the deepseek4 operations, the NCCL option, Q5_0, Q5_1 and IQ4_NL, graphs if measured | per-family witnesses, the DeepSeek fixture | per item |

Every card session is one scripted batch with a deadline; the first runs step 1 as a smoke tier and continues to step 2 only if it passes.
The checks that need no card run on every change: nvcc builds of every kernel for every target, SASS witnesses (the integer dot and tensor-core instructions present where declared, no local-memory spills), host-core tests against a fake `Api`, and a host model of each `port.h` mapping.
The shared kernels keep running on ROCm, so most kernel logic is exercised on every ROCm change.
A hosted job installs the toolkit without a GPU, builds with the CUDA option on and runs the host-only tests and the SASS witnesses.

## Final gate (planned)

The first landing needs, on the cards each model needs: HF on Qwen3-0.6B Q8_0 and Qwen3.5-0.8B with the headroom recorded per path, the full suite and CTest on `cuda:0`, every dtype, and a timing round against the reference runtime's CUDA build on the same card in the same session.
Both arms are built the same way and interleaved; each run records the driver, clocks, power limit, host CPU activity and both binaries' `--version` and hash, at default clocks and power.
A figure from another runtime build or another machine is reported separately, never subtracted.
SASS for an SM version no card has run is compile coverage, marked untested.

## Open questions

- Whether host-staged writes are fast enough for a two-card group without peer access to beat a layer split; step 6 measures it.
- How far the F16 class trails the reference runtime's 8-bit tensor-core prefill on NVIDIA; int8 is the matched cell, and the FP16 tensor-core candidate is the answer if the gap is wide.
- Whether a vendor GEMM is ever needed for F32-class prefill on cards with matrix cores; decided per kernel with the numbers in hand.
