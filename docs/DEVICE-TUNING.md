# Device tuning

What the kernel and speed work on llmx has learned, distilled so a new tuning step starts from it rather than from nothing.
It holds the measured device limits, the rules that turn those limits into kernel shapes, the changes that worked, the ones that did not, how to measure, and the headroom still open.
The backends' own designs stay in [ROCM](ROCM.md), [VULKAN](VULKAN.md), [CUDA](CUDA.md) and [DEVICE-EXECUTION](DEVICE-EXECUTION.md); their dated records stay in [STATUS](STATUS.md) and the monthly files beside it.

## Using this page

Read it before any kernel or speed work, and check the failed ledger before building an arm: most ideas worth trying have been tried once.
A null result there rejects an implementation, not always the idea; each line says which, and a later win over an earlier loss is listed beside it.
Every result of a tuning step, a null one included, is added here in the same change, with its number, its date, and its commit or record path.
Profile before changing a hot path and keep the trace: the rocm-perfetto-profile skill for ROCm (rocprofv3 traces as Perfetto files), vulkan-radv-profile for Vulkan (RGP and shader statistics) and cpu-perf-profile for the CPU.

### Source keys (2026-11-23)

Sources are abbreviated in the sections below.
D:N is line N of the shared devlog `collab/llmx/devlog.md`, A:N line N of its archive `devlog-archive-20261120.md`, and I:N line N of the ideas ledger `collab/llmx/ideas.md`, all on the rig under `/opt/claude-work/`.
A bare stream name such as keff33 is that stream's directory there (its `handoff.md` and `out/`); a hex string is an llmx commit.
Percent of HBM always names its denominator: the MI50's nominal 1024 GB/s, or a measured read ceiling.

## Device limits, measured to 2026-11-23

### MI50 gfx906 under ROCm

| Limit | Value | Condition | Source |
|---|---|---|---|
| HBM nominal | 1024 GB/s | HBM2, four stacks | spec |
| Read, one 6.8 GB stream | 937 GB/s | one read kernel | cuts14, D:1345 |
| Read, real product bytes as a stream | 889 GB/s | same 27B tw4 bytes, 7.66 ms | cuts14 |
| Read-only graph, 257 launches | 844 GB/s (82%) | 6.6 GB int8 rows, 10 s sustained | bypass24, D:553 |
| Persistent kernel with next-row prefetch | 855 GB/s | best arm of the same bench | bypass24 |
| Ideal Q8_0 loader | 838 GB/s | layout costs about 0 against a raw grid stride at 840 | D:602 |
| Read ceiling by call size | 868-880, 713 at 7 MB, 536 at 2.8 MB | decode shapes | A:12688, A:17430 |
| Floor fit for held Q8_0 | 3.3 us + bytes / 889 GB/s | slope 897 small, 857 from 4352 rows | cuts27, cuts28, D:1777 |
| Held Q8_0 pair launch against two | 2.3-2.4 us saved a pair, back to back; the reference's c_gemv2 body alone reaches 341-358 GB/s against 460-546 | 512 to 4300 rows of 2304, int8 | inkspd1aq mb9 |
| Held Q8_0 list launch of four calls against two pairs | flat grid 22.3-22.8 to 19.8-20.3 us back to back, 27.4-28.0 to 24.6-25.4 flushed; the same calls over grid y 24.8-24.9 (empty blocks cost more than the launch saves) | 4300 and 4332 rows of 2304 on three twins, int8 | inkspd1ar mb10 |
| Normed query mix of three queries (one launch, per-query norm weight, one parts twin) against the mix, three `rms_norm` twins and two list launches | us a layer chain p50 of 5 reps at 1/2/4 items: 41.6/45.5/46.2 to 28.6/29.9/31.8 at 9 parts, 29.9/31.1/33.0 at 3, 34.1/38.1/37.4 at 1; every rep gains at least 5 us; bits equal | one row of 2304, int8, 6 launches to 2 | inkspd1as mb11 |
| Inkling-2-Mini decode launches a token, both cards | int8 1567 (spd/dense 1bc7baac), 1518 with the held list launch, 1485 with the normed query mix; f16 1661/1651/1585; the decode A/B of the last two is inside the host-noise band (fast-mode int8 means 67.2 against 68.0 tok/s, best 69.8 against 68.7) | tw2, rocprofv3 kernel counts, (tg48 - tg16) / 32 | inkspd1au t3, t4 |
| Production Q8_0 products | 755 GB/s, 9.01 ms a token | 27B Q8_0 int8 tw4, per-shape graphs | cuts14 |
| Production by projection | gate_up 723, down 671, qkv 621, head 767 GB/s | 27B, 11-14 | A:20342 |
| Ladder from read to product | 855 read, 800 held loads, 775 with y store, 692 with dot, 636 main | q_b 22.3 MB | A:28356 |
| Routed MXFP4 | 592 gate+up, 430 down vs 882/840 ceilings | one row | A:21801 |
| Routed, ALU bound | 349 GB/s, VALUBusy 87-89%, MemUnitStalled 0.4% | routed8 with cache-resident loads | A:17346 |
| sdot4 peak | 53.6 TOPS | compute bound above 16-20 rows a group | A:14593 |
| Host launch, hipModuleLaunchKernel | 1.2-2.0 us; 3.0-3.5 us with four threads to four devices | empty kernel | bypass31 D:1500, qdev28 |
| hipGraphLaunch / own AQL packet | 0.03 / 0.006-0.009 us per kernel | host side | bypass31 |
| Device cost per dispatch | 0.80 us gap, 1.25-1.34 us per empty dependent packet | same for graph and own AQL | bypass7, bypass31 |
| Device cost per small read kernel | about 0.88 us launch and drain plus 1.0 us dependency ramp | Fable fit | D:1500 |
| Boundary in the product stream | about 2.4 us a launch | (B-E)/443 | bypass22/24, D:562 |
| Dependent tiny kernel | about 6 us device time, 5.1 in a graph | 8B decode | STATUS, ROCm decode launches and host time (2026-10-30) |
| Hidden kernel arguments | 1.4-2.9 us a launch from reading gridDim.y | | backends-gpu Kernel notes |
| Barrier-AND, system fences, host kernargs | +2.0 us a dependency, +1.3 us, +1.2 us a packet | raw AQL | bypass7 |
| Two AQL queues | 5.72 vs 11.40 us a kernel: they overlap; one queue never does | | bypass31 |
| Dirty L2 after a writing kernel | about 1 us + 1 us/MB before a product | | D:1466 |
| Graph launch device time | 4-9 us; break-even 10-16 kernels a run | | launch1 |
| Host record cost | 5.3 us a launch (Inkling), 3.7 us x 864 = 3.2 ms a member (27B 1x4) | | inkgpu6, qdev26 |
| Dispatch stall | 510-512 workgroups of 256 threads stall about 40 us; 508 and 513 do not | RADV shows none | STATUS, The tile reduce within the resident workgroups (2026-11-19) |
| Code fetch | delta_step, about 27 KB of code, 32.2 us in situ vs 20.8 in bench: evicted from L2 and I-cache by the weight stream | | cuts22, D:1493 |
| Cold instruction fetch | one block, 8 KB of s_nop after an I-cache thrash, code still in L2: +0.1-0.15 us/KB straight, plus 25-50 ns a taken branch into a cold line (16/64/128 shuffled chunks +1.7/+2.9/+4.3 us); a kernel waits for its slowest CU, so one cold CU costs the whole launch | Inkling tw2 pre-expert chain about 8 us a chain in situ (query_mix4 runs about 6.5 KB of its 29.7 KB object; route_step_f32 25.6 KB; matmul_normed_f32 50.4 KB) | inkdec4/lr3, inkdec5/lr4 |
| First memory access | a kernel's first dependent L2 load 1.28 us, each later dependent load, glc load or atomic with return 0.25 us (one thread); a kernel cannot hide it, so a launch chain pays it per launch | | inkdec6/lr5 |
| Clock | idle mclk 350 MHz; after 2-9 ms idle the core runs about 1478 MHz for 1.3-9 ms; a 2 ms idle costs the next gate+up 22% | power cap 225 W | A:12722, D:297, D:1467 |
| Cross-card hop | 1.70-1.75 us one way (kernel polling own VRAM flag); 3.7-3.9 us through the CP barrier-AND | | bypass mb1c, sums4 |
| Link | 14.25 GB/s push per pair, Gen4 x8; 9.0 each way duplex; renderD137 Gen3 x8 about 7 | | vksum17 D:419, sums29 |
| Sum ceilings | kernel_sum_ceiling 8.71 / 8.30 / 8.74 GB/s per member at W2/3/4; all-reduce 9.3 / 8.99 / 8.97 | | sums4, D:1323 |
| Sum floor | per-epoch 3.9-5.4 us; real W4 decode sum 9.5 us = dispatch 2 + work 6.4 + re-arm 1.1 | small n | D:1294, D:1843 |
| Sums against RCCL | per window 5.82 / 7.60 / 10.95 us vs RCCL 16.79 / 24.87 / 31.07 at 256 floats W2/3/4 | | STATUS, Sums timed per window of calls (2026-11-23) |
| Peer reread | stale through the target's HDP read cache; only a device HDP flush fixes it | | sums46, sums47 |
| Timer | s_memrealtime runs at 25.0 MHz, steps 0.16 us | | bypass26, D:664 |

### MI50 under Vulkan (RADV)

| Limit | Value | Condition | Source |
|---|---|---|---|
| Read | 805 GB/s | | MULTI-DEVICE, The hardware, measured |
| Dependent dispatch | 4.1-5 us whatever it does; two streams on one card 1.02x | | same |
| Barrier floors | no barrier 0.05 us; exec only 0.72-1.15; compute memory barrier 2.20-2.30; product full barrier 3.60 | dependent dispatches | vksum43 dfloor |
| Pipeline switch | 3.1-3.2 us on the compute-only queue, 0.1 us on the graphics family | | A:1360 |
| Extra submission | 10-36 us of device time | | VULKAN, Structure |
| Row kernels | time follows instructions, not bytes: 64 and 2048 rows both about 10 us; the 151936x2048 head 138 GB/s | early 2026-10 | VULKAN, Kernels |
| Timestamps | inflate small dispatches, 26 us without vs 39.5 with | Q4_K 12288x2048 | VULKAN, Kernel notes |
| Routed tiles | MXFP4 float tile 24-28 GB/s vs 288 dense; one-row routed 272 | | A:15087, A:17555 |
| Sums | tagged vec4 lag 1 n1 11.1 / 13.0 / 15.4 us; ceiling 9.3 GB/s; large n 91-93% of it, small n 15-27% of the hop | | vksum29, vksum plan |
| Write-back-only barrier | W2 n1024 9.38 to 8.29 us a sum | | vksum44 |

### Radeon VII (Windows, AMD driver)

| Limit | Value | Source |
|---|---|---|
| Memory | about 1 TB/s; Q4_K matmul reads 390 GB/s of it, latency bound | VULKAN, Kernels |
| Shape | subgroup 64, 32 KiB shared, 128-byte push constants, no cooperative matrix | VULKAN, The device this is designed against |
| Issue | v_mad_u64_u32 at quarter rate took a third of the Q4_K kernel's issue slots; one buffer_load_ubyte per byte | VULKAN, Kernels |
| Channels | power-of-two-wide F32 rows hit one channel: 4096x2048 took 7-10x a 4096x2080 tile, on either card | VULKAN, Kernel notes |
| Missing | no launch, barrier or sum figure is recorded | gap |

### CUDA

| Limit | Value | Source |
|---|---|---|
| Shared memory an SM | sm_86/89 100 KB, sm_80 164 KB, sm_90 228 KB; block cap 48 KiB | CUDA, The int8 prompt tile on other cards |
| Registers | 128 registers sets two 256-thread blocks an SM; `__launch_bounds__(kThreads,1)` took 128 to 182-187 and halved blocks | same, A:24667 |
| IMMA fold | about 16 FP32 instructions per 32-fold; the tile caps near 35-45% of integer peak on Ampere, 50-60 on Ada and Hopper | same |
| Host issue of a sum | 10.6 us host vs 10.0 us member busy at 2048 floats; timed cuEventRecord 1.2-1.9 us, untimed 0.18 | STATUS, Tensor-group sums against NCCL (2026-11-18) |
| Links without peer access | 3080 Ti copy engines 33.9 down / 32.1 up GB/s, host cap 40-45; kernel stores 4.8-11 | sums4, sums25 |
| sm_90 | only tile8_tall spills of 4/8 B recorded; no bandwidth figure | cuda82 |

### CPU (Ryzen 7 5800X, AVX2)

About 50 GB/s; DOT_ROWS is the fused kernel's width, not a tunable; no software prefetch on sequential weight streams ([backends-cpu](src/backends-cpu.md)).
On the rig host, `CpuBackend::moe_step` read about 27 GB/s at host load 20 (A:19587).

## Radeon AI PRO R9700 (gfx1201), measured 2026-11-28

The hardware sheet for kernel work on the R9700 (Navi 48, RDNA4), written before further tuning there.
Every number says where it comes from: **DS** is an AMD datasheet or AMD document, **2nd** a secondary site quoting one, **ISA** the LLVM AMDGPU sources or AMD's matrix instruction calculator, **F** a formula from those, and **M** a measurement by the r97spec1 probes.
The probes are HIP kernels built with TheRock 10.1 `hipcc` for gfx1201 and run through the driver's `amdhip64_7.dll` (HIP runtime 7.2), and Vulkan shaders run on the AMD proprietary driver 26.6.4 (LLPC, driverVersion 2.0.331), on the R9700 sandbox box (rocm:1, vulkan:0), at default clocks.
Sources, logs and the box commands are in r97spec1 (`hip/`, `vk/`, `out/run2.log`, `out/vk4.log`); peak rates are the best of five timed launches (six in Vulkan) of a kernel of 3 to 24 ms, memory and clock runs the best of two or three; logs `out/run2.log`, `out/vk4.log` and, with random operands and buffers, `out/run3.log`.
HIP lists one R9700 (rocm:1); Vulkan lists it twice with one device UUID, a duplicate driver entry of the same card (per COORD), and the probes take the first.

### Specs

| Item | Value | Source |
|---|---|---|
| Compute units | 64 CUs, 32 WGPs; two SIMD32 a CU (4096 stream processors) | DS: ROCm GPU hardware specifications table (rocm.docs.amd.com/en/docs-7.2.4/reference/gpu-arch-specs.html); WGPs and SIMDs derived |
| Matrix units | 128 AI accelerators, two a CU | 2nd: kad8.com R9700 launch article |
| Wave size | wave32 native and the HIP default; wave64 supported | DS: ROCm table; ISA: GCNFeatures.td FeatureISAVersion12 |
| Boost clock | 2920 MHz; base and game clock not found | 2nd: servethehome.com R9700 article |
| Clock under load | 3.23-3.26 GHz on FMAs (one workgroup or every CU), 3.08-3.09 GHz on f16 WMMA over every CU, held for a 1.47 s kernel | M: shader cycle counter against the realtime counter, taken as 100 MHz (the Vulkan timestamp period is 10 ns; the counter measured 98.8-99.1 MHz against event time, which includes launch); inferred, not a clock register |
| Memory | 32 GiB GDDR6, 20 Gbps, 256-bit, 640 GB/s | DS: ROCm table (size); 2nd: servethehome, wccftech (speed, width, bandwidth) |
| L2 | 8 MiB (6 MiB on Navi 31), four 2 MiB slices | DS: ROCm table; 2nd: chipsandcheese.com RDNA4 architecture article (slices) |
| Infinity Cache (MALL) | 64 MiB; bandwidth not published | DS: ROCm table |
| Graphics L1 | none: on RDNA4 it is a read/write coalescing buffer, not a cache | DS: ROCm table (N/A); 2nd: chipsandcheese |
| L0 vector / scalar / instruction | 32 / 16 / 32 KiB | DS: ROCm table |
| LDS | 128 KiB physical, a WGP (inferred from the ROCm table, which gives no scope); 64 KiB addressable by one workgroup, 1 KiB granules, 32 banks | DS: ROCm table; ISA: GCNFeatures.td |
| VGPRs | 1536 a SIMD lane, 192 KiB a SIMD (768 KiB a WGP in the ROCm table); at most 256 a wave, so 256 VGPRs leave 6 waves a SIMD and 96 or fewer the full 16 (16 assumed from gfx11, inferred); dynamic VGPR allocation exists for wave32 compute but is off by default | ISA: Feature1536VGPRs, dynamic-vgpr; DS: ROCm table |
| Cache lines | 128 bytes for data and instructions | ISA: GCNFeatures.td |
| Board | 300 W TBP, PCIe 5.0 (x16 not confirmed) | 2nd: servethehome, kad8 |

### Peak per dtype

Formula: peak = ops a clock a CU x 64 CUs x 2.92 GHz, the clock AMD's figures use.
A wave32 16x16x16 WMMA does 8192 operations; f16 and bf16 take 16 cycles on a SIMD and the 8-bit forms 8 (ISA: AMD matrix instruction calculator, `-a rdna4`), so with two SIMDs a CU the f16 rate is 2 x 8192 / 16 = 1024 a clock a CU, and the 8-bit rate 2048.
The probes ran faster than 2.92 GHz, so a measurement can exceed a datasheet figure; the last column divides matrix rates by the measured 3.08 GHz and VALU rates by 3.23 GHz.
Matrix rates depend on the data: with random operands the Vulkan f16 rate fell from 185-209 to 169-184 T/s and the int8 rate from 396-409 to 350-357 (zero and splat operands ran faster), so the random-data figures are the ones to plan with; the HIP operands are small arithmetic sequences, not random.

| Dtype and path | DS dense / sparse (TFLOPS or TOPS) | F: ops/clk/CU x 64 x 2.92 GHz | M: HIP (T/s) | M: Vulkan (T/s) | M: ops/clk/CU at the measured clock |
|---|---|---|---|---|---|
| FP32 vector | 47.8 | 256 (128 FMA lanes x 2, dual issue counted) = 47.8 | v_fma_f32 23.2-23.4; v_dual_fmac_f32 41-42; wave64 v_fma_f32 43-45 | fma 24.2-24.7 (subgroup 32), 39.7-40.8 (subgroup 64) | 112-113 / 198-204 / 210-216 |
| FP16 vector, packed | 95.7 | 512 = 95.7 | v_pk_fma_f16 50-52 (wave32 and wave64 alike); v_dot2_f32_f16 45-47, wave64 79, VOPD v_dual_dot2acc_f32_f16 75 (inline asm) | f16vec2 fma 53 | 242-252 (pk_fma), 363-382 (dot2 dual issue or wave64) |
| FP16 matrix | 191.4 / 382.7 | 1024 = 191.4 | WMMA f32 accumulate 183-193, f16 accumulate 192-200, wave64 195-198; SWMMAC 16x16x32 336-389 | random operands: coopmat f32 accumulate 169 (32) / 180 (64), f16 accumulate 173 / 184 | 857-1015 |
| BF16 matrix | 191.4 / 382.7 | 1024 = 191.4 | WMMA 192-202; SWMMAC 349-364 | not usable (below) | 974-1025 |
| FP8 / BF8 matrix | 382.7 / 765.5 | 2048 = 382.7 | WMMA 325-333 (all four A/B pairs); SWMMAC 608-622 | not usable (below) | 1649-1689 |
| INT8 matrix | 382.7 / 765.5 | 2048 = 382.7 | WMMA iu8 351-352, wave64 353-360; SWMMAC 658-672 | random operands: coopmat s8 353, u8 357 (32); 350 (64) | 1776-1826 |
| INT4 matrix | 765.5 / 1531 | 4096 = 765.5 | WMMA 16x16x32 iu4 663-682 (16x16x16 iu4 runs at the int8 rate, 348-357); SWMMAC 16x16x64 1242-1286, while SWMMAC 16x16x32 iu4 (659-683) gains nothing over dense 16x16x32 | not exposed | 3363-3460 |
| INT8 dot, VALU | none | v_dot4: 8 ops x 64 lanes = 512 | v_dot4_i32_iu8 99-101, wave64 167; v_dot4_u32_u8 99-101 | dotPacked4x8AccSat 104-105 (32), 169-174 (64) | 479-489 / 808-842 |
| INT4 dot, VALU | none | v_dot8: 16 x 64 = 1024 | v_dot8_i32_iu4 198-202 | not exposed | 958-977 |
| FP8 dot, VALU | none | 512 | v_dot4_f32_fp8_fp8 100-101 | not exposed | 484-489 |
| Dependent WMMA, one wave | | 16 issue cycles (f16) | v_wmma_f32_16x16x16_f16: 27.9 cycles a dependent step, so a wave needs two independent accumulators, or a SIMD two waves, to issue at the full rate | | |

DS figures: AMD's R9700 spec table as quoted by servethehome, neowin.net and flopper.io (2nd; AMD's own page did not load, so its footnote was not read).
Sparse rows use the 4:2 structured-sparsity SWMMAC instructions and count dense-equivalent operations, as AMD's sparse figures do.
Discarded: a Vulkan `dotPacked4x8 + add` loop and the f16-accumulate and iu8 latency loops were folded by the compiler (impossible rates), and are not in the table.

**The FP16 matrix figure is 191 TFLOPS dense, not 95.** 95.7 is AMD's FP16 *vector* (packed VALU) figure; FP16/BF16 matrix is 191.4 dense and 382.7 with sparsity, and WMMA through HIP measured 183-202 T/s dense and cooperative matrix through Vulkan 169-184 with random operands: 87-99% of 1024 operations a clock a CU at the measured 3.08 GHz.
v_pk_fma_f16 has no dual-issue form and measured 50-53 T/s on both wave sizes; the 16-bit VALU paths that come nearest the 95.7 figure are v_dot2_f32_f16 in wave64 (79) and its VOPD pair v_dual_dot2acc_f32_f16 in wave32 (75), which the compiler did not emit on its own.
In wave64, FP32 FMA and the dot instructions run 1.7-1.9x their wave32 VOP3 rate, about what VOPD pairs give in wave32; v_pk_fma_f16 gains nothing.
int8 WMMA reaches 87-89% of twice the f16 rate on both APIs once the Vulkan operands are random.

### Memory and caches, measured

A read, write or copy kernel of 1024 workgroups of 256 threads sweeps a buffer of the given size as float4 at grid stride, passes times, and the rate counts the bytes requested (M: r97spec1 `out/run3.log`, HIP, with buffers of random data; Vulkan read in `out/vk4.log` and `out/run3.log`).
The read kernel issues one load a thread an iteration, so the cache rows are lower bounds of what the caches give; a variant with four loads in flight and a smaller grid read less below 40 MiB, so occupancy, not load depth, limits these rows.
Up to 4 MiB a thread rereads its own addresses each pass, so L0 hits may inflate those rows.

| Buffer | Read GB/s (HIP / Vulkan) | Write GB/s | Copy GB/s (read + write bytes) | Reading |
|---|---|---|---|---|
| 1-8 MiB | 6100-9200 / 5500-13000 | 3800-5000 | 6400-7800 | caches; no separate 8 MiB L2 edge shows |
| 10-32 MiB | 4600-7400 / 3300-6300 | 4100-4500 | 1600-6500 | copy touches twice the buffer, so its fall past 12 MiB is the same 32-40 MiB edge as reads |
| 40-128 MiB | 1500-2200 / 1000-1900 | 1800-2500 | 1200-1500 | the drop comes at 32-40 MiB, below the 64 MiB MALL size; inferred to be MALL hits |
| 256 MiB | 1100-1300 / 1290-1300 | 1580-1850 | 550 | unexplained: twice DRAM at four times the MALL size, with random data, so not compression |
| 512 MiB-2 GiB | 613-624 / 618-619 | 600-614 | 542-548 | GDDR6: 97% of 640 GB/s read |

A latency chase would place the cache edges exactly and is not done (gap).
LDS: ds_load_b128 at conflict-free addresses read 12.0-13.9 TB/s across the card in the default WGP mode and 17.0-17.2 TB/s compiled with `-mcumode`, and ds_load_b64 14.4 and 17.9 at 512 workgroups; at 3.2 GHz that is 64-70 bytes a clock a CU in WGP mode and 83-87 in CU mode.
The hardware LDS peak is not published; 128 bytes a clock a CU (32 banks x 4 bytes) is an assumption, so these figures are not called a fraction of it.

### Instruction support

ISA columns from LLVM's GCNFeatures.td (FeatureISAVersion9_0_6, 11 and 12) and BuiltinsAMDGPU.td; HIP and Vulkan columns are what the probes ran on this box.

| Operation | gfx906 (MI50) | gfx11 (RDNA3) | gfx12 (R9700) | HIP on gfx1201 | Vulkan, AMD driver 26.6.4 |
|---|---|---|---|---|---|
| WMMA f16, bf16 -> f32 / f16 / bf16, 16x16x16 | none | yes, A/B duplicated across lane halves, 32 cycles a wave32 f16 op | yes, no duplication, 16 cycles | all four (`__builtin_amdgcn_wmma_*_w32_gfx12`, `_w64_gfx12`) | f16 -> f32 and f16 -> f16; bf16 listed, no extension |
| WMMA fp8 / bf8 -> f32 (four A/B pairs) | none | none | new, OCP E4M3FN / E5M2, 8 cycles | yes | two float-accumulating types listed as 1000567000 and 1000567001, inferred to be E4M3 and E5M2 under a pre-release enum (the SDK header has 1000491002/3), VK_EXT_shader_float8 absent |
| WMMA iu8 -> i32 (signedness per operand, clamp) | none | yes | yes, 8 cycles | yes | s8, u8 and the mixed pairs, with and without saturation |
| WMMA iu4 -> i32, 16x16x16 and 16x16x32 | none | 16x16x16 | both | both | not exposed; two vendor types 1000142000/1000142001 with s32 results are listed and unidentified |
| SWMMAC (4:2 sparse A, index operand) f16, bf16, fp8/bf8, iu8, iu4 | none | none | new | yes, 2x the dense rate | not exposed |
| v_dot2_f32_f16 | yes | yes | yes | yes | through the compiler |
| v_dot2_f32_bf16, v_dot2_f16_f16, v_dot2_bf16_bf16 | none | yes | yes | yes | bf16 not exposed |
| v_dot4_i32_i8 (signed only), v_dot8_i32_i4, v_dot2_i32_i16 | yes | none | none | | |
| v_dot4_i32_iu8, v_dot8_i32_iu4 (mixed sign) | none | yes | yes | yes | 4x8 packed signed, unsigned and mixed reported accelerated; 16-bit dot not |
| v_dot4_u32_u8, v_dot8_u32_u4 | yes | yes | yes | yes | |
| v_dot4_f32_fp8/bf8 (four pairs) | none | none | new | yes | not exposed |
| v_pk_fma_f16, v_pk_mul_f16, v_pk_add_f16, v_fma_mix | yes | yes | yes | yes, no dual issue | yes |
| Packed f32 (v_pk_fma_f32) | none | none | none | | |
| VOPD dual issue (wave32) | none | yes | yes, includes dot2acc f16 and bf16 | v_dual_fmac_f32 emitted | |
| fp8 conversions (v_cvt_f32_fp8, v_cvt_pk_fp8_f32, stochastic) | none | none | new, OCP | builtins present | |
| v_cvt_pk_bf16_f32 | none | none | none (gfx950 and gfx1250 only) | | |
| Transpose loads global_load_tr_b64 / b128 | none | none | new (no LDS transpose load) | builtins present | |
| s_prefetch_inst / s_prefetch_data | none | none | new | builtins present | |
| Split wait counters (s_wait_loadcnt, dscnt, kmcnt...) | | | new | | |

WMMA operand layout on gfx12, wave32, 16x16x16 f16 (ISA: AMD matrix instruction calculator): A is 4 VGPRs a lane, lane `16*((k/4)%2) + i` holds row i, so lanes 0-15 carry k 0-3 and 8-11 and lanes 16-31 carry k 4-7 and 12-15, two halves a VGPR; B is the same with column j for row i.
The f32 result is 8 VGPRs, lane `16*(i/8) + j`, VGPR `i%8`; an f16 result packs two rows a VGPR in 4.
The 8-bit forms take 2 VGPRs for A and B, lane `16*(k/8) + i`, byte `k%4`; in wave64 every operand halves.
gfx11 code needs 8 VGPRs for A and B, duplicated in lanes i and i+16, and interleaves result rows by parity, so a gfx11 WMMA kernel does not run on gfx12 unchanged.
A WMMA whose A or B reads the previous WMMA's result needs one independent instruction in between; a C operand that does is stalled by the hardware (ISA: GCNHazardRecognizer.cpp, fixWMMAHazards).

The Vulkan driver reports subgroup size 64 by default with 32 to 64 selectable, `maxComputeSharedMemorySize` 32768 bytes (HIP gives a workgroup 64 KiB), cooperative matrix only at 16x16x16 subgroup scope, and neither `shaderBFloat16CooperativeMatrix` nor `shaderFloat8CooperativeMatrix`; creating a bf16 or fp8 cooperative-matrix pipeline anyway killed the process.
llama.cpp marks proprietary drivers below driverVersion 2.0.416 (Adrenalin 26.10.1) as having a large cooperative-matrix tile slowdown (ggml-org/llama.cpp PR 31587); this box reports 2.0.331.
RADV on gfx12 exposes the same 16x16x16 shapes with f16, bf16, int8 and float8 types (Mesa `radv_physical_device.c`); it was not measured here.

### What the reference engines use on RDNA4

Read as a reference only.
llama.cpp's Vulkan backend picks KHR cooperative matrix f16 tiles, an int8 cooperative-matrix MMQ on RDNA3 and RDNA4 (PR 27952, up to 1.12x on Q8_0 and 1.22x on Q3_K on an R9700, with Q4_1, Q5_1, Q4_K and Q5_K left on f16 because int8 was slower there), a 128x128 tile on fixed drivers (PR 31587, pp512 +22-32% on a 9070 XT), and cooperative-matrix flash attention (PR 31323).
Its HIP backend runs quantized products as int8 WMMA MMQ on RDNA4 (PR 17156) with the 16x16x16 f16, bf16 and iu8 builtins, and flash attention on the f16 WMMA tile; it uses no fp8, iu4 or sparse instruction.

### Implications for llmx

1. **Prompt tile: f16 WMMA first, int8 as the opt-in.**
   The default f16 class has a 191 T/s matrix peak, four times the packed VALU rate, and Vulkan exposes it now; R97TUNE1's cooperative-matrix Q8_0 tile measured 47.7 TFLOPS on 1024 x 4096 x 512 (`r97tune/cm1.log`), 25% of it.
   int8 cooperative matrix is fully exposed in Vulkan at twice that peak (350-357 T/s measured with random operands) and stays an opt-in under docs/PRECISION.md.
   A dependent WMMA takes about 28 cycles against 16 to issue, so a tile keeps at least two independent accumulators a wave.
2. **Ridge: the matrix cores need wide tiles or cache-resident weights.**
   At 195 T/s over the 620 GB/s DRAM read the ridge is about 315 operations a byte, so a weight read from DRAM is compute-bound only above about 89 prompt rows for Q4_K and 167 for Q8_0 in f16, and about 160 and 300 in int8 at 352 T/s (Where prefill crosses to compute-bound, below, gives the method).
   From the caches, at 4.6 TB/s or more up to 32 MiB (a lower bound), the ridge falls to about 42 operations a byte or less, so a tile order that rereads weight panels from them matters more here than on the MI50.
3. **Shared memory is 32 KiB in Vulkan on the AMD driver.**
   A Vulkan tile there has half the shared memory a HIP kernel gets (RADV's limit is not measured), so staging both operands in LDS limits the tile; loading B straight from global memory into the matrix operand, as R97TUNE4's xh candidate does, is the direction.
4. **Choose the subgroup size per type.**
   With random operands f16 cooperative matrix ran 6% faster at subgroup 64 and int8 1-2% faster at 32; FP32 FMA and the dot instructions run 1.7-1.9x faster in wave64 than as single VOP3 in wave32, about the rate VOPD pairs give in wave32, and v_pk_fma_f16 gains nothing from either.
   LDS reads ran 25-30% faster in CU mode than in the default WGP mode.
5. **K-quant dequantization is VALU work, far below the matrix rate.**
   The f16 arithmetic of an unpack runs on v_pk_fma_f16 without dual issue at about 50 T/s, a quarter of the matrix rate; the rates of the bit-field, permute and conversion instructions it also needs are not measured (inferred, gap); integer unpacking into an int8 WMMA with the block scales applied in f32 afterwards, as llama.cpp's HIP MMQ does, keeps the unpack off the f16 path.
6. **fp8, bf16, int4 and sparse are HIP-only on this driver.**
   They measured at their datasheet ratios through HIP; the Vulkan driver lists bf16 and fp8 types without their extensions, so a Vulkan kernel cannot use them until a driver exposes VK_KHR_shader_bfloat16 and VK_EXT_shader_float8.
   gfx12 has no f32-to-bf16 conversion instruction.
7. **Decode reads at 97% of nominal.**
   613-622 GB/s read from DRAM against 640; decode figures on this card take 620 GB/s as the ceiling.

Gaps: no cache latency chase, no sustained clock beyond 1.5 s or power reading, no bit-field or conversion rates, no RADV measurement, no sparse path in Vulkan (none exists), and the datasheet footnotes were not read on amd.com.

## From specs to shapes (2026-11-23)

A memory-bound kernel's shape follows from a few device numbers, and most DeviceProfile knobs are a measured knee of one of the rules below.

### The numbers that decide a shape, on the MI50

| Number | MI50 value | State |
|---|---|---|
| Practical bandwidth by pattern | stream 889-937, product 755-855, routed 206-623 GB/s | measured (Device limits, above) |
| Memory latency under load | not measured; 1-2 us assumed below | missing: a pointer-chase probe at load |
| L2 | 4 MB, 64-byte line, 16 channels | spec, not probed |
| VGPR budget | 256 per lane a SIMD: waves = floor(256 / VGPRs), at most 10; 84 gives 3, 128 gives 2, 129 gives 1 | measured cliffs: 84 at 3 waves (keff43), 128 vs 129 (cuts20) |
| LDS | 64 KB a CU; 23552 to 19456 B a block gave a third block a CU (keff17) | measured |
| Wave size | 64 | spec |
| CUs | 60, four SIMDs each | spec |
| Per-launch device cost | 1.25-2.4 us; about 4 us a dependent dispatch in a real pass | measured |

### Little's law

Bytes in flight needed = bandwidth x latency: at 900 GB/s and 1-2 us that is about 0.9-1.8 MB over 60 CUs, or 15-30 KB a CU.
At three waves a SIMD (12 a CU) that is 1.3-2.5 KB a wave, 20-40 bytes a lane: the 27B held kernel issues 20 dword loads before its first wait (80 bytes a lane, D:514) and reaches 711-822 GB/s.
At one wave a SIMD (4 a CU, a kernel above 128 VGPRs) each wave must hold 4-8 KB, 64-128 bytes a lane, which a K-quant row loop does not issue.
A call also pays a ramp: the first bytes arrive a latency after launch and the last wave drains alone, so a call of a few MB loses several us of a 5-10 us transfer.

Why single-token MoE decode sits at 16-30% of HBM, from moebench1 (Qwen3.6-35B-A3B Q4_K_M, one MI50 ROCm, 2026-11-23):

| Kernel | Calls a token | Bytes a call | Time a call | At 889 GB/s | Gap |
|---|---|---|---|---|---|
| q4_k_routed16 | 61 | 7.7 MB | 37.5 us | 8.7 us | 4.3x |
| q4_k_rows16_1 | 162 | 4.0 MB | 20.4 us | 4.5 us | 4.5x |
| q6_k_rows16_1 | 41 | 15.4 MB | 51.0 us | 17.3 us | 2.9x |
| q6_k_routed16 | 20 | 6.9 MB | 41.0 us | 7.8 us | 5.3x |

On q4_k_rows16_1 the 3-6 us fixed cost would account for 19-38% of the 15.9 us gap to the streaming floor. The other three rows give 9-21%. The table alone cannot distinguish memory latency, occupancy and instruction costs.
Limited bytes in flight are one candidate: the K-quant row builds held 206-249 VGPRs in the last count (A:15144), one wave a SIMD.
They also spend VALU on decode: Q4_K rows run 5.9M VALU for 50 MB where Q8_0 runs 5.8M for 95 MB (keff1).
The largest call, the Q6_K lm head in rows16_1, gets closest (29%), as the rule predicts.
Inkling routed16 (Q8_0, 84-ish VGPR class) reaches 625 GB/s in its microbench at nin 2304 and 491 at nin 1152, but 33% of HBM in situ (inkgpu7, inkgpu13): there the per-call ramp and about 2454 dispatches a pass dominate, not the loop.

### Shape rules

1. Grid: launch at least k waves a CU, with k the occupancy the VGPR count allows times the SIMDs, so every CU holds its share of the bytes in flight; a call of fewer rows splits along k.
2. Loads ahead: each wave issues its bytes-in-flight share before its first wait, kept batched (an asm "+v" over all loaded registers, bypass23), and no more: extra depth costs VGPRs and drops an occupancy step.
3. Split along k when rows are few: a narrow call's lanes or workgroups take parts of the inner dimension and sum once.
4. Merge launches: below about 4 MB a call is ramp-bound, so join adjacent products, fuse small chains, and record ahead; a launch removed saves 1.3-4 us of device time.
5. Stage a tile within LDS per occupancy step: size the staged operand so blocks a CU stays at the step its VGPRs allow (64 KB / blocks), never one byte over.
6. Fewer bytes and fewer instructions a byte beat more loads: decode products are fill-bound near 700-850 GB/s, so aligned planes, word loads and integer dots win, wider loads and prefetch lose (Patterns that failed, below).

### DeviceProfile knobs against the rules

The knobs live in `src/backends/device_profile.hpp`; the table gives the rule that predicts each and how the measured knee compares.

| Knob | Default (MI50 HIP / RADV) | Rule | Prediction against measurement |
|---|---|---|---|
| q8_decode_cols | 8 (- / 32) | 5: reuse a weight until VGPRs cross a step | ROCm: 10 columns at 84 VGPR, 3 waves (keff43, keff50); agrees. RADV 32 is not predicted |
| q8_decode_two_rows_from | 2560 (- / 4096) | 1: two rows a wave once rows / 2 still fill 60 CUs x 4 SIMDs x 3-5 waves, about 1440-2400 rows | measured knee 1280-2560 on ROCm; agrees. RADV 4096 is higher, its build holding more VGPRs |
| q8_decode_four_steps_from | 256 blocks | 2: four steps ahead once a lane has enough blocks | measured on the MI50; agrees in kind, not derived |
| q8_decode_narrow_below | 1024 | 2: a narrow row has too few blocks a lane to load ahead | measured; agrees |
| q8_decode_one_row_below_16bit | 1024 (RVII 1280) | 1 | measured; agrees in kind |
| k45_row_lanes | 32 | 2 and 3: lanes a row = row bytes / bytes a lane needs in flight; a 2048-wide Q4_K row is 1152 B, 36 B a lane at 32 | not measured per device; predicts fewer lanes for narrow rows |
| row_four_waves_per_cu | 8 | 1: k = 8 waves a CU before four rows a cluster | agrees with 2-3 waves a SIMD |
| tile_split_per_cu, _narrow, float_tile_split_per_cu | 8, 4, 4 | 3 | measured; agrees |
| tile_tall_per_cu, _8bit, _narrow | 1 (6 / 4, 4, 8) | 1 and 5: a tall tile needs enough workgroups to hide loads | ROCm knee about 6 a CU (STATUS, ROCm tall tile); RADV 4-8; agrees |
| qtile_tall_from, _8bit | 0 (4096 / 768, 2560) | 5: a narrow row gives too few blocks for the taller tile | measured; not derived |
| tile_from_8bit, _other and narrow forms | 32/64 (- / 16-40) | arithmetic intensity: the tile wins once a weight byte feeds enough rows | measured crossover; moves with the driver |
| moe_tile_from*, qtile_narrow_per_expert* | 32-96, 16 | the same, per expert's rows | measured; not derived |
| prompt_rows_below | 2 (9 / -) | intensity, ROCm decode rows up to 8 columns | measured |
| attention_split_chunk, _max, _max_wide, attention_tile_part | 32, 64, 64, 1024 | 3 along the history | measured; 1024 within 1% of 2048 and 4096 |
| attention_group_from, sparse_split_ids, argmax_slice, select_wide_from | 2048, 256, 4096, 4096 | 1: keep workgroups a CU until the work is long enough | measured |
| topk_chunk | 0 (MI50 HIP 4096) | 1: a block a row leaves 52 of 60 CUs idle at 8 to 16 rows; chunks a block, then a merge over their picks | measured on one MI50, 248320 scores, 16 picks: 8 rows 2.0 -> 0.18 ms, 15 rows 1.9-4.7 -> 0.23-0.28 ms, chunks 2048-16384 within 0.1 ms, ids equal |
| dispatch_chunk | 64 | 4 and the host-overlap rule | not swept against per-launch cost |
| held_push_members | 0 (3 / -) | sums: push pays only when the hop overlaps enough product | push lost at two members; agrees |
| prefer_integer_dot, lookup_integer_dot | false | none: a compiler property | Mesa gains 15%, the AMD driver loses 2% |

### Numbers DeviceCaps lacks

DeviceCaps holds subgroup size, CU count, shared bytes a workgroup and the integer-dot flag only.
A shape derived rather than swept would need these, each a probe:
- practical read bandwidth by pattern (stream, row dot, routed) and loaded memory latency;
- L2 size and line, VGPRs a SIMD with the waves each count allows, and LDS a CU;
- per-launch and per-dependency device cost, and host record cost a launch;
- cross-device hop latency and link bandwidth.
None of these is probed today; the MI50 values above come from separate microbenchmarks.

## Roofline recipe (2026-11-23)

1. Bytes a call: weight bytes read (routed: entries x expert tensor bytes) plus activations and outputs; count from the GGUF tensor sizes, as moebench1 did.
2. Floor a call: fixed cost + bytes / ceiling, with the MI50 held fit 3.3 us + bytes / 889 GB/s, or 1.3 us + bytes / 805 GB/s on RADV; use the ceiling of the access pattern, not the nominal 1024.
3. Compute check: integer ops / 53.6 TOPS (sdot4) on the MI50; a call whose ALU time exceeds its byte time is VALU-bound, and only fewer instructions help.
4. Take kernel times from a rocprofv3 or RGP trace at default clocks, with the per-dispatch tracing overhead (about 45 us under rocprofv3 kernel trace, fuse12) kept out of short kernels' sums.
5. Gap = measured / floor; sort by gap x time a token, and check the rule it breaks in From specs to shapes (above).
6. Pass level: the floor of every call plus one dispatch cost each (Device limits, above) gives the device time a pass can reach; host time beyond it is recording.

## Patterns that worked, to 2026-11-23

| Change | Mechanism | Gain | Kernel, device | Source |
|---|---|---|---|---|
| Convolution head on the waves the sum leaves idle | in a one-row llmx_conv_ring_sum, waves 2-3 copy the ring slot, load the weights and form every tap but the current one while waves 0-1 push and take; the summed row stays in shared memory for the last tap's fma and the post step (SGPR spills 25 to 58, scratch 16 to 0 B a lane, LDS 18.6 to 23.7 KB) | per call int8 13.0 to 11.8 us, f16 a tie (12.2 to 12.3 us) | llmx_conv_ring_sum, two MI50 ROCm | inkp2/mb8, inkp2/rv |
| A kernel table in device memory | `attention_rel_sink` took its views and block table from the argument ring in host memory (`arg`), so a pooled row read a block index over PCIe a key; `pass_arg` puts it in device memory as the other attention ops have theirs | traced p50 at tw2 int8, 64 tokens: global layers 51.2 to 40.0 us, ring 36.5 to 33.9 us, ring p75 50.9 to 35.0; about 0.17 ms a token, ids equal | llmx_attention_rel_lanes_g3, MI50 ROCm, Inkling-2-Mini | inkcore17/c3 |
| Push and take in separate waves | a one-row sum of at most a wave of float4 units: wave 0 loads its units and pushes them while wave 1 takes the peers' into the sum, so the take no longer waits behind the push stores and their barrier | 13.92 to 12.0-12.3 us a call (f16 probe); decode int8 68.3 to 73.3 tok/s, f16 a tie | llmx_conv_ring_sum, two MI50 ROCm | inkp2/rf, inkp2/rG, inkp2/rH, SUMINK9 |
| Code prefetch into L2 | a 64-byte line a thread up to the end label; the weight stream had evicted the code | 35.84 to 28.64 us bench; 66.02 to 67.20 tok/s | llmx_delta_step, MI50 ROCm | 57ee77537, cuts22, cuts25 |
| Code prefetch into L2 for attention's merge tail | after the q loads, a 64-byte line a thread up to the end label; the layer's other kernels evict the lanes build's 12 KB between layers | traced p50 at tw2 int8 over the table fix: ring layers 35.0-36.3 to 32.3-32.8 us, p75 50 to 41.6, global 40.8-41.9 to 40.0-40.2; about 0.1 ms a token, ids equal | llmx_attention_rel_lanes_g3, MI50 ROCm, Inkling-2-Mini | inkcore18/x1 |
| Four-way dot for Q8_0 rows | dot4_i8 for a scalar byte loop | 60 v_bfe_i32 and 43 multiplies to 2 and 0; 0.227 ms a token | llmx_delta_step | 2362d1fe3 |
| Mix loads without branches | query_mix_held loads a column past the row from the row's last column (no sum or store takes it) instead of a per-lane masked load, gates the sums by a select, and reads each item from a wave-uniform base; the 45 masked loads had compiled to 63 branches and split the batch | llmx_query_mix4 at 2304: n4 6.14 to 4.94 us in a graph, 8.24 to 6.95 L2-evicted, n2 4.10 to 3.56; llmx_query_mix_norm n4 7.90 to 6.68; bits equal over 5 dims, n 1-4, 10 seeds | llmx_query_mix4, llmx_query_mix_norm, MI50 ROCm | ink100/qm r11 |
| Router rows in one batch | route_step's exact_row loads 144 bytes a lane a batch (`kRouteBatchBytes`), a 2304-wide F32 row in one batch of nine steps instead of 4 + 4 + 1, same chain order | 10.40 to 9.28 us warm, 11.68 to 10.56 L2-evicted, 12.00 to 10.40 I-cache cold, bits equal | llmx_route_step_f32, MI50 ROCm, Inkling-2-Mini router shape | inkdec6/lr5 |
| int8 routed down walk + combine | a wave an entry with its twin step held, eight rows a block, the combine in the launch | 45.6 (half8s1 + moe_combine) to 21.7 us, bits same; the reference engine's down standalone 21.9 | q8_0_walk_combine int8, MI50 ROCm | inkspd1aj mb1 |
| Glu walk rows interleaved over the waves | wave w walks rows w, w + 8, ... in place of 4w to 4w + 3, so a block's waves read adjacent rows at each step; the int8 build held at three waves a SIMD | Inkling-2-Mini tw2 glu on one MI50: 38.8-40.0 to 34.1-35.2 us a layer (582 to 661 GB/s), bits equal | q8_0_walk_glu, MI50 ROCm | inkp5/mb out.r2, inkp5/runs |
| int8 routed down walk, two rows in flight | a row's buffer takes the row two ahead right after its dots and before its `wave_sum`, in the same two buffers; the old order waited `vmcnt(0)` every row, one 612 B row in flight a wave | Inkling-2-Mini tw2 down on one MI50: launch 26.7-27.2 to 25.8-25.9 us, bits equal, 37 VGPRs | q8_0_walk_combine int8, MI50 ROCm | inkp8/mb out.r3, out.r4 |
| int8 routed gate/up walk + GLU | each lane holds up to three half-block steps of the entry's twin, gate then up rows, the GLU and its int8 twin in the launch | in-model 70.1/74.0 (moe_group + routed_half8_g2 + silu_mul_twin8) to 64.6/60.4 us a layer, launches -2 a layer, ids same; standalone 42.1 us loop | q8_0_walk_glu int8, MI50 ROCm | inkspd1aj p4, inkspd1al mb5 |
| Fused step on the base's products | the base's alpha and beta products, then llmx_delta_step reads them: three launches to one at f16 and other types | 70.9 to 63.2 us bench, 27B 1x4 one row; 27B tw4 f16 tg 48.57 to 48.70 tok/s, inside the noise | llmx_delta_step, MI50 ROCm | perf/delta-f16 |
| Split-plane Q8_0 | scales apart, codes aligned | token 7749 to 7466 us, 12/12 rounds, bits same | held decode, MI50 | keff33 |
| Planes on multi-column rows and tiles | aligned loads issued together | 0.94-0.98 of file layout; int8 prompt tile 0.90-0.97; c16_r4 with hoisted loads 0.85-0.90 | MI50 ROCm | keff34, keff36, keff41 |
| 10-column build at 84 VGPR, 3 waves | stay on the occupancy step | 0.53-0.85 vs 0.77-0.97 | Q8_0 decode | 607bdf679, keff43 |
| 9-10 columns to the 11-column build | the scheduler serialises a 1-2 column tail | 0.615 of c10, bits same | Q8_0 decode | keff50 |
| Zero-min tile for IQ and Q1/Q2 types | fewer float ops; LDS 23552 to 19456 B gives a third block a CU | +8.4 to +38.6% on 11 types, hashes equal | EB tile, MI50 | keff17, keff22 |
| x staged in LDS once a block | fewer global loads | -2.1 to -7.6%, bits same | held Q8_0 | keff24, keff25 |
| DPP row_ror for wave sums | fewer LDS round trips | -1.6%, only at occupancy 3 (+27 us at 4) | held Q8_0 | keff30 |
| Spill removal | 256 VGPR and 124 B scratch to 206 and 0 | Q6_K 250 to 513 GB/s at 1-4 entries; Q5_0 242 to 464 | routed stream, MI50 | A:15142 |
| Routed stream kernel | per-entry weight streaming | groups of 1-4 Q4_K 209 to 554 GB/s; real routing 154 to 212 | MI50 ROCm | A:14809 |
| Lossless byte-plane loads | one 8-byte weight load | nin 1152 47.9 to 41.4 us, flat at 2304 | Inkling routed16 | inkgpu11 |
| Norm in 256-thread parts | parts sized to the CU | 9.4 to 8.3 us a call, about 0.15 ms a token | rms_norm | c53e9eed, cuts32 |
| Fused MoE step | 3 launches from 13 | 2x2 tg64 47.4 to 52.6 tok/s; route_step 27.9 to 19.3 us | MI50 ROCm | STATUS, ROCm decode loop (2026-11-06); A:20323 |
| Attention split by heads over a tensor group | each member its q heads and their KV heads, output by columns, summed and joined | Inkling-2-Mini Q8_0 f16 tw2 pp512 767.9 to 829.2; on spd/prompt 1122.2 to 1266.2 tok/s | MI50 ROCm, two cards | inkspd4 f2, g |
| Grouped and joined launches | projections in one launch, adjacent runs joined | 253 to 145 launches; 8B int8 x4 78.1 to 211.0 tok/s | ROCm products | STATUS, ROCm products join adjacent runs (2026-11-02) |
| Loads up front in small dispatches | issue every load before any wait | 2048-wide norm 11.4 to 7.4 us; 30B-A3B 120.0 to 129.0 tok/s | Vulkan | VULKAN, Kernel notes |
| Word loads for byte loads | fewer memory instructions | Q8_0 rows 32 to 201 GB/s; K tile 8B 193 to 240 tok/s | Vulkan | VULKAN, Kernels |
| Hoisted sub-scales | out of the per-value decode | 8B Q4_K_M 96.5 to 191.9 tok/s | Vulkan | VULKAN, Kernels |
| One module per type family | a shared module cost Q8_0 40% in registers | | Vulkan | VULKAN, Kernels |
| Static VRAM argument block | no host-pool kernargs | 5.80 to 3.86-4.13 us eager | norm_rope_kv | bypass17 |
| Raw AQL queue, VRAM kernargs | host at its floor | 13.71-13.83 ms vs graph 13.85-14.66, eager 15.45-18.37 | 27B tw4 | bypass17, cuts2 |
| A3 gated-attention fold | norm_rope and kv_write into attention | 7.2 us a layer at len 1; 0.55 ms a token on 27B tw4 | MI50 | fuse7, fuse13 |
| Sums per window, self-staged pull shares | one submit a window; shares rounded to 256 B | all 33 cells beat RCCL; W4 2.6M floats 1625 vs 2305 us | MI50 ROCm | sums43 |
| Square K tile | 128x128 for Q4_K-Q6_K | 27B pp512 901 to 1024 tok/s; A6000 702 to 935 | CUDA 3090 | cuda99 |
| Per-row attention over the decode rows alone | a mixed round launched blocks for its tiled prompt rows that returned at once (about 163500 of 168320) | prompt-round decode attention 8348 to 1710 us (two hosts); c16 1197-1224 to 1266-1303 tok/s (+6.1%), Qwen3-0.6B BF16; MI50 mixed round 1074 to 639 us; bits same | attention_vec and its merge, CUDA 3090 and MI50 ROCm | cuda151, cuda150/PLAN.md section 8 |
| exact_cols reduce-scatter | one weight load feeds G columns | c16 step 4869 vs 10503 us | CUDA | cuda123 |
| Whole-block staging in the Q8_0 prompt tile | a thread stages one 34-byte block (two four-word loads and a word) and one twin block (two four-word loads): 6 loads a step for 16, loop 1045 to 929 instructions at 84 VGPRs | routed 8-bit gate-up 4213 to 3435-3453 us, down 2362-2373 to 1940-1952 (reference tile 3726-3728, 1934-1935); 16-bit half-block 6175-6196 to 6045-6081 | MI50 ROCm, `q8_0_tile` | INKSPD3ab, inkspd3aa/mb res_g |
| Waves a multiple of the SIMDs | the rel tile's three heads' rows packed over 16-lane slots, 4 waves a block (2 blocks a CU) where 6 waves on 4 SIMDs left two SIMDs paced by the other two at every barrier | 2048 rows, 6 heads over 2 kv heads: 1.893 to 1.281 ms, window 1024 1.529 to 0.999, same bits; 8 waves x 21 rows 1.334, 1 row a slot 1.865, 3 rows a slot (154 VGPRs) 2.134 | MI50 ROCm, `llmx_attention_rel_tile_g3` | INKSPD3af, inkspd3af/mb |
| A build per item count range | a kernel takes the registers of its widest template case: `query_mix` up to 4 items in its own build, 64 VGPRs against 186 | 1024 rows x 2304: 2 items 84.6 to 60.0 us, 4 items 155.4 to 84.2, same bits; a 5-item build stays at 132 VGPRs and gains 4-8% | MI50 ROCm, `llmx_query_mix4` | INKSPD3af, inkspd3af/qmb |
| A block a row over all its queries | an item of at most five held rows loads its row and its squares once and loops over its queries, one launch an item and one build a row count (`llmx_query_mix_rows1` to `5`); a call of fewer rows than compute units keeps the block a (row, query) builds | 2048 rows x 2304 on one MI50: 3 rows x 3 queries 350 to 191 us, 4 x 4 576 to 343, 4 and 5 rows x 4 and 3 queries 1763 to 899, same bits; one query on 5 rows 307 to 410 us, so those stay on the old builds; a build of 5 items at 132 VGPRs and a peeled first query (23 to 130 VGPRs) were slower; Inkling-2-Mini Q8_0 tw2 pp2048 int8 2362.5 to 2384.9 tok/s (+1.0%), f16 1751.9 to 1764.3 (+0.7%), medians of 16 slices over 4 interleaved rounds | MI50 ROCm, `llmx_query_mix_rows1` to `5` | INKSPD3al, inkspd3ah/j3 |
| Constant shape behind one entry switch | a grid-barrier kernel of runtime width (llmx_normed_gate_mix) switches once at entry to a body whose width, loop passes and loads are constants; the item is taken by value, since a reference into the call kept a 376-byte private copy (384 B/lane scratch) and put the barrier word in vector registers | one item 15.8 to 12.8 us, three 21.9 to 20.6 us in a graph, bits equal | MI50 ROCm | ink100/qm qn4 rn5 |
| Route build per decoder pair | a runtime F16/BF16 branch held 148 VGPR in the F32 build; one build a router and row-gate decoder pair keeps the F32 one at 65-68 | route r1/r4/r8 17/29/52 to 13/19.5/33 us (trace p50) | llmx_moe_step_route, MI50 ROCm | dspine28 p1, dspine29 kcf-16 |
| Entry-fastest step grids | rows of h the grid's fast dimension, so an expert several rows pick is read from DRAM once | down_q8_0 r8 143 to 130 us; Q8_0 gate step r8 414 to 354 us | llmx_moe_step_up/down, Qwen3.6-35B-A3B shapes | dspine28 m8 |
| Ids, weights and y0 staged with the twin | the down's combine waited on 2k serial global loads and each task on one; the limit was the vmcnt chain of dependent L2 reads, removed by the staging | step medians r1/r4/r8, main to staged: Q5_1 down 62.8/170.2/274.9 to 62.2/147.9/258.1 us, Q8_0 69.3/169.6/311.2 to 65.4/152.7/250.3 us | llmx_moe_step_down, Qwen3.6-35B-A3B shapes | dspine29 m10 |
| lane_row: twin before the weights' decode, shift trip count, fold by selects | the twin's LDS reads and sum run under the weight load's wait; the runtime stride's division and branchy fold were the instruction excess against main | down_q5_1 r1 25.1 (w12) to 20.8 us (trace p50); step Q5_1 r1/r4/r8 62.8/151.4/274.4 (main) to 60.1/143.2/241.4 us | lane_row, MI50 ROCm | dspine29 m11 |
| exact_cols wave32 geometry, q/k/v and gate/up one launch | R4G2 over 8 waves past 8 rows, R2G2 over 4 below; fewer launches | 0.6B 28-layer round at 16 rows 4.79 to 3.10 ms, 8 rows 2.86 to 2.21, 0 bits differ (harness) | CUDA 3090 | cuda143, cuda145 |
| exact_cols many-row build without spills | last whole step widens at load, the others hoist raw bits | 112 B stack and 160 B spills to 0 at 128 registers, sm_86 ptxas | CUDA | cuda145 |
| exact_cols F32 column builds without spills | no raw hoist on wave32; many-row build one block an SM | few 8 B spill to 0 at 96 registers, many 32 B to 0 at 166; at 128 registers every F32 form spills 68-320 B, sm_86 ptxas | CUDA | cuda148 |
| exact_cols 16-bit many-row build without spills on sm_70 and sm_120 | one block an SM there, two elsewhere (`LLMX_COLS16_BLOCKS`) | 16 B (sm_70) and 20 B (sm_120) spills at 128 registers to 0; no card of those targets timed | CUDA | cuda149 |
| Ring conv prompt rows in parts | the staged build walks each 16-channel strip's rows one dependent load and store at a time (more staged blocks, any part size: 108-111 us, null); llmx_conv_ring_parts gives a thread a channel over 8 rows on grid.z, its halo from the input and all its loads before its first store, from 128 rows where no item writes its own input | 1024 x 2304 added at a stride 109.9 to 57.8 us, bits equal; Inkling-2-Mini Q8_0 int8 pp2048 tw4 3257.5-3260.0 to 3312.0-3316.5 tok/s (+1.7%), tw2 2401.8-2404.1 to 2427.8-2428.1 in its clean rounds; post convs 24.2 to 12.8 ms a pass | llmx_conv_ring_parts, MI50 ROCm | inkpp5/e mb3, runs/time.out |

## Patterns that failed, to 2026-11-23

Each line is the arm, the reason as read at the time, and its numbers; "implementation" marks a loss later reversed by another build of the idea.

| What | Why | Numbers | Source |
|---|---|---|---|
| Decode exchange as producer push + one flag (INKP2, PLAN P2): fenced 16 B block pushes in the int8 attention and down producers, the flag raised at llmx_conv_ring_sum entry | each producer block waits on its cross-card stores at the system fence (vmcnt(0) + L1 invalidate) before it retires, and the wait for the flag serializes with a plain take where the armed take overlapped the skew; a 5.4 us conv entry floor that both forms pay was never the push | Inkling-2-Mini Q8_0 int8 tw2, two MI50: tg256 57.9-67.0 to 52.0-60.8 tok/s, per call 13.9-14.4 to 15.7-16.3 us, ids equal | inkp2/r4, inkp2/PLAN-FEEDBACK.md |
| Flagged push in the int8 routed down walk (barrier, LDS, two 16 B stores a peer, fence) | same registers (26 VGPRs); the barrier carries vmcnt(0) on the y stores and the fence the cross-card ack, so every block lives longer, paid over two generations (240 of 288 resident) | +1.7 to +2.4 us an exchange, bits same | inkp2/mb2 |
| Alternating the members' flush order at two members | none at W=2: the early card waits push + hop and the late one finds its rows, whatever the order | not built; an alternating 2 us skew case measures the other ideas instead | Fable a3adf3b5 |
| Q8_0 half words as one unaligned dwordx4 plus a dword (X4D) | slower at every depth | down 18.75 to 22.7-32.3 us a layer, glu 40 to 49-50 | inkp5/mb out.r1 |
| Walk rows staged through LDS (unique dwords or dwordx4 rows) | the staging costs more than the loads it shares | down 18.75 to 22.5-44.9 us; glu 39.5-40 to 36.8-38.4, below the interleaved rows | inkp5/mb out.r1 |
| Deeper row prefetch (4 or 8 rows) or more rows a block (16 to 64) in the down walk | the registers cut the waves a unit holds | D8 28-34 us; 16 rows 17.9-19.4 (= production); 32 rows 21-25; 64 rows 26-29 | inkp5/mb out.r1-r3 |
| Residency cap by dynamic LDS on the down and glu walks | fewer resident blocks stream slower | down 19.5 to 19.7-34.3 us; glu 40.1 to 40.1-43.0 | inkp5/mb out.r3 |
| Walk blocks looping groups in a smaller grid (down G144/96/72, glu 8 or 4 waves, G72/48/36, stride, contiguous, entry- or chunk-major, carry of the next row 0) | the loop's barrier and partial buffers cost more than the ramp they save | Inkling tw2 one MI50: down 19.2-19.6 to 24.4-36.9 us a layer, glu 34.1-34.9 to 36.7-43.9, bits equal | inkp7/mb out.r1, out.r2 |
| Touch-ahead of the next group's lines in the looped walks | the touches take issue slots and hold registers | down 24.6-28.1 to 36.0-44.0 us a layer, glu 36.7 to 65-94 | inkp7/mb out.r2 |
| Down walk at 16 outputs a block (144 blocks, all resident) | the same throughput in one round; with two rows in flight 0.3-1.3 us slower | launch 26.7-26.9 (depth 1), 28.0-28.2 (depth 2) against 26.7-26.9 | inkp8/mb out.r3 |
| Fewer vector instructions in the down walk (scalar row address by `first_lane`, one select of the dot words) | does not stack with two rows in flight; at L2 residency the time does not follow the instruction count (21.4 us at 467 and 374 vector instructions) | launch 25.6-26.1 against two rows in flight 25.8-25.9 | inkp8/mb out.r4 |
| Query mix as four column-owned blocks (block b = wave b, wave partials through global memory and a flag) | bits equal, but every thread still runs all nine columns of every item, so the split cuts no per-thread work and adds the exchange | n4 7.2-7.6 against 6.14 us in a graph | ink100/qm r2, r5 |
| route_step pick: score and bias loads issued before any arithmetic; a radix-select top-k (32 ballot steps); an LDS rank pick | the pick is the issue rate of one wave's dependent vector instructions (about 0.23 us a round of about 60), not load latency | pick alone 4.6 us to 5.0, 5.0 and 7.5 (the reference's rank pick 6.7) | inkdec6/lr5 |
| route_step's norm share and twin stored after the counter by waves 1 and up | within noise once the rows take one batch | 9.44 alone, 9.28 with one-batch rows against 9.28 | inkdec6/lr5 |
| 2-byte aligned 16 B loads in the int8 walk; all eight rows loaded ahead | the word realignment is not the cost; eight rows of registers cut occupancy | 21.7 to 22.0 us; 21.7 to 30.1 us; bits same | inkspd1aj mb1b |
| Word or wider loads in routed16 | VALU-bound, not load-bound | word loads 7% slower, bits same; 16 B lanes 77.7 vs 73.2 us; three aligned dwords 72-83 to 77-89 | inkgpu8, inkgpu11, D:4422 |
| More rows a wave or workgroup | waves drop, barrier coupling | 8-row workgroups +8.3%; R=2 held 8.53 to 9.50 ms; R4 no gain, R8 worse on routed16; two rows a wave 36.6 vs 29.5 us | keff32, cuts16, inkgpu8, STATUS ROCm decode products |
| Deeper loads, depth 2-3, load-ahead | VGPRs cross a step, waves 2 to 1 | spills 12-61 VGPR, 8.79 vs 7.94 ms; K rows 89.6 to 102.0 us at 60 to 81 VGPR; routed8 2-4 ahead 310/295/258 vs 349 GB/s | bypass23, keff1, A:17349 |
| Early loads at 16 columns | registers | 1.32x slower | STATUS, ROCm few decode columns (2026-11-20) |
| Q8_0 planar repack, first builds | implementation | qkv 29.99 vs 31.91 us; +4.7% under its 800 GB/s rule; loadpl 661 GB/s vs 800 | cuts14/15, D:4635, inkgpu13 (keff33 later won) |
| Group-interleaved planes | | 0.937-0.987, never better than row planes; tall tile 1.025 | keff39, keff34 |
| L2 touch-ahead of the next product | competes for HBM | 9.84/10.75 vs 8.26 ms; +1.5-2.3 ms | bypass22, D:499 |
| Persistent kernels and grid barriers | a launch boundary costs about 0 on a fill-bound stream | 8.35 vs 8.26 ms; MB4b 15.6 vs graph 13.6 ms; needed -1.0 ms, got 0.15-0.5 | bypass23, bypass26 |
| Own AQL ring, two-queue chaining | device side equals hipGraph | -0.33 to -0.56 us a dispatch, under the kill rule | bypass31 |
| Graph replay of held ops | the about 4 us floor is device side | saves only host time; fuse first | I: 2026-11-23 inkling-gpu-core |
| Barrier removal on Vulkan | barriers are 0.7-3.6 us of a 4 us dispatch | range-based hazard tracking estimated at about 1 ms a token, untried | I: 2026-11-23, vksum43 |
| Nontemporal stores | | 8.28 vs 8.29 us | vksum44 |
| Per-lane global x without LDS | | +13.4%; XLGX 8901 vs 7743 | keff32, keff33 |
| Unaligned 16 B code load | | +12.1% / +7.8% | keff29 |
| held_t on tiles | 129 vs 128 VGPR occupancy cliff | c16r4 +27 to +55% | cuts18, cuts20 |
| LDS cap once VGPRs bind | | 1.06-2.18x; uniform pad o 12.15 to 16.24 | keff41, keff28 |
| One barrier rule for all C>8 | occupancy 2 to 1 | c16_r4 1.35-1.72x | keff49 |
| Take+norm, silu_mul and norm folds | under the 1 us rule, or more bytes a launch | within 0.3 us; 0.6B +1.5% but 27B loss; SiLU tail missed -0.2 ms in every column | cuts12, fuse5, 5843b0f29, STATUS (2026-11-01) |
| Tile push in the product | drain-bound, about 8 ns a remote store | tg256 63.8 = 63.8; lost at two members | cuts14, cuts16 |
| Recorder lanes per stage, more prompt slices | devices already 75-77% busy | l4 pp2048 2113 to 1954; 9 slices 2013, 16 slices 1698 tok/s | fngate2/3/4 |
| Kernel copies under 2 MB | the slicer chose 17 slices | pp2048 1486 to 1139; at 5 slices 1496 vs 1491 | dsdev263, dsdev265 |
| Guess-ahead expert pull | | tg128 60.0 to 55.8/49.8 | moeoff21b |
| Copy queue on the transfer family | | -38% on 27B 1x4 | tscq1 |
| No-P2P ring, CPU reduce, LL8/LL16 sums | link-bound bytes | 5641 vs 2924 us; 1015 vs 552 us; 2x bytes | sums22, A:18171 |
| Pulls (host writes) for the prompt sum of an Inkling-2-Mini chunk at width 2 | a load-time pick took pulls in 2 of 12 loads on a 0.03 ms edge and a pass then shows 96 host-visible gaps | `rates forms`, 2048 x 2304 (18.9 MB) behind a Q8_0 product, 5 rounds, MI50 pair: shares and one_shot 2.0-2.1 ms a sum every round, pulls 2.01-3.94 ms (rounds 3.15, 2.16, 2.01, 3.94, 2.08), so pulls never clearly win and sometimes cost 2x; kept as a guard, pulls and copies are picked only below 0.7 of the best kernel form in the interleaved duel | SUMINK1, SUMINK6, SUMINK7 |
| Sixteen lanes a block (Vulkan) | in-order load counter | 245-272 GB/s, fewer registers | VULKAN, Kernels |
| One-column build of wide Q8_0 (Vulkan) | | -9% of 8B decode | VULKAN, Kernel notes |
| Attention four heads a block (ROCm) | | 24.6 vs 17.7 us | STATUS, ROCm decode launches and host time |
| K rows: waves_per_eu 4/5, lane-fixed index, K45 pair | spills, VALU | 10-37% slower | keff1 |
| Launcher thread per GPU backend, sums as launcher operations (two cards) | members drift; sum spin waits grow from 6.7 to 140-270 ms a trace; the members' replicated attention, not host recording, bounds width 2 | Inkling-2-Mini Q8_0 f16 ROCm tw2 tg64 28.09/30.63 vs base 37.64/32.01, one card 37.01/37.41 vs 38.23/34.18 tok/s | inkspd4 t2, STATUS two cards for Inkling-2-Mini |
| Flush each tensor-group member once its part is recorded | implementation; same drift | tw2 tg64 33.39/16.28 vs 37.64/32.01 tok/s | inkspd4 t2 |
| Member threads with a join at every part (two cards) | 96 joins a token on a loaded host | tw2 tg 21.98/20.75 vs 28.97/31.56 tok/s | inkspd4 a1 |
| CUDA BF16 prompts, cols8, KT_FULL | | pp512 2988 vs 9379; cols8 912/938 vs 1245/1516 tok/s; no gain | cuda102, cuda121, cuda87kt |
| route_norm loop rolled (unroll 1 or 4, x re-read for the store, same sums) to cut cold code | the warm loss exceeds the cold gain | route_step_f32 warm 14.4-14.7 to 18.6-19.5 us, after an I-cache thrash 17.0-17.1 to 20.3-21.9 | inkdec5/lr4 |
| Code prefetch into L2 for the pre-expert chain; a warming kernel before it | the code is already in L2 and the cost is the per-CU instruction cache; walk and attention evict between | null in lr3 | inkdec4/lr3 |
| Fused step: no next load on the last trip (lane_row) | the guard made the Q4_K up wait on each unit (vmcnt) | up r8 126.7 to 138.6 us; a one-trip path before the loop took its place | dspine28 m8 |
| Fused step down through lane_row at a runtime stride | division, clamps and a branchy fold: about 120 more instructions a wave at 4 waves a SIMD | down_q5_1 r1 22.4 to 25.1 us (trace p50) against the hand-written main kernel | dspine28 m9, Fable |
| Prompt chain conv + mix + norm + twin in one row kernel (Inkling FFN side, 1024 rows x 2304) | the ring conv keeps its lags in registers, so the fused kernel builds at about 200 VGPRs (one wave a SIMD) and loses the staged conv's parallelism; fusing only the mix, norm and twin behind the conv wins in the microbenchmark (next row) | chain 343.8 us; fused a block a row 414.9 us (214 VGPRs), 2/4/8 rows a block 383.0/413.8/465.8 us (198 VGPRs); bits identical | inkpp5/mb2 r7 |
| Prompt chain mix + norm + twin in the row-form query_mix launch behind the staged conv (Inkling FFN side, 1024 rows x 2304) | projected <1%, parked, not measured integrated (go3 A/B contaminated by concurrent chains); its candidate also failed 12 launch-count and twin-read checks in tests/backend_rocm.cpp | microbenchmark chain 343.8 to 275.0 us, -69 us a chain, about 6.6 ms of a 0.85 s tw2 pp2048 pass over 96 chains; bits identical | inkpp5/mb2 r7, go3 |
| moe_step_down: staged global reads, shared Q8 row, staged twin | not load-issue bound | 0-6 ns a workgroup; twin slopes Q5_1 37 to 39, Q5_K 49 to 49, Q6_K 72 to 78, Q8_0 45 to 42 ns, VGPR +4 to +8 | dspine30 m6, m7, m8 (2026-11-26) |
| moe_step_down: full-row lane_row, branchless k45_scale_min, Q6_K signed decode | VGPR occupancy, not VALU: eight extra dot4 a segment and a hot row are both null | full row Q5_K 46 to 131 ns at 80 to 136 VGPR, Q5_1 39 to 52 at 56 to 84, Q6_K 63 to 63; branchless Q5_K 47; signed Q6_K 63 to 71 at 68 VGPR, inside its +-10 ns band; bits identical | dspine30 m9 (2026-11-27) |
| moe_step_down: staging per 16 or 32 rows | the fixed cost is the dispatch | exit before staging 4.9 us a call at any grid, staging and barrier 2 ns a workgroup of 47-64; at most 2-3 ns to gain | dspine30 m10 (2026-11-27) |
| moe_step_down: amdgpu_waves_per_eu(4) on every down build | occupancy is confirmed as the bound, but forcing 64 VGPRs spills: Q5_K needs a smaller lane state, not a register cap | dummy LDS for 4/3/2 workgroups a CU: null / Q5_1 40 to 49, Q8_0 43 to 56, Q5_K 47 to 48 / Q5_1 66, Q5_K 66, Q8_0 78 ns a workgroup; Q5_1 padded to 68 and 83 VGPR both 49; the cap takes Q5_K from 79 to 64 VGPR with 40 bytes a lane of scratch, 47 to 58-59 ns a workgroup and 19.7 to 28 us at one row; bits identical | dspine30 m11 (2026-11-27) |
| moe_step_down: late K45 header with packed scales (rd), and with the header before the next prefetch (m13) | rd takes Q5_K to 63 VGPR, no scratch, 4 workgroups a CU and the slope from 47 to 41 ns a workgroup, but the header read lands on each unit step and the scale packing in the unsplit load also slows the shared dense k_rows | down_q5_k at one row 19.7 to 20.5 us, up_q4_k_c2_w 27.5 to 31.4 us; Qwen3.6-35B-A3B UD-Q4_K_XL int8 on one MI50 plain 144.14 to 139.71 tok/s (-3.1%), MTP k7 228.49 to 221.71 (-3.0%); Qwen3.5-9B Q4_K_M plain f16 83.22 to 77.75 (-6.6%), int8 91.41 to 86.55 (-5.3%); m13 down_q5_k 21.3 us, slope 41, also null | dspine30 t35, t36, m13 (2026-11-27) |
| Host ring of 16 slots in place of kRingSlots 4 (ROCm decode sums) | the period-3 decode burst is host issue under CPU load, not ring reuse | 27B Q8_0 int8 tg256, 16 vs 4 slots: tw4 71.75/71.16 vs 71.53/71.13, tw2 45.60 vs 45.59 tok/s | sums97, D:18041 |

Recurring readings across the ledger:
- Microbenchmark wins shrink or reverse in situ: tile push, held_t and the held-stride arm all did, from a wrong shape or build flag, a cold I-cache or a dirty L2.
- Occupancy steps decide more than arithmetic: a change that costs a step loses whatever it saves.
- Decode products are fill-bound near 700-850 GB/s, so fewer bytes and fewer instructions a byte win; more loads in flight do not.
- Device-side floors of about 4 us a dependent dispatch cap fusion gains and make replay a host-only saving.

## Prefill: the compute side (2026-11-23)

Decode is a memory problem and most of this page is about it; a prompt pass is the other half.
With 64 or more rows against each weight read, the weight bytes are amortised and the arithmetic sets the time, so the tools change from bytes and dispatch counts to instruction counts, registers and LDS.
Peaks below are derived from the hardware or the spec sheet unless a source names a measurement; a derived figure is marked so.

### Compute roofline per device

| Device | Peak | Value | Basis |
|---|---|---|---|
| MI50 gfx906, ROCm | F32 FMA | 13.3 TFLOPS | 60 CUs x 64 lanes x 2 x 1.725 GHz; derived |
| MI50 | packed f16 (v_pk_fma_f16) | 26.5 TFLOPS | two FMAs a lane; derived; no llmx source names it (unverified that any kernel emits it) |
| MI50 | dot2 (v_dot2_i32_i16, v_dot2_f32_f16) | 26.5 TOPS | two MACs a lane; the 16-bit twin tiles (the f16 class) run on it; keff1 uses 26.5-26.8 |
| MI50 | dot4 (v_dot4_i32_i8) | 53.0-53.6 TOPS | four MACs a lane; the `--dtype int8` tiles; keff1, A:14593 |
| MI50 | no MFMA | none | gfx906 has no matrix core and no cooperative-matrix path, so every prompt product is a VALU dot |
| MI50, effective | clock under load | 1567-1609 MHz of 1725 | GRBM_GUI_ACTIVE over duration at default clocks; 7-9% of any nominal peak is clock, not code; keff1 KEFF8 |
| Radeon VII | same gfx906 silicon | about 13.8 TFLOPS F32, double and quadruple that for dot2 and dot4 | 60 CUs, 1.8 GHz peak boost; derived, no prefill rate is recorded on it (gap) |
| MI50 under RADV | float tile | 4.65-4.87 TFLOPS, 35-37% of F32 peak | 4096 x 14336 over 512 rows, Mesa, 2026-09 history; VULKAN, Kernels |
| RTX 3090 sm_86 | F32 | 35 TFLOPS | F2DEV review, A:21649 |
| RTX 3090 sm_86 | tensor cores | f16 and int8 spec-sheet peaks not recorded here | unverified; measured int8 tile rates are in the CUDA rows below |
| sm_90 | tensor cores | no rate measured | only compile checks and tile8_tall spills exist (cuda82); CUDA.md says Hopper doubles the IMMA rate an SM |

Measured CUDA prompt rates on sm_86: the int8 IMMA tile reads 59 TOPS on the square gate_up shape and 43-48 on the 64-output qkv and down products (I:607), and Q6_K tall reads 49 (I:608).
CUDA.md caps the tile at 35-45% of integer peak on Ampere and 50-60% on Ada and Hopper by the fold's 16 FP32 instructions an IMMA; the peak it means is not stated there, so the percentage is unverified against the spec sheet.

### Where prefill crosses to compute-bound

A tile that feeds R prompt rows from one read of a weight does 2R operations per weight.
The device's ridge is its peak divided by its read ceiling; on the MI50 the read-only graph reached 844 GB/s (bypass24, D:553), so dot2 has a ridge of about 31 ops a byte and dot4 about 63.
Rows at the ridge R = ridge x bytes a weight / 2, derived with Q4_K at 0.5625 bytes a weight and Q8_0 at 1.0625:

| Type | dot2 ridge | dot4 ridge |
|---|---|---|
| Q4_K | about 9 rows | about 18 rows |
| Q8_0 | about 17 rows | about 33 rows |

This agrees with the measured crossovers: the sdot4 product is compute bound above 16-20 rows a group (A:14593) and the profile's `tile_from_8bit` is 32 (From specs to shapes, above).
A 64-row column tile at pp512 sits above every ridge, so the K tiles are compute-bound and the counters say so: VALUBusy 80-85% and MemUnitBusy 17-28% on the 27B shapes (keff1, KEFF6).
The derivation ignores activation reads, and the tile re-reads weights once a column tile from L2, which is why a tile reaches 40-55% of the peak and not more.
Below the ridge, at 1 to 8 prompt rows, the product is a rows kernel and the decode rules apply.

### What sets a tile's ceiling

1. **Instruction class, by precision rule.**
   The default path computes in the class its dtype names: the f16 class runs the 16-bit twin on dot2 at 26.5 TOPS peak and int8 on dot4 at 53, twice the rate.
   8-bit activations are an opt-in with their own bound and never the default (docs/PRECISION.md; AGENTS.md, Traps the gates exist for), so a prompt speedup from int8 is reported beside the f16 figure and never replaces it.
   On the 27B Q4_K_M, int8 reads 353-393 tok/s against f16 251-262 (keff1, KEFF7 fm7), 1.4x for a 2x peak, because int8 reaches 43-44% of its peak and f16 55%.
2. **The dot share of VALU, then busy, then clock.**
   KEFF8 closed the model: percent of nominal dot peak = dynamic dot share x VALUBusy x clock / 1725 MHz.
   Q4_K f16 0.691 x 0.863 x 0.928 = 55.3 (measured 54.9); Q4_K int8 0.552 x 0.849 x 0.933 = 43.7 (43.5); Q6_K f16 46.3 (46.0); Q6_K int8 35.1 (35.2); all keff1, KEFF8.
   Only the first factor is code: the rest of a sub-block's VALU is the float epilogue, conversions and staging.
3. **Dequantize and convert overhead per dot.**
   A Q4_K f16 sub-block issues 256 dot2 beside 32 fmac, 20 cvt, 16 mul and 36 ds_read_b128 (keff1 ISA count); a Q6_K int8 step spends about 110 of 1052 VALU on 64-bit address math (KEFF8).
   The float epilogue is about 4 VALU an accumulator, 28% of Q4_K int8 VALU, and caps Q4_K at about 69% of dot2 peak and 55-58% of dot4 even at full issue (KEFF6); Q4_K sits at about 75% of that reachable limit.
4. **Register tile against occupancy.**
   gfx906 gives a lane 256 VGPRs a SIMD, so 64 keeps four waves, 84 three and 128 two.
   Every tile that grew past 84 lost: a 128-row tile at 115 VGPR and two waves ran Q4_K f16 8-25% slower, and load-ahead builds at 105-109 VGPR ran 7-26% slower (KEFF6, KEFF9).
   The shipped tiles sit at 71-76 VGPR and 19.5-21.5 KB of LDS, three blocks to a CU.
5. **LDS staging and bank conflicts.**
   LDS time is about 15% of VALU time and a minor co-limit (KEFF8); conflicts were 0 on Q4_K and 3.6% on Q6_K f16, but 74.6% of LDS-active cycles on Q6_K int8, the one kernel with no pad.
   Exposed staging is latency, not VALU: removing the staging time (a wrong-math probe) bounds the gain at +15-19% for Q4_K f16, and the exposed staging share is 11-34% a type (KEFF9 pr1).
6. **Attention is a different roof.**
   The attention tile is F32 (docs/PRECISION.md) and read 4.5-5.0 TFLOPS on the MI50 at every depth, 34-38% of the F32 peak, a per-tile rate limit (A:2935).
   A GQA group re-reading its K/V and a 16-key step with two barriers explain it (A:21649).

### Measured prefill facts, with sources

Setups: one MI50, ROCm, `--dtype` as named, p512 unless stated.

| Fact | Model, setup | Number | Source |
|---|---|---|---|
| Prompt time by kernel | Qwen3.6-27B Q4_K_M, ROCm, auto, pp512 | 208 tok/s; q4_k_tile16 72.3%, q6_k_tile16 17.4%, attention_vec256 3.3%, delta 2.6% | keff1 profile 2026-11-19 |
| K tile rate, base | same shapes, 512 rows | Q4_K f16 46-53% of dot2 peak, Q6_K 39-45%; int8 Q4_K 36-43%, Q6_K 27-34% of dot4 | keff1 KEFF2, KEFF6 |
| Staging 3 (ebias, pad 8) | 27B pp512 f16 / int8, A=C ids | f16 251.3-252.6 to 261.3-262.2 (+3.9%); int8 352.6-385.4 to 366.9-393.1; mx 245.9-247.8; contended host | keff1 KEFF7 fm7, READY 372ff2d70 |
| Uniform weight index (c4) | same tiles, bits equal | Q4_K f16 55.0 to 55.7%, int8 43.7 to 44.9, Q6_K f16 46.2 to 47.6; every type +0.5-3.3% | KEFF9 kt8a |
| Inkling prefill profile | Inkling-2-Mini Q8_0, auto f16, ROCm | attention_rel_sink 60% (15.3 ms a call); routed tiles 15% at about 4 TFLOPS (15% of f16); matmul_rows_f32 8%; conv_ring 5% | D:4277, inkgpu7 |
| Inkling attention_rel_sink | same, baseline | 1077 of 1839 ms of pp512 (59%) | I:2246 |
| Inkling rel_sink tile build | Vulkan, one card, pp512 | 395.0 / 397.9 to 552.9 tok/s, tg unchanged | inkgpu10 t1 (wip/inkgpu10 f9730d52) |
| Inkling expert tiles | same | 354 ms of 512 rows at about 33% of f16 peak, which disagrees with 15% above; unverified which peak or count is right | I:2306, D:4277 |
| MoE end to end | Qwen3.6-35B-A3B Q4_K_M, A B x3, host load 49-60 | ROCm pp512 1445 vs mx b12500 895; Vulkan 1296 vs mainline 6b95a9908 863 | moebench1, D:4716 |
| MI50 attention_tile | 30B-A3B Q4_K_M / 27B Q4_1, pp512 int8 | 27.7 / 116.1 / 235.5 ms a call at 16k / 64k / 128k (82-96% of the pass); 47.0 / 176.8 / 341.5 ms (32-78%) | A:2935 |
| QSA prompt attention | Flash-Next, layer split | vec256_g4 is 15.5% of prompt kernel time at about 1.2 TFLOPS, a decode-shaped kernel on a dense causal job | I:101 |
| DeepSeek sparse tile | dsv4, 512 rows, 32 heads | 7.03 to 4.91 ms for the tile32 probe; 5.6 TF/s is 43% of peak | I:658, I:634 |
| Float vs integer tile (Vulkan, 2026-09) | 4096 x 14336, 512 rows | Q4_K 4.65 to 11.48 TFLOPS, Q8_0 4.87 to 7.72; history, the default is now the 16-bit twin | VULKAN, Kernels |
| CUDA square K tile | 27B Q4_K_M, 3090 int8 | pp512 901 to 1024 tok/s; mx 1213 | cuda99, I:605 |
| Q8_0 8-bit routed groups of 1 to 16 entries in one stream (`llmx_q8_0_routed_stream16`) in place of the stream to 8 and the narrow tile | Inkling-2-Mini Q8_0 shapes, one MI50 microbench; pp tw2 on two MI50s (renderD130/131), 12 slices an arm, host load 51-64 | microbench gate-up 335.6/332.4 to 338.8/336.1 us, down 187.9/185.9 to 175.4/175.9, same bits; pp2048 int8 2392.3 to 2398.9 tok/s, auto 1773.7 to 1762.1; pp512 int8 1874.6 to 1870.2, auto 1396.3 to 1412.5 (parity; a kernel and a tile path fewer) | INKSPD3ak rs.txt, inkspd3ak/run.txt |
| CUDA BF16 prompt rows | 0.6B BF16, 3090 | pp512 2988-3454 tok/s against 9379 for Q8_0 and a 10279 target: no tensor-core tile for BF16 | I:654, cuda128 |

Tile builds that lost, with the arm and the cost, as the failed ledger above lists the decode ones:

| Arm | Result | Source |
|---|---|---|
| 128-row 16-bit tile (r128, 4x8 thread) | Q4_K f16 -8 to -25%, 115 VGPR and two waves; parity with the split fix, dropped | KEFF6 kt0, t10 |
| Column swizzle (xor 8 a part) | Q4_K f16 +3.2-4.7%, but Q6_K -1.5-3%, MXFP4 int8 -2-4%, IQ4_NL int8 -8%; the xor does not fold into ds_read offsets | KEFF6 |
| Staging pad in place | f16 +2-5% on most types, IQ4_NL f16 -4, Q6_K int8 -8, Q4_1 int8 -3; +8-12 VGPR | KEFF6 kt3 |
| STEP 2 and next-step loads (c2, c3) | -7 to -26%, 105-109 VGPR, below the occupancy step | KEFF9 kt8a |
| Part rotation (c5) | Q6_K f16 -4% and changes every kernel | KEFF9 |
| Saddr-style weight loads (c6, c7) | the compiler still emits v_add_co and `off` loads; Q6_K int8 static VALU 1274 to 1253 only; a lead, not screened | KEFF9 |
| Plain int16 object read as the shipped one | gfx906 runs int16dot.hipfb; counts from the plain module misled the first ISA reads | keff1 KEFF8 |
| Q8_0 tile at 2 waves a SIMD (waves_per_eu 2, or a full-stage hoist at 100 VGPRs) | the 8-bit tile is issue-bound, and 3 waves hide its LDS reads | 3781 and 2280 us against 3435 and 1952 at 3 waves | INKSPD3ab res_f |
| Whole-block tile left at the compiler's registers in the product build | `llmx_q8_0_routed_tile8` compiled to 88 VGPRs (2 waves) where the microbenchmark's copy had 84; check the trace's VGPR_Count against the microbenchmark's | Inkling-2-Mini pp2048 tw2 int8 trace: gate-up 4092 to 3776 us and down 2351 to 2323 only, then 3415 and 1936 at 84 VGPRs under `LLMX_WAVES_BOUNDS(kThreads, 3)` | INKSPD3ac t, h2 |
| Dense Q8_0 tile left at the compiler's registers | `llmx_q8_0_tile8` compiled to 85 VGPRs (2 waves); `LLMX_WAVES(3)` holds it at 84 | Inkling-2-Mini trace: 12288x32 1271 to 1156 us, 9216x32 633 to 571; pp2048 tw2 int8 2146-2188 to 2202-2208 tok/s, f16 1657-1662 to 1662-1665, same pair, 3 interleaved rounds | INKSPD3ad j |
| 16-bit twin split into high and low byte planes against 8-bit weights | dot4 and dot2 issue at the same rate, so the dots do not drop; two accumulators cost 105 VGPRs, 2 waves | gate-up 6019-6065 to 6647-6663 us, down 3209-3215 to 3688-3691, same bits | INKSPD3ab res_h |
| Quad-shared activations in the 8-bit routed tile (ds_read2st64 and four DPP quad_perm moves in place of a b128 read) | 128 more VALU and 16 s_nop a step, 903 against 775; the tile is issue-bound | gate-up 2054 to 2288 us, down 1167 to 1288, same bits | INKSPD3ak, inkspd3ai/mb rg.txt |
| 128-output routed tiles (`q8_0_tile<., 8, true>`, 112 and 95 VGPRs) | fewer workgroups, more registers | 8-bit gate-up 2121 to 2597 us, down 1196 to 1441; 16-bit 3003 to 3492, 1578 to 1778, same bits | INKSPD3ak, rf.txt |
| Q8_0 8-bit stream at 9-16 entries beside the stream to 8 (`routed_stream<Q8_0, 8, 16>`, as the K-types) | lost to the narrow tile | 171 to 203 us gate-up, 93 to 102 down | INKSPD3ak, rs.txt |
| Next word's LDS reads one word ahead in the tile | 3 waves already hide them | 16-bit -1%, 8-bit within noise, 88 VGPRs | INKSPD3ab res_g |

The Q8_0 8-bit routed tile (`llmx_q8_0_routed_tile8`, Inkling-2-Mini tw2 shapes, one MI50, INKSPD3ak) is bound by VALU issue, with LDS a near co-limit.
A step of its loop issues 775 VALU (512 v_dot4, 64 each of cvt, mul and fmac for the block scales) and 72 ds_read_b128 at 84 VGPRs, 3 waves: 3100 SIMD cycles against about 2300 CU cycles of LDS.
Cutting a third of the activation reads (a wrong-math probe) left gate-up at 2067 against 2054 us, and adding 128 VALU cost 11%; the 16-bit tile is the same shape, 691 VALU of which 512 dot2.
The dots and the bits-equal epilogue fma(wd * xd, float(isum), acc) are 91% of the step, so further gains need fewer staging VALU and fewer LDS reads together, or less wasted work (empty waves of a group's last 64 rows, workgroups that return at once).

The routed tile8 rotates a wave's row quarter by workgroup, runs two of a thread's four rows where a wave holds eight or fewer live rows, and drops the out-of-range selects of whole-block staging (INKSPD3an, INKSPD3ao, INKSPD3ap).
It builds at 78 VGPRs, was 83, and the bits are equal.
Microbenchmark, real24 sets, one MI50, 3 reps (gate-up / down, us): base 2010 / 1126, candidate 1981 / 1117 on the production kernels (-1.4% / -0.7%); the tile16 and mid16 builds are equal in bits and not slower.
Levers on d_tile8 (gate-up / down): ROT +0.5% / +0.5%, ROT+NOSEL -1.0% / -2.7%, TAILJ -1.0% / -1.8%, all three -1.75% / -3.05%, two workgroups a CU +11% / +15%, an extra 64 dot4 +6% / +3%.
Nulls: a 32-bit trip count with the half-row waves (k32t) reaches -3.1% / -3.5% on d_tile8 and is not in this change; blocks aligned to 36 bytes (no v_perm, the bound for split quant and scale planes) add nothing beyond k32t (gate-up 1989 against 1990 us), so the split-planes loader change is not pursued.
Prefetch ceiling: with every block of a row identical, skipping the loads saves 4.0% of gate-up (2017 to 1938 us) and 4.6% of down, bits equal, so memory latency is at most about 4%.
About 704 of 776 VALU a step are the dots and the epilogue, so about 9% is removable; the remaining prompt levers are each under 0.5% of pp2048.

### Open prefill headroom, ranked

Ranked by the largest stated payoff, with the evidence behind each; all but the first two are untried.
1. Inkling prefill from 553 tok/s (Vulkan, one card) toward the estimated 1100 (D:4277):
   tile attention on ROCm (inkgpu2/t8 building, I:2246), routed tiles from about 15% of f16 (int8 dot4 tiles, an opt-in, I:2306), F32 weights on the f16 tile family (I:2280),
   and a conv_ring row grid (339 to 50-80 us estimated, I:2279, I:2312).
2. Attention at depth, where it is 82-96% of a 30B pass at 16k and beyond (A:2935): the tile at 4.5-5 TFLOPS of 13.3 is a third of peak.
   Levers: a query group in one block, deeper keys a step and K/V staged once (A:21649), and a flash tile for QSA at about 1.2 TFLOPS (I:101).
3. K tiles at 55% of dot2 and 44% of dot4: the reachable limit is about 69% and 55-58%, and removing exposed staging bounds +15-19% (KEFF6, KEFF9).
   Leads: per-type tile height (Q6_K int8 at 128 rows +5-7% on big shapes, bits change) and a base-plus-32-bit-offset load (KEFF7, KEFF9).
4. CUDA prompts: a BF16 tensor-core tile (3454 against a 10279 target, I:654), the f16-class IMMA tile by byte planes (8B AWQ f16 pp512 704 to a 220-260 ms target, I:1228),
   a square tile for the 64-output qkv and down products (+8-10% estimated, I:607), and prompt attention on tensor cores (I:1096).
5. Non-matmul work in a prompt pass: gated_delta_rule 53 of 491 ms in a 3090 27B pp512 profile, matmul_rows_f32 at 595 against 60 us, and the per-pass dispatches fusions (I:611, D:4277).
6. Packed f16 FMA (v_pk_fma_f16) has the same 26.5 TFLOPS roof as dot2, so it offers no roof gain over the shipped f16 class; no source names it, and whether the compiler emits it anywhere is unverified.
   The only 2x over dot2 is int8 dot4, which stays an opt-in under docs/PRECISION.md.

## Measurement method (2026-11-23)

### Layout band and controls

In one translation unit, an unrelated edit moves the hot path: unused-function probes moved 0.6B prefill -1.34 / -0.5 / +2.52% and failed the advance rule once (`docs/benchmarks/layout-sensitivity-20260921/`).
The band depends on where the edit is: an end-of-file probe in cpu_backend.hpp moved -1.16 to +0.63%, a four-line edit in Tokenizer::decode -3.13 / -2.17 / -3.09%.
Those cells are in `docs/benchmarks/code-read-20260921/`; perturb the same file as the change, as the Principles of [AGENTS](../AGENTS.md) say.
Build identity alone moved 0.6B prefill -8.03% against -0.08% with both arms built from detached worktrees; build both arms the same way and record `--version`.
On the rig, run ABCCBA with a same-file layout control and treat a 1-4% move as inside the band (D:2177).

### Contention

tools/ab_runner.py flags a pair when either arm's whole-run CPU exceeds the session median by 25%, never drops it, and advances on mean and median paired ratio at least 0.97 with baseline wins under the binomial 5% threshold.
On the rig, host load of 20-70 swamped 0.15-0.3 ms effects (37.5 vs 51.6 tok/s from a 1 s stall, D:665); wrap timed blocks in arc-cap.sh and log load, clocks and card activity in every arm.
Timing runs at default clocks; a 2 ms idle drops the next call 22%, so warm up and run sustained blocks.
The rig timing cores 12-15 are SMT siblings of cores 4-7, so load on cores 0-11 reaches a timed run: 27B Q8_0 int8 tg256 at tw4 fell from 71.58-71.79 tok/s quiet to 68.22 under eight nice-19 compiles, as host issue per token rose from 9.3 to 14.1 ms (sums97, D:18041).

### Microbenchmark harnesses

In the tree:
- llmx-backend-rocm-test rates: decode GB/s and hash by weight type;
- llmx-backend-vulkan-test with --decode-rates and --decode-columns;
- llmx-moe-kernel-bench (tools/moe_kernel_bench.cpp) and tools/decode_probe.cpp;
- tools/sums_vs_reference.py for llmx sums against RCCL, and tools/ab_runner.py for paired end-to-end arms.

On the rig, as records:
- keff rw*.hip arms with mk*.py, go*.sh and a predeclared rule*.txt; keff1 r.sh and vgt.sh (VGPR, LDS, scratch);
- bypass20/23/25 mb1, mb2 and mb4a.hip (read arms); bypass31 lp.cpp (launch floors); cuts27 hl2.sh (slope and intercept fit); cuts22 ds7 (code eviction);
- inkgpu13 mb/go.sh (routed planes with bitdiff); sums29 lk probe and limits2 kernel_sum_ceiling; vksum43 probe/dfloor.cpp;
- moebench1 go.sh rocm or vulkan: interleaved llmx and reference bench plus rocprofv3 kernel stats.
A microbenchmark compiles with the production flags of its target (gfx906: -DLLMX_DOT16): the held-stride arm built without it won 6-12% and lost 6% in replay (cuts4, cuts6).
Keep loads batched in a microbenchmark with one asm "+v" over all loaded registers; an empty memory clobber understated GB/s (D:513).

### Profiling

ROCm: the rocm-perfetto-profile skill, rocprofv3 kernel traces written as Perfetto files.
rocprofv3 adds about 45 us a dispatch (fuse12) and dense Q8_0 read 14 ms traced vs 11.65 untraced (A:17430): time from untraced runs, attribute from traces.
In containers rocprofv3 needs libdw and a writable working directory (AGENT-RULES).
Vulkan: the vulkan-radv-profile skill, GPU timestamps, RADV shader statistics and RGP instruction traces; timestamps inflate small dispatches.
CPU: the cpu-perf-profile skill, process-scoped perf with the exact binary's disassembly.
Device ticks: s_memrealtime is 25 MHz on the MI50; a hop printed as 0.43 us was 1.70-1.75.
Stamps behind s_waitcnt vmcnt(0) charge the load drain to the body (A:27056).

## Open headroom (2026-11-23)

| Kernel | Model, quant | Device | Now | Source |
|---|---|---|---|---|
| q4_k_rows16_1 | Qwen3.6-35B-A3B Q4_K_M | 1 MI50 ROCm | 195 GB/s, 19% of 1024; 3.31 ms a token | moebench1 |
| q4_k_routed16 | same | same | 206 GB/s, 20%; 2.29 ms | moebench1 |
| q6_k_rows16_1 (lm head) | same | same | 302 GB/s, 29%; 2.09 ms | moebench1 |
| q6_k_routed16 | same | same | 168 GB/s, 16%; 0.82 ms | moebench1 |
| q8_0_routed16 | Inkling-2-Mini Q8_0 | MI50 | about 33%, 9.8 ms a token; microbench 625 GB/s at nin 2304 | inkgpu7, inkgpu13 |
| q4_k / q6_k rows, decode | Qwen3.6-27B Q4_K_M | 1 MI50 ROCm | tg 20.6 tok/s, 42 ms of kernels vs a 20 ms floor | keff1 |
| q4_k_tile16 prefill | same | same | pp512 208 tok/s at keff1; the tile read 46-55% of dot2 peak in later counters (Prefill: the compute side, above) | keff1 |
| q8_0_half8_1h5 gate+up | 27B Q8_0 int8 tw4 | 4 MI50 ROCm | 764 GB/s (best call 829); 1h5p 635 | cuts38, cuts39 |
| All products | same | same | 755 GB/s, 9.01 ms vs 7.66 ms as one stream | cuts14 |
| 9-column Q8_0 decode | 5120x5120 | MI50 ROCm | 0.77 of main through the 11-column build | keff50 |
| BF16 prompt rows | 0.6B BF16 | RTX 3090 | pp512 3454 tok/s against a 10279 target | cuda128, cuda129 |

End to end on moebench1 llmx leads the reference (ROCm pp512 1445 vs 895, tg128 72.0 vs 40.4; Vulkan 1296 vs 863, 86.1 vs 56.9), so this headroom is against the device, not a competitor.

Hypotheses, ranked by stated payoff:
1. K-quant rows and routed kernels are VALU and occupancy bound (keff1, inkgpu8): one wave a SIMD at 206-249 VGPRs leaves far less than the 15-30 KB a CU in flight; byte planes, LDS-staged columns and twin half-sums, under 128 VGPRs.
2. Dispatch count on Inkling: 2261-2454 dispatches a pass at about 4 us device floor; fusing chains toward 500-1400 is estimated at 4-7 ms a pass (I: aa4b2022).
3. Host recording at tensor width 2: 32.6 vs 24.9 ms a pass from recording members in turn; per-member threads estimated at -8 ms a pass (inkgpu12).
4. Held Q8_0 gap is a per-launch intercept, about 0.84 ms a token: kernel code 0.65, x reload 0.42 (cuts27, bypass25); merge launches or cap the grid near 120 blocks with x loaded once.
5. Code fetch after a weight stream for other long kernels (rms_norm twins, attention_vec_major, group_rms_norm): 0.3-0.6 ms a token by the delta_step precedent (cuts22).
6. Planes ported to prepare_matrix and routed MoE: nin 2304 73 to 60-65 us estimated (keff36).
7. Vulkan range-based hazard tracking for a barrier a dispatch: about 1 ms a token estimated.
8. Pre-enqueue token N+1 with a device argmax: 0.5-1.5 ms a token (bypass27).
9. moe_step_down on the 27B drafter waits on load latency at its occupancy: about 7.6 KB of LDS caps a CU at 8 workgroups and Q5_K at 80 VGPR at 6; Q5_K at 64 VGPR or less is predicted at 47 to 35 ns a workgroup (dspine30 m9, untested).

## Vision encoder encode time, Inkling-2-Mini mmproj on one MI50 (2026-11-27, INKVIS22, land/inkling6 8dcf6260)

Measured with `llmx-encode-image` `--repeat 4` on the same MI50 under the timing lock, each cell the median of four repeats after the first encode.

| Image (tokens) | ROCm gfx906 | Vulkan RADV | Source |
|---|---|---|---|
| 300x200 (40) | 62.7 ms | 51.4 ms | inkvis22/out/r-rocm, r-vk |
| 800x600 (300) | 978 ms | 196 ms | same |
| 1920x1080 (1296) | 12301 ms | 978 ms | same |

On ROCm, rocprofv3 (kernel trace, 827 dispatches, 12.28 s of kernels) puts 10.80 s in `llmx_attention_chunk` (23 calls, 470 ms each, about 0.23 TFLOPS against about 110 GFLOP a layer) and 1.38 s in `llmx_matmul_rows_bf16`; everything else is under 35 ms.
The non-causal call has 64-wide heads, so it misses the tile route (causal, head_dim 128 or 256) and the vector route and takes the one-row-per-block chunk kernel, which reads K and V from global memory once per query row with a quarter of its threads live in the value sum (src/backends/gpu/backend_core.hpp attend, kernels/f32.gpu attention_chunk; Fable read-through, not built).
The candidate is `attention_tile` for head_dim 64 with a non-causal flag and a 256-key fold, which changes ROCm and CUDA vision sums within the 4e-6 rule of docs/PRECISION.md and needs its own failing test (`vision::dominant_key` at 1296 and 4096 keys) and the CUDA vision check; it is not done.
Vulkan 1080p is within its bound at 8.53e-6 of 9.37e-6, ROCm 4.38e-6, CPU 4.03e-6.
