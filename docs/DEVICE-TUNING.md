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
| Code prefetch into L2 | a 64-byte line a thread up to the end label; the weight stream had evicted the code | 35.84 to 28.64 us bench; 66.02 to 67.20 tok/s | llmx_delta_step, MI50 ROCm | 57ee77537, cuts22, cuts25 |
| Four-way dot for Q8_0 rows | dot4_i8 for a scalar byte loop | 60 v_bfe_i32 and 43 multiplies to 2 and 0; 0.227 ms a token | llmx_delta_step | 2362d1fe3 |
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
| exact_cols reduce-scatter | one weight load feeds G columns | c16 step 4869 vs 10503 us | CUDA | cuda123 |
| Route build per decoder pair | a runtime F16/BF16 branch held 148 VGPR in the F32 build; one build a router and row-gate decoder pair keeps the F32 one at 65-68 | route r1/r4/r8 17/29/52 to 13/19.5/33 us (trace p50) | llmx_moe_step_route, MI50 ROCm | dspine28 p1, dspine29 kcf-16 |
| Entry-fastest step grids | rows of h the grid's fast dimension, so an expert several rows pick is read from DRAM once | down_q8_0 r8 143 to 130 us; Q8_0 gate step r8 414 to 354 us | llmx_moe_step_up/down, Qwen3.6-35B-A3B shapes | dspine28 m8 |
| Ids, weights and y0 staged with the twin | the down's combine waited on 2k serial global loads and each task on one; the limit was the vmcnt chain of dependent L2 reads, removed by the staging | step medians r1/r4/r8, main to staged: Q5_1 down 62.8/170.2/274.9 to 62.2/147.9/258.1 us, Q8_0 69.3/169.6/311.2 to 65.4/152.7/250.3 us | llmx_moe_step_down, Qwen3.6-35B-A3B shapes | dspine29 m10 |
| lane_row: twin before the weights' decode, shift trip count, fold by selects | the twin's LDS reads and sum run under the weight load's wait; the runtime stride's division and branchy fold were the instruction excess against main | down_q5_1 r1 25.1 (w12) to 20.8 us (trace p50); step Q5_1 r1/r4/r8 62.8/151.4/274.4 (main) to 60.1/143.2/241.4 us | lane_row, MI50 ROCm | dspine29 m11 |
| exact_cols wave32 geometry, q/k/v and gate/up one launch | R4G2 over 8 waves past 8 rows, R2G2 over 4 below; fewer launches | 0.6B 28-layer round at 16 rows 4.79 to 3.10 ms, 8 rows 2.86 to 2.21, 0 bits differ (harness) | CUDA 3090 | cuda143, cuda145 |
| exact_cols many-row build without spills | last whole step widens at load, the others hoist raw bits | 112 B stack and 160 B spills to 0 at 128 registers, sm_86 ptxas | CUDA | cuda145 |
| exact_cols F32 column builds without spills | no raw hoist on wave32; many-row build one block an SM | few 8 B spill to 0 at 96 registers, many 32 B to 0 at 166; at 128 registers every F32 form spills 68-320 B, sm_86 ptxas | CUDA | cuda148 |
| exact_cols 16-bit many-row build without spills on sm_70 and sm_120 | one block an SM there, two elsewhere (`LLMX_COLS16_BLOCKS`) | 16 B (sm_70) and 20 B (sm_120) spills at 128 registers to 0; no card of those targets timed | CUDA | cuda149 |

## Patterns that failed, to 2026-11-23

Each line is the arm, the reason as read at the time, and its numbers; "implementation" marks a loss later reversed by another build of the idea.

| What | Why | Numbers | Source |
|---|---|---|---|
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
| Sixteen lanes a block (Vulkan) | in-order load counter | 245-272 GB/s, fewer registers | VULKAN, Kernels |
| One-column build of wide Q8_0 (Vulkan) | | -9% of 8B decode | VULKAN, Kernel notes |
| Attention four heads a block (ROCm) | | 24.6 vs 17.7 us | STATUS, ROCm decode launches and host time |
| K rows: waves_per_eu 4/5, lane-fixed index, K45 pair | spills, VALU | 10-37% slower | keff1 |
| CUDA BF16 prompts, cols8, KT_FULL | | pp512 2988 vs 9379; cols8 912/938 vs 1245/1516 tok/s; no gain | cuda102, cuda121, cuda87kt |
| Fused step: no next load on the last trip (lane_row) | the guard made the Q4_K up wait on each unit (vmcnt) | up r8 126.7 to 138.6 us; a one-trip path before the loop took its place | dspine28 m8 |
| Fused step down through lane_row at a runtime stride | division, clamps and a branchy fold: about 120 more instructions a wave at 4 waves a SIMD | down_q5_1 r1 22.4 to 25.1 us (trace p50) against the hand-written main kernel | dspine28 m9, Fable |

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
