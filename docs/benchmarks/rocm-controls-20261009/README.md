# ROCm collective controls, 2026-10-09

This is a direct collective comparison, not model throughput or a passed
phase-final mx gate. All 168 planned processes pass the exact final residual
and guard checks: 3,360 measured chains and 504 warmups. The preceding 24
smoke processes pass another 552 chains. Every planned run and slow sample
is retained. No production backend code changes.

## Workload and identities

Each chain has 128 sums into a residual, with zero or four dependent increments
after every sum. Three warmups precede 20 measured chains per process. Each
table cell gives the two process medians, forward and reverse arm order, in
microseconds per sum or sum-plus-increments unit; lower is better. The raw
record also keeps every sample, p95, enqueue time, device interval and rank skew.
No preparation, capture or validation is timed.

The common image is `d5591273e00b5735bd3419009ffd7575821dcf957897010e03ba1d1c6a0ae14f`:
HIP 7.2.1, clang 22, Mesa 25.2.8 and shaderc 2025.2. All arms now set the
vendor prerequisite `HSA_FORCE_FINE_GRAIN_PCIE=1` and `GPU_MAX_HW_QUEUES=8`.
No ISA override is used. Older measurements with the fine-grain setting unset
stay separate; their difference does not isolate the environment's effect.

The llmx control is `f80709f204a8eb13749f78b7dcb31da5ae343fde`, whose runtime
tree is unchanged in the documentation landing `56e69e686`. The retained
Vulkan and custom HIP binaries are used again, not rebuilt for this comparison.
mx is its shipped library from image
`a83c92cf625046e09959de1a7b1e7b241e723646b5e9c6d6f47e6ce28fdc83e3`, source
`eefc4e7321c869496146697d63362f073941aed6`. Its exported prepare/launch calls
are linked through the matching unmodified header in the reference scratch
directory. No foreign source is copied into llmx's runtime. The path record
confirms broadcast, three blocks at 5,120 values and 16 at 65,537, 512 threads;
these shapes do not select two-shot. RCCL reports version 22707 (2.27.7).

Custom HIP fuses the ordered F32 sum into the residual. mx produces a sum and
our harness adds it into the residual with a separate kernel; both implement
the measured contract, with their respective costs included. The graph node
counts are held to 128 * (1 + adds) for custom HIP and 128 * (2 + adds) for mx.
Every member finishes submitting before any host wait. mx's one-time prepare
is outside the chain, so this does not measure its complete model scheduler.

HIP copies use peer transfers and events, including consumption before scratch
reuse, then an ordered local F32 sum. RCCL uses grouped send/recv and the same
ordered local sum. Native RCCL allreduce is not substituted for that ordering;
its earlier failed order-conformance cells remain in [STATUS](../../STATUS.md).
These conservative controls use one enqueue coordinator and are eager only.
Captured conservative transport remains unqualified. The four increments are
launch placeholders: scalar HIP adds versus Vulkan's ones buffer, not identical
model compute. New controls also verify input preservation and sum/staging
guards; the custom HIP probe retains its sequence/status checks. Only final
chain results are saved, not every intermediate output.

Cards are rig GPU 2/3/4/5 under PCI root 0000:80. Width two uses cards 4/5,
three 4/5/7 and four 4/5/6/7. CPU allowance is 4/5/6/7 with siblings 12/13/14/15
reserved. Default clocks and the one-second monitor apply to every arm.
All owned containers were removed and every selected card returned to its
exact initial 10,932,224 bytes of VRAM use.

## 5,120 F32 values

| Members | Adds | Vulkan | HIP eager | HIP graph | mx eager | mx graph | HIP copies | RCCL copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 | 94.81 / 95.27 | 11.90 / 11.87 | 10.60 / 10.49 | 14.47 / 14.27 | 12.48 / 12.45 | 63.05 / 63.94 | 36.97 / 36.93 |
| 2 | 4 | 101.11 / 101.34 | 20.04 / 20.06 | 15.97 / 16.00 | 22.78 / 23.58 | 17.86 / 17.85 | 83.19 / 83.86 | 57.45 / 53.92 |
| 3 | 0 | 144.11 / 152.02 | 19.74 / 19.00 | 16.47 / 16.61 | 23.79 / 19.11 | 16.60 / 16.65 | 111.40 / 92.44 | 49.76 / 49.57 |
| 3 | 4 | 220.67 / 226.84 | 26.52 / 40.49 | 22.05 / 22.58 | 46.15 / 37.89 | 22.64 / 22.74 | 136.20 / 156.30 | 96.32 / 101.86 |
| 4 | 0 | 362.60 / 310.91 | 30.31 / 31.87 | 24.50 / 23.77 | 34.52 / 34.93 | 23.89 / 24.08 | 206.19 / 234.02 | 93.00 / 70.25 |
| 4 | 4 | 356.80 / 361.87 | 40.04 / 51.08 | 29.07 / 29.32 | 59.19 / 56.90 | 29.75 / 28.91 | 287.42 / 254.94 | 111.60 / 163.29 |

## 65,537 F32 values

| Members | Adds | Vulkan | HIP eager | HIP graph | mx eager | mx graph | HIP copies | RCCL copies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0 | 95.29 / 91.70 | 54.18 / 54.09 | 52.73 / 52.65 | 44.68 / 45.03 | 44.16 / 44.19 | 60.15 / 64.20 | 96.33 / 97.12 |
| 2 | 4 | 99.20 / 93.45 | 63.34 / 63.56 | 60.13 / 59.91 | 55.24 / 57.00 | 51.55 / 51.29 | 70.63 / 75.02 | 106.61 / 106.24 |
| 3 | 0 | 223.08 / 144.13 | 112.91 / 112.64 | 107.20 / 107.33 | 108.51 / 106.86 | 95.27 / 95.64 | 137.29 / 134.95 | 184.14 / 183.66 |
| 3 | 4 | 156.66 / 212.33 | 117.88 / 121.58 | 114.23 / 114.45 | 110.46 / 126.56 | 102.48 / 102.21 | 159.72 / 151.50 | 192.43 / 192.51 |
| 4 | 0 | 342.25 / 416.80 | 159.45 / 159.75 | 154.59 / 154.69 | 176.81 / 184.45 | 159.96 / 159.49 | 311.09 / 224.36 | 270.60 / 271.02 |
| 4 | 4 | 305.63 / 409.99 | 186.63 / 175.73 | 162.92 / 162.89 | 195.60 / 190.27 | 168.91 / 168.93 | 275.10 / 240.29 | 278.95 / 282.52 |

## Activity and interpretation

The predeclared unrelated-CPU flag overlaps 96 of 168 calls: 13 at width two,
27 at width three and all 56 at width four. Observed work outside the benchmark
cgroups includes package/build processes and clang/cc1plus; aggregate observed
unrelated CPU peaks at 15.02 cores. Ownership of those jobs is not established
by this monitor. No call triggers the disk or outside-group GPU threshold.
No GPU counter is missing and no process has a leading/trailing coverage gap.
Short-lived and inaccessible processes remain limitations; the monitor covers
whole calls including initialization and warmup, not individual millisecond
chains. All flags and samples remain in the result, with no discarded or
replacement runs. This imbalance limits small performance conclusions.

At 5,120 values, graph replay puts the four-card custom probe and mx near
24 us per sum, and near 29 us with four increments. This is evidence of a
competitive communication mechanism in this workload, not proof of a small
win or a full-model speed gate. The two-card small-vector custom graph rows
are faster in both blocks; three-card rows are close. Eager launch varies and
the conservative transfer paths remain costlier in the complete matrix.

The larger-vector losses remain visible: custom graph takes about 52.7 us
against mx's 44.2 at width two, and 107.3 against 95.5 at width three, without
increments. Its width-four graph rows are lower than mx's in these blocks,
but every width-four call overlapped flagged CPU activity. A vector-size sweep
and instruction/traffic evidence, not these two blocks alone, would qualify
a new dispatch choice. The current comparison does not include mx's two-shot
crossover or large-message RCCL policy.

The next admission evidence is the refreshed same-environment model budget,
with the wrong-output no-sum arm explicitly diagnostic. Q8 decode, full prompt
attention, model correctness and the other backend milestones in
[ROCM](../../ROCM.md) remain required. The existing closed kernel/driver-patch
route and the Vulkan tensor-split prerequisite are unchanged.

## Evidence

Canonical directory: `/zpool1/llmx-xdev-validation/rocm-probe-20261009`.
The workstation mirror is `.tmp-rocm-probe-20261009`.

- `controls-protocol.md`, SHA256 `2e4f858c24222a9fefc0dc911ae337641e12b01bb4f45984ac327785fc298368`, and `controls-plan.json` freeze every cell, arm order, bound and activity flag.
- `latency-controls.cpp`, `build-controls-2/sha256.txt` and the retained libraries/headers identify the new harness and the shipped reference. Both wrapper targets compile; the reference library's gfx906 code is not gfx1151 qualification.
- `controls-smoke-timing-1`, `controls-smoke-activity-1-w*` and `controls-smoke-summary.json` retain all smoke cases.
- `controls-timing-timing-1`, `controls-timing-activity-1-w*` and `controls-summary.json` retain every planned timed process, sample, activity record, path witness and cleanup. The summary was independently verified on the rig and workstation.
- `reference-inspection-2` retains the failed unsupported `llama-bench --version` call and overly broad symbol listing. The narrower `reference-inspection-3` proves the required exports, image source label, headers and library identities. `build-controls-1` retains the compiler-reported HIP function-name typo; build 2 corrects it before any run.

No result from a failed setup attempt is counted as a completed run. No injected
missing-rank experiment is run against mx's unbounded device barriers.
