# ROCm Q8 compute probe, 2026-10-10

The first HIP Q8 matrix-vector probe is slower than the existing Vulkan path.
A focused load change reduces HIP graph latency by 28-41 percent, but remains
9-31 percent slower than Vulkan in the measured cells. The communication gain
therefore does not yet establish an overall backend win. The compute gap and
full prompt attention remain open before the admission decision in [ROCM](../../ROCM.md).
No production backend or build option is added by this record.

## Workload and bounds

One MI50, PCI 0000:89:00.0 under pci0000:80, CPUs 4/5/6/7 with their siblings
reserved, default clocks and power. Every arm uses image
`d5591273e00b5735bd3419009ffd7575821dcf957897010e03ba1d1c6a0ae14f`,
HIP 7.2.1, AMD clang 22, Mesa 25.2.8 and the retained Vulkan library built from
clean main `fb82404338a6c93f5cf355279ce968583aaf57cc`. Its runtime is unchanged
through the starting main `c6d828aaa9aea5fc804f7b9a1fb7f9c592b28842`.
Vendor prerequisites HSA_FORCE_FINE_GRAIN_PCIE=1 and GPU_MAX_HW_QUEUES=8 are
identical between arms; no ISA override, hidden llmx knob or host modification.

Canonical 34-byte Q8_0 blocks, square matrices of width 4096 and 5120, and
eight distinct weight/input banks, at least 136 MiB of weights, beyond cache.
Every operation includes activation preparation and matvec. Rotating input
descriptors forces Vulkan's preparation to run; HIP prepares the used form,
while Vulkan int8 also prepares the int16 twin and sum table. This difference
is included in the end-to-end measurement, not hidden as equivalent work.

F16 denotes the approved block-int16 policy; int8 is block-int8, both with F32
scaling and accumulation. Inputs have a block peak of one and odd multiples
of 1/256 elsewhere, away from rounding ties; signed finite normal F16 scales
and the full signed-byte weight range are covered. An independent host oracle
constructs activation codes, decodes weights and sums in double. Every output
must be finite and within `1e-6 + 1e-5 * sum(abs(products))`, frozen before runs.
Guards, original inputs, all HIP codes/scales and Vulkan executed-path witnesses
are checked. The 96-by-19 smoke covers odd block counts and row tails.
This establishes ordinary finite cases, not extreme-range, model or HF gates.

Each process runs three warmup chains and 20 measured chains of 1024 operations,
with eight banks rotating equally. Upload, graph capture, readback and checks
are outside timing; preparation, products, enqueue and completion are inside.
The graph has exactly 2048 nodes. Two blocks reverse arm order for each shape
and dtype. Figures below are microseconds per operation, block 1 / block 2
medians, lower better. All individual samples, minima, p95 and maxima remain
in the raw summary. Repeated outputs and graph replay are not a full model's
dependencies. No mx kernel was measured for this fixture; its separate
[model targets](../rocm-model-budget-20261010/README.md) remain open.

## First compute matrix

All 24 processes pass: 480 measured chains and 72 warmups. HIP eager and graph
both lose to Vulkan. The graph's host enqueue cost is small beside completed
device time; changing launch mode alone does not remove the compute gap.

| Square width | Dtype | vulkan | hip-eager | hip-graph | mx kernel target |
| --- | --- | --- | --- | --- | --- |
| 4096 | f16 | 30.46 / 30.48 | 64.38 / 64.47 | 62.61 / 62.63 | Not measured |
| 4096 | int8 | 29.75 / 29.65 | 61.25 / 61.14 | 59.44 / 59.41 | Not measured |
| 5120 | f16 | 43.80 / 43.79 | 80.93 / 80.98 | 79.65 / 79.74 | Not measured |
| 5120 | int8 | 42.89 / 42.87 | 79.61 / 79.65 | 77.66 / 77.66 | Not measured |

## Profile and load change

ROCTX markers bracket the measured region. The profiler is validated first on
the tail fixture; the full captures each contain 4096 preparation and 4096
product dispatches, with 1024 of each inside the measured region and no copies
there. Traced product means are 68.81 us for F16 and 66.21 us for int8, versus
4.90/4.95 us for preparation. Scratch is zero. The profiler doubles wall time:
F16 126.59 us/op versus unprofiled controls 62.23/62.24, int8 127.88 versus
59.11/59.11. Those instrumented timings locate work; they are not speed results.
The marked build's device assembly matches the original after normalizing only
the source filename and HIP translation-unit identity. The private profiling
image adds missing libdw to the same base, and both controls use that image.

The original helper computes wide byte addresses and branches per lane when
Q8 bytes straddle words. The candidate computes each full-width row base once,
uses row-relative offsets and loads five consecutive words before combining
the four words needed. It retains canonical storage, source arithmetic and
reduction order. It follows the existing Vulkan strategy, written in the
standalone HIP probe; no foreign implementation is copied into production.
The change tests addressing and load scheduling together, not their separate
contributions. The speed result below supports that combined change.

gfx906 emits native dot2/dot4. Compiler-reported product VGPRs are 39 for both
original paths, 46 for candidate F16 and 39 for candidate int8; neither spills
or uses private scratch. The register increase is a remaining tuning question,
not proof of the cause of the residual gap. gfx1151 variants compile with zero
reported spills; no Strix Halo hardware test or support claim is made.

## Matched candidate matrix

All 40 processes pass: 800 measured chains and 120 warmups, with original HIP
and Vulkan controls rebuilt by neither tuning nor profiling. The candidate
had first passed 12 full-chain checks, including tails and both launch modes.
Its name below is `loads`; original HIP remains `hip`.

| Square width | Dtype | vulkan | hip-eager | hip-graph | loads-eager | loads-graph | mx kernel target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | f16 | 30.45 / 30.37 | 64.40 / 64.48 | 62.65 / 62.68 | 38.38 / 38.38 | 36.71 / 36.73 | Not measured |
| 4096 | int8 | 29.72 / 29.63 | 61.14 / 61.16 | 59.36 / 59.37 | 36.72 / 36.68 | 34.77 / 34.79 | Not measured |
| 5120 | f16 | 43.81 / 43.86 | 81.06 / 81.13 | 79.86 / 79.97 | 58.91 / 58.94 | 57.49 / 57.55 | Not measured |
| 5120 | int8 | 42.85 / 42.81 | 79.64 / 79.61 | 77.62 / 77.65 | 48.36 / 48.36 | 46.63 / 46.64 | Not measured |

| Width | Dtype | Candidate latency reduction vs original HIP graph | Candidate extra latency vs Vulkan | Candidate useful weight GB/s |
| --- | --- | --- | --- | --- |
| 4096 | f16 | 41.41% / 41.41% | 20.55% / 20.94% | 485.64 / 485.38 |
| 4096 | int8 | 41.42% / 41.40% | 17.00% / 17.40% | 512.63 / 512.39 |
| 5120 | f16 | 28.00% / 28.04% | 31.24% / 31.20% | 484.45 / 484.01 |
| 5120 | int8 | 39.93% / 39.94% | 8.80% / 8.93% | 597.37 / 597.19 |

Bandwidth counts useful Q8 bytes divided by completed time, not measured
physical traffic. The material reduction repeats in both blocks; no small
win or native-model speed parity is claimed. The candidate's F16 5120 case
retains the largest cost versus Vulkan and needs further measurement.

## Activity, failures and evidence

Both matrices use the same one-second CPU/process/disk/GPU monitor. Frozen
flags: unrelated CPU at least one observed core, an individual disk at least
100 MiB/s, or an outside GPU at least 20 percent. Zero of 64 calls and zero
of 1280 measured chains cross these flags. Maximum observed unrelated CPU
in a measured interval is 0.58 cores; all intervals and samples remain.
No missing GPU counters or leading/trailing coverage gaps; longest monitor
interval is 1.012 seconds. Short-lived/new/inaccessible processes may be missed,
and disk counters are system-wide; this does not assert complete idle time.

Every timing output passes its bound, 54,263,808 checked values across the two
matrices including warmups. The largest observed fraction of the frozen bound
is 0.001210. Repeated values count repeated checks, not unique test cases.
Initial smoke passes 18 processes; full-chain original checks pass six and
candidate checks pass twelve. The marked profile series passes seven processes.
Build 1's ignored cleanup return values, the first monitor-summary end-record
parsing error and the first profile check's inclusion of runtime copy kernels
are retained. These were build/harness failures, not discarded slow samples.

Each native process has a 150-second timeout plus 10-second grace; supervised
groups are bounded and all containers drain and are removed. Exact starting
VRAM, 10,932,224 bytes, is restored. No own competing build, test or download
runs during timing. No production process, driver, package or clock is changed.
All frozen source/binary/assembly hashes pass before and after the work.

Evidence lives at `/zpool1/llmx-xdev-validation/rocm-probe-20261009`, mirrored
under `.tmp-rocm-probe-20261009` on the workstation: both frozen protocols and
manifests, build sources/logs/assembly/hashes, stdout/stderr and call records,
activity and cleanup, profiler CSVs and both performance JSON summaries.
The checkpoint archive is `q8-evidence-20261010.tar.gz`; binaries remain on the
rig. Standalone experimental source is private and is not runtime support.

The follow-up below measures preparation and row layout; the initial results
above remain historical evidence. Full backend gates stay open.

## Phase and row-layout follow-up (2026-10-10)

The follow-up keeps the earlier matrices and binaries frozen. It measures
preparation separately, tests the F16 register limit, then changes wave row
ownership. All use the same ordinary finite fixtures, image, compiler flags,
device and monitor described above. Starting main is `5948174e9`; no runtime
source changes. The reference Vulkan library is still the same retained build.

### Phase isolation and a rejected occupancy hint

The `parts` diagnostic runs full, preparation-only and product-only graph
chains. Product-only prepares all eight inputs before timing; preparation-only
runs products after timing to verify every output. Both retain exact code,
scale, input and guard checks. Isolated timings are diagnostic and are not
assumed to add to interleaved model work. Three warmups and five measured
1024-operation chains per process, two arm orders, 56 processes in total.
The retained `loads-graph` is a separate binary control for the added modes.

An experimental gfx906-only compiler hint requests six waves per execution
unit. F16 goes from 46 to 40 vector registers but spills six registers into
28 bytes of private scratch. Int8 remains at 39 registers with no spills.
The forced F16 path is substantially slower, so the hint is rejected.
These are compiler resource limits, not measured achieved occupancy.

Microseconds per operation, both block medians, lower better:

| Width | Dtype | vulkan | loads-graph | base-full | base-prepare | base-product | waves6-full | waves6-product | mx kernel target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | f16 | 30.36 / 30.39 | 36.49 / 36.60 | 36.54 / 36.58 | 1.81 / 1.81 | 35.11 / 35.15 | 62.92 / 62.95 | 61.26 / 61.20 | Not measured |
| 4096 | int8 | 29.73 / 29.72 | 34.76 / 34.78 | 34.78 / 34.78 | 1.80 / 1.80 | 32.80 / 32.80 | 34.77 / 34.79 | 32.80 / 32.81 | Not measured |
| 5120 | f16 | 44.04 / 43.84 | 57.29 / 57.28 | 57.31 / 57.28 | 1.79 / 1.79 | 55.83 / 55.81 | 86.90 / 86.92 | 85.09 / 85.16 | Not measured |
| 5120 | int8 | 42.83 / 42.79 | 46.66 / 46.50 | 46.61 / 46.69 | 1.79 / 1.79 | 44.88 / 44.99 | 46.57 / 46.49 | 45.09 / 45.08 | Not measured |

Preparation is about 1.8 us; the F16 product at width 5120 is about 55.8 us.
The product therefore remains the main target. All 56 processes pass their
bounded checks: 280 measured chains, 168 warmups and 16,515,072 output checks.
One call overlaps the predeclared CPU flag: width 5120, int8, forced-six-wave
full, first block, maximum observed unrelated CPU 3.15 cores. Four measured
chains and all three warmups in that call are flagged and retained. No missing
GPU counters or coverage gaps; longest monitor interval is 1.017 seconds.

### Wave-uniform row ownership

Each wave owns the same two rows across its lanes. The next candidate makes
their row-base index explicitly uniform using `readfirstlane`; the raw loads,
source arithmetic and reduction order stay. A second control uses one row
per wave, reducing activation reuse while retaining the per-row computation.

| gfx906 product | Retained loads VGPRs | Uniform two-row VGPRs | Uniform one-row VGPRs |
| --- | --- | --- | --- |
| F16 | 46 | 51 | 25 |
| int8 | 39 | 35 | 22 |

Neither new layout spills or uses private scratch. Both compile for gfx1151
with no spills too, but gfx1151 remains compile-only. The full-chain smoke
passes 24 processes, 96 chains and 2,364,160 output checks across both launch
modes, dtypes, square widths and 96-by-19 tails. The timing matrix uses graph
replay, three warmups and 20 measured chains per process, two arm orders:

| Width | Dtype | vulkan | loads-graph | uniform-graph | onerow-graph | mx kernel target |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | f16 | 30.36 / 30.45 | 36.66 / 36.70 | 34.39 / 34.43 | 69.05 / 69.11 | Not measured |
| 4096 | int8 | 29.75 / 29.68 | 34.77 / 34.78 | 39.70 / 39.69 | 65.36 / 65.37 | Not measured |
| 5120 | f16 | 43.81 / 43.86 | 57.40 / 57.43 | 47.23 / 47.34 | 68.75 / 68.69 | Not measured |
| 5120 | int8 | 42.83 / 42.82 | 46.60 / 46.60 | 48.91 / 48.90 | 50.25 / 50.25 | Not measured |

All 32 processes pass: 640 measured chains, 96 warmups and 27,131,904 output
checks. No call or measured chain crosses a declared activity flag; maximum
observed unrelated CPU is 0.51 cores. No missing counters or coverage gaps;
longest monitor interval is 1.011 seconds. Maximum error is 0.001210 of the
frozen bound. Both series restore exactly the initial VRAM and remove their
owned run containers; all before/after hashes match.

The two-row change lowers measured F16 latency by 6.2 percent at width 4096
and 17.6-17.7 percent at 5120. It increases int8 latency by 14.1-14.2 and
4.9-5.0 percent respectively. The one-row control loses in every cell despite
its much smaller register count. These results reject using register count
alone to choose a kernel. Keep the retained int8 and uniform two-row F16
candidates separate in the next screen. No same-behavior layout perturbation
control was added here, so the smaller differences are not attributed to one
instruction alone. F16 still costs about 13 percent and 8 percent over Vulkan
at the two widths; no mx microkernel or model speed gate is established.

The frozen `q8-rows-plan.json` hash is
`a8693273e1f57d914b2147dd075821118c361a4b258ee10c05abe3d78f7338e9`.
The raw phase and row-layout outputs, activity, call order, source copies,
compiler assembly and hashes are retained in the same evidence directory.
Their summaries are `q8-parts-performance-1.json` and
`q8-rows-performance-1.json`. The separate follow-up archive is
`q8-followup-evidence-20261010.tar.gz`, 418 files, SHA256
`9f8885e9b466018f53736d64c52cc6a6a4b97a8ef743f78d4947dba055cbd55b`.
The original archive remains unchanged and predates this follow-up; executable
binaries remain on the rig. All negative controls and the flagged call stay.

Next is the planned prompt-attention probe, isolating QK and PV then timing
the complete operation. The Q8 gap remains part of the combined compute and
communication budget before backend admission; the full model/HF, lifetime
and mx gates remain open. Core dtype is already merged. CPU emulation retains
correctness coverage but its speed is nonblocking; GPU and native CPU arithmetic
remain the performance priorities. This record adds no production backend.
