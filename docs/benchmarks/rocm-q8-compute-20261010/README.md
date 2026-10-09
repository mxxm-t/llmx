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

Next: measure the remaining F16 register/load-scheduling cost and preparation
cost before choosing another kernel change, then complete the prompt-attention
probe. The existing full backend, file-exact HF, lifetime, model and mx gates
remain required. This documentation-only landing runs docs/dead-code; it does
not merge the candidate or declare ROCm admission.
