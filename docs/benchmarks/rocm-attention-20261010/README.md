# ROCm prompt attention, 2026-10-10

The first full HIP attention probe is correct on its ordinary-finite fixtures
but takes 9-49 percent longer than Vulkan. Isolated QK/PV lane layouts give small,
mixed changes. Full-kernel phase clocks then identify KV staging and its barrier
as the largest recorded phase, selecting a wider-load experiment. These are
private standalone probes, not a production ROCm backend or a model gate.
The [ROCm admission rule](../../ROCM.md) remains unchanged.

## Scope

One MI50 at PCI 0000:89:00.0, default clocks and power, CPU 4/5/6/7 and their
siblings reserved. The full-op arms use pinned image
`d5591273e00b5735bd3419009ffd7575821dcf957897010e03ba1d1c6a0ae14f`,
HIP 7.2.1, clang 22, Mesa 25.2.8, and the retained Vulkan library from clean
`fb82404338a6c93f5cf355279ce968583aaf57cc`. Runtime code is unchanged through
starting main `ad8f44149e71e4a93256c2f78ff4515fa8ea750b`. Panel timing and traces
use its derivative image `8bf6f488e3d79cc1c672294e36627c484209a0fd41da4ac9b0c00b8b409ac38a`,
which adds profiler dependencies; all their controls use that same image.
Vendor prerequisites HSA_FORCE_FINE_GRAIN_PCIE=1 and GPU_MAX_HW_QUEUES=8 are
explicit and identical between arms. No host, power, clock or driver change.

F32 queries, scores, probabilities and sums, with exactly decoded F16 K/V.
Causal GQA uses 64-token pages in reversed physical order, dimension 128 and
two deterministic input banks. Queries include values not representable in F16.
Attention has no dtype argument: the matrix policy does not authorize narrowing
these operations. The pinned mx generic half tile uses additional rounding;
that source finding is not a dispatch or speed witness for this fixture.
No foreign kernel was copied. Core dtype is already merged; CPU emulation speed
does not block it, while correctness and GPU/native CPU performance remain required.

The double-precision host oracle is separate from the kernels: dense scores,
stable softmax and weighted sum. Its frozen full-op bound is
`5e-5 + 2e-5 * sum(abs(weighted_values))`. Zero-score and constant-value cases
also have analytic expected results. Small cases check every output; large cases
check predeclared boundary rows and heads, all values for finiteness, and whole
captured outputs against the control. The cross-backend triangle bound is 1.4e-4;
changes claiming unchanged arithmetic require bit identity. This is not an
extreme-range, nonfinite, independent HF, lifetime or full-model qualification.

The full-op matrix has 64 query heads and 8 KV heads, three warmups and ten
measured chains of four operations. Two blocks reverse arm order. Uploads,
cache construction, graph capture, readback and checking are outside timing;
operation completion, submission and waiting stay inside. Captures are written
only in correctness runs. Numbers below are both block medians; lower latency
is better. Every raw sample, including slow or flagged samples, is retained.
No standalone mx attention fixture was measured. Its separate
[model performance targets](../rocm-model-budget-20261010/README.md) remain open.

## Initial complete operation

The Vulkan control passes 37 processes and 6,623,232 oracle output checks.
The full comparison passes 115 processes, 460 chains, 19,980,288 oracle checks
and 230 whole-output comparisons. All 76 HIP eager/graph bank comparisons are
bit-identical; maximum Vulkan/HIP absolute difference is 2.38418579102e-7.
The 24-process timing matrix passes another 5,750,784 oracle checks. HIP graphs
do not remove the large gap to Vulkan.

| Prompt / history | Vulkan ms/op | HIP eager ms/op | HIP graph ms/op | HIP graph / Vulkan latency | mx target |
| --- | --- | --- | --- | --- | --- |
| 512 / 0 | 1.845 / 1.847 | 2.023 / 2.018 | 2.013 / 2.015 | 1.09x / 1.09x | Not measured |
| 2048 / 0 | 14.603 / 14.571 | 21.265 / 21.272 | 21.299 / 21.282 | 1.46x / 1.46x | Not measured |
| 4096 / 0 | 53.660 / 53.713 | 80.059 / 80.148 | 80.064 / 80.092 | 1.49x / 1.49x | Not measured |
| 512 / 2048 | 9.674 / 9.685 | 13.163 / 13.163 | 13.170 / 13.190 | 1.36x / 1.36x | Not measured |

## Isolated QK and PV layouts

Each panel holds 32 queries, 64 keys and dimension 128. QK multiplies F32 queries
by exactly decoded F16 keys; PV uses positive normalized F32 probabilities and
F16 values. Both accumulate in F32. The independent double oracle checks every
output at the predeclared `1e-6 + 1e-5 * sum(abs(products))` bound. The adjacent
layout assigns four adjacent values to a lane; the strided layout assigns every
eighth value. Compiler output witnesses 128-bit shared reads instead of paired
32-bit reads, with no spills on gfx906 or the compile-only gfx1151 target.
This changes the dot's addition grouping and is held to the original bound.

Eight small processes pass 1,966,080 output checks. Sixteen timing processes
pass 736,100,352 checks, across 160 measured chains and 48 warmups of eight
kernels. Each batch has the panel count shown; these cache/launch and occupancy
conditions differ from fused attention, so the component times are not additive
full-op estimates. The small, mixed changes do not establish its bottleneck.

| Panel operation | Panels | Strided us/batch | Adjacent us/batch | Adjacent / strided latency | mx target |
| --- | --- | --- | --- | --- | --- |
| qk | 128 | 59.09 / 58.96 | 57.18 / 57.61 | 0.968x / 0.977x | Not measured |
| pv | 128 | 64.21 / 63.86 | 65.01 / 64.55 | 1.012x / 1.011x | Not measured |
| qk | 1024 | 261.22 / 260.87 | 247.13 / 256.12 | 0.946x / 0.982x | Not measured |
| pv | 1024 | 238.22 / 238.42 | 233.98 / 233.98 | 0.982x / 0.981x | Not measured |

A separate profiler matrix witnesses all four kernels and warmup/measured ROCTX
ranges, with eight panel dispatches and no copy helper in each measured range.
Unprofiled controls bracket every trace. Trace/control wall ratios are 1.13 for
strided QK and 1.01-1.02 for the others; use unprofiled figures for speed.
All 13 processes pass 37,765,120 output checks. The first trace reader mistakenly
counted 24 ROCm copyBuffer helper kernels beside 32 panel kernels in its smoke;
the native program passed. That attempt and reader remain, and the corrected
reader separates known copy helpers while refusing unknown extra kernels.

## Full-kernel phase clocks

Original HIP, matched clock-off and clock-on builds preserve every operation.
On gfx906, clock-off uses 88 VGPR and clock-on 109, both without spills.
Five actual device clock reads mark load plus barrier, QK, softmax/rescaling,
and PV plus barrier. The first lane of every workgroup records its cumulative
cycles and exact iteration count. Compiler boundaries retain the phase outputs.
These are instrumented workgroup cycles: workgroups overlap and their totals
cannot be summed into GPU wall time or treated as hardware stall counters.

All 24 correctness processes pass 4,325,376 oracle checks and 32 bit-identical
whole-output comparisons. The 24 timing processes pass 5,750,784 further checks.
Clock overhead is 5-8 percent against the matched off control, which tracks the
original kernel closely. A slow 51.10 ms clocked history-case correctness sample
remains in the raw smoke record; it is not silently replaced by the timing matrix.

| Prompt / history | Original ms/op | Clock off ms/op | Clock on ms/op | Clock on / off latency | mx target |
| --- | --- | --- | --- | --- | --- |
| 512 / 0 | 2.016 / 2.014 | 2.011 / 2.026 | 2.145 / 2.141 | 1.07x / 1.06x | Not measured |
| 2048 / 0 | 21.295 / 21.327 | 21.329 / 21.350 | 22.810 / 22.865 | 1.07x / 1.07x | Not measured |
| 4096 / 0 | 80.036 / 80.092 | 80.061 / 80.081 | 86.133 / 86.181 | 1.08x / 1.08x | Not measured |
| 512 / 2048 | 13.165 / 13.184 | 13.144 / 13.153 | 13.809 / 13.788 | 1.05x / 1.05x | Not measured |

| Prompt / history | Load and barrier, percent | QK, percent | Softmax, percent | PV and barrier, percent |
| --- | --- | --- | --- | --- |
| 512 / 0 | 40.88 / 40.94 | 25.66 / 25.64 | 5.06 / 5.05 | 28.41 / 28.38 |
| 2048 / 0 | 37.72 / 37.75 | 26.89 / 26.88 | 5.29 / 5.29 | 30.10 / 30.08 |
| 4096 / 0 | 37.30 / 37.30 | 27.05 / 27.05 | 5.32 / 5.32 | 30.33 / 30.33 |
| 512 / 2048 | 41.82 / 41.85 | 25.60 / 25.59 | 4.90 / 4.89 | 27.68 / 27.67 |

Loading accounts for 37-42 percent of the recorded cycles. This selects a
narrow experiment: stage eight exact half values per 128-bit global load and
write the decoded values to the same shared tile with float4 stores, keeping
QK, softmax, PV and their addition order unchanged. The stage includes waiting
for other waves; the timing experiment tests the combined loading/barrier cost,
not a claim that every recorded cycle is physical DRAM traffic.

## Wider KV staging

The candidate compiles for gfx906 and gfx1151, with 92 and 101 VGPR respectively
and zero scratch/spills. Its matched scalar controls use 88 and 100. The intended
128-bit global loads and shared stores are present in the assembly. The first
build's Werror rejection of an unused helper in the scalar arm is retained;
conditional compilation removes it from that arm without disabling warnings.

All 114 correctness processes pass: 456 chains, 19,906,560 independent oracle
checks and 152 whole-output comparisons bit-identical to original HIP, covering
all 38 random, analytic, row/page-tail and large fixtures. All 32 matched timing
processes pass another 7,667,712 oracle checks over 320 measured chains and 96
warmups. The vector candidate reduces HIP latency by 18-25 percent, beats Vulkan
by about 12 percent at 512 fresh rows, stays about 2 percent behind with history,
and remains 20-23 percent behind at 2048/4096 fresh rows. The paired scalar
control tracks original HIP, so this gain exceeds that measured layout variation.
It does not establish a standalone mx or model-performance gate.

| Prompt / history | Vulkan ms/op | Original HIP ms/op | Scalar control ms/op | Vector staging ms/op | Vector / Vulkan latency | mx target |
| --- | --- | --- | --- | --- | --- | --- |
| 512 / 0 | 1.848 / 1.845 | 2.018 / 2.023 | 2.011 / 2.019 | 1.623 / 1.626 | 0.88x / 0.88x | Not measured |
| 2048 / 0 | 14.587 / 14.624 | 21.270 / 21.284 | 21.286 / 21.294 | 17.458 / 17.491 | 1.20x / 1.20x | Not measured |
| 4096 / 0 | 53.502 / 53.808 | 80.016 / 80.119 | 80.064 / 80.061 | 65.630 / 65.647 | 1.23x / 1.22x | Not measured |
| 512 / 2048 | 9.684 / 9.662 | 13.176 / 13.157 | 13.160 / 13.159 | 9.855 / 9.826 | 1.02x / 1.02x | Not measured |

## Activity, artifacts and remaining work

The same low-overhead one-second monitor runs before and throughout each arm.
Flags were frozen before measuring: unrelated CPU at least one observed core,
system disk traffic at least 100 MiB/s, other GPU activity at least 20 percent.
No samples are removed, replaced or selected for an idle interval.

| Timing matrix | Flagged calls / measured chains | Maximum observed unrelated CPU cores | Inaccessible process observations |
| --- | --- | --- | --- |
| Initial full op | 0 / 0 | 0.600 | 22 |
| Isolated panels | 0 / 0 | 0.390 | 4 |
| Phase clocks | 4 / 2 | 3.958 | 23 |
| Vector staging | 0 / 0 | 0.490 | 39 |

All four matrices have complete interval coverage and no missing GPU counters.
Inaccessible or short-lived processes are unknown, not zero activity; disk
counters are system-wide. Each matrix keeps its flagged calls, warmups and slow
samples alongside both planned blocks. Inputs, source/binary hashes, native
exit status, container removal and exact starting VRAM restoration are checked.
All owned runs are terminal and their reservations released.

The private artifact root on the rig is
`/zpool1/llmx-xdev-validation/rocm-probe-20261009`, mirrored in the workstation's
`.tmp-rocm-probe-20261009`. The immutable `attention-evidence-20261010.tar.gz`
contains 1279 files, 5966089 bytes, SHA256
`b673f653723dd86286c7012d7a7b5c75233a13d889724ea4acfcb71d4b1b3103`.
Its manifest pins every included source, build log, assembly, protocol, plan,
raw process output, trace, monitor and verified summary. ELF executables and
full captured output tensors remain on the rig; earlier compressed metadata
archives are not duplicated. Failed launch/build/parser attempts remain too.

The principal summaries are `attention-full-performance-1.json`,
`attention-panels-performance-1.json`, `attention-panels-trace-summary-2.json`,
`attention-clocks-performance-1.json` and `attention-stage-performance-1.json`.
Each has its raw run directories and frozen plan. The initial HIP build's spills
were removed by declaring its actual 256-thread bound before any GPU run;
that rejected build remains alongside the measured builds.

The candidate is private. Remaining work is a matched mx attention control,
the residual large-prompt gap, and the combined compute/collective admission
budget beside the [Q8 costs](../rocm-q8-compute-20261010/README.md) and
[model budget](../rocm-model-budget-20261010/README.md). Production integration
then needs the full independent HF, model, range, lifetime, split and matched
mx gates; no new ROCm flag, runtime dependency or empty backend is shipped.
gfx1151 compilation is not Strix Halo hardware validation.
