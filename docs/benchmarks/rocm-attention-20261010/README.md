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

At this initial checkpoint the candidate is private. Remaining work is a matched
mx attention control (completed in the follow-up below), the residual large-prompt gap, and the combined compute/collective admission
budget beside the [Q8 costs](../rocm-q8-compute-20261010/README.md) and
[model budget](../rocm-model-budget-20261010/README.md). Production integration
then needs the full independent HF, model, range, lifetime, split and matched
mx gates; no new ROCm flag, runtime dependency or empty backend is shipped.
gfx1151 compilation is not Strix Halo hardware validation.

## Matched shipped mx control (2026-10-10 follow-up)

The missing reference comparison is now measured. These are complete native
attention operations on the same MI50, not equal-precision kernel comparisons.
All arms receive the same two logical Q/K/V fixtures, causal mask, head width
128 and 64 query heads over eight KV heads. Vulkan and the private HIP probe
keep the paged cache; mx takes its contiguous F16 cache padded to 256 positions.
Layout conversion, uploads, graph capture and readback are outside timing.
Execution includes every kernel and synchronization needed for completion.

The mx wrapper links the shipped library at
`eefc4e7321c869496146697d63362f073941aed6`; `libggml-hip.so` and its loaded
versioned library both have SHA256
`cda090783159dfcff2f75fc30caa02b67ca305e10a7b37204f66aa9652ee4dfa`.
All four loaded GGML libraries are checked against the pinned copies. Its
default precision and graph policy are retained. The wrapper uses the public
backend API, without copying or rebuilding mx's kernels.

All arms run in the same profiler-capable image
`sha256:8bf6f488e3d79cc1c672294e36627c484209a0fd41da4ac9b0c00b8b409ac38a`,
on PCI `0000:89:00.0`, CPUs 4-7, default clocks, with the same vendor
prerequisites and monitoring as the earlier probes. The Vulkan and vector-HIP
binaries are unchanged from that evidence. The added image library is present
in the traced and untraced arms alike.

The matrix uses four operations per chain, three warmups and ten measured
chains per process. The two numbers retain the forward and reverse arm orders;
lower milliseconds per operation are better. No traced time enters this table.

| Prompt / history | Vulkan ms/op | Vector HIP ms/op | mx HIP target ms/op | HIP / mx latency |
| --- | --- | --- | --- | --- |
| 512 / 0 | 1.842 / 1.843 | 1.634 / 1.630 | 1.659 / 1.657 | 0.98x / 0.98x |
| 2048 / 0 | 14.603 / 14.588 | 17.459 / 17.474 | 11.581 / 11.592 | 1.51x / 1.51x |
| 4096 / 0 | 53.488 / 53.749 | 65.623 / 65.632 | 35.865 / 35.867 | 1.83x / 1.83x |
| 512 / 2048 | 9.708 / 9.640 | 9.856 / 9.834 | 5.075 / 5.076 | 1.94x / 1.94x |

The short fresh prompt is slightly faster than mx in both blocks. The private
HIP probe remains 51 percent slower at 2048 rows, 83 percent at 4096, and
94 percent slower with the 2048-token history. Vulkan also retains a reference
gap in those cells. These are standalone costs, not model tokens per second
or evidence that the ROCm backend is ready.

### Dispatch and precision witnessed

Five before/trace/after groups cover 33/7 and all four large shapes. Each
measured region launches one HIP graph and contains two attention calls,
each reaching `flash_attn_tile<128,128,8,8,false>` and the combine kernel.
At 2048/4096 rows each also reaches the mask-to-KV-maximum helper, which
lets fully masked tiles be skipped. No copy, capture or graph instantiation
occurs inside the measured region. Traced/untraced ratios are about 1.20 for
the short case and 1.00-1.01 for the large cases; all controls are retained.

The tile covers eight query rows across eight query heads, sharing K/V across
the GQA group. Source inspection shows F16 query/probability storage and F16
PV accumulation, with F32 QK sums and softmax denominators. llmx's probe
retains F32 queries, probabilities and accumulation. Reuse, tiling and narrower
arithmetic are candidate explanations to isolate, not a measured attribution
of the full speed difference to any one of them.

The original harness required a singleton causal row to equal its sole V.
mx instead returned 0.513599157333 for the exact value 0.513671875. Its source
adds `3.0f * 0.6931f` to the softmax maximum: the singleton probability rounds
to 0.125 in the F16 numerator while the F32 denominator is about 0.1250177.
An independent calculation predicts 0.513599136548. The corrected reference
control checks that derived arithmetic within eight F32 epsilons; the initial
failure and the diagnostic build remain in the archive. A separate earlier
launch typo used an unsupported Vulkan wrapper mode and is retained too.

No llmx bound is widened. Every reference call still reports its error against
the original double oracle and the count beyond llmx's unchanged bound. A
reference-only gross sanity envelope was fixed before its first run, alongside
analytic zero-query/constant-value controls. It checks the harness, not a new
llmx numerical policy. The complete smoke matrix's largest reference absolute
error is 0.000924885; 2512172 of its 6365184 reference checks exceed llmx's
bound, including repeated warmup/chain checks. Large-shape results are:

| Prompt / history | mx maximum absolute error | Maximum fraction of llmx bound |
| --- | --- | --- |
| 512 / 0 | 0.000808607 | 12.91x |
| 2048 / 0 | 0.000627659 | 10.44x |
| 4096 / 0 | 0.000819639 | 13.48x |
| 512 / 2048 | 0.000339345 | 5.65x |

### Verification, artifacts and next work

All 108 smoke processes pass their stated controls: 432 chains and 19095552
independent oracle comparisons across the three arms, with input fixture
hashes equal, finite outputs, unchanged inputs, preserved sentinels and
repeatable complete output hashes. All 15 trace/control processes pass, with
1695744 oracle comparisons. All 24 timing processes pass, retaining 240
measured chains, 72 warmups and 7667712 actual oracle comparisons (counting
correction explained in the reuse follow-up below). These ordinary
finite synthetic fixtures do not establish HF, full-model or extreme-range
correctness for any backend.

One smoke call has background CPU activity flagged; no measured smoke chain
does. Timing and trace controls have no declared flags. The timing monitor's
maximum unrelated CPU use is 0.620 cores, with 38 inaccessible process
observations retained as unknown. All intervals are covered and no GPU counter
is missing. Every planned sample remains; cleanup and exact starting VRAM
restoration pass for all three completed stages.

The follow-up archive is `attention-mx-evidence-20261010.tar.gz`,
533 files, 536626 bytes, SHA256
`711a4f0ac721997f643371aa4961f96a68153c046dcff469cffb5fe1aada1617`, verified on workstation and rig at the artifact
root above. It contains frozen protocols, plans, wrapper/build sources,
native output, traces, monitors, summaries and failed attempts. Full output
tensors and executable binaries remain on the rig; library hashes identify
the shipped reference. The original attention archive remains unchanged.

Next are separate GQA-reuse and KV-tiling controls with llmx's current F32
arithmetic, then combined compute/collective admission against the model
budget. The full HF/model/range/lifetime/split and matched mx release gates
remain open. No production backend, dependency, flag or Windows executable
is added by this documentation checkpoint.

## GQA reuse and larger exact loads (2026-10-10 follow-up)

The next screen separates three changes: regrouping the same 32 query/head
pairs, doubling the pairs sharing a KV load, and staging more KV positions
while retaining the existing 16-token softmax steps. Query scaling, QK
reductions, softmax and PV sums stay F32. It follows the public checkpoint at
`753182723afab0abb1903b8f209e8c2237fef383`; all kernels remain standalone.

An arm named `q8h8k64` processes eight query rows across eight query heads
and stages 64 KV positions. The earlier vector HIP is `prior`; `q32h1k16`
is its rebuilt control in the new source. Vulkan and shipped mx are the same
reference binaries used above. mx retains its narrower arithmetic and
contiguous cache; its error report and all llmx bounds remain unchanged.

The short smoke passes 27 processes, 7299072 actual oracle comparisons and
48 bit-identical whole-output comparisons. All eight gfx906 builds have zero
scratch spills. The screen then runs all 44 planned processes in both orders,
with three warmups and three measured chains of four operations per process.
Both block medians remain below; lower latency is better.

| Arm | 2048 / 0 ms/op | 512 / 2048 ms/op |
| --- | --- | --- |
| vulkan | 14.605 / 14.606 | 9.619 / 9.606 |
| prior | 17.491 / 17.469 | 9.810 / 9.822 |
| q32h1k16 | 17.120 / 17.082 | 9.616 / 9.620 |
| q16h2k16 | 16.585 / 16.537 | 9.522 / 9.544 |
| q64h1k16 | 18.644 / 18.650 | 10.133 / 10.108 |
| q32h2k16 | 17.726 / 17.751 | 10.080 / 10.066 |
| q16h4k16 | 17.097 / 17.086 | 9.933 / 9.896 |
| q8h8k16 | 16.808 / 16.838 | 9.806 / 9.832 |
| q8h8k32 | 16.356 / 16.339 | 9.495 / 9.500 |
| q8h8k64 | 16.116 / 16.134 | 9.375 / 9.384 |
| mx | 11.590 / 11.588 | 5.061 / 5.065 |

Grouping heads without increasing the pairs per block gives a modest gain.
Doubling rows alone loses, as does 32 rows across two heads in these cells.
The 8/8/64 arm is the best of the tested configurations in both screened
shapes and orders, but remains well behind mx. The rebuilt control is about
two percent faster than the previous binary despite the same intended
arithmetic, so the candidate is compared with both rather than assigning
that control difference to reuse. No losing arm or slow sample is removed.

| Query rows / heads / load positions | VGPRs | SGPRs | LDS bytes | Scratch bytes |
| --- | --- | --- | --- | --- |
| 16 / 2 / 16 | 94 | 34 | 16384 | 0 |
| 16 / 4 / 16 | 83 | 35 | 16384 | 0 |
| 32 / 1 / 16 | 92 | 35 | 16384 | 0 |
| 32 / 2 / 16 | 83 | 35 | 16384 | 0 |
| 64 / 1 / 16 | 92 | 36 | 16384 | 0 |
| 8 / 8 / 16 | 83 | 35 | 16384 | 0 |
| 8 / 8 / 32 | 75 | 36 | 32768 | 0 |
| 8 / 8 / 64 | 84 | 31 | 65536 | 0 |

The selected source also compiles for gfx1151. This is compile coverage only,
without Strix Halo hardware or performance claims. The fixed prototype
requires a GQA ratio divisible by eight; it implements no generic fallback.

### Selected configuration checked more broadly

The prior binary, rebuilt control and candidate pass 66 processes over rows
1/19/32/33/65, histories 0/7/67, a second GQA shape with two groups sharing
each KV head, all four large shapes and the zero-query/constant-value
controls. All 88 complete-output comparisons are bit-identical; 13596672
actual independent oracle comparisons pass. Inputs and sentinels are checked,
and candidate outputs repeat across warmup and graph replays.

The larger timing matrix keeps all five arms, both orders, three warmups and
ten measured chains per process, four operations per chain. All 40 processes
pass with 11501568 actual oracle comparisons.

| Arm | 512 / 0 ms/op | 2048 / 0 ms/op | 4096 / 0 ms/op | 512 / 2048 ms/op |
| --- | --- | --- | --- | --- |
| vulkan | 1.844 / 1.838 | 14.570 / 14.565 | 53.612 / 53.738 | 9.665 / 9.662 |
| prior | 1.636 / 1.634 | 17.438 / 17.471 | 65.661 / 65.653 | 9.832 / 9.827 |
| q32h1k16 | 1.595 / 1.597 | 17.106 / 17.096 | 64.197 / 64.167 | 9.596 / 9.608 |
| q8h8k64 | 1.369 / 1.369 | 16.130 / 16.134 | 61.984 / 61.974 | 9.348 / 9.376 |
| mx | 1.657 / 1.658 | 11.586 / 11.577 | 35.946 / 35.873 | 5.069 / 5.076 |

Candidate/mx latency ratios in row/history order 512/0, 2048/0, 4096/0 and
512/2048 are 0.83x / 0.83x, 1.39x / 1.39x, 1.72x / 1.73x, 1.84x / 1.85x. The probe's improvements do not
close the large-prompt or history gaps, and no standalone result is promoted
to a model-level speed claim.

### Activity, counting correction and artifacts

Every stage uses the same fixed image, MI50, CPU affinity, default clocks and
one-second monitor as the matched mx run. All planned samples remain.

| Stage | Processes | Flagged calls / measured chains | Inaccessible process observations |
| --- | --- | --- | --- |
| Screen | 44 | 0 / 0 | 20 |
| Wider smoke | 66 | 0 / 0 | 11 |
| Full timing | 40 | 1 / 1 | 54 |

The one CPU-flagged measured chain is the rebuilt control at 4096 rows in
the reverse block: 64.267 ms/op, retained beside that block's 64.167 ms/op
median and every other sample. No arm or block is replaced.

All intervals are covered, no GPU counter is missing, and exact starting
VRAM and container cleanup checks pass. Inaccessible activity remains unknown.
The full timing matrix's maximum observed unrelated CPU use is 3.619 cores.

The preceding matched-mx timing count is corrected from 11501568 to 7667712.
HIP and Vulkan overwrite the same two bank outputs during a four-operation
chain and verify only those final outputs; mx verifies all four distinct
outputs. The old report incorrectly counted four for every arm. No outputs,
timings, arithmetic bounds or pass/fail result change. The original archive
and native metadata remain, with the explicit corrected checker/summary in
this follow-up archive. New counts follow actual comparisons by arm.

The immutable `attention-reuse-evidence-20261010.tar.gz` contains 523 files,
552417 bytes, SHA256
`ea5998c026cf030a75aa1351754e7f5925bf62eca778cb8b3c2dc2136f25de19`, verified at the same workstation and rig artifact roots.
It retains sources, build/assembly reports, frozen protocols/plans, every
process output, exact-output comparison, monitor and summary. Full tensors
and executables remain on the rig. Earlier archives are unchanged.

Next is profiling the remaining QK/PV and synchronization cost of the selected
F32 path, followed by the combined compute/collective admission against the
model budget. These fixtures do not establish extreme-range, independent HF,
full-model, lifetime or multi-device correctness. Production ROCm and the full
release gates remain open; no runtime flag, dependency or executable is added
by this documentation checkpoint.
