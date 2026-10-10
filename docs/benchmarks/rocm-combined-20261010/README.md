# Dependent Q8 compute and collective screen (2026-10-10)

The private HIP graph prototype fits the earlier estimated decode budget in
this structured workload on two and four MI50s. This advances the investigation
to actual model projection shapes; it does not qualify a production ROCm backend
or establish full-model parity with mx. [ROCM](../../ROCM.md) owns that plan.

## Workload and checks

Eight physically distinct Q8_0 square matrices per rank rotate over the chain.
Every product consumes the previous result. One or four products precede each
ordered sum; an intermediate product adds its input, and the last rank sum adds
to the global state, which feeds the next step. Each rank owns its buffers.
The local control replaces communication with one local residual add. Its
amplitudes differ, while compute kernels, dimensions and weight bytes are the
same; it is a compute-only floor, not an equivalent model implementation.

Every block contains magnitudes 127, 9 and thirty 4s, permuted by bank/block,
with row/column signs. The exact binary16 scale is 2^-24. For the sign vector
s, W*s = c*s, with c = n/2^21. This gives an independent analytic recurrence:
amplitude after k steps is (1 + members*c*(1+c)^(products-1))^k. The local
control substitutes one for members. These are structured finite inputs,
not random models or a general range proof.

The bound was frozen before compilation: 1e-7 + amplitude*k*FLT_EPSILON*
(2+(32*products+members)*gain), with gain the per-step added amplitude.
It budgets residual rounding and the smaller product contribution for this
screen only; no production dtype or HF bound changes. A scalar decoder checks
all elements of 512 small rows against the coefficient. Analytic mutation
controls show the timed-width envelope rejects stale inputs, omitted
intermediate products and omitted rank sums.

Every native chain checks all final values against that recurrence, guards,
unchanged weights and exact rank/replay hashes. HIP also checks every block's
collective completion/epoch. Vulkan must report the executed block-int16 or
block-int8 path. There is no cross-backend bit-identity claim.

The 48 smoke processes cover widths 64 and 5120, both group sizes and dtypes,
one/four products and local/sum modes: 192 chains, 1007616 final-state values.
The complete timing matrix covers 64 processes and 512 chains, including 320
measured and 192 warmup chains, with 7864320 final-state comparisons. All pass.
Largest bound fractions are 0.10335/0.02966 for two/four-card smoke and
0.01716/0.01001 for timing. Replay identity holds throughout these fixtures.

## Environment and ownership

Published main is a0acdaedc. The unchanged Vulkan library is from clean
fb8240433; src, CMake and cmake have no difference between those commits.
Both wrappers use pinned HIP 7.2.1/clang22, -O3 -ffp-contract=off and default
clocks in image
`8bf6f488e3d79cc1c672294e36627c484209a0fd41da4ac9b0c00b8b409ac38a`.
HSA_FORCE_FINE_GRAIN_PCIE=1 and GPU_MAX_HW_QUEUES=8 are identical vendor
prerequisites in both arms, not new llmx runtime knobs.

The two-card group is PCI83/86, the four-card group PCI83/86/89/8c, all under
pci0000:80. CPU4-7 and siblings12-15 were reserved. Production GPUs8/9 were
untouched. No owned build, transfer or download overlaps timing. All kernels
compile for gfx906 and gfx1151 with zero scratch spills; gfx1151 is compile-only.
The gfx906 F16 product uses 51 VGPRs/27 SGPRs, occupancy 4; int8 uses 39/21,
occupancy 6. These compiler resource counts are not measured hardware stalls.

The first build attempt stopped before compilation because its container lacked
the source's absolute mount. Its files/log remain alongside the corrected build.
The corrected host oracle, both HIP targets and Vulkan wrapper compile cleanly.
Sources, assembly, binary/library hashes and exact commands are in the archive.
No mx kernel source is copied; the selected private llmx kernels are reused.

## Matched latency

Width 5120, 128 dependent steps, five measured rounds after three warmups per
process. Cells are forward/reverse order medians in us per step, lower better.
The host clock ends after every member completes; setup and final readback are
outside it. HIP device events and host enqueue times are retained separately.

| GPUs | Dtype | Products/sum | HIP local | HIP sum | Vulkan local | Vulkan sum |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | f16 | 1 | 47.89 / 48.25 | 57.29 / 58.23 | 56.96 / 64.37 | 166.75 / 167.83 |
| 2 | f16 | 4 | 196.74 / 195.55 | 202.08 / 204.58 | 190.27 / 190.83 | 253.84 / 240.43 |
| 2 | int8 | 1 | 48.34 / 48.46 | 57.50 / 57.52 | 53.12 / 53.57 | 149.53 / 161.59 |
| 2 | int8 | 4 | 195.49 / 190.59 | 200.42 / 201.84 | 191.34 / 191.08 | 248.62 / 250.15 |
| 4 | f16 | 1 | 54.92 / 54.88 | 73.06 / 74.18 | 72.03 / 71.80 | 233.15 / 232.97 |
| 4 | f16 | 4 | 198.09 / 199.25 | 216.55 / 219.57 | 222.81 / 224.67 | 301.72 / 291.02 |
| 4 | int8 | 1 | 55.49 / 54.88 | 71.59 / 73.93 | 66.68 / 67.95 | 232.61 / 236.93 |
| 4 | int8 | 4 | 197.32 / 198.26 | 217.18 / 218.05 | 214.48 / 213.46 | 293.21 / 294.88 |

HIP sum is faster than Vulkan sum in every measured cell. To screen whether
extra computation consumes the earlier communication margin, subtract the
fresh Vulkan local control from HIP sum. Negative means the combined HIP arm
is faster than that local Vulkan control; it is not a negative collective time.
This difference includes backend and host-submission costs, not just transport.

The available budgets below come from the earlier
[model-budget experiment](../rocm-model-budget-20261010/README.md), using a
private Vulkan no-sum diagnostic whose model outputs are wrong by construction.
Use them as screening estimates, not a new merge gate or full-model result.
The smaller of its two available-budget estimates is shown.

| GPUs | Dtype | Products/sum | HIP sum minus Vulkan local us | Available estimate us | Matched mx coupled us |
| --- | --- | --- | --- | --- | --- |
| 2 | f16 | 1 | 0.33 / -6.15 | 18.88 | Not measured |
| 2 | f16 | 4 | 11.80 / 13.75 | 18.88 | Not measured |
| 2 | int8 | 1 | 4.38 / 3.95 | 19.49 | Not measured |
| 2 | int8 | 4 | 9.08 / 10.76 | 19.49 | Not measured |
| 4 | f16 | 1 | 1.03 / 2.38 | 28.36 | Not measured |
| 4 | f16 | 4 | -6.26 / -5.11 | 28.36 | Not measured |
| 4 | int8 | 1 | 4.91 / 5.98 | 29.32 | Not measured |
| 4 | int8 | 4 | 2.70 / 4.59 | 29.32 | Not measured |

The historical matched mx full-model decode targets remain 33.63/33.69 tok/s
for two cards and 51.21/51.23 tok/s for four cards, from that budget campaign.
This probe reports latency of a different workload, so those throughput figures
are not converted to a claimed model speedup. Full-model mx parity remains open.

## Activity and evidence

The monitor runs every second throughout every arm. Frozen flags are unrelated
CPU >=1 core, any measured disk >=100 MiB/s and outside-group GPU >=20 percent.
Every planned run, warmup, measured sample, flag and unknown observation stays.

| GPUs | Stage | Flagged calls / measured chains | Max unrelated CPU cores | Inaccessible process observations |
| --- | --- | --- | --- | --- |
| 2 | smoke | 0 / 0 | 0.410 | 39 |
| 2 | timing | 0 / 0 | 0.360 | 12 |
| 4 | smoke | 1 / 1 | 3.149 | 41 |
| 4 | timing | 0 / 0 | 0.390 | 17 |

The four-card smoke has one unrelated-CPU flag; no timing chain has a declared
flag. This does not assert the machine was idle. Inaccessible observations remain
unknown. No coverage gaps or missing GPU counters were found. Hash validation,
owned-container removal, default clocks and exact starting VRAM restoration pass.

Immutable `combined-evidence-20261010.tar.gz`: 395 files, 342685 bytes,
SHA256 `30f25b3920a8fe1982f7d17ecbfc76cb9b7f3d991fb5d4b31515c8597a9cc713`,
verified on rig and workstation. Local evidence is under `.tmp-rocm-probe-20261009/`;
the rig root is `/zpool1/llmx-xdev-validation/rocm-probe-20261009/`.
Native binaries remain on the rig; the archive includes source, compiler
assembly, oracle, frozen protocol/plan, all samples, monitors and checks.

## Decision and remaining work

The dependent square-matrix bracket passes its screening estimate in both
orders. Next verify actual Qwen3-32B projection shapes and data dependencies,
including expansion/contraction and their preparation costs, before treating
this as model admission. Prompt attention remains a separate measured gap.
Full HF/model/range, failure/lifetime, splits and matched mx qualification are
still required for production. No new runtime code, dependency, flag or Windows
executable ships here. Core dtype remains complete; CPU emulation performance
is nonblocking while its correctness coverage remains required.
