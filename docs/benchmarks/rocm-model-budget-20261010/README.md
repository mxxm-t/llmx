# ROCm model budget, 2026-10-10

The current Vulkan model path remains below shipped mx on this Qwen3-32B Q8_0
workload. This record refreshes the communication budget before the ROCm
admission decision; it does not qualify a new backend or close an HF gate.
All 20 planned processes complete: 120 measured phases and 40 warmup phases.
Every sample is retained, including activity flags and each process's first
measured decode after its prompt passes.

## Workload and identities

One sequence, prompt 512 and decode 128 forced token IDs, independently from
empty history; KV 512 tokens in F16, ubatch 512, four CPU threads, direct load.
LCG seeds 12345 and 777 give identical input IDs in both runtimes, without
sampling, drafts or text conversion. One prompt/decode warmup precedes three
prompt and three decode samples. Each width runs main-F16, diagnostic-F16, mx,
main-int8, diagnostic-int8, then the reverse. Rates below are tokens/s, the
two block medians, higher better; the raw summary preserves ranges and samples.
The mx target is repeated for the separately requested llmx policies, not a
claim that mx has an equivalent dtype flag or arithmetic policy.

The measured llmx main is `fb82404338a6c93f5cf355279ce968583aaf57cc`; private
diagnostic `73d2917e64336a6e6530841a5c654fe080fb169e` differs by seven lines
that omit only one-row sums and count omissions. Both are clean detached
trees with identical build routes. The wrapper adds timestamps and completed
matrix-path witnesses outside timing, with no runtime source instrumentation
in the main arm. It returns and copies full-vocabulary logits inside timing,
as both public APIs do. Load, reset, logging and validation are outside timing.

mx is the unmodified shipped library from image
`a83c92cf625046e09959de1a7b1e7b241e723646b5e9c6d6f47e6ce28fdc83e3`,
source `eefc4e7321c869496146697d63362f073941aed6`. Its public
`llama_get_logits_ith` synchronizes before returning. Logs confirm all 65
layers offloaded, the requested tensor group, context, cache and flash attention,
custom F32 all-reduce initialization and the token graph path. Its pinned
dispatcher uses small-message custom F32 sums and large-message RCCL BF16
transfer; this is a source-based path inference, not per-kernel tracing.
Its Q8 products use its shipped activation quantization; llmx reports the
block-int16 or block-int8 paths actually executed on every measured device.

Every arm runs inside image
`d5591273e00b5735bd3419009ffd7575821dcf957897010e03ba1d1c6a0ae14f`,
HIP 7.2.1, AMD clang 22, Mesa 25.2.8, Vulkan 1.3.275 and shaderc 2025.2.
The shipped HIP library also records AMD clang 22 / ROCm 7.2.1. Wrappers use
the same explicit C++17/O3/AVX2/FMA/F16C flags. All arms set the vendor
prerequisites HSA_FORCE_FINE_GRAIN_PCIE=1 and GPU_MAX_HW_QUEUES=8,
GGML_ENABLE_CUSTOM_AR=1 and the same explicit pinned-library search path;
there is no ISA spoof or collective gate override. These are experiment
prerequisites, not new llmx runtime knobs. Vulkan and ROCm use their respective
drivers in that same environment.

Widths two and four use PCI 83/86 and 83/86/89/8c under pci0000:80 on the MI50
rig, with CPUs 4/5/6/7 and siblings 12/13/14/15 reserved. Production uses
other cards. Default clocks/power remain unchanged. Model SHA256 is
`2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169`,
34,817,718,912 bytes, freshly hashed before and after the matrix. Every
process checks its size and modification time and the frozen binary hashes.

## Model throughput

The diagnostic's decode results are intentionally wrong. Its rate is only
useful for estimating available time; it is never a working model result.
Prompt sums remain active and prompt output hashes match main within each dtype.
Small prompt differences between those builds do not establish a tuning gain.

| GPUs | Dtype | Phase | llmx | mx target | llmx vs mx | Invalid diagnostic |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | f16 | pp | 360.12 / 358.74 | 567.34 / 567.49 | -36.53% / -36.79% | 359.74 / 358.98 |
| 2 | f16 | tg | 29.64 / 29.74 | 33.63 / 33.69 | -11.88% / -11.72% | 36.64 / 36.68 |
| 2 | int8 | pp | 532.31 / 532.35 | 567.34 / 567.49 | -6.18% / -6.19% | 532.15 / 532.29 |
| 2 | int8 | tg | 30.02 / 29.83 | 33.63 / 33.69 | -10.74% / -11.46% | 36.79 / 36.78 |
| 4 | f16 | pp | 461.86 / 461.99 | 804.94 / 800.92 | -42.62% / -42.32% | 461.79 / 461.48 |
| 4 | f16 | tg | 29.34 / 28.28 | 51.21 / 51.23 | -42.72% / -44.80% | 62.91 / 62.96 |
| 4 | int8 | pp | 587.86 / 589.90 | 804.94 / 800.92 | -26.97% / -26.35% | 588.96 / 589.87 |
| 4 | int8 | tg | 28.28 / 28.84 | 51.21 / 51.23 | -44.78% / -43.70% | 63.91 / 63.42 |

## Decode budget

With 128 sums per generated token, available time for one sum is
`(1/mx_tps - 1/diagnostic_tps) * 1000000/128`. Removed cost substitutes
main for mx. Both are estimates: omitting sums changes intermediate values,
launches and scheduling, so replacement kernels need a real model test.
The earlier matched collective screen is shown separately; it is not a new
HIP model measurement and retains its previously reported CPU contention.

| GPUs | Dtype | Available budget, us/sum | Removed cost, us/sum | Earlier HIP graph screen, us/sum |
| --- | --- | --- | --- | --- |
| 2 | f16 | 19.07 / 18.88 | 50.38 / 49.66 | 10.60 / 10.49 |
| 2 | int8 | 19.94 / 19.49 | 47.89 / 49.51 | 10.60 / 10.49 |
| 4 | f16 | 28.36 / 28.43 | 142.11 / 152.20 | 24.50 / 23.77 |
| 4 | int8 | 30.31 / 29.32 | 154.00 / 147.69 | 24.50 / 23.77 |

A graph collective below this budget supports continuing the measured
communication approach, but does not prove that a complete HIP backend wins.
Q8 decode throughput, launch coordination with real compute and full prompt
attention remain the admission work. Prompt and width-two findings stay
separate; F16 and int8 are not pooled to hide a default-path loss.

## Activity and bounded checks

One-second monitoring spans all calls. The frozen flags are unrelated CPU
at least one observed core, any individual disk at least 100 MiB/s and an
outside-group GPU at least 20 percent. 20 of 20 whole calls and
3 of 120 measured phases overlap a flag; every one is retained.
The maximum observed unrelated CPU during measured phases is 2.92 cores.
Missing GPU-counter observations in those phases: 0; combined leading
and trailing coverage gaps: 0.000000 seconds. Longest monitor interval:
1.058 seconds. System-wide disk counters include
model loading, and an interval at a timed boundary can overlap that loading.
Short-lived/new/inaccessible processes can be missed; the raw activity retains
these limits rather than treating them as idle time.

Every call crosses the disk flag while loading; 16 also cross the CPU flag.
The three flagged measured phases cross only the unrelated-CPU flag: width-two
reverse-block diagnostic-F16 decode samples 1 and 2, and width-four reverse-block
mx decode sample 2 (zero-based sample indexes). No measured phase crosses the
disk or outside-GPU flag. These samples remain in the medians and full record.

All final vectors are finite and phase hashes stable within each arm;
main and diagnostic prompt hashes agree at both widths/dtypes, while their
decode hashes differ. Every diagnostic process reports exactly 65,536 omitted
sums. This confirms the negative control, not its correctness. The preceding
corrected smoke passes three processes and 12 phases; the first smoke's mx
pre-load exit 127 (missing transitive library path) is retained. RCCL prints
warnings about unexposed topology nodes during initialization; those logs
remain visible. The initial verifier's stdout parsing and Windows text-mode
errors are also retained as harness errors, not failed numerical samples.

Each process has a 300-second timeout plus 10-second kill grace, each group
3300 seconds. Containers drain and are removed; all participating cards return
to their exact 10,932,224-byte starting VRAM. No production process, driver,
package, clock or power setting is changed. No own competing build/test/model
download runs on the rig during timing; the independent documentation checks
run on the workstation.

## Evidence and remaining work

Canonical evidence is `/zpool1/llmx-xdev-validation/rocm-probe-20261009`, with
a workstation copy under `.tmp-rocm-probe-20261009`: frozen protocol/plan,
model-budget.cpp and both build manifests, clean revision IDs and private
diagnostic patch, model hashes, timing stdout/stderr/call records, activity
and cleanup logs, per-width checked records and `model-budget-summary-w2-4.json`.
The full summary includes each sample, range, activity window and derived budget.
The diagnostic branch is private and does not merge. This documentation-only
checkpoint runs documentation/dead-code checks; it adds no runtime support.
The next bounded work is the Q8 decode and full attention probes in [ROCM](../../ROCM.md).
