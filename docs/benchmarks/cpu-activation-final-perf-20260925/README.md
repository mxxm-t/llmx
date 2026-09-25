# CPU activation final performance evidence, 2026-09-25

This package preserves all 128 common-harness and 72 CLI calls from the frozen
final performance plans. It contains raw stdout/stderr, terminal records and
monitoring for every call. No sample was discarded or replaced. This is a
performance evidence checkpoint, not a claim of performance neutrality,
universal mx parity, external HF correctness or merge approval.

The measured candidate has an observed **-6.568% CLI Q5 decode cost**, with all
six paired blocks slower than base (range -11.884% to -0.919%). The layout
control is -4.110% in that cell, with five slower blocks and one faster block.
The common harness gives +0.427% candidate/base and +4.692% candidate/mx for
Q5 decode. These are different prompts, histories and translation units;
they do not cancel the CLI result. Layout sensitivity and activity limit
causal attribution, but do not justify erasing the observed cost or calling
it proven noise. Adoption requires assessing these costs alongside the
correctness repair and the complete workload tradeoff.

## Measurement scope

The host was an AMD Ryzen 7 5800X (8 cores, 16 logical CPUs), Windows, MSVC
19.50 Release. Base is `11f5859a4be5`, candidate is `356b7457398a`, the
same-file unused-helper layout control is `e149fc05acee`, and mx-llama.cpp is
`5542318e748c`. Exact revisions, binary versions/hashes, model hashes and
compiler commands are preserved in [plan.json](plan.json),
[cli-plan.json](cli-plan.json), [build-identities.json](build-identities.json),
[build-commands.json](build-commands.json) and
[reference-identity.json](reference-identity.json). The recorded post-run
[identity-after.json](identity-after.json) retains all 15 input identities.
Packaging did not reread those executable, DLL or model payloads.

The common harness uses identical 215 prompt and 32 teacher-forced continuation
token IDs, six CPU threads, F16 K/V, context 512 and ubatch 128; flash attention
is disabled. Loading, tokenization and sampling are outside timing; logits
materialization is timed for both runtimes. Each invocation retains run 0 as
the prospectively excluded warmup and measures run 1. Eight rotating/reversing
orders cover four arms and four models.

The CLI uses its synthetic LCG prompt, pp215 and independent decode32 from
empty history, six CPU threads, F16 K/V, default ubatch 512 and model context.
It performs an internal warmup per phase and one measured run. Six balanced
orders cover three arms and four models. CLI has no matched mx arm, and its
printed rates are rounded to 0.01 tok/s.

## Common harness results

Rates below are medians in tokens/second. Paired percentages are geometric
means of same-round throughput ratios, not ratios of marginal medians.
Positive percentages mean higher throughput. Each common pair has eight
blocks; each CLI pair has six. Full per-round ratios, faster/slower counts,
minima/maxima and activity subsets remain in the linked JSON and CSV.

| Model | Phase | base tok/s | candidate tok/s | layout tok/s | mx tok/s | candidate/base paired % | layout/base paired % | candidate/mx paired % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B-Q8_0.gguf | pp215 | 538.59 | 539.08 | 521.46 | 286.84 | +1.224 | -2.729 | +90.439 |
| Qwen3-0.6B-Q8_0.gguf | tg32 | 53.31 | 55.52 | 54.29 | 49.47 | +2.392 | -0.494 | +11.006 |
| Qwen3-0.6B-Q4_0.gguf | pp215 | 460.74 | 483.38 | 472.95 | 440.11 | +2.367 | +3.549 | +4.033 |
| Qwen3-0.6B-Q4_0.gguf | tg32 | 80.11 | 78.19 | 80.44 | 77.38 | -1.439 | +2.206 | +0.828 |
| Qwen3-0.6B-Q5_K_M.gguf | pp215 | 439.75 | 433.69 | 439.89 | 256.27 | -0.292 | +0.858 | +72.699 |
| Qwen3-0.6B-Q5_K_M.gguf | tg32 | 69.36 | 70.78 | 71.57 | 68.14 | +0.427 | +1.996 | +4.692 |
| Qwen3-8B-Q8_0.gguf | pp215 | 40.61 | 41.18 | 40.43 | 21.91 | +1.527 | -0.730 | +88.518 |
| Qwen3-8B-Q8_0.gguf | tg32 | 4.58 | 4.60 | 4.55 | 4.60 | +0.316 | -1.734 | +0.126 |

Candidate/mx paired means are positive for these eight cells, including
+0.126% for 8B decode. That narrow observed margin does not establish a stable
advantage or parity for every model, quant, prompt, backend or hardware.

## CLI results

| Model | Phase | base tok/s | candidate tok/s | layout tok/s | candidate/base paired % | layout/base paired % |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B-Q8_0.gguf | pp215 | 557.86 | 538.41 | 539.70 | -1.187 | -1.317 |
| Qwen3-0.6B-Q8_0.gguf | tg32 | 56.70 | 57.47 | 56.50 | +1.197 | +0.411 |
| Qwen3-0.6B-Q4_0.gguf | pp215 | 511.76 | 504.98 | 513.53 | -0.946 | +1.862 |
| Qwen3-0.6B-Q4_0.gguf | tg32 | 87.94 | 86.53 | 84.88 | -1.325 | -1.196 |
| Qwen3-0.6B-Q5_K_M.gguf | pp215 | 487.06 | 482.96 | 493.04 | -1.873 | +1.390 |
| Qwen3-0.6B-Q5_K_M.gguf | tg32 | 77.38 | 71.24 | 73.23 | -6.568 | -4.110 |
| Qwen3-8B-Q8_0.gguf | pp215 | 40.74 | 40.61 | 40.69 | -0.867 | -0.404 |
| Qwen3-8B-Q8_0.gguf | tg32 | 4.64 | 4.62 | 4.67 | +0.828 | +2.146 |

## Activity and retention

All 128 common and 72 CLI records have successful terminal workload results,
monitor exits, structural records and bracket coverage of the full invocation.
Observed activity flags affect 126/128 common and 70/72 CLI invocations. All
200 invocations also have some unknown process CPU deltas, so unflagged must
not be read as quiet. Every paired block includes at least one flagged arm;
flagged-pair summaries therefore equal the complete summaries. System disk
and GPU counters describe whole-system activity, including the workload;
they are not proof of unrelated work. Monitoring spans loading and warmup as
well as measured execution. No inference about causality follows from flags
alone, and no block is excluded because of them.

The common harness retains printed top-1 and a rounded final-logit sum as a
limited output witness. It does not preserve full logits or establish an
independent numerical HF oracle. The strict chat gate and external HF
correctness records remain separate evidence. Recorded readiness requires
the strict nine-call gate and stopped competing own work; those guard and
preflight records are included here.

## Archive inventory and verification

| Archive | Members | Original bytes | Compressed bytes |
|---|---:|---:|---:|
| matrix.tar.xz | 642 | 179610539 | 3937284 |
| cli-matrix.tar.xz | 362 | 91086774 | 1994324 |
| Total | 1004 | 270697313 | 5931608 |

The archives use standard USTAR plus XZ preset 6, with sorted relative file
paths and deterministic ownership, mode and timestamp metadata. A verified
ZIP trial was superseded because solid compression reduced its 25493600 bytes
to 5931608 bytes; the trial sizes/hashes remain in the manifest.

Every member was decompressed and its SHA256 matched to the original source
bytes during packaging. The 84 copied supporting files (2394685 bytes) were
also compared byte for byte by SHA256. [package-manifest.json](package-manifest.json)
records every member, size and hash. It includes no executable, object, DLL or
model payload. Original local artifacts were left unchanged. The original
plans/freezes/drivers are retained, including preparatory `executed: false`
fields; successful terminal matrix records establish later completion.
The old unguarded drivers are frozen provenance. Actual execution used
`run_matrix_guarded.py` and `run_cli_matrix_guarded.py`.

This directory's `.gitattributes` disables text conversion to preserve exact
recorded bytes. Original absolute paths inside records are provenance and
must remain unchanged. The analyzer reads unpacked files relative to its
location; it does not launch recorded commands or open model paths.

## Reproduce the evidence audit

Copy this directory to a scratch location and run from that copy. Python's
standard library is sufficient; no model, compiler, GPU or inference is needed.

```text
python verify_package.py
python -m tarfile -e matrix.tar.xz .
python -m tarfile -e cli-matrix.tar.xz .
python review/analyze_final.py
python review/analyze_final.py --cli
```

The first command rechecks archive hashes, every decompressed member and all
copied supporting files. Extraction restores `matrix/` and `cli-matrix/`.
The analyzer requires exact planned order/counts, successful terminal records,
matching commands and raw stdout agreement. It writes new timestamped audit
directories rather than changing saved summaries. Saved absolute command
strings are compared to the frozen plan, without executing them. Reproduced
paths and audit timestamps will differ; measured cells and paired arithmetic
must agree. For the retained generated-fixture checks, run
`python review/test_analyze_final.py`; this creates new small fixture files.
Do not run guarded workload drivers merely to audit existing evidence.

Saved complete audits:

- [Common summary](review/common-analysis-20260925T022107018446Z/summary.md), [JSON](review/common-analysis-20260925T022107018446Z/summary.json), [paired blocks](review/common-analysis-20260925T022107018446Z/blocks.csv).
- [CLI summary](review/cli-analysis-20260925T023611951658Z/summary.md), [JSON](review/cli-analysis-20260925T023611951658Z/summary.json), [paired blocks](review/cli-analysis-20260925T023611951658Z/blocks.csv).

The analyzer gives no automatic cutoff, noise, adoption or merge verdict.
