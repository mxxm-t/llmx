# llmx — Development Status

Living tracker. This is the disposable file: notes here are only useful while
work is in progress. When a feature ships, delete its block below and mark the
row `Done` in the table. Read it together with `docs/ROADMAP.md` (the stable
plan) and `docs/ARCHITECTURE.md` (the layer rules) — STATUS carries where each
feature currently stands right now.

## Status table

| Feature                                  | Status   |
|------------------------------------------|----------|
| Layered restructure                      | Done     |
| Build config (config.hpp + CMake + build.bat) | Done |
| Test suite (roundtrip / perf / tokenizer)| Done     |
| Perf `bench` command                     | Done     |
| CPU backend optimization                 | Done     |
| More quant formats (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K read) | Done |
| More model architectures (Llama, ...)    | Planned  |
| More formats (safetensors, ...)          | Planned  |
| Device execution model (GPU prerequisite) | Planned |
| GPU backends (ROCm first, Vulkan portability) | Planned |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |
| Correctness baseline vs HF reference     | In Progress |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | In Progress |
| CPU attention in backend (ROADMAP #4a)  | In Progress |
| CPU row streaming / parallel prefill   | In Progress |
| CPU attention value accumulation      | In Progress |
| CPU grouped projections              | In Progress |
| CPU Q8 scale / load scheduling       | In Progress |
| GitHub CPU CI                          | Done     |
| HF integration (pull + Hub formats)      | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

## Active feature blocks

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### F32 embedding/matrix inference

- **Goal:** load and run F32 embeddings and matrices in dense Qwen3, with
  external HF numerical validation and measured CPU performance (ROADMAP #8).
- **Done:** direct F32 rows, float-aligned tensor blobs, deterministic HF
  full-logit/NLL fixtures (tied/untied, odd widths, batches, threads). Windows
  and Linux full suites pass; UBSan passes and catches the pre-fix alignment
  fault. New sanitizer CI job passes workflow lint. All 311 real Qwen3-0.6B
  F32 tensors verified against original HF weights; all real HF checks pass.
  Copy/direct controls match matrix hashes and all logits over a 1,943-token
  prompt plus 32 greedy tokens. Measurements/provenance are in ASSETS.
- **Left:** close the measured F32 CPU gap versus public mx-llama.cpp
  `5542318e74`, then merge and observe the expanded five-job hosted CI.
  The latest matched F32 results are in the Q8 scale/load block below and
  `docs/benchmarks/q8-scale-load-cpu-20260919.json`; external decode parity
  remains unproven. Model loading is excluded and each process warms up.
- **Findings:** activation tiling and a fully spinning worker pool did not
  establish a win. Profiling instead identifies scalar attention as roughly
  170-180 ms of prefill. Backend attention now improves prefill by 24.8%
  against its interleaved control and passes HF bounds, while changing
  summation order. Remaining work is the CPU performance gap, not numerical
  validation of this attention implementation.
  `tools/compare_cpu.cpp` / `.py` preserve the matched measurement procedure;
  commands and limitations are in ASSETS.
- **Gotchas:** do not treat success on quantized weights widened to F32 as
  parity with the original HF weights. Validate tied/untied output and prefill.

### CPU attention in backend

- **Goal:** move causal GQA attention out of the model and into the backend
  (ROADMAP #4a), sharing the decode and prefill implementation and improving
  CPU throughput with measured, numerically bounded vectorization.
- **Done:** `Backend::attention` now handles both forward paths, with CPU-owned
  score scratch, causal GQA, AVX2 dots and value accumulation, and scalar tails.
  Deleted duplicate model-layer attention loops. Regenerated the tiny HF
  fixture at head width 10 to cover vector tails. Windows and Linux full
  suites pass, including real Q8/Q4 HF checks; Linux UBSan synthetic suite
  passes. Tiny maximum HF logit error is 6.2e-7. A future-token attention
  mutant fails with error 0.03787. Real F32 excerpt/window NLL differs from HF
  by at most 4.5e-6. At 1,943 prompt tokens plus 32 generated tokens, greedy
  output matches HF and all 5,013,888 logits are within 0.001 (max 0.0001252).
  Eight interleaved rounds give F32 prefill 286.84 -> 357.88 tok/s (+24.8%);
  decode 13.34 -> 13.57 tok/s (small change with overlapping ranges).
  Q8 synthetic guardrails pass; matmul 123.07 -> 121.80 GFLOPS, prefill
  4828 -> 4800 and decode 4522 -> 4875 tok/s, with overlapping ranges.
- **Left:** close the remaining external CPU floor gap, then merge and run
  hosted CI. See grouped projections below for the latest measurements;
  its F32 decode still trails mx.
  Profiling after vectorization finds prefill attention around 60-67 ms and
  decode attention around 97-100 ms; matrix operations now dominate decode.
  Sequential F32 row streaming and parallel batched elementwise work are
  implemented on the next feature branch; see its validation block below.
- **Gotchas:** SIMD dot reductions change summation order. A matching top token
  or logit sum is not a numerical gate. This step does not implement device
  buffers, resident activations or async execution, which remain prerequisites
  for GPU backends.

### CPU row streaming / parallel prefill

- **Goal:** close the F32 CPU performance gap without changing weights or
  weakening the HF numerical gate (ROADMAP #8).
- **Done:** contiguous single-row F32 decode and parallel batched
  norm/RoPE/SiLU work, guarded to keep fewer than two rows per worker serial.
  Final eight interleaved rounds: prefill 371.65 -> 383.06 tok/s (+3.1%,
  overlapping ranges), decode 13.87 -> 14.68 (+5.8%, disjoint ranges).
  Same-run mx is 398.97 / 14.89: both floors remain open (-4.0% / -1.4%).
  Windows and Linux full suites pass with required Q8/Q4 HF fixtures; UBSan
  synthetic suite passes. Real F32 HF logits/excerpt/chunked NLL pass.
  Long-prompt full logits have maximum error 0.00012636 under 0.001, with
  all greedy IDs matching HF. Q8 step guard passes with overlapping ranges.
  The batched guard caught tiny-prompt scheduling regressions; after the
  serial guard, small-batch ranges overlap the control and B=64 retains a
  17% latency improvement. ASSETS and the row-scheduling JSON contain all
  initial/final samples and validation scope.
- **Left:** close both remaining external gaps before merge and hosted CI.
  Instrumented pool profiling finds 26-28 ms after worker callbacks finish
  during 32 decode steps. Bounded completion polling did not establish a win
  over its atomic-only control (eight rounds); no pool change was adopted.
  Full spinning was already rejected. Query scheduling remains scratch-only;
  the value-accumulation kernel below is the next validated checkpoint.
- **Gotchas:** single-row dots change reduction order. Printed sums alone
  are not the numerical gate. Initial unconditional scheduling appeared above
  mx for prefill, but final guarded measurements did not; use the final run.
  The legacy bench "prefill" is repeated step(), not batched prefill, so keep
  the separate batched regression guard. Reader alignment was not adopted.

### CPU attention value accumulation

- **Goal:** reduce CPU attention output loads/stores by retaining value-sum
  lanes in SIMD registers, while preserving per-lane summation order (#4a/#8).
- **Done:** normalized coefficients and register value sums implemented, with
  32-lane blocks, eight-lane remainders and scalar tails. Expanded the tiny HF
  fixture to head width 42, deriving its tensor shapes and HF config together.
  SIMD and forced scalar-value branches pass at maximum HF error 0.00000070
  under 0.00002; a missing SIMD-block output mutant fails at 0.19888665.
  Windows/Linux full suites pass with required real Q8/Q4 fixtures; Linux
  UBSan synthetic suite passes. Real F32 HF logits and excerpt/window NLL pass.
  All 5,013,888 logits over 1,943 prompt plus 32 greedy tokens are byte-identical
  to c4436fd; HF maximum error 0.00012636 under 0.001, greedy IDs 32/32 exact.
  Q8 step and batched guards show overlapping control/candidate ranges.
  Final eight interleaved rounds:

  | Mean tok/s | Control c4436fd | Value kernel | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Prefill | 384.86 | 397.63 | 397.94 | -0.08% |
  | Decode | 14.56 | 14.57 | 14.89 | -2.10% |

  Prefill improves 3.32% with narrowly overlapping ranges; decode is unchanged
  within noise. ASSETS and `benchmarks/attention-values-cpu-20260919.json`
  retain all samples, hashes, validation logs and reproduction harnesses.
- **Left:** close the external performance gap before merge/hosted CI.
  Instrumented profiling attributes about 2,073 of 2,187 ms decode to matmul,
  including 528 ms in the output projection; attention is about 87 ms.
  Investigate matrix operations next, using matched controls and HF gates.
- **Gotchas:** means close to mx are not proof of parity. This is an interactive
  workstation and no outliers were discarded. Query scheduling and polling
  remain scratch-only. The value kernel keeps sequence order per lane; byte
  identity is established for the recorded Windows long-prompt case, not all
  inputs or compilers. Forced scalar values do not prove no-AVX ISA support.

### CPU grouped projections

- **Goal:** reduce worker-pool dispatches for Q/K/V and FFN gate/up projections
  sharing activations, preserving float arithmetic (ROADMAP #4a/#8).
- **Done:** direct grouped decode through existing F32/Q8_0/Q4_K row kernels.
  Other types, batches and small jobs use the sequential fallback. No TLS,
  nested dispatch or activation conversion. Added CTest coverage to CI.
  Windows/Linux full suites pass with required real Q8/Q4 HF fixtures;
  real F32 HF and UBSan synthetic/backend checks pass. Backend tests cover
  540 cases and 141,750 outputs per platform; a missing-row mutant fails.
  Instrumented Q8 run observes 3,584 groups avoiding 5,376 dispatches across
  warmup plus measured sequences. Four excerpt/window NLL cases per model
  match the previous runtime exactly. All 5,013,888 long-prompt logits per
  model (F32 and Q8) are byte-identical to the previous runtime. F32 remains
  within the independent HF bound, with identical greedy continuation.
  Final eight interleaved rounds after a discarded warmup round:

  | Mean tok/s | Previous 5a9518c | Grouped | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8_0 prefill | 418.59 | 419.06 | 267.76 | +56.51% |
  | Q8_0 decode | 43.92 | 45.18 | 48.34 | -6.55% |
  | F32 prefill | 397.41 | 394.25 | 390.07 | +1.07% |
  | F32 decode | 14.47 | 14.64 | 14.86 | -1.49% |

  Q8 decode mean/median improve 2.86%/3.19%; ranges overlap due to one slower
  candidate sample, retained in the result. F32 prefill/decode changes are
  within overlapping ranges. Synthetic matmul mean 129.57 -> 127.64 GFLOPS
  (median 129.78 -> 129.51, ranges overlap); step-based prefill/decode improve
  4998/4928 -> 6861/6921 tok/s. Batched guards retain overlapping ranges except
  the faster six-thread single-token case. Raw samples, hashes, exact-output
  gates and reproduction sources are in ASSETS and
  `benchmarks/grouped-projections-cpu-20260919.json`.
  The follow-up real 8B Q8_0 diagnostic also preserves all 5,013,888 logits
  on its matched 215+32-token history. Three measured rounds, after warmup:

  | 8B mean tok/s | Previous | Grouped | mx |
  |---|---:|---:|---:|
  | Prefill | 29.12 | 29.25 | 21.20 |
  | Decode | 4.31 | 4.33 | 4.49 |

  Control/group ranges overlap; this does not establish a small speedup.
  Decode remains below mx. Exact-vector equality is against previous llmx,
  not an independent 8B HF baseline. Evidence and the full-vector control are
  in `benchmarks/grouped-projections-8b-20260919.json`.
- **Left:** close the external decode floors before merge and hosted CI.
  Native Q8 scale/load scheduling is the next bounded investigation below.
- **Gotchas:** exact equality is scoped to tested inputs/platform, not a
  full-corpus or maximum-context proof. The legacy bench prefill uses step();
  batched prefill has a separate guard. All outliers retained; do not pool
  absolute rates from separate sessions. Candidate is validated but unmerged.

### CPU Q8 scale / load scheduling

- **Goal:** reduce native Q8 decode instruction overhead while preserving
  float activations, per-lane accumulation order and exact weight scales (#8).
- **Done:** selected direct memory half broadcast plus direct byte-load sign
  extension. Assembly confirms the intended instructions; scalar fallback and
  accumulator order are unchanged. Feature specialization did not establish a
  further gain and is excluded. Exhaustive finite-half scale/signed-weight
  regression passes, and a wrong-half-offset mutant fails. Windows/Linux full
  suites with required Q8/Q4 HF fixtures, real F32 HF checks, backend CTest and
  UBSan synthetic/backend checks pass. Instrumentation confirms real Q8 use.
  Recorded F32/Q8 long vectors and excerpt/window NLL are exactly unchanged
  from `b6a890f`; the same-weight 8B vector comparison also passes.

  | Eight-round mean tok/s | Before b6a890f | Candidate | mx |
  |---|---:|---:|---:|
  | Qwen3-0.6B Q8 prefill | 422.99 | 418.43 | 273.51 |
  | Qwen3-0.6B Q8 decode | 45.30 | 46.54 | 48.35 |
  | Qwen3-0.6B F32 prefill | 398.17 | 395.23 | 394.92 |
  | Qwen3-0.6B F32 decode | 14.70 | 14.77 | 14.98 |

  Q8 prefill and F32 ranges overlap. Q8 decode improves but remains below mx.
  Root CLI and production comparator code hashes match their validated arms.
  The larger-model diagnostic also improves decode but does not establish
  external parity:

  | Qwen3-8B Q8 mean tok/s | Before | Candidate | mx |
  |---|---:|---:|---:|
  | Prefill | 28.97 | 29.04 | 21.13 |
  | Decode | 4.29 | 4.45 | 4.49 |

  The short synthetic guard's threaded single-token regression does not
  retain disjoint ranges in the longer follow-up. Paired changes run in both
  directions; the higher candidate mean remains visible:

  | Synthetic prefill mean ms | Before | Candidate |
  |---|---:|---:|
  | 1 thread, 1 token | 0.16995 | 0.12580 |
  | 6 threads, 1 token | 0.15864 | 0.16167 |

  ASSETS and `benchmarks/q8-scale-load-cpu-20260919.json` retain all samples,
  controls, hashes, HF/platform logs, prototype patches and reproduction code.
- **Left:** close the external decode floors before merge and hosted CI.
  Continue matrix-operation work with matched model measurements; retain the
  threaded tiny-prompt guard because zero slowdown has not been established.
- **Gotchas:** retain F16 subnormal/negative-scale behavior and scalar fallback;
  do not multiply activations by scales instead, which changes rounding.
  Exact equality is scoped to recorded inputs/platforms. The 8B equality check
  is against previous llmx, not an independent HF 8B reference. All samples
  are retained; no external parity claim.

### Correctness baseline vs HF reference

- **Goal:** give the suite an external ground truth. Correctness is measured
  against the HF reference, never against llmx itself (`docs/ROADMAP.md` #8).
- **Done:**
  - `tools/gen_baseline.py` emits golden fixtures using `tokenizers` +
    `huggingface_hub` only (no torch, no transformers); output committed to
    `tests/data/baseline_tokenizer.json`.
  - `tests/baseline.py` compares llmx against the committed goldens and is
    wired into `tests/run_tests.py`. It SKIPS when no fixture model is on disk,
    so the rest of the suite still runs anywhere.
  - Tokenizer parity: **20/20 cases match pinned Qwen/Qwen3-0.6B.**
  - HF fp32 top-10 logit goldens for six prompts landed in `1c2e102`.
    The baseline checks top-1, top-5 set overlap and a magnitude bound on
    Qwen3-0.6B Q8_0 and mixed Q4_0 fixtures. Both passed all six prompts in
    that commit; the mixed fixture catches the f16 subnormal regression.
  - Local golden generation works in an isolated environment with
    numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2.
  - Pinned HF fp32 PPL golden for a 247-token wikitext excerpt: exact token
    IDs/count and finite NLL/PPL checks, with per-quant absolute NLL bounds.
    HF PPL 28.7974; Q8_0 28.8371 (NLL delta 0.001374 <= 0.01); mixed Q4_0
    32.8463 (delta 0.131554 <= 0.16). Both llmx arms repeated identically at
    printed precision. Generator, provenance and bounds are in `docs/ASSETS.md`.
    Eight injected bad-output cases were rejected; build and full suite pass.
  - Nine bugs found and fixed via this path, all of which survived a green
    suite: attention missing 1/sqrt(head_dim); temperature cancelling in the
    sampler; RoPE read past context_length; the GPT-2 whitespace guard that
    could never fire; attention width hardcoded to n_embd; tied embeddings
    unsupported; Windows argv delivered in the ANSI codepage so any non-ASCII
    prompt was mangled before llmx saw it; and the pretokenizer implementing
    the GPT-2 regex instead of the Qwen2/Qwen3 one; the byte encoder incorrectly
    including soft-hyphen byte 0xAD in its printable set. Four new HF cases
    reject the preserved old encoder. Whole-wikitext file input now tokenizes
    298,938 tokens and scores the requested window limit successfully.
- **Left:**
  - Per-layer activation and full-corpus PPL goldens, plus long-context validation.
    Continuous and chunked excerpt gates are implemented, with an explicit
    disjoint-window scoring policy (`docs/USAGE.md`). They are not full-corpus
    coverage. Two-token windows show large quantized/HF deviations even under
    the unchanged old scorer; diagnostics and bounds are in `docs/ASSETS.md`.
    The existing ranking gate does not bound full-vector numerical error.
  - Tolerance bands: F32 vs reference tight, Q8_0 vs reference needs a
    quantization-appropriate bound.
- **Gotchas:**
  - Round-trip and synthetic tests alone are not an external correctness gate;
    preserve the HF tokenizer/logit checks and extend their coverage.
  - The 8B hides bugs the 0.6B exposes: `n_head * head_dim == n_embd` holds for
    Qwen3-8B (32*128 == 4096) and fails for 0.6B/1.7B/4B.
  - torch is a fixture-GENERATION dependency only, never needed to run the
    suite and never at runtime.
  - F32 embedding/matrix support and a tight HF numerical gate are implemented
    on the active feature branch. Its external performance gate remains open;
    see the F32 block above before merging.
  - Generation's legacy reasoning filter searches `thinking_start/end`, not
    Qwen3's actual `<think>` / `</think>` markers. Its docs now state that limit.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Latest checkpoint:** native Q8 scale/load scheduling preserves all tested
  output bits and improves Q8 decode on both measured model sizes. External
  Q8 and F32 decode floors remain open. See its block and evidence JSON above
  for current model measurements, HF gates and the tiny-prompt limitation.
- **Latest investigation (no runtime change):** paired native Q8 rows regress;
  direct pointer increments and explicit row-kernel inlining do not establish
  a decode gain. Exact row/tail/fallback checks pass. Assembly confirms shared
  activation loads without inner-loop spills, removal of native-loop address
  multiplication, and removal of row calls in the respective prototypes.

  | Separate Qwen3-0.6B Q8 studies, mean decode tok/s | Control | Prototype | mx |
  |---|---:|---:|---:|
  | Paired rows | 46.00 | 43.81 | 48.31 |
  | Pointer increments, longer run | 46.22 | 45.86 | 48.22 |
  | Explicit inlining | 46.56 | 46.04 | 48.46 |
  | Inlining plus pointers | 46.56 | 46.38 | 48.46 |

  No prototype is adopted. Do not pool absolute rates across these sessions.
  ASSETS and `benchmarks/q8-row-instructions-20260919.json` retain patches,
  assembly, exact checks, complete samples, code hashes and reproduction.
  These scratch prototypes did not enter the full HF/platform adoption gate.
- **Previous investigation:** AVX2 integer dots with vectorized activation packing
  were tested at 8-bit and 16-bit precision. Q16 improves matched mean decode
  by 2.79% on 0.6B and 4.06% on 8B, but still trails mx by 5.12% / 1.21%.
  Both variants pass existing Windows HF fixture bounds; Q16 stays much closer
  to the current float path. Independent packing/product controls pass,
  including signed weight extremes, half subnormals and fallback cases.
  Across four excerpt/window cases, Q16's maximum absolute NLL change versus
  current llmx is 0.00003155. Across 5,013,888 logits on a 1,943+32-token
  forced-HF continuation, maximum change is 0.002213; maximum and RMS error
  against HF are slightly lower in this case. This is a nonzero precision
  change, not a lossless result. Full corpus, independent 8B HF, maximum
  context and cross-platform validation remain open. No candidate was adopted;
  all patches, samples and numerical controls are preserved in
  `benchmarks/q8-integer-activation-20260919.json` and ASSETS.
- **Earlier investigations:** paired F32 decode rows did not improve throughput;
  packed F32 prefill variants regressed. None was adopted. Exact patches,
  samples and diagnostics are in ASSETS and the paired-decode/packed-prefill
  benchmark JSON files. These experiments used the `5a9518c` runtime.
  Profiling scalar exponentials finds only 7.45 ms SiLU wall time and 9.42 ms
  summed softmax worker time during 2,237.50 ms decode; these are not the main
  remaining cost. The summed worker measurement is not wall time.
  The preceding matched Q8 comparison covers both the HF fixture and real 8B
  model against pinned public mx `5542318e74`: three alternating pairs,
  215+32 pinned tokens, six threads, ubatch 128 and F32 KV.

  | Q8_0 model / phase | llmx mean tok/s | mx mean tok/s | Gap |
  |---|---:|---:|---:|
  | Qwen3-0.6B prefill | 380.50 | 265.35 | +43.39% |
  | Qwen3-0.6B decode | 42.73 | 46.81 | -8.71% |
  | Qwen3-8B prefill | 25.94 | 20.07 | +29.27% |
  | Qwen3-8B decode | 4.10 | 4.39 | -6.64% |

  Prefill exceeds the reference on both models, but decode remains below it;
  both comparisons have disjoint arm ranges in each phase. Raw samples,
  hashes, flags and scope are in `benchmarks/q8-external-floor-20260919.json`.
  Source inspection confirms mx uses quantized Q8 activations and integer
  dots, while llmx retains float activations. Any analogous optimization
  needs a measured numerical cost bound before adoption.
- **Historical stand-in comparison (different conditions; not pooled):**
  Qwen3-8B Q8_0, 343-token wikitext prompt, -t 16, this workstation. Reference
  is the CPU AVX2 llama.cpp shipped with LM Studio, stock `llama-server`, same
  machine, so no rig time was used.
    - pp   llmx 37.23 / 37.86   llama.cpp 37.70 / 37.34   -> parity
    - tg   llmx  3.91           llama.cpp  4.99 / 5.02    -> 22% under
  These are the original matched measurements. Later Q8_0 decode commits
  report 4.12 tok/s with independent accumulators (`54ea063`) and 4.24/4.18
  with F16C (`0c12570`); these are not a new matched mx-llama.cpp comparison.
  Separate decode/prefill thread flags landed in `dada6c0`.
- **How prefill got there, 3.89 -> 37.5 tok/s (9.6x), each step A/B measured:**
  - persistent worker pool instead of spawning threads per call (decode -23%)
  - attention through `Backend::parallel_for` instead of its own threads
  - batched prefill: matrix-matrix instead of one token at a time (3.89 -> 13.2)
  - dequantize each weight row once per batch, not once per column (-> 15.3)
  - four independent accumulators in the f32 dot (-> 17.0)
  - fused 4-row kernel sharing one activation load (-> 24.0)
  - two activation columns per four rows, 0.75 loads/FMA (-> 33.7)
  - three activation columns per four rows, 0.58 loads/FMA (-> 37.5)
- **The lesson worth keeping:** the kernel was LOAD bound, not FMA bound. Each
  naive dot needs 2 loads per FMA and Zen3 sustains about 2 loads/cycle against
  2 FMAs/cycle, so it ran at half of FMA peak no matter how the batch was
  blocked. Every win after the first came from raising the FMA:load ratio.
- **Left:**
  - Close the measured Q8 and F32 decode gaps against mx-llama.cpp itself.
    Q8_0 decode is memory-bandwidth bound (early thread scaling was flat:
    4/8/16 threads give 3.83/4.13/3.90 tok/s) at about 32 GB/s against
    llama.cpp's 37, so the ceiling on the whole gap is bandwidth efficiency.
    Allocation churn, layout fragmentation and software prefetch are measured
    NULL. mmap has not been established as a throughput improvement.
  - The integer-activation study above found a modest decode gain despite the
    earlier bandwidth prediction, but did not meet the external floor and
    introduces a precision change. Keep its measured cost visible if revisited.
  - Investigate the remaining matrix-operation cost after grouped dispatch
    and native Q8 scale/load scheduling (latest results in its block above).
    The larger-model diagnostic did not close the gap. Keep float arithmetic
    unchanged where possible and apply independent HF gates to any change.
- **Gotchas:**
  - Synthetic `bench` throughput does not establish real-model speed. Small
    projections may stay serial depending on thread count. Grouping improves
    the six-thread synthetic case, but the real-model external floor still
    fails; use the matched model measurements.
  - Do not tune the row block as a byte budget. Measured worse at every size
    (64/128/196/256 KB gave 22.37/22.04/23.68/21.12 against 24.04 for a flat
    4); the knee follows the fused kernel width, so it is `DOT_ROWS`.
  - ubatch barely matters once the kernel is right, and 343 vs 512 on a
    343-token prompt is the SAME computation - do not read noise as signal.

Nothing else is in flight. When you start a feature, open a block above
before writing code — see `AGENTS.md` → "Starting a feature".
