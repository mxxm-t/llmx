# llmx - Development Status

Living tracker. This is the disposable file: notes here are only useful while
work is in progress. When a feature ships, delete its block below and mark the
row `Done` in the table. Read it together with `docs/ROADMAP.md` (the stable
plan) and `docs/ARCHITECTURE.md` (the layer rules) - STATUS carries where each
feature currently stands right now.

## Status table

**Resumed after explicit user authorization following the reboot.** Branch
`fix/gguf-tensor-extents`, isolated from placement checkpoint `5859762` and
based on validated JSON checkpoint `a61c414` and
validated production runtime `bf122fd`.
Pinned reference tooling is validated; broader 8B HF coverage remains open.
The TUI watcher on **8181** is
restarted with its saved cursor. Root `llmx.exe` was updated to validated streaming
build `9cfe43f` after the user reported buffered 8B chat output; the old executable
is backed up in `%TEMP%/llmx-live-generation/root-before-streaming.exe`.
CLI thread corrections are validated on the active branch; benchmark
comparator code is unchanged. Automatic build identification and the README/ASCII cleanup are validated. Live generation/loading progress is validated on Windows and Linux. Gitea is reachable
again; feature checkpoints may be backed up there, with performance gates still
required before merging to main. GitHub publication is authorized after the
requirements pass. GitHub stays main-only; feature branches go to Gitea.

**Documentation review (2026-09-20):** rechecked all 25 tracked Markdown files
against current source, CLI, CMake/CI and recorded evidence. Independent source
and documentation review found no correctness blocker. The optional 8B consumer
now records frozen bounds, strict parsing, independent fixtures and rejection
tests without adding default CI model downloads. Its Windows and Linux real-model gates
pass 37/37 each, and both full 11-component required-HF suites pass. Linux
uses diagnostic-only perf timing. A separate isolated current-runtime timing
session confirms both Q8 decode means/medians remain below mx; prefill exceeds
mx on both models. Q16 scratch passed its own integer arithmetic/witness
checks, then failed the unchanged native double-dot accuracy contract on both
compilers. It is rejected; HF cost and performance comparisons were not run.
The exact Q8 epilogue candidate passed native/HF and exact-output checks, but
its complete nine-round timing found no decode gain. It is rejected and remains
outside production: both decode means/medians trail current and mx, with no
candidate wins over mx in either model. All 25 Markdown files are reviewed at
this rejection checkpoint.
Subsequent Q8 split-storage and exact decode SwiGLU callback studies pass their
scoped numerical checks but miss their frozen synthetic timing screens. Both
remain outside production; their complete samples and reviews are archived below.
The all-phase placement candidate remains rejected. A separate prefill-only
placement screen passes, with its independent timing audit and full 25-file
Markdown review complete. Production integration and external gates remain open.
The separate observer-free placement screen also passes; its post-run audit
and full Markdown review are complete. CPU-local per-operation placement is implemented only in scratch and rejected
by its completed screen because small-model decode regresses. Its independent
timing audit and full 25-file Markdown checkpoint review are complete.
The subsequent synchronous callback integration also fails its frozen screen:
8B decode mean is 0.574% below disabled prototype, beyond the 0.5% limit.
That original screen stopped adoption; the user tradeoff clarification below
reopens assessment. Independent timing/archive audit and the full
25-file Markdown checkpoint review are complete.
The reopened candidate now passes fresh Windows native, lifecycle, active HF
logit and 0.6B long-continuation checks, with machine activity recording
implemented and exercised. New monitored timing remains open; see its block.
Prior prefill/HF evidence remains archived.
Historical measurements and the root streaming executable remain unchanged.
External performance requirements still block main/GitHub publication.

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
| JSON syntax and Unicode validation      | In Progress |
| GGUF reader size and tensor extent validation | In Progress |
| Device execution model (GPU prerequisite) | Planned |
| GPU backends (ROCm first, Vulkan portability) | Planned |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |
| Chat follow-up cache validation          | In Progress |
| Correctness baseline vs HF reference     | In Progress |
| Pinned HF reference generation           | In Progress |
| Optional Qwen3-8B HF consumer             | In Progress |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Matched CPU comparison thread selection | In Progress |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | In Progress |
| CPU attention in backend (ROADMAP #4a)  | In Progress |
| CPU row streaming / parallel prefill   | In Progress |
| CPU attention value accumulation      | In Progress |
| CPU grouped projections              | In Progress |
| CPU Q8 scale / load scheduling       | In Progress |
| Head-major CPU KV storage             | In Progress |
| CPU worker exception safety           | In Progress |
| CPU worker cost profile                 | In Progress |
| CPU ordered prefill reductions          | In Progress |
| CLI thread settings                    | In Progress |
| Automatic build identification          | In Progress |
| Live generation and loading progress     | In Progress |
| GitHub CPU CI                          | Done     |
| HF integration (pull + Hub formats)      | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

**Performance tradeoff clarification (2026-09-20):** the user asks that a
large gain in one phase not be automatically rejected for a minor loss in
another. Assess and report the complete workload tradeoff, retaining HF
correctness and explicit matched mx comparisons. Prior frozen-screen results
remain historical facts; the prefill-placement decision is being reassessed
under this clarified preference. Fresh correctness checks pass below, but no
candidate has yet been adopted or timed under that assessment.

**Machine contention requirement (2026-09-20):** the user requires checking
whether other demanding work is using the PC during measurements. Upcoming
placement validation must capture background process CPU use and system
CPU/disk/GPU activity before and throughout every arm. Use predefined
contamination criteria, preserve affected matched blocks as inconclusive and
repeat complete blocks after contention clears. Existing small differences
cannot retroactively be certified contention-free without the needed evidence.
Windows recording, controlled-load detection and the comparison driver's
preflight screen are now exercised. Completed model timing and assessment of
observer effects remain open.

## Active feature blocks

### GGUF reader size and tensor extent validation

- **Goal:** reject malformed lengths, dimensions, arithmetic overflow and tensor
  extents before allocating payload storage or reporting loading progress;
  honor the file's declared alignment. This closes the documented format-layer
  error-handling gap and supports future Hub/sharded-format work.
- **Done:** bounded reads, checked size arithmetic and subtraction-based file
  ranges reject malformed input before payload allocation/progress. Reader and
  writer honor positive uint32 alignments divisible by eight, including 24.
  Quantized row widths must contain whole blocks. Array depth is limited to
  256 and tensor rank to four; valid empty tensors retain mathematical size zero.
  Isolated branch `fix/gguf-tensor-extents` starts at `5859762`; placement
  experiments and fixed binaries remain separate.
- **Done:** 122 GGUF cases pass on Windows/Linux: independently constructed
  fixtures plus the writer-alignment round trip. All nine
  native tests and all eleven required-HF suite components pass on both.
  Current-reader ASan+UBSan passes both format/progress tests. The initial full
  Windows build lacked the MSVC include environment; that failure is retained
  and the complete run passes after initializing vcvars64. No source workaround.
- **Done:** the pinned 8B file loads completely with byte-identical `info`
  output against the prior validated control. All six fresh short HF cases
  match top-1 and top-5 overlap 5/5. This is not a new 8B NLL/long-context gate.
  All jobs are terminal. All 25 project Markdown files reviewed and stale
  `info`, quantized-row and loader validation descriptions corrected.
- **Left:** merge with the runtime stack once its separate performance gate
  passes; the root executable and main/GitHub remain unchanged. No inference
  hot path changed and no new performance result is claimed. Evidence:
  [`gguf-reader-validation-20260920.json`](benchmarks/gguf-reader-validation-20260920.json).
- **Gotchas:** model configuration, tensor names/shapes required by a model,
  token IDs, JSON tensor-dimension arithmetic, writer validation and future
  request recovery remain separate validation work. Do not
  claim that file-extent checks make arbitrary models executable. Preserve
  nested GGUF arrays within a documented depth limit and valid non-power-of-two
  alignments that are multiples of eight.

### Prefill placement reassessment with machine activity monitoring

- **Goal:** complete the reopened whole-prefill placement assessment against
  production and matched mx, including HF/lossless and short follow-ups.
- **Done:** preserved the historical candidate and its failed original screen;
  JSON checkpoint `a61c414` passes Windows/Linux correctness suites.
- **Done:** fresh scratch control and default-enabled candidate retain current
  JSON/CLI fixes and pass 8/8 Windows CTests each. The candidate full required-HF
  suite passes 11/11; timings are diagnostic only. Rebuilt callback contracts
  pass 17 cases/4,626 exact values and Windows lifecycle checks pass 27
  cases/1,176 exact values. All 28 active-prefill HF logit cases pass: ten tiny
  F32, six real 0.6B Q8, six real 0.6B F32 and six 8B Q8. Every candidate
  process verifies six applies/restores on distinct target CPUs, no errors or
  leftover restriction, and exact printed logits against current production.
  Tiny F32 maximum HF error is 6.991024018e-7 against the unchanged 2e-5 bound;
  all 18 real-model cases match top-1 and all five top-5 IDs.
- **Done:** the active 1,943-token prefill plus 32 forced continuation steps
  passes for real 0.6B F32 and Q8. Each model's 33 full vectors (5,013,888
  finite floats per arm) are byte-identical to current production, with exact
  per-target NLL. Both candidate processes verify six applies/restores and
  zero decode setters. All 36 HF/prompt/continuation inputs match the earlier
  committed evidence before execution. F32 maximum HF logit error is
  0.000126362 <= 0.001; absolute mean continuation NLL differences are
  0.000000645211 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8).
- **Done:** `tools/monitor_windows.py` records timestamped system CPU/disk/GPU
  counters and per-process CPU deltas keyed by PID plus creation time. The
  24-sample controlled-load check detects the known CPU process at a median
  99.995% of one logical CPU; recorder CPU is 0.53125 s over 24.01487 s,
  including initialization. Intervals and query errors remain in the log.
- **Left:** complete prospectively planned matched comparisons and check observer
  effects once a quiet measurement window is available. The driver records
  machine activity before and throughout every matched block; no model timing
  has yet passed its preflight screen.
- **Done:** fresh Linux candidate pass-through build, native 8/8 and full
  required-HF suite 11/11 pass with unchanged snapshot source. Matched mx
  primary/one-token/nine-token continuation harness builds against the pinned
  CPU DLLs. The activity evaluator passes 20 synthetic/known-load checks.
- **In progress:** prospective three-arm comparison plan uses one outer
  warmup round plus eight measured rounds per model/workload, with complete
  matched-block replacement for detected contention (at most two replacements).
  The first preflight defers before any model launch: accessible unrelated CPU
  is 23-44% of one logical CPU and physical disk busy is 29-41% across eleven
  samples. Raw logs are retained; all 56 frozen identities recheck unchanged.
  No performance result or candidate rejection follows. Small observer effects
  remain an explicit unresolved limit.
- **Done:** fresh enabled-candidate optional 8B regression passes 37/37 checks
  against unchanged HF fixture bounds. This includes serial-step NLL coverage;
  active placement is established by the separate callback witnesses, not by
  the serial NLL path. Its timings are not performance evidence.
- **Done:** primary, one-token and nine-token follow-up correctness passes
  for both Q8 models: 12 processes, six byte-identical full-vector pairs and
  911,616 finite floats. All 20 callbacks independently verify six distinct
  physical cores and full restoration (120 applies and 120 restores total),
  with zero decode setters, placement errors or leftover restrictions.
  These fixed-token prefix/suffix checks do not replace interactive chat tests.
- **Checkpoint:** all 25 project Markdown files reviewed; stale status wording
  corrected. Supplemental commands, sources and results are archived in
  [`prefill-contention-followup-20260920.json`](benchmarks/prefill-contention-followup-20260920.json).
  All correctness jobs are terminal. Timing remains deferred; preserve the
  first preflight and use a fresh output directory for the next comparison.
- **Latest preflight:** the next attempt also defers before launching a model:
  sustained unrelated CPU exceeds the unchanged screen. Recorder exit is zero;
  all 58 identities recheck unchanged. A minimally changed runner now requires
  a fresh `--output` path, preserving both attempts and the original runner.
  [Second preflight evidence](benchmarks/prefill-preflight-02-20260920.json).
  Turning off the only recorder would remove during-run contention evidence;
  an extra-recorder sensitivity diagnostic would not prove zero-recorder cost.
  Keep that limit explicit and assess whether such a diagnostic is useful after
  the matched comparison, without creating more measurement infrastructure now.
- **Gotchas:** activity monitoring is evidence, not proof of no interference.
  Keep observer overhead and unavailable counters explicit; no automatic
  adoption or retroactive noise claim follows from the policy change.
  The local check retains 124-126 inaccessible processes as unknown; system
  counters remain available. Controlled disk/GPU saturation and benchmark
  timing perturbation are untested. This continuation check is not full-corpus,
  maximum-context or 8B long-context HF coverage. Production runtime source
  remains unchanged; the candidate is still scratch-only.
  Full 25-file Markdown review and saved checkpoint evidence:
  [`prefill-reassessment-correctness-20260920.json`](benchmarks/prefill-reassessment-correctness-20260920.json).

### JSON validation, Unicode decoding and output escaping

- **Goal:** fix current model-description JSON handling and the documented
  ROADMAP #9b prerequisite for HF metadata/safetensors, without dependencies.
- **Done:** strict number/literal/escape syntax, classic-locale finite-double
  conversion, validated UTF-8 and UTF-16 surrogate-pair decoding, and a
  256-container nesting bound. Value/API and duplicate get-first behavior stay
  unchanged. A small string quoting helper now escapes both dequantize path
  and tensor-name fields, including quotes, backslashes and control bytes.
- **Done:** final Windows/Linux builds and native tests pass 8/8; both full
  required-HF Python suites pass 11/11. Native JSON has 1,033 checks, including
  independent expected bytes, numeric/locale limits and malformed input.
  Linux ASan+UBSan passes the same checks. Actual Q8/Q4 CLI round trips retain
  BMP/supplementary Unicode, quotes, backslashes, newline and source paths.
  Old parser/CLI regressions fail as expected. The initial new-reader Windows
  suite exposed the existing output-escaping bug; its failed result remains
  archived alongside successful final runs. A missing temporary Linux build
  directory caused one later launch to fail before executing any build/test;
  final Linux validation uses a persistent owned build directory.
- **Left:** include the validated fix with the runtime stack when its external
  requirements pass; hosted macOS execution remains unobserved locally.
  All project Markdown is reviewed at this checkpoint. Evidence:
  [`json-validation-20260920.json`](benchmarks/json-validation-20260920.json).
- **Gotchas:** 1,033 counts checks, not independent input documents. Numeric
  storage is double, not exact arbitrary-precision integers. Lone surrogates,
  invalid UTF-8, overflow and nonzero underflow to zero are rejected by policy.
  Quoting expects valid UTF-8; tensor-schema/range/extent arithmetic and general
  filesystem-path handling remain separate. No inference arithmetic changes.
  Suite timings are diagnostic; existing HF/mx merge requirements stay open.

### Native CPU decode sampling (diagnostic complete)

- **Goal:** identify sampled native instruction/function locations during current
  8B decode without adding timers to runtime source. This is a separate
  diagnostic after caller attribution proved no production saving.
- **Done:** optimized PDB harness builds against 17 unchanged runtime files.
  Session 65416 completes the discarded pair, then stops on xperf's unquoted
  commas in C++ symbol fields. A separate parser recovers the saved trace;
  the initial failure and all 41 original identities remain unchanged.
  Independent recovery review pins 51 additional identities. Session 21434
  runs exactly the remaining six invocations and terminates with exit 0.
  All eight planned invocations and 16 finite full vectors pass; vectors are
  byte-identical to the historical unchanged-runtime reference.
- **Done:** independent audit rehashes both manifests, checks collector ownership,
  Running/Stopped states, fixed ready/done holds, and reconstructs all target
  samples directly from saved exports: 39,209 discarded, then 40,115, 39,733
  and 39,571 measured. Every trace has zero lost events/buffers and 100%
  named application-symbol coverage. All unknown/system samples are retained.
  Of 119,419 measured samples, 116,184 (97.291051%) land in `dot_row_impl`.

| 8B phase, mean elapsed ms | Plain | Sampled | Sampled/plain change |
|---|---:|---:|---:|
| Prefill, 215 tokens | 7060.177800 | 7022.348833 | -0.535808% |
| Decode, 32 tokens | 6970.996900 | 7034.000533 | +0.903797% |

- **Done:** exact-binary mapping verifies 345 instructions and all 1,476 code
  bytes against the frozen executable. Actual image bases and PE exception
  ranges resolve every dot sample to an instruction start; padding is excluded.
  All non-dot and caller/other-thread counts remain in the evidence.
- **Left:** external HF/mx requirements remain open. This diagnostic selects no
  production optimization and does not reopen rejected studies. Main/GitHub
  remain unchanged. Full checkpoint evidence and Markdown review are archived
  in [`cpu-native-decode-sampling-20260920.json`](benchmarks/cpu-native-decode-sampling-20260920.json).
- **Gotchas:** three measured pairs, six threads, ubatch 128 and F32 KV; one
  internal warmup per process. Decode paired elapsed changes are +2.915900%,
  -0.611112% and +0.453332%. These combine profiler/handshake/state effects
  and variability, not pure tool overhead. Samples include outside-clock gate
  activity and identify execution locations, not hardware-stall causes or
  elapsed per-operation costs. This exact-vector check supplements earlier
  HF evidence; it does not replace the independent correctness gate.

### Current decode caller-cost attribution (diagnostic complete)

- **Goal:** separate caller work from the previously mixed dispatch residual
  before selecting another optimization. No placement study is reopened.
- **Done:** unchanged production, legacy worker probes and added caller probes
  complete all 12 fixed 8B invocations in session 21493, exit 0. Six threads,
  ubatch 128, F32 KV, 215 prompt plus 32 forced tokens; one outer triplet and
  each process's first iteration are prospective warmups. All 24 full vectors
  are finite and byte-identical to the pinned unchanged-runtime vector. All 93
  frozen identities recheck unchanged. Each of four attributed traces has 32
  steps, 5,792 dispatches and 37,152 caller events, with zero accounting gap.
  Builds, native grouped/error checks, active-probe error/reuse, synthetic
  accounting and independent malformed-trace checks pass. Native Windows
  sampling resolves 6,205 of 6,221 samples to two named C++ test functions
  using local PDBs; this earlier capability checkpoint does not sample a model
  or establish a memory-stall diagnosis. The later model study is above.
- **Finding:** observed means are 217.602291 ms/token in dispatch, 1.972685
  model-side, 0.080984 backend caller work, 0.015842 harness and 0.067954
  explicit caller-observer brackets, plus 0.000014 outer timer fringe.
  Model-side includes serial backend norms/RoPE. SwiGLU is 1.157311 ms/token
  within the model-side total. Attributed decode elapsed is 2.140880% above
  plain and 1.863856% above legacy spans; paired differences against spans
  change sign. Perturbation/variation is comparable to or larger than the
  individual residual regions. Explicit brackets do not capture all observer
  effects, and standalone calibration is not subtracted from model timings.
- **Decision:** no recoverable production saving is proved. Stop the
  outside-kernel optimization direction without probe tuning, a repeat timing
  screen or runtime implementation. Keep all samples. Native function sampling
  is exercised in the separate model study above; none was selected or run
  in this earlier caller-attribution study. External HF/mx gates remain open.
- **Left:** the existing external decode performance gaps remain open. Any
  separate native model sampling needs its own fixed plan, output checks and
  unprofiled control. No production implementation follows this diagnostic.
  Independent terminal audit rechecks all 93 identities, 24 vectors and
  148,608 caller events, reproducing every integer time partition.
- **Gotchas:** the old 2.010823 ms/token residual is historical motivation,
  not a current serial-cost estimate. Region times are instrumented intervals,
  not production savings; exact final vectors are not an external HF gate.
  Production source, tests, build/CI and root executable remain unchanged.
  Full evidence: [`cpu-decode-caller-cost-20260920.json`](benchmarks/cpu-decode-caller-cost-20260920.json).

### Synchronous CPU prefill placement (old screen failed; reassessment reopened)

- **Goal:** test one backend-owned synchronous callback around the complete
  prefill, after the smaller operation-local candidate failed. Keep platform
  details below Model and include setup, callback and checked cleanup costs.
- **Done:** all 36 primary invocations completed with exit 0. Both prefill cases
  pass against fresh production and disabled prototype, with 5/5 wins against
  each. However, 8B decode mean loses 0.573915% against disabled, beyond the
  frozen 0.5% limit. Reject this integration and stop placement adoption.
  Under that original rule, follow-up timing and conditional HF/mx runs did
  not proceed. The user has since requested evaluating large gains against
  minor losses: adoption is reopened for further validation, with this failed
  screen and all samples preserved.
  Windows native checks pass 7/7; callback/tiny-model contracts pass 17 cases
  and 4,626 exact values, plus 27 Windows placement cases and 1,176 exact
  values. Linux unchanged native assertions and callback/no-op guards pass.
  Both one-thread perf smoke floors pass (placement inactive).
  All 18 separate real-model witnesses pass: exact fresh-production vectors
  for initial and one/nine-token continuation cases on both models. Enabled
  processes have 12 applies/restores for primary and 24 for follow-ups across
  two iterations; disabled witnesses and decode have zero setters. Production
  has no instrumentation. All 36 timed vectors match their witness controls.
  Preflight passes 103 checks; 216 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-prefill-callback-20260920.json`.
  Independent timing/archive audit and the full 25-file Markdown checkpoint
  review are complete.
- **Left:** fresh active-path HF/lossless checks pass in the reassessment block;
  short-follow-up and fresh matched mx validation remain open under the user's
  clarified tradeoff preference. Retain the
  bounded callback and existing mapping. Report uncertainty for small decode
  differences and gains across phases; the old cutoff failure stays recorded.
  Production scheduling remains unchanged until adoption is validated.
- **Gotchas:** scratch Backend/Model/CPU only; production source, tests, build
  files, public API and root executable are unchanged. Successful placement
  uses one apply and one checked restore pool dispatch. Partial application
  restores all changed participants before unbound fallback. Cleanup retries
  once but reports its first error; persistent refusal does not establish safe
  pool/model reuse. Initial scope is six supported Windows participants,
  mechanically quant-independent, with no wider performance claim. Source
  guards reject nesting/thread changes but do not make Model concurrent.
  Timed policy permits fallback; witness counters prove their own processes.
  Exact vectors are not an independent HF gate. The failed primary screen
  originally prevented the planned short-follow-up performance runs, so their
  timing remains unverified despite passing numerical/activation witnesses.

| Model / phase | Production mean tok/s | Disabled mean tok/s | Enabled mean tok/s | Mean vs production | Mean vs disabled | Median vs disabled |
|---|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 474.695274 | 472.415155 | 548.326819 | +15.511% | +16.069% | +17.878% |
| 0.6b / tg | 48.835786 | 50.004903 | 50.342125 | +3.084% | +0.674% | +0.222% |
| 8b / pp | 30.392742 | 30.891701 | 41.452967 | +36.391% | +34.188% | +34.854% |
| 8b / tg | 4.628661 | 4.649671 | 4.622986 | -0.123% | -0.574% | +0.004% |

### CPU-local batched-matmul placement (scratch candidate screened out)

- **Goal:** keep placement inside the CPU backend and existing callbacks,
  avoiding a generic Backend phase API or model-level platform code.
- **Done:** all 24 invocations completed with exit 0. Prefill meets the frozen
  >=5% mean/median and 4/5-win screen in both models, but 0.6B decode loses
  2.62% mean and 2.72% median throughput, beyond the 0.5% limit. Reject this
  candidate; no sample removal, rerun or inherited phase-wide result.
  All final vectors match within model and against prior unchanged controls.
  Fresh Windows lifecycle gate passes 41 cases and 11,849 exact values;
  native backend-group passes unchanged 540 cases per arm. Linux native/no-op
  checks and both one-thread perf smoke floors pass. Four real-model witness
  processes verify exact prior outputs, candidate apply/restore counts of
  4,704 for 0.6B and 6,048 for 8B across two iterations, and zero decode/control
  setters. All 130 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-matmul-placement-20260920.json`.
  Independent timing audit and full 25-file Markdown checkpoint review pass.
- **Left:** the later whole-prefill callback above is the preferred reopened
  assessment under the user's clarified tradeoff preference. This per-operation
  arm and its failure remain archived. Production is unchanged; its prepared
  active-path HF plan was not run.
- **Gotchas:** six-thread Q8_0 nbatch > 1 scope only; other paths fall back.
  Every eligible operation queries topology and each nonempty callback pays
  apply/checked restore. Timing compares enabled/disabled prototype policy,
  with inert scaffolding in control; it is not the exact production baseline.
  Instrumented witness activation covers its own processes, not every timed
  callback. Allocation failure before dispatch propagates; dual body/restore
  failure can replace the task payload with cleanup failure. Persistent restore
  refusal is reported and only the synthetic test performs manual rescue.
  No production source, public flag, Backend API or root executable changed.

| Model / phase | Disabled mean tok/s | Enabled mean tok/s | Mean change | Median change | Enabled wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 471.700388 | 504.994000 | +7.06% | +8.73% | 4/5 |
| 0.6b / tg | 49.597335 | 48.299841 | -2.62% | -2.72% | 1/5 |
| 8b / pp | 30.040204 | 40.211011 | +33.86% | +34.11% | 5/5 |
| 8b / tg | 4.502460 | 4.575066 | +1.61% | +1.60% | 5/5 |

### Prefill placement without diagnostic observers (scratch screening passed)

- **Goal:** establish whether the prefill placement benefit survives removal
  of shared observer dispatches before considering production integration.
- **Done:** all 24 invocations completed with exit 0, and the frozen screen
  passes. Both prefill means/medians improve at least 5%, with 5/5 wins per
  model; decode means/medians stay within the 0.5% regression limit.
  Each model's 12 finite final vectors match exactly, with internal warmup
  identity also checked. All 48 source/build/check identities were frozen.
  The new parser accepts seven archived lifecycle records and rejects 34
  corruptions; six rule-boundary checks pass. Unchanged helper lifecycle
  evidence is reused explicitly, not claimed as fresh execution.
  Independent timing audit and review of all 25 Markdown files are complete.
  Full results are in `docs/benchmarks/cpu-prefill-observer-free-20260920.json`.
- **Left:** the CPU-local batched-matmul prototype above is rejected by its
  separate performance screen. The synchronous callback is implemented only
  in scratch and failed its old primary cutoff. Its adoption decision is now
  reopened above; the production Backend API remains unchanged.
  Independent HF and fresh matched mx gates remain required.
- **Gotchas:** scheduler mode constructs no placement Session. Candidate
  construction/apply/prefill/verify/checked restoration are timed; decode
  follows immediately. Stored pre-decode witnesses are serialized afterward,
  and the already-restored destructor performs no pool/affinity call.
  Formatting and inert object disposal are outside timing. There is no
  post-decode mask observation. Six threads, one prompt, Windows only;
  this remains scratch evidence, not production or HF/mx acceptance.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 479.228537 | 529.632147 | +10.52% | +11.02% | 5/5 |
| 0.6b / tg | 49.518222 | 49.633298 | +0.23% | +0.10% | 3/5 |
| 8b / pp | 29.027817 | 40.985134 | +41.19% | +40.29% | 5/5 |
| 8b / tg | 4.581855 | 4.600768 | +0.41% | +0.79% | 4/5 |

### Prefill-only CPU placement (scratch screening passed)

- **Goal:** test the measured prefill opportunity while restoring normal
  scheduling before decode, including recurring placement costs in timing.
- **Done:** the fixed 24-invocation comparison completed with exit 0 and passes
  its frozen screen. Both models improve prefill mean/median by at least 5%
  with 5/5 wins, and decode mean/median stay within the 0.5% regression limit.
  All 12 saved final vectors per model match exactly; each internal warmup
  also matches its measured iteration. Lifecycle tests pass 3,636 exact Q8 value
  comparisons, fresh iterations, exception cleanup and same-pool reuse.
  The parser accepts seven lifecycle records and rejects fifteen corruptions.
  Independent preflight passes; 46 identities were frozen before timing.
  Independent post-run audit and all 25 Markdown checkpoint reviews pass.
  Complete results are in `docs/benchmarks/cpu-prefill-placement-20260920.json`.
- **Left:** the separate observer-free study above passes its frozen screen.
  Broader thread/phase lifecycles, independent HF correctness and
  fresh matched mx performance remain required before adoption.
- **Gotchas:** this is a screening pass, not production readiness. Restoring
  masks does not reset cache, boost or scheduling state. The full candidate
  Session lifecycle, including report serialization and destruction, is timed
  as prefill. Common observation work outside both clocks can influence decode.
  The result covers one 215-token prompt, 32 forced decode steps, six threads,
  ubatch 128, F32 KV and Windows Ryzen 7 5800X only. No HF/mx floor follows.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 467.836623 | 535.466490 | +14.46% | +11.93% | 5/5 |
| 0.6b / tg | 49.293600 | 49.418118 | +0.25% | +0.39% | 3/5 |
| 8b / pp | 29.521998 | 40.948992 | +38.71% | +40.36% | 5/5 |
| 8b / tg | 4.594650 | 4.623768 | +0.63% | -0.30% | 3/5 |

### Explicit CPU worker placement (all-phase candidate screened out)

- **Goal:** compare scheduler-selected placement with six workers on six
  distinct queried physical cores, including caller worker zero, without
  changing kernels or production options.
- **Done:** helper success/unbound, caller/worker failure cleanup, destructor
  restoration and 1,818 exact grouped Q8 value checks pass. Independent
  preflight verifies source fidelity, masks, actual CPU witnesses and cleanup.
  All 24 real-model invocations pass. Fixed-arm before/after snapshots verify
  logical CPUs 0/2/4/6/8/10 from the same queried topology and process mask;
  every original thread mask is restored. Saved finite final vectors match
  byte-for-byte within each model, across both arms and all invocations.

  | Model / phase | Scheduler mean tok/s | Fixed mean tok/s | Mean change | Median change | Fixed wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 479.071849 | 541.704220 | +13.07% | +11.58% | 5/5 |
  | 0.6B decode | 49.314603 | 47.022156 | -4.65% | -5.70% | 0/5 |
  | 8B prefill | 30.183415 | 40.902861 | +35.51% | +35.72% | 5/5 |
  | 8B decode | 4.564495 | 4.649338 | +1.86% | +2.00% | 5/5 |

  The frozen rule requires at least 0.5% higher decode mean and median with
  4/5 wins in both models, and no prefill mean/median regression above 3%.
  Small-model decode fails; the all-phase candidate stays outside production.
  All samples are retained; no new mx or independent HF gate was run.
- **Left:** retain scheduler-selected production behavior. The separate
  prefill-only study above passed its own frozen screen with transition costs
  included; this all-phase candidate remains rejected.
  Matched external decode requirements remain open.
- **Gotchas:** topology and allowed mask are queried in each owned child;
  adjacent CPU IDs are observed, not assumed. Core placement includes serial
  caller work and warmup first-touch, and cannot isolate migration/SMT effects.
  Final-position vectors do not prove full-corpus/deep-context correctness.
  No mapping search, production flag, API or cross-platform affinity promise.
  Evidence: `benchmarks/cpu-worker-placement-20260920.json`.

### Exact decode SwiGLU callback fusion (screened out)

- **Goal:** measure unchanged SwiGLU inside the existing equal-row Q8 gate/up
  worker callback, preserving all buffers, arithmetic and projection order.
- **Done:** MSVC and GCC each pass 1,440 matrix arm comparisons, 126,990
  finite bit comparisons and 27,054 nonfinite classifications. A separate
  instrumented copy witnesses 16,920 rows exactly once; an arithmetic mutant
  compiles and is rejected numerically. Actual unchanged grouped projections
  plus an independent literal SwiGLU expression supply the synthetic reference.
  Assembly confirms original dots and scalar exp/divide/multiply order.
  All 40 fixed timing samples complete with exact gate/up/output checks.

  | Synthetic shape | Serial mean ms/call | Fused mean ms/call | Mean change | Median change | Fused wins |
  |---|---:|---:|---:|---:|---:|
  | Small gate/up + SwiGLU | 0.089504 | 0.085051 | -4.98% | -5.91% | 7/9 |
  | Large gate/up + SwiGLU | 2.442995 | 2.383465 | -2.44% | +0.25% | 7/9 |

  The frozen screen requires lower mean and median in both shapes with at
  least 6/9 wins each, plus at least 2% lower mean and median in one shape.
  The large median fails; no model integration follows. All samples retained.
- **Left:** retain current production behavior; do not weaken the prospective
  screen or repeat this experiment merely to obtain a passing sample.
  Independent HF and matched external decode requirements remain open.
- **Gotchas:** both timed arms use one concrete Q8 helper, not the generic
  production dispatch. Repeated synthetic weights can remain cached. This is
  neither a model slowdown finding nor an HF/external performance result.
  Numerical checks use the standard floating environment and compare within
  each compiler; NaN payload and alternate rounding-mode identity are unclaimed.
  Evidence: `benchmarks/q8-swiglu-fusion-screening-20260920.json`.

### Native Q8 bounded inner-loop follow-up (rejected)

- **Goal:** test one ordinary two-trip loop with local accumulators, preserving
  one copy of the native block body and every arithmetic operation.
- **Done:** MSVC emits two block bodies without accumulator stack traffic.
  Independent instruction review finds 55 F16C-path instructions per pair
  versus 54 for two control iterations. Arithmetic and branch counts are
  unchanged; the extra instruction reloads the feature flag inside the loop.
  No second-block work moves before the first block's final FMA/exit check.
  This fails the useful-scheduling gate; no numerical/model/timing runs follow.
- **Left:** stop this unrolling exploration. Historical and current-source 8B
  whole-operation diagnostics are complete below. Within-group attribution
  remains unresolved; use the matrix-cost findings to select the next study.
- **Gotchas:** instruction counts are not micro-op counts or measured latency.
  No hints, forced inlining, feature specialization or duplicated kernel bodies
  were used. Production remains unchanged. Evidence:
  `benchmarks/q8-bounded-inner-loop-rejection-20260920.json`.

### Native Q8 block scheduling study (rejected at codegen gate)

- **Goal:** expose two consecutive native Q8 blocks to compiler scheduling while
  preserving every weight product, four FMA chains and the original reduction.
- **Done:** isolated control/candidate comparators build with MSVC. Independent
  scalar/control oracles pass 612,267 finite bit checks and 1,939 nonfinite
  classifications on both MSVC and GCC. A repeated-block mutant compiles and
  fails numerically. The two-block lambda preserves the original HADD epilogue,
  block order and odd tail, but MSVC emits eight unconditional 32-byte
  accumulator stack stores per two-block iteration, including the F16C path.
  This fails the planned no-spill codegen gate; the formulation is rejected.
- **Left:** no adoption or model timing for this formulation. Continue the
  external decode performance work from the unchanged production kernel.
- **Gotchas:** this is a codegen rejection, not a measured slowdown. The oracle
  is scoped numerical evidence, not an HF/model validation claim. Prior
  pointer, feature-specialization and integer-unroll studies remain distinct.
  Evidence is in
  `benchmarks/q8-native-block-scheduling-rejection-20260920.json`.

### Exact Q8 horizontal reduction study (rejected)

- **Goal:** reduce Q8 row-dot epilogue cost while preserving all float products,
  FMA chains and contributing addition order.
- **Done:** the scratch shuffle/add sequence emits the intended instructions.
  Native CTest passes 7/7 and required-HF suites pass 11/11 for control/candidate
  on Windows/Linux. MSVC scalar checks pass 612,267 finite bit comparisons and
  1,939 nonfinite classifications; a wrong shuffle is rejected. Windows repeated
  control and candidate match 5,013,888 long-prompt logits and four serial NLL
  cases, within HF bounds. Windows candidate 8B HF checks pass 37/37.
  All 60 fixed timing processes completed: one discarded warmup plus nine
  measured rounds per model/arm, with every sample retained.

  | Mean tok/s | Current | Candidate | mx | Candidate/current | Candidate/mx |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 473.669 | 453.120 | 277.206 | -4.34% | +63.46% |
  | 0.6B decode | 48.966 | 48.723 | 49.969 | -0.50% | -2.49% |
  | 8B prefill | 29.646 | 29.936 | 21.348 | +0.98% | +40.23% |
  | 8B decode | 4.579 | 4.549 | 4.610 | -0.65% | -1.31% |

  Both decode medians also trail current/mx. Candidate wins only 3/9 decode
  pairs against current and 0/9 against mx on each model. The reduction is
  rejected; production stays at `bf122fd`, and the native regression proposal
  remains unapplied. Validation evidence is in
  `docs/benchmarks/q8-exact-reduction-validation-20260920.json`; completed timing
  is in `docs/benchmarks/q8-exact-reduction-performance-20260920.json`.
- **Left:** close the separate production decode floor. Current control means
  in this session trail mx by 2.01% (0.6B) and 0.67% (8B); do not pool this with
  earlier sessions or revive the rejected epilogue based on selected samples.
- **Gotchas:** the low 0.6B candidate prefill sample (357.111 tok/s) stays in the
  mean. Its median is 471.589 versus current 476.152; do not describe the mean
  gap as a universal causal slowdown. Correctness alone does not justify this
  performance change. NaN payloads and alternate rounding modes remain unclaimed.


### Optional Qwen3-8B HF consumer

- **Goal:** compare the verified local Q8_0 GGUF against the independent pinned
  8B HF tokenizer, logit and PPL goldens without adding large CI downloads.
- **Done:** original HF generation and provenance evidence are committed in
  `bf122fd`. Before any llmx 8B comparison, declare exact tokenizer/input IDs,
  six exact top-1 matches and top-5 set overlap 5/5; all ten printed logits must
  be finite, sorted, unique-token and within absolute magnitude 100. NLL delta
  limits are 0.01 continuous and 0.02 for each windowed case, prospectively
  reusing the existing Q8 quality budget, not calibrated from 8B results.
  Consumer implementation and independent review are complete. Both platforms pass
  all 37 checks: 20 tokenizer cases, six prompt-ID/ranking pairs, PPL IDs and
  four NLL cases. Largest NLL delta is 0.002185355, under its 0.02 limit. The
  full required-HF suite passes 11/11 on each platform, including consumer
  rejection tests (Linux perf is diagnostic only). The official pinned GGUF is
  downloaded and hash-verified in the Linux cache documented in ASSETS. The
  successful Linux gate used an identical staged copy, removed only after its
  tests finished. Interrupted mounted-file results and separate intervention
  metadata are preserved. Copy plus verification took 59.17 seconds; direct
  download plus verification took 486.61 seconds. These are operational I/O
  observations, not inference measurements. Prefer an existing verified copy
  when faster and reuse the completed Linux cache. Windows original untouched.
  All 25 Markdown files reviewed and stale performance wording corrected.
  Evidence: `benchmarks/hf-8b-validation-20260920.json`.
- **Left:** merge the validated consumer with the runtime stack after its
  external performance gates pass. Keep failures with original bounds;
  investigate rather than relaxing thresholds to fit observations.
- **Gotchas:** exact original GGUF conversion revision is undocumented. The
  official model-family link and file hash do not prove identical source
  weights. Fixed excerpt/rank checks do not cover full corpus or all logits.

### CPU ordered prefill reductions

- **Goal:** determine whether the four-row/three-column prefill kernel's
  addressable accumulator array adds avoidable stack traffic or reduction
  overhead, without changing per-lane FMA or final addition order.
- **Done:** explicit ordered reductions match 1,824 scalar-FMA outputs across
  dimension tails and unaligned inputs. MSVC assembly removes most epilogue
  accumulator stack traffic; the FMA loops do not spill in either arm. The
  function grows from 1,119 to 2,931 bytes. Nine alternating rounds improve
  mean Q8/F32 prefill by 4.59%/7.01%, winning 8/9 and 9/9 pairs. A separate
  nine-round Q8 follow-up repeats the prefill gain (+6.25%, 9/9 pairs).
  Q8 decode changes from -1.81% to +0.71% versus control between sessions;
  no stable decode regression or universal external parity is established.
  All control/candidate final-vector hashes match. Windows/Linux native checks
  pass 7/7 and required-HF suites pass 10/10. Linux real-F32 tokenizer/logit/NLL
  checks also pass. Both platforms check each arm against the same scalar-FMA
  oracle; an MSVC mutant swapping final additions is rejected. Fresh long
  comparisons have 5,013,888 byte-identical logits per F32/Q8 model. F32 maximum
  HF error is 0.000126362 <= 0.001; prefilled continuation NLL deltas are
  0.000000645 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8). Four separate serial
  NLL cases equal control and pass existing HF bounds. All 25 Markdown files
  reviewed; evidence: `benchmarks/prefill-ordered-reduction-20260920.json`.
- **Left:** retain this validated feature checkpoint on Gitea; merge with the
  runtime stack only when broader external performance requirements pass.
- **Gotchas:** no reassociation or new activation quantization. A synthetic
  gain alone does not establish the external floor. Keep worker implementation
  unchanged and isolate timing from other builds/tests and user inference.

### Pinned HF reference generation

- **Goal:** reuse independent HF tokenizer/logit/PPL generation for explicitly
  pinned models, keeping larger-model fixtures separate from the existing suite.
- **Done:** model/revision/output and associated GGUF labels are explicit;
  numerical loaders share pinned CPU float32 eager execution. Alternate models
  require separate output and cannot overwrite the default fixture directory.
  Windows full required-HF suite passes 10/10 components against the unchanged
  validated 9cfe43f executable; Linux generator safeguards pass 2/2 tests.
  Two offline generations from cached 0.6B HF weights reproduce every numerical
  field. Against committed goldens: 20 tokenizer cases, six top-10 ID lists,
  247 PPL token IDs, four NLL values and synthetic F32 JSON match exactly.
  Rounded top-10 logit values differ by at most 0.0001. Existing fixtures,
  acceptance bounds and default CI model downloads are unchanged. Evidence:
  `benchmarks/hf-reference-tools-20260919.json`.
  Parallel rig work generated actual 8B tokenizer/logit/PPL references from
  verified original `Qwen/Qwen3-8B` at pinned `b968826d9c46dd6066d109eabc6255188de91218`.
  All three modes pass with CPU FP32 eager execution. Measured memory reaches
  the owned container's 40 GiB cap including file cache (3,098 limit events,
  zero OOM/kill). Evidence: `benchmarks/hf-8b-reference-20260920.json`.
- **Left:** merge the tooling with the runtime stack after its external gates
  pass. The separate 8B consumer passes Windows and Linux checks under
  predeclared bounds. Official GGUF metadata links
  the base model and matches the local Q8 digest, but exact original conversion
  revision is undocumented. Default CI downloads remain unchanged.
- **Gotchas:** do not overwrite small-model goldens with another model or expand
  default CI downloads. Tooling support alone is not an 8B correctness result.
  The tooling-only checkpoint changed no hot path and ran after worker timing.
  The parallel 8B HF work ran on the separate rig; it is not a performance gate.

### Separate Q8 scale/payload storage study (screened out)

- **Goal:** test a scratch storage view with original half-scale bytes separate
  from contiguous 32-byte weight blocks, preserving all arithmetic and values.
- **Done:** MSVC and GCC each pass 612,267 finite bit comparisons, 1,939
  nonfinite classifications and 141 packing cases. Grouped/standalone witnesses
  confirm the split dot runs; a corrupted-scale mutant compiles then fails.
  Native assembly preserves the FMA chains/HADD with no loop accumulator spills.
  Fixed synthetic timing completes all six shapes and 96 samples, with exact
  output checks throughout. No samples are dropped.

  | Shape | Original mean ms/call | Split mean ms/call | Mean change | Median change | Split wins |
  |---|---:|---:|---:|---:|---:|
  | Small up | 0.047025 | 0.047276 | +0.53% | -0.51% | 4/7 |
  | Small gate/up | 0.078415 | 0.078916 | +0.64% | +0.69% | 1/7 |
  | Small down | 0.042013 | 0.041720 | -0.70% | -1.84% | 3/7 |
  | Large up | 0.865281 | 0.844761 | -2.37% | -6.07% | 5/7 |
  | Large gate/up | 2.406585 | 2.359071 | -1.97% | -2.83% | 6/7 |
  | Large down | 0.859395 | 0.850048 | -1.09% | -0.20% | 3/7 |

  The frozen advancement rule requires at least 3% mean and median improvement
  with at least 5/7 wins in every large case, and no small-case mean/median
  regression above 3%. It fails; no model integration follows this study.
- **Left:** retain production storage. Revisit only with a distinct hypothesis;
  do not weaken the screening rule or infer a real-model improvement from these
  short cached matrix measurements. External decode requirements remain open.
- **Gotchas:** packing takes 8.96/19.04/9.09 ms in the three large cases and
  adds another weight-sized retained allocation in this diagnostic. Logical
  retained bytes exclude allocator overhead and an additional unpack-validation
  temporary. No peak-RSS, model-loading, HF or external-performance claim.
  Evidence: `benchmarks/q8-split-storage-screening-20260920.json`.

### Current 8B worker-span diagnostic

- **Goal:** measure instrumentation impact on the current 8B runtime and
  attribute decode dispatch intervals to operations before selecting a change.
- **Done:** current source snapshots, plain/instrumented builds and fault check
  pass. All eight fixed processes pass: one discarded outer warmup pair and
  three alternating measured pairs. Every process internally warms up. Saved
  151,936-float final vectors match byte-for-byte across all eight invocations;
  internal warmups use FNV64. Each instrumented trace has 865 prefill and 5,792
  decode records with valid ordered timestamps and no overflow.

  | Phase time, ms | Plain mean | Plain median | Spans mean | Spans median | Mean change |
  |---|---:|---:|---:|---:|---:|
  | Prefill, 215 tokens | 7224.912 | 7215.189 | 7188.779 | 7186.463 | -0.50% |
  | Decode, 32 tokens | 6955.863 | 6934.154 | 6887.725 | 6864.781 | -0.98% |

  Paired changes reverse direction in both phases; no probe speedup is claimed.
  Instrumented decode dispatch is 213.231 ms/token within 215.241 ms/token phase
  time. Gate/up accounts for 46.82% and FFN down for 23.76% of dispatch time;
  all matrix projections total 98.54%, attention 1.46%.
- **Left:** the separate Q8 scale/payload study above missed its screening
  rule. Do not change workers or extrapolate an external performance pass from
  this three-pair diagnostic. No model integration is selected.
- **Gotchas:** all builds/tests finished before timing; all measured samples
  remain. No mx comparison or independent HF gate was run. Last-finisher entry
  can overlap other workers' compute and callback intervals can include
  descheduling. These intervals are not all recoverable overhead. Evidence:
  `benchmarks/current-8b-worker-spans-20260920.json`.

### Archived decode operation attribution

- **Goal:** label the existing 0.6B decode worker spans by operation, using the
  exact archived model/backend source and shapes to prove dispatch order.
- **Done:** all 12 traces and 54,144 decode records map to 141 dispatches per
  token: five per layer across 28 layers, then vocabulary projection. Archived
  source, guards and shapes prove the order. Exact last-finisher decomposition
  passes per operation and sums back to the original dispatch totals. In the
  historical instrumented current Q8 arm, gate/up accounts for 26.07% and
  vocabulary projection for 21.90% of dispatch time. Each of the five layer
  operations changes its control/current delta sign across three pairs.
- **Left:** within-group Q/K/V and gate/up member timing remains unresolved.
  Do not rewrite workers on this evidence. The separate current-source 8B
  plain/spans diagnostic is complete in the block above.
- **Gotchas:** archived current is `9cfe43f`, not a fresh `bf122fd` measurement.
  Model and decode paths are unchanged, but prefill source/compiler layout
  differs. Instrumentation perturbs timings; these are neither external-floor
  results nor evidence of a causal regression. Evidence:
  `benchmarks/worker-decode-operation-attribution-20260920.json`.

### CPU worker cost profile

- **Goal:** localize the remaining prefill/decode costs before selecting another
  hot-path change; compare the pre-error worker control with the retained runtime.
- **Done:** operation-level profiles cover 24 processes: Q8/F32, one/six
  workers, both source arms, three alternating pairs and two instrumented
  sequences after an uninstrumented warmup. Final-vector hashes agree with
  warmups, across source arms and thread counts. The initial instrument double
  counted attention's nested parallel_for; its consistency check rejected the
  run and the corrected probe excludes nested calls. Full samples and sources:
  `benchmarks/cpu-worker-profile-20260919.json`.
  Release definitions, DLL imports and the archived DLL hash establish OpenMP
  workers/barriers in the mx reference. Historical measurements are retained;
  ASSETS and the original artifact now carry a dated interpretation correction.
  The separate per-participant probe is complete: 24 serial Q8/F32 processes,
  six participants, three alternating pairs of plain/instrumented builds for
  both snapshots. Final full vectors match byte-for-byte across every arm and
  mode; all traces have 673 prefill and 4512 decode dispatches with ordered
  timestamps. The instrumented current failure/drain/reuse check passes.
  Evidence: `benchmarks/cpu-worker-spans-20260919.json`.
- **Left:** whole decode operations are now labeled by the verified archived
  dispatch order (see attribution block above). Individual projections inside
  grouped callbacks remain unresolved, and instrumentation changes timing
  materially. No production runtime change is selected.
  Keep the existing worker implementation and the external performance gate;
  do not repeat rejected dispatch variants on this evidence.
- **Gotchas:** instrumentation changes timing. The three-process comparison
  locates costs but does not clear a small regression or prove causality.
  Keep profiles isolated from builds/tests and user inference. Reference source
  may be inspected but cannot be copied into llmx.

| Instrumented phase mean ms, six workers | Before worker fix | Retained runtime |
|---|---:|---:|
| Q8 prefill | 562.43 | 573.38 |
| Q8 decode, 32 steps | 764.09 | 762.44 |
| F32 prefill | 617.72 | 626.08 |
| F32 decode, 32 steps | 2443.77 | 2431.62 |

Q/K/V and gate/up account for 10.57 ms of the 10.95 ms mean Q8 prefill difference
and 7.94 ms of the 8.36 ms F32 difference. These include dispatch/wait time.
Q8 prefill medians reverse the small mean ordering (569.89 vs 565.93 ms), so
this diagnostic does not establish a stable regression magnitude. The next
table is the separate completed span probe; do not pool the two sessions.

| Span-probe phase mean ms | Before plain | Retained plain | Before instrumented | Retained instrumented |
|---|---:|---:|---:|---:|
| Q8 prefill | 632.945 | 567.707 | 591.131 | 574.741 |
| Q8 decode, 32 steps | 803.943 | 762.698 | 779.716 | 783.178 |
| F32 prefill | 638.665 | 627.730 | 645.636 | 635.732 |
| F32 decode, 32 steps | 2463.925 | 2479.028 | 2458.478 | 2562.645 |

Q8 decode's control/current ordering reverses with instrumentation. F32 decode
differs by +0.61% in plain builds but +4.24% in instrumented builds. The exact
last-finisher decomposition separates entry, callback and final completion;
it does not turn overlapping participant spans into additive phase costs.
These results do not identify a stable worker regression or prove its absence.
No mx benchmark or new independent HF gate was run by this scratch probe.

### Live generation and loading progress

- **Goal:** stream generated text immediately in chat/generate, show prompt
  processing before the first token, and make loader progress reusable by
  current CLI consumers and future serving (ROADMAP #7).
- **Done:** optional synchronous loader byte-progress and inference text callbacks
  are implemented; CLI generate/chat owns terminal detection, stderr status and
  stdout flushing. Prompt-processing status appears before prefill. Normal
  Qwen3 output and --think stream before the next model step; legacy retroactive
  filters retain their prior buffered behavior. Stop/EOS and follow-up cache
  accounting remain unchanged.
  Windows/Linux full required-HF suites pass all nine components. After review
  tightened completion ordering, final native checks pass 7/7 on both platforms;
  Linux follow-up chat/progress and version checks pass again. Four MSVC mutants
  fail as intended: delayed delivery, missing flush, ignored read failures and
  completion before a failing trailing seek. Full documentation review covers
  all 25 Markdown files, including current capabilities, CLI defaults and test
  scope. Evidence: `benchmarks/live-generation-20260919.json`.
- **Left:** merge with the enclosing runtime stack only after its external
  performance gates pass. Gitea holds feature checkpoints; GitHub remains
  main-only. Next runtime work should profile the remaining CPU costs.
- **Gotchas:** callbacks are synchronous and do not provide concurrent execution
  or resumable-session recovery. Byte chunks can split UTF-8 characters. Loading
  counts tensor payload bytes, not metadata/padding or model preparation; final
  completion follows every read/seek. Comprehensive size/extent validation is
  still separate work. Legacy filtering buffers text when future markers can
  retroactively discard it; no server framework is added.

| Qwen3-0.6B Q8_0, 64 greedy tokens, median of 3 pairs | Before 8226e17 | Streaming |
|---|---:|---:|
| First visible text from process start (s) | 2.663 | 0.662 |
| Whole process elapsed (s) | 2.738 | 2.741 |
| CLI generation (tok/s) | 31.48 | 31.77 |

Same model/prompt, six CPU workers, stdout pipe and --think; one outer warmup
pair excluded. Every output byte matches. Builds/tests were stopped during
these runs. This is end-user delivery latency, including loading and prefill,
not a kernel-speedup or external mx-llama.cpp parity claim.

### Automatic build identification

- **Goal:** identify each CMake/plain MSVC build by release version plus Git
  revision, with a dirty marker for tracked changes and a clear archive fallback.
- **Done:** CMake refreshes build revision on each build, without rewriting an
  unchanged header. Plain build.bat emits the same metadata. --version and the
  usage banner show release plus Git revision and tracked-dirty state, with
  unknown fallback outside a checkout. Windows plain/CMake clean, dirty,
  new-commit, archive and no-op cases pass in a path containing spaces; Linux
  clean/dirty/new-commit rebuilds and version smoke pass. The current project
  build also reports its actual HEAD plus dirty state. Windows full required-HF suite passes with
  the new version regression. The Windows for/f equals-sign parsing issue was
  caught and fixed before the passing rerun.
  README now separates implemented CPU capabilities from future execution,
  serving and HF goals. All source/comments/docs and new messages use ASCII;
  Unicode fixture data is preserved. The requirement is recorded in AGENTS.
- **Left:** merge with the validated stack after its external gates pass.
  Live generation/progress reached checkpoint 9cfe43f; subsequent CPU cost
  profiles are recorded above. Full evidence for build identification is in
  `benchmarks/build-version-20260919.json`; all 25 Markdown files were reviewed
  for current capabilities, future goals, build behavior and ASCII compliance.
- **Gotchas:** untracked files do not mark a build dirty. Source archives report
  unknown even when nested in another repo. No timestamps, automatic release
  increments, commits/tags or new build/runtime dependencies are introduced.

### CLI thread settings

- **Goal:** make existing auto, decode and prefill thread flags work consistently
  for generate, follow-up chat and bench without changing kernel arithmetic.
- **Done:** generate/chat capture the resolved decode count, apply the prefill
  count and restore decode for every turn. Bench keeps auto selection; actual
  phase counts are visible through verbose/benchmark output. The new regression
  covers 32 generation/chat configurations and four bench cases per platform,
  with 64 HF-golden replies, and rejects all three reintroduced bug mutants.
  Windows full required-HF suite passes. Linux native tests and every correctness
  component pass; its initial automatic-thread synthetic floor fails. Pinning
  that test to its original one-worker conditions passes on both platforms,
  without changing floors. Logs retain the initial failure and the targeted
  reruns. Kernel/model comparator code is byte-identical to c072af2.
- **Left:** merge with the enclosing validated runtime stack once its external
  performance gates pass. Evidence: `benchmarks/cli-threads-20260919.json`.
- **Gotchas:** changing phase counts recreates the CPU pool. Automatic counts
  can be slower for tiny synthetic jobs, particularly under WSL; do not compare
  old serial-default benchmark results with new auto-default results. GPU
  backends will retain CPU-worker meaning for these flags; ubatch remains
  prompt tokens per forward pass. No GPU execution interface was introduced.

### CPU worker exception safety

- **Goal:** propagate CPU task failures after every participant finishes, with
  safe job lifetime and reusable dispatch state; clean up partial pool startup.
  This repairs the existing backend before ROADMAP #4a/#7 execution work.
- **Done:** dispatch catches caller/worker failures and waits for completion
  before rethrowing; startup joins partially created pools. Windows/Linux
  native checks, full suites with required real HF fixtures, Linux UBSan native
  tests and allocation/task fault sweeps pass. The original pool terminates on
  the task and partial-startup regressions. Independent real F32 HF checks pass;
  long F32/Q8 vectors and continuous/window NLL are exact against the control.
  Full samples, hashes, commands, logs and harnesses are archived in
  `docs/benchmarks/worker-errors-cpu-20260919.json`.
- **Left:** investigate the Q8 prefill cost before
  adoption, and close the external Q8 decode floor. No merge. The old root
  executable was retained at that checkpoint; current deployment is recorded above. Paired candidate Q8 prefill loses
  eight of nine rounds despite overlapping ranges; do not dismiss that as noise.
- **Gotchas:** dispatch recovery does not roll back partially written outputs
  or establish Model/session recovery. No concurrent submissions are supported.
  Comprehensive GGUF size/extent validation and model config/tensor validation are separate
  known gaps from the same review. Control is `3a82284`; no merge. Gitea holds
  the feature checkpoint, while the public/default branch remains unchanged.

- **Post-reboot decision:** retain the existing c072af2 implementation. The
  stored-call variant failed its longer comparison. Failure-only exception
  bookkeeping loses all five paired decode rounds. A shared non-template
  dispatch body improves exploratory prefill but does not improve decode;
  it is not adopted. No more dispatch variants are planned without profiling
  evidence. The user explicitly asked to keep the implementation simple.
  Full samples and source patches for the two latest scratch studies are in
  `docs/benchmarks/worker-cold-errors-cpu-20260919.json` and
  `docs/benchmarks/worker-shared-dispatch-cpu-20260919.json`. Both pass initial
  MSVC allocation/task-fault and grouped-kernel checks; neither entered full
  HF/platform adoption gates. Those experiments changed neither runtime source
  nor the root executable.
  Next performance work should localize the remaining cost before changing code.
  User authorization includes merging main and publishing GitHub once the
  requirements pass; current performance evidence does not clear that gate.

Stored-call experiment is **not adopted**. Full HF, exact vectors/NLL,
Windows/Linux suites and Linux UBSan native checks pass, but the longer run
does not establish a performance gain. Evidence and commands:
`docs/benchmarks/worker-invocation-cpu-20260919.json`.

| Mean tok/s, nine rounds | Before errors | Error checkpoint | Stored call | mx |
|---|---:|---:|---:|---:|
| Q8 prefill | 422.70 | 406.76 | 404.32 | 262.84 |
| Q8 decode | 44.69 | 44.93 | 44.53 | 46.32 |
| F32 prefill | 354.95 | 346.77 | 343.63 | 369.07 |
| F32 decode | 13.44 | 13.43 | 13.30 | 13.33 |

All ranges overlap. Stored-call Q8 prefill loses every pair against the
pre-error control, and no mean beats the error checkpoint. Preserve the early
five-round result as exploratory, not a reason to select the variant.

| Mean tok/s, nine matched rounds | Control | Candidate | mx-llama.cpp |
|---|---:|---:|---:|
| Q8 prefill | 408.39 | 386.18 | 263.80 |
| Q8 decode | 43.88 | 43.94 | 45.81 |
| F32 prefill | 362.85 | 377.11 | 370.26 |
| F32 decode | 13.91 | 14.06 | 13.69 |

Q8 prefill mean changes by -5.44%; Q8 decode remains -4.09% below mx. All
candidate/control ranges overlap; F32 means lead in this session without an
equivalence claim. Separate synthetic timing also has lower prefill/decode
means (see ASSETS). All samples are retained; builds/tests do not overlap timing.

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
  Current F32 results are in the KV and worker-error blocks; the older
  scale/load comparison records that earlier checkpoint. Sustained external
  parity remains unproven. Loading is excluded and each process warms up.
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
  hosted CI. See the KV and worker-error blocks for later measurements.
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
  Native Q8 scale/load scheduling is implemented and validated below.
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

### CPU comparison thread scaling

- **Goal:** locate the remaining CPU performance gap using matched thread
  counts and matrix-shape measurements, following ROADMAP #8.
- **Done:** explicit `--threads` in the comparator and runner, with requested
  counts echoed and checked. Windows llmx/mx and Linux llmx builds pass;
  invalid arguments and missing/wrong thread metadata are rejected. A real-model
  runner smoke passes. Matched Q8/F32 scaling is complete; those measurements
  used runtime `475f312`, before the KV and worker-error changes.
  Full vectors are byte-identical across all measured counts; continuous NLL
  is unchanged from the prior runtime and passes the independent HF fixture.
  Projection and matrix probes are complete. F32 matrix ranges overlap mx;
  Q8 matrix latency remains higher at the default comparison count. ASSETS and
  `benchmarks/cpu-thread-scaling-20260919.json` contain complete results.
- **Left:** merge the tool with the validated stack once its external floor
  is met. The resulting contiguous per-head KV change is implemented and
  validated below; it is no longer a pending experiment.
- **Gotchas:** use the same thread count in both arms and record it with every
  result. Keep the pinned model, tokens, reference revision, warmup and KV
  settings. Scaling diagnostics do not waive the existing external floor.

### Head-major CPU KV storage

- **Goal:** make each KV head's history contiguous to improve attention reads,
  preserve arithmetic order, and separate concrete CPU storage from logical
  sequence state without adding speculative device/server interfaces.
- **Done:** `HostKVCache` owns bounded growth and token-major projection writes;
  `Model` owns valid length/reset; backend attention receives an explicit head
  stride. Promoted the exact validated headers from the scratch candidate.
  Growth/reset/mixed histories match all 229,758 control values on Windows
  and Linux (the latter with nonrecovering UBSan). The direct storage oracle
  is in CTest and rejects a wrong-head relocation mutant.
  Real F32 HF logits and continuous/window NLL pass; long HF error is at most
  0.00012636185 under 0.001, with all 32 greedy IDs matching. F32 and Q8 each
  retain all 5,013,888 long-history logits and four full-precision NLL cases
  exactly versus `475f312`. Windows/Linux full suites with required real HF
  fixtures and Linux UBSan native/synthetic suites pass. Integrated MSVC code
  has the same `.text` hash as the validated candidate; final native CTest
  integration passes on Linux and UBSan.
  Nine interleaved matched rounds, all outliers retained:

  | Mean tok/s | Control | Head-major KV | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8 prefill | 366.46 | 416.37 | 262.34 | +58.72% |
  | Q8 decode | 42.36 | 44.41 | 45.51 | -2.42% |
  | F32 prefill | 345.42 | 361.38 | 362.44 | -0.29% |
  | F32 decode | 13.06 | 13.47 | 13.27 | +1.55% |

  Candidate/control ranges overlap; paired candidate wins are 9/9, 7/9,
  7/9 and 8/9 respectively. These results support an incremental selection,
  not a claim that the external floors are closed. The separate tiny synthetic
  decode mean declines 2.88% with overlapping ranges; that remains a recorded
  limitation. ASSETS and `benchmarks/head-major-kv-cpu-20260919.json` preserve
  all samples, hashes, scopes and reproduction sources. Earlier short-run
  diagnostics remain in `head-major-kv-initial-20260919.json`.
- **Left:** close the remaining external Q8 decode and F32 prefill gaps, then
  merge the validated stack and observe hosted CI. No merge or publish yet.
- **Gotchas:** reset retains allocation but must not expose stale tokens.
  Growth temporarily holds old and replacement storage together; future
  multi-user memory budgets must account for that peak. A CPU head-major layout
  is not a requirement for future device buffers, paging or shared prefixes.

### Chat follow-up cache validation

- **Goal:** preserve correct conversation history across follow-up prompts,
  reusing KV only when its exact token prefix matches the rendered transcript.
- **Done:** review found that `cmd_chat` skips cached tokens by count alone,
  renders twice per turn, and can pass empty logits to generation when the
  template does not add a generation suffix. Generated stop tokens and the
  unconditional EOS step also need accurate cache accounting.
  The new HF-backed chat regression reproduced a crash on the old binary.
  It also exposed double consumption of template block terminators, which
  skips adjacent content and breaks nested conditionals/loops.
  Implemented exact fed-token tracking and reset/refill for changed prefixes;
  removed unconditional EOS insertion and the redundant prefill/render pass.
  Fixed block terminator consumption and first keyword argument parsing. The
  latter affected Qwen namespace state and removal of old reasoning. New
  end-to-end HF reply fixtures pass; a token-count-only reuse mutant fails.
  The real Qwen template matches Jinja2 across initial and follow-up histories.
  Windows and Linux full suites with required real HF fixtures pass. MSVC
  renderer test, Linux CTest and Linux UBSan native/synthetic suites pass.
  Linux first exposed a missing `<cmath>` include in the standalone renderer;
  that is fixed. Chat coverage is 54 runs of nine scenarios across thread and
  batch settings; the real template has 12 independent Jinja2 cases. Logs,
  fixture hashes and reproduction commands are in
  `benchmarks/chat-followup-validation-20260919.json`.
- **Left:** merge with a validated runtime stack and observe hosted CI. The
  enclosing stack still needs its external performance floors; chat fixes do
  not establish those floors. Numerical kernels and model forward paths are
  unchanged by this fix.
- **Gotchas:** prefill continuation tests alone do not validate chat-template
  reuse. Keep messages separate from cached tokens; rendered text may change
  earlier turns or retokenize their boundary. This is single-user chat, not
  multi-user or concurrent request support.

### Correctness baseline vs HF reference

- **Goal:** give the suite an external ground truth. Correctness is measured
  against the HF reference, never against llmx itself (`docs/ROADMAP.md` #8).
- **Done:**
  - The tokenizer mode of `tools/gen_baseline.py` emits golden fixtures using
    `tokenizers` + `huggingface_hub`; numerical modes additionally need torch
    and transformers. Tokenizer output is committed to
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
  - Per-layer activation and full-corpus PPL goldens, plus maximum-context validation.
    The 1,943-token/32-step independent F32 HF check is complete; its scope and
    full-vector quantized diagnostics are recorded in ASSETS.
    Continuous and chunked excerpt gates are implemented, with an explicit
    disjoint-window scoring policy (`docs/USAGE.md`). They are not full-corpus
    coverage. Two-token windows show large quantized/HF deviations even under
    the unchanged old scorer; diagnostics and bounds are in `docs/ASSETS.md`.
    The existing ranking gate does not bound full-vector numerical error.
  - Existing tiny F32 full-vector and real-model excerpt bounds are implemented.
    Broader Q8 full-vector and full-corpus acceptance bounds remain open.
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
  - The CLI thread-settings block records the validated auto/prefill/decode
    corrections found during the documentation review.
  - JSON syntax and Unicode validation are implemented in the active parser
    block above; tensor-schema/range validation remains separate. See
    docs/src/core-json.md.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Preceding two-arm result (2026-09-20):** unchanged validated `bf122fd`
  runtime versus pinned mx `5542318e74`, nine alternating measured pairs per
  model after one discarded warmup pair; six workers, ubatch 128, 215 prompt
  plus 32 forced tokens and F32 KV. Every process exits zero and llmx final
  output hashes repeat within each model. No timing overlaps other compute.

  | Q8 model / phase | llmx mean tok/s | mx mean tok/s | Mean gap | llmx / mx median | Paired wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 440.479 | 274.766 | +60.31% | 448.159 / 277.473 | 9/9 |
  | 0.6B decode | 47.280 | 47.589 | -0.65% | 47.362 / 48.315 | 3/9 |
  | 8B prefill | 30.170 | 22.124 | +36.37% | 30.481 / 21.994 | 9/9 |
  | 8B decode | 4.403 | 4.440 | -0.82% | 4.465 / 4.501 | 2/9 |

  Both decode means and medians remain below mx. The external gate remains
  open; close results do not meet the required floor. Preserve all measured
  rounds, including the slower first measured round, and do not pool older
  sessions. Evidence: `benchmarks/q8-current-floor-20260920.json`.
  Independent review of all 27 pending commits found no additional concrete
  publication blocker; the existing Windows/Linux HF/native evidence remains
  valid for this unchanged runtime. Hosted CI follows eventual publication.
  Earlier ordered-prefill comparisons also exceed mx in F32 prefill/decode;
  their initial and Q8 follow-up sessions retain their separate scope.
- **Earlier instruction study (no runtime change):** paired native Q8 rows regress;
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
  - Close the current Q8 decode deficits while retaining prefill gains. The
    latest fixed `bf122fd` control/candidate/mx session measures control mean
    gaps of -2.01% (0.6B) and -0.67% (8B), with both medians below mx. The
    preceding two-arm session measured -0.65%/-0.82%; keep the sessions separate. Earlier bandwidth/thread observations and null allocation,
    fragmentation and prefetch experiments do not predict this current gap;
    mmap has not been established as a throughput improvement.
  - Grouped Q16 activation packing is now rejected. Its own scalar/integer
    arithmetic and activation tests passed, but unchanged native accuracy
    checks fail on MSVC and GCC (control 7/7, candidate 6/7). First diagnosed
    absolute error is 0.000219106674 against the existing 0.000206180004 limit,
    on a standalone one-worker Q8 dot. No test tolerance was changed; model
    HF-cost/performance runs were stopped before execution. Evidence:
    `benchmarks/q16-group-native-rejection-20260920.json`. Bounded-cost research
    does not satisfy AGENTS' lossless requirement. Preserve the rejection;
    do not tune a new tolerance to that failing case.
  - The exact Q8 horizontal reduction is rejected by complete timing despite
    passing numerical gates. Read-only review also found two redundant `h_`
    clears in `Model::step`, but no evidence that their cost closes the gap.
    The native two-block lambda is rejected at the no-spill codegen gate.
    The ordinary inner-loop follow-up also fails the useful-scheduling gate:
    no spills, but one extra feature reload and no useful cross-block work.
    Unrolling remains closed. Caller attribution and native sampling are
    complete above; neither established a recoverable production cost. The
    separately reopened prefill-placement assessment remains active. Prior
    F16C specialization, pointer increments and row pairing are nulls.
  - Thread and matrix-shape diagnostics are complete (see CPU comparison
    thread scaling above). F32 matrix ranges overlap mx, while Q8 matrix
    latency still trails it. The resulting head-major KV layout is now validated.
    Further changes should follow profiling of the current runtime, not repeat
    completed instruction/dispatch studies. The larger-model diagnostic did
    not close the external gap.
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
before writing code - see `AGENTS.md` -> "Starting a feature".
