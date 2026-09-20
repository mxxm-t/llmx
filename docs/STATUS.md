# llmx - Development Status

Current implementation and remaining work. Historical checkpoints, failed
experiments and raw evidence remain in [ASSETS](ASSETS.md) and
`docs/benchmarks/`; their dated next steps are not current blockers.

## Prefill placement checkpoint (2026-09-20)

Backend-owned prefill placement is accepted for main under the performance
tradeoff policy. Publication and hosted validation of the new tests are pending.
On supported Windows topology, six-worker prefill uses separate physical cores
and checks restoration before decode. Other configurations fall back. There is
no runtime affinity flag, NUMA memory policy, arithmetic change or concurrent
submission support. Persistent OS refusal to restore is reported as an error.

| Final source check | Result |
|---|---:|
| Windows native / Linux native | 12/12 / 11/11 |
| Required HF Python suites, Windows / Linux | 11/11 / 11/11 |
| Active-placement HF cases | 28/28 |
| Exact Q8 follow-up pairs | 6/6 |
| F32/Q8 long-continuation vectors | 33/33 each |
| Q8 monitored blocks / processes | 24/24 / 72/72 |
| F32 monitored blocks / processes | 4/4 / 12/12 |
| Frozen identities rechecked | 79/79 in each run |

| Primary model | Paired prefill gain vs main | Paired combined gain vs mx |
|---|---:|---:|
| 0.6B Q8 | +26.40% | +37.49% |
| 8B Q8 | +32.84% | +34.84% |
| 0.6B F32 | +13.70% | +2.86% |

All three measured rounds are retained. Competing developer builds/model jobs
overlapped Q8 despite the reservation and materially limit causal claims.
Observed Q8 follow-up combined means range from -4.39% to +1.44% versus main,
with wide uncertainty; F32 decode is -1.26%, while its combined measurement is
+1.27% versus main. These costs are accepted alongside the primary gains, not
relabelled as zero or used to claim universal parity. Full phase, absolute-time,
activity and uncertainty tables are in the
[final comparison](ASSETS.md#final-prefill-placement-comparison-2026-09-20).

Independent review verified all 84 outputs/vectors and reproduced the statistics
and activity summaries. The single-worker performance smoke passes both arms;
its one-pair numbers are diagnostic. Source `250846a` is unchanged since final
correctness and timing. All 26 Markdown files were reviewed for this
checkpoint. The original working tree and executable remain untouched.

## Runtime base release checkpoint (2026-09-20)

The completed CPU runtime stack is accepted for release under the user's
performance tradeoff policy: large gains may justify smaller costs elsewhere,
with HF correctness and matched mx comparisons retained. This is a scoped
release decision, not a claim of universal per-phase or per-quant superiority.
The optional placement candidate is not part of that published base; its
completed integration is recorded in the checkpoint above.

Release `08351b0` combined runtime checkpoint `0d41a7b` with public main `9511a4a`.
All `src/` files at that reconciliation matched the runtime parent after Git
newline normalization. CI retry/cache handling and plain Windows build-header error
checks are retained. No inference arithmetic changed during reconciliation.
The original Windows working tree, user files and its root executable are not
replaced by this release.

| Final reconciliation check | Windows MSVC | Linux GCC 13.3 / WSL |
|---|---:|---:|
| CMake Release build | Pass | Pass |
| Native CTest | 10/10 | 10/10 |
| Offline downloader cases | 15/15 | 15/15 |
| Python suite with both HF models required | 11/11 | 11/11 |
| Tiny F32 full-logit maximum HF error | 0.00000070 | 0.00000070 |

These are correctness/build checks; simultaneous jobs and diagnostic synthetic
timings do not supply new performance measurements. The rig's HF container
lacked CMake, so the Linux run used the configured WSL toolchain instead.
Independent merge review verifies that native tests, all eleven Python
components, UBSan, required HF fixtures and downloader checks remain wired.
All 25 Markdown files were reviewed for source alignment, ASCII and local links.

## Base release performance decision

The base release's Q8 evidence is the fixed monitored comparison: six workers,
ubatch 128, F32 KV, eight measured rounds per model/workload, with all samples
retained. The table reports paired combined-work speed changes for production
versus pinned mx `5542318e74`; positive means faster. Follow-up prefix loading
is outside the timer. See the [complete phase tables and monitoring limits](ASSETS.md#monitored-prefill-comparison-2026-09-20).

| Model / workload | Combined speed vs mx | Approximate 95% interval |
|---|---:|---:|
| 0.6B Q8, primary 215+32 | +19.69% | +15.87% to +23.50% |
| 0.6B Q8, follow-up 1+32 | -2.58% | -12.82% to +7.67% |
| 0.6B Q8, follow-up 9+32 | +12.46% | -15.74% to +40.66% |
| 8B Q8, primary 215+32 | +24.41% | +10.65% to +38.17% |
| 8B Q8, follow-up 1+32 | -4.72% | -16.01% to +6.58% |
| 8B Q8, follow-up 9+32 | -2.87% | -7.61% to +1.88% |

The primary gains support release while the smaller follow-up costs remain
explicit. Zero-crossing intervals do not prove parity or make the costs zero.
Background imbalance, competing-agent jobs and coarse telemetry limit causal
attribution. No sample was discarded or corrected. An idle PC is not required;
future timing must keep monitoring and must not overlap competing agent work.

The base release used the separate nine-round F32 comparison from the
[ordered-reduction study](ASSETS.md#ordered-prefill-accumulator-reductions-2026-09-20):

| 0.6B F32 phase | llmx tok/s | mx tok/s | Mean-rate difference |
|---|---:|---:|---:|
| Primary prefill | 382.921 | 364.721 | +4.99% |
| Decode | 13.735 | 13.524 | +1.56% |

The base release's backend, inference and quantization sources match that
`bf122fd` study; later model edits validate construction. This is applicable
historical evidence, not a new F32 measurement on the reconciled tree. Older
F32 deficits and the earlier sub-percent Q8 decode veto are superseded as
release blockers by the later evidence and clarified tradeoff policy. Their
original results remain in ASSETS. Other hardware, quants and workloads need
their own measurements; K-quant optimization remains separate work below.

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
| JSON syntax and Unicode validation      | Done |
| GGUF reader size and tensor extent validation | Done |
| JSON quantize tensor validation | Done |
| Qwen model construction validation | Done |
| Device execution model (GPU prerequisite) | In Progress (LDEV, separate branch) |
| GPU backends (ROCm first, Vulkan portability) | Planned |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |
| Chat follow-up cache validation          | Done |
| Correctness baseline vs HF reference     | In Progress |
| Pinned HF reference generation           | Done |
| Optional Qwen3-8B HF consumer             | Done |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Matched CPU comparison thread selection | Done |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | Done |
| CPU attention in backend (ROADMAP #4a)  | Done |
| CPU row streaming / parallel prefill   | Done |
| CPU attention value accumulation      | Done |
| CPU grouped projections              | Done |
| CPU Q8 scale / load scheduling       | Done |
| Head-major CPU KV storage             | Done |
| CPU worker exception safety           | Done |
| CPU worker cost profile                 | Done |
| CPU ordered prefill reductions          | Done |
| Backend-owned prefill placement | Done (hosted validation pending) |
| CLI thread settings                    | Done |
| Automatic build identification          | Done (main `9511a4a`) |
| Live generation and loading progress     | Done |
| GitHub CPU CI                          | Done     |
| HF fixture download retries and CI cache | Done |
| Hosted numeric/path portability repair | Done (five jobs green at `851d375`) |
| HF integration (pull + Hub formats)      | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

`Done` denotes implemented and validated functionality in this release tree.
Runtime `08351b0` is published on both main remotes. Its initial five-check
hosted run `35512421834` passed ordinary Ubuntu and required HF, but failed
Windows reference-generator path spelling, UBSan exact scalar-tail comparison,
and macOS JSON subnormal conversion. Repair `851d375` passed all five jobs in
[run 35512954742](https://github.com/mxxm-t/llmx/actions/runs/35512954742):
Windows, macOS Intel, Linux, Linux UBSan and required HF. The repair changes
JSON conversion plus test portability, preserving inference kernels and bounds.
Evidence is in `docs/benchmarks/ci-portability-20260920.json`; prior four-check
passes at `b266650` and `9511a4a` cover those smaller releases.
See [CI](CI.md) for the precise workflow scope and local reproduction commands.

## Active feature blocks

### Scoped correctness coverage and remaining HF work

- **Goal:** keep independent HF ground truth and extend coverage where the roadmap requires it.
- **Done:** exact tokenizer fixtures; tiny tied/untied F32 full logits and NLL; real Q8/Q4 ranking and excerpt PPL; HF/Jinja2 follow-up chat fixtures; pinned reference generation and strict consumers. The unchanged real 8B consumer previously passed 37/37 on Windows and Linux with frozen bounds. Real 0.6B F32/Q8 1,943-token plus 32-step continuation checks are archived in ASSETS.
- **Left:** broader full-corpus, maximum-context and per-layer references, plus prospective numerical bounds for any new lossy kernels. Short 8B rankings/excerpts are not deep-context validation.
- **Gotchas:** self-consistency is supplementary. Exact comparison against another llmx path cannot replace HF. Model construction validation does not establish finite weights, arbitrary token-ID safety, request budgets or failed-session recovery.

### K-quant and device execution work (separate developer branch)

- **Goal:** improve the remaining K-quant decode path and continue ROADMAP #4a without overlapping this release/placement work.
- **Done:** LDEV owns the separate `design/device-execution-model` branch and K-quant experiments. Its changes are not incorporated by this release. TUI measurements and source identities must be reviewed before adoption. LDEV reports that quantized-activation commits `357d68d` and `97d52e8` fail `backend-group`; they remain isolated and are not merge-ready. Arithmetic-preserving dispatch work is being separated onto a passing base.
- **Left:** prospective correctness/performance validation for any new quantized-activation path; backend-owned weights/activations and execution lifetime before vendor GPU kernels. Coordinate rebases after release and announce timing reservations.
- **Gotchas:** earlier grouped Q16 failed the unchanged native double-dot accuracy contract. Do not reuse it as a lossless baseline or weaken bounds after observing results. CPU Q8 results do not establish K-quant parity.

## Working rules and ownership

XDEV owns runtime release, validation and optional placement. LDEV owns its
separate K-quant/device work. Coordination uses the TUI on port 8181 and the
shared devlog; the notification watcher is polled explicitly. Builds and tests
may run in parallel when no timing reservation is active. Keep every planned
performance sample, record ordinary machine activity, and report missing
telemetry honestly. GitHub receives main only; feature checkpoints stay on Gitea.
