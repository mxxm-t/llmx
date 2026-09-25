COMPLETE: 128/128 rows. No adoption verdict.

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

Rates are medians. Paired percentages are geometric means of same-round ratios; positive means higher throughput.

| Model | Phase | Pair | All blocks | Flagged blocks | Unknown-monitor blocks |
|---|---|---|---:|---:|---:|
| Qwen3-0.6B-Q8_0.gguf | pp | candidate/base | +1.224% (n=8) | +1.224% (n=8) | +1.224% (n=8) |
| Qwen3-0.6B-Q8_0.gguf | pp | layout/base | -2.729% (n=8) | -2.729% (n=8) | -2.729% (n=8) |
| Qwen3-0.6B-Q8_0.gguf | pp | candidate/mx | +90.439% (n=8) | +90.439% (n=8) | +90.439% (n=8) |
| Qwen3-0.6B-Q8_0.gguf | tg | candidate/base | +2.392% (n=8) | +2.392% (n=8) | +2.392% (n=8) |
| Qwen3-0.6B-Q8_0.gguf | tg | layout/base | -0.494% (n=8) | -0.494% (n=8) | -0.494% (n=8) |
| Qwen3-0.6B-Q8_0.gguf | tg | candidate/mx | +11.006% (n=8) | +11.006% (n=8) | +11.006% (n=8) |
| Qwen3-0.6B-Q4_0.gguf | pp | candidate/base | +2.367% (n=8) | +2.367% (n=8) | +2.367% (n=8) |
| Qwen3-0.6B-Q4_0.gguf | pp | layout/base | +3.549% (n=8) | +3.549% (n=8) | +3.549% (n=8) |
| Qwen3-0.6B-Q4_0.gguf | pp | candidate/mx | +4.033% (n=8) | +4.033% (n=8) | +4.033% (n=8) |
| Qwen3-0.6B-Q4_0.gguf | tg | candidate/base | -1.439% (n=8) | -1.439% (n=8) | -1.439% (n=8) |
| Qwen3-0.6B-Q4_0.gguf | tg | layout/base | +2.206% (n=8) | +2.206% (n=8) | +2.206% (n=8) |
| Qwen3-0.6B-Q4_0.gguf | tg | candidate/mx | +0.828% (n=8) | +0.828% (n=8) | +0.828% (n=8) |
| Qwen3-0.6B-Q5_K_M.gguf | pp | candidate/base | -0.292% (n=8) | -0.292% (n=8) | -0.292% (n=8) |
| Qwen3-0.6B-Q5_K_M.gguf | pp | layout/base | +0.858% (n=8) | +0.858% (n=8) | +0.858% (n=8) |
| Qwen3-0.6B-Q5_K_M.gguf | pp | candidate/mx | +72.699% (n=8) | +72.699% (n=8) | +72.699% (n=8) |
| Qwen3-0.6B-Q5_K_M.gguf | tg | candidate/base | +0.427% (n=8) | +0.427% (n=8) | +0.427% (n=8) |
| Qwen3-0.6B-Q5_K_M.gguf | tg | layout/base | +1.996% (n=8) | +1.996% (n=8) | +1.996% (n=8) |
| Qwen3-0.6B-Q5_K_M.gguf | tg | candidate/mx | +4.692% (n=8) | +4.692% (n=8) | +4.692% (n=8) |
| Qwen3-8B-Q8_0.gguf | pp | candidate/base | +1.527% (n=8) | +1.527% (n=8) | +1.527% (n=8) |
| Qwen3-8B-Q8_0.gguf | pp | layout/base | -0.730% (n=8) | -0.730% (n=8) | -0.730% (n=8) |
| Qwen3-8B-Q8_0.gguf | pp | candidate/mx | +88.518% (n=8) | +88.518% (n=8) | +88.518% (n=8) |
| Qwen3-8B-Q8_0.gguf | tg | candidate/base | +0.316% (n=8) | +0.316% (n=8) | +0.316% (n=8) |
| Qwen3-8B-Q8_0.gguf | tg | layout/base | -1.734% (n=8) | -1.734% (n=8) | -1.734% (n=8) |
| Qwen3-8B-Q8_0.gguf | tg | candidate/mx | +0.126% (n=8) | +0.126% (n=8) | +0.126% (n=8) |

Every observed planned row is retained; flagged and unknown subsets overlap and do not replace complete results.
Common run0 is retained as prospective warmup; only run1 contributes measured rates. CLI warmup is internal and unreported.
Activity spans entire invocations, including loading and warmup; system disk/GPU flags do not establish unrelated contention. Missing/inaccessible processes and telemetry remain unknown.
CLI rates are rounded to 0.01 tokens/second; reconstructed milliseconds are approximate.
A single layout control does not define all possible layout variation. Same-model witnesses and independent HF/depth gates are separate.
Warmed pp215 plus 32 teacher-forced continuation tokens; loading/tokenization/sampling excluded. This harness is separate from normal CLI layout and does not establish every workload or default serving performance.
