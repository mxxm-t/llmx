COMPLETE: 72/72 rows. No adoption verdict.

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

Rates are medians. Paired percentages are geometric means of same-round ratios; positive means higher throughput.

| Model | Phase | Pair | All blocks | Flagged blocks | Unknown-monitor blocks |
|---|---|---|---:|---:|---:|
| Qwen3-0.6B-Q8_0.gguf | pp | candidate/base | -1.187% (n=6) | -1.187% (n=6) | -1.187% (n=6) |
| Qwen3-0.6B-Q8_0.gguf | pp | layout/base | -1.317% (n=6) | -1.317% (n=6) | -1.317% (n=6) |
| Qwen3-0.6B-Q8_0.gguf | tg | candidate/base | +1.197% (n=6) | +1.197% (n=6) | +1.197% (n=6) |
| Qwen3-0.6B-Q8_0.gguf | tg | layout/base | +0.411% (n=6) | +0.411% (n=6) | +0.411% (n=6) |
| Qwen3-0.6B-Q4_0.gguf | pp | candidate/base | -0.946% (n=6) | -0.946% (n=6) | -0.946% (n=6) |
| Qwen3-0.6B-Q4_0.gguf | pp | layout/base | +1.862% (n=6) | +1.862% (n=6) | +1.862% (n=6) |
| Qwen3-0.6B-Q4_0.gguf | tg | candidate/base | -1.325% (n=6) | -1.325% (n=6) | -1.325% (n=6) |
| Qwen3-0.6B-Q4_0.gguf | tg | layout/base | -1.196% (n=6) | -1.196% (n=6) | -1.196% (n=6) |
| Qwen3-0.6B-Q5_K_M.gguf | pp | candidate/base | -1.873% (n=6) | -1.873% (n=6) | -1.873% (n=6) |
| Qwen3-0.6B-Q5_K_M.gguf | pp | layout/base | +1.390% (n=6) | +1.390% (n=6) | +1.390% (n=6) |
| Qwen3-0.6B-Q5_K_M.gguf | tg | candidate/base | -6.568% (n=6) | -6.568% (n=6) | -6.568% (n=6) |
| Qwen3-0.6B-Q5_K_M.gguf | tg | layout/base | -4.110% (n=6) | -4.110% (n=6) | -4.110% (n=6) |
| Qwen3-8B-Q8_0.gguf | pp | candidate/base | -0.867% (n=6) | -0.867% (n=6) | -0.867% (n=6) |
| Qwen3-8B-Q8_0.gguf | pp | layout/base | -0.404% (n=6) | -0.404% (n=6) | -0.404% (n=6) |
| Qwen3-8B-Q8_0.gguf | tg | candidate/base | +0.828% (n=6) | +0.828% (n=6) | +0.828% (n=6) |
| Qwen3-8B-Q8_0.gguf | tg | layout/base | +2.146% (n=6) | +2.146% (n=6) | +2.146% (n=6) |

Every observed planned row is retained; flagged and unknown subsets overlap and do not replace complete results.
Common run0 is retained as prospective warmup; only run1 contributes measured rates. CLI warmup is internal and unreported.
Activity spans entire invocations, including loading and warmup; system disk/GPU flags do not establish unrelated contention. Missing/inaccessible processes and telemetry remain unknown.
CLI rates are rounded to 0.01 tokens/second; reconstructed milliseconds are approximate.
A single layout control does not define all possible layout variation. Same-model witnesses and independent HF/depth gates are separate.
Normal CLI translation-unit layout check: synthetic LCG prompt pp215, then independent decode32 from empty history. Default ubatch512 and model context; F16 KV, one sequence and six CPU threads. One warmup per phase followed by one measured run per invocation. This is distinct from the common harness and has no matched mx arm.
