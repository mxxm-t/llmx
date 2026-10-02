# CPU dtype release comparison (2026-10-02)

Main runtime `0bf4fcae` is unchanged in integration main `25549f02`. Candidate source archive `e66d21c7` includes the native F16 policy and CPU fit reserve. The llmx arms use GNU 14.2, the same Release configuration and archive build route, with version `llmx 0.1.0+unknown`; the reference is mx-llama.cpp `eefc4e73` on the CPU. Default clocks and the frozen token histories are used throughout.

The [machine-readable evidence](dtype-cpu-final-20261002.json) contains model hashes, binary identities, every block delta, activity alignment and final cgroup verification. The reporter control changes only the post-timing CLI report call; it has the candidate's arithmetic.

The release accepts the measured workload tradeoff: Q4 prefill is faster in four blocks and Q5 decode is faster in four, while Q5 prefill loses in three (median paired -4.39 percent). Uneven activity limits causal attribution. This does not establish mx parity or universal nonregression, and the reference gaps remain work to do.

Final CPU comparison: four threads, CPUs 0-3, pp512/tg128, F16 KV, 32/32 valid calls. Rates are tok/s.

| Model | Phase | Main | Candidate | Reporter control | mx reference | Candidate/main paired % | Candidate/mx paired % |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Q4_K_M | pp | 13.550 | 16.555 | 16.215 | 20.390 | +19.12 | -18.66 |
| Qwen3-8B Q4_K_M | tg | 5.345 | 5.840 | 5.085 | 6.160 | +1.43 | -12.18 |
| Qwen3-0.6B Q5_K_M | pp | 186.735 | 191.430 | 177.145 | 111.330 | -4.39 | +73.27 |
| Qwen3-0.6B Q5_K_M | tg | 44.175 | 46.820 | 44.330 | 48.470 | +5.69 | -3.89 |

Percent changes are medians of within-block ratios, not ratios of median rates. All four block deltas follow.

| Model | Phase | Comparison | Block 0 % | Block 1 % | Block 2 % | Block 3 % | Median paired % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B Q4_K_M | pp | candidate_vs_main | +11.96 | +11.76 | +56.67 | +26.27 | +19.12 |
| Qwen3-8B Q4_K_M | pp | candidate_vs_reference | -22.29 | -18.56 | -18.77 | +55.23 | -18.66 |
| Qwen3-8B Q4_K_M | pp | control_vs_main | +11.69 | +6.48 | +57.14 | +24.55 | +18.12 |
| Qwen3-8B Q4_K_M | pp | candidate_vs_control | +0.24 | +4.96 | -0.30 | +1.38 | +0.81 |
| Qwen3-8B Q4_K_M | tg | candidate_vs_main | +4.40 | -1.54 | +67.70 | -18.96 | +1.43 |
| Qwen3-8B Q4_K_M | tg | candidate_vs_reference | -10.69 | -13.66 | +6.80 | -28.52 | -12.18 |
| Qwen3-8B Q4_K_M | tg | control_vs_main | -14.61 | -15.58 | +67.42 | +4.59 | -5.01 |
| Qwen3-8B Q4_K_M | tg | candidate_vs_control | +22.27 | +16.63 | +0.17 | -22.52 | +8.40 |
| Qwen3-0.6B Q5_K_M | pp | candidate_vs_main | -2.62 | +13.71 | -7.09 | -6.16 | -4.39 |
| Qwen3-0.6B Q5_K_M | pp | candidate_vs_reference | +64.35 | +73.19 | +76.38 | +73.36 | +73.27 |
| Qwen3-0.6B Q5_K_M | pp | control_vs_main | -1.03 | -2.46 | -8.19 | -6.60 | -4.53 |
| Qwen3-0.6B Q5_K_M | pp | candidate_vs_control | -1.61 | +16.58 | +1.20 | +0.47 | +0.84 |
| Qwen3-0.6B Q5_K_M | tg | candidate_vs_main | +2.66 | +0.60 | +12.99 | +8.72 | +5.69 |
| Qwen3-0.6B Q5_K_M | tg | candidate_vs_reference | +1.60 | -3.63 | -4.15 | -11.26 | -3.89 |
| Qwen3-0.6B Q5_K_M | tg | control_vs_main | +2.11 | -9.55 | +13.23 | +8.75 | +5.43 |
| Qwen3-0.6B Q5_K_M | tg | candidate_vs_control | +0.54 | +11.22 | -0.21 | -0.02 | +0.26 |

Activity counts are flagged one-second intervals within each arm. Intervals crossing a call boundary can appear in both calls.

| Arm | Calls | Intervals | Busy CPU | I/O wait | Unrelated CPU | Disk | Gaps | Unknown CPU | Missed process observations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | 8 | 620 | 82 | 55 | 260 | 135 | 0 | 0 | 307 |
| candidate | 8 | 534 | 2 | 16 | 203 | 124 | 0 | 0 | 246 |
| control | 8 | 517 | 0 | 22 | 83 | 95 | 0 | 0 | 245 |
| reference | 8 | 535 | 0 | 29 | 115 | 143 | 0 | 0 | 209 |

Q4 activity alignment around the largest slowdowns:

| Block | Arm | Intervals | Busy CPU | I/O wait | Unrelated CPU | Disk | Gaps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | control | 118 | 0 | 11 | 11 | 27 | 0 |
| 2 | reference | 101 | 0 | 13 | 13 | 36 | 0 |
| 2 | candidate | 115 | 0 | 16 | 1 | 19 | 0 |
| 2 | main | 155 | 81 | 15 | 75 | 21 | 0 |
| 3 | reference | 167 | 0 | 0 | 77 | 62 | 0 |
| 3 | main | 169 | 1 | 0 | 146 | 30 | 0 |
| 3 | control | 120 | 0 | 0 | 4 | 6 | 0 |
| 3 | candidate | 139 | 2 | 0 | 76 | 55 | 0 |

Tradeoffs and limits:

- Qwen3-8B Q4_K_M pp: candidate/main +19.12% (4/4 positive blocks); candidate/mx -18.66% (1/4 positive blocks).
- Qwen3-8B Q4_K_M tg: candidate/main +1.43% (2/4 positive blocks); candidate/mx -12.18% (1/4 positive blocks).
- Qwen3-0.6B Q5_K_M pp: candidate/main -4.39% (1/4 positive blocks); candidate/mx +73.27% (4/4 positive blocks).
- Qwen3-0.6B Q5_K_M tg: candidate/main +5.69% (4/4 positive blocks); candidate/mx -3.89% (1/4 positive blocks).
- Four threads on CPUs 0-3 with an explicit four-CPU quota; no subtraction from historical six-thread rates.
- All 32 valid planned calls retained. Activity flags identify possible disturbance, not its causal effect.
- Block 2 main Q4 slowed sharply and subsequent Q5 arms also slowed. Its paired candidate/main gain is not evidence of a clean kernel effect; block deltas and activity alignment remain visible.
- Monitoring windows cover each complete invocation, including load, warm-up and timed phases. They do not provide phase-specific causal attribution. System CPU reads were complete, but missed process observations remain unknown.
- Reporter control perturbs only the CLI reporting site; it is not a universal kernel-layout bound. It deliberately has no completed matrix-path record.
- All llmx arms: fit_kv=false, identical GGUF logical context budget, KV backed on demand, identical fixed 512-token prompt and fresh 128-token decode. Warm-up backs 512 tokens before timing; strides/work follow touched blocks and actual history, not logical context. mx reserves 512 in its own cache layout. This capacity difference is disclosed and is not an effect of the CPU reserve fix.
- Stopped invalid quota1 attempt retained separately; it is an orchestration/configuration failure, not a discarded slow performance sample.
- Short-lived processes between samples may be missed; disappeared/new and inaccessible processes are retained as coverage limits. Unavailable GPU counters are not zero. GPU busy/VRAM counters are retained per device, without per-process attribution; no causal contention claim. Benchmark container and known docker launchers excluded from unrelated CPU.

Evidence:

- Invalid attempt: /zpool1/llmx-xdev-validation/dtype-main0bf-20261002/model-screen-reporter-control-v6
- Invalid resource proof: /zpool1/llmx-xdev-validation/dtype-main0bf-20261002/perf-prepared-v6/container-build-limits.json
- Corrected live cgroup/memory preflight: /zpool1/llmx-xdev-validation/dtype-main0bf-20261002/perf-prepared-v6/prelaunch-corrected.json
- Full valid samples, activity and source/binary identity: /zpool1/llmx-xdev-validation/dtype-main0bf-20261002/model-screen-reporter-control-v6-corrected
- Source archive SHA256: e66d21c7cdbdc6fc332dca878d4fcc8a48aab7f06a57d63ca36c7a61ec6fbf4f
