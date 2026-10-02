# Expert projection type validation checkpoint - 2026-10-02

This records the completed local gates for base `46f60b649133` and candidate `234b5a00c5b2`, before final integration. The fix declares the existing grouped gate/up operation through one architecture helper and backend capability; it changes neither matrix arithmetic nor shaders. An unsupported resident pair fails before model allocation/adoption, while an unsupported streaming destination leaves the layer on its capable host. A separately typed down projection remains valid.

The integration refresh below is complete on main `fb366b16`; exact-head hosted CI is still required before landing. The original model and timing results retain their original source identities. The [machine-readable report](expert-types-20261002.json) pins every binary, model, capture and source evidence file by SHA256.

## Validation

| Gate | Result | Scope or limitation |
|---|---|---|
| Test-first regression | Both tests fail before the fix | Unsupported pair accepted; qwen35 plan lacks the operation |
| Linux native | 42/42 pass | Includes actual Vulkan capability assertion; no native skip |
| Linux strict device suite | 25/25 components pass | All six pinned fixtures and required tools/types; Windows short-path alias unittest skipped on Linux |
| Windows native | 43/43 pass | Actual Radeon VII backend; no native skip |
| Windows strict CPU | 25/25 components covered | Original 23 pass plus scoped CLI/roundtrip recovery; original failures retained |
| Windows Radeon VII | MoE and qwen35 pass | POSIX file-size-limit injection is the CPU roundtrip platform skip |
| CPU and Linux Vulkan identity | 8/8 cases; 24/24 byte captures | Dense Qwen3-0.6B Q8, synthetic Qwen3MoE Q8 and qwen35moe F32, real Qwen3-30B-A3B Q4_K_M |
| Compiled Vulkan modules | 99/99 identical to base | Generated module SHA256 equality |
| Same-owner layout control | Native checks and 8 identity cases pass | Different generated text; never shipped |
| mx-llama.cpp reference | Phase-level gate, not repeated here | No new reference rates or parity claim |

Identity uses 33 fixed input tokens, complete batched and row-decode logits, and 64 greedy steps through the production `llmx-model-logits` tool. The JSON records also match apart from their build version. The synthetic fixtures extend context to 640 for this identity workload; they do not replace the independent HF fixtures in the strict suites. The real model SHA256 is `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`.

The Windows sandbox initially denied output-path resolution in CLI and roundtrip. Published `8af97e88` and candidate binaries both reproduce filesystem error 5 in the same sandbox/cwd with fresh output paths; both reach the expected missing-input error with normal filesystem access. Only CLI/roundtrip were rerun under normal access and passed. This is 25 components covered across retained attempts, not a second full-suite run or an initial 25/25 result.

## Matched timing

Linux arms use clean detached worktrees, the same Release build and image `sha256:2787625f463391c440798579bac0d012c77fdfa949d086d036cdbd77f0b8d600`. The device is GPU0, RADV VEGA20 device `66a1`, Mesa `25.0.7-2+deb13u1`, PCI `03:00.0`, renderD136, UUID `00000000-0300-0000-0000-000000000000`; llvmpipe GPU1 is not selected. Timing uses CPU0-3, four threads/quota, 32 GiB cap, default device clocks, `--p 512 --n 128 --r 1 --ubatch 512 --cache-type-k f16 --cache-type-v f16 --dtype auto`. Each model/device retains the predeclared base/candidate/control/control/candidate/base order.

The control `b9de22aade89` adds only a volatile local gate-id read inside the new `routed_ops` helper. It changes generated code in the same owner, preserves the checked behavior and stays outside the release branch. It is a layout/noise control, not another proposed implementation.

Rates below are median tokens/s. Percentage changes are medians of the forward/reverse paired percentages, not ratios of the displayed medians. Positive means higher throughput. The mx reference gate remains phase-level; these are only before/after/control measurements.

| Model / device | Phase | Base | Candidate | Control | Candidate/base | Control/base | mx phase gate |
|---|---|---:|---:|---:|---:|---:|---|
| 0.6B Q8 / CPU | pp512 | 219.910 | 220.165 | 213.725 | +0.12% | -2.82% | Not rerun |
| 0.6B Q8 / CPU | tg128 | 43.985 | 44.270 | 43.065 | +0.65% | -2.10% | Not rerun |
| 0.6B Q8 / MI50 | pp512 | 8758.725 | 8757.100 | 8767.210 | -0.02% | +0.10% | Not rerun |
| 0.6B Q8 / MI50 | tg128 | 394.920 | 397.030 | 393.825 | +0.54% | -0.28% | Not rerun |
| 30B-A3B Q4_K_M / CPU | pp512 | 29.040 | 29.045 | 28.875 | +0.02% | -0.57% | Not rerun |
| 30B-A3B Q4_K_M / CPU | tg128 | 12.910 | 13.505 | 13.490 | +4.88% | +4.75% | Not rerun |
| 30B-A3B Q4_K_M / MI50 | pp512 | 1114.075 | 1114.435 | 1116.595 | +0.03% | +0.23% | Not rerun |
| 30B-A3B Q4_K_M / MI50 | tg128 | 126.765 | 126.480 | 126.635 | -0.22% | -0.10% | Not rerun |

No material slowdown is demonstrated by this round. Most candidate/base cells are near flat. The real-MoE CPU decode base is 12.24 then 13.58 tok/s, while candidate is 13.48/13.53 and control 13.42/13.56. The candidate/base paired changes are +10.13% and -0.37%; the +4.88% median does not establish a speedup. The same-behavior control shows the same apparent gain. The dense CPU control also has one low sample, 207.45/41.59 versus its later 220.00/44.54. All remain in the report; neither low result is assigned a cause or deleted.

## Every planned call and observed activity

All 24 calls returned 0. There are 463 monitor observations and 462 intervals, with a stop record. Predeclared flags are host idle below 25%, iowait above 5%, an unrelated process at least 100% of one CPU, any disk above 50 MB/s, or a sample gap above 2.5 seconds. The unique interval totals are respectively 0, 5, 80, 23 and 0. Per-call columns below show iowait/unrelated-CPU/disk flagged intervals; busy and gap are zero for every call.

The first real-MoE CPU base call overlaps 29 unrelated-CPU flagged intervals, versus 2/11 for the two candidates, 1/0 for controls and 0 for the last base. That is activity imbalance, not proof of a phase-specific cause. The last dense CPU base and second real-MoE CPU candidate also overlap disk flags. Whole-call monitoring includes loading and both benchmark phases.

| Model / device | Call arm in order | pp512 | tg128 | Intervals | Flags I/U/D | Missed/new/departed process observations |
|---|---|---:|---:|---:|---|---|
| 0.6B Q8 / CPU | 1 base | 219.18 | 43.84 | 13 | 0/6/0 | 7/55/53 |
| 0.6B Q8 / CPU | 2 candidate | 220.64 | 44.27 | 12 | 0/7/0 | 6/45/51 |
| 0.6B Q8 / CPU | 3 control | 207.45 | 41.59 | 12 | 0/5/0 | 5/60/57 |
| 0.6B Q8 / CPU | 4 control | 220.00 | 44.54 | 12 | 0/8/0 | 5/44/55 |
| 0.6B Q8 / CPU | 5 candidate | 219.69 | 44.27 | 12 | 0/7/0 | 1/94/52 |
| 0.6B Q8 / CPU | 6 base | 220.64 | 44.13 | 13 | 5/5/10 | 7/135/83 |
| 0.6B Q8 / MI50 | 1 base | 8760.35 | 396.42 | 3 | 0/1/0 | 0/60/8 |
| 0.6B Q8 / MI50 | 2 candidate | 8755.55 | 397.54 | 2 | 0/1/0 | 0/8/6 |
| 0.6B Q8 / MI50 | 3 control | 8773.30 | 395.44 | 2 | 0/1/0 | 0/10/11 |
| 0.6B Q8 / MI50 | 4 control | 8761.12 | 392.21 | 3 | 0/1/0 | 0/16/16 |
| 0.6B Q8 / MI50 | 5 candidate | 8758.65 | 396.52 | 3 | 0/0/0 | 1/14/13 |
| 0.6B Q8 / MI50 | 6 base | 8757.10 | 393.42 | 3 | 0/0/0 | 1/34/25 |
| 30B-A3B Q4_K_M / CPU | 1 base | 28.87 | 12.24 | 60 | 0/29/0 | 16/316/291 |
| 30B-A3B Q4_K_M / CPU | 2 candidate | 28.95 | 13.48 | 58 | 0/2/0 | 19/240/300 |
| 30B-A3B Q4_K_M / CPU | 3 control | 28.70 | 13.42 | 58 | 0/1/0 | 12/197/204 |
| 30B-A3B Q4_K_M / CPU | 4 control | 29.05 | 13.56 | 58 | 0/0/0 | 5/187/197 |
| 30B-A3B Q4_K_M / CPU | 5 candidate | 29.14 | 13.53 | 60 | 0/11/13 | 30/357/237 |
| 30B-A3B Q4_K_M / CPU | 6 base | 29.21 | 13.58 | 57 | 0/0/0 | 11/226/327 |
| 30B-A3B Q4_K_M / MI50 | 1 base | 1106.56 | 126.92 | 7 | 0/0/0 | 2/61/58 |
| 30B-A3B Q4_K_M / MI50 | 2 candidate | 1108.49 | 126.65 | 8 | 0/0/0 | 4/58/34 |
| 30B-A3B Q4_K_M / MI50 | 3 control | 1120.12 | 126.40 | 7 | 0/0/0 | 1/25/95 |
| 30B-A3B Q4_K_M / MI50 | 4 control | 1113.07 | 126.87 | 8 | 0/0/0 | 8/52/23 |
| 30B-A3B Q4_K_M / MI50 | 5 candidate | 1120.38 | 126.31 | 7 | 0/0/0 | 3/40/35 |
| 30B-A3B Q4_K_M / MI50 | 6 base | 1121.59 | 126.61 | 6 | 0/0/0 | 3/22/31 |

CPU/disk counters and GPU busy/VRAM observations are available throughout the sampled intervals; zero unknown counters does not establish an idle host. Across unique intervals there are 142 missed process observations, 2,167 new processes without deltas and 2,127 departures. Short-lived work between samples remains unmeasured. GPU0 and other cards have activity; counters do not attribute it to a process. Per-call GPU ranges, unknown counts and timestamps are in the JSON. Overlapping boundary intervals can appear in two calls and must not be added as unique counts. The last call has 0.838 seconds beyond the last bracketing interval; all other calls are fully bracketed.

## Reproduction and retained evidence

The rig root is `/zpool1/llmx-xdev-validation/expert-types-20261002`; local evidence is `.tmp-expert-types-run-20261002/`. The rig retains `functional-complete.json`, `binaries.json`, `shader-hashes.json`, `identity/complete.json` with raw captures and fixture inventory, `layout-control/`, and `timing-layout/{plan.json,complete.json,activity.jsonl}` with every call log. Local copies use `identity-complete.json`, `control-complete.json`, `timing-complete.json` and `timing-plan.json`. The local activity analysis reuses the storage gate auditor, preserving every observation and reporting unknowns. The report JSON hashes the raw evidence, scripts and final cleanup audit. The container is stopped and CPU0-3/GPU0 released.

Windows evidence is `.tmp-expert-types-windows-20261002/`, including initial `state.json`, `cpu-suite.log`, paired `access-probe-*.json`, scoped `recovery.json` and `completion-audit.json`. The fresh executable is `.tmp-expert-types-windows-20261002/build/Release/llmx.exe`, version `0.1.0+g234b5a00c5b2`, SHA256 `68626a99a6480973030b850c91b1372a1e5f885a892eec20af7f7546982a205f`. Its Radeon VII is UUID `00000000-2d00-0000-0000-000000000000`; Windows results are separate from Linux timing.

Representative commands, run from the candidate source after its fresh CMake build:

```text
ctest --test-dir build --output-on-failure
python -X utf8 tests/run_tests.py --exe build/llmx --device vulkan:0 --dtype auto --require-baseline --require-tools --require-device-types F32,Q8_0,Q4_0,Q4_1,Q4_K,Q5_K,Q6_K,MXFP4 --no-perf-floor
llmx-model-logits MODEL IDS PREFIX DEVICE f16 16 0
llmx bench --threads 4 --p 512 --n 128 --r 1 --ubatch 512 --cache-type-k f16 --cache-type-v f16 --dtype auto --model MODEL --device DEVICE
```

On Windows CTest adds `-C Release`; the CPU suite selects `--device cpu` and the executable under `build/Release/`. Scoped recovery adds `--only cli,roundtrip`; the Radeon check selects `--device vulkan:0 --only moe,qwen35`. Exact configure/build/test commands, timestamps and statuses remain in the cited manifests. No numerical bounds were changed, and no new reference-runtime campaign was run.

## Integration refresh on current main

At clean `47232075dbd2`, rebased onto `fb366b16`, Windows passes 43/43 native tests with zero skips and the docs, dead-code and architecture-boundary components. Linux passes 42/42 native tests with zero skips, and all 99 shader modules match the original candidate. Only STATUS conflicted during rebase; the two commits' source/test/build range-diff is unchanged. The final gate-record amendment changes only documentation. No model suite, identity capture or timing campaign was repeated.

The Windows executable is `.tmp-expert-types-integration-windows-20261002/build/Release/llmx.exe`, version `0.1.0+g47232075dbd2`, SHA256 `7c47aba7277a1eb77e9c221ac8d97ca3061356f43a21ba7960483397e4fe3e2c`. The Linux executable has the same version and SHA256 `6637c8e8165e6a81ebd3d0a191a92111a75f816f6c58bce2757258f2e40bee6f`. The Linux container was verified idle after its checks and stopped. Windows initially held its preflight on an existing MSBuild process; a two-second observation showed zero CPU delta and no compiler/link/test child, identifying an idle reusable node. That observation and the initial refusal are retained; the node was left untouched.
