# Vulkan weight dispatch checkpoint - 2026-10-02

Correctness and model identity are complete at measured private candidate `4ad79007fd2d` against base `fb366b16cef4`. **The timing screen is accepted at the measured repeatability. Integration onto main `2c293678` merged without code conflicts; fresh Windows/Linux build/CTest checks pass. Publication requires hosted CI green at the final commit and an unchanged-main fast-forward.** The original roughly 50% real-MoE CPU decode deficit did not reproduce with the same binaries, and a targeted ten-repetition GPU check narrowed the smaller dense decode decline. All 54 calls remain in three separate groups. This supports the bounded screen decision, not a speedup, proof of exact nonregression or reference parity. The specific cause of the first-round deficit remains unproven.

## Current-main integration

The single feature commit is based on published `2c293678`. Only STATUS conflicted; the normalized source/test/CMake delta inherited from main is exact. Fresh detached builds at `d4db2515` pass Windows 43/43 and Linux 42/42 native tests, with no native test skipped and all 331 dispatch/refusal cases executed on each. Both pass docs, dead-code and architecture checks. All 99 Linux shader modules match the measured candidate. The Linux container stopped and has no remaining device clients. The final amendment changes documentation only. Exact-head hosted CI and an unchanged-main check precede fast-forward publication.

The JSON integration record pins both binaries, source manifests, complete native logs and conditional numerical subcases. These refresh checks do not relabel the original model suites, identities or timing as measurements of the integrated head.

## Scope and validation

The change gathers existing Vulkan type choices into one private descriptor and kernel IDs/names/module bindings into one ordered list. It adds no type, numerical policy, threshold or shader arithmetic. All 91 existing kernel IDs and bindings remain equivalent by the development comparison; all 99 generated shader modules match base. The measured candidate excludes the independent mixed-expert capability fix; integration inherits that published fix from main.

| Check | Result | Scope |
|---|---|---|
| Linux native | 42/42 pass | Actual RADV integer-dot device; 331 dispatch/refusal cases |
| Linux strict device suite | 25/25 components pass | Six pinned fixtures; required tools and all eight current weight types |
| Windows native | 43/43 pass | Fresh clean-commit MSVC Vulkan build; Radeon VII |
| Windows docs/dead-code/architecture | Pass | 13 known/16 planted; 7 known/18 planted; 22 files/0 findings |
| Base/candidate model identity | 22/22 cases, 66/66 byte capture pairs | 11 fixtures on CPU and device, records equal except build version |
| Layout control | Native and four model/device identities pass | Same-owner perturbation; 99 shader modules identical |
| Timing | 54/54 calls returned 0 | Original 24, rotated follow-up 24, dense GPU R10 six; accepted at measured repeatability |
| mx-llama.cpp reference | Not rerun | Reference-speed comparison remains the phase-level gate |

Identity covers the six pinned small real fixtures, real Qwen3-30B-A3B Q4_K_M, Qwen3MoE Q4_1, qwen35moe F32 and dense-tied/routed MXFP4 fixtures. Captures compare full batched/decode logits and greedy output, not only top picks. Both devices collectively cover storage IDs 0, 2, 3, 8, 12, 13, 14 and 39. [Results JSON](results.json) pins model/capture/binary hashes and every raw evidence file.

No native tests were skipped. Linux skips the Windows-only short-path-alias unittest. Optional numerical subcases without float preservation retain their documented skips in the backend logs; a passing component is not a claim that unavailable optional hardware was exercised. The prior dirty development executable and its focused checks remain historical; the clean-commit Windows executable is `.tmp-vulkan-dispatch-windows-20261002/build/Release/llmx.exe`, version `0.1.0+g4ad79007fd2d`, SHA256 `fff71543ef89f77756f2176173baf4aeb8c7f1a6bf7ea4495830beb33e160fd2`.

## First timing round (retained)

All Linux arms are clean detached worktrees built alike in image `sha256:2787625f463391c440798579bac0d012c77fdfa949d086d036cdbd77f0b8d600`, Release/Vulkan enabled. The device is RADV VEGA20, PCI `03:00.0`, renderD136, Mesa `25.0.7-2+deb13u1`, UUID `00000000-0300-0000-0000-000000000000`. CPU0-3, four threads/quota, 32 GiB cap and default device clocks are shared. Every model/device uses the planned base/candidate/control/control/candidate/base order. Workload: `bench --threads 4 --p 512 --n 128 --r 1 --ubatch 512 --cache-type-k f16 --cache-type-v f16 --dtype auto`, with the model and device added.

The same-owner control `968345fe7039` adds a volatile device-index round trip during Vulkan construction, with no per-token volatile operation. Generated text differs, checked behavior and shaders do not. This single perturbation does not establish a universal layout band, explain the CPU loss, or qualify a regression for release. It never ships.

Retained-binary diagnosis finds candidate/control identical in addresses, sizes and bytes for all 159 compared CPU/q8 functions, 22 bench functions and the 432-byte CPU vtable. Selected base/candidate Q4/Q5 decode and dispatch instruction sequences match after target-address/RIP-relative normalization. Other data sections were not proved equal; generic q8 dispatch retains unverified constant-symbol offsets. This excludes a difference in those compared candidate/control functions, not every possible layout or runtime cause. The diagnosis artifact and SHA256 are pinned in the JSON.

The completed follow-up uses the same 24 workloads in control/candidate/base/base/candidate/control order, adding child CPU time, faults and I/O plus cgroup snapshots. A separate dense-GPU check uses ten internal repetitions per call. Their results follow the untouched first-round samples below; neither replaces an original sample.

Rates are median tokens/s; changes are medians of the forward/reverse paired percentages, not ratios of these displayed medians. Positive means faster. The mx gate remains separate and has no fresh rates here.

| Model / device | Phase | Base | Candidate | Control | Candidate/base | Pair changes | mx phase gate |
|---|---|---:|---:|---:|---:|---|---|
| 0.6B Q8 / CPU | pp512 | 121.695 | 138.280 | 134.230 | +31.53% | -15.73%/+78.78% | Not rerun |
| 0.6B Q8 / CPU | tg128 | 23.595 | 24.840 | 21.510 | +17.85% | -10.18%/+45.89% | Not rerun |
| 0.6B Q8 / MI50 | pp512 | 8673.280 | 8736.665 | 8746.170 | +0.74% | +0.06%/+1.41% | Not rerun |
| 0.6B Q8 / MI50 | tg128 | 372.465 | 366.400 | 371.975 | -1.62% | -0.61%/-2.64% | Not rerun |
| 30B-A3B Q4_K_M / CPU | pp512 | 21.620 | 20.330 | 26.075 | -6.41% | -8.17%/-4.65% | Not rerun |
| 30B-A3B Q4_K_M / CPU | tg128 | 7.920 | 3.890 | 7.185 | -50.44% | -49.28%/-51.60% | Not rerun |
| 30B-A3B Q4_K_M / MI50 | pp512 | 1128.045 | 1118.300 | 1112.915 | -0.86% | -1.99%/+0.27% | Not rerun |
| 30B-A3B Q4_K_M / MI50 | tg128 | 126.425 | 125.640 | 126.490 | -0.62% | -0.14%/-1.10% | Not rerun |

CPU 30B decode is 2.47/5.31 tok/s for candidate against base 4.87/10.97, with control 4.41/9.96. Candidate/base is -49.28%/-51.60%; this was a material unresolved deficit that required the follow-up below; the original observations are not deleted or relabeled as parity. CPU prefill is also lower in both 30B pairs. Dense CPU rates change strongly over time, so their positive paired medians are not established improvements. MI50 decode declines in both pairs for both models, about -1.62% on 0.6B and -0.62% on 30B by the paired medians. The smaller declines remain visible alongside the large CPU loss. These first-round results alone established neither cause nor performance acceptance; the completed follow-up and its limits are below.

## All calls and activity

Every row below is retained. Flags are counts of overlapping intervals in order B/I/U/D/G: host idle below 25%, iowait above 5%, unrelated process at least 100% of one CPU, any disk above 50 MB/s, sample gap above 2.5 seconds. Process columns count missed/new-without-delta/departed observations. These are whole-call observations, including load, prefill and decode; they do not identify the cause of a phase result.

| Model / device | Arm in order | pp512 | tg128 | Intervals | B/I/U/D/G | Missed/new/departed |
|---|---|---:|---:|---:|---|---|
| 0.6B Q8 / CPU | 1 base | 167.79 | 34.18 | 18 | 0/0/8/1/0 | 13/133/69 |
| 0.6B Q8 / CPU | 2 candidate | 141.40 | 30.70 | 16 | 0/0/9/0/0 | 10/108/96 |
| 0.6B Q8 / CPU | 3 control | 168.08 | 24.96 | 17 | 3/0/11/2/0 | 23/165/172 |
| 0.6B Q8 / CPU | 4 control | 100.38 | 18.06 | 23 | 22/0/6/0/0 | 35/244/227 |
| 0.6B Q8 / CPU | 5 candidate | 135.16 | 18.98 | 27 | 27/0/10/0/0 | 35/226/249 |
| 0.6B Q8 / CPU | 6 base | 75.60 | 13.01 | 31 | 31/0/4/0/0 | 41/230/293 |
| 0.6B Q8 / MI50 | 1 base | 8731.21 | 370.02 | 3 | 3/0/1/0/0 | 3/58/21 |
| 0.6B Q8 / MI50 | 2 candidate | 8736.11 | 367.78 | 2 | 2/0/0/0/0 | 2/80/9 |
| 0.6B Q8 / MI50 | 3 control | 8754.80 | 369.16 | 3 | 3/0/0/0/0 | 1/50/11 |
| 0.6B Q8 / MI50 | 4 control | 8737.54 | 374.79 | 3 | 3/0/0/0/0 | 2/30/29 |
| 0.6B Q8 / MI50 | 5 candidate | 8737.22 | 365.02 | 2 | 2/0/0/0/0 | 1/15/14 |
| 0.6B Q8 / MI50 | 6 base | 8615.35 | 374.91 | 3 | 3/0/1/0/0 | 2/28/20 |
| 30B-A3B Q4_K_M / CPU | 1 base | 16.15 | 4.87 | 160 | 150/0/82/8/0 | 169/1305/1389 |
| 30B-A3B Q4_K_M / CPU | 2 candidate | 14.83 | 2.47 | 173 | 54/16/59/85/0 | 282/1918/1697 |
| 30B-A3B Q4_K_M / CPU | 3 control | 23.78 | 4.41 | 157 | 0/0/7/40/0 | 108/607/603 |
| 30B-A3B Q4_K_M / CPU | 4 control | 28.37 | 9.96 | 89 | 0/0/71/3/0 | 27/293/487 |
| 30B-A3B Q4_K_M / CPU | 5 candidate | 25.83 | 5.31 | 80 | 0/1/41/16/0 | 89/468/530 |
| 30B-A3B Q4_K_M / CPU | 6 base | 27.09 | 10.97 | 76 | 0/38/8/57/0 | 21/604/470 |
| 30B-A3B Q4_K_M / MI50 | 1 base | 1130.41 | 126.56 | 9 | 0/4/0/5/0 | 2/49/20 |
| 30B-A3B Q4_K_M / MI50 | 2 candidate | 1107.86 | 126.38 | 8 | 0/0/0/0/0 | 1/83/68 |
| 30B-A3B Q4_K_M / MI50 | 3 control | 1129.10 | 126.53 | 10 | 0/8/1/8/0 | 9/69/24 |
| 30B-A3B Q4_K_M / MI50 | 4 control | 1096.73 | 126.45 | 10 | 0/2/0/3/0 | 1/129/39 |
| 30B-A3B Q4_K_M / MI50 | 5 candidate | 1128.74 | 124.90 | 17 | 0/8/13/15/0 | 2/72/79 |
| 30B-A3B Q4_K_M / MI50 | 6 base | 1125.68 | 126.29 | 10 | 0/1/0/5/0 | 5/70/43 |

The monitor retains 926 samples and 925 intervals, plus its stop record. Unique interval flags are B=293, I=76, U=327, D=244, G=0; overlapping per-call counts are not additive. Unrelated hot observations include `llmx`, compiler processes, split/model tools and system/container work. Available CPU/disk and GPU busy/VRAM counters have no unknown entries in the sampled intervals, but missed, new and departed processes still leave unmeasured work. GPU ranges and all process coverage counts remain in the JSON; complete observations remain in `activity.jsonl` and the existing full analysis. The last call has 0.348 seconds beyond the final bracketing interval; the others are fully bracketed. None of these observations proves the CPU deficit is environmental.

## Completed follow-up and measured repeatability

The rotated 24-call block used the identical three binaries, models and workload. Real-MoE CPU decode medians are candidate 13.535, base 13.425 and control 13.550 tok/s. Candidate/base pairs are +0.52%/+1.12%, versus the original -49.28%/-51.60%; candidate/control pairs are -0.37%/+0.15%. The large deficit therefore did not reproduce. The compared candidate/control CPU and benchmark code is exact, as recorded above. This does not prove that paging caused the original difference: the first block lacks the per-call resource counters needed for that attribution.

Dense MI50 decode still had a small negative paired median in the rotated R1 block, so a six-call R10 check measured the same model with ten internal repetitions. Its decode medians are candidate 392.875, base 390.855 and control 392.410 tok/s; candidate/base pairs are -0.23%/+1.28%, and candidate/control +0.41%/-0.17%. The full means and within-call standard deviations are retained below. GPU prefill and the first/follow-up comparisons still have mixed signs. There is no speedup or exact nonregression claim.

Taken together, the unchanged binaries, non-reproducing large CPU deficit, same-owner control and longer GPU samples support accepting this timing screen at the measured repeatability. All unfavorable samples, activity imbalance and uncertainty remain. This is a branch comparison against base; the mx reference-speed gate remains separate at phase level.

Follow-up rates below are medians of the two calls per arm; percentages show the two directional pairs. They must not be pooled with the first block or with the different R10 workload.

| Group / model / device | Phase | Base | Candidate | Control | Candidate/base pairs | mx phase gate |
|---|---|---:|---:|---:|---|---|
| R1 / 0.6B Q8 / CPU | pp512 | 225.155 | 223.245 | 219.690 | -2.75%/+1.11% | Not rerun |
| R1 / 0.6B Q8 / CPU | tg128 | 46.055 | 46.910 | 38.615 | +0.64%/+3.09% | Not rerun |
| R1 / 0.6B Q8 / MI50 | pp512 | 8763.660 | 8766.745 | 8766.740 | -0.12%/+0.19% | Not rerun |
| R1 / 0.6B Q8 / MI50 | tg128 | 395.395 | 391.065 | 393.855 | -2.01%/-0.17% | Not rerun |
| R1 / 30B-A3B Q4_K_M / CPU | pp512 | 28.115 | 28.330 | 28.180 | +0.60%/+0.92% | Not rerun |
| R1 / 30B-A3B Q4_K_M / CPU | tg128 | 13.425 | 13.535 | 13.550 | +0.52%/+1.12% | Not rerun |
| R1 / 30B-A3B Q4_K_M / MI50 | pp512 | 1111.290 | 1127.545 | 1115.330 | +0.31%/+2.65% | Not rerun |
| R1 / 30B-A3B Q4_K_M / MI50 | tg128 | 126.445 | 126.470 | 126.725 | +0.19%/-0.15% | Not rerun |
| R10 / 0.6B Q8 / MI50 | pp512 | 8647.930 | 8712.125 | 8702.080 | -0.81%/+2.33% | Not rerun |
| R10 / 0.6B Q8 / MI50 | tg128 | 390.855 | 392.875 | 392.410 | -0.23%/+1.28% | Not rerun |

The rotated block has 505 observations/504 intervals and B/I/U/D/G totals 0/25/11/51/0; the R10 block has 32 observations/31 intervals and 0/0/1/0/0. Both have stop records and no unknown sampled CPU/disk counters. Per-call process misses, new/departed observations, GPU ranges and unknown coverage remain in the JSON and raw monitors. All 30 added calls have zero increments in cgroup memory max/OOM/OOM-kill counters; this is not evidence about the first round. Whole-call child statistics include loading, warmup and both phases, so they do not assign a fault or I/O event to decode.

The first 30B CPU control call in the rotated block records 1,591,749 major faults and 17,848,232 input blocks; the next candidate records 14 major faults and no input blocks, and subsequent CPU calls record zero of both. That distinguishes the recorded cold read from later cached reads in this block, not a retrospective diagnosis of the original slowdown. The six CPU calls use 222.25 to 237.77 seconds of child CPU time.

### All 24 rotated calls

Order per model/device is control/candidate/base/base/candidate/control. Flags use the first block's B/I/U/D/G definitions. Selected child usage, cgroup deltas, coverage unknowns and per-call timestamps are in the JSON; full counters remain in the hashed raw evidence.

| Model / device | Arm in order | pp512 | tg128 | B/I/U/D/G | Major faults | Input blocks |
|---|---|---:|---:|---|---:|---:|
| 0.6B Q8 / CPU | 1 control | 223.13 | 46.75 | 0/0/0/2/0 | 154678 | 2471073 |
| 0.6B Q8 / CPU | 2 candidate | 221.70 | 46.85 | 0/0/0/0/0 | 15 | 0 |
| 0.6B Q8 / CPU | 3 base | 227.98 | 46.55 | 0/0/0/0/0 | 0 | 0 |
| 0.6B Q8 / CPU | 4 base | 222.33 | 45.56 | 0/0/0/0/0 | 0 | 0 |
| 0.6B Q8 / CPU | 5 candidate | 224.79 | 46.97 | 0/0/0/0/0 | 0 | 0 |
| 0.6B Q8 / CPU | 6 control | 216.25 | 30.48 | 0/8/0/8/0 | 0 | 0 |
| 0.6B Q8 / MI50 | 1 control | 8765.38 | 397.26 | 0/2/0/2/0 | 17 | 15 |
| 0.6B Q8 / MI50 | 2 candidate | 8753.57 | 388.13 | 0/1/0/2/0 | 17 | 0 |
| 0.6B Q8 / MI50 | 3 base | 8764.24 | 396.10 | 0/0/0/1/0 | 127 | 0 |
| 0.6B Q8 / MI50 | 4 base | 8763.08 | 394.69 | 0/0/0/0/0 | 0 | 0 |
| 0.6B Q8 / MI50 | 5 candidate | 8779.92 | 394.00 | 0/0/0/0/0 | 0 | 0 |
| 0.6B Q8 / MI50 | 6 control | 8768.10 | 390.45 | 0/0/0/0/0 | 0 | 0 |
| 30B-A3B Q4_K_M / CPU | 1 control | 28.27 | 13.58 | 0/0/1/8/0 | 1591749 | 17848232 |
| 30B-A3B Q4_K_M / CPU | 2 candidate | 28.29 | 13.53 | 0/0/4/6/0 | 14 | 0 |
| 30B-A3B Q4_K_M / CPU | 3 base | 28.12 | 13.46 | 0/7/0/10/0 | 0 | 0 |
| 30B-A3B Q4_K_M / CPU | 4 base | 28.11 | 13.39 | 0/0/0/0/0 | 0 | 0 |
| 30B-A3B Q4_K_M / CPU | 5 candidate | 28.37 | 13.54 | 0/0/5/0/0 | 0 | 0 |
| 30B-A3B Q4_K_M / CPU | 6 control | 28.09 | 13.52 | 0/0/1/0/0 | 0 | 0 |
| 30B-A3B Q4_K_M / MI50 | 1 control | 1116.61 | 126.71 | 0/0/0/0/0 | 0 | 0 |
| 30B-A3B Q4_K_M / MI50 | 2 candidate | 1127.49 | 126.75 | 0/0/0/0/0 | 0 | 2219744 |
| 30B-A3B Q4_K_M / MI50 | 3 base | 1124.05 | 126.51 | 0/0/0/4/0 | 18 | 6933440 |
| 30B-A3B Q4_K_M / MI50 | 4 base | 1098.53 | 126.38 | 0/0/0/0/0 | 0 | 1185344 |
| 30B-A3B Q4_K_M / MI50 | 5 candidate | 1127.60 | 126.19 | 0/0/0/0/0 | 61 | 1138688 |
| 30B-A3B Q4_K_M / MI50 | 6 control | 1114.05 | 126.74 | 0/9/0/11/0 | 2548 | 2416688 |

### All six dense MI50 calls at ten repetitions

Each mean and SD is in tokens/s as printed by that call; SD is not a confidence interval. Order is base/control/candidate/candidate/control/base.

| Arm in order | pp512 mean | pp512 SD | tg128 mean | tg128 SD | B/I/U/D/G |
|---|---:|---:|---:|---:|---|
| 1 base | 8750.74 | 22.14 | 393.72 | 1.06 | 0/0/0/0/0 |
| 2 control | 8694.88 | 136.25 | 391.20 | 1.12 | 0/0/0/0/0 |
| 3 candidate | 8679.70 | 101.19 | 392.81 | 0.67 | 0/0/0/0/0 |
| 4 candidate | 8744.55 | 18.64 | 392.94 | 0.80 | 0/0/0/0/0 |
| 5 control | 8709.28 | 64.59 | 393.62 | 0.84 | 0/0/0/0/0 |
| 6 base | 8545.12 | 232.13 | 387.99 | 7.32 | 0/0/1/0/0 |

The additional evidence groups are local `followup/` and `gpu-repeat/` under `.tmp-vulkan-dispatch-run-20261002/`: each retains plan, complete results, analysis, full activity, per-call stdout/usage and final resource audit. Their raw hashes and the explicit decision record `performance-assessment.json` are pinned in the JSON. These are completed measurements, not replacement samples.

## Reproduction, failures and retained evidence

Rig evidence root: `/zpool1/llmx-xdev-validation/vulkan-weight-dispatch-20261002`. Local timing/identity evidence: `.tmp-vulkan-dispatch-run-20261002/`; Linux functional logs/manifests: `.tmp-agent-vulkan-weight-dispatch-20261002/linux-functional/`; Windows logs/manifests: `.tmp-vulkan-dispatch-windows-20261002/`. The rig retains `identity/`, `layout-control/` and `timing-layout/`; local flattened copies are named `identity-complete.json`, `control-complete.json`, `timing-plan.json`, `timing-complete.json` and `final-audit.json`. The report links their SHA256 identities rather than embedding large logits or the full activity analysis.

The initial control setup failed with return code 128 before building or measuring because its worktree metadata path was unavailable in that setup context. It remains in rig `layout-control-attempt-1/`, `control-launch-attempt-1.json` and `control-runner-attempt-1.log`. The later successful setup is separate. No timing sample was discarded. The first timing block stopped the owned measurement container at 18:22:33 local, as recorded in `final-audit.json`. The container restarted at 18:38:11 local for the follow-up. The final stop after the targeted GPU check is recorded separately at 18:50:46 local, with 0% GPU busy and 10,932,224 bytes of VRAM; the first-block audit describes only its own earlier stop.

The preceding checkpoint screened all 86 tracked Markdown files for the affected descriptor/kernel names and links. This documentation-only completion rechecks the changed record and carries forward that unaffected review. Removed helper names under dated historical findings remain historical. Source, tests, numerical bounds and publication state are unchanged by this record. The timing screen is accepted at measured repeatability. Current-main integration, required build/CTest refresh and exact-head hosted CI remain outstanding; the branch is not merged.
