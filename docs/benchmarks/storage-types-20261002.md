# Storage metadata gate evidence (2026-10-02)

**Correctness passes. The branch timing check is accepted with limited precision: the apparent CPU prefill loss is smaller than the measured spread between builds with identical benchmark code.** This does not prove universal nonregression or identify a performance gain. Hosted CI at the landing head remains a separate required check.

Candidate `8923c376` separates storage metadata/sizing from execution support. The measured main is `46f60b64`; the info-path control is `dc72bbd1`. No new decoding arithmetic or shader expressions are included. The [companion JSON](storage-types-20261002.json) retains full source, executable, tool, fixture, prompt-ID and capture hashes.

| Arm | Source commit | Linux executable SHA256 |
|---|---|---|
| base | `46f60b6491335b62785caf2d0647998864b0283b` | `870b994c30c7e8593fa1911e4897158b51dd5b6f476371d5f95971a1c2a7ced0` |
| candidate | `8923c3762af9cb4befaf4a7eda00365f4b23da62` | `be11f0a844d1459d18c2848c2a528e7b573aac20dc1504fc82acfa5a8056f798` |
| control | `dc72bbd108d59fea1103738e9078aacdea414e42` | `7ace0daf963d0563aeae9015145fa46bed8d3e8d94a8de8e4972bf1e1260c475` |

Each version is `llmx 0.1.0+g` followed by its first 12 commit characters. The matching clean detached builds use CMake Release with Vulkan enabled in image `sha256:2787625f463391c440798579bac0d012c77fdfa949d086d036cdbd77f0b8d600`. Runs use CPU0-3, four-CPU quota and a 16 GiB memory/swap cap. Vulkan device 0 is MI50/RADV, PCI `0000:03:00.0`, renderD136, UUID `00000000-0300-0000-0000-000000000000`, Mesa 25.0.7-2+deb13u1; the timing preflight requires automatic clocks.

| Completed check | Result |
|---|---|
| Windows CPU | 38/38 native tests; 25/25 suite components |
| Linux / MI50 | 42/42 native tests; 25/25 device suite components |
| Current-main model identity | 12/12: six files on CPU and Vulkan; all batched/decode float captures, 64 greedy rows/IDs and completed path reports equal |
| Compiled shader identity | 99/99 modules byte-identical |

Both full suites require the pinned fixtures, tools and all eight implemented weight types. Identity files are Qwen3-0.6B Q8_0, Q4_0, Q5_K_M and Q4_K_M, plus Qwen3.5-0.8B Q8_0 and Q4_K_M. Identity uses F16 caches, AUTO activation precision and microbatch 16; each comparison is within one backend. The Windows suite result pins its command and times but not an executable digest; the JSON labels the available current digest as a post-run observation.

Timing uses Qwen3-0.6B Q8_0 (`9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`), `--threads 4 --p 512 --n 128 --r 1 --ubatch 512 --cache-type-k f16 --cache-type-v f16 --dtype auto`. Rates are tokens/s. All 14 planned calls returned zero and remain below; no slow or active sample was discarded. The later control block is separate from the initial block. No new mx measurement was run for this branch screen.

| Block | Device | Arm (run order) | pp512 | tg128 | Activity U/B/D / intervals |
|---|---|---|---:|---:|---|
| initial | cpu | base | 220.03 | 30.12 | 14/0/0 / 15 |
| initial | cpu | candidate | 217.58 | 29.17 | 15/0/0 / 15 |
| initial | cpu | candidate | 218.75 | 30.11 | 15/0/0 / 15 |
| initial | cpu | base | 227.41 | 30.58 | 14/0/1 / 14 |
| initial | vulkan:0 | base | 8765.32 | 395.68 | 2/0/1 / 2 |
| initial | vulkan:0 | candidate | 8755.38 | 378.27 | 3/0/0 / 3 |
| initial | vulkan:0 | candidate | 8777.92 | 393.12 | 2/0/0 / 2 |
| initial | vulkan:0 | base | 8763.89 | 389.67 | 2/0/0 / 2 |
| control | cpu | base | 208.84 | 43.20 | 7/0/0 / 13 |
| control | cpu | candidate | 201.32 | 42.07 | 4/4/0 / 12 |
| control | cpu | control | 214.20 | 39.61 | 5/6/0 / 13 |
| control | cpu | control | 209.41 | 39.67 | 6/6/0 / 13 |
| control | cpu | candidate | 208.33 | 42.86 | 6/9/0 / 12 |
| control | cpu | base | 210.93 | 42.00 | 5/8/0 / 12 |

U/B/D count whole-call intervals flagged for an unrelated process using at least one CPU, host idle below 25%, and a whole disk exceeding 50 MB/s. Flags also cover iowait above 5% and sample gaps above 2.5 seconds; neither occurred. Boundary intervals can belong to neighboring calls, so per-call counts must not be summed as unique observations.

| Block / device | Phase | Main median | Candidate median | Control median | Candidate/main paired change | mx target |
|---|---|---:|---:|---:|---:|---|
| initial / cpu | pp512 | 223.720 | 218.165 | - | -2.46% | not remeasured |
| initial / cpu | tg128 | 30.350 | 29.640 | - | -2.35% | not remeasured |
| initial / vulkan:0 | pp512 | 8764.605 | 8766.650 | - | +0.02% | not remeasured |
| initial / vulkan:0 | tg128 | 392.675 | 385.695 | - | -1.76% | not remeasured |
| control / cpu | pp512 | 209.885 | 204.825 | 211.805 | -2.42% | not remeasured |
| control / cpu | tg128 | 42.600 | 42.465 | 39.640 | -0.28% | not remeasured |

Percentages are medians of forward/reverse matched-pair changes, not ratios of rate medians. In the control block, control/main is +0.92% prefill and -6.93% decode; candidate/control is -3.26% and +7.13%. The control changes info-only CLI code and has identical Q8 CPU captures. Full `.text` comparison attributes all 5,309 different bytes to `main`, `main.cold`, CRT startup/teardown stubs and alignment padding. All 22 benchmark-related symbols have identical addresses, sizes and bytes; model and backend function bytes are unchanged too. This is a repeat with the same benchmark machine code, not a hot-code layout perturbation. Startup code differs, and ELF data sections were not compared, so it is not a claim that every executed instruction or initial state is identical.

**Assessment:** retain the negative candidate/main prefill result in every pair. The candidate/control prefill pairs are -6.01% and -0.52%, while decode is +6.21% and +8.04%, despite identical benchmark functions. That repeatability cannot distinguish the approximately 2.4% main comparison from run/initial-state variation. Accept this metadata change's timing screen at that precision, without claiming a layout band, exact nonregression, a causal attribution to background activity or reference parity. No control-only code is included in the feature. Base-to-candidate inspection also shows the same normalized 1,237-instruction Q8 prompt worker at a different address; the indexed row-size lookup does change executed code, so source-level arithmetic identity alone was not used as the timing verdict.

The initial monitor retained 63 samples/62 intervals; the control monitor retained 72/71. Both stop records are present. Background CPU work and unequal activity remain in the results. Observations cover loading and both phases together; phase-specific overlap and GPU process attribution are unavailable. New, departed, inaccessible and short-lived processes limit coverage; missing observations are not zero activity. Disk devices are not summed across storage layers. Detailed flags, GPU ranges and coverage counts remain in the JSON and original analyses.

Prior harness attempts remain recorded: missing NumPy stopped before builds; an earlier Windows stdout comparison mistakenly included timing text and was corrected using the existing text/ID parser; a declaration-order control emitted identical code and was never timed; and an extra `nm` argument was corrected without rebuilding the final control. These are retained harness/setup records, not discarded performance samples.

Original evidence is under `/zpool1/llmx-xdev-validation/storage-types-20261002`: `functional-complete.json`, `identity/complete.json`, `binaries.json`, `shader-hashes.json`, `fixture-inventory.json`, both `timing-analysis.json` files, and `layout-control-info/verified.json`. Windows evidence is under `.tmp-storage-types-run-20261002`: `windows-suite-result.json`, `windows-suite.log` and `rebase-ctest.log`. The companion JSON hashes every retrieved record and retains all 14 calls and their log paths.

## Integration refresh onto split-hold main

The peer feature landed at `58a696ce` while the storage hosted gate was running. Storage rebased to `ae20c5d6` with no code conflict; only the two new STATUS entries needed combining. Its source/test/build patch has the same stable patch ID as before the rebase. The obsolete hosted run was cancelled, not treated as a failure or a passing landing gate.

Fresh Windows and Linux Vulkan builds pass 38/38 and 42/42 native tests respectively. All 99 shader modules remain byte-identical. The companion JSON records both new executable hashes and build/test commands. Existing full-suite, model-identity and timing results stay scoped to their original commits; they are retained under the project rebase rule rather than relabeled as measurements of this build. Final documentation-only amendments require their own exact-head hosted pass before main advances.
