# Restore-host submission drain gate (2026-10-17)

Measured baseline `2a0221627cf6bb8791bd7b951ac370cc72789040` and fix `c898d417728e6f4fceb80356b58cf17705de5c55`. The final integration onto docs-only main `3b70f5e4f48f129b028ecd2c8d261819d5458552` preserves the measured runtime and test changes and requires its own build, CTest and hosted CI before landing. This record does not qualify that final binary.

## Measured identities

Both arms used fresh detached Linux worktrees, Release Vulkan, the same configuration and image `sha256:2787625f463391c440798579bac0d012c77fdfa949d086d036cdbd77f0b8d600`.

| Arm | Actual version | CLI SHA256 |
| --- | --- | --- |
| base | `llmx 0.1.0+g2a0221627cf6 numerics 52aa97b4761d52c8` | `10df3150ffbe003a7d9baef99bb7c33af9af9ce8c4387c3636c59a843c56b7ec` |
| fix | `llmx 0.1.0+gc898d417728e numerics 888d8972bfd4898b` | `153ec0266feca90de5146c4333d4b2042089d2c1b31afb64abbd2c31ec826506` |

Model-logits tool SHA256: base `5771d4f67ab15afa79c8e4bd2ebd34bed178d45d9fa47d398b89265e22d78faf`, fix `8149ca639eacdbfdd1a9ae27bcec63fe2560ee313e16ad2660698c1e4641ca79`. Both backend archives were byte-identical, SHA256 `44078f06a6083805ca17fbdfd7fa4876a4928664b42789b33dfa3a33a152cf08`; generated configuration SHA256 `cffa6da5cbbeb9d258e8ed6d4505d16a512511ae681cc798d3afb30dea978749`. Source worktrees remained clean and all six binary/archive hashes were unchanged after the gates.

## Correctness and lifetime coverage

- The test-first build failed with "a failed restore submission did not drain borrowed host copies before returning its slot"; the fixed architecture test passed 808 checks. It injects a submit failure on each member of a two-device split and verifies drain, slot return, retry and continuation logits.
- Linux candidate CTest: 53/53, 887.71 s. Actual Vulkan checks ran; the second-device collective subcase was explicitly skipped because only physical GPU 0 was exposed. Quantization also reports its named cases skipped without float preservation.
- Candidate full device suite: all 28 components passed, 572.20 s. Dense Qwen3-0.6B Q8 follow-ups promoted host donors, reused 960 tokens and matched CLI greedy text; the existing check requires at least two promotions and positive bytes moved. The synthetic hybrid pause/resume check requires an actual pause and promotion and exact replies versus running alone.
- Matched identity: eight dense Qwen3-0.6B Q8 / hybrid Qwen3.5-0.8B Q8 cells over CPU and GPU, each with F32 and F16 caches, passed in 235.14 s. Complete 247-row batched and token-by-token logits, 64-row greedy logits and greedy IDs matched byte for byte between arms. The tool uses six threads constrained to CPU 0/1. These ordinary model-output cells do not themselves force restore; the reached restore tests provide that witness.
- Windows at c898: fresh MSVC build; CTest 49/49 in 715.86 s. The CPU suite initially passed 26 components and failed `cli` and `roundtrip` under the sandbox. Both failed components passed when rerun unrestricted without source changes. A retained direct CLI contrast produced filesystem error 5 in the sandbox and the expected missing-file error unrestricted; this is not a claim that the first full invocation passed.
- [Hosted run 39012261051](https://github.com/mxxm-t/llmx/actions/runs/39012261051) passed all seven jobs at c898. Final-head hosted status must be checked by that final commit, not inferred from this run.

A redundant Linux baseline CPU suite was deliberately stopped after scope review (container exit 143, not OOM); no whole-suite pass is claimed. No broad fix CPU or baseline device suite was added.

## Matched timing

One ABBA block per model on physical GPU 0 (PCI `0000:03:00.0`), CPU 0/1, F32 cache: `bench --model MODEL --device vulkan:0 --p 512 --n 128 --r 1 --threads 2 --ubatch 128 --cache-type-k f32 --cache-type-v f32`. All eight invocations were retained; total container time 13.80 s. Values below are tok/s in actual execution order.

| Model | Phase | Base A1 | Fix B1 | Fix B2 | Base A2 | Mean fix/base change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B-Q8_0.gguf | pp512 | 5619.50 | 5616.49 | 5613.35 | 5612.17 | -0.0163% |
| Qwen3-0.6B-Q8_0.gguf | tg128 | 401.59 | 402.65 | 401.70 | 396.85 | +0.7402% |
| Qwen3.5-0.8B-Q8_0.gguf | pp512 | 5271.67 | 5272.82 | 5268.67 | 5273.07 | -0.0308% |
| Qwen3.5-0.8B-Q8_0.gguf | tg128 | 351.00 | 343.49 | 349.61 | 342.06 | +0.0058% |

Twelve timestamped host/CPU/disk/process/GPU snapshots were retained with the same monitor in both arms. Peer physical GPUs 4-7 were busy throughout; root Flash jobs were terminal. No build process was observed, and sampled non-llmx CPU peaks stayed below 0.25 core (maximum 0.109). Disk activity was retained. Sampling can miss short-lived activity and does not fully attribute IO or unknown llmx processes. No sample was discarded or replaced.

The small differences are one timing round, without a layout perturbation control; they establish no performance improvement or universal regression bound. Bench covers ordinary prefill/decode, not the intentional restore-submit exception path. Linux containers used `--memory 4g --memory-swap 4g` (4 GiB total cap), fresh resource guards, no network and only owned containers; every owned container was removed, with no OOM.

## Retained evidence

Rig evidence remains under `/zpool1/llmx-xdev-restore-submit-20261017`; local small records remain under `.tmp-restore-submit-gate-20261017`. Large identity binaries stay on the rig. The compact gate index SHA256 is `cf53ada7c14ca76be5e148e82981016d06caf8d8e9fb3a5e0a3e6203a9204407`.

| Record | SHA256 |
| --- | --- | --- |
| `ctest/summary.json` | `640c34d853271bd25ee61567a4081ab888298f01a5c11a768b832aeeffc8f95d` |
| `device-fix/summary.json` | `b10821b5fb91a6899dc06ee1176f72b7088520e4340774906dd7ab0c7033535c` |
| `identity/summary.json` | `9bfa83785907e4d85854bd92e94234c56b9071bc25a61503d0f0528af8ffe9d4` |
| `timing/summary.json` | `c5cd107d1d65809bfacb0b8faaea8f45970d2f1eb944a10a8a53777e8711fa66` |
| `cpu-base/cancelled-by-scope-review.json` | `f9878978f7ab5426b81b88ea0bca3fbf9360c22c19ae324cebb8b8cef4d5d7ec` |
| Windows `final-windows-ctest.log` | `09b054b2c36ad82f48b5bc678a56a7099da9a3a105fa5b694c1aed3f604f8d9f` |
| Windows `final-windows-cpu-suite.log` | `c14b4da99147b5833281e42c7aa4e1e5adecea300aa66032cfe00bc2e80650f2` |
| Windows `final-windows-cpu-suite-failed-components-escalated.log` | `7406d5faa3e3ffa66a9559cb2ae021a2b6334a45742d0b0ce4b24b2ba3086a65` |
| Windows `windows-cli-sandbox-contrast.json` | `c7443fc00e7a98a9327409313e7f9584bd81d740efe71d896823a933a4a05926` |

Final landing remains a fast-forward only after the integrated head passes its required fresh build, CTest and hosted CI. No new arithmetic, model qualification or successful final-head gate is claimed here.
