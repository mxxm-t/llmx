# Two-row decode builds for the Q4 and K-quant rows, 2026-09-29

The second route of the [server investigation](../server-investigation-20260929/README.md)'s first ranked fix: decode builds of several columns for the Q4_0, Q4_1, Q4_K, Q5_K and Q6_K row kernels, the Q6_K and Q4 output heads included, with every column's bits those of the one-column build.
Measured first with a probe on an 8B pass's decode matmuls, then built on branch `perf/kquant-decode-columns` and served against main on the same MI50s.

**In short:**

- Two adjacent rows a cluster, with every column a build holds computed, took an 8B pass's decode matmuls at 64 columns from 268 to 195 ms on Q4_K_M, 351 to 261 on Q6_K and 254 to 156 on Q4_0 on one MI50; one column is unchanged (the one-column build compiles to the same instructions as main's).
- Served, one MI50, 16 to 64 users: Q4_K_M 183 to 228-231 tok/s (+25 to +26 percent), Q6_K 144 to 177-178 (+23 to +26), Q4_0 190 to 265-268 (+39 to +42); two MI50s +22 to +39 percent. One user is level.
- Every column is main's bits: layer 0's tensors and the head of each file give main's digests in every build, `backend-vulkan` checks every decode column against the column alone, and the CLI and served outputs are byte for byte main's (the branch's STATUS block lists the gates).
- What limited these rows was each column's own activation loads, their wait and a branch before the next column, not memory traffic: reading every column from one address, which hits the cache, cost the same, and staging activations in shared memory changed nothing. Three and four rows a cluster ran two waves a SIMD and lost, and so did issuing column loads in groups.
- Under the AMD proprietary driver on the Radeon VII a first form of these builds changed every build's bits, the one-column one included, and cost one column 25 percent; the builds are a separate path and only the MI50 under RADV takes them (`row_decode_cols`).

## What ran

- **llmx:** main `660b0aed` with the probe tool added as one commit (`llmx 0.1.0+gdf6d21fa2506`, sha256 `5361ec33...9889`), against the branch's kernel commit on top of the same probe commit (`llmx 0.1.0+g2dd823013f3a`, sha256 `b1982d34...928b`), whose `src/` equals the branch head `f35c9103`; both built fresh in the build image with Vulkan, Release, the same way. `serve --max-seqs 64 --threads 6`, one pass in flight per stage.
- **Probe:** `llmx-kq-probe` ([`probe-src/probe2.patch`](probe-src/probe2.patch), never merged): `time` runs an 8B pass's decode matmuls (each layer's q, k and v grouped, o added, gate and up grouped, down added, then the head) with decode rows of extent 1 at each column count, ten passes after a warm one; `check` compares every column of calls of 1 to 64 columns with the column alone on layer 0's tensors and the head, grouped as the model groups them, and prints a digest of the columns alone to compare builds. The prototype trees are [`probe-src/proto7-full.patch`](probe-src/proto7-full.patch) to [`proto9-full.patch`](probe-src/proto9-full.patch), whose environment variables chose rows, columns, column groups, shared-memory staging and a branchless column loop.
- **Hardware:** MI50s (gfx906) under RADV on Linux, clocks held high on the cards in use and restored to auto after; the Radeon VII under the AMD proprietary driver on Windows for the bits and timing of the builds it keeps.
- **Models:** Qwen3-8B Q4_K_M (lmstudio-community), Q6_K and Q4_0 as in the server investigation (its `models.sha256`); Qwen3-0.6B Q4_K_M, Q4_0 and Q5_K_M (unsloth) for identity.
- **Load:** `tools/server_load.py` from main, closed loop at 1, 8, 16, 32 and 64 users, 128-token prompts and replies with the end of text ignored, greedy, one warm-up request, two rounds a level; each arm's server twice in the order main, branch, main, branch on one card (one MI50) or two (a layer split), the reference figures the server investigation's best of any setting.

## Decode matmuls of an 8B pass, one MI50

ms a pass, both rounds in [`micro-tables.md`](micro-tables.md):

| model | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| Qwen3-8B Q4_K_M, main / branch | 9.1 / 9.2 | 12.8 / 11.7 | 19.6 / 15.7 | 33.8 / 24.7 | 67.2 / 49.0 | 134.4 / 97.8 | 268.3 / 195.3 |
| Qwen3-8B Q6_K, main / branch | 15.6 / 15.6 | 19.7 / 19.9 | 27.4 / 25.1 | 44.1 / 35.9 | 87.9 / 65.5 | 175.7 / 130.6 | 351.3 / 260.9 |
| Qwen3-8B Q4_0, main / branch | 7.8 / 7.8 | 11.0 / 9.4 | 17.9 / 12.9 | 31.9 / 21.7 | 63.5 / 39.4 | 126.8 / 78.4 | 253.6 / 156.4 |

The prototype's variants, one MI50, ms at 8 and 64 columns ([`prototype-tables.md`](prototype-tables.md)); the kernel's time at 8 columns on Q4_K_M is the eight-column build's `matmul_row_k4_dot` (timestamps):

| variant | Q4_K_M 8 / 64 | Q6_K 8 / 64 | Q4_0 8 / 64 | Q4_K kernel at 8 |
|---|---|---|---|---|
| main | 33.8 / 268.5 | 44.3 / 351.1 | 31.9 / 253.7 | 22.8 |
| two rows | 28.3 / 225.3 | 39.1 / 312.0 | 26.8 / 212.0 | 16.3 |
| three rows | 28.7 / 226.1 | 49.0 / 389.4 | 21.5 / 170.4 | - |
| four rows | 45.5 / 362.0 | 57.6 / 457.8 | 31.2 / 247.7 | - |
| every column's activations from one address | - | - | - | 22.5 |
| activations staged in shared memory, two rows | - | - | - | 15.8 |
| column loads issued four at a time, two rows | 32.1 / 254.7 | 48.4 / 383.9 | 28.9 / 228.5 | - |
| two rows, no branch in the column loop (the branch's form, 8 columns) | 24.8 / 196.2 | 36.0 / 285.8 | 21.4 / 169.6 | 14.7 |
| the same, 16 columns | 41.0 / 193.2 | 57.5 / 260.6 | 34.6 / 156.6 | - |

## Served, tok/s

Best round of each arm's two servers; every round in [`server-tables.md`](server-tables.md):

| model, cards | users | main | branch | branch / main | reference best (2026-09-29) |
|---|---:|---:|---:|---:|---:|
| Q4_K_M, one MI50 | 1 / 8 / 16 / 32 / 64 | 89 / 179 / 183 / 183 / 182 | 89 / 225 / 228 / 231 / 229 | 0 / +26 / +25 / +26 / +26% | 69 / 164 / 96 / 158 / 170 |
| Q6_K, one MI50 | 1 / 8 / 16 / 32 / 64 | 57 / 142 / 144 / 143 / 141 | 57 / 165 / 177 / 178 / 177 | 0 / +16 / +23 / +24 / +26% | 61 / 145 / 152 / 230 / 260 |
| Q4_0, one MI50 | 1 / 8 / 16 / 32 / 64 | 99 / 186 / 190 / 189 / 190 | 100 / 243 / 266 / 268 / 265 | +1 / +31 / +40 / +42 / +39% | 77 / 191 / 235 / 273 / 262 |
| Q4_K_M, two MI50s | 1 / 8 / 16 / 32 / 64 | 87 / 271 / 318 / 326 / 330 | 87 / 305 / 387 / 407 / 415 | 0 / +13 / +22 / +25 / +26% | 67 / 169 / 98 / 160 / 162 |
| Q6_K, two MI50s | 1 / 8 / 16 / 32 / 64 | 57 / 208 / 255 / 261 / 263 | 57 / 220 / 295 / 322 / 326 | 0 / +6 / +16 / +23 / +24% | 62 / 158 / 160 / 244 / 292 |
| Q4_0, two MI50s | 1 / 8 / 16 / 32 / 64 | 94 / 286 / 330 / 339 / 344 | 97 / 346 / 421 / 469 / 478 | +3 / +21 / +28 / +38 / +39% | 79 / 201 / 235 / 278 / 257 |

- Inter-token p50 at 64 users falls with the throughput: one MI50 Q4_K_M 295 to 221 ms, Q6_K 386 to 292, Q4_0 281 to 184.
- Against the reference on one MI50 the branch now leads Q4_0 at 64 users (265 against 262) and trails it by 2 percent at 32; Q6_K still trails at 32 and 64 users (178 and 177 against 230 and 260), the gap the tile route of the server investigation would close.
- The prototype served on cores 4-7,12-15 while other lanes' builds held them at 100 percent gained 5 percent where `bench --seqs 64` gained 36 (Q4_K_M, 220.6 to 299.4 tok/s): a faster pass leaves the host less time, so those servers' figures are void; the ones above ran on idle cores (0-3,8-11 for one MI50's Q4_K_M and Q6_K, 4-7,12-15 for the rest).

## The Radeon VII

- The first form, under the AMD proprietary driver: Qwen3-8B Q4_K_M decode matmuls 14.7 / 74.6 / 596.7 ms at 1 / 8 / 64 columns on main against 18.5 / 68.8 / 548.6, and 5603 of 5655 checked columns off main's bits. So the branch keeps the one-row code as it was and the Radeon VII's profile does not set `row_decode_cols`.
- The branch there: `backend-vulkan --isa` passes, layer 0's and the head's digests equal main's on Q4_K_M, Q4_0 and Q6_K, and the matmuls are level (Q4_K_M at 64 columns 582 to 585 ms on both). Records in `radeon-vii.tar.xz`.

## Activity during the runs

The flags of [`scripts/FLAGS.txt`](scripts/FLAGS.txt) were fixed before the first timed run; [`flags-p1.txt`](flags-p1.txt) and [`flags-p2.txt`](flags-p2.txt) give every timed server level its samples, the host's busy range, its flags and the busiest processes outside this work.
Of 120 levels 68 carry the CPU flag and 36 the disk flag (the other arms' and tracks' model loads); other tracks' tests held cores 4-7,12-15 while one MI50's Q4_K_M and Q6_K ran on 0-3,8-11. Nothing was stopped or dropped; the arms of each comparison ran on one card in the order main, branch, main, branch, and the two runs of each arm agree within a few percent.

## Files

- [`micro-tables.md`](micro-tables.md), [`prototype-tables.md`](prototype-tables.md), [`server-tables.md`](server-tables.md): the tables above with every round.
- `micro.tar.xz`: every probe run's output (the checks with their digests, the timings, the per-kernel times), including the prototype variants; `server.tar.xz`: the load tool's tables, level windows, health and progress of every server; `radeon-vii.tar.xz`: the Radeon VII checks and timings; `monitor.jsonl.xz`: the monitor's samples. SHA-256: micro `c3fde628200e0b2fa4509bea1131070a3311b5defaeb675ea5d075f073a37fad`, server `2f9446a242b12bf9438efca461194df35218d79c44f78b1d4523d3f730d5d1e6`, monitor `c5d816173b2471fe522763aa2a67c0c4f52382e66ab2480c9ad5492e83531d13`.
- [`scripts/`](scripts/): the monitor, the flags, the build, probe, server and gate scripts, and the tables and float-operation counts.
- [`probe-src/`](probe-src/): the probe and the prototype trees as diffs against `660b0aed` (the probe) and `fcff27ec` (the prototypes).
