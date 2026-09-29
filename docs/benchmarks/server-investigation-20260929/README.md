# Server speed investigation, 2026-09-29

Measurement only: no runtime change and no merge.
The question was why the server's throughput at 16 to 64 users is low on every quant type, and whether the server, the scheduler or the backend is the cause.
Measured on main `6147753a`, built fresh, against mx-llama.cpp's ROCm server on the same MI50s, with Qwen3-8B in Q8_0, Q4_K_M, Q6_K and Q4_0.

**In short:**

- **The backend, not the server or the scheduler.** One MI50 is 93 to 97 percent busy from 16 users on, on every type; the host exposes 3 to 7 percent of the wall. The decode matmuls with the head are 78 to 93 percent of a 64-row decode pass.
- **The decode row kernels of the 4- and 6-bit types do not scale.** Q4_K, Q6_K and Q4_0 rows cost 4.0 to 5.5 ms a pass for every row past the first few (Q8_0's decode kernel 1.4), and their head is Q6_K at 0.6 ms a row, so those files serve 140 to 190 tok/s from 8 users on, however many users come. That is where llmx trails the reference on one card: Q6_K by 38 and 46 percent at 32 and 64 users, Q4_0 by 21 to 32 percent at 16 to 64. On Q8_0 and Q4_K_M llmx leads or is level on one card (+1 to +55 percent, +7 to +89), and on the two-card split it leads on every type from 8 users on but Q6_K at 64 users.
- **The prompt tile keeps a decode row's bits whatever shares its call**, and at 64 rows costs 56 to 70 ms a pass against the row kernels' 94 (Q8_0) to 351 ms (Q6_K). Served with decode rows through it, one MI50 at 64 users goes from 177 to 333-340 tok/s on Q4_K_M, 140 to 298-300 on Q6_K, 180 to 329-336 on Q4_0 and 357-361 to 431-437 on Q8_0, and the server's batch-invariance check (ids and log-probabilities alone, together and skewed) passes on Q8_0 and Q4_K_M. At one user it loses 69 to 80 percent, and 46 to 73 percent with the fastest row-shaped kernel below.
- **The missing piece is a row-shaped kernel with the tile's arithmetic at the row kernels' speed.** Five such kernels were written; every one gives the tile's bits on every type, so a call could switch between it and the tile by row count without any row depending on the batch, but the best is still 1.9 to 4.1 times slower than today's row kernels at one row.
- **On two MI50s** stage 1 runs the head (19.5 ms of a 78 to 101 ms stage for the Q6_K heads at 32 rows a stage) while stage 0 idles 20 to 24 percent; moving two layers to stage 0 gained 3 to 5 percent on Q4_K_M and lost on Q8_0. Receive waits are backpressure, not withheld work, and P = S + 1 served 7 to 16 percent less than P = S.

## What ran

- **llmx:** main `6147753a` built fresh in the build image with Vulkan (`llmx 0.1.0+g6147753a24f5`, sha256 `914ce2813e9f907841e2997492c7f705c83b68b25952cd87b0f349d55ebbe5b0`), `serve --max-seqs 64 --threads 6` at its default passes (P = S), no `--timing`: every llmx throughput in the comparison comes from it.
  The traced arms are the same commit with a probe patch ([`probe-src/probe.diff`](probe-src/probe.diff), `llmx 0.1.0+g6147753a24f5.dirty`) that timestamps every dispatch against the host's clock and records the scheduler thread's intervals, run with `--timing`; they served within 2 percent of the untimed probe and of main at 64 users (Q8_0 on one MI50: 365 against 370 and 362 to 373 tok/s), so the breakdowns describe the untimed server.
  The decode-tile experiments are further probe trees of the same commit ([`probe-src/`](probe-src/), one diff per tree); nothing of them is on any branch.
- **Reference:** mx-llama.cpp's ROCm image `gfx906-b10951-eefc4e732` (`llama-server` 0.3.0-dev build 10951, commit `eefc4e732`), `-ngl 99 -fa 1 -lm dio`, `-sm layer` on two cards, `-np` equal to the users with 1152 tokens of context a slot and a server per level, its batch and ubatch swept at 16 to 64 users (`-b 2048 -ub 512`, its default, `-b 4096 -ub 1024` and `-b 1024 -ub 256`), and the setting of 2026-09-28, `-np 64` at every level, run again beside them.
  Each server's container mapped only its cards' render nodes, with `HIP_VISIBLE_DEVICES` naming them; the witness runs below show the devices it used by PCI address.
- **Hardware:** MI50s (32 GB, gfx906) under RADV (Mesa 25.0.7) on Linux 6.17, clocks held high on the cards in use and restored to auto after: one card for Q8_0 and Q4_K_M, one for Q6_K and Q4_0, two for the split, one for the traced arms, and one (on loan) for the microbench and the decode-tile servers. Every comparison's arms ran on one card, interleaved.
- **Models:** Qwen3-8B Q8_0 and Q6_K (Qwen's repository, revision `7c41481f`), Q4_K_M (lmstudio-community, `07ebe812`: Q4_K with Q6_K in half of `attn_v` and `ffn_down` and in the head) and Q4_0 (bartowski's conversion, `0b69f75b`: `ffn_down` in Q4_1, the head Q6_K); the four files' SHA-256 are in [`models.sha256`](models.sha256).
- **Load:** `tools/server_load.py` from main on both servers, closed loop at 1, 8, 16, 32 and 64 users, 128-token prompts and 128-token replies with the end of text ignored, greedy, one warm-up request, two rounds a level; each figure is a server's best round, both rounds in [`summary.txt`](summary.txt). A third interleaved round at 16 to 64 users ran later with fewer of this investigation's lanes beside it.
- **Microbench:** the probe tool (in the probe diffs) times an 8B pass's matmuls, every layer's projections grouped as the model groups them and the head, per column count, ten passes after a warm one, through the decode path (decode rows, extent 1), the prompt tile (extent 64) and the decode-tile kernels, and checks their bits.

## 1. Where the time goes

From the traced arms: the decode pass's device time by class (matmul, the head, attention with the KV writes, the rest) for the most common decode pass of the level, the share of all device time the passes with prompt rows took, each device's busy share, its idle time by cause, the round period, and a decode pass's critical path (formed to its logits read) against its dispatches summed.
Busy is the union of a device's dispatch intervals; idle is attributed to the host activity covering it (recording a stage, the sampling pool's run, stepping the requests, the logits wait) or, on a split, to the previous stage still running.
On a split each decode pass holds half the users' rows, the two stages side by side.


#### Qwen3-8B Q8_0, one MI50

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 64 | 1 | 10.6 + 0.9 + 1.3 + 0.9 = 13.7 | 6% | 91% | 9%: recording 4.2, pool run 1.5, logits wait 1.0 | 15.8 | 14.7 / 13.7 |
| 8 | 274 | 8 | 14.2 + 1.1 + 2.9 + 1.1 = 19.3 | 25% | 87% | 13%: recording 4.9, pool run 4.7, logits wait 0.9 | 29.2 | 21.2 / 19.3 |
| 16 | 338 | 16 | 23.1 + 1.8 + 4.9 + 1.4 = 31.2 | 31% | 93% | 7%: pool run 3.1, recording 1.8, step 0.7 | 45.8 | 32.0 / 30.7 |
| 32 | 367 | 32 | 40.6 + 3.3 + 9.5 + 1.7 = 55.1 | 36% | 94% | 6%: pool run 3.4, recording 1.3, step 0.6 | 81.7 | 55.3 / 53.7 |
| 64 | 365 | 64 | 81.1 + 6.8 + 22.0 + 2.4 = 112.3 | 38% | 93% | 7%: pool run 3.2, thread idle 1.3, step 0.6 | 163.1 | 111.0 / 109.0 |

#### Qwen3-8B Q8_0, a layer split over two MI50s

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 59 | 1 | 5.3 + 0.0 + 0.7 + 0.5 = 6.5; 5.3 + 0.9 + 0.7 + 0.5 = 7.3 | 6% | 39% / 44% | 61%: logits wait 39.6, recording 12.3, wait ticket 4.5 ; 56%: previous stage 47.6, wait ticket 3.3, recording 3.2 | 8.8 | 16.4 / 13.8 |
| 8 | 267 | 4 | 5.9 + 0.0 + 1.0 + 0.5 = 7.4; 5.9 + 0.9 + 1.0 + 0.5 = 8.3 | 17% | 56% / 61% | 44%: recording 21.7, wait ticket 8.2, pool run 8.2 ; 39%: previous stage 23.6, wait ticket 8.0, recording 4.7 | 15.8 | 25.6 / 15.6 |
| 16 | 445 | 8 | 7.1 + 0.0 + 1.4 + 0.5 = 9.1; 7.1 + 1.1 + 1.4 + 0.6 = 10.1 | 26% | 67% / 73% | 33%: recording 16.2, pool run 7.9, logits wait 3.3 ; 27%: previous stage 19.9, recording 3.5, wait ticket 1.4 | 17.8 | 25.4 / 19.1 |
| 32 | 543 | 16 | 11.6 + 0.0 + 2.4 + 0.7 = 14.7; 11.6 + 1.7 + 2.4 + 0.7 = 16.4 | 30% | 70% / 76% | 30%: recording 12.4, pool run 7.0, logits wait 2.8 ; 24%: previous stage 15.9, recording 2.8, wait ticket 2.2 | 29.1 | 39.8 / 30.8 |
| 64 | 662 | 32 | 20.0 + 0.0 + 4.7 + 0.8 = 25.6; 20.3 + 3.3 + 4.7 + 0.8 = 29.2 | 36% | 78% / 86% | 22%: recording 7.5, pool run 5.7, step 2.7 ; 14%: previous stage 8.9, recording 2.0, thread idle 1.4 | 47.6 | 62.3 / 54.1 |

#### Qwen3-8B Q4_K_M, one MI50

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 76 | 1 | 7.5 + 1.2 + 1.3 + 0.9 = 11.0 | 7% | 89% | 11%: recording 5.4, pool run 1.6, logits wait 1.4 | 13.2 | 12.2 / 11.0 |
| 8 | 168 | 8 | 28.9 + 4.7 + 2.9 + 1.1 = 37.6 | 16% | 89% | 11%: recording 4.9, pool run 2.9, logits wait 1.1 | 48.1 | 39.6 / 37.0 |
| 16 | 176 | 16 | 57.3 + 9.5 + 4.9 + 1.4 = 73.0 | 17% | 95% | 5%: pool run 1.9, logits wait 0.9, recording 0.8 | 89.1 | 74.4 / 72.4 |
| 32 | 180 | 32 | 114.0 + 19.1 + 9.4 + 1.7 = 144.2 | 20% | 96% | 4%: pool run 1.6, step 0.8, recording 0.4 | 166.4 | 141.7 / 139.9 |
| 64 | 180 | 64 | 227.5 + 38.5 + 21.4 + 2.3 = 289.7 | 23% | 96% | 4%: pool run 1.5, thread idle 1.3, step 0.7 | 329.0 | 281.7 / 279.4 |

#### Qwen3-8B Q4_K_M, a layer split over two MI50s

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 72 | 1 | 3.8 + 0.0 + 0.7 + 0.5 = 4.9; 3.9 + 1.2 + 0.7 + 0.5 = 6.2 | 7% | 37% / 46% | 63%: logits wait 41.8, recording 13.8, wait ticket 2.9 ; 54%: previous stage 46.5, recording 3.6, wait ticket 1.8 | 7.1 | 13.1 / 11.1 |
| 8 | 246 | 4 | 8.1 + 0.0 + 1.0 + 0.5 = 9.6; 8.3 + 2.8 + 1.0 + 0.5 = 12.6 | 14% | 65% / 82% | 35%: recording 17.0, logits wait 8.2, pool run 5.2 ; 18%: previous stage 12.1, recording 3.7, wait ticket 0.5 | 16.9 | 26.6 / 22.1 |
| 16 | 304 | 8 | 14.3 + 0.0 + 1.4 + 0.6 = 16.3; 14.5 + 4.8 + 1.4 + 0.6 = 21.3 | 16% | 74% / 92% | 26%: recording 9.0, logits wait 8.6, pool run 4.0 ; 8%: previous stage 5.9, recording 0.5, wait ticket 0.4 | 26.0 | 43.1 / 37.4 |
| 32 | 319 | 16 | 28.4 + 0.0 + 2.4 + 0.7 = 31.5; 28.8 + 9.6 + 2.4 + 0.7 = 41.6 | 17% | 76% / 96% | 24%: logits wait 11.5, recording 5.4, pool run 3.4 ; 4%: previous stage 2.8, recording 0.2, wait ticket 0.1 | 48.5 | 82.0 / 72.3 |
| 64 | 327 | 32 | 56.5 + 0.0 + 4.7 + 0.8 = 62.0; 57.3 + 19.4 + 4.7 + 0.8 = 82.2 | 19% | 76% / 97% | 24%: logits wait 12.8, recording 3.1, pool run 2.8 ; 3%: thread idle 1.7, previous stage 1.0, logits wait 0.1 | 95.1 | 159.8 / 142.1 |

#### Qwen3-8B Q6_K, one MI50

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 51 | 1 | 13.4 + 1.2 + 1.3 + 0.9 = 16.8 | 6% | 90% | 10%: recording 3.4, logits wait 2.5, pool run 1.0 | 19.5 | 18.2 / 16.8 |
| 8 | 140 | 8 | 37.8 + 4.7 + 2.8 + 1.1 = 46.4 | 16% | 95% | 5%: recording 1.8, pool run 1.7, logits wait 0.6 | 55.9 | 47.4 / 46.1 |
| 16 | 145 | 16 | 75.1 + 9.6 + 4.8 + 1.4 = 90.9 | 17% | 96% | 4%: pool run 1.3, recording 0.6, step 0.5 | 107.2 | 91.4 / 90.0 |
| 32 | 146 | 32 | 149.9 + 19.3 + 9.4 + 1.7 = 180.3 | 19% | 97% | 3%: pool run 1.4, step 0.6, recording 0.4 | 205.5 | 177.0 / 175.2 |
| 64 | 146 | 64 | 299.7 + 38.7 + 21.3 + 2.3 = 362.0 | 22% | 96% | 4%: thread idle 1.3, pool run 1.3, step 0.5 | 406.0 | 352.2 / 349.9 |

#### Qwen3-8B Q6_K, a layer split over two MI50s

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 52 | 1 | 6.8 + 0.0 + 0.7 + 0.4 = 7.9; 6.8 + 1.2 + 0.6 + 0.5 = 9.0 | 6% | 43% / 49% | 57%: logits wait 44.7, recording 8.1, wait ticket 1.6 ; 51%: previous stage 47.5, recording 2.1, wait ticket 0.5 | 9.7 | 18.3 / 17.0 |
| 8 | 207 | 4 | 11.8 + 0.0 + 1.0 + 0.5 = 13.2; 11.7 + 2.8 + 1.0 + 0.5 = 16.0 | 12% | 78% / 93% | 22%: recording 9.9, logits wait 6.7, pool run 2.5 ; 7%: previous stage 6.0, recording 0.4, logits wait 0.1 | 18.9 | 32.6 / 29.2 |
| 16 | 257 | 8 | 18.9 + 0.0 + 1.4 + 0.5 = 20.8; 18.9 + 4.8 + 1.4 + 0.6 = 25.7 | 16% | 80% / 95% | 20%: logits wait 10.3, recording 5.8, pool run 2.1 ; 5%: previous stage 3.9, recording 0.2, logits wait 0.2 | 30.2 | 51.4 / 46.2 |
| 32 | 264 | 16 | 37.5 + 0.0 + 2.4 + 0.7 = 40.6; 38.0 + 9.7 + 2.5 + 0.7 = 50.9 | 17% | 80% / 97% | 20%: logits wait 12.8, recording 3.3, pool run 1.6 ; 3%: previous stage 2.0, logits wait 0.1, recording 0.1 | 58.3 | 100.8 / 90.4 |
| 64 | 269 | 32 | 74.9 + 0.0 + 4.7 + 0.8 = 80.4; 75.8 + 19.5 + 4.7 + 0.8 = 100.8 | 19% | 80% / 97% | 20%: logits wait 12.3, recording 2.2, pool run 1.9 ; 3%: thread idle 1.7, previous stage 1.0, logits wait 0.1 | 115.4 | 197.7 / 178.6 |

#### Qwen3-8B Q4_0, one MI50

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 87 | 1 | 6.3 + 1.2 + 1.3 + 0.9 = 9.7 | 8% | 91% | 9%: recording 4.6, pool run 1.6, logits wait 0.5 | 11.5 | 10.5 / 9.7 |
| 8 | 181 | 8 | 27.2 + 4.6 + 2.8 + 1.1 = 35.7 | 17% | 95% | 5%: recording 2.0, pool run 1.3, logits wait 0.3 | 42.5 | 35.8 / 34.7 |
| 16 | 187 | 16 | 54.0 + 9.4 + 4.8 + 1.4 = 69.6 | 18% | 97% | 3%: pool run 1.2, recording 0.8, step 0.3 | 81.5 | 69.2 / 67.9 |
| 32 | 190 | 32 | 107.7 + 19.0 + 9.4 + 1.6 = 137.7 | 20% | 97% | 3%: pool run 1.1, recording 0.4, step 0.3 | 156.8 | 134.2 / 132.7 |
| 64 | 189 | 64 | 214.9 + 38.4 + 21.4 + 2.3 = 276.9 | 23% | 97% | 3%: thread idle 1.2, pool run 1.1, step 0.4 | 313.3 | 269.1 / 267.2 |

#### Qwen3-8B Q4_0, a layer split over two MI50s

| users | tok/s, traced | decode rows a pass | decode pass, device ms by stage: matmul + head + attention/KV + other = sum | prompt passes, share of device time | device busy, by stage | device idle and its largest causes, share of wall | round, ms | decode pass critical path / its dispatches summed, ms |
|---:|---:|---:|---|---:|---|---|---:|---|
| 1 | 79 | 1 | 3.2 + 0.0 + 0.6 + 0.4 = 4.3; 3.2 + 1.2 + 0.6 + 0.5 = 5.4 | 8% | 37% / 46% | 63%: logits wait 41.1, recording 14.3, wait ticket 2.0 ; 54%: previous stage 48.1, recording 3.2, wait ticket 0.8 | 6.3 | 11.3 / 9.7 |
| 8 | 267 | 4 | 7.5 + 0.0 + 1.0 + 0.5 = 9.0; 7.5 + 2.8 + 1.0 + 0.5 = 11.8 | 14% | 70% / 89% | 30%: recording 13.1, logits wait 6.5, pool run 5.5 ; 11%: previous stage 8.1, recording 1.3, wait ticket 0.7 | 14.8 | 24.2 / 20.7 |
| 16 | 322 | 8 | 13.6 + 0.0 + 1.4 + 0.5 = 15.5; 13.6 + 4.8 + 1.4 + 0.5 = 20.3 | 17% | 76% / 95% | 24%: logits wait 11.6, recording 6.7, pool run 3.2 ; 5%: previous stage 4.1, logits wait 0.2, recording 0.2 | 24.0 | 40.0 / 35.4 |
| 32 | 334 | 16 | 27.0 + 0.0 + 2.4 + 0.7 = 30.1; 27.1 + 9.6 + 2.4 + 0.7 = 39.8 | 18% | 76% / 97% | 24%: logits wait 11.6, recording 4.8, pool run 3.7 ; 3%: previous stage 2.2, logits wait 0.2, recording 0.1 | 45.8 | 77.3 / 68.8 |
| 64 | 341 | 32 | 53.8 + 0.0 + 4.7 + 0.8 = 59.3; 53.9 + 19.4 + 4.7 + 0.8 = 78.8 | 20% | 76% / 97% | 24%: logits wait 13.2, pool run 3.0, recording 2.8 ; 3%: thread idle 1.4, previous stage 1.0, logits wait 0.1 | 91.0 | 152.3 / 135.9 |

- **One MI50 is device-bound.** From 16 users on the card is 93 to 97 percent busy on every type, and the host exposes 3 to 7 percent: the sampling pool's run 1.1 to 3.4, stepping 0.3 to 0.8, recording under 2. A decode pass's critical path is its dispatches plus 1 to 2 ms.
- **The decode matmuls are the pass.** At 64 rows the projections are 72 (Q8_0) to 83 percent (Q6_K) of the decode pass and the head 6 (Q8_0) to 14 percent (Q4_0); attention with the KV writes is 20 percent of Q8_0's pass (22 ms) and 6 to 8 percent of the others'.
  From 8 rows on a Q4_K_M, Q6_K or Q4_0 pass grows in step with its rows, so their throughput stops at 8 users: 168 to 190 tok/s for Q4_K_M and Q4_0, 140 to 146 for Q6_K.
  The Q6_K head of those three files, read against 16-bit activations as the head keeps them, costs 38.5 ms at 64 rows.
- **Prompt passes** take 22 to 38 percent of device time at 64 users in this workload (a 128-token prompt for every 128 replies), so the prompt path's speed is a third of this benchmark already.
- **The split** keeps stage 1 86 to 97 percent busy at 64 users and stage 0 76 to 80 percent: stage 1 runs the head, and stage 0's idle is the host's logits wait, recording, sampling and stepping (16 to 19 percent together at 64 users) and the unequal stages.
  From 16 users on, receive waits hold the scheduler thread 66 to 92 percent of the wall, but the producer is busy 82 to 94 percent of that time and the consumer 62 to 84, and the host wakes 61 to 105 us after the producer finishes: the blocking relay withholds little.
  A decode pass's critical path is 8 to 19 ms longer than its dispatches summed at 64 users.

## 2. llmx against the reference server

Every server's best round in run order: llmx's servers first (the first rounds, then the third round and, on two cards, the equal splits of the stage-balance runs), then the reference at `-np` = users with its three batch settings, then its old `-np 64`.
The ratio is llmx's best server against the reference's best server of any setting; the latencies are those two servers'.
The last column keeps the figures of 2026-09-28 (main `3da159b9`): llmx, the reference at `-np 64`, and the reference at `-np` = users where it ran.


#### Qwen3-8B Q8_0, one MI50

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 72, 70 | 60, 60 ; - ; - | 58 | +19% | 110 / 13.1 / 14 | 184 / 14.8 / 24 | 65 ; 62 ; - |
| 8 | 298, 292 | 205, 194 ; - ; - | 182 | +46% | 881 / 20.0 / 30 | 827 / 32.1 / 44 | 249 ; 198 ; - |
| 16 | 341, 336, 341, 338 | 207, 204, 221 ; 209, 219 ; 187 | 193 | +55% | 1784 / 33.3 / 345 | 1574 / 59.8 / 73 | 305 ; 208 ; 217 |
| 32 | 372, 366, 366, 362 | 271, 302, 302 ; 307, 326 ; 245 | 164 | +14% | 3704 / 58.5 / 478 | 2940 / 74.7 / 111 | 325 ; 190 ; 326 |
| 64 | 367, 361, 360, 357 | 325, 296, 318 ; 285, 365 ; 268 | 259 | +1% | 7926 / 119.4 / 544 | 6004 / 128.1 / 197 | 323 ; 358 ; 366 |

#### Qwen3-8B Q8_0, a layer split over two MI50s

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 70, 65 | 63, 59 ; - ; - | 51 | +12% | 107 / 13.5 / 15 | 184 / 14.5 / 18 | 61 ; - ; - |
| 8 | 355, 315 | 187, 202 ; - ; - | 188 | +76% | 630 / 17.5 / 32 | 631 / 33.1 / 71 | 296 ; - ; - |
| 16 | 535, 479, 522, 511, 513 | 214, 212, 231 ; 215, 219 ; 197 | 202 | +132% | 1080 / 21.4 / 28 | 1392 / 58.7 / 64 | 417 ; 210 ; 234 |
| 32 | 634, 602, 634, 637, 636 | 315, 269, 343 ; 302, 340 ; 255 | 161 | +85% | 1990 / 35.0 / 398 | 3020 / 70.2 / 134 | 508 ; 249 ; 341 |
| 64 | 719, 661, 711, 717, 717 | 322, 321, 394 ; 340, 371 ; 258 | 310 | +83% | 3908 / 59.4 / 480 | 6255 / 114.0 / 1528 | 559 ; 335 ; - |

#### Qwen3-8B Q4_K_M, one MI50

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 85, 83 | 26, 69 ; - ; - | 68 | +22% | 125 / 10.7 / 15 | 260 / 12.3 / 16 | 87 ; 72 ; - |
| 8 | 173, 175 | 148, 164 ; - ; - | 157 | +7% | 957 / 38.3 / 68 | 1259 / 38.9 / 46 | 172 ; 163 ; - |
| 16 | 181, 178, 178, 181 | 67, 90, 96 ; 96, 96 ; 87 | 95 | +89% | 1964 / 74.1 / 401 | 2332 / 147.7 / 203 | 174 ; 95 ; 96 |
| 32 | 180, 181, 180, 182 | 127, 147, 150 ; 157, 158 ; 140 | 85 | +15% | 4227 / 147.5 / 558 | 4631 / 166.6 / 198 | 174 ; 89 ; 155 |
| 64 | 179, 181, 180, 180 | 147, 153, 157 ; 157, 170 ; 138 | 150 | +7% | 9481 / 297.5 / 726 | 9747 / 300.5 / 462 | 173 ; 157 ; - |

#### Qwen3-8B Q4_K_M, a layer split over two MI50s

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 80, 81 | 53, 67 ; - ; - | 67 | +22% | 118 / 11.4 / 14 | 266 / 12.4 / 26 | 84 ; - ; - |
| 8 | 256, 267 | 147, 169 ; - ; - | 165 | +59% | 673 / 24.7 / 31 | 996 / 39.2 / 48 | 271 ; - ; - |
| 16 | 310, 317, 315, 317 | 66, 95 ; 98 ; 89 | 94 | +222% | 1149 / 41.7 / 48 | 1837 / 148.7 / 165 | 317 ; 95 ; 98 |
| 32 | 324, 327, 326, 326 | 131, 152 ; 160 ; 142 | 80 | +104% | 2157 / 82.3 / 467 | 4165 / 167.1 / 185 | 326 ; 144 ; 150 |
| 64 | 328, 330, 330, 330 | 146, 159 ; 159 ; 146 | 162 | +104% | 4429 / 164.0 / 576 | 10916 / 310.6 / 2519 | 330 ; 168 ; - |

#### Qwen3-8B Q6_K, one MI50

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 57, 54 | 61, 54 ; - ; - | 61 | -7% | 140 / 16.7 / 17 | 274 / 14.1 / 19 | - ; - ; - |
| 8 | 141, 139 | 145, 131 ; - ; - | 140 | -3% | 1150 / 48.4 / 53 | 1250 / 44.4 / 86 | - ; - ; - |
| 16 | 143, 141, 143, 142 | 145, 120, 146 ; 138, 152 ; 126 | 137 | -6% | 2394 / 95.1 / 355 | 2191 / 88.0 / 97 | - ; - ; - |
| 32 | 143, 142, 143, 142 | 198, 165, 203 ; 194, 230 ; 179 | 128 | -38% | 5194 / 189.1 / 684 | 4342 / 105.1 / 120 | - ; - ; - |
| 64 | 142, 142, 141, 142 | 206, 201, 240 ; 242, 260 ; 211 | 236 | -46% | 11892 / 383.0 / 876 | 9019 / 170.3 / 531 | - ; - ; - |

#### Qwen3-8B Q6_K, a layer split over two MI50s

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 50, 54 | 60, 60 ; - ; - | 62 | -12% | 135 / 17.4 / 20 | 256 / 14.2 / 18 | - ; - ; - |
| 8 | 212, 213 | 158, 154 ; - ; - | 157 | +34% | 801 / 31.5 / 35 | 1018 / 42.3 / 49 | - ; - ; - |
| 16 | 262, 261, 261 | 154, 151, 155 ; 159, 160 ; 143 | 151 | +63% | 1378 / 50.8 / 58 | 1680 / 87.1 / 91 | - ; - ; - |
| 32 | 267, 266, 267 | 221, 233, 234 ; 238, 244 ; 219 | 194 | +9% | 2625 / 101.2 / 551 | 3817 / 102.1 / 108 | - ; - ; - |
| 64 | 270, 270, 270 | 283, 273, 279 ; 292, 289 ; 244 | 271 | -7% | 5338 / 201.3 / 688 | 7892 / 156.4 / 334 | - ; - ; - |

#### Qwen3-8B Q4_0, one MI50

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 93, 95 | 72, 77 ; - ; - | 76 | +23% | 121 / 9.6 / 12 | 224 / 11.1 / 14 | - ; - ; - |
| 8 | 180, 181 | 176, 191 ; - ; - | 187 | -5% | 942 / 36.8 / 51 | 1035 / 33.7 / 42 | - ; - ; - |
| 16 | 185, 185, 187, 187 | 212, 221, 232 ; 218, 235 ; 205 | 224 | -21% | 1958 / 71.4 / 398 | 1866 / 53.4 / 61 | - ; - ; - |
| 32 | 186, 187, 186, 186 | 211, 251, 262 ; 251, 273 ; 231 | 181 | -32% | 4213 / 143.4 / 554 | 3720 / 88.3 / 101 | - ; - ; - |
| 64 | 186, 186, 186, 184 | 219, 236, 244 ; 218, 262 ; 210 | 231 | -29% | 9503 / 287.1 / 715 | 7645 / 183.4 / 529 | - ; - ; - |

#### Qwen3-8B Q4_0, a layer split over two MI50s

| users | llmx, tok/s | reference -np = users: -b 2048 -ub 512 (default) ; 4096/1024 ; 1024/256, tok/s | reference -np 64, tok/s | llmx best / reference best | llmx TTFT p99 / ITL p50 / ITL p99, ms | reference best: TTFT p99 / ITL p50 / ITL p99, ms | 2026-09-28, tok/s: llmx ; reference -np 64 ; reference -np = users |
|---:|---|---|---|---:|---|---|---|
| 1 | 92, 90 | 77, 79 ; - ; - | 79 | +15% | 116 / 9.9 / 13 | 220 / 10.8 / 14 | - ; - ; - |
| 8 | 284, 279 | 196, 201 ; - ; - | 192 | +42% | 667 / 22.9 / 29 | 800 / 33.4 / 42 | - ; - ; - |
| 16 | 325, 329 | 221, 235 ; 227 ; 207 | 234 | +40% | 1142 / 39.9 / 56 | 1794 / 53.3 / 65 | - ; - ; - |
| 32 | 339, 339 | 257, 274 ; 278 ; 253 | 194 | +22% | 2146 / 78.9 / 455 | 3327 / 88.7 / 119 | - ; - ; - |
| 64 | 343, 344 | 249, 241 ; 257 ; 225 | 251 | +34% | 4401 / 156.9 / 568 | 7036 / 190.3 / 278 | - ; - ; - |

**Witness** (the reference's own log at `-lv 4`, a separate run of each measured command at 16 users): on one MI50 `using device ROCm0 (AMD Instinct MI60 / MI50) (0000:89:00.0)`, `n_seq_max = 16`, `n_batch = 2048` or `4096`, `n_ubatch = 512` or `1024`, `flash_attn = enabled`; on two MI50s every type and both batch settings log `llama_context: pipeline parallelism enabled` and `sched copies = 4`, on devices `0000:03:00.0` and `0000:44:00.0`, all 37 layers offloaded.
The measured runs have no `-lv 4`, whose logging could slow the server; their logs and the witness logs are in `raw.tar.xz` (`witness-*`).

- **Q8_0:** llmx leads on one card from 1 to 32 users (+14 to +55 percent) and is level at 64 (+1 percent, 367 against the reference's best 365, which it reached only in the third round); on two cards llmx leads 1.1x to 2.3x.
  llmx has gained since 2026-09-28 (the half-block order and the sampling pool): 323 to 367 tok/s on one card at 64 users, 559 to 719 on two.
  The reference's layer split gains 5 to 8 percent over one card at 16 to 64 users (its best 231, 343 and 394 tok/s on two cards against 221, 326 and 365 on one) although its pipeline parallelism is enabled.
- **Q4_K_M:** llmx leads on one card (+7 to +89 percent), but both are low: llmx 173 to 182 tok/s from 8 users on, the reference 96 to 170 with its own dip at 16 users. On two cards llmx is 1.2x to 3.2x the reference.
- **Q6_K and Q4_0: llmx trails on one card** from 8 users on, by up to 46 percent (Q6_K at 64 users, 142 against 260 tok/s) and 32 percent (Q4_0 at 32 users, 186 against 273), because its decode matmuls stop scaling at 8 rows; its inter-token p50 is higher by about the same factor. At one user llmx is ahead on Q4_0 (+23 percent) and behind on Q6_K (-7 percent). On two cards llmx leads Q4_0 at every level (+15 to +42 percent) and Q6_K from 8 to 32 users (+9 to +63 percent), and trails Q6_K at 64 users (-7 percent) and 1 user.
- **Tails:** on one card the reference's best server has the lower inter-token p99 in all 12 cells at 16 to 64 users (61 to 531 ms against llmx's 345 to 876): a pass that reads a prompt of up to a ubatch of 512 rows stalls that pass's decoders, which shows in llmx's p99 while its p50 is the lower on Q8_0 and Q4_K_M. On two cards llmx's time to first token p99 is the lower at 8 to 64 users on every type.
- **The reference's settings:** matching `-np` to the users raised it at 32 users (on Q8_0 one card from 164 at `-np 64` to 271 to 326 tok/s), as on 2026-09-28; its batch setting moved it by up to 27 percent, `-b 1024 -ub 256` the worst and `-b 4096 -ub 1024` the best or level with the default on most cells. The reference's server keeps one CPU busy, and its figures rose in the third round, when fewer of this investigation's lanes shared its cores (Q8_0 at 64 users: on one card 285 to 325 tok/s in the first rounds and 318 to 365 in the third, on two 321 to 340 and 371 to 394); llmx's moved less (357 to 367 and 661 to 719). The ratios use each side's best server, so the third round sets the reference's side where it was best.

## 3. The general fix

### Batch invariance of the prompt tile at a decode split

The probe tool's `check` sends 160 columns of each tensor of layer 0 (and layer 3 of Q4_K_M) and the head, of every file, alone through the tile at extent 64 and then in calls of 1 to 160 columns, in order and reversed, and beside decode rows in one call.
Every column is the bits it has alone: 0 of 1836 per tensor and extent, 0 of 984 in the mixed calls, for Q8_0, Q4_K, Q6_K, Q4_0 and Q4_1 ([`micro/micro/`](micro/micro/)).
The tile keeps each row's activation form (the 8-bit activations of `quantize_x8`), its order (each block's exact integer dot, then the type's float expression, summed over the blocks in order) and its inner-dimension split, which follows the extent and not the call: extents 64 and 512 give different bits since the split changes, every extent up to 64 the same.
A decode row's column through the tile differs from today's decode kernel's in every column checked (another summation order, and 8-bit activations where the K-quant head reads 16-bit ones), so decode rows through the tile change decode numerics once: they need the HF gate with its headroom and the near-tie check.

### The decode matmuls per type

An 8B pass's matmuls, ms a pass, the mean of the runs (min and max and every run in [`micro/microtab.md`](micro/microtab.md)): the decode path, the tile, and the fastest decode-tile form at that width.


| columns | Qwen3-8B Q8_0 decode / tile / best form | Qwen3-8B Q4_K_M decode / tile / best form | Qwen3-8B Q6_K decode / tile / best form | Qwen3-8B Q4_0 decode / tile / best form |
|---:|---|---|---|---|
| 1 | 11.6 / 46.0 / 24.4 (form 2) | 9.3 / 52.4 / 38.0 (form 1) | 16.0 / 58.0 / 31.0 (form 4) | 7.9 / 51.6 / 28.6 (form 3) |
| 2 | 11.9 / 46.7 / 24.6 (form 2) | 13.0 / 53.0 / 38.1 (form 1) | 19.6 / 58.3 / 31.2 (form 4) | 11.3 / 52.4 / 28.6 (form 3) |
| 4 | 12.9 / 47.3 / 24.8 (form 2) | 19.9 / 54.4 / 38.5 (form 1) | 27.4 / 59.0 / 33.2 (form 4) | 18.6 / 53.0 / 28.9 (form 3) |
| 8 | 16.2 / 47.9 / 29.2 (form 3) | 34.4 / 55.1 / 54.8 (form 2) | 44.1 / 59.8 / 51.8 (form 4) | 32.4 / 53.8 / 30.9 (form 3) |
| 16 | 26.8 / 48.8 / 35.2 (form 3) | 68.5 / 55.6 / 73.9 (form 2) | 87.7 / 60.9 / 51.3 (form 3) | 64.5 / 54.8 / 35.2 (form 3) |
| 32 | 47.1 / 50.9 / - | 136.2 / 57.7 / - | 175.3 / 63.3 / - | 129.2 / 56.7 / - |
| 64 | 94.2 / 58.1 / - | 272.4 / 63.1 / - | 350.6 / 69.5 / - | 258.0 / 62.1 / - |

- The decode path costs 4.0 (Q4_0), 4.2 (Q4_K_M) and 5.5 ms (Q6_K) a pass for every column past the first few, against Q8_0's 1.4: the Q8_0 decode kernel reads a weight once for up to 32 columns and each further column costs its dot, while the other types' row kernels pay each column's unpacking again.
- The tile costs 46 to 58 ms at one column and 58 to 70 at 64, so it is 1.6x (Q8_0) to 5x (Q6_K) cheaper at 64 columns and 3.6x to 6.5x dearer at one; it passes the decode path between 32 and 64 columns on Q8_0 and between 8 and 16 on the others.

### Five decode-tile kernels with the tile's bits

Each probe kernel computes the tile's arithmetic in a row kernel's shape: the tile's staged words, its exact block dots, its float expression and order within the same parts, the parts added by the same reduce. Form 1 gives a row 8 lanes, a word each, the block sums meeting in a clustered add; form 2 has each of 8 lanes take a whole block and exchange the sums through shared memory; form 3 gives each lane a whole row; form 4 is form 2 with the exchange by shuffles; form 5 (Q8_0 only) loads a row's blocks with consecutive lanes on consecutive words.
Every form gives every column of calls of 1 to 16 columns, first and later columns, the tile's bits for Q8_0, Q4_K, Q6_K, Q4_0 and Q4_1: 0 mismatches in every check ([`micro/`](micro/)).


| model | columns | decode path | tile | form 1 | form 2 | form 3 | form 4 | form 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B Q8_0 | 1 | 11.6 | 46.0 | 25.2 | 24.4 | 26.8 | 27.6 | 56.6 |
| Qwen3-8B Q8_0 | 8 | 16.2 | 47.9 | 41.9 | 31.3 | 29.2 | 49.5 | 71.0 |
| Qwen3-8B Q8_0 | 16 | 26.8 | 48.8 | 78.2 | 47.0 | 35.2 | 95.3 | 99.0 |
| Qwen3-8B Q4_K_M | 1 | 9.3 | 52.4 | 38.0 | 46.3 | 138.8 | 45.4 | - |
| Qwen3-8B Q4_K_M | 8 | 34.4 | 55.1 | 58.3 | 54.8 | 140.9 | 64.2 | - |
| Qwen3-8B Q4_K_M | 16 | 68.5 | 55.6 | 100.5 | 73.9 | 143.8 | 121.3 | - |
| Qwen3-8B Q6_K | 1 | 16.0 | 58.0 | 46.2 | 49.2 | 53.7 | 31.0 | - |
| Qwen3-8B Q6_K | 8 | 44.1 | 59.8 | 73.2 | 61.7 | 59.2 | 51.8 | - |
| Qwen3-8B Q6_K | 16 | 87.7 | 60.9 | 128.6 | 91.9 | 51.3 | 103.8 | - |
| Qwen3-8B Q4_0 | 1 | 7.9 | 51.6 | 32.1 | 29.0 | 28.6 | 30.4 | - |
| Qwen3-8B Q4_0 | 8 | 32.4 | 53.8 | 51.2 | 36.4 | 30.9 | 50.1 | - |
| Qwen3-8B Q4_0 | 16 | 64.5 | 54.8 | 90.8 | 54.5 | 35.2 | 96.4 | - |

- None reaches the row kernels at one column: the best is 1.9x (Q6_K, 31.0 against 16.0 ms), 2.1x (Q8_0, 24.4 against 11.6), 3.6x (Q4_0, 28.6 against 7.9) and 4.1x (Q4_K_M, 38.0 against 9.3). They read 300 to 350 GB/s where the Q8_0 decode kernel reads about 700.
  Each form either reads a row's blocks lane by lane, so its loads are not coalesced, or carries the in-order float chain of a part through shared memory with barriers; form 3's general build spills on Q4_K (139 ms).
  Giving the decode class its own split, parts of 16 or 32 blocks, left one column's cost where it was and made the tile dearer at 64 rows ([`micro/micro5/`](micro/micro5/)).
- What would reach it, untested: the Q8_0 decode kernel's coalesced layout feeding exact per-block integer sums, with each part's float chain kept in registers on one lane across the subgroup's steps.
  Form 3 is already faster than the decode path at 8 columns on Q4_0 and at 16 on Q4_0 and Q6_K.

### Served

The probe with decode rows given extent 64, so that their matmuls take the tile at its decode split while one-row attention views keep the per-row kernel, then with calls of up to 16 columns through form 2, interleaved with the unchanged probe on one MI50 (`base` and `base2` the unchanged probe before and after):

#### Decode rows through the tile, one MI50 (tok/s)

| model | users | base | tile | dtile | base2 |
|---|---:|---|---|---|---|
| Qwen3-8B Q8_0 | 1 | 70 (ITL p50 13.4) | 22 (ITL p50 45.7) | 38 (ITL p50 25.7) | 70 (ITL p50 13.5) |
| Qwen3-8B Q8_0 | 8 | 286 (ITL p50 21.0) | 139 (ITL p50 50.0) | 193 (ITL p50 34.5) | 291 (ITL p50 20.4) |
| Qwen3-8B Q8_0 | 16 | 334 (ITL p50 33.9) | 228 (ITL p50 55.6) | 239 (ITL p50 52.9) | 337 (ITL p50 33.6) |
| Qwen3-8B Q8_0 | 32 | 360 (ITL p50 60.9) | 341 (ITL p50 64.0) | 343 (ITL p50 64.0) | 365 (ITL p50 59.9) |
| Qwen3-8B Q8_0 | 64 | 357 (ITL p50 122.9) | 432 (ITL p50 87.8) | 437 (ITL p50 87.2) | 361 (ITL p50 121.1) |
| Qwen3-8B Q4_K_M | 1 | 82 (ITL p50 11.1) | 19 (ITL p50 51.8) | 22 (ITL p50 45.7) | 85 (ITL p50 10.9) |
| Qwen3-8B Q4_K_M | 8 | 173 (ITL p50 39.0) | 118 (ITL p50 60.1) | 120 (ITL p50 59.3) | 173 (ITL p50 39.1) |
| Qwen3-8B Q4_K_M | 16 | 176 (ITL p50 76.5) | 191 (ITL p50 68.3) | 162 (ITL p50 83.7) | 176 (ITL p50 76.5) |
| Qwen3-8B Q4_K_M | 32 | 177 (ITL p50 151.8) | 269 (ITL p50 87.1) | 269 (ITL p50 87.4) | 177 (ITL p50 152.6) |
| Qwen3-8B Q4_K_M | 64 | 177 (ITL p50 306.0) | 333 (ITL p50 128.1) | 340 (ITL p50 124.5) | 176 (ITL p50 306.0) |
| Qwen3-8B Q6_K | 1 | 55 (ITL p50 17.2) | 17 (ITL p50 57.0) | 20 (ITL p50 48.5) | 55 (ITL p50 17.1) |
| Qwen3-8B Q6_K | 8 | 138 (ITL p50 49.2) | 106 (ITL p50 65.3) | 106 (ITL p50 66.5) | 137 (ITL p50 50.0) |
| Qwen3-8B Q6_K | 16 | 139 (ITL p50 97.9) | 171 (ITL p50 74.9) | 133 (ITL p50 102.1) | 138 (ITL p50 99.2) |
| Qwen3-8B Q6_K | 32 | 141 (ITL p50 194.4) | 240 (ITL p50 95.8) | 241 (ITL p50 94.2) | 139 (ITL p50 196.6) |
| Qwen3-8B Q6_K | 64 | 140 (ITL p50 389.4) | 298 (ITL p50 138.7) | 300 (ITL p50 138.1) | 139 (ITL p50 391.3) |
| Qwen3-8B Q4_0 | 1 | 95 (ITL p50 9.4) | 19 (ITL p50 50.9) | 34 (ITL p50 28.4) | 97 (ITL p50 9.4) |
| Qwen3-8B Q4_0 | 8 | 176 (ITL p50 38.2) | 120 (ITL p50 58.8) | 165 (ITL p50 41.0) | 179 (ITL p50 37.2) |
| Qwen3-8B Q4_0 | 16 | 180 (ITL p50 74.9) | 191 (ITL p50 67.9) | 200 (ITL p50 65.2) | 181 (ITL p50 73.9) |
| Qwen3-8B Q4_0 | 32 | 179 (ITL p50 149.6) | 270 (ITL p50 87.2) | 272 (ITL p50 87.0) | 180 (ITL p50 148.9) |
| Qwen3-8B Q4_0 | 64 | 180 (ITL p50 300.0) | 329 (ITL p50 131.0) | 336 (ITL p50 128.1) | 178 (ITL p50 301.7) |

#### The decode tile again on probe8, its attention guard keyed on the decode extent, one MI50 (tok/s)

| model | users | base | dtile |
|---|---:|---|---|
| Qwen3-8B Q8_0 | 1 | 70 (ITL p50 13.4) | 38 (ITL p50 25.8) |
| Qwen3-8B Q8_0 | 16 | 342 (ITL p50 33.2) | 238 (ITL p50 52.9) |
| Qwen3-8B Q8_0 | 64 | 361 (ITL p50 121.4) | 431 (ITL p50 90.2) |
| Qwen3-8B Q4_K_M | 1 | - | 22 (ITL p50 45.7) |
| Qwen3-8B Q4_K_M | 16 | - | 160 (ITL p50 84.3) |
| Qwen3-8B Q4_K_M | 64 | - | 335 (ITL p50 128.0) |

- Decode rows through the tile lift 64 users by 21 percent on Q8_0, 88 to 92 percent on Q4_K_M, 113 percent on Q6_K and 83 to 87 percent on Q4_0, and 32 users by 51 to 71 percent on the three 4- and 6-bit types, where Q8_0 loses 5 percent; below 16 users they lose, at one user 69 to 80 percent through the tile and 46 to 73 percent with form 2.
  With it, one MI50 serves Q4_K_M, Q6_K and Q4_0 at 298 to 340 tok/s at 64 users, above the reference's best at 64 users (170, 260 and 262).
- **Batch invariance, served:** `tools/server_mix_check.py --logprobs`, 32 requests alone, all 32 at once and skewed with clients leaving, over decode rows through the tile and form 2: every request's ids and log-probabilities match alone on Q8_0 and on Q4_K_M, and so does the control without the decode tile.
  The first run of it failed (2 of 32 requests together and 1 of 24 skewed, first on the log-probability of the first generated token): the probe's attention guard then kept every one-row view on the per-row kernel, and a prompt that the scheduler read in slices had a one-row slice take that kernel when read beside others, so its last row depended on the batch. Keyed on the decode extent instead, as the kernel choice is, the rerun passes (`raw.tar.xz`, lanes e and e2). The guard's figures above are from the first probe; the second served the same (Q8_0 at 64 users 431 against 437, Q4_K_M 335 against 340).
- Without any attention guard, decode rows at extent 64 also took the tiled attention kernel: on the traced lane's card that served Q8_0 at 64 users 357 tok/s against 362 to 373 for main, and Q4_K_M, Q6_K and Q4_0 56, 84 and 57 percent above main (`raw.tar.xz`, lane d, arms `*-tile`), so the guard is part of the fix.

### Host costs and the split

- **Logits copy:** rows are read in place since step 4. Copying each row to host memory first costs 0.14 ms of CPU a row (8.8 ms a 64-row pass) beside the draw's 0.34, and served 347 to 350 against 365 to 367 tok/s at 32 and 64 users on Q8_0, both traced: in place stays.
- **Sampling:** the pool's run is 1.1 to 3.4 percent of the wall on one card at 16 to 64 users and 1.6 to 7.9 percent on the split, where it idles stage 0.
- **Blocking relay:** from 16 users on, receive waits hold the split's scheduler thread 66 to 92 percent of the wall, but the producer is busy 82 to 94 percent of that time, the consumer 62 to 84, and the host wakes 61 to 105 us after the producer finishes: backpressure, not withheld work.
- **P = S + 1:** Q8_0 on two MI50s at 16, 32 and 64 users served 448, 590 and 620 tok/s against 479 to 535, 602 to 637 and 661 to 719 at P = S (lane b, `q8-llmxp3`), as on 2026-09-28's first measurement.
- **Stage balance** (`--layer-shares`, llmx main on the two split cards, the equal split before and after):

#### Stage balance on two MI50s (tok/s)

| model | users | eq | s1917 | s2016 | s2115 | eq2 |
|---|---:|---|---|---|---|---|
| Qwen3-8B Q8_0 | 16 | 522 (ITL p50 21.7) | 510 (ITL p50 22.7) | 501 (ITL p50 23.1) | - | 511 (ITL p50 22.6) |
| Qwen3-8B Q8_0 | 32 | 634 (ITL p50 35.0) | 617 (ITL p50 36.1) | 595 (ITL p50 37.7) | - | 637 (ITL p50 35.0) |
| Qwen3-8B Q8_0 | 64 | 711 (ITL p50 60.6) | 686 (ITL p50 62.7) | 658 (ITL p50 65.4) | - | 717 (ITL p50 59.9) |
| Qwen3-8B Q4_K_M | 16 | 315 (ITL p50 41.8) | - | 324 (ITL p50 40.2) | 307 (ITL p50 42.7) | 317 (ITL p50 41.9) |
| Qwen3-8B Q4_K_M | 32 | 326 (ITL p50 82.5) | - | 341 (ITL p50 76.5) | 322 (ITL p50 81.6) | 326 (ITL p50 82.4) |
| Qwen3-8B Q4_K_M | 64 | 330 (ITL p50 164.4) | - | 347 (ITL p50 151.4) | 338 (ITL p50 156.0) | 330 (ITL p50 164.2) |

On Q4_K_M, 20 layers to stage 0 and 16 to stage 1 gained 3 to 5 percent over the equal split, 21 and 15 less; on Q8_0, whose head is 3.3 ms at 32 rows, every move lost 2 to 8 percent.

## Ranked fixes

By measured gain on this workload (one MI50 unless said), with what each needs:

1. **Decode rows of Q4_K, Q5_K, Q6_K, Q4_0 and Q4_1, and the Q6_K head, through the tile at many rows: +83 to +113 percent at 64 users, +51 to +71 at 32** (served, batch-invariant, the served check passing on Q4_K_M). Needs:
   - a row-shaped kernel with the tile's arithmetic at the row kernels' speed for 1 to about 12 rows, which the five forms show possible for the bits and not yet for the speed (1.9x to 4.1x slower at one row), so that one user does not lose 64 to 80 percent;
   - decode rows taking the tile's arithmetic at every batch, a one-time decode numerics change: the HF gate with its headroom and the near-tie check, and `backend-vulkan`'s column-invariance tests over the tile and the new kernel;
   - one-row attention views kept on the per-row kernel by the extent, as the probe's second guard does.
   - A second route, not measured: Q8_0-style decode builds for these types, each weight unpacked once and dotted against up to 32 columns with every column's bits the same in every build, as the Q8_0 decode kernel already does. It keeps one user's speed; its gain is bounded by Q8_0's 1.4 ms a column against their 3.9 to 5.4.
2. **The same for Q8_0: +21 percent at 64 users**, -5 percent at 32 and a loss below; same needs.
3. **Stage balance on the split: +3 to +5 percent on Q4_K_M at 16 to 64 users** with 20 and 16 layers; the layer fit needs to count the head's time (a Q6_K head costs about as much as 5 layers of the stage at 32 rows), and Q8_0 is best equal.
4. **Host work on the split's stage 0:** recording, sampling and stepping expose about 16 percent of stage 0 at 64 users on Q8_0, but stage 1 is 86 percent busy and waits on stage 0 for 9 percent, so the bound is about 9 percent on Q8_0 and 3 on the K-quant files; needs recording off the scheduler thread (step 7's).
5. **Attention and the KV writes at many rows:** 22 ms of a 112 ms 64-row Q8_0 pass, and 25 to 35 percent of the pass once fix 1 or 2 lands; not measured as a fix.
- **Not fixes:** P = S + 1 (-7 to -16 percent), copying rows before sampling (-5 percent), and the relay (backpressure).

## Activity during the runs

The flags of [`scripts/FLAGS.txt`](scripts/FLAGS.txt) were fixed before the first timed run; [`flags.txt`](flags.txt) gives every timed level its samples, the host CPU busy range, this investigation's own CPU and the busiest processes outside it.
Every level is flagged. Until 11:26 up to five of this investigation's lanes shared cores 4-7,12-15, 80 to 99 percent busy; from 11:06 a timing round of another track ran on cores 4-7 and its HF and test containers on 12-15, and the other developer's work ran on 0-3,8-11; each lane's own model loads raise the disk flag.
Nothing was stopped or dropped. The effects the findings rest on are far larger than the rounds' spread; the host-side shares are upper bounds, and the reference, whose server keeps one CPU busy, gained more than llmx when contention eased (above).


| lane | timed levels | flagged cpu | flagged disk | unflagged | host CPU busy, range of samples | this investigation's CPU, mean of samples | outside processes seen busiest |
|---|---:|---:|---:|---:|---|---:|---|
| lane-a | 62 | 61 | 56 | 0 | 37-100% | 285% | llmx(llmx-p2-q35v-gateB) 252%, llmx(llmx-p2-q35v-gateC) 207%, dockerd(host) 186% |
| lane-b | 156 | 155 | 134 | 0 | 25-100% | 209% | llmx(llmx-p2-q35v-gateB) 390%, llmx(llmx-p2-q35v-gateC) 274%, python(llmx-p2-q35v-hffe) 230% |
| lane-c | 62 | 62 | 57 | 0 | 37-100% | 308% | llmx(llmx-p2-q35v-gateB) 244%, llmx(llmx-p2-q35v-gateC) 207%, dockerd(host) 186% |
| lane-d | 112 | 110 | 99 | 0 | 25-100% | 245% | llmx(llmx-p2-q35v-gateB) 390%, llmx(llmx-p2-q35v-gateC) 274%, dockerd(host) 186% |
| lane-e | 80 | 79 | 70 | 0 | 25-62% | 160% | llmx(llmx-p2-q35v-gateB) 390%, llmx(llmx-p2-q35v-gateC) 274%, python(llmx-p2-q35v-hffe) 230% |
| lane-e2 | 9 | 9 | 4 | 0 | 29-48% | 95% | llmx(llmx-p2-q35v-gateB) 226%, python(llmx-p2-q35v-hffe) 214%, cc1plus(llmx-p2-x-mxfp4-gpu2-20260928) 100% |
| lane-f | 24 | 24 | 22 | 0 | 29-60% | 179% | llmx(llmx-p2-q35v-gateB) 264%, python(llmx-p2-q35v-hffe) 230%, llama-bench(llmx-p2-q35v-time7) 100% |
| lane-g | 24 | 24 | 17 | 0 | 29-69% | 151% | llmx(llmx-p2-q35v-gateB) 243%, python(llmx-p2-q35v-hffe) 221%, cc1plus(llmx-p2-x-mxfp4-gpu2-20260928) 100% |
| lane-h | 42 | 42 | 20 | 0 | 27-54% | 98% | llmx(llmx-p2-q35v-gateB) 259%, python(llmx-p2-q35v-hffe) 235%, cc1plus(llmx-p2-x-mxfp4-gpu2-20260928) 100% |
| witness-a | 4 | 4 | 4 | 0 | 46-69% | 346% | python(llmx-p2-q35v-hffe) 214%, llmx(llmx-p2-q35v-gateB) 213%, containerd(host) 16% |
| witness-b | 8 | 8 | 8 | 0 | 35-69% | 264% | llmx(llmx-p2-q35v-gateB) 231%, python(llmx-p2-q35v-hffe) 214%, containerd(host) 18% |
| witness-c | 4 | 4 | 4 | 0 | 35-54% | 224% | llmx(llmx-p2-q35v-gateB) 259%, python(llmx-p2-q35v-hffe) 235%, containerd(host) 15% |

## Files

- [`index.json`](index.json): every file of this record, its size, SHA-256 and what it holds.
- [`summary.txt`](summary.txt), [`summary.json`](summary.json): every server's every level, both rounds, TTFT and ITL p50 and p99, failures, and the reference's configuration where its log gives it.
- [`analysis/`](analysis/): each traced arm's breakdown per level, text and JSON.
- [`micro/`](micro/): the probe tool's checks and timings per run, and [`micro/microtab.md`](micro/microtab.md).
- [`flags.txt`](flags.txt), [`flags.json`](flags.json): the activity flags per timed level.
- [`previous-20260928.txt`](previous-20260928.txt), [`previous-20260928.json`](previous-20260928.json): the 2026-09-28 servers' figures, read from their load records.
- `raw.tar.xz`, kept outside the repository in `llmx-evidence/server-investigation-20260929/`, as [ASSETS](../../ASSETS.md) keeps raw payloads, with its size and SHA-256 in `index.json`: every lane's raw records but the load tool's JSON (its tables, server logs, level windows, health, progress, clock logs, the mix checks' ids and log-probabilities), the witness runs, the microbench runs, the build and fetch logs and the monitor's samples.
- `records.tar.xz`, kept outside the repository in `llmx-evidence/server-investigation-20260929/`, as [ASSETS](../../ASSETS.md) keeps raw payloads, with its size and SHA-256 in `index.json`: the load tool's JSON records of every level, every request with its inter-token gaps, each time rounded to 0.1 ms to keep the archive small. The full-precision records and the dispatch traces (30 to 60 MB each) stay on the machine that ran them.
- [`scripts/`](scripts/): the lane, microbench, witness, build, fetch and packaging scripts, the monitor, and the analysis and table scripts.
- [`probe-src/`](probe-src/): the probe trees as diffs against `6147753a`: `probe.diff` (the trace, the logits copy mode and the decode extent), `probe2.diff` to `probe7.diff` (the decode-tile forms, cumulative, with the first attention guard and the split override), and `probe8.diff` (form 2 with the attention guard keyed on the decode extent).
- [`binaries.sha256`](binaries.sha256), [`models.sha256`](models.sha256).
