# llmx - Development Status

Current implementation and remaining work. Historical checkpoints, failed
experiments and raw evidence remain in [ASSETS](ASSETS.md) and
`docs/benchmarks/`; their dated next steps are not current blockers.

## Multi-device phase 1: layer split across devices (ROADMAP #5) (2026-09-24, branch feat/multi-device-phase1)

- **Goal:** phase 1 of `docs/MULTI-DEVICE.md`: a model split by layers over the devices `--device` lists, each device's share chosen by a fit against its free memory (`Backend::memory_available`), admission that counts every KV pool in its own block size, sharded GGUF mapped shard by shard so a model past host memory loads, and weights uploaded to the devices in parallel. Today's crossing (a read and a write) stays; pipelining is phase 2.
- **Done:** `Backend::memory_available` (CPU: `core/host_memory.hpp`; Vulkan: `VK_EXT_memory_budget`); `model/layer_split.hpp`, architecture-neutral, fitting consecutive layers per device from the architecture's `footprint` (`arch_qwen.hpp`, which alone knows the tensors) with the arena's slot widths shared with `ensure`; `--device A,B,...` and `--layer-shares` (proportions) on every command, a device listed once; `--verbose` prints each device's share. The placement test checks the fit and a fitted split over two CPU backends bit-identical to one. On the Radeon VII with the CPU at shares 1,1 the whole Python suite with the three HF fixtures passes; the single-device suite and the 22 native tests are unchanged.
- **Done, admission per pool:** the server reserves each request's blocks in every cache pool in that pool's own block size (`blocks_for`, `room_for`), donors and growth steps too, and the model refuses pools whose block sizes do not nest, since a shared prefix ends on a whole block of the largest. Counting everything in the largest block refused every request of a 16-token budget on a 64-token device pool beside a 128-token CPU pool; the server component caught it.
- **Done, on two MI50s (rocm-smi GPU[2] and GPU[3]):** one card against a 1:1 split is byte-identical in the top-10 logits of the last 8 positions of the 247-token excerpt and in 64 greedy tokens on Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M. Qwen3-32B Q8_0, which no one card holds, fits as 32 layers each (16.2 GiB of weights and 5.0 GiB of cache per card at the default context): excerpt NLL 2.253, `bench` pp64 188, pp512 222, tg128 13.28 tok/s, against the llama.cpp Vulkan build's layer split on the same cards at 81, 205, 13.3 and its ROCm fork's at 195, 309, 18.8. Qwen3-30B-A3B Q8_0 as 24 layers each: NLL 2.427, pp64 430, pp512 1250, tg128 63.9. One request on a layer split pays the waiting card's clock (phase 0); throughput comes with passes in flight (phases 2 and 3).
- **Done, bounded shard loading:** every GGUF file is mapped, a sharded model as one segment per shard in one offset space, so no shard is copied into host memory; a metadata-only shard maps nothing, and a shard truncated before loading is refused before any progress. The shard test releases a model's mappings before rewriting its files, since Windows keeps a mapped file from being rewritten or removed.
- **Done, review of b86236a..f332209 (six findings):** the fit tries every first and last device, each carrying at least one layer and its endpoint weights, and keeps the plan with the fewest layers on host devices, so a host device returns when moving the embedding leaves a device short (the reviewer's reproduction is a test); a tied head on the embedding's device shares its buffer and the head's norm is counted; the server fits a prompt's ubatch plus `--max-seqs` rows; `Backend::resident_bytes` lets the Vulkan backend count its padded F32 copies; `--device` entries are canonical before de-duplication (`vulkan`, `vulkan:0`, `vulkan:00`); `memory_available` returns nothing when it cannot tell, and zero stays a full device. Docs: per-pool admission in MULTI-DEVICE, the multi-device STATUS row, and the contract that a mapped model's files stay unchanged while loaded.
- **Measured once, Qwen3-235B-A22B Q4_K_M on six MI50s (rocm-smi GPU[2] to GPU[7]):** the fit gives 15 or 16 layers each (20.5 to 23.5 GiB of weights per card); greedy text sane, tg32 18.96 tok/s (the llama.cpp Vulkan layer split on the same six reads tg128 16.3, its ROCm fork 29.1); loading took 744 s. Further runs use smaller models first.
- **Done, gate evidence at e9ee1aa on two MI50s (rocm-smi GPU[2] and GPU[3]):** every one of the 151,936 logits printed for each of the last 32 positions of the 247-token excerpt is identical between one card and a 1:1 split, and 64 greedy tokens match, on Qwen3-0.6B Q8_0 and Qwen3-8B Q8_0. Single device, main b86236a against e9ee1aa on GPU[2] alone, `bench --p 512 --n 128 --r 3`, two interleaved rounds (tok/s): 0.6B pp512 9394 / 9328 against 9391 / 9395, tg128 322.2 / 323.8 against 324.3 / 323.1; 8B pp512 897.5 / 899.8 against 896.7 / 893.0, tg128 66.78 / 66.49 against 66.54 / 66.39: level within the spread.
- **Done, loading:** a payload larger than the host's available memory is no longer touched page by page before the devices copy it, since those pages were evicted in between and read from disk twice: Qwen3-235B-A22B Q4_K_M loads and generates 16 tokens on six MI50s in 370 s against 744 s, the same greedy text. Loading is bound by the disk at about 400 MB/s cold; a model the host can hold is touched as before. Parallel uploads are not taken: a warm Qwen3-8B uploads to two cards in 1.6 s, and a cold load waits on the disk, not on the uploads.
- **Done, re-review of 52bde20:** layers are assigned by a small dynamic program over their actual sizes for every first and last device, fewest host layers then the lightest busiest device, so uneven layers (400, 10, 10, 10 MiB on 800 and 500 MiB) fit and three equal devices take one of three equal layers each (both the reviewer's probes, now tests); only matrices a product reads are charged a padded F32 copy, not the embedding's gather or routed expert stacks; the head's logits rows count on the output device; MULTI-DEVICE's opening says each phase merges on its own row's gates.
- **Done, remaining gate runs:** the whole Python suite with the three HF fixtures required and CTest pass on two MI50s split 1:1 (tiny HF F32, MoE and shard models within 7e-7 of HF), as on the Radeon VII with the CPU. Radeon VII single device, main b86236a against this branch, bench pp512 and tg128 over interleaved rounds: 8B Q8_0 pp 338.1 / 339.2 against 338.6 / 339.2, tg 41.31 / 41.47 against 41.73 / 41.84; 0.6B Q8_0 decode reads either about 188 or about 196 tok/s from run to run in both binaries alike (rounds 5 to 10: branch 195.9, 195.7, 195.8, 195.8, 188.5, 188.7; main 195.9, 187.5, 195.8, 188.8, 195.7, 196.1), a machine state rather than a code difference.
- **Left:** the reviewer's confirmation of the latest fit; then merge. The gates still open are those of later phases: exact against one device on CPU+CPU and on two identical MI50s (tiny HF model, 0.6B, 8B, 30B-A3B), HF bounds on CPU+Vulkan, Qwen3-30B-A3B Q8_0 and Qwen3-32B Q8_0 on two MI50s, Qwen3-235B-A22B on six, the Radeon VII with CPU stages, no regression on one device.
- **Gotchas:** a tied embedding and head on different devices are uploaded to both; `--n-cpu-moe` with several devices is refused until expert tiers (phase 5b).

## Multi-device phase 0: measurements (ROADMAP #5) (2026-09-24, branch feat/multi-device-phase0)

- **Goal:** the numbers phase 0 of `docs/MULTI-DEVICE.md` asks for, before any split is written: the Vulkan handoff between two MI50s, P over S, S+1, S+2 and 2S, whether a host-relayed group sum and an expert exchange pay under Vulkan, and the baselines (llama.cpp Vulkan and ROCm on pinned cards, the vLLM gfx906 fork brought up and checked). No runtime change lands from this branch; its tools and records do.
- **Done, device queries** (`llmx-vk-handoff probe`, nine MI50s under RADV, Mesa 25.0.7): binary semaphores export and import as sync files; timeline semaphores only as opaque descriptors, and each card has its own device UUID, so a timeline does not cross cards and the host relay stays the baseline; host memory imports into every card (4096-byte alignment, a cached coherent host type); device memory exports and imports as dma-buf; no device groups, every card is a group of one. Every card links at Gen4 x8 to its root port; the root complexes hold cards 03, 44, four at 83 to 8c and four at c3 to cc.
- **Done, handoff** (`llmx-vk-handoff time`, 200 timed repeats, median): a submission of one command buffer costs 50 to 70 us from submit to host wake even when empty (4 us with no command buffer), each further command buffer about 20 us more, and a small copy inside a command buffer about 4 us. So a handoff's copy belongs in the stage's own submission, where what remains is the transfer. Card to card: 20 KB 113 to 165 us whichever way, since two submissions dominate; 10 MB 2.07 ms through imported host memory with a sync-file wait (two crossings at 10.9 GB/s each), 2.7 ms through today's read and write. A dma-buf import lands in host-visible memory on the importing card: 9.2 GB/s from a card on the same root complex with clocks held high (one crossing), 4.8 at automatic clocks, 1.1 GB/s across complexes.
- **Done, concurrency** (`llmx-multi-device-bench concurrent`): 2, 4 and 8 cards in one process, each on its own thread and backend, run their decode-shaped and 64-row passes within 1 percent of alone.
- **Done, pipeline** (`pipeline`, two stages of about 10 and 3 ms, host sampling 0.3 ms): with P = 2 on two stages the rate is 99 to 104 percent of the slower stage's back-to-back rate, and P of 3, 4 and 6 add latency only, so P = S while the host's time per pass is small against a stage. With one pass at a time the waiting card drops its clock: a pass through both 10 ms stages takes 29.4 ms at automatic clocks and 19.5 with clocks held high. A waiting stage that keeps its card busy with a slice of its up projection (`pipeline ... warm_rows`) recovers part of it, 30.0 to 26.1 ms with 256 rows at a time and 24.2 with 2048, at a constant load on an idle card; it is not adopted.
- **Done, group sum and exchange** (`groupsum`, `exchange`): a host-relayed sum of one decode row takes 227, 346 and 639 us on 2, 4 and 8 cards, 8 to 25 ms for 512 rows; two a layer is at least 29 ms per token on a 64-layer model, more than the decode it would split, so a Vulkan tensor group through the host is not worth writing.
- **Done, device-side chain** (`llmx-vk-handoff pingpong`, 200 hops, median of five chains): bytes bounced between two cards, each hop reading the other card's memory through dma-buf and writing its own. A 20 KB hop costs 30 us as back-to-back submissions on one card, 81 to 91 us with the host waiting on every hop, and 55 us with the whole chain queued ahead through sync files, on the same root complex and across complexes alike; 160 KB 72 us on one complex and 132 across. A tensor group of two summing this way pays about 7 ms a token on Qwen3-32B (two sums a layer, 64 layers) against about 30 ms of each card's half of the weights, and keeps both cards busy, so it is the candidate for one user on a model past one card: an estimated 26 tok/s against the Vulkan reference's 13.3 below. The fork's ROCm tensor split reads 33.6, which a Vulkan group at this hop cost does not reach; that needs ROCm's peer stores. Prompt sums of 512 rows want a pair on one root complex (about 1.1 ms a sum there, 9 ms across).
- **Baselines, Qwen3-32B Q8_0 on two MI50s (rocm-smi GPU[2] and GPU[3], one root complex), llama-bench -p 64,512 -n 128 -r 3, two interleaved rounds, tok/s:** the upstream Vulkan build (7ab4ee7ba) with a layer split pp64 79.9 to 81.0, pp512 198.4 to 204.7, tg128 13.24 to 13.32; the fork's ROCm image (eefc4e732) with a layer split 195.0 to 195.1, 307.4 to 309.5, 18.69 to 18.78; the same with its tensor split 373.5 to 373.6, 569.6 to 570.3, 33.57 to 33.66. The 235B download ran beside these (network and disk only).
- **Baselines, Qwen3-235B-A22B Q4_K_M** (132.4 GiB in five shards; llama-bench -p 64,512 -n 128 -r 3, one round, tok/s): the upstream Vulkan build's layer split on six MI50s (rocm-smi GPU[2] to GPU[7]) pp64 45.7, pp512 160.9, tg128 16.3; the fork's ROCm layer split on the same six 59.4, 174.7, 29.1; its tensor split on eight (GPU[0] to GPU[7]) 179.4, 459.7, 40.6 with a spread of 6.2 in decode. The tensor split runs at width 8 although the model has 4 KV heads.
- **Server baselines, Qwen3-32B Q8_0 on the two cards above** (`tools/server_load.py`, 128 greedy tokens a request on short prompts, 32 slots, best of two rounds; decoded tok/s, time to first token p50, inter-token p50): the fork's ROCm layer split 17.6 / 56.3 / 84.0 / 151.6 tok/s at 1 / 4 / 16 / 32 requests, TTFT 104 ms to 2.55 s, ITL 54 to 188 ms; its tensor split 32.5 / 86.1 / 124.3 / 213.3, TTFT 88 ms to 2.49 s, ITL 30 to 131 ms; the upstream Vulkan layer split 13.1 / 40.6 / 34.9 / 64.4, TTFT 106 ms to 4.81 s, ITL 76 to 463 ms. The vLLM gfx906 fork (image v0.12.0-rocm6.3, archived) runs Qwen3-32B-AWQ with tensor parallelism 2 on the same cards once its RCCL peer transport is disabled (`NCCL_P2P_DISABLE=1`; with it, `hipIpcGetMemHandle` fails with an invalid argument): greedy text is sane, mean NLL over the 247 targets of the fixed wikitext excerpt 2.366. Its own `vllm bench serve` (random 128 tokens in, 128 out, end of text ignored) gives 35.0 / 101.7 / 153.5 / 184.8 output tok/s at 1 / 4 / 16 / 32 concurrent, median TTFT 382 ms to 4.58 s, median ITL 26 to 133 ms; the same load tool as above gives 38.1 / 123.3 / 208.6 / 237.7 tok/s, TTFT 52 to 728 ms, ITL 26 to 130 ms. It is the strongest of the references at every level measured. The load tool sends short prompts and lets a request stop at its end of text; the comparison with vLLM's serving benchmark needs prompt-length distributions and fixed output lengths, which it gains in phase 3.
- **An expert exchange** took 0.85 to 4 ms a layer on 2 to 8 ranks at 8 tokens a rank (80 to 380 ms a pass of Qwen3-235B-A22B), and its floor of two host round trips a layer is over 40 ms a pass, against about 20 to 25 ms of decode: through the host, expert parallelism loses to the layer split. Chained on the devices through dma-buf and sync files the floor would be near 10 ms a pass (188 exchanges at about 55 us), which is not yet measured with more than two cards waiting on each other.
- **Measured, not adopted:** the backend's submission chunk of 64 dispatches, swept at 16, 24, 32, 48, 64, 128, 256 and 1024 on one MI50 over three interleaved rounds: nothing beats 64 outside noise (tg128 0.6B Q8_0 317 to 329 tok/s at 24 to 64), 16 costs Qwen3-30B-A3B 3 percent and 1024 costs it 17.
- **Left:** the device-side expert exchange on more than two ranks and vLLM on four cards, which open phase 6b; everything else phase 0 asked for is above and in `docs/MULTI-DEVICE.md`, Phase 0 results. Keeping a waiting stage's clock up in a single-stream layer split is a phase 2 question: llmx does not change a machine's power settings.
- **Gotchas:** multi-card runs need cards with no neighbour and link speed checked first; a llama.cpp reference run is pinned to its cards or it spreads over all ten. A clock held high must be restored by a trap.

## MoE decode's small kernels and the float tile's F32 rows (2026-09-24, branch feat/moe-decode-small-ops)

- **Goal:** the loose ends of MoE decode after the grouped decode merged: a token's time beyond its weight rows.
- **Found:** timed alone without timestamps (`llmx-moe-kernel-bench`, built with the Vulkan backend), Qwen3-30B-A3B's routed rows read at 580-680 GB/s on an MI50, while a token is about 600 dependent dispatches at a 4.1 us floor and its small kernels sat far above it: the RMS norm 11.4 us, the F32 router 13.4, the routing 10.9, the combine about 8. Each waited on its loads one at a time. Separately, F32 rows 2048 wide put every row of a tile step on one memory channel, and the router's 128 rows gave the float tile four workgroups: 360-400 us a layer at 32 to 512 prompt rows.
- **Done:** the small kernels load up front with their sums in the same order, and the norm's last tree steps run in one subgroup (logits over 64 positions and greedy text byte-identical to before on Qwen3-8B and 30B-A3B Q4_K_M); the float tile rotates F32 rows' inner steps by 128-row block and splits starved calls (`docs/VULKAN.md`, Kernel notes). On one MI50, base against this branch (tok/s): 30B-A3B tg128 120.0 / 129.0, pp32 357 / 410-412, pp128 628 / 663, pp512 1243 / 1248; 8B Q4_K_M tg128 93.1 / 96.4, pp512 871-873 / 874-876. The Radeon VII's Qwen3-0.6B prefill is level (Q4_0 pp512 +1 percent, pp64 within the spread).
- **Rejected:** two adjacent Q4_K or Q5_K rows a cluster (`research/k45-two-rows-rejected`): 8B +0.7 percent, 30B-A3B +0.2, 0.6B Q5_K_M -2.3.
- **Also rejected:** the combine folded into the down projection (`research/moe-combine-fold-rejected`): the last workgroup of a token's block of rows, found through an atomic counter, added the slots in the combine's order, exact but 1.6 percent slower single-stream and 4 percent at four sequences on the MI50, since every workgroup then ends on an atomic round trip while holding its slot.
- **MoE gate on main aead8e3, 2026-09-24:** one MI50, files one after another, llmx against the reference's Vulkan build, two interleaved rounds (mean ratio): Q4_0 pp64 182 percent, pp247 145, pp512 108, tg128 124; Q4_1 180, 175, 112, 119; Q4_K_M 128, 150, 109, 120; Q5_K_M 151, 159, 128, 115; Q6_K 156, 143, 126, 104. Q6_K decode, 99 percent before this branch, is ahead.
- **Left:** the dispatch count itself, about twelve a MoE layer, each at least 4.1 us on the MI50 and 7.6 on the Radeon VII; a fold through a completion count costs more than the dispatch it saves, so fewer dispatches needs kernels that do two steps' work without one. F32 weights 2048 wide now go through padded copies (branch `feat/f32-padded-rows`: 30B-A3B pp512 +1.8 percent on the MI50), but the router's activation rows are 2048 floats apart too, and at 512 prompt rows it still takes about 400 us a layer.

## MoE decode with many requests (2026-09-24, branch feat/moe-batched-decode)

- **Goal:** a server's MoE throughput growing with concurrent requests, well ahead of the reference's server at every level (the server gate).
- **Done:** a pass whose generated tokens carry at least two entries an expert groups them by expert and reads each expert's rows once per run of up to eight entries (`docs/VULKAN.md`, "Generated tokens beside each other"); `llmx bench --seqs N` measures decode passes of N sequences, which is what showed the plateau outside the server. Checked bit for bit against each token alone, the HF MoE gate and the suites on both machines. Server on one MI50, Qwen3-30B-A3B Q4_K_M, 64 tokens a request, main / this branch (tok/s): 115 / 115 at 1, 194 / 194 at 4, 217-218 / 218-222 at 8, 226-227 / 230-232 at 16, 225-226 / 262-263 at 32, time to first token at 32 from 848 to 606-614 ms; the reference's server gave 94, 149, 164, 110 and 203 on the same card earlier the same day. Plain decode is unchanged on both cards, since the grouped mode is a pipeline of its own (specialization constant 8 of the wide build): computed per column in every row kernel it had cost the MI50's integer-dot builds up to 7.5 percent, and tables filled up front the Radeon VII's Q8_0 build a fifth.
- **Tried and reverted:** grouping every batch of generated tokens, lone entries in the wide build (152 against 192 tok/s at four concurrent) and then split off to the one-column build (154): the loss was the grouping dispatch and its extra launches, not the build.
- **Measured after the merge, 2026-09-24:** `bench --profile` now reads the interval since its last reading, so it profiles a batched decode pass rather than a process's first 4096 dispatches. At eight sequences a pass the Q4_K gate and up rows of the routed entries take 40 percent of it and the Q6_K down rows 19, each entry about 7.9 us for 1.77 MB, 224 GB/s, where a dense Qwen3-8B decode reads its rows at about 460: an expert's matrix is 768 rows of 2048 values, so a dispatch is small whichever way it is cut. Capping a Q4_K or Q5_K row at 32 lanes in the integer-dot row kernels (`k45_row_lanes`) gave single-sequence MoE decode 120.6 to 123.2 tok/s and left eight a pass (249) and dense Qwen3-8B (94.0) where they were; 16 lanes cost the 8B 3 percent.
- **Left:** each routed entry is a small dispatch reading its rows at a quarter to half of what a dense decode reaches, so MoE throughput grows slowly with concurrency (1.23 times the reference's server at one request, 1.29 at 32); fusing an entry's gate, up and down into fewer, larger kernels is the next lever.

## CPU prompt rows on the prompt dots (2026-09-24, branch feat/cpu-dense-prefill)

- **Goal:** the CPU's dense prompt rows through the prompt dots that routed experts took on main (`q8_dots.hpp` `dot_block`), where that beats the batched float path, at or above the reference's CPU prefill.
- **Done:** K-quant rows (Q4_K, Q5_K, Q6_K) at least 4096 wide (`CpuBackend::kPromptDotsFrom`) meet a prompt through the prompt dots; narrower K-quant rows, Q8_0, Q4_0 and Q4_1 keep the float path, and generated tokens the decode dots, so a matrix takes one path per kind of row and a row computes the same however it is batched (`q8-dots` checks a prompt's rows beside a generated token against each alone, bit for bit, at 256, 2048 and 4096 wide). The full CPU suite and every real-model HF baseline pass on both machines.
- **Measured on the Linux machine's CPU (EPYC 7262, 16 threads, two interleaved rounds, tok/s, main / prompt dots / the reference's CPU path):** Qwen3-8B Q4_K_M pp64 21.9-22.1 / 30.7-31.2 / 38.4-39.1, pp247 33.4-33.7 / 32.2-32.3 / 39.4-40.3, pp512 19.9-20.0 / 31.4-31.8 / 40.7-40.8. Qwen3-0.6B Q5_K_M with the prompt dots on every width lost at 247 and 512 prompt tokens (415-431 against 378, 393-399 against 345-370), where its 1024-wide rows' dequantized blocks stay in the first-level cache; from 2048 wide it still lost 6 percent at 247, so the rule is 4096, which leaves the 0.6B files on main's path. The Q4_0 drop at 64 tokens seen in the first measurement did not repeat: main read 349 plus or minus 73 and 402 in two rounds, every arm within that.
- **Tried and reverted:** Q4_0 and Q4_1 rows on 8-bit activations, as the device takes them, with an output head's rows kept on 16 bits through `matmul_logits`. The HF gate's Q4_0 file then fails on the CPU ("The capital of France is": top-5 overlap 3 of 5 against a bound of 4). The two types stay on 16-bit activations.
- **Left:** the reference's CPU is still ahead on Qwen3-8B Q4_K_M (31-32 against 39-41 tok/s) and on Qwen3-0.6B Q4_0 (390-400 against 420-470); weights repacked into interleaved rows at load, so the prompt dots read several rows per load, are the next lever. The prompt dots' per-group horizontal sums follow from the activations' scale per 32 values and do not go away by blocking.
- **Gotchas:** the desktop's CPU timings wander with its other load; measure on the Linux machine, alone.

## Mixture of experts: qwen3moe on both backends (ROADMAP #2) (2026-09-23, branch feat/moe)

- **Goal:** Qwen3's mixture-of-experts form (`general.architecture = qwen3moe`, Qwen3-30B-A3B and Qwen3-Coder-30B-A3B) on the CPU and Vulkan backends, gated against HF, at or above llama.cpp's own Vulkan backend on the MI50 and the Radeon VII, and with experts on the CPU where the device is too small.
- **Done:** a layer is routed when its GGUF has `ffn_gate_inp`, so files that mix dense and routed layers load. Three backend ops carry a routed layer (`backend.hpp`): `route_experts` (softmax over the router scores, the top k, renormalized), `matmul_experts` (gate and up of each token's chosen experts) and `matmul_experts_add` (the down projection, weighted and summed in slot order into the residual). Expert ids and weights stay in the activation arena, so nothing leaves the device.
- **Done, CPU:** entries grouped by expert; a generated token's entries take the decode dots, all of a call's in one pool dispatch, and a prompt's entries a batched matmul per expert over its rows.
- **Done, CPU decode dots:** a decode row's activations are quantized once per call, 8-bit for Q8_0, Q4_K and Q5_K and 16-bit for Q4_0, Q4_1 and Q6_K, and meet the packed weights in integers (`backends/cpu/q8_dots.hpp`). The float dots converted every weight and were bound by arithmetic. On 8 bits for every type the HF gate's Q4_0 file, whose head is Q6_K, reached top-5 3/5 on "The capital of France is" against a bound of 4, as the device had found, so the three types the device reads on 16 bits read 16 bits here too; the gate then passes on all three fixtures. The CPU now picks a row's path by its runs as the device does, a generated token the decode dots and a prompt's rows the batched float path, dense and routed alike. `q8-dots` checks every type against a double-precision reference and a decode row alone, beside others and grouped, bit for bit. Qwen3-30B-A3B Q4_K_M on the 5800X, 8 threads: whole-model decode 8.0 to 16.9 tok/s (the reference's release build, whose CPU path this is, 11.3).
- **Done, Vulkan:** decode runs one entry per workgroup row through the row kernels, which take an expert offset on the weight rows; a prompt whose extent reaches its weight type's `moe_tile_from` (32 for Q8_0 and Q6_K, 48 for Q5_K, 64 for Q4_K, 96 for Q4_0 and Q4_1, measured on both cards; docs/VULKAN.md) takes the tile kernels over each expert's entries, grouped on the device by `moe_group.comp` (a workgroup per expert, stable order) and never split, so an entry computes the same whatever else is routed beside it. `moe_route.comp` routes through subgroup reductions and `moe_combine.comp` adds the weighted slots. The experts read the activation twin the router's input already has.
- **Done, placement:** `--n-cpu-moe N` and `--cpu-moe` put the experts of the first N (or all) routed layers on the CPU beside a device, through the per-role placement the model layer already had; attention, the dense blocks, the embedding and the head stay on the device.
- **Gates:** `tests/moe.py`, a tiny random-weight qwen3moe (two routed layers, one dense, 8 experts, top 3) against HF `Qwen3MoeForCausalLM` (`tools/gen_baseline.py moe`): all 257 logits within 7.5e-7 and windowed NLL within 1e-5 on the CPU, the Radeon VII and the MI50, across batch widths, threads and both placements. `tests/backend_vulkan.cpp` checks routing and the routed projections against the CPU for every supported type on the row kernel, the tile kernel and a batch mixing decode rows with a prompt. Qwen3-30B-A3B Q4_K_M gives the same greedy text on the CPU and the MI50.
- **Measured, Qwen3-30B-A3B Q4_K_M, one MI50 (idle), two interleaved rounds:**

  | test | llmx | llama.cpp Vulkan | share |
  |---|---:|---:|---:|
  | pp64 | 398 | 340 | 117% |
  | pp247 | 859 | 602 | 143% |
  | pp512 | 1165 | 1093 | 107% |
  | tg32 | 123.5 | 107.7 | 115% |
  | tg128 | 116.6 | 107.5 | 108% |

  Through the row kernels alone prefill was 383 against 1112; grouping by expert took it to 1048, and a grouping workgroup per expert, reused by the down projection, to 1214. Decode went from 99.3 to 116.6 with the twin reused past the router, subgroup routing (8 to 4 percent of decode time) and F32 router rows four values a load (8.8 to 4.0 percent).
- **Measured, experts on the CPU (b11075 on the Radeon VII, 8 threads; the reference on the same MI50, 16 threads):**

  | where | experts on CPU | test | llmx | llama.cpp | share |
  |---|---:|---|---:|---:|---:|
  | Radeon VII | 12 of 48 | pp512 | 196 | 121 | 162% |
  | Radeon VII | 12 of 48 | tg32 | 31.3 | 27.4 | 114% |
  | Radeon VII | 48 | pp512 | 75.6 | 76.7 | 99% |
  | Radeon VII | 48 | tg32 | 11.4 | 15.7 | 72% |
  | MI50 | 12 | pp512 | 223 | 404 | 55% |
  | MI50 | 12 | tg32 | 27.2 | 25.8 | 106% |
  | MI50 | 48 | pp512 | 69 | 166 | 42% |
  | MI50 | 48 | tg32 | 11.0 | 12.2 | 91% |

  12 routed layers on the CPU is what lets the rest fit the Radeon VII's 16 GB. With the decode dots over quantized activations (below), decode on the Radeon VII is 44.7 tok/s with 12 layers' experts on the CPU (163%) and 20.1 with all 48 (128%), and on the MI50 34.3 (133%) and 12.9 (106%); prefill is 195 and 76 on the Radeon VII, and 180 and 52 on the MI50. Some of the Radeon VII runs overlapped the model downloads and are to be repeated on a quiet machine.
- **Prefill with experts on the CPU, the MI50's gap:** the Linux machine is an EPYC 7262, 8 Zen 2 cores with eight memory channels, and the MI50 sits on PCIe 4.0 x16. The reference prefilled 151 tok/s at 512 rows with every expert on the CPU and 88.5 with `--no-op-offload 1`: from its CPU alone it is ahead of llmx's 52, and copying the CPU-held weights to the device for a large batch gains it another 1.7 times, which the Radeon VII's PCIe 3.0 link halves. Both are llmx's to close: a faster CPU path for a prompt's experts, then the same copy.
- **Done, mapped loading:** a single-file GGUF is mapped read-only instead of read into one heap allocation (`format/mapped_file.hpp`). Qwen3-30B-A3B Q8_0 is 32.5 GB: on the Radeon VII's 32 GB host with thirty layers' experts on the CPU the heap copy paged through every pass, and on the Linux machine's 62 GB host it sat beside its own page cache. The mapping lets the OS drop what the device copied. Sharded files keep the allocation that assembles them.
- **Done, device-held pages leave the host:** a model with experts on the host keeps the file mapped for them, and the pages of every tensor a device copied stayed in the process beside them: Qwen3-30B-A3B Q8_0 with thirty layers on the CPU held 21 to 22 GB on the Radeon VII's 32 GB host, which then read pages back from the file through every pass. Once the weights are resolved, the whole pages of each tensor only devices hold are given back (`MappedFile::drop`: `VirtualUnlock` on Windows, `madvise(MADV_DONTNEED)` elsewhere), and the process settles at 16.8 GB. Prefill at 512 rows went from 119.7 +- 33.5 to 129.6 +- 1.2 tok/s over eight runs, decode unchanged.
- **Done, prompt dots for experts on the CPU:** a prompt's routed entries met each expert through the float path, which converts every weight to a float for each block of four rows; they now meet it through quantized activations, each weight row unpacked 256 values at a time once for all of that expert's entries, integer sums per group of 32, eight groups' scales in one vector multiply-add (`q8_dots.hpp` `dot_block`). Generated tokens keep the fused decode dots, one kernel per kind of row, so a row computes the same alone or beside others (`q8-dots` checks prompt and decode entries beside each other against each alone, bit for bit). An earlier form that unpacked per four columns lost at 512 rows: each weight row swept every column's whole activation row, which no cache holds, and the block now walks the inner dimension so a column's 256 values stay in the first-level cache across eight rows. Qwen3-30B-A3B, experts on the CPU, streaming off (tok/s):

  | where | test | llama.cpp Vulkan | llmx before | llmx now |
  |---|---|---:|---:|---:|
  | MI50, Q8_0, 12 layers, 16 threads | pp64 | 54-57 | 105-109 | 166 |
  | MI50, Q8_0, 12 layers | pp247 | 147-148 | 180-186 | 267-270 |
  | MI50, Q8_0, 12 layers | pp512 | 287-293 | 230-235 | 314-316 |
  | MI50, Q8_0, 12 layers | tg16 | 20.0-20.3 (tg128) | 30.1-30.6 | 30.2-30.8 |
  | Radeon VII, Q4_K_M, 10 layers, 8 threads | pp64 | - | 82 | 123 |
  | Radeon VII, Q4_K_M, 10 layers | pp247 | - | 177-179 | 229-230 |
  | Radeon VII, Q4_K_M, 10 layers | pp512 | - | 230-240 | 271-272 |
  | Radeon VII, Q4_K_M, 10 layers | tg32 | - | 51 | 51 |

  The MI50's reference figures are from the three-way run on the same card, the llmx columns interleaved with each other. With this the CPU path beats the reference at 512 rows without copying experts, so `--moe-stream-from` now defaults to 0.
- **Done, streamed experts (`--moe-stream-from`, default 0 since the prompt dots; measured while it was 512):** from that prompt extent a host-placed routed layer runs on its attention device, the norm and router copied there at load and the experts copied into one window per device once per pass; decode rows and shorter prompts stay on the host, a mixed server pass split into groups of consecutive entries. The host upload now fills one half of staging while the device copies from the other. The copy is a fixed cost per pass, so the break-even differs by link: on the MI50 7.7 GB (twelve Q8_0 layers) takes 0.9 s (11 GB/s by DMA alone, under what the MI50's PCIe 4.0 x16 link allows), on the Radeon VII 19.2 GB (thirty layers) takes 3.3 s (6.4 GB/s by DMA alone). First keyed on extent like every kernel choice, which made a short follow-up in a conversation past the threshold pay the whole copy; it now follows the tokens the request prefills, its reused prefix excluded (`BatchEntry::fresh`), so every slice of a prompt still takes one path, and a reply on a cached prefix may take the CPU where one pass over the whole conversation would stream. On the MI50 with twelve Q8_0 layers on the CPU and the prompt dots, streaming gives 411 tok/s at 512 rows against 311 on the CPU and the reference's 296-299, and loses below that (223 against 268 at 247). Same greedy text as the host path on Qwen3-30B-A3B Q8_0; `tests/moe.py` adds streamed placements (every run, and prompts from extent 4) within 7.2e-7 of HF, and `tests/server.py` checks the synthetic MoE model's ids alone and four at a time with streamed prompt rows beside host decode rows.

  | where | test | host path | streamed |
  |---|---|---:|---:|
  | MI50, 12 layers on CPU | pp16 | 50.8 | 17.8 |
  | MI50, 12 layers on CPU | pp64 | 128 | 64.7 |
  | MI50, 12 layers on CPU | pp128 | 123 | 125 |
  | MI50, 12 layers on CPU | pp247 | 175 | 226 |
  | MI50, 12 layers on CPU | pp512 | 230 | 412 |
  | Radeon VII, 30 layers on CPU | pp128 | 92.7 | 37.1 |
  | Radeon VII, 30 layers on CPU | pp247 | 114 | 69.4 |
  | Radeon VII, 30 layers on CPU | pp512 | 118 | 132 |

  A 256 MB staging buffer instead of 64 MB changed nothing. What is left is the copy overlapping the previous layer's compute, which needs a second window and a transfer queue (at pp512 on the MI50 the compute is about 0.35 s of the pass's 1.25 s).
- **Measured, Qwen3-30B-A3B Q8_0 with experts on the CPU (one MI50 with 12 layers and 16 threads, the reference pinned to the same card; Radeon VII with 30 layers and 8 threads against b11075; two interleaved rounds each):**

  | where | test | llmx | llama.cpp Vulkan | share |
  |---|---|---:|---:|---:|
  | MI50 | pp64 | 102-106 | 56-58 | 180% |
  | MI50 | pp247 | 188-212 | 152-153 | 123-139% |
  | MI50 | pp512, host path | 229-241 | 291-305 | 75-83% |
  | MI50 | pp512, streamed | 412 | 291-305 | 135-141% |
  | MI50 | tg128 | 28.3-28.7 | 20.6-20.8 | 138% |
  | Radeon VII | pp64 | 56-57 | 6.1-7.6 | 740% |
  | Radeon VII | pp247 | 93-107 | 49-50 | 188-216% |
  | Radeon VII | pp512 | 117-120 | 90-91 | 129-132% |
  | Radeon VII | tg128 | 18.5-18.9 | 14.2 | 131% |

  The Radeon VII's llmx prefill still varies by up to 35 tok/s between repeats, the host holding 19 GB of experts in 32 GB; the MI50 streamed row is a single round.
- **Done, decode on the 8-bit twin:** the first MI50 gate run had Q6_K, Q4_0 and Q4_1 decode at 87 to 93 percent, all three rows reading the 16-bit twin because the HF gate's Q4_0 fixture had failed on 8 bits. That failure was the Q6_K output head: `Backend::matmul_logits` names the head, which keeps the 16-bit twin for those types, and every other Q6_K, Q4_0 and Q4_1 row reads the 8-bit twin (Q6_K folding its -32 into each weight byte). Decode at tg128 went from 82.0 to 92.4 tok/s on Q6_K (with at most 32 lanes a row, `q6k_row_lanes`), from 101.8 to 131.8 on Q4_0 and from 101.4 to 128.8 on Q4_1; the HF gate passes on all three fixtures. A second gate run with these is going.
- **Done, server:** an uncapped request reserved its whole reach, the whole pool, so Open WebUI's chats and background requests ran one at a time and each admission dropped the cached prefixes. It now reserves its prompt and a step and grows, the latest admitted uncapped request pausing (history kept as a donor) when the pool runs out; `tests/server.py` checks three uncapped requests sharing a small pool with at least one pause. The compatible replies carry a `timings` object, which Open WebUI shows, and each request logs a line.
- **Tried and reverted, 2026-09-23:** the integer-dot tile reading the 8-bit twin a producer wrote, in position order, instead of quantizing its own block-major copy. The values are the same, but at 4096 wide a column's blocks are 4 KB apart and the loads stop coalescing: Qwen3-8B Q4_K_M prefill on the MI50 fell from 693 to 368 tok/s at 64 rows and from 845 to 775 at 512, with no gain on the 0.6B files.
- **Tried and reverted, 2026-09-23:** a decode token's down projection summing its eight slots into the residual inside the row kernel, one token per workgroup row, instead of writing the slots and adding them with `moe_combine`. Same arithmetic, but a workgroup per token running its slots in turn left 256 workgroups where there had been 2048, and MI50 decode fell from 92.8 to 90.7 tok/s on Q6_K and from 118.7 to 118.0 on Q4_K_M (the reference 93.2 and 107.6 in the same interleaved runs). Q6_K decode stays at 99.5 percent of the reference; the F32 router rows already take a whole subgroup each.
- **Tried and reverted:** a prompt's routed entries through the decode dots as well, a dot per weight row and entry. Each dot unpacks the row's nibbles and scales again, where the batched float path unpacks a row once for all its expert's entries, and prefill with every expert on the CPU went from 52 to 35 tok/s on the Linux machine (during a download). A prompt's entries want a multi-column kernel that unpacks a row once.
- **Left:** the prompt dots for the CPU's dense prompt rows, which still take the float path; the copy overlapping compute;  and the dense CPU cells before and after the decode dots are to be measured; a real file of every supported type (Q5_K_M, Q6_K, Q8_0, Q4_0, Q4_1 downloading on both machines) through the gate cells; CPU expert decode (the fused dots against float activations) and prefill with experts on the CPU, where the reference likely runs large batches on the device from host-held weights; a server check of routed layers on a real model.
- **16k check, redefined 2026-09-24:** a hash across the CPU and the device parted at the first near-tie: on Qwen3-0.6B-Q8_0 they agreed for 68 characters, where the CPU's top two logits were 18.498 and 18.379 and the device's 18.383 and 18.346, the two tokens swapped, the device's logits sitting up to 0.23 from the CPU's after the 16k prompt as the 8-bit activations shift them. With the user's agreement `tools/long_context_check.py` now requires the device to give the same 512 greedy tokens after a 16384-token prompt on two runs from fresh servers, and the CPU, reading the prompt and those tokens (`llmx logits --last`), to rank each within 0.5 logits of its top choice. Passed on every run: Qwen3-0.6B-Q8_0 on the MI50 (the CPU's top choice at 509 of 512 tokens, the largest gap 0.135) and on the Radeon VII (512 of 512, 0.000), Qwen3-8B Q4_K_M on the MI50 (510 of 512, 0.014).

## Main documentation checkpoint (2026-09-24)

Corrected fixture counts, ticket/completion descriptions, vendor quant support,
device selection, Q8 overflow scope and historical GPU status against main.
Recorded the user's independent-feature branch rule in AGENTS. All 35 Markdown
files are ASCII, 118 local links/anchors resolve and `git diff --check` passes.
No source, tests, build configuration or historical measurement tables changed.
Main remains dense Qwen3 with capped server requests; unmerged feature claims
were excluded. This documentation change is separate from the help fix below.

The parent help commit `5afc1c7` passed all six hosted jobs in
[CI run 35964866124](https://github.com/mxxm-t/llmx/actions/runs/35964866124):
Windows, Ubuntu, macOS, Linux UBSan, Vulkan build and required real-model HF.

## Command help checkpoint (2026-09-24)

`llmx --help` prints a grouped overview; each of the 12 commands accepts
`--help` or `-h` for its own options, defaults and example. Execution options
share one renderer. Help returns before model, backend or Hub access. Existing
command parsing and positional text behavior are unchanged. The help and
reference now describe seed zero as retaining the fixed default RNG state.

This change is based directly on `cdf1cdb`, independently of in-flight features.
The main-based Windows Release build passes:

| Check | Observed | Required |
|---|---:|---:|
| Help routes | 26 passed | 26 |
| Missing or unsupported advertised parser flags/aliases | 0 | 0 |
| Exit/positional-text checks | 5 passed | 5 |
| Native tests | 20 passed | 20 |
| Python components with all required real-model fixtures | 13 passed | 13 |
| Real-model HF top-1, each of Q8_0/Q4_0/Q5_K_M | 6/6 | 6/6 |

| HF NLL comparison, maximum over batched and per-token paths | Observed absolute error | Existing bound |
|---|---:|---:|
| Q8_0 continuous | 0.001284 | 0.01 |
| Q8_0 windowed | 0.012376 | 0.02 |
| Q4_0 continuous | 0.131554 | 0.16 |
| Q4_0 windowed | 0.167600 | 0.20 |
| Q5_K_M continuous | 0.026144 | 0.05 |
| Q5_K_M windowed | 0.129480 | 0.16 |

Reproduction: CMake Release build, `ctest --test-dir build -C Release
--output-on-failure`, then `python -u -X utf8 tests/run_tests.py --exe
build/Release/llmx.exe --no-perf-floor --require-baseline`. The standalone help
check also compares each command's emitted options to its parser, rejecting
both missing and unsupported flags. Raw logs and help output remain under
`%TEMP%/llmx-help-main-20260924/build/`. Performance timings are diagnostic only;
this change makes no new performance claim.

All 35 tracked Markdown files were reviewed against main: ASCII and 116 local
links/anchors checked. Help-related claims were corrected here. Independent
findings were subsequently addressed by the separate documentation checkpoint
above; they were not included in the help implementation commit.

## Multi-user server (ROADMAP #7, EXECUTION step 7) (2026-09-22)

- **Goal:** the HTTP front-end over the model layer the execution plan
  built for it: one shared model, a `Sequence` per request, continuous
  batching with chunked prefill through one `Model::forward` per scheduler
  iteration, streaming responses, prefix reuse through `fork`, admission by
  the KV pool's budget. Dependency-free transport. The design, protocol,
  scheduler loop, gates and order of work are `docs/SERVER.md`.
- **Done:** the design and steps 1 to 6 (3a and 4 to 6 further down).
  `src/server/http.hpp`: HTTP/1.1 over blocking sockets, Winsock or BSD, a listener, one request with a
  Content-Length body per connection, a whole response or a chunked
  stream, a client for tests; the `http` CTest covers a whole response, an
  echoed body, a three-chunk stream arriving as written, 413, 400, 404 and
  the listener closed from another thread, on Windows and on the Linux
  machine. `src/server/scheduler.hpp`: the loop of SERVER.md, admission by
  the pool's capacity with every admitted request's blocks reserved up
  front (admitting on blocks merely free let four requests into a
  one-block pool), decode entries then chunked prompt slices in one
  `forward`, per-request seeded sampling, channels, cancellation, a
  failed pass ending its requests and not the loop. `src/server/api.hpp`
  and `llmx serve`: `/v1/generate`, `/v1/chat` through the template
  renderer, `/v1/health`, `/v1/models`, streamed as server-sent events
  with characters held until complete and any bytes that never form a
  character replaced by U+FFFD, since a byte-level vocabulary under
  sampling produces them. The `server` Python component runs on the
  synthetic F32 model without a download and on the Q8_0 fixture:
  greedy through the server equals `generate --temp 0` alone and four at
  a time, a stream carries the same ids, a seeded request repeats,
  refusals, a client leaving mid-stream leaves nothing active, a chat
  turn; it passes on the CPU and on the device. `tools/server_load.py`
  measures aggregate decode throughput at N concurrent requests against
  llmx's route or the reference's `/completion`. On the device, Qwen3-0.6B
  Q8_0, 64 tokens per request, best of two rounds, both servers in the
  same minutes (the reference with 16 slots over an 8192 context):

  | concurrency | reference server | `llmx serve` | llmx share |
  |---|---:|---:|---:|
  | 1 | 176.9 tok/s | 210.6 | 119% |
  | 4 | 367.6 tok/s | 328.9 | 89% |
  | 8 | 453.3 tok/s | 369.1 | 81% |
  | 16 | 237.1 tok/s | 244.9 | 103% |

  Per step llmx cost about 4 ms more per extra sequence, which was the
  device's multi-view path: `kv_write` and `attention` dispatched once
  per view, and `norm_rope_kv` fused only a single view and fell back to
  its three-op default for a batch, so a sixteen-sequence step was on the
  order of a thousand dispatches. SERVER.md step 3a closed it: every
  cache kernel now takes a view table (`shaders/views.glsl`: per view its
  batch row, dispatch-local row, row count, history, block-table offset
  and length, then every view's block ids, uploaded through the args
  arena) and runs once per layer over every view; attention splits a
  batch into the views the tiled kernel takes (32 rows or more, 128-wide
  heads) and the rest for the per-row kernel, at most two dispatches, so
  a decode row never sits in a tile staging its whole history for one
  live row; the merge kernel reads the same table since a dispatch's rows
  need not be a prefix of the batch. On the way a latent hazard surfaced:
  a split or subgroup that saw no token held -inf and its state came out
  as exp(-inf - -inf); both merges now skip such a part, which the
  kv-cache checks over two views had not reached before. Every kernel
  check, the HF gate on the device and the server component pass; the
  same load again, both servers in the same minutes:

  | concurrency | reference server | `llmx serve` before | `llmx serve` now | llmx share |
  |---|---:|---:|---:|---:|
  | 1 | 177.5 tok/s | 210.6 | 209.0 | 118% |
  | 4 | 365.6 tok/s | 328.9 | 409.2 | 112% |
  | 8 | 462.6 tok/s | 369.1 | 504.3 | 109% |
  | 16 | 237.1 tok/s | 244.9 | 297.1 | 125% |

  The throughput gate is met at every level. Both servers fall at 16:
  for llmx the row kernel holds eight columns per dispatch, so sixteen
  sequences stream the weights twice per matmul, which is recorded, not
  fixed.
  SERVER.md step 4, prefix reuse: a finished request's history stays as a
  donor (at most `max_seqs` of them, the oldest dropped when a request
  needs its blocks) and a new prompt forks the donor sharing the longest
  run of full blocks, rolled back to those blocks through the new
  `Model::truncate`, so only the rest of the prompt is prefilled. Tokens
  are compared, not hashed: a server holds a handful of donors for one
  model, and the hashed key of KV-CACHE is for an index that outlives a
  process. The `server` component sends the 247-token excerpt with two
  endings and requires the second to reuse blocks and give the CLI's
  greedy text; it passes on the CPU and on the device. `/v1/health`
  reports `donors`, `prefix_hits` and `prefix_tokens`, a reply
  `reused_tokens`. Whole-request wall time at the client, 16 generated
  tokens, one request at a time:

  | model | backend | prompt | first request | with a donor | reused |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | Vulkan | 1995 tokens | 1.53 s | 0.16 s | 1984 |
  | Qwen3-8B-Q8_0 | Vulkan | 1995 tokens | 8.74 s | 0.72 s | 1984 |
  | Qwen3-0.6B-Q8_0 | CPU | 1995 tokens | 6.95 s | 0.95 s | 1920 |

  The reuse is bounded by the block size (64 on the device, 128 on the
  CPU) and by the last prompt token, which is always prefilled for its
  logits.
  SERVER.md step 5, the second execution context, measured and not added.
  A timing build split every scheduler pass on the device into the
  `forward` call (recording and submitting), the wait for its logits and
  the host's work until the next `forward`, Qwen3-0.6B-Q8_0, 64 tokens
  per request, 200 passes each:

  | sequences | forward call | wait | host between passes | per pass |
  |---:|---:|---:|---:|---:|
  | 1 | 1.6 ms | 3.4 ms | 0.02 ms | 5.0 ms |
  | 4 | 2.9 ms | 6.6 ms | 0.02 ms | 9.5 ms |
  | 8 | 4.5 ms | 10.7 ms | 0.03 ms | 15.2 ms |
  | 16 | 18.7 ms | 30.9 ms | 0.03 ms | 49.6 ms |

  A second context could hide only the last column, and there is nothing
  there to hide: greedy sampling over the vocabulary is tens of
  microseconds. The same build with one submission per pass instead of
  one every 64 dispatches puts pure recording at 0.7 ms per pass at one
  sequence and 1.7 at eight, the rest of the `forward` column being the
  chunked queue submissions, and costs 2 percent of `bench` decode (218
  to 213 tok/s), so the chunking stays. The recording cannot overlap the
  device either way, since the next pass's tokens come from this one; a
  recorded pass replayed with the next tokens, positions and view table
  written into the arena would recover it, at most 15 percent of a
  0.6B decode pass and about 3 percent of an 8B one. Recorded as an open
  backend lever, not built: it is a small-model gain and the device is
  already past the reference on those.
  SERVER.md step 6, the compatible routes the user asked for so that
  the tools people already run connect to `llmx serve` unchanged:
  `/v1/chat/completions`, `/v1/completions` and `/v1/models` in the
  shape the OpenAI clients speak, the shape vLLM and the reference's
  server expose too. They are a JSON mapping in `src/server/api.hpp`
  over the same scheduler: one parse, one request, one drain loop, the
  native routes' knobs accepted as extra fields and the clients'
  synonyms beside them, everything validated before the model, errors
  in the clients' `{"error": {"message", "type"}}`. The `server`
  component checks greedy equality with the CLI through
  `/v1/completions` whole and streamed, the usage counts, the role in
  the first chat chunk and the finish reason in the last, text content
  parts and the refusals; it passes on the CPU and on the device.
  The serving gate, restated by the user: the reference's server is the
  weaker of the serving runtimes at concurrency and the one that runs on
  this hardware, so llmx must beat it by a wide margin on the figures a
  serving runtime is judged by, not reach parity. `tools/server_load.py`
  now streams every request and reports time to first token and
  inter-token latency at the median and the 99th percentile, tokens per
  second and requests per second. Qwen3-0.6B-Q8_0 on the device, 64
  tokens per request, the reference with 32 slots over an 8192 context,
  llmx with 32 sequences, both servers in the same minutes, best of two
  rounds, the two passes' spread shown where it matters:

  | concurrent | tok/s reference | tok/s llmx | TTFT p50 reference | TTFT p50 llmx | ITL p50 reference | ITL p50 llmx | ITL p99 reference | ITL p99 llmx |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 1 | 170, 173 | 187, 187 | 21, 29 ms | 26, 26 ms | 5.5 ms | 5.0 ms | 6.5, 6.6 ms | 5.5, 5.6 ms |
  | 4 | 357, 361 | 401, 406 | 105, 113 ms | 61, 64 ms | 9.6 ms | 9.0, 9.1 ms | 11.1, 11.2 ms | 9.7 ms |
  | 8 | 455, 470 | 509, 511 | 106, 150 ms | 61, 62 ms | 15.4, 15.6 ms | 15.0 ms | 17.1, 18.3 ms | 15.8, 15.9 ms |
  | 16 | 231, 232 | 546, 555 | 325, 353 ms | 78, 98 ms | 64.6, 64.7 ms | 27.9, 28.0 ms | 69.0, 74.4 ms | 29.7, 33.2 ms |
  | 32 | 544, 548 | 560, 563 | 469, 525 ms | 109, 119 ms | 51.1, 51.3 ms | 56.0, 56.1 ms | 58.5, 65.2 ms | 57.3, 58.0 ms |

  This table is with the tile threshold, the tool's warm-up and the
  chosen tile shape all in place, both servers interleaved in the same
  minutes, two passes each. Throughput now leads at every level
  measured: 108, 112, 110, 238 and 103 percent of the reference at 1,
  4, 8, 16 and 32 concurrent. The 32 case was the last one behind, at
  95 percent, and what closed it was not a server change at all but the
  prefill tile shape becoming a choice (the Vulkan block's thirtieth
  paragraph): a prompt chunk joining a decode batch arrives at 16 to 32
  rows, which is exactly the band that changed, and the server went
  from 517 to 563 tok/s there with time to first token from 165 to
  109 ms. Time to first token is 1.7 to 4.8 times shorter from 4
  concurrent up, since a prompt joins the running batch as a chunk
  rather than waiting for a slot; the inter-token p99 sits within 2 ms
  of the median at every level on llmx. Before the threshold, 16
  concurrent had been 282 tok/s with a p99 of 55 ms: the pass a prompt
  chunk joined at 16 rows and up took the tile kernel, which costs a
  64-row tile whatever its fill. Two earlier readings were the tool,
  not the server: a p99 of 13 and 62 ms at 1 and 4 concurrent was the
  server's first pass past sixty tokens and its first four-way pass,
  which an eight-token warm-up did not reach; twenty single requests
  in a row show one gap above 9 ms, in the first request at token 60.
  The tool now warms with one request of the measured length. A
  warm-up at the widest level is not neutral, it halved the reference's
  rate at 4 to 16, so the tool does not do that. llmx's TTFT at 1
  concurrent reads 27 ms here against 9 ms in steady state: the first
  request after an idle spell costs 20 to 35 ms more, whatever the
  prompt, on the client's first connection or the device's clocks, and
  a median of two rounds carries it. Prefix reuse, measured: an
  830-token prompt sharing six full blocks with a donor takes 77 ms to
  its first token against 160 ms fresh; its encode is 0.7 ms and the
  fork under 0.1 ms, and the 77 is the pass over the 62 rows that
  follow the shared blocks, which the bench prices at 56 ms without a
  history. On this model a pass of 32 to 128 rows costs a near-flat 50
  to 77 ms (the Vulkan block's twenty-seventh paragraph), so that is
  what a short prompt or a hit's tail pays. The 16-column kernel
  remains open.
  The limits, asked for by the user as the flags a deployment sets:
  the KV pool's budget had been one model context in total, shared by
  every request with no knob, and the queue unbounded. `--ctx-size`
  (`-c`) on `llmx serve` is now the pool's total token budget, the
  model context by default, rounded up to whole blocks, and the
  ceiling on one request's prompt plus `max_tokens` is the smaller of
  the context and that budget, refused with 413 in the API and in the
  scheduler. `--max-queue` (default 64) bounds the requests waiting
  for admission; past it a submit throws `QueueFull`, which the API
  answers with 503. The server test starts a one-sequence, one-queue
  server over 512 tokens and checks the 413 and, with three requests
  arriving 0.2 s apart, statuses 200, 200 and 503. Found on the way:
  `serve` was not among the commands the suite's `--device` reached,
  so the server had run on the CPU in every device pass while the CLI
  it was compared with ran on the device; it is now, and the test
  passes with both on the device.
- **Left:** the 16-column row kernel if sixteen-way batches turn out to
  matter; replaying a recorded decode pass, above, if small-model decode
  becomes the target.
- **Gotchas:** the scheduler thread is the only caller of `forward` for its
  devices, by contract; connection threads queue and drain. A request is
  admitted only when the pool holds its prompt plus `max_tokens`; admitted
  requests are never evicted, donors are. A donor's blocks are shared
  read-only, and a fork rolled back to a block boundary appends into fresh
  blocks, which is what `KVSequence::prepare` requires. Per-request seeded
  sampling keeps a request reproducible whatever it is batched with.

## Vulkan backend, sub-step 1 of docs/VULKAN.md (2026-09-21)

- **Goal:** the first vendor backend over the Radeon VII: storage and
  submission first, kernels in the following sub-steps.
- **Done:** `src/backends/vulkan/vulkan_backend.cpp`, the one translation
  unit outside the header-only runtime, built as a static library only
  with `LLMX_HAS_BACKEND_VULKAN=ON`. The loader is loaded at run time and
  every entry point fetched through it, so nothing links against
  `vulkan-1` and a build without the option is byte-for-byte what it was.
  One instance, the physical device by index, one compute queue without
  graphics, timeline semaphores required and 8- and 16-bit storage and
  arithmetic enabled where present, push descriptors enabled where
  present. Buffers are one `VkBuffer` on their own memory: device-local
  and not host-visible for `Memory::device`, host-visible, coherent and
  cached for `Memory::host_visible`, mapped for their lifetime. `adopt`,
  `write` and `read` go through a 64 MiB staging buffer in chunks;
  `alloc` zero-fills in stream order; `copy` is a device copy. `submit`
  ends the open command buffer and signals the timeline with the ticket,
  `wait` blocks on the value, `sync` submits what is open and waits on the
  last ticket, and a ring of four command buffers is reused as tickets
  retire. One full barrier between consecutive commands. The compute ops
  throw naming their sub-step. `backend-vulkan` passes 13 checks on the
  Radeon VII and exits 77, which CTest reports as skipped, without a
  loader or device. The CMake option-to-flag conversion that the generated
  config needed is exercised by every configure and the default tree is
  unchanged: 19/19 native, Python 12/12.
- **Slip, recorded:** these files were swept into `757dee9`, the commit
  that recorded the fork gate, before they had been built, by a `git add
  -A` that should have been scoped. The default tree was verified within
  minutes and was never broken, since the option is off; the Vulkan tree
  was built and its test run right after, and this commit carries the
  description the previous one lacked.
- **Done: sub-step 2.** Six GLSL compute shaders under
  `src/backends/vulkan/shaders/`, compiled by `glslc` at build time into
  the generated include directory as numeric arrays and embedded, with a
  shared `q.glsl` for block decoding. Pipelines are built on first use
  with a push-descriptor set layout and 128 bytes of push constants; a
  dispatch binds, pushes buffers and constants, launches and fences.
  Elementwise kernels are one invocation per element; the norms are one
  workgroup per row or per (row, head) with a shared-memory reduction, so
  they do not depend on the subgroup size; `embed` decodes F32 and Q8_0
  rows, the table bound once as floats and once as bytes. `backend-vulkan`
  compares every kernel against the CPU backend on random inputs, bounds
  fixed in the test before the first run: exact for add, gather and both
  embed paths, 1e-6 relative for SiLU, 1e-5 for the norms and RoPE;
  160,688 outputs match on the Radeon VII, and out-of-range rows, positions
  and ids are refused on the host. The default tree is untouched by any of
  it.
- **Done: sub-step 3, the row kernel.** `matmul` for F32 and Q8_0: one
  subgroup per output row, lanes striding over blocks, batch columns in
  chunks of eight held in registers, one `subgroupAdd` per column. Q8_0
  rows with an even block count are read as 32-bit words over block pairs
  with the activations as 16-byte vectors, which took the 4096-square Q8_0
  matvec from 32 GB/s to 201 GB/s on the Radeon VII; odd block counts keep
  the 16-bit path. The subgroup size is queried and must divide 256.
  Checked against the CPU backend over batch widths 1, 3, 8 and 13 and
  both parities at 1e-4 relative; 167,388 kernel outputs match in all.
- **Done: sub-step 4.** KV blocks on the device, one K and one V buffer
  per layer holding `[kv_head][token][head_dim]` in 64-token blocks, grown
  by allocate and copy on the queue with the buffers the copy reads from
  kept alive until it retires. `kv_write` scatters a view's rows; `kv_copy`
  copies a block within each layer's buffer; `attention` is one workgroup
  per (row, head) with the subgroups taking tokens round robin, one
  `subgroupAdd` per token for the score, an online softmax so no score
  array is needed, and a shared-memory merge of the subgroups. Checked
  against the CPU backend through each backend's own storage and block
  size on histories of 0, 63, 64, 65 and 131 tokens with 1 and 3 queries,
  two views in one call, and a copied block attending like its source, at
  1e-4 relative; 179,228 kernel outputs match in all. The device's bounds
  checks caught a query offset in the test that the CPU backend reads
  through silently.
- **Done: sub-step 5, all but the floor.** `--device cpu|vulkan:N` on
  `generate`, `chat`, `logits`, `perplexity` and `bench`; a build without
  the backend says so rather than falling back. The Python suite takes
  `--device` and passes it through `LLMX_DEVICE`, test configuration like
  `LLMX_BASELINE_GGUF`; a fixture the device has no kernel for is reported
  as skipped, with the reason, rather than failed. On the Radeon VII the
  whole suite passes with every command on the device: the synthetic F32
  and sharded HF cases, the chat and thread goldens, and the real Qwen3
  Q8_0 baselines, whose top logits and all four perplexity cases match the
  CPU's numbers to the digit (PPL 28.8371 continuous, 38.2140 at c=123).
  Q4_0 was skipped until sub-step 6 gave the device its Q4_1 and Q6_K
  decoders; it passes now, see below.

  Throughput after the arena fix, 247-token prompt and 32 greedy tokens,
  one run each, no gate: 0.6B prefill 382 tok/s and decode 67.6, against
  the CPU's 431 and 46.7 at six threads; 8B prefill 40.3 and decode 21.3,
  against the CPU's 37 and about 4. The first numbers were 19 and 10.9
  tok/s decode, because every dispatch taking ids or positions allocated
  a device buffer, over a hundred `vkAllocateMemory` calls per token; a
  host-visible arena per ring slot removed that. Prefill on the device is
  the row kernel streaming the weights once per eight columns, which the
  tile kernel is for.
- **The floor, measured.** The pinned mx revision's Vulkan backend did
  not build here: its shader generator's nested configure fails under
  CMake 4.x with "CMAKE_C_COMPILER not set", with both the pip and the
  Visual Studio CMake, and is left for a CMake 3.x retry. The reference
  used instead is today's upstream llama.cpp Windows Vulkan release,
  b11075, which is a stricter bar since its kernels are newer. Same card,
  same files, 247-token prompt and 32 generated tokens; llmx numbers are
  single runs after the changes below, the reference is `llama-bench`
  with three repeats.

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | prefill | 660 tok/s | 1195 (three runs within 1%) | 181% |
  | Qwen3-0.6B-Q8_0 | decode | 198 tok/s | 111 | 56% |
  | Qwen3-0.6B-Q4_0 | prefill | 636 tok/s | 1110 (997, 1110, 1110) | 175% |
  | Qwen3-0.6B-Q4_0 | decode | 209 tok/s | 107 (90, 106, 108) | 51% |
  | Qwen3-8B-Q8_0 | prefill | 99 tok/s | 220 | 222% |
  | Qwen3-8B-Q8_0 | decode | 39.7 tok/s | 32.7 | 82% |

  Prefill clears the reference on both models since the tile kernel;
  decode is at about half of it, and the roadmap's bar is the reference,
  not the CPU. Where the time goes, measured per kernel at the 0.6B shapes rather
  than guessed: a near-empty dispatch with its barrier costs 6.3 us, so
  the four hundred dispatches of a token are under 3 of its milliseconds;
  the Q8_0 matvec has a floor of about 17 us at 1 MB and streams at 65 to
  125 GB/s on the 1 to 3 MB layer matrices and about 210 GB/s at 8B's
  sizes, on a card whose memory does 1 TB/s; decode attention was 134 us
  per layer with one workgroup per head walking the whole history. Three
  changes so far: the inter-dispatch barrier names the compute and
  transfer stages instead of all commands (decode +14% on 0.6B); a
  subgroup takes several rows with a lane cluster each, so a 1024-wide
  row no longer idles three quarters of its lanes (neutral, so occupancy
  was not the limit); and attention splits the history into 32-token
  chunks across workgroups with a merge kernel, 134 to 36 us per layer,
  decode 78 to 99.5 tok/s on 0.6B. `backend-vulkan` reports the per-shape
  timings so the next change is measured against them. Fourth, the tile
  kernel for wide batches: a workgroup computes a 64 x 64 output tile
  with the weights dequantized once into shared memory and every thread
  accumulating a 4 x 4 micro-tile, so a weight is read once per pass
  rather than once per eight columns; batches of 16 and up take it.
  Prefill 449 to 1025 tok/s on 0.6B and 40 to 220 on 8B, checked against
  the CPU at batch widths 16, 64, 100 and 247 at 1e-4.
  Fifth, the matvec's access pattern: eight consecutive lanes share one
  block pair, lane t loading words t, t + 8 and, for lane 0, word 16, so
  a load instruction touches 32 contiguous bytes per pair instead of
  four, and every lane reads the two scales directly, which the hardware
  serves as one broadcast and which beat a shuffle. The 8B matrices went
  from 210 to about 295 GB/s and 8B decode from 21.8 to 28.1 tok/s. Two
  variants measured and rejected: a generic loop over words with
  accumulator arrays indexed by column spilled to scratch and ran at 27
  GB/s, and two pairs per iteration cost occupancy and lost a few percent.
  Sixth, grouped projections: the row kernel takes up to three
  projections of one X in one dispatch, workgroups handed to projections
  in order so a workgroup's buffers are selected once, which is dynamic
  indexing of a storage buffer array and a device feature the backend
  now requires. q, k and v are one dispatch, gate and up another, checked
  bit for bit against the same projections one at a time. Decode 104 to
  110 tok/s on 0.6B and 28.1 to 29.3 on 8B.
  Seventh, the lanes per block pair swept: one, two, four, eight and
  sixteen give 200, 190, 336, 295 and 185 GB/s at the 8B shapes, so four
  it is, each lane with five loads in flight over 16 contiguous bytes.
  8B decode 29.3 to 32.7 tok/s.
  Eighth, sub-step 6 for the Q4_0 fixture: Q4_0, Q4_1 and Q6_K in
  `embed`, the tile kernel and the row kernel, checked against the CPU
  (exact for embed, 1e-4 for the matmuls; 599,588 outputs), and the
  fixture's HF baselines pass on the device with the CPU's numbers
  (PPL 32.8463 continuous, 42.5740 at c=123). Two findings on the way.
  The first version put every type's branch in the one row module and
  Q8_0 decode fell from 105 to 65 tok/s with no change to any executed
  Q8_0 instruction, the register demand of the whole module setting the
  occupancy of every path; the row kernel is now one module per family
  of types, F32 and Q8_0, Q4_0 and Q4_1, Q6_K, built from one source with
  a define, and Q8_0 is back at 110 to 112 against the parent commit's
  110 to 112 in an interleaved run. The second: byte loads with scalar
  activation loads gave Q4_0 17 GB/s, Q4_1 35 and Q6_K 6 to 17 at the 8B
  shapes, and the Q6_K head of the fixture, 151,936 rows, took 22 ms of a
  38 ms token. The word paths read the quants as 32-bit words, Q4_0 as a
  block per lane over the 9-word pair with the first block's nibble
  words assembled from two loads, Q4_1 as one lane per 5-word block,
  Q6_K as sixteen lanes per block each holding three words of quants,
  two of sub-scales and the scale, with every other block's words
  assembled from two loads since 210 bytes is not a multiple of four;
  and the activations as aligned 16-byte vectors, one load for four.
  Now 149, 188 and 170 GB/s at the 8B shapes and the head at 102 GB/s,
  1.25 ms. Q4_0 decode 26 to 107 tok/s; llama.cpp does 209 on the same
  file.
  Ninth, the rest of sub-step 6: Q4_K and Q5_K in `embed`, the tile
  kernel and the row kernel, checked against the CPU the same way
  (723,796 outputs across the seven types). The row kernel is now five
  modules from one source: F32 and Q8_0, Q4_0 and Q4_1, Q4_K, Q5_K,
  Q6_K. Q4_K and Q5_K beside Q6_K cost Q6_K 40 percent (170 to 99 GB/s)
  the way every type beside Q8_0 had cost Q8_0, and Q4_K alone gained 30
  percent over Q4_K beside Q5_K (104 to 135 GB/s); at the 8B shapes the
  row kernel reads Q4_K at 135 GB/s, Q5_K at 123 and Q6_K at 163. Eight
  lanes share a Q4_K or Q5_K block, each taking the sixteen nibble bytes
  of one half of one 64-value chunk, with the three packed sub-scale
  words read by every lane. `Qwen3-0.6B-Q5_K_M.gguf` (168 Q5_K, 29 Q6_K,
  113 F32; same repo and revision as the Q4_0 file) is a third
  `BASELINE_MODELS` fixture, as the fused Q5_K block below proposed, with
  bounds set from the CPU's measured HF deltas plus margin: top-5 overlap
  4, NLL delta 0.05 continuous (measured 0.026) and 0.16 per chunk
  (measured 0.130, 0.068, 0.024). On the device its logits and all four
  perplexity cases match the CPU's numbers to the digit (PPL 29.5612
  continuous, 38.6676 at c=123).

  A confirmation run of the reference at the end of the session, asked
  for by the user, moved with the machine: llama.cpp 0.6B Q8_0 measured
  602 prefill and 182 decode against 660 and 198 three hours earlier,
  and llmx in the same minutes 1000 to 1157 and 102.7 to 106.5 against
  1195 and 111; Q4_0 the same, 624/209 against 636/209 and llmx 878 to
  1056 and 95.7 to 97.4 against 1110 and 107. Both arms fell together, so
  the shares hold and neither run is a regression; the table keeps the
  earlier pairs and the session-end pairs side by side.

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q5_K_M | prefill | 492 tok/s | 640 (533, 638, 645) | 130% |
  | Qwen3-0.6B-Q5_K_M | decode | 205 tok/s | 87 (81, 88, 87) | 42% |
  | Qwen3-0.6B-Q8_0, session end | prefill | 602 tok/s | 1000 to 1157 | 166 to 192% |
  | Qwen3-0.6B-Q8_0, session end | decode | 182 tok/s | 102.7 to 106.5 | 56 to 59% |
  | Qwen3-0.6B-Q4_0, session end | prefill | 624 tok/s | 878 to 1056 | 141 to 169% |
  | Qwen3-0.6B-Q4_0, session end | decode | 209 tok/s | 95.7 to 97.4 | 46% |
  Tenth, two decode wins found by timing the host against the device.
  Temporary instrumentation put a 0.6B decode token at 0.5 ms of host
  recording and 7.4 ms of device time over 396 dispatches, so the
  device is the story, and the row kernel's final reduction turned out
  to run its xor-shuffle chain over all eight column slots whether one
  column was live or eight: 48 shuffles per lane after five loads.
  Reducing only the live columns took the 1024-square Q8_0 matvec from
  17.1 to 13.2 us, 1024x3072 from 33 to 22 us and the 8B shapes from 313
  to 373 GB/s, and 0.6B decode from 122 to 145 tok/s over 128 tokens.
  Then the pass is submitted in chunks of 64 dispatches as it is
  recorded, so the device starts while the host records the rest: 145
  to 158 tok/s; chunks of 16, 32, 64, 128 and 256 gave 150, 156, 158,
  152 and 149, and 8B prefill is unchanged by it (203 to 208 either
  way). Both changes leave every kernel output identical and the greedy
  text the same. Three interleaved rounds of the four files afterwards:

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | prefill | 660 tok/s | 1029, 1105, 1109 | 156 to 168% |
  | Qwen3-0.6B-Q8_0 | decode | 198 tok/s | 137.9, 137.1, 136.5 | 69% |
  | Qwen3-0.6B-Q4_0 | prefill | 636 tok/s | 934, 962, 1040 | 147 to 164% |
  | Qwen3-0.6B-Q4_0 | decode | 209 tok/s | 128.2, 129.4, 127.3 | 61% |
  | Qwen3-0.6B-Q5_K_M | prefill | 492 tok/s | 681, 736, 742 | 138 to 151% |
  | Qwen3-0.6B-Q5_K_M | decode | 205 tok/s | 122.1, 110.5, 118.4 | 54 to 60% |
  | Qwen3-8B-Q8_0 | prefill | 99 tok/s | 208, 208, 209 | 210% |
  | Qwen3-8B-Q8_0 | decode | 39.7 tok/s | 35.5, 35.5, 35.7 | 89% |

  Decode over 128 tokens reads higher than over 32 (158 against 138 on
  0.6B Q8_0) because the first tokens carry the clock ramp; llama-bench
  warms up before its 32, so the 32-token llmx figure is the
  conservative one and the table keeps it.
  Eleventh, the residual add folded into the matmul's store. The model
  asks for `matmul_add`, Y += W X, and each backend produces it its own
  way: the CPU computes into scratch and adds, the same arithmetic as
  before to the bit, the device sets a flag in the row and tile kernels'
  push constants and the store becomes an accumulate. Two dispatches
  fewer per layer; 0.6B Q8_0 decode 137 to 140-146 tok/s, 8B 35.5 to
  35.8, checked against the CPU at batch widths 1, 3 and 64.

  Twelfth, and the one that changes the reading of every table above:
  the user asked whether the measurement was right, and it was not
  matched. The reference's bench tool warms up, then averages three
  repeats of prompt processing into an empty history and of generating
  32 tokens from an empty history, model time only. `generate --verbose`
  gave one cold run whose decode came after the 247-token prompt, so
  with attention over 250 to 280 tokens every step, with sampling and
  text output inside the timer and the clock ramp in the first tokens.
  `llmx bench --model` now runs the reference's protocol (docs/USAGE.md),
  and the same binary that read 138 tok/s under `generate` reads 202
  under it. Both arms in the same minutes, same card and files:

  | model | phase | reference b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp247 | 651.2 +- 4.3 tok/s | 1507.6 +- 5.1 | 232% |
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 +- 0.6 tok/s | 201.9 +- 0.2 | 103% |
  | Qwen3-0.6B-Q4_0 | pp247 | 669.7 +- 3.1 tok/s | 1367.1 +- 5.0 | 204% |
  | Qwen3-0.6B-Q4_0 | tg32 | 220.8 +- 0.9 tok/s | 194.8 +- 1.5 | 88% |
  | Qwen3-0.6B-Q5_K_M | pp247 | 518.1 +- 1.6 tok/s | 875.1 +- 3.5 | 169% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 +- 1.0 tok/s | 169.4 +- 0.7 | 77% |
  | Qwen3-8B-Q8_0 | pp247 | 97.0 +- 0.5 tok/s | 232.5 +- 0.4 | 240% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 +- 0.1 tok/s | 40.6 +- 0.1 | 105% |

  Q8_0 decode is at the floor on both models under the matched protocol.
  The earlier tables stand as what `generate --verbose` measured, both
  arms' figures at the time; they are not the floor comparison.

  Thirteenth, the attention inputs as one op. `norm_rope_kv` is what the
  model asks for at every layer: q normed and rotated in place, k normed
  and rotated and written with v into the view's KV blocks. The base
  class default is the three ops it replaces, which is what the CPU
  runs, so its arithmetic is unchanged; the device runs one kernel, a
  workgroup per (row, head) over the q, k and v heads, the k heads
  written straight into their blocks, three dispatches fewer per layer.
  Checked against the CPU's three ops with a 70-token history, q
  directly at 1e-5 and k and v through attention at 1e-4. Under the
  matched protocol, same session as the table above:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 tok/s | 201.9 | 221.3 +- 0.8 | 113% |
  | Qwen3-0.6B-Q4_0 | tg32 | 220.8 tok/s | 194.8 | 205.4 +- 0.4 | 93% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 tok/s | 169.4 | 176.7 +- 0.4 | 80% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 tok/s | 40.6 | 41.0 +- 0.0 | 106% |

  Prefill moved within noise (1494, 1352, 873 and 230 tok/s).

  Fourteenth, the norm folded into the matmul, tried three ways and
  rejected, not committed. The op was `matmul_group_normed`, the RMS
  norm of X against w and the projections in one call, the CPU norming
  into scratch first and the device folding it into the row kernel for
  one column. (1) The column staged in 16 KB of shared memory per
  workgroup, normed there, every read from it: 8B decode 41.0 to 13.4
  tok/s and 0.6B 221 to 101, with the plain modules carrying the array
  too; in separate normed modules, so the plain ones had none, 8B 22.3
  and 0.6B 165. Sixteen KB of shared memory per workgroup caps what a
  compute unit holds. (2) No staging, each read scaled by the row's
  factor and w on the way in: 8B 12.5, 0.6B 137; with the reduction
  removed and the scale alone left, 8B 11.7, so the per-read scale and
  its second load are what the tuned Q8_0 load pattern cannot absorb.
  (3) A 4 KB staging for rows up to 1024 wide: 0.6B Q8_0 203, Q4_0 199,
  Q5_K_M 176 against 221, 205 and 177 unfused. Every variant also
  moved the prefill's norm into a backend scratch buffer, and that
  alone cost 8B prefill 232 to 166 tok/s; the buffer was device-local
  and allocated once, and the cause was not found before the whole
  change was reverted, so a backend-allocated activation buffer is a
  thing to measure before using again. The two norm dispatches per
  layer stay: on 0.6B they are about a tenth of a decode token and no
  fusion tried gets them back.

  Fifteenth, the greedy-output hash the user asked for: `generate` with
  the 247-token excerpt, 128 greedy tokens, on the CPU and on the
  device, output hashed. Qwen3-0.6B Q8_0, Q4_0 and Q5_K_M give the same
  hash on both backends (e41aee746cc31abf, c19367d54fb40354,
  b18a64cda0208547), so the device's reduction-order differences never
  flipped an argmax over 128 steps on any of the three quantizations,
  and Qwen3-8B-Q8_0 gives the same hash on both over 64 tokens
  (379c733a478ec58a). The check is CPU against device on the same file:
  a greedy sequence from a quantized file cannot be checked against the
  fp32 reference beyond its first tokens, since quantization legitimately
  moves the argmax, so HF stays the ranking and NLL gate.

  Sixteenth, the long-context case the user asked for, `bench --model`
  with 16384 prompt tokens and 512 generated, two repeats after a
  warm-up, both arms in the same minutes:

  | model | test | reference b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 155.2 +- 3.7 | 30% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 192.5 +- 0.2 | 109% |
  | Qwen3-8B-Q8_0 | pp16384 | 97.3 +- 0.5 tok/s | bad allocation | - |
  | Qwen3-8B-Q8_0 | tg512 | 37.6 +- 0.1 tok/s | bad allocation | - |

  Two findings. Decode over a 512-token history still leads, but a
  16384-token prompt runs at 30 percent of the reference: the device
  attention kernel is one workgroup per (query row, head) with an
  online softmax, which is the flash form for decode, but prefill has
  no query tiling, so every query row re-reads its whole K/V history
  and the traffic is quadratic. At 247 tokens that is invisible behind
  the matmuls; at 16384 it is the prompt. And Qwen3-8B does not fit a
  16k context on the 16 GB card: the KV cache is F32, 16896 tokens of it
  are about 5 GB on top of 8.7 GB of weights, and the allocation
  failed, where the reference's f16 KV fits.
  Seventeenth, the tiled prefill attention. A wide pass of 128-wide
  heads takes `attention_tile`: a workgroup per 32 query rows and head,
  the head's K and V streamed through shared memory in tiles of 16
  tokens, so a tile is read once per 32 rows instead of once per row;
  eight lanes share a row, each holding 16 elements of it, a score is
  three xor shuffles, the softmax is online per row as before. Rows
  past the causal limit are masked inside the tile that holds them and
  later tiles are not read. Checked against the CPU at 32 and 45 rows
  after 0 and 70 tokens at 1e-4. Same session as the table above, the
  8B case at 8k because 16k does not fit its f32 cache:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 155.2 | 513.3 +- 0.2 | 100% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 192.5 | 191.9 +- 0.5 | 108% |
  | Qwen3-8B-Q8_0 | pp8192 | 115.7 +- 0.2 tok/s | - | 185.9 +- 1.3 | 161% |
  | Qwen3-8B-Q8_0 | tg512 | 37.3 +- 0.1 tok/s | - | 39.2 +- 0.0 | 105% |
  | Qwen3-0.6B-Q8_0 | pp247 | 656.0 +- 2.1 tok/s | 1507.6 | 1610.5 +- 4.1 | 246% |
  | Qwen3-0.6B-Q8_0 | tg32 | 198.3 +- 0.3 tok/s | 221.3 | 226.9 +- 0.4 | 114% |

  The 16k prompt went from 30 percent of the reference to level with
  it, and the kernel does not yet share a K/V tile across the query
  heads of a KV group, which is the next factor available there.
  Eighteenth, the f16 cache sides. `--cache-type-k` and `--cache-type-v`
  take `f32` or `f16` for `generate`, `chat`, `logits`, `perplexity` and
  `bench --model`, the same on the CPU and the device; the type reaches
  each backend through `kv_alloc`, the CPU converts with F16C on write
  and read, the device builds four variants of every cache kernel and
  the storage picks one, and the device writes halves with an explicit
  round-to-nearest-even in the bits because `packHalf2x16` leaves the
  rounding to the driver and the first build differed from the CPU by
  an f16 ulp. The two backends now hold identical cache bytes; checked
  for every combination of K and V types through `kv_write`,
  `norm_rope_kv`, `kv_copy` and both attention kernels, device against
  CPU at 1e-4 and both against the f32 cache at 2e-2. The HF gate with
  both sides f16 (`run_tests.py --cache-type f16`) passes on both
  backends with the same numbers to three places: Q8_0 NLL delta
  0.001254 against 0.001374 with f32 caches, Q4_0 0.131584, Q5_K_M
  0.026174; the synthetic F32 and shard gates, which compare exact f32
  arithmetic, skip under f16 caches by design. Halving the cache lets
  Qwen3-8B run the 16k context that failed to allocate with f32:

  | model | test | reference b11075 Vulkan | llmx f32 cache | llmx f16 cache | llmx share (f16) |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q8_0 | pp16384 | 97.3 +- 0.5 tok/s | bad allocation | 141.7 +- 0.3 | 146% |
  | Qwen3-8B-Q8_0 | tg512 | 37.6 +- 0.1 tok/s | bad allocation | 38.7 +- 0.0 | 103% |
  | Qwen3-8B-Q8_0 | pp8192 | 115.7 +- 0.2 tok/s | 185.9 +- 1.3 | 189.9 +- 2.7 | 164% |
  | Qwen3-8B-Q8_0 | tg512 (8k) | 37.3 +- 0.1 tok/s | 39.2 +- 0.0 | 39.5 +- 0.0 | 106% |
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 513.3 +- 0.2 | 512.9 +- 0.7 | 100% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 191.9 +- 0.5 | 203.6 +- 0.1 | 115% |
  | Qwen3-0.6B-Q8_0 | pp247 | 656.0 +- 2.1 tok/s | 1610.5 +- 4.1 | 1620.9 +- 3.7 | 247% |
  | Qwen3-0.6B-Q8_0 | tg32 | 198.3 +- 0.3 tok/s | 226.9 +- 0.4 | 225.1 +- 0.5 | 113% |

  The default is f16 on both sides, taken as a decision on 2026-09-22
  after these measurements: it is what the reference runtime stores by
  default, so a comparison is like-for-like without flags; the HF gate
  passes with it on both backends, the 8B Q8_0 excerpt reading a
  perplexity delta of 0.001254 against f32's 0.001374 inside a 0.01
  bound; a decode step reads the whole cache, so a 512-token generation
  on Qwen3-0.6B gains 6 percent; and the 8B runs a 16k context that f32
  cannot allocate on a 16 GB card. `f32` on both sides stores the cache
  exactly and is one flag away. The three components that compare exact
  f32 arithmetic against independent fixtures (`f32`, `shards`,
  `server`) now ask for f32 sides themselves, so they test what they
  tested before; `--cache-type` overrides them, and under `f16` they
  skip as they did. All three suites pass at the new default: the CPU
  suite, the CPU suite with `--cache-type f32`, and the device suite,
  with the Q5_K_M perplexity delta reading 0.067623, 0.067633 and
  0.067753 across them.
  Nineteenth, mixed groups partitioned by type. A Q5_K_M layer's q and k
  are Q5_K and its v is Q6_K, and a group of mixed types fell back to
  one dispatch per projection; it is now one dispatch per type, two for
  that group instead of three. Q5_K_M tg32 176.7 to 185.7 +- 0.5 tok/s
  (84 percent of the reference's 219.8); Q4_0, whose groups are pure,
  read 211.2 +- 0.3 in the same minutes against 205.4 earlier, which is
  the session's drift, not the change. After the reduction fix the row
  kernel's per-type readings at the 8B shapes are Q8_0 415 GB/s, Q6_K
  215, Q5_K 179, Q4_0 178, Q4_K 167, and the Q6_K head of the 0.6B
  files, 151,936 rows of 1024, takes 693 us at 184 GB/s, 7 percent of a
  Q5_K_M token; at 1024 wide every row's lanes re-read the whole
  activation row, 620 MB of cache traffic against 127 MB of weights.
  Twentieth, two rows per lane cluster in the Q6_K module, tried and
  rejected, not committed: each lane decoded its sixteenth of two rows'
  blocks and applied both to the activation vectors it had loaded once,
  halving the activation re-read. The second row's quants and scales
  doubled the live registers and the occupancy lost outweighed the
  traffic saved: the head went from 693 to 1184 us, the 8B shape from
  215 to 129 GB/s, and Q5_K_M tg32 from 185.7 to 161.1 tok/s. The
  activation re-read is the traffic, but the answer is not more state
  per lane. Twenty-first, the other way round, a 4 KB shared-memory
  copy of the activation row per workgroup in the Q6_K module for
  one-column passes up to 1024 wide, every lane reading from it: also
  rejected, the head 693 to 771 us, Q5_K_M tg32 185.7 to 180.1, and
  even the 4096-wide shape, which was not staged, fell from 215 to 175
  GB/s from the select on every read. The K-quant row kernels at these
  shapes are not limited by activation traffic in a way either fix can
  reach; what remains there is the per-dispatch floor and the decode
  work per lane, and it is left at 84 percent for now.

  Twenty-second, the block-size screening VULKAN.md sub-step 7 asked
  for, on the device, same session, `bench --model --p 247 --n 512`,
  two repeats: 32, 64 and 128 tokens per block give 201.4, 200.6 and
  201.9 tok/s on 0.6B and 40.9, 41.0 and 41.0 on 8B, all within a
  repeat's spread, every kernel check passing at each size. 64 stays.
  Twenty-third, integer activations for the decode row kernel
  (VULKAN.md sub-step 8). The 4- and 5-bit files sat at 88 and 77
  percent of the reference's decode and the row kernel's per-type
  readings said why: Q4_0 moved 182 GB/s of weights where Q8_0 moved
  413, the same weights per second, so the 4-bit paths were bound by
  what they did per weight, eight nibble extractions, conversions and
  multiply-adds per word, and by the float activation bytes every path
  re-read per row. The device accelerates packed 8-bit and 16-bit
  integer dots, so three probes at the 8B shape, wrong numerics and the
  kernel alone, put a number on each form:

  | shape | float | int8, GGUF word order | int8, aligned words | int16 |
  |---|---:|---:|---:|---:|
  | Q8_0 4096 x 12288 | 130 us | 147 | 118 | 112 |
  | Q8_0 12288 x 4096 | 152 us | 154 | 120 | 109 |
  | Q4_0 4096 x 12288 | 155 us | 95 to 102 | - | 105 to 110 |

  Q8_0 in the GGUF word order was slower than float: its first block's
  values sit two bytes into each word, and the per-word handling of that
  cost more than the dots saved; aligned, the int path won. 16-bit
  activations took the design over 8-bit on two counts: the CPU
  experiment in ASSETS.md had put per-block 8-bit activations at 0.0093
  of NLL against the 0.010 bound on the 8B excerpt and 16-bit at
  0.00003, and 16-bit dots were the faster of the two on Q8_0 besides.
  The implementation (`shaders/xquant.glsl`, `matmul_row.comp`): every
  quantized row meets the activations as signed 16-bit values in blocks
  of 32, the block's largest magnitude at 32767, with the scale and the
  scaled sums, whole and per half of 16, in a table; a weight word's
  values pair off with activation words in the order nibble and byte
  words unpack in, a block's integer sum is scaled once, and a type's
  offset is folded through the block sum, from one lane of each group
  that reduces together where lanes share a group. The twin is written
  by whichever kernel produces the input, the norm, the SiLU and the
  per-row attention or its merge, tagged for the next row matmul on that
  buffer, so a decode token quantizes nothing in a dispatch of its own;
  an input without a producer takes one. The CPU backend and the tile
  kernel keep float activations. Three findings on the way, each
  measured on the 8B shape: the standalone quantize dispatch costs 5 to
  9 us, which is why the producers write the twin; the same dots over
  8-byte activation loads ran Q4_0 at 150 us against 105 with 16-byte
  loads, so the load count and not the arithmetic bounds these paths;
  and for Q8_0 the per-word handling of the misaligned first block cost
  a quarter of the kernel (selects 154 us, aligned form 125, the narrow
  path beside the wide one in the module another third), so the wide
  path shifts its two first-block words by a half word with the word
  after, fetched from the next lane of the pair by a shuffle, and lives
  in a module of its own. Kernel readings, float to integer, the
  integer figure including the standalone quantize dispatch the test's
  matmul takes:

  | shape | float | integer |
  |---|---:|---:|
  | Q8_0 4096 x 12288 | 130 us, 413 GB/s | 134 to 136 us, 392 to 399 GB/s |
  | Q8_0 12288 x 4096 | 152 us, 350 GB/s | 133 us, 400 GB/s |
  | Q4_0 4096 x 12288 | 155 us, 182 GB/s | 107 to 110 us, 257 to 264 GB/s |
  | Q4_1 12288 x 4096 | 167 us, 189 GB/s | 98 us, 320 GB/s |
  | Q4_K 4096 x 12288 | 163 us, 173 GB/s | 118 to 121 us, 234 to 240 GB/s |
  | Q5_K 4096 x 12288 | 219 us, 158 GB/s | 160 to 166 us, 208 to 216 GB/s |
  | Q6_K 4096 x 12288 | 247 us, 167 GB/s | 223 to 226 us, 183 to 186 GB/s |
  | Q6_K 1024 x 151936 | 825 us, 155 GB/s | 780 to 818 us, 156 to 164 GB/s |

  Correctness: `backend-vulkan` feeds the CPU reference the activations
  quantized the same way, so the comparison is about the dots and the
  reduction order at 1e-4 over every type and both block-count
  parities, and checks a norm, a SiLU and an attention into a buffer
  followed by a matmul from it; the whole CTest and device Python
  suites pass. The HF gate on the device, the tile path against the row
  kernel forced with `--ubatch 8` and `--ubatch 1`, mean NLL over the
  247-token excerpt, HF 3.360286:

  | model | CPU float | device tile | device row kernel | bound |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 3.361660 | 3.361670 | 3.361670 | 0.010 |
  | Qwen3-0.6B-Q4_0 | 3.491840 | 3.491820 | 3.491820 | 0.160 |
  | Qwen3-0.6B-Q5_K_M | 3.386460 | 3.386460 | 3.386460 | 0.050 |

  The integer row kernel gives the tile path's NLL to six decimals. A
  bug surfaced on the first model run and is fixed: a stream-ordered
  scratch outgrown mid-pass, the twin's and the attention split
  states', was freed while the open command buffer still named it, and
  the device hung; `grow` now retires the old buffer with the ring slot.
  The floor under the matched protocol, three arms in the same minutes,
  the before arm built from a detached worktree at 37a020f, two rounds:

  | model | test | reference b11075 Vulkan | llmx before | llmx after | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 197.7, 197.9 tok/s | 217.0, 225.4 | 213.7, 206.3 | 104 to 108% |
  | Qwen3-0.6B-Q4_0 | tg32 | 226.4, 226.5 tok/s | 202.6, 197.1 | 206.5, 206.0 | 91% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 223.7, 223.5 tok/s | 179.6, 185.3 | 176.7, 179.9 | 79 to 80% |
  | Qwen3-8B-Q8_0 | tg32 | 39.9, 39.7 tok/s | 41.0 | 41.0, 40.7 | 103% |
  | Qwen3-8B-Q4_K_M | pp247 | 78.5, 79.0 tok/s | 155.5 | 155.3, 155.4 | 197% |
  | Qwen3-8B-Q4_K_M | tg32 | 52.3, 52.3 tok/s | 34.8, 35.0 | 41.5, 41.6 | 79% |

  The tradeoff, reported together: the 8B 4-bit file, measured for the
  first time here, gains 19 percent of decode and goes from 67 to 79
  percent of the reference; 8B Q8_0 is unchanged; on the 0.6B files the
  change is within a few percent either way, Q4_0 up, Q8_0 and Q5_K_M
  down, all inside the session's own spread between rounds, because a
  0.6B token is about three hundred dispatches at the per-dispatch
  floor and its 1024-wide matmuls run at 60 to 160 GB/s whatever the
  arithmetic. The gain is where the weights are, and it is kept.
  Twenty-fourth, the Q6_K path. It was the weak type of a Q4_K_M file
  at 186 GB/s against Q4_K's 240, half the weights per second at one
  and a half times the bytes, because sixteen lanes of four positions
  each loaded three funnelled quant words, two sub-scale words, the
  scale, four activation pairs and four table entries for sixteen
  weights, with every other block's words costing two loads since 210
  bytes is not a multiple of four. Now eight lanes of eight positions:
  six quant words as three consecutive pairs of three loads each when
  unaligned, the activations as four 16-byte loads, the four group
  scales and the two lanes' half sums as two 16-byte loads each. Every
  check passes; at the 8B shape 218 to 175 us (189 to 236 GB/s) and the
  0.6B files' 151,936-row head 686 to 578 us, from 825 before the
  integer activations. Three arms in the same minutes, the before arm
  a detached worktree at 524ad46, two rounds:

  | model | test | reference b11075 Vulkan | llmx before | llmx after | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 195.4, 194.9 tok/s | 210.3, 210.1 | 209.8, 209.3 | 107% |
  | Qwen3-0.6B-Q4_0 | tg32 | 223.4, 223.1 tok/s | 211.0, 211.3 | 221.1, 212.4 | 95 to 99% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 220.5, 220.6 tok/s | 177.6, 177.8 | 188.7, 188.6 | 85% |
  | Qwen3-8B-Q4_K_M | tg32 | 51.6, 51.7 tok/s | 41.1, 40.8 | 44.8, 44.8 | 87% |

  Q8_0 has no Q6_K and does not move; the Q4_0 file's head and the
  Q5_K_M and Q4_K_M files' v and down projections are where it lands.
  Twenty-fifth, the Q4_K and Q5_K path. Vector loads first, a lane's
  four quant words and the scale with the three packed sub-scale words
  as 16 bytes each and a group pair's table entries as one, which were
  correct and bought nothing at the 8B shape: unlike Q6_K these paths
  were not load-bound. Timing-only variants then put the cost where it
  was: with the sub-scale decode replaced by constants Q4_K ran 121 to
  110 us and Q5_K 165 to 138, and the min term another 5 percent. Each
  of a block's eight lanes decoded its two groups through byte selects
  on branches that diverge across the lanes. The decode is now
  branch-free with selects over the two words the lane's groups sit in,
  Q4_K 121 to 112 us (232 to 256 GB/s) and Q5_K 165 to 138 (210 to
  250), every check passing. Three arms in the same minutes, the before
  arm a detached worktree at adec4f7, two rounds:

  | model | test | reference b11075 Vulkan | llmx before | llmx after | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 197.1, 196.8 tok/s | 204.3, 204.6 | 211.5, 204.1 | 104 to 107% |
  | Qwen3-0.6B-Q4_0 | tg32 | 224.5, 224.4 tok/s | 221.8, 221.8 | 221.8, 221.1 | 99% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 220.9, 221.7 tok/s | 189.2, 183.9 | 195.9, 196.8 | 89% |
  | Qwen3-8B-Q4_K_M | tg32 | 51.9, 51.9 tok/s | 44.7, 45.1 | 46.5, 47.3 | 90 to 91% |

  Twenty-sixth, the tile kernel's threshold. It took every batch of 16
  rows and up, a number set before the row kernel existed in its
  present form; the server's inter-token p99 at 1 and 4 concurrent
  (13 and 62 ms against medians of 5 and 9) was the pass in which a new
  prompt's chunk joined the decoders and tipped the batch into the tile
  kernel, which costs a 64-row tile whatever its fill. Prompt
  processing at 8 to 256 rows, the tile at its old threshold against
  the row kernel taking every width, three models, three runs each:

  | model | rows | tile kernel | row kernel |
  |---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 8 | (row) 712 tok/s | 727 |
  | | 16 | 333 | 791 |
  | | 32 | 640 | 805 |
  | | 64 | 831 | 830 |
  | | 128 | 1810 | 832 |
  | | 256 | 1478 | 838 |
  | Qwen3-8B-Q8_0 | 8 | (row) 72.4 | 72.7 |
  | | 16 | 52.3 | 74.2 |
  | | 32 | 100.7 | 74.9 |
  | | 64 | 173.1 | 75.2 |
  | | 128 | 224.3 | 75.1 |
  | | 256 | 243.3 | 75.3 |
  | Qwen3-8B-Q4_K_M | 8 | (row) 97.7 | 98.3 |
  | | 16 | 33.7 | 100.9 |
  | | 32 | 63.4 | 102.5 |
  | | 64 | 102.0 | 103.3 |
  | | 128 | 143.0 | 103.6 |
  | | 256 | 156.7 | 103.6 |

  The row kernel's rate is flat in the width, a weight pass per eight
  columns, and the tile's climbs with its fill; the tile loses at 16 on
  every file and wins from about 24 rows on 8B Q8_0 and 64 on the other
  two. The threshold is now 32 rows when every projection of the
  dispatch is F32 or Q8_0 and 64 otherwise, which follows the three
  measurements within a few percent and costs the 0.6B Q8_0 up to a
  fifth at 32 to 63 rows against its own best. With the thresholds in
  place, three runs each: 8B Q8_0 at 16, 32 and 64 rows 74.6, 102.4 and
  174.7 tok/s (52.3, 100.7 and 173.1 before), 8B Q4_K_M 101.4, 103.0 and
  102.6 (33.7, 63.4 and 102.0). The server's figures are in the server
  block. The backend-vulkan test's row-kernel reference follows the two
  thresholds, the Linux build passes CTest with the test skipping on
  llvmpipe, and the device suite passes.
  Twenty-seventh, the cost of a prompt pass in rows, from the server's
  time to first token. A sweep of the bench on 0.6B Q8_0, pass time in
  ms at 31, 32, 48, 63, 64, 65, 96, 128, 129, 192, 193, 256 and 257
  rows: 38, 50, 54, 57, 77, 69, 73, 70, 84, 99, 113, 173, 185. Each new
  64-row tile adds about 13 ms to the pass, which is the tile kernel's
  price per tile across the layers; exactly 64 rows costs 8 ms more
  than 65; and the tile from 193 to 256 rows grows from 0.5 to 1.7 ms
  per added row before resetting at 257. The three are open, and so is
  the tiled attention over a long history with few query tiles: a
  62-row tail over 768 shared tokens is two tiles by sixteen heads,
  32 workgroups walking 830 keys each, where the per-row kernel splits
  such a history across up to 64 workgroups. The scheduler's host side
  is not in this: a pass of 62 prompt rows returns from forward in 29
  ms with the command ring four chunks deep, and the rest is the
  device. Against the reference, back to back, three runs each,
  tok/s:

  | rows | 0.6B Q8_0 reference | llmx | share | 8B Q8_0 reference | llmx | share | 8B Q4_K_M reference | llmx | share |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 32 | 953 | 641 | 67% | 97.5 | 102.4 | 105% | 100.5 | 102.7 | 102% |
  | 64 | 929 | 838 | 90% | 79.9 | 174.1 | 218% | 58.3 | 102.6 | 176% |
  | 128 | 620 | 1831 | 296% | 91.3 | 225.5 | 247% | 66.7 | 143.8 | 216% |
  | 256 | 651 | 1487 | 228% | 100.3 | 244.2 | 244% | 79.1 | 157.2 | 199% |
  | 512 | 1662 | 1795 | 108% | 146.1 | 267.5 | 183% | 111.2 | 176.5 | 159% |

  The prompt gate holds on both 8B files at every width and on 0.6B
  from 128 rows; 0.6B at 32 and 64 rows is where the tile kernel's
  price per tile shows, a pass of 50 ms against the reference's 34.
  Twenty-eighth, the K-quant row kernel by its registers. The backend
  now captures the driver's per-kernel statistics when the device has
  `VK_KHR_pipeline_executable_properties` and the test prints them:
  vector and scalar registers, shared memory and scratch per kernel.
  They put occupancy behind the row paths' ranking: the wide Q8_0 path
  at 41 vector registers runs five waves per SIMD and 394 GB/s at
  4096 x 12288, Q4_K at 75 three waves and 261, Q5_K at 93 two waves
  and 254, Q6_K at 78 three and 238. Folding a block's scales once and
  unpacking Q5_K's fifth bits per nibble word, so nothing unpacked
  stays live across the columns:

  | path | registers before | after |
  |---|---:|---:|
  | Q4_K | 75 | 74 |
  | Q5_K | 93 | 84 |

  Q4_K did not move because the compiler hoists all four activation
  loads whatever the source order. The bandwidth first recorded here
  for Q5_K, 254 to 296 GB/s, was wrong: the 254 came from a tree
  before the register trim rather than from the arm being compared.
  Interleaved against the same tree afterwards, both this change and
  the extension's removal leave Q5_K level at 295 to 306, so the trim
  is kept for the registers and not for a rate. Three layouts measured worse and
  were dropped: sixteen lanes per block (55 and 59 registers, four
  waves, 245 and 272 GB/s, a lane then keeping half the weight bytes
  in flight per load), the next block's nibble words loaded before
  this block's dots (81 and 92 registers, 251 and 257 GB/s, the
  hardware's in-order load counter making a wait for this block's
  loads wait for the prefetch), and both at once (226 and 260). On the
  models the change is flat: 8B Q4_K_M decode 46.95 to 46.99 tok/s
  against the reference's 51.8 (91 percent), 0.6B Q5_K_M 195 to 202
  against 221 (88 to 91 percent), where the 0.6B shapes are bound by
  dispatch latency rather than bandwidth (Q5_K at 1024 x 3072 reads
  128 GB/s). The lever that remained for Q4_K was below the
  source, and the same extension served it: with `diagnostics` the
  backend captures the driver's disassembly of every kernel and
  `backend-vulkan --isa DIR` writes one file per kernel. Reading the
  wide Q8_0 and Q4_K listings showed the integer dot product extension
  buying nothing: its 16-bit dot lowered to the same multiply-add pairs
  a plain expression gives, with the operands sign-extended first.
  Written as multiplies of sign-extended halves and bytes instead,
  interleaved against the extension, two passes each:

  | path | extension | multiplies |
  |---|---:|---:|
  | Q4_K 4096 x 12288 | 261, 267 GB/s | 271, 282 |
  | Q6_K 1024 x 151936 | 223, 224 | 232, 233 |
  | Q8_0 4096 x 12288 | 393, 398 | 397, 398 |
  | Q5_K 4096 x 12288 | 305, 295 | 299, 297 |
  | 8B Q4_K_M decode | 47.1, 47.4 tok/s | 48.5, 48.6 |
  | 8B Q8_0 decode | 40.6, 40.7 | 40.9, 41.0 |

  Q4_0, Q4_1 and the 0.6B files are level, the last because those
  shapes are bound by dispatch latency rather than bandwidth. 8B
  Q4_K_M decode is now 92 percent of the reference's 52.3 tok/s and 8B
  Q8_0 103 percent of its 39.7. The second consequence is the larger
  one: no shader uses the extension, so the backend no longer requires
  `VK_KHR_shader_integer_dot_product` of a device, which is one fewer
  refusal between llmx and a card that lacks it.
  Twenty-ninth, what a kernel boundary costs and how many a decode pass
  has. The same tiny dispatch, timed four thousand times on the Radeon
  VII, under four barrier forms:

  | between dispatches | us each |
  |---|---:|
  | the barrier in the tree | 4.27 |
  | compute stages and access bits only | 4.18 |
  | execution dependency, no memory barrier | 3.05 |
  | nothing, incorrect and for the measurement only | 0.49 |

  A boundary costs about 3.8 us, of which 2.6 is the execution
  dependency itself, the device draining and relaunching, and 1.2 is
  the cache maintenance a shader write to shader read requires.
  Narrowing the barrier to the compute stages, which is all a dispatch
  needs, saves 0.1 us and does not pay for a second barrier flavour.
  Counting the dispatches of a decode pass on Qwen3-0.6B-Q8_0 gives
  256: nine per layer over 28 layers, being one norm-rope-and-cache
  write, four matmuls since q, k and v share a dispatch and gate and up
  share another, two RMS norms, one SiLU and one attention, plus the
  embedding, a gather, the final norm and the head. The residual adds
  are not among them; the matmul accumulates them already. At 3.8 us
  each that is 0.97 ms of a 4.88 ms token, a fifth of decode, and the
  nine are a strict chain, so none of it is idle time other work could
  fill. The way out is fewer boundaries rather than cheaper ones, and
  the two norms are the candidates, worth 0.21 ms if folded into the
  matmul that follows. What makes that awkward is the integer
  activations: a norm writes the 16-bit twin its consumer reads, so a
  matmul that normalised on the fly would have to quantise on the fly
  too, per 32-value block, in every workgroup that reads the row. Not
  attempted; the measurement is recorded so the next attempt knows what
  it is buying.

  Thirtieth, the row kernels by what their multiply and their registers
  cost, which closed the two decode cells that were short. The driver's
  disassembly showed one `v_mad_u64_u32` per product of a quant and an
  activation, a 32-bit integer multiply this chip runs at a quarter
  rate: 32 of the Q4_K kernel's 249 vector instructions and a third of
  its issue slots. It will not narrow that to the full-rate 24-bit form
  however the operands are written, and removing the weight side's dead
  sign extension changed the instruction count not at all. Every
  product on the nibble and K-quant paths is a non-negative quant of at
  most six bits against a 16-bit activation and no accumulator reaches
  the 16,777,216 a float counts exactly, so those dots multiply as
  floats and return the same integer at full rate; Q8_0's weights are
  signed and its blocks sum past that range, so it keeps the integer
  multiply. Worth 3.4 percent of 8B Q4_K_M decode with Q8_0 and every
  prefill cell flat.

  That a third of the issue slots bought 3 percent says these kernels
  are not issue bound, and at 390 GB/s of a thousand they are not
  bandwidth bound either. They are latency bound, and the register file
  is what limits the latency the chip can hide: three waves per SIMD on
  the K-quant kernels against five on the wide Q8_0 one. Eight of those
  registers are the batch columns a lane keeps, and a single-sequence
  decode uses one, so the column count is now specialization constant 0
  and each row kernel is built twice. The narrow build fits a fourth
  wave on Q4_0/Q4_1 (69 to 57 registers), Q4_K (73 to 63) and Q6_K (72
  to 62), and Q5_K misses it at 69. The wide Q8_0 kernel is excluded:
  its eight-column build is the one that is not register starved, the
  narrow build takes it from five waves to eight, and 8B Q8_0 decode
  fell 9 percent by it with its matmul going 338 to 367 ms of device
  time. The same kernel on the 0.6B files gained 12 percent, one work
  unit per lane there against four, so the direction follows the shape
  as well as the path and the larger model decides it.

  Both arms built from detached worktrees at their own commits, two
  interleaved blocks per cell, under the matched protocol. The
  reference column is the b11075 Vulkan figure recorded earlier on this
  card; that binary is not on this machine and was not re-run today:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | tg32 | 51.9 tok/s | 48.64 | 58.16 | 112% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 tok/s | 205.43 | 227.00 | 103% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 tok/s | 41.62 | 41.40 | 107% |
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 tok/s | 213.97 | 214.21 | 110% |

  Prefill is flat on all four, within 0.4 percent. Both decode cells
  that were below the reference on this card now clear it. The whole
  suite passes on the device backend, the HF baseline included; one
  perf-floor run failed with the bench process exiting non-zero and did
  not reproduce in two further runs.

  The MI50 in the container, where the reference was re-run on the same
  card in the same session. That card takes the integer dot form, so
  none of the float multiply reaches it and all of the gain is the
  register change:

  | model | test | reference Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | tg32 | 36.75 tok/s | 65.6, 65.0 | 79.8, 79.8 | 217% |
  | Qwen3-8B-Q8_0 | tg32 | 25.65 | 45.7, 46.8 | 49.6, 45.6 | 182% |
  | Qwen3-0.6B-Q8_0 | tg32 | 100.59 | 270.1, 270.5 | 269.9, 270.4 | 269% |

  Prefill is flat there too, on both models measured as the control.

  **Correction (2026-09-22, thirty-fourth paragraph):** the reference column above was measured with all ten MI50s visible, so the reference split the model across ten cards. These are not same-card figures and its shares are void; the one-card comparison is in the thirty-fourth paragraph.
  Whether the float multiply beats the integer dot under Mesa is not
  measured; the two forms still compile from one source and the profile
  still chooses.

  Thirty-first, where prompt processing actually stands, which the
  figures quoted until now understated. The 43, 72 and 77 percent of
  the reference reported through the day are Q8_0 and 0.6B cells. On
  the MI50 at 247 rows against that card's own reference build, in the
  same session as the table above:

  | model | llmx pp247 | reference pp247 | llmx share |
  |---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 2653 tok/s | 3359 | 79% |
  | Qwen3-8B-Q8_0 | 265 | 606 | 44% |
  | Qwen3-8B-Q4_K_M | 98 | 530 | 19% |

  The Q4_K_M file prefills slower in absolute terms than the Q8_0 file
  of the same model on the same card, 98 tok/s against 265, while
  reading a little over half the bytes. So it is not bandwidth and not
  the tile shape: it is the tile kernel dequantizing K-quant weights
  into shared memory on every pass, which the 8-bit path does not pay.
  This is the largest gap open and is the next thing taken.

  **Correction (2026-09-22, thirty-fourth paragraph):** the reference column above was measured with all ten MI50s visible, so the reference split the model across ten cards. These are not same-card figures and its shares are void; the one-card comparison is in the thirty-fourth paragraph. The gap between the Q4_K_M and Q8_0 files is llmx's own and stands.

  Thirty-second, that gap closed. The tile kernel staged K-quant weights
  through the per-value decoders in `qdecode.glsl`, which re-read and
  re-unpack a block's packed sub-scale and sub-min for every value,
  three to five byte loads and the unpacking each time. A thread's run
  of values lies inside one group of 32 whichever tile height is built,
  since it stages 8 or 16 values starting at a multiple of that, so the
  sub-scale, the sub-min, which nibble half the run takes and the byte
  it starts at are invariant across the run and are read once. Then the
  loads: this driver issues one `buffer_load_ubyte` per byte and joins
  none of them, 71 of them in this kernel, so every byte now comes from
  a word. `qdecode.glsl` reaches its bytes through a `QBYTE(i)` macro
  rather than naming an array, so a shader serves them from whatever
  view it binds, and the tile kernel's byte binding went away rather
  than a word binding being added. It issues 86 dword loads and no byte
  loads, against 32 dword, 72 byte and 5 short before. Nothing about
  the arithmetic or the staged values changed, and the HF perplexities
  are identical to the digit.

  | device | model | rows | before | after | reference |
  |---|---|---:|---:|---:|---:|
  | Radeon VII | Qwen3-8B-Q4_K_M | 64 | 95.98 tok/s | 170.75 | - |
  | Radeon VII | Qwen3-8B-Q4_K_M | 247 | 96.51 | 240.31 | - |
  | Radeon VII | Qwen3-8B-Q4_K_M | 512 | 109.21 | 281.73 | - |
  | Radeon VII | Qwen3-0.6B-Q5_K_M | 247 | 1158.02 | 2381.28 | - |
  | Radeon VII | Qwen3-0.6B-Q5_K_M | 512 | 1072.31 | 2565.02 | - |
  | MI50 | Qwen3-8B-Q4_K_M | 64 | 93.0, 94.2 | 157.1, 157.6 | - |
  | MI50 | Qwen3-8B-Q4_K_M | 247 | 98.2, 98.5 | 243.8, 244.4 | 530.4 |
  | MI50 | Qwen3-8B-Q4_K_M | 512 | 113.1, 113.1 | 297.5, 297.7 | - |

  The 8-bit controls are flat on both cards, their staging source
  untouched: on the MI50 Qwen3-8B-Q8_0 reads 265.1 against 264.8 at 247
  rows and 327.0 against 327.1 at 512, and Qwen3-0.6B-Q8_0 is level
  inside its spread. So K-quant prompt processing went from 19 percent
  of the reference on that card to 46 (shares against the ten-card reference, corrected in the thirty-fourth paragraph), which is where the 8-bit path
  already was. Both cards pass every suite afterwards, the HF gate
  included.

  Thirty-third, three questions about prompt processing answered by measurement, each against the explanation I had given.

  Whether the tile wants occupancy. The 128-row tile runs two waves per SIMD, held there by 100 registers and 24,576 bytes of shared memory, and the 64-row tile runs three on 67 and 16,384. Forcing each on the Radeon VII, two interleaved blocks:

  | model | rows | 64-row tile | 128-row tile | shipped choice |
  |---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | 247 | 228.2 tok/s | 240.7 | 243.3 |
  | Qwen3-8B-Q4_K_M | 512 | 261.4 | 284.6 | 284.8 |
  | Qwen3-8B-Q8_0 | 247 | 245.8 | 281.0 | 280.9 |
  | Qwen3-8B-Q8_0 | 512 | 280.6 | 340.2 | 340.4 |

  The taller tile wins everywhere despite the lost wave, so work per thread is worth more than occupancy here, unlike the row kernels, and the shipped height choice is already right.

  Whether the host feeds the device in time. The card read as fully occupied at about three quarters of its power which raised whether prefill is recorded late. With runs short enough that the profiler samples every dispatch, the kernels' own execution times sum to the wall time: 345.7 against 345.2 ms for Qwen3-0.6B-Q8_0 at 512 rows, 3632.6 against 3632.9 for Qwen3-8B-Q4_K_M. A queue waiting on the host would leave time no kernel covers. None is left, so the device never waits and the shortfall is inside the kernels.

  What the reference does instead. On the MI50 its Vulkan build reports `int dot: 1` and `matrix cores: none`, and its quantized prefill path multiplies 8-bit activations through the four-wide integer dot, where ours multiplies dequantized floats one product per instruction on float tiles. That is a quarter of the instructions and a quarter of the shared memory per product. It is the next thing built, for devices whose integer dot is native, which the profile already records for the MI50 under Mesa; the precision of 8-bit activations against the HF bounds decides whether it ships.

  Also measured and not kept: the prefill attention kernel with its online softmax taken four or two keys at a time, one rescale per chunk rather than per key. Flat within one percent on Qwen3-0.6B-Q8_0 at 512 and 4096 rows and Qwen3-8B-Q4_K_M at 2048.

  Thirty-fourth, a correction to every MI50 reference figure above. The reference's Vulkan build uses every device it can see, and the Linux machine has ten MI50s, so `llama-bench -ngl 99` without `GGML_VK_VISIBLE_DEVICES` split the model across all ten. Measured back to back on Qwen3-0.6B-Q8_0, that is pp247 3364 and tg32 101.3 tok/s with ten visible against 6941 and 299.0 on one. The decode leads of 217, 182 and 269 percent, the earlier 2.4 times and the prefill shares were all against the ten-card split. The Radeon VII has one device, so its tables stand, and the per-shape TFLOPS comparison ran pinned to one device and stands. The ROCm arms of the table above are not known to have been pinned either and are unverified. The reference is now pinned to one card.

  The one-card comparison, llmx at `cb2eb5b` with the integer-dot tile against the reference's Vulkan build on the same MI50, interleaved, two passes, three cards in parallel with one model on each:

  | model | test | llmx | reference | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp64 | 1098.9 tok/s | 4654.9 | 24% |
  | Qwen3-0.6B-Q8_0 | pp247 | 4698.6 | 6954.7 | 68% |
  | Qwen3-0.6B-Q8_0 | pp512 | 4871.1 | 6676.8 | 73% |
  | Qwen3-0.6B-Q8_0 | tg32 | 273.0 | 303.1 | 90% |
  | Qwen3-8B-Q4_K_M | pp64 | 249.5 | 260.0 | 96% |
  | Qwen3-8B-Q4_K_M | pp247 | 409.5 | 634.4 | 65% |
  | Qwen3-8B-Q4_K_M | pp512 | 498.0 | 763.4 | 65% |
  | Qwen3-8B-Q4_K_M | tg32 | 78.8 | 86.8 | 91% |
  | Qwen3-8B-Q8_0 | pp64 | 248.8 | 528.3 | 47% |
  | Qwen3-8B-Q8_0 | pp247 | 411.0 | 727.2 | 57% |
  | Qwen3-8B-Q8_0 | pp512 | 499.8 | 863.7 | 58% |
  | Qwen3-8B-Q8_0 | tg32 | 50.0 | 58.4 | 86% |

  Each arm's best pass is shown. So on the MI50 llmx trails in both phases: decode at 86 to 91 percent, prompt processing at 24 to 96. Decode on the MI50 is an open gate again, not a lead.

  Thirty-fifth, the integer-dot tile for every K-quant and Q8_0 loading a word at a time. Q8_0's 34-byte blocks start either on a word boundary or two bytes past one, so its staging loads one extra word and funnels where the block straddles, instead of two 16-bit loads per quant word. Q6_K scales each half of a 32-value group apart, so its own build sums the halves separately, and its offset of 32 goes into each staged byte. Q5_K is Q4_K plus a fifth bit. At the 8B feed-forward shape on one MI50, against the reference's integer-dot tile on the same card:

  | type | float tile | integer-dot tile | reference |
  |---|---:|---:|---:|
  | Q8_0 | 4.87 TFLOPS | 12.07 | 13.30 |
  | Q4_K | 4.65 | 11.44 | 11.42 |
  | Q6_K | 3.62 | 9.60 | 7.03 |

  One module for all four cost Q8_0 and Q4_K 4 percent through Q6_K's split sums, so Q6_K is its own module. Device allocations are now whole words, because a tensor with an odd block count ended two bytes into a word its 32-bit view could not reach.

  The one-card gate at `db249b8`, same protocol as the thirty-fourth paragraph, now with the Q5_K_M file:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q5_K_M | 2123 vs 2931 tok/s, 72% | 5225 vs 4433, 118% | 5676 vs 5723, 99% | 328.9 vs 312.4, 105% |
  | Qwen3-0.6B-Q8_0 | 1097 vs 4654, 24% | 5474 vs 6949, 79% | 5930 vs 6652, 89% | 272.4 vs 300.4, 91% |
  | Qwen3-8B-Q4_K_M | 283.4 vs 260.5, 109% | 557.3 vs 634.3, 88% | 666.7 vs 761.3, 88% | 78.6 vs 86.6, 91% |
  | Qwen3-8B-Q8_0 | 299.7 vs 528.3, 57% | 584.1 vs 733.8, 80% | 721.6 vs 863.4, 84% | 50.0 vs 58.4, 86% |

  These ran while the HF suite and a perplexity job used two other cards of the same machine, so the host was shared; both arms were interleaved on each card, and one llmx decode pass on Qwen3-0.6B-Q8_0 read 243.8 against 272.4 in the other and is reported rather than dropped. Every HF perplexity cell passes in both scoring modes on the MI50. On the 8B Q4_K_M file, 40 wikitext windows of 512 score mean NLL 2.4695 against the float tile's 2.47023.

  What is left on the MI50 is short prompts, where the 0.6B Q8_0 file reads a quarter of the reference at 64 rows, and decode at 86 to 91 percent except on the Q5_K_M file.

  Thirty-sixth, the thresholds. The crossover from the per-row kernel to the tile on the MI50 had been measured against the float tile, 96 rows on a narrow 8-bit projection. Forcing each kernel and sweeping against the integer-dot tile, one model per card:

  | model | row kernel wins up to | tile wins from | threshold before |
  |---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 24 rows | 32 | 96 |
  | Qwen3-0.6B-Q5_K_M | 32 | 48 | 64 |
  | Qwen3-8B-Q8_0 | 8 | 16 | 32 |
  | Qwen3-8B-Q4_K_M | 16 | 24 | 64 |

  So the measured profile carries four thresholds per device, 8-bit and other types each split at 4096 wide, and the MI50's row is 16, 32, 24 and 40; a row measured with the integer-dot tile applies only where the device has the integer dot. Q4_0 and Q4_1 went through the integer-dot tile too, so that the K-quant thresholds they share are the ones measured for it: Q4_0 folds its offset of 8 into each byte, and Q4_1 adds its minimum. The other types held within half a percent at the feed-forward shape. The one-card gate at `2b770f6`:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 2052 vs 4928, 42% | 5025 vs 7252, 69% | 5355 vs 6823, 78% | 328 vs 328, 100% |
  | Qwen3-0.6B-Q5_K_M | 2086 vs 2969, 70% | 5199 vs 4434, 117% | 5668 vs 5722, 99% | 334 vs 313, 107% |
  | Qwen3-0.6B-Q8_0 | 2223 vs 4638, 48% | 5452 vs 6950, 78% | 5918 vs 6659, 89% | 270 vs 299, 90% |
  | Qwen3-8B-Q4_K_M | 283 vs 261, 109% | 555 vs 634, 88% | 665 vs 761, 87% | 78 vs 86, 91% |
  | Qwen3-8B-Q8_0 | 300 vs 528, 57% | 585 vs 733, 80% | 721 vs 863, 84% | 50 vs 58, 86% |

  The HF suite ran on another card of the same machine meanwhile, and every cell passes in both modes. One margin narrowed: the Q4_0 fixture's continuous cell, scored in batched passes through 8-bit activations, is at an NLL delta of 0.139 against its 0.160 bound, where the float tile gave 0.131. Short prompts on the 0.6B files are what is left in prompt processing: 64 rows yields a single column tile, and the projections give too few row tiles to fill sixty compute units.

  Thirty-seventh, a third tile height. A 32-row build of each tile kernel, the second variant of the 64-row one, doubles the workgroups of a short prompt without reading a weight more often. Taken whenever the 64-row tile underfills, it gave the 0.6B files 11 to 23 percent at 48 to 64 prompt rows but cost the 8B files up to 5 percent at 96 to 128: profiled there, the 8B's 4096-wide k and v took 72.5 ms on the small tile against 51.5 on the middle one. The small tile does half the arithmetic per barrier, which a 1024-wide projection's 32 inner steps absorb and a 4096-wide one's 128 do not. So a narrow projection takes it whenever the middle tile underfills and a wide one only below half fill. Against the build before it on five MI50 cards, one model each, best of two interleaved passes:

  | model | pp32 | pp48 | pp64 | pp96 | pp128 | pp247 |
  |---|---:|---:|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | +1% | +23% | +20% | +11% | +8% | 0% |
  | Qwen3-0.6B-Q5_K_M | 0% | +15% | +11% | +5% | +3% | 0% |
  | Qwen3-0.6B-Q8_0 | +14% | +16% | +12% | +7% | +5% | 0% |
  | Qwen3-8B-Q4_K_M | +3% | +2% | -1% | -1% | 0% | 0% |
  | Qwen3-8B-Q8_0 | +3% | +2% | -2% | 0% | -1% | 0% |

  The 8B cells at 64 to 128 rows read 0 to +1 percent under the same rule in the run before, so their -1 and -2 are inside run-to-run spread. Qwen3-0.6B-Q8_0 at 64 rows is 2502 tok/s, 54 percent of the reference's 4638.

  Thirty-eighth, two quant blocks per barrier and where decode stands on the MI50. The integer-dot tile staged one 32-value block per step, two barriers per block and each block's loads waited for alone. Two blocks per step, on one MI50 at the 8B feed-forward shape: Q8_0 11.94 to 12.15 TFLOPS, Q4_K 11.35 to 11.63, Q6_K 9.48 to 10.02; the 8B files gain 2.8 to 4.8 percent at 247 and 512 prompt rows, the 0.6B files 4 to 4.5 at 64, and 8B Q8_0 at 64 rows loses 4.3. Four blocks per step lost 10 to 23 percent, the larger shared arrays costing occupancy. A first build of this failed the MI50's Q4_0 check because a local variable in the Q4_0 and Q6_K staging shadowed the new block index; the Radeon VII never takes this path, so its tests cannot catch such a bug, and a change to this kernel is checked on the MI50 before it is committed.

  Decode, matrix-vector at an 8B down projection, 4096 rows over 14336 inputs, on one MI50 against the reference's per-operation benchmark on the same card:

  | type | llmx | reference | llmx share |
  |---|---:|---:|---:|
  | Q8_0 | 162.5 us | 117.9 | 73% |
  | Q4_K | 124.5 | 56.7 | 46% |
  | Q6_K | 170.9 | 126.4 | 74% |

  The reference's decode matvec quantizes activations to 8 bits and multiplies through the four-wide 8-bit dot; ours reads the 16-bit twin through the two-wide 16-bit dot, twice the dot instructions and twice the activation bytes per weight. The float-multiply row build, which the Radeon VII runs, was tried on the MI50 in place of the integer-dot build and is 10 to 24 percent slower there on the 8B shapes, so it is not the answer. The next piece is an 8-bit twin for the integer-dot devices: every producer of decode activations writes it and every row family's integer-dot build reads it, with Q6_K's offset of 32 and Q4_0's of 8 folded into the weight bytes as the tile does. Batched HF scoring through 8-bit activations on the MI50 put the Q8_0 continuous cell at 0.0023 against its 0.010 bound, so the precision is not expected to be what stops it.

  Thirty-ninth, an 8-bit activation twin for decode on the integer-dot devices. The row kernels read activations as a 16-bit twin through the two-wide 16-bit dot. The reference's decode reads 8-bit activations through the four-wide 8-bit dot, and a probe reading 8-bit on the Q4_K and Q5_K row kernel alone took the Q4_K matvec at 4096 x 12288 from 92.4 to 75.8 us. Putting every family on it measured this way on one MI50, 16-bit against 8-bit:

  | matvec | 16-bit twin | 8-bit twin |
  |---|---:|---:|
  | Q4_0, 4096 x 12288 | 135.1 us | 74.7 |
  | Q4_1, 12288 x 4096 | 137.1 | 82.8 |
  | Q4_K, 14336 x 4096 | 123.4 | 92.8 |
  | Q5_K, 4096 x 12288 | 107.0 | 93.2 |
  | Q6_K, 4096 x 12288 | 150.3 | 137.6 |
  | Q8_0, 4096 x 12288 | 153.7 | 169.5 |

  Q8_0 was slower on it in every loop shape tried: its eight-bit build fell from 40 registers to 32 and ran eight waves per SIMD, the over-occupancy the one-column build also caused it, and loading two pairs per iteration to use the registers recovered the large shapes but cost the small ones. So on such a device the producers write both twins, the 8-bit one after the 16-bit one at a 256-byte offset, and each family reads the one it is fastest and precise enough on. The HF gate then decided which. With Q4_0 or Q6_K on the 8-bit twin the Q4_0 fixture, whose only K-quant is its tied Q6_K output head, ranked a different fifth token for "The capital of France is", a top-5 overlap of 3 against its frozen 4. The bound was not moved, so Q4_0, Q4_1, Q6_K and Q8_0 read the 16-bit twin and Q4_K and Q5_K the 8-bit one.

  Writing a twin nobody reads cost the small models: the producers with the 8-bit writer behind a runtime branch took Qwen3-0.6B-Q4_0 decode down 2.6 percent without ever taking it. So the 8-bit writer is a second build of each producer, specialization constant 7, which the backend dispatches only once a matmul that reads the 8-bit twin has run; the one pass before that makes it through the fallback quantizer. Decode on one MI50, best of two interleaved passes, reference pinned to one card:

  | model | before | now | reference | llmx share |
  |---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | 78.3 tok/s | 85.5 | 86.3 | 99% |
  | Qwen3-0.6B-Q5_K_M | 333.5 | 337.4 | 305.7 | 110% |
  | Qwen3-0.6B-Q4_0 | 331.2 | 330.7 | 324.5 | 102% |
  | Qwen3-0.6B-Q8_0 | 273.4 | 270.1 | 299.4 | 90% |
  | Qwen3-8B-Q8_0 | 50.0 | 50.0 | 57.4 | 87% |

  Every suite passes on the MI50 and the Radeon VII, the HF baseline in both scoring modes. On the 8B Q4_K_M file ten wikitext windows of 512 scored one token at a time move from mean NLL 2.69356 to 2.69409. The backend test feeds each family's reference the activations of the twin it reads; against the 8-bit twin one quant can round the other way on the device, whose reciprocal is a few ulps from the host's, worth about the weight times the block's step, so those comparisons take 1e-2 where an indexing error is worth the output itself. The Radeon VII has no native integer dot and is unchanged. The decode gap left on the MI50 is the two Q8_0 files, and Q6_K, which the gate keeps on the 16-bit twin.

  Fortieth, the one-card gate at `300d812`, same protocol as the thirty-fourth paragraph, llmx share of the reference's Vulkan build on the same MI50, best of two interleaved passes:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 2512 vs 4910, 51% | 5084 vs 7250, 70% | 5403 vs 6832, 79% | 328 vs 331, 99% |
  | Qwen3-0.6B-Q5_K_M | 2443 vs 2959, 83% | 5197 vs 4433, 117% | 5643 vs 5725, 99% | 337 vs 314, 107% |
  | Qwen3-0.6B-Q8_0 | 2592 vs 4652, 56% | 5494 vs 6960, 79% | 5964 vs 6667, 89% | 274 vs 303, 90% |
  | Qwen3-8B-Q4_K_M | 288 vs 260, 111% | 574 vs 634, 91% | 692 vs 761, 91% | 86 vs 87, 99% |
  | Qwen3-8B-Q8_0 | 290 vs 526, 55% | 604 vs 732, 83% | 742 vs 862, 86% | 50 vs 59, 85% |

  The largest gaps left are short prompts on every file but the 8B Q4_K_M, and Q8_0 in both phases. The reference's decode matvec loads each activation word once per thread and reuses it across several rows, where a lane of ours serves one row and loads the activations again for every row; two rows per lane cluster is the next piece for Q8_0 decode.

  Tried and not kept: two adjacent rows per cluster in the wide Q8_0 kernel, each activation word loaded once for both rows. On the Radeon VII, interleaved three times, it is 20 to 30 percent slower at every shape, 133 to 167 us at 4096 x 12288: the kernel went from 42 registers to 51 and its workgroups halved, and for this kernel that costs more than the shared loads save.

  Also not kept: the wide Q8_0 kernel on the 8-bit twin with its occupancy capped. It lost to the 16-bit twin at 32 registers and eight waves per SIMD, and an unused 16 KB of shared memory brings it to four waves; on one MI50 at 4096 x 12288 that reads 172 us against 160 uncapped and 151 on the 16-bit twin, twice over. So occupancy is not why Q8_0 loses on the 8-bit twin, and Q8_0 decode stays on the 16-bit one.

  Documentation review at this checkpoint: every Markdown file read against the code, CLI, tests and build. About fifty stale claims corrected across README, AGENTS, STATUS's table and feature blocks, VULKAN, USAGE, ROADMAP, ARCHITECTURE, EXECUTION, KV-CACHE, DEVICE-EXECUTION, SERVER, CI, ASSETS and the per-source pages. The largest were the ten-card MI50 figures presented as current in README and the status table, VULKAN.md saying no shader uses the integer dot, the tile described as two heights and two thresholds, the perplexity scorer described as one token at a time, and ROADMAP and ARCHITECTURE still calling the server and the Vulkan backend planned. The numbered measurement paragraphs above are left as history.

  And a memory fix. A model on a device backend held every weight twice: the loader reads the file into one host allocation, the model kept it for its lifetime, and the device backend copies each weight into its own memory. The model now records at adoption whether any weight still reads those bytes in place, and the CLI releases them when none does. Qwen3-8B-Q4_K_M on the Radeon VII, steady host memory 4.62 to 0.18 GB, decode unchanged; the CPU backend adopts by aliasing and keeps them. The peak is still the whole file, 4.84 GB, since it is read before the upload. Streaming the file to the device during the load, read directly into staging and uploaded asynchronously so the disk and the copies overlap, would remove that peak and is not done.

  Forty-first, short prompts through the integer-dot tile. A profile of 8B Q8_0 at 64 rows put 53 percent of device time in the 64-row tile, and timing single calls on one MI50 at 64 columns showed where. The down projection, 4096 x 12288, took 2970 us, 2.2 TFLOPS, where the gate projection, 12288 x 4096, read 6.2. Splitting the inner dimension across more workgroups, tried first on its own, made most shapes slower: 4096 x 4096 went from 617 to 896 us. So the calls were not short of workgroups but of memory locality.

  The cause was the 8-bit activations' order. They were stored column after column, so each step of the tile read its 64 columns a row width apart. They are now ordered by block of the inner dimension, then by column, which makes a step's activations one 2 KB run. That alone takes the down projection to 1299 us, 4096 x 4096 to 499, and the 512-column feed-forward shape from 9.56 to 11.0 TFLOPS. Perplexity is bit-identical.

  With that order in place, splitting helps. A call with fewer workgroups than eight per compute unit splits its inner dimension into parts of at least 16 quant blocks. Each part writes partial sums to a scratch buffer, and `matmul_reduce.comp` adds them in part order. The target was swept at 2, 4 and 8 per compute unit on all five files, and eight wins or ties everywhere except 0.6B Q8_0 at 512 rows, which loses 3 percent. Together, the down projection goes to 731 us and 4096 x 4096 to 354.

  Split and unsplit outputs agree to 4e-7 relative at every shape probed. The re-quantization to 8 bits downstream turns that reordering into a few rounding flips, so 20 windows of 512 score 13.6049 against 13.5942 on 8B Q8_0, and 40 windows of 64 score 67.310 against 67.474 on 0.6B Q8_0. The HF gate passes every cell both ways, and the backend test passes on the MI50, where its 1024-wide calls now split in two.

  The one-card gate at this change, same protocol as the fortieth paragraph:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 3744 vs 4795, 78% | 5610 vs 6962, 81% | 5590 vs 6695, 83% | 310 vs 308, 101% |
  | Qwen3-0.6B-Q5_K_M | 3553 vs 2917, 122% | 5543 vs 4380, 127% | 5712 vs 5609, 102% | 310 vs 290, 107% |
  | Qwen3-0.6B-Q8_0 | 3381 vs 4600, 73% | 5973 vs 6867, 87% | 6011 vs 6585, 91% | 262 vs 292, 90% |
  | Qwen3-8B-Q4_K_M | 618 vs 261, 237% | 767 vs 630, 122% | 812 vs 755, 107% | 84 vs 84, 101% |
  | Qwen3-8B-Q8_0 | 677 vs 526, 129% | 819 vs 732, 112% | 866 vs 873, 99% | 50 vs 56, 90% |

  Both 8B files now clear the reference in prompt processing, except Q8_0 at 512 rows, which is level. What remains below it is 0.6B Q4_0 and Q8_0 prompt processing and Q8_0 decode. `bench --profile` now also prints the driver's register, shared memory and waves-per-SIMD figures for each kernel that ran.

  Forty-second, the prefill attention tile and batch invariance. At 512 rows the causal attention tile was 28 percent of Qwen3-0.6B-Q8_0's device time on an MI50, about 0.6 TFLOPS. It finished every score in every lane with three shuffles, and took two exponentials and a rescale of the accumulator per token. It now takes a tile of 16 keys as a unit. Scores come eight tokens at a time, and a butterfly over the row's lanes finishes and deals them out one to a lane, 7 shuffles per 8 tokens. Then one maximum, one exponential per score, and one rescale per tile. Lanes own interleaved dimensions, so a row reads a staged token as contiguous 128-byte runs. On one MI50, 0.6B Q8_0 goes from 6029 to 7513 tok/s at 512 rows and from 3267 to 5566 at 2048, and 8B Q8_0 from 668 to 809 at 2048; on the Radeon VII 0.6B Q8_0 gains 3.5 percent at 512 rows and 12 at 2048. Finishing a whole tile's scores at once held 208 registers and one wave per SIMD, and a 32-key tile left one workgroup per compute unit; both were slower.

  Its output is within 1.3e-6 of the CPU's, as the old tile's was, yet the server test failed: a request reusing a cached prefix gave different greedy text from the CLI over the whole prompt. Two causes. The first was the test. Its server was started without the device and cache flags, since the helper that adds them looked at the executable path rather than the command, so every device suite had compared a CPU server with a device CLI. The CLI side also ran the default f16 cache where the server side asked for f32. On this prompt the first token is a near-tie, margin 0.26 on the CPU and 0.15 on the old tile, and the new tile tipped it.

  The second was real: a row's result depended on its batch. The row kernels and the tiles round differently and the kernel was chosen by the call's width, so a prompt's reused-prefix tail took the row kernel where one pass took the tile; the integer-dot tile's split followed the call's workgroups; and the per-row attention split its history by the dispatch's longest row and not at all past 256 (row, head) pairs.
  - A batch entry now carries its extent, the position one past its prompt's last token or 1 for a generated token, and the model passes calls their rows as runs (`backend::RowRun`). A row takes the tile when its extent reaches the tile threshold, and a call that mixes kernels becomes one call per kernel.
  - The tile's split is the one a pass over the row's whole prompt takes, up to 512 rows. Taking it from the shape alone as if every call were one column tile was also invariant, but cost 8B Q8_0 at 512 rows 866 to 802 tok/s.
  - Attention takes the tile by the view's extent, and the per-row kernel splits a row's history in parts of 32 tokens, doubling until at most 64 cover it, from the row's length alone.

  `backend-vulkan` checks this bitwise at 2048 and 6144 outputs over 249 rows, where the whole prompt takes the tallest tile and its tail the shortest. At the model API, one pass, a forked reused prefix and a prefix-then-tail pass give bit-identical logits on the MI50. The server test now runs its server on the device with f32 caches on both sides, and passes on both cards.

  Two things did not hold. Every prompt row through the tile, which would have kept many short prompts on one tile call, failed the HF gate on the short-prompt fixtures (a top-5 overlap of 3 against 4, and the exact F32 logits) and halved a 5-token prompt, 769 to 394 tok/s. The row kernel looping over the call's columns inside one dispatch took 24-row prompts from 1063 to 1202 tok/s but slowed the Q8_0 matvec from 167 to 209 us at 14336 x 4096 with the same registers, so it is not kept. The cost that stays is first-token time for many short prompts arriving together: 16 concurrent requests on 0.6B Q8_0 wait 134 to 138 ms at the median where they waited 49 to 102, since each prompt now takes the row kernel it takes alone rather than joining one tile call. Throughput and inter-token latency are unchanged.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4015 vs 4915, 82% | 6341 vs 7217, 88% | 6855 vs 6821, 101% | 322 vs 328, 98% |
  | Qwen3-0.6B-Q5_K_M | 3853 vs 2976, 129% | 6342 vs 4448, 143% | 7033 vs 5750, 122% | 341 vs 314, 109% |
  | Qwen3-0.6B-Q8_0 | 4006 vs 4651, 86% | 6865 vs 6980, 98% | 7504 vs 6690, 112% | 272 vs 297, 91% |
  | Qwen3-8B-Q4_K_M | 632 vs 261, 242% | 801 vs 633, 126% | 872 vs 763, 114% | 87 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 690 vs 527, 131% | 848 vs 736, 115% | 914 vs 865, 106% | 50 vs 58, 86% |

  Prompt processing now clears the reference on every file at 512 rows and on both 8B files everywhere. Below it: 0.6B Q4_0 and Q8_0 at 64 and 247 rows, decode on the 8-bit files at 86 and 91 percent, and many concurrent short prompts' first token.

  Forty-third, a layer's projections of one type in one integer-dot tile dispatch. At 64 columns on one MI50 a 0.6B layer's q, k and v took 232 us as three tile calls and 89 us as one call over their 4096 rows, and gate and up 177 against 120: each projection alone left most of the device idle. The tile now takes up to three projections of one type as the row kernel does, their row tiles on consecutive workgroups, and a split call's reduce adds all their parts in one dispatch. A reduce workgroup that straddled two projections indexed their output buffers non-uniformly and lost the second one's writes; each projection now has whole workgroups of its own, which a new check of three grouped projections at 64 and 249 columns caught. The split is the fused group's, so a row still sums the same parts however its prompt is batched. Qwen3-0.6B-Q8_0 at 64 rows goes from 3754 to 5047 tok/s; short prompts, decode and server load are unchanged.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4917 vs 4883, 101% | 6858 vs 7224, 95% | 7452 vs 6795, 110% | 323 vs 331, 98% |
  | Qwen3-0.6B-Q5_K_M | 4528 vs 2971, 152% | 6692 vs 4436, 151% | 7455 vs 5747, 130% | 337 vs 314, 107% |
  | Qwen3-0.6B-Q8_0 | 5036 vs 4654, 108% | 7383 vs 6974, 106% | 8093 vs 6692, 121% | 271 vs 298, 91% |
  | Qwen3-8B-Q4_K_M | 696 vs 261, 267% | 848 vs 634, 134% | 894 vs 763, 117% | 86 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 760 vs 524, 145% | 889 vs 735, 121% | 936 vs 865, 108% | 50 vs 58, 86% |

  Prompt processing clears the reference in every cell but 0.6B Q4_0 at 247 rows, 95 percent. Decode is the gap now: 86 and 91 percent on the Q8_0 files, 97 and 98 on 8B Q4_K_M and 0.6B Q4_0.

  Forty-fourth, a Q8_0 decode kernel over the 8-bit twin. The reference's Q8_0 matvec at 4096 x 14336 on one MI50 takes 120.8 us, about 516 GB/s, where ours took 167 us at the same shape. Ours read the 16-bit twin, so every row's lanes read the whole activation vector at two bytes a value, and its wide path spread each load instruction over 16-byte pieces of every block pair; on the 8-bit twin that path was slower still. `matmul_vec_q8.comp` gives a subgroup two rows. Lane l covers quarter l % 4 of every (S / 4)-th block, so a step reads a contiguous run of each row, and loads its eight 8-bit activation values per column once for both rows, two four-wide dots per quarter. The matvec goes to 135 us; one row per subgroup read 151 and four 146. It is the row kernel for Q8_0 wherever the integer dot is native, so short prompts and a server's many decode rows take it too: a 5-token prompt on 0.6B Q8_0 goes from 774 to 1191 tok/s, and 16 concurrent requests from 583 to 625 tok/s to 875 to 1085, their first token at the median from 130 to 142 ms to 56 to 71. The HF gate passes every cell, and the backend test's references now take the 8-bit rounding for Q8_0 rows on such a device.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4956 vs 4878, 102% | 6866 vs 7230, 95% | 7446 vs 6815, 109% | 325 vs 328, 99% |
  | Qwen3-0.6B-Q5_K_M | 4489 vs 2867, 157% | 6712 vs 4447, 151% | 7441 vs 5755, 129% | 342 vs 314, 109% |
  | Qwen3-0.6B-Q8_0 | 5063 vs 4636, 109% | 7383 vs 6978, 106% | 8020 vs 6662, 120% | 325 vs 300, 109% |
  | Qwen3-8B-Q4_K_M | 697 vs 260, 268% | 848 vs 634, 134% | 896 vs 764, 117% | 86 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 757 vs 525, 144% | 889 vs 735, 121% | 934 vs 864, 108% | 60 vs 58, 103% |

  Three cells remain below the reference: 8B Q4_K_M decode at 97 percent, 0.6B Q4_0 decode at 99, and 0.6B Q4_0 prompt processing at 247 rows at 95.

  Forty-fifth, Q4_0 and Q4_1 staging in the integer-dot tile. A block's staged words w and w + 4 are the low and high nibbles of one source word, and two threads each loaded it and, for Q4_0, funnelled it. A thread that stages two or more words now takes source words whole and stages both halves. 0.6B Q4_0 prompt processing goes from about 6879 to 7285 tok/s at 247 rows and from 7467 to 7922 at 512.

  Tried and not kept: a Q4_K decode kernel on the Q8_0 kernel's plan, a subgroup on two rows. With one nibble word a lane it read 4 bytes of weights against 16 of header and 24 of activations per step, and the 14336 x 4096 matvec took 201 us against the row kernel's 92. With sixteen bytes a lane it took 103, and 8B Q4_K_M decode read 80 tok/s against 84, so two rows sharing their activations does not pay for Q4_K here. The reference's Q4_K matvec at that shape takes 61 us and its Q6_K 132 against our 169; that file's output head is Q6_K, about 510 MB a token, which is where its decode gap is.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4998 vs 4881, 102% | 7277 vs 7221, 101% | 7918 vs 6840, 116% | 321 vs 330, 97% |
  | Qwen3-0.6B-Q5_K_M | 4488 vs 2959, 152% | 6656 vs 4445, 150% | 7428 vs 5751, 129% | 339 vs 312, 109% |
  | Qwen3-0.6B-Q8_0 | 5095 vs 4659, 109% | 7360 vs 6976, 105% | 8043 vs 6698, 120% | 326 vs 299, 109% |
  | Qwen3-8B-Q4_K_M | 696 vs 261, 267% | 846 vs 634, 133% | 894 vs 764, 117% | 86 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 759 vs 526, 144% | 888 vs 736, 121% | 934 vs 864, 108% | 61 vs 58, 104% |

  Prompt processing clears the reference in every cell. Decode on 8B Q4_K_M and 0.6B Q4_0 is at 97 percent, the one gap left.

  Forty-sixth, the RMS norm over several workgroups a row. A decode profile put `rms_norm_rows` at 18 percent of 8B Q4_K_M's device time, and alone on an MI50 one 1024-wide row took 15 to 16.6 us against 4.2 for an empty dispatch. One workgroup took the row, and on such a device it also writes the 8-bit twin, a chain of dependent shuffles per value, so a 4096-wide row was sixteen serial steps on one compute unit. Now a row's 256-value chunks each have a workgroup; every one sums the whole row's squares in the same order, so all reach the same scale bit for bit, and writes its own chunk and its twin. When dst and src overlap, as before the output head, the row keeps one workgroup, since a chunk's output would change what another is summing.

  Decode goes from 331 to 351 tok/s on 0.6B Q4_0, from 85 to 94 on 8B Q4_K_M and from 58 to 65 on 8B Q8_0. Reducing the sum of squares through subgroups rather than the shared-memory tree took the norm's barriers from eight to one but added in another order, and the HF gate's Q8_0 fixture then ranked a different fifth token on one prompt, a top-5 overlap of 4 against its frozen 5; the tree stays and the norm's arithmetic is unchanged.

  Also tried and not kept, all on one MI50: a Q6_K decode kernel with a subgroup on two rows sharing their 16-bit activations, 175 us at 14336 x 4096 against the row kernel's 169; Q6_K with sixteen lanes a block, 225; and Q4_K with four lanes a block, 134 against 92. None of the row kernel's Q4_K or Q6_K lane layouts beats the one it has.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 5037 vs 4889, 103% | 7268 vs 7223, 101% | 7901 vs 6836, 116% | 340 vs 329, 103% |
  | Qwen3-0.6B-Q5_K_M | 4506 vs 2985, 151% | 6676 vs 4429, 151% | 7427 vs 5754, 129% | 359 vs 314, 115% |
  | Qwen3-0.6B-Q8_0 | 5108 vs 4653, 110% | 7361 vs 6978, 105% | 8028 vs 6688, 120% | 345 vs 299, 116% |
  | Qwen3-8B-Q4_K_M | 698 vs 261, 268% | 837 vs 633, 132% | 886 vs 764, 116% | 96 vs 89, 108% |
  | Qwen3-8B-Q8_0 | 762 vs 526, 145% | 879 vs 735, 120% | 925 vs 865, 107% | 65 vs 58, 111% |

  Every cell of the one-card MI50 gate clears the reference. The thinnest is 0.6B Q4_0 at 247 rows, 101 percent, within what one run can move.

  Forty-seventh, the gate on both platforms. The user set the device gate as llmx against llama.cpp's own Vulkan backend, the same backend type, on the Linux MI50 and on the Windows Radeon VII. A second MI50 run at `8c07a29`, four interleaved passes per file on cards 3 to 7, best of each arm:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 5044 vs 4883, 103% | 7259 vs 7104, 102% | 7869 vs 6739, 117% | 325 vs 318, 102% |
  | Qwen3-0.6B-Q5_K_M | 4512 vs 2932, 154% | 6651 vs 4362, 152% | 7385 vs 5677, 130% | 348 vs 303, 115% |
  | Qwen3-0.6B-Q8_0 | 5006 vs 4645, 108% | 7318 vs 6912, 106% | 8004 vs 6611, 121% | 331 vs 290, 114% |
  | Qwen3-8B-Q4_K_M | 695 vs 261, 267% | 831 vs 633, 131% | 874 vs 761, 115% | 93 vs 84, 110% |
  | Qwen3-8B-Q8_0 | 759 vs 524, 145% | 881 vs 732, 120% | 926 vs 863, 107% | 65 vs 57, 114% |

  The Radeon VII against llama.cpp b11075's Vulkan backend on the same card, two interleaved passes of five repetitions, best of each arm:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 1040 vs 952, 109% | 2241 vs 670, 335% | 2528 vs 1825, 138% | 228 vs 224, 102% |
  | Qwen3-0.6B-Q5_K_M | 982 vs 647, 152% | 2395 vs 526, 455% | 2613 vs 1174, 223% | 228 vs 219, 104% |
  | Qwen3-0.6B-Q8_0 | 1019 vs 922, 110% | 2869 vs 653, 440% | 3111 vs 1655, 188% | 215 vs 194, 111% |
  | Qwen3-8B-Q4_K_M | 175 vs 58, 300% | 239 vs 78, 308% | 280 vs 111, 252% | 60 vs 52, 115% |
  | Qwen3-8B-Q8_0 | 221 vs 80, 275% | 278 vs 99, 281% | 338 vs 147, 230% | 42 vs 39, 107% |

  The 0.6B Q4_0 decode cell was rerun four times alone, 228 to 239 against 223.5 tok/s; llmx decodes after a 247-token prompt and the reference from an empty one. Every cell on both platforms clears the reference.

  For information, not the gate: the mx fork `eefc4e732` for gfx906 with ROCm, pinned to one MI50, reads ahead of llmx's Vulkan backend on the Q8_0 files, 8B Q8_0 at 512 rows 1301 tok/s against 904 and its decode 71 against 66, and behind it on the 4- and 5-bit ones, 0.6B Q5_K_M at 64 rows 896 against 4388. A ROCm backend is where that comparison belongs.
- **Left:** the device gate passes on both platforms (the forty-seventh paragraph), the thinnest cells at 102 percent; the Q6_K and Q4_0 matvecs are still 1.28 and about 2.3 times the reference's; loading, which reads the whole file into host memory before uploading it; folding a layer's
  two RMS norms into the matmul that follows, worth a fifth of the
  barrier time measured in the twenty-ninth; the
  prompt pass at 32 to 128 rows (the twenty-seventh paragraph): the
  tile kernel's cost per tile, the step at exactly 64 rows, and
  splitting the tiled attention over a long history. On 0.6B the bound is the
  dispatch count, not a kernel; replaying a recorded pass (the server
  block above) is the lever there. On 8B the K-quant paths sit at 250
  to 256 GB/s against Q8_0's 400 with their loads and their sub-scale
  decode already trimmed; what remains is the per-lane work of the
  nibble unpacking itself, and 8-bit activations would buy another
  tenth at a numerical cost the CPU experiment measured near the bound. The same backend now runs on the Linux machine's MI50s
  under Linux through `docker/Dockerfile`, which needs no change to that
  shared machine: Debian's own Mesa driver enumerates all ten cards
  through `/dev/dri` and its own shader compiler is new enough, so the
  image carries the driver, the loader, the headers and the compiler and
  the host carries nothing. The whole CTest suite passes there with
  `backend-vulkan` running rather than skipping, its 1,172,518 kernel
  outputs matching the CPU backend on an MI50, which is the first device
  other than the Radeon VII to run these kernels. Reported bandwidth is
  lower than the Radeon VII's under the Windows driver (Q8_0 at
  4096 x 12288 reads 315 GB/s against 397, Q6_K's wide head 248 against
  232 the other way), untuned and not a like-for-like comparison, since
  the driver differs as well as the card.

  A real model followed. `llmx pull` fetched Qwen3-0.6B-Q8_0 on the Linux machine
  through the image's curl, at the same revision the Windows copy has,
  and the whole Python suite passes there on `vulkan:0`, the HF gate
  included: the continuous excerpt reads an NLL delta of 0.001264
  against a 0.010 bound and the windowed cases 0.002393 to 0.012396
  against 0.020, the same numbers the Radeon VII returns. That is the
  external correctness gate met on a second device.

  Performance there was measured against three reference arms on the
  same card, which is worth doing because they disagree by more than
  llmx does. Qwen3-0.6B-Q8_0, two passes each, five runs per point,
  arms interleaved in the same minutes. The reference's Vulkan build is
  the same commit as its ROCm one, `7ab4ee7ba` (build 11100), compiled
  in the dev container; the fork is mx-llama.cpp `eefc4e732` built for
  gfx906, the arm the gate in `AGENTS.md` names. None of these is the
  b11075 Vulkan build the Radeon VII tables above use, so the tables
  are not a series.

  | test | reference Vulkan | reference ROCm | mx-llama.cpp ROCm | llmx Vulkan |
  |---|---:|---:|---:|---:|
  | pp64 | 2009, 2016 tok/s | 1774, 1980 | 4549 | 830, 826 |
  | pp256 | 3453, 3474 | 5419, 5434 | - | 1430, 1477 |
  | pp512 | 3550, 3560 | 6087, 6114 | 6782 | 1783, 1783 |
  | tg64 | 102.6, 101.3 | 226.4, 229.6 | 229.8 | 246.2, 246.2 |

  Read against the like-for-like arm, the reference's own Vulkan
  backend on the same card and driver, llmx decodes 2.4 times faster
  and prompt-processes at 41 to 50 percent. Decode is the striking
  figure: the reference's Vulkan decode collapses to 102 tok/s here
  while its ROCm decode holds 226, and llmx's Vulkan decode reaches
  246, past the vendor path. **Correction (2026-09-22):** the reference arms in this table ran with all ten MI50s visible and split the model across them, so neither its decode collapse nor the 2.4 times is a same-card result; see the thirty-fourth paragraph. That says the reference's Vulkan backend
  is far more driver-sensitive than ours, since on the Radeon VII under
  the Windows driver the same comparison is 205 against about 199.
  Prompt processing is where llmx is behind on this card by every arm:
  50 percent of the reference's Vulkan, 29 of its ROCm, 26 of the
  fork's. The fork matters: it reads 4549 tok/s at a 64-token prompt
  against upstream's 1774, so a share quoted against upstream flatters
  llmx, and it is the arm the gate names where it is available.

  Why prompt processing is behind, from the MI50 figures alone, since
  those are the ones taken in a single environment. At pp512 there the
  reference's Vulkan reads 3555 tok/s and llmx 1783. The model's
  prompt pass is about 450 GFLOP at 512 tokens, so llmx runs at 1.57
  of this card's 13.4 fp32 teraflops, 12 percent of peak, and the
  reference at 23. The weights are read once per pass, 640 MB in 287
  ms, nowhere near the card's bandwidth, and the 256 dispatches of a
  pass cost under a millisecond of it, so neither arm is bound by
  memory or by launch overhead. What does bind `matmul_tile.comp` is
  shared-memory traffic per operation: at a 4 by 4 micro-tile a thread
  reads eight floats per sixteen multiply-adds, two operations per
  read, and at this card's shared-memory bandwidth that ratio lands
  close to the 12 percent measured. The work is to raise the
  operations per read, which means a wider micro-tile.

  Done, and the shape is now chosen rather than fixed. A thread
  accumulates `TILE_ROWS / 16` rows by four columns, and `TILE_ROWS` is
  a specialization constant, so one SPIR-V module builds both a 64-row
  and a 128-row pipeline and the backend picks per dispatch. The taller
  tile reads twelve values from shared memory per thirty-two products
  where the shorter reads eight per sixteen, two thirds of the traffic
  for the same work; it also halves the workgroups, so a call whose
  taller form would produce fewer groups than the device has compute
  units takes the shorter one. Compute units come from
  `VK_AMD_shader_core_properties` where the driver has it, and a
  deliberately small assumption otherwise, which prefers the shorter
  tile and starves nothing. Measured on the Radeon VII, three runs a
  point, interleaved, the two fixed heights against the choice:

  | case | 4 by 4 fixed | 8 by 4 fixed | chosen |
  |---|---:|---:|---:|
  | 0.6B pp64 | 837 tok/s | 732 | 844 |
  | 0.6B pp128 | 1814 | - | 1919 |
  | 0.6B pp256 | 1473 | 2344 | 2772 |
  | 0.6B pp512 | 1795 | 2820 | 2821 |
  | 8B pp256 | 243 | 262 | 265 |
  | 8B pp512 | 267 | 309 | 309 |

  Choosing beats both fixed heights everywhere, and beats the taller
  one at 256 rows because the projections of a dispatch differ: a
  1024-row projection and a 3072-row one do not want the same tile.
  Decode is untouched, the row kernel taking those batches, and reads
  197 to 205 tok/s across runs either way, which is this session's
  drift.

  Verified where the reference comparison lives, all three arms on the
  MI50 in one container, five runs a point, two passes:

  | test | reference Vulkan | llmx before | llmx after |
  |---|---:|---:|---:|
  | pp64 | 2027 tok/s | 828 | 831 |
  | pp256 | 3474 | 1454 | 2253 |
  | pp512 | 3555 | 1783 | 2619 |
  | tg64 | 101.6 | 246 | 246 |

  Prompt processing goes from 50 to 74 percent of the reference's
  Vulkan at 512 rows and from 42 to 65 at 256; 64 rows is unchanged at
  41 percent, that call being too small to fill the card either way,
  and decode stays 2.4 times ahead, a ratio void for the reason corrected in the thirty-fourth paragraph. The kernels remain exact there,
  1,172,518 outputs against the CPU backend.

  What this fixes beyond the number is the structure. The workgroup
  size, tile shape and shared-memory arrays of this kernel were
  literals tuned to one card; the row kernel had long adapted, reading
  the device's subgroup width and computing lanes per row and rows per
  subgroup on the host. The tile kernel now adapts too, by the
  mechanism Vulkan provides for it, so a device with a different
  subgroup width, a different shared-memory limit or a different ratio
  of shared-memory bandwidth to arithmetic gets a different shape from
  the same source. The shared-memory limit was the next input tried to
  that policy, and it did not pay. A 128 by 128 shape halves the shared
  memory read per product again, sixteen reads per sixty-four products
  against twelve per thirty-two, and its 33792 bytes fit the 64 KB Mesa
  reports where they do not fit the 32 KB the AMD proprietary driver
  does, so it would have been the first thing that capability bought.
  On the MI50 it took prompt processing at 512 rows from 2619 to 1144
  tok/s. Eight columns to a thread means the inner loop holds eight of
  them, and indexing that by the loop variable put them where the fast
  path does not want them; the per-row kernel had learned the same
  thing earlier, where accumulator arrays indexed by column cost it ten
  times. Generalising the inner loop to a column count cost the
  4-column form as well, 0.6B pp256 reading 2642 tok/s against 2796
  with the named scalars while gaining 2 percent at 512, so the scalars
  stay and the column count stays a literal. A wider tile was then
  revisited written as named scalars per column, compiled as its own
  module from the same source under a define the way the row kernel's
  families are, so nothing was indexed and nothing was generalised. It
  is still bad, and worse than the indexed version suggested. Two
  shapes became selectable, 64 by 128 needing 25344 bytes and 128 by
  128 needing 33792, and the narrower fits the 32 KB the AMD
  proprietary driver reports, so it ran on both cards: 0.6B pp512 fell
  from 2818 to 458 tok/s on the Radeon VII and from 2628 to 576 on the
  MI50, with the 8B falling from 292 to 95 there. Prompt sizes that did
  not select it were unchanged, so the four-column path was intact and
  the eight-column one is simply slow.

  That is worth stating plainly because the arithmetic said otherwise.
  A 64 by 128 tile and a 128 by 64 tile read the same shared memory per
  product, hold the same 25344 bytes, and the wider one re-reads the
  weights half as often, so it should have been at least even. It is
  six times slower. Two explanations were tested and both fail. Shared
  memory bank conflicts: with 33-float padding, sixteen threads taking
  eight contiguous columns each land in four banks where four columns
  each land in eight, so the columns a thread owns were restrided by
  sixteen to spread them over sixteen banks, and pp512 still read 457
  tok/s. Register pressure or spilling: the driver reports the
  eight-column kernel at 96 vector registers with no scratch at all,
  against 87 for the four-column one, which is the same two waves per
  SIMD. So it is not the reads per product that the twelve-against-eight
  argument counts, not the banks those reads fall in, and not spilling.
  Until it is understood a wider micro-tile is not the lever it looked
  like, and four columns stay.

  The disassembly was then read, and it answered a different question
  than the one it was opened for. The inner loop of the tile kernel
  issued, per step: six address calculations, six shared-memory reads,
  two waits and sixteen multiply-adds. Half the instructions were not
  arithmetic. Two causes, neither visible in the source. The loop was
  not unrolled, so the driver recomputed every address each step rather
  than folding the step into the read's immediate offset. And the
  shared tiles were held row-major, `[row][k]`, so the four values a
  thread wants for one k sat 33 floats apart and needed six reads.

  Both were fixed by changing where the values sit rather than what the
  kernel computes. The tiles are k-major now, `[k][row]` and
  `[k][column]`, which makes a thread's four rows and four columns
  adjacent: the driver emits one 128-bit read and two paired reads
  where it used to emit six, and registers fell from 87 to 79. The k
  loop is unrolled by four on top of that, so the four steps differ by
  a constant. Interleaved on the Radeon VII, two passes, three models,
  every cell improves and none regresses:

  | case | before | after |
  |---|---:|---:|
  | 0.6B pp128 | 1904 tok/s | 1925, 1935 |
  | 0.6B pp256 | 2744 | 2795, 2799 |
  | 0.6B pp512 | 2798, 2800 | 2946, 2960 |
  | 8B Q8_0 pp256 | 264 | 288 |
  | 8B Q8_0 pp512 | 307 | 333 |
  | 8B Q4_K_M pp512 | 108 | 109 |

  On the MI50 it is worth more: 8B pp32 86 to 103 tok/s, pp64 158 to
  171, pp512 291 to 323, and 0.6B pp96 972 to 1208. That the same
  source change is worth twice as much under Mesa is the same pattern
  as everything else here, the driver deciding what a shape costs.

  The unroll was then deepened from four to eight, which removes more
  of the address arithmetic still being emitted. Interleaved, two
  passes, three models, every cell improves: 0.6B pp512 2999 to 3033
  tok/s, pp256 2851 to 2867, 8B Q8_0 pp512 339 to 345, pp256 294 to
  297, 8B Q4_K_M pp512 110.7 to 111.2. Sixteen and thirty-two measured
  the same as eight, so eight is where the return stops and the
  shallower form is kept.

  Splitting the inner dimension was then written and rejected. At 64
  prompt rows a tile dispatch yields 16 workgroups where the MI50 has
  64 compute units, so the kernel is starved and the row kernel wins
  there by default; the fix for that is to cut the inner dimension into
  slices, run a dispatch per slice, and add the partial sums in one
  pass. It was implemented, a `matmul_combine` kernel and a k range on
  the tile kernel, correct on both cards, and slower everywhere: on the
  Radeon VII 0.6B pp64 781 tok/s against 850 and pp128 1879 against
  1961, and on the MI50, with the tile kernel forced so the split could
  engage at all, pp32 528, pp64 941, pp96 1126 and pp128 1589 against
  1071, 1042, 1168 and 1610 unsplit.

  The arithmetic says why, and it was predictable: a slice writes a
  whole nbatch by nout array of floats and the combine pass reads them
  all back, so the traffic added is twice slices times nbatch times
  nout times four bytes. At pp64 on a 1024 by 1024 projection that is
  2 MB against the 1.1 MB of weights the call reads at all, roughly
  tripling its memory traffic to buy four times the workgroups.
  Splitting pays only where the weights dwarf the output, which is not
  this shape. Reverted; the row kernel keeps those batches.

  A profiler followed, because until now every claim about where time
  went was inferred from kernels timed alone or read off their machine
  code, neither of which sees overlap or idle. A diagnostics backend
  now writes a timestamp either side of every dispatch and
  `bench --model --profile` reports device time per kernel. The first
  run, 8B Q4_K_M decode on the Radeon VII, 64 tokens:

  | kernel | device ms | share |
  |---|---:|---:|
  | matmul_row_k4 | 229.5 | 67.1% |
  | matmul_row_k | 78.1 | 22.8% |
  | rms_norm_rows | 20.0 | 5.9% |
  | attention_kv16 | 6.1 | 1.8% |
  | norm_rope_kv_kv16 | 4.6 | 1.4% |
  | silu_mul, quantize_x, embed, gather_rows | 4.0 | 1.2% |

  Ninety percent is the two K-quant matmuls, so nothing is hiding
  outside the kernels this cell is about. The line that changes the
  plan is `rms_norm_rows` at 5.9 percent: its work is one row of 4096
  values, trivial, and at two dispatches a layer over 36 layers times
  the 3.8 us a boundary costs (the twenty-ninth paragraph) that is
  0.24 ms of the 0.31 ms it shows. It is almost entirely the boundary.
  Removing those dispatches is worth about 4.6 percent of decode, which
  is most of the 6 percent this cell is short.

  The profile also settles what the dominant kernel is bound by, which
  three earlier guesses had not. Sampling 4096 dispatches, about 14
  tokens, the Q4_K matmul takes 9.5 ms of a 20.7 ms token and reads the
  model's 3.705 GB of Q4_K weights in that time, which is 390 GB/s
  against roughly 1000 of this card's peak. The Q6_K matmul reads its
  1.316 GB at about 306 GB/s. Neither is close to memory bound; both
  are instruction bound, at about four operations per weight with a
  quarter-rate 32-bit multiply among them. Making that multiply full
  rate is worth roughly 40 percent of the dominant kernel, which is far
  more than this cell is short.

  And the instruction exists, which four earlier probes had missed
  because they all asked for integer dots. This driver refuses to emit
  an integer dot under any formulation tried: the extension, packed
  8-bit dots, and two ways of stating operand widths all produce the
  same wide multiplies. Asked for a half-precision dot it emits
  `v_dot2_f32_f16`, the chip's native one, which takes two half
  products into a float accumulator at full rate. The probe that found
  it pays for it in conversions, 195 int-to-float and 96 float-to-half
  against 48 dots, because it was handed integer activations; the point
  is only that the instruction is reachable here.

  What that implies is a different activation format for the row
  kernel: halves rather than 16-bit integers with a block scale. A
  weight nibble becomes a half in one conversion, the dot takes two
  products at full rate, and the block scale still multiplies the sum
  afterwards, so the arithmetic is about 2.5 full-rate operations per
  weight against the present five effective. The integer twin exists
  because integer dots needed a shared scale; halves carry their own
  exponent and need none. Not yet written, and it touches the twin's
  producers as well as every row path, so it is the next substantial
  piece rather than a tune.

  Eight columns to a thread was retried on the k-major layout, where a
  thread's eight columns are contiguous rather than 33 floats apart,
  which was the objection to the first two attempts. It is still
  catastrophic: 0.6B pp512 547 tok/s against 2953, 8B pp512 127 against
  333, at 97 registers with no scratch. That is three attempts, as an
  indexed array, as named scalars row-major, and as named scalars
  k-major, all within a factor of the same result, so the effect is a
  property of the hardware rather than of any one way of writing it.
  Widening a thread's rows from four to eight helped; widening its
  columns the same way does not, and the asymmetry is unexplained. Four
  columns stay, and the remaining inner loop is close to its minimum:
  two reads and sixteen multiply-adds per step, about three quarters of
  issued instructions being arithmetic.

  Declaring the tiles as four-wide vectors, to force both reads to
  128 bits rather than leaving the columns as a pair of reads, was also
  measured and is slightly worse: 8B pp512 320 tok/s against 333.

  The same method then found the decode lever, and it is a driver
  difference again. The per-row matmul's inner loop spent, per 16-bit
  product, a sign-extend pair and a `v_mul_lo_u32`, which is a full
  32-bit multiply and runs at quarter rate on gfx906. The chip has a
  native 16-bit dot instruction and the integer dot product extension
  maps to it, but only under Mesa: the AMD proprietary driver lowers
  that extension back to the same multiplies with the operands widened
  first, which is why the extension was dropped this morning on the
  strength of measurements taken here. Under Mesa the dot form emits
  128 of the native instructions where the multiply form emits 35 wide
  multiplies, and per-type bandwidth on the MI50 goes from 222 to 241
  GB/s on Q4_K, 239 to 284 on Q5_K and 213 to 254 on Q6_K.

  Neither form wins everywhere, so both are compiled from the one
  source under a define and the backend picks per device, availability
  being a capability and worth being a measurement:

  | model | multiplies | dots | device |
  |---|---:|---:|---|
  | 8B Q8_0 | 39.8, 39.4 tok/s | 45.7, 45.7 | MI50, Mesa |
  | 0.6B Q8_0 | 246.8, 245.8 | 262.7, 261.3 | MI50, Mesa |
  | 8B Q8_0 | 40.9, 40.7 | 40.1, 40.3 | Radeon VII |
  | 8B Q4_K_M | 47.9, 48.0 | 46.6, 47.0 | Radeon VII |

  So 15 percent of 8B decode and 6 of 0.6B on the MI50, and this
  workstation keeps the multiplies and is unchanged. One ordering bug
  on the way, worth noting because the structure invites it: the
  capability flags were filled before the extension scan that
  discovers them, so the MI50 silently kept the slower form until the
  profile decision moved after the scan.

  Against the reference's own Vulkan build on the MI50, both arms in
  one container, five runs a point, two passes, after all of the day's
  kernel work: pp64 1075 tok/s against 2016, pp256 2518 against 3469,
  pp512 2743 against 3556, decode 262 against 102. Prompt processing is
  53, 73 and 77 percent of it, against 41, 65 and 74 this morning, and
  decode is 2.6 times ahead rather than 2.4. On the same card the
  per-type sweeps read 0.6B pp16 to pp96 at 1029, 1071, 1092, 1103 and
  1218 tok/s against 851, 883, 897, 904 and 1208, the short prompts
  gaining a fifth from the dot-form row kernel; the 8B reads pp32
  104.5, pp64 180.5 and pp512 327.

  The tile threshold turns out to be a property of the driver, not only
  of the card, which the profile can hold but cannot yet derive. It was
  measured at 32 rows for 8-bit projections on the Radeon VII under the
  AMD proprietary driver. On the MI50 under Mesa, the same silicon, the
  row kernel wins all the way to 96: at 16, 32, 48, 64, 96 and 128 rows
  the tile reads 230, 422, 614, 825, 940 and 1355 tok/s on
  Qwen3-0.6B-Q8_0 while the row kernel reads 852, 883, 897, 900, 890
  and 889. So between 32 and 96 rows the shipped threshold picks the
  slower kernel there, by up to a factor of two at 32.

  The first attempt was to make the threshold follow the shape rather
  than the device, since the crossover also moves with how much work a
  row carries: 64 rows on a 1024-wide 8-bit projection against about 24
  on a 4096-wide one. Interleaved on the Radeon VII across three
  models, two passes each, that was a regression, 0.6B pp48 reading 825
  tok/s against 978, and a wash everywhere else, so it is not in the
  tree. The shape does move the crossover, but less than the driver
  does, and a rule fitted to one device mispredicts the other.

  Measured again with the current tile kernel, which had moved the
  crossover since the table above, the picture is that width matters as
  much as the driver. Forcing each kernel and sweeping rows on
  Qwen3-0.6B-Q8_0 and Qwen3-8B-Q8_0:

  | device | 1024-wide projection | 4096-wide |
  |---|---:|---:|
  | Radeon VII, AMD driver | 40 rows | 26 |
  | MI50, Mesa | 96 | 30 |

  A narrow projection wants a much higher threshold than a wide one on
  both devices, which is why one constant of 32 was wrong for the small
  model everywhere and right for the 8B everywhere. The threshold is
  now 64 rows when a row carries fewer than 4096 values and 32
  otherwise, the boundary put at 4096 because it is what separates the
  two models: the 0.6B's widest projection is 3072 and the 8B's
  narrowest is 4096. An earlier attempt at the same idea put the
  boundary at 2048, which left the 0.6B feed-forward projection on the
  wide side, so half the call still took the wrong kernel and the two
  arms measured the same.

  Interleaved, two passes, three models on the Radeon VII: 0.6B pp32
  goes from 694 and 705 to 834 and 845 tok/s, pp48 from 973 and 978
  down to 851 and 852, and pp16, pp64, pp96, pp128, pp512 and every 8B
  figure are unchanged. On the MI50 the same change reads off the
  sweeps as 422 to 883 tok/s at 32 rows, 614 to 897 at 48 and 825 to
  900 at 64, with nothing given up, since its crossover is further from
  the shipped value. So it costs 13 percent in one band on one device
  to gain 20 percent in another there and 9 to 109 percent across three
  bands on the other.

  Verified on the MI50 afterwards rather than predicted from the
  sweeps: 0.6B pp16, pp32, pp48, pp64 and pp96 read 851, 883, 897, 856
  and 972 tok/s, against 852, 422, 614, 825 and 940 before, and the 8B
  reads 85.9, 157 and 292 at pp32, pp64 and pp512 against 83.7, 154 and
  about 290. So the band that was picking the slower kernel there is
  closed, and nothing regressed.

  The per-device ideal being 48 rows here and 96 there, the compromise
  is now only what an unmeasured device gets. `measured_profiles` in
  `backends/device_profile.hpp` is a table keyed by what a device and
  its driver call themselves, holding what that combination actually
  wanted; the two measured entries are the Radeon VII under the AMD
  proprietary driver at 48 and the MI50 under Mesa at 96. A device in
  the table runs better than the defaults, one that is not runs exactly
  as before, and bringing up hardware is running the sweeps and adding
  a row. It keys on device and driver together because the driver is
  what the measurement moved with: the same Vega20 wants 40 rows under
  one and 96 under the other.

  Measured after wiring it: the Radeon VII reads 971 tok/s at pp48
  against 851 with the compromise, which is the 13 percent the single
  number had given up, and the MI50 reads 896 at pp64 against 856.
  Nothing else moved on either.

  A second, separate observation, and only an observation: the two
  drivers report different limits for the same Vega20 silicon, 32 KB
  of shared memory per workgroup from the AMD proprietary driver on
  Windows and 64 KB from Mesa. Our tile is fixed at 64 by 64 using
  16896 bytes and the backend never reads
  `maxComputeSharedMemorySize`, so whatever a device offers above that
  goes unused, which is worth fixing on its own terms. It is not
  evidence about the reference: an earlier revision of this paragraph
  argued from llmx reading 1795 on the Radeon VII against 1783 on the
  MI50 while the reference went 1662 to 3555, and that comparison is
  void. The two reference figures are different builds (11075 against
  11100), different compilers, operating systems and drivers, three
  changes at once, which is the mistake the build-identity rule in
  AGENTS.md exists to prevent, applied to environments rather than
  binaries. Every prompt-processing measurement from here is llmx
  before against llmx after against the reference, all three in one
  environment.

  The tiled attention does not yet share a K/V tile across the query
  heads of a KV group.

## KV cache fork, step 2 of the KV design (2026-09-21)

- **Goal:** a second history with the same committed tokens, sharing every
  full block and copying the partial tail, which the server needs for
  prefix reuse and branching and which the fork rule in KV-CACHE.md had
  described without implementing.
- **Done:** `KVSequence::fork` retains every full block and takes a fresh
  block for a partial tail, reporting the ids so the caller fills it;
  `Backend::kv_copy` copies every layer's K and V of one block into
  another, because only the backend knows the layout; `Model::fork` does
  both on every storage and the fork inherits the tickets of the passes
  that wrote what it shares. Shared blocks are read-only: `prepare`
  refuses to append into one, which a history truncated into a shared
  block would do, so such a history is forked again instead. `kv-cache`
  checks a fork across a block edge (shared block at refcount two, tail
  private and copied, both histories diverging without touching the
  shared block, a boundary fork taking no tail, the refused append,
  releases following the refcounts) and, through the model, that a forked
  sequence continues exactly as a fresh one fed the same tokens while the
  original continues exactly as if never forked. Native 19/19, Python
  12/12 with both HF models.
- **Done: gate**, base `c85aa9d`, candidate `215ad24`, layout control
  `d8d508e`. Five 0.6B cells at 15 pairs, one 8B at 9. System CPU averaged
  47 and 43 percent in the first two 0.6B cells and 38 to 39 in the rest,
  against the benchmark's own 37.5, so other activity was present during
  the two cells that moved most. Evidence in
  `docs/benchmarks/kv-fork-20260921/`, raw monitors archived and hashed;
  every sample kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -1.39% / -2.54% / 9/15 | -1.59% / -1.05% / 9/15 |
  | 0.6B-1 | decode  | +3.38% / +2.15% / 5/15 | +3.23% / +0.95% / 7/15 |
  | 0.6B-2 | prefill | **-3.92% / -3.51% / 11/15 FAIL** | +1.53% / +2.26% / 5/15 |
  | 0.6B-2 | decode  | **-3.66% / -4.01% / 12/15 FAIL** | +1.31% / -0.16% / 8/15 |
  | 0.6B-3 | prefill | -0.76% / -2.13% / 9/15 | +1.45% / +1.36% / 6/15 |
  | 0.6B-3 | decode  | -1.12% / -0.06% / 8/15 | -0.24% / -0.39% / 8/15 |
  | 0.6B-4 | prefill | -2.20% / -2.35% / 11/15 | -1.45% / -0.74% / 9/15 |
  | 0.6B-4 | decode  | +0.01% / +0.55% / 7/15 | +0.28% / -0.21% / 8/15 |
  | 0.6B-5 | prefill | -1.77% / -1.53% / 10/15 | -0.48% / -0.97% / 8/15 |
  | 0.6B-5 | decode  | -0.60% / -0.40% / 10/15 | -1.96% / -2.10% / 10/15 |
  | 8B | prefill | -0.23% / +1.08% / 4/9 | +0.01% / +0.55% / 4/9 |
  | 8B | decode  | +0.31% / +0.00% / 4/9 | -1.00% / -0.22% / 6/9 |

  Cell 2 failed both phases at about -3.8 percent with the control positive
  in the same cell; the three cells after it pass and 8B is flat. What the
  measured path gained from this change is one refcount comparison per
  pass in `prepare`. Two things are written down rather than argued away.
  0.6B decode, the phase that comparison sits on, is flat over five cells:
  +3.38, -3.66, -1.12, +0.01, -0.60, mean -0.4 against a control mean of
  +0.5. 0.6B prefill is under base in all five cells, mean -2.0 against a
  control mean of -0.1, which is the same size and sign as the step 2 gate
  and the opposite of the step 6 gate on changes of the same character;
  it is inside the band AGENTS.md documents for this file and this model,
  and it joins the running list of candidates if 0.6B prefill is ever
  found a few points low against an older baseline.
- **Left:** nothing. The prefix index is step 4 of the KV design and lands
  with the server.

## Execution model for batching and placement (ROADMAP #5, #7) (2026-09-21)

- **Goal:** fix what the backend interface and the model layer need for the
  per-layer and per-tensor splits and for continuous batching, now that both
  are scoped and per-row split is dropped, so the first vendor backend
  implements each signature once.
- **Done:** the design, `docs/EXECUTION.md`: `submit`/`wait` tickets with
  no events, host-visible memory and `write` returning with the transfer as
  its caller, per-row positions with the RoPE table as a buffer, batched KV
  views with `gather_rows`, and `Model` / `Sequence` / `ExecContext` /
  `Batch`. Seven ordered steps; steps 1 to 4 change the interface and go
  before Vulkan. ROADMAP #4a marked done, #5 rewritten without per-row, #7
  pointed at the design; stale "steps 5 and 6 remain" claims corrected in
  ARCHITECTURE, DEVICE-EXECUTION, README and the backend page.
- **Done:** amended for architectures beyond dense Qwen (hybrid compressed
  attention, MoE, lookup-table memory, residual mixing): tables are per
  storage counted in that storage's entries, `kv_alloc` describes an entry
  by key and value widths, placement is per tensor role with attention and
  feed-forward separate, and a sequence may hold private unpaged state.
  The "Beyond dense Qwen" section lists the assumptions steps 3, 4 and 6
  must not make.
- **Done: step 1.** `norm_rope_rows` takes the cos/sin tables as buffers
  and one position per row; the model adopts its table once and passes a
  slice of an identity position table. `Backend::rope` is deleted; `bench`
  times `norm_rope_rows` instead of a function the runtime never called.
  `backend-group` checks three rows at positions 5, 2 and 9 against a
  double-precision reference reading the table at each row's own position.
  Native 18/18, Python 12/12 with both HF models.
- **Done: step 1 gate**, three arms built the same way from detached
  worktrees: base `f1e0be4`, candidate `7519170`, and a layout control
  `769761e` that is base plus an unused function appended to each of the
  two files the change edits (`cpu_backend.hpp`, `arch_qwen.hpp`), per the
  AGENTS rule that the control must perturb the same files. Qwen3-0.6B-Q8_0
  five times at 15 pairs, Qwen3-8B-Q8_0 once at 9, six threads, 250-token
  prompt, 32 generated tokens. Evidence in
  `docs/benchmarks/rope-positions-20260921/`; raw monitors archived beside
  the repo and hashed in each cell's `monitor-summary.json`. System CPU
  averaged 40 to 46 percent across cells against the benchmark's own 37.5,
  so other activity was present throughout; every sample is kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | +3.83% / +3.84% / 4/15 | -2.61% / -0.42% / 9/15 |
  | 0.6B-1 | decode  | +1.47% / +1.80% / 5/15 | +0.33% / +3.32% / 5/15 |
  | 0.6B-2 | prefill | -0.29% / -0.06% / 8/15 | -1.33% / +0.16% / 7/15 |
  | 0.6B-2 | decode  | +0.46% / +0.91% / 5/15 | +1.45% / +0.87% / 6/15 |
  | 0.6B-3 | prefill | -0.35% / +2.29% / 6/15 | +0.73% / +0.03% / 7/15 |
  | 0.6B-3 | decode  | **-2.31% / -1.75% / 13/15 FAIL** | -2.22% / -1.61% / 10/15 |
  | 0.6B-4 | prefill | +1.60% / +0.06% / 7/15 | +0.78% / +0.38% / 7/15 |
  | 0.6B-4 | decode  | -0.08% / -0.27% / 8/15 | -1.29% / -1.39% / 10/15 |
  | 0.6B-5 | prefill | -0.86% / -0.81% / 8/15 | +0.44% / +0.09% / 7/15 |
  | 0.6B-5 | decode  | +0.51% / +0.34% / 6/15 | +2.06% / +1.84% / 6/15 |
  | 8B | prefill | +1.85% / +0.23% / 3/9 | +2.01% / +0.08% / 4/9 |
  | 8B | decode  | +0.14% / +0.45% / 4/9 | +0.75% / +0.45% / 2/9 |

  Fails at 12 of 15 and 8 of 9 base wins. One cell failed, 0.6B-3 decode,
  on the win count. In that same cell the control, which executes the
  same instructions as base, lost 2.22 percent with 10 base wins, so the
  cell moved against both later arms rather than against the change, and
  the two reruns the goal requires came back at -0.08 and +0.51 with 8 and
  6 base wins. Over five 0.6B cells the candidate's decode mean is
  +0.01 percent. Recorded as noise confirmed by rerun, with the failing
  cell kept. The +3.83 prefill in cell 1 is not claimed either; the
  control spans -2.61 to +0.78 on the same measurement.
- **Done: step 2.** `submit()` returns a monotonic ticket and `wait()`
  blocks on one; `sync()` stays what the failure paths call, because a
  failed pass has ops queued behind no ticket. `alloc` takes a `Memory`
  kind; the logits buffer is host visible and the host reads it in place
  after the wait, so no read op copies a row it can already see. The model
  submits once per pass, waits on that ticket for the logits, and `reset`
  waits on the last ticket instead of draining. `kv-cache` counts the
  calls: one submission waited on once per step, no read op for logits, a
  three-token prompt at ubatch 2 submitting twice and waiting once, and a
  reset that waits without a sync. Native 18/18, Python 12/12 with both HF
  models.
- **Done: step 2 gate**, same shape as step 1: base `2bbfd34`, candidate
  `4ee6f48`, layout control `bd0cdc7` perturbing `cpu_backend.hpp` and
  `arch_qwen.hpp`. Three 0.6B cells at 15 pairs, one 8B at 9. System CPU
  averaged 37 to 40 percent per cell against the benchmark's own 37.5, so
  the machine was close to quiet; every sample is kept. Evidence in
  `docs/benchmarks/tickets-20260921/`, raw monitors archived and hashed.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -1.72% / -2.45% / 10/15 | -1.03% / -1.20% / 9/15 |
  | 0.6B-1 | decode  | +0.58% / +1.04% / 4/15 | +0.79% / +0.21% / 7/15 |
  | 0.6B-2 | prefill | -1.10% / -1.54% / 9/15 | -1.98% / -0.24% / 9/15 |
  | 0.6B-2 | decode  | +1.02% / -0.04% / 8/15 | -0.84% / -1.84% / 9/15 |
  | 0.6B-3 | prefill | -2.02% / -2.84% / 10/15 | +0.21% / -0.43% / 9/15 |
  | 0.6B-3 | decode  | +0.67% / +1.03% / 6/15 | +2.02% / -0.54% / 8/15 |
  | 8B | prefill | +0.54% / +0.36% / 3/9 | +0.74% / +1.00% / 1/9 |
  | 8B | decode  | +0.86% / +0.66% / 3/9 | +0.98% / +0.66% / 2/9 |

  All eight cells pass. The candidate's 0.6B prefill is below base in all
  three cells, by 1.1 to 2.0 percent on the mean, which is inside the band
  the control itself spans (-1.98 to +0.21) and the change adds nothing a
  prefill executes beyond one counter increment per pass. Recorded rather
  than explained away; if a later step finds 0.6B prefill a point or two
  low against an older baseline, this is a candidate along with the code
  read's three points.
- **Done: step 3.** `attention` and `kv_write` take an array of `KVView`s
  laid out in row order, and a view carries `nq`, the rows of the pass that
  are its own; `pos` and `batch` are gone because both follow from the
  view, and the sequence fills `nq` from what it has prepared and not
  committed. The CPU backend takes the views one after another around the
  code it had, so one view computes what it did before. `kv-cache` checks
  two sequences with histories of different lengths across a block edge,
  written and attended in one call, against the same two taken separately,
  bit for bit. `gather_rows` is deferred to step 4, where a `Batch` gives
  it a real caller. Native 18/18, Python 12/12 with both HF models.
- **Done: step 3 gate**, base `c850149`, candidate `8a3ff82`, layout
  control `a09c584` perturbing the two edited files. Three 0.6B cells at
  15 pairs, one 8B at 9. System CPU averaged 37 to 39 percent per cell
  against the benchmark's own 37.5; every sample is kept. Evidence in
  `docs/benchmarks/views-20260921/`, raw monitors archived and hashed.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -0.60% / -0.32% / 8/15 | +0.97% / +0.81% / 6/15 |
  | 0.6B-1 | decode  | **-1.21% / -1.59% / 12/15 FAIL** | -0.76% / -0.53% / 8/15 |
  | 0.6B-2 | prefill | +1.83% / +2.08% / 5/15 | +2.98% / +1.27% / 7/15 |
  | 0.6B-2 | decode  | +0.10% / +0.18% / 7/15 | +0.52% / +0.81% / 5/15 |
  | 0.6B-3 | prefill | +1.82% / +1.37% / 4/15 | +2.45% / +2.25% / 4/15 |
  | 0.6B-3 | decode  | +1.75% / +2.59% / 5/15 | +2.12% / +1.98% / 5/15 |
  | 8B | prefill | +1.60% / +0.03% / 4/9 | +1.42% / -0.31% / 6/9 |
  | 8B | decode  | -0.10% / +0.22% / 4/9 | -0.08% / -0.65% / 5/9 |

  One cell failed, 0.6B-1 decode, on the win count alone at exactly the
  threshold; its mean and median are inside the noise band and the control
  was negative in the same cell. The two cells run after it, identical in
  plan, came back at +0.10 and +1.75 with 7 and 5 base wins, and 8B decode
  is flat. Recorded as noise confirmed by rerun, with the failing cell
  kept.
- **Done: step 4.** `Model` owns the weights, the cache's pool and physical
  storage and the backend, read-only after construction apart from pool
  bookkeeping. `Sequence` is one request's history and its last ticket,
  `ExecContext` one pass in flight, and `Model::forward` runs one pass over
  a batch of entries, each a sequence with tokens to append and whether it
  wants logits; rows that want logits are compacted by `gather_rows`, which
  gets its first caller, and the head runs once over them. `forward`
  submits and returns; the context waits on the ticket the first time its
  logits are read, so a caller with two contexts can submit the next pass
  before reading this one. `step` and `prefill` are wrappers on a sequence
  and a context the model keeps, so the CLI, `generate`, `chat` and
  `perplexity` are unchanged. Decode runs the same row-batched graph as
  prefill with one row, which retires the separate decode arena and the
  single-row helpers. `kv-cache` runs two sequences in one pass, one
  decoding over a history while the other prefills, against the two alone,
  and refuses a sequence listed twice; `backend-group` checks the gather.
  Native 18/18, Python 12/12 with both HF models.
- **Done: step 4 gate**, base `56ee2d1`, candidate `18b9c74`, layout
  control `1a05110` perturbing `cpu_backend.hpp` and `arch_qwen.hpp`.
  Three 0.6B cells at 15 pairs, one 8B at 9. Evidence in
  `docs/benchmarks/model-split-20260921/`, raw monitors archived and
  hashed; every sample is kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | **-2.29% / -2.21% / 12/15 FAIL** | -0.54% / -1.08% / 10/15 |
  | 0.6B-1 | decode  | +0.84% / +0.77% / 5/15 | +0.19% / +0.37% / 6/15 |
  | 0.6B-2 | prefill | -0.66% / -0.70% / 8/15 | +0.27% / +0.40% / 5/15 |
  | 0.6B-2 | decode  | +1.35% / +1.18% / 5/15 | +0.66% / +1.00% / 5/15 |
  | 0.6B-3 | prefill | +1.57% / +2.50% / 5/15 | +3.01% / +0.97% / 5/15 |
  | 0.6B-3 | decode  | +3.03% / +1.45% / 2/15 | +1.68% / +1.88% / 6/15 |
  | 8B | prefill | -0.05% / +0.54% / 4/9 | -0.28% / -0.03% / 5/9 |
  | 8B | decode  | +0.32% / +0.44% / 3/9 | +0.27% / +0.22% / 2/9 |

  One cell failed, 0.6B-1 prefill, on the win count at exactly the
  threshold with mean and median inside the band; the two identical cells
  after it came back at -0.66 and +1.57 with 8 and 5 base wins, and 8B
  prefill is flat. Recorded as noise confirmed by rerun, failing cell kept.
  Decode is positive in every cell, which is the number this step could
  have moved: the decode loop now runs the row-batched graph with one row
  rather than its own path, and it did not cost anything measurable.
- **Done: the Vulkan backend page**, `docs/VULKAN.md`, designed against
  the Radeon VII's reported properties: wave64, 8- and 16-bit storage,
  timeline semaphores behind `submit`/`wait`, push descriptors, a 4-byte
  storage offset alignment that closes the `adopt` alignment question, no
  cooperative matrix so matmul is subgroup dots, and the memory types each
  `Memory` kind maps to. Loader loaded at run time so one binary runs
  without Vulkan; GLSL in the tree compiled by `glslc` at build time and
  embedded. Seven sub-steps with a CTest that skips without a device.
- **Done: step 6, placement**, taken ahead of the Vulkan backend because
  its first test needs no device. `Model` takes several backends and a
  `Placement`, a device per tensor role: each layer's attention and
  feed-forward block, the embedding table and the output head. Each weight
  is adopted by the backend that hosts its role; each device that runs
  attention gets a `KVStorage` for exactly its layers with its own pool,
  block size and adopted RoPE tables; a `Sequence` holds a table per
  storage and a ticket per device, and records the model that made it.
  Wherever the placement changes, the residual stream crosses through the
  context's staging vector, a `read` then a `write`, which gives `write`
  its caller. The `placement` CTest splits a two-layer model over two CPU
  backends so both crossings fall inside a layer and requires the bytes of
  one backend for a prompt, decode steps, a history across a block edge, a
  reset and a two-sequence pass; it counts reads and writes so crossings
  are exactly where the placement changes and absent on one device. No
  flag selects a placement yet: with the CPU as the only backend there is
  nothing to place, so `--device` and `--n-gpu-layers` land with Vulkan.
  Native 19/19, Python 12/12 with both HF models.
- **Done: step 6 gate**, base `6514b17`, candidate `3a5aa7b`, layout
  control `c6743a1`. Three 0.6B cells at 15 pairs, one 8B at 9, all eight
  pass. Evidence in `docs/benchmarks/placement-20260921/`, raw monitors
  archived and hashed; every sample kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | +2.92% / +3.09% / 4/15 | +0.44% / +0.61% / 7/15 |
  | 0.6B-1 | decode  | -1.16% / +0.08% / 7/15 | -1.05% / -1.28% / 9/15 |
  | 0.6B-2 | prefill | +2.03% / +1.37% / 3/15 | +0.70% / -0.15% / 10/15 |
  | 0.6B-2 | decode  | +0.96% / +0.55% / 6/15 | +1.08% / +0.46% / 6/15 |
  | 0.6B-3 | prefill | +2.61% / +3.05% / 1/15 | +0.46% / +0.97% / 7/15 |
  | 0.6B-3 | decode  | +0.32% / +0.12% / 6/15 | -0.35% / -1.07% / 8/15 |
  | 8B | prefill | +0.28% / +0.67% / 3/9 | +0.51% / +0.92% / 3/9 |
  | 8B | decode  | -0.21% / -0.43% / 5/9 | +0.04% / +0.22% / 4/9 |

  0.6B prefill is two to three percent up in every cell against a control
  under one percent, with base wins of 4, 3 and 1 of 15. Not claimed as a
  win: the single-device path adds a role lookup per layer and nothing it
  could have saved, and the step 2 gate had the same measurement two
  percent the other way on a change of the same character. It is the
  layout band again, this time in the candidate's favour, and the two
  cancel.
- **Left:** the Vulkan backend, sub-step 1 of `docs/VULKAN.md`. This
  workstation has the loader and `vulkaninfo` but not the SDK: `VULKAN_SDK`
  is unset and there is no `glslc`, so the LunarG SDK has to be installed
  before sub-step 1 can build here. Then the placement flags with it.
- **Gotchas:** `sync()` stays `noexcept`; `wait` is too. The other
  developer's last recorded position predates the last five merges to
  main; the design is posted for review but does not wait on it.

## ROCm on Windows is not available for this hardware (2026-09-21)

Checked before planning any GPU work on the workstation. The Windows HIP SDK
supports RDNA3, RDNA3.5 and RDNA4 only, lists no Instinct card, and states
that it does not support gfx906 (Vega 20). gfx906 entered ROCm maintenance
mode in 5.7 and is deprecated on Linux too, where it still runs but AMD no
longer builds for it.

Consequence for the plan, which it confirms rather than changes: the ROCm
backend is developed and validated on the Linux machine. Windows keeps the
CPU backend. If this workstation ever needs GPU acceleration, the route is
Vulkan, already the roadmap's portability target and supported by AMD's
Windows drivers.

## Device execution model complete (ROADMAP #4a) (2026-09-21)

All six steps are merged. Weights and activations are `Buffer` handles, every
op takes a buffer and a float offset, KV blocks are buffers the backend
allocates, and ops enqueue with one sync per forward pass. The model layer
holds no host address and computes no offset into KV storage. Per-step blocks
are deleted per the checkpoint rule; the plan and the step table live in
[DEVICE-EXECUTION](DEVICE-EXECUTION.md) and every gate's evidence is under
`docs/benchmarks/`.

What a vendor backend supplies: `alloc`, `adopt`, `read` and `copy` over its
own allocator, `kv_alloc` and a `KVStorage` it owns, the ops, and a `sync()`
that means it. What it does not have to invent is a place for activations or
a cache layout.

Three things are worth carrying forward rather than rediscovering.

- **`sync()` is `noexcept` by contract.** Three of its four callers are
  exception paths releasing KV blocks, so a sync that threw would replace the
  error that got there and leave the sequence half-released. A backend that
  cannot establish that its work finished must fail hard.
- **A few percent of Qwen3-0.6B prefill means nothing on this tree without a
  control.** The same measurement moved -3 points during the code read and +6
  at step 6, both times from code placement rather than work. AGENTS.md
  carries the rule and the evidence; the short version is that the control
  has to perturb the same file the change does.
- **`Backend::write` was deleted** after step 5 declined to give it a caller.
  Weights arrive through `adopt` and every other value is produced by an op.
  The first backend that genuinely needs a host-to-device write adds it back
  alongside that caller.


## Full code read before the first vendor backend (2026-09-21)

- **Goal:** read every line of `src/`, `tests/` and `tools/` before starting a
  GPU backend, fix what is actually broken, record the rest. Steps 1 to 4 of
  the device execution migration had landed and it was the right point to stop
  and look at the whole tree.
- **Done:** the read. `src/model/kv_cache.hpp` is the only file with nothing
  to report.
- **Done: two real bugs.** `f32_to_f16` OR-ed the rounding carry into the
  exponent field instead of adding it, so any value whose mantissa rounded up
  out of ten bits came back exactly half the right size whenever the exponent
  was odd; 1.999756 encoded to 1.0 rather than 2.0. It reached only
  `llmx quantize`, where it scales a whole block, and nothing had ever
  compared that direction against a reference. `tests/fp16.cpp` now does,
  using binary16 as its own oracle over the whole finite range. Separately,
  `bpe::Tokenizer::decode` indexed the vocabulary with an unchecked id, which
  `llmx detokenize <id>` passes straight from the command line.
- **Done: two hangs and a silent truncation reachable from a model file.**
  A chat template calling `replace`, `count` or `split` with an empty needle
  looped forever, because each advanced its cursor by the needle's length.
  The template is GGUF metadata, so that was untrusted input reaching an
  unkillable loop. A model declaring no EOS id had token zero, an ordinary
  token, treated as the stop token; and a reasoning-start marker with no
  matching end marker suppressed the entire reply.
- **Done: three tests asserted less than they claimed.** Two were coverage
  this project lost to its own activation-arena change and did not notice:
  `prefill-scope` counted 64- and 96-byte allocations that stopped existing
  when nine vectors became one arena, and `kv-cache` aimed an injected
  allocation failure at "the second large batch buffer" for the same reason.
  The third predates it: the tokenizer fixture typed its special token 2,
  which is GGUF's *unknown* rather than *control*, so the special-token path
  its docstring advertises was never exercised, and its unicode case checked
  only an exit code.
- **Done:** the vocabulary vector was zero filled twice per forward pass with
  a `read` overwriting all of it immediately after, an extra 608 KB per
  decoded token on Qwen3-8B; comments across the backend, the CPU backend and
  the model still described the pre-migration interface; `ARCHITECTURE`,
  `ROADMAP`, `DEVICE-EXECUTION` and the backend page said the interface takes
  raw host pointers, which stopped being true at step 4.
- **Done: the gate failed, and the cause was code layout, not the change.**
  Worth reading before the next few-percent argument. The first 0.6B prefill
  cell came in at -4.58% and failed the advance rule. Reruns gave -0.81,
  -2.52, -1.92 and -2.33, so two thirds failed and it was clearly not simple
  noise. A layout control built by appending an unused function to
  `cpu_backend.hpp` moved the same number by only -1.16/+0.63/-1.13/+0.33,
  which made the candidate look like a real regression sitting outside the
  band.

  Bisecting the seven commits found it. The f16 commit alone was clean at
  -0.56% mean. Adding the next commit, a four-line bounds check inside
  `Tokenizer::decode`, produced -3.13/-2.17/-3.09. `Model::prefill` never
  calls `decode`; the prefill timer cannot execute one instruction of it. The
  entire measured difference came from where those four lines pushed the code
  that follows them in the single translation unit.

  Two things follow. The check is now `vocab.at(id)`, one token instead of
  four lines, with the readable message at the CLI call site where the caller
  knows the vocabulary size. And the control has to perturb the same file as
  the change: appending to a different header understated the band by more
  than a factor of two. AGENTS.md carries this now; cells are `06-layout*`,
  `06-fp16*`, `06-tok*`, `06-pre*` and `06-mid*` under
  `docs/benchmarks/code-read-20260921/`.
- **Merged with one performance cell unresolved.** Recorded so nobody later
  reads this as four green cells. Correctness was green throughout: native
  18/18, Python 12/12 with both HF models, 260-token logits byte-identical.
  0.6B prefill came in at -3.87/-1.59/-3.71 across three runs, failing the
  advance rule twice, while 8B prefill gained 3.56% and both decode cells
  were flat. The bisection above is why it merged anyway: the loss tracks
  where the compiler places the code, not work the change added, and the
  clearest single case is a four-line guard in a function that prefill never
  calls moving the same number by three points. If a later step finds 0.6B
  prefill about three points low against an older baseline, this is where it
  went.
- **Left:** nothing blocking.
- **Gotchas:** `Backend::write` and `Backend::copy` are implemented and have
  no caller anywhere, which AGENTS.md forbids. They stay only because step 5
  is their consumer and is next; if step 5 does not use `write`, delete it
  there. `Backend::rope` is the last op taking raw host pointers, and its only
  caller outside the CPU backend is `bench`, which therefore measures a
  function the runtime never calls; `norm_rope_rows` is what the model runs
  and it still takes raw `cos`/`sin` pointers. Moving both is a measured
  change and belongs in its own gated commit, not this one.

  Known and deliberately not fixed: `strip_ws` in the template renderer eats a
  leading unary minus, so `{{ -1 }}` renders `1`; `{% for k, v in x %}` binds
  one variable literally named "k, v" rather than unpacking; `quant/` includes
  `format/gguf.hpp`, which reaches up one layer and is already recorded in
  ARCHITECTURE as a known exception; `quantize_row_q4_1` has no production
  path, since the CLI writes only q8_0 and q4_0; `metadata_u64`,
  `Tokenizer::token_id` and `pad_id` have no callers.

## Code layout moves this benchmark more than the rule allows (2026-09-21)

Moving the embedding gather into the backend measured -8.03% on 0.6B prefill,
twice. The gather itself measures 0.0014 ms at that shape and the logits are
byte-identical, so the work cannot account for 38 ms. Builds that behave
identically, 15 pairs each, against one reference build:

| Build | Prefill | Decode |
|---|---:|---:|
| baseline + unused function | -1.34% | -1.35% |
| baseline + small unused function | -0.50% | -1.18% |
| baseline + larger unused function | +2.52% | -2.00% |
| embed as written | -8.03% | -0.84% |
| embed + that function as dead code | -1.89% | +0.42% |
| embed, gather forced out of line | -3.75% | -0.90% |
| embed, prefill body forced out of line | -4.36% | -1.89% |
| embed, token id in a member slot | -0.60% | -2.21% |

Three things follow, and the second one matters most.

**Layout alone spans about four points of prefill**, wider than the runner's
3% band. The first probe, an unused function and nothing else, **failed the
advance rule** on decode at 12 of 15 baseline wins. The rule can therefore
report a regression for a relink.

**A single-build A/B cannot gate a few-percent change on this cell.** A
candidate needs comparing against several perturbed builds of the same
behaviour, not one. Earlier few-percent prefill conclusions on this model
carry that uncertainty, as does the 0.6B prefill cell's own A/A failure at
-3.53%.

**The -8% build was still real.** It sits outside the layout band, so it was
a genuinely poor layout, not noise: one extra local in a very large inline
body was enough. Forcing functions out of line made it worse; moving the
local to a member slot restored it to -0.60%, inside the band. That is the
shipped form, with the reason in the code.

Root cause is structural. The whole runtime is one translation unit of
headers and the forward pass is one enormous inline body, so any edit can
reshuffle it. Splitting the model layer into its own translation unit would
bound this; AGENTS already allows `.cpp` files with one per logical unit.
Evidence in `docs/benchmarks/layout-sensitivity-20260921/`.

**Then the confound in my own comparison.** The build embeds the Git
revision and a dirty marker. The candidate had been built from a dirty tree
and the baseline from a clean worktree, so the two binaries differed in an
embedded string as well as in code. Rebuilding both the same way, from
detached worktrees at their two commits, the same change reads:

| Cell | Paired mean | Paired median | Baseline wins |
|---|---:|---:|---:|
| 0.6B prefill | +1.09% | -0.08% | 8/15 |
| 0.6B decode | +0.13% | -1.37% | 8/15 |
| 8B prefill | +0.35% | +0.83% | 3/9 |
| 8B decode | +0.51% | +0.46% | 4/9 |

Dead level, against -8.03% when the arms also differed in build identity.
The rule for every comparison from here: **build both arms the same way**,
same source of the version string and same tree state, so the only
difference is the change. An embedded string is enough to move this
benchmark by several points.

## Matched mx gate on a quiet machine (2026-09-21)

The comparison repeated under the rule below, with recorded system CPU of
39% on 8B and 43% on 0.6B against the benchmark's own 37.5%. Eight pairs per
model, alternating arms, llmx from `0c0dec7`, mx `5542318e74`. Evidence in
`docs/benchmarks/kv-mx-quiet-20260921/`.

| Model | Phase | llmx | mx | Paired median | llmx wins |
|---|---|---:|---:|---:|---:|
| 0.6B Q8_0 | prefill | 524.57 | 277.54 | +85.64% | 8/8 |
| 0.6B Q8_0 | decode | 49.09 | 47.77 | +3.91% | 7/8 |
| 8B Q8_0 | prefill | 40.77 | 22.16 | +84.54% | 8/8 |
| 8B Q8_0 | decode | 4.64 | 4.67 | -1.53% | 3/8 |

Three of four cells clear the floor, prefill by a wide margin on both models.
8B decode is 1.53% under it, against a harness A/A of 0.06% on that cell, so
the sign is outside harness noise; but llmx wins 3 of 8 pairs and the
per-pair spread runs -7.5% to +1.2%, so the size is not well determined. The
earlier figure of -3.5% came from the loaded run and overstated it.

This is the current state of ROADMAP #8's performance gate: met everywhere
except 8B decode, which is under by roughly one and a half percent.

## The matched mx comparison ran under background load (2026-09-20 run)

`tools/compare_cpu.py` passes its own A/A: one binary as both arms moves at
most 0.80%, and 0.06% on the 8B decode cell. The harness is sound.

**Corrected cause.** This block first blamed memory contention between the
two arms, because llmx copies the model into the heap while mx maps it. That
was a hypothesis and it is wrong. Two things refute it. The same depression
appears on Qwen3-0.6B, where both processes together need about 1.2 GB on a
32 GB machine, so there is no pressure to have. And the recorded monitors
give the real answer: the comparison ran at 55% mean system CPU against the
benchmark's own 37.5%, while the later A/A runs sat at 36 to 40%. Roughly a
fifth of the machine was doing something else during the comparison, which
is why both arms were slow. Medians, tok/s:

| Condition | 0.6B pp | 0.6B tg | 8B pp | 8B tg |
|---|---:|---:|---:|---:|
| llmx alternating with mx | 464.92 | 43.56 | 37.34 | 3.89 |
| mx alternating with llmx | 259.21 | 41.62 | 20.58 | 4.02 |
| llmx alternating with itself | 548.42 | 50.94 | 41.49 | 4.73 |
| mx alternating with itself | 278.13 | 49.64 | 21.45 | 4.78 |

Both binaries lose about a fifth under that load, llmx 3.89 against 4.73 and
mx 4.02 against 4.78, and no drift appears across pairs, so it is not heat
building up. Pairing is what makes the comparison survive this: both arms
alternate inside the same conditions, so the relative result holds even
though the absolute rates do not represent a quiet machine. The 8B decode
deficit reads -3.2% paired under load against -1.0% unpaired when quiet, and
the unpaired figure compares two runs three hours apart, which is the
comparison this project has been burned by. The honest statement is a
deficit of roughly 1% to 3.5%, still under the floor, smaller than first
reported, and not caused by the KV work.

Prefill is unaffected: llmx is roughly double mx in every condition measured.

What this changes for method: the runner already records activity, but
nothing acted on it. A comparison worth publishing should be repeated when
the recorded system CPU is near the benchmark's own share. Evidence in
`docs/benchmarks/mx-harness-aa-20260921/`.

## A/A calibration of the A/B runner (2026-09-20)

Run after the screening and sampler results below, not before them, which is
the wrong order and is why this block exists. The same binary published as
both arms, `tools/ab_runner.py`, 247-token prompt, 6 threads, evidence in
`docs/benchmarks/aa-calibration-20260920/`.

| Model | Phase | Paired mean | Paired median | Baseline wins | Per-pair spread |
|---|---|---:|---:|---:|---|
| 0.6B Q8_0 | prefill | -2.04% | -3.53% | 7/9 | -8.09% to +8.96% |
| 0.6B Q8_0 | decode | -0.03% | +0.30% | 4/9 | -5.94% to +3.03% |
| 8B Q8_0 | prefill | -0.00% | +0.07% | 2/5 | -4.49% to +2.57% |
| 8B Q8_0 | decode | +1.93% | +2.27% | 2/5 | -2.24% to +5.67% |

**0.6B prefill fails its own A/A**: identical code reports a 3.53% median
loss, past the runner's 3% band. 8B decode moves 2.27% on identical code.
So on this machine today, a single cell below about 4% in those two places
is not evidence either way. What survives: 0.6B decode, where the A/A holds
to 0.30% median.

Consequences, applied to the blocks below rather than left for a reader to
work out:

- The sampler's 0.6B decode gain, +32.04% with 0 of 9 baseline wins, is far
  outside this band and stands.
- The sampler's prefill cells, +2.79% and +2.17%, are inside the band and
  are **withdrawn**; the change has no measured prefill effect.
- The sampler's 8B decode cell was already reported as unresolved and stays
  so.
- In the KV screening, per-cell prefill differences under about 4% carry no
  weight. The rejection of 64 rested on five of eight plans losing prefill,
  two of them by 6.5% and 7.8%, which is a pattern rather than one cell, but
  it is weaker evidence than that table implied. The choice of 128 over 256
  rests on allocated-versus-used bytes, which is counted rather than timed
  and is unaffected.
- The matched mx comparison uses a different harness
  (`tools/compare_cpu.py`, in-process timing) which has **not** been A/A
  calibrated. Its 8B decode cell of -3.5% should be treated as provisional
  until it is.

## Greedy sampling cost (2026-09-20)

Greedy sampling sorted the whole 151936-token vocabulary before reading one
element. `sample` now takes a linear maximum at temperature zero and a
partial sort over the top-k window above it; the repetition penalty is read
through a lambda instead of materialized. Ties now take the lowest token id,
where the unstable sort left them unspecified.

| Paired A/B, generate --temp 0, 247-token prompt, 6 threads | Prefill | Decode |
|---|---:|---:|
| Qwen3-0.6B Q8_0, 9 pairs | +2.79% | +32.04% |
| Qwen3-8B Q8_0, 5 pairs | +2.17% | -0.72% |

The saved work is a constant per token, about 7 ms, so it dominates a 0.6B
token and is inside the runner's noise band on 8B at five pairs, where it is
reported as unresolved rather than as a win. Evidence in
`docs/benchmarks/sampler-greedy-20260920/`. The matched mx gate is unaffected:
that harness times model inference only and excludes sampling.

## Paged KV cache design (2026-09-20)

- **Goal:** replace the single-sequence contiguous `HostKVCache` with a paged
  cache whose logical view (per-sequence block table, positions, refcounts) is
  backend-neutral and whose physical block size and layout are backend-owned,
  so the multi-user server and GPU backends do not inherit a CPU layout.
- **Done:** isolated cold, shuffled paging microbenchmark, seven paired
  repeats, two geometries, three lengths: block 16 costs +25-33% on decode
  attention, 128 costs +5-10%, 256 costs +1-3%. Design in
  [KV-CACHE](KV-CACHE.md), opened for review by the second developer.
- **Done (step 1):** `model/kv_cache.hpp` (`BlockPool`, `KVSequence`),
  backend `kv_layout`/`kv_alloc`/`kv_write` and view-form `attention`,
  `CpuKVStorage` backed on demand, `HostKVCache` and the raw-pointer
  attention removed. The review findings are folded in: each step and each
  prompt is one transaction, bookkeeping vectors are reserved so failure
  paths never allocate, pool and sequence own their ids (non-copyable,
  release on destruction), retain rejects free ids, growth copies into
  exact-size buffers with retained and peak bytes reported, the budget
  crosses the seam in tokens, arithmetic is checked. `generate --verbose`
  prints `kv: allocated/peak/used`. Native suite 17/17; Python suite with
  both HF models passes; main and paged logits byte-identical on 0.6B and
  8B (11 to 841 tokens) and greedy text identical. Provisional block size
  128 behind the temporary `LLMX_KV_BLOCK` knob.
- **Done (screening):** 64/128/256 against contiguous main on 0.6B and 8B
  Q8_0, eight frozen plans, all samples kept, tables in
  [KV-CACHE](KV-CACHE.md). Block fixed at 128; the knob is deleted.
- **Done (mx gate):** matched comparison on the paged runtime, tables in
  [KV-CACHE](KV-CACHE.md). Prefill +79.8% (0.6B) and +82.7% (8B) over mx,
  8/8 pairs; decode -0.2% and -3.5%. The 8B decode cell is under the floor
  and is the existing decode bandwidth item, not a paging cost.
- **Left:** re-review of the final revision; a decision on the 8B
  decode cell; then main integration. Fork/COW and device buffers are later
  steps. F16 KV is out of scope.
- **Gotchas:** the microbenchmark is isolated attention with a cold cache and
  is not an end-to-end decode cost. Memory waste cuts against large blocks:
  224 KiB per token on 0.6B means a partial 256-token tail wastes up to
  55.8 MiB per sequence, 27.8 MiB at 128.

## Native HF download checkpoint (2026-09-20)

ROADMAP #9a is implemented: native `llmx pull`, immutable revision resolution,
verified cache, `HF_TOKEN` credentials, bounded parallel downloads (four streams
by default) and aggregate sharded GGUF loading. This checkpoint is locally
validated and prepared for main publication; hosted CI is still pending.
HF native safetensors/tokenizer/config support and device kernels remain planned.

| Local check | Windows | Linux |
|---|---:|---:|
| Native suite | 16/16 | 15/15 with UBSan |
| Required-HF Python components | 12/12 | 12/12 with UBSan |
| Shard/Unicode fixtures | 66/66 | 66/66 with ASan/UBSan |
| Transport fixtures | 58/58 | 58/58 |
| Sharded HF full-logit cases | 20/20 | 20/20 |

The real 639,446,688-byte public Q8 download passes independent SHA256 after
four-stream assembly. Cache reuse passes on Windows and Linux. Five measured
loader pairs average 427.246 ms before and 428.494 ms after; paired speed change
is -0.283% with a descriptive interval [-3.706%, +3.140%]. This tiny observed
cost is unresolved; all activity-flagged samples are retained. Complete loaded
payload/tensor/metadata hashes match in all 12 processes. No inference arithmetic
changed, and this is not a fresh mx inference comparison. Full evidence and
limitations are in the [checkpoint](ASSETS.md#native-hf-pull-checkpoint-2026-09-20).
All 28 Markdown files were reviewed. Live gated-repository use and controlled
credential redirects remain untested; native credential fixtures and public
HTTPS pass. A maintained curl 8.4+ supplies HTTPS and redirect behavior.

## Prefill placement checkpoint (2026-09-20)

Backend-owned prefill placement is merged and published on both main remotes
at `3c5d4b9`, under the performance tradeoff policy. All five hosted jobs pass in
[run 35516912422](https://github.com/mxxm-t/llmx/actions/runs/35516912422): Windows,
macOS Intel, Linux, Linux UBSan and required HF. The new native targets are
included (12 Windows, 11 Linux/macOS).
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
timings do not supply new performance measurements. The Linux machine's HF container
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
| Paged KV cache (block pool, backend-owned blocks) | Done |
| Device execution model (ROADMAP #4a)     | Done     |
| Execution model: tickets, batched views, placement (`docs/EXECUTION.md`) | Done: steps 1 to 7, step 7 being the server, see the server row; `--device` lists select a layer split (multi-device row) |
| KV cache fork (KV-CACHE step 2)          | Done     |
| Multi-device split (per-layer, per-tensor) | In progress (`docs/MULTI-DEVICE.md`): phase 0 measured and merged; phase 1, the layer split over a `--device` list fitted to free memory, on its branch; pipelining, the scheduler over passes in flight and tensor groups follow |
| GPU backends (Vulkan first to write, ROCm first-class) | Vulkan implemented and the recorded dense-model device gate passed on both platforms (forty-seventh checkpoint above): Radeon VII decode 102-115% and prefill 109-455% of the same-card reference Vulkan build; one MI50 decode 102-115% and prefill 102-267%. These are dated gate results, not new measurements from this documentation review. ROCm planned |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Done (`docs/SERVER.md` steps 1 to 6): `llmx serve`, correctness gates pass on both backends, throughput 109 to 125 percent of the reference server at 1 to 16 concurrent on the device (short of the wide margin `docs/SERVER.md` gates on), prefix reuse through fork, a second execution context measured to have nothing to hide, the OpenAI-compatible routes |
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
| Backend-owned prefill placement | Done (main `3c5d4b9`, five hosted jobs green) |
| CLI thread settings                    | Done |
| Automatic build identification          | Done (main `9511a4a`) |
| Focused CLI help and complete current option coverage | Done (2026-09-24 checkpoint) |
| Live generation and loading progress     | Done |
| GitHub CPU CI                          | Done     |
| HF fixture download retries and CI cache | Done |
| Hosted numeric/path portability repair | Done (five jobs green at `851d375`) |
| HF model download and sharded GGUF (ROADMAP #9a) | Done (local gates pass; hosted CI pending publication) |
| HF native formats (ROADMAP #9b)          | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

`Done` denotes implemented and validated functionality in this release tree.
The earlier runtime base `08351b0` was published on both main remotes. Its initial five-check
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

### Vulkan prefill through the 8-bit integer dot (2026-09-22)

- **Goal:** prompt processing on the MI50 at least level with the reference, where it was 44 to 79 percent against a reference split across ten cards and is 24 to 73 percent against one (the thirty-fourth paragraph, measured after the tile below). Measured on that card within one environment at a 4096 x 14336 projection over 512 rows, our float tile reads 4.87 TFLOPS for Q8_0, 4.65 for Q4_K and 3.62 for Q6_K, level with the reference's own float tile at 4.77, while its 8-bit integer-dot tile reads 13.30, 11.42 and 7.03. So the gap is that path, not scheduling or tile shape (STATUS, thirty-third paragraph above).
- **Plan:** a kernel quantizes each activation column to 8-bit values per 32-value block with the block's scale and scaled sum; a tile kernel stages one quant block per row and column per step as packed 8-bit words and scales, and multiplies with the four-wide integer dot, one float multiply-add per block for the scale and one more for a type's minimum. Q8_0 and Q4_K first, then Q6_K and Q5_K. Used only where the device's integer dot is native, which the profile records as `prefer_integer_dot`; elsewhere the float tile stays.
- **Done:** the diagnosis above, and `backend-vulkan` now times the tile at that shape in TFLOPS. `quantize_x8.comp` and `matmul_tile_q.comp` for Q8_0 and Q4_K (`cb2eb5b`), taken where the profile says `prefer_integer_dot`. On the MI50 at the 8B feed-forward shape Q8_0 goes 4.87 to 7.72 TFLOPS and Q4_K 4.65 to 11.48, the reference's being 13.30 and 11.42. Qwen3-8B-Q4_K_M prompt processing 297.8 to 488.7 tok/s at 512 rows. Correctness: every HF perplexity cell in both scoring modes on the MI50 with all three 0.6B fixtures, the backend test's 1,172,518 outputs with its reference rounded the same way, and on the 8B Q4_K_M file, which no fixture covers, 40 wikitext windows of 512 at mean NLL 2.47005 against the float tile's 2.47023. Scoring through batched passes (`fc261f9`) is what made the HF gate reach this path at all. Then the thirty-fifth to thirty-seventh paragraphs: Q8_0 staging a word at a time (12.07 TFLOPS), Q6_K in its own module (9.60 against the reference's 7.03) and Q5_K through the tile (`c1bdb09`, `db249b8`); Q4_0 and Q4_1 through it too and the measured profile carrying four thresholds, the MI50's row 16, 32, 24 and 40 (`2b770f6`); and a third tile height of 32 rows for short prompts, 0.6B Q8_0 at 64 rows 2502 tok/s, 54 percent of the reference's 4638 (`33933a9`).
- **Left:** staging several quant blocks per barrier (being measured); MI50 prompt processing at 69 to 89 percent at 247 rows and more; MI50 decode at 86 to 91 percent on the 8-bit files.
- **Gotchas:** 8-bit activations cost 0.009 of NLL in the decode kernel earlier, close to the 0.010 bound on one HF cell, so the device suite on the MI50 decides whether this ships, per type. The AMD Windows driver lowers the integer dot extension to widened multiplies, so the Radeon VII must keep the float tile.

### External floor of merged main (2026-09-20)

Matched against mx `5542318e74`, six threads, eight rounds, identical committed
token IDs in both arms. Runtime is `main` after the device-execution steps and
the fused Q5_K/Q6_K decode dots; the quantized-activation kernels are NOT in it.

| Model / phase | llmx | mx | ratio | |
|---|---:|---:|---:|---|
| Q8_0 prefill | 495.23 | 271.75 | 1.82x | above |
| Q8_0 decode | 46.01 | 44.94 | 1.02x | above |
| Q5_K_M prefill | 408.70 | 242.41 | 1.69x | above |
| Q5_K_M decode | 34.92 | 60.93 | 0.57x | **below** |

Three of four clear the floor, and Q8_0 clears both phases: decode crossed from
0.99x earlier today to 1.02x. **ROADMAP #8 is met for Q8_0 and NOT met
overall** - K-quant decode is the single remaining failure, with mx 1.75x
faster. Absolute rates are higher than this afternoon on both arms because host
load fell; the ratios barely moved (Q5_K decode 0.589x then, 0.573x here).
The 1.02x is inside the range an A/A can produce and should not be leaned on;
the 1.69x-1.82x margins and the 0.57x shortfall are not.
Evidence: `benchmarks/main-external-floor-20260920.json`.


### Scoped correctness coverage and remaining HF work

- **Goal:** keep independent HF ground truth and extend coverage where the roadmap requires it.
- **Done:** exact tokenizer fixtures; tiny tied/untied F32 full logits and NLL; real Q8/Q4 ranking and excerpt PPL; HF/Jinja2 follow-up chat fixtures; pinned reference generation and strict consumers. The unchanged real 8B consumer previously passed 37/37 on Windows and Linux with frozen bounds. Real 0.6B F32/Q8 1,943-token plus 32-step continuation checks are archived in ASSETS.
- **Left:** broader full-corpus, maximum-context and per-layer references, plus prospective numerical bounds for any new lossy kernels. Short 8B rankings/excerpts are not deep-context validation.
- **Gotchas:** self-consistency is supplementary. Exact comparison against another llmx path cannot replace HF. Model construction validation does not establish finite weights, arbitrary token-ID safety, request budgets or failed-session recovery.

### K-quant and device execution work (separate developer branch)

- **Goal:** improve the remaining K-quant decode path and continue ROADMAP #4a without overlapping this release/placement work.
- **Done:** the separate `design/device-execution-model` branch and the K-quant experiments are not incorporated by this release. Their measurements and source identities must be reviewed before adoption. Quantized-activation commits `357d68d` and `97d52e8` fail `backend-group`; they remain isolated and are not merge-ready. Arithmetic-preserving dispatch work is being separated onto a passing base.
- **Left:** prospective correctness/performance validation for any new quantized-activation path. The device execution model is complete on main (see the 2026-09-21 block above); K-quant decode remains. Coordinate rebases and announce timing reservations.
- **Gotchas:** earlier grouped Q16 failed the unchanged native double-dot accuracy contract. Do not reuse it as a lossless baseline or weaken bounds after observing results. CPU Q8 results do not establish K-quant parity.

## Working rules and ownership

One developer owns native HF download/cache and sharded GGUF loading; the other owns its
separate K-quant/device work. Coordination happens in a shared log outside
this repository. Builds and tests
may run in parallel when no timing reservation is active. Keep every planned
performance sample, record ordinary machine activity, and report missing
telemetry honestly. GitHub receives main only; feature work stays on its branch until its gates pass.

## Historical feature blocks (2026-09-19 to 2026-09-21)

The blocks from here to the end are checkpoints of features that have since
shipped or been closed; the status table above is the current state. Their
Left lines, such as merging with the runtime stack or closing an external
floor, record what was open at the time and are not current work. The two
exceptions are the blocks whose table rows are still `In Progress`,
"Correctness baseline vs HF reference" and "Performance floor vs
mx-llama.cpp".

### GGUF reader size and tensor extent validation

- **Goal:** reject malformed lengths, dimensions, arithmetic overflow and tensor
  extents before allocating payload storage or reporting loading progress;
  honor the file's declared alignment. This closes the documented format-layer
  error-handling gap and supports future Hub/sharded-format work.
- **Done:** bounded reads, checked size arithmetic and subtraction-based file
  ranges reject malformed input before payload allocation/progress. Reader and
  writer honor positive uint32 alignments divisible by eight, including 24.
  Quantized row widths must contain whole blocks. Array depth is limited to
  256 and tensor rank to four; valid empty tensors retain mathematical size zero.
  Isolated branch `fix/gguf-tensor-extents` starts at `5859762`; placement
  experiments and fixed binaries remain separate.
- **Done:** 122 GGUF cases pass on Windows/Linux: independently constructed
  fixtures plus the writer-alignment round trip. All nine
  native tests and all eleven required-HF suite components pass on both.
  Current-reader ASan+UBSan passes both format/progress tests. The initial full
  Windows build lacked the MSVC include environment; that failure is retained
  and the complete run passes after initializing vcvars64. No source workaround.
- **Done:** the pinned 8B file loads completely with byte-identical `info`
  output against the prior validated control. All six fresh short HF cases
  match top-1 and top-5 overlap 5/5. This is not a new 8B NLL/long-context gate.
  All jobs are terminal. All 25 project Markdown files reviewed and stale
  `info`, quantized-row and loader validation descriptions corrected.
- **Left:** merge with the runtime stack once its separate performance gate
  passes; the root executable and main/GitHub remain unchanged. No inference
  hot path changed and no new performance result is claimed. Evidence:
  [`gguf-reader-validation-20260920.json`](benchmarks/gguf-reader-validation-20260920.json).
- **Gotchas:** token IDs, numerical weight contents, writer validation and future
  request recovery remain separate validation work. The later Qwen construction
  block checks configuration geometry and required tensor layouts.
  JSON conversion dimensions are covered by the later block
  above. Do not
  claim that file-extent checks make arbitrary models executable. Preserve
  nested GGUF arrays within a documented depth limit and valid non-power-of-two
  alignments that are multiples of eight.

### Prefill placement reassessment with machine activity monitoring

- **Method clarification after review:** the frozen runner rotates and
  reverses the three arm orders inside each model/workload block; model and
  workload order also rotate across one warmup and eight measured rounds.
  Report mean/median rates and elapsed time, paired ranges, sample deviation
  and approximate paired 95% intervals, retaining all originals. A third
  contaminated block ends the study as inconclusive; it is not accepted.
  Benchmark and recorder CPU are excluded by PID plus creation time. All arms
  use the same model path/hash for a given model. Activity telemetry does not
  measure DRAM bandwidth or prove absence of short/inaccessible activity;
  sub-percent observer effects also remain unresolved. No clean-preflight
  result alone establishes that a small performance difference is real.
- **Goal:** complete the reopened whole-prefill placement assessment against
  production and matched mx, including HF/lossless and short follow-ups.
- **Done:** preserved the historical candidate and its failed original screen;
  JSON checkpoint `a61c414` passes Windows/Linux correctness suites.
- **Done:** fresh scratch control and default-enabled candidate retain current
  JSON/CLI fixes and pass 8/8 Windows CTests each. The candidate full required-HF
  suite passes 11/11; timings are diagnostic only. Rebuilt callback contracts
  pass 17 cases/4,626 exact values and Windows lifecycle checks pass 27
  cases/1,176 exact values. All 28 active-prefill HF logit cases pass: ten tiny
  F32, six real 0.6B Q8, six real 0.6B F32 and six 8B Q8. Every candidate
  process verifies six applies/restores on distinct target CPUs, no errors or
  leftover restriction, and exact printed logits against current production.
  Tiny F32 maximum HF error is 6.991024018e-7 against the unchanged 2e-5 bound;
  all 18 real-model cases match top-1 and all five top-5 IDs.
- **Done:** the active 1,943-token prefill plus 32 forced continuation steps
  passes for real 0.6B F32 and Q8. Each model's 33 full vectors (5,013,888
  finite floats per arm) are byte-identical to current production, with exact
  per-target NLL. Both candidate processes verify six applies/restores and
  zero decode setters. All 36 HF/prompt/continuation inputs match the earlier
  committed evidence before execution. F32 maximum HF logit error is
  0.000126362 <= 0.001; absolute mean continuation NLL differences are
  0.000000645211 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8).
- **Done:** `tools/monitor_windows.py` records timestamped system CPU/disk/GPU
  counters and per-process CPU deltas keyed by PID plus creation time. The
  24-sample controlled-load check detects the known CPU process at a median
  99.995% of one logical CPU; recorder CPU is 0.53125 s over 24.01487 s,
  including initialization. Intervals and query errors remain in the log.
- **Left:** complete prospectively planned matched comparisons and check observer
  effects once a quiet measurement window is available. The driver records
  machine activity before and throughout every matched block; no model timing
  has yet passed its preflight screen.
- **Done:** fresh Linux candidate pass-through build, native 8/8 and full
  required-HF suite 11/11 pass with unchanged snapshot source. Matched mx
  primary/one-token/nine-token continuation harness builds against the pinned
  CPU DLLs. The activity evaluator passes 20 synthetic/known-load checks.
- **In progress:** prospective three-arm comparison plan uses one outer
  warmup round plus eight measured rounds per model/workload, with complete
  matched-block replacement for detected contention (at most two replacements).
  The first preflight defers before any model launch: accessible unrelated CPU
  is 23-44% of one logical CPU and physical disk busy is 29-41% across eleven
  samples. Raw logs are retained; all 56 frozen identities recheck unchanged.
  No performance result or candidate rejection follows. Small observer effects
  remain an explicit unresolved limit.
- **Done:** fresh enabled-candidate optional 8B regression passes 37/37 checks
  against unchanged HF fixture bounds. This includes serial-step NLL coverage;
  active placement is established by the separate callback witnesses, not by
  the serial NLL path. Its timings are not performance evidence.
- **Done:** primary, one-token and nine-token follow-up correctness passes
  for both Q8 models: 12 processes, six byte-identical full-vector pairs and
  911,616 finite floats. All 20 callbacks independently verify six distinct
  physical cores and full restoration (120 applies and 120 restores total),
  with zero decode setters, placement errors or leftover restrictions.
  These fixed-token prefix/suffix checks do not replace interactive chat tests.
- **Checkpoint:** all 25 project Markdown files reviewed; stale status wording
  corrected. Supplemental commands, sources and results are archived in
  [`prefill-contention-followup-20260920.json`](benchmarks/prefill-contention-followup-20260920.json).
  All correctness jobs are terminal. Timing remains deferred; preserve the
  first preflight and use a fresh output directory for the next comparison.
- **Latest preflight:** the next attempt also defers before launching a model:
  sustained unrelated CPU exceeds the unchanged screen. Recorder exit is zero;
  all 58 identities recheck unchanged. A minimally changed runner now requires
  a fresh `--output` path, preserving both attempts and the original runner.
  [Second preflight evidence](benchmarks/prefill-preflight-02-20260920.json).
  Turning off the only recorder would remove during-run contention evidence;
  an extra-recorder sensitivity diagnostic would not prove zero-recorder cost.
  Keep that limit explicit and assess whether such a diagnostic is useful after
  the matched comparison, without creating more measurement infrastructure now.
- **Gotchas:** activity monitoring is evidence, not proof of no interference.
  Keep observer overhead and unavailable counters explicit; no automatic
  adoption or retroactive noise claim follows from the policy change.
  The local check retains 124-126 inaccessible processes as unknown; system
  counters remain available. Controlled disk/GPU saturation and benchmark
  timing perturbation are untested. This continuation check is not full-corpus,
  maximum-context or 8B long-context HF coverage. Production runtime source
  remains unchanged; the candidate is still scratch-only.
  Full 25-file Markdown review and saved checkpoint evidence:
  [`prefill-reassessment-correctness-20260920.json`](benchmarks/prefill-reassessment-correctness-20260920.json).

### JSON validation, Unicode decoding and output escaping

- **Goal:** fix current model-description JSON handling and the documented
  ROADMAP #9b prerequisite for HF metadata/safetensors, without dependencies.
- **Done:** strict number/literal/escape syntax, classic-locale finite-double
  conversion, validated UTF-8 and UTF-16 surrogate-pair decoding, and a
  256-container nesting bound. Value/API and duplicate get-first behavior stay
  unchanged. A small string quoting helper now escapes both dequantize path
  and tensor-name fields, including quotes, backslashes and control bytes.
- **Done:** final Windows/Linux builds and native tests pass 8/8; both full
  required-HF Python suites pass 11/11. Native JSON has 1,033 checks, including
  independent expected bytes, numeric/locale limits and malformed input.
  Linux ASan+UBSan passes the same checks. Actual Q8/Q4 CLI round trips retain
  BMP/supplementary Unicode, quotes, backslashes, newline and source paths.
  Old parser/CLI regressions fail as expected. The initial new-reader Windows
  suite exposed the existing output-escaping bug; its failed result remains
  archived alongside successful final runs. A missing temporary Linux build
  directory caused one later launch to fail before executing any build/test;
  final Linux validation uses a persistent owned build directory.
- **Left:** include the validated fix with the runtime stack when its external
  requirements pass; hosted macOS execution remains unobserved locally.
  All project Markdown is reviewed at this checkpoint. Evidence:
  [`json-validation-20260920.json`](benchmarks/json-validation-20260920.json).
- **Gotchas:** 1,033 counts checks, not independent input documents. Numeric
  storage is double, not exact arbitrary-precision integers. Lone surrogates,
  invalid UTF-8, overflow and nonzero underflow to zero are rejected by policy.
  Quoting expects valid UTF-8. The later JSON conversion block checks tensor
  dimensions and byte extents; the Qwen block validates construction geometry
  and tensor layouts. Token/request checks and general filesystem-path handling
  remain separate. No inference arithmetic changes.
  Suite timings are diagnostic; existing HF/mx merge requirements stay open.

### Native CPU decode sampling (diagnostic complete)

- **Goal:** identify sampled native instruction/function locations during current
  8B decode without adding timers to runtime source. This is a separate
  diagnostic after caller attribution proved no production saving.
- **Done:** optimized PDB harness builds against 17 unchanged runtime files.
  Session 65416 completes the discarded pair, then stops on xperf's unquoted
  commas in C++ symbol fields. A separate parser recovers the saved trace;
  the initial failure and all 41 original identities remain unchanged.
  Independent recovery review pins 51 additional identities. Session 21434
  runs exactly the remaining six invocations and terminates with exit 0.
  All eight planned invocations and 16 finite full vectors pass; vectors are
  byte-identical to the historical unchanged-runtime reference.
- **Done:** independent audit rehashes both manifests, checks collector ownership,
  Running/Stopped states, fixed ready/done holds, and reconstructs all target
  samples directly from saved exports: 39,209 discarded, then 40,115, 39,733
  and 39,571 measured. Every trace has zero lost events/buffers and 100%
  named application-symbol coverage. All unknown/system samples are retained.
  Of 119,419 measured samples, 116,184 (97.291051%) land in `dot_row_impl`.

| 8B phase, mean elapsed ms | Plain | Sampled | Sampled/plain change |
|---|---:|---:|---:|
| Prefill, 215 tokens | 7060.177800 | 7022.348833 | -0.535808% |
| Decode, 32 tokens | 6970.996900 | 7034.000533 | +0.903797% |

- **Done:** exact-binary mapping verifies 345 instructions and all 1,476 code
  bytes against the frozen executable. Actual image bases and PE exception
  ranges resolve every dot sample to an instruction start; padding is excluded.
  All non-dot and caller/other-thread counts remain in the evidence.
- **Left:** external HF/mx requirements remain open. This diagnostic selects no
  production optimization and does not reopen rejected studies. Main/GitHub
  remain unchanged. Full checkpoint evidence and Markdown review are archived
  in [`cpu-native-decode-sampling-20260920.json`](benchmarks/cpu-native-decode-sampling-20260920.json).
- **Gotchas:** three measured pairs, six threads, ubatch 128 and F32 KV; one
  internal warmup per process. Decode paired elapsed changes are +2.915900%,
  -0.611112% and +0.453332%. These combine profiler/handshake/state effects
  and variability, not pure tool overhead. Samples include outside-clock gate
  activity and identify execution locations, not hardware-stall causes or
  elapsed per-operation costs. This exact-vector check supplements earlier
  HF evidence; it does not replace the independent correctness gate.

### Current decode caller-cost attribution (diagnostic complete)

- **Goal:** separate caller work from the previously mixed dispatch residual
  before selecting another optimization. No placement study is reopened.
- **Done:** unchanged production, legacy worker probes and added caller probes
  complete all 12 fixed 8B invocations in session 21493, exit 0. Six threads,
  ubatch 128, F32 KV, 215 prompt plus 32 forced tokens; one outer triplet and
  each process's first iteration are prospective warmups. All 24 full vectors
  are finite and byte-identical to the pinned unchanged-runtime vector. All 93
  frozen identities recheck unchanged. Each of four attributed traces has 32
  steps, 5,792 dispatches and 37,152 caller events, with zero accounting gap.
  Builds, native grouped/error checks, active-probe error/reuse, synthetic
  accounting and independent malformed-trace checks pass. Native Windows
  sampling resolves 6,205 of 6,221 samples to two named C++ test functions
  using local PDBs; this earlier capability checkpoint does not sample a model
  or establish a memory-stall diagnosis. The later model study is above.
- **Finding:** observed means are 217.602291 ms/token in dispatch, 1.972685
  model-side, 0.080984 backend caller work, 0.015842 harness and 0.067954
  explicit caller-observer brackets, plus 0.000014 outer timer fringe.
  Model-side includes serial backend norms/RoPE. SwiGLU is 1.157311 ms/token
  within the model-side total. Attributed decode elapsed is 2.140880% above
  plain and 1.863856% above legacy spans; paired differences against spans
  change sign. Perturbation/variation is comparable to or larger than the
  individual residual regions. Explicit brackets do not capture all observer
  effects, and standalone calibration is not subtracted from model timings.
- **Decision:** no recoverable production saving is proved. Stop the
  outside-kernel optimization direction without probe tuning, a repeat timing
  screen or runtime implementation. Keep all samples. Native function sampling
  is exercised in the separate model study above; none was selected or run
  in this earlier caller-attribution study. External HF/mx gates remain open.
- **Left:** the existing external decode performance gaps remain open. Any
  separate native model sampling needs its own fixed plan, output checks and
  unprofiled control. No production implementation follows this diagnostic.
  Independent terminal audit rechecks all 93 identities, 24 vectors and
  148,608 caller events, reproducing every integer time partition.
- **Gotchas:** the old 2.010823 ms/token residual is historical motivation,
  not a current serial-cost estimate. Region times are instrumented intervals,
  not production savings; exact final vectors are not an external HF gate.
  Production source, tests, build/CI and root executable remain unchanged.
  Full evidence: [`cpu-decode-caller-cost-20260920.json`](benchmarks/cpu-decode-caller-cost-20260920.json).

### Synchronous CPU prefill placement (old screen failed; reassessment reopened)

- **Goal:** test one backend-owned synchronous callback around the complete
  prefill, after the smaller operation-local candidate failed. Keep platform
  details below Model and include setup, callback and checked cleanup costs.
- **Done:** all 36 primary invocations completed with exit 0. Both prefill cases
  pass against fresh production and disabled prototype, with 5/5 wins against
  each. However, 8B decode mean loses 0.573915% against disabled, beyond the
  frozen 0.5% limit. Reject this integration and stop placement adoption.
  Under that original rule, follow-up timing and conditional HF/mx runs did
  not proceed. The user has since requested evaluating large gains against
  minor losses: adoption is reopened for further validation, with this failed
  screen and all samples preserved.
  Windows native checks pass 7/7; callback/tiny-model contracts pass 17 cases
  and 4,626 exact values, plus 27 Windows placement cases and 1,176 exact
  values. Linux unchanged native assertions and callback/no-op guards pass.
  Both one-thread perf smoke floors pass (placement inactive).
  All 18 separate real-model witnesses pass: exact fresh-production vectors
  for initial and one/nine-token continuation cases on both models. Enabled
  processes have 12 applies/restores for primary and 24 for follow-ups across
  two iterations; disabled witnesses and decode have zero setters. Production
  has no instrumentation. All 36 timed vectors match their witness controls.
  Preflight passes 103 checks; 216 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-prefill-callback-20260920.json`.
  Independent timing/archive audit and the full 25-file Markdown checkpoint
  review are complete.
- **Left:** fresh active-path HF/lossless checks pass in the reassessment block;
  short-follow-up and fresh matched mx validation remain open under the user's
  clarified tradeoff preference. Retain the
  bounded callback and existing mapping. Report uncertainty for small decode
  differences and gains across phases; the old cutoff failure stays recorded.
  Production scheduling remains unchanged until adoption is validated.
- **Gotchas:** scratch Backend/Model/CPU only; production source, tests, build
  files, public API and root executable are unchanged. Successful placement
  uses one apply and one checked restore pool dispatch. Partial application
  restores all changed participants before unbound fallback. Cleanup retries
  once but reports its first error; persistent refusal does not establish safe
  pool/model reuse. Initial scope is six supported Windows participants,
  mechanically quant-independent, with no wider performance claim. Source
  guards reject nesting/thread changes but do not make Model concurrent.
  Timed policy permits fallback; witness counters prove their own processes.
  Exact vectors are not an independent HF gate. The failed primary screen
  originally prevented the planned short-follow-up performance runs, so their
  timing remains unverified despite passing numerical/activation witnesses.

| Model / phase | Production mean tok/s | Disabled mean tok/s | Enabled mean tok/s | Mean vs production | Mean vs disabled | Median vs disabled |
|---|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 474.695274 | 472.415155 | 548.326819 | +15.511% | +16.069% | +17.878% |
| 0.6b / tg | 48.835786 | 50.004903 | 50.342125 | +3.084% | +0.674% | +0.222% |
| 8b / pp | 30.392742 | 30.891701 | 41.452967 | +36.391% | +34.188% | +34.854% |
| 8b / tg | 4.628661 | 4.649671 | 4.622986 | -0.123% | -0.574% | +0.004% |

### CPU-local batched-matmul placement (scratch candidate screened out)

- **Goal:** keep placement inside the CPU backend and existing callbacks,
  avoiding a generic Backend phase API or model-level platform code.
- **Done:** all 24 invocations completed with exit 0. Prefill meets the frozen
  >=5% mean/median and 4/5-win screen in both models, but 0.6B decode loses
  2.62% mean and 2.72% median throughput, beyond the 0.5% limit. Reject this
  candidate; no sample removal, rerun or inherited phase-wide result.
  All final vectors match within model and against prior unchanged controls.
  Fresh Windows lifecycle gate passes 41 cases and 11,849 exact values;
  native backend-group passes unchanged 540 cases per arm. Linux native/no-op
  checks and both one-thread perf smoke floors pass. Four real-model witness
  processes verify exact prior outputs, candidate apply/restore counts of
  4,704 for 0.6B and 6,048 for 8B across two iterations, and zero decode/control
  setters. All 130 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-matmul-placement-20260920.json`.
  Independent timing audit and full 25-file Markdown checkpoint review pass.
- **Left:** the later whole-prefill callback above is the preferred reopened
  assessment under the user's clarified tradeoff preference. This per-operation
  arm and its failure remain archived. Production is unchanged; its prepared
  active-path HF plan was not run.
- **Gotchas:** six-thread Q8_0 nbatch > 1 scope only; other paths fall back.
  Every eligible operation queries topology and each nonempty callback pays
  apply/checked restore. Timing compares enabled/disabled prototype policy,
  with inert scaffolding in control; it is not the exact production baseline.
  Instrumented witness activation covers its own processes, not every timed
  callback. Allocation failure before dispatch propagates; dual body/restore
  failure can replace the task payload with cleanup failure. Persistent restore
  refusal is reported and only the synthetic test performs manual rescue.
  No production source, public flag, Backend API or root executable changed.

| Model / phase | Disabled mean tok/s | Enabled mean tok/s | Mean change | Median change | Enabled wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 471.700388 | 504.994000 | +7.06% | +8.73% | 4/5 |
| 0.6b / tg | 49.597335 | 48.299841 | -2.62% | -2.72% | 1/5 |
| 8b / pp | 30.040204 | 40.211011 | +33.86% | +34.11% | 5/5 |
| 8b / tg | 4.502460 | 4.575066 | +1.61% | +1.60% | 5/5 |

### Prefill placement without diagnostic observers (scratch screening passed)

- **Goal:** establish whether the prefill placement benefit survives removal
  of shared observer dispatches before considering production integration.
- **Done:** all 24 invocations completed with exit 0, and the frozen screen
  passes. Both prefill means/medians improve at least 5%, with 5/5 wins per
  model; decode means/medians stay within the 0.5% regression limit.
  Each model's 12 finite final vectors match exactly, with internal warmup
  identity also checked. All 48 source/build/check identities were frozen.
  The new parser accepts seven archived lifecycle records and rejects 34
  corruptions; six rule-boundary checks pass. Unchanged helper lifecycle
  evidence is reused explicitly, not claimed as fresh execution.
  Independent timing audit and review of all 25 Markdown files are complete.
  Full results are in `docs/benchmarks/cpu-prefill-observer-free-20260920.json`.
- **Left:** the CPU-local batched-matmul prototype above is rejected by its
  separate performance screen. The synchronous callback is implemented only
  in scratch and failed its old primary cutoff. Its adoption decision is now
  reopened above; the production Backend API remains unchanged.
  Independent HF and fresh matched mx gates remain required.
- **Gotchas:** scheduler mode constructs no placement Session. Candidate
  construction/apply/prefill/verify/checked restoration are timed; decode
  follows immediately. Stored pre-decode witnesses are serialized afterward,
  and the already-restored destructor performs no pool/affinity call.
  Formatting and inert object disposal are outside timing. There is no
  post-decode mask observation. Six threads, one prompt, Windows only;
  this remains scratch evidence, not production or HF/mx acceptance.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 479.228537 | 529.632147 | +10.52% | +11.02% | 5/5 |
| 0.6b / tg | 49.518222 | 49.633298 | +0.23% | +0.10% | 3/5 |
| 8b / pp | 29.027817 | 40.985134 | +41.19% | +40.29% | 5/5 |
| 8b / tg | 4.581855 | 4.600768 | +0.41% | +0.79% | 4/5 |

### Prefill-only CPU placement (scratch screening passed)

- **Goal:** test the measured prefill opportunity while restoring normal
  scheduling before decode, including recurring placement costs in timing.
- **Done:** the fixed 24-invocation comparison completed with exit 0 and passes
  its frozen screen. Both models improve prefill mean/median by at least 5%
  with 5/5 wins, and decode mean/median stay within the 0.5% regression limit.
  All 12 saved final vectors per model match exactly; each internal warmup
  also matches its measured iteration. Lifecycle tests pass 3,636 exact Q8 value
  comparisons, fresh iterations, exception cleanup and same-pool reuse.
  The parser accepts seven lifecycle records and rejects fifteen corruptions.
  Independent preflight passes; 46 identities were frozen before timing.
  Independent post-run audit and all 25 Markdown checkpoint reviews pass.
  Complete results are in `docs/benchmarks/cpu-prefill-placement-20260920.json`.
- **Left:** the separate observer-free study above passes its frozen screen.
  Broader thread/phase lifecycles, independent HF correctness and
  fresh matched mx performance remain required before adoption.
- **Gotchas:** this is a screening pass, not production readiness. Restoring
  masks does not reset cache, boost or scheduling state. The full candidate
  Session lifecycle, including report serialization and destruction, is timed
  as prefill. Common observation work outside both clocks can influence decode.
  The result covers one 215-token prompt, 32 forced decode steps, six threads,
  ubatch 128, F32 KV and Windows Ryzen 7 5800X only. No HF/mx floor follows.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 467.836623 | 535.466490 | +14.46% | +11.93% | 5/5 |
| 0.6b / tg | 49.293600 | 49.418118 | +0.25% | +0.39% | 3/5 |
| 8b / pp | 29.521998 | 40.948992 | +38.71% | +40.36% | 5/5 |
| 8b / tg | 4.594650 | 4.623768 | +0.63% | -0.30% | 3/5 |

### Explicit CPU worker placement (all-phase candidate screened out)

- **Goal:** compare scheduler-selected placement with six workers on six
  distinct queried physical cores, including caller worker zero, without
  changing kernels or production options.
- **Done:** helper success/unbound, caller/worker failure cleanup, destructor
  restoration and 1,818 exact grouped Q8 value checks pass. Independent
  preflight verifies source fidelity, masks, actual CPU witnesses and cleanup.
  All 24 real-model invocations pass. Fixed-arm before/after snapshots verify
  logical CPUs 0/2/4/6/8/10 from the same queried topology and process mask;
  every original thread mask is restored. Saved finite final vectors match
  byte-for-byte within each model, across both arms and all invocations.

  | Model / phase | Scheduler mean tok/s | Fixed mean tok/s | Mean change | Median change | Fixed wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 479.071849 | 541.704220 | +13.07% | +11.58% | 5/5 |
  | 0.6B decode | 49.314603 | 47.022156 | -4.65% | -5.70% | 0/5 |
  | 8B prefill | 30.183415 | 40.902861 | +35.51% | +35.72% | 5/5 |
  | 8B decode | 4.564495 | 4.649338 | +1.86% | +2.00% | 5/5 |

  The frozen rule requires at least 0.5% higher decode mean and median with
  4/5 wins in both models, and no prefill mean/median regression above 3%.
  Small-model decode fails; the all-phase candidate stays outside production.
  All samples are retained; no new mx or independent HF gate was run.
- **Left:** retain scheduler-selected production behavior. The separate
  prefill-only study above passed its own frozen screen with transition costs
  included; this all-phase candidate remains rejected.
  Matched external decode requirements remain open.
- **Gotchas:** topology and allowed mask are queried in each owned child;
  adjacent CPU IDs are observed, not assumed. Core placement includes serial
  caller work and warmup first-touch, and cannot isolate migration/SMT effects.
  Final-position vectors do not prove full-corpus/deep-context correctness.
  No mapping search, production flag, API or cross-platform affinity promise.
  Evidence: `benchmarks/cpu-worker-placement-20260920.json`.

### Exact decode SwiGLU callback fusion (screened out)

- **Goal:** measure unchanged SwiGLU inside the existing equal-row Q8 gate/up
  worker callback, preserving all buffers, arithmetic and projection order.
- **Done:** MSVC and GCC each pass 1,440 matrix arm comparisons, 126,990
  finite bit comparisons and 27,054 nonfinite classifications. A separate
  instrumented copy witnesses 16,920 rows exactly once; an arithmetic mutant
  compiles and is rejected numerically. Actual unchanged grouped projections
  plus an independent literal SwiGLU expression supply the synthetic reference.
  Assembly confirms original dots and scalar exp/divide/multiply order.
  All 40 fixed timing samples complete with exact gate/up/output checks.

  | Synthetic shape | Serial mean ms/call | Fused mean ms/call | Mean change | Median change | Fused wins |
  |---|---:|---:|---:|---:|---:|
  | Small gate/up + SwiGLU | 0.089504 | 0.085051 | -4.98% | -5.91% | 7/9 |
  | Large gate/up + SwiGLU | 2.442995 | 2.383465 | -2.44% | +0.25% | 7/9 |

  The frozen screen requires lower mean and median in both shapes with at
  least 6/9 wins each, plus at least 2% lower mean and median in one shape.
  The large median fails; no model integration follows. All samples retained.
- **Left:** retain current production behavior; do not weaken the prospective
  screen or repeat this experiment merely to obtain a passing sample.
  Independent HF and matched external decode requirements remain open.
- **Gotchas:** both timed arms use one concrete Q8 helper, not the generic
  production dispatch. Repeated synthetic weights can remain cached. This is
  neither a model slowdown finding nor an HF/external performance result.
  Numerical checks use the standard floating environment and compare within
  each compiler; NaN payload and alternate rounding-mode identity are unclaimed.
  Evidence: `benchmarks/q8-swiglu-fusion-screening-20260920.json`.

### Native Q8 bounded inner-loop follow-up (rejected)

- **Goal:** test one ordinary two-trip loop with local accumulators, preserving
  one copy of the native block body and every arithmetic operation.
- **Done:** MSVC emits two block bodies without accumulator stack traffic.
  Independent instruction review finds 55 F16C-path instructions per pair
  versus 54 for two control iterations. Arithmetic and branch counts are
  unchanged; the extra instruction reloads the feature flag inside the loop.
  No second-block work moves before the first block's final FMA/exit check.
  This fails the useful-scheduling gate; no numerical/model/timing runs follow.
- **Left:** stop this unrolling exploration. Historical and current-source 8B
  whole-operation diagnostics are complete below. Within-group attribution
  remains unresolved; use the matrix-cost findings to select the next study.
- **Gotchas:** instruction counts are not micro-op counts or measured latency.
  No hints, forced inlining, feature specialization or duplicated kernel bodies
  were used. Production remains unchanged. Evidence:
  `benchmarks/q8-bounded-inner-loop-rejection-20260920.json`.

### Native Q8 block scheduling study (rejected at codegen gate)

- **Goal:** expose two consecutive native Q8 blocks to compiler scheduling while
  preserving every weight product, four FMA chains and the original reduction.
- **Done:** isolated control/candidate comparators build with MSVC. Independent
  scalar/control oracles pass 612,267 finite bit checks and 1,939 nonfinite
  classifications on both MSVC and GCC. A repeated-block mutant compiles and
  fails numerically. The two-block lambda preserves the original HADD epilogue,
  block order and odd tail, but MSVC emits eight unconditional 32-byte
  accumulator stack stores per two-block iteration, including the F16C path.
  This fails the planned no-spill codegen gate; the formulation is rejected.
- **Left:** no adoption or model timing for this formulation. Continue the
  external decode performance work from the unchanged production kernel.
- **Gotchas:** this is a codegen rejection, not a measured slowdown. The oracle
  is scoped numerical evidence, not an HF/model validation claim. Prior
  pointer, feature-specialization and integer-unroll studies remain distinct.
  Evidence is in
  `benchmarks/q8-native-block-scheduling-rejection-20260920.json`.

### Exact Q8 horizontal reduction study (rejected)

- **Goal:** reduce Q8 row-dot epilogue cost while preserving all float products,
  FMA chains and contributing addition order.
- **Done:** the scratch shuffle/add sequence emits the intended instructions.
  Native CTest passes 7/7 and required-HF suites pass 11/11 for control/candidate
  on Windows/Linux. MSVC scalar checks pass 612,267 finite bit comparisons and
  1,939 nonfinite classifications; a wrong shuffle is rejected. Windows repeated
  control and candidate match 5,013,888 long-prompt logits and four serial NLL
  cases, within HF bounds. Windows candidate 8B HF checks pass 37/37.
  All 60 fixed timing processes completed: one discarded warmup plus nine
  measured rounds per model/arm, with every sample retained.

  | Mean tok/s | Current | Candidate | mx | Candidate/current | Candidate/mx |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 473.669 | 453.120 | 277.206 | -4.34% | +63.46% |
  | 0.6B decode | 48.966 | 48.723 | 49.969 | -0.50% | -2.49% |
  | 8B prefill | 29.646 | 29.936 | 21.348 | +0.98% | +40.23% |
  | 8B decode | 4.579 | 4.549 | 4.610 | -0.65% | -1.31% |

  Both decode medians also trail current/mx. Candidate wins only 3/9 decode
  pairs against current and 0/9 against mx on each model. The reduction is
  rejected; production stays at `bf122fd`, and the native regression proposal
  remains unapplied. Validation evidence is in
  `docs/benchmarks/q8-exact-reduction-validation-20260920.json`; completed timing
  is in `docs/benchmarks/q8-exact-reduction-performance-20260920.json`.
- **Left:** close the separate production decode floor. Current control means
  in this session trail mx by 2.01% (0.6B) and 0.67% (8B); do not pool this with
  earlier sessions or revive the rejected epilogue based on selected samples.
- **Gotchas:** the low 0.6B candidate prefill sample (357.111 tok/s) stays in the
  mean. Its median is 471.589 versus current 476.152; do not describe the mean
  gap as a universal causal slowdown. Correctness alone does not justify this
  performance change. NaN payloads and alternate rounding modes remain unclaimed.


### Optional Qwen3-8B HF consumer

- **Goal:** compare the verified local Q8_0 GGUF against the independent pinned
  8B HF tokenizer, logit and PPL goldens without adding large CI downloads.
- **Done:** original HF generation and provenance evidence are committed in
  `bf122fd`. Before any llmx 8B comparison, declare exact tokenizer/input IDs,
  six exact top-1 matches and top-5 set overlap 5/5; all ten printed logits must
  be finite, sorted, unique-token and within absolute magnitude 100. NLL delta
  limits are 0.01 continuous and 0.02 for each windowed case, prospectively
  reusing the existing Q8 quality budget, not calibrated from 8B results.
  Consumer implementation and independent review are complete. Both platforms pass
  all 37 checks: 20 tokenizer cases, six prompt-ID/ranking pairs, PPL IDs and
  four NLL cases. Largest NLL delta is 0.002185355, under its 0.02 limit. The
  full required-HF suite passes 11/11 on each platform, including consumer
  rejection tests (Linux perf is diagnostic only). The official pinned GGUF is
  downloaded and hash-verified in the Linux cache documented in ASSETS. The
  successful Linux gate used an identical staged copy, removed only after its
  tests finished. Interrupted mounted-file results and separate intervention
  metadata are preserved. Copy plus verification took 59.17 seconds; direct
  download plus verification took 486.61 seconds. These are operational I/O
  observations, not inference measurements. Prefer an existing verified copy
  when faster and reuse the completed Linux cache. Windows original untouched.
  All 25 Markdown files reviewed and stale performance wording corrected.
  Evidence: `benchmarks/hf-8b-validation-20260920.json`.
- **Left:** merge the validated consumer with the runtime stack after its
  external performance gates pass. Keep failures with original bounds;
  investigate rather than relaxing thresholds to fit observations.
- **Gotchas:** exact original GGUF conversion revision is undocumented. The
  official model-family link and file hash do not prove identical source
  weights. Fixed excerpt/rank checks do not cover full corpus or all logits.

### CPU ordered prefill reductions

- **Goal:** determine whether the four-row/three-column prefill kernel's
  addressable accumulator array adds avoidable stack traffic or reduction
  overhead, without changing per-lane FMA or final addition order.
- **Done:** explicit ordered reductions match 1,824 scalar-FMA outputs across
  dimension tails and unaligned inputs. MSVC assembly removes most epilogue
  accumulator stack traffic; the FMA loops do not spill in either arm. The
  function grows from 1,119 to 2,931 bytes. Nine alternating rounds improve
  mean Q8/F32 prefill by 4.59%/7.01%, winning 8/9 and 9/9 pairs. A separate
  nine-round Q8 follow-up repeats the prefill gain (+6.25%, 9/9 pairs).
  Q8 decode changes from -1.81% to +0.71% versus control between sessions;
  no stable decode regression or universal external parity is established.
  All control/candidate final-vector hashes match. Windows/Linux native checks
  pass 7/7 and required-HF suites pass 10/10. Linux real-F32 tokenizer/logit/NLL
  checks also pass. Both platforms check each arm against the same scalar-FMA
  oracle; an MSVC mutant swapping final additions is rejected. Fresh long
  comparisons have 5,013,888 byte-identical logits per F32/Q8 model. F32 maximum
  HF error is 0.000126362 <= 0.001; prefilled continuation NLL deltas are
  0.000000645 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8). Four separate serial
  NLL cases equal control and pass existing HF bounds. All 25 Markdown files
  reviewed; evidence: `benchmarks/prefill-ordered-reduction-20260920.json`.
- **Left:** keep this validated feature branch; merge with the
  runtime stack only when broader external performance requirements pass.
- **Gotchas:** no reassociation or new activation quantization. A synthetic
  gain alone does not establish the external floor. Keep worker implementation
  unchanged and isolate timing from other builds/tests and user inference.

### Pinned HF reference generation

- **Goal:** reuse independent HF tokenizer/logit/PPL generation for explicitly
  pinned models, keeping larger-model fixtures separate from the existing suite.
- **Done:** model/revision/output and associated GGUF labels are explicit;
  numerical loaders share pinned CPU float32 eager execution. Alternate models
  require separate output and cannot overwrite the default fixture directory.
  Windows full required-HF suite passes 10/10 components against the unchanged
  validated 9cfe43f executable; Linux generator safeguards pass 2/2 tests.
  Two offline generations from cached 0.6B HF weights reproduce every numerical
  field. Against committed goldens: 20 tokenizer cases, six top-10 ID lists,
  247 PPL token IDs, four NLL values and synthetic F32 JSON match exactly.
  Rounded top-10 logit values differ by at most 0.0001. Existing fixtures,
  acceptance bounds and default CI model downloads are unchanged. Evidence:
  `benchmarks/hf-reference-tools-20260919.json`.
  Parallel work on the Linux machine generated actual 8B tokenizer/logit/PPL references from
  verified original `Qwen/Qwen3-8B` at pinned `b968826d9c46dd6066d109eabc6255188de91218`.
  All three modes pass with CPU FP32 eager execution. Measured memory reaches
  the owned container's 40 GiB cap including file cache (3,098 limit events,
  zero OOM/kill). Evidence: `benchmarks/hf-8b-reference-20260920.json`.
- **Left:** merge the tooling with the runtime stack after its external gates
  pass. The separate 8B consumer passes Windows and Linux checks under
  predeclared bounds. Official GGUF metadata links
  the base model and matches the local Q8 digest, but exact original conversion
  revision is undocumented. Default CI downloads remain unchanged.
- **Gotchas:** do not overwrite small-model goldens with another model or expand
  default CI downloads. Tooling support alone is not an 8B correctness result.
  The tooling-only checkpoint changed no hot path and ran after worker timing.
  The parallel 8B HF work ran on the separate Linux machine; it is not a performance gate.

### Separate Q8 scale/payload storage study (screened out)

- **Goal:** test a scratch storage view with original half-scale bytes separate
  from contiguous 32-byte weight blocks, preserving all arithmetic and values.
- **Done:** MSVC and GCC each pass 612,267 finite bit comparisons, 1,939
  nonfinite classifications and 141 packing cases. Grouped/standalone witnesses
  confirm the split dot runs; a corrupted-scale mutant compiles then fails.
  Native assembly preserves the FMA chains/HADD with no loop accumulator spills.
  Fixed synthetic timing completes all six shapes and 96 samples, with exact
  output checks throughout. No samples are dropped.

  | Shape | Original mean ms/call | Split mean ms/call | Mean change | Median change | Split wins |
  |---|---:|---:|---:|---:|---:|
  | Small up | 0.047025 | 0.047276 | +0.53% | -0.51% | 4/7 |
  | Small gate/up | 0.078415 | 0.078916 | +0.64% | +0.69% | 1/7 |
  | Small down | 0.042013 | 0.041720 | -0.70% | -1.84% | 3/7 |
  | Large up | 0.865281 | 0.844761 | -2.37% | -6.07% | 5/7 |
  | Large gate/up | 2.406585 | 2.359071 | -1.97% | -2.83% | 6/7 |
  | Large down | 0.859395 | 0.850048 | -1.09% | -0.20% | 3/7 |

  The frozen advancement rule requires at least 3% mean and median improvement
  with at least 5/7 wins in every large case, and no small-case mean/median
  regression above 3%. It fails; no model integration follows this study.
- **Left:** retain production storage. Revisit only with a distinct hypothesis;
  do not weaken the screening rule or infer a real-model improvement from these
  short cached matrix measurements. External decode requirements remain open.
- **Gotchas:** packing takes 8.96/19.04/9.09 ms in the three large cases and
  adds another weight-sized retained allocation in this diagnostic. Logical
  retained bytes exclude allocator overhead and an additional unpack-validation
  temporary. No peak-RSS, model-loading, HF or external-performance claim.
  Evidence: `benchmarks/q8-split-storage-screening-20260920.json`.

### Current 8B worker-span diagnostic

- **Goal:** measure instrumentation impact on the current 8B runtime and
  attribute decode dispatch intervals to operations before selecting a change.
- **Done:** current source snapshots, plain/instrumented builds and fault check
  pass. All eight fixed processes pass: one discarded outer warmup pair and
  three alternating measured pairs. Every process internally warms up. Saved
  151,936-float final vectors match byte-for-byte across all eight invocations;
  internal warmups use FNV64. Each instrumented trace has 865 prefill and 5,792
  decode records with valid ordered timestamps and no overflow.

  | Phase time, ms | Plain mean | Plain median | Spans mean | Spans median | Mean change |
  |---|---:|---:|---:|---:|---:|
  | Prefill, 215 tokens | 7224.912 | 7215.189 | 7188.779 | 7186.463 | -0.50% |
  | Decode, 32 tokens | 6955.863 | 6934.154 | 6887.725 | 6864.781 | -0.98% |

  Paired changes reverse direction in both phases; no probe speedup is claimed.
  Instrumented decode dispatch is 213.231 ms/token within 215.241 ms/token phase
  time. Gate/up accounts for 46.82% and FFN down for 23.76% of dispatch time;
  all matrix projections total 98.54%, attention 1.46%.
- **Left:** the separate Q8 scale/payload study above missed its screening
  rule. Do not change workers or extrapolate an external performance pass from
  this three-pair diagnostic. No model integration is selected.
- **Gotchas:** all builds/tests finished before timing; all measured samples
  remain. No mx comparison or independent HF gate was run. Last-finisher entry
  can overlap other workers' compute and callback intervals can include
  descheduling. These intervals are not all recoverable overhead. Evidence:
  `benchmarks/current-8b-worker-spans-20260920.json`.

### Archived decode operation attribution

- **Goal:** label the existing 0.6B decode worker spans by operation, using the
  exact archived model/backend source and shapes to prove dispatch order.
- **Done:** all 12 traces and 54,144 decode records map to 141 dispatches per
  token: five per layer across 28 layers, then vocabulary projection. Archived
  source, guards and shapes prove the order. Exact last-finisher decomposition
  passes per operation and sums back to the original dispatch totals. In the
  historical instrumented current Q8 arm, gate/up accounts for 26.07% and
  vocabulary projection for 21.90% of dispatch time. Each of the five layer
  operations changes its control/current delta sign across three pairs.
- **Left:** within-group Q/K/V and gate/up member timing remains unresolved.
  Do not rewrite workers on this evidence. The separate current-source 8B
  plain/spans diagnostic is complete in the block above.
- **Gotchas:** archived current is `9cfe43f`, not a fresh `bf122fd` measurement.
  Model and decode paths are unchanged, but prefill source/compiler layout
  differs. Instrumentation perturbs timings; these are neither external-floor
  results nor evidence of a causal regression. Evidence:
  `benchmarks/worker-decode-operation-attribution-20260920.json`.

### CPU worker cost profile

- **Goal:** localize the remaining prefill/decode costs before selecting another
  hot-path change; compare the pre-error worker control with the retained runtime.
- **Done:** operation-level profiles cover 24 processes: Q8/F32, one/six
  workers, both source arms, three alternating pairs and two instrumented
  sequences after an uninstrumented warmup. Final-vector hashes agree with
  warmups, across source arms and thread counts. The initial instrument double
  counted attention's nested parallel_for; its consistency check rejected the
  run and the corrected probe excludes nested calls. Full samples and sources:
  `benchmarks/cpu-worker-profile-20260919.json`.
  Release definitions, DLL imports and the archived DLL hash establish OpenMP
  workers/barriers in the mx reference. Historical measurements are retained;
  ASSETS and the original artifact now carry a dated interpretation correction.
  The separate per-participant probe is complete: 24 serial Q8/F32 processes,
  six participants, three alternating pairs of plain/instrumented builds for
  both snapshots. Final full vectors match byte-for-byte across every arm and
  mode; all traces have 673 prefill and 4512 decode dispatches with ordered
  timestamps. The instrumented current failure/drain/reuse check passes.
  Evidence: `benchmarks/cpu-worker-spans-20260919.json`.
- **Left:** whole decode operations are now labeled by the verified archived
  dispatch order (see attribution block above). Individual projections inside
  grouped callbacks remain unresolved, and instrumentation changes timing
  materially. No production runtime change is selected.
  Keep the existing worker implementation and the external performance gate;
  do not repeat rejected dispatch variants on this evidence.
- **Gotchas:** instrumentation changes timing. The three-process comparison
  locates costs but does not clear a small regression or prove causality.
  Keep profiles isolated from builds/tests and user inference. Reference source
  may be inspected but cannot be copied into llmx.

| Instrumented phase mean ms, six workers | Before worker fix | Retained runtime |
|---|---:|---:|
| Q8 prefill | 562.43 | 573.38 |
| Q8 decode, 32 steps | 764.09 | 762.44 |
| F32 prefill | 617.72 | 626.08 |
| F32 decode, 32 steps | 2443.77 | 2431.62 |

Q/K/V and gate/up account for 10.57 ms of the 10.95 ms mean Q8 prefill difference
and 7.94 ms of the 8.36 ms F32 difference. These include dispatch/wait time.
Q8 prefill medians reverse the small mean ordering (569.89 vs 565.93 ms), so
this diagnostic does not establish a stable regression magnitude. The next
table is the separate completed span probe; do not pool the two sessions.

| Span-probe phase mean ms | Before plain | Retained plain | Before instrumented | Retained instrumented |
|---|---:|---:|---:|---:|
| Q8 prefill | 632.945 | 567.707 | 591.131 | 574.741 |
| Q8 decode, 32 steps | 803.943 | 762.698 | 779.716 | 783.178 |
| F32 prefill | 638.665 | 627.730 | 645.636 | 635.732 |
| F32 decode, 32 steps | 2463.925 | 2479.028 | 2458.478 | 2562.645 |

Q8 decode's control/current ordering reverses with instrumentation. F32 decode
differs by +0.61% in plain builds but +4.24% in instrumented builds. The exact
last-finisher decomposition separates entry, callback and final completion;
it does not turn overlapping participant spans into additive phase costs.
These results do not identify a stable worker regression or prove its absence.
No mx benchmark or new independent HF gate was run by this scratch probe.

### Live generation and loading progress

- **Goal:** stream generated text immediately in chat/generate, show prompt
  processing before the first token, and make loader progress reusable by
  current CLI consumers and future serving (ROADMAP #7).
- **Done:** optional synchronous loader byte-progress and inference text callbacks
  are implemented; CLI generate/chat owns terminal detection, stderr status and
  stdout flushing. Prompt-processing status appears before prefill. Normal
  Qwen3 output and --think stream before the next model step; legacy retroactive
  filters retain their prior buffered behavior. Stop/EOS and follow-up cache
  accounting remain unchanged.
  Windows/Linux full required-HF suites pass all nine components. After review
  tightened completion ordering, final native checks pass 7/7 on both platforms;
  Linux follow-up chat/progress and version checks pass again. Four MSVC mutants
  fail as intended: delayed delivery, missing flush, ignored read failures and
  completion before a failing trailing seek. Full documentation review covers
  all 25 Markdown files, including current capabilities, CLI defaults and test
  scope. Evidence: `benchmarks/live-generation-20260919.json`.
- **Left:** merge with the enclosing runtime stack only after its external
  performance gates pass. GitHub remains
  main-only. Next runtime work should profile the remaining CPU costs.
- **Gotchas:** callbacks are synchronous and do not provide concurrent execution
  or resumable-session recovery. Byte chunks can split UTF-8 characters. Loading
  counts tensor payload bytes, not metadata/padding or model preparation; final
  completion follows every read/seek. The later GGUF validation checkpoint
  checks file extents before progress starts, and the Qwen checkpoint validates
  construction geometry/layouts. Token/request checks remain separate work.
  Legacy filtering buffers text when future markers can
  retroactively discard it; no server framework is added.

| Qwen3-0.6B Q8_0, 64 greedy tokens, median of 3 pairs | Before 8226e17 | Streaming |
|---|---:|---:|
| First visible text from process start (s) | 2.663 | 0.662 |
| Whole process elapsed (s) | 2.738 | 2.741 |
| CLI generation (tok/s) | 31.48 | 31.77 |

Same model/prompt, six CPU workers, stdout pipe and --think; one outer warmup
pair excluded. Every output byte matches. Builds/tests were stopped during
these runs. This is end-user delivery latency, including loading and prefill,
not a kernel-speedup or external mx-llama.cpp parity claim.

### Automatic build identification

- **Goal:** identify each CMake/plain MSVC build by release version plus Git
  revision, with a dirty marker for tracked changes and a clear archive fallback.
- **Done:** CMake refreshes build revision on each build, without rewriting an
  unchanged header. Plain build.bat emits the same metadata. --version and the
  usage banner show release plus Git revision and tracked-dirty state, with
  unknown fallback outside a checkout. Windows plain/CMake clean, dirty,
  new-commit, archive and no-op cases pass in a path containing spaces; Linux
  clean/dirty/new-commit rebuilds and version smoke pass. The current project
  build also reports its actual HEAD plus dirty state. Windows full required-HF suite passes with
  the new version regression. The Windows for/f equals-sign parsing issue was
  caught and fixed before the passing rerun.
  README now separates implemented CPU capabilities from future execution,
  serving and HF goals. All source/comments/docs and new messages use ASCII;
  Unicode fixture data is preserved. The requirement is recorded in AGENTS.
- **Left:** merge with the validated stack after its external gates pass.
  Live generation/progress reached checkpoint 9cfe43f; subsequent CPU cost
  profiles are recorded above. Full evidence for build identification is in
  `benchmarks/build-version-20260919.json`; all 25 Markdown files were reviewed
  for current capabilities, future goals, build behavior and ASCII compliance.
- **Gotchas:** untracked files do not mark a build dirty. Source archives report
  unknown even when nested in another repo. No timestamps, automatic release
  increments, commits/tags or new build/runtime dependencies are introduced.

### CLI thread settings

- **Goal:** make existing auto, decode and prefill thread flags work consistently
  for generate, follow-up chat and bench without changing kernel arithmetic.
- **Done:** generate/chat capture the resolved decode count, apply the prefill
  count and restore decode for every turn. Bench keeps auto selection; actual
  phase counts are visible through verbose/benchmark output. The new regression
  covers 32 generation/chat configurations and four bench cases per platform,
  with 64 HF-golden replies, and rejects all three reintroduced bug mutants.
  Windows full required-HF suite passes. Linux native tests and every correctness
  component pass; its initial automatic-thread synthetic floor fails. Pinning
  that test to its original one-worker conditions passes on both platforms,
  without changing floors. Logs retain the initial failure and the targeted
  reruns. Kernel/model comparator code is byte-identical to c072af2.
- **Left:** merge with the enclosing validated runtime stack once its external
  performance gates pass. Evidence: `benchmarks/cli-threads-20260919.json`.
- **Gotchas:** changing phase counts recreates the CPU pool. Automatic counts
  can be slower for tiny synthetic jobs, particularly under WSL; do not compare
  old serial-default benchmark results with new auto-default results. GPU
  backends will retain CPU-worker meaning for these flags; ubatch remains
  prompt tokens per forward pass. No GPU execution interface was introduced.

### CPU worker exception safety

- **Goal:** propagate CPU task failures after every participant finishes, with
  safe job lifetime and reusable dispatch state; clean up partial pool startup.
  This repairs the existing backend before ROADMAP #4a/#7 execution work.
- **Done:** dispatch catches caller/worker failures and waits for completion
  before rethrowing; startup joins partially created pools. Windows/Linux
  native checks, full suites with required real HF fixtures, Linux UBSan native
  tests and allocation/task fault sweeps pass. The original pool terminates on
  the task and partial-startup regressions. Independent real F32 HF checks pass;
  long F32/Q8 vectors and continuous/window NLL are exact against the control.
  Full samples, hashes, commands, logs and harnesses are archived in
  `docs/benchmarks/worker-errors-cpu-20260919.json`.
- **Left:** investigate the Q8 prefill cost before
  adoption, and close the external Q8 decode floor. No merge. The old root
  executable was retained at that checkpoint; current deployment is recorded above. Paired candidate Q8 prefill loses
  eight of nine rounds despite overlapping ranges; do not dismiss that as noise.
- **Gotchas:** dispatch recovery does not roll back partially written outputs
  or establish Model/session recovery. No concurrent submissions are supported.
  Later GGUF and Qwen checkpoints address file extents, configuration geometry
  and required tensor layouts; token/request checks remain open. Control is
  `3a82284`; no merge. The feature branch holds
  the checkpoint, while the public/default branch remains unchanged.

- **Post-reboot decision:** retain the existing c072af2 implementation. The
  stored-call variant failed its longer comparison. Failure-only exception
  bookkeeping loses all five paired decode rounds. A shared non-template
  dispatch body improves exploratory prefill but does not improve decode;
  it is not adopted. No more dispatch variants are planned without profiling
  evidence. The user explicitly asked to keep the implementation simple.
  Full samples and source patches for the two latest scratch studies are in
  `docs/benchmarks/worker-cold-errors-cpu-20260919.json` and
  `docs/benchmarks/worker-shared-dispatch-cpu-20260919.json`. Both pass initial
  MSVC allocation/task-fault and grouped-kernel checks; neither entered full
  HF/platform adoption gates. Those experiments changed neither runtime source
  nor the root executable.
  Next performance work should localize the remaining cost before changing code.
  User authorization includes merging main and publishing GitHub once the
  requirements pass; current performance evidence does not clear that gate.

Stored-call experiment is **not adopted**. Full HF, exact vectors/NLL,
Windows/Linux suites and Linux UBSan native checks pass, but the longer run
does not establish a performance gain. Evidence and commands:
`docs/benchmarks/worker-invocation-cpu-20260919.json`.

| Mean tok/s, nine rounds | Before errors | Error checkpoint | Stored call | mx |
|---|---:|---:|---:|---:|
| Q8 prefill | 422.70 | 406.76 | 404.32 | 262.84 |
| Q8 decode | 44.69 | 44.93 | 44.53 | 46.32 |
| F32 prefill | 354.95 | 346.77 | 343.63 | 369.07 |
| F32 decode | 13.44 | 13.43 | 13.30 | 13.33 |

All ranges overlap. Stored-call Q8 prefill loses every pair against the
pre-error control, and no mean beats the error checkpoint. Preserve the early
five-round result as exploratory, not a reason to select the variant.

| Mean tok/s, nine matched rounds | Control | Candidate | mx-llama.cpp |
|---|---:|---:|---:|
| Q8 prefill | 408.39 | 386.18 | 263.80 |
| Q8 decode | 43.88 | 43.94 | 45.81 |
| F32 prefill | 362.85 | 377.11 | 370.26 |
| F32 decode | 13.91 | 14.06 | 13.69 |

Q8 prefill mean changes by -5.44%; Q8 decode remains -4.09% below mx. All
candidate/control ranges overlap; F32 means lead in this session without an
equivalence claim. Separate synthetic timing also has lower prefill/decode
means (see ASSETS). All samples are retained; builds/tests do not overlap timing.

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### F32 embedding/matrix inference

- **Goal:** load and run F32 embeddings and matrices in dense Qwen3, with
  external HF numerical validation and measured CPU performance (ROADMAP #8).
- **Done:** direct F32 rows, float-aligned tensor blobs, deterministic HF
  full-logit/NLL fixtures (tied/untied, odd widths, batches, threads). Windows
  and Linux full suites pass; UBSan passes and catches the pre-fix alignment
  fault. New sanitizer CI job passes workflow lint. All 311 real Qwen3-0.6B
  F32 tensors verified against original HF weights; all real HF checks pass.
  Copy/direct controls match matrix hashes and all logits over a 1,943-token
  prompt plus 32 greedy tokens. Measurements/provenance are in ASSETS.
- **Left:** close the measured F32 CPU gap versus public mx-llama.cpp
  `5542318e74`, then merge and observe the expanded five-job hosted CI.
  Current F32 results are in the KV and worker-error blocks; the older
  scale/load comparison records that earlier checkpoint. Sustained external
  parity remains unproven. Loading is excluded and each process warms up.
- **Findings:** activation tiling and a fully spinning worker pool did not
  establish a win. Profiling instead identifies scalar attention as roughly
  170-180 ms of prefill. Backend attention now improves prefill by 24.8%
  against its interleaved control and passes HF bounds, while changing
  summation order. Remaining work is the CPU performance gap, not numerical
  validation of this attention implementation.
  `tools/compare_cpu.cpp` / `.py` preserve the matched measurement procedure;
  commands and limitations are in ASSETS.
- **Gotchas:** do not treat success on quantized weights widened to F32 as
  parity with the original HF weights. Validate tied/untied output and prefill.

### CPU attention in backend

- **Goal:** move causal GQA attention out of the model and into the backend
  (ROADMAP #4a), sharing the decode and prefill implementation and improving
  CPU throughput with measured, numerically bounded vectorization.
- **Done:** `Backend::attention` now handles both forward paths, with CPU-owned
  score scratch, causal GQA, AVX2 dots and value accumulation, and scalar tails.
  Deleted duplicate model-layer attention loops. Regenerated the tiny HF
  fixture at head width 10 to cover vector tails. Windows and Linux full
  suites pass, including real Q8/Q4 HF checks; Linux UBSan synthetic suite
  passes. Tiny maximum HF logit error is 6.2e-7. A future-token attention
  mutant fails with error 0.03787. Real F32 excerpt/window NLL differs from HF
  by at most 4.5e-6. At 1,943 prompt tokens plus 32 generated tokens, greedy
  output matches HF and all 5,013,888 logits are within 0.001 (max 0.0001252).
  Eight interleaved rounds give F32 prefill 286.84 -> 357.88 tok/s (+24.8%);
  decode 13.34 -> 13.57 tok/s (small change with overlapping ranges).
  Q8 synthetic guardrails pass; matmul 123.07 -> 121.80 GFLOPS, prefill
  4828 -> 4800 and decode 4522 -> 4875 tok/s, with overlapping ranges.
- **Left:** close the remaining external CPU floor gap, then merge and run
  hosted CI. See the KV and worker-error blocks for later measurements.
  Profiling after vectorization finds prefill attention around 60-67 ms and
  decode attention around 97-100 ms; matrix operations now dominate decode.
  Sequential F32 row streaming and parallel batched elementwise work are
  implemented on the next feature branch; see its validation block below.
- **Gotchas:** SIMD dot reductions change summation order. A matching top token
  or logit sum is not a numerical gate. This step does not implement device
  buffers, resident activations or async execution, which remain prerequisites
  for GPU backends.

### CPU row streaming / parallel prefill

- **Goal:** close the F32 CPU performance gap without changing weights or
  weakening the HF numerical gate (ROADMAP #8).
- **Done:** contiguous single-row F32 decode and parallel batched
  norm/RoPE/SiLU work, guarded to keep fewer than two rows per worker serial.
  Final eight interleaved rounds: prefill 371.65 -> 383.06 tok/s (+3.1%,
  overlapping ranges), decode 13.87 -> 14.68 (+5.8%, disjoint ranges).
  Same-run mx is 398.97 / 14.89: both floors remain open (-4.0% / -1.4%).
  Windows and Linux full suites pass with required Q8/Q4 HF fixtures; UBSan
  synthetic suite passes. Real F32 HF logits/excerpt/chunked NLL pass.
  Long-prompt full logits have maximum error 0.00012636 under 0.001, with
  all greedy IDs matching HF. Q8 step guard passes with overlapping ranges.
  The batched guard caught tiny-prompt scheduling regressions; after the
  serial guard, small-batch ranges overlap the control and B=64 retains a
  17% latency improvement. ASSETS and the row-scheduling JSON contain all
  initial/final samples and validation scope.
- **Left:** close both remaining external gaps before merge and hosted CI.
  Instrumented pool profiling finds 26-28 ms after worker callbacks finish
  during 32 decode steps. Bounded completion polling did not establish a win
  over its atomic-only control (eight rounds); no pool change was adopted.
  Full spinning was already rejected. Query scheduling remains scratch-only;
  the value-accumulation kernel below is the next validated checkpoint.
- **Gotchas:** single-row dots change reduction order. Printed sums alone
  are not the numerical gate. Initial unconditional scheduling appeared above
  mx for prefill, but final guarded measurements did not; use the final run.
  The legacy bench "prefill" is repeated step(), not batched prefill, so keep
  the separate batched regression guard. Reader alignment was not adopted.

### CPU attention value accumulation

- **Goal:** reduce CPU attention output loads/stores by retaining value-sum
  lanes in SIMD registers, while preserving per-lane summation order (#4a/#8).
- **Done:** normalized coefficients and register value sums implemented, with
  32-lane blocks, eight-lane remainders and scalar tails. Expanded the tiny HF
  fixture to head width 42, deriving its tensor shapes and HF config together.
  SIMD and forced scalar-value branches pass at maximum HF error 0.00000070
  under 0.00002; a missing SIMD-block output mutant fails at 0.19888665.
  Windows/Linux full suites pass with required real Q8/Q4 fixtures; Linux
  UBSan synthetic suite passes. Real F32 HF logits and excerpt/window NLL pass.
  All 5,013,888 logits over 1,943 prompt plus 32 greedy tokens are byte-identical
  to c4436fd; HF maximum error 0.00012636 under 0.001, greedy IDs 32/32 exact.
  Q8 step and batched guards show overlapping control/candidate ranges.
  Final eight interleaved rounds:

  | Mean tok/s | Control c4436fd | Value kernel | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Prefill | 384.86 | 397.63 | 397.94 | -0.08% |
  | Decode | 14.56 | 14.57 | 14.89 | -2.10% |

  Prefill improves 3.32% with narrowly overlapping ranges; decode is unchanged
  within noise. ASSETS and `benchmarks/attention-values-cpu-20260919.json`
  retain all samples, hashes, validation logs and reproduction harnesses.
- **Left:** close the external performance gap before merge/hosted CI.
  Instrumented profiling attributes about 2,073 of 2,187 ms decode to matmul,
  including 528 ms in the output projection; attention is about 87 ms.
  Investigate matrix operations next, using matched controls and HF gates.
- **Gotchas:** means close to mx are not proof of parity. This is an interactive
  workstation and no outliers were discarded. Query scheduling and polling
  remain scratch-only. The value kernel keeps sequence order per lane; byte
  identity is established for the recorded Windows long-prompt case, not all
  inputs or compilers. Forced scalar values do not prove no-AVX ISA support.

### CPU grouped projections

- **Goal:** reduce worker-pool dispatches for Q/K/V and FFN gate/up projections
  sharing activations, preserving float arithmetic (ROADMAP #4a/#8).
- **Done:** direct grouped decode through existing F32/Q8_0/Q4_K row kernels.
  Other types, batches and small jobs use the sequential fallback. No TLS,
  nested dispatch or activation conversion. Added CTest coverage to CI.
  Windows/Linux full suites pass with required real Q8/Q4 HF fixtures;
  real F32 HF and UBSan synthetic/backend checks pass. Backend tests cover
  540 cases and 141,750 outputs per platform; a missing-row mutant fails.
  Instrumented Q8 run observes 3,584 groups avoiding 5,376 dispatches across
  warmup plus measured sequences. Four excerpt/window NLL cases per model
  match the previous runtime exactly. All 5,013,888 long-prompt logits per
  model (F32 and Q8) are byte-identical to the previous runtime. F32 remains
  within the independent HF bound, with identical greedy continuation.
  Final eight interleaved rounds after a discarded warmup round:

  | Mean tok/s | Previous 5a9518c | Grouped | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8_0 prefill | 418.59 | 419.06 | 267.76 | +56.51% |
  | Q8_0 decode | 43.92 | 45.18 | 48.34 | -6.55% |
  | F32 prefill | 397.41 | 394.25 | 390.07 | +1.07% |
  | F32 decode | 14.47 | 14.64 | 14.86 | -1.49% |

  Q8 decode mean/median improve 2.86%/3.19%; ranges overlap due to one slower
  candidate sample, retained in the result. F32 prefill/decode changes are
  within overlapping ranges. Synthetic matmul mean 129.57 -> 127.64 GFLOPS
  (median 129.78 -> 129.51, ranges overlap); step-based prefill/decode improve
  4998/4928 -> 6861/6921 tok/s. Batched guards retain overlapping ranges except
  the faster six-thread single-token case. Raw samples, hashes, exact-output
  gates and reproduction sources are in ASSETS and
  `benchmarks/grouped-projections-cpu-20260919.json`.
  The follow-up real 8B Q8_0 diagnostic also preserves all 5,013,888 logits
  on its matched 215+32-token history. Three measured rounds, after warmup:

  | 8B mean tok/s | Previous | Grouped | mx |
  |---|---:|---:|---:|
  | Prefill | 29.12 | 29.25 | 21.20 |
  | Decode | 4.31 | 4.33 | 4.49 |

  Control/group ranges overlap; this does not establish a small speedup.
  Decode remains below mx. Exact-vector equality is against previous llmx,
  not an independent 8B HF baseline. Evidence and the full-vector control are
  in `benchmarks/grouped-projections-8b-20260919.json`.
- **Left:** close the external decode floors before merge and hosted CI.
  Native Q8 scale/load scheduling is implemented and validated below.
- **Gotchas:** exact equality is scoped to tested inputs/platform, not a
  full-corpus or maximum-context proof. The legacy bench prefill uses step();
  batched prefill has a separate guard. All outliers retained; do not pool
  absolute rates from separate sessions. Candidate is validated but unmerged.

### CPU Q8 scale / load scheduling

- **Goal:** reduce native Q8 decode instruction overhead while preserving
  float activations, per-lane accumulation order and exact weight scales (#8).
- **Done:** selected direct memory half broadcast plus direct byte-load sign
  extension. Assembly confirms the intended instructions; scalar fallback and
  accumulator order are unchanged. Feature specialization did not establish a
  further gain and is excluded. Exhaustive finite-half scale/signed-weight
  regression passes, and a wrong-half-offset mutant fails. Windows/Linux full
  suites with required Q8/Q4 HF fixtures, real F32 HF checks, backend CTest and
  UBSan synthetic/backend checks pass. Instrumentation confirms real Q8 use.
  Recorded F32/Q8 long vectors and excerpt/window NLL are exactly unchanged
  from `b6a890f`; the same-weight 8B vector comparison also passes.

  | Eight-round mean tok/s | Before b6a890f | Candidate | mx |
  |---|---:|---:|---:|
  | Qwen3-0.6B Q8 prefill | 422.99 | 418.43 | 273.51 |
  | Qwen3-0.6B Q8 decode | 45.30 | 46.54 | 48.35 |
  | Qwen3-0.6B F32 prefill | 398.17 | 395.23 | 394.92 |
  | Qwen3-0.6B F32 decode | 14.70 | 14.77 | 14.98 |

  Q8 prefill and F32 ranges overlap. Q8 decode improves but remains below mx.
  Root CLI and production comparator code hashes match their validated arms.
  The larger-model diagnostic also improves decode but does not establish
  external parity:

  | Qwen3-8B Q8 mean tok/s | Before | Candidate | mx |
  |---|---:|---:|---:|
  | Prefill | 28.97 | 29.04 | 21.13 |
  | Decode | 4.29 | 4.45 | 4.49 |

  The short synthetic guard's threaded single-token regression does not
  retain disjoint ranges in the longer follow-up. Paired changes run in both
  directions; the higher candidate mean remains visible:

  | Synthetic prefill mean ms | Before | Candidate |
  |---|---:|---:|
  | 1 thread, 1 token | 0.16995 | 0.12580 |
  | 6 threads, 1 token | 0.15864 | 0.16167 |

  ASSETS and `benchmarks/q8-scale-load-cpu-20260919.json` retain all samples,
  controls, hashes, HF/platform logs, prototype patches and reproduction code.
- **Left:** close the external decode floors before merge and hosted CI.
  Continue matrix-operation work with matched model measurements; retain the
  threaded tiny-prompt guard because zero slowdown has not been established.
- **Gotchas:** retain F16 subnormal/negative-scale behavior and scalar fallback;
  do not multiply activations by scales instead, which changes rounding.
  Exact equality is scoped to recorded inputs/platforms. The 8B equality check
  is against previous llmx, not an independent HF 8B reference. All samples
  are retained; no external parity claim.

### CPU comparison thread scaling

- **Goal:** locate the remaining CPU performance gap using matched thread
  counts and matrix-shape measurements, following ROADMAP #8.
- **Done:** explicit `--threads` in the comparator and runner, with requested
  counts echoed and checked. Windows llmx/mx and Linux llmx builds pass;
  invalid arguments and missing/wrong thread metadata are rejected. A real-model
  runner smoke passes. Matched Q8/F32 scaling is complete; those measurements
  used runtime `475f312`, before the KV and worker-error changes.
  Full vectors are byte-identical across all measured counts; continuous NLL
  is unchanged from the prior runtime and passes the independent HF fixture.
  Projection and matrix probes are complete. F32 matrix ranges overlap mx;
  Q8 matrix latency remains higher at the default comparison count. ASSETS and
  `benchmarks/cpu-thread-scaling-20260919.json` contain complete results.
- **Left:** merge the tool with the validated stack once its external floor
  is met. The resulting contiguous per-head KV change is implemented and
  validated below; it is no longer a pending experiment.
- **Gotchas:** use the same thread count in both arms and record it with every
  result. Keep the pinned model, tokens, reference revision, warmup and KV
  settings. Scaling diagnostics do not waive the existing external floor.

### Head-major CPU KV storage

- **Goal:** make each KV head's history contiguous to improve attention reads,
  preserve arithmetic order, and separate concrete CPU storage from logical
  sequence state without adding speculative device/server interfaces.
- **Done:** `HostKVCache` owns bounded growth and token-major projection writes;
  `Model` owns valid length/reset; backend attention receives an explicit head
  stride. Promoted the exact validated headers from the scratch candidate.
  Growth/reset/mixed histories match all 229,758 control values on Windows
  and Linux (the latter with nonrecovering UBSan). The direct storage oracle
  is in CTest and rejects a wrong-head relocation mutant.
  Real F32 HF logits and continuous/window NLL pass; long HF error is at most
  0.00012636185 under 0.001, with all 32 greedy IDs matching. F32 and Q8 each
  retain all 5,013,888 long-history logits and four full-precision NLL cases
  exactly versus `475f312`. Windows/Linux full suites with required real HF
  fixtures and Linux UBSan native/synthetic suites pass. Integrated MSVC code
  has the same `.text` hash as the validated candidate; final native CTest
  integration passes on Linux and UBSan.
  Nine interleaved matched rounds, all outliers retained:

  | Mean tok/s | Control | Head-major KV | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8 prefill | 366.46 | 416.37 | 262.34 | +58.72% |
  | Q8 decode | 42.36 | 44.41 | 45.51 | -2.42% |
  | F32 prefill | 345.42 | 361.38 | 362.44 | -0.29% |
  | F32 decode | 13.06 | 13.47 | 13.27 | +1.55% |

  Candidate/control ranges overlap; paired candidate wins are 9/9, 7/9,
  7/9 and 8/9 respectively. These results support an incremental selection,
  not a claim that the external floors are closed. The separate tiny synthetic
  decode mean declines 2.88% with overlapping ranges; that remains a recorded
  limitation. ASSETS and `benchmarks/head-major-kv-cpu-20260919.json` preserve
  all samples, hashes, scopes and reproduction sources. Earlier short-run
  diagnostics remain in `head-major-kv-initial-20260919.json`.
- **Left:** close the remaining external Q8 decode and F32 prefill gaps, then
  merge the validated stack and observe hosted CI. No merge or publish yet.
- **Gotchas:** reset retains allocation but must not expose stale tokens.
  Growth temporarily holds old and replacement storage together; future
  multi-user memory budgets must account for that peak. A CPU head-major layout
  is not a requirement for future device buffers, paging or shared prefixes.

### Chat follow-up cache validation

- **Goal:** preserve correct conversation history across follow-up prompts,
  reusing KV only when its exact token prefix matches the rendered transcript.
- **Done:** review found that `cmd_chat` skips cached tokens by count alone,
  renders twice per turn, and can pass empty logits to generation when the
  template does not add a generation suffix. Generated stop tokens and the
  unconditional EOS step also need accurate cache accounting.
  The new HF-backed chat regression reproduced a crash on the old binary.
  It also exposed double consumption of template block terminators, which
  skips adjacent content and breaks nested conditionals/loops.
  Implemented exact fed-token tracking and reset/refill for changed prefixes;
  removed unconditional EOS insertion and the redundant prefill/render pass.
  Fixed block terminator consumption and first keyword argument parsing. The
  latter affected Qwen namespace state and removal of old reasoning. New
  end-to-end HF reply fixtures pass; a token-count-only reuse mutant fails.
  The real Qwen template matches Jinja2 across initial and follow-up histories.
  Windows and Linux full suites with required real HF fixtures pass. MSVC
  renderer test, Linux CTest and Linux UBSan native/synthetic suites pass.
  Linux first exposed a missing `<cmath>` include in the standalone renderer;
  that is fixed. Chat coverage is 54 runs of nine scenarios across thread and
  batch settings; the real template has 12 independent Jinja2 cases. Logs,
  fixture hashes and reproduction commands are in
  `benchmarks/chat-followup-validation-20260919.json`.
- **Left:** merge with a validated runtime stack and observe hosted CI. The
  enclosing stack still needs its external performance floors; chat fixes do
  not establish those floors. Numerical kernels and model forward paths are
  unchanged by this fix.
- **Gotchas:** prefill continuation tests alone do not validate chat-template
  reuse. Keep messages separate from cached tokens; rendered text may change
  earlier turns or retokenize their boundary. This is single-user chat, not
  multi-user or concurrent request support.

### Correctness baseline vs HF reference

- **Goal:** give the suite an external ground truth. Correctness is measured
  against the HF reference, never against llmx itself (`docs/ROADMAP.md` #8).
- **Done:**
  - The tokenizer mode of `tools/gen_baseline.py` emits golden fixtures using
    `tokenizers` + `huggingface_hub`; numerical modes additionally need torch
    and transformers. Tokenizer output is committed to
    `tests/data/baseline_tokenizer.json`.
  - `tests/baseline.py` compares llmx against the committed goldens and is
    wired into `tests/run_tests.py`. It SKIPS when no fixture model is on disk,
    so the rest of the suite still runs anywhere.
  - Tokenizer parity: **20/20 cases match pinned Qwen/Qwen3-0.6B.**
  - HF fp32 top-10 logit goldens for six prompts landed in `1c2e102`.
    The baseline checks top-1, top-5 set overlap and a magnitude bound on
    Qwen3-0.6B Q8_0 and mixed Q4_0 fixtures. Both passed all six prompts in
    that commit; the mixed fixture catches the f16 subnormal regression.
  - Local golden generation works in an isolated environment with
    numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2.
  - Pinned HF fp32 PPL golden for a 247-token wikitext excerpt: exact token
    IDs/count and finite NLL/PPL checks, with per-quant absolute NLL bounds.
    HF PPL 28.7974; Q8_0 28.8371 (NLL delta 0.001374 <= 0.01); mixed Q4_0
    32.8463 (delta 0.131554 <= 0.16). Both llmx arms repeated identically at
    printed precision. Generator, provenance and bounds are in `docs/ASSETS.md`.
    Eight injected bad-output cases were rejected; build and full suite pass.
  - Nine bugs found and fixed via this path, all of which survived a green
    suite: attention missing 1/sqrt(head_dim); temperature cancelling in the
    sampler; RoPE read past context_length; the GPT-2 whitespace guard that
    could never fire; attention width hardcoded to n_embd; tied embeddings
    unsupported; Windows argv delivered in the ANSI codepage so any non-ASCII
    prompt was mangled before llmx saw it; and the pretokenizer implementing
    the GPT-2 regex instead of the Qwen2/Qwen3 one; the byte encoder incorrectly
    including soft-hyphen byte 0xAD in its printable set. Four new HF cases
    reject the preserved old encoder. Whole-wikitext file input now tokenizes
    298,938 tokens and scores the requested window limit successfully.
- **Left:**
  - Per-layer activation and full-corpus PPL goldens, plus maximum-context validation.
    The 1,943-token/32-step independent F32 HF check is complete; its scope and
    full-vector quantized diagnostics are recorded in ASSETS.
    Continuous and chunked excerpt gates are implemented, with an explicit
    disjoint-window scoring policy (`docs/USAGE.md`). They are not full-corpus
    coverage. Two-token windows show large quantized/HF deviations even under
    the unchanged old scorer; diagnostics and bounds are in `docs/ASSETS.md`.
    The existing ranking gate does not bound full-vector numerical error.
  - Existing tiny F32 full-vector and real-model excerpt bounds are implemented.
    Broader Q8 full-vector and full-corpus acceptance bounds remain open.
- **Gotchas:**
  - Round-trip and synthetic tests alone are not an external correctness gate;
    preserve the HF tokenizer/logit checks and extend their coverage.
  - The 8B hides bugs the 0.6B exposes: `n_head * head_dim == n_embd` holds for
    Qwen3-8B (32*128 == 4096) and fails for 0.6B/1.7B/4B.
  - torch is a fixture-GENERATION dependency only, never needed to run the
    suite and never at runtime.
  - F32 embedding/matrix support and a tight HF numerical gate are implemented
    on the active feature branch. Its external performance gate remains open;
    see the F32 block above before merging.
  - Generation's legacy reasoning filter searches `thinking_start/end`, not
    Qwen3's actual `<think>` / `</think>` markers. Its docs now state that limit.
  - The CLI thread-settings block records the validated auto/prefill/decode
    corrections found during the documentation review.
  - JSON syntax/Unicode and conversion tensor dimensions/extents are implemented
    in their active blocks above. Qwen construction now validates configuration
    geometry and required tensor layouts; token/request checks remain separate. See
    docs/src/core-json.md.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Preceding two-arm result (2026-09-20):** unchanged validated `bf122fd`
  runtime versus pinned mx `5542318e74`, nine alternating measured pairs per
  model after one discarded warmup pair; six workers, ubatch 128, 215 prompt
  plus 32 forced tokens and F32 KV. Every process exits zero and llmx final
  output hashes repeat within each model. No timing overlaps other compute.

  | Q8 model / phase | llmx mean tok/s | mx mean tok/s | Mean gap | llmx / mx median | Paired wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 440.479 | 274.766 | +60.31% | 448.159 / 277.473 | 9/9 |
  | 0.6B decode | 47.280 | 47.589 | -0.65% | 47.362 / 48.315 | 3/9 |
  | 8B prefill | 30.170 | 22.124 | +36.37% | 30.481 / 21.994 | 9/9 |
  | 8B decode | 4.403 | 4.440 | -0.82% | 4.465 / 4.501 | 2/9 |

  Both decode means and medians remain below mx. The external gate remains
  open; close results do not meet the required floor. Preserve all measured
  rounds, including the slower first measured round, and do not pool older
  sessions. Evidence: `benchmarks/q8-current-floor-20260920.json`.
  Independent review of all 27 pending commits found no additional concrete
  publication blocker; the existing Windows/Linux HF/native evidence remains
  valid for this unchanged runtime. Hosted CI follows eventual publication.
  Earlier ordered-prefill comparisons also exceed mx in F32 prefill/decode;
  their initial and Q8 follow-up sessions retain their separate scope.
- **Earlier instruction study (no runtime change):** paired native Q8 rows regress;
  direct pointer increments and explicit row-kernel inlining do not establish
  a decode gain. Exact row/tail/fallback checks pass. Assembly confirms shared
  activation loads without inner-loop spills, removal of native-loop address
  multiplication, and removal of row calls in the respective prototypes.

  | Separate Qwen3-0.6B Q8 studies, mean decode tok/s | Control | Prototype | mx |
  |---|---:|---:|---:|
  | Paired rows | 46.00 | 43.81 | 48.31 |
  | Pointer increments, longer run | 46.22 | 45.86 | 48.22 |
  | Explicit inlining | 46.56 | 46.04 | 48.46 |
  | Inlining plus pointers | 46.56 | 46.38 | 48.46 |

  No prototype is adopted. Do not pool absolute rates across these sessions.
  ASSETS and `benchmarks/q8-row-instructions-20260919.json` retain patches,
  assembly, exact checks, complete samples, code hashes and reproduction.
  These scratch prototypes did not enter the full HF/platform adoption gate.
- **Previous investigation:** AVX2 integer dots with vectorized activation packing
  were tested at 8-bit and 16-bit precision. Q16 improves matched mean decode
  by 2.79% on 0.6B and 4.06% on 8B, but still trails mx by 5.12% / 1.21%.
  Both variants pass existing Windows HF fixture bounds; Q16 stays much closer
  to the current float path. Independent packing/product controls pass,
  including signed weight extremes, half subnormals and fallback cases.
  Across four excerpt/window cases, Q16's maximum absolute NLL change versus
  current llmx is 0.00003155. Across 5,013,888 logits on a 1,943+32-token
  forced-HF continuation, maximum change is 0.002213; maximum and RMS error
  against HF are slightly lower in this case. This is a nonzero precision
  change, not a lossless result. Full corpus, independent 8B HF, maximum
  context and cross-platform validation remain open. No candidate was adopted;
  all patches, samples and numerical controls are preserved in
  `benchmarks/q8-integer-activation-20260919.json` and ASSETS.
- **Earlier investigations:** paired F32 decode rows did not improve throughput;
  packed F32 prefill variants regressed. None was adopted. Exact patches,
  samples and diagnostics are in ASSETS and the paired-decode/packed-prefill
  benchmark JSON files. These experiments used the `5a9518c` runtime.
  Profiling scalar exponentials finds only 7.45 ms SiLU wall time and 9.42 ms
  summed softmax worker time during 2,237.50 ms decode; these are not the main
  remaining cost. The summed worker measurement is not wall time.
  The preceding matched Q8 comparison covers both the HF fixture and real 8B
  model against pinned public mx `5542318e74`: three alternating pairs,
  215+32 pinned tokens, six threads, ubatch 128 and F32 KV.

  | Q8_0 model / phase | llmx mean tok/s | mx mean tok/s | Gap |
  |---|---:|---:|---:|
  | Qwen3-0.6B prefill | 380.50 | 265.35 | +43.39% |
  | Qwen3-0.6B decode | 42.73 | 46.81 | -8.71% |
  | Qwen3-8B prefill | 25.94 | 20.07 | +29.27% |
  | Qwen3-8B decode | 4.10 | 4.39 | -6.64% |

  Prefill exceeds the reference on both models, but decode remains below it;
  both comparisons have disjoint arm ranges in each phase. Raw samples,
  hashes, flags and scope are in `benchmarks/q8-external-floor-20260919.json`.
  Source inspection confirms mx uses quantized Q8 activations and integer
  dots, while llmx retains float activations. Any analogous optimization
  needs a measured numerical cost bound before adoption.
- **Historical stand-in comparison (different conditions; not pooled):**
  Qwen3-8B Q8_0, 343-token wikitext prompt, -t 16, this workstation. Reference
  is the CPU AVX2 llama.cpp shipped with LM Studio, stock `llama-server`, same
  machine, so no time on the Linux machine was used.
    - pp   llmx 37.23 / 37.86   llama.cpp 37.70 / 37.34   -> parity
    - tg   llmx  3.91           llama.cpp  4.99 / 5.02    -> 22% under
  These are the original matched measurements. Later Q8_0 decode commits
  report 4.12 tok/s with independent accumulators (`54ea063`) and 4.24/4.18
  with F16C (`0c12570`); these are not a new matched mx-llama.cpp comparison.
  Separate decode/prefill thread flags landed in `dada6c0`.
- **How prefill got there, 3.89 -> 37.5 tok/s (9.6x), each step A/B measured:**
  - persistent worker pool instead of spawning threads per call (decode -23%)
  - attention through `Backend::parallel_for` instead of its own threads
  - batched prefill: matrix-matrix instead of one token at a time (3.89 -> 13.2)
  - dequantize each weight row once per batch, not once per column (-> 15.3)
  - four independent accumulators in the f32 dot (-> 17.0)
  - fused 4-row kernel sharing one activation load (-> 24.0)
  - two activation columns per four rows, 0.75 loads/FMA (-> 33.7)
  - three activation columns per four rows, 0.58 loads/FMA (-> 37.5)
- **The lesson worth keeping:** the kernel was LOAD bound, not FMA bound. Each
  naive dot needs 2 loads per FMA and Zen3 sustains about 2 loads/cycle against
  2 FMAs/cycle, so it ran at half of FMA peak no matter how the batch was
  blocked. Every win after the first came from raising the FMA:load ratio.
- **Left:**
  - Close the current Q8 decode deficits while retaining prefill gains. The
    latest fixed `bf122fd` control/candidate/mx session measures control mean
    gaps of -2.01% (0.6B) and -0.67% (8B), with both medians below mx. The
    preceding two-arm session measured -0.65%/-0.82%; keep the sessions separate. Earlier bandwidth/thread observations and null allocation,
    fragmentation and prefetch experiments do not predict this current gap;
    mmap has not been established as a throughput improvement.
  - Grouped Q16 activation packing is now rejected. Its own scalar/integer
    arithmetic and activation tests passed, but unchanged native accuracy
    checks fail on MSVC and GCC (control 7/7, candidate 6/7). First diagnosed
    absolute error is 0.000219106674 against the existing 0.000206180004 limit,
    on a standalone one-worker Q8 dot. No test tolerance was changed; model
    HF-cost/performance runs were stopped before execution. Evidence:
    `benchmarks/q16-group-native-rejection-20260920.json`. Bounded-cost research
    does not satisfy AGENTS' lossless requirement. Preserve the rejection;
    do not tune a new tolerance to that failing case.
  - The exact Q8 horizontal reduction is rejected by complete timing despite
    passing numerical gates. Read-only review also found two redundant `h_`
    clears in `Model::step`, but no evidence that their cost closes the gap.
    The native two-block lambda is rejected at the no-spill codegen gate.
    The ordinary inner-loop follow-up also fails the useful-scheduling gate:
    no spills, but one extra feature reload and no useful cross-block work.
    Unrolling remains closed. Caller attribution and native sampling are
    complete above; neither established a recoverable production cost. The
    separately reopened prefill-placement assessment remains active. Prior
    F16C specialization, pointer increments and row pairing are nulls.
  - Thread and matrix-shape diagnostics are complete (see CPU comparison
    thread scaling above). F32 matrix ranges overlap mx, while Q8 matrix
    latency still trails it. The resulting head-major KV layout is now validated.
    Further changes should follow profiling of the current runtime, not repeat
    completed instruction/dispatch studies. The larger-model diagnostic did
    not close the external gap.
- **Gotchas:**
  - Synthetic `bench` throughput does not establish real-model speed. Small
    projections may stay serial depending on thread count. Grouping improves
    the six-thread synthetic case, but the real-model external floor still
    fails; use the matched model measurements.
  - Do not tune the row block as a byte budget. Measured worse at every size
    (64/128/196/256 KB gave 22.37/22.04/23.68/21.12 against 24.04 for a flat
    4); the knee follows the fused kernel width, so it is `DOT_ROWS`.
  - ubatch barely matters once the kernel is right, and 343 vs 512 on a
    343-token prompt is the SAME computation - do not read noise as signal.

### Device execution model (ROADMAP #4a)

- **Process note:** this block was opened after the code was written, which
  `AGENTS.md` forbids.
- **Goal:** land the backend-agnostic execution model #4b depends on, in the
  six steps of `docs/DEVICE-EXECUTION.md`. Bar per step is no measured
  regression, not a win.
- **Done:** the design, `b1e4904`. Step 1, weights resolved once at load,
  `edd617f`: suite green, HF logits and PPL unchanged on Q8_0 and Q4_0. Doc
  page refreshed in `1d4ffa4`.
- **Done:** step 2, batched elementwise ops, committed on the feature branch.
  Suite green and HF perplexity bit-identical to step 1.
- **Done:** step 2 measured under protocol after 57294 went terminal, with the
  plan, advance rule and contamination criteria frozen and hashed before any
  timing. Nine measured pairs, arms alternating and reversing, every arm under
  `monitor_windows.py`, no sample dropped. **Result: does not advance.**

  | Phase | base mean | cand mean | Mean | Median | Baseline wins |
  |---|---:|---:|---:|---:|---:|
  | Prefill | 337.91 | 346.55 | +2.56% | +2.66% | 2/9 |
  | Decode | 27.48 | 27.69 | +0.79% | -0.50% | 5/9 |

  Prefill passes every criterion. Decode fails paired wins (5/9 against a
  rule of <= 4) while its mean and median both sit inside the 1% noise band,
  so that phase is better described as indistinguishable than regressed. The
  rule is not weakened and the run is not repeated to obtain a pass. Evidence:
  `benchmarks/device-exec-step2-20260920.json`.
- **Done:** all six steps, merged and gated (`docs/DEVICE-EXECUTION.md`, and
  the "Device execution model complete" block above); the status table row
  is `Done`. The Left lines below are the open items as of step 2, kept as
  history and since resolved.
- **Left (as of step 2):** decide step 2's disposition. The frozen rule is stricter than
  `AGENTS.md`'s own tradeoff principle, which says a large gain can justify a
  minor loss and warns against rejecting on an isolated per-case cutoff. That
  tension is a judgement call and must not be resolved by editing the rule
  afterwards. Options: re-measure with more pairs under a NEW prospective plan,
  or keep decode on the single-row ops so only prefill changes.
- **Left (as of step 2):** step 1 still has no admissible measurement of its
  own; it was timed off-protocol during 57294 (13:33:05-13:52:07 +0300, disclosed at the time).
  Expected neutral, unproven.
- **Left (as of step 2):** the external mx-llama.cpp floor is untouched by
  this runner and still applies to any step that advances.
- **Left (as of step 2, since done):** steps 3-6 (buffers, arena, KV on
  buffers, sync) untouched. No vendor backend is writable until step 6. Step 3 was started as interface
  plumbing only and reverted: `Buffer` with no caller is a speculative seam,
  which `AGENTS.md` forbids. It must land together with the weight conversion
  that uses it, which means touching `arch_qwen.hpp` and waiting for the other
  developer's
  stack.
- **Gotchas:**
  - Step 2's first decode reading of +3.40% was an artifact: its two winning
    pairs were the two lowest-throughput rounds, the other six gave +0.63%.
    Decode runs B=1 and cannot benefit. Prefill is the real effect, 8/8 paired
    wins. All samples retained.
  - Not a SwiGLU result. The screened-out "Exact decode SwiGLU callback fusion"
    measured a different change and its verdict stands.

### Fused Q5_K and Q6_K row dots (ROADMAP #1)

- **Goal:** give Q5_K and Q6_K the decode row dot Q4_K already has, so no
  dequantized value is materialised. Q4_K works because `d*q - m` factorises
  into `d*sum(q*x) - m*sum(x)`; Q5_K carries the same scale/min pair plus a
  high bit, and Q6_K has signed group scales and no min.
- **Done:** Q5_K fused dot, adopted. `dot_row_q5_K` mirrors `dot_row_q4_K`
  with the fifth bit taken from `qh`, whose mask shifts two places every 64
  values while `qh` itself does not advance. Dispatch now covers both types
  through one threading block rather than a copy.
- **Done:** correctness against the fp32 HF golden, not against llmx. The
  ordinary suite's fixtures are Q8_0 and Q4_0, so it does not cover this path
  at all; `Qwen3-0.6B-Q5_K_M.gguf` was scored directly against
  `tests/data/baseline_logits.json`: top-1 6/6, worst top-5 overlap 4/5, mean
  4.50, max absolute logit 24.46 against the 100 bound. The file is 29 tensors
  of Q6_K as well, so that path is exercised incidentally.
- **Done:** throughput, plan frozen before timing, 15 measured pairs.

  | Phase | base mean | cand mean | Mean | Median | Baseline wins |
  |---|---:|---:|---:|---:|---:|
  | Decode | 8.424 | 12.751 | +51.369% | +51.12% | 0/15 |
  | Prefill | - | - | -0.93% | -0.64% | 10/15 |

  Evidence: `benchmarks/fused-q5k-decode-20260920.json`.
  **Corrected 2026-09-20 by an independent recomputation of all 15 pairs**: the arm
  means were 8.424 and 12.751333 tok/s, not 8.96 and 13.33. The paired
  +51.369% is unchanged, and Q6_K recomputes to +68.0186%. The original
  figures were read from a single round rather than the arm means.
- **Done:** the matched external floor, which the above does NOT establish.
  `tools/compare_cpu.py`, mx `5542318e74`, six threads, eight matched rounds,
  identical committed HF token IDs in both arms:

  | Phase | llmx | mx | Ratio | Gate |
  |---|---:|---:|---:|---|
  | Prefill | 324.33 | 230.33 | 1.41x | above |
  | Decode | 15.98 | 58.30 | 0.27x | **BELOW** |

  Q5_K decode does not meet ROADMAP #8. The fused dot is a real 51% gain over
  the previous llmx build and still leaves mx 3.65x faster. Reporting the
  self-comparison alone would have read as success.
  Evidence: `benchmarks/q5k-external-floor-20260920.json`.
- **Done:** located the gap. Same machine, session, binaries and harness, two
  quants:

  | | llmx | mx | ratio |
  |---|---:|---:|---:|
  | Q8_0 decode | 40.93 | 41.44 | 0.99x |
  | Q5_K decode | 15.98 | 58.30 | 0.27x |

  Decode is bandwidth bound, so Q5_K (about 0.69 bytes/weight) should beat
  Q8_0 (about 1.06). mx does that: 41.44 -> 58.30, +41%. llmx goes backwards:
  40.93 -> 15.98, -61%. A smaller quant making llmx slower means the K-quant
  decode path is limited by per-weight unpacking and float conversion, not by
  memory traffic. Q8_0 sits at parity because its fused dequant+FMA matvec is
  cheap enough to stay bandwidth bound.
  Evidence: `benchmarks/kquant-decode-bound-20260920.json`.
- **Left:** prefill work cannot close this. Both quants are already above the
  floor in prefill, 1.50x and 1.41x in the same runs. The whole gap is K-quant
  decode.
- **Left:** test whether quantizing the activation vector to 8 bits and taking
  an integer dot product restores bandwidth-bound behaviour, as mx does. That
  is a much larger change than a fused dot, affects every quant type, and
  changes arithmetic, so it needs its own prospective plan and its own HF
  gate. It is the highest-value remaining decode work.
- **Done:** the same treatment for Q6_K, which has signed group scales and no
  min, so the dot is `sum(d_g * sum(q*x))` with no `sum(x)` term:
  `dot_row_q6_K`, adopted at +68.02% paired decode over 15 pairs, 0/15
  baseline wins. Evidence: `benchmarks/fused-q6k-decode-20260920.json`.
- **Done 2026-09-21:** `Qwen3-0.6B-Q5_K_M.gguf` is in `BASELINE_MODELS`
  (see the Vulkan block above), so the external gate covers Q5_K/Q6_K
  permanently on both backends. It is a third download for
  `tools/fetch_test_models.py`, 419 MB from the same repo and revision as
  the Q4_0 file.
- **Gotchas:**
  - Prefill does not benefit and cannot: the fused dot only fires at
    `nbatch == 1`, and the batched path already dequantizes each row once and
    reuses it across the batch. That made prefill an accidental control in the
    Q5_K run - identical code measured -0.93% mean and -0.64% median, which
    puts this setup's noise floor near 1%. Useful when reading small effects.
  - Qwen3-0.6B-Q5_K_M is 168 Q5_K / 29 Q6_K / 113 F32, so it exercises both
    paths in one model. A fused Q5_K dot alone will show a partial effect.
  - The roadmap's 2.16 -> ~2.6 tok/s is Q4_K's result on its own model, not a
    target for these types.

Nothing else is in flight. When you start a feature, open a block above
before writing code - see `AGENTS.md` -> "Starting a feature".
