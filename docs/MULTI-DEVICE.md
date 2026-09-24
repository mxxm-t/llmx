# Multi-device execution: layer, tensor and staged splits

Design for ROADMAP #5, written before any code and open for review. It extends [EXECUTION](EXECUTION.md), whose placement, tickets and crossing already run a model over two backends, and replaces its decision to drop tensor-parallel work: that returns here as the tensor group, built after the layer split and designed with it. The server side extends [SERVER](SERVER.md).

## Why, and what must hold

Models past one card's memory cannot be run or gated today (Qwen3-32B Q8_0 at about 35 GB, Qwen3-30B-A3B Q8_0 at 32.5 GB without CPU experts, Qwen3-235B-A22B at about 142 GB in Q4_K_M), and a server with many users should turn every added card into throughput. The design is judged on the multi-user case first.

Gates, all required before a phase merges:

1. **No regression** on any current path: single device, CPU, CPU experts beside a device, streamed experts, the server. Interleaved A/B against main with a layout control, on both machines.
2. **Correctness.** A layer split is bit-identical to one device: logits over every position and greedy text, for prompts, decode and a mixed server pass. A tensor group passes the HF gate within its bounds, is identical run to run, and computes a prompt the same however it is batched. Server and CLI give the same greedy text within one placement at every concurrency.
3. **Ahead of llama.cpp by a wide margin** on the same cards, model and quant: `llama-bench` prefill and decode, and its server under the same load.
4. **Ahead of vLLM by a wider margin** on vLLM's own serving metrics: request and output throughput, time to first token and inter-token latency at p50 and p99, end-to-end latency, over a sweep of arrival rates and concurrency, with vLLM's benchmark method. vLLM runs on gfx906 only through a community fork and prefers 4-bit AWQ or GPTQ weights to GGUF, so the comparison is at equal bits on the same model and the format difference is reported with every table.

## The hardware, measured

On the Linux machine, with ten MI50 32 GB cards, an EPYC 7262 (8 cores, 16 threads) and 62 GB of RAM, from earlier work with mx-llama.cpp on the same machine:

- Every card sits behind its own switch on a PCIe Gen4 x8 root port, with no XGMI. Host to device and back run at 14.3-14.4 GB/s, and a HIP peer copy between any two of the 56 pairs at 14.3 GB/s. Link training is random per boot and a card can come up at Gen1 or Gen3, so the link speed is checked before any timing.
- Peer to peer works under HIP: DMA peer copies at full link rate, and kernel peer stores at 3-6 GB/s for 16 KB messages with 6-17 us latency, correct only with fine-grained memory, which makes other kernels reading that memory slower. Vulkan has no peer copy between devices; everything crosses through host memory.
- Memory reads at 805 GB/s. A dependent dispatch costs 4.1 to 5 us whatever it does, which bounds decode by dispatch count once the rows are fast (`docs/VULKAN.md`, kernel notes).
- Two streams on one card do not overlap real kernels (1.02x), so a card runs one pass at a time and overlap comes from different cards.

The Windows workstation has one Radeon VII (16 GB). There the only second device is the CPU, so a layer split is the only split it can use, and the multi-card gates run on Linux.

## Three splits, one placement

- **Layer split.** Consecutive layers on different devices. Only the residual crosses a boundary, `rows x n_embd` floats per pass: 20 KB for a decode token of a 5120-wide model, 10 MB for a 512-row chunk (0.7 ms at link rate). Arithmetic is unchanged, so results are bit-identical.
- **Tensor group.** Every layer on several devices at once: q, k, v, gate and up split by output rows, the attention output and down by input columns, attention by heads, each member keeping the KV of its heads, and two sums over the group per layer. Each member reads a share of the weights, so one request decodes faster. The sums cost two synchronizations per layer.
- **Staged tensor.** Stages of a layer split, each stage a tensor group.

One description covers all three, so the scheduler and the model code do not branch on the mode:

```
Group      an ordered set of devices that run the same layers together; width 1 is a single device.
Stage      a contiguous range of layers on one group; the embedding and the head are placed as roles of their own.
Placement  the stages in layer order, plus where the embedding, the head and each layer's feed-forward block run.
```

A layer split is stages of width-1 groups, a tensor split one stage of width N, and staged tensor anything between. Today's `Placement` (a device per role per layer) is the width-1 case and keeps working: experts on the CPU beside a device are a layer's feed-forward block on another group, as now.

### Which split for which case

| Case | Best split | Why |
|---|---|---|
| Model fits one card, many users | Replicas (one model copy per card or group) | No communication at all. Four width-2 replicas gave 4.43 times the throughput of one width-8 group on this hardware |
| Too big for one card, many users | Layer split with passes in flight | Throughput grows with cards once enough passes are in flight: 20 to 122 tokens/s from 1 to 16 concurrent on 8 cards |
| Too big for one card, few users | Tensor group of 2 to 4 | Per-request decode scales: dense 27B 21.5 to 46.8 tokens/s on 1 to 4 cards. Layer split stays at one card's speed (22.3 to 21.3) |
| Long prompt | Layer split, prompt chunks pipelined | A prompt's chunks flow through the stages like separate passes |
| Mixture of experts | Layer split; expert tiers where a model barely does not fit; see Mixture of experts | Below |

Group widths beyond 4 lose on this hardware: the sum's wait is the slowest of N-1 PCIe latencies, 8.2 us at 4 cards and 20.3 us at 8, which is 27 percent of an 8-card decode token.

## Multi-user execution

### Passes in flight

A pass is one forward over a batch through every stage. With one pass at a time, each stage idles while the others work, and a layer split is no faster than one card. So the server keeps **P passes in flight**, each a separate batch of different sequences, one per stage and one being sampled:

```
stage 0:  [A][B][C][A'][B'] ...
stage 1:     [A][B][C][A'] ...
host:           sample A, form A'
```

Each stage runs its passes in order. P is the stage count plus one (the plus one hides the host's sampling and batch assembly), and it is not a flag, because the stage count determines it. A sequence is in at most one pass at a time, since its next token needs its logits.

This is between-step filling, one scheduler and one KV manager for all passes, as vLLM V1 does it (`max_concurrent_batches` = pipeline size). Two alternatives were measured on this hardware and are not taken: static per-engine request partitions (V0 virtual engines, +19 percent at K=4 but a fixed split of slots and caches) and in-step ubatch pipelining of one batch (+21 percent). The two did not compound: they compete for the same requests.

### Decode

Decoding sequences are spread across the passes in flight, each sequence staying in its pass while it decodes, and new decoders going to the pass with the least work. Merging all decoders into one pass would leave all stages but one idle; spreading them keeps every stage busy, and a card reads its weights once per pass for all the pass's rows. With fewer decoders than P, the passes are fewer and some stages idle, which is the case a tensor group serves better (below).

### Prefill beside decode

A pass carries decode rows first and then prompt chunks, as the server does today, but its size is set by **predicted stage time**, not only by a token count. A pass whose chunk takes twice as long stalls every pass behind it at every stage, so each pass is assembled to about the same predicted time, and to at most the inter-token latency the decoders are promised. The cost model per stage is measured at load and corrected from observed pass times: decode rows cost about their weight reads, prompt tokens their compute plus attention over their history.

A long prompt's chunks go to consecutive passes, so chunk i+1 enters stage 0 as soon as chunk i leaves it. Its layers at stage 0 need only chunk i's keys and values at stage 0, which are complete by then. A chunk later in a long prompt attends to more history and costs more, so chunk size shrinks as the prompt grows to keep the stage time level, as SGLang's dynamic chunking does. Prompt boundaries depend on the prompt alone, never on what else is in flight, so a prompt computes the same however it is scheduled.

### More requests than stages

This is the normal case. Passes grow wider rather than more numerous: P stays at the stage count plus one and each pass carries more sequences, up to the KV budget. Past the point where a pass turns compute-bound, around 35 rows on an MI50, wider passes stop being free, and the scheduler caps pass width by the same predicted time.

### KV, admission and the arenas

Each device keeps the KV of the layers it runs (per head within a tensor group), in its own pool, as now. Admission checks every pool a request touches, and the tightest one decides. Prefix reuse keeps one block table per storage, as now. Each pass in flight needs its own activation arena on every device, so memory is `weights + KV + P x arena`. The fit is computed at load from each device's reported free memory, and when P passes do not fit, the server says so and runs with fewer. It never silently falls back to one pass in flight.

## Execution structure

Layers as in ARCHITECTURE: the model knows stages, the inference layer runs them, and the server only schedules.

- **Model** (`model/`). `forward` becomes three calls: `begin(ctx, entries)` (positions, block reservations, the embedding), `run_stage(ctx, s)` (one stage's layers, the handoff out of it) and `finish(ctx)` (the head). `forward` stays as those three in a row, so the CLI and every test run unchanged. A pass commits its sequences' histories when its last stage is enqueued. A failure in a later stage fails the pass's requests and releases their blocks after the stages' tickets retire.
- **Pipeline** (`inference/pipeline.hpp`). One host thread per stage, and the only thread that drives that stage's backends, which keeps the one-thread-per-backend contract. Each takes passes from its queue in order, runs `run_stage`, and hands the pass to the next stage's queue. A CPU stage computes on its own thread, so CPU and GPU stages overlap too. The CLI's long prompts use the same pipeline.
- **Handoff.** Phase 1 keeps today's crossing, a blocking read and a write, which is correct and slow. The pipelined handoff needs no new backend call: the source copies the residual into a host-visible buffer with the existing `copy`, the stage thread waits that submission's ticket, and the next stage `write`s it. Zero-copy variants, host memory imported into both devices or a device-side wait on the source's semaphore, are backend-private improvements measured in phase 0. A ROCm backend adds a peer copy when it exists. On HIP, cross-device stream waits were unreliable with more than four hardware queues, so that path gets a dedicated copy stream and is checked for ordering.
- **Scheduler** (`server/scheduler.hpp`). P execution contexts, batch assembly by predicted time, admission over several pools, and cancellation and pausing only for sequences not in flight.
- **Backend** (`backends/`). One addition in phase 1: `memory_available()`, the device's free memory (Vulkan through `VK_EXT_memory_budget`, CPU from the OS), which the fit needs. The group sum for tensor groups is added in its phase and not before.

Flag names are chosen for what fits llmx best; an established name is kept where it is the best fit, and no name is taken only because another runtime uses it. A sketch, to be settled when the flags land: `--devices vulkan:0,vulkan:1` lists the devices; the layer split is the default for more than one; `--group-width N` forms tensor groups of N consecutive devices; `--layer-shares 3,2` overrides the automatic balance. The flags mean the same on every backend or are refused, as the KV cache types are.

## Placement and balance

- Layers go to stages by **measured cost**, not layer count: a stage's time sets the pipeline's pace. The head (0.6 to 1.3 GB and a large matmul at a 151936-row vocabulary) and the embedding are placed as roles with their own cost, not pinned to the last and first device. A head pinned to the last card capped a 10-card split before, and moving it to the CPU lost 31 to 43 percent.
- Per-layer costs are not uniform. Attention grows with context, and MoE and dense layers differ, so the balance is set at a representative context and the chunk sizing absorbs the rest. Weights do not move after load.
- A stage boundary never splits a layer's attention from its feed-forward block, except where the placement asks for it (CPU experts).
- A second card beside another job halved layer-split prefill in earlier measurements, so gates run on cards with no neighbour.

## Mixture of experts

A MoE layer is a small attention block and a large set of experts of which each token uses a few: Qwen3-30B-A3B has 128 experts of 768 values per layer and uses 8, Qwen3-235B-A22B 128 of 1536 and uses 8. Most of the bytes are experts, few are read per token, and with many users most experts are read every pass anyway: 32 decode tokens of Qwen3-30B-A3B touch about 111 of the 128. Measured on this hardware, MoE decode is not bound by memory bandwidth (31 percent of the ceiling against 72 for a dense model) but by dispatches and small matmuls, which is also what llmx's own MoE decode showed (`docs/STATUS.md`).

The placement already puts a layer's feed-forward block on its own group, separate from its attention (the CPU experts use it). Four ways to split the experts follow from that, in order of preference on this hardware:

1. **Layer split** (the default). Whole MoE layers per stage, attention and experts together. One crossing per boundary per pass, and with passes in flight each card reads its layers' experts once per pass for all of that pass's tokens, so aggregate expert bandwidth grows with the cards. This is the multi-user choice, and the only one needed for Qwen3-30B-A3B Q8_0 on two cards or Qwen3-235B-A22B on six.
2. **Expert tiers.** A layer's experts divided by id between two devices, the most used on the faster one: a card and the CPU (the Radeon VII with a model slightly past its 16 GB, where today whole layers go to the CPU), or two cards when the last stage overflows by a few experts. The router runs on the attention device; each token's entries go to the device that holds their expert; the slots come back and are combined on the attention device in slot order, as `moe_combine` does now. Two crossings per MoE layer, as the CPU experts pay today, and both devices compute at once. The split is chosen from usage counts that `route_experts` collects per expert and layer, measured on a calibration run and stored with the placement, and it stays fixed while serving. This generalizes `--n-cpu-moe` from whole layers to single experts and replaces most of what streamed experts do for long prompts.
3. **Expert parallelism** across a group of cards: every card holds a share of each layer's experts, and every MoE layer exchanges token states out to the cards holding their experts and slot results back. It scales a single pass's expert bandwidth with the cards, which is its only advantage over a layer split with passes in flight. On this hardware it lost 3 to 1 single-stream in earlier work (29 against 81 tokens/s on gpt-oss-120B), mostly to host sorting, a readback and a synchronization per call, and lost fusion. If it is built, it keeps llmx's device-side grouping (`moe_group.comp`), never synchronizes the host inside a layer, combines in slot order, and replicates the most used experts on several cards to balance load, as DeepSeek's EPLB does. It needs peer copies to be worth measuring, so it waits for ROCm, and its exchange cost through host memory is measured in phase 0 first.
4. **Expert tensor split.** Each expert's matrices split across a tensor group, with the group's sums. The split must fall on whole quant blocks: 768 splits into 3 by 256 but not into 2 for K-quants, 1536 into 2, 3 or 6. MoE tensor decode gained little on this hardware (each added card cost about 0.9 ms of host launch time per token against a 9.9 ms device pass), so this follows the tensor groups and is used only where the attention is tensor-split anyway.

Shared experts (Qwen2-MoE, DeepSeek) go with the routed ones and are never copied to every member: a copied shared expert was 89 percent of each card's feed-forward bytes in an earlier tensor split.

A token's routed entries computed on the same kind of device in the same slot order give the same result wherever the experts sit, so an expert split across two identical cards is bit-identical to one card. Across a card and the CPU the dot kernels differ, so the gate there is the HF bound, determinism and batch invariance, as for the CPU experts today.

## Tensor groups

Built after the layer split and on the same structure.

- Weights are sharded at load: column-parallel projections by output rows (contiguous rows, no repacking), row-parallel ones by input blocks (each row's block range copied into a per-member matrix). The split falls on quant block boundaries, so a width is legal only where `n_embd`, the head counts and the feed-forward widths divide into whole blocks per member. MoE experts of 768 values split into 384, which is not a whole number of 256-value K-quant blocks, so expert projections take whole experts per member or stay replicated; that is an open question per model.
- The group sum is deterministic: partial sums added in a fixed member order on every member. Earlier, an 8-card reduction that added peers in per-rank order silently diverged at 100k context, and a reused scatter region raced when message sizes grew; both are design constraints here.
- On ROCm the sum uses peer stores for small messages and a ring for large ones (written here, not linked). On Vulkan it crosses through host memory, twice per layer. Whether that pays is measured in phase 0 before any Vulkan tensor code is written.
- A tensor group does not reduce dispatch count per device: each member runs every layer's kernels. The per-token dispatch floor stays, so tensor decode scales less on MoE models, where two thirds of decode time did not parallelize.

## Replicas

For a model that fits one card or one group, several independent copies behind one scheduler beat any split for many users. Replicas share the scheduler and route requests by prefix affinity, and each keeps its own KV and prefix index. This is ROADMAP #5's cheapest high-throughput mode, and it is in scope once the pipeline exists.

## Problems known from other systems, and how this avoids them

- **llama.cpp.** Its layer split overlaps only prompt ubatches (copies of the graph inputs, `GGML_SCHED_MAX_COPIES`); decode across cards is sequential. The overlap breaks whenever another context decodes between ubatches, which halves multi-card prefill with a draft model (issue 27428). The head sits on the last device. A deep ring that does not fit falls back to one copy with only a log line. Its row split scatters activations from one main device and gathers them back per matmul. Here: passes in flight for decode, the head as its own role, a loud fit, and no row split.
- **vLLM.** Pipeline parallelism on one node is reported far worse than tensor parallelism for inter-token latency (13 against 52 tokens/s of output in one published comparison). Its fast all-reduce needs peer access and falls back to NCCL on PCIe cards without it; hangs at startup on PCIe peer tests are a recurring issue. Here: tensor groups only where the sum is measured to pay, and passes in flight so a layer split is judged against a filled pipeline.
- **Megatron and pipeline schedules.** The bubble is (S-1)/(M+S-1) for S stages and M microbatches in flight, so M must exceed S. The embedding and output stages unbalance the pipeline, and tensor parallelism belongs where the interconnect is fast. Here: P = S+1, the head and the embedding as costed roles, and group width at most 4 over PCIe.
- **SGLang.** Chunked pipeline prefill with shrinking chunks keeps stages balanced as context grows. Taken as described above.
- **Earlier work on this hardware.** Batch dependence came from server checkpoint rules changing ubatch shapes, not from grouping: prompt boundaries here follow the prompt alone. Hand-built stage subgraphs lost fusion (16 percent) until their bookkeeping was complete: stages here are the model's own layer code, not partitioned graphs. Cross-device reads of cache views staged 251 MB per pipeline copy: each device reads only its own KV.

## Order of work

Each phase is its own branch from main, merged on its own gates, with a STATUS block opened before its code.

| # | Phase | Gate |
|---|---|---|
| 0 | Measurement only. Handoff latency on two MI50s under Vulkan: blocking read and write, host-visible copy with a ticket, imported host memory, host-signalled device wait. Two Vulkan devices in one process running concurrently. A host-relayed group sum. Baselines: llama.cpp (Vulkan and ROCm, cards pinned) and vLLM (gfx906 fork, AWQ) on 2 and 4 cards, bench and server. Qwen3-32B and Qwen3-235B-A22B downloaded | The numbers that choose phase 2's handoff and whether Vulkan tensor groups are worth writing |
| 1 | Stages and groups of width 1, flags, `memory_available`, automatic fit, parallel loading, today's crossing | Bit-identity with one device (tiny HF model, 0.6B, 8B, 30B-A3B) on CPU+CPU, CPU+Vulkan and Vulkan+Vulkan. 30B-A3B Q8_0 and 32B Q8_0 fully on two MI50s through the HF gates. Radeon VII + CPU layer split. No regression on any single-device path |
| 2 | `begin` / `run_stage` / `finish`, one thread per stage, pipelined handoff | The same bit-identity; single-stream decode no slower than phase 1 |
| 3 | Scheduler: P passes in flight, batch assembly by predicted stage time, admission over several pools | Server gates at 1 to 64 users and a rate sweep against llama.cpp and vLLM on the same cards; server/CLI equality across placements |
| 4 | Pipelined long prompts with shrinking chunks | Prefill at 512 to 16384 tokens against both references on 2 and 4 cards |
| 5 | Replicas | Aggregate throughput against one split instance on the same cards |
| 5b | Expert tiers: per-expert placement by measured use, card and CPU or two cards | Radeon VII with Qwen3-30B-A3B Q4_K_M and Q8_0 against today's whole-layer CPU experts and against the references' expert offload; HF MoE gate; bit-identity across two cards |
| 6 | Tensor groups (ROCm first, Vulkan if phase 0 says it pays) | HF gate, determinism and batch invariance; decode at 1 to 4 users against the layer split and both references |
| 7 | Staged tensor | Prefill and decode against the best single mode at each concurrency |

## Open questions and risks

1. **Handoff on Vulkan.** Whether a device-side wait on another device's semaphore works on RADV between two MI50s. If not, a host wake of 10-50 us per handoff; negligible for decode passes of tens of milliseconds, visible only on tiny passes.
2. **Host memory.** 62 GB of RAM, much of it used by other work. A 142 GB model loads through the map and streams; load time and page-cache pressure need measuring. Uploads to several cards in parallel must not multiply peak RAM.
3. **Arena memory times P.** Measured before on a 10-card split: 445 MB of compute buffer at one copy against 2.8 GB at ten. The fit must count it, and runs with a different P must be visible.
4. **Balance under changing load.** Stage costs shift with context length and with the prefill/decode mix, while weights are placed once. The cost model and chunk sizing absorb what they can; the rest shows as a bubble, reported by the pipeline.
5. **Mid-pipeline failure.** A stage failing after the pass committed its histories. The design fails the pass's requests; recovery is not designed, as for device loss today.
6. **Cancel and pause** only between passes. A request cancelled while in flight completes its pass first.
7. **Threads.** Stage threads, the CPU backend's pool and the HTTP threads share 16 hardware threads, and a CPU stage competes with all of them.
8. **Tensor legality.** Head counts and widths must divide by the group width in whole quant blocks. MoE expert widths may not; per model.
9. **Determinism across splits.** A tensor group computes different sums than one device. Its CPU-vs-device check compares within a bound, as the device checks do now.
10. **The vLLM baseline.** It depends on a community fork for gfx906 and on AWQ or GPTQ weights. If it does not run, the gate says so rather than dropping the comparison.
11. **Shared machine.** Multi-card timing needs cards with no neighbour for the whole run, and link speed checked at the start.
12. **Windows.** One GPU, so the Windows gates cover the Radeon VII with CPU stages; multi-card gates run on Linux only.
13. **Expert use drifts.** Expert tiers placed from a calibration run lose when a workload uses other experts. Placement stays fixed while serving; moving experts between passes is possible later but not designed.
14. **Expert exchange cost.** Expert parallelism crosses twice per MoE layer; through host memory that is about 94 x 2 synchronizations per pass on Qwen3-235B-A22B. Phase 0 measures it before anything is built.
15. **Downloads.** The Linux machine's uplink measured about 2.5 MB/s, so a 142 GB model takes most of a day to fetch; phase 0 starts the downloads first.

## Sources

- vLLM on pipeline versus tensor parallelism on one node, and the V1 engine: https://docs.vllm.ai/en/latest/configuration/optimization/ ; https://croz.net/run-your-own-ai-at-scale-vol-1-tuning-vllm/
- vLLM custom all-reduce and PCIe peer access: https://discuss.vllm.ai/t/what-means-there-is-no-p2p-support/1928 ; https://github.com/vllm-project/vllm/issues/4996
- llama.cpp pipeline parallelism and its limits: https://github.com/ggml-org/llama.cpp/pull/6017 ; https://github.com/ggml-org/llama.cpp/issues/27428 ; https://github.com/ggml-org/llama.cpp/discussions/20252
- SGLang chunked pipeline parallelism: https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/
- vLLM for gfx906: https://github.com/nlzy/vllm-gfx906
- Expert load balancing by replicating used experts (DeepSeek EPLB): https://github.com/deepseek-ai/EPLB
- Hybrid CPU and GPU expert placement: https://github.com/kvcache-ai/ktransformers
- Measurements on this hardware: mx-llama.cpp research notes (`tp-notes/split-modes-explained.md` in that repository) and the records summarized above.
