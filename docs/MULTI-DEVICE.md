# Multi-device execution: layer, tensor and staged splits

Design for ROADMAP #5, written before any code and open for review. It extends [EXECUTION](EXECUTION.md), whose placement, tickets and crossing already run a model over two backends, and replaces its decision to drop tensor-parallel work: that returns here as the tensor group, built after the layer split and designed with it. The server side extends [SERVER](SERVER.md).

## Why, and what must hold

Models past one card's memory cannot be run or gated today (Qwen3-32B Q8_0 at about 35 GB, Qwen3-30B-A3B Q8_0 at 32.5 GB without CPU experts, Qwen3-235B-A22B at about 142 GB in Q4_K_M), and a server with many users should turn every added card into throughput. The design is judged on the multi-user case first.

Gates, all required before a phase merges:

1. **No regression** on any current path: single device, CPU, CPU experts beside a device, streamed experts, the server. Interleaved A/B against main with a layout control, on both machines.
2. **Correctness.** Exact where the arithmetic is the same: pipelined against serialized execution of the same placement, a split over identical devices (CPU+CPU, two identical cards) against one of them, compared as logits over every position and greedy text for prompts, decode and a mixed server pass. Where the devices differ (CPU+Vulkan: the CPU sums in double, the device in its own order), the HF bounds apply independently to the split, as they do to each device alone. A tensor group passes the HF gate within its bounds, is identical run to run, and computes a prompt the same however it is batched. Server and CLI give the same greedy text within one placement at every concurrency.
3. **Ahead of llama.cpp by a wide margin** on the same cards, model and quant: `llama-bench` prefill and decode, and its server under the same load.
4. **Ahead of vLLM by a wider margin** on vLLM's own serving metrics: request and output throughput, time to first token and inter-token latency at p50 and p99, end-to-end latency, over a sweep of arrival rates and concurrency, with vLLM's benchmark method. vLLM runs on gfx906 only through a community fork, which is archived; its maintainer reports Qwen3-30B-A3B GPTQ-Int4 partly working and slow on v0.9.2. So the baseline is a bring-up task, not a given: phase 0 establishes that the pinned fork runs, which quant and backend it supports, and that its outputs are correct, before any table uses it. Equal bits do not mean equal quality, so every table carries a quality measure for both runtimes beside the format difference, and a baseline that does not run is reported as such, never dropped.

## The hardware, measured

On the Linux machine, with ten MI50 32 GB cards, an EPYC 7262 (8 cores, 16 threads) and 62 GB of RAM, from earlier work with mx-llama.cpp on the same machine:

- Every card sits behind its own switch on a PCIe Gen4 x8 root port, with no XGMI. Host to device and back run at 14.3-14.4 GB/s, and a HIP peer copy between any two of the 56 pairs at 14.3 GB/s. Link training is random per boot and a card can come up at Gen1 or Gen3, so the link speed is checked before any timing.
- Peer to peer works under HIP: DMA peer copies at full link rate, and kernel peer stores at 3-6 GB/s for 16 KB messages with 6-17 us latency, correct only with fine-grained memory, which makes other kernels reading that memory slower. The Vulkan backend has no peer copy today. The API has device groups with peer memory, and whether RADV forms a group of two MI50s is a phase 0 query. A timeline semaphore exported as an opaque file descriptor can only be imported by a device with the same driver and device UUIDs, so importing one card's timeline into another is not a portable path: each device keeps its own timeline, and a host thread relays between them. Binary semaphores exported as sync files (temporary import, copy semantics) are the candidate for a device-side wait, measured in phase 0 on the installed driver after querying the supported handle types.
- Memory reads at 805 GB/s. A dependent dispatch costs 4.1 to 5 us whatever it does, which bounds decode by dispatch count once the rows are fast (`docs/VULKAN.md`, kernel notes).
- Two streams on one card do not overlap real kernels (1.02x), so a card runs one pass at a time and overlap comes from different cards.

The Windows workstation has one Radeon VII (16 GB). There the only second device is the CPU, so a layer split is the only split it can use, and the multi-card gates run on Linux.

## Three splits, one placement

- **Layer split.** Consecutive layers on different devices. Only the residual crosses a boundary, `rows x n_embd` floats per pass: 20 KB for a decode token of a 5120-wide model, 10 MB for a 512-row chunk (0.7 ms at link rate). Arithmetic is unchanged, so a split over identical devices, and pipelined against serialized execution of the same placement, is bit-identical; across different devices (CPU and a card) the HF bounds apply.
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
| Mixture of experts, many users | Layer split first, then data-parallel attention with expert parallelism | Experts are most of the bytes; sharding them divides each pass's reads by the cards while attention and KV stay local (below) |

Group widths beyond 4 lose on this hardware: the sum's wait is the slowest of N-1 PCIe latencies, 8.2 us at 4 cards and 20.3 us at 8, which is 27 percent of an 8-card decode token.

## Multi-user execution

### Passes in flight

A pass is one forward over a batch through every stage. With one pass at a time, each stage idles while the others work, and a layer split is no faster than one card. So the server keeps **P passes in flight**, each a separate batch of different sequences, one per stage and one being sampled:

```
stage 0:  [A][B][C][A'][B'] ...
stage 1:     [A][B][C][A'] ...
host:           sample A, form A'
```

Each stage runs its passes in order. P starts at the stage count plus one (the plus one covers the host's sampling and batch assembly), but that is a heuristic, not a proven depth: P is sized from the measured round trip of a pass and the target interval between a request's tokens, bounded by memory, and phase 0 sweeps S, S+1, S+2 and 2S. A host that cannot sample and assemble within the target interval is a bottleneck no P removes. No tuner runs at load until measurements show one is needed. A sequence is in at most one pass at a time, since its next token needs its logits, and `KVSequence::prepare` already refuses a second pending step.

This is between-step filling, one scheduler and one KV manager for all passes, as vLLM V1 does it (`max_concurrent_batches` = pipeline size). Two alternatives were measured on this hardware and are not taken: static per-engine request partitions (V0 virtual engines, +19 percent at K=4 but a fixed split of slots and caches) and in-step ubatch pipelining of one batch (+21 percent). The two did not compound: they compete for the same requests.

### Decode

Decoding sequences are spread across the passes in flight, each sequence staying in its pass while it decodes, and new decoders going to the pass with the least work. Merging all decoders into one pass would leave all stages but one idle; spreading them keeps every stage busy, and a card reads its weights once per pass for all the pass's rows. With fewer decoders than P, the passes are fewer and some stages idle, which is the case a tensor group serves better (below).

### Prefill beside decode

A pass carries decode rows first and then prompt chunks, as the server does today, but its size is set by **predicted stage time**, not only by a token count. A pass whose chunk takes twice as long stalls every pass behind it at every stage, so each pass is assembled to about the same predicted time, and to at most the inter-token latency the decoders are promised. The cost model per stage is measured at load and corrected from observed pass times: decode rows cost about their weight reads, prompt tokens their compute plus attention over their history.

Through phase 3 a prompt, like every sequence, has one pass in flight. Phase 4 lets a long prompt's chunks go to consecutive passes, so chunk i+1 enters stage 0 as soon as chunk i leaves it: its layers at stage 0 need only chunk i's keys and values at stage 0, which are complete by then. That breaks the one-pass rule, so phase 4 first gives a sequence per-storage progress: blocks reserved and positions written per stage, a committed history that is the part every storage has written, and cancellation that removes every later chunk that depended on a failed or cancelled one. A chunk later in a long prompt attends to more history and costs more, so chunk size shrinks as the prompt grows to keep the stage time level, as SGLang's dynamic chunking does. Chunk boundaries and each row's extent depend on the prompt alone, never on the cost model or on what else is in flight: the predictor may change which requests share a pass, but not what any row computes.

### More requests than stages

This is the normal case. Passes grow wider rather than more numerous: P stays at its measured depth and each pass carries more sequences, up to the KV budget. Past the point where a pass turns compute-bound, around 35 rows on an MI50, wider passes stop being free, and the scheduler caps pass width by the same predicted time.

### KV, admission and the arenas

Each device keeps the KV of the layers it runs (per head within a tensor group), in its own pool, as now. Pools differ in block size (128 tokens on the CPU, 64 on a Vulkan device), so admission is not one scalar: a request reserves blocks in **every** pool it touches, each counted in that pool's own units, atomically (all or none). The count includes blocks held by prefix donors and shared with them, and blocks reserved by passes still in flight. A single token budget would be wrong: 129 tokens are 2 CPU blocks and 3 device blocks, but two requests of 65 tokens need 2 CPU blocks and 4 device blocks. Prefix reuse keeps one block table per storage, as now, and a reusable prefix ends on a boundary that is a whole block in every storage.

The fit is computed at load from each device's reported free memory, and it counts everything a device holds: weights, including a tied embedding and head uploaded to two devices; KV and its growth while a pass reserves ahead; the activation arena of each pass in flight; RoPE tables, logits and handoff buffers; the attention and tile scratch; streamed expert windows; and padded F32 copies. On the host it counts any payload a loader materializes. When P passes do not fit, the server says so and runs with fewer; it never silently falls back to one pass in flight.

## Execution structure

Layers as in ARCHITECTURE: the model knows stages, the inference layer runs them, and the server only schedules.

- **Model** (`model/`). `forward` becomes three calls: `begin(ctx, entries)` (positions, block reservations, the embedding), `run_stage(ctx, s)` (one stage's layers, the handoff out of it) and `finish(ctx)` (the head). `forward` stays as those three in a row, so the CLI and every test run unchanged. A pass commits its sequences' histories after `finish` has enqueued the head and every stage's submission has succeeded, as `forward` commits after its submissions today. Each stage's ticket is kept with the pass. A failure before commit aborts the reservations, after draining any work a stage enqueued without a ticket (a sync on that backend, as the error path does now). Device loss exits the process, since memory whose completion is unknown cannot be released safely.
- **Pipeline** (`inference/pipeline.hpp`). One host thread per stage. Each takes passes from its queue in order, runs `run_stage`, and hands the pass to the next stage's queue. A CPU stage computes on its own thread, so CPU and GPU stages overlap too. The CLI's long prompts use the same pipeline.
- **Ownership.** Every backend has exactly one owner thread for **every** operation on it: the stage ops, the embedding and the head if they sit there, and the sequence operations (`reset`, `truncate`, a donor fork) and error paths. A backend that serves several roles, such as a card holding the last stage and the head, or the CPU holding experts for layers of two stages, belongs to one worker, and every role on it runs on that worker. The scheduler never calls a backend itself; it posts sequence operations to the owning workers. A placement that would give one backend two owners is refused at load.
- **Handoff.** Phase 1 keeps today's crossing, a blocking read and a write, which is correct and slow. The pipelined handoff needs no new backend call: the source copies the residual into a host-visible buffer with the existing `copy`, the stage thread waits that submission's ticket, and the next stage `write`s it. Zero-copy variants, host memory imported into both devices or a device-side wait on the source's semaphore, are backend-private improvements measured in phase 0. A ROCm backend adds a peer copy when it exists. On HIP, cross-device stream waits were unreliable with more than four hardware queues, so that path gets a dedicated copy stream and is checked for ordering.
- **Scheduler** (`server/scheduler.hpp`). P execution contexts, batch assembly by predicted time, admission over several pools, and cancellation and pausing only for sequences not in flight.
- **Backend** (`backends/`). One addition in phase 1: `memory_available()`, the device's free memory (Vulkan through `VK_EXT_memory_budget`, CPU from the OS), which the fit needs. The group sum for tensor groups is added in its phase and not before.

Flag names are chosen for what fits llmx best; an established name is kept where it is the best fit, and no name is taken only because another runtime uses it. A sketch, to be settled when the flags land: `--devices vulkan:0,vulkan:1` lists the devices; the layer split is the default for more than one; `--group-width N` forms tensor groups of N consecutive devices; `--layer-shares 3,2` overrides the automatic balance. The flags mean the same on every backend or are refused, as the KV cache types are.

## Loading

Only a single-file GGUF is mapped today. A sharded GGUF and a native safetensors model assemble their payload in host memory, which a model past host RAM cannot do, and Qwen3-235B-A22B ships as sharded GGUF. So bounded shard loading, each shard mapped and its pages dropped once the devices hold them, is a prerequisite of phase 1 for any model larger than host memory. The native HF path stays dense-only and its half-precision weights CPU-only until that path gains devices. A payload stays owned for as long as any loader, replica, upload or buffer borrowing it is alive, and is released only after the last of them.

## Placement and balance

- Layers go to stages by **measured cost**, not layer count: a stage's time sets the pipeline's pace. The head (0.6 to 1.3 GB and a large matmul at a 151936-row vocabulary) and the embedding are placed as roles with their own cost, not pinned to the last and first device. A head pinned to the last card capped a 10-card split before, and moving it to the CPU lost 31 to 43 percent.
- Per-layer costs are not uniform. Attention grows with context, and MoE and dense layers differ, so the balance is set at a representative context and the chunk sizing absorbs the rest. Weights do not move after load.
- A stage boundary never splits a layer's attention from its feed-forward block, except where the placement asks for it (CPU experts).
- A second card beside another job halved layer-split prefill in earlier measurements, so gates run on cards with no neighbour.

## Mixture of experts

A MoE layer is a small attention block and a large set of experts of which each token uses a few: Qwen3-30B-A3B has 128 experts of 768 values per layer and uses 8, Qwen3-235B-A22B 128 of 1536 and uses 8. Most of the bytes are experts, few are read per token, and with many users most experts are read every pass anyway: 32 decode tokens of Qwen3-30B-A3B touch about 111 of the 128. Measured on this hardware, MoE decode is not bound by memory bandwidth (31 percent of the ceiling against 72 for a dense model) but by dispatches and small matmuls, which is also what llmx's own MoE decode showed (`docs/STATUS.md`).

The placement already puts a layer's feed-forward block on its own group, separate from its attention (the CPU experts use it). The splits, in the order they are built:

1. **Layer split** (first, and the default). Whole MoE layers per stage, attention and experts together. One crossing per boundary per pass, and with passes in flight each card reads its layers' experts once per pass for all of that pass's tokens. This is what phase 1 gives Qwen3-30B-A3B Q8_0 on two cards and Qwen3-235B-A22B on six.
2. **Data-parallel attention with expert parallelism** (the MoE serving path of vLLM, SGLang and Megatron, adapted below). Built after the layer split, on the group collective the tensor groups also use, and only if phase 0's measurements say it pays.
3. **Expert tiers.** A layer's experts divided by measured use between a card and the CPU, or two cards.
4. **Expert tensor split.** Each expert's matrices across a tensor group; last, and only where attention is tensor-split anyway.

### Data-parallel attention with expert parallelism

Each card of an expert group (a **rank**) holds the attention, router, norms, embedding and head, and runs attention for **its own requests** with its own KV. Each card holds a share of every MoE layer's experts. So KV is never duplicated and never split by heads (Qwen3 MoE models have only 4 KV heads, which caps a tensor split's width), and the expert bytes read per pass are divided by the cards. For Qwen3-235B-A22B on eight MI50s this is about 4 GB of replicated attention in Q4_K and 17 GB of experts per card; whether KV, arenas and the exchange buffers fit beside them is the fit's calculation (Capacity, below), not assumed here.

Every MoE layer, on every rank at once:

1. **Route** locally: the router and `route_experts`, as now. Each token's global choices and probabilities are kept as routed, with no renormalization at the destination.
2. **Dispatch**: each entry goes to the rank holding its expert. An entry is `(home rank, token row, original slot, expert, extent)` plus the token's activations, sent once per destination however many of the token's experts sit there. A pack kernel writes each destination's entries into its outbox with the counts in a header, so the host never reads counts.
3. **Expert matmuls**: the received entries grouped by local expert (`moe_group.comp`) and run through the routed kernels. The received list is compact and variable per token, unlike today's `Routing`, which assumes `rows x k` uniform entries. This is a new seam: routed projections over an explicit entry list.
4. **Return and combine**: each entry's **raw** down-projection vector goes back to its home rank, which applies the weights once, in original slot order, as `moe_combine` does now. The destination never weights or sums. So `matmul_experts_add`, which combines locally today, splits into the raw down projection and a combine over returned vectors.

**Exchange epochs.** Removing the count readback does not remove synchronization: each dispatch and each return is a group-wide exchange. Each has an epoch, `(pass, layer, micro-batch, direction)`, and three events: the producer's work that wrote the outbox has completed (its ticket), the consumer may read (the relay has seen every producer of that epoch), and the consumer has finished reading (its ticket), before the buffer is reused. Buffers alternate by epoch parity, and a producer never writes an epoch's buffer until the previous user's read has retired. That is 188 exchanges a pass on Qwen3-235B-A22B with one micro-batch, and 376 with two unless their exchanges are coalesced.

**Overlap.** Each rank's pass may split into two micro-batches of its requests, so one exchanges while the other computes (vLLM's dual batch overlap, SGLang's two-batch overlap). Both report that it pays only above 64 to 128 tokens per batch. Here the overlap is not assumed: phase 0 measures serialized against two micro-batches, and it engages only from the size where it measured a gain.

**The exchange.** Under Vulkan the baseline is ordinary staging: each outbox is copied to host-visible memory, relayed, and written to each destination. Host memory imported into every card (`VK_EXT_external_memory_host`) would remove one copy, but it is not implemented and must be probed per device: supported handle and memory types, the minimum import alignment, and the memory properties. An imported host allocation stays alive until every import and every reader has retired. HIP's peer numbers say nothing about Vulkan. Under ROCm, a rank writes into the other ranks' inboxes by peer store (6 to 17 us for small messages on these cards).

**Capacity.** The worst destination receives `G x R x K` entries: G ranks, R tokens per rank per micro-batch, K experts per token. Its memory is about `3 x G x R x K x F x 4` bytes for gate, up and activation, plus `G x R x K x E x 4` for the raw returns, plus routing, activation twins, outboxes and alignment, for every epoch that can be live, plus any replicated experts. Reserving a fixed buffer per rank pair grows with the square of the ranks and can take most of host memory. So R has a cap that admission enforces, the fit counts all of the above, and llmx stays dropless within that cap: an entry is never dropped, and a pass that would exceed the cap is split before it runs.

**Bit identity is a proof obligation, not a given.** Today's routed kernels choose by local counts: per-entry or grouped execution at `entries < 2 x n_expert`, the tile and its shape by grouped counts and the device profile, and the activation path by type, extent and profile. Moving entries between ranks changes the local counts. An incoming F32 row also arrives without the twin that its producer cached, so it is quantized again on arrival. The path therefore claims identity with one card only for combinations tested bit for bit: each routed path (per-entry, grouped, tile), decode and prompt extents, every weight type, entries moved against the same entries on one card. It preserves each projection's extent per entry, sends F32 values exactly, and quantizes on arrival with the same blocks and rounding as the producer, or sends the twin itself where that is exact. Until a case is tested, it is held to the HF bounds.

**Ranks and scheduling.** Requests are rank-local for their whole life: decode, pause and resume, and prefix reuse. A request's KV and its donors live on its rank. Prefix affinity chooses among donors on the rank a request is assigned to, and free KV on another rank does not admit a request here; moving a history between ranks is migration or recomputation, not in scope. Every rank joins every exchange of a pass, so a rank with no requests runs an explicit empty pass that contributes empty outboxes, not a `forward` over an empty batch, which refuses. The scheduler assigns each new request to a rank (least loaded, then prefix affinity) and assembles every rank's pass to the same predicted time. One rank's long prefill stalls every rank at every MoE layer; predicted-time assembly is the first answer, prefill and decode on separate cards the later one.

**Expert ownership.** Static first: each expert has one owner rank, fixed at load. Redundant copies of the most used experts, as DeepSeek's EPLB places them in vLLM and SGLang, need versioned routing: a new mapping applies from a pass onward, and the old copies are released only after every in-flight pass that routed through the old mapping has retired. Between passes is not enough while other passes are in flight. Balancing is added only after phase 0 or serving shows measured skew that justifies it.

**With stages.** An expert group can be one stage of a layer split, for example two stages of four cards each, with passes in flight across the stages as above.

**When it pays.** It divides the expert bytes per pass by the ranks and keeps attention and KV local, so it is the candidate for many users on a large MoE model. A single request gets no attention speedup and pays two exchanges per layer, so a few users stay on the layer split. Whether it pays under Vulkan, and how much ROCm changes that, cannot be decided from the evidence so far; phase 0 measures it first.

### Expert tiers

A layer's experts divided by id between two devices, the most used on the faster one. This covers a card and the CPU (the Radeon VII with a model slightly past its 16 GB, where today whole layers go to the CPU), or two cards when a stage overflows by a few experts. The router runs on the attention device, each token's entries go to the device holding their expert, and the slots come back and are combined in slot order. That is two crossings per MoE layer, as the CPU experts pay today, and both devices compute at once. The split comes from use counts collected per expert and layer on a calibration run, stored with the placement, and fixed while serving. This generalizes `--n-cpu-moe` from whole layers to single experts.

### Expert tensor split

Each expert's matrices split across a tensor group, with the group's sums. Legality is per axis. Splitting output rows keeps whole rows: Qwen3-30B-A3B's gate and up (768 rows of 2048) split into two halves of 384 rows at no cost. Splitting input columns must fall on quant blocks: its down projection (2048 rows of 768) split into halves of 384 cuts 256-value K-quant blocks, and 256 + 512 is legal but unbalanced. Width 3 fails the attention anyway (32 query heads and 4 KV heads). Expert parallelism is not introduced to get around tensor legality; it is chosen, or not, on its own measurements. MoE tensor decode gained little on this hardware (each added card cost about 0.9 ms of host launch time per token against a 9.9 ms device pass), so it comes last.

Shared experts (Qwen2-MoE, DeepSeek) go with the routed ones and are never copied to every member: a copied shared expert was 89 percent of each card's feed-forward bytes in an earlier tensor split.

An expert split across two identical cards is claimed bit-identical to one card only for the path combinations tested bit for bit, under the same conditions as expert parallelism above (each entry's kernel path and extent preserved, exact transfers, slot-order combine). Across a card and the CPU the dot kernels differ, so the gate there is the HF bound, determinism and batch invariance, as for the CPU experts today.

## Tensor groups

Built after the layer split and on the same structure.

- Weights are sharded at load: column-parallel projections by output rows (contiguous rows, no repacking, any whole-row split), row-parallel ones by input blocks (each row's block range copied into a per-member matrix, legal only on quant block boundaries). A width is legal only where the head and KV-head counts divide by it and every row-parallel input width divides into whole blocks per member, checked per projection axis at load; an illegal width is refused, not rounded.
- The group sum is deterministic: partial sums added in a fixed member order on every member. Earlier, an 8-card reduction that added peers in per-rank order silently diverged at 100k context, and a reused scatter region raced when message sizes grew; both are design constraints here.
- On ROCm the sum uses peer stores for small messages and a ring for large ones (written here, not linked). On Vulkan it crosses through host memory, twice per layer. Whether that pays is measured in phase 0 before any Vulkan tensor code is written.
- A tensor group does not reduce dispatch count per device: each member runs every layer's kernels. The per-token dispatch floor stays, so tensor decode scales less on MoE models, where two thirds of decode time did not parallelize.

## Replicas

For a model that fits one card or one group, several independent copies behind one scheduler beat any split for many users. Replicas share the scheduler and route requests by prefix affinity, and each keeps its own KV and prefix index. This is ROADMAP #5's cheapest high-throughput mode, and it is in scope once the pipeline exists.

## Problems known from other systems, and how this avoids them

- **llama.cpp.** Its layer split overlaps only prompt ubatches (copies of the graph inputs, `GGML_SCHED_MAX_COPIES`); decode across cards is sequential. The overlap breaks whenever another context decodes between ubatches, which halves multi-card prefill with a draft model (issue 27428). The head sits on the last device. A deep ring that does not fit falls back to one copy with only a log line. Its row split scatters activations from one main device and gathers them back per matmul. Here: passes in flight for decode, the head as its own role, a loud fit, and no row split.
- **vLLM.** Pipeline parallelism on one node is reported far worse than tensor parallelism for inter-token latency (13 against 52 tokens/s of output in one published comparison). Its fast all-reduce needs peer access and falls back to NCCL on PCIe cards without it; hangs at startup on PCIe peer tests are a recurring issue. Here: tensor groups only where the sum is measured to pay, and passes in flight so a layer split is judged against a filled pipeline.
- **Megatron and pipeline schedules.** The bubble is (S-1)/(M+S-1) for S stages and M microbatches in flight, so M must exceed S. The embedding and output stages unbalance the pipeline, and tensor parallelism belongs where the interconnect is fast. Here: P starts at S+1 and is measured (S, S+1, S+2 and 2S in phase 0), the head and the embedding are costed roles, and group width is at most 4 over PCIe.
- **SGLang.** Chunked pipeline prefill with shrinking chunks keeps stages balanced as context grows. Taken as described above.
- **Earlier work on this hardware.** Batch dependence came from server checkpoint rules changing ubatch shapes, not from grouping: prompt boundaries here follow the prompt alone. Hand-built stage subgraphs lost fusion (16 percent) until their bookkeeping was complete: stages here are the model's own layer code, not partitioned graphs. Cross-device reads of cache views staged 251 MB per pipeline copy: each device reads only its own KV.

## Order of work

Each phase is its own branch from main, merged on its own gates, with a STATUS block opened before its code.

| # | Phase | Gate |
|---|---|---|
| 0 | Measurement only. Query the semaphore handle types and import/export support, and whether RADV forms a device group of two MI50s with peer memory. Handoff latency on two MI50s under Vulkan: blocking read and write, host-visible copy with a ticket, imported host memory, host relay between per-device timelines, a sync-file binary semaphore wait. Two Vulkan devices in one process running concurrently. A host-relayed group sum. The full expert dispatch and return on 2, 4 and 8 ranks: empty ranks, maximum skew, epoch reuse, small decode and large prefill passes, serialized against two micro-batches, reporting aggregate bytes, per-rank latency and exposed wait. Imported host memory probed per device (handle and memory types, alignment). P swept over S, S+1, S+2 and 2S on a two-stage split. Baselines: llama.cpp (Vulkan and ROCm, cards pinned), and the vLLM gfx906 fork brought up, its supported quant found and its outputs checked, on 2 and 4 cards. Downloads of Qwen3-32B and Qwen3-235B-A22B started first | The numbers that choose phase 2's handoff, P, and whether Vulkan tensor groups and a Vulkan expert exchange are worth writing |
| 1 | Stages and groups of width 1, flags, `memory_available`, the full fit, bounded shard loading, parallel uploads, one owner per backend, admission over pools of different block sizes, today's crossing | Exact against one device on CPU+CPU and on two identical cards (tiny HF model, 0.6B, 8B, 30B-A3B); HF bounds on CPU+Vulkan. 30B-A3B Q8_0 and 32B Q8_0 fully on two MI50s through the HF gates. Radeon VII + CPU layer split. No regression on any single-device path |
| 2 | `begin` / `run_stage` / `finish`, one thread per stage, pipelined handoff | Pipelined exact against serialized execution of the same placement; single-stream decode no slower than phase 1 |
| 3 | Scheduler: P passes in flight, batch assembly by predicted stage time, admission over several pools | Server gates at 1 to 64 users and a rate sweep against llama.cpp and vLLM on the same cards; server/CLI equality across placements |
| 4 | Per-storage progress for a sequence (reserved, written, committed), cancellation of dependent chunks, then pipelined long prompts with shrinking chunks | Exact against unpipelined prefill; cancellation mid-prompt leaves a consistent history; prefill at 512 to 16384 tokens against both references on 2 and 4 cards |
| 5 | Replicas | Aggregate throughput against one split instance on the same cards |
| 5b | Expert tiers: per-expert placement by measured use, card and CPU or two cards | Radeon VII with Qwen3-30B-A3B Q4_K_M and Q8_0 against today's whole-layer CPU experts and against the references' expert offload; HF MoE gate; bit-identity across two cards |
| 6b | Data-parallel attention with expert parallelism, if phase 0 says it pays: exchange epochs, compact entry lists, raw returns combined at home, capped capacity, rank-local requests and empty passes, static expert ownership | Bit-identity with one card for every tested path combination, HF bounds elsewhere; Qwen3-30B-A3B and Qwen3-235B-A22B serving at 1 to 64 users and a rate sweep against the layer split and against vLLM's expert-parallel serving on the same cards. Redundant experts only after measured skew |
| 6 | Tensor groups (ROCm first, Vulkan if phase 0 says it pays) | HF gate, determinism and batch invariance; decode at 1 to 4 users against the layer split and both references |
| 7 | Staged tensor | Prefill and decode against the best single mode at each concurrency |

## Open questions and risks

1. **Handoff on Vulkan.** Timeline semaphores do not import across two cards with different device UUIDs, so the baseline is a host relay between per-device timelines: a host wake of 10-50 us per handoff, negligible for decode passes of tens of milliseconds and visible only on tiny passes. Sync-file binary semaphores are the phase 0 candidate for a device-side wait.
2. **Host memory.** 62 GB of RAM, much of it used by other work. Only a single-file GGUF streams through the map today; sharded GGUF and safetensors assemble in host memory, so bounded shard loading comes first. Load time and page-cache pressure need measuring, and uploads to several cards in parallel must not multiply peak RAM.
3. **Arena memory times P.** Measured before on a 10-card split: 445 MB of compute buffer at one copy against 2.8 GB at ten. The fit must count it, and runs with a different P must be visible.
4. **Balance under changing load.** Stage costs shift with context length and with the prefill/decode mix, while weights are placed once. The cost model and chunk sizing absorb what they can; the rest shows as a bubble, reported by the pipeline.
5. **Mid-pipeline failure.** Histories commit only after every stage's submission succeeded, so a failure before commit aborts cleanly once untracked work is drained. Device loss exits the process; recovery is not designed.
6. **Cancel and pause** only between passes. A request cancelled while in flight completes its pass first.
7. **Threads.** Stage threads, the CPU backend's pool and the HTTP threads share 16 hardware threads, and a CPU stage competes with all of them.
8. **Tensor legality.** Two separate checks per projection axis: the head and KV-head counts must divide by the group width, and every row-parallel input width must split into whole quant blocks per member (output-row splits need only whole rows). MoE down projections may fail the second; per model.
9. **Determinism across splits.** A tensor group computes different sums than one device. Its CPU-vs-device check compares within a bound, as the device checks do now.
10. **The vLLM baseline.** The gfx906 fork is archived and reported slow and partly working on Qwen3-30B-A3B GPTQ-Int4. It is brought up and checked for correctness in phase 0; if it does not run, the gate says so rather than dropping the comparison.
11. **Shared machine.** Multi-card timing needs cards with no neighbour for the whole run, and link speed checked at the start.
12. **Windows.** One GPU, so the Windows gates cover the Radeon VII with CPU stages; multi-card gates run on Linux only.
13. **Expert use drifts.** Expert tiers placed from a calibration run lose when a workload uses other experts. Placement stays fixed while serving; moving experts between passes is possible later but not designed.
14. **Expert exchange cost.** Expert parallelism synchronizes its group twice per MoE layer: 188 epochs per pass on Qwen3-235B-A22B, 376 with two micro-batches unless coalesced. Overlap hides them only if each layer's compute per micro-batch outlasts the exchange, and that is measured, not assumed.
15. **Downloads.** The Linux machine's uplink measured about 2.5 MB/s, so a 142 GB model takes most of a day to fetch; phase 0 starts the downloads first.
16. **Rank imbalance.** Under data-parallel attention one rank's long prefill stalls every rank at every MoE layer. Predicted-time assembly is the first answer, prefill and decode on separate cards the second.
17. **Backend ownership.** Roles that share a backend (the last stage and the head, CPU experts of two stages) must share its one owner thread, or the placement is refused.
18. **Admission across pools.** Different block sizes make any single-number budget wrong; reservations are per pool, atomic, and count donor and in-flight blocks.
19. **Expert capacity.** A destination may receive `ranks x tokens x k` entries; without an admission-enforced cap, per-pair buffers grow with the square of the ranks.
20. **Rank-local state.** Requests and donors cannot move between ranks without migration, so a busy rank cannot borrow another rank's free KV.

## Sources

- vLLM on pipeline versus tensor parallelism on one node, and the V1 engine: https://docs.vllm.ai/en/latest/configuration/optimization/ ; https://croz.net/run-your-own-ai-at-scale-vol-1-tuning-vllm/
- vLLM custom all-reduce and PCIe peer access: https://discuss.vllm.ai/t/what-means-there-is-no-p2p-support/1928 ; https://github.com/vllm-project/vllm/issues/4996
- llama.cpp pipeline parallelism and its limits: https://github.com/ggml-org/llama.cpp/pull/6017 ; https://github.com/ggml-org/llama.cpp/issues/27428 ; https://github.com/ggml-org/llama.cpp/discussions/20252
- SGLang chunked pipeline parallelism: https://www.lmsys.org/blog/2026-01-15-chunked-pipeline/
- vLLM for gfx906: https://github.com/nlzy/vllm-gfx906
- Expert load balancing by replicating used experts (DeepSeek EPLB): https://github.com/deepseek-ai/EPLB
- Hybrid CPU and GPU expert placement: https://github.com/kvcache-ai/ktransformers
- vLLM expert-parallel deployment and dual batch overlap: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/ ; https://docs.vllm.ai/en/latest/design/dbo/
- SGLang large-scale expert parallelism (DP attention, DeepEP, two-batch overlap, EPLB, prefill and decode on separate cards): https://www.lmsys.org/blog/2025-05-05-large-scale-ep/
- Megatron Core MoE token dispatcher: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
- Measurements on this hardware: mx-llama.cpp research notes (`tp-notes/split-modes-explained.md` in that repository) and the records summarized above.
- Vulkan host-pointer import: https://docs.vulkan.org/refpages/latest/refpages/source/VkImportMemoryHostPointerInfoEXT.html
- Vulkan external semaphore compatibility and device groups: https://docs.vulkan.org/spec/latest/chapters/capabilities.html ; https://docs.vulkan.org/refpages/latest/refpages/source/VkSemaphoreGetFdInfoKHR.html ; https://docs.vulkan.org/refpages/latest/refpages/source/vkGetDeviceGroupPeerMemoryFeatures.html
- vLLM gfx906 fork status: https://github.com/nlzy/vllm-gfx906/issues/29
- Qwen3-30B-A3B configuration: https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
