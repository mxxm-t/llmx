# Tensor split and the staged tensor split (planned)

Plan for phases 6 and 7 of [MULTI-DEVICE](MULTI-DEVICE.md), written on 2026-10-03 before any code and open for review.
It refines that page's Tensor split section and its Three splits, one placement description with research, a measurement and an order of work.
It is a separate page because the research, the skew analysis, serving and speculative decoding over a group do not fit inside a section of the multi-device design; MULTI-DEVICE keeps the placement model all splits share and points here.

## 1. Goal

A tensor group runs every layer of its stage on 2 to 4 devices at once, so one request decodes faster than on one device or on a layer split, and a model past one device's memory runs at more than one device's speed per request.
The staged tensor split is a layer split whose stages are tensor groups (for example 2 stages of 4 devices on 8 cards), and it is designed in from the first branch, so the first version needs no rewrite for it.
It serves many users through the same scheduler, pass API and speculative decoding as every other placement, with no separate path.

What must hold, from MULTI-DEVICE's gates and AGENTS.md:
- No regression: every placement without a group (one device, CPU, CPU experts, streamed experts, the layer split, the server) keeps main's bytes and speed.
- Correctness: the rule in section 4.4, held to the HF reference.
- Speed: single-request decode scales with the group's width, and the group is measured at 1 to 64 users against the layer split, mx-llama.cpp's ROCm tensor split and vLLM on the same cards, model and bits.
- Skew: measured and bounded at every width offered (section 4.5).

## 2. Research summary

### 2.1 What llmx already has

- **Placement per role.** `Placement` names a backend for the embedding, the head and each layer's mixer and feed-forward part (`model/runtime.hpp`), a stage is a run of layers whose mixer sits on one device, and `place_model` (`model/place.hpp`) fits layers to devices by their free memory (`model/layer_split.hpp`).
- **Passes in flight.** The pass API (`reserve_passes`, `begin_pass`, `run_pass_stage`, `pass_logits`, `end_pass`, `abort_pass`) keeps P passes of different sequences in one context; each pass owns a handoff buffer per sending device, a range of logits rows and its tickets, and each device runs its passes in formation order.
- **Crossings.** The residual leaves a stage through a host-visible handoff buffer copied inside the stage's own submission, the host waits that ticket and the next stage writes it (`send`, `receive`, `cross`).
- **Devices kept busy.** A device of a split holds a device-side wait between its submissions (`Backend::hold_between_submissions`), so a waiting card keeps its clock.
- **The architecture contract** (`model/architecture.hpp`): a module declares roles and runs each part (embed, mixer, ffn, head, draft) as backend ops on the device its `Step` holds.
  Every residual add a part makes is a row-parallel product at the part's end: `attn_output` or `ssm_out` in the mixer, and in the feed-forward part `ffn_down`, or the routed down projection followed by the shared expert's (`blocks::swiglu`, `blocks::routed_experts`).
  So in every module today the reductions of a tensor split fall exactly at part ends.
- **The planned contract fields** (ADDING-AN-ARCHITECTURE, Future split modes): a `Role` says how its tensor shards, and a `Step` carries the member, the width, the member's heads and a group-sum hook that is the identity at width 1.
- **Exactness machinery.** Kernels choose by an entry's extent class (`Backend::row_class`), never by the batch, so a row's bits follow its entry; the head split (`feat/split-head`, kept as a record, `8b60b7aeb`) showed that a call over some rows of a matrix gives the whole matrix's bits only if the device shapes its kernels by the whole matrix, since the Vulkan tile chooses its split of K from the workgroup count (`split_blocks` in `backends/vulkan/vulkan_backend.cpp`), which follows the matrix's rows.
- **What MULTI-DEVICE decided:** groups of width 2 to 4, column-parallel q, k, v, gate and up, row-parallel attention output and down, attention by heads with each member's KV, two sums a layer, a fixed member order for the sum, a width refused unless heads and quant blocks divide by it, and the tensor split built after the layer split.

### 2.2 Megatron-LM

- Column-parallel then row-parallel MLP and attention split by heads, so each layer needs two all-reduces forward, one after the attention output and one after the MLP's second matrix; layer norm, dropout and the residual are duplicated on every device (Shoeybi et al., arXiv 1909.08053, section 3).
- The embedding is split by vocabulary and summed; the output layer keeps the logits split and fuses cross-entropy so only b x s scalars cross (same paper; `megatron/core/tensor_parallel/cross_entropy.py`).
- Sequence parallelism replaces each all-reduce by a reduce-scatter and an all-gather with the norms sharded along the sequence: the same bytes and twice the synchronization points (Korthikanti et al., arXiv 2205.05198, section 4.2.2), a training saving that adds latency at decode.
- PP x TP: tensor parallelism inside a server's fast links, pipeline stages across slow ones, because tensor parallelism's all-reduces over slow links are impractical (Narayanan et al., arXiv 2104.04473, takeaway 1); a stage's tensor ranks each send 1/t of the boundary tensor and the receivers all-gather.

### 2.3 vLLM (v0.30.0)

- `ColumnParallelLinear`, `MergedColumnParallelLinear`, `QKVParallelLinear` and `RowParallelLinear` in `vllm/model_executor/layers/linear.py`; when the width exceeds the KV heads, each rank holds one KV head and the KV cache is replicated (`num_kv_head_replicas`).
- Row-parallel quantized weights must split on whole quantization groups, refused otherwise (`quantization/auto_awq.py`).
- Two all-reduces a layer for dense Qwen3; for Qwen3-MoE the router is replicated, each expert's intermediate width is split, and the shared expert's output joins before one final all-reduce, so two a layer (`models/qwen3_moe.py`, `fused_moe/moe_output.py`); expert parallelism without data parallelism gives whole experts to ranks and still sums at the layer's end.
- The Gated DeltaNet of Qwen3.5 shards by heads: `in_proj` column-parallel by q, k, v and z sections, the conv by channels, `A_log` and `dt_bias` by V head, the state per head shard, `out_proj` row-parallel: two all-reduces a layer (`layers/mamba/gdn/qwen_gdn_linear_attn.py`); the attention gate is fused into Q and shards with its head (`models/qwen3_next.py`).
- The custom all-reduce (`device_communicators/custom_all_reduce.py`, `csrc/custom_all_reduce.cuh`): one-shot below 256 to 512 KB, where every rank reads every peer and sums in the same order, so all ranks get the same bits; two-shot above, whose peer order rotates by rank; disabled for more than two PCIe-only GPUs and, on ROCm, enabled only on gfx94 and gfx95, so an MI50 falls back to RCCL.
- `VLLM_BATCH_INVARIANT=1` turns the custom all-reduce off and pins NCCL to one algorithm, because by default the reduction's algorithm follows the message size, which follows the batch (`model_executor/determinism/batch_invariant.py`).
- PP x TP: ranks laid out with TP innermost, `IntermediateTensors` (hidden state and residual) crossing stages, pipeline-size batches in flight, and the docs advise pipeline over tensor parallelism without NVLink (`docs/serving/parallelism_scaling.md`).
- MTP and EAGLE drafters run inside the target's tensor group (`vllm/v1/spec_decode/llm_base_proposer.py`, read for SPECULATIVE section 7).

### 2.4 SGLang (v0.5.21)

- Its parallel linear layers are adapted from vLLM's, so the sharding rules match; Qwen3-Next's GDN shards the same way and fuses the reduction with the residual add and the next norm at a layer boundary (`python/sglang/srt/layers/layer_boundary/`).
- DP attention for MoE: attention data-parallel with no replicated KV, hidden rows gathered before the MoE layer and scattered after it, the gather exact because each rank writes only its own rows.
- Deterministic inference forces the one-shot all-reduce on ROCm ("fixed accumulation ordering") and does a reduce-scatter as an all-reduce plus a slice (`kernels/aot/csrc/allreduce/deterministic_all_reduce.hip`, `distributed/parallel_state.py`).

### 2.5 mx-llama.cpp, the user's fork, on these MI50s

- **Per-op rules** (upstream's meta backend, `ggml/src/ggml-backend-meta.cpp`, weights in `src/llama-model.cpp`): weights AXIS_1 (output rows) for Q, K, V, gate, up, the Qwen3.5 gate and the head; AXIS_0 (input) for the attention output, down and `ssm_out`, whose products come out PARTIAL; norms, embedding and router MIRRORED; KV by whole KV heads; the recurrent projections and state split by heads with Qwen3.5's interleaved V heads segmented at K scale; MoE experts split inside each expert, the reduction delayed past the expert weighting so a MoE block costs one sum.
  Two sums a layer: about 129 a token on the dense Qwen3.6-27B and 81 on Qwen3.6-35B-A3B.
- **Uneven work:** card shares from `--tensor-split` rounded to whole KV-head groups and quant blocks, the last card taking the remainder; a card can get nothing (Qwen3-235B's 4 KV heads at width 8) and joins the sum with zeros.
- **The all-reduce** (`ggml/src/ggml-cuda/tp-allreduce.cu`, a port of vLLM's adapted to gfx906): one-shot broadcast of F32 partials by peer stores with flags in fine-grained memory below 32768 to 262144 elements, RCCL in BF16 above; on gfx906 peer stores need fine-grained memory or they sit in the source card's L2 until the kernel ends; the sum `0 + p0 + ... + p(N-1)` is the same on every rank, but the algorithm and the wire precision follow the message size, which grows with the batch, so results depend on the batch and the card count.
- **Bugs met:** a warp race on the release flag, missing system-scope ordering at the end barrier, a two-shot scatter overrunning a slower peer's region, relaxed atomics giving garbage, and a silent fallback to a slow path from a wrong build flag (`tp-notes/car_audit_findings.md`, commits `093f2a38f`, `6d82eb5f8`, `92607b5d1`).
- **Measured decode, tg128, Q8_0** (`tp-notes/BENCH_RESULTS.md`): Qwen3-14B 40.7 tok/s on one card, 63.2 / 74.9 / 76.5 at width 2 / 3 / 4, 54.3 at 8 and 30.2 at 10; Qwen3.6-27B 21.0 on one card, 31.8 at width 2, 40.9 at 4 (48.7 to 51.7 later with whole-token graphs), 16.7 at 9.
  The all-reduce is 25 to 27 percent of decode kernel time; host enqueue took 18.5 ms a token against 3.9 ms of waiting before concurrent per-card dispatch, ranks arrived about 90 us apart, and about 80 percent of all-reduce time was waiting at the barrier (`751b6114cd`, `5d9efc8cad`).
- **Staged** (`-tps T`, `tp-notes/SM_TENSOR_PIPELINE.md`): one meta device knows the stages, layers go to stages contiguously, the residual and every mirrored tensor cross a stage seam card k to card k; full TP wins dense decode at 4 cards and staged wins at 6 and more (Qwen3-14B on 9 cards: 66.9 tok/s at 3 stages of 3 against 45.6 at width 9).
- **On llmx's own measurement of the reference** (`docs/STATUS.md`): Qwen3-32B Q8_0 on two MI50s, tg128 18.7 tok/s with its layer split and 33.6 with its tensor split, and a server at 1 / 4 / 16 / 32 requests of 32.5 / 86.1 / 124.3 / 213.3 tok/s; Qwen3-8B Q8_0 at 512/128 served 78 to 92 tok/s on 4 cards with the tensor split at 16 and 32 users, below one card's 100 to 136.
- **Determinism:** the fork accepts different hashes across modes, greedy output changes with the ubatch even on its layer split, and staged greedy output differs from layer and full tensor on long prompts.

### 2.6 Measured on the Linux machine

Phase 0 (MULTI-DEVICE, Phase 0 results): binary semaphores cross cards as sync files, timeline semaphores do not; a card reads another's memory exported as dma-buf at 9.2 GB/s on one root complex with clocks held high, 4.8 at automatic clocks, and 1.1 GB/s across complexes; a host-relayed sum costs 227 / 346 / 639 us on 2 / 4 / 8 cards; a 20 KB hop chained on the devices through sync files costs 55 us one way.

New on 2026-10-03, two MI50s on one root complex (PCI 83:00 and 86:00), RADV Mesa 25.0.7, Linux 6.17.13, default clocks, a standalone probe built from `tools/vulkan_handoff.cpp`'s helpers, 200 exchanges a chain, median of 5 chains.
An exchange is the all-reduce of width 2: each card writes its F32 partial into the peer's inbox, waits for the peer, and adds both in member order; the probe checks every sum.

| message | dispatch floor, no peer | sync files, one thread | sync files, a thread per card |
|---|---:|---:|---:|
| 20 KB (one 5120-wide row) | 13 us | 137 to 140 us | 188 us |
| 160 KB (8 rows) | 14 us | 137 to 167 us | 207 us |
| 1.25 MB (64 rows) | 26 us | 367 to 387 us | 415 us |
| 10 MB (512 rows) | 88 us | 2.0 ms | 2.1 ms |

- Every sum through sync files was correct, and the concurrent submission tested, a submitting thread per card, did not lower the cost; telling the kernel driver's wait from host recording and member arrival needs step 0's separate timings.
- A device-side wait inside one submission, spinning on a flag the peer writes, never saw the peer's writes within 20 ms: with the inbox in device-local memory of the `VK_AMD_device_coherent_memory` uncached type exported as dma-buf, with host memory imported into both cards, and with the flag written by a shader store, after a host-release barrier, and by a command-processor buffer marker.
  The writes became visible only at the writer's submission boundary, which matches HIP on gfx906 without fine-grained memory; under Vulkan the mapping of an imported buffer is the driver's, so this path is closed to llmx without a driver change.
  The first run's spin, before it was bounded, cost each card one ring timeout with a soft recovery and nothing else.
- The same wait with the Vulkan memory model, on the coordinator's review, also never saw the peer's writes: shaders with the `VulkanMemoryModel` capability and `vulkanMemoryModel` and `vulkanMemoryModelDeviceScope` enabled on both cards, the data and the flag as `atomicStore` with release and `gl_SemanticsMakeAvailable` after an acquire-release barrier, the spin and the sum as `atomicLoad` with acquire and `gl_SemanticsMakeVisible`, at `gl_ScopeDevice` and at `gl_ScopeQueueFamily`, on both inbox placements: every wait timed out.
  The driver's ISA (`RADV_DEBUG=shaders`) shows why: the release stores and acquire loads become `buffer_store_dword` and `buffer_load_dword` with `glc` and `s_waitcnt`, which bypass only the L1, and nothing writes the L2 back or invalidates it, for which gfx906 has no shader instruction; device scope, the widest Vulkan gives, is satisfied inside one card's L2, and another card is outside every Vulkan scope.
  So on this driver and these cards a sum waits at a submission boundary; a driver that maps imported or shared memory uncached on the writer, as HIP's fine-grained memory does, would change that, and step 0 reruns the test on each driver it meets.

### 2.7 What the research decides

1. Two reductions a layer is the floor for every Qwen layer kind (attention, Gated DeltaNet, dense and MoE feed-forward); sequence parallelism doubles the synchronization points and is not taken.
2. The reductions sit at part ends in llmx's modules, so the runtime can sum between parts and a module needs no mid-part hook.
3. A fixed reduction order gives every member the same bits, as vLLM's one-shot and the fork's broadcast do; an algorithm or wire precision chosen by message size breaks batch invariance, as the fork's results and vLLM's batch-invariant mode show, and is not taken.
4. On Vulkan here a reduction costs about 137 us at decode sizes, so a width-2 group pays about 17.5 ms of sums a token on a 64-layer model; that, not bandwidth, sets Vulkan's gain (section 5).
5. Widths above 4 lose on this hardware in every measurement found (the fork, MULTI-DEVICE's skew figures), and the staged form recovers them.

## 3. Words used here

```
Group      an ordered set of devices of one backend kind and one device profile that run the same layers together; width 1 is a single device.
Member     one device of a group; member 0 is the first listed.
Stage      a contiguous range of layers on one group; the embedding goes with the first stage and the head with the last.
Partial    a member's share of a row-parallel product, summed over the group at the end of a part.
```

A layer split is stages of width-1 groups, a tensor split one stage of width W, and a staged tensor split S stages of width W.

## 4. Design

### 4.1 One placement for all three splits

- `Placement` gains its groups: the backends of each group in member order, every group of one width.
  A role names a group, not a backend; with no groups given, each backend is a group of one, which is today's placement and keeps today's code path.
- A stage is a run of layers on one group, so the pass API, passes in flight, the handoff buffers, admission over pools and the fit's balance by layer counts are unchanged in shape: the residual leaves a stage from member 0's handoff buffer (the residual is the same on every member) and the next stage writes it into every member's residual.
- The scheduler keeps P passes in flight over S stages, P = S by default; a single group is one stage and runs one pass at a time, as one device does.

### 4.2 How each op shards

Each role declares how it shards, and one owner, a new `model/shard.hpp`, turns that into each member's spans of rows or columns and checks legality.
A declaration is an axis, a unit and an order:

- **axis**: output rows (column-parallel, each member's product complete for its rows), input columns (row-parallel, each member's product a partial summed at the part's end), or none (replicated, the whole tensor on every member);
- **unit**: the indivisible run along that axis, a head, a head and its gate, a KV head, or a single row or block;
- **order**: contiguous units per member, or tiled as Qwen3.5's V heads (unit j = s Hk + h belongs to K head h), so a member takes its K heads' V heads from every tile, on either axis: `attn_qkv`'s v rows, z, alpha, beta, `ssm_a` and `ssm_dt` are tiled rows, and `ssm_out` is tiled columns with the same permutation.

A tiled member's share is several spans; on the column axis each span starts and ends on a whole quant block, checked span by span and not only over the member's total.

| role | qwen3 / qwen3moe | qwen35 full attention | qwen35 linear attention |
|---|---|---|---|
| attention / mixer norm, q and k norms | replicated | replicated | replicated; `ssm_norm` is one V head wide and serves every head |
| q, k, v | rows by unit: q heads, KV heads | rows by unit: a q head with its gate, KV heads | `attn_qkv`: q and k rows by K head, v rows tiled |
| gate z, alpha, beta, `ssm_a`, `ssm_dt` | | | rows tiled by V head |
| conv taps | | | the channels of the member's q, k and v rows |
| attention output, `ssm_out` | columns by head | columns by head | columns tiled by V head |
| feed-forward norm | replicated | replicated | replicated |
| gate, up | rows | rows | rows |
| down | columns | columns | columns |
| router, routed experts, shared expert | later (section 4.10) | | |
| embedding | replicated | replicated | replicated |
| head | rows (vocabulary) | rows | rows |
| output norm | replicated | replicated | replicated |

- **Attention, KV and state:** each member runs attention and the recurrence over its own heads, keeps the KV of its KV heads and the recurrent state of its V heads, and needs no exchange inside the mixer.
  Where the width exceeds the KV heads and divides into them evenly, each KV head is replicated on width / Hkv members as vLLM does, its K and V products computed on each; the fit counts the replicated cache.
- **The tables and the head:** every member holds the rope tables; the head's vocabulary rows are split evenly, each member writes its slice of every logits row into one host buffer imported into every member, so the host reads whole rows with no join copy.
- **Legality, refused at load with the projection named:** the width divides the q heads, the K heads and the V heads, and divides the KV heads or is a multiple of them; every column split falls on whole quant blocks per member; every split is even, so no member holds a remainder (section 4.5); a group's members are one backend kind with one device profile, so a sum computes the same bits on every member.
  On the models in view: Qwen3-32B (64 q, 8 KV heads, F 25600) is legal at 2 and 4 in every type and at 8 only where `ffn_down` is Q8_0 (25600 / 8 = 3200 is not whole 256-value blocks); Qwen3.6-27B (24 q, 4 KV, 16 K and 48 V heads, F 17408) is legal at 2 and 4, and at 8 only with KV replicated and `ffn_down` in Q8_0.

Each member's arithmetic is the one-device kernel over its shard, so a group adds one new op, the sum.

### 4.3 The reductions

- **Where:** at the end of the mixer and of the feed-forward part of every layer, two a layer, and none in the embedding (replicated) or the head (its slices are joined in host memory).
- **What a module changes:** the part's final residual projection goes through one block helper that adds into the residual at width 1, as today, and writes the member's partial into the slot the group gives at width W; the module reads its member's head counts from the `Step`.
  At width 1 every module runs the same ops as on main, which is how gate 1 holds.
- **What the runtime does:** it runs each part on every member, then asks the group to sum the partials into every member's residual, then runs the next part.
- **The sum** is `x + (((p0 + p1) + p2) + p3)` in member order, F32, the same kernel on every member, so every member's residual keeps the same bits; it is one algorithm at every message size, and a two-shot form for large messages is allowed only if it adds each element in the same order, so its bits are the one-shot's.
- **The collective is a backend concern** with two implementations from its first branch, which is the second use that justifies the seam:
  - **CPU:** the members are CPU backends in one process; the partials are host memory and the sum reads them in member order.
    It makes every group test runnable on the hosted runners.
  - **Vulkan:** each member's partial is written into every peer's inbox, device memory exported as dma-buf, and the member's submission signals a binary semaphore exported as a sync file; each member's sum runs in its next submission, which waits on its peers' semaphores, and reads its inbox.
    Inboxes alternate by parity, so a member writes an inbox only after every peer has read its previous contents, which the wait chain guarantees, as MULTI-DEVICE's exchange epochs require.
    It is Linux-only, as sync files and dma-buf are; on Windows a group of Vulkan devices is refused by name and the Radeon VII, a single device, is untouched.
  - **ROCm,** when that backend exists: peer stores into fine-grained memory with flags, the fork's measured path, behind the same call.
- **Cost on Vulkan:** about 137 us a reduction at decode sizes on two cards (section 2.6), 128 a token on a 64-layer model, about 17.5 ms; at 512 prompt rows about 2.0 ms a reduction.
- **If a device-side wait works** on some driver (step 0's first measurement) and that driver documents a cross-device visibility and ordering contract for the memory involved, the Vulkan collective keeps its call and its inboxes and replaces the sync files by a release store of a flag per peer after the partial and an acquire spin before the sum, bounded far below the ring timeout with an error flag the host checks; a member's work then stays in one submission per stage, as on one device, and a sum costs about the dispatch floor plus the PCIe write, an estimated 15 to 25 us (the measured three-dispatch floor is 13 us).
  A passing probe alone does not select flags: the Vulkan shader specification disallows the CrossDevice scope and gives each device its own Device-scope instance, so a probe shows one driver's behaviour, not a guarantee.
  Without both the contract and the probe, as on RADV and gfx906 today, the collective keeps API synchronization, the sync files, with a submission boundary at every sum.

### 4.4 Exactness and the correctness gate

A row-parallel projection's partials regroup the sum over its input, so a tensor group does not give one device's bits; no rule here weakens an existing one, and the rule a group keeps is new.

**The rule proposed:** within one group shape (the same width, backend kind, device profile and shard boundaries), a row's logits are a function of its entry alone, as on one device today:
- identical run to run;
- identical however the row is batched: alone or beside other sequences, in slices of a prompt, at any P, in the server and the CLI, and recomputed by class on a resume (the kernels on each member follow `row_class`, and the sum is elementwise in a fixed order);
- identical across stage counts at one width: a staged split of S stages of width W gives the bits of one stage of width W, since the layer split between groups is exact;
- **not** identical across widths or to one device: the difference is the regrouped sums of the row-parallel projections, computed in F32.

**The gate:** every existing gate holds unchanged for width 1; a group is held to the HF reference at the bounds of the precision its path computes in (F32 sums, the same activation dtype as one device), on the tiny F32 fixtures (new ones with even head and width counts, generated by `tools/gen_baseline.py`), on the pinned Qwen3-8B Q8_0 and on the 0.8B qwen35 files of the gate; to one device's logits through the device-reference criterion of `tests/common.py` (ranking, NLL, calibration and greedy agreement), as a device is held to the CPU today; and with the count of changed greedy tokens against one device on the fixed set reported at every merge.
The long-context checks (raw and chat 16k against file-exact HF) run before the first device merge, as AGENTS.md requires of a change of prompt or decode numerics on a device.

**Considered and not recommended, an exact mode:** every product split by output rows only, with gathers of the attention output, the residual and the SwiGLU output instead of sums, and every column-parallel kernel shaped by the whole matrix as the head split did, which gives one device's bits at every width.
It needs four exchanges a layer instead of two and more bytes: on Qwen3-32B a row's logical payload is Q + 2E + F = 44032 floats a layer against 2E = 10240 for the two sums, 4.3 times; what one member receives at width W is (W - 1) / W of each gathered vector against (W - 1) E for each sum, so at width 2 it receives 22016 floats a layer against 10240, 2.15 times.
At decode the exchanges' latency dominates, about 35 ms a token on Vulkan instead of 17.5, and the whole-shape argument is needed on every product.
Decision 1 (section 8) keeps it as the fallback, not built now.

### 4.5 Skew

**What it is:** every reduction waits for the slowest member, so each sum costs the exchange plus the spread of the members' arrival times, and that spread grows with the width.
Its sources here:
- **Launch skew:** one host thread records each member's part in turn, so member W-1 starts later than member 0 by W - 1 recordings; recording an 8B pass took 2.6 to 5.2 ms on one card (`docs/STATUS.md`, layer split phase 3), about 4.6 to 9.2 ms a member a token on a 64-layer model, so at width 4 one thread records 18 to 37 ms a token against about 18 ms of device work, and the host, not the devices, sets the pace.
  The fork met the same: 18.5 ms of host enqueue a token and ranks 90 us apart until it dispatched per card.
- **The exchange's own wait:** a sync-file wait per peer, N - 1 of them a sum, each through the kernel driver; the slowest of N - 1 PCIe latencies is 8.2 us at 4 cards and 20.3 us at 8 for HIP peer stores (MULTI-DEVICE), and Vulkan's spread at 4 and 8 is not measured yet.
- **Uneven work:** a member with more heads, more rows, a remainder or a routed expert imbalance arrives last at every sum.
- **Card variance:** clocks, temperature and link training differ per card (a link can train at Gen1 or Gen3), and a card beside another job runs slower.
- **Topology:** a group across root complexes reads its peers at 1.1 GB/s instead of 9.2 and its 160 KB hop took 132 us instead of 72.

**What it costs a token** on Qwen3-32B Q8_0 (64 layers, 128 sums), single request, from section 2.6 and the fork's skew figures; widths 4 to 8 add HIP's peer-store skew to the measured two-card Vulkan sum, an optimistic extrapolation and not a proven bound, until step 0 measures wider Vulkan groups:

| width | sum, us | sums a token, ms | share of a token |
|---:|---:|---:|---:|
| 2 | 137 (measured) | 17.5 | 35 percent |
| 4 | 145 (est.) | 18.6 | 51 percent |
| 6 | 150 (est.) | 19.2 | 60 percent |
| 8 | 157 (est.) | 20.1 | 66 percent |

Width 6 is not legal on Qwen3-32B (64 q heads) and shows the trend only.
The exchange and its skew do not shrink with the width while the compute does, so past 4 the sums are most of a token on Vulkan, and on ROCm, where a sum costs 6 to 17 us plus skew, skew is the larger part at 8 (the fork's 27 percent of an 8-card token).

**What each system does about it, and what llmx takes:**
- **Fewer reductions:** two a layer is the floor for these models (2.7); sequence parallelism and per-two-layer schemes add synchronization points or change the math, so llmx keeps two a layer, fused with the residual add.
- **Amortize per pass:** a pass of R rows pays each sum once, so wider passes and verify rows of speculative decoding divide its fixed cost by the rows (sections 4.7 and 4.8), which llmx takes.
- **Overlap:** two micro-batches of one pass, one exchanging while the other computes (vLLM's and SGLang's dual-batch overlap), pays only from 64 to 128 rows and does nothing for one request, so llmx takes it later, for prompts, only where measured (step 7).
- **Balanced work:** even splits only, with no remainder member, no zero-width member (the fork's width 8 on 4 KV heads) and replicated KV heads rather than uneven ones, which llmx takes as a legality rule.
- **Launch skew:** a submitting thread per member from width 4 on, under the one-owner-per-backend rule of MULTI-DEVICE (each member's backend belongs to one submitter), measured first at width 2 and 4 in step 0; the fork's concurrent per-card dispatch is the same remedy.
- **Device-side waits:** the chain stays queued ahead on the devices through sync files, never relayed by the host (227 us); a wait inside a submission is closed on Vulkan here (2.6) and is the ROCm backend's path.
- **Topology:** members on one root complex; the fit reads each device's PCI address and refuses a group across complexes by name unless the user forces it with the flag's own form.
- **Bounded width, staged beyond it:** a group is at most 4 wide, and more cards form stages: 8 cards as 2 stages of 4 keep width 4's skew per sum, take the same time per token for one request (each token crosses both stages, each with half the layers), and with P = 2 serve twice the requests; the fork measured staged ahead of full width from 6 cards.

**Recommended:** width at most 4, 2 on Vulkan unless step 0's 4-card measurement shows width 4 pays there, even shards, one root complex per group, a submitting thread per member from width 4, and stages beyond 4 cards.
At 8 cards the model's optimistic figures favour width 8 for one request (section 5), but every measurement found puts width 8 below width 4 and the staged form above both from 6 cards, the figures leave out Vulkan's unmeasured skew at 8, one thread cannot record 8 members in time, the eight cards span two root complexes here (four at 83 to 8c, four at c3 to cc), and 2 stages of 4 serve twice the requests at P = 2; width 8 stays refused until step 0 measures a group of 8 on one complex, which this machine does not have.
**Gate:** at every width offered and at 2 stages of 4, the exposed wait a sum and a token, read from GPU timestamps on every member, and the per-token cost against the model above, recorded at each merge that touches the group's execution.

### 4.6 Placement, fit and flags

- **Flag:** `--group-width N` (MULTI-DEVICE's sketch) forms groups of N consecutive devices of the `--device` list; the list's length must be a multiple of N, and the groups become the stages, so `--device vulkan:0,vulkan:1,vulkan:2,vulkan:3 --group-width 2` is 2 stages of 2 and `--group-width 4` one stage of 4.
  `--layer-shares` gives one share per group.
  The flag means the same on every backend or is refused, and it is added to `docs/USAGE.md` and `print_usage` with its branch.
- **Fit:** the fit treats a group as one device whose budget is the least member's free memory and whose footprint is a member's share: its rows and columns of each matrix, its heads' KV and state, the replicated tensors whole, its arena, its share of the logits rows, the collective's partial slot and inboxes, and, on member 0 of every stage but the last, the handoff buffers.
  The inboxes are counted at their peak live allocation for the rows the context reserves: with two parity slots and an inbox per peer, 2 (W - 1) x rows x E x 4 bytes a member (about 42 MiB at width 2 and 1088 rows of Qwen3-32B, 128 MiB at width 4), in checked arithmetic, with a hand count and a refusal case in the existing fit tests (`placement`) and no new fit owner.
  A group that does not fit is refused with the member and the bytes, as a device is today.
- **Loading:** column-parallel rows are a contiguous span of the file per member, or one span a tile for tiled V heads; a row-parallel shard is each row's block span or spans (several for `ssm_out`'s tiled columns), which the loader packs on the host as it streams into one upload per member, so no member holds a whole tensor.

### 4.7 Serving many users

- **Passes:** the scheduler, the pass API and the policy core are unchanged; a stage is a group, P = S passes in flight by default, and every member of a stage runs each pass in lockstep, recorded by its submitter.
  A single group runs one pass at a time, so its host gap (sampling and the first recording of the next pass) is exposed as on one device.
- **Width of a pass:** passes widen up to the decode kernel's columns as on one device, and a wider pass pays each sum once: on Qwen3-32B at width 2, a 32-row pass costs about 170 ms of member work and 32 ms of sums (640 KB a sum, between the measured 160 KB and 1.25 MB), 158 tok/s, where the 2-card layer split at P = 2 gives about 188 (estimates from the measured 8B curve scaled by bytes).
  So at 16 to 64 users the layer split keeps the throughput lead on Vulkan and the tensor split the latency lead; the staged form sits between, and the choice is the user's per deployment, as MULTI-DEVICE's Which split for which case says.
- **Predicted stage time:** the cost model of a pass (phase 3, step 9) adds the group's sum cost per reduction and per row, measured at load, so assembly keeps passes level.
- **Prefill beside decode:** unchanged; a 512-row chunk pays about 2.0 ms a sum on two cards, 0.26 s over a 64-layer model against about 0.78 s of member work, so long prompts prefer the layer split or the staged form until overlap (step 7) hides it.
- **KV, state and admission:** one block pool per group, whose members' storages share block ids, so a sequence keeps one block table per stage as today and admission, prefix reuse, forks, donors and pauses are unchanged; a member's KV is its heads' share, so per-card KV capacity matches an even layer split's where the KV heads divide by the width; where the width exceeds the KV heads, the replicated heads raise each member's KV bytes by W / Hkv against that layer split (twice at width 4 on two KV heads), which the fit counts.
  The recurrent state's slots are shared ids across the members' state storages, each holding its V heads.
- **Host tier and checkpoints (SPECULATIVE 2b):** `save_host` and `restore_host` copy each member's blocks and state slot into that member's slabs; a checkpoint is a slot id held by every member.
- **What vLLM and SGLang do that applies:** one scheduler with continuous batching and chunked prefill over SPMD workers that run each batch in lockstep, which is llmx's scheduler with members recorded per pass; CUDA-graph fixed shapes pad batches to captured sizes, which llmx's row classes replace without padding; DP attention is the MoE path of MULTI-DEVICE phase 6b, not this plan.
- **Gates:** `tools/server_load.py` at 1, 4, 16, 32 and 64 users, 128/128, 1024/128 and the mixed set, greedy and at the defaults, output throughput, time to first token and inter-token p50 and p99, for the tensor split and the staged form against the layer split on the same cards, mx-llama.cpp's ROCm tensor split and vLLM (matched bits, as phase 3 decided); server and CLI greedy text equal within one group shape at every concurrency; `tools/server_mix_check.py` over a group and over 2 stages of 2.

### 4.8 Speculative decoding over a group

One system, as SPECULATIVE section 3 requires: proposers over one verify, accept and retract path, with no path specific to groups.

- **The embedded MTP drafter** (SPECULATIVE section 7) is a layer of the target's last stage: its roles shard as a layer's do (its attention by heads with its KV in the group's pool, its feed-forward rows and columns, `eh_proj` replicated, a 2E by E product the draft chain reads whole), and the head is the target's vocabulary-split head.
  The draft chain's argmax over the vocabulary becomes each member's argmax over its slice and one small exchange of (value, id) pairs, the largest value winning and the lower id on a tie, which gives one device's id exactly; every member then embeds the same id from its replicated table.
  A draft row costs two sums and one argmax exchange, about 0.4 ms on two Vulkan cards, beside its share of the block and the head.
- **The verify** is a pass with k + 1 rows a sequence, so it pays each of the 2L sums once for all its rows: on a group, whose decode is latency-bound by the sums, speculative decoding gains more than on one device, and the scheduler's measured pass price (`spec::draft_depths` over `PassCost`, landed with SPECULATIVE's step 5) sees a pass whose fixed cost the sums raise, so it drafts more there with no change of its own.
- **Rollback:** `retract` returns blocks in the group's pool, the same table on every member; the state rerun reads each member's saved inputs of its own V heads and writes its own slot, the conv and the recurrence being per channel and per head, so the rerun is local to each member with no sum, and SPECULATIVE's estimate of 1 to 2 ms a rejected round on the 27B is divided by the width.
- **vLLM and SGLang** run MTP and EAGLE drafters inside the target's tensor group, as here.
- **Gates:** drafts on against off identical within one group shape, greedy and seeded (the `spec` CTest over CPU groups, `common.check_drafts` over a group); acceptance on Qwen3.6-27B-MTP within the margin of the same file on one device; decode tok/s at 1 to 4 users with drafts against without on a group of two.

### 4.9 Code owners and the Backend interface

- `model/shard.hpp` (new): the shard plan, its legality and refusals, and a member's footprint; one owner for how a model splits over a group.
- `model/architecture.hpp` and the modules: a shard way per role, the member's head counts and partial slot in `Step`, and the residual block helper in `model/arch/blocks.hpp`.
- `model/runtime.hpp`: groups in `Placement`, a part run per member and the sum between parts, per-member arenas, one pool per group, the head's slices into the shared host rows; the per-member bookkeeping goes into a file of its own if `runtime.hpp` (1379 lines) would otherwise grow by more than the hooks; decision 6 moves it out first, in step 0b.
- `model/layer_split.hpp` and `model/place.hpp`: groups as fit units, the topology check, `PlacementRequest::group_width`.
- `inference/load.hpp`: shard spans and packed column shards in the upload entries.
- `backends/backend.hpp`: one interface, the group's collective (made once for a list of backends of one kind, giving each member's partial slot and summing a pass's rows into every member's residual), and an output row stride on `matmul_logits`; nothing else.
- `backends/cpu/` and `backends/vulkan/`: the two collectives.
- `cli/main.cpp`: `--group-width`.
- `server/` and `inference/spec.hpp`: nothing new beyond the pass cost's sum term.

### 4.10 Out of scope

- MoE layers in a group (qwen3moe, qwen35moe): the fork measured little gain (Qwen3.6-35B-A3B on 8 cards at 4 stages of 2: 64.7 tok/s against its layer split's 75.2) and MoE decode is bound by dispatches, which a group does not divide; refused by name until step 8 measures expert rows against experts by member.
  When built, the routed and the shared expert's partials accumulate into one cleared partial slot before the part's single sum, and width 1 keeps today's order of the two residual adds.
- Data-parallel attention with expert parallelism (MULTI-DEVICE phase 6b), replicas (phase 5), multi-node.
- Uneven member shares, groups mixing device kinds or profiles, a group of the CPU and a card.
- A wire format other than F32, and any algorithm chosen by message size that changes the order of a sum.
- Command buffers recorded once and replayed (phase 3's Not doing).

## 5. What it would deliver

Single-request decode on Qwen3-32B Q8_0, modeled from measured parts: a token is 61.7 ms of device work on two MI50s at one row (the layer split's measured pass), of which about 2.9 ms is the dispatch floor that does not split (about 10 dispatches a layer at 4.1 to 5 us); a member takes (61.7 - 2.9) / W + 2.9 ms, plus the sums of section 4.5.

| placement | tok/s, Vulkan sums (137 us measured at width 2, HIP skew added past it) | tok/s, a 15 us sum plus HIP's skew (ROCm peer stores) | measured references |
|---|---:|---:|---|
| layer split, 2 cards | 16.2 (measured) | | mx-llama.cpp layer 18.7, Vulkan reference 13.3 |
| tensor split, width 2 | 20.1 | 29.2 | mx-llama.cpp tensor 33.6, vLLM AWQ TP 2 38.1 |
| tensor split, width 4 | 27.6 (est.) | 48.6 (est.) | |
| tensor split, width 8 (only where `ffn_down` is Q8_0) | 32.9 (est.); 13.5 to 27 with one recording thread | 67.7 (est.) | mx-llama.cpp: width 8 below width 4 on Qwen3-14B (54.3 against 76.5) |
| 2 stages of 4, 8 cards, one request | 27.5 (est.) | 48.3 (est.) | mx-llama.cpp: staged ahead of full width from 6 cards |

- With a device-side wait at 15 to 25 us a sum, on a driver that documents it, Vulkan's figures would move to the ROCm column's: about 28 to 29 tok/s at width 2 and 46 to 49 at width 4 on Qwen3-32B Q8_0 (estimates), and the plan would build step 3 on flags; without it the Vulkan column stands.
- On Vulkan a group of two gains about 1.24 times the layer split for one request, and width 4 about 1.7 times if the optimistic extrapolation holds; reaching the ROCm references needs a cheaper cross-card wait (the ROCm backend, or a Vulkan driver that maps imported memory uncached) and per-card decode near the fork's, whose 33.6 tok/s implies about 25 ms of member work against llmx's modeled 32.
- At 16 to 64 users the layer split keeps the throughput lead on Vulkan (section 4.7); the staged form and the group are measured against it and the references there.
- With speculative decoding the group's per-pass sums are shared by k + 1 verify rows, which is where the group's single-request lead grows most (section 4.8).
- Every figure above but the measured ones is an estimate; step 0 and each step's timing replace them.

## 6. Order of work

Each step is a branch from main, at most two commits, with a STATUS block opened before its code; every step keeps width 1 byte-identical to main on the CPU, one MI50 and a layer split, with `llmx-split-check` bit-identical, and runs the merge gates of its tier in AGENTS.md.

| # | branch | brings | gate |
|---|---|---|---|
| 0 | `tools/tp-exchange` | `llmx-vk-handoff exchange`: the probe of section 2.6 as a mode of the existing tool, with N cards; measurements only | first, the device-side flag wait with the Vulkan memory model (release and acquire with MakeAvailable and MakeVisible at device and queue-family scope, both inbox placements, the spin bounded) and the ISA checked for an L2 writeback or invalidate, on every driver at hand, which with a documented driver contract for cross-device visibility and ordering decides flags, and otherwise keeps sync files, for step 3; then the exchange at widths 2, 3 and 4 on one root complex and at 2 across complexes, 20 KB to 10 MB, each sum checked; the members' arrival spread; recording time a member a layer on Qwen3-32B and Qwen3.6-27B; the numbers recorded in STATUS and section 5 updated; the decision on the Vulkan collective (decision 2) |
| 0b | `refactor/runtime-split` | decision 6: a minimal move-only split of `model/runtime.hpp` by concern, in parallel with step 0, nothing renamed or changed in behaviour | byte identity against main on the CPU and one MI50 and a layer split, `llmx-split-check` bit-identical, CTest, the suite's docs and dead-code components, the hosted run, and a timing round against main with a perturbed-layout control on the hot path |
| 1 | `feat/tp-shard` | the shard ways on the roles of qwen3 and qwen35, `model/shard.hpp` with legality and refusals, a member's footprint, the loader's shard spans and packed column shards | `model-validation` with each refusal's text in `tests/data/model_refusals.txt`; every member's shards reassemble each tensor's bytes, for every type, the fused q, k and v sections of `attn_qkv` and the tiled V heads on both axes (`ssm_out`'s columns included); the fit's member footprint against hand counts; width 1 unchanged; the dead-code list names step 2 for the interfaces it leaves to step 2 |
| 2 | `feat/tp-cpu` | groups in `Placement`, parts per member and the sum between parts, the CPU collective, one pool per group, the head's slices, `--group-width`, staged groups with passes in flight, the fit over groups | hosted: tiny F32 fixtures with even shapes at width 2 against HF at the F32 bounds (`f32`, `qwen35`); 1 stage of width 2 against 2 stages of width 2 on four CPU backends bit for bit (`split`, `placement`); the row-class and batch checks over a group (slices, two sequences, P = S and 2S, failures mid-pass); `server-passes-cpu` and `server-resume` over CPU groups; `cli` refusals; device-reference criterion against width 1 |
| 3 | `feat/tp-vulkan` | the Vulkan collective (inboxes, sync files, parity), a submitting thread per member if step 0 asks for it, the topology check | `backend-vulkan`: the sum's bits equal on every member and to the CPU's order, every size and refusal; `vulkan-lifetime` for inboxes and semaphores; on two MI50s: HF gate (Qwen3-8B Q8_0 baseline, tiny fixtures), the 16k checks, the identity rules of 4.4, greedy agreement against one device reported; timing: Qwen3-8B and 32B Q8_0 decode at 1 to 8 users and pp512 to pp16384 against the layer split, mx-llama.cpp's ROCm tensor split and vLLM; the skew gate of 4.5; the Radeon VII and Windows builds unchanged |
| 4 | `feat/tp-staged-serve` | the server over groups and stages: the sum term in the pass cost, `--passes` over staged groups, the fit's checks of handoff and logits on groups | section 4.7's gates at 1 to 64 users on 2 and 4 MI50s and 2 stages of 4 on 8, with the skew gate at 8; `tools/server_mix_check.py`; server and CLI equal within one group shape |
| 5 | `feat/tp-qwen35` | the linear attention's shards (K heads with their tiled V heads, conv channels, per-head state), states, checkpoints and marks per member | `qwen35` component at width 2 on CPU groups against HF; `arch-qwen35` checks over groups (forks at checkpoints, retract, marks, rerun) bit-identical within a group shape; Qwen3.6-27B Q8_0 on 2 and 4 MI50s: HF gate on the qwen35 gate files, 16k checks, timing against the layer split and the references |
| 6 | `feat/tp-spec` | the embedded drafter's shards and the argmax exchange, the verify and rerun over groups in the CLI and the server | section 4.8's gates |
| 7 | `perf/tp-overlap` | two micro-batches of a prompt pass, one exchanging while the other computes, only from the size where step 0 or step 3 measured a gain | prompt speed against step 3 at 512 to 16384 tokens with bits unchanged (the sum's order does not change) |
| 8 | `feat/tp-moe` | if measured worth it: experts by rows where every down shard is whole blocks, else by member, with the router replicated | the MoE HF gate at width 2, Qwen3-30B-A3B and Qwen3.6-35B-A3B decode and serving against the layer split; not built if the measurement shows no gain |

Steps 0 and 0b run first, in parallel, after this plan lands; steps 1 and 2 follow them and need no device beyond the probe's cards; step 3 is the first device merge; steps 5 and 6 follow the order of SPECULATIVE's own steps where they share files.

## 7. Risks and how each is measured early

1. **The Vulkan sum costs more than the group saves.** Measured at 137 us a sum on two cards (2.6); step 0 measures widths 3 and 4 and the spread before any Vulkan group code, and the user decides on the collective with those numbers.
2. **The host records too slowly for a group.** One thread records W members; step 0 measures recording per member a layer, and step 3 adds a submitting thread per member only where that measurement says the host limits.
3. **A device-side wait stays closed on Vulkan.** Measured closed on RADV and gfx906, with and without the Vulkan memory model, and explained by the ISA (2.6); the design does not depend on it, and a driver or a ROCm backend that offers it enters behind the same collective.
4. **Skew past width 4.** Bounded by the legality and width rules and measured at each merge (4.5).
5. **Quant blocks forbid a width.** Checked per projection at load with the projection named (4.2); the model list's legal widths are written in step 1.
6. **Memory.** Replicated embedding, norms and tables on every member, and replicated KV heads past the KV head count; the fit counts them, and step 1's footprint tests hold them.
7. **The exactness rule fails a gate.** Decision 1 accepted it; the exact mode's cost stays written down as the fallback.
8. **`runtime.hpp` grows past its owner.** Decision 6: the move-only split of step 0b comes first.
9. **Windows and the Radeon VII.** A group of Vulkan devices is refused on Windows by name; width 1 keeps every path, and the Windows build and the Radeon VII identity check run at every device merge.

## 8. Decisions (2026-10-03)

The user delegated the plan's questions to the coordinator and the other developer, in these words (2026-10-03, about 21:50): "Open questions to me goes to [the other developer] and when you two get to agreement then continue", the bracket replacing the internal name.
The coordinator proposed answers in the shared development log (PROPOSAL re:tensor-split, 22:05) and the other developer agreed with two refinements (ANSWER re:tensor-split, 22:07): the reference floor stays visibly open, and the delegation is recorded here.

1. **Exactness:** the rule of 4.4, identical run to run, however a request is batched, at any P and across stage counts at one width, and not identical to one card or across widths; held to the HF reference at the bounds of its precision and to one card through the device-reference criterion, with the changed greedy ids reported at every merge.
   No exact mode now; it stays written in 4.4 as the fallback if a gate fails.
2. **The Vulkan collective:** step 0 runs before it is chosen; flags only with a passing probe and a documented driver or platform contract for cross-device visibility and ordering, otherwise sync files; whether step 3 is built now, or steps 1 and 2 land alone for a later ROCm backend to reuse, is decided on step 0's numbers.
3. **Width:** at most 4 per group, staged beyond it; head, KV-head, V-head and quant-block legality checked at load; wider groups refused until measured on one root complex.
4. **Flag:** `--group-width N`.
5. **MoE:** deferred to step 8, where the routed and the shared expert's partials accumulate into one cleared partial slot before one sum, and width 1 keeps today's order of the two residual adds.
6. **`runtime.hpp`:** a separate, minimal, move-only split by concern first, with its own gates (byte identity on the CPU and a device, CTest, the hosted run, and code-layout scrutiny of the hot path), so the group's bookkeeping lands in a file of its own.
7. **The merge gate:** the tensor split's purpose and gate is its single-request and few-user gain over the layer split; every 16 to 64 user figure and every reference figure (mx-llama.cpp's ROCm tensor split, vLLM) is reported beside it, and a reference floor not met stays visibly open under the first-support policy of AGENTS.md, never called passed because the layer split is beaten.
   Every width-1 gate stays unchanged, and no dtype or half-weight prerequisite is waived by these choices.

## 9. Sources

- Megatron-LM: Shoeybi et al., https://arxiv.org/pdf/1909.08053 ; Korthikanti et al., https://arxiv.org/pdf/2205.05198 ; Narayanan et al., https://arxiv.org/pdf/2104.04473 ; https://github.com/NVIDIA/Megatron-LM/blob/core_v0.19.2/megatron/core/tensor_parallel/
- vLLM v0.30.0: https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/linear.py ; .../layers/vocab_parallel_embedding.py ; .../layers/mamba/gdn/qwen_gdn_linear_attn.py ; .../models/qwen3_moe.py ; .../models/qwen3_next.py ; .../distributed/device_communicators/custom_all_reduce.py ; https://github.com/vllm-project/vllm/blob/v0.30.0/csrc/custom_all_reduce.cuh ; .../model_executor/determinism/batch_invariant.py ; .../docs/serving/parallelism_scaling.md
- SGLang v0.5.21: https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/layers/linear.py ; .../srt/layers/dp_attention.py ; .../srt/distributed/parallel_state.py ; .../docs/docs/advanced_features/deterministic_inference.mdx ; https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/
- Vulkan scopes (CrossDevice disallowed, a Device-scope instance per device): https://docs.vulkan.org/spec/latest/chapters/shaders.html
- Batch invariance: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ; tensor-parallel invariance at a cost: https://arxiv.org/html/2511.17826v2
- llama.cpp's tensor split on PCIe cards: https://github.com/ggml-org/llama.cpp/pull/19378
- mx-llama.cpp: `tp-notes/TENSOR_PARALLEL.md`, `tp-notes/SM_TENSOR_PIPELINE.md`, `tp-notes/BENCH_RESULTS.md`, `tp-notes/car_audit_findings.md`, `tp-notes/research/custom-ar-tg-broadcast-mi50.md`, `tp-notes/research/custom-ar-bf16-on-wire.md`, `tp-notes/research/mi50-current-stack-profile.md`; code `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cuda/tp-allreduce.cu`, `src/llama-model.cpp`; commits `093f2a38f`, `6d82eb5f8`, `92607b5d1`, `5f65f9fa3`, `19d784ad0`, `751b6114cd`, `5d9efc8cad`.
- llmx: MULTI-DEVICE (Phase 0 results), `docs/STATUS.md` (Multi-device phase 0, layer split phase 3), the head split record `feat/split-head` (`8b60b7aeb`, `d48798487`), and the probe of section 2.6.
