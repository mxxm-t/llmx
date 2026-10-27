# DeepSeek-V4.1-Flash-1017 and DeepSeek-V4 support (planned, 2026-10-27)

This page is the plan for running DeepSeek's V4-class models in llmx: DeepSeek-V4.1-Flash-1017 first (GGUF architecture `deepseek41`), then DeepSeek-V4-Flash (`deepseek4`) as a variant of the same module.
It replaces the plan of 2026-09-26, which the owner decided on that day and which was kept outside the tree; its decisions still hold and are carried in section 1.3, and only what 1017 and llmx's growth since then change is new here.
The owner decided its questions on 2026-10-27 (section 9); building starts only when the coordinator assigns a developer to its first building step.
Written by DSDEV, read at main dd4163407.

## 1. Goals

### 1.1 What a user gets

- `llmx generate`, `chat`, `logits`, `perplexity` and `serve` run DeepSeek-V4.1-Flash-1017 from its GGUF files, text only, on the CPU and on Vulkan devices, on one device where it fits and as a layer split where it does not, with the Engram tables read from the file through a bounded hot-row store in host memory.
- DSpark drafting through the one speculative system, with output identical to drafting off.
- The same for DeepSeek-V4-Flash (the 0731 release and the April one), if the owner says yes to question 1.
- Any hardware: the fit decides the placement from the devices' and the host's memory; the 10-MI50 machine is the test bed, not the target.
- Out of scope here: vision (the files carry `general.tags = image-text-to-text` and separate projector files; it follows ROADMAP #10), V4-Pro, and parsing DSML tool calls (they are rendered and returned as text).

### 1.2 What it is held to

- **Correctness:** the owner's gate revision of 2026-09-26 (section 5): tiny models built by DeepSeek's own inference code with a CPU port of its kernels, and a layered full-model reference on the real file's weights.
- **Batch invariance and exact reuse,** as for every model (ADDING-AN-ARCHITECTURE.md, Rules): the server equals the CLI equals each request alone, and a reused row or state is one the CLI would have computed the same way.
- **Speed, the owner's targets:** way over the fork. At least 2x the fork's layer-split server and 1.5x its tensor split at 16 to 64 users with p99 no worse, and single-stream prefill and decode clearly above the fork's layer split, the number proposed after the first measurement (section 6).

### 1.3 Decisions of 2026-09-26 that still hold

The owner approved every recommendation of the earlier plan on 2026-09-26:
- the reference is DeepSeek's official `inference/` code with a CPU replacement of its kernels, in a third isolated environment, fetched by pinned revision and SHA-256;
- the model-defined cache formats (the FP8 window, the FP4 index keys, the FP4 entries) are emulated exactly as model math; the engine's FP8 activations, BF16 activations and BF16 index scores are not;
- bring-up runs on the published files, and a lossless V4.1 file is decided once V4.1 runs;
- a V4 file without its YaRN keys is refused by name, and V4 is gated on a copy with the keys;
- no single-card or other-runtime Vulkan gate; the Radeon VII runs the tiny suites and its figures are recorded only;
- the exact V4.1 prefill skip is a branch of its own, gated by bitwise equality, with no user flag;
- DSpark is a proposer of the unified speculative system, embedded or from a sidecar;
- the default context is what the fit admits, printed at load;
- a cache type `native` (`--cache-type-k native --cache-type-v native`), the default for these architectures, with `f32` giving identical outputs and `f16` refused; by F2DEV's review it means "the architecture's declared cache format", one meaning for every model, refused where a model declares none, as the owner's one-rule-per-flag of 2026-10-05 asks;
- work starts step by step as prerequisites land;
- thresholds are proposed after a first measurement, for the owner's OK;
- the gate revision: a layered full-model reference from GGUFs on the host, V4 through transformers' `deepseek_v4` one layer at a time on the exact UD-Q8_K_XL weights, V4.1 through the official code with the CPU kernel port on the published file's decoded weights (file-exact); the fork is a side check.

## 2. Measured facts (2026-10-27)

### 2.1 What DeepSeek-V4.1-Flash-1017 is

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at revision 1bae0e1088993b9b82de08f59de5be5a34e42170 (created 2026-10-17) is, in its own card's words, "a minor update to DeepSeek-V4.1-Flash", compared through the Hub's file hashes against [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) at dba1be0a40aa (the earlier plan's pin) and at 314113f8a7da (its head, 2026-10-16):

| | 1017 against the earlier pin |
|---|---|
| `config.json`, `inference/config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors.index.json` | identical: same architecture, shapes, dtypes and tensor names |
| `inference/kernel.py`, `inference/engram.py`, `convert.py` | identical |
| the 48 weight shards | every one has a new SHA-256: new weights |
| `inference/model.py` | two changes, both also on V4.1-Flash's head (commits 8b3cec70, the index-key fix, and 4dd75a8b, of 2026-10-12) |
| `requirements.txt` | TileLang 0.1.12 in place of 0.1.8 |
| `encoding/encoding.py` | content given as blocks, a user message holding only tool results no longer starts a turn, image checks before merging |
| chat template | none: "This release does not include a Jinja-format chat template"; V4.1-Flash's head gained an official `chat_template.jinja` of 15348 bytes |

The two `model.py` changes:
- **Index keys in decode.** A layer that owns index keys (2, 8, 14 and 20) now publishes its key cache on every step, not only when its compression group completes (`model.py:536-549` at 1bae0e10). Before, in a decode step where a ratio-2 source's group was incomplete, its readers scored against the previous source's keys; DeepSeek's discussion #12 gives a random-weight test where this moved logits by up to 1.99. It is a fix of the reference, not of the model: the earlier plan already defined each layer as reading its latest source's keys, so its design is unchanged and only the reference pin moves.
- **FP8 of the attention output.** `act_quant(o)`, FP8 with an E8M0 scale per 32, now runs before the grouped output projection, in the trunk and in the DSpark block (`model.py:787` and `model.py:1073` at 1bae0e10). It brings the reference in line with what deployments run (FlashMLA ends its fused kernel in an FP8 cast). It is the engine's activation precision, the same as the `act_quant` before every FP8 GEMM, so under decision 1.3 it is not emulated (question 7).

So 1017 needs nothing the earlier V4.1 design lacked: the architecture is V4.1's.

### 2.2 The 1017 files on the test machine

The other project converted 1017 on 2026-10-26 with its September converter and staged it for its Hub repository; its files are in `/zpool1/models/DeepSeek-V4.1-Flash-1017-GGUF/`.
I read their GGUF headers only (provenance at the end of this page).

- **Main file:** 12 shards, 1006 tensors, 375.79 GiB, `general.architecture = deepseek41`, `general.version = 1017`: the same tensor set, types and metadata keys as the earlier V4.1 conversion the 2026-09-26 plan surveyed.
  - MXFP4 routed experts (exact from the checkpoint's FP4), Q8_0 for every FP8 linear (lossy, relative L2 about 0.0055), BF16 router, compressors, indexer projection, `engram_q`/`engram_k`, embedding and head, F32 norms, sinks and hyper-connection tensors.
  - Engram: `blk.1.engram_embed.weight` [256, 384006168] and `blk.14.engram_embed.weight` [256, 384016682], MXFP4 at 136 bytes a row, 48.6 GiB each, alone in shards 9 and 11; the checkpoint holds them as FP8 with E8M0 scales, and the other project's own verification log gives relative L2 0.12 for the requantization.
  - Sizes: experts 268.95 GiB, a layer 6.891 to 7.062 GiB without Engram, embedding and head 1.23 GiB each, Engram 97.28 GiB.
  - The file carries the converter's chat template (6945 bytes), not DeepSeek's.
- **DSpark sidecar:** `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf`, `general.architecture = dflash`, 3 stages of 128 experts at top 3, `target_layers` [37, 38, 39], block 5, 7.42 GiB.
  The first build took its experts from the trunk again (the fork's F14 repeated, kept as `BAD-trunk-experts.gguf`); the file in the GGUF folder is the corrected one.
- **The earlier V4.1 and V4 GGUFs are no longer in `/zpool1/models`.** The V4-Flash-0731 MXFP4 (156.4 GB) and UD-Q8_K_XL (161.9 GB) files survive on the backup share `/mnt/external-data1/zpool1/models/`; no V4 or earlier V4.1 file is on the pools.
- **Capacity, 2026-10-27:** `zpool1` 610 GB free, the NVMe root pool 299 GB; host memory 62 GiB; production holds GPUs 0-3, so 6 MI50s (4-9) are free beside it.

### 2.3 The other `deepseek41` layout

Upstream's converter (llama.cpp #31086, merged 2026-10-25) writes a different layout under the same architecture name, which Unsloth's published 1017 files use ([unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) at 60ad3651; headers read by range requests, no tensor data):
- metadata renamed: `attention.kv_source_layers`, `attention.indexer.source_layers`, `attention.indexer.candidate_*`, and the Engram parameters as `ple.layers`, `ple.head_offsets`, `ple.head_vocab_sizes`, `ple.layer_multipliers`; DSpark's parameters in the main file (`dspark.*`);
- one Engram table for both layers, `per_layer_token_embd.weight` [256, 768022850], the offsets of layer 14's heads continuing after layer 1's;
- the table in GGML type 43, MXFP8 (E4M3 with an E8M0 scale per 32), 202.8 GB: the checkpoint's own bytes, lossless; swappable for Q6_K, Q5_K or IQ4_NL shards;
- `engram_gate.weight` F32 [5120, 4] in place of `engram_q`/`engram_k`, `engram_kv`, `indexer.attn_k`, `indexer.k_norm`, and `exp_probs_b_vl.bias` for image tokens;
- in UD-Q8_K_XL every FP8 linear is BF16 and the router F32: 473.46 GiB in all, 284.6 GiB without the table, 508.4 GB on disk, value-exact by construction as V4's UD file was (BF16 holds an FP8 value times a power-of-two scale while the scale stays in range), not yet checked against the checkpoint;
- llmx has no MXFP8 decoder.

### 2.4 DeepSeek-V4

- [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (April, 60d8d707) and [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (7872f01b) have the same trunk configuration and kernels; 0731 adds the DSpark drafter (`dspark_*` keys, three MTP layers) and a few type and call-signature edits in `model.py`. One module variant serves both.
- transformers has `deepseek_v4` (since 5.8.0) and no `deepseek_v41` at 5.21.

### 2.5 What llmx has now that the 2026-09-26 plan waited for

| Needed | llmx at dd4163407 |
|---|---|
| MXFP4, BF16 and I32 tensors | read (`src/quant/types.hpp`, `src/quant/mxfp4.hpp`) |
| several residual streams | `ModelPlan::residual` holds four streams for qwen4exp |
| lookup tables read from the file | `LayerPlan::lookup`, `WeightRows`, `--lookup-storage`, the hot-row store and the read ahead (CACHE-PLAN.md section 9, steps 1 and 2 in flight) |
| token ids in a layer | `LayerPlan::token_history`, today only on one layer that keeps a state |
| per-position side rows and pooled groups | `LayerPlan::kv_aux`, `LayerPlan::kv_pool`, `Backend::pool_kv` |
| fixed per-sequence state, checkpoints, marks | qwen35's state slots, checkpoint slots, mark and retract |
| a drafter | the drafter hooks and `normalize_drafter` |
| splits | layer split, tensor groups of 2 to 4, staged groups; the pass API with passes in flight |
| precision | `--dtype auto|f16|bf16|f32|int8` (PRECISION.md) |

Still missing, all named in section 4: the `joyai-llm` pretokenizer (llmx treats every byte from 0x80 as a letter, and the Unicode classes it needs are not in the tree), YaRN, the mHC Sinkhorn mix, the compressor, DeepSeek's indexer (FP4 keys, top-512, candidate blocks), attention with a sink over a window and selected entries, the grouped output projection, sqrt-softplus and hash routing, the SwiGLU clamp, the model's FP8 and FP4 encoders, a layer that keeps paged entries and a fixed state, token history on two layers that may sit on different stages, and an MXFP8 decoder for the upstream layout.
qwen4exp's `select_kv` and `sparse_attention` are not DeepSeek's math (mean-pooled F32 keys, attention over full-resolution K/V), and its `mix_streams` and `inject_streams` have no Sinkhorn step; they are reused only where the math is the same (section 4.2).

### 2.6 How the big runtimes run it

From a survey of the code at pinned commits (vLLM 5d35ce39, SGLang 2b36c718, TensorRT-LLM 35cfb8cd, FlashMLA 395cb3af, DeepGEMM 5081f851, DeepEP f6aa38fe, llama.cpp 74bca395) and of their documents, techniques only, no code taken:

| Topic | What they do | What llmx takes |
|---|---|---|
| support | vLLM since #56214 (V4.1, 2026-09-11), SGLang with Engram since #39666, TensorRT-LLM since #19727 (Blackwell only), upstream llama.cpp since #31086 (2026-10-25, with the V4.1 indexer and candidate blocks, which the fork lacks); vLLM calls 1017 a weights-only update | the architecture is settled; 1017 is a file |
| cache rows | FlashMLA and TensorRT-LLM: 288 B FP4 entries with E4M3 scales per 16, 68 B MXFP4 index keys, 890 B a token in all; 528 B FP8 window rows | the same row formats (section 4.3) |
| compressor tail under drafting | vLLM and SGLang keep a ring of the open group's rows sized for the drafts and overwrite rejected slots; TensorRT-LLM recomputes incomplete groups | a ring with k_max headroom, restored by length |
| top-k | the reference leaves ties undefined; vLLM's decode top-k admits approximate boundary bins; TensorRT-LLM's radix top-k is exact but not deterministic; upstream llama.cpp sorts the picks before the gather for determinism | one total order (score, then lower index) on every backend |
| mHC | fused kernels in every engine; upstream llama.cpp keeps the mix in F32 because the mixes are near-tie sensitive | F32, one op per stage of the mix |
| prefix reuse | all three serving engines adopt DeepSeek's "SWA bounded replay", which their own docs say is not bit-exact; SGLang refuses deterministic inference for V4.1 and vLLM's batch-invariant list stops at V3.1 | not taken: exact checkpoints only |
| Engram | vLLM: pinned host tables read by the GPU over UVA, prefetched on a side stream, huge pages, heads sharded over ranks; SGLang: tables sharded in HBM over TP, host gather during the previous forward; TensorRT-LLM: pinned host only; DeepEP: an experimental RDMA engram buffer; none has a disk tier or a hot-row cache, and none publishes an overhead figure | the file with a hot-row store (CACHE-PLAN.md section 9), rows read ahead since ids depend only on tokens |
| MoE | DeepGEMM's fused EP dispatch with FP8 x FP4 GEMMs, EPLB in vLLM | experts as today; expert parallelism stays MULTI-DEVICE's |

### 2.7 The fork today

The fork's last DeepSeek commit is c4d12e2c00 (2026-10-15), which turns on prompt checkpoints for V4 and V4.1; restored turns are not bit-exact (log-probabilities differ by up to 0.21).
It still has no V4.1 indexer, top-512 or candidate blocks: V4.1 attends densely to every compressed entry.
No run of 1017 is recorded.
Its latest V4.1 figures (the earlier weights, MXFP4, 10 MI50s, Engram shards on NVMe, 2026-10-15): layer split decode 24.5 tok/s, prefill 323 tok/s at 3.3k and 328 at 15k (293 with checkpoints); tensor split `-tps 2` decode 19.8, prefill about 226 warm; with DSpark at `-tps 2` decode 23.0 at acceptance 0.49.
No server or concurrency figure exists for either model.
Pitfalls since the earlier plan's F1 to F18: F19 a BF16 matrix past 512 MiB written with the wrong row stride in multi-row products (single-token decode correct, perplexity 2.5M), F20 too many views per tensor-split container, F21 the DSpark trunk-experts bug again, F22 the Engram tables requantized to relative L2 0.12, F23 two incompatible `deepseek41` layouts under one name, F24 top-k picks in no fixed order, F25 inexact checkpoint restore and 1.2 s a prompt to save one.

## 3. Does V4.1-1017 support cover DeepSeek-V4?

No: a V4 file is refused until V4 is built, but V4 is a variant of the same module, not a second one.

**What differs** (DeepSeek's two `model.py` files, as the earlier plan read them):

| | V4-Flash | V4.1-Flash(-1017) |
|---|---|---|
| shape | d 4096, 43 layers, 256 experts top 6 | d 5120, 40 layers, 384 experts top 6 |
| compressed layers | ratio 4 (overlapping groups with a learned position term `ape`) on 21 layers and ratio 128 on 20, each layer its own | ratio 2 on layers 2, 8, 14 and ratio 1 on layer 20; the other layers read those (CSA2 sharing) |
| indexer | 64 heads, a 128-point Hadamard before FP4, on every ratio-4 layer | 32 heads, no Hadamard, on 8 layers; candidate blocks on layer 20 reused by 24 to 36 |
| window format | FP8 over dims 0-447 in blocks of 64, the 64 rope dims unquantized | FP8 over all 512 dims in blocks of 32 |
| stored entries | FP8, 711 B | FP4 with E4M3 scales, 288 B |
| query | a per-head RMS norm without weight | none |
| routing | layers 0-2 by `tid2eid[token]` (hash), the rest by score | by score everywhere |
| mHC | two passes per sublayer, `hc_head` before the output | one pass, carrying `pre` to the next sublayer; no `hc_head` |
| eps | 1e-6 | 1e-20, and +1e-20 in the routing normalization |
| Engram | none | layers 1 and 14 |
| DSpark | taps at the outputs of layers 40-42, 256 experts top 6 | taps at the inputs of layers 37-39, 128 experts top 3 |
| empty attention row | NaN in its kernel | zeros |

**Shared:** the tokenizer and pretokenizer, YaRN, the 512-wide single-vector attention with a sink, the grouped output projection, the mHC coefficient and mix ops (with an order flag), sqrt-softplus routing with bias, the clamp, the FP4-E8M0 index keys, top-k, the compressor's pooling, the sidecar drafter, and every runtime addition of section 4.3.

**One module, two registry entries,** as qwen3/qwen3moe and qwen35/qwen35moe are: `deepseek41` and `deepseek4` reach `src/model/arch/deepseek4.hpp` with a variant the registry entry names, read once into the configuration; nothing is inferred from which tensors a file holds.

**What V4 costs on top of V4.1** (estimates): the ratio-4 overlapping compressor with `ape` and its carried half-groups, the ratio-128 compressor, the Hadamard, hash routing with its I32 table, the two-pass mHC and `hc_head`, the query norm and the FP8 entry format: about a quarter to a third of V4.1's model work, CPU and Vulkan, with no runtime additions of its own.
Its gates are cheaper than V4.1's: V4 fits six MI50s beside production (V4.1 needs all ten), it has an independent HF implementation for the layered reference, and its lossless UD-Q8_K_XL file is on the backup share.

## 4. Design

### 4.1 Where the pieces live

| Concern | Owner |
|---|---|
| keys, refusals, tensor roles, layer kinds, the variant, both `deepseek41` layouts, the math | the module `src/model/arch/deepseek4.hpp` (a directory once past about 1,000 lines) |
| which file runs it | `src/model/arch/registry.hpp`: `deepseek41` and `deepseek4`, declared dtype BF16, `dflash` drafters paired through `normalize_drafter` |
| the YaRN tables | the module's `fill_tables` (CACHE-PLAN.md section 10 (b)); the rope owner of the shared graph pieces once a second architecture declares YaRN |
| the model's FP8 and FP4 encoders and scale rules | one C++ owner under `src/quant/` and its GLSL twin (XDEV's area) |
| the MXFP8 decoder (upstream layout) | `src/quant/`, beside MXFP4 (XDEV's area) |
| the Engram hash | the module's `lookup_ids` hook, from token ids and the token history |
| the Engram rows | `WeightRows` and its hot-row store, under `--host-cache-bytes` (section 4.4) |
| storages, state slots, checkpoints, forks, rollback | the runtime and the backends, extended once (section 4.3) |
| the kernels | the backends, every new op named for its math in `src/backends/backend.hpp` |

**The two layouts of `deepseek41`.** The reader recognises a layout by its metadata key set (`engram.layer_ids` against `ple.layers`) and refuses a file mixing them; each layout has one name table, and both resolve to the same plan.
The fork layout's separate `engram_q` and `engram_k` and the upstream layout's product `engram_gate` feed the same gate op (section 4.2).

### 4.2 The layer math as backend ops

Every new op has a CPU form with a unit test against a reference written from the math, then a Vulkan form held to the CPU (ADDING-AN-ARCHITECTURE.md, New backend ops); kernels follow the row's extent, every multiply-add is an explicit `fma` and order-sensitive accumulators are `precise`.

| Piece | Op | New or reused |
|---|---|---|
| mHC coefficients: 24 mixes as an F32 matmul, then the rsqrt, sigmoids and 20 Sinkhorn rounds per row in the reference's order | matmul, then `hc_coeffs` | new; one invocation per row, so batch-invariant |
| collapse u = sum of pre_k X_k, then RMSNorm | `hc_collapse_norm` | new |
| expand X'_k = post_k F(u) + sum over j of comb[j][k] X_j, in j order | `hc_expand` | new; V4.1 carries `pre` in the arena |
| query, window key, the shared vector's RMSNorm and rope on its last 64 dims, inverse rope on the output | `norm_rope_partial` with adjacent pairs and an offset, an inverse mode, and two tables | extended under a mode, existing SPIR-V unchanged by hash |
| compressor: softmax pooling per channel over a group's slots, from the sequence's stored tail and this pass's rows, by absolute position | `compress_pool` | new; a group's entry never depends on where a pass cut it (F6) |
| index scores sum over heads of w_h ReLU(q_h . k_j), keys decoded from FP4 on the fly | `indexer_scores` | new |
| top-k by score then lower index, a running best-k set merged over key tiles (O(k) workspace a row at any context), returned by position | `topk_rows` | new (F8, F24) |
| candidate blocks on layer 20, carried to 24-36 | `candidate_blocks` | new |
| attention over the window then the selected entries, one 512-wide vector as K and V, a learned sink added after the max, empty rows zeros | `sparse_attention_sink` | new; qwen4exp's `sparse_attention` reads full-resolution K/V and has no sink |
| grouped output projection, 8 groups of 4096 to 1024 | `matmul_experts` with a constant routing made at load | reused, no reshape of placed weights (F17) |
| sqrt-softplus scores, selection by score plus bias, weights from the unbiased scores normalized and scaled by 1.5; V4's hash mode | `route_experts` scoring modes | extended under a mode |
| SwiGLU clamp silu(min(g, L)) times clamp(u, plus or minus L) on every path | `silu_mul` clamp variant | extended; a fused path without the clamp is refused (F4) |
| Engram gate: per stream, RMS-normalized X and K against `engram_q`, `engram_k` (or their product), dot over sqrt(d), signed square root clamped at 1e-6 with -0 on the negative branch, then X_c += g_c v | `group_rms_norm`, `signed_sqrt_gate`, an add per stream | reused where the math is the same (qwen4exp's PLE gate has this form); the -0 branch and eps 1e-20 tested first |
| V4: Hadamard, query norm, ratio-4 and ratio-128 pooling | the same ops in modes | with V4 only |

### 4.3 Caches and state

- **Paged entries.** Compressed entries and index keys are paged storage in the model's own formats, beside K/V: FP4 entries with E4M3 scales (288 B) and MXFP4 index keys (68 B) on V4.1, decoded by the existing MXFP4 decoder; FP8 entries (711 B) on V4.
  Every such storage uses blocks spanning 128 tokens, so a multiple of 128 is a whole block in every storage and forks share whole blocks as today.
  V4.1 keeps them on layers 2, 8, 14 and 20 only; the other layers hold views of their source.
- **Per-sequence state.** The 128-token window ring with k_max headroom for drafts and the compressor's open group are a fixed, private state per sequence, held in state slots allocated at load, in the model's format (`native`) or F32.
  So a DeepSeek layer keeps paged entries and a fixed state at once, which no layer does today: one runtime addition, a cache kind holding both, landed in a commit of its own with DeepSeek as its first user and every existing architecture byte-identical.
- **Token history.** Layers 1 and 14 both read the three preceding token ids, possibly on different stages. Today `Model::prepare_tokens` snapshots the tail for one stage (`token_stage_`) and that stage's commit advances it (`passes.hpp:596-602` and `passes.hpp:698-701` at dd4163407).
  The change, the narrow extension of today's mechanism (XDEV's second review): the pass snapshots each entry's preceding ids when its first lookup stage is prepared, that stage's commit advances the tail once, and every later lookup stage reads the pass's retained snapshot, never the live tail.
  A continuing chunk may begin once the previous chunk's first stage has committed (`passes.hpp:42` at dd4163407), before that chunk's later lookup stage has run; stages run passes in formation order, so the next chunk's first lookup stage is prepared after the previous chunk's has committed, and the same-stage order supplies the right tail without serializing the pipeline.
  One tail a sequence, never one per lookup, with the host identity and serialization unchanged in owner.
  Tests: both lookups on separate stages, with chunk B begun after chunk A's first stage commits and before A's last lookup stage runs, each lookup reading its correct history; overlapping passes, a checkpoint restore and a partial draft acceptance.
- **Crossing rows.** On a split the source layers' new entries, index keys, index lists, candidate sets and the carried `pre` cross with the residual for this pass's rows only (F10); readers on later stages keep replicas.
- **Reuse.** A checkpoint is the full state at a multiple of 128, where V4.1's ratio-2 and ratio-1 tails are empty, about 2.6 MiB on V4.1 in the model's format; batch assembly ends a slice there, legal because the compressor is cut-invariant.
  On V4 the ratio-4 compressor's overlapping groups carry the previous group's first halves across every boundary, so V4's checkpoint holds them as state (about 20 KiB a ratio-4 layer, 420 KiB in all, a checkpoint about 4.1 MiB), and ratio 128's tail is empty only at a multiple of 128.
  Prefix reuse, pause and resume go through qwen35's checkpoint slots and `make_room`.
- **Rollback by length** (XDEV's review). qwen35's retract reruns the recurrence from a mark's saved inputs (`Model::rerun`, `Architecture::recur`); this cache kind restores a prefix without either, in the history and cache owners:
  - every piece is indexed by absolute position: the window ring's slot for position q is q mod (128 + k_max), the compressor's open-group rows a ring of r + k_max slots the same way (2r + k_max on V4's overlapping ratio 4, which also keeps the previous group), and the entries' and index keys' lengths floor(L / r);
  - after a pass of k rows past length L (k at most k_max), no slot of a position in [L - 128, L) has been written, since the slot of q in [L, L + k) last held q - 128 - k_max, which is below L - 128; so retracting to any accepted length L + a (0 to k) sets the lengths and nothing else, and entries that rejected rows completed are dropped with them;
  - replicas on later stages hold the same lengths, set by the same retract;
  - a pass that fails after a source stage commits is restored the same way where its rows fit the headroom, and otherwise back to the checkpoint, or to 0, as for qwen35.
  Tests: accepting every prefix from 0 through all drafts, then one more decode, across group and 128-token boundaries, after a fork, and with a failure injected after a source stage commits; each equal to decoding the accepted tokens one at a time.
  DeepSeek's bounded replay is not used, since it is not exact (F25 shows the cost of inexact restore).
- **Sizes.** The source layers' entries and index keys are 890 B a token on V4.1 in the model's format (720 B of entries and 170 B of keys over layers 2, 8, 14 and 20), against 6,400 B for the same values in F32; the window rings are a fixed 2.6 MiB a sequence.

### 4.4 Engram on the cache plan's owners

- **One owner per table.** The fork layout's two tables are two `WeightRows`; the upstream layout's one table, read by layers 1 and 14 at offset ranges, is one `WeightRows` serving two lookups.
  No second mechanism: the module declares each layer's `lookup` (24 rows a token, the table's row width and type) and its `lookup_ids` hook computes the ids; the runtime reads, stores, uploads and gathers.
- **The hash.** Each id goes through the 129,280-entry `token_map` to one of 99,092 compressed ids, the products with the per-layer odd multipliers are bounded so int64 never overflows, they are XOR-folded over the token and the three before it, reduced modulo each head's prime and offset; pad id 2 fills positions before the start.
  It depends on token ids alone, so rows are read ahead for a prompt and for drafts, and layer 14's rows are read while layers 1 to 13 run.
- **The store.** Each table's hot rows live in the store of CACHE-PLAN.md section 9 under the one `--host-cache-bytes` budget, rows first; its cap of 1.5M tokens' rows is 10.3 GiB for the fork layout's 136-byte rows and 18.9 GiB for MXFP8's 264-byte rows.
  The survey there puts DeepSeek's hit rate at 0.60 with 2 GiB and 0.66 with 16 GiB; what misses are n-grams never seen, which only the file serves.
- **Where the table is.** `--lookup-storage file` on the test machine: 97.3 GiB (or 188.8) exceeds the host, and every MI50 is full of weights. On a host whose budget holds a table, the store at full size is the table in memory (CACHE-PLAN.md section 9, step 3); a split by rows over devices (its step 4) only on a measured gain.
- **The fork's lessons:** no prefault and no page-fault gathers (F11), the history from tokens across checkpoints, forks and pauses (F12), `engram_q` times `engram_k` never a host node in the pass (F11).

### 4.5 Numerics

- **Model formats, exact:** the window's FP8 (E4M3 with an E8M0 scale per 32 on V4.1; per 64 over dims 0-447 on V4), the index keys' and queries' FP4 with an E8M0 scale per 32, V4.1's stored entries in FP4 with an E4M3 scale per 16.
  Scales follow the reference's rule, 2 to the ceiling of log2(amax / max) from one correctly rounded F32 multiply and integer operations, not the OCP rule of llmx's MXFP4 writer; codes are chosen by exact comparisons, so the CPU and Vulkan store identical bytes for identical inputs.
  Decoding is exact to F32, subnormal scales included, under the quant plan's denormal policy.
  Each GLSL module of these encoders and decoders declares the float controls it relies on (denormal preserve, round to nearest even) through the kernel table's `controls`, and a device without them refuses that kernel by name; nothing relies on an undeclared control.
- **Activations follow `--dtype`** (PRECISION.md): the registry declares BF16, so `auto` is F16 on the MI50 and an AVX2 CPU, and `int8` is the opt-in matched to the fork's 8-bit activations.
  F32 stays for the router, the compressor, mHC, the index scores and the head's mix, as in the reference.
  The engine's FP8 activation quantization, the new FP8 of `o` among it, and its BF16 index scores are not emulated (question 7).
- **One site per function** (exp, rsqrt, the pool's division), identical in C++ and GLSL where byte identity is claimed, else an ULP bound in the op's gate.

### 4.6 Tokenizer and chat

- The `joyai-llm` pretokenizer: digits in runs of 1 to 3, runs of CJK ideographs and kana, then the split over letters, marks, punctuation and symbols, which needs real Unicode general categories: a generated table in `src/core/`, dependency-free, used by this pretokenizer only so qwen2's and qwen35's ids do not move.
  The table is of the Unicode version the reference tokenizer's regex engine uses; its generator pins the UCD files by version and SHA-256, and the id test carries characters of the newest Unicode versions, where versions disagree.
- Chat renders the file's own template, as for every model; the fork-layout file's is the converter's and the upstream files carry none.
  Its renders are compared with DeepSeek's `encoding.py` on its five official cases and the differences recorded, not gated, since the template is not DeepSeek's (decision 1.3).
  A file without a template is refused by `chat` and `serve` as today.

### 4.7 DSpark

- A sidecar `dflash` file (or, in the upstream layout, DSpark keys beside the trunk, no file of it read yet) is paired by `normalize_drafter` after tokenizer equality; its taps are refused outside the target (the earlier `dsv41-broken-hold` file).
- It drafts through the proposer-independent verify (SPECULATIVE.md): one entry of k+1 rows, exact acceptance, the ring and tails restored by length, the Engram history following the kept tokens; output with it on equals output with it off.
- Its experts are checked against the trunk's at load for byte equality, and a sidecar whose experts equal the trunk's is refused (F14, F21).

### 4.8 Placement and memory

- **Test machine, V4.1:** weights without Engram are 278.5 GiB in the fork layout and 284.6 GiB in the lossless upstream layout, so V4.1 needs all ten MI50s at about four layers a card: the first card held 28.97 GiB of weights in the earlier plan's count, short of room for a 1M-token budget, which is why the default context is what the fit admits.
  With production on GPUs 0-3, V4.1's device gates run in whole-machine windows (question 5); development runs on the six free cards with truncated real files (the first N layers) and the tiny models.
- **Test machine, V4:** 145.6 GiB (MXFP4) or 150.8 GiB (UD) fits GPUs 4-9 beside production.
- **Generic hardware:** V4.1 needs about 280 GiB of device memory for weights plus the KV budget (eight 48 GB cards, four 80 GB cards, two 192 GB cards), and its Engram table in host memory where the host holds it, else read from the file through the store; V4 about half.
  Host experts beside one large device follow `--n-cpu-moe` where the host holds the experts (269 GiB on V4.1).
- **Tensor groups:** 64 query heads over one shared vector, 32 index heads and 384 experts shard by heads and experts, and the 8 output groups allow widths 2 and 4; built after the layer split, on the owner's 1.5x tensor-split target (step 13).
- **Not here:** expert parallelism (MULTI-DEVICE.md).

## 5. Correctness gates

1. **Tiny models** (the gate revision of 2026-09-26): `tools/` builds tiny `deepseek41`, `deepseek4` and `dflash` models through DeepSeek's `ModelArgs` and `Transformer` at the pinned code (1017's 1bae0e10 for V4.1, 0731's 7872f01b for V4), its kernels replaced by F32 CPU code, its top-k by llmx's total order, the model formats' QDQ on and the activation QDQ off, the real tokenizer, with every layer kind, selection and candidate blocks engaged within 256 tokens, and writes them in the fork's layout, and in upstream's too if question 2 says yes.
   They hold 2e-5 per logit and 1e-5 NLL across ubatches, threads, decode after prefill, scoring, `--then-ids`, cuts at every offset around group edges, and splits; decisions within a measured margin of a tie run in forced mode (the port takes llmx's selections). Hosted CI runs them.
   For V4 the port is cross-checked against transformers' `deepseek_v4` on the same weights, and a fork at a multiple of 128 must continue bit-identically, the carried ratio-4 halves included.
2. **Ops:** every new op against its math reference; QDQ exhaustively against decoders written from the format description; Sinkhorn against float64; top-k at widths 1 to 2^18 with ties, NaN and -inf fillers and any tiling.
3. **The real file, layered and file-exact:** the official code with the CPU kernel port, one layer at a time, on the GGUF's weights as `tests/spec_decode.py` decodes them, over short prompts and the perplexity excerpt; for V4 transformers' `deepseek_v4` on the UD weights.
4. **The 16k rule** (AGENTS.md, Traps the gates exist for): two fresh 16,384-token inputs, raw and chat-rendered, against the layered reference, under the frozen empirical depth contract of 2026-10-19 (QWEN38-FLASH.md section 6), its calibration pair being the reference in F32 against the reference at the engine's precision (question 6).
5. **Devices and splits:** the tiny suites on one MI50 and the Radeon VII; slice invariance bitwise on the device; two layer shares giving identical bytes; `llmx-split-check` over CPU stages bit-identical to one CPU; the device against the CPU path on the real file by the device-against-CPU criterion.
6. **Serving:** server equals CLI equals each request alone with prefix splits at 1, 3, 127, 128, 129; a checkpoint fork continues bit-identically; `tools/server_mix_check.py`; two fresh servers give the same bytes; DSpark on equals off.
7. **Everything else unchanged:** every step keeps existing models byte-identical to main, checked by what the step can break under AGENTS.md's merge gates at landing (a docs or tokenizer step needs no device qualification), with unchanged evidence carried explicitly and a focused fix rerunning only what it touches.

## 6. Speed gates

- **Arms:** llmx against the fork at its newest DeepSeek commit with the same 1017 file, on the same cards, at default clocks, cards pinned, rounds interleaved, contention recorded; llmx `--dtype int8` is the arm matched to the fork's 8-bit activations, and `auto` is reported beside it. Upstream llama.cpp (which has the indexer) is reported where it runs, gating nothing.
- **Cells:** the architecture gate's pp64, pp247, pp512, pp4096, tg32, tg128 and pp16384 then tg512 (ADDING-AN-ARCHITECTURE.md, Gates), a 16k agent follow-up turn, cold and warm Engram with bytes read; the server through `tools/server_load.py` at 1 to 64 users, 4096/256 and 16384/128, with TTFT, ITL and end-to-end p50 and p99.
- **Pass:** at least 2x the fork's layer-split server and 1.5x its tensor split at 16 to 64 users with ITL p99 no worse; single-stream prefill and decode above the fork's layer split by the margin proposed after step 9's first measurement.
- The comparison with the fork is the phase's final gate (AGENTS.md, Merge gates), after steps 9 to 13, not each step's; each step records its own cells and holds existing models level with main.
- First support may land below a cell only as AGENTS.md allows, with the cells listed and their recovery named.

## 7. Steps

Each step is a branch off main with its own gates, landing as AGENTS.md says; every step keeps existing models byte-identical.

| | Step | Gate |
|---|---|---|
| 0 | Measure and vet, no llmx code: the fork on the 1017 file at the cells of section 6 in a whole-machine window; TileLang 0.1.12's casts and fast-math read; the reference environment (a torch with `float8_e8m0fnu` and `float4_e2m1fn_x2`) | numbers recorded here |
| 1 | This page | F2DEV's and XDEV's reviews, the owner's decisions |
| 2 | The reference tool: tiny generator, GGUF writer in both layouts, goldens, the layered file-exact reference | the gates of section 5.1 on the port itself |
| 3 | The `joyai-llm` pretokenizer and the Unicode table | ids equal to `tokenizers` on the multilingual set, the specials and a long prompt; qwen2 and qwen35 ids unchanged |
| 4 | The module opened: configuration, refusals, plan, footprint, `info`; the plan refuses the architecture by name | `arch-deepseek4` with every refusal's text |
| 5 | The runtime additions: the cache kind with entries and state, token history on two stages, crossing rows, `native` | existing architectures byte-identical; their own tests |
| 6 | The CPU ops | each op's unit test |
| 7 | V4.1 on the CPU | section 5.1 on the tiny models; split-check over CPU stages; the real file recorded |
| 8 | The Vulkan ops | `backend-vulkan` against the CPU; existing SPIR-V unchanged or level on both cards |
| 9 | V4.1 on ten MI50s, Engram from the file | section 5.3 to 5.5; the speed cells measured and recorded |
| 10 | Serving: checkpoints, pause, reuse | section 5.6; the server cells measured and recorded |
| 11 | The exact prefill skip | logits, states and forks bit-identical on and off; speed recorded |
| 12 | DSpark | section 5.6's drafting cases; acceptance recorded |
| 13 | Tensor groups for these models | section 5.5 on groups; cells recorded |
| 14 | V4 (question 1) | section 5 for V4, on six MI50s |
| 15 | The upstream layout and MXFP8 (question 2) | the tiny models in that layout; the layered reference on its file |

## 8. Not chosen

- **Two modules for V4 and V4.1:** they share every runtime addition and most ops; a variant keyed by the registry entry is how llmx runs qwen3/qwen3moe.
- **The window as paged history** released past 128 tokens: forks at older positions would still need state, and the ring reuses qwen35's checkpoint machinery.
- **Bounded replay, dense attention over V4.1's entries, an Engram-off switch, a running-softmax compressor, an f16 cache, emulated FP8 activations:** none is the model, or none is exact.
- **Engram tables resident on the devices on the test machine:** every MI50 is full of weights; a split by rows only on a measured gain elsewhere.
- **A converter in llmx:** llmx reads files.

## 9. The owner's decisions (2026-10-27)

Decided by the owner on 2026-10-27, after F2DEV's and XDEV's clearance, every question as recommended; decision 6 is an owner-approved gate change.

| | Question | Recommendation | Decided |
|---|---|---|---|
| 1 | Support DeepSeek-V4 as well, as a variant of the same module after V4.1's CPU step (about a quarter to a third more model work, no runtime work of its own)? | yes: it fits six MI50s beside production and has an independent HF reference | yes, V4 too, after V4.1's CPU step |
| 2 | Read both `deepseek41` layouts: the fork's (the file on the host) first, upstream's (Unsloth's published files, whose UD-Q8_K_XL is lossless by construction) after an MXFP8 decoder from XDEV? | yes | yes, both layouts, the local one first, upstream's MXFP8 after XDEV adds the decoder |
| 3 | May llmx read the other project's 1017 GGUF and its NVMe copy of the table shards, read-only, once that project says its conversion and publishing are done, rather than download a copy of 404 GB? | yes | yes, read-only once that project says it is finished, no download |
| 4 | Keep V4.1's real-model gate file-exact on that file, with no download of the lossless 508 GB file now (610 GB free less the 300 GiB reserve)? | yes; revisit when the pool has room | yes, checked against the local file's own weights, the 508 GB download skipped |
| 5 | Book whole-machine windows, production stopped and announced, for V4.1's ten-card gates (steps 0, 9, 10, 12)? | yes | yes, announced windows with production stopped |
| 6 | For the 16k rule, the depth contract of 2026-10-19 with the reference in F32 against the reference at the engine's precision as its calibration pair (a gate revision)? | yes | yes, the 2026-10-19 depth rules calibrated as proposed (a gate change) |
| 7 | Leave 1017's new FP8 of the attention output unemulated, as every activation FP8 is? | yes | yes, not emulated |
| 8 | The fork at its newest commit on the same 1017 file as the speed reference, upstream llama.cpp reported beside it? | yes | yes |
| 9 | Copy V4-0731's UD-Q8_K_XL (162 GB) back from the backup share when step 14 starts? | yes | yes, when the V4 step starts |

## 10. Review dispositions

Each finding of F2DEV's and XDEV's reviews, rechecked against the code and the files, fixed or rejected with the evidence beside it; both cleared the plan for the owner at 07:33 and 07:36 on 2026-10-27.

| Reviewer | Finding | Disposition |
|---|---|---|
| F2DEV 07:11 | (1) a checkpoint at 128 finds the tails empty only on V4.1; V4's ratio-4 overlap carries half a group | taken: 4.3 states V4's carried halves as checkpoint state (the 2026-09-26 plan's count, 420 KiB), section 5.1 adds the V4 fork at 128 |
| F2DEV 07:11 | (2) the Unicode table must be the reference regex engine's version, pinned | taken: 4.6 |
| F2DEV 07:11 | (3) the FP8/FP4 GLSL modules declare their float controls through the kernel table | taken: 4.5; checked that the kernel table's `controls` and `preserved_controls` exist (`vulkan_kernels.hpp:1501-1530` at dd4163407) |
| F2DEV 07:11 | (4) `native` as the architecture's declared cache format, one meaning, refused where none is declared | taken: 1.3 |
| F2DEV 07:11 | the fork comparison is the phase's final gate, not each step's | taken: section 6 and the step table |
| XDEV 07:27 | (1) retract by length is unspecified: today's retract reruns through `Architecture::recur`; every piece must describe the same prefix | taken: 4.3's rollback by length, positions-indexed rings with the headroom argument, replicas, failure after a source stage commits, and its tests; checked `Model::rerun` (`history.hpp:856-898` at dd4163407) |
| XDEV 07:27 | (2) token history is snapshotted for one stage and advanced at its commit; two lookups on two stages need one immutable snapshot and one advancing owner | taken: 4.3's token history; checked `prepare_tokens` and the commit's advance in `passes.hpp` at dd4163407 |
| XDEV 07:27 | (3) "at every step" means the existing-model invariant, checked by AGENTS.md's changed-path rules, evidence carried | taken: section 5.7 |
| XDEV 07:33 | the begin_pass snapshot can read a stale tail, since the next chunk may begin after only the previous chunk's first stage commits | taken: snapshot at the first lookup stage, advance at its commit, later lookup stages read the retained snapshot, with the ordering test; checked `passes.hpp:42` and `passes.hpp:60` at dd4163407 |
| F2DEV 07:33 | cleared for the owner; XDEV's changes reopen none of the four points | recorded |
| F2DEV 07:36 and 07:41 | a correction: XDEV's ordering point was right and his own 07:33 too quick; cleared again at 5d2a028bd with it folded in | recorded |
| XDEV 07:36 | cleared for the owner's decisions (design only, not the numerical-policy questions or implementation gates) | recorded |

## Provenance

- **GGUF headers on the test machine,** read on 2026-10-27 between 06:33 and 06:50 UTC by a header-only script piped over ssh (metadata and tensor infos, no tensor data; nothing written there), after `fuser` showed no process holding the files: the 12 shards `DeepSeek-V4.1-Flash-MXFP4-000{01..12}-of-00012.gguf` and `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf` in `/zpool1/models/DeepSeek-V4.1-Flash-1017-GGUF/`, and `DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf` and `BAD-trunk-experts.gguf` in `/zpool1/models/dsv41-1017-sidecar-fixed/`; the chat template was taken from shard 1's header.
  The coordinator's reminder of 06:14 forbids even header reads of these files; the coordinator's decision of 06:31 then allowed DSDEV header reads only, of files no process writes, with every file listed, and this list is that disclosure.
  Listings only: those folders, `/zpool1/models/DeepSeek-V4.1-Flash-1017/` (the checkpoint), `/opt/models/dsv41-engram-nvme` and the backup share's V4 files.
  A research helper also ran one `grep` on the checkpoint's `config.json` with its output discarded, outside the brief; disclosed here, nothing came of it.
- **DeepSeek's code and configs:** fetched from the Hub at the revisions in section 2.1 and 2.4 (no weights); the upstream layout from range reads of the Unsloth headers at 60ad3651.
- **The fork:** its git history and the other project's logs under `/opt/claude-work`, read-only.
