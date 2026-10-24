# Qwen3.8-Flash-Next text support

This page describes llmx's text inference for the open-weight Qwen3.8-Flash-Next, whose GGUF architecture is `qwen4exp`.
It is not the already supported Qwen3.8-27B, which uses `qwen35`, or a promise to reproduce the hosted API.
The architecture module is registered, and ordinary GGUF loading and the CLI and server commands run it.
The released Q8_0 file is qualified under the released-file contract (section 6): its corpus cells and its 16K depth cells pass against canonical HF on the file's own weights.
Its speed gate is open on decode: llmx leads the reference runtime's ROCm build on prompt throughput in every comparable split and trails it on decode (section 6, Speed gate); section 7 holds the open items.
[STATUS](STATUS.md) carries the feature block.

What the implementation has passed, each within the scope its record states:

- The complete tiny graph, with the GDN output gate as the released contract's sigmoid, against independent HF logits on Windows and Linux CPU and MI50 Vulkan, within the tiny-fixture bounds of 2e-5 per logit and 1e-5 NLL ([record](benchmarks/qwen38-sigmoid-gate-20261015.json)).
- The public loader in each load mode and each lookup storage, with its retained memory and file owners, against the same HF logits (`llmx-qwen4exp-model-test`).
- Tiny tensor-group fixtures over two CPU backends and two MI50s against the same HF rows, with memory- and file-backed lookup passes, marked rollback continuations and follow-up decode ([record](benchmarks/qwen38-vulkan-tensor-group-20261016.json)).
- A disk restore streamed to the devices that carries the complete Flash history image: auxiliary KV, every layer's state, the MTP carry and the host token history ([record](benchmarks/qwen38-streamed-restore-20261019.json)).
- MTP through the existing drafter hooks: the tiny fixture matrix on one Vulkan device, a width-two group, two width-two stages and width four ([record](benchmarks/qwen38-mtp-main6e-vulkan-20261019.json)), and released draft-state and marked-rollback controls (MTP development checkpoint, below).
- The released-file contract of section 6 at the qualified candidate: all eight corpus cells within the file-exact bounds, and both fresh 16K depth inputs within the frozen reference-calibrated limits.
- Natural-stopping MTP acceptance at or above the reference runtime's on each of three prompts (section 6).

The tiny HF server fixture, the edit, cut and follow-up checks against fresh servers, and the POSIX disk restart and crash recovery checks are held by `tests/qwen4exp_fixture.py`, which the suite's `reference-consumer` component runs; they are tiny-fixture checks, not released-model serving qualification.
Broader split and serving qualification, other precision modes and the decode side of the speed gate remain open.

Follow [ADDING-AN-ARCHITECTURE](ADDING-AN-ARCHITECTURE.md), [ARCHITECTURE](ARCHITECTURE.md), [EXECUTION](EXECUTION.md), [TENSOR-SPLIT](TENSOR-SPLIT.md) and the gates in [AGENTS](../AGENTS.md).

## 1. Scope and evidence

The production target is Vulkan with enough devices and host memory for the selected file.
The CPU backend runs the complete graph, which the tiny fixtures check against HF, and serves the operator tests; F16/BF16 weight decoding is implemented, but released-model CPU-only execution has not been qualified against the required corpus.

| Capability | Support | Evidence |
|---|---|---|
| Text prefill, decode, chat and scoring | Yes | Tiny HF fixture; the released Q8_0 file under the contract of section 6 |
| CPU-only released model | Unqualified: BF16 decoding exists; released-file CPU corpus checks remain | Tiny full-graph CPU fixtures |
| CPU experts beside Vulkan | `--n-cpu-moe` or `--cpu-moe` beside one device, through the existing placement | Tiny HF fixture with a witness of where the routed rows ran (`tests/qwen4exp_model.cpp`) |
| Host experts streamed to Vulkan | `--moe-stream-from`, copying whole expert stacks into their windows | Tiny HF fixture with a witness of the window copies (`tests/qwen4exp_model.cpp`) |
| Lookup table on the devices or read from its file | `--lookup-storage` `auto`, `device` or `file` (section 3) | Tiny HF fixture in every load mode and lookup storage; the released file qualified with its rows read from the file (section 6) |
| Layer split | Yes | Tiny fixtures against one device; the released file on three stages of width-two groups (section 6) |
| Tensor groups | `--tensor-width` 2 to 4; width 3 is refused on this model's vocabulary | Tiny CPU and MI50 tensor-group fixtures against HF; the released file on three stages of width-two groups (section 6) |
| Follow-ups, multiple users, reset, cut and failure recovery | Yes, through the shared runtime and server | Tiny fixtures against fresh controls (`tests/qwen4exp_fixture.py`) |
| MTP | Embedded block or sidecar file (section 4) | Tiny graph against a scalar oracle; on the released file, replies with MTP on equal those with it off, and natural-text acceptance is in section 6 (MTP development checkpoint, below) |
| Vision/video, native HF-directory loading | No | Separate follow-on work, never inferred from text support or the fixture generator |
| Further quantization types | Only the types the existing decoders read; another is refused by name | The decoder owners' own gates |

The runtime does not invent a model-file layout or convert weights.
Unsupported inference modes and tensor forms are refused before execution.

The implementation follows the [Qwen config revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json).
Transformers `v5.16.1` resolves to immutable commit [`93c8b7b485963a10800c91f55304db6be211c2bd`](https://github.com/huggingface/transformers/commit/93c8b7b485963a10800c91f55304db6be211c2bd); its [Qwen4-Exp text config](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py) and [text model implementation](https://github.com/huggingface/transformers/blob/93c8b7b485963a10800c91f55304db6be211c2bd/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py) are the reference contract.
The GGUF map was inspected in the reference runtime's [Qwen4Exp converter](https://github.com/mxxm-t/mx-llama.cpp/blob/c061df19838ff60970faf54fd7e414953590125d/conversion/qwen4exp.py), [model loader](https://github.com/mxxm-t/mx-llama.cpp/blob/017ad444637cbb9acb4528a396b7d7ba6fea0a50/src/models/qwen4exp.cpp), [tensor-name map](https://github.com/mxxm-t/mx-llama.cpp/blob/017ad444637cbb9acb4528a396b7d7ba6fea0a50/gguf-py/gguf/tensor_mapping.py) and [GGUF architecture metadata](https://github.com/mxxm-t/mx-llama.cpp/blob/017ad444637cbb9acb4528a396b7d7ba6fea0a50/src/llama-arch.cpp).
These are reference evidence, not code to copy.
The reference preparation (`tools/qwen4exp_reference.py`) also pins the Torch version, attention implementation, dtype, cache settings and fixture hashes, which Transformers alone does not.

### 1.1 Pinned text config and refusals

The pinned outer config is `model_type=qwen4_exp` with `architectures=["Qwen4ExpForConditionalGeneration"]`, and contains separate `text_config` and `vision_config` objects.
llmx runs the text model only: it reads the text fields below and has no image or video input path.
Packaged vision tensors are not unsupported text inputs; the text graph does not read them, and the MTP block runs only as a drafter (section 4).
The pinned Transformers source also exposes `Qwen4ExpForCausalLM` over a `Qwen4ExpTextConfig` for a text-only reference.

The released text config uses BF16 weights and has hidden width 2560, vocabulary 248320, 48 decoder layers, 24 attention heads of width 256, 2 main KV heads, maximum position 262144, RMS epsilon 1e-6, RoPE theta 10000000, partial rotary factor 0.25 and interleaved MRoPE sections `[11, 11, 10]`.
It sets four hyper-connection streams with low rank 320; 16 linear-attention key heads and 48 value heads, both width 128, with convolution width 4; and 512 experts with top-10 routing, expert/shared intermediate width 640, SiLU and sigmoid output gating.
Layers are linear attention except every fourth layer, which is QSA full attention.
QSA has four query heads, one key head, width 128, token budget 2048 and compression ratio 4.
PLE is enabled on one-based layer 2, with 3-grams, eight heads per n-gram order, embedding width 2560, depthwise convolution width 4, base vocabulary 20,000,000, 128-way table padding and EOS id 248044.

The pinned `@strict` text config validates conditions that the module's reader enforces on the GGUF metadata as well: layer kinds are only `linear_attention` or `qwen_sparse_attention` after `full_attention` entries have been normalized; `hc_count > 1`; expert count and both feed-forward widths are positive; and `1 <= num_experts_per_tok <= num_experts`.
If any QSA field is present, all five (`indexer_n_heads`, `indexer_kv_heads`, `indexer_head_dim`, `indexer_budget`, `indexer_compress_ratio`) are required and positive, `indexer_kv_heads == 1`, the budget is divisible by the compression ratio, and `int(head_dim * partial_rotary_factor) <= indexer_head_dim`.
PLE layers are one-based in HF and zero-based in the GGUF, in range and linear-attention only; the total head count `(ngram_size - 1) * heads_per_ngram` and embedding width must be positive, the width must divide evenly across heads, and EOS must be present.
HF supports only `sigmoid` and `silu` output gates; the GGUF carries no gate key, and the module applies the released contract's sigmoid (section 4).
Explicit malformed fields are refused; defaults apply only where the pinned config defines them.

### 1.2 GGUF metadata and Qwen4Exp tensor map

The GGUF architecture string is `qwen4exp`; architecture metadata keys below therefore use the `qwen4exp.` prefix.
The converter writes trunk layer kinds as `qwen4exp.attention.recurrent_layers` (true for GDN, false for QSA) and emits `qwen4exp.attention.compress_ratios` with 4 on QSA layers and 0 elsewhere.
For this checkpoint the PLE metadata is `qwen4exp.ple.layers=[1]` (zero-based), `qwen4exp.ple.ngram_size=3`, `qwen4exp.ple.heads_per_ngram=8`, `qwen4exp.ple.conv_kernel=4`, `qwen4exp.ple.eos_token_id=248044`, `qwen4exp.embedding_length_per_layer_input=160`, plus the exact integer arrays `qwen4exp.ple.layer_multipliers` (3 entries), `qwen4exp.ple.head_offsets` (16 entries) and `qwen4exp.ple.head_vocab_sizes` (16 entries).
The QSA keys are `qwen4exp.attention.indexer.head_count=4`, `qwen4exp.attention.indexer.key_length=128` and `qwen4exp.attention.indexer.top_k=2048`.
Hyper-connection metadata is `qwen4exp.hyper_connection.count=4` and `qwen4exp.hyper_connection.low_rank=320`.
The GDN geometry is described by `qwen4exp.ssm.conv_kernel=4`, `qwen4exp.ssm.inner_size=6144`, `qwen4exp.ssm.state_size=128`, `qwen4exp.ssm.time_step_rank=48` and `qwen4exp.ssm.group_count=16`.

Names and stored shapes below use GGUF order (input dimensions first).
The standard embedding, output, attention, GDN and MoE weights use the existing Qwen mappings.
Architecture-specific tensors are:

| HF state name | GGUF tensor | Shape for pinned config |
|---|---|---:|
| `model.hyper_connection_mixer.hc_norm.weight` | `output_hc_norm.weight` | `[2560, 4]` |
| `model.hyper_connection_mixer.input_mix_weight_down.weight` | `output_hc_down.weight` | `[10240, 320]` |
| `model.hyper_connection_mixer.input_mix_weight_up.weight` | `output_hc_up.weight` | `[320, 10240]` |
| `model.layers.i.attn_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}` | `blk.i.hc_attn_{norm,down,up,inject}.weight` | `[2560,4]`, `[10240,320]`, `[320,10240]`, `[10240,4]` |
| `model.layers.i.mlp_hyper_connection.{hc_norm,input_mix_weight_down,input_mix_weight_up,block_inject_weight}` | `blk.i.hc_ffn_{norm,down,up,inject}.weight` | `[2560,4]`, `[10240,320]`, `[320,10240]`, `[10240,4]` |
| `model.layers.i.self_attn.indexer.index_qk_proj.weight` | split to `blk.i.indexer.q_proj.weight`, `blk.i.indexer.k_proj.weight` | `[2560,512]`, `[2560,128]` |
| `model.layers.i.self_attn.indexer.{q_layernorm,k_layernorm}.weight` | `blk.i.indexer.{q_norm,k_norm}.weight` | `[128]` each |
| `model.layers.1.ple.{key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}.weight` | `blk.1.ple_{key,value,norm_key,norm_query,norm_conv,conv1d}.weight` | `[2560,10240]`, `[2560,2560]`, three `[2560,4]`, `[4,10240]` |
| `model.layers.1.ple.ple_embedding.ngram_embedding.weight` shards | `per_layer_token_embd.weight` | `[160,320001536]` when padded to 128 rows |

The converter adds one to each zero-centered HF norm gamma before writing it; runtime norm application is ordinary multiplication by the stored gamma.
It splits the combined indexer Q/K projection at `indexer_n_heads * indexer_head_dim`, squeezes the singleton conv axis, and concatenates PLE checkpoint shards in numeric shard order.
Hash constants are read as int64 before generic conversion can cast non-floats to F32.
The source table is converted to F32 before any requested GGUF quantization; its row width is 160, while gather ranges define the required vocabulary extent.
PLE input width is the row width times the number of n-gram heads; it need not equal the model hidden width.
The pinned released config has SHA-256 `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b`.
The six GGUF shard identities are pinned against the Unsloth conversion at `38bb39ee97821de2c9009abb7e93950eec396e66`; the file-exact output records each shard's size and SHA-256 and checks the GGUF architecture metadata against the pinned HF configuration.

The 16 PLE heads each use the next prime after 20,000,000 through the 16th prime after that base.
Their combined vocabulary is 320001446 rows and the divisor 128 pads it to 320001536; at width 160 this is 51,200,245,760 parameters, or about 95.4 GiB in BF16.
This is why the table is a bounded lookup whose storage placement chooses before the loader reads any of it (section 3), not an ordinary weight loaded whole.

### 1.3 Reference math that must remain exact

PLE hashes valid token IDs with signed 64-bit integer arithmetic.
The pinned multiplier generator bounds each positive odd multiplier below `floor((2^63-1)/vocab_size)`, so the product for `0 <= token_id < vocab_size` fits signed int64.
For each order `n=2..ngram_size`, the current and preceding `n-1` token IDs are multiplied by the corresponding stored odd multipliers, the products XORed, the result taken modulo each head's stored prime vocabulary, and its stored row offset added.
Shifts that cross an EOS boundary or sequence start supply EOS; the per-sequence token tail has length `ngram_size-1`, starts as EOS and is checkpointed with the sequence.
There are eight independent heads for each n-gram order, each gathering a 160-wide row; order and duplicate gathers are preserved.
The stored offsets, multipliers and sizes are part of the file's integer contract, never recomputed from rounded floats.

QSA stores one raw index key per visible token.
For each query, the visible tokens in cache order divide into complete consecutive groups of four and a causal partial tail.
The native selector (`select_kv`) averages each complete group's raw F32 keys, applies F32 RMSNorm with the stored GGUF multiplier (which already includes HF's `1 + gamma`), and applies split-half RoPE at the group's first token position.
Its query input is already normalized and RoPE-rotated.
It computes F32 dot products per indexer head, applies ReLU to each head score, sums heads and divides by `sqrt(dim)`, selects up to `budget/compression` complete groups and appends every visible token in the partial tail.
IDs are emitted chronologically and padded with `UINT32_MAX` to `budget + compression - 1`.
The deterministic tie rule selects earlier groups first; HF `topk` does not specify tied-index order, so exact tie membership is not a cross-backend HF contract.
This primitive is all-F32 and does not itself run sparse attention.
A separate backend operation (`sparse_attention`) attends over selected full-resolution K/V with GQA and independently supports F32/F16 K and V cache sides; it ignores padding and IDs beyond each causal end, and returns zeros when no selected ID is valid.
The module's QSA layers call the selector and then sparse attention.
Tensor groups declare head, expert and vocabulary shards; hyper-connection injection produces full-stream partials for the existing collective; a lookup's rows are read once per group and uploaded to each member's own buffer.
PLE saved rows are replicated, while GDN saved inputs follow its head shards.

The HF source makes key precision boundaries observable: QSA pooled keys and indexer scores are explicitly F32; delta-rule inputs/state are accumulated in F32 and its output is cast back to the input dtype; RMSNorm computes from F32 input and `(1 + stored_gamma)` in F32, then casts to the source dtype.
RoPE frequency construction, cosine and sine run with autocast disabled in F32 and are cast to activation dtype.
Chunk GDN uses chunk size 64, while one-token cached decode uses the sequential recurrent rule.
The tiny reference fixes its attention backend, dtype and cache mode as part of the numeric oracle; that pin alone does not settle PyTorch kernel-level tie or reduction behavior.
The native selector does not add HF's model-dtype casts around the pooling and normalization stages; its operator and tests use F32 operands.

For a one-sequence text cache, the layer kinds own distinct history.
A linear layer carries its GDN convolution tail (3 positions of `2*2048 + 6144 = 10240` channels), its F32 recurrent state `[48,128,128]`, and no attention KV.
The PLE layer also carries its dilated-convolution tail `[10240,9]` because the kernel is 4 with dilation 3, plus its integer token tail `[2]`; these are separate from the GDN state.
A QSA layer carries ordinary full-resolution K/V (2 KV heads, head width 256) and one raw indexer key `[128]` per token.
Grouped index keys are derived from this cache and are not a replacement for the main K/V.
Every active history is captured at the same logical sequence position by checkpoint, fork, retract and failure rollback, while retaining its layer ownership: PLE adds convolution and integer token history to its GDN layer; QSA owns its ordinary K/V cache and auxiliary raw indexer-key history.
`DynamicCache`'s `conv_states` indices are 0 for GDN, 1 for PLE convolution and 2 for PLE token history.

The pinned hyper-connection mixer normalizes each of the four hidden-width streams separately, applies `(1 + gamma)`, maps the flattened streams through a SiLU of `down(x)/4` and a sigmoid `up`, then forms the mean of gate-weighted normalized streams.
Its injection is `2*sigmoid(inject(x)/4)`.
The block's single output vector is multiplied by its per-stream injection weights and added back to the corresponding unnormalized residual streams.
The same mixer shape is used before attention, before the MoE and once for the final head.
This defines the reference structure; the test oracle keeps the matmul, reduction, activation and cast order of the pinned HF implementation.

PLE adds to the stream residual before the attention mixer.
Its 160-wide-head embeddings are concatenated into 2560 features; separate linear projections produce a 10240-wide key and 2560-wide value.
The key and stream query use grouped RMSNorm, the gate is `dot(key, query)/sqrt(2560)` per stream, then the sign of the gate times `sqrt(max(abs(gate), 1e-6))`; sigmoid gates the repeated value.
A grouped norm of this gated value feeds the causal depthwise width-4 convolution with dilation 3 and SiLU.
The returned PLE output is the unnormalized gated value plus that convolution result.
Convolution history therefore spans nine prior positions, and the PLE weights are not just a token embedding lookup.

## 2. Architecture and ownership

The model combines gated delta-net layers with sparse attention, routed and shared experts, learned mixing of four residual streams, and per-layer n-gram embeddings.
The supplied configuration has 48 layers, 512 routed experts with 10 selected, and a roughly 51B-parameter n-gram table.
The table is about 102 GB at two bytes per parameter, before overhead; quantized sizes differ.

Sparse attention compresses the indexer's view for selecting blocks; it still attends to original per-token K/V, and the main cache is not replaced by a compressed-KV approximation.
The partial block at a causal boundary matters.
N-grams need both integer token history and convolution state.
`StateShape` remains the common GDN prefix; `LayerPlan::state_tail` supplies additional per-layer F32 rows after it, including the nine preceding raw rows needed by the PLE dilated convolution.
`dilated_conv_silu_add` applies the causal depthwise convolution and SiLU, adds it to the residual, and updates that F32 tail, on the CPU and on Vulkan.

For token IDs, `LayerPlan::token_history` declares a fixed-width preceding-ID tail and `LayerPlan::token_padding` its sequence-start value, the PLE EOS id.
The runtime keeps one exact U32 array on the host, indexed by the existing state slot IDs, for the one recurrent layer that declares a tail; it does not store token IDs in F32 state or duplicate the ID array on tensor-group members.
Before the owning stage runs, the pass snapshots each entry's preceding IDs into its `Step`; after that stage commits, the runtime advances the slot's tail with the current IDs.
`keep` and fork reuse the slot mapping.
Marks save the current U32 IDs, and retract rebuilds the live tail from the mark's tail and accepted IDs only.

The module `src/model/arch/qwen4exp.hpp` runs the PLE hash, gather and convolution, QSA selection and sparse attention, routed and shared experts, GDN recurrence, and learned residual mixing and injection.
`src/model/arch/registry.hpp` maps `qwen4exp` to the module with the BF16 architecture default and the module's drafter normalization, and ordinary loading and the CLI and server commands dispatch to it.

The hash, EOS handling, pooling and normalization order, residual gate math, precision boundaries and cache updates follow the pinned reference (section 1.3).
The native selector's tie rule is fixed to earlier groups.
HF top-k tie behavior is unspecified, so the tests claim no HF tie-membership parity and do not hide selection changes behind a wider logit tolerance.

| Concern | Owner |
|---|---|
| Model keys, tensor names, shapes, refusals and graph | `src/model/arch/qwen4exp.hpp`, one module |
| Registration | `src/model/arch/registry.hpp`: `qwen4exp` to `qwen4exp::open`, the BF16 default and `qwen4exp::normalize_drafter` |
| Shared graph math | `src/model/arch/blocks.hpp` and the existing delta-net and MoE operations, where their math matches |
| Numerical operations | Mathematical contracts in `src/backends/backend.hpp`; CPU implementations under `src/backends/cpu/`, Vulkan implementations and shaders under `src/backends/vulkan/` |
| Persistent sequence state and rollback | The shared runtime, history and backend storage, extended by declared state requirements (section 4) rather than architecture-name branches |
| Immutable lookup rows | `src/model/weight_rows.hpp` (`WeightRows`) reads bounded rows from memory or a file; `TensorView::row_source` connects a declared lookup to the runtime, and `Model::start_lookup` and `Model::finish_lookup` in `src/model/passes.hpp` stage each pass's rows |
| Lookup storage | `src/model/place.hpp`: `PlacementRequest::lookup_storage` and the fit choose `device` or `file` for each table (section 3) |
| File opening, model spans and loading progress | `src/inference/load.hpp` builds each table's row source over its exact `format::FileSpan` before placement, and holds the mapping or the direct reader the chosen storage reads through |
| Memory fit and placement | `src/model/place.hpp`, `layer_split.hpp` and `shard.hpp` over the module's declarations |
| CLI and server | Parse supported settings and invoke the existing model path; no hash, tensor-layout, loading or table-residency logic |

`WeightRows` is not a second loader or a general cache: it reads the rows a pass names from a retained mapping or through the file's reader.

## 3. Lookup table storage

The PLE table is declared as a lookup (`LayerPlan::lookup`) over a mixer `RoleKind::gather` role, and the architecture computes each pass's row ids (`Architecture::lookup_ids`) from the tokens and the layer's token history.
Hashing, row layout interpretation and the learned operations stay in the architecture; the storage and the transfer are generic, so another architecture with an immutable lookup table reuses them.

`--lookup-storage` chooses where each declared table lives, independent of `--load-mode` ([USAGE](USAGE.md)):

- `device` keeps the whole table on the devices that run its layer; under a fitted KV budget the budget is fitted beside the table, and otherwise a table that does not fit is refused.
- `file` reads each pass's rows from the model file and uploads only those packed rows: through the file's mapping in `--load-mode auto` and `mapped`, and through direct reads in `direct`.
- `auto` takes `device` where the table fits beside the fitted KV budget, and `file` otherwise.

The loader (`src/inference/load.hpp`) builds a `WeightRows` source over each table's exact file span, and placement (`plan_placement` in `src/model/place.hpp`) chooses each table's storage before the loader reads any whole table.
A table read from its file is counted in the fit by its host and device row staging alone, and never joins the load's stream or the warming of `mapped`.

Each pass, `Model::start_lookup` has the architecture compute the row ids on the group's first member as the stage begins and starts reading each distinct row once on the reader's threads (`WeightRows::start` sorts and deduplicates the ids, coalesces contiguous rows and keeps an ordered remap), and `Model::finish_lookup` takes them just before the layer and uploads the packed rows to each member of the group.
The backend's gather then reads them in the request's order, duplicates included.
Ordinary passes and passes recorded apart have separate row staging and remap storage, since the ordinary embedding reuses its index scratch.
The recurrent rerun after a retract uses the saved PLE inputs and reads no rows again.
With the table on the devices, the module gathers from the resident table and reads nothing from the file.

The released Vulkan gather uses 32-bit byte offsets for quantized resident tables.
The 26.8 GiB IQ4_NL PLE table exceeds that range: high resident rows wrap, while the file-backed path uses wide host offsets and small remapped uploads. An independent failing-first Vulkan reproducer confirms the defect on both host-ID and device-ID gathers; it also confirms that the full-table descriptor exceeds the device's storage-buffer range. The authoritative record is `worktrees/vulkan-large-lookup-20261020/docs/benchmarks/vulkan-large-lookup-repro-20261020.json` at commit `6a3adff52fe8d3da76bff750feb2f5c4b4dd32d2`.
Use file-backed PLE for this released model until both the large-table addressing and descriptor-range constraints are fixed and qualified.
The resident timing and repeated-prompt controls are retained in [the 2026-10-20 record](benchmarks/flash-q4-p16384-warmup-20261020.json); its resident PP 706.774 repeated mean is correctness-unqualified and must not be interpreted as valid inference performance. Successful loading alone does not prove correct resident lookup.

Weights stay immutable under the model file's lifetime; there is no writeback and no second on-disk copy.
A file changed or truncated is refused where the readers can detect it; a mapping does not make external truncation safe.
The server's host and disk tiers hold conversation histories, not table rows: model code does not depend on them, and table rows never enter donor ranking or persisted cache files.
Lookup storage neither prefetches nor retains rows across passes (section 5).

## 4. Expert placement and sequence state

The routed expert stacks are declared as whole-stack windows (`Stream::window`) and the router, shared expert and its gate as copies (`Stream::copy`), so the existing expert placement applies unchanged.
Experts stay on the devices by default.
`--n-cpu-moe N` or `--cpu-moe` runs the first N routed layers' experts, or every layer's, on the CPU beside one device, and is refused with several devices or a tensor split.
`--moe-stream-from N` copies those layers' whole gate, up and down stacks into device windows for prompts from N tokens (`Model::ffn_split`); it transfers whole stacks, not only the selected experts.
Routing, shared-expert gating, dtype and reduction are the same on every path.

State follows the layer kind.
A linear-attention layer keeps its GDN convolution tail and F32 recurrent state in a state slot, `StateShape` giving the common prefix.
The PLE layer adds its dilated convolution's nine raw rows as a per-layer F32 tail after that prefix (`LayerPlan::state_tail`) and its preceding token IDs as the host token-history tail (`LayerPlan::token_history`).
A QSA layer keeps ordinary attention K/V and a per-position F32 raw indexer key in an auxiliary plane of the same KV storage (`LayerPlan::kv_aux`); pooled keys and partial-block selection are derived from it, and the MTP block's QSA keys use the same plane.
Allocation, growth, host copies and the persisted identity include the auxiliary plane, and tensor-group members each keep the complete auxiliary row and the complete state tail beside their head-sharded prefix.
No layer combines GDN recurrence with attention K/V, so there is no combined KV and state cache mode.

State creation, checkpoint, fork, retract, reset, context cut, disk save and restore and failure rollback cover the complete state at one logical position through the shared runtime, with no architecture branch.
The whole-host image, the host restore and the streamed restore walk the same runs (`detail::block_runs` and `detail::state_runs` in `src/model/history.hpp`): auxiliary KV, every layer's state with its tail, the MTP carry and the host token history.
The host identity is version 2 where a token history exists, naming its owner stage, width and padding beside the auxiliary widths and state strides, so a persisted history of another layout is not misread.
A restore mode a model does not support is refused, not silently missing state.
Cut prompts are recomputed at their new positions under the normal cut owner, with no recurrent-state splicing, and a failed pass drains its tickets before storage is reused.

`Model::keep` and `Model::mark` transfer the ownership of a live state slot without copying a tensor, and a fork shares its immutable history blocks and checkpoint until execution needs a writable destination.
The PLE convolution's F32 tail follows its state slot, and the U32 token tail follows the same slot IDs through the host array, with a mark's rows saved separately and accepted IDs replayed on retract.
A partial speculative rejection reruns only the accepted rows' two convolutions and GDN recurrence from their saved inputs; projections, lookup, stream mixing, attention and experts are not rerun.
QSA index rows are truncated with K/V, so rejected rows cannot affect later selection.

MTP drafts through the existing drafter hooks: an embedded block after the layers is planned by `Architecture::plan_drafter`, and a sidecar file is paired with the target through the registry entry's `normalize_drafter`, which adapts the observed sidecar headers to the block-only contract after tokenizer equality (MTP development checkpoint, below).

Residual width, per-part hidden width and tensor-group shard width are distinct: the residual row holds four streams (`ModelPlan::residual`), and the fit counts the four-stream activations and boundary transfers.
Indexer heads and the shared lookup result are declared once, and reductions and top-k are global where the math requires it.
A table kept on the devices is an unsharded weight, whole on every member of a group; a table read from its file is staged once per group and uploaded to each member.

The GDN output gate is sigmoid for the released Qwen4Exp GGUF contract, unlike Qwen35's SiLU.
HF's `output_gate_type` must be explicitly set to `sigmoid` in reference configurations: omitting it selects SiLU through `hidden_act` and produces a different graph that the keyless GGUF metadata cannot represent.

## 5. Remaining coverage

Section 7 holds the open items of the gates; these are the coverage items the built paths still lack.

- Lookup storage on the released table: cold and warm reads, low and high reuse, prompt and decode batches and concurrent requests, with useful and physical bytes, wait time and end-to-end throughput; row retention across passes, known-token prefetch and a hot-row device cache follow only where such a profile shows they pay.
- Failure and concurrency on the released file: device failure recovery, interruption during disk writes and concurrent persistence.
- Released-file serving: multi-user performance and tensor-group serving beyond the recorded cells.
- Checkpoint cost on this model: bytes copied and allocated, and latency, of keep, mark, fork, first continuation and retract at short and long prefixes and several accepted draft lengths, device timings separating submission from completion.

## 6. Correctness and performance evidence

### Released-file contract (agreed 2026-10-19)

The contract below was agreed on 2026-10-19 and supersedes the 2026-10-17 contract.
The historical 492/512 failed the former 512/512 depth rule and is not a pass under it.

1. Before any reference forward, freeze all six shard SHA-256 identities, tokenizer/config, full HF revision (the retained `93c8b7b4` revision), modeling/helper/sequential-control hashes, execution settings, both input texts and ID vectors, scoring rules, candidate source/binary and flags.
   The initial runtime is `f073e6b51b3c240758ec5c45d366dd915161d3b9`, binary `b1d79c468f8daccaea4c2914a440461e7956208b4530b69031b58dde456d1405`, with F32 compute/K/V on physical GPUs 0-5 in three width-two stages.
   Reuse verified full model hashes when file identities still match.
2. Keep exact model/tokenizer/storage/state/checkpoint contracts, independent operator oracles, and tiny full-graph bounds of 2e-5 per logit and 1e-5 NLL.
   Released family prompts retain exact top-1, top-five overlap 5/5 under the existing 0.1 boundary rule, finite sorted top-ten and magnitude limit 100.
   Continuous cases at 512/4096 in batched and per-token modes require absolute aggregate mean-NLL delta <=0.01 against canonical file-exact HF; windowed cases require <=0.02.
   These are existing `FILE_EXACT_BOUNDS` from `tests/baseline.py`, applied by `common.check_ppl`, not allowances fitted to Flash.
   Retain exact counts, finite NLL/PPL and per-window data; the window limit applies to the complete case's target-weighted mean, not a newly imposed per-window cutoff.
   Corpus counts remain 1,207/4,565 continuous and 1,205/4,564 windowed targets.
   No extra reference calibration is required for these corpus cells; a failure remains a correctness failure.
3. Use exactly two fresh 16,384-token inputs, one raw and one chat-rendered, held out from development and used jointly for calibration and comparison.
   Retain 513 prediction rows at positions 15871..16383.
   Classify the final 512 prediction rows at 15872..16383 with `common.hf_depth_top1`: exact canonical top-1 or its second choice when the top-two gap is <=0.1.
   The 512 NLL targets are input positions 15872..16383, predicted by the preceding rows.
   Use full-history teacher forcing; exclude no rows.
4. Run canonical HF chunk-64 and HF sequential GDN in F32 on both inputs, with identical stored weights, IDs and pinned attention/cache settings.
   For each input, let e be its 512 paired NLL differences, sequential minus canonical.
   Freeze three limits: twice the larger input's top-1 miss count, twice the larger RMS(e), and twice the larger abs(mean(e)).
   The factor two is an empirical policy, with reference-only precedent in `tools/calibrate_dtype.py`, not a proven error envelope.
   Hash all four outputs and record numerical limits before native reads either input.
   Add no floor for zero spread; adjust nothing from native results.
5. Run the frozen native candidate once per input against canonical HF.
   Each input must independently meet all three limits.
   Also report signed mean and every miss's position, canonical margin and reference-arm agreement; do not cancel biases across inputs or exclude ambiguous rows.
   An exceeded limit means NOT QUALIFIED, with its cause still to investigate.
6. Retain existing common-input CPU F32/Vulkan F32/F64 two-chunk evidence without repeating it.
   It does not reproduce accumulated native 16K history.
   Full-model CPU execution is no longer a precondition.
   Any optional bounded F64/operator experiment is attribution only, neither a waiver nor proof of a defect by itself.
7. A later native run requires a named code change and a recorded hypothesis.
   Pin its new source/binary; reuse the same inputs, references and limits.
   Those inputs are then regression data, no longer unseen.
   Model, tokenizer, reference, input or scoring changes require a new experiment freeze.
8. Merge still requires the exact and calibrated depth cells, serving/MTP requirements, and all applicable AGENTS model/device gates: supported-model identity against main, device suite, Radeon VII checks, matched timing and hosted CI.
   Other precision modes keep separate qualification.
   Performance waives nothing; the incomplete split comparison remains separate.

No repeat-control headroom, additional scheduling pilot, cache-split arm, decay candidate, confidence interval, partial-row KL, candidate-derived threshold or localization without a named violation is part of this contract.

### Qualification (2026-10-19)

**Candidate and inputs (item 1).** The [preparation record](benchmarks/qwen38-reference-qualification-preparation-20261019.json) freezes the model's six shard hashes, the HF pins, the scoring rules and two fresh 16,384-token inputs, raw and chat-rendered, chosen after the candidate was pinned and disjoint from every earlier input (raw ids SHA-256 `5583a03811186ad1efaba01f32a6f929b351bbf742eef533d510e2434816fb6e`, chat `7029be5df711c1b8e662cb965519455cf29e8020add7295aca9d4e8670d8bf04`).
Before native read either input, the candidate was re-pinned to the integrated merge head: source `14e4867a9`, binary SHA-256 `8b38f9533f70ccf4e301748e5e57c5bfdcf622a976edb7f2ed6a84647e430866`, numerics `b9bfb15f0016ba2d`, whose Vulkan library is byte-identical to the first candidate's.
Native flags: `--dtype f32`, f32 caches, six MI50s as three stages of width-two tensor groups, `--lookup-storage file --load-mode auto` (run on 2026-10-19 under the residency then named `disk`, whose rows were read through a buffered reader rather than the mapping, the same bytes).

**Corpus cells (item 2).** Every cell is within `FILE_EXACT_BOUNDS` of canonical HF on the file's own weights, with every target count exact:

| Cell | Native mean NLL | HF mean NLL | Native - HF | Bound |
|---|---|---|---|---|
| 512 window, batched / per token | 1.338838830 / 1.338869209 | 1.338869135 | -0.000030 / +0.0000001 | 0.02 |
| 512 continuous, batched / per token | 0.669878793 / 0.670253060 | 0.670253136 | -0.000374 / -0.0000001 | 0.01 |
| 4096 continuous, batched / per token | 0.353694912 / 0.354778478 | 0.354255545 | -0.000561 / +0.000523 | 0.01 |
| 4096 window, batched / per token | 0.427022647 / 0.427114248 | 0.427103883 | -0.000081 / +0.000010 | 0.02 |

The 4096 window reference ran HF eager attention, which the contract leaves out of scope for these cells.

**Depth limits (item 4).** HF chunk-64 against HF sequential GDN, F32, on both inputs, gave these spreads (outputs SHA-256: raw chunk-64 `fac51e596260dec78ea9d9e46274c143ed20fe24887c542efeb05657eb90bec3`, raw sequential `93746fe9db2a7bce2aa2c0574a7252fb42ce5a02f98ed772f9f2bf56c54b05db`, chat chunk-64 `ed48b56764f6041f17773a8bcee954c15c8fe55961545edd07813f4266347cd3`, chat sequential `f669e08102a388136ddd3581e101cfd6707a0b8828a475eb62c199f551eb40d6`):

| Input | Top-1 exact / tie / miss | RMS(e) | signed mean(e) |
|---|---|---|---|
| raw | 510 / 1 / 1 | 0.034035 | +0.002263 |
| chat | 499 / 4 / 9 | 0.715498 | +0.049302 |

The frozen limits, twice the larger input's figure and held per input, are 18 top-1 misses, RMS(d) 1.430996 and abs(mean(d)) 0.098604 (limits record SHA-256 `28d4ec481ee89a37f6a49120da8dd083785c579a72a5a90dca474e10bdd4e3af`).
The chat input's HF arms part by up to 14.94 nats at single positions, which makes the RMS limit about 42 times the raw input's spread; the limits stand as item 4 froze them.

**Native run (item 5), once per input: qualified.**

| Input | Top-1 (exact + tie) of 512 | Misses | RMS(d) | signed mean(d) | Within the frozen limits | Within twice the input's own HF spread |
|---|---|---|---|---|---|---|
| raw | 511 (508 + 3) | 1 | 0.040377 | +0.001247 | yes | yes (limits 2, 0.0681, 0.0045) |
| chat | 499 (495 + 4) | 13 | 0.737503 | +0.008458 | yes | yes (the same as the frozen limits) |

The raw miss, at 15950, is where the two HF arms disagree and native takes the sequential arm's choice.
Of the chat input's 13 misses, 4 are where the HF arms disagree and native takes the sequential arm's choice, and 9 are where they agree, 7 of those turning on token 248046 (the GGUF end of text) against canonical margins of 0.14 to 2.59.
The largest per-token difference, +14.82 nats at 16196, is where the HF arms part by +14.94, native following the sequential arm.
Every non-exact row, with its position, margin and the arms' agreement, is in the native score records (SHA-256 raw `c26f2311b9e12d0cd06f3cf9579efa86c6de8d3b80d6c93763c40937246c7a9b`, chat `03065b0f36609f955df5d5d049ea0ec3d34df2d68dd0d184143129960d3b2425`).
The historical 492/512 failed the rule this contract replaced and is no evidence either way.

**Natural-stopping MTP acceptance.** Three prompts, greedy, natural end of text, draft depth up to 3:

| Prompt | llmx, kept/drafted | mx-llama.cpp ROCm, kept/drafted |
|---|---|---|
| code_fix | 122/138 = 88.4% (170 tokens) | 99/117 = 84.6% (138 tokens) |
| incident_summary | 83/93 = 89.2% (115 tokens) | 83/96 = 86.5% (116 tokens) |
| ordering | 87/100 = 87.0% (136 tokens) | 114/138 = 82.6% (161 tokens) |

llmx ran the qualified candidate at F32 over three width-two stages, and every reply with MTP on equals its reply with MTP off.
mx-llama.cpp ran its ROCm build on a six-device layer split, ubatch 512, f16 caches; its staged split and its f32-cache servers failed on the first request (`CUBLAS_STATUS_INTERNAL_ERROR`) and its Vulkan staged server returned NaN log-probabilities, so those are not references.
Each runtime is scored on its own reply, so the comparison is per prompt and not per position.

### Speed gate (2026-10-19)

Mode by mode on MI50s: llmx `--dtype int8` (the gate arm, the precision of the reference's 8-bit activations) against the ROCm build of the reference runtime, mx-llama.cpp, in that mode, each side at its best ubatch; llmx `--dtype auto` (f16) and mx-llama.cpp's Vulkan build are reported beside them.
Every arm used the same devices, CPUs 6-13 with 8 threads, a 16 GiB host cap and f16 caches; mx-llama.cpp took the environment and flags its documentation recommends (fine-grain PCIe, eight hardware queues, its custom AllReduce, `-lm dio`, `-fa on`) and a batch of the whole prompt, so its microbatches outnumber its stages, and logged "pipeline parallelism enabled" in every run.
One warmup and three timed runs per process, tokens per second:

| Mode, devices | llmx int8 PP16384 / PP4096 / TG256 | llmx auto | mx-llama.cpp ROCm | mx-llama.cpp Vulkan | int8 against mx-llama.cpp ROCm |
|---|---|---|---|---|---|
| Layer, 6 | 2313.3 / 1910.3 / 28.82 | 1784.8 / 1455.5 / 28.77 | 300.3 / 306.5 / 35.53 | 263.2 / 273.8 / 17.45 | prompt 7.7x / 6.2x; decode -18.9% |
| Staged 3x2, 6 | 871.2 / 928.1 / 32.61 | 753.0 / 790.4 / 32.56 | 284.1 / 288.3 / 37.84 | nonfinite / 257.3 / 10.59 | prompt 3.1x / 3.2x; decode -13.8% |
| Staged 2x4, 8 | 470.7 / 508.4 / 27.16 | 439.5 / - / 26.99 | 286.3 / 305.3 / 37.87 | not run | prompt 1.6x / 1.7x; decode -28.3% |
| Layer, 10 | 3008.1 / - / 22.21 | not run | 212.3 / - / 31.45 | not run | prompt 14.2x; decode -29.4% |

Cells without a comparison count for neither side: a single tensor group of six (llmx groups hold two to four devices; mx-llama.cpp ROCm 287.2 / 317.5 / 39.69), two stages of three (width 3 does not divide the 248320 vocabulary rows; mx-llama.cpp ROCm 293.6 / 314.3 / 39.55), and a lone group of four or of two, which neither runtime fits in device memory.

**Open: decode.** TG256 is below mx-llama.cpp ROCm in every comparable mode, by 18.9% (layer, 6), 13.8% (staged 3x2), 28.3% (staged 2x4) and 29.4% (layer, 10).
A traced budget of one token on the six-device layer split, taken before the lookup change below, with the reference measured beside it at 28.1 ms a token, puts 33.1 of llmx's 38.5 ms in GPU busy, the six stages of a token running one after another, and about 4.5 ms on the host: the lookup rows read before the first stage (2.6 to 3.0 ms, buffered reads through a compressed file system), each stage's first chunk recorded while its GPU waits (about 1.5 ms) and five crossings (about 0.6 ms).
On staged 3x2 the 93 cross-device sums of a token add about 5.3 ms of waits, and every member runs the stream mixing, the routers and the routing at full size.
The rows' read went with this change, `file` reading them through the mapping (H1 below): TG256 on the six-device layer split went from 26.40 and 26.29 to 29.05 and 29.19 tokens per second, 3.6 ms a token less (37.96 to 34.34 ms), and prompt throughput at 512 tokens from 272 to 653.
The named recovery is three host items, H1 the rows through a read-only mapping (2.6 to 3.0 ms, done here), H2 a stage's first chunk flushed early (about 1.2 ms) and H3 the crossing rows copied on the device (about 0.3 ms), and four kernels, K1 `group_rms_norm` (about 2.2 ms), K2 the low-rank stream-mixing products (about 1.7 ms, a numerics change), K3 the activation quantizes folded into their producers (about 0.9 ms) and K4 `moe_route` (about 0.8 ms).
Their projection is 35.0 tokens per second against the reference's 35.5 on the layer split and 38.3 against 37.8 on staged 3x2, parity at best, so the gate stays open after them; the sums' floor on staged and the per-member duplicated work are outside them.

### Reference and performance method

A reduced random-weight HF fixture covers every operation and state kind, including tiny tables, routed/shared experts and both attention kinds.
It is the complete graph, not a substitute network that avoids the hard operations.
Separate real-dimension operator cases reach the kernels that small models skip.
Check integer hashes, table addressing, EOS boundaries, convolution dilation, indexer compression boundaries, partial blocks, selection ties and causal masks.

For the released model, the file-exact companion runs HF's math on the exact stored weights through a memory-bounded reference execution: it reads only the selected lookup rows and experts with the spec decoders, not llmx's decoder, and changes storage access only, not the model's math or integer hashing.
If an end-to-end cell cannot be run, name the gap; never call a layer check that cell.
Preserve requested cache type and precision throughout every mode.
Batch, slice and permutation checks, cache eviction and page refill must not silently change row arithmetic.
Keep unsupported modes explicit rather than skipping a requested test and reporting PASS.

For performance, use the reference runtime's revision and backend that support this model, on the same GPUs, file, dtype/cache configuration, prompt and device split, for Vulkan and ROCm and for the layer, tensor and staged tensor splits.
A backend that cannot run the cell is N/A with its reason, not a zero-throughput baseline.
Record cold and warm load, cold and warm table access, pp64/247/512/4096, tg32/128 and pp16384/tg512, with host CPU utilization, bytes read and transferred, host and device peak memory and multi-user TTFT/TPOT percentiles.
Do not flush a shared host's file cache; label warm and cold conditions only where isolated files or measured misses establish them.

| File, context, placement and residency | Metric | Main control | Candidate | Matched reference | Candidate/reference | Activity flags / gate status |
|---|---|---|---|---|---|---|
| Frozen per cell before measuring | tok/s or milliseconds, direction stated | N/A for new architecture; existing-model regression control separately | Measured | Measured or unsupported | Throughput >= 1; latency <= 1 | Full result and open recovery work |

Build arms the same way; record versions/hashes, device identity, clocks, driver, memory budgets and timestamped contention.
Existing-model regressions are not excused by this being new support.
Under AGENTS, initial support may land with specifically recorded open reference-speed cells and immediate recovery work; neither HF correctness nor existing-path regression checks are waived by that allowance.

## MTP development checkpoint (2026-10-18)

MTP runs through the existing architecture drafter hooks and the paired loader, with exact verify and rollback; nothing in the runtime names the architecture.

The wrapper follows the converter at `c061df19838ff60970faf54fd7e414953590125d` and mx-llama.cpp graph at `017ad444637cbb9acb4528a396b7d7ba6fea0a50`, already linked in the source pins above.
Pinned HF modeling source SHA-256 `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459` explicitly ignores `mtp.*`; its HC, attention, expert and decoder equations are available, but there is no official HF MTP wrapper to claim parity with.
The independent scalar-double test is an assembled reference: it reconstructs the input/carry wrapper from the reference runtime's converter and expresses the HF block equations directly, without calling native graph helpers.

For token j, the input uses the target's raw HC-wide residual at j-1, with a zero residual before sequence start.
Each stream has its own `nextn.hnorm` gamma; `nextn.enorm` normalizes the common embedding.
The fused projection reads `[embedding; hidden_stream]`; the older layout uses square `eh_proj` and `fc_hidden` matrices and sums them.
The attention and FFN each use their own HC mixer/injection, then `nextn.hc_head_*` mixes the raw draft residual before the output matrix.
A second draft carries that raw residual, before the HC head.
Context rows compute only the input and attention cache writes.
The runtime's existing residual-sized carry and mark storage and drafter `kv_aux` hold these rows and QSA raw indexer keys, including rejection.

Read-only inspection of four sidecar headers found two legitimate attention contracts.
The unsloth fused Q8_0 sidecar has 34 tensors, including its own embedding and output, and compression tail 0 (ordinary causal attention).
Its shared-target variant has 32 tensors and `nextn_shared_target_tensors=true`.
Both preserve the trunk compression prefix and omit PLE metadata.
The older `mtp-heads/*-v2.gguf` files have 35 tensors and split input projections; their MTP tail is 4 (QSA), while all trunk compression entries are zero.
Their BF16 and Q8_0 variants differ in matrix storage.
Headers alone do not establish weight provenance or pair compatibility.

The existing generic pair/join path expects block-only tensors and matching architecture metadata, allowing per-block arrays to extend the trunk.
The registry now allows one optional module-owned normalization callback, called after tokenizer equality.
Qwen4Exp validates compatible math metadata, the private-global shapes and permitted tensor names, then normalizes those observed layouts to the existing contract.
Private embedding/output tensors are renamed into the MTP block's namespace and read only by the drafter; an explicit shared-target sidecar borrows the target's globals.
Complete PLE omission is filled from the target.
Only the observed legacy all-zero trunk compression prefix is replaced, preserving its MTP tail.
Unrelated metadata differences are refused.
Header changes preserve tensor order, file spans, offsets, payload size and owners.
Ordinary same-architecture draft models, including those carrying their own MTP block, retain their original behavior.
A read-only comparison of the four actual sidecar headers against the released target found no other math-metadata differences.
No released weights were loaded or executed; this is format evidence, not numerical qualification.

`qwen4exp-mtp` covers fused and split projections with dense attention and QSA ratios 2 and 4 against a trunk ratio of 2.
The scalar oracle checks two draft steps and asserts that nonzero auxiliary keys change selected pools.
Native checks hold target logits with the drafter loaded to those without it, repeated drafts to their own bytes, unequal two-sequence chains in reverse length order to separate calls, and partial-rejection and edited-continuation drafts and logits to a fresh history.
Temporary GGUF target/sidecar pairs pass through `infer::load_model` and match the in-memory graph byte for byte, for block-only, private-global and explicit shared-target layouts; refusals cover incompatible metadata, malformed private globals, unsupported tensors, invalid geometry and a tokenizer mismatch.
A split-capable fixture adds a width-two CPU group, two stages of width-two groups and width four, against the same fixture on one backend.
`server-spec` runs the paired fixture through the public loader and the scheduler, on one backend, one width-two group and two width-two stages: loaded/off and enabled replies match fresh no-sidecar controls exactly, drafts are fed, kept and rejected, and a follow-up and an edited suffix reuse the 128-token checkpoint.
It also injects a failure after both lookup replays and one GDN member update over three width-two CPU stages and retries the same mark against a clean control.
Its CPU server component exports the fixture for the CLI, where ordinary and streamed requests match across transport and off/on arms.
These are tiny F32 checks; the device matrix is in the [fixture record](benchmarks/qwen38-mtp-main6e-vulkan-20261019.json).

On the released target and its shared Q8_0 sidecar, over physical GPUs 0-5 as three width-two stages with F32 activations and K/V and disk-backed lookup:

- The exact server-geometry fit passes, and off/on replies of a short prompt and a two-request chat history are identical, with drafts fed and kept and no failures.
- Checkpoint reuse needs full prompts of one row class: a first exercise at 170 and 218 tokens reused nothing because the two extents fall in different Vulkan arithmetic classes, which the scheduler's guard requires, not a lost checkpoint.
  At 521 and 569 tokens, with context and KV pool 1024, warm MTP-off follow-up and edit reuse 512 tokens and match fresh full-history controls exactly.
- Under the production pricing the scheduler may choose zero drafts after reuse, and did on that follow-up; an instrumented run showed pricing choosing zero at every eligible step.
  With the existing test-only unpriced seam, initial, follow-up and edited requests drafted/kept 27/13, 24/15 and 33/11 with zero failures, each reply equal to its off control ([forced coverage record](benchmarks/qwen38-forced-mtp-checkpoint-20261019.json)).
  The shared sidecar's 32 block tensors and borrowed target globals are byte-identical to the private sidecar's.
- The same 569-token follow-up, prefilled fresh and resumed from checkpoint 512, gives byte-identical target and draft rows ([draft-state record](benchmarks/qwen38-mtp-draft-state-20261019.json)).
  After verifying four rows and retracting to the first accepted row, the resumed target and drafter equal a fresh continuation byte for byte ([rollback record](benchmarks/qwen38-mtp-rollback-state-20261019.json)).
  Both arms of each control could share a conditioning defect; neither is an HF, acceptance-quality or reference-runtime comparison, and in the forced 24-token sample the target ranks the end token first while `ignore_eos` forces another, so it is not representative acceptance.
- Natural-stopping controls: llmx with MTP off ended three requests at their natural end token after 170, 115 and 136 tokens, with exact tokenization and finite log-probabilities ([natural-control record](benchmarks/qwen38-mtp-natural-controls-20261019.json)).
  The reference's Vulkan MTP-off server returned 192 copies of token 248319 with null log-probabilities ([failure record](benchmarks/qwen38-mtp-natural-reference-failure-20261019.json)); the acceptance comparison of section 6 uses its ROCm build instead.

Released-sidecar accuracy against HF, broader split and serving modes and MTP performance remain open.

## 7. Open items

- Decode speed on split placements, the open side of the speed gate (section 6, Speed gate): H2 and H3 on the host, K1 to K4 in the kernels, which reach parity at best, then the sums' floor of a staged split and the work every group member repeats.
- Follow-ons outside this feature: released tensor-group and serving qualification beyond the recorded cells, other precision modes, multimodal input and selective expert streaming.

## History (2026-10-14 to 2026-10-20)

Dated summary of the development record; the retained records and their hashes are the evidence.

- **2026-10-14 to 2026-10-15:** support started and the design pinned (see also the [runtime patterns review](research/qwen38-runtime-patterns-20261014.md)).
  The design was written as a plan of seven steps: reference and schema, module and CPU fixtures, a Vulkan vertical slice, host-backed weights, splits and serving, the released-file gate, and measured follow-ons such as selective expert transfer or hot-row retention.
  It took the lookup storage's shape from the table's size and checked it against DeepSeek's [Engram at revision 517ef625df97ec57aadc91b67506a57c20fdc5bb](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/517ef625df97ec57aadc91b67506a57c20fdc5bb/inference/engram.py), whose token mapping differs, so address generation stays in the architecture, and vLLM's [Engram path at 4f49b55c1f753183cec09f0a32f5e3a84467e1ec](https://github.com/vllm-project/vllm/blob/4f49b55c1f753183cec09f0a32f5e3a84467e1ec/docs/features/engram.md), which keeps tables resident on the GPU or pinned on the host and has no bounded demand reads from disk.
  Selective expert streaming, a separate hot-row device cache and asynchronous prefetch were left until profiles show they pay; none is built.
  The tiny reference had selected HF's default SiLU GDN output gate; the released contract is sigmoid, and the first incoherent released smoke exposed it.
  After the correction the tiny graph passed and the first released Q8_0 smoke on eight MI50s began "Paris.
  Paris is a city.", with explicit F32 and automatic precision giving the same eight IDs.
  The Radeon VII full-graph result of the SiLU fixture is no evidence for the sigmoid graph.
  The first 13-token released Q8_0 file-exact comparison, llmx on eight MI50s against original HF on CPU over all 248320 logits, gave a maximum absolute error of `1.660766601574437e-5`, top-1 on all 13 positions and exact top-5 overlap on every row, for that input only ([record](benchmarks/qwen38-file-reference-20261015.json)).
  The public loader's 12 combinations of load mode and the lookup residencies of the time passed on Windows CPU and Radeon VII Vulkan and Linux CPU and MI50 Vulkan ([record](benchmarks/qwen38-loader-20261015.json)).
- **2026-10-16:** the tiny-fixture limits of 2e-5 and 1e-5 had been applied to the released model and called released gates; that was corrected.
  A 2,057-token released comparison matched all 33 tail top-1 choices with a minimum top-5 overlap of 4/5; original HF itself changed its result when the same input ran as cached chunks, and the first differing expert set sat across a router gap of `4.76837e-7` (routing record SHA-256 `f0512bddbb0144ca23d1c867c1aa37c0036d417d41c552b818a9f43fda4a4372`).
  The eight corpus cells then measured absolute mean-NLL differences against HF eager of `9.2452e-5`/`2.263e-7` continuous at 512, `4.6822e-4`/`7.7334e-4` continuous at 4096, `2.9833e-5`/`1.597e-7` windowed at 512 and `4.8055e-4`/`9.5596e-4` windowed at 4096 (batched/per token), on that day's runtime (corpus record SHA-256 `20cacd689232fa723846dd5f7f35fef7a091bdf5cef5d5ebbeeeff6033fe5892`).
  Forced direct loading had bypassed the file cache for later lookup reads; normal auto loading keeps them buffered.
- **2026-10-17:** `fill_tables` now forms each rotary frequency as a positive power and takes its reciprocal, as pinned HF does.
  The 16K comparisons then gave 499/513 tail top-1 on the raw input and 483/513 on the chat input (chat record SHA-256 `2ea1924200fc7986aca0e2301d19b821608f07e6fb8606e01bb1e69770f28a59`).
  The contract of that day required every one of the last 512 predictions of each 16K input to be HF's top-1 or its second choice within 0.1 (512/512); the chat input gave 492/512 (482 exact, 10 accepted ties), which failed that rule, and the 2026-10-19 contract replaced it.
  Diagnostics over routing, QSA selection, GDN decay and recurrence, hyper-connection accounting, attention and expert projections found local numerical differences and no semantic defect; a private decay candidate scored 488/512 and was not adopted.
  On the same input HF's own sequential GDN scored 489/512 against its chunk-64 GDN (sequential record SHA-256 `ef61a8edbbf75f264be68be02f72f6b3ffef4127b8acfe9e4a515ba1fef9fe61`), and on a held-out chat input 501/512 (held-out record SHA-256 `bb2563c038f0df31baaf72f8c239849083129b8e067813f87a1b9f07bf45e176`), which led to the reference-calibrated depth limits of section 6.
- **2026-10-18:** MTP was implemented through the drafter hooks with the pairing adapter, and the tiny CPU and Vulkan matrices passed; the released fit, off/on smoke and two-request history passed (header inventory SHA-256 `e03be7161590c83ef9e4e96bafeb11f7824d7c09bccee79f879f2388ee3dc8ad`, server-device SHA-256 `b8c27372035c4b310b1834bba051d2972c707fe3d1a3bfb62c9c02eca93ad1af`).
- **2026-10-19:** a failing test showed the streamed disk restore writing 8352 of a saved image's 12744 bytes, because its destination walk missed the Flash history's parts.
  The fix has the whole-host and streamed images share `detail::block_runs` and `detail::state_runs`, covering auxiliary KV, complete per-layer state, the MTP carry and the host token history; KV destinations resolve their buffers after growth, host copies keep no ring pointer, and device tickets hold ring slots until they retire.
  The regression runs on its selected backend at the whole block extent (CPU 128, Vulkan 64) and compares uninterrupted history, whole-host restore, 1 KiB chunk copies and a disk store's streamed read, follow-up and draft logits bit for bit ([record](benchmarks/qwen38-streamed-restore-20261019.json)).
  The same day the released-file contract was agreed and qualified (section 6), and the MTP draft-state, rollback and natural-stopping controls ran (MTP development checkpoint, above).
  The first speed cells, P512/G128 on six MI50s against the Vulkan build of mx-llama.cpp, measured llmx at 322.12/27.163 prompt/decode tokens/s against 232.20/17.643 on a layer split ([record](benchmarks/qwen38-layer-performance-20261019.json)), while the reference runtime's staged run failed with a nonfinite value (record SHA-256 `122aac6c24e7d14f31c76ee35390f686bb0e883c19339968e41ff698839c68bb`); the mode-by-mode speed gate of section 6 replaced them.
- **2026-10-20:** `--lookup-storage` was reduced to `auto`, `device` and `file`: the host copy of a table and its `host` value were removed, and `disk` became `file`, read through the file's mapping in `--load-mode auto` and `mapped` and through direct reads in `direct`.
  The plan's step table was retired with the sections it planned, which now describe the built paths; what stays open is in sections 5 and 7.
