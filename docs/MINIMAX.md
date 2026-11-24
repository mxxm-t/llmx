# MiniMax-M3.1: the minimax-m3 architecture

Design for MiniMax-M3.1-Flash, written before the code (`docs/ADDING-AN-ARCHITECTURE.md`, The order of the work, step 1).
The steps, their gates and the decisions are in [STATUS](STATUS.md), in the block "MiniMax-M3.1".
The model runs on the CPU, held to HF on tiny fixtures (Classes and checks, below), where `generate`, `logits` and `serve` run those fixtures.
A real file is refused at load until step 2, since a pretokenizer other than `qwen2`, `qwen35` and `joyai-llm` is refused and the file's is `minimax-m2` ([USAGE](USAGE.md)); the fixtures name none.
A device refuses the model at load by op, against the device each part is placed on: of the ops the plan names, `sigmoid_routing`, which every routed layer's feed-forward part names, is the one Vulkan, ROCm and CUDA lack (`mixed_experts` too, where an expert's gate and up stacks differ in storage type), so the refusal comes from the routed layers' feed-forward part, and a part placed on the CPU (the experts' offload flags) does not meet it.
With `--cpu-moe` the attention runs on Vulkan and ROCm and the experts on the CPU, held to HF on the same fixtures (`tests/minimax.py`); Vulkan takes its five attention input products as calls of three.
A tensor width above 1 is refused, as the module declares no role that splits.

Sources, read on 2026-11-15:

- the minimax_m3_vl modeling file of transformers 5.23.0 (SHA-256 52b4860c...), the text path of `MiniMaxM3VLForCausalLM`;
- the model's `config.json`, `tokenizer.json` and `chat_template.jinja` (`MiniMaxAI/MiniMax-M3.1-Flash`);
- the GGUF headers of the seven shards of `unsloth/MiniMax-M3.1-Flash-GGUF` UD-Q8_K_XL, and a few small F32 tensors;
- the checkpoint's norm tensors of layer 0 and the final norm, by range request;
- the converter and graph of llama.cpp pull request 31192 at 6b95a9908, the reference that wrote the file.

## The model

Every GGUF key is prefixed by `minimax-m3.`.

| symbol | meaning | value | GGUF key |
|---|---|---|---|
| E | residual width | 6144 | `embedding_length` |
| L | decoder layers | 60 | `block_count` |
| Ld | leading dense layers | 3 | `leading_dense_block_count` |
| Hq, Hkv | query and KV heads | 64, 4 | `attention.head_count`, `attention.head_count_kv` |
| D | head width | 128 | `attention.key_length`, `attention.value_length` |
| R | rotary width (NeoX halves) | 64 | `rope.dimension_count` |
| theta | RoPE base | 1e7 | `rope.freq_base` |
| Hi, Di | index heads (one per KV head) and index width | 4, 128 | `attention.indexer.head_count`, `attention.indexer.key_length` |
| B | positions per block, a power of two (llmx's limit: the devices split a position by shift and mask) | 128 | `attention.indexer.block_size` |
| K | blocks a query keeps, its own included | 16 | `attention.indexer.top_k` |
| Kl | local blocks pinned | 1 | `attention.indexer.local_blocks` |
| F | dense FFN width | 12288 | `feed_forward_length` |
| X, k | experts, experts per token | 128, 4 | `expert_count`, `expert_used_count` |
| Fe | routed and shared expert width | 3072 | `expert_feed_forward_length` |
| S | shared experts | 1 | `expert_shared_count` |
| V | vocabulary | 200064 | `tokenizer.ggml.tokens` |

The router's gating function is sigmoid (`expert_gating_func` 2), the chosen weights are normalized (`expert_weights_norm`) and scaled by 2.0 (`expert_weights_scale`).
RMS epsilon is 1e-6, the context 1048576 positions.
About 428B parameters, 23B active a token.
The GLU's constants, alpha 1.702 and limit 7, are not in the GGUF; the reader holds them as the architecture's.

## The forward pass

Every layer is one attention block and one feed-forward block, each pre-normed, with the residual added after each:

- **Norms.** All norms are RMSNorm in F32 with weight `w`, the checkpoint's Gemma `1 + w` folded in by the converter.
  Checked against the checkpoint: the final norm reads 1.47656 there and 2.47656 in the GGUF, layer 0's input norm -0.96094 and 0.03906, its q norm 0.11133 and 1.11133, its index q norm -0.09424 and 0.90576.
  HF adds the 1 in F32 to a BF16 value, which is exact, so the stored weight is HF's scale and the module reads a plain RMSNorm.
- **Attention.** q = x Wq (Hq x D), k = x Wk, v = x Wv (Hkv x D); q and k get a per-head RMSNorm over D with one D-wide weight, then NeoX RoPE on their first R values.
  Scores are q . k / sqrt(D), causal, GQA with Hq / Hkv query heads per KV head; output through Wo (Hq D x E).
  No output gate, no biases.
- **Sparse selection (MSA), on every layer.** The index query qi = x Wiq (Hi x Di) and the index key ki = x Wik (one head of Di) get their own RMSNorm and the same partial RoPE.
  ki is kept per position beside K and V.
  For a query at position p and index head h, a position t <= p scores qi[h] . ki[t] in F32, unscaled, with no ReLU and no head weights; later positions score -inf.
  A block's score is the maximum over its B positions, the query's own block p / B is set to +inf, and the K best blocks are kept, ties by lower index (HF's top-k leaves ties unordered), -inf blocks dropped.
  K counts the pinned block, so a query keeps every block while p / B < K: below position 2048 the attention is dense.
  Index head h serves KV head h: the Hq / Hkv query heads of KV group h attend only to the positions of h's kept blocks, still causally inside the newest one.
- **Feed-forward.** Layers below Ld are dense: down((clamp(u) + 1) * glu(g)), with g = x Wg and u = x Wu over F.
  The rest are routed: sigmoid router scores s = sigmoid(x Wr); the k experts with the largest s + b (b = `exp_probs_b`) are chosen, weighted by their s normalized to sum to one, times 2.0; plus one shared expert of the same form, unweighted.
  The GLU form: g' = min(g, 7), u' = clamp(u, -7, 7), out = (u' + 1) * g' * sigmoid(1.702 g').
  In HF the factor 2.0 multiplies the summed expert output, which equals scaling each weight by 2.0 bit for bit, a power of two.
- **Head.** Final RMSNorm, then an untied BF16 head over V.

## The file

UD-Q8_K_XL, 241.3 GiB in seven shards, 1359 tensors:

- routed experts NVFP4, `[E, Fe, X]` gate and up and `[Fe, E, X]` down, each with a per-expert F32 `.scale` companion (`[X]`), which the loader already pairs (`src/format/weight_scales.hpp`);
- attention, dense and shared-expert matrices, embedding and head BF16; router `ffn_gate_inp` `[E, X]` and bias `exp_probs_b.bias` `[X]` F32;
- index projections F32 (`indexer.q_proj` `[E, Hi Di]`, `indexer.k_proj` `[E, Di]`) and every norm F32.

The checkpoint stores dense matrices as MXFP8 with E8M0 block scales and the experts as NVFP4 with a global scale.
MXFP8 values and power-of-two scales are BF16 values, and the converter keeps NVFP4 as stored, so the file holds the checkpoint's weights exactly: an HF reference built from the GGUF's weights is file-exact.

The tokenizer is byte-level BPE with pretokenizer `minimax-m2` and an NFC normalizer (`tokenizer.ggml.normalizer.nfc`); BOS 200019, EOS 200020.

## What a layer keeps

Per position and layer: K and V, Hkv x D each, and the index key ki, Di F32 values (`LayerPlan::kv_aux`).
At f16 K and V that is 2 KiB plus 0.5 KiB a position a layer, 150 KiB a position over 60 layers.

## The ops

The module is `src/model/arch/minimax_m3.hpp`, reached through the registry name `minimax-m3`; it reads the configuration, plans the file's tensors and runs the forward pass above with the ops below.
Each op change goes under its existing owner (F2DEV's reviews of 2026-11-14 22:45 and 2026-11-15 02:46), and each new form is an `Op` mode entry the plan names, so a backend lacking it refuses the model at load:

- Done in step 4a: `RouteRule::Score::sigmoid` beside softmax and sqrt_softplus, the mode `sigmoid_routing`.
- Done in step 4b: the GLU form with alpha and the +1 on the up half, `GluForm::quick_gelu` beside the clamp limit on `silu_mul`, `MoeStep` and `MoeShared`, the mode `glu_quick_gelu`, so existing clamped-GLU callers keep their code path.
- Done in step 4c: `indexer_scores` gains a raw per-head form, `IndexForm::raw_blocks` and the mode `raw_index_scores`, over the per-position index keys `kv_write` stores, a loop body of its own beside deepseek41's, writing block maxima, rows x heads x blocks, so no rows x positions score matrix is stored; blocks past a row's length score -inf.
- Done in step 4d: `candidate_blocks` runs over those maxima unchanged, block size 1, lengths in blocks, and every layer's mixer names it.
- Done in step 4e: `sparse_attention` is extended in place with a group stride, so each KV head reads its own ids, and a block size, so an id names B positions; groups 1 and block 1 are today's chain, held by qwen4exp's byte identity; the mode is `sparse_attention_blocks`.

## Classes and checks

- Below position K B every selection keeps every block, so short checks never reach the sparse path: the tiny fixture uses B 4 and K 3, and the real-file checks include the raw and the chat-rendered 16k sequences and decode after them.
- The tiny fixtures (`tests/fixture_minimax.py`, their goldens from HF's own model through `tools/minimax_reference.py`, docs/ASSETS.md) are `small`, 3 layers at B 4 and K 3 over 64 tokens, first dropping a block at position 12, and `wide`, 2 layers at the file's head, rotary and index widths (D 128, R 64, Di 128) and block (B 128, K 2) over 272 tokens, first dropping at 256.
  The suite's `minimax` component holds llmx on the CPU to 2e-5 a logit and 1e-5 a target's NLL, the F32 class's bounds, since the fixtures are F32 with F32 caches; HF's rows past the first drop sit 0.92 and 0.10 from dense attention's on the same weights, so rows within the bound took the selection.
- The router's biases sit near 20, where F32 steps by 1.9e-6, so two experts' biased scores round to one value at the cut, and HF's top-k takes either at such a tie; llmx's picks there are its own lowest id among its own scores, which differ from HF's by up to 49 ulps of the k-th score over the first 64 raw positions (2.6e-4 a router logit).
  The file-exact reference therefore takes llmx's pick wherever every expert the two picks differ in scores within 98 ulps of HF's k-th score, and reports each such pick; a pick differing beyond it fails the row (docs/ASSETS.md, The minimax-m3 file-exact reference).
  The margin is a frozen budget: the distance, measured on HF's own scores, within which HF's F32 cannot order the experts, sized from raw[:64] at twice its largest deviation so that a tie llmx resolves by an ulp of its own never fails, and fixed before the full rerun.
  It is sound at any llmx deviation, since a pick taken from llmx must still sit within the margin on HF's scores, so a larger deviation can only fail a row, never pass one; a differing pick between 98 ulps and a row's measured deviation is a question for the F32 class, never a reason to widen the margin after the fact.
- Scores are compared in F32 and a flipped block pick is not rounding: the layered real-file check reports per layer how many picks agree with HF beside the logit bound.
