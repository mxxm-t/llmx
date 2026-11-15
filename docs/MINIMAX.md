# MiniMax-M3.1: the minimax-m3 architecture

Design for MiniMax-M3.1-Flash, written before the code (`docs/ADDING-AN-ARCHITECTURE.md`, The order of the work, step 1).
The steps, their gates and the decisions are in [STATUS](STATUS.md), in the block "MiniMax-M3.1".
Nothing of it runs yet; the files and ops it adds are under "The ops (planned)", below.

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
| B | positions per block | 128 | `attention.indexer.block_size` |
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

## The ops (planned)

The module is `src/model/arch/minimax_m3.hpp`, reached through the registry name `minimax-m3`.
Each op change goes under its existing owner (F2DEV's review of 2026-11-14 22:45):

- `RouteRule::Score::sigmoid` beside softmax and sqrt_softplus.
- The GLU form with alpha and the +1 on the up half, as its own form so existing clamped-GLU callers keep their code path.
- `indexer_scores` gains a raw per-head form over per-position auxiliary keys, writing block maxima, rows x heads x blocks, so no rows x positions score matrix is stored.
- `candidate_blocks` runs over those maxima unchanged, block size 1, lengths in blocks.
- `sparse_attention` gains a group stride, so each KV head reads its own ids, and a block granularity, so an id names B positions.

## Classes and checks

- Below position K B every selection keeps every block, so short checks never reach the sparse path: the tiny fixture uses B 4 and K 3, and the real-file checks include the raw and the chat-rendered 16k sequences and decode after them.
- Scores are compared in F32 and a flipped block pick is not rounding: the layered real-file check reports per layer how many picks agree with HF beside the logit bound.
