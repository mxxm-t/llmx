# Qwen 3.5, 3.6 and 3.8: the qwen35 and qwen35moe architectures

Design for the next architectures of ROADMAP #2, written before any code and open for review.
This page holds what the code implements: the models and their shapes, the forward pass of each layer kind, the conventions of the GGUF files the kernels read, the files on hand and how each is handled, and the row classes that keep the recurrent layers exact.
The plan, its branches and gates, the decisions taken and the questions still open are in [STATUS](STATUS.md), in the block "Qwen 3.5, 3.6 and 3.8".
The full-attention layers use the paged KV cache of [KV-CACHE](KV-CACHE.md), and the recurrent state is the fixed-size, private, per-sequence state that [EXECUTION](EXECUTION.md), "Beyond dense Qwen", leaves room for.

Sources, read on 2026-09-25:

- the `qwen3_5` and `qwen3_5_moe` modeling code of transformers 5.17.0 and of its main branch;
- llama.cpp's converter and `qwen35` graph at 4b1a27fa, and master's converter, `qwen35.cpp` and Vulkan gated delta-net shader;
- the GGUF headers of every qwen35, qwen35moe and dflash file on the Linux MI50 machine, plus a few small F32 tensors;
- the HF checkpoints: the layer-0 tensors by range request, and every `config.json`.

## The models

Symbols used on this page, with the GGUF keys they come from.
Every key is prefixed by the architecture name, `qwen35.` or `qwen35moe.`.

| symbol | meaning | GGUF key |
|---|---|---|
| E | residual width | `embedding_length` |
| L | decoder layers | `block_count` minus `nextn_predict_layers` |
| Hq, Hkv | query and KV heads of the full-attention layers | `attention.head_count`, `attention.head_count_kv` |
| D | full-attention head width, 256 | `attention.key_length`, `attention.value_length` |
| R | rotary width, 64 | `rope.dimension_count` |
| Hk | linear-attention K heads, 16 in every file | `ssm.group_count` |
| Hv | linear-attention V heads, a multiple of Hk | `ssm.time_step_rank` |
| Dk | K head width, and rows of the state, 128 | `ssm.state_size` |
| Dv | V head width, and columns of the state, 128 | `ssm.inner_size` / Hv |
| C | conv channels, 2 Hk Dk + Hv Dv | none, derived |
| F | dense FFN width | `feed_forward_length` |
| X, K | experts, and experts per token | `expert_count`, `expert_used_count` |
| Fe, Fs | routed and shared expert widths | `expert_feed_forward_length`, `expert_shared_feed_forward_length` |
| V | vocabulary, 248320 | `tokenizer.ggml.tokens` |

| model | arch | layers (full attention) | E | Hq / Hkv | Hk / Hv | FFN | head | state per sequence, F32 | KV per token, f16 |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B | qwen35 | 24 (6: 3, 7, ..., 23) | 1024 | 8 / 2 | 16 / 16 | 3584 | tied, Q6_K | 19.3 MiB | 12 KiB |
| Qwen3.5-2B | qwen35 | 24 (6) | 2048 | 8 / 2 | 16 / 16 | 6144 | tied | 19.3 MiB | 12 KiB |
| Qwen3.5-4B | qwen35 | 32 (8) | 2560 | 16 / 4 | 16 / 32 | 9216 | tied, Q6_K | 50.3 MiB | 32 KiB |
| Qwen3.5-9B | qwen35 | 32 (8) | 4096 | 16 / 4 | 16 / 32 | 12288 | own, Q6_K | 50.3 MiB | 32 KiB |
| Qwen3.6-27B, Qwen3.8-27B | qwen35 | 64 (16), plus one MTP block in the files that carry it (`block_count` 65, `nextn_predict_layers` 1) | 5120 | 24 / 4 | 16 / 48 | 17408 | own | 149.6 MiB | 64 KiB, plus 4 KiB for the MTP layer |
| Qwen3.6-35B-A3B | qwen35moe | 40 (10), plus an MoE MTP block in the files that carry it | 2048 | 16 / 2 | 16 / 32 | 256 experts of 512, top 8, and a shared expert of 512 | own, Q6_K | 62.8 MiB | 20 KiB |
| Qwen3.5-122B-A10B | qwen35moe | 48 (12) | 3072 | 32 / 2 | 16 / 64 | 256 experts of 1024, top 8, and a shared expert of 1024 | own, Q8_0 | 149.1 MiB | 24 KiB |

- The state per sequence is, over the linear-attention layers, Hv Dk Dv floats of recurrent state plus 3 C floats of conv rows, all F32.
- The KV per token is, over the full-attention layers, 2 Hkv D values of 2 bytes.
- Layer l, counted from 0, is a full-attention layer when (l + 1) mod `full_attention_interval` is 0, and a linear-attention layer otherwise.
  Every file here has an interval of 4, so layers 3, 7, 11 and so on are full attention, and three layers in four are linear.
- No file here has the per-layer `attention.recurrent_layers` array.
  llama.cpp's current converter writes one with `block_count` entries, the MTP block's included and false, and its graph reads it at that length.
  The config reads the array when a file has it, and refuses it unless it has `block_count` entries with every MTP entry false.
  The resolver refuses a layer whose tensors disagree with its kind.

Common to every qwen35 and qwen35moe file here:

- RMS epsilon 1e-6.
- Rope base 1e7, rotating R = 64 of the 256 head dims, with `rope.dimension_sections` [11, 11, 10, 0].
- Linear attention: conv kernel 4, state size 128, 16 K heads and `full_attention_interval` 4.
- Context 262144, with no rope scaling.
- Tokenizer `gpt2` with pre `qwen35` and 248320 tokens.
  No BOS is added, and EOS is 248046.
  The GGUF lists 7 control tokens (248070 to 248076, among them `<|audio_start|>`) that the checkpoint's `tokenizer.json` lacks.
  transformers adds them from `tokenizer_config.json`, so it encodes each as one id, as llmx does; only the `tokenizers` library reading `tokenizer.json` alone splits their text into several.

## The forward pass

Notation, for one token's row:

- RMSNorm(x; w) = x / sqrt(mean(x^2) + 1e-6) * w, elementwise, with w the GGUF tensor as stored.
  The stored norm weights already hold HF's 1 + w (GGUF conventions, below), so this is llmx's existing RMS norm.
- W x is the product of GGUF matrix W with the row x, as llmx's matmul computes it: W's first dimension is the input width.
- silu(x) = x * sigmoid(x), sigmoid(x) = 1 / (1 + exp(-x)), and softplus(x) = log(1 + exp(x)), taken as x above 20.
- `*` between rows is elementwise, and a row index [a, b) is the half-open range.

The model:

```
x = token_embd[id]                                  no scaling, and no BOS
for each layer l < L:
    x = x + Mixer_l(RMSNorm(x; blk.l.attn_norm))
    x = x + FFN_l(RMSNorm(x; blk.l.post_attention_norm))
logits = head RMSNorm(x; output_norm)               head = output, or token_embd when tied
```

Mixer_l is the linear attention or the gated attention below, by the layer's kind.
FFN_l is the dense FFN on qwen35 and the MoE FFN on qwen35moe.

### Linear attention: the gated delta net

With a the normed row of the token at position t:

1. **Projections.**
   p = `attn_qkv` a, C values, split as [q: Hk Dk | k: Hk Dk | v: Hv Dv].
   z = `attn_gate` a, Hv Dv values.
   b = `ssm_beta` a and alpha = `ssm_alpha` a, Hv values each.
2. **Conv.**
   A causal depthwise conv of width 4 over p, per channel c: u_t[c] = silu(w[0, c] p_{t-3}[c] + w[1, c] p_{t-2}[c] + w[2, c] p_{t-1}[c] + w[3, c] p_t[c]), with w = `ssm_conv1d`.
   Tap 3 multiplies the current token, and rows before the sequence's start are zero.
   The three earlier raw rows p_{t-3}, p_{t-2} and p_{t-1} are the conv part of the state.
3. **Normed q and k.**
   Split u as q, k and v, the same way as p.
   For K head h: qn_h = q_h / sqrt(sum(q_h^2) + 1e-6) / sqrt(128), and kn_h = k_h / sqrt(sum(k_h^2) + 1e-6).
4. **Gates.**
   For V head j: beta_j = sigmoid(b_j), and g_j = `ssm_a`[j] * softplus(alpha_j + `ssm_dt.bias`[j]).
   The decay factor is exp(g_j), set to 0 when it is below 2^-126, on every backend.
5. **Recurrence.**
   For V head j, with kappa = j mod Hk and the state S_j a Dk x Dv matrix, zero at the sequence's start:

   ```
   S = exp(g_j) S                 decay first
   m = S^T kn_kappa               what the state recalls for this key, Dv values
   d = beta_j (v_j - m)           the correction
   S = S + kn_kappa d^T           a rank-1 write
   o_j = S^T qn_kappa             the read, from the updated state
   ```

6. **Output.**
   y_j = RMSNorm(o_j; `ssm_norm`) * silu(z_j), per V head, with the one Dv-wide `ssm_norm` shared by every head.
   The mixer's output is `ssm_out` y.

The order is HF's: the decay first, then m from the decayed state.

**The chunked form.**
Over a chunk of rows, the same recurrence can be written with matrix products.
Per V head, let G_i be the sum of g over the chunk's rows up to row i, which is never positive.
The chunk's delta-rule updates form a unit lower-triangular system (the WY or UT form), solved in F32; the chunk's outputs are its causally masked intra-chunk attention plus a read of the state from the chunk's start, and the state after the chunk is one more product.
Every decay factor it needs is exp(G_i - G_j) with i >= j, exp(G_i) or exp(G_last - G_i), each at most 1.
A kernel must never form exp(G_i - G_j) for i < j: over a 64-row chunk the cumulative log-decay reaches the thousands on these files, and that factor overflows.
On cards without matrix units the chunked form takes about 1.7 times the recurrence's arithmetic, and its sequential chain is one step per chunk instead of one per token.
HF's full-sequence forward runs this form at a chunk of 64, and its cached single-token steps run the recurrence.

**Decay range.**
`ssm_a` reaches -139 on the 27B, and `ssm_dt.bias` reaches 19.25.
With alpha = 0, the per-token log-decay on the 35B-A3B goes as low as -91.6.
In F32 that decay factor is already denormal, and a GPU that flushes denormals makes it zero, which is why the flush below 2^-126 is defined once and applied everywhere.

### Gated attention, every fourth layer

With a the normed row of the token at position pos:

1. r = `attn_q` a, Hq 2D values.
   For query head h, q_h = r[2Dh, 2Dh + D) and gate_h = r[2Dh + D, 2Dh + 2D).
2. k = `attn_k` a and v = `attn_v` a, Hkv D values each.
3. q_h = RMSNorm(q_h; `attn_q_norm`) and k_h = RMSNorm(k_h; `attn_k_norm`), over all 256 dims of each head.
4. NeoX rope on dims 0 to 63 only: for i from 0 to 31, the pair (i, i + 32) rotates by the angle pos * 1e7^(-2i/64).
   Dims 64 to 255 pass unchanged.
   The sections [11, 11, 10, 0] give each frequency to one of three position streams (time, height and width); for text the three streams hold the same position, so the sections reduce exactly to this rope.
5. Causal softmax attention at scale 1/sqrt(256) = 1/16, with query head h reading KV head floor(h / (Hq / Hkv)), as on Qwen3.
   On the 27B each KV head serves 6 query heads.
6. The mixer's output is `attn_output` (o * sigmoid(gate)), where o is the attention output and the gate applies per element of each head's 256 values.

The KV cache holds k after its norm and rope, and v as projected, as on Qwen3.

### The dense FFN

It is Qwen3's: with f the normed row, the output is `ffn_down` (silu(`ffn_gate` f) * `ffn_up` f).

### The MoE FFN, with the shared expert and its gate

With f the normed row:

1. **Routing.**
   s = `ffn_gate_inp` f, X router scores.
   A softmax over the X scores in F32, the top K, and the K probabilities renormalized to sum to 1, giving weights w_1 to w_K for experts e_1 to e_K.
   This is llmx's existing `route_experts`.
2. **Routed sum.**
   routed = sum over k of w_k `ffn_down_exps`[e_k] (silu(`ffn_gate_exps`[e_k] f) * `ffn_up_exps`[e_k] f), as on qwen3moe.
3. **Shared expert.**
   sg = sigmoid(`ffn_gate_inp_shexp` . f), one scalar per row, the dot product of the E-wide gate vector with f.
   shared = `ffn_down_shexp` (sg * silu(`ffn_gate_shexp` f) * `ffn_up_shexp` f).
4. The FFN's output is routed + shared.

HF scales the shared expert's output by sg.
llmx scales the down projection's input instead, and the down projection accumulates into the routed sum through the existing `matmul_add`.
The projection is linear, so this is HF's product in a different order, within the HF bounds.

### The MTP block

Files with `nextn_predict_layers` 1 carry one more block, stored as `blk.L`.
It proposes the token after next: at position i it reads the main model's row h_i and the embedding of the token t_{i+1}.

1. u = `nextn.eh_proj` [RMSNorm(`token_embd`[t_{i+1}]; `nextn.enorm`), RMSNorm(h_i; `nextn.hnorm`)], the two E-wide rows concatenated in that order.
   h_i is the main model's row after `output_norm`, which llama.cpp's graph also feeds it (`t_h_nextn`, taken after `output_norm` in `qwen35.cpp`).
2. One gated attention layer and its FFN run over u, with a decoder layer's residual structure and the tensors under `blk.L` (`attn_norm`, `attn_q`, `attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output`, `post_attention_norm` and the FFN's).
   It runs at rotary position i.
   Its K and V are one more attention layer in the target's KV storage, with the row layout that STATUS gives.
   On qwen35moe its FFN is the MoE FFN above.
3. With x the block's output row, the draft logits are `output` RMSNorm(x; `nextn.shared_head_norm`), with the main model's head, or `token_embd` when tied.

For a second draft step the block reads its own row after `nextn.shared_head_norm` as h, which is the row llama.cpp's graph exports for it.
The acceptance-rate gate of the MTP step checks that choice at a draft length above 1, since a draft of length 1 never reads it.
The block has no embedding or head of its own: no file here has `nextn.embed_tokens` or `nextn.shared_head_head`.
HF drops `mtp.*` at load, so HF gives no reference for this block.

## The recurrent state

Each linear-attention layer holds, per sequence:

- S: Hv x Dk x Dv floats, 128 x 128 per V head, in F32;
- the conv rows: the 3 C raw projection rows before the next token, in F32.

The state is F32 only, with no flag, on every backend.
A carried state is stored at slice boundaries, and those depend on the batch, so a narrower type would round at points the batch chooses.
A decode step reads and writes S, 144 MiB each way on the 27B, so 288 MiB of traffic a sequence.
A recurrent state exists only at the end of what it has read, so a sequence cannot be forked or truncated at an arbitrary position; where it can be resumed is the checkpoint design in STATUS.

## GGUF conventions

Weights load as the file stores them, with no byte transform, so the loader keeps streaming in file order.
The kernels read the conventions below.

### Tensors

Shapes are in GGUF order, input width first, as `llmx info` prints them.

| tensor | shape | layers | as stored |
|---|---|---|---|
| `token_embd.weight` | [E, V] | | the embedding, and the head when `output.weight` is absent |
| `output_norm.weight` | [E] | | 1 + w |
| `output.weight` | [E, V] | | absent when tied (0.8B, 2B and 4B) |
| `blk.N.attn_norm.weight` | [E] | all | 1 + w |
| `blk.N.post_attention_norm.weight` | [E] | all | 1 + w; the norm before the FFN, which Qwen3 calls `ffn_norm` |
| `blk.N.attn_qkv.weight` | [E, C] | linear | rows [q: Hk Dk, k: Hk Dk, v: Hv Dv]; the v rows in tiled order |
| `blk.N.attn_gate.weight` | [E, Hv Dv] | linear | z; rows in tiled order |
| `blk.N.ssm_alpha.weight` | [E, Hv] | linear | rows in tiled order |
| `blk.N.ssm_beta.weight` | [E, Hv] | linear | rows in tiled order |
| `blk.N.ssm_conv1d.weight` | [4, C] | linear | F32; tap 3 multiplies the current token; the v channels in tiled order |
| `blk.N.ssm_a` | [Hv] | linear | F32, -exp(A_log), in tiled order; the name has no `.weight` |
| `blk.N.ssm_dt.bias` | [Hv] | linear | F32, HF's `dt_bias`, in tiled order |
| `blk.N.ssm_norm.weight` | [Dv] | linear | F32, raw, not 1 + w; shared by every V head |
| `blk.N.ssm_out.weight` | [Hv Dv, E] | linear | input columns in tiled order |
| `blk.N.attn_q.weight` | [E, Hq 2D] | full | each head's rows as [q: D, gate: D] |
| `blk.N.attn_k.weight`, `blk.N.attn_v.weight` | [E, Hkv D] | full | |
| `blk.N.attn_q_norm.weight`, `blk.N.attn_k_norm.weight` | [D] | full | 1 + w |
| `blk.N.attn_output.weight` | [Hq D, E] | full | |
| `blk.N.ffn_gate.weight`, `blk.N.ffn_up.weight` | [E, F] | all, qwen35 | |
| `blk.N.ffn_down.weight` | [F, E] | all, qwen35 | |
| `blk.N.ffn_gate_inp.weight` | [E, X] | all, qwen35moe | the router; F32 in the decoder layers (types, below) |
| `blk.N.ffn_gate_exps.weight`, `blk.N.ffn_up_exps.weight` | [E, Fe, X] | all, qwen35moe | |
| `blk.N.ffn_down_exps.weight` | [Fe, E, X] | all, qwen35moe | |
| `blk.N.ffn_gate_inp_shexp.weight` | [E] | all, qwen35moe | the shared expert's gate, F32 in the decoder layers (types, below); HF's `shared_expert_gate`, [1, E] there |
| `blk.N.ffn_gate_shexp.weight`, `blk.N.ffn_up_shexp.weight` | [E, Fs] | all, qwen35moe | |
| `blk.N.ffn_down_shexp.weight` | [Fs, E] | all, qwen35moe | |
| `blk.L.nextn.eh_proj.weight` | [2E, E] | MTP | HF's `mtp.fc` |
| `blk.L.nextn.enorm.weight`, `blk.L.nextn.hnorm.weight` | [E] | MTP | 1 + w; HF's `mtp.pre_fc_norm_embedding` and `mtp.pre_fc_norm_hidden` |
| `blk.L.nextn.shared_head_norm.weight` | [E] | MTP | 1 + w; HF's `mtp.norm` |

The MTP block's attention, norm and FFN tensors sit under `blk.L` with the names of a full-attention layer.

Types in the files here:

- The norms, `ssm_a`, `ssm_dt.bias` and `ssm_conv1d` are F32, and so are the router and the shared expert's gate of every decoder layer.
  The MTP block of the Qwen3.6-35B-A3B-MTP Q8_0 and UD-Q4_K_XL files stores its router and shared expert's gate in BF16.
- `ssm_alpha` and `ssm_beta` are quantized in most files, although they have only 16 to 64 output rows.
  The Qwen3.6-27B-MTP Q4_1 and Qwen3.6-35B-A3B-MTP UD-Q4_K_XL files store them in F32, and the Qwen3.6-27B BF16, Qwen3.5-35B-A3B-UD-Q8_K_XL and Qwen3.5-122B-A10B-UD-Q8_K_XL files store them in BF16 or F16.
- Mixed files mix types within a layer: in the 27B Q4_K_M, layer 0's `attn_qkv` is Q6_K while its `attn_gate`, `ssm_alpha` and `ssm_beta` are Q4_K, and in the 0.8B `attn_qkv` is Q5_K.

### What the converter folds

Checked against the HF weights:

- Every norm weight except `ssm_norm` is stored as 1 + w.
  HF's norm multiplies by 1 + w, so llmx's RMS norm applies the stored weight unchanged.
  In every file read, the stored value minus 1 is an exact bf16 number.
- `ssm_norm` is stored raw, and equals HF's exactly: HF's gated norm multiplies by w itself.
- `ssm_a` is -exp(A_log), so it is already negative, and `ssm_dt.bias` is HF's `dt_bias`.
- `ssm_conv1d` is HF's conv weight with its middle axis dropped.
- `attn_q` keeps HF's layout, which puts each head's q and gate side by side.
- `mtp.*` becomes `blk.L`, and its fc and its three norms take the `nextn.` names above.
- Not folded, so the kernels apply them: the attention scale 1/16, the L2 epsilon 1e-6, the query scale 1/sqrt(128) and the rope frequencies.

Folding done twice or missed gives wrong numbers that still look like text: 1 added again to the norms, 1 added to `ssm_norm`, or exp applied to `ssm_a`.

### V heads in tiled order

HF stores the linear-attention V heads grouped by K head: with r = Hv / Hk, HF's V head j reads K head floor(j / r).
Since llama.cpp PR 19468 (2026-02-10), the converter stores them in tiled order whenever Hv differs from Hk: GGUF V head j = s Hk + h holds HF's V head h r + s, and reads K head j mod Hk.

- The permutation applies to every tensor indexed by V head: the v rows of `attn_qkv`, the v channels of `ssm_conv1d`, the rows of `attn_gate`, `ssm_alpha` and `ssm_beta`, the entries of `ssm_a` and `ssm_dt.bias`, and the input columns of `ssm_out`.
  The whole V side moves together, so the model computes HF's function once each V head reads K head j mod Hk.
- Checked on the 4B, the 9B, both 27B and the 35B-A3B: `ssm_dt.bias` matches HF under the tiled order with error 0, and misses by 18 to 23 under the grouped order.
- No metadata records which order a file uses, so a file converted before that change cannot be told apart.
  Comparing `ssm_dt.bias` with HF under both orders is the vetting step for any new file with Hv different from Hk.
- The 0.8B and 2B have Hv = Hk, so they cannot reveal a wrong mapping.
  Only a file with Hv above Hk catches it: the tiny fixture with Hv = 3 Hk, the 4B, and the 9B, the 27B and the 35B-A3B through the layered HF reference.
  The tiny fixture's writer applies the tiled order itself, so it cannot catch a misreading that the writer and the kernels share; a hosted check of the writer's permutation against real `dt_bias` values covers that (STATUS).

### Settings the files carry that the math ignores

The 27B configs carry `output_gate_type: "swish"`, `attn_output_gate` and `mtp_use_dedicated_embeddings`.
HF, llama.cpp and vLLM all ignore these and apply a sigmoid gate.
llmx follows HF, and the HF gate checks it.

## The files on the Linux MI50 machine

The step numbers are those of the plan in STATUS.

| files | arch | tensor types | handling |
|---|---|---|---|
| Qwen3.5-0.8B, 2B, 4B and 9B Q4_K_M | qwen35 | K-quants, F32 | load from step 4 |
| Qwen3.6-27B Q4_K_M, Q8_0 and Q5_K_M (the Q5_K_M is a review copy requantized from Q8_0); Qwen3.8-27B Q8_0, which carries an MTP block | qwen35 | Q4_K, Q5_K, Q6_K, Q8_0, F32 | load from step 4, which ignores an MTP block; step 9 uses it |
| Qwen3.6-27B-MTP Q8_0 and Q4_1; Qwen3.6-27B revision 6a9e13bd Q4_K-pure and Q4_K-imatrix | qwen35 with an MTP block | Q4_1, Q4_K, Q5_K, Q6_K, Q8_0; the Q4_1 file stores `ssm_alpha` and `ssm_beta` in F32 | load from step 4, which ignores the MTP block; step 9 uses it |
| Qwen3.6-35B-A3B Q4_K_M, Q5_K_M (plus a requantized review copy), Q6_K and Q8_0 | qwen35moe | K-quants, Q8_0 | load from step 7 |
| Qwen3.6-27B-MXFP4 (all 498 matrices MXFP4); Qwen3.6-35B-A3B-MXFP4 (120 expert tensors MXFP4, the rest Q8_0) | qwen35, qwen35moe | MXFP4 | refused, naming the type, until the quantization plan's MXFP4 branches merge, then step 10; both are requantized from other quants, so they only compare one runtime with another on the same file |
| Qwen3.6-35B-A3B-IQ4_NL (381 IQ4_NL tensors, plus Q5_K and Q6_K) | qwen35moe | IQ4_NL | refused, naming the type, until the quantization plan's IQ4 branches merge, then step 10 |
| Qwen3.6-27B revision 6a9e13bd BF16 (50.9 GiB, with an MTP block); Qwen3.8-27B-UD-Q8_K_XL (BF16 `attn_q`, `attn_k`, `attn_v`, `output` and `eh_proj`); Qwen3.5-35B-A3B-UD-Q8_K_XL (311 BF16 tensors); Qwen3.6-35B-A3B-MTP Q8_0 and UD-Q4_K_XL (BF16 only in the MTP block's router and shared expert's gate; the UD-Q4_K_XL stores `ssm_alpha` and `ssm_beta` in F32); Qwen3.5-122B-A10B-UD-Q8_K_XL (F16, with an F16 `token_embd` and a Q8_0 `output`) | qwen35, qwen35moe | BF16, F16 | refused, naming the type, until the quantization plan's 16-bit branch merges, then step 10 |
| Qwen3.6-27B-Q5_1 (497 Q5_1 tensors) | qwen35 | Q5_1 | refused, naming the type; Q5_0 and Q5_1 are not on the quantization roadmap, and whether they join it is an open question |
| Qwen3.6-27B revision 6a9e13bd Q4_S8X and Q4_H2I2-imatrix | qwen35 | type id 53, which is not a ggml type (experimental repacked quants) | refused as an unknown type |
| Qwen3.6-35B-A3B-DFlash Q8_0 (6 blocks, block size 16, target layers 2, 7, ..., 38, a 4096-token sliding window on 5 of 6 layers); Qwen3.8-27B-DFlash2 Q8_0 (5 blocks, block size 8, non-causal attention, conv kernel 2, a selector of rank 256 at top 16, target layers 6, 20, 34, 48 and 62, a 2048-token window) | dflash | Q8_0 | another architecture: drafters that read the target's hidden rows from chosen layers; DFlash2 is step 5 of the speculative decoding plan in STATUS, and DFlash1 follows only if that plan measures a gain on the MoE target |
| Qwen3.8-Flash-Next | qwen4exp | Q8_0, UD-Q4_K_XL | another architecture: hyper-connections, an indexer and compressed attention, hashed n-gram per-layer embeddings, and 512 experts at top 10; it gets its own plan after DeepSeek V4.1 |

The chat templates in these GGUFs are Unsloth's edits of the official templates.

## Row classes

The recurrence is where a runtime can silently depend on the batch: a carried state is stored wherever a slice ends, and a chunked kernel adds its own grouping.
These rules keep every result batch-invariant, and every reused state one the CLI would have computed the same way.

- **On the CPU**, one arithmetic serves every row: for each (sequence, V head), tokens run in order through the recurrence above.
- **On Vulkan**, the rows of an extent-1 entry run that same per-token recurrence.
  These are the generated rows (decode steps, the exact-resume replay and the verify rows of speculative decoding) and a one-token prompt.
- **The prompt rows of an entry whose extent is above 1** run the chunked form on Vulkan, on a grid of absolute multiples of 64 from position 0, with no other extent threshold.
  This comes with its own step of the plan (step 6), and is kept only if it measures faster; until then these rows run the recurrence too.
- So a row's class is its entry's extent, 1 or above 1.
  The row classes of `fix/server-exact-resume` (`RowClass`: the end, extent and fresh count of each stretch of a history) record it.

Within a class, a row's output and the state after it depend only on the state before the row and the row's own inputs.
So all of these give the same bits: slicing a prompt into passes, the server's budget slices, other sequences in the pass, a reused prefix, and the exact-resume replay of generated rows.

**The chunk grid** comes with the chunked form (step 6); while every row runs the per-token recurrence, no cut is needed.
One owner, the model's prompt cut, ends every slice of a hybrid model's prompt at an absolute multiple of 64, or at the prompt's end.
The CLI's ubatch split, the pipelined chunks of the layer split and the server's batch assembly all use it.
A chunk is partial only at the prompt's first or last fresh row.
A checkpoint position is a multiple of the KV block size, which 64 divides, so it lies on the grid.

**The conv's carried rows** are the raw projection rows, held in F32 like the arena rows the prompt path reads.

**Rules the kernels keep:**

- The layout (lanes per column, reduction trees, column split, chunk size) is a constant per model, never chosen by the token or sequence count.
- No pipeline is specialized on the token count.
  Checkpoint stores, and the source and destination state slots, are push constants of one pipeline.
- Every multiply-add is an explicit fused multiply-add: GLSL `fma()` on the device, and `std::fma` or the intrinsic on the CPU, because the build contracts by default.
- The recurrence, conv, L2-norm and gated-norm accumulators are `precise` (NoContraction), as `matmul_tile_q.comp` already is, since SPIR-V gives no invariance without it.
- exp, softplus and sigmoid are each computed at one site.
  The decay flush below 2^-126 is defined at that one site, not left to each driver's denormal handling.
- The operation order is HF's: the decay first, then m from the decayed state.

**Against HF.**
HF's full-sequence forward uses the chunked form, and its cached steps use the recurrence; the HF gate records how far the two disagree in F32.
The tiny fixtures' goldens come from HF's token-by-token cached forward, whose recurrence matches llmx's per-token arithmetic, with the full forward recorded beside them and its distance.
Both HF paths cast q, k, v, beta and the decay to F32 inside, so HF has no FP64 path for these layers.
