# `src/model/arch/qwen35.hpp` - the qwen35 and qwen35moe architectures

Attention, recurrent-input, output and shared-expert gate projections forward `Step::dtype`. The shared routed block keeps the router F32 and forwards the policy to the experts. The architecture does not implement dtype selection or conversion; its existing non-matrix F32 operations are unchanged.

Qwen 3.5, 3.6 and 3.8, in namespace `infer::qwen35`: the dense `qwen35`
architecture and its mixture-of-experts form `qwen35moe`, which the
registry ([registry](model-arch-registry.md)) reaches through `open_dense`
and `open_routed`. It implements the architecture contract
([architecture](model-architecture.md)); nothing outside `model/arch/`
names it. The math, the files' conventions and the recurrent state are in
[QWEN35](../QWEN35.md); the plan and its steps are in `docs/STATUS-2026-09.md`.
It runs on the CPU, on a Vulkan device and on a ROCm device, whose backends all implement
the ops each part names (`LayerPlan::ops`), and `llmx serve` reuses a
conversation's prefix through state checkpoints ([server](server.md)).

- `Config`, `read_config(file, prefix, moe)`: the configuration, read
  under the prefix the registry hands it through
  [metadata](model-arch-metadata.md).
  - With `moe`, a qwen35moe file: no `feed_forward_length`, the routing
    keys through `metadata::experts`, and
    `expert_shared_feed_forward_length`, the shared expert's width.
  - The decoder layers are `block_count` less `nextn_predict_layers`
    (absent is 0), whose MTP block the model does not run; more than one
    MTP block, or as many as the blocks, is refused.
  - `embedding_length`, `feed_forward_length`, `attention.head_count`,
    `attention.head_count_kv` (default the query heads),
    `attention.key_length`, which `attention.value_length` must equal,
    `context_length`, `rope.freq_base` (default 10000) and
    `attention.layer_norm_rms_epsilon` (default 1e-6), with a tensor layout
    other than `reference` refused, as Qwen3's reader refuses it; the scaling keys and the request go to
    `rope::resolve` over the rotary width ([rope](model-rope.md)), as in Qwen3.
  - `rope.dimension_count`, the leading dims of each head that rotate
    (default the head width), even and at most the head width. For text
    every section's position stream holds the same position, so the
    sections reduce to the plain rope over those dims; when
    `rope.dimension_sections` holds sections, twice their sum must be that
    width, and an absent key or an empty array is not checked.
    With three sections or more the plan gives `image_streams`: an image row's (start, row, column) interleaved over the rotated pairs as HF's Qwen 3.5 interleaves them, pair j taking the row where j mod 3 is 1 inside three times the second section, the column where it is 2 inside three times the third, and the start elsewhere; the full-attention layers pass a pass's `Step::streams` to `norm_rope_partial`.
  - The linear attention: `ssm.conv_kernel`, which must be
    `backend::kConvTaps` (4), `ssm.state_size` (the K head width),
    `ssm.group_count` (K heads), `ssm.time_step_rank` (V heads) and
    `ssm.inner_size`, which must be a whole number of V heads and divided
    by them gives the V head width.
  - A layer is full attention where the per-block
    `attention.recurrent_layers` array says it is not recurrent, when a
    file has one, which must have `block_count` entries with every MTP
    entry false; otherwise every `full_attention_interval`-th layer is.
  - `check_config`: at least one decoder layer, whole query heads per KV
    head and V heads per K head, a rotary width that is even and inside a
    head, at most 256 experts a token and no more than a layer has,
    projections and a state head (`k_dim * v_dim`) within `int`, and
    context storage fitting float vectors.
- `Qwen35`: the architecture over a `Config`, with its roles as the ids of
  `Role` and its layer kinds `linear` and `full`.
  - `plan(index)`: the embedding gives the vocabulary. Each layer's roles
    are its `attn_norm`, its mixer's, then `post_attention_norm` and the
    dense `ffn_gate`, `ffn_up` and `ffn_down`, or on qwen35moe a routed
    layer's `ffn_gate_inp` router (`[E, X]`), the `ffn_{gate,up,down}_exps`
    stacks, `ffn_gate_inp_shexp`, the shared expert's gate as an F32
    vector of E, and the shared expert's `ffn_{gate,up,down}_shexp`,
    issuing `sigmoid_mul` in the feed-forward part. `blocks::routed_ops` also declares `routed_experts`, and `mixed_experts` when the gate and up stacks differ in storage type, for both layer kinds; the down stack is a separate product. A routed layer run
    beside its mixer (`Placement::stream_from`) copies its norm, router and
    shared expert there and writes its stacks into a window. A full-attention layer
    keeps KV (`Cache::kv`) and reads `attn_q` as `[E, 2 Hq D]`, each
    head's q and gate side by side, `attn_k`, `attn_v`, the per-head
    `attn_q_norm` and `attn_k_norm` and `attn_output`, issuing
    `norm_rope_partial` and `sigmoid_mul`. A linear-attention layer keeps
    a state (`Cache::state`) and reads `attn_qkv` (`[E, C]`, C being
    2 Hk Dk + Hv Dv), `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`
    as a `table` (`[4, C]`, F32), `ssm_a` and `ssm_dt.bias` as F32
    vectors of Hv, `ssm_norm` (Dv) and `ssm_out`, issuing
    `causal_conv_silu`, `gated_delta_rule` and `gated_rms_norm`, and saves
    for a mark the raw qkv rows (slot 2, C wide), alpha and beta (slot 5,
    Hv wide each, beta `rows` rows in). A layer
    holding the other kind's first projection is refused, naming the
    tensor, and so is a router, and blocks past the decoder layers are not
    read; a router in a qwen35 file is refused. The head's `output.weight` takes `token_embd.weight` as its
    alias, the tie. The plan's state is `StateShape{Hk, Hv, Dk, Dv}` and
    the two rope tables hold the covered positions at `rope_dim / 2`
    each.
    Each role declares its split over a tensor group ([shard](model-shard.md)): the head by its vocabulary rows; a full-attention layer's `attn_q` by heads of 2 D rows, a q head with its gate, `attn_k` and `attn_v` by KV heads, replicated where the width is a multiple of them, and `attn_output` by the heads' columns; a linear-attention layer by K heads, `attn_qkv` and the conv's channels as the q and k rows of a member's K heads and, from each of the Hv / Hk tiles of V heads, the V heads that read them, `attn_gate`, `ssm_alpha` and `ssm_beta` by those V heads' rows, `ssm_a` and `ssm_dt` by their elements and `ssm_out` by their columns; the dense block by `blocks::shard_swiglu`, and on qwen35moe a routed layer's expert stacks by `blocks::shard_experts`, each expert's hidden rows, and its shared expert by the same call as one more expert; norms, `ssm_norm`, the embedding, the router and the shared expert's gate vector stay whole.
  - `slot_widths(config)`: the floats one row takes in each of the ten
    arena slots, the widest use either layer kind makes of it: x, h,
    attn_q's rows or the raw qkv rows, k or the conv's output, v or z, the
    normed and rotated q or alpha then beta, the attention's or the
    recurrence's output, and the feed-forward block's gate, up and output,
    as wide as the dense block or k routed expert rows and the shared
    expert's after them, whichever is wider. A qwen35moe plan adds four: the router's scores,
    the expert ids and weights, and the shared expert's gate.
  - `fill_tables`: the rope's cos and sin over the rotated pairs,
    `[pos * (rope_dim / 2) + i]` at frequency `rope_theta^(-2i/rope_dim)`, through `rope::fill`.
  - `embed`, `ffn` and `head` are the shared pieces of
    [blocks](model-arch-blocks.md), the feed-forward block after the
    `post_attention_norm` norm: `swiglu` on qwen35, and on qwen35moe
    `moe_step` with the shared expert and its gate, one backend op whose
    base runs the routed experts, then the shared expert's gate as a
    one-column `matmul` and its SwiGLU scaled by that row gate, each adding
    into the residual. `mixer` norms the residual with
    `attn_norm` and runs the layer's kind:
    - full attention: q with its gates, k and v through one
      `matmul_group`; one `gated_attention_layer`, whose base runs
      `norm_rope_partial` from q's rows between the gates into contiguous
      heads and in place on k, `kv_write` and `gated_attention`, the
      attention gated in place by each head's gate read where `attn_q`
      left it, a device fusing it where it gives the base's bits;
      `attn_output` added into the residual.
    - linear attention: one `Backend::delta_step` with the SiLU gate,
      the qkv, z, alpha and beta products, the conv, the delta rule and
      the gated norm by z in place, a device fusing it where it gives the
      base's bits; `ssm_out` added into the residual.
  - `recur`: a linear-attention layer's state update from its slots, run
    by a retract's rerun:
    `causal_conv_silu` from the raw rows and the state's carried ones into
    the conv's output, and `gated_delta_rule` from the sequence's slot over
    it with alpha and beta into the recurrence's output, V head `j` reading
    K head `j mod Hk`; the mixer runs it, and a retract into a verify runs
    it again over the rows the mark saved.
  - `plan_drafter`: the MTP block after the decoder layers as an embedded
    drafter (docs/SPECULATIVE.md, section 7), planned only when a caller
    asks for one and refused for a file without the block: the target's
    embedding, final norm and head, the block's `nextn` norms and
    `eh_proj`, and its full-attention layer and feed-forward block under
    the decoder layers' names, all `Part::draft`, with the drafter's ops
    and four more arena slots, the first `draft_h`.
    A joined DFlash sidecar (`Config::dflash`, [dflash](model-arch-dflash.md)) drafts in place of the MTP block (`plan_dflash`): `fc` and `enc.output_norm` over the taps, each drafter layer's roles from `dflash_layers` on (`dflash_role`), the code bridge, its final norm and the target's embedding and head, `draft_kv_layers` KV layers, `draft_window` its window and `draft_causal` its attention. A DFlash2 sidecar (`selector_rank` set) takes each layer's convolution projections and bases (`attn_conv_proj`, `attn_conv_base`, `ffn_conv_proj`, `ffn_conv_base`, the bases tables of `[width, taps, sides]`) in place of the bridge's read and write, and the selector's `selector_hidden`, `selector_predecessor` and `selector_successor` in place of the codebook and gates, with `tap_weighted_rows` in place of `unit_rows` and `sigmoid_gate_rows`. Where its K and V heads or width differ from the target's, `draft_kv` gives its layers their own shape in the head's KV storage, and the tables gain the drafter's own rope over its whole head as tables 2 and 3. Each tapped layer's kind carries `writes_tap`, so its mixer first copies the residual entering it into its plane of the taps' slot, which crosses with the residual from the first tap on.
  - `draft_rows`: the block's KV rows for every row of a pass: the final
    norm over every row into `draft_h`, the row before each (an entry's
    carried row for its first, the normed row before it for the rest),
    `blocks::nextn_input`, `attn_norm`, K and V with the k norm and the
    partial rope, written to the block's cache; q, attention, the
    feed-forward block and the head do not run for them.
  - `draft`: a draft row for each drafting sequence: the drafted ids'
    rows through `embed_ids` and the rows before them as the block's
    input, gathered where a step reads carried rows, the block's
    full-attention layer and feed-forward block on its residual,
    `shared_head_norm`, the target's head and `argmax_rows` into the next
    ids.
- `open_dense(file, prefix)`, `open_routed(file, prefix)`: the
  registry's readers of a qwen35 file and a qwen35moe file, each the
  architecture over the configuration `read_config` reads.
- `hf_table()`, `hf_derive(config, kv, prefix)`: the table that reads a dense Qwen 3.5 or 3.6 Hugging Face directory (`Qwen3_5ForConditionalGeneration`, `model_type` `qwen3_5`) as the qwen35 GGUF file the converter writes from it ([hf_config](model-arch-hf_config.md)): the text model's keys from `text_config`, the vision tower dropped, the MTP layer `mtp.*` under `blk.{L}`, and the norms, `ssm_a`, `ssm_conv1d` and the tiled V heads made at load in the forms of [QWEN35](../QWEN35.md), GGUF conventions; `hf_derive` writes the keys no single value gives.
  Qwen 3.6's keys are checked: `output_gate_type` is `swish` only, `partial_rotary_factor` beside `rope_parameters` must equal its factor, and the top level's `language_model_only` is false only; an FP8 directory's V-side linears and their scale grids move in tiled order ([hf_config](model-arch-hf_config.md)).
  The suite's `safetensors` component holds tiny directories, F32 and BF16, to their GGUF twins' outputs byte for byte and to HF.

CTest `arch-qwen35` holds every refused key and tensor to its text, the
plan of each layer kind and the footprint on a tiny model built in memory,
the runtime's rules for a model that keeps a state, and the same model as
qwen35moe; the suite's
`qwen35` component holds the tiny HF fixtures to their goldens, and
`tests/baseline_qwen35.py` and `tests/baseline_layered.py` the real files
to theirs.

## Arena slots

The arena holds, per row of a pass, the widest use either layer kind makes of each slot: 0 x, 1 h, 2 attn_q's rows or the raw qkv rows, 3 k or the conv's output, 4 v or z, 5 the normed and rotated q or alpha then beta, 6 the attention's or the recurrence's output, gated, 7 gate, 8 up, 9 ffn.
On qwen35moe slots 10 to 13 hold the router scores, the expert ids, the expert weights and the shared expert's gate.
With a DFlash drafter three slots follow: the taps, one plane of the pass's rows each, the taps of each row side by side, and `fc`'s normed output.
The block's follow them: its codes (the block's rows and the anchors' after them), the bridge's gate, pred_gate and prev rows, its messages, each row's root mean square, the last layer's rows before its bridge, and the tail's refined rows, scores and candidates; the tail's are `ceil(top_k / block_size)` times as wide, so a step's `R * top_k` rows fit the room of the call's `R * block_size`.
A DFlash2 block's follow the same three instead: the convolution's coefficients and output, the selector's gates, predecessor rows, scores and candidates, and the candidates' lengths.

## A DFlash drafter's context rows

For every row of a pass the drafter gathers the row's taps side by side, runs `fc` and `enc.output_norm` over them, and for each of its layers writes K, normed and rotated over the whole head by the drafter's rope (`Config::draft_rope`, YaRN at the target's factor where `--yarn` asks it of the target), and V into that layer's cache.

## A DSpine block

`dflash_block` follows `tools/dspine_reference.py` (`block`) on the CPU, one call a chain over its sequences' blocks of `block_size` rows (`ModelPlan::draft_anchor` 1, `draft_block` one less): each sequence's last pick at the history's length, then mask rows.
Each layer writes the K and V of all the block's rows before any row attends, through the window, over the context rows and the block's rows up to its own, so a stale row of an earlier block at those positions is never read.
After each layer's feed-forward block the code bridge takes the row's root mean square (`unit_rows`, the reference's `sqrt(max(mean, FLT_MIN))` with eps outside), `dspine_read` and its unit code times sqrt(C); each anchor row's code is the codebook row of its pick times sqrt(C) instead, kept after the block's rows; and row r gets `sigmoid_gate_rows`' sigmoid(gate(code r) + pred_gate(code r-1) + bias) * prev(code r-1) times its root mean square, an anchor row none, through `dspine_write` into its residual.
The target's head over the final-normed rows past the anchor gives each draft's logits, which are what `draft_logits` returns, not the draft's own scores, and `topk_rows` their `top_k` candidates by ascending id.
The tail runs the last layer's bridge once more for draft t's row under each candidate code of the step before (the last pick for the first), `matmul_at_ids` scores each refined row against its candidates' head rows times `output_norm`, and `chain_pick` walks from the first row's best through each step's row of the candidate taken.

## A DFlash2 block

`dflash2_block` follows `tools/dspine_reference.py` (`dflash2_block`) on the CPU, taken in place of `dflash_block` where the sidecar has a `selector_rank`, over the same blocks of the last pick then mask rows.
Its views are non-causal (`ModelPlan::draft_causal` false), so each row attends, through the window counted from its own position, the context rows and every row of its block.
Around each layer's attention and feed-forward a convolution within the block: `attn_conv_proj` or `ffn_conv_proj` over the normed row gives its coefficients, and `tap_weighted_rows` with side 0 of the base feeds q, k and v or the gate and up products, and with side 1 adds the attention or feed-forward output into the residual, row r's tap t reading row r - t of its block and none before its first.
The target's head over the final-normed rows past the anchor gives each draft's logits and `topk_rows` their `top_k` candidates, and `selector_hidden` over the same rows each draft's gate.
Step t takes `selector_predecessor`'s rows of draft t - 1's candidates (the last pick for the first) through `embed_ids`, `matmul_at_ids` scores each under the draft's gate against its candidates' `selector_successor` rows plus each candidate's own logit, and `chain_pick` walks the scores.

## The MTP layer's cache rows

For every row of a pass the block's cache takes the final norm over the row into `draft_h`, the row before it (an entry's carried row for its first row, the normed row before it for the others), the block's input, then attn_norm, K and V, the k norm and the partial rope.
