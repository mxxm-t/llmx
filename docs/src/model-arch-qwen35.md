# `src/model/arch/qwen35.hpp` - the qwen35 and qwen35moe architectures

Qwen 3.5, 3.6 and 3.8, in namespace `infer::qwen35`: the dense `qwen35`
architecture and its mixture-of-experts form `qwen35moe`, which the
registry ([registry](model-arch-registry.md)) reaches through `open_dense`
and `open_routed`. It implements the architecture contract
([architecture](model-architecture.md)); nothing outside `model/arch/`
names it. The math, the files' conventions and the recurrent state are in
[QWEN35](../QWEN35.md); the plan and its steps are in `docs/STATUS.md`.
It runs on the CPU and on a Vulkan device, whose backends both implement
the ops each part names (`LayerPlan::ops`), and `llmx serve` holds its
requests without donors ([server](server.md)).

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
    `attention.layer_norm_rms_epsilon` (default 1e-6), with RoPE scaling
    and a tensor layout other than `reference` refused, as Qwen3's reader
    refuses them.
  - `rope.dimension_count`, the leading dims of each head that rotate
    (default the head width), even and at most the head width. For text
    every section's position stream holds the same position, so the
    sections reduce to the plain rope over those dims; when
    `rope.dimension_sections` holds sections, twice their sum must be that
    width, and an absent key or an empty array is not checked.
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
    issuing `sigmoid_mul` in the feed-forward part. A routed layer run
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
    `causal_conv_silu`, `gated_delta_rule` and `gated_rms_norm`. A layer
    holding the other kind's first projection is refused, naming the
    tensor, and so is a router, and blocks past the decoder layers are not
    read; a router in a qwen35 file is refused. The head's `output.weight` takes `token_embd.weight` as its
    alias, the tie. The plan's state is `StateShape{Hk, Hv, Dk, Dv}` and
    the two rope tables hold the context's positions at `rope_dim / 2`
    each.
  - `slot_widths(config)`: the floats one row takes in each of the ten
    arena slots, the widest use either layer kind makes of it: x, h,
    attn_q's rows or the raw qkv rows, k or the conv's output, v or z, the
    normed and rotated q or alpha then beta, the attention's or the
    recurrence's output, and the feed-forward block's gate, up and output,
    as wide as the dense block, k routed expert rows or the shared expert,
    whichever is widest. A qwen35moe plan adds four: the router's scores,
    the expert ids and weights, and the shared expert's gate.
  - `fill_tables`: the rope's cos and sin over the rotated pairs,
    `[pos * (rope_dim / 2) + i]` at frequency `rope_theta^(-2i/rope_dim)`.
  - `embed`, `ffn` and `head` are the shared pieces of
    [blocks](model-arch-blocks.md), the feed-forward block after the
    `post_attention_norm` norm: `swiglu` on qwen35, and on qwen35moe
    `routed_experts`, then the shared expert's gate as a one-column
    `matmul` and `swiglu` with that row gate, each adding into the
    residual. `mixer` norms the residual with
    `attn_norm` and runs the layer's kind:
    - full attention: q with its gates, k and v through one
      `matmul_group`; `norm_rope_partial` from q's rows between the gates
      into contiguous heads, and in place on k; `kv_write` and
      `attention`; `sigmoid_mul` gating the output in place by each head's
      gate read where `attn_q` left it; `attn_output` added into the
      residual.
    - linear attention: the raw qkv rows through their own `matmul`, then
      z, alpha and beta in one `matmul_group`; `causal_conv_silu` from the rows and the state's
      carried ones; `gated_delta_rule` from the sequence's slot, V head `j`
      reading K head `j mod Hk`; `gated_rms_norm` by z in place; `ssm_out`
      added into the residual.
- `open_dense(file, prefix)`, `open_routed(file, prefix)`: the
  registry's readers of a qwen35 file and a qwen35moe file, each the
  architecture over the configuration `read_config` reads.

CTest `arch-qwen35` holds every refused key and tensor to its text, the
plan of each layer kind and the footprint on a tiny model built in memory,
the runtime's rules for a model that keeps a state, and the same model as
qwen35moe; the suite's
`qwen35` component holds the tiny HF fixtures to their goldens, and
`tests/baseline_qwen35.py` and `tests/baseline_layered.py` the real files
to theirs.
