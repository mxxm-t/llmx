# `src/model/arch/qwen3.hpp` - the Qwen3 architecture

Qwen3's architecture, in namespace `infer::qwen3`: dense Qwen3 and its
mixture-of-experts form, `qwen3moe`, which the registry
([registry](model-arch-registry.md)) reaches through `open_dense` and
`open_routed`. It implements the architecture contract
([architecture](model-architecture.md)); nothing outside `model/arch/`
names it.

- Mixture of experts: a `qwen3moe` file has every key read under the
  `qwen3moe.` prefix, plus `expert_count`, `expert_used_count`,
  `expert_feed_forward_length` and an optional `expert_weights_norm`
  (default true); a sigmoid gate, shared experts or scaled expert weights
  are refused. A layer is routed when `blk.N.ffn_gate_inp.weight` is
  present, its experts the stacked `ffn_{gate,up,down}_exps` tensors, so
  dense and routed layers can mix. A routed feed-forward block is the
  router matmul, `route_experts`, `matmul_experts` for gate and up,
  `silu_mul` over every slot and `matmul_experts_add` into the residual;
  the arena gains the router scores, expert ids and weights as slots 9 to
  11, and the feed-forward slots are as wide as a dense layer or k
  experts, whichever is wider.

- `Config`, `read_config(file, prefix, moe)`: the configuration, read
  under the prefix the registry hands it (`qwen3.`, or `qwen3moe.` with
  `moe`): `block_count`, `embedding_length`, `feed_forward_length`,
  `attention.head_count[_kv]`, `attention.key_length`, `context_length`,
  `rope.freq_base` and `attention.layer_norm_rms_epsilon`, through
  [metadata](model-arch-metadata.md) (the gating function through
  `choice`, UINT32 1 or absent), so consumed integer fields accept
  positive INT32/UINT32/INT64/UINT64 values up to `INT_MAX` and consumed
  float fields finite positive F32/F64 values representable as nonzero F32.
  Duplicate consumed keys and wrong types fail.
  Optional defaults apply only when absent: KV heads equal query heads, key
  width is an exact embedding/head quotient, context is 4096, RoPE base is
  10000 and RMS epsilon is 1e-6. Explicit key width can differ from that quotient.
  Declared value/rotary widths must equal key width, and tensor layout
  must be `reference`. RoPE scaling is unsupported: type must be absent or
  `none`, and current/legacy factors absent or exactly one. These keys use
  the [GGUF metadata vocabulary](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- `check_config(config)`: the rules a configuration keeps whichever file it
  came from, which `read_config` calls last: head width even,
  GQA head counts dividing and projection widths fitting the runtime's
  integer indices, at most 256 experts a token and no more than a layer has,
  and context storage fitting float vectors.
- `Qwen3`: Qwen3's architecture over a `Config`, with its roles as the ids
  of `Role` and its layer kinds `dense` and `routed`.
  - `plan(index)`: the embedding gives the vocabulary, refused when it is
    not a positive `int`. A layer is routed when
    `blk.N.ffn_gate_inp.weight` is present, which a dense architecture
    refuses, and a dense layer needs the dense width. A layer's roles are
    its attention's norm, q and k norms and four projections, then its
    feed-forward norm, then the router and three expert stacks or the three
    dense matrices; a routed layer's norm and router are `copy` roles and
    its stacks `window` roles. The head's `output.weight` takes
    `token_embd.weight` as its alias, which is the tie. The context is the
    configuration's, the residual the embedding width, the slots
    `slot_widths` with the dense width counted when some layer is
    dense, and the two RoPE tables hold the context's positions at half a
    head each.
  - `fill_tables`: the RoPE cos and sin tables, `[pos*(head_dim/2) + i]`.
  - `embed`, `mixer`, `ffn` and `head`: the embedding gather; the attention
    (norm, grouped q, k and v projections, `norm_rope_kv`, `attention` and
    the output projection added into the residual); the feed-forward block,
    dense or routed by the layer's kind, reading whichever row of weights it
    is handed; and the head over the rows that want logits, compacted first.
    The embedding, the dense block and the head are the shared pieces of
    [blocks](model-arch-blocks.md), and so is the projection the grouped
    and routed products take.
    - `attention` runs over the paged KV cache; score scratch, causal
      masking and head scheduling belong to the backend. `norm_rope_kv`
      norms and rotates q and k and writes k and v into the views' blocks in
      one op.
    - Q/K/V and the dense gate and up share their input and go through
      `matmul_group`. The CPU groups eligible decode projections; a batched
      prefill keeps sequential matrix calls through the backend's fallback.
    - The norms, the per-head norm and RoPE, the SiLU and the residual adds
      are backend ops (`rms_norm_rows`, `norm_rope_kv`, `silu_mul`, and
      `matmul_add`, which folds the residual add into the output
      projections). Each row keeps the same arithmetic, and each op finishes
      before the matrix operations or cache writes that depend on it begin.
    - The routed down projection reads the SiLU's output as k entries a
      token row, so the part rebuilds the runs for it in the step's run list
      (`Step::scratch`), which the runtime has reserved.
- `slot_widths(config, dense)`: the floats one row takes in each of Qwen3's twelve arena slots, which its plan holds, `ensure` allocates and `footprint` counts.
- `open_dense(file, prefix)`, `open_routed(file, prefix)`: the registry's
  readers of a qwen3 file and a qwen3moe file, each the architecture over
  the configuration `read_config` reads.
- `synthetic_model(name, ...)`: a model of this architecture with a given
  shape and random weights, Q8_0 matrices and F32 norms, written as a file
  of the dense entry `name` the registry passes, which it writes as
  `general.architecture` and as the prefix of every key; the registry's
  `synthetic_model` hands it to `bench` to time without a file.

Supports dense and mixture-of-experts Qwen3 with Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K weights
and F32 embeddings/matrices/norms. F32 embedding rows are copied directly;
F32 matmul reads weight rows without staging. Missing
`output.weight` selects tied token embeddings for the output projection.
