# `src/model/arch/minimax_m3.hpp` - the MiniMax-M3.1 architecture

MiniMax-M3.1's text model in namespace `infer::minimax_m3`, reached through the registry's `minimax-m3` entry ([registry](model-arch-registry.md)); the design is [MINIMAX](../MINIMAX.md).
It implements the architecture contract ([architecture](model-architecture.md)); its plan is refused by name ("inference: minimax-m3 does not run yet; its plan is in docs/MINIMAX.md") until the ops of the design page run on the CPU.

- `Config`, `read_config(file, prefix)`: the configuration under `minimax-m3.`, through [metadata](model-arch-metadata.md).
  Every key that changes the math is required: the layer counts (`block_count`, `leading_dense_block_count`), the widths, the attention and rotary shape, `rope.freq_base`, the RMS epsilon, the experts and the index (`attention.indexer.head_count`, `key_length`, `top_k`, `block_size`, `local_blocks`).
  Held to one value, refused as `unsupported metadata` otherwise: sigmoid gating (`expert_gating_func` 2), normalized weights, one shared expert as wide as the routed ones, index heads equal to KV heads, one local block, value width equal to key width; `expert_weights_scale` is kept as the router's scale.
- `check_config`: a routed layer after the dense lead, whole query heads per KV head, an even rotation no wider than the head and the index key, at most 256 experts a token and no more than a layer has, and a cache row (K, V and the index key) whose context fits an allocation.
- `MinimaxM3::layout(tensors)`: the plan of a file's tensors.
  Every layer has attention with per-head q and k norms, the index projections and their norms, and keeps its index key a position (`LayerPlan::kv_aux`, the index key width); layers below `leading_dense_block_count` run the dense GLU block, the rest the router with its bias (`exp_probs_b.bias`), the expert stacks and one shared expert without a gate.
  A tensor the layer's kind does not read (a router or expert stack on a dense layer, a dense matrix on a routed one) is refused by name.
  Arena slots 0 to 13 are those of `slot_widths`; the slots of the selection come with its ops.
  `plan` builds this and refuses the architecture by name.
- `fill_tables`: NeoX RoPE over the first `rope.dimension_count` values of a head, its frequencies over that width.
- `embed` and `head` are the shared blocks ([blocks](model-arch-blocks.md)) over `token_embd`, `output_norm` and an untied `output`; `mixer` and `ffn` refuse by name until their ops land.
