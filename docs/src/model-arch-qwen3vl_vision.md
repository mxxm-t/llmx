# `src/model/arch/qwen3vl_vision.hpp` - the Qwen vision tower

The vision tower of Qwen 3.5, 3.6, 3.8 and Qwen3.8-Flash-Next, read from an mmproj GGUF whose `general.architecture` is `clip` and `clip.projector_type` is `qwen3vl_merger`, which `model/arch/registry.hpp`'s `kVisionEncoders` maps here ([VISION](../VISION.md), section 3).

`read_config` reads `clip.vision.*`: the width, MLP, blocks, heads, patch, merge, output width, the LayerNorm epsilon, the normalization's mean and std, and the learned grid's side from `v.position_embd.weight`.
It refuses `clip.use_gelu` false or absent, a deepstack layer, a merge other than 2, a head width that is not a multiple of 4 or past 256, and a missing position table, each with its text; the processor's pixel bounds, which the file does not carry, are the family's documented 65536 and 16777216.

The plan's roles are the patch embedding's two temporal kernels and bias, the learned positions, each block's twelve and the merger's six, by the converter's names; the patch kernels must be `[patch, patch, 3, width]` and the position table square.

`encode` is HF's `Qwen3_5VisionModel.forward` as backend ops: both temporal kernels on the one prepared frame, the bias and the learned positions' bilinear taps (`mix_rows`); in each block LayerNorm, the q, k and v rows of `attn_qkv` as three projections and their biases, the rope without the norm over the patch's row and column (`norm_rope_partial` with two streams, the first `head_dim / 4` frequencies by row, the rest by column), `kv_write` and `attention` through the non-causal view, the output projection and its bias before the residual add, LayerNorm, the MLP with tanh GELU, its bias before the add; then the merger's LayerNorm, four consecutive patch rows read as one row, a projection, erf GELU and the output projection, each with its bias.
`inputs` gives each patch's row and column in merge-block order and its four bilinear taps of the learned grid with aligned corners, as HF's vision utilities compute them in F32.
`tests/arch_qwen3vl_vision.cpp` holds the refusals, the plan and the inputs; the suite's `vision` component holds the rows to HF.
