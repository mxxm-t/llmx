# `src/model/arch/inkling2_vision.hpp` - the Inkling-2 vision tower

The vision tower of Inkling-2, read from an mmproj GGUF whose `general.architecture` is `clip` and projector `inkling2v`, which `model/arch/registry.hpp`'s `kVisionEncoders` maps here ([VISION](../VISION.md), section 12).
It is a third module rather than flags on `qwen3vl_vision.hpp`, since nearly every line of `encode` differs (no patch bias, a class row, a norm before the blocks and none after, separate q, k and v, layer scale, an adjacent-pair rope from position 1, half-cell position taps, a merger without biases); the pieces both share live in `model/encoder.hpp`.

`read_config` reads `clip.vision.*`: the width, MLP, blocks, heads, patch, merge, output width, the LayerNorm epsilon, the normalization's mean and std, the learned grid's side from `v.position_embd.weight` and the merger's hidden width from `mm.1.weight`.
It refuses a merge other than 2, a head width that is not a multiple of 4 or past 256, and a missing position table or merger, each with its text.

`preprocess` is `Inkling2ImageProcessor`'s: `image::Fit::pad_end` (no resize, the bottom and right padded with -1 on the 0..255 scale to whole 40 px tokens), `Matte::checker`, and the token bound `max_rows`, 4096 by default and at most (`kMaxTokens`, the model's practical bound, so 1080p and 1440p screenshots fit); HF never resizes, so an image past the bound is refused with a message naming `--image-max-tokens`, and the bound also bounds the rope table and the attention's rows.
`bound` is `max_rows` merge blocks of patches, its attention pieces over them and the class row, which the runner adds as the plan's extra row.
Its rows are plain text positions, marked `<|content_image|>`, `<|unused_200054|>` a token and `<|end_message|>`.

The plan's roles are the patch kernel (`[patch, patch, 3, width]`, no bias), the learned positions, `v.class_embd` and `v.class_position_embd.weight`, `v.pre_ln`, each block's eighteen (LayerNorms, `attn_q`, `attn_k`, `attn_v` and `attn_out` with biases, `ls1`, `ffn_up` and `ffn_down` with biases, `ls2`) and the merger's `mm.1` and `mm.2`.
`extra_rows` is 1: the class row is row n, after the patches, which the non-causal attention and the stored rope positions make the same as HF's row 0.

`encode` is HF's `Inkling2VisionModel.forward` as backend ops: the patch product and the learned positions' four bilinear taps (`mix_rows`), the class row copied from `class_embd` and `class_position_embd` added; LayerNorm over every row; in each block LayerNorm, q, k and v as one `matmul_group`, their biases, the rope without the norm in adjacent pairs (`RopeForm` adjacent, two streams: the first `head_dim / 4` pairs by the row's column + 1, the rest by its row + 1, the class row at (0, 0)), `kv_write` and attention through the non-causal view, the output product with its bias times `ls1` (`add_bias`'s scale) before the residual add, LayerNorm, the MLP with erf GELU, its bias times `ls2` before the add; then the merger reads the patch rows only, four consecutive rows as one, `mm.1`, erf GELU and `mm.2`.
`pieces` cuts it like the Qwen tower's: the embeddings, each block's projections, its attention in runs of query rows over the n + 1 rows, and the rest, then the merger.
`inputs` gives each patch's positions and taps in merge-block order, the taps at half-cell centres (`bilinear_taps` without aligned corners) as HF's vision utilities compute them in F32.
`tools/inkling2_vision_reference.py --encode` holds its rows to HF's `pooler_output` on the mmproj's own weights ([ASSETS](../ASSETS.md), The vision encoder references).
