# `src/model/arch/dflash.hpp` - DFlash drafters beside a model

DFlash drafters (docs/SPECULATIVE.md, section 11) in namespace `infer::dflash`: a `dflash` sidecar file that drafts a block of tokens from a few target layers' rows, its checks against the target and its configuration.
DeepSeek-V4.1's DSpark rewrites its sidecar into its own blocks ([deepseek4](model-arch-deepseek4.md)); qwen35 keeps a DSpine sidecar under a `dflash.` namespace, and its drafter is being built (docs/STATUS.md, DSpine drafters for qwen35).

- `sidecar(side)`: whether a file's `general.architecture` is `dflash`.
- `check_sidecar(target, side, p, own, name)`: the checks every sidecar passes, each `dflash.` key not in `own` equal to the target's key under `p`, and a `tokenizer.ggml.mask_token_id`, which it returns; DSpark's normalization calls it.
- `normalize_beside(target, side)`: the registry's `normalize_drafter` for qwen35 and qwen35moe ([registry](model-arch-registry.md)). A sidecar's geometry is its own; its width, context and norm epsilon must be the target's, its `target_layers` ascending distinct decoder layers, and a target that already carries a DFlash drafter refuses it. Its tensors are renamed under `dflash.` and its keys under the target's prefix and `dflash.`, with the mask token as `dflash.mask_token_id`, so pairing ([pair](inference-pair.md)) gives `DrafterKind::joined` and joins it to the target.
- `Config` and `read_config(m, p)`: the joined configuration, its layers, block, attention window and candidates a row, taps, heads, rope base, feed-forward width and mask token; attention other than causal over a window on every layer, or a rope over part of a head, is refused by key.
