# `src/format/weight_scales.hpp` - GGUF weight scale association and F8_E4M3 scale grids

`gguf::weight_scales` returns an optional companion tensor index for each tensor. It associates an NVFP4 or F8_E4M3 name ending in `.weight` with the same stem ending in `.scale`, without owning duplicate-name validation; `TensorIndex` retains that existing error ownership. It does not interpret `.input_scale`, which belongs to activation quantization. The decoder GGUF adapter stores these indices in `TensorView.scale`; model validation owns scale shape/type checks rather than this format lookup. An F8_E4M3 `.weight` without a `.scale` is refused by name, since its codes carry no scale of their own.

Callers that cannot consume scales pass `supported = false`. If an NVFP4 or F8_E4M3 weight has an associated `.scale` tensor, the format owner refuses it with the weight's name before the caller loads or adopts its tensors. The vision mmproj loader uses this check because its encoder does not yet consume scaled weights. An NVFP4 tensor without a companion is not refused by this check alone; its eventual reader still determines whether that type is supported.

## F8_E4M3 scale grids

An F8_E4M3 weight (`quant::GGML_TYPE_F8_E4M3`, one E4M3 code a byte) runs with one F32 scale per 128 inputs (`kFp8ScaleInputs`, `quant::F8_E4M3_SCALE_INPUTS`) of every 2^shift rows of each expert, its grid, shaped `{inputs / 128, ceil(rows / 2^shift), experts}` with the fastest dimension first.
Every kernel reads row `r`'s scale for input `k` at `(r >> shift) * (inputs / 128) + k / 128`, one index expression for every shift, so no kernel has a second build.
The shift is 0 for a scale a row (compressed-tensors' channel strategy) and 7 for one per 128 x 128 block, which is also the form a scale a tensor or an expert expands to.
A channel file's grid holds each row's scale `inputs / 128` times, 8.25 bits a weight against the block files' 8.002; that cost is known and measured at the device step, and a column shift is added only if the measurement asks for it.

- `fp8_scale_shift(weight, scale)`: the shift whose grid shape `scale` pairs with `weight`'s shape, or none. A weight is a matrix or a stack of experts whose inputs are whole 128-input blocks.
- `fp8_grid_shape(weight, shift)`: the grid shape of `weight` at `shift`.
- `fp8_grid_count(nin, nout, shift, experts)`: the scales of the grids of `experts` matrices of `nout` rows of `nin` inputs, or 0 where no grid fits; the one count rule the CPU's `fp8_scales` and Vulkan's `check_weight_scales` check a grid against.
- `fp8_grid(source, type, n, out, count)`: writes `count` F32 scales from `n` F32, F16 or BF16 scales, each widened exactly and repeated over `count / n` consecutive grid entries, so one function makes a grid from a scale a tensor, an expert or a row, or from a block grid. `hf_quant::repack` calls it for the composite kind `kFp8GridComposite` ([hf_quant](format-hf_quant.md)).
- `fp8_grids(m)`: a GGUF file's F8_E4M3 scales, an F32 scalar or one an expert (the one form a file of this type holds), each made the composite of its weight's grid at shift 7, which the loader makes as it makes a directory's repacked tensors ([load](inference-load.md)). `infer::read_model` calls it on every GGUF file ([directory](inference-directory.md)). A `.scale` of any other shape or type is refused.

llmx writes no file with a grid: a grid companion under id 43 would be a dialect the files' writers do not read, so writing one, when a use appears, carries its own metadata key or type, decided then; the decision is recorded in [STATUS](../STATUS.md).
