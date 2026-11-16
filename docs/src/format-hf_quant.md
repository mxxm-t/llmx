# `src/format/hf_quant.hpp` - quantized Hugging Face checkpoints

`hf_quant` reads a directory's `config.json` `quantization_config` and makes each quantized linear's companion tensors one tensor of a type llmx already runs, whose values are the checkpoint's exactly (vLLM formats, quant step 1, `docs/STATUS.md`).
It owns the key: `infer::read_directory` ([directory](inference-directory.md)) hands it `quantization_config` and erases it before the architecture's table reads the rest of `config.json`, so the table never sees it.

- `apply(qc, m)`: `qc` is the key's value or null. A null `qc` reads nothing; otherwise `quant_method` must name a method llmx reads, `awq`, `gptq` or `compressed-tensors`, and every key of the object is read, checked or refused by name.
  Afterwards any tensor still I8, I32 or I64, which only a quantization rule consumes, is refused by name ("holds integers that no quantization rule of the directory reads").
- `repack(c, t, parts, out)`: writes composite `c`'s bytes for tensor `t` at `out` from its parts' bytes, read whole, in the order of `c.parts`; the loader calls it ([load](inference-load.md)).
- `linears(m, main, companions, required, make)`: the one table edit of every method: each `X.<main>` holding integers (`qweight`, `weight_packed`, or int-quantized's `weight`) with its companions `X.<suffix>` becomes one composite `X.weight` at the main's place, a companion without its main or a main without a required companion is refused, and `make` checks the parts and sets the kind, type and shape.
- `kRepackVersion`: what the repacks write, carried in the disk tier's identity (`server::disk_identity`, [server](server.md)) as `repack: N`, so cached histories of a directory change with any change to a repack's bytes; it moves with every such change.

## AWQ

The GEMM packing of 4-bit AWQ with zero points (`version` `gemm`).
`quantization_config` keys:
- `quant_method` `awq`; `bits` 4; `zero_point` true; `version` `gemm` in any case (AutoAWQ's own files write `GEMM`); `backend` `autoawq` when present; `group_size` a multiple of 32 from 32 to 65536, required.
- `modules_to_not_convert`: null or a list of module names; the linears it names have no `qweight` and read as ordinary weights.
- `do_fuse`, `fuse_max_seq_len`, `modules_to_fuse` and `exllama_config` are taken as they are: they name how a runtime fuses or schedules kernels and change no value.
- Any other key, or another value of a checked one, is refused with the key named.

A linear `X` is `X.qweight` I32 [in, out / 8], `X.qzeros` I32 [in / group, out / 8] and `X.scales` F16 or BF16 [in / group, out] (Hugging Face order, slowest first).
A companion beside a tensor that is not the integer main, such as a `weight_scale` beside a 16-bit `weight`, is refused by name.
Word j of a row holds output 8j + order[s] in nibble s, order = {0, 2, 4, 6, 1, 3, 5, 7}, for codes and zero points alike, and the weight at output n and input k is (q - z) * s.
`apply` replaces the three by one tensor `X.weight` of type Q5_0 and shape [in, out] at `qweight`'s place in the table, a `gguf::GGUFModel::Composite` of kind `awq_q5_0` ([gguf](format-gguf.md)), and refuses a companion without its `qweight`, a `qweight` without both companions, other storage types, and shapes that do not agree with the group.

`repack` writes Q5_0 with d = s in F16 and code q - z + 16, which is 1 to 31, so each value is the checkpoint's exactly.
An F16 scale is taken as it is; a BF16 scale is taken through F16 only when the round trip f16(bf16) gives the same value, subnormals included, and otherwise the load is refused naming the tensor, the scale, its output and its group.
Q5_0 holds 22 bytes for 32 values where AWQ holds 16 bytes of codes plus its share of the zero points and scales, so the repacked tensor is about 29 percent larger than its source.
The repack is the default path, not a fallback: a native 4-bit-with-zero type is added only on a measured need against the same model's Q4_K_M and Q4_0 files (F2DEV's review of the quant proposal).

`tests/spec_decode.py` decodes AWQ linears and their Q5_0 repack from the packing description, `raw-blocks` holds both to it, and `load-progress` loads a synthetic AWQ directory to its Q5_0 twin's logits ([TESTS](../TESTS.md)).

## GPTQ

GPTQ linears in the `gptq` and `gptq_v2` checkpoint formats, packed in int32, without act-order groups.
`quantization_config` keys:
- `quant_method` `gptq`; `bits` 4 or 8, required; `group_size` a multiple of 32 from 32 to 65536, required (-1, one group a row, is refused); `sym` true or false, true when absent as in transformers' `GPTQConfig`; `checkpoint_format` `gptq` (v1, the default) or `gptq_v2`; `lm_head` false; `pack_dtype` `int32`.
- 8-bit with `sym` false is refused: its codes q - z span -255 to 255, which no existing 8-bit code holds.
- Keys that name how the checkpoint was quantized or how a runtime schedules kernels and change no stored value are taken as they are: `desc_act`, `static_groups`, `true_sequential`, `hyb_act`, `damp_percent`, `damp_auto_increment`, `mse`, `meta`, the quantizer's name keys and transformers' `GPTQConfig` runtime keys (`use_exllama`, `exllama_config`, `backend` and the like); `desc_act` is taken because `g_idx` itself shows whether a linear's groups follow its inputs.
- Any other key, or another value of a checked one, is refused with the key named.

A linear `X` is `X.qweight` I32 [in * bits / 32, out], `X.qzeros` I32 [in / group, out * bits / 32], `X.scales` F16 or BF16 [in / group, out] and, when present, `X.g_idx` I32 [in] (Hugging Face order, slowest first).
A word holds 32 / bits codes from its low bits up, along the inputs in `qweight` and along the outputs in `qzeros`; v1 stores each zero point less one, v2 as it is, and the weight at output n and input k is (q - z) * s of group k / group.
`apply` replaces them by one tensor `X.weight` [in, out], a composite whose kind and type follow the configuration: `gptq_q4_0` and Q4_0 for 4-bit symmetric zero points, `gptq_q5_0` and Q5_0 for other 4-bit ones and `gptq_q8_0` and Q8_0 for 8-bit symmetric ones, with `zero_offset` 1 under v1.

`repack` writes d = s in F16 and the code q (Q4_0, whose value is q - 8), q - z + 16 (Q5_0, 0 to 31 since a v1 zero point is 1 to 16) or q - 128 (Q8_0), so each value is the checkpoint's exactly; scales are taken as AWQ's are.
A symmetric linear whose zero point is not the middle code, 8 or 128, is refused naming the tensor, the zero point, its output and its group, and a `g_idx` entry other than k / group (act-order groups, whose scales change inside a block of 32) is refused naming the tensor and the input.
Q4_0 holds 18 bytes for 32 values where GPTQ holds 16 plus its share of zero points and scales, and Q8_0 34 where 8-bit GPTQ holds 32, so the repacked tensors are about 8 and 4 percent larger than their sources.

`tests/spec_decode.py` decodes GPTQ linears and their repacks from the packing description, `raw-blocks` holds both to it, and `load-progress` loads synthetic GPTQ directories of each kind to their twins' logits ([TESTS](../TESTS.md)).

## compressed-tensors

Integer weights of the `pack-quantized` and `int-quantized` formats, as llm-compressor writes them; `float-quantized` (FP8) is a later step of the plan.
`quantization_config` keys:
- `quant_method` `compressed-tensors`; `format` `pack-quantized` or `int-quantized`, required; `quantization_status` `compressed`; `kv_cache_scheme` null; `sparsity_config` null or empty; `ignore` null or naming only `lm_head`, which no rule here quantizes; `global_compression_ratio` and `version` taken as they are; the keys taken are those llm-compressor's files carry.
- `config_groups`: each group's `weights` is one scheme, and every group must give the same one; `targets` must include `Linear`, so no linear is left unquantized by a narrower target, `input_activations` is taken and not applied (the run's `--dtype` decides activations, `docs/PRECISION.md`) and, when not null, written in words as `quantization.input_activations` for `llmx info`, `output_activations` must be null.
- A weights scheme: `num_bits` 4 or 8; `type` `int`; `symmetric` true or false; `strategy` `group`, with `group_size` a multiple of 32 from 32 to 65536, or `channel`, one scale a row; `dynamic` false; `block_structure` null; `actorder`, `observer` and `observer_kwargs` taken as they are, `actorder` because `weight_g_idx` itself shows whether a linear's groups follow its inputs.
- 8-bit asymmetric schemes, and asymmetric `int-quantized` ones, are refused: no existing code holds their values.
- The strategy and group rules hold the weights only: an activation scheme of strategy `token`, as int8 `int-quantized` files carry, is taken.
- Any other key, or another value of a checked one, is refused with the key and its group named.

A linear `X` is `X.weight_packed` I32 [out, in * bits / 32] with `X.weight_shape` I64 [2] holding [out, in] (`pack-quantized`), or `X.weight` I8 [out, in] (`int-quantized`); `X.weight_scale` F16 or BF16 [out, in / group] (one column under `channel`); under an asymmetric scheme `X.weight_zero_point` I32 [out * bits / 32, in / group]; and, when present, `X.weight_g_idx` I32 [in] (Hugging Face order, slowest first).
Codes and zero points are signed, -2^(bits - 1) to 2^(bits - 1) - 1; `pack-quantized` stores each plus 2^(bits - 1), 32 / bits a word from its low bits up, codes along the inputs and zero points along the outputs, and the weight at output n and input k is (q - z) * s of group k / group, z being 0 when symmetric.
`apply` replaces them by one tensor `X.weight` [in, out]: `ct_q4_0` and Q4_0 for 4-bit symmetric codes, `ct_q5_0` and Q5_0 for 4-bit asymmetric ones, and `ct_q8_0` and Q8_0 for 8-bit ones.

`repack` writes d = s in F16 and the code q + 8 (Q4_0, whose value is q), q - z + 16 (Q5_0, 1 to 31) or q (Q8_0), so each value is the checkpoint's exactly; scales are taken as AWQ's are, and the channel scale is each block's d.
A `weight_shape` other than the tensor's, a `weight_g_idx` entry other than k / group and an int8 code outside a 4-bit scheme's range are refused naming the tensor.
Q4_0 holds 18 bytes for 32 values where 4-bit codes take 16 plus their share of scales, Q5_0 22 where an asymmetric group of 64 takes 16 plus a zero point and a scale per 64, and Q8_0 34 where 8-bit codes take 32, so the repacked tensors are about 6 to 29 percent larger than their sources.

`tests/spec_decode.py` decodes compressed-tensors linears and their repacks from the packing description, `raw-blocks` holds both to it, and `load-progress` and the suite's `safetensors` component load synthetic directories of each kind to their twins' logits ([TESTS](../TESTS.md)).
