# `src/format/hf_quant.hpp` - quantized Hugging Face checkpoints

`hf_quant` reads a directory's `config.json` `quantization_config` and makes each quantized linear's companion tensors one tensor of a type llmx already runs, whose values are the checkpoint's exactly (vLLM formats, quant step 1, `docs/STATUS.md`).
It owns the key: `infer::read_directory` ([directory](inference-directory.md)) hands it `quantization_config` and erases it before the architecture's table reads the rest of `config.json`, so the table never sees it.

- `apply(qc, m)`: `qc` is the key's value or null. A null `qc` reads nothing; otherwise `quant_method` must name a method llmx reads, `awq` today, and every key of the object is read, checked or refused by name.
  Afterwards any tensor still I32 or I64, which only a quantization rule consumes, is refused by name ("holds integers that no quantization rule of the directory reads").
- `repack(c, t, parts, out)`: writes composite `c`'s bytes for tensor `t` at `out` from its parts' bytes, read whole, in the order of `c.parts`; the loader calls it ([load](inference-load.md)).
- `kRepackVersion`: what the repacks write, carried in the disk tier's identity (`server::disk_identity`, [server](server.md)) as `repack: N`, so cached histories of a directory change with any change to a repack's bytes; it moves with every such change.

## AWQ

The GEMM packing of 4-bit AWQ with zero points (`version` `gemm`).
`quantization_config` keys:
- `quant_method` `awq`; `bits` 4; `zero_point` true; `version` `gemm` in any case (AutoAWQ's own files write `GEMM`); `backend` `autoawq` when present; `group_size` a multiple of 32 from 32 to 65536, required.
- `modules_to_not_convert`: null or a list of module names; the linears it names have no `qweight` and read as ordinary weights.
- `do_fuse`, `fuse_max_seq_len`, `modules_to_fuse` and `exllama_config` are taken as they are: they name how a runtime fuses or schedules kernels and change no value.
- Any other key, or another value of a checked one, is refused with the key named.

A linear `X` is `X.qweight` I32 [in, out / 8], `X.qzeros` I32 [in / group, out / 8] and `X.scales` F16 or BF16 [in / group, out] (Hugging Face order, slowest first).
Word j of a row holds output 8j + order[s] in nibble s, order = {0, 2, 4, 6, 1, 3, 5, 7}, for codes and zero points alike, and the weight at output n and input k is (q - z) * s.
`apply` replaces the three by one tensor `X.weight` of type Q5_0 and shape [in, out] at `qweight`'s place in the table, a `gguf::GGUFModel::Composite` of kind `awq_q5_0` ([gguf](format-gguf.md)), and refuses a companion without its `qweight`, a `qweight` without both companions, other storage types, and shapes that do not agree with the group.

`repack` writes Q5_0 with d = s in F16 and code q - z + 16, which is 1 to 31, so each value is the checkpoint's exactly.
An F16 scale is taken as it is; a BF16 scale is taken through F16 only when the round trip f16(bf16) gives the same value, subnormals included, and otherwise the load is refused naming the tensor, the scale, its output and its group.
Q5_0 holds 22 bytes for 32 values where AWQ holds 16 bytes of codes plus its share of the zero points and scales, so the repacked tensor is about 29 percent larger than its source.
The repack is the default path, not a fallback: a native 4-bit-with-zero type is added only on a measured need against the same model's Q4_K_M and Q4_0 files (F2DEV's review of the quant proposal).

`tests/spec_decode.py` decodes AWQ linears and their Q5_0 repack from the packing description, `raw-blocks` holds both to it, and `load-progress` loads a synthetic AWQ directory to its Q5_0 twin's logits ([TESTS](../TESTS.md)).
