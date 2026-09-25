# `src/quant/convert.hpp` - raw F32 tensors to and from GGUF

A model as raw F32 tensors is two files: `model.json` names the model and each
tensor with its shape, and `model.bin` holds their floats in that order. This
is what `llmx quantize` reads and `llmx dequantize` writes. Namespace `quant`.

- `raw_tensors(root, type)`: the tensors `model.json` describes, each to be
  written as `type`; a missing array, name or shape, a rank outside 1 to 4, or
  a dimension that is not an integer from 1 to 2^53-1 throws.
- `file_type_of(type)`: GGUF's `general.file_type` for a model whose matrices
  are all Q8_0 (7) or Q4_0 (2).
- `quantize_raw(json, bin, out, type) -> tensor count`: the model as a GGUF
  file with every tensor quantized to `type` through the registry
  (`quant/quant.hpp`). The size of `model.bin` must match the shapes, and
  storage past what a vector or a stream can hold is refused before any
  allocation.
- `dequantize_to_raw(in, json, bin)`: a GGUF file's tensors as raw F32, F32
  tensors copied and quantized ones decoded through the registry.

`tests/roundtrip.py` quantizes and decodes both types and checks refusals.
