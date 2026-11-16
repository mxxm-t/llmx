# `src/format/safetensors.hpp` - safetensors files and directories

`safetensors::read_directory` reads the headers of a Hugging Face directory's weight files into a `gguf::GGUFModel`: its tensor table, offsets and one segment per file, laid out as `gguf::read_gguf` lays out a set of shards, so `gguf::map_payload`, `tensor_data`, `span` and the loader's stream read them unchanged.
It reads no metadata (`kv` stays empty) and keeps the tensors' Hugging Face names; the architecture's table turns `config.json` and those names into the GGUF keys and names (vLLM formats step 1, `docs/STATUS.md`).
Nothing past the headers is read.

A file is an 8-byte little-endian header length, a JSON object of that many bytes, and the tensors' bytes.
The directory's `model.safetensors.index.json` names each tensor's file in `weight_map`; without one the directory holds a single `model.safetensors`.
Files are read in name order, and a file's tensors in the order of their bytes.
Shapes list the slowest dimension first and are reversed into `ne`, fastest first.

Refused, each with the file and tensor named:
- a header length over 100 MiB (the parsed JSON takes many times its size) or past the end of the file;
- a header that is not an object, repeats a key, or whose `__metadata__` is not an object of strings;
- a tensor entry with keys other than exactly `dtype`, `shape` and `data_offsets`, a shape of rank over 4, or a shape or offset that is not a non-negative integer up to 2^53;
- a dtype other than `F32`, `F16`, `BF16`, `I8`, `I32`, `I64` and `F8_E4M3`, by name; what an integer or F8_E4M3 tensor means is the quantization rule's, and one no rule consumes is refused there ([hf_quant](format-hf_quant.md)). `F8_E4M3` is OCP's E4M3 (bias 7, no infinities, 0x7F and 0xFF NaN) and becomes `GGML_TYPE_F8_E4M3`; `F8_E5M2` and the FNUZ variants, whose bytes decode to other values, are refused as 8-bit floats llmx does not read;
- offsets that end before they begin or do not span the shape's bytes, tensors that overlap, leave a gap or do not end where the file does;
- a tensor at a file offset that is not a multiple of its values' alignment (`gguf::value_alignment`: 4 for F32 and the integers, 2 for F16 and BF16, 1 for I8 and F8_E4M3, so an FP8 checkpoint's odd-length 16-bit scales may precede any tensor), which the loader's alignment rule needs and which is refused rather than copied;
- an index that repeats a tensor, names a file by a path rather than a name beside it, or does not name exactly the tensors a file holds.
