# `src/format/gguf.hpp` — GGUF v3 reader/writer

From-scratch implementation of the GGUF file format (v3) for `Q8_0`, `Q4_0`,
`Q4_1`, `Q6_K` and `F32` tensors, in namespace `gguf`. Those are not an
arbitrary set: a llama.cpp "Q4_0" file is MIXED, so all of them are needed to
load one at all.

- Constants: `MAGIC` (`'GGUF'`), `VERSION=3`, `ALIGNMENT=32`, GGML type ids
  (`GGML_TYPE_F32=0`, `Q4_0=2`, `Q4_1=3`, `Q8_0=8`, `Q6_K=14`) and the block
  size / bytes-per-block for each: 32/18 (Q4_0), 32/20 (Q4_1), 32/34 (Q8_0),
  256/210 (Q6_K).
- `TensorInfo::data_size()` switches on the type here rather than reading
  `quant::Registry`, because `quant/` includes `format/` and not the reverse.
  A new type therefore needs an entry in BOTH places.
- `MetaValue`: typed metadata value (all GGUF value types incl. arrays).
- `TensorInfo`: name, dims (`ne[0]` fastest), type, offset; `n_elements()` and
  `data_size()`.
- `GGUFModel`: metadata KVs, tensor infos, and ONE contiguous `blob` holding
  all tensor data with per-tensor `offsets`; `tensor_data(i)` / `tensor_bytes(i)`
  address it. `read_gguf` sizes the blob exactly and reads each tensor straight
  into place.
- `read_gguf(path)` / `write_gguf(m, path)` with the on-disk layout:
  header, metadata KVs, tensor infos (each padded to `ALIGNMENT`), then tensor
  data (each padded to `ALIGNMENT`).

This is the format the CLI and the `infer::Model` layer consume.
