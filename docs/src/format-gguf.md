# `src/format/gguf.hpp` — GGUF v3 reader/writer

From-scratch implementation of the GGUF file format (v3) for `Q8_0` and `F32`
tensors, in namespace `gguf`.

- Constants: `MAGIC` (`'GGUF'`), `VERSION=3`, `ALIGNMENT=32`, GGML type ids
  (`GGML_TYPE_F32=0`, `GGML_TYPE_Q8_0=8`), `Q8_0_BLOCK=32`, `Q8_0_TYPESIZE=34`.
- `MetaValue`: typed metadata value (all GGUF value types incl. arrays).
- `TensorInfo`: name, dims (`ne[0]` fastest), type, offset; `n_elements()` and
  `data_size()`.
- `GGUFModel`: flat lists of `(key, MetaValue)` and `(TensorInfo, raw bytes)`.
- `read_gguf(path)` / `write_gguf(m, path)` with the on-disk layout:
  header, metadata KVs, tensor infos (each padded to `ALIGNMENT`), then tensor
  data (each padded to `ALIGNMENT`).

This is the format the CLI and the `infer::Model` layer consume.
