# `src/format/gguf.hpp` — GGUF v3 reader/writer

From-scratch implementation of the GGUF file format (v3) for `Q8_0`, `Q4_0`,
`Q4_1`, `Q4_K`, `Q5_K`, `Q6_K` and `F32` tensors, in namespace `gguf`. Those are not an
arbitrary set: real GGUF files mix types. The pinned Q4_0 fixture requires
Q4_0, Q4_1, Q6_K and F32; other mixtures use the other supported types.

- Constants: `MAGIC` (`'GGUF'`), `VERSION=3`, `ALIGNMENT=32`, GGML type ids
  (`GGML_TYPE_F32=0`, `Q4_0=2`, `Q4_1=3`, `Q8_0=8`, `Q4_K=12`, `Q5_K=13`, `Q6_K=14`) and the block
  size / bytes-per-block for each: 32/18 (Q4_0), 32/20 (Q4_1), 32/34 (Q8_0),
  256/144 (Q4_K), 256/176 (Q5_K), 256/210 (Q6_K).
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
- Both `read_gguf` and `add_tensor_data` preserve `alignof(float)` between
  in-memory tensors. A 34-byte quantized tensor must not misalign a following
  F32 tensor when the loader removes on-disk padding.
- `read_gguf(path)` / `write_gguf(m, path)` with the on-disk layout:
  header, metadata KVs, contiguous tensor infos, then an aligned data section
  with each tensor payload aligned to `ALIGNMENT`. Tensor infos have no
  individual padding.

This is the format the CLI and the `infer::Model` layer consume. Stream failures,
file extents and overflow in metadata-derived sizes are not comprehensively
validated yet; successful `info` output is not a strict file-validation gate.
