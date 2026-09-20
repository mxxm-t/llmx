# `src/format/gguf.hpp` - GGUF v3 reader/writer

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
  `data_size()` use checked arithmetic. Quantized rows must contain a whole
  number of blocks, even when the total element count would be divisible.
- `GGUFModel`: metadata KVs, tensor infos, and ONE contiguous `blob` holding
  all tensor data with per-tensor `offsets`; `tensor_data(i)` / `tensor_bytes(i)`
  address it. `read_gguf` sizes the blob exactly and reads each tensor straight
  into place.
- Both `read_gguf` and `add_tensor_data` preserve `alignof(float)` between
  in-memory tensors. A 34-byte quantized tensor must not misalign a following
  F32 tensor when the loader removes on-disk padding.
- `read_gguf(path, progress = {})` / `write_gguf(m, path)` with the on-disk layout:
  header, metadata KVs, contiguous tensor infos, then an aligned data section
  with each tensor payload aligned to `general.alignment` (default `ALIGNMENT`).
  The reader and writer require a unique uint32 alignment that is positive and
  a multiple of eight; non-power-of-two values such as 24 are supported.
  Tensor infos have no individual padding.

This is the format the CLI and the `infer::Model` layer consume. Reads/seeks
throw on stream failure. The internal `Reader` obtains the extent from the
opened stream and bounds strings, arrays and field reads before allocation.
Only version 3 and at most four tensor dimensions are accepted. Nested arrays
are supported up to `MAX_ARRAY_DEPTH` (256 containers); an empty array still
needs a valid element type. Rank-zero F32 and zero-sized tensors are accepted
as file objects; model execution imposes separate shape requirements.

Before allocating the payload blob or reporting progress, the reader checks
all tensor byte counts, aligned in-memory totals and on-disk ranges, including
zero-sized tensor offsets. Unordered or overlapping ranges are accepted if
each lies within the data section. Reads cannot use wrapped offsets or an
allocation total smaller than the validated payload. Overlap can amplify memory
use, and this is not a resource quota or complete metadata/model-schema validator.
String encoding, tensor-name semantics and general writer hardening remain separate.

The optional `format::LoadProgress` callback starts at `(0, total)` before
payload allocation and after structural validation, advances after successful reads of up to 8 MiB directly
into tensor storage, and ends at `(total, total)`. Empty models report `(0, 0)`
once. Failed reads throw before reporting those bytes as complete. Percentages
and console output belong to the caller; no extra tensor copy is introduced.
