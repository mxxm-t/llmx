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
- `GGUFModel`: metadata KVs, tensor infos, and all tensor data addressed by
  per-tensor `offsets`: an in-memory model's in one `blob`, a file's data
  section mapped read-only, one `Segment` per shard placed after the one
  before in offset order (see [mapped_file](format-mapped_file.md)), so a
  sharded model larger than host memory loads without a copy.
  `payload_size()` is the extent the offsets address and `holds(p)` whether
  a pointer lies in the tensor bytes; `tensor_data(i)` / `tensor_bytes(i)`
  address a tensor in whichever holds it, and a mapped model's bytes are
  read-only. `read_gguf` maps every file and touches every page once in the
  steps the progress reports. `release_payload()` drops the mappings or
  frees the blob once a model on device backends alone has copied every
  weight into device memory (`Model::holds_payload`), so the host does not
  hold the weights twice. A mapped model keeps its files open, which Windows
  will not let another writer rewrite or remove while it is loaded.
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
throw on stream failure. Read/write paths are UTF-8 and converted through
`std::filesystem::u8path` so Unicode cache paths also work on Windows.
The internal `Reader` obtains the extent from the
opened stream and bounds strings, arrays and field reads before allocation.
Only version 3 and at most four tensor dimensions are accepted. Nested arrays
are supported up to `MAX_ARRAY_DEPTH` (256 containers); an empty array still
needs a valid element type. Rank-zero F32 and zero-sized tensors are accepted
as file objects; model execution imposes separate shape requirements.

Before mapping payloads or reporting progress, the reader checks
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

Sharded files require the complete typed `split.no` (uint16), `split.count`
(uint16) and `split.tensors.count` (int32) metadata trio. Open the canonical first
`-00001-of-0000N.gguf` file; the reader discovers sibling names and keeps the
validated streams open. Model/tokenizer metadata is authoritative in the first
shard, which may contain no tensors. Later shards may omit it; repeated keys
must match exactly by type and value, except each file owns its own alignment.
Duplicate keys/tensors and mismatched indices/counts/totals fail before progress
or payload allocation. Extremely large shard sets may exceed OS open-file limits.

All shards share one offset space and one aggregate progress total; each is
mapped in place, with no copy, and a shard holding metadata alone maps
nothing. Each file's extent is fixed when it is mapped, so a shard truncated
before loading is refused before any progress. The assembled model removes split
bookkeeping so writing it as one GGUF remains valid. Native fixtures exercise
structure, payload bytes and error paths; `tests/shards.py` compares sharded
synthetic model logits and NLL against the committed HF references.
