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
  section in that file, one `Segment {path, file, start, base, size, first}` per
  shard placed after the one before in offset order, so a sharded model's
  tensors are addressed as one payload. A tensor's segment is found by its
  index (`first` is the segment's first tensor, and a file's tensors follow
  one another), not by its offset, since a zero-sized tensor at the end of
  one file has the offset the next file starts at. A segment's `file` is null until
  `map_payload` maps it read-only (see [mapped_file](format-mapped_file.md)),
  so a sharded model larger than host memory loads without a copy.
  `payload_size()` is the extent the offsets address; `tensor_data(i)` / `tensor_bytes(i)`
  address a tensor in whichever holds it, `tensor_data(i)` is null while its
  file is not mapped, and a mapped model's bytes are read-only.
  `span(i)` is where tensor i's bytes lie in its file, a `format::FileSpan`,
  mapped or not; it throws for a model built in memory, which has no file,
  and after `release_payload()`, which clears the segments. `release_payload()` drops the mappings or
  frees the blob once a model on device backends alone has copied every
  weight into device memory, which the loader tells from the backends that
  took each weight ([load](inference-load.md)), so the host does not
  hold the weights twice. `drop_pages(i)` lets one mapped tensor's pages leave the
  host's working set. On Windows a mapped model keeps its file handles open,
  preventing another writer from rewriting or removing the files while loaded.
  POSIX closes each descriptor after mapping; the files must still remain unchanged.
- `find(key)`: the metadata value under `key`, or null. Every reader of
  metadata looks keys up through it. `read_gguf` refuses a file that repeats
  a key, so the first match is the only one.
- `add_tensor_data` keeps `alignof(float)` between in-memory tensors, so a
  34-byte quantized tensor does not misalign a following F32 tensor.
  A file's tensors are addressed in place, padding included, and `read_gguf`
  refuses a tensor offset that is not float-aligned; each shard's data
  starts at a float-aligned offset.
- `read_gguf(path)` / `write_gguf(m, path)` with the on-disk layout:
  header, metadata KVs, contiguous tensor infos, then an aligned data section
  with each tensor payload aligned to `general.alignment` (default `ALIGNMENT`).
  The reader and writer require a uint32 alignment that is positive and
  a multiple of eight; non-power-of-two values such as 24 are supported.
  Tensor infos have no individual padding. Before it opens the output,
  `write_gguf` refuses a model whose files are not mapped, so a refusal
  leaves the file at `path` as it was.
  - `read_gguf` reads the headers alone: it parses and checks them and lays
    out the segments, and maps nothing, so `info`, `tokenize` and
    `detokenize`, which need the metadata alone, neither read the payload
    nor hold the files. It refuses a tensor name that repeats, in one file
    or across shards.
  - `map_payload(m)` maps every segment not yet mapped. It refuses a file
    whose size changed since its header was read, since the extents
    `read_gguf` checked no longer describe it. The shards mapped before the
    refused one stay mapped until the model is dropped.
  - `warm(m, tensors, progress = {})` reads the pages of the given tensors
    into memory in the order given, one byte of every page (`core::page_size`)
    in steps of up to 8 MiB, so their first reader does not fault them in.
    Their files must be mapped; a tensor that is not is refused before any
    progress. `bytes_of(m, tensors)` is their bytes, padding excluded. Whether to warm at all is the loader's rule
    ([load](inference-load.md)).

This is the format the CLI and the `infer::Model` layer consume. Metadata reads
and seeks throw on stream failure; mapped payloads follow the lifetime contract below. Read/write paths are UTF-8 and converted through
`std::filesystem::u8path` so Unicode cache paths also work on Windows.
The internal `Reader` obtains the extent from the
opened stream and bounds strings, arrays and field reads before allocation.
Only version 3 and at most four tensor dimensions are accepted. Nested arrays
are supported up to `MAX_ARRAY_DEPTH` (256 containers); an empty array still
needs a valid element type. Rank-zero F32 and zero-sized tensors are accepted
as file objects; model execution imposes separate shape requirements.

While reading the headers, before anything is mapped or any progress
reported, the reader checks all tensor byte counts and on-disk ranges
against each file's size, including zero-sized tensor offsets. Unordered or
overlapping ranges are accepted if each lies within the data section. Offsets
across shards use checked arithmetic, and `map_payload` checks that each
file still has the size its ranges were checked against. Overlap can amplify memory
use, and this is not a resource quota or complete metadata/model-schema validator.
String encoding, tensor-name semantics beyond uniqueness and general writer
hardening remain separate.

`warm`'s optional `format::LoadProgress` callback starts at `(0, total)`,
`total` being the given tensors' bytes, advances after each step of up to
8 MiB and ends at `(total, total)`. Tensors holding no bytes report `(0, 0)`
once. Completion describes pages read in, not GPU upload or model readiness;
it does not guarantee that every page remains resident. Callback exceptions
propagate. Percentages and console output belong to the caller; no extra
tensor copy is introduced.

Sharded files require the complete typed `split.no` (uint16), `split.count`
(uint16) and `split.tensors.count` (int32) metadata trio. Open the canonical first
`-00001-of-0000N.gguf` file; the reader discovers sibling names and reads each
header through a validated stream, closing it once read. Model/tokenizer metadata is authoritative in the first
shard, which may contain no tensors. Later shards may omit it; repeated keys
must match exactly by type and value, except each file owns its own alignment.
Duplicate keys/tensors and mismatched indices/counts/totals fail before progress
or payload allocation. Reading holds one file open at a time; once mapped, a
set holds no descriptor on POSIX and two handles per shard on Windows.

All shards share one offset space and one aggregate progress total; each is
mapped in place, with no copy, and a shard holding metadata alone has no
segment. Each file's extent is checked when its header is read and its size
again when it is mapped, so a shard truncated before loading, or between
reading and mapping, is refused before any progress. A mapped model
requires its files to stay unchanged while it is loaded: a file truncated
or rewritten under a live mapping is not detected, and reading the lost
pages can end the process (SIGBUS on Linux); Windows refuses such writes. The assembled model removes split
bookkeeping so writing it as one GGUF remains valid. Native fixtures exercise
structure, payload bytes and error paths; `tests/shards.py` compares sharded
synthetic model logits and NLL against the committed HF references.
