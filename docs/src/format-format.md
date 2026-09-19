# `src/format/format.hpp` - format abstraction

Pluggable model-file format interface in namespace `format`. A format knows how
to open a file and enumerate its tensors plus metadata. Container independence
is the intended integration, not the current CLI/model behavior.

- `Tensor`: a tensor as described by metadata (name, shape, GGML type id) - not
  yet loaded.
- `ModelFormat` (abstract): `tensors()`, `metadata_string(key)`, `metadata_u64(key)`.
- `ModelFormatPtr` / `open(path, progress = {})`: sniff the magic and return a GGUF adapter,
  or `nullptr` for an unrecognized format. The inline definition lives at the
  bottom of `gguf.hpp`, after the reader and adapter definitions.

`gguf::GGUFFormat` derives from `ModelFormat` and wraps a `GGUFModel` for tensor
and metadata access. CLI/model code still uses `gguf::GGUFModel` directly;
using the format-independent interface throughout remains future work.

`LoadProgress(completed, total)` reports tensor payload bytes, excluding metadata
and padding. Callbacks are synchronous and optional; exceptions propagate.
The GGUF adapter forwards the callback to the reader.
