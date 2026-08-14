# `src/format/format.hpp` — format abstraction

Pluggable model-file format interface in namespace `format`. A format knows how
to open a file and enumerate its tensors plus metadata; the rest of the stack
never cares which container was used.

- `Tensor`: a tensor as described by metadata (name, shape, GGML type id) — not
  yet loaded.
- `ModelFormat` (abstract): `tensors()`, `metadata_string(key)`, `metadata_u64(key)`.
- `ModelFormatPtr` / `open(path)`: auto-detect format from the header, return
  `nullptr` if unrecognized.

GGUF is the first implementation (`format/gguf.hpp`). Future formats
(safetensors, raw) implement the same interface.
