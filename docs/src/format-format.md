# `src/format/format.hpp` — format abstraction

Pluggable model-file format interface in namespace `format`. A format knows how
to open a file and enumerate its tensors plus metadata; the rest of the stack
never cares which container was used.

- `Tensor`: a tensor as described by metadata (name, shape, GGML type id) — not
  yet loaded.
- `ModelFormat` (abstract): `tensors()`, `metadata_string(key)`, `metadata_u64(key)`.
- `ModelFormatPtr` / `open(path)`: intended auto-detection interface; `open`
  has no definition yet.

The standalone GGUF reader (`format/gguf.hpp`) does not derive from
`ModelFormat`; CLI/model code currently uses `gguf::GGUFModel` directly.
Integrating this interface belongs with the next format.
