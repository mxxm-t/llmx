# `src/format/format.hpp` - what a reader hands on

The types in namespace `format` that describe a model file's contents whatever
the container, for the loader to use without knowing it.

- `FileSpan {file, offset, bytes}`: where a tensor's bytes lie in a model file,
  `bytes` bytes from `offset` in the file at `file`, a UTF-8 path.
  `gguf::GGUFModel::span(i)` gives one for each tensor of a file it has read,
  whether its payload is mapped or not, and the loader plans its reads of the
  weights a device copies from them ([load](inference-load.md)).
- `LoadProgress(completed, total)` reports tensor payload bytes, excluding
  metadata and padding. Callbacks are synchronous and optional; exceptions
  propagate. `gguf::warm` reports through it, and the loader passes the
  caller's.

A second format is a reader that produces the model's input,
`infer::QwenWeights` (`model/arch_qwen.hpp`), as `infer::gguf_weights` does
for GGUF, and a `FileSpan` for each tensor, plus one branch in
`infer::load_model`. The tokenizer and the chat format still read
`gguf::GGUFModel`.
