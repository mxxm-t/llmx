# `src/format/format.hpp` - what a reader reports

The types in namespace `format` that a reader shares whatever the container.

- `LoadProgress(completed, total)` reports tensor payload bytes, excluding
  metadata and padding. Callbacks are synchronous and optional; exceptions
  propagate. `gguf::warm` reports through it, and the loader passes the
  caller's.

A second format is a reader that produces the model's input,
`infer::QwenWeights` (`model/arch_qwen.hpp`), as `infer::gguf_weights` does
for GGUF, plus one branch in `infer::load_model`. The tokenizer and the chat format still read
`gguf::GGUFModel`.
