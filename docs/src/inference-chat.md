# `src/inference/chat.hpp` — Jinja2-subset chat template renderer

Minimal Jinja2-subset renderer for GGUF `tokenizer.chat_template` strings, in
namespace `chat` (with a `chat::jj` value/render core).

- `Message`: `{ role, content }`.
- `chat::jj::Value`: tagged value (`NONE/BOOL/NUM/STR/LIST/DICT`) with
  constructors and helpers (`truthy`, `get`, etc.).
- `chat::render(tpl, messages, add_generation_prompt, bos, eos) -> string`:
  render a template.

Supports the control-flow and expressions used by common chat templates
(Qwen2/3, Llama, Mistral, Gemma): `{{ ... }}` output, `{% if/elif/else/for/set
%}`, dict/list/string access, `.get()`/`.keys()`/etc., and the `messages`,
`add_generation_prompt`, `bos_token`, `eos_token` context variables. Undefined
variables evaluate to none (empty) so unknown templates degrade gracefully
rather than throwing.
