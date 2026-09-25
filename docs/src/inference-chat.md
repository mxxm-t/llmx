# `src/inference/chat.hpp` - Jinja2-subset chat template renderer

Minimal Jinja2-subset renderer for GGUF `tokenizer.chat_template` strings, in
namespace `chat` (with a `chat::jj` value/render core).

- `Message`: `{ role, content }`.
- `chat::jj::Value`: tagged value (`NONE/BOOL/NUM/STR/LIST/DICT`) with
  constructors and the helpers `truthy`, `to_str` and `val_eq`.
- `chat::render(tpl, messages, add_generation_prompt, bos, eos) -> string`:
  render a template.
- `chat::chat_format(file, tok) -> ChatFormat`: how a model's conversations
  are written, for `chat` and the server alike: the file's
  `tokenizer.chat_template`, or ChatML when it carries
  none, and the text of its start and end tokens.

Supports the control-flow and expressions used by common chat templates
(Qwen2/3, Llama, Mistral, Gemma): `{{ ... }}` output, `{% if/elif/else/for/set
%}`, dict/list/string access, `.get()`/`.keys()`/etc., and the `messages`,
`add_generation_prompt`, `bos_token`, `eos_token` and `tools` context
variables. `tools` is none, as a request without tools passes it; no other
tool variable is set. The template methods are in `Call::call_method`, the
filters in `Filter::eval` and the `is` tests in `Test::eval`. Undefined
variables evaluate to none (empty) so unknown templates degrade gracefully
rather than throwing.

`is defined` checks presence, not the value. A variable is defined when the
context holds its name, an attribute or index when the dict holds the key,
and any other expression when its value is not none. So `tools` is defined
though none, a `namespace` field set to none is defined, and a template's
`{% if not date_string is defined %}` default applies when the caller passes
no date.

Compound statements consume their terminator once; adjacent output and nested
blocks remain visible to the enclosing parser. Function calls accept keyword
arguments from the first argument, including Qwen's `namespace(...)` state.
`tests/chat_template.cpp` checks the pinned real Qwen template against Jinja2
goldens across initial and follow-up turns, with and without reasoning and
generation headers. It also checks `is defined` and `is not defined` against
what Jinja2 renders for variables, attributes, indexes, none values and a date
default. `tests/chat.py` separately checks end-to-end replies against HF
fixtures. This does not establish support for every Jinja2 template or tools.
