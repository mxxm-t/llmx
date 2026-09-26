# `src/inference/chat.hpp` - chat template renderer

Renders a GGUF file's `tokenizer.chat_template`, a template in the Jinja language, byte for byte as the HF reference renders it: transformers' `apply_chat_template`, which runs Jinja2 in its immutable sandbox with `trim_blocks`, `lstrip_blocks` and the loop controls, a `tojson` that keeps key order and escapes no HTML, and the globals `raise_exception` and `strftime_now`.
Values behave as Python's do: `str()` and `repr()` of every kind, string methods over characters rather than bytes, integer and float arithmetic, the undefined value's messages, and Python's own messages for the type and division errors a render can raise.
Namespace `chat`, with the language itself in `chat::jj`.

- `Message`: `{ role, content, reasoning_content }`, the last absent when a turn has no reasoning apart from its reply.
- `assistant_turn(text) -> Message`: an assistant reply split as the Qwen templates split one themselves, the reply after the last `</think>` without the newlines that open it and the reasoning before the first `</think>`, after the last `<think>` there, without the newlines around it.
  A text without `</think>` is all reply.
  It is the one owner of the split.
- `ChatFormat`: the template parsed once (`program`), or the reason it is refused (`refusal`), the text of the start and end tokens, and whether a conversation keeps an assistant turn split under it (`split_turns`).
  `render(messages, add_generation_prompt)` renders a prompt and raises `TemplateError` where the reference raises; `require()` raises `Refused` with the refusal.
  `assistant(text)` is an assistant turn as a conversation keeps it: split by `assistant_turn` when `split_turns`, and whole otherwise.
  `chat` records each of its replies through it, and the server reads an assistant message a client sends back through it when the message carries no `reasoning_content`, so a conversation renders the same through both.
- `chat_format(source, bos, eos)` parses a template's source; `chat_format(file, tok)` takes the file's template, or ChatML when it carries none, with its tokenizer's start and end text.
- `Refused` is a template the renderer does not take; `TemplateError` is a render that fails.

The variables a render reads are those the reference passes for a conversation without tools or documents: `messages` (each a dict of `role`, `content` and, when present, `reasoning_content`, in that order), `tools` and `documents` as none, `add_generation_prompt`, `bos_token` and `eos_token`.
The thinking variables (`enable_thinking` and the like) are not passed, so a template takes its own default.
`context(messages, add_generation_prompt, bos, eos)` builds them, and `jj::Template::render` renders any such dict, which the fixture test uses for tools, tool calls and content given as parts.

## Which turns are split

The rule, for `chat` and the server alike:

- An assistant turn is split only under a template that reads `reasoning_content` and does not split a reply at `</think>` itself; among the pinned templates that is the two Qwen 3.8 ones.
- There, a turn that comes without reasoning apart from its text (each of `chat`'s own replies, and a server message whose `reasoning_content` is absent or null) is split by `assistant_turn` before the render: the text after its last `</think>` becomes the content and the reasoning before it `reasoning_content`.
- Under every other template the turn is rendered whole, as it came, and the template does with a `</think>` in it whatever it does.
- A message that carries a `reasoning_content` string is taken as sent, its content and its reasoning unchanged, under every template.

So each model sees earlier reasoning in the form its template was written for, whichever way a client sends it back, inline in the content or in `reasoning_content`.
A prompt differs from a render of the messages exactly as sent only in one case: a template that reads `reasoning_content` and does not split a turn itself, given reasoning inline.

`split_turns` holds when the template reads `reasoning_content`, as an attribute, a constant subscript, a `get()` or a filter's attribute, and does not itself split text at the constant `'</think>'`; the parser records both (`jj::Template::reads`, `jj::Template::splits_at`).
Only the Qwen 3.8 templates among the pinned ones meet it: they show earlier reasoning only from `reasoning_content`, so a reply kept whole would carry its `<think>` block inside the content.
The Qwen3, Qwen 3.5 and Qwen 3.6 templates split a whole turn themselves, and keeping it whole renders it byte for byte as the reference renders the text a client sends back.
Splitting it first would not: those templates trim a turn's whole content before they split it, but trim a reply that arrives split, so spaces or a tab opening the reply would be lost.
A template that knows no reasoning, Qwen2.5's or the ChatML fallback, takes the turn as it came, so a reply holding `</think>` keeps the text before it.

## The part of the language it takes

- Tags: `if`/`elif`/`else`, `for` with unpacking, an `if` filter and `else`, `set` including namespace attributes and block sets with filters, `macro` with defaults and keyword arguments, `break` and `continue`, and the reference's `generation` block.
  Comments and the whitespace markers `-` and `+` on every tag.
- Expressions: literals of every kind with Python's string escapes, lists, tuples and dicts, `x if c else y`, `and`/`or`/`not`, chained comparisons, `in`, arithmetic with Python's floor division and modulo for integers and floats, `~`, attributes and subscripts with the sandbox's rules, and slices.
- The loop variable's `index`, `index0`, `revindex`, `revindex0`, `first`, `last`, `length`, `depth`, `depth0`, `previtem` and `nextitem`.
- Filters: `count`, `d`/`default`, `dictsort`, `first`, `items`, `join`, `last`, `length`, `list`, `lower`, `map`, `reject`, `rejectattr`, `replace`, `reverse`, `safe`, `select`, `selectattr`, `string`, `title`, `tojson`, `trim`, `upper`.
- Tests: every one Jinja has but `sameas`, `escaped`, `filter` and `test`.
- Methods: a string's `startswith`, `endswith`, `strip`, `lstrip`, `rstrip`, `split`, `replace`, `lower`, `upper`, `title`, `count`, `find` and `join`; a dict's `get`, `keys`, `values` and `items`, which give views that print, compare and hold items as Python's do; a list's `count`.
- Globals: `range` (at most 100000 items, as the sandbox allows), `namespace`, `raise_exception` and `strftime_now`, which reads the local time through the C runtime's `strftime`, so a conversion that runtime does not take fails the render, as Python's `strftime` fails on the same platform.

Anything else is refused when the template is parsed: another tag, filter, test, method or global, unpacked call arguments, a keyword argument given twice, a macro reading `varargs`, `kwargs` or `caller`, a recursive loop, and anything Jinja would not compile, such as trailing tokens, a macro naming a parameter twice, or `loop` assigned anywhere inside a `for`, its target included.
A filter or test Jinja does not have at all is refused as Jinja refuses it, except inside an `if` or a conditional expression, where Jinja compiles it and fails only when it runs, and so does this renderer.
A template nesting its statements and expressions more than 100 deep, counting each link of a chain such as `a + b + c` or `a.b.c`, is refused too, since each level takes host stack to parse and render; the Qwen templates nest about 20 deep.
`chat` and `serve` raise the refusal before they take a turn or listen (`cli/main.cpp`); `generate` and the other commands never render a template, so they run on such a file.

Where Python's semantics reach past what this renderer holds, a render fails with `TemplateError` rather than giving other text: an integer past 64 bits, case mapping (`upper`, `lower`, `title`, the case tests and `dictsort`'s folding) of text beyond ASCII, printf-style formatting with `%`, set operations on views, and a namespace holding a namespace, which keeps any value from holding itself.
The other way round, `map`, `items`, `reverse` and the `select` filters give lists where Jinja gives iterators, so the few uses Jinja refuses on an iterator, such as its `length`, render here.
A render also fails once it runs more than ten million loop iterations and macro calls, nests its statements and expressions more than 250 deep at once, macro calls included, builds a value nested more than 100 deep, or builds a string or list past 256 MiB, where the reference would run on or fail at its own recursion limit.
Those bounds keep parsing and rendering within about 100 KB and 200 KB of host stack, well inside the 512 KiB of the smallest thread stack a server's render runs on.

## Checks

`tests/chat_template.cpp` (CTest `chat-template`) reads `tests/data/baseline_chat_template.json`, which `tools/gen_chat_baseline.py` writes with transformers' renderer:

- the pinned real templates, each held to its SHA-256, over 34 conversations each: the twelve the Qwen3 fixture always held, replies with their reasoning inline and split, tool turns, tools with text beyond ASCII and nested objects, tool calls with their arguments as a dict and as a string, content given as parts, conversations the templates raise on, and text JSON or whitespace handling could change;
- for each of them, whether the reference's renders keep an assistant turn split, which `split_turns` must match, and a two-turn conversation with seven replies kept through `ChatFormat::assistant`, among them replies with two `</think>`, none, and spaces or a tab after it, each rendering as the reference renders the turn kept that way;
- small feature templates over three conversations each, for the parts of the language the real templates reach only in branches their conversations do not take;
- templates the renderer must refuse;
- texts `assistant_turn` must split as the templates' own expression does.

Every case must render byte for byte, or fail where the reference fails; the message must match too when the template raised it, an undefined value did, or Python raised a type or division error.
The test also renders `strftime_now` with conversions a C runtime may not take, which must render or fail the render, never end the process, and templates past each limit above, among them a sum of 200000 terms, a macro recursing through ten statements a level, a list nested 1000 deep and a namespace holding itself, which must be refused or fail.
The same binary checks a `--scan` of the templates in every GGUF file on a machine by hand.
`tests/chat.py` checks `chat`'s follow-up replies against HF fixtures, and that a refused template stops `chat` and `serve` while `generate` runs; `tests/server.py` checks that the server renders an assistant turn sent back with its reasoning inline, in `reasoning_content` or with it null, as `chat` renders its own reply, and that two `chat` turns under a Qwen 3.8 template of the fixture are the lengths of the reference's renders, the reply kept split.
