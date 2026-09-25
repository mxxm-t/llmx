# llmx - Development Status

Current implementation and remaining work. Historical checkpoints, failed
experiments and raw evidence remain in [ASSETS](ASSETS.md) and
`docs/benchmarks/`; their dated next steps are not current blockers.

## Chat templates rendered as the HF reference renders them (2026-09-26, branch feat/chat-template-jinja)

- **Why:** this is step 2 of the plan in "Qwen 3.5, 3.6 and 3.8" below, whose step 2 now records this branch and the differences listed under Decided here. The renderer gave wrong output, with no error, on every Qwen 3.5 and later template: `macro` was skipped as an unknown tag, unknown filters returned none, `[::-1]` read as `[-1]`, `raise_exception` did nothing, ternaries and tuples did not parse, and trailing tokens were ignored, so `llmx chat` with the Qwen3.5-0.8B template dropped every message's content. The Qwen 3.8 templates also show earlier reasoning only when it comes as `reasoning_content`, which `chat` never passed.
- **Done:**
  - `src/inference/chat.hpp` is a new renderer of the Jinja language as transformers' `apply_chat_template` renders it: the lexer applies trim_blocks, lstrip_blocks and the `-` and `+` markers as Jinja does, the parser follows Jinja's grammar with its soft frames for unknown filters and tests, and values behave as Python's (str and repr of every kind, dict views, character-based string methods, integer and float arithmetic, the sandbox's attribute and subscript rules, the undefined value's messages and Python's own type and division errors). The part of the language it takes, and what it refuses, is listed in `docs/src/inference-chat.md`. A template is parsed once, where the old renderer parsed it again for every prompt.
  - A template outside that part is refused when it is parsed; `ChatFormat` records the refusal, and only `chat` and `serve` raise it, before a turn or before listening. The CLI's `Opened` builds the format once for every command, as the loader plan's `LoadedModel::chat` will, and `generate`, `logits` and the other commands never render a template.
  - `chat::Message` has `reasoning_content`. `chat::assistant_turn` is the one owner of the split at `</think>`, with the Qwen templates' own expression, and `ChatFormat::assistant` applies it where the template needs it (Decided here): `chat` records its replies through it, and the server reads an assistant message through it unless the message carries `reasoning_content` (a string or null; another type is 400). A render the template raises on is 400 with the template's message.
  - `tools/gen_chat_baseline.py` renders with transformers' own renderer (`render_jinja_template`) in a second isolated environment with transformers 5.17.0, whose versions the tool lists and holds transformers and Jinja2 to (`docs/ASSETS.md`); the 4.55.2 environment stays for the other goldens. `--add` pins a template by SHA-256, each pin's origin a public repository, revision and file, and `--extract` and `--scan` check every template a machine holds.
  - The thinking variables (`enable_thinking`, `preserve_thinking`, `reasoning_effort`) are not passed; each template takes its own default. They are the plan's follow-up branch, so the plan's step-2 cases that pass `enable_thinking` are left to it.
- **Decided here, approved by the user (2026-09-26):**
  - The plan splits every assistant turn; this branch splits a turn only under a template that reads `reasoning_content` and does not split a reply at `</think>` itself, both read from the parsed template (`jj::Template::reads`, `jj::Template::splits_at`), not from its text. Among the pinned templates that is the two Qwen 3.8 ones, which show earlier reasoning only from `reasoning_content`. The Qwen3, 3.5 and 3.6 templates split a whole turn themselves, and splitting it first is not neutral there: they trim a turn's whole content before they split it but trim a reply that arrives split, so a reply opening with spaces or a tab after `</think>` lost them, and the server's prompt differed from transformers' render of the text an OpenAI client sends back. Kept whole, every turn renders there as the reference renders its text, which the fixture checks for replies with no `</think>`, a leading `<think>`, two `</think>` and spaces or a tab after it. A template that knows no reasoning, Qwen2.5's or the ChatML fallback, keeps a reply holding `</think>` whole, as main did, rather than losing the text before it.
  - The plan counted 9 distinct Qwen 3.5 to 3.8 templates; the two machines and the official repositories hold 8.
- **Review fixes:**
  - The first version's structure was kept, and these were wrong in it: the reply was taken after the first `</think>` where the templates take the last; `map(attribute=...)` held a statement that could not have compiled and applied its default only at the end of the path; lstrip_blocks tested a condition that was always true; string escapes called a function that does not exist; integer arithmetic overflowed as signed C++ (undefined behaviour) where Python grows; a repeat, a replace or a loop could run without bound; the `for` loop's `else` and a block set's filters were parsed in the enclosing soft frame; the `generation` tag was refused; and case mapping beyond ASCII silently differed from Python, where it now fails.
  - A second review found a `break` or `continue` in a loop's `else` or in a block set dropped instead of reaching the loop around it, `callable` false for an undefined value and for the loop object, `tojson`'s indent capped at 64 spaces, the conditional expression's undefined value without the line Jinja names, and `strftime_now` with a conversion the MSVC runtime does not take ending the process on Windows.
  - A third review found the split not neutral under the Qwen 3.5 and 3.6 templates (Decided here); `select`, `reject`, `selectattr` and `rejectattr` refused, which refused the Qwen3-Coder template of a GGUF on the workstation; a template chaining 5,000 sums, or a recursive macro through nested statements, ending the process by exhausting the stack, where parsing now counts each link of a chain against its nesting limit, and a render counts every statement and expression it nests, macro calls included, against one limit; templates Jinja does not compile rendered (`loop` assigned inside a loop, a macro parameter or a keyword argument given twice); `keys()`, `values()` and `items()` giving lists that printed and compared as lists; float `//` computed as `floor(x / y)`, so `1 // 0.1` gave 10.0 where Python gives 9.0; and error texts that differed from Python's for concatenation, repetition, comparison and division. Values nested past 100 containers, and a namespace holding a namespace, now fail too, since printing, comparing and freeing them followed the nesting on the host stack.
  - Against the fixture as it now stands, the code before the third review fails 70 cases and ends the process on the limit checks (status 139 on Linux), and main's renderer renders 144 of its 652 template and feature cases as the reference does.
- **Exact:** on Qwen3-0.6B Q8_0 on the CPU, against main dc31dd4: `generate` greedy and seeded without their timing lines, `logits`, a three-turn `chat` whose every reply holds `</think>` with its prompt token count per turn, and the server's `/v1/chat` ids and prompt tokens and `/v1/chat/completions` text over a conversation whose assistant turn keeps its reasoning inline, all give the same bytes. Where the old renderer was wrong the prompts change: every Qwen2.5 prompt loses the line feed the template's last line ends in, which Jinja drops and the old renderer kept; three Qwen3 templates (lmstudio-community's, the official repositories' and Qwen3-4B-Instruct-2507's) render a conversation ending in an assistant turn as the reference does; and every Qwen 3.5 and later prompt changes. On the Qwen3-0.6B GGUF's template the old renderer matched the reference on every conversation it could hold but the empty one, which the template now refuses as it does in transformers, and which neither `chat` nor the server sends.
- **Tests:**
  - Byte-equal to transformers 5.17.0 on 16 pinned templates, the 8 distinct Qwen 3.5, 3.6 and 3.8 ones, 7 Qwen3 ones and Qwen2.5's, over 34 conversations each with tools, tool calls and content given as parts among them: 544 of 544 cases, 56 of them failures the template raises, with the same message. Each keeps an assistant turn as the rule read from Jinja's own parse of it says, and as transformers' renders say where they decide it, and renders a two-turn conversation with seven replies kept that way as the reference does, 128 renders. With 36 feature templates, 10 limit checks and the Qwen3 case the test holds in its source (the audit round below), 791 of 791 cases; 33 of 33 refusals; 11 of 11 splits (CTest `chat-template`, on Linux and with MSVC on Windows).
  - The fixtures regenerate byte for byte in the reference environment on Linux and on Windows. `baseline_chat.json` keeps the prompt and reply IDs of all 25 turns of its nine cases, and the twelve Qwen3-0.6B cases render the old Jinja2 goldens byte for byte.
  - No template refused: the Linux MI50 machine's model directories (205 GGUF files through their links, two dangling links skipped) and every GGUF location on the workstation (7848 files: the HF cache, llmx's cache, LM Studio's models, and the project and temporary directories) hold 30 distinct templates, DeepSeek V4 and V4.1, Gemma 4, gpt-oss, MiMo, MiniMax, Command R, Phi-3, Qwen2 and the Qwen families. None is refused, and all 1377 cases of the scan are byte-equal on Linux and on Windows, gpt-oss's `strftime_now` date read in the reference's time zone; the rule for keeping a turn agrees with transformers' renders on every template they decide it for.
  - Refusal: a template with `{% filter %}` stops `chat` (status 1, nothing on stdout) and `serve` (status 1, before listening), with the reason, and `generate` runs on the same file (`tests/chat.py`).
  - Server equals CLI: on a synthetic model whose reply is `qr</think>o`, `chat` records the turn split under a template that reads `reasoning_content`, and the server's prompt for the same conversation has the same length as `chat`'s whether the assistant turn comes with its reasoning inline, in `reasoning_content` or with it null; under a template without reasoning both keep the turn whole; under the fixture's Qwen 3.8 template, `chat`'s two prompts are as long as transformers' renders with the reply split, the second beginning with the first, and the server's second prompt is the same (`tests/server.py`, `check_reasoning`).
  - Linux, CPU only in the build image, on this commit: CTest 22 of 22 and all 17 suite components (`--device cpu --require-tools`, the four Qwen3-0.6B fixtures on disk), the real model's server checks included, in 742 s at a load average of 12 to 27; the `chat-template` test built with the address and undefined-behaviour sanitizers over the fixture and the scan, and 28,397 broken versions of the fixture's templates and 36,267 of the scan's, parsed or refused, whose 70,797 renders give text or a `TemplateError`, with no sanitizer report. MSVC builds `llmx` and the test at /W4 without a warning.
  - At the limits a parse takes at most 91 KB of host stack with MSVC and 77 KB with GCC, and a render 176 KB and 123 KB, measured by a probe in each nesting level; the Qwen templates nest 22 deep to parse and 17 to render.
  - A render of 202 messages, 395 KB of text, takes 2.5 to 2.7 ms on the Qwen3 template and 3.8 to 4.0 ms on the Qwen 3.5 one, level with the code before the third review, interleaved on the Windows machine.
- **Rebased onto main `007b504`** (2026-09-26), past the `/v1/tokenize` and `/v1/detokenize` routes, the spec decoders, the Q4_0 zero-sign fix and `ignore_eos`, still one commit.
  `/v1/tokenize` renders `messages` through the same `messages_of` and `ChatFormat::render` as the chat routes, so its ids come from the new renderer and the one split owner, and a conversation the template raises on is 400 there as on the chat routes; `check_reasoning` now holds `/v1/tokenize` to the chat routes' counts and refusals.
  The chat fixture no longer has the old file's top-level `cases`, so the server's tokenize check reads the twelve cases carried over from the Jinja2 goldens under the template the fixture pins for the Q8_0 file, the same messages and texts as before.
  `ignore_eos` and the tokenize routes are unchanged; the conflicts were in STATUS, CI, SERVER, USAGE, `docs/src/cli-main.md` and `tests/server.py`, each kept with both sides' content.
  The chat and server pages state the split rule plainly: which turns are split, when, and that a message carrying `reasoning_content` is taken as sent.
- **Audit round** (2026-09-26), four commits around the renderer's:
  - Before it, `b4d3e0b` adds to main's own `chat-template` test a conversation ending in an assistant turn under the Qwen3 template of the official repositories (the `tokenizer_config.json` of Qwen/Qwen3-0.6B, 8B and 14B).
    Main's renderer reads that template's `messages[::-1]` as `messages[-1]`, finds the wrong last user turn and drops the reasoning of the reply after it, so the case fails on main, the test exiting 1 with the reply rendered without its reasoning; the renderer's rewritten test keeps the case, and it passes there, as the fixture's `legacy-0-complete-nogen` case of that template does.
  - `369220a`, test first, and `54b3aaa`: a `map` filter naming `map` applied `map` to every item one level deeper on the host stack and copied the names left at each level, so a template chaining thousands of them could end `chat`, or `serve` for every client.
    Each such `map` now counts one level of the render's depth and reads the names after it by index, so `'a'|map('map', ... 5000 times ..., 'upper')` fails the render with `maximum recursion depth exceeded`.
    Before the fix, 5000 names held 1,247 MiB before the value depth limit stopped the render, and 20000 ran out of a 6 GiB memory cap; with it, both fail at the depth limit within 4 and 10 MiB.
  - The last commit: `docs/src/inference-chat.md` says that a method or global reached other than by a call where the template writes it fails at render, not at parse, and that the list bound of `*` and `+` counts 2^28 items rather than 256 MiB; the header comments of `chat.hpp` and the test are two sentences each, and no comment names a template family the tests do not hold.
- **Gates on the stack's tip** (2026-09-26, the Linux machine's CPU, EPYC 7262, with no GPU device, against main `007b504` built beside it, at a load average of 12 to 34 on 16 threads; the tree gated is this commit's but for this block and the audit round's numbers):
  - The `chat-template` test at each commit of the audit round: `b4d3e0b` exits 1 on the Qwen3 case, `395c88c` passes 790 of 790 cases, `369220a` fails 1 of 791, the chained `map` stopped by the value depth limit instead, and `54b3aaa` passes 791 of 791.
  - Both builds configured fresh and built in full (24 and 30 objects), and main's CPU build: 0 warnings and 0 errors each.
  - CTest 22/22 on the CPU build and 25/25 on the Vulkan-enabled one, `chat-template` included, `backend-vulkan` and `vulkan-lifetime` skipped without a device.
  - The suite with `--no-perf-floor --require-tools --require-baseline --device cpu`: 18 of 18 on each build (669 and 611 s), `server` with its real-model, `ignore_eos` and `check_reasoning` checks included.
  - Byte identity with main on Qwen3-0.6B Q8_0 and on Q4_K_M: `generate -n 64 --temp 0`, `generate -n 64 --seed 42` and `logits --top 20` on four prompts (exit status, stdout without its timing and stderr), a three-turn `chat` whose three replies hold `</think>` with its prompt token count per turn (31, 48 and 63), and `/v1/chat`, `/v1/chat/completions` and `/v1/tokenize` over a conversation whose assistant turn keeps its reasoning inline, the same with the reasoning in `reasoning_content`, and one user turn: 49 of 50 the same on each model.
    The one difference, on both models, is `/v1/tokenize` of a conversation ending in an assistant turn that carries `reasoning_content`: main read no such field, so the Qwen3 template rendered that turn with an empty think block (34 tokens), and the stack renders its reasoning (44 tokens), as transformers does in the fixture's `reply-split-last` case; the same turn with its reasoning inline gives the same 44 tokens on both.
  - `tools/server_mix_check.py` on Qwen3-0.6B Q8_0: 16 requests of 928 tokens alone, 0 of 16 differ together, 0 of 12 skewed with 4 clients leaving, and the 4 CLI checks equal.
  - MSVC on Windows builds `llmx` and `llmx-chat-template-test` of this tree at /W4 without a warning.
  - The single-commit stack before the audit round passed the same host gates and all six hosted jobs.
- **Merged** at `f944091` (2026-09-26), with all six hosted jobs passing on the stack's tip. The user approved Decided here on 2026-09-26 and keeps the split rule: it gives every Qwen model the history format it was trained on whichever way a client sends earlier reasoning, and llmx's prompt deliberately differs from other engines' only where a template reads `reasoning_content` and a client sends the reasoning inline. No device, model, kernel or loader code changed, so the merge rules ask for no device suite.

## A reply that ends only at its limit: `ignore_eos` (2026-09-26, branch feat/ignore-eos)

- **Why:** a fixed reply length, which the serving gate's load (`tools/server_load.py --output-len`) and any throughput comparison need, held only while the model did not emit its end of text: the routes read no `ignore_eos` (a gap the load tool's block below lists), and the CLI had no way to ask either.
- **Done:** `infer::Sampling::ignore_eos`, off by default; `generate --ignore-eos` and `chat --ignore-eos`; a boolean `ignore_eos` on `/v1/generate`, `/v1/chat`, `/v1/completions` and `/v1/chat/completions`, refused with 400 when it is anything but `true` or `false`, `null` included.
  It is one rule, in the sampler: `infer::sample(logits, settings, end, gen, rng)` is the call `infer::generate` and the scheduler both make, with `Tokenizer::eos_id`, and with the setting on that id scores negative infinity before any step, greedy included, and a draw leaves it out of the ranked window before top-k, top-p and the softmax, so a nucleus whose float sum falls short of the draw cannot fall back on it.
  Greedy's scan starts past a masked id 0, so a row whose other scores are all negative infinity or NaN does not give the masked id either.
  The id masked is the one `Tokenizer::is_eos` ends a reply at: Qwen3's files name `<|im_end|>` (151645) there, the chat template's `eos_token` is the same id, and llmx ends a reply at no other (the `<|endoftext|>` the files name as start and padding token is an ordinary token to the end check), so there is one id to mask.
  A reply still ends at its token limit, the context's for an uncapped compatible request, and at a stop text.
- **The context on the CLI:** `generate` checks nothing up front, so a `-n` up to what the prompt leaves of the context runs to `-n` with the server's ids, and past that the command prints one token more than fits and stops with the context error, as main does with the option off; the server refuses such a capped request with 400 before it starts.
  Ending the CLI's reply at the context instead would change main's output with the option off, so it is left open as a fix of its own.
- **Penalties and log-probabilities:** the masked id does not exist for the draw, so the penalty cannot bring it back and top-k and top-p count only the other tokens.
  The mask writes nothing to the logits row.
  The base reports no log-probabilities; `feat/server-logprobs` reports the row as the model gave it, before the penalty, the temperature, top-k and top-p, and that stays so: under `ignore_eos` a token's log-probability is its share of the model's own distribution, the end token's share in it, and the end token can appear among `top_logprobs` as a token the penalty lowered can.
- **tools/server_load.py:** `--output-len` already sent `ignore_eos` on every route, the one way the tool fixes a reply's length; llmx now honours it, and the self-test checks that every request of a workload read with `--output-len` sends it and one read with `--tokens` does not.
- **Tests:** `sampler` holds the masked id to its definition (greedy, the penalty, top-k, top-p, the draw's frequencies, an id past the row, a masked id 0 beside scores that are all negative infinity or NaN, the settings' overload); `generation-stream` runs a reply past the masked EOS to its limit and to a stop text; `server` writes the synthetic model again with an end token from its own greedy reply, so a greedy reply ends there after 9 of 12 tokens and runs to 12 with `ignore_eos`, uncapped to the 16-token context, which `generate --ignore-eos` asked for the room the prompt leaves fills with the same ids, and on Qwen3-0.6B Q8_0 a turn answered without thinking ends after 13 of 48 tokens and runs to 48: the CLI and the server give the same ids, greedy and seeded, requests with and without it give their own ids side by side, and a non-boolean value is refused on every route; `cli` takes `--ignore-eos` on the `generate` and `chat` pages, refuses it on every other command and refuses a value written after it, checked by the reason, which a parser without the switch would not give.
  The masked id 0 checks fail on the sampler without the greedy scan start.
- **Gates on the Linux machine's CPU (EPYC 7262) on this change's source on main `b7b585f`, against main built beside it, with other work loading the machine (load average 15 to 64 on 16 threads); the commit gated differs from this one in its base, in docs and in the two test checks the review round below tightened:**
  - The branch's CPU and Vulkan-enabled builds, configured fresh and built in full (24 and 30 objects), and main's CPU build: 0 warnings each.
  - CTest 22/22 on the CPU build and 25/25 on the Vulkan-enabled one, `backend-vulkan` and `vulkan-lifetime` skipped without a device.
  - The suite with `--no-perf-floor --require-tools --require-baseline --device cpu`: 16 of 17 on each build (26 and 23 minutes), `server` failing in its uncapped check, the requests past the 600 s client timeout, as main's did on the same machine at such load, since each test server starts a worker per logical CPU inside a quota of 6.
    That component's checks then ran one at a time with the servers and the CLI at `--threads 6`, the branch and main at once: all 11 pass on the branch and all 9 on main, the uncapped check in 542 and 535 s, the synthetic `ignore_eos` check ending after 9 of 12 tokens and the Q8_0 one after 13 of 48.
  - `tools/server_mix_check.py` on Qwen3-0.6B Q8_0: 16 requests of 928 tokens alone, 0 of 16 differ together, 0 of 12 skewed with 4 clients leaving, and the 4 CLI checks equal.
  - Byte identity with main with the option off, on Qwen3-0.6B Q8_0: `generate -n 64 --temp 0`, `generate -n 64 --seed 42` at the default sampling and `logits --top 20` for three prompts, 9 of 9 the same; 96 requests on the four routes, greedy, temperature 0.8 with seed 7 and the defaults with seed 11, each with `ignore_eos` absent and `false`, whole and streamed, 96 of 96 the same, of the 48 whole replies 24 ending at the length and 24 at the end token.
    On Qwen3-8B Q4_K_M the 9 CLI cases are the same.
  - Review round (2026-09-26): the branch rebased onto main `dc31dd4`, whose one code change past `b7b585f` is the tokenizer's constructor reading the `qwen35` pretokenizer name, so this change's own source is the one gated above.
    The `cli` check of a value after `--ignore-eos` now reads the reason, and the load tool's `--tokens` request comes from a workload read without `--output-len`.
    On the rebased tree both builds have 0 warnings, CTest passes 22/22 and 25/25 (the two Vulkan tests skipped), and the suite's 16 components other than `server` pass on the CPU build with `--require-tools --require-baseline` (load average 17 to 34); `server` is the hosted HF reference job's.
    Against main's binary, which has no `--ignore-eos`, the `cli` check now fails where it passed before, and with a workload that always asks to ignore the end of text the self-test now fails on every API where it passed before.
    The hosted jobs pass on this commit, the HF reference job running the whole suite, `server` included.
- **Merged** at `7f5ecd3` (2026-09-26), rebased onto main `575a2ec` behind the Q4_0 fix below: on the stack's tip both builds have 0 warnings, CTest passes 22/22 and 25/25, the suite passes 18 of 18 on each build on the host CPU, `server` included, and the hosted jobs pass.
  The change is host code (the sampler, the routes, the CLI flag), which the merge rules gate on the CPU and the hosted jobs, and a device changes the logits the rule reads, not the rule.
  Open: the CLI at the context, above, and the test servers' worker count inside a CPU quota, which puts the uncapped check past its timeout on a loaded machine on main too.
  A model whose reply ends at more than one token needs each of them masked: the mask reads `Tokenizer::eos_id`, the one id `is_eos` ends a reply at today, so the work that gives `is_eos` a second end token (Qwen 3.x, DeepSeek 4.x) widens the mask with it.

## Q4_0 decodes -0 at nibble 8 under a negative scale (2026-09-26, branch fix/q4_0-negative-zero, merged at `7f5ecd3`)

- **Why:** the format decodes a Q4_0 value as `d*(nibble-8)`, which is -0 at nibble 8 when the scale is negative, and llmx gave +0 there, so `llmx dequantize` of the Qwen3-0.6B Q4_0 file differed from a decoder written from the format description in the sign of 26,631,920 zeros ([ASSETS](ASSETS.md)).
- **Done:** the CPU decode and the Vulkan embed both give the format's value, the embed with the sign set as bits; `tests/roundtrip.py` decodes raw Q4_0 blocks under every scale of its set, negatives included, against the spec decoder, and `backend-vulkan` embeds raw Q4_0 rows under negative scales against the CPU.
- **Gates** (on the stack's tip against main `575a2ec`, on the Linux machine's CPU): the round trip fails on the test commit alone, at 8 of 512 values, and passes with the fix; `llmx dequantize` equals the spec decoder bit for bit on every tensor of the 0.6B Q8_0, Q4_0, Q5_K_M and Q4_K_M files; `generate -n 64 --temp 0` and `logits --top 20` on two prompts on 0.6B Q8_0 and Q4_K_M, 16 of 16 the same as main; builds, CTest and the suite as in the `ignore_eos` block above; `backend-vulkan` passes on an MI50 and, after the merge, on the Radeon VII, built on Windows with no new warning; the hosted jobs pass.

## The qwen35 pretokenizer (2026-09-26, branch feat/tokenizer-qwen35)

- **Goal:** the tokenizer reads the Qwen3.5, 3.6 and 3.8 files (`tokenizer.ggml.pre` `qwen35`) and gives HF's ids, and every other pretokenizer name stays refused.
- **Done:**
  - `Tokenizer` accepts `qwen35` beside `qwen2`, and a refusal names the key and the implemented values.
  - `pretokenize` is unchanged: the qwen35 regex adds `\p{M}` to qwen2's letter runs and to the class its punctuation runs exclude, and llmx reads every byte at or above 0x80 as a letter, so marks already join letter runs.
  - `tools/gen_baseline.py tokenizer-qwen35` writes `tests/data/baseline_tokenizer_qwen35.json` (45 KB) from the `tokenizer.json` and `tokenizer_config.json` of Qwen/Qwen3.5-0.8B at 2fc06364 with `tokenizers` 0.23.2, and refuses either file if its SHA-256 is not the pinned one.
    It holds HF's ids for the 20 texts of the Qwen3 fixture, 4 Thai, 3 Devanagari, 5 CJK punctuation and 5 special-token texts, the 1,251 tokens and 969 merges those texts reach, and apart from them the 7 control tokens only `tokenizer_config.json` adds.
    Every added token has the type the GGUF files give it: control for HF's special tokens and for those written `<|name|>`, user-defined for the other 6.
  - `tests/tokenizer.py` writes a file from it that names `qwen35` and requires HF's ids for all 37 texts and one id for each of the 7 control tokens, alone and side by side, in every CI job and without a model; `qwen3` joins the refused names it checks.
  - The generator keeps every merge that forms a substring of a text, so the file gives the ids the whole vocabulary gives however llmx cuts the text.
    `tests/reference_generator.py` holds it to that and to the files' token types on a made-up vocabulary, and holds the committed golden to the generator's texts, commit and digests.
  - The reference is a second isolated venv on the Linux machine, which the other qwen35 branches share: Python 3.12.13, torch 2.5.1+cpu, transformers 5.17.0, tokenizers 0.23.2, huggingface_hub 1.33.0, safetensors 0.8.0, numpy 2.2.6, Jinja2 3.1.6 (`docs/ASSETS.md`).
- **Gate:**
  - The 37 texts through the real files give HF's ids, 37 of 37 on each, and each control token's text gives its one id, alone and side by side: Qwen3.5-0.8B, 2B, 4B and 9B Q4_K_M, Qwen3.6-27B Q8_0, Qwen3.6-35B-A3B Q4_K_M and Qwen3.8-27B Q8_0.
    The Qwen3.5-122B-A10B UD-Q8_K_XL file is refused by the GGUF reader for its F16 tensors before its tokenizer is read.
  - All 45 distinct Qwen3.5, 3.6 and 3.8 GGUFs on the Linux machine hold identical tokens, types and merges: `tokenizer.json`'s, plus 7 control tokens and 243 padding entries.
    They are 18 qwen35 and 11 qwen35moe models, 2 DFlash drafters (`dflash`) and 14 files of Qwen3.8-Flash-Next (`qwen4exp`) and its MTP heads.
    Every one names EOS 248046; some also name BOS 248044 with `add_bos_token` false or absent, and llmx adds no BOS.
    The golden's tokens, control tokens and added-token types equal those of the 8 files checked: the 7 above and the first Qwen3.8-Flash-Next Q8_0 shard.
  - The golden regenerates byte for byte in the venv above, offline, from the pinned files.
    A file the generator builds the same way from 11 other texts, 7 of which differ from HF on the 0.8B file (curly quotes, U+3000 runs, U+00BD, U+00BB, a decomposed accent, symbols before punctuation, no-break spaces), gives the 0.8B file's ids on all 11 and on the 7 control tokens side by side, so the fixture cannot hide a pretokenizer difference.
  - A sweep of the Wikitext test file in 33 pieces and 18 multilingual paragraphs (297,747 HF tokens) on the 0.8B file: 48 of 51 texts equal HF, and 3 differ in one place each, a space before U+00BD twice and U+00BB followed by a comma.
    On Qwen3-0.6B the same sweep gives 42 of 51, with Thai, CJK period and newlines, U+3000 and NFC among the differences.
  - That sweep understates English text with typographic quotes, which Wikitext does not use.
    Of its 501 paragraphs that hold quotes, 270 differ from HF on the 0.8B file and 269 on Qwen3-0.6B once the quotes are made curly, against 1 of 501 with straight quotes; 243 of the 501 hold a closing quote before a period or a comma, which is one HF token and two in llmx.
  - Qwen3 is unchanged: main b7b585f and this branch give the same output on Qwen3-0.6B Q8_0 for the 37 golden texts, the 8 control-token texts, the 51 sweep texts and the 501 curly-quote paragraphs.
- **Known differences** (`docs/src/tokenizer-tokenizer.md`): no NFC; non-ASCII spaces, numbers, punctuation and symbols read as letters, most often a closing curly quote before a period or a comma, as the typographic-quote sweep above measures; the 7 control tokens 248070 to 248076 that the GGUF files add, each one id in llmx and several in `tokenizer.json`.
  Exact Unicode classes and NFC come later as their own branch (decided 2026-09-25); the typographic-quote rate above bears on its priority.
- **Gotcha for the qwen35 references:** transformers 5.17.0 loads Qwen3.5 as `Qwen2Tokenizer`, which puts back the qwen2 regex and adds the 7 control tokens, so its ids differ from `tokenizer.json` on Thai and Devanagari and on those tokens; `gen_logits` tokenizes through `AutoTokenizer`, so a qwen35 golden made that way must be checked against `tokenizers`.
- **Tests:** on main b7b585f, whose code main bd73203 keeps, on Linux, CPU only: CTest 22/22 and 16 of the 17 suite components pass, the tokenizer's 20 Qwen3 fixture cases and the Qwen3 logit and PPL gates among them.
  The server component timed out in its uncapped case, three replies of about 1,020 tokens under a 600 s limit, at a host load average near 48, and main failed the same way in the same run; this branch changes neither the server nor the model.
  On Windows the MSVC build compiles with no warning, and the tokenizer and reference-generator components pass.
  Only the tokenizer's constructor changed, no hot path, so no speed run was made.
- **Left:** the qwen35 CPU model branch runs this golden against its pinned 0.8B file in `tests/baseline.py`.

## Tokenize routes on the server (2026-09-25, branch feat/server-tokenize)

- **Goal:** the loaded model's tokenizer over HTTP with the CLI's results, so a client counts a prompt as the server reads it.
- **Done:** `POST /v1/tokenize` (`{"text"}`, or `{"messages"}` in its place, replying `{"tokens", "count"}`) and `POST /v1/detokenize` (`{"tokens"}`, replying `{"text"}`) in `server/api.hpp`, answered on the connection thread without the scheduler. A text is encoded as a prompt is, through the generating routes' `encode`, a special token's text reading as that token and no template applied; `messages` go through the chat routes' `render_messages`, so the ids are what a chat request with them reads. Each id is read through `integer_value`, the check `integer` makes, and detokenized text gets `utf8_sanitize`, which gives a whole reply's ids back as its text. Every POST route reads its body through `body_of`, and the native reply's id list and the tokenize reply's come from `ids_json`. `tools/server_load.py` counts prompts through `/v1/tokenize` where a server has no `/tokenize`, so on `llmx serve` a prompt's length is exact by count and no longer rests on every word of its list being one token.
- **`add_special`:** no route adds a start or end token to a prompt (the Qwen3 files set `tokenizer.ggml.add_bos_token` false, and a prompt reads the ids `encode` gives), so `add_special` is not read: whatever its value, the ids are the ones a prompt reads, which is what a client that sends it to count a prompt asks for. Honouring the file's `add_bos_token` in this route alone would count a token the model never reads, and refusing a value that is not true or false would be a second rule for boolean fields beside `flag`, which reads `stream` and `include_usage` as false unless they are true.
- **Review fixes:** a text the tokenizer cannot encode was a 500 on `/v1/tokenize` and a 400 on the generating routes, and `encode` now gives all of them the 400 (the Qwen3 files hold every byte token, so only a file without one reaches it). `detokenize` read ids with its own copy of `integer`'s check, now `integer_value`, which `integer` calls. `add_special` is no longer refused when it is not true or false. The tests check the routes while the queue is full, a text past a 512-token context, the refusal on a synthetic vocabulary without the byte token `q`, and that the vocabulary edge they take from `/v1/models` is the tokenizer's, through the CLI. `docs/CI.md` lists the new checks, and `docs/SERVER.md` keeps its error lines out of the tokenize paragraph.
- **Gates** (at 8ed1a04, this commit before its gate lines and its rebase onto main bd73203, which changes only docs, against main b7b585f on the MI50 host's CPU, in a container of 6 CPUs with no GPU device, the load average 31 to 50 from other work): the unencodable-text check fails on bf6176d, the branch before the review fixes, where `/v1/tokenize` answered 500 as a text and as messages, and passes here; builds with Vulkan on and off, clean, 0 warnings, and the `build.bat` binary on Windows without a warning; CTest 25 of 25 with Vulkan on (`backend-vulkan` and `vulkan-lifetime` skipped with no device) and 22 of 22 with it off; `run_tests.py --no-perf-floor --require-tools --require-baseline --device cpu` passes 16 of 17 components, and `server`, whose `check_uncapped` request on the Q8_0 passed its 600 s timeout at a load of 45 to 50 after the tokenize checks and the limits had passed, passes run alone at 31 to 38; `tools/server_mix_check.py` on 0.6B Q8_0 finds all requests matching (16 alone, 0 of 16 differing together, 0 of 12 skewed with 4 clients leaving early, 4 checked against the CLI). On 0.6B Q8_0 and Q4_0, `generate` greedy and seeded, `logits`, `tokenize` and `detokenize` are byte-identical to main in stdout, stderr and exit status, and the help pages differ only in the version line and `serve`'s new route line; 34 server requests a model on the other routes (native and compatible, whole and streamed, greedy and seeded, the refusals, the whole-number fields `integer` reads, a `stream` that is not a boolean, `/v1/health` and `/v1/models`) give replies byte-identical to main's with `created` and `timings` masked, 68 of 68. `tools/server_load.py` counts its prompts through `/v1/tokenize` on `--api llmx` and `openai` against the branch's server and by its two probe requests against main's.
- **Hosted CI:** all six jobs passed on eee2a84, the commit before its rebase onto main dc31dd4, which changes only docs and the tokenizer.

## Prefill kernels on the MI50 (2026-09-25, branch perf/prefill-kernels)

- **Goal:** single-card prefill on the MI50, where the integer-dot tile was 89.4 percent of an 8B Q8_0 512-row pass. Three changes measured apart earlier the same day (branches `perf/tile-staging-loads`, `perf/q8-tile-step4`, and `feat/quantize-x8-vec4` to `feat/x8-row-pad`) are combined on main 34bebc3 in that order, each re-applied on top of the one before and gated again as a layer. After review the branch was rebased onto main aea6e34 and gated again there (Rebased, below).
- **Commits:**
  1. Every staging load of a step goes out before the first wait. The disassembly had every load under a branch the compiler could not prove uniform, each followed by a wait for everything in flight, so a step of two blocks waited for ten memory round trips in a row. A load out of range now reads a valid address and its zero is selected at the store.
  2. The math takes word w of all of a thread's rows against word w of its four columns, so the block's dot sums stay live and a column word is read once: 91 registers before scheduling where holding four columns' words took 115.
  3. Q4_K, Q5_K, Q6_K, Q4_0 and Q4_1 stage the same way, the K-quants' sub-scales unpacked from the 16-byte head of the 256-value block held in registers; each module carries only its own types.
  4. Q6_K takes the math of 2 in groups of four rows, since eight rows' two half sums would take 64 registers.
  5. Q8_0 gets its own module (`LLMX_Q8`): one float scale per row and per column, no minimum term, no type branch; the general module no longer compiles a Q8_0 path.
  6. The Q8_0 module stages four blocks a step and unrolls their math.
  7. `quantize_x8` takes four values a lane.
  8. The RMS norm splits a row only until the pass has four workgroups a compute unit.
  9. The norm, the SiLU and the wide attention write the tile's 8-bit copy in place of the twin when one tile call reads the whole batch next, and `quantize_x8` does not run.
  10. Rows of the copy are padded to an odd number of 128-byte runs.
  11. `alloc` and `adopt` drop the tags of the twin, the 8-bit copy and the routed grouping, as a write from the host does: a tag names a buffer by its handle, which a buffer made after one is freed can take over, and a matmul of the same shape over the new buffer with no dispatch before it read the freed buffer's copy. The model never met this (its buffers outlive its passes and every pass dispatches first); main's twin tag had the same gap.
- **Changed from the separate branches:** 1 to 4 are that branch's code with its `first` in the Q4_0/Q4_1 staging renamed `qoff`, since it shadowed the routed call's first column, and the `store_scales` comment saying that a step of two divides the threads per row at every tile height. 5 and 6 are rewritten onto 1's staging rather than applied: the Q8_0 staging is its own block with no type branch; a row's first thread takes the half scale from words it already loads (the extra word of a run is the one after the thread's share where a block straddles a word boundary and the one before where it does not, which for the first thread holds the scale), so the module has no scale load and never uses `store_scales`, whose step must divide the row's threads and a step of four on the 128-row tile would not; each of a column's four threads loads one activation scale and the column's scale array always has four slots, so a step of two stores without a branch. 9 and 10 are that branch's code with the tile's activation addressing moved onto 1's loads, the measurements in two code comments moved here, and `docs/src/backends-vulkan.md` updated.
- **Exact:** no result changes, by argument and by the gate below.
  - Staging (1, 3): only data movement; a thread stages the same words and scales as main, and an out-of-range value is zero as it was.
  - Math (2, 4): each dot sum adds a block's eight words (Q6_K: four per half) in the same order, so it is the same int32 (the largest Q8_0 sum is about 520k), and each output's float expression is main's operand for operand, blocks in the same order within the same parts.
  - Q8_0 module (5, 6): main computes `acc + (ws.x*c.x*float(s) - ws.z*c.y)` with `ws.z` zero for Q8_0; whichever contraction a compiler makes there the result is `acc + round(round(d*c)*s)`, since the subtraction stands between the product and the add. The module computes exactly that with `precise` products; in its SPIR-V all eight float multiplies carry NoContraction and none of its five float adds, the four accumulations and the output's, does, while the integer adds, multiplies and divisions `precise` also marks are exact either way. Blocks past a part's end in a step of four add +0 to an accumulator that starts at +0 and cannot become -0.
  - `quantize_x8` (7) and the producers (9): the per-value arithmetic is unchanged and neither a block's maximum nor its integer sum depends on the order it is taken in; the norm's vec4 form is `(src*r)*w`, only correctly rounded multiplies. The SiLU's vec4 and scalar forms use `exp` and a division, whose precision Vulkan leaves to the driver, so their equality rests on RADV lowering both alike, which the backend test checks.
  - Norm split (8): every chunk sums the whole row in the same tree order. Padding (10): addresses only.
  - Batch invariance: the tile height, which producer writes the copy and the norm's chunk count follow the batch, and none changes an output's arithmetic; `backend-vulkan`'s invariance checks pass.
- **Gates** (one MI50, rocm-smi GPU[1]): after each layer, all 151,936 logits at the last 32 positions of the 248-token excerpt (`logits --file --last 32 --top 151936`, printed to six decimals) and 64 greedy tokens byte-identical to main on Qwen3-8B Q8_0, 0.6B Q8_0, 8B Q4_K_M, 30B-A3B Q4_K_M, 0.6B Q5_K_M and 0.6B Q4_0; 32 greedy tokens after the 6823-token prompt on 8B Q8_0 identical as well. `llmx-backend-vulkan-test` passes on every layer, on the Q8_0 module at a step of two, and on each of the six other commits built alone. No gated file has Q4_1 weights, so commit 3's Q4_1 staging is checked only by `backend-vulkan`, against the CPU within its tolerance. The binaries measured were built on 34bebc3: layer 1 from the branch's 61268ea, the other arms from f325b99, ae84e69 and d323a09 and the per-commit backend tests from f38cc18, 68e8412 and 76b0fae, each of which differs from the branch's commit for the same step by one comment line. The tip's 58 SPIR-V modules are byte-identical to the measured final build's, before the rebase and after it.

  | layer (tip measured) | llmx binary sha256 | logits and 64 tokens, six files | 32 tokens after 6823 | backend-vulkan |
  |---|---|---|---|---|
  | main 34bebc3 | 989dce63d6e25623 | reference | reference | pass, 25,373,118 outputs |
  | 1 (commits 1-4) | 3c88479a3a3712a4 | identical | identical | pass, 25,373,118 |
  | Q8_0 module at a step of two | 08464d5541c6d068 | identical | identical | pass, 25,373,118 |
  | 1+2 (commits 1-6) | 4d388fe83bbc21b3 | identical | identical | pass, 25,373,118 |
  | 1+2+3 (commits 1-10) | 2e9b16334fc8b535 | identical | identical | pass, 25,827,774 |

  Windows, before the rebase: the CPU and Vulkan trees build with MSVC, CPU CTest 22/22 and the CPU suite pass (no GPU used).
- **Rebased** onto main aea6e34 after review. The one conflict, the wide attention's tile count, keeps main's `kAttentionTileRows` beside the branch's `tile_reads` flag and push constants. The review's fixes: the comments on `kper` (a multiple of two only when a call is split) and on the Q8_0 staging's extra word, a comment broken mid-sentence, commit 11, and a `backend-vulkan` check of a routed down projection over the MoE SiLU's per-entry copy against its own pass, with every output in those checks allocated before its producer runs, since a new buffer now drops the copy. A check of a matmul over a new buffer after a freed buffer's copy was tried and dropped: a diagnostic build showed RADV giving the new buffer another handle, so it passed without commit 11 as well.
  - Identity, one MI50 (rocm-smi GPU[1]), with `logits --file PATH` as main's CLI now takes it: the code tip 2b0b1d1 against main aea6e34 gives byte-identical logits and 64 greedy tokens on the six files, the same bytes as before the rebase, and identical 32 tokens after the 6823-token prompt; the tip's 58 SPIR-V modules equal the measured final build's.

    | build | llmx binary sha256 | logits and 64 tokens, six files | 32 tokens after 6823 | backend-vulkan |
    |---|---|---|---|---|
    | main aea6e34 | 42cb84082de4dc6c | reference | reference | pass, 25,401,660 outputs |
    | code tip 2b0b1d1 | d57cf12017b9573b | identical | identical | pass, 25,868,604 |

  - Mixed batches: `tools/server_mix_check.py` on 1124b91 (the tip's sources but for the dropped check) over the 6823-token prompt's text, 16 requests of about 30 to 2700 tokens alone, all at once, and staggered with four clients leaving, then four through the CLI, on 8B Q8_0 and 30B-A3B: every request gives its tokens alone. So a batch in which a decode row or a short prompt sends the copy through `quantize_x8` gives what the prompt gives alone, where its producers write the copy.
  - Main against 1124b91 on rocm-smi GPU[0] at high clocks, two interleaved rounds, tok/s: 8B Q8_0 pp512 935.6 / 934.3 against 1330.1 / 1326.7 (+42 percent), pp4096 751.9 / 761.8 against 987.4 / 1002.3 (+31); 8B Q4_K_M pp512 877.6 / 877.7 against 1229.9 / 1233.4 (+40); 30B-A3B pp512 1245.1 / 1243.9 against 1625.7 / 1628.9 (+31); 0.6B Q8_0 pp512 9324 / 9294 against 13251 / 13236 (+42). Decode moved within the noise of a host other runs loaded (load average 8 to 23): 30B-A3B tg128 read 136.4 and 132.4 on main against 132.6 and 131.4, where the two layer binaries below, whose decode kernels are the same, read 130.2 to 134.7. The tip reads within 2 percent of the final arm's means in the session below (8B Q8_0 pp512 1328 against 1318), so neither the rebase nor commit 11 costs anything.
- **Measured** (one session on rocm-smi GPU[0] at high clocks, the four arms interleaved with their order rotating each round, three rounds since two differed by more than 1 percent in several cells, `bench --model M --p P --n 128 --r 3`, 32 generated tokens at 64 rows; tok/s per round; the reference is mx-llama.cpp's gfx906 image, its ROCm build, on the same card, `llama-bench -ngl 99 -fa 1 -p 64,512,4096 -n 128 -r 3 -lm dio`, once between rounds 2 and 3):

  | cell | main | layer 1 | 1+2 | 1+2+3 | reference |
  |---|---|---|---|---|---:|
  | 8B Q8_0 pp512 | 935.7 / 915.3 / 910.9 | 1134.1 / 1118.0 / 1109.4 | 1270.2 / 1260.6 / 1224.5 | 1324.6 / 1323.2 / 1307.1 | 1323.1 |
  | 8B Q8_0 pp4096 | 761.4 / 734.1 / 729.6 | 885.3 / 855.3 / 862.6 | 968.0 / 934.7 / 926.6 | 992.3 / 970.1 / 965.5 | 1167.0 |
  | 8B Q8_0 pp64 | 810.6 / 810.2 / 810.0 | 1025.7 / 1031.2 / 1032.8 | 1156.0 / 1136.4 / 1156.2 | 1168.7 / 1118.6 / 1168.5 | 859.7 |
  | 8B Q8_0 tg128 | 67.5 / 68.5 / 68.4 | 67.3 / 68.3 / 68.3 | 67.3 / 68.4 / 68.4 | 67.4 / 68.4 / 68.3 | 71.3 |
  | 8B Q4_K_M pp512 | 872.6 / 858.8 / 853.0 | 1181.2 / 1151.0 / 1140.2 | 1179.3 / 1162.6 / 1150.5 | 1230.3 / 1218.8 / 1215.7 | 813.5 |
  | 8B Q4_K_M pp4096 | 710.3 / 695.9 / 699.2 | 891.2 / 880.2 / 877.7 | 892.9 / 877.1 / 877.0 | 935.8 / 913.8 / 912.4 | 752.0 |
  | 8B Q4_K_M pp64 | 739.5 / 723.5 / 719.4 | 1039.7 / 1034.0 / 1014.4 | 1040.5 / 1024.3 / 1020.8 | 1056.1 / 1041.4 / 1055.7 | 255.8 |
  | 8B Q4_K_M tg128 | 100.2 / 100.2 / 99.8 | 100.5 / 100.2 / 100.1 | 100.3 / 100.0 / 100.5 | 100.1 / 99.8 / 99.9 | 87.5 |
  | 30B-A3B Q4_K_M pp512 | 1211.9 / 1207.2 / 1200.5 | 1586.8 / 1547.9 / 1546.7 | 1586.6 / 1506.5 / 1549.7 | 1631.7 / 1589.8 / 1585.9 | 1155.3 |
  | 30B-A3B Q4_K_M pp4096 | 890.9 / 917.3 / 902.1 | 1041.0 / 1027.0 / 1078.7 | 1031.3 / 1022.7 / 1025.0 | 1067.8 / 1099.1 / 1057.3 | 1015.5 |
  | 30B-A3B Q4_K_M pp64 | 431.1 / 440.4 / 439.6 | 579.8 / 582.4 / 582.5 | 572.4 / 584.7 / 584.5 | 575.1 / 585.1 / 580.7 | 403.3 |
  | 30B-A3B Q4_K_M tg128 | 137.0 / 136.8 / 136.3 | 137.3 / 136.9 / 136.4 | 137.2 / 136.6 / 136.7 | 136.5 / 136.9 / 137.0 | 114.5 |
  | 0.6B Q8_0 pp512 | 9369 / 9357 / 9370 | 11509 / 11509 / 11464 | 12482 / 12424 / 12423 | 13378 / 13394 / 13368 | 7200 |
  | 0.6B Q8_0 pp4096 | 4508 / 4630 / 4631 | 4929 / 5099 / 5099 | 5150 / 5281 / 5281 | 5184 / 5436 / 5436 | 5948 |
  | 0.6B Q8_0 pp64 | 5408 / 5349 / 5362 | 6444 / 6462 / 6447 | 7135 / 7098 / 7109 | 7090 / 7070 / 7099 | 5013 |
  | 0.6B Q8_0 tg128 | 353.8 / 353.6 / 353.7 | 356.3 / 353.5 / 356.0 | 355.0 / 352.3 / 343.6 | 354.2 / 354.6 / 352.6 | 323.4 |
  | 0.6B Q5_K_M pp512 | 8460 / 8442 / 8448 | 11144 / 11140 / 11150 | 11178 / 11165 / 11167 | 11869 / 11871 / 11843 | - |
  | 0.6B Q4_0 pp512 | 9282 / 9277 / 9285 | 11588 / 11583 / 11574 | 11598 / 11591 / 11588 | 12315 / 12327 / 12329 | - |

  - Most 8B cells at 512 and 4096 rows read 1 to 4 percent lower in rounds 2 and 3 than in round 1, on every arm, least on the final arm at 512 rows (its 8B Q8_0 pp512 by 0.1 and 1.3 percent), while 8B Q8_0 at 64 rows held on main and layer 1, so the steps are read per round. Layer 1 gives 8B Q8_0 pp512 +21.2, +22.1, +21.8 percent, 8B Q4_K_M +35.4, +34.0, +33.7, 30B-A3B +30.9, +28.2, +28.8. Layer 2 gives 8B Q8_0 +12.0, +12.8, +10.4 at 512 rows, +9.3, +9.3, +7.4 at 4096 and +12.7, +10.2, +11.9 at 64, and 0.6B Q8_0 +8.5, +8.0, +8.4 at 512; files without Q8_0 matrices are level (30B-A3B has none: no Q8_0 module runs in its profile). Its pp4096 read -0.9, -0.4 and -5.0 percent from layer 1 to 1+2 here, round 3 of layer 1 reading high, and four interleaved rounds of the same two binaries after the review read 1078.3 to 1078.7 tok/s on layer 1 against 1079.5 to 1080.1 on 1+2. Layer 3 gives +4.3, +5.0, +6.7 at 8B Q8_0 pp512, +4.3, +4.8, +5.7 at 8B Q4_K_M, +2.8, +5.5, +2.3 at 30B-A3B, +7.2, +7.8, +7.6 at 0.6B Q8_0 and about +6.2 at 0.6B Q5_K_M and Q4_0; at 64 rows, where the producers do not write the copy since `tile_reads` takes the highest of every type's thresholds, 96 rows for routed Q4_0, it is level (8B Q8_0 +1.1, -1.6, +1.1; 0.6B Q8_0 -0.6, -0.4, -0.1).
  - No layer loses its gain combined: layer 1 gives what it measured alone on main (8B Q8_0 pp512 +21 percent, 8B Q4_K_M +35), and layers 2 and 3 give more than they did alone on main (Q8_0 module and step of four +8 to +10 percent there, the 8-bit copy +2.9 to +3.2 percent at 512 rows), since the tile they now sit beside is faster. The norm's row split (8) alone on main took the norm from 8.8 to 5.9 ms and pp512 up about 0.5 percent. All ten measured commits are kept.
  - Decode is level on every file (tg128 within 1 percent of main in every round, apart from single 0.6B runs).
  - Against the ROCm reference on this card, 8B Q8_0 is now level at 512 rows (1318 against 1323 by the mean of rounds) and at 84 percent at 4096 (976 against 1167), from 70 and 64 percent on main; 8B Q4_K_M leads by 50 and 22 percent, 30B-A3B by 39 and 6 percent, 0.6B Q8_0 by 86 percent at 512 rows and trails at 4096 (5352 against 5948).
- **Device time of a 512-row pass** (`bench --profile`, one run per arm between rounds 2 and 3, ms):

  | file, arm | pass | tile | quantize_x8 | norm | SiLU | attention |
  |---|---:|---:|---:|---:|---:|---:|
  | 8B Q8_0, main | 547.6 | 489.6 | 11.22 | 8.71 | 5.47 | 20.80 |
  | 8B Q8_0, 1 | 452.2 | 392.8 | 11.47 | 8.74 | 5.52 | 21.73 |
  | 8B Q8_0, 1+2 | 402.5 | 343.5 | 11.42 | 8.93 | 5.49 | 21.39 |
  | 8B Q8_0, 1+2+3 | 386.0 | 344.1 | - | 3.59 | 4.67 | 21.79 |
  | 8B Q4_K_M, main | 583.3 | 525.1 | 11.23 | 8.72 | 5.46 | 20.79 |
  | 8B Q4_K_M, 1 | 431.7 | 372.7 | 11.35 | 8.83 | 5.47 | 21.26 |
  | 8B Q4_K_M, 1+2 | 430.6 | 371.5 | 11.36 | 8.82 | 5.47 | 21.32 |
  | 8B Q4_K_M, 1+2+3 | 414.4 | 372.9 | - | 3.54 | 4.65 | 21.26 |
  | 30B-A3B, main | 411.6 | 340.6 | 7.23 | 4.73 | 3.71 | 27.73 |
  | 30B-A3B, 1 | 324.7 | 252.4 | 7.56 | 4.87 | 3.71 | 27.73 |
  | 30B-A3B, 1+2 | 323.1 | 251.9 | 7.28 | 4.76 | 3.71 | 27.73 |
  | 30B-A3B, 1+2+3 | 314.1 | 252.6 | - | 2.63 | 3.29 | 28.03 |
  | 0.6B Q8_0, main | 55.3 | 37.0 | 2.46 | 1.54 | 1.24 | 9.28 |
  | 0.6B Q8_0, 1 | 45.1 | 26.8 | 2.46 | 1.54 | 1.23 | 9.28 |
  | 0.6B Q8_0, 1+2 | 41.6 | 23.4 | 2.47 | 1.54 | 1.24 | 9.27 |
  | 0.6B Q8_0, 1+2+3 | 38.7 | 23.4 | - | 1.04 | 1.12 | 9.40 |

  The tile is the sum of the integer-dot tile modules (on Q4_K_M the Q4_K and Q6_K ones, on 30B-A3B the routed and dense ones), the norm and SiLU the builds that write a copy or a twin.
- **Kernel statistics** (driver, from `backend-vulkan`: registers, registers before scheduling, shared memory, waves per SIMD):

  | module | main | layers 1+2+3 |
  |---|---|---|
  | `matmul_tile_q_tall` (128 rows) | 128, 115, 17,408 B, 2 | 128, 99, 17,408 B, 2 |
  | `matmul_tile_q` (64 rows) | 84, 82, 11,264 B, 3 | 64, 60, 11,264 B, 4 |
  | `matmul_tile_q_small` (32 rows) | 84, 76, 8,192 B, 3 | 48, 45, 8,192 B, 5 |
  | `matmul_tile_q6_tall` | 128, 119, 17,408 B, 2 | 128, 120, 17,408 B, 2 |
  | `matmul_tile_q6` | 128, 86, 11,264 B, 2 | 84, 76, 11,264 B, 3 |
  | `matmul_tile_q6_small` | 84, 81, 8,192 B, 3 | 64, 52, 8,192 B, 4 |
  | `matmul_tile_q8_tall` | - | 128, 86, 27,648 B, 2 (a step of two: 128, 86, 14,336 B, 2) |
  | `matmul_tile_q8` | - | 84, 52, 18,432 B, 3 (64, 52, 9,728 B, 4) |
  | `matmul_tile_q8_small` | - | 64, 38, 13,824 B, 4 (48, 38, 7,680 B, 5) |
  | `quantize_x8`, `silu_mul_x8`, `rms_norm_rows_x8` | 12, 16, 20; 10 waves | 24, 24, 24; 10 waves |
  | `attention_tile_kv16` | 128, 125, 18,432 B, 2 | unchanged |

  Layer 2 leaves the general module's statistics as layer 1 had them, without its Q8_0 path.
- **The step of four, re-derived on layer 1** (the same card, two interleaved rounds, tok/s; the 8B Q8_0 pass's Q8_0 tile from one profile run):

  | cell | layer 1 | Q8_0 module, step of two | step of four, unrolled |
  |---|---|---|---|
  | 8B Q8_0 pp512 | 1134.4 / 1130.8 | 1230.3 / 1221.4 | 1277.2 / 1269.5 |
  | 8B Q8_0 pp4096 | 888.8 / 885.8 | 943.0 / 930.3 | 969.7 / 958.6 |
  | 8B Q8_0 pp64 | 1029.8 / 1031.7 | 1131.8 / 1128.1 | 1156.3 / 1153.4 |
  | 0.6B Q8_0 pp512 | 11465 / 11460 | 12428 / 12354 | 12467 / 12406 |
  | 0.6B Q8_0 pp4096 | 5098 / 5098 | 5278 / 5277 | 5281 / 5280 |
  | 8B Q8_0 tile, 512-row pass | 393.5 ms | 360.1 | 344.2 |

  The step of four still pays after the word-major math (+3.9 percent at 8B Q8_0 pp512, +2.9 at 4096, +2.2 at 64, the 0.6B file level) and still fits the 128-row tile in 128 registers and two waves; the 64-row and 32-row Q8_0 tiles go from 64 and 48 registers to 84 and 64 (four and five waves to three and four) without a loss on the 0.6B file.
- **Measured and not taken** (on the three separate branches, on main 1a0d4be or a2b732f, one MI50 at high clocks, two interleaved rounds unless stated):
  - Staging diagnostic, 8B Q8_0 pp512 tile (`research/tile-d2-diagnostic`): main 490.0 and 490.9 ms, math alone (staged once, the math every step) 301.5 and 302.6, staging alone 362.4 and 362.0 at 28 registers; after commits 1 to 3 the full tile 393.1 and 392.1, math alone 264.8 and 266.0, staging alone 283.2 and 283.2.
  - Loading a step ahead of the math (`research/tile-prefetch-rejected`): all loads with row-at-a-time math 651 ms, 133 registers before scheduling and one wave per SIMD; activations only 439 to 441 ms against 424 to 426; on the word-major math 422.7 and 424.5 against 391.0 and 392.4, with the Q6_K 128-row tile at 224 registers.
  - Two shared-memory buffers and one barrier a step (`research/tile-double-buffer-rejected`): 428.4 and 429.6 ms against 423.6 and 426.0; it fits in 30 KB with the row scales trimmed to pairs but is no faster.
  - A row tile's column tiles started together, to share weights through L2 (`research/tile-column-order-rejected`): 8B Q8_0 tile 433.4 and 435.1 ms against 391.7 and 394.3; Q4_K tile 365.7 and 366.4 against 312.5 and 312.7; Q6_K 57.1 and 57.2 against 55.0 and 55.1.
  - The word-major math in groups of four rows for the general types: 394.0 and 394.3 ms against 391 to 392.
  - Q6_K's word-major math over all eight rows at once: its 128-row tile 59.0 ms against 55.0 in groups of four.
  - The Q8_0 module's step of four left a loop (`research/q8-tile-step4-rolled-rejected`): 8B Q8_0 pp512 964 and 972 tok/s against 949 to 983 at a step of two, tile 467.6 ms against 468.0; 0.6B Q8_0 pp512 9780 and 9782 against 10007 to 10030 (-2.3 percent), 8B pp64 842 and 843 against 866 and 869 (-3 percent).
  - Skipping the zero-filled blocks of a part's last step of four: not measured; it matters only where a part's block count is not a multiple of four, on 8B only for q, k and v at 64 rows, estimated under 1 percent there.
  - `quantize_x8` at eight values a lane: 8.3 to 8.4 ms, the same as four.
  - A workgroup over 32 columns of one block, writing one run (`research/quantize-x8-column-run-rejected`): 9.0 ms against 8.2 to 8.3.
  - The SiLU writing nothing and `quantize_x8` after it: 3.81 plus 4.80 ms against 8.75 for the SiLU writing the copy (8B Q4_K_M), a tie; the fused form is kept since it removes a pass.
  - Four scalar loads a lane in the producers in place of vec4 buffer views: level, so the views are kept for clarity.
  - Padding the copy's rows by 1, 4, 8, 16, 32 and 64 blocks: `quantize_x8` 4.50, 4.05, 4.21, 4.19, 4.25 and 4.51 ms against 8.28 unpadded; four blocks, an odd number of 128-byte runs, is kept.
  - Ruled out by the tile's disassembly, not measured: fusing the dot's accumulate, since the 256 `v_dot4` of a thread's block already chain their accumulator, and wider weight loads, since a thread's weights already load as a `dwordx4` and a `dword`.
- **Left before merge:** the Radeon VII gate (its profile has no `prefer_integer_dot`, so the tile modules, `quantize_x8` and the producers' copy never run there, but the norm's split and the attention epilogue's sixth binding and push constants do), and the merge-gate cells against the pinned Vulkan reference build on both cards (pp247, tg32). `backend-vulkan` checks each producer's copy against the pass's, the routed down projection's too, and a copy dropped by a host write or a kernel; it does not yet assert that the producer path ran (a diagnostic build printed the copy in use in the dense norm, SiLU and attention checks) or cover the overlap checks after the float tile and routing. The Windows build is to be repeated on the rebased branch with that gate.
## The bench's KV pool holds its sequences (2026-09-25, branch fix/bench-seqs)

- **Why:** `bench --model tiny-f32.gguf --p 4 --n 2 --r 2 --seqs 2` exited with `KV cache: block budget exhausted`, and `--seqs 1` ran. The pool was one model context, the budget `generate` gets, since only `serve` reads `--ctx-size`; the tiny model's 16-token context fills one block, and each sequence takes whole blocks. Batched decode also made its sequences without clearing the one sequence, so the last prompt's blocks stayed beside them.
- **Done:** `PlacementRequest` carries the histories a caller holds at once and the tokens each reaches. `place_model` counts them in each backend's blocks, each up to the model's context, so a run that asks for more than the context does not grow the pool before it is refused; where the budget would leave a storage short, it becomes what they take in the largest blocks, which the fit then counts, and otherwise it is unchanged. `bench --model` names its sequences and what each holds: a batched one its prompt and its tokens, the one sequence its depth and the longer test. Every bench test starts from a cleared history, batched decode included. With either change alone the tiny model still fails.
- **Exact:** on Qwen3-0.6B Q8_0 on the CPU, `generate -n 64 --temp 0` without its `pp:` and `tg:` lines, `logits --top 20`, `perplexity` batched and `--per-token`, and `bench --model` with `--seqs 1`, with `--depth 32` and with `--seqs 4`, rates masked, give the same bytes as main f1a979c. No difference is intended.
- **Tests:** `f32` runs `bench --model --seqs 2` on the tiny model and requires `pp4` and `x2 tg2`, two runs each, at finite rates; on main it fails with the budget error. From the merged cli-surface review, `check_logits_input` now requires each `--last` row printed once and in order, the `--ubatch 5` rows byte-identical to the one-pass rows, and the `--then-ids` rows byte-identical to them at positions 3 to 12, which they are on the CPU. `placement` checks that the budget grows only where it falls short, that one history of one block leaves the two-block budget as it is rather than shrinking it, and that a history past the context is counted at the context, and runs three 100-token histories in one pass over a two-block context on one backend and over two. CTest 22/22 and the CPU suite pass on Windows; the Vulkan tree compiles.
- **Left before merge:** the suites on the Radeon VII and on an MI50, where `f32` now runs `bench --seqs 2` over 64-token blocks, and `bench --model --seqs` on a split.

## The serving load tool: request rates, fixed lengths, per-request records (2026-09-25, branch tools/server-load-v2)

- **Why:** phase 3's gate (`docs/MULTI-DEVICE.md`, Order of work) is the server at 1 to 64 users and a rate sweep against the references on the same cards, and the comparison with vLLM's serving benchmark (random prompts of a fixed length, replies of a fixed length with the end of text ignored, Poisson arrivals) needs the same load and figures. `tools/server_load.py` sent eight short prompts, let a reply stop at its end of text, and reported TTFT and ITL at p50 and p99, tok/s and req/s.
- **Done:** the tool keeps its flags, defaults and first eight columns, and gains: closed-loop levels 1 to 64 by default with `--num-prompts` a level; open-loop `--rate` levels with Poisson arrivals from `--seed` (`inf` for all at once); `--input-len` and `--input-len-range LO:HI` prompts of an exact token length, a different one per request; `--output-len` replies with `ignore_eos` sent; TTFT mean, p50, p90 and p99, TPOT mean, p50 and p99, e2e p50 and p99, total tok/s, completed, failed and short counts and the mean reused prompt tokens; `--warmup`, failures counted and listed by reason; `--json` with every request's record; `--self-test`, run by the suite as component `server-load` (about 3 s, no model). Prompt lengths are counted through the server's `/tokenize` route where it has one, else by two probe requests that show every word of the list is one token (it is in Qwen3's vocabulary).
- **Review fixes:**
  - A closed level shows the round with the most output tok/s among the rounds with no failed request, and the notes below the table list the failures of every round; before, a faster round hid a slower round's refusal from the table.
  - The tool exits 3 when any timed request failed, after the table, the notes and `--json`; a 503 among the failures adds a line on `llmx serve --max-queue`, and USAGE and the docstring say to start the server with `--max-queue` at least `--num-prompts` for an open level, since on its defaults 35 of the default 100 at `inf` were refused.
  - The prompt lengths come from the seed alone, before any prompt is built, and the words from a second stream keyed by level and round, so the rounds of a level and two servers that count a start token differently get the same lengths; before, the rounds drew different lengths and best-of favoured the lighter one.
  - TPOT is (end - first token) / (tokens - 1), the end being the reply's last event, as the brief and vLLM's benchmark ((latency - TTFT) / (output tokens - 1)) have it; the first version used the last token's arrival.
  - `--timeout` keeps the earlier tool's meaning, 600 s with nothing arriving, and `--total-timeout` (21600 s, as vLLM's benchmark allows) bounds a whole request; the first version made `--timeout` a 600 s total, which failed long queued requests and cut the slower server's tail.
  - The stream is read by a function over (time, line) pairs: `/completion` ends at its stop event, not at the stream's close; an `error:` event fails the request with its message; events past a reply's own token count (llmx's held piece of a split character on `/v1/completions`) are dropped.
  - The table's `reused` column is the mean reused prompt tokens, since every prompt starts with "the" and a server that matches token by token reuses one or two tokens a request, which a hit count would count as a hit; `/completion` keeps `cache_prompt: false` and the OpenAI route leaves every server's cache at its default, which USAGE says.
  - Every request records its connect time and, in the open loop, its send lag; the notes flag lag at the 99th percentile over 10 ms or a twentieth of the mean arrival gap, and connects over 100 ms, as a burst past a listen backlog gives. The open loop starts every request's thread before the level, each sleeping to its time, so starting threads no longer delays a burst's sends. A prompt whose tokenize count cannot reach its length is noted.
  - The self-test checks the stream reading of all three APIs exactly on made-up timelines (events of several tokens, a held piece, error events, a missing last event, a reply of no token) and bounds the socket runs' times only from below and loosely from above; it no longer depends on the machine being idle.
- **Checked:**
  - The reviews' reproductions, against a scripted server in the same process: `/completion` events of 1, 3 and 1 tokens at 100, 300 and 400 ms with the stop event at 420 give TTFT 105.9 ms, ITL 200.0 and 101.4, TPOT 80.2 (80 by hand) and e2e 426.7; the same stream held open 500 ms after its stop event gives e2e 422.0 (925 before); llmx's held piece gives 4 events for 4 tokens and ITL 99.5, 99.7 and 100.3; an `error:` event fails with its message. Two rounds at two users with a 503 in the first show the second with fail 0, list `conc 2, round 1 of 2, not the one shown: 1 failed: HTTP 503` and exit 3. Rounds and servers with and without a start token get the same lengths.
  - The self-test passes 8 of 8 runs with 24 spinning processes on the 16 logical CPUs (10 to 18 s each), where the first version failed 4 of 4 under the same load, and each of 16 mutations of the tool fails it: TTFT from the second token, TPOT from the last token, the finish chunk counted, a held piece kept, `/completion` read past its stop, `error:` lines skipped, lengths drawn with the words, the best round by tok/s alone, exit 0 on failures, e2e at the first token, an unshown round's notes dropped, the idle limit ignored, `[DONE]` ending a native stream with no `done`, the health counters ignored, the send lag not recorded, a 503 read as a stream.
  - Send lag against a server in another process, with the CPU busy with other work, median of three: threads started at their send time lag 60 to 490 ms at the median at `inf` with 100 and 200 requests; threads started before the level lag 9 to 18 ms, and 1.4 to 2.6 ms at 20 and 50 requests a second either way. A burst of 64 on a listen backlog of 5 has 48 connects over 100 ms (709 ms at most), noted; 200 at once on `llmx serve`, whose backlog is 64, connect in 70 ms at most.
  - `llmx serve` on the CPU with Qwen3-0.6B Q8_0 on the Windows machine: the earlier invocation (`--concurrency 1 2 --tokens 8`) keeps its first eight columns; closed at 1, 2 and 4 users with 64-token prompts and 16-token replies on `/v1/generate` and `/v1/completions`, open at 0.5, 2 and inf on `/v1/completions` and at 1 and inf on `/v1/generate` with 16 to 96 tokens, and 200 at once with `--max-queue 256`: every request completed at its length, every record's TPOT is (e2e - TTFT) / 15, the rounds share their lengths, all 88 prompts rebuilt from the seed and recounted with `llmx tokenize` have their target length, and `/v1/completions` reports each one's. On `--max-seqs 2 --max-queue 2`, 8 at once fail 5 with the 503 reason and the queue line and exit 3; `--timeout 0.3` and `--total-timeout 0.4` fail requests with their own reasons. Other work shared the CPU, so these runs check the tool, not llmx's speed.
- **Gaps in llmx the tool meets, proposed and not changed here:** the routes read no `ignore_eos`, so a greedy reply ends at the model's end of text whatever its cap and fixed output lengths hold only while the model does not emit it (the `short` count shows when it does); a boolean `ignore_eos` on the four generation routes that masks the end-of-text ids before sampling would close it (closed by the `ignore_eos` block above). The native stream's `done` event carries `tokens` but not the `prompt_tokens` and `reused_tokens` a whole reply carries, so on `/v1/generate` the tool reads reused tokens from `/v1/health` and cannot check prompt lengths per request. There is no tokenize route (`/v1/tokenize` came later, branch feat/server-tokenize, and the tool counts prompts through it since). A burst is admitted up to about `--max-queue`, not `--max-seqs` plus `--max-queue`, since a request waits in the queue until the scheduler's next pass makes it active: 8 at once on 2 and 2 admitted 3, and 100 at once on the defaults 65.
- **Left:** the gate's measurement on the cards, llmx against mx-llama.cpp's server and vLLM with the same load at 1 to 64 users and a rate sweep; `--api completion` and the reference server's `/tokenize` have run here only against in-process servers, since no reference server is built on this machine.

## One owner for the default cache type (2026-09-25, branch fix/kv-cache-default)

- **Why:** the CLI's parameters said f16 while `ModelOptions` said f32, so a model built without the CLI's flags stored f32 caches. The synthetic bench, which the `perf` floors read, and `llmx-split-check`, the layer split's bit-identity gate, never ran the caches users and the server run. Separately, `run_tests.py --cache-type f16` or `f32` failed `perf` and `threads`, because `device_args` gave the synthetic bench cache flags it refuses.
- **Done:** `ModelOptions` holds the one default, f16; the CLI starts from it, changes a side only when its flag is given and prints the default from it in the help; the two cache type names sit beside `KVType`; `llmx-split-check` takes a cache type after the ubatch. The kv-cache transaction check, which counts cache bytes as floats, and `tools/compare_cpu.cpp`, whose reference arm stores f32, ask for f32 sides. `model-validation`, `generation-stream`, `prefill-scope` and `placement` compare two models built with the same options or read zero-weight fixtures, so they keep the default and now run f16. `device_args` gives the synthetic bench neither layer shares nor cache types. The CLI checks a cache type as its flag is read and keeps its one spelling, so an empty or unknown name, `-ctk ""` among them, is refused before any model file is read, as `cli-output` checks. The help prints `f16 (default) or f32` from `ModelOptions{}`, and `--help` of `generate`, `chat`, `logits`, `perplexity`, `bench` and `serve` matches a build of 1a0d4be apart from the version line. `llmx-split-check` reads its cache type before the model file, and the Vulkan backend test names cache types through `kv_type_name`.
- **Exact:** `generate`, `logits`, `perplexity` and `chat` on Qwen3-0.6B Q8_0 give the same bytes as 2946b8a apart from the timing lines, on the CPU and on the Radeon VII, with the default, both sides f16, both f32, and each side alone f32. `llmx-split-check` on 0.6B is bit-identical at f16 and at f32 over two and three CPU stages with 64-token chunks.
- **Left before merge:** `llmx-split-check` at f16 and at f32 over 0.6B, 8B and 30B-A3B on the MI50s, and the suites with `--cache-type f16` and `f32` there.
- **Tests:** CTest 24/24; the suites on the CPU and the Radeon VII pass with the default cache and with `--cache-type f16` and `f32`; on 2946b8a `--cache-type f16` and `f32` fail `perf` and `threads` on the bench's refusal.
- **Tests after the rebase onto 1a0d4be:** CTest 24/24, with `hub-pull`, `model-validation` and `backend-vulkan` passing on a rerun after failing or timing out while other suites held the machine; the CPU suite passes, and `server` and `threads` pass on the Radeon VII.
- **Synthetic bench** (`bench --size 2048 --iters 5 --threads 1`, the `perf` command, 2946b8a with f32 caches against this branch with f16, 30 interleaved runs each, medians):

  | device | matmul GFLOPS | prefill tok/s | decode tok/s |
  |---|---:|---:|---:|
  | CPU, before | 58.4 | 10364 | 9869 |
  | CPU, after | 56.0 | 10003 | 9160 |
  | Radeon VII, before | 53.3 | 1947 | 1977 |
  | Radeon VII, after | 58.4 | 1666 | 2539 |

  This block ran with the machine otherwise quiet (system CPU 13 percent). On the CPU the interquartile ranges do not overlap: the synthetic model's decode is 7.2 percent and its prefill 3.5 percent slower with f16 caches. The matmul line, which no cache touches, moved 4 percent, so the prefill change is within the layout band and the decode change is not. The Radeon VII's 64-token passes vary by half between runs of one binary. The floors hold in the quiet block (the branch's lowest runs 33.8 GFLOPS, 5443 and 5099 tok/s against 8, 1000 and 800). Three more blocks ran while other test suites held 11 to 16 cores; there single matmul runs of either arm fell below the 8 GFLOPS floor, and neither arm differed from the other beyond the spread.

## Layer split phase 3: passes in flight (planned 2026-09-25, decided 2026-09-26, branches feat/split-passes and perf/decode-columns)

- **Goal:** on a pipelined layer split, the server keeps several passes of different requests in flight, so every stage works while the host samples and forms the next pass. Throughput at many users then scales with the stage count, as phase 3 of `docs/MULTI-DEVICE.md` (Order of work) plans. Every request's ids and logprobs stay equal to its run alone and to the CLI's, and identical across one MI50 and 2 to 4 MI50s. Together with the batch-invariant decode kernel branch `perf/decode-columns`, this phase passes the layer split's final gate (Gate, below), and the layer split is not done until it does.
- **Decided by the user (2026-09-25 and 2026-09-26):**
  - The gate: at 16, 32 and 64 users, output throughput at least 2 times both layer-split servers (mx-llama.cpp's ROCm layer split and llama.cpp's Vulkan layer split) and at least 1.5 times mx-llama.cpp's ROCm tensor split and the gfx906 vLLM, with time to first token and inter-token p99 no worse.
  - vLLM is compared at matched bits: llmx Q4_K_M against vLLM AWQ, and llmx Q8_0 against a vLLM 8-bit checkpoint (GPTQ-Int8 or W8A16) where one runs on gfx906, each named. The comparison is rerun on the identical checkpoint once llmx loads AWQ. Both the newest vLLM build that runs and the fastest are reported, with their versions and weights.
  - That gate is the final gate of the layer split, owned jointly by this phase and `perf/decode-columns`, which started 2026-09-26. This phase's scheduler steps merge on their own correctness gates plus being faster than today at every load. The layer split is not done until the full gate passes.
  - Threads: one scheduler thread, a sampling pool, a 16-slot command ring and the fast sampler (`perf/sampler-select`, approved). The thread is measured after step 4 and after the kernel step, and step 7 applies the rule decided with it.
  - A decode kernel that needs a different per-column summation order, fixed and independent of the batch, changes decode output once. It needs the user's OK first, with the count of changed greedy tokens on a fixed set, the HF results and the speed gain. Until then every new build is bit-identical to today's per column.
  - Slice boundaries may follow the cost model; an entry's extent and fresh count never do. `docs/MULTI-DEVICE.md` says so from this commit.
  - Batch invariance and exact reuse only: every request's output equals its run alone and the CLI's, and is identical across one card and 2 to 4 cards.
- **Rules it keeps:**
  - A row's arithmetic follows its entry (token, position, extent, fresh count, its own cache), never what else is in flight.
  - The scheduler steps change no kernel, and `perf/decode-columns` keeps every column's bits until the user approves another order.
  - Exact reuse only: this branch reuses no rows of its own, and a donor is parked only after its last pass has returned.
  - P = 1, every single-device path and the CLI keep main's bytes and speed.
  - One owner per rule, as in the exact-resume plan.
  - A flag means the same on every backend or is refused.
- **Found**, read at d48f2b2 for the model and the backends and at bf17d6a (`feat/server-logprobs`) for the scheduler:
  - The server runs one `forward` per iteration and reads its logits before it forms the next pass. `forward` runs `begin`, every `run_stage` and `finish` in a row, so on a split every stage but one idles.
  - Phase 2 already supplies most of the pieces. There is a `Pass` plan per pass in flight, and `run_stage` reserves and commits one storage. `send` and `receive` go through host-visible buffers. One activation arena per device is shared, because each device runs its passes in order.
  - Five things tie the model to one server pass:
    - `ExecContext` has one logits buffer and one ticket. The head writes at row 0, and `finish` takes the output device's latest ticket, not the pass's own `p.sent`.
    - Each device has two handoff buffers, picked by chunk parity. Picked by slot at an odd pass count, two consecutive passes on one device would share one. With any queueing between stages, parity by start order is not safe either.
    - `begin` calls `ensure`, which replaces the arena, handoff and logits buffers when a pass needs more rows. A pass between `send` and `receive` would lose its residual.
    - `begin`, `run_stage` and `finish` are private.
    - Nothing checks at run time that a sequence is in only one pass. `KVSequence::prepare` refuses a second pending step within one storage only, so a sequence at stage 1 in one pass and at stage 0 in another passes the check.
  - `pipelined_` holds only when there is more than one stage, the embedding is on the first stage's device, the head is on the last stage's device, and every feed-forward block sits beside its attention. Elsewhere `cross` uses handoff buffer 0 inside a stage, so passes in flight are for pipelined placements only.
  - `Sequence::length()` reads the first storage, which a pass commits a stage ahead of the others. The scheduler may read it, and change a sequence, only while that sequence is not in flight.
  - One host thread blocks in seven places: `receive` waiting on the previous stage's ticket; `receive`'s upload into the destination, which waits for the copy of the previous upload out of the same half of staging (every upload under 32 MiB starts at half 0), and so for everything queued on that device before it; `open` when every command slot is busy; the logits wait; a CPU stage (which computes on the calling thread); `reset` waiting on a sequence's last tickets; and `retire`, which drains every device after a failure.
  - The Vulkan backend holds 4 command buffers in flight (`kRing`). An 8B stage on two cards is 3 to 4 submissions plus its receive's upload, so a thread recording ahead onto a busy device blocks in `open`. Phase 2 measured 16 and 32 slots as neutral for prompts. Each slot holds 1 MiB of host memory, which `host_resident` counts.
  - The Q8_0 decode kernel on integer-dot devices, `matmul_vec_q8.comp`, keeps up to 8 columns, so a pass reads the weights once per 8 rows; the K-quant row kernels in `matmul_row.comp` keep 8 too. Each column's sum is fixed by the lane layout and the reduction: one subgroup reduction per row and column in `matmul_vec_q8.comp`, and in the K-quant row kernels a cluster of lanes per row that reduce with xor shuffles, the cluster set by type, row length and the device's profile, never by the column count. So a build with more columns that keeps both computes the same bits, as long as the compiler contracts `acc += dw * dx * float(s)` the same way in every build.
  - The 8-column build loads every column's activations whatever the pass's column count, and holds an accumulator per row and column and a word pair and a scale per column in registers. So its fixed cost grows with its width: at 32 columns about 160 registers a lane, counted from those arrays (64 accumulators, 64 words, 32 scales) before the compiler's own, one wave per SIMD. A one-column chunk takes a one-column build, which frees those registers.
  - Step 0's first runs, on one MI50 with clocks held high, llmx at db0f8c3, Qwen3-8B Q8_0, 512-token prompts and 128-token replies with the end of text ignored, host load average 15 to 54 (vLLM's runs 34 to 78), two rounds each:
    - An 8B pass costs 16.1 ms at 1 row, 28.3 at 2, 31.6 at 4, 39.5 at 8, 78.2 at 16, 157 at 32 and 320 at 64, read from the server's inter-token p50 with host sampling included (the 64-slot server's second round, at host load 34 to 54, read 4 to 7 percent higher: 167 at 32, 336 at 64). From 2 rows that is about 25 ms for each block of 8 rows plus 1.75 ms a row; one row takes the one-column build. Eight rows cost 2.5 times one row, and every further 8 rows cost about another 8-row pass. `docs/MULTI-DEVICE.md` said wider passes were free up to about 35 rows, and now points to this curve.
    - `llmx bench --seqs` on one MI50 (step 0, 128- and 512-token prompts, two rounds each, one-minute load average 5 to 29) gives 14.8 to 15.2 ms a pass at 1 row, 27.0 to 28.8 at 2, 28.8 to 31.9 at 4, 34.0 to 38.7 at 8, 47.3 to 53.1 at 9, 66.2 to 77.2 at 16, 132 to 156 at 32 and 265 to 294 at 64. That is the same shape, between 2 percent above and 17 percent below the server's figures, which include host sampling; the targets below still use the server's.
    - llmx serves 118 to 125 tok/s at 16 users, 118 to 124 at 32 and 113 to 124 at 64, decoding 190 to 207 tok/s by inter-token p50.
    - The ROCm reference (mx-llama.cpp's server) serves 100 to 129, 113 to 136 and 111 to 146 tok/s, and steps at 212 to 261, 298 to 337 and 339 to 430 tok/s by inter-token p50. That is well below its kernel: its batched bench decodes 16, 32 and 64 rows at 295 to 307, 518 to 559 and 645 to 751 tok/s on one pinned MI50 (step 0).
    - vLLM, the gfx906 fork's v0.12.0 image on FP16 weights, serves 19.8 to 21.1, 18.0 to 18.8, 36.6 to 38.9, 55.1 to 58.2, 75.9 to 79.2, 93.6 to 96.2 and 118.2 to 120.5 tok/s at 1, 2, 4, 8, 16, 32 and 64 users. At 64 users it is level with llmx, and its inter-token p50 of 177 ms is a decode rate of 361 tok/s, 1.8 times llmx's.
    - The same image on the Q8_0 GGUF gave 61.0 tok/s at 16 users, and the newest image that ran (aiinfos/vllm-gfx906-mobydick, which reports vLLM 0.1.dev17732+gff063e44e and is published as 0.23.1rc0; FP16) 46.7 to 47.7 with a 15.8 to 18.5 s time to first token. The 0.29 source line is unbuilt, and no GPTQ-Int8 or W8A16 run of 8B exists yet (AWQ: What the targets say). The Q8_0 GGUF run is 8-bit, but not one of the formats the decision names.
    - A 512-token prompt costs 385 to 400 ms of one MI50 (time to first token 0.397 s at one user, pp512 1330 tok/s), against 40 ms for 8 decode rows. A pass that carries a whole ubatch of prompt holds every pass behind it at every stage.
  - 32B Q8_0, about 35 GB, does not fit one 32 GB MI50. Step 0 measured it on two MI50s at P = 1 (`bench --seqs` on 128-token prompts, two rounds): 61.7 to 62.0 ms a pass at 1 row, 123 at 8, 247 at 16 and 495 to 496 at 32. That is 3.4 to 4.1 times 8B's one-card pass in the same runs, so the targets' 4.0 times sits near the top of it. At 2, 4, 9 and 24 rows in both rounds, and at 48 and 64 in one, the model did not fit, because other processes held memory on those two cards.
  - The fit balances layer counts. Modeled, the head is 7 percent of an 8B pass and 2 percent of a 32B pass. So for 8B the last stage is about 15 percent heavier than the others on two cards and 30 percent on four, which caps two cards near 1.9 times one card and four near 3.3, and for 32B about 4 percent on two.
  - Sampling per row on the EPYC 7262, with the scheduler's 594 KiB row copy, medians at host load average 36 to 39: 0.25 ms greedy, 0.45 at the defaults (temperature 0.8, top-k 40, top-p 0.95), 1.55 greedy with a penalty of 1.1, 1.68 at the defaults with it, and 15.4 with top-k 0 (18.3 with top-p 0.95), which the compatible routes give a client's `top_k: -1`. The row copy alone takes 0.054 ms.
    - Main's penalty looks up a hash set for each of the 151,936 entries, which is the 1.31 to 1.35 ms it adds at a load average of 11 to 12. `perf/sampler-select` writes the penalized scores into a copy of the row instead, and with it a penalty adds 0.02 to 0.10 ms.
    - `perf/sampler-select` selects only what top-k and top-p keep, so the defaults take less than main's, and top-k 0 takes 0.85 ms at top-p 1 and 6.0 to 6.1 ms at top-p 0.95 (below).
    - Step 0 timed both again at a one-minute load average of 11.2 to 11.9, two rounds each.
      - main: 0.125 to 0.128 ms greedy, 0.256 to 0.258 at the defaults, 1.47 to 1.48 greedy with a penalty of 1.1, 1.56 to 1.57 at the defaults with it, 12.6 to 12.7 with top-k 0, and 12.7 to 13.7 with top-k 0 and top-p 0.95.
      - `perf/sampler-select`: 0.124 to 0.132, 0.216 to 0.221, 0.142 to 0.227, 0.234 to 0.240, 0.85 with top-k 0, and 6.0 to 6.1 with top-k 0 and top-p 0.95. That last figure is on uniform logits, which keep nearly every token under top-p.
      - The row copy is 0.017 to 0.025 ms.
      - Both samplers are still 5 to 10 times the 25 microseconds a pass that the server's step 5 timing build recorded at 1 to 16 sequences (`docs/SERVER.md`), and step 0 times that host gap again.
  - Host share of a round, modeled: the busiest host thread's work per round over the round, for 8B at P = S with today's decode cost, the worst of 16, 32 and 64 users, read as the defaults / the defaults with a penalty of 1.1, from step 0's sampler medians at a load average of 11 to 12. Recording is scaled from 0.6B's measured recording (about 0.9 to 2.2 ms a pass at 8B's 36 layers), and a crossing costs 0.11 to 0.17 ms.

    | Host | S = 2 | S = 3 | S = 4 |
    |---|---|---|---|
    | One thread, main's sampler | 22% / 72% | 29% / 92% | 41% / 129% |
    | One thread, the new sampler | 21% / 21% | 28% / 28% | 39% / 40% |
    | One thread and four sampling threads, the new sampler | 14% / 15% | 20% / 20% | 28% / 28% |

    - With the new sampler the penalty no longer needs the pool. Top-k 0 at top-p 1 (0.85 ms) takes up to 81 percent of a round at S = 4 without the pool and 38 percent with it. At top-p 0.95, the bench's 6.0 ms takes at worst 69 to 70, 89 to 90 and 125 to 126 percent of a round at S = 2, 3 and 4 even with the pool. Real logits keep far fewer tokens, so step 0 times it on the model's own logits before the pool is sized.
    - A faster pass shortens the round. At the pass costs 32B needs (8 rows in 21.5 ms and 16 in 25 ms, 8B-equivalent, Targets) and with head-aware shares, one thread with the pool is modeled at 62 to 63 percent of a round at S = 4 with 32 users and 84 to 86 percent with 64, recording being the largest part.
  - Whether one thread keeps the devices fed is not modeled reliably. The first model put it at 55 to 71 percent of the device-bound rate at four stages in a fixed round order, and at 83 to 91 percent serving stages as they complete. A later simulation put it near 100 percent with the pool, but it does not state its order and is unchecked. Phase 0's P sweep ran a thread per stage, so step 0's pipeline bench, driven by one thread and by a thread per stage, measures it.
  - Elsewhere, servers that keep decode passes in flight across stages give each device a process (vLLM, SGLang, TensorRT-LLM) or a thread (Orca, in C++), and all but SGLang keep one central scheduler. LMDeploy's TurboMind, also C++, gives each device a scheduling thread and an executor thread but has no pipeline stages. None relays activations through the host as llmx does.
  - mx-llama.cpp's continuous-serving research executor (not landed), under ROCm, where the host does not wait between stages, kept 2 to 8 MI50s at 95.6 to 100.6 percent of the stage count times one request's rate, with one host thread and one queue per device. One spare lane gained 4.5 and 10.2 percent at 8 and 10 lanes on ten MI50s, and a second spare lost.
  - mx-llama.cpp's tensor split sums the cards' partials in a fixed order and sends them as F32 or BF16 by message size, which grows with the batch, so its outputs depend on the batch and on the card count. llmx cannot copy that reduction.
  - A tensor group that splits output rows and gathers them computes every element as one card does, since the Q8_0 kernel adds a lane's blocks one after another, and a split by input blocks would regroup that sum. At 16 to 64 users such a group reads the same bytes per token as P = S, so this gate is won by the pipeline and the kernel.
- **Targets**, modeled for 8B Q8_0 and 30B-A3B Q4_K_M from their measured one-card curves and for 32B Q8_0 from 8B's; 8B and 32B Q4_K_M, 30B-A3B Q8_0 and the Radeon VII with the CPU wait for step 0's curves:
  - Throughput is decode-bound output tok/s, greedy; prompt loads follow below. P is S or step 8's P, whichever gives the higher rate. The pool is in place and the host does not limit. Crossings (0.11 to 0.17 ms) are ignored, and 32B costs 4.0 times 8B.
  - Every reference figure marked "est." is an estimate. Step 0's first multi-card runs put the ROCm layer split within about 40 percent of it, but the tensor split 21 to 75 percent below. The tensor split's tg128 ratios do not carry over to its server, which served less than one card at 16 and 32 users on 2 and 4 cards. Step 0's measurements replace its column. The 1.5 times bar it sets is not the highest one: that is 2 times the layer split, or vLLM. The method predicts the ROCm 32B layer split at about 59 tok/s at 16 users, where 84.0 was measured.
    - On 4 MI50s at 512/128 (two rounds, host load 21 to 74), decode by inter-token p50: the tensor split 219 to 237, 160 to 172 and 238 to 249 tok/s at 16, 32 and 64 users, 21 to 75 percent under the estimates below; the layer split 195 to 214, 227 to 288 and 276 to 347, inside them.
    - The tensor split served 78 to 82 and 92 tok/s on 4 cards at 512/128 at 16 and 32 users, against one card's 100 to 129 and 113 to 136, and 133 and 199 on 2 cards at 128/128 (one round), against one card's 154 and 216.
  - The ROCm layer split is taken at about one card, since its tg128 on 2 to 4 cards is 70.5 to 69.7 against 71.0 on one. One card is the ROCm reference's decode rate by inter-token p50 (Found).
  - The ROCm tensor split is taken at that rate times 1.41 to 1.51 on 2 cards, 1.41 to 1.77 on 3 and 1.41 to 1.89 on 4: its tg128 ratios, and its 32B ratio to the layer split at 32 users. Step 0 measured it far lower (above).
  - The Vulkan layer split's 8B server on two cards served 96, 124 and 269 tok/s at 16, 32 and 64 users (128/128, one round), 0.58 to 0.71 times the ROCm layer split's 151, 213 and 379. So the ROCm layer split still sets the 2 times bar.
  - "Needed" is the lowest rate that meets 2 times the layer split and 1.5 times the tensor split, from the low and the high ends of the estimates.
  - "Today's kernel" is P = S or step 8's P. "+ head shares" adds step 6. "+ wider build" adds step 5's 16- and 32-column builds at 39.5 ms plus 2.0 ms a row beyond 8. That is an estimate, not a bound either way, until a 16-column prototype is timed. The build's fixed cost grows with its width (Found). The reference's batched bench decodes 16 rows in 52.1 to 54.3 ms and 32 in 57.3 to 61.8, faster than this model.
  - "Pass cost needed" is the 8B pass cost at which the needed rate is met, with head-aware shares at P = S.

  **Qwen3-8B Q8_0 on 2, 3 and 4 MI50s:**

  | Cards | Users | ROCm layer split (est.) | ROCm tensor split (est.) | Needed | llmx, today's kernel | + head shares | + wider build | Pass cost needed, ms |
  |---:|---:|---|---|---|---:|---:|---:|---|
  | 2 | 16 | 212-261 | 299-394 | 448-591 | 379 | 405 | 405 | 8 rows 27.1-35.7 (39.5 today) |
  | 2 | 32 | 298-337 | 420-509 | 630-763 | 382 | 409 | 577 | 16 rows 41.9-50.8 (78.2) |
  | 2 | 64 | 339-430 | 478-649 | 717-974 | 381 | 408 | 731 | 32 rows 65.7-89.3 (157) |
  | 3 | 16 | 212-261 | 299-462 | 448-693 | 409 | 466 | 466 | 5 to 6 rows 23.1-35.7 (34.3) |
  | 3 | 32 | 298-337 | 420-596 | 630-895 | 533 | 608 | 714 | 10 to 11 rows 35.8-50.8 (68.7) |
  | 3 | 64 | 339-430 | 478-761 | 717-1142 | 533 | 608 | 967 | 21 to 22 rows 56.1-89.3 (112.3) |
  | 4 | 16 | 212-261 | 299-493 | 448-740 | 418 | 506 | 506 | 4 rows 21.6-35.7 (31.6) |
  | 4 | 32 | 298-337 | 420-637 | 630-955 | 670 | 810 | 810 | 8 rows 33.5-50.8 (39.5) |
  | 4 | 64 | 339-430 | 478-813 | 717-1219 | 676 | 818 | 1153 | 16 rows 52.5-89.3 (78.2) |

  **Qwen3-32B Q8_0 on 2 MI50s**, with the references measured on 2026-09-24 on short prompts with the end of text allowed:

  | Users | ROCm layer split | Vulkan layer split | ROCm tensor split | vLLM, AWQ, TP 2 | Needed, Q8_0 | llmx, today's kernel | + head shares | + wider build | Pass cost needed, 8B-equivalent |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
  | 16 | 84.0 | 34.9 | 124.3 | 208.6 | 186 | 99 | 101 | 101 | 8 rows 21.5 ms (39.5 today) |
  | 32 | 151.6 | 64.4 | 213.3 | 237.7 | 320 | 100 | 102 | 144 | 16 rows 25.0 ms (78.2 today, 55.5 wider) |
  | 64 | step 0 | step 0 | step 0 | step 0 | step 0 | 100 | 102 | 183 | step 0 |

  - The AWQ figures face llmx's Q4_K_M, which needs 313 and 357 tok/s at 16 and 32 users; llmx's 32B Q4_K_M curve is unmeasured. The vLLM 8-bit 32B, which the Q8_0 row faces, is unmeasured too.
  - At the gate's 128/128 load (step 0) the references served less than on 2026-09-24.
    - ROCm tensor split: 86 to 92, 125 to 126 and 173 to 179 tok/s at 16, 32 and 64 users.
    - vLLM AWQ with tensor parallelism 2: 144, 159 to 160 and 164 to 166 (v0.12.0), and 126 to 140, 147 to 153 and 163 (0.23.1rc0).
    - llmx at P = 1: 54 (one round).
    - The bars in this table come down when step 0 re-derives it.

  **Qwen3-30B-A3B Q4_K_M on 2 and 3 MI50s**, modeled from llmx's one-card server (115, 194, 218 to 222, 230 to 232 and 262 to 263 tok/s at 1, 4, 8, 16 and 32 users, 64-token replies on short prompts, 2026-09-24) with 8B's head share. The only reference figures are the reference's server on one card the same day (94, 149, 164, 110 and 203 tok/s at 1, 4, 8, 16 and 32 users), standing in for the layer splits. Its 110 at 16 users is below its 164 at 8, so the 16-user bar (220) is probably low; step 0 re-measures it.

  | Cards | Users | Layer split (est.) | Needed, 2 times it | llmx, today's kernel | + head shares |
  |---:|---:|---:|---:|---:|---:|
  | 2 | 16 | 110 | 220 | 411 | 440 |
  | 2 | 32 | 203 | 406 | 432 | 462 |
  | 2 | 64 | step 0 | step 0 | 491 | 525 |
  | 3 | 16 | 110 | 220 | 543 | 619 |
  | 3 | 32 | 203 | 406 | 593 | 676 |
  | 3 | 64 | step 0 | step 0 | 647 | 737 |

  - The tensor split and vLLM on this model are unmeasured. MoE tensor decode gained little on this hardware (`docs/MULTI-DEVICE.md`), and the gfx906 vLLM was reported partly working on it. The Q8_0 file (32.5 GB) needs two cards, and step 0 models it from its own curve.

  **The Radeon VII with the CPU:** the CPU stage computes on the scheduler thread, so P = 2 overlaps the card with the CPU and is bound by the CPU's stage. The reference is llama.cpp's Vulkan build on the Radeon VII with the same layers on the CPU. Step 0 models it from both devices' measured pass costs, and its bar waits for the user (Open).

  **Prompt loads**, modeled as device time per output token = (prompt tokens per reply token) / prefill rate + 1 / decode rate. These are estimates at 32 users, with today's kernel and the head shares:
  - Prefill on S cards is taken as 0.9 S times one card for llmx, since prompts of different requests fill the stages, and as the references' measured pp4096 ratios: the layer split 1.62 on 2 cards and 2.59 on 4, the tensor split 1.56 and 1.98. One card prefills 8B at about 1200 tok/s for llmx and 1000 for the reference on 128-token prompts (their pp64 and pp512 lie either side), and about 1250 and 1300 on 1024-token prompts (their pp512 and pp4096).
  - 128/128: llmx about 344 tok/s on 2 cards against 252 to 279 (layer split) and 331 to 384 (tensor split), and 682 on 4 against 267 to 298 and 347 to 482. Prompts take about 16 percent of llmx's device time on 4 cards, so the decode-bound tables above are optimistic by about that.
  - 1024/128: llmx about 167 tok/s on 2 cards against 140 to 148 and 158 to 169, and 332 on 4 against 174 to 187 and 182 to 214. Here a prompt costs more device time than its reply, and llmx's prefill per card is level with the reference's at 512 tokens (1318 against 1323 tok/s by the mean of rounds) and 0.84 times at 4096 (976 against 1167).
  - On 2 cards, 2 times the layer split at 1024/128 (280 to 296 tok/s) is at or past what llmx gives even with free decode (281). With decode at about 700 tok/s, which the kernel has to give there anyway, it needs llmx's prefill 1.7 to 1.8 times faster per card than today. On 4 cards, 332 misses 2 times the layer split (349 to 374), so there too it needs a cheaper decode pass or faster prefill. So at that load the gate rests on prefill work as much as on this phase (Open, below).
  - **What the targets say:**
    - At the low end of the estimates, the kernel is needed for 8B on 2 cards at every load and on 3 cards at 32 and 64 users, and for 32B everywhere. 8B on 4 cards, and on 3 cards at 16 users, clears, decode-bound, the two llama.cpp-based references with the head shares and today's kernel, by 4 to 29 percent, inside the estimates' spread (vLLM's 8-bit rate is unmeasured).
    - At the high end, every 8B placement needs a faster pass than the wider build's estimate gives, so the pass at 2 to 8 rows must lose fixed cost too: 4 rows in 21.6 ms and 8 in 27.1 ms, against 31.6 and 39.5 today.
    - 32B needs the most: an 8-row pass 1.84 times faster than today and a 16-row pass 3.1 times faster, 2.2 times beyond the wider build's estimate. That is estimated possible from the MI50's sustained read ceiling, which mx-llama.cpp measured at 786 GB/s. A 32B stage on two cards reads about 17.4 GB: about 22 ms at that rate, and about 26 ms at the 650 to 685 GB/s that llmx's dense projections reach (`docs/VULKAN.md`). The budget is 43 ms for an 8-row stage, but no kernel on this card has shown it.
    - 30B-A3B clears 2 times the one-card reference with today's kernel, narrowly at 32 users on 2 cards (432 against 406). Its tensor split and vLLM rows are unknown.
    - vLLM's AWQ build sets the highest bar measured so far from 16 users on, the one llmx Q4_K_M faces. At 128/128 on one MI50 v0.12.0 serves 310 to 314, 350 to 353 and 354 to 361 tok/s at 16, 32 and 64 users. That is 1.7 to 2.0 times llmx Q4_K_M's 176 to 179. With tensor parallelism 2 it serves 418, 543 and 577 (one round so far).
    - 0.23.1rc0 serves less on one card (238 to 289, 312 to 344 and 310 to 352), and 359 to 438, 454 to 478 and 558 to 566 on two over two rounds, above v0.12.0's one round at 16 users only. By the faster build's higher round, the gate asks llmx Q4_K_M for 657, 814 and 866 tok/s on two cards.
    - vLLM's FP16 tensor parallelism scales little on these cards: 337.5 tok/s at 64 users on two, and 162 to 165 at 512/128 on four (decode by inter-token p50 381 to 390, against 361 on one). Its 8-bit checkpoint (GPTQ-Int8, downloaded) has not run yet, and its FP16 rates stand in for the bar llmx Q8_0 faces only until it has.
    - The head shares are needed on 3 and 4 cards, step 8 on 3 cards at 32 and 64 users (from 409 and 500 tok/s to 533), the pool and the new sampler for the sampled loads, and step 9 and prefill work for the prompt loads.
- **Levers**, each batch-invariant:
  - Wider decode builds (step 5): one weight read for up to 16 or 32 columns. A build keeps batch invariance only if each column keeps its lanes and its reduction (Found: one subgroup reduction in `matmul_vec_q8.comp`, a cluster's xor shuffles in the K-quant row kernels); the row count may change only how many columns share a weight fetch. Counterexamples: Marlin picks its tile by batch, llama.cpp switches kernels at 8 columns, and the gfx906 vLLM's AWQ kernel switches at 32 rows and adds with FP16 atomics.
  - A cheaper pass at 2 to 8 rows (step 5): builds that load only the columns a pass has.
  - Stage shares that count the head (step 6): any split is bit-identical to one card (`llmx-split-check`).
  - P from the kernel's column width (step 8) and assembly by predicted stage time (step 9): the cost model chooses which entries share a pass and where a slice ends, never an entry's extent or fresh count.
  - Host work off the critical path (steps 3 and 4): the 16-slot ring, the pool, rows read in place and the new sampler, which changes seeded draws once and leaves greedy unchanged.
  - Considered, not planned: greedy argmax on the device for rows that ask for neither logprobs nor a penalty. It is exact by construction (comparisons only, the lowest id on a tie, the host's handling of NaN kept), and it would take the row's pass over the vocabulary and part of the logits wait off the thread. If step 7's measurement shows the logits wait or greedy sampling limiting, it goes to the user beside step 7's numbers. It is not built before that.
  - Not in this phase: prefill speed, which the prompt loads rest on; tensor groups (phase 6); command buffers recorded once and replayed (Not doing).
- **Design:** one host thread, the scheduler's, drives every stage of every pass, with a pool of sampling threads the scheduler owns, a 16-slot command ring and the new sampler (Decided). It does so through a small public pass API over one `ExecContext` with P pass slots. Four designs were compared:
  - This one.
  - A relay that does not block, with P = S + 1. Each crossing waits on the device through a sync file, and the destination reads the handoff without staging, from host memory imported into it or from the source's memory through dma-buf, so neither the source's ticket nor staging holds the thread (on Linux, where every multi-card placement runs, a 20 KB hop took 55 us through dma-buf with the chain queued ahead, and 160 KB took 72 us on one root complex and 132 across; a dma-buf read across complexes runs at 1.1 GB/s against 9.2 on one, so a prompt-sized handoff across complexes takes host memory imported into the destination). Or the thread polls each device's timeline and serves stages in the order they complete. One pass more than the stages hides a round of sampling and assembly, as vLLM's pp + 1 does.
    - It needs no contract change, and it is step 7's first answer when the thread is found blocked while devices idle.
    - Its costs: backend-private code, a non-blocking ticket query for the polling form, and fewer rows per pass at P = S + 1, which today's curve charges (modeled: 346 against 418 tok/s at 16 users on 4 cards while the devices bound).
    - A device-side wait is trusted only after a garbage check at 16k tokens and more, since mx-llama.cpp's ROCm stage transfers raced only past about 3K tokens.
  - A submitter thread per device that runs the (pass, stage) jobs the scheduler posts, in order, and decides nothing. It changes the backend's one-thread contract, needs the ticket read atomic and the block pool kept on one thread or locked, and races on the MoE run vectors unless each device has its own. It goes to the user only by step 7's rule.
  - A strict rotation of exactly S passes with buffers picked by start parity. It is not taken as it stands, for three reasons. Its buffers are safe only under strict rotation. It cannot run more passes than stages. It retries growth only when a request's own pass returns, which can give room made for an older request to a younger one.
  - **Model, no arithmetic touched:**
    - `reserve_passes(ctx, slots, rows, logit_rows)` runs once before serving. It sizes the arena for `ubatch + max_seqs` rows, `max(2, slots)` handoff buffers per sending device, and the logits rows. The context is then frozen: `begin` throws `logic_error` before any work if a pass would need more, and nothing is replaced while serving. The CLI never calls it and keeps `ensure` as today.
    - `Pass::parity` becomes `handoff`, the handoff buffer the pass owns until it ends. `Pass` also gains `logits_base`, its first logits row. The server sets `handoff` to the pass's slot and `logits_base` to the range the scheduler gave it; `prefill` sets chunk parity and row 0 as today.
    - The head writes at `logits_base`, and `finish` records `p.sent` as the pass's logits ticket.
    - `begin_pass` marks each of its sequences in flight and refuses one already marked; `end_pass` and `abort_pass` clear the mark. `reset` and `fork` refuse a sequence in flight. `forward` and `prefill` do not use the mark, since a pipelined prompt has several chunks in flight on purpose.
    - Public calls: `stage_count`, `pipelined`, `reserve_passes`, `begin_pass(ctx, slot, entries, n, logits_base)`, `run_pass_stage(ctx, slot, s)`, `pass_logits(ctx, slot, i)` (which waits on the pass's own ticket), `end_pass(ctx, slot)` and `abort_pass(ctx, slot)`. `abort_pass` is today's `roll_back`: every device is drained, then only that pass's entries are truncated to where the pass found them.
    - `forward`, `prefill`, `step` and `score` keep their code.
  - **Logits:**
    - One host-visible range of `2 * max_seqs * (k + 1)` rows sits on the output device. It is carved in formation order and released when a pass ends.
    - Wanting rows in flight never exceed `max_seqs * (k + 1)`, since a sequence is in one pass and wants at most one row, or k + 1 as a speculative verify entry of k drafts (k = 0 without speculative decoding), so a contiguous range of `2 * max_seqs * (k + 1)` rows always fits. An aborted pass leaves a gap, reclaimed once the passes formed before it have ended.
    - The scheduler's policy core owns the offsets, and the model only checks them against the reservation.
    - At `max_seqs` 64 on the 151,936-token vocabulary the range is 74 MiB at any P without drafts, and k + 1 times that with them. The fit's `ubatch + max_seqs` logits rows cover it while `2 * max_seqs * (k + 1)` is at most that (k up to 3 at `max_seqs` 64 and ubatch 512).
  - **The round**, which the scheduler thread repeats:
    1. Under the lock, new submissions join the queue, and cancellations are swept in the queue, the paused list and the stalled list.
    2. Advance from the last stage down to stage 1: each device stage takes its oldest waiting pass. Each `receive` waits on a submission made in an earlier round.
    3. Retire, oldest first, every pass whose last stage was recorded in an earlier round. Wait on its logits, sample its wanting rows, push the tokens and end the pass. Its members leave flight, ends and cancels are applied, and a member that needs more blocks joins the stalled list.
    4. Room, in the exact-resume order. Stalled requests go first, oldest first; then resumes, oldest first; then new admissions, and only when nothing is stalled or paused.
    5. Form new passes while a slot and a logits range are free, work is ready and fewer than P passes are in flight: assemble, `begin_pass`, then stage 0.
    6. Run the CPU stages due this round, after every device stage of the round has been submitted, since a CPU stage computes on this thread.
    7. With nothing in flight and nothing ready, wait on the queue.
  - Each pass advances at most one stage a round, and each stage takes its oldest waiting pass. So every device runs its passes in formation order, which is what makes the shared arena safe. Advancing from the last stage down gives the later devices their work before the host samples.
  - **P:**
    - Step 3 runs P = S on a pipelined placement.
    - Step 8 changes it to P = max(P0, ceil(D / W)) for D decode rows, capped by `--passes`, where P0 is S, or S + 1 once step 7 has taken the relay that does not block, and W is the decode build's column count: 8 today, and step 5's width once it merges. A speculative verify entry counts its k + 1 rows. Passes then carry whole blocks of W decode rows, and no pass is twice as heavy as the rest. Decode rows per pass are at most ceil(D / P).
    - Step 7 runs P = S + 1 with that relay when its rule calls for them.
    - Other placements run P = 1, and `--passes` above 1 is refused there.
  - **Room and pausing:**
    - `make_room` stays the one owner of who gives up blocks. Each round it plans over every request, in flight or not, by the exact-resume rule: donors first, oldest first; then, for growth only, uncapped requests admitted after the asker, latest first.
    - It returns what it may take now and which requests in flight it waits for, and it takes nothing until the whole plan is out of flight.
    - A request the plan waits for is held: when its pass returns it is not formed into a new pass, and the next round's plan, recomputed from scratch, takes it.
    - The plan is never stored, so if the asker is cancelled or room frees up elsewhere, the hold ends within one round. A stall lasts at most one lap of the passes after its plan is made.
  - **Assembly:**
    - Step 3 keeps today's rule: decode entries first, then prompt slices by exact-resume's row classes, one stretch per pass, up to the ubatch.
    - Step 9 sizes passes by predicted stage time instead. Slices come in whole 64-row tiles, costed with each stage's measured coefficients and corrected by observed pass times. Replays of generated rows are costed as decode rows, which replaces `kReplayRows`. When nothing else is decoding, a prompt gets the whole ubatch, so time to first token does not regress at low load.
    - The cost model chooses which entries share a pass and where a slice ends, never an entry's extent or fresh count.
  - **Sampling:** rows are sampled per wanting row in entry order, each request with its own random generator, so the result does not depend on the pass, on P or on order. From step 4, after `perf/sampler-select` has merged, a pass's rows are sampled on a small pool of threads private to the scheduler. They read the mapped row in place and copy it only when logprobs are asked. The scheduler thread makes the logits wait (`pass_logits`) and hands the pool the row pointers. Every pool read finishes before `end_pass` releases the pass's logits range. Only the scheduler thread touches channels, the ledger and backends.
  - **Failure:**
    - A throw in any pass call reaches `abort_pass`. Every device is drained and that pass's entries are truncated in every storage. Its requests end with "error" and are released, never parked.
    - Other passes continue, because their rows sit in their own storages, handoff buffers and logits ranges. The drain submits and completes what the failed pass recorded (`sync`), so no tag describes work that did not run, and a stage that starts with `receive` also resets the tags through its host write.
    - This narrows today's rule, where a failed pass ends every active request. Device loss still exits.
  - **Cancel and stop:** a request not in flight ends at once. A request in flight completes its pass, is not sampled, ends with "cancel" and is parked, since its cache is consistent. `stop()` aborts every pass in flight and then ends everything as today, leaving the ledger and every pool at zero.
  - **Health:** `/v1/health` adds `passes` (the cap and the count in flight), the round period, the thread's working time per round (recording, relaying, sampling and assembly), its blocked time in `receive` (the source's ticket and the staging wait apart), in `open` and in the logits wait, each stage's idle share and the device-bound rate, both only from a build with GPU timestamps (below). The timers are sampled and can be switched off, and their cost is measured against a build without them before any round time is trusted, since mx-llama.cpp's per-submission timers cost it 9 percent.
  - **The device-bound rate** is the rows a pass carries over the busiest stage's device time per pass. A stage's device time comes from GPU timestamps at the start and end of its submissions, in a timing build run on the same load. Ticket times the host sees are not used, because the command ring hides stage edges from the host. Throughput is always read from a build without timestamps.
  - **Memory** for Qwen3-8B at `max_seqs` 64 and ubatch 512:
    - The arena is 216 KiB a row for 576 rows, 121.5 MiB per device, and does not grow with P. A context per pass would add 121.5 MiB per device for every extra pass.
    - Handoff is 9 MiB of host memory per slot per sending device: 72 MiB at 8 slots against 18 MiB today.
    - Logits take 74 MiB at any P without drafts (Logits).
    - The 16-slot ring adds 12 MiB of host memory per device.
    - The placement request carries the slot count, and the fit counts `max(2, slots)` handoff buffers. When they do not fit, the server says so at start and runs with fewer.
- **Owners:**
  - What is in flight, and in which order: the round.
  - Who gives up blocks for whom: `make_room`.
  - Where a pass's logits go: the policy core's range allocator.
  - How a row is computed: the request's row classes, from the exact-resume branch.
  - How a decode column is summed: the kernel's lane layout and reduction, the same in every build.
  - Buffer sizes: `reserve_passes`, once.
  - Whether the host gets more threads: step 7's rule.
  - The final gate: Gate, below, shared with `perf/decode-columns`.
- **Plan:** each step of `feat/split-passes` is one commit. Before the next step starts, every step keeps `generate`, `logits`, `perplexity` and `chat` byte-identical to main on the CPU, the Radeon VII and one MI50. It also keeps `llmx-split-check` bit-identical on 2, 3 and 4 MI50s for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M. Each of this phase's steps from 1 to 9 other than step 5 merges on its own correctness gates plus being faster than today at every load (Decided), read as Open says until the user confirms it.
  0. **Measure; no change to the model, the backends or the server**, with cards pinned, clocks held high and the host at a load average under 12. Each program's effective P and ring depth are read from its own log, never from its command line. Which cards are free is recorded before each run, since other processes held memory on two of them during 32B's `--seqs`. Done so far:
     - the one-card runs in Found, at a host load of 15 to 54;
     - 4-card servers at 512/128, two rounds: llmx at P = 1 (109 to 116 tok/s), the ROCm layer and tensor splits, and vLLM FP16 with tensor parallelism 4;
     - the sampler timing;
     - `bench --seqs` for 8B Q8_0 on one card and 32B Q8_0 on two;
     - the ROCm and Vulkan batched benches;
     - 128/128 servers on one and two cards, a first round of each; the second round is running.

     The one-card runs in Found and the 4-card runs were at a host load above 12 and are first readings only: they are repeated at a load average under 12 before any target uses them. Left:
     - `llmx bench --model M --seqs N`, one run for each N of 1, 2, 4, 8, 9, 16, 24, 32 and 64, for 8B Q4_K_M and 32B Q4_K_M on one MI50, 32B Q8_0 on two MI50s with one pass in flight at the row counts that did not fit (Found), and 30B-A3B Q4_K_M on one and Q8_0 on two. These are the pass cost curves every target rests on.
     - A timing build on 2, 3 and 4 MI50s measures:
       - recording per stage at 1 to 64 rows;
       - one crossing;
       - the head's share of the last stage;
       - each stage's device time from GPU timestamps;
       - `llmx-multi-device-bench pipeline` at P = S and S + 1, with the measured stage times, against 4 and 16 command-ring slots (the 16-slot arm a build with `kRing` 16), driven by one thread and by a thread per stage. The bench runs two stages with a thread each today, so a `tools/` commit of its own first gives it 3 and 4 stages and a one-thread driver.
     - The server's host time between a pass's logits and the next pass, which `docs/SERVER.md` records far lower than the sampler's time a row, and top-k 0 with top-p 0.95 timed on the model's own logits.
     - Today's server and the references on the same cards in the same minutes, under the gate's loads and server settings (Gate): 8B on 1 to 4 MI50s, 32B on 2, 30B-A3B on 2 and 3, and the Q4_K_M rows beside vLLM's AWQ. The references' 8B servers on 2, 3 and 4 cards are measured for the first time, every reference runs at 48 and 64 users too, and vLLM runs as its newest and its fastest build, in 8 bits and in AWQ where they run.
     - Gate: the numbers recorded in this block. Every target is re-derived for every placement and load the gate runs, prompts included, with the estimates replaced by same-minute measurements. The host share of a round is predicted for S = 2, 3 and 4 at today's decode cost and at step 5's targets.
  1. **The pass API**, as in Design (Model): `reserve_passes`, `Pass::handoff`, `logits_base`, the in-flight mark, the frozen context and the fit's slot count. The server still calls `forward`.
     - Gate: placement test 1 below.
     - Gate: the split plan prints identical to main for every placement the suites run.
     - Gate: `llmx bench` tg128 and pp512 level with main on one MI50 and on the Radeon VII, and `server_load.py` level with main.
     - Docs: the `ExecContext` comment.
  2. **The scheduler over the pass API at P = 1.** The policy core (the round, the room plan and the logits ranges) becomes free functions, and property test 2 arrives with them.
     - Gate: server replies byte-identical to the base (`server.py`, `server_mix_check.py` with logprobs, `server-resume`) on the CPU, the Radeon VII, one MI50 and a 3-card split.
     - Gate: the property test is clean over at least 100,000 schedules under UBSan.
     - Gate: `server_load.py` level with the base on both machines.
  3. **P = S on pipelined placements, and the 16-slot command ring.** This step brings the round, holds, deferred cancel and park, failure limited to one pass, `--passes` refused above 1 on placements that are not pipelined, and the health fields.
     - Gate: tests 3 to 6.
     - Gate: P = 1 byte-identical to step 2 and level with it in speed, and one user at P = S level with P = 1.
     - Gate: the 16-slot ring keeps tg128 and pp512 level with main on one MI50 and on the Radeon VII.
     - Gate: for the ring, the standing device merge gate, every cell against llama.cpp's pinned Vulkan build on the MI50 and on the Radeon VII.
     - Gate: throughput above step 2 at every load from 2 users on 2 to 4 MI50s, with time to first token and inter-token p99 no worse, recorded against step 0's targets.
     - Docs: the loop and cancellation in `docs/SERVER.md`, `--passes` and the health fields in USAGE.
  4. **The sampling pool and in-place rows**, as in Design (Sampling), after `perf/sampler-select` has merged on its own gate: greedy unchanged, seeded CLI equal to the server, and the count of changed draws on a fixed set reported.
     - Gate: outputs byte-identical, and TSan clean over the pool.
     - Gate: step 7's first measurement, recorded.
     - Target, recorded: at 32 to 64 users on 4 MI50s, default sampling within 15 percent of greedy.
     - Docs: the threads and sampling in `docs/SERVER.md`.
  5. **`perf/decode-columns`**, a branch of its own, started 2026-09-26 and built beside steps 1 to 4:
     - Its first commit times a 16-column prototype of `matmul_vec_q8.comp` with `bench --seqs` on one MI50. That time replaces the wider build's estimate in Targets.
     - `matmul_vec_q8.comp`, and the K-quant row kernels the Q4_K_M rows need, gain builds that read each weight once for up to 16 and 32 columns. Builds for 2 and 4 columns load only the columns a pass has, so the pass at 2 to 8 rows loses as much of the 8-column build's fixed cost as it can.
     - The sum-order rule (Decided) holds for every build that merges: every column bit-identical to today's. A faster build that needs another per-column order, fixed and independent of the batch, is built and timed only as a prototype that does not merge, and goes to the user with the count of changed greedy tokens on a fixed set, the HF results (the pinned fixtures and the 8B check, per token included) and the speed gain before any build's order changes. If the user approves, the order changes in every build at once, the one-column build that single-user decode and the CLI take included, so no row computes differently by pass width.
     - Gate: in `backend-vulkan`, every column `memcmp`-equal to the one-column build at 1 to 64 columns, for each type the builds cover, plain, routed and grouped by expert (the grouped path is a pipeline of the same shader). It runs on RADV on the MI50 and on the Windows driver on the Radeon VII, for every build each device takes. The accumulation carries no `precise` qualifier, so each build's ISA is checked for the same multiply and add contraction as today's build.
     - Gate: `generate`, `logits`, `perplexity`, `chat` and the server byte-identical to main on the Radeon VII and one MI50, and `llmx-split-check` bit-identical on 2, 3 and 4 MI50s. Both add a Q8_0 MoE model, Qwen3-30B-A3B Q8_0, which no one card holds: its outputs on two and three MI50s byte-identical to main's on the same splits.
     - Gate: the standing device merge gate, every cell against llama.cpp's pinned Vulkan build on the MI50 and on the Radeon VII.
     - Gate: `bench --seqs` on one MI50 faster than main at 2 to 4 and 9 to 64 rows, and level at 1 and at 5 to 8 unless a build that changes the 8-column pass merges, in which case it is faster there too. It must also be no slower than the reference's batched step, llmx's slower round against the reference's faster one: 52.1 ms at 16 rows and 57.3 at 32, and 27.9 at 8 once the 8-column pass is changed (step 0). The builds as modeled (55.5 and 87.5 ms, and today's 39.5 at 8 rows) miss it at every one of these widths, so the branch needs forms cheaper than the model. At 64 rows the reference's batched step (85.2 to 99.2 ms) beats the two 32-column chunks (up to 175 ms), so a 64-column build is measured. A form fast enough that needs another per-column order goes to the user under the sum-order rule.
     - Targets, recorded whether met or not: the pass costs in Targets, down to 8 rows in 21.5 ms and 16 in 25.0 ms for 8B, which 32B on two cards needs.
  6. **Stage shares that count the head**, a branch of its own after `refactor/loader` merges, taken if step 0 measures the last stage more than 15 percent heavier than the others:
     - The fit gives the last stage fewer layers by the head's measured cost, and by the MTP block's when speculative decoding is on, and the CLI prints the new plan.
     - Gate: every split stays bit-identical to one card on 2, 3 and 4 MI50s, and every changed split plan is listed.
     - Gate: pp4096 on 2 to 4 MI50s level with main.
     - Target, recorded: on 8B, the last stage within 5 percent of the mean stage time at S = 2, 3 and 4, and throughput at S = 4 at 32 and 64 users at least 10 percent above the step before.
  7. **The thread decision, by the rule decided 2026-09-26.** It is measured after step 4 and again after step 5, since a faster pass shortens the round: 8B Q8_0 on 4 MI50s, 16 to 64 users, greedy and at the defaults, host load average under 12. The measurement records the thread's working time; its blocked time in `receive` (the source's ticket and the staging wait apart), in `open` and in the logits wait, each apart; each device's idle time, with what the thread was doing when it went idle; and step 0's pipeline bench driven by one thread and by a thread per stage.
     - At 90 percent of the device-bound rate or more everywhere, nothing changes.
     - Below 90 percent with devices idle while the thread is blocked: first the relay that does not block (sync-file waits or completion-order polling) and P = S + 1, with no contract change.
     - Only if the thread is busy 80 percent of the round or more do per-device submitter threads go to the user, with the numbers. The scheduler keeps all policy, and TSan checks the threads. Command-buffer replay, the other answer to a thread busy recording, goes beside them with its cost.
     - If submitter threads are approved, the step needs:
       - an atomic ticket in the Vulkan backend, published only after `vkQueueSubmit` succeeds;
       - per-device copies of the MoE run vectors;
       - the block pool's bookkeeping on its device's thread, with the scheduler's sequence operations posted there;
       - a failing worker's pass drained on every device before any storage is truncated;
       - `wait` across threads written down as the one exception to `docs/MULTI-DEVICE.md`'s single owner per backend, since a device's thread waits on the previous device's ticket and the scheduler on the output device's;
       - the admission ledger named as the one owner of who holds room;
       - `docs/SERVER.md`'s "the scheduler is that thread" and `docs/ARCHITECTURE.md`'s "a server's scheduler is that thread" reworded;
       - a TSan job.
     - Gate for either change: tests 1 to 6, TSan clean where threads are added, and faster than today at every load (Plan). The share of the device-bound rate at S = 4 is recorded, and while it is below 90 percent the rule above is applied again.
  8. **P from the kernel's column width:** P = max(P0, ceil(D / W)) up to `--passes`, with P0 as in Design (P) and decode rows per pass at most ceil(D / P). It is kept only if it measures better where P = P0 would put more than W rows into a pass.
     - Gate: outputs unchanged.
     - Gate: throughput and inter-token p99 at 17, 24, 33, 48 and 64 users against the step before it.
  9. **Assembly by predicted stage time.** Under long prompts this is a throughput step as well as a latency step.
     - Gate: outputs unchanged, and faster than today at every load (Plan).
     - Target, recorded: on the rate sweep and the 1024/128 closed loop, time to first token p99 and inter-token p99 better than step 8, and throughput higher at 16 to 64 users with 1024-token prompts.
  10. **The final gate and the docs** (Gate, below). Each doc changes in the commit that changes what it describes, as the steps above list; this step marks the phase 3 row of `docs/MULTI-DEVICE.md` done and closes this block. `docs/MULTI-DEVICE.md` already carries what the decisions change there: the slice-boundary rule, one context with P pass slots (the scheduler bullet, the fit and risk 3), one thread owning every backend, the measured decode curve in place of passes free up to about 35 rows, the thread rule and the final gate.
- **Tests:**
  1. `placement` CTest (CPU, every job, UBSan included):
     - Passes are interleaved through the pass API over 2, 3 and 4 CPU stages at P = S, S + 1 and 2S, with five sequences including a forked prefix and prompts sliced at random. Every logits row must be `memcmp`-equal to the same sequence run alone through `prefill` and `step` on one CPU backend.
     - `FailingCpu` fails one pass at a random stage, one of them a pass of several sequences. The failed pass's storages return to where they were, the other passes' rows still equal their runs alone, and a retry equals the run alone.
     - `begin_pass` refuses a sequence already in flight and a pass beyond the reservation.
  2. `server-passes` CTest, a property test of the policy core:
     - A simulated executor drives it with random stage times, arrivals, growth, pauses, cancellations, failures and stops. It covers S from 1 to 4, P from 1 to 2S, and two pools of different block sizes. Cancellations include one that arrives while its request's pass is formed but not yet recorded.
     - After every event, a sequence is in at most one pass and no pass is empty.
     - Each device runs passes in formation order.
     - A handoff buffer and a logits range belong to one pass until it ends.
     - No pool is over-reserved, and nothing in flight is paused, parked, reset, forked from or cancelled.
     - The oldest request is never refused room that younger requests or donors hold.
     - A stall ends within one lap after the requests it waits for leave flight.
     - Every decoder that is not stalled gets a token within one lap, and a cancellation ends within one lap.
     - Admission is first-come, and no free slot idles while work is ready.
     - After stop, the ledger and every pool are at zero.
  3. `server-passes-cpu` CTest runs the real scheduler over the synthetic Q8_0 model on 1, 2 and 3 CPU stages at P = 1, S, S + 1 and 2S:
     - The load mixes prompts, including ones longer than the ubatch, capped and uncapped requests with pauses, and greedy and seeded sampling with `top_logprobs` 5. It adds cancellations in flight, an injected stage failure and a stop mid-run.
     - Every surviving request's ids and logprobs equal its run alone and its run at P = 1, and the CLI's, for a forked prefix too once exact-resume's follow-up branch has merged. Only the failed pass's requests end with "error".
     - Exact-resume's `server-resume` cases rerun at P above 1, including a victim held while in flight.
  4. Replay: the passes test 3 ran are replayed in their order through `forward`, and every logits row must be `memcmp`-equal.
  5. `llmx-split-check` gains a passes phase:
     - P passes go through the pass API with random host delays between stage recordings, so overlap and queue depth vary.
     - Every logit must be `memcmp`-equal to the same passes run serialized and, on the MI50s, to one card. It runs on 2, 3 and 4 MI50s for 0.6B, 8B and 30B-A3B, and on the Radeon VII with the CPU.
     - Qwen3-32B Q8_0, which no one card holds, must be byte-identical on 2, 3 and 4 MI50s and at P = 1 and the chosen P.
     - CPU ops complete on the calling thread, so this is the one test that can see a hazard in device ordering.
  6. `server_mix_check.py` phases (alone, together, skewed, uncapped, and against the CLI) run with logprobs at P = 1, S, S + 1 and 2S:
     - Placements: 1 to 4 MI50s with 8B, 0.6B and 30B-A3B, and the Radeon VII with the CPU.
     - Ids and top logprobs must be byte-equal to each request alone and to the CLI on the same placement, and across one card and 2 to 4 MI50s.
- **Gate**, the layer split's final gate, shared with `perf/decode-columns`, with cards pinned, clocks held high, the host at a load average under 12 and every server run in the same minutes:
  - Servers:
    - llmx at P = 1 and at the chosen P;
    - mx-llama.cpp's ROCm server with its layer split and with its tensor split, the tensor split's log showing RCCL initialized and its custom all-reduce active, since a build without them is 10 to 25 percent slower;
    - llama.cpp's Vulkan server with its layer split;
    - the gfx906 vLLM, as the newest build that runs and as the fastest, each with its version and weights in every table: in 8 bits (GPTQ-Int8 or W8A16, named) against llmx Q8_0 where it runs on gfx906, and in AWQ against llmx Q4_K_M, rerun on the identical checkpoint once llmx loads AWQ.
    - Every table carries each runtime's NLL on the fixed excerpt beside its format, as `docs/MULTI-DEVICE.md` asks.
    - Every reference server gets at least 1152 tokens a slot, since the 1024/128 load needs them and step 0's first reference runs had 1024, and every server gets the context the skewed runs' prompts need.
  - Placements:
    - Qwen3-8B Q8_0 on 1, 2, 3 and 4 MI50s, and Qwen3-32B Q8_0 on 2.
    - Qwen3-8B Q4_K_M on 2, 3 and 4 MI50s and Qwen3-32B Q4_K_M on 2, with every llama.cpp-based reference on the same Q4_K_M file and vLLM on AWQ.
    - Qwen3-30B-A3B Q4_K_M and Q8_0 on 2 and 3 MI50s.
    - On 3 MI50s, where vLLM's tensor parallelism cannot split Qwen3-8B's or 30B-A3B's 32 query heads, vLLM runs with pipeline parallelism 3, named as such, or is reported as not running.
    - On Windows, the Radeon VII with the CPU against llama.cpp's Vulkan build on the Radeon VII with the same layers on the CPU, with the bar in Open.
  - Loads, all through `server_load.py` with the end of text ignored, and fixed lengths except the mixed set's prompts:
    - a closed loop at 1, 2, 4, 8, 16, 32, 48 and 64 users with 128/128, 1024/128 and a mixed set: prompt lengths drawn by `server_load.py --input-len-range LO:HI --seed N` with 128-token replies, where LO, HI and N are recorded in step 0 before any server runs;
    - an open-loop rate sweep up to saturation;
    - skewed runs with long prompts beside short ones, staggered arrivals and clients that leave, their shapes recorded in step 0 before any server runs: four 4096-token prompts arriving together beside users decoding at 128/128, and four prompts near the context every server is given.
    - Every load runs greedy, and once more at llmx's defaults (temperature 0.8, top-k 40, top-p 0.95), sent explicitly to every server with every sampler the request does not name switched off.
  - Reported:
    - output and request throughput;
    - time to first token, inter-token latency and TPOT, at p50 and p99;
    - end-to-end latency;
    - each stage's idle share, the host time per round and the device-bound rate.
  - Pass criteria:
    - Every correctness check above holds, and a follow-up turn on a forked prefix equals the CLI's. That needs exact-resume's follow-up branch (Sequencing), and the gate is not passed before it merges.
    - At 16, 32 and 64 users of every closed-loop set (128/128, 1024/128 and the mixed set), greedy and at the defaults, on every placement of two or more MI50s, llmx's output throughput is at least 2 times each layer-split server's and at least 1.5 times the ROCm tensor split's and the faster vLLM build's, with time to first token p99 and inter-token p99 no worse than each. 48 users is reported beside them, and the rate sweep and the skewed runs are reported against the same servers.
    - "At least N times" means llmx's lower round is at least N times the reference's higher round, over two rounds each.
    - On one MI50, where no split exists, P = 1 is level with main and reported beside the references.
    - At 1 to 8 users, throughput is at least 0.95 times llmx on one card, or, for a model no one card holds, llmx at P = 1 on the same placement.
    - A placement or load that misses its multiple is reported with its gap, and the layer split is not done until the gap is closed.
    - The targets for every placement and load are modeled from step 0's numbers before the gate runs.
- **Open, for the user:**
  - The user's gate names 16, 32 and 64 users but no load. This block applies it to every closed-loop set, greedy and at the defaults, and against the faster vLLM build, until the user limits it. At 1024/128 that needs prefill work that is in neither this phase nor `perf/decode-columns` (Targets): 2 times the layer split on two cards at that load needs llmx's prefill 1.7 to 1.8 times faster per card than today.
  - "Faster than today at every load" (Decided) is read here against main before this phase's first scheduler step, at the commit step 0's tables name. Where passes overlap, a step is above that baseline. Everywhere, it is no worse than the main it merges onto, within two rounds' spread, with time to first token and inter-token p99 no worse. Steps 1 and 2, one user and one card cannot overlap passes, so they are held level. The other reading, above the main each step merges onto at every load, would stop steps 4, 8 and 9, since each helps only some loads. Confirm, or name the bar.
  - The Radeon VII with the CPU: the gate's multiples name MI50 servers, and against llama.cpp's Vulkan build with the same layers on the CPU a two-stage pipeline gains at most (CPU + card) / max(CPU, card) over a serial run, so 2 times needs llmx's stages well ahead of the reference's. Until the user sets its bar, that placement is gated on correctness and on being faster than today at every load (level at P = 1, where the fit does not make the split pipelined), and reported beside the reference.
- **Not doing, and why:**
  - One `ExecContext` per pass: the arena would grow with P. On 8B that is 121.5 MiB per extra pass per device, and a 10-card split measured 2.8 GB against 445 MB.
  - A thread per device by default: it changes the backend's one-thread contract and the block pool's single owner, and it brings new classes of race. Step 7's rule decides it on llmx's own measurements.
  - A process per stage: a C++ host does not need one, and it adds a control protocol, memory shared across processes and failure detection across processes to everything submitter threads need.
  - Buffers picked by start parity under a strict rotation: they are correct only while no pass queues between stages, and nothing checks that at run time. Buffers owned by a slot cost 9 MiB per slot.
  - Command buffers recorded once and replayed: recording is modeled at 2 to 6 percent of a round per stage today, and replay needs every dispatch's parameters in buffers. After step 5 recording becomes the thread's largest share, so step 7 lists replay beside submitter threads. Any replay path must fail closed: mx-llama.cpp once replayed 4 of 108 subgraphs a token and looked several times faster while producing garbage.
  - Sampling on the device: it must match the host sampler bit for bit, and the host pool does not need that. Greedy argmax is the one form considered (Levers).
  - A long prompt's chunks in consecutive passes: that breaks the one-pass rule and is phase 4, after per-storage progress. mx-llama.cpp measured its largest mixed-load gain from exactly this, which is phase 4's case.
  - Moving a running decoder between passes: forming passes at each return already balances them, and a move would need the sequence out of flight anyway.
  - A tuner for P at load: P follows the decoder count by formula, and `docs/MULTI-DEVICE.md` says no tuner runs until measurements show one is needed.
  - More than one pass in flight on a one-device placement: `pipelined_` needs more than one stage, and a pass's tokens come from the pass before it for the same requests.
  - Rewriting `prefill`'s rotation onto the server's driver: the CLI's bytes and phase 2's measurements stay as they are, and phase 4 unifies the two.
  - Tensor groups: phase 6, and not a lever for this gate (Found).
- **Sequencing:**
  - `feat/server-logprobs` merges first, since the tests read logprobs. It also carries the exact-resume block this block cites ("Exact resume of a paused request").
  - `fix/server-cancel` and `fix/server-exact-resume` both edit `scheduler.hpp`. Whichever starts second rebases on the first, as the exact-resume block says.
  - Phase 3's scheduler steps, 2 onward, start from main after exact-resume's steps 2 and 3 and its take-back step have merged. The take-back step, added for the Qwen 3.x plan, gives a resumed request whose own donor survives that donor back whole. The round, the holds and assembly are built on the row classes, the stalls, `make_room` and the take-back.
  - The gate's check that a follow-up turn equals the CLI needs the branch after exact-resume, which stops first admissions forking across row classes. Until that branch merges, tests compare a forked prefix only with the same request run alone.
  - Step 0 changes no model, backend or server code; its pipeline bench needs that `tools/` commit, and its one-card runs are in. Step 1 touches only the model and the fit's host count, so it is built beside exact-resume.
  - `perf/sampler-select`, approved as its own branch, merges before step 4. Greedy is unchanged, seeded draws change once, and its gate reports how many draws differ on a fixed set.
  - `perf/decode-columns` touches only the Vulkan backend's decode builds and their tests. It runs beside steps 1 to 4, merges on its own gate, and step 8's W follows what it merged.
  - The loader (`refactor/loader`) rewrites `PlacementRequest` and the fit's budgets, which step 1's slot count and step 6's shares also touch. Step 6 comes after it, and whichever of the loader and step 1 merges second rebases.
  - `feat/ignore-eos` merged at `7f5ecd3`, before the gate runs, so every server's output lengths are fixed. Step 0's llmx runs so far ended no reply short.
  - The prefill kernels merged at db0f8c3 and `perf/decode-columns` move the one-card numbers, so every gate table names the commit it ran on.
  - The Qwen 3.x plan builds beside this one, with this phase and `perf/decode-columns` first on the cards. Its serve fit follows this phase's server work, and its checkpoints need this phase.
  - Its speculative decoding verifies k drafts as one extent-1 entry of k + 1 rows of one sequence, so verify rows ride in passes as decode rows do. The one-pass rule holds, the rows count against W and in step 8's D, and the MTP block runs on the output device, which step 6's shares count when speculative decoding is on.
  - This phase changes one line of the exact-resume plan. While a request is stalled, the room it needs may be held by capped requests, by requests admitted before it, or by requests held until their pass returns. The exact-resume block takes this wording in phase 3's step 3 commit, or in `fix/server-exact-resume` if that branch has not merged by then.

## Layer split phase 2: a prompt pipelined over the stages (2026-09-25, branch feat/split-pipeline, done)

- **Goal:** phase 2's targets (`docs/MULTI-DEVICE.md`, Order of work): prefill on a layer split about one device's times the stage count, single-stream decode about one device's, and pipelined output exact against the same placement run serialized, so still exact against one device.
- **Measured before any code** (main 5ef32e5, Qwen3-8B Q8_0, clocks held high on two MI50s, two rounds, tok/s): pp4096 on one card 738, 730 and on a 1:1 split 1101, 1103, 1.50 times; tg16 68.2, 68.4 and 67.2, 68.1. Decode meets its target already. Prefill stops at 1.5 times because a stage's work is submitted only when the crossing reads its output, and the host waits there: the first card idles while the host writes the handoff into the second and records the next chunk. Chunks of one prompt overlap only by that accident.
- **Plan**, in order, each step with outputs byte-identical to main before the next:
  1. Stages: the model derives them once from the placement, runs of consecutive layers whose attention sits on one device, each with the cache storage it writes. A placement without a layer split has one stage.
  2. `forward` becomes `begin` (the checks, rows and positions of a pass, in a plan per pass), `run_stage` (the stage's cache blocks prepared and committed, the embedding before the first stage, its layers, the head after the last, and its submission) and `finish`. `forward` runs them in a row, so the server, decode and every test are unchanged. A pass that fails rolls every storage back to where it started, committed stages included.
  3. One crossing in two halves: the source copies the residual rows into a host-visible buffer inside its own submission and keeps the ticket (`send`); the destination waits that ticket and writes the rows (`receive`). Today's `cross` becomes the two back to back. It uses existing backend calls only (`copy`, `submit`, `wait`, `write`). There are two such buffers per used device on a pipelined split, one when crossings happen only inside a stage, which the fit counts in host memory.
  4. `prefill` over more than one stage runs as a software pipeline on the calling thread: step t submits stage 0 of chunk t, then receives and submits stage 1 of chunk t-1, and so on down the stages. Each device runs its chunks in order, so the activation arenas are shared and only the pass plan is kept per chunk in flight. Chunks stay the ubatch, so a split computes exactly what one device computes. Over K chunks and S stages the gain is K·S/(K+S-1): 1.78 for pp4096 over two, 1.94 for pp16384. The handoff subtracts 2.1 to 2.7 ms per 512-row chunk (phase 0) from stage times near 350 ms.
- **Done:** steps 1 to 4 as planned, plus:
  - A device's attention layers must form one run, or the placement is refused, so each storage is written by one stage; `ensure()` waits for a context's last submission on a device before it replaces storage a pass may still use.
  - The fit gives the busiest device as few layers as the budgets allow before it balances bytes: it had placed 12/13/11 on three MI50s and 8/10/10/8 on four, because the embedding and head sit on the end devices, and the pipeline runs at its slowest stage.
  - `llmx-split-check` takes a device list and a ubatch; `tools/server_mix_check.py` checks many users at once.
- **Exact:** `llmx-split-check` with 64-token chunks, so the prefill is pipelined, is bit-identical to one card on 2, 3 and 4 MI50s for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M, with an f32 cache and with the f16 cache users run. CTest, and the suites on one card and 2- and 3-card splits on the MI50s and on the Radeon VII with the CPU, pass. Many users through the server (16 requests, prompts and decodes together, then skewed with four clients leaving) match their text alone and through the CLI on 4 cards (8B) and 3 cards (30B-A3B, 0.6B). The 16k greedy check on a 2-card split: the same 512 tokens from two fresh servers, the CPU's top choice 508 of 512 times, largest gap 0.177 logits.
- **Measured** (Qwen3-8B Q8_0, clocks held high, cards pinned, two interleaved rounds, tok/s; the reference is mx-llama.cpp on the same cards, its layer and tensor split):

  | cards | llmx pp4096 | llmx pp16384 | llmx tg | mx pp4096, layer / tensor | mx pp16384, layer / tensor | mx tg, layer / tensor |
  |---|---:|---:|---:|---:|---:|---:|
  | 1 | 746, 715 | 442, 419 | 67.6, 67.8 | 1168 | 812 | 71.0 |
  | 2 | 1225, 1217 | 821, 812 | 65.9, 66.2 | 1890 / 1823 | 1467 / 1375 | 70.5 / 107 |
  | 3 | 1686, 1678 | 1224, 1212 | 65.7, 66.2 | 2542 / 2298 | 2068 / 1768 | 70.4 / 126 |
  | 4 | 2010, 2011 | 1546, 1553 | 64.7, 65.3 | 3021 / 2311 | 2602 / 1916 | 69.7 / 134 |

  Four cards prefill 16k tokens at 3.5 times one card and 4k tokens at 2.7 times, the pipeline's bound for chunks that grow with depth (the last chunk drains through every stage alone); main's split had given 1.5 times. Decode on the split is 96 to 98 percent of one card. The split scales as the reference's layer split does (2.6 to 2.7 times on four cards at pp4096), so the gap to it is the single card: its prefill is 1.56 times llmx's at 4k tokens and 1.92 times at 16k, and on one card 86.6 percent of llmx's pp512 device time is the integer-dot matmul tile (`matmul_tile_q_tall`). Its tensor split decodes at 107 to 134 tok/s, which llmx's layer split cannot: that is tensor groups (phase 6).
- **Measured and not taken:**
  - A deeper command ring (16 or 32 slots): it took the host's block in recording away but moved throughput under 2 percent, since every device already had its next chunk queued.
  - Recording the next stage before its rows arrive (a held input the host fills before the submit): decode on 3 and 4 cards gained under 1.5 percent, 2 cards lost a little in decode and prefill; a stage's first 64 dispatches already start while the rest is recorded.
- **Left, in later phases:** passes of different sequences in flight (phase 3) with the server gate against the reference's server; per-storage progress visible to the scheduler (phase 4); tensor groups for multi-card decode (phase 6); a thread per stage only if the host is measured to limit a pipeline; pipelined scoring for perplexity. The single-card prefill kernels come after the correctness-bug branches and the loader.

## Quantization coverage: 16-bit, MXFP4, IQ4, Q3_K and Q2_K weights (planned 2026-09-25)

- **Goal:** llmx opens and runs the GGUF files people download for the architectures it runs today and for the ones planned next (Qwen3.5, 3.6 and 3.8, [QWEN35](QWEN35.md)), on the CPU and on both Vulkan cards.
  Each new tensor type must meet six conditions:
  - it is exact against a decoder written independently from the format description;
  - it is within HF bounds on a model a host here can hold;
  - it is batch-invariant, and prefix reuse with it stays exact;
  - it is at least level with the reference's Vulkan build on the MI50 and on the Radeon VII, and with the reference's CPU build on the Linux host's CPU;
  - every existing type keeps byte-identical outputs and its speed;
  - types are taken in the order of the files they open and how often those files are published, weighed against the effort.
- **Priority:** built now in the background, following AGENTS.md and docs/ARCHITECTURE.md, with a full review of every step.
  Layer split phase 3 keeps priority on the cards, so a branch's device gates and speed cells run when the cards are free of it (user, 2026-09-25).
- **Decided (2026-09-25),** numbered as the plan's questions were:
  - **1. Order:** 16-bit, MXFP4, IQ4, Q3_K, Q2_K.
    MXFP4 goes before IQ4 by the user's choice, although IQ4 opens far more files and the order rule above put IQ4 first.
    So MXFP4's branches write the shared 16-entry lookup that IQ4's then reuse, in C++ and in GLSL.
    MXFP4 also comes before DeepSeek 4.x and does not wait for it.
  - **2. Fixtures:** six unsloth Qwen3-0.6B files join the HF gate, from the revision of the Q4_0 and Q5_K_M fixtures (50968a44, 3.41 GB): BF16, UD-Q8_K_XL, IQ4_NL, IQ4_XS, Q2_K and Q3_K_S.
    BF16 is gated as byte-identical to the existing Qwen3-0.6B-F32.gguf once its widened tensors are shown equal to it, so it inherits that file's bounds (max logit error 0.001, NLL 1e-4).
    UD-Q8_K_XL is held at Q8_0's bounds, and the lower types' bounds are measured on the CPU and proposed then.
    Only UD-Q8_K_XL, IQ4_XS and Q2_K (1.51 GB) go into the hosted HF job, since the tiny models, IQ4 dense and MoE ones included, cover every new type in CI.
    Those three enter `tests/data/fixtures.json` with their repo (`unsloth/Qwen3-0.6B-GGUF`), revision and SHA-256, and the other three are pinned the same way in ASSETS.
  - **3. File-exact reference:** required for MXFP4, IQ4, Q3_K and Q2_K before their bounds are proposed.
    HF runs on the fixture's weights as the numpy spec decoder decodes them, held to Q8_0-class bounds, since bounds against the fp32 reference alone are too wide at 2 to 4 bits to catch a mis-scaled sub-block.
  - **4. MXFP4 fixtures:** a deterministic Python writer in `tools/` writes all-matrix MXFP4, the tied embedding included, from the pinned 0.6B BF16 file.
    It uses the OCP scale rule, with round-to-nearest-even and saturation, all written down.
    Its sha256 goes in ASSETS, and the file is checked by hand, not hosted.
  - **5. MXFP4 edge values:** -0 decodes as +0, and e = 255 as 2^127 by the same construction as every other exponent, with no scan at load, both documented.
    e <= 1 keeps its subnormal scale on every backend, with DenormPreserve on the devices.
    If either driver does not offer fp32 denormal preservation, every backend flushes e <= 1 scales to zero, the CPU and the spec decoder included: a documented departure from the spec for values below 3 x 2^-126 that keeps every path bit-identical.
  - **6. CPU before Vulkan:** a type's CPU branch may merge while Vulkan refuses that type at load, until its Vulkan branch merges.
    Step 0's load-time device check makes the refusal explicit, streamed layers of that type stay on the host, and the skipped device fixtures are recorded.
  - **7. 8-bit CPU activations:** MXFP4, IQ4, Q3_K and Q2_K may use 8-bit activations on the CPU for non-head rows where they pass the bounds and gain speed, both recorded.
    Their heads always read 16 bits, and the existing types' heads keep today's path.
  - **8. 16-bit identity:** a 16-bit file equals an F32 file of the widened weights, bit for bit, on the CPU and each card.
    If a driver fuses the 16-bit build differently and `precise` would change F32's bytes or speed, the work stops and the user is asked; the fallback then is equality with the CPU within F32's own device bounds.
  - **9. 16-bit prefill:** if llmx's lossless f32 float tile for 16-bit weights falls short of the reference at pp512 or pp4096, that is accepted and recorded, with decode and pp64 still required at least level.
    No lossy path is added unless the user asks for one as an opt-in, which every backend then honours identically or refuses.
  - **10. Covered by the exact-reuse rule:** rows are reused only where they were computed as the CLI would compute them.
    Its owner is exact resume's first-admission branch, and step 0's kernel-class function is offered to it.
    Until then this plan's server checks run with no donor hits or pauses, which each check asserts.
- **Open, not yet asked:**
  - **11.** Should IQ3_S be the first type after this plan, as its own plan?
    It opens the Qwen3.5/3.6-35B-A3B UD-IQ4 files.
    Copying the IQ1/IQ2 codebooks raises a licence-notice question against the rule on naming other projects in code, to settle then.
    Recommendation: yes.
- **Found:** from main d48f2b2, from GGUF headers of the Linux host's model files, from Hub headers read by range requests, and from samples of 40,000 blocks per tensor.
  No model was run.
  The first draft read db0f8c3; the eight commits since then change only the KV storage in `src/`.
  Main has since moved to b7b585f, where `cleanup/backend-contract-helpers` merged as 6aac250 and moved lines in the backends, so the line numbers below are those of d48f2b2 unless b7b585f is named.
  - **The reader today.**
    - llmx reads F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K and Q6_K.
    - `gguf::TensorInfo::data_size` throws on any other type, and it runs for every tensor when a file is opened.
      So one tensor of another type makes the whole file unreadable, `info`, `tokenize` and `detokenize` included.
      The message names neither the type nor the tensor.
    - Type sizes are written in `format/gguf.hpp`, in the registry (`quant/quant.hpp`) and again in `q.glsl`.
    - Vulkan refuses a type it has no kernel for only when the kernel is first called ("unsupported matrix type").
      `tests/common.py:device_lacks_kernel` turns that message into a SKIP under `LLMX_DEVICE`, so a device run can pass with nothing checked.
    - Six checks use F16 (type 1) as the unsupported type, at b7b585f: in `backend_vulkan.cpp`, the matmul at 983, and `check_refusals`' matmul (1537), routed product (1587) and embed (1595) through its F16 constant (1444); and `model_validation.cpp:413` and `:421`.
  - **Work in flight that this plan touches.**
    - `feat/hf-loader-main-20260924` (b56e4e3, 234 commits behind main) has a `core/storage.hpp` that is only named constants.
      It has no lookup table, registry entries for F16 and BF16 with a CPU dequantize fallback, and a refusal of 16-bit weights on Vulkan.
    - `perf/prefill-kernels`: its commits are on main (4c17361 to 68486a4).
      Its STATUS block still owes the Radeon VII gate and the merge cells pp247 and tg32.
    - `cleanup/quant-owner` (eaf1118) and `cleanup/vulkan-constants` are on main.
    - `cleanup/backend-contract-helpers` (14) merged as 6aac250 after this plan was read.
      It adds `quant::row_bytes` and sets F32's block size to 1.
    - `cleanup/cpu-kernels` (16) has no branch yet.
    - The loader plan's step 4 ("plan, then fill") checks every role before any byte moves.
      That plan lists checking each device's weight types at load as a follow-up it does not do.
  - **Silent defaults that would send a new type to another type's kernel:**
    - CPU:
      - `q8_dots.hpp`: the `default:` of `dot_block` (544) and of `dot` (556) go to Q6_K;
      - `q8::reads16` (499);
      - `row_dot`'s default (1241), which goes to `dot_row_dequant`, accumulating in double;
      - `is_kquant` (1118).
    - Shaders:
      - `embed.comp`: its `else` goes to `q6_k_at`;
      - `q.glsl`: `block_bytes` falls through to Q6_K's size, and `block_values` returns 32 for every type that is not a K-quant;
      - `matmul_tile_q.comp`: its `else` goes to Q4_0/Q4_1;
      - `matmul_tile.comp`: the float tile's last `else` reads weights as F32 (146);
      - the `matmul_row.comp` plain build treats any type other than Q8_0 as F32 (281);
      - `qtile`: its last branch.
    - Host side of Vulkan:
      - `row_plan`'s `default: break` (1596);
      - `row_twin` gives the raw activations only to F32 (1620), so a 16-bit row would get the quantized twin;
      - `reads_x8` (1851), `is_row_kernel`, `row_kernel_builds_one_column` and `row_dot_variant`'s default are hand-written lists, and a new 8-bit row kernel missing from `reads_x8` reads the 16-bit twin as bytes;
      - `integer_dot_tile` is true on the MI50 for every type except F32;
      - `tile_from`'s choice between "8-bit" and "float" (1383);
      - `moe_tile_from_for`'s default (device_profile.hpp:131).
    - Test helpers place scales by `type >= Q4_K`, and `d` at offset 0 or 2.
  - **Modules shared by every type through a push constant.**
    `embed` and the float tile branch on `p.type` at run time and read `q.glsl`'s `block_bytes`, `block_values` and `is_kquant`.
    So adding a type to them changes their SPIR-V.
  - **`tile_reads`** takes the maximum of every type's threshold (1956-1957).
    A new, higher MoE threshold would move, for every model, the range of prompts in which the producers write the 8-bit copy.
  - **Device buffers** are rounded only to whole words (vulkan_backend.cpp:581).
  - **Vulkan `matmul_experts`** throws when the gate and up experts differ in type (1721).
    No file read mixes them.
  - **Streamed experts:** with experts on the host, long prompts copy the expert stacks into device windows and compute them there (`arch_qwen.hpp ffn_split`).
  - **Float controls:** nothing in `src/` or CMake requests Vulkan float controls.
    RADV flushes fp32 denormals unless a module declares DenormPreserve.
    The CPU sets no flush-to-zero.
  - **CPU paths.**
    - Decode takes `dot_f32` for F32 only.
      Other types without a fused dot are dequantized and take `dot_f32_x4`'s order (426-470).
    - MoE decode calls `row_dot`.
    - The prompt dots run only for K-quants at 4096 wide and up (`kPromptDotsFrom`), so they never run on the 0.6B files.
    - MoE prompt entries take `dot_block` for any type with a fused dot, whatever the width.
    - The CPU has no `matmul_logits` override: the head falls through to `matmul` (backend.hpp:195).
  - **Heads.**
    The Q4_0 fixture failed top-5 because of its Q6_K head on the 8-bit twin (VULKAN.md).
    The Vulkan `matmul_logits` keeps the 16-bit twin for Q4_0, Q4_1 and Q6_K heads, and every other row of those types reads 8 bits.
  - **Server.**
    Prefix reuse chooses a donor by token match alone (`scheduler.hpp best_donor`).
    A new prompt's own rows follow its whole prompt's extent, as the CLI's do.
    A donor's reused rows were computed at the donor's extent, and its generated tokens' rows at extent 1, so their kernel class can differ from the CLI's.
    Pause and resume has the known gap recorded in the second code audit's block below.
  - **Radeon VII limits.**
    - Its float tile is one module for every type.
      It uses 67 registers, three more than a fourth wave allows.
    - Its plain float row path is exact only while a block's partial sums stay below 2^24.
  - **Qwen3.x files on the Linux host that need no new type:**
    - Qwen3.5 0.8B, 2B, 4B and 9B Q4_K_M.
    - Qwen3.6-27B Q4_K_M, Q8_0 and pure Q4_K.
    - Qwen3.6-35B-A3B Q4_K_M, Q5_K_M, Q6_K and Q8_0.
    - Qwen3.8-27B Q8_0.
    - These hold only F32, Q8_0, Q4_K, Q5_K and Q6_K, so the Qwen3.x architecture work does not wait on this plan.
  - **Qwen3.x files on the Linux host that do need a new type:**

    | file | architecture | new type it needs |
    |---|---|---|
    | Qwen3.8-27B UD-Q8_K_XL | qwen35 | BF16, 5.14 GB (16.4 percent): full-attention q, k and v in 17 layers, the head, the multi-token-prediction projection |
    | Qwen3.5-35B-A3B UD-Q8_K_XL | qwen35moe | BF16, 3.84 GB: every matrix outside the routed experts |
    | Qwen3.5-122B-A10B UD-Q8_K_XL | qwen35moe | F16, 87.4 GB (51 percent), all 48 routed down projections included |
    | Qwen3.6-27B BF16 | qwen35 | BF16 throughout, 54.6 GB |
    | Qwen3.6-35B-A3B UD-Q4_K_XL, the copy with the prediction layer | qwen35moe | BF16 in two router tensors of layer 40 only; the published file (Hub a483e9e6) has none |
    | Qwen3.6-35B-A3B IQ4_NL | qwen35moe | IQ4_NL on experts, shared experts, the linear-attention matrices and the embedding |
    | Qwen3.6-27B and 35B-A3B MXFP4, local requantizations | qwen35, qwen35moe | MXFP4: 498 tensors (27B), 120 expert tensors (35B-A3B) |
    | Qwen3.6-27B Q5_1 | qwen35 | Q5_1 on 497 matrices |

  - **Files of architectures llmx runs today:**
    - **Linux host, local requantizations** (in a review directory, sha256 to be recorded, not reproducible):
      - Qwen3-14B IQ4_NL, 8,597,069,184 bytes.
        The Hub's main file is 8,541,363,584, so gates use a pinned Hub download instead.
      - Qwen3-14B MXFP4, with every matrix, the embedding and the head in MXFP4.
      - Qwen3-14B Q4_K, Q5_K and Q6_K beside them, which calibrate the device-against-CPU criterion.
    - No file on the host holds Q2_K, Q3_K or IQ4_XS.
    - **Qwen3-30B-A3B UD-Q8_K_XL:** BF16 gate, up and down experts in 5 layers, BF16 attn_v in all 48 layers, and a BF16 embedding and head.
    - **Heads of the pinned files:**
      - The 0.6B IQ4_NL, IQ4_XS, Q2_K and Q3_K_S files all have a tied Q6_K token_embd.
      - The 8B and 30B IQ4, Q2 and Q3 files all have a Q6_K output.
      - 8B Q2_K and 30B-A3B Q2_K have a Q2_K token_embd.
      - So no real fixture puts IQ4, Q2_K or Q3_K in the head.
  - **Other architectures on the Linux host:**
    - gpt-oss-120b: MXFP4 experts in rows of 90 blocks.
    - DeepSeek-V4-Flash: 129 MXFP4 expert tensors and a BF16 router.
      Its UD-Q8_K_XL holds 14.5 GB of BF16.
    - DeepSeek-V4.1-Flash: 120 MXFP4 expert tensors and two MXFP4 embedding tables.
    - Qwen3.8-Flash-Next (qwen4exp): a 28.8 GB IQ4_NL per-layer embedding, Q5_1 expert down projections 640 wide, and BF16 indexer projections.
    - Two Qwen3.6-27B files use type 53, a research format.
  - **MXFP4 block samples:**
    - Exponents run from 116 to 127, and never 255.
    - Code 8 (-0) is 5.9 to 8.6 percent of values in checkpoints trained in MXFP4 (gpt-oss, the DeepSeek experts).
      Converted files never use it.
    - In Qwen3.6-35B-A3B's layer 0 gate experts, 26,432 of 65,536 sampled blocks have exponent 0 with all-zero codes, and 13 of 256 experts are entirely exponent 0.
    - No sampled block has an exponent of 0 or 1 with a nonzero code.
  - **Hub headers:**
    - unsloth Qwen3-0.6B at 50968a44, the revision of the Q4_0 and Q5_K_M fixtures, publishes seven relevant files:
      - BF16;
      - UD-Q8_K_XL (F16 plus Q8_0);
      - IQ4_NL (381,566,656 bytes);
      - IQ4_XS;
      - Q2_K, which holds Q2_K and Q3_K;
      - Q3_K_S;
      - Q3_K_M.
    - Every unsloth Qwen3 and Qwen3.x repo read publishes IQ4_NL, IQ4_XS, Q3_K_S, Q3_K_M and the UD-Q2/Q3_K_XL files.
    - MXFP4 appears only as MXFP4_MOE for Qwen3.5-35B-A3B and Qwen3.6-35B-A3B.
      The 3.6 file is 21.7 GB: MXFP4 gate and up experts in 39 layers, Q5_K and Q6_K down experts, and Q8_0 elsewhere.
    - For qwen3moe there is a community Qwen3-Coder-30B-A3B MXFP4_MOE: 17,082,450,208 bytes, MXFP4 on all 144 expert tensors and Q8_0 elsewhere.
    - UD-Q4_K_XL:
      - 0.6B, 8B and 32B need IQ4_XS on 20 to 28 tensors (q, k, gate and up of five to seven layers), and nothing else new.
      - Qwen3.6-27B needs IQ4_XS on 12 tensors.
      - Qwen3-30B-A3B (d5b1d57b) and Qwen3.6-35B-A3B hold only types llmx reads.
    - Qwen3-235B-A22B UD-Q2_K_XL (88.0 GB in two shards) has Q2_K gate and up experts, Q3_K down experts, and Q4_K, Q5_K, Q6_K and F32 for the rest.
      The first of the three shards of its UD-Q3_K_XL (103.7 GB) adds Q3_K alone.
    - The 30B-A3B UD-Q2_K_XL (11.8 GB) and UD-Q3_K_XL (13.8 GB) need only Q2_K and Q3_K.
    - Dense UD-Q2_K_XL files also need IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS (Qwen3-32B read).
    - Qwen3.6-35B-A3B's UD-IQ4 files put IQ3_S on the gate and up experts.
  - **Formats**, from the GGUF type definitions and the OCP MX specification:
    - **F16 (1) and BF16 (30):** 2 bytes per value, and widening to f32 is exact.
      0.18 to 0.22 percent of the F16 weights in real files are subnormal.
    - **IQ4_NL (20):** 32 values in 18 bytes.
      An f16 scale, then nibbles in Q4_0's order that index the 16-entry int8 table {-127 ... 113}.
    - **IQ4_XS (23):** 256 values in 136 bytes, as eight sub-blocks with signed 6-bit scales.
      99.66 percent of the 0.6B file's super-blocks have a subnormal f16 scale.
    - **MXFP4 (39):** 32 values in 17 bytes.
      An E8M0 exponent comes first, then nibbles in Q4_0's order that index the doubled E2M1 values {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12}.
      - Each value is table[c] x 2^(e-128).
      - For e <= 1 the scale is an f32 subnormal (2^-128 or 2^-127).
        The plain bit construction turns e = 0 into -inf, and -inf x 0 is NaN, so the scale needs its own branch there.
      - Values overflow to inf only at e >= 253.
    - **Q3_K (11):** 110 bytes, with signed 6-bit scales per 16 values and values from -4 to 3.
    - **Q2_K (10):** 84 bytes, with a 4-bit scale and min per 16 values.
    - **Exactness.**
      Every product these decoders form is exact in f32, and Q2_K's scale minus min is one rounding that is exact in double.
      So every correct decoder gives the same bits, as long as subnormals are kept.
      f16 subnormal scales widen to normal f32 values; the Q4_0 fixture's subnormal Q6_K scale already passes on both cards.
      MXFP4's e <= 1 scales stay subnormal in f32, so they need float controls on a device (Design).
  - **Speed bar.**
    - The reference's Vulkan source has integer-dot matmul and mat-vec kernels for MXFP4, Q2_K and Q3_K.
      They are compiled only under `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` and run only on a device with integer dot.
      IQ4 goes through dequantization to f16.
    - For F16 weights, its matmul pipeline accumulates in f16 when precision is default and the device has fp16, which both cards do.
      Its BF16 path converts activations to BF16.
      Both are lossy.
      llmx's lossless f32 float tile, which the BF16 rule requires, may therefore not reach it at pp512 and pp4096.
    - mx-llama.cpp's ROCm build has gfx906 tuning for MXFP4 and an IQ4_NL prefill repack.
  - **Radeon VII fit.**
    30B-A3B Q2_K does not fit at the model's 40,960-token context.
    The 11.26 GB file, plus the f16 cache at 96 KiB a token (3.75 GiB), plus the reserve (256 MiB and a twentieth of free memory) exceed 16 GB.
  - **HF container:** transformers 4.55.2, the version the 8B goldens record.
- **Order:**

  | types | branches (size) | opens on qwen3 and qwen3moe | opens later | shares kernels with |
  |---|---|---|---|---|
  | F16, BF16 | 1 `feat/half-weights` (medium) | every BF16/F16 Qwen3 file; the UD-Q8_K_XL family (0.6B, 8B, 30B-A3B, Coder-30B-A3B, 30B-A3B-2507, 4B-2507); native HF weights on devices | Qwen3.5/3.6/3.8 UD-Q8_K_XL and BF16, F16 for the 122B; DeepSeek-V4 UD | F32's plain row build, float tile, embed and CPU decode dot |
  | MXFP4 | 2 `feat/mxfp4-cpu` (small), 3 `feat/mxfp4-vulkan` (medium) | Qwen3-Coder-30B-A3B MXFP4_MOE; the local Qwen3-14B MXFP4 | unsloth Qwen3.5/3.6-35B-A3B MXFP4_MOE; gpt-oss and DeepSeek-V4/V4.1 experts, with those architectures | Q4_0's nibble order; the 16-entry lookup, which it writes for IQ4; the Q8_0 module's scale-only math |
  | IQ4_NL, IQ4_XS | 4 `feat/iq4-cpu` (medium), 5 `feat/iq4-vulkan` (large) | IQ4 files of every Qwen3 size up to 235B-A22B; dense UD-Q4_K_XL (0.6B, 8B, 32B); the pinned Qwen3-14B IQ4_NL | Qwen3.6-27B IQ4_XS and UD-Q4_K_XL; Qwen3.6-35B-A3B IQ4_NL; Qwen3.5 IQ4 | IQ4_NL: Q4_0's staging and row pairs; IQ4_XS: Q4_1's aligned staging, KQ45_K's step with no min; both: MXFP4's 16-entry lookup |
  | Q3_K | 6 `feat/k-low-cpu` (medium, with Q2_K), 7 `feat/q3k-vulkan` (medium) | Q3_K_S/M/L of every Qwen3; the first shard of the 235B-A22B UD-Q3_K_XL | Qwen3.6-27B Q3_K_M; Qwen3.5-0.8B Q3_K_S | Q6_K's signed per-16 scales, KQ6_K dots, Q6 tile math and row lanes |
  | Q2_K | 6, 8 `feat/q2k-vulkan` (large) | Q2_K of every Qwen3; UD-Q2_K_XL of 30B-A3B and 235B-A22B | none planned | Q3_K's per-16 sums; KQ45_K's unsigned dot with a min |

  - Branch 0, `refactor/storage-types` (medium), comes before all of them.
  - MXFP4 writes the shared lookup, in C++ in branch 2 and in GLSL in branch 3, and the IQ4 branches reuse it.
- **MXFP4 comes before DeepSeek 4.x and does not wait for it:**
  - **It can be gated on qwen3 and qwen3moe:**
    - raw blocks against a decoder written from the OCP bit fields;
    - tiny HF models built from random MXFP4 blocks;
    - a real Qwen3-0.6B written in MXFP4, tied embedding included, against HF;
    - the local Qwen3-14B MXFP4 and the Hub's Qwen3-Coder-30B-A3B MXFP4_MOE on each device against the CPU.
  - **What DeepSeek's files add is covered without that architecture:**
    - Code -0 is tested at the rate the native checkpoints show.
    - The MXFP4 embedding and head are covered by the 0.6B writer's file and the 14B's.
    - Its expert widths (128, 64, 160 and 72 blocks per row) are multiples of four blocks, like every Qwen width.
      gpt-oss's 90-block rows take the general path, which a tiny-model width covers.
  - **Why the type lands first:** an architecture branch then finds its tensor types already proven, so a failed HF bound there points at the architecture.
  - **First public file:** unsloth's Qwen3.6-35B-A3B MXFP4_MOE, which the Qwen3.x MoE step can run once branch 3 lands.
    DeepSeek 4.x follows Qwen 3.x.
- **Design:**
  - **One owner per rule:**
    - **Storage table.** `core/storage.hpp` holds one table over every GGML type id: name, values per block and bytes per block, with F32 at a block size of 1.
      - `gguf::data_size`, the registry, `quant::row_bytes` and the safetensors reader all read it.
      - Support is a separate predicate for each backend, not part of the table.
      - A CTest asserts that `q.glsl`'s type ids and sizes equal the table's.
    - **Decoders.** Each type has one C++ decoder in `quant/` and one GLSL decoder in `qdecode.glsl`, with its sizes in `q.glsl`.
      The embed checks prove the two agree bit for bit.
    - **Lookup.** MXFP4, IQ4_NL and IQ4_XS share one lookup, parameterised by the table, in each language.
      MXFP4's branches write it, and IQ4's reuse it.
    - **Python.** One Python module holds a spec decoder for each type in a pure form and a numpy form.
      The numpy form is checked against the pure one on the raw-block corpus.
      `roundtrip.py`, the reference generator and the tiny-model builder use it.
    - **Kernel choice** is an explicit table per backend (row family, tile module, row twin and thresholds), with no default branch.
      `KernelId`, `kKernelNames` and `kKernels` become one list.
    - **Refusal.** A type missing from a backend's table is refused at load, naming the type, the tensor and the device.
    - **Float types.** One predicate names the float storage types F32, F16 and BF16.
      It feeds `row_twin`, `row_plan`, `integer_dot_tile`, `tile_from`'s choice between 8-bit and float, `moe_tile_from_for` and the CPU's decode branch.
    - **Kernel class.** The profile table owns one "kernel class of (model, extent)" function covering `tile_from`, `split_tiles_of` and `moe_tile_from`.
      `tile_reads` reads it, and exact reuse can key donors on it (Decided, 10).
    - **Per-16 sums.** Per-16 activation sums for Q2_K and Q3_K have one producer.
    - **Device-against-CPU criterion.** It is written once in `tests/common.py`, beside `top5_overlap`.
      Over the pinned excerpt:
      - top-1 is equal wherever the CPU's top-2 gap exceeds the tie margin;
      - top-5 overlap is 5/5 with the margin;
      - NLL is within 0.01 of the CPU's;
      - the largest logit gap is no larger than an existing type of the same model or architecture shows, device against CPU;
      - the 64 greedy tokens are equal up to the first near-tie.
  - **Storage.** Weights stay in their file's type on every backend.
    They are read in place on the CPU and uploaded unchanged to a device; nothing is converted at load.
  - **Shader modules.**
    - Every new type gets its own builds of each module it needs: embed, the float tile, the plain rows, the integer tile and the routed builds.
      They are built from the existing sources under a define, so the existing SPIR-V stays byte-identical.
    - A hash list of existing modules decides whether that holds.
      Any module whose hash changes is A/B'd level on both cards.
  - **CPU.**
    - **16-bit types.** Fused widen dots in `dot_f32`'s order: F16 through F16C, BF16 by a 16-bit shift.
      They are used in `row_dot`, in `matmul_raw`'s decode branch beside F32's, and in `matmul_group`'s single-entry path.
      Prompt rows are dequantized into the row buffer and take F32's four-row kernels, which is F32's path over the same values.
      So a BF16 file gives the same bits as an F32 file of the widened weights, MoE decode included.
      16-bit rows never take the prompt dots or 8-bit activations.
    - **MXFP4 and IQ4** dots look codes up with `pshufb`.
      Q3_K takes KQ6_K's shape, and Q2_K takes the unsigned dot with mins.
    - **Heads.** A CPU `matmul_logits` override reads 16-bit activations for heads of the new types only.
      Q8_0, Q4_K and Q5_K heads keep today's path and bytes.
    - **Non-head rows** start at 16 bits.
      A type moves to 8 bits only if it passes the HF bounds and gains speed, and both are recorded (Decided, 7).
    - **Prompt dots.** Whether prompt rows take the prompt dots is decided by the matrix alone.
      A type that joins them makes its 8B file a required gate, since the 0.6B rows are narrower than 4096.
  - **Vulkan on the MI50.**
    - Integer-dot tile modules per family:
      - MXFP4 and IQ4_NL are modelled on the Q8_0 module: one scale, no min term.
      - IQ4_XS uses Q4_1's word-aligned staging.
      - Q3_K uses the Q6 module.
      - Q2_K uses a Q6-shaped module with a min term.
    - Decode rows read the 8-bit copy, and heads read the 16-bit copy through `matmul_logits`.
    - 16-bit types stay on the float tile and the plain row build, with F32's row plan (units, cluster, rows per group) and F32's `tile_from` and `moe_tile_from`.
  - **Vulkan on the Radeon VII.**
    - Each new family gets a float-tile build and a plain row build.
    - MXFP4 uses signed 16-bit lanes, whose sums stay at or below 12.58M.
    - IQ4's partial sums reach 66.6M, past 2^24, so its rows use integer byte dots, or split each table value into a high plane and a low plane.
    - Q3_K and Q2_K fold their offsets through the half sums, as Q6_K does.
  - **16-bit exactness on the cards.**
    - The 16-bit builds share F32's source and accumulate expressions, with only the load swapped.
      The gate requires a BF16 file to equal the F32 file bit for bit on each card.
    - If a driver fuses the two builds' multiply-adds differently, `precise` on the accumulate in both builds is the fallback.
      It is taken only if F32's outputs stay byte-identical and F32's speed stays level on that card.
      Otherwise the work stops and the user is asked (Decided, 8).
  - **MXFP4 edge values.**
    - -0 decodes to +0, which is what every integer path computes.
    - e = 255 decodes to 2^127 by the same construction as any other exponent, with no scan at load.
    - The scale is never held as f16.
    - e <= 1 keeps its subnormal scale:
      - the CPU sets no flush;
      - MXFP4 modules declare DenormPreserve for 32-bit floats, through GL_EXT_spirv_intrinsics (glslc targets Vulkan 1.2), where the driver reports `shaderDenormPreserveFloat32`;
      - embed builds value bits with integer operations, which is exact without float controls.
      - Both drivers are queried first.
        If either lacks the property, every backend flushes e <= 1 scales to zero, the CPU and the spec decoder included (Decided, 5).
  - **16-bit rows a power of two wide** may need a padded copy on the float tile, as F32 rows do: without one, F32 rows ran 7 to 10 times slower.
    This is measured first.
    If padding is needed:
    - one padded upload goes through the loader's upload entry, with no second copy;
    - every reader takes the padded addressing: the plain rows, the embed of a tied F16 or BF16 embedding, and the routed stacks, which are bound unpadded today;
    - `resident_bytes` counts the padded size in place of the unpadded, rather than adding it.
  - **Batch invariance.**
    - Every threshold lives in the profile table.
    - `tile_reads` takes its threshold from the loaded model's weight types, fixed at load.
      It is still independent of the batch, and existing files are measured level.
    - Kernel choice follows the prompt's extent, never the batch.
      Each tile module gives the same bytes at its 32, 64 and 128-row heights, and each row family gives the same bytes in its one-column, wide and grouped builds.
  - **Loading and placement.**
    - Open reads any type in the table, so `info`, `tokenize` and `detokenize` work on every file.
    - Model construction refuses a tensor whose type the backend holding it cannot run, used or not, naming the type and the tensor.
      The device check sits in the loader's plan-then-fill step.
    - A routed layer whose expert type its stream device lacks gets no stream device and runs on the host, which `--verbose` names.
    - Vulkan refuses at load a routed layer whose gate and up experts differ in type, naming the layer.
  - **HF references.**
    - Each fixture is checked against the existing fp32 reference with its own bounds.
      This measures the format's loss.
    - For MXFP4, IQ4, Q3_K and Q2_K, a file-exact reference also runs before their bounds are proposed (Decided, 3).
      HF runs on the fixture's weights as the numpy spec decoder decodes them, which holds llmx's own arithmetic to Q8_0-class bounds.
      Its goldens are committed under `tests/data` and stay checked after the merge: the hosted job checks the IQ4_XS and Q2_K files against theirs in the same run as their fp32 bounds, with no extra download, and the other files are checked against theirs by hand.
    - The spec decoder writes inf explicitly for overflow cells, since `struct.pack('<f')` raises OverflowError on them.
    - BF16 is gated by identity to the independent F32 file, which already carries its HF bounds.
  - **Read-only.** There is no quantizer for these types.
    Fixtures come from pinned downloads, and the MXFP4 fixtures come from a deterministic Python writer.
  - **Risks:**
    - **ALU cost on the MI50.** The lookup may make MI50 decode ALU-bound: dense MXFP4 at full bandwidth is estimated at about 9 T lookup operations per second against about 6.6 T available.
      Each Vulkan branch measures SWAR, a shared-memory table and `bitfieldExtract` against each other.
    - **Strong reference kernels.** The reference's Q2_K and Q3_K kernels are mature.
    - **The 16-bit prefill bar.** The reference accumulates in f16 or BF16, which llmx will not do by default; a shortfall at pp512 or pp4096 is accepted and recorded (Decided, 9).
    - **Subnormal f16 scales.** Almost every IQ4_XS block has one, and device paths must keep them.
    - **Driver contraction** of the 16-bit builds.
- **Plan**, each branch off main and merged on its own gates:
  - **Every branch's gate:**
    - **Suites:** CTest and the Python suites on the Linux MI50s, and on the Windows Radeon VII with the CPU.
      Windows otherwise serves only the compile check.
    - **Byte identity with main:** logits over the excerpt and 64 greedy tokens.
      - Files: tiny F32, 0.6B F32 (3,012,480,832 bytes), Q8_0, Q4_0, Q5_K_M and Q4_K_M, 8B Q8_0, and 30B-A3B Q4_K_M.
      - Devices: the CPU, one MI50, two MI50s split 1:1, the Radeon VII, and the Radeon VII with `--n-cpu-moe 12`.
    - **SPIR-V:** existing modules byte-identical by hash.
      A module that changes is A/B'd for pp and tg level with main on both cards.
    - **Split:** `llmx-split-check` bit-identical on two MI50s over a file of the new type.
    - **Server:** `tools/server_mix_check.py` on a file of the new type.
      Prompts are distinct and the pool large enough that no request finds a donor or pauses, until exact reuse covers donors (Decided, 10).
      The check asserts that `/v1/health` still reports no `prefix_tokens` and no `pauses` after each run.
      Each request alone gives the same result as when batched and as through the CLI.
    - **Skips:** a Vulkan branch requires zero device skips for its types, through a suite switch beside `--require-baseline` and `--require-tools` that fails a run in which `device_lacks_kernel` skips a named type.
      A CPU-only merge records which device fixtures skipped.
    - **Hosted CI:** a green hosted run on the branch's `gate/<name>` push, with its job times recorded in STATUS.
    - **Device speed:** against the reference's pinned Vulkan build on one pinned MI50 and on the Radeon VII.
      - Cells: pp64, pp512, pp4096 and tg128, plus the merge cells pp247 and tg32.
      - Q4_0 and Q4_K_M of the same model run as controls.
      - The reference's numbers sit beside llmx's, and llmx must be at least level.
      - Step 0 records that the pinned builds on both machines were compiled with integer-dot support and that their devices report it; otherwise both are rebuilt at the same commit with it.
    - **CPU speed:** against the reference's CPU build on the Linux host's CPU.
    - **Measured variants:** every lookup, crossover and activation-width variant is measured in the branch and recorded, whether taken or not.
    - **Docs:** a `docs/src` page for each new source file, the USAGE and VULKAN type lists, and ASSETS rows with sha256.
      Local requantizations are marked as local and not reproducible.
  0. **`refactor/storage-types`** (medium, no behaviour change except `info`, `tokenize` and `detokenize` now opening files that hold unsupported types):
     - **What changes:**
       - The storage table in `core/storage.hpp` over every GGML id.
         `data_size`, the registry and `quant::row_bytes` read it, with F32's block size 1 there.
         The native HF branch rebases onto it.
       - An id outside the table is refused at open, naming the id and the tensor.
       - The CPU refuses at construction a tensor whose type it cannot run.
       - The per-device weight-type check goes into the loader's plan-then-fill step, stream windows included: a streamed layer the device cannot run stays on the host.
         `device_lacks_kernel` matches the new message.
       - Vulkan refuses mixed gate and up expert types at load.
       - Explicit per-backend tables replace every silent default listed under Found.
       - The kernel list becomes one list.
       - The kernel-class function lives in the profile table.
         `tile_reads` reads it over the loaded model's weight types.
         If exact reuse has already added its own kernel-class function, step 0 makes that function read the table.
       - Test helpers take block sizes and scale offsets from the registry.
       - The `q.glsl` constants test.
       - The suite switch that fails on device skips of named types, and the `tools/server_mix_check.py` assertion of no donor hits or pauses, which every later branch's gate uses.
     - **Gates:**
       - the standard byte-identity set, stdout and the split plan identical to main;
       - existing modules byte-identical, except those whose `else` became explicit, which are A/B'd level on both cards;
       - `tile_reads` from the model's types measured level or better on every byte-identity file;
       - a test that every registered type has a row in each backend's table, or a refusal;
       - `info` prints a tiny file holding a Q2_K tensor, and `generate` refuses it, naming the tensor and type.
  1. **`feat/half-weights`** (F16 and BF16, medium):
     - **Measured first, on both cards:**
       - 16-bit prefill against the reference at pp512 and pp4096, where a shortfall is accepted and recorded (Decided, 9);
       - power-of-two rows on the float tile.
     - **Code:**
       - F16 and BF16 supported in the storage table;
       - `bf16_to_f32` in `core/fp16.hpp` and the widen in `f16.glsl`;
       - the CPU fused dots in `row_dot`, the decode branch and `matmul_group`;
       - the float-type predicate threaded through the six decisions;
       - Vulkan 16-bit builds of the plain rows, the float tile on both cards, the routed calls, embed and heads, all from F32's sources.
       - The six unsupported-type checks (Found) move to a still-unsupported id: `backend_vulkan.cpp`'s matmul at 983 and `check_refusals`' F16 constant (1444), which its matmul, routed product and embed use, and `model_validation.cpp:413` and `:421`.
     - **Side effect:** the native HF branch's safetensors half weights run on devices through the same kernels.
     - **Exactness and HF:**
       - Every tensor of unsloth 0.6B BF16 (50968a44), widened by the Python spec decoder rather than by llmx, must equal the independent Qwen3-0.6B-F32.gguf.
       - Then the BF16 logits and 64 greedy tokens must be byte-identical to that F32 file on the CPU and each card.
       - 0.6B UD-Q8_K_XL is held at Q8_0's bounds.
       - A tiny dense model in F32, F16, BF16 and mixed variants, and a tiny MoE with BF16 experts (the CPU decode path), each with the 16-bit type as an untied head and as a tied embedding, give identical logits in CI.
       - Device kernels are checked against the CPU, including F16 subnormals.
       - 30B-A3B UD-Q8_K_XL is checked on each device against the CPU by the criterion.
     - **Speed:**
       - CPU: 0.6B BF16 and 8B BF16.
       - Devices: 0.6B BF16, 4B-2507 F16 and 8B UD-Q8_K_XL on both cards; 8B BF16 on an MI50; 30B-A3B UD-Q8_K_XL on two MI50s and on the Radeon VII with `--n-cpu-moe`.
  2. **`feat/mxfp4-cpu`** (small):
     - **Code:**
       - format, the E8M0 scale with its e <= 1 branch, and the table;
       - the shared C++ lookup, parameterised by the table, which branch 4 reuses for IQ4;
       - CPU dots and heads;
       - the spec decoder with explicit inf;
       - `tools/` writer: deterministic all-matrix MXFP4 with the embedding (Decided, 4).
         The scale is floor(log2 amax) minus E2M1's top exponent 2; values round to nearest even on E2M1 and saturate at +-6.
         The rule is written in the tool and in ASSETS.
       - Vulkan refuses the type at load.
     - **Raw blocks:**
       - every exponent from 0 to 254 against every code in every position (255 blocks, 8,160 values);
       - e = 255 under the chosen rule;
       - the overflow cells as inf;
       - -0;
       - `gguf-validation` refuses a partial row.
     - **Tiny HF models:** dense qwen3 and qwen3moe built from random blocks, run in CI.
       - Widths are multiples of 32, and one has a block count that is not a multiple of four.
       - Blocks with exponent 0 and 1 carry nonzero codes, one block has exponent 0 with all-zero codes, and -0 codes appear at the native rate.
       - An untied MXFP4 head and a tied MXFP4 embedding are included.
     - **Real model:**
       - 0.6B from the writer, sha256 recorded, against the fp32 and file-exact references, with bounds measured and then proposed.
       - 8B from the writer for speed only.
     - **Speed:** CPU, the 0.6B from the writer and the local 14B MXFP4.
  3. **`feat/mxfp4-vulkan`** (medium):
     - **Code:**
       - the float-control query and DenormPreserve;
       - an embed build that makes its value bits with integer operations;
       - a float-tile build;
       - an `LLMX_MXFP4` integer tile on the Q8_0 module's math;
       - the GLSL lookup, written once, which branch 5 reuses for IQ4;
       - four blocks form one word-aligned 68-byte run where the block count divides by four, and a general funnel covers the other cases without reading past a tensor's last word;
       - rows use four lanes per 68-byte group on the 8-bit copy on the MI50, and signed 16-bit lanes on the Radeon VII;
       - routed builds and thresholds.
     - **Gates:**
       - HF on both cards with zero skips;
       - blocks with e <= 1 and nonzero codes bit-exact against the CPU in embed and in every dot, a required outcome;
       - a tiny tensor of odd byte size placed last in its buffer;
       - the local Qwen3-14B MXFP4 and the Coder-30B-A3B MXFP4_MOE on each device against the CPU, by the criterion.
     - **Speed:**
       - 0.6B, 8B and 14B MXFP4 on both cards;
       - Coder-30B-A3B MXFP4_MOE on an MI50, and on the Radeon VII with `--n-cpu-moe`.
  4. **`feat/iq4-cpu`** (IQ4_NL and IQ4_XS, medium, after branch 2):
     - **Code:**
       - format, registry and decoders on branch 2's shared C++ lookup;
       - CPU dots: IQ4_NL on `dot_q4_0`'s shape, IQ4_XS on KQ45_K's step with no min;
       - the CPU head path for the new types;
       - the numpy spec decoder and the tiny-model builder;
       - Vulkan refuses both types at load.
     - **Raw blocks:**
       - every code in both nibbles;
       - `d` positive, negative, zero, subnormal and at the largest finite value;
       - all six scale bits of every sub-block, 0 and 63 included.
     - **Tiny models in CI:** dense and MoE, with IQ4_NL and IQ4_XS on matrices and experts, as an untied head and as a tied embedding, against HF.
     - **HF:**
       - 0.6B IQ4_NL and IQ4_XS from 50968a44, against the fp32 and the file-exact references, with bounds measured on the CPU and then proposed;
       - 8B IQ4_XS and UD-Q4_K_XL become required in the 8B check if IQ4_XS joins the prompt dots.
     - **Speed:** CPU, 0.6B and 8B IQ4_NL and IQ4_XS.
  5. **`feat/iq4-vulkan`** (large, after branch 3):
     - **Code:**
       - embed and float-tile builds on branch 3's GLSL lookup;
       - an `LLMX_IQ4` integer tile: IQ4_NL on Q4_0's staging, and IQ4_XS on Q4_1's with its scale head read once;
       - rows on the 8-bit copy on the MI50, and on integer byte dots or two planes on the Radeon VII;
       - heads on the 16-bit copy;
       - routed builds and profile rows, MoE thresholds included.
     - **Gates:**
       - HF on both cards with the CPU's bounds and zero skips;
       - the pinned Hub Qwen3-14B IQ4_NL on each device against the CPU, by the criterion;
       - subnormal f16 scales on both cards;
       - tile heights and row builds giving the same bytes.
     - **Speed:**
       - 8B IQ4_NL, 8B IQ4_XS, 8B UD-Q4_K_XL and 14B IQ4_NL on both cards;
       - 30B-A3B IQ4_XS on an MI50, and on the Radeon VII with `--n-cpu-moe`.
  6. **`feat/k-low-cpu`** (Q3_K and Q2_K, medium):
     - **Code:**
       - format and dequantizers, with one Q3_K scale unpacker;
       - per-16 activation sums built from the already-quantized integers, so no existing result changes;
       - Q3_K dots on KQ6_K's shape, and Q2_K on the unsigned dot with mins;
       - the CPU head path for both;
       - `is_kquant` includes both, so 8B rows take the prompt dots;
       - Vulkan refuses both types at load.
     - **Raw blocks:**
       - Q3_K: scale-bit patterns, and high-bit patterns keyed to each value's address, which catch an inverted -4;
       - Q2_K: scale and min patterns, with `d` and `dmin` paired over subnormals and negatives.
     - **Tiny models:** a K-quant MoE and a dense model with Q2_K and Q3_K, as an untied head and as a tied embedding.
       Widths are multiples of 256, with a down projection 768 wide.
     - **HF:**
       - 0.6B Q2_K, which holds both types, and Q3_K_S from 50968a44, against the fp32 and file-exact references;
       - 8B Q2_K (required) for the prompt dots and the Q2_K embedding.
     - **Speed:** CPU, 0.6B and 8B Q2_K and Q3_K_M.
  7. **`feat/q3k-vulkan`** (medium):
     - **Code:**
       - embed and float-tile builds;
       - a Q6-shaped integer tile build: the low-bit and high-bit runs are funnelled, and the -4 is folded in by bit flip;
       - rows on the 8-bit copy with the -4 folded per byte on the MI50, and on the 16-bit copy through the half sums on the Radeon VII;
       - routed rows 768 wide.
     - **Gates:** HF on both cards with zero skips; 30B-A3B UD-Q3_K_XL on each device against the CPU.
     - **Speed:** 8B Q3_K_M and 30B-A3B Q3_K_M on both cards, the Radeon VII with experts on the CPU where the file does not fit.
  8. **`feat/q2k-vulkan`** (large):
     - **Code:**
       - an embed build, since the 8B and 30B-A3B Q2_K files have a Q2_K token_embd;
       - the Radeon VII float-tile build, and its plain rows through the half sums;
       - a Q2_K tile whose min term comes from per-column half-block sums, which the activation stagers compute;
       - rows on the MI50 that compute their own activation byte sums with one integer dot;
       - routed builds.
     - **Gates:** HF on both cards with zero skips; 30B-A3B UD-Q2_K_XL on each device against the CPU.
     - **Speed:**
       - 8B Q2_K and 30B-A3B Q2_K on both cards.
         On the Radeon VII, 30B-A3B runs with every expert on the card at the context the fit admits (recorded), and with `--n-cpu-moe` at full context.
       - Once: Qwen3-235B-A22B UD-Q2_K_XL on the fewest MI50s it fits, with `llmx-split-check` bit identity, and the reference's layer split beside it.
- **Tests:**
  - **Every CPU job:**
    - raw-block round trips for each type, from the spec decoders, with the numpy form checked against the pure one;
    - `q8-dots`, `backend-group` and `fused-dot-overflow` rows for each type, MoE prompt entries through `dot_block` included;
    - the tiny 16-bit, MXFP4, IQ4 and K-quant models, dense and MoE, against HF, each new type as an untied head and as a tied embedding;
    - the storage-table completeness test and the `q.glsl` constants test.
  - **The hosted HF job:** the UD-Q8_K_XL, IQ4_XS and Q2_K files against their fp32 bounds, and the IQ4_XS and Q2_K files against their file-exact goldens.
    The job took 15 min 13 s of its 30-minute limit at 73f4f78 (CI.md), and each 0.6B fixture adds about 3 minutes with its f32 pass, so these three add about 9.
    With the Qwen 3.x plan's two 0.8B files the job would near its limit.
    So the branch of either plan that would take the job past 20 minutes, two thirds of its limit, adds a second HF job with its own fixture list and cache key instead.
  - **By hand on the cards:**
    - `backend-vulkan` for each type:
      - embed bit-exact against the CPU;
      - tile, both row variants and routed calls at model shapes;
      - invariance across tile heights and one-column, wide and grouped builds, alone and beside other rows, grouped and routed;
      - subnormal and e <= 1 cases;
      - odd-size tensors placed last.
    - The HF bounds with `LLMX_DEVICE` on both cards and zero skips.
    - The split and server checks, and speed.
- **Not doing:**
  - **IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS.**
    They need about 33 KB of lattice codebooks, and every file that uses them mixes 8 or 9 types.
    IQ3_S is proposed as the first candidate after this plan (open question 11, not yet asked).
  - **Q5_0 and Q5_1.**
    Only one local Qwen3.6-27B Q5_1 file and the qwen4exp architecture use them.
    They are small to add once a planned file needs them (whether they join is the Qwen 3.x plan's open question 12).
  - **Q8_K, TQ1_0, TQ2_0 and type 53.** No file of a planned architecture uses them.
  - **Converting at load,** whether 16-bit to F32 or Q8_0, or any type to another.
    It doubles memory or moves results.
  - **Widening tiny 16-bit tensors at load**, which the user allows.
    Routers go through the same `matmul` as every other matrix, so branch 1's kernels serve them.
    This is revisited for an operation that reads only F32.
  - **A lossy 16-bit prefill** (f16 accumulation or BF16 activations) as a default.
    It would come only as an opt-in the user asks for, honoured identically or refused on every backend.
  - **A quantizer or requantize command for the new types.**
    They stay read-only, like the K-quants.
    The MXFP4 writer is a test tool.
  - **A 16-bit-activation integer tile on the MI50.** It cost 17 to 62 percent where it was tried.
  - **F16 weights through f16-rounded activations.** That moves results away from the F32 path.
  - **Scanning MXFP4 exponents at load.** No sampled block has e = 255.
  - **The gpt-oss and DeepSeek 4.x architectures.**
    They belong to the architecture roadmap.
    This plan only makes their MXFP4 tensors readable.
- **Sequencing:**
  - **Background:** the branches are built now, and layer split phase 3 keeps the cards first, so a branch's device gates run when the cards are free and nothing merges before its gates pass.
  - **Before step 0:**
    - The prefill-kernels block's owed Radeon VII gate and its merge cells (pp247, tg32) pass on main, where its commits already are.
      Step 0 then records its baselines.
    - `cleanup/cpu-kernels` (16) merges, since it rewrites the decode dots branch 1 edits.
    - Loader steps 3 and 4 are on main, because step 0's device check goes into plan-then-fill and its `data_size` change sits beside the reader step 3 rewrites.
    - If any of these is not ready, step 0 waits for it rather than rebasing across it.
  - **Native HF loading branch.**
    It rebases onto step 0's table and branch 1's F16/BF16 entries.
    Branch 1 gives its half weights their device kernels.
  - **Exact resume and exact reuse.**
    The kernel-class function is offered to exact resume's first-admission branch, which owns the exact-reuse rule (Decided, 10).
    The server checks here run without donor hits or pauses until that branch covers donors.
  - **Qwen3.x architecture** ([QWEN35](QWEN35.md) and its block below).
    It runs in parallel, since its first targets need no new type.
    - After step 0's one kernel list, a Qwen3.x Vulkan branch and a type Vulkan branch may be open together; whichever merges second rebases.
    - Branch 1 lands before its UD-Q8_K_XL and BF16 files are used.
    - Branch 3 lands before its MoE step runs the MXFP4_MOE files.
    - Branch 5 lands before its IQ4 files.
    - Branch 7 lands before Qwen3.6-27B Q3_K_M.
  - **Within this plan:**
    - Step 0 comes first, then branch 1.
    - Branch 4 follows branch 2 (the C++ lookup), and branch 5 follows branch 3 (the GLSL lookup).
    - A type's CPU branch may be open while the previous type's Vulkan branch is being gated.
    - Two Vulkan type branches are never open at once, because they edit the same selection tables and shader headers.
  - **DeepSeek 4.x** comes after Qwen 3.x and after branch 3.

## Qwen 3.5, 3.6 and 3.8: the qwen35 and qwen35moe architectures (planned 2026-09-25)

- **Goal:** run the current Qwen generation on both backends, text only.
  The models are Qwen3.5 0.8B to 9B, Qwen3.6-27B and Qwen3.8-27B (`general.architecture = qwen35`), and Qwen3.6-35B-A3B and Qwen3.5-35B-A3B (`qwen35moe`).
  - The work goes in this order.
    The step numbers are the ones in the plan below.
    1. The text path: the chat template renderer and the tokenizer (steps 2 and 3).
    2. The dense models, exact on the CPU against HF (step 4).
    3. Vulkan speed at or above llama.cpp's Vulkan backend, on the MI50 and on the Radeon VII (steps 5 and 6).
    4. The MoE form (step 7).
    5. Serving with the recurrent state: memory fitted at load, pause, checkpoints with reuse, and the layer split (step 8).
    6. The multi-token-prediction (MTP) layer as a proposer of the speculative decoding plan, and the recurrent state's part of its `Model::retract` (step 9).
    7. Cross-gates with the quantization coverage plan's 16-bit, MXFP4 and IQ4 branches (step 10).
  - Every result stays batch-invariant.
  - Every reused cache row or recurrent state is one the CLI would have computed the same way.
- **Design page:** [QWEN35](QWEN35.md) holds the models and their shapes, the forward pass of each layer kind, the GGUF conventions (tensor names and shapes, the folded constants, the V-to-K head mapping), the files on hand and how each is handled, the recurrent state and the row classes.
  This block holds the plan: what llmx lacks today, how the code changes, the branches and their gates, and the decisions.
- **Speculative decoding** is one system for every proposer, set out in the speculative decoding plan, which the user approved on 2026-09-26 and whose block comes to STATUS with its own docs branch.
  It owns verify, acceptance and retract; this plan's step 9 is its step 4, and DFlash drafters are its step 5.
- **Priority:** built now in the background, following AGENTS.md and docs/ARCHITECTURE.md, with a full review of every step.
  Layer split phase 3 keeps priority on the cards, so a step's device gates and speed cells run when the cards are free of it (user, 2026-09-25).
- **Decided (2026-09-25),** numbered as the plan's questions were:
  - **1. HF environment and models:** a second, isolated reference environment with transformers 5.17.0, with 4.55.2 left untouched.
    The gate takes the tiny qwen35 fixtures (Hv = Hk and Hv = 3 Hk), with goldens from HF's token-by-token cached path, and Qwen3.5-0.8B (a pinned Q8_0 plus the Q4_K_M) in the hosted HF job with 512-token windows.
    The 4096-token windows and Qwen3.5-4B run by hand.
  - **2. Layered HF reference** for the 9B, the 27B and the 35B-A3B: HF's own decoder-layer and MoE-block modules run in order, one layer's weights at a time, with peaks of about 9 GB, 11 GB and 8 GB against 18 to 22 GiB free on the Linux host.
    On the 0.8B it must equal the full forward bit for bit.
    It needs bf16 safetensors downloads on the Linux host: about 19 GB for the 9B, 55 GB for the 27B and 70 GB for the 35B-A3B.
    It gives the 27B, a real model with Hv = 3 Hk, and the MoE an HF reference in place of greedy sanity and the CPU-against-device fallback.
  - **3. Exactness classes and the state:** the CPU runs the per-token recurrence for every row.
    Vulkan runs it for generated rows, and the chunked form for every prompt row on an absolute 64-row grid (step 6), kept only if measured faster.
    The state is F32 only, with no flag for a narrower type, since a narrower state breaks batch invariance.
    A one-token prompt is an extent-1 entry and runs the recurrence, as [QWEN35](QWEN35.md), Row classes, defines.
  - **5. Serving before step 8b:** `serve` refuses qwen35 and qwen35moe until the state-aware scheduler lands, so there is no half-exact serving.
  - **6. Failed passes:** a failed pass marks its sequences' states as lost.
    The server ends those requests, as it already does, and the CLI keeps one slot and ends the command, as it does for Qwen3.
  - **8. Take-back resume:** "a resumed request whose own donor survives takes it back whole" goes into `fix/server-exact-resume`, as a step after its step 2.
    Hybrid models need it, and it also removes recompute for dense models.
  - **11. The acceptance rule and the MTP gate** (2026-09-26, through the speculative decoding plan, whose recommendations the user approved): a draft is kept only if it equals the sampler's own pick, and the acceptance-rate gate holds MTP within that plan's margin (5 percent, re-derived in its step 0) of the exact reference build.
    Only the assembled tiny MTP reference is still open (question 11 below).
  - **13. Thinking controls, the chat reference and the tokenizer:**
    - `reasoning_content` goes into step 2, with one split owner shared by the CLI and the server's message parser, because the 3.8 templates render history wrongly without it.
    - The chat reference moves to transformers' renderer, whose `tojson` differs from Jinja2's.
      The Qwen3 chat goldens may change, and any change is recorded and fixed in step 2.
    - The template variables (`enable_thinking`, `preserve_thinking`, `reasoning_effort`) come through `chat_template_kwargs` on the server and one CLI flag, in a small follow-up branch after step 2.
    - The tokenizer's byte-class approximation stays.
      Exact Unicode classes and NFC come later as their own branch, which also fixes Qwen3's Thai gap.
    - The 7 GGUF-only control tokens were to be documented as a difference, but a review on 2026-09-26 found none against transformers 5.17.0, which encodes each as one id, as llmx does (Found, Tokenizer).
      So there is no difference to document, and the tokenizer reference is transformers' tokenizer rather than `tokenizer.json` read alone.
  - **14. Early start:** answered by background development.
    Steps 2 and 3 need no card and gate on hosted CI; the later steps run their card gates when phase 3 frees the cards.
- **Open questions** (asked 2026-09-25, not yet answered):
  - **4. The serve memory fit, step 8a, for every model.**
    `serve` would fit its KV budget at load after the weights, activations and state slots, back it whole at load, and cap `token_limit` at it.
    This changes dense serving too: it takes its whole budget at start instead of growing.
    The alternative keeps growth and counts its peak, which leaves only about half to two thirds of the room reachable as KV.
    Recommendation: fit and back at load.
  - **7. Checkpoint policy, slots and flag.**
    At first, only the checkpoint at the end of the stable prefix, kept on the device and stored in-kernel at a named row.
    The checkpoint slot count is fixed at load, and defaults to what fits in a quarter of the room left after the live slots, at most `--max-seqs`; on the 27B Q8_0 on one MI50 that is 4.
    Message-boundary and every-N checkpoints would come later, on measurement, and the host tier is decided with exact resume's step 3 numbers.
    A suggested flag name is `--state-checkpoints N`.
    Recommendation: approve, and choose the flag name.
  - **9. Speed gate files and cells.**
    MI50: the 0.8B, the 9B Q4_K_M, the 27B Q4_K_M and Q8_0, and the 35B-A3B Q4_K_M and Q8_0.
    Radeon VII: the 0.8B, the 9B Q4_K_M, and the 35B-A3B Q4_K_M with experts on the CPU, since the 27B does not fit its 16 GB.
    Cells: pp64, pp247, pp512, pp4096, tg32 and tg128, plus pp16384/tg512 from the start.
    The files would be copied from the Linux host to the workstation.
    Recommendation: yes.
  - **10. Serving references.**
    Steps 8c and 8d would gate against llama.cpp's server and against mx-llama.cpp's layer and tensor split.
    A gfx906 vLLM build, if it runs `qwen3_5`, would be recorded beside them, not gated.
    Recommendation: approve, and say whether vLLM should also be a gate.
  - **11. The assembled tiny MTP reference.**
    The tiny fixture's MTP logits would be checked against a reference assembled from HF's own decoder layer, since HF has no MTP reference of its own.
    The acceptance rule and the acceptance gate are decided (above).
    Recommendation: approve.
  - **12. Scope.**
    "Qwen 3.8" means Qwen3.8-27B, which comes with qwen35.
    Two lines of the question were settled by later plans on 2026-09-26: Flash-Next (`qwen4exp`) gets its own plan after DeepSeek V4.1, and DFlash drafters are the speculative decoding plan's step 5.
    The MXFP4, IQ4_NL and BF16 files come from the quantization plan's branches and are gated here in step 10; the MXFP4 files are requantized, so they are gated runtime against runtime only.
    The Q5_1 and type-53 files stay refused, and the 122B-A10B comes later, needing F16 and a split.
    Should Q5_0 and Q5_1 join the quantization roadmap?
    Recommendation: approve, and leave Q5_0 and Q5_1 off the roadmap.
- **Found** by four read-only reviews (the model math, the kernels, the recurrent state, the loading and text path) and one review of the plan itself.
  - Sources read:
    - main at d48f2b2, and `refactor/loader` at bdcd2f2;
    - the GGUF headers of every qwen35, qwen35moe and dflash file on the Linux host, plus a few small F32 tensors;
    - the HF checkpoints: layer-0 tensors by range request, and every `config.json`;
    - the `qwen3_5` and `qwen3_5_moe` modeling code in transformers 5.17.0 and on main;
    - llama.cpp 4b1a27fa's converter and `qwen35` graph, and master's `qwen35.cpp` and Vulkan gated-delta-net shader.
  - Nothing ran on a GPU and nothing was changed.
  - The model shapes, the GGUF conventions, what the converter folds, the tiled V-head order, the decay range and the files on the Linux host are in [QWEN35](QWEN35.md).
  - **What llmx refuses or gets wrong today** (line numbers on d48f2b2):
    - The architecture is refused (`arch_qwen.hpp` 99).
    - The rotary width must equal the head dim (130).
    - The rope table uses the head dim as its frequency denominator.
      `footprint` sizes it by the head dim (435), which comes to 256 MiB at 262144 positions.
    - `norm_rope_rows` rotates whole, contiguous heads.
    - Shared experts are refused (148 to 149).
    - `block_count` is read as the layer count (111), which would run the MTP block as a 65th decoder layer.
    - There is no BF16, MXFP4, IQ4_NL or Q5_1.
    - The tokenizer refuses pre `qwen35` (`tokenizer.hpp` 69).
    - Head dim 256 runs only on the per-row attention kernel, because the attention tile and `attention_vec` are fixed at 128.
    - The activation arena has 12 slots (`kSlots`, 278; `slot_widths`, 322 to 330).
      None fits r = 2 q_dim (12288 on the 27B), the raw qkv rows (10240), the conv output, z, alpha, beta and g, o or y.
    - `footprint` holds one cache size for every layer (`cache_per_layer`, 434).
      `kv_used_bytes` multiplies by `n_layer` (795).
      `kv_alloc` is given `attn_layers` (567), which would include the linear layers.
    - `matmul_group` takes at most three projections (`vulkan_backend.cpp` 1432).
      A group of mixed types is split into one dispatch per type.
    - `attention()` writes the 8-bit copy of its output, or the 16-bit twin, for `attn_output` to read (2045 to 2052, 2059, 2097).
      A gate applied between the two would leave that copy ungated.
    - The KV storage grows by doubling and copying, and holds the old and new buffers together (`kv_storage.hpp` 20 to 21, 41 to 73).
      It backs a block when an op writes to it (87), in the middle of a pass.
    - `serve` defaults to `--max-seqs` 16 (`server/api.hpp` 26), and its KV budget is the model context.
      For these files that is 262144 tokens, 16 GiB of KV on the 27B, so the server admits more than one card can back.
    - `tools/long_context_check.py`, the 16k check, runs its device side through `llmx serve`.
    - On `refactor/loader`, `load_model` builds the chat format for every command (`inference/load.hpp` 75).
    - `tools/gen_chat_baseline.py` renders with Jinja2's plain sandbox (63).
      Its built-in `tojson` sorts keys and escapes HTML characters and non-ASCII text, and transformers' override, `json.dumps` with `ensure_ascii=False` and `sort_keys=False`, does none of these.
      Every qwen35 template uses `tojson`.
    - Every CLI command ends on an error (`cli/main.cpp` 1123 to 1125).
      `cmd_chat` only resets or extends its history, and never truncates it.
  - **Tokenizer:**
    - The qwen35 regex adds `\p{M}` to two alternatives: runs of letters, and the run of punctuation and symbols.
    - llmx treats every byte at or above 0x80 as a letter, so marks already join words.
      That is qwen35's behaviour.
    - Example: the Thai title for a lady, 11 code points with its combining marks, is 2 tokens in both HF with qwen35 and llmx.
      With Qwen3's vocabulary HF gives 6 and llmx 2, so this is an existing Qwen3 gap.
    - Deviations that remain, shared with Qwen3: no NFC, runs of U+3000, and one of the 20 multilingual texts (a line of math symbols).
    - The GGUF lists 7 control tokens (248070 to 248076, among them `<|audio_start|>`) that `tokenizer.json` lacks, and llmx encodes each as one id.
      transformers 5.17.0 adds them from `tokenizer_config.json`'s `added_tokens_decoder` and encodes each as one id too (`Qwen2Tokenizer`, 248077 entries, at `Qwen/Qwen3.5-0.8B` 2fc06364).
      Only the `tokenizers` library reading `tokenizer.json` alone splits them, `<|audio_start|>` into 6 ids.
    - No BOS is added.
      EOS is 248046.
      HF's generation config also stops on 248044, the same gap Qwen3 has today.
  - **The chat template renderer gives wrong output, with no error, on every Qwen3.5 and later template:**
    - 16 of 16 fixture cases differ from Jinja2, on the 0.8B, 9B, 3.6 and 3.8 templates.
    - `llmx chat` with the 0.8B template drops every message's content.
      It prefilled 99 tokens for both a 9-byte and a 27-byte message.
    - Causes in `chat.hpp`:
      - `macro` is skipped as an unknown tag.
      - Unknown filters return none.
      - `[::-1]` lexes as `[-1]`.
      - `raise_exception` does nothing.
      - `default` returns its argument for any false value (415), where Jinja returns it only for an undefined one.
      - The tests `undefined`, `true` and `false` are unknown.
      - Ternaries and tuple literals do not parse, and trailing tokens are ignored.
      - Missing: `loop.previtem` and `loop.nextitem`, `startswith` and `endswith`, the `string`, `safe` and `items` filters, and `for a, b in`.
  - **Thinking across turns:**
    - The 3.5 and 3.6 templates drop earlier reasoning.
      Turn N+1 matches turn N's prompt only through `<|im_start|>assistant\n`, which is 2 to 4 tokens before turn N's prompt ends.
    - The 3.8 templates keep reasoning.
      They match 71 of 71 tokens, but only when the reply's reasoning is passed as `reasoning_content`.
      Stored as content, as `cmd_chat` stores it, the history is malformed.
    - The 0.8B and 2B templates default to thinking off.
      The 4B and larger default to thinking on.
    - The GGUF templates are Unsloth's edits of the official templates.
  - **HF reference:**
    - The reference environment has transformers 4.55.2, which has no `qwen3_5`.
    - The versions that matter:
      - 5.2.0 is the first release with `qwen3_5`.
      - Before 5.7.0, a cached multi-token forward restarted the linear layers from a zero state.
      - 5.16.0 fixes the recurrent fallback.
      - 5.17.0 (2026-09-09) has all of these.
        It runs on torch 2.5.1 and needs tokenizers 0.23, huggingface-hub 1.5 or later, and safetensors 0.8 or later.
    - Version 5 loads these checkpoints in bf16 unless a dtype is passed.
    - Its `Qwen2Tokenizer` class rebuilds the qwen2 regex.
    - HF drops `mtp.*` at load (`_keys_to_ignore_on_load_unexpected`), so no HF reference exists for the MTP block.
    - A full forward runs the chunked fallback.
      A cached step runs the recurrent fallback only when `use_precomputed_states and seq_len == 1`.
      Both fallbacks cast q, k, v, beta and the decay to F32 inside, so HF's own code has no FP64 path.
    - FP32 sizes of a whole model: 0.8B about 3.4 GB, 2B about 8 GB, 4B about 18 GB.
      The 9B's weights alone are about 36 GB.
      The Linux host has 62 GiB, of which 18 to 22 GiB was free when read.
    - Running HF's own decoder-layer and MoE-block modules one layer at a time peaks at about embedding + head + one layer: about 11 GB for the 27B, 9 GB for the 9B, 8 GB for the 35B-A3B and 4 GB for the 4B.
    - No small qwen35moe model exists.
  - **Speed references:**
    - No Vulkan qwen35 measurement exists on the Linux host.
    - Upstream, llama.cpp's Vulkan linear attention has had decode pathologies: issue 26795 reports the 35B-A3B at tg128 3.52 against pp512 3159 on an RDNA4 card under RADV.
    - The ROCm build of mx-llama.cpp on one MI50: 27B Q4_K_M pp2048 215 to 235, tg128 25.3 to 25.4.
    - That build's profiles were probably taken on a two-card layer split:
      - 27B Q8_0 pp512: the recurrence is 4.4 percent of device time, at 3.9 us per token step.
      - 35B-A3B pp512: 8.0 percent, at 2.9 us per step.
      - 27B decode: 1.9 percent, at 19.6 us per layer.
    - That build and llama.cpp's Vulkan shader both run prompts through the same per-token loop as decode.
    - **What the per-token loop costs in prefill.**
      At a target of 1 us per token step, the 0.8B's 18 linear layers take 9.2 ms per 512-token pass.
      Qwen3-0.6B's pp512 on one MI50 is about 39 ms, so the recurrence would be roughly 15 to 20 percent of the 0.8B's prefill.
      It would be about 6 percent on the 4B, 4 percent on the 35B-A3B and 2 percent on the 27B.
      At 3 us per step, all of these triple.
  - **Memory, estimated.**
    Every setup here has its KV backed whole at load (step 8a), with a fixed number of state slots.

    | setup | room for KV and state | slots | state | KV tokens |
    |---|---|---|---|---|
    | 27B Q8_0 (26.63 GiB) on one MI50 | about 4.7 GiB | 16 live (serve's default) + 16 checkpoints | 4.67 GiB | about 0.4K: refused by the fit |
    | | | 16 live + 4 checkpoints (the fitted default) | 2.92 GiB | about 29K |
    | | | 8 live + 4 checkpoints | 1.75 GiB | about 48K |
    | | | 8 live, no checkpoints | 1.17 GiB | about 58K |
    | 27B-MTP Q8_0 (27.05 GiB) on one MI50, MTP on | about 4.3 GiB | 8 live + a pool of 4 verify slots + saved inputs | 1.81 GiB | about 38K at 68 KiB a token, given as `--ctx-size`: a drafter takes only the room the no-drafter fit leaves |
    | 27B Q4_K_M (15.41 GiB) on one MI50 | about 15.9 GiB | 16 + 16 | 4.67 GiB | about 184K |
    | | | 32 + 16 | 7.0 GiB | about 146K |
    | 35B-A3B Q4_K_M (19.71 GiB) on one MI50 | about 11.7 GiB | 16 + 16 | 1.96 GiB | about 508K |
    | | | 32 + 32 | 3.93 GiB | about 405K |
    | 9B Q4_K_M (5.24 GiB) on the Radeon VII | about 10.1 GiB | 16 + 16 | 1.57 GiB | about 280K |
    | | | 32 + 32 | 3.14 GiB | about 229K |

    - Under today's growth rule, the last doubling holds the old and new buffers together, so only about half to two thirds of these KV figures is reachable.
    - The 27B does not fit the Radeon VII alone.
    - A 27B decode step reads and writes its 144 MiB of S per sequence, 288 MiB of traffic.
      That adds 14 percent to the weight stream at 8 sequences and 57 percent at 32.
- **Design, the model:**
  - One model path.
    `QwenConfig` stays the one metadata reader, `QwenWeights` the one input, and `Model` the one class, extended as qwen3moe was.
    A second class would copy about 1,000 lines of code that does not depend on the architecture: stages, pools, fork, the arena and placement.
  - The config gains:
    - `n_layer = block_count - nextn_predict_layers`, and `n_nextn`.
      More than one MTP layer is refused.
    - A mixer kind per layer, from `full_attention_interval`, or from the per-layer `attention.recurrent_layers` array when a file has one, whose length and MTP entries are checked as [QWEN35](QWEN35.md) says.
      The resolver refuses a layer whose tensors disagree with it.
    - `rope_dim`: even, at most the head dim, and twice the sum of the sections.
      qwen3 keeps its refusal.
    - The linear-attention widths: K width = state size, V width = inner size / time-step rank, and the K and V head counts, with V a multiple of K.
    - The gated attention flag.
    - The shared-expert width, for qwen35moe only.
      qwen3moe keeps its refusal.
  - The resolver:
    - Maps `post_attention_norm` onto the pre-FFN norm role.
    - Never resolves or reads blocks at or above `n_layer`, unless MTP is asked for.
    - Gains the linear-attention roles (`attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_out`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_norm`) and the shared-expert roles, all of which `footprint` counts.
  - Weights load as the file stores them, with no byte transform, so the loader keeps streaming in file order.
    The kernels read the GGUF conventions of [QWEN35](QWEN35.md): the folded norms, `ssm_a` as stored, V head j against K head j mod Hk, and `attn_q`'s interleaved rows.
    `TensorView` documents that its values follow the GGUF conventions.
  - **Arena and fit:**
    - The arena's slots take the widest use over both mixer kinds, since a layer uses only one.
      The q slot holds r on attention layers and the raw qkv rows on linear layers.
      The k slot holds the conv output.
      The v slot holds z.
      The attention output slot holds o, then y.
      One new slot holds alpha, beta and g (3 Hv).
      `kSlots` becomes 13.
    - `Footprint` gives each layer its own cache: KV tokens for a full-attention layer, and slots x state for a linear layer.
      `tables` is sized by rope_dim / 2: 64 MiB at 262144 positions instead of 256 MiB.
    - `kv_used_bytes` and `kv_alloc` count only the full-attention layers.
  - KV storage covers only the full-attention layers, plus the MTP layer when MTP is on.
- **Design, the forward pass and exactness:** the math of each layer kind, the MTP block, the row classes, the chunk grid and the rules the kernels keep are in [QWEN35](QWEN35.md).
- **Design, the state:**
  - The layout and type (S and the conv rows, F32 only, on every backend) are in [QWEN35](QWEN35.md), The recurrent state.
  - The logical side lives in `model/`: a slot pool, with one slot per sequence in each state storage.
  - The physical side lives in the backend, beside `BlockKVStorage`, without its growth:
    - `state_alloc` allocates, at load, one buffer per layer that holds every slot.
      It is counted in the fit and never grows, so no pass allocates state.
    - `state_copy` copies a slot.
      It restores a checkpoint or a take-back into a live slot.
    - Length 0 means a zero state, so a recycled slot needs no clearing.
      A slot returns to the pool only after the tickets of every pass that touched it have retired, which is the KV storage's retirement rule.
  - A layer's state sits on the device that runs that layer's mixer, so a stage may hold only states.
    `Sequence::length()` and `history()` therefore read a committed length that does not depend on which storage exists.
  - **One slot count per model**, `ModelOptions::state_slots`, which each command sets:
    - 1 for `generate`, `logits`, `perplexity` and `chat`;
    - `--seqs` for `bench`;
    - 2 for `llmx-split-check`;
    - `serve`: `--max-seqs` live slots, plus checkpoint slots, plus the verify-slot pool the fit sizes from the room left after the KV budget when a drafter is on.
  - A pass updates the state in place, so "a failed pass leaves every history as it was" no longer holds by itself.
    A failed pass marks its entries' states as lost (Decided, 6).
    A CLI command ends on the error, as it does for Qwen3.
    The server already ends every request in a failed pass.
  - Until checkpoints exist (step 8c):
    - `truncate` to any length other than 0 or the current length is refused.
    - Forks and prefix reuse are refused for these models.
- **Design, the kernels:**
  - **CPU:**
    - One routine for all rows.
    - Tasks are (sequence, V head), and each head's arithmetic runs on one thread, so the thread count cannot change a result.
    - The state is laid out as [K row][V column], vectorized over 8 columns, with fixed-order sums over K.
    - The conv, the L2 norms, g and beta run in a prologue, and the gated norm runs after.
  - **The Vulkan recurrence**, one shader for the rows of every extent-1 entry, and for every prompt row until step 6:
    - Candidate layout: a 256-lane workgroup covers 32 V columns by 8 lanes, and each lane holds 16 K rows of one column in registers, so a head takes 4 workgroups.
      It is measured against a one-subgroup (64-lane) workgroup per 8 columns.
    - The next token's q, k, v, g and beta are prefetched while the current token computes.
      o_t and m_{t+1} both read S_t, so their reductions share one butterfly phase.
      State loads and stores are 16 bytes wide.
      The L2 norms are fused into the prologue.
    - It reads the state from a source slot and writes it to a destination slot, which are the same slot except in a verify.
    - One checkpoint mechanism: a pass may name one row p and a checkpoint slot.
      The recurrence stores S at p from its registers, and the conv dispatch writes the three raw rows before p, from the arena or the carried rows.
      Batch assembly names the row, and no slice is cut for it.
      The checkpoint becomes valid when the pass commits.
      A verify entry never names a checkpoint row.
    - Targets, which are estimates and not gates: at most 1 us per token step in prefill, and at most 10 us per layer in 27B decode.
  - **The Vulkan chunked form (step 6)**, only for the prompt rows of entries whose extent is above 1: per 64-row chunk, the intra-chunk products and the WY/UT correction in F32, forming only exp(G_i - G_j) for i >= j, then the state passed from chunk to chunk.
    It is about 1.7 times the recurrence's arithmetic, on cards without matrix units, but the sequential chain drops from one step per token to one step per 64 tokens.
  - **The conv** is its own dispatch, with one invocation per (entry, channel).
  - **The gated RMSNorm x silu(z)** is its own op.
    It takes runs, like `silu_mul`, and writes the activation copy that `ssm_out` reads.
    Fusing it into the recurrence is a later, measured step.
  - **Projection groups**, since `matmul_group` holds three projections.
    The candidates, to be measured: {`attn_qkv`, `attn_gate`} + {`ssm_alpha`, `ssm_beta`}, against {`attn_qkv`} + {`attn_gate`, `ssm_alpha`, `ssm_beta`}.
    The second is one dispatch per group in the Q4_K_M files, where those three are all Q4_K.
    On the MoE, the shared-expert gate joins the F32 router's group, and the shared expert's gate and up form one group.
  - **Activation copies:**
    - `sigmoid_mul` writes the 8-bit copy, or the 16-bit twin, of its output by runs, for the matmul that reads it next.
    - `attention` is told when its output is not a matmul's next input, and then skips its copy.
    - Every new op that writes an arena slot (conv, recurrence, gated norm, `sigmoid_mul`, state copies) clears the tags it overlaps, as 1492, 1672 and 1818 do.
  - **Attention at head dim 256:**
    - Decode: `attention_vec` at 256, with 32 lanes per token and one more shuffle stage.
      On the 27B each KV head serves a group of 6 query heads.
    - The prefill tile at 256 needs 34 KiB of shared memory at TK 16 in F32, over the Radeon VII's 32 KiB.
      The candidates, to be measured: 16 lanes per row, with two query heads of one KV head per workgroup; f16 staging, which is exact for the f16 cache; TK 8.
    - TK sets the online softmax's rescale points, so it is a constant of the head dim and the cache type, never of the batch.
      The tile threshold stays by extent.
  - **`norm_rope`** takes a source head stride, and a rotary width separate from the norm width.
    It writes contiguous heads, so `attn_q` loads unchanged and the attention's Q layout stays.
  - **The output gate** is a `sigmoid_mul` op with runs, which reads the gate in place.
    The shared expert's scale is the same op with one value per row, applied to the down projection's input.
    Fusing either into an epilogue is a later, measured step.
  - **Expected on one MI50** (estimates from llmx's measured 8B and 30B-A3B, not gates):
    - 27B Q4_K_M: pp512 about 330 to 350, tg128 about 27 to 28.
    - 27B Q8_0: tg about 20.
    - 35B-A3B Q4_K_M: pp512 about 1200 to 1400, tg128 about 115 to 130.
- **Design, MoE placement:**
  - Under `--n-cpu-moe`, a host-placed layer's whole FFN goes to the CPU with its experts: the router, the shared expert and its gate.
    That is qwen3moe's rule.
    Placing the shared expert beside attention is a later, measured step.
  - Under `--moe-stream-from`, the shared expert and its gate are copied to the attention device at load, beside the norm and router.
- **Design, serving (step 8):**
  - **The fit (8a, every model; question 4):**
    - `serve` fits its KV budget at load after the weights, activations and state slots, and backs it whole at load, so no pass grows the KV.
    - `token_limit` is the smaller of the context, `--ctx-size` and the fitted budget.
    - A drafter never changes the fitted budget; it takes only the room left after it (the speculative decoding plan).
    - Load is refused when the live slots leave no room for KV.
      The startup log gives the live slots, checkpoint slots and KV tokens.
  - **Admission:**
    - An admitted request takes one live slot, and there are `--max-seqs` of them, so admission never waits on a slot.
  - **Checkpoints (question 7):**
    - A checkpoint is a checkpoint slot's copy of every state storage at a position p, plus the row classes of [0, p).
    - p is a multiple of `kv_block_tokens()`, which 64 divides, so the KV fork stays whole-block and p lies on the chunk grid.
    - The checkpoint slot count is fixed at load.
      It defaults to the number that fits in a quarter of the room left after the live slots, at most `--max-seqs`.
    - A checkpoint is skipped, never forced, when no checkpoint slot is free.
  - **Where checkpoints go.** Batch assembly is the one owner.
    - Always: the largest block boundary at or below E - g, where E is the prompt's end and g is the length of the generation-prompt suffix.
    - On the 3.5 and 3.6 templates the next turn diverges right after `<|im_start|>assistant\n`, so this one checkpoint serves both retries and follow-up turns.
    - On the 3.8 templates the next turn keeps the whole reply when its reasoning is passed as `reasoning_content`, and the same checkpoint still serves, since a follow-up turn recomputes the reply's generated rows as prompt rows.
    - Message-boundary and every-N checkpoints are options, off at first.
  - **Choosing a donor:**
    - `best_donor` considers only checkpoints whose tokens, and whose row classes over [0, p), equal what the CLI would compute for the new prompt.
    - A finished request's final state follows generated rows, which a follow-up turn recomputes as prompt rows.
      So at finish a donor is trimmed to its last checkpoint, and a hybrid donor with no checkpoint is released.
  - **Pause:**
    - If a resumed request's own donor survives, the request takes it back whole, KV and live slot included, and recomputes nothing.
      This take-back lands in `fix/server-exact-resume`, as a step after its step 2 (Decided, 8).
    - Otherwise the exact-resume replay runs from 0, or from a checkpoint whose classes match.
    - A 27B state is 150 MiB, which takes 13 to 31 ms each way to host memory at 5 to 12 GB/s.
      Recomputing is estimated at about 4 s per 1000 tokens.
      That weighs toward exact resume's host tier for these models.
  - **`make_room`** stays the one owner, now over KV blocks and checkpoint slots together.
    It frees in this order: the oldest finished donors' checkpoints, then paused donors, then active requests' optional checkpoints.
  - **Layer split:**
    - Each stage holds its own layers' states, and phase 2's pipelined chunks, cut on the chunk grid, advance them in order.
      So the CLI's split needs nothing more.
    - Phase 4's per-storage progress includes the state storages.
      A checkpoint is valid only once every stage has taken it.
- **Design, speculative decoding (step 9):**
  - **Verify, acceptance and retract** are the speculative decoding plan's (`infer::accept`, `Model::retract`) for every proposer; this plan supplies the qwen35 state's part of `retract`.
  - **State:**
    - The verify reads the live slot and writes a verify slot from the pool, so the state from before the verify is kept; a request drafts only while it holds one.
    - Each linear layer's verify inputs are saved to a per-sequence buffer counted in the fit: the raw conv rows, plus the normed q and k, v, g and beta after the conv, for k + 1 rows.
      That is about 16 MB per sequence on the 27B at k = 3.
      The arena cannot hold them, because every layer reuses it.
    - After a rejection, the recurrence reruns over the accepted rows from the kept state, using the saved inputs.
      The per-token arithmetic is the same, so the bits are the same.
      When every draft is accepted, the two slots swap roles.
      A sequence with a restore pending is not parked, paused, forked, checkpointed or cancelled into a donor until the restore has run.
    - `Model::retract` drops rejected rows through the model's one storage list, calls this plan's state restore, and drops any checkpoint past the new length.
      The public truncate stays refused.
    - Snapshots after every row (144 MiB each on the 27B) are not planned (the speculative decoding plan's Not doing).
  - **The MTP proposer:**
    - It loads only when `--drafter embedded` or a sidecar path asks for it.
    - Its KV is one more attention layer in the target's KV storage, 4 KiB per token on the 27B, with the target's length, blocks, forks and truncation.
      Every pass computes its rows (`eh_proj`, the norms, K and V; no attention or FFN) for every row it feeds, prompt rows included.
    - The MTP row that reads h_{i-1} and t_i sits at index i, at rotary position i - 1, and its attention reads indices 1 to i.
      So a prefix of p rows carries exactly the MTP rows its tokens determine, and a fork takes them with the target's blocks.
      Each MTP block keeps h of its last position, and the sequence keeps its last h.
    - On a layer split, the MTP block runs on the output device, which holds h, `output` and a copy of `token_embd` counted by the fit.
      The draft chain (argmax, embedding, next step) is one submission there.
    - On the 35B-A3B, the MTP block's FFN is step 7's MoE FFN.
- **Design, text:**
  - **Tokenizer:**
    - Accept pre `qwen35` beside `qwen2`.
    - `pretokenize` needs no change.
  - **Chat template:**
    - The renderer gains:
      - `macro`/`endmacro`, with default arguments, their own scope, and namespaces shared by reference;
      - slicing;
      - `x if c else y`, and tuple literals;
      - the tests `undefined`, `true` and `false`;
      - the filters `trim`, `string`, `safe` and `items`;
      - `startswith` and `endswith`;
      - `loop.previtem` and `loop.nextitem`;
      - `for a, b in`;
      - `raise_exception` as a real error;
      - `default` returning its argument only for an undefined value, as Jinja does, so that the 3.8 templates' `reasoning_effort|default('xhigh')` raises on a null, as transformers does;
      - `tojson` as transformers renders it, Python's `json.dumps(x, ensure_ascii=False, sort_keys=False)` with its `indent` and `separators` arguments: keys in their given order, no HTML escaping, non-ASCII text kept as it is, Python's escapes (`\b` and `\f` among them), and numbers spelled as Python spells the int or float the JSON text parsed to.
    - Any unknown tag, filter, test or method, and any trailing tokens, make the template refused.
      The refusal is recorded in `ChatFormat` and raised only by `chat` and `serve`.
      `generate`, `logits` and the other commands are unaffected.
  - **Messages:** `chat::Message` gains `reasoning_content`.
    One owner in `inference/chat.hpp`, beside the renderer and `chat::Message` and below both `server/api.hpp` and the CLI, splits an assistant turn as the 3.5 and 3.6 templates split it: `reasoning_content` is the text before the first `</think>`, after the last `<think>` in it, and the content is the text after the last `</think>`, with the templates' newline trims.
    A reply with no `</think>` is all content.
    Both the CLI's turn recording and the server's message parser use it, so a server render equals the CLI render, and a split turn renders on those templates as its unsplit text does.
  - **Thinking variables** (Decided, 13): `enable_thinking`, `preserve_thinking` and `reasoning_effort` reach the template through `chat_template_kwargs` on the server and one CLI flag, named for best fit and added to USAGE and `print_usage`, in a small branch after step 2.
- **Design, the HF reference:**
  - A second, isolated environment with transformers 5.17.0 on torch 2.5.1+cpu, so the 4.55.2 goldens stay reproducible (Decided, 1).
    Its exact package versions are recorded in the repo's reference tooling, as the existing environment's are.
  - The generator:
    - passes float32 explicitly and uses eager attention;
    - runs offline with no `kernels` package, so the torch fallbacks run;
    - loads `Qwen3_5ForCausalLM`;
    - asserts that the only unexpected keys are `mtp.*` and `model.visual.*`, and that no key is missing;
    - counts, through a hook on HF's recurrent function, that every cached step of a tiny golden ran the token-by-token recurrence, and records the full forward's distance from it, which must be non-zero;
    - is held to all of the above by `reference-generator`'s doubles: the version, float32, eager attention, no `kernels` package, and the unexpected- and missing-key assertions.
  - The tiny fixtures' goldens come from HF's token-by-token cached forward, whose recurrent form matches llmx's per-token arithmetic.
    HF's full forward, which uses the chunked form, is recorded beside them with its distance.
    HF has no FP64 path for these layers.
  - The real models' goldens come from HF's full forward.
    The 0.8B's weights and tokenizer come from `Qwen/Qwen3.5-0.8B` at 2fc06364, and both 0.8B files from `unsloth/Qwen3.5-0.8B-GGUF` at 6ab46149 (Q8_0 811,843,840 bytes, Q4_K_M 532,517,120 bytes), which enter `tests/data/fixtures.json` with their SHA-256.
    The Linux host's 0.8B Q4_K_M is another build (527,502,816 bytes) with no known Hub source, so it is not a gate file.
  - The 4B's full forward takes about 18 GB in FP32 against the 18 to 22 GiB found free, before its 4096-token windows.
    If it does not fit, the fallback is a layered run for the 4B as for the 9B, and using it needs the user's OK first, as a gate revision.
  - The layered reference (Decided, 2) runs HF's own decoder-layer and MoE-block modules in order, with the model's own rotary embedding and masks, loading one layer's weights at a time.
    It covers the 9B, 27B and 35B-A3B.
    On the 0.8B it must equal the full forward bit for bit.
    Its goldens follow the 8B's: committed under `tests/data`, each recording the HF repo and revision it ran on and the SHA-256 of the GGUF it is for, and read by a consumer like `tests/baseline_8b.py` that checks the model's and the goldens' hashes and writes a report.
  - Goldens are stored as JSON, as today, so the hosted HF job needs no new environment.
  - The tokenizer reference is transformers 5.17.0's tokenizer at `Qwen/Qwen3.5-0.8B` 2fc06364, which adds the entries of `tokenizer_config.json`'s `added_tokens_decoder` to `tokenizer.json`; the `tokenizers` library on `tokenizer.json` alone would split the 7 control tokens (Found, Tokenizer).
  - The chat reference renders with transformers 5.17.0's own template renderer, in the second environment (its `tojson`, `raise_exception` and loop controls), not with a plain Jinja2 sandbox.
    Each chat golden records the transformers version that rendered it.
  - The fixture writers apply the converter's transforms themselves (1 + w, -exp, the tiled V order).
    HF then runs the raw weights, and llmx reads the GGUF conventions.
    A hosted check holds the writer's tiled permutation to real values: it maps HF's `dt_bias` of one 4B layer onto that layer's `ssm_dt.bias` in the 4B GGUF bit for bit, from the 32 values of each committed under `tests/data`, so a misreading that the writer and the kernels share cannot pass the tiny Hv = 3 Hk fixture.
- **Plan.** Each branch comes off main and merges on its own gates, which always include a green hosted run on its `gate/<name>` push, with the job times recorded in STATUS.
  Sizes are relative to the Qwen3 MoE work: `feat/moe` took about 1,200 lines of source and tests for the model, routing, the expert kernels and the tiny HF gate (beside its stored baseline), then about a day of speed work.
  1. **`docs/qwen35-design`** (this branch): this block, the quantization coverage block above, the design page [QWEN35](QWEN35.md) for the math, the GGUF conventions, the file inventory and the row classes, ROADMAP #1, #2 and #9b brought in line, and the design page listed in README.
     - Gate: review of the design.
     - Size: small.
  2. **`feat/chat-template-jinja`:** the renderer features above, the refusal recorded in `ChatFormat`, `reasoning_content` with its one split owner, and the chat reference moved to transformers' renderer.
     - Gates:
       - byte-equal to transformers' renderer on all 9 distinct qwen35 templates (pinned by sha256) and their cases, which cover:
         - tools, so `tojson` meets non-ASCII text and nested key order;
         - `tool_calls` arguments as a dict and as a string, and tool responses;
         - content-part lists;
         - conversations with and without a system message;
         - both values of `add_generation_prompt`, and of `enable_thinking` passed to the renderer;
         - `reasoning_content` in earlier assistant turns on the 3.8 templates, with thinking on and off;
         - cases where transformers raises, which llmx must refuse rather than render;
       - a `chat-template` CTest case for each new renderer construct;
       - the Qwen3 fixtures regenerated with that renderer, with any difference from the old goldens recorded and fixed in this branch;
       - no template in any GGUF on either machine (deduplicated by sha256) becomes refused;
       - unsupported syntax refused by `chat` and `serve`, and `generate` still runs on a file whose template is refused;
       - server renders with `reasoning_content` equal to CLI renders;
       - split-owner cases: a reply with no `</think>`, with a leading `<think>`, and with two `</think>`, each rendering on the 3.5 and 3.6 templates as its unsplit text does;
       - a two-turn `llmx chat` test on a 3.8 template: the history the CLI records through the split owner renders byte-equal to transformers' render of the same {content, reasoning_content} messages, and turn 2's prompt begins with turn 1's;
       - hosted CI.
     - Size: about 0.6.
     - **Done** on `feat/chat-template-jinja` (its block at the top of this file), with three differences from the gates above that need the user's OK: the split owner splits a turn only under a template that reads `reasoning_content` and does not split `</think>` itself, the Qwen 3.8 ones among the pinned templates, since splitting first under the 3.5 and 3.6 templates can lose a reply's leading whitespace; the machines and the official repositories hold 8 distinct Qwen 3.5 to 3.8 templates, not 9; and the cases passing `enable_thinking` wait for the follow-up branch, which passes the thinking variables.
     - **The follow-up branch** (Decided, 13): the thinking variables through `chat_template_kwargs` and one CLI flag.
       Gates, as this step's: renders with each variable set byte-equal to transformers' renderer, the server's render equal to the CLI's, and hosted CI.
       Size: small.
  3. **`feat/tokenizer-qwen35`**, which also lets `tests/data/fixtures.json` and `tests/baseline.py` take a second model family (Tests).
     - Gates:
       - A golden in `tests/data` from transformers' tokenizer, recording the tokenizer's repo and revision: the Qwen3 golden's 20 texts, the 20 multilingual texts of the review, Thai, Devanagari, CJK punctuation, combining marks after letters, digits, punctuation and spaces, and every entry of `added_tokens_decoder` (248044 to 248076, the 7 control tokens included), alone and inside text.
       - llmx's ids equal the golden's, except the known deviations (the math-symbol line, runs of U+3000, and any text NFC would change), each listed in the golden as an expected difference with both id lists.
       - Every other pre name still refused.
       - Hosted CI, with the 0.8B Q8_0 pinned as a tokenizer-only fixture until step 4 adds its bounds.
     - Size: about 0.1.
  4. **`feat/qwen35-cpu`**, the dense models on the CPU.
     It brings:
     - the config, the resolver and the mixer kind;
     - the arena slots, the per-layer cache in `Footprint`, the rope table by rope_dim, and the KV counters over attention layers only;
     - the CPU conv, recurrence, gated norm, partial rope and output gate;
     - state storage allocated at load, the slot count per command, the retirement rule, and the failed-pass and truncate rules;
     - refusal, with a clear message, by the device backends and by `serve` (Decided, 5);
     - the HF environment and the fixtures.
     - Gates:
       - Tiny qwen35 fixtures within today's bounds (2e-5 per logit, 1e-5 NLL) against HF's token-by-token goldens.
         The fixtures cover Hv = Hk, Hv = 3 Hk, tied and untied heads, and a file with an MTP block whose logits must not change.
         The bounds must hold across ubatches, batch widths, threads, decode after prefill, and scoring.
       - Qwen3.5-0.8B (the pinned Q8_0 and Q4_K_M downloads) and Qwen3.5-4B against HF: tokenizer ids, top-1 and top-5, PPL windows, and the chat render and its ids.
         The hosted job keeps windows at 512 tokens, and 4096-token windows run by hand.
         Bounds are set from a first measurement, as `tests/baseline.py` does, and both 0.8B files are hosted fixtures of the qwen35 family from then on.
       - The hosted check of the fixture writer's tiled permutation against the 4B's `dt_bias`.
       - The 9B and 27B Q4_K_M against the layered HF reference (Decided, 2).
       - Bitwise slice invariance.
       - `bench --seqs` and a two-CPU `llmx-split-check`, including a stage that holds only states.
       - Logits and greedy text on the Qwen3 files byte-identical to main.
     - Size: about 1.3.
  5. **`feat/qwen35-vulkan`:** the device ops (conv, the per-token recurrence with source, destination and checkpoint-row push constants, the gated norm, `sigmoid_mul`, and the copy and tag rules), the projection groups, device state storage, attention at head dim 256, strided partial rope, the CLI's layer split with states, and a CLI mode for `tools/long_context_check.py` (two fresh `generate` runs, plus `logits --last` on the baseline).
     - Gates:
       - The tiny fixtures, the 0.8B and the 4B within bounds on both cards.
       - Bitwise slice invariance on the device.
       - A backend-vulkan test of attention, then `sigmoid_mul`, then a matmul, on the tile and on the row path, against the CPU.
       - The 16k check in its CLI mode on the 9B and the 27B, against CPU baselines on the Q4_K_M files.
       - `llmx-split-check` identical to one card on 2 and 3 MI50s, including a stage that holds only linear layers.
         The Radeon VII with the CPU is held to the HF bounds and to the CPU-against-device margins, not to bitwise identity.
         The check's fork and replay phase skips hybrids until step 8c.
       - The Qwen3 files byte-identical on both cards.
       - Speed at or above llama.cpp Vulkan on every cell: pp64, pp247, pp512, pp4096, tg32, tg128 and pp16384/tg512.
         The files are the 0.8B Q4_K_M, the 9B Q4_K_M, and the 27B Q4_K_M and Q8_0 on the MI50, and the 0.8B and 9B Q4_K_M on the Radeon VII (question 9).
     - Size: about 1.6 to 2.1.
  6. **`feat/qwen35-chunked`:** the chunked form on Vulkan for the prompt rows of every entry whose extent is above 1, and the model's prompt cut on the absolute 64-row grid.
     - Gates:
       - The tiny fixtures, the 0.8B and the 4B within HF bounds on both cards.
       - Bitwise slice invariance with cuts at 1, 63, 64, 65 and 511 rows, where the cut rounds to the grid, and with prompts that start off the grid.
       - Server budget slices and the split's pipelined chunks land on the grid.
       - The 16k check.
       - The 0.8B, 4B, 9B and 35B-A3B prefill cells faster than step 5 on both cards, and no tg cell slower.
         The form is kept only if it passes, and a loss is recorded (Decided, 3).
       - Speed still at or above llama.cpp Vulkan.
     - Size: about 0.6 to 0.8.
  7. **`feat/qwen35moe`:** the shared expert and its gate on both backends, 256 experts (the routing kernel already holds up to 1024), and the `--n-cpu-moe` and `--moe-stream-from` placements.
     - Gates:
       - A tiny qwen35moe fixture (8 experts, top 3, expert FFN 11, shared FFN 13, the shared gate, the router scaled as `moe.py` does, the minimum routing gap recorded) on the CPU and both cards, in both placements.
       - The 35B-A3B against the layered HF reference (Decided, 2).
       - The 35B-A3B Q4_K_M on the MI50 against the CPU, with the existing CPU-against-device margin check and the 16k check.
         The Q8_0 files do not fit the host's free memory for a CPU baseline.
       - Qwen3, qwen3moe and qwen35 dense byte-identical.
       - Speed at or above llama.cpp Vulkan: the 35B-A3B Q4_K_M and Q8_0 on the MI50, and the Q4_K_M with experts on the CPU on the Radeon VII (question 9).
     - Size: about 0.4 to 0.5.
  8. **Serving,** as four branches:
     - a. **`fix/serve-kv-fit`,** for every model: the fit, backing at load, and the `token_limit` cap (question 4).
       - Gates:
         - The server suites unchanged on dense and MoE models.
         - A configuration that asks for more than fits is refused at load.
         - `tools/server_mix_check.py` and the serving load tool on the 8B and the 30B-A3B, with no throughput loss against main.
       - Size: about 0.2.
     - b. **`feat/qwen35-serve`:** live slots and admission, and pause and resume by take-back or exact replay.
       No prefix reuse.
       It lifts `serve`'s refusal of these models.
       - Gates:
         - `server-resume` cases on a synthetic hybrid model: take-back, a recompute after eviction, cancellation back to zero slots, and repeated requests on recycled slots with two contexts in flight.
         - `tools/server_mix_check.py` with uncapped pausing on the 9B and the 35B-A3B: each request equals itself alone and the CLI.
         - The randomized `make_room` test, with the slot ledger.
       - Size: about 0.6.
     - c. **`feat/qwen35-checkpoints`:** checkpoint slots, the in-kernel store, class-aware `best_donor`, donor trimming, and `make_room` over blocks and slots (question 7).
       The split check's fork and replay phase comes back for hybrids.
       - Gates:
         - A fork from a checkpoint continues bit-identically to a fresh run.
         - Server equals CLI, with reuse on, under mixed and skewed load.
         - vLLM-style metrics (TTFT, ITL and throughput percentiles, 1 to 64 users, a rate sweep) heavily ahead of llama.cpp's server and of mx-llama.cpp on the same cards (question 10).
           Their logs are checked for whether their checkpoints were actually restored.
           A gfx906 vLLM build, if it runs `qwen3_5`, is recorded beside them, not gated.
       - Size: about 0.7.
     - d. **The server on a layer split**, with states in phase 4's per-storage progress.
       - Gate: the split's server gate with a hybrid model on 2 or more cards, against mx-llama.cpp's layer and tensor split (question 10).
       - Size: about 0.3.
  9. **`feat/qwen35-mtp`** (step 4 of the speculative decoding plan): the verify-slot pool and saved inputs behind `Model::retract` for qwen35, the MTP proposer embedded and as a sidecar, its rows indexed by the token they read, the on-device draft chain, and its placement on a split.
     - Gates (Decided, 11):
       - Tokens and logits identical with MTP on and off, greedy and seeded sampling, on the CPU and both cards, through the CLI and the server.
       - The acceptance rate on a fixed prompt set, at draft lengths 1 and 3, within the speculative decoding plan's acceptance margin of the exact reference build.
         On-and-off identity alone passes even with a wrong block, and only a draft length above 1 reads the block's own row after `nextn.shared_head_norm`.
       - The tiny fixture's MTP logits against a reference built from HF's own decoder layer, if question 11 approves it.
       - Decode speedup and memory cost on the 27B at Q8_0 and Q4_K_M, beside llama.cpp with its MTP.
       - The 35B-A3B MTP waits for the 16-bit branch.
       - Embedded and sidecar MTP give the same drafts.
       - With MTP loaded and k = 0, logits are identical to no drafter, decode is within 3 percent, and pp512 and pp16384 are within 2 percent.
       - The synthetic proposers reject at j = 0, 1, 2 and k.
       - On the Radeon VII, the tiny fixtures run written both ways (a gate revision approved with the speculative decoding plan).
     - Size: about 1.0.
  10. **Cross-gates with the quantization plan,** run once both sides have merged.
      No branch is made unless a fix is found.
      - MXFP4: the 27B and 35B-A3B MXFP4 files.
        CPU against device with the margin check, split identity, PPL beside llama.cpp on the same file, and speed at or above llama.cpp Vulkan.
        There is no HF bound, since both files are requantized.
      - IQ4_NL: the 35B-A3B IQ4_NL, with the same checks.
      - BF16: the UD files, the 27B BF16 on a two-MI50 split against the layered HF reference, and the 35B-A3B MTP files.
      - Size: about 0.1 each.
  - **Totals:**
    - Steps 4 to 7 are about 3.9 to 4.7 times the MoE work.
      That is above the earlier estimate of 2 to 3 times, because it now includes the chunked form, the arena and fit work, and the split and 16k-check paths.
    - Steps 2 and 3 add about 0.7.
      They were not in that estimate: the template breakage was found by this review.
    - Serving adds about 1.8, MTP about 1.0 (verify, acceptance and retract are in the speculative decoding plan's steps 1 to 3), and the cross-gates about 0.3.
    - The whole plan is about 7.5 to 8.5 times the MoE work.
- **Tests.** Everything that needs no GPU runs in the hosted workflow.
  Device checks run by hand on both cards and are named in the step's STATUS entry.
  - **Fixtures and references:**
    - `tests/qwen35.py` and a qwen35moe case, with goldens from `tools/gen_baseline.py` in the new environment.
    - `tests/data/fixtures.json` and `tests/baseline.py` take a second model family in step 3.
      Each pinned entry names its family, each family has its own goldens and its own vocabulary size and context (151936 and 40960 for Qwen3, 248320 and 262144 for qwen35), and an entry may be tokenizer-only until its bounds exist.
    - The 0.8B HF check in the hosted HF job, with 512-token windows.
      The 4096-token windows and the 4B check run by hand.
    - Hosted time: the two 0.8B files add an estimated 6 to 8 minutes to the HF job with their f32 pass, under the limit rule in the quantization coverage block's Tests.
    - The layered goldens, read by hand by their consumer.
    - Chat template fixtures against transformers' renderer, and tokenizer ids against transformers' tokenizer.
    - The hosted check of the fixture writer's tiled permutation.
    - A scan that renders or refuses every GGUF template on both machines.
  - **Guards on the recurrence:**
    - Slice invariance: one pass against splits at 1, 3, 63, 64, 65 and 511 tokens, beside other sequences, with checkpoint stores on and off, compared with `memcmp` on logits and state.
      It also covers a conv with fewer than 3 earlier tokens, and Hv / Hk of 1, 2 and 3.
    - Extent-1 replay: k rows in one extent-1 entry equal k decode steps.
    - Decay flush: a fixture row with a decay factor below 2^-126 gives the same state on the CPU and the device.
  - **`backend-vulkan`:**
    - The conv, the recurrence, the chunked form, the gated norm, `sigmoid_mul`, strided partial rope, and dim-256 attention on the vec, tile and per-row kernels, each against the CPU and a double-precision reference.
    - Attention, then `sigmoid_mul`, then a matmul, on the tile and row paths.
    - The existing tail-after-reuse and decode-beside-longer checks, at 256.
  - **Model:**
    - The slot pool, and slot retirement with two contexts in flight.
    - An injected failed pass marks states lost, and the command ends.
    - `truncate` is refused.
    - A file with an MTP block runs `block_count - nextn` layers.
      `nextn_predict_layers` above 1 is refused.
    - A `recurrent_layers` array of the wrong length, or with an MTP entry true, is refused.
    - Bad head counts are refused.
    - qwen3 with a rotary width different from its head dim is still refused.
    - A two-CPU split equals one CPU, with a stage that holds only states.
    - MXFP4, IQ4_NL, BF16, Q5_1 and type-53 files are refused, each naming its type, until their branches land.
  - **Server:**
    - The fit refuses at load, and `token_limit` follows the fit.
    - The hybrid `server-resume` cases.
    - `make_room` with slots.
    - Checkpoint forks.
    - MTP identity under the speculative decoding plan's synthetic proposers, and `Model::retract`'s state restore.
- **Risks:**
  - A wrong V-to-K head mapping gives plausible text.
    Only Hv different from Hk catches it: the tiny Hv = 3 Hk fixture, the 4B, and the 9B, 27B and 35B-A3B through the layered reference, with the hosted permutation check covering a misreading that the fixture's writer and the kernels could share.
    A file converted before the tiling change cannot be told apart from its metadata.
  - Folding done twice or missed: 1 added again to the norms, 1 added to `ssm_norm`, or exp applied to `ssm_a`, which is already negative.
  - The recurrence's step latency: on the 0.8B it is 15 to 20 percent of prefill at 1 us per step.
    The chunked form is the answer, and it is a hard kernel.
  - The dim-256 tile's shared memory and registers.
  - Silent inexactness from contraction without `precise`, from driver `exp` or denormal flushing, or from a pipeline specialized on the token count.
  - State corruption across requests on recycled slots, which llama.cpp's fork has logged.
    The retirement rule and the recycled-slot tests cover it.
  - Checkpoint slots compete with KV: on the 27B Q8_0 on one MI50, serve's defaults with a checkpoint per sequence leave no KV.
  - The requantized MXFP4 and review files are lossy twice over, so they compare runtimes only.
  - The 27B configs carry `output_gate_type: "swish"`, `attn_output_gate` and `mtp_use_dedicated_embeddings`.
    HF, llama.cpp and vLLM all ignore these and apply a sigmoid gate.
    llmx follows HF, and the HF gate checks it.
- **Not doing, with reasons** (question 12 covers the scope lines):
  - **Qwen3.8-Flash-Next (`qwen4exp`):** a different architecture, with its own plan after DeepSeek V4.1.
    Qwen3.8-27B is plain qwen35 and comes with this plan.
  - **DFlash drafters (`dflash`):** a different architecture.
    They draft blocks of 8 or 16 tokens from hidden rows exported by 5 to 8 chosen target layers, with non-causal or sliding-window attention and conv projections.
    They need an export of hidden rows from chosen layers and a block drafter.
    They are step 5 of the speculative decoding plan, a proposer over the same verify and loader path.
  - **Vision and video input:** text only.
    For text positions the M-RoPE reduces exactly to plain rope.
  - **A chunked form keyed on the batch, on an extent threshold above 1, or for extent-1 entries:** a row's class is its entry's extent, 1 or above 1, nothing else.
  - **A narrower state, or compressed checkpoints** (such as llama.cpp's INT8 and INT16 checkpoint codec): they break batch invariance or exact reuse.
  - **Forking or truncating at arbitrary positions:** a recurrent state exists only at the end of what it has read.
  - **BF16, F16, MXFP4 and IQ4_NL files:** they load once the quantization plan's branches merge, and step 10 gates them.
  - **Q5_1 and type-53 files:** Q5_0 and Q5_1 are not on the quantization roadmap, and type 53 is not a ggml type.
    Both are refused, each naming its type.
  - **Qwen3.5-122B-A10B as a gate file:** it needs F16 and a multi-card split, and adds no code of its own.
  - **Fused `ffn_gate_up_exps` from newer converters:** no file here has it, so it is refused by name until a gate file does.
  - **Exact Unicode classes and NFC:** a separate tokenizer branch (Decided, 13).
    The approximation already matches qwen35 on marks.
  - **A safetensors qwen35 reader:** it would need byte transforms that the streamed loader cannot express.
- **Sequencing against the work in flight:**
  - Implementation runs now in the background (user, 2026-09-25).
    Layer split phase 3 keeps priority on the cards, and each step merges only on its own gates.
  - `refactor/loader` (at bdcd2f2) merges before step 4, which extends its `QwenConfig`, `QwenWeights`, resolver and single load owner.
    Step 2's refusal lives in the `ChatFormat` that its `load_model` builds.
  - Steps 2 and 3 touch neither the loader nor the model, need no card, and gate on hosted CI.
  - Step 5:
    - The dim-256 tile builds on `feat/attn-tile-split` after it merges, and on the prefill kernels just merged.
    - Its speed gate needs the cards that the phase 3 server gate uses.
  - Step 6 follows step 5.
    Step 7 needs step 5, and does not need step 6.
  - Step 8:
    - 8a does not depend on qwen35, and follows phase 3's server work.
    - 8b needs 8a, and `fix/server-exact-resume` for the replay and its take-back step.
    - 8c needs exact resume's class-aware first-admission follow-up for `best_donor`, and phase 3.
    - 8d needs phase 4.
  - Step 9:
    - It needs step 5 for the CLI and 8b for the server, and the speculative decoding plan's steps 1 to 3.
    - It uses exact resume's multi-row extent-1 entries.
    - Speculative decoding is listed as not in the server's first version (docs/SERVER.md), so SERVER.md changes with the speculative decoding plan's step 3.
  - Step 10 waits for the quantization plan's `feat/half-weights`, `feat/mxfp4-vulkan` and `feat/iq4-vulkan`, and for steps 5 and 7.
  - DeepSeek 4.x comes after this plan.

## Test and CI coverage (planned 2026-09-25)

- **Goal:** every feature a user can reach has a test the hosted workflow runs, or its STATUS block names the hand check that covers it and says why no hosted runner can. A bug fix lands its failing test first. The workflow runs everything that needs no GPU. What needs the cards is still run on them by hand, as it is today.
- **Found** by three read-only audits: tests against features, the workflow as it runs, and software Vulkan on a hosted runner. Nothing was run, and every time below is an estimate, because the job durations could not be read here.
  - The workflow already runs every registered CTest and every `run_tests.py` component. The three CPU jobs and the UBSan job run both. The HF job adds the three pinned 0.6B models. The Vulkan job builds and runs CTest, where `backend-vulkan` and `vulkan-lifetime` skip without a device.
  - The sampler has no test at all, although `generate` and `chat` sample by default (temperature 0.8, top-k 40, top-p 0.95).
  - The server's `stop`, `top_k`, `top_p` and penalty fields have no test.
  - The layer split's pipelined prefill has one CI case, 2 chunks over 2 stages (`placement.cpp`, the `place_model` case). The reuse of pass slots and handoff buffers, three stages, a MoE split and the rollback of a failed pipelined prompt are reached only on the MI50s.
  - Q4_K and Q4_1 are checked only against llmx's own `dequantize`, although Q4_K_M is the most common download.
  - The known server bugs have no failing test. `check_uncapped` counts pauses but never compares a paused request's output with the same request run alone.
  - `run_tests.py --cache-type` fails today, and AGENTS.md describes it as working. `common.device_args` adds cache flags to the synthetic bench, which refuses them, so `perf` and `threads` fail.
  - The workflow runs on pushes to main and on pull requests, and only main is pushed, so it has never checked a branch before its merge. STATUS records no hosted run since `11f5859`, 57 commits back.
  - Hosted runners cannot run the device paths. The software Vulkan driver their Linux image can install reports subgroups of 8 lanes, or 16 at most. The backend refuses fewer than 32 when it opens a device, and three kernel families really need 32: the activation quantization's shuffles, the general attention kernel's per-subgroup arrays, and MoE routing's partial results.
- **Every gate:** these branches change tests, tools, the build and the workflow, not the runtime. The gate is CTest and the Python suites on the CPU on Windows and Linux, on the Radeon VII and on an MI50, plus a green hosted run. A new test that fails on main is a finding, fixed in its own branch before the test merges. Each merge records its hosted run and job times in STATUS.
- **Done (2026-09-25), at `a2b732f`:** `test/sampler`, `test/quant-decode`, `test/cli-surface` and `test/reference-8b-per-token`. Gates: CTest and the suites on the CPU on Windows and Linux, suites on one MI50 and on a 2-card split, and the 8B HF check 41 of 41 on one MI50. The hosted run on `a2b732f` passed all six jobs: HF reference 7 min 33 s, macOS 3 min 34 s, Windows 3 min 7 s, the Vulkan build 2 min 9 s, UBSan 2 min 6 s and Linux 1 min 40 s.
- **Also done:** `hub-pull` waits for its four ranges to meet instead of sleeping 30 ms, and `model-validation` may take 120 s, since both failed only on a loaded machine; the Windows job builds with `build.bat` too and runs the suite on that binary; `docs/BUILD.md` covers building on every platform.
- **Done on `ci/hosted-coverage` (2026-09-25, written on `gate/merge-4` at `ea3a255` and rebased onto `gate/merge-8`, merged with it on 2026-09-26):**
  - Item 9: `tests/server.py` runs its MoE server check on the CPU too, each prompt's ids alone against four at a time, without the host-expert flags; on a single device it keeps them.
  - Item 6's leftover: `run_tests.py --only` runs the components it names and refuses an unknown or empty one, and the HF job runs `baseline` a second time with `--cache-type f32`.
  - The CI changes as listed above: `--parallel 4`; the Vulkan job's suite with `--device cpu --require-tools`; the HF job's `llmx-split-check cpu cpu,cpu 8 64` over the perplexity excerpt and `tools/server_mix_check.py --requests 8 --cli 2`, and no CTest step; `tests/data/fixtures.json`, the pinned specs alone, read by `baseline.py` and `tools/fetch_test_models.py` and hashed by the cache key, `baseline.py` refusing to load unless each pinned file has bounds and each bounded file is pinned once, with restore and save as separate steps and the save right after the verified fetch; `cancel-in-progress` for pull requests only, every other run in a concurrency group of its own so no push to main or to a `gate/<name>` branch is cancelled or dropped from the queue; the Ubuntu job byte-compiles `tests/` and `tools/` and runs every tool's `--help`. Two tools needed a change for that: `fetch_test_models.py` downloaded on any argument, and `gen_chat_baseline.py` imported torch before reading its arguments. The Windows job's run on the `build.bat` binary stays without `--require-tools`.
  - The decided Q4_K_M fixture: `Qwen3-0.6B-Q4_K_M.gguf` from `unsloth/Qwen3-0.6B-GGUF` at the Q4_0 file's revision (Qwen's own GGUF repo has only Q8_0), 168 Q4_K, 29 Q6_K and 113 F32 tensors. Bounds by the Q5_K_M rule, the measured deltas plus the margins Q5_K_M left: measured 0.105 continuous and 0.215 per chunk on the CPU, bounds 0.13 and 0.25, top-5 overlap 4 (`docs/ASSETS.md`).
  - The docs-only list, each item checked against the code first: all were still wrong except the two branch 3 and `test/split-coverage` had already fixed (`llmx-split-check` built only with Vulkan, `--cache-type` failing). `docs/CI.md` also drops the self-hosted runner, which was decided against, and puts the f32 pass at 5 to 7 minutes; the paragraphs this branch touched in CI and AGENTS hold whole sentences per line.
  - **Gates on Windows at `2e7a353` (now `7b25cca` after the rebases):** CTest 22/22 on the CPU tree. The Vulkan tree builds, and its 23 CTests that need no device pass; `backend-vulkan` and `vulkan-lifetime` were not run, since this branch's gate keeps the local GPU out. The suite with `--require-tools --device cpu` and all four fixtures on disk: 16 of 16 in 7 min 40 s, the real-model server checks, the four fixtures' HF gate and the MoE server check on the CPU included, with no component skipped. The Vulkan job's command, the suite with `--device cpu --require-tools` on the Vulkan-enabled binary, with the loader finding no driver and no HF cache as on a hosted runner: 16 of 16 in 20 s. The `--cache-type f32 --only baseline` pass holds all four fixtures in 4 min 37 s. `--only` with an empty name, a stray comma or an unknown name exits 2, and a fixture list with a file missing, an unbounded file or a file pinned twice stops `baseline.py` with both lists named. The Ubuntu job's byte-compile and nine `--help` runs, the fetcher's 15 offline cases and the four fixtures' SHA-256 verification pass.
  - **Gates on Windows at `1a21c8b` (now `43d153f`):** the HF job's other added commands, run on the build of `ea3a255`, since no C++ source differs: `llmx-split-check cpu cpu,cpu 8 64` on the Q8_0 over the excerpt is bit-identical (247 positions, the chunked prefill and 8 steps, 69 mixed rows) in 14 s; `server_mix_check.py --requests 8 --cli 2` matches every request (0 of 8 differ together, 0 of 6 skewed with 2 clients leaving, both CLI checks) in 5 min 21 s, with other suites loading the machine. The Ubuntu job's byte-compile and `--help` loop pass on Windows and under WSL's Python 3.12.3, where the base's `gen_chat_baseline.py --help` failed without torch. The workflow parses and validates against the published GitHub workflow schema. CPU outputs (`generate -n 64 --temp 0`, `logits --top 20`, `perplexity` both ways, `tokenize` on Q8_0, Q4_0 and Q5_K_M) are byte-identical to `ea3a255`; no runtime code changed.
  - **Merged with `gate/merge-8` (2026-09-26),** together with `cleanup/cli-help`, `refactor/cli-exec-options` and `fix/top5-llmx-tie`, gated at `73f4f78` against main `d48f2b2` built beside it on an MI50:
    - CTest 25/25 on the card, and the Python suite with `--require-baseline` 17 of 17 on one MI50 and on a two-MI50 split, the Q4_K_M fixture running on the device within the bounds set on the CPU.
    - On an MI50 and CPU split every component passed except `server`, whose uncapped requests timed out while the host ran at load average 40 to 64; that component then passed alone for main and for the stack.
    - CLI output byte-identical to main in 32 cases on the card and 26 on the CPU, and `tools/server_mix_check.py` equal for every request (0 of 16 differ together, 0 of 12 skewed, the CLI checks equal).
    - 8B Q8_0 on one MI50, main, stack, stack, main, three times: pp512 1297 to 1303 tok/s against 1300 to 1303, and tg128 medians 62.0 against 62.7.
    - The CPU suite on Linux ran in the hosted jobs on the same commit. The Radeon VII suites were not run, since the stack changes no device, model, kernel or loader code, which the merge rules exempt.
    - The hosted run on `73f4f78` passed all six jobs: HF reference 15 min 13 s, macOS 3 min 54 s, Windows 2 min 58 s, the Vulkan build with its CPU suite 2 min 6 s, UBSan 1 min 54 s, Linux 1 min 5 s. The HF job keeps its 30-minute limit, about twice its time, and `server_mix_check.py` keeps 8 requests there.
    - `fix/top5-llmx-tie`, approved by the user on 2026-09-25: the HF top-5 check also forgives the reference's 5th and 6th trading places when llmx's own logits for the two are within 0.1. On an MI50 and CPU split, 0.6B Q8_0 put " black" and " orange" 0.076 apart where the reference has 0.203, which had failed that split's baseline on main; every placement agrees on the four tokens above them.
- **Found while fixing those flakes:** creating a CPU backend starts a worker per logical CPU, and the `set_threads` that follows stops that pool and starts another, so `model-validation`, which builds hundreds of small models, starts thousands of threads. `fix/cpu-pool` starts the pool once, at the size used.
- **Missing tests, in order of value:**
  1. `test/sampler` adds a CTest `sampler` (`tests/sampler.cpp`, CPU, every job) that calls `infer::sample` on hand-computed logits:
     - Temperature 0 takes the largest score, and the lowest id on a tie.
     - The penalty divides a seen token's positive logit and multiplies a negative one, so a repeated leader loses to the runner-up.
     - `top_k` 1 is greedy at any temperature, and `top_k` k never returns a token outside the k best over 10,000 seeded draws.
     - A `top_p` between the first probability and the sum of the first two keeps exactly two tokens.
     - Draw frequencies at temperatures 1 and 0.5 fall within three standard deviations of the softmax at that temperature. This also catches a temperature applied twice.
     - A seed repeats its sequence, and seed 0 keeps the default state.
  2. The server's request fields are tested in `fix/server-http` (branch 2), which rewrites how they are read. The checks go in `tests/server.py` on the synthetic model, every CPU job:
     - A seeded sampled request with `temperature`, `top_k`, `top_p` and `repetition_penalty` (native `penalty`) gives the ids of `generate` with the same flags and seed, as the batch-invariance rule requires.
     - `stop`, as a string and as an array, ends the reply where `generate --stop` ends it, with `finish_reason` stop.
     - Bad JSON, a body that is not an object, a non-text content part, and each number field outside the range the branch sets are refused with 400.
     - The branch's own bugs get their failing tests first in the `http` CTest: after a failed `accept()` the listener still answers, and a streamed handler that fails after its first chunk ends the body without a second status line inside it.
  3. `test/split-coverage` covers the pipelined prefill, after the layer split merges (CPU, every job, UBSan included):
     - In `tests/placement.cpp`, a three-layer tiny model goes through `place_model` over two and three CPU backends at ubatch 3 with a 13-token prompt. That is 5 chunks, more than the stages, so pass slots and both handoff buffers are reused. Prefill, three decode steps, `score()` row by row, a two-sequence pass, and a second prompt continuing the first must all be exact against one backend, with `n_tokens` and `kv_used_bytes` equal.
     - In the same file, a backend on the last stage fails after the first stage has committed chunks, once mid-prompt and once at the head on the last chunk, on top of an existing history. Every storage's length and `kv_used_bytes` must be back where they were before the call, and the same prompt run again must be exact. For this, `FailingCpu` moves from `kv_cache.cpp` into `tests/tiny_qwen.hpp`.
     - `llmx-split-check` is built in every CMake configuration and links the Vulkan library only when that backend is on. It builds its devices with `make_backends`, which does not apply the CLI's listed-once rule, so `cpu,cpu` works.
     - A new `split` component in `run_tests.py` writes the tiny F32 model (tied and untied) and the tiny MoE model. It runs the tool on `cpu` against `cpu,cpu`, and on the three-layer MoE against `cpu,cpu,cpu`, at ubatch 1, 3 and 16, with 3 decode steps over a 13-token text, inside the 16-token context.
     - The component looks for the tool beside `--exe`. A new flag, `--require-tools`, turns a missing tool into a failure instead of a skip, and every CI job passes it.
  4. `fix/server-pause-prefill` (branch 1) lands its failing test first, in `tests/server.py` on the real Q8_0 in the HF job. `check_uncapped` compares each paused request's ids with the same request run alone. It also adds prompts long enough that the pool runs out while one of them is still prefilling. `fix/server-cancel` keeps the tests it already plans.
  5. `test/quant-decode` extends `tests/roundtrip.py` (CPU, every job) to Q4_1 and Q4_K:
     - The test writes a one-tensor GGUF of raw Q4_1 blocks and one of Q4_K super-blocks, with bytes chosen to reach every scale, min and nibble bit.
     - `llmx dequantize` must match, bit for bit, a decoder written from the format description, as the Q8_0 and Q4_0 check does now.
     - `q8-dots` and `backend-group`, which compare against `dequantize`, then rest on an independent decode.
     - Q5_K and Q6_K stay with the HF Q5_K_M fixture, which covers them end to end.
  6. `fix/kv-cache-default` (branch 3) also takes on the following:
     - `common.device_args` leaves the cache flags off the synthetic bench.
     - The model layer's default becomes f16, so CTest and `llmx-split-check` run what users run, and the exact-f32 cases ask for f32 themselves.
     - `kv-cache` checks f16 and both mixed pairs on the CPU storage against a double-precision reference computed over the f16-rounded K and V.
     - `run_tests.py` gets `--only`, and the HF job gets a second pass with `--cache-type f32 --only baseline`, about 3 to 4 minutes.
  7. `test/cli-surface` adds a `cli` component (`tests/cli.py`, CPU, every job) and extends `tests/f32.py`:
     - `--device vulkan:0` exits non-zero with the no-Vulkan message in a CPU build, and with the device error in a Vulkan build that has no device.
     - `info` on the synthetic model names its architecture, layer count and tensor count.
     - `logits --file` matches the inline prompt, and the `--last N` and `--then-ids` rows fall within the fixture's HF bound at their positions.
     - `bench --model --seqs 2` runs and reports finite speeds.
     - Branch 5 (`cleanup/cli-arguments`) adds to this component the refusals from 010d7d9 (model flags on the synthetic bench, `generate --system`, `-n 0`), plus bad and negative numbers. It also has the server refuse a `temperature`, `top_k`, `top_p` or `penalty` outside the range the CLI's flag takes with 400, a `top_k` of -1 still taken as 0 on the compatible routes, and `tests/server.py` checks both.
     - Branch 10 (`cleanup/cli-help`) adds the help check: every flag a command's help lists is accepted by that command, a flag it does not list is refused, and help needs no model.
  8. `fix/tokenizer-metadata` (8) and `fix/chat-template-defined` (9) each land their failing test first. For branch 8, a synthetic GGUF declaring another tokenizer is refused with a message (`tests/tokenizer.py`). For branch 9, a template testing `is defined` takes the else branch for a variable that is absent (`chat-template` CTest).
  9. `tests/server.py` also runs its MoE server check on the CPU, as part of `ci/hosted-coverage`. The routed model's ids for each request alone must equal its ids four at a time, without the host-expert flags that need a device. This adds about 10 seconds per job.
- **CI changes** go in `ci/hosted-coverage`, which comes after `test/split-coverage` and edits `.github/workflows/ci.yml`:
  - Builds use `--parallel 4`, the hosted runners' core count. That saves about 10 to 20 runner-minutes per run.
  - The Vulkan job installs Python and runs the suite with `--require-tools` on the CPU path of the Vulkan-enabled binary. That is the build Linux GPU users make, and no job runs its suite today. About 3 minutes.
  - The HF job adds two runs on the 0.6B Q8_0:
    - `llmx-split-check cpu cpu,cpu 8 64` over the perplexity excerpt, about 1 minute.
    - `tools/server_mix_check.py` on the CPU with `--requests 8 --cli 2`, about 4 to 6 minutes, cut to 4 requests if the first run measures over 8 minutes. It would be the only hosted check of long prompts landing while others decode, and of clients leaving.
  - The HF job drops its CTest step, which the Ubuntu job already runs on the same build. That saves about 1 minute.
  - The HF cache key hashes a file holding only the pinned model specs (`tests/data/fixtures.json`, read by `baseline.py` and `tools/fetch_test_models.py`) instead of all of `baseline.py`, which changes far more often than its model list does. Restore and save become separate steps, with the save right after the verified fetch, so a red run keeps its 1.5 GB of downloads.
  - `cancel-in-progress` applies to pull requests only, so every push to main keeps its run as the record of that merge.
  - The Ubuntu job byte-compiles `tests/` and `tools/` and runs each Python tool's `--help`, so a tool broken by a CLI change fails there instead of on the cards. About 0.2 minutes.
  - **Net:** about -5 to +10 runner-minutes on an estimated 60 to 85 per run. The HF job grows by about 7 to 10 minutes against its 30-minute limit, and the first run's measured times decide the limit. docs/CI.md records those times.
- **Docs, fixed now in a docs-only commit:**
  - AGENTS.md, working rules: add that a feature lands with a test the workflow runs, or its STATUS block names the hand check and says why no hosted runner can run it. Add that a bug fix lands its failing test first.
  - AGENTS.md, test descriptions:
    - `http` runs on Linux, Windows and macOS, not only Windows and Linux.
    - `placement` also covers the fit, `place_model` and one pipelined prefill.
    - Round-trip starts from an F32 model, quantizes to Q8_0 and Q4_0, and decodes both from the format description.
    - The server's uncapped check confirms each request finishes and counts pauses. It does not yet compare output with the request run alone.
    - The MoE component also runs streamed placements on a device.
    - `vulkan-buffer` needs no loader, because a fake device supplies every call.
    - `llmx-split-check` is built only with Vulkan today, and it accepts `cpu,cpu` because it builds its backends without the CLI's device-list check.
    - `--cache-type` fails in `perf` and `threads` until branch 3 fixes it.
  - docs/CI.md:
    - `vulkan-buffer` needs no loader, in the table and in the prefill-scope section.
    - The Vulkan-only CTests run in one job, where two of the three skip, and that job runs no Python.
    - The HF row adds the real-model server checks (limits, pausing, prefix reuse) and the HF chat and thread replies.
    - The only layer split a hosted job runs is the placement CTest's two CPU backends.
    - "legacy filtering" goes, since the thinking filter was removed.
    - The historical pass counts get their commit next to them.
    - The workflow never checks a branch before its merge.
  - The `run_tests.py --require-baseline` help text says "any of the three fixtures", not "either".
  - Each test branch above updates these descriptions for what it adds.
- **Stays hand-run on the cards, and why:**
  - The Vulkan kernels and attention combinations (`backend-vulkan`), `vulkan-lifetime`, the suites with `--device vulkan:0`, HF bounds on a device, experts on the host and streamed, pools of different block sizes beside the CPU, splits over cards (`--device a,b` and `llmx-split-check` on 2 to 4 MI50s), and `ensure()` waiting on a device:
    - Hosted runners have no GPU, and the software driver's narrow subgroups are refused.
    - Kernels rewritten for narrow subgroups would run only in CI, in the code most sensitive to precision.
    - Even then, `backend-vulkan` would take about 4 to 8 minutes against its 120-second limit, and the suite on 0.6B would take about 0.5 to 2 hours.
  - Performance floors, A/B runs and every comparison with the reference need quiet, pinned cards.
  - Serving speed (`tools/server_load.py`), the 16k check (`tools/long_context_check.py`, a device against the CPU), many users on a card split (`tools/server_mix_check.py`) and the 8B HF check (`tests/baseline_8b.py`) need the cards, or weights and time beyond a hosted runner.
  - The merge gate on both platforms, the Linux MI50s and the Windows Radeon VII, stays by hand whatever the workflow runs.
- **Decided (2026-09-25):**
  - No self-hosted runner on the Linux MI50 machine: it runs other work, and the gates on the cards stay by hand on both platforms.
  - Branches are not pushed to GitHub before their merge; each branch's local gate runs the same commands. Changed later that day: a stack about to merge is pushed as `gate/<name>` so the hosted workflow runs on it first, and the branch is deleted after the merge.
  - A pinned Qwen3-0.6B Q4_K_M fixture joins the HF job with HF bounds, so K-quant decoding meets the reference in CI.
  - `test/reference-8b-per-token` goes ahead: the 8B HF check scores batched and per-token with its bounds unchanged, and the 0.6B check takes the 8B check's stricter validators.
  - The hosted Windows job also builds with `build.bat`, which now finds Visual Studio itself, requires its binary to report the CMake build's version, and runs the suite on it, so the first Windows route in the build docs has an automated check.
- **Not doing:**
  - A software Vulkan job: the kernel changes it needs would be tested only by that job.
  - A test-only way past the subgroup check to run the storage and submission checks: `vulkan-buffer` already covers them with its fake device and no runtime hook.
  - A CPU alias so the CLI can split on the CPU: the listed-once refusal exists because two stages would drive one backend and count its free memory twice. The placement CTest and `llmx-split-check` cover the split without it.
  - A nightly long-context run on the CPU against itself: it proves only determinism, which the server checks already require.
  - The real-model server pass on Windows and macOS: the synthetic server pass already runs on every platform.
  - `MappedFile::drop` with a payload larger than host memory, and the split lines of `--verbose`: they need memory or devices a hosted runner lacks.
  - Tests that would repeat existing ones: `--stop` in the CLI, the tokenizer against HF, perplexity, device-list parsing, and Q5_K and Q6_K.
- **Sequencing:**
  - Everything starts after the layer split merges.
  - `test/split-coverage` goes first, since it covers this branch's feature.
  - `test/sampler`, `test/quant-decode`, `test/cli-surface` and `ci/hosted-coverage` run in parallel with the correctness branches, since they touch other files. The exceptions are `run_tests.py`, `CMakeLists.txt` and `ci.yml`, which these branches change in that order.
  - Branch 3's `--only` flag and cache-type pass land after `ci/hosted-coverage`.
  - Branches 5 and 10 add to `tests/cli.py` after `test/cli-surface`.
  - The server field tests go with branch 2, and the pause test with branch 1.

## Loader in one place, with a load mode (planned 2026-09-25, branch refactor/loader off main e039b62)

- **Goal:**
  - Loading a model has one owner, `infer::load_model` in a new `src/inference/load.hpp`. It reads the files, builds the tokenizer and the chat format, places the model, fills the weights and settles the host copy.
  - The format layer parses, maps and reads files. The model checks roles and asks the loader for each weight's storage. Backends allocate and copy. The CLI and the tools turn flags into a request.
  - The model takes a format-neutral input, `QwenWeights`: the config plus one view per tensor, with the fields the safetensors branch already uses. The loader streams from format-neutral file spans. A second format is then a reader that produces both, plus one branch in `load_model`.
  - Weights reach a device through large reads in file order, overlapped with the uploads, instead of page faults inside the upload copy. `--load-mode auto|mapped|direct` chooses how files are read, direct I/O included, with the same meaning on every backend.
  - Every step leaves logits, greedy text, stdout and the split plan byte-identical to main. No load is slower than main on either machine, cold or warm.
  - Expected, estimated from measured rates but not yet measured:
    - Qwen3-235B-A22B Q4_K_M on six MI50s goes from 370 s to about 60 to 70 s, with progress shown throughout.
    - A cold Qwen3-8B Q8_0 on one MI50 goes from about 22 s at the measured fault rate to about 4 s.
    - Warm loads stay level with main.
- **Decided (2026-09-25):**
  - Backends are created before the file is read, so bad flags fail fast.
  - `format::ModelFormat` is removed.
  - Direct-I/O loading, with reads overlapped with uploads, is selected by a load-mode flag.
  - The loader is one tight, reusable design, never worse than today.
- **Problems, at e039b62:**
  1. **Loading is spread over three layers.**
     - `format/gguf.hpp:read_gguf` touches every page (557-579) before any placement exists.
     - `model/arch_qwen.hpp:Model::Model` drops the pages of copied tensors (549-551).
     - `cli/main.cpp:open_model` releases the host copy (264).
     - `tools/split_check.cpp:main` and `tools/compare_cpu.cpp:main` repeat parts of that sequence and never release.
  2. **Two signals answer "does a host read this weight in place".** `Model::note_reader` (965) compares `host_ptr()` with `GGUFModel::holds`. `place_model`, `Model::resolve_tensors` and `model/layer_split.hpp:budgets_for` ask `Backend::reads_in_place`.
  3. **The seam kept for a second format is not the one loading uses.** `format::ModelFormat`, `gguf::GGUFFormat` and `format::open` carry no tensor bytes, and only `tests/load_progress.cpp:38` and `tests/gguf_shards.cpp:162,179` call them. Meanwhile `Model`, `footprint`, `routed_layers`, `place_model`, the tokenizer, `chat::chat_format` and `server::Api` all take a `gguf::GGUFModel`; since `feat/chat-template-jinja`, `server::Api` takes a `chat::ChatFormat` instead.
  4. **One load parses the config three times.** `load_config` runs in `Model::Model` (469), `footprint` (399) and `place_model` (1372). `footprint` also decides dense layers by a `.ffn_gate.weight` substring (424), where the resolver and `routed_layers` look for the router tensor.
  5. **The tensor table is checked twice.** `read_gguf` refuses duplicate names only across shards (500-507). `Model::Model` then checks count, duplicates, rank and extent again (529-541).
  6. **Commands that need only metadata read the whole payload.** `cmd_info`, `cmd_tokenize` and `cmd_detokenize` map and touch it through `read_gguf`.
  7. **Lifetime rests on caller conventions.**
     - `Model` keeps `const GGUFModel* m_` (837).
     - The CLI's `Opened` is "built in place and never moved".
     - `server::serve` and `Api` took the whole file only to call `chat_format` (`server/api.hpp:64`); `feat/chat-template-jinja` has them take a `const chat::ChatFormat&`, built once in the CLI's `Opened`.
  8. **The written `adopt` contract is the opposite of what the release relies on.**
     - `backend.hpp:160`, `docs/src/backends-backend.md` and `docs/DEVICE-EXECUTION.md` say the source must outlive the buffer.
     - Releasing the host copy relies on a copying backend having consumed the source when `adopt` returns (`VulkanBackend::upload`, 2405).
     - `tests/model_validation.cpp:LoadingBackend` reports `reads_in_place()` false but aliases the source.
  9. **Reads are page faults.** The touch reads one byte in every 4096, in 8 MiB steps. A diagnostic measured page-fault reads at about 400 MB/s, against 2.1 to 2.4 GB/s for 16 MiB reads from the same ZFS pool, buffered and direct alike (Multi-device phase 1, Done, loading).
  10. **Large models fault silently in role order.**
      - Above available host memory, `read_gguf` reports 100% at once (560-562).
      - Every page is then faulted inside the memcpy of `vulkan_backend.cpp:VulkanBackend::upload` (2413), one tensor at a time. The order is `Model::resolve_tensors`' role order (869-951), not the file's, so reads jump around the file.
      - The disk reads only inside that host copy, so it idles through every allocation, submit and wait.
      - The 235B spends 370 s under "Preparing model..." with no progress.
  11. **A tensor two devices take is read twice.** This covers a tied head on another device (`resolve_tensors`, 895) and a streamed layer's norm and router (922-923).
  12. **Some failures come after a lot of uploading.** A missing tensor, or a cache that does not fit, is found only after every weight before it has been uploaded.
  13. **ZFS caches a mapped file twice**, in the ARC and in the page cache. The Linux host has about 22 GB available beside other work.
- **Plan.** Each step is one commit. Outputs are byte-identical to main in every step.
  0. **Docs.** This block, plus agreeing `TensorView`, `AdoptWeight` and `format::FileSpan` with the safetensors branch before step 2.
  1. **One loader owns the load sequence.**
     - **New:** `inference/load.hpp` with:
       - `struct LoadedModel { gguf::GGUFModel file; std::vector<core::HostPages> host; std::optional<bpe::Tokenizer> tok; chat::ChatFormat chat; std::unique_ptr<Model> model; std::string plan; }`. `model` is declared last, so it is destroyed first.
       - `load_model(path, backends, request, options, progress)`, which runs today's `open_model` sequence.
     - **Moved:** backends are now made before the file is read, so a bad `--device`, `--cache-type` or `--layer-shares` fails before "Reading model metadata...".
     - **The CLI keeps:** device specs, `make_backends`, the request and options built from flags, the progress renderer (it prints "Preparing model..." when the bar completes), the plan print and `set_threads`.
     - **Other callers:**
       - `bench --profile` gets its backend through an out-parameter.
       - `cmd_chat` and `cmd_serve` use `loaded->chat` where `feat/chat-template-jinja` has them use `opened->chat`, so `load_model` takes over building the format from `open_model`.
       - `server::serve` and `Api` take a `const chat::ChatFormat&`, as they do since `feat/chat-template-jinja`.
       - `split_check` makes two `load_model` calls, and `compare_cpu` makes one.
     - **Deleted:** `Opened`, the CLI's `load_model`, the release at `main.cpp:264`, and the tools' own open sequences.
     - **Tests:** `load-progress` writes `tiny_qwen` with tokenizer metadata and loads it twice. On the CPU the payload is kept. On a CPU subclass that copies what it adopts and reports `reads_in_place()` false, `payload_size()` must be 0. In both cases the logits must be bit-identical to the model built in memory.
     - **Docs:**
       - new `docs/src/inference-load.md`;
       - `docs/src/cli-main.md` and `docs/src/server.md`;
       - the ARCHITECTURE table, layer diagram and error-handling paragraph;
       - the AGENTS description of `load-progress`.
  2. **The model is built from `QwenWeights`, and the loader decides each weight's storage.**
     - **New in `arch_qwen.hpp`:**
       - `TensorView {name, shape, type, data, bytes}`, where `data` is null when the bytes are not in memory.
       - `QwenWeights {config, tensors}`, tensors in file order. Tensor `i` is the file's tensor `i`.
       - `gguf_weights(const GGUFModel&)`. It runs `load_config` once and takes over the table checks from 529-541.
       - `using AdoptWeight = std::function<backend::BufferPtr(size_t tensor, backend::Backend&)>`. The default is `b.adopt(view.data, view.bytes)`.
     - **Changed signatures:**
       - `Model(const QwenWeights&, backends, Placement, ModelOptions = {}, const AdoptWeight& = {})` and `place_model(const QwenWeights&, ..., const AdoptWeight& = {})`.
       - The hook is called at the two adoption sites (888 in the check lambda, 959 in `experts`). RoPE (588-589) still adopts inline.
       - The two GGUF constructors become one-line forwards through `gguf_weights`, for the fixtures and the synthetic bench.
       - `footprint` and `routed_layers` take `QwenWeights`, and `footprint` takes dense layers from `routed_layers`.
     - **The loader's hook** still adopts inline, as today. It records per tensor whether `b.reads_in_place()`. After construction it releases the payload when no host reads a weight, and otherwise drops the pages of every tensor no host reads.
     - **Deleted:**
       - from `Model`: `m_`, the `tindex_` and `out_name_` members (they become constructor locals), `host_reads_`, `copied_`, `holds_payload_`, `holds_payload()`, `note_reader`, and the drop loop at 549-551;
       - `GGUFModel::holds`.
     - **Tests:**
       - `placement.cpp` wraps its fixtures in `gguf_weights`.
       - `model_validation`, through a recording hook: nothing is read in place on `LoadingBackend`. With a device plus a streaming host, exactly `blk.0`'s `ffn_norm`, router and three expert stacks are read in place.
       - `LoadingBackend` copies what it adopts.
       - A one-off check shows `footprint` unchanged on every local model file.
     - **Docs:**
       - The real `adopt` contract in `backend.hpp:160`, `backends-backend.md` and `DEVICE-EXECUTION.md`: a backend that reads in place borrows the source for the buffer's life and does not read it inside `adopt`, and a copying backend has consumed it when `adopt` returns.
       - `docs/src/model-arch_qwen.md` (the pointer rule, and 199-201).
       - `docs/src/format-mapped_file.md`.
       - `docs/EXECUTION.md:234`: "costs no RAM" is wrong, since the page cache is RAM.
  3. **Reading a file maps and touches nothing, and `ModelFormat` goes.**
     - **`read_gguf(path)`:**
       - It parses, checks and lays out `Segment{path, file, start, base, size}`, with `file` null until mapped.
       - It refuses duplicate tensor names in every file, not only across shards.
       - It loses its progress parameter.
     - **`map_payload(GGUFModel&)`** maps each segment. It refuses a file whose size changed since its header was read, taking over the extent checks at 544-550.
     - **`GGUFModel::span(i)`** returns a `format::FileSpan{file, offset, bytes}`.
     - **`gguf_weights`** gives a view a null `data` while its tensor's segment is unmapped, deciding per segment by its `file`, not by whether segments exist.
     - **`format/format.hpp`** keeps `LoadProgress` and gains `FileSpan`. `ModelFormat`, `format::Tensor`, `ModelFormatPtr`, `format::open` and `GGUFFormat` are deleted.
     - **`gguf::warm(m, tensors, progress)`** is today's loop at 564-579 over the given tensors, with the same 8 MiB steps. The host-memory rule moves into the loader.
     - **`core::page_size()`** in `core/host_memory.hpp` replaces `MappedFile::page_size` and the literal 4096.
     - **The loader** calls `map_payload` and then `warm` over every tensor right after `read_gguf`, the same point and progress as today.
     - **Callers:** `dequantize` calls `map_payload`. `info`, `tokenize` and `detokenize` map nothing, so on Windows they no longer lock the file.
     - **Tests:**
       - Progress assertions call `read_gguf`, `map_payload` and `warm` in that order.
       - The adapter checks become direct `GGUFModel` checks.
       - `gguf-validation` gains a refusal of duplicate names in a single file.
       - A file truncated between `read_gguf` and `map_payload` is refused.
     - **Docs:**
       - `format-format.md`, `format-gguf.md` and `core-host_memory.md`;
       - ARCHITECTURE: the `format/` line, the table, "Progress and text delivery", and the loader paragraph in the present tense;
       - the `format/` row in AGENTS;
       - ROADMAP §3;
       - the audit block's "`ModelFormat` stays" note;
       - the USAGE progress paragraph.
  4. **Plan, then fill: every weight is allocated before any is filled, and filled in file order.**
     - **`Backend::alloc_weight(bytes)`**, which defaults to `alloc(bytes)`, gives storage the caller fills with `write` before any op reads it. The Vulkan override makes a device-local buffer with no fill and sets `adopted`, so `padded_f32` still applies. Vulkan's `adopt` becomes `alloc_weight` plus `upload`.
     - **The loader's hook:**
       - A backend that copies gets `alloc_weight`, and the loader records an upload (tensor, backend, buffer).
       - A backend that reads in place adopts the mapped address, and the loader records a host read.
       - When the constructor returns, every role has been checked and every weight, window, cache and RoPE table has been allocated, in today's order.
     - **`detail::fill`** writes each upload in file order through `Backend::write`, in pieces of at most 16 MiB taken from the mapping. A tensor with two destinations is read once and written twice.
     - **Warming** is today's: every tensor, before placement, under today's rule.
     - **Effects:**
       - A missing tensor or a cache that does not fit now fails before any upload, with the same message.
       - Faults above host memory now happen in file order.
     - **Tests:**
       - `load-progress` loads the fixture from disk. Every device buffer is read back and compared with the file, and the logits must be identical to the in-memory model.
       - The fill also runs with 4 KiB pieces, so tensors cross piece boundaries.
       - `LoadingBackend` overrides `alloc_weight` and `write`, with a failure injected at write N, and must show `premature == 0`.
       - `backend-vulkan`: `alloc_weight` plus piecewise writes equals `adopt`, and the padded path is still taken.
     - **Docs:** `backend.hpp`, `backends-backend.md`, `DEVICE-EXECUTION.md`, `model-arch_qwen.md` and `inference-load.md`.
  5. **`--load-mode auto|mapped`: buffered reads on a reader thread, overlapped with the uploads.**
     - **New `format/file_reader.hpp:FileReader(path, direct)`**, buffered path:
       - POSIX: `open`, `posix_fadvise(SEQUENTIAL)`, then `pread` in a loop, retrying on `EINTR`.
       - Windows: `CreateFileW` with `FILE_SHARE_READ | FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED`, and `ReadFile` with an `OVERLAPPED` per call plus `GetOverlappedResult(TRUE)`. Synchronous completion and `ERROR_HANDLE_EOF` are both handled.
       - A read is short only at end of file, and it is safe from several threads.
       - `size()`; `granule()` is max(page, `st_blksize`), which is the recordsize on ZFS and the page on Windows.
     - **New `core::HostPages(bytes)`:** page-aligned memory llmx owns (`VirtualAlloc`, or anonymous `mmap`), freed on destruction.
     - **`enum class LoadMode { automatic, mapped, direct }`**, and `load_mode_of` beside `kv_type_of`.
     - **`mapped`** is step 4's path.
     - **`auto`:**
       - Pieces are planned from the uploads' spans. Each starts on the granule, is at most 16 MiB, and is merged with the one before while the file is the same. A gap longer than a granule starts a new piece.
       - One reader thread fills a ring of four 16 MiB `HostPages` slots in file order. The main thread writes each fragment to its destinations and then frees the slot.
       - Files are mapped only when a host reads a weight in place. Host-read tensors are warmed after the stream, under the host-memory rule applied to their bytes alone.
       - The payload is unmapped when no host reads a weight.
     - **Progress:**
       - `mapped` reports as today.
       - `auto` reports 0 after construction, then per piece and per 8 MiB warm step. The total is the bytes of every tensor some backend takes, each counted once. It reaches 100% after the last upload.
       - On the CPU alone, `auto` gives today's sequence whenever every tensor has a role.
     - **Timing:** under `--verbose`, and always under `bench`, one line gives the mode, the read method per file, and the time spent constructing, reading, uploading and waiting for reads.
     - **Measured in the same session, kept only if they win, recorded either way:**
       - a second reader thread;
       - 32 MiB pieces;
       - `MADV_WILLNEED`, `MADV_POPULATE_READ` or `PrefetchVirtualMemory` before warming the tensors a host reads.
     - **Tests:**
       - A new CTest, `file-reader`: aligned ranges, short reads at end of file, tiny files, several threads.
       - `load-progress` in `auto` and `mapped`, on the CPU and on the copying backend: buffers compared with the file, logits identical, and progress that starts at 0, only rises and ends at the total. Pieces of one granule are used, so tensors cross pieces and a piece holds several tensors.
       - Three failures must each drain every backend, join the reader and free the ring: a throw at the Nth write, a progress callback that throws, and a file truncated after `read_gguf` ("`<path>` ended at N bytes, before its tensors").
       - Shards, including a metadata-only shard.
       - `run_tests.py --load-mode M` sets `LLMX_LOAD_MODE`, which `tests/common.py` passes to the binary as a flag.
     - **Docs:**
       - the flag in `print_usage` and USAGE;
       - ARCHITECTURE "Progress and text delivery";
       - new `docs/src/format-file_reader.md`;
       - `core-host_memory.md` and `inference-load.md`.
  6. **Direct reads, `--load-mode direct`, and `auto` reading direct when the host cannot cache the model.**
     - **`FileReader`, direct path on Linux:**
       - `open(O_DIRECT)`, then `statx(STATX_DIOALIGN)` by raw syscall, using the kernel's struct layout.
       - It is refused when the mask lacks the bit or the memory alignment is zero. That covers kernels before 6.1, ZFS with `direct=disabled`, and file systems without direct I/O.
       - It is also refused on `EINVAL` from `open`, or when the memory alignment is larger than a page.
       - `align()` is max(page, `stx_dio_offset_align`), which is the recordsize on ZFS.
     - **`FileReader`, direct path on Windows:**
       - `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`.
       - `align()` is max(page, `LogicalBytesPerSector`, `PhysicalBytesPerSectorForPerformance`) from `FileStorageInfo`.
       - One aligned probe read at open must succeed.
     - **Misaligned requests:** a misaligned direct request is a `logic_error`, because ZFS 2.4 serves one through the ARC instead of refusing it. Buffers are always `HostPages`, and offsets and lengths are aligned by construction.
     - **`direct`:**
       - Device pieces are read direct into the ring.
       - Each run of consecutive host-read tensors is read direct into one `HostPages`, from the aligned start below the run, in the same front-to-back pass. Each tensor keeps its in-page offset from the file, so CPU kernels see the same alignment the mapping gives them.
       - The hook adopts those addresses. Nothing is mapped.
     - **`auto`** reads a file's device pieces direct when the bytes the load reads exceed `core::host_memory_available()` and the file takes direct reads. Otherwise it reads them buffered.
     - **Refusals**, before any byte is read:
       - "--load-mode direct: `<path>` is on a file system that does not take direct reads", at open, before the model is built.
       - "--load-mode direct: the host reads X GiB of weights in place and has Y GiB available; auto and mapped map them instead", after construction.
     - **Measured in the same session:** a CPU decode A/B between `mapped` and `direct` on both machines, since anonymous pages behave differently from file pages, with `MADV_HUGEPAGE` on the host copy as a variant.
     - **Tests:**
       - The step 5 loader tests run in `direct` where the temporary file system takes direct reads, and check the documented refusal where it does not.
       - `file-reader` covers the alignment and a rounded read past end of file.
       - A ZFS dataset with `direct=disabled` is checked by hand on the Linux host.
     - **Docs:** USAGE and `format-file_reader.md`.
  7. **Measured, taken only if it wins, recorded either way: devices copy from the ring without the host copy into staging.**
     - It is prototyped first.
     - If it wins, it lands as one backend call that wraps caller memory as a host-visible buffer. The CPU aliases the memory; Vulkan imports it once per slot through `VK_EXT_external_memory_host`, which both cards take at 4096-byte alignment.
     - The fill then copies each fragment with `copy`, and waits for a slot's last ticket before refilling it.
- **Done:**
  - Step 1 as planned, except that `LoadedModel` has no `host` member until step 5 brings `core::HostPages`.
    - A device that cannot be made now fails before "Reading model metadata...", which `tests/chat.py` checks. Main has since refused a bad `--layer-shares` or cache type as a usage error while the flags are read, so the check keeps only the devices.
    - Main's `cli` component, which came after this step, tries `bench --profile`'s line with `vulkan:0` on a model that does not exist and expected the missing file's error. Since the devices are made first, it also takes that device's refusal on a build or a machine without one.
    - Stdout is byte-identical to main on the CPU for Qwen3-0.6B Q8_0, Q4_0 and Q5_K_M: 64 greedy tokens, logits, perplexity batched and per token, `info`, `tokenize`, `detokenize` and `chat` with a fixed seed.
    - On one MI50, 64 greedy tokens and logits are byte-identical to main for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M, and so are a split over the MI50 and the CPU, `--n-cpu-moe 12` and `llmx-split-check`, apart from the free memory they report.
    - Warm loads of Qwen3-8B Q8_0 on one MI50 are level with main: 2.62 to 2.81 s against 2.82 to 3.23 s.
  - Step 2 as planned, except that it landed before the safetensors branch agreed to its types, which step 0 put first.
    - `TensorView` and `QwenWeights` use the safetensors branch's field names and meanings. Its `vocab_size` and `tie_output` are left out: a reader ties the head by leaving out `output.weight`, as a GGUF file does. They went to that branch's developer for objections, with `AdoptWeight` and step 3's `format::FileSpan`, before the step was committed, and no answer had come when it was.
    - The loader drops the pages of every tensor no host reads, which also covers a tensor no backend took; main dropped only the tensors a device copied.
    - The model also refuses a duplicate tensor name, since a second format's views reach it without `gguf_weights`' checks, and `model-validation` records through the loader's own hook, `recording_adopt`.
    - `format-gguf.md`, `format-format.md`, ROADMAP §3 and the ARCHITECTURE diagram named `GGUFModel::holds`, `Model::holds_payload` or a model that reads `GGUFModel`, so they changed with the step; step 3 still rewrites the last three.
    - `footprint` is unchanged on all 17 local model files: the 12 Qwen3 files give the same footprint, and the other 5 are refused with the same message.
    - Stdout and stderr are byte-identical to main on the CPU for Qwen3-0.6B Q8_0, Q4_0 and Q5_K_M (64 greedy tokens, logits, perplexity batched and per token, `info`, `tokenize`, `detokenize`, `chat` with a fixed seed), for Qwen3-30B-A3B Q4_K_M (16 greedy tokens and logits) and for the plan of a one-share split, apart from the free memory it reports.
    - On one MI50, 64 greedy tokens and logits are byte-identical to main for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M, and so are a split over the MI50 and the CPU, `--n-cpu-moe 12` and `llmx-split-check`, stderr included.
    - Host memory once a server serves is main's to 2 MiB: 130 MiB for Qwen3-8B Q8_0 on the MI50, 4427 MiB for Qwen3-30B-A3B with `--n-cpu-moe 12` (4323 MiB of it file pages), 511 MiB for Qwen3-0.6B split over the MI50 and the CPU, and 702 MiB for it on the CPU.
    - Loads of Qwen3-8B Q8_0 on one MI50 are level with main: 2.62 to 2.75 s against 2.67 to 2.79 s, eight each, interleaved, all of them warm. Half came after `posix_fadvise(DONTNEED)` on the file, which leaves a ZFS file's pages cached.
    - On the Windows CPU, placing the model on warm pages takes 117 to 415 ms on either side for Qwen3-0.6B, Qwen3-8B and Qwen3-30B-A3B, and whole loads of Qwen3-0.6B are level; whole loads of Qwen3-8B took 8.2 to 18.0 s on main and 9.5 to 43.7 s on the branch while other processes were loading models on the same machine.
  - Step 3 as planned, except that `format::FileSpan` was fixed before the safetensors branch answered, which step 0 put first.
    - `FileSpan {file, offset, bytes}` names the file by its UTF-8 path, the offset from the file's start, as proposed to that branch.
    - A `Segment` also holds the index of its first tensor, and `span`, `tensor_data` and `drop_pages` find a tensor's segment by its index, since a zero-sized tensor at the end of one shard has the offset the next shard starts at.
    - `GGUFModel::tensor_data` is null for a tensor whose file is not mapped, which is how `gguf_weights` gives such a view a null `data`, deciding per segment by its `file`.
    - The host-memory rule is `infer::detail::warm` in `load.hpp`. Above available memory it reports 0 and then the whole payload itself, as `read_gguf` did, and `gguf::bytes_of` gives the bytes it compares.
    - `map_payload` refuses a file that grew as well as one that shrank. `read_gguf` closes each header stream once it is read, and keeps the float-alignment check on tensor offsets.
    - A repeated tensor name is refused as "duplicate GGUF tensor: name" in one file and across shards, where shards said "duplicate GGUF shard tensor".
    - `write_gguf` refuses a model whose files are not mapped before it opens the output, and `warm` refuses a tensor that is not mapped before it reports progress, since `tensor_data` can now be null.
    - `core::page_size` throws when the operating system gives no positive size, and also gives the page size in the fallback of `core::host_memory_available`.
    - Stdout and stderr are byte-identical to main on the CPU for Qwen3-0.6B Q8_0, Q4_0 and Q5_K_M (64 greedy tokens, logits, perplexity batched and per token, `info`, `tokenize`, `detokenize`, `chat` with a fixed seed), for Qwen3-30B-A3B Q4_K_M (16 greedy tokens, logits, `info`), and for the F32 export of a small Q8_0 file (`dequantize`).
    - Above available host memory, Qwen3-30B-A3B Q8_0 (32.5 GB) on the Windows CPU with 9.5 GiB available prints the same "Loading tensor data: 0%", "100%" and "Preparing model..." as main.
    - On one MI50, 64 greedy tokens, logits, `info` and `tokenize` are byte-identical to main for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M, and so are a split over the MI50 and the CPU, `--n-cpu-moe 12` and `llmx-split-check`, stderr included.
    - Host memory once a server serves is main's: 129 MiB for Qwen3-8B Q8_0 on the MI50 and 4427 MiB for Qwen3-30B-A3B with `--n-cpu-moe 12`.
    - `info` on Qwen3-8B Q8_0 takes 0.11 s against main's 1.21 to 1.26 s, which read the whole payload from the cache.
    - Loads of Qwen3-8B Q8_0 on one MI50 are level with main: 2.64 to 2.71 s against 2.64 to 2.72 s, twelve each, interleaved, all of them warm, since `posix_fadvise(DONTNEED)` before half of them leaves a ZFS file's pages cached.
    - On the Windows CPU, reading Qwen3-0.6B Q8_0 in took 305 to 1849 ms on the branch and 362 to 1692 ms on main while other processes ran models on the same machine, so it gives no comparison.
    - After the review fixes, the same outputs are byte-identical again on the CPU, where the synthetic F32 model as one file and as three shards was added, and on the MI50.
    - Then, on the MI50, loads of Qwen3-8B Q8_0 took 2.66 to 2.90 s against main's 2.62 to 2.81 s, with medians of 2.70 and 2.69 s over twelve each, interleaved, and host memory once a server serves was main's to 1 MiB.
    - CTest and the Python suites pass on the MI50 with every command on the device.
    - Cold loads on one MI50 are level with main 3fcb428. Each run reads from a ZFS dataset with `primarycache=metadata`, remounted before it so that neither the ARC nor the page cache holds the file, interleaved with main:
      - Qwen3-8B Q8_0: 25.4 to 30.5 s against 23.9 to 29.3 s, medians 27.1 and 28.4 s, six each.
      - Qwen3-30B-A3B Q4_K_M: 54.1 to 118.4 s against 54.1 to 187.9 s, and with `--n-cpu-moe 12` 51.7 to 57.5 s against 46.1 to 59.1 s, four each. The slowest run on each side came in one round while other work took host memory.
  - Before the merge, steps 1 to 3 were rebased onto main 575a2ec and then 007b504 and gated against each; only step 1's `tests/cli.py` change came of it.
    - Stdout is byte-identical to main for 64 greedy tokens, logits, perplexity batched and per token, `info` and `tokenize`: on the CPU for Qwen3-0.6B Q8_0, and on one MI50 for Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M, alone and with `--n-cpu-moe 12`. So are a split over the MI50 and the CPU and `llmx-split-check`.
    - Over two MI50s split 1:1, the same outputs are byte-identical to main for Qwen3-0.6B and Qwen3-8B, and so are 64 greedy tokens and logits for Qwen3-30B-A3B. `llmx-split-check` finds the split bit-identical to one card on all three.
    - On the Radeon VII, the same outputs are byte-identical to main for Qwen3-0.6B and Qwen3-8B, for Qwen3-30B-A3B with `--n-cpu-moe 12`, and for Qwen3-0.6B split over the card and the CPU.
    - CTest and the Python suites pass on the CPU, on one MI50, and on the Radeon VII alone and split with the CPU, and every job of the hosted run passes.
    - Loads on one MI50 against main 575a2ec, interleaved, medians against main's: Qwen3-8B Q8_0 3.47 against 3.80 s warm and 22.59 against 23.07 s cold, six each; Qwen3-30B-A3B 7.78 against 7.45 s warm, four each; with `--n-cpu-moe 12`, 7.16 against 6.55 s warm, four each, and 48.18 against 47.27 s cold, eight each.
    - Against 007b504, Qwen3-8B Q8_0 cold takes 21.76 against 21.91 s over six pairs, with the same 2.12 million major faults, and Qwen3-30B-A3B with `--n-cpu-moe 12` cold 45.57 against 44.85 s over five.
    - Qwen3-0.6B Q8_0 warm on the CPU, timed inside the process over twenty interleaved pairs, takes 0.391 s against main's 0.411 s at the median.
    - Qwen3-30B-A3B cold on one MI50 is level on a quiet host: 45.1 to 52.3 s against 45.0 to 49.9 s over eight pairs, with the same 4.53 million major faults. While other work took host memory, single runs on either side took 177 to 496 s.
    - Under a 16 GiB container limit, six pairs of it took 144 to 390 s against 172 to 303 s, medians 190 and 256 s, each side faster in three. The warm and the major faults are the same on both sides, and the uploads are the same code, so the spread is the host's.
    - The `--n-cpu-moe 12` cold medians of 221.0 against 208.7 s at d48f2b2 came from host memory pressure: both sides read the payload in, and on a quieter host the case is level.
  - Open before the merge to main:
    - the safetensors branch's agreement to `TensorView`, `QwenWeights`, `AdoptWeight` and `format::FileSpan`, which steps 2 and 3 fixed without an answer;
    - an A/B of whole Qwen3-8B Q8_0 loads on the Windows CPU with nothing else running, since no load may be slower than main on either machine.
- **The `--load-mode` flag, for docs/USAGE.md.** The execution options of chat, generate, serve, logits, perplexity and `bench --model` take it. Help line: `--load-mode M           How weights are read: auto (default), mapped or direct`.
  - **`auto` (default):**
    - Weights a device copies are read from the file in large sequential reads, overlapped with the uploads.
    - The reads go through the operating system's file cache when the host has memory for the bytes the load reads, so a reload is served from the cache. When it does not, and the file system allows it, they go around the cache with direct I/O.
    - Weights the CPU reads in place stay memory-mapped and are read in after the uploads.
    - A model on the CPU alone loads as with `mapped`.
  - **`mapped`:**
    - Every file is memory-mapped, and its pages are read in before the model is placed when the host has room for the whole payload.
    - Devices copy from the mapping.
    - This is how models loaded before this option existed.
  - **`direct`:**
    - Every weight is read with direct I/O (`O_DIRECT`, or `FILE_FLAG_NO_BUFFERING`), bypassing the file cache. Nothing is mapped.
    - Weights a device copies are streamed to it.
    - Weights the CPU reads are read into memory llmx allocates, which the system cannot page back to the file.
    - Refused before any weight is read when a model file's file system does not take direct reads, or when the weights the CPU reads exceed the host's available memory. Examples of the first case: ZFS with `direct=disabled`, and Linux before 6.1, where the alignment cannot be queried.
  - **Errors and progress:**
    - An unknown value is refused before anything is opened: "unknown load mode 'x' (auto, mapped or direct)".
    - "Loading tensor data: N%" counts the bytes of the weights read. With a device in `auto` or `direct`, it reaches 100% after the last upload.
    - `--verbose` adds one line with the mode, the read method per file, and the time spent reading and uploading.
- **Not in this plan:**
  - **Direct reads for every load in `auto`.**
    - Direct reads fill no cache: ZFS skips the ARC on a miss, and ext4, xfs and NTFS skip the page cache. Every reload would then run at disk speed: about 4 s for the 8B on one MI50, against about 2 s warm today.
    - `direct` is one flag away.
    - Changing `auto`'s rule on Windows after its cold and warm A/B is a separate decision with its own gate.
  - **io_uring, and more parallelism.**
    - Container seccomp profiles block io_uring, `kernel.io_uring_disabled` can refuse it, and it adds nothing for a few large sequential reads.
    - A reader per file or per device, and parallel uploads to several devices, are also out. One pool limits the Linux host, and the upload link limits the Radeon VII.
  - **Direct reads into Vulkan staging.** Linux refuses to pin the driver's mapping, and Windows is unverified. Step 7 is the measured route.
  - **Sub-allocating weights from large device blocks.** It changes the device memory layout and the fit, so it gets its own branch.
  - **One submit for many small tensors.** The stage times decide this first.
  - **Cache behaviour after loading.**
    - Pruning the cache behind `auto` (`RWF_DONTCACHE`, `POSIX_FADV_DONTNEED`) is out, and so is an `auto` that checks the page cache with `mincore`.
    - `mapped` keeps touching device tensors before upload, so it stays the earlier loading.
  - **The fit.** The ring and `direct`'s host copy are not counted in it, and completing the fit is out: single-device and `--n-cpu-moe` loads, stream windows, padded F32 copies.
  - **Small items:**
    - cancelling a read in flight (a failed load waits for at most one 16 MiB read);
    - halving pieces on `ERROR_NOT_ENOUGH_QUOTA`;
    - `mlock`;
    - `MADV_DONTFORK` (nothing forks while loading);
    - tuning for disks that seek.
  - **Converting bytes at load, and row-range adoption for tensor groups and expert tiers.** An upload entry (a file span mapped to a buffer offset) is where either would go.
  - **Checking every tensor through a table of roles.** Plan, then fill already checks every role before any byte moves.
  - **The other follow-ups:**
    - type ids and sizes in `core/storage.hpp`, which the safetensors branch brings;
    - the shard filename rule, written in both `format/gguf.hpp` and `hub/manifest.hpp:shard_name`;
    - parsing the header from the mapping;
    - `PlacementRequest` fields a path ignores silently;
    - `set_ubatch` still public;
    - `split_check`'s own device spelling;
    - checking at load that a device supports every weight type.
- **Gates:**
  - **Suites, every step:** CTest and the Python suites on the Linux MI50s and on the Windows Radeon VII with the CPU. From step 5, in every load mode.
  - **Outputs, every step:**
    - Logits over the excerpt and 64 greedy tokens byte-identical to main, on Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M.
    - Devices: the CPU, one MI50, two MI50s split 1:1, the Radeon VII, and the Radeon VII with `--n-cpu-moe 12`.
    - `llmx-split-check` bit-identical.
    - `info`, `tokenize` and `detokenize` stdout and the `--verbose` plan diffed against main.
    - stderr without `--verbose` identical to main through step 4, and in `mapped` after it.
    - The `tests/chat.py` progress assertions pass in every mode.
  - **Host memory after load:** no more than main.
    - A device-only model releases its copy.
    - `--n-cpu-moe` keeps only the tensors a host reads.
    - `direct` holds exactly the host-read weights plus their aligned edges.
  - **Load time, from step 4 on:**
    - Cold and warm, at least three interleaved rounds, on both machines, every mode against main.
    - mx-llama.cpp's Vulkan build loads the same files on the same cards with `-lm dio` and `-lm mmap`, pinned to that card, and its load times go beside llmx's.
    - Cases: 8B Q8_0 on one MI50; the 8B split 1:1 over two MI50s; 30B-A3B on one MI50; the 8B on the Radeon VII; 30B-A3B on the Radeon VII with `--n-cpu-moe 12`; the 8B on the CPU of each machine; the 235B on six MI50s once, cold, against main's 370 s.
    - `auto` must be no slower than main in any case, and every result lists the stage times.
    - The two A/B arms are built from different commits in separate build trees.
  - **Getting a cold cache:**
    - Linux: a scratch ZFS dataset with `primarycache=metadata`. Where that cannot be created, the round is recorded as uncontrolled.
    - Windows: the standby list emptied with RAMMap before each cold round.
  - **Before step 6:** confirm that the Linux host's kernel is 6.1 or later, or `direct` is refused there.
  - **Step 6:** the CPU decode A/B between `mapped` and `direct` must be level, or better, on both machines.
- **Decided with the plan (the design was delegated):**
  1. `auto` reads through the cache when the host can hold the bytes the load reads, and direct otherwise, instead of direct wherever the file system allows. The reason is the "never worse than today" rule.
  2. The value is `mapped`, not `mmap`, since Windows maps with `MapViewOfFile`.
  3. With a device, in `auto` and `direct`, the bar counts bytes read and reaches 100% after the last upload. "Preparing model..." still follows it.
  4. `direct` refuses when the weights the host reads exceed available memory, rather than counting them in the fit.
  5. The reader refuses duplicate tensor names in a single file.

## Cleanup from the second code audit (planned 2026-09-25)

- **Goal:** a second read-only audit of the commands, KV cache, server, inference, hub and tokenizer, backends, and tests and tools found 72 findings that survived adversarial verification (5 were refuted). They are fixed under the one-owner rule in 18 branches, each off main and merged on its own gate. Model loading and the layer split pipeline stay out of it.
- **Every gate:** CTest and the Python suites on the Radeon VII and on an MI50. Where runtime code changes, outputs byte-identical to main on the pinned fixtures (0.6B Q8_0, Q4_0 and Q5_K_M, 8B Q8_0, 30B-A3B), on the CPU and on both cards, unless the branch names an intended change. Where a hot path changes, pp and tg level with main on both cards.
- **Correctness bugs first:**
  1. `fix/server-pause-prefill`: a request paused while still prefilling loses the unread part of its prompt, so the client gets text unrelated to it.
  2. `fix/server-http`: one `accept()` error stops the server; a streamed request whose pass fails gets a second HTTP response inside its body; request numbers are cast from double unchecked; the limit rules and the compatible-route decision are each split across files.
  3. `fix/kv-cache-default` (implemented, see its section): the model layer defaults caches to f32 while the CLI says f16, so the synthetic bench and `llmx-split-check` never run the default users run; `run_tests.py --cache-type` fails.
  4. `fix/server-prefix-fork`: the reservation ledger counts shared prefix blocks twice, so a follow-up turn evicts the donor it would fork; a fork copies a tail block only to truncate it.
  - `fix/server-cancel`, after branch 2 since both edit `http.hpp` and `api.hpp`: a request is cancelled only when a write to its client fails, so a client that leaves during a streamed request's prefill, while its request is queued, or at any point of a whole reply is not noticed, and the request runs to its end (an uncapped one up to the token limit), holding a slot and its blocks. The connection thread waits for tokens with a timeout of about 100 ms and, whenever nothing arrives, probes the socket without blocking and cancels the request once the client has closed it. Tests: a whole reply, a streamed long prompt during prefill and a queued request, each client leaving, and the server with nothing active and its blocks back within seconds.
- **Then the input and text bugs:** 5 `cleanup/cli-arguments` (numbers parsed four ways, garbage and negatives accepted, flags still ignored), 6 `cleanup/utf8` (UTF-8 written three times; the server can send invalid UTF-8 in JSON), 8 `fix/tokenizer-metadata` (a file with another tokenizer tokenizes silently wrong), 9 `fix/chat-template-defined` (`is defined` always true).
- **Then refactors that change no behaviour:** 7 `cleanup/quant-owner` (registry set up in 13 places, UTF-8 paths for quantize), 10 `cleanup/cli-help` (defaults and output the help misstates; merged 2026-09-26), 11 `refactor/cli-exec-options` (execution flags in the sampler header, sampling defaults twice; merged 2026-09-26), 12 `cleanup/test-helpers`, 13 `cleanup/vulkan-constants` (shader-fixed numbers in the tunable profile), 14 `cleanup/backend-contract-helpers` (row bytes and the row-run walker written several times), 15 `cleanup/kv-storage` (the CPU and Vulkan KV storages duplicate their bookkeeping and have drifted), 16 `cleanup/cpu-kernels` (dead runtime AVX2 checks, a duplicated decode dot), 17 `cleanup/hub-limits`.
- **Approved (2026-09-25):** 18 `test/reference-8b-per-token`. Since perplexity became batched by default, the 8B HF check scores only the batched path, so the 8B decode kernels have no reference check; the per-token mode is added, bounds unchanged, and the 0.6B check takes the 8B check's stricter validators.
- **Sequencing:** 3, 4, 5, 7 and 12 touch files the layer split branch still has open and start after it merges; branches sharing a file land in order (1, 2, 4 on `scheduler.hpp`; 5, 10, 11 on `main.cpp`; 7, 14 on the registry; 4, 15 on `kv_copy`; 14, 16 on `matmul_raw`).
- **Done (2026-09-25):** 8, 9 and 17 at `60f4799`; 1, 2 and 18 at `a2b732f`; 4 and then 3 at `80d53a1`, where 3 and 4 also gave greedy text and logits byte-identical to main for 0.6B, 8B and 30B-A3B on an MI50, and `llmx-split-check` bit-identical to one card at f16 and f32. Their gates: CTest and the suites on one card and on a split on the MI50s and on the Radeon VII; many users through the server on one MI50 and on a 3-card split (16 requests, together, then skewed with four clients leaving) matching their text alone and through the CLI; the 8B HF check 41 of 41 on one MI50.
- **Open, found while gating 1:** a request paused and resumed is not exact against the CLI. It resumes by forking its own full cache blocks and prefilling the rest of its history, so the tokens it generated in its last, partial block get their cache from the prompt path, not the decode path that first computed them. Whether a request is paused depends on the other requests, so its later tokens can depend on the batch, which the batch-invariance rule forbids. The server check compares paused requests' ids with the same request alone and passes because the greedy text agrees on the fixtures, not by construction. The fix makes a resumed cache identical to the one it replaces, either by keeping a paused request's blocks (moved to host memory) or by recomputing its generated tail through decode steps; it is planned after this cleanup.
- **Not acting on, with reasons recorded in the audit:** the KV growth peak in the fit (the loader plan's), one generation state for the CLI and server (a design change: they already share `sample` and `is_eos` and give the same ids), a float-scratch dot for routed Q4_0 and Q4_1 decode on the CPU (changes outputs), and helpers that would only carry differences as parameters.

## Cleanup from the code audit (2026-09-25, branches refactor/split-tight and cleanup/audit)

- **Goal:** every finding of a read-only audit of `src/`, `tests/`, `tools/` and the docs fixed under the one-owner rule (`docs/ARCHITECTURE.md`, Each concern has one owner): dead code removed, a concern implemented once where it belongs, docs matching the code. Behaviour and output unchanged unless a finding is a bug; each branch passes the suites on both machines and, where it touches kernels or placement, byte-identical outputs against main.
- **Layer split (refactor/split-tight):** the placement owns the ubatch and the rows it implies (one `kDefaultUbatch`); the kernels' scratch is a backend query (`Backend::scratch_reserve`), not a Vulkan figure in the device-neutral fit; expert streaming asks `reads_in_place`; one per-position cache size for the fit and `kv_used_bytes`, which counted f16 caches as floats, while the cache allocation takes the token budget and the types; `routed_layers` for the experts placement; `place_model` tests; every command opens its model through one `open_model`.
- **Commands and server:** one parser for the execution flags five commands copy; flags a command ignores are refused (`bench` without `--model`, `generate --system`, `-n` below 1); the chat template fallback and the start and end token lookup, the end-of-generation rule (`Tokenizer::is_eos`), and the sampling defaults each in one place below the CLI; the synthetic bench model and the quantize metadata writer out of `main.cpp`; the server reads the model's ubatch itself; unused scheduler members, config macros (`LLMX_DEFAULT_THREADS`, version parts, backends that do not exist) and includes removed.
- **Model, format, tokenizer:** unused `Model`, `BlockPool`, `Tokenizer` and `GGUFModel` members removed (`ModelFormat` stayed as the seam for safetensors, on the roadmap; the loader plan later removed it, since loading used another seam); `perplexity` scores a token through `token_nll` once; metadata is looked up through one `GGUFModel::find`, the reader refusing a file that repeats a key where five readers had walked the list and four checked for repeats on their own; the thinking filter that matches no Qwen3 token removed with the `--think` flag that only disabled it.
- **Backends:** the integer-dot feature struct chained into device creation only where the extension is present (a bug); Vulkan block sizes from `quant::Registry`; write-only `DeviceCaps` fields and the `Device` copies of `caps` removed; the range check, checked size arithmetic and KV block count shared in `backend.hpp`; `rms_norm` is `rms_norm_rows` over one row; the integer-dot tile call set up once; the unreachable split of more than three projections replaced by the throw it can never reach; push-constant fields no shader reads removed; `MeasuredProfile` renamed `TunedDevice`, the table of devices the profiles were measured on.
- **Tests and tools:** `chat.py` passes the configured device (it ran on the CPU under `--device`); one tiny-Qwen fixture for the placement, cache and prefill tests; the HF comparison, logits and NLL parsing and the f32-cache skip in `tests/common.py`; one server-starting helper for `server.py` and `long_context_check.py`; `make_test.py` and `verify_gguf.py` removed for the `roundtrip` component; the unbuilt `paged_attn_bench.cpp` moved to its evidence folder.
- **Docs:** SERVER.md (requests are paused, prefix reuse across block sizes is done, `pauses` in health), KV-CACHE.md and EXECUTION.md (f16 caches, block budgets, release on the last pass, fitting), `format-gguf.md` (files mapped in place), VULKAN.md (no validation option exists), README and ROADMAP (the layer split is implemented, the long-context check), the ARCHITECTURE table, the AGENTS test descriptions, a `device_profile` page, and no client named by its product.
- **Not done, on purpose:** one table of Qwen tensor roles (the resolver stays the one owner of shapes), one generated kernel table for the Vulkan kernel list, a shared `require` for the C++ tests, folding every file-hash helper in the tools, and one helper for the two float tile calls, which share two lines: each costs more than the duplication it removes. `pull --parallel` keeps two checks, the CLI's as it reads the flag and `pull`'s for its other callers, against one named limit (`hub::max_parallel_streams`), and `curl --version` runs once per byte range, at most 16 a file, where caching it would add shared state for no measurable time. The loader, spread over the format layer, the model constructor and `open_model`, gets its own plan before any code.

## Vulkan attention coverage checkpoint (2026-09-25)

The native attention oracle covers 80 head-width, head-ratio and K/V-type
combinations, including all implemented vector widths and a non-vector width.
Both rows of mixed short/long histories must match separate calls exactly.
The existing numerical bound is unchanged. The host-visible test now uses
its 777-byte fixture's size instead of two invalid 1000-byte requests.

A fresh Windows Release build on base `c602ace` passed all 24 native tests on
the Radeon VII, with no skips. Runtime source and build configuration are
unchanged. No new external HF or performance result is claimed. The
[evidence](benchmarks/attention-coverage-20260925/README.md) retains both the
initial run and final corrected run. The checkpoint review covered all 46
project Markdown files and 158 relative file targets: all ASCII, all targets
resolved, and changed claims match source and retained validation evidence.

## Long-context decode and the 16k check (2026-09-25)

- **Goal:** the pp16384 / tg512 case and the long-context greedy check asked for once decode reached its floor, on one card and on a layer split, beside the reference's Vulkan build on the same cards and file.
- **Done, 16k check** (`tools/long_context_check.py`, Qwen3-8B Q8_0, main 32914a9, rocm-smi GPU[2] and GPU[2]+GPU[3]): one card and the 1:1 split each give the same 512 tokens on two fresh servers, the same text as each other; the CPU reading the prompt and those tokens ranks 505 of them first, the largest gap 0.041 logits. Passed again with the vectorized decode attention below (508 of 512, largest gap 0.177).
- **Done, matched long-context bench:** `llmx bench --depth N` fills an N-token history outside the timer before every repeat, the protocol reference bench tools use for the same depth (`--seqs 1` had decoded from an empty history). Qwen3-8B Q8_0, two interleaved rounds, tok/s:

  | | llmx one MI50 | reference one MI50 | llmx split | reference split |
  |---|---:|---:|---:|---:|
  | pp16384 | 411.7, 448.7 | 354.9, 354.9 | 646.7 | 528.1, 528.0 |
  | tg512 from empty | 64.6, 64.7 | 58.0, 58.0 | 38.7 | 37.0, 36.5 |
  | pp512 @ d16384 | 261.2, 257.7 | 213.7, 213.6 | 251.8, 252.9 | 212.3, 212.1 |
  | tg @ d16384, main 7e195ff | 19.1, 19.3 | 44.5, 44.5 | 15.6, 15.6 | 29.4, 28.9 |
  | tg @ d16384, now | 30.4, 28.5 | | 21.4, 20.9 | |

  Both runtimes process a 16k prompt faster split than on one card; one card alone over a long prefill is likely held back by its power or thermal limit, not checked.
- **Done, decode attention on a long history, two steps** (`docs/VULKAN.md`, attention): a workgroup takes up to four query heads of one KV head once the history fills every split, so the history is read once for them, bit-identical to before (merged f8e0c5a); then heads 128 wide take `attention_vec.comp`, a token's row read in one load a lane and several tokens a subgroup (runtime merged acaa399; top-5 test policy changed in 435d610). One MI50, tg128 at a 16384-token history, main then grouped then vectorized: 0.6B 51.0, 64.4, 86.3; 8B Q8_0 21.2, 27.9, 29.4; 30B-A3B 21.0, 29.1, 31.4; level or faster at every shorter history on the MI50 and the Radeon VII. The vectorized kernel sits closer to the CPU than the one it replaces and resolves the MI50's reversed 8B near-tie (the pinned 8B HF check passes 37 of 37 on one MI50); it swapped the 0.6B Q8_0 fixture's 5th and 6th tokens for "The capital of France is", 0.024 apart on the CPU, and with the user's agreement both HF checks now count a swap at the 5th place when the reference puts both tokens within 0.1 of its 5th value (`common.top5_overlap`).
- **Measured and not taken, 16-bit activations on the MI50:** the integer-dot prefill tile over 16-bit activations fixed the 8B near-tie but cost prefill 17 to 62 percent (8B Q8_0 pp512 898 to 339), and the Q8_0 head on the 16-bit twin cost decode 3.5 to 7 percent; the reference's build reverses the same near-tie with 8-bit activations.
- **Measured, layer-split decode and the card clocks** (main da325cb, Qwen3-8B Q8_0, tg128, two rounds): one MI50 67.66, 67.50 at the automatic performance level and 67.58, 67.34 held high; split over two MI50s 38.98, 39.09 automatic and 66.59, 66.18 held high. The split's single-stream decode gap is the idle card lowering its clock between stages, not the handoff; with the clocks held high (`rocm-smi --setperflevel high`, which the operator sets; llmx changes no power settings, `docs/USAGE.md`) it meets phase 2's decode target already. pp512 is level too (891 split, 899 one card); prefill above one card needs prompts pipelined across the stages. Split comparisons against the reference hold both at the same clock level.
- **Left:** decode at a 16384-token history is 29.4 tok/s against the reference's 44.5 on one MI50; on the split 21 against 29. Phase 2 (`docs/MULTI-DEVICE.md`): prefill about one device's times the stages, and passes in flight for throughput at several users.

## CPU activation range checkpoint (2026-09-25)

Tiny finite CPU activation blocks now retain a representable scale and
nearest-even packed integers when the float reciprocal overflows. The ordinary
SIMD arithmetic is unchanged. The change is confined to CPU activation packing;
no backend interface, model dependency, runtime flag or GPU shader changes.
This integrates the reviewed repair onto main `11f5859`.

| Correctness check | Observed | Requirement |
|---|---:|---:|
| Current-main native tests | 21/21 | All |
| Required-HF CPU components | 14/14 | All |
| Real-model NLL, batched and per-token | 24/24 | Frozen HF bounds |
| Independent activation checks | 10,004 blocks / 320,128 values | Range, reconstruction, signs, zeros, sums and guards |
| Ordinary activation controls | 2,686,976 values unchanged | Exact packed bytes, scales and sums |
| Strict native-chat depth comparisons | 9/9 | Baseline/repeat/candidate exact through EOS |

[Integration evidence](benchmarks/cpu-activation-range-current-20260925/report.json)
retains the fresh native/HF logs and exact source identities. The
[strict chat gate](benchmarks/cpu-activation-range-current-20260925/strict-chat/README.md)
uses 19,820 prompt tokens on each of Q8, Q4 and Q5, with respectively 244, 302
and 229 generated tokens, zero reuse and no manual generation cap. Its frozen
older binaries and source-scope bridge remain explicit: this is scoped output
equality at depth, not independent HF numerical correctness there. The original
[raw-completion EOS gate](benchmarks/cpu-activation-range-perf-20260924/raw-completion/README.md)
remains incomplete after the baseline filled its context; no candidate result
is inferred from it. Gradual underflow is assumed; nonfinite inputs and
flush-to-zero behavior are outside the activation repair's claim.

The [final performance package](benchmarks/cpu-activation-final-perf-20260925/README.md)
retains all 128 common-harness and 72 normal-CLI measurements, balanced orders,
raw outputs and activity telemetry. All 15 model/executable/DLL hashes and
seven versions match after timing. The common comparison measures detached
candidate `356b745` against main `11f5859`, layout control `e149fc0` and mx
`5542318`, built alike on the same Windows machine. All eight paired candidate/mx
point estimates are positive; 8B decode is effectively tied (+0.126%, four of
eight pairs faster), so universal performance parity is not established.

| Normal CLI Q5 decode | Before | Candidate | Layout control |
|---|---:|---:|---:|
| Median throughput, tok/s | 77.38 | 71.24 | 73.23 |
| Paired throughput change vs before | Reference | -6.568% | -4.110% |

Accept this as a correctness repair with that measured cost. All six Q5 CLI
pairs are slower; candidate/control remains -2.563%, so layout does not explain
away the cost. The common harness uses a different prompt, microbatch and decode
history and cannot cancel the CLI result. Ordinary packed values, external HF
bounds and scoped depth equality support the repair; no cause for the slowdown
or performance neutrality is claimed. All other gains and losses and the prior
200 historical measurements remain available with their original scope.
Observed activity flags affect 126/128 common and 70/72 CLI invocations; all
retain unknown process CPU deltas. No competing own builds, tests or downloads
overlap timing. Telemetry covers whole invocations, not precise phase activity.

A [fresh CPU-only Release checkpoint build](benchmarks/cpu-activation-range-current-20260925/checkpoint/checkpoint-focused.json)
passes version/banner, F32 HF (maximum error 0.00000069 against 0.00002) and 56
thread-control cases (maximum NLL error 0.00000093 against 0.00001). Runtime,
tests and build configuration match the measured candidate; this later dirty
build identity is recorded separately, without a binary-performance equivalence
claim. All 45 project Markdown files received an affected-claim, ASCII and
relative-file-link review, including both analyzer summaries and the original
failed depth attempt. Historical measurements retain their original scope;
remote URLs and their anchors were not revalidated. The evidence archive
preserves all 1,004 raw files byte for byte in 5.66 MiB of compressed data.
CPU RowRuns validation and LDEV attention remain separate work.

## Vulkan cache cleanup checkpoint (2026-09-25)

- **Goal:** close the three reproduced Vulkan ownership failures without
  changing successful kernel arithmetic or adding queue waits.
- **Done:** failed kernel creation cleans local handles before retry and
  publishes only successful outputs; diagnostic query pools are destroyed
  after device idle; padded-cache invalidation reserves retention capacity
  before moving entries. Shader/layout/query failure outputs are deliberately
  poisoned in tests. Independent corrected source/test review found no blocker.
- **Done:** fresh Windows native 24/24 and required-HF Vulkan 14/14 passed,
  including 24 real-model NLL cells. All eight new regression cases pass;
  unchanged main fails seven. The existing seven queued-lifetime cases pass
  both arms. Local WSL GCC 13.3 builds the test and passes eight constructor
  cases; device checks skip on llvmpipe's subgroup width, not pass.
- **Done:** clean committed rebuild `c0d02cc0b28e`, with working Git metadata,
  passed all 24 native checks and focused version/F32/thread HF checks on Vulkan.
  The executable reports `llmx 0.1.0+gc0d02cc0b28e`; SHA-256 is recorded in
  `clean-commit.json` with commands and raw logs. Earlier pre-commit `unknown`
  binaries retain their original identity and evidence. Independent source,
  test and evidence review passed. Published main `11f5859` contains this
  runtime unchanged; all six hosted CI jobs passed in
  [run 36074805315](https://github.com/mxxm-t/llmx/actions/runs/36074805315).
  Hosted Vulkan coverage builds the backend; the local Radeon VII run
  supplies numerical coverage.
- **Gotchas:** CPU malformed RowRuns is a separate follow-up. No numerical
  shader, arithmetic, dispatch selection or successful queue-wait change.
  No performance claim is made while long-context correctness runs; suite
  timings are diagnostic. The first draft also passed native/HF checks before
  review required safe ownership of undefined failed-create outputs, so green
  happy-path tests alone did not establish that property.

| Check | Unchanged main | Candidate | Requirement |
|---|---:|---:|---:|
| New cleanup / failure / retry cases | 1/8 | 8/8 | 8/8 |
| Existing queued storage cases | 7/7 | 7/7 | 7/7 |
| Windows native suite | Prior 24/24 | 24/24 | All |
| Required-HF Vulkan components | Prior 14/14 | 14/14 | All |
| Real-model NLL cells | Prior 24/24 | 24/24 | Frozen HF bounds |

| Vulkan HF check | Maximum absolute error | HF bound |
|---|---:|---:|
| Synthetic F32 logits | 0.00000066 | 0.00002 |
| Synthetic MoE logits | 0.00000072 | 0.00002 |
| Q8_0 continuous NLL | 0.001224 | 0.01 |
| Q8_0 windowed NLL | 0.012346 | 0.02 |
| Q4_0 continuous NLL | 0.131524 | 0.16 |
| Q4_0 windowed NLL | 0.167600 | 0.2 |
| Q5_K_M continuous NLL | 0.026294 | 0.05 |
| Q5_K_M windowed NLL | 0.129520 | 0.16 |

[Evidence](benchmarks/vulkan-cache-cleanup-20260925/report.json) retains exact
sources, build commands, raw logs, controls and scope. Original audit artifacts
remain unchanged. Query tests use real diagnostic dispatches; transfer/kernel
failure tests intercept calls, including unsafe handles in negative controls.
The GCC warnings come from the test's deliberate global new/free injection;
no warning was suppressed. No Linux GPU or MI50 numerical claim is made.
All 40 Markdown files received affected-claim, ASCII and link review
(133 relative file targets, none missing), including updated test coverage
and the distinction between the historical audit and this fix.

## Backend architecture audit follow-up (2026-09-25)

- **Goal:** verify backend boundaries, ownership, error behavior and development
  rules without mixing changes into the active attention or activation work.
- **Done:** source review at published `e475c3f`, independent CPU/Vulkan review,
  and isolated failure probes. All 33 backend files pass the mechanical
  upward-include, ASCII and runtime-environment scans. The existing buffer,
  ticket, physical-KV and single-submitter boundaries fit the architecture.
- **Done:** three Vulkan findings reproduce: pipeline retry orphans one module,
  descriptor layout and pipeline layout; diagnostic query-pool teardown is
  absent; partial padded-cache invalidation leaves a null entry after allocation
  failure. CPU malformed RowRuns can execute beyond the declared batch before
  rejecting; model-generated malformed runs were not observed.
- **Follow-up:** the separate Vulkan cache cleanup block above implements the
  three Vulkan fixes and permanent regression checks. CPU malformed RowRuns
  remains open. The activation-range candidate's depth and performance gates
  remain separate; the dated audit itself made no runtime change.
- **Gotchas:** this is a source/failure-path audit, not a claim that every backend
  operation or shader is numerically validated. Earlier passing lifetime checks
  covered different paths. The three Vulkan probes use intercepted resource
  calls; CPU probes count callbacks without performing invalid memory accesses.

[Review and reproduction](benchmarks/backend-audit-20260925/README.md) records
findings, code locations, raw outputs and the exact reviewed revision. No
architecture rewrite or speculative abstraction is indicated.
All 40 Markdown files received affected-claim, ASCII and relative-link
review (132 file targets, none missing). Corrected historical buffer
signatures, implemented placement flags, the concrete CPU helper description
and the CPU/Vulkan precision comparison. No executable source changed, so
this documentation checkpoint relies on the unchanged main build and its
recorded 24 native checks; the new probes supplement those scoped results.

## Vulkan allocation lifetime checkpoint (2026-09-25)

Vulkan buffer construction releases acquired handles on failure. Failed KV
growth drains queued work before releasing new buffers and preserves the prior
backing and peak accounting for retry. Padded weight copies retain ownership
before recording; argument arena overflow retains queued storage before the
replacement allocation can fail. No successful-path wait is added.

This integrates reviewed `01010e1` onto main `da2b47a`. The Vulkan implementation,
regression source and CMake configuration match that reviewed feature; shaders,
attention kernels and CPU arithmetic are unchanged. The backend keeps ownership
and queue details below the model layer, with no new runtime knobs or dependencies.

| Check | Observed | Requirement |
|---|---:|---:|
| Constructor cases, including success/empty allocation | 8 pass | No leaked handles |
| Queued storage ownership cases | 7 pass | No premature release |
| Windows native suite, Radeon VII included | 24/24 | All tests |
| Required-HF Vulkan components | 14/14 | All components |
| Real-model NLL, batched and per-token | 24/24 | Frozen bounds |
| HF top-1 per model | 6/6 | 6/6 |

| Vulkan HF correctness | Maximum absolute error | Bound |
|---|---:|---:|
| Synthetic F32 logits | 0.00000066 | 0.00002 |
| Synthetic MoE logits | 0.00000072 | 0.00002 |
| Q8_0 continuous NLL | 0.001224 | 0.01 |
| Q8_0 windowed NLL | 0.012346 | 0.02 |
| Q4_0 continuous NLL | 0.131524 | 0.16 |
| Q4_0 windowed NLL | 0.167600 | 0.2 |
| Q5_K_M continuous NLL | 0.026294 | 0.05 |
| Q5_K_M windowed NLL | 0.129520 | 0.16 |

[Integration evidence](benchmarks/vulkan-allocation-lifetime-main-20260925/report.json)
records the fresh MSVC Release build, source/binary hashes and raw outputs.
The original failure controls remain in
[the feature evidence](benchmarks/vulkan-allocation-lifetime-20260924.json):
three of eight constructor cases and all seven queued-storage cases failed
before the fix. Those controls were not rerun here; their base Vulkan source
is identical on this integration's base. Intercepted transfer tests establish
ownership ordering; the actual-device kernel and HF checks supply numerical
coverage. This checkpoint makes no Linux/MI50 numerical, long-context or matched
performance claim; suite timings are diagnostic.

All 39 project Markdown files received affected-claim, ASCII and relative-link
review, with an independent source/architecture review. Documentation now names
the allocation-failure coverage and correctly says the change adds no successful
wait; existing ring reuse can still wait. Historical measurements, remote URLs
and anchors were not revalidated.

## CPU interface contract checkpoint (2026-09-25)

Published as `da2b47a` to both main remotes. Its clean committed EXE passed
backend-errors and focused version/F32/thread checks; GitHub run `36067923135`
passed all six jobs, including Linux Vulkan build and the required-HF CPU job.

Valid zero-byte CPU reads, writes and copies return after bounds/source
validation. `set_threads(0)` preserves the current worker pool, including
inside a prefill scope. Previously empty writes/copies could fail and a zero
thread hint reduced the pool to one worker. Grouped matmul now documents the
same asynchronous completion contract as the other backend operations.

This integrates reviewed `5c8e89c` onto main `2b67f62`. The executable diff is
four early returns; kernel arithmetic, CLI automatic thread selection and
negative thread-hint behavior are unchanged. The original test fails on the
old implementation; its dependency source is identical on this integration's
base. That archived negative executable was not rerun here.

| Check | Observed | Requirement |
|---|---:|---:|
| Valid empty/end transfers | 15 pass | No writes |
| Invalid ranges/sources | 13 refused | Reject |
| Zero thread hints | 3 pass | Preserve pool |
| Windows native suite, initial and forced final build | 21/21 each | All tests |
| Required-HF CPU components | 14/14 | All components |
| Real-model NLL, batched and per-token | 24/24 | Frozen bounds |
| HF top-1, each of three models | 6/6 | 6/6 |

| CPU HF correctness | Maximum absolute error | Bound |
|---|---:|---:|
| Synthetic F32 logits | 0.00000069 | 0.00002 |
| Synthetic MoE logits | 0.00000065 | 0.00002 |
| Q8_0 continuous NLL | 0.007366 | 0.01 |
| Q8_0 windowed NLL | 0.012356 | 0.02 |
| Q4_0 continuous NLL | 0.131554 | 0.16 |
| Q4_0 windowed NLL | 0.167600 | 0.2 |
| Q5_K_M continuous NLL | 0.027534 | 0.05 |
| Q5_K_M windowed NLL | 0.129440 | 0.16 |

The initial full suite and the forced final build are identified separately
in [the integration evidence](benchmarks/cpu-interface-main-20260925/report.json).
Two comment-only follow-ups corrected host-visible access wording and comment
indentation. MSBuild skipped those newer headers on an incremental invocation
under TEMP; that invocation is not counted as a fresh rebuild. A forced clean
build compiled final source, then all native tests and focused version/F32/thread
HF checks passed. AGENTS records the fresh-build rule for temporary MSVC trees.

All 39 Markdown files received affected-claim, ASCII and relative-link review,
with an independent interface/source review. Historical measurements, remote
URLs and anchors were not revalidated. The original focused gate remains in
`benchmarks/cpu-interface-contracts-20260924.json`. No new local Linux, macOS,
Vulkan numerical or matched performance claim belongs to this CPU integration;
standard suite timings are diagnostic during other correctness work.

## Model loading lifetime checkpoint (2026-09-25)

Failed construction and model teardown drain used backends before model-owned
buffers are released. A Vulkan adoption failure keeps its partial destination
alive until queued work retires; streamed-expert windows retain each successful
allocation before requesting another. Successful adoption remains asynchronous.
This integrates reviewed feature `83cca18` on main `7e195ff`, preserving the
perplexity thread fix and changing no attention shader or CPU arithmetic.

| Integration check | Observed | Requirement |
|---|---:|---:|
| Windows native suite, including Vulkan | 22/22 | All tests |
| Model validation, including failed loading | 380 checks | All checks |
| Required-HF CPU components | 14/14 | All components |
| Required-HF Vulkan components, Radeon VII | 14/14 | All components |
| Real-model NLL cases, batched and per-token | 48/48 | Frozen HF bounds |
| HF top-1 per model, both backends | 6/6 | 6/6 |

| Loading error witness | Original | Fixed |
|---|---:|---:|
| Missing final projection: buffers released while pending | 13 | 0 |
| Fourth adoption failure: buffers released while pending | 3 | 0 |
| Later streamed-window allocation failure | Original/intermediate regression fails | Pass |

The original and intermediate implementations fail independent asynchronous
ownership witnesses. Full CPU/Vulkan suites use the reviewed implementation;
a final rebuild after two comment-only corrections passes the model regression
and version check again. Those comments now describe physical microbatching and
activation slots without the stale one-sequence/nine-slot claims.

| HF correctness | CPU maximum absolute error | Vulkan maximum absolute error | Bound |
|---|---:|---:|---:|
| Synthetic F32 logits | 0.00000069 | 0.00000066 | 0.00002 |
| Synthetic MoE logits | 0.00000065 | 0.00000072 | 0.00002 |
| Q8_0 continuous NLL | 0.007366 | 0.001224 | 0.01 |
| Q8_0 windowed NLL | 0.012356 | 0.012346 | 0.02 |
| Q4_0 continuous NLL | 0.131554 | 0.131524 | 0.16 |
| Q4_0 windowed NLL | 0.167600 | 0.167600 | 0.2 |
| Q5_K_M continuous NLL | 0.027534 | 0.026294 | 0.05 |
| Q5_K_M windowed NLL | 0.129440 | 0.129520 | 0.16 |

[Current-main evidence](benchmarks/model-load-lifetime-main-20260925/report.json)
records commands, binary/source hashes and raw outputs. The earlier failure
controls remain in `benchmarks/model-load-lifetime-20260924.json`. This is a
lifetime/error-path change, with no matched performance claim; suite timings
are diagnostic while other correctness work runs. Linux CPU checks on descendant
`1f778af` cover this ownership implementation, not Linux Vulkan. Hosted CI on loader commit `2b67f62` passed all six jobs, including the Linux
Vulkan build, in run `36065492877`. That build job adds compile coverage; it
does not establish Linux Vulkan numerical or allocation-failure behavior.

All 39 Markdown files were reviewed for affected claims, ASCII and relative
file targets, with independent source/ownership review. Corrected grouped
integer-dot coverage, streamed uploads, host-visible model buffers, F16 CPU KV,
compiled Vulkan source and runtime layer-split configuration. Historical
measurements, remote URLs and anchors were not independently revalidated.

Separate Gitea-only work remains outside main: tiny-activation correction and
its evidence `e1060e7`, and native HF format integration `4c890c8`. Vulkan
allocation lifetime `01010e1` is integrated in the checkpoint above.
The separate activation long-context gate stopped on the Q8_0 base: a 19,812
token prompt plus 12,956 generated tokens filled the 32,768 context, returning
`length` rather than the required EOS after 6,093.875 seconds. No candidate
long comparison was reached. The [raw response and failure record](benchmarks/cpu-activation-range-perf-20260924/raw-completion/README.md)
are retained; this establishes an incomplete gate, not a candidate numerical result.

## Multi-device phase 1: layer split across devices (ROADMAP #5) (2026-09-24, branch feat/multi-device-phase1)

- **Goal:** phase 1 of `docs/MULTI-DEVICE.md`: a model split by layers over the devices `--device` lists, each device's share chosen by a fit against its free memory (`Backend::memory_available`), admission that counts every KV pool in its own block size, sharded GGUF mapped shard by shard so a model past host memory loads, and weights uploaded to the devices in parallel. Today's crossing (a read and a write) stays; pipelining is phase 2.
- **Done:** `Backend::memory_available` (CPU: `core/host_memory.hpp`; Vulkan: `VK_EXT_memory_budget`); `model/layer_split.hpp`, architecture-neutral, fitting consecutive layers per device from the architecture's `footprint` (`arch_qwen.hpp`, which alone knows the tensors) with the arena's slot widths shared with `ensure`; `--device A,B,...` and `--layer-shares` (proportions) on every command, a device listed once; `--verbose` prints each device's share. The placement test checks the fit and a fitted split over two CPU backends bit-identical to one. On the Radeon VII with the CPU at shares 1,1 the whole Python suite with the three HF fixtures passes; the single-device suite and the 22 native tests are unchanged.
- **Done, admission per pool:** the server reserves each request's blocks in every cache pool in that pool's own block size (`blocks_for`, `room_for`), donors and growth steps too, and the model refuses pools whose block sizes do not nest, since a shared prefix ends on a whole block of the largest. Counting everything in the largest block refused every request of a 16-token budget on a 64-token device pool beside a 128-token CPU pool; the server component caught it.
- **Done, on two MI50s (rocm-smi GPU[2] and GPU[3]):** one card against a 1:1 split is byte-identical in the top-10 logits of the last 8 positions of the 247-token excerpt and in 64 greedy tokens on Qwen3-0.6B Q8_0, Qwen3-8B Q8_0 and Qwen3-30B-A3B Q4_K_M. Qwen3-32B Q8_0, which no one card holds, fits as 32 layers each (16.2 GiB of weights and 5.0 GiB of cache per card at the default context): excerpt NLL 2.253, `bench` pp64 188, pp512 222, tg128 13.28 tok/s, against the llama.cpp Vulkan build's layer split on the same cards at 81, 205, 13.3 and its ROCm fork's at 195, 309, 18.8. Qwen3-30B-A3B Q8_0 as 24 layers each: NLL 2.427, pp64 430, pp512 1250, tg128 63.9. One request on a layer split pays the waiting card's clock (phase 0); throughput comes with passes in flight (phases 2 and 3).
- **Done, bounded shard loading:** every GGUF file is mapped, a sharded model as one segment per shard in one offset space, so no shard is copied into host memory; a metadata-only shard maps nothing, and a shard truncated before loading is refused before any progress. The shard test releases a model's mappings before rewriting its files, since Windows keeps a mapped file from being rewritten or removed.
- **Done, review of b86236a..f332209 (six findings):** the fit tries every first and last device, each carrying at least one layer and its endpoint weights, and keeps the plan with the fewest layers on host devices, so a host device returns when moving the embedding leaves a device short (the reviewer's reproduction is a test); a tied head on the embedding's device shares its buffer and the head's norm is counted; the server fits a prompt's ubatch plus `--max-seqs` rows; `Backend::resident_bytes` lets the Vulkan backend count its padded F32 copies; `--device` entries are canonical before de-duplication (`vulkan`, `vulkan:0`, `vulkan:00`); `memory_available` returns nothing when it cannot tell, and zero stays a full device. Docs: per-pool admission in MULTI-DEVICE, the multi-device STATUS row, and the contract that a mapped model's files stay unchanged while loaded.
- **Measured once, Qwen3-235B-A22B Q4_K_M on six MI50s (rocm-smi GPU[2] to GPU[7]):** the fit gives 15 or 16 layers each (20.5 to 23.5 GiB of weights per card); greedy text sane, tg32 18.96 tok/s (the llama.cpp Vulkan layer split on the same six reads tg128 16.3, its ROCm fork 29.1); loading took 744 s. Further runs use smaller models first.
- **Done, gate evidence at e9ee1aa on two MI50s (rocm-smi GPU[2] and GPU[3]):** the printed text of all 151,936 logits for each of the last 32 positions of the 247-token excerpt is identical between one card and a 1:1 split (equality at printed precision; the raw floats are compared below), and 64 greedy tokens match, on Qwen3-0.6B Q8_0 and Qwen3-8B Q8_0. Single device, main b86236a against e9ee1aa on GPU[2] alone, `bench --p 512 --n 128 --r 3`, two interleaved rounds (tok/s): 0.6B pp512 9394 / 9328 against 9391 / 9395, tg128 322.2 / 323.8 against 324.3 / 323.1; 8B pp512 897.5 / 899.8 against 896.7 / 893.0, tg128 66.78 / 66.49 against 66.54 / 66.39: level within the spread.
- **Done, loading:** a payload larger than the host's available memory is no longer touched page by page before the devices copy it, since those pages were evicted in between and read from disk twice: Qwen3-235B-A22B Q4_K_M loads and generates 16 tokens on six MI50s in 370 s against 744 s, the same greedy text. A diagnostic measurement, with the file system's cache state and the machine's other load not controlled, read the mapped file through page faults at about 400 MB/s against 2.1 to 2.4 GB/s from the same pool with large sequential reads (16 MiB calls, and direct I/O alike); it points at the loader rather than the disk without proving a cold fault ceiling, and the loader branch measures reads and uploads stage by stage. A model the host can hold is touched as before. Parallel uploads are not taken: a warm Qwen3-8B uploads to two cards in 1.6 s, and a cold load spends its time reading the file, not uploading it.
- **Done, re-review of 52bde20:** layers are assigned by a small dynamic program over their actual sizes for every first and last device, fewest host layers then the lightest busiest device, so uneven layers (400, 10, 10, 10 MiB on 800 and 500 MiB) fit and three equal devices take one of three equal layers each (both the reviewer's probes, now tests); only matrices a product reads are charged a padded F32 copy, not the embedding's gather or routed expert stacks; the head's logits rows count on the output device; MULTI-DEVICE's opening says each phase merges on its own row's gates.
- **Done, remaining gate runs:** the whole Python suite with the three HF fixtures required and CTest pass on two MI50s split 1:1 (tiny HF F32, MoE and shard models within 7e-7 of HF), as on the Radeon VII with the CPU. Radeon VII single device, main b86236a against this branch, bench pp512 and tg128 over interleaved rounds: 8B Q8_0 pp 338.1 / 339.2 against 338.6 / 339.2, tg 41.31 / 41.47 against 41.73 / 41.84; 0.6B Q8_0 decode reads either about 188 or about 196 tok/s from run to run in both binaries alike (rounds 5 to 10: branch 195.9, 195.7, 195.8, 195.8, 188.5, 188.7; main 195.9, 187.5, 195.8, 188.8, 195.7, 196.1), a machine state rather than a code difference.
- **Done, host fit (review of e3aa7ed):** the fit tries every set of devices that runs layers, each taking one layer at least, and charges the host exactly for that set: the logits rows, the RoPE vectors it keeps while the model lives, the residual stream staged between devices when more than one runs layers, and the host memory of the backends used, never of one left out. A device whose staging the host cannot hold is left out when the others can run the model, and a CPU stage counts the tables it reads in place once. The reviewer's probes are placement tests: two devices staging 68 MiB beside 64 MiB of tables use one device with 140 MiB of the host free and are refused with 100 MiB, and shares of 1,0 charge the used device only. The logits rows stay a bound (a scored text returns a row for every row of a pass).
- **Done, exact as raw floats (`llmx-split-check`, which compares every logit with `memcmp`):** on two MI50s (rocm-smi GPU[2] and GPU[3]) one card against a 1:1 split is bit-identical at every position of the 248-token excerpt, all 151,936 logits each, then in the prefill and 32 greedy steps, on Qwen3-0.6B Q8_0, Qwen3-8B Q8_0, Qwen3-8B Q4_K_M and Qwen3-30B-A3B Q4_K_M; the tiny HF fixtures (F32 tied and untied, MoE) are bit-identical over their 13 positions and two steps. The CPU against CPU+CPU is bit-identical on the same tiny fixtures and on Qwen3-0.6B Q8_0 and Qwen3-8B Q8_0.
- **Done, the two Q8_0 files no one card holds:** no HF fixture exists for Qwen3-32B Q8_0 or Qwen3-30B-A3B Q8_0, and HF in BF16 needs 61 to 65 GB of host memory, more than either machine has free, so they are held against llmx's CPU path, which the HF gate covers on the fixture models. On the excerpt, CPU against the two-MI50 split: 32B NLL 2.2501 against 2.2533 (delta 0.0032), the top token and top-5 set equal at all of the last 8 positions; 30B-A3B NLL 2.4200 against 2.4269 (delta 0.0069), the top token equal at all 8, the top-5 sets equal at 5 of 8 and differing by one near-tied candidate (within 0.1) at the other three. Both deltas are inside the Q8_0 fixture's NLL bound of 0.01 against HF; the rank-5 swaps are the CPU against the device, since the split is bit-identical to one card.
- **Done, gates at 335dee0 on the Radeon VII with CPU stages (shares 1,1):** the whole Python suite with the three HF fixtures required passes, and CTest 22 of 22.
- **Done, the mixed pass (`llmx-split-check` at 83f9a8e):** three passes of a sequence decoding beside a fresh prompt of up to 64 tokens, every row's logits, are bit-identical between one device and the split, as the prompt and decode paths are: on two MI50s for the three tiny fixtures, Qwen3-0.6B Q8_0, Qwen3-8B Q8_0, Qwen3-8B Q4_K_M and Qwen3-30B-A3B Q4_K_M, and on the CPU against CPU+CPU for the tiny fixtures, Qwen3-0.6B Q8_0 and Qwen3-8B Q8_0.
- **Done, HF on the split with a model whose reference fits a host:** the pinned Qwen3-8B Q8_0 float32 HF check (`tests/baseline_8b.py`, which now takes `--device` and `--layer-shares`) passes all 37 checks on the Radeon VII with the CPU at shares 1,1, the NLL within 0.0016 of HF on the continuous excerpt and 0.0022 at most windowed. On the two-MI50 split it passes 36: the four NLL cases (0.0035 continuous, 0.0045 at most windowed) and five of six rankings, and fails one top-1 at a near-tie, `The capital of France is Paris. The capital of Italy is Rome. The capital of`, where HF puts token 17689 at 24.364 over 9856 at 24.290 and the MI50 9856 at 24.662 over 17689 at 24.624 (the Radeon VII keeps HF's order, 24.512 over 24.491). One MI50 alone gives the same logits and the same failure with this branch and with main b86236a, so the split adds nothing to it.
- **Phase 1 gates, row by row:** exact against one device on CPU+CPU and on two identical cards (tiny HF model, 0.6B, 8B, 30B-A3B): the raw-float results above, the mixed pass included. HF bounds on CPU+Vulkan: the Radeon VII suite and its 8B HF check above. 30B-A3B Q8_0 and 32B Q8_0 fully on two MI50s: fitted, generated, NLL and bench above, held against the CPU path; the HF gates on the split run with the 8B and the tiny MoE, whose references a host here holds (`docs/MULTI-DEVICE.md`, phase 1 row). Radeon VII + CPU layer split: the same suite and check. No regression on any single-device path: level on the MI50 and the Radeon VII, above. The evidence, with commands, commits and binary hashes, is in `docs/benchmarks/multi-device-phase1-20260924.json`.
- **Left:** the reviewer's confirmation, then merge. After merge and before the loader branch: the single-card MI50 top-1 at the 8B near-tie, which the pinned 8B check first ran on a device here to show; the MI50 sits 0.30 above HF on both tokens where the Radeon VII sits 0.15 to 0.20 above, so which of its kernels differs is the first question.
- **Gotchas:** a tied embedding and head on different devices are uploaded to both; `--n-cpu-moe` with several devices is refused until expert tiers (phase 5b).

## Multi-device phase 0: measurements (ROADMAP #5) (2026-09-24, branch feat/multi-device-phase0)

- **Goal:** the numbers phase 0 of `docs/MULTI-DEVICE.md` asks for, before any split is written: the Vulkan handoff between two MI50s, P over S, S+1, S+2 and 2S, whether a host-relayed group sum and an expert exchange pay under Vulkan, and the baselines (llama.cpp Vulkan and ROCm on pinned cards, the vLLM gfx906 fork brought up and checked). No runtime change lands from this branch; its tools and records do.
- **Done, device queries** (`llmx-vk-handoff probe`, nine MI50s under RADV, Mesa 25.0.7): binary semaphores export and import as sync files; timeline semaphores only as opaque descriptors, and each card has its own device UUID, so a timeline does not cross cards and the host relay stays the baseline; host memory imports into every card (4096-byte alignment, a cached coherent host type); device memory exports and imports as dma-buf; no device groups, every card is a group of one. Every card links at Gen4 x8 to its root port; the root complexes hold cards 03, 44, four at 83 to 8c and four at c3 to cc.
- **Done, handoff** (`llmx-vk-handoff time`, 200 timed repeats, median): a submission of one command buffer costs 50 to 70 us from submit to host wake even when empty (4 us with no command buffer), each further command buffer about 20 us more, and a small copy inside a command buffer about 4 us. So a handoff's copy belongs in the stage's own submission, where what remains is the transfer. Card to card: 20 KB 113 to 165 us whichever way, since two submissions dominate; 10 MB 2.07 ms through imported host memory with a sync-file wait (two crossings at 10.9 GB/s each), 2.7 ms through today's read and write. A dma-buf import lands in host-visible memory on the importing card: 9.2 GB/s from a card on the same root complex with clocks held high (one crossing), 4.8 at automatic clocks, 1.1 GB/s across complexes.
- **Done, concurrency** (`llmx-multi-device-bench concurrent`): 2, 4 and 8 cards in one process, each on its own thread and backend, run their decode-shaped and 64-row passes within 1 percent of alone.
- **Done, pipeline** (`pipeline`, two stages of about 10 and 3 ms, host sampling 0.3 ms): with P = 2 on two stages the rate is 99 to 104 percent of the slower stage's back-to-back rate, and P of 3, 4 and 6 add latency only, so P = S while the host's time per pass is small against a stage. With one pass at a time the waiting card drops its clock: a pass through both 10 ms stages takes 29.4 ms at automatic clocks and 19.5 with clocks held high. A waiting stage that keeps its card busy with a slice of its up projection (`pipeline ... warm_rows`) recovers part of it, 30.0 to 26.1 ms with 256 rows at a time and 24.2 with 2048, at a constant load on an idle card; it is not adopted.
- **Done, group sum and exchange** (`groupsum`, `exchange`): a host-relayed sum of one decode row takes 227, 346 and 639 us on 2, 4 and 8 cards, 8 to 25 ms for 512 rows; two a layer is at least 29 ms per token on a 64-layer model, more than the decode it would split, so a Vulkan tensor group through the host is not worth writing.
- **Done, device-side chain** (`llmx-vk-handoff pingpong`, 200 hops, median of five chains): bytes bounced between two cards, each hop reading the other card's memory through dma-buf and writing its own. A 20 KB hop costs 30 us as back-to-back submissions on one card, 81 to 91 us with the host waiting on every hop, and 55 us with the whole chain queued ahead through sync files, on the same root complex and across complexes alike; 160 KB 72 us on one complex and 132 across. A tensor group of two summing this way pays about 7 ms a token on Qwen3-32B (two sums a layer, 64 layers) against about 30 ms of each card's half of the weights, and keeps both cards busy, so it is the candidate for one user on a model past one card: an estimated 26 tok/s against the Vulkan reference's 13.3 below. The fork's ROCm tensor split reads 33.6, which a Vulkan group at this hop cost does not reach; that needs ROCm's peer stores. Prompt sums of 512 rows want a pair on one root complex (about 1.1 ms a sum there, 9 ms across).
- **Baselines, Qwen3-32B Q8_0 on two MI50s (rocm-smi GPU[2] and GPU[3], one root complex), llama-bench -p 64,512 -n 128 -r 3, two interleaved rounds, tok/s:** the upstream Vulkan build (7ab4ee7ba) with a layer split pp64 79.9 to 81.0, pp512 198.4 to 204.7, tg128 13.24 to 13.32; the fork's ROCm image (eefc4e732) with a layer split 195.0 to 195.1, 307.4 to 309.5, 18.69 to 18.78; the same with its tensor split 373.5 to 373.6, 569.6 to 570.3, 33.57 to 33.66. The 235B download ran beside these (network and disk only).
- **Baselines, Qwen3-235B-A22B Q4_K_M** (132.4 GiB in five shards; llama-bench -p 64,512 -n 128 -r 3, one round, tok/s): the upstream Vulkan build's layer split on six MI50s (rocm-smi GPU[2] to GPU[7]) pp64 45.7, pp512 160.9, tg128 16.3; the fork's ROCm layer split on the same six 59.4, 174.7, 29.1; its tensor split on eight (GPU[0] to GPU[7]) 179.4, 459.7, 40.6 with a spread of 6.2 in decode. The tensor split runs at width 8 although the model has 4 KV heads.
- **Server baselines, Qwen3-32B Q8_0 on the two cards above** (`tools/server_load.py`, 128 greedy tokens a request on short prompts, 32 slots, best of two rounds; decoded tok/s, time to first token p50, inter-token p50): the fork's ROCm layer split 17.6 / 56.3 / 84.0 / 151.6 tok/s at 1 / 4 / 16 / 32 requests, TTFT 104 ms to 2.55 s, ITL 54 to 188 ms; its tensor split 32.5 / 86.1 / 124.3 / 213.3, TTFT 88 ms to 2.49 s, ITL 30 to 131 ms; the upstream Vulkan layer split 13.1 / 40.6 / 34.9 / 64.4, TTFT 106 ms to 4.81 s, ITL 76 to 463 ms. The vLLM gfx906 fork (image v0.12.0-rocm6.3, archived) runs Qwen3-32B-AWQ with tensor parallelism 2 on the same cards once its RCCL peer transport is disabled (`NCCL_P2P_DISABLE=1`; with it, `hipIpcGetMemHandle` fails with an invalid argument): greedy text is sane, mean NLL over the 247 targets of the fixed wikitext excerpt 2.366. Its own `vllm bench serve` (random 128 tokens in, 128 out, end of text ignored) gives 35.0 / 101.7 / 153.5 / 184.8 output tok/s at 1 / 4 / 16 / 32 concurrent, median TTFT 382 ms to 4.58 s, median ITL 26 to 133 ms; the same load tool as above gives 38.1 / 123.3 / 208.6 / 237.7 tok/s, TTFT 52 to 728 ms, ITL 26 to 130 ms. It is the strongest of the references at every level measured. The load tool sends short prompts and lets a request stop at its end of text; the comparison with vLLM's serving benchmark needs prompt-length distributions and fixed output lengths, which it gains in phase 3.
- **An expert exchange** took 0.85 to 4 ms a layer on 2 to 8 ranks at 8 tokens a rank (80 to 380 ms a pass of Qwen3-235B-A22B), and its floor of two host round trips a layer is over 40 ms a pass, against about 20 to 25 ms of decode: through the host, expert parallelism loses to the layer split. Chained on the devices through dma-buf and sync files the floor would be near 10 ms a pass (188 exchanges at about 55 us), which is not yet measured with more than two cards waiting on each other.
- **Measured, not adopted:** the backend's submission chunk of 64 dispatches, swept at 16, 24, 32, 48, 64, 128, 256 and 1024 on one MI50 over three interleaved rounds: nothing beats 64 outside noise (tg128 0.6B Q8_0 317 to 329 tok/s at 24 to 64), 16 costs Qwen3-30B-A3B 3 percent and 1024 costs it 17.
- **Left:** the device-side expert exchange on more than two ranks and vLLM on four cards, which open phase 6b; everything else phase 0 asked for is above and in `docs/MULTI-DEVICE.md`, Phase 0 results. Keeping a waiting stage's clock up in a single-stream layer split is a phase 2 question: llmx does not change a machine's power settings.
- **Gotchas:** multi-card runs need cards with no neighbour and link speed checked first; a llama.cpp reference run is pinned to its cards or it spreads over all ten. A clock held high must be restored by a trap.

## MoE decode's small kernels and the float tile's F32 rows (2026-09-24, branch feat/moe-decode-small-ops)

- **Goal:** the loose ends of MoE decode after the grouped decode merged: a token's time beyond its weight rows.
- **Found:** timed alone without timestamps (`llmx-moe-kernel-bench`, built with the Vulkan backend), Qwen3-30B-A3B's routed rows read at 580-680 GB/s on an MI50, while a token is about 600 dependent dispatches at a 4.1 us floor and its small kernels sat far above it: the RMS norm 11.4 us, the F32 router 13.4, the routing 10.9, the combine about 8. Each waited on its loads one at a time. Separately, F32 rows 2048 wide put every row of a tile step on one memory channel, and the router's 128 rows gave the float tile four workgroups: 360-400 us a layer at 32 to 512 prompt rows.
- **Done:** the small kernels load up front with their sums in the same order, and the norm's last tree steps run in one subgroup (logits over 64 positions and greedy text byte-identical to before on Qwen3-8B and 30B-A3B Q4_K_M); the float tile rotates F32 rows' inner steps by 128-row block and splits starved calls (`docs/VULKAN.md`, Kernel notes). On one MI50, base against this branch (tok/s): 30B-A3B tg128 120.0 / 129.0, pp32 357 / 410-412, pp128 628 / 663, pp512 1243 / 1248; 8B Q4_K_M tg128 93.1 / 96.4, pp512 871-873 / 874-876. The Radeon VII's Qwen3-0.6B prefill is level (Q4_0 pp512 +1 percent, pp64 within the spread).
- **Rejected:** two adjacent Q4_K or Q5_K rows a cluster (`research/k45-two-rows-rejected`): 8B +0.7 percent, 30B-A3B +0.2, 0.6B Q5_K_M -2.3.
- **Also rejected:** the combine folded into the down projection (`research/moe-combine-fold-rejected`): the last workgroup of a token's block of rows, found through an atomic counter, added the slots in the combine's order, exact but 1.6 percent slower single-stream and 4 percent at four sequences on the MI50, since every workgroup then ends on an atomic round trip while holding its slot.
- **MoE gate on main aead8e3, 2026-09-24:** one MI50, files one after another, llmx against the reference's Vulkan build, two interleaved rounds (mean ratio): Q4_0 pp64 182 percent, pp247 145, pp512 108, tg128 124; Q4_1 180, 175, 112, 119; Q4_K_M 128, 150, 109, 120; Q5_K_M 151, 159, 128, 115; Q6_K 156, 143, 126, 104. Q6_K decode, 99 percent before this branch, is ahead.
- **Left:** the dispatch count itself, about twelve a MoE layer, each at least 4.1 us on the MI50 and 7.6 on the Radeon VII; a fold through a completion count costs more than the dispatch it saves, so fewer dispatches needs kernels that do two steps' work without one. F32 weights 2048 wide now go through padded copies (branch `feat/f32-padded-rows`: 30B-A3B pp512 +1.8 percent on the MI50), but the router's activation rows are 2048 floats apart too, and at 512 prompt rows it still takes about 400 us a layer.

## MoE decode with many requests (2026-09-24, branch feat/moe-batched-decode)

- **Goal:** a server's MoE throughput growing with concurrent requests, well ahead of the reference's server at every level (the server gate).
- **Done:** a pass whose generated tokens carry at least two entries an expert groups them by expert and reads each expert's rows once per run of up to eight entries (`docs/VULKAN.md`, "Generated tokens beside each other"); `llmx bench --seqs N` measures decode passes of N sequences, which is what showed the plateau outside the server. Checked bit for bit against each token alone, the HF MoE gate and the suites on both machines. Server on one MI50, Qwen3-30B-A3B Q4_K_M, 64 tokens a request, main / this branch (tok/s): 115 / 115 at 1, 194 / 194 at 4, 217-218 / 218-222 at 8, 226-227 / 230-232 at 16, 225-226 / 262-263 at 32, time to first token at 32 from 848 to 606-614 ms; the reference's server gave 94, 149, 164, 110 and 203 on the same card earlier the same day. Plain decode is unchanged on both cards, since the grouped mode is a pipeline of its own (specialization constant 8 of the wide build): computed per column in every row kernel it had cost the MI50's integer-dot builds up to 7.5 percent, and tables filled up front the Radeon VII's Q8_0 build a fifth.
- **Tried and reverted:** grouping every batch of generated tokens, lone entries in the wide build (152 against 192 tok/s at four concurrent) and then split off to the one-column build (154): the loss was the grouping dispatch and its extra launches, not the build.
- **Measured after the merge, 2026-09-24:** `bench --profile` now reads the interval since its last reading, so it profiles a batched decode pass rather than a process's first 4096 dispatches. At eight sequences a pass the Q4_K gate and up rows of the routed entries take 40 percent of it and the Q6_K down rows 19, each entry about 7.9 us for 1.77 MB, 224 GB/s, where a dense Qwen3-8B decode reads its rows at about 460: an expert's matrix is 768 rows of 2048 values, so a dispatch is small whichever way it is cut. Capping a Q4_K or Q5_K row at 32 lanes in the integer-dot row kernels (`k45_row_lanes`) gave single-sequence MoE decode 120.6 to 123.2 tok/s and left eight a pass (249) and dense Qwen3-8B (94.0) where they were; 16 lanes cost the 8B 3 percent.
- **Left:** each routed entry is a small dispatch reading its rows at a quarter to half of what a dense decode reaches, so MoE throughput grows slowly with concurrency (1.23 times the reference's server at one request, 1.29 at 32); fusing an entry's gate, up and down into fewer, larger kernels is the next lever.

## CPU prompt rows on the prompt dots (2026-09-24, branch feat/cpu-dense-prefill)

- **Goal:** the CPU's dense prompt rows through the prompt dots that routed experts took on main (`q8_dots.hpp` `dot_block`), where that beats the batched float path, at or above the reference's CPU prefill.
- **Done:** K-quant rows (Q4_K, Q5_K, Q6_K) at least 4096 wide (`CpuBackend::kPromptDotsFrom`) meet a prompt through the prompt dots; narrower K-quant rows, Q8_0, Q4_0 and Q4_1 keep the float path, and generated tokens the decode dots, so a matrix takes one path per kind of row and a row computes the same however it is batched (`q8-dots` checks a prompt's rows beside a generated token against each alone, bit for bit, at 256, 2048 and 4096 wide). The full CPU suite and every real-model HF baseline pass on both machines.
- **Measured on the Linux machine's CPU (EPYC 7262, 16 threads, two interleaved rounds, tok/s, main / prompt dots / the reference's CPU path):** Qwen3-8B Q4_K_M pp64 21.9-22.1 / 30.7-31.2 / 38.4-39.1, pp247 33.4-33.7 / 32.2-32.3 / 39.4-40.3, pp512 19.9-20.0 / 31.4-31.8 / 40.7-40.8. Qwen3-0.6B Q5_K_M with the prompt dots on every width lost at 247 and 512 prompt tokens (415-431 against 378, 393-399 against 345-370), where its 1024-wide rows' dequantized blocks stay in the first-level cache; from 2048 wide it still lost 6 percent at 247, so the rule is 4096, which leaves the 0.6B files on main's path. The Q4_0 drop at 64 tokens seen in the first measurement did not repeat: main read 349 plus or minus 73 and 402 in two rounds, every arm within that.
- **Tried and reverted:** Q4_0 and Q4_1 rows on 8-bit activations, as the device takes them, with an output head's rows kept on 16 bits through `matmul_logits`. The HF gate's Q4_0 file then fails on the CPU ("The capital of France is": top-5 overlap 3 of 5 against a bound of 4). The two types stay on 16-bit activations.
- **Left:** the reference's CPU is still ahead on Qwen3-8B Q4_K_M (31-32 against 39-41 tok/s) and on Qwen3-0.6B Q4_0 (390-400 against 420-470); weights repacked into interleaved rows at load, so the prompt dots read several rows per load, are the next lever. The prompt dots' per-group horizontal sums follow from the activations' scale per 32 values and do not go away by blocking.
- **Gotchas:** the desktop's CPU timings wander with its other load; measure on the Linux machine, alone.

## Mixture of experts: qwen3moe on both backends (ROADMAP #2) (2026-09-23, branch feat/moe)

- **Goal:** Qwen3's mixture-of-experts form (`general.architecture = qwen3moe`, Qwen3-30B-A3B and Qwen3-Coder-30B-A3B) on the CPU and Vulkan backends, gated against HF, at or above llama.cpp's own Vulkan backend on the MI50 and the Radeon VII, and with experts on the CPU where the device is too small.
- **Done:** a layer is routed when its GGUF has `ffn_gate_inp`, so files that mix dense and routed layers load. Three backend ops carry a routed layer (`backend.hpp`): `route_experts` (softmax over the router scores, the top k, renormalized), `matmul_experts` (gate and up of each token's chosen experts) and `matmul_experts_add` (the down projection, weighted and summed in slot order into the residual). Expert ids and weights stay in the activation arena, so nothing leaves the device.
- **Done, CPU:** entries grouped by expert; a generated token's entries take the decode dots, all of a call's in one pool dispatch, and a prompt's entries a batched matmul per expert over its rows.
- **Done, CPU decode dots:** a decode row's activations are quantized once per call, 8-bit for Q8_0, Q4_K and Q5_K and 16-bit for Q4_0, Q4_1 and Q6_K, and meet the packed weights in integers (`backends/cpu/q8_dots.hpp`). The float dots converted every weight and were bound by arithmetic. On 8 bits for every type the HF gate's Q4_0 file, whose head is Q6_K, reached top-5 3/5 on "The capital of France is" against a bound of 4, as the device had found, so the three types the device reads on 16 bits read 16 bits here too; the gate then passes on all three fixtures. The CPU now picks a row's path by its runs as the device does, a generated token the decode dots and a prompt's rows the batched float path, dense and routed alike. `q8-dots` checks every type against a double-precision reference and a decode row alone, beside others and grouped, bit for bit. Qwen3-30B-A3B Q4_K_M on the 5800X, 8 threads: whole-model decode 8.0 to 16.9 tok/s (the reference's release build, whose CPU path this is, 11.3).
- **Done, Vulkan:** decode runs one entry per workgroup row through the row kernels, which take an expert offset on the weight rows; a prompt whose extent reaches its weight type's `moe_tile_from` (32 for Q8_0 and Q6_K, 48 for Q5_K, 64 for Q4_K, 96 for Q4_0 and Q4_1, measured on both cards; docs/VULKAN.md) takes the tile kernels over each expert's entries, grouped on the device by `moe_group.comp` (a workgroup per expert, stable order) and never split, so an entry computes the same whatever else is routed beside it. `moe_route.comp` routes through subgroup reductions and `moe_combine.comp` adds the weighted slots. The experts read the activation twin the router's input already has.
- **Done, placement:** `--n-cpu-moe N` and `--cpu-moe` put the experts of the first N (or all) routed layers on the CPU beside a device, through the per-role placement the model layer already had; attention, the dense blocks, the embedding and the head stay on the device.
- **Gates:** `tests/moe.py`, a tiny random-weight qwen3moe (two routed layers, one dense, 8 experts, top 3) against HF `Qwen3MoeForCausalLM` (`tools/gen_baseline.py moe`): all 257 logits within 7.5e-7 and windowed NLL within 1e-5 on the CPU, the Radeon VII and the MI50, across batch widths, threads and both placements. `tests/backend_vulkan.cpp` checks routing and the routed projections against the CPU for every supported type on the row kernel, the tile kernel and a batch mixing decode rows with a prompt. Qwen3-30B-A3B Q4_K_M gives the same greedy text on the CPU and the MI50.
- **Measured, Qwen3-30B-A3B Q4_K_M, one MI50 (idle), two interleaved rounds:**

  | test | llmx | llama.cpp Vulkan | share |
  |---|---:|---:|---:|
  | pp64 | 398 | 340 | 117% |
  | pp247 | 859 | 602 | 143% |
  | pp512 | 1165 | 1093 | 107% |
  | tg32 | 123.5 | 107.7 | 115% |
  | tg128 | 116.6 | 107.5 | 108% |

  Through the row kernels alone prefill was 383 against 1112; grouping by expert took it to 1048, and a grouping workgroup per expert, reused by the down projection, to 1214. Decode went from 99.3 to 116.6 with the twin reused past the router, subgroup routing (8 to 4 percent of decode time) and F32 router rows four values a load (8.8 to 4.0 percent).
- **Measured, experts on the CPU (b11075 on the Radeon VII, 8 threads; the reference on the same MI50, 16 threads):**

  | where | experts on CPU | test | llmx | llama.cpp | share |
  |---|---:|---|---:|---:|---:|
  | Radeon VII | 12 of 48 | pp512 | 196 | 121 | 162% |
  | Radeon VII | 12 of 48 | tg32 | 31.3 | 27.4 | 114% |
  | Radeon VII | 48 | pp512 | 75.6 | 76.7 | 99% |
  | Radeon VII | 48 | tg32 | 11.4 | 15.7 | 72% |
  | MI50 | 12 | pp512 | 223 | 404 | 55% |
  | MI50 | 12 | tg32 | 27.2 | 25.8 | 106% |
  | MI50 | 48 | pp512 | 69 | 166 | 42% |
  | MI50 | 48 | tg32 | 11.0 | 12.2 | 91% |

  12 routed layers on the CPU is what lets the rest fit the Radeon VII's 16 GB. With the decode dots over quantized activations (below), decode on the Radeon VII is 44.7 tok/s with 12 layers' experts on the CPU (163%) and 20.1 with all 48 (128%), and on the MI50 34.3 (133%) and 12.9 (106%); prefill is 195 and 76 on the Radeon VII, and 180 and 52 on the MI50. Some of the Radeon VII runs overlapped the model downloads and are to be repeated on a quiet machine.
- **Prefill with experts on the CPU, the MI50's gap:** the Linux machine is an EPYC 7262, 8 Zen 2 cores with eight memory channels, and the MI50 sits on PCIe 4.0 x16. The reference prefilled 151 tok/s at 512 rows with every expert on the CPU and 88.5 with `--no-op-offload 1`: from its CPU alone it is ahead of llmx's 52, and copying the CPU-held weights to the device for a large batch gains it another 1.7 times, which the Radeon VII's PCIe 3.0 link halves. Both are llmx's to close: a faster CPU path for a prompt's experts, then the same copy.
- **Done, mapped loading:** a single-file GGUF is mapped read-only instead of read into one heap allocation (`format/mapped_file.hpp`). Qwen3-30B-A3B Q8_0 is 32.5 GB: on the Radeon VII's 32 GB host with thirty layers' experts on the CPU the heap copy paged through every pass, and on the Linux machine's 62 GB host it sat beside its own page cache. The mapping lets the OS drop what the device copied. Sharded files keep the allocation that assembles them.
- **Done, device-held pages leave the host:** a model with experts on the host keeps the file mapped for them, and the pages of every tensor a device copied stayed in the process beside them: Qwen3-30B-A3B Q8_0 with thirty layers on the CPU held 21 to 22 GB on the Radeon VII's 32 GB host, which then read pages back from the file through every pass. Once the weights are resolved, the whole pages of each tensor only devices hold are given back (`MappedFile::drop`: `VirtualUnlock` on Windows, `madvise(MADV_DONTNEED)` elsewhere), and the process settles at 16.8 GB. Prefill at 512 rows went from 119.7 +- 33.5 to 129.6 +- 1.2 tok/s over eight runs, decode unchanged.
- **Done, prompt dots for experts on the CPU:** a prompt's routed entries met each expert through the float path, which converts every weight to a float for each block of four rows; they now meet it through quantized activations, each weight row unpacked 256 values at a time once for all of that expert's entries, integer sums per group of 32, eight groups' scales in one vector multiply-add (`q8_dots.hpp` `dot_block`). Generated tokens keep the fused decode dots, one kernel per kind of row, so a row computes the same alone or beside others (`q8-dots` checks prompt and decode entries beside each other against each alone, bit for bit). An earlier form that unpacked per four columns lost at 512 rows: each weight row swept every column's whole activation row, which no cache holds, and the block now walks the inner dimension so a column's 256 values stay in the first-level cache across eight rows. Qwen3-30B-A3B, experts on the CPU, streaming off (tok/s):

  | where | test | llama.cpp Vulkan | llmx before | llmx now |
  |---|---|---:|---:|---:|
  | MI50, Q8_0, 12 layers, 16 threads | pp64 | 54-57 | 105-109 | 166 |
  | MI50, Q8_0, 12 layers | pp247 | 147-148 | 180-186 | 267-270 |
  | MI50, Q8_0, 12 layers | pp512 | 287-293 | 230-235 | 314-316 |
  | MI50, Q8_0, 12 layers | tg16 | 20.0-20.3 (tg128) | 30.1-30.6 | 30.2-30.8 |
  | Radeon VII, Q4_K_M, 10 layers, 8 threads | pp64 | - | 82 | 123 |
  | Radeon VII, Q4_K_M, 10 layers | pp247 | - | 177-179 | 229-230 |
  | Radeon VII, Q4_K_M, 10 layers | pp512 | - | 230-240 | 271-272 |
  | Radeon VII, Q4_K_M, 10 layers | tg32 | - | 51 | 51 |

  The MI50's reference figures are from the three-way run on the same card, the llmx columns interleaved with each other. With this the CPU path beats the reference at 512 rows without copying experts, so `--moe-stream-from` now defaults to 0.
- **Done, streamed experts (`--moe-stream-from`, default 0 since the prompt dots; measured while it was 512):** from that prompt extent a host-placed routed layer runs on its attention device, the norm and router copied there at load and the experts copied into one window per device once per pass; decode rows and shorter prompts stay on the host, a mixed server pass split into groups of consecutive entries. The host upload now fills one half of staging while the device copies from the other. The copy is a fixed cost per pass, so the break-even differs by link: on the MI50 7.7 GB (twelve Q8_0 layers) takes 0.9 s (11 GB/s by DMA alone, under what the MI50's PCIe 4.0 x16 link allows), on the Radeon VII 19.2 GB (thirty layers) takes 3.3 s (6.4 GB/s by DMA alone). First keyed on extent like every kernel choice, which made a short follow-up in a conversation past the threshold pay the whole copy; it now follows the tokens the request prefills, its reused prefix excluded (`BatchEntry::fresh`), so every slice of a prompt still takes one path, and a reply on a cached prefix may take the CPU where one pass over the whole conversation would stream. On the MI50 with twelve Q8_0 layers on the CPU and the prompt dots, streaming gives 411 tok/s at 512 rows against 311 on the CPU and the reference's 296-299, and loses below that (223 against 268 at 247). Same greedy text as the host path on Qwen3-30B-A3B Q8_0; `tests/moe.py` adds streamed placements (every run, and prompts from extent 4) within 7.2e-7 of HF, and `tests/server.py` checks the synthetic MoE model's ids alone and four at a time with streamed prompt rows beside host decode rows.

  | where | test | host path | streamed |
  |---|---|---:|---:|
  | MI50, 12 layers on CPU | pp16 | 50.8 | 17.8 |
  | MI50, 12 layers on CPU | pp64 | 128 | 64.7 |
  | MI50, 12 layers on CPU | pp128 | 123 | 125 |
  | MI50, 12 layers on CPU | pp247 | 175 | 226 |
  | MI50, 12 layers on CPU | pp512 | 230 | 412 |
  | Radeon VII, 30 layers on CPU | pp128 | 92.7 | 37.1 |
  | Radeon VII, 30 layers on CPU | pp247 | 114 | 69.4 |
  | Radeon VII, 30 layers on CPU | pp512 | 118 | 132 |

  A 256 MB staging buffer instead of 64 MB changed nothing. What is left is the copy overlapping the previous layer's compute, which needs a second window and a transfer queue (at pp512 on the MI50 the compute is about 0.35 s of the pass's 1.25 s).
- **Measured, Qwen3-30B-A3B Q8_0 with experts on the CPU (one MI50 with 12 layers and 16 threads, the reference pinned to the same card; Radeon VII with 30 layers and 8 threads against b11075; two interleaved rounds each):**

  | where | test | llmx | llama.cpp Vulkan | share |
  |---|---|---:|---:|---:|
  | MI50 | pp64 | 102-106 | 56-58 | 180% |
  | MI50 | pp247 | 188-212 | 152-153 | 123-139% |
  | MI50 | pp512, host path | 229-241 | 291-305 | 75-83% |
  | MI50 | pp512, streamed | 412 | 291-305 | 135-141% |
  | MI50 | tg128 | 28.3-28.7 | 20.6-20.8 | 138% |
  | Radeon VII | pp64 | 56-57 | 6.1-7.6 | 740% |
  | Radeon VII | pp247 | 93-107 | 49-50 | 188-216% |
  | Radeon VII | pp512 | 117-120 | 90-91 | 129-132% |
  | Radeon VII | tg128 | 18.5-18.9 | 14.2 | 131% |

  The Radeon VII's llmx prefill still varies by up to 35 tok/s between repeats, the host holding 19 GB of experts in 32 GB; the MI50 streamed row is a single round.
- **Done, decode on the 8-bit twin:** the first MI50 gate run had Q6_K, Q4_0 and Q4_1 decode at 87 to 93 percent, all three rows reading the 16-bit twin because the HF gate's Q4_0 fixture had failed on 8 bits. That failure was the Q6_K output head: `Backend::matmul_logits` names the head, which keeps the 16-bit twin for those types, and every other Q6_K, Q4_0 and Q4_1 row reads the 8-bit twin (Q6_K folding its -32 into each weight byte). Decode at tg128 went from 82.0 to 92.4 tok/s on Q6_K (with at most 32 lanes a row, `q6k_row_lanes`), from 101.8 to 131.8 on Q4_0 and from 101.4 to 128.8 on Q4_1; the HF gate passes on all three fixtures. A second gate run with these is going.
- **Done, server:** an uncapped request reserved its whole reach, the whole pool, so Open WebUI's chats and background requests ran one at a time and each admission dropped the cached prefixes. It now reserves its prompt and a step and grows, the latest admitted uncapped request pausing (history kept as a donor) when the pool runs out; `tests/server.py` checks three uncapped requests sharing a small pool with at least one pause. The compatible replies carry a `timings` object, which Open WebUI shows, and each request logs a line.
- **Tried and reverted, 2026-09-23:** the integer-dot tile reading the 8-bit twin a producer wrote, in position order, instead of quantizing its own block-major copy. The values are the same, but at 4096 wide a column's blocks are 4 KB apart and the loads stop coalescing: Qwen3-8B Q4_K_M prefill on the MI50 fell from 693 to 368 tok/s at 64 rows and from 845 to 775 at 512, with no gain on the 0.6B files.
- **Tried and reverted, 2026-09-23:** a decode token's down projection summing its eight slots into the residual inside the row kernel, one token per workgroup row, instead of writing the slots and adding them with `moe_combine`. Same arithmetic, but a workgroup per token running its slots in turn left 256 workgroups where there had been 2048, and MI50 decode fell from 92.8 to 90.7 tok/s on Q6_K and from 118.7 to 118.0 on Q4_K_M (the reference 93.2 and 107.6 in the same interleaved runs). Q6_K decode stays at 99.5 percent of the reference; the F32 router rows already take a whole subgroup each.
- **Tried and reverted:** a prompt's routed entries through the decode dots as well, a dot per weight row and entry. Each dot unpacks the row's nibbles and scales again, where the batched float path unpacks a row once for all its expert's entries, and prefill with every expert on the CPU went from 52 to 35 tok/s on the Linux machine (during a download). A prompt's entries want a multi-column kernel that unpacks a row once.
- **Left:** the prompt dots for the CPU's dense prompt rows, which still take the float path; the copy overlapping compute;  and the dense CPU cells before and after the decode dots are to be measured; a real file of every supported type (Q5_K_M, Q6_K, Q8_0, Q4_0, Q4_1 downloading on both machines) through the gate cells; CPU expert decode (the fused dots against float activations) and prefill with experts on the CPU, where the reference likely runs large batches on the device from host-held weights; a server check of routed layers on a real model.
- **16k check, redefined 2026-09-24:** a hash across the CPU and the device parted at the first near-tie: on Qwen3-0.6B-Q8_0 they agreed for 68 characters, where the CPU's top two logits were 18.498 and 18.379 and the device's 18.383 and 18.346, the two tokens swapped, the device's logits sitting up to 0.23 from the CPU's after the 16k prompt as the 8-bit activations shift them. With the user's agreement `tools/long_context_check.py` now requires the device to give the same 512 greedy tokens after a 16384-token prompt on two runs from fresh servers, and the CPU, reading the prompt and those tokens (`llmx logits --last`), to rank each within 0.5 logits of its top choice. Passed on every run: Qwen3-0.6B-Q8_0 on the MI50 (the CPU's top choice at 509 of 512 tokens, the largest gap 0.135) and on the Radeon VII (512 of 512, 0.000), Qwen3-8B Q4_K_M on the MI50 (510 of 512, 0.014).

## Main documentation checkpoint (2026-09-24)

Corrected fixture counts, ticket/completion descriptions, vendor quant support,
device selection, Q8 overflow scope and historical GPU status against main.
Recorded the user's independent-feature branch rule in AGENTS. All 35 Markdown
files are ASCII, 118 local links/anchors resolve and `git diff --check` passes.
No source, tests, build configuration or historical measurement tables changed.
Main remains dense Qwen3 with capped server requests; unmerged feature claims
were excluded. This documentation change is separate from the help fix below.

The parent help commit `5afc1c7` passed all six hosted jobs in
[CI run 35964866124](https://github.com/mxxm-t/llmx/actions/runs/35964866124):
Windows, Ubuntu, macOS, Linux UBSan, Vulkan build and required real-model HF.

## Perplexity batch-thread checkpoint (2026-09-25)

Batched perplexity applies `--threads-batch` / `-tb`; omitted or zero values
follow `--threads`, and `--per-token` uses the decode count. `--verbose`
reports the selected phase and actual count on stderr. This CLI fix changes
no backend arithmetic or inference API and has no unmerged feature dependency.

| Check | Windows MSVC | Linux GCC 13.3 | Requirement |
|---|---:|---:|---:|
| Effective counts, aliases, modes and HF NLL | 56 pass | 56 pass | All cases |
| Maximum absolute HF NLL error | 0.00000093 | 0.00000093 | 0.00001 |
| Native suite | 21/21 | 20/20 | All platform tests |
| Full required-HF CPU components | 14/14 | Focused checks only | 14/14 when run |
| Restored ignored-batch bug | Rejected | Not run | Batch 1 must not select 16 |

The negative control retains verbose support and removes only effective batch
selection. Linux uses a hash-verified source snapshot without Git metadata,
so its build identifier is `+unknown`. There is no new Vulkan or matched
performance claim; standard smoke timing is diagnostic during other
correctness work. Evidence: `docs/benchmarks/perplexity-batch-threads-20260924.json`;
raw logs: `TEMP/llmx-perplexity-batch-threads-evidence-20260924/`.

All 39 Markdown files were reviewed for affected claims, ASCII and 123 relative
file targets. Usage, CLI and test coverage were updated, and touched CLI
comments were swept. Unrelated architecture corrections remain in separate
checkpoint 1f778af; historical numbers and remote links were not revalidated.

## Command help checkpoint (2026-09-24)

`llmx --help` prints a grouped overview; each of the 12 commands accepts
`--help` or `-h` for its own options, defaults and example. Execution options
share one renderer. Help returns before model, backend or Hub access. Existing
command parsing and positional text behavior are unchanged. The help and
reference now describe seed zero as retaining the fixed default RNG state.

This change is based directly on `cdf1cdb`, independently of in-flight features.
The main-based Windows Release build passes:

| Check | Observed | Required |
|---|---:|---:|
| Help routes | 26 passed | 26 |
| Missing or unsupported advertised parser flags/aliases | 0 | 0 |
| Exit/positional-text checks | 5 passed | 5 |
| Native tests | 20 passed | 20 |
| Python components with all required real-model fixtures | 13 passed | 13 |
| Real-model HF top-1, each of Q8_0/Q4_0/Q5_K_M | 6/6 | 6/6 |

| HF NLL comparison, maximum over batched and per-token paths | Observed absolute error | Existing bound |
|---|---:|---:|
| Q8_0 continuous | 0.001284 | 0.01 |
| Q8_0 windowed | 0.012376 | 0.02 |
| Q4_0 continuous | 0.131554 | 0.16 |
| Q4_0 windowed | 0.167600 | 0.20 |
| Q5_K_M continuous | 0.026144 | 0.05 |
| Q5_K_M windowed | 0.129480 | 0.16 |

Reproduction: CMake Release build, `ctest --test-dir build -C Release
--output-on-failure`, then `python -u -X utf8 tests/run_tests.py --exe
build/Release/llmx.exe --no-perf-floor --require-baseline`. The standalone help
check also compares each command's emitted options to its parser, rejecting
both missing and unsupported flags. Raw logs and help output remain under
`%TEMP%/llmx-help-main-20260924/build/`. Performance timings are diagnostic only;
this change makes no new performance claim.

All 35 tracked Markdown files were reviewed against main: ASCII and 116 local
links/anchors checked. Help-related claims were corrected here. Independent
findings were subsequently addressed by the separate documentation checkpoint
above; they were not included in the help implementation commit.

## Multi-user server (ROADMAP #7, EXECUTION step 7) (2026-09-22)

- **Goal:** the HTTP front-end over the model layer the execution plan
  built for it: one shared model, a `Sequence` per request, continuous
  batching with chunked prefill through one `Model::forward` per scheduler
  iteration, streaming responses, prefix reuse through `fork`, admission by
  the KV pool's budget. Dependency-free transport. The design, protocol,
  scheduler loop, gates and order of work are `docs/SERVER.md`.
- **Done:** the design and steps 1 to 6 (3a and 4 to 6 further down).
  `src/server/http.hpp`: HTTP/1.1 over blocking sockets, Winsock or BSD, a listener, one request with a
  Content-Length body per connection, a whole response or a chunked
  stream, a client for tests; the `http` CTest covers a whole response, an
  echoed body, a three-chunk stream arriving as written, 413, 400, 404 and
  the listener closed from another thread, on Windows and on the Linux
  machine. `src/server/scheduler.hpp`: the loop of SERVER.md, admission by
  the pool's capacity with every admitted request's blocks reserved up
  front (admitting on blocks merely free let four requests into a
  one-block pool), decode entries then chunked prompt slices in one
  `forward`, per-request seeded sampling, channels, cancellation, a
  failed pass ending its requests and not the loop. `src/server/api.hpp`
  and `llmx serve`: `/v1/generate`, `/v1/chat` through the template
  renderer, `/v1/health`, `/v1/models`, streamed as server-sent events
  with characters held until complete and any bytes that never form a
  character replaced by U+FFFD, since a byte-level vocabulary under
  sampling produces them. The `server` Python component runs on the
  synthetic F32 model without a download and on the Q8_0 fixture:
  greedy through the server equals `generate --temp 0` alone and four at
  a time, a stream carries the same ids, a seeded request repeats,
  refusals, a client leaving mid-stream leaves nothing active, a chat
  turn; it passes on the CPU and on the device. `tools/server_load.py`
  measures aggregate decode throughput at N concurrent requests against
  llmx's route or the reference's `/completion`. On the device, Qwen3-0.6B
  Q8_0, 64 tokens per request, best of two rounds, both servers in the
  same minutes (the reference with 16 slots over an 8192 context):

  | concurrency | reference server | `llmx serve` | llmx share |
  |---|---:|---:|---:|
  | 1 | 176.9 tok/s | 210.6 | 119% |
  | 4 | 367.6 tok/s | 328.9 | 89% |
  | 8 | 453.3 tok/s | 369.1 | 81% |
  | 16 | 237.1 tok/s | 244.9 | 103% |

  Per step llmx cost about 4 ms more per extra sequence, which was the
  device's multi-view path: `kv_write` and `attention` dispatched once
  per view, and `norm_rope_kv` fused only a single view and fell back to
  its three-op default for a batch, so a sixteen-sequence step was on the
  order of a thousand dispatches. SERVER.md step 3a closed it: every
  cache kernel now takes a view table (`shaders/views.glsl`: per view its
  batch row, dispatch-local row, row count, history, block-table offset
  and length, then every view's block ids, uploaded through the args
  arena) and runs once per layer over every view; attention splits a
  batch into the views the tiled kernel takes (32 rows or more, 128-wide
  heads) and the rest for the per-row kernel, at most two dispatches, so
  a decode row never sits in a tile staging its whole history for one
  live row; the merge kernel reads the same table since a dispatch's rows
  need not be a prefix of the batch. On the way a latent hazard surfaced:
  a split or subgroup that saw no token held -inf and its state came out
  as exp(-inf - -inf); both merges now skip such a part, which the
  kv-cache checks over two views had not reached before. Every kernel
  check, the HF gate on the device and the server component pass; the
  same load again, both servers in the same minutes:

  | concurrency | reference server | `llmx serve` before | `llmx serve` now | llmx share |
  |---|---:|---:|---:|---:|
  | 1 | 177.5 tok/s | 210.6 | 209.0 | 118% |
  | 4 | 365.6 tok/s | 328.9 | 409.2 | 112% |
  | 8 | 462.6 tok/s | 369.1 | 504.3 | 109% |
  | 16 | 237.1 tok/s | 244.9 | 297.1 | 125% |

  The throughput gate is met at every level. Both servers fall at 16:
  for llmx the row kernel holds eight columns per dispatch, so sixteen
  sequences stream the weights twice per matmul, which is recorded, not
  fixed.
  SERVER.md step 4, prefix reuse: a finished request's history stays as a
  donor (at most `max_seqs` of them, the oldest dropped when a request
  needs its blocks) and a new prompt forks the donor sharing the longest
  run of full blocks, rolled back to those blocks through the new
  `Model::truncate`, so only the rest of the prompt is prefilled. Tokens
  are compared, not hashed: a server holds a handful of donors for one
  model, and the hashed key of KV-CACHE is for an index that outlives a
  process. The `server` component sends the 247-token excerpt with two
  endings and requires the second to reuse blocks and give the CLI's
  greedy text; it passes on the CPU and on the device. `/v1/health`
  reports `donors`, `prefix_hits` and `prefix_tokens`, a reply
  `reused_tokens`. Whole-request wall time at the client, 16 generated
  tokens, one request at a time:

  | model | backend | prompt | first request | with a donor | reused |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | Vulkan | 1995 tokens | 1.53 s | 0.16 s | 1984 |
  | Qwen3-8B-Q8_0 | Vulkan | 1995 tokens | 8.74 s | 0.72 s | 1984 |
  | Qwen3-0.6B-Q8_0 | CPU | 1995 tokens | 6.95 s | 0.95 s | 1920 |

  The reuse is bounded by the block size (64 on the device, 128 on the
  CPU) and by the last prompt token, which is always prefilled for its
  logits.
  SERVER.md step 5, the second execution context, measured and not added.
  A timing build split every scheduler pass on the device into the
  `forward` call (recording and submitting), the wait for its logits and
  the host's work until the next `forward`, Qwen3-0.6B-Q8_0, 64 tokens
  per request, 200 passes each:

  | sequences | forward call | wait | host between passes | per pass |
  |---:|---:|---:|---:|---:|
  | 1 | 1.6 ms | 3.4 ms | 0.02 ms | 5.0 ms |
  | 4 | 2.9 ms | 6.6 ms | 0.02 ms | 9.5 ms |
  | 8 | 4.5 ms | 10.7 ms | 0.03 ms | 15.2 ms |
  | 16 | 18.7 ms | 30.9 ms | 0.03 ms | 49.6 ms |

  A second context could hide only the last column, and there is nothing
  there to hide: greedy sampling over the vocabulary is tens of
  microseconds. The same build with one submission per pass instead of
  one every 64 dispatches puts pure recording at 0.7 ms per pass at one
  sequence and 1.7 at eight, the rest of the `forward` column being the
  chunked queue submissions, and costs 2 percent of `bench` decode (218
  to 213 tok/s), so the chunking stays. The recording cannot overlap the
  device either way, since the next pass's tokens come from this one; a
  recorded pass replayed with the next tokens, positions and view table
  written into the arena would recover it, at most 15 percent of a
  0.6B decode pass and about 3 percent of an 8B one. Recorded as an open
  backend lever, not built: it is a small-model gain and the device is
  already past the reference on those.
  SERVER.md step 6, the compatible routes the user asked for so that
  the tools people already run connect to `llmx serve` unchanged:
  `/v1/chat/completions`, `/v1/completions` and `/v1/models` in the
  shape the OpenAI clients speak, the shape vLLM and the reference's
  server expose too. They are a JSON mapping in `src/server/api.hpp`
  over the same scheduler: one parse, one request, one drain loop, the
  native routes' knobs accepted as extra fields and the clients'
  synonyms beside them, everything validated before the model, errors
  in the clients' `{"error": {"message", "type"}}`. The `server`
  component checks greedy equality with the CLI through
  `/v1/completions` whole and streamed, the usage counts, the role in
  the first chat chunk and the finish reason in the last, text content
  parts and the refusals; it passes on the CPU and on the device.
  The serving gate, restated by the user: the reference's server is the
  weaker of the serving runtimes at concurrency and the one that runs on
  this hardware, so llmx must beat it by a wide margin on the figures a
  serving runtime is judged by, not reach parity. `tools/server_load.py`
  now streams every request and reports time to first token and
  inter-token latency at the median and the 99th percentile, tokens per
  second and requests per second. Qwen3-0.6B-Q8_0 on the device, 64
  tokens per request, the reference with 32 slots over an 8192 context,
  llmx with 32 sequences, both servers in the same minutes, best of two
  rounds, the two passes' spread shown where it matters:

  | concurrent | tok/s reference | tok/s llmx | TTFT p50 reference | TTFT p50 llmx | ITL p50 reference | ITL p50 llmx | ITL p99 reference | ITL p99 llmx |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 1 | 170, 173 | 187, 187 | 21, 29 ms | 26, 26 ms | 5.5 ms | 5.0 ms | 6.5, 6.6 ms | 5.5, 5.6 ms |
  | 4 | 357, 361 | 401, 406 | 105, 113 ms | 61, 64 ms | 9.6 ms | 9.0, 9.1 ms | 11.1, 11.2 ms | 9.7 ms |
  | 8 | 455, 470 | 509, 511 | 106, 150 ms | 61, 62 ms | 15.4, 15.6 ms | 15.0 ms | 17.1, 18.3 ms | 15.8, 15.9 ms |
  | 16 | 231, 232 | 546, 555 | 325, 353 ms | 78, 98 ms | 64.6, 64.7 ms | 27.9, 28.0 ms | 69.0, 74.4 ms | 29.7, 33.2 ms |
  | 32 | 544, 548 | 560, 563 | 469, 525 ms | 109, 119 ms | 51.1, 51.3 ms | 56.0, 56.1 ms | 58.5, 65.2 ms | 57.3, 58.0 ms |

  This table is with the tile threshold, the tool's warm-up and the
  chosen tile shape all in place, both servers interleaved in the same
  minutes, two passes each. Throughput now leads at every level
  measured: 108, 112, 110, 238 and 103 percent of the reference at 1,
  4, 8, 16 and 32 concurrent. The 32 case was the last one behind, at
  95 percent, and what closed it was not a server change at all but the
  prefill tile shape becoming a choice (the Vulkan block's thirtieth
  paragraph): a prompt chunk joining a decode batch arrives at 16 to 32
  rows, which is exactly the band that changed, and the server went
  from 517 to 563 tok/s there with time to first token from 165 to
  109 ms. Time to first token is 1.7 to 4.8 times shorter from 4
  concurrent up, since a prompt joins the running batch as a chunk
  rather than waiting for a slot; the inter-token p99 sits within 2 ms
  of the median at every level on llmx. Before the threshold, 16
  concurrent had been 282 tok/s with a p99 of 55 ms: the pass a prompt
  chunk joined at 16 rows and up took the tile kernel, which costs a
  64-row tile whatever its fill. Two earlier readings were the tool,
  not the server: a p99 of 13 and 62 ms at 1 and 4 concurrent was the
  server's first pass past sixty tokens and its first four-way pass,
  which an eight-token warm-up did not reach; twenty single requests
  in a row show one gap above 9 ms, in the first request at token 60.
  The tool now warms with one request of the measured length. A
  warm-up at the widest level is not neutral, it halved the reference's
  rate at 4 to 16, so the tool does not do that. llmx's TTFT at 1
  concurrent reads 27 ms here against 9 ms in steady state: the first
  request after an idle spell costs 20 to 35 ms more, whatever the
  prompt, on the client's first connection or the device's clocks, and
  a median of two rounds carries it. Prefix reuse, measured: an
  830-token prompt sharing six full blocks with a donor takes 77 ms to
  its first token against 160 ms fresh; its encode is 0.7 ms and the
  fork under 0.1 ms, and the 77 is the pass over the 62 rows that
  follow the shared blocks, which the bench prices at 56 ms without a
  history. On this model a pass of 32 to 128 rows costs a near-flat 50
  to 77 ms (the Vulkan block's twenty-seventh paragraph), so that is
  what a short prompt or a hit's tail pays. The 16-column kernel
  remains open.
  The limits, asked for by the user as the flags a deployment sets:
  the KV pool's budget had been one model context in total, shared by
  every request with no knob, and the queue unbounded. `--ctx-size`
  (`-c`) on `llmx serve` is now the pool's total token budget, the
  model context by default, rounded up to whole blocks, and the
  ceiling on one request's prompt plus `max_tokens` is the smaller of
  the context and that budget, refused with 413 in the API and in the
  scheduler. `--max-queue` (default 64) bounds the requests waiting
  for admission; past it a submit throws `QueueFull`, which the API
  answers with 503. The server test starts a one-sequence, one-queue
  server over 512 tokens and checks the 413 and, with three requests
  arriving 0.2 s apart, statuses 200, 200 and 503. Found on the way:
  `serve` was not among the commands the suite's `--device` reached,
  so the server had run on the CPU in every device pass while the CLI
  it was compared with ran on the device; it is now, and the test
  passes with both on the device.
- **Left:** the 16-column row kernel if sixteen-way batches turn out to
  matter; replaying a recorded decode pass, above, if small-model decode
  becomes the target.
- **Gotchas:** the scheduler thread is the only caller of `forward` for its
  devices, by contract; connection threads queue and drain. A request is
  admitted only when the pool holds its prompt plus `max_tokens`; admitted
  requests are never evicted, donors are. A donor's blocks are shared
  read-only, and a fork rolled back to a block boundary appends into fresh
  blocks, which is what `KVSequence::prepare` requires. Per-request seeded
  sampling keeps a request reproducible whatever it is batched with.

## Vulkan backend, sub-step 1 of docs/VULKAN.md (2026-09-21)

- **Goal:** the first vendor backend over the Radeon VII: storage and
  submission first, kernels in the following sub-steps.
- **Done:** `src/backends/vulkan/vulkan_backend.cpp`, the one translation
  unit outside the header-only runtime, built as a static library only
  with `LLMX_HAS_BACKEND_VULKAN=ON`. The loader is loaded at run time and
  every entry point fetched through it, so nothing links against
  `vulkan-1` and a build without the option is byte-for-byte what it was.
  One instance, the physical device by index, one compute queue without
  graphics, timeline semaphores required and 8- and 16-bit storage and
  arithmetic enabled where present, push descriptors enabled where
  present. Buffers are one `VkBuffer` on their own memory: device-local
  and not host-visible for `Memory::device`, host-visible, coherent and
  cached for `Memory::host_visible`, mapped for their lifetime. `adopt`,
  `write` and `read` go through a 64 MiB staging buffer in chunks;
  `alloc` zero-fills in stream order; `copy` is a device copy. `submit`
  ends the open command buffer and signals the timeline with the ticket,
  `wait` blocks on the value, `sync` submits what is open and waits on the
  last ticket, and a ring of four command buffers is reused as tickets
  retire. One full barrier between consecutive commands. The compute ops
  throw naming their sub-step. `backend-vulkan` passes 13 checks on the
  Radeon VII and exits 77, which CTest reports as skipped, without a
  loader or device. The CMake option-to-flag conversion that the generated
  config needed is exercised by every configure and the default tree is
  unchanged: 19/19 native, Python 12/12.
- **Slip, recorded:** these files were swept into `757dee9`, the commit
  that recorded the fork gate, before they had been built, by a `git add
  -A` that should have been scoped. The default tree was verified within
  minutes and was never broken, since the option is off; the Vulkan tree
  was built and its test run right after, and this commit carries the
  description the previous one lacked.
- **Done: sub-step 2.** Six GLSL compute shaders under
  `src/backends/vulkan/shaders/`, compiled by `glslc` at build time into
  the generated include directory as numeric arrays and embedded, with a
  shared `q.glsl` for block decoding. Pipelines are built on first use
  with a push-descriptor set layout and 128 bytes of push constants; a
  dispatch binds, pushes buffers and constants, launches and fences.
  Elementwise kernels are one invocation per element; the norms are one
  workgroup per row or per (row, head) with a shared-memory reduction, so
  they do not depend on the subgroup size; `embed` decodes F32 and Q8_0
  rows, the table bound once as floats and once as bytes. `backend-vulkan`
  compares every kernel against the CPU backend on random inputs, bounds
  fixed in the test before the first run: exact for add, gather and both
  embed paths, 1e-6 relative for SiLU, 1e-5 for the norms and RoPE;
  160,688 outputs match on the Radeon VII, and out-of-range rows, positions
  and ids are refused on the host. The default tree is untouched by any of
  it.
- **Done: sub-step 3, the row kernel.** `matmul` for F32 and Q8_0: one
  subgroup per output row, lanes striding over blocks, batch columns in
  chunks of eight held in registers, one `subgroupAdd` per column. Q8_0
  rows with an even block count are read as 32-bit words over block pairs
  with the activations as 16-byte vectors, which took the 4096-square Q8_0
  matvec from 32 GB/s to 201 GB/s on the Radeon VII; odd block counts keep
  the 16-bit path. The subgroup size is queried and must divide 256.
  Checked against the CPU backend over batch widths 1, 3, 8 and 13 and
  both parities at 1e-4 relative; 167,388 kernel outputs match in all.
- **Done: sub-step 4.** KV blocks on the device, one K and one V buffer
  per layer holding `[kv_head][token][head_dim]` in 64-token blocks, grown
  by allocate and copy on the queue with the buffers the copy reads from
  kept alive until it retires. `kv_write` scatters a view's rows; `kv_copy`
  copies a block within each layer's buffer; `attention` is one workgroup
  per (row, head) with the subgroups taking tokens round robin, one
  `subgroupAdd` per token for the score, an online softmax so no score
  array is needed, and a shared-memory merge of the subgroups. Checked
  against the CPU backend through each backend's own storage and block
  size on histories of 0, 63, 64, 65 and 131 tokens with 1 and 3 queries,
  two views in one call, and a copied block attending like its source, at
  1e-4 relative; 179,228 kernel outputs match in all. The device's bounds
  checks caught a query offset in the test that the CPU backend reads
  through silently.
- **Done: sub-step 5, all but the floor.** `--device cpu|vulkan:N` on
  `generate`, `chat`, `logits`, `perplexity` and `bench`; a build without
  the backend says so rather than falling back. The Python suite takes
  `--device` and passes it through `LLMX_DEVICE`, test configuration like
  `LLMX_BASELINE_GGUF`; a fixture the device has no kernel for is reported
  as skipped, with the reason, rather than failed. On the Radeon VII the
  whole suite passes with every command on the device: the synthetic F32
  and sharded HF cases, the chat and thread goldens, and the real Qwen3
  Q8_0 baselines, whose top logits and all four perplexity cases match the
  CPU's numbers to the digit (PPL 28.8371 continuous, 38.2140 at c=123).
  Q4_0 was skipped until sub-step 6 gave the device its Q4_1 and Q6_K
  decoders; it passes now, see below.

  Throughput after the arena fix, 247-token prompt and 32 greedy tokens,
  one run each, no gate: 0.6B prefill 382 tok/s and decode 67.6, against
  the CPU's 431 and 46.7 at six threads; 8B prefill 40.3 and decode 21.3,
  against the CPU's 37 and about 4. The first numbers were 19 and 10.9
  tok/s decode, because every dispatch taking ids or positions allocated
  a device buffer, over a hundred `vkAllocateMemory` calls per token; a
  host-visible arena per ring slot removed that. Prefill on the device is
  the row kernel streaming the weights once per eight columns, which the
  tile kernel is for.
- **The floor, measured.** The pinned mx revision's Vulkan backend did
  not build here: its shader generator's nested configure fails under
  CMake 4.x with "CMAKE_C_COMPILER not set", with both the pip and the
  Visual Studio CMake, and is left for a CMake 3.x retry. The reference
  used instead is today's upstream llama.cpp Windows Vulkan release,
  b11075, which is a stricter bar since its kernels are newer. Same card,
  same files, 247-token prompt and 32 generated tokens; llmx numbers are
  single runs after the changes below, the reference is `llama-bench`
  with three repeats.

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | prefill | 660 tok/s | 1195 (three runs within 1%) | 181% |
  | Qwen3-0.6B-Q8_0 | decode | 198 tok/s | 111 | 56% |
  | Qwen3-0.6B-Q4_0 | prefill | 636 tok/s | 1110 (997, 1110, 1110) | 175% |
  | Qwen3-0.6B-Q4_0 | decode | 209 tok/s | 107 (90, 106, 108) | 51% |
  | Qwen3-8B-Q8_0 | prefill | 99 tok/s | 220 | 222% |
  | Qwen3-8B-Q8_0 | decode | 39.7 tok/s | 32.7 | 82% |

  Prefill clears the reference on both models since the tile kernel;
  decode is at about half of it, and the roadmap's bar is the reference,
  not the CPU. Where the time goes, measured per kernel at the 0.6B shapes rather
  than guessed: a near-empty dispatch with its barrier costs 6.3 us, so
  the four hundred dispatches of a token are under 3 of its milliseconds;
  the Q8_0 matvec has a floor of about 17 us at 1 MB and streams at 65 to
  125 GB/s on the 1 to 3 MB layer matrices and about 210 GB/s at 8B's
  sizes, on a card whose memory does 1 TB/s; decode attention was 134 us
  per layer with one workgroup per head walking the whole history. Three
  changes so far: the inter-dispatch barrier names the compute and
  transfer stages instead of all commands (decode +14% on 0.6B); a
  subgroup takes several rows with a lane cluster each, so a 1024-wide
  row no longer idles three quarters of its lanes (neutral, so occupancy
  was not the limit); and attention splits the history into 32-token
  chunks across workgroups with a merge kernel, 134 to 36 us per layer,
  decode 78 to 99.5 tok/s on 0.6B. `backend-vulkan` reports the per-shape
  timings so the next change is measured against them. Fourth, the tile
  kernel for wide batches: a workgroup computes a 64 x 64 output tile
  with the weights dequantized once into shared memory and every thread
  accumulating a 4 x 4 micro-tile, so a weight is read once per pass
  rather than once per eight columns; batches of 16 and up take it.
  Prefill 449 to 1025 tok/s on 0.6B and 40 to 220 on 8B, checked against
  the CPU at batch widths 16, 64, 100 and 247 at 1e-4.
  Fifth, the matvec's access pattern: eight consecutive lanes share one
  block pair, lane t loading words t, t + 8 and, for lane 0, word 16, so
  a load instruction touches 32 contiguous bytes per pair instead of
  four, and every lane reads the two scales directly, which the hardware
  serves as one broadcast and which beat a shuffle. The 8B matrices went
  from 210 to about 295 GB/s and 8B decode from 21.8 to 28.1 tok/s. Two
  variants measured and rejected: a generic loop over words with
  accumulator arrays indexed by column spilled to scratch and ran at 27
  GB/s, and two pairs per iteration cost occupancy and lost a few percent.
  Sixth, grouped projections: the row kernel takes up to three
  projections of one X in one dispatch, workgroups handed to projections
  in order so a workgroup's buffers are selected once, which is dynamic
  indexing of a storage buffer array and a device feature the backend
  now requires. q, k and v are one dispatch, gate and up another, checked
  bit for bit against the same projections one at a time. Decode 104 to
  110 tok/s on 0.6B and 28.1 to 29.3 on 8B.
  Seventh, the lanes per block pair swept: one, two, four, eight and
  sixteen give 200, 190, 336, 295 and 185 GB/s at the 8B shapes, so four
  it is, each lane with five loads in flight over 16 contiguous bytes.
  8B decode 29.3 to 32.7 tok/s.
  Eighth, sub-step 6 for the Q4_0 fixture: Q4_0, Q4_1 and Q6_K in
  `embed`, the tile kernel and the row kernel, checked against the CPU
  (exact for embed, 1e-4 for the matmuls; 599,588 outputs), and the
  fixture's HF baselines pass on the device with the CPU's numbers
  (PPL 32.8463 continuous, 42.5740 at c=123). Two findings on the way.
  The first version put every type's branch in the one row module and
  Q8_0 decode fell from 105 to 65 tok/s with no change to any executed
  Q8_0 instruction, the register demand of the whole module setting the
  occupancy of every path; the row kernel is now one module per family
  of types, F32 and Q8_0, Q4_0 and Q4_1, Q6_K, built from one source with
  a define, and Q8_0 is back at 110 to 112 against the parent commit's
  110 to 112 in an interleaved run. The second: byte loads with scalar
  activation loads gave Q4_0 17 GB/s, Q4_1 35 and Q6_K 6 to 17 at the 8B
  shapes, and the Q6_K head of the fixture, 151,936 rows, took 22 ms of a
  38 ms token. The word paths read the quants as 32-bit words, Q4_0 as a
  block per lane over the 9-word pair with the first block's nibble
  words assembled from two loads, Q4_1 as one lane per 5-word block,
  Q6_K as sixteen lanes per block each holding three words of quants,
  two of sub-scales and the scale, with every other block's words
  assembled from two loads since 210 bytes is not a multiple of four;
  and the activations as aligned 16-byte vectors, one load for four.
  Now 149, 188 and 170 GB/s at the 8B shapes and the head at 102 GB/s,
  1.25 ms. Q4_0 decode 26 to 107 tok/s; llama.cpp does 209 on the same
  file.
  Ninth, the rest of sub-step 6: Q4_K and Q5_K in `embed`, the tile
  kernel and the row kernel, checked against the CPU the same way
  (723,796 outputs across the seven types). The row kernel is now five
  modules from one source: F32 and Q8_0, Q4_0 and Q4_1, Q4_K, Q5_K,
  Q6_K. Q4_K and Q5_K beside Q6_K cost Q6_K 40 percent (170 to 99 GB/s)
  the way every type beside Q8_0 had cost Q8_0, and Q4_K alone gained 30
  percent over Q4_K beside Q5_K (104 to 135 GB/s); at the 8B shapes the
  row kernel reads Q4_K at 135 GB/s, Q5_K at 123 and Q6_K at 163. Eight
  lanes share a Q4_K or Q5_K block, each taking the sixteen nibble bytes
  of one half of one 64-value chunk, with the three packed sub-scale
  words read by every lane. `Qwen3-0.6B-Q5_K_M.gguf` (168 Q5_K, 29 Q6_K,
  113 F32; same repo and revision as the Q4_0 file) is a third
  `BASELINE_MODELS` fixture, as the fused Q5_K block below proposed, with
  bounds set from the CPU's measured HF deltas plus margin: top-5 overlap
  4, NLL delta 0.05 continuous (measured 0.026) and 0.16 per chunk
  (measured 0.130, 0.068, 0.024). On the device its logits and all four
  perplexity cases match the CPU's numbers to the digit (PPL 29.5612
  continuous, 38.6676 at c=123).

  A confirmation run of the reference at the end of the session, asked
  for by the user, moved with the machine: llama.cpp 0.6B Q8_0 measured
  602 prefill and 182 decode against 660 and 198 three hours earlier,
  and llmx in the same minutes 1000 to 1157 and 102.7 to 106.5 against
  1195 and 111; Q4_0 the same, 624/209 against 636/209 and llmx 878 to
  1056 and 95.7 to 97.4 against 1110 and 107. Both arms fell together, so
  the shares hold and neither run is a regression; the table keeps the
  earlier pairs and the session-end pairs side by side.

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q5_K_M | prefill | 492 tok/s | 640 (533, 638, 645) | 130% |
  | Qwen3-0.6B-Q5_K_M | decode | 205 tok/s | 87 (81, 88, 87) | 42% |
  | Qwen3-0.6B-Q8_0, session end | prefill | 602 tok/s | 1000 to 1157 | 166 to 192% |
  | Qwen3-0.6B-Q8_0, session end | decode | 182 tok/s | 102.7 to 106.5 | 56 to 59% |
  | Qwen3-0.6B-Q4_0, session end | prefill | 624 tok/s | 878 to 1056 | 141 to 169% |
  | Qwen3-0.6B-Q4_0, session end | decode | 209 tok/s | 95.7 to 97.4 | 46% |
  Tenth, two decode wins found by timing the host against the device.
  Temporary instrumentation put a 0.6B decode token at 0.5 ms of host
  recording and 7.4 ms of device time over 396 dispatches, so the
  device is the story, and the row kernel's final reduction turned out
  to run its xor-shuffle chain over all eight column slots whether one
  column was live or eight: 48 shuffles per lane after five loads.
  Reducing only the live columns took the 1024-square Q8_0 matvec from
  17.1 to 13.2 us, 1024x3072 from 33 to 22 us and the 8B shapes from 313
  to 373 GB/s, and 0.6B decode from 122 to 145 tok/s over 128 tokens.
  Then the pass is submitted in chunks of 64 dispatches as it is
  recorded, so the device starts while the host records the rest: 145
  to 158 tok/s; chunks of 16, 32, 64, 128 and 256 gave 150, 156, 158,
  152 and 149, and 8B prefill is unchanged by it (203 to 208 either
  way). Both changes leave every kernel output identical and the greedy
  text the same. Three interleaved rounds of the four files afterwards:

  | model | phase | llama.cpp b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | prefill | 660 tok/s | 1029, 1105, 1109 | 156 to 168% |
  | Qwen3-0.6B-Q8_0 | decode | 198 tok/s | 137.9, 137.1, 136.5 | 69% |
  | Qwen3-0.6B-Q4_0 | prefill | 636 tok/s | 934, 962, 1040 | 147 to 164% |
  | Qwen3-0.6B-Q4_0 | decode | 209 tok/s | 128.2, 129.4, 127.3 | 61% |
  | Qwen3-0.6B-Q5_K_M | prefill | 492 tok/s | 681, 736, 742 | 138 to 151% |
  | Qwen3-0.6B-Q5_K_M | decode | 205 tok/s | 122.1, 110.5, 118.4 | 54 to 60% |
  | Qwen3-8B-Q8_0 | prefill | 99 tok/s | 208, 208, 209 | 210% |
  | Qwen3-8B-Q8_0 | decode | 39.7 tok/s | 35.5, 35.5, 35.7 | 89% |

  Decode over 128 tokens reads higher than over 32 (158 against 138 on
  0.6B Q8_0) because the first tokens carry the clock ramp; llama-bench
  warms up before its 32, so the 32-token llmx figure is the
  conservative one and the table keeps it.
  Eleventh, the residual add folded into the matmul's store. The model
  asks for `matmul_add`, Y += W X, and each backend produces it its own
  way: the CPU computes into scratch and adds, the same arithmetic as
  before to the bit, the device sets a flag in the row and tile kernels'
  push constants and the store becomes an accumulate. Two dispatches
  fewer per layer; 0.6B Q8_0 decode 137 to 140-146 tok/s, 8B 35.5 to
  35.8, checked against the CPU at batch widths 1, 3 and 64.

  Twelfth, and the one that changes the reading of every table above:
  the user asked whether the measurement was right, and it was not
  matched. The reference's bench tool warms up, then averages three
  repeats of prompt processing into an empty history and of generating
  32 tokens from an empty history, model time only. `generate --verbose`
  gave one cold run whose decode came after the 247-token prompt, so
  with attention over 250 to 280 tokens every step, with sampling and
  text output inside the timer and the clock ramp in the first tokens.
  `llmx bench --model` now runs the reference's protocol (docs/USAGE.md),
  and the same binary that read 138 tok/s under `generate` reads 202
  under it. Both arms in the same minutes, same card and files:

  | model | phase | reference b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp247 | 651.2 +- 4.3 tok/s | 1507.6 +- 5.1 | 232% |
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 +- 0.6 tok/s | 201.9 +- 0.2 | 103% |
  | Qwen3-0.6B-Q4_0 | pp247 | 669.7 +- 3.1 tok/s | 1367.1 +- 5.0 | 204% |
  | Qwen3-0.6B-Q4_0 | tg32 | 220.8 +- 0.9 tok/s | 194.8 +- 1.5 | 88% |
  | Qwen3-0.6B-Q5_K_M | pp247 | 518.1 +- 1.6 tok/s | 875.1 +- 3.5 | 169% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 +- 1.0 tok/s | 169.4 +- 0.7 | 77% |
  | Qwen3-8B-Q8_0 | pp247 | 97.0 +- 0.5 tok/s | 232.5 +- 0.4 | 240% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 +- 0.1 tok/s | 40.6 +- 0.1 | 105% |

  Q8_0 decode is at the floor on both models under the matched protocol.
  The earlier tables stand as what `generate --verbose` measured, both
  arms' figures at the time; they are not the floor comparison.

  Thirteenth, the attention inputs as one op. `norm_rope_kv` is what the
  model asks for at every layer: q normed and rotated in place, k normed
  and rotated and written with v into the view's KV blocks. The base
  class default is the three ops it replaces, which is what the CPU
  runs, so its arithmetic is unchanged; the device runs one kernel, a
  workgroup per (row, head) over the q, k and v heads, the k heads
  written straight into their blocks, three dispatches fewer per layer.
  Checked against the CPU's three ops with a 70-token history, q
  directly at 1e-5 and k and v through attention at 1e-4. Under the
  matched protocol, same session as the table above:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 tok/s | 201.9 | 221.3 +- 0.8 | 113% |
  | Qwen3-0.6B-Q4_0 | tg32 | 220.8 tok/s | 194.8 | 205.4 +- 0.4 | 93% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 tok/s | 169.4 | 176.7 +- 0.4 | 80% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 tok/s | 40.6 | 41.0 +- 0.0 | 106% |

  Prefill moved within noise (1494, 1352, 873 and 230 tok/s).

  Fourteenth, the norm folded into the matmul, tried three ways and
  rejected, not committed. The op was `matmul_group_normed`, the RMS
  norm of X against w and the projections in one call, the CPU norming
  into scratch first and the device folding it into the row kernel for
  one column. (1) The column staged in 16 KB of shared memory per
  workgroup, normed there, every read from it: 8B decode 41.0 to 13.4
  tok/s and 0.6B 221 to 101, with the plain modules carrying the array
  too; in separate normed modules, so the plain ones had none, 8B 22.3
  and 0.6B 165. Sixteen KB of shared memory per workgroup caps what a
  compute unit holds. (2) No staging, each read scaled by the row's
  factor and w on the way in: 8B 12.5, 0.6B 137; with the reduction
  removed and the scale alone left, 8B 11.7, so the per-read scale and
  its second load are what the tuned Q8_0 load pattern cannot absorb.
  (3) A 4 KB staging for rows up to 1024 wide: 0.6B Q8_0 203, Q4_0 199,
  Q5_K_M 176 against 221, 205 and 177 unfused. Every variant also
  moved the prefill's norm into a backend scratch buffer, and that
  alone cost 8B prefill 232 to 166 tok/s; the buffer was device-local
  and allocated once, and the cause was not found before the whole
  change was reverted, so a backend-allocated activation buffer is a
  thing to measure before using again. The two norm dispatches per
  layer stay: on 0.6B they are about a tenth of a decode token and no
  fusion tried gets them back.

  Fifteenth, the greedy-output hash the user asked for: `generate` with
  the 247-token excerpt, 128 greedy tokens, on the CPU and on the
  device, output hashed. Qwen3-0.6B Q8_0, Q4_0 and Q5_K_M give the same
  hash on both backends (e41aee746cc31abf, c19367d54fb40354,
  b18a64cda0208547), so the device's reduction-order differences never
  flipped an argmax over 128 steps on any of the three quantizations,
  and Qwen3-8B-Q8_0 gives the same hash on both over 64 tokens
  (379c733a478ec58a). The check is CPU against device on the same file:
  a greedy sequence from a quantized file cannot be checked against the
  fp32 reference beyond its first tokens, since quantization legitimately
  moves the argmax, so HF stays the ranking and NLL gate.

  Sixteenth, the long-context case the user asked for, `bench --model`
  with 16384 prompt tokens and 512 generated, two repeats after a
  warm-up, both arms in the same minutes:

  | model | test | reference b11075 Vulkan | llmx Vulkan | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 155.2 +- 3.7 | 30% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 192.5 +- 0.2 | 109% |
  | Qwen3-8B-Q8_0 | pp16384 | 97.3 +- 0.5 tok/s | bad allocation | - |
  | Qwen3-8B-Q8_0 | tg512 | 37.6 +- 0.1 tok/s | bad allocation | - |

  Two findings. Decode over a 512-token history still leads, but a
  16384-token prompt runs at 30 percent of the reference: the device
  attention kernel is one workgroup per (query row, head) with an
  online softmax, which is the flash form for decode, but prefill has
  no query tiling, so every query row re-reads its whole K/V history
  and the traffic is quadratic. At 247 tokens that is invisible behind
  the matmuls; at 16384 it is the prompt. And Qwen3-8B does not fit a
  16k context on the 16 GB card: the KV cache is F32, 16896 tokens of it
  are about 5 GB on top of 8.7 GB of weights, and the allocation
  failed, where the reference's f16 KV fits.
  Seventeenth, the tiled prefill attention. A wide pass of 128-wide
  heads takes `attention_tile`: a workgroup per 32 query rows and head,
  the head's K and V streamed through shared memory in tiles of 16
  tokens, so a tile is read once per 32 rows instead of once per row;
  eight lanes share a row, each holding 16 elements of it, a score is
  three xor shuffles, the softmax is online per row as before. Rows
  past the causal limit are masked inside the tile that holds them and
  later tiles are not read. Checked against the CPU at 32 and 45 rows
  after 0 and 70 tokens at 1e-4. Same session as the table above, the
  8B case at 8k because 16k does not fit its f32 cache:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 155.2 | 513.3 +- 0.2 | 100% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 192.5 | 191.9 +- 0.5 | 108% |
  | Qwen3-8B-Q8_0 | pp8192 | 115.7 +- 0.2 tok/s | - | 185.9 +- 1.3 | 161% |
  | Qwen3-8B-Q8_0 | tg512 | 37.3 +- 0.1 tok/s | - | 39.2 +- 0.0 | 105% |
  | Qwen3-0.6B-Q8_0 | pp247 | 656.0 +- 2.1 tok/s | 1507.6 | 1610.5 +- 4.1 | 246% |
  | Qwen3-0.6B-Q8_0 | tg32 | 198.3 +- 0.3 tok/s | 221.3 | 226.9 +- 0.4 | 114% |

  The 16k prompt went from 30 percent of the reference to level with
  it, and the kernel does not yet share a K/V tile across the query
  heads of a KV group, which is the next factor available there.
  Eighteenth, the f16 cache sides. `--cache-type-k` and `--cache-type-v`
  take `f32` or `f16` for `generate`, `chat`, `logits`, `perplexity` and
  `bench --model`, the same on the CPU and the device; the type reaches
  each backend through `kv_alloc`, the CPU converts with F16C on write
  and read, the device builds four variants of every cache kernel and
  the storage picks one, and the device writes halves with an explicit
  round-to-nearest-even in the bits because `packHalf2x16` leaves the
  rounding to the driver and the first build differed from the CPU by
  an f16 ulp. The two backends now hold identical cache bytes; checked
  for every combination of K and V types through `kv_write`,
  `norm_rope_kv`, `kv_copy` and both attention kernels, device against
  CPU at 1e-4 and both against the f32 cache at 2e-2. The HF gate with
  both sides f16 (`run_tests.py --cache-type f16`) passes on both
  backends with the same numbers to three places: Q8_0 NLL delta
  0.001254 against 0.001374 with f32 caches, Q4_0 0.131584, Q5_K_M
  0.026174; the synthetic F32 and shard gates, which compare exact f32
  arithmetic, skip under f16 caches by design. Halving the cache lets
  Qwen3-8B run the 16k context that failed to allocate with f32:

  | model | test | reference b11075 Vulkan | llmx f32 cache | llmx f16 cache | llmx share (f16) |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q8_0 | pp16384 | 97.3 +- 0.5 tok/s | bad allocation | 141.7 +- 0.3 | 146% |
  | Qwen3-8B-Q8_0 | tg512 | 37.6 +- 0.1 tok/s | bad allocation | 38.7 +- 0.0 | 103% |
  | Qwen3-8B-Q8_0 | pp8192 | 115.7 +- 0.2 tok/s | 185.9 +- 1.3 | 189.9 +- 2.7 | 164% |
  | Qwen3-8B-Q8_0 | tg512 (8k) | 37.3 +- 0.1 tok/s | 39.2 +- 0.0 | 39.5 +- 0.0 | 106% |
  | Qwen3-0.6B-Q8_0 | pp16384 | 514.1 +- 1.7 tok/s | 513.3 +- 0.2 | 512.9 +- 0.7 | 100% |
  | Qwen3-0.6B-Q8_0 | tg512 | 177.2 +- 0.1 tok/s | 191.9 +- 0.5 | 203.6 +- 0.1 | 115% |
  | Qwen3-0.6B-Q8_0 | pp247 | 656.0 +- 2.1 tok/s | 1610.5 +- 4.1 | 1620.9 +- 3.7 | 247% |
  | Qwen3-0.6B-Q8_0 | tg32 | 198.3 +- 0.3 tok/s | 226.9 +- 0.4 | 225.1 +- 0.5 | 113% |

  The default is f16 on both sides, taken as a decision on 2026-09-22
  after these measurements: it is what the reference runtime stores by
  default, so a comparison is like-for-like without flags; the HF gate
  passes with it on both backends, the 8B Q8_0 excerpt reading a
  perplexity delta of 0.001254 against f32's 0.001374 inside a 0.01
  bound; a decode step reads the whole cache, so a 512-token generation
  on Qwen3-0.6B gains 6 percent; and the 8B runs a 16k context that f32
  cannot allocate on a 16 GB card. `f32` on both sides stores the cache
  exactly and is one flag away. The three components that compare exact
  f32 arithmetic against independent fixtures (`f32`, `shards`,
  `server`) now ask for f32 sides themselves, so they test what they
  tested before; `--cache-type` overrides them, and under `f16` they
  skip as they did. All three suites pass at the new default: the CPU
  suite, the CPU suite with `--cache-type f32`, and the device suite,
  with the Q5_K_M perplexity delta reading 0.067623, 0.067633 and
  0.067753 across them.
  Nineteenth, mixed groups partitioned by type. A Q5_K_M layer's q and k
  are Q5_K and its v is Q6_K, and a group of mixed types fell back to
  one dispatch per projection; it is now one dispatch per type, two for
  that group instead of three. Q5_K_M tg32 176.7 to 185.7 +- 0.5 tok/s
  (84 percent of the reference's 219.8); Q4_0, whose groups are pure,
  read 211.2 +- 0.3 in the same minutes against 205.4 earlier, which is
  the session's drift, not the change. After the reduction fix the row
  kernel's per-type readings at the 8B shapes are Q8_0 415 GB/s, Q6_K
  215, Q5_K 179, Q4_0 178, Q4_K 167, and the Q6_K head of the 0.6B
  files, 151,936 rows of 1024, takes 693 us at 184 GB/s, 7 percent of a
  Q5_K_M token; at 1024 wide every row's lanes re-read the whole
  activation row, 620 MB of cache traffic against 127 MB of weights.
  Twentieth, two rows per lane cluster in the Q6_K module, tried and
  rejected, not committed: each lane decoded its sixteenth of two rows'
  blocks and applied both to the activation vectors it had loaded once,
  halving the activation re-read. The second row's quants and scales
  doubled the live registers and the occupancy lost outweighed the
  traffic saved: the head went from 693 to 1184 us, the 8B shape from
  215 to 129 GB/s, and Q5_K_M tg32 from 185.7 to 161.1 tok/s. The
  activation re-read is the traffic, but the answer is not more state
  per lane. Twenty-first, the other way round, a 4 KB shared-memory
  copy of the activation row per workgroup in the Q6_K module for
  one-column passes up to 1024 wide, every lane reading from it: also
  rejected, the head 693 to 771 us, Q5_K_M tg32 185.7 to 180.1, and
  even the 4096-wide shape, which was not staged, fell from 215 to 175
  GB/s from the select on every read. The K-quant row kernels at these
  shapes are not limited by activation traffic in a way either fix can
  reach; what remains there is the per-dispatch floor and the decode
  work per lane, and it is left at 84 percent for now.

  Twenty-second, the block-size screening VULKAN.md sub-step 7 asked
  for, on the device, same session, `bench --model --p 247 --n 512`,
  two repeats: 32, 64 and 128 tokens per block give 201.4, 200.6 and
  201.9 tok/s on 0.6B and 40.9, 41.0 and 41.0 on 8B, all within a
  repeat's spread, every kernel check passing at each size. 64 stays.
  Twenty-third, integer activations for the decode row kernel
  (VULKAN.md sub-step 8). The 4- and 5-bit files sat at 88 and 77
  percent of the reference's decode and the row kernel's per-type
  readings said why: Q4_0 moved 182 GB/s of weights where Q8_0 moved
  413, the same weights per second, so the 4-bit paths were bound by
  what they did per weight, eight nibble extractions, conversions and
  multiply-adds per word, and by the float activation bytes every path
  re-read per row. The device accelerates packed 8-bit and 16-bit
  integer dots, so three probes at the 8B shape, wrong numerics and the
  kernel alone, put a number on each form:

  | shape | float | int8, GGUF word order | int8, aligned words | int16 |
  |---|---:|---:|---:|---:|
  | Q8_0 4096 x 12288 | 130 us | 147 | 118 | 112 |
  | Q8_0 12288 x 4096 | 152 us | 154 | 120 | 109 |
  | Q4_0 4096 x 12288 | 155 us | 95 to 102 | - | 105 to 110 |

  Q8_0 in the GGUF word order was slower than float: its first block's
  values sit two bytes into each word, and the per-word handling of that
  cost more than the dots saved; aligned, the int path won. 16-bit
  activations took the design over 8-bit on two counts: the CPU
  experiment in ASSETS.md had put per-block 8-bit activations at 0.0093
  of NLL against the 0.010 bound on the 8B excerpt and 16-bit at
  0.00003, and 16-bit dots were the faster of the two on Q8_0 besides.
  The implementation (`shaders/xquant.glsl`, `matmul_row.comp`): every
  quantized row meets the activations as signed 16-bit values in blocks
  of 32, the block's largest magnitude at 32767, with the scale and the
  scaled sums, whole and per half of 16, in a table; a weight word's
  values pair off with activation words in the order nibble and byte
  words unpack in, a block's integer sum is scaled once, and a type's
  offset is folded through the block sum, from one lane of each group
  that reduces together where lanes share a group. The twin is written
  by whichever kernel produces the input, the norm, the SiLU and the
  per-row attention or its merge, tagged for the next row matmul on that
  buffer, so a decode token quantizes nothing in a dispatch of its own;
  an input without a producer takes one. The CPU backend and the tile
  kernel keep float activations. Three findings on the way, each
  measured on the 8B shape: the standalone quantize dispatch costs 5 to
  9 us, which is why the producers write the twin; the same dots over
  8-byte activation loads ran Q4_0 at 150 us against 105 with 16-byte
  loads, so the load count and not the arithmetic bounds these paths;
  and for Q8_0 the per-word handling of the misaligned first block cost
  a quarter of the kernel (selects 154 us, aligned form 125, the narrow
  path beside the wide one in the module another third), so the wide
  path shifts its two first-block words by a half word with the word
  after, fetched from the next lane of the pair by a shuffle, and lives
  in a module of its own. Kernel readings, float to integer, the
  integer figure including the standalone quantize dispatch the test's
  matmul takes:

  | shape | float | integer |
  |---|---:|---:|
  | Q8_0 4096 x 12288 | 130 us, 413 GB/s | 134 to 136 us, 392 to 399 GB/s |
  | Q8_0 12288 x 4096 | 152 us, 350 GB/s | 133 us, 400 GB/s |
  | Q4_0 4096 x 12288 | 155 us, 182 GB/s | 107 to 110 us, 257 to 264 GB/s |
  | Q4_1 12288 x 4096 | 167 us, 189 GB/s | 98 us, 320 GB/s |
  | Q4_K 4096 x 12288 | 163 us, 173 GB/s | 118 to 121 us, 234 to 240 GB/s |
  | Q5_K 4096 x 12288 | 219 us, 158 GB/s | 160 to 166 us, 208 to 216 GB/s |
  | Q6_K 4096 x 12288 | 247 us, 167 GB/s | 223 to 226 us, 183 to 186 GB/s |
  | Q6_K 1024 x 151936 | 825 us, 155 GB/s | 780 to 818 us, 156 to 164 GB/s |

  Correctness: `backend-vulkan` feeds the CPU reference the activations
  quantized the same way, so the comparison is about the dots and the
  reduction order at 1e-4 over every type and both block-count
  parities, and checks a norm, a SiLU and an attention into a buffer
  followed by a matmul from it; the whole CTest and device Python
  suites pass. The HF gate on the device, the tile path against the row
  kernel forced with `--ubatch 8` and `--ubatch 1`, mean NLL over the
  247-token excerpt, HF 3.360286:

  | model | CPU float | device tile | device row kernel | bound |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 3.361660 | 3.361670 | 3.361670 | 0.010 |
  | Qwen3-0.6B-Q4_0 | 3.491840 | 3.491820 | 3.491820 | 0.160 |
  | Qwen3-0.6B-Q5_K_M | 3.386460 | 3.386460 | 3.386460 | 0.050 |

  The integer row kernel gives the tile path's NLL to six decimals. A
  bug surfaced on the first model run and is fixed: a stream-ordered
  scratch outgrown mid-pass, the twin's and the attention split
  states', was freed while the open command buffer still named it, and
  the device hung; `grow` now retires the old buffer with the ring slot.
  The floor under the matched protocol, three arms in the same minutes,
  the before arm built from a detached worktree at 37a020f, two rounds:

  | model | test | reference b11075 Vulkan | llmx before | llmx after | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 197.7, 197.9 tok/s | 217.0, 225.4 | 213.7, 206.3 | 104 to 108% |
  | Qwen3-0.6B-Q4_0 | tg32 | 226.4, 226.5 tok/s | 202.6, 197.1 | 206.5, 206.0 | 91% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 223.7, 223.5 tok/s | 179.6, 185.3 | 176.7, 179.9 | 79 to 80% |
  | Qwen3-8B-Q8_0 | tg32 | 39.9, 39.7 tok/s | 41.0 | 41.0, 40.7 | 103% |
  | Qwen3-8B-Q4_K_M | pp247 | 78.5, 79.0 tok/s | 155.5 | 155.3, 155.4 | 197% |
  | Qwen3-8B-Q4_K_M | tg32 | 52.3, 52.3 tok/s | 34.8, 35.0 | 41.5, 41.6 | 79% |

  The tradeoff, reported together: the 8B 4-bit file, measured for the
  first time here, gains 19 percent of decode and goes from 67 to 79
  percent of the reference; 8B Q8_0 is unchanged; on the 0.6B files the
  change is within a few percent either way, Q4_0 up, Q8_0 and Q5_K_M
  down, all inside the session's own spread between rounds, because a
  0.6B token is about three hundred dispatches at the per-dispatch
  floor and its 1024-wide matmuls run at 60 to 160 GB/s whatever the
  arithmetic. The gain is where the weights are, and it is kept.
  Twenty-fourth, the Q6_K path. It was the weak type of a Q4_K_M file
  at 186 GB/s against Q4_K's 240, half the weights per second at one
  and a half times the bytes, because sixteen lanes of four positions
  each loaded three funnelled quant words, two sub-scale words, the
  scale, four activation pairs and four table entries for sixteen
  weights, with every other block's words costing two loads since 210
  bytes is not a multiple of four. Now eight lanes of eight positions:
  six quant words as three consecutive pairs of three loads each when
  unaligned, the activations as four 16-byte loads, the four group
  scales and the two lanes' half sums as two 16-byte loads each. Every
  check passes; at the 8B shape 218 to 175 us (189 to 236 GB/s) and the
  0.6B files' 151,936-row head 686 to 578 us, from 825 before the
  integer activations. Three arms in the same minutes, the before arm
  a detached worktree at 524ad46, two rounds:

  | model | test | reference b11075 Vulkan | llmx before | llmx after | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 195.4, 194.9 tok/s | 210.3, 210.1 | 209.8, 209.3 | 107% |
  | Qwen3-0.6B-Q4_0 | tg32 | 223.4, 223.1 tok/s | 211.0, 211.3 | 221.1, 212.4 | 95 to 99% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 220.5, 220.6 tok/s | 177.6, 177.8 | 188.7, 188.6 | 85% |
  | Qwen3-8B-Q4_K_M | tg32 | 51.6, 51.7 tok/s | 41.1, 40.8 | 44.8, 44.8 | 87% |

  Q8_0 has no Q6_K and does not move; the Q4_0 file's head and the
  Q5_K_M and Q4_K_M files' v and down projections are where it lands.
  Twenty-fifth, the Q4_K and Q5_K path. Vector loads first, a lane's
  four quant words and the scale with the three packed sub-scale words
  as 16 bytes each and a group pair's table entries as one, which were
  correct and bought nothing at the 8B shape: unlike Q6_K these paths
  were not load-bound. Timing-only variants then put the cost where it
  was: with the sub-scale decode replaced by constants Q4_K ran 121 to
  110 us and Q5_K 165 to 138, and the min term another 5 percent. Each
  of a block's eight lanes decoded its two groups through byte selects
  on branches that diverge across the lanes. The decode is now
  branch-free with selects over the two words the lane's groups sit in,
  Q4_K 121 to 112 us (232 to 256 GB/s) and Q5_K 165 to 138 (210 to
  250), every check passing. Three arms in the same minutes, the before
  arm a detached worktree at adec4f7, two rounds:

  | model | test | reference b11075 Vulkan | llmx before | llmx after | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | tg32 | 197.1, 196.8 tok/s | 204.3, 204.6 | 211.5, 204.1 | 104 to 107% |
  | Qwen3-0.6B-Q4_0 | tg32 | 224.5, 224.4 tok/s | 221.8, 221.8 | 221.8, 221.1 | 99% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 220.9, 221.7 tok/s | 189.2, 183.9 | 195.9, 196.8 | 89% |
  | Qwen3-8B-Q4_K_M | tg32 | 51.9, 51.9 tok/s | 44.7, 45.1 | 46.5, 47.3 | 90 to 91% |

  Twenty-sixth, the tile kernel's threshold. It took every batch of 16
  rows and up, a number set before the row kernel existed in its
  present form; the server's inter-token p99 at 1 and 4 concurrent
  (13 and 62 ms against medians of 5 and 9) was the pass in which a new
  prompt's chunk joined the decoders and tipped the batch into the tile
  kernel, which costs a 64-row tile whatever its fill. Prompt
  processing at 8 to 256 rows, the tile at its old threshold against
  the row kernel taking every width, three models, three runs each:

  | model | rows | tile kernel | row kernel |
  |---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 8 | (row) 712 tok/s | 727 |
  | | 16 | 333 | 791 |
  | | 32 | 640 | 805 |
  | | 64 | 831 | 830 |
  | | 128 | 1810 | 832 |
  | | 256 | 1478 | 838 |
  | Qwen3-8B-Q8_0 | 8 | (row) 72.4 | 72.7 |
  | | 16 | 52.3 | 74.2 |
  | | 32 | 100.7 | 74.9 |
  | | 64 | 173.1 | 75.2 |
  | | 128 | 224.3 | 75.1 |
  | | 256 | 243.3 | 75.3 |
  | Qwen3-8B-Q4_K_M | 8 | (row) 97.7 | 98.3 |
  | | 16 | 33.7 | 100.9 |
  | | 32 | 63.4 | 102.5 |
  | | 64 | 102.0 | 103.3 |
  | | 128 | 143.0 | 103.6 |
  | | 256 | 156.7 | 103.6 |

  The row kernel's rate is flat in the width, a weight pass per eight
  columns, and the tile's climbs with its fill; the tile loses at 16 on
  every file and wins from about 24 rows on 8B Q8_0 and 64 on the other
  two. The threshold is now 32 rows when every projection of the
  dispatch is F32 or Q8_0 and 64 otherwise, which follows the three
  measurements within a few percent and costs the 0.6B Q8_0 up to a
  fifth at 32 to 63 rows against its own best. With the thresholds in
  place, three runs each: 8B Q8_0 at 16, 32 and 64 rows 74.6, 102.4 and
  174.7 tok/s (52.3, 100.7 and 173.1 before), 8B Q4_K_M 101.4, 103.0 and
  102.6 (33.7, 63.4 and 102.0). The server's figures are in the server
  block. The backend-vulkan test's row-kernel reference follows the two
  thresholds, the Linux build passes CTest with the test skipping on
  llvmpipe, and the device suite passes.
  Twenty-seventh, the cost of a prompt pass in rows, from the server's
  time to first token. A sweep of the bench on 0.6B Q8_0, pass time in
  ms at 31, 32, 48, 63, 64, 65, 96, 128, 129, 192, 193, 256 and 257
  rows: 38, 50, 54, 57, 77, 69, 73, 70, 84, 99, 113, 173, 185. Each new
  64-row tile adds about 13 ms to the pass, which is the tile kernel's
  price per tile across the layers; exactly 64 rows costs 8 ms more
  than 65; and the tile from 193 to 256 rows grows from 0.5 to 1.7 ms
  per added row before resetting at 257. The three are open, and so is
  the tiled attention over a long history with few query tiles: a
  62-row tail over 768 shared tokens is two tiles by sixteen heads,
  32 workgroups walking 830 keys each, where the per-row kernel splits
  such a history across up to 64 workgroups. The scheduler's host side
  is not in this: a pass of 62 prompt rows returns from forward in 29
  ms with the command ring four chunks deep, and the rest is the
  device. Against the reference, back to back, three runs each,
  tok/s:

  | rows | 0.6B Q8_0 reference | llmx | share | 8B Q8_0 reference | llmx | share | 8B Q4_K_M reference | llmx | share |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 32 | 953 | 641 | 67% | 97.5 | 102.4 | 105% | 100.5 | 102.7 | 102% |
  | 64 | 929 | 838 | 90% | 79.9 | 174.1 | 218% | 58.3 | 102.6 | 176% |
  | 128 | 620 | 1831 | 296% | 91.3 | 225.5 | 247% | 66.7 | 143.8 | 216% |
  | 256 | 651 | 1487 | 228% | 100.3 | 244.2 | 244% | 79.1 | 157.2 | 199% |
  | 512 | 1662 | 1795 | 108% | 146.1 | 267.5 | 183% | 111.2 | 176.5 | 159% |

  The prompt gate holds on both 8B files at every width and on 0.6B
  from 128 rows; 0.6B at 32 and 64 rows is where the tile kernel's
  price per tile shows, a pass of 50 ms against the reference's 34.
  Twenty-eighth, the K-quant row kernel by its registers. The backend
  now captures the driver's per-kernel statistics when the device has
  `VK_KHR_pipeline_executable_properties` and the test prints them:
  vector and scalar registers, shared memory and scratch per kernel.
  They put occupancy behind the row paths' ranking: the wide Q8_0 path
  at 41 vector registers runs five waves per SIMD and 394 GB/s at
  4096 x 12288, Q4_K at 75 three waves and 261, Q5_K at 93 two waves
  and 254, Q6_K at 78 three and 238. Folding a block's scales once and
  unpacking Q5_K's fifth bits per nibble word, so nothing unpacked
  stays live across the columns:

  | path | registers before | after |
  |---|---:|---:|
  | Q4_K | 75 | 74 |
  | Q5_K | 93 | 84 |

  Q4_K did not move because the compiler hoists all four activation
  loads whatever the source order. The bandwidth first recorded here
  for Q5_K, 254 to 296 GB/s, was wrong: the 254 came from a tree
  before the register trim rather than from the arm being compared.
  Interleaved against the same tree afterwards, both this change and
  the extension's removal leave Q5_K level at 295 to 306, so the trim
  is kept for the registers and not for a rate. Three layouts measured worse and
  were dropped: sixteen lanes per block (55 and 59 registers, four
  waves, 245 and 272 GB/s, a lane then keeping half the weight bytes
  in flight per load), the next block's nibble words loaded before
  this block's dots (81 and 92 registers, 251 and 257 GB/s, the
  hardware's in-order load counter making a wait for this block's
  loads wait for the prefetch), and both at once (226 and 260). On the
  models the change is flat: 8B Q4_K_M decode 46.95 to 46.99 tok/s
  against the reference's 51.8 (91 percent), 0.6B Q5_K_M 195 to 202
  against 221 (88 to 91 percent), where the 0.6B shapes are bound by
  dispatch latency rather than bandwidth (Q5_K at 1024 x 3072 reads
  128 GB/s). The lever that remained for Q4_K was below the
  source, and the same extension served it: with `diagnostics` the
  backend captures the driver's disassembly of every kernel and
  `backend-vulkan --isa DIR` writes one file per kernel. Reading the
  wide Q8_0 and Q4_K listings showed the integer dot product extension
  buying nothing: its 16-bit dot lowered to the same multiply-add pairs
  a plain expression gives, with the operands sign-extended first.
  Written as multiplies of sign-extended halves and bytes instead,
  interleaved against the extension, two passes each:

  | path | extension | multiplies |
  |---|---:|---:|
  | Q4_K 4096 x 12288 | 261, 267 GB/s | 271, 282 |
  | Q6_K 1024 x 151936 | 223, 224 | 232, 233 |
  | Q8_0 4096 x 12288 | 393, 398 | 397, 398 |
  | Q5_K 4096 x 12288 | 305, 295 | 299, 297 |
  | 8B Q4_K_M decode | 47.1, 47.4 tok/s | 48.5, 48.6 |
  | 8B Q8_0 decode | 40.6, 40.7 | 40.9, 41.0 |

  Q4_0, Q4_1 and the 0.6B files are level, the last because those
  shapes are bound by dispatch latency rather than bandwidth. 8B
  Q4_K_M decode is now 92 percent of the reference's 52.3 tok/s and 8B
  Q8_0 103 percent of its 39.7. The second consequence is the larger
  one: no shader uses the extension, so the backend no longer requires
  `VK_KHR_shader_integer_dot_product` of a device, which is one fewer
  refusal between llmx and a card that lacks it.
  Twenty-ninth, what a kernel boundary costs and how many a decode pass
  has. The same tiny dispatch, timed four thousand times on the Radeon
  VII, under four barrier forms:

  | between dispatches | us each |
  |---|---:|
  | the barrier in the tree | 4.27 |
  | compute stages and access bits only | 4.18 |
  | execution dependency, no memory barrier | 3.05 |
  | nothing, incorrect and for the measurement only | 0.49 |

  A boundary costs about 3.8 us, of which 2.6 is the execution
  dependency itself, the device draining and relaunching, and 1.2 is
  the cache maintenance a shader write to shader read requires.
  Narrowing the barrier to the compute stages, which is all a dispatch
  needs, saves 0.1 us and does not pay for a second barrier flavour.
  Counting the dispatches of a decode pass on Qwen3-0.6B-Q8_0 gives
  256: nine per layer over 28 layers, being one norm-rope-and-cache
  write, four matmuls since q, k and v share a dispatch and gate and up
  share another, two RMS norms, one SiLU and one attention, plus the
  embedding, a gather, the final norm and the head. The residual adds
  are not among them; the matmul accumulates them already. At 3.8 us
  each that is 0.97 ms of a 4.88 ms token, a fifth of decode, and the
  nine are a strict chain, so none of it is idle time other work could
  fill. The way out is fewer boundaries rather than cheaper ones, and
  the two norms are the candidates, worth 0.21 ms if folded into the
  matmul that follows. What makes that awkward is the integer
  activations: a norm writes the 16-bit twin its consumer reads, so a
  matmul that normalised on the fly would have to quantise on the fly
  too, per 32-value block, in every workgroup that reads the row. Not
  attempted; the measurement is recorded so the next attempt knows what
  it is buying.

  Thirtieth, the row kernels by what their multiply and their registers
  cost, which closed the two decode cells that were short. The driver's
  disassembly showed one `v_mad_u64_u32` per product of a quant and an
  activation, a 32-bit integer multiply this chip runs at a quarter
  rate: 32 of the Q4_K kernel's 249 vector instructions and a third of
  its issue slots. It will not narrow that to the full-rate 24-bit form
  however the operands are written, and removing the weight side's dead
  sign extension changed the instruction count not at all. Every
  product on the nibble and K-quant paths is a non-negative quant of at
  most six bits against a 16-bit activation and no accumulator reaches
  the 16,777,216 a float counts exactly, so those dots multiply as
  floats and return the same integer at full rate; Q8_0's weights are
  signed and its blocks sum past that range, so it keeps the integer
  multiply. Worth 3.4 percent of 8B Q4_K_M decode with Q8_0 and every
  prefill cell flat.

  That a third of the issue slots bought 3 percent says these kernels
  are not issue bound, and at 390 GB/s of a thousand they are not
  bandwidth bound either. They are latency bound, and the register file
  is what limits the latency the chip can hide: three waves per SIMD on
  the K-quant kernels against five on the wide Q8_0 one. Eight of those
  registers are the batch columns a lane keeps, and a single-sequence
  decode uses one, so the column count is now specialization constant 0
  and each row kernel is built twice. The narrow build fits a fourth
  wave on Q4_0/Q4_1 (69 to 57 registers), Q4_K (73 to 63) and Q6_K (72
  to 62), and Q5_K misses it at 69. The wide Q8_0 kernel is excluded:
  its eight-column build is the one that is not register starved, the
  narrow build takes it from five waves to eight, and 8B Q8_0 decode
  fell 9 percent by it with its matmul going 338 to 367 ms of device
  time. The same kernel on the 0.6B files gained 12 percent, one work
  unit per lane there against four, so the direction follows the shape
  as well as the path and the larger model decides it.

  Both arms built from detached worktrees at their own commits, two
  interleaved blocks per cell, under the matched protocol. The
  reference column is the b11075 Vulkan figure recorded earlier on this
  card; that binary is not on this machine and was not re-run today:

  | model | test | reference b11075 Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | tg32 | 51.9 tok/s | 48.64 | 58.16 | 112% |
  | Qwen3-0.6B-Q5_K_M | tg32 | 219.8 tok/s | 205.43 | 227.00 | 103% |
  | Qwen3-8B-Q8_0 | tg32 | 38.8 tok/s | 41.62 | 41.40 | 107% |
  | Qwen3-0.6B-Q8_0 | tg32 | 195.1 tok/s | 213.97 | 214.21 | 110% |

  Prefill is flat on all four, within 0.4 percent. Both decode cells
  that were below the reference on this card now clear it. The whole
  suite passes on the device backend, the HF baseline included; one
  perf-floor run failed with the bench process exiting non-zero and did
  not reproduce in two further runs.

  The MI50 in the container, where the reference was re-run on the same
  card in the same session. That card takes the integer dot form, so
  none of the float multiply reaches it and all of the gain is the
  register change:

  | model | test | reference Vulkan | llmx before | llmx now | llmx share |
  |---|---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | tg32 | 36.75 tok/s | 65.6, 65.0 | 79.8, 79.8 | 217% |
  | Qwen3-8B-Q8_0 | tg32 | 25.65 | 45.7, 46.8 | 49.6, 45.6 | 182% |
  | Qwen3-0.6B-Q8_0 | tg32 | 100.59 | 270.1, 270.5 | 269.9, 270.4 | 269% |

  Prefill is flat there too, on both models measured as the control.

  **Correction (2026-09-22, thirty-fourth paragraph):** the reference column above was measured with all ten MI50s visible, so the reference split the model across ten cards. These are not same-card figures and its shares are void; the one-card comparison is in the thirty-fourth paragraph.
  Whether the float multiply beats the integer dot under Mesa is not
  measured; the two forms still compile from one source and the profile
  still chooses.

  Thirty-first, where prompt processing actually stands, which the
  figures quoted until now understated. The 43, 72 and 77 percent of
  the reference reported through the day are Q8_0 and 0.6B cells. On
  the MI50 at 247 rows against that card's own reference build, in the
  same session as the table above:

  | model | llmx pp247 | reference pp247 | llmx share |
  |---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 2653 tok/s | 3359 | 79% |
  | Qwen3-8B-Q8_0 | 265 | 606 | 44% |
  | Qwen3-8B-Q4_K_M | 98 | 530 | 19% |

  The Q4_K_M file prefills slower in absolute terms than the Q8_0 file
  of the same model on the same card, 98 tok/s against 265, while
  reading a little over half the bytes. So it is not bandwidth and not
  the tile shape: it is the tile kernel dequantizing K-quant weights
  into shared memory on every pass, which the 8-bit path does not pay.
  This is the largest gap open and is the next thing taken.

  **Correction (2026-09-22, thirty-fourth paragraph):** the reference column above was measured with all ten MI50s visible, so the reference split the model across ten cards. These are not same-card figures and its shares are void; the one-card comparison is in the thirty-fourth paragraph. The gap between the Q4_K_M and Q8_0 files is llmx's own and stands.

  Thirty-second, that gap closed. The tile kernel staged K-quant weights
  through the per-value decoders in `qdecode.glsl`, which re-read and
  re-unpack a block's packed sub-scale and sub-min for every value,
  three to five byte loads and the unpacking each time. A thread's run
  of values lies inside one group of 32 whichever tile height is built,
  since it stages 8 or 16 values starting at a multiple of that, so the
  sub-scale, the sub-min, which nibble half the run takes and the byte
  it starts at are invariant across the run and are read once. Then the
  loads: this driver issues one `buffer_load_ubyte` per byte and joins
  none of them, 71 of them in this kernel, so every byte now comes from
  a word. `qdecode.glsl` reaches its bytes through a `QBYTE(i)` macro
  rather than naming an array, so a shader serves them from whatever
  view it binds, and the tile kernel's byte binding went away rather
  than a word binding being added. It issues 86 dword loads and no byte
  loads, against 32 dword, 72 byte and 5 short before. Nothing about
  the arithmetic or the staged values changed, and the HF perplexities
  are identical to the digit.

  | device | model | rows | before | after | reference |
  |---|---|---:|---:|---:|---:|
  | Radeon VII | Qwen3-8B-Q4_K_M | 64 | 95.98 tok/s | 170.75 | - |
  | Radeon VII | Qwen3-8B-Q4_K_M | 247 | 96.51 | 240.31 | - |
  | Radeon VII | Qwen3-8B-Q4_K_M | 512 | 109.21 | 281.73 | - |
  | Radeon VII | Qwen3-0.6B-Q5_K_M | 247 | 1158.02 | 2381.28 | - |
  | Radeon VII | Qwen3-0.6B-Q5_K_M | 512 | 1072.31 | 2565.02 | - |
  | MI50 | Qwen3-8B-Q4_K_M | 64 | 93.0, 94.2 | 157.1, 157.6 | - |
  | MI50 | Qwen3-8B-Q4_K_M | 247 | 98.2, 98.5 | 243.8, 244.4 | 530.4 |
  | MI50 | Qwen3-8B-Q4_K_M | 512 | 113.1, 113.1 | 297.5, 297.7 | - |

  The 8-bit controls are flat on both cards, their staging source
  untouched: on the MI50 Qwen3-8B-Q8_0 reads 265.1 against 264.8 at 247
  rows and 327.0 against 327.1 at 512, and Qwen3-0.6B-Q8_0 is level
  inside its spread. So K-quant prompt processing went from 19 percent
  of the reference on that card to 46 (shares against the ten-card reference, corrected in the thirty-fourth paragraph), which is where the 8-bit path
  already was. Both cards pass every suite afterwards, the HF gate
  included.

  Thirty-third, three questions about prompt processing answered by measurement, each against the explanation I had given.

  Whether the tile wants occupancy. The 128-row tile runs two waves per SIMD, held there by 100 registers and 24,576 bytes of shared memory, and the 64-row tile runs three on 67 and 16,384. Forcing each on the Radeon VII, two interleaved blocks:

  | model | rows | 64-row tile | 128-row tile | shipped choice |
  |---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | 247 | 228.2 tok/s | 240.7 | 243.3 |
  | Qwen3-8B-Q4_K_M | 512 | 261.4 | 284.6 | 284.8 |
  | Qwen3-8B-Q8_0 | 247 | 245.8 | 281.0 | 280.9 |
  | Qwen3-8B-Q8_0 | 512 | 280.6 | 340.2 | 340.4 |

  The taller tile wins everywhere despite the lost wave, so work per thread is worth more than occupancy here, unlike the row kernels, and the shipped height choice is already right.

  Whether the host feeds the device in time. The card read as fully occupied at about three quarters of its power which raised whether prefill is recorded late. With runs short enough that the profiler samples every dispatch, the kernels' own execution times sum to the wall time: 345.7 against 345.2 ms for Qwen3-0.6B-Q8_0 at 512 rows, 3632.6 against 3632.9 for Qwen3-8B-Q4_K_M. A queue waiting on the host would leave time no kernel covers. None is left, so the device never waits and the shortfall is inside the kernels.

  What the reference does instead. On the MI50 its Vulkan build reports `int dot: 1` and `matrix cores: none`, and its quantized prefill path multiplies 8-bit activations through the four-wide integer dot, where ours multiplies dequantized floats one product per instruction on float tiles. That is a quarter of the instructions and a quarter of the shared memory per product. It is the next thing built, for devices whose integer dot is native, which the profile already records for the MI50 under Mesa; the precision of 8-bit activations against the HF bounds decides whether it ships.

  Also measured and not kept: the prefill attention kernel with its online softmax taken four or two keys at a time, one rescale per chunk rather than per key. Flat within one percent on Qwen3-0.6B-Q8_0 at 512 and 4096 rows and Qwen3-8B-Q4_K_M at 2048.

  Thirty-fourth, a correction to every MI50 reference figure above. The reference's Vulkan build uses every device it can see, and the Linux machine has ten MI50s, so `llama-bench -ngl 99` without `GGML_VK_VISIBLE_DEVICES` split the model across all ten. Measured back to back on Qwen3-0.6B-Q8_0, that is pp247 3364 and tg32 101.3 tok/s with ten visible against 6941 and 299.0 on one. The decode leads of 217, 182 and 269 percent, the earlier 2.4 times and the prefill shares were all against the ten-card split. The Radeon VII has one device, so its tables stand, and the per-shape TFLOPS comparison ran pinned to one device and stands. The ROCm arms of the table above are not known to have been pinned either and are unverified. The reference is now pinned to one card.

  The one-card comparison, llmx at `cb2eb5b` with the integer-dot tile against the reference's Vulkan build on the same MI50, interleaved, two passes, three cards in parallel with one model on each:

  | model | test | llmx | reference | llmx share |
  |---|---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | pp64 | 1098.9 tok/s | 4654.9 | 24% |
  | Qwen3-0.6B-Q8_0 | pp247 | 4698.6 | 6954.7 | 68% |
  | Qwen3-0.6B-Q8_0 | pp512 | 4871.1 | 6676.8 | 73% |
  | Qwen3-0.6B-Q8_0 | tg32 | 273.0 | 303.1 | 90% |
  | Qwen3-8B-Q4_K_M | pp64 | 249.5 | 260.0 | 96% |
  | Qwen3-8B-Q4_K_M | pp247 | 409.5 | 634.4 | 65% |
  | Qwen3-8B-Q4_K_M | pp512 | 498.0 | 763.4 | 65% |
  | Qwen3-8B-Q4_K_M | tg32 | 78.8 | 86.8 | 91% |
  | Qwen3-8B-Q8_0 | pp64 | 248.8 | 528.3 | 47% |
  | Qwen3-8B-Q8_0 | pp247 | 411.0 | 727.2 | 57% |
  | Qwen3-8B-Q8_0 | pp512 | 499.8 | 863.7 | 58% |
  | Qwen3-8B-Q8_0 | tg32 | 50.0 | 58.4 | 86% |

  Each arm's best pass is shown. So on the MI50 llmx trails in both phases: decode at 86 to 91 percent, prompt processing at 24 to 96. Decode on the MI50 is an open gate again, not a lead.

  Thirty-fifth, the integer-dot tile for every K-quant and Q8_0 loading a word at a time. Q8_0's 34-byte blocks start either on a word boundary or two bytes past one, so its staging loads one extra word and funnels where the block straddles, instead of two 16-bit loads per quant word. Q6_K scales each half of a 32-value group apart, so its own build sums the halves separately, and its offset of 32 goes into each staged byte. Q5_K is Q4_K plus a fifth bit. At the 8B feed-forward shape on one MI50, against the reference's integer-dot tile on the same card:

  | type | float tile | integer-dot tile | reference |
  |---|---:|---:|---:|
  | Q8_0 | 4.87 TFLOPS | 12.07 | 13.30 |
  | Q4_K | 4.65 | 11.44 | 11.42 |
  | Q6_K | 3.62 | 9.60 | 7.03 |

  One module for all four cost Q8_0 and Q4_K 4 percent through Q6_K's split sums, so Q6_K is its own module. Device allocations are now whole words, because a tensor with an odd block count ended two bytes into a word its 32-bit view could not reach.

  The one-card gate at `db249b8`, same protocol as the thirty-fourth paragraph, now with the Q5_K_M file:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q5_K_M | 2123 vs 2931 tok/s, 72% | 5225 vs 4433, 118% | 5676 vs 5723, 99% | 328.9 vs 312.4, 105% |
  | Qwen3-0.6B-Q8_0 | 1097 vs 4654, 24% | 5474 vs 6949, 79% | 5930 vs 6652, 89% | 272.4 vs 300.4, 91% |
  | Qwen3-8B-Q4_K_M | 283.4 vs 260.5, 109% | 557.3 vs 634.3, 88% | 666.7 vs 761.3, 88% | 78.6 vs 86.6, 91% |
  | Qwen3-8B-Q8_0 | 299.7 vs 528.3, 57% | 584.1 vs 733.8, 80% | 721.6 vs 863.4, 84% | 50.0 vs 58.4, 86% |

  These ran while the HF suite and a perplexity job used two other cards of the same machine, so the host was shared; both arms were interleaved on each card, and one llmx decode pass on Qwen3-0.6B-Q8_0 read 243.8 against 272.4 in the other and is reported rather than dropped. Every HF perplexity cell passes in both scoring modes on the MI50. On the 8B Q4_K_M file, 40 wikitext windows of 512 score mean NLL 2.4695 against the float tile's 2.47023.

  What is left on the MI50 is short prompts, where the 0.6B Q8_0 file reads a quarter of the reference at 64 rows, and decode at 86 to 91 percent except on the Q5_K_M file.

  Thirty-sixth, the thresholds. The crossover from the per-row kernel to the tile on the MI50 had been measured against the float tile, 96 rows on a narrow 8-bit projection. Forcing each kernel and sweeping against the integer-dot tile, one model per card:

  | model | row kernel wins up to | tile wins from | threshold before |
  |---|---:|---:|---:|
  | Qwen3-0.6B-Q8_0 | 24 rows | 32 | 96 |
  | Qwen3-0.6B-Q5_K_M | 32 | 48 | 64 |
  | Qwen3-8B-Q8_0 | 8 | 16 | 32 |
  | Qwen3-8B-Q4_K_M | 16 | 24 | 64 |

  So the measured profile carries four thresholds per device, 8-bit and other types each split at 4096 wide, and the MI50's row is 16, 32, 24 and 40; a row measured with the integer-dot tile applies only where the device has the integer dot. Q4_0 and Q4_1 went through the integer-dot tile too, so that the K-quant thresholds they share are the ones measured for it: Q4_0 folds its offset of 8 into each byte, and Q4_1 adds its minimum. The other types held within half a percent at the feed-forward shape. The one-card gate at `2b770f6`:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 2052 vs 4928, 42% | 5025 vs 7252, 69% | 5355 vs 6823, 78% | 328 vs 328, 100% |
  | Qwen3-0.6B-Q5_K_M | 2086 vs 2969, 70% | 5199 vs 4434, 117% | 5668 vs 5722, 99% | 334 vs 313, 107% |
  | Qwen3-0.6B-Q8_0 | 2223 vs 4638, 48% | 5452 vs 6950, 78% | 5918 vs 6659, 89% | 270 vs 299, 90% |
  | Qwen3-8B-Q4_K_M | 283 vs 261, 109% | 555 vs 634, 88% | 665 vs 761, 87% | 78 vs 86, 91% |
  | Qwen3-8B-Q8_0 | 300 vs 528, 57% | 585 vs 733, 80% | 721 vs 863, 84% | 50 vs 58, 86% |

  The HF suite ran on another card of the same machine meanwhile, and every cell passes in both modes. One margin narrowed: the Q4_0 fixture's continuous cell, scored in batched passes through 8-bit activations, is at an NLL delta of 0.139 against its 0.160 bound, where the float tile gave 0.131. Short prompts on the 0.6B files are what is left in prompt processing: 64 rows yields a single column tile, and the projections give too few row tiles to fill sixty compute units.

  Thirty-seventh, a third tile height. A 32-row build of each tile kernel, the second variant of the 64-row one, doubles the workgroups of a short prompt without reading a weight more often. Taken whenever the 64-row tile underfills, it gave the 0.6B files 11 to 23 percent at 48 to 64 prompt rows but cost the 8B files up to 5 percent at 96 to 128: profiled there, the 8B's 4096-wide k and v took 72.5 ms on the small tile against 51.5 on the middle one. The small tile does half the arithmetic per barrier, which a 1024-wide projection's 32 inner steps absorb and a 4096-wide one's 128 do not. So a narrow projection takes it whenever the middle tile underfills and a wide one only below half fill. Against the build before it on five MI50 cards, one model each, best of two interleaved passes:

  | model | pp32 | pp48 | pp64 | pp96 | pp128 | pp247 |
  |---|---:|---:|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | +1% | +23% | +20% | +11% | +8% | 0% |
  | Qwen3-0.6B-Q5_K_M | 0% | +15% | +11% | +5% | +3% | 0% |
  | Qwen3-0.6B-Q8_0 | +14% | +16% | +12% | +7% | +5% | 0% |
  | Qwen3-8B-Q4_K_M | +3% | +2% | -1% | -1% | 0% | 0% |
  | Qwen3-8B-Q8_0 | +3% | +2% | -2% | 0% | -1% | 0% |

  The 8B cells at 64 to 128 rows read 0 to +1 percent under the same rule in the run before, so their -1 and -2 are inside run-to-run spread. Qwen3-0.6B-Q8_0 at 64 rows is 2502 tok/s, 54 percent of the reference's 4638.

  Thirty-eighth, two quant blocks per barrier and where decode stands on the MI50. The integer-dot tile staged one 32-value block per step, two barriers per block and each block's loads waited for alone. Two blocks per step, on one MI50 at the 8B feed-forward shape: Q8_0 11.94 to 12.15 TFLOPS, Q4_K 11.35 to 11.63, Q6_K 9.48 to 10.02; the 8B files gain 2.8 to 4.8 percent at 247 and 512 prompt rows, the 0.6B files 4 to 4.5 at 64, and 8B Q8_0 at 64 rows loses 4.3. Four blocks per step lost 10 to 23 percent, the larger shared arrays costing occupancy. A first build of this failed the MI50's Q4_0 check because a local variable in the Q4_0 and Q6_K staging shadowed the new block index; the Radeon VII never takes this path, so its tests cannot catch such a bug, and a change to this kernel is checked on the MI50 before it is committed.

  Decode, matrix-vector at an 8B down projection, 4096 rows over 14336 inputs, on one MI50 against the reference's per-operation benchmark on the same card:

  | type | llmx | reference | llmx share |
  |---|---:|---:|---:|
  | Q8_0 | 162.5 us | 117.9 | 73% |
  | Q4_K | 124.5 | 56.7 | 46% |
  | Q6_K | 170.9 | 126.4 | 74% |

  The reference's decode matvec quantizes activations to 8 bits and multiplies through the four-wide 8-bit dot; ours reads the 16-bit twin through the two-wide 16-bit dot, twice the dot instructions and twice the activation bytes per weight. The float-multiply row build, which the Radeon VII runs, was tried on the MI50 in place of the integer-dot build and is 10 to 24 percent slower there on the 8B shapes, so it is not the answer. The next piece is an 8-bit twin for the integer-dot devices: every producer of decode activations writes it and every row family's integer-dot build reads it, with Q6_K's offset of 32 and Q4_0's of 8 folded into the weight bytes as the tile does. Batched HF scoring through 8-bit activations on the MI50 put the Q8_0 continuous cell at 0.0023 against its 0.010 bound, so the precision is not expected to be what stops it.

  Thirty-ninth, an 8-bit activation twin for decode on the integer-dot devices. The row kernels read activations as a 16-bit twin through the two-wide 16-bit dot. The reference's decode reads 8-bit activations through the four-wide 8-bit dot, and a probe reading 8-bit on the Q4_K and Q5_K row kernel alone took the Q4_K matvec at 4096 x 12288 from 92.4 to 75.8 us. Putting every family on it measured this way on one MI50, 16-bit against 8-bit:

  | matvec | 16-bit twin | 8-bit twin |
  |---|---:|---:|
  | Q4_0, 4096 x 12288 | 135.1 us | 74.7 |
  | Q4_1, 12288 x 4096 | 137.1 | 82.8 |
  | Q4_K, 14336 x 4096 | 123.4 | 92.8 |
  | Q5_K, 4096 x 12288 | 107.0 | 93.2 |
  | Q6_K, 4096 x 12288 | 150.3 | 137.6 |
  | Q8_0, 4096 x 12288 | 153.7 | 169.5 |

  Q8_0 was slower on it in every loop shape tried: its eight-bit build fell from 40 registers to 32 and ran eight waves per SIMD, the over-occupancy the one-column build also caused it, and loading two pairs per iteration to use the registers recovered the large shapes but cost the small ones. So on such a device the producers write both twins, the 8-bit one after the 16-bit one at a 256-byte offset, and each family reads the one it is fastest and precise enough on. The HF gate then decided which. With Q4_0 or Q6_K on the 8-bit twin the Q4_0 fixture, whose only K-quant is its tied Q6_K output head, ranked a different fifth token for "The capital of France is", a top-5 overlap of 3 against its frozen 4. The bound was not moved, so Q4_0, Q4_1, Q6_K and Q8_0 read the 16-bit twin and Q4_K and Q5_K the 8-bit one.

  Writing a twin nobody reads cost the small models: the producers with the 8-bit writer behind a runtime branch took Qwen3-0.6B-Q4_0 decode down 2.6 percent without ever taking it. So the 8-bit writer is a second build of each producer, specialization constant 7, which the backend dispatches only once a matmul that reads the 8-bit twin has run; the one pass before that makes it through the fallback quantizer. Decode on one MI50, best of two interleaved passes, reference pinned to one card:

  | model | before | now | reference | llmx share |
  |---|---:|---:|---:|---:|
  | Qwen3-8B-Q4_K_M | 78.3 tok/s | 85.5 | 86.3 | 99% |
  | Qwen3-0.6B-Q5_K_M | 333.5 | 337.4 | 305.7 | 110% |
  | Qwen3-0.6B-Q4_0 | 331.2 | 330.7 | 324.5 | 102% |
  | Qwen3-0.6B-Q8_0 | 273.4 | 270.1 | 299.4 | 90% |
  | Qwen3-8B-Q8_0 | 50.0 | 50.0 | 57.4 | 87% |

  Every suite passes on the MI50 and the Radeon VII, the HF baseline in both scoring modes. On the 8B Q4_K_M file ten wikitext windows of 512 scored one token at a time move from mean NLL 2.69356 to 2.69409. The backend test feeds each family's reference the activations of the twin it reads; against the 8-bit twin one quant can round the other way on the device, whose reciprocal is a few ulps from the host's, worth about the weight times the block's step, so those comparisons take 1e-2 where an indexing error is worth the output itself. The Radeon VII has no native integer dot and is unchanged. The decode gap left on the MI50 is the two Q8_0 files, and Q6_K, which the gate keeps on the 16-bit twin.

  Fortieth, the one-card gate at `300d812`, same protocol as the thirty-fourth paragraph, llmx share of the reference's Vulkan build on the same MI50, best of two interleaved passes:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 2512 vs 4910, 51% | 5084 vs 7250, 70% | 5403 vs 6832, 79% | 328 vs 331, 99% |
  | Qwen3-0.6B-Q5_K_M | 2443 vs 2959, 83% | 5197 vs 4433, 117% | 5643 vs 5725, 99% | 337 vs 314, 107% |
  | Qwen3-0.6B-Q8_0 | 2592 vs 4652, 56% | 5494 vs 6960, 79% | 5964 vs 6667, 89% | 274 vs 303, 90% |
  | Qwen3-8B-Q4_K_M | 288 vs 260, 111% | 574 vs 634, 91% | 692 vs 761, 91% | 86 vs 87, 99% |
  | Qwen3-8B-Q8_0 | 290 vs 526, 55% | 604 vs 732, 83% | 742 vs 862, 86% | 50 vs 59, 85% |

  The largest gaps left are short prompts on every file but the 8B Q4_K_M, and Q8_0 in both phases. The reference's decode matvec loads each activation word once per thread and reuses it across several rows, where a lane of ours serves one row and loads the activations again for every row; two rows per lane cluster is the next piece for Q8_0 decode.

  Tried and not kept: two adjacent rows per cluster in the wide Q8_0 kernel, each activation word loaded once for both rows. On the Radeon VII, interleaved three times, it is 20 to 30 percent slower at every shape, 133 to 167 us at 4096 x 12288: the kernel went from 42 registers to 51 and its workgroups halved, and for this kernel that costs more than the shared loads save.

  Also not kept: the wide Q8_0 kernel on the 8-bit twin with its occupancy capped. It lost to the 16-bit twin at 32 registers and eight waves per SIMD, and an unused 16 KB of shared memory brings it to four waves; on one MI50 at 4096 x 12288 that reads 172 us against 160 uncapped and 151 on the 16-bit twin, twice over. So occupancy is not why Q8_0 loses on the 8-bit twin, and Q8_0 decode stays on the 16-bit one.

  Documentation review at this checkpoint: every Markdown file read against the code, CLI, tests and build. About fifty stale claims corrected across README, AGENTS, STATUS's table and feature blocks, VULKAN, USAGE, ROADMAP, ARCHITECTURE, EXECUTION, KV-CACHE, DEVICE-EXECUTION, SERVER, CI, ASSETS and the per-source pages. The largest were the ten-card MI50 figures presented as current in README and the status table, VULKAN.md saying no shader uses the integer dot, the tile described as two heights and two thresholds, the perplexity scorer described as one token at a time, and ROADMAP and ARCHITECTURE still calling the server and the Vulkan backend planned. The numbered measurement paragraphs above are left as history.

  And a memory fix. A model on a device backend held every weight twice: the loader reads the file into one host allocation, the model kept it for its lifetime, and the device backend copies each weight into its own memory. The model now records at adoption whether any weight still reads those bytes in place, and the CLI releases them when none does. Qwen3-8B-Q4_K_M on the Radeon VII, steady host memory 4.62 to 0.18 GB, decode unchanged; the CPU backend adopts by aliasing and keeps them. The peak is still the whole file, 4.84 GB, since it is read before the upload. Streaming the file to the device during the load, read directly into staging and uploaded asynchronously so the disk and the copies overlap, would remove that peak and is not done.

  Forty-first, short prompts through the integer-dot tile. A profile of 8B Q8_0 at 64 rows put 53 percent of device time in the 64-row tile, and timing single calls on one MI50 at 64 columns showed where. The down projection, 4096 x 12288, took 2970 us, 2.2 TFLOPS, where the gate projection, 12288 x 4096, read 6.2. Splitting the inner dimension across more workgroups, tried first on its own, made most shapes slower: 4096 x 4096 went from 617 to 896 us. So the calls were not short of workgroups but of memory locality.

  The cause was the 8-bit activations' order. They were stored column after column, so each step of the tile read its 64 columns a row width apart. They are now ordered by block of the inner dimension, then by column, which makes a step's activations one 2 KB run. That alone takes the down projection to 1299 us, 4096 x 4096 to 499, and the 512-column feed-forward shape from 9.56 to 11.0 TFLOPS. Perplexity is bit-identical.

  With that order in place, splitting helps. A call with fewer workgroups than eight per compute unit splits its inner dimension into parts of at least 16 quant blocks. Each part writes partial sums to a scratch buffer, and `matmul_reduce.comp` adds them in part order. The target was swept at 2, 4 and 8 per compute unit on all five files, and eight wins or ties everywhere except 0.6B Q8_0 at 512 rows, which loses 3 percent. Together, the down projection goes to 731 us and 4096 x 4096 to 354.

  Split and unsplit outputs agree to 4e-7 relative at every shape probed. The re-quantization to 8 bits downstream turns that reordering into a few rounding flips, so 20 windows of 512 score 13.6049 against 13.5942 on 8B Q8_0, and 40 windows of 64 score 67.310 against 67.474 on 0.6B Q8_0. The HF gate passes every cell both ways, and the backend test passes on the MI50, where its 1024-wide calls now split in two.

  The one-card gate at this change, same protocol as the fortieth paragraph:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 3744 vs 4795, 78% | 5610 vs 6962, 81% | 5590 vs 6695, 83% | 310 vs 308, 101% |
  | Qwen3-0.6B-Q5_K_M | 3553 vs 2917, 122% | 5543 vs 4380, 127% | 5712 vs 5609, 102% | 310 vs 290, 107% |
  | Qwen3-0.6B-Q8_0 | 3381 vs 4600, 73% | 5973 vs 6867, 87% | 6011 vs 6585, 91% | 262 vs 292, 90% |
  | Qwen3-8B-Q4_K_M | 618 vs 261, 237% | 767 vs 630, 122% | 812 vs 755, 107% | 84 vs 84, 101% |
  | Qwen3-8B-Q8_0 | 677 vs 526, 129% | 819 vs 732, 112% | 866 vs 873, 99% | 50 vs 56, 90% |

  Both 8B files now clear the reference in prompt processing, except Q8_0 at 512 rows, which is level. What remains below it is 0.6B Q4_0 and Q8_0 prompt processing and Q8_0 decode. `bench --profile` now also prints the driver's register, shared memory and waves-per-SIMD figures for each kernel that ran.

  Forty-second, the prefill attention tile and batch invariance. At 512 rows the causal attention tile was 28 percent of Qwen3-0.6B-Q8_0's device time on an MI50, about 0.6 TFLOPS. It finished every score in every lane with three shuffles, and took two exponentials and a rescale of the accumulator per token. It now takes a tile of 16 keys as a unit. Scores come eight tokens at a time, and a butterfly over the row's lanes finishes and deals them out one to a lane, 7 shuffles per 8 tokens. Then one maximum, one exponential per score, and one rescale per tile. Lanes own interleaved dimensions, so a row reads a staged token as contiguous 128-byte runs. On one MI50, 0.6B Q8_0 goes from 6029 to 7513 tok/s at 512 rows and from 3267 to 5566 at 2048, and 8B Q8_0 from 668 to 809 at 2048; on the Radeon VII 0.6B Q8_0 gains 3.5 percent at 512 rows and 12 at 2048. Finishing a whole tile's scores at once held 208 registers and one wave per SIMD, and a 32-key tile left one workgroup per compute unit; both were slower.

  Its output is within 1.3e-6 of the CPU's, as the old tile's was, yet the server test failed: a request reusing a cached prefix gave different greedy text from the CLI over the whole prompt. Two causes. The first was the test. Its server was started without the device and cache flags, since the helper that adds them looked at the executable path rather than the command, so every device suite had compared a CPU server with a device CLI. The CLI side also ran the default f16 cache where the server side asked for f32. On this prompt the first token is a near-tie, margin 0.26 on the CPU and 0.15 on the old tile, and the new tile tipped it.

  The second was real: a row's result depended on its batch. The row kernels and the tiles round differently and the kernel was chosen by the call's width, so a prompt's reused-prefix tail took the row kernel where one pass took the tile; the integer-dot tile's split followed the call's workgroups; and the per-row attention split its history by the dispatch's longest row and not at all past 256 (row, head) pairs.
  - A batch entry now carries its extent, the position one past its prompt's last token or 1 for a generated token, and the model passes calls their rows as runs (`backend::RowRun`). A row takes the tile when its extent reaches the tile threshold, and a call that mixes kernels becomes one call per kernel.
  - The tile's split is the one a pass over the row's whole prompt takes, up to 512 rows. Taking it from the shape alone as if every call were one column tile was also invariant, but cost 8B Q8_0 at 512 rows 866 to 802 tok/s.
  - Attention takes the tile by the view's extent, and the per-row kernel splits a row's history in parts of 32 tokens, doubling until at most 64 cover it, from the row's length alone.

  `backend-vulkan` checks this bitwise at 2048 and 6144 outputs over 249 rows, where the whole prompt takes the tallest tile and its tail the shortest. At the model API, one pass, a forked reused prefix and a prefix-then-tail pass give bit-identical logits on the MI50. The server test now runs its server on the device with f32 caches on both sides, and passes on both cards.

  Two things did not hold. Every prompt row through the tile, which would have kept many short prompts on one tile call, failed the HF gate on the short-prompt fixtures (a top-5 overlap of 3 against 4, and the exact F32 logits) and halved a 5-token prompt, 769 to 394 tok/s. The row kernel looping over the call's columns inside one dispatch took 24-row prompts from 1063 to 1202 tok/s but slowed the Q8_0 matvec from 167 to 209 us at 14336 x 4096 with the same registers, so it is not kept. The cost that stays is first-token time for many short prompts arriving together: 16 concurrent requests on 0.6B Q8_0 wait 134 to 138 ms at the median where they waited 49 to 102, since each prompt now takes the row kernel it takes alone rather than joining one tile call. Throughput and inter-token latency are unchanged.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4015 vs 4915, 82% | 6341 vs 7217, 88% | 6855 vs 6821, 101% | 322 vs 328, 98% |
  | Qwen3-0.6B-Q5_K_M | 3853 vs 2976, 129% | 6342 vs 4448, 143% | 7033 vs 5750, 122% | 341 vs 314, 109% |
  | Qwen3-0.6B-Q8_0 | 4006 vs 4651, 86% | 6865 vs 6980, 98% | 7504 vs 6690, 112% | 272 vs 297, 91% |
  | Qwen3-8B-Q4_K_M | 632 vs 261, 242% | 801 vs 633, 126% | 872 vs 763, 114% | 87 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 690 vs 527, 131% | 848 vs 736, 115% | 914 vs 865, 106% | 50 vs 58, 86% |

  Prompt processing now clears the reference on every file at 512 rows and on both 8B files everywhere. Below it: 0.6B Q4_0 and Q8_0 at 64 and 247 rows, decode on the 8-bit files at 86 and 91 percent, and many concurrent short prompts' first token.

  Forty-third, a layer's projections of one type in one integer-dot tile dispatch. At 64 columns on one MI50 a 0.6B layer's q, k and v took 232 us as three tile calls and 89 us as one call over their 4096 rows, and gate and up 177 against 120: each projection alone left most of the device idle. The tile now takes up to three projections of one type as the row kernel does, their row tiles on consecutive workgroups, and a split call's reduce adds all their parts in one dispatch. A reduce workgroup that straddled two projections indexed their output buffers non-uniformly and lost the second one's writes; each projection now has whole workgroups of its own, which a new check of three grouped projections at 64 and 249 columns caught. The split is the fused group's, so a row still sums the same parts however its prompt is batched. Qwen3-0.6B-Q8_0 at 64 rows goes from 3754 to 5047 tok/s; short prompts, decode and server load are unchanged.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4917 vs 4883, 101% | 6858 vs 7224, 95% | 7452 vs 6795, 110% | 323 vs 331, 98% |
  | Qwen3-0.6B-Q5_K_M | 4528 vs 2971, 152% | 6692 vs 4436, 151% | 7455 vs 5747, 130% | 337 vs 314, 107% |
  | Qwen3-0.6B-Q8_0 | 5036 vs 4654, 108% | 7383 vs 6974, 106% | 8093 vs 6692, 121% | 271 vs 298, 91% |
  | Qwen3-8B-Q4_K_M | 696 vs 261, 267% | 848 vs 634, 134% | 894 vs 763, 117% | 86 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 760 vs 524, 145% | 889 vs 735, 121% | 936 vs 865, 108% | 50 vs 58, 86% |

  Prompt processing clears the reference in every cell but 0.6B Q4_0 at 247 rows, 95 percent. Decode is the gap now: 86 and 91 percent on the Q8_0 files, 97 and 98 on 8B Q4_K_M and 0.6B Q4_0.

  Forty-fourth, a Q8_0 decode kernel over the 8-bit twin. The reference's Q8_0 matvec at 4096 x 14336 on one MI50 takes 120.8 us, about 516 GB/s, where ours took 167 us at the same shape. Ours read the 16-bit twin, so every row's lanes read the whole activation vector at two bytes a value, and its wide path spread each load instruction over 16-byte pieces of every block pair; on the 8-bit twin that path was slower still. `matmul_vec_q8.comp` gives a subgroup two rows. Lane l covers quarter l % 4 of every (S / 4)-th block, so a step reads a contiguous run of each row, and loads its eight 8-bit activation values per column once for both rows, two four-wide dots per quarter. The matvec goes to 135 us; one row per subgroup read 151 and four 146. It is the row kernel for Q8_0 wherever the integer dot is native, so short prompts and a server's many decode rows take it too: a 5-token prompt on 0.6B Q8_0 goes from 774 to 1191 tok/s, and 16 concurrent requests from 583 to 625 tok/s to 875 to 1085, their first token at the median from 130 to 142 ms to 56 to 71. The HF gate passes every cell, and the backend test's references now take the 8-bit rounding for Q8_0 rows on such a device.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4956 vs 4878, 102% | 6866 vs 7230, 95% | 7446 vs 6815, 109% | 325 vs 328, 99% |
  | Qwen3-0.6B-Q5_K_M | 4489 vs 2867, 157% | 6712 vs 4447, 151% | 7441 vs 5755, 129% | 342 vs 314, 109% |
  | Qwen3-0.6B-Q8_0 | 5063 vs 4636, 109% | 7383 vs 6978, 106% | 8020 vs 6662, 120% | 325 vs 300, 109% |
  | Qwen3-8B-Q4_K_M | 697 vs 260, 268% | 848 vs 634, 134% | 896 vs 764, 117% | 86 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 757 vs 525, 144% | 889 vs 735, 121% | 934 vs 864, 108% | 60 vs 58, 103% |

  Three cells remain below the reference: 8B Q4_K_M decode at 97 percent, 0.6B Q4_0 decode at 99, and 0.6B Q4_0 prompt processing at 247 rows at 95.

  Forty-fifth, Q4_0 and Q4_1 staging in the integer-dot tile. A block's staged words w and w + 4 are the low and high nibbles of one source word, and two threads each loaded it and, for Q4_0, funnelled it. A thread that stages two or more words now takes source words whole and stages both halves. 0.6B Q4_0 prompt processing goes from about 6879 to 7285 tok/s at 247 rows and from 7467 to 7922 at 512.

  Tried and not kept: a Q4_K decode kernel on the Q8_0 kernel's plan, a subgroup on two rows. With one nibble word a lane it read 4 bytes of weights against 16 of header and 24 of activations per step, and the 14336 x 4096 matvec took 201 us against the row kernel's 92. With sixteen bytes a lane it took 103, and 8B Q4_K_M decode read 80 tok/s against 84, so two rows sharing their activations does not pay for Q4_K here. The reference's Q4_K matvec at that shape takes 61 us and its Q6_K 132 against our 169; that file's output head is Q6_K, about 510 MB a token, which is where its decode gap is.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 4998 vs 4881, 102% | 7277 vs 7221, 101% | 7918 vs 6840, 116% | 321 vs 330, 97% |
  | Qwen3-0.6B-Q5_K_M | 4488 vs 2959, 152% | 6656 vs 4445, 150% | 7428 vs 5751, 129% | 339 vs 312, 109% |
  | Qwen3-0.6B-Q8_0 | 5095 vs 4659, 109% | 7360 vs 6976, 105% | 8043 vs 6698, 120% | 326 vs 299, 109% |
  | Qwen3-8B-Q4_K_M | 696 vs 261, 267% | 846 vs 634, 133% | 894 vs 764, 117% | 86 vs 89, 97% |
  | Qwen3-8B-Q8_0 | 759 vs 526, 144% | 888 vs 736, 121% | 934 vs 864, 108% | 61 vs 58, 104% |

  Prompt processing clears the reference in every cell. Decode on 8B Q4_K_M and 0.6B Q4_0 is at 97 percent, the one gap left.

  Forty-sixth, the RMS norm over several workgroups a row. A decode profile put `rms_norm_rows` at 18 percent of 8B Q4_K_M's device time, and alone on an MI50 one 1024-wide row took 15 to 16.6 us against 4.2 for an empty dispatch. One workgroup took the row, and on such a device it also writes the 8-bit twin, a chain of dependent shuffles per value, so a 4096-wide row was sixteen serial steps on one compute unit. Now a row's 256-value chunks each have a workgroup; every one sums the whole row's squares in the same order, so all reach the same scale bit for bit, and writes its own chunk and its twin. When dst and src overlap, as before the output head, the row keeps one workgroup, since a chunk's output would change what another is summing.

  Decode goes from 331 to 351 tok/s on 0.6B Q4_0, from 85 to 94 on 8B Q4_K_M and from 58 to 65 on 8B Q8_0. Reducing the sum of squares through subgroups rather than the shared-memory tree took the norm's barriers from eight to one but added in another order, and the HF gate's Q8_0 fixture then ranked a different fifth token on one prompt, a top-5 overlap of 4 against its frozen 5; the tree stays and the norm's arithmetic is unchanged.

  Also tried and not kept, all on one MI50: a Q6_K decode kernel with a subgroup on two rows sharing their 16-bit activations, 175 us at 14336 x 4096 against the row kernel's 169; Q6_K with sixteen lanes a block, 225; and Q4_K with four lanes a block, 134 against 92. None of the row kernel's Q4_K or Q6_K lane layouts beats the one it has.

  The one-card gate at this change, same protocol:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 5037 vs 4889, 103% | 7268 vs 7223, 101% | 7901 vs 6836, 116% | 340 vs 329, 103% |
  | Qwen3-0.6B-Q5_K_M | 4506 vs 2985, 151% | 6676 vs 4429, 151% | 7427 vs 5754, 129% | 359 vs 314, 115% |
  | Qwen3-0.6B-Q8_0 | 5108 vs 4653, 110% | 7361 vs 6978, 105% | 8028 vs 6688, 120% | 345 vs 299, 116% |
  | Qwen3-8B-Q4_K_M | 698 vs 261, 268% | 837 vs 633, 132% | 886 vs 764, 116% | 96 vs 89, 108% |
  | Qwen3-8B-Q8_0 | 762 vs 526, 145% | 879 vs 735, 120% | 925 vs 865, 107% | 65 vs 58, 111% |

  Every cell of the one-card MI50 gate clears the reference. The thinnest is 0.6B Q4_0 at 247 rows, 101 percent, within what one run can move.

  Forty-seventh, the gate on both platforms. The user set the device gate as llmx against llama.cpp's own Vulkan backend, the same backend type, on the Linux MI50 and on the Windows Radeon VII. A second MI50 run at `8c07a29`, four interleaved passes per file on cards 3 to 7, best of each arm:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 5044 vs 4883, 103% | 7259 vs 7104, 102% | 7869 vs 6739, 117% | 325 vs 318, 102% |
  | Qwen3-0.6B-Q5_K_M | 4512 vs 2932, 154% | 6651 vs 4362, 152% | 7385 vs 5677, 130% | 348 vs 303, 115% |
  | Qwen3-0.6B-Q8_0 | 5006 vs 4645, 108% | 7318 vs 6912, 106% | 8004 vs 6611, 121% | 331 vs 290, 114% |
  | Qwen3-8B-Q4_K_M | 695 vs 261, 267% | 831 vs 633, 131% | 874 vs 761, 115% | 93 vs 84, 110% |
  | Qwen3-8B-Q8_0 | 759 vs 524, 145% | 881 vs 732, 120% | 926 vs 863, 107% | 65 vs 57, 114% |

  The Radeon VII against llama.cpp b11075's Vulkan backend on the same card, two interleaved passes of five repetitions, best of each arm:

  | model | pp64 | pp247 | pp512 | tg32 |
  |---|---:|---:|---:|---:|
  | Qwen3-0.6B-Q4_0 | 1040 vs 952, 109% | 2241 vs 670, 335% | 2528 vs 1825, 138% | 228 vs 224, 102% |
  | Qwen3-0.6B-Q5_K_M | 982 vs 647, 152% | 2395 vs 526, 455% | 2613 vs 1174, 223% | 228 vs 219, 104% |
  | Qwen3-0.6B-Q8_0 | 1019 vs 922, 110% | 2869 vs 653, 440% | 3111 vs 1655, 188% | 215 vs 194, 111% |
  | Qwen3-8B-Q4_K_M | 175 vs 58, 300% | 239 vs 78, 308% | 280 vs 111, 252% | 60 vs 52, 115% |
  | Qwen3-8B-Q8_0 | 221 vs 80, 275% | 278 vs 99, 281% | 338 vs 147, 230% | 42 vs 39, 107% |

  The 0.6B Q4_0 decode cell was rerun four times alone, 228 to 239 against 223.5 tok/s; llmx decodes after a 247-token prompt and the reference from an empty one. Every cell on both platforms clears the reference.

  For information, not the gate: the mx fork `eefc4e732` for gfx906 with ROCm, pinned to one MI50, reads ahead of llmx's Vulkan backend on the Q8_0 files, 8B Q8_0 at 512 rows 1301 tok/s against 904 and its decode 71 against 66, and behind it on the 4- and 5-bit ones, 0.6B Q5_K_M at 64 rows 896 against 4388. A ROCm backend is where that comparison belongs.
- **Left:** the device gate passes on both platforms (the forty-seventh paragraph), the thinnest cells at 102 percent; the Q6_K and Q4_0 matvecs are still 1.28 and about 2.3 times the reference's; loading, which reads the whole file into host memory before uploading it; folding a layer's
  two RMS norms into the matmul that follows, worth a fifth of the
  barrier time measured in the twenty-ninth; the
  prompt pass at 32 to 128 rows (the twenty-seventh paragraph): the
  tile kernel's cost per tile, the step at exactly 64 rows, and
  splitting the tiled attention over a long history. On 0.6B the bound is the
  dispatch count, not a kernel; replaying a recorded pass (the server
  block above) is the lever there. On 8B the K-quant paths sit at 250
  to 256 GB/s against Q8_0's 400 with their loads and their sub-scale
  decode already trimmed; what remains is the per-lane work of the
  nibble unpacking itself, and 8-bit activations would buy another
  tenth at a numerical cost the CPU experiment measured near the bound. The same backend now runs on the Linux machine's MI50s
  under Linux through `docker/Dockerfile`, which needs no change to that
  shared machine: Debian's own Mesa driver enumerates all ten cards
  through `/dev/dri` and its own shader compiler is new enough, so the
  image carries the driver, the loader, the headers and the compiler and
  the host carries nothing. The whole CTest suite passes there with
  `backend-vulkan` running rather than skipping, its 1,172,518 kernel
  outputs matching the CPU backend on an MI50, which is the first device
  other than the Radeon VII to run these kernels. Reported bandwidth is
  lower than the Radeon VII's under the Windows driver (Q8_0 at
  4096 x 12288 reads 315 GB/s against 397, Q6_K's wide head 248 against
  232 the other way), untuned and not a like-for-like comparison, since
  the driver differs as well as the card.

  A real model followed. `llmx pull` fetched Qwen3-0.6B-Q8_0 on the Linux machine
  through the image's curl, at the same revision the Windows copy has,
  and the whole Python suite passes there on `vulkan:0`, the HF gate
  included: the continuous excerpt reads an NLL delta of 0.001264
  against a 0.010 bound and the windowed cases 0.002393 to 0.012396
  against 0.020, the same numbers the Radeon VII returns. That is the
  external correctness gate met on a second device.

  Performance there was measured against three reference arms on the
  same card, which is worth doing because they disagree by more than
  llmx does. Qwen3-0.6B-Q8_0, two passes each, five runs per point,
  arms interleaved in the same minutes. The reference's Vulkan build is
  the same commit as its ROCm one, `7ab4ee7ba` (build 11100), compiled
  in the dev container; the fork is mx-llama.cpp `eefc4e732` built for
  gfx906, the arm the gate in `AGENTS.md` names. None of these is the
  b11075 Vulkan build the Radeon VII tables above use, so the tables
  are not a series.

  | test | reference Vulkan | reference ROCm | mx-llama.cpp ROCm | llmx Vulkan |
  |---|---:|---:|---:|---:|
  | pp64 | 2009, 2016 tok/s | 1774, 1980 | 4549 | 830, 826 |
  | pp256 | 3453, 3474 | 5419, 5434 | - | 1430, 1477 |
  | pp512 | 3550, 3560 | 6087, 6114 | 6782 | 1783, 1783 |
  | tg64 | 102.6, 101.3 | 226.4, 229.6 | 229.8 | 246.2, 246.2 |

  Read against the like-for-like arm, the reference's own Vulkan
  backend on the same card and driver, llmx decodes 2.4 times faster
  and prompt-processes at 41 to 50 percent. Decode is the striking
  figure: the reference's Vulkan decode collapses to 102 tok/s here
  while its ROCm decode holds 226, and llmx's Vulkan decode reaches
  246, past the vendor path. **Correction (2026-09-22):** the reference arms in this table ran with all ten MI50s visible and split the model across them, so neither its decode collapse nor the 2.4 times is a same-card result; see the thirty-fourth paragraph. That says the reference's Vulkan backend
  is far more driver-sensitive than ours, since on the Radeon VII under
  the Windows driver the same comparison is 205 against about 199.
  Prompt processing is where llmx is behind on this card by every arm:
  50 percent of the reference's Vulkan, 29 of its ROCm, 26 of the
  fork's. The fork matters: it reads 4549 tok/s at a 64-token prompt
  against upstream's 1774, so a share quoted against upstream flatters
  llmx, and it is the arm the gate names where it is available.

  Why prompt processing is behind, from the MI50 figures alone, since
  those are the ones taken in a single environment. At pp512 there the
  reference's Vulkan reads 3555 tok/s and llmx 1783. The model's
  prompt pass is about 450 GFLOP at 512 tokens, so llmx runs at 1.57
  of this card's 13.4 fp32 teraflops, 12 percent of peak, and the
  reference at 23. The weights are read once per pass, 640 MB in 287
  ms, nowhere near the card's bandwidth, and the 256 dispatches of a
  pass cost under a millisecond of it, so neither arm is bound by
  memory or by launch overhead. What does bind `matmul_tile.comp` is
  shared-memory traffic per operation: at a 4 by 4 micro-tile a thread
  reads eight floats per sixteen multiply-adds, two operations per
  read, and at this card's shared-memory bandwidth that ratio lands
  close to the 12 percent measured. The work is to raise the
  operations per read, which means a wider micro-tile.

  Done, and the shape is now chosen rather than fixed. A thread
  accumulates `TILE_ROWS / 16` rows by four columns, and `TILE_ROWS` is
  a specialization constant, so one SPIR-V module builds both a 64-row
  and a 128-row pipeline and the backend picks per dispatch. The taller
  tile reads twelve values from shared memory per thirty-two products
  where the shorter reads eight per sixteen, two thirds of the traffic
  for the same work; it also halves the workgroups, so a call whose
  taller form would produce fewer groups than the device has compute
  units takes the shorter one. Compute units come from
  `VK_AMD_shader_core_properties` where the driver has it, and a
  deliberately small assumption otherwise, which prefers the shorter
  tile and starves nothing. Measured on the Radeon VII, three runs a
  point, interleaved, the two fixed heights against the choice:

  | case | 4 by 4 fixed | 8 by 4 fixed | chosen |
  |---|---:|---:|---:|
  | 0.6B pp64 | 837 tok/s | 732 | 844 |
  | 0.6B pp128 | 1814 | - | 1919 |
  | 0.6B pp256 | 1473 | 2344 | 2772 |
  | 0.6B pp512 | 1795 | 2820 | 2821 |
  | 8B pp256 | 243 | 262 | 265 |
  | 8B pp512 | 267 | 309 | 309 |

  Choosing beats both fixed heights everywhere, and beats the taller
  one at 256 rows because the projections of a dispatch differ: a
  1024-row projection and a 3072-row one do not want the same tile.
  Decode is untouched, the row kernel taking those batches, and reads
  197 to 205 tok/s across runs either way, which is this session's
  drift.

  Verified where the reference comparison lives, all three arms on the
  MI50 in one container, five runs a point, two passes:

  | test | reference Vulkan | llmx before | llmx after |
  |---|---:|---:|---:|
  | pp64 | 2027 tok/s | 828 | 831 |
  | pp256 | 3474 | 1454 | 2253 |
  | pp512 | 3555 | 1783 | 2619 |
  | tg64 | 101.6 | 246 | 246 |

  Prompt processing goes from 50 to 74 percent of the reference's
  Vulkan at 512 rows and from 42 to 65 at 256; 64 rows is unchanged at
  41 percent, that call being too small to fill the card either way,
  and decode stays 2.4 times ahead, a ratio void for the reason corrected in the thirty-fourth paragraph. The kernels remain exact there,
  1,172,518 outputs against the CPU backend.

  What this fixes beyond the number is the structure. The workgroup
  size, tile shape and shared-memory arrays of this kernel were
  literals tuned to one card; the row kernel had long adapted, reading
  the device's subgroup width and computing lanes per row and rows per
  subgroup on the host. The tile kernel now adapts too, by the
  mechanism Vulkan provides for it, so a device with a different
  subgroup width, a different shared-memory limit or a different ratio
  of shared-memory bandwidth to arithmetic gets a different shape from
  the same source. The shared-memory limit was the next input tried to
  that policy, and it did not pay. A 128 by 128 shape halves the shared
  memory read per product again, sixteen reads per sixty-four products
  against twelve per thirty-two, and its 33792 bytes fit the 64 KB Mesa
  reports where they do not fit the 32 KB the AMD proprietary driver
  does, so it would have been the first thing that capability bought.
  On the MI50 it took prompt processing at 512 rows from 2619 to 1144
  tok/s. Eight columns to a thread means the inner loop holds eight of
  them, and indexing that by the loop variable put them where the fast
  path does not want them; the per-row kernel had learned the same
  thing earlier, where accumulator arrays indexed by column cost it ten
  times. Generalising the inner loop to a column count cost the
  4-column form as well, 0.6B pp256 reading 2642 tok/s against 2796
  with the named scalars while gaining 2 percent at 512, so the scalars
  stay and the column count stays a literal. A wider tile was then
  revisited written as named scalars per column, compiled as its own
  module from the same source under a define the way the row kernel's
  families are, so nothing was indexed and nothing was generalised. It
  is still bad, and worse than the indexed version suggested. Two
  shapes became selectable, 64 by 128 needing 25344 bytes and 128 by
  128 needing 33792, and the narrower fits the 32 KB the AMD
  proprietary driver reports, so it ran on both cards: 0.6B pp512 fell
  from 2818 to 458 tok/s on the Radeon VII and from 2628 to 576 on the
  MI50, with the 8B falling from 292 to 95 there. Prompt sizes that did
  not select it were unchanged, so the four-column path was intact and
  the eight-column one is simply slow.

  That is worth stating plainly because the arithmetic said otherwise.
  A 64 by 128 tile and a 128 by 64 tile read the same shared memory per
  product, hold the same 25344 bytes, and the wider one re-reads the
  weights half as often, so it should have been at least even. It is
  six times slower. Two explanations were tested and both fail. Shared
  memory bank conflicts: with 33-float padding, sixteen threads taking
  eight contiguous columns each land in four banks where four columns
  each land in eight, so the columns a thread owns were restrided by
  sixteen to spread them over sixteen banks, and pp512 still read 457
  tok/s. Register pressure or spilling: the driver reports the
  eight-column kernel at 96 vector registers with no scratch at all,
  against 87 for the four-column one, which is the same two waves per
  SIMD. So it is not the reads per product that the twelve-against-eight
  argument counts, not the banks those reads fall in, and not spilling.
  Until it is understood a wider micro-tile is not the lever it looked
  like, and four columns stay.

  The disassembly was then read, and it answered a different question
  than the one it was opened for. The inner loop of the tile kernel
  issued, per step: six address calculations, six shared-memory reads,
  two waits and sixteen multiply-adds. Half the instructions were not
  arithmetic. Two causes, neither visible in the source. The loop was
  not unrolled, so the driver recomputed every address each step rather
  than folding the step into the read's immediate offset. And the
  shared tiles were held row-major, `[row][k]`, so the four values a
  thread wants for one k sat 33 floats apart and needed six reads.

  Both were fixed by changing where the values sit rather than what the
  kernel computes. The tiles are k-major now, `[k][row]` and
  `[k][column]`, which makes a thread's four rows and four columns
  adjacent: the driver emits one 128-bit read and two paired reads
  where it used to emit six, and registers fell from 87 to 79. The k
  loop is unrolled by four on top of that, so the four steps differ by
  a constant. Interleaved on the Radeon VII, two passes, three models,
  every cell improves and none regresses:

  | case | before | after |
  |---|---:|---:|
  | 0.6B pp128 | 1904 tok/s | 1925, 1935 |
  | 0.6B pp256 | 2744 | 2795, 2799 |
  | 0.6B pp512 | 2798, 2800 | 2946, 2960 |
  | 8B Q8_0 pp256 | 264 | 288 |
  | 8B Q8_0 pp512 | 307 | 333 |
  | 8B Q4_K_M pp512 | 108 | 109 |

  On the MI50 it is worth more: 8B pp32 86 to 103 tok/s, pp64 158 to
  171, pp512 291 to 323, and 0.6B pp96 972 to 1208. That the same
  source change is worth twice as much under Mesa is the same pattern
  as everything else here, the driver deciding what a shape costs.

  The unroll was then deepened from four to eight, which removes more
  of the address arithmetic still being emitted. Interleaved, two
  passes, three models, every cell improves: 0.6B pp512 2999 to 3033
  tok/s, pp256 2851 to 2867, 8B Q8_0 pp512 339 to 345, pp256 294 to
  297, 8B Q4_K_M pp512 110.7 to 111.2. Sixteen and thirty-two measured
  the same as eight, so eight is where the return stops and the
  shallower form is kept.

  Splitting the inner dimension was then written and rejected. At 64
  prompt rows a tile dispatch yields 16 workgroups where the MI50 has
  64 compute units, so the kernel is starved and the row kernel wins
  there by default; the fix for that is to cut the inner dimension into
  slices, run a dispatch per slice, and add the partial sums in one
  pass. It was implemented, a `matmul_combine` kernel and a k range on
  the tile kernel, correct on both cards, and slower everywhere: on the
  Radeon VII 0.6B pp64 781 tok/s against 850 and pp128 1879 against
  1961, and on the MI50, with the tile kernel forced so the split could
  engage at all, pp32 528, pp64 941, pp96 1126 and pp128 1589 against
  1071, 1042, 1168 and 1610 unsplit.

  The arithmetic says why, and it was predictable: a slice writes a
  whole nbatch by nout array of floats and the combine pass reads them
  all back, so the traffic added is twice slices times nbatch times
  nout times four bytes. At pp64 on a 1024 by 1024 projection that is
  2 MB against the 1.1 MB of weights the call reads at all, roughly
  tripling its memory traffic to buy four times the workgroups.
  Splitting pays only where the weights dwarf the output, which is not
  this shape. Reverted; the row kernel keeps those batches.

  A profiler followed, because until now every claim about where time
  went was inferred from kernels timed alone or read off their machine
  code, neither of which sees overlap or idle. A diagnostics backend
  now writes a timestamp either side of every dispatch and
  `bench --model --profile` reports device time per kernel. The first
  run, 8B Q4_K_M decode on the Radeon VII, 64 tokens:

  | kernel | device ms | share |
  |---|---:|---:|
  | matmul_row_k4 | 229.5 | 67.1% |
  | matmul_row_k | 78.1 | 22.8% |
  | rms_norm_rows | 20.0 | 5.9% |
  | attention_kv16 | 6.1 | 1.8% |
  | norm_rope_kv_kv16 | 4.6 | 1.4% |
  | silu_mul, quantize_x, embed, gather_rows | 4.0 | 1.2% |

  Ninety percent is the two K-quant matmuls, so nothing is hiding
  outside the kernels this cell is about. The line that changes the
  plan is `rms_norm_rows` at 5.9 percent: its work is one row of 4096
  values, trivial, and at two dispatches a layer over 36 layers times
  the 3.8 us a boundary costs (the twenty-ninth paragraph) that is
  0.24 ms of the 0.31 ms it shows. It is almost entirely the boundary.
  Removing those dispatches is worth about 4.6 percent of decode, which
  is most of the 6 percent this cell is short.

  The profile also settles what the dominant kernel is bound by, which
  three earlier guesses had not. Sampling 4096 dispatches, about 14
  tokens, the Q4_K matmul takes 9.5 ms of a 20.7 ms token and reads the
  model's 3.705 GB of Q4_K weights in that time, which is 390 GB/s
  against roughly 1000 of this card's peak. The Q6_K matmul reads its
  1.316 GB at about 306 GB/s. Neither is close to memory bound; both
  are instruction bound, at about four operations per weight with a
  quarter-rate 32-bit multiply among them. Making that multiply full
  rate is worth roughly 40 percent of the dominant kernel, which is far
  more than this cell is short.

  And the instruction exists, which four earlier probes had missed
  because they all asked for integer dots. This driver refuses to emit
  an integer dot under any formulation tried: the extension, packed
  8-bit dots, and two ways of stating operand widths all produce the
  same wide multiplies. Asked for a half-precision dot it emits
  `v_dot2_f32_f16`, the chip's native one, which takes two half
  products into a float accumulator at full rate. The probe that found
  it pays for it in conversions, 195 int-to-float and 96 float-to-half
  against 48 dots, because it was handed integer activations; the point
  is only that the instruction is reachable here.

  What that implies is a different activation format for the row
  kernel: halves rather than 16-bit integers with a block scale. A
  weight nibble becomes a half in one conversion, the dot takes two
  products at full rate, and the block scale still multiplies the sum
  afterwards, so the arithmetic is about 2.5 full-rate operations per
  weight against the present five effective. The integer twin exists
  because integer dots needed a shared scale; halves carry their own
  exponent and need none. Not yet written, and it touches the twin's
  producers as well as every row path, so it is the next substantial
  piece rather than a tune.

  Eight columns to a thread was retried on the k-major layout, where a
  thread's eight columns are contiguous rather than 33 floats apart,
  which was the objection to the first two attempts. It is still
  catastrophic: 0.6B pp512 547 tok/s against 2953, 8B pp512 127 against
  333, at 97 registers with no scratch. That is three attempts, as an
  indexed array, as named scalars row-major, and as named scalars
  k-major, all within a factor of the same result, so the effect is a
  property of the hardware rather than of any one way of writing it.
  Widening a thread's rows from four to eight helped; widening its
  columns the same way does not, and the asymmetry is unexplained. Four
  columns stay, and the remaining inner loop is close to its minimum:
  two reads and sixteen multiply-adds per step, about three quarters of
  issued instructions being arithmetic.

  Declaring the tiles as four-wide vectors, to force both reads to
  128 bits rather than leaving the columns as a pair of reads, was also
  measured and is slightly worse: 8B pp512 320 tok/s against 333.

  The same method then found the decode lever, and it is a driver
  difference again. The per-row matmul's inner loop spent, per 16-bit
  product, a sign-extend pair and a `v_mul_lo_u32`, which is a full
  32-bit multiply and runs at quarter rate on gfx906. The chip has a
  native 16-bit dot instruction and the integer dot product extension
  maps to it, but only under Mesa: the AMD proprietary driver lowers
  that extension back to the same multiplies with the operands widened
  first, which is why the extension was dropped this morning on the
  strength of measurements taken here. Under Mesa the dot form emits
  128 of the native instructions where the multiply form emits 35 wide
  multiplies, and per-type bandwidth on the MI50 goes from 222 to 241
  GB/s on Q4_K, 239 to 284 on Q5_K and 213 to 254 on Q6_K.

  Neither form wins everywhere, so both are compiled from the one
  source under a define and the backend picks per device, availability
  being a capability and worth being a measurement:

  | model | multiplies | dots | device |
  |---|---:|---:|---|
  | 8B Q8_0 | 39.8, 39.4 tok/s | 45.7, 45.7 | MI50, Mesa |
  | 0.6B Q8_0 | 246.8, 245.8 | 262.7, 261.3 | MI50, Mesa |
  | 8B Q8_0 | 40.9, 40.7 | 40.1, 40.3 | Radeon VII |
  | 8B Q4_K_M | 47.9, 48.0 | 46.6, 47.0 | Radeon VII |

  So 15 percent of 8B decode and 6 of 0.6B on the MI50, and this
  workstation keeps the multiplies and is unchanged. One ordering bug
  on the way, worth noting because the structure invites it: the
  capability flags were filled before the extension scan that
  discovers them, so the MI50 silently kept the slower form until the
  profile decision moved after the scan.

  Against the reference's own Vulkan build on the MI50, both arms in
  one container, five runs a point, two passes, after all of the day's
  kernel work: pp64 1075 tok/s against 2016, pp256 2518 against 3469,
  pp512 2743 against 3556, decode 262 against 102. Prompt processing is
  53, 73 and 77 percent of it, against 41, 65 and 74 this morning, and
  decode is 2.6 times ahead rather than 2.4. On the same card the
  per-type sweeps read 0.6B pp16 to pp96 at 1029, 1071, 1092, 1103 and
  1218 tok/s against 851, 883, 897, 904 and 1208, the short prompts
  gaining a fifth from the dot-form row kernel; the 8B reads pp32
  104.5, pp64 180.5 and pp512 327.

  The tile threshold turns out to be a property of the driver, not only
  of the card, which the profile can hold but cannot yet derive. It was
  measured at 32 rows for 8-bit projections on the Radeon VII under the
  AMD proprietary driver. On the MI50 under Mesa, the same silicon, the
  row kernel wins all the way to 96: at 16, 32, 48, 64, 96 and 128 rows
  the tile reads 230, 422, 614, 825, 940 and 1355 tok/s on
  Qwen3-0.6B-Q8_0 while the row kernel reads 852, 883, 897, 900, 890
  and 889. So between 32 and 96 rows the shipped threshold picks the
  slower kernel there, by up to a factor of two at 32.

  The first attempt was to make the threshold follow the shape rather
  than the device, since the crossover also moves with how much work a
  row carries: 64 rows on a 1024-wide 8-bit projection against about 24
  on a 4096-wide one. Interleaved on the Radeon VII across three
  models, two passes each, that was a regression, 0.6B pp48 reading 825
  tok/s against 978, and a wash everywhere else, so it is not in the
  tree. The shape does move the crossover, but less than the driver
  does, and a rule fitted to one device mispredicts the other.

  Measured again with the current tile kernel, which had moved the
  crossover since the table above, the picture is that width matters as
  much as the driver. Forcing each kernel and sweeping rows on
  Qwen3-0.6B-Q8_0 and Qwen3-8B-Q8_0:

  | device | 1024-wide projection | 4096-wide |
  |---|---:|---:|
  | Radeon VII, AMD driver | 40 rows | 26 |
  | MI50, Mesa | 96 | 30 |

  A narrow projection wants a much higher threshold than a wide one on
  both devices, which is why one constant of 32 was wrong for the small
  model everywhere and right for the 8B everywhere. The threshold is
  now 64 rows when a row carries fewer than 4096 values and 32
  otherwise, the boundary put at 4096 because it is what separates the
  two models: the 0.6B's widest projection is 3072 and the 8B's
  narrowest is 4096. An earlier attempt at the same idea put the
  boundary at 2048, which left the 0.6B feed-forward projection on the
  wide side, so half the call still took the wrong kernel and the two
  arms measured the same.

  Interleaved, two passes, three models on the Radeon VII: 0.6B pp32
  goes from 694 and 705 to 834 and 845 tok/s, pp48 from 973 and 978
  down to 851 and 852, and pp16, pp64, pp96, pp128, pp512 and every 8B
  figure are unchanged. On the MI50 the same change reads off the
  sweeps as 422 to 883 tok/s at 32 rows, 614 to 897 at 48 and 825 to
  900 at 64, with nothing given up, since its crossover is further from
  the shipped value. So it costs 13 percent in one band on one device
  to gain 20 percent in another there and 9 to 109 percent across three
  bands on the other.

  Verified on the MI50 afterwards rather than predicted from the
  sweeps: 0.6B pp16, pp32, pp48, pp64 and pp96 read 851, 883, 897, 856
  and 972 tok/s, against 852, 422, 614, 825 and 940 before, and the 8B
  reads 85.9, 157 and 292 at pp32, pp64 and pp512 against 83.7, 154 and
  about 290. So the band that was picking the slower kernel there is
  closed, and nothing regressed.

  The per-device ideal being 48 rows here and 96 there, the compromise
  is now only what an unmeasured device gets. `measured_profiles` in
  `backends/device_profile.hpp` is a table keyed by what a device and
  its driver call themselves, holding what that combination actually
  wanted; the two measured entries are the Radeon VII under the AMD
  proprietary driver at 48 and the MI50 under Mesa at 96. A device in
  the table runs better than the defaults, one that is not runs exactly
  as before, and bringing up hardware is running the sweeps and adding
  a row. It keys on device and driver together because the driver is
  what the measurement moved with: the same Vega20 wants 40 rows under
  one and 96 under the other.

  Measured after wiring it: the Radeon VII reads 971 tok/s at pp48
  against 851 with the compromise, which is the 13 percent the single
  number had given up, and the MI50 reads 896 at pp64 against 856.
  Nothing else moved on either.

  A second, separate observation, and only an observation: the two
  drivers report different limits for the same Vega20 silicon, 32 KB
  of shared memory per workgroup from the AMD proprietary driver on
  Windows and 64 KB from Mesa. Our tile is fixed at 64 by 64 using
  16896 bytes and the backend never reads
  `maxComputeSharedMemorySize`, so whatever a device offers above that
  goes unused, which is worth fixing on its own terms. It is not
  evidence about the reference: an earlier revision of this paragraph
  argued from llmx reading 1795 on the Radeon VII against 1783 on the
  MI50 while the reference went 1662 to 3555, and that comparison is
  void. The two reference figures are different builds (11075 against
  11100), different compilers, operating systems and drivers, three
  changes at once, which is the mistake the build-identity rule in
  AGENTS.md exists to prevent, applied to environments rather than
  binaries. Every prompt-processing measurement from here is llmx
  before against llmx after against the reference, all three in one
  environment.

  The tiled attention does not yet share a K/V tile across the query
  heads of a KV group.

## KV cache fork, step 2 of the KV design (2026-09-21)

- **Goal:** a second history with the same committed tokens, sharing every
  full block and copying the partial tail, which the server needs for
  prefix reuse and branching and which the fork rule in KV-CACHE.md had
  described without implementing.
- **Done:** `KVSequence::fork` retains every full block and takes a fresh
  block for a partial tail, reporting the ids so the caller fills it;
  `Backend::kv_copy` copies every layer's K and V of one block into
  another, because only the backend knows the layout; `Model::fork` does
  both on every storage and the fork inherits the tickets of the passes
  that wrote what it shares. Shared blocks are read-only: `prepare`
  refuses to append into one, which a history truncated into a shared
  block would do, so such a history is forked again instead. `kv-cache`
  checks a fork across a block edge (shared block at refcount two, tail
  private and copied, both histories diverging without touching the
  shared block, a boundary fork taking no tail, the refused append,
  releases following the refcounts) and, through the model, that a forked
  sequence continues exactly as a fresh one fed the same tokens while the
  original continues exactly as if never forked. Native 19/19, Python
  12/12 with both HF models.
- **Done: gate**, base `c85aa9d`, candidate `215ad24`, layout control
  `d8d508e`. Five 0.6B cells at 15 pairs, one 8B at 9. System CPU averaged
  47 and 43 percent in the first two 0.6B cells and 38 to 39 in the rest,
  against the benchmark's own 37.5, so other activity was present during
  the two cells that moved most. Evidence in
  `docs/benchmarks/kv-fork-20260921/`, raw monitors archived and hashed;
  every sample kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -1.39% / -2.54% / 9/15 | -1.59% / -1.05% / 9/15 |
  | 0.6B-1 | decode  | +3.38% / +2.15% / 5/15 | +3.23% / +0.95% / 7/15 |
  | 0.6B-2 | prefill | **-3.92% / -3.51% / 11/15 FAIL** | +1.53% / +2.26% / 5/15 |
  | 0.6B-2 | decode  | **-3.66% / -4.01% / 12/15 FAIL** | +1.31% / -0.16% / 8/15 |
  | 0.6B-3 | prefill | -0.76% / -2.13% / 9/15 | +1.45% / +1.36% / 6/15 |
  | 0.6B-3 | decode  | -1.12% / -0.06% / 8/15 | -0.24% / -0.39% / 8/15 |
  | 0.6B-4 | prefill | -2.20% / -2.35% / 11/15 | -1.45% / -0.74% / 9/15 |
  | 0.6B-4 | decode  | +0.01% / +0.55% / 7/15 | +0.28% / -0.21% / 8/15 |
  | 0.6B-5 | prefill | -1.77% / -1.53% / 10/15 | -0.48% / -0.97% / 8/15 |
  | 0.6B-5 | decode  | -0.60% / -0.40% / 10/15 | -1.96% / -2.10% / 10/15 |
  | 8B | prefill | -0.23% / +1.08% / 4/9 | +0.01% / +0.55% / 4/9 |
  | 8B | decode  | +0.31% / +0.00% / 4/9 | -1.00% / -0.22% / 6/9 |

  Cell 2 failed both phases at about -3.8 percent with the control positive
  in the same cell; the three cells after it pass and 8B is flat. What the
  measured path gained from this change is one refcount comparison per
  pass in `prepare`. Two things are written down rather than argued away.
  0.6B decode, the phase that comparison sits on, is flat over five cells:
  +3.38, -3.66, -1.12, +0.01, -0.60, mean -0.4 against a control mean of
  +0.5. 0.6B prefill is under base in all five cells, mean -2.0 against a
  control mean of -0.1, which is the same size and sign as the step 2 gate
  and the opposite of the step 6 gate on changes of the same character;
  it is inside the band AGENTS.md documents for this file and this model,
  and it joins the running list of candidates if 0.6B prefill is ever
  found a few points low against an older baseline.
- **Left:** nothing. The prefix index is step 4 of the KV design and lands
  with the server.

## Execution model for batching and placement (ROADMAP #5, #7) (2026-09-21)

- **Goal:** fix what the backend interface and the model layer need for the
  per-layer and per-tensor splits and for continuous batching, now that both
  are scoped and per-row split is dropped, so the first vendor backend
  implements each signature once.
- **Done:** the design, `docs/EXECUTION.md`: `submit`/`wait` tickets with
  no events, host-visible memory and `write` returning with the transfer as
  its caller, per-row positions with the RoPE table as a buffer, batched KV
  views with `gather_rows`, and `Model` / `Sequence` / `ExecContext` /
  `Batch`. Seven ordered steps; steps 1 to 4 change the interface and go
  before Vulkan. ROADMAP #4a marked done, #5 rewritten without per-row, #7
  pointed at the design; stale "steps 5 and 6 remain" claims corrected in
  ARCHITECTURE, DEVICE-EXECUTION, README and the backend page.
- **Done:** amended for architectures beyond dense Qwen (hybrid compressed
  attention, MoE, lookup-table memory, residual mixing): tables are per
  storage counted in that storage's entries, `kv_alloc` describes an entry
  by key and value widths, placement is per tensor role with attention and
  feed-forward separate, and a sequence may hold private unpaged state.
  The "Beyond dense Qwen" section lists the assumptions steps 3, 4 and 6
  must not make.
- **Done: step 1.** `norm_rope_rows` takes the cos/sin tables as buffers
  and one position per row; the model adopts its table once and passes a
  slice of an identity position table. `Backend::rope` is deleted; `bench`
  times `norm_rope_rows` instead of a function the runtime never called.
  `backend-group` checks three rows at positions 5, 2 and 9 against a
  double-precision reference reading the table at each row's own position.
  Native 18/18, Python 12/12 with both HF models.
- **Done: step 1 gate**, three arms built the same way from detached
  worktrees: base `f1e0be4`, candidate `7519170`, and a layout control
  `769761e` that is base plus an unused function appended to each of the
  two files the change edits (`cpu_backend.hpp`, `arch_qwen.hpp`), per the
  AGENTS rule that the control must perturb the same files. Qwen3-0.6B-Q8_0
  five times at 15 pairs, Qwen3-8B-Q8_0 once at 9, six threads, 250-token
  prompt, 32 generated tokens. Evidence in
  `docs/benchmarks/rope-positions-20260921/`; raw monitors archived beside
  the repo and hashed in each cell's `monitor-summary.json`. System CPU
  averaged 40 to 46 percent across cells against the benchmark's own 37.5,
  so other activity was present throughout; every sample is kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | +3.83% / +3.84% / 4/15 | -2.61% / -0.42% / 9/15 |
  | 0.6B-1 | decode  | +1.47% / +1.80% / 5/15 | +0.33% / +3.32% / 5/15 |
  | 0.6B-2 | prefill | -0.29% / -0.06% / 8/15 | -1.33% / +0.16% / 7/15 |
  | 0.6B-2 | decode  | +0.46% / +0.91% / 5/15 | +1.45% / +0.87% / 6/15 |
  | 0.6B-3 | prefill | -0.35% / +2.29% / 6/15 | +0.73% / +0.03% / 7/15 |
  | 0.6B-3 | decode  | **-2.31% / -1.75% / 13/15 FAIL** | -2.22% / -1.61% / 10/15 |
  | 0.6B-4 | prefill | +1.60% / +0.06% / 7/15 | +0.78% / +0.38% / 7/15 |
  | 0.6B-4 | decode  | -0.08% / -0.27% / 8/15 | -1.29% / -1.39% / 10/15 |
  | 0.6B-5 | prefill | -0.86% / -0.81% / 8/15 | +0.44% / +0.09% / 7/15 |
  | 0.6B-5 | decode  | +0.51% / +0.34% / 6/15 | +2.06% / +1.84% / 6/15 |
  | 8B | prefill | +1.85% / +0.23% / 3/9 | +2.01% / +0.08% / 4/9 |
  | 8B | decode  | +0.14% / +0.45% / 4/9 | +0.75% / +0.45% / 2/9 |

  Fails at 12 of 15 and 8 of 9 base wins. One cell failed, 0.6B-3 decode,
  on the win count. In that same cell the control, which executes the
  same instructions as base, lost 2.22 percent with 10 base wins, so the
  cell moved against both later arms rather than against the change, and
  the two reruns the goal requires came back at -0.08 and +0.51 with 8 and
  6 base wins. Over five 0.6B cells the candidate's decode mean is
  +0.01 percent. Recorded as noise confirmed by rerun, with the failing
  cell kept. The +3.83 prefill in cell 1 is not claimed either; the
  control spans -2.61 to +0.78 on the same measurement.
- **Done: step 2.** `submit()` returns a monotonic ticket and `wait()`
  blocks on one; `sync()` stays what the failure paths call, because a
  failed pass has ops queued behind no ticket. `alloc` takes a `Memory`
  kind; the logits buffer is host visible and the host reads it in place
  after the wait, so no read op copies a row it can already see. The model
  submits once per pass, waits on that ticket for the logits, and `reset`
  waits on the last ticket instead of draining. `kv-cache` counts the
  calls: one submission waited on once per step, no read op for logits, a
  three-token prompt at ubatch 2 submitting twice and waiting once, and a
  reset that waits without a sync. Native 18/18, Python 12/12 with both HF
  models.
- **Done: step 2 gate**, same shape as step 1: base `2bbfd34`, candidate
  `4ee6f48`, layout control `bd0cdc7` perturbing `cpu_backend.hpp` and
  `arch_qwen.hpp`. Three 0.6B cells at 15 pairs, one 8B at 9. System CPU
  averaged 37 to 40 percent per cell against the benchmark's own 37.5, so
  the machine was close to quiet; every sample is kept. Evidence in
  `docs/benchmarks/tickets-20260921/`, raw monitors archived and hashed.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -1.72% / -2.45% / 10/15 | -1.03% / -1.20% / 9/15 |
  | 0.6B-1 | decode  | +0.58% / +1.04% / 4/15 | +0.79% / +0.21% / 7/15 |
  | 0.6B-2 | prefill | -1.10% / -1.54% / 9/15 | -1.98% / -0.24% / 9/15 |
  | 0.6B-2 | decode  | +1.02% / -0.04% / 8/15 | -0.84% / -1.84% / 9/15 |
  | 0.6B-3 | prefill | -2.02% / -2.84% / 10/15 | +0.21% / -0.43% / 9/15 |
  | 0.6B-3 | decode  | +0.67% / +1.03% / 6/15 | +2.02% / -0.54% / 8/15 |
  | 8B | prefill | +0.54% / +0.36% / 3/9 | +0.74% / +1.00% / 1/9 |
  | 8B | decode  | +0.86% / +0.66% / 3/9 | +0.98% / +0.66% / 2/9 |

  All eight cells pass. The candidate's 0.6B prefill is below base in all
  three cells, by 1.1 to 2.0 percent on the mean, which is inside the band
  the control itself spans (-1.98 to +0.21) and the change adds nothing a
  prefill executes beyond one counter increment per pass. Recorded rather
  than explained away; if a later step finds 0.6B prefill a point or two
  low against an older baseline, this is a candidate along with the code
  read's three points.
- **Done: step 3.** `attention` and `kv_write` take an array of `KVView`s
  laid out in row order, and a view carries `nq`, the rows of the pass that
  are its own; `pos` and `batch` are gone because both follow from the
  view, and the sequence fills `nq` from what it has prepared and not
  committed. The CPU backend takes the views one after another around the
  code it had, so one view computes what it did before. `kv-cache` checks
  two sequences with histories of different lengths across a block edge,
  written and attended in one call, against the same two taken separately,
  bit for bit. `gather_rows` is deferred to step 4, where a `Batch` gives
  it a real caller. Native 18/18, Python 12/12 with both HF models.
- **Done: step 3 gate**, base `c850149`, candidate `8a3ff82`, layout
  control `a09c584` perturbing the two edited files. Three 0.6B cells at
  15 pairs, one 8B at 9. System CPU averaged 37 to 39 percent per cell
  against the benchmark's own 37.5; every sample is kept. Evidence in
  `docs/benchmarks/views-20260921/`, raw monitors archived and hashed.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | -0.60% / -0.32% / 8/15 | +0.97% / +0.81% / 6/15 |
  | 0.6B-1 | decode  | **-1.21% / -1.59% / 12/15 FAIL** | -0.76% / -0.53% / 8/15 |
  | 0.6B-2 | prefill | +1.83% / +2.08% / 5/15 | +2.98% / +1.27% / 7/15 |
  | 0.6B-2 | decode  | +0.10% / +0.18% / 7/15 | +0.52% / +0.81% / 5/15 |
  | 0.6B-3 | prefill | +1.82% / +1.37% / 4/15 | +2.45% / +2.25% / 4/15 |
  | 0.6B-3 | decode  | +1.75% / +2.59% / 5/15 | +2.12% / +1.98% / 5/15 |
  | 8B | prefill | +1.60% / +0.03% / 4/9 | +1.42% / -0.31% / 6/9 |
  | 8B | decode  | -0.10% / +0.22% / 4/9 | -0.08% / -0.65% / 5/9 |

  One cell failed, 0.6B-1 decode, on the win count alone at exactly the
  threshold; its mean and median are inside the noise band and the control
  was negative in the same cell. The two cells run after it, identical in
  plan, came back at +0.10 and +1.75 with 7 and 5 base wins, and 8B decode
  is flat. Recorded as noise confirmed by rerun, with the failing cell
  kept.
- **Done: step 4.** `Model` owns the weights, the cache's pool and physical
  storage and the backend, read-only after construction apart from pool
  bookkeeping. `Sequence` is one request's history and its last ticket,
  `ExecContext` one pass in flight, and `Model::forward` runs one pass over
  a batch of entries, each a sequence with tokens to append and whether it
  wants logits; rows that want logits are compacted by `gather_rows`, which
  gets its first caller, and the head runs once over them. `forward`
  submits and returns; the context waits on the ticket the first time its
  logits are read, so a caller with two contexts can submit the next pass
  before reading this one. `step` and `prefill` are wrappers on a sequence
  and a context the model keeps, so the CLI, `generate`, `chat` and
  `perplexity` are unchanged. Decode runs the same row-batched graph as
  prefill with one row, which retires the separate decode arena and the
  single-row helpers. `kv-cache` runs two sequences in one pass, one
  decoding over a history while the other prefills, against the two alone,
  and refuses a sequence listed twice; `backend-group` checks the gather.
  Native 18/18, Python 12/12 with both HF models.
- **Done: step 4 gate**, base `56ee2d1`, candidate `18b9c74`, layout
  control `1a05110` perturbing `cpu_backend.hpp` and `arch_qwen.hpp`.
  Three 0.6B cells at 15 pairs, one 8B at 9. Evidence in
  `docs/benchmarks/model-split-20260921/`, raw monitors archived and
  hashed; every sample is kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | **-2.29% / -2.21% / 12/15 FAIL** | -0.54% / -1.08% / 10/15 |
  | 0.6B-1 | decode  | +0.84% / +0.77% / 5/15 | +0.19% / +0.37% / 6/15 |
  | 0.6B-2 | prefill | -0.66% / -0.70% / 8/15 | +0.27% / +0.40% / 5/15 |
  | 0.6B-2 | decode  | +1.35% / +1.18% / 5/15 | +0.66% / +1.00% / 5/15 |
  | 0.6B-3 | prefill | +1.57% / +2.50% / 5/15 | +3.01% / +0.97% / 5/15 |
  | 0.6B-3 | decode  | +3.03% / +1.45% / 2/15 | +1.68% / +1.88% / 6/15 |
  | 8B | prefill | -0.05% / +0.54% / 4/9 | -0.28% / -0.03% / 5/9 |
  | 8B | decode  | +0.32% / +0.44% / 3/9 | +0.27% / +0.22% / 2/9 |

  One cell failed, 0.6B-1 prefill, on the win count at exactly the
  threshold with mean and median inside the band; the two identical cells
  after it came back at -0.66 and +1.57 with 8 and 5 base wins, and 8B
  prefill is flat. Recorded as noise confirmed by rerun, failing cell kept.
  Decode is positive in every cell, which is the number this step could
  have moved: the decode loop now runs the row-batched graph with one row
  rather than its own path, and it did not cost anything measurable.
- **Done: the Vulkan backend page**, `docs/VULKAN.md`, designed against
  the Radeon VII's reported properties: wave64, 8- and 16-bit storage,
  timeline semaphores behind `submit`/`wait`, push descriptors, a 4-byte
  storage offset alignment that closes the `adopt` alignment question, no
  cooperative matrix so matmul is subgroup dots, and the memory types each
  `Memory` kind maps to. Loader loaded at run time so one binary runs
  without Vulkan; GLSL in the tree compiled by `glslc` at build time and
  embedded. Seven sub-steps with a CTest that skips without a device.
- **Done: step 6, placement**, taken ahead of the Vulkan backend because
  its first test needs no device. `Model` takes several backends and a
  `Placement`, a device per tensor role: each layer's attention and
  feed-forward block, the embedding table and the output head. Each weight
  is adopted by the backend that hosts its role; each device that runs
  attention gets a `KVStorage` for exactly its layers with its own pool,
  block size and adopted RoPE tables; a `Sequence` holds a table per
  storage and a ticket per device, and records the model that made it.
  Wherever the placement changes, the residual stream crosses through the
  context's staging vector, a `read` then a `write`, which gives `write`
  its caller. The `placement` CTest splits a two-layer model over two CPU
  backends so both crossings fall inside a layer and requires the bytes of
  one backend for a prompt, decode steps, a history across a block edge, a
  reset and a two-sequence pass; it counts reads and writes so crossings
  are exactly where the placement changes and absent on one device. No
  flag selects a placement yet: with the CPU as the only backend there is
  nothing to place, so `--device` and `--n-gpu-layers` land with Vulkan.
  Native 19/19, Python 12/12 with both HF models.
- **Done: step 6 gate**, base `6514b17`, candidate `3a5aa7b`, layout
  control `c6743a1`. Three 0.6B cells at 15 pairs, one 8B at 9, all eight
  pass. Evidence in `docs/benchmarks/placement-20260921/`, raw monitors
  archived and hashed; every sample kept.

  | cell | phase | candidate mean / median / base wins | control mean / median / base wins |
  |---|---|---|---|
  | 0.6B-1 | prefill | +2.92% / +3.09% / 4/15 | +0.44% / +0.61% / 7/15 |
  | 0.6B-1 | decode  | -1.16% / +0.08% / 7/15 | -1.05% / -1.28% / 9/15 |
  | 0.6B-2 | prefill | +2.03% / +1.37% / 3/15 | +0.70% / -0.15% / 10/15 |
  | 0.6B-2 | decode  | +0.96% / +0.55% / 6/15 | +1.08% / +0.46% / 6/15 |
  | 0.6B-3 | prefill | +2.61% / +3.05% / 1/15 | +0.46% / +0.97% / 7/15 |
  | 0.6B-3 | decode  | +0.32% / +0.12% / 6/15 | -0.35% / -1.07% / 8/15 |
  | 8B | prefill | +0.28% / +0.67% / 3/9 | +0.51% / +0.92% / 3/9 |
  | 8B | decode  | -0.21% / -0.43% / 5/9 | +0.04% / +0.22% / 4/9 |

  0.6B prefill is two to three percent up in every cell against a control
  under one percent, with base wins of 4, 3 and 1 of 15. Not claimed as a
  win: the single-device path adds a role lookup per layer and nothing it
  could have saved, and the step 2 gate had the same measurement two
  percent the other way on a change of the same character. It is the
  layout band again, this time in the candidate's favour, and the two
  cancel.
- **Left:** the Vulkan backend, sub-step 1 of `docs/VULKAN.md`. This
  workstation has the loader and `vulkaninfo` but not the SDK: `VULKAN_SDK`
  is unset and there is no `glslc`, so the LunarG SDK has to be installed
  before sub-step 1 can build here. Then the placement flags with it.
- **Gotchas:** `sync()` stays `noexcept`; `wait` is too. The other
  developer's last recorded position predates the last five merges to
  main; the design is posted for review but does not wait on it.

## ROCm on Windows is not available for this hardware (2026-09-21)

Checked before planning any GPU work on the workstation. The Windows HIP SDK
supports RDNA3, RDNA3.5 and RDNA4 only, lists no Instinct card, and states
that it does not support gfx906 (Vega 20). gfx906 entered ROCm maintenance
mode in 5.7 and is deprecated on Linux too, where it still runs but AMD no
longer builds for it.

Consequence for the plan, which it confirms rather than changes: the ROCm
backend is developed and validated on the Linux machine. Windows keeps the
CPU backend. If this workstation ever needs GPU acceleration, the route is
Vulkan, already the roadmap's portability target and supported by AMD's
Windows drivers.

## Device execution model complete (ROADMAP #4a) (2026-09-21)

All six steps are merged. Weights and activations are `Buffer` handles, every
op takes a buffer and a float offset, KV blocks are buffers the backend
allocates, and ops enqueue with one sync per forward pass. The model layer
holds no host address and computes no offset into KV storage. Per-step blocks
are deleted per the checkpoint rule; the plan and the step table live in
[DEVICE-EXECUTION](DEVICE-EXECUTION.md) and every gate's evidence is under
`docs/benchmarks/`.

What a vendor backend supplies: `alloc`, `adopt`, `read` and `copy` over its
own allocator, `kv_alloc` and a `KVStorage` it owns, the ops, and a `sync()`
that means it. What it does not have to invent is a place for activations or
a cache layout.

Three things are worth carrying forward rather than rediscovering.

- **`sync()` is `noexcept` by contract.** Three of its four callers are
  exception paths releasing KV blocks, so a sync that threw would replace the
  error that got there and leave the sequence half-released. A backend that
  cannot establish that its work finished must fail hard.
- **A few percent of Qwen3-0.6B prefill means nothing on this tree without a
  control.** The same measurement moved -3 points during the code read and +6
  at step 6, both times from code placement rather than work. AGENTS.md
  carries the rule and the evidence; the short version is that the control
  has to perturb the same file the change does.
- **`Backend::write` was deleted** after step 5 declined to give it a caller.
  Weights arrive through `adopt` and every other value is produced by an op.
  The first backend that genuinely needs a host-to-device write adds it back
  alongside that caller.


## Full code read before the first vendor backend (2026-09-21)

- **Goal:** read every line of `src/`, `tests/` and `tools/` before starting a
  GPU backend, fix what is actually broken, record the rest. Steps 1 to 4 of
  the device execution migration had landed and it was the right point to stop
  and look at the whole tree.
- **Done:** the read. `src/model/kv_cache.hpp` is the only file with nothing
  to report.
- **Done: two real bugs.** `f32_to_f16` OR-ed the rounding carry into the
  exponent field instead of adding it, so any value whose mantissa rounded up
  out of ten bits came back exactly half the right size whenever the exponent
  was odd; 1.999756 encoded to 1.0 rather than 2.0. It reached only
  `llmx quantize`, where it scales a whole block, and nothing had ever
  compared that direction against a reference. `tests/fp16.cpp` now does,
  using binary16 as its own oracle over the whole finite range. Separately,
  `bpe::Tokenizer::decode` indexed the vocabulary with an unchecked id, which
  `llmx detokenize <id>` passes straight from the command line.
- **Done: two hangs and a silent truncation reachable from a model file.**
  A chat template calling `replace`, `count` or `split` with an empty needle
  looped forever, because each advanced its cursor by the needle's length.
  The template is GGUF metadata, so that was untrusted input reaching an
  unkillable loop. A model declaring no EOS id had token zero, an ordinary
  token, treated as the stop token; and a reasoning-start marker with no
  matching end marker suppressed the entire reply.
- **Done: three tests asserted less than they claimed.** Two were coverage
  this project lost to its own activation-arena change and did not notice:
  `prefill-scope` counted 64- and 96-byte allocations that stopped existing
  when nine vectors became one arena, and `kv-cache` aimed an injected
  allocation failure at "the second large batch buffer" for the same reason.
  The third predates it: the tokenizer fixture typed its special token 2,
  which is GGUF's *unknown* rather than *control*, so the special-token path
  its docstring advertises was never exercised, and its unicode case checked
  only an exit code.
- **Done:** the vocabulary vector was zero filled twice per forward pass with
  a `read` overwriting all of it immediately after, an extra 608 KB per
  decoded token on Qwen3-8B; comments across the backend, the CPU backend and
  the model still described the pre-migration interface; `ARCHITECTURE`,
  `ROADMAP`, `DEVICE-EXECUTION` and the backend page said the interface takes
  raw host pointers, which stopped being true at step 4.
- **Done: the gate failed, and the cause was code layout, not the change.**
  Worth reading before the next few-percent argument. The first 0.6B prefill
  cell came in at -4.58% and failed the advance rule. Reruns gave -0.81,
  -2.52, -1.92 and -2.33, so two thirds failed and it was clearly not simple
  noise. A layout control built by appending an unused function to
  `cpu_backend.hpp` moved the same number by only -1.16/+0.63/-1.13/+0.33,
  which made the candidate look like a real regression sitting outside the
  band.

  Bisecting the seven commits found it. The f16 commit alone was clean at
  -0.56% mean. Adding the next commit, a four-line bounds check inside
  `Tokenizer::decode`, produced -3.13/-2.17/-3.09. `Model::prefill` never
  calls `decode`; the prefill timer cannot execute one instruction of it. The
  entire measured difference came from where those four lines pushed the code
  that follows them in the single translation unit.

  Two things follow. The check is now `vocab.at(id)`, one token instead of
  four lines, with the readable message at the CLI call site where the caller
  knows the vocabulary size. And the control has to perturb the same file as
  the change: appending to a different header understated the band by more
  than a factor of two. AGENTS.md carries this now; cells are `06-layout*`,
  `06-fp16*`, `06-tok*`, `06-pre*` and `06-mid*` under
  `docs/benchmarks/code-read-20260921/`.
- **Merged with one performance cell unresolved.** Recorded so nobody later
  reads this as four green cells. Correctness was green throughout: native
  18/18, Python 12/12 with both HF models, 260-token logits byte-identical.
  0.6B prefill came in at -3.87/-1.59/-3.71 across three runs, failing the
  advance rule twice, while 8B prefill gained 3.56% and both decode cells
  were flat. The bisection above is why it merged anyway: the loss tracks
  where the compiler places the code, not work the change added, and the
  clearest single case is a four-line guard in a function that prefill never
  calls moving the same number by three points. If a later step finds 0.6B
  prefill about three points low against an older baseline, this is where it
  went.
- **Left:** nothing blocking.
- **Gotchas:** `Backend::write` and `Backend::copy` are implemented and have
  no caller anywhere, which AGENTS.md forbids. They stay only because step 5
  is their consumer and is next; if step 5 does not use `write`, delete it
  there. `Backend::rope` is the last op taking raw host pointers, and its only
  caller outside the CPU backend is `bench`, which therefore measures a
  function the runtime never calls; `norm_rope_rows` is what the model runs
  and it still takes raw `cos`/`sin` pointers. Moving both is a measured
  change and belongs in its own gated commit, not this one.

  Known and deliberately not fixed: `strip_ws` in the template renderer eats a
  leading unary minus, so `{{ -1 }}` renders `1`; `{% for k, v in x %}` binds
  one variable literally named "k, v" rather than unpacking; `quant/` includes
  `format/gguf.hpp`, which reaches up one layer and is already recorded in
  ARCHITECTURE as a known exception; `quantize_row_q4_1` has no production
  path, since the CLI writes only q8_0 and q4_0; `metadata_u64`,
  `Tokenizer::token_id` and `pad_id` have no callers.

## Code layout moves this benchmark more than the rule allows (2026-09-21)

Moving the embedding gather into the backend measured -8.03% on 0.6B prefill,
twice. The gather itself measures 0.0014 ms at that shape and the logits are
byte-identical, so the work cannot account for 38 ms. Builds that behave
identically, 15 pairs each, against one reference build:

| Build | Prefill | Decode |
|---|---:|---:|
| baseline + unused function | -1.34% | -1.35% |
| baseline + small unused function | -0.50% | -1.18% |
| baseline + larger unused function | +2.52% | -2.00% |
| embed as written | -8.03% | -0.84% |
| embed + that function as dead code | -1.89% | +0.42% |
| embed, gather forced out of line | -3.75% | -0.90% |
| embed, prefill body forced out of line | -4.36% | -1.89% |
| embed, token id in a member slot | -0.60% | -2.21% |

Three things follow, and the second one matters most.

**Layout alone spans about four points of prefill**, wider than the runner's
3% band. The first probe, an unused function and nothing else, **failed the
advance rule** on decode at 12 of 15 baseline wins. The rule can therefore
report a regression for a relink.

**A single-build A/B cannot gate a few-percent change on this cell.** A
candidate needs comparing against several perturbed builds of the same
behaviour, not one. Earlier few-percent prefill conclusions on this model
carry that uncertainty, as does the 0.6B prefill cell's own A/A failure at
-3.53%.

**The -8% build was still real.** It sits outside the layout band, so it was
a genuinely poor layout, not noise: one extra local in a very large inline
body was enough. Forcing functions out of line made it worse; moving the
local to a member slot restored it to -0.60%, inside the band. That is the
shipped form, with the reason in the code.

Root cause is structural. The whole runtime is one translation unit of
headers and the forward pass is one enormous inline body, so any edit can
reshuffle it. Splitting the model layer into its own translation unit would
bound this; AGENTS already allows `.cpp` files with one per logical unit.
Evidence in `docs/benchmarks/layout-sensitivity-20260921/`.

**Then the confound in my own comparison.** The build embeds the Git
revision and a dirty marker. The candidate had been built from a dirty tree
and the baseline from a clean worktree, so the two binaries differed in an
embedded string as well as in code. Rebuilding both the same way, from
detached worktrees at their two commits, the same change reads:

| Cell | Paired mean | Paired median | Baseline wins |
|---|---:|---:|---:|
| 0.6B prefill | +1.09% | -0.08% | 8/15 |
| 0.6B decode | +0.13% | -1.37% | 8/15 |
| 8B prefill | +0.35% | +0.83% | 3/9 |
| 8B decode | +0.51% | +0.46% | 4/9 |

Dead level, against -8.03% when the arms also differed in build identity.
The rule for every comparison from here: **build both arms the same way**,
same source of the version string and same tree state, so the only
difference is the change. An embedded string is enough to move this
benchmark by several points.

## Matched mx gate on a quiet machine (2026-09-21)

The comparison repeated under the rule below, with recorded system CPU of
39% on 8B and 43% on 0.6B against the benchmark's own 37.5%. Eight pairs per
model, alternating arms, llmx from `0c0dec7`, mx `5542318e74`. Evidence in
`docs/benchmarks/kv-mx-quiet-20260921/`.

| Model | Phase | llmx | mx | Paired median | llmx wins |
|---|---|---:|---:|---:|---:|
| 0.6B Q8_0 | prefill | 524.57 | 277.54 | +85.64% | 8/8 |
| 0.6B Q8_0 | decode | 49.09 | 47.77 | +3.91% | 7/8 |
| 8B Q8_0 | prefill | 40.77 | 22.16 | +84.54% | 8/8 |
| 8B Q8_0 | decode | 4.64 | 4.67 | -1.53% | 3/8 |

Three of four cells clear the floor, prefill by a wide margin on both models.
8B decode is 1.53% under it, against a harness A/A of 0.06% on that cell, so
the sign is outside harness noise; but llmx wins 3 of 8 pairs and the
per-pair spread runs -7.5% to +1.2%, so the size is not well determined. The
earlier figure of -3.5% came from the loaded run and overstated it.

This is the current state of ROADMAP #8's performance gate: met everywhere
except 8B decode, which is under by roughly one and a half percent.

## The matched mx comparison ran under background load (2026-09-20 run)

`tools/compare_cpu.py` passes its own A/A: one binary as both arms moves at
most 0.80%, and 0.06% on the 8B decode cell. The harness is sound.

**Corrected cause.** This block first blamed memory contention between the
two arms, because llmx copies the model into the heap while mx maps it. That
was a hypothesis and it is wrong. Two things refute it. The same depression
appears on Qwen3-0.6B, where both processes together need about 1.2 GB on a
32 GB machine, so there is no pressure to have. And the recorded monitors
give the real answer: the comparison ran at 55% mean system CPU against the
benchmark's own 37.5%, while the later A/A runs sat at 36 to 40%. Roughly a
fifth of the machine was doing something else during the comparison, which
is why both arms were slow. Medians, tok/s:

| Condition | 0.6B pp | 0.6B tg | 8B pp | 8B tg |
|---|---:|---:|---:|---:|
| llmx alternating with mx | 464.92 | 43.56 | 37.34 | 3.89 |
| mx alternating with llmx | 259.21 | 41.62 | 20.58 | 4.02 |
| llmx alternating with itself | 548.42 | 50.94 | 41.49 | 4.73 |
| mx alternating with itself | 278.13 | 49.64 | 21.45 | 4.78 |

Both binaries lose about a fifth under that load, llmx 3.89 against 4.73 and
mx 4.02 against 4.78, and no drift appears across pairs, so it is not heat
building up. Pairing is what makes the comparison survive this: both arms
alternate inside the same conditions, so the relative result holds even
though the absolute rates do not represent a quiet machine. The 8B decode
deficit reads -3.2% paired under load against -1.0% unpaired when quiet, and
the unpaired figure compares two runs three hours apart, which is the
comparison this project has been burned by. The honest statement is a
deficit of roughly 1% to 3.5%, still under the floor, smaller than first
reported, and not caused by the KV work.

Prefill is unaffected: llmx is roughly double mx in every condition measured.

What this changes for method: the runner already records activity, but
nothing acted on it. A comparison worth publishing should be repeated when
the recorded system CPU is near the benchmark's own share. Evidence in
`docs/benchmarks/mx-harness-aa-20260921/`.

## A/A calibration of the A/B runner (2026-09-20)

Run after the screening and sampler results below, not before them, which is
the wrong order and is why this block exists. The same binary published as
both arms, `tools/ab_runner.py`, 247-token prompt, 6 threads, evidence in
`docs/benchmarks/aa-calibration-20260920/`.

| Model | Phase | Paired mean | Paired median | Baseline wins | Per-pair spread |
|---|---|---:|---:|---:|---|
| 0.6B Q8_0 | prefill | -2.04% | -3.53% | 7/9 | -8.09% to +8.96% |
| 0.6B Q8_0 | decode | -0.03% | +0.30% | 4/9 | -5.94% to +3.03% |
| 8B Q8_0 | prefill | -0.00% | +0.07% | 2/5 | -4.49% to +2.57% |
| 8B Q8_0 | decode | +1.93% | +2.27% | 2/5 | -2.24% to +5.67% |

**0.6B prefill fails its own A/A**: identical code reports a 3.53% median
loss, past the runner's 3% band. 8B decode moves 2.27% on identical code.
So on this machine today, a single cell below about 4% in those two places
is not evidence either way. What survives: 0.6B decode, where the A/A holds
to 0.30% median.

Consequences, applied to the blocks below rather than left for a reader to
work out:

- The sampler's 0.6B decode gain, +32.04% with 0 of 9 baseline wins, is far
  outside this band and stands.
- The sampler's prefill cells, +2.79% and +2.17%, are inside the band and
  are **withdrawn**; the change has no measured prefill effect.
- The sampler's 8B decode cell was already reported as unresolved and stays
  so.
- In the KV screening, per-cell prefill differences under about 4% carry no
  weight. The rejection of 64 rested on five of eight plans losing prefill,
  two of them by 6.5% and 7.8%, which is a pattern rather than one cell, but
  it is weaker evidence than that table implied. The choice of 128 over 256
  rests on allocated-versus-used bytes, which is counted rather than timed
  and is unaffected.
- The matched mx comparison uses a different harness
  (`tools/compare_cpu.py`, in-process timing) which has **not** been A/A
  calibrated. Its 8B decode cell of -3.5% should be treated as provisional
  until it is.

## Greedy sampling cost (2026-09-20)

Greedy sampling sorted the whole 151936-token vocabulary before reading one
element. `sample` now takes a linear maximum at temperature zero and a
partial sort over the top-k window above it; the repetition penalty is read
through a lambda instead of materialized. Ties now take the lowest token id,
where the unstable sort left them unspecified.

| Paired A/B, generate --temp 0, 247-token prompt, 6 threads | Prefill | Decode |
|---|---:|---:|
| Qwen3-0.6B Q8_0, 9 pairs | +2.79% | +32.04% |
| Qwen3-8B Q8_0, 5 pairs | +2.17% | -0.72% |

The saved work is a constant per token, about 7 ms, so it dominates a 0.6B
token and is inside the runner's noise band on 8B at five pairs, where it is
reported as unresolved rather than as a win. Evidence in
`docs/benchmarks/sampler-greedy-20260920/`. The matched mx gate is unaffected:
that harness times model inference only and excludes sampling.

## Paged KV cache design (2026-09-20)

- **Goal:** replace the single-sequence contiguous `HostKVCache` with a paged
  cache whose logical view (per-sequence block table, positions, refcounts) is
  backend-neutral and whose physical block size and layout are backend-owned,
  so the multi-user server and GPU backends do not inherit a CPU layout.
- **Done:** isolated cold, shuffled paging microbenchmark, seven paired
  repeats, two geometries, three lengths: block 16 costs +25-33% on decode
  attention, 128 costs +5-10%, 256 costs +1-3%. Design in
  [KV-CACHE](KV-CACHE.md), opened for review by the second developer.
- **Done (step 1):** `model/kv_cache.hpp` (`BlockPool`, `KVSequence`),
  backend `kv_layout`/`kv_alloc`/`kv_write` and view-form `attention`,
  `CpuKVStorage` backed on demand, `HostKVCache` and the raw-pointer
  attention removed. The review findings are folded in: each step and each
  prompt is one transaction, bookkeeping vectors are reserved so failure
  paths never allocate, pool and sequence own their ids (non-copyable,
  release on destruction), retain rejects free ids, growth copies into
  exact-size buffers with retained and peak bytes reported, the budget
  crosses the seam in tokens, arithmetic is checked. `generate --verbose`
  prints `kv: allocated/peak/used`. Native suite 17/17; Python suite with
  both HF models passes; main and paged logits byte-identical on 0.6B and
  8B (11 to 841 tokens) and greedy text identical. Provisional block size
  128 behind the temporary `LLMX_KV_BLOCK` knob.
- **Done (screening):** 64/128/256 against contiguous main on 0.6B and 8B
  Q8_0, eight frozen plans, all samples kept, tables in
  [KV-CACHE](KV-CACHE.md). Block fixed at 128; the knob is deleted.
- **Done (mx gate):** matched comparison on the paged runtime, tables in
  [KV-CACHE](KV-CACHE.md). Prefill +79.8% (0.6B) and +82.7% (8B) over mx,
  8/8 pairs; decode -0.2% and -3.5%. The 8B decode cell is under the floor
  and is the existing decode bandwidth item, not a paging cost.
- **Left:** re-review of the final revision; a decision on the 8B
  decode cell; then main integration. Fork/COW and device buffers are later
  steps. F16 KV is out of scope.
- **Gotchas:** the microbenchmark is isolated attention with a cold cache and
  is not an end-to-end decode cost. Memory waste cuts against large blocks:
  224 KiB per token on 0.6B means a partial 256-token tail wastes up to
  55.8 MiB per sequence, 27.8 MiB at 128.

## Native HF download checkpoint (2026-09-20)

ROADMAP #9a is implemented: native `llmx pull`, immutable revision resolution,
verified cache, `HF_TOKEN` credentials, bounded parallel downloads (four streams
by default) and aggregate sharded GGUF loading. This checkpoint is locally
validated and prepared for main publication; hosted CI is still pending.
HF native safetensors/tokenizer/config support and device kernels remain planned.

| Local check | Windows | Linux |
|---|---:|---:|
| Native suite | 16/16 | 15/15 with UBSan |
| Required-HF Python components | 12/12 | 12/12 with UBSan |
| Shard/Unicode fixtures | 66/66 | 66/66 with ASan/UBSan |
| Transport fixtures | 58/58 | 58/58 |
| Sharded HF full-logit cases | 20/20 | 20/20 |

The real 639,446,688-byte public Q8 download passes independent SHA256 after
four-stream assembly. Cache reuse passes on Windows and Linux. Five measured
loader pairs average 427.246 ms before and 428.494 ms after; paired speed change
is -0.283% with a descriptive interval [-3.706%, +3.140%]. This tiny observed
cost is unresolved; all activity-flagged samples are retained. Complete loaded
payload/tensor/metadata hashes match in all 12 processes. No inference arithmetic
changed, and this is not a fresh mx inference comparison. Full evidence and
limitations are in the [checkpoint](ASSETS.md#native-hf-pull-checkpoint-2026-09-20).
All 28 Markdown files were reviewed. Live gated-repository use and controlled
credential redirects remain untested; native credential fixtures and public
HTTPS pass. A maintained curl 8.4+ supplies HTTPS and redirect behavior.

## Prefill placement checkpoint (2026-09-20)

Backend-owned prefill placement is merged and published on both main remotes
at `3c5d4b9`, under the performance tradeoff policy. All five hosted jobs pass in
[run 35516912422](https://github.com/mxxm-t/llmx/actions/runs/35516912422): Windows,
macOS Intel, Linux, Linux UBSan and required HF. The new native targets are
included (12 Windows, 11 Linux/macOS).
On supported Windows topology, six-worker prefill uses separate physical cores
and checks restoration before decode. Other configurations fall back. There is
no runtime affinity flag, NUMA memory policy, arithmetic change or concurrent
submission support. Persistent OS refusal to restore is reported as an error.

| Final source check | Result |
|---|---:|
| Windows native / Linux native | 12/12 / 11/11 |
| Required HF Python suites, Windows / Linux | 11/11 / 11/11 |
| Active-placement HF cases | 28/28 |
| Exact Q8 follow-up pairs | 6/6 |
| F32/Q8 long-continuation vectors | 33/33 each |
| Q8 monitored blocks / processes | 24/24 / 72/72 |
| F32 monitored blocks / processes | 4/4 / 12/12 |
| Frozen identities rechecked | 79/79 in each run |

| Primary model | Paired prefill gain vs main | Paired combined gain vs mx |
|---|---:|---:|
| 0.6B Q8 | +26.40% | +37.49% |
| 8B Q8 | +32.84% | +34.84% |
| 0.6B F32 | +13.70% | +2.86% |

All three measured rounds are retained. Competing developer builds/model jobs
overlapped Q8 despite the reservation and materially limit causal claims.
Observed Q8 follow-up combined means range from -4.39% to +1.44% versus main,
with wide uncertainty; F32 decode is -1.26%, while its combined measurement is
+1.27% versus main. These costs are accepted alongside the primary gains, not
relabelled as zero or used to claim universal parity. Full phase, absolute-time,
activity and uncertainty tables are in the
[final comparison](ASSETS.md#final-prefill-placement-comparison-2026-09-20).

Independent review verified all 84 outputs/vectors and reproduced the statistics
and activity summaries. The single-worker performance smoke passes both arms;
its one-pair numbers are diagnostic. Source `250846a` is unchanged since final
correctness and timing. All 26 Markdown files were reviewed for this
checkpoint. The original working tree and executable remain untouched.

## Runtime base release checkpoint (2026-09-20)

The completed CPU runtime stack is accepted for release under the user's
performance tradeoff policy: large gains may justify smaller costs elsewhere,
with HF correctness and matched mx comparisons retained. This is a scoped
release decision, not a claim of universal per-phase or per-quant superiority.
The optional placement candidate is not part of that published base; its
completed integration is recorded in the checkpoint above.

Release `08351b0` combined runtime checkpoint `0d41a7b` with public main `9511a4a`.
All `src/` files at that reconciliation matched the runtime parent after Git
newline normalization. CI retry/cache handling and plain Windows build-header error
checks are retained. No inference arithmetic changed during reconciliation.
The original Windows working tree, user files and its root executable are not
replaced by this release.

| Final reconciliation check | Windows MSVC | Linux GCC 13.3 / WSL |
|---|---:|---:|
| CMake Release build | Pass | Pass |
| Native CTest | 10/10 | 10/10 |
| Offline downloader cases | 15/15 | 15/15 |
| Python suite with both HF models required | 11/11 | 11/11 |
| Tiny F32 full-logit maximum HF error | 0.00000070 | 0.00000070 |

These are correctness/build checks; simultaneous jobs and diagnostic synthetic
timings do not supply new performance measurements. The Linux machine's HF container
lacked CMake, so the Linux run used the configured WSL toolchain instead.
Independent merge review verifies that native tests, all eleven Python
components, UBSan, required HF fixtures and downloader checks remain wired.
All 25 Markdown files were reviewed for source alignment, ASCII and local links.

## Base release performance decision

The base release's Q8 evidence is the fixed monitored comparison: six workers,
ubatch 128, F32 KV, eight measured rounds per model/workload, with all samples
retained. The table reports paired combined-work speed changes for production
versus pinned mx `5542318e74`; positive means faster. Follow-up prefix loading
is outside the timer. See the [complete phase tables and monitoring limits](ASSETS.md#monitored-prefill-comparison-2026-09-20).

| Model / workload | Combined speed vs mx | Approximate 95% interval |
|---|---:|---:|
| 0.6B Q8, primary 215+32 | +19.69% | +15.87% to +23.50% |
| 0.6B Q8, follow-up 1+32 | -2.58% | -12.82% to +7.67% |
| 0.6B Q8, follow-up 9+32 | +12.46% | -15.74% to +40.66% |
| 8B Q8, primary 215+32 | +24.41% | +10.65% to +38.17% |
| 8B Q8, follow-up 1+32 | -4.72% | -16.01% to +6.58% |
| 8B Q8, follow-up 9+32 | -2.87% | -7.61% to +1.88% |

The primary gains support release while the smaller follow-up costs remain
explicit. Zero-crossing intervals do not prove parity or make the costs zero.
Background imbalance, competing-agent jobs and coarse telemetry limit causal
attribution. No sample was discarded or corrected. An idle PC is not required;
future timing must keep monitoring and must not overlap competing agent work.

The base release used the separate nine-round F32 comparison from the
[ordered-reduction study](ASSETS.md#ordered-prefill-accumulator-reductions-2026-09-20):

| 0.6B F32 phase | llmx tok/s | mx tok/s | Mean-rate difference |
|---|---:|---:|---:|
| Primary prefill | 382.921 | 364.721 | +4.99% |
| Decode | 13.735 | 13.524 | +1.56% |

The base release's backend, inference and quantization sources match that
`bf122fd` study; later model edits validate construction. This is applicable
historical evidence, not a new F32 measurement on the reconciled tree. Older
F32 deficits and the earlier sub-percent Q8 decode veto are superseded as
release blockers by the later evidence and clarified tradeoff policy. Their
original results remain in ASSETS. Other hardware, quants and workloads need
their own measurements; K-quant optimization remains separate work below.

## Status table

| Feature                                  | Status   |
|------------------------------------------|----------|
| Layered restructure                      | Done     |
| Build config (config.hpp + CMake + build.bat) | Done |
| Test suite (roundtrip / perf / tokenizer)| Done     |
| Perf `bench` command                     | Done     |
| CPU backend optimization                 | Done     |
| CPU tiny-activation range repair | Done; measured CLI Q5 decode cost retained in the checkpoint above |
| Vulkan allocation failure ownership | Done |
| Vulkan attention width and mixed-cache validation | Done |
| More quant formats (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K read) | Done |
| Quantization coverage: F16/BF16, MXFP4, IQ4, Q3_K, Q2_K | Planned (block above), built in the background |
| More model architectures (Llama, ...)    | Planned  |
| Qwen 3.5, 3.6 and 3.8 (`qwen35`, `qwen35moe`) | Planned (block above, design in [QWEN35](QWEN35.md)), built in the background |
| More formats (safetensors, ...)          | Planned  |
| JSON syntax and Unicode validation      | Done |
| GGUF reader size and tensor extent validation | Done |
| JSON quantize tensor validation | Done |
| Qwen model construction validation | Done |
| Paged KV cache (block pool, backend-owned blocks) | Done |
| Device execution model (ROADMAP #4a)     | Done     |
| Execution model: tickets, batched views, placement (`docs/EXECUTION.md`) | Done: steps 1 to 7, step 7 being the server, see the server row; `--device` lists select a layer split (multi-device row) |
| KV cache fork (KV-CACHE step 2)          | Done     |
| Multi-device split (per-layer, per-tensor) | In progress (`docs/MULTI-DEVICE.md`): phase 0 measured and merged; phase 1, the layer split over a `--device` list fitted to free memory, on its branch; pipelining, the scheduler over passes in flight and tensor groups follow |
| GPU backends (Vulkan first to write, ROCm first-class) | Vulkan implemented and the recorded dense-model device gate passed on both platforms (forty-seventh checkpoint above): Radeon VII decode 102-115% and prefill 109-455% of the same-card reference Vulkan build; one MI50 decode 102-115% and prefill 102-267%. These are dated gate results, not new measurements from this documentation review. ROCm planned |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Done (`docs/SERVER.md` steps 1 to 6): `llmx serve`, correctness gates pass on both backends, throughput 109 to 125 percent of the reference server at 1 to 16 concurrent on the device (short of the wide margin `docs/SERVER.md` gates on), prefix reuse through fork, a second execution context measured to have nothing to hide, the OpenAI-compatible routes |
| Chat follow-up cache validation          | Done |
| Correctness baseline vs HF reference     | In Progress |
| Pinned HF reference generation           | Done |
| Optional Qwen3-8B HF consumer             | Done |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Matched CPU comparison thread selection | Done |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | Done |
| CPU attention in backend (ROADMAP #4a)  | Done |
| CPU row streaming / parallel prefill   | Done |
| CPU attention value accumulation      | Done |
| CPU grouped projections              | Done |
| CPU Q8 scale / load scheduling       | Done |
| Head-major CPU KV storage             | Done |
| CPU worker exception safety           | Done |
| CPU worker cost profile                 | Done |
| CPU ordered prefill reductions          | Done |
| Backend-owned prefill placement | Done (main `3c5d4b9`, five hosted jobs green) |
| CLI thread settings, including batched/per-token perplexity | Done |
| Automatic build identification          | Done (main `9511a4a`) |
| Focused CLI help and complete current option coverage | Done (2026-09-24 checkpoint) |
| Live generation and loading progress     | Done |
| Model loading and teardown buffer lifetime | Done (2026-09-25 checkpoint) |
| CPU zero-byte transfer and zero-thread hint contracts | Done (2026-09-25 checkpoint) |
| GitHub CPU CI                          | Done     |
| HF fixture download retries and CI cache | Done |
| Hosted numeric/path portability repair | Done (five jobs green at `851d375`) |
| HF model download and sharded GGUF (ROADMAP #9a) | Done (included in main; five hosted jobs passed at `7e195ff`) |
| HF native formats (ROADMAP #9b)          | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

`Done` denotes implemented and validated functionality in this release tree.
The earlier runtime base `08351b0` was published on both main remotes. Its initial five-check
hosted run `35512421834` passed ordinary Ubuntu and required HF, but failed
Windows reference-generator path spelling, UBSan exact scalar-tail comparison,
and macOS JSON subnormal conversion. Repair `851d375` passed all five jobs in
[run 35512954742](https://github.com/mxxm-t/llmx/actions/runs/35512954742):
Windows, macOS Intel, Linux, Linux UBSan and required HF. The repair changes
JSON conversion plus test portability, preserving inference kernels and bounds.
Evidence is in `docs/benchmarks/ci-portability-20260920.json`; prior four-check
passes at `b266650` and `9511a4a` cover those smaller releases.
See [CI](CI.md) for the precise workflow scope and local reproduction commands.

## Active feature blocks

### Scoped correctness coverage and remaining HF work

- **Goal:** keep independent HF ground truth and extend coverage where the roadmap requires it.
- **Done:** exact tokenizer fixtures; tiny tied/untied F32 full logits and NLL; real Q8/Q4 ranking and excerpt PPL; HF/Jinja2 follow-up chat fixtures; pinned reference generation and strict consumers. The unchanged real 8B consumer previously passed 37/37 on Windows and Linux with frozen bounds. Real 0.6B F32/Q8 1,943-token plus 32-step continuation checks are archived in ASSETS.
- **Left:** broader full-corpus, maximum-context and per-layer references, plus prospective numerical bounds for any new lossy kernels. Short 8B rankings/excerpts are not deep-context validation.
- **Gotchas:** self-consistency is supplementary. Exact comparison against another llmx path cannot replace HF. Model construction validation does not establish finite weights, arbitrary token-ID safety, request budgets or failed-session recovery.

## Working rules and ownership

Current feature ownership and timing reservations are recorded in the shared
collaboration log outside this repository. Confirm ownership there before
starting work; historical branch names below are not active assignments. Builds and tests
may run in parallel when no timing reservation is active. Keep every planned
performance sample, record ordinary machine activity, and report missing
telemetry honestly. GitHub receives main only; feature work stays on its branch until its gates pass.

## Historical feature blocks (2026-09-19 to 2026-09-22)

These dated blocks preserve the results, plans and open items from their
checkpoints. Use the status table and current feature blocks above for the
present state. Historical Left lines can refer to work that later shipped
or was superseded. The overall HF correctness and mx performance requirements
remain in progress; historical results establish only their recorded scope.

### Vulkan prefill through the 8-bit integer dot (2026-09-22)

- **Goal:** prompt processing on the MI50 at least level with the reference, where it was 44 to 79 percent against a reference split across ten cards and is 24 to 73 percent against one (the thirty-fourth paragraph, measured after the tile below). Measured on that card within one environment at a 4096 x 14336 projection over 512 rows, our float tile reads 4.87 TFLOPS for Q8_0, 4.65 for Q4_K and 3.62 for Q6_K, level with the reference's own float tile at 4.77, while its 8-bit integer-dot tile reads 13.30, 11.42 and 7.03. So the gap is that path, not scheduling or tile shape (STATUS, thirty-third paragraph above).
- **Plan:** a kernel quantizes each activation column to 8-bit values per 32-value block with the block's scale and scaled sum; a tile kernel stages one quant block per row and column per step as packed 8-bit words and scales, and multiplies with the four-wide integer dot, one float multiply-add per block for the scale and one more for a type's minimum. Q8_0 and Q4_K first, then Q6_K and Q5_K. Used only where the device's integer dot is native, which the profile records as `prefer_integer_dot`; elsewhere the float tile stays.
- **Done:** the diagnosis above, and `backend-vulkan` now times the tile at that shape in TFLOPS. `quantize_x8.comp` and `matmul_tile_q.comp` for Q8_0 and Q4_K (`cb2eb5b`), taken where the profile says `prefer_integer_dot`. On the MI50 at the 8B feed-forward shape Q8_0 goes 4.87 to 7.72 TFLOPS and Q4_K 4.65 to 11.48, the reference's being 13.30 and 11.42. Qwen3-8B-Q4_K_M prompt processing 297.8 to 488.7 tok/s at 512 rows. Correctness: every HF perplexity cell in both scoring modes on the MI50 with all three 0.6B fixtures, the backend test's 1,172,518 outputs with its reference rounded the same way, and on the 8B Q4_K_M file, which no fixture covers, 40 wikitext windows of 512 at mean NLL 2.47005 against the float tile's 2.47023. Scoring through batched passes (`fc261f9`) is what made the HF gate reach this path at all. Then the thirty-fifth to thirty-seventh paragraphs: Q8_0 staging a word at a time (12.07 TFLOPS), Q6_K in its own module (9.60 against the reference's 7.03) and Q5_K through the tile (`c1bdb09`, `db249b8`); Q4_0 and Q4_1 through it too and the measured profile carrying four thresholds, the MI50's row 16, 32, 24 and 40 (`2b770f6`); and a third tile height of 32 rows for short prompts, 0.6B Q8_0 at 64 rows 2502 tok/s, 54 percent of the reference's 4638 (`33933a9`).
- **Left:** staging several quant blocks per barrier (being measured); MI50 prompt processing at 69 to 89 percent at 247 rows and more; MI50 decode at 86 to 91 percent on the 8-bit files.
- **Gotchas:** 8-bit activations cost 0.009 of NLL in the decode kernel earlier, close to the 0.010 bound on one HF cell, so the device suite on the MI50 decides whether this ships, per type. The AMD Windows driver lowers the integer dot extension to widened multiplies, so the Radeon VII must keep the float tile.

### External floor of merged main (2026-09-20)

Matched against mx `5542318e74`, six threads, eight rounds, identical committed
token IDs in both arms. Runtime is `main` after the device-execution steps and
the fused Q5_K/Q6_K decode dots; the quantized-activation kernels are NOT in it.

| Model / phase | llmx | mx | ratio | |
|---|---:|---:|---:|---|
| Q8_0 prefill | 495.23 | 271.75 | 1.82x | above |
| Q8_0 decode | 46.01 | 44.94 | 1.02x | above |
| Q5_K_M prefill | 408.70 | 242.41 | 1.69x | above |
| Q5_K_M decode | 34.92 | 60.93 | 0.57x | **below** |

Three of four clear the floor, and Q8_0 clears both phases: decode crossed from
0.99x earlier today to 1.02x. **ROADMAP #8 is met for Q8_0 and NOT met
overall** - K-quant decode is the single remaining failure, with mx 1.75x
faster. Absolute rates are higher than this afternoon on both arms because host
load fell; the ratios barely moved (Q5_K decode 0.589x then, 0.573x here).
The 1.02x is inside the range an A/A can produce and should not be leaned on;
the 1.69x-1.82x margins and the 0.57x shortfall are not.
Evidence: `benchmarks/main-external-floor-20260920.json`.

### K-quant and device execution work (separate developer branch)

- **Goal:** improve the remaining K-quant decode path and continue ROADMAP #4a without overlapping this release/placement work.
- **Done:** the separate `design/device-execution-model` branch and the K-quant experiments are not incorporated by this release. Their measurements and source identities must be reviewed before adoption. Quantized-activation commits `357d68d` and `97d52e8` fail `backend-group`; they remain isolated and are not merge-ready. Arithmetic-preserving dispatch work is being separated onto a passing base.
- **Left:** prospective correctness/performance validation for any new quantized-activation path. The device execution model is complete on main (see the 2026-09-21 block above); K-quant decode remains. Coordinate rebases and announce timing reservations.
- **Gotchas:** earlier grouped Q16 failed the unchanged native double-dot accuracy contract. Do not reuse it as a lossless baseline or weaken bounds after observing results. CPU Q8 results do not establish K-quant parity.


### GGUF reader size and tensor extent validation

- **Goal:** reject malformed lengths, dimensions, arithmetic overflow and tensor
  extents before allocating payload storage or reporting loading progress;
  honor the file's declared alignment. This closes the documented format-layer
  error-handling gap and supports future Hub/sharded-format work.
- **Done:** bounded reads, checked size arithmetic and subtraction-based file
  ranges reject malformed input before payload allocation/progress. Reader and
  writer honor positive uint32 alignments divisible by eight, including 24.
  Quantized row widths must contain whole blocks. Array depth is limited to
  256 and tensor rank to four; valid empty tensors retain mathematical size zero.
  Isolated branch `fix/gguf-tensor-extents` starts at `5859762`; placement
  experiments and fixed binaries remain separate.
- **Done:** 122 GGUF cases pass on Windows/Linux: independently constructed
  fixtures plus the writer-alignment round trip. All nine
  native tests and all eleven required-HF suite components pass on both.
  Current-reader ASan+UBSan passes both format/progress tests. The initial full
  Windows build lacked the MSVC include environment; that failure is retained
  and the complete run passes after initializing vcvars64. No source workaround.
- **Done:** the pinned 8B file loads completely with byte-identical `info`
  output against the prior validated control. All six fresh short HF cases
  match top-1 and top-5 overlap 5/5. This is not a new 8B NLL/long-context gate.
  All jobs are terminal. All 25 project Markdown files reviewed and stale
  `info`, quantized-row and loader validation descriptions corrected.
- **Left:** merge with the runtime stack once its separate performance gate
  passes; the root executable and main/GitHub remain unchanged. No inference
  hot path changed and no new performance result is claimed. Evidence:
  [`gguf-reader-validation-20260920.json`](benchmarks/gguf-reader-validation-20260920.json).
- **Gotchas:** token IDs, numerical weight contents, writer validation and future
  request recovery remain separate validation work. The later Qwen construction
  block checks configuration geometry and required tensor layouts.
  JSON conversion dimensions are covered by the later block
  above. Do not
  claim that file-extent checks make arbitrary models executable. Preserve
  nested GGUF arrays within a documented depth limit and valid non-power-of-two
  alignments that are multiples of eight.

### Prefill placement reassessment with machine activity monitoring

- **Method clarification after review:** the frozen runner rotates and
  reverses the three arm orders inside each model/workload block; model and
  workload order also rotate across one warmup and eight measured rounds.
  Report mean/median rates and elapsed time, paired ranges, sample deviation
  and approximate paired 95% intervals, retaining all originals. A third
  contaminated block ends the study as inconclusive; it is not accepted.
  Benchmark and recorder CPU are excluded by PID plus creation time. All arms
  use the same model path/hash for a given model. Activity telemetry does not
  measure DRAM bandwidth or prove absence of short/inaccessible activity;
  sub-percent observer effects also remain unresolved. No clean-preflight
  result alone establishes that a small performance difference is real.
- **Goal:** complete the reopened whole-prefill placement assessment against
  production and matched mx, including HF/lossless and short follow-ups.
- **Done:** preserved the historical candidate and its failed original screen;
  JSON checkpoint `a61c414` passes Windows/Linux correctness suites.
- **Done:** fresh scratch control and default-enabled candidate retain current
  JSON/CLI fixes and pass 8/8 Windows CTests each. The candidate full required-HF
  suite passes 11/11; timings are diagnostic only. Rebuilt callback contracts
  pass 17 cases/4,626 exact values and Windows lifecycle checks pass 27
  cases/1,176 exact values. All 28 active-prefill HF logit cases pass: ten tiny
  F32, six real 0.6B Q8, six real 0.6B F32 and six 8B Q8. Every candidate
  process verifies six applies/restores on distinct target CPUs, no errors or
  leftover restriction, and exact printed logits against current production.
  Tiny F32 maximum HF error is 6.991024018e-7 against the unchanged 2e-5 bound;
  all 18 real-model cases match top-1 and all five top-5 IDs.
- **Done:** the active 1,943-token prefill plus 32 forced continuation steps
  passes for real 0.6B F32 and Q8. Each model's 33 full vectors (5,013,888
  finite floats per arm) are byte-identical to current production, with exact
  per-target NLL. Both candidate processes verify six applies/restores and
  zero decode setters. All 36 HF/prompt/continuation inputs match the earlier
  committed evidence before execution. F32 maximum HF logit error is
  0.000126362 <= 0.001; absolute mean continuation NLL differences are
  0.000000645211 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8).
- **Done:** `tools/monitor_windows.py` records timestamped system CPU/disk/GPU
  counters and per-process CPU deltas keyed by PID plus creation time. The
  24-sample controlled-load check detects the known CPU process at a median
  99.995% of one logical CPU; recorder CPU is 0.53125 s over 24.01487 s,
  including initialization. Intervals and query errors remain in the log.
- **Left:** complete prospectively planned matched comparisons and check observer
  effects once a quiet measurement window is available. The driver records
  machine activity before and throughout every matched block; no model timing
  has yet passed its preflight screen.
- **Done:** fresh Linux candidate pass-through build, native 8/8 and full
  required-HF suite 11/11 pass with unchanged snapshot source. Matched mx
  primary/one-token/nine-token continuation harness builds against the pinned
  CPU DLLs. The activity evaluator passes 20 synthetic/known-load checks.
- **In progress:** prospective three-arm comparison plan uses one outer
  warmup round plus eight measured rounds per model/workload, with complete
  matched-block replacement for detected contention (at most two replacements).
  The first preflight defers before any model launch: accessible unrelated CPU
  is 23-44% of one logical CPU and physical disk busy is 29-41% across eleven
  samples. Raw logs are retained; all 56 frozen identities recheck unchanged.
  No performance result or candidate rejection follows. Small observer effects
  remain an explicit unresolved limit.
- **Done:** fresh enabled-candidate optional 8B regression passes 37/37 checks
  against unchanged HF fixture bounds. This includes serial-step NLL coverage;
  active placement is established by the separate callback witnesses, not by
  the serial NLL path. Its timings are not performance evidence.
- **Done:** primary, one-token and nine-token follow-up correctness passes
  for both Q8 models: 12 processes, six byte-identical full-vector pairs and
  911,616 finite floats. All 20 callbacks independently verify six distinct
  physical cores and full restoration (120 applies and 120 restores total),
  with zero decode setters, placement errors or leftover restrictions.
  These fixed-token prefix/suffix checks do not replace interactive chat tests.
- **Checkpoint:** all 25 project Markdown files reviewed; stale status wording
  corrected. Supplemental commands, sources and results are archived in
  [`prefill-contention-followup-20260920.json`](benchmarks/prefill-contention-followup-20260920.json).
  All correctness jobs are terminal. Timing remains deferred; preserve the
  first preflight and use a fresh output directory for the next comparison.
- **Latest preflight:** the next attempt also defers before launching a model:
  sustained unrelated CPU exceeds the unchanged screen. Recorder exit is zero;
  all 58 identities recheck unchanged. A minimally changed runner now requires
  a fresh `--output` path, preserving both attempts and the original runner.
  [Second preflight evidence](benchmarks/prefill-preflight-02-20260920.json).
  Turning off the only recorder would remove during-run contention evidence;
  an extra-recorder sensitivity diagnostic would not prove zero-recorder cost.
  Keep that limit explicit and assess whether such a diagnostic is useful after
  the matched comparison, without creating more measurement infrastructure now.
- **Gotchas:** activity monitoring is evidence, not proof of no interference.
  Keep observer overhead and unavailable counters explicit; no automatic
  adoption or retroactive noise claim follows from the policy change.
  The local check retains 124-126 inaccessible processes as unknown; system
  counters remain available. Controlled disk/GPU saturation and benchmark
  timing perturbation are untested. This continuation check is not full-corpus,
  maximum-context or 8B long-context HF coverage. Production runtime source
  remains unchanged; the candidate is still scratch-only.
  Full 25-file Markdown review and saved checkpoint evidence:
  [`prefill-reassessment-correctness-20260920.json`](benchmarks/prefill-reassessment-correctness-20260920.json).

### JSON validation, Unicode decoding and output escaping

- **Goal:** fix current model-description JSON handling and the documented
  ROADMAP #9b prerequisite for HF metadata/safetensors, without dependencies.
- **Done:** strict number/literal/escape syntax, classic-locale finite-double
  conversion, validated UTF-8 and UTF-16 surrogate-pair decoding, and a
  256-container nesting bound. Value/API and duplicate get-first behavior stay
  unchanged. A small string quoting helper now escapes both dequantize path
  and tensor-name fields, including quotes, backslashes and control bytes.
- **Done:** final Windows/Linux builds and native tests pass 8/8; both full
  required-HF Python suites pass 11/11. Native JSON has 1,033 checks, including
  independent expected bytes, numeric/locale limits and malformed input.
  Linux ASan+UBSan passes the same checks. Actual Q8/Q4 CLI round trips retain
  BMP/supplementary Unicode, quotes, backslashes, newline and source paths.
  Old parser/CLI regressions fail as expected. The initial new-reader Windows
  suite exposed the existing output-escaping bug; its failed result remains
  archived alongside successful final runs. A missing temporary Linux build
  directory caused one later launch to fail before executing any build/test;
  final Linux validation uses a persistent owned build directory.
- **Left:** include the validated fix with the runtime stack when its external
  requirements pass; hosted macOS execution remains unobserved locally.
  All project Markdown is reviewed at this checkpoint. Evidence:
  [`json-validation-20260920.json`](benchmarks/json-validation-20260920.json).
- **Gotchas:** 1,033 counts checks, not independent input documents. Numeric
  storage is double, not exact arbitrary-precision integers. Lone surrogates,
  invalid UTF-8, overflow and nonzero underflow to zero are rejected by policy.
  Quoting expects valid UTF-8. The later JSON conversion block checks tensor
  dimensions and byte extents; the Qwen block validates construction geometry
  and tensor layouts. Token/request checks and general filesystem-path handling
  remain separate. No inference arithmetic changes.
  Suite timings are diagnostic; existing HF/mx merge requirements stay open.

### Native CPU decode sampling (diagnostic complete)

- **Goal:** identify sampled native instruction/function locations during current
  8B decode without adding timers to runtime source. This is a separate
  diagnostic after caller attribution proved no production saving.
- **Done:** optimized PDB harness builds against 17 unchanged runtime files.
  Session 65416 completes the discarded pair, then stops on xperf's unquoted
  commas in C++ symbol fields. A separate parser recovers the saved trace;
  the initial failure and all 41 original identities remain unchanged.
  Independent recovery review pins 51 additional identities. Session 21434
  runs exactly the remaining six invocations and terminates with exit 0.
  All eight planned invocations and 16 finite full vectors pass; vectors are
  byte-identical to the historical unchanged-runtime reference.
- **Done:** independent audit rehashes both manifests, checks collector ownership,
  Running/Stopped states, fixed ready/done holds, and reconstructs all target
  samples directly from saved exports: 39,209 discarded, then 40,115, 39,733
  and 39,571 measured. Every trace has zero lost events/buffers and 100%
  named application-symbol coverage. All unknown/system samples are retained.
  Of 119,419 measured samples, 116,184 (97.291051%) land in `dot_row_impl`.

| 8B phase, mean elapsed ms | Plain | Sampled | Sampled/plain change |
|---|---:|---:|---:|
| Prefill, 215 tokens | 7060.177800 | 7022.348833 | -0.535808% |
| Decode, 32 tokens | 6970.996900 | 7034.000533 | +0.903797% |

- **Done:** exact-binary mapping verifies 345 instructions and all 1,476 code
  bytes against the frozen executable. Actual image bases and PE exception
  ranges resolve every dot sample to an instruction start; padding is excluded.
  All non-dot and caller/other-thread counts remain in the evidence.
- **Left:** external HF/mx requirements remain open. This diagnostic selects no
  production optimization and does not reopen rejected studies. Main/GitHub
  remain unchanged. Full checkpoint evidence and Markdown review are archived
  in [`cpu-native-decode-sampling-20260920.json`](benchmarks/cpu-native-decode-sampling-20260920.json).
- **Gotchas:** three measured pairs, six threads, ubatch 128 and F32 KV; one
  internal warmup per process. Decode paired elapsed changes are +2.915900%,
  -0.611112% and +0.453332%. These combine profiler/handshake/state effects
  and variability, not pure tool overhead. Samples include outside-clock gate
  activity and identify execution locations, not hardware-stall causes or
  elapsed per-operation costs. This exact-vector check supplements earlier
  HF evidence; it does not replace the independent correctness gate.

### Current decode caller-cost attribution (diagnostic complete)

- **Goal:** separate caller work from the previously mixed dispatch residual
  before selecting another optimization. No placement study is reopened.
- **Done:** unchanged production, legacy worker probes and added caller probes
  complete all 12 fixed 8B invocations in session 21493, exit 0. Six threads,
  ubatch 128, F32 KV, 215 prompt plus 32 forced tokens; one outer triplet and
  each process's first iteration are prospective warmups. All 24 full vectors
  are finite and byte-identical to the pinned unchanged-runtime vector. All 93
  frozen identities recheck unchanged. Each of four attributed traces has 32
  steps, 5,792 dispatches and 37,152 caller events, with zero accounting gap.
  Builds, native grouped/error checks, active-probe error/reuse, synthetic
  accounting and independent malformed-trace checks pass. Native Windows
  sampling resolves 6,205 of 6,221 samples to two named C++ test functions
  using local PDBs; this earlier capability checkpoint does not sample a model
  or establish a memory-stall diagnosis. The later model study is above.
- **Finding:** observed means are 217.602291 ms/token in dispatch, 1.972685
  model-side, 0.080984 backend caller work, 0.015842 harness and 0.067954
  explicit caller-observer brackets, plus 0.000014 outer timer fringe.
  Model-side includes serial backend norms/RoPE. SwiGLU is 1.157311 ms/token
  within the model-side total. Attributed decode elapsed is 2.140880% above
  plain and 1.863856% above legacy spans; paired differences against spans
  change sign. Perturbation/variation is comparable to or larger than the
  individual residual regions. Explicit brackets do not capture all observer
  effects, and standalone calibration is not subtracted from model timings.
- **Decision:** no recoverable production saving is proved. Stop the
  outside-kernel optimization direction without probe tuning, a repeat timing
  screen or runtime implementation. Keep all samples. Native function sampling
  is exercised in the separate model study above; none was selected or run
  in this earlier caller-attribution study. External HF/mx gates remain open.
- **Left:** the existing external decode performance gaps remain open. Any
  separate native model sampling needs its own fixed plan, output checks and
  unprofiled control. No production implementation follows this diagnostic.
  Independent terminal audit rechecks all 93 identities, 24 vectors and
  148,608 caller events, reproducing every integer time partition.
- **Gotchas:** the old 2.010823 ms/token residual is historical motivation,
  not a current serial-cost estimate. Region times are instrumented intervals,
  not production savings; exact final vectors are not an external HF gate.
  Production source, tests, build/CI and root executable remain unchanged.
  Full evidence: [`cpu-decode-caller-cost-20260920.json`](benchmarks/cpu-decode-caller-cost-20260920.json).

### Synchronous CPU prefill placement (old screen failed; reassessment reopened)

- **Goal:** test one backend-owned synchronous callback around the complete
  prefill, after the smaller operation-local candidate failed. Keep platform
  details below Model and include setup, callback and checked cleanup costs.
- **Done:** all 36 primary invocations completed with exit 0. Both prefill cases
  pass against fresh production and disabled prototype, with 5/5 wins against
  each. However, 8B decode mean loses 0.573915% against disabled, beyond the
  frozen 0.5% limit. Reject this integration and stop placement adoption.
  Under that original rule, follow-up timing and conditional HF/mx runs did
  not proceed. The user has since requested evaluating large gains against
  minor losses: adoption is reopened for further validation, with this failed
  screen and all samples preserved.
  Windows native checks pass 7/7; callback/tiny-model contracts pass 17 cases
  and 4,626 exact values, plus 27 Windows placement cases and 1,176 exact
  values. Linux unchanged native assertions and callback/no-op guards pass.
  Both one-thread perf smoke floors pass (placement inactive).
  All 18 separate real-model witnesses pass: exact fresh-production vectors
  for initial and one/nine-token continuation cases on both models. Enabled
  processes have 12 applies/restores for primary and 24 for follow-ups across
  two iterations; disabled witnesses and decode have zero setters. Production
  has no instrumentation. All 36 timed vectors match their witness controls.
  Preflight passes 103 checks; 216 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-prefill-callback-20260920.json`.
  Independent timing/archive audit and the full 25-file Markdown checkpoint
  review are complete.
- **Left:** fresh active-path HF/lossless checks pass in the reassessment block;
  short-follow-up and fresh matched mx validation remain open under the user's
  clarified tradeoff preference. Retain the
  bounded callback and existing mapping. Report uncertainty for small decode
  differences and gains across phases; the old cutoff failure stays recorded.
  Production scheduling remains unchanged until adoption is validated.
- **Gotchas:** scratch Backend/Model/CPU only; production source, tests, build
  files, public API and root executable are unchanged. Successful placement
  uses one apply and one checked restore pool dispatch. Partial application
  restores all changed participants before unbound fallback. Cleanup retries
  once but reports its first error; persistent refusal does not establish safe
  pool/model reuse. Initial scope is six supported Windows participants,
  mechanically quant-independent, with no wider performance claim. Source
  guards reject nesting/thread changes but do not make Model concurrent.
  Timed policy permits fallback; witness counters prove their own processes.
  Exact vectors are not an independent HF gate. The failed primary screen
  originally prevented the planned short-follow-up performance runs, so their
  timing remains unverified despite passing numerical/activation witnesses.

| Model / phase | Production mean tok/s | Disabled mean tok/s | Enabled mean tok/s | Mean vs production | Mean vs disabled | Median vs disabled |
|---|---:|---:|---:|---:|---:|---:|
| 0.6b / pp | 474.695274 | 472.415155 | 548.326819 | +15.511% | +16.069% | +17.878% |
| 0.6b / tg | 48.835786 | 50.004903 | 50.342125 | +3.084% | +0.674% | +0.222% |
| 8b / pp | 30.392742 | 30.891701 | 41.452967 | +36.391% | +34.188% | +34.854% |
| 8b / tg | 4.628661 | 4.649671 | 4.622986 | -0.123% | -0.574% | +0.004% |

### CPU-local batched-matmul placement (scratch candidate screened out)

- **Goal:** keep placement inside the CPU backend and existing callbacks,
  avoiding a generic Backend phase API or model-level platform code.
- **Done:** all 24 invocations completed with exit 0. Prefill meets the frozen
  >=5% mean/median and 4/5-win screen in both models, but 0.6B decode loses
  2.62% mean and 2.72% median throughput, beyond the 0.5% limit. Reject this
  candidate; no sample removal, rerun or inherited phase-wide result.
  All final vectors match within model and against prior unchanged controls.
  Fresh Windows lifecycle gate passes 41 cases and 11,849 exact values;
  native backend-group passes unchanged 540 cases per arm. Linux native/no-op
  checks and both one-thread perf smoke floors pass. Four real-model witness
  processes verify exact prior outputs, candidate apply/restore counts of
  4,704 for 0.6B and 6,048 for 8B across two iterations, and zero decode/control
  setters. All 130 identities were frozen before timing.
  Evidence: `docs/benchmarks/cpu-matmul-placement-20260920.json`.
  Independent timing audit and full 25-file Markdown checkpoint review pass.
- **Left:** the later whole-prefill callback above is the preferred reopened
  assessment under the user's clarified tradeoff preference. This per-operation
  arm and its failure remain archived. Production is unchanged; its prepared
  active-path HF plan was not run.
- **Gotchas:** six-thread Q8_0 nbatch > 1 scope only; other paths fall back.
  Every eligible operation queries topology and each nonempty callback pays
  apply/checked restore. Timing compares enabled/disabled prototype policy,
  with inert scaffolding in control; it is not the exact production baseline.
  Instrumented witness activation covers its own processes, not every timed
  callback. Allocation failure before dispatch propagates; dual body/restore
  failure can replace the task payload with cleanup failure. Persistent restore
  refusal is reported and only the synthetic test performs manual rescue.
  No production source, public flag, Backend API or root executable changed.

| Model / phase | Disabled mean tok/s | Enabled mean tok/s | Mean change | Median change | Enabled wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 471.700388 | 504.994000 | +7.06% | +8.73% | 4/5 |
| 0.6b / tg | 49.597335 | 48.299841 | -2.62% | -2.72% | 1/5 |
| 8b / pp | 30.040204 | 40.211011 | +33.86% | +34.11% | 5/5 |
| 8b / tg | 4.502460 | 4.575066 | +1.61% | +1.60% | 5/5 |

### Prefill placement without diagnostic observers (scratch screening passed)

- **Goal:** establish whether the prefill placement benefit survives removal
  of shared observer dispatches before considering production integration.
- **Done:** all 24 invocations completed with exit 0, and the frozen screen
  passes. Both prefill means/medians improve at least 5%, with 5/5 wins per
  model; decode means/medians stay within the 0.5% regression limit.
  Each model's 12 finite final vectors match exactly, with internal warmup
  identity also checked. All 48 source/build/check identities were frozen.
  The new parser accepts seven archived lifecycle records and rejects 34
  corruptions; six rule-boundary checks pass. Unchanged helper lifecycle
  evidence is reused explicitly, not claimed as fresh execution.
  Independent timing audit and review of all 25 Markdown files are complete.
  Full results are in `docs/benchmarks/cpu-prefill-observer-free-20260920.json`.
- **Left:** the CPU-local batched-matmul prototype above is rejected by its
  separate performance screen. The synchronous callback is implemented only
  in scratch and failed its old primary cutoff. Its adoption decision is now
  reopened above; the production Backend API remains unchanged.
  Independent HF and fresh matched mx gates remain required.
- **Gotchas:** scheduler mode constructs no placement Session. Candidate
  construction/apply/prefill/verify/checked restoration are timed; decode
  follows immediately. Stored pre-decode witnesses are serialized afterward,
  and the already-restored destructor performs no pool/affinity call.
  Formatting and inert object disposal are outside timing. There is no
  post-decode mask observation. Six threads, one prompt, Windows only;
  this remains scratch evidence, not production or HF/mx acceptance.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 479.228537 | 529.632147 | +10.52% | +11.02% | 5/5 |
| 0.6b / tg | 49.518222 | 49.633298 | +0.23% | +0.10% | 3/5 |
| 8b / pp | 29.027817 | 40.985134 | +41.19% | +40.29% | 5/5 |
| 8b / tg | 4.581855 | 4.600768 | +0.41% | +0.79% | 4/5 |

### Prefill-only CPU placement (scratch screening passed)

- **Goal:** test the measured prefill opportunity while restoring normal
  scheduling before decode, including recurring placement costs in timing.
- **Done:** the fixed 24-invocation comparison completed with exit 0 and passes
  its frozen screen. Both models improve prefill mean/median by at least 5%
  with 5/5 wins, and decode mean/median stay within the 0.5% regression limit.
  All 12 saved final vectors per model match exactly; each internal warmup
  also matches its measured iteration. Lifecycle tests pass 3,636 exact Q8 value
  comparisons, fresh iterations, exception cleanup and same-pool reuse.
  The parser accepts seven lifecycle records and rejects fifteen corruptions.
  Independent preflight passes; 46 identities were frozen before timing.
  Independent post-run audit and all 25 Markdown checkpoint reviews pass.
  Complete results are in `docs/benchmarks/cpu-prefill-placement-20260920.json`.
- **Left:** the separate observer-free study above passes its frozen screen.
  Broader thread/phase lifecycles, independent HF correctness and
  fresh matched mx performance remain required before adoption.
- **Gotchas:** this is a screening pass, not production readiness. Restoring
  masks does not reset cache, boost or scheduling state. The full candidate
  Session lifecycle, including report serialization and destruction, is timed
  as prefill. Common observation work outside both clocks can influence decode.
  The result covers one 215-token prompt, 32 forced decode steps, six threads,
  ubatch 128, F32 KV and Windows Ryzen 7 5800X only. No HF/mx floor follows.

| Model / phase | Scheduler mean tok/s | Prefill-only mean tok/s | Mean change | Median change | Candidate wins |
|---|---:|---:|---:|---:|---:|
| 0.6b / pp | 467.836623 | 535.466490 | +14.46% | +11.93% | 5/5 |
| 0.6b / tg | 49.293600 | 49.418118 | +0.25% | +0.39% | 3/5 |
| 8b / pp | 29.521998 | 40.948992 | +38.71% | +40.36% | 5/5 |
| 8b / tg | 4.594650 | 4.623768 | +0.63% | -0.30% | 3/5 |

### Explicit CPU worker placement (all-phase candidate screened out)

- **Goal:** compare scheduler-selected placement with six workers on six
  distinct queried physical cores, including caller worker zero, without
  changing kernels or production options.
- **Done:** helper success/unbound, caller/worker failure cleanup, destructor
  restoration and 1,818 exact grouped Q8 value checks pass. Independent
  preflight verifies source fidelity, masks, actual CPU witnesses and cleanup.
  All 24 real-model invocations pass. Fixed-arm before/after snapshots verify
  logical CPUs 0/2/4/6/8/10 from the same queried topology and process mask;
  every original thread mask is restored. Saved finite final vectors match
  byte-for-byte within each model, across both arms and all invocations.

  | Model / phase | Scheduler mean tok/s | Fixed mean tok/s | Mean change | Median change | Fixed wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 479.071849 | 541.704220 | +13.07% | +11.58% | 5/5 |
  | 0.6B decode | 49.314603 | 47.022156 | -4.65% | -5.70% | 0/5 |
  | 8B prefill | 30.183415 | 40.902861 | +35.51% | +35.72% | 5/5 |
  | 8B decode | 4.564495 | 4.649338 | +1.86% | +2.00% | 5/5 |

  The frozen rule requires at least 0.5% higher decode mean and median with
  4/5 wins in both models, and no prefill mean/median regression above 3%.
  Small-model decode fails; the all-phase candidate stays outside production.
  All samples are retained; no new mx or independent HF gate was run.
- **Left:** retain scheduler-selected production behavior. The separate
  prefill-only study above passed its own frozen screen with transition costs
  included; this all-phase candidate remains rejected.
  Matched external decode requirements remain open.
- **Gotchas:** topology and allowed mask are queried in each owned child;
  adjacent CPU IDs are observed, not assumed. Core placement includes serial
  caller work and warmup first-touch, and cannot isolate migration/SMT effects.
  Final-position vectors do not prove full-corpus/deep-context correctness.
  No mapping search, production flag, API or cross-platform affinity promise.
  Evidence: `benchmarks/cpu-worker-placement-20260920.json`.

### Exact decode SwiGLU callback fusion (screened out)

- **Goal:** measure unchanged SwiGLU inside the existing equal-row Q8 gate/up
  worker callback, preserving all buffers, arithmetic and projection order.
- **Done:** MSVC and GCC each pass 1,440 matrix arm comparisons, 126,990
  finite bit comparisons and 27,054 nonfinite classifications. A separate
  instrumented copy witnesses 16,920 rows exactly once; an arithmetic mutant
  compiles and is rejected numerically. Actual unchanged grouped projections
  plus an independent literal SwiGLU expression supply the synthetic reference.
  Assembly confirms original dots and scalar exp/divide/multiply order.
  All 40 fixed timing samples complete with exact gate/up/output checks.

  | Synthetic shape | Serial mean ms/call | Fused mean ms/call | Mean change | Median change | Fused wins |
  |---|---:|---:|---:|---:|---:|
  | Small gate/up + SwiGLU | 0.089504 | 0.085051 | -4.98% | -5.91% | 7/9 |
  | Large gate/up + SwiGLU | 2.442995 | 2.383465 | -2.44% | +0.25% | 7/9 |

  The frozen screen requires lower mean and median in both shapes with at
  least 6/9 wins each, plus at least 2% lower mean and median in one shape.
  The large median fails; no model integration follows. All samples retained.
- **Left:** retain current production behavior; do not weaken the prospective
  screen or repeat this experiment merely to obtain a passing sample.
  Independent HF and matched external decode requirements remain open.
- **Gotchas:** both timed arms use one concrete Q8 helper, not the generic
  production dispatch. Repeated synthetic weights can remain cached. This is
  neither a model slowdown finding nor an HF/external performance result.
  Numerical checks use the standard floating environment and compare within
  each compiler; NaN payload and alternate rounding-mode identity are unclaimed.
  Evidence: `benchmarks/q8-swiglu-fusion-screening-20260920.json`.

### Native Q8 bounded inner-loop follow-up (rejected)

- **Goal:** test one ordinary two-trip loop with local accumulators, preserving
  one copy of the native block body and every arithmetic operation.
- **Done:** MSVC emits two block bodies without accumulator stack traffic.
  Independent instruction review finds 55 F16C-path instructions per pair
  versus 54 for two control iterations. Arithmetic and branch counts are
  unchanged; the extra instruction reloads the feature flag inside the loop.
  No second-block work moves before the first block's final FMA/exit check.
  This fails the useful-scheduling gate; no numerical/model/timing runs follow.
- **Left:** stop this unrolling exploration. Historical and current-source 8B
  whole-operation diagnostics are complete below. Within-group attribution
  remains unresolved; use the matrix-cost findings to select the next study.
- **Gotchas:** instruction counts are not micro-op counts or measured latency.
  No hints, forced inlining, feature specialization or duplicated kernel bodies
  were used. Production remains unchanged. Evidence:
  `benchmarks/q8-bounded-inner-loop-rejection-20260920.json`.

### Native Q8 block scheduling study (rejected at codegen gate)

- **Goal:** expose two consecutive native Q8 blocks to compiler scheduling while
  preserving every weight product, four FMA chains and the original reduction.
- **Done:** isolated control/candidate comparators build with MSVC. Independent
  scalar/control oracles pass 612,267 finite bit checks and 1,939 nonfinite
  classifications on both MSVC and GCC. A repeated-block mutant compiles and
  fails numerically. The two-block lambda preserves the original HADD epilogue,
  block order and odd tail, but MSVC emits eight unconditional 32-byte
  accumulator stack stores per two-block iteration, including the F16C path.
  This fails the planned no-spill codegen gate; the formulation is rejected.
- **Left:** no adoption or model timing for this formulation. Continue the
  external decode performance work from the unchanged production kernel.
- **Gotchas:** this is a codegen rejection, not a measured slowdown. The oracle
  is scoped numerical evidence, not an HF/model validation claim. Prior
  pointer, feature-specialization and integer-unroll studies remain distinct.
  Evidence is in
  `benchmarks/q8-native-block-scheduling-rejection-20260920.json`.

### Exact Q8 horizontal reduction study (rejected)

- **Goal:** reduce Q8 row-dot epilogue cost while preserving all float products,
  FMA chains and contributing addition order.
- **Done:** the scratch shuffle/add sequence emits the intended instructions.
  Native CTest passes 7/7 and required-HF suites pass 11/11 for control/candidate
  on Windows/Linux. MSVC scalar checks pass 612,267 finite bit comparisons and
  1,939 nonfinite classifications; a wrong shuffle is rejected. Windows repeated
  control and candidate match 5,013,888 long-prompt logits and four serial NLL
  cases, within HF bounds. Windows candidate 8B HF checks pass 37/37.
  All 60 fixed timing processes completed: one discarded warmup plus nine
  measured rounds per model/arm, with every sample retained.

  | Mean tok/s | Current | Candidate | mx | Candidate/current | Candidate/mx |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 473.669 | 453.120 | 277.206 | -4.34% | +63.46% |
  | 0.6B decode | 48.966 | 48.723 | 49.969 | -0.50% | -2.49% |
  | 8B prefill | 29.646 | 29.936 | 21.348 | +0.98% | +40.23% |
  | 8B decode | 4.579 | 4.549 | 4.610 | -0.65% | -1.31% |

  Both decode medians also trail current/mx. Candidate wins only 3/9 decode
  pairs against current and 0/9 against mx on each model. The reduction is
  rejected; production stays at `bf122fd`, and the native regression proposal
  remains unapplied. Validation evidence is in
  `docs/benchmarks/q8-exact-reduction-validation-20260920.json`; completed timing
  is in `docs/benchmarks/q8-exact-reduction-performance-20260920.json`.
- **Left:** close the separate production decode floor. Current control means
  in this session trail mx by 2.01% (0.6B) and 0.67% (8B); do not pool this with
  earlier sessions or revive the rejected epilogue based on selected samples.
- **Gotchas:** the low 0.6B candidate prefill sample (357.111 tok/s) stays in the
  mean. Its median is 471.589 versus current 476.152; do not describe the mean
  gap as a universal causal slowdown. Correctness alone does not justify this
  performance change. NaN payloads and alternate rounding modes remain unclaimed.


### Optional Qwen3-8B HF consumer

- **Goal:** compare the verified local Q8_0 GGUF against the independent pinned
  8B HF tokenizer, logit and PPL goldens without adding large CI downloads.
- **Done:** original HF generation and provenance evidence are committed in
  `bf122fd`. Before any llmx 8B comparison, declare exact tokenizer/input IDs,
  six exact top-1 matches and top-5 set overlap 5/5; all ten printed logits must
  be finite, sorted, unique-token and within absolute magnitude 100. NLL delta
  limits are 0.01 continuous and 0.02 for each windowed case, prospectively
  reusing the existing Q8 quality budget, not calibrated from 8B results.
  Consumer implementation and independent review are complete. Both platforms pass
  all 37 checks: 20 tokenizer cases, six prompt-ID/ranking pairs, PPL IDs and
  four NLL cases. Largest NLL delta is 0.002185355, under its 0.02 limit. The
  full required-HF suite passes 11/11 on each platform, including consumer
  rejection tests (Linux perf is diagnostic only). The official pinned GGUF is
  downloaded and hash-verified in the Linux cache documented in ASSETS. The
  successful Linux gate used an identical staged copy, removed only after its
  tests finished. Interrupted mounted-file results and separate intervention
  metadata are preserved. Copy plus verification took 59.17 seconds; direct
  download plus verification took 486.61 seconds. These are operational I/O
  observations, not inference measurements. Prefer an existing verified copy
  when faster and reuse the completed Linux cache. Windows original untouched.
  All 25 Markdown files reviewed and stale performance wording corrected.
  Evidence: `benchmarks/hf-8b-validation-20260920.json`.
- **Left:** merge the validated consumer with the runtime stack after its
  external performance gates pass. Keep failures with original bounds;
  investigate rather than relaxing thresholds to fit observations.
- **Gotchas:** exact original GGUF conversion revision is undocumented. The
  official model-family link and file hash do not prove identical source
  weights. Fixed excerpt/rank checks do not cover full corpus or all logits.

### CPU ordered prefill reductions

- **Goal:** determine whether the four-row/three-column prefill kernel's
  addressable accumulator array adds avoidable stack traffic or reduction
  overhead, without changing per-lane FMA or final addition order.
- **Done:** explicit ordered reductions match 1,824 scalar-FMA outputs across
  dimension tails and unaligned inputs. MSVC assembly removes most epilogue
  accumulator stack traffic; the FMA loops do not spill in either arm. The
  function grows from 1,119 to 2,931 bytes. Nine alternating rounds improve
  mean Q8/F32 prefill by 4.59%/7.01%, winning 8/9 and 9/9 pairs. A separate
  nine-round Q8 follow-up repeats the prefill gain (+6.25%, 9/9 pairs).
  Q8 decode changes from -1.81% to +0.71% versus control between sessions;
  no stable decode regression or universal external parity is established.
  All control/candidate final-vector hashes match. Windows/Linux native checks
  pass 7/7 and required-HF suites pass 10/10. Linux real-F32 tokenizer/logit/NLL
  checks also pass. Both platforms check each arm against the same scalar-FMA
  oracle; an MSVC mutant swapping final additions is rejected. Fresh long
  comparisons have 5,013,888 byte-identical logits per F32/Q8 model. F32 maximum
  HF error is 0.000126362 <= 0.001; prefilled continuation NLL deltas are
  0.000000645 <= 0.0001 (F32) and 0.007011817 <= 0.01 (Q8). Four separate serial
  NLL cases equal control and pass existing HF bounds. All 25 Markdown files
  reviewed; evidence: `benchmarks/prefill-ordered-reduction-20260920.json`.
- **Left:** keep this validated feature branch; merge with the
  runtime stack only when broader external performance requirements pass.
- **Gotchas:** no reassociation or new activation quantization. A synthetic
  gain alone does not establish the external floor. Keep worker implementation
  unchanged and isolate timing from other builds/tests and user inference.

### Pinned HF reference generation

- **Goal:** reuse independent HF tokenizer/logit/PPL generation for explicitly
  pinned models, keeping larger-model fixtures separate from the existing suite.
- **Done:** model/revision/output and associated GGUF labels are explicit;
  numerical loaders share pinned CPU float32 eager execution. Alternate models
  require separate output and cannot overwrite the default fixture directory.
  Windows full required-HF suite passes 10/10 components against the unchanged
  validated 9cfe43f executable; Linux generator safeguards pass 2/2 tests.
  Two offline generations from cached 0.6B HF weights reproduce every numerical
  field. Against committed goldens: 20 tokenizer cases, six top-10 ID lists,
  247 PPL token IDs, four NLL values and synthetic F32 JSON match exactly.
  Rounded top-10 logit values differ by at most 0.0001. Existing fixtures,
  acceptance bounds and default CI model downloads are unchanged. Evidence:
  `benchmarks/hf-reference-tools-20260919.json`.
  Parallel work on the Linux machine generated actual 8B tokenizer/logit/PPL references from
  verified original `Qwen/Qwen3-8B` at pinned `b968826d9c46dd6066d109eabc6255188de91218`.
  All three modes pass with CPU FP32 eager execution. Measured memory reaches
  the owned container's 40 GiB cap including file cache (3,098 limit events,
  zero OOM/kill). Evidence: `benchmarks/hf-8b-reference-20260920.json`.
- **Left:** merge the tooling with the runtime stack after its external gates
  pass. The separate 8B consumer passes Windows and Linux checks under
  predeclared bounds. Official GGUF metadata links
  the base model and matches the local Q8 digest, but exact original conversion
  revision is undocumented. Default CI downloads remain unchanged.
- **Gotchas:** do not overwrite small-model goldens with another model or expand
  default CI downloads. Tooling support alone is not an 8B correctness result.
  The tooling-only checkpoint changed no hot path and ran after worker timing.
  The parallel 8B HF work ran on the separate Linux machine; it is not a performance gate.

### Separate Q8 scale/payload storage study (screened out)

- **Goal:** test a scratch storage view with original half-scale bytes separate
  from contiguous 32-byte weight blocks, preserving all arithmetic and values.
- **Done:** MSVC and GCC each pass 612,267 finite bit comparisons, 1,939
  nonfinite classifications and 141 packing cases. Grouped/standalone witnesses
  confirm the split dot runs; a corrupted-scale mutant compiles then fails.
  Native assembly preserves the FMA chains/HADD with no loop accumulator spills.
  Fixed synthetic timing completes all six shapes and 96 samples, with exact
  output checks throughout. No samples are dropped.

  | Shape | Original mean ms/call | Split mean ms/call | Mean change | Median change | Split wins |
  |---|---:|---:|---:|---:|---:|
  | Small up | 0.047025 | 0.047276 | +0.53% | -0.51% | 4/7 |
  | Small gate/up | 0.078415 | 0.078916 | +0.64% | +0.69% | 1/7 |
  | Small down | 0.042013 | 0.041720 | -0.70% | -1.84% | 3/7 |
  | Large up | 0.865281 | 0.844761 | -2.37% | -6.07% | 5/7 |
  | Large gate/up | 2.406585 | 2.359071 | -1.97% | -2.83% | 6/7 |
  | Large down | 0.859395 | 0.850048 | -1.09% | -0.20% | 3/7 |

  The frozen advancement rule requires at least 3% mean and median improvement
  with at least 5/7 wins in every large case, and no small-case mean/median
  regression above 3%. It fails; no model integration follows this study.
- **Left:** retain production storage. Revisit only with a distinct hypothesis;
  do not weaken the screening rule or infer a real-model improvement from these
  short cached matrix measurements. External decode requirements remain open.
- **Gotchas:** packing takes 8.96/19.04/9.09 ms in the three large cases and
  adds another weight-sized retained allocation in this diagnostic. Logical
  retained bytes exclude allocator overhead and an additional unpack-validation
  temporary. No peak-RSS, model-loading, HF or external-performance claim.
  Evidence: `benchmarks/q8-split-storage-screening-20260920.json`.

### Current 8B worker-span diagnostic

- **Goal:** measure instrumentation impact on the current 8B runtime and
  attribute decode dispatch intervals to operations before selecting a change.
- **Done:** current source snapshots, plain/instrumented builds and fault check
  pass. All eight fixed processes pass: one discarded outer warmup pair and
  three alternating measured pairs. Every process internally warms up. Saved
  151,936-float final vectors match byte-for-byte across all eight invocations;
  internal warmups use FNV64. Each instrumented trace has 865 prefill and 5,792
  decode records with valid ordered timestamps and no overflow.

  | Phase time, ms | Plain mean | Plain median | Spans mean | Spans median | Mean change |
  |---|---:|---:|---:|---:|---:|
  | Prefill, 215 tokens | 7224.912 | 7215.189 | 7188.779 | 7186.463 | -0.50% |
  | Decode, 32 tokens | 6955.863 | 6934.154 | 6887.725 | 6864.781 | -0.98% |

  Paired changes reverse direction in both phases; no probe speedup is claimed.
  Instrumented decode dispatch is 213.231 ms/token within 215.241 ms/token phase
  time. Gate/up accounts for 46.82% and FFN down for 23.76% of dispatch time;
  all matrix projections total 98.54%, attention 1.46%.
- **Left:** the separate Q8 scale/payload study above missed its screening
  rule. Do not change workers or extrapolate an external performance pass from
  this three-pair diagnostic. No model integration is selected.
- **Gotchas:** all builds/tests finished before timing; all measured samples
  remain. No mx comparison or independent HF gate was run. Last-finisher entry
  can overlap other workers' compute and callback intervals can include
  descheduling. These intervals are not all recoverable overhead. Evidence:
  `benchmarks/current-8b-worker-spans-20260920.json`.

### Archived decode operation attribution

- **Goal:** label the existing 0.6B decode worker spans by operation, using the
  exact archived model/backend source and shapes to prove dispatch order.
- **Done:** all 12 traces and 54,144 decode records map to 141 dispatches per
  token: five per layer across 28 layers, then vocabulary projection. Archived
  source, guards and shapes prove the order. Exact last-finisher decomposition
  passes per operation and sums back to the original dispatch totals. In the
  historical instrumented current Q8 arm, gate/up accounts for 26.07% and
  vocabulary projection for 21.90% of dispatch time. Each of the five layer
  operations changes its control/current delta sign across three pairs.
- **Left:** within-group Q/K/V and gate/up member timing remains unresolved.
  Do not rewrite workers on this evidence. The separate current-source 8B
  plain/spans diagnostic is complete in the block above.
- **Gotchas:** archived current is `9cfe43f`, not a fresh `bf122fd` measurement.
  Model and decode paths are unchanged, but prefill source/compiler layout
  differs. Instrumentation perturbs timings; these are neither external-floor
  results nor evidence of a causal regression. Evidence:
  `benchmarks/worker-decode-operation-attribution-20260920.json`.

### CPU worker cost profile

- **Goal:** localize the remaining prefill/decode costs before selecting another
  hot-path change; compare the pre-error worker control with the retained runtime.
- **Done:** operation-level profiles cover 24 processes: Q8/F32, one/six
  workers, both source arms, three alternating pairs and two instrumented
  sequences after an uninstrumented warmup. Final-vector hashes agree with
  warmups, across source arms and thread counts. The initial instrument double
  counted attention's nested parallel_for; its consistency check rejected the
  run and the corrected probe excludes nested calls. Full samples and sources:
  `benchmarks/cpu-worker-profile-20260919.json`.
  Release definitions, DLL imports and the archived DLL hash establish OpenMP
  workers/barriers in the mx reference. Historical measurements are retained;
  ASSETS and the original artifact now carry a dated interpretation correction.
  The separate per-participant probe is complete: 24 serial Q8/F32 processes,
  six participants, three alternating pairs of plain/instrumented builds for
  both snapshots. Final full vectors match byte-for-byte across every arm and
  mode; all traces have 673 prefill and 4512 decode dispatches with ordered
  timestamps. The instrumented current failure/drain/reuse check passes.
  Evidence: `benchmarks/cpu-worker-spans-20260919.json`.
- **Left:** whole decode operations are now labeled by the verified archived
  dispatch order (see attribution block above). Individual projections inside
  grouped callbacks remain unresolved, and instrumentation changes timing
  materially. No production runtime change is selected.
  Keep the existing worker implementation and the external performance gate;
  do not repeat rejected dispatch variants on this evidence.
- **Gotchas:** instrumentation changes timing. The three-process comparison
  locates costs but does not clear a small regression or prove causality.
  Keep profiles isolated from builds/tests and user inference. Reference source
  may be inspected but cannot be copied into llmx.

| Instrumented phase mean ms, six workers | Before worker fix | Retained runtime |
|---|---:|---:|
| Q8 prefill | 562.43 | 573.38 |
| Q8 decode, 32 steps | 764.09 | 762.44 |
| F32 prefill | 617.72 | 626.08 |
| F32 decode, 32 steps | 2443.77 | 2431.62 |

Q/K/V and gate/up account for 10.57 ms of the 10.95 ms mean Q8 prefill difference
and 7.94 ms of the 8.36 ms F32 difference. These include dispatch/wait time.
Q8 prefill medians reverse the small mean ordering (569.89 vs 565.93 ms), so
this diagnostic does not establish a stable regression magnitude. The next
table is the separate completed span probe; do not pool the two sessions.

| Span-probe phase mean ms | Before plain | Retained plain | Before instrumented | Retained instrumented |
|---|---:|---:|---:|---:|
| Q8 prefill | 632.945 | 567.707 | 591.131 | 574.741 |
| Q8 decode, 32 steps | 803.943 | 762.698 | 779.716 | 783.178 |
| F32 prefill | 638.665 | 627.730 | 645.636 | 635.732 |
| F32 decode, 32 steps | 2463.925 | 2479.028 | 2458.478 | 2562.645 |

Q8 decode's control/current ordering reverses with instrumentation. F32 decode
differs by +0.61% in plain builds but +4.24% in instrumented builds. The exact
last-finisher decomposition separates entry, callback and final completion;
it does not turn overlapping participant spans into additive phase costs.
These results do not identify a stable worker regression or prove its absence.
No mx benchmark or new independent HF gate was run by this scratch probe.

### Live generation and loading progress

- **Goal:** stream generated text immediately in chat/generate, show prompt
  processing before the first token, and make loader progress reusable by
  current CLI consumers and future serving (ROADMAP #7).
- **Done:** optional synchronous loader byte-progress and inference text callbacks
  are implemented; CLI generate/chat owns terminal detection, stderr status and
  stdout flushing. Prompt-processing status appears before prefill. Normal
  Qwen3 output and --think stream before the next model step; legacy retroactive
  filters retain their prior buffered behavior. Stop/EOS and follow-up cache
  accounting remain unchanged.
  Windows/Linux full required-HF suites pass all nine components. After review
  tightened completion ordering, final native checks pass 7/7 on both platforms;
  Linux follow-up chat/progress and version checks pass again. Four MSVC mutants
  fail as intended: delayed delivery, missing flush, ignored read failures and
  completion before a failing trailing seek. Full documentation review covers
  all 25 Markdown files, including current capabilities, CLI defaults and test
  scope. Evidence: `benchmarks/live-generation-20260919.json`.
- **Left:** merge with the enclosing runtime stack only after its external
  performance gates pass. GitHub remains
  main-only. Next runtime work should profile the remaining CPU costs.
- **Gotchas:** callbacks are synchronous and do not provide concurrent execution
  or resumable-session recovery. Byte chunks can split UTF-8 characters. Loading
  counts tensor payload bytes, not metadata/padding or model preparation; final
  completion follows every read/seek. The later GGUF validation checkpoint
  checks file extents before progress starts, and the Qwen checkpoint validates
  construction geometry/layouts. Token/request checks remain separate work.
  Legacy filtering buffers text when future markers can
  retroactively discard it; no server framework is added.

| Qwen3-0.6B Q8_0, 64 greedy tokens, median of 3 pairs | Before 8226e17 | Streaming |
|---|---:|---:|
| First visible text from process start (s) | 2.663 | 0.662 |
| Whole process elapsed (s) | 2.738 | 2.741 |
| CLI generation (tok/s) | 31.48 | 31.77 |

Same model/prompt, six CPU workers, stdout pipe and --think; one outer warmup
pair excluded. Every output byte matches. Builds/tests were stopped during
these runs. This is end-user delivery latency, including loading and prefill,
not a kernel-speedup or external mx-llama.cpp parity claim.

### Automatic build identification

- **Goal:** identify each CMake/plain MSVC build by release version plus Git
  revision, with a dirty marker for tracked changes and a clear archive fallback.
- **Done:** CMake refreshes build revision on each build, without rewriting an
  unchanged header. Plain build.bat emits the same metadata. --version and the
  usage banner show release plus Git revision and tracked-dirty state, with
  unknown fallback outside a checkout. Windows plain/CMake clean, dirty,
  new-commit, archive and no-op cases pass in a path containing spaces; Linux
  clean/dirty/new-commit rebuilds and version smoke pass. The current project
  build also reports its actual HEAD plus dirty state. Windows full required-HF suite passes with
  the new version regression. The Windows for/f equals-sign parsing issue was
  caught and fixed before the passing rerun.
  README now separates implemented CPU capabilities from future execution,
  serving and HF goals. All source/comments/docs and new messages use ASCII;
  Unicode fixture data is preserved. The requirement is recorded in AGENTS.
- **Left:** merge with the validated stack after its external gates pass.
  Live generation/progress reached checkpoint 9cfe43f; subsequent CPU cost
  profiles are recorded above. Full evidence for build identification is in
  `benchmarks/build-version-20260919.json`; all 25 Markdown files were reviewed
  for current capabilities, future goals, build behavior and ASCII compliance.
- **Gotchas:** untracked files do not mark a build dirty. Source archives report
  unknown even when nested in another repo. No timestamps, automatic release
  increments, commits/tags or new build/runtime dependencies are introduced.

### CLI thread settings

- **Goal:** make existing auto, decode and prefill thread flags work consistently
  for generate, follow-up chat and bench without changing kernel arithmetic.
- **Done:** generate/chat capture the resolved decode count, apply the prefill
  count and restore decode for every turn. Bench keeps auto selection; actual
  phase counts are visible through verbose/benchmark output. The new regression
  covers 32 generation/chat configurations and four bench cases per platform,
  with 64 HF-golden replies, and rejects all three reintroduced bug mutants.
  Windows full required-HF suite passes. Linux native tests and every correctness
  component pass; its initial automatic-thread synthetic floor fails. Pinning
  that test to its original one-worker conditions passes on both platforms,
  without changing floors. Logs retain the initial failure and the targeted
  reruns. Kernel/model comparator code is byte-identical to c072af2.
- **Left:** merge with the enclosing validated runtime stack once its external
  performance gates pass. Evidence: `benchmarks/cli-threads-20260919.json`.
- **Gotchas:** changing phase counts recreates the CPU pool. Automatic counts
  can be slower for tiny synthetic jobs, particularly under WSL; do not compare
  old serial-default benchmark results with new auto-default results. GPU
  backends will retain CPU-worker meaning for these flags; ubatch remains
  prompt tokens per forward pass. No GPU execution interface was introduced.

### CPU worker exception safety

- **Goal:** propagate CPU task failures after every participant finishes, with
  safe job lifetime and reusable dispatch state; clean up partial pool startup.
  This repairs the existing backend before ROADMAP #4a/#7 execution work.
- **Done:** dispatch catches caller/worker failures and waits for completion
  before rethrowing; startup joins partially created pools. Windows/Linux
  native checks, full suites with required real HF fixtures, Linux UBSan native
  tests and allocation/task fault sweeps pass. The original pool terminates on
  the task and partial-startup regressions. Independent real F32 HF checks pass;
  long F32/Q8 vectors and continuous/window NLL are exact against the control.
  Full samples, hashes, commands, logs and harnesses are archived in
  `docs/benchmarks/worker-errors-cpu-20260919.json`.
- **Left:** investigate the Q8 prefill cost before
  adoption, and close the external Q8 decode floor. No merge. The old root
  executable was retained at that checkpoint; current deployment is recorded above. Paired candidate Q8 prefill loses
  eight of nine rounds despite overlapping ranges; do not dismiss that as noise.
- **Gotchas:** dispatch recovery does not roll back partially written outputs
  or establish Model/session recovery. No concurrent submissions are supported.
  Later GGUF and Qwen checkpoints address file extents, configuration geometry
  and required tensor layouts; token/request checks remain open. Control is
  `3a82284`; no merge. The feature branch holds
  the checkpoint, while the public/default branch remains unchanged.

- **Post-reboot decision:** retain the existing c072af2 implementation. The
  stored-call variant failed its longer comparison. Failure-only exception
  bookkeeping loses all five paired decode rounds. A shared non-template
  dispatch body improves exploratory prefill but does not improve decode;
  it is not adopted. No more dispatch variants are planned without profiling
  evidence. The user explicitly asked to keep the implementation simple.
  Full samples and source patches for the two latest scratch studies are in
  `docs/benchmarks/worker-cold-errors-cpu-20260919.json` and
  `docs/benchmarks/worker-shared-dispatch-cpu-20260919.json`. Both pass initial
  MSVC allocation/task-fault and grouped-kernel checks; neither entered full
  HF/platform adoption gates. Those experiments changed neither runtime source
  nor the root executable.
  Next performance work should localize the remaining cost before changing code.
  User authorization includes merging main and publishing GitHub once the
  requirements pass; current performance evidence does not clear that gate.

Stored-call experiment is **not adopted**. Full HF, exact vectors/NLL,
Windows/Linux suites and Linux UBSan native checks pass, but the longer run
does not establish a performance gain. Evidence and commands:
`docs/benchmarks/worker-invocation-cpu-20260919.json`.

| Mean tok/s, nine rounds | Before errors | Error checkpoint | Stored call | mx |
|---|---:|---:|---:|---:|
| Q8 prefill | 422.70 | 406.76 | 404.32 | 262.84 |
| Q8 decode | 44.69 | 44.93 | 44.53 | 46.32 |
| F32 prefill | 354.95 | 346.77 | 343.63 | 369.07 |
| F32 decode | 13.44 | 13.43 | 13.30 | 13.33 |

All ranges overlap. Stored-call Q8 prefill loses every pair against the
pre-error control, and no mean beats the error checkpoint. Preserve the early
five-round result as exploratory, not a reason to select the variant.

| Mean tok/s, nine matched rounds | Control | Candidate | mx-llama.cpp |
|---|---:|---:|---:|
| Q8 prefill | 408.39 | 386.18 | 263.80 |
| Q8 decode | 43.88 | 43.94 | 45.81 |
| F32 prefill | 362.85 | 377.11 | 370.26 |
| F32 decode | 13.91 | 14.06 | 13.69 |

Q8 prefill mean changes by -5.44%; Q8 decode remains -4.09% below mx. All
candidate/control ranges overlap; F32 means lead in this session without an
equivalence claim. Separate synthetic timing also has lower prefill/decode
means (see ASSETS). All samples are retained; builds/tests do not overlap timing.

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### F32 embedding/matrix inference

- **Goal:** load and run F32 embeddings and matrices in dense Qwen3, with
  external HF numerical validation and measured CPU performance (ROADMAP #8).
- **Done:** direct F32 rows, float-aligned tensor blobs, deterministic HF
  full-logit/NLL fixtures (tied/untied, odd widths, batches, threads). Windows
  and Linux full suites pass; UBSan passes and catches the pre-fix alignment
  fault. New sanitizer CI job passes workflow lint. All 311 real Qwen3-0.6B
  F32 tensors verified against original HF weights; all real HF checks pass.
  Copy/direct controls match matrix hashes and all logits over a 1,943-token
  prompt plus 32 greedy tokens. Measurements/provenance are in ASSETS.
- **Left:** close the measured F32 CPU gap versus public mx-llama.cpp
  `5542318e74`, then merge and observe the expanded five-job hosted CI.
  Current F32 results are in the KV and worker-error blocks; the older
  scale/load comparison records that earlier checkpoint. Sustained external
  parity remains unproven. Loading is excluded and each process warms up.
- **Findings:** activation tiling and a fully spinning worker pool did not
  establish a win. Profiling instead identifies scalar attention as roughly
  170-180 ms of prefill. Backend attention now improves prefill by 24.8%
  against its interleaved control and passes HF bounds, while changing
  summation order. Remaining work is the CPU performance gap, not numerical
  validation of this attention implementation.
  `tools/compare_cpu.cpp` / `.py` preserve the matched measurement procedure;
  commands and limitations are in ASSETS.
- **Gotchas:** do not treat success on quantized weights widened to F32 as
  parity with the original HF weights. Validate tied/untied output and prefill.

### CPU attention in backend

- **Goal:** move causal GQA attention out of the model and into the backend
  (ROADMAP #4a), sharing the decode and prefill implementation and improving
  CPU throughput with measured, numerically bounded vectorization.
- **Done:** `Backend::attention` now handles both forward paths, with CPU-owned
  score scratch, causal GQA, AVX2 dots and value accumulation, and scalar tails.
  Deleted duplicate model-layer attention loops. Regenerated the tiny HF
  fixture at head width 10 to cover vector tails. Windows and Linux full
  suites pass, including real Q8/Q4 HF checks; Linux UBSan synthetic suite
  passes. Tiny maximum HF logit error is 6.2e-7. A future-token attention
  mutant fails with error 0.03787. Real F32 excerpt/window NLL differs from HF
  by at most 4.5e-6. At 1,943 prompt tokens plus 32 generated tokens, greedy
  output matches HF and all 5,013,888 logits are within 0.001 (max 0.0001252).
  Eight interleaved rounds give F32 prefill 286.84 -> 357.88 tok/s (+24.8%);
  decode 13.34 -> 13.57 tok/s (small change with overlapping ranges).
  Q8 synthetic guardrails pass; matmul 123.07 -> 121.80 GFLOPS, prefill
  4828 -> 4800 and decode 4522 -> 4875 tok/s, with overlapping ranges.
- **Left:** close the remaining external CPU floor gap, then merge and run
  hosted CI. See the KV and worker-error blocks for later measurements.
  Profiling after vectorization finds prefill attention around 60-67 ms and
  decode attention around 97-100 ms; matrix operations now dominate decode.
  Sequential F32 row streaming and parallel batched elementwise work are
  implemented on the next feature branch; see its validation block below.
- **Gotchas:** SIMD dot reductions change summation order. A matching top token
  or logit sum is not a numerical gate. This step does not implement device
  buffers, resident activations or async execution, which remain prerequisites
  for GPU backends.

### CPU row streaming / parallel prefill

- **Goal:** close the F32 CPU performance gap without changing weights or
  weakening the HF numerical gate (ROADMAP #8).
- **Done:** contiguous single-row F32 decode and parallel batched
  norm/RoPE/SiLU work, guarded to keep fewer than two rows per worker serial.
  Final eight interleaved rounds: prefill 371.65 -> 383.06 tok/s (+3.1%,
  overlapping ranges), decode 13.87 -> 14.68 (+5.8%, disjoint ranges).
  Same-run mx is 398.97 / 14.89: both floors remain open (-4.0% / -1.4%).
  Windows and Linux full suites pass with required Q8/Q4 HF fixtures; UBSan
  synthetic suite passes. Real F32 HF logits/excerpt/chunked NLL pass.
  Long-prompt full logits have maximum error 0.00012636 under 0.001, with
  all greedy IDs matching HF. Q8 step guard passes with overlapping ranges.
  The batched guard caught tiny-prompt scheduling regressions; after the
  serial guard, small-batch ranges overlap the control and B=64 retains a
  17% latency improvement. ASSETS and the row-scheduling JSON contain all
  initial/final samples and validation scope.
- **Left:** close both remaining external gaps before merge and hosted CI.
  Instrumented pool profiling finds 26-28 ms after worker callbacks finish
  during 32 decode steps. Bounded completion polling did not establish a win
  over its atomic-only control (eight rounds); no pool change was adopted.
  Full spinning was already rejected. Query scheduling remains scratch-only;
  the value-accumulation kernel below is the next validated checkpoint.
- **Gotchas:** single-row dots change reduction order. Printed sums alone
  are not the numerical gate. Initial unconditional scheduling appeared above
  mx for prefill, but final guarded measurements did not; use the final run.
  The legacy bench "prefill" is repeated step(), not batched prefill, so keep
  the separate batched regression guard. Reader alignment was not adopted.

### CPU attention value accumulation

- **Goal:** reduce CPU attention output loads/stores by retaining value-sum
  lanes in SIMD registers, while preserving per-lane summation order (#4a/#8).
- **Done:** normalized coefficients and register value sums implemented, with
  32-lane blocks, eight-lane remainders and scalar tails. Expanded the tiny HF
  fixture to head width 42, deriving its tensor shapes and HF config together.
  SIMD and forced scalar-value branches pass at maximum HF error 0.00000070
  under 0.00002; a missing SIMD-block output mutant fails at 0.19888665.
  Windows/Linux full suites pass with required real Q8/Q4 fixtures; Linux
  UBSan synthetic suite passes. Real F32 HF logits and excerpt/window NLL pass.
  All 5,013,888 logits over 1,943 prompt plus 32 greedy tokens are byte-identical
  to c4436fd; HF maximum error 0.00012636 under 0.001, greedy IDs 32/32 exact.
  Q8 step and batched guards show overlapping control/candidate ranges.
  Final eight interleaved rounds:

  | Mean tok/s | Control c4436fd | Value kernel | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Prefill | 384.86 | 397.63 | 397.94 | -0.08% |
  | Decode | 14.56 | 14.57 | 14.89 | -2.10% |

  Prefill improves 3.32% with narrowly overlapping ranges; decode is unchanged
  within noise. ASSETS and `benchmarks/attention-values-cpu-20260919.json`
  retain all samples, hashes, validation logs and reproduction harnesses.
- **Left:** close the external performance gap before merge/hosted CI.
  Instrumented profiling attributes about 2,073 of 2,187 ms decode to matmul,
  including 528 ms in the output projection; attention is about 87 ms.
  Investigate matrix operations next, using matched controls and HF gates.
- **Gotchas:** means close to mx are not proof of parity. This is an interactive
  workstation and no outliers were discarded. Query scheduling and polling
  remain scratch-only. The value kernel keeps sequence order per lane; byte
  identity is established for the recorded Windows long-prompt case, not all
  inputs or compilers. Forced scalar values do not prove no-AVX ISA support.

### CPU grouped projections

- **Goal:** reduce worker-pool dispatches for Q/K/V and FFN gate/up projections
  sharing activations, preserving float arithmetic (ROADMAP #4a/#8).
- **Done:** direct grouped decode through existing F32/Q8_0/Q4_K row kernels.
  Other types, batches and small jobs use the sequential fallback. No TLS,
  nested dispatch or activation conversion. Added CTest coverage to CI.
  Windows/Linux full suites pass with required real Q8/Q4 HF fixtures;
  real F32 HF and UBSan synthetic/backend checks pass. Backend tests cover
  540 cases and 141,750 outputs per platform; a missing-row mutant fails.
  Instrumented Q8 run observes 3,584 groups avoiding 5,376 dispatches across
  warmup plus measured sequences. Four excerpt/window NLL cases per model
  match the previous runtime exactly. All 5,013,888 long-prompt logits per
  model (F32 and Q8) are byte-identical to the previous runtime. F32 remains
  within the independent HF bound, with identical greedy continuation.
  Final eight interleaved rounds after a discarded warmup round:

  | Mean tok/s | Previous 5a9518c | Grouped | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8_0 prefill | 418.59 | 419.06 | 267.76 | +56.51% |
  | Q8_0 decode | 43.92 | 45.18 | 48.34 | -6.55% |
  | F32 prefill | 397.41 | 394.25 | 390.07 | +1.07% |
  | F32 decode | 14.47 | 14.64 | 14.86 | -1.49% |

  Q8 decode mean/median improve 2.86%/3.19%; ranges overlap due to one slower
  candidate sample, retained in the result. F32 prefill/decode changes are
  within overlapping ranges. Synthetic matmul mean 129.57 -> 127.64 GFLOPS
  (median 129.78 -> 129.51, ranges overlap); step-based prefill/decode improve
  4998/4928 -> 6861/6921 tok/s. Batched guards retain overlapping ranges except
  the faster six-thread single-token case. Raw samples, hashes, exact-output
  gates and reproduction sources are in ASSETS and
  `benchmarks/grouped-projections-cpu-20260919.json`.
  The follow-up real 8B Q8_0 diagnostic also preserves all 5,013,888 logits
  on its matched 215+32-token history. Three measured rounds, after warmup:

  | 8B mean tok/s | Previous | Grouped | mx |
  |---|---:|---:|---:|
  | Prefill | 29.12 | 29.25 | 21.20 |
  | Decode | 4.31 | 4.33 | 4.49 |

  Control/group ranges overlap; this does not establish a small speedup.
  Decode remains below mx. Exact-vector equality is against previous llmx,
  not an independent 8B HF baseline. Evidence and the full-vector control are
  in `benchmarks/grouped-projections-8b-20260919.json`.
- **Left:** close the external decode floors before merge and hosted CI.
  Native Q8 scale/load scheduling is implemented and validated below.
- **Gotchas:** exact equality is scoped to tested inputs/platform, not a
  full-corpus or maximum-context proof. The legacy bench prefill uses step();
  batched prefill has a separate guard. All outliers retained; do not pool
  absolute rates from separate sessions. Candidate is validated but unmerged.

### CPU Q8 scale / load scheduling

- **Goal:** reduce native Q8 decode instruction overhead while preserving
  float activations, per-lane accumulation order and exact weight scales (#8).
- **Done:** selected direct memory half broadcast plus direct byte-load sign
  extension. Assembly confirms the intended instructions; scalar fallback and
  accumulator order are unchanged. Feature specialization did not establish a
  further gain and is excluded. Exhaustive finite-half scale/signed-weight
  regression passes, and a wrong-half-offset mutant fails. Windows/Linux full
  suites with required Q8/Q4 HF fixtures, real F32 HF checks, backend CTest and
  UBSan synthetic/backend checks pass. Instrumentation confirms real Q8 use.
  Recorded F32/Q8 long vectors and excerpt/window NLL are exactly unchanged
  from `b6a890f`; the same-weight 8B vector comparison also passes.

  | Eight-round mean tok/s | Before b6a890f | Candidate | mx |
  |---|---:|---:|---:|
  | Qwen3-0.6B Q8 prefill | 422.99 | 418.43 | 273.51 |
  | Qwen3-0.6B Q8 decode | 45.30 | 46.54 | 48.35 |
  | Qwen3-0.6B F32 prefill | 398.17 | 395.23 | 394.92 |
  | Qwen3-0.6B F32 decode | 14.70 | 14.77 | 14.98 |

  Q8 prefill and F32 ranges overlap. Q8 decode improves but remains below mx.
  Root CLI and production comparator code hashes match their validated arms.
  The larger-model diagnostic also improves decode but does not establish
  external parity:

  | Qwen3-8B Q8 mean tok/s | Before | Candidate | mx |
  |---|---:|---:|---:|
  | Prefill | 28.97 | 29.04 | 21.13 |
  | Decode | 4.29 | 4.45 | 4.49 |

  The short synthetic guard's threaded single-token regression does not
  retain disjoint ranges in the longer follow-up. Paired changes run in both
  directions; the higher candidate mean remains visible:

  | Synthetic prefill mean ms | Before | Candidate |
  |---|---:|---:|
  | 1 thread, 1 token | 0.16995 | 0.12580 |
  | 6 threads, 1 token | 0.15864 | 0.16167 |

  ASSETS and `benchmarks/q8-scale-load-cpu-20260919.json` retain all samples,
  controls, hashes, HF/platform logs, prototype patches and reproduction code.
- **Left:** close the external decode floors before merge and hosted CI.
  Continue matrix-operation work with matched model measurements; retain the
  threaded tiny-prompt guard because zero slowdown has not been established.
- **Gotchas:** retain F16 subnormal/negative-scale behavior and scalar fallback;
  do not multiply activations by scales instead, which changes rounding.
  Exact equality is scoped to recorded inputs/platforms. The 8B equality check
  is against previous llmx, not an independent HF 8B reference. All samples
  are retained; no external parity claim.

### CPU comparison thread scaling

- **Goal:** locate the remaining CPU performance gap using matched thread
  counts and matrix-shape measurements, following ROADMAP #8.
- **Done:** explicit `--threads` in the comparator and runner, with requested
  counts echoed and checked. Windows llmx/mx and Linux llmx builds pass;
  invalid arguments and missing/wrong thread metadata are rejected. A real-model
  runner smoke passes. Matched Q8/F32 scaling is complete; those measurements
  used runtime `475f312`, before the KV and worker-error changes.
  Full vectors are byte-identical across all measured counts; continuous NLL
  is unchanged from the prior runtime and passes the independent HF fixture.
  Projection and matrix probes are complete. F32 matrix ranges overlap mx;
  Q8 matrix latency remains higher at the default comparison count. ASSETS and
  `benchmarks/cpu-thread-scaling-20260919.json` contain complete results.
- **Left:** merge the tool with the validated stack once its external floor
  is met. The resulting contiguous per-head KV change is implemented and
  validated below; it is no longer a pending experiment.
- **Gotchas:** use the same thread count in both arms and record it with every
  result. Keep the pinned model, tokens, reference revision, warmup and KV
  settings. Scaling diagnostics do not waive the existing external floor.

### Head-major CPU KV storage

- **Goal:** make each KV head's history contiguous to improve attention reads,
  preserve arithmetic order, and separate concrete CPU storage from logical
  sequence state without adding speculative device/server interfaces.
- **Done:** `HostKVCache` owns bounded growth and token-major projection writes;
  `Model` owns valid length/reset; backend attention receives an explicit head
  stride. Promoted the exact validated headers from the scratch candidate.
  Growth/reset/mixed histories match all 229,758 control values on Windows
  and Linux (the latter with nonrecovering UBSan). The direct storage oracle
  is in CTest and rejects a wrong-head relocation mutant.
  Real F32 HF logits and continuous/window NLL pass; long HF error is at most
  0.00012636185 under 0.001, with all 32 greedy IDs matching. F32 and Q8 each
  retain all 5,013,888 long-history logits and four full-precision NLL cases
  exactly versus `475f312`. Windows/Linux full suites with required real HF
  fixtures and Linux UBSan native/synthetic suites pass. Integrated MSVC code
  has the same `.text` hash as the validated candidate; final native CTest
  integration passes on Linux and UBSan.
  Nine interleaved matched rounds, all outliers retained:

  | Mean tok/s | Control | Head-major KV | mx | Gap vs mx |
  |---|---:|---:|---:|---:|
  | Q8 prefill | 366.46 | 416.37 | 262.34 | +58.72% |
  | Q8 decode | 42.36 | 44.41 | 45.51 | -2.42% |
  | F32 prefill | 345.42 | 361.38 | 362.44 | -0.29% |
  | F32 decode | 13.06 | 13.47 | 13.27 | +1.55% |

  Candidate/control ranges overlap; paired candidate wins are 9/9, 7/9,
  7/9 and 8/9 respectively. These results support an incremental selection,
  not a claim that the external floors are closed. The separate tiny synthetic
  decode mean declines 2.88% with overlapping ranges; that remains a recorded
  limitation. ASSETS and `benchmarks/head-major-kv-cpu-20260919.json` preserve
  all samples, hashes, scopes and reproduction sources. Earlier short-run
  diagnostics remain in `head-major-kv-initial-20260919.json`.
- **Left:** close the remaining external Q8 decode and F32 prefill gaps, then
  merge the validated stack and observe hosted CI. No merge or publish yet.
- **Gotchas:** reset retains allocation but must not expose stale tokens.
  Growth temporarily holds old and replacement storage together; future
  multi-user memory budgets must account for that peak. A CPU head-major layout
  is not a requirement for future device buffers, paging or shared prefixes.

### Chat follow-up cache validation

- **Goal:** preserve correct conversation history across follow-up prompts,
  reusing KV only when its exact token prefix matches the rendered transcript.
- **Done:** review found that `cmd_chat` skips cached tokens by count alone,
  renders twice per turn, and can pass empty logits to generation when the
  template does not add a generation suffix. Generated stop tokens and the
  unconditional EOS step also need accurate cache accounting.
  The new HF-backed chat regression reproduced a crash on the old binary.
  It also exposed double consumption of template block terminators, which
  skips adjacent content and breaks nested conditionals/loops.
  Implemented exact fed-token tracking and reset/refill for changed prefixes;
  removed unconditional EOS insertion and the redundant prefill/render pass.
  Fixed block terminator consumption and first keyword argument parsing. The
  latter affected Qwen namespace state and removal of old reasoning. New
  end-to-end HF reply fixtures pass; a token-count-only reuse mutant fails.
  The real Qwen template matches Jinja2 across initial and follow-up histories.
  Windows and Linux full suites with required real HF fixtures pass. MSVC
  renderer test, Linux CTest and Linux UBSan native/synthetic suites pass.
  Linux first exposed a missing `<cmath>` include in the standalone renderer;
  that is fixed. Chat coverage is 54 runs of nine scenarios across thread and
  batch settings; the real template has 12 independent Jinja2 cases. Logs,
  fixture hashes and reproduction commands are in
  `benchmarks/chat-followup-validation-20260919.json`.
- **Left:** merge with a validated runtime stack and observe hosted CI. The
  enclosing stack still needs its external performance floors; chat fixes do
  not establish those floors. Numerical kernels and model forward paths are
  unchanged by this fix.
- **Gotchas:** prefill continuation tests alone do not validate chat-template
  reuse. Keep messages separate from cached tokens; rendered text may change
  earlier turns or retokenize their boundary. This is single-user chat, not
  multi-user or concurrent request support.

### Correctness baseline vs HF reference

- **Goal:** give the suite an external ground truth. Correctness is measured
  against the HF reference, never against llmx itself (`docs/ROADMAP.md` #8).
- **Done:**
  - The tokenizer mode of `tools/gen_baseline.py` emits golden fixtures using
    `tokenizers` + `huggingface_hub`; numerical modes additionally need torch
    and transformers. Tokenizer output is committed to
    `tests/data/baseline_tokenizer.json`.
  - `tests/baseline.py` compares llmx against the committed goldens and is
    wired into `tests/run_tests.py`. It SKIPS when no fixture model is on disk,
    so the rest of the suite still runs anywhere.
  - Tokenizer parity: **20/20 cases match pinned Qwen/Qwen3-0.6B.**
  - HF fp32 top-10 logit goldens for six prompts landed in `1c2e102`.
    The baseline checks top-1, top-5 set overlap and a magnitude bound on
    Qwen3-0.6B Q8_0 and mixed Q4_0 fixtures. Both passed all six prompts in
    that commit; the mixed fixture catches the f16 subnormal regression.
  - Local golden generation works in an isolated environment with
    numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2.
  - Pinned HF fp32 PPL golden for a 247-token wikitext excerpt: exact token
    IDs/count and finite NLL/PPL checks, with per-quant absolute NLL bounds.
    HF PPL 28.7974; Q8_0 28.8371 (NLL delta 0.001374 <= 0.01); mixed Q4_0
    32.8463 (delta 0.131554 <= 0.16). Both llmx arms repeated identically at
    printed precision. Generator, provenance and bounds are in `docs/ASSETS.md`.
    Eight injected bad-output cases were rejected; build and full suite pass.
  - Nine bugs found and fixed via this path, all of which survived a green
    suite: attention missing 1/sqrt(head_dim); temperature cancelling in the
    sampler; RoPE read past context_length; the GPT-2 whitespace guard that
    could never fire; attention width hardcoded to n_embd; tied embeddings
    unsupported; Windows argv delivered in the ANSI codepage so any non-ASCII
    prompt was mangled before llmx saw it; and the pretokenizer implementing
    the GPT-2 regex instead of the Qwen2/Qwen3 one; the byte encoder incorrectly
    including soft-hyphen byte 0xAD in its printable set. Four new HF cases
    reject the preserved old encoder. Whole-wikitext file input now tokenizes
    298,938 tokens and scores the requested window limit successfully.
- **Left:**
  - Per-layer activation and full-corpus PPL goldens, plus maximum-context validation.
    The 1,943-token/32-step independent F32 HF check is complete; its scope and
    full-vector quantized diagnostics are recorded in ASSETS.
    Continuous and chunked excerpt gates are implemented, with an explicit
    disjoint-window scoring policy (`docs/USAGE.md`). They are not full-corpus
    coverage. Two-token windows show large quantized/HF deviations even under
    the unchanged old scorer; diagnostics and bounds are in `docs/ASSETS.md`.
    The existing ranking gate does not bound full-vector numerical error.
  - Existing tiny F32 full-vector and real-model excerpt bounds are implemented.
    Broader Q8 full-vector and full-corpus acceptance bounds remain open.
- **Gotchas:**
  - Round-trip and synthetic tests alone are not an external correctness gate;
    preserve the HF tokenizer/logit checks and extend their coverage.
  - The 8B hides bugs the 0.6B exposes: `n_head * head_dim == n_embd` holds for
    Qwen3-8B (32*128 == 4096) and fails for 0.6B/1.7B/4B.
  - torch is a fixture-GENERATION dependency only, never needed to run the
    suite and never at runtime.
  - F32 embedding/matrix support and a tight HF numerical gate are implemented
    on the active feature branch. Its external performance gate remains open;
    see the F32 block above before merging.
  - Generation's legacy reasoning filter searches `thinking_start/end`, not
    Qwen3's actual `<think>` / `</think>` markers. Its docs now state that limit.
  - The CLI thread-settings block records the validated auto/prefill/decode
    corrections found during the documentation review.
  - JSON syntax/Unicode and conversion tensor dimensions/extents are implemented
    in their active blocks above. Qwen construction now validates configuration
    geometry and required tensor layouts; token/request checks remain separate. See
    docs/src/core-json.md.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Preceding two-arm result (2026-09-20):** unchanged validated `bf122fd`
  runtime versus pinned mx `5542318e74`, nine alternating measured pairs per
  model after one discarded warmup pair; six workers, ubatch 128, 215 prompt
  plus 32 forced tokens and F32 KV. Every process exits zero and llmx final
  output hashes repeat within each model. No timing overlaps other compute.

  | Q8 model / phase | llmx mean tok/s | mx mean tok/s | Mean gap | llmx / mx median | Paired wins |
  |---|---:|---:|---:|---:|---:|
  | 0.6B prefill | 440.479 | 274.766 | +60.31% | 448.159 / 277.473 | 9/9 |
  | 0.6B decode | 47.280 | 47.589 | -0.65% | 47.362 / 48.315 | 3/9 |
  | 8B prefill | 30.170 | 22.124 | +36.37% | 30.481 / 21.994 | 9/9 |
  | 8B decode | 4.403 | 4.440 | -0.82% | 4.465 / 4.501 | 2/9 |

  Both decode means and medians remain below mx. The external gate remains
  open; close results do not meet the required floor. Preserve all measured
  rounds, including the slower first measured round, and do not pool older
  sessions. Evidence: `benchmarks/q8-current-floor-20260920.json`.
  Independent review of all 27 pending commits found no additional concrete
  publication blocker; the existing Windows/Linux HF/native evidence remains
  valid for this unchanged runtime. Hosted CI follows eventual publication.
  Earlier ordered-prefill comparisons also exceed mx in F32 prefill/decode;
  their initial and Q8 follow-up sessions retain their separate scope.
- **Earlier instruction study (no runtime change):** paired native Q8 rows regress;
  direct pointer increments and explicit row-kernel inlining do not establish
  a decode gain. Exact row/tail/fallback checks pass. Assembly confirms shared
  activation loads without inner-loop spills, removal of native-loop address
  multiplication, and removal of row calls in the respective prototypes.

  | Separate Qwen3-0.6B Q8 studies, mean decode tok/s | Control | Prototype | mx |
  |---|---:|---:|---:|
  | Paired rows | 46.00 | 43.81 | 48.31 |
  | Pointer increments, longer run | 46.22 | 45.86 | 48.22 |
  | Explicit inlining | 46.56 | 46.04 | 48.46 |
  | Inlining plus pointers | 46.56 | 46.38 | 48.46 |

  No prototype is adopted. Do not pool absolute rates across these sessions.
  ASSETS and `benchmarks/q8-row-instructions-20260919.json` retain patches,
  assembly, exact checks, complete samples, code hashes and reproduction.
  These scratch prototypes did not enter the full HF/platform adoption gate.
- **Previous investigation:** AVX2 integer dots with vectorized activation packing
  were tested at 8-bit and 16-bit precision. Q16 improves matched mean decode
  by 2.79% on 0.6B and 4.06% on 8B, but still trails mx by 5.12% / 1.21%.
  Both variants pass existing Windows HF fixture bounds; Q16 stays much closer
  to the current float path. Independent packing/product controls pass,
  including signed weight extremes, half subnormals and fallback cases.
  Across four excerpt/window cases, Q16's maximum absolute NLL change versus
  current llmx is 0.00003155. Across 5,013,888 logits on a 1,943+32-token
  forced-HF continuation, maximum change is 0.002213; maximum and RMS error
  against HF are slightly lower in this case. This is a nonzero precision
  change, not a lossless result. Full corpus, independent 8B HF, maximum
  context and cross-platform validation remain open. No candidate was adopted;
  all patches, samples and numerical controls are preserved in
  `benchmarks/q8-integer-activation-20260919.json` and ASSETS.
- **Earlier investigations:** paired F32 decode rows did not improve throughput;
  packed F32 prefill variants regressed. None was adopted. Exact patches,
  samples and diagnostics are in ASSETS and the paired-decode/packed-prefill
  benchmark JSON files. These experiments used the `5a9518c` runtime.
  Profiling scalar exponentials finds only 7.45 ms SiLU wall time and 9.42 ms
  summed softmax worker time during 2,237.50 ms decode; these are not the main
  remaining cost. The summed worker measurement is not wall time.
  The preceding matched Q8 comparison covers both the HF fixture and real 8B
  model against pinned public mx `5542318e74`: three alternating pairs,
  215+32 pinned tokens, six threads, ubatch 128 and F32 KV.

  | Q8_0 model / phase | llmx mean tok/s | mx mean tok/s | Gap |
  |---|---:|---:|---:|
  | Qwen3-0.6B prefill | 380.50 | 265.35 | +43.39% |
  | Qwen3-0.6B decode | 42.73 | 46.81 | -8.71% |
  | Qwen3-8B prefill | 25.94 | 20.07 | +29.27% |
  | Qwen3-8B decode | 4.10 | 4.39 | -6.64% |

  Prefill exceeds the reference on both models, but decode remains below it;
  both comparisons have disjoint arm ranges in each phase. Raw samples,
  hashes, flags and scope are in `benchmarks/q8-external-floor-20260919.json`.
  Source inspection confirms mx uses quantized Q8 activations and integer
  dots, while llmx retains float activations. Any analogous optimization
  needs a measured numerical cost bound before adoption.
- **Historical stand-in comparison (different conditions; not pooled):**
  Qwen3-8B Q8_0, 343-token wikitext prompt, -t 16, this workstation. Reference
  is the CPU AVX2 llama.cpp shipped with LM Studio, stock `llama-server`, same
  machine, so no time on the Linux machine was used.
    - pp   llmx 37.23 / 37.86   llama.cpp 37.70 / 37.34   -> parity
    - tg   llmx  3.91           llama.cpp  4.99 / 5.02    -> 22% under
  These are the original matched measurements. Later Q8_0 decode commits
  report 4.12 tok/s with independent accumulators (`54ea063`) and 4.24/4.18
  with F16C (`0c12570`); these are not a new matched mx-llama.cpp comparison.
  Separate decode/prefill thread flags landed in `dada6c0`.
- **How prefill got there, 3.89 -> 37.5 tok/s (9.6x), each step A/B measured:**
  - persistent worker pool instead of spawning threads per call (decode -23%)
  - attention through `Backend::parallel_for` instead of its own threads
  - batched prefill: matrix-matrix instead of one token at a time (3.89 -> 13.2)
  - dequantize each weight row once per batch, not once per column (-> 15.3)
  - four independent accumulators in the f32 dot (-> 17.0)
  - fused 4-row kernel sharing one activation load (-> 24.0)
  - two activation columns per four rows, 0.75 loads/FMA (-> 33.7)
  - three activation columns per four rows, 0.58 loads/FMA (-> 37.5)
- **The lesson worth keeping:** the kernel was LOAD bound, not FMA bound. Each
  naive dot needs 2 loads per FMA and Zen3 sustains about 2 loads/cycle against
  2 FMAs/cycle, so it ran at half of FMA peak no matter how the batch was
  blocked. Every win after the first came from raising the FMA:load ratio.
- **Left:**
  - Close the current Q8 decode deficits while retaining prefill gains. The
    latest fixed `bf122fd` control/candidate/mx session measures control mean
    gaps of -2.01% (0.6B) and -0.67% (8B), with both medians below mx. The
    preceding two-arm session measured -0.65%/-0.82%; keep the sessions separate. Earlier bandwidth/thread observations and null allocation,
    fragmentation and prefetch experiments do not predict this current gap;
    mmap has not been established as a throughput improvement.
  - Grouped Q16 activation packing is now rejected. Its own scalar/integer
    arithmetic and activation tests passed, but unchanged native accuracy
    checks fail on MSVC and GCC (control 7/7, candidate 6/7). First diagnosed
    absolute error is 0.000219106674 against the existing 0.000206180004 limit,
    on a standalone one-worker Q8 dot. No test tolerance was changed; model
    HF-cost/performance runs were stopped before execution. Evidence:
    `benchmarks/q16-group-native-rejection-20260920.json`. Bounded-cost research
    does not satisfy AGENTS' lossless requirement. Preserve the rejection;
    do not tune a new tolerance to that failing case.
  - The exact Q8 horizontal reduction is rejected by complete timing despite
    passing numerical gates. Read-only review also found two redundant `h_`
    clears in `Model::step`, but no evidence that their cost closes the gap.
    The native two-block lambda is rejected at the no-spill codegen gate.
    The ordinary inner-loop follow-up also fails the useful-scheduling gate:
    no spills, but one extra feature reload and no useful cross-block work.
    Unrolling remains closed. Caller attribution and native sampling are
    complete above; neither established a recoverable production cost. The
    separately reopened prefill-placement assessment remains active. Prior
    F16C specialization, pointer increments and row pairing are nulls.
  - Thread and matrix-shape diagnostics are complete (see CPU comparison
    thread scaling above). F32 matrix ranges overlap mx, while Q8 matrix
    latency still trails it. The resulting head-major KV layout is now validated.
    Further changes should follow profiling of the current runtime, not repeat
    completed instruction/dispatch studies. The larger-model diagnostic did
    not close the external gap.
- **Gotchas:**
  - Synthetic `bench` throughput does not establish real-model speed. Small
    projections may stay serial depending on thread count. Grouping improves
    the six-thread synthetic case, but the real-model external floor still
    fails; use the matched model measurements.
  - Do not tune the row block as a byte budget. Measured worse at every size
    (64/128/196/256 KB gave 22.37/22.04/23.68/21.12 against 24.04 for a flat
    4); the knee follows the fused kernel width, so it is `DOT_ROWS`.
  - ubatch barely matters once the kernel is right, and 343 vs 512 on a
    343-token prompt is the SAME computation - do not read noise as signal.

### Device execution model (ROADMAP #4a)

- **Process note:** this block was opened after the code was written, which
  `AGENTS.md` forbids.
- **Goal:** land the backend-agnostic execution model #4b depends on, in the
  six steps of `docs/DEVICE-EXECUTION.md`. Bar per step is no measured
  regression, not a win.
- **Done:** the design, `b1e4904`. Step 1, weights resolved once at load,
  `edd617f`: suite green, HF logits and PPL unchanged on Q8_0 and Q4_0. Doc
  page refreshed in `1d4ffa4`.
- **Done:** step 2, batched elementwise ops, committed on the feature branch.
  Suite green and HF perplexity bit-identical to step 1.
- **Done:** step 2 measured under protocol after 57294 went terminal, with the
  plan, advance rule and contamination criteria frozen and hashed before any
  timing. Nine measured pairs, arms alternating and reversing, every arm under
  `monitor_windows.py`, no sample dropped. **Result: does not advance.**

  | Phase | base mean | cand mean | Mean | Median | Baseline wins |
  |---|---:|---:|---:|---:|---:|
  | Prefill | 337.91 | 346.55 | +2.56% | +2.66% | 2/9 |
  | Decode | 27.48 | 27.69 | +0.79% | -0.50% | 5/9 |

  Prefill passes every criterion. Decode fails paired wins (5/9 against a
  rule of <= 4) while its mean and median both sit inside the 1% noise band,
  so that phase is better described as indistinguishable than regressed. The
  rule is not weakened and the run is not repeated to obtain a pass. Evidence:
  `benchmarks/device-exec-step2-20260920.json`.
- **Done:** all six steps, merged and gated (`docs/DEVICE-EXECUTION.md`, and
  the "Device execution model complete" block above); the status table row
  is `Done`. The Left lines below are the open items as of step 2, kept as
  history and since resolved.
- **Left (as of step 2):** decide step 2's disposition. The frozen rule is stricter than
  `AGENTS.md`'s own tradeoff principle, which says a large gain can justify a
  minor loss and warns against rejecting on an isolated per-case cutoff. That
  tension is a judgement call and must not be resolved by editing the rule
  afterwards. Options: re-measure with more pairs under a NEW prospective plan,
  or keep decode on the single-row ops so only prefill changes.
- **Left (as of step 2):** step 1 still has no admissible measurement of its
  own; it was timed off-protocol during 57294 (13:33:05-13:52:07 +0300, disclosed at the time).
  Expected neutral, unproven.
- **Left (as of step 2):** the external mx-llama.cpp floor is untouched by
  this runner and still applies to any step that advances.
- **Left (as of step 2, since done):** steps 3-6 (buffers, arena, KV on
  buffers, sync) untouched. No vendor backend is writable until step 6. Step 3 was started as interface
  plumbing only and reverted: `Buffer` with no caller is a speculative seam,
  which `AGENTS.md` forbids. It must land together with the weight conversion
  that uses it, which means touching `arch_qwen.hpp` and waiting for the other
  developer's
  stack.
- **Gotchas:**
  - Step 2's first decode reading of +3.40% was an artifact: its two winning
    pairs were the two lowest-throughput rounds, the other six gave +0.63%.
    Decode runs B=1 and cannot benefit. Prefill is the real effect, 8/8 paired
    wins. All samples retained.
  - Not a SwiGLU result. The screened-out "Exact decode SwiGLU callback fusion"
    measured a different change and its verdict stands.

### Fused Q5_K and Q6_K row dots (ROADMAP #1)

- **Goal:** give Q5_K and Q6_K the decode row dot Q4_K already has, so no
  dequantized value is materialised. Q4_K works because `d*q - m` factorises
  into `d*sum(q*x) - m*sum(x)`; Q5_K carries the same scale/min pair plus a
  high bit, and Q6_K has signed group scales and no min.
- **Done:** Q5_K fused dot, adopted. `dot_row_q5_K` mirrors `dot_row_q4_K`
  with the fifth bit taken from `qh`, whose mask shifts two places every 64
  values while `qh` itself does not advance. Dispatch now covers both types
  through one threading block rather than a copy.
- **Done:** correctness against the fp32 HF golden, not against llmx. The
  ordinary suite's fixtures are Q8_0 and Q4_0, so it does not cover this path
  at all; `Qwen3-0.6B-Q5_K_M.gguf` was scored directly against
  `tests/data/baseline_logits.json`: top-1 6/6, worst top-5 overlap 4/5, mean
  4.50, max absolute logit 24.46 against the 100 bound. The file is 29 tensors
  of Q6_K as well, so that path is exercised incidentally.
- **Done:** throughput, plan frozen before timing, 15 measured pairs.

  | Phase | base mean | cand mean | Mean | Median | Baseline wins |
  |---|---:|---:|---:|---:|---:|
  | Decode | 8.424 | 12.751 | +51.369% | +51.12% | 0/15 |
  | Prefill | - | - | -0.93% | -0.64% | 10/15 |

  Evidence: `benchmarks/fused-q5k-decode-20260920.json`.
  **Corrected 2026-09-20 by an independent recomputation of all 15 pairs**: the arm
  means were 8.424 and 12.751333 tok/s, not 8.96 and 13.33. The paired
  +51.369% is unchanged, and Q6_K recomputes to +68.0186%. The original
  figures were read from a single round rather than the arm means.
- **Done:** the matched external floor, which the above does NOT establish.
  `tools/compare_cpu.py`, mx `5542318e74`, six threads, eight matched rounds,
  identical committed HF token IDs in both arms:

  | Phase | llmx | mx | Ratio | Gate |
  |---|---:|---:|---:|---|
  | Prefill | 324.33 | 230.33 | 1.41x | above |
  | Decode | 15.98 | 58.30 | 0.27x | **BELOW** |

  Q5_K decode does not meet ROADMAP #8. The fused dot is a real 51% gain over
  the previous llmx build and still leaves mx 3.65x faster. Reporting the
  self-comparison alone would have read as success.
  Evidence: `benchmarks/q5k-external-floor-20260920.json`.
- **Done:** located the gap. Same machine, session, binaries and harness, two
  quants:

  | | llmx | mx | ratio |
  |---|---:|---:|---:|
  | Q8_0 decode | 40.93 | 41.44 | 0.99x |
  | Q5_K decode | 15.98 | 58.30 | 0.27x |

  Decode is bandwidth bound, so Q5_K (about 0.69 bytes/weight) should beat
  Q8_0 (about 1.06). mx does that: 41.44 -> 58.30, +41%. llmx goes backwards:
  40.93 -> 15.98, -61%. A smaller quant making llmx slower means the K-quant
  decode path is limited by per-weight unpacking and float conversion, not by
  memory traffic. Q8_0 sits at parity because its fused dequant+FMA matvec is
  cheap enough to stay bandwidth bound.
  Evidence: `benchmarks/kquant-decode-bound-20260920.json`.
- **Left:** prefill work cannot close this. Both quants are already above the
  floor in prefill, 1.50x and 1.41x in the same runs. The whole gap is K-quant
  decode.
- **Left:** test whether quantizing the activation vector to 8 bits and taking
  an integer dot product restores bandwidth-bound behaviour, as mx does. That
  is a much larger change than a fused dot, affects every quant type, and
  changes arithmetic, so it needs its own prospective plan and its own HF
  gate. It is the highest-value remaining decode work.
- **Done:** the same treatment for Q6_K, which has signed group scales and no
  min, so the dot is `sum(d_g * sum(q*x))` with no `sum(x)` term:
  `dot_row_q6_K`, adopted at +68.02% paired decode over 15 pairs, 0/15
  baseline wins. Evidence: `benchmarks/fused-q6k-decode-20260920.json`.
- **Done 2026-09-21:** `Qwen3-0.6B-Q5_K_M.gguf` is in `BASELINE_MODELS`
  (see the Vulkan block above), so the external gate covers Q5_K/Q6_K
  permanently on both backends. It is a third download for
  `tools/fetch_test_models.py`, 419 MB from the same repo and revision as
  the Q4_0 file.
- **Gotchas:**
  - Prefill does not benefit and cannot: the fused dot only fires at
    `nbatch == 1`, and the batched path already dequantizes each row once and
    reuses it across the batch. That made prefill an accidental control in the
    Q5_K run - identical code measured -0.93% mean and -0.64% median, which
    puts this setup's noise floor near 1%. Useful when reading small effects.
  - Qwen3-0.6B-Q5_K_M is 168 Q5_K / 29 Q6_K / 113 F32, so it exercises both
    paths in one model. A fused Q5_K dot alone will show a partial effect.
  - The roadmap's 2.16 -> ~2.6 tok/s is Q4_K's result on its own model, not a
    target for these types.

When you start a feature, open a block above before writing code - see
`AGENTS.md` -> "Starting a feature".
