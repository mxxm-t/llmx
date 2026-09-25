# `src/cli/main.cpp` - CLI dispatcher

Thin command-line entry point. Only argument parsing and glue live here; format
logic is in `format/`, quantization in `quant/`, inference in `inference/`, and
the model in `model/`.

Commands and their entry points:

- `--help` / `-h`: grouped command overview. Every command also accepts either
  spelling as its sole argument, for a focused page with options, defaults and
  an example. Help returns before opening a model or creating a backend;
  positional text in `tokenize`, `logits` and `perplexity` remains text.
  `print_usage(command, out)` writes the overview or a command's page to the stream it is given and returns false for a name that is no command.
- `--version`: release version plus the build revision, without loading a model.
- Usage errors: a command line a command cannot take throws `UsageError`, and `main` prints that command's page on stderr (the overview for `--help` followed by anything), then `error:` and the reason, and returns 2.
  This covers a missing or extra argument, an unknown flag, a flag without its value, a number out of form or range and the refusals below, for every command, `serve` and `pull` included.
  An unknown command prints `unknown command:` and returns 2, bare `llmx` prints the overview on stdout and returns 1, and any other error prints `error:` and returns 1.
- Readers: `flag_value` takes the value after a flag and refuses a flag at the end of the line.
  `int_arg(argc, argv, i, flag, lo, hi)` reads a decimal whole number through `whole_number` (digits only, read by `std::from_chars`, so no space, sign, base prefix, fraction or trailing character), and `float_arg` a decimal number of digits, a point and an exponent (no infinity, NaN or hexadecimal form); each refuses a value outside `lo` to `hi`, which defaults to the type's largest.
  Every numeric flag reads through them with its range: `--threads` and `-tb` from 0 (automatic), `--ubatch`, `-n`, `--top`, `--last`, `--max-seqs`, `--max-queue`, the context sizes, `--chunks`, `--iters`, `--p`, `--n`, `--r` and `--seqs` from 1, `--port` 0 to 65535 (0 asks the system for a free port), `pull --parallel` 1 to `hub::max_parallel_streams`, `--size` from 32, `--depth`, `--n-cpu-moe` and `--moe-stream-from` from 0, `--temp`, `--topk`, `--topp` and `--penalty` within the sampler's ranges beside `GenParams` (`infer::kTempRange` and the others), and `--seed` as a decimal unsigned 64-bit value.
- `token_ids(text, vocab_size)`: ids separated by commas or any whitespace, any other character refused, and each id compared with the vocabulary while it is still 64 bits wide, so an id past 2^32 is refused rather than narrowed into range.
  `detokenize` and `logits --then-ids` read their ids through it.
- `read_text_file(path, command)`: a file's bytes as they are, opened by its UTF-8 path, the refusal naming the command.
  `text_arg` takes the text `logits` and `perplexity` score right after the model, inline or as `--file <path>` (`-f`), and `no_second_text` refuses a `--file` among the flags after it.
- `pull`: parse repository/quant, revision, explicit file, cache and stream count;
  pass HF credentials to `hub::pull`, render status on stderr and print the
  verified model path on stdout. See [Hub acquisition](hub.md).

- `quantize` / `dequantize`: the dispatch in `main` checks the argument count, and `cmd_quantize` / `cmd_dequantize` call `quant::quantize_raw` / `quant::dequantize_to_raw` (`quant/convert.hpp`), which read and write `model.json`/`model.bin` by their UTF-8 paths.
  `cmd_quantize` maps the quant type name, `q8_0` when omitted, through `quant::quant_type_of`, and a name quantize does not write is a usage error with exit status 2, before any file is opened.
  Quantize requires one to four positive integral dimensions in the JSON
  parser's consecutive integer range (`1..2^53-1`) and whole 32-value rows.
  Shared GGUF tensor arithmetic checks products and output bytes; the CLI
  checks total float32 input size and allocation limits before loading data.
  Binary length must match exactly, and read/seek failures throw. These checks
  happen before opening the output. Empty tensor lists remain supported.
- `info`: `cmd_info` (dump metadata + tensor list).
- `tokenize` / `detokenize`: `cmd_tokenize` / `cmd_detokenize`.
- `perplexity`: `cmd_perplexity` loads and tokenizes inline or `-f/--file` UTF-8 text (`text_arg`, `read_text_file`), then delegates scoring to `infer::perplexity`.
  `-c/--ctx-size` chooses window size; `--chunks` limits windows; `--per-token` scores one token at a time instead of in batched passes.
  Batched scoring selects `--threads-batch` / `-tb`, falling back to `--threads` for an omitted or zero override; per-token scoring uses `--threads`.
  `--verbose` reports the selected phase and actual count on stderr.
  See `inference-perplexity.md`.
- `logits`: `cmd_logits` (top-N next-token logits; this is what the correctness
  gate compares against a full-precision reference, since sampled text hides
  everything except argmax flips).
  It takes its text inline or from `--file <path>` (`-f`) right after the model, as `perplexity` does, and appends the `--then-ids` file's ids.
- `generate`: `cmd_generate` (prefill + generate; prints `pp:`/`tg:` timings).
- `chat`: `cmd_chat` (interactive loop using the chat template). Tracks the
  exact IDs fed into the model separately from message text. Prefills only
  an exact-prefix extension; resets and refills changed, shortened or identical
  prompts to obtain valid next-token logits. A returned stop token may not yet
  be cached, and EOS is supplied by the next rendered transcript rather than
  appended unconditionally. These are single-sequence semantics.
- `bench`: `cmd_bench` (hot-path micro-benchmark, timed after one untimed matmul so that one-time setup such as the CPU pool's start stays out, then synthetic end-to-end TPS),
  or with `--model` the matched real-model measurement: warm-up, then `--r`
  repeats of `pp N` and `tg N`, model time only, `--seqs N` for decode
  passes carrying one token of each of N sequences, `--depth N` for tests
  run on top of an N-token history filled outside the timer, `--profile`
  for device time per kernel and the driver's statistics of each kernel
  (registers, occupancy) where it reports them.
  Its branch refuses `--profile` unless `--device` lists one Vulkan device and no layer shares, before any model is opened.
- `layer_shares`: `--layer-shares` as one whole-number proportion per device (`core::comma_list`), each 0 to 999999; `exec_flag` reads the list through it when it meets the flag, so a bad share is a usage error before the model is opened.
- `model_options`: the model options the flags ask for, starting from `infer::ModelOptions{}`. A cache side changes, through `backend::kv_type_of`, only when its flag is given, so the model layer's default is the one the commands run. `print_usage` prints that default from `ModelOptions{}` as well, as `f16 (default) or f32` while f16 is the default.
- `exec_flag`: the execution flags every model command takes, read in one place: `--device`, `--layer-shares`, `--n-cpu-moe`, `--cpu-moe`, `--moe-stream-from`, `--threads`, `--ubatch` and the cache types. A cache type is checked and written in its one spelling as it is read (`cache_type_arg`, through `backend::kv_type_of` and `backend::kv_type_name`), so an empty or unknown name is a usage error before any model file is read. `generate`, `chat` and `perplexity` add `--threads-batch`.
  Each command's branch refuses what it would otherwise ignore or overwrite: `bench` without `--model` the flags only a model run reads, and `bench --model` `--size` and `--iters`; `generate` `--system` and a second prompt; `chat` any positional argument, since its messages come from stdin; and `generate` and `chat` a second `--stop`. `-tb` with `perplexity --per-token` stays accepted and unused, as USAGE documents.
- `open_model`: how every model-building command opens its model (`Opened`: the file, its tokenizer and the model, built in place since the model keeps the file's address): refuse `--moe-stream-from` without experts on the CPU as a usage error, which the flags alone decide; read the file, with progress for `generate`, `chat` and `serve`; place the model over the backends `--device` names (`backend::device_specs`, `backend::make_backends`) through `infer::place_model`, with the flags' shares, experts on the CPU, `--moe-stream-from` and `--ubatch`, and `serve`'s `--max-seqs` or `bench`'s `--seqs` as the generated rows a pass carries beside a prompt; print a split's plan with `--verbose` (always for `bench`); release the host's copy of the weights when no weight reads it in place; and set the thread count.
- `serve`: parses host, port, sequence and queue limits, the KV budget
  (`--ctx-size`), `--ubatch`, `--threads`, `--device`, `--layer-shares`,
  the experts' placement (`--n-cpu-moe`, `--cpu-moe`, `--moe-stream-from`)
  and the cache types, then runs `server::serve` (see [server](server.md)).
  The sequence and queue limits are refused below 1 here, the one place they are checked.
  The model's name is its file name, read as UTF-8 as the loader reads the path (`u8path`, `u8string`), so on Windows it does not pass through the system code page.

`generate` and each `chat` turn apply the prefill worker count and restore the
resolved decode count, including automatic selection. Existing `--verbose`
reports the actual counts on stderr. `bench` retains the backend's automatic
count for zero/omitted threads and prints that resolved count on stdout.

Generate/chat use `load_model` to render the format's loading progress on
stderr when attached to a terminal or when verbose; it ends when the file is
read and mapped, before the weights are uploaded and the model is ready. They show processing before
prefill and generating before sampling. `emit_text` writes and flushes inference
text chunks to stdout; the caller appends a newline per reply.

The synthetic bench's model comes from `infer::synthetic_model`
(`model/arch_qwen.hpp`). Also holds `print_usage`. See `docs/USAGE.md` for
the full command reference.
