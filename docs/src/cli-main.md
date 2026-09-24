# `src/cli/main.cpp` - CLI dispatcher

Thin command-line entry point. Only argument parsing and glue live here; format
logic is in `format/`, quantization in `quant/`, inference in `inference/`, and
the model in `model/`.

Commands and their entry points:

- `--help` / `-h`: grouped command overview. Every command also accepts either
  spelling as its sole argument, for a focused page with options, defaults and
  an example. Help returns before opening a model or creating a backend;
  positional text in `tokenize`, `logits` and `perplexity` remains text.
- `--version`: release version plus the build revision, without loading a model.
- `pull`: parse repository/quant, revision, explicit file, cache and stream count;
  pass HF credentials to `hub::pull`, render status on stderr and print the
  verified model path on stdout. See [Hub acquisition](hub.md).

- `quantize` / `dequantize`: `cmd_quantize` / `cmd_dequantize` (Q8_0/Q4_0
  writing and supported-type dequantization via `model.json`/`model.bin`).
  Input uses the core JSON parser; output quotes paths and tensor names through
  its string helper, preserving UTF-8 and escaping JSON special characters.
  Quantize requires one to four positive integral dimensions in the JSON
  parser's consecutive integer range (`1..2^53-1`) and whole 32-value rows.
  Shared GGUF tensor arithmetic checks products and output bytes; the CLI
  checks total float32 input size and allocation limits before loading data.
  Binary length must match exactly, and read/seek failures throw. These checks
  happen before opening the output. Empty tensor lists remain supported.
- `info`: `cmd_info` (dump metadata + tensor list).
- `tokenize` / `detokenize`: `cmd_tokenize` / `cmd_detokenize`.
- `perplexity`: `cmd_perplexity` loads and tokenizes inline or `-f/--file`
  UTF-8 text, then delegates scoring to `infer::perplexity`. `-c/--ctx-size`
  chooses window size; `--chunks` limits windows; `--per-token` scores one token
  at a time instead of in batched passes. See `inference-perplexity.md`.
- `logits`: `cmd_logits` (top-N next-token logits; this is what the correctness
  gate compares against a full-precision reference, since sampled text hides
  everything except argmax flips).
- `generate`: `cmd_generate` (prefill + generate; prints `pp:`/`tg:` timings).
- `chat`: `cmd_chat` (interactive loop using the chat template). Tracks the
  exact IDs fed into the model separately from message text. Prefills only
  an exact-prefix extension; resets and refills changed, shortened or identical
  prompts to obtain valid next-token logits. A returned stop token may not yet
  be cached, and EOS is supplied by the next rendered transcript rather than
  appended unconditionally. These are single-sequence semantics.
- `bench`: `cmd_bench` (hot-path micro-benchmark + synthetic end-to-end TPS),
  or with `--model` the matched real-model measurement: warm-up, then `--r`
  repeats of `pp N` and `tg N`, model time only, `--profile` for device time
  per kernel and the driver's statistics of each kernel (registers,
  occupancy) where it reports them.
- `serve`: parses host, port, sequence and queue limits, the KV budget
  (`--ctx-size`), `--ubatch`, `--threads`, `--device` and the cache types,
  then runs `server::serve` (see [server](server.md)).

`generate` and each `chat` turn apply the prefill worker count and restore the
resolved decode count, including automatic selection. Existing `--verbose`
reports the actual counts on stderr. `bench` retains the backend's automatic
count for zero/omitted threads and prints that resolved count on stdout.

Generate/chat use `load_model` to render completed tensor-byte progress on
stderr when attached to a terminal or when verbose. They show processing before
prefill and generating before sampling. `emit_text` writes and flushes inference
text chunks to stdout; the caller appends a newline per reply.

Also holds the `build_synthetic_model` helper (an in-memory random Qwen3 model
for the end-to-end TPS measurement) and `print_usage`. See `docs/USAGE.md` for
the full command reference.
