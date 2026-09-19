# `src/cli/main.cpp` - CLI dispatcher

Thin command-line entry point. Only argument parsing and glue live here; format
logic is in `format/`, quantization in `quant/`, inference in `inference/`, and
the model in `model/`.

Commands and their entry points:

- `--version`: release version plus the build revision, without loading a model.

- `quantize` / `dequantize`: `cmd_quantize` / `cmd_dequantize` (Q8_0/Q4_0
  writing and supported-type dequantization via `model.json`/`model.bin`).
- `info`: `cmd_info` (dump metadata + tensor list).
- `tokenize` / `detokenize`: `cmd_tokenize` / `cmd_detokenize`.
- `perplexity`: `cmd_perplexity` loads and tokenizes inline or `-f/--file`
  UTF-8 text, then delegates scoring to `infer::perplexity`. `-c/--ctx-size`
  chooses window size; `--chunks` limits windows. See `inference-perplexity.md`.
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
- `bench`: `cmd_bench` (hot-path micro-benchmark + synthetic end-to-end TPS).

`generate` and each `chat` turn apply the prefill worker count and restore the
resolved decode count, including automatic selection. Existing `--verbose`
reports the actual counts on stderr. `bench` retains the backend's automatic
count for zero/omitted threads and prints that resolved count on stdout.

Also holds the `build_synthetic_model` helper (an in-memory random Qwen3 model
for the end-to-end TPS measurement) and `print_usage`. See `docs/USAGE.md` for
the full command reference.
