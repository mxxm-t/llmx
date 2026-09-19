# `src/cli/main.cpp` — CLI dispatcher

Thin command-line entry point. Only argument parsing and glue live here; format
logic is in `format/`, quantization in `quant/`, inference in `inference/`, and
the model in `model/`.

Commands and their entry points:

- `quantize` / `dequantize`: `cmd_quantize` / `cmd_dequantize` (Q8_0/F32
  conversion via `model.json`/`model.bin`).
- `info`: `cmd_info` (dump metadata + tensor list).
- `tokenize` / `detokenize`: `cmd_tokenize` / `cmd_detokenize`.
- `perplexity`: `cmd_perplexity` loads and tokenizes inline or `-f/--file`
  UTF-8 text, then delegates scoring to `infer::perplexity`. `-c/--ctx-size`
  chooses window size; `--chunks` limits windows. See `inference-perplexity.md`.
- `logits`: `cmd_logits` (top-N next-token logits; this is what the correctness
  gate compares against a full-precision reference, since sampled text hides
  everything except argmax flips).
- `generate`: `cmd_generate` (prefill + generate; prints `pp:`/`tg:` timings).
- `chat`: `cmd_chat` (interactive loop using the chat template).
- `bench`: `cmd_bench` (hot-path micro-benchmark + synthetic end-to-end TPS).

Also holds the `build_synthetic_model` helper (an in-memory random Qwen3 model
for the end-to-end TPS measurement) and `print_usage`. See `docs/USAGE.md` for
the full command reference.
