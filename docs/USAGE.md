# llmx — Usage

Command-line reference for the `llmx` binary. All commands take the form
`llmx <command> [args...] [flags...]`. Run `llmx` with no arguments to print a
short usage summary.

## Global conventions

- A model file is a GGUF v3 container (see `docs/src/format/gguf.md`).
- Text arguments containing spaces must be quoted so they arrive as one argv
  element (`"The capital of France is"`).
- Token ids in `detokenize` are comma- or space-separated integers.
- `--threads 0` means auto (default: the CPU's hardware concurrency).

## `llmx quantize <model.json> <model.bin> <out.gguf> [q8_0|q4_0]`

Convert a raw float32 model into a quantized GGUF file.

- `model.json` describes the tensor names and shapes; `model.bin` holds each
  tensor's float32 data concatenated in the same order (row-major, with the
  fastest-varying dimension first).
- The optional last argument selects the output quant type: `q8_0` (default) or
  `q4_0`.
- Every tensor must have a number of elements divisible by 32 (the block size
  for both Q8_0 and Q4_0).
- The output is a GGUF v3 file with all tensors quantized to the chosen type.

> Note: Q4_0 inference is currently correct-but-slow (a generic dequant-to-f32
> matmul path, not a fused kernel) — see `docs/src/quant-quant.md`.

`model.json` schema:

```json
{
  "name": "MyModel",
  "tensors": [
    { "name": "tok_embeddings.weight", "shape": [512, 256] },
    { "name": "norm.weight",           "shape": [256] }
  ]
}
```

`shape[0]` is the fastest-varying dimension (maps to GGUF `ne[0]`).

## `llmx dequantize <in.gguf> <out.json> <out.bin>`

Inverse of `quantize`: read a Q8_0/F32 GGUF and write the tensors back out as
raw float32. Produces a `model.json`-compatible `out.json` plus the concatenated
float32 data in `out.bin`. Useful for round-trip verification and for feeding
data back into `quantize`.

## `llmx info <in.gguf>`

Inspect a GGUF file without running inference. Prints:

- file path, tensor count, metadata entry count
- every metadata key/value (typed dump)
- the tensor list: type, name, shape, element count, and on-disk byte size

Use this as the authoritative check that a model file parsed correctly.

## `llmx tokenize <in.gguf> "<text>"`

Encode `text` with the model's tokenizer and print the resulting token ids as a
comma-separated list on one line.

## `llmx detokenize <in.gguf> <id1,id2,...>`

Decode a comma- or space-separated list of token ids back into text and print it.

## `llmx perplexity <in.gguf> "<text>" [flags...]`

Compute the loss-based perplexity of `text` under the model.

- Requires at least 2 tokens.
- Prints `tokens`, `mean NLL`, and `perplexity`.

Flags:

| Flag            | Meaning                                        |
|-----------------|------------------------------------------------|
| `--threads N`   | worker thread count (0 = auto)                 |
| `--ubatch N`    | prefill physical batch (default 512)           |

## Physical batch (`--ubatch`)

`--ubatch` is how many prompt tokens go through **one forward pass** of the
graph. It sets the matmul width and the size of the prefill scratch buffers,
and it only affects prompt processing; generation is one token at a time.

It is llama.cpp's `n_ubatch` (`-ub`), not `n_batch`. llmx has no logical batch:
there is one sequence and no queue, so the prompt is the batch. That
distinction starts to matter only with the multi-user server in
`docs/ROADMAP.md` #7, where tokens from different sequences are merged into one
pass.

Scratch is sized to the smaller of `--ubatch` and the actual prompt, so a short
prompt does not allocate a full-width buffer.

## `llmx generate <in.gguf> "<prompt>" [flags...]`

Prompt-process `prompt`, then autoregressively generate tokens until eos or
`--max-tokens`. Prints the generated text (the Qwen3 `<thinking>` reasoning
block is hidden unless `--think`).

Prints `pp:` (prompt-processing) and `tg:` (text-generation) timing lines:
`N tok, <ms>, <tok/s>`.

| Flag                    | Meaning                                              | Default |
|-------------------------|------------------------------------------------------|---------|
| `-n`, `--max-tokens N`  | max tokens to generate                               | 32      |
| `--temp F`              | sampling temperature (0 = argmax/greedy)             | 0.8     |
| `--topk N`              | top-k truncation (0 = off)                           | 40      |
| `--topp F`              | top-p nucleus truncation (1.0 = off)                 | 0.95    |
| `--penalty F`           | repetition penalty (>= 1)                            | 1.0     |
| `--threads N`           | worker thread count (0 = auto)                       | 0       |
| `--ubatch N`            | prefill physical batch (llama.cpp `n_ubatch` / `-ub`) | 512     |
| `--seed N`              | RNG seed (0 = non-deterministic)                     | 0       |
| `--stop "<text>"`       | stop generating once decoded output contains this    | (none)  |
| `--think`               | show the Qwen3 `<thinking>` block                    | off     |
| `--verbose`             | print prompt-token count                             | off     |

## `llmx chat <in.gguf> [--system "<text>"] [flags...]`

Interactive chat loop reading lines from stdin. Uses the model's
`tokenizer.chat_template` (Jinja2-subset renderer) to format the conversation.
Supports the same sampling flags as `generate`, plus `--system` to set the
system message (default: `You are a helpful assistant.`).

## `llmx bench [--size N] [--iters N] [--threads N] [--p N] [--n N]`

Micro-benchmark of the backend hot paths, plus end-to-end TPS:

- `matmul`: Q8_0 matvec on an `N x N` matrix (`--size`, default 1024). Reports
  ms and GFLOPS.
- `rms_norm`: RMSNorm on `N` elements.
- `rope`: rotary position embedding on `N/2` pairs.
- End-to-end: prompt-process `--p` tokens (default 64) into a fresh KV cache and
  report pp tok/s, then decode `--n` tokens (default 64) over the warm cache and
  report tg tok/s.

This is the command `tests/perf.py` uses as the perf-regression gate.

| Flag            | Meaning                                      | Default |
|-----------------|----------------------------------------------|---------|
| `--size N`      | hot-path vector/matrix size (multiple of 32) | 1024    |
| `--iters N`     | repetitions for hot-path timing              | 5       |
| `--threads N`   | worker thread count (0 = auto)               | 0       |
| `--p N`         | tokens to prompt-process for the TPS gate    | 64      |
| `--n N`         | tokens to decode for the TPS gate            | 64      |
