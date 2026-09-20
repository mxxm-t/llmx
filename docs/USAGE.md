# llmx — Usage

Command-line reference for the `llmx` binary. All commands take the form
`llmx <command> [args...] [flags...]`. Run `llmx` with no arguments to print a
short usage summary.

## `llmx --version`

Print the release and build identifier, then exit successfully. For example:
`llmx 0.1.0+g0123456789ab`. Tracked changes add `.dirty`; source archives or
builds without Git report `unknown` instead of a revision. Untracked files do
not affect the identifier. The usage banner reports the same version.

## Global conventions

- A model file is a GGUF v3 container (see `docs/src/format-gguf.md`).
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

## `llmx logits <in.gguf> "<text>" [--top N] [--threads N] [--ubatch N]`

Print the top-N next-token logits for `text`, one `id value` pair per line
after a `tokens:` header. `--top` defaults to 10.

This exists for the correctness gate. Comparing llmx against a reference
through sampled text hides everything except argmax flips, so
`tests/baseline.py` uses this to compare the ranking directly against a
full-precision reference (`docs/ROADMAP.md` #8).

Reading the output: llmx runs a quantized GGUF, so the VALUES differ from an
fp32 reference by quantization error and are not comparable directly. The
ranking is what is stable, and even then two tokens within about 0.01 logits of
each other can legitimately swap.

## `llmx perplexity <in.gguf> "<text>" [flags...]`

Compute the loss-based perplexity of `text` under the model.

To read text from disk instead, use `llmx perplexity <in.gguf> --file <path>`
(`-f` is an alias). Put the input immediately after the model and any remaining
flags after the input. Choose one source: inline text or one file.

```
llmx perplexity model.gguf --file "corpus.txt" --ctx-size 512 --chunks 4 --threads 6
```

- Files must contain UTF-8 text without a BOM. Bytes, including CRLF/LF line
  endings, are preserved; no trimming or newline conversion is performed.
- Tokenize the entire input once, without adding BOS/EOS, then split into
  disjoint windows. Each window resets the KV cache and RoPE positions.
- Score every token after the first in each window. Include a partial last
  window if it has at least two tokens; a final singleton has no target.
- Aggregate the sum of negative log probabilities divided by the total number
  of scored targets, then exponentiate. Do not average window perplexities.
- The default window is the model's context length; all windows are evaluated
  unless `--chunks` limits them. Input shorter than one window keeps its
  previous continuous-sequence score. The whole text and token list stay in RAM.
- Requires at least 2 tokens.
- Prints input `tokens`, `used tokens` in evaluated windows (including each
  window's first token), `scored tokens`, `chunks`, `context size`, `mean NLL`
  and `perplexity`. Unused suffixes and singleton tails appear in input tokens
  but not used/scored tokens.

Flags:

| Flag            | Meaning                                        |
|-----------------|------------------------------------------------|
| `-f`, `--file <path>` | read the input text from a UTF-8 file instead of an argument |
| `-c`, `--ctx-size N` | tokens per window, from 2 through the model's context length |
| `--chunks N` | maximum windows to evaluate (positive integer; default all) |
| `--threads N`   | worker thread count (0 = auto)                 |

Perplexity evaluates one token at a time to obtain every target's logits.
`--ubatch` and `--threads-batch` / `-tb` remain accepted for compatibility but
do not affect this command. This all-target window policy differs from
llama.cpp modes that exclude a warmup half-window; compare scores only with
identical input bytes, token IDs, window boundaries and target selection.

## Threads: generation vs prefill

`--threads` is the thread count for **generation** (decode) and
`--threads-batch` / `-tb` is the count for **prefill**, defaulting to
`--threads`. These are llama.cpp's `-t` and `-tb`.

They are separate because the two phases have different bottlenecks. Prefill is
compute bound and scales nearly linearly: on a Ryzen 7 5800X, 4/8/16 threads
gave 5.07/8.55/13.24 tok/s. Decode is memory-bandwidth bound and peaks BELOW
the logical core count, because SMT adds contention rather than bandwidth: the
same machine measured 3.11/4.39/4.42/4.23/4.12 tok/s at 2/4/6/8/16 threads.

The defaults leave both at hardware concurrency. If you are tuning, raise
`-tb` to every logical core and lower `--threads` towards the physical core
count, then measure - the knee is machine-specific.

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
`--max-tokens`. Prints generated text. The legacy reasoning filter recognizes
`thinking_start` / `thinking_end` token names; it does not currently recognize
Qwen3's `<think>` / `</think>` markers. `--think` disables that filtering.

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
| `-tb`, `--threads-batch N` | threads for prefill                               | = `--threads` |
| `--seed N`              | RNG seed (0 = non-deterministic)                     | 0       |
| `--stop "<text>"`       | stop generating once decoded output contains this    | (none)  |
| `--think`               | disable legacy reasoning-token filtering             | off     |
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
