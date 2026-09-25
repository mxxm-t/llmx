# llmx - Usage

Command-line reference for the `llmx` binary. All commands take the form
`llmx <command> [args...] [flags...]`. Run `llmx --help` for the grouped command
overview, or `llmx <command> --help` for that command's options, defaults and
example. `-h` is equivalent. Help needs no model, device or network connection.

```powershell
.\llmx.exe --help
.\llmx.exe chat --help
.\llmx.exe serve --help
```

Explicit help exits successfully and prints to stdout. Running without arguments
prints the overview and exits with status 1. Help is recognized immediately
after the command, without other arguments. Positional text remains text in
commands such as `tokenize`, `logits` and `perplexity`; `generate` still rejects
unrecognized arguments beginning with a dash.

## `llmx --version`

Print the release and build identifier, for example `llmx 0.1.0+g0123456789ab`.
Tracked changes add `.dirty`; untracked files are excluded. CMake and the plain
Windows build refresh this identifier on each build, including after commits
without reconfiguration. A source archive or unavailable Git reports `+unknown`.
The usage banner shows the same version. Builds do not create commits/tags,
change the release number, or embed timestamps.

## Global conventions

- A model file is a GGUF v3 container (see `docs/src/format-gguf.md`).
- Every command that reads a model's tokenizer refuses a file that names a `tokenizer.ggml.model` other than `gpt2` or a `tokenizer.ggml.pre` other than `qwen2`, the byte-level BPE with Qwen2 pretokenization that llmx implements.
  A key the file omits is not checked.
- Text arguments containing spaces must be quoted so they arrive as one argv
  element (`"The capital of France is"`).
- Token ids in `detokenize` are comma- or space-separated integers.
- `--threads` omitted or zero keeps the CPU backend's automatic hardware
  thread count, including in `bench`. Specify a positive count for matched
  performance comparisons.

## `llmx pull <owner/repo>:<quant>`

Download a GGUF model from Hugging Face and print its absolute local path on
stdout. Status goes to stderr. For example:

```powershell
$model = .\llmx.exe pull Qwen/Qwen3-0.6B-GGUF:Q8_0 --parallel 4
.\llmx.exe chat "$model" --threads 6 --temp 0 -n 256
```

The command requires curl 8.4 or newer for HTTPS. There is no Python runtime or
linked TLS library. `HF_TOKEN` supplies a credential for private/gated models;
access must already be granted by the repository owner. Tokens are passed to
curl through stdin and omitted from its arguments, environment and diagnostics.
HTTPS redirects retain certificate checks and strip authorization on a change
of origin. Normal curl proxy and CA environment settings remain supported.

| Flag | Meaning |
|---|---|
| `--revision <ref>` | Branch, tag or full commit SHA; default `main`. Metadata resolves it to a SHA before downloading. |
| `--file <name>` | Exact repository-relative file when several models match the quant. Selecting any shard downloads the entire set. |
| `--cache-dir <path>` | Cache root; default `<home>/.cache/llmx`. |
| `--parallel N` | Maximum streams per file, from 1 to 16; default 4. Large files use concurrent byte ranges. |

Quant matching is case insensitive and uses the filename's quant suffix.
Ambiguous matches and incomplete shard sets fail with an error. Files smaller
than 16 MiB use one stream; larger files use up to N streams with at least
8 MiB per stream. A server must honor ranges with exact HTTP 206 responses;
an ignored or mismatched range fails, rather than assembling incorrect bytes.
Use `--parallel 1` if the server does not support ranges.

The cache layout is
`<cache>/models--<owner>--<repo>/snapshots/<commit-sha>/<repository-file>`.
Every pull refreshes revision metadata. Cache hits are verified against the
published size and SHA256 for LFS files, or Git blob SHA1 for ordinary files.
Existing verified bytes are reused. Corrupt entries are replaced only after a
complete verified replacement is available. There is no offline mode.

Up to five attempts retry transient network/HTTP failures, including HTTP 429,
with bounded backoff and numeric Retry-After delays up to 60 seconds. Longer
server waits fail with an instruction to retry later, rather than retry early. Failed attempts
restart their stream; partial transfers are not resumed across invocations.
Range parts are assembled and hashed in bounded memory before atomic per-file
publication. Temporary disk use can approach twice a file's size. Temporary
files are removed on ordinary failure; a forcibly terminated process may leave
its `.pull-*` directory for manual removal. Simultaneous pulls use separate
temporary directories. Completed shards remain reusable if a later shard fails.

All GGUF commands accept the first `-00001-of-0000N.gguf` shard and discover
the siblings beside it. Loading validates all shard metadata and tensor extents,
then maps every shard in place, so a sharded model larger than host memory loads.
A loaded model's files must not change while it runs: a file truncated or
rewritten under the mapping is not detected and can end the process. The first shard may contain metadata
only. Successful download does not establish that llmx implements the model's
architecture, tokenizer or tensor types; current runtime coverage still applies.

## `llmx quantize <model.json> <model.bin> <out.gguf> [q8_0|q4_0]`

Convert a raw float32 model into a quantized GGUF file.

- `model.json` is UTF-8 JSON; escaped Unicode tensor names are decoded to UTF-8.
  Invalid JSON syntax/Unicode is rejected. See `docs/src/core-json.md` for
  parser limits. The conversion command validates tensor dimensions separately.
- `model.json` describes the tensor names and shapes; `model.bin` holds each
  tensor's float32 data concatenated in the same order (row-major, with the
  fastest-varying dimension first).
- The optional last argument selects the output quant type: `q8_0` (default) or
  `q4_0`.
- Every tensor's fastest-varying dimension must be divisible by 32 (the block
  size for both Q8_0 and Q4_0), so each quantized row contains whole blocks.
- A tensor has one to four dimensions. Each parsed dimension must be a positive
  integer at most `2^53-1`, within the JSON parser's consecutive integer range.
  Products, total byte sizes and allocation limits are checked; `model.bin`
  must contain exactly the described float32 data. Invalid shapes or payload
  lengths fail before creating or replacing the output. An empty tensor list
  is supported with an empty binary input.
- The output is a GGUF v3 file with all tensors quantized to the chosen type.

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

Read a GGUF containing any supported tensor types and write the tensors as
raw float32. Produces tensor descriptions in `out.json` plus the concatenated
float32 data in `out.bin`. JSON output escapes path and tensor-name quotes,
backslashes and control characters, preserving UTF-8 tensor names. Reusing this
output with `quantize` requires the shape rules above: GGUF can also hold scalar,
zero-sized or F32 tensors whose rows do not contain whole quantization blocks.

## `llmx info <in.gguf>`

Inspect a GGUF file without running inference. Prints:

- file path, tensor count, metadata entry count
- every metadata key/value (typed dump)
- the tensor list: type, name, shape, element count, and on-disk byte size

The reader validates field lengths, tensor sizes, alignment and file extents
before loading payloads. Successful `info` output does not establish valid
model configuration, required tensor shapes/names or metadata string encoding.

## `llmx tokenize <in.gguf> "<text>"`

Encode `text` with the model's tokenizer and print the resulting token ids as a
comma-separated list on one line.

## `llmx detokenize <in.gguf> <id1,id2,...>`

Decode a comma- or space-separated list of token ids back into text and print it.

## `llmx logits <in.gguf> "<text>" [--file] [--then-ids F] [--last N] [--top N] [--threads N] [--ubatch N] [--device D] [--layer-shares A,B] [--n-cpu-moe N] [--cpu-moe] [--moe-stream-from N] [--cache-type-k T] [--cache-type-v T]`

Print the top-N next-token logits for `text`, one `id value` pair per line
after a `tokens:` header. `--top` defaults to 10. `--file` reads the text
from the file named in its place, for a text longer than a command line
holds. `--then-ids F` appends the whitespace-separated token ids in `F`
after the text's tokens, so a generated reply is read as the tokens it
was. `--last N` prints each of the last `N` positions instead, one line
of its position followed by its top-N `id value` pairs, from the batched
passes a prompt takes; `tools/long_context_check.py` reads a device's
reply this way.

This exists for the correctness gate. Comparing llmx against a reference
through sampled text hides everything except argmax flips, so
`tests/baseline.py` uses this to compare the ranking directly against a
full-precision reference (`docs/ROADMAP.md` #8).

F32 models support direct numerical comparison under the HF fixture bounds.
Quantized models add weight error, so comparisons need quantization-specific
bounds on values and NLL as well as rankings. Close rankings can change;
matching the top token alone does not establish numerical correctness.

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
| `--per-token` | score one token at a time, the decode path, instead of in batched passes |
| `--verbose` | show scoring phase and actual worker count on stderr |
| `--threads N`   | worker thread count (0 = auto)                 |
| `-tb`, `--threads-batch N` | threads for batched passes (omitted or 0: `--threads`); ignored with `--per-token` |
| `--ubatch N`    | tokens per batched pass (default 512)          |
| `-ctk`, `--cache-type-k T` / `-ctv`, `--cache-type-v T` | KV cache storage per side, `f16` (default) or `f32` |
| `--device D`    | backend: `cpu`, or `vulkan:N` in a build with it |
| `--layer-shares A,B` | with several devices, their proportions of the layers |
| `--n-cpu-moe N`, `--cpu-moe` | experts of the first `N` routed layers, or of all, on the CPU beside a device |
| `--moe-stream-from N` | run those experts on the device for a prompt of at least `N` new tokens (default 0, never) |

By default a window goes through the model in batched passes of up to `--ubatch` tokens, the way a prompt does, with logits taken for every position; the output head then runs once per pass over all of its rows. `--per-token` scores the same targets one token at a time instead, which is the path generation takes after the prompt. On a device the two paths use different kernels, so a score from each checks different code; they agree to within the rounding of their reductions. This all-target window policy differs from
scoring modes elsewhere that exclude a warmup half-window; compare scores only with
identical input bytes, token IDs, window boundaries and target selection.

## Threads: generation vs prefill

For `generate` and `chat`, `--threads` is the CPU worker count for **decode** and
`--threads-batch` / `-tb` is the count for **prefill**, defaulting to
`--threads`. `-tb` is the short form of `--threads-batch`; `--threads` has none.

Prefill and decode can favor different counts. Measure the chosen model and
hardware; the matched thread-scaling tables in ASSETS record the tested cases.
An omitted or zero `-tb` uses the resolved decode count. After every prefill,
the runtime restores that count, including automatic selection and follow-up
chat turns. `--verbose` reports each phase's actual count on stderr; `bench`
prints its resolved count on stdout. Perplexity uses the batch count for batched
scoring and the decode count with `--per-token`; its `--verbose` output reports
the selected phase and actual count on stderr. Changing counts recreates the CPU pool.

On supported Windows topology, six-worker prefill automatically places workers
on separate physical cores and checks restoration of their original affinity
before decode. Other configurations use the normal scheduler. See the
[placement policy and limits](src/backends-cpu-placement.md).

On a GPU backend these flags retain their CPU-worker meaning; they
do not select GPU workgroup sizes or launch dimensions. `--ubatch` controls
prompt tokens per forward pass across backends.

## Device selection (`--device`)

`--device cpu` is the default. `--device vulkan:N` runs the model on Vulkan
device `N`, counted as the loader lists them, in a build configured with
`-DLLMX_HAS_BACKEND_VULKAN=ON` (`docs/VULKAN.md`); a build without it says
so rather than falling back. `generate`, `chat`, `logits`, `perplexity`,
`serve` and `bench` take the flag. On a device `--threads` and `--threads-batch` do
nothing and `--verbose` reports 0 threads, unless experts run on the CPU
beside it (below); `--ubatch` keeps its meaning.

## Several devices (`--device A,B,...`, `--layer-shares`)

A comma-separated `--device` list splits the model by layers over the
devices in the order listed: the first runs the embedding and the first
layers, the last runs the final layers and the head, and the residual
stream crosses once at each boundary per pass. Each device's layers are
fitted to the memory it reports free, counting its layers' weights, their
cache for the whole `--ctx-size` budget (the model context by default),
the embedding and head where they sit, one pass of activations and a
reserve for kernel scratch. Devices that hold weights in their own memory
share the layers as evenly as that allows; the CPU, whose weights read the
mapped file in place, takes only the layers the others cannot hold. A
model that does not fit is refused with the layer count that has no room.
`--verbose` prints what each device was given.

```powershell
.\llmx.exe generate Qwen3-32B-Q8_0.gguf "The capital of France is" --device vulkan:0,vulkan:1 --verbose
.\llmx.exe chat Qwen3-30B-A3B-Q4_K_M.gguf --device vulkan:0,cpu
```

`--layer-shares A,B,...` overrides the fit with each device's proportion of
the layers, one whole number per listed device: `1,1` halves the layers,
`3,1` gives the first device three quarters; the fit is still checked. A
device may be listed once. Experts on the CPU (`--n-cpu-moe`, `--cpu-moe`)
are a placement of one device and are refused with a list; list the CPU
as a device to give it layers. `bench` without `--model` measures the
first device listed, and `--profile` takes one device. Every command that
takes `--device` takes a list.

One request at a time leaves each device idle while the others run their
layers, and a card left at its automatic clock level drops its clock in
those gaps: Qwen3-8B Q8_0 split over two MI50s decodes at 39 tokens per
second at the automatic level and at 66, level with one card's 67, with
both cards held high. llmx does not change a machine's power settings;
hold the clocks up with the system's own tool where a split serves one
stream, on Linux with AMD cards `rocm-smi -d 2 3 --setperflevel high` for
the cards in the split, and `--setperflevel auto` to return them.

## Experts on the CPU (`--n-cpu-moe N`, `--cpu-moe`)

A mixture-of-experts model (`qwen3moe`, such as Qwen3-30B-A3B) larger than
the device's memory can keep its experts in host memory:
`--n-cpu-moe N` runs the routed feed-forward block of the first `N` routed
layers on the CPU, `--cpu-moe` that of every routed layer. Attention, dense
feed-forward blocks, the embedding table and the output head stay on the
`--device`, and the residual stream crosses to the CPU and back once per
offloaded layer. `--threads` then sets the CPU's workers. With
`--device cpu` the flags change nothing, and on a model without routed
layers they are refused. `generate`, `chat`, `logits`, `perplexity`,
`serve` and `bench --model` take them. For example, Qwen3-30B-A3B Q4_K_M
fits a 16 GB card with twelve layers' experts on the CPU:

```powershell
.\llmx.exe generate Qwen3-30B-A3B-Q4_K_M.gguf "The capital of France is" --device vulkan:0 --n-cpu-moe 12
```

A long prompt makes those layers the bottleneck: its tokens between them
use nearly every expert, and the work grows with the prompt. The CPU meets
a prompt's rows with each expert's weights unpacked once for all of them,
which is usually fast enough. `--moe-stream-from N` (default 0, never) runs such a layer on the device
instead for a prompt of at least `N` new tokens, its experts copied there
once per pass of up to 512 tokens. The copy is a fixed cost per pass,
about 0.9 s for twelve Q8_0 layers over the MI50's link and 3 s for thirty
over the Radeon VII's, so it pays only for long prompts: on the MI50 with
twelve Q8_0 layers on the CPU, 512 tokens prefill at 411 tok/s streamed
against 311 on the CPU, while at 247 the CPU is ahead (268 against 223);
on the Radeon VII the two meet at about 512. `512` suits a machine that
mostly reads long documents. The count is the request's new tokens, a
reused conversation prefix excluded, so a short reply in a long chat stays
on the CPU; all slices of one prompt take the same path, alone or beside
other requests. A server reply that reuses a cached prefix can therefore
take the CPU for tokens a single pass over the whole conversation would
stream, and differ from it by rounding. Generated tokens never stream. The device holds one layer's experts for this
(about 640 MB for Qwen3-30B-A3B Q8_0).

## KV cache types (`--cache-type-k`, `--cache-type-v`)

Each side of the KV cache is stored as `f16`, the default, or `f32`,
chosen separately because keys feed every attention score while values
are averaged under the softmax, so values tolerate less precision first. An f16 side is
written with round-to-nearest and read back exactly as stored, so what
differs between the types is the stored precision, not the arithmetic.
The flags mean the same thing on every backend (`generate`, `chat`,
`logits`, `perplexity`, `serve` and `bench --model` all take them); a backend that
cannot store a type refuses it rather than substituting. `f16` halves the
cache, which is what a long context on a small card needs: Qwen3-8B at
a 16k context does not fit a 16 GB card with an f32 cache. It is the
default because it costs nothing the correctness gate can see (the
perplexity delta against the reference implementation is 0.001254 on
the 8B Q8_0 excerpt against f32's 0.001374, both well inside a 0.01
bound) and because a decode step reads the whole cache, so a 512-token
generation on Qwen3-0.6B runs 6 percent faster. Pass `f32` on both
sides to store the cache exactly.

## Physical batch (`--ubatch`)

`--ubatch` is how many prompt tokens go through **one forward pass** of the
graph. It sets the matmul width and the size of the prefill scratch buffers,
and it only affects prompt processing; generation is one token at a time.

It is the physical batch, not a logical batch. llmx has no logical batch flag:
in the CLI there is one sequence and no queue, so the prompt is the batch. In
`llmx serve` tokens from different sequences are merged into one pass, and
`--ubatch` bounds the tokens of that pass (`docs/SERVER.md`).

Scratch is sized to the smaller of `--ubatch` and the actual prompt, so a short
prompt does not allocate a full-width buffer.

## `llmx generate <in.gguf> "<prompt>" [flags...]`

Prompt-process `prompt`, then autoregressively generate tokens until eos or
`--max-tokens`. Streams generated text as tokens arrive, reasoning included.
Stop matching retains the matching token in output, including any suffix
within that token, as before.

Generate and chat show model-loading percentages and processing/generating
phases on stderr when it is a terminal, or when `--verbose` is set. Loading
percentages count completed tensor payload reads, excluding metadata and padding;
model preparation follows. The processing message reports the prompt token
count before prefill begins, not a token-by-token completion percentage.
Redirected stderr stays quiet by default. Text continues to stream when stdout
is redirected.

Prints `pp:` (prompt-processing) and `tg:` (text-generation) timing lines:
`N tok, <ms>, <tok/s>`.

| Flag                    | Meaning                                              | Default |
|-------------------------|------------------------------------------------------|---------|
| `-n`, `--max-tokens N`  | max tokens to generate                               | 64      |
| `--temp F`              | sampling temperature (0 = argmax/greedy)             | 0.8     |
| `--topk N`              | top-k truncation (0 = off)                           | 40      |
| `--topp F`              | top-p nucleus truncation (1.0 = off)                 | 0.95    |
| `--penalty F`           | repetition penalty (>= 1)                            | 1.0     |
| `--threads N`           | worker thread count (0 = auto)                       | 0       |
| `--ubatch N`            | prefill physical batch                       | 512     |
| `-ctk`, `--cache-type-k T` | KV cache storage for keys: `f16` or `f32`      | `f16`   |
| `-ctv`, `--cache-type-v T` | KV cache storage for values: `f16` or `f32`    | `f16`   |
| `-tb`, `--threads-batch N` | threads for prefill                               | = `--threads` |
| `--device D`            | backend: `cpu`, or `vulkan:N` in a build with it     | `cpu`   |
| `--layer-shares A,B`    | with several devices, their proportions of the layers | fitted to free memory |
| `--n-cpu-moe N`         | experts of the first `N` routed layers on the CPU    | 0       |
| `--cpu-moe`             | experts of every routed layer on the CPU             | off     |
| `--moe-stream-from N`   | new prompt tokens from which those experts run on the device | 0 (never) |
| `--seed N`              | RNG seed (0 retains the fixed default state)        | 0       |
| `--stop "<text>"`       | stop generating once decoded output contains this    | (none)  |
| `--verbose`             | print prompt-token/thread counts, KV allocated/peak/used bytes and loading/processing status | off   |

## `llmx chat <in.gguf> [--system "<text>"] [flags...]`

Interactive chat loop reading lines from stdin. Uses the model's
`tokenizer.chat_template` (Jinja2-subset renderer) to format the conversation.
Supports the same sampling flags as `generate`, plus `--system` to set the
system message (default: `You are a helpful assistant.`).

Each input line is a follow-up in the same conversation. The runtime renders
the complete conversation with its assistant-generation header and reuses KV
only when the cached token IDs are an exact prefix. If the template rewrites
earlier turns (for example, removing old reasoning), it rebuilds the cache.
The template supplies turn-ending tokens; chat does not insert an extra EOS.
An empty rendered prompt is an error. History must fit the model context;
automatic truncation and concurrent conversations are not implemented.

For a quick follow-up check, keep one chat process open:

```powershell
.\llmx.exe chat "model.gguf" --threads 6 --temp 0 -n 256
```

Enter `My name is Marko. Remember it.`, wait for the reply, then enter
`What is my name?`. Each line continues the same conversation; starting a new
process starts a new history. Press Ctrl+C to exit.

## `llmx bench [--size N] [--iters N] [--threads N] [--p N] [--n N] [--device D]`

Micro-benchmark of the backend hot paths, plus end-to-end TPS:

- `matmul`: Q8_0 matvec on an `N x N` matrix (`--size`, default 1024). Reports
  ms and GFLOPS.
- `rms_norm`: RMSNorm on `N` elements.
- `norm_rope`: per-head RMS norm followed by rotary position embedding on
  one row of `N` floats, the op the model runs.
- End-to-end: prompt-process `--p` tokens (default 64) into a fresh KV cache
  by repeated single-token `step()` calls and report pp tok/s, then decode
  `--n` tokens (default 64) over the warm cache and
  report tg tok/s.

This is the command `tests/perf.py` uses as the perf-regression gate. Its pp
metric does not measure batched `Model::prefill`; use the matched real-model
comparison below for that path.

| Flag            | Meaning                                      | Default |
|-----------------|----------------------------------------------|---------|
| `--size N`      | hot-path vector/matrix size (multiple of 32) | 1024    |
| `--iters N`     | repetitions for hot-path timing              | 5       |
| `--threads N`   | CPU worker count (0 = auto)                  | 0       |
| `--p N`         | tokens to prompt-process for the TPS gate    | 64      |
| `--n N`         | tokens to decode for the TPS gate            | 64      |

## `llmx serve <in.gguf> [--host H] [--port N] [--max-seqs N] [--max-queue N] [--ctx-size N] [--ubatch N] [--threads N] [--device D] [--layer-shares A,B] [--n-cpu-moe N] [--cpu-moe] [--moe-stream-from N] [--cache-type-k T] [--cache-type-v T]`

The multi-user server (`docs/SERVER.md`): one model, a sequence per
request, every active request advanced by one token per pass with a slice
of a new request's prompt beside them, tokens streamed as they are sampled.
HTTP/1.1 without dependencies or TLS; put a reverse proxy in front of it
when it faces a network. Defaults: `127.0.0.1:8080`, 16 sequences, a
queue of 64. `--max-seqs` is how many requests decode at once, the rest
wait in the queue, and past `--max-queue` waiting requests a new one is
refused with 503. `--ctx-size` (`-c`) is the KV pool's total token budget shared
by every request, the model context by default: with 16 sequences over a
40k-token model that is 2.5k tokens each on average, so a deployment that
serves long conversations sets it to what its memory holds, rounded up
to whole KV blocks (128 tokens on the CPU, 64 on a Vulkan device), and a request whose prompt plus
`max_tokens` exceeds the budget is refused with 413.

| Route | Body | Reply |
|---|---|---|
| `POST /v1/generate` | `{"prompt": "...", "max_tokens": 64, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "penalty": 1.0, "seed": 0, "stop": ["..."], "stream": false}` | `{"text", "ids", "finish", "prompt_tokens", "reused_tokens", "tokens"}`, `finish` one of `eos`, `stop`, `length` |
| `POST /v1/chat` | `{"messages": [{"role": "user", "content": "..."}], ...}` (the same sampling fields) | as above; the prompt is the model's chat template over the messages |
| `GET /v1/health` | | `{"status": "ok", "model", "active", "queued", "donors", "prefix_hits", "prefix_tokens", "pauses"}` |
| `GET /v1/models` | | `{"object": "list", "data": [{"id", "object": "model", "created", "owned_by", "context_length", "vocab"}]}` |
| `POST /v1/chat/completions` | `{"messages": [...], "max_tokens" or "max_completion_tokens", "temperature", "top_p", "seed", "stop", "stream", "stream_options": {"include_usage"}}`, plus `top_k`, `penalty` or `repetition_penalty` | `{"id", "object": "chat.completion", "created", "model", "choices": [{"index": 0, "message": {"role", "content"}, "finish_reason"}], "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}}` |
| `POST /v1/completions` | `{"prompt": "...", ...}` (the same fields) | as above with `"object": "text_completion"` and `choices[0].text` |

On the last two an absent `max_tokens`, or `-1`, means no cap, as the standard has it: the reply runs to the model's end of text or to what the request may hold. Such a request reserves its prompt and grows its reservation as it generates, so uncapped requests run side by side; when the KV pool runs out, cached prefixes are dropped first, then the most recently admitted uncapped request is paused and resumes from its history once there is room. A capped request reserves its whole reach up front and is never paused. The compatible routes also return a `timings` object beside `usage`, in the fields clients that display speed read: `prompt_n` and `cache_n` (prompt tokens prefilled and reused), `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second` and `queued_ms`. Each finished request logs one line on stderr. The native routes keep a default of 64.

The last two are the shape the OpenAI clients speak, so a UI, an SDK or a script written for any such server connects to `llmx serve` unchanged: it lists `/v1/models`, sends the `id` it finds there as the model and streams `/v1/chat/completions`.
Streamed, each `data:` line is a chunk whose first delta carries the role, the last carries `finish_reason` (`stop` for the end of text or a stop string, `length` for the token limit), a usage chunk follows when asked for, then `data: [DONE]`.
A message's content is a string or an array of `{"type": "text", "text"}` parts; `n` other than 1 and non-text parts are refused with 400 in the clients' error shape, `{"error": {"message", "type"}}`.
A stream whose pass fails ends with one `data:` event holding the error in that shape, without `data: [DONE]`, since its 200 head has gone out.
The native routes carry what the shape cannot: token ids, the `eos` finish and the reused-prefix count.

With `"stream": true` the reply is `text/event-stream`: one `data:` line per token holding its id and text (a character split across tokens is held until complete), then `data: {"done": true, "finish": ..., "tokens": N}` and `data: [DONE]`.
A stream whose pass fails ends with one `data:` event holding the error in the native shape, `{"error": "..."}`, in place of the `done` event and without `data: [DONE]`.
A request is admitted when the KV pool can hold its prompt plus `max_tokens`, otherwise it waits in the queue; a prompt that cannot fit the context at all is refused with 413.
A greedy request gives the ids `generate --temp 0` gives for the same prompt, alone or beside other requests, and a seeded request is reproducible whatever it is batched with.
A finished request's cache stays a while as a donor: a new prompt that repeats its tokens shares those KV blocks read-only and prefills only what follows, `reused_tokens` in the reply, whole blocks only and never the last prompt token.
Donors give their blocks up, oldest first, when a request needs them.

```
llmx serve Qwen3-0.6B-Q8_0.gguf --device vulkan:0 --port 8080
curl -N -d '{"prompt":"The capital of France is","max_tokens":16,"stream":true}' http://127.0.0.1:8080/v1/generate
```

## `llmx bench --model <in.gguf> [--p N] [--n N] [--r N] [--seqs N] [--depth N] [--threads N] [--ubatch N] [--device D] [--layer-shares A,B] [--n-cpu-moe N] [--cpu-moe] [--moe-stream-from N] [--cache-type-k T] [--cache-type-v T] [--profile]`

The matched real-model measurement: a warm-up of each test, then `--r`
repeats (default 3) of prompt-processing `--p` tokens in one batch into an
empty history (`pp N`) and of generating `--n` tokens one at a time from
an empty history (`tg N`). Only model time is inside the timer: token ids
are fixed, sampling and text output are excluded, and every repeat starts
from a cleared history. The report is the mean and standard deviation of
tokens per second, the protocol reference runtimes' bench tools use for
the same `-p N -n N -r R`, so the figures compare directly.
`generate --verbose` also prints `pp` and `tg`, but its `tg` is a single
cold run decoding after the prompt with sampling inside the timer, which
is a different measurement.

`--seqs N` measures decode the way a server runs it: `N` sequences each
prefilled with the `--p` prompt, then `--n` passes of one token from every
sequence, reported as `xN tg` in tokens per second over all of them.

`--depth N` measures a long context: before every repeat, and outside the
timer, the history is filled with `N` tokens, and `pp` and `tg` then run on
top of it, reported as `pp P @ dN` and `tg G @ dN`, the protocol reference
bench tools use for the same `-d N`. It takes one sequence.

`--profile`, on a device backend, runs one more prompt run and one more
decode run after the timed runs, each reported on its own as the device
time each kernel spent (`profile pp`, then `profile tg` or
`profile batched tg`, which starts after its sequences' prompts); the
first 4096 dispatches of each are timed. It takes one device.

```
llmx bench --model Qwen3-0.6B-Q8_0.gguf --device vulkan:0 --p 247 --n 32 --r 3
```

For matched CPU measurements against the reference runtime, use
`tools/compare_cpu.py --llmx PATH --reference PATH --reference-revision SHA
--model MODEL.gguf --output NEW_DIRECTORY [--threads N] [--rounds N]`.
The driver sets the same thread count for prefill and decode in both arms,
records it, and rejects mismatched results. The default is six threads; the
accepted range is 1-64. See [ASSETS.md](ASSETS.md#matched-external-cpu-benchmark)
for wrapper builds, pinned inputs and measurement scope.
