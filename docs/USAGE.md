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

A usage error prints the command's page on stderr, or the overview for `--help` or `--version` followed by anything, then `error:` and the reason, and exits with status 2.
That is a missing or extra argument, an unknown flag, a flag without its value, a number out of its form or range, or a flag or argument the command would ignore or overwrite, and it includes `serve` and `pull` without arguments.
A flag that takes a value is given once: given again, in the same spelling or its other one (`-n` and `--max-tokens`), it is refused, and so are `--cpu-moe` and `--n-cpu-moe` together; a switch given again changes nothing.
An empty value that would read as the flag not given is refused as well: `--stop`, `--then-ids`, `--layer-shares`, `bench --model`, `pull --file` and `pull --cache-dir`.
Numbers are decimal and within the flag's range, and a whole number is digits only, with no sign, space, base prefix or fraction.
An unknown command prints `unknown command:` and exits with status 2.
Any other failure, such as a file that cannot be read, prints `error:` and exits with status 1.

## `llmx --version`

Print the release, the build identifier and the numerics fingerprint, for example `llmx 0.1.0+g0123456789ab numerics 3fa9c1d2e07b4a15`.
The fingerprint changes only when a source that can change a result's bits does (`docs/BUILD.md`, The build identifier).
It takes nothing after it.
Tracked changes add `.dirty`; untracked files are excluded. CMake and the plain
Windows build refresh this identifier on each build, including after commits
without reconfiguration. A source archive or unavailable Git reports `+unknown`.
The usage banner shows the same version. Builds do not create commits/tags,
change the release number, or embed timestamps.

## Precision

Commands that load a model accept `--dtype auto|f16|bf16|f32|int8` (default `auto`) and print one `dtype:` record to stderr. It gives the request, resolved policy, architecture default and each device's native, emulated or wider fallback implementation, with a warning for emulation or fallback. The same record appears in `/v1/health` as `dtype`, with `requested`, `declared`, `effective` and `devices`; each device has `device`, `how`, `paths` and its own `effective` dtype. `paths` groups possible matrix families by activation form, including retained F32 operations; it describes the implementation, not which paths a particular request executed. A mixed run retains the requested policy at the top level and lists the wider dtype each device that falls back runs; when all devices run one dtype, that is the top-level effective dtype.

On the supported AVX2 CPU, MI50 and Radeon VII paths, `auto` selects F16. The Qwen architectures declare BF16, but these backends prefer their supported F16 policy because they do not implement BF16 natively. Explicit `f32` keeps original F32 matrix inputs. Explicit `bf16` rounds inputs to BF16 and widens for F32 arithmetic on the CPU and on Vulkan devices that preserve F32 denormals, signed zeros, infinities and NaNs; otherwise it reports F32 fallback. F16 uses qualifying block-int16 kernels or documented wider F32 products, not a conversion of the whole model to half precision. Weights remain exact, and norms, softmax, rope, recurrent state, residuals and routers retain their F32 operations. Dtype is independent of the KV cache storage flags and does not add support for F16 or BF16 weight tensors.

Explicit `int8` rounds the inputs of every quantized matrix product, the output head included, to 8-bit integers per block of 32, prompts and generated tokens alike, for speed below the default precision; `auto` never chooses it. Products without an 8-bit build, F32 weights and MXFP4, take their F16 form. It runs on a Vulkan device that prefers the integer dot product, the MI50 under Mesa; elsewhere, on the CPU and the Radeon VII, the device runs `f16` and the record warns, naming it. Its budget is in `docs/PRECISION.md`.

What `int8` gives up for its speed, as measured on an MI50 (`docs/STATUS.md`): rankings near a tie can turn, and routed models choose other experts more often.
Two cells of the real-model HF gate that `f16` passes fail under it: Qwen3-0.6B Q4_0 keeps 3 of HF's top five tokens for "The capital of France is" where the gate asks for 4, and Qwen3.5-0.8B Q8_0 keeps 4 of 5 on one chat prompt where it asks for 5; every perplexity cell of those files passes.
A routed model picks another set of experts than F32 inputs give in 9 to 17 percent of its token-layer routings, against 1 to 1.5 percent under `f16` (Qwen3-30B-A3B Q8_0 and Qwen3.6-35B-A3B Q4_K_M), and its top token then differs from `f16`'s at 6 to 8 of 512 positions.
After 16384 tokens of context Qwen3.5-9B Q4_K_M chose another token than the HF reference at 4 or 5 of 512 positions of raw text and at 1 of a chat, where `f16` chose HF's token at every one.

| `--dtype` | quantized-weight products, the output head included, at every row count | F32-weight products | routers | attention and KV cache |
|---|---|---|---|---|
| `auto` | the preferred common dtype: `f16` on the MI50, the Radeon VII and an AVX2 CPU | as that dtype | F32 | outside `--dtype` |
| `f32` | F32 inputs | F32 | F32 | outside `--dtype` |
| `f16` | F16 or wider: blocks of 32 scaled to 16-bit integers, or F32 | F32 | F32 | outside `--dtype` |
| `bf16` | inputs rounded to BF16, widened to F32 | rounded to BF16 | F32 | outside `--dtype` |
| `int8` | blocks of 32 scaled to 8-bit integers; MXFP4, which has no 8-bit build yet, as `f16` | as `f16` | F32 | outside `--dtype` |

`logits`, `perplexity` and model `bench` also write one `matrix-paths:` JSON record to stderr after computation, containing the effective `dtype` and a `devices` list of the matrix paths actually dispatched. The correctness tools use it to select their precision bound; the startup capability description is not execution evidence. Numerical stdout is unchanged.

## Global conventions

- A model file is a GGUF v3 container (see `docs/src/format-gguf.md`).
- Every command that reads a model's tokenizer refuses a file that names a `tokenizer.ggml.model` other than `gpt2` or a `tokenizer.ggml.pre` other than `qwen2` or `qwen35`, the byte-level BPE with Qwen2 and Qwen3.5 pretokenization that llmx implements.
  A key the file omits is not checked.
- Text arguments containing spaces must be quoted so they arrive as one argv
  element (`"The capital of France is"`).
- Token ids in `detokenize` and in a `logits --then-ids` file are separated by commas or any whitespace.
  Any other character, or an id outside the vocabulary, is refused.
- `--threads` omitted or zero takes the CPU backend's automatic count, including in `bench` and `serve`: the CPUs the process may use, from its hardware threads, its affinity and its CPU quota ([Threads](#threads-generation-vs-prefill)).
  Specify a positive count for matched performance comparisons.

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
then reads the weights as `--load-mode` says (below). In `mapped` mode every shard is mapped;
`auto` also maps the files when a host backend reads weights in place, so
host weights can exceed available RAM; with `direct` those weights are copied into
memory of the process's own, and a model whose CPU weights are more than the
host has available is refused. A shard whose size changed in between is
refused in every mode.
A loaded model's mapped files must not change while it runs: a file truncated
or rewritten under the mapping is not detected and can end the process. The first shard may contain metadata
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
- The output is a GGUF v3 file with all tensors quantized to the chosen type. Writes and close must succeed before it replaces the destination; failures name the path and preserve an existing destination during preparation. Output paths must be absent or regular files, not symbolic links or devices. Replacement needs space beside an existing file and write access to its parent directory.

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

Conversion uses hidden `.llmx-output-*` staging directories beside its outputs.
An interrupted process can leave one behind; remove that directory only after
confirming its conversion has stopped. These writes do not guarantee durability
across power loss.

## `llmx dequantize <in.gguf> <out.json> <out.bin>`

Read a GGUF containing F32 or types with an implemented decoder and write the tensors as
raw float32. Produces tensor descriptions in `out.json` plus the concatenated
float32 data in `out.bin`. JSON output escapes path and tensor-name quotes,
backslashes and control characters, preserving UTF-8 tensor names. Reusing this
output with `quantize` requires the shape rules above: GGUF can also hold scalar,
zero-sized or F32 tensors whose rows do not contain whole quantization blocks.
The decoder types are F32, Q8_0, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and MXFP4.
A known storage layout without a decoder, including F16 and BF16, is refused
by tensor name before mapping the payload or allocating decoded buffers;
existing output files remain unchanged.

Both raw outputs are prepared and closed successfully before either is published.
Their paths must name different files, each absent or regular. Publication replaces
JSON then binary; if the second rename fails, the error names its path and the
first complete replacement may remain. The pair is not a transaction and is not
guaranteed durable across power loss.

## `llmx info <in.gguf>`

Inspect a GGUF file without running inference. Prints:

- file path, tensor count, metadata entry count
- every metadata key/value (typed dump)
- the tensor list: type, name, shape, element count, and on-disk byte size

The reader validates field lengths, tensor sizes, alignment, file extents
and unique tensor names from the file's headers alone: `info`, `tokenize` and
`detokenize` never read or map the tensor data, and do not hold the file
while they run. These commands accept all 35 known storage layouts, including
types without inference kernels; unknown and removed type IDs are refused.
Successful `info` output does not establish decoder or backend support, valid
model configuration, required tensor shapes/names or metadata string encoding.

## `llmx tokenize <in.gguf> "<text>"`

Encode `text` with the model's tokenizer and print the resulting token ids as a
comma-separated list on one line.

## `llmx detokenize <in.gguf> <id1,id2,...>`

Decode a comma- or whitespace-separated list of token ids back into text and print it.

## `llmx logits <in.gguf> ("<text>" | --file <path>) [--chat] [--then-ids F] [--last N] [--per-token] [--top N] [--threads N] [--ubatch N] [--device D] [--layer-shares A,B] [--tensor-width N] [--n-cpu-moe N] [--cpu-moe] [--moe-stream-from N] [--cache-type-k T] [--cache-type-v T] [--load-mode M] [--dtype T]`

`--dtype` selects activation precision (Precision, above).

Print the top-N next-token logits for `text`, one `id value` pair per line after a `tokens:` header.
The list is most likely first, a tie going to the lower id.
`--top` defaults to 10.
`--file <path>` (`-f`) in place of the text reads it from a UTF-8 file, as `perplexity` does, for a text longer than a command line holds.
`--chat` reads the text as `generate --chat` reads its prompt, one user message through the model's chat template.
`--then-ids F` appends the token ids in `F`, separated by commas or whitespace, after the text's tokens, so a generated reply is read as the tokens it was.
`--last N` prints each of the last `N` positions instead, one line of its position followed by its top-N `id value` pairs, from the batched passes a prompt takes; `tools/long_context_check.py` reads a device's reply this way.
That tool takes `--dtype auto|f16|bf16|f32` to select the same request for prompt sizing, both fresh generations and both backends' scoring passes.
`--per-token` reads every token one at a time through decode steps, the path a generated token takes, instead of the batched passes a prompt takes, as `perplexity --per-token` does; on a device the two paths run different kernels, so a reply's `--then-ids` read this way gives the logits its decode steps gave.
`--top` and `--last` are at least 1.

This exists for the correctness gate. Comparing llmx against a reference
through sampled text hides everything except argmax flips, so
`tests/baseline.py` uses this to compare the ranking directly against a
full-precision reference (`docs/ROADMAP.md` #8).

F32 models support direct numerical comparison under the HF fixture bounds.
Quantized models add weight error, so comparisons need quantization-specific
bounds on values and NLL as well as rankings. Close rankings can change;
matching the top token alone does not establish numerical correctness.

## `llmx perplexity <in.gguf> "<text>" [flags...]`

`--dtype` selects activation precision (Precision, above).

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
  disjoint windows. Each window resets the model's history (its KV cache, and a
  qwen35 model's recurrent state) and RoPE positions.
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
- With `--verbose`, a line for each window comes first, in order: `chunk K: scored N, mean NLL X`, its scored targets and their mean NLL to nine decimals, so two builds can be compared window by window.

Flags:

| Flag            | Meaning                                        |
|-----------------|------------------------------------------------|
| `-f`, `--file <path>` | read the input text from a UTF-8 file instead of an argument |
| `-c`, `--ctx-size N` | tokens per window, from 2 through the model's context length |
| `--chunks N` | maximum windows to evaluate (positive integer; default all) |
| `--per-token` | score one token at a time, the decode path, instead of in batched passes |
| `--verbose` | show scoring phase and actual worker count on stderr, and each window's scored tokens and mean NLL on stdout |
| `--threads N`   | worker thread count (0 = auto)                 |
| `-tb`, `--threads-batch N` | threads for batched passes (omitted or 0: `--threads`); refused with `--per-token`, which scores on `--threads` |
| `--ubatch N`    | tokens per batched pass (default 512)          |
| `-ctk`, `--cache-type-k T` / `-ctv`, `--cache-type-v T` | KV cache storage per side, `f16` (default) or `f32` |
| `--device D`    | backend: `cpu`, or `vulkan:N` in a build with it |
| `--layer-shares A,B` | with several devices, their proportions of the layers |
| `--tensor-width N` | devices each layer is split across, the listed devices forming groups of `N` (default 1, below) |
| `--n-cpu-moe N`, `--cpu-moe` | experts of the first `N` routed layers, or of all, on the CPU beside a device |
| `--moe-stream-from N` | run those experts on the device for a prompt of at least `N` tokens, two at the least (default 0, never) |
| `--load-mode M` | how the weights are read: `auto` (default), `mapped` or `direct` (below) |

By default a window goes through the model in batched passes of up to `--ubatch` tokens, the way a prompt does, with logits taken for every position; the output head then runs once per pass over all of its rows. `--per-token` scores the same targets one token at a time instead, which is the path generation takes after the prompt. On a device the two paths use different kernels, so a score from each checks different code; they agree to within the rounding of their reductions. This all-target window policy differs from
scoring modes elsewhere that exclude a warmup half-window; compare scores only with
identical input bytes, token IDs, window boundaries and target selection.

## Threads: generation vs prefill

With `--threads` omitted or 0 the count is automatic, the fewest of:

- the hardware threads;
- the CPUs the process's affinity allows: `sched_getaffinity` on Linux; on Windows the process affinity mask when the process's threads lie in one processor group, and every processor of their groups when they span several;
- the CPUs its CPU quota allows, rounded up: on Linux its cgroup CPU quota, `cpu.max` on cgroup v2 and `cpu.cfs_quota_us` over `cpu.cfs_period_us` on v1, in the process's own cgroup and each above it that its cgroup mount shows; on Windows the CPU rate hard cap of its job object, which a process-isolated container's `--cpus` sets.

It is at least 1 and at most 64, and a quota or affinity that cannot be read changes nothing.
So a container limited to 6 CPUs on a 16-thread host starts 6 workers rather than 16 workers sharing 6 CPUs of quota, and one limited to 1.5 CPUs starts 2.

For `generate` and `chat`, `--threads` is the CPU worker count for **decode** and
`--threads-batch` / `-tb` is the count for **prefill**, defaulting to
`--threads`. `-tb` is the short form of `--threads-batch`; `--threads` has none.

Prefill and decode can favor different counts.
Measure the chosen model and hardware; the matched thread-scaling tables in ASSETS record the tested cases.
An omitted or zero `-tb` uses the resolved decode count.
After every prefill, the runtime restores that count, including automatic selection and follow-up chat turns.
`--verbose` reports each phase's actual count on stderr.
`bench` without `--model` prints its resolved count on stdout.
Perplexity uses the batch count for batched scoring and the decode count with `--per-token`, which refuses `-tb`; its `--verbose` output reports the selected phase and actual count on stderr.
Changing counts recreates the CPU pool.

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
`serve` and `bench` take the flag. A model whose layers need an op the
device's backend lacks is refused as it loads, naming the op; every op the
`qwen3` and `qwen35` files need runs on a Vulkan device. On a device `--threads` and `--threads-batch` do
nothing and `--verbose` reports 0 threads, unless experts run on the CPU
beside it (below); `--ubatch` keeps its meaning.

## Several devices (`--device A,B,...`, `--layer-shares`)

A comma-separated `--device` list runs the model as a layer split over the
devices in the order listed: the first runs the embedding and the first
layers, the last runs the final layers and the head, and the residual
stream crosses once at each boundary per pass. Each device's layers are
fitted to the memory it reports free, counting its layers' weights, their
cache, KV for the whole `--ctx-size` budget (the model context by default, or what `bench --seqs` holds when that is more) or a linear-attention layer's recurrent state for each sequence decoding at once,
the embedding and head where they sit, one pass of activations and a
reserve for kernel scratch. Devices that hold weights in their own memory
share the layers as evenly as that allows; the CPU, which reads its weights
in place, from the mapped file or with `--load-mode direct` from its own
copy, takes only the layers the others cannot hold.
A model that does not fit is refused: the fit names the model's layer count, and with `--layer-shares` the error names the device, the layers it was given, the memory they need and the memory it has free.
`--verbose` prints what each device was given.
The CPU reports what the process can still take of the host's memory: the host's available memory, or less where a container's cgroup or a job object's memory limit leaves less; the loader's "available memory" below is the same figure.

```powershell
.\llmx.exe generate Qwen3-32B-Q8_0.gguf "The capital of France is" --device vulkan:0,vulkan:1 --verbose
.\llmx.exe chat Qwen3-30B-A3B-Q4_K_M.gguf --device vulkan:0,cpu
```

`--layer-shares A,B,...` overrides the fit with each device's proportion of
the layers, one whole number per listed device: `1,1` halves the layers,
`3,1` gives the first device three quarters; the fit is still checked. A
device may be listed once. Expert offload (`--n-cpu-moe`, `--cpu-moe`)
is a placement of one device and is refused with a list; list the CPU
as a device to give it layers. `bench` without `--model` measures the
first device listed.
`--profile` takes one Vulkan device and is refused with a list or layer shares.
Every command that takes `--device` takes a list.

### Tensor split (`--tensor-width N`)

`--tensor-width N` splits every layer across `N` devices (`docs/TENSOR-SPLIT.md`): the listed devices form groups of `N` consecutive devices, each device of a group holding its share of every layer's heads and hidden rows and of the head's vocabulary, and the groups are the stages of a layer split, so `--layer-shares` gives one share a group.
1, the default, is the layer split alone; a width that is the whole list is one group.
A group sums its devices' partial products twice a layer in a fixed order, so its output is the same run to run and however a request is batched, and across stage counts at one width, but not the same bits as one device or another width.
A group runs the dense `qwen3` models and the hybrid `qwen35` ones (Qwen 3.5, 3.6 and 3.8): of a linear-attention layer each device holds its share of the K heads and the V heads that read them, and keeps the recurrent state of those heads, so state checkpoints, drafting by lookup and the host and disk tiers work over a group as on one device.
`--drafter embedded` works over a group too: each device runs its share of the MTP block, a draft row costing two more sums, and reads the output head whole for a draft row, which it holds beside its own share of the head, so every device of the group finds the same drafted token with no exchange; the fit counts that copy.
Refused before a model file is read: a list that is not whole groups, a width above 4, a share count other than the groups and a group of devices of different kinds; refused once the model is read: a width that does not divide its heads, KV heads (or is not a multiple of them), K or V heads or vocabulary rows, a column split off whole quant blocks, and routed experts, which a group does not split yet. A group whose devices share no memory or semaphores on the platform, as on Windows, is refused there too, once the file's headers are read.
A device is listed once, so the command line forms groups of Vulkan devices, whose sum crosses the cards through dma-buf and sync files on Linux; a backend without a sum across its devices on this build or platform, the Vulkan backend on Windows, is refused naming it.
A group may span PCI root complexes, which cost its sums the same on the machines measured; where a command prints its placement (`serve`, `bench --model`, and any command with `--verbose`) each group's devices are printed with their roots, and where the listed devices allow every group under one root a line says which order gives that; a group whose devices cannot share memory or semaphores is refused, naming each device and its root.
Measured figures for groups of MI50s, against the layer split and by user count and precision, are in docs/STATUS.md (Tensor groups serving, and Llmx against the reference in the same topology, which names each shape and the precision), which also lists the cells still open against the speed gate.
Groups of CPU backends, which hold the split's arithmetic to its rules on one host, are formed by the test tools rather than the command line (`llmx-split-check` with a tensor width, docs/TENSOR-SPLIT.md, step 2).

One request at a time leaves each device idle while the others run their
layers, and a card left at its automatic clock level drops its clock in
those gaps: Qwen3-8B Q8_0 split over two MI50s decodes at 39 tokens per
second at the automatic level and at 66, level with one card's 67, with
both cards held high. llmx does not change a machine's power settings;
hold the clocks up with the system's own tool where a split serves one
stream, on Linux with AMD cards `rocm-smi -d 2 3 --setperflevel high` for
the cards in the split, and `--setperflevel auto` to return them.

## Expert offload (`--n-cpu-moe N`, `--cpu-moe`)

With expert offload, a mixture-of-experts model (`qwen3moe`, such as Qwen3-30B-A3B) larger than
the device's memory keeps its experts in host memory:
`--n-cpu-moe N` runs the routed feed-forward block of the first `N` routed
layers on the CPU, `--cpu-moe` that of every routed layer. Attention, dense
feed-forward blocks, the embedding table and the output head stay on the
`--device`, and the residual stream crosses to the CPU and back once per
offloaded layer. `--threads` then sets the CPU's workers. With
`--device cpu` the flags change nothing, and a model without routed
layers refuses them on every device, the CPU included, naming the flag
given. `generate`, `chat`, `logits`, `perplexity`,
`serve` and `bench --model` take them. For example, Qwen3-30B-A3B Q4_K_M
fits a 16 GB card with twelve layers' experts on the CPU:

```powershell
.\llmx.exe generate Qwen3-30B-A3B-Q4_K_M.gguf "The capital of France is" --device vulkan:0 --n-cpu-moe 12
```

A long prompt makes those layers the bottleneck: its tokens between them
use nearly every expert, and the work grows with the prompt. The CPU meets
a prompt's rows with each expert's weights unpacked once for all of them,
which is usually fast enough. `--moe-stream-from N` (default 0, never) runs such a layer on the device
instead for a prompt of at least `N` tokens, its experts copied there
once per pass of up to `--ubatch` tokens, 512 by default. A layer whose streamed weights include a
type the device cannot execute stays on the CPU; eligible layers still stream.
A weight type unsupported by its assigned home backend is refused at load,
before weight adoption or model buffer allocation. A tensor no model role uses
is also refused if none of the selected backends supports its type.
The copy is a fixed cost per pass,
about 0.9 s for twelve Q8_0 layers over the MI50's link and 3 s for thirty
over the Radeon VII's, so it pays only for long prompts: on the MI50 with
twelve Q8_0 layers on the CPU, 512 tokens prefill at 411 tok/s streamed
against 311 on the CPU, while at 247 the CPU is ahead (268 against 223);
on the Radeon VII the two meet at about 512. `512` suits a machine that
mostly reads long documents.
The count is the whole prompt's length, a reused conversation prefix included, so every row a prompt computes takes the same path in every slice, alone or beside other requests, and whether or not a server had part of it cached.
A server forks a cached prefix only where it took the path the new prompt takes, so the reply equals `generate`'s.
A short follow-up in a long chat therefore streams too and pays the copy, and `serve` makes that copy inside the pass that carries every other request's next token, so it delays every request sharing that pass, not only the follow-up.
With `bench --depth` the depth counts toward the length as well.
Generated tokens never stream, and neither does a one-token prompt, so `1` streams the prompts `2` does.
The device holds one layer's experts for this (about 640 MB for Qwen3-30B-A3B Q8_0).
Without `--n-cpu-moe` or `--cpu-moe` there are no experts on the CPU to stream, so a nonzero `--moe-stream-from` is refused.

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

## Reading the weights (`--load-mode`)

Every command that runs a model takes `--load-mode`, which says how the weights are read from the file; it means the same on every backend.

- `auto` (the default) fits and constructs the model before streaming the weights a device copies, so a placement that does not fit fails before those reads. Weight storage is reserved at construction; execution scratch and physical KV backing grow as needed. The loader reads copied weights in large reads, in file order, on up to two reader threads while earlier uploads proceed.
  The reads go through the operating system's file cache, so a model loaded again is served from it while the host has room; when the weights a device copies are more than the host's available memory, and the file system takes direct reads, they go around the cache.
  When a host backend reads weights in place, each file is mapped as a whole; the loader warms the tensors that host reads after the uploads, when the host has room for them.
  The progress starts once the model is built and reaches 100% after the last upload.
  On the CPU alone it uses mapped weights and warms their payload after construction.
- `mapped` maps each whole file, warms its tensor payload before placement when the host has room for it, and copies each weight a device takes out of the mapping.
- `direct` reads every weight around the file cache and maps nothing: those a device copies as `auto` streams them, and those the CPU reads into memory of its own laid out as the file, in the same pass.
  It is refused, before the model is built, where a file's file system does not take direct reads, with the reason: on Linux that needs 6.1 or later and a file system that reports the alignment, on Windows a volume that reports its sector sizes and takes an unbuffered read, and macOS and other systems have no direct reads. After the model is built, before a byte is read, it is refused when the weights the CPU reads are more than the host's available memory, or when the system will not commit the memory for them.
  On the CPU, a cold load from a file system whose mapped reads are slow, ZFS among them, is several times faster with `direct`: Qwen3-8B on six cores loads in about 6.4 s against about 22 s in `auto`. Warm, `auto`'s mapping is faster (about 2.3 s against 6.1), and `direct`'s copy is the process's own memory, which the host cannot reclaim or share with another process as it can a mapped file's pages.

All three give the same model, bit for bit. With `--verbose`, and always with `bench --model`, a line after the split's plan gives the mode and where the load's time went: building the model, and for a streamed load the files read through the cache and around it, the reads, the uploads, the bytes the devices copied straight out of the reads where they can read host memory in place, and the uploads' waits for a read.

## Physical batch (`--ubatch`)

`--ubatch` is how many prompt tokens go through **one forward pass** of the
graph. It sets the matmul width and the size of the prefill scratch buffers,
and it only affects prompt processing; generation is one token at a time.

It is the physical batch, not a logical batch. llmx has no logical batch flag:
in the CLI there is one sequence and no queue, so the prompt is the batch.
`llmx serve` merges tokens from different sequences into one pass. Its
`--ubatch` budgets prompt and replay work beside the selected decode rows,
so a pass can hold up to `--ubatch` plus `--max-seqs` rows (`docs/SERVER.md`).

For a CLI prompt, scratch grows to the smaller of `--ubatch` and that prompt.
The server reserves its pass capacity at startup, including decode rows.

When serving, `--ubatch` is also how long a decoding request waits while a prompt is read: its row rides a pass, and its token comes when that pass has left the last stage.
On Qwen3-32B Q8_0 over four MI50s at `--dtype int8`, a pass of 512 prompt rows takes about 1.0 s on two stages of tensor groups of two and 1.6 s on a layer split of four, and one of 128 rows 0.27 and 0.42 s.
So a smaller value shortens the longest gap between a request's tokens and costs output, since smaller passes read a prompt slower (8 percent at 128 rows, 16 to 24 at 64).
Measured there: on tensor groups `--ubatch 128` cut the inter-token p99 at 32 and 64 users two to three times (946 to 335 ms at 32 users on 128-token prompts) for 1 to 4 percent of the output on short prompts and 8 to 11 on long ones, and raised it at 16 users (56 to 396 ms), where requests start decoding sooner and then ride more, shorter prompt passes; on a layer split it helped the closed loads the same way and, with long prompts arriving beside decoding users, made the p99 worse (94 to 546 ms) for a fifth of the output.
The default of 512 is the setting for output and for a layer split; 128 is the setting for the steadiest tokens on tensor groups at 32 users and more; 64 was worse than 128 on every count. `docs/STATUS.md` has the tables.

## `llmx generate <in.gguf> ("<prompt>" | --file <path>) [flags...]`

`--dtype` selects activation precision (Precision, above).

Prompt-process `prompt`, then autoregressively generate tokens until eos or
`--max-tokens`. Streams generated text as tokens arrive, reasoning included.
`--file <path>` (`-f`) in place of the prompt reads it from a UTF-8 file, as `logits` and `perplexity` read their text, for a prompt longer than a command line holds.
The prompt is raw text unless `--chat` is given, which sends it as one user message through the model's `tokenizer.chat_template` with the assistant's header after it and no system message, as `/v1/chat` renders a conversation of that one message; a template the renderer refuses stops the command, as it stops `chat`.
Stop matching retains the matching token in output, including any suffix
within that token, as before.

Generate and chat show model-loading percentages and processing/generating
phases on stderr when it is a terminal, or when `--verbose` is set. Loading
percentages exclude metadata and padding. In `mapped` mode they count payload
warming before placement; an oversized payload is not warmed and goes
straight from 0 to complete. In `auto` and `direct`, progress starts after
construction and counts streamed tensors and any host-weight warming.
The processing message reports the prompt token count before prefill begins,
not a token-by-token completion percentage.
Redirected stderr still carries the dtype report; progress is hidden unless
`--verbose` is set. Text continues to stream when stdout is redirected.

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
| `--tensor-width N`      | devices each layer is split across, the listed devices forming groups of `N` | 1 |
| `--n-cpu-moe N`         | experts of the first `N` routed layers on the CPU    | 0       |
| `--cpu-moe`             | experts of every routed layer on the CPU             | off     |
| `--moe-stream-from N`   | prompt length, two at the least, from which those experts run on the device | 0 (never) |
| `--load-mode M`         | how the weights are read: `auto`, `mapped` or `direct` | auto    |
| `--seed N`              | RNG seed (0 retains the fixed default state)        | 0       |
| `--stop "<text>"`       | stop generating once decoded output contains this    | (none)  |
| `--ignore-eos`          | never end at the model's end-of-text token           | off     |
| `--drafter D`           | draft tokens to verify in one pass: `off`, `lookup`, `embedded` or a drafter file beside the model | `off`  |
| `--draft-max N`         | most drafts a verify takes, 1 to 63                  | 3       |
| `-f`, `--file <path>`   | read the prompt from a UTF-8 file, right after the model | (none) |
| `--chat`                | send the prompt as one user message through the model's chat template | off (raw text) |
| `--verbose`             | print prompt-token/thread counts, KV allocated/peak/used bytes and loading/processing status, and after `tg:` the generated token ids as `ids: a,b,...`, which `logits --then-ids` reads back, and with a drafter the drafts kept by position | off   |

`--seed` is a decimal whole number up to 2^64 - 1, so a leading zero does not make it octal and a `0x` prefix is refused.
`--temp` and `--topk` are at least 0, `--topp` is 0 to 1 and `--penalty` is at least 1, the ranges the server takes for the same settings.
A sampled token is drawn from the tokens `--topk` and `--topp` keep, ranked by score with a tie going to the lower id, so a `--seed` gives the same tokens on every run and through the server with the same settings.
`--topk 0` ranks only the best tokens, 64 at first and more as the `--topp` nucleus needs them, and with `--topp 1` ranks none: the draw walks every token in id order.
`--ignore-eos` takes the end-of-text token out of every draw, greedy included, so the reply runs to `-n` unless a `--stop` match ends it first; the server's `ignore_eos` is the same rule, and the two give the same tokens for the same settings.
The model's context still bounds the reply: a `-n` up to what the prompt leaves of it runs to `-n`, and past that the command stops with the context error, as it does without the option, where the server refuses such a request before it starts.
`--drafter lookup` drafts the tokens that followed the latest earlier occurrence of the last three tokens, else two, else one, of the prompt and the reply so far, and verifies the last token and up to `--draft-max` drafts in one pass of generated tokens (docs/SPECULATIVE.md, section 3).
Each row is sampled as the token without drafts would be, so the text, the ids and every draw are the same as with `--drafter off`, greedy or seeded; drafts that match save passes, which pays where the reply repeats its prompt or itself, and a reply whose drafts kept average below half a draft a verify drafts nothing for its next 16 tokens, then tries again.
On a model that keeps a recurrent state a verify keeps the state it started from in a slot of its own, and a rejected draft runs the state's update again over the rows it keeps.
`--drafter embedded` drafts with the MTP block a qwen35 file carries after its layers (docs/SPECULATIVE.md, section 7): the model is loaded with the block on the device of its head, every pass writes the block's cache rows for the tokens it feeds, and each round's drafts are one chain of the block's rows on that device; a file without the block is refused by name.
The text, the ids and every draw are those of `--drafter off` here too.
`--drafter PATH` names a drafter file beside the model, which must pair with it before any byte of either is read (docs/SPECULATIVE.md, step 6): the same tokenizer (tokens, token types, merges, pre-tokenizer and end of text), and for the kind of drafter it is, its widths and blocks; a file that does not pair is refused, naming both files and both values.
A file of the model's MTP blocks, as `llmx-drafter-pack split` writes one from a file that embeds them, drafts as `embedded` does, the same drafts bit for bit; a draft model, a model of its own with the same tokenizer such as Qwen3-0.6B for Qwen3-8B or Qwen3.5-0.8B for Qwen3.6-27B, is loaded on the same devices once the model has taken its room and drafts greedily on its own history, one pass to catch up and then a token a step; a DFlash drafter is refused, as llmx does not run one yet.
With `--verbose`, a line after the ids gives the drafts kept and fed at each draft position, as `drafts kept: 7/10 4/8 ...`.
A verify costs more than a step, and on a device its cost steps where its rows cross a build of the Q8_0 decode kernel, which is built for 1, 2, 4, 8, 16 and 32 columns and takes the narrowest that holds a pass's rows: on an MI50 the kernel's time rises by about 30 percent from a verify of 4 rows (depth 3) to one of 5 (depth 4), which takes the 8-column build, and by about 60 percent from 8 rows to 9 (depth 8), which takes the 16-column build, so depth 4 can give fewer tokens a second than depth 3 and depth 8 than depth 7. `generate` and `chat` verify the drafts `--draft-max` allows and the default stays 3; the server prices each pass by its rows and drafts only where that gains (docs/SERVER.md, Drafts). docs/STATUS.md has the measurements.

## `llmx chat <in.gguf> [--system "<text>"] [flags...]`

`--dtype` selects activation precision (Precision, above).

Interactive chat loop reading lines from stdin.
Uses the model's `tokenizer.chat_template` to format the conversation, rendered byte for byte as the Jinja template language defines it (`docs/src/inference-chat.md` lists what the renderer takes).
Supports the same sampling and drafting flags as `generate`, plus `--system` to set the system message (default: `You are a helpful assistant.`).
Messages come only from stdin, so a positional argument after the model is refused.
A template that uses a part of the template language the renderer does not take is refused, with the reason, before the first turn; `generate` without `--chat` still runs on that file.
A template can also refuse a conversation itself, and that ends the command with the template's message.
Each reply is kept in the history as the model wrote it, except under a template that reads `reasoning_content` and does not split a reply at `</think>` itself, such as the Qwen 3.8 ones: there the reply after its `</think>` is kept as the content and the reasoning before it as `reasoning_content`, which is the only way those templates show earlier reasoning.

Each input line is a follow-up in the same conversation. The runtime renders
the complete conversation with its assistant-generation header and reuses KV
only when the cached token IDs are an exact prefix. If the template rewrites
earlier turns (for example, removing old reasoning), it rebuilds the cache.
The template supplies turn-ending tokens; chat does not insert an extra EOS.
With `--ignore-eos` every reply runs to `-n` tokens, and the next rendered turn closes it as it closes any other.
An empty rendered prompt is an error. History must fit the model context;
automatic truncation and concurrent conversations are not implemented.

For a quick follow-up check, keep one chat process open:

```powershell
.\llmx.exe chat "model.gguf" --threads 6 --temp 0 -n 256
```

Enter `My name is Marko. Remember it.`, wait for the reply, then enter
`What is my name?`. Each line continues the same conversation; starting a new
process starts a new history. Press Ctrl+C to exit.

## `llmx bench [--size N] [--iters N] [--threads N] [--p N] [--n N] [--device D] [--model PATH] [--dtype T]`

`--dtype` selects activation precision only with `--model` (Precision, above); the synthetic benchmark refuses it.

Micro-benchmark of the backend hot paths, plus end-to-end TPS:

- `matmul`: Q8_0 matvec on an `N x N` matrix (`--size`, default 1024), timed after one untimed call so that one-time setup stays out.
  Reports ms and GFLOPS.
- `rms_norm`: RMSNorm on `N` elements.
- `norm_rope`: per-head RMS norm followed by rotary position embedding on
  one row of `N` floats, the op the model runs.
- End-to-end: prompt-process `--p` tokens (default 64) into a reset KV cache
  by repeated single-token `step()` calls and report pp tok/s, then decode
  `--n` tokens (default 64) over the warm cache and
  report tg tok/s; the whole run goes once untimed first, so neither figure
  counts the model's one-time setup on its backend (its cache's first
  growth, kernels made on first use, a device leaving its idle clocks),
  which the first repetition of `bench --model` still shows.

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

## `llmx serve <in.gguf> [--host H] [--port N] [--max-seqs N] [--max-queue N] [--passes N] [--state-checkpoints N] [--host-cache-bytes N] [--disk-cache-bytes N] [--disk-cache-dir PATH] [--disk-cache-floor N] [--disk-cache-keep] [--disk-cache-max-age TIME] [--timing] [--ctx-size N] [--drafter D] [--draft-max N] [--ubatch N] [--threads N] [--device D] [--layer-shares A,B] [--tensor-width N] [--n-cpu-moe N] [--cpu-moe] [--moe-stream-from N] [--cache-type-k T] [--cache-type-v T] [--load-mode M] [--dtype T]`

`--dtype` selects activation precision (Precision, above).

The multi-user server (`docs/SERVER.md`): one model, a sequence per
request, selected ready decoding requests advanced by one token in a pass
with prompt or replay slices beside them, tokens streamed as they are sampled.
A pipelined split can keep several passes in flight.
HTTP/1.1 without dependencies or TLS; put a reverse proxy in front of it
when it faces a network. Defaults: `127.0.0.1:8080`, 16 sequences, a
queue of 64. `--max-seqs` is how many requests decode at once, the rest
wait in the queue, and past `--max-queue` queued requests a new one is
refused with 503.
Paused requests wait apart and are not counted.
`--ctx-size` (`-c`) is the most the KV pool's total token budget, shared
by every request, may take, the model context by default, rounded up to
whole KV blocks (128 tokens on the CPU, 64 on a Vulkan device).
The server fits the budget as it loads: where the devices cannot hold it
beside the weights, a pass's activations and a recurrent state for each of
the `--max-seqs` requests, it takes the most whole blocks they hold, and it
backs the whole budget at load, so no request grows the cache; a model that
leaves no room for one block is refused before the server listens.
Over several devices the server first prints what each device was given, as `bench --model` does, and the fit reads the devices' free memory only once it has risen no further for five seconds, or after thirty, so a server restarted on cards its predecessor held places its layers as on idle cards; every command placing a split by free memory, without `--layer-shares`, waits the same.
The line the server prints as it starts gives the budget it took: with 16
sequences over a 40k-token budget that is 2.5k tokens each on average, and
a request whose prompt plus `max_tokens` exceeds the budget or the model
context, whichever is smaller, is refused with 413.

The flags, in the groups of the help page (`llmx serve --help`); the paragraphs below explain each, and [OPERATING](OPERATING.md) has setups to copy and what to do when a server misbehaves.

| Group | Flag | Default | What it sets |
|---|---|---|---|
| Server | `--host H` | `127.0.0.1` | The listen address |
| Server | `--port N` | `8080` | The listen port; 0 picks a free one |
| Server | `--ctx-size N`, `-c` | model context | Tokens of conversation memory (KV cache) shared by all requests |
| Server | `--timing` | off | Times the rounds and each device's work for `/v1/health` |
| Limits | `--max-seqs N` | `16` | Requests served at once |
| Limits | `--max-queue N` | `64` | Requests waiting for a place; more get a 503 |
| Limits | `--passes N` | one per device of a layer split, else 1 | Batches the devices work on at once |
| Prefix cache | `--host-cache-bytes N` | `--max-seqs` full histories within half the free host memory | Host memory for prefixes the devices evict |
| Prefix cache | `--disk-cache-bytes N` | `0` | Disk for what the host cache drops |
| Prefix cache | `--disk-cache-dir PATH` | `<home>/.cache/llmx/kv` | Where the disk cache lives |
| Prefix cache | `--disk-cache-floor N` | larger of 16 GiB and a twentieth of the disk | Free space kept after every write |
| Prefix cache | `--disk-cache-keep` | off | Keeps the entries for the next server |
| Prefix cache | `--disk-cache-max-age TIME` | `24h` | Deletes entries unused for longer |
| Prefix cache | `--state-checkpoints N` | fitted, up to `--max-seqs` | Saved states of models with recurrent layers |
| Speculative decoding | `--drafter D` | `off` | Proposes tokens and checks them in one pass |
| Speculative decoding | `--draft-max N` | `3` | Most drafted tokens checked at once |
| Execution | `--device D`, `--dtype T`, `--layer-shares A,B`, `--tensor-width N`, `--threads N`, `--ubatch N`, `--cache-type-k T`, `--cache-type-v T`, `--n-cpu-moe N`, `--cpu-moe`, `--moe-stream-from N`, `--load-mode M` | | Where and how the model runs, as for every model command |

`--port` is 0 to 65535, 0 asking the system for a free port, which the server prints as it starts, and `--max-seqs`, `--max-queue`, `--passes` and `--ctx-size` are at least 1.
`--passes` is how many passes the server keeps in flight: on a layer split whose every device runs its layers whole, a pass per stage by default, so every device works on some pass while the host samples another; one elsewhere, where a number above 1 is refused as the server starts.
The server prints the number it keeps, and passes whose buffers the memory cannot hold are dropped at start with a line on stderr.
`--timing` times the rounds and each device's work for `/v1/health`, its dispatches between timestamps, which slows serving: throughput is read from a server without it.
A model whose layers keep a recurrent state, a `qwen35` file such as Qwen3.5 or Qwen3.6-27B, holds a state for each of the `--max-seqs` requests it runs at once, and keeps a request's state where its prompt's last whole block ends within what a follow-up turn begins with, so a chat's next turn forks that state and reads only the rest (docs/SERVER.md).
`--state-checkpoints N` is how many such states the server keeps, each the size of one request's state (149.6 MiB on Qwen3.6-27B), by default the fewer of `--max-seqs` and the most that take at most a quarter of the KV budget's room; the oldest finished conversation's goes first, and 0 keeps none, so a follow-up turn recomputes its whole prompt and a paused request resumes from its start.
The line the server prints as it starts gives the number it took and the KV tokens they took from the budget.
`--host-cache-bytes N` is how much host memory the server keeps prefixes in that the devices evict, by default what `--max-seqs` histories take at the most one request may hold (the model context or the KV pool, whichever is smaller), within half of the memory the host has free once the model and its caches are loaded, and none where every cache sits on the CPU, whose copies would only move host memory into more of it (an explicit value is still taken), 0 keeping none; a copy whose new memory would leave the host less free than the reserve the fit keeps on it is not kept: a donor the devices evict is copied there, and a request that matches it is given it back, so more conversations than the devices hold resume rather than read their history again; eviction follows the conversation policy described in [SERVER](SERVER.md).
`--disk-cache-bytes N` keeps up to N bytes on disk of what the host cache would drop next ([DISK-TIER](DISK-TIER.md)), bit for bit as host memory holds it, 0, the default, keeping none; it needs a host cache, and is refused where there is none.
The server writes an entry while it is still in host memory, once the entries not yet on disk fill three quarters of the host cache, so room the host cache needs later releases entries already on disk at once and a request never waits for a write.
`--disk-cache-keep` keeps the entries for the next server: at a clean exit the server first writes what memory holds, the conversations that came back before the rest, newest first, within a bound it prints (20 seconds, or longer where what is left needs it at the disk's measured rate), and leaves its directory for the next server of the same model and the same numerics fingerprint under the same directory, which adopts the entries, so an update that touches no kernel, model or decoder keeps them, and one that does says which part of the identity differs; without it a clean exit removes them.
Once no request has been active or waiting for five seconds, a server under `--disk-cache-keep` writes the same entries ahead, so a stop finds little left; give the server at least the bound it prints before killing it (`docker stop -t 30` or more).
What is written is what a turn added: its new blocks, 68 KiB a token on a 27B model over two cards, and its state, 150 MiB on that model, never the conversation again, so an idle moment after a turn writes 0.165 GB, or nothing where the turn filled no block, where the whole copy at 76k tokens is 5.2 GB. A crash loses at most the turn not yet written, and every message boundary's state stays on disk within the size and the age limit, so an edit of any earlier message reads only that message again.
`--disk-cache-max-age TIME` deletes entries unused for longer than TIME, a number of seconds or one followed by `s`, `m`, `h` or `d`, by default `24h`, `0` keeping them until room takes them; a server under keep adopts only entries younger than its limit.
`--disk-cache-dir PATH` is where it lives, by default `<home>/.cache/llmx/kv`, each server in a directory of its own, readable by its owner alone on Linux and macOS and with the access of `PATH` on Windows, removed when the server exits and by the next server to start if it crashed; `--disk-cache-floor N` is the free space the file system keeps after every write, by default the larger of 16 GiB and a twentieth of the file system.
SIGTERM or SIGINT, or on Windows Ctrl-C, Ctrl-Break or closing its console, stops the server cleanly: it takes no new request, ends the ones it runs as cancelled and, under `--disk-cache-keep`, writes memory to disk, then exits with status 0; a second one ends it at once.
As it starts the server prints the directory, the file system's free space and the floor, and refuses to start where the cap and the floor together exceed the free space; a write that fails, the disk full among them, stops writing until a check a minute later finds the floor and a tenth of the cap free.
A request whose history an entry on disk shares more of than anything in memory has it read back into host memory as it waits to be admitted, at most two reads at once, and is admitted once it is there; requests behind it that fit go ahead meanwhile, and a read that fails, or would take longer than computing the shared tokens at the prompt rate the server measures, leaves the request to compute them.
`--drafter lookup|embedded|PATH` and `--draft-max N` draft as `generate` does (`generate`, above), each decoding request in the scheduler's passes beside the others, and every reply is the one it gives without drafts, greedy or seeded, alone or among others (docs/SERVER.md, Drafts). A drafter file must hold MTP blocks; the server refuses a draft model by name, since it drafts on its own history.
A request drafts where its pass has decode columns to spare, the rows the device's decode kernels read each weight once for less a row for each request decoding in that pass, or alone, and where the server's measured cost of its passes finds a gain in the drafts it expects the request to keep, so drafting goes off as more requests decode; not while a request waits to be admitted or to resume; its drafts stay inside the room its reservation holds, so they never pause, stall or evict anything.
A request drafting holds a mark of its history for the verify: on a model whose layers keep a recurrent state each mark is a state slot and its saved rows beside the ones the requests run in, the KV budget is fitted as without drafts and never lowered for them, the embedded drafter and one mark fitting beside it with the automatic state checkpoints giving way, the most that still fit, which the start line counts; a `--state-checkpoints` given by number is held, and where nothing makes room the server is refused, naming the budget and the checkpoints, a smaller `--ctx-size` or fewer `--state-checkpoints` leaving the room; more marks, up to half the devices' decode columns and `--max-seqs`, take only what is then left; the requests that draft at once are at most the marks.
The server draws a pass's tokens on its scheduler thread and up to four sampling threads beside it, one fewer sampling thread than the CPUs the process may use where that is fewer, whatever `--threads` says, which counts the CPU backend's workers; it prints the number of sampling threads as it starts.

| Route | Body | Reply |
|---|---|---|
| `POST /v1/generate` | `{"prompt": "...", "max_tokens": 64, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "penalty": 1.0, "seed": 0, "stop": ["..."], "ignore_eos": false, "stream": false, "logprobs": false, "top_logprobs": 0}` | `{"text", "ids", "finish", "prompt_tokens", "reused_tokens", "tokens"}`, `finish` one of `eos`, `stop`, `length`, and with `logprobs` `"logprobs"` beside `ids` and `"top_logprobs"` |
| `POST /v1/chat` | `{"messages": [{"role": "user", "content": "...", "reasoning_content": "..."}], ...}` (the same sampling fields; `reasoning_content` is optional) | as above; the prompt is the model's chat template over the messages |
| `POST /v1/tokenize` | `{"text": "..."}`, or `{"messages": [...]}` in place of the text | `{"tokens": [ids], "count": n}` |
| `POST /v1/detokenize` | `{"tokens": [ids]}` | `{"text": "..."}` |
| `GET /v1/health` | | `{"status": "ok", "server", "precision", "requests", "reuse", "pressure", "reread", "drafting", "passes"}`, each group's numbers split into `now` and `since_start`, and `"timing"` with `--timing`; every field is in the table below |
| `GET /v1/live` | | `{"status": "ok"}`, answered without the scheduler |
| `GET /v1/models` | | `{"object": "list", "data": [{"id", "object": "model", "created", "owned_by", "context_length", "vocab"}]}` |
| `POST /v1/chat/completions` | `{"messages": [...], "max_tokens" or "max_completion_tokens", "temperature", "top_p", "seed", "stop", "stream", "stream_options": {"include_usage"}, "logprobs", "top_logprobs"}`, plus `top_k`, `penalty` or `repetition_penalty`, and `ignore_eos` | `{"id", "object": "chat.completion", "created", "model", "choices": [{"index": 0, "message": {"role", "reasoning_content", "content"}, "logprobs", "finish_reason"}], "usage": {"prompt_tokens", "completion_tokens", "total_tokens"}}`, `reasoning_content` only for a reply that reasons, `logprobs` only when asked |
| `POST /v1/completions` | `{"prompt": "...", ...}` (the same fields, with `logprobs` a count) | as above with `"object": "text_completion"` and `choices[0].text` |

On the last two an absent `max_tokens`, or `-1`, means no cap, as the standard has it: the reply runs to the model's end of text or to what the request may hold. Such a request reserves its prompt and grows its reservation as it generates, so uncapped requests run side by side; when the KV pool runs out, cached prefixes are dropped first, then uncapped requests admitted after the one that must grow are paused, the latest first, and each resumes from its history once there is room, taking its cache back whole if nothing needed it meanwhile and otherwise recomputing what it lost the way it was first computed, so its reply is the one it gives never paused. Room goes by first admission: a request that still cannot grow waits a pass with its cache as it is rather than pausing itself, paused requests resume oldest first and do not count against `--max-queue`, and new requests are admitted only once none is paused. A capped request reserves its whole reach up front and is never paused. The compatible routes also return a `timings` object beside `usage`, in the fields clients that display speed read: `prompt_n` and `cache_n` (prompt tokens prefilled and reused), `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second` and `queued_ms`. Each finished request logs one line on stderr. The native routes keep a default of 64. A `seed` of `-1`, which clients send for a random one, is taken on these two routes as no seed and sampled as a request without one is; the native routes refuse it as any other seed below 0.

The last two are the shape the OpenAI clients speak, so a UI, an SDK or a script written for any such server connects to `llmx serve` unchanged: it lists `/v1/models`, sends the `id` it finds there as the model and streams `/v1/chat/completions`.
Streamed, each `data:` line is a chunk whose first delta carries the role, the last carries `finish_reason` (`stop` for the end of text or a stop string, `length` for the token limit), a usage chunk follows when asked for, then `data: [DONE]`.
A message's content is a string or an array of `{"type": "text", "text"}` parts; `n` other than 1 and non-text parts are refused with 400 in the clients' error shape, `{"error": {"message", "type"}}`.
On both chat routes an assistant message may carry its reasoning in `reasoning_content`, a string or null.
A message with a `reasoning_content` string is taken as sent.
An assistant message without one, or with null, is kept as `chat` keeps its own replies: under a template that reads `reasoning_content` and does not split a reply at `</think>` itself, such as the Qwen 3.8 ones, the text after its last `</think>` becomes the content and the reasoning before it `reasoning_content`, and under every other template it is rendered whole, as sent.
So a conversation renders as it does in `chat`, and a model sees earlier reasoning as its template expects it whether a client sends it inline or in `reasoning_content`.
A conversation the model's template raises on, such as one without a user message for the Qwen 3.5 templates, is refused with 400 and the template's message.
`/v1/chat/completions` gives a reply's reasoning apart from its answer, as reasoning models' replies are shown: when the template leaves the reply inside an open `<think>`, as the Qwen 3.5 templates do, or the reply opens one itself, as Qwen3's do, the text up to the first `</think>` is `reasoning_content` and the rest `content`, whole and in each streamed delta; a reply cut off while it reasons is all `reasoning_content`, and one that opens no `<think>` is all `content`, as written. `/v1/chat` keeps the text whole.
While idle the server reads such a reply again for the next turn as its content alone, as clients send it back; a client that also returns `reasoning_content`, under a template that keeps it, reuses the conversation only up to that turn's start.
`chat_template_kwargs`, an object of booleans, numbers, strings or null, sets template variables on the chat routes and `/v1/tokenize`: `{"enable_thinking": false}` makes the Qwen templates close the `<think>` in the prompt, so the model answers without reasoning. A name the render sets itself (`messages`, `tools`, `documents`, `add_generation_prompt`, `bos_token`, `eos_token`) is refused with 400.
A model whose template the renderer refuses is refused as `serve` starts, before it listens.
A stream whose pass fails ends with one `data:` event holding the error in that shape, without `data: [DONE]`, since its 200 head has gone out.
The native routes carry what the shape cannot: token ids and the `eos` finish.
On every route `temperature` and `top_k` are at least 0, `top_p` is 0 to 1 and `penalty`, or `repetition_penalty` on the compatible routes, is at least 1, as the CLI's flags are, and a value outside is refused with 400.
The compatible routes also take a `top_k` of -1, which clients send for no top-k, as 0, which keeps every token.
`ignore_eos`, on every route, makes a reply end only at its limit: the model's end-of-text token is taken out of every draw, greedy included, so the reply runs to `max_tokens`, or uncapped on the compatible routes to what the request may hold, unless a stop text ends it first.
The token does not exist for the draw: the penalty cannot bring it back, and `top_k` and `top_p` count only the other tokens.
It is `false` by default, a value other than `true` or `false` is refused with 400, and a request gives the ids `generate --ignore-eos` gives with the same settings.
The compatible replies carry the reused-prefix count as `timings.cache_n`.

Every generating route gives log-probabilities when asked, at most 20 of the most likely tokens a position: `"logprobs": true` with `"top_logprobs": k` on `/v1/generate`, `/v1/chat` and `/v1/chat/completions`, and `"logprobs": k` on `/v1/completions`, as the compatible APIs take them.
Each value is the log-softmax of the model's logits for that position, before the penalty, the temperature, top-k and top-p, written as the shortest decimal that reads back as the same 32-bit float.
The native routes add `"logprobs": [v, ...]` beside `ids`, and with a `top_logprobs` above 0 `"top_logprobs": [[{"id", "logprob"}, ...], ...]`, and a streamed event its token's `logprob`, and its `top_logprobs` when asked.
`/v1/chat/completions` gives `choices[0].logprobs.content`, one `{"token", "logprob", "bytes", "top_logprobs"}` per token, beside a null `refusal`, and `/v1/completions` gives `choices[0].logprobs` as `tokens`, `token_logprobs`, `top_logprobs` (a map from text to value, the sampled token included) and `text_offset`; a streamed chunk carries its own token's.
A token that splits a character is written `bytes:\xe2\x80` and so on.
A value JSON has no number for, minus infinity for a token given no probability, is `null` on the native routes; the compatible routes, whose fields are numbers, write -9999 for it and for any value below -9999.
The values cost a pass over the vocabulary per token, so they are computed only for a request that asks, and a reply that does not ask is unchanged.

With `"stream": true` the reply is `text/event-stream`: one `data:` line per token holding its id and text (a character split across tokens is held until complete), then `data: {"done": true, "finish": ..., "tokens": N}` and `data: [DONE]`.
A stream whose pass fails ends with one `data:` event holding the error in the native shape, `{"error": "..."}`, in place of the `done` event and without `data: [DONE]`.
A request is admitted when the KV pool can hold its prompt plus `max_tokens`, otherwise it waits in the queue; one whose prompt plus `max_tokens` passes the smaller of the model context and the pool is refused with 413 before it waits.
A greedy request gives the ids `generate --temp 0` gives for the same prompt, alone or beside other requests, and a seeded request gives the ids `generate` gives with the same settings and `--seed`, whatever it is batched with.
A finished request's cache stays a while as a donor: a new prompt that repeats its tokens shares those KV blocks read-only and prefills only what follows, `reused_tokens` in the reply, whole blocks only and never the last prompt token.
Donors give their blocks up, oldest first, when a request needs them, except that the donor a request forks is kept and, if the pool is still short, consumed by it: the blocks it shares pass to the request and the rest are freed.
A follow-up turn or a resumed request, which shares every full block of its donor, consumes that donor before any other gives its blocks up.
Only rows computed as the new prompt computes them are shared: a follow-up turn shares the previous turn's prompt and computes the previous reply again as rows of its own prompt, so every reply equals `generate`'s for the same prompt.

`/v1/tokenize` gives the ids `llmx tokenize` prints for `text`, the ids a prompt of that text reads: no chat template is applied, and the text of a special token such as `<|im_start|>` reads as that token.
With `messages` in place of `text`, as `/v1/chat` takes them, the model's chat template renders them first, the assistant's header included and an assistant message's `reasoning_content` read as the chat routes read it, so the ids are the ones a chat request with those messages reads, and a conversation the template raises on is refused with 400 as there.
No start or end token is added, since no route adds one to a prompt, so `add_special`, which clients of other servers send true to count one, is not read.
Neither route waits in the queue, and a text past the context is counted rather than refused.
`/v1/detokenize` gives the text `llmx detokenize` prints for the ids, with each byte that starts no UTF-8 character replaced by U+FFFD as in a reply, so the `ids` of a whole reply give back its `text`.
A body that is not a JSON object, a `text` that is not a string, both `text` and `messages` or neither, a text the tokenizer cannot encode, as the generating routes refuse it, and a token id that is not a whole number within the vocabulary are refused with 400, and a body past 64 MiB with 413, in the native error shape.

```
llmx serve Qwen3-0.6B-Q8_0.gguf --device vulkan:0 --port 8080
curl -N -d '{"prompt":"The capital of France is","max_tokens":16,"stream":true}' http://127.0.0.1:8080/v1/generate
curl -d '{"text":"The capital of France is"}' http://127.0.0.1:8080/v1/tokenize
```

### `/v1/health` and `/v1/live`

`GET /v1/live` answers `{"status": "ok"}` from the HTTP layer without asking the scheduler, for a liveness probe that must not wait.
`GET /v1/health` reads the scheduler's counters, which takes its lock, so a reply can wait up to the time of a stage; poll it with a timeout longer than that.
Its groups put what is true now under `now` and counters that rise from zero at start under `since_start`, with the unit in the name (`_bytes`, `_tokens`, `_ms`, `_s`).
A server without a disk tier or a drafter still prints the fields, as zeros, `false` and an empty list.

```
{"status": "ok",
 "server": {"version": "0.1.0+g1234567", "numerics": "0123456789abcdef", "uptime_s": 8123, "model": "Qwen3-8B-Q8_0.gguf", "context_tokens": 40960, "devices": ["vulkan:0"]},
 "precision": {"requested": "auto", "declared": "bf16", "effective": "f16", "devices": [{"device": "vulkan:0", "how": "native", "paths": "...", "effective": "f16"}]},
 "requests": {"now": {"active": 3, "queued": 0, "paused": 0}, "limits": {"active": 16, "queued": 64},
              "since_start": {"finished": 912, "prompt_tokens": 481203, "generated_tokens": 90112}},
 "reuse": {"since_start": {"forks": 311, "tokens": 205112},
           "device": {"now": {"entries": 4, "state_checkpoints": 0}},
           "host": {"now": {"entries": 12, "bytes": 913047552, "limit_bytes": 17179869184}, "since_start": {"promotions": 40, "bytes_moved": 4093640704}},
           "disk": {"now": {"entries": 90, "bytes": 8011472896, "limit_bytes": 214748364800, "in_flight": 0, "ready": true, "writing": true},
                    "since_start": {"hits": 7, "bytes_read": 612368384, "bytes_written": 9100574720, "waits": 7, "wait_ms": 412, "errors": 0, "dropped_for_cap": 0, "lost_before_written": 0}},
           "boundaries": {"now": {"entries": 14}, "since_start": {"hits": 21}}},
 "pressure": {"since_start": {"pauses": 0, "stalls": 0, "waits": 0, "recomputed_tokens": 0, "resumes_taking_history_back": 0}},
 "reread": {"since_start": {"jobs": 55, "rows": 31040, "cancelled": 3}},
 "drafting": {"since_start": {"drafted": 4096, "kept": 2780, "by_position": [{"position": 1, "drafted": 1024, "kept": 901}]}},
 "passes": {"limit": 2, "in_flight": 1, "sampling_threads": 3}}
```

| Field | Unit | Kind | Meaning | Worth a look when |
|---|---|---|---|---|
| `server.version`, `server.numerics` | text | fixed | The build, as `--version` prints it, and the first 16 characters of the fingerprint of the sources that decide a result's bits; the disk cache adopts entries only of its own fingerprint | two servers of one deployment differ |
| `server.uptime_s` | seconds | now | Since the server started | it is small and you did not restart it |
| `server.model`, `server.context_tokens`, `server.devices` | text, tokens, names | fixed | The model file's name, its context length, and the devices it runs on | |
| `precision` | | fixed | The activation precision requested and the one each device runs, with `how` it runs (`native`, `emulated` or `fallback`) | a device shows `fallback` |
| `requests.now.active` | requests | now | Requests running in passes | it sits at `requests.limits.active` |
| `requests.now.queued` | requests | now | Requests waiting for a place | it stays above 0, or reaches `requests.limits.queued`, where new requests get a 503 |
| `requests.now.paused` | requests | now | Requests paused for want of KV room, waiting to resume; not counted in `queued` | above 0 for long |
| `requests.limits.active`, `.queued` | requests | fixed | `--max-seqs` and `--max-queue` | |
| `requests.since_start.finished` | requests | since start | Clients' requests that were admitted and ended, by any cause, a read-again job not counted; a request is added just after its reply is complete, so a poll right after a reply may not show it yet | |
| `requests.since_start.prompt_tokens`, `.generated_tokens` | tokens | since start | Prompt tokens and generated tokens of those requests | |
| `reuse.since_start.forks` | requests | since start | Requests that started from a shared history rather than from nothing | close to 0 for a client that sends conversations |
| `reuse.since_start.tokens` | tokens | since start | Prompt tokens those requests did not have to read | |
| `reuse.device.now.entries` | histories | now | Finished or paused conversations kept in device memory, ready to share | |
| `reuse.device.now.state_checkpoints` | states | now | Saved conversation states, on a model whose layers keep one | |
| `reuse.host.now.entries`, `.bytes`, `.limit_bytes` | histories, bytes | now | Histories in host memory, their bytes, and the cap `--host-cache-bytes` set (0 where there is no host cache) | `bytes` at `limit_bytes` is normal; the tier makes room by dropping entries |
| `reuse.host.since_start.promotions` | histories | since start | Histories copied from host memory back to a device for a request | |
| `reuse.host.since_start.bytes_moved` | bytes | since start | Bytes copied between devices and host memory, both ways | |
| `reuse.disk.now.entries`, `.bytes`, `.limit_bytes` | files, bytes | now | Entries on disk, their bytes, and the cap `--disk-cache-bytes` set (0 where there is no disk tier) | |
| `reuse.disk.now.in_flight` | operations | now | The write and the reads the disk tier has under way | |
| `reuse.disk.now.ready` | yes or no | now | The store is made, which waits for the model file's digest (about twenty seconds for a 27 GB file not hashed before) | false for long after start |
| `reuse.disk.now.writing` | yes or no | now | The tier has not stopped writing | false after `ready`, with a tier configured: it stopped after a failed write and waits for room |
| `reuse.disk.since_start.hits`, `.bytes_read` | entries, bytes | since start | Entries read back for a request, and their bytes | |
| `reuse.disk.since_start.bytes_written` | bytes | since start | Every finished write's bytes | rising fast with a small cap |
| `reuse.disk.since_start.waits`, `.wait_ms` | requests, milliseconds | since start | Requests that waited for a read, and the time they waited | `wait_ms` grows far faster than `waits` |
| `reuse.disk.since_start.errors` | operations | since start | Writes and reads that failed | above 0 |
| `reuse.disk.since_start.dropped_for_cap` | entries | since start | Entries deleted to stay under the disk cap | rising: the cap is small for the load |
| `reuse.disk.since_start.lost_before_written` | copies | since start | Copies host memory gave up room for before any file held them | rising: the disk is too slow or small for the rate conversations end |
| `reuse.boundaries.now.entries`, `.since_start.hits` | states, requests | now, since start | Message boundaries' states held in host memory, and requests that forked one | |
| `pressure.since_start.pauses` | pauses | since start | Times a request was paused to give blocks to another | rising: the pool is small for the load (`--ctx-size`, `--max-seqs`) |
| `pressure.since_start.stalls` | passes | since start | Passes a request sat out, unable to grow | rising with pauses |
| `pressure.since_start.waits` | passes | since start | Of those, the ones whose room waited on a request in flight | |
| `pressure.since_start.recomputed_tokens` | tokens | since start | Rows resumes computed again | the cost of the pauses |
| `pressure.since_start.resumes_taking_history_back` | resumes | since start | Resumes that took their own kept history back whole, computing nothing | |
| `reread.since_start.jobs`, `.rows`, `.cancelled` | jobs, rows, jobs | since start | Background jobs that read a reply again so the next turn finds it kept, the rows they read, and the jobs that gave way to a request at a pass boundary | |
| `drafting.since_start.drafted`, `.kept` | tokens | since start | Drafted tokens that verifies fed, and the ones they kept; `by_position` has the same by draft position, from 1 | `kept` far below `drafted` |
| `passes.limit`, `.in_flight` | passes | fixed, now | `--passes` and the passes under way | |
| `passes.sampling_threads` | threads | fixed | Threads that sample beside the scheduler's | |
| `timing` | milliseconds | since start | With `--timing` only: the means of the rounds' parts, `stage_idle_share` (a fraction of the span for each stage) and `device_bound_rows_per_s` | |

The flat field names this reply had before (`prefix_hits`, `host_donors`, `disk_ready`, `drafted` and the rest) are gone, and `docs/SERVER.md` (The grouped `/v1/health`) maps each to its place.

## Serving load (`tools/server_load.py`)

A running server under load, measured the way serving runtimes are compared: `python tools/server_load.py --url http://127.0.0.1:8080 [flags]`, standard library only.
Every request streams a greedy reply, or with `--sampled` one drawn at llmx's defaults, so the arrival of each token is timed and the work of a request is fixed.

| Flag | Meaning | Default |
|---|---|---|
| `--url U` | the server | `http://127.0.0.1:8080` |
| `--api A` | `llmx` (`/v1/generate`), `openai` (`/v1/completions`) or `completion` (the reference server's `/completion`) | `llmx` |
| `--model M` | the model id `--api openai` sends | the first id `/v1/models` lists |
| `--concurrency C ...` | closed-loop levels: C users, each sending its next request when its reply ends | `1 2 4 8 16 32 64`, none when `--rate` is given alone |
| `--rate R ...` | open-loop levels: requests arriving at R a second as a Poisson process, each sent at its time whatever is still running; `inf` sends them all at once | none |
| `--num-prompts N` | requests a level | one a user in the closed loop, 100 in the open loop |
| `--rounds N` | repeats of each closed-loop level; the table shows the round with the most output tokens per second among those in which no request failed | 2 |
| `--tokens N` | tokens a request asks for, a reply ending at the end of text | 64 |
| `--output-len N` | in place of `--tokens`: exactly N tokens a request, with `ignore_eos` sent | |
| `--input-len N` | every request its own prompt of N tokens | the eight short fixed prompts |
| `--input-len-range LO:HI` | prompt lengths uniform from LO to HI | |
| `--seed S` | seeds the prompt lengths, the prompts and the arrivals | 0 |
| `--sampled` | every request drawn at temperature 0.8, top-k 40 and top-p 0.95, llmx's defaults, with seeds 1 up to the level's request count, sent explicitly on every API, and on `--api completion` the reference server's other samplers sent switched off | greedy |
| `--warmup N` | requests before any level, at the longest prompt and the full reply | 1 |
| `--timeout S` | seconds a timed request may go with nothing arriving before it fails, the limit the tool always put on every read | 600 |
| `--total-timeout S` | seconds a timed request may take in all before it fails | 21600 |
| `--json PATH` | every request's record and every level's figures, rewritten after each level | |
| `--self-test` | check the stream reading and the figures on made-up times and against an in-process server, and exit | |

An open level sends its requests whatever the server is still running, and the server queues what it cannot run yet.
`llmx serve` runs 16 requests and queues 64 more by default and refuses the rest with 503, which count as failures, so for an open level start it with `--max-queue` at least `--num-prompts`.

A prompt of `--input-len N` is the word "the" and random words from a fixed list after it, a different prompt for every request, so requests share no prefix beyond the first word but by chance.
The prompt lengths come from the seed alone, before any prompt is built, so the n-th request of every level and round has the same length, whatever the server under test counts; the words come from the seed and the level and round.
The same seed gives the same prompts, so a second run against a server that kept the first run's prefixes can reuse them; the `reused` column shows it, and another `--seed` avoids it.
A prompt's length counts every token the server reads, a start token it adds included.
Where the server has a tokenize route (`POST /tokenize`, or `POST /v1/tokenize` as `llmx serve` has) each prompt is counted there and trimmed or extended to its length, and a line below the table lists any prompt that cannot reach it; otherwise two one-token requests on the word list, once and twice, show whether every word is one token, and when it is a prompt's length is exact by construction.
Where the replies report their prompt tokens (`usage` on `--api openai`, `tokens_evaluated` on `--api completion`) a line below the table lists any that miss their target.
`--output-len` asks every route to ignore the end of text, which llmx's routes do (`ignore_eos`), so there every reply has its length; a server that does not honour `ignore_eos` can still end a reply at the model's end of text, and the `short` column counts such replies.

One row per level: the first eight columns are the tool's earlier table (output tokens per second, requests per second, time to first token and inter-token latency at the median and the 99th percentile), then prompt and output tokens per second (`all tok/s`), time to first token at the mean and the 90th percentile, time per output token after the first (`tpot`, per request (end - first token) / (tokens - 1), the end being the reply's last event) at the mean, median and 99th percentile, end-to-end latency to the reply's last event at the median and 99th percentile, the mean prompt and output tokens, the counts (completed, failed and short) and the mean prompt tokens a request reused from the server's prefix cache.
Figures are over the completed requests, from a level's first send to its last reply's end; time to first token counts from before the connection is opened.
A reply's tokens are the count it reports where it reports one, so an event of several tokens counts them all and an event holding only the rest of a character split across tokens counts none.
A request fails on an HTTP error, a refused or dropped connection, an error event, a stream that ends before its last event, or either timeout; each failure is counted and its reason listed below the table, for every round of a closed level, the ones the table does not show included.
Reused tokens are read from the replies (`timings.cache_n`) or, on the native route, from the difference of `/v1/health`'s `reuse.since_start.tokens` around the level.
Every prompt starts with "the", so a server that matches prompts token by token reuses that word and any start token, one or two tokens a request.
`--api completion` sends `cache_prompt: false`, as the tool always has, so the reference server's `/completion` reuses nothing; `--api openai` leaves every server's prefix cache at its default.
Below the table an open level also notes sends that fell behind their arrival times at the 99th percentile by over 10 ms or a twentieth of the mean gap between arrivals, whichever is longer, and any level the requests that took over 100 ms to connect, as a burst past the server's listen backlog does; every request's send lag and connect time are in `--json`.

The tool exits 0 when every timed request completed and 3 when any failed, after printing the table and its notes and writing `--json`; it exits 1 when the server cannot be reached or the warm-up fails.

```
python tools/server_load.py --input-len 128 --output-len 128 --concurrency 1 2 4 8 16 32 64 --num-prompts 128 --json closed.json
llmx serve Qwen3-0.6B-Q8_0.gguf --max-queue 256
python tools/server_load.py --input-len-range 64:1024 --output-len 128 --rate 1 2 4 8 inf --num-prompts 200 --json open.json
```

## `llmx bench --model <in.gguf> [--p N] [--n N] [--r N] [--seqs N] [--depth N] [--threads N] [--ubatch N] [--device D] [--layer-shares A,B] [--tensor-width N] [--n-cpu-moe N] [--cpu-moe] [--moe-stream-from N] [--cache-type-k T] [--cache-type-v T] [--load-mode M] [--profile] [--drafter D]`

The matched real-model measurement: a warm-up of each test, then `--r`
repeats (default 3) of prompt-processing `--p` tokens in one batch into an
empty history (`pp N`) and of generating `--n` tokens one at a time from
an empty history (`tg N`). Only model time is inside the timer: token ids
are fixed, sampling and text output are excluded, and every repeat starts
from a cleared history. The report is the mean and standard deviation of
tokens per second, the protocol reference runtimes' bench tools use for
the same `-p N -n N -r R`, so the figures compare directly.
`generate` always prints `pp` and `tg` as well, but its `tg` is a single cold run decoding after the prompt with sampling inside the timer, which is a different measurement.

`--seqs N` measures decode the way a server runs it: `N` sequences each
prefilled with the `--p` prompt, then `--n` passes of one token from every
sequence, reported as `xN tg` in tokens per second over all of them.
The KV pool is the model context, as for `generate`, and grows to hold the `N` sequences' prompts and tokens at once, each in whole KV blocks, when they need more.

`--depth N` measures a long context: before every repeat, and outside the
timer, the history is filled with `N` tokens, and `pp` and `tg` then run on
top of it, reported as `pp P @ dN` and `tg G @ dN`, the protocol reference
bench tools use for the same `-d N`. It takes one sequence.
With `--moe-stream-from N` the depth counts toward a prompt's length, so `pp` at a depth streams once the depth plus `--p` reaches `N`.

`--drafter embedded`, or a file of MTP blocks beside the model, loads the MTP block as `generate` does and runs its cache rows in every pass, drafting nothing, so the figures show what carrying the block costs prompt and decode; `off`, the default, loads no drafter.
It then times a round's rollback, after a mark and a verify of 3 drafts, the history kept at each of the verify's 4 rows: the retract and the decode step after it as completed work up to the step's logits, each repeat taking every position in turn, and gives the median over the repeats of each position less keeping all 4 in the same repeat, whose retract runs nothing, with the middle half of those differences and the median time of the retract call itself on the host, as `bench: rollback keeping K of 4 rows`, beside the step after keeping all and the mark itself, over `--r` runs after one more that is not counted; it needs `--n` of 5 or more and one sequence, and where the depth, the prompt, the verify's 4 rows and the step after them reach past the model's context it prints `bench: rollback skipped` with that reach instead.

`--profile`, on a device backend, runs one more prompt run and one more
decode run after the timed runs, each reported on its own as the device
time each kernel spent (`profile pp`, then `profile tg` or
`profile batched tg`, which starts after its sequences' prompts); the
first 4096 dispatches of each are timed.
It takes one Vulkan device and is refused on the CPU, with a device list or with layer shares.

`--size` and `--iters` belong to the synthetic bench and are refused with `--model`, as the flags only a model run reads are refused without it.

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


For request consistency checks, `python tools/server_mix_check.py --model MODEL.gguf --text corpus.txt --device cpu --requests 4 --max-seqs 4 --fresh-phases --ctx-size 8192 --logprobs --cache-type f32` compares requests alone, concurrent and staggered, then against the CLI.
The tool's `--cache-type f16|f32` selects both cache sides for every server and CLI call; omit it to test the runtime default.
`--fresh-phases` starts each capped phase on a fresh server and refuses prefix reuse or pauses, so the pool must hold the requests at once; it cannot be combined with `--uncapped`.
These are validation-tool options; the runtime itself selects sides with `--cache-type-k` and `--cache-type-v`.

## MXFP4 files

MXFP4 GGUF matrices are read-only and execute through the ordinary model commands on the CPU or on Vulkan devices with the required float-preservation and double support. `quantize` still writes only Q8_0 and Q4_0. On Vulkan, F16 uses integer rows and, on integer-tile device profiles, eligible dense integer prompts. Other products use the documented wider F32 path; explicit F32 uses F32 inputs, and BF16 uses rounded inputs. Existing range repair remains. A device missing the required properties refuses the type before loading; format support does not silently select another device. [PRECISION](PRECISION.md) describes the arithmetic contract, and [STATUS](STATUS.md) records the measured performance and any open reference-speed cells.
