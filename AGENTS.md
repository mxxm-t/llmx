# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** - a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs quantized or F32 Qwen3-style transformers on x86 CPU with AVX2/FMA/F16C
or on a Vulkan device, and is structured so formats, quantizations, backends, and multi-device / cluster
serving can be added later without touching the core.

## Build

`docs/BUILD.md` is the one page on building: the prerequisites and commands on Windows, Linux and macOS, the Vulkan backend, the Docker image, where the binary lands, the build identifier and the test commands.
In short, `build.bat` writes `llmx.exe` to the repo root, and `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` followed by `cmake --build build --config Release` builds the binary and the native tests.
The notes below are for working on llmx itself.

For MSVC builds in temporary worktrees, use a fresh build directory or `cmake --build build --config Release --clean-first` after header changes.
An incremental invocation here emitted MSB8029 and left the executable older than edited headers, despite exiting successfully.
Preserve the compile log and verify that the changed source was actually rebuilt before claiming a gate.

The build dir's `generated/config.hpp` is produced from `cmake/llmx-config.hpp.in`; the checked-in `src/config.hpp` is the fallback used by the plain `build.bat` path.
Keep the two in sync when you add build knobs.

The Linux MI50 machine builds and runs the Vulkan backend in the image from `docker/Dockerfile` (`docs/BUILD.md`, Docker), which carries the loader, the Mesa driver, the headers and the shader compiler and takes the cards from the host through `/dev/dri`.
That leaves the shared machine's packages untouched.

## Principles

- **Performance first.** llmx is a *runtime*: a slow-but-correct implementation
  is not enough. Every change to a hot path (quantized matmul, RMSNorm, RoPE,
  attention, KV cache) should state the perf impact and be benchmarked, not just
  verified for correctness. When correctness and speed trade off, prefer the
  fast path and prove it is lossless (see `docs/ROADMAP.md` correctness gate).
- **Assess performance tradeoffs.** Report gains and regressions together across
  phases and models. A large gain can justify a minor loss elsewhere; do not
  automatically reject it on an isolated per-case cutoff. Preserve all results
  and explain the workload tradeoff. HF correctness and matched mx comparisons
  remain required.
- **Build both arms the same way, and run them in the same place.** A
  comparison is only about the change if nothing else differs between the
  binaries, and the same applies to where they run: a figure taken on one
  machine, driver and operating system does not subtract from one taken on
  another. Two reference figures that differed by build, compiler, operating
  system and driver at once were briefly read here as a driver effect; they
  said nothing. Compare llmx before, llmx after and the reference within one
  environment, and quote across environments only as separate results. The build embeds the Git
  revision and a dirty marker, and that string alone moved 0.6B prefill by
  several percent: the same change measured -8.03% with mismatched build
  identity and -0.08% with both arms built from detached worktrees at their
  own commits. Build both arms the same way, from the same kind of tree, and
  record each binary's `--version` alongside its hash.
- **Code layout is part of the measurement.** In a header-only runtime with
  one translation unit, an unrelated edit can reshuffle the hot path.
  Behaviourally identical builds, differing only by an unused function, span
  about four points of prefill on Qwen3-0.6B, and one of them failed the
  advance rule outright. Before believing a few-percent result on a hot-path
  cell, compare against a perturbed build of the same behaviour and check the
  difference is outside that band. Evidence in
  `docs/benchmarks/layout-sensitivity-20260921/`.

  **How big the band is depends on where you edit, so choose the control to
  match.** Appending an unused function to the end of `cpu_backend.hpp` moved
  0.6B prefill by -1.16/+0.63/-1.13/+0.33 percent over four runs. Adding a
  four-line bounds check inside `Tokenizer::decode`, a function that
  `Model::prefill` never calls, moved the same measurement by
  -3.13/-2.17/-3.09 percent and failed the advance rule twice. Same tree, same
  binaries built the same way, no change to any executed instruction of the
  measured path. A control that perturbs a different file than your change
  will understate the band and convince you a layout artifact is a
  regression; that is what happened here before the cause was bisected.
  Evidence in `docs/benchmarks/code-read-20260921/`, cells `06-layout*` and
  `06-tok*`.
- **Check machine contention.** Record timestamped background process CPU use
  and system CPU, disk and GPU activity before and throughout performance runs,
  using the same low-overhead monitoring in every arm. Separate benchmark and
  monitor activity from unrelated work. Do not run competing builds, tests or
  downloads. An idle machine is not required. Define activity flags before
  measuring, retain every planned matched block and report potentially affected
  runs alongside the complete results. Interleave and repeat arms to assess
  noise and activity imbalance; do not automatically stop, discard or replace
  a run because background activity is present. Do not discard isolated slow
  samples. If monitoring is unavailable, report that limitation instead of
  assuming the machine was idle or treating unknown activity as zero.
- **Dependency-free.** No external libs. The whole point is to control the full
  stack; reaching for a library erodes that.
- **Lean, not clever.** Add a seam only when a second implementation is on the
  roadmap. No speculative abstraction, no empty stubs.
- **One owner per concern, kept tight.** A feature is implemented once, in the
  lowest layer that holds what it needs, and callers reach it through one call
  (`docs/ARCHITECTURE.md`, Each concern has one owner). Before adding code,
  find the concern's owner; if a second caller needs the same steps, move them
  into the owner instead of copying them. The CLI and the server read flags
  and requests and call down; they hold no model, placement or loading logic.
  Remove what nothing reaches (functions, flags, kernels, diagnostic switches,
  experiment code) in the change that leaves it unreached.
- **Not overengineer, not underdo, no bloat.** This is a ground-up runtime with a
  deliberately small surface, so the default is to build *only* what the current
  feature needs and no more. But "lean" is not an excuse to ship a half-built
  feature: if a piece is claimed or a path is promised, it must actually work and
  be covered. Prefer the simplest thing that fully solves the stated problem;
  leave generic seams, flags, and scaffolding out until a concrete second use
  exists. When in doubt, ask whether the extra code pays for itself today.

## Configuration: flags, not environment variables

Runtime knobs are **CLI flags**. Environment variables are not used for runtime
configuration, because they are invisible in a command line, do not appear in
`--help`, and silently change results between runs - which is exactly what you
do not want while measuring.

- **A knob a user would set** is a flag: `--threads`, `--ubatch`. Choose the
  name that fits it best. An established name is a good candidate where it
  fits, but no name is taken only because another runtime uses it, and no
  behaviour is copied with it (llmx has no `n_batch`, see below).
- **A value the code can determine** is not a knob at all. Row blocking in the
  CPU matmul was briefly `LLMX_ROW_BLOCK`; it is now `DOT_ROWS`, the width the
  fused kernel handles, because measurement showed the knee follows the kernel
  and not the machine. A constant that is a property of the code does not
  belong in the environment.
- **Temporary A/B knobs get deleted** once they have answered their question.
  Both `LLMX_ROW_BLOCK` and `LLMX_ROW_BLOCK_BYTES` existed only to find a
  number and were removed with the finding recorded in `docs/STATUS.md`.
- **The runtime credential exception is `HF_TOKEN`** for `llmx pull` gated-repo
  access. It is not a tuning knob and must not appear in child argv or logs.
- **Test configuration**: `LLMX_BASELINE_GGUF` points `tests/baseline.py` at a fixture model; `LLMX_DEVICE`, set by `run_tests.py --device`, appends `--device` to every command that takes it so the suite runs on a device backend, except where a component names its own devices (`threads` names the CPU, whose thread counts it checks, and `split` names the CPU backends its tool splits over); `LLMX_CACHE_TYPE`, set by `run_tests.py --cache-type`, appends `--cache-type-k` and `--cache-type-v` the same way so the HF gate runs with a chosen cache type; `LLMX_LAYER_SHARES`, set by `run_tests.py --layer-shares`, appends `--layer-shares` so a device list is tested at a split the fit would not choose.
  The synthetic bench (`bench` without `--model`) takes only `--device` and `--threads`, so it gets neither shares nor cache types, and `split` gives its tool equal shares and the cache types it names itself.
  The runtime stores f16 by default, so the components that check exact f32 arithmetic against independent fixtures (`f32`, `moe`, `shards`, `server`) ask for f32 sides themselves and skip when `--cache-type` asks for another type (`common.f32_cache_skip`).
  That is test configuration, not runtime configuration, and it reaches the binary only as the flags.

If you add a flag, add it to `docs/USAGE.md` and to `print_usage` in the same
change, or it does not exist as far as a user is concerned.

## Verify

The round-trip component (`tests/roundtrip.py`, under Tests below) checks
the format and quantization paths: it builds a random F32 model, quantizes
it to Q8_0 and Q4_0 through the CLI, dequantizes it back and bounds the
error, and decodes the written blocks from the format description rather
than with llmx's own reader.
Q4_1 and Q4_K, which `quantize` does not write, are decoded the same way from raw blocks the test writes, chosen so every scale, min and nibble bit reaches a decoded value, and `dequantize` must match bit for bit.
Run it alone against the root `llmx.exe` that `build.bat` writes, or as part of the suite for a CMake build:
```
python tests/roundtrip.py
python tests/run_tests.py --exe build/Release/llmx.exe
```
`llmx.exe info FILE` lists the metadata and tensors of a written GGUF.

## Tests

Run the native backend, model, chat-template, KV storage, loader and streaming checks
after a CMake build:
```
ctest --test-dir build -C Release --output-on-failure
```

`json` checks syntax, numeric/locale boundaries, UTF-8 and escaped Unicode,
malformed input, nesting limits and JSON output string escaping. The Q8/Q4 round-trip test also checks
escaped Unicode tensor names through the actual CLI.

`gguf-validation` checks independent binary fixtures for field lengths/counts,
array depth, tensor arithmetic, file extents, quantized row widths and custom
alignment. These are format checks; they do not establish model-schema safety.
`load-progress` also checks early rejection and a file truncated before loading, refused before any progress.

`gguf-shards` covers complete shard sets, metadata-only first shards, exact
payloads, inconsistent metadata, truncation, aggregate progress and Unicode
file paths. `hub-manifest`, `hub-pull` and `hub-transport` are offline tests of
variant selection/hash vectors, concurrent range assembly/cache repair and
native curl child lifetime/response handling. Real transfers are separate
integration checks. The Python `shards` component compares sharded synthetic
logits and NLL against the independent HF fixture.

`model-validation` checks Qwen configuration ranges/defaults, required tensor
layouts and in-memory storage before model execution buffers are allocated.
It covers tied/untied output, supported matrix types and singleton axes. An
asynchronous test backend also checks loading failure and model teardown drain
pending work before releasing buffers, including split placements and backend
reuse. It does not validate numeric weights, arbitrary token IDs or failed-session recovery.
Each model it builds runs on a one-thread CPU backend that must start no worker threads.

`backend-group` checks mixed types, uneven rows, batches, thread counts,
output boundaries and fallback behavior against separate calls and double dots.
It also checks every finite f16 scale against signed Q8 weight extremes using
one-hot inputs with exact expected products, and the three-column prefill
reduction against ordered scalar FMA across dimension tails and unaligned
inputs. The independent HF fixtures below remain the external correctness gate.

`fused-dot-overflow` pins the two kernel families apart. `dot_row_impl` folds
the scale into each weight before the activation, avoiding the demonstrated
scale-after-sum overflow; accumulation can still overflow under cancellation.
The fused K-quant dots accumulate first and apply the scale after,
which is what makes them fast and what lets a large activation reach infinity
before a small scale could bound it. Those rows fall back to dequantizing.
Sixteen cases across Q8_0/Q4_K/Q5_K/Q6_K cover tiny and zero scales against
huge and ordinary inputs, once on the float dots and once on the decode dots
over quantized activations, which cannot overflow since each block is scaled
first and are bounded against the sum of magnitudes.

`q8-dots` checks those decode dots (`backends/cpu/q8_dots.hpp`), 8-bit for
Q8_0/Q4_K/Q5_K and 16-bit for Q4_0/Q4_1/Q6_K, against a double-precision
reference fed the same quantized activations, and that a decode row computes
the same alone, beside other rows and in a grouped call, bit for bit. It also
checks activation quantization independently against the original inputs over
float exponent boundaries, tiny/subnormal blocks and reciprocal-overflow
thresholds: packed range/sign/zero, exact integer sums, nearest reconstruction,
ties to even and partial-block guards. Fallback reconstruction checks are
strict; ordinary SIMD controls allow its existing float-rounding error.
These run under gradual underflow;
they do not establish nonfinite-input handling or flush-to-zero behavior.

`prefill-scope` uses self-generated model fixtures to check caller-once execution,
nesting/thread guards, allocation and microbatch boundaries, error draining and
scope reuse. Windows-only `prefill-placement` covers real eligible topology,
unsupported topology fallback and synthetic apply/restore failures without
requiring a six-core hosted runner. Neither replaces the independent HF gate.

`fp16` checks binary32 to binary16 conversion without an oracle library:
every finite half must encode back to its own bits, and every encoded float
must be at least as close as either neighbouring half, ties to even.

`sampler` calls `infer::sample` on hand-picked logits with expectations taken from the definitions.
Temperature 0 takes the largest score and the lowest id on a tie.
The repetition penalty divides a seen token's positive score and multiplies a negative one, by the penalty and once however often the token was seen, so a repeated leader loses to the runner-up.
`top_k` 1 is greedy at any temperature, `top_k` 3 never draws outside the three best, and a `top_p` between the first probability and the sum of the first two keeps exactly those two.
The nucleus is measured after the temperature and within the `top_k` window, which default requests combine: at temperature 2 a `top_p` of 0.7 keeps three tokens, and `top_k` 2 with `top_p` 0.6 keeps one.
Draws at temperatures 1 and 0.5 fall within three binomial standard deviations of the softmax at that temperature, a bound that rejects the temperature applied twice or ignored.
A seed repeats its sequence and seed 0 keeps the default state; every seed is fixed, so the frequency checks draw the same tokens on every run.

`backend-errors` injects task and startup-allocation failures, checks completion
before error propagation, and exercises pool reuse and thread reconfiguration.
It also checks valid empty CPU transfers, rejected offsets/null sources,
unchanged storage and a zero thread hint preserving the current pool.
It pins that construction and count changes start no threads, and that the first dispatch at a count starts one pool of that size, which later dispatches reuse.
A start that fails partway fails its dispatch and keeps the count, and the next dispatch starts the whole pool without a new count.
It does not establish recovery of partially executed model sessions.

`backend-vulkan` exists only in a build with `LLMX_HAS_BACKEND_VULKAN=ON`.
It opens device 0, round-trips buffers through adopt, copy, write and read, checks zeroed allocations, host-visible memory read in place after a wait, monotonic tickets and the refusal of a KV budget that overflows, then runs every implemented kernel against the CPU backend on random inputs with bounds fixed in the test: exact where the arithmetic is the same operation in the same order, a stated relative tolerance where a transcendental or a reduction order differs.
The decode row kernel reads quantized rows against 16-bit integer activations, and on
a device whose profile prefers the integer dot the wide tile reads 8-bit
ones, so the CPU reference is fed the activations quantized the same way and
the comparison is about the dots; the norm, SiLU and attention kernels' twin
of their output is checked through a matmul from it. After the checks it
prints the matvec bandwidth per type and, when the device reports them,
the driver's per-kernel statistics (registers, shared memory, scratch);
`--isa DIR` additionally writes the driver's disassembly of each kernel
to that directory.
Attention additionally covers 80 combinations of head widths 32/40/64/128/256,
query/KV head ratios 1/2/4/8 and all four F32/F16 cache-side pairs, with nonzero
inputs at long histories. Both rows of a mixed short/long pass must equal the
same rows taken separately, bit for bit; CPU comparisons retain the bound
`1e-4 * (1 + abs(reference))`.
A group of a Q8_0 and a Q4_0 projection whose batch reaches the 8-bit tile crossover but not the other types' must equal each type alone forced onto the row kernel, bit for bit, on a device whose profile puts batches between the two.
It exits 77, which CTest reports as skipped, when there is no loader, no
device or a driverless loader.

`vulkan-buffer` checks constructor cleanup on a fake device that supplies every Vulkan call it makes, so it needs no loader and runs wherever the backend builds.
`vulkan-lifetime` opens a device, intercepts transfers and injects allocation failures to check queued storage ownership during KV growth, padded-copy creation/replacement/invalidation and argument-arena overflow.
It checks retry and unchanged KV accounting after failed growth.
Five kernel-construction cases substitute calls to check cleanup, poisoned failure outputs, retry and cache reuse.
Two query cases use a real diagnostic add dispatch to check creation failure/retry and destruction after device idle; these cases skip if diagnostic timestamps are unavailable.
Transfer ownership cases intercept copies so old failures cannot submit references to freed memory.
Broad device arithmetic remains covered by `backend-vulkan` and HF.

`http` starts the server's HTTP layer (`src/server/http.hpp`) on a system-chosen port from a thread and drives it with the layer's own client: a whole response, a body echoed back, a chunked stream whose chunks arrive as written, a whole response refused inside a stream, an oversized body refused with 413, a malformed request line refused with 400, an unknown route 404, a client seen by `peer_closed` as open while it waits for its answer, also after one urgent (out-of-band) byte, and as closed once it leaves, a write to it then throwing `ClientGone`, and the listener closed from the main thread ending the accept loop.
It runs on Linux, Windows and macOS.

`server-utf8` checks that the server's `utf8_sanitize` (`src/server/api.hpp`) turns a surrogate (ED A0 80), an overlong form (E0 80 80) and a value above U+10FFFF (F4 90 80 80) into U+FFFD, one per byte, as it does truncated and stray bytes, while valid text of every length stays unchanged, and that `utf8_complete` holds back a character whose bytes have not all arrived and lets a whole one or a stray continuation byte through; `error_json` repairs a stray byte in an error message in both reply shapes.

`placement` splits a two-layer model over two CPU backends with a device per
tensor role (`docs/EXECUTION.md`) and requires the bytes of the same model on
one backend for a prompt, decode steps, a history across a block edge, a
reset and a two-sequence pass. It counts copies, writes and submissions so
the residual stream crosses exactly where the placement changes and never on
one device, and each stage and crossing submits the devices it records on. It
refuses malformed placements, among them a device whose attention layers are
not one run, and sequences of another model.
It checks the fit to device budgets (`model/layer_split.hpp`): even shares where room allows, a device without room left out, a host device given only what the others cannot hold, layers placed by their own sizes, the busiest device given as few layers as fit, tied weights counted once, the host's tables, handoff buffers and staging counted where they sit, shares honored or refused, and the fitted placement exact against one device.
`place_model`, the one placement entry, asks the backends for their budgets, applies the request's ubatch, and splits by shares exactly as one device computes.
A three-layer model placed by `place_model` over two and three CPU backends at ubatch 3 takes a 13-token prompt in five chunks, more than the stages, so the pipelined prefill reuses its pass slots and both handoff buffers; the prompt, three decode steps, a second prompt continuing the history, every row of `score()` and a two-sequence pass must be exact against one backend, with the same `n_tokens` and `kv_used_bytes`.
A backend on the last stage then fails while the first stage is chunks ahead, on top of a history, once at an attention mid-prompt and once at the head on the last chunk (`FailingCpu` in `tests/tiny_qwen.hpp`, which `kv-cache` also uses): every storage's length and `kv_used_bytes` must be back at the history, and the same prompt again must be exact.
`place_model` refuses experts on the CPU beside several devices, and a stream point without experts on the CPU.
A request's histories grow the cache budget only where the context's blocks cannot hold them, each counted up to the context, and three histories that need three blocks run in one pass on one backend and over a split.

Run the Python suite (synthetic fixtures are generated locally; real-model HF
checks skip when their models are absent):
```
python tests/run_tests.py
```

For a CMake build, pass `--exe <path-to-built-llmx>`.
`--only` runs just the components it names, comma separated (`--only baseline`, `--only split,server`), and refuses a name the suite does not have.
CI uses `--no-perf-floor` for shared runners, `--require-tools` on every CMake build it runs the suite on so a tool missing beside the executable fails rather than skips (the `build.bat` binary has no tools beside it), and `--require-baseline` in its real-model job so missing fixtures fail.
That job also runs `--only baseline` with `--cache-type f32`, `llmx-split-check` on the Q8_0 over two CPU backends and `tools/server_mix_check.py` on the Q8_0; the Vulkan job runs the suite with `--device cpu` on the Vulkan-enabled binary.
Local performance floors remain enabled by default. See `docs/CI.md` for workflow coverage and reproduction commands.

- **Version** (`tests/version.py`): `--version` matches the CMake project
  version and build identifier format, and the usage banner starts with it.
- **CLI** (`tests/cli.py`): the command-line surface the numerical components do not reach.
  A Vulkan device is refused with an error and nothing on stdout, never run on the CPU instead, through a model command and through the synthetic bench.
  A build without the Vulkan backend refuses it, and so does a Vulkan build that cannot open it; where device 0 opens, an index no machine has stands in for the missing device.
  `info` on the synthetic MoE model names its architecture and layer count, and lists every tensor written with its type, shape and size.
  It also checks the CLI's usage errors, all refused before a model is opened: an unknown command, `serve` and `pull` without arguments, missing and extra arguments, unknown flags and flags without a value, a chat positional argument, a second prompt or `--stop`, the synthetic bench's flags with `--model` and the model run's without it, `--profile` off a single Vulkan device, `--moe-stream-from` without experts on the CPU, an unknown cache type, and numbers out of their form or range.
  Each exits with status 2, nothing on stdout and the command's page then the reason on stderr.
  The help check reads the overview and every command's page, each shown by `--help` and `-h` alike with status 0 and without a model.
  Every flag a page lists, in each spelling, is taken by its command: the line is read in full and fails only on a missing input file, or pull on its empty quant, or with `--size` and `--iters` runs the synthetic bench.
  Every flag another page lists, and one no page lists, is refused by a command whose page does not list it, as a usage error.
  It needs no device, so it runs in every job.
- **Round-trip** (`tests/roundtrip.py`): build a random F32 model and quantize it to Q8_0 and Q4_0 through the CLI, each dequantized and checked against its own bound.
  Regression gate for `quant/` + `format/`.
  Also checks quantize's JSON tensor schema/dimension and binary-length rejection,
  output preservation on validation failure, and valid one-to-four-dimensional
  conversion for both writable types.
  `dequantize` must match, bit for bit, a decode written from the format description: Q8_0 and Q4_0 on the blocks quantize writes, Q4_1 and Q4_K on raw blocks that reach every scale, min and nibble bit, so the Q8_0, Q4_0, Q4_1 and Q4_K references of `q8-dots` and `backend-group`, which take them from the same decoders, rest on an independent decode; their Q5_K and Q6_K references still rest on the real-model HF checks.
  Both types quantize and dequantize under a non-ASCII directory to the same bytes as under an ASCII one, and a type name quantize does not write is refused with exit status 2.
- **Perf** (`tests/perf.py`): time matmul / RMSNorm / RoPE hot paths and print
  throughput, so perf-first changes can be checked for regressions. Assert a
  generous floor so catastrophic slowdowns fail loudly without being flaky.
- **Tokenizer** (`tests/tokenizer.py`): encode/decode round-trips incl. unicode and special tokens, and refusal of a file naming another tokenizer or pretokenizer.
- **Perplexity** (`tests/perplexity.py`): a synthetic model with an analytic
  scoring oracle checks window boundaries, chunk limits, target counts, file
  and inline input parity and invalid flags.
  On the same model `logits` gives the same output for inline text and `--file` or `-f`, appends `--then-ids` ids separated by commas or whitespace, and refuses any other separator and an id past the vocabulary, 2^32 plus a valid id included.
- **Chat** (`tests/chat.py`): follow-up replies against independent HF/Jinja2
  goldens, including changed prefixes, stop/EOS and token-limit endings.
  CTest also runs `chat-template`, comparing the real Qwen template against
  Jinja2-rendered conversation fixtures. Regenerate these with
  `python tools/gen_chat_baseline.py`; running them needs no external libraries.
  The same test holds `is defined` and `is not defined` cases inline, each
  beside the text Jinja2 renders for it.
- **Thread controls** (`tests/threads.py`): actual auto/explicit phase counts,
  restoration after prefill, follow-up chat and HF-golden replies.
  Perplexity also checks batched/per-token counts, both batch-thread aliases
  and automatic/zero selection against an independent HF NLL fixture.
- **Loading and streaming** (CTest `load-progress`, `generation-stream`,
  `cli-output`):
  mapped-byte reporting, files truncated before loading, callback failures, early text
  delivery, split UTF-8 bytes and stop/EOS accounting;
  `cli-output` also reads `--device` lists as the commands do (canonical spellings, a device once, malformed entries refused), and the cache types as `exec_flag` reads them (one spelling each, an empty or unknown name refused before any model file is read).
  It runs the CLI's number readers (`int_arg`, `float_arg` with the sampler's ranges, `--seed`'s decimal 64-bit read) and `token_ids` over every malformed form: a missing value, a sign, space, base prefix, fraction or trailing character, infinity and NaN, a value past its range or its type, and an id that would narrow into the vocabulary.
- **KV cache** (`tests/kv_cache.cpp`, CTest `kv-cache`): block pool reuse and
  exhaustion, sequence prepare/commit/abort/reset, on-demand storage growth
  and retained reset across block boundaries, paged attention over two
  block tables against a double-precision reference, two sequences batched
  in one attention call against the same two taken separately, the ticket
  and release contract (one submission per pass, waits and syncs counted on
  every release path), the model transaction on failure, a two-entry
  `forward` against the entries run alone, and forks: a whole-block length
  sharing its blocks without allocating one, a length inside a block or
  past the history refused, refused appends into shared blocks, refcounted
  release, and a sequence forked at a block boundary continuing exactly as
  a fresh one fed the same history. This oracle supplements the
  independent HF gate.
  The shared KV storage (`BlockKVStorage`) is held to the doubling rule it replaced, written out in the test: every growth step and the peak, with out-of-order ids and mixed cache types, an overflowing budget refused at allocation, attention over a block no write backed refused, and the growth hooks, each old buffer retired once, the backend drained and the accounting unchanged when a growth fails, and a retry.
- **Server** (`tests/server.py`): `llmx serve` on a system-chosen port
  against the CLI on the same file, the synthetic F32 model without a
  download and the Q8_0 fixture when present: greedy through `/v1/generate`
  equals `generate --temp 0` alone and four at a time, a stream carries the
  same ids, a seeded request repeats, refusals, a client leaving mid-stream
  leaves nothing active, a chat turn, the compatible `/v1/completions`
  and `/v1/chat/completions` whole and streamed in the OpenAI clients'
  shape, a prompt repeating a finished request's tokens reuses its
  blocks with the CLI's greedy text, and the limits: a KV budget below the
  context bounds a request and a full queue answers 503.
  A sampling field outside the range the CLI's flag takes is refused with 400 on every route, `repetition_penalty` on the compatible routes included, and a `top_k` of -1 is refused on the native route and sampled as `top_k` 0 on the compatible one.
  The synthetic model's file name holds a byte that is not UTF-8 on Linux, and elsewhere characters beyond ASCII whose UTF-8 bytes code pages 932, 936, 949, 950 and 1257 cannot map, so a name read in the system code page there fails; `/v1/health` and `/v1/models` must name it as UTF-8, with a U+FFFD for each byte that belongs to no UTF-8 character.
  With the Q8_0 fixture, uncapped requests share a pool too small for all of them: a request is paused when it runs out and resumes from its history, and the check confirms each runs to its own end and the server counts a pause, though it does not yet compare a paused request's output with the same request run alone; a long prompt read one token a pass is paused while it is still prefilling and, resumed, gives the CLI's greedy text; a conversation of six turns on a 1024-token pool, its history growing past half the pool, reuses the last turn's history on every follow-up (its `reused_tokens` and the server's `prefix_tokens` grow each time) with each turn's greedy text equal to the CLI's; and a follow-up that fits the pool only once one donor goes, beside an unrelated donor, consumes the turn it repeats, so a later prompt repeating the unrelated request's history still reuses it with the CLI's greedy text.
  With the Q8_0 fixture too, a client that leaves a whole reply while it is generated, a streamed prompt while it is read one token a pass, or a request waiting for the one slot is noticed within seconds, though nothing written to it fails, and the server then holds nothing active or queued and starts a request reaching the whole pool at once; a client that shuts only its sending side during a whole reply gets no answer, the connection just closing; and each request left behind would run for thousands of passes, so a server that noticed nothing fails on any device.
  The synthetic MoE model's prompts must each give the same ids alone and four at a time, where a pass routes one request's prompt rows beside another's decode rows.
  On the CPU it runs as it is, so every job checks it; on a single device other than the CPU it runs with its experts on the host and prompts from three tokens streamed, where a pass holds streamed prompt rows beside host decode rows.
  A device list skips it, and the whole component skips under `--cache-type f16`.
  Throughput is measured separately with `tools/server_load.py`.
- **Serving load** (`tools/server_load.py`, suite component `server-load`): a running server's figures under closed-loop levels of concurrent users and open-loop Poisson request rates, through `/v1/generate`, `/v1/completions` or the reference server's `/completion`.
  Prompts of an exact token length come from a seeded word list, their lengths from the seed alone, replies of a fixed length ask to ignore the end of text, and each level reports time to first token, time per output token, inter-token and end-to-end latency percentiles, output and total tokens per second, and failures by reason; `--json` keeps every request's record, and the tool exits 3 when any timed request failed.
  The suite runs its `--self-test`, which needs no model: the stream reading of every API and the figures on made-up arrival times exactly, then every API in both loads against an in-process server with a refusal, a dropped stream, both timeouts and a short reply, checked on counts, reasons and time bounds a busy machine cannot break, then the whole tool's table, notes and exit status.
- **Many users** (`tools/server_mix_check.py`): a real model through
  `llmx serve` on any placement, a layer split above all. Requests mixing
  short and long prompts with short and long replies are run alone, then
  all at once, then skewed: long prompts land while others decode and every
  fourth client leaves mid-stream. Each request that finishes must give its
  ids alone, clients that left must leave nothing active, and the first
  requests must give the same text through `generate --temp 0`, whose
  prompt a split pipelines over its stages.
  The HF job runs it on the Q8_0 fixture on the CPU with `--requests 8 --cli 2`.
- **Long context** (`tools/long_context_check.py`): one 16k-token
  summarization prompt from `tests/data/wiki.test.raw`, greedy, 512
  generated tokens by default, sent to `llmx serve` on the device under
  test twice from fresh servers, which must give the same tokens; then the
  CPU reads the prompt and those tokens (`llmx logits --last`), and at
  every generated position the device's token must be the CPU's top choice
  or within `--margin` (0.5) logits of it. Nothing else here reaches a
  prompt that fills thousands of KV blocks and then decodes from that
  history. A hash across two backends is not the check: their activations
  round differently, and after a long prompt greedy decoding meets
  near-ties where either token is right, so identical text is only
  required of one backend against itself. It needs a real model and is run
  by hand, not by `run_tests.py`.
- **Layer split** (`tools/split_check.cpp`, target `llmx-split-check`, built in every configuration with tests, which is the default, and linked to the Vulkan backend when that is on): a model on one device against the same model split in equal shares over a comma-separated list of devices of the same kind (default `0,1`; a device is `cpu` or a Vulkan index), with optional decode steps, ubatch and cache type (`f16`, the default, or `f32`, both sides of both models), as raw float logits compared with `memcmp`: every position of a scored text through the prompt path, the prefill in chunks of the ubatch, which a split pipelines over its stages, and greedy decode steps, then three passes of a decoding sequence beside a fresh prompt, every row's logits.
  The split must be bit-identical, since each layer runs the same kernels on the same rows wherever it sits; a split over different backends is held to the HF bounds instead.
  Each listed device is a backend of its own, without the CLI's listed-once rule, so `cpu,cpu` is two CPU backends.
  The `split` component (`tests/split.py`) runs the tool found beside `--exe` on the tiny F32 model, tied and untied, and the tiny MoE model, one CPU against `cpu,cpu` and the MoE also against `cpu,cpu,cpu`, at ubatch 1, 3 and 16 and with f16 and f32 caches, with 3 decode steps after a 13-token text, which fills their 16-token context.
  It skips when the tool is not there, unless `--require-tools` is given, and the configured device, shares and cache type do not reach it.
  The HF job runs it on the Q8_0 fixture over the perplexity excerpt, `cpu` against `cpu,cpu` with 8 steps and 64-token chunks; other real models and splits over devices are run by hand.
- **F32** (`tests/f32.py`): deterministic small-model weights with full logits
  and windowed NLL generated independently by HF. Covers tied/untied weights,
  odd dimensions, batch tails and threads without downloading a model,
  and `bench --model` at a depth and with `--seqs 2`, two sequences where the model's context fills one cache block.
  `logits --file` must print what the same prompt inline does.
  The `--last` rows of the prompt, in one pass and in several, and of its first three tokens continued by `--then-ids`, are held to the HF bound at their positions.
  Each row is printed once, and the rows over several passes and after `--then-ids` are the bytes of the one-pass rows at the same positions.
- **MoE** (`tests/moe.py`): the same for a tiny `qwen3moe` model against HF `Qwen3MoeForCausalLM` (`tools/gen_baseline.py moe`), two routed layers and one dense, across batch widths and threads and, on a device, with the experts of one or every routed layer on the CPU (`--n-cpu-moe`, `--cpu-moe`).
  On a device it also streams those layers to the device for prompts from a length on (`--moe-stream-from`): from 0, from 1, which a generated token never reaches, and from 4, which only the longer prompts reach.
- **Baseline** (`tests/baseline.py`): real-model EXTERNAL ground truth.
  Compares llmx against golden fixtures generated once from the HF reference by `tools/gen_baseline.py` and committed to `tests/data/`.
  Needs a real model, so it SKIPS when none is on disk; point it at one with `LLMX_BASELINE_GGUF`.
  Otherwise each check, the tokenizer's included, reads its model from the HF cache at the revision `BASELINE_MODELS` pins, the path `tools/fetch_test_models.py` downloads to.
  Its models are pinned in `tests/data/fixtures.json` (repo, revision, file and SHA-256), which `tools/fetch_test_models.py` downloads, and their bounds sit in `tests/baseline.py`, which refuses to load unless each pinned file has bounds and each bounded file is pinned once.
  Every perplexity cell is scored twice, in batched passes (the default) and with `--per-token`, so the prompt and decode kernels both meet the reference.
  Its logit and PPL outputs go through the validators the 8B check uses, `common.check_logits` and `common.check_ppl`, at each fixture model's bounds: the exact prompt token count, ten unique in-vocabulary IDs with finite logits sorted from the top, and exactly the PPL fields with every count exact.
  Regenerating tokenizer fixtures needs `tokenizers` and `huggingface_hub`; numerical fixtures also need `torch` and `transformers`.
  RUNNING the suite needs none of these packages.
- **Reference generator** (`tests/reference_generator.py`): standard-library
  checks for pinned reference selection, separate alternate-model output and
  forwarding the revision/float32/eager settings to the HF loaders. Actual
  reference generation and model correctness remain separate checks.
- **Reference consumer** (`tests/reference_consumer.py`): standard-library rejection tests for changed 8B fixtures, damaged logits/PPL, wrong model identity and failed launches, and a passing run over simulated outputs that must have 41 checks with each NLL case scored in both modes.
  It is included in the ordinary suite; it does not load or download the 8B model.
- **Fixture downloader** (`tests/fetch_models.py`): fifteen offline tests
  of `tools/fetch_test_models.py` against simulated responses: a verified
  download and its cached reuse, a corrupt cached file replaced, bounded
  retries with backoff on 429, transient server errors and network
  failures (Retry-After as seconds or a date, HF reset headers, malformed
  headers), no early retry when the server asks for too long a wait, no
  retry on permanent HTTP or local write errors, interrupted and short
  reads restarted from a clean temporary file, and a hash mismatch failing
  with the existing file kept. It is not a `run_tests.py` component; CI
  runs it as a step of its own (`docs/CI.md`).

The optional real 8B check is separate from the ordinary suite and default CI:

```
python -X utf8 tests/baseline_8b.py --exe build/Release/llmx.exe --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

Use a new output directory; on Linux use `--exe build/llmx`.
The consumer verifies the model/fixture hashes and writes raw outputs plus `report.json`, including failures.
It requires exact tokenizer/input IDs, six top-1 matches, top-5 overlap 5/5 and valid top-10 logits; absolute NLL bounds are 0.01 for the continuous excerpt and 0.02 per windowed case.
Each NLL case is scored twice, in batched passes (`ppl-NN`) and with `--per-token` (`ppl-NN-per-token`), as in `tests/baseline.py`, so a run has 41 checks.
These prospective Q8 bounds were frozen before the 8B comparison.
Since 2026-09-25 the overlap, here and in `tests/baseline.py`, counts a swap at the 5th place as agreement when the reference puts both tokens within 0.1 logits of its 5th value (`common.top5_overlap`): such near ties reorder with any summation order.
See `docs/ASSETS.md` for provenance and scope: short rankings/excerpts do not establish full-corpus or deep-context correctness, and the exact original GGUF conversion revision is undocumented.

The two project gates are external and are defined in `docs/ROADMAP.md` #8:
**correctness is the HF reference**, and **performance must be at least
mx-llama.cpp** on the same model, quant, prompt and hardware. Do not substitute
a self-consistency check for either. llmx passed a fully green suite while eight
correctness bugs were live, because every test compared llmx against llmx.

When you change a hot path, run `tests/perf.py` and note the before/after in the
commit message. The lossless correctness gate (path-controlled perplexity on a
real model) is tracked in `docs/ROADMAP.md`.

Report numerical results in comparison tables: before / candidate / mx-llama.cpp
for performance, and observed error / HF bound for correctness. Include units
and distinguish measured results from targets or unverified claims.

Real models and corpora for manual verification (the Qwen3-8B Q8_0 model, the
wikitext test set) are documented in `docs/ASSETS.md`.

## Architecture

See `docs/ARCHITECTURE.md` for the layer diagram and rules. The rule that
matters: **each layer depends only on the layers below it** -
`cli > server > inference > model > backends > tokenizer > format > quant > core`;
`server/` uses the inference layer and adds only scheduling and transport.

| Directory    | Contents                                        |
|--------------|-------------------------------------------------|
| `core/`      | fp16 <-> f32, JSON parser, UTF-8, common types  |
| `hub/`       | CLI acquisition path: Hub metadata, curl HTTPS and verified multi-stream cache |
| `quant/`     | QuantType registry + Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K kernels |
| `format/`    | ModelFormat interface + GGUF v3 impl           |
| `tokenizer/` | byte-level BPE, Qwen2/Qwen3 pretokenizer       |
| `model/`     | Qwen3 config + forward pass (dense and qwen3moe), KV cache, layer split over devices |
| `backends/`  | Backend interface + cpu/ (AVX2) and vulkan/ impls; one worker pool; `device_profile.hpp`, the device numbers a GPU backend shapes its kernels by |
| `inference/` | sampler, generate, perplexity, chat template renderer      |
| `server/`    | multi-user server (`docs/SERVER.md`): HTTP layer, scheduler with prefix reuse, routes |
| `cli/`       | thin argument parsing + dispatch               |

## Starting a feature

A fresh agent (or human) can jump straight into a feature by reading, in order:
1. `AGENTS.md` - this file: what the project is, how to build and verify.
2. `docs/ARCHITECTURE.md` - the layers and the dependency rule.
3. `docs/ROADMAP.md` - the stable plan; pick or confirm the feature there.
4. `docs/STATUS.md` - what is already in flight and where each feature stands.

When you start (or pick up) a feature:
- Keep each independent feature on its own branch based on current main.
  Do not base unrelated work on another unmerged feature. If a dependency is
  necessary, name it in STATUS and keep the dependent change separate.
  Unrelated documentation corrections belong in a separate commit; a feature's
  own documentation ships with that feature.
- Open a new per-feature block in `docs/STATUS.md` (or update the existing one)
  **before** writing code: **Goal / Done / Left / Gotchas**. That block is what
  lets the next agent pick the feature back up with a "continue feature X"
  prompt, so keep it current.
- A feature lands with a test the hosted workflow runs, or its STATUS block names the hand check that covers it and says why no hosted runner can run it.
- A bug fix lands its failing test first: a commit that adds the test, failing on the unfixed code, then the fix that makes it pass.
- Don't invent new directions - follow the roadmap. When the feature ships,
  delete its block and mark the row `Done` in the STATUS table.

## Checkpoints

Update `docs/STATUS.md` and commit at each meaningful checkpoint - at minimum
when a feature, a milestone, or a discrete chunk of work is complete. Each
commit should leave `docs/STATUS.md` accurate: `Done`/`Left` reflect reality,
the build passes, and tests are green. A fresh agent should be able to read
STATUS.md and resume exactly where the last commit left off.

At every completed checkpoint, review all project Markdown files against the
current code, tests, CLI, build configuration and recorded results. Correct stale
claims and links, distinguish historical measurements from current status, and
record the review in STATUS.md. This includes README, AGENTS, all docs/ pages
and per-source documentation; update only what needs changing.

Build identification is automatic: both supported build paths combine the
explicit release version with Git revision and tracked-dirty state. Update
CMake's project version and the fallback config together for a release; do not
increment versions or create tags merely because a build ran. `--version`
reports the embedded value, with `unknown` for builds without Git metadata.

## Build-time vs runtime

- **Backends** are the only compile-time concern (GPU SDKs are heavy). Gated by
  `LLMX_HAS_BACKEND_*` in `src/config.hpp` (see `cmake/llmx-config.hpp.in`).
  Vulkan is implemented; the ROCm, CUDA and SYCL options exist without code.
- **Model architectures** are planned to be compiled in and selected from
  metadata; today Qwen3, dense and mixture of experts (`qwen3moe`), is implemented.
- **Split mode** is a runtime flag: a `--device` list splits by layers
  (`docs/MULTI-DEVICE.md`); tensor groups and node count are planned. See
  `docs/ROADMAP.md`.

## Conventions

- mx-llama.cpp may be inspected as a reference, but do not copy its source.
  Implement llmx changes independently in this project's architecture and style.

- Use ASCII characters in code, comments, documentation and commit messages.
  Preserve Unicode test coverage using escaped literals and fixture data.

- Code comments are short and carry only what helps read the code: what a thing is, an invariant it keeps, or a non-obvious reason, in a sentence or two.
- No walls of text in code. Measurements, timings, rejected alternatives and history belong in the docs (`docs/STATUS.md`, `docs/VULKAN.md` and the like); a comment may point there.
- Do not break a sentence across lines in code comments or commit messages. A line ends where a sentence ends; a long sentence stays on one line rather than wrapping at a column.
- A file's comments follow these rules once the file is touched, and a branch's files are swept before it merges.

- CPU and shared runtime code use headers (`#pragma once` + `inline`),
  compiled via `src/cli/main.cpp`. The optional Vulkan backend has its own
  compiled translation unit. Keep one TU per logical unit when adding `.cpp` files.
- Include paths are relative to `src/` root: `#include "format/gguf.hpp"`.
- Every source file under `src/` is described in `docs/src/`: a page per file
  (`<dir>-<file>.md`), or one page for a directory whose files form one unit
  (`server.md`, `hub.md`, `backends-vulkan.md`). A page changes in the same
  commit as its file whenever what it says changes; a new file comes with its
  description, and a removed file takes it with it.
- No comments in code unless they explain a non-obvious decision or algorithm (e.g. the fp16 rounding, the GGUF padding rules, the AVX2 dequant+FMA path).
- Cross-platform (Windows / Linux / macOS): guard MSVC-vs-GCC intrinsics with
  `#if defined(_MSC_VER)`; use `<intrin.h>`/`<cpuid.h>` appropriately.
- Don't overengineer. Add a seam (interface) only when a second implementation
  is actually on the roadmap. Empty stubs are discouraged.

## Roadmap

`docs/ROADMAP.md` lists: more quant formats, more architectures, more formats,
backends (Vulkan written first, ROCm first-class on Linux), per-layer and
per-tensor multi-device split, multi-node cluster, a multi-user server, and
Hugging Face integration (`llmx pull` plus reading what the Hub actually
hosts). Follow the roadmap before inventing new directions.

GPU backends are **not** drop-in the way a quant type is. The device
execution model (`docs/ROADMAP.md` #4a, `docs/DEVICE-EXECUTION.md`) and the
Vulkan backend over it (`docs/VULKAN.md`) are complete; a further vendor
backend implements the whole `Backend` interface over its own allocator,
kernels and queue, and the Vulkan backend is the measure of what that
takes. Don't pick up "add the ROCm backend" expecting the "one file + one
registry entry" experience Q4_0 had.
