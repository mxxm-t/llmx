# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** - a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs quantized or F32 Qwen3-style transformers on x86 CPU with AVX2/FMA/F16C
or on a Vulkan device, and is structured so formats, quantizations, backends, and multi-device / cluster
serving can be added later without touching the core.

## Build

Windows (MSVC):
```
build.bat
```
produces `llmx.exe` in the repo root.

Cross-platform (Windows / Linux / macOS), CMake:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```
For MSVC builds in temporary worktrees, use a fresh build directory or
`cmake --build build --config Release --clean-first` after header changes.
An incremental invocation here emitted MSB8029 and left the executable older
than edited headers, despite exiting successfully. Preserve the compile log
and verify that the changed source was actually rebuilt before claiming a gate.

The build dir's `generated/config.hpp` is produced from
`cmake/llmx-config.hpp.in`; the checked-in `src/config.hpp` is the fallback used
by the plain `build.bat` path. Keep the two in sync when you add build knobs.

A Linux machine with AMD cards but no Vulkan driver or SDK of its own
builds and runs the Vulkan backend through `docker/Dockerfile`, which
carries the loader, the Mesa driver, the headers and the shader compiler
and takes the cards from the host through `/dev/dri`. The file itself
gives the two commands. This is how the Linux MI50 machine runs the backend, and it
leaves that shared machine's packages untouched.

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
- **Test configuration**: `LLMX_BASELINE_GGUF` points
  `tests/baseline.py` at a fixture model; `LLMX_DEVICE`, set by
  `run_tests.py --device`, appends `--device` to every command that takes
  it so the suite runs on a device backend; `LLMX_CACHE_TYPE`, set by
  `run_tests.py --cache-type`, appends `--cache-type-k` and
  `--cache-type-v` the same way so the HF gate runs with a chosen cache
  type; `LLMX_LAYER_SHARES`, set by `run_tests.py --layer-shares`,
  appends `--layer-shares` so a device list is tested at a split the fit
  would not choose. The runtime stores f16 by default, so the components that check
  exact f32 arithmetic against independent fixtures (`f32`, `shards`,
  `server`) ask for f32 sides themselves unless `--cache-type` overrides
  them. That
  is test configuration, not runtime configuration, and it reaches the
  binary only as the flags.

If you add a flag, add it to `docs/USAGE.md` and to `print_usage` in the same
change, or it does not exist as far as a user is concerned.

## Verify

Round-trip test (generates fixtures, quantizes, dequantizes, compares):
```
python make_test.py
llmx.exe quantize test_model.json test_model.bin test_model.gguf
llmx.exe dequantize test_model.gguf test_out.json test_out.bin
python -c "import struct;a=open('test_orig.bin','rb').read();b=open('test_out.bin','rb').read();fa=struct.unpack('<%df'%(len(a)//4),a);fb=struct.unpack('<%df'%(len(b)//4),b);print('max err',max(abs(x-y) for x,y in zip(fa,fb)))"
```
Note: `verify_gguf.py` is an independent spec parser that aligns between tensor
infos; our writer packs them contiguously. That script's alignment is stricter
than the spec requires, so use `llmx.exe info` as the authoritative check.

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

`backend-errors` injects task and startup-allocation failures, checks completion
before error propagation, and exercises pool reuse and thread reconfiguration.
It also checks valid empty CPU transfers, rejected offsets/null sources,
unchanged storage and a zero thread hint preserving the current pool.
It does not establish recovery of partially executed model sessions.

`backend-vulkan` exists only in a build with `LLMX_HAS_BACKEND_VULKAN=ON`. It
opens device 0, round-trips buffers through adopt, copy, write and read,
checks zeroed allocations, host-visible memory read in place after a wait
and monotonic tickets, then runs every implemented kernel against the CPU
backend on random inputs with bounds fixed in the test: exact where the
arithmetic is the same operation in the same order, a stated relative
tolerance where a transcendental or a reduction order differs. The decode
row kernel reads quantized rows against 16-bit integer activations, and on
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
It exits 77, which CTest reports as skipped, when there is no loader, no
device or a driverless loader.

`vulkan-buffer` checks constructor cleanup with substituted Vulkan calls and
needs only the loader. `vulkan-lifetime` opens a device, intercepts transfers
and injects allocation failures to check queued storage ownership during KV
growth, padded-copy creation/replacement/invalidation and argument-arena
overflow. It checks retry and unchanged KV accounting after failed growth.
Five kernel-construction cases substitute calls to check cleanup, poisoned
failure outputs, retry and cache reuse. Two query cases use a real diagnostic
add dispatch to check creation failure/retry and destruction after device idle;
these cases skip if diagnostic timestamps are unavailable. Transfer ownership
cases intercept copies so old failures cannot submit references to freed
memory. Broad device arithmetic remains covered by `backend-vulkan` and HF.

`http` starts the server's HTTP layer (`src/server/http.hpp`) on a
system-chosen port from a thread and drives it with the layer's own client:
a whole response, a body echoed back, a chunked stream whose chunks arrive
as written, an oversized body refused with 413, a malformed request line
refused with 400, an unknown route 404, and the listener closed from the
main thread ending the accept loop. Windows and Linux.

`placement` splits a two-layer model over two CPU backends with a device per
tensor role (`docs/EXECUTION.md`) and requires the bytes of the same model on
one backend for a prompt, decode steps, a history across a block edge, a
reset and a two-sequence pass. It counts reads and writes so the residual
stream crosses exactly where the placement changes and never on one device,
and refuses malformed placements and sequences of another model.

Run the Python suite (synthetic fixtures are generated locally; real-model HF
checks skip when their models are absent):
```
python tests/run_tests.py
```

For a CMake build, pass `--exe <path-to-built-llmx>`. CI uses
`--no-perf-floor` for shared runners and `--require-baseline` in its real-model
job so missing fixtures fail. Local performance floors remain enabled by
default. See `docs/CI.md` for workflow coverage and reproduction commands.

- **Version** (`tests/version.py`): `--version` matches the CMake project
  version and build identifier format, and the usage banner starts with it.
- **Round-trip** (`tests/roundtrip.py`): build a random Q8_0 model, quantize,
  dequantize, assert max error below a Q8_0-appropriate bound. Regression gate
  for `quant/` + `format/`.
  Also checks quantize's JSON tensor schema/dimension and binary-length rejection,
  output preservation on validation failure, and valid one-to-four-dimensional
  conversion for both writable types.
- **Perf** (`tests/perf.py`): time matmul / RMSNorm / RoPE hot paths and print
  throughput, so perf-first changes can be checked for regressions. Assert a
  generous floor so catastrophic slowdowns fail loudly without being flaky.
- **Tokenizer** (`tests/tokenizer.py`): encode/decode round-trips incl. unicode
  and special tokens.
- **Perplexity** (`tests/perplexity.py`): a synthetic model with an analytic
  scoring oracle checks window boundaries, chunk limits, target counts, file
  and inline input parity and invalid flags.
- **Chat** (`tests/chat.py`): follow-up replies against independent HF/Jinja2
  goldens, including changed prefixes, stop/EOS and token-limit endings.
  CTest also runs `chat-template`, comparing the real Qwen template against
  Jinja2-rendered conversation fixtures. Regenerate these with
  `python tools/gen_chat_baseline.py`; running them needs no external libraries.
- **Thread controls** (`tests/threads.py`): actual auto/explicit phase counts,
  restoration after prefill, follow-up chat and HF-golden replies.
  Perplexity also checks batched/per-token counts, both batch-thread aliases
  and automatic/zero selection against an independent HF NLL fixture.
- **Loading and streaming** (CTest `load-progress`, `generation-stream`,
  `cli-output`):
  mapped-byte reporting, files truncated before loading, callback failures, early text
  delivery, split UTF-8 bytes, legacy filtering and stop/EOS accounting;
  `cli-output` also reads `--device` lists as the commands do (canonical
  spellings, a device once, malformed entries refused).
- **KV cache** (`tests/kv_cache.cpp`, CTest `kv-cache`): block pool reuse and
  exhaustion, sequence prepare/commit/abort/reset, on-demand storage growth
  and retained reset across block boundaries, paged attention over two
  block tables against a double-precision reference, two sequences batched
  in one attention call against the same two taken separately, the ticket
  and release contract (one submission per pass, waits and syncs counted on
  every release path), the model transaction on failure, a two-entry
  `forward` against the entries run alone, and forks: shared full blocks,
  a copied tail, refused appends into shared blocks, refcounted release,
  and a forked sequence continuing exactly as a fresh one fed the same
  history. This oracle supplements the independent HF gate.
- **Server** (`tests/server.py`): `llmx serve` on a system-chosen port
  against the CLI on the same file, the synthetic F32 model without a
  download and the Q8_0 fixture when present: greedy through `/v1/generate`
  equals `generate --temp 0` alone and four at a time, a stream carries the
  same ids, a seeded request repeats, refusals, a client leaving mid-stream
  leaves nothing active, a chat turn, the compatible `/v1/completions`
  and `/v1/chat/completions` whole and streamed in the OpenAI clients'
  shape, a prompt repeating a finished request's tokens reuses its
  blocks with the CLI's greedy text, and the limits: a KV budget below the
  context bounds a request and a full queue answers 503. Skips under
  `--cache-type f16`.
  Throughput is measured separately with `tools/server_load.py`.
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
- **Layer split** (`tools/split_check.cpp`, target `llmx-split-check`): a
  model on one device against the same model split 1:1 over two of the
  same kind (a device is `cpu` or a Vulkan index), as raw float logits
  compared with `memcmp`: every position of a scored text through the
  prompt path, the prefill and greedy decode steps, then three passes of a
  decoding sequence beside a fresh prompt, every row's logits. The split must be
  bit-identical, since each layer runs the same kernels on the same rows
  wherever it sits; a split over different backends is held to the HF
  bounds instead. It takes the tiny fixtures `tests/f32.py` and
  `tests/moe.py` write as well as real models, and is run by hand.
- **F32** (`tests/f32.py`): deterministic small-model weights with full logits
  and windowed NLL generated independently by HF. Covers tied/untied weights,
  odd dimensions, batch tails and threads without downloading a model,
  and `bench --model` at a depth.
- **MoE** (`tests/moe.py`): the same for a tiny `qwen3moe` model against HF
  `Qwen3MoeForCausalLM` (`tools/gen_baseline.py moe`), two routed layers and
  one dense, across batch widths and threads and, on a device, with the
  experts of one or every routed layer on the CPU (`--n-cpu-moe`).
- **Baseline** (`tests/baseline.py`): real-model EXTERNAL ground truth.
  Compares llmx against golden fixtures generated once from the HF
  reference by `tools/gen_baseline.py` and committed to `tests/data/`. Needs a
  real model, so it SKIPS when none is on disk; point it at one with
  `LLMX_BASELINE_GGUF`. Every perplexity cell is scored twice, in batched
  passes (the default) and with `--per-token`, so the prompt and decode
  kernels both meet the reference. Regenerating tokenizer fixtures needs `tokenizers` and
  `huggingface_hub`; numerical fixtures also need `torch` and `transformers`.
  RUNNING the suite needs none of these packages.
- **Reference generator** (`tests/reference_generator.py`): standard-library
  checks for pinned reference selection, separate alternate-model output and
  forwarding the revision/float32/eager settings to the HF loaders. Actual
  reference generation and model correctness remain separate checks.
- **Reference consumer** (`tests/reference_consumer.py`): standard-library
  rejection tests for changed 8B fixtures, damaged logits/PPL, wrong model
  identity and failed launches. It is included in the ordinary suite;
  it does not load or download the 8B model.

The optional real 8B check is separate from the ordinary suite and default CI:

```
python -X utf8 tests/baseline_8b.py --exe build/Release/llmx.exe --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
```

Use a new output directory; on Linux use `--exe build/llmx`. The consumer
verifies the model/fixture hashes and writes raw outputs plus `report.json`,
including failures. It requires exact tokenizer/input IDs, six top-1 matches,
top-5 overlap 5/5 and valid top-10 logits; absolute NLL bounds are 0.01 for the
continuous excerpt and 0.02 per windowed case. These prospective Q8 bounds
were frozen before the 8B comparison. Since 2026-09-25 the overlap, here and
in `tests/baseline.py`, counts a swap at the 5th place as agreement when the
reference puts both tokens within 0.1 logits of its 5th value
(`common.top5_overlap`): such near ties reorder with any summation order. See `docs/ASSETS.md` for provenance and
scope: short rankings/excerpts do not establish full-corpus or deep-context
correctness, and the exact original GGUF conversion revision is undocumented.

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
| `core/`      | fp16 <-> f32, JSON parser, common types         |
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
