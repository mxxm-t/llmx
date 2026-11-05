# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** - a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs quantized or F32 Qwen3, Qwen3-MoE, the Qwen 3.5 hybrid (qwen35, qwen35moe) and Qwen3.8-Flash-Next (qwen4exp) transformers on x86 CPU with AVX2/FMA/F16C
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
- **Never wait; keep the pipeline busy.** The pipeline is the whole path from input to output: a request arriving, its tokens, loading and reading weights, every pass on every device, sampling, and the text streamed back.
  Nothing on it waits behind anything else while work exists for it: no device behind the host, no stage behind another, no request behind a step that could run beside it.
  The host prepares and queues work ahead of time, as asynchronous submissions with device-side dependencies; where that is not possible, a thread does the work beside the devices.
  A host round trip inside a pass is allowed only when handing the work to the host lets a device keep going where that work would otherwise stall it, and then it overlaps device work rather than sitting on the critical path.
  A change that adds a wait states why no alternative exists; device idle time, not host time, is the measure.
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
  Every job checks this (Dead code and stale docs, below).
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
  number and were removed with the finding recorded in `docs/STATUS-2026-09.md`.
- **The runtime credential exception is `HF_TOKEN`** for `llmx pull` gated-repo
  access. It is not a tuning knob and must not appear in child argv or logs.
- **Test configuration**: `LLMX_BASELINE_GGUF` points `tests/baseline.py` at a fixture model; `LLMX_DEVICE`, set by `run_tests.py --device`, appends `--device` to every command that takes it so the suite runs on a device backend, except where a component names its own devices (`threads` names the CPU, whose thread counts it checks, and `split` names the CPU backends its tool splits over); `LLMX_CACHE_TYPE`, set by `run_tests.py --cache-type`, appends `--cache-type-k` and `--cache-type-v` the same way so the HF gate runs with a chosen cache type; `LLMX_LAYER_SHARES`, set by `run_tests.py --layer-shares`, appends `--layer-shares` so a device list is tested at a split the fit would not choose; `LLMX_TENSOR_WIDTH`, set by `run_tests.py --tensor-width`, appends `--tensor-width` so a device list runs as tensor groups, and `tensor-split` then captures on that list (docs/TENSOR-SPLIT.md); the components whose synthetic fixtures a group refuses, an odd feed-forward width or one head (`perplexity`, `f32`, `int8`, `moe`, `mxfp4`, `qwen35`, `drafters` and `shards`), skip under a width above 1 (`common.tensor_width_skip`), while `server` and `chat` run on a model of the tensor-split fixtures' shape there.
  The synthetic bench (`bench` without `--model`) takes only `--device` and `--threads`, so it gets neither shares, cache types nor a load mode, and `split` gives its tool equal shares and the cache types it names itself.
  The runtime stores f16 by default, so the components that check exact f32 arithmetic against independent fixtures (`f32`, `moe`, `qwen35`, `shards`, `server`) ask for f32 sides themselves and skip when `--cache-type` asks for another type (`common.f32_cache_skip`).
  `LLMX_LOAD_MODE`, set by `run_tests.py --load-mode`, appends `--load-mode` to every model command the same way, so the suite runs with the weights read in any mode.
  `LLMX_DTYPE`, set by `run_tests.py --dtype`, appends `--dtype` to model commands that do not name it themselves; it never reaches the synthetic benchmark or metadata-only commands. The split component passes it through its tool's optional dtype argument to both placements.
  That is test configuration, not runtime configuration, and it reaches the binary only as the flags.

If you add a flag, add it to `docs/USAGE.md` and to `print_usage` in the same
change, or it does not exist as far as a user is concerned.

## Verify

The round-trip component (`tests/roundtrip.py`, described in `docs/TESTS.md`) checks the format and quantization paths.
It builds a random F32 model, quantizes it to Q8_0 and Q4_0 through the CLI, dequantizes it back and bounds the error, and decodes the written blocks with the spec decoders of `tests/spec_decode.py`, written from the format descriptions, rather than with llmx's own reader.
Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K and Q6_K, which `quantize` does not write, and Q8_0 and Q4_0 under negative scales, which their quantizers never write, are decoded the same way from raw blocks the test writes, chosen so every scale, min, high bit and nibble reaches a decoded value, and `dequantize` must match bit for bit, the sign of zero included. Q2_K and Q3_K are read-only and have no writers.
Under a negative scale Q4_0's nibble 8 must decode as -0, as the format's d*(nibble - 8) gives it (`docs/ASSETS.md`).
Run it alone against the root `llmx.exe` that `build.bat` writes, or as part of the suite for a CMake build:
```
python tests/roundtrip.py
python tests/run_tests.py --exe build/Release/llmx.exe
```
`llmx.exe info FILE` lists the metadata and tensors of a written GGUF.
On Windows with `LongPathsEnabled` set, the round-trip component also reads a GGUF through a path longer than 260 characters and checks identical decoded payload bytes and the preserved full model path in its metadata; it records a skip of this case if the policy is disabled or unavailable. Both MSVC build routes embed the shared `cmake/windows.manifest`; no test or runtime enables the system policy.

## Tests

What each test checks is in `docs/TESTS.md`: every CTest and suite component is described there, and a new one is described there in the change that adds it (the `docs` component's `test-name` check holds that).

Vulkan `--isa` count checks follow the declared build shape and repair property,
as described in `docs/TESTS.md`. A compiler may retain a repair loop instead of
copying its body for every row. Such builds retain the operation-kind and
same-row count checks, with cross-row coverage reported separately and held by
numerical repair tests. Do not change shader arithmetic to satisfy a static
count equation that assumes those loops were unrolled.

Run the native backend, model, chat-template, KV storage, loader and streaming checks
after a CMake build:
```
ctest --test-dir build -C Release --output-on-failure
```

Run the Python suite (synthetic fixtures are generated locally; real-model HF
checks skip when their models are absent):
```
python tests/run_tests.py
```

For a CMake build, pass `--exe <path-to-built-llmx>`.
`--only` runs just the components it names, comma separated (`--only baseline`, `--only split,server`), and refuses a name the suite does not have.
CI uses `--no-perf-floor` for shared runners, `--require-tools` on every CMake build it runs the suite on so a tool missing beside the executable, or numpy for raw-blocks, fails rather than skips (the `build.bat` binary has no tools beside it), and `--require-baseline` in its real-model job so missing fixtures fail.
That job also runs `--only baseline` with `--cache-type f32`, `llmx-split-check` on the Q8_0 over two CPU backends and `tools/server_mix_check.py` on the Q8_0; the Vulkan job runs the suite with `--device cpu` on the Vulkan-enabled binary.
The TSan job builds `http` and `server-passes-cpu` with `-fsanitize=thread` and runs them with address randomization off (`setarch -R`), which ThreadSanitizer needs on the runner's kernel.
Local performance floors remain enabled by default. `--require-device-types Q8_0,Q4_K` makes a selected backend's refusal of any named weight type fail rather than skip through `common.device_lacks_kernel`; names are the case-sensitive storage names in `tests/spec_decode.py`.
Types not named keep their ordinary skips, and naming a type does not prove it was exercised, require a missing model, or change native CTest's driverless skips.
The `reference-consumer` component checks early and backend refusals, unrelated failures, required and optional types, runner exit status and invalid selections.
See `docs/CI.md` for workflow coverage and reproduction commands.

The real-model checks run by hand, the optional 8B check and the qwen35 files' checks, are described in `docs/TESTS.md` (`docs/TESTS.md`, Real-model checks by hand):

```
python -X utf8 tests/baseline_8b.py --exe build/Release/llmx.exe --model path/to/Qwen3-8B-Q8_0.gguf --output-dir hf-8b-review
python -X utf8 tests/baseline_qwen35.py --exe build/llmx --model path/to/Qwen3.5-4B-Q4_K_M.gguf --output-dir qwen35-review --context 4096
```

`tests/baseline_layered.py` runs the same checks on the layered qwen35 reference's files.

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
`cli > server > inference > model > backends > tokenizer > format > quant > core`, with `image/` beside `tokenizer/` on `core/` alone;
`server/` uses the inference layer and adds only scheduling and transport.

| Directory    | Contents                                        |
|--------------|-------------------------------------------------|
| `core/`      | fp16 and bf16 <-> f32, JSON parser, UTF-8, file hashes, the host memory a process can still take and owned pages, comma-separated lists, the CPUs a process may use and the cgroups its limits are read from, threads that run a job over its indices |
| `hub/`       | CLI acquisition path: Hub metadata, curl HTTPS and verified multi-stream cache |
| `quant/`     | storage metadata and checked row sizes in types.hpp, decoder registry, F16/BF16 widening and the Q, K, IQ, MXFP4 and TQ family decoders |
| `format/`    | GGUF v3 reader/writer (headers, then mapping, then reading in), file spans, a file read at offsets, output files published whole, raw F32 tensors to and from GGUF |
| `tokenizer/` | byte-level BPE, Qwen2/Qwen3/Qwen3.5 pretokenizer |
| `image/`     | image files decoded, oriented and resized for a vision encoder, beside the tokenizer; depends on core alone |
| `model/`     | runtime (sequences and their histories, passes, stages, the arena, placement), one module per architecture under `arch/` chosen by the registry (qwen3 and qwen3moe, qwen35 and qwen35moe, qwen4exp) with their shared graph pieces, KV cache and recurrent state slots, layer split over devices, tensor shards |
| `backends/`  | Backend interface with cpu/ (AVX2), vulkan/ and optional Linux rocm/ implementations; one CPU worker pool; `device_profile.hpp`, the device numbers a GPU backend shapes its kernels by |
| `inference/` | model loading, sampler, log-probabilities, generate, perplexity, chat template renderer |
| `server/`    | multi-user server (`docs/SERVER.md`): HTTP layer, scheduler, the cache behind one owner, its holders, room, ranking, parking, jobs and tiers (`cache.hpp`), its policy core and sampling threads, routes |
| `cli/`       | thin argument parsing + dispatch               |

## Starting a feature

A fresh agent (or human) can jump straight into a feature by reading, in order:
1. `AGENTS.md` - this file: what the project is, how to build and verify.
2. `docs/ARCHITECTURE.md` - the layers and the dependency rule.
3. `docs/ROADMAP.md` - the stable plan; pick or confirm the feature there.
4. `docs/STATUS.md` - what is already in flight and where each feature stands.

When you start (or pick up) a feature:
- A new model architecture is a module under `src/model/arch/`, built in the order and held to the gates `docs/ADDING-AN-ARCHITECTURE.md` gives.
- Keep each independent feature on its own branch based on current main.
  Do not base unrelated work on another unmerged feature. If a dependency is
  necessary, name it in STATUS and keep the dependent change separate.
  A feature's own documentation ships in the same commit as the code it
  describes; a documentation correction unrelated to the feature goes on a
  branch of its own.
- Open a new per-feature block in `docs/STATUS.md` (or update the existing one)
  **before** writing code: **Goal / Done / Left / Gotchas**. That block is what
  lets the next agent pick the feature back up with a "continue feature X"
  prompt, so keep it current.
- A feature lands with a test the hosted workflow runs, or its STATUS block names the hand check that covers it and says why no hosted runner can run it.
- A bug fix lands its failing test first: a commit that adds the test, failing on the unfixed code, then the fix that makes it pass.
- A branch lands as at most two commits: the failing test, where it has one, then the change with its tests, its documentation and its STATUS entry.
  Its working commits, review fixes and checkpoints are squashed into those before it lands, so main's history holds what changed and why, not how the branch got there; a docs-only branch lands as one commit.
- Don't invent new directions - follow the roadmap. When the feature ships,
  delete its block and mark the row `Done` in the STATUS table.

## Merge gates

Choose checks by the changed paths and behavior, not by the number or metadata of commits. Do not repeat the full gates during development: run focused checks for the changed coverage, then the applicable landing checks once the change is complete. After a failure or fix, rerun the checks that cover the failure and the changed code.

During development a step runs only the build, the focused test of what it changed and short microbenchmarks on the real shapes, profiling before it changes a hot path.
CTest, the suites, device suites, long benchmarks, the Radeon VII check, gate pushes and the hosted run wait for the head that lands and run there once.
A focused test may run on a second device or driver when the step's risk is there, such as a kernel whose sums a driver could contract differently.
Fewer checks never loosen the architecture: each step states its layer and the one owner it changes, adds no second path, and is read by the reviewer before the next step starts.

- Static dashboard presentation changes confined to HTML, CSS and browser JavaScript, with no collector, health API, server or runtime change: run syntax checks and focused browser checks for the changed states and responsive layout, plus the `docs` and `dead-code` components. Do not require HF, CPU/device numerical correctness or inference performance gates, or wait for unrelated hosted inference jobs. A workflow may still start automatically; that does not make its unrelated jobs a gate for this scope.
- Docs only: the suite's `docs` and `dead-code` components on the tree that lands; unrelated hosted inference jobs are not waited for.
- Tests or tools only: rerun the tests and checks affected by the change; apply the relevant hosted checks for its scope, without requiring unrelated inference gates for static tools.
- Standalone Python collector changes: focused Python tests for collection, schema and security behavior, plus protocol integration checks when the health contract changes. A collector-only change does not require runtime CTest, HF numerical or GPU inference gates.
- CLI, server, health API, scheduler or other llmx host/runtime logic: CTest, the CPU suite, Qwen3-0.6B byte identity against main, and the hosted run; host code specific to Windows also runs CTest and the suite on Windows.
  A change to the server, the scheduler or the suite's `server` component also runs that component on an MI50 (`--device vulkan:0 --only server`): its checks that send signals do not run on Windows, so the Radeon VII does not cover them, and no hosted job has a device.
- Model, kernel, device, loader or placement code: byte identity against main on the CPU and a device for the models the change reaches, the suite on a device, the Radeon VII check on Windows for device code, one timing round against main, and the hosted run.
  A change of numerics also records its error against the HF reference and the headroom to its bound.
- Carry forward prior evidence only when the checked source, tests, fixtures, configuration and relevant build inputs are unchanged. Record the exact tested revision and binary identity, then state the intervening delta; rerun checks when that delta changes their coverage or inputs. Commit hashes alone do not invalidate evidence.

A branch lands when its applicable checks pass for the tree that lands and main has not moved since: main is fast-forwarded to it.
Branches ready at the same time land together: stacked in landing order, each branch's checks run once at the stacked head with the reviews already given, and main is fast-forwarded to that head.
A branch rebased without a conflict over commits that touch none of its checks' inputs reruns its builds and CTest, and takes the hosted run of the head that lands. The hosted workflow policy remains as listed above for runtime and numerical changes; static UI-only changes do not wait for unrelated hosted inference jobs.
The landing commit's own STATUS entry is the record: it gives what the applicable gates measured and says the change lands by fast-forward, and the merge adds no commit of its own. Where a hosted run is part of the applicable gate, its commit hash identifies it on GitHub.
Comparisons against the reference runtimes belong to the phase's final gate, not to each branch.

### Traps the gates exist for

Each of these let a wrong result or a wrong measurement through here once.
They are part of the gates, not advice.

- **A check counts only for the path it reaches.** A tiny model below a fast path's size threshold runs another kernel, and a green result says nothing about the path it skipped. A numerics change names the path each check exercised and proves it ran, through a counter, a log line or a forced-path unit check, and exercises the threshold's edges.
- **Short checks do not bound long ones.** A device prompt path with 8-bit activations passed every tiny fixture and the 512-token HF windows, then missed 6 to 8 of 512 top-1 tokens on a 9B model at 16k tokens against HF on the file's own weights. A change of prompt or decode numerics on a device runs the long-context checks, the raw and the chat-template 16k sequences against file-exact HF, before it merges.
- **A bound follows the precision a path computes in, never the weight type.** F32 bounds hold F32 paths. A path in another precision is held to a budget calibrated for that precision: a reference run with the exact weights, activations rounded to that precision and F32 sums, measured against file-exact HF over the dense, tied and MoE fixtures and extra seeds, then frozen with its headroom stated before any candidate is rerun. See `docs/PRECISION.md` for the sole user-approved, exact IQ4_XS split-MoE F32-logit fixture exception and its unchanged limits. A budget is never set just above a failing candidate, and the real-model bounds are not widened to let a path through.
- **Correctness is not traded for speed by default.** A faster path in a lower precision than the rule gives (8-bit activations, say) is an opt-in with its own bound, never the default, and never decides a merge of the default path. A fallback is never narrower than the precision asked for, and says so when taken.
- **Measure what a user runs.** Timing runs at the device's default clocks and power state. Setting performance levels or holding clocks high flatters a kernel and makes its figures incomparable with any other run; every arm, llmx before, llmx after and the reference, runs on the same device in the same environment (Principles, above).
- **An open speed gate is recorded, not waived.** First model support may merge below the reference with its deficits and recovery work recorded. Quant support follows the Vulkan-first sequence in `docs/ROADMAP.md`: meet mx Vulkan before moving to the next quant, then tune all quants against mx ROCm while retaining Vulkan performance. Record ROCm deficits for that second stage; they do not block first-stage quant landings. Neither allowance covers a slowdown on a path main already has, which the timing round against main catches at every merge.
- **A review finding is rechecked before it is acted on.** A reviewer's finding, a person's or a tool's, is checked against the code, then fixed or rejected with the evidence beside it.

## Checkpoints

Update `docs/STATUS.md` and commit at each meaningful checkpoint - at minimum
when a feature, a milestone, or a discrete chunk of work is complete. Each
commit should leave `docs/STATUS.md` accurate: `Done`/`Left` reflect reality,
the build passes, and the focused tests of the change pass. A fresh agent should be able to read
STATUS.md and resume exactly where the last commit left off.
These checkpoints live on the branch; it is squashed as it lands (Starting a feature, above).

A checkpoint during development updates the docs its change touches and its STATUS block.
At the head that lands, review all project Markdown files against the
current code, tests, CLI, build configuration and recorded results. Correct stale
claims and links, distinguish historical measurements from current status, and
record the review in STATUS.md. This includes README, AGENTS, all docs/ pages
and per-source documentation; update only what needs changing.

Build identification is automatic: both supported build paths combine the
explicit release version with Git revision and tracked-dirty state. Update
CMake's project version and the fallback config together for a release; do not
increment versions or create tags merely because a build ran. `--version`
reports the embedded value, with `unknown` for builds without Git metadata.

## Dead code and stale docs

What nothing reaches is removed in the change that leaves it unreached (Principles), and a doc names only what the tree has: a change that removes or renames a path, a flag, a command, a test, a heading or a name in the code corrects every doc that names it.
Every hosted job holds the tree to both, and a finding fails the job:

- The compilers: an unused function of one translation unit, or an unused local, is an error in the CMake builds and in `build.bat`.
- The suite's first two components, `dead-code` and `docs` (`docs/TESTS.md`, Suite components), read the source, the Markdown and the comments in a few seconds.
  The `docs` component's `comment` check fails a comment block of more than 2 sentences or a line of more than 240 characters; what a comment loses that a reader still needs goes to the doc that owns the topic (`docs/src/` page of the file, or the design doc), and a file's line in `tests/data/comment_allowance.txt` goes when the file is swept.
- The Vulkan job also runs `python -X utf8 tests/dead_code.py --linked build-linked`, which builds every target once more to see which functions the product's executables keep.

**A finding is fixed in the change that makes it**: remove the code, or correct the doc.
Only a finding that has to stay is listed, in `tests/data/known_findings.txt`, as `check | file | name | times | reason`, `times` being how often it occurs and the reason saying what it is and what removes it, such as the branch it goes with.
A listed finding that no longer occurs, or occurs another number of times, fails its check too, and so does a line whose check neither component has, so the change that removes one also removes its line, and the list does not rot.
A new line needs a reason a reviewer accepts, not a wish to merge first.
An interface that a later step of an approved plan gives its product caller is such a reason: the line names the plan and the step, as `layer split phase 3, step 2`, and goes in that step.

**What the checks rely on:**
- A struct whose fields mirror a layout something else reads, such as a kernel's structure, is pinned by a `static_assert` with `offsetof` or `sizeof`, which tells the name check its fields are read by position.
- An enum that a value is cast to from an integer has its values reached by arithmetic, so the name check leaves them out; kernel ids are the shader check's.
- A doc's sections: a heading carrying a date opens a record, and a heading saying "planned", "Order of work", "Open questions", "Not chosen", "Out of scope", "Non-goals" or "Migration order", like all of `docs/ROADMAP.md` and `docs/QWEN35.md`, a plan.
  A record's or a plan's names may be gone or not exist yet, so only their links and line numbers are checked.
  In `docs/STATUS.md` only the status table, the active blocks and the working rules are live, and the monthly files beside it, STATUS- and the year and month in the name, each a month's records moved from it, are records throughout; elsewhere a section is live unless a heading above it opens a record or a plan, and live text is held to the tree.
- A line number in a doc names the commit it points into, `vulkan_backend.cpp:581 at 0123abc`, since the line moves with the next edit.
- A reference to a heading of the same doc says where it is, `(Tests, above)`, and one to another doc's names the doc, `(AGENTS.md, Tests)`; a heading's name in prose without either is not seen.
- A source CMake does not build is built by the linked check beside its targets (`HAND_BUILT` in `tests/dead_code.py`); a doc naming a source does not count as building it.
- The tree the checks read is what a commit would hold: the files git tracks and the new ones it does not ignore, or every file of a copy without `.git`.
- A `docs/src` page's title names its file or its directory (`` # `src/core/json.hpp` - minimal JSON parser ``), and each source file is named by exactly one title.

**What no mechanical check sees** is left to the review of the Markdown at the head that lands (Checkpoints, above).
In the docs: whether prose about behaviour, a refusal, a crossover or an exit status is still true; numbers not written beside the constant they are; a counted fact repeated in several docs (the suite's components, the consumer's checks); whether a measurement is current or history; history paragraphs in live text, which move under a dated heading; names of other programs; planned names in live design text; whether a page describes its file as it is now; what a pinned line says; a finished plan still headed "planned"; and one concern described differently in two docs.
In the code: code that runs but serves nothing, paths for inputs or devices that never occur, whether a function only tests call is a probe worth keeping, a virtual function whose name is too common for the override check to tell its calls apart, generality nobody uses (a default never overridden, a template parameter with one instantiation, a seam with one implementation), table rows and enum values no path selects, and a feature implemented twice.

## Build-time vs runtime

- **Backends** are the only compile-time concern (GPU SDKs are heavy). Gated by
  `LLMX_HAS_BACKEND_*` in `src/config.hpp` (see `cmake/llmx-config.hpp.in`).
  Vulkan and Linux ROCm are optional backends. CUDA and SYCL remain planned and get their options with their implementations.
- **Model architectures** are compiled in and selected from metadata by
  `src/model/arch/registry.hpp`, one module per architecture under `src/model/arch/`
  on a runtime they share; today qwen3, qwen3moe, qwen35, qwen35moe and
  qwen4exp. A new one follows
  `docs/ADDING-AN-ARCHITECTURE.md`.
- **Split mode** is runtime configuration: a `--device` list selects the devices,
  and `--tensor-width` groups them for tensor splitting; the default width of one
  gives a layer split (`docs/MULTI-DEVICE.md`). Multi-node execution is planned
  in `docs/ROADMAP.md`.

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
- Cross-platform (Windows / Linux / macOS): guard MSVC-vs-GCC intrinsics with `#if defined(_MSC_VER)`.
  AVX2, FMA and F16C are the build baseline (`docs/BUILD.md`), so the x86 kernels include `<immintrin.h>` and use them with no runtime check; a kernel beyond that set, such as the deferred AVX-512 path (`docs/src/backends-cpu.md`), needs its own runtime dispatch.
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

Q5_1 raw-block checks cover all five-bit codes at every position, independent high-bit lanes and scale/minimum edge pairs. CPU row-class and grouped-matrix checks include Q5_1; Vulkan uses the existing affine row and tile families, extending their packed-code unpacking. The feature's current qualification status is recorded in STATUS.

Q5_0 Vulkan focused checks extend the same backend and offset-range owners: independent 22-byte blocks exercise alternating alignment, all five-bit codes, signed scales and embedding zero signs. F32/BF16/F16 matrix checks include Q5_0 through ordinary, logits, additive, grouped and routed calls, decode-column identity and row classes. Execution witnesses check explicit int8 requests on wider-path devices; native int8 and integer-tile checks run only where the device advertises them. Center codes at 16 before the dot to retain a unit residual beside 65504 under a tiny scale; the existing offset-range bounds must not be relaxed. These checks do not establish HF, sidecar or matched performance qualification.
