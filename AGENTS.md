# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** - a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs quantized or F32 Qwen3-style transformers on CPU (AVX2 where available),
and is structured so formats, quantizations, backends, and multi-device / cluster
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
The build dir's `generated/config.hpp` is produced from
`cmake/llmx-config.hpp.in`; the checked-in `src/config.hpp` is the fallback used
by the plain `build.bat` path. Keep the two in sync when you add build knobs.

## Principles

- **Performance first.** llmx is a *runtime*: a slow-but-correct implementation
  is not enough. Every change to a hot path (quantized matmul, RMSNorm, RoPE,
  attention, KV cache) should state the perf impact and be benchmarked, not just
  verified for correctness. When correctness and speed trade off, prefer the
  fast path and prove it is lossless (see `docs/ROADMAP.md` correctness gate).
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

- **A knob a user would set** is a flag: `--threads`, `--ubatch`. Name it after
  the equivalent in llama.cpp where one exists, so the vocabulary carries over
  (`--ubatch` is `n_ubatch` / `-ub`; llmx has no `n_batch`, see below).
- **A value the code can determine** is not a knob at all. Row blocking in the
  CPU matmul was briefly `LLMX_ROW_BLOCK`; it is now `DOT_ROWS`, the width the
  fused kernel handles, because measurement showed the knee follows the kernel
  and not the machine. A constant that is a property of the code does not
  belong in the environment.
- **Temporary A/B knobs get deleted** once they have answered their question.
  Both `LLMX_ROW_BLOCK` and `LLMX_ROW_BLOCK_BYTES` existed only to find a
  number and were removed with the finding recorded in `docs/STATUS.md`.
- **The one environment variable is for tests**: `LLMX_BASELINE_GGUF` points
  `tests/baseline.py` at a fixture model. That is test configuration, not
  runtime configuration, and it never reaches the binary.

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

Run the native backend, chat-template, KV storage, loader and streaming checks
after a CMake build:
```
ctest --test-dir build -C Release --output-on-failure
```

`backend-group` checks mixed types, uneven rows, batches, thread counts,
output boundaries and fallback behavior against separate calls and double dots.
It also checks every finite f16 scale against signed Q8 weight extremes using
one-hot inputs with exact expected products, and the three-column prefill
reduction against ordered scalar FMA across dimension tails and unaligned
inputs. The independent HF fixtures below remain the external correctness gate.

`backend-errors` injects task and startup-allocation failures, checks completion
before error propagation, and exercises pool reuse and thread reconfiguration.
It does not establish recovery of partially executed model sessions.

Run the Python suite (synthetic fixtures are generated locally; real-model HF
checks skip when their models are absent):
```
python tests/run_tests.py
```

For a CMake build, pass `--exe <path-to-built-llmx>`. CI uses
`--no-perf-floor` for shared runners and `--require-baseline` in its real-model
job so missing fixtures fail. Local performance floors remain enabled by
default. See `docs/CI.md` for workflow coverage and reproduction commands.

- **Round-trip** (`tests/roundtrip.py`): build a random Q8_0 model, quantize,
  dequantize, assert max error below a Q8_0-appropriate bound. Regression gate
  for `quant/` + `format/`.
- **Perf** (`tests/perf.py`): time matmul / RMSNorm / RoPE hot paths and print
  throughput, so perf-first changes can be checked for regressions. Assert a
  generous floor so catastrophic slowdowns fail loudly without being flaky.
- **Tokenizer** (`tests/tokenizer.py`): encode/decode round-trips incl. unicode
  and special tokens.
- **Chat** (`tests/chat.py`): follow-up replies against independent HF/Jinja2
  goldens, including changed prefixes, stop/EOS and token-limit endings.
  CTest also runs `chat-template`, comparing the real Qwen template against
  Jinja2-rendered conversation fixtures. Regenerate these with
  `python tools/gen_chat_baseline.py`; running them needs no external libraries.
- **Thread controls** (`tests/threads.py`): actual auto/explicit phase counts,
  restoration after prefill, follow-up chat and HF-golden replies.
- **Loading and streaming** (CTest `load-progress`, `generation-stream`,
  `cli-output`):
  completed-byte reporting, truncated reads, callback failures, early text
  delivery, split UTF-8 bytes, legacy filtering and stop/EOS accounting.
- **KV storage** (`tests/kv_cache.cpp`, CTest `kv-cache`): distinct
  layer/head/position/lane values across growth, retained-capacity reset and
  invalid extents. This storage oracle supplements the independent HF gate.
- **F32** (`tests/f32.py`): deterministic small-model weights with full logits
  and windowed NLL generated independently by HF. Covers tied/untied weights,
  odd dimensions, batch tails and threads without downloading a model.
- **Baseline** (`tests/baseline.py`): real-model EXTERNAL ground truth.
  Compares llmx against golden fixtures generated once from the HF
  reference by `tools/gen_baseline.py` and committed to `tests/data/`. Needs a
  real model, so it SKIPS when none is on disk; point it at one with
  `LLMX_BASELINE_GGUF`. Regenerating tokenizer fixtures needs `tokenizers` and
  `huggingface_hub`; numerical fixtures also need `torch` and `transformers`.
  RUNNING the suite needs none of these packages.
- **Reference generator** (`tests/reference_generator.py`): standard-library
  checks for pinned reference selection, separate alternate-model output and
  forwarding the revision/float32/eager settings to the HF loaders. Actual
  reference generation and model correctness remain separate checks.
- **Reference consumer** (`tests/reference_consumer.py`): standard-library
  rejection tests for changed 8B fixtures, damaged logits/PPL, wrong model
  identity and failed launches. This is the ordinary suite's eleventh component;
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
were frozen before the 8B comparison. See `docs/ASSETS.md` for provenance and
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
`cli > inference > model > backends > tokenizer > format > quant > core`.

| Directory    | Contents                                        |
|--------------|-------------------------------------------------|
| `core/`      | fp16 <-> f32, JSON parser, common types         |
| `quant/`     | QuantType registry + Q8_0/Q4_0/Q4_1/Q4_K/Q5_K/Q6_K kernels |
| `format/`    | ModelFormat interface + GGUF v3 impl           |
| `tokenizer/` | byte-level BPE, Qwen2/Qwen3 pretokenizer       |
| `model/`     | Qwen3 config + forward pass, KV cache          |
| `backends/`  | Backend interface + cpu/ (AVX2) impl; one worker pool |
| `inference/` | sampler, generate, perplexity, chat template renderer      |
| `cli/`       | thin argument parsing + dispatch               |

## Starting a feature

A fresh agent (or human) can jump straight into a feature by reading, in order:
1. `AGENTS.md` - this file: what the project is, how to build and verify.
2. `docs/ARCHITECTURE.md` - the layers and the dependency rule.
3. `docs/ROADMAP.md` - the stable plan; pick or confirm the feature there.
4. `docs/STATUS.md` - what is already in flight and where each feature stands.

When you start (or pick up) a feature:
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
  The names/options exist, but GPU backends are not implemented.
- **Model architectures** are planned to be compiled in and selected from
  metadata; today only dense Qwen3 is implemented.
- **Split mode / node count** are planned runtime params, not implemented flags. See
  `docs/ROADMAP.md`.

## Conventions

- mx-llama.cpp may be inspected as a reference, but do not copy its source.
  Implement llmx changes independently in this project's architecture and style.

- Use ASCII characters in code, comments, documentation and commit messages.
  Preserve Unicode test coverage using escaped literals and fixture data.

- Header-only for now (everything is `#pragma once` + `inline`), compiled via
  `src/cli/main.cpp`. If we add `.cpp` files later, keep one TU per logical unit.
- Include paths are relative to `src/` root: `#include "format/gguf.hpp"`.
- No comments in code unless they explain a non-obvious decision or algorithm
  (e.g. the fp16 rounding, the GGUF padding rules, the AVX2 dequant+FMA path).
- Cross-platform (Windows / Linux / macOS): guard MSVC-vs-GCC intrinsics with
  `#if defined(_MSC_VER)`; use `<intrin.h>`/`<cpuid.h>` appropriately.
- Don't overengineer. Add a seam (interface) only when a second implementation
  is actually on the roadmap. Empty stubs are discouraged.

## Roadmap

`docs/ROADMAP.md` lists: more quant formats, more architectures, more formats,
backends (ROCm first-class, Vulkan for portability), multi-device split,
multi-node cluster, a multi-user server, and Hugging Face integration (`llmx
pull` plus reading what the Hub actually hosts). Follow the roadmap before
inventing new directions.

GPU backends are **not** drop-in the way a quant type is. They require the
device execution model refactor in `docs/ROADMAP.md` #4a - device buffers,
resident activations, attention moved into the backend, async submit/sync -
before any vendor backend is worth writing. Don't pick up "add the ROCm
backend" expecting the "one file + one registry entry" experience Q4_0 had.
