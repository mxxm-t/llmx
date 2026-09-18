# AGENTS.md

Guidance for AI agents (and humans) working in this repo. Read this before
making changes.

## What this is

**llmx** — a ground-up, dependency-free LLM inference runtime. It reads/writes
GGUF v3, runs Q8_0 / F32 Qwen3-style transformers on CPU (AVX2 where available),
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
cmake --build build
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

Run the full suite (all generate their own fixtures, no real models needed):
```
python tests/run_tests.py
```

- **Round-trip** (`tests/roundtrip.py`): build a random Q8_0 model, quantize,
  dequantize, assert max error below a Q8_0-appropriate bound. Regression gate
  for `quant/` + `format/`.
- **Perf** (`tests/perf.py`): time matmul / RMSNorm / RoPE hot paths and print
  throughput, so perf-first changes can be checked for regressions. Assert a
  generous floor so catastrophic slowdowns fail loudly without being flaky.
- **Tokenizer** (`tests/tokenizer.py`): encode/decode round-trips incl. unicode
  and special tokens.
- **Baseline** (`tests/baseline.py`): the only test with an EXTERNAL ground
  truth. Compares llmx against golden fixtures generated once from the HF
  reference by `tools/gen_baseline.py` and committed to `tests/data/`. Needs a
  real model, so it SKIPS when none is on disk; point it at one with
  `LLMX_BASELINE_GGUF`. Regenerating fixtures needs `tokenizers` and
  `huggingface_hub`; RUNNING the suite needs neither, and never needs torch.

The two project gates are external and are defined in `docs/ROADMAP.md` #8:
**correctness is the HF reference**, and **performance must be at least
mx-llama.cpp** on the same model, quant, prompt and hardware. Do not substitute
a self-consistency check for either. llmx passed a fully green suite while eight
correctness bugs were live, because every test compared llmx against llmx.

When you change a hot path, run `tests/perf.py` and note the before/after in the
commit message. The lossless correctness gate (path-controlled perplexity on a
real model) is tracked in `docs/ROADMAP.md`.

Real models and corpora for manual verification (the Qwen3-8B Q8_0 model, the
wikitext test set) are documented in `docs/ASSETS.md`.

## Architecture

See `docs/ARCHITECTURE.md` for the layer diagram and rules. The rule that
matters: **each layer depends only on the layers below it** —
`cli > inference > model > backends > tokenizer > format > quant > core`.

| Directory    | Contents                                        |
|--------------|-------------------------------------------------|
| `core/`      | fp16 <-> f32, JSON parser, common types         |
| `quant/`     | QuantType registry + Q8_0 kernels              |
| `format/`    | ModelFormat interface + GGUF v3 impl           |
| `tokenizer/` | GPT-2 byte-level BPE                           |
| `model/`     | Qwen3 config + forward pass, KV cache          |
| `backends/`  | Backend interface + cpu/ (AVX2) impl           |
| `inference/` | sampler, generate, chat template renderer      |
| `cli/`       | thin argument parsing + dispatch               |

## Starting a feature

A fresh agent (or human) can jump straight into a feature by reading, in order:
1. `AGENTS.md` — this file: what the project is, how to build and verify.
2. `docs/ARCHITECTURE.md` — the layers and the dependency rule.
3. `docs/ROADMAP.md` — the stable plan; pick or confirm the feature there.
4. `docs/STATUS.md` — what is already in flight and where each feature stands.

When you start (or pick up) a feature:
- Open a new per-feature block in `docs/STATUS.md` (or update the existing one)
  **before** writing code: **Goal / Done / Left / Gotchas**. That block is what
  lets the next agent pick the feature back up with a "continue feature X"
  prompt, so keep it current.
- Don't invent new directions — follow the roadmap. When the feature ships,
  delete its block and mark the row `Done` in the STATUS table.

## Checkpoints

Update `docs/STATUS.md` and commit at each meaningful checkpoint — at minimum
when a feature, a milestone, or a discrete chunk of work is complete. Each
commit should leave `docs/STATUS.md` accurate: `Done`/`Left` reflect reality,
the build passes, and tests are green. A fresh agent should be able to read
STATUS.md and resume exactly where the last commit left off.

## Build-time vs runtime

- **Backends** are the only compile-time concern (GPU SDKs are heavy). Gated by
  `LLMX_HAS_BACKEND_*` in `src/config.hpp` (see `cmake/llmx-config.hpp.in`).
- **Model architectures** are compiled in, selected at runtime from metadata.
- **Split mode / node count** are runtime params, not build options. See
  `docs/ROADMAP.md`.

## Conventions

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
device execution model refactor in `docs/ROADMAP.md` #4a — device buffers,
resident activations, attention moved into the backend, async submit/sync —
before any vendor backend is worth writing. Don't pick up "add the ROCm
backend" expecting the "one file + one registry entry" experience Q4_0 had.
