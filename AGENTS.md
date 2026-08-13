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
backends (ROCm/CUDA/Vulkan), multi-device split, multi-node cluster, and a
multi-user server. Follow the roadmap before inventing new directions.
