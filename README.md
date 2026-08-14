# llmx

A ground-up, **dependency-free** LLM inference runtime. It reads and writes
GGUF v3, runs Q8_0 / F32 transformers on CPU (AVX2 where available), and is
structured so more formats, quantizations, backends, and even multi-device /
multi-node serving can be added later without touching the core.

No external libraries. No CUDA, no ONNX Runtime — just C++ and your CPU.

## Status

llmx is a young runtime. Today it runs **Qwen3-style** models quantized to
**Q8_0** (plus F32) on CPU, with a GPT-2 byte-level BPE tokenizer and a
Jinja2-subset chat-template renderer. See `docs/STATUS.md` for exactly what's
done and what's in flight.

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

## Quick start

```
llmx.exe generate model.gguf "The capital of France is" -n 64
llmx.exe chat      model.gguf --system "You are a terse assistant."
llmx.exe perplexity model.gguf "The quick brown fox jumps over the lazy dog."
```

Run `llmx.exe` with no arguments for the full command list, or see
`docs/USAGE.md` for the complete command reference. A real Q8_0 Qwen3 model and
the wikitext corpus for manual verification are documented in `docs/ASSETS.md`.

## Test

All tests generate their own synthetic fixtures — no real models needed:

```
python tests/run_tests.py
```

- **Round-trip**: quantize / dequantize a random Q8_0 model, assert max error
  within a Q8_0-appropriate bound.
- **Perf**: time the Q8_0 matmul / RMSNorm / RoPE hot paths and end-to-end
  prefill/decode TPS, asserting generous floors so catastrophic regressions fail
  loudly without being flaky.
- **Tokenizer**: encode/decode round-trips incl. unicode and special tokens.

## Documentation

| File                        | What it is                                    |
|-----------------------------|-----------------------------------------------|
| `docs/ARCHITECTURE.md`      | Layer diagram and dependency rules            |
| `docs/USAGE.md`             | Full command reference                        |
| `docs/ASSETS.md`            | Real models and corpora for manual verification|
| `docs/ROADMAP.md`           | The stable long-term plan                     |
| `docs/STATUS.md`            | Living tracker of what's done / in flight     |
| `docs/src/`                 | Per-file docs, linked from ARCHITECTURE       |
| `AGENTS.md`                 | Guidance for AI agents working in this repo   |

## Design at a glance

```
cli > inference > model > backends > tokenizer > format > quant > core
```

Each layer depends only on the layers below it — nothing below the model layer
knows what the model is, nothing below the format layer knows what a file is.
That's what keeps every dimension (formats, quantizations, backends, devices)
independently replaceable. See `docs/ARCHITECTURE.md`.
