# llmx — Architecture

**llmx** is a ground-up, dependency-free LLM inference runtime. It reads and
writes GGUF v3, runs Q8_0 / F32 transformers on CPU (AVX2 where available), and
is structured so more formats, quantizations, backends, and even multi-device /
multi-node serving can be added without touching the core.

## Layer diagram

```
cli/           argument parsing, command dispatch, usage text
   |
   v
inference/     sampler (RNG + top-k/top-p/temp/penalty), generate loop,
               chat template rendering, perplexity driver
   |
   v
model/         in-memory Model + Tensor + KV cache, architecture registry,
               Qwen3 forward graph (arch_qwen)
   |
   v
backends/      Backend interface (matmul / RMSNorm / RoPE), cpu/ impl
   |
   v
tokenizer/     GPT-2 byte-level BPE (encode / decode / specials)
   |
   v
format/        ModelFormat interface; gguf/ is the first implementation
   |
   v
quant/         QuantType registry; Q8_0 block quant / dequant kernels
   |
   v
core/          fp16 <-> f32, minimal JSON parser, common types
```

The rule is: **each layer depends only on the layers below it.** Nothing below
the model layer knows what the model is; nothing below the format layer knows
what a file is. That is what makes each dimension independently replaceable.

## What lives where

| Directory       | Contents                                                              |
|-----------------|-----------------------------------------------------------------------|
| `core/`         | `fp16.hpp` (half <-> float), `json.hpp` (recursive-descent parser)    |
| `quant/`        | `quant.hpp` (type registry), `q8_0.hpp` kernels (moved from main)     |
| `format/`       | `format.hpp` (ModelFormat interface), `gguf.hpp` (GGUF v3 + adapter)  |
| `tokenizer/`    | `tokenizer.hpp` (GPT-2 BPE)                                           |
| `model/`        | `arch_qwen.hpp` (Qwen3 config + forward pass, KV cache)               |
| `backends/`     | `backend.hpp` (interface), `cpu/cpu_backend.hpp` (AVX2 impl)          |
| `inference/`    | `sampler.hpp`, `generate.hpp`, `chat.hpp` (Jinja2-subset renderer)    |
| `cli/`          | `main.cpp` (thin dispatcher)                                          |

## Build-time vs runtime

- **Backends** are the *only* compile-time concern: GPU backends pull in heavy
  SDKs, so they are opt-in via `LLMX_HAS_BACKEND_*` in `config.hpp`. CPU is
  always on (no external deps).
- **Model architectures** are compiled in and selected at runtime from the
  model file's metadata. All are on, always.
- **Split mode and node count** are runtime parameters chosen at launch, not
  build options. See `ROADMAP.md`.

## Multi-device / multi-node design notes

The `backend::Backend` interface is deliberately device-agnostic. A model can
later be split across several Backends — one per device, or per cluster node —
using strategies at the model layer (per-layer, per-tensor, per-row). None of
that is implemented yet; the interface simply does not preclude it. See
`ROADMAP.md` for the plan.
