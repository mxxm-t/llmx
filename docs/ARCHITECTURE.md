# llmx — Architecture

**llmx** is a ground-up, dependency-free LLM inference runtime. It reads and
writes GGUF v3, runs Q8_0 / Q4_0 / Q4_1 / Q6_K / F32 transformers on CPU (AVX2
where available), and
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
backends/      Backend interface (type-generic matmul / RMSNorm / RoPE /
               parallel_for), cpu/ impl
   |
   v
tokenizer/     byte-level BPE, Qwen2/Qwen3 pretokenizer (encode / decode)
   |
   v
format/        ModelFormat interface; gguf/ is the first implementation
   |
   v
quant/         QuantType registry; Q8_0 / Q4_0 / Q4_1 / Q6_K kernels
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
| `quant/`        | `quant.hpp` (type registry + all block kernels)                       |
| `format/`       | `format.hpp` (ModelFormat interface), `gguf.hpp` (GGUF v3)            |
| `tokenizer/`    | `tokenizer.hpp` (byte-level BPE, Qwen2/Qwen3 pretokenizer)             |
| `model/`        | `arch_qwen.hpp` (Qwen3 config + forward pass, KV cache)               |
| `backends/`     | `backend.hpp` (interface), `cpu/cpu_backend.hpp` (AVX2 impl)          |
| `inference/`    | `sampler.hpp`, `generate.hpp`, `chat.hpp` (Jinja2-subset renderer)    |
| `cli/`          | `main.cpp` (thin dispatcher)                                          |

Per-file documentation lives in `docs/src/` — one page per source file, covering
what each header does, its public surface, and its place in the layering. See
`docs/src/cli-main.md` for the CLI entry points and `docs/USAGE.md` for the
command reference.

## Build-time vs runtime

- **Backends** are the *only* compile-time concern: GPU backends pull in heavy
  SDKs, so they are opt-in via `LLMX_HAS_BACKEND_*` in `config.hpp`. CPU is
  always on (no external deps).
- **Model architectures** are compiled in and selected at runtime from the
  model file's metadata. All are on, always.
- **Split mode and node count** are runtime parameters chosen at launch, not
  build options. See `ROADMAP.md`.

## Multi-device / multi-node design notes

The `backend::Backend` interface is device-agnostic in *shape* — nothing in it
names a vendor — but it is host-pointer based today: every call takes raw host
pointers and returns synchronously, so a backend cannot own device memory or
keep activations resident. That is a prerequisite, not a detail. See
`ROADMAP.md` #4a (device execution model), which has to land before any GPU
backend is worth writing.

Once it has, a model can be split across several Backends — one per device, or
per cluster node — using strategies at the model layer (per-layer, per-tensor,
per-row). None of that is implemented yet. See `ROADMAP.md` for the plan.
