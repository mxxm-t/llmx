# llmx - Architecture

**llmx** is a ground-up, dependency-free LLM inference runtime. It reads and
writes GGUF v3, runs Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / F32 transformers
on x86 CPU with AVX2/FMA/F16C, and
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
model/         Qwen3 Model + KV cache; architecture registry planned,
               Qwen3 forward graph (arch_qwen)
   |
   v
backends/      Backend interface (type-generic matmul / attention / RMSNorm / RoPE /
               parallel_for), cpu/ impl
   |
   v
tokenizer/     byte-level BPE, Qwen2/Qwen3 pretokenizer (encode / decode)
   |
   v
format/        GGUF reader/writer + ModelFormat adapter/open;
               CLI/model still consume GGUFModel directly
   |
   v
quant/         QuantType registry; Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K kernels
   |
   v
core/          fp16 <-> f32, minimal JSON parser, common types
```

The rule is: **each layer depends only on the layers below it.** Nothing below
the model layer knows what the model is; nothing below the format layer knows
what a file is. That is what makes each dimension independently replaceable.

This is the intended dependency rule. The current quant registry imports GGUF
type constants from `format/gguf.hpp`; this existing exception needs resolving
when adding another format. Dense F32 and supported block-quant matrices
share the CPU float dot kernels; F32 rows need no dequantization buffer.

## What lives where

| Directory       | Contents                                                              |
|-----------------|-----------------------------------------------------------------------|
| `core/`         | `fp16.hpp` (half <-> float), `json.hpp` (recursive-descent parser)    |
| `quant/`        | `quant.hpp` (registry + block quants), `k_quants.hpp` (K-quants)                       |
| `format/`       | `format.hpp` (ModelFormat interface), `gguf.hpp` (GGUF v3)            |
| `tokenizer/`    | `tokenizer.hpp` (byte-level BPE, Qwen2/Qwen3 pretokenizer)             |
| `model/`        | `arch_qwen.hpp` (Qwen3 config + forward pass), `host_kv_cache.hpp` (CPU KV storage) |
| `backends/`     | `backend.hpp` (interface), `cpu/cpu_backend.hpp` (AVX2 impl)          |
| `inference/`    | `sampler.hpp`, `generate.hpp`, `perplexity.hpp`, `chat.hpp`    |
| `cli/`          | `main.cpp` (thin dispatcher)                                          |

Per-file documentation lives in `docs/src/` - one page per source file, covering
what each header does, its public surface, and its place in the layering. See
`docs/src/cli-main.md` for the CLI entry points and `docs/USAGE.md` for the
command reference.

## Build-time vs runtime

- **Backends** are the *only* compile-time concern: GPU backends pull in heavy
  SDKs, so they are opt-in via `LLMX_HAS_BACKEND_*` in `config.hpp`. CPU is
  always on (no external deps). The GPU options currently define macros only;
  no vendor implementation is built.
- **Model architectures** will be compiled in and selected from metadata.
  Today the model layer implements dense Qwen3 only.
- **Split mode and node count** are planned runtime parameters, not implemented
  build options or CLI flags. See `ROADMAP.md`.

`--threads` and `--threads-batch` select CPU workers for decode and prefill.
Future GPU backends must keep that meaning for applicable CPU work; GPU launch
dimensions belong to the backend. `--ubatch` is the number of prompt tokens
per forward pass and remains relevant to device execution.

## KV state and concurrent execution

Today `Model` combines a reference to model weights, one sequence's KV cache
and token position, activation scratch, and a backend. It supports one sequence
at a time. Parallel work inside a forward pass does not make concurrent calls
to the same `Model` safe. Backend worker dispatch and attention scratch also
need explicit ownership before concurrent submissions can be supported.

`HostKVCache` centralizes CPU storage, growth and writes. Each head's history
is contiguous; the backend receives its physical stride separately from the
valid sequence length. `Model` retains logical position and reset ownership.
This is a concrete host implementation, not the future device-memory interface.

The planned device and server work (ROADMAP #4a and #7) must preserve these
boundaries:

- Loaded weights are shared read-only. Each sequence owns its logical token
  positions and mutable KV state; resetting or cancelling one sequence must
  not change another sequence's history.
- KV storage owns allocation, growth and layout. The forward graph supplies
  the layer and positions to write; backend attention consumes an explicit
  layout and valid sequence extent. Allocated capacity is not valid history.
- Activation and attention scratch belong to an execution context, or have
  exclusive scheduled use. A shared worker pool alone does not provide safe
  concurrent execution.
- Continuous batching must describe each sequence's positions and causal
  boundaries independently. The current prefill microbatch contains tokens
  from one sequence; it is not a batch of independent users.
- With device execution, backend buffers own physical storage and submission
  completion controls its lifetime. Cache growth, reset and reuse must not
  invalidate storage still used by an in-flight operation. Model-level routing
  determines placement across devices without introducing vendor types into
  sequence state.
- Future prefix reuse may share immutable KV blocks only when the model,
  positions and relevant execution configuration match. Mutable suffixes stay
  private, and shared blocks remain alive until all users and operations finish.

These are design constraints, not implemented server features. The current
CPU cache centralizes concrete storage operations without adding unused paging,
scheduling or device interfaces. A contiguous CPU layout must
not become a requirement imposed on future device backends.

## Progress and text delivery

The format layer reports completed tensor payload bytes through an optional
`LoadProgress` callback. Inference reports decoded byte chunks through an
optional generation callback. Both run synchronously on their caller, hold no
global subscriber state, and leave terminal formatting to the CLI. Callback
exceptions propagate; consumers must not reenter the same model. Future serving
can adapt these callbacks without importing console code into lower layers.
The callbacks do not provide scheduling, cancellation or concurrent sessions.

## Error handling

Runtime validation throws exceptions; the CLI catches `std::exception`, prints
the diagnostic and returns failure. Checks on external input and resource
limits must remain active in release builds. Assertions, if added, are for
internal programmer invariants and must not replace those checks.

CPU dispatch waits for every participant before propagating a task exception
on the calling thread. This keeps the borrowed callable alive and leaves the
pool reusable. Other participants finish their work; there is no cancellation
or rollback. Partially written outputs are invalid, and pool recovery alone
does not establish that a failed model/session can resume. Partial pool startup
joins threads already created; a failed thread-count change leaves the backend
in serial mode, from which it can be configured again.

GGUF reads and seeks throw on stream failure, including truncated payloads.
The reader bounds metadata lengths/counts by the opened file extent, limits
array nesting, checks tensor-size arithmetic and validates every payload range
before payload allocation or loading progress. It honors declared file alignment.
These are structural checks: metadata string encoding and model configuration,
required tensor names/shapes and token IDs still need validation before execution.
Valid large files or overlapping tensor ranges can still exceed available memory;
there is no per-request memory budget. The JSON parser validates syntax and
Unicode with bounded nesting and finite-double storage. The quantize CLI
separately checks parsed dimensions, rank, quantized row width, checked byte
totals and exact binary length before payload allocation or output creation.
Its float input buffer owns properly aligned float objects. These conversion
checks do not validate model execution schemas. A future server must define
request/session recovery rather than treating the CLI's process-level catch as request isolation.

## Multi-device / multi-node design notes

The `backend::Backend` interface is device-agnostic in *shape* - nothing in it
names a vendor - but it is host-pointer based today: every call takes raw host
pointers and returns synchronously, so a backend cannot own device memory or
keep activations resident. That is a prerequisite, not a detail. See
`ROADMAP.md` #4a (device execution model), which has to land before any GPU
backend is worth writing.

Once it has, a model can be split across several Backends - one per device, or
per cluster node - using strategies at the model layer (per-layer, per-tensor,
per-row). None of that is implemented yet. See `ROADMAP.md` for the plan.
