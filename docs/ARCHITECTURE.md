# llmx - Architecture

**llmx** is a ground-up, dependency-free LLM inference runtime. It reads and
writes GGUF v3, runs Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / F32 transformers
on x86 CPU with AVX2/FMA/F16C or on a Vulkan device, and
is structured so more formats, quantizations, backends, and even multi-device /
multi-node serving can be added without touching the core.

## Layer diagram

```
cli/           argument parsing, command dispatch, usage text
   |
   v
server/        HTTP transport, scheduler, native and compatible routes (SERVER.md)
   |
   v
inference/     sampler (RNG + top-k/top-p/temp/penalty), generate loop,
               chat template rendering, perplexity driver
   |
   v
model/         Qwen3 Model + logical KV cache (block pool, sequence);
               architecture registry planned, forward graph (arch_qwen)
   |
   v
backends/      Backend interface (type-generic matmul / attention / RMSNorm /
               RoPE / batched elementwise ops), cpu/ and vulkan/ impls
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
| `core/`         | `fp16.hpp` (half <-> float), `json.hpp` (recursive-descent parser), `sha.hpp` (Hub file hashes) |
| `hub/`          | Hub metadata/quant selection, curl HTTPS transport, verified download cache |
| `quant/`        | `quant.hpp` (registry + block quants), `k_quants.hpp` (K-quants)                       |
| `format/`       | `format.hpp` (ModelFormat interface), `gguf.hpp` (GGUF v3), `mapped_file.hpp` (read-only mapping) |
| `tokenizer/`    | `tokenizer.hpp` (byte-level BPE, Qwen2/Qwen3 pretokenizer)             |
| `model/`        | `arch_qwen.hpp` (Qwen3 config + forward pass), `kv_cache.hpp` (logical KV: block pool, sequence) |
| `backends/`     | `backend.hpp` (interface), `device_profile.hpp` (what a GPU backend shapes its kernels by, shared across vendors), `cpu/cpu_backend.hpp` (AVX2 impl), `cpu/prefill_placement.hpp` (Windows policy), `vulkan/` (the Vulkan backend and its GLSL kernels, `VULKAN.md`) |
| `inference/`    | `sampler.hpp`, `generate.hpp`, `perplexity.hpp`, `chat.hpp`    |
| `server/`       | `http.hpp` (HTTP/1.1 over sockets, no dependencies), `scheduler.hpp` (admission, batching, sampling, prefix reuse), `api.hpp` (the native and OpenAI-compatible routes), per `SERVER.md` |
| `cli/`          | `main.cpp` (thin dispatcher)                                          |

Source and subsystem documentation lives in `docs/src/`, covering
what each header does, its public surface, and its place in the layering. See
`docs/src/cli-main.md` for the CLI entry points and `docs/USAGE.md` for the
command reference.

`hub/` is a CLI-invoked acquisition path beside the inference stack. It depends
on core JSON and hashing, never on a model or backend. Curl is an external HTTPS
process; no Python or TLS library is linked. Bounded range downloads write into
private temporary files; final size/hash verification precedes cache publication.
The CLI owns credentials, options and status rendering. GGUF shard interpretation
belongs to `format/`, so locally supplied and downloaded shards load identically.

## Build-time vs runtime

- **Backends** are the *only* compile-time concern: GPU backends pull in heavy
  SDKs, so they are opt-in via `LLMX_HAS_BACKEND_*` in `config.hpp`. CPU is
  always on (no external deps). `LLMX_HAS_BACKEND_VULKAN` builds the Vulkan
  backend; the ROCm, CUDA and SYCL options define macros only.
- **Model architectures** will be compiled in and selected from metadata.
  Today the model layer implements dense Qwen3 only.
- **Split mode and node count** are planned runtime parameters, not implemented
  build options or CLI flags. See `ROADMAP.md`.

`--threads` and `--threads-batch` select CPU workers for decode and prefill.
GPU backends keep that meaning for applicable CPU work; GPU launch
dimensions belong to the backend. `--ubatch` is the number of prompt tokens
per forward pass and remains relevant to device execution.

CPU prefill enters a synchronous `Backend::run_prefill` scope once around
buffer preparation and all prompt microbatches. The default implementation
invokes the body once on the caller. CPU owns optional Windows worker placement
and cleanup; model code contains no platform scheduling types. Decode steps
remain outside the scope. The scope does not add asynchronous or concurrent
submission support.

## KV state and concurrent execution

`Model` holds the weights, the cache's pool and physical storage, and the
backend, and is read-only after construction apart from pool bookkeeping. A
`Sequence` is one request's history, an `ExecContext` is one pass in flight
(activation arena, logits rows, ticket), and `Model::forward` runs one pass
over a batch of entries, each a sequence with tokens to append. The CLI uses
one sequence and one context through `step` and `prefill`. Parallel work
inside a forward pass does not make concurrent calls to the same `Model`
safe: a backend is driven by one thread at a time, and a server's scheduler
is that thread.

The KV cache is paged (`docs/KV-CACHE.md`). `model/kv_cache.hpp` owns the
logical side, a block pool and one sequence's block table and committed
length; the backend owns the physical blocks, their size and layout, and
backs them on demand. The model hands the backend a view and never computes
an offset into KV storage. A fork shares full blocks and copies the tail,
and a block returns to the pool only after the backend has retired the work
that read it; the server reuses a finished request's blocks for a prompt
that repeats its tokens (`docs/SERVER.md`).

The device and server work (ROADMAP #4a and #7) preserves these
boundaries, and further work must too:

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
  boundaries independently. A prefill microbatch is one entry of one
  sequence; the server's passes mix such entries with decode entries of
  other sequences, each at its own positions.
- With device execution, backend buffers own physical storage and submission
  completion controls its lifetime. Cache growth, reset and reuse must not
  invalidate storage still used by an in-flight operation. Model-level routing
  determines placement across devices without introducing vendor types into
  sequence state.
- Prefix reuse may share immutable KV blocks only when the model,
  positions and relevant execution configuration match. Mutable suffixes stay
  private, and shared blocks remain alive until all users and operations finish.

The paging and block ownership above are implemented, and a block returns to
the pool only after the backend has retired the work that read it;
per-request sequences, scheduling and prefix sharing are the server's
(`docs/SERVER.md`). The
CPU block size and intra-block layout are that backend's choices and must not
become requirements imposed on future device backends.

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
These are structural format checks. Qwen model construction separately validates
consumed configuration values, attention geometry, required tensor names/shapes,
normalization types and in-memory payload ranges before model activation/KV/RoPE
allocation. Explicit malformed values cannot select optional metadata defaults.
The backend may already exist before these model checks. Borrowed model metadata
and weights must stay unchanged for the model's lifetime, except that the CLI
releases the host payload (`GGUFModel::release_payload`) once no weight reads it
in place (`Model::holds_payload`), as on device backends. Metadata string encoding,
numeric weight contents, arbitrary token IDs and dynamic request limits are not
fully validated by construction.
Valid large files or overlapping tensor ranges can still exceed available memory;
there is no per-request memory budget. The JSON parser validates syntax and
Unicode with bounded nesting and finite-double storage. The quantize CLI
separately checks parsed dimensions, rank, quantized row width, checked byte
totals and exact binary length before payload allocation or output creation.
Its float input buffer owns properly aligned float objects. These conversion
checks do not validate model execution schemas. The server ends a request, not
its loop, when a pass fails; that is not the CLI's process-level catch.

## Multi-device / multi-node design notes

The `backend::Backend` interface is device-agnostic in *shape* - nothing in it
names a vendor - and, since the device execution migration
(`DEVICE-EXECUTION.md`, complete), in substance too: weights, activations and
KV blocks are `Buffer` handles, every op takes a buffer and an offset, ops
enqueue on one implicit stream with one `sync()` per forward pass, and the
model layer holds no host address.

A model is split across several Backends by a `Placement` at the model
layer: a device per tensor role, so per-layer and per-tensor splits are the
same mechanism and the CPU is one of the devices. The residual stream
crosses at a boundary through `read` and `write`. Per-row split is not
planned. This, the tickets, the batched views and the `Model` /
`Sequence` / `ExecContext` split are designed in `EXECUTION.md` and
implemented, as are the Vulkan backend (#4b) and the multi-user server (#7);
a second vendor backend and the placement flags are what `ROADMAP.md` #4b
and #5 still carry.
