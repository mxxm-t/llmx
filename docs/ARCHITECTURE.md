# llmx - Architecture

**llmx** is a ground-up, dependency-free LLM inference runtime. It reads and
writes GGUF v3, runs Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / F32 transformers
on x86 CPU with AVX2/FMA/F16C or on a Vulkan device, including read-only MXFP4 where the device provides its required float preservation and double arithmetic, and
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
inference/     model loading (load), sampler (RNG + top-k/top-p/temp/penalty),
               generate loop, chat template rendering, perplexity driver
   |
   v
model/         runtime (Model: sequences, passes, stages, the arena),
               placement (place), one module per architecture under arch/
               chosen by the registry from general.architecture, logical
               KV cache (block pool, sequence), layer split over devices
               (layer_split)
   |
   v
backends/      Backend interface (type-generic matmul / attention / RMSNorm /
               RoPE / batched elementwise ops), cpu/ and vulkan/ impls
   |
   v
tokenizer/     byte-level BPE, Qwen2/Qwen3/Qwen3.5 pretokenizer (encode / decode)
   |
   v
format/        GGUF reader/writer (headers read, payload mapped and read
               in as separate steps), file spans, a file read at
               offsets, raw F32 tensors to and from GGUF; the model is
               built from ModelWeights, the CLI still consumes GGUFModel
   |
   v
quant/         storage metadata and checked row sizes, decoder registry;
               Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / MXFP4 kernels
   |
   v
core/          fp16 and bf16 <-> f32, minimal JSON parser, UTF-8, file hashes, available
               host memory and owned pages, comma-separated lists, the CPUs a
               process may use
```

The rule is: **each layer depends only on the layers below it.** Nothing below
the model layer knows what the model is; nothing below the format layer knows
what a file is. That is what makes each dimension independently replaceable.

### Each concern has one owner

A feature lives in as few files as it can, in the lowest layer that holds
everything it needs, and every other layer reaches it through one call. A
caller calls the owner; it does not repeat the owner's steps. When a second
caller needs the same sequence, the sequence moves down into the owner rather
than being copied. A feature that spans layers by nature keeps only each
layer's own part in that layer, and the part that decides in one file: the
layer split has the backends report their memory (`Backend::memory_available`
and its neighbours), the model fit and place the layers (`layer_split.hpp`,
`infer::place_model`), and the server admit requests per cache pool, while
the CLI only reads `--device` and `--layer-shares`.

What this rules out, from cases found here: the CLI building a placement
from a model's tensor names (moved to `infer::place_model`), a tool building
the same device budgets again (`budgets_for`), and model loading spread over
the format layer (touching pages), the model constructor (dropping pages)
and every command (releasing the host copy), which `infer::load_model` now
does in one place: the format layer reads, maps and reads in files, the
model asks the loader for each weight's storage, and the commands and tools
turn flags into a request. Code that nothing reaches any more is removed in
the change that leaves it unreached.

The quant layer names storage layouts and sizes their rows in one owner,
`quant/types.hpp` (`storage_type`, `row_bytes`). The format layer uses that
metadata to validate GGUF tensor spans without requiring a decoder. The
decoder registry in `quant/quant.hpp` takes its sizes from the same table;
it contains only implemented types. Raw F32 conversion
(`format/raw_convert.hpp`) opens files in the format layer and reaches the
blocks through that registry, checking every decoder before mapping a GGUF
payload or allocating decoded buffers. Dense F32 and
supported block-quant matrices share the CPU float dot kernels; F32 rows
need no dequantization buffer.

### Model architectures

The model layer is one runtime that names no architecture and one module per architecture.
- The runtime (`model/runtime.hpp`, `model/history.hpp` and `model/place.hpp`) owns sequences and the operations on their histories (`model/history.hpp`), passes, stages, the arena, the crossings between devices, placement and the fit, experts on the host and streamed to a device, and the cache storages.
  It indexes a file's tensors by name once and finds each role's tensor there, reads the plan a module declares, and calls the module's parts.
  Before adopting weights it checks each assigned backend's type support and the ops the plan names. A tensor no role uses is refused if none of the model's backends supports its type; metadata-only file access remains independent of these execution checks.
  It decides each host layer's streaming eligibility once, keeping a layer
  on its host when its destination lacks a weight type or an op of its
  feed-forward part; the backend owns the type and op queries over its
  existing kernel support.
- A module (`model/arch/qwen3.hpp` for qwen3 and qwen3moe, `model/arch/qwen35.hpp` for qwen35 and qwen35moe) reads its configuration from the file's metadata and declares a plan: its layers, each layer's kind, cache (KV, a recurrent state or none), tensor roles and the ops some backends lack, the arena slots, the residual width, the context length, the K and V geometry, the state's shape and the position tables.
  The graph pieces more than one module runs are in `model/arch/blocks.hpp`.
  It supplies its math as backend ops, which the runtime calls once per layer part: `embed`, `mixer`, `ffn` and `head`.
- The registry (`model/arch/registry.hpp`) maps each `general.architecture` value to its module and is the only place such a name is accepted.

Inside `model/` the dependencies run one way:
- The runtime and a module meet only in the contract, `architecture.hpp` and `weights.hpp`.
- A module includes the contract, `arch/metadata.hpp`, the backend interface, the format headers its readers read and the quant headers its synthetic model's writer uses. It may include another module's graph header when it runs that module's blocks, and it never includes the runtime or the registry.
- The runtime never includes a module.
- The registry includes every module, and only the loader (`inference/load.hpp`), the tests and tools that build a model without the loader, and the CLI, for `bench`'s synthetic model, include the registry.

`tests/arch_boundary.py` holds the runtime, the loader, the passes, the server, the CLI and the headers the modules share to naming no registered architecture and no tensor.
A new architecture is added as [ADDING-AN-ARCHITECTURE](ADDING-AN-ARCHITECTURE.md) describes.

## What lives where

| Directory       | Contents                                                              |
|-----------------|-----------------------------------------------------------------------|
| `src/` root     | `config.hpp` (build configuration: version and the `LLMX_HAS_BACKEND_*` switches) |
| `core/`         | `fp16.hpp` (half <-> float), `bf16.hpp` (BF16 <-> float), `json.hpp` (recursive-descent parser), `utf8.hpp` (UTF-8 encoding and validation), `sha.hpp` (Hub file hashes), `host_memory.hpp` (the host memory a process can still take, within its cgroup or job object memory limits, the page size, `HostPages`: owned page-aligned memory, and address space reserved and committed by range), `list.hpp` (comma-separated values), `cpus.hpp` (the CPUs a process may use, by its affinity and its CPU quota, and `automatic_threads`, the worker count a pool takes when given none), `cgroup.hpp` (the directories of the Linux cgroups over a process, where its limits are read) |
| `hub/`          | `manifest.hpp` (Hub metadata/quant selection), `transport.hpp` (curl HTTPS transport), `pull.hpp` (verified download cache) |
| `quant/`        | `types.hpp` (storage metadata and checked `row_bytes`), `quant.hpp` (decoder registry + block quants), `k_quants.hpp` (K-quants) |
| `format/`       | `format.hpp` (`FileSpan`, where a tensor lies in its file, and `LoadProgress`), `file_reader.hpp` (a file read at given offsets by several threads, through the file cache or around it, which the loader streams weights through), `gguf.hpp` (GGUF v3: `read_gguf` reads the headers, `map_payload` maps the payload, `warm` reads it in), `mapped_file.hpp` (read-only mapping), `output_file.hpp` (checked staging and publication of conversion outputs), `raw_convert.hpp` (raw F32 tensors to and from GGUF, for `quantize` and `dequantize`) |
| `tokenizer/`    | `tokenizer.hpp` (byte-level BPE, Qwen2/Qwen3/Qwen3.5 pretokenizer)     |
| `model/`        | `weights.hpp` (the format-neutral weights a model is built from, `ModelWeights`, and the resolved `Weight`), `architecture.hpp` (the contract an architecture implements: its plan and its parts), `runtime.hpp` (the runtime that runs a plan and its parts: sequences, passes, stages, the arena, crossings, `Placement` of each tensor role), `history.hpp` (the operations on a sequence's history, members of the runtime's `Model`: fork, reset, retract, mark, keep and checkpoint), `place.hpp` (the memory footprint from the plan, and `place_model`, which places a model over its backends), `arch/registry.hpp` (the architectures by `general.architecture`, and `gguf_weights`), `arch/metadata.hpp` (typed metadata reads), `arch/blocks.hpp` (the graph pieces modules share), `arch/qwen3.hpp` (Qwen3 and qwen3moe), `arch/qwen35.hpp` (Qwen 3.5, 3.6 and 3.8), `kv_cache.hpp` (logical KV: block pool, sequence; the recurrent state's slots, `SlotPool` and `StateSlot`), `layer_split.hpp` (layers per device fitted to their free memory, architecture-neutral) |
| `backends/`     | `backend.hpp` (interface), `kv_storage.hpp` (the paged KV storage the backends derive theirs from: buffers, accounting, growth and view checks), `devices.hpp` (the backend a device spec names: `device_specs`, `make_backends`), `device_profile.hpp` (what a GPU backend shapes its kernels by, shared across vendors), `cpu/cpu_backend.hpp` (AVX2 impl), `cpu/q8_dots.hpp` (the CPU's dots against quantized activations), `cpu/prefill_placement.hpp` (Windows policy), `vulkan/` (the Vulkan backend and its GLSL kernels, `VULKAN.md`) |
| `inference/`    | `load.hpp` (`load_model`, the one load sequence, in the mode `--load-mode` names: file read; mapped and read in when the host has room (`mapped`), mapped only for a host that reads in place (`auto`), or never mapped, a host's weights read into its own copy laid out as the file (`direct`); tokenizer, chat format, weights, placed model with each weight's reader and each streamed copy's storage recorded; copies streamed in file order on up to two reader threads, around the file cache in `direct` and in `auto` when they would not fit in it; host copy released or unread pages dropped), `sampler.hpp`, `logprobs.hpp` (log-softmax of a logits row), `generate.hpp`, `perplexity.hpp`, `chat.hpp` |
| `server/`       | `http.hpp` (HTTP/1.1 over sockets, no dependencies), `scheduler.hpp` (admission, batching, the rounds over the model's pass API, sampling, prefix reuse), `policy.hpp` (the scheduler's policy core: room, the round's stages, the logits rows), `sampling_pool.hpp` (the scheduler's sampling threads), `api.hpp` (the native and OpenAI-compatible routes), per `SERVER.md` |
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
  backend, the only optional backend today; ROCm, CUDA and SYCL are planned and
  each gets its option with its implementation.
- **Model architectures** are compiled in and selected from metadata by
  `model/arch/registry.hpp`; today qwen3, qwen3moe, qwen35 and qwen35moe.
- **Split mode** is a runtime parameter: `--device` with several devices
  runs the model as a layer split over them (`MULTI-DEVICE.md`); the tensor
  split and node count are planned. See `ROADMAP.md`.

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

`Model` holds the architecture and its plan, the weights, a KV pool and
storage on each device whose layers keep KV, a state storage on each device
whose layers keep a recurrent state and the slots of that state, the
backends it is placed over, and the one sequence and context its own `step`, `prefill` and `score` use; apart from those and pool bookkeeping it is read-only after construction. A
`Sequence` is one request's history, an `ExecContext` is where passes run
(activation arenas, handoff buffers, logits rows, tickets, the plan of each
pass in flight), and `Model::forward` runs one pass over a batch of entries,
each a sequence with tokens to append, stage by stage. The CLI uses
one sequence and one context through `step` and `prefill`, and a server's
scheduler one context reserved for its passes in flight
(`Model::reserve_passes`), which it drives stage by stage through the pass
API. Parallel work
inside a forward pass does not make concurrent calls to the same `Model`
safe: a backend is driven by one thread at a time, and a server's scheduler
is that thread.

The KV cache is paged (`docs/KV-CACHE.md`). `model/kv_cache.hpp` owns the
logical side, a block pool and one sequence's block table and committed
length; the backend owns the physical blocks, their size and layout, and
backs them on demand. The model hands the backend a view and never computes
an offset into KV storage. A fork shares whole physical blocks without copying their
contents; it allocates its own logical block tables. A block returns to the
pool only after the backend has retired the work
that read it; the server reuses a finished request's blocks for a prompt
that repeats its tokens (`docs/SERVER.md`). A model whose layers keep a
recurrent state is forked only at a sequence's checkpoint, a state kept at a
position (`docs/SPECULATIVE.md`).

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

Loading reports progress through an optional `LoadProgress` callback, which the loader (`infer::load_model`) drives.
With `--load-mode mapped` it counts the payload bytes whose pages were read in (`gguf::warm`), in file order, before the model is placed; a payload larger than available host memory is not read in, and its progress goes from 0 straight to complete.
With `auto` and `direct` it starts once the model is built and counts the bytes of every tensor a backend took, each once: those streamed to the devices as each read's copies are made, and those a host reads in place as they are read, into its own copy (`direct`) or through the mapping after the uploads (`auto`), so it reaches complete after the last upload.
Reading a file's headers (`gguf::read_gguf`) and mapping its payload report nothing.
Inference reports decoded byte chunks through the optional generation callback of `infer::generate`, which the CLI's `generate` and `chat` drive.
Both callbacks run synchronously on their caller, hold no global subscriber state, and leave terminal formatting to the CLI.
Callback exceptions propagate; consumers must not reenter the same model.
The server does not use the generation callback: its scheduler samples every request's logits itself, ends a request with its own per-token check over the same sampler and `Tokenizer::is_eos`, and hands each token to the request's connection through a channel (`docs/SERVER.md`).
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
does not establish that a failed model/session can resume.
The pool starts on the first dispatch that needs it.
A partial start joins the threads already created and fails that dispatch; the count stays and the next dispatch retries, so a lasting failure fails every dispatch rather than leaving the backend serial.

Model loading failures drain each used backend before constructor members
unwind; model destruction drains them before owned buffers are released.
A failed Vulkan adoption also drains while its partially uploaded destination
is still alive. Successful adoption remains asynchronous, with later work
ordered after the uploads on the same backend. Vulkan KV growth also drains
failed copies before releasing new storage, leaving its prior capacity and peak
accounting intact for retry. Buffer construction releases each acquired handle
on failure. Cached padded weights and argument arenas retain ownership before
recording or replacing queued storage, without adding a wait to successful operations.
Kernel creation also cleans partial resources before retry and publishes
ownership only after success; diagnostic query pools are released after device
idle. Padded-cache invalidation reserves retirement capacity before moving any
entry, so allocation failure leaves the cache intact. The dated
[backend audit](benchmarks/backend-audit-20260925/README.md) retains the original
failure controls. Its separate CPU row-run finding is closed: `for_each_run`
(`src/backends/backend.hpp`) checks that the runs are in row order and end at
the call's last row before any callback, so a malformed list reaches no rows.

GGUF metadata reads and seeks throw on stream failure. Payload extents are
checked when the headers are read, and a file whose size changed before its
payload is mapped is refused, so a file truncated before loading is refused. Mapped files
must remain unchanged for their lifetime; accessing pages removed by a later
truncation can terminate the process on POSIX instead of throwing an exception.
The reader bounds metadata lengths/counts by the opened file extent, limits
array nesting, checks tensor-size arithmetic and validates every payload range
before mapping payloads or reporting loading progress. It honors declared file alignment.
These are structural format checks. Taking a GGUF model's weights (`infer::gguf_weights`)
separately validates consumed configuration values, attention geometry, tensor
ranks and in-memory payload ranges (`read_gguf` has refused repeated names). The module's plan names the tensors
and shapes, and the runtime validates each by its role's kind, normalization types included, all before model
activation/KV/RoPE allocation. Explicit malformed values cannot select optional metadata defaults.
The loader's callers make the backends before the file is read, so a device that cannot be opened fails first.
Weights a host backend reads in place must stay unchanged for the model's lifetime once the load has returned. In a direct load they are the loader's own copy of each file (`LoadedModel::host`), which it fills after the model is built and before anything reads it, since a backend that reads in place does not read a weight as it adopts it.
In the streamed loads, `auto` (the default) and `direct`, a backend that copies gets its weights' storage while the model is built (`Backend::alloc_weight`), so a model that cannot be placed fails before any weight is read, and the loader (`infer::load_model`) then streams them from the file in file order; once they are written it releases the host payload (`GGUFModel::release_payload`) when no host reads a weight in place, as on device backends alone.
Metadata string encoding, numeric weight contents, arbitrary token IDs and dynamic request limits are not fully validated by construction.
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
enqueue on one implicit stream with ticket submission per backend. Results and
resource lifetimes determine when to wait. Backends own device arithmetic and
storage layout; model contexts retain host-visible logits and handoff buffers.

A model is split across several Backends by a `Placement` at the model
layer: a device per tensor role, so per-layer and per-tensor splits are the
same mechanism and the CPU is one of the devices. The residual stream
crosses at a boundary as a `copy` into the source's host-visible handoff
buffer inside its own submission, then a wait on that ticket and a `write`
into the destination. Per-row split is not
planned. This, the tickets, the batched views and the `Model` /
`Sequence` / `ExecContext` split are designed in `EXECUTION.md` and
implemented, as are the Vulkan backend (#4b) and the multi-user server (#7).
A layer split over the devices `--device` lists is fitted by
`model/layer_split.hpp` from the `footprint` of the architecture's plan
(`model/place.hpp`) and each backend's `memory_available()`; the split knows
no architecture and the architecture knows no device. A prompt's chunks
pipeline over the stages, and the server keeps a pass in flight per stage;
the tensor split and a second vendor backend are what `MULTI-DEVICE.md` and
`ROADMAP.md` #4b and #5 still carry.
