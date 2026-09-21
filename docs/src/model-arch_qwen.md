# `src/model/arch_qwen.hpp` - Qwen3 forward pass

Qwen3-style transformer forward pass, from scratch, in namespace `infer`. The
compute primitives (matmul, attention, RMSNorm, RoPE) are delegated to a
`backend::Backend`.

- `QwenConfig` + `load_config(GGUFModel)`: reads Qwen3 metadata
  (`block_count`, `embedding_length`, `feed_forward_length`,
  `attention.head_count[_kv]`, `attention.key_length`, `context_length`,
  `rope.freq_base`, `attention.layer_norm_rms_epsilon`, with the `qwen3.` prefix).
  Consumed integer fields accept positive INT32/UINT32/INT64/UINT64 values up
  to `INT_MAX`. Consumed float fields accept finite positive F32/F64 values
  representable as nonzero F32. Duplicate consumed keys and wrong types fail.
  Optional defaults apply only when absent: KV heads equal query heads, key
  width is an exact embedding/head quotient, context is 4096, RoPE base is
  10000 and RMS epsilon is 1e-6. Explicit key width can differ from that quotient.
  Head width must be even; GQA head counts must divide and projection widths
  fit the runtime's integer indices. Context storage must fit float vectors.
  Declared value/rotary widths must equal key width. Declared architecture and
  tensor layout must be `qwen3` and `reference`; absence remains accepted for
  existing synthetic models. RoPE scaling is unsupported: type must be absent
  or `none`, and current/legacy factors absent or exactly one. These keys use
  the [GGUF metadata vocabulary](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- `Weight` / `LayerWeights`: a tensor resolved once at load - type, storage
  pointer and the two dimensions - and the eleven per-layer weights grouped
  together. `Weight::f32()` is the whole row for a normalization weight, which
  validation guarantees is F32. The forward pass indexes `layers_[l]` instead
  of rebuilding `"blk.N."` and hashing a tensor name for every projection of
  every layer of every token. It exists so a device backend can recognize the
  same weight across calls and keep it resident; no admissible CPU performance
  measurement exists yet. See `docs/DEVICE-EXECUTION.md` step 1.
- `Placement`: a device index per tensor role: each layer's attention and
  feed-forward block, the embedding table and the output head. Empty means
  everything on device 0. Per role rather than per layer so expert offload
  can later put a layer's experts on the CPU while its attention stays on
  the device (`docs/EXECUTION.md`).
- `Sequence`: one request's history over a model's cache, made by
  `Model::make_sequence`: a block table per storage and the committed
  length, and per device the ticket of the last pass that touched it, which
  a reset waits on. Movable, not copyable. The server keeps one per request;
  the CLI's model keeps one.
- `ExecContext`: one pass in flight, plain data the model fills: the
  activation arena (nine slots at 64-byte offsets in one backend allocation),
  the host-visible logits rows and the submission's ticket. Allocated by the
  first forward that needs it and grown to the largest pass seen. Two
  contexts let a scheduler keep one pass on the device while it reads
  another's logits. `logits(i)` is row `i` of the last pass, in entry order.
- `BatchEntry`: what one sequence contributes to a pass: tokens appended to
  it and whether the logits after its last token are wanted. A prefill
  microbatch is one entry with many tokens, a decode batch is many entries
  with one, and they mix.
- `Model`: loads tensors from a `GGUFModel` over one backend, or over
  several with a `Placement`. Each weight is adopted by the backend that
  hosts its role, which on the CPU is the mapped file and costs no RAM.
  Each device that runs attention gets a `KVStorage` for exactly its layers
  with its own pool, block size and adopted RoPE tables. The residual
  stream crosses devices wherever the placement changes, through the
  context's staging vector: a `read` from the source, a `write` into the
  destination, once per boundary per pass. One default sequence and context
  serve the single-sequence entry points. Read-only after construction
  apart from pool bookkeeping.
  - `forward(ctx, entries, n)`: one pass over every entry. Each sequence's
    tokens go through the graph at their own positions and attend through
    their own history via one view per entry and per storage; the rows that
    want logits are gathered, normed and projected once on the output
    device; the pass is one submission per device, waited on only when
    logits are wanted. It is one transaction: every sequence
    commits only once the logits exist, and a failure anywhere drains the
    backend and leaves every history as it was. A sequence listed twice is
    refused.
  - `make_sequence()`, `reset(sequence)`: a fresh history, and one returned
    to the pool after waiting on its last ticket.
  - `set_threads(n)`, `threads_available()`, `n_tokens()`, `head_dim()`,
    `context_length()`. The thread getter reports the resolved backend count,
    allowing the CLI to restore automatic decode settings after prefill.
  - `step(token_id) -> logits`: one entry of one token through `forward` on
    the model's own sequence and context. This is the decode path.
  - `prefill(ids) -> logits`: the prompt in chunks of `ubatch()` tokens, one
    entry per chunk, inside one backend prefill scope, so each weight row is
    read once per chunk instead of once per token. Only the last chunk asks
    for logits. The prompt is one transaction across its chunks.
  - `set_ubatch(n)` / `ubatch()`: physical batch, llama.cpp's `n_ubatch`, set
    by `--ubatch`. llmx has no logical batch; see `docs/USAGE.md`.
  - `reset()`: the default sequence's history returns to the pool while
    allocated KV capacity is retained.
  - Both forward paths call `Backend::attention` over the KV cache; score
    scratch, causal masking and head scheduling belong to the backend.
    Storage grows before the forward pass, preserving the used prefix of
    every head and layer. Projected token-major K/V rows are written into
    contiguous per-head histories, with an explicit head stride for attention.
  - Q/K/V and FFN gate/up share activations and use `matmul_group` in both
    forward paths. CPU groups eligible decode projections; batched prefill
    retains sequential matrix calls through the backend fallback.
  - Batched norms, per-head norm/RoPE, SiLU and the residual adds are backend
    ops (`rms_norm_rows`, `norm_rope_rows`, `silu_mul`, `add`), so the model
    holds no elementwise loops and needs no host parallelism of its own. Each
    row keeps the same arithmetic, and the operations finish before dependent
    matrix operations or KV writes begin. Whether to spread a stage across
    workers is the backend's decision, not the model's.
  - `matvec` / `matmul` / `dequant_row`: helpers taking a resolved `Weight`,
    which carries the type and dimensions, so they dispatch through
    `quant::Registry` without a name lookup. Q8_0 uses the backend's fused AVX2
    matvec; Q4_K has a fused decode dot; other supported quants use a
    generic dequant-row-to-f32 + dot path.
  - The constructor calls `quant::register_builtins()` (idempotent) so the
    quant registry is populated before any tensor is processed.
  - Before model activation/KV/RoPE allocation, construction checks tensor-name
    uniqueness, offset count/alignment/ranges, supported storage types and all
    required layouts. `resolve_tensors` performs this validation and returns
    the `Weight` for each tensor from the same check, so a resolved handle is
    well-formed by construction and no other path produces one.
    Norms are F32 vectors. Matrices have the expected input
    and output dimensions, with equal embedding/output vocabulary sizes.
    Trailing singleton dimensions up to rank four are accepted. Valid payload
    aliases and unused scalar/empty F32 tensors remain supported. The backend
    can already have allocated its worker pool before these checks.

Supports dense Qwen3 with Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K weights
and F32 embeddings/matrices/norms. F32 embedding rows are copied directly;
F32 matmul reads weight rows without staging. Missing
`output.weight` selects tied token embeddings for the output projection.

The borrowed GGUF model must outlive `Model` and remain unchanged. Construction
does not scan numerical weight contents, validate every possible metadata
extension, check arbitrary token IDs or establish recovery after an execution
failure. Those require separate input/session checks; they are not guarantees
of the configuration and layout validation above.

`prefill` enters one backend-owned synchronous scope around batch-buffer
allocation and every microbatch, including the final projection. Empty input
is rejected before entry; individual decode steps do not enter that scope.
