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
- `Model`: loads tensors from a `GGUFModel`, owns one sequence's logical token
  count and a `HostKVCache` for physical CPU storage.
  - `set_threads(n)`, `threads_available()`, `n_tokens()`, `head_dim()`,
    `context_length()`. The thread getter reports the resolved backend count,
    allowing the CLI to restore automatic decode settings after prefill.
  - `step(token_id) -> logits`: run one token through the full forward pass
    (embedding, per-block attention + FFN, output norm + head), updating the KV
    cache. This is the decode path.
  - `prefill(ids) -> logits`: run a whole prompt through matrix-matrix matmuls
    in chunks of `ubatch()` tokens, so each weight row is read once per chunk
    instead of once per token. Only the final token's logits are produced, so
    the vocab projection stays a single matvec. The limiting resource depends
    on the model, batch size, hardware and competing workloads.
  - `set_ubatch(n)` / `ubatch()`: physical batch, llama.cpp's `n_ubatch`, set
    by `--ubatch`. llmx has no logical batch; see `docs/USAGE.md`.
  - `reset()`: reset logical history while retaining allocated KV capacity.
    Future attention sees only the newly written sequence extent.
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
