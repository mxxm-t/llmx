# `src/model/arch_qwen.hpp` — Qwen3 forward pass

Qwen3-style transformer forward pass, from scratch, in namespace `infer`. The
compute primitives (quantized matmul, RMSNorm, RoPE) are delegated to a
`backend::Backend`.

- `QwenConfig` + `load_config(GGUFModel)`: reads Qwen3 metadata
  (`block_count`, `embedding_length`, `feed_forward_length`,
  `attention.head_count[_kv]`, `key_length`, `context_length`, `rope_theta`,
  `rms_eps`).
- `Model`: loads tensors from a `GGUFModel`, owns the KV cache.
  - `set_threads(n)`, `n_tokens()`, `head_dim()`.
  - `step(token_id) -> logits`: run one token through the full forward pass
    (embedding, per-block attention + FFN, output norm + head), updating the KV
    cache. This is the decode path.
  - `prefill(ids) -> logits`: run a whole prompt through matrix-matrix matmuls
    in chunks of `ubatch()` tokens, so each weight row is read once per chunk
    instead of once per token. Only the final token's logits are produced, so
    the vocab projection stays a single matvec. This is the prefill path and is
    compute bound, unlike decode.
  - `set_ubatch(n)` / `ubatch()`: physical batch, llama.cpp's `n_ubatch`, set
    by `--ubatch`. llmx has no logical batch; see `docs/USAGE.md`.
  - `reset()`: clear KV cache / internal state.
  - `attend_heads` / `attend_head`: (parallel) attention over the KV cache.
  - `matvec` / `dequant_row`: per-tensor matmul helpers that dispatch on the
    tensor's type via `quant::Registry`. Q8_0 uses the backend's fused AVX2
    matvec; other types use a correct generic dequant-row-to-f32 + dot path.
  - The constructor calls `quant::register_builtins()` (idempotent) so the
    quant registry is populated before any tensor is processed.

Supports dense Q8_0 / F32 and Q4_0 tensors (see the tensor layout comment in the
header). Q4_0 uses the generic (correct-but-slower) matmul path until a fused
kernel lands.
