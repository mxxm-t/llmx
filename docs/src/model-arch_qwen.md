# `src/model/arch_qwen.hpp` — Qwen3 forward pass

Qwen3-style transformer forward pass, from scratch, in namespace `infer`. The
compute primitives (quantized matmul, RMSNorm, RoPE) are delegated to a
`backend::Backend`.

- `QwenConfig` + `load_config(GGUFModel)`: reads Qwen3 metadata
  (`block_count`, `embedding_length`, `feed_forward_length`,
  `attention.head_count[_kv]`, `key_length`, `context_length`, `rope_theta`,
  `rms_eps`).
- `Model`: loads tensors from a `GGUFModel`, owns the KV cache.
  - `set_threads(n)`, `n_tokens()`, `head_dim()`, `context_length()`.
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
    matvec; Q4_K has a fused decode dot; other supported quants use a
    generic dequant-row-to-f32 + dot path.
  - The constructor calls `quant::register_builtins()` (idempotent) so the
    quant registry is populated before any tensor is processed.

Supports dense Qwen3 with Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K weights
and F32 embeddings/matrices/norms. F32 embedding rows are copied directly;
F32 matmul reads weight rows without staging. Missing
`output.weight` selects tied token embeddings for the output projection.
