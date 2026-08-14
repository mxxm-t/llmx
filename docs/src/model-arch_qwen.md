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
    cache.
  - `reset()`: clear KV cache / internal state.
  - `attend_heads` / `attend_head`: (parallel) attention over the KV cache.
  - `matvec` / `dequant_row`: per-tensor matmul helpers.

Supports dense Q8_0 / F32 tensors only (see the tensor layout comment in the
header).
