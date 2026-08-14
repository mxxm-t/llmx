# `src/inference/generate.hpp` — high-level inference drivers

High-level drivers built on the model + tokenizer. Namespace `infer`.

- `prefill(model, ids) -> logits`: feed every id through the model (updating the
  KV cache), return the last token's logits (distribution over the next token).
- `generate(model, tok, gp, rng, logits) -> vector<uint32_t>`: sample
  autoregressively until eos or `gp.max_tokens`, respecting `gp.stop`; by
  default hides the Qwen3 `<thinking>...<thinking_end>` block (pass
  `gp.show_thinking` to keep it). Returns the generated ids (excluding eos).

Also includes `find_token_by_substr` for reasoning-block detection.
