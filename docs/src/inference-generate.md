# `src/inference/generate.hpp` - high-level inference drivers

High-level drivers built on the model + tokenizer. Namespace `infer`.

- `prefill(model, ids) -> logits`: feed every id through the model (updating the
  KV cache), return the last token's logits (distribution over the next token).
- `generate(model, tok, gp, rng, logits, emit = {}) -> vector<uint32_t>`: sample
  autoregressively until eos or `gp.max_tokens`, respecting `gp.stop`; by
  default attempts to hide tokens matching `thinking_start` / `thinking_end`
  (pass `gp.show_thinking` to disable that filter). These substring markers do
  not cover Qwen3's `<think>` / `</think>` tokens. Returns generated ids
  (excluding eos, including a matched stop token). The optional synchronous callback receives decoded byte
  chunks before the next model step; a chunk may split a UTF-8 character.
  Callback consumers must copy retained text and incrementally decode bytes
  when they need complete Unicode characters. No callback means no text output.
  CLI callers own stdout, flushing and the final newline.
  Legacy thinking/answer markers can retroactively discard text, so that
  vocabulary/filter mode emits only after generation completes. `--think`
  bypasses filtering and streams immediately. A callback exception propagates
  without executing the next step; it does not define session recovery.

Also includes `find_token_by_substr` for reasoning-block detection.
