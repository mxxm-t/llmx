# `src/inference/generate.hpp` - high-level inference drivers

High-level drivers built on the model + tokenizer. Namespace `infer`.

- `generate(model, tok, gp, rng, logits, emit = {}) -> vector<uint32_t>`: sample
  autoregressively until eos or `gp.max_tokens`, respecting `gp.stop`; a model
  without an EOS id has no stop token, rather than stopping on token zero.
  With `gp.ignore_eos` the sampler never draws the EOS id (`infer::sample` over `Sampling`), so the reply runs to `gp.max_tokens` or a stop match.
  Returns generated ids
  (excluding eos, including a matched stop token). The optional synchronous callback receives decoded byte
  chunks before the next model step; a chunk may split a UTF-8 character.
  Callback consumers must copy retained text and incrementally decode bytes
  when they need complete Unicode characters. No callback means no text output.
  CLI callers own stdout, flushing and the final newline.
  Each token's text is delivered before the next step. A callback exception propagates
  without executing the next step; it does not define session recovery.

