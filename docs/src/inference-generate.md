# `src/inference/generate.hpp` - high-level inference drivers

High-level drivers built on the model + tokenizer. Namespace `infer`.

- `generate(model, tok, gp, rng, logits, emit = {}, drafting = nullptr) -> vector<uint32_t>`: sample
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
  Every token is fed but an eos and the token that completes the stop text, the last one `gp.max_tokens` allows included, so the history the model holds afterwards is the same however the tokens came.
  With `drafting` ([spec](inference-spec.md)) each round asks `spec::draft_length` how many drafts to take, the proposer for them, marks the history (`Model::mark`), feeds the last pick and the drafts in one pass of generated tokens (`Model::step(ids, n)`), samples its rows through `infer::accept` and retracts the history to what the run without drafts holds (`Model::retract`), feeding a last token the limit ends on by a step where the verify did not; where it drafts nothing or no mark is free it takes a single step. The tokens, the draws and the history are those of the run without drafts, and text is delivered as each row is sampled, after the verify's pass.

