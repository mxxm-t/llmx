# `src/inference/sampler.hpp` — sampling + generation params

Sampling logic and generation parameters, split out of the CLI so the same
sampler drives generate, perplexity, and chat. Namespace `infer`.

- `RNG`: minimal deterministic xorshift64 PRNG (no `<random>` dependency),
  `seed()`, `next()`, `unit()`.
- `GenParams`: `max_tokens`, `temp`, `top_k`, `top_p`, `threads`, `penalty`,
  `seed`, `stop`, `show_prompt_tokens`, `show_thinking`.
- `sample(logits, temp, top_k, top_p, penalty, gen, rng) -> uint32_t`:
  temperature + top-k + top-p nucleus sampling with repetition penalty.
  Returns the chosen token id.
