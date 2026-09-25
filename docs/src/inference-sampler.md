# `src/inference/sampler.hpp` - sampling + generation params

Sampling logic and generation parameters, split out of the CLI so the same
sampler drives generate and chat. Perplexity reuses thread parameters only. Namespace `infer`.

- `RNG`: minimal deterministic xorshift64 PRNG (no `<random>` dependency),
  `seed()`, `next()`, `unit()`.
- `GenParams`: `max_tokens`, `temp`, `top_k`, `top_p`, `threads`, `threads_batch`, `ubatch`,
  `cache_type_k`, `cache_type_v`, `kv_tokens`, `device`, `penalty`,
  `seed`, `stop`, `show_prompt_tokens`.
- `sample(logits, temp, top_k, top_p, penalty, gen, rng) -> uint32_t`:
  temperature + top-k + top-p nucleus sampling with repetition penalty.
  Returns the chosen token id.
  At `temp <= 0` it takes a linear maximum and returns, allocating nothing;
  ties go to the lowest token id. Above zero the top-k window comes from
  `std::partial_sort`, because nothing after it is read. Ordering the whole
  vocabulary first cost 12.5 ms per token on Qwen3-8B for a result that
  reads one element. The repetition penalty is applied through a lambda over
  the caller's logits rather than into a copy.
