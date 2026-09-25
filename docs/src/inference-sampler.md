# `src/inference/sampler.hpp` - sampling + generation params

Sampling logic and generation parameters, split out of the CLI so the same
sampler drives generate and chat. Perplexity reuses thread parameters only. Namespace `infer`.

- `RNG`: minimal deterministic xorshift64 PRNG (no `<random>` dependency),
  `seed()`, `next()`, `unit()`.
- `GenParams`: `max_tokens`, `temp`, `top_k`, `top_p`, `threads`, `threads_batch`, `ubatch`,
  `cache_type_k`, `cache_type_v`, `kv_tokens`, `device`, `layer_shares`,
  `cpu_moe`, `moe_stream_from`, `penalty`, `seed`, `stop`,
  `show_prompt_tokens`. `cache_type_k` and `cache_type_v` hold a cache
  type only as `--cache-type-k` and `--cache-type-v` give it and are empty
  otherwise, which keeps the model's default (`ModelOptions`,
  `model/arch_qwen.hpp`).
- `SampleRange` and the four ranges beside `GenParams`: `kTempRange` from 0 (greedy), `kTopKRange` from 0 (every token kept), `kTopPRange` 0 to 1 and `kPenaltyRange` from 1 (no penalty).
  The CLI reads `--temp`, `--topk`, `--topp` and `--penalty` against them.
- `sample(logits, temp, top_k, top_p, penalty, gen, rng) -> uint32_t`:
  temperature + top-k + top-p nucleus sampling with repetition penalty.
  Returns the chosen token id.
  At `temp <= 0` it takes a linear maximum and returns, allocating nothing;
  ties go to the lowest token id. Above zero the top-k window comes from
  `std::partial_sort`, because nothing after it is read. Ordering the whole
  vocabulary first cost 12.5 ms per token on Qwen3-8B for a result that
  reads one element. The repetition penalty is applied through a lambda over
  the caller's logits rather than into a copy.
