# `src/inference/sampler.hpp` - the sampler and its settings

The sampler and the sampling settings it reads, with their defaults and ranges. Namespace `infer`.
Its callers are `infer::generate` (`inference/generate.hpp`), which the CLI's `generate` and `chat` drive, and the server's scheduler (`server/scheduler.hpp`), which samples each request's logits itself.

- `RNG`: minimal deterministic xorshift64 PRNG (no `<random>` dependency), `seed()`, `next()`, `unit()`.
- `SampleRange<T>`: the values a setting takes, from `lo` to `hi`; `holds` is false for NaN.
- `Sampling`: one generation's settings, `max_tokens`, `temp`, `top_k`, `top_p`, `penalty` and `seed`, each default written here once.
  Beside them are its four ranges, `temp_range` from 0 (greedy), `top_k_range` from 0 (every token kept), `top_p_range` 0 to 1 and `penalty_range` from 1 (no penalty).
  The CLI's flags start from these defaults and read `--temp`, `--topk`, `--topp` and `--penalty` against these ranges, and the server's requests do the same for `temperature`, `top_k`, `top_p` and `penalty`, so the two take the same defaults and refuse the same values, except the `top_k` of -1 that the compatible routes take as 0.
- `GenParams`: `Sampling` plus the one `stop` text of the CLI's `generate` and `chat`, which is what `infer::generate` reads.
  The server's `SampleParams` is `Sampling` plus its list of stop texts and `until_limit` (see [server](server.md)).
- `sample(logits, temp, top_k, top_p, penalty, gen, rng) -> uint32_t`: temperature + top-k + top-p nucleus sampling with repetition penalty.
  Returns the chosen token id.
  At `temp <= 0` it takes a linear maximum and returns, allocating nothing; ties go to the lowest token id.
  Above zero the top-k window comes from `std::partial_sort`, because nothing after it is read.
  Ordering the whole vocabulary first cost 12.5 ms per token on Qwen3-8B for a result that reads one element.
  The repetition penalty is applied through a lambda over the caller's logits rather than into a copy.
