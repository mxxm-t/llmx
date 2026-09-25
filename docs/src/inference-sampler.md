# `src/inference/sampler.hpp` - the sampler and its settings

The sampler and the sampling settings it reads, with their defaults and ranges. Namespace `infer`.
Its callers are `infer::generate` (`inference/generate.hpp`), which the CLI's `generate` and `chat` drive, and the server's scheduler (`server/scheduler.hpp`), which samples each request's logits itself.

- `RNG`: minimal deterministic xorshift64 PRNG (no `<random>` dependency), `seed()`, `next()`, `unit()`.
- `SampleRange<T>`: the values a setting takes, from `lo` to `hi`; `holds` is false for NaN.
- `Sampling`: one generation's settings, `max_tokens`, `temp`, `top_k`, `top_p`, `penalty`, `seed` and `ignore_eos`, each default written here once.
  Beside them are its four ranges, `temp_range` from 0 (greedy), `top_k_range` from 0 (every token kept), `top_p_range` 0 to 1 and `penalty_range` from 1 (no penalty).
  The CLI's flags start from these defaults and read `--temp`, `--topk`, `--topp` and `--penalty` against these ranges, and the server's requests do the same for `temperature`, `top_k`, `top_p` and `penalty`, so the two take the same defaults and refuse the same values, except the `top_k` of -1 that the compatible routes take as 0.
  `ignore_eos`, off by default, is the CLI's `--ignore-eos` and a request's `ignore_eos`: a reply that ends only at its token limit or a stop text.
- `GenParams`: `Sampling` plus the one `stop` text of the CLI's `generate` and `chat`, which is what `infer::generate` reads.
  The server's `SampleParams` is `Sampling` plus its list of stop texts and `until_limit` (see [server](server.md)).
- `sample(logits, temp, top_k, top_p, penalty, gen, rng, masked = -1) -> uint32_t`: temperature + top-k + top-p nucleus sampling with repetition penalty.
  Returns the chosen token id.
  A `masked` id of the row scores negative infinity whatever the penalty, so greedy never takes it, and a draw leaves it out before top-k, top-p and the softmax, so no rounding in the nucleus's sum can fall back on it; a row holding nothing else keeps it, and -1 or an id past the row masks nothing.
  The mask reads the caller's logits as the penalty does and writes nothing to them.
  At `temp <= 0` it takes a linear maximum and returns, allocating nothing; ties go to the lowest token id.
  The scan starts past a masked id 0, so greedy does not give the masked id even when every other score is negative infinity or NaN.
  Above zero the top-k window comes from `std::partial_sort`, because nothing after it is read.
  Ordering the whole vocabulary first cost 12.5 ms per token on Qwen3-8B for a result that reads one element.
  The repetition penalty is applied through a lambda over the caller's logits rather than into a copy.
- `sample(logits, s, end, gen, rng) -> uint32_t`: the next token of a reply under the settings `s`, where `end` is the id that ends a reply (`bpe::Tokenizer::eos_id`, -1 for none), masked when `s.ignore_eos` is set.
  `infer::generate` and the scheduler both sample through it, so the rule is written once and the CLI and the server draw the same tokens for the same settings.
