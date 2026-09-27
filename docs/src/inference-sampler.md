# `src/inference/sampler.hpp` - the sampler and its settings

The sampler and the sampling settings it reads, with their defaults and ranges. Namespace `infer`.
Its callers are `infer::generate` (`inference/generate.hpp`), which the CLI's `generate` and `chat` drive, and the server's scheduler (`server/scheduler.hpp`), which draws each request's token from its row of the pass's logits, read in place, on its sampling threads (`server/sampling_pool.hpp`).

- `RNG`: minimal deterministic xorshift64 PRNG (no `<random>` dependency), `seed()`, `next()`, `unit()`.
- `SampleRange<T>`: the values a setting takes, from `lo` to `hi`; `holds` is false for NaN.
- `Sampling`: one generation's settings, `max_tokens`, `temp`, `top_k`, `top_p`, `penalty`, `seed` and `ignore_eos`, each default written here once.
  Beside them are its four ranges, `temp_range` from 0 (greedy), `top_k_range` from 0 (every token kept), `top_p_range` 0 to 1 and `penalty_range` from 1 (no penalty).
  The CLI's flags start from these defaults and read `--temp`, `--topk`, `--topp` and `--penalty` against these ranges, and the server's requests do the same for `temperature`, `top_k`, `top_p` and `penalty`, so the two take the same defaults and refuse the same values, except the `top_k` of -1 that the compatible routes take as 0.
  `ignore_eos`, off by default, is the CLI's `--ignore-eos` and a request's `ignore_eos`: a reply that ends only at its token limit or a stop text.
- `GenParams`: `Sampling` plus the one `stop` text of the CLI's `generate` and `chat`, which is what `infer::generate` reads.
  The server's `SampleParams` is `Sampling` plus its list of stop texts, `until_limit`, `logprobs` and `top_logprobs` (see [server](server.md)).
- `sample(logits, n, temp, top_k, top_p, penalty, gen, rng, masked = -1) -> uint32_t`: temperature + top-k + top-p nucleus sampling with repetition penalty, over the `n` logits of one row.
  Returns the chosen token id.
  It reads the row in place and never writes it, so the server draws from a pass's mapped logits without copying them.
  A `masked` id of the row is passed over by every path whatever the penalty, so greedy never takes it, and a draw leaves it out before top-k, top-p and the softmax, so no rounding in the nucleus's sum can fall back on it; a row holding nothing else keeps it, and -1 or an id past the row masks nothing.
  The mask writes nothing to the caller's logits.
  The repetition penalty is applied into a copy of the logits, made only when a token is penalized, and a token seen several times is penalized once.
  At `temp <= 0` it takes a linear maximum and returns; ties go to the lowest token id.
  The scan starts past a masked id 0, so greedy does not give the masked id even when every other score is negative infinity or NaN.
  Above zero tokens rank by score and a tie by the lower id, through `detail::rank_key`, one integer per token whose order is that ranking, so no sort's or selection's handling of equal scores reaches the result.
  A token's weight is `exp((score - best score) / temp)`.
  The kept tokens' weights are summed in id order, and the nucleus is the shortest ranked prefix whose weight, summed best first, reaches `top_p` of that sum, so no sum depends on the order a selection leaves its candidates in.
  The draw walks the drawn tokens to `r` times their weight in the order that weight was summed in: the nucleus best first, or without top-p the kept tokens in id order.
  `detail::Ranking` ranks the kept tokens only as far as they are read: a top-k set of up to 4096 in one pass over the scores with a heap, a nucleus without a top-k to 512 in at most two such passes, and more by laying out the keys not yet ranked once behind the ranked ones and selecting each further prefix with `std::nth_element`.
  The heap's pass forms a token's key only when its score is not below the worst score the heap holds, since a lower score has the smaller key whatever its id.
  A top-k below the vocabulary is selected first, and its best token is then found within it.
  Without a top-k, or with one past 4096, a nucleus is ranked 64 tokens first and then 512, through heap passes without a top-k and selections within one, and past that each selection doubles the ranked prefix, so top-k 0 with top-p below 1 ranks 64 tokens, fewer than eight times a nucleus of up to 512, or fewer than twice a larger one, and the whole vocabulary only when the nucleus holds more than half of it.
  A nucleus leaves the heap at 512 because on rows whose nucleus runs to tens of thousands of tokens a heap pass of 4096 cost more than the selections it saves.
  With every token kept each weight is held by id as the sum takes it, in a buffer that is not zeroed, so the nucleus and the draw take no exp again; within a top-k the nucleus takes its tokens' weights again, at most k of them.
  With top-k 0 and top-p 1 nothing is ranked: one pass finds the best score, one sums the weights and the draw walks the ids in order.
- `sample(logits, n, s, end, gen, rng) -> uint32_t`: the next token of a reply under the settings `s`, where `end` is the id that ends a reply (`bpe::Tokenizer::eos_id`, -1 for none), masked when `s.ignore_eos` is set.
  `infer::generate` and the scheduler both sample through it, so the rule is written once and the CLI and the server draw the same tokens for the same settings.
