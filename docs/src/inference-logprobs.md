# `src/inference/logprobs.hpp` - log-probabilities of a logits row

The model's distribution over the next token as log-probabilities: the log-softmax of one row of logits.
It has two readers, perplexity's scores (`token_nll`) and the server's `logprobs` (`server/scheduler.hpp`), so both compute the same quantity the same way.
Namespace `infer`.

- `log_sum_exp(logits, n) -> double`: the row's largest value plus the log of the sum of `exp` of each value less it, summed in double, so no term overflows however large the logits are.
  The largest value is sought from the row's first rather than from a fixed floor, so a row lying wholly below any floor still shifts by its own; an empty row gives minus infinity.
- `logprob(logits, lse, id) -> float`: the logit less `lse`, in double, rounded once to float. `token_nll` is the same difference in double, negated, without the rounding.
- `TokenLogprob`: an id and its log-probability.
- `top_logprobs(logits, n, lse, k)`: the `k` most likely tokens, most likely first, a tie going to the lower id as greedy sampling's does, from one pass over the row that keeps the best `k` in a heap with the least likely on top; a `k` past the row lists the whole row, and 0 lists nothing.
  `llmx logits --top` and `llmx-decode-probe` rank their rows through it too, printing the raw logits, so a list of the whole vocabulary costs a sort rather than an insertion per token.

The `logprobs` CTest holds every value of 151936-token rows and of a row below -1e30 within half a float step of a reference that shifts by the maximum and sums with compensation, and the top lists to a full sort by logit and id.
A row costs one pass of `exp` in double over the vocabulary, about a millisecond for 151936 tokens, which is why the server computes it only for a request that asks, and in the thread that reads the request's tokens unless that thread has fallen behind (`Request::kRowsWaiting`), when the draw of the token computes it on the scheduler's sampling threads or its own thread (`server/sampling_pool.hpp`).
