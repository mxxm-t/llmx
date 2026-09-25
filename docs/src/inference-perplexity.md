# `src/inference/perplexity.hpp` - windowed perplexity

`infer::perplexity(model, ids, context_size=0, max_chunks=0, per_token=false)`
scores an already tokenized sequence. Zero selects the model context or all chunks, respectively.
A window holds at least `kMinPerplexityWindow` tokens, two, since its first token is only context, and at most the model's context.
The CLI reads `--ctx-size` against that floor and `--chunks` from one, so a command line below them is refused before the model loads; a text of fewer than two tokens and a window past the model's context are refused here, once it has.

Windows are disjoint, with reset KV cache and positions. Each first token is
context only; all following tokens are targets. A partial window needs at
least two tokens. No padding, BOS/EOS insertion, overlap or warmup exclusion
occurs. By default each window goes through `Model::score`, batched passes of
up to `ubatch()` tokens with logits for every position, the prompt path; with
`per_token` (the CLI's `--per-token`) it is fed one token at a time through
`step`, the decode path, and the final target needs no forward pass because
its logits are unused. On a device the two paths use different kernels, and
`tests/baseline.py` and `tests/baseline_8b.py` score every HF perplexity cell both ways.

Each target's negative log-likelihood is `token_nll`: its logit less `log_sum_exp` of its row (`inference/logprobs.hpp`, which the server's logprobs read too), in double, negated.
`PerplexityResult` stores the window size it scored with (`context`, the one given or the model's context), used input tokens, scored targets, chunks and total NLL.
`mean_nll()` divides total NLL by scored targets.
`docs/USAGE.md` documents the CLI counters and examples.
The analytic test covers boundaries without a real model; pinned HF goldens cover continuous and chunked real-model scores.
