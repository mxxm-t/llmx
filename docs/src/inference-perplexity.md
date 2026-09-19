# `src/inference/perplexity.hpp` - windowed perplexity

`infer::perplexity(model, ids, context_size=0, max_chunks=0)` scores an already
tokenized sequence. Zero selects the model context or all chunks, respectively;
the CLI accepts only positive explicit values. Context must be at least two
and no larger than the model's context.

Windows are disjoint, with reset KV cache and positions. Each first token is
context only; all following tokens are targets. A partial window needs at
least two tokens. No padding, BOS/EOS insertion, overlap or warmup exclusion
occurs. The final target needs no forward pass because its logits are unused.

`PerplexityResult` stores used input tokens, scored targets, chunks and total
NLL. `mean_nll()` divides total NLL by scored targets. `docs/USAGE.md` documents
the CLI counters and examples. The analytic test covers boundaries without a
real model; pinned HF goldens cover continuous and chunked real-model scores.
