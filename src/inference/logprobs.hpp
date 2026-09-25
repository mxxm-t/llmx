#pragma once
// The model's distribution over the next token as log-probabilities: the log-softmax of one logits row, which perplexity scores and the server reports.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace infer {

// The log of the sum of exp over a row, in double: its largest value plus the log of the sum of exp of each value less it, so no term overflows.
// A token's log-probability is its logit less this.
// The maximum is sought from the row's first value rather than from a fixed floor, so a row lying wholly below any floor still shifts by its own; an empty row gives minus infinity, the log of an empty sum.
inline double log_sum_exp(const float* logits, size_t n) {
    if (n == 0) return -std::numeric_limits<double>::infinity();
    float maxv = logits[0];
    for (size_t v = 1; v < n; ++v) maxv = std::max(maxv, logits[v]);
    double sum = 0.0;
    for (size_t v = 0; v < n; ++v) sum += std::exp((double)logits[v] - maxv);
    return maxv + std::log(sum);
}

// The log-probability of `id` in a row whose log_sum_exp is `lse`, computed in double and rounded once to float.
inline float logprob(const float* logits, double lse, uint32_t id) {
    return (float)((double)logits[id] - lse);
}

struct TokenLogprob {
    uint32_t id;
    float logprob;
};

// The `k` most likely tokens of a row, most likely first, a tie going to the lower id as greedy sampling's does; a `k` past the row lists the whole row.
inline std::vector<TokenLogprob> top_logprobs(const float* logits, size_t n, double lse, size_t k) {
    k = std::min(k, n);
    std::vector<uint32_t> best;
    best.reserve(k + 1);
    for (size_t v = 0; v < n && k; ++v) {
        if (best.size() == k && !(logits[v] > logits[best.back()])) continue;
        // After every id with an equal logit, which is lower since ids arrive in order.
        auto at = std::upper_bound(best.begin(), best.end(), logits[v],
                                   [&](float x, uint32_t b) { return x > logits[b]; });
        best.insert(at, (uint32_t)v);
        if (best.size() > k) best.pop_back();
    }
    std::vector<TokenLogprob> top;
    top.reserve(best.size());
    for (uint32_t id : best) top.push_back({id, logprob(logits, lse, id)});
    return top;
}

} // namespace infer
