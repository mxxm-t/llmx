#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/arch_qwen.hpp"

namespace infer {

struct PerplexityResult {
    size_t used_tokens = 0;
    size_t scored_tokens = 0;
    size_t chunks = 0;
    double nll = 0.0;

    double mean_nll() const { return nll / (double)scored_tokens; }
};

// The negative log-likelihood of `target` under one row of logits.
inline double token_nll(const float* logits, size_t vocab, uint32_t target) {
    float maxv = -1e30f;
    for (size_t v = 0; v < vocab; ++v) maxv = std::max(maxv, logits[v]);
    double sum = 0.0;
    for (size_t v = 0; v < vocab; ++v) sum += std::exp((double)logits[v] - maxv);
    return -((double)logits[target] - (maxv + std::log(sum)));
}

// Windows of `context_size` tokens, each from an empty history. By default a window is scored through the batched passes a prompt takes, logits for every position of a microbatch at once, which is the path prompt processing uses and on a device a different set of kernels from decode's. `per_token` scores it one token at a time through step instead, the decode path; the HF gate runs both so each set of kernels meets the reference.
inline PerplexityResult perplexity(Model& model, const std::vector<uint32_t>& ids,
                                   int context_size = 0, int max_chunks = 0, bool per_token = false) {
    if (ids.size() < 2) throw std::runtime_error("perplexity: need at least 2 tokens");
    const int context = context_size == 0 ? model.context_length() : context_size;
    if (context < 2 || context > model.context_length())
        throw std::runtime_error("perplexity: context size must be between 2 and model context " +
                                 std::to_string(model.context_length()));
    if (max_chunks < 0) throw std::runtime_error("perplexity: chunks must be nonnegative");

    PerplexityResult result;
    for (size_t begin = 0; begin < ids.size();) {
        const size_t count = std::min((size_t)context, ids.size() - begin);
        if (count < 2 || (max_chunks > 0 && result.chunks >= (size_t)max_chunks)) break;
        if (!per_token) {
            const std::vector<uint32_t> window(ids.begin() + (std::ptrdiff_t)begin,
                                               ids.begin() + (std::ptrdiff_t)(begin + count));
            model.score(window, [&](size_t pos, const float* logits) {
                // The last position predicts past the window; its logits are unused.
                if (pos + 1 < count) result.nll += token_nll(logits, model.n_vocab(), window[pos + 1]);
            });
            result.used_tokens += count;
            result.scored_tokens += count - 1;
            result.chunks++;
            begin += count;
            continue;
        }
        model.reset();
        std::vector<float> logits = model.step((int)ids[begin]);
        for (size_t i = begin + 1; i < begin + count; i++) {
            const uint32_t target = ids[i];
            float maxv = -1e30f;
            for (float l : logits) maxv = std::max(maxv, l);
            double sum = 0.0;
            for (float l : logits) sum += std::exp((double)l - maxv);
            const double logsumexp = maxv + std::log(sum);
            result.nll -= (double)logits[target] - logsumexp;
            // The last token is a target only; its next-token logits are unused.
            if (i + 1 < begin + count) logits = model.step((int)target);
        }
        result.used_tokens += count;
        result.scored_tokens += count - 1;
        result.chunks++;
        begin += count;
    }
    return result;
}

} // namespace infer
