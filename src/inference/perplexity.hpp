#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference/logprobs.hpp"
#include "model/runtime.hpp"

namespace infer {

// The fewest tokens a window holds: its first token is only context, so it takes a second to score anything.
constexpr int kMinPerplexityWindow = 2;

// One window's scored tokens and the sum of their NLL.
struct PerplexityWindow {
    size_t scored = 0;
    double nll = 0.0;
};

struct PerplexityResult {
    size_t context = 0;   // the window's size in tokens: the one asked for, or the model's context
    size_t used_tokens = 0;
    size_t scored_tokens = 0;
    double nll = 0.0;
    std::vector<PerplexityWindow> windows;   // the chunks, in order; `nll` sums every token itself, so the total does not depend on them

    double mean_nll() const { return nll / (double)scored_tokens; }
};

// The negative log-likelihood of `target` under one row of logits, in double: the server's logprob of it before rounding, negated.
inline double token_nll(const float* logits, size_t vocab, uint32_t target) {
    return -((double)logits[target] - log_sum_exp(logits, vocab));
}

// Windows of `context_size` tokens, each from an empty history. By default a window is scored through the batched passes a prompt takes, logits for every position of a microbatch at once, which is the path prompt processing uses and on a device a different set of kernels from decode's. `per_token` scores it one token at a time through step instead, the decode path; the HF gate runs both so each set of kernels meets the reference.
inline PerplexityResult perplexity(Model& model, const std::vector<uint32_t>& ids,
                                   int context_size = 0, int max_chunks = 0, bool per_token = false) {
    if (ids.size() < (size_t)kMinPerplexityWindow)
        throw std::runtime_error("perplexity: need at least " + std::to_string(kMinPerplexityWindow) + " tokens");
    const int context = context_size == 0 ? model.context_length() : context_size;
    if (context < kMinPerplexityWindow || context > model.context_length())
        throw std::runtime_error("perplexity: context size must be between " + std::to_string(kMinPerplexityWindow) + " and model context " +
                                 std::to_string(model.context_length()));
    if (max_chunks < 0) throw std::runtime_error("perplexity: chunks must be nonnegative");

    PerplexityResult result;
    result.context = (size_t)context;
    for (size_t begin = 0; begin < ids.size();) {
        const size_t count = std::min((size_t)context, ids.size() - begin);
        if (count < (size_t)kMinPerplexityWindow || (max_chunks > 0 && result.windows.size() >= (size_t)max_chunks)) break;
        PerplexityWindow part{count - 1, 0.0};
        auto take = [&](double nll) {
            result.nll += nll;
            part.nll += nll;
        };
        if (!per_token) {
            const std::vector<uint32_t> window(ids.begin() + (std::ptrdiff_t)begin,
                                               ids.begin() + (std::ptrdiff_t)(begin + count));
            model.score(window, [&](size_t pos, const float* logits) {
                // The last position predicts past the window; its logits are unused.
                if (pos + 1 < count) take(token_nll(logits, model.n_vocab(), window[pos + 1]));
            });
        } else {
            model.reset();
            std::vector<float> logits = model.step((int)ids[begin]);
            for (size_t i = begin + 1; i < begin + count; i++) {
                const uint32_t target = ids[i];
                take(token_nll(logits.data(), logits.size(), target));
                // The last token is a target only; its next-token logits are unused.
                if (i + 1 < begin + count) logits = model.step((int)target);
            }
        }
        result.windows.push_back(part);
        result.used_tokens += count;
        result.scored_tokens += count - 1;
        begin += count;
    }
    return result;
}

} // namespace infer
