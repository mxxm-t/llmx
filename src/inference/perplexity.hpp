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

inline PerplexityResult perplexity(Model& model, const std::vector<uint32_t>& ids,
                                   int context_size = 0, int max_chunks = 0) {
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
