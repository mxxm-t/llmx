#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#include "model/arch_qwen.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"

// Generation driven over the model and tokenizer, one sampled token a step.

namespace infer {

// Generate tokens starting from `logits` (the prediction after the last fed token), stopping at eos.
// Returns generated ids (excluding the eos token).
// Text callbacks run synchronously, each token's text before the next step, and may split a UTF-8 character between chunks.
inline std::vector<uint32_t> generate(infer::Model& model, bpe::Tokenizer& tok,
                                      const infer::GenParams& gp, infer::RNG& rng,
                                      std::vector<float> logits,
                                      const std::function<void(const std::string&)>& emit = {}) {
    std::vector<uint32_t> gen;
    std::string decoded;
    for (int t = 0; t < gp.max_tokens; t++) {
        uint32_t id = infer::sample(logits, gp.temp, gp.top_k, gp.top_p, gp.penalty, gen, rng);
        if (tok.is_eos(id)) break;
        gen.push_back(id);
        const std::string text = tok.decode({ id });
        if (emit) emit(text);
        decoded += text;
        if (!gp.stop.empty() && decoded.find(gp.stop) != std::string::npos) break;
        logits = model.step((int)id); // predict token after `id`
    }

    return gen;
}

} // namespace infer
