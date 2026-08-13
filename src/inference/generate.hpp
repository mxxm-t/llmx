#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

#include "model/arch_qwen.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"
#include "inference/chat.hpp"

// High-level inference drivers built on the model + tokenizer: prefill,
// generate, and the generate / perplexity / chat entry points.

namespace infer {

// Feed every id in `ids` through the model (prefill / continue), updating the
// KV cache. Returns the logits predicted by the last token (i.e. the
// distribution over the next token).
inline std::vector<float> prefill(infer::Model& model,
                                  const std::vector<uint32_t>& ids) {
    std::vector<float> logits;
    for (uint32_t id : ids) logits = model.step((int)id);
    return logits;
}

// Find a vocab token whose string contains `sub`, or -1.
inline int find_token_by_substr(const bpe::Tokenizer& tok, const std::string& sub) {
    for (size_t i = 0; i < tok.vocab.size(); i++)
        if (tok.vocab[i].find(sub) != std::string::npos) return (int)i;
    return -1;
}

// Generate tokens starting from `logits` (the prediction after the last fed
// token), stopping at eos. Returns generated ids (excluding the eos token).
// By default the Qwen3 <thinking_start>...<thinking_end> reasoning block is
// hidden; only the final answer is printed. Pass gp.show_thinking to keep it.
inline std::vector<uint32_t> generate(infer::Model& model, bpe::Tokenizer& tok,
                                      const infer::GenParams& gp, infer::RNG& rng,
                                      std::vector<float> logits) {
    uint32_t eos = (uint32_t)((tok.eos_id >= 0) ? tok.eos_id : 0);
    std::vector<uint32_t> gen;
    std::string decoded;
    for (int t = 0; t < gp.max_tokens; t++) {
        uint32_t id = infer::sample(logits, gp.temp, gp.top_k, gp.top_p, gp.penalty, gen, rng);
        if (id == eos) break;
        gen.push_back(id);
        decoded += tok.decode({ id });
        if (!gp.stop.empty() && decoded.find(gp.stop) != std::string::npos) break;
        logits = model.step((int)id); // predict token after `id`
    }

    size_t begin = 0;
    size_t end = gen.size();
    if (!gp.show_thinking) {
        int tstart = find_token_by_substr(tok, "thinking_start");
        int tend   = find_token_by_substr(tok, "thinking_end");
        int astart = find_token_by_substr(tok, "answer_start");
        int aend   = find_token_by_substr(tok, "answer_end");
        if (tstart >= 0) {
            auto ts = std::find(gen.begin(), gen.end(), (uint32_t)tstart);
            if (ts != gen.end()) {
                auto te = std::find(gen.begin(), gen.end(), (uint32_t)tend);
                begin = (te != gen.end()) ? (size_t)(te - gen.begin() + 1) : gen.size();
            }
        }
        if (astart >= 0) {
            auto as = std::find(gen.begin(), gen.end(), (uint32_t)astart);
            if (as != gen.end()) begin = (size_t)(as - gen.begin() + 1);
        }
        if (aend >= 0) {
            auto ae = std::find(gen.begin(), gen.end(), (uint32_t)aend);
            if (ae != gen.end()) end = (size_t)(ae - gen.begin());
        }
    }

    for (size_t i = begin; i < end; i++)
        std::cout << tok.decode({ gen[i] }) << std::flush;
    std::cout << "\n";
    return gen;
}

} // namespace infer
