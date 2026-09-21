#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

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
    return model.prefill(ids);
}

// Find a vocab token whose string contains `sub`, or -1.
inline int find_token_by_substr(const bpe::Tokenizer& tok, const std::string& sub) {
    for (size_t i = 0; i < tok.vocab.size(); i++)
        if (tok.vocab[i].find(sub) != std::string::npos) return (int)i;
    return -1;
}

// Generate tokens starting from `logits` (the prediction after the last fed
// token), stopping at eos. Returns generated ids (excluding the eos token).
// Text callbacks run synchronously and may split a UTF-8 character between chunks.
// Legacy reasoning filters need the full sequence because later markers can discard earlier text.
inline std::vector<uint32_t> generate(infer::Model& model, bpe::Tokenizer& tok,
                                      const infer::GenParams& gp, infer::RNG& rng,
                                      std::vector<float> logits,
                                      const std::function<void(const std::string&)>& emit = {}) {
    // A model without an EOS id has no stop token at all. Folding that to 0
    // made token zero, which is an ordinary token, end every generation.
    const bool has_eos = tok.eos_id >= 0;
    const uint32_t eos = has_eos ? (uint32_t)tok.eos_id : 0;
    const int tstart = gp.show_thinking ? -1 : find_token_by_substr(tok, "thinking_start");
    const int tend = gp.show_thinking ? -1 : find_token_by_substr(tok, "thinking_end");
    const int astart = gp.show_thinking ? -1 : find_token_by_substr(tok, "answer_start");
    const int aend = gp.show_thinking ? -1 : find_token_by_substr(tok, "answer_end");
    const bool buffered = tstart >= 0 || astart >= 0 || aend >= 0;
    std::vector<uint32_t> gen;
    std::string decoded;
    for (int t = 0; t < gp.max_tokens; t++) {
        uint32_t id = infer::sample(logits, gp.temp, gp.top_k, gp.top_p, gp.penalty, gen, rng);
        if (has_eos && id == eos) break;
        gen.push_back(id);
        const std::string text = tok.decode({ id });
        if (emit && !buffered) emit(text);
        decoded += text;
        if (!gp.stop.empty() && decoded.find(gp.stop) != std::string::npos) break;
        logits = model.step((int)id); // predict token after `id`
    }

    size_t begin = 0;
    size_t end = gen.size();
    if (buffered) {
        // Both markers have to exist for the filter to mean anything. With a
        // start and no end, the search for (uint32_t)-1 failed and the whole
        // reply was dropped rather than the reasoning block.
        if (tstart >= 0 && tend >= 0) {
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

    if (emit && buffered)
        for (size_t i = begin; i < end; i++) emit(tok.decode({ gen[i] }));
    return gen;
}

} // namespace infer
