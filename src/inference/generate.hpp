#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#include "model/runtime.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"
#include "inference/spec.hpp"

// Generation driven over the model and tokenizer, one sampled token a step, or with drafts several a verify (docs/SPECULATIVE.md, section 3).

namespace infer {

// Generate tokens starting from `logits` (the prediction after the last fed token), stopping at eos unless gp.ignore_eos masks it.
// Returns generated ids (excluding the eos token).
// Text callbacks run synchronously, each token's text before the next step, and may split a UTF-8 character between chunks.
// Every generated token is fed but an eos and one that completes the stop text, the last one the limit allows included, so the model's history is the same however the tokens were generated.
// With `drafting`, each round verifies the last pick and the proposer's drafts in one pass of generated tokens, keeps the rows whose picks equalled their drafts and retracts the rest: the tokens, the draws and the history are those without drafts.
inline std::vector<uint32_t> generate(infer::Model& model, bpe::Tokenizer& tok,
                                      const infer::GenParams& gp, infer::RNG& rng,
                                      std::vector<float> logits,
                                      const std::function<void(const std::string&)>& emit = {},
                                      spec::Drafting* drafting = nullptr) {
    std::vector<uint32_t> gen;
    std::string decoded;
    if (gp.max_tokens <= 0) return gen;
    // How a pick ends the reply: not at all, unfed (an eos, or the token completing the stop text) or fed (the last one the limit allows).
    enum class End { none, unfed, fed } end = End::none;
    auto take = [&](uint32_t id) {
        if (tok.is_eos(id)) { end = End::unfed; return false; }
        gen.push_back(id);
        const std::string text = tok.decode({ id });
        if (emit) emit(text);
        decoded += text;
        if (!gp.stop.empty() && decoded.find(gp.stop) != std::string::npos) { end = End::unfed; return false; }
        if (gen.size() >= (size_t)gp.max_tokens) { end = End::fed; return false; }
        return true;
    };
    const size_t n_vocab = logits.size();
    uint32_t y = infer::sample(logits.data(), n_vocab, gp, tok.eos_id, gen, rng);
    bool going = take(y);
    if (end == End::fed) model.step((int)y);
    std::vector<uint32_t> drafts, feed, history;
    while (going) {
        // y is the last pick, not yet fed.
        size_t k = 0;
        if (drafting) {
            k = spec::draft_length(drafting->draft_max, (size_t)gp.max_tokens - gen.size(), (size_t)model.context_length() - (size_t)model.n_tokens(),
                                   drafting->acceptance);
            if (k) {
                history = drafting->history;
                history.insert(history.end(), gen.begin(), gen.end());
                drafting->proposer->draft(history, k, drafts);
                // A draft past the vocabulary, and those after it, are not fed: the verify needs nothing a decode does not.
                const auto past = std::find_if(drafts.begin(), drafts.end(), [&](uint32_t id) { return id >= n_vocab; });
                drafts.erase(past, drafts.end());
                k = std::min(k, drafts.size());
            }
        }
        if (!k || !model.mark()) {
            if (drafting) drafting->acceptance.stepped();
            logits = model.step((int)y);   // predict the token after y
            y = infer::sample(logits.data(), n_vocab, gp, tok.eos_id, gen, rng);
            going = take(y);
            if (end == End::fed) model.step((int)y);
            continue;
        }
        const size_t start = (size_t)model.n_tokens();
        feed.assign(1, y);
        feed.insert(feed.end(), drafts.begin(), drafts.begin() + (std::ptrdiff_t)k);
        const float* rows = model.step(feed.data(), feed.size());
        const Accepted a = infer::accept(rows, n_vocab, drafts.data(), k, gp, tok.eos_id, gen, rng, [&](uint32_t id) { return going = take(id); });
        // y and the drafts its picks equalled were fed, and a last pick the reply ends on fed where the verify fed it as its draft.
        const bool last_fed = end == End::fed && a.rows <= k && a.last == drafts[a.rows - 1];
        model.retract(start + a.rows + (last_fed ? 1 : 0));
        drafting->acceptance.verified(a.rows - 1);
        y = a.last;
        if (end == End::fed && !last_fed) model.step((int)y);
    }
    return gen;
}

} // namespace infer
