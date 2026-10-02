#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "model/runtime.hpp"
#include "inference/sampler.hpp"

// Speculative decoding (docs/SPECULATIVE.md, section 3): proposers that draft the next tokens, the length of a draft, and the acceptance of a verify's rows, which keeps every token, draw and end what the run without drafts gives.

namespace infer {

// What `accept` ends with: the rows it sampled, at least one, and the last pick.
struct Accepted {
    size_t rows;
    uint32_t last;
};

// The rows of a verify of the drafts `drafts[0..k)`, row i the logits after the last pick and the first i drafts, sampled in order exactly as the run without drafts samples a token after each: one call of `sample` a row, with the request's own settings, history and generator.
// `pick` takes each token as the run without drafts does and says whether the request goes on (an end token, a stop text or the length limit end it), and it adds the token to `gen` where that run would.
// Sampling stops at the first pick that ends the request or differs from its draft, or after row k: every earlier pick equalled its draft, so it was fed as that run feeds it.
template <class Pick>
Accepted accept(const float* rows, size_t n_vocab, const uint32_t* drafts, size_t k, const Sampling& s, int32_t end, const std::vector<uint32_t>& gen,
                RNG& rng, Pick&& pick) {
    for (size_t i = 0;; ++i) {
        const uint32_t y = sample(rows + i * n_vocab, n_vocab, s, end, gen, rng);
        if (!pick(y) || i == k || y != drafts[i]) return Accepted{i + 1, y};
    }
}

namespace spec {

// The most drafts a verify takes: with the last pick 64 rows, the most generated tokens a pass's kernels are held to give the bits of single steps for (AGENTS.md, backend-vulkan).
inline constexpr int kMaxDrafts = 63;

// A drafter of the next tokens of a history: it proposes, the verify decides, so a wrong proposal costs time and never output.
// A proposer drafts deterministically and never touches a request's generator.
class Proposer {
public:
    virtual ~Proposer() = default;
    // Up to `k` tokens likely to follow `history`, in `out`; fewer, or none, where it has nothing to propose.
    virtual void draft(const std::vector<uint32_t>& history, size_t k, std::vector<uint32_t>& out) = 0;
};

// Prompt lookup: the tokens that followed the latest earlier occurrence of the history's last three tokens, else two, else one.
class Lookup final : public Proposer {
public:
    void draft(const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        const size_t size = h.size();
        for (size_t n = std::min<size_t>(3, size); n >= 1 && k; --n) {
            const auto tail = h.end() - (std::ptrdiff_t)n;
            // An occurrence that ends before the history's last token, so a token follows it.
            for (size_t i = size - n; i-- > 0;) {
                if (!std::equal(h.begin() + (std::ptrdiff_t)i, h.begin() + (std::ptrdiff_t)(i + n), tail)) continue;
                for (size_t j = i + n; j < size && out.size() < k; ++j) out.push_back(h[j]);
                return;
            }
        }
    }
};

// The embedded drafter a model was loaded with (docs/SPECULATIVE.md, section 7): its chain of drafts after the history's last pick, which the model drafts from the row its history carries.
class Embedded final : public Proposer {
public:
    explicit Embedded(Model& model) : model_(model) {}
    void draft(const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        if (!h.empty()) model_.draft(h.back(), k, out);
    }

private:
    Model& model_;
};

// A request's one acceptance figure (docs/SPECULATIVE.md, section 3): the average of the drafts its verifies kept, each verify moving it an eighth of the way to what that one kept, and below kBreakEven the request drafts nothing for its next 16 tokens, then verifies once more.
// It also counts, by draft position, the drafts its verifies fed and those they kept.
class Acceptance {
public:
    // A verify that fed `fed` drafts and kept the first `kept` of them.
    void verified(size_t kept, size_t fed) {
        average_ += ((double)kept - average_) / kWeight;
        if (average_ < kBreakEven) rest_ = kRest;
        if (drafted_.size() < fed) drafted_.resize(fed), kept_.resize(fed);
        for (size_t i = 0; i < fed; ++i) {
            ++drafted_[i];
            kept_[i] += i < kept;
        }
    }
    // A token generated without drafts.
    void stepped() {
        if (rest_) --rest_;
    }
    bool resting() const { return rest_ > 0; }
    const std::vector<size_t>& drafted() const { return drafted_; }
    const std::vector<size_t>& kept() const { return kept_; }

private:
    static constexpr double kBreakEven = 0.5, kWeight = 8.0;
    static constexpr size_t kRest = 16;
    double average_ = 2.0;
    size_t rest_ = 0;
    std::vector<size_t> drafted_, kept_;
};

// How many drafts a request verifies next, the one rule for every caller: at most `draft_max`, one fewer than the tokens it may still generate and than the positions its context has left, since the verify feeds the last pick and every draft, and none while it rests (Acceptance).
inline size_t draft_length(size_t draft_max, size_t tokens_left, size_t context_left, const Acceptance& acceptance) {
    if (acceptance.resting() || !tokens_left || !context_left) return 0;
    return std::min({draft_max, tokens_left - 1, context_left - 1});
}

// A request's drafting: its proposer, its most drafts a verify, the tokens its history held before it generated, and its acceptance.
struct Drafting {
    Proposer* proposer = nullptr;
    size_t draft_max = 0;
    std::vector<uint32_t> history;
    Acceptance acceptance;
};

} // namespace spec
} // namespace infer
