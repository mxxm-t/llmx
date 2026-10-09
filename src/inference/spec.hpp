#pragma once
#include <algorithm>
#include <cmath>
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

// The rows of a verify of the drafts `drafts[0..k)`, row i the logits after the last pick and the first i drafts, sampled in order exactly as the run without drafts samples: one call of `sample` a row with the request's own settings.
// `pick` takes each token as that run does and says whether the request goes on, and sampling stops at the first pick that ends the request or differs from its draft, or after row k.
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
    // `seq` is the history's sequence in the model, or null for the model's own, as `generate` runs it.
    virtual void draft(Sequence* seq, const std::vector<uint32_t>& history, size_t k, std::vector<uint32_t>& out) = 0;

    // One history's draft among several: its sequence in the model, its history, the most drafts it takes and its drafts.
    struct Ask {
        Sequence* seq = nullptr;
        std::vector<uint32_t> history;
        size_t k = 0;
        std::vector<uint32_t> out;
    };
    // Each of `n` asks' drafts, each what draft gives it alone; a proposer that drafts several histories at once does so here.
    virtual void draft_all(Ask* asks, size_t n) {
        for (size_t i = 0; i < n; ++i) draft(asks[i].seq, asks[i].history, asks[i].k, asks[i].out);
    }
};

// Prompt lookup: the tokens that followed the latest earlier occurrence of the history's last three tokens, else two, else one.
class Lookup final : public Proposer {
public:
    void draft(Sequence*, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
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
    void draft(Sequence* seq, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        if (h.empty()) return;
        if (seq) model_.draft(*seq, h.back(), k, out);
        else model_.draft(h.back(), k, out);
    }
    // Every ask's chain in one batch of the model's (Model::draft), so the drafter's weights and the head are read once a step for all of them; each ask names its sequence.
    void draft_all(Ask* asks, size_t n) override {
        batch_.clear();
        for (size_t i = 0; i < n; ++i) {
            asks[i].out.clear();
            if (!asks[i].history.empty()) batch_.push_back({asks[i].seq, asks[i].history.back(), asks[i].k, &asks[i].out});
        }
        model_.draft(batch_.data(), batch_.size());
    }

private:
    Model& model_;
    std::vector<Model::DraftAsk> batch_;
};

// A draft model (docs/SPECULATIVE.md, step 6): a model of its own sharing the target's tokenizer, drafting greedily, ties to the lowest id, on its own history, which follows the one it is given.
// A draft takes that history back to what it shares with the given one, feeds the rest in one pass (keeping the state there on a model that keeps one), then drafts a token a step.
class DraftModel final : public Proposer {
public:
    explicit DraftModel(Model& model) : model_(model) {}
    void draft(Sequence* seq, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        if (seq) throw std::logic_error("inference: a draft model drafts for one history, not a sequence of another model");
        if (h.empty() || !k) return;
        size_t shared = 0;
        while (shared < fed_.size() && shared < h.size() && fed_[shared] == h[shared]) ++shared;
        // The last token is fed again where the history holds no more, since its logits are what the first draft is drawn from.
        shared = std::min(shared, h.size() - 1);
        if (shared < fed_.size()) fed_.resize(model_.retract(shared));
        if (h.size() + k > (size_t)model_.context_length()) return;
        const std::vector<uint32_t> rest(h.begin() + (std::ptrdiff_t)fed_.size(), h.end());
        // What a verify kept and its pick, at most a verify's rows, go in as generated tokens through the decode kernels; a longer rest, a new prompt, goes through the prompt path.
        std::vector<float> logits;
        if (rest.size() <= size_t(kMaxDrafts) + 1) {
            const float* rows = model_.step(rest.data(), rest.size());
            logits.assign(rows + (rest.size() - 1) * model_.n_vocab(), rows + rest.size() * model_.n_vocab());
        } else {
            logits = model_.prefill(rest);
        }
        fed_ = h;
        if (model_.keeps_state()) model_.keep();
        for (;;) {
            const uint32_t d = (uint32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            out.push_back(d);
            if (out.size() == k) return;
            logits = model_.step((int)d);
            fed_.push_back(d);
        }
    }

private:
    Model& model_;
    std::vector<uint32_t> fed_;   // the tokens the model's history holds
};

// A request's acceptance figure (docs/SPECULATIVE.md, section 3): a running average of the drafts its verifies kept, each moving it an eighth of the way to what that one kept.
// Below kBreakEven the request drafts nothing for its next 16 tokens, and it counts by draft position the drafts its verifies fed and kept.
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
    // The chance a verify keeps its draft at position i: the share these verifies kept there, weighed with kPrior verifies at `prior`'s share, or at one half where neither has fed one.
    double keeps(size_t i, const Acceptance& prior) const {
        const double p = i < prior.drafted_.size() && prior.drafted_[i] ? (double)prior.kept_[i] / (double)prior.drafted_[i] : 0.5;
        const double n = i < drafted_.size() ? (double)drafted_[i] : 0.0, k = i < kept_.size() ? (double)kept_[i] : 0.0;
        return (k + kPrior * p) / (n + kPrior);
    }

private:
    static constexpr double kBreakEven = 0.5, kWeight = 8.0, kPrior = 4.0;
    static constexpr size_t kRest = 16;
    double average_ = 2.0;
    size_t rest_ = 0;
    std::vector<size_t> drafted_, kept_;
};

// How many drafts a request verifies next, the one rule for every caller: at most `draft_max`, one fewer than the tokens it may still generate and than the positions its context has left, and none while it rests (Acceptance).
// It is also at most `columns`, the rows the pass's decode kernels read each weight once for, past which a draft row costs a weight read of its own (docs/SPECULATIVE.md, section 3).
inline size_t draft_length(size_t draft_max, size_t tokens_left, size_t context_left, size_t columns, const Acceptance& acceptance) {
    if (acceptance.resting() || !tokens_left || !context_left) return 0;
    return std::min({draft_max, tokens_left - 1, context_left - 1, columns});
}

// What a pass of generated rows costs, `base_ms` and `row_ms` a row, and a step of the drafter's chains, which run before the pass, `step_ms` and `step_row_ms` a chain.
// The pass's part is unknown until passes of different rows have been measured (PassTimes), and `seen_rows` is the rows of those measured, 0 for none.
struct PassCost {
    double base_ms = 0, row_ms = 0, step_ms = 0, step_row_ms = 0, seen_rows = 0;
    bool known = false;
};

// The pass cost measured as passes run: lines through the passes of generated rows (rows against milliseconds) and through the chains' steps, each measurement weighing kForget of the one after it.
// A line is known while the rows it was fitted to spread by half a row or more, so a width run long enough is forgotten and measured again; an unknown chain line is its mean step.
class PassTimes {
public:
    void pass(size_t rows, double ms) { passes_.add((double)rows, ms); }
    void chain(size_t steps, size_t chains, double ms) {
        if (steps) chains_.add((double)chains, ms / (double)steps);
    }
    // What a pass of `rows` generated rows takes, from the line where it is known and else the mean of those seen; 0 before any.
    double pass_ms(double rows) const {
        double base = 0, row = 0;
        return passes_.fit(base, row) ? base + row * rows : passes_.mean();
    }
    PassCost cost() const {
        PassCost c;
        if (!chains_.fit(c.step_ms, c.step_row_ms)) c.step_ms = chains_.mean();
        c.known = passes_.fit(c.base_ms, c.row_ms);
        if (!c.known) c.seen_rows = passes_.mean_x();
        return c;
    }

private:
    static constexpr double kForget = 0.99;
    struct Line {
        double w = 0, x = 0, y = 0, xx = 0, xy = 0;
        void add(double a, double b) {
            w = w * kForget + 1;
            x = x * kForget + a;
            y = y * kForget + b;
            xx = xx * kForget + a * a;
            xy = xy * kForget + a * b;
        }
        double mean() const { return w ? y / w : 0.0; }
        double mean_x() const { return w ? x / w : 0.0; }
        // The line's value at 0 and its slope, neither below 0.
        bool fit(double& at0, double& slope) const {
            if (w < 2) return false;
            const double mx = x / w, my = y / w, var = xx / w - mx * mx;
            if (var < 0.25) return false;
            slope = std::max(0.0, (xy / w - mx * my) / var);
            at0 = std::max(0.0, my - slope * mx);
            return true;
        }
    };
    Line passes_, chains_;
};

// The drafts each of a pass's drafting requests verifies (docs/SPECULATIVE.md, section 3), from the pass cost `c`, the pass's `decoders` and each request's chance of keeping each draft its cap allows (`keeps[i]`), into `takes`.
// The depth taken is the one with the most tokens a millisecond, request i getting min(K, its cap) drafts, or none where no depth beats the pass without drafts; free rows or an unknown cost give every request its cap.
inline void draft_depths(const PassCost& c, size_t decoders, const std::vector<std::vector<double>>& keeps, std::vector<size_t>& takes) {
    takes.assign(keeps.size(), 0);
    size_t depth = 0, capped = decoders;
    for (const auto& k : keeps) depth = std::max(depth, k.size()), capped += k.size();
    if (!c.known && c.seen_rows > 0 && std::fabs(c.seen_rows - (double)capped) < 0.5) return;
    if (!c.known || c.row_ms <= 0) {
        for (size_t i = 0; i < keeps.size(); ++i) takes[i] = keeps[i].size();
        return;
    }
    const double plain_ms = c.base_ms + c.row_ms * (double)decoders, rate = (double)decoders / plain_ms;
    const auto take = [&](size_t K, std::vector<size_t>* out) {
        double tokens = (double)decoders, ms = plain_ms;
        size_t chains = 0;
        for (size_t i = 0; i < keeps.size(); ++i) {
            const size_t k = std::min(K, keeps[i].size());
            double kept = 0;
            for (size_t j = 0; j < k; ++j) kept += keeps[i][j];
            if (!k || kept < rate * c.row_ms * (double)k) continue;
            tokens += kept;
            ms += c.row_ms * (double)k;
            ++chains;
            if (out) (*out)[i] = k;
        }
        if (chains) ms += (double)K * (c.step_ms + c.step_row_ms * (double)chains);
        return tokens / ms;
    };
    double best = rate;
    size_t best_depth = 0;
    for (size_t K = 1; K <= depth; ++K) {
        const double r = take(K, nullptr);
        if (r > best) best = r, best_depth = K;
    }
    if (best_depth) take(best_depth, &takes);
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
