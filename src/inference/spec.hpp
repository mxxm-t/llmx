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

// The most drafts a request's own history gives a verify where that is more than its proposer's `--draft-max`: with the last pick eight rows, the price deciding each pass how many of them pay (docs/SPECULATIVE.md, section 8).
inline constexpr size_t kOwnDrafts = 7;

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
    // Whether a history's own convincing continuation is drafted in this proposer's place where it has one (lookup_first), as the commands ask of a proposer that runs a model.
    virtual bool after_lookup() const { return false; }
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

    // The match that stands in for a model's draft, in tokens; docs/SPECULATIVE.md, section 8, has the measurement it comes from.
    static constexpr size_t kConvincing = 8;
    // Up to `k` tokens that followed the latest earlier occurrence of the history's last kConvincing tokens, in `out`, or none where they occur nowhere earlier.
    // An occurrence that overlaps the history's end goes on into the tokens drafted, as a repeating run does.
    static void convincing(const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) {
        out.clear();
        const size_t size = h.size(), n = kConvincing;
        if (size <= n) return;
        const uint32_t last = h[size - 1];
        for (size_t i = size - n; i-- > 0;) {
            if (h[i + n - 1] != last || !std::equal(h.begin() + (std::ptrdiff_t)i, h.begin() + (std::ptrdiff_t)(i + n - 1), h.end() - (std::ptrdiff_t)n)) continue;
            for (size_t j = i + n; out.size() < k; ++j) out.push_back(j < size ? h[j] : out[j - size]);
            return;
        }
    }
};

// The embedded drafter a model was loaded with (docs/SPECULATIVE.md, section 7): its chain of drafts after the history's last pick, which the model drafts from the row its history carries.
class Embedded final : public Proposer {
public:
    explicit Embedded(Model& model, bool after_lookup = false) : model_(model), after_lookup_(after_lookup) {}
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
    bool after_lookup() const override { return after_lookup_; }

private:
    Model& model_;
    const bool after_lookup_;
    std::vector<Model::DraftAsk> batch_;
};

// A draft model (docs/SPECULATIVE.md, step 6): a model of its own sharing the target's tokenizer, drafting greedily, ties to the lowest id, on its own history, which follows the one it is given.
// A draft takes that history back to what it shares with the given one, feeds the rest in one pass (keeping the state there on a model that keeps one), then drafts a token a step.
class DraftModel final : public Proposer {
public:
    explicit DraftModel(Model& model, bool after_lookup = false) : model_(model), after_lookup_(after_lookup) {}
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
    bool after_lookup() const override { return after_lookup_; }

private:
    Model& model_;
    const bool after_lookup_;
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
    // `n` tokens generated without this acceptance's drafts.
    void stepped(size_t n = 1) { rest_ -= std::min(rest_, n); }
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

// Where a draft comes from: the request's own history ahead of a model proposer (lookup_first), or its proposer.
enum class Source : uint8_t { lookup, proposer };

// A request's drafting record: an acceptance for each source, so neither's misses rest the other, and the source of the draft its verify in flight feeds.
struct Record {
    Acceptance of[2];
    Source drafted = Source::proposer;
    Acceptance& operator[](Source s) { return of[(size_t)s]; }
    const Acceptance& operator[](Source s) const { return of[(size_t)s]; }
    // A verify that fed `fed` drafts of the source `drafted` and kept the first `kept`, the tokens it generated counted against the other source's rest.
    void verified(size_t kept, size_t fed) {
        of[(size_t)drafted].verified(kept, fed);
        of[1 - (size_t)drafted].stepped(kept + 1);
    }
    // A token generated without drafts.
    void stepped() {
        for (Acceptance& a : of) a.stepped();
    }
};

// How many drafts a request verifies next, the one rule for every caller: at most `draft_max`, one fewer than the tokens it may still generate and than the positions its context has left, and none while it rests (Acceptance).
// It is also at most `columns`, the rows the pass's decode kernels read each weight once for, past which a draft row costs a weight read of its own (docs/SPECULATIVE.md, section 3).
inline size_t draft_length(size_t draft_max, size_t tokens_left, size_t context_left, size_t columns, const Acceptance& acceptance) {
    if (acceptance.resting() || !tokens_left || !context_left) return 0;
    return std::min({draft_max, tokens_left - 1, context_left - 1, columns});
}

// A request's draft ahead of its proposer (docs/SPECULATIVE.md, section 8): the convincing continuation of its own `history` (Lookup::convincing), up to the drafts draft_length allows lookup's acceptance, in `out`.
// It sets the record's source and returns the drafts; with none the source is the proposer's, which the caller asks under that source's acceptance.
inline size_t lookup_first(Record& r, size_t draft_max, size_t tokens_left, size_t context_left, size_t columns, const std::vector<uint32_t>& history,
                           std::vector<uint32_t>& out) {
    out.clear();
    r.drafted = Source::proposer;
    const size_t k = draft_length(std::max(draft_max, kOwnDrafts), tokens_left, context_left, columns, r[Source::lookup]);
    if (k) Lookup::convincing(history, k, out);
    if (!out.empty()) r.drafted = Source::lookup;
    return out.size();
}

// What a pass of generated rows costs, `base_ms` and `row_ms` a row, and a step of the drafter's chains, which run before the pass, `step_ms` and `step_row_ms` a chain.
// The pass's part is unknown until passes of different rows have been measured (PassTimes), and `seen_rows` is the rows of those measured, 0 for none.
struct PassCost {
    double base_ms = 0, row_ms = 0, step_ms = 0, step_row_ms = 0, seen_rows = 0;
    bool known = false;
    // A pass's time by its rows where passes of that many have run, 0 elsewhere: a pass's cost steps where its kernels change build, which the line cannot hold.
    std::vector<double> width_ms;
    // The line's price of a pass of `rows` generated rows, which is the price of the pass without drafts and gives the least a row may cost.
    double line_ms(size_t rows) const { return base_ms + row_ms * (double)rows; }
    // What a pass widened by drafts costs: the line's, and what its width measured beyond that, so a measured width can add a step's cost and never take a row's away.
    double pass_ms(size_t rows) const { return std::max(line_ms(rows), rows < width_ms.size() ? width_ms[rows] : 0.0); }
};

// The pass cost measured as passes run: lines through the passes of generated rows (rows against milliseconds) and through the chains' steps, each measurement weighing kForget of the one after it.
// A line is known while the rows it was fitted to spread by half a row or more, so a width run long enough is forgotten and measured again; an unknown chain line is its mean step.
class PassTimes {
public:
    void pass(size_t rows, double ms) {
        passes_.add((double)rows, ms);
        if (rows < kWidths) {
            widths_[rows].w = widths_[rows].w * kForget + 1;
            widths_[rows].ms = widths_[rows].ms * kForget + ms;
        }
    }
    void chain(size_t steps, size_t chains, double ms) {
        if (steps) chains_.add((double)chains, ms / (double)steps);
    }
    // What a pass of `rows` generated rows takes, from the line where it is known and else the mean of those seen; 0 before any.
    // The widths' own means reach the draft price through cost() alone.
    double pass_ms(double rows) const {
        double base = 0, row = 0;
        return passes_.fit(base, row) ? base + row * rows : passes_.mean();
    }
    PassCost cost() const {
        PassCost c;
        if (!chains_.fit(c.step_ms, c.step_row_ms)) c.step_ms = chains_.mean();
        c.known = passes_.fit(c.base_ms, c.row_ms);
        if (!c.known) c.seen_rows = passes_.mean_x();
        c.width_ms.assign(kWidths, 0.0);
        for (size_t i = 0; i < kWidths; ++i)
            if (widths_[i].w > 0) c.width_ms[i] = widths_[i].ms / widths_[i].w;
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
    // A forgetting mean of the passes of each width up to a verify's most rows and a row.
    static constexpr size_t kWidths = size_t(kMaxDrafts) + 2;
    struct Width {
        double w = 0, ms = 0;
    };
    Width widths_[kWidths];
};

// The drafts each of a pass's drafting requests verifies (docs/SPECULATIVE.md, section 3), from the pass cost `c`, the pass's `decoders` and each request's chance of keeping each draft its cap allows (`keeps[i]`), into `takes`.
// The depth taken gives the most tokens a millisecond, request i getting min(K, its cap) drafts, or none where no depth beats the pass without drafts; free rows or an unknown cost give each its cap, and `chained` marks a chain's requests.
inline void draft_depths(const PassCost& c, size_t decoders, const std::vector<std::vector<double>>& keeps, std::vector<size_t>& takes,
                         const std::vector<uint8_t>* chained = nullptr) {
    takes.assign(keeps.size(), 0);
    size_t depth = 0, capped = decoders;
    for (const auto& k : keeps) depth = std::max(depth, k.size()), capped += k.size();
    if (!c.known && c.seen_rows > 0 && std::fabs(c.seen_rows - (double)capped) < 0.5) return;
    if (!c.known || c.row_ms <= 0) {
        for (size_t i = 0; i < keeps.size(); ++i) takes[i] = keeps[i].size();
        return;
    }
    const double plain_ms = c.line_ms(decoders), rate = (double)decoders / plain_ms;
    // A depth is weighed with the chain and without it, where only the requests that need no chain draft, so a dear chain does not take their drafts with it.
    const auto take = [&](size_t K, bool chain, std::vector<size_t>* out) {
        const auto drafts = [&](size_t i) {
            const bool chains_it = !chained || (*chained)[i];
            return chains_it && !chain ? size_t(0) : std::min(K, keeps[i].size());
        };
        // A row's cost at this depth is the pass's with every request drafting over its rows, so a step in the pass's cost is spread over the rows that cross it.
        size_t asked = 0;
        for (size_t i = 0; i < keeps.size(); ++i) asked += drafts(i);
        if (!asked) return rate;
        const double row_ms = (c.pass_ms(decoders + asked) - plain_ms) / (double)asked;
        double tokens = (double)decoders;
        size_t chains = 0, steps = 0, rows = decoders;
        for (size_t i = 0; i < keeps.size(); ++i) {
            const size_t k = drafts(i);
            double kept = 0;
            for (size_t j = 0; j < k; ++j) kept += keeps[i][j];
            if (!k || kept < rate * row_ms * (double)k) continue;
            tokens += kept;
            rows += k;
            if (!chained || (*chained)[i]) ++chains, steps = std::max(steps, k);
            if (out) (*out)[i] = k;
        }
        double ms = c.pass_ms(rows);
        if (chains) ms += (double)steps * (c.step_ms + c.step_row_ms * (double)chains);
        return tokens / ms;
    };
    double best = rate;
    size_t best_depth = 0;
    bool best_chain = true;
    for (size_t K = 1; K <= depth; ++K)
        for (bool chain : {true, false}) {
            const double r = take(K, chain, nullptr);
            if (r > best) best = r, best_depth = K, best_chain = chain;
        }
    if (best_depth) take(best_depth, best_chain, &takes);
}

// A request's drafting: its proposer, its most drafts a verify, the tokens its history held before it generated, and its record.
struct Drafting {
    Proposer* proposer = nullptr;
    size_t draft_max = 0;
    std::vector<uint32_t> history;
    Record record;
};

} // namespace spec
} // namespace infer
