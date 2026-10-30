#pragma once
#include <algorithm>
#include <array>
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

// The rows of a history's images, [first, end) each in order, as its sequence's runs give them (Model::image_rows).
using ImageRows = std::vector<std::pair<size_t, size_t>>;

// A drafter of the next tokens of a history: it proposes, the verify decides, so a wrong proposal costs time and never output.
// A proposer drafts deterministically and never touches a request's generator.

class Proposer {
public:
    virtual ~Proposer() = default;
    // Up to `k` tokens likely to follow `history`, in `out`; fewer, or none, where it has nothing to propose.
    // `seq` is the history's sequence in the model, or null for the model's own, as `generate` runs it; `images` are the history's image rows, which are not tokens to draft from.
    virtual void draft(Sequence* seq, const std::vector<uint32_t>& history, size_t k, std::vector<uint32_t>& out, const ImageRows& images = {}) = 0;

    // One history's draft among several: its sequence in the model, its history and its image rows, the most drafts it takes and its drafts.
    struct Ask {
        Sequence* seq = nullptr;
        std::vector<uint32_t> history;
        ImageRows images;
        size_t k = 0;
        std::vector<uint32_t> out;
    };
    // Each of `n` asks' drafts, each what draft gives it alone; a proposer that drafts several histories at once does so here.
    virtual void draft_all(Ask* asks, size_t n) {
        for (size_t i = 0; i < n; ++i) draft(asks[i].seq, asks[i].history, asks[i].k, asks[i].out, asks[i].images);
    }
    // Whether a history's own convincing continuation is drafted in this proposer's place where it has one (lookup_first), as the commands ask of a proposer that runs a model.
    virtual bool after_lookup() const { return false; }
    // Whether its drafts run on the model's devices, as they may unless it says not: on the head stage's alone, which a server records on that stage's thread, or on others too, for which it holds every stage.
    virtual bool on_device() const { return true; }
    virtual bool on_head() const { return false; }
    // For a proposer on the head's devices, the asks' chain begun for a caller that records it on the head stage's thread behind the work queued there and ends it (Model::Chain); null where no ask drafts.
    virtual std::unique_ptr<Model::Chain> begin_all(Ask*, size_t) { throw std::logic_error("inference: a proposer without a chain on the head's devices"); }
};

// Whether row `at` of a history is an image's.
inline bool image_row(const ImageRows& images, size_t at) {
    for (const auto& [first, end] : images)
        if (first <= at && at < end) return true;
    return false;
}
// Whether rows [from, to) hold no image row.
inline bool text_rows(const ImageRows& images, size_t from, size_t to) {
    for (const auto& [first, end] : images)
        if (first < to && from < end) return false;
    return true;
}

// Prompt lookup: the tokens that followed the latest earlier occurrence of the history's last three tokens, else two, else one.
// An image's rows are no tokens (docs/VISION.md, section 5): no occurrence or tail holds one, and a draft stops where one begins.
class Lookup final : public Proposer {
public:
    void draft(Sequence*, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out, const ImageRows& images = {}) override {
        out.clear();
        const size_t size = h.size();
        for (size_t n = std::min<size_t>(3, size); n >= 1 && k; --n) {
            const auto tail = h.end() - (std::ptrdiff_t)n;
            if (!text_rows(images, size - n, size)) continue;
            // An occurrence that ends before the history's last token, so a token follows it.
            for (size_t i = size - n; i-- > 0;) {
                if (!std::equal(h.begin() + (std::ptrdiff_t)i, h.begin() + (std::ptrdiff_t)(i + n), tail) || !text_rows(images, i, i + n)) continue;
                for (size_t j = i + n; j < size && out.size() < k && !image_row(images, j); ++j) out.push_back(h[j]);
                return;
            }
        }
    }

    // The match that stands in for a model's draft, in tokens; docs/SPECULATIVE.md, section 8, has the measurement it comes from.
    static constexpr size_t kConvincing = 8;
    // The longest match a draft's acceptance is told apart by (match_bucket).
    static constexpr size_t kLongestMatch = 32;
    // Up to `k` tokens that followed the latest earlier occurrence of the history's last kConvincing tokens, in `out`, an occurrence that overlaps the history's end going on into the tokens drafted.
    // Returns the match's length, the occurrence extended backwards while it still matches the history's end, at most kLongestMatch, or 0 for none.
    static size_t convincing(const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out, const ImageRows& images = {}) {
        out.clear();
        const size_t size = h.size(), n = kConvincing;
        if (size <= n || !k || !text_rows(images, size - n, size)) return 0;
        const uint32_t last = h[size - 1];
        for (size_t i = size - n; i-- > 0;) {
            if (h[i + n - 1] != last || !std::equal(h.begin() + (std::ptrdiff_t)i, h.begin() + (std::ptrdiff_t)(i + n - 1), h.end() - (std::ptrdiff_t)n) ||
                !text_rows(images, i, i + n))
                continue;
            for (size_t j = i + n; out.size() < k && !image_row(images, j); ++j) out.push_back(j < size ? h[j] : out[j - size]);
            size_t run = n;
            while (run < kLongestMatch && i + n > run && h[i + n - 1 - run] == h[size - 1 - run] && !image_row(images, i + n - 1 - run)) ++run;
            return run;
        }
        return 0;
    }
    bool on_device() const override { return false; }
};

// The embedded drafter a model was loaded with (docs/SPECULATIVE.md, section 7): its chain of drafts after the history's last pick, which the model drafts from the row its history carries.
class Embedded final : public Proposer {
public:
    explicit Embedded(Model& model, bool after_lookup = false) : model_(model), after_lookup_(after_lookup) {}
    void draft(Sequence* seq, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out, const ImageRows& = {}) override {
        out.clear();
        if (h.empty()) return;
        if (seq) model_.draft(*seq, h.back(), k, out);
        else model_.draft(h.back(), k, out);
    }
    // Every ask's chain in one batch of the model's (Model::draft), so the drafter's weights and the head are read once a step for all of them; each ask names its sequence.
    void draft_all(Ask* asks, size_t n) override {
        batch(asks, n);
        model_.draft(batch_.data(), batch_.size());
    }
    std::unique_ptr<Model::Chain> begin_all(Ask* asks, size_t n) override {
        batch(asks, n);
        return model_.draft_begin(batch_.data(), batch_.size(), true);
    }
    bool after_lookup() const override { return after_lookup_; }
    bool on_device() const override { return true; }
    bool on_head() const override { return true; }

private:
    void batch(Ask* asks, size_t n) {
        batch_.clear();
        for (size_t i = 0; i < n; ++i) {
            asks[i].out.clear();
            if (!asks[i].history.empty()) batch_.push_back({asks[i].seq, asks[i].history.back(), asks[i].k, &asks[i].out});
        }
    }

    Model& model_;
    const bool after_lookup_;
    std::vector<Model::DraftAsk> batch_;
};

// A draft model (docs/SPECULATIVE.md, step 6): a model of its own sharing the target's tokenizer, drafting greedily, ties to the lowest id, on its own history, which follows the one it is given.
// A draft takes that history back to what it shares with the given one, feeds the rest in one pass (keeping the state there on a model that keeps one), then drafts a token a step.
class DraftModel final : public Proposer {
public:
    explicit DraftModel(Model& model, bool after_lookup = false) : model_(model), after_lookup_(after_lookup) {}
    void draft(Sequence* seq, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out, const ImageRows& = {}) override {
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
        const size_t n_vocab = model_.n_vocab();
        std::vector<float> prompt;
        const float* logits = nullptr;
        if (rest.size() <= size_t(kMaxDrafts) + 1) {
            logits = model_.step(rest.data(), rest.size()) + (rest.size() - 1) * n_vocab;
        } else {
            prompt = model_.prefill(rest);
            logits = prompt.data();
        }
        fed_ = h;
        if (model_.keeps_state()) model_.keep();
        for (;;) {
            const uint32_t d = (uint32_t)(std::max_element(logits, logits + n_vocab) - logits);
            out.push_back(d);
            if (out.size() == k) return;
            logits = model_.step((int)d);
            fed_.push_back(d);
        }
    }
    bool after_lookup() const override { return after_lookup_; }
    bool on_device() const override { return true; }

private:
    Model& model_;
    const bool after_lookup_;
    std::vector<uint32_t> fed_;   // the tokens the model's history holds
};

// What each verify a request's acceptance counts, and each pass or chain step the pass cost measures, weighs of the next, so about the last hundred count and a figure set by a few early ones gives way to later ones.
inline constexpr double kForget = 0.99;

// A request's acceptance figure (docs/SPECULATIVE.md, section 3): a running average of the drafts its verifies kept, each moving it an eighth of the way to what that one kept.
// Below kBreakEven the request drafts nothing for its next 16 tokens, and it counts by draft position the drafts its verifies fed and kept.
class Acceptance {
public:
    // A verify that fed `fed` drafts and kept the first `kept` of them.
    void verified(size_t kept, size_t fed) {
        average_ += ((double)kept - average_) / kWeight;
        if (average_ < kBreakEven) rest_ = kRest;
        if (drafted_.size() < fed) drafted_.resize(fed), kept_.resize(fed), reached_.resize(fed), kept_w_.resize(fed), reached_w_.resize(fed);
        for (size_t i = 0; i < kept_w_.size(); ++i) kept_w_[i] *= kForget, reached_w_[i] *= kForget;
        for (size_t i = 0; i < fed; ++i) {
            ++drafted_[i];
            kept_[i] += i < kept;
            reached_[i] += i <= kept;
            kept_w_[i] += i < kept;
            reached_w_[i] += i <= kept;
        }
    }
    // `n` tokens generated without this acceptance's drafts.
    void stepped(size_t n = 1) { rest_ -= std::min(rest_, n); }
    bool resting() const { return rest_ > 0; }
    const std::vector<size_t>& drafted() const { return drafted_; }
    const std::vector<size_t>& kept() const { return kept_; }
    const std::vector<size_t>& reached() const { return reached_; }
    // The chance a verify keeps its drafts up to position i: the product of each position's chance over the verifies that reached it (docs/SPECULATIVE.md, section 3).
    // Each is weighed with kPrior verifies at `prior`'s, that at the last reached position's (one half for the first draft), and each verify weighs kForget of the next.
    double keeps(size_t i, const Acceptance& prior) const {
        double chance = 1.0, before = 1.0;
        for (size_t j = 0; j <= i; ++j) {
            const double p = (prior.kept_at(j) + kPrior * (j ? before : 0.5)) / (prior.reached_at(j) + kPrior);
            const double c = (kept_at(j) + kPrior * p) / (reached_at(j) + kPrior);
            chance *= c;
            if (prior.reached_at(j) + reached_at(j) > 0) before = c;
        }
        return chance;
    }

private:
    static constexpr double kBreakEven = 0.5, kWeight = 8.0, kPrior = 4.0;
    static constexpr size_t kRest = 16;
    double kept_at(size_t i) const { return i < kept_w_.size() ? kept_w_[i] : 0.0; }
    double reached_at(size_t i) const { return i < reached_w_.size() ? reached_w_[i] : 0.0; }
    double average_ = 2.0;
    size_t rest_ = 0;
    std::vector<size_t> drafted_, kept_, reached_;   // by position, the drafts fed, those kept and those fed after every draft before them was kept
    std::vector<double> kept_w_, reached_w_;          // the kept and reached the chances read, each verify weighing kForget of the next
};

// Where a draft comes from: the request's own history ahead of a model proposer (lookup_first), or its proposer.
enum class Source : uint8_t { lookup, proposer };

// The ranges of a history draft's match length at its start that keep their acceptance apart: 8 to 15, 16 to 31, and 32 or more tokens.
// Measured on the match profile (docs/SPECULATIVE.md, section 8): a match of 32 kept its 16th draft 0.82 of the time, one of 8 to 11 0.25.
inline constexpr size_t kMatchBuckets = 3;
inline size_t match_bucket(size_t match) { return match >= 32 ? 2 : match >= 16 ? 1 : 0; }

// A request's drafting record: an acceptance for the proposer and one for each match bucket of the history's own drafts, so none's misses rest another, and the source and bucket of the draft its verify in flight feeds.
struct Record {
    Acceptance lookup[kMatchBuckets], proposer;
    Source drafted = Source::proposer;
    size_t bucket = 0;
    // The acceptance of source `s`, a history draft's in match bucket `b`.
    Acceptance& at(Source s, size_t b) { return s == Source::lookup ? lookup[b] : proposer; }
    const Acceptance& at(Source s, size_t b) const { return s == Source::lookup ? lookup[b] : proposer; }
    // The acceptance of the draft in flight.
    Acceptance& drafting() { return at(drafted, bucket); }
    // Whether every match bucket of the history's own drafts rests, so no walk of the history can give a draft.
    bool lookup_resting() const { return lookup[0].resting() && lookup[1].resting() && lookup[2].resting(); }
    // A verify that fed `fed` drafts of the draft in flight and kept the first `kept`, the tokens it generated counted against every other acceptance's rest.
    void verified(size_t kept, size_t fed) {
        Acceptance& d = drafting();
        d.verified(kept, fed);
        for (Acceptance* a : all())
            if (a != &d) a->stepped(kept + 1);
    }
    // A token generated without drafts.
    void stepped() {
        for (Acceptance* a : all()) a->stepped();
    }
    // The drafts fed and kept by position over source `s`, its buckets summed.
    void tally(Source s, std::vector<size_t>& drafted_at, std::vector<size_t>& kept_at) const {
        drafted_at.clear();
        kept_at.clear();
        for (size_t b = 0; b < (s == Source::lookup ? kMatchBuckets : 1); ++b) {
            const Acceptance& a = at(s, b);
            if (drafted_at.size() < a.drafted().size()) drafted_at.resize(a.drafted().size()), kept_at.resize(a.drafted().size());
            for (size_t i = 0; i < a.drafted().size(); ++i) drafted_at[i] += a.drafted()[i], kept_at[i] += a.kept()[i];
        }
    }

private:
    std::array<Acceptance*, kMatchBuckets + 1> all() { return {&lookup[0], &lookup[1], &lookup[2], &proposer}; }
};

// How many drafts a request verifies next, the one rule for every caller: at most `draft_max`, one fewer than the tokens it may still generate and than the positions its context has left, and none while it rests (Acceptance).
// It is also at most `columns`, the rows the pass's decode kernels read each weight once for, past which a draft row costs a weight read of its own (docs/SPECULATIVE.md, section 3).
inline size_t draft_length(size_t draft_max, size_t tokens_left, size_t context_left, size_t columns, const Acceptance& acceptance) {
    if (acceptance.resting() || !tokens_left || !context_left) return 0;
    return std::min({draft_max, tokens_left - 1, context_left - 1, columns});
}

// A request's draft ahead of its proposer (docs/SPECULATIVE.md, section 8): the convincing continuation of its own `history` (Lookup::convincing), up to the drafts draft_length allows the acceptance of its match's bucket, in `out`.
// It sets the record's source and bucket and returns the drafts; with none the source is the proposer's, which the caller asks under that source's acceptance.
inline size_t lookup_first(Record& r, size_t draft_max, size_t tokens_left, size_t context_left, size_t columns, const std::vector<uint32_t>& history,
                           std::vector<uint32_t>& out, const ImageRows& images = {}) {
    out.clear();
    r.drafted = Source::proposer;
    if (r.lookup_resting()) return 0;
    static const Acceptance fresh;
    const size_t k = draft_length(draft_max, tokens_left, context_left, columns, fresh);
    const size_t match = k ? Lookup::convincing(history, k, out, images) : 0;
    if (out.empty()) return 0;
    if (r.lookup[match_bucket(match)].resting()) {
        out.clear();
        return 0;
    }
    r.drafted = Source::lookup;
    r.bucket = match_bucket(match);
    return out.size();
}

// What a pass of generated rows costs, `base_ms` and `row_ms` a row, and a step of the drafter's chains, which run before the pass, `step_ms` and `step_row_ms` a chain.
// The pass's part is unknown until passes of different rows have been measured (PassTimes), `seen_rows` being their rows, and `rows_ms[r]` is r rows' price from the server's price table in server time, 0 where it has none (pass_ms).
struct PassCost {
    double base_ms = 0, row_ms = 0, step_ms = 0, step_row_ms = 0, seen_rows = 0;
    bool known = false;
    std::vector<double> rows_ms;
};

// A pass of `rows` generated rows: the price table's where it prices them, past its last priced width or before its first that width's price on the line's slope, else the line.
// Continuing from the table rather than jumping to the line keeps a draft past the measured widths from looking free where the line sits below them.
inline double pass_ms(const PassCost& c, size_t rows) {
    if (rows < c.rows_ms.size() && c.rows_ms[rows] > 0) return c.rows_ms[rows];
    size_t first = c.rows_ms.size(), last = 0;
    for (size_t r = 0; r < c.rows_ms.size(); ++r)
        if (c.rows_ms[r] > 0) first = std::min(first, r), last = r;
    if (first == c.rows_ms.size()) return c.base_ms + c.row_ms * (double)rows;
    if (rows > last) return c.rows_ms[last] + c.row_ms * (double)(rows - last);
    return std::max(0.0, c.rows_ms[first] - c.row_ms * (double)(first - rows));
}

// The pass cost measured as passes run: lines through passes' rows and chains' steps, and the server's time past the table's prices, each weighing kForget of the next.
// A line is known while its inputs spread by half a unit or more, an unknown chain line being its mean step.
class PassTimes {
public:
    void pass(size_t rows, double ms) { passes_.add((double)rows, ms); }
    // A pass whose slowest stage the price table priced at `stage_ms` took `ms` of the server's time.
    void reconcile(double stage_ms, double ms) { stages_.add(stage_ms, ms); }
    // The server time of a pass priced at `stage_ms`: that price and the reconciled passes' mean difference, so the table's shape, its steps included, is kept; 0 for an unpriced pass.
    double server_ms(double stage_ms) const { return stage_ms > 0 ? stage_ms + stages_.mean() - stages_.mean_x() : 0.0; }
    void chain(size_t steps, size_t chains, double ms) {
        if (steps) chains_.add((double)chains, ms / (double)steps);
    }
    PassCost cost() const {
        PassCost c;
        if (!chains_.fit(c.step_ms, c.step_row_ms)) c.step_ms = chains_.mean();
        c.known = passes_.fit(c.base_ms, c.row_ms);
        if (!c.known) c.seen_rows = passes_.mean_x();
        return c;
    }

private:
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
    Line passes_, chains_, stages_;
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
    const double plain_ms = pass_ms(c, decoders), rate = (double)decoders / plain_ms;
    // A depth is weighed with the chain and without it, where only the requests that need no chain draft, so a dear chain does not take their drafts with it.
    const auto take = [&](size_t K, bool chain, std::vector<size_t>* out) {
        double tokens = (double)decoders;
        size_t rows = decoders, chains = 0, steps = 0;
        for (size_t i = 0; i < keeps.size(); ++i) {
            const bool chains_it = !chained || (*chained)[i];
            const size_t k = chains_it && !chain ? 0 : std::min(K, keeps[i].size());
            double kept = 0;
            for (size_t j = 0; j < k; ++j) kept += keeps[i][j];
            if (!k || kept < rate * (pass_ms(c, rows + k) - pass_ms(c, rows))) continue;
            tokens += kept;
            rows += k;
            if (chains_it) ++chains, steps = std::max(steps, k);
            if (out) (*out)[i] = k;
        }
        double ms = pass_ms(c, rows);
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
    ImageRows images;   // the history's image rows
    Record record;
};

} // namespace spec
} // namespace infer
