#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>

// The sampler and the sampling settings it reads, with their defaults and ranges.
// generate (the CLI's generate and chat) and the server's scheduler both sample through it.

namespace infer {

// Minimal xorshift64 PRNG (no <random> dependency; deterministic + portable).
struct RNG {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    void seed(uint64_t x) { if (x) s = x; }
    uint64_t next() {
        uint64_t x = s;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s = x;
        return x;
    }
    // uniform float in [0,1)
    float unit() { return (float)((next() >> 40) * (1.0 / 16777216.0)); }
};

// The values a sampling setting takes, from lo to hi.
template <class T>
struct SampleRange {
    T lo, hi;
    bool holds(T v) const { return v >= lo && v <= hi; }   // false for NaN
};

// One generation's sampling settings, each default and each range written once.
// The CLI's flags and the server's request fields start from these defaults and refuse a value outside these ranges, so the two take the same values.
struct Sampling {
    int max_tokens = 64;    // tokens generated at most
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    float penalty = 1.0f;   // repetition penalty
    uint64_t seed = 0;      // 0 keeps the fixed default RNG state
    bool ignore_eos = false;    // the token that ends a reply is never drawn, so the reply runs to its limit or a stop text

    static constexpr SampleRange<float> temp_range{0.0f, std::numeric_limits<float>::max()};    // 0 is greedy
    static constexpr SampleRange<int> top_k_range{0, std::numeric_limits<int>::max()};            // 0 keeps every token
    static constexpr SampleRange<float> top_p_range{0.0f, 1.0f};                                  // 1 keeps every token
    static constexpr SampleRange<float> penalty_range{1.0f, std::numeric_limits<float>::max()};  // 1 is none, and below 1 would favor repeats
};

// What generate reads: the sampling settings and the one stop text of the CLI's generate and chat.
struct GenParams : Sampling {
    std::string stop;       // stop generating when decoded output contains this
};

namespace detail {

// A token's place in the ranking as one integer, larger for the better token: the higher score first, and the lower id on a tie.
// The score's bits are mapped so that unsigned order is float order, and the id is complemented so that the lower id gives the larger key.
inline uint64_t rank_key(float score, uint32_t id) {
    score += 0.0f;  // -0 + 0 is +0, so the two zeros tie as they compare equal
    uint32_t bits;
    std::memcpy(&bits, &score, sizeof bits);
    bits = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    return ((uint64_t)bits << 32) | (uint32_t)~id;
}

inline uint32_t key_id(uint64_t key) { return ~(uint32_t)key; }

// Ids in increasing order: a sort for a few, one pass over a mark per token for many.
inline void in_id_order(std::vector<uint32_t>& ids, size_t n) {
    if (ids.size() < n / 32) {
        std::sort(ids.begin(), ids.end());
        return;
    }
    std::vector<uint8_t> mark(n, 0);
    for (uint32_t id : ids) mark[id] = 1;
    ids.clear();
    for (size_t i = 0; i < n; i++)
        if (mark[i]) ids.push_back((uint32_t)i);
}

// The kept tokens, the best of the n scores but the one at `skip` (n or more for none), put in rank order only as far as they are read.
// A top-k set of up to `heap_limit`, and a nucleus up to `nucleus_heap`, is ranked in one pass over the scores with a heap of the best so far.
// Past that the kept keys not yet ranked are laid out once behind the ranked ones, and each further prefix is ranked by a selection over the unranked rest.
class Ranking {
public:
    Ranking(const float* scores, size_t n, size_t skip, size_t kept)
        : s_(scores), n_(n), skip_(skip), m_(skip < n ? n - 1 : n), kept_(kept) {}

    // The id of the i-th best token, for i below the kept count.
    uint32_t id(size_t i) {
        if (i >= ranked_) rank(i + 1);
        return key_id(keys_[i]);
    }

    // The kept tokens' ids, in no set order.
    std::vector<uint32_t> kept_ids() {
        if (kept_ > heap_limit) lay_out();
        else if (ranked_ < kept_) rank(kept_);
        std::vector<uint32_t> ids(kept_);
        for (size_t i = 0; i < kept_; i++) ids[i] = key_id(keys_[i]);
        return ids;
    }

private:
    static constexpr size_t heap_limit = 4096;
    static constexpr size_t nucleus_heap = 512;
    static constexpr size_t first_rank = 64;

    // Ranks at least the best c.
    // A small kept set is ranked whole, since its ids are all read.
    // Otherwise a nucleus is read a token at a time: a heap pass ranks eight times what the last one did up to `nucleus_heap`, and past that each selection doubles the ranked prefix, so a large nucleus has fewer than twice its tokens ranked.
    // A nucleus leaves the heap at 512 rather than 4096 because on rows whose nucleus runs to tens of thousands of tokens a heap pass of 4096 cost more than the selections it saves.
    void rank(size_t c) {
        if (kept_ < m_ && kept_ <= heap_limit) c = kept_;
        else c = std::min(kept_, std::max({c, (8 * ranked_ <= nucleus_heap ? 8 : 2) * ranked_, first_rank}));
        if (!laid_out_ && c <= (kept_ < m_ ? heap_limit : nucleus_heap)) {
            heap_select(c);
            return;
        }
        lay_out();
        const auto from = keys_.begin() + (std::ptrdiff_t)ranked_;
        const auto to = keys_.begin() + (std::ptrdiff_t)c;
        if (to != keys_.end()) std::nth_element(from, to, keys_.end(), std::greater<uint64_t>());
        std::sort(from, to, std::greater<uint64_t>());
        ranked_ = c;
    }

    // The best c keys in rank order, from a heap whose front is the worst key it holds.
    void heap_select(size_t c) {
        const std::greater<uint64_t> worse_first{};
        keys_.clear();
        keys_.reserve(c);
        size_t i = 0;
        for (; i < n_ && keys_.size() < c; i++) {
            if (i == skip_) continue;
            keys_.push_back(rank_key(s_[i], (uint32_t)i));
            std::push_heap(keys_.begin(), keys_.end(), worse_first);
        }
        // A score below the worst held one has the smaller key whatever its id, so it is passed over without a key.
        float worst = s_[key_id(keys_.front())];
        for (; i < n_; i++) {
            if (s_[i] < worst || i == skip_) continue;
            const uint64_t key = rank_key(s_[i], (uint32_t)i);
            if (key > keys_.front()) {
                std::pop_heap(keys_.begin(), keys_.end(), worse_first);
                keys_.back() = key;
                std::push_heap(keys_.begin(), keys_.end(), worse_first);
                worst = s_[key_id(keys_.front())];
            }
        }
        std::sort_heap(keys_.begin(), keys_.end(), worse_first);
        ranked_ = c;
    }

    // Every kept key, the ranked ones first as they stand and the rest unranked behind them, less those a selection puts past the kept count.
    // Keys are distinct, so the ids not yet ranked are those whose key is below the last ranked one.
    void lay_out() {
        if (laid_out_) return;
        const bool all = ranked_ == 0;
        const uint64_t last = all ? 0 : keys_[ranked_ - 1];
        keys_.resize(m_);
        size_t j = ranked_;
        for (size_t i = 0; i < n_; i++) {
            if (i == skip_) continue;
            const uint64_t key = rank_key(s_[i], (uint32_t)i);
            if (all || key < last) keys_[j++] = key;
        }
        if (kept_ < m_) {
            std::nth_element(keys_.begin() + (std::ptrdiff_t)ranked_, keys_.begin() + (std::ptrdiff_t)kept_, keys_.end(),
                             std::greater<uint64_t>());
            keys_.resize(kept_);
        }
        laid_out_ = true;
    }

    const float* s_;
    size_t n_, skip_, m_, kept_;  // m_ ids take part: all n_ but skip_
    std::vector<uint64_t> keys_;  // the best ranked_ in rank order, then when laid out the rest of the kept set
    size_t ranked_ = 0;
    bool laid_out_ = false;
};

} // namespace detail

// Temperature + top-k + top-p nucleus sampling with repetition penalty.
// `penalty` >= 1: divide the score of each already-generated token by penalty to discourage repeats.
// `masked`, an id of the row or -1 for none, is passed over whatever the penalty: greedy never takes it, and a draw leaves it out before top-k, top-p and the softmax, so it does not exist for the draw.
// Returns the chosen token id.
// Tokens rank by score, and a tie by the lower id, so no sort's handling of equal scores reaches the result.
// Each sum of weights is taken in id order, or best first for the nucleus, so none depends on the order a selection leaves its candidates in.
inline uint32_t sample(const std::vector<float>& logits, float temp, int top_k,
                       float top_p, float penalty, const std::vector<uint32_t>& gen,
                       RNG& rng, int64_t masked = -1) {
    const size_t n = logits.size();
    // The id every path passes over, n for none; a row holding nothing else keeps the masked id, since a draw needs a token.
    const size_t skip = (masked >= 0 && (uint64_t)masked < n && n > 1) ? (size_t)masked : n;
    const size_t m = skip < n ? n - 1 : n;  // the ids a draw can give

    // Repetition penalty, in a copy made only when a token is penalized.
    // A seen token's score is set from its logit, so a token seen several times is penalized once.
    const float* score = logits.data();
    std::vector<float> penalized;
    if (penalty > 0.0f && penalty != 1.0f && !gen.empty()) {
        penalized = logits;
        for (uint32_t id : gen) {
            if (id >= n) continue;
            const float v = logits[id];
            penalized[id] = (v > 0.0f) ? (v / penalty) : (v * penalty);
        }
        score = penalized.data();
    }

    // Greedy needs the largest score, not an ordering of the rest.
    // Ties take the lowest token id.
    // The scan starts past a masked id 0, so a row whose other scores are all negative infinity or NaN still does not give it.
    if (temp <= 0.0f) {
        size_t best = skip == 0 ? 1 : 0;
        float best_score = score[best];
        for (size_t i = best + 1; i < n; i++) {
            const float v = score[i];
            if (v > best_score && i != skip) { best_score = v; best = i; }
        }
        return (uint32_t)best;
    }

    // A token's weight is exp((score - best score) / temp), its softmax probability before the division by the sum.
    // Temperature is applied exactly once, here: pre-scaling the scores by temp as well would cancel this division and make --temp a no-op at every value > 0.
    // The draw is r times the drawn tokens' weight, found by walking them in the order that weight was summed in.
    const size_t keep = (top_k > 0 && (size_t)top_k < m) ? (size_t)top_k : m;
    const float r = rng.unit();

    // Every token kept: the draw walks the whole vocabulary, so nothing is ranked.
    if (keep == m && top_p >= 1.0f) {
        size_t best = skip == 0 ? 1 : 0;
        for (size_t i = best + 1; i < n; i++)
            if (score[i] > score[best] && i != skip) best = i;
        const float best_score = score[best];
        std::unique_ptr<float[]> w(new float[n]);  // not zeroed, since every weight read is written first
        double sum = 0.0;
        for (size_t i = 0; i < n; i++)
            if (i != skip) sum += w[i] = std::exp((score[i] - best_score) / temp);
        const double target = r * sum;
        double acc = 0.0;
        for (size_t i = 0; i < n; i++)
            if (i != skip && target < (acc += w[i])) return (uint32_t)i;
        return (uint32_t)best;
    }

    detail::Ranking ranking(score, n, skip, keep);
    // Fewer than all kept: the kept set is selected first, so the best token is then found within it.
    std::vector<uint32_t> kept;
    if (keep < m) {
        kept = ranking.kept_ids();
        detail::in_id_order(kept, n);
    }
    const float best_score = score[ranking.id(0)];
    const auto weight = [&](uint32_t id) { return std::exp((score[id] - best_score) / temp); };

    // The kept tokens' weight, the softmax's sum, over their ids in increasing order.
    // Each weight is held as it is summed, by place in `kept` or, with every token kept, by id, since a nucleus can then reach most of the row.
    // The buffer is not zeroed, since every weight read is written first, and zeroing would add a pass over the row that a small nucleus gains nothing from.
    std::unique_ptr<float[]> w(new float[keep < m ? keep : n]);
    double sum = 0.0;
    if (keep < m) {
        for (size_t j = 0; j < keep; j++) sum += w[j] = weight(kept[j]);
    } else {
        for (size_t i = 0; i < n; i++)
            if (i != skip) sum += w[i] = weight((uint32_t)i);
    }

    // Without top-p the draw walks the kept tokens in id order.
    if (top_p >= 1.0f) {
        const double target = r * sum;
        double acc = 0.0;
        for (size_t j = 0; j < keep; j++)
            if (target < (acc += w[j])) return kept[j];
        return ranking.id(0);
    }

    // top-p: the shortest ranked prefix whose weight, summed best first, reaches top_p of the kept weight.
    // It is ranked only as far as it reaches, and the draw walks it best first.
    // Within a top-k a nucleus token's weight is taken again, which is at most k of them; with every token kept it is read.
    const auto kept_weight = [&](uint32_t id) { return keep < m ? weight(id) : w[id]; };
    const double goal = top_p * sum;
    double nucleus_weight = 0.0;
    size_t nucleus = 0;
    do {
        nucleus_weight += kept_weight(ranking.id(nucleus));
        nucleus++;
    } while (nucleus < keep && nucleus_weight < goal);
    const double target = r * nucleus_weight;
    double acc = 0.0;
    for (size_t i = 0; i < nucleus; i++)
        if (target < (acc += kept_weight(ranking.id(i)))) return ranking.id(i);
    return ranking.id(0);
}

// The next token of a reply under `s`, drawn from the logits after the tokens `gen` it holds so far.
// `end` is the id that ends a reply (bpe::Tokenizer::eos_id, -1 for none); with s.ignore_eos it is masked, so the reply runs on to its token limit.
inline uint32_t sample(const std::vector<float>& logits, const Sampling& s, int32_t end,
                       const std::vector<uint32_t>& gen, RNG& rng) {
    return sample(logits, s.temp, s.top_k, s.top_p, s.penalty, gen, rng, s.ignore_eos ? end : -1);
}

} // namespace infer
