#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <unordered_set>

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

    static constexpr SampleRange<float> temp_range{0.0f, std::numeric_limits<float>::max()};    // 0 is greedy
    static constexpr SampleRange<int> top_k_range{0, std::numeric_limits<int>::max()};            // 0 keeps every token
    static constexpr SampleRange<float> top_p_range{0.0f, 1.0f};                                  // 1 keeps every token
    static constexpr SampleRange<float> penalty_range{1.0f, std::numeric_limits<float>::max()};  // 1 is none, and below 1 would favor repeats
};

// What generate reads: the sampling settings and the one stop text of the CLI's generate and chat.
struct GenParams : Sampling {
    std::string stop;       // stop generating when decoded output contains this
};

// Temperature + top-k + top-p nucleus sampling with repetition penalty.
// `penalty` >= 1: divide the score of each already-generated token by penalty to discourage repeats.
// Returns the chosen token id.
inline uint32_t sample(const std::vector<float>& logits, float temp, int top_k,
                       float top_p, float penalty, const std::vector<uint32_t>& gen,
                       RNG& rng) {
    const size_t n = logits.size();

    // Repetition penalty, read through rather than materialized: the greedy path below never needs a second array.
    std::unordered_set<uint32_t> seen;
    const bool repeat = (penalty > 0.0f && penalty != 1.0f && !gen.empty());
    if (repeat) for (uint32_t id : gen) seen.insert(id);
    const auto score = [&](size_t i) {
        const float v = logits[i];
        if (!repeat || !seen.count((uint32_t)i)) return v;
        return (v > 0.0f) ? (v / penalty) : (v * penalty);
    };

    // Greedy needs the largest score, not an ordering of the rest.
    // Ties take the lowest token id.
    if (temp <= 0.0f) {
        size_t best = 0;
        float best_score = score(0);
        for (size_t i = 1; i < n; i++) {
            const float v = score(i);
            if (v > best_score) { best_score = v; best = i; }
        }
        return (uint32_t)best;
    }

    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(n);
    for (size_t i = 0; i < n; i++) ranked.push_back({ score(i), (uint32_t)i });

    const auto by_score = [](const std::pair<float, uint32_t>& a,
                             const std::pair<float, uint32_t>& b) {
        return a.first > b.first;
    };

    // top-k truncation.
    // Nothing below reads past `keep`, so the tail is left unordered.
    const size_t keep = (top_k > 0 && (size_t)top_k < n) ? (size_t)top_k : n;
    if (keep < n)
        std::partial_sort(ranked.begin(), ranked.begin() + (ptrdiff_t)keep,
                          ranked.end(), by_score);
    else
        std::sort(ranked.begin(), ranked.end(), by_score);

    // softmax over the kept window.
    // Temperature is applied exactly once, here: pre-scaling the scores by temp as well would cancel this division and make --temp a no-op at every value > 0.
    float maxv = ranked[0].first;
    std::vector<float> p(keep);
    double sum = 0.0;
    for (size_t i = 0; i < keep; i++) {
        float v = std::exp((ranked[i].first - maxv) / temp);
        p[i] = v;
        sum += v;
    }
    for (size_t i = 0; i < keep; i++) p[i] = (float)(p[i] / sum);

    // top-p nucleus truncation
    size_t nuc = keep;
    if (top_p < 1.0f) {
        float acc = 0.0f;
        nuc = 0;
        while (nuc < keep && acc < top_p) { acc += p[nuc]; nuc++; }
        if (nuc < 1) nuc = 1;
        // renormalize over the nucleus
        float nsum = 0.0f;
        for (size_t i = 0; i < nuc; i++) nsum += p[i];
        for (size_t i = 0; i < nuc; i++) p[i] /= nsum;
    }

    float r = rng.unit();
    float acc = 0.0f;
    for (size_t i = 0; i < nuc; i++) {
        acc += p[i];
        if (r < acc) return ranked[i].second;
    }
    return ranked[nuc - 1].second;
}

} // namespace infer
