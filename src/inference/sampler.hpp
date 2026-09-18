#pragma once
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>

// Sampling logic and generation parameters, split out of the CLI so the same
// sampler can drive generate, perplexity, and chat.

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

struct GenParams {
    int max_tokens = 64;
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    int threads = 0;        // 0 = auto
    float penalty = 1.0f;   // repetition penalty (>= 1)
    uint64_t seed = 0;      // 0 = non-deterministic
    std::string stop;       // stop generating when decoded output contains this
    bool show_prompt_tokens = false;
    bool show_thinking = false; // show Qwen3 <thinking> block
};

// Temperature + top-k + top-p nucleus sampling with repetition penalty.
// `penalty` >= 1: divide the score of each already-generated token by penalty
// to discourage repeats. Returns the chosen token id.
inline uint32_t sample(const std::vector<float>& logits, float temp, int top_k,
                       float top_p, float penalty, const std::vector<uint32_t>& gen,
                       RNG& rng) {
    size_t n = logits.size();

    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(n);
    for (size_t i = 0; i < n; i++) ranked.push_back({ logits[i], (uint32_t)i });

    // repetition penalty
    if (penalty > 0.0f && penalty != 1.0f && !gen.empty()) {
        std::unordered_set<uint32_t> seen;
        for (uint32_t id : gen) seen.insert(id);
        for (auto& pr : ranked) {
            if (seen.count(pr.second)) {
                pr.first = (pr.first > 0.0f) ? (pr.first / penalty) : (pr.first * penalty);
            }
        }
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // top-k truncation
    size_t keep = (top_k > 0 && (size_t)top_k < n) ? (size_t)top_k : n;

    if (temp <= 0.0f) return ranked[0].second; // argmax (no randomness)

    // softmax over the kept window. Temperature is applied exactly once, here:
    // pre-scaling the scores by temp as well would cancel this division and
    // make --temp a no-op at every value > 0.
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
