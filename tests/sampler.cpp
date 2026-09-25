// infer::sample against expectations taken from the definitions rather than from the code.
// Greedy is the argmax, the penalty divides a seen positive score and multiplies a seen negative one, top-k keeps the k best, the nucleus is the shortest ranked prefix whose probability reaches top_p, and a draw follows the softmax of the kept scores over the temperature.
// Frequencies are held to three binomial standard deviations per token from fixed seeds, so every run draws the same tokens.
// Every draw is also held, token for token and with the generator's state after it, to a slow reference that sorts all scores but a masked one, ties by the lower id, which fixes the ranking, the tokens kept, the order a draw walks them in and the generator's use.
// The reference sums the softmax in id order as the definition does, but a sum in another order differs only in its last bits, so no draw here can show that order.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference/sampler.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

const std::vector<uint32_t> no_history;
constexpr int draws = 10000;

uint32_t greedy(const std::vector<float>& logits, float penalty = 1.0f,
                const std::vector<uint32_t>& gen = no_history, int64_t masked = -1) {
    infer::RNG rng;
    return infer::sample(logits, 0.0f, 40, 0.95f, penalty, gen, rng, masked);
}

std::vector<int> counts(const std::vector<float>& logits, float temp, int top_k, float top_p,
                        uint64_t seed, int64_t masked = -1) {
    infer::RNG rng;
    rng.seed(seed);
    std::vector<int> n(logits.size(), 0);
    for (int i = 0; i < draws; i++) {
        const uint32_t id = infer::sample(logits, temp, top_k, top_p, 1.0f, no_history, rng, masked);
        if (id >= logits.size())
            throw std::runtime_error("sampled id " + std::to_string(id) + " is out of range");
        n[id]++;
    }
    return n;
}

// The softmax of logits / temp over the `kept` tokens, zero for the rest.
std::vector<double> softmax(const std::vector<float>& logits, double temp,
                            const std::vector<uint32_t>& kept) {
    std::vector<double> p(logits.size(), 0.0);
    double sum = 0.0;
    for (uint32_t id : kept) sum += p[id] = std::exp(logits[id] / temp);
    for (double& v : p) v /= sum;
    return p;
}

std::vector<uint32_t> every(const std::vector<float>& logits) {
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < (uint32_t)logits.size(); i++) ids.push_back(i);
    return ids;
}

// Empty when each count is within three standard deviations of draws * p, else the first token that is not.
// A probability of zero or one has no spread, so that token must be drawn never or every time.
std::string misfit(const std::vector<int>& n, const std::vector<double>& p) {
    for (size_t i = 0; i < n.size(); i++) {
        const double mean = draws * p[i];
        const double bound = 3.0 * std::sqrt(draws * p[i] * (1.0 - p[i]));
        if (std::fabs(n[i] - mean) > bound) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "token %zu drawn %d times, expected %.1f +- %.1f", i,
                          n[i], mean, bound);
            return buf;
        }
    }
    return {};
}

void fits(const std::string& name, const std::vector<int>& n, const std::vector<double>& p) {
    const std::string why = misfit(n, p);
    require(why.empty(), name + ": " + why);
}

void temperature_zero() {
    require(greedy({0.5f, 2.0f, -1.0f, 1.5f}) == 1, "temperature 0 missed the largest score");
    require(greedy({-2.0f, -0.5f, -1.0f}) == 1, "temperature 0 missed the largest negative score");
    require(greedy({1.0f, 3.0f, 0.0f, 3.0f}) == 1, "a tie did not take the lowest id");
    require(greedy({3.0f, 3.0f, 3.0f}) == 0, "an all-equal row did not take id 0");
}

// Penalty 2 takes the leader 3 to 1.5 and the leader -1 to -2, so a runner-up just either side of those values pins the factor.
void penalty() {
    const std::vector<uint32_t> seen = {0};
    require(greedy({3.0f, 1.6f, 0.0f}, 2.0f, seen) == 1, "a seen positive leader was not divided");
    require(greedy({3.0f, 1.4f, 0.0f}, 2.0f, seen) == 0, "a seen positive leader lost more than the penalty");
    require(greedy({-1.0f, -1.9f, -3.0f}, 2.0f, seen) == 1, "a seen negative leader was not multiplied");
    require(greedy({-1.0f, -2.1f, -3.0f}, 2.0f, seen) == 0, "a seen negative leader lost more than the penalty");
    require(greedy({3.0f, 1.4f, 0.0f}, 2.0f, {0, 0, 0}) == 0, "a token seen three times was penalized more than once");
    require(greedy({3.0f, 1.6f, 0.0f}, 1.0f, seen) == 0, "penalty 1 changed a score");
    require(greedy({3.0f, 1.6f, 0.0f}, 2.0f, no_history) == 0, "an empty history penalized a token");

    infer::RNG rng;
    require(infer::sample({3.0f, 1.6f, 0.0f}, 1.0f, 1, 1.0f, 2.0f, seen, rng) == 1,
            "the sampling path ranked the unpenalized score");
}

void top_k() {
    // The three best are ids 1, 6 and 3, and the fourth, id 4, is close enough that keeping one too many shows.
    const std::vector<float> logits = {0.2f, 1.5f, -0.3f, 1.1f, 0.9f, -1.0f, 1.3f, 0.0f};
    for (float temp : {0.5f, 1.0f, 4.0f, 100.0f}) {
        char name[48];
        std::snprintf(name, sizeof name, "top_k 1 at temperature %g", temp);
        fits(name, counts(logits, temp, 1, 1.0f, 2), softmax(logits, 1.0, {1}));
    }
    // At temperature 2 almost half the full softmax lies outside the three best.
    fits("top_k 3", counts(logits, 2.0f, 3, 1.0f, 3), softmax(logits, 2.0, {1, 6, 3}));
}

void top_p() {
    const std::vector<float> logits = {0.0f, 1.2f, -1.0f, 1.7f, -0.5f};
    const std::vector<double> full = softmax(logits, 1.0, every(logits));
    require(full[3] < 0.65 && 0.65 < full[3] + full[1], "top_p 0.65 does not fall between the first two");
    fits("top_p 0.65", counts(logits, 1.0f, 0, 0.65f, 4), softmax(logits, 1.0, {3, 1}));
    // At temperature 2 the first two hold only 0.636, so top_p 0.7 keeps three; a nucleus measured before the temperature would keep two.
    const std::vector<double> warm = softmax(logits, 2.0, every(logits));
    require(warm[3] + warm[1] < 0.7 && 0.7 < warm[3] + warm[1] + warm[0], "top_p 0.7 does not fall between the second and third at temperature 2");
    fits("top_p 0.7 at temperature 2", counts(logits, 2.0f, 0, 0.7f, 5), softmax(logits, 2.0, {3, 1, 0}));
    // Within the two best the first holds 0.622, so top_p 0.6 keeps it alone; a nucleus measured over the whole row would keep both.
    fits("top_k 2 with top_p 0.6", counts(logits, 1.0f, 2, 0.6f, 6), softmax(logits, 1.0, {3}));
}

void temperature() {
    const std::vector<float> logits = {1.0f, 0.0f, 2.0f, 0.5f, -0.5f};
    const std::vector<uint32_t> ids = every(logits);
    // Seed 0 is the state every unseeded run samples from, so its draws are held to the bound too.
    fits("temperature 1", counts(logits, 1.0f, 0, 1.0f, 0), softmax(logits, 1.0, ids));
    const std::vector<int> half = counts(logits, 0.5f, 0, 1.0f, 1);
    fits("temperature 0.5", half, softmax(logits, 0.5, ids));
    require(!misfit(half, softmax(logits, 0.25, ids)).empty(),
            "the bound cannot tell temperature 0.5 from the temperature applied twice");
    require(!misfit(half, softmax(logits, 1.0, ids)).empty(),
            "the bound cannot tell temperature 0.5 from an ignored temperature");
}

std::vector<uint32_t> sequence(infer::RNG rng) {
    const std::vector<float> logits = {1.0f, 0.0f, 2.0f, 0.5f, -0.5f};
    std::vector<uint32_t> ids;
    for (int i = 0; i < 1000; i++)
        ids.push_back(infer::sample(logits, 1.0f, 0, 1.0f, 1.0f, no_history, rng));
    return ids;
}

infer::RNG seeded(uint64_t seed) {
    infer::RNG rng;
    rng.seed(seed);
    return rng;
}

void seeds() {
    require(sequence(seeded(42)) == sequence(seeded(42)), "seed 42 did not repeat its sequence");
    require(sequence(seeded(42)) != sequence(seeded(43)), "seeds 42 and 43 drew the same sequence");
    require(sequence(seeded(0)) == sequence(infer::RNG{}), "seed 0 moved the default state");
}

// A masked id, the end of text under ignore_eos, does not exist for the draw: greedy takes the best of the rest, the penalty cannot bring it back, top-k and top-p count only the other tokens, and a draw follows their softmax.
void masked() {
    const std::vector<float> row = {0.5f, 2.0f, -1.0f, 1.5f};
    require(greedy(row, 1.0f, no_history, 1) == 3, "greedy took the masked leader");
    require(greedy(row, 1.0f, no_history, 0) == 1, "masking another id moved greedy");
    require(greedy(row, 1.0f, no_history, 4) == 1, "an id past the row masked a token");
    // Penalty 2 would take a seen leader of 3 to 1.5, above the rest; masked, it stays below them.
    require(greedy({3.0f, -5.0f, -6.0f}, 2.0f, {0}, 0) == 1, "the penalty brought a masked token back");
    // Beside scores that are all negative infinity or NaN a masked id 0 is still not given, greedy or drawn on every path: nothing ranked, a top-k's kept set with and without a nucleus, and a nucleus over the whole row.
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    require(greedy({1.0f, -inf, -inf}, 1.0f, no_history, 0) == 1, "greedy gave a masked id 0 over negative infinities");
    require(greedy({1.0f, nan, nan}, 1.0f, no_history, 0) == 1, "greedy gave a masked id 0 over NaN");
    for (float other : {-inf, nan})
        for (int top_k : {0, 1})
            for (float top_p : {1.0f, 0.95f}) {
                char name[96];
                std::snprintf(name, sizeof name, "a draw at top_k %d and top_p %g gave a masked id 0 over %s", top_k, top_p,
                              std::isnan(other) ? "NaN" : "negative infinities");
                require(counts({1.0f, other, other}, 1.0f, top_k, top_p, 12, 0)[0] == 0, name);
            }

    const std::vector<float> logits = {1.0f, 0.0f, 2.0f, 0.5f, -0.5f};
    fits("temperature 1 with the leader masked", counts(logits, 1.0f, 0, 1.0f, 7, 2), softmax(logits, 1.0, {0, 1, 3, 4}));
    require(counts(logits, 1.0f, 0, 1.0f, 8, 5) == counts(logits, 1.0f, 0, 1.0f, 8), "an id past the row changed the draws");
    // With the best, id 1, masked, top_k 3 keeps the next three.
    const std::vector<float> wide = {0.2f, 1.5f, -0.3f, 1.1f, 0.9f, -1.0f, 1.3f, 0.0f};
    fits("top_k 3 with the leader masked", counts(wide, 2.0f, 3, 1.0f, 9, 1), softmax(wide, 2.0, {6, 3, 4}));
    // Without id 3 the first of the rest holds 0.627, so top_p 0.7 keeps two; a nucleus that counted the masked id's 0.508 would keep id 1 alone.
    const std::vector<float> nucleus = {0.0f, 1.2f, -1.0f, 1.7f, -0.5f};
    fits("top_p 0.7 with the leader masked", counts(nucleus, 1.0f, 0, 0.7f, 10, 3), softmax(nucleus, 1.0, {1, 0}));
    // A row holding only the masked id still gives a token.
    fits("a row of the masked id alone", counts({7.0f}, 1.0f, 0, 1.0f, 11, 0), {1.0});

    // The settings' overload masks the end id the caller names only with ignore_eos.
    infer::Sampling s;
    s.temp = 0.0f;
    infer::RNG rng;
    require(infer::sample(row, s, 1, no_history, rng) == 1, "the end id was masked without ignore_eos");
    s.ignore_eos = true;
    require(infer::sample(row, s, 1, no_history, rng) == 3, "ignore_eos drew the end id");
    require(infer::sample(row, s, -1, no_history, rng) == 1, "ignore_eos masked a token of a model without an end id");
}

// infer::sample as its definition reads, slowly: every token but a masked one ranked by a full sort on score and then id, and the kept tokens' softmax summed in id order.
// Without top-p the draw walks the kept tokens in id order; with it the nucleus is summed best first and the draw walks it best first.
// Its draws pin the ranking, the tie rule, the walk order and the generator's use; its sums follow the definition's order too, though no draw shows the order of a sum.
uint32_t reference(const std::vector<float>& logits, float temp, int top_k, float top_p, float penalty,
                   const std::vector<uint32_t>& gen, infer::RNG& rng, int64_t masked) {
    const size_t n = logits.size();
    std::vector<float> s = logits;
    if (penalty != 1.0f)
        for (uint32_t id : gen) s[id] = logits[id] > 0.0f ? logits[id] / penalty : logits[id] * penalty;
    std::vector<uint32_t> order;
    for (uint32_t i = 0; i < (uint32_t)n; i++)
        if (n < 2 || (int64_t)i != masked) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [&](uint32_t a, uint32_t b) { return s[a] > s[b] || (s[a] == s[b] && a < b); });
    if (temp <= 0.0f) return order[0];

    const size_t m = order.size();
    const size_t keep = (top_k > 0 && (size_t)top_k < m) ? (size_t)top_k : m;
    const float best = s[order[0]];
    const auto weight = [&](uint32_t id) { return std::exp((s[id] - best) / temp); };
    std::vector<uint32_t> kept(order.begin(), order.begin() + (std::ptrdiff_t)keep);
    std::sort(kept.begin(), kept.end());
    double sum = 0.0;
    for (uint32_t id : kept) sum += weight(id);
    const float r = rng.unit();
    if (top_p >= 1.0f) {
        const double target = r * sum;
        double acc = 0.0;
        for (uint32_t id : kept)
            if (target < (acc += weight(id))) return id;
        return order[0];
    }
    const double goal = top_p * sum;
    double nucleus_weight = 0.0;
    size_t nucleus = 0;
    do {
        nucleus_weight += weight(order[nucleus]);
        nucleus++;
    } while (nucleus < keep && nucleus_weight < goal);
    const double target = r * nucleus_weight;
    double acc = 0.0;
    for (size_t i = 0; i < nucleus; i++)
        if (target < (acc += weight(order[i]))) return order[i];
    return order[0];
}

// n scores from a seed.
// With `levels` they take that many values a quarter apart, so ties are everywhere and fall across every cut, and a zero is -0 or +0 at random, which compare equal.
// Without they spread as a model's do, with one in 200 far ahead.
std::vector<float> scores(size_t n, int levels, uint64_t seed) {
    infer::RNG rng = seeded(seed);
    std::vector<float> s(n);
    for (float& v : s) {
        if (levels) {
            v = (float)(rng.next() % (uint64_t)levels) * 0.25f - 1.0f;
            if (v == 0.0f && rng.next() % 2) v = -0.0f;
        } else {
            v = (rng.unit() + rng.unit() + rng.unit() - 1.5f) * 4.0f;
            if (rng.next() % 200 == 0) v += 12.0f;
        }
    }
    return s;
}

// `steps` draws of infer::sample and of the reference, each from its own copy of one seeded state and with the history growing by each draw, must be the same tokens and leave the same state.
void agrees(const std::vector<float>& logits, float temp, int top_k, float top_p, float penalty, uint64_t seed,
            int steps, int64_t masked) {
    infer::RNG fast = seeded(seed), slow = seeded(seed);
    std::vector<uint32_t> gen;
    for (int i = 0; i < steps; i++) {
        const uint32_t got = infer::sample(logits, temp, top_k, top_p, penalty, gen, fast, masked);
        const uint32_t want = reference(logits, temp, top_k, top_p, penalty, gen, slow, masked);
        if (got != want || fast.s != slow.s) {
            char buf[256];
            std::snprintf(buf, sizeof buf,
                          "%zu scores, temperature %g, top_k %d, top_p %g, penalty %g, masked %lld, seed %llu, draw %d: token %u, the reference's %u",
                          logits.size(), temp, top_k, top_p, penalty, (long long)masked, (unsigned long long)seed, i, got, want);
            require(false, buf);
        }
        ++checks;
        gen.push_back(got);
    }
}

// The id greedy takes from a row: the best score, the lowest id on a tie.
int64_t leader(const std::vector<float>& logits) {
    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

void against_reference() {
    const float temps[] = {0.0f, 0.2f, 0.8f, 1.5f};
    const float top_ps[] = {0.1f, 0.95f, 1.0f};
    const float penalties[] = {1.0f, 1.1f};
    uint64_t seed = 1;
    // Half the cells mask the row's leader, as ignore_eos masks a likely end of text, so the next best leads and every path passes over an id.
    // Each top-k, temperature and top-p masks under one penalty and not under the other, which penalty alternating, so both penalties draw with and without a mask.
    const auto grid = [&](const std::vector<float>& logits, std::initializer_list<int> top_ks, int steps) {
        for (int top_k : top_ks)
            for (float temp : temps)
                for (float top_p : top_ps)
                    for (float penalty : penalties) {
                        agrees(logits, temp, top_k, top_p, penalty, seed, steps, (seed >> 1) % 2 ? -1 : leader(logits));
                        seed++;
                    }
    };
    // Five levels put the zeros, of both signs, first, so the top-k cut falls among them.
    for (int levels : {5, 12, 0})
        for (uint64_t row = 0; row < 2; row++) grid(scores(1000, levels, 100 * row + (uint64_t)levels), {0, 1, 40, 1000}, 12);
    // The sampler ranks a nucleus with a heap only to 512 tokens and a top-k set only to 4096, so on 40000 a flat nucleus grows through the selections past the heap, and a top-k of 5000 is selected from every key.
    for (int levels : {5, 0}) grid(scores(40000, levels, 7 + (uint64_t)levels), {0, 40, 5000, 40000}, 3);
}

}  // namespace

int main() {
    try {
        temperature_zero();
        penalty();
        top_k();
        top_p();
        temperature();
        seeds();
        masked();
        against_reference();
        std::cout << "sampler: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
