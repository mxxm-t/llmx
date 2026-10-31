// infer::sample against expectations taken from the definitions rather than from the code: greedy is the argmax, the penalty divides a seen nonnegative score and multiplies a negative one, top-k keeps the k best.
// Frequencies are held to three binomial standard deviations from fixed seeds, and every draw, with the generator's state after it, to a slow reference that sorts all scores but a masked one and sums the softmax in id order.
#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <random>

#include "inference/sampler.hpp"
#include "inference/spec.hpp"
#include "server/policy.hpp"

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
    return infer::sample(logits.data(), logits.size(), 0.0f, 40, 0.95f, penalty, gen, rng, masked);
}

std::vector<int> counts(const std::vector<float>& logits, float temp, int top_k, float top_p,
                        uint64_t seed, int64_t masked = -1) {
    infer::RNG rng;
    rng.seed(seed);
    std::vector<int> n(logits.size(), 0);
    for (int i = 0; i < draws; i++) {
        const uint32_t id = infer::sample(logits.data(), logits.size(), temp, top_k, top_p, 1.0f, no_history, rng, masked);
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
    const std::vector<float> row = {3.0f, 1.6f, 0.0f};
    require(infer::sample(row.data(), row.size(), 1.0f, 1, 1.0f, 2.0f, seen, rng) == 1,
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
        ids.push_back(infer::sample(logits.data(), logits.size(), 1.0f, 0, 1.0f, 1.0f, no_history, rng));
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
    require(infer::sample(row.data(), row.size(), s, 1, no_history, rng) == 1, "the end id was masked without ignore_eos");
    s.ignore_eos = true;
    require(infer::sample(row.data(), row.size(), s, 1, no_history, rng) == 3, "ignore_eos drew the end id");
    require(infer::sample(row.data(), row.size(), s, -1, no_history, rng) == 1, "ignore_eos masked a token of a model without an end id");
}

// infer::sample as its definition reads, slowly: every token but a masked one ranked by a full sort on score and then id, the kept tokens' softmax summed in id order, and with top-p the nucleus summed and walked best first.
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

// n scores from a seed: with `levels` they take that many values a quarter apart, so ties are everywhere and fall across every cut, and a zero is -0 or +0 at random.
// Without, they spread as a model's do, with one in 200 far ahead.
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
        const uint32_t got = infer::sample(logits.data(), logits.size(), temp, top_k, top_p, penalty, gen, fast, masked);
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

// infer::accept against the loop without drafts written out here: row i sampled after the picks before it, one draw a row, the reply ending at an end token or the limit, and the drafts deciding only where sampling stops.
// Rows of twelve scores on five levels, so greedy meets ties; drafts are the loop's own picks with one replaced at a random place, or a random id, with an end token and a limit falling at every row.
void acceptance() {
    std::mt19937 gen_rows(20261001u);
    const size_t n = 12;
    const infer::Sampling settings[] = {
        [] { infer::Sampling s; s.temp = 0; return s; }(),
        [] { infer::Sampling s; s.temp = 0.9f; s.top_k = 0; s.top_p = 0.9f; return s; }(),
        [] { infer::Sampling s; s.temp = 1.3f; s.top_k = 5; s.top_p = 1; s.penalty = 1.4f; return s; }(),
    };
    for (int trial = 0; trial < 3000; ++trial) {
        const infer::Sampling& s = settings[trial % 3];
        const size_t k = 1 + (size_t)trial % 8;
        std::vector<float> rows((k + 1) * n);
        for (float& v : rows) v = (float)(gen_rows() % 5);
        const int32_t end = trial % 4 == 0 ? (int32_t)(gen_rows() % n) : -1;
        const size_t limit = 1 + gen_rows() % (k + 3);
        const uint64_t seed = 1 + (uint64_t)trial;
        // The loop without drafts over these rows, which also gives the drafts it would keep.
        std::vector<uint32_t> gen0{(uint32_t)(trial % n)}, picks;
        infer::RNG r0;
        r0.seed(seed);
        for (size_t i = 0; i <= k; ++i) {
            const uint32_t y = infer::sample(rows.data() + i * n, n, s, end, gen0, r0);
            picks.push_back(y);
            if (end >= 0 && y == (uint32_t)end) break;
            gen0.push_back(y);
            if (gen0.size() - 1 >= limit) break;
        }
        std::vector<uint32_t> drafts(picks.begin(), picks.end());
        drafts.resize(k, 0);
        const size_t miss = gen_rows() % (k + 2);
        if (miss < k) drafts[miss] = (drafts[miss] + 1 + (uint32_t)(gen_rows() % (n - 1))) % (uint32_t)n;
        // Where the loop without drafts stops: the first pick that ends the reply or differs from its draft, or row k.
        size_t want_rows = 0;
        for (size_t i = 0; i < picks.size(); ++i) {
            want_rows = i + 1;
            const bool ended = (end >= 0 && picks[i] == (uint32_t)end) || i + 1 >= limit;
            if (ended || i == k || picks[i] != drafts[i]) break;
        }
        std::vector<uint32_t> gen{(uint32_t)(trial % n)};
        infer::RNG r;
        r.seed(seed);
        const infer::Accepted a = infer::accept(rows.data(), n, drafts.data(), k, s, end, gen, r, [&](uint32_t id) {
            if (end >= 0 && id == (uint32_t)end) return false;
            gen.push_back(id);
            return gen.size() - 1 < limit;
        });
        require(a.rows == want_rows && a.last == picks[want_rows - 1], "accept stopped at another row than the loop without drafts, trial " + std::to_string(trial));
        // The draws are the loop's: replaying its first want_rows rows leaves the generator where accept left it.
        std::vector<uint32_t> gen1{(uint32_t)(trial % n)};
        infer::RNG r1;
        r1.seed(seed);
        for (size_t i = 0; i < want_rows; ++i) {
            const uint32_t y = infer::sample(rows.data() + i * n, n, s, end, gen1, r1);
            if (!(end >= 0 && y == (uint32_t)end)) gen1.push_back(y);
        }
        require(r1.s == r.s && gen1 == gen, "accept drew otherwise than the loop without drafts, trial " + std::to_string(trial));
    }
}

bool close_to(double a, double b, double tolerance = 1e-12) { return std::fabs(a - b) <= tolerance; }

// spec::Acceptance, the one acceptance average: from 2, each empty verify moves it an eighth of the way to 0, so ten still leave it at half a draft or more and the eleventh rests the request.
// 16 tokens stepped then end the rest, a verify keeping three moves the average back above the break-even, and its counts by draft position keep a prefix of each verify.
void acceptance_rest() {
    infer::spec::Acceptance a;
    for (int i = 0; i < 10; ++i) {
        a.verified(0, 3);
        require(!a.resting() && infer::spec::draft_length(3, 100, 100, 3, a) == 3, "the acceptance rested after " + std::to_string(i + 1) + " empty verifies");
    }
    a.verified(0, 3);
    require(a.resting() && infer::spec::draft_length(3, 100, 100, 3, a) == 0, "eleven empty verifies did not rest the request");
    for (int i = 0; i < 15; ++i) a.stepped();
    require(a.resting(), "the rest ended before 16 tokens");
    a.stepped();
    require(!a.resting() && infer::spec::draft_length(3, 100, 100, 3, a) == 3, "16 tokens did not end the rest");
    require(infer::spec::draft_length(3, 100, 100, 2, a) == 2 && infer::spec::draft_length(3, 100, 100, 0, a) == 0, "a draft past the decode columns its pass leaves");
    a.verified(3, 3);
    require(!a.resting(), "a verify keeping three left the average below the break-even");
    a.verified(0, 1);
    a.verified(1, 4);
    require(!a.resting(), "the average forgot the verify that kept three");
    require(a.drafted() == std::vector<size_t>({14, 13, 13, 1}) && a.kept() == std::vector<size_t>({2, 1, 1, 0}) && a.reached() == std::vector<size_t>({14, 2, 1, 0}),
            "the drafts fed and kept by position are not the verifies' counts");
    // The chance of keeping the drafts up to a position: each position's share over the verifies that reached it, weighed with four at the prior's, which is weighed with four at the last reached position's, one half for the first draft.
    // Position 0 was reached by all 14 verifies, 1 by the two that kept a draft (the 12th and the 14th), 2 by the 12th, and 3 by none, which goes on at position 2's chance; each verify weighs 0.99 of the next.
    infer::spec::Acceptance prior;
    prior.verified(1, 2);
    const double f = 0.99, w12 = f * f;
    double all = 0;
    for (int t = 0; t < 14; ++t) all += std::pow(f, 13 - t);
    const double c0 = (w12 + 1 + 4 * ((1 + 4 * 0.5) / (1 + 4))) / (all + 4), c1 = (w12 + 4 * ((0 + 4 * c0) / (1 + 4))) / (w12 + 1 + 4),
                 c2 = (w12 + 4 * c1) / (w12 + 4);
    require(close_to(a.keeps(0, prior), c0) && close_to(a.keeps(1, prior), c0 * c1) && close_to(a.keeps(2, prior), c0 * c1 * c2) &&
                close_to(a.keeps(5, prior), c0 * c1 * c2 * c2 * c2 * c2),
            "a position's chance of being kept is not the product of the reached verifies' shares weighed with the prior's");
    const infer::spec::Acceptance none;
    require(close_to(none.keeps(0, none), 0.5) && close_to(none.keeps(5, none), 0.5), "a record of no verifies does not keep its first draft at one half and the rest whenever it is");
}

// The pass price (spec::PassTimes, spec::draft_depths), on passes that cost 34 ms and 10 ms a row and chains of 4 ms a step and 1 ms a chain.
void pass_price() {
    infer::spec::PassTimes t;
    for (int i = 0; i < 20; ++i) t.pass(1, 44);
    require(!t.cost().known, "passes of one width priced a row");
    t.pass(4, 74);
    t.pass(8, 114);
    for (int i = 0; i < 20; ++i) t.chain(3, 1 + i % 3, 3 * (4.0 + (1 + i % 3)));
    const infer::spec::PassCost c = t.cost();
    require(c.known && close_to(c.base_ms, 34, 1e-6) && close_to(c.row_ms, 10, 1e-6) && close_to(c.step_ms, 4, 1e-6) && close_to(c.step_row_ms, 1, 1e-6),
            "the measured price is not the passes' and the chains' lines");
    // Passes of one width long enough forget the spread, so the price is measured again.
    for (int i = 0; i < 1000; ++i) t.pass(8, 114);
    require(!t.cost().known, "a thousand passes of one width kept their price");

    std::vector<size_t> takes;
    // One decoder that keeps most drafts takes three: 1 + 2.4 tokens in 44 + 30 + 3 x 5 ms beats one in 44.
    infer::spec::draft_depths(c, 1, {{0.8, 0.8, 0.8}}, takes);
    require(takes == std::vector<size_t>({3}), "a lone decoder that keeps most drafts took " + std::to_string(takes[0]));
    // A decoder that keeps few drafts takes none: their rows cost more than they give.
    infer::spec::draft_depths(c, 1, {{0.1, 0.1, 0.1}}, takes);
    require(takes == std::vector<size_t>({0}), "a decoder that keeps few drafts drafted");
    // Eight decoders: 8 tokens in 114 ms, which three drafts kept at one half each do not pay for, so none drafts.
    infer::spec::draft_depths(c, 8, std::vector<std::vector<double>>(8, {0.5, 0.5, 0.5}), takes);
    require(takes == std::vector<size_t>(8, 0), "eight decoders drafted where their rows cost more than they give");
    // Beside one that keeps few drafts, two that keep most draft, and the depth is the one most tokens a millisecond give, within each request's cap.
    infer::spec::draft_depths(c, 3, {{0.9, 0.9}, {0.1, 0.1, 0.1}, {0.99, 0.99, 0.99}}, takes);
    require(takes == std::vector<size_t>({2, 0, 3}), "the depth did not follow the requests that pay for their rows");
    // A price not known, or rows that cost nothing, give every request its cap, unless every pass measured had the rows the caps give, when the pass drafts nothing, so another width is measured.
    infer::spec::draft_depths(infer::spec::PassCost{}, 8, {{0.1}, {0.1, 0.1}}, takes);
    require(takes == std::vector<size_t>({1, 2}), "a price not known held drafts back");
    infer::spec::PassCost seen;
    seen.seen_rows = 4;
    infer::spec::draft_depths(seen, 1, {{0.9, 0.9, 0.9}}, takes);
    require(takes == std::vector<size_t>({0}), "passes all of the capped width did not measure one without drafts");
    seen.seen_rows = 1;
    infer::spec::draft_depths(seen, 1, {{0.9, 0.9, 0.9}}, takes);
    require(takes == std::vector<size_t>({3}), "passes all without drafts did not measure one with them");
    for (int i = 0; i < 20; ++i) t.pass(8, 114);
    require(!t.cost().known && close_to(t.cost().seen_rows, 8, 1e-3), "passes of one width do not give that width as the rows seen");
}

// The pass cost as the scheduler gives it to the depth rule (Scheduler::propose): passes of each of `widths`, `rounds` times, taking `ms` of stage time at band 0 and `host_ms` more of the server's.
// The table prices its band's widths through PassTimes::server_ms and the line the rest; `band` is the band read.
infer::spec::PassCost table_cost(const std::function<double(size_t)>& ms, const std::vector<size_t>& widths, size_t most, double host_ms = 0, size_t band = 0, int rounds = 10) {
    server::PriceTable table;
    infer::spec::PassTimes t;
    for (int round = 0; round < rounds; ++round)
        for (size_t rows : widths) {
            server::PassWork w;
            w.decode(rows, 0);
            const double cell = table.decode_cell_ms(0, rows);
            table.observe(w, ms(rows));
            t.pass(rows, ms(rows) + host_ms);
            if (cell > 0) t.reconcile(cell, ms(rows) + host_ms);
        }
    infer::spec::PassCost c = t.cost();
    c.rows_ms = table.decode_curve(band, most);
    for (double& x : c.rows_ms) x = t.server_ms(x);
    return c;
}

std::vector<size_t> widths_from(size_t first, size_t last, size_t skip_from = 0, size_t skip_to = 0) {
    std::vector<size_t> w;
    for (size_t r = first; r <= last; ++r)
        if (r < skip_from || r > skip_to) w.push_back(r);
    return w;
}

// A pass cost with a step, as production's Qwen3.8-Flash-Next 2x2 measured it at 128k tokens: 2.4 ms a row to 7 rows, then 10 ms more from 8 (docs/STATUS.md).
// A line through those passes prices 7 rows too dear and 9 too cheap; the price table's cells put the depth before the step, where it gives the most tokens a millisecond.
void stepped_price() {
    const double host = 1.5;
    const auto ms = [](size_t rows) { return 12.0 + 2.4 * (double)rows + (rows >= 8 ? 10.0 : 0.0); };
    const infer::spec::PassCost c = table_cost(ms, widths_from(1, 20), 20, host);
    require(c.known, "passes of 1 to 20 rows did not give a price");
    require(close_to(c.rows_ms[5], ms(5) + host, 1e-6) && close_to(c.rows_ms[9], ms(9) + host, 1e-6), "the table's stage times were not brought to the server's time");
    // Kept given the ones before at 0.95 to the 8th draft and 0.9 after: six drafts (7 rows) give the most tokens a millisecond at the stepped cost, nine at the line.
    std::vector<double> keeps(16);
    double chance = 1;
    for (size_t j = 0; j < keeps.size(); ++j) chance *= j < 8 ? 0.95 : 0.9, keeps[j] = chance;
    size_t best = 0;
    double rate = 1 / (ms(1) + host);
    for (size_t k = 1; k <= keeps.size(); ++k) {
        double tokens = 1;
        for (size_t j = 0; j < k; ++j) tokens += keeps[j];
        if (tokens / (ms(k + 1) + host) > rate) rate = tokens / (ms(k + 1) + host), best = k;
    }
    require(best == 6, "the stepped cost's best depth is not six");
    std::vector<size_t> takes;
    infer::spec::draft_depths(c, 1, {keeps}, takes);
    require(takes[0] == best, "the depth was " + std::to_string(takes[0]) + " where the measured passes give the most tokens a millisecond at six");
    infer::spec::PassCost line = c;
    line.rows_ms.clear();
    std::vector<size_t> on_line;
    infer::spec::draft_depths(line, 1, {keeps}, on_line);
    require(on_line[0] == 9, "the line's depth was " + std::to_string(on_line[0]) + ", not nine");
    // A band the table has not measured is priced by the line, so a reply crossing into a deeper band is not priced at a shallower band's cells.
    const infer::spec::PassCost deeper = table_cost(ms, widths_from(1, 20), 20, host, 1);
    infer::spec::draft_depths(deeper, 1, {keeps}, takes);
    require(takes == on_line, "an unmeasured band gave depth " + std::to_string(takes[0]) + " against " + std::to_string(on_line[0]) + " on the line");
    // Widths the table has not measured: between measured cells the line through them, never the nearest cell's price, and past the last measured the line.
    const infer::spec::PassCost gap = table_cost(ms, widths_from(1, 12, 7, 9), 16, host);
    const double between = ms(6) + host + (ms(10) - ms(6)) * 2.0 / 4.0;
    require(close_to(gap.rows_ms[8], between, 1e-6) && !close_to(infer::spec::pass_ms(gap, 8), ms(6) + host, 0.5) && !close_to(infer::spec::pass_ms(gap, 8), ms(10) + host, 0.5),
            "8 rows between measured 6 and 10 were priced " + std::to_string(infer::spec::pass_ms(gap, 8)) + ", not " + std::to_string(between));
    require(gap.rows_ms[14] == 0 && close_to(infer::spec::pass_ms(gap, 14), gap.rows_ms[12] + gap.row_ms * 2, 1e-9), "14 rows past the measured cells were not priced from 12 rows' price on the line's slope");
    // A line below the table's last width: past it a pass costs that width's price and the line's slope more, never less, so a draft there never looks free.
    infer::spec::PassCost low;
    low.known = true;
    low.base_ms = 1;
    low.row_ms = 0.5;
    low.rows_ms = {0, 0, 0, 30, 32, 40};
    require(close_to(infer::spec::pass_ms(low, 7), 41, 1e-12) && infer::spec::pass_ms(low, 6) > infer::spec::pass_ms(low, 5), "past the last measured width the price fell to the line");
    require(close_to(infer::spec::pass_ms(low, 2), 29.5, 1e-12) && close_to(infer::spec::pass_ms(low, 4), 32, 1e-12), "before the first measured width the price was not continued from it");
}

// A decode cell's first measure pays its width's one-time costs, as a kernel build made on first use did for 2 rows on an MI50 (docs/STATUS.md): 1 to 4 rows cost 49, 55, 92 and 97 ms, the first 2-row pass 96.
// That pass alone must not price 2 rows, which would move a request kept at 0.6 and 0.3 from one draft to two and keep it from ever measuring 2 rows again; a second 2-row pass prices them.
void first_measure() {
    const auto ms = [](size_t rows) { return rows == 1 ? 49.0 : rows == 2 ? 55.0 : rows == 3 ? 92.0 : 97.0; };
    const std::vector<std::vector<double>> keeps = {{0.6, 0.3, 0.15}};
    server::PriceTable table;
    infer::spec::PassTimes t;
    const auto run = [&](size_t rows, double took) {
        server::PassWork w;
        w.decode(rows, 0);
        const double cell = table.decode_cell_ms(0, rows);
        table.observe(w, took);
        t.pass(rows, took);
        if (cell > 0) t.reconcile(cell, took);
    };
    const auto cost = [&] {
        infer::spec::PassCost c = t.cost();
        c.rows_ms = table.decode_curve(0, 4);
        for (double& x : c.rows_ms) x = t.server_ms(x);
        return c;
    };
    for (int round = 0; round < 10; ++round)
        for (size_t rows : {1, 3, 4}) run(rows, ms(rows));
    run(2, 96.0);
    std::vector<size_t> takes;
    infer::spec::draft_depths(cost(), 1, keeps, takes);
    require(takes[0] == 1, "one dear first 2-row pass moved the depth to " + std::to_string(takes[0]) + " drafts, where 1 to 4 rows at 49, 55, 92 and 97 ms give one");
    run(2, ms(2));
    const infer::spec::PassCost c = cost();
    require(close_to(c.rows_ms[2], ms(2), 1e-6), "a second 2-row pass priced 2 rows at " + std::to_string(c.rows_ms[2]) + ", not its 55 ms");
}

// Cheaper passes never draft shallower at equal chances: every width cheaper by one factor keeps the depth, and a cheaper row at the same fixed cost never lowers it.
// A cheaper fixed cost alone may lower it, since the drafts then pay back less of it; that is the price's point, not a fault.
void cheaper_not_shallower() {
    infer::RNG rng;
    rng.seed(7);
    const auto unit = [&]() { return (double)(rng.next() >> 11) / 9007199254740992.0; };
    const auto depth = [](const std::function<double(size_t)>& ms, const std::vector<double>& keeps) {
        std::vector<size_t> takes;
        infer::spec::draft_depths(table_cost(ms, widths_from(1, 24), 24), 1, {keeps}, takes);
        return takes[0];
    };
    for (int trial = 0; trial < 200; ++trial) {
        const double base = 5 + 30 * unit(), row = 0.5 + 4 * unit(), jump = 15 * unit();
        const size_t at = 2 + (size_t)(unit() * 20);
        const double scale = 0.5 + 0.45 * unit(), cheaper_row = row * unit();
        const auto ms = [&](size_t r) { return base + row * (double)r + (r >= at ? jump : 0.0); };
        std::vector<double> keeps(23);
        double chance = 1;
        const double a = 0.5 + 0.5 * unit(), b = 0.5 + 0.5 * unit();
        for (size_t j = 0; j < keeps.size(); ++j) chance *= j < 8 ? a : b, keeps[j] = chance;
        const size_t before = depth(ms, keeps), scaled = depth([&](size_t r) { return scale * ms(r); }, keeps);
        const size_t line = depth([&](size_t r) { return base + row * (double)r; }, keeps), cheaper = depth([&](size_t r) { return base + cheaper_row * (double)r; }, keeps);
        require(scaled == before && cheaper >= line, "cheaper passes drafted shallower: depth " + std::to_string(before) + " and " + std::to_string(scaled) + " with every width " +
                                                         std::to_string(scale) + " as dear, " + std::to_string(line) + " and " + std::to_string(cheaper) + " with a row " +
                                                         std::to_string(cheaper_row) + " ms in place of " + std::to_string(row));
    }
}

// A draft position's chance follows the verifies that reached it, a draft's depth censoring what it says of later positions (docs/SPECULATIVE.md, section 3).
// As production met it at --draft-max 63: the first verifies fed 63 drafts and missed within three, and later ones fed eight and kept them; the deep misses say nothing of position 9, so a request keeping its eight drafts is priced deeper.
void censored_depth() {
    infer::spec::Acceptance prior, own;
    for (int i = 0; i < 80; ++i) prior.verified((size_t)(i % 3), 63);
    for (int i = 0; i < 2000; ++i) prior.verified(i % 50 ? 8 : 7, 8);
    for (int i = 0; i < 200; ++i) own.verified(i % 50 ? 8 : 7, 8);
    std::vector<double> keeps(63);
    for (size_t j = 0; j < keeps.size(); ++j) keeps[j] = own.keeps(j, prior);
    infer::spec::PassCost c;
    c.known = true;
    c.base_ms = 16.7;
    c.row_ms = 3.3;
    std::vector<size_t> takes;
    infer::spec::draft_depths(c, 1, {keeps}, takes);
    require(takes[0] > 15, "a request keeping its eight drafts was held at " + std::to_string(takes[0]) + " by deep verifies that missed early");
    // A request whose drafts fall away, kept given the ones before at 0.9, 0.5 and 0.2 over verifies of three, still stops within them: a position no verify reached goes on at about 0.2, not above it.
    infer::spec::Acceptance falling;
    for (int i = 0; i < 1000; ++i) falling.verified(i % 100 < 10 ? 0 : i % 100 < 55 ? 1 : i % 100 < 91 ? 2 : 3, 3);
    for (size_t j = 0; j < keeps.size(); ++j) keeps[j] = falling.keeps(j, falling);
    // About the last hundred verifies count, and each hundred ends with the nine that kept three, so position 2's 0.2 reads as 0.239.
    require(keeps[3] < keeps[2] * 0.25, "a position past the measured ones was priced above the last measured chance");
    infer::spec::draft_depths(c, 1, {keeps}, takes);
    require(takes[0] >= 1 && takes[0] <= 3, "a request whose drafts fall away took " + std::to_string(takes[0]));
}

// A frontier set by a few early verifies relaxes (docs/SPECULATIVE.md, section 3): five verifies kept 20 drafts and missed the 21st, then 400 fed 20 and kept them all.
// Nothing feeds position 21 again, so only forgetting the early verifies moves it: its chance must come back to within 5 percent of position 20's times the conditional measured before it.
void frontier_relaxes() {
    infer::spec::Acceptance tally;
    for (int i = 0; i < 5; ++i) tally.verified(20, 21);
    for (int i = 0; i < 400; ++i) tally.verified(20, 20);
    const infer::spec::Acceptance none;
    const double at19 = none.keeps(18, tally), at20 = none.keeps(19, tally), at21 = none.keeps(20, tally);
    require(at21 >= 0.95 * at20 * (at20 / at19), "position 21 stayed at " + std::to_string(at21 / at20) + " of position 20 after 400 verifies kept every draft up to it");
}

// Verifies in a loop at a price of `base_ms` and `row_ms` a row: each feeds the depth draft_depths takes and keeps each draft by `truth`, its chance given the ones before.
// It starts after the verifies `before` gives as (kept, fed), and gives the depth each verify fed.
std::vector<size_t> closed_loop(const std::vector<double>& truth, size_t verifies, uint64_t seed, const std::vector<std::pair<size_t, size_t>>& before,
                                double base_ms = 16.7, double row_ms = 3.3) {
    infer::spec::Acceptance own, server;
    for (const auto& b : before) own.verified(b.first, b.second), server.verified(b.first, b.second);
    infer::spec::PassCost c;
    c.known = true;
    c.base_ms = base_ms;
    c.row_ms = row_ms;
    infer::RNG rng = seeded(seed);
    std::vector<std::vector<double>> keeps(1, std::vector<double>(truth.size()));
    std::vector<size_t> takes, fed;
    for (size_t v = 0; v < verifies; ++v) {
        for (size_t j = 0; j < truth.size(); ++j) keeps[0][j] = own.keeps(j, server);
        infer::spec::draft_depths(c, 1, keeps, takes);
        size_t kept = 0;
        while (kept < takes[0] && rng.unit() < truth[kept]) ++kept;
        own.verified(kept, takes[0]);
        server.verified(kept, takes[0]);
        fed.push_back(takes[0]);
    }
    return fed;
}

// The frontier over eight seeds: kept at 0.99 everywhere, the depth passes 21 within 400 verifies (301 to 343 measured).
// With position 21 truly kept at 0.2, at most 40 of 2000 verifies feed it (12 to 17 measured).
void frontier_loop() {
    const std::vector<std::pair<size_t, size_t>> early(5, {20, 21});
    std::vector<double> truth(infer::spec::kMaxDrafts, 0.99);
    for (uint64_t seed = 1; seed <= 8; ++seed) {
        const std::vector<size_t> fed = closed_loop(truth, 400, seed, early);
        require(std::any_of(fed.begin(), fed.end(), [](size_t k) { return k > 21; }), "a depth kept at 0.99 everywhere stayed within 21 for 400 verifies after five early misses at 21");
    }
    truth[20] = 0.2;
    for (uint64_t seed = 1; seed <= 8; ++seed) {
        const std::vector<size_t> fed = closed_loop(truth, 2000, seed, early);
        const size_t feeding = (size_t)std::count_if(fed.begin(), fed.end(), [](size_t k) { return k > 20; });
        require(feeding <= 40, std::to_string(feeding) + " of 2000 verifies fed position 21, which is kept at 0.2");
    }
}

// The depth that gives the most tokens a millisecond at the price, where position j is kept with the product of truth up to j.
size_t best_depth(const std::vector<double>& truth, double base_ms, double row_ms) {
    double chance = 1, tokens = 1, best = 1 / (base_ms + row_ms);
    size_t depth = 0;
    for (size_t k = 1; k <= truth.size(); ++k) {
        chance *= truth[k - 1];
        tokens += chance;
        const double rate = tokens / (base_ms + row_ms * (double)(k + 1));
        if (rate > best) best = rate, depth = k;
    }
    return depth;
}

// On code edits a draft that survived to a position is usually inside a long copy, so the chance of keeping it given the ones before rises with depth: 0.8 at the first, 0.99 from the 20th.
// At a price where 63 drafts give the most tokens a millisecond, the depth opens to 63 from a fresh start and, after 300 verifies that missed within three, within 200 verifies (60 to 128 measured), and stays there.
void rising_loop() {
    std::vector<double> truth(infer::spec::kMaxDrafts);
    for (size_t j = 0; j < truth.size(); ++j) truth[j] = 0.8 + 0.19 * (double)std::min<size_t>(j, 19) / 19.0;
    const double base = 30, row = 0.1;
    require(best_depth(truth, base, row) == 63, "the rising profile's best depth at its price is not 63");
    std::vector<std::pair<size_t, size_t>> shorts;
    for (int i = 0; i < 300; ++i) shorts.push_back({(size_t)(i % 3), 3});
    for (int start = 0; start < 2; ++start)
        for (uint64_t seed = 1; seed <= 4; ++seed) {
            const std::vector<size_t> fed = closed_loop(truth, 1000, seed, start ? shorts : std::vector<std::pair<size_t, size_t>>{}, base, row);
            const size_t first = (size_t)(std::find_if(fed.begin(), fed.end(), [](size_t k) { return k >= 63; }) - fed.begin());
            const size_t deep = (size_t)std::count_if(fed.begin() + 500, fed.end(), [](size_t k) { return k >= 63; });
            require(first <= (start ? 200u : 0u) && deep >= 475, "a rising profile opened to 63 at verify " + std::to_string(first) + " and held it in " + std::to_string(deep) + " of the last 500");
        }
}

// Each bucket prices from its own verifies, at censored_depth's price: 0.97 a position (32+ on production's reruns) stays within 26 in 9 of 10 verifies.
// 0.998 a position (its verbatim copies) takes 60 or more in two of three (225 of 300 on the worst seed measured).
void bucket_depths() {
    const std::vector<double> mixed(infer::spec::kMaxDrafts, 0.97), copies(infer::spec::kMaxDrafts, 0.998);
    for (uint64_t seed = 1; seed <= 4; ++seed) {
        const std::vector<size_t> a = closed_loop(mixed, 600, seed, {}), b = closed_loop(copies, 600, seed, {});
        const size_t a_shallow = (size_t)std::count_if(a.begin() + 300, a.end(), [](size_t k) { return k <= 26; });
        const size_t b_deep = (size_t)std::count_if(b.begin() + 300, b.end(), [](size_t k) { return k >= 60; });
        require(a_shallow >= 270 && b_deep >= 200, "of the last 300 verifies, " + std::to_string(a_shallow) + " at 0.97 a position stayed within 26 and " + std::to_string(b_deep) + " at 0.998 took 60 or more");
    }
}

// The additive penalties of the compatible APIs, subtracted from a seen token's score before the temperature: presence once however often the token was seen, frequency once for each time.
infer::Sampling settings(float temp, int top_k, float top_p) {
    infer::Sampling s;
    s.temp = temp;
    s.top_k = top_k;
    s.top_p = top_p;
    return s;
}

uint32_t pick(const std::vector<float>& logits, const infer::Sampling& s, const std::vector<uint32_t>& gen = no_history, uint64_t seed = 1) {
    infer::RNG rng;
    rng.seed(seed);
    return infer::sample(logits.data(), logits.size(), s, -1, gen, rng);
}

std::vector<int> counts_of(const std::vector<float>& logits, const infer::Sampling& s, const std::vector<uint32_t>& gen, uint64_t seed) {
    infer::RNG rng;
    rng.seed(seed);
    std::vector<int> n(logits.size(), 0);
    for (int i = 0; i < draws; i++) n[infer::sample(logits.data(), logits.size(), s, -1, gen, rng)]++;
    return n;
}

void additive_penalties() {
    const std::vector<float> row = {3.0f, 2.5f, 0.0f};
    infer::Sampling s = settings(0.0f, 40, 1.0f);
    s.presence_penalty = 0.6f;
    require(pick(row, s, {0}) == 1, "a presence penalty did not lower a seen leader");
    require(pick(row, s, {0, 0, 0}) == 1, "a presence penalty of 0.6 on a token seen three times did not lower it");
    s.presence_penalty = 0.4f;
    require(pick(row, s, {0, 0, 0}) == 0, "a presence penalty was taken once for each time the token was seen");
    require(pick(row, s, {}) == 0, "a presence penalty lowered a token not seen");
    s.presence_penalty = 0.0f;
    s.frequency_penalty = 0.2f;
    require(pick(row, s, {0, 0, 0}) == 1, "a frequency penalty of 0.2 on a token seen three times did not take 0.6");
    require(pick(row, s, {0}) == 0, "a frequency penalty of 0.2 on a token seen once took more than 0.2");
    s.frequency_penalty = 0.1f;
    require(pick(row, s, {0, 0, 0}) == 0, "a frequency penalty took more than once for each time");
    // A negative penalty favors a seen token.
    s.frequency_penalty = 0.0f;
    s.presence_penalty = -1.0f;
    require(pick({1.0f, 1.5f, 0.0f}, s, {0}) == 0, "a negative presence penalty did not raise a seen token");
    // It is subtracted from the score before the temperature, and beside the repetition penalty, which is applied first: 3 / 2 - 0.4 is 1.1 against 1.5.
    s = settings(0.0f, 40, 1.0f);
    s.penalty = 2.0f;
    s.presence_penalty = 0.4f;
    require(pick({3.0f, 1.2f, 0.0f}, s, {0}) == 1, "the additive penalty was not taken from the repetition-penalized score");
    require(pick({3.0f, 1.0f, 0.0f}, s, {0}) == 0, "the additive penalty took more than its amount");
    // Drawn at temperature 2, the scores 0 and 1 with a presence penalty of 1 on the second are the softmax of 0 and 0.
    const std::vector<float> pair = {0.0f, 1.0f};
    s = settings(2.0f, 0, 1.0f);
    s.presence_penalty = 1.0f;
    fits("a presence penalty before the temperature", counts_of(pair, s, {1}, 9), {0.5, 0.5});
    // Neutral values are no penalty: the draws and the generator's state are those of the call without the three.
    for (uint64_t seed = 1; seed < 40; seed++) {
        infer::RNG a = seeded(seed), b = seeded(seed);
        const std::vector<float> logits = scores(300, 0, seed);
        const std::vector<uint32_t> gen = {1, 2, 2, 7};
        infer::Sampling n = settings(0.8f, 40, 0.95f);
        n.penalty = 1.2f;
        const uint32_t x = infer::sample(logits.data(), logits.size(), n, -1, gen, a);
        const uint32_t y = infer::sample(logits.data(), logits.size(), 0.8f, 40, 0.95f, 1.2f, gen, b, -1);
        require(x == y && a.s == b.s, "neutral additive penalties and min_p changed a draw");
    }
}

// min_p keeps the tokens whose probability, at the temperature, is at least min_p times the likeliest token's, after top-k and top-p.
void min_p() {
    const std::vector<float> logits = {0.0f, 1.2f, -1.0f, 1.7f, -0.5f};
    infer::Sampling s = settings(1.0f, 0, 1.0f);
    // Against the likeliest, id 3, the others weigh 0.607, 0.183, 0.111 and 0.067 at temperature 1.
    s.min_p = 0.5f;
    fits("min_p 0.5", counts_of(logits, s, no_history, 11), softmax(logits, 1.0, {3, 1}));
    s.min_p = 0.15f;
    fits("min_p 0.15", counts_of(logits, s, no_history, 12), softmax(logits, 1.0, {3, 1, 0}));
    // At temperature 2 the weights are their square roots, 0.78, 0.43, 0.33 and 0.26, so the same 0.4 keeps three where temperature 1 keeps two.
    s = settings(2.0f, 0, 1.0f);
    s.min_p = 0.4f;
    fits("min_p 0.4 at temperature 2", counts_of(logits, s, no_history, 13), softmax(logits, 2.0, {3, 1, 0}));
    s = settings(1.0f, 0, 1.0f);
    s.min_p = 0.4f;
    fits("min_p 0.4 at temperature 1", counts_of(logits, s, no_history, 14), softmax(logits, 1.0, {3, 1}));
    // After top-k and top-p: top_k 2 with min_p 0.15 keeps the two best; a nucleus of 0.65 keeps two and min_p 0.7 cuts the second.
    s = settings(1.0f, 2, 1.0f);
    s.min_p = 0.15f;
    fits("top_k 2 with min_p 0.15", counts_of(logits, s, no_history, 15), softmax(logits, 1.0, {3, 1}));
    s = settings(1.0f, 0, 0.65f);
    s.min_p = 0.7f;
    fits("top_p 0.65 with min_p 0.7", counts_of(logits, s, no_history, 16), softmax(logits, 1.0, {3}));
    // min_p 1 keeps the likeliest token alone, whatever the temperature, and a tie keeps both.
    s = settings(5.0f, 40, 0.95f);
    s.min_p = 1.0f;
    for (uint64_t seed = 1; seed < 20; seed++) require(pick(logits, s, no_history, seed) == 3, "min_p 1 drew a token below the likeliest");
    fits("min_p 1 on a tie", counts_of({2.0f, 2.0f, 1.0f}, s, no_history, 17), {0.5, 0.5, 0.0});
    // A masked leader leaves the next best as the likeliest.
    infer::RNG rng = seeded(3);
    s = settings(1.0f, 0, 1.0f);
    s.min_p = 1.0f;
    s.ignore_eos = true;
    for (int i = 0; i < 20; i++) require(infer::sample(logits.data(), logits.size(), s, 3, no_history, rng) == 1, "min_p ranked the masked token");
}

// A history draft whose match opens a bucket no verify has reached starts from the next shorter bucket's record (spec::Record::seed, through spec::lookup_first), so a verbatim copy drafts at that bucket's depth from its first verify.
// A bucket with verifies of its own keeps them.
void seeded_bucket() {
    infer::spec::Record own, server;
    for (int i = 0; i < 300; ++i) {
        own.lookup[2].verified(i % 50 ? 63 : 62, 63);
        server.lookup[2].verified(i % 50 ? 63 : 62, 63);
    }
    own.lookup[1].verified(0, 3);
    std::vector<uint32_t> history(200);
    for (size_t i = 0; i < history.size(); ++i) history[i] = (uint32_t)(1000 + i);
    history.insert(history.end(), history.begin(), history.begin() + 150);
    std::vector<uint32_t> out;
    const size_t k = infer::spec::lookup_first(own, 63, 1000, 1000, 63, history, out);
    require(k == 63 && own.drafted == infer::spec::Source::lookup && own.bucket == 3, "a copy of 150 tokens drafted " + std::to_string(k) + " in bucket " + std::to_string(own.bucket));
    server.seed(3);
    std::vector<double> keeps(63);
    for (size_t j = 0; j < keeps.size(); ++j) keeps[j] = own.lookup[3].keeps(j, server.lookup[3]);
    infer::spec::PassCost c;
    c.known = true;
    c.base_ms = 16.7;
    c.row_ms = 3.3;
    std::vector<size_t> takes;
    infer::spec::draft_depths(c, 1, {keeps}, takes);
    require(takes[0] == 63, "a copy's first verify in a fresh bucket took " + std::to_string(takes[0]) + " drafts beside a shorter bucket's 63");
    own.seed(1);
    require(own.lookup[1].drafted().size() == 3 && own.lookup[1].kept()[0] == 0, "a bucket with a verify of its own was seeded over");
}

int main() {
    try {
        temperature_zero();
        penalty();
        additive_penalties();
        min_p();
        top_k();
        top_p();
        temperature();
        seeds();
        masked();
        against_reference();
        acceptance();
        acceptance_rest();
        pass_price();
        censored_depth();
        stepped_price();
        first_measure();
        cheaper_not_shallower();
        frontier_relaxes();
        frontier_loop();
        rising_loop();
        bucket_depths();
        seeded_bucket();
        std::cout << "sampler: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
