// infer::sample against expectations taken from the definitions rather than from the code.
// Greedy is the argmax, the penalty divides a seen positive score and multiplies a seen negative one, top-k keeps the k best, the nucleus is the shortest ranked prefix whose probability reaches top_p, and a draw follows the softmax of the kept scores over the temperature.
// Frequencies are held to three binomial standard deviations per token from fixed seeds, so every run draws the same tokens.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
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
                const std::vector<uint32_t>& gen = no_history) {
    infer::RNG rng;
    return infer::sample(logits, 0.0f, 40, 0.95f, penalty, gen, rng);
}

std::vector<int> counts(const std::vector<float>& logits, float temp, int top_k, float top_p,
                        uint64_t seed) {
    infer::RNG rng;
    rng.seed(seed);
    std::vector<int> n(logits.size(), 0);
    for (int i = 0; i < draws; i++) {
        const uint32_t id = infer::sample(logits, temp, top_k, top_p, 1.0f, no_history, rng);
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

}  // namespace

int main() {
    try {
        temperature_zero();
        penalty();
        top_k();
        top_p();
        temperature();
        seeds();
        std::cout << "sampler: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
