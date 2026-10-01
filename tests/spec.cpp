// Speculative decoding's round (docs/SPECULATIVE.md, section 3) held to the run without drafts: infer::generate with test-only proposers that keep every draft, miss at a chosen draft, draw at random or propose hostile ids, and with lookup, on a dense, a routed and a hybrid model on one to four CPU stages, greedy and seeded, must give the ids, the fed history and the next logits of the run without drafts.
// Then the model's history calls under a verify: every row of a verify equals single steps, a retract to any row continues as if never drafted, the hybrid model's state rerun from its mark included, and a rerun that fails keeps the mark for a retry.
#include <iostream>
#include <random>

#include "inference/generate.hpp"
#include "server_harness.hpp"
#include "tiny_qwen.hpp"

namespace {

using infer::spec::Proposer;

// The tokens a run without drafts generated, proposed as drafts: each draft the true next token, but the one at `miss` (counted from the first draft of every verify) replaced by another, and past the run's end its end token where it has one.
struct Oracle final : Proposer {
    const std::vector<uint32_t>* truth = nullptr;
    size_t prompt = 0, miss = SIZE_MAX, vocab = 0;
    int64_t end = -1;
    void draft(const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        const size_t g = h.size() - prompt;
        for (size_t i = 0; i < k; ++i) {
            uint32_t id;
            if (g + i < truth->size()) id = (*truth)[g + i];
            else if (end >= 0) id = (uint32_t)end;
            else break;
            if (i == miss) id = (id + 1) % (uint32_t)vocab;
            out.push_back(id);
        }
    }
};

// Drafts drawn from a fixed sequence, inside the vocabulary.
struct Random final : Proposer {
    std::mt19937 rng{20261001u};
    size_t vocab = 0;
    void draft(const std::vector<uint32_t>&, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        for (size_t i = 0; i < k; ++i) out.push_back((uint32_t)(rng() % vocab));
    }
};

// The end token, the vocabulary's last id and one past it, whose drafts after it are never fed.
struct Hostile final : Proposer {
    size_t vocab = 0;
    uint32_t end = 0;
    void draft(const std::vector<uint32_t>&, size_t k, std::vector<uint32_t>& out) override {
        const uint32_t ids[] = {end, (uint32_t)vocab - 1, (uint32_t)vocab + 3, 1};
        out.assign(ids, ids + std::min<size_t>(k, 4));
    }
};

struct Run {
    std::vector<uint32_t> ids;
    int fed = 0;
    size_t passes = 0;         // the passes that generated them, counted at the first stage's submissions
    std::vector<float> next;   // the logits after the history and one more token, which hold the state the run left
};

// A model of `weights` placed over `stages` CPU backends of one thread, with a mark of up to 17 rows where it keeps a state; `submits`, when given, counts the first stage's submissions.
std::unique_ptr<infer::Model> placed(const gguf::GGUFModel& weights, size_t stages, size_t* submits = nullptr) {
    std::vector<backend::BackendPtr> backends;
    for (const auto& h : hooked(stages)) backends.push_back(h);
    if (submits) static_cast<Hooked&>(*backends[0]).hook = [submits] { ++*submits; };
    infer::PlacementRequest request;
    for (size_t i = 0; i < stages; ++i) request.names.push_back("cpu " + std::to_string(i));
    if (stages > 1) request.shares.assign(stages, 1);
    infer::ModelOptions options;
    options.mark_slots = 1;
    options.mark_rows = 17;
    return std::move(infer::place_model(infer::gguf_weights(weights), std::move(backends), request, options).model);
}

Run generate(const gguf::GGUFModel& weights, size_t stages, const std::vector<uint32_t>& prompt, const infer::GenParams& gp,
             Proposer* proposer, size_t draft_max) {
    size_t submits = 0;
    auto model = placed(weights, stages, &submits);
    const std::vector<float> logits = model->prefill(prompt);
    const size_t prefilled = submits;
    bpe::Tokenizer tok(weights);
    infer::RNG rng;
    rng.seed(gp.seed);
    std::unique_ptr<infer::spec::Drafting> drafting;
    if (proposer) {
        drafting = std::make_unique<infer::spec::Drafting>();
        drafting->proposer = proposer;
        drafting->draft_max = draft_max;
        drafting->history = prompt;
    }
    Run r;
    r.ids = infer::generate(*model, tok, gp, rng, logits, {}, drafting.get());
    r.passes = submits - prefilled;
    r.fed = model->n_tokens();
    r.next = model->step(7);
    return r;
}

void same(const Run& want, const Run& got, const std::string& what) {
    require(got.ids == want.ids, what + ": the generated ids differ from the run without drafts");
    require(got.fed == want.fed, what + ": fed " + std::to_string(got.fed) + " tokens, without drafts " + std::to_string(want.fed));
    require(got.next.size() == want.next.size() && !std::memcmp(got.next.data(), want.next.data(), got.next.size() * sizeof(float)),
            what + ": the history left differs from the run without drafts");
}

// The model's weights with `end` as its end token.
gguf::GGUFModel ending(gguf::GGUFModel w, uint32_t end) {
    gguf::MetaValue v;
    v.vtype = gguf::V_UINT32;
    v.u = end;
    w.kv.push_back({"tokenizer.ggml.eos_token_id", v});
    return w;
}

// Every proposer, sampler and placement against the run without drafts on one CPU.
void rounds(const std::string& name, const gguf::GGUFModel& plain, size_t vocab, const std::vector<size_t>& stages) {
    std::vector<infer::GenParams> samplers(3);
    samplers[0].temp = 0;
    samplers[1].temp = 0.8f, samplers[1].top_k = 40, samplers[1].top_p = 0.95f, samplers[1].seed = 5;
    samplers[2].temp = 1.5f, samplers[2].top_k = 0, samplers[2].top_p = 1.0f, samplers[2].penalty = 1.3f, samplers[2].seed = 9;
    // The greedy run's 21st token ends a reply under the model given it as its end token.
    infer::GenParams g0 = samplers[0];
    g0.max_tokens = 40;
    const Run greedy = generate(plain, 1, prompt_of(3, 10, (uint32_t)vocab), g0, nullptr, 0);
    const uint32_t end = greedy.ids[20];
    const gguf::GGUFModel ended = ending(plain, end);
    for (const gguf::GGUFModel* w : {&plain, &ended})
        for (size_t s = 0; s < samplers.size(); ++s)
            for (size_t prompt_len : {10, 127, 128, 129}) {
                const bool has_end = w == &ended;
                infer::GenParams gp = samplers[s];
                gp.max_tokens = 40;
                const std::vector<uint32_t> prompt = prompt_of(3, prompt_len, (uint32_t)vocab);
                const Run want = generate(*w, 1, prompt, gp, nullptr, 0);
                const std::string at = name + (has_end ? " with an end token" : "") + ", sampler " + std::to_string(s) + ", prompt " +
                                       std::to_string(prompt_len);
                for (size_t st : stages) {
                    Oracle oracle;
                    oracle.truth = &want.ids;
                    oracle.prompt = prompt.size();
                    oracle.vocab = vocab;
                    oracle.end = has_end ? (int64_t)end : -1;
                    for (size_t k : {1, 4, 16}) {
                        for (size_t miss : {SIZE_MAX, size_t(0), size_t(1), size_t(2), k - 1}) {
                            oracle.miss = miss;
                            const Run got = generate(*w, st, prompt, gp, &oracle, k);
                            same(want, got, at + ", " + std::to_string(st) + " stages, " + std::to_string(k) + " drafts missing at " + std::to_string((int64_t)miss));
                            // Drafts that are all kept are verified, so the reply takes fewer passes than tokens.
                            if (miss == SIZE_MAX && k == 16 && want.ids.size() > 8)
                                require(got.passes * 4 < want.passes, at + ": " + std::to_string(got.passes) + " passes with drafts all kept, " +
                                                                          std::to_string(want.passes) + " without");
                        }
                    }
                    Random random;
                    random.vocab = vocab;
                    same(want, generate(*w, st, prompt, gp, &random, 4), at + ", random drafts");
                    Hostile hostile;
                    hostile.vocab = vocab;
                    hostile.end = end;
                    same(want, generate(*w, st, prompt, gp, &hostile, 4), at + ", hostile drafts");
                    infer::spec::Lookup lookup;
                    same(want, generate(*w, st, prompt, gp, &lookup, 8), at + ", lookup");
                }
            }
    // A stop text that a draft's verify passes, and a limit at every token of a verify of oracle drafts.
    const Run free = generate(plain, 1, prompt_of(3, 10, (uint32_t)vocab), g0, nullptr, 0);
    bpe::Tokenizer tok(plain);
    infer::GenParams stop = g0;
    stop.stop = tok.decode({free.ids[11], free.ids[12]});
    const Run stopped = generate(plain, 1, prompt_of(3, 10, (uint32_t)vocab), stop, nullptr, 0);
    require(stopped.ids.size() < free.ids.size(), name + ": the stop text did not end the reply");
    for (size_t limit = 1; limit <= 12; ++limit) {
        infer::GenParams lim = g0;
        lim.max_tokens = (int)limit;
        const Run want = generate(plain, 1, prompt_of(3, 10, (uint32_t)vocab), lim, nullptr, 0);
        Oracle oracle;
        oracle.truth = &free.ids;
        oracle.prompt = 10;
        oracle.vocab = vocab;
        same(want, generate(plain, stages.back(), prompt_of(3, 10, (uint32_t)vocab), lim, &oracle, 16), name + ": a limit of " + std::to_string(limit));
    }
    for (size_t miss : {SIZE_MAX, size_t(0), size_t(3)}) {
        Oracle oracle;
        oracle.truth = &free.ids;
        oracle.prompt = 10;
        oracle.vocab = vocab;
        oracle.miss = miss;
        same(stopped, generate(plain, stages.back(), prompt_of(3, 10, (uint32_t)vocab), stop, &oracle, 16), name + ": a stop text inside a verify");
    }
}

// A CPU backend whose conv throws once where a test arms it, for a rerun that fails.
struct FailingConv : backend::CpuBackend {
    bool fail = false;
    void causal_conv_silu(backend::Slice out, backend::CSlice x, backend::CSlice w, size_t layer, const backend::StateView* views, size_t n_views) override {
        if (fail) { fail = false; throw std::runtime_error("injected"); }
        backend::CpuBackend::causal_conv_silu(out, x, w, layer, views, n_views);
    }
};

bool bits(const float* a, const float* b, size_t n) { return !std::memcmp(a, b, n * sizeof(float)); }

// The history calls under verifies: the hybrid model fed a fixed walk of tokens one step at a time against verifies of k + 1 of them, each retracted to keep j rows, repeated through a long generation; every verify row and every row after a retract the bits of single steps.
void history(const gguf::GGUFModel& w, size_t vocab, size_t stages) {
    const std::vector<uint32_t> walk = prompt_of(5, 300, (uint32_t)vocab), prompt = prompt_of(2, 20, (uint32_t)vocab);
    auto single = placed(w, 1);
    single->prefill(prompt);
    std::vector<std::vector<float>> want;
    for (uint32_t id : walk) want.push_back(single->step((int)id));
    const size_t n = want[0].size();
    auto model = placed(w, stages);
    model->prefill(prompt);
    size_t at = 0, round = 0;
    while (at + 17 < walk.size()) {
        const size_t k = 1 + round % 16, keep = round % 5 == 4 ? k + 1 : (round * 7) % (k + 1);
        require(model->mark(), "a mark was refused");
        const float* rows = model->step(walk.data() + at, k + 1);
        for (size_t i = 0; i <= k; ++i)
            require(bits(rows + i * n, want[at + i].data(), n), "verify row " + std::to_string(i) + " of " + std::to_string(k + 1) + " differs from a step");
        const size_t reached = model->retract(prompt.size() + at + keep);
        require(reached == prompt.size() + at + keep, "a retract inside a mark did not reach its length");
        at += keep;
        // A retract to the mark keeps nothing: the next token is fed again by a single step.
        if (!keep) {
            require(bits(model->step((int)walk[at]).data(), want[at].data(), n), "a step after a retract to the mark differs");
            ++at;
        }
        ++round;
    }
    require(bits(model->step((int)walk[at]).data(), want[at].data(), n), "the step after the last retract differs");
}

// A rerun that fails keeps the mark, so the retract is called again and reaches its length with the bits of single steps; and what a mark refuses.
void failures(const gguf::GGUFModel& w, size_t vocab) {
    const std::vector<uint32_t> walk = prompt_of(5, 20, (uint32_t)vocab), prompt = prompt_of(2, 20, (uint32_t)vocab);
    auto single = placed(w, 1);
    single->prefill(prompt);
    std::vector<std::vector<float>> want;
    for (uint32_t id : walk) want.push_back(single->step((int)id));
    auto conv = std::make_shared<FailingConv>();
    conv->set_threads(1);
    infer::ModelOptions options;
    options.mark_slots = 1;
    options.mark_rows = 8;
    infer::Model model(infer::gguf_weights(w), conv, options);
    model.prefill(prompt);
    require(model.mark(), "a mark was refused");
    model.step(walk.data(), 6);
    conv->fail = true;
    bool threw = false;
    try { model.retract(prompt.size() + 3); } catch (const std::runtime_error&) { threw = true; }
    require(threw, "an injected rerun failure did not fail the retract");
    require(model.retract(prompt.size() + 3) == prompt.size() + 3, "the retry of a failed retract did not reach its length");
    require(bits(model.step((int)walk[3]).data(), want[3].data(), want[3].size()), "the step after a retried retract differs");
    // Refusals: a second mark, a second pass after a mark, a pass of more rows than a mark saves; a mark with none free gives false.
    require(model.mark(), "a mark was refused");
    bool second = false;
    try { model.mark(); } catch (const std::logic_error&) { second = true; }
    require(second, "a second mark was not refused");
    bool wide = false;
    try { model.step(walk.data(), 9); } catch (const std::logic_error&) { wide = true; }
    require(wide, "a pass after a mark beyond its rows was not refused");
    model.step(walk.data() + 4, 2);
    bool twice = false;
    try { model.step(walk.data() + 6, 1); } catch (const std::logic_error&) { twice = true; }
    require(twice, "a second pass after a mark was not refused");
    require(model.retract(model.n_tokens()) == (size_t)model.n_tokens(), "a retract keeping every row moved the history");
    infer::ModelOptions none;
    infer::Model unmarked(infer::gguf_weights(w), backend::make_cpu_backend(), none);
    unmarked.prefill(prompt);
    require(!unmarked.mark(), "a mark with none free was given");
}

} // namespace

int main() {
    try {
        const gguf::GGUFModel dense = served(kCpu), hybrid = served_hybrid(kHybrid);
        const gguf::GGUFModel routed = with_tokens(tiny_qwen_moe(3, 512, false), 16);
        rounds("the dense model", dense, (size_t)kCpu.vocab, {1, 2});
        rounds("the routed model", routed, 16, {1, 3});
        rounds("the hybrid model", hybrid, (size_t)kHybrid.vocab, {1, 2, 4});
        for (size_t st : {1, 2, 4}) history(hybrid, (size_t)kHybrid.vocab, st);
        history(dense, (size_t)kCpu.vocab, 2);
        failures(hybrid, (size_t)kHybrid.vocab);
        std::cout << "spec: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "spec: FAIL: " << e.what() << "\n";
        return 1;
    }
}
