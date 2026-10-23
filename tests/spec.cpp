// Speculative decoding's round (docs/SPECULATIVE.md, section 3) held to the run without drafts: test-only proposers, lookup and the embedded drafter, on dense, routed and hybrid models over one to four CPU stages (AGENTS.md, Tests).
// Then the history calls under a verify: every row equals single steps, a retract to any row continues as if never drafted, and a rerun that fails keeps the mark for a retry.
#include <chrono>
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
    void draft(infer::Sequence*, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
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
    void draft(infer::Sequence*, const std::vector<uint32_t>&, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        for (size_t i = 0; i < k; ++i) out.push_back((uint32_t)(rng() % vocab));
    }
};

// The end token, the vocabulary's last id and one past it, whose drafts after it are never fed.
struct Hostile final : Proposer {
    size_t vocab = 0;
    uint32_t end = 0;
    void draft(infer::Sequence*, const std::vector<uint32_t>&, size_t k, std::vector<uint32_t>& out) override {
        const uint32_t ids[] = {end, (uint32_t)vocab - 1, (uint32_t)vocab + 3, 1};
        out.assign(ids, ids + std::min<size_t>(k, 4));
    }
};

// A proposer that stands behind lookup as a model drafter does, drafting as `inner` does and counting its calls.
struct Behind final : Proposer {
    Proposer* inner = nullptr;
    size_t calls = 0;
    void draft(infer::Sequence* seq, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        ++calls;
        inner->draft(seq, h, k, out);
    }
    bool after_lookup() const override { return true; }
};

struct Run {
    std::vector<uint32_t> ids;
    int fed = 0;
    size_t passes = 0;         // the passes that generated them, counted at the first stage's submissions
    std::vector<float> next;   // the logits after the history and one more token, which hold the state the run left
    size_t drafted = 0;        // the drafts its verifies fed
    size_t own = 0, own_kept = 0;   // those of them its own history gave ahead of the proposer, and the ones kept
    bool rested = false;       // whether the proposer's acceptance ended resting
};

// The tensor width every model here is placed at (docs/TENSOR-SPLIT.md): above 1 each stage is a group of that many CPU backends, and a run is held to the run on one such group, since a group gives its own bits.
size_t group_width = 1;

// A model of `weights` placed over `stages` stages of one-thread CPU backends, with a mark of up to 17 rows where it keeps a state and with `drafter` the file's embedded drafter; `submits`, when given, counts the first backend's submissions.
std::unique_ptr<infer::Model> placed(const gguf::GGUFModel& weights, size_t stages, size_t* submits = nullptr, bool drafter = false) {
    std::vector<backend::BackendPtr> backends;
    for (const auto& h : hooked(stages * group_width)) backends.push_back(h);
    if (submits) static_cast<Hooked&>(*backends[0]).hook = [submits] { ++*submits; };
    infer::PlacementRequest request;
    for (size_t i = 0; i < backends.size(); ++i) request.names.push_back("cpu " + std::to_string(i));
    if (stages > 1) request.shares.assign(stages, 1);
    request.width = group_width;
    request.drafter = drafter;
    infer::ModelOptions options;
    options.mark_slots = 1;
    options.mark_rows = 32;
    return std::move(infer::place_model(infer::gguf_weights(weights), std::move(backends), request, options).model);
}

// With `embedded` and no proposer, the model is loaded with the file's embedded drafter and drafts with it.
// `shown`, when given, stands before the prompt in the history the proposers see, as an earlier turn the model was never fed, so a test chooses what lookup finds.
Run generate(const gguf::GGUFModel& weights, size_t stages, const std::vector<uint32_t>& prompt, const infer::GenParams& gp,
             Proposer* proposer, size_t draft_max, bool embedded = false, const std::vector<uint32_t>* shown = nullptr, bool behind = false) {
    size_t submits = 0;
    auto model = placed(weights, stages, &submits, embedded);
    const std::vector<float> logits = model->prefill(prompt);
    const size_t prefilled = submits;
    bpe::Tokenizer tok(weights);
    infer::RNG rng;
    rng.seed(gp.seed);
    std::unique_ptr<infer::spec::Drafting> drafting;
    std::unique_ptr<infer::spec::Embedded> own;
    if (embedded && !proposer) {
        own = std::make_unique<infer::spec::Embedded>(*model, behind);
        proposer = own.get();
    }
    if (proposer) {
        drafting = std::make_unique<infer::spec::Drafting>();
        drafting->proposer = proposer;
        drafting->draft_max = draft_max;
        if (shown) drafting->history = *shown;
        drafting->history.insert(drafting->history.end(), prompt.begin(), prompt.end());
    }
    Run r;
    r.ids = infer::generate(*model, tok, gp, rng, logits, {}, drafting.get());
    if (drafting) {
        std::vector<size_t> fed, kept;
        drafting->record.tally(infer::spec::Source::proposer, fed, kept);
        for (size_t n : fed) r.drafted += n;
        drafting->record.tally(infer::spec::Source::lookup, fed, kept);
        for (size_t n : fed) r.drafted += n, r.own += n;
        for (size_t n : kept) r.own_kept += n;
        r.rested = drafting->record.proposer.resting();
    }
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
                    for (size_t k : {1, 4, 7, 15, 16, 31}) {
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

// The embedded drafter of a hybrid model with an MTP block (docs/SPECULATIVE.md, section 7) against the run without drafts of the file without the block, on one, two and four CPU stages, greedy and seeded, at 1, 3 and 8 drafts a verify.
void embedded(const gguf::GGUFModel& plain, const gguf::GGUFModel& mtp, size_t vocab, const std::vector<size_t>& stages = {1, 2, 4}) {
    std::vector<infer::GenParams> samplers(2);
    samplers[0].temp = 0;
    samplers[1].temp = 0.8f, samplers[1].top_k = 40, samplers[1].top_p = 0.95f, samplers[1].seed = 5;
    for (infer::GenParams gp : samplers)
        for (size_t prompt_len : {10, 127, 129}) {
            gp.max_tokens = 40;
            const std::vector<uint32_t> prompt = prompt_of(3, prompt_len, (uint32_t)vocab);
            const Run want = generate(plain, 1, prompt, gp, nullptr, 0);
            require(want.ids == generate(mtp, 1, prompt, gp, nullptr, 0).ids, "the file with an MTP block generates otherwise without its drafter");
            for (size_t st : stages)
                for (size_t k : {1, 3, 8}) {
                    const Run got = generate(mtp, st, prompt, gp, nullptr, k, true);
                    const std::string at = "the embedded drafter, " + std::to_string(st) + " stages, " + std::to_string(k) + " drafts, prompt " +
                                           std::to_string(prompt_len) + (gp.temp > 0 ? ", seeded" : "");
                    same(want, got, at);
                    require(got.drafted > 0, at + ": no draft was fed");
                }
            // Shown its reply as an earlier turn, the drafter stands behind the reply's own continuation.
            for (size_t st : stages) {
                // Made as the tests above make it, the drafter is alone: it looks nowhere, whatever the history shows.
                require(generate(mtp, st, prompt, gp, nullptr, 3, true, &want.ids).own == 0, "the embedded drafter alone drafted from the history it was shown");
                const Run got = generate(mtp, st, prompt, gp, nullptr, 3, true, &want.ids, true);
                const std::string at = "the embedded drafter behind lookup, " + std::to_string(st) + " stages, prompt " + std::to_string(prompt_len) + (gp.temp > 0 ? ", seeded" : "");
                same(want, got, at);
                require(got.own_kept > 0 && got.drafted > got.own, at + ": " + std::to_string(got.own_kept) + " drafts of the reply's own continuation kept of " + std::to_string(got.drafted) + " fed");
            }
        }
}

// Lookup::convincing and spec::Record on their own: the latest earlier occurrence of the last eight tokens, a run that repeats into its own draft, and each source's rest apart.
void convincing_rule() {
    using infer::spec::Lookup;
    using infer::spec::Source;
    require(Lookup::kConvincing == 8, "the convincing match is " + std::to_string(Lookup::kConvincing) + " tokens, and the cases below are written for 8");
    std::vector<uint32_t> out{99};
    const std::vector<uint32_t> run{1, 2, 3, 4, 5, 6, 7, 8};
    auto history = [&](std::initializer_list<std::vector<uint32_t>> parts) {
        std::vector<uint32_t> h;
        for (const auto& p : parts) h.insert(h.end(), p.begin(), p.end());
        return h;
    };
    Lookup::convincing(history({run, {20, 21, 22}, run}), 2, out);
    require(out == std::vector<uint32_t>({20, 21}), "the continuation of an earlier run of eight was not drafted");
    Lookup::convincing(history({run, {20, 21, 22}, run}), 16, out);
    require(out == history({{20, 21, 22}, run, {20, 21, 22}, {1, 2}}), "a continuation that reaches the history's end did not go on into its own draft");
    Lookup::convincing(history({run, {20}, run, {30}, run}), 1, out);
    require(out == std::vector<uint32_t>({30}), "the earlier of two occurrences was drafted from, not the latest");
    Lookup::convincing(history({{2, 3, 4, 5, 6, 7, 8}, {20}, run}), 4, out);
    require(out.empty(), "a match of seven tokens was taken as convincing");
    Lookup::convincing(history({run, {20}, {9, 2, 3, 4, 5, 6, 7, 8}}), 4, out);
    require(out.empty(), "a run differing in its first token was taken as convincing");
    Lookup::convincing(run, 4, out);
    require(out.empty(), "a history of eight tokens drafted from itself");
    Lookup::convincing(std::vector<uint32_t>(9, 5), 3, out);
    require(out == std::vector<uint32_t>(3, 5), "a token repeated nine times did not draft itself on");
    Lookup::convincing(history({run, {20, 21, 22}, run}), 0, out);
    require(out.empty(), "a draft of no tokens gave some");

    // Eleven verifies that kept nothing rest a source, and the other still drafts; the tokens the other's verifies generate count its rest down.
    infer::spec::Record r;
    r.drafted = Source::lookup;
    for (int i = 0; i < 11; ++i) r.verified(0, 3);
    require(r.lookup[0].resting() && !r.proposer.resting(), "lookup's misses rested the proposer, or not lookup");
    require(infer::spec::lookup_first(r, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 0 && r.drafted == Source::proposer,
            "a resting lookup drafted ahead of the proposer");
    r.drafted = Source::proposer;
    for (int i = 0; i < 4; ++i) r.verified(3, 3);
    require(!r.lookup[0].resting(), "16 tokens the proposer's verifies generated did not end lookup's rest");
    require(infer::spec::lookup_first(r, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 3 && r.drafted == Source::lookup &&
                out == std::vector<uint32_t>({20, 21, 22}),
            "lookup did not draft ahead of the proposer once its rest ended");
    require(infer::spec::lookup_first(r, 3, 100, 100, 3, history({run, {20, 21, 22}}), out) == 0 && r.drafted == Source::proposer && out.empty(),
            "a history without a convincing run left a draft or lookup's source");
    infer::spec::Record p;
    for (int i = 0; i < 11; ++i) p.verified(0, 3);
    require(p.proposer.resting() && !p.lookup[0].resting(), "the proposer's misses rested lookup, or not the proposer");
    require(infer::spec::lookup_first(p, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 3, "a resting proposer stopped lookup");

    // Lookup's acceptance is kept by the match's length at the draft's start (F2DEV, 2026-10-23): eleven misses of an eight-token match rest that bucket alone, so a match of 32 tokens still drafts.
    std::vector<uint32_t> long_run(32);
    for (size_t i = 0; i < long_run.size(); ++i) long_run[i] = 100 + (uint32_t)i;
    infer::spec::Record keyed;
    keyed.drafted = Source::lookup;
    for (int i = 0; i < 11; ++i) keyed.verified(0, 3);
    require(infer::spec::lookup_first(keyed, 3, 100, 100, 3, history({long_run, {20, 21, 22}, long_run}), out) == 3 && keyed.drafted == Source::lookup &&
                out == std::vector<uint32_t>({20, 21, 22}),
            "a short match's misses rested the drafts of a match of 32 tokens");
    require(infer::spec::lookup_first(keyed, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 0 && keyed.drafted == Source::proposer,
            "a resting eight-token match drafted");
    // Only once every bucket rests does lookup_first skip the walk, as main skipped it while lookup's one acceptance rested: one bucket resting leaves the others drafting.
    require(!keyed.lookup_resting(), "one resting bucket made the history's drafts rest");
    infer::spec::Record rested;
    rested.drafted = Source::lookup;
    // In turns, since each verify counts down the other buckets' rests.
    for (int i = 0; i < 11; ++i)
        for (size_t bucket = 0; bucket < infer::spec::kMatchBuckets; ++bucket) {
            rested.bucket = bucket;
            rested.verified(0, 3);
        }
    require(rested.lookup_resting() && infer::spec::lookup_first(rested, 3, 100, 100, 3, history({long_run, {20, 21, 22}, long_run}), out) == 0 && out.empty() &&
                rested.drafted == Source::proposer,
            "every bucket resting, a history draft was still taken");

    // The walk gives the match's length, the occurrence extended backwards while it matches, at most 32, and that names the bucket a verify records into.
    std::vector<uint32_t> mid(20);
    for (size_t i = 0; i < mid.size(); ++i) mid[i] = 200 + (uint32_t)i;
    require(Lookup::convincing(history({run, {20, 21, 22}, run}), 2, out) == 8 && Lookup::convincing(history({mid, {20, 21, 22}, mid}), 2, out) == 20 &&
                Lookup::convincing(history({{7}, long_run, {20, 21, 22}, {7}, long_run}), 2, out) == 32 && Lookup::convincing(history({run, {20}}), 2, out) == 0,
            "the walk gave a match's length wrong");
    require(infer::spec::match_bucket(8) == 0 && infer::spec::match_bucket(15) == 0 && infer::spec::match_bucket(16) == 1 && infer::spec::match_bucket(31) == 1 &&
                infer::spec::match_bucket(32) == 2,
            "the match buckets are not 8 to 15, 16 to 31 and 32 or more");
    infer::spec::Record b;
    require(infer::spec::lookup_first(b, 3, 100, 100, 3, history({mid, {20, 21, 22}, mid}), out) == 3 && b.drafted == Source::lookup && b.bucket == 1,
            "a match of 20 tokens drafted in another bucket");
    b.verified(2, 3);
    std::vector<size_t> fed, kept;
    b.tally(Source::lookup, fed, kept);
    require(b.lookup[1].drafted() == std::vector<size_t>({1, 1, 1}) && b.lookup[0].drafted().empty() && b.lookup[2].drafted().empty() &&
                fed == std::vector<size_t>({1, 1, 1}) && kept == std::vector<size_t>({1, 1, 0}),
            "a verify recorded outside its match's bucket, or the tally did not sum the buckets");

    // What a round pays to look: the walk of a history that holds no convincing run, and the copy of that history the scheduler makes to look in, printed and held to nothing.
    for (size_t tokens : {size_t(32768), size_t(262144)}) {
        std::mt19937 rng(20261012u);
        std::vector<uint32_t> h(tokens), copy;
        for (uint32_t& t : h) t = rng() % 150000u;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) Lookup::convincing(h, 3, out);
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) copy.assign(h.begin(), h.end());
        const auto t2 = std::chrono::steady_clock::now();
        require(out.empty() && copy.size() == tokens, "a random history convinced");
        const auto t3 = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) infer::spec::lookup_first(rested, 3, 100, 100, 3, h, out);
        const auto t4 = std::chrono::steady_clock::now();
        std::cout << "spec: looking in " << tokens << " tokens: the walk " << std::chrono::duration<double, std::micro>(t1 - t0).count() / 100 << " us, the copy "
                  << std::chrono::duration<double, std::micro>(t2 - t1).count() / 100 << " us, every bucket resting " << std::chrono::duration<double, std::micro>(t4 - t3).count() / 100
                  << " us" << std::endl;
    }

    // A chain's steps are priced only where a request's drafts come from one: 34 ms a pass and 10 ms a row, a chain step 40 ms.
    infer::spec::PassCost c;
    c.known = true;
    c.base_ms = 34, c.row_ms = 10, c.step_ms = 40;
    std::vector<size_t> takes;
    const std::vector<uint8_t> chain{1}, own{0};
    infer::spec::draft_depths(c, 1, {{0.9, 0.9, 0.9}}, takes, &chain);
    require(takes == std::vector<size_t>({0}), "a chain dearer than its drafts was taken");
    infer::spec::draft_depths(c, 1, {{0.9, 0.9, 0.9}}, takes, &own);
    require(takes == std::vector<size_t>({3}), "drafts of a request's own history were priced with a chain");
    // Two requests in a pass, one of each: the chain is charged for its own request alone, 4 ms a step and 1 ms a chain, and every flag set gives the price without flags.
    c.step_ms = 4, c.step_row_ms = 1;
    const std::vector<std::vector<double>> two{{0.9, 0.9, 0.9}, {0.9, 0.9, 0.9}};
    const std::vector<uint8_t> mixed{1, 0}, both{1, 1};
    std::vector<size_t> flagged, plain;
    infer::spec::draft_depths(c, 2, two, flagged, &both);
    infer::spec::draft_depths(c, 2, two, plain);
    require(flagged == plain && plain == std::vector<size_t>({3, 3}), "every request chained was priced otherwise than without flags");
    infer::spec::draft_depths(c, 2, two, flagged, &mixed);
    require(flagged == std::vector<size_t>({3, 3}), "a pass of a chain's request and a history's did not take both");
    // A chain too dear for its own request leaves the history's drafts, which cost rows alone: 60 ms a step against rows of 10.
    c.step_ms = 60;
    infer::spec::draft_depths(c, 2, {{0.5, 0.5, 0.5}, {0.9, 0.9, 0.9}}, flagged, &mixed);
    infer::spec::draft_depths(c, 2, {{0.5, 0.5, 0.5}, {0.9, 0.9, 0.9}}, plain, &both);
    std::cout << "spec: a dear chain beside a history's drafts takes " << flagged[0] << " and " << flagged[1] << ", both chained " << plain[0] << " and " << plain[1] << std::endl;
    require(flagged[1] == 3, "a dear chain took a history's drafts with it");

    // A draft of the history is cut to what draft_length allows: the tokens left, the positions left and the columns.
    infer::spec::Record cut;
    require(infer::spec::lookup_first(cut, 16, 3, 100, 16, history({run, {20, 21, 22}, run}), out) == 2, "a draft of the history passed the tokens left");
    require(infer::spec::lookup_first(cut, 16, 100, 2, 16, history({run, {20, 21, 22}, run}), out) == 1, "a draft of the history passed the positions left");
    require(infer::spec::lookup_first(cut, 16, 100, 100, 2, history({run, {20, 21, 22}, run}), out) == 2, "a draft of the history passed the columns");
    require(infer::spec::lookup_first(cut, 2, 100, 100, 16, history({run, {20, 21, 22}, run}), out) == 2, "a draft of the history passed the most drafts");
}

// Lookup ahead of a proposer (docs/SPECULATIVE.md, section 8) against the run without drafts: the proposers are shown the reply as an earlier turn, so once eight tokens are generated the reply's own continuation is found and kept.
// Then a turn whose every run of eight is followed by a wrong token, so lookup misses until it rests and the proposer drafts on.
void lookup_first(const std::string& name, const gguf::GGUFModel& plain, size_t vocab, const std::vector<size_t>& stages) {
    std::vector<infer::GenParams> samplers(2);
    samplers[0].temp = 0;
    samplers[1].temp = 0.8f, samplers[1].top_k = 40, samplers[1].top_p = 0.95f, samplers[1].seed = 5;
    const std::vector<uint32_t> prompt = prompt_of(3, 10, (uint32_t)vocab);
    for (size_t s = 0; s < samplers.size(); ++s) {
        infer::GenParams gp = samplers[s];
        gp.max_tokens = 80;
        const Run want = generate(plain, 1, prompt, gp, nullptr, 0);
        // Every window of eight of the reply followed by a token the reply does not hold there.
        std::vector<uint32_t> misleading;
        for (size_t j = 0; j + 8 < want.ids.size(); ++j) {
            misleading.insert(misleading.end(), want.ids.begin() + (std::ptrdiff_t)j, want.ids.begin() + (std::ptrdiff_t)(j + 8));
            misleading.push_back((want.ids[j + 8] + 1) % (uint32_t)vocab);
        }
        for (size_t st : stages) {
            const std::string at = name + ", sampler " + std::to_string(s) + ", " + std::to_string(st) + " stages";
            Oracle wrong, right;
            for (Oracle* o : {&wrong, &right}) {
                o->truth = &want.ids;
                o->vocab = vocab;
            }
            wrong.prompt = want.ids.size() + prompt.size();
            wrong.miss = 0;
            for (size_t k : {1, 3, 16}) {
                Behind behind;
                behind.inner = &wrong;
                const Run got = generate(plain, st, prompt, gp, &behind, k, false, &want.ids);
                same(want, got, at + ", the reply shown, " + std::to_string(k) + " drafts");
                require(got.own_kept * 2 > got.own, at + ": " + std::to_string(got.own_kept) + " of " + std::to_string(got.own) + " drafts of the reply's own continuation kept");
                // Until eight tokens are generated the proposer is asked, a token a round, and never after.
                require(behind.calls >= 1 && behind.calls <= 9, at + ": the proposer was asked " + std::to_string(behind.calls) + " times beside a history that convinces from the eighth token on");
                if (k == 16) require(got.passes * 3 < want.passes, at + ": " + std::to_string(got.passes) + " passes with the reply's continuation drafted, " + std::to_string(want.passes) + " without");
            }
            // A proposer that does not stand behind lookup is asked every round, whatever its history shows.
            const Run alone = generate(plain, st, prompt, gp, &wrong, 3, false, &want.ids);
            same(want, alone, at + ", a proposer alone shown the reply");
            require(alone.own == 0, at + ": " + std::to_string(alone.own) + " drafts of the history fed for a proposer that does not stand behind lookup");
            right.prompt = misleading.size() + prompt.size();
            Behind behind;
            behind.inner = &right;
            const Run got = generate(plain, st, prompt, gp, &behind, 3, false, &misleading);
            same(want, got, at + ", a misleading turn shown");
            require(got.own >= 11 && got.own_kept == 0, at + ": " + std::to_string(got.own_kept) + " of " + std::to_string(got.own) + " misleading drafts kept");
            require(behind.calls > 0 && !got.rested && got.drafted > got.own, at + ": lookup's misses stopped the proposer, asked " + std::to_string(behind.calls) + " times");
        }
    }
    // A stop text and each limit inside a draft of the reply's own continuation, and an end token inside one.
    infer::GenParams g0 = samplers[0];
    g0.max_tokens = 40;
    const Run free = generate(plain, 1, prompt, g0, nullptr, 0);
    Oracle none;
    none.truth = &free.ids;
    none.vocab = vocab;
    none.prompt = SIZE_MAX / 2;
    Behind behind;
    behind.inner = &none;
    bpe::Tokenizer tok(plain);
    infer::GenParams stop = g0;
    stop.stop = tok.decode({free.ids[11], free.ids[12]});
    const Run stopped = generate(plain, 1, prompt, stop, nullptr, 0);
    same(stopped, generate(plain, stages.back(), prompt, stop, &behind, 16, false, &free.ids), name + ": a stop text inside a draft of the reply's own continuation");
    for (size_t limit = 9; limit <= 20; ++limit) {
        infer::GenParams lim = g0;
        lim.max_tokens = (int)limit;
        const Run want = generate(plain, 1, prompt, lim, nullptr, 0);
        same(want, generate(plain, stages.back(), prompt, lim, &behind, 16, false, &free.ids), name + ": a limit of " + std::to_string(limit) + " inside a draft of the reply's own continuation");
    }
    // An end token the reply first holds past its twelfth token, so drafts of its own continuation reach it.
    size_t at_end = 12;
    while (at_end < free.ids.size() && std::find(free.ids.begin(), free.ids.begin() + (std::ptrdiff_t)at_end, free.ids[at_end]) != free.ids.begin() + (std::ptrdiff_t)at_end) ++at_end;
    require(at_end < free.ids.size(), name + ": the reply holds no token first met past its twelfth");
    const uint32_t end = free.ids[at_end];
    const gguf::GGUFModel ended = ending(plain, end);
    const Run want = generate(ended, 1, prompt, g0, nullptr, 0);
    require(want.ids.size() == at_end, name + ": the end token did not end the reply where it stands");
    const Run got = generate(ended, stages.back(), prompt, g0, &behind, 16, false, &free.ids);
    same(want, got, name + ": an end token inside a draft of the reply's own continuation");
    require(got.own > 0, name + ": no draft of the reply's own continuation met the end token");
}

// A member's footprint with the embedded drafter (infer::footprint with a width) against counts made here, on the hybrid model a group of two splits whole: the block's shards, its projection and norms whole, the head and embedding whole.
void drafter_footprint(const gguf::GGUFModel& mtp) {
    const infer::ModelWeights w = infer::gguf_weights(mtp);
    const infer::ModelPlan plan = infer::plan_model(w, true);
    const infer::ModelOptions o;
    const infer::Footprint one = infer::footprint(w, plan, o), half = infer::footprint(w, plan, o, 2, 0);
    const auto bytes = [](const std::vector<infer::Matrix>& v) {
        size_t n = 0;
        for (const infer::Matrix& m : v) n += m.bytes;
        return n;
    };
    const auto q8 = [](size_t rows, size_t cols) { return rows * quant::row_bytes(quant::GGML_TYPE_Q8_0, cols); };
    const size_t E = (size_t)kHybridEven.embd, F = (size_t)kHybridEven.ff, D = (size_t)kHybridEven.head_dim, H = (size_t)kHybridEven.heads, HKV = (size_t)kHybridEven.kv_heads;
    const size_t V = (size_t)kHybridEven.vocab;
    // Five norms of E and two of a head, eh_proj E rows of 2 E.
    const size_t whole = (5 * E + 2 * D) * sizeof(float) + q8(E, 2 * E);
    require(bytes(one.drafter) == whole + q8(2 * H * D, E) + 2 * q8(HKV * D, E) + q8(E, H * D) + 2 * q8(F, E) + q8(E, F), "the drafter's weights on one device");
    require(bytes(half.drafter) == whole + q8(H * D, E) + 2 * q8(HKV * D / 2, E) + q8(E, H * D / 2) + 2 * q8(F / 2, E) + q8(E, F / 2) + q8(V, E),
            "a member's shards of the drafter's block, with the head whole");
    require(half.drafter_embedding.bytes == q8(V, E) && one.drafter_embedding.bytes == q8(V, E), "the drafter's embedding table is not whole on a member");
    require(one.drafter_cache - half.drafter_cache == infer::kv_tokens(plan, o) * infer::kv_bytes_per_position(plan, o) / 2, "a member's drafter KV is not its KV heads'");
}

// A CPU backend whose conv throws once where a test arms it, for a rerun that fails, and that remembers whether it was left recording unordered.
struct FailingConv : backend::CpuBackend {
    bool fail = false, unordered_on = false;
    void unordered(bool on) override { unordered_on = on; }
    void causal_conv_silu(backend::Slice out, backend::CSlice x, backend::CSlice w, size_t layer, const backend::StateView* views, size_t n_views) override {
        if (fail) { fail = false; throw std::runtime_error("injected"); }
        backend::CpuBackend::causal_conv_silu(out, x, w, layer, views, n_views);
    }
};

bool bits(const float* a, const float* b, size_t n) { return !std::memcmp(a, b, n * sizeof(float)); }

// The history calls under verifies: the hybrid model fed a fixed walk of tokens one step at a time against verifies of k + 1 of them, each retracted to keep j rows; every verify row and every row after a retract the bits of single steps.
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
    require(!conv->unordered_on, "a failed rerun left its backend recording unordered");
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
        convincing_rule();
        lookup_first("the dense model behind lookup", dense, (size_t)kCpu.vocab, {1, 2});
        lookup_first("the hybrid model behind lookup", hybrid, (size_t)kHybrid.vocab, {1, 2, 4});
        rounds("the dense model", dense, (size_t)kCpu.vocab, {1, 2});
        rounds("the routed model", routed, 16, {1, 3});
        rounds("the hybrid model", hybrid, (size_t)kHybrid.vocab, {1, 2, 4});
        for (size_t st : {1, 2, 4}) history(hybrid, (size_t)kHybrid.vocab, st);
        embedded(hybrid, served_hybrid(kHybrid, true), (size_t)kHybrid.vocab);
        history(dense, (size_t)kCpu.vocab, 2);
        failures(hybrid, (size_t)kHybrid.vocab);
        // The hybrid model over tensor groups of two CPUs, one stage and two: each member keeps the state of its own heads and reruns it from the mark on its own, and with the embedded drafter runs its shards of the block and reads the head whole.
        const gguf::GGUFModel even = served_hybrid(kHybridEven);
        group_width = 2;
        rounds("the hybrid model on groups of two", even, (size_t)kHybridEven.vocab, {1, 2});
        for (size_t st : {1, 2}) history(even, (size_t)kHybridEven.vocab, st);
        embedded(even, served_hybrid(kHybridEven, true), (size_t)kHybridEven.vocab, {1, 2});
        group_width = 1;
        drafter_footprint(served_hybrid(kHybridEven, true));
        std::cout << "spec: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "spec: FAIL: " << e.what() << "\n";
        return 1;
    }
}
