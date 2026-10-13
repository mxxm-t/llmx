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
    size_t own_depth = 0;           // the most drafts of its own history one verify fed
    bool rested = false;       // whether the proposer's acceptance ended resting
};

// The tensor width every model here is placed at (docs/TENSOR-SPLIT.md): above 1 each stage is a group of that many CPU backends, and a run is held to the run on one such group, since a group gives its own bits.
size_t group_width = 1;

// The rows a mark saves on the models placed here, 17 unless a case sets it.
size_t mark_rows = 17;

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
    options.mark_rows = mark_rows;
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
    if (drafting)
        for (const infer::spec::Acceptance& a : drafting->record.of)
            for (size_t n : a.drafted()) r.drafted += n;
    if (drafting) {
        const infer::spec::Acceptance& own_history = drafting->record[infer::spec::Source::lookup];
        for (size_t n : own_history.drafted()) r.own += n;
        for (size_t n : own_history.kept()) r.own_kept += n;
        r.own_depth = own_history.drafted().size();
        r.rested = drafting->record[infer::spec::Source::proposer].resting();
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
    require(r[Source::lookup].resting() && !r[Source::proposer].resting(), "lookup's misses rested the proposer, or not lookup");
    require(infer::spec::lookup_first(r, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 0 && r.drafted == Source::proposer,
            "a resting lookup drafted ahead of the proposer");
    r.drafted = Source::proposer;
    for (int i = 0; i < 4; ++i) r.verified(3, 3);
    require(!r[Source::lookup].resting(), "16 tokens the proposer's verifies generated did not end lookup's rest");
    require(infer::spec::lookup_first(r, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 3 && r.drafted == Source::lookup &&
                out == std::vector<uint32_t>({20, 21, 22}),
            "lookup did not draft ahead of the proposer once its rest ended");
    require(infer::spec::lookup_first(r, 3, 100, 100, 3, history({run, {20, 21, 22}}), out) == 0 && r.drafted == Source::proposer && out.empty(),
            "a history without a convincing run left a draft or lookup's source");
    infer::spec::Record p;
    for (int i = 0; i < 11; ++i) p.verified(0, 3);
    require(p[Source::proposer].resting() && !p[Source::lookup].resting(), "the proposer's misses rested lookup, or not the proposer");
    require(infer::spec::lookup_first(p, 3, 100, 100, 3, history({run, {20, 21, 22}, run}), out) == 3, "a resting proposer stopped lookup");

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
        std::cout << "spec: looking in " << tokens << " tokens: the walk " << std::chrono::duration<double, std::micro>(t1 - t0).count() / 100 << " us, the copy "
                  << std::chrono::duration<double, std::micro>(t2 - t1).count() / 100 << " us" << std::endl;
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
    // Its own cap is kOwnDrafts where the proposer's most drafts are fewer, and the proposer's where they are more.
    require(infer::spec::kOwnDrafts == 7, "the history's own cap is " + std::to_string(infer::spec::kOwnDrafts) + ", and the cases below are written for 7");
    require(infer::spec::lookup_first(cut, 2, 100, 100, 16, history({run, {20, 21, 22}, run}), out) == 7, "a draft of the history stopped short of its own cap");
    require(infer::spec::lookup_first(cut, 12, 100, 100, 16, history({run, {20, 21, 22}, run}), out) == 12, "a draft of the history stopped short of the proposer's most drafts");

    // The price by a pass's width: a pass of 2 to 4 rows costs about a millisecond a row, the fifth row 14 more and each after it 5, as a device's decode builds step.
    infer::spec::PassCost stepped;
    stepped.known = true;
    stepped.base_ms = 49.4, stepped.row_ms = 0.95;   // the line through passes of 2 to 4 rows, all a server may have seen
    infer::spec::PassCost line = stepped;
    stepped.width_ms = {0, 43.0, 51.2, 52.4, 53.1, 67.7, 73.1, 78.0, 83.1};
    const std::vector<uint8_t> own_only{0};
    const std::vector<double> good{0.9, 0.9, 0.9, 0.65, 0.65, 0.65, 0.65}, poor{0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.3};
    infer::spec::draft_depths(stepped, 1, {good}, takes, &own_only);
    require(takes == std::vector<size_t>({7}), "a history kept at 0.65 past its third draft did not take its seven, but " + std::to_string(takes[0]));
    infer::spec::draft_depths(stepped, 1, {poor}, takes, &own_only);
    require(takes == std::vector<size_t>({3}), "a history kept at 0.3 past its third draft crossed the step, taking " + std::to_string(takes[0]));
    infer::spec::draft_depths(line, 1, {poor}, takes, &own_only);
    require(takes == std::vector<size_t>({7}), "the line alone was expected to cross the step it cannot see, and took " + std::to_string(takes[0]));
    // The pass without drafts and the slope are the line's, and a measured width adds only what it costs beyond the line, so the rows a draft adds never cost less than the line's slope.
    // Sixteen decoders whose own width measured dear, 30 ms where the line gives 26, beside a width of 19 at the line: drafts kept at a half do not pay the line's slope and are not taken.
    infer::spec::PassCost crowded;
    crowded.known = true;
    crowded.base_ms = 10, crowded.row_ms = 1, crowded.step_ms = 0.1;
    infer::spec::PassCost crowded_line = crowded;
    crowded.width_ms.assign(32, 0.0);
    crowded.width_ms[16] = 30, crowded.width_ms[19] = 29;
    const std::vector<std::vector<double>> halves{{0.5, 0.5, 0.5}};
    std::vector<size_t> by_line;
    infer::spec::draft_depths(crowded_line, 16, halves, by_line);
    infer::spec::draft_depths(crowded, 16, halves, takes);
    require(by_line == std::vector<size_t>({0}) && takes == by_line, "a plain width measured above a wider one admitted " + std::to_string(takes[0]) + " drafts the line refuses");
    // A width measured below the line prices as the line.
    crowded.width_ms[16] = 0, crowded.width_ms[19] = 12;
    const std::vector<std::vector<double>> sure{{0.9, 0.9, 0.9}};
    infer::spec::draft_depths(crowded_line, 16, sure, by_line);
    infer::spec::draft_depths(crowded, 16, sure, takes);
    require(takes == by_line && by_line[0] > 0, "a width measured below the line changed the line's answer to " + std::to_string(takes[0]));
    // At one depth the table takes a subset of what the line takes: one decoder, line 10 + 1 a row, a width of 7 measured 35, and two requests to three drafts, kept at 0.9 and at 0.2 a position.
    // The line takes both; the table, whose rows there cost 4 ms, takes the first and not the second.
    infer::spec::PassCost lone;
    lone.known = true;
    lone.base_ms = 10, lone.row_ms = 1;
    infer::spec::PassCost stepped_at_seven = lone, stepped_at_four = lone;
    stepped_at_seven.width_ms.assign(16, 0.0), stepped_at_four.width_ms.assign(16, 0.0);
    stepped_at_seven.width_ms[7] = 35, stepped_at_four.width_ms[4] = 30;
    const std::vector<uint8_t> no_chain{0, 0};
    const std::vector<std::vector<double>> subset{{0.9, 0.9, 0.9}, {0.2, 0.2, 0.2}};
    infer::spec::draft_depths(lone, 1, subset, by_line, &no_chain);
    infer::spec::draft_depths(stepped_at_seven, 1, subset, takes, &no_chain);
    require(by_line == std::vector<size_t>({3, 3}) && takes == std::vector<size_t>({3, 0}),
            "at one depth the line took " + std::to_string(by_line[0]) + " and " + std::to_string(by_line[1]) + " and the table " + std::to_string(takes[0]) + " and " + std::to_string(takes[1]));
    // Across depths the two may differ, which is the table seeing a step: a width of 4 measured 30 makes the third draft dear, so the table stops at two.
    // There the second request's two drafts pay the line's slope, while the line goes to three and gives that request none.
    const std::vector<std::vector<double>> depths{{0.9, 0.9, 0.9}, {0.1, 0.1, 0.0}};
    infer::spec::draft_depths(lone, 1, depths, by_line, &no_chain);
    infer::spec::draft_depths(stepped_at_four, 1, depths, takes, &no_chain);
    require(by_line == std::vector<size_t>({3, 0}) && takes == std::vector<size_t>({2, 2}),
            "across depths the line took " + std::to_string(by_line[0]) + " and " + std::to_string(by_line[1]) + " and the table " + std::to_string(takes[0]) + " and " + std::to_string(takes[1]));
    // Whatever the widths measured, where the line's price takes no draft the table's takes none: 2000 tables of random widths, decoders and keeps.
    uint64_t state = 0x9E3779B97F4A7C15ull;
    const auto draw = [&] {
        state ^= state << 13, state ^= state >> 7, state ^= state << 17;
        return (double)(state >> 11) / (double)(1ull << 53);
    };
    size_t refused = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        infer::spec::PassCost c;
        c.known = true;
        c.base_ms = 10 + 40 * draw(), c.row_ms = 0.2 + 3 * draw(), c.step_ms = 4 * draw();
        const infer::spec::PassCost only_line = c;
        c.width_ms.assign(64, 0.0);
        for (double& w : c.width_ms) w = draw() < 0.5 ? 0 : 5 + 150 * draw();
        const size_t decoders = 1 + (size_t)(20 * draw());
        std::vector<std::vector<double>> keeps(1 + (size_t)(3 * draw()));
        for (auto& k : keeps) {
            k.resize(1 + (size_t)(7 * draw()));
            double p = draw();
            for (double& x : k) x = p, p *= 0.5 + 0.5 * draw();
        }
        infer::spec::draft_depths(only_line, decoders, keeps, by_line);
        infer::spec::draft_depths(c, decoders, keeps, takes);
        if (std::all_of(by_line.begin(), by_line.end(), [](size_t t) { return t == 0; })) {
            ++refused;
            require(std::all_of(takes.begin(), takes.end(), [](size_t t) { return t == 0; }), "measured widths admitted drafts the line's price refuses, trial " + std::to_string(trial));
        }
    }
    require(refused > 100, "the random tables seldom met a line that refuses: " + std::to_string(refused));
    // A chain's request at its three beside a history's at its seven, in one call.
    infer::spec::PassCost cheap;
    cheap.known = true;
    cheap.base_ms = 34, cheap.row_ms = 1, cheap.step_ms = 1;
    infer::spec::draft_depths(cheap, 2, {{0.9, 0.9, 0.9}, good}, takes, &mixed);
    require(takes == std::vector<size_t>({3, 7}), "a chain's request and a history's did not each take its own cap");
    // Passes of one width leave the line unknown, and that width is still priced by its own passes; a second width prices itself and the line the rest.
    infer::spec::PassTimes times;
    for (int i = 0; i < 10; ++i) times.pass(4, 53.0);
    require(!times.cost().known && std::fabs(times.pass_ms(4) - 53.0) < 1e-9 && std::fabs(times.cost().pass_ms(4) - 53.0) < 1e-9, "passes of one width did not price that width");
    for (int i = 0; i < 10; ++i) times.pass(8, 83.0);
    require(times.cost().known && std::fabs(times.pass_ms(8) - 83.0) < 1e-9 && std::fabs(times.pass_ms(4) - 53.0) < 1e-9, "a second width changed the first's price or missed its own");
    require(times.pass_ms(6) > 53.0 && times.pass_ms(6) < 83.0 && std::fabs(times.cost().pass_ms(6) - times.pass_ms(6)) < 1e-9, "a width never run was not priced by the line");
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
            // At three drafts for the proposer the history's own draft still runs to its cap of seven, and on a model whose mark saves five rows to four.
            {
                Behind short_proposer;
                short_proposer.inner = &wrong;
                const Run deep = generate(plain, st, prompt, gp, &short_proposer, 3, false, &want.ids);
                require(deep.own_depth == infer::spec::kOwnDrafts, at + ": the history's own draft ran to " + std::to_string(deep.own_depth) + " drafts beside a proposer of three");
                mark_rows = 5;
                Behind marked;
                marked.inner = &wrong;
                const Run cut_run = generate(plain, st, prompt, gp, &marked, 3, false, &want.ids);
                mark_rows = 17;
                same(want, cut_run, at + ", a mark of five rows");
                const size_t most = placed(plain, 1)->keeps_state() ? 4 : infer::spec::kOwnDrafts;
                require(cut_run.own_depth == most, at + ": under a mark of five rows the history's own draft ran to " + std::to_string(cut_run.own_depth) + " drafts, not " + std::to_string(most));
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
