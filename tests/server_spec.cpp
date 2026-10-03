// Drafting in the scheduler (docs/SPECULATIVE.md, section 3): requests that verify drafts beside each other give, token for token, the ids and log-probabilities they give alone without drafts.
// Over the synthetic Q8_0 model with lookup, and the hybrid model with an MTP block with its embedded drafter and with lookup, on one CPU and over a two-CPU split, greedy, sampled, capped, uncapped and ended by a stop string; and a pool too small for every request, where they pause and resume.
#include <iostream>
#include <numeric>

#include "server_harness.hpp"

namespace {

constexpr size_t kDraftMax = 3;

// A CPU backend whose decode kernels hold 16 columns, so a pass of several decoding requests has columns to spare for their drafts.
struct Wide : backend::CpuBackend {
    Wide() { set_threads(1); }
    size_t decode_columns() const override { return 16; }
};

std::vector<backend::BackendPtr> wide(size_t n) {
    std::vector<backend::BackendPtr> v;
    for (size_t i = 0; i < n; ++i) v.push_back(std::make_shared<Wide>());
    return v;
}

using MakeProposer = std::function<std::unique_ptr<infer::spec::Proposer>(infer::Model&)>;

size_t total(const std::vector<size_t>& v) { return std::accumulate(v.begin(), v.end(), size_t(0)); }

// Each request alone on a fresh model without drafts, then alone with them, then all at once with them: every reply its reply alone, and drafts fed and kept, which a scheduler `priced` by its passes' timing need not feed.
void against_alone(const Make& plain, const Make& drafting, const MakeProposer& proposer, const bpe::Tokenizer& tok, size_t pool, size_t max_seqs,
                   size_t passes, const std::vector<Req>& reqs, const std::string& what, bool pauses = false, bool priced = false,
                   size_t draft_max = kDraftMax) {
    std::vector<Reply> alone;
    for (const Req& r : reqs) {
        auto model = plain(pool, 0);
        alone.push_back(serve(*model, tok, max_seqs, {{r}}, nullptr, passes).back());
    }
    for (size_t i = 0; i < reqs.size(); ++i) {
        auto model = drafting(pool, 0);
        auto p = proposer(*model);
        same(alone[i], serve(*model, tok, max_seqs, {{reqs[i]}}, nullptr, passes, 0, p.get(), draft_max, priced).back(),
             what + ", request " + std::to_string(i) + " alone with drafts");
    }
    auto model = drafting(pool, 0);
    auto p = proposer(*model);
    server::Scheduler::Stats stats;
    const std::vector<Reply> together = serve(*model, tok, max_seqs, {reqs}, &stats, passes, 0, p.get(), draft_max, priced);
    for (size_t i = 0; i < reqs.size(); ++i) same(alone[i], together[i], what + ", request " + std::to_string(i) + " beside the others with drafts");
    require(priced || (total(stats.drafted) > 0 && total(stats.kept) > 0),
            what + ": " + std::to_string(total(stats.drafted)) + " drafts fed and " + std::to_string(total(stats.kept)) + " kept");
    if (pauses) require(stats.pauses >= 1, what + ": nothing paused");
    std::cout << "server-spec: " << what << ": " << total(stats.kept) << " of " << total(stats.drafted) << " drafts kept, " << stats.pauses << " pauses" << std::endl;
}

// A wide CPU backend that runs `hook` as the model submits its work (Hooked), so a case acts while a pass is in flight.
struct WideHooked : Hooked {
    WideHooked() { set_threads(1); }
    size_t decode_columns() const override { return 16; }
};

// A request cancelled while its pass is in flight, at each of several submissions so some cancels find a verify, ends cancelled with its history back at its last pick; a request after it gives its reply on a fresh model without drafts, and every block comes back.
void cancelled_in_flight(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, const MakeProposer& proposer, bool drafter,
                         size_t states, const std::string& what) {
    const Req v{prompt_of(1, 40, vocab), 300}, after{prompt_of(2, 30, vocab), 40};
    auto fresh = on(weights, [] { return cpus(1); }, states)(1024, 0);
    const Reply want = serve(*fresh, tok, 3, {{after}}).back();
    for (size_t at = 12; at < 20; ++at) {
        const std::string case_what = what + ", cancelled at submission " + std::to_string(at);
        auto device = std::make_shared<WideHooked>();
        auto model = on(weights, [device] { return std::vector<backend::BackendPtr>{device}; }, states, 0, 4, kDraftMax + 1, drafter)(1024, 0);
        auto p = proposer(*model);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 0, p.get(), kDraftMax, false);
        std::shared_ptr<server::Request> hv = sched.submit(v.prompt, params_of(v));
        size_t submits = 0;
        device->hook = [&submits, &hv, at] { if (++submits == at) hv->cancel(); };
        std::thread runner([&] { sched.run(); });
        try {
            collect(*hv);
            require(hv->finish() == "cancel", case_what + ": it ended with " + hv->finish());
            device->hook = nullptr;
            same(want, drain(*sched.submit(after.prompt, params_of(after))), case_what + ": the request after it");
            ledger(sched.stats(), *model, case_what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        whole_pool_free(*model, 1024, vocab, case_what);
    }
}

// Greedy and sampled, capped and uncapped, and one ended by a stop string its reply alone holds.
std::vector<Req> requests(const Make& plain, const bpe::Tokenizer& tok, uint32_t vocab) {
    std::vector<Req> reqs = {{prompt_of(1, 40, vocab), 160}, {prompt_of(2, 9, vocab), 120, "", 0.8f, 7}, {prompt_of(3, 23, vocab), 0}};
    Req stopped{prompt_of(4, 30, vocab), 140};
    auto model = plain(1024, 0);
    const Reply reply = serve(*model, tok, 4, {{stopped}}).back();
    std::vector<uint32_t> ids;
    for (const auto& t : reply) ids.push_back(t.id);
    stopped.stop = stop_at(tok, ids, 90);
    reqs.push_back(stopped);
    return reqs;
}

}   // namespace

int main() {
    try {
        const gguf::GGUFModel q8 = served(kCpu);
        const bpe::Tokenizer q8_tok(q8);
        const uint32_t q8_vocab = (uint32_t)kCpu.vocab;
        const MakeProposer lookup = [](infer::Model&) { return std::make_unique<infer::spec::Lookup>(); };
        const MakeProposer embedded = [](infer::Model& m) { return std::make_unique<infer::spec::Embedded>(m); };
        for (size_t stages : {size_t(1), size_t(2)}) {
            const std::string where = stages == 1 ? "one CPU" : "a two-CPU split";
            const Make plain = on(q8, [stages] { return cpus(stages); });
            const Make drafting = on(q8, [stages] { return wide(stages); }, 8, 0, 4, kDraftMax + 1);
            against_alone(plain, drafting, lookup, q8_tok, 4096, 4, stages, requests(plain, q8_tok, q8_vocab), "Q8_0 with lookup on " + where);
            // Three uncapped requests on 8 blocks: they pause and resume, with drafts as without.
            against_alone(plain, drafting, lookup, q8_tok, 1024, 3, stages, {{prompt_of(1, 40, q8_vocab)}, {prompt_of(2, 9, q8_vocab)}, {prompt_of(3, 23, q8_vocab)}},
                          "Q8_0 with lookup on " + where + ", paused", true);
        }
        // Three requests, each a lone decoder in a pass of its own over three CPU stages at three passes, each verifying the 63 drafts lookup finds in a prompt that ends as it began: the draft rows in flight stay within the 64 the logits rows hold, and every reply is its reply alone.
        {
            const Make plain = on(q8, [] { return cpus(3); });
            const Make deep = on(q8, [] { return wide(3); }, 8, 0, 4, infer::spec::kMaxDrafts + 1);
            std::vector<Req> reqs;
            for (uint32_t r = 0; r < 3; ++r) {
                std::vector<uint32_t> prompt = prompt_of(10 + r, 80, q8_vocab);
                prompt.insert(prompt.end(), {prompt[0], prompt[1], prompt[2]});
                reqs.push_back({prompt, 100});
            }
            against_alone(plain, deep, lookup, q8_tok, 4096, 3, 3, reqs, "Q8_0 with lookup at 63 drafts over three CPUs at three passes", false, false,
                          infer::spec::kMaxDrafts);
        }
        const gguf::GGUFModel hybrid = served_hybrid(kHybrid, true);
        const bpe::Tokenizer hybrid_tok(hybrid);
        const uint32_t hybrid_vocab = (uint32_t)kHybrid.vocab;
        for (size_t stages : {size_t(1), size_t(2)}) {
            const std::string where = stages == 1 ? "one CPU" : "a two-CPU split";
            const Make plain = on(hybrid, [stages] { return cpus(stages); }, 4);
            const Make with_drafter = on(hybrid, [stages] { return wide(stages); }, 4, 0, 4, kDraftMax + 1, true);
            const Make with_marks = on(hybrid, [stages] { return wide(stages); }, 4, 0, 4, kDraftMax + 1);
            const std::vector<Req> reqs = requests(plain, hybrid_tok, hybrid_vocab);
            against_alone(plain, with_drafter, embedded, hybrid_tok, 4096, 4, stages, reqs, "the hybrid model with its embedded drafter on " + where);
            against_alone(plain, with_marks, lookup, hybrid_tok, 4096, 4, stages, reqs, "the hybrid model with lookup on " + where);
            // Priced by the passes' timing, which decides how much each pass drafts, every reply is still its reply alone.
            against_alone(plain, with_drafter, embedded, hybrid_tok, 4096, 4, stages, reqs, "the hybrid model with its embedded drafter on " + where + ", priced", false, true);
        }
        cancelled_in_flight(q8, q8_tok, q8_vocab, lookup, false, 8, "Q8_0 with lookup");
        cancelled_in_flight(hybrid, hybrid_tok, hybrid_vocab, embedded, true, 4, "the hybrid model with its embedded drafter");
        std::cout << "server-spec: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-spec: FAIL: " << e.what() << "\n";
        return 1;
    }
}
