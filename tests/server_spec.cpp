// Drafting in the scheduler (docs/SPECULATIVE.md, section 3): requests that verify drafts beside each other give, token for token, the ids and log-probabilities they give alone without drafts.
// The models, placements and request kinds it runs over are listed under `server-spec` in AGENTS.md, Tests.
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

// A request cancelled while its pass is in flight, at each of several submissions so some cancels find a verify, ends cancelled with its history back at its last pick.
// A request after it gives its reply on a fresh model without drafts, and every block comes back.
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

// A request resumed by a fork of its whole history drafts nothing until a pass has fed it, and nobody ends with an error (docs/SPECULATIVE.md, section 3).
// Two uncapped requests on a pool of 4096 tokens with a host tier take turns, each resume a fork of a copy promoted from host memory; both replies are the replies alone, nothing recomputed.
void resumed_by_fork(const Make& plain, const Make& drafting, const MakeProposer& proposer, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what) {
    const Req a{prompt_of(1, 2306, vocab)}, b{prompt_of(2, 641, vocab)};
    std::vector<Reply> alone;
    for (const Req& r : {a, b}) {
        auto model = plain(4096, 0);
        alone.push_back(serve(*model, tok, 4, {{r}}).back());
    }
    auto model = drafting(4096, 0);
    auto p = proposer(*model);
    server::Scheduler sched(*model, tok, 4, 64, 0, false, size_t(1) << 28, p.get(), kDraftMax, false);
    std::thread runner([&] { sched.run(); });
    Reply got_a, got_b;
    std::string end_a, end_b;
    server::Scheduler::Stats stats;
    try {
        const auto ha = sched.submit(a.prompt, params_of(a));
        server::Request::Token t;
        while (got_a.size() < 50) {
            require(ha->next(t, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, what + ": the first request ended or gave nothing before its 50th token");
            got_a.push_back(t);
        }
        const auto hb = sched.submit(b.prompt, params_of(b));
        for (const Reply& more : {collect(*ha)}) got_a.insert(got_a.end(), more.begin(), more.end());
        got_b = collect(*hb);
        end_a = ha->finish() + " " + ha->error();
        end_b = hb->finish() + " " + hb->error();
        stats = sched.stats();
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    require(end_a == "length " && end_b == "length ", what + ": the requests ended with \"" + end_a + "\" and \"" + end_b + "\", against the pool's end for both");
    same(alone[0], got_a, what + ", the request that ran to the pool's end");
    same(alone[1], got_b, what + ", the request paused beside it");
    require(stats.pauses >= 1 && stats.host_hits >= 1 && stats.recomputed == 0,
            what + ": " + std::to_string(stats.pauses) + " pauses, " + std::to_string(stats.host_hits) + " promotions from host memory and " + std::to_string(stats.recomputed) + " tokens recomputed, against one at least of each and none");
    std::cout << "server-spec: " << what << ": " << total(stats.kept) << " of " << total(stats.drafted) << " drafts kept" << std::endl;
}

// A request drafts while another waits for room, and not while one waits for a seat (docs/SPECULATIVE.md, section 3).
// Queued for room, the drafts fed are those the two feed each alone; queued for the one seat, the first feeds none; every reply is the request's reply alone.
void drafts_while_one_waits(const Make& plain, const Make& drafting, const MakeProposer& proposer, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what) {
    std::vector<Req> reqs;
    for (uint32_t r = 0; r < 2; ++r) {
        std::vector<uint32_t> prompt = prompt_of(20 + r, 80, vocab);
        prompt.insert(prompt.end(), {prompt[0], prompt[1], prompt[2]});
        reqs.push_back({prompt, 60});
    }
    reqs.push_back({prompt_of(22, 900, vocab), 20});
    std::vector<Reply> alone;
    std::vector<size_t> fed;
    for (const Req& r : reqs) {
        auto model = plain(1024, 0);
        alone.push_back(serve(*model, tok, 2, {{r}}).back());
        auto with = drafting(1024, 0);
        auto p = proposer(*with);
        server::Scheduler::Stats stats;
        same(alone.back(), serve(*with, tok, 2, {{r}}, &stats, 0, 0, p.get(), kDraftMax, false).back(), what + ", a request alone with drafts");
        fed.push_back(total(stats.drafted));
    }
    require(fed[0] > 0 && fed[1] > 0, what + ": lookup found no draft for the requests alone");
    const auto together = [&](size_t seats, const std::vector<size_t>& which, size_t want, const std::string& name) {
        auto model = drafting(1024, 0);
        auto p = proposer(*model);
        server::Scheduler::Stats stats;
        std::vector<Req> wave;
        for (size_t i : which) wave.push_back(reqs[i]);
        const std::vector<Reply> got = serve(*model, tok, seats, {wave}, &stats, 0, 0, p.get(), kDraftMax, false);
        for (size_t i = 0; i < which.size(); ++i) same(alone[which[i]], got[i], what + ", " + name + ", request " + std::to_string(i));
        require(total(stats.drafted) == want, what + ", " + name + ": " + std::to_string(total(stats.drafted)) + " drafts fed, against " + std::to_string(want));
    };
    together(2, {0, 2}, fed[0] + fed[2], "a request queued for room");
    together(1, {0, 1}, fed[1], "a request queued for the one seat");
    std::cout << "server-spec: " << what << ": " << fed[0] << " drafts fed beside a request waiting for room, none beside one waiting for the seat" << std::endl;
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
        // Three requests, each a lone decoder in a pass of its own over three CPU stages at three passes, each verify the 63 drafts lookup finds in a prompt that ends as it began.
        // The draft rows in flight stay within the 64 the logits rows hold, and every reply is its reply alone.
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
        resumed_by_fork(on(hybrid, [] { return cpus(1); }, 4, 3), on(hybrid, [] { return wide(1); }, 4, 3, 4, kDraftMax + 1, true), embedded, hybrid_tok, hybrid_vocab,
                        "a request resumed by a fork of its whole history, the hybrid model with its embedded drafter");
        drafts_while_one_waits(on(q8, [] { return cpus(1); }), on(q8, [] { return wide(1); }, 8, 0, 4, kDraftMax + 1), lookup, q8_tok, q8_vocab, "drafts while a request waits, Q8_0 with lookup");
        cancelled_in_flight(q8, q8_tok, q8_vocab, lookup, false, 8, "Q8_0 with lookup");
        cancelled_in_flight(hybrid, hybrid_tok, hybrid_vocab, embedded, true, 4, "the hybrid model with its embedded drafter");
        std::cout << "server-spec: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-spec: FAIL: " << e.what() << "\n";
        return 1;
    }
}
