// Drafting in the scheduler (docs/SPECULATIVE.md, section 3): requests that verify drafts beside each other give, token for token, the ids and log-probabilities they give alone without drafts.
// The models, placements and request kinds it runs over are listed under `server-spec` in AGENTS.md, Tests.
#include <iostream>
#include <cmath>
#include <cstring>
#include <atomic>
#include <numeric>

#include "server_harness.hpp"
#include "inference/load.hpp"
#include "qwen4exp_mtp_fixture.hpp"

namespace {

constexpr size_t kDraftMax = 3;

// A CPU backend whose decode kernels hold 16 columns, so a pass of several decoding requests has columns to spare for their drafts.
struct Wide : backend::CpuBackend {
    Wide() { set_threads(1); }
    size_t decode_columns() const override { return 16; }
};

struct FlashRerunFault {
    std::atomic<bool> armed{false}, threw{false}, retrying{false};
    std::atomic<size_t> marked_ple_replays{0}, marked_gdn_completions{0}, retry_ple_replays{0};
};

// Fails GDN only after the marked rerun has replayed PLE's stateful convolution.
struct FlashRerunFailingWide : Wide {
    std::shared_ptr<FlashRerunFault> fault;
    bool fail_member = false, ple_after_mark = false;
    bool unordered_on = false;
    FlashRerunFailingWide(std::shared_ptr<FlashRerunFault> f, bool member) : fault(std::move(f)), fail_member(member) {}
    void unordered(bool on) override { unordered_on = on; }
    void dilated_conv_silu_add(backend::Slice out, backend::CSlice x, backend::CSlice residual, backend::CSlice w,
                               size_t channels, size_t taps, size_t dilation, size_t layer,
                               const backend::StateView* views, size_t n_views) override {
        backend::CpuBackend::dilated_conv_silu_add(out, x, residual, w, channels, taps, dilation, layer, views, n_views);
        if (fault->armed.load()) { ple_after_mark = true; ++fault->marked_ple_replays; }
        if (fault->retrying.load()) ++fault->retry_ple_replays;
    }
    void gated_delta_rule(backend::Slice out, backend::CSlice qkv, backend::CSlice alpha, backend::CSlice b,
                          backend::CSlice a, backend::CSlice dt_bias, size_t layer,
                          const backend::StateView* views, size_t n_views) override {
        if (fault->armed.load() && fail_member && ple_after_mark) {
            fault->armed.store(false);
            fault->threw.store(true);
            throw std::runtime_error("injected Flash GDN rerun failure");
        }
        backend::CpuBackend::gated_delta_rule(out, qkv, alpha, b, a, dt_bias, layer, views, n_views);
        if (fault->armed.load()) ++fault->marked_gdn_completions;
    }
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
                   size_t draft_max = kDraftMax, server::Scheduler::Stats* counted = nullptr) {
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
    if (!pauses) {
        uint64_t prompt = 0;
        for (const Req& r : reqs) prompt += r.prompt.size();
        require(stats.prefilled_tokens == prompt - stats.prefix_tokens, what + ": verify or drafter rows counted as prompt work");
    }
    require(priced || (total(stats.drafted) > 0 && total(stats.kept) > 0),
            what + ": " + std::to_string(total(stats.drafted)) + " drafts fed and " + std::to_string(total(stats.kept)) + " kept");
    if (pauses) require(stats.pauses >= 1, what + ": nothing paused");
    if (counted) *counted = stats;
    std::cout << "server-spec: " << what << ": " << total(stats.kept) << " of " << total(stats.drafted) << " drafts kept, " << stats.pauses << " pauses" << std::endl;
}

// A timed scheduler (serve --timing) counts the thread's time in the drafters' chains and in the verifies' retracts apart from forming passes, and the reply is its reply without drafts.
void timed_phases(const Make& plain, const Make& drafting, const MakeProposer& proposer, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what) {
    const Req r{prompt_of(1, 40, vocab), 160};
    auto fresh = plain(1024, 0);
    const Reply want = serve(*fresh, tok, 3, {{r}}).back();
    auto model = drafting(1024, 0);
    auto p = proposer(*model);
    server::Scheduler sched(*model, tok, 3, 64, 0, true, 0, p.get(), kDraftMax, false);
    std::thread runner([&] { sched.run(); });
    server::Scheduler::Stats stats;
    try {
        same(want, drain(*sched.submit(r.prompt, params_of(r))), what);
        stats = sched.stats();
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    const server::Scheduler::Timing& t = stats.timing;
    require(stats.timed && total(stats.drafted) > 0, what + ": " + std::to_string(total(stats.drafted)) + " drafts fed under a timed scheduler");
    require(t.drafting_ms > 0 && t.settling_ms > 0 && t.assembly_ms >= 0,
            what + ": chains " + std::to_string(t.drafting_ms) + " ms, retracts " + std::to_string(t.settling_ms) + " ms, forming " + std::to_string(t.assembly_ms) + " ms");
    std::cout << "server-spec: " << what << ": chains " << t.drafting_ms << " ms, retracts " << t.settling_ms << " ms over " << t.rounds << " rounds" << std::endl;
}

// A proposer that stands behind lookup as a model drafter does and drafts the token after each history's last, which a reply seldom holds next.
struct Astray final : infer::spec::Proposer {
    uint32_t vocab = 0;
    void draft(infer::Sequence*, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override { out.assign(k, (h.back() + 1) % vocab); }
    bool after_lookup() const override { return true; }
};

// Lookup ahead of a proposer in the scheduler (docs/SPECULATIVE.md, section 8): greedy replies that come to repeat themselves draft their own continuation, the proposer drafting for the others in the same passes, each reply its own alone.
void own_history(const Make& plain, const Make& drafting, const bpe::Tokenizer& tok, uint32_t vocab, size_t stages, const std::string& what) {
    const MakeProposer astray = [vocab](infer::Model&) {
        auto p = std::make_unique<Astray>();
        p->vocab = vocab;
        return p;
    };
    std::vector<Req> reqs;
    for (uint32_t seed : {3u, 4u, 5u}) reqs.push_back({prompt_of(seed, 40, vocab), 300});
    reqs.push_back({prompt_of(6, 40, vocab), 300, "", 0.8f, 7});
    server::Scheduler::Stats stats;
    against_alone(plain, drafting, astray, tok, 4096, 4, stages, reqs, what, false, false, kDraftMax, &stats);
    const size_t own = total(stats.own_drafted), kept = total(stats.own_kept);
    require(kept * 2 > own && kept >= 100, what + ": " + std::to_string(kept) + " of " + std::to_string(own) + " drafts of the replies' own continuations kept");
    require(total(stats.drafted) > own && total(stats.kept) >= kept, what + ": the proposer drafted nothing beside them, " + std::to_string(total(stats.drafted)) + " drafts fed in all");
    std::cout << "server-spec: " << what << ": " << kept << " of " << own << " drafts of the replies' own continuations kept, " << total(stats.drafted) - own << " of the proposer's fed" << std::endl;
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
// Two uncapped requests on 4096 tokens with a host tier take turns, each resume a fork of a copy promoted from host memory, the second submitted from the scheduler's thread at the first's 50th sampled pass; nothing is recomputed.
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
    std::mutex m;
    std::shared_ptr<server::Request> ha, hb;
    size_t sampled = 0;
    sched.on_retire = [&](const server::Scheduler::Retired& t) {
        std::lock_guard<std::mutex> lock(m);
        for (size_t i = 0; i < t.requests.size(); ++i)
            if (t.requests[i] == ha.get() && t.want[i] && ++sampled == 50) hb = sched.submit(b.prompt, params_of(b));
    };
    {
        std::lock_guard<std::mutex> lock(m);
        ha = sched.submit(a.prompt, params_of(a));
    }
    std::thread runner([&] { sched.run(); });
    Reply got_a, got_b;
    std::string end_a, end_b;
    server::Scheduler::Stats stats;
    try {
        got_a = collect(*ha);
        {
            std::lock_guard<std::mutex> lock(m);
            require(hb != nullptr, what + ": the first request ended before its 50th sampled pass");
        }
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

// Delegates every draft unchanged and witnesses that simultaneous requests reach the model's batched drafter.
struct FlashDrafts final : infer::spec::Proposer {
    infer::spec::Embedded embedded;
    bool batched = false;
    explicit FlashDrafts(infer::Model& model) : embedded(model) {}
    void draft(infer::Sequence* seq, const std::vector<uint32_t>& history, size_t k, std::vector<uint32_t>& out) override {
        embedded.draft(seq, history, k, out);
    }
    void draft_all(Ask* asks, size_t n) override {
        batched = batched || n > 1;
        embedded.draft_all(asks, n);
    }
};

// Public sidecar loading and real scheduler verifies share the native Flash fixture, without an HTTP transport claim.
void flash_sidecar(bool split, int compression, int layout, const std::vector<std::string>& devices, size_t passes = 0) {
    const bool grouped = devices.size() > 1;
    flash_mtp::Fixture fixture(split, compression, grouped);
    fixture.c.context_length = 512;
    // Positive random A makes the original short fixture overflow after 73 rows; long server histories need genuine decay.
    auto& decay = fixture.data.at("blk.0.ssm_a");
    std::fill(decay.begin(), decay.end(), -0.5f);
    auto [target, side] = flash_mtp::headers(fixture, layout);
    std::map<std::string, std::vector<float>> private_data;
    if (layout == 2) private_data = flash_mtp::private_globals(fixture, side);
    flash_mtp::Files files(fixture, std::move(target), std::move(side), private_data);
    std::vector<backend::BackendPtr> selected;
    if (std::any_of(devices.begin(), devices.end(), [](const std::string& name) { return name != "cpu"; }))
        for (const auto& name : devices) selected.push_back(name == "cpu" ? std::make_shared<Wide>() : backend::make_backend(name));
    const auto load = [&](bool paired) {
        infer::ModelOptions options;
        options.kv_tokens = 1024; options.state_slots = 3; options.checkpoint_slots = 3;
        options.mark_slots = 3; options.mark_rows = kDraftMax + 1;
        infer::PlacementRequest request;
        request.dtype = backend::Dtype::f32;
        request.width = grouped ? 2 : 1;
        if (grouped) {
            request.names = devices;
            request.shares.assign(devices.size() / 2, 1);
        }
        auto loaded = infer::load_model(files.target_path, selected.empty() ? wide(devices.size()) : selected, request, options, {},
                                        infer::LoadMode::mapped, paired ? files.side_path : "");
        if (grouped) {
            require(loaded->model->stage_count() == devices.size() / 2, "Flash public loader stage count");
            for (size_t stage = 0; stage < loaded->model->stage_count(); ++stage)
                require(loaded->model->stage_grouped(stage), "Flash public loader tensor-group stage");
        }
        return loaded;
    };
    std::string where;
    for (const auto& name : devices) where += (where.empty() ? "" : ",") + name;
    const std::string what = "Flash " + std::string(split ? "split QSA" : "fused dense") + (layout == 1 ? " shared" : " private") +
                             " on " + where + " width=" + std::to_string(grouped ? 2 : 1) + " stages=" + std::to_string(grouped ? devices.size() / 2 : 1);
    const std::vector<Req> reqs{{prompt_of(1, 151, flash_mtp::V), 27}, {prompt_of(3, 137, flash_mtp::V), 19, "", 0.8f, 7}};
    std::vector<Reply> wants;
    for (const auto& req : reqs) {
        auto plain = load(false);
        wants.push_back(serve(*plain->model, *plain->tok, 2, {{req}}, nullptr, passes).back());
        for (const auto& token : wants.back()) {
            require(std::isfinite(token.logprob), what + ": nonfinite target logprob");
            for (const auto& top : token.top) require(std::isfinite(top.logprob), what + ": nonfinite top logprob");
        }
        auto off = load(true);
        require(off->drafter == infer::spec::DrafterKind::mtp, what + ": public loader selected MTP");
        same(wants.back(), serve(*off->model, *off->tok, 2, {{req}}, nullptr, passes).back(), what + ": loaded/off");
    }
    Req follow{reqs[0].prompt, 17}, edited{reqs[0].prompt, 13};
    for (const auto& token : wants[0]) follow.prompt.push_back(token.id);
    follow.prompt.insert(follow.prompt.end(), {2, 4, 1});
    for (size_t i = 135; i < edited.prompt.size(); ++i) edited.prompt[i] = (edited.prompt[i] + 3) % flash_mtp::V;
    for (const auto& req : {follow, edited}) {
        auto fresh = load(false);
        wants.push_back(serve(*fresh->model, *fresh->tok, 2, {{req}}, nullptr, passes).back());
    }
    auto loaded = load(true);
    FlashDrafts proposer(*loaded->model);
    server::Scheduler::Stats stats;
    const auto got = serve(*loaded->model, *loaded->tok, 2, {reqs, {follow}, {edited}}, &stats, passes, 0, &proposer, kDraftMax);
    for (size_t i = 0; i < wants.size(); ++i) same(wants[i], got[i], what + ": verified reply " + std::to_string(i));
    require(total(stats.drafted) > total(stats.kept) && total(stats.kept) > 0,
            what + ": need fed, kept and rejected drafts, got " + std::to_string(total(stats.kept)) + "/" + std::to_string(total(stats.drafted)));
    require(stats.prefix_hits == 2 && stats.prefix_tokens == 256, what + ": follow-up/edit did not reuse the exact checkpoints");
    if (devices.size() != 4 || passes == 1) require(proposer.batched, what + ": unequal requests never drafted together");
    if (grouped) require(stats.passes == (passes ? passes : devices.size() / 2), what + ": scheduler did not retain the requested pass count");
    require(wants[0][0].logprob != wants[3][0].logprob, what + ": edited suffix did not change the target row");
    require(stats.draft_failures == 0, what + ": drafter failed");
    std::cout << "server-spec: " << what << ": " << total(stats.kept) << " of " << total(stats.drafted)
              << " drafts kept, " << stats.prefix_tokens << " prefix tokens reused, passes=" << stats.passes << " batched=" << proposer.batched << '\n';
}

void flash_sidecars(const std::vector<std::string>& devices) {
    const size_t before = checks;
    flash_sidecar(false, 0, 1, devices);
    flash_sidecar(false, 0, 2, devices);
    flash_sidecar(true, 4, 2, devices);
    if (devices.size() == 4) flash_sidecar(true, 4, 2, devices, 1);
    std::cout << "server-spec: Flash scheduler/public-loader " << devices.size() << " backends: " << checks - before << " checks pass\n";
}

// A marked MTP verify whose accepted-state replay fails after PLE has mutated its tail, then retries the same mark.
void flash_marked_rerun_failure() {
    flash_mtp::Fixture fixture(true, 4, true, true, 3);
    auto [target, side] = flash_mtp::headers(fixture, 1);
    flash_mtp::Files files(fixture, std::move(target), std::move(side));
    struct Loaded {
        std::unique_ptr<infer::LoadedModel> model;
        std::vector<std::shared_ptr<FlashRerunFailingWide>> injected;
    };
    const auto load = [&](const std::shared_ptr<FlashRerunFault>& fault) {
        std::vector<backend::BackendPtr> backends;
        std::vector<std::shared_ptr<FlashRerunFailingWide>> injected;
        for (size_t i = 0; i < 6; ++i) {
            if (fault) {
                auto b = std::make_shared<FlashRerunFailingWide>(fault, i % 2 == 1);
                injected.push_back(b);
                backends.push_back(std::move(b));
            } else {
                backends.push_back(std::make_shared<Wide>());
            }
        }
        infer::PlacementRequest request;
        request.width = 2;
        request.drafter = true;
        request.shares.assign(3, 1);
        for (size_t i = 0; i < backends.size(); ++i) request.names.push_back("device " + std::to_string(i));
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        options.kv_tokens = 1024;
        options.state_slots = 1;
        options.mark_slots = 1;
        options.mark_rows = kDraftMax + 1;
        auto loaded = infer::load_model(files.target_path, std::move(backends), request, options, {}, infer::LoadMode::mapped, files.side_path);
        require(loaded->drafter == infer::spec::DrafterKind::mtp, "Flash rerun fixture did not load its MTP sidecar");
        require(loaded->model->stage_count() == 3, "Flash rerun fixture did not make three stages");
        for (size_t s = 0; s < loaded->model->stage_count(); ++s)
            require(loaded->model->stage_grouped(s), "Flash rerun fixture has an ungrouped stage");
        return Loaded{std::move(loaded), std::move(injected)};
    };
    const auto exercise = [&](bool check_draft) {
        auto fault = std::make_shared<FlashRerunFault>();
        auto clean = load({});
        auto broken = load(fault);
        const std::vector<uint32_t> prompt = prompt_of(6, 9, flash_mtp::V);
        const size_t mark_at = prompt.size();
        clean.model->model->prefill(prompt);
        broken.model->model->prefill(prompt);
        require(clean.model->model->mark() && broken.model->model->mark(), "Flash rerun mark was refused");
        const uint32_t verify[] = {1, 4, 2};
        const float* expected_rows = clean.model->model->step(verify, 3);
        const float* actual_rows = broken.model->model->step(verify, 3);
        require(std::all_of(expected_rows, expected_rows + 3 * flash_mtp::V, [](float x) { return std::isfinite(x); }) &&
                    std::memcmp(expected_rows, actual_rows, 3 * flash_mtp::V * sizeof(float)) == 0,
                "Flash rerun pre-failure verify logits differ from the clean control");
        fault->armed.store(true);
        bool threw = false;
        try { (void)broken.model->model->retract(mark_at + 2); }
        catch (const std::runtime_error& e) { threw = std::string(e.what()) == "injected Flash GDN rerun failure"; }
        require(threw && fault->threw.load() && fault->marked_ple_replays.load() >= 2 && fault->marked_gdn_completions.load() == 1,
                "Flash rerun fault did not follow both PLE replays and exactly one member's GDN completion");
        for (const auto& b : broken.injected)
            require(!b->unordered_on, "a failed Flash rerun left a backend recording unordered");
        fault->retrying.store(true);
        require(broken.model->model->retract(mark_at + 2) == mark_at + 2,
                "Flash rerun retry of the same mark did not reach the kept length");
        fault->retrying.store(false);
        require(fault->retry_ple_replays.load() > 0, "Flash retry did not replay the saved PLE inputs");
        require(clean.model->model->retract(mark_at + 2) == mark_at + 2, "Flash clean control did not reach the kept length");
        require(clean.model->model->n_tokens() == broken.model->model->n_tokens(), "Flash retry history length differs from the clean control");
        if (check_draft) {
            std::vector<uint32_t> want, got;
            clean.model->model->draft(verify[1], 2, want);
            broken.model->model->draft(verify[1], 2, got);
            require(want.size() == 2 && got.size() == 2 && want == got,
                    "Flash MTP drafts after failed-rerun retry differ from the clean control");
            for (size_t i = 0; i < got.size(); ++i) {
                const float* expected = clean.model->model->draft_logits(i);
                const float* actual = broken.model->model->draft_logits(i);
                require(std::all_of(expected, expected + flash_mtp::V, [](float x) { return std::isfinite(x); }) &&
                            std::memcmp(expected, actual, flash_mtp::V * sizeof(float)) == 0,
                        "Flash MTP draft logits after failed-rerun retry differ from the clean control");
            }
        } else {
            const uint32_t continuation[] = {5, 3, 7, 1, 6, 2, 4, 0, 3, 5, 1, 7};
            for (uint32_t id : continuation) {
                const std::vector<float> want = clean.model->model->step(int(id));
                const std::vector<float> got = broken.model->model->step(int(id));
                require(std::all_of(want.begin(), want.end(), [](float x) { return std::isfinite(x); }) &&
                            std::memcmp(want.data(), got.data(), flash_mtp::V * sizeof(float)) == 0,
                        "Flash target logits after failed-rerun retry differ from the clean control");
            }
        }
    };
    exercise(false);
    exercise(true);
    std::cout << "server-spec: Flash MTP marked rerun failure after both PLE replays and one GDN member, then retry; width 2 over three CPU stages: pass\n";
}

void write_flash_http(const std::filesystem::path& directory) {
    require(std::filesystem::is_directory(directory) && std::filesystem::is_empty(directory), "Flash HTTP fixture needs an empty directory");
    flash_mtp::Fixture fixture(false, 0);
    fixture.c.context_length = 512;
    auto& decay = fixture.data.at("blk.0.ssm_a");
    std::fill(decay.begin(), decay.end(), -0.5f);
    auto [target, side] = flash_mtp::headers(fixture, 1);
    const auto format = flash_mtp::string("{% for m in messages %}{{ m.content }}{% endfor %}{% if add_generation_prompt %}a{% endif %}");
    target.kv.push_back({"tokenizer.chat_template", format});
    side.kv.push_back({"tokenizer.chat_template", format});
    flash_mtp::Files files(fixture, std::move(target), std::move(side));
    std::filesystem::copy_file(std::filesystem::u8path(files.target_path), directory / "model.gguf");
    std::filesystem::copy_file(std::filesystem::u8path(files.side_path), directory / "mtp.gguf");
}

}   // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write-flash-http") {
            write_flash_http(std::filesystem::u8path(argv[2]));
            return 0;
        }
        const char* usage = "usage: llmx-server-spec-test [--flash-only [--device cpu|vulkan:N] [--group-devices D0,D1[,D2,D3]]] | --write-flash-http EMPTY_DIRECTORY";
        bool flash_only = false, device_given = false, groups_given = false;
        std::string device = "cpu";
        std::vector<std::string> group_devices;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--flash-only") flash_only = true;
            else if (flag == "--device" && i + 1 < argc) { device = backend::canonical_device(argv[++i]); device_given = true; }
            else if (flag == "--group-devices" && i + 1 < argc) { group_devices = core::comma_list(argv[++i]); groups_given = true; }
            else throw std::runtime_error(usage);
        }
        if ((device_given || groups_given) && !flash_only) throw std::runtime_error("device options require --flash-only");
        if (groups_given) {
            if (group_devices.size() != 2 && group_devices.size() != 4) throw std::runtime_error(usage);
            for (size_t i = 0; i < group_devices.size(); ++i) {
                group_devices[i] = backend::canonical_device(group_devices[i]);
                for (size_t j = 0; j < i; ++j)
                    if (group_devices[i] != "cpu" && group_devices[i] == group_devices[j])
                        throw std::runtime_error("--group-devices requires distinct device names except cpu");
            }
        }
        flash_sidecars({device});
        if (device == "cpu") flash_marked_rerun_failure();
        if (groups_given) flash_sidecars(group_devices);
        else {
            flash_sidecars({"cpu", "cpu"});
            flash_sidecars({"cpu", "cpu", "cpu", "cpu"});
        }
        if (flash_only) {
            std::cout << "server-spec: " << checks << " Flash checks pass\n";
            return 0;
        }
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
            own_history(plain, drafting, q8_tok, q8_vocab, stages, "Q8_0 with lookup ahead of a proposer on " + where);
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
        timed_phases(on(hybrid, [] { return cpus(1); }, 4), on(hybrid, [] { return wide(1); }, 4, 0, 4, kDraftMax + 1, true), embedded, hybrid_tok, hybrid_vocab,
                     "a timed scheduler, the hybrid model with its embedded drafter");
        cancelled_in_flight(q8, q8_tok, q8_vocab, lookup, false, 8, "Q8_0 with lookup");
        cancelled_in_flight(hybrid, hybrid_tok, hybrid_vocab, embedded, true, 4, "the hybrid model with its embedded drafter");
        std::cout << "server-spec: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-spec: FAIL: " << e.what() << "\n";
        return 1;
    }
}
