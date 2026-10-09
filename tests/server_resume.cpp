// Requests the scheduler pauses and resumes give, token for token, the ids and log-probabilities they give alone, over the synthetic Q8_0 model, and room goes by first admission (docs/SERVER.md).
// Usage: llmx-server-resume-test [cpu|device], both by default, the device cases on Vulkan device 0 when it opens; the cases are listed in AGENTS.md, Tests.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>

#include "server/disk_index.hpp"
#include "server_harness.hpp"

namespace {

// Each request alone on a fresh model, after the `setup` requests, which leave the donors a case needs; then the setup again and every request at once on another fresh model.
// Each reply must equal its reply alone, and the run together must pause.
server::Scheduler::Stats alone_then_together(const Make& make, const bpe::Tokenizer& tok, size_t pool, int ubatch, size_t max_seqs,
                                             const std::vector<Req>& setup, const std::vector<Req>& reqs, const std::string& what) {
    std::vector<std::vector<Req>> waves;
    for (const Req& s : setup) waves.push_back({s});
    std::vector<Reply> alone;
    for (const Req& r : reqs) {
        auto model = make(pool, ubatch);
        waves.push_back({r});
        alone.push_back(serve(*model, tok, max_seqs, waves).back());
        waves.pop_back();
    }
    auto model = make(pool, ubatch);
    waves.push_back(reqs);
    server::Scheduler::Stats stats;
    const std::vector<Reply> together = serve(*model, tok, max_seqs, waves, &stats);
    require(stats.pauses >= 1, what + ": nothing paused");
    for (size_t i = 0; i < reqs.size(); ++i)
        same(alone[i], together[setup.size() + i], what + ", request " + std::to_string(i));
    return stats;
}

// Three uncapped requests whose prompts differ in their first token, so none forks another, on 8 blocks of 128 tokens.
// Two fit at first; the first's growth pauses the second with generated tokens in its partial block and its donor taken, so it recomputes its whole history, and the third then runs beside the first and is paused in turn.
void three_uncapped(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what) {
    const std::vector<Req> reqs = {{prompt_of(1, 40, vocab)}, {prompt_of(2, 9, vocab)}, {prompt_of(3, 23, vocab)}};
    const server::Scheduler::Stats stats = alone_then_together(make, tok, 1024, 0, 3, {}, reqs, what);
    // A resume recomputed more rows than every pause's longest prompt, so generated tokens among them.
    require(stats.recomputed > stats.pauses * 40, what + ": no resume recomputed a generated token");
}

// A prompt read one token a pass is still prefilling when the request beside it has to grow and the pool is short, so it is paused part-way through its prompt, its donor then going to the one that grew.
void prefilling_victim(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::vector<Req> reqs = {{prompt_of(1, 4, vocab)}, {prompt_of(2, 384, vocab)}};
    const auto s = alone_then_together(make, tok, 1024, 1, 3, {}, reqs, "a victim still prefilling");
    require(s.taken_back == 0 && s.recomputed >= 2 * kBlock, "a victim still prefilling: " + std::to_string(s.taken_back) + " taken back and " +
            std::to_string(s.recomputed) + " rows recomputed, against 0 and at least the two blocks of its prompt it had read");
}

// A follow-up turn repeating a 20-token prompt and its reply forks none of that history, whose only whole blocks hold the reply's decode rows, and computes it as its own prompt.
// Paused beside an uncapped request with a short prompt, it must recompute its history as it first computed it.
void follow_up(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req first{prompt_of(5, 20, vocab), 200};
    auto model = make(1024, 0);
    const Reply turn = serve(*model, tok, 3, {{first}})[0];
    std::vector<uint32_t> again = first.prompt;
    for (const auto& t : turn) again.push_back(t.id);
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    const auto s = alone_then_together(make, tok, 1024, 0, 3, {first}, {{prompt_of(1, 9, vocab)}, {again}}, "a follow-up turn paused");
    // It sat out passes at the end of its 512-token reservation and was paused there, so a resume that forked nothing recomputes those 512 rows.
    require(s.taken_back == 0 && s.recomputed >= 4 * kBlock, "a follow-up turn paused: " + std::to_string(s.taken_back) + " taken back and " +
            std::to_string(s.recomputed) + " rows recomputed, against 0 and at least 512");
}

// On a device, a 100-token prompt sharing the first block of a 600-token prompt's history, whose rows the longer prompt's tile split computed: it forks nothing, since the classes differ.
// Admitted after an uncapped request with a short prompt, it is paused when that one grows, and its donor then goes too, so it recomputes its rows at its own extent.
void device_classes(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req donor{prompt_of(9, 600, vocab), 4};
    std::vector<uint32_t> forked(donor.prompt.begin(), donor.prompt.begin() + 64);
    const std::vector<uint32_t> own = prompt_of(10, 36, vocab);
    forked.insert(forked.end(), own.begin(), own.end());
    const auto s = alone_then_together(make, tok, 1024, 0, 3, {donor}, {{prompt_of(1, 9, vocab)}, {forked}}, "on a device, a forked prefix of another extent");
    // It had grown past its first 384-token reservation when it was paused, so a resume that recomputed at least that much forked nothing and computed the forked prefix again.
    require(s.taken_back == 0 && s.recomputed >= 384, "on a device, a forked prefix of another extent: " + std::to_string(s.taken_back) + " taken back and " +
            std::to_string(s.recomputed) + " rows recomputed, against 0 and at least 384");
    three_uncapped(make, tok, vocab, "on a device, three uncapped requests");
}

// The ids of a reply.
std::vector<uint32_t> ids_of(const Reply& r) {
    std::vector<uint32_t> ids;
    for (const auto& t : r) ids.push_back(t.id);
    return ids;
}

// A first admission forks only rows computed as it would compute them (Model::row_class): the reply after `setup` must equal the same request on a fresh model with no donor, and `forked` tokens must have been reused.
void as_on_fresh(const Make& make, const bpe::Tokenizer& tok, const std::vector<Req>& setup, const Req& r, size_t forked, const std::string& what) {
    auto fresh = make(1024, 0);
    const Reply alone = serve(*fresh, tok, 3, {{r}})[0];
    std::vector<std::vector<Req>> waves;
    for (const Req& s : setup) waves.push_back({s});
    waves.push_back({r});
    auto model = make(1024, 0);
    server::Scheduler::Stats stats;
    const Reply after = serve(*model, tok, 3, waves, &stats).back();
    same(alone, after, what);
    require(stats.prefix_tokens == forked, what + ": " + std::to_string(stats.prefix_tokens) + " tokens reused, against " + std::to_string(forked));
}

// A follow-up turn repeating an `n`-token prompt and its reply: it forks the prompt's whole blocks, `forked` tokens, and computes the reply's rows, which decode computed, as rows of its own prompt.
void follow_up_as_cli(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, size_t n, size_t forked, const std::string& what) {
    const Req first{prompt_of(5, n, vocab), 100};
    auto model = make(1024, 0);
    std::vector<uint32_t> again = first.prompt;
    for (uint32_t id : ids_of(serve(*model, tok, 3, {{first}})[0])) again.push_back(id);
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    as_on_fresh(make, tok, {first}, Req{again, 64}, forked, what);
}

// On a device, a 100-token prompt sharing the first block of a 600-token prompt, whose tile split its sums another way: it forks nothing.
void other_split_as_cli(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req donor{prompt_of(9, 600, vocab), 4};
    std::vector<uint32_t> forked(donor.prompt.begin(), donor.prompt.begin() + 64);
    const std::vector<uint32_t> own = prompt_of(10, 36, vocab);
    forked.insert(forked.end(), own.begin(), own.end());
    as_on_fresh(make, tok, {donor}, Req{forked, 32}, 0, "on a device, a prompt sharing a block of another split's prompt");
    // A 500-token prompt sharing the first 448 tokens of the 600-token one takes the same split, and forks them.
    std::vector<uint32_t> longer(donor.prompt.begin(), donor.prompt.begin() + 448);
    const std::vector<uint32_t> tail = prompt_of(11, 52, vocab);
    longer.insert(longer.end(), tail.begin(), tail.end());
    as_on_fresh(make, tok, {donor}, Req{longer, 32}, 448, "on a device, a prompt sharing blocks of a prompt of its own split");
}

// `victim` given a stop string that ends it with its reply's token `at`, found from its reply alone after the `setup` requests.
Req stopping(const Make& make, const bpe::Tokenizer& tok, size_t pool, const std::vector<Req>& setup, Req victim, size_t at) {
    std::vector<std::vector<Req>> waves;
    for (const Req& s : setup) waves.push_back({s});
    waves.push_back({victim});
    auto model = make(pool, 0);
    victim.stop = stop_at(tok, ids_of(serve(*model, tok, 3, waves).back()), at);
    return victim;
}

// On 13 blocks uncapped A (40 tokens), uncapped B (60) and capped C (380 in all) leave 4 free; B takes its step of 3 blocks at 384 tokens, and A's step 20 passes later pauses B, whose unused headroom covers it, so B's donor stays.
// C ends 16 passes later and B takes its donor back whole, recomputing nothing, and its stop string ends it before either needs room again.
void take_back(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 13 * kBlock;
    const Req a{prompt_of(1, 40, vocab)}, c{prompt_of(3, 20, vocab), 360};
    const Req b = stopping(make, tok, pool, {}, Req{prompt_of(2, 60, vocab)}, 420);
    const auto s = alone_then_together(make, tok, pool, 0, 3, {}, {a, b, c}, "a paused request taking its donor back");
    require(s.pauses == 1 && s.taken_back == 1 && s.recomputed == 0,
            "a paused request whose donor stayed: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) + " taken back, " +
            std::to_string(s.recomputed) + " rows recomputed, against 1, 1 and 0");
}

// A follow-up turn, which forks nothing of the previous turn's reply rows at its first admission, paused with its donor intact as B above is (14 blocks, the previous turn's donor going to its first growth step): it takes back its donor whole.
void take_back_follow_up(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 14 * kBlock;
    const Req first{prompt_of(5, 20, vocab), 200};
    auto model = make(pool, 0);
    std::vector<uint32_t> again = first.prompt;
    for (uint32_t id : ids_of(serve(*model, tok, 3, {{first}})[0])) again.push_back(id);
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    const Req a{prompt_of(1, 40, vocab)}, c{prompt_of(3, 20, vocab), 360};
    const Req b = stopping(make, tok, pool, {first}, Req{again}, 420);
    const auto s = alone_then_together(make, tok, pool, 0, 3, {first}, {a, b, c}, "a follow-up turn taking its donor back");
    require(s.pauses == 1 && s.taken_back == 1 && s.recomputed == 0,
            "a follow-up turn whose donor stayed: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) + " taken back, " +
            std::to_string(s.recomputed) + " rows recomputed, against 1, 1 and 0");
}

// B and capped C fork the first block of the setup's 200-token prompt; on 14 blocks B cannot take its step at 512 tokens beside A and C, and A's step pauses B and takes its donor, so once C ends only C's donor holds that block.
// B forks that block, computed as its own was, and recomputes the other 384 rows in their classes.
void partial_eviction(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 14 * kBlock;
    const Req setup{prompt_of(7, 200, vocab), 4};
    const auto after_shared = [&](uint32_t first, size_t n) {
        std::vector<uint32_t> p(setup.prompt.begin(), setup.prompt.begin() + kBlock);
        const std::vector<uint32_t> own = prompt_of(first, n, vocab);
        p.insert(p.end(), own.begin(), own.end());
        return p;
    };
    const Req a{prompt_of(1, 40, vocab)}, c{after_shared(3, 12), 380};
    const Req b = stopping(make, tok, pool, {setup}, Req{after_shared(2, 60)}, 420);
    const auto s = alone_then_together(make, tok, pool, 0, 3, {setup}, {a, b, c}, "a paused request whose donor went, part of its history kept");
    require(s.pauses == 1 && s.taken_back == 0 && s.recomputed == 512 - kBlock,
            "a paused request with part of its history kept: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) + " taken back, " +
            std::to_string(s.recomputed) + " rows recomputed, against 1, 0 and 384");
}

// A donor that matches a paused request's history by tokens further than by how it was computed: C's prompt is B's prompt and B's first 80 tokens alone, so C's donor holds B's generated rows 188 to 268 as prompt rows.
// Paused as in partial_eviction, B must fork only the first block, which both computed as the setup's prompt, and recompute the other 384 rows.
void fork_within_class(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 14 * kBlock;
    const Req setup{prompt_of(7, 200, vocab), 4};
    std::vector<uint32_t> prompt(setup.prompt.begin(), setup.prompt.begin() + kBlock);
    const std::vector<uint32_t> own = prompt_of(2, 60, vocab);
    prompt.insert(prompt.end(), own.begin(), own.end());
    auto model = make(pool, 0);
    const std::vector<uint32_t> reply = ids_of(serve(*model, tok, 3, {{setup}, {Req{prompt}}}).back());
    Req b{prompt};
    b.stop = stop_at(tok, reply, 420);
    std::vector<uint32_t> longer = prompt;
    longer.insert(longer.end(), reply.begin(), reply.begin() + 80);
    const Req a{prompt_of(1, 40, vocab)}, c{longer, 370};
    const auto s = alone_then_together(make, tok, pool, 0, 3, {setup}, {a, b, c}, "a paused request beside a donor holding its reply as prompt rows");
    require(s.pauses == 1 && s.taken_back == 0 && s.recomputed == 512 - kBlock,
            "a paused request beside a donor holding its reply as prompt rows: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) +
            " taken back, " + std::to_string(s.recomputed) + " rows recomputed, against 1, 0 and 384");
}

// B paused as in take_back, its donor intact, and cancelled before it resumes, which C's longer reply (720 tokens in all, on 16 blocks) leaves time for: it ends where it waits, taking nothing back, and A and C finish as they do alone.
void cancel_while_paused_donor(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    const Req a{prompt_of(1, 40, vocab)}, b{prompt_of(2, 60, vocab)}, c{prompt_of(3, 20, vocab), 700};
    std::vector<Reply> alone;
    for (const Req& r : {a, c}) {
        auto model = make(pool, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(pool, 0);
    {
        server::Scheduler sched(*model, tok, 3, 64);
        std::thread runner([&] { sched.run(); });
        try {
            std::vector<std::shared_ptr<server::Request>> h;
            for (const Req& r : {a, b, c}) h.push_back(sched.submit(r.prompt, params_of(r)));
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (sched.stats().pauses == 0) {
                require(std::chrono::steady_clock::now() < until, "a paused request cancelled: nothing paused in 60 seconds");
                std::this_thread::yield();
            }
            h[1]->cancel();
            same(alone[0], drain(*h[0]), "beside a paused request cancelled, the uncapped request");
            same(alone[1], drain(*h[2]), "beside a paused request cancelled, the capped request");
            server::Request::Token t;
            while (h[1]->next(t, server::Request::Clock::now() + std::chrono::seconds(60)) == server::Request::Next::id) {}
            require(h[1]->finish() == "cancel", "a paused request cancelled ended with " + h[1]->finish());
            const auto s = sched.stats();
            require(s.pauses == 1 && s.taken_back == 0 && s.active == 0 && s.queued == 0,
                    "a paused request cancelled: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) + " taken back, " +
                    std::to_string(s.active) + " active and " + std::to_string(s.queued) + " queued, against 1, 0, 0 and 0");
            ledger(s, *model, "after a paused request with its donor was cancelled");
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(model->kv_used_bytes() == 0, "a pool holds blocks once the scheduler has stopped, after a paused request was cancelled");
}

// On a split of `stages` CPUs, a request cancelled from inside a stage of its pass, at the last device's tenth submission, is in flight until a later round retires that pass.
// It ends then with its history kept as a donor, which a request repeating its prompt forks, and once the scheduler has stopped the whole pool is free.
void cancelled_in_flight(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, size_t stages) {
    const std::string what = "a request cancelled in flight on " + std::to_string(stages) + " CPUs";
    const std::vector<std::shared_ptr<Hooked>> devices = hooked(stages);
    auto model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(1024, 0);
    const Req v{prompt_of(1, 130, vocab), 300}, f{prompt_of(1, 130, vocab), 4};
    std::shared_ptr<server::Request> hv;
    size_t submits = 0;
    {
        server::Scheduler sched(*model, tok, 3, 64);
        hv = sched.submit(v.prompt, params_of(v));
        devices.back()->hook = [&submits, &hv] { if (++submits == 10) hv->cancel(); };
        std::thread runner([&] { sched.run(); });
        try {
            size_t got = 0;
            server::Request::Token t;
            server::Request::Next next;
            while ((next = hv->next(t, server::Request::Clock::now() + std::chrono::seconds(120))) == server::Request::Next::id) ++got;
            require(next == server::Request::Next::end && hv->finish() == "cancel" && got < 300,
                    what + ": it ended with " + hv->finish() + " after " + std::to_string(got) + " tokens, against cancel before 300");
            const Reply forked = drain(*sched.submit(f.prompt, params_of(f)));
            const auto s = sched.stats();
            require(forked.size() == 4 && s.prefix_hits == 1 && s.prefix_tokens == kBlock,
                    what + ": a request repeating its prompt gave " + std::to_string(forked.size()) + " tokens and reused " + std::to_string(s.prefix_tokens) +
                    ", against 4 and " + std::to_string(kBlock));
            ledger(s, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    devices.back()->hook = nullptr;
    whole_pool_free(*model, 1024, vocab, what);
}

// On `stages` CPUs, a stop from inside the first stage of a pass, at the first device's sixth submission, finds the pass in flight.
// The scheduler abandons it before it releases the request, which ends cancelled, and the whole pool is free while the request's handle still holds its sequence.
void stopped_in_flight(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, size_t stages) {
    const std::string what = "a stop with a pass in flight on " + std::to_string(stages) + " CPU" + (stages == 1 ? "" : "s");
    const std::vector<std::shared_ptr<Hooked>> devices = hooked(stages);
    auto model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(1024, 0);
    const Req v{prompt_of(1, 130, vocab), 300};
    std::shared_ptr<server::Request> hv;
    {
        server::Scheduler sched(*model, tok, 3, 64);
        hv = sched.submit(v.prompt, params_of(v));
        size_t submits = 0;
        devices.front()->hook = [&submits, &sched] { if (++submits == 6) sched.stop(); };
        std::thread runner([&] { sched.run(); });
        runner.join();
        devices.front()->hook = nullptr;
        server::Request::Token t;
        while (hv->next(t, server::Request::Clock::now() + std::chrono::seconds(10)) == server::Request::Next::id) {}
        require(hv->finish() == "cancel", what + ": the request ended with " + hv->finish());
        const auto s = sched.stats();
        require(s.active == 0 && s.queued == 0 && s.paused == 0 && s.donors == 0, what + ": requests or donors were left once it stopped");
        for (size_t p = 0; p < s.reserved.size(); ++p) require(s.reserved[p] == 0, what + ": the ledger holds blocks once it stopped");
    }
    whole_pool_free(*model, 1024, vocab, what);
}

// Every request submitted before the scheduler starts, so its first admission takes them in order and their prompts share its first pass.
std::vector<Reply> started_together(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, const std::vector<Req>& reqs,
                                    server::Scheduler::Stats& stats) {
    server::Scheduler sched(model, tok, max_seqs, 64);
    std::vector<std::shared_ptr<server::Request>> handles;
    for (const Req& r : reqs) handles.push_back(sched.submit(r.prompt, params_of(r)));
    std::thread runner([&] { sched.run(); });
    std::vector<Reply> replies;
    try {
        for (auto& h : handles) replies.push_back(drain(*h));
        stats = sched.stats();
        ledger(stats, model, "after requests started together");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    return replies;
}

// A resumed request that cannot grow waits for room rather than being paused again, since only requests admitted after it give room up for it.
// On 16 blocks A's growth pauses B, B takes its donor back once A ends at its stop string, and B's next step at 768 tokens finds the room held by capped K and by L, which started beside it and ends 15 passes later.
void resumed_short_of_room(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    const Req a = stopping(make, tok, pool, {}, Req{prompt_of(1, 20, vocab)}, 380);
    const Req b = stopping(make, tok, pool, {}, Req{prompt_of(2, 120, vocab)}, 800);
    const Req k{prompt_of(3, 20, vocab), 700}, l{prompt_of(4, 20, vocab), 300};
    const std::vector<Req> reqs = {a, b, k, l};
    std::vector<Reply> alone;
    for (const Req& r : reqs) {
        auto model = make(pool, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(pool, 0);
    server::Scheduler::Stats s;
    const std::vector<Reply> together = started_together(*model, tok, 3, reqs, s);
    for (size_t i = 0; i < reqs.size(); ++i) same(alone[i], together[i], "beside a resumed request short of room, request " + std::to_string(i));
    require(s.pauses == 1 && s.taken_back == 1 && s.recomputed == 0 && s.stalls > 0,
            "a resumed request short of room for its own growth: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) +
            " taken back, " + std::to_string(s.recomputed) + " rows recomputed and " + std::to_string(s.stalls) + " passes sat out, against 1, 1, 0 and some");
}

// A request that cannot grow sits out passes for longer than a block while a capped request holds the room it needs, and nothing is admitted meanwhile.
// On 10 blocks uncapped A and capped E (128 tokens and 300) start together and capped D (138 and 600) takes the room E leaves, so A's step at 384 tokens waits for D's end, and capped F, submitted meanwhile, starts only after it.
void stall_holds_room(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 10 * kBlock;
    const Req a{prompt_of(1, 40, vocab)}, e{prompt_of(3, 128, vocab), 300}, d{prompt_of(4, 138, vocab), 600}, f{prompt_of(5, 10, vocab), 20};
    std::vector<Reply> alone;
    for (const Req& r : {a, e, d, f}) {
        auto model = make(pool, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(pool, 0);
    server::Scheduler sched(*model, tok, 3, 64);
    std::vector<std::shared_ptr<server::Request>> h;
    for (const Req& r : {a, e, d}) h.push_back(sched.submit(r.prompt, params_of(r)));
    std::thread runner([&] { sched.run(); });
    try {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (sched.stats().stalls == 0) {
            require(std::chrono::steady_clock::now() < until, "a request short of room: nothing sat out a pass in 60 seconds");
            std::this_thread::yield();
        }
        h.push_back(sched.submit(f.prompt, params_of(f)));
        server::Request::Token t;
        require(h[3]->next(t, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, "a request submitted during a stall gave nothing");
        require(h[2]->finish() == "length", "a request submitted while another sat out passes started before the capped request holding its room ended");
        Reply late = {t};
        for (const auto& rest : drain(*h[3])) late.push_back(rest);
        same(alone[3], late, "a request submitted during a stall");
        for (size_t i = 0; i < 3; ++i) same(alone[i], drain(*h[i]), "beside a stall, request " + std::to_string(i));
        const auto s = sched.stats();
        require(s.stalls > kBlock && s.pauses == 0, "a stall held by a capped request: " + std::to_string(s.stalls) + " passes sat out and " +
                std::to_string(s.pauses) + " pauses, against more than a block and 0");
        ledger(s, *model, "after a stall");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// An older request's growth step that falls due in the iteration a newer request could first be admitted takes the room first.
// On 10 blocks uncapped A and capped E start together and capped D waits; E ends before A's step at 384 tokens falls due, and D fits only in the room that step needs, so D waits for A's stop rather than A sitting out D's reply.
void growth_before_admission(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 10 * kBlock;
    const Req a = stopping(make, tok, pool, {}, Req{prompt_of(1, 40, vocab)}, 500);
    const Req e{prompt_of(3, 128, vocab), 345}, d{prompt_of(4, 138, vocab), 600};
    const std::vector<Req> reqs = {a, e, d};
    std::vector<Reply> alone;
    for (const Req& r : reqs) {
        auto model = make(pool, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(pool, 0);
    server::Scheduler::Stats s;
    const std::vector<Reply> together = started_together(*model, tok, 3, reqs, s);
    for (size_t i = 0; i < reqs.size(); ++i) same(alone[i], together[i], "beside a growth step due at an admission, request " + std::to_string(i));
    require(s.stalls == 0, "a growth step due in the iteration a newer request could be admitted: the older request sat out " + std::to_string(s.stalls) +
            " passes, against 0");
}

// A paused request cancelled while its donor holds less than a block leaves no donor behind, since no fork can share one.
// On 8 blocks uncapped A and capped E start together and uncapped B waits for E's end; A's step at 384 tokens pauses B with its donor kept, and B is cancelled while it waits, so once A ends A's history is the one donor.
void cancelled_short_donor(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 8 * kBlock;
    const Req a = stopping(make, tok, pool, {}, Req{prompt_of(1, 40, vocab)}, 650);
    const Req e{prompt_of(3, 20, vocab), 300}, b{prompt_of(2, 10, vocab)};
    std::vector<Reply> alone;
    for (const Req& r : {a, e}) {
        auto model = make(pool, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    auto model = make(pool, 0);
    server::Scheduler sched(*model, tok, 3, 64);
    std::vector<std::shared_ptr<server::Request>> h;
    for (const Req& r : {a, e, b}) h.push_back(sched.submit(r.prompt, params_of(r)));
    // The cancel comes from the scheduler's own thread, as the first pass after the pause retires, so no round can admit the paused request between its pause and its cancel.
    std::atomic<bool> cancelled{false};
    sched.on_retire = [&](const server::Scheduler::Retired&) {
        if (cancelled.load() || !sched.stats().pauses) return;
        h[2]->cancel();
        cancelled.store(true);
    };
    std::thread runner([&] { sched.run(); });
    try {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!cancelled.load()) {
            require(std::chrono::steady_clock::now() < until, "a paused request with a short donor: nothing paused in 60 seconds");
            std::this_thread::yield();
        }
        same(alone[0], drain(*h[0]), "beside a paused request with a short donor cancelled, the uncapped request");
        same(alone[1], drain(*h[1]), "beside a paused request with a short donor cancelled, the capped request");
        server::Request::Token t;
        while (h[2]->next(t, server::Request::Clock::now() + std::chrono::seconds(60)) == server::Request::Next::id) {}
        require(h[2]->finish() == "cancel", "a paused request with a short donor cancelled ended with " + h[2]->finish());
        const auto s = sched.stats();
        require(s.pauses == 1 && s.taken_back == 0 && s.paused == 0 && s.donors == 1,
                "a paused request with a short donor cancelled: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) + " taken back, " +
                std::to_string(s.paused) + " paused and " + std::to_string(s.donors) + " donors, against 1, 0, 0 and 1");
        ledger(s, *model, "after a paused request with a short donor was cancelled");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// A request refused room evicts no donor for it.
// On 8 blocks the setup leaves a donor of 2 blocks, N finds no room beside capped K even with it gone, and K ends holding less than a block, so N then starts beside the donor, which a prompt repeating the setup's still forks.
void refused_evicts_nothing(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 8 * kBlock;
    const Req setup{prompt_of(5, 200, vocab), 1};
    const Req k = stopping(make, tok, pool, {}, Req{prompt_of(1, 10, vocab), 600}, 60);
    const Req n{prompt_of(2, 10, vocab), 600};
    auto model = make(pool, 0);
    server::Scheduler sched(*model, tok, 3, 64);
    std::thread runner([&] { sched.run(); });
    try {
        drain(*sched.submit(setup.prompt, params_of(setup)));
        const auto hk = sched.submit(k.prompt, params_of(k)), hn = sched.submit(n.prompt, params_of(n));
        server::Request::Token t;
        require(hn->next(t, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, "a request refused room gave nothing");
        require(!hk->finish().empty(), "a request that finds no room beside a capped request started before that request ended");
        drain(*hk);
        drain(*hn);
        drain(*sched.submit(setup.prompt, params_of(Req{setup.prompt, 4})));
        const auto s = sched.stats();
        require(s.prefix_hits == 1 && s.prefix_tokens == kBlock,
                "a donor evicted for a request then refused: a prompt repeating it reused " + std::to_string(s.prefix_tokens) + " tokens against " +
                std::to_string(kBlock));
        ledger(s, *model, "after a request refused room");
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// Paused requests wait apart from the queue, so they do not fill the queue --max-queue bounds: with a queue of one, A's growth pauses B as in three_uncapped and a request submitted meanwhile is queued, not refused.
// The scheduler's own thread submits B inside A's first pass and C inside the first pass that finds B paused, so the case follows from the blocks alone and not from when the test's thread runs.
void paused_outside_queue(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::vector<std::shared_ptr<Hooked>> devices = hooked(1);
    auto model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(1024, 0);
    {
        server::Scheduler sched(*model, tok, 3, 1);
        const std::shared_ptr<server::Request> a = sched.submit(prompt_of(1, 40, vocab), params_of(Req{}));
        std::shared_ptr<server::Request> b, c;
        std::atomic<int> submitted{0};   // 1 once B is submitted, 2 once C is, 3 once C was refused
        devices[0]->hook = [&] {
            const int s = submitted.load();
            if (s == 0) {
                b = sched.submit(prompt_of(2, 9, vocab), params_of(Req{}));
                submitted.store(1);
            } else if (s == 1 && sched.stats().paused > 0) {
                try {
                    c = sched.submit(prompt_of(3, 10, vocab), params_of(Req{{}, 4}));
                    submitted.store(2);
                } catch (const server::QueueFull&) {
                    submitted.store(3);
                }
            }
        };
        std::thread runner([&] { sched.run(); });
        try {
            // Only a hang guard: C is submitted once B is paused, whenever this thread runs.
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (submitted.load() < 2) {
                require(std::chrono::steady_clock::now() < limit, "paused requests and the queue: nothing paused in 60 seconds");
                std::this_thread::yield();
            }
            require(submitted.load() == 2, "a request submitted while another was paused was refused as though the queue were full");
            // New requests wait until no request is paused, so the queued one starts only once B has resumed.
            server::Request::Token t;
            require(c->next(t, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, "a request queued while another was paused gave nothing");
            require(sched.stats().paused == 0, "a request queued while another was paused started before the paused request resumed");
            for (const auto& r : {a, b, c}) drain(*r);
            require(sched.stats().pauses == 1, "paused requests and the queue: " + std::to_string(sched.stats().pauses) + " pauses, against one");
        } catch (...) {
            sched.stop();
            runner.join();
            devices[0]->hook = nullptr;
            throw;
        }
        sched.stop();
        runner.join();
    }
    devices[0]->hook = nullptr;
}

// A paused request whose client leaves ends where it waits, and once the scheduler stops every pool is empty again.
void cancelled_while_paused(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    auto model = make(1024, 0);
    {
        server::Scheduler sched(*model, tok, 3, 64);
        std::thread runner([&] { sched.run(); });
        try {
            std::vector<std::shared_ptr<server::Request>> handles;
            for (uint32_t first : {1u, 2u, 3u}) handles.push_back(sched.submit(prompt_of(first, 9 + first, vocab), params_of(Req{})));
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (sched.stats().pauses == 0) {
                require(std::chrono::steady_clock::now() < until, "a cancelled request: nothing paused in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            for (auto& h : handles) h->cancel();
            for (auto& h : handles) {
                server::Request::Token t;
                while (h->next(t, server::Request::Clock::now() + std::chrono::seconds(60)) == server::Request::Next::id) {}
                require(h->finish() == "cancel" || h->finish() == "length", "a cancelled request ended with " + h->finish());
            }
            const auto s = sched.stats();
            require(s.active == 0 && s.queued == 0, "cancelled requests left active or queued");
            ledger(s, *model, "after paused requests were cancelled");
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(model->kv_used_bytes() == 0, "a pool holds blocks once the scheduler has stopped");
}

// A model whose layers keep a recurrent state keeps no donor, so every resume recomputes its request's history from its start, and nothing is forked or taken back.
void no_donors(const server::Scheduler::Stats& s, const std::string& what) {
    require(s.donors == 0 && s.taken_back == 0 && s.prefix_hits == 0, what + ": " + std::to_string(s.donors) + " donors, " + std::to_string(s.taken_back) +
            " taken back and " + std::to_string(s.prefix_hits) + " prefixes reused, against none");
}

// `n` capped requests at once on `model`, each of which must run whole: with the scheduler's max_seqs at the model's state slots, every slot is free.
void every_slot_free(infer::Model& model, const bpe::Tokenizer& tok, uint32_t vocab, size_t n, const std::string& what) {
    std::vector<Req> reqs;
    for (uint32_t i = 0; i < n; ++i) reqs.push_back({prompt_of(20 + i, 30, vocab), 40});
    const std::vector<Reply> got = serve(model, tok, n, {reqs});
    for (size_t i = 0; i < n; ++i) require(got[i].size() == 40, what + ": request " + std::to_string(i) + " on the recycled slots gave " + std::to_string(got[i].size()) + " tokens");
}

// The hybrid model: requests paused and resumed give their replies alone on one CPU and over a two-CPU split with passes in flight, a follow-up turn recomputes its history, and cancels and stops leave every block and state slot free.
// With checkpoint slots (docs/SPECULATIVE.md, section 2) a finished request keeps its state at its prompt's last whole block for a follow-up to fork, and a paused request keeps its whole history as its checkpoint.
void hybrid_checkpoints(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const auto follow_up_of = [&](const Make& make, const Req& first) {
        auto model = make(1024, 0);
        std::vector<uint32_t> again = first.prompt;
        for (uint32_t id : ids_of(serve(*model, tok, 3, {{first}})[0])) again.push_back(id);
        const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
        again.insert(again.end(), more.begin(), more.end());
        return Req{again, 32};
    };
    const Make kept = on(weights, [] { return cpus(1); }, 3, 3);
    const Req a{prompt_of(5, 300, vocab), 40}, b{prompt_of(7, 300, vocab), 40};
    as_on_fresh(kept, tok, {a}, follow_up_of(kept, a), 2 * kBlock, "a hybrid model's follow-up turn from its checkpoint");
    const Make split = on(weights, [] { return cpus(2); }, 3, 3);
    as_on_fresh(split, tok, {a}, follow_up_of(split, a), 2 * kBlock, "a hybrid model's follow-up turn from its checkpoint over a two-CPU split");
    const Make one_slot = on(weights, [] { return cpus(1); }, 3, 1);
    as_on_fresh(one_slot, tok, {a, b}, follow_up_of(one_slot, a), 0, "a hybrid model's older conversation, its checkpoint taken by a newer one");
    as_on_fresh(one_slot, tok, {a, b}, follow_up_of(one_slot, b), 2 * kBlock, "a hybrid model's newer conversation, one checkpoint slot");
    take_back(kept, tok, vocab);
    // A donor parked at its checkpoint holds none of the live slots admission counts on: with one, the same prompt again forks it and runs, and with three, three requests at once beside the donor each find their slot.
    {
        auto model = on(weights, [] { return cpus(1); }, 1, 1)(1024, 0);
        server::Scheduler::Stats stats;
        const std::vector<Reply> r = serve(*model, tok, 1, {{a}, {a}}, &stats);
        same(r[0], r[1], "a hybrid model's repeated prompt with one live slot");
        require(stats.prefix_tokens == 2 * kBlock, "a hybrid model's repeated prompt with one live slot reused " + std::to_string(stats.prefix_tokens) + " tokens");
    }
    {
        const std::vector<Req> three = {{prompt_of(1, 40, vocab), 64}, {prompt_of(2, 9, vocab), 64}, {prompt_of(3, 23, vocab), 64}};
        auto model = kept(1024, 0);
        const std::vector<Reply> r = serve(*model, tok, 3, {{a}, three});
        for (size_t i = 0; i < three.size(); ++i) {
            auto fresh = kept(1024, 0);
            same(serve(*fresh, tok, 3, {{three[i]}})[0], r[1 + i], "a hybrid model's request beside a donor, " + std::to_string(i));
        }
    }
}

// Message boundaries (docs/SPECULATIVE.md, section 2, Host tier): a hybrid model with three checkpoint slots and a host tier, six turns each read again, each job's donor superseding the one before, its state thinned to four in host memory.
// A 2000-token request then evicts every donor, and an edit of turn 2 and a regenerated turn 6 fork the boundaries and give their replies on a fresh model; `whole`, where given, is how many boundaries the six turns leave in host memory.
void message_boundaries(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what, size_t host = (size_t)1 << 30,
                        const server::DiskOptions& disk = {}, size_t whole = 0) {
    // Under a disk tier its prompts read in passes of 16 rows, which measure no prompt rate, so its requests wait for their reads however fast the model computes.
    const int ub = disk.bytes ? 16 : 0;
    auto model = make(2048, ub);
    std::vector<std::vector<uint32_t>> prompts, nexts;
    Reply edit_reply, regen_reply;
    std::vector<uint32_t> edit;
    size_t edit_reused = 0, regen_reused = 0;
    server::Scheduler::Stats after_turns, after_long, stats;
    {
        server::Scheduler sched(*model, tok, 3, 64, 0, false, host, nullptr, 0, false, disk);
        std::thread runner([&] { sched.run(); });
        try {
            // The disk tier writes nothing until the model file's digest is known and its store made, which a slow runner reaches only after the turns would have ended.
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60); disk.bytes && !sched.stats().disk_ready;) {
                require(std::chrono::steady_clock::now() < until, what + ": the disk tier was not made in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::vector<uint32_t> prompt = prompt_of(5, 300, vocab);
            for (size_t turn = 1; turn <= 6; ++turn) {
                prompts.push_back(prompt);
                const auto h = sched.submit(prompt, params_of(Req{prompt, 100}));
                std::vector<uint32_t> next = prompt;
                for (uint32_t id : ids_of(drain(*h))) next.push_back(id);
                next.push_back(1);
                next.push_back(2);
                nexts.push_back(next);
                sched.follow(h, next, true);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills < turn) {
                    require(std::chrono::steady_clock::now() < until, what + ": a reply was not read again in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                // Under a disk tier each turn's writes end before the next turn, so what reaches the disk does not hang on how fast the turns come.
                while (disk.bytes && sched.stats().disk_in_flight) {
                    require(std::chrono::steady_clock::now() < until, what + ": the disk tier's writes did not end in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                prompt = next;
                const std::vector<uint32_t> tail = prompt_of(10 + (int)turn, 50, vocab);
                prompt.insert(prompt.end(), tail.begin(), tail.end());
            }
            after_turns = sched.stats();
            const std::vector<uint32_t> long_prompt = prompt_of(9, 2000, vocab);
            drain(*sched.submit(long_prompt, params_of(Req{long_prompt, 40})));
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60); disk.bytes && sched.stats().disk_in_flight;) {
                require(std::chrono::steady_clock::now() < until, what + ": the disk tier's writes did not end in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            after_long = sched.stats();
            edit = nexts[0];
            const std::vector<uint32_t> other = prompt_of(30, 50, vocab);
            edit.insert(edit.end(), other.begin(), other.end());
            const auto e = sched.submit(edit, params_of(Req{edit, 32}));
            edit_reply = drain(*e);
            edit_reused = e->reused();
            const auto g = sched.submit(prompts[5], params_of(Req{prompts[5], 32}));
            regen_reply = drain(*g);
            regen_reused = g->reused();
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(disk.bytes || after_long.boundaries == 4, what + ": " + std::to_string(after_long.boundaries) + " boundaries in host memory, against 4");
    require(!whole || after_turns.boundaries == whole,
            what + ": the turns left " + std::to_string(after_turns.boundaries) + " boundaries in host memory, against " + std::to_string(whole));
    require(!disk.bytes || whole || (stats.disk_hits >= 1 && stats.disk_errors == 0), what + ": " + std::to_string(stats.disk_hits) + " entries read back from disk");
    const size_t first = nexts[0].size() / kBlock * kBlock, newest = nexts[4].size() / kBlock * kBlock;
    // Under a small host tier beside a disk tier, a boundary room takes mid-write hangs on until the writer reaches it, so a request may fork an earlier boundary than the one before its message.
    const auto forked = [&](size_t reused, size_t most) { return disk.bytes && !whole ? reused && reused <= most && reused % kBlock == 0 : reused == most; };
    require(forked(edit_reused, first), what + ": the edit of turn 2 reused " + std::to_string(edit_reused) + " tokens, against " + std::to_string(first));
    require(forked(regen_reused, newest), what + ": the regenerated turn 6 reused " + std::to_string(regen_reused) + " tokens, against " + std::to_string(newest));
    // Under a disk tier a request may also fork a boundary read back where the history it would fork has left memory.
    require(disk.bytes ? stats.boundary_hits >= 2 : stats.boundary_hits == 2, what + ": " + std::to_string(stats.boundary_hits) + " requests forked a boundary, against 2");
    auto fresh = make(2048, ub);
    same(serve(*fresh, tok, 3, {{Req{edit, 32}}})[0], edit_reply, what + ", the edit of turn 2");
    fresh = make(2048, ub);
    same(serve(*fresh, tok, 3, {{Req{prompts[5], 32}}})[0], regen_reply, what + ", the regenerated turn 6");
}

// A boundary kept where a request's prompt passes its last user message's start (docs/SPECULATIVE.md, section 2, Host tier): a regenerate given that start keeps a boundary at the whole block below it beside its prompt-end checkpoint.
// An edit of the same message then forks that boundary with the regenerate's blocks and gives its reply on a fresh model, while without the start the regenerate keeps no boundary and the edit forks nothing.
void edit_after_regenerate(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::vector<uint32_t> before = prompt_of(5, 300, vocab);
    std::vector<uint32_t> regen = before, edit = before;
    const std::vector<uint32_t> message = prompt_of(6, 200, vocab), other = prompt_of(7, 50, vocab);
    regen.insert(regen.end(), message.begin(), message.end());
    edit.insert(edit.end(), other.begin(), other.end());
    for (const bool known : {true, false}) {
        const std::string what = std::string("an edit after a regenerate, ") + (known ? "the message's start given" : "no start given");
        auto model = make(2048, 0);
        Reply reply;
        size_t reused = 0;
        server::Scheduler::Stats stats;
        {
            server::Scheduler sched(*model, tok, 3, 64, 0, false, (size_t)1 << 30);
            std::thread runner([&] { sched.run(); });
            try {
                const size_t start = known ? before.size() : 0;
                drain(*sched.submit(regen, params_of(Req{regen, 32}), regen.size(), start));
                const auto e = sched.submit(edit, params_of(Req{edit, 32}), edit.size(), start);
                reply = drain(*e);
                reused = e->reused();
                stats = sched.stats();
                ledger(stats, *model, what);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        const size_t want = known ? before.size() / kBlock * kBlock : 0;
        require(reused == want, what + ": the edit reused " + std::to_string(reused) + " tokens, against " + std::to_string(want));
        require(stats.boundary_hits == (known ? 1u : 0u), what + ": " + std::to_string(stats.boundary_hits) + " requests forked a boundary");
        auto fresh = make(2048, 0);
        same(serve(*fresh, tok, 3, {{Req{edit, 32}}})[0], reply, what + ", the edit");
    }
    // A message that starts in its prompt's last whole block, where the request's own checkpoint sits, keeps its boundary too, so the state outlives the request's donor.
    const std::string what = "a short last message";
    auto model = make(2048, 0);
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 3, 64, 0, false, (size_t)1 << 30);
        std::thread runner([&] { sched.run(); });
        try {
            drain(*sched.submit(edit, params_of(Req{edit, 32}), edit.size(), before.size()));
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(stats.boundaries == 1, what + ": " + std::to_string(stats.boundaries) + " boundaries kept where the message starts, against 1");
}

// Message boundaries whose copies fail: three turns on a hybrid model with three checkpoint slots and a host tier, each reply read again, an unrelated request, and an edit of turn 2 while the last job's donor is on the devices.
// Copies to host memory failing keep no boundary and no host copy, copies from it failing keep the boundaries and the edit reads its prompt from the start, and either way every reply is its reply alone and the ledger adds up.
void boundary_faults(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, bool reads) {
    const std::string what = std::string("message boundaries, copies ") + (reads ? "to" : "from") + " host memory failing";
    std::shared_ptr<FailingCopies> failing;
    const Make make = on(weights, [&] {
        failing = std::make_shared<FailingCopies>();
        failing->set_threads(1);
        return std::vector<backend::BackendPtr>{failing};
    }, 3, 3);
    auto model = make(2048, 0);
    failing->fail_read = reads;
    failing->fail_write = !reads;
    std::vector<uint32_t> first, edit;
    Reply edit_reply;
    size_t edit_reused = 0;
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 3, 64, 0, false, (size_t)1 << 30);
        std::thread runner([&] { sched.run(); });
        try {
            std::vector<uint32_t> prompt = prompt_of(5, 300, vocab);
            for (size_t turn = 1; turn <= 3; ++turn) {
                const auto h = sched.submit(prompt, params_of(Req{prompt, 100}));
                std::vector<uint32_t> next = prompt;
                for (uint32_t id : ids_of(drain(*h))) next.push_back(id);
                next.push_back(1);
                next.push_back(2);
                if (turn == 1) first = next;
                sched.follow(h, next, true);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills < turn) {
                    require(std::chrono::steady_clock::now() < until, what + ": a reply was not read again in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                prompt = next;
                const std::vector<uint32_t> tail = prompt_of(10 + (int)turn, turn == 1 ? 200 : 50, vocab);
                prompt.insert(prompt.end(), tail.begin(), tail.end());
            }
            // An unrelated request's donor takes the place of the oldest superseded one, the first job's.
            const std::vector<uint32_t> unrelated = prompt_of(40, 300, vocab);
            drain(*sched.submit(unrelated, params_of(Req{unrelated, 32})));
            edit = first;
            const std::vector<uint32_t> other = prompt_of(30, 50, vocab);
            edit.insert(edit.end(), other.begin(), other.end());
            const auto g = sched.submit(edit, params_of(Req{edit, 32}));
            edit_reply = drain(*g);
            edit_reused = g->reused();
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    if (reads) {
        require(stats.boundaries == 0 && stats.host_donors == 0 && stats.host_bytes == 0,
                what + ": " + std::to_string(stats.boundaries) + " boundaries and " + std::to_string(stats.host_donors) + " donors in host memory");
    } else {
        require(stats.boundaries == 3 && stats.boundary_hits == 0 && edit_reused == 0,
                what + ": " + std::to_string(stats.boundaries) + " boundaries, " + std::to_string(stats.boundary_hits) + " forked, the edit reusing " +
                    std::to_string(edit_reused) + " tokens");
    }
    auto fresh = make(2048, 0);
    failing->fail_read = failing->fail_write = false;
    same(serve(*fresh, tok, 3, {{Req{edit, 32}}})[0], edit_reply, what + ", the edit of turn 2");
}

// A reply read again as prompt rows (docs/SPECULATIVE.md, section 2, Idle re-prefill): the ids a 300-token prompt's next turn begins with go to the scheduler, whose job reads them on a fork and keeps a donor for a follow-up to fork.
// The variants (idle, `interrupt` k, `small`, `writing`, `at_once`) are listed in AGENTS.md, Tests, and each gives every reply its reply alone.
enum class When { idle, writing, at_once };

void reprefilled(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, size_t interrupt, When when, const std::string& what, bool small = false) {
    const bool writing = when != When::idle;
    const Req first{prompt_of(5, 300, vocab), writing ? 400 : 100}, other{prompt_of(9, small ? 40 : 440, vocab), 20};
    // Answered at once, the follow-up must fit beside the job and the donor it forked, or the job gives way to it.
    const size_t pool = when == When::at_once ? 4096 : writing || !interrupt ? 2048 : 1024;
    std::vector<uint32_t> early = first.prompt, later = first.prompt;
    if (writing) {
        auto alone = make(pool, 16);
        const std::vector<uint32_t> ids = ids_of(serve(*alone, tok, 3, {{first}})[0]);
        early.insert(early.end(), ids.begin(), ids.begin() + 100);
        later.insert(later.end(), ids.begin(), ids.begin() + 380);
    }
    auto model = make(pool, 16);
    Reply follow_reply, other_reply;
    std::vector<uint32_t> next;
    {
        server::Scheduler sched(*model, tok, 3, 64);
        std::shared_ptr<server::Request> h, interrupter;
        size_t job_passes = 0, request_passes = 0;
        std::mutex handles;
        // A pass that holds none of the case's requests is the job's.
        sched.on_retire = [&](const server::Scheduler::Retired& t) {
            std::lock_guard<std::mutex> lock(handles);
            for (const server::Request* r : t.requests)
                if (r == h.get() || r == interrupter.get()) {
                    // 19 passes read the prompt, so the 120th has written more than 100 tokens and the 400th more than 380.
                    if (writing && r == h.get()) {
                        ++request_passes;
                        if (request_passes == 120) sched.follow(h, early, false);
                        if (request_passes == 400 && when == When::writing) sched.follow(h, later, false);
                    }
                    return;
                }
            if (++job_passes == interrupt) interrupter = sched.submit(other.prompt, params_of(other));
        };
        std::thread runner([&] { sched.run(); });
        try {
            {
                std::lock_guard<std::mutex> lock(handles);
                h = sched.submit(first.prompt, params_of(first));
            }
            const Reply reply = drain(*h);
            if (writing) {
                const size_t rows = sched.stats().reprefill_rows;
                const size_t want = when == When::writing ? 384 : 128;
                require(rows == want, what + ": the job read " + std::to_string(rows) + " rows while the reply was written, against " + std::to_string(want));
            }
            next = first.prompt;
            for (uint32_t id : ids_of(reply)) next.push_back(id);
            next.push_back(1);
            next.push_back(2);
            sched.follow(h, next, true);
            const auto read_again = [&] {
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills == 0) {
                    require(std::chrono::steady_clock::now() < until, what + ": the reply was not read again in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            };
            if (when != When::at_once) read_again();
            // A regenerated reply resends the prompt alone: it forks the request's own donor, which the job's donor stands beside, at the prompt's last whole block, and gives the reply again.
            if (when == When::idle && !interrupt) {
                const auto g = sched.submit(first.prompt, params_of(first));
                same(reply, drain(*g), what + ", the regenerated reply");
                require(g->reused() == 2 * kBlock, what + ": the regenerated reply reused " + std::to_string(g->reused()) + " tokens, against " + std::to_string(2 * kBlock));
            }
            if (interrupt) {
                std::shared_ptr<server::Request> o;
                {
                    std::lock_guard<std::mutex> lock(handles);
                    o = interrupter;
                }
                require(o != nullptr, what + ": the job took fewer than " + std::to_string(interrupt) + " passes");
                other_reply = drain(*o);
            }
            std::vector<uint32_t> again = next;
            const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
            again.insert(again.end(), more.begin(), more.end());
            const Req follow{again, 32};
            const auto f = sched.submit(follow.prompt, params_of(follow));
            follow_reply = drain(*f);
            const size_t kept = next.size() / kBlock * kBlock;
            if (when == When::at_once) {
                // With passes in flight the job may be in a pass as the follow-up comes, when it forks the donor instead.
                const size_t least = (sched.stats().passes > 1 ? 2 : 3) * kBlock;
                require(f->reused() >= least && f->reused() % kBlock == 0,
                        what + ": the follow-up turn reused " + std::to_string(f->reused()) + " tokens, against " + std::to_string(least) + " or more");
                read_again();
            } else {
                require(f->reused() == kept, what + ": the follow-up turn reused " + std::to_string(f->reused()) + " tokens, against " + std::to_string(kept));
            }
            const server::Scheduler::Stats stats = sched.stats();
            require(stats.reprefills == 1 && stats.reprefill_cancels == (interrupt && !small ? 1u : 0u),
                    what + ": " + std::to_string(stats.reprefills) + " jobs done and " + std::to_string(stats.reprefill_cancels) + " cancelled");
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    std::vector<uint32_t> again = next;
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    auto fresh = make(pool, 16);
    same(serve(*fresh, tok, 3, {{Req{again, 32}}})[0], follow_reply, what + ", the follow-up turn");
    if (interrupt) {
        fresh = make(pool, 16);
        same(serve(*fresh, tok, 3, {{other}})[0], other_reply, what + ", the request that cancelled the job");
    }
}

// A job beside a decoding request over a two-CPU split, with a pass in flight on each stage and a pass budget past a job's chunk (Scheduler, kJobChunk).
// `idle`: a job begun once the reply has ended takes no pass from that request's first pass to its last, while otherwise a job begun mid-reply reads beside its decode rows, at most a chunk a pass.
void job_beside_decode(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, bool idle) {
    const std::string what = idle ? "a job begun at idle beside a decoding request over a two-CPU split" : "a job begun while its reply is written over a two-CPU split";
    const size_t kJobChunk = 64, ubatch = 256;
    const Make make = on(weights, [] { return cpus(2); });
    const Req first{prompt_of(5, 300, vocab), idle ? 100 : 400}, other{prompt_of(9, 40, vocab), 200};
    Reply alone;
    std::vector<uint32_t> early = first.prompt;
    {
        auto model = make(4096, (int)ubatch);
        const std::vector<Reply> replies = serve(*model, tok, 3, {{idle ? other : first}});
        alone = replies[0];
        if (!idle) {
            const std::vector<uint32_t> ids = ids_of(alone);
            early.insert(early.end(), ids.begin(), ids.begin() + 100);
        }
    }
    auto model = make(4096, (int)ubatch);
    Reply reply;
    std::vector<std::pair<bool, size_t>> passes;   // per retired pass, whether it held the decoding request and the job's rows in it
    size_t followed_at = 0;                        // the passes retired when the job's ids were given
    {
        server::Scheduler sched(*model, tok, 3, 64);
        require(sched.stats().passes == 2, what + ": the scheduler keeps " + std::to_string(sched.stats().passes) + " passes in flight, against 2");
        std::shared_ptr<server::Request> h, o;
        size_t request_passes = 0;
        std::mutex handles;
        sched.on_retire = [&](const server::Scheduler::Retired& t) {
            std::lock_guard<std::mutex> lock(handles);
            const server::Request* decoding = idle ? o.get() : h.get();
            bool held = false;
            size_t job = 0;
            for (size_t i = 0; i < t.requests.size(); ++i) {
                if (t.requests[i] == decoding) held = true;
                else if (t.requests[i] != h.get()) job += t.rows[i];
            }
            passes.push_back({held, job});
            // The prompt takes two passes, so the 120th of the request has written more than 100 tokens.
            if (!idle && held && ++request_passes == 120) {
                sched.follow(h, early, false);
                followed_at = passes.size();
            }
        };
        std::thread runner([&] { sched.run(); });
        try {
            {
                std::lock_guard<std::mutex> lock(handles);
                h = sched.submit(first.prompt, params_of(first));
            }
            if (idle) {
                std::vector<uint32_t> next = first.prompt;
                for (uint32_t id : ids_of(drain(*h))) next.push_back(id);
                const std::vector<uint32_t> more = prompt_of(3, 1200, vocab);
                next.insert(next.end(), more.begin(), more.end());
                {
                    std::lock_guard<std::mutex> lock(handles);
                    sched.follow(h, next, true);
                    followed_at = passes.size();
                    o = sched.submit(other.prompt, params_of(other));
                }
                reply = drain(*o);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills == 0) {
                    require(std::chrono::steady_clock::now() < until, what + ": the job did not complete in 60 seconds once the request had ended");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else {
                reply = drain(*h);
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    same(alone, reply, what);
    size_t first_pass = passes.size(), last_pass = 0, most = 0, read = 0;
    for (size_t i = followed_at; i < passes.size(); ++i)
        if (passes[i].first) {
            first_pass = std::min(first_pass, i);
            last_pass = i;
        }
    require(first_pass < passes.size(), what + ": no pass held the request once the job's ids were given");
    for (size_t i = first_pass; i <= last_pass; ++i) {
        most = std::max(most, passes[i].second);
        read += passes[i].second;
    }
    if (idle) {
        require(most == 0, what + ": a pass carried " + std::to_string(most) + " rows of the job while the request decoded, against none");
    } else {
        require(most <= kJobChunk, what + ": a pass carried " + std::to_string(most) + " rows of the job beside the reply it follows, against at most " + std::to_string(kJobChunk));
        require(read > 0, what + ": the job read nothing while its reply was written");
    }
}

// A job begun while its reply is written forks its request's prompt blocks whenever its ids arrive, also in a round where a pass holds the request, and never reads the prompt again.
// Eight requests in turn over a two-CPU split, each a 300-token prompt and 200 tokens, give their next turn's ids from the reader's thread once 100 tokens are read and whole at the end, each job reading the one block past its request's two.
void job_forks_running(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "a job begun while its reply is written";
    auto model = on(weights, [] { return cpus(2); })(4096, 256);
    server::Scheduler sched(*model, tok, 3, 64);
    std::thread runner([&] { sched.run(); });
    try {
        for (uint32_t k = 0; k < 8; ++k) {
            const Req r{prompt_of(20 + k, 300, vocab), 200};
            const size_t before = sched.stats().reprefill_rows;
            const auto h = sched.submit(r.prompt, params_of(r));
            std::vector<uint32_t> next = r.prompt;
            server::Request::Token t;
            for (;;) {
                const auto got = h->next(t, server::Request::Clock::now() + std::chrono::seconds(120));
                if (got == server::Request::Next::end) break;
                require(got == server::Request::Next::id, what + ": a channel gave nothing for 120 seconds");
                next.push_back(t.id);
                if (next.size() == r.prompt.size() + 100) sched.follow(h, next, false);
            }
            next.push_back(1);
            next.push_back(2);
            sched.follow(h, next, true);
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60); sched.stats().reprefills != k + 1;) {
                require(std::chrono::steady_clock::now() < until, what + ": the job of request " + std::to_string(k) + " did not complete in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const size_t rows = sched.stats().reprefill_rows - before, past = next.size() / kBlock * kBlock - r.prompt.size() / kBlock * kBlock;
            require(rows <= past, what + ": the job of request " + std::to_string(k) + " read " + std::to_string(rows) + " rows, against the " + std::to_string(past) + " past its request's prompt blocks");
        }
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
}

// A job whose ids grow while its reply is written reserves the blocks they take before it reads them: on 16 blocks a request needing 5 finds no free room beside a 700-token request and its 896-token job, so the job gives way.
// Every request runs to its length with its reply alone, and once the reply has ended the job starts again and the follow-up forks its 896 tokens.
void writing_growth(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "a job whose ids grow while its reply is written";
    const Req first{prompt_of(5, 300, vocab), 700}, other{prompt_of(11, 500, vocab), 100};
    std::vector<uint32_t> early = first.prompt, later = first.prompt;
    {
        auto alone = make(2048, 16);
        const std::vector<uint32_t> ids = ids_of(serve(*alone, tok, 3, {{first}})[0]);
        early.insert(early.end(), ids.begin(), ids.begin() + 100);
        later.insert(later.end(), ids.begin(), ids.begin() + 600);
    }
    auto model = make(2048, 16);
    Reply first_reply, other_reply, follow_reply;
    std::vector<uint32_t> again;
    {
        server::Scheduler sched(*model, tok, 3, 64);
        std::shared_ptr<server::Request> h, o;
        size_t passes = 0;
        std::mutex handles;
        // 19 passes read the prompt, so the 120th has written more than 100 tokens, the 620th more than 600, and by the 640th the job has read its 896.
        sched.on_retire = [&](const server::Scheduler::Retired& t) {
            std::lock_guard<std::mutex> lock(handles);
            if (std::find(t.requests.begin(), t.requests.end(), h.get()) == t.requests.end()) return;
            ++passes;
            if (passes == 120) sched.follow(h, early, false);
            if (passes == 620) sched.follow(h, later, false);
            if (passes == 640) o = sched.submit(other.prompt, params_of(other));
        };
        std::thread runner([&] { sched.run(); });
        try {
            {
                std::lock_guard<std::mutex> lock(handles);
                h = sched.submit(first.prompt, params_of(first));
            }
            first_reply = drain(*h);
            std::shared_ptr<server::Request> other_handle;
            {
                std::lock_guard<std::mutex> lock(handles);
                other_handle = o;
            }
            require(other_handle != nullptr, what + ": the request needing the room was not submitted");
            other_reply = drain(*other_handle);
            std::vector<uint32_t> next = first.prompt;
            for (uint32_t id : ids_of(first_reply)) next.push_back(id);
            next.push_back(1);
            next.push_back(2);
            sched.follow(h, next, true);
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (sched.stats().reprefills == 0) {
                require(std::chrono::steady_clock::now() < until, what + ": the reply was not read again in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            again = next;
            const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
            again.insert(again.end(), more.begin(), more.end());
            const auto f = sched.submit(again, params_of(Req{again, 32}));
            follow_reply = drain(*f);
            require(f->reused() == 7 * kBlock, what + ": the follow-up turn reused " + std::to_string(f->reused()) + " tokens, against " + std::to_string(7 * kBlock));
            const server::Scheduler::Stats stats = sched.stats();
            require(stats.reprefill_cancels >= 1, what + ": the job did not give way to the request needing its room");
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    auto fresh = make(2048, 16);
    same(serve(*fresh, tok, 3, {{first}})[0], first_reply, what + ", its request");
    fresh = make(2048, 16);
    same(serve(*fresh, tok, 3, {{other}})[0], other_reply, what + ", the request needing its room");
    fresh = make(2048, 16);
    same(serve(*fresh, tok, 3, {{Req{again, 32}}})[0], follow_reply, what + ", the follow-up turn");
}

// Donors kept in host memory (docs/SPECULATIVE.md, section 2, Host tier): two conversations of a 300-token prompt alternate on a pool of 4 blocks of 128, so each turn's admission evicts the other conversation's donor.
// With a host tier each follow-up promotes its conversation's donor back and forks its 256 tokens, without one it reuses nothing, and with `fail` a failing copy to or from host memory loses only the copy; every reply is its reply alone.
enum class HostFault { none, write_back, promotion };

// With a `width` above 1 the devices form tensor groups of that many (docs/TENSOR-SPLIT.md), each member's storage copied to host memory and back on its own.
void host_tier(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, size_t devices, size_t checkpoints, HostFault fault,
               const std::string& what, size_t width = 1) {
    std::shared_ptr<FailingCopies> failing;
    const Make make = on(weights, [&] {
        failing = std::make_shared<FailingCopies>();
        failing->set_threads(1);
        std::vector<backend::BackendPtr> b{failing};
        for (size_t d = 1; d < devices; ++d) b.push_back(cpus(1)[0]);
        return b;
    }, 3, checkpoints, 0, 0, false, width);
    const Req a{prompt_of(5, 300, vocab), 40}, b{prompt_of(7, 300, vocab), 40};
    const auto follow = [&](const Req& first, uint32_t seed) {
        auto model = make(512, 0);
        std::vector<uint32_t> again = first.prompt;
        for (uint32_t id : ids_of(serve(*model, tok, 3, {{first}})[0])) again.push_back(id);
        const std::vector<uint32_t> more = prompt_of(seed, 30, vocab);
        again.insert(again.end(), more.begin(), more.end());
        return Req{again, 32};
    };
    const Req fa = follow(a, 6), fb = follow(b, 8);
    std::vector<Reply> alone;
    for (const Req& r : {a, b, fa, fb}) {
        auto model = make(512, 0);
        alone.push_back(serve(*model, tok, 3, {{r}})[0]);
    }
    for (const bool host : {false, true}) {
        auto model = make(512, 0);
        failing->fail_read = fault == HostFault::write_back;
        failing->fail_write = fault == HostFault::promotion;
        server::Scheduler::Stats stats;
        const std::vector<Reply> got = serve(*model, tok, 3, {{a}, {b}, {fa}, {fb}}, &stats, 0, host ? (size_t)1 << 30 : 0);
        const std::string arm = what + (host ? ", with a host tier" : ", without one");
        for (size_t i = 0; i < got.size(); ++i) same(alone[i], got[i], arm + ", request " + std::to_string(i));
        const bool hits = host && fault == HostFault::none;
        require(stats.host_hits == (hits ? 2u : 0u) && stats.prefix_tokens == (hits ? 4 * kBlock : 0u),
                arm + ": " + std::to_string(stats.host_hits) + " promotions and " + std::to_string(stats.prefix_tokens) + " tokens reused");
        // The second follow-up evicts the first's donor too: on the dense model its 384 tokens are a third entry, and on the hybrid one its checkpoint is at the first turn's 256, whose entry it renews; a failed write-back keeps nothing.
        const size_t held = !host || fault == HostFault::write_back ? 0 : checkpoints ? 2 : 3;
        require(stats.host_donors == held && (stats.host_bytes == 0) == (held == 0),
                arm + ": " + std::to_string(stats.host_donors) + " donors held in host memory, against " + std::to_string(held));
    }
}

// A job's donor supersedes the request's donor of the same conversation (one copy per conversation in each tier): on 16 blocks, C's donor and A's request donor and, read again, A's job's donor of 384 tokens.
// An 800-token request then takes a donor's room: A's superseded donor goes though C is older, and is not copied to host memory, so C's repeat forks 256 tokens, A's follow-up its job's 384, and every reply is its reply alone.
void superseded_donor(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    for (const size_t host : {(size_t)0, (size_t)1 << 30}) {
        const std::string what = std::string("a superseded donor") + (host ? " with a host tier" : "");
        const Req c{prompt_of(3, 300, vocab), 40}, a{prompt_of(5, 300, vocab), 100}, big{prompt_of(9, 800, vocab), 40};
        auto model = make(2048, 0);
        Reply a_reply, c_again_reply, follow_reply;
        std::vector<uint32_t> next, c_again;
        size_t c_reused = 0, follow_reused = 0;
        server::Scheduler::Stats stats;
        {
            // Four at once, so the donor count, which max_seqs bounds, evicts nothing here.
            server::Scheduler sched(*model, tok, 4, 64, 0, false, host);
            std::thread runner([&] { sched.run(); });
            try {
                drain(*sched.submit(c.prompt, params_of(c)));
                const auto h = sched.submit(a.prompt, params_of(a));
                a_reply = drain(*h);
                next = a.prompt;
                for (uint32_t id : ids_of(a_reply)) next.push_back(id);
                next.push_back(1);
                next.push_back(2);
                sched.follow(h, next, true);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills == 0) {
                    require(std::chrono::steady_clock::now() < until, what + ": the reply was not read again in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                drain(*sched.submit(big.prompt, params_of(big)));
                c_again = c.prompt;
                const std::vector<uint32_t> more = prompt_of(7, 30, vocab);
                c_again.insert(c_again.end(), more.begin(), more.end());
                const auto ca = sched.submit(c_again, params_of(Req{c_again, 32}));
                c_again_reply = drain(*ca);
                c_reused = ca->reused();
                stats = sched.stats();
                std::vector<uint32_t> follow = next;
                const std::vector<uint32_t> tail = prompt_of(6, 30, vocab);
                follow.insert(follow.end(), tail.begin(), tail.end());
                const auto f = sched.submit(follow, params_of(Req{follow, 32}));
                follow_reply = drain(*f);
                follow_reused = f->reused();
                next = follow;
                ledger(sched.stats(), *model, what);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        require(c_reused == 2 * kBlock, what + ": the older conversation's repeat reused " + std::to_string(c_reused) + " tokens, against " + std::to_string(2 * kBlock));
        require(follow_reused == 3 * kBlock, what + ": the follow-up reused " + std::to_string(follow_reused) + " tokens, against " + std::to_string(3 * kBlock));
        require(stats.host_donors == 0 && stats.host_bytes_moved == 0,
                what + ": " + std::to_string(stats.host_donors) + " donors in host memory and " + std::to_string(stats.host_bytes_moved) + " bytes moved");
        auto fresh = make(2048, 0);
        same(serve(*fresh, tok, 3, {{Req{c_again, 32}}})[0], c_again_reply, what + ", the older conversation's repeat");
        fresh = make(2048, 0);
        same(serve(*fresh, tok, 3, {{Req{next, 32}}})[0], follow_reply, what + ", the follow-up");
    }
}

// One copy per conversation in host memory: conversation A takes two turns, each reply read again, so its second job's donor supersedes its second request's donor and the first job's, beside an unrelated conversation C.
// A request needing the whole pool of 16 blocks evicts every donor, host memory keeps C's and A's second job's alone, and C's repeat and A's third turn fork 256 and 384 tokens from host memory, every reply its reply alone.
void one_copy_per_conversation(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "one copy per conversation in host memory";
    const Req c{prompt_of(3, 300, vocab), 40}, a{prompt_of(5, 300, vocab), 40}, big{prompt_of(9, 2000, vocab), 40};
    auto model = make(2048, 0);
    std::vector<uint32_t> next, c_again;
    Reply c_again_reply, third_reply;
    size_t c_reused = 0, third_reused = 0, kept = 0;
    {
        server::Scheduler sched(*model, tok, 4, 64, 0, false, (size_t)1 << 30);
        std::thread runner([&] { sched.run(); });
        try {
            drain(*sched.submit(c.prompt, params_of(c)));
            next = a.prompt;
            for (size_t turn = 1; turn <= 2; ++turn) {
                const auto h = sched.submit(next, params_of(Req{next, 40}));
                for (uint32_t id : ids_of(drain(*h))) next.push_back(id);
                next.push_back(1);
                next.push_back(2);
                sched.follow(h, next, true);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills < turn) {
                    require(std::chrono::steady_clock::now() < until, what + ": a reply was not read again in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (turn == 1) {
                    const std::vector<uint32_t> tail = prompt_of(6, 30, vocab);
                    next.insert(next.end(), tail.begin(), tail.end());
                }
            }
            drain(*sched.submit(big.prompt, params_of(big)));
            kept = sched.stats().host_donors;
            c_again = c.prompt;
            const std::vector<uint32_t> more = prompt_of(7, 30, vocab);
            c_again.insert(c_again.end(), more.begin(), more.end());
            const auto ca = sched.submit(c_again, params_of(Req{c_again, 32}));
            c_again_reply = drain(*ca);
            c_reused = ca->reused();
            const std::vector<uint32_t> tail = prompt_of(8, 30, vocab);
            next.insert(next.end(), tail.begin(), tail.end());
            const auto t = sched.submit(next, params_of(Req{next, 32}));
            third_reply = drain(*t);
            third_reused = t->reused();
            ledger(sched.stats(), *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(kept == 2, what + ": " + std::to_string(kept) + " donors in host memory once every donor was evicted, against 2");
    require(c_reused == 2 * kBlock, what + ": the older conversation's repeat reused " + std::to_string(c_reused) + " tokens, against " + std::to_string(2 * kBlock));
    require(third_reused == 3 * kBlock, what + ": the third turn reused " + std::to_string(third_reused) + " tokens, against " + std::to_string(3 * kBlock));
    auto fresh = make(2048, 0);
    same(serve(*fresh, tok, 3, {{Req{c_again, 32}}})[0], c_again_reply, what + ", the older conversation's repeat");
    fresh = make(2048, 0);
    same(serve(*fresh, tok, 3, {{Req{next, 32}}})[0], third_reply, what + ", the third turn");
}

// Host memory keeps the conversations that came back against newcomers (Scheduler::write_back): one request at a time, each finished request's donor evicting the one before it, each host copy one 64 MiB slab.
// X' repeats X and so came back, then new prompts Y, Z and W evict donors: with room for two copies a prompt repeating X' forks from host memory, with room for one it forks nothing; every reply is its reply alone.
void conversations_that_came_back(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    for (const size_t slabs : {(size_t)2, (size_t)1}) {
        const std::string what = "host memory for " + std::to_string(slabs) + " cop" + (slabs == 1 ? "y" : "ies") + " keeping conversations that came back";
        const std::vector<uint32_t> x = prompt_of(5, 300, vocab), y = prompt_of(3, 300, vocab), z = prompt_of(9, 300, vocab), w = prompt_of(11, 300, vocab);
        std::vector<uint32_t> x1 = x;
        const std::vector<uint32_t> t1 = prompt_of(6, 200, vocab);
        x1.insert(x1.end(), t1.begin(), t1.end());
        std::vector<uint32_t> x2 = x1, z1 = z;
        const std::vector<uint32_t> t2 = prompt_of(7, 30, vocab);
        x2.insert(x2.end(), t2.begin(), t2.end());
        z1.insert(z1.end(), t2.begin(), t2.end());
        const std::vector<std::vector<uint32_t>> order = slabs == 2 ? std::vector<std::vector<uint32_t>>{x, x1, y, z, w, x2} : std::vector<std::vector<uint32_t>>{x, x1, y, z, w, z1, x2};
        auto model = make(2048, 0);
        std::vector<Reply> replies;
        std::vector<size_t> reused;
        {
            server::Scheduler sched(*model, tok, 1, 64, 0, false, slabs * ((size_t)64 << 20));
            std::thread runner([&] { sched.run(); });
            try {
                for (const auto& p : order) {
                    const auto h = sched.submit(p, params_of(Req{p, 20}));
                    replies.push_back(drain(*h));
                    reused.push_back(h->reused());
                }
                ledger(sched.stats(), *model, what);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        const std::vector<size_t> want = slabs == 2 ? std::vector<size_t>{0, 2 * kBlock, 0, 0, 0, 3 * kBlock} : std::vector<size_t>{0, 2 * kBlock, 0, 0, 0, 2 * kBlock, 0};
        for (size_t i = 0; i < order.size(); ++i) {
            require(reused[i] == want[i], what + ": request " + std::to_string(i) + " reused " + std::to_string(reused[i]) + " tokens, against " + std::to_string(want[i]));
            auto fresh = make(2048, 0);
            same(serve(*fresh, tok, 1, {{Req{order[i], 20}}})[0], replies[i], what + ", request " + std::to_string(i));
        }
    }
}

// The donor count gives up the donors a job's donor supersedes: with two at most, A's job completing gives up A's request's donor, not an unrelated C's.
// So a prompt repeating C forks its 256 tokens, A's follow-up forks the job's 384, and with a host tier nothing has been copied to host memory.
void count_gives_up_superseded(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    for (const size_t host : {(size_t)0, (size_t)1 << 30}) {
        const std::string what = std::string("the donor count giving up a superseded donor") + (host ? " with a host tier" : "");
        const Req c{prompt_of(3, 300, vocab), 40}, a{prompt_of(5, 300, vocab), 100};
        auto model = make(2048, 0);
        std::vector<uint32_t> next, c_again, follow;
        Reply c_again_reply, follow_reply;
        size_t c_reused = 0, follow_reused = 0;
        server::Scheduler::Stats stats;
        {
            server::Scheduler sched(*model, tok, 2, 64, 0, false, host);
            std::thread runner([&] { sched.run(); });
            try {
                drain(*sched.submit(c.prompt, params_of(c)));
                const auto h = sched.submit(a.prompt, params_of(a));
                next = a.prompt;
                for (uint32_t id : ids_of(drain(*h))) next.push_back(id);
                next.push_back(1);
                next.push_back(2);
                sched.follow(h, next, true);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills == 0) {
                    require(std::chrono::steady_clock::now() < until, what + ": the reply was not read again in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                stats = sched.stats();
                c_again = c.prompt;
                const std::vector<uint32_t> more = prompt_of(7, 30, vocab);
                c_again.insert(c_again.end(), more.begin(), more.end());
                const auto ca = sched.submit(c_again, params_of(Req{c_again, 32}));
                c_again_reply = drain(*ca);
                c_reused = ca->reused();
                follow = next;
                const std::vector<uint32_t> tail = prompt_of(6, 30, vocab);
                follow.insert(follow.end(), tail.begin(), tail.end());
                const auto f = sched.submit(follow, params_of(Req{follow, 32}));
                follow_reply = drain(*f);
                follow_reused = f->reused();
                ledger(sched.stats(), *model, what);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        require(stats.host_donors == 0 && stats.host_bytes_moved == 0,
                what + ": " + std::to_string(stats.host_donors) + " donors in host memory and " + std::to_string(stats.host_bytes_moved) + " bytes moved once the job completed");
        require(c_reused == 2 * kBlock, what + ": the older conversation's repeat reused " + std::to_string(c_reused) + " tokens, against " + std::to_string(2 * kBlock));
        require(follow_reused == 3 * kBlock, what + ": the follow-up reused " + std::to_string(follow_reused) + " tokens, against " + std::to_string(3 * kBlock));
        auto fresh = make(2048, 0);
        same(serve(*fresh, tok, 3, {{Req{c_again, 32}}})[0], c_again_reply, what + ", the older conversation's repeat");
        fresh = make(2048, 0);
        same(serve(*fresh, tok, 3, {{Req{follow, 32}}})[0], follow_reply, what + ", the follow-up");
    }
}

// A host copy that a job's donor of the same history renews stands for that donor again: J1, A's job donor evicted to host memory, is promoted by a regenerated reply whose job supersedes J1's copy, and long requests press host memory twice.
// The renewed copy is not taken as superseded, so A's follow-up forks its 384 tokens from host memory; every reply is its reply alone.
void renewed_copy_current(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "a renewed host copy standing for its donor";
    const Req a{prompt_of(5, 300, vocab), 100};
    auto model = make(2048, 0);
    std::vector<uint32_t> next, follow;
    Reply a_reply, regen_reply, follow_reply;
    size_t regen_reused = 0, follow_reused = 0;
    {
        server::Scheduler sched(*model, tok, 4, 64, 0, false, (size_t)2 << 26);
        std::thread runner([&] { sched.run(); });
        const auto read_again = [&](size_t n) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (sched.stats().reprefills < n) {
                require(std::chrono::steady_clock::now() < until, what + ": a reply was not read again in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        };
        const auto long_request = [&](int seed) {
            const std::vector<uint32_t> p = prompt_of(seed, 2000, vocab);
            drain(*sched.submit(p, params_of(Req{p, 40})));
        };
        try {
            const auto h = sched.submit(a.prompt, params_of(a));
            a_reply = drain(*h);
            next = a.prompt;
            for (uint32_t id : ids_of(a_reply)) next.push_back(id);
            next.push_back(1);
            next.push_back(2);
            sched.follow(h, next, true);
            read_again(1);
            long_request(9);
            const auto g = sched.submit(a.prompt, params_of(a));
            regen_reply = drain(*g);
            regen_reused = g->reused();
            sched.follow(g, next, true);
            read_again(2);
            long_request(10);
            long_request(11);
            long_request(12);
            follow = next;
            const std::vector<uint32_t> tail = prompt_of(6, 30, vocab);
            follow.insert(follow.end(), tail.begin(), tail.end());
            const auto f = sched.submit(follow, params_of(Req{follow, 32}));
            follow_reply = drain(*f);
            follow_reused = f->reused();
            ledger(sched.stats(), *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    same(regen_reply, a_reply, what + ", the regenerated reply against the first");
    require(regen_reused == 2 * kBlock, what + ": the regenerated reply reused " + std::to_string(regen_reused) + " tokens, against " + std::to_string(2 * kBlock));
    require(follow_reused == 3 * kBlock, what + ": the follow-up reused " + std::to_string(follow_reused) + " tokens, against " + std::to_string(3 * kBlock));
    auto fresh = make(2048, 0);
    same(serve(*fresh, tok, 3, {{Req{follow, 32}}})[0], follow_reply, what + ", the follow-up");
}

// A promotion that fails after evicting the device donor a request would otherwise fork: with host memory for one copy, making room for a host copy evicts the second request's donor, whose copy takes the memory the first held.
// The promotion fails, and the request runs from what is left with its reply alone.
void failed_promotion(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req a{prompt_of(5, 300, vocab), 40};
    std::vector<uint32_t> fork = a.prompt;
    fork.resize(kBlock);
    const std::vector<uint32_t> tail = prompt_of(9, 200, vocab);
    fork.insert(fork.end(), tail.begin(), tail.end());
    std::vector<uint32_t> again = a.prompt;
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    const std::vector<Req> reqs = {a, Req{fork, 40}, Req{again, 32}};
    auto model = make(1024, 0);
    server::Scheduler::Stats stats;
    const std::vector<Reply> got = serve(*model, tok, 1, {{reqs[0]}, {reqs[1]}, {reqs[2]}}, &stats, 0, (size_t)64 << 20);
    for (size_t i = 0; i < reqs.size(); ++i) {
        auto fresh = make(1024, 0);
        same(serve(*fresh, tok, 1, {{reqs[i]}})[0], got[i], "a failed promotion, request " + std::to_string(i));
    }
    require(stats.host_hits == 0 && stats.host_donors == 1, "a failed promotion: " + std::to_string(stats.host_hits) + " promotions and " +
                                                            std::to_string(stats.host_donors) + " donors in host memory");
}

void hybrid(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Make one = on(weights, [] { return cpus(1); }, 3);
    const std::vector<Req> three = {{prompt_of(1, 40, vocab)}, {prompt_of(2, 9, vocab)}, {prompt_of(3, 23, vocab)}};
    auto s = alone_then_together(one, tok, 1024, 0, 3, {}, three, "a hybrid model's three uncapped requests");
    no_donors(s, "a hybrid model's three uncapped requests");
    require(s.recomputed > s.pauses * 40, "a hybrid model's three uncapped requests: no resume recomputed a generated token");
    s = alone_then_together(one, tok, 1024, 1, 3, {}, {{prompt_of(1, 4, vocab)}, {prompt_of(2, 384, vocab)}}, "a hybrid model's victim still prefilling");
    no_donors(s, "a hybrid model's victim still prefilling");
    // A follow-up turn repeating a finished request's history forks nothing, so it recomputes that history as its prompt.
    const Req first{prompt_of(5, 20, vocab), 200};
    auto model = one(1024, 0);
    std::vector<uint32_t> again = first.prompt;
    const Reply turn = serve(*model, tok, 3, {{first}})[0];
    for (const auto& t : turn) again.push_back(t.id);
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    s = alone_then_together(one, tok, 1024, 0, 3, {first}, {{prompt_of(1, 9, vocab)}, {again}}, "a hybrid model's follow-up turn");
    no_donors(s, "a hybrid model's follow-up turn");
    s = alone_then_together(on(weights, [] { return cpus(2); }, 3), tok, 1024, 0, 3, {}, three, "a hybrid model on a two-CPU split");
    no_donors(s, "a hybrid model on a two-CPU split");
    require(s.passes == 2, "a hybrid model on a two-CPU split kept " + std::to_string(s.passes) + " passes in flight, against 2");
    // Paused requests cancelled, then the slots and blocks they held taken by as many requests at once.
    model = one(1024, 0);
    {
        server::Scheduler sched(*model, tok, 3, 64);
        std::thread runner([&] { sched.run(); });
        try {
            std::vector<std::shared_ptr<server::Request>> handles;
            for (const Req& r : three) handles.push_back(sched.submit(r.prompt, params_of(r)));
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (sched.stats().pauses == 0) {
                require(std::chrono::steady_clock::now() < until, "a hybrid model's cancelled requests: nothing paused in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            for (auto& h : handles) h->cancel();
            for (auto& h : handles) collect(*h);
            ledger(sched.stats(), *model, "a hybrid model's paused requests cancelled");
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    every_slot_free(*model, tok, vocab, 3, "a hybrid model after paused requests were cancelled");
    require(model->kv_used_bytes() == 0, "a hybrid model's pool holds blocks once its schedulers have stopped");
    // A request cancelled in flight and a stop in flight, on a split of three CPUs.
    for (const bool stop : {false, true}) {
        const std::string what = stop ? "a hybrid model stopped in flight" : "a hybrid model's request cancelled in flight";
        const std::vector<std::shared_ptr<Hooked>> devices = hooked(3);
        auto split = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); }, 3)(1024, 0);
        {
            server::Scheduler sched(*split, tok, 3, 64);
            const Req v{prompt_of(1, 130, vocab), 300};
            const auto hv = sched.submit(v.prompt, params_of(v));
            size_t submits = 0;
            devices.back()->hook = [&] { if (++submits == 10) (stop ? sched.stop() : hv->cancel()); };
            std::thread runner([&] { sched.run(); });
            try {
                const Reply got = collect(*hv);
                require(hv->finish() == "cancel" && got.size() < 300, what + ": it ended with " + hv->finish() + " after " + std::to_string(got.size()) + " tokens");
                if (!stop) ledger(sched.stats(), *split, what);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
            no_donors(sched.stats(), what);
        }
        devices.back()->hook = nullptr;
        every_slot_free(*split, tok, vocab, 3, what);
        whole_pool_free(*split, 1024, vocab, what);
    }
    // A scheduler that would run more requests at once than the model holds states is refused as it is made.
    bool refused = false;
    try {
        server::Scheduler sched(*model, tok, 4, 64);
    } catch (const std::logic_error& e) {
        refused = std::string(e.what()).find("recurrent state slots") != std::string::npos;
    }
    require(refused, "a scheduler of 4 requests at once over a hybrid model holding 3 states was not refused");
}

namespace fs = std::filesystem;

// A disk tier's root for one case, removed after it, with a small file standing for the model, whose digest the entries' identity carries.
struct DiskRoot {
    fs::path root;
    std::string model;
    explicit DiskRoot(const std::string& name) {
        root = fs::temp_directory_path() / ("llmx-server-resume-" + name + "-" + std::to_string(std::random_device{}()));
        fs::remove_all(root);
        fs::create_directories(root);
        model = (root / "model.bin").u8string();
        std::ofstream(root / "model.bin", std::ios::binary) << "a model file";
    }
    ~DiskRoot() {
        std::error_code ec;
        fs::remove_all(root, ec);
    }
    server::DiskOptions options(uint64_t bytes, std::chrono::milliseconds pace = std::chrono::milliseconds(0)) const {
        server::DiskOptions o;
        o.bytes = bytes;
        o.dir = root.u8string();
        o.floor = 0;
        o.model_path = model;
        o.pace = pace;
        return o;
    }
    // The servers' directories under the root.
    std::vector<fs::path> servers() const {
        std::vector<fs::path> v;
        for (const auto& e : fs::directory_iterator(root))
            if (e.is_directory() && e.path().filename().u8string().rfind("server-", 0) == 0) v.push_back(e.path());
        return v;
    }
    // The files of extension `ext` in the servers' directories, and their bytes.
    std::vector<fs::path> files(const std::string& ext, uint64_t* bytes = nullptr) const {
        std::vector<fs::path> v;
        if (bytes) *bytes = 0;
        for (const fs::path& d : servers())
            for (const auto& e : fs::directory_iterator(d))
                if (e.path().extension().u8string() == ext) {
                    v.push_back(e.path());
                    if (bytes) *bytes += (uint64_t)fs::file_size(e.path());
                }
        return v;
    }
};

// Polls `done` for up to a minute.
void within_a_minute(const std::function<bool()>& done, const std::string& what) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!done()) {
        require(std::chrono::steady_clock::now() < until, what + " within a minute");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// Demotion to disk (docs/DISK-TIER.md): eight conversations in turn evict each donor to a host tier of four copies, the oldest being written to disk while it stays in host memory and released at once when room is needed.
// So after turn k the disk holds the k - 3 oldest, and with a disk tier of two entries the two newest; every reply is its reply alone, the files in the directory are the entries counted, and a clean exit leaves no directory.
void disk_demotion(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, size_t devices) {
    uint64_t entry_bytes = 0;
    for (const bool small : {false, true}) {
        const std::string what = "demotion to disk on " + std::to_string(devices) + " CPU" + (devices > 1 ? "s" : "") + (small ? ", a disk tier of two entries" : "");
        DiskRoot disk("demotion");
        auto model = make(2048, 0);
        std::vector<Req> reqs;
        for (uint32_t k = 0; k < 8; ++k) reqs.push_back(Req{prompt_of(20 + k, 300, vocab), 20});
        std::vector<Reply> replies;
        server::Scheduler::Stats stats;
        uint64_t file_bytes = 0;
        size_t kv_files = 0, tmp_files = 0;
        {
            const uint64_t cap = small ? 2 * entry_bytes + entry_bytes / 2 : uint64_t(1) << 30;
            server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * devices * ((size_t)64 << 20), nullptr, 0, false, disk.options(cap));
            std::thread runner([&] { sched.run(); });
            try {
                for (size_t k = 0; k < reqs.size(); ++k) {
                    replies.push_back(drain(*sched.submit(reqs[k].prompt, params_of(reqs[k]))));
                    const size_t written = k >= 4 ? k - 3 : 0, kept = small ? std::min<size_t>(written, 2) : written;
                    within_a_minute([&] { const auto st = sched.stats(); return st.disk_entries == kept && (!entry_bytes || st.disk_bytes_written == written * entry_bytes); },
                                    what + ": " + std::to_string(kept) + " entries on disk after turn " + std::to_string(k));
                    if (!entry_bytes && written) {
                        entry_bytes = sched.stats().disk_bytes;
                        require(entry_bytes > 0, what + ": an entry of no bytes");
                    }
                }
                stats = sched.stats();
                ledger(stats, *model, what);
                kv_files = disk.files(".kv", &file_bytes).size();
                tmp_files = disk.files(".tmp").size();
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        const size_t kept = small ? 2 : 4;
        require(stats.disk_entries == kept && stats.host_donors == 4 && stats.disk_writing && stats.disk_errors == 0,
                what + ": " + std::to_string(stats.disk_entries) + " entries on disk and " + std::to_string(stats.host_donors) + " in host memory, against " +
                    std::to_string(kept) + " and 4");
        // Every copy that left host memory had its file, and the small tier deleted two entries for its cap.
        require(stats.host_unwritten == 0 && stats.disk_capped == 4 - kept,
                what + ": " + std::to_string(stats.host_unwritten) + " copies dropped unwritten and " + std::to_string(stats.disk_capped) + " entries deleted for the cap, against 0 and " + std::to_string(4 - kept));
        require(stats.disk_bytes == kept * entry_bytes && stats.disk_bytes_written == 4 * entry_bytes,
                what + ": " + std::to_string(stats.disk_bytes) + " bytes on disk and " + std::to_string(stats.disk_bytes_written) + " written");
        require(kv_files == kept && file_bytes == stats.disk_bytes && tmp_files == 0,
                what + ": " + std::to_string(kv_files) + " entry files of " + std::to_string(file_bytes) + " bytes and " + std::to_string(tmp_files) + " temporary files");
        require(disk.servers().empty(), what + ": the server's directory is left after a clean exit");
        for (size_t k = 0; k < reqs.size(); ++k) {
            auto fresh = make(2048, 0);
            same(serve(*fresh, tok, 1, {{reqs[k]}})[0], replies[k], what + ", turn " + std::to_string(k));
        }
    }
}

// A history a later turn supersedes stays on disk as that turn's base (docs/DISK-TIER.md, Entries written as what changed): A's first job donor, evicted to a host tier of two copies, is written to disk.
// A's second turn promotes it and forks its 384 tokens, its read-again reply supersedes the first job's donor in host memory while the file stays, nothing else is written, and the reply is its reply on a fresh model.
void disk_superseded(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "a superseded history on disk";
    DiskRoot disk("superseded");
    auto model = make(2048, 0);
    const Req a{prompt_of(5, 300, vocab), 100}, b{prompt_of(7, 300, vocab), 20}, c{prompt_of(9, 300, vocab), 20};
    Req a2;
    Reply a2_reply;
    size_t a2_reused = 0, kv_files = 0;
    server::Scheduler::Stats before, stats;
    {
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 2 * ((size_t)64 << 20), nullptr, 0, false, disk.options(uint64_t(1) << 30));
        std::thread runner([&] { sched.run(); });
        try {
            const auto read_again = [&](const std::shared_ptr<server::Request>& h, const Req& r, size_t jobs) {
                std::vector<uint32_t> next = r.prompt;
                for (uint32_t id : ids_of(drain(*h))) next.push_back(id);
                next.push_back(1);
                next.push_back(2);
                sched.follow(h, next, true);
                within_a_minute([&] { return sched.stats().reprefills == jobs; }, what + ": reply " + std::to_string(jobs) + " read again");
                return next;
            };
            const std::vector<uint32_t> next = read_again(sched.submit(a.prompt, params_of(a)), a, 1);
            drain(*sched.submit(b.prompt, params_of(b)));
            drain(*sched.submit(c.prompt, params_of(c)));
            within_a_minute([&] { return sched.stats().disk_entries == 1; }, what + ": the first job's donor on disk");
            a2.prompt = next;
            const std::vector<uint32_t> tail = prompt_of(6, 30, vocab);
            a2.prompt.insert(a2.prompt.end(), tail.begin(), tail.end());
            a2.cap = 100;
            const auto h2 = sched.submit(a2.prompt, params_of(a2));
            a2_reply = drain(*h2);
            a2_reused = h2->reused();
            before = sched.stats();
            std::vector<uint32_t> next2 = a2.prompt;
            for (uint32_t id : ids_of(a2_reply)) next2.push_back(id);
            next2.push_back(1);
            next2.push_back(2);
            sched.follow(h2, next2, true);
            within_a_minute([&] { return sched.stats().reprefills == 2; }, what + ": the second reply read again");
            stats = sched.stats();
            ledger(stats, *model, what);
            kv_files = disk.files(".kv").size();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(a2_reused == 3 * kBlock, what + ": the second turn reused " + std::to_string(a2_reused) + " tokens, against " + std::to_string(3 * kBlock));
    require(before.host_hits == 1 && before.disk_entries == 1, what + ": " + std::to_string(before.host_hits) + " promotions and " + std::to_string(before.disk_entries) + " entries on disk, against 1 and 1");
    require(stats.disk_entries == 1 && kv_files == 1, what + ": " + std::to_string(stats.disk_entries) + " entries and " + std::to_string(kv_files) + " files on disk once A's first job's donor was superseded, against 1 and 1");
    require(stats.disk_bytes_written == before.disk_bytes, what + ": " + std::to_string(stats.disk_bytes_written) + " bytes written, against the first job's donor's " + std::to_string(before.disk_bytes));
    auto fresh = make(2048, 0);
    same(serve(*fresh, tok, 1, {{a2}})[0], a2_reply, what + ", A's second turn");
}

// Writes never hold up a request (docs/DISK-TIER.md, Demotion): with every write held for a minute a chunk, the turn after the host tier of `copies` copies fills ends in far less than the write would take.
// With four copies the next oldest copy gives the room and the write stays in flight, with one the copy being written is the only one to take, so its write is cancelled and its entry not kept; every reply is its reply alone.
void disk_never_blocks(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, size_t copies) {
    const std::string what = copies > 1 ? "a held write left in flight" : "a held write giving way";
    DiskRoot disk("held");
    auto model = make(2048, 0);
    std::vector<Req> reqs;
    for (uint32_t k = 0; k < copies + 2; ++k) reqs.push_back(Req{prompt_of(40 + k, 300, vocab), 20});
    std::vector<Reply> replies;
    server::Scheduler::Stats stats;
    double last_ms = 0;
    bool first_tmp_left = true, first_kv = true;
    {
        server::Scheduler sched(*model, tok, 1, 64, 0, false, copies * ((size_t)64 << 20), nullptr, 0, false, disk.options(uint64_t(1) << 30, std::chrono::minutes(1)));
        std::thread runner([&] { sched.run(); });
        try {
            // The store is made before the turns: made after them, on a slow runner, its write probe's temporary file was taken for the entry's, and no write was in flight when the last turn came.
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
            for (size_t k = 0; k <= copies; ++k) replies.push_back(drain(*sched.submit(reqs[k].prompt, params_of(reqs[k]))));
            std::vector<fs::path> writing;
            within_a_minute([&] { return !(writing = disk.files(".tmp")).empty(); }, what + ": the oldest copy's write in flight");
            const fs::path first = writing[0];
            require(first.filename().u8string().rfind("entry-", 0) == 0, what + ": the file in flight is " + first.filename().u8string() + ", not an entry's");
            const auto start = std::chrono::steady_clock::now();
            replies.push_back(drain(*sched.submit(reqs[copies + 1].prompt, params_of(reqs[copies + 1]))));
            last_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (copies == 1) within_a_minute([&] { return !fs::exists(first); }, what + ": the cancelled write's temporary file removed");
            first_tmp_left = fs::exists(first);
            first_kv = fs::exists(fs::path(first).replace_extension(".kv"));
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(last_ms < 20000, what + ": the turn needing room took " + std::to_string(last_ms) + " ms");
    require(first_tmp_left == (copies > 1) && !first_kv && stats.disk_entries == 0,
            what + ": the write's temporary file " + (first_tmp_left ? "is in place, " : "is gone, ") + std::to_string(stats.disk_entries) + " entries kept");
    require(stats.host_donors == copies && stats.disk_errors == 0 && stats.disk_writing,
            what + ": " + std::to_string(stats.host_donors) + " copies in host memory, against " + std::to_string(copies) + ", " + std::to_string(stats.disk_errors) +
                " disk errors");
    require(disk.servers().empty(), what + ": the server's directory is left after a clean exit");
    for (size_t k = 0; k < reqs.size(); ++k) {
        auto fresh = make(2048, 0);
        same(serve(*fresh, tok, 1, {{reqs[k]}})[0], replies[k], what + ", turn " + std::to_string(k));
    }
}

// Six conversations in turn through a host tier of four copies, each turn's writes waited for, so the first conversation's copy is on disk alone (docs/DISK-TIER.md, Restore).
// Then that conversation's follow-up, its prompt, reply and 30 tokens more, is asked for.
struct Demoted {
    std::vector<Req> turns;
    std::vector<Reply> replies;
    Req follow;
};
// `files` is how many files a copy is on disk: its blocks' segment and, on a model that keeps a state, its state.
Demoted demote(server::Scheduler& sched, uint32_t vocab, const std::string& what, size_t files = 1) {
    Demoted d;
    for (uint32_t k = 0; k < 6; ++k) d.turns.push_back(Req{prompt_of(30 + k, 300, vocab), 20});
    for (size_t k = 0; k < d.turns.size(); ++k) {
        d.replies.push_back(drain(*sched.submit(d.turns[k].prompt, params_of(d.turns[k]))));
        const size_t written = (k >= 4 ? k - 3 : 0) * files;
        within_a_minute([&] { return sched.stats().disk_entries == written; }, what + ": " + std::to_string(written) + " entries on disk after turn " + std::to_string(k));
    }
    d.follow.prompt = d.turns[0].prompt;
    for (uint32_t id : ids_of(d.replies[0])) d.follow.prompt.push_back(id);
    const std::vector<uint32_t> more = prompt_of(50, 30, vocab);
    d.follow.prompt.insert(d.follow.prompt.end(), more.begin(), more.end());
    d.follow.cap = 32;
    return d;
}

// A conversation demoted to disk and asked for again: its history is read back into host memory, promoted and forked at 256 tokens, with the reply it gives on a fresh model, on the Q8_0 model and the hybrid one.
// Its prompts read in passes of 16 rows measure no prompt rate, so the follow-up waits for its read however fast the model computes.
void disk_restore(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, size_t devices, const std::string& what) {
    DiskRoot disk("restore");
    auto model = make(2048, 16);
    Demoted d;
    Reply got;
    size_t reused = 0;
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * devices * ((size_t)64 << 20), nullptr, 0, false, disk.options(uint64_t(1) << 30));
        std::thread runner([&] { sched.run(); });
        try {
            d = demote(sched, vocab, what, model->keeps_state() ? 2 : 1);
            const auto h = sched.submit(d.follow.prompt, params_of(d.follow));
            got = drain(*h);
            reused = h->reused();
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(reused == 2 * kBlock, what + ": the follow-up reused " + std::to_string(reused) + " tokens, against " + std::to_string(2 * kBlock));
    require(stats.disk_hits == 1 && stats.disk_waits == 1 && stats.disk_bytes_read > 0 && stats.host_hits == 1,
            what + ": " + std::to_string(stats.disk_hits) + " entries read from disk for " + std::to_string(stats.disk_waits) + " waits and " + std::to_string(stats.host_hits) +
                " promotions, against 1, 1 and 1");
    auto fresh = make(2048, 16);
    same(serve(*fresh, tok, 1, {{d.follow}})[0], got, what + ", the follow-up");
}

// The rule a recording backend keeps (backends/backend.hpp, wait), on a guarded stage's devices: while a stage's thread records on a device, no other thread calls it but to wait.
// A recording is held still when armed, counting the calls other threads make while it is parked (`met`) and the scheduler's own (`own`), a letter each in `calls`: S copy into host history memory, R copy out, A allocation, s submission.
struct Recording {
    std::mutex m;
    std::condition_variable cv;
    std::thread::id scheduler, holder;
    bool armed = false, parked = false, go = false;
    size_t met = 0, own = 0;
    std::string calls;                        // the scheduler thread's calls on the stage's devices, a letter each (Guarded::seen)
    std::set<const backend::Buffer*> slabs;   // the host memory a model keeps copies of histories in
    void arm() {
        std::lock_guard<std::mutex> lk(m);
        armed = true;
        go = false;
    }
    bool wait_parked() {
        std::unique_lock<std::mutex> lk(m);
        return cv.wait_for(lk, std::chrono::seconds(60), [&] { return parked; });
    }
    void release() {
        {
            std::lock_guard<std::mutex> lk(m);
            go = true;
        }
        cv.notify_all();
    }
    size_t count(size_t Recording::* which) {
        std::lock_guard<std::mutex> lk(m);
        return this->*which;
    }
};
struct Guarded : backend::CpuBackend {
    Recording* rule = nullptr;
    bool guarded = false;
    // A stage of one backend goes to a thread of its own only where it is not the host's, so the layer split's backends say they are no CPU.
    bool device = false;
    bool is_cpu() const override { return !device; }
    void seen(char kind, const backend::Buffer* dst = nullptr, const backend::Buffer* src = nullptr) {
        if (!guarded) return;
        const std::thread::id me = std::this_thread::get_id();
        std::lock_guard<std::mutex> lk(rule->m);
        if (me == rule->scheduler) {
            ++rule->own;
            rule->calls += dst && rule->slabs.count(dst) ? 'S' : src && rule->slabs.count(src) ? 'R' : kind;
        }
        if (rule->parked && me != rule->holder) ++rule->met;
    }
    backend::Ticket submit() override {
        if (guarded) {
            std::unique_lock<std::mutex> lk(rule->m);
            const std::thread::id me = std::this_thread::get_id();
            if (rule->armed && me != rule->scheduler) {
                rule->armed = false;
                rule->parked = true;
                rule->holder = me;
                rule->cv.notify_all();
                rule->cv.wait(lk, [&] { return rule->go; });
                rule->parked = false;
            }
        }
        seen('s');
        return CpuBackend::submit();
    }
    void sync() noexcept override {
        seen('y');
        CpuBackend::sync();
    }
    double device_ms() override {
        seen('t');
        return CpuBackend::device_ms();
    }
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        const bool slab = where == backend::Memory::host_visible && bytes >= ((size_t)1 << 20);
        seen(slab ? 'A' : 'a');
        backend::BufferPtr b = CpuBackend::alloc(bytes, where);
        if (slab && guarded) {
            std::lock_guard<std::mutex> lk(rule->m);
            rule->slabs.insert(b.get());
        }
        return b;
    }
    void read(const backend::Buffer& src, size_t off, void* dst, size_t bytes) override {
        seen('r');
        CpuBackend::read(src, off, dst, bytes);
    }
    void write(backend::Buffer& dst, size_t off, const void* src, size_t bytes) override {
        seen('w', &dst);
        CpuBackend::write(dst, off, src, bytes);
    }
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        seen('c', &dst, &src);
        CpuBackend::copy(dst, dst_off, src, src_off, bytes);
    }
};

// What a scheduler's thread does on a stage's devices outside a pass's stage while that stage's recording is parked: room sending a donor to host memory, a promotion from host memory, and a read back from disk.
// `width` 2 holds it over two stages of groups, `width` 1 over a layer split of two with two requests decoding, and with a `share` that split is fed in pieces and reached only under a hold (Scheduler::hold).
void recorder_rule(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, size_t width, double share = 0) {
    const std::string what = std::string(width > 1 ? "the recorder rule over two stages of groups" : "the recorder rule over a layer split of two") + (share > 0 ? " fed in pieces" : "");
    Recording rule;
    const Make make = on(weights, [&rule, width] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 2 * width; ++i) {
            auto g = std::make_shared<Guarded>();
            g->set_threads(1);
            g->rule = &rule;
            g->guarded = i >= width;
            g->device = width == 1;
            v.push_back(g);
        }
        return v;
    }, 8, 0, 0, 0, false, width);
    const Make plain = on(weights, [width] { return cpus(2 * width); }, 8, 0, 0, 0, false, width);
    DiskRoot disk("recorder");
    auto model = make(2048, 16);
    std::vector<Req> turns;
    std::vector<Reply> replies;
    // A conversation's next turn: its prompt, its reply and 30 tokens more.
    const auto follow = [&](size_t k, uint32_t seed) {
        Req r{turns[k].prompt, 32};
        for (uint32_t id : ids_of(replies[k])) r.prompt.push_back(id);
        const std::vector<uint32_t> more = prompt_of(seed, 30, vocab);
        r.prompt.insert(r.prompt.end(), more.begin(), more.end());
        return r;
    };
    struct Step {
        std::string name;
        Req trigger, beside, pair;
        Reply got, got_beside, got_pair;
        size_t own_parked = 0, own_after = 0, met = 0;
        std::string calls;   // the scheduler thread's calls on the stage's devices once the recording was let go
        server::Scheduler::Stats before, after;
    };
    std::vector<Step> steps(4);
    Step first_read;
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    // No pass retires while a step's requests are queued, so the first cannot run to its end before the second is in.
    std::mutex submitting;
    // One step: a request beside parks a recording, the trigger is submitted and given the hold, and the recording is let go.
    const auto hold = [&](server::Scheduler& sched, Step& s) {
        // The step before it has its copies written, so which tier holds a conversation does not turn on a write's timing.
        within_a_minute([&] { return !sched.stats().disk_in_flight; }, what + ": the writes before " + s.name);
        s.before = sched.stats();
        rule.arm();
        // On the layer split a second request decodes beside the first, each in a pass of its own, both under one block, which leaves no donor.
        std::shared_ptr<server::Request> beside, pair;
        {
            std::lock_guard<std::mutex> lock(submitting);
            beside = sched.submit(s.beside.prompt, params_of(s.beside));
            if (width == 1) pair = sched.submit(s.pair.prompt, params_of(s.pair));
        }
        if (!rule.wait_parked()) {
            const auto st = sched.stats();
            require(false, what + ", " + s.name + ": no recording parked within a minute; active " + std::to_string(st.active) + ", queued " + std::to_string(st.queued) + ", passes " + std::to_string(st.passes) + ", in flight " + std::to_string(st.in_flight) + ", disk in flight " + std::to_string(st.disk_in_flight));
        }
        const size_t own0 = rule.count(&Recording::own);
        {
            std::lock_guard<std::mutex> lk(rule.m);
            rule.calls.clear();
        }
        const auto h = sched.submit(s.trigger.prompt, params_of(s.trigger));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        s.own_parked = rule.count(&Recording::own) - own0;
        s.met = rule.count(&Recording::met);
        rule.release();
        s.got = drain(*h);
        s.got_beside = drain(*beside);
        if (pair) s.got_pair = drain(*pair);
        s.own_after = rule.count(&Recording::own) - own0 - s.own_parked;
        s.after = sched.stats();
        {
            std::lock_guard<std::mutex> lk(rule.m);
            s.calls = rule.calls;
        }
    };
    // The requests beside a step: one over groups, and on the layer split two, which decode beside each other.
    const auto company = [&](Step& s, uint32_t seed) {
        s.beside = Req{prompt_of(seed, 20, vocab), width > 1 ? 8 : 60};
        if (width == 1) s.pair = Req{prompt_of(10 + seed, 20, vocab), 60};
    };
    {
        server::Scheduler sched(*model, tok, width > 1 ? 2 : 3, 64, width > 1 ? 2 : 3, false, 2 * 2 * width * ((size_t)64 << 20), nullptr, 0, false, options, share);
        sched.on_retire = [&](const server::Scheduler::Retired&) { std::lock_guard<std::mutex> lock(submitting); };
        std::thread runner([&] {
            {
                std::lock_guard<std::mutex> lk(rule.m);
                rule.scheduler = std::this_thread::get_id();
            }
            sched.run();
        });
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
            // Conversations in turn, each turn's writes waited for, until the oldest copy has left host memory for disk.
            for (uint32_t k = 0; k < (width > 1 ? 6u : 7u); ++k) {
                turns.push_back(Req{prompt_of(30 + k, 300, vocab), 20});
                replies.push_back(drain(*sched.submit(turns[k].prompt, params_of(turns[k]))));
                within_a_minute([&] { return !sched.stats().disk_in_flight; }, what + ": the writes after turn " + std::to_string(k));
            }
            require(sched.stats().disk_entries >= 2 && sched.stats().host_donors == 2, what + ": " + std::to_string(sched.stats().disk_entries) + " entries on disk and " + std::to_string(sched.stats().host_donors) + " copies in host memory after " + std::to_string(turns.size()) + " conversations");
            steps[0].name = "a follow-up whose room sends a donor to host memory";
            steps[0].trigger = follow(3, 51);
            steps[1].name = "a new conversation";
            steps[1].trigger = Req{prompt_of(60, 300, vocab), 20};
            steps[2].name = "a conversation promoted from host memory";
            steps[2].trigger = follow(turns.size() - 1, 52);
            steps[3].name = "a conversation read back from disk and promoted";
            steps[3].trigger = follow(0, 50);
            for (size_t k = 0; k < steps.size(); ++k) {
                company(steps[k], 1 + (uint32_t)k);
                hold(sched, steps[k]);
            }
        } catch (...) {
            rule.release();
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    // A read that has to allocate: a second scheduler with a host tier smaller than an entry adopts the entries, and a conversation read back goes through host memory beyond the tier, taking new slabs, a call on each device.
    {
        auto again = make(2048, 16);
        {
            std::lock_guard<std::mutex> lk(rule.m);
            rule.slabs.clear();
        }
        server::Scheduler sched(*again, tok, width > 1 ? 2 : 3, 64, width > 1 ? 2 : 3, false, (size_t)1 << 16, nullptr, 0, false, options, share);
        sched.on_retire = [&](const server::Scheduler::Retired&) { std::lock_guard<std::mutex> lock(submitting); };
        std::thread runner([&] {
            {
                std::lock_guard<std::mutex> lk(rule.m);
                rule.scheduler = std::this_thread::get_id();
            }
            sched.run();
        });
        try {
            within_a_minute([&] { return sched.stats().disk_entries >= 2; }, what + ": the entries adopted after a restart");
            first_read.name = "a conversation read back through host memory beyond the tier";
            first_read.trigger = follow(1, 53);
            company(first_read, 9);
            hold(sched, first_read);
        } catch (...) {
            rule.release();
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    steps.push_back(first_read);
    for (const Step& s : steps) {
        require(s.met == 0, what + ", " + s.name + ": " + std::to_string(s.met) + " calls reached the stage's devices from another thread while its recording was parked");
        require(s.own_parked == 0, what + ", " + s.name + ": the scheduler's thread made " + std::to_string(s.own_parked) + " calls on the stage's devices while its recording was parked");
        require(s.own_after > 0 || &s == &steps[1], what + ", " + s.name + ": the scheduler's thread made no call on the stage's devices once the recording was let go, so the case did not reach the call it holds");
        auto fresh = plain(2048, 16);
        same(serve(*fresh, tok, 1, {{s.trigger}})[0], s.got, what + ", " + s.name);
        auto fresh_beside = plain(2048, 16);
        same(serve(*fresh_beside, tok, 1, {{s.beside}})[0], s.got_beside, what + ", the request beside " + s.name);
        if (width > 1) continue;
        auto fresh_pair = plain(2048, 16);
        same(serve(*fresh_pair, tok, 1, {{s.pair}})[0], s.got_pair, what + ", the second request beside " + s.name);
    }
    // What each step held is the first call its path makes on the stage's devices: a copy into host memory, and a copy out of it.
    require(steps[0].calls[0] == 'S' && steps[0].after.host_bytes_moved > steps[0].before.host_bytes_moved, what + ", " + steps[0].name + ": its first call there was not a donor's copy to host memory (" + steps[0].calls + ")");
    require((steps[2].calls[0] == 'R' || (width == 1 && steps[2].calls[0] == 'S' && steps[2].calls.find('R') != std::string::npos)) && steps[2].after.host_hits == steps[2].before.host_hits + 1 && steps[2].after.disk_hits == steps[2].before.disk_hits,
            what + ", " + steps[2].name + ": its first call there was not the copy back from host memory, or it was read from disk (" + steps[2].calls + ", " +
                std::to_string(steps[2].after.host_hits - steps[2].before.host_hits) + " promotions, " + std::to_string(steps[2].after.disk_hits - steps[2].before.disk_hits) + " reads from disk)");
    require(steps[3].after.disk_hits == steps[3].before.disk_hits + 1 && steps[3].after.host_hits == steps[3].before.host_hits + 1, what + ", " + steps[3].name + ": it was not read back from disk and promoted (" + steps[3].calls + ")");
    require(steps[4].calls[0] == 'A' && steps[4].after.disk_hits == steps[4].before.disk_hits + 1,
            what + ", " + steps[4].name + ": its first call there was not the allocation of the host memory it is read into, or it was not read back from disk (" + steps[4].calls + ")");
    std::cout << "server-resume: " << what << ": a donor's copy to host memory, a promotion from it, a conversation read back from disk and one read back through host memory beyond the tier, which allocates it, each waited for a parked recording\n";
}

// A request waiting for its read keeps its place (docs/DISK-TIER.md, Restore): with every read held two seconds a chunk, a follow-up on disk waits while a later unrelated request runs to its end, then forks the copy read back.
// A follow-up whose read fails its checksum, a byte flipped, computes its history, the entry gone and the failure counted; every reply is its reply alone.
void disk_read_waits(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    for (const bool corrupt : {false, true}) {
        const std::string what = corrupt ? "a read failing its checksum" : "a request waiting for its read";
        DiskRoot disk(corrupt ? "corrupt" : "wait");
        auto model = make(2048, 16);
        Demoted d;
        const Req other{prompt_of(55, 300, vocab), 20};
        Reply got, other_got;
        size_t reused = 0;
        double other_ms = 0, follow_ms = 0;
        server::Scheduler::Stats stats;
        {
            server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false,
                                    disk.options(uint64_t(1) << 30, corrupt ? std::chrono::milliseconds(0) : std::chrono::milliseconds(2000)));
            std::thread runner([&] { sched.run(); });
            try {
                d = demote(sched, vocab, what);
                if (corrupt) {
                    // The oldest entry is the first conversation's: its payload's first byte, past the header's first mebibyte at most, flipped in place.
                    std::vector<fs::path> files = disk.files(".kv");
                    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
                        return std::stoull(a.stem().u8string().substr(6)) < std::stoull(b.stem().u8string().substr(6));
                    });
                    std::fstream f(files.at(0), std::ios::in | std::ios::out | std::ios::binary);
                    f.seekg(-(std::streamoff)server::DiskStore::kAlign, std::ios::end);
                    const char byte = (char)f.peek();
                    f.seekp(-(std::streamoff)server::DiskStore::kAlign, std::ios::end);
                    f.put((char)(byte ^ 0x5a));
                }
                const auto start = std::chrono::steady_clock::now();
                const auto h = sched.submit(d.follow.prompt, params_of(d.follow));
                const auto o = sched.submit(other.prompt, params_of(other));
                other_got = drain(*o);
                other_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                got = drain(*h);
                follow_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                reused = h->reused();
                stats = sched.stats();
                ledger(stats, *model, what);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        if (corrupt) {
            require(reused == 0 && stats.disk_hits == 0 && stats.disk_errors == 1,
                    what + ": the follow-up reused " + std::to_string(reused) + " tokens, " + std::to_string(stats.disk_hits) + " entries read, " +
                        std::to_string(stats.disk_errors) + " errors");
        } else {
            require(other_ms < 1500 && follow_ms >= 1900, what + ": the later request ended after " + std::to_string(other_ms) + " ms and the follow-up after " + std::to_string(follow_ms));
            require(reused == 2 * kBlock && stats.disk_hits == 1 && stats.disk_waits == 1 && stats.disk_wait_ms >= 1900,
                    what + ": the follow-up reused " + std::to_string(reused) + " tokens after " + std::to_string(stats.disk_wait_ms) + " ms waiting");
        }
        auto fresh = make(2048, 16);
        same(serve(*fresh, tok, 1, {{d.follow}})[0], got, what + ", the follow-up");
        fresh = make(2048, 16);
        same(serve(*fresh, tok, 1, {{other}})[0], other_got, what + ", the later request");
    }
}

// A read past its bound (docs/DISK-TIER.md, Restore): prompts read in passes of the default size measure the prompt rate, by which recomputing the follow-up's 256 tokens takes far less than a read held two seconds a chunk.
// So the follow-up is admitted without the read, computes its history and ends before the read could, with its reply alone.
void disk_read_bound(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "a read past its bound";
    DiskRoot disk("bound");
    auto model = make(2048, 0);
    Demoted d;
    Reply got;
    size_t reused = 0;
    double ms = 0;
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, disk.options(uint64_t(1) << 30, std::chrono::milliseconds(2000)));
        std::thread runner([&] { sched.run(); });
        try {
            d = demote(sched, vocab, what);
            const auto start = std::chrono::steady_clock::now();
            const auto h = sched.submit(d.follow.prompt, params_of(d.follow));
            got = drain(*h);
            ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            reused = h->reused();
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(reused == 0 && ms < 1900 && stats.disk_waits == 1,
            what + ": the follow-up reused " + std::to_string(reused) + " tokens in " + std::to_string(ms) + " ms after " + std::to_string(stats.disk_waits) + " waits");
    auto fresh = make(2048, 0);
    same(serve(*fresh, tok, 1, {{d.follow}})[0], got, what + ", the follow-up");
}

// Entries kept across a restart (docs/DISK-TIER.md, Keeping entries across a restart): under --disk-cache-keep three conversations leave three entry files at the stop, which a second scheduler on a fresh model adopts and forks from.
// Variants: `restart_host` bytes for the second scheduler's host tier, `first_host` for the first's, a stop before the idle moment, a flush held to its 20 second bound, an entry past the age limit, and repeated idle periods.
void disk_flush(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    for (const int form : {0, 1, 2}) {
        const std::string what = form == 0 ? "a stop before the idle moment" : form == 1 ? "a flush that reaches its bound" : "a second idle period";
        DiskRoot disk("flush");
        server::DiskOptions options = disk.options(uint64_t(1) << 30, std::chrono::milliseconds(form == 1 ? 60000 : 0));
        options.keep = true;
        auto model = make(2048, 16);
        double stop_s = 0;
        {
            server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
            std::thread runner([&] { sched.run(); });
            try {
                within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
                for (uint32_t k = 0; k < (form == 0 ? 3u : form == 1 ? 1u : 2u); ++k) {
                    const Req r{prompt_of(40 + k, 300, vocab), 20};
                    drain(*sched.submit(r.prompt, params_of(r)));
                    if (form == 2) within_a_minute([&] { return sched.stats().disk_entries == k + 1 && !sched.stats().disk_in_flight; }, what + ": conversation " + std::to_string(k) + " written while idle");
                }
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            const auto start = std::chrono::steady_clock::now();
            sched.stop();
            runner.join();
            stop_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        // What an entry's description holds at this version of the store's layout (Scheduler::disk_blob), read out of the three files: a change to it without a new DiskStore::kVersion moves these digests.
        // So does a change of behaviour with the layout unchanged, which turn's copy came back or how a prompt's rows are classed, and then the digests are taken again and the version stays.
        if (form == 0) {
            std::vector<std::string> blobs;
            for (const fs::path& file : disk.files(".kv")) {
                std::ifstream in(file, std::ios::binary);
                std::string head((size_t)1 << 20, 0);
                in.read(head.data(), (std::streamsize)head.size());
                uint64_t blob = 0, chunks = 0;
                uint32_t runs = 0;
                std::memcpy(&blob, head.data() + 52, 8);
                std::memcpy(&runs, head.data() + 60, 4);
                std::memcpy(&chunks, head.data() + 64 + 8 * (size_t)runs, 8);
                core::Sha sha(true);
                sha.update(head.data() + 72 + 8 * (size_t)runs + 4 * (size_t)chunks, (size_t)blob);
                blobs.push_back(sha.hex());
            }
            std::sort(blobs.begin(), blobs.end());
            const std::vector<std::string> golden{"1ea3a633430894b6d66b569b8ed56fea7bff1e474327716404c0f72cd115e35b", "4ed81cad3a8d1968469bad680e7a6cc3715157b8b591555fbb1ed69ceecb88c4", "53e3a59594258645edf48056a2abcc07be5945d6414d57cb29c62ca664c98096"};
            std::string got;
            for (const std::string& b : blobs) got += " " + b;
            require(blobs == golden, what + ": the entries' descriptions are" + got + ", not those of layout version 2: bump DiskStore::kVersion if DiskIndex::describe changed, or take the digests again if what the turns leave did");
        }
        const size_t files = disk.files(".kv").size(), want = form == 0 ? 3 : form == 1 ? 0 : 2;
        require(files == want && disk.files(".tmp").empty(), what + ": " + std::to_string(files) + " entry files and " + std::to_string(disk.files(".tmp").size()) + " temporary ones left, against " + std::to_string(want) + " and 0");
        if (form == 1) require(stop_s >= 19 && stop_s < 45, what + ": the stop took " + std::to_string(stop_s) + " s, against the 20 s bound");
    }
}

// Disk writes write what changed (docs/DISK-TIER.md, Entries written as what changed): one conversation with replies read again, on `make`'s model, under --disk-cache-keep, writes after each of four turns what it added and nothing twice.
// The bytes written are the bytes on disk and the stop writes nothing more, and a second server adopts the files, so a follow-up forks the whole conversation and an edit of its third message forks below it.
void disk_increment(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what) {
    DiskRoot disk("increment");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    options.idle = std::chrono::seconds(1);   // the idle writes start a second after a turn, to keep the case short
    const auto on_disk = [&] {
        uint64_t bytes = 0;
        disk.files(".kv", &bytes);
        return bytes;
    };
    std::vector<std::vector<uint32_t>> prompts;
    std::vector<uint32_t> prompt = prompt_of(49, 1200, vocab);
    size_t files_before = 0;
    {
        auto model = make(4096, 16);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
            const size_t more[4] = {0, 50, 600, 40};
            for (size_t turn = 0; turn < 4; ++turn) {
                if (more[turn]) {
                    const std::vector<uint32_t> tail = prompt_of(50 + (uint32_t)turn, more[turn], vocab);
                    prompt.insert(prompt.end(), tail.begin(), tail.end());
                }
                prompts.push_back(prompt);
                const uint64_t before = sched.stats().disk_bytes_written;
                const auto h = sched.submit(prompt, params_of(Req{prompt, 20}));
                for (uint32_t id : ids_of(drain(*h))) prompt.push_back(id);
                prompt.push_back(1);
                prompt.push_back(2);
                sched.follow(h, prompt, true);
                within_a_minute([&] { return sched.stats().reprefills == turn + 1; }, what + ": reply " + std::to_string(turn + 1) + " read again");
                // The idle writes begin a second after the turn and end when nothing more is written for two.
                uint64_t seen = 0;
                auto still = std::chrono::steady_clock::now();
                within_a_minute([&] {
                    const auto stats = sched.stats();
                    if (stats.disk_bytes_written != seen || stats.disk_in_flight) {
                        seen = stats.disk_bytes_written;
                        still = std::chrono::steady_clock::now();
                    }
                    return seen > before && std::chrono::steady_clock::now() - still > std::chrono::seconds(2);
                }, what + ": turn " + std::to_string(turn + 1) + " written while idle");
                require(seen == on_disk(), what + ": after turn " + std::to_string(turn + 1) + " " + std::to_string(seen) + " bytes were written and " + std::to_string(on_disk()) +
                                               " are on disk, so something was written twice or replaced");
                const size_t files = disk.files(".kv").size();
                require(model->keeps_state() || (turn != 1 && turn != 3) || files == files_before + 1,
                        what + ": a turn of a block wrote " + std::to_string(files - files_before) + " files, against the one segment it added");
                files_before = files;
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        const uint64_t before_stop = on_disk();
        sched.stop();
        runner.join();
        require(on_disk() == before_stop, what + ": the stop wrote " + std::to_string(on_disk() - before_stop) + " bytes after the idle writes");
    }
    // The follow-up goes on from the whole conversation; the edit replaces the third message and what followed it.
    Req follow{prompt, 24}, edit{prompts[2], 24};
    const std::vector<uint32_t> tail = prompt_of(47, 30, vocab);
    follow.prompt.insert(follow.prompt.end(), tail.begin(), tail.end());
    edit.prompt.resize(prompts[1].size() + 22);
    edit.prompt.insert(edit.prompt.end(), tail.begin(), tail.end());
    Reply follow_got, edit_got;
    size_t follow_reused = 0, edit_reused = 0;
    {
        auto model = make(4096, 16);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_entries > 0; }, what + ": the files adopted");
            const auto e = sched.submit(edit.prompt, params_of(edit));
            edit_got = drain(*e);
            edit_reused = e->reused();
            const auto f = sched.submit(follow.prompt, params_of(follow));
            follow_got = drain(*f);
            follow_reused = f->reused();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    // On a model that keeps a state a request forks where a state was kept: the conversation's last, and the one where the third message starts.
    const bool state = make(4096, 16)->keeps_state();
    const size_t whole = prompt.size() / kBlock * kBlock, third = (prompts[1].size() + 22) / kBlock * kBlock;
    require(follow_reused == whole, what + ": the follow-up after the restart reused " + std::to_string(follow_reused) + " tokens, against " + std::to_string(whole));
    require(state ? edit_reused == third : edit_reused >= third, what + ": the edit of the third message reused " + std::to_string(edit_reused) + " tokens, against " + std::to_string(third));
    auto fresh = make(4096, 16);
    same(serve(*fresh, tok, 3, {{follow}})[0], follow_got, what + ", the follow-up");
    fresh = make(4096, 16);
    same(serve(*fresh, tok, 3, {{edit}})[0], edit_got, what + ", the edit");
}

// What a file of a kept directory is, read from its header as a start reads it: a segment's range or a state's position.
struct OnDisk {
    fs::path path;
    bool state = false;
    size_t first = 0, end = 0;
};
std::vector<OnDisk> described(const DiskRoot& disk) {
    std::vector<OnDisk> out;
    for (const fs::path& file : disk.files(".kv")) {
        std::ifstream in(file, std::ios::binary);
        std::string head(size_t(1) << 20, 0);
        in.read(head.data(), (std::streamsize)head.size());
        uint64_t blob = 0, chunks = 0;
        uint32_t runs = 0;
        std::memcpy(&blob, head.data() + 52, 8);
        std::memcpy(&runs, head.data() + 60, 4);
        std::memcpy(&chunks, head.data() + 64 + 8 * (size_t)runs, 8);
        const size_t from = 72 + 8 * (size_t)runs + 4 * (size_t)chunks;
        server::DiskIndex::File f;
        require(in.gcount() == (std::streamsize)head.size() && from + blob <= head.size() && server::DiskIndex::parse(head.substr(from, (size_t)blob), f), "a kept file's description did not read: " + file.u8string());
        out.push_back(OnDisk{file, f.state, f.first, f.end});
    }
    std::sort(out.begin(), out.end(), [](const OnDisk& a, const OnDisk& b) { return a.state != b.state ? !a.state : a.end < b.end; });
    return out;
}

// Files lost or damaged between two servers (docs/DISK-TIER.md, Crash safety, case by case), on `make`'s model: a three-turn conversation's directory is copied with one fault in it for each case.
// The start must keep exactly the files a whole path from an empty history still reaches, and the next turn must fork what they give with the reply of a fresh model, for a lost or damaged segment or state or a leftover temporary file.
void disk_faults(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, const std::string& what) {
    DiskRoot kept("faults");
    const auto options_of = [](const DiskRoot& d) {
        server::DiskOptions o = d.options(uint64_t(1) << 30);
        o.keep = true;
        o.idle = std::chrono::seconds(1);
        return o;
    };
    std::vector<uint32_t> prompt = prompt_of(41, 1200, vocab);
    {
        auto model = make(4096, 16);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options_of(kept));
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
            const size_t more[3] = {0, 200, 500};
            for (size_t turn = 0; turn < 3; ++turn) {
                if (more[turn]) {
                    const std::vector<uint32_t> tail = prompt_of(42 + (uint32_t)turn, more[turn], vocab);
                    prompt.insert(prompt.end(), tail.begin(), tail.end());
                }
                const uint64_t before = sched.stats().disk_bytes_written;
                const auto h = sched.submit(prompt, params_of(Req{prompt, 20}));
                for (uint32_t id : ids_of(drain(*h))) prompt.push_back(id);
                prompt.push_back(1);
                prompt.push_back(2);
                sched.follow(h, prompt, true);
                within_a_minute([&] { return sched.stats().reprefills == turn + 1; }, what + ": reply " + std::to_string(turn + 1) + " read again");
                uint64_t seen = 0;
                auto still = std::chrono::steady_clock::now();
                within_a_minute([&] {
                    const auto stats = sched.stats();
                    if (stats.disk_bytes_written != seen || stats.disk_in_flight) {
                        seen = stats.disk_bytes_written;
                        still = std::chrono::steady_clock::now();
                    }
                    return seen > before && std::chrono::steady_clock::now() - still > std::chrono::seconds(2);
                }, what + ": turn " + std::to_string(turn + 1) + " written while idle");
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    const bool state = make(4096, 16)->keeps_state();
    const std::vector<OnDisk> whole = described(kept);
    size_t segments = 0;
    for (const OnDisk& f : whole) segments += !f.state;
    require(segments >= 3 && whole[0].first == 0 && whole[0].end == 1024 && (!state || whole.size() >= segments + 2),
            what + ": the conversation was kept as " + std::to_string(segments) + " segments and " + std::to_string(whole.size() - segments) + " states, too few for the faults");
    Req follow{prompt, 24};
    const std::vector<uint32_t> tail = prompt_of(40, 30, vocab);
    follow.prompt.insert(follow.prompt.end(), tail.begin(), tail.end());
    auto fresh = make(4096, 16);
    const Reply alone = serve(*fresh, tok, 3, {{follow}})[0];

    // What a start keeps of the files `left`, and what the next turn forks of them: the segments from 0 end to end, the states on them, and the highest of those or, without states, the segments' end.
    // A server that loses a segment while it runs (`running`) keeps the states above it until its next start: the history is computed again and its blocks written again under them.
    struct Usable {
        size_t files = 0, states = 0, reuse = 0;
    };
    const auto usable = [&](const std::vector<OnDisk>& left, bool running) {
        Usable u;
        size_t cover = 0;
        for (bool more = true; more;) {
            more = false;
            for (const OnDisk& f : left)
                if (!f.state && f.first == cover && f.end > cover) {
                    cover = f.end;
                    ++u.files;
                    more = true;
                }
        }
        for (const OnDisk& f : left)
            if (f.state && (running || f.end <= cover)) {
                ++u.files;
                ++u.states;
                u.reuse = std::max(u.reuse, f.end);
            }
        if (!state) u.reuse = cover;
        return u;
    };
    // One case: `fault` damages a copy of the kept directory; `late` says the start cannot see it, so every file is adopted and the read finds it.
    const auto run = [&](const std::string& name, const std::function<void(const std::vector<OnDisk>&, std::vector<OnDisk>&)>& fault, bool late = false) {
        const std::string at = what + ", " + name;
        DiskRoot disk("fault");
        fs::copy(kept.root, disk.root, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        const std::vector<OnDisk> files = described(disk);
        std::vector<OnDisk> left = files;
        fault(files, left);
        const Usable u = usable(left, late);
        auto model = make(4096, 16);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options_of(disk));
        std::thread runner([&] { sched.run(); });
        Reply got;
        size_t reused = 0;
        server::Scheduler::Stats after;
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, at + ": the disk tier made");
            const auto f = sched.submit(follow.prompt, params_of(follow));
            got = drain(*f);
            reused = f->reused();
            after = sched.stats();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        require(disk.files(".tmp").empty(), at + ": a temporary file is left");
        require(after.disk_entries == u.files && after.disk_states == u.states, at + ": " + std::to_string(after.disk_entries) + " files are kept, " + std::to_string(after.disk_states) + " of them states, against the " + std::to_string(u.files) + " and " + std::to_string(u.states) + " a whole path reaches");
        require(late ? after.disk_errors == 1 : (after.disk_errors == 0 && reused == u.reuse),
                at + ": the next turn reused " + std::to_string(reused) + " tokens with " + std::to_string(after.disk_errors) + " errors, against " + std::to_string(u.reuse));
        same(alone, got, at);
    };
    const auto lose = [](std::vector<OnDisk>& left, const fs::path& file) {
        fs::remove(file);
        left.erase(std::remove_if(left.begin(), left.end(), [&](const OnDisk& f) { return f.path == file; }), left.end());
    };
    run("a segment lost from the middle of the path", [&](const std::vector<OnDisk>& files, std::vector<OnDisk>& left) { lose(left, files[1].path); });
    run("the first segment lost", [&](const std::vector<OnDisk>& files, std::vector<OnDisk>& left) { lose(left, files[0].path); });
    run("a temporary file a crash left", [&](const std::vector<OnDisk>& files, std::vector<OnDisk>&) { std::ofstream(files[0].path.parent_path() / "entry-9999.tmp", std::ios::binary) << "half a file"; });
    run("a byte flipped in the second segment", [&](const std::vector<OnDisk>& files, std::vector<OnDisk>& left) {
        std::fstream f(files[1].path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekg((std::streamoff)(size_t(1) << 20) + 100);
        char c = 0;
        f.get(c);
        f.seekp((std::streamoff)(size_t(1) << 20) + 100);
        f.put((char)(c ^ 0x40));
        // The read finds it: the file goes, and the segments above it with it.
        left.erase(std::remove_if(left.begin(), left.end(), [&](const OnDisk& x) { return !x.state && x.first >= files[1].first; }), left.end());
    }, true);
    if (!state) return;
    run("the newest state lost", [&](const std::vector<OnDisk>& files, std::vector<OnDisk>& left) { lose(left, files.back().path); });
    run("every state lost", [&](const std::vector<OnDisk>& files, std::vector<OnDisk>& left) {
        for (const OnDisk& f : files)
            if (f.state) lose(left, f.path);
    });
}

// The cap never takes a file another stands on (docs/DISK-TIER.md, Room, the cap and the age limit): with room for two segments only, a second turn's segment finds only files it would stand on, so it is not written and nothing is deleted.
// The two files stay through four seconds, the idle writes and the stop, and a second scheduler forks their 1152 tokens for the next turn with the reply of a fresh model.
void disk_cap_base(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "the cap and a conversation's base";
    DiskRoot disk("capbase");
    server::DiskOptions options = disk.options(5 << 20);
    options.keep = true;
    options.idle = std::chrono::seconds(1);
    std::vector<uint32_t> prompt = prompt_of(45, 1200, vocab);
    {
        auto model = make(4096, 16);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
            for (size_t turn = 0; turn < 2; ++turn) {
                if (turn) {
                    const std::vector<uint32_t> tail = prompt_of(46, 200, vocab);
                    prompt.insert(prompt.end(), tail.begin(), tail.end());
                }
                const auto h = sched.submit(prompt, params_of(Req{prompt, 20}));
                for (uint32_t id : ids_of(drain(*h))) prompt.push_back(id);
                prompt.push_back(1);
                prompt.push_back(2);
                sched.follow(h, prompt, true);
                within_a_minute([&] { return sched.stats().reprefills == turn + 1; }, what + ": reply " + std::to_string(turn + 1) + " read again");
                if (!turn) within_a_minute([&] { return sched.stats().disk_entries == 2 && !sched.stats().disk_in_flight; }, what + ": the first turn's two segments written while idle");
                else std::this_thread::sleep_for(std::chrono::seconds(4));
                const auto stats = sched.stats();
                require(stats.disk_entries == 2 && stats.disk_capped == 0 && stats.disk_errors == 0 && stats.disk_writing,
                        what + ": after turn " + std::to_string(turn + 1) + " " + std::to_string(stats.disk_entries) + " files are on disk and " + std::to_string(stats.disk_capped) + " were deleted for the cap, against the first turn's two and none");
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    const std::vector<OnDisk> files = described(disk);
    require(files.size() == 2 && files[0].first == 0 && files[0].end == 1024 && files[1].first == 1024 && files[1].end == 1152, what + ": the stop did not leave the first turn's two segments");
    Req follow{prompt, 24};
    const std::vector<uint32_t> tail = prompt_of(40, 30, vocab);
    follow.prompt.insert(follow.prompt.end(), tail.begin(), tail.end());
    options.bytes = uint64_t(1) << 30;
    Reply got;
    size_t reused = 0;
    {
        auto model = make(4096, 16);
        server::Scheduler sched(*model, tok, 3, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the second disk tier made");
            const auto f = sched.submit(follow.prompt, params_of(follow));
            got = drain(*f);
            reused = f->reused();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(reused == 1152, what + ": the next turn after the restart reused " + std::to_string(reused) + " tokens, against the 1152 on disk");
    auto fresh = make(4096, 16);
    same(serve(*fresh, tok, 3, {{follow}})[0], got, what);
}

void disk_kept(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab, size_t restart_host = 4 * ((size_t)64 << 20), size_t first_host = 4 * ((size_t)64 << 20)) {
    const bool through = restart_host < ((size_t)64 << 20), beyond = first_host < ((size_t)64 << 20);
    const std::string what = beyond ? "entries written from a host tier smaller than an entry" : through ? "entries kept across a restart into a host tier smaller than an entry" : "entries kept across a restart";
    DiskRoot disk("kept");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    const size_t kept = beyond ? 1 : 3;
    std::vector<Req> turns;
    std::vector<Reply> replies;
    for (uint32_t k = 0; k < 3; ++k) turns.push_back(Req{prompt_of(10 + k, 300, vocab), 20});
    server::Scheduler::Stats first;
    {
        auto model = make(2048, 16);
        server::Scheduler sched(*model, tok, 1, 64, 0, false, first_host, nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            for (const Req& r : turns) replies.push_back(drain(*sched.submit(r.prompt, params_of(r))));
            within_a_minute([&] { return sched.stats().disk_entries == kept && !sched.stats().disk_in_flight; }, what + ": the idle server's entries written");
            first = sched.stats();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(first.disk_entries == kept && (!beyond || first.host_bytes <= first_host),
            what + ": " + std::to_string(first.disk_entries) + " entries on disk before the stop, " + std::to_string(first.host_bytes) + " bytes in host memory");
    const std::vector<fs::path> servers = disk.servers();
    require(servers.size() == 1 && fs::exists(servers[0] / "kept") && disk.files(".kv").size() == kept,
            what + ": " + std::to_string(servers.size()) + " directories and " + std::to_string(disk.files(".kv").size()) + " entry files left, against 1 and " + std::to_string(kept));
    if (beyond) return;
    std::vector<Req> follows;
    for (const size_t k : {(size_t)0, (size_t)2}) {
        Req f{turns[k].prompt, 32};
        for (uint32_t id : ids_of(replies[k])) f.prompt.push_back(id);
        const std::vector<uint32_t> more = prompt_of(50, 30, vocab);
        f.prompt.insert(f.prompt.end(), more.begin(), more.end());
        follows.push_back(f);
    }
    std::vector<Reply> got;
    std::vector<size_t> reused;
    server::Scheduler::Stats second;
    size_t slabs = 0;
    {
        auto model = make(2048, 16);
        server::Scheduler sched(*model, tok, 1, 64, 0, false, restart_host, nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_entries == 3; }, what + ": three entries adopted");
            for (const Req& f : follows) {
                const auto h = sched.submit(f.prompt, params_of(f));
                got.push_back(drain(*h));
                reused.push_back(h->reused());
            }
            second = sched.stats();
            ledger(second, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        slabs = model->host_allocated();
    }
    require(second.disk_hits == 2 && reused == std::vector<size_t>{2 * kBlock, 2 * kBlock},
            what + ": " + std::to_string(second.disk_hits) + " entries read back, the follow-ups reusing " + std::to_string(reused[0]) + " and " + std::to_string(reused[1]) + " tokens");
    for (size_t i = 0; i < follows.size(); ++i) {
        auto fresh = make(2048, 16);
        same(serve(*fresh, tok, 1, {{follows[i]}})[0], got[i], what + ", follow-up " + std::to_string(i));
    }
    if (through) {
        // Neither the copies nor the host memory they were read through outlive their requests.
        require(second.host_donors == 0 && slabs == 0, what + ": " + std::to_string(second.host_donors) + " copies and " + std::to_string(slabs) + " bytes of slabs left in host memory");
        return;
    }
    // The second scheduler left its entries kept too; one made two days old is past the age limit and not adopted.
    std::vector<fs::path> files = disk.files(".kv");
    require(!files.empty(), what + ": the second scheduler kept nothing");
    fs::last_write_time(files[0], fs::file_time_type::clock::now() - std::chrono::hours(48));
    {
        auto model = make(2048, 16);
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        size_t adopted = 0;
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the third disk tier made");
            within_a_minute([&] { return sched.stats().disk_entries + 1 == files.size(); }, what + ": all but the old entry adopted");
            adopted = sched.stats().disk_entries;
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        require(adopted + 1 == files.size(), what + ": " + std::to_string(adopted) + " of " + std::to_string(files.size()) + " entries adopted, the old one among them");
    }
}

// The age limit on a running server (docs/DISK-TIER.md, Age): with entries unused for five seconds deleted, the copies six turns leave on disk are gone within a few seconds, their files with them, and the server writes on.
// A copy whose entry the limit deleted is unwritten again, so the check follows the files the turns left, taken as they are when the turns end, and the limit of five seconds leaves the first copy written standing on a slow runner.
void disk_age(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "the age limit";
    DiskRoot disk("age");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.max_age = 5;
    auto model = make(2048, 16);
    size_t files = 1;
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            demote(sched, vocab, what);
            const std::vector<fs::path> left = disk.files(".kv");
            require(!left.empty(), what + ": no entry file after six turns");
            // A file being deleted can refuse the question on Windows; it is then still there, and asked again.
            const auto remaining = [&] {
                return (size_t)std::count_if(left.begin(), left.end(), [](const fs::path& f) {
                    std::error_code ec;
                    return fs::exists(f, ec) || ec;
                });
            };
            within_a_minute([&] { return remaining() == 0; }, what + ": the entries deleted");
            files = remaining();
            stats = sched.stats();
            ledger(stats, *model, what);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(files == 0 && stats.disk_writing && stats.disk_bytes_written > 0, what + ": " + std::to_string(files) + " entry files left");
}

// The age limit at a restart follows a conversation, not a file (docs/DISK-TIER.md, Age): a first turn's files made two days old below a second turn's left fresh.
// A second scheduler under a limit of a day adopts them all, and the third turn forks the whole of the second.
void disk_age_restart(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::string what = "the age limit at a restart below a file used since";
    DiskRoot disk("age-restart");
    server::DiskOptions options = disk.options(uint64_t(1) << 30);
    options.keep = true;
    const auto follow = [&](const Req& before, const Reply& reply, uint32_t seed) {
        Req f{before.prompt, 20};
        for (uint32_t id : ids_of(reply)) f.prompt.push_back(id);
        const std::vector<uint32_t> more = prompt_of(seed, 200, vocab);
        f.prompt.insert(f.prompt.end(), more.begin(), more.end());
        return f;
    };
    std::vector<Req> turns = {{prompt_of(21, 300, vocab), 20}};
    std::vector<fs::path> first;
    size_t kept = 0;
    {
        auto model = make(2048, 16);
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            Reply reply = drain(*sched.submit(turns[0].prompt, params_of(turns[0])));
            within_a_minute([&] { return sched.stats().disk_entries > 0 && !sched.stats().disk_in_flight; }, what + ": the first turn's files written");
            first = disk.files(".kv");
            turns.push_back(follow(turns[0], reply, 22));
            reply = drain(*sched.submit(turns[1].prompt, params_of(turns[1])));
            within_a_minute([&] { return disk.files(".kv").size() > first.size() && !sched.stats().disk_in_flight; }, what + ": the second turn's files written");
            turns.push_back(follow(turns[1], reply, 23));
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    kept = disk.files(".kv").size();
    require(!first.empty() && kept > first.size(), what + ": " + std::to_string(first.size()) + " files after the first turn and " + std::to_string(kept) + " kept, against some and more");
    // The first turn's files by their names, which the stop keeps, made two days old; the second turn's stay as written.
    size_t aged = 0;
    for (const fs::path& f : disk.files(".kv"))
        if (std::any_of(first.begin(), first.end(), [&](const fs::path& g) { return g.filename() == f.filename(); })) {
            fs::last_write_time(f, fs::file_time_type::clock::now() - std::chrono::hours(48));
            ++aged;
        }
    require(aged == first.size(), what + ": " + std::to_string(aged) + " of the first turn's " + std::to_string(first.size()) + " files found after the stop");
    options.max_age = 24 * 3600;
    const size_t whole = (turns[1].prompt.size() + 20) / kBlock * kBlock;
    size_t reused = 0, adopted = 0;
    Reply got;
    {
        auto model = make(2048, 16);
        server::Scheduler sched(*model, tok, 1, 64, 0, false, 4 * ((size_t)64 << 20), nullptr, 0, false, options);
        std::thread runner([&] { sched.run(); });
        try {
            within_a_minute([&] { return sched.stats().disk_ready; }, what + ": the disk tier made");
            // The entries are entered in the round after the tier is made; ten seconds for them, and what is there then is what was adopted.
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (sched.stats().disk_entries < kept && std::chrono::steady_clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            adopted = sched.stats().disk_entries;
            const auto h = sched.submit(turns[2].prompt, params_of(turns[2]));
            got = drain(*h);
            reused = h->reused();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(adopted == kept && reused == whole, what + ": " + std::to_string(adopted) + " of " + std::to_string(kept) + " files adopted and the third turn reused " + std::to_string(reused) +
                                                    " tokens, against all of them and the " + std::to_string(whole) + " of the second turn");
    auto fresh = make(2048, 16);
    same(serve(*fresh, tok, 1, {{turns[2]}})[0], got, what + ", the third turn");
    std::cout << "server-resume: " << what << ": " << adopted << " files adopted, " << aged << " of them two days old, and the third turn reused " << reused << " tokens\n";
}

} // namespace

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    if (!only.empty() && only != "cpu" && only != "device") {
        std::fprintf(stderr, "usage: llmx-server-resume-test [cpu|device]\n");
        return 2;
    }
    try {
        if (only != "device") {
            const gguf::GGUFModel weights = served(kCpu);
            const bpe::Tokenizer tok(weights);
            const Make one = on(weights, [] { return cpus(1); });
            const uint32_t vocab = (uint32_t)kCpu.vocab;
            three_uncapped(one, tok, vocab, "three uncapped requests");
            prefilling_victim(one, tok, vocab);
            follow_up(one, tok, vocab);
            follow_up_as_cli(one, tok, vocab, 200, kBlock, "a follow-up turn against its prompt on a fresh model");
            three_uncapped(on(weights, [] { return cpus(2); }), tok, vocab, "a two-CPU layer split");
            const gguf::GGUFModel deep = served(kSplit);
            for (size_t stages = 2; stages <= 3; ++stages) cancelled_in_flight(deep, tok, vocab, stages);
            for (size_t stages = 1; stages <= 3; stages += 2) stopped_in_flight(deep, tok, vocab, stages);
            cancelled_while_paused(one, tok, vocab);
            take_back(one, tok, vocab);
            take_back_follow_up(one, tok, vocab);
            partial_eviction(one, tok, vocab);
            fork_within_class(one, tok, vocab);
            cancel_while_paused_donor(one, tok, vocab);
            resumed_short_of_room(one, tok, vocab);
            refused_evicts_nothing(one, tok, vocab);
            paused_outside_queue(weights, tok, vocab);
            growth_before_admission(one, tok, vocab);
            stall_holds_room(one, tok, vocab);
            cancelled_short_donor(one, tok, vocab);
            const gguf::GGUFModel mixed = served_hybrid(kHybrid);
            hybrid(mixed, bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab);
            hybrid_checkpoints(mixed, bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab);
            for (size_t devices = 1; devices <= 2; ++devices)
                message_boundaries(on(mixed, [devices] { return cpus(devices); }, 3, 3), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab,
                                   "message boundaries on " + std::to_string(devices) + " CPU" + (devices > 1 ? "s" : ""));
            for (const bool reads : {true, false}) boundary_faults(mixed, bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, reads);
            edit_after_regenerate(on(mixed, [] { return cpus(1); }, 3, 3), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab);
            for (size_t devices = 1; devices <= 2; ++devices) disk_demotion(on(weights, [devices] { return cpus(devices); }), tok, vocab, devices);
            disk_superseded(one, tok, vocab);
            for (const size_t copies : {size_t(4), size_t(1)}) disk_never_blocks(one, tok, vocab, copies);
            for (size_t devices = 1; devices <= 2; ++devices) {
                const std::string on_cpus = std::to_string(devices) + " CPU" + (devices > 1 ? "s" : "");
                DiskRoot bounds_disk("boundaries");
                message_boundaries(on(mixed, [devices] { return cpus(devices); }, 3, 3), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab,
                                   "message boundaries read back from disk on " + on_cpus, 3 * devices * ((size_t)64 << 20), bounds_disk.options(uint64_t(1) << 30));
                if (devices == 1) {
                    DiskRoot whole_disk("boundaries-whole");
                    message_boundaries(on(mixed, [] { return cpus(1); }, 3, 3), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab,
                                       "message boundaries beside a disk tier, none thinned, on " + on_cpus, (size_t)1 << 30, whole_disk.options(uint64_t(1) << 30), 6);
                }
                disk_restore(on(weights, [devices] { return cpus(devices); }), tok, vocab, devices, "a conversation read back from disk on " + on_cpus);
                disk_restore(on(mixed, [devices] { return cpus(devices); }, 3, 2), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, devices,
                             "a hybrid conversation read back from disk on " + on_cpus);
            }
            recorder_rule(weights, tok, vocab, 2);
            recorder_rule(weights, tok, vocab, 1);
            recorder_rule(weights, tok, vocab, 1, 0.5);
            disk_read_waits(one, tok, vocab);
            disk_read_bound(one, tok, vocab);
            disk_flush(one, tok, vocab);
            disk_increment(one, tok, vocab, "disk writes of what changed");
            disk_increment(on(mixed, [] { return cpus(1); }, 3, 3), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, "disk writes of what changed, a model keeping a state");
            disk_faults(one, tok, vocab, "files lost between two servers");
            disk_faults(on(mixed, [] { return cpus(1); }, 3, 3), bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, "files lost between two servers, a model keeping a state");
            disk_cap_base(one, tok, vocab);
            disk_kept(one, tok, vocab);
            disk_kept(one, tok, vocab, (size_t)1 << 16);
            disk_kept(one, tok, vocab, 4 * ((size_t)64 << 20), (size_t)1 << 16);
            disk_age(one, tok, vocab);
            disk_age_restart(one, tok, vocab);
            for (size_t devices = 1; devices <= 2; ++devices)
                host_tier(mixed, bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, devices, 1, HostFault::none,
                          "a hybrid model's donors in host memory on " + std::to_string(devices) + " CPU" + (devices > 1 ? "s" : ""));
            host_tier(mixed, bpe::Tokenizer(mixed), (uint32_t)kHybrid.vocab, 1, 1, HostFault::promotion, "a hybrid model's donors in host memory, a promotion failing");
            // The job reads 128 rows in passes of 16, so it can be cancelled at each of the seven boundaries before its last pass.
            writing_growth(one, tok, vocab);
            job_beside_decode(weights, tok, vocab, true);
            job_beside_decode(weights, tok, vocab, false);
            job_forks_running(weights, tok, vocab);
            for (size_t devices = 1; devices <= 2; ++devices)
                host_tier(weights, tok, vocab, devices, 0, HostFault::none, "donors in host memory on " + std::to_string(devices) + " CPU" + (devices > 1 ? "s" : ""));
            host_tier(weights, tok, vocab, 1, 0, HostFault::write_back, "donors in host memory, a write-back failing");
            host_tier(weights, tok, vocab, 1, 0, HostFault::promotion, "donors in host memory, a promotion failing");
            failed_promotion(one, tok, vocab);
            // Over tensor groups of two CPUs (docs/TENSOR-SPLIT.md, step 2), each reply against its run alone on one group, since a group gives its own bits: pauses, take-backs, forks and donors in host memory, each member's storage copied on its own.
            const Make group = on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2);
            three_uncapped(group, tok, vocab, "three uncapped requests on a group of two CPUs");
            three_uncapped(on(weights, [] { return cpus(4); }, 8, 0, 0, 0, false, 2), tok, vocab, "three uncapped requests on two stages of groups of two CPUs");
            take_back(group, tok, vocab);
            follow_up_as_cli(group, tok, vocab, 200, kBlock, "a follow-up turn on a group of two CPUs against its prompt on a fresh model");
            host_tier(weights, tok, vocab, 2, 0, HostFault::none, "donors in host memory on a group of two CPUs", 2);
            // A hybrid model over a group of two CPUs, each member keeping the state of its own heads under the shared slot ids: pauses, a fork at a checkpoint, a state kept and taken back, and a donor's state copied to host memory and back.
            {
                const gguf::GGUFModel even = served_hybrid(kHybridEven);
                const bpe::Tokenizer even_tok(even);
                const uint32_t v = (uint32_t)kHybridEven.vocab;
                no_donors(alone_then_together(on(even, [] { return cpus(2); }, 3, 0, 0, 0, false, 2), even_tok, 1024, 0, 3, {},
                                              {{prompt_of(1, 40, v)}, {prompt_of(2, 9, v)}, {prompt_of(3, 23, v)}}, "a hybrid model's three uncapped requests on a group of two CPUs"),
                          "a hybrid model's three uncapped requests on a group of two CPUs");
                const Make kept = on(even, [] { return cpus(2); }, 3, 3, 0, 0, false, 2);
                auto model = kept(1024, 0);
                server::Scheduler::Stats stats;
                const Req a{prompt_of(5, 300, v), 40};
                const std::vector<Reply> r = serve(*model, even_tok, 1, {{a}, {a}}, &stats);
                same(r[0], r[1], "a hybrid model's repeated prompt on a group of two CPUs");
                require(stats.prefix_tokens == 2 * kBlock, "a hybrid model's repeated prompt on a group of two CPUs reused " + std::to_string(stats.prefix_tokens) + " tokens");
                take_back(kept, even_tok, v);
                host_tier(even, even_tok, v, 2, 1, HostFault::none, "a hybrid model's donors in host memory on a group of two CPUs", 2);
            }
            superseded_donor(one, tok, vocab);
            one_copy_per_conversation(one, tok, vocab);
            conversations_that_came_back(one, tok, vocab);
            count_gives_up_superseded(one, tok, vocab);
            renewed_copy_current(one, tok, vocab);
            for (const When w : {When::idle, When::writing, When::at_once}) {
                const std::string when = w == When::writing ? " while it is written" : w == When::at_once ? " and answered at once" : "";
                reprefilled(one, tok, vocab, 0, w, "a reply read again" + when);
                reprefilled(on(weights, [] { return cpus(2); }), tok, vocab, 0, w, "a reply read again over a two-CPU split" + when);
                const bpe::Tokenizer mixed_tok(mixed);
                const Make kept = on(mixed, [] { return cpus(1); }, 3, 3);
                for (size_t k = 0; k <= (w == When::idle ? 7u : 0u); ++k)
                    reprefilled(kept, mixed_tok, (uint32_t)kHybrid.vocab, k, w,
                                "a hybrid model's reply read again" + when + (k ? ", cancelled after pass " + std::to_string(k) : std::string()));
                if (w == When::idle) reprefilled(kept, mixed_tok, (uint32_t)kHybrid.vocab, 3, w, "a hybrid model's reply read again beside a request that fits", true);
                reprefilled(on(mixed, [] { return cpus(2); }, 3, 3), mixed_tok, (uint32_t)kHybrid.vocab, 0, w, "a hybrid model's reply read again over a two-CPU split" + when);
            }
            std::printf("server-resume: CPU cases pass\n");
        }
        if (only != "cpu") {
            backend::BackendPtr device;
#if LLMX_HAS_BACKEND_VULKAN
            try { device = backend::make_backend("vulkan:0"); } catch (const std::exception&) {}
#endif
            if (device) {
                device.reset();
                const gguf::GGUFModel weights = served(kDevice);
                const bpe::Tokenizer tok(weights);
                const Make on_device = on(weights, [] { return std::vector<backend::BackendPtr>{backend::make_backend("vulkan:0")}; });
                device_classes(on_device, tok, (uint32_t)kDevice.vocab);
                // 500 and 630 tokens take the same tile split, so the follow-up forks the first prompt's seven whole blocks of 64.
                follow_up_as_cli(on_device, tok, (uint32_t)kDevice.vocab, 500, 448, "on a device, a follow-up turn against its prompt on a fresh model");
                other_split_as_cli(on_device, tok, (uint32_t)kDevice.vocab);
                std::printf("server-resume: device cases pass\n");
            } else {
                std::printf("server-resume: no Vulkan device, device cases skipped\n");
            }
        }
        std::cout << "server-resume: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "server-resume: " << error.what() << '\n';
        return 1;
    }
}
