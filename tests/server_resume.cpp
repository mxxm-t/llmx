// Requests the scheduler pauses and resumes give, token for token, the ids and log-probabilities they give alone, over the synthetic Q8_0 model whose prompt and decode rows take different CPU paths, and room goes by first admission (docs/SERVER.md).
// A hybrid model, whose linear-attention layers keep a recurrent state, keeps no donor, and its requests resume by recomputing from their start.
// A request cancelled, or a scheduler stopped, while a pass is in flight leaves every block to come back and every donor free to fork.
// Usage: llmx-server-resume-test [cpu|device]; both by default, the device cases on Vulkan device 0 when it opens.
#include <atomic>
#include <chrono>
#include <iostream>

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
// Two fit at first; the first reaches its reservation's end ahead of the second, which is paused with generated tokens in its partial block and, its donor taken for the first's growth, recomputes its whole history; the third then runs beside the first and is paused in turn.
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

// B paused as in take_back, its donor intact, and cancelled before it resumes, which C's longer reply (720 tokens in all, on 16 blocks) leaves time for: it ends where it waits, taking nothing back, A and C finish as they do alone, and once the scheduler stops every pool is empty.
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
// On 10 blocks uncapped A and capped E (128 tokens and 345) start together and capped D (138 and 600) waits; E ends in the pass before A's step at 384 tokens falls due, and D fits only in the room that step needs, so D waits for A's stop rather than A sitting out D's reply.
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
// On 8 blocks uncapped A and capped E (20 tokens and 300) start together and uncapped B (10) waits for E's end; A's step at 384 tokens pauses B at 54 tokens with its donor kept, and B is cancelled while it waits, so once A ends at its stop string A's history is the one donor.
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
    std::thread runner([&] { sched.run(); });
    try {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (sched.stats().pauses == 0) {
            require(std::chrono::steady_clock::now() < until, "a paused request with a short donor: nothing paused in 60 seconds");
            std::this_thread::yield();
        }
        h[2]->cancel();
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
// On 8 blocks the setup leaves a donor of 2 blocks, and N finds no room beside capped K even with that donor gone; K ends at its stop string holding less than a block, so N then starts beside the donor, which a prompt repeating the setup's still forks.
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

// Paused requests wait apart from the queue, so they do not fill the queue --max-queue bounds.
// With a queue of one, A's growth pauses B as in three_uncapped, and B waits for A's end; a request submitted meanwhile is queued, not refused.
// The scheduler's own thread submits B inside A's first pass, while A holds three of the pool's eight blocks, and C inside the first pass that finds B paused, so the case follows from the blocks alone and not from when the test's thread runs.
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

// The hybrid model: requests paused and resumed give their replies alone on one CPU and over a two-CPU split with passes in flight, a follow-up turn recomputes its history rather than forking, and requests cancelled paused, in flight or by a stop leave every block and every state slot free.
// A hybrid model with checkpoint slots (docs/SPECULATIVE.md, section 2): a finished request keeps its state at its prompt's last whole block, so a follow-up turn forks it and gives the reply of its prompt on a fresh model; with one slot the newer conversation's checkpoint takes the older's; and a paused request keeps its whole history as its checkpoint and takes it back, recomputing nothing, as take_back has it.
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
