// Requests the scheduler pauses and resumes give, token for token, the ids and log-probabilities they give alone, over the synthetic Q8_0 model whose prompt and decode rows take different CPU paths, and room goes by first admission (docs/SERVER.md).
// Usage: llmx-server-resume-test [cpu|device]; both by default, the device cases on Vulkan device 0 when it opens.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "backends/devices.hpp"
#include "model/arch_qwen.hpp"
#include "server/scheduler.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

struct Shape {
    int layers, embd, ff, heads, kv_heads, head_dim, vocab;
};
// Small enough that a thousand-token reply takes a fraction of a second on the CPU.
const Shape kCpu{2, 64, 128, 4, 2, 16, 64};
// Heads of 128, which take the device's wide attention for a prompt and its per-row attention for a generated token.
const Shape kDevice{2, 256, 512, 2, 1, 128, 256};

// The synthetic model with a token list and no end token, so an uncapped reply runs to what the request may hold.
gguf::GGUFModel served(const Shape& s) {
    gguf::GGUFModel m = infer::synthetic_model(s.layers, s.embd, s.ff, s.heads, s.kv_heads, s.head_dim, s.vocab, 20260925u);
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (int i = 0; i < s.vocab; ++i) {
        gguf::MetaValue t;
        t.vtype = gguf::V_STRING;
        t.s = "t" + std::to_string(i);
        tokens.arr.push_back(t);
    }
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    return m;
}

// A prompt of n ids starting with `first`, the rest a fixed walk through the vocabulary.
std::vector<uint32_t> prompt_of(uint32_t first, size_t n, uint32_t vocab) {
    std::vector<uint32_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = i ? (uint32_t)((first * 13 + i * 7) % vocab) : first;
    return p;
}

using Reply = std::vector<server::Request::Token>;

// One request: its prompt, and a cap, or none (0), which makes it uncapped as a compatible route's request without max_tokens is; and a stop string, which ends it where its text first holds it.
struct Req {
    std::vector<uint32_t> prompt;
    int cap = 0;
    std::string stop = {};
};

server::SampleParams params_of(const Req& r) {
    server::SampleParams p;
    p.temp = 0.0f;
    p.logprobs = true;
    p.top_logprobs = 5;
    p.until_limit = r.cap == 0;
    if (r.cap) p.max_tokens = r.cap;
    if (!r.stop.empty()) p.stop = {r.stop};
    return p;
}

Reply drain(server::Request& r) {
    Reply out;
    server::Request::Token t;
    for (;;) {
        const auto got = r.next(t, server::Request::Clock::now() + std::chrono::seconds(120));
        if (got == server::Request::Next::end) break;
        require(got == server::Request::Next::id, "a channel gave nothing for 120 seconds");
        out.push_back(t);
    }
    require(r.finish() == "length" || r.finish() == "stop", "a request ended with " + r.finish() + " " + r.error());
    return out;
}

// A stop string that ends a request whose reply is `ids` with its token `at`: the text of its tokens up to `at`, from the latest start whose text first appears there.
std::string stop_at(const bpe::Tokenizer& tok, const std::vector<uint32_t>& ids, size_t at) {
    std::string whole;
    std::vector<size_t> ends;
    for (uint32_t id : ids) {
        whole += tok.decode({id});
        ends.push_back(whole.size());
    }
    for (size_t from = at + 1; from-- > 0;) {
        const size_t begin = from ? ends[from - 1] : 0;
        const std::string s = whole.substr(begin, ends[at] - begin);
        if (whole.find(s) == begin) return s;
    }
    throw std::runtime_error("no stop string ends the reply at its token " + std::to_string(at));
}

// Runs `waves` through one scheduler over `model`: a wave's requests are submitted together and every one drained before the next wave starts.
// The replies come in submission order; the scheduler's counters at the end go to `stats`.
std::vector<Reply> serve(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs,
                         const std::vector<std::vector<Req>>& waves, server::Scheduler::Stats* stats = nullptr) {
    server::Scheduler sched(model, tok, max_seqs, 64);
    std::thread runner([&] { sched.run(); });
    std::vector<Reply> replies;
    try {
        for (const auto& wave : waves) {
            std::vector<std::shared_ptr<server::Request>> handles;
            for (const Req& r : wave) handles.push_back(sched.submit(r.prompt, params_of(r)));
            for (auto& h : handles) replies.push_back(drain(*h));
        }
        if (stats) *stats = sched.stats();
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    return replies;
}

// A reply against the same request's reply alone: every id, logprob and top entry equal, bit for bit.
void same(const Reply& alone, const Reply& got, const std::string& what) {
    const size_t n = std::min(alone.size(), got.size());
    for (size_t i = 0; i < n; ++i) {
        bool eq = alone[i].id == got[i].id && alone[i].logprob == got[i].logprob && alone[i].top.size() == got[i].top.size();
        for (size_t j = 0; eq && j < alone[i].top.size(); ++j)
            eq = alone[i].top[j].id == got[i].top[j].id && alone[i].top[j].logprob == got[i].top[j].logprob;
        if (!eq) {
            char detail[160];
            std::snprintf(detail, sizeof detail, ": token %zu of %zu differs from the request alone (id %u, logprob %.9g; alone id %u, logprob %.9g)",
                          i, alone.size(), got[i].id, (double)got[i].logprob, alone[i].id, (double)alone[i].logprob);
            require(false, what + detail);
        }
    }
    require(alone.size() == got.size(), what + ": " + std::to_string(got.size()) + " tokens against " + std::to_string(alone.size()) + " alone");
}

// A fresh model over the backends `backends` makes, with a pool of `pool` tokens and prompts taken `ubatch` tokens a pass (the default when 0).
using Make = std::function<std::unique_ptr<infer::Model>(size_t pool, int ubatch)>;

Make on(const gguf::GGUFModel& weights, std::function<std::vector<backend::BackendPtr>()> backends) {
    return [&weights, backends](size_t pool, int ubatch) {
        infer::ModelOptions options;
        options.kv_tokens = pool;
        std::vector<backend::BackendPtr> b = backends();
        infer::PlacementRequest request;
        for (size_t i = 0; i < b.size(); ++i) request.names.push_back("device " + std::to_string(i));
        if (b.size() > 1) request.shares.assign(b.size(), 1);
        request.ubatch = ubatch;
        return std::move(infer::place_model(infer::gguf_weights(weights), std::move(b), request, options).model);
    };
}

std::vector<backend::BackendPtr> cpus(size_t n) {
    std::vector<backend::BackendPtr> v;
    for (size_t i = 0; i < n; ++i) {
        auto c = std::make_shared<backend::CpuBackend>();
        c->set_threads(1);
        v.push_back(c);
    }
    return v;
}

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
    alone_then_together(make, tok, 1024, 1, 3, {}, reqs, "a victim still prefilling");
}

// A follow-up turn forks the whole blocks of the previous turn's history, its reply rows among them, computed as generated tokens.
// Admitted after an uncapped request with a short prompt, it reaches its reservation's end first and is paused; the other's growth then takes its donor, so it recomputes all it holds.
// Its forked reply rows must be recomputed as generated tokens again, which recomputing them as rows of its own prompt would get wrong.
void follow_up(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req first{prompt_of(5, 20, vocab), 200};
    auto model = make(1024, 0);
    const Reply turn = serve(*model, tok, 3, {{first}})[0];
    std::vector<uint32_t> again = first.prompt;
    for (const auto& t : turn) again.push_back(t.id);
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    alone_then_together(make, tok, 1024, 0, 3, {first}, {{prompt_of(1, 9, vocab)}, {again}}, "a follow-up turn paused");
}

// On a device, a 100-token prompt that forks the first block of a 600-token prompt's history: those rows were computed at the longer prompt's extent, whose tile splits its sums another way than a prompt under 449 tokens.
// Admitted after an uncapped request with a short prompt, it is paused when that one grows, and its donor then goes too, so it recomputes the forked rows at the extent that first computed them and its own at its own.
void device_classes(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req donor{prompt_of(9, 600, vocab), 4};
    std::vector<uint32_t> forked(donor.prompt.begin(), donor.prompt.begin() + 64);
    const std::vector<uint32_t> own = prompt_of(10, 36, vocab);
    forked.insert(forked.end(), own.begin(), own.end());
    alone_then_together(make, tok, 1024, 0, 3, {donor}, {{prompt_of(1, 9, vocab)}, {forked}}, "on a device, a forked prefix of another extent");
    three_uncapped(make, tok, vocab, "on a device, three uncapped requests");
}

// The ids of a reply.
std::vector<uint32_t> ids_of(const Reply& r) {
    std::vector<uint32_t> ids;
    for (const auto& t : r) ids.push_back(t.id);
    return ids;
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

// The CPU's 128-token blocks, of which a take-back case's pool holds a given number.
constexpr size_t kBlock = 128;

// Three requests on 13 blocks: an uncapped one with a 40-token prompt (A), an uncapped one with 60 (B) and a capped one (C, 380 tokens in all), which leave 4 blocks free.
// B takes its first growth step, 3 blocks, at 384 tokens; A takes its own 20 passes later, and with one block free pauses B, whose step leaves 2 of its blocks unused, so B's donor stays.
// C ends 16 passes later and B takes its donor back whole: it recomputes nothing, and its stop string ends it before either needs room again.
void take_back(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 13 * kBlock;
    const Req a{prompt_of(1, 40, vocab)}, c{prompt_of(3, 20, vocab), 360};
    const Req b = stopping(make, tok, pool, {}, Req{prompt_of(2, 60, vocab)}, 420);
    const auto s = alone_then_together(make, tok, pool, 0, 3, {}, {a, b, c}, "a paused request taking its donor back");
    require(s.pauses == 1 && s.taken_back == 1 && s.recomputed == 0,
            "a paused request whose donor stayed: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) + " taken back, " +
            std::to_string(s.recomputed) + " rows recomputed, against 1, 1 and 0");
}

// A follow-up turn, which forked the first block of the previous turn's history at its first admission, paused with its donor intact as B above is (14 blocks, the previous turn's donor going to its first growth step): it takes back its donor, forked prefix and reply rows included.
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

// Part of a paused request's history outlives its own donor: B and the capped C forked the first block of the setup's 200-token prompt at their first admissions, so both hold it as that prompt's rows.
// On 14 blocks B cannot take its growth step at 512 tokens beside A and C, and is paused, its donor then going to A's step; C ends and leaves the shared block in a donor of its own.
// B cannot take its donor back and forks that block, computed as its own was, recomputing the other 384 rows in their classes.
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
    require(s.pauses == 1 && s.taken_back == 1 && s.recomputed == 0,
            "a resumed request short of room for its own growth: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.taken_back) +
            " taken back, " + std::to_string(s.recomputed) + " rows recomputed, against 1, 1 and 0");
}

// A request refused room evicts no donor for it.
// On 8 blocks the setup leaves a donor of 2 blocks, and N finds no room beside capped K even with that donor gone; K ends at its stop string holding less than a block, so N then starts beside the donor, which a prompt repeating the setup's still forks.
void refused_evicts_nothing(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 8 * kBlock;
    const Req setup{prompt_of(5, 200, vocab), 1};
    const Req k = stopping(make, tok, pool, {}, Req{prompt_of(1, 10, vocab), 600}, 60);
    const Req n{prompt_of(2, 10, vocab), 600};
    auto model = make(pool, 0);
    server::Scheduler::Stats s;
    serve(*model, tok, 3, {{setup}, {k, n}, {Req{setup.prompt, 4}}}, &s);
    require(s.prefix_hits == 1 && s.prefix_tokens == kBlock,
            "a donor evicted for a request then refused: a prompt repeating it reused " + std::to_string(s.prefix_tokens) + " tokens against " +
            std::to_string(kBlock));
}

// Paused requests wait apart from the queue, so they do not fill the queue --max-queue bounds.
// With a queue of one, A's growth pauses B as in three_uncapped, and B waits for A's end; a request submitted meanwhile is queued, not refused.
void paused_outside_queue(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    auto model = make(1024, 0);
    server::Scheduler sched(*model, tok, 3, 1);
    std::thread runner([&] { sched.run(); });
    try {
        const auto until = [&](const std::function<bool(const server::Scheduler::Stats&)>& done, const std::string& what) {
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(60);
            while (!done(sched.stats())) {
                require(std::chrono::steady_clock::now() < limit, "paused requests and the queue: " + what + " in 60 seconds");
                std::this_thread::yield();
            }
        };
        std::vector<std::shared_ptr<server::Request>> h;
        h.push_back(sched.submit(prompt_of(1, 40, vocab), params_of(Req{})));
        until([](const server::Scheduler::Stats& s) { return s.queued == 0; }, "the first request not admitted");
        h.push_back(sched.submit(prompt_of(2, 9, vocab), params_of(Req{})));
        until([](const server::Scheduler::Stats& s) { return s.queued == 0; }, "the second request not admitted");
        until([](const server::Scheduler::Stats& s) { return s.pauses > 0; }, "nothing paused");
        try {
            h.push_back(sched.submit(prompt_of(3, 10, vocab), params_of(Req{{}, 4})));
        } catch (const server::QueueFull&) {
            require(false, "a request submitted while another was paused was refused as though the queue were full");
        }
        for (auto& r : h) drain(*r);
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
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
            three_uncapped(on(weights, [] { return cpus(2); }), tok, vocab, "a two-CPU layer split");
            cancelled_while_paused(one, tok, vocab);
            take_back(one, tok, vocab);
            take_back_follow_up(one, tok, vocab);
            partial_eviction(one, tok, vocab);
            cancel_while_paused_donor(one, tok, vocab);
            resumed_short_of_room(one, tok, vocab);
            refused_evicts_nothing(one, tok, vocab);
            paused_outside_queue(one, tok, vocab);
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
                device_classes(on(weights, [] { return std::vector<backend::BackendPtr>{backend::make_backend("vulkan:0")}; }),
                               tok, (uint32_t)kDevice.vocab);
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
