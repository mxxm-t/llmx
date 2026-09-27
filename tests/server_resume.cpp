// Requests the scheduler pauses and resumes give, token for token, the ids and log-probabilities they give alone, over the synthetic Q8_0 model whose prompt and decode rows take different CPU paths, and room goes by first admission (docs/SERVER.md).
// A request cancelled, or a scheduler stopped, while a pass is in flight leaves every block to come back and every donor free to fork.
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
// The CPU shape with a layer for each of up to four stages, for the cases that split it over three CPUs.
const Shape kSplit{4, 64, 128, 4, 2, 16, 64};

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

// Once every request has ended, the blocks the scheduler holds reserved are the donors' blocks, never more than a pool has.
void ledger(const server::Scheduler::Stats& s, const infer::Model& model, const std::string& what) {
    require(s.reserved.size() == model.kv_pools() && s.donor_blocks.size() == model.kv_pools(), what + ": the ledger does not name every pool");
    for (size_t p = 0; p < s.reserved.size(); ++p)
        require(s.reserved[p] <= model.kv_pool_blocks(p) && s.reserved[p] == s.donor_blocks[p],
                what + ": pool " + std::to_string(p) + " holds " + std::to_string(s.reserved[p]) + " blocks reserved and its donors " +
                std::to_string(s.donor_blocks[p]) + ", of " + std::to_string(model.kv_pool_blocks(p)));
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
            ledger(sched.stats(), model, "after a wave");
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

// A CPU backend that runs `hook` whenever the model submits its work, which under the scheduler happens only inside a pass's stages, so a case acts in the scheduler's own thread while that pass is in flight.
struct Hooked : backend::CpuBackend {
    std::function<void()> hook;
    backend::Ticket submit() override {
        if (hook) hook();
        return CpuBackend::submit();
    }
};

std::vector<std::shared_ptr<Hooked>> hooked(size_t n) {
    std::vector<std::shared_ptr<Hooked>> v;
    for (size_t i = 0; i < n; ++i) {
        v.push_back(std::make_shared<Hooked>());
        v.back()->set_threads(1);
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

// The CPU's 128-token blocks, of which a case's pool holds a given number.
constexpr size_t kBlock = 128;

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

// A follow-up turn forks the whole blocks of the previous turn's history, its reply rows among them, computed as generated tokens.
// Paused beside an uncapped request with a short prompt, which then takes its donor, it must recompute those reply rows as generated tokens again, not as rows of its own prompt.
void follow_up(const Make& make, const bpe::Tokenizer& tok, uint32_t vocab) {
    const Req first{prompt_of(5, 20, vocab), 200};
    auto model = make(1024, 0);
    const Reply turn = serve(*model, tok, 3, {{first}})[0];
    std::vector<uint32_t> again = first.prompt;
    for (const auto& t : turn) again.push_back(t.id);
    const std::vector<uint32_t> more = prompt_of(6, 30, vocab);
    again.insert(again.end(), more.begin(), more.end());
    const auto s = alone_then_together(make, tok, 1024, 0, 3, {first}, {{prompt_of(1, 9, vocab)}, {again}}, "a follow-up turn paused");
    // It sat out passes at the end of its 512-token reservation and was paused there, so a resume that forked nothing recomputes those 512 rows, the forked reply rows among them.
    require(s.taken_back == 0 && s.recomputed >= 4 * kBlock, "a follow-up turn paused: " + std::to_string(s.taken_back) + " taken back and " +
            std::to_string(s.recomputed) + " rows recomputed, against 0 and at least 512");
}

// On a device, a 100-token prompt that forks the first block of a 600-token prompt's history: those rows were computed at the longer prompt's extent, whose tile splits its sums another way than a prompt under 449 tokens.
// Admitted after an uncapped request with a short prompt, it is paused when that one grows, and its donor then goes too, so it recomputes the forked rows at the extent that first computed them and its own at its own.
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

// Once the scheduler has stopped, the model's own history takes the whole pool of `tokens`, so every block has come back, a block of a request whose handle is still held included.
void whole_pool_free(infer::Model& model, size_t tokens, uint32_t vocab, const std::string& what) {
    try {
        model.prefill(prompt_of(7, tokens, vocab));
    } catch (const std::exception& e) {
        require(false, what + ": a prompt as long as the pool failed once the scheduler had stopped (" + e.what() + ")");
    }
    model.reset();
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
        // New requests wait until no request is paused, so the queued one starts only once B has resumed.
        server::Request::Token t;
        require(h[2]->next(t, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, "a request queued while another was paused gave nothing");
        require(sched.stats().paused == 0, "a request queued while another was paused started before the paused request resumed");
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
            paused_outside_queue(one, tok, vocab);
            growth_before_admission(one, tok, vocab);
            stall_holds_room(one, tok, vocab);
            cancelled_short_donor(one, tok, vocab);
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
