#pragma once
// What the scheduler's native tests share: the synthetic Q8_0 model served over CPU backends or a device, requests and their replies, a run of waves through one scheduler, and replies compared bit for bit.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "backends/devices.hpp"
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/arch/registry.hpp"
#include "server/scheduler.hpp"

inline size_t checks = 0;

inline void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

struct Shape {
    int layers, embd, ff, heads, kv_heads, head_dim, vocab;
};
// Small enough that a thousand-token reply takes a fraction of a second on the CPU.
inline const Shape kCpu{2, 64, 128, 4, 2, 16, 64};
// Heads of 128, which take the device's wide attention for a prompt and its per-row attention for a generated token.
inline const Shape kDevice{2, 256, 512, 2, 1, 128, 256};
// The CPU shape with a layer for each of up to four stages, for the cases that split it over several CPUs.
inline const Shape kSplit{4, 64, 128, 4, 2, 16, 64};

// The synthetic model with a token list and no end token, so an uncapped reply runs to what the request may hold.
inline gguf::GGUFModel served(const Shape& s) {
    gguf::GGUFModel m = infer::synthetic_model({s.layers, s.embd, s.ff, s.heads, s.kv_heads, s.head_dim, s.vocab, 20260925u});
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
inline std::vector<uint32_t> prompt_of(uint32_t first, size_t n, uint32_t vocab) {
    std::vector<uint32_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = i ? (uint32_t)((first * 13 + i * 7) % vocab) : first;
    return p;
}

using Reply = std::vector<server::Request::Token>;

// One request: its prompt, and a cap, or none (0), which makes it uncapped as a compatible route's request without max_tokens is; a stop string, which ends it where its text first holds it; and a temperature above 0 with a seed for a sampled one.
struct Req {
    std::vector<uint32_t> prompt;
    int cap = 0;
    std::string stop = {};
    float temp = 0.0f;
    uint64_t seed = 0;
};

inline server::SampleParams params_of(const Req& r) {
    server::SampleParams p;
    p.temp = r.temp;
    p.seed = r.seed;
    p.logprobs = true;
    p.top_logprobs = 5;
    p.until_limit = r.cap == 0;
    if (r.cap) p.max_tokens = r.cap;
    if (!r.stop.empty()) p.stop = {r.stop};
    return p;
}

// Every token a request's channel gives until it ends.
inline Reply collect(server::Request& r) {
    Reply out;
    server::Request::Token t;
    for (;;) {
        const auto got = r.next(t, server::Request::Clock::now() + std::chrono::seconds(120));
        if (got == server::Request::Next::end) break;
        require(got == server::Request::Next::id, "a channel gave nothing for 120 seconds");
        out.push_back(t);
    }
    return out;
}

// A request's reply, which must run to its length or its stop string.
inline Reply drain(server::Request& r) {
    Reply out = collect(r);
    require(r.finish() == "length" || r.finish() == "stop", "a request ended with " + r.finish() + " " + r.error());
    return out;
}

// A stop string that ends a request whose reply is `ids` with its token `at`: the text of its tokens up to `at`, from the latest start whose text first appears there.
inline std::string stop_at(const bpe::Tokenizer& tok, const std::vector<uint32_t>& ids, size_t at) {
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
inline void ledger(const server::Scheduler::Stats& s, const infer::Model& model, const std::string& what) {
    require(s.reserved.size() == model.kv_pools() && s.donor_blocks.size() == model.kv_pools(), what + ": the ledger does not name every pool");
    for (size_t p = 0; p < s.reserved.size(); ++p)
        require(s.reserved[p] <= model.kv_pool_blocks(p) && s.reserved[p] == s.donor_blocks[p],
                what + ": pool " + std::to_string(p) + " holds " + std::to_string(s.reserved[p]) + " blocks reserved and its donors " +
                std::to_string(s.donor_blocks[p]) + ", of " + std::to_string(model.kv_pool_blocks(p)));
}

// Runs `waves` through one scheduler over `model` with `passes` passes in flight, the scheduler's own number when 0: a wave is fully queued before any pass retires, and every request is drained before the next wave starts.
// The replies come in submission order; the scheduler's counters at the end go to `stats`.
inline std::vector<Reply> serve(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs,
                         const std::vector<std::vector<Req>>& waves, server::Scheduler::Stats* stats = nullptr, size_t passes = 0) {
    server::Scheduler sched(model, tok, max_seqs, 64, passes);
    std::mutex submitting;
    // The first pass may start while a wave is queued, but cannot retire and advance its request ahead of the rest.
    sched.on_retire = [&](const server::Scheduler::Retired&) { std::lock_guard<std::mutex> lock(submitting); };
    std::thread runner([&] { sched.run(); });
    std::vector<Reply> replies;
    try {
        for (const auto& wave : waves) {
            std::vector<std::shared_ptr<server::Request>> handles;
            {
                std::lock_guard<std::mutex> lock(submitting);
                for (const Req& r : wave) handles.push_back(sched.submit(r.prompt, params_of(r)));
            }
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

// A reply against the same request's reply alone: every id, logprob and top entry equal, bit for bit; a reply that ended early, a cancelled one's, may be a prefix of it.
inline void same(const Reply& alone, const Reply& got, const std::string& what, bool prefix = false) {
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
    require(alone.size() == got.size() || (prefix && got.size() < alone.size()),
            what + ": " + std::to_string(got.size()) + " tokens against " + std::to_string(alone.size()) + " alone");
}

// A fresh model over the backends `backends` makes, with a pool of `pool` tokens and prompts taken `ubatch` tokens a pass (the default when 0).
using Make = std::function<std::unique_ptr<infer::Model>(size_t pool, int ubatch)>;

inline Make on(const gguf::GGUFModel& weights, std::function<std::vector<backend::BackendPtr>()> backends) {
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

inline std::vector<backend::BackendPtr> cpus(size_t n) {
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

inline std::vector<std::shared_ptr<Hooked>> hooked(size_t n) {
    std::vector<std::shared_ptr<Hooked>> v;
    for (size_t i = 0; i < n; ++i) {
        v.push_back(std::make_shared<Hooked>());
        v.back()->set_threads(1);
    }
    return v;
}

// The CPU's 128-token blocks, of which a case's pool holds a given number.
inline constexpr size_t kBlock = 128;

// Once the scheduler has stopped, the model's own history takes the whole pool of `tokens`, so every block has come back, a block of a request whose handle is still held included.
inline void whole_pool_free(infer::Model& model, size_t tokens, uint32_t vocab, const std::string& what) {
    try {
        model.prefill(prompt_of(7, tokens, vocab));
    } catch (const std::exception& e) {
        require(false, what + ": a prompt as long as the pool failed once the scheduler had stopped (" + e.what() + ")");
    }
    model.reset();
}
