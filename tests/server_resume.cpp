// A request the scheduler pauses and resumes must give, token for token, the ids and log-probabilities it gives run alone, where it never pauses (docs/SERVER.md, pausing).
// Uncapped greedy requests share a pool too small for all of them, through the scheduler over the synthetic Q8_0 model, whose decode rows take the CPU's 8-bit dots and whose prompt rows take the float path, so a generated token recomputed as a prompt row shows in its values.
// Each case runs its requests alone first, then together, and compares every channel's ids, logprobs and top five; a difference names its first token.
// Usage: llmx-server-resume-test [cpu|device]; both by default, the device cases on Vulkan device 0 when the build has the backend and the device opens.
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

// One request: its prompt, and a cap, or none (0), which makes it uncapped as a compatible route's request without max_tokens is.
struct Req {
    std::vector<uint32_t> prompt;
    int cap = 0;
};

server::SampleParams params_of(const Req& r) {
    server::SampleParams p;
    p.temp = 0.0f;
    p.logprobs = true;
    p.top_logprobs = 5;
    p.until_limit = r.cap == 0;
    if (r.cap) p.max_tokens = r.cap;
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
    require(r.finish() == "length", "a request ended with " + r.finish() + " " + r.error());
    return out;
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
