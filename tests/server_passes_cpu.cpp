// The scheduler with passes in flight (docs/SERVER.md, the round) over the synthetic Q8_0 model and a hybrid one, on one CPU and split over two and three, at P = 1, S, S + 1 and 2S, each request equal to its run alone.
// Cancels, stops and failures from inside a stage, and the replay of a never-pausing load through Model::forward, are checked too, each leaving every block free (AGENTS.md, Tests).
#include <thread>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <set>

#include "server_harness.hpp"

namespace {

// Prompts past it take several passes.
constexpr int kUbatch = 16;
constexpr size_t kSeqs = 6;

// A placement of S CPUs with P passes in flight.
struct Run {
    size_t stages, passes;
};
std::vector<Run> runs() {
    std::vector<Run> r{{1, 1}};
    for (size_t s = 2; s <= 3; ++s)
        for (size_t p : {size_t(1), s, s + 1, 2 * s}) r.push_back({s, p});
    return r;
}
std::string name(const Run& r) {
    return std::to_string(r.stages) + " CPU" + (r.stages == 1 ? "" : "s") + " at P = " + std::to_string(r.passes);
}

// Every request alone on one CPU with one pass in flight, each on a fresh model with a pool of `pool` tokens.
std::vector<Reply> alone(const Make& one, const bpe::Tokenizer& tok, size_t pool, const std::vector<Req>& reqs) {
    std::vector<Reply> out;
    for (const Req& r : reqs) {
        auto model = one(pool, kUbatch);
        out.push_back(serve(*model, tok, kSeqs, {{r}}, nullptr, 1)[0]);
    }
    return out;
}

// Two uncapped requests, one sampled, beside capped ones with prompts past the ubatch, on 8 blocks of 128 tokens, which the uncapped ones outgrow, so they pause and resume.
std::vector<Req> paused_load(uint32_t vocab) {
    return {{prompt_of(1, 40, vocab), 60},
            {prompt_of(2, 5, vocab), 0, {}, 0.8f, 7},
            {prompt_of(3, 70, vocab), 30, {}, 1.0f, 11},
            {prompt_of(4, 12, vocab)},
            {prompt_of(5, 3, vocab), 90},
            {prompt_of(6, 25, vocab), 45, {}, 0.7f, 3}};
}

// Two uncapped requests on 9 blocks, which they fill: the older's growth step falls due four tokens before the younger's, so the younger is in flight as the older's plan pauses it, and the plan waits for its pass to retire.
std::vector<Req> held_load(uint32_t vocab) {
    return {{prompt_of(21, 5, vocab)}, {prompt_of(22, 385, vocab)}};
}

// Capped requests on a pool that holds them all, so nothing pauses or forks: prompts past the ubatch and short ones, greedy and sampled.
std::vector<Req> steady_load(uint32_t vocab) {
    return {{prompt_of(1, 40, vocab), 50},
            {prompt_of(2, 5, vocab), 70, {}, 0.8f, 7},
            {prompt_of(3, 33, vocab), 20},
            {prompt_of(4, 9, vocab), 60, {}, 1.0f, 5},
            {prompt_of(5, 2, vocab), 40}};
}

// The paused load and the held load at every run, each reply its reply alone; it gives the passes apart that stages recorded between a prompt pass's pieces.
// With a `share` some pass must be formed apart on every run, and the held load's waits are not asked for, the younger request's rows being in a pass of their own kind.
uint64_t paused(const Make& one, const std::function<Make(size_t)>& split, const bpe::Tokenizer& tok, uint32_t vocab, double share = 0) {
    uint64_t between = 0;
    const std::vector<Req> reqs = paused_load(vocab), held = held_load(vocab);
    const std::vector<Reply> ref = alone(one, tok, 8 * kBlock, reqs), held_ref = alone(one, tok, 9 * kBlock, held);
    for (const Run& run : runs()) {
        // With a share one pass slot a stage is enough: the passes apart have slots of their own.
        if (share > 0 && run.passes != run.stages) continue;
        auto model = split(run.stages)(8 * kBlock, kUbatch);
        server::Scheduler::Stats s;
        const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, &s, run.passes, 0, nullptr, 0, false, server::Pacing::fixed(share));
        require(s.passes == run.passes, name(run) + ": the scheduler kept " + std::to_string(s.passes) + " passes in flight");
        require(s.pauses > 0, "the paused load on " + name(run) + ": nothing paused");
        require((s.apart > 0) == (share > 0), "the paused load on " + name(run) + ": " + std::to_string(s.apart) + " passes formed apart");
        between += s.between;
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], "the paused load on " + name(run) + ", request " + std::to_string(i));
        model = split(run.stages)(9 * kBlock, kUbatch);
        const std::vector<Reply> kept = serve(*model, tok, kSeqs, {held}, &s, run.passes, 0, nullptr, 0, false, server::Pacing::fixed(share));
        require(s.pauses == 1 && (share > 0 || run.passes != run.stages || run.stages == 1 || s.waits > 0),
                "the held load on " + name(run) + ": " + std::to_string(s.pauses) + " pauses and " + std::to_string(s.waits) +
                " plans waiting on a request in flight, against 1 and some at P = S");
        for (size_t i = 0; i < held.size(); ++i) same(held_ref[i], kept[i], "the held load on " + name(run) + ", request " + std::to_string(i));
        between += s.between;
    }
    return between;
}

// The steady load at every run, each pass recorded as it retires, then replayed in that order through forward on a fresh model of the same placement: every logits row bit for bit, and every reply its reply alone.
void replayed(const Make& one, const std::function<Make(size_t)>& split, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs()) {
        const std::string what = "the replay on " + name(run);
        auto model = split(run.stages)(pool, kUbatch);
        std::vector<server::Scheduler::Retired> passes;
        std::vector<std::shared_ptr<server::Request>> handles;
        std::vector<Reply> got;
        {
            server::Scheduler sched(*model, tok, kSeqs, 64, run.passes);
            sched.on_retire = [&passes](const server::Scheduler::Retired& t) { passes.push_back(t); };
            for (const Req& r : reqs) handles.push_back(sched.submit(r.prompt, params_of(r)));
            std::thread runner([&] { sched.run(); });
            try {
                for (auto& h : handles) got.push_back(drain(*h));
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], what + ", request " + std::to_string(i));
        // Each request's history, its prompt then its reply, which its entries read from.
        std::map<const server::Request*, size_t> index;
        std::vector<std::vector<uint32_t>> history;
        for (size_t i = 0; i < reqs.size(); ++i) {
            index[handles[i].get()] = i;
            history.push_back(reqs[i].prompt);
            for (const auto& t : got[i]) history.back().push_back(t.id);
        }
        auto fresh = split(run.stages)(pool, kUbatch);
        std::vector<infer::Sequence> seqs;
        for (size_t i = 0; i < reqs.size(); ++i) seqs.push_back(fresh->make_sequence());
        infer::ExecContext ctx;
        size_t rows = 0;
        for (size_t p = 0; p < passes.size(); ++p) {
            const server::Scheduler::Retired& t = passes[p];
            std::vector<infer::BatchEntry> entries;
            for (size_t e = 0; e < t.requests.size(); ++e) {
                const size_t i = index.at(t.requests[e]);
                require(seqs[i].length() == t.from[e] && t.from[e] + t.rows[e] <= history[i].size(),
                        what + ": pass " + std::to_string(p) + " continues request " + std::to_string(i) + " from another history");
                infer::BatchEntry b{&seqs[i], history[i].data() + t.from[e], t.rows[e], t.want[e] != 0};
                b.extent = t.extent[e];
                entries.push_back(b);
            }
            fresh->forward(ctx, entries.data(), entries.size());
            require(ctx.n_logits == t.logits.size(), what + ": pass " + std::to_string(p) + " gave another count of logits rows");
            for (size_t w = 0; w < t.logits.size(); ++w, ++rows)
                require(std::memcmp(ctx.logits(w), t.logits[w].data(), t.logits[w].size() * sizeof(float)) == 0,
                        what + ": pass " + std::to_string(p) + ", logits row " + std::to_string(w) + " differs through forward");
        }
        for (auto& s : seqs) fresh->reset(s);
        require(rows > 0, what + ": no logits rows replayed");
    }
}

// A lone request's prompt slices on two CPU stages with a ubatch of 256 (prompt_slice): a burst of four 512-token prompts takes whole ubatches, and a lone 768-token prompt's slices shrink until a request arrives.
// Every reply is its reply alone.
void slices(const Make& one, const std::function<Make(size_t)>& split, const bpe::Tokenizer& tok, uint32_t vocab) {
    constexpr int kWide = 256;
    const size_t pool = 32 * kBlock;
    const auto run = [&](const std::vector<Req>& queued, const Req* arriving, std::vector<server::Scheduler::Retired>& passes,
                         std::vector<std::shared_ptr<server::Request>>& handles) {
        auto model = split(2)(pool, kWide);
        std::vector<Reply> got;
        server::Scheduler sched(*model, tok, kSeqs, 64, 2);
        // The arriving request is submitted on the scheduler thread as the first pass retires, so it is queued before that round's admissions.
        std::shared_ptr<server::Request> late;
        std::atomic<bool> sent{false};
        sched.on_retire = [&](const server::Scheduler::Retired& t) {
            passes.push_back(t);
            if (arriving && !sent.load()) {
                late = sched.submit(arriving->prompt, params_of(*arriving));
                sent.store(true);
            }
        };
        for (const Req& r : queued) handles.push_back(sched.submit(r.prompt, params_of(r)));
        std::thread runner([&] { sched.run(); });
        try {
            for (auto& h : handles) got.push_back(drain(*h));
            if (arriving) {
                while (!sent.load()) std::this_thread::yield();
                handles.push_back(late);
                got.push_back(drain(*late));
            }
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        return got;
    };
    const auto alone_on_one = [&](const Req& r) {
        auto model = one(pool, kWide);
        return serve(*model, tok, kSeqs, {{r}}, nullptr, 1)[0];
    };
    // The burst: every prompt slice a whole ubatch.
    {
        std::vector<Req> burst;
        for (uint32_t i = 0; i < 4; ++i) burst.push_back({prompt_of(40 + i, 512, vocab), 4});
        std::vector<server::Scheduler::Retired> passes;
        std::vector<std::shared_ptr<server::Request>> handles;
        const std::vector<Reply> got = run(burst, nullptr, passes, handles);
        for (const auto& t : passes)
            for (size_t e = 0; e < t.requests.size(); ++e)
                require(t.rows[e] == 1 || t.rows[e] == (size_t)kWide, "a burst of four prompts took a cut slice of " + std::to_string(t.rows[e]) + " rows");
        for (size_t i = 0; i < burst.size(); ++i) same(alone_on_one(burst[i]), got[i], "the burst, request " + std::to_string(i));
    }
    // The arrival mid-prompt: the lone prompt's first slice a quarter of its 768 rows, and after the second request's submission 64 rows in a pass no other prompt shares, then whole slices.
    {
        const Req lone{prompt_of(50, 768, vocab), 4}, late{prompt_of(51, 40, vocab), 4};
        std::vector<server::Scheduler::Retired> passes;
        std::vector<std::shared_ptr<server::Request>> handles;
        const std::vector<Reply> got = run({lone}, &late, passes, handles);
        std::vector<size_t> rows;
        for (const auto& t : passes)
            for (size_t e = 0; e < t.requests.size(); ++e)
                if (t.requests[e] == handles[0].get() && t.rows[e] > 1) {
                    rows.push_back(t.rows[e]);
                    if (rows.size() == 2)
                        for (size_t o = 0; o < t.requests.size(); ++o)
                            require(o == e || t.rows[o] == 1, "the slice that brought the lone prompt back to a whole ubatch shared its pass with another prompt");
                } else if (t.requests[e] == handles[1].get() && t.rows[e] > 1) {
                    require(rows.size() >= 2, "the request that arrived took a prompt slice before the lone prompt was back on a whole ubatch");
                }
        require(rows.size() == 4 && rows[0] == 192, "the lone prompt's first slice was not a quarter of its rows");
        require(rows[1] == 64 && rows[2] == (size_t)kWide && rows[3] == (size_t)kWide, "the lone prompt was not brought back to whole slices after a request arrived");
        same(alone_on_one(lone), got[0], "the lone prompt");
        same(alone_on_one(late), got[1], "the request that arrived mid-prompt");
    }
}

// The steady load at every run without logprobs, so no row is copied out of a pass's logits: every id its id alone, where the row is copied for its values, and no values.
void in_place(const Make& one, const std::function<Make(size_t)>& split, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs()) {
        const std::string what = "the steady load without logprobs on " + name(run);
        auto model = split(run.stages)(pool, kUbatch);
        std::vector<Reply> got;
        {
            server::Scheduler sched(*model, tok, kSeqs, 64, run.passes);
            std::vector<std::shared_ptr<server::Request>> handles;
            for (const Req& r : reqs) {
                server::SampleParams p = params_of(r);
                p.logprobs = false;
                p.top_logprobs = 0;
                handles.push_back(sched.submit(r.prompt, p));
            }
            std::thread runner([&] { sched.run(); });
            try {
                for (auto& h : handles) got.push_back(drain(*h));
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        for (size_t i = 0; i < reqs.size(); ++i) {
            require(got[i].size() == ref[i].size(), what + ", request " + std::to_string(i) + ": " + std::to_string(got[i].size()) + " tokens against " +
                    std::to_string(ref[i].size()) + " alone");
            for (size_t t = 0; t < got[i].size(); ++t)
                require(got[i][t].id == ref[i][t].id && got[i][t].top.empty() && got[i][t].logprob == 0.0f,
                        what + ", request " + std::to_string(i) + ": token " + std::to_string(t) + " differs from alone or carries values");
        }
    }
}

// `reqs` on `run` over a fresh model in `model`, whose last device runs `hook` with the requests and its count of submissions at each; their replies and how each ended.
struct Ended {
    std::vector<Reply> replies;
    std::vector<std::string> finish, error;
};
Ended hooked_run(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, const Run& run, size_t pool,
                 const std::function<void(const std::vector<std::shared_ptr<server::Request>>&, size_t)>& hook,
                 const std::vector<Req>& reqs, std::unique_ptr<infer::Model>& model, double share = 0) {
    const std::vector<std::shared_ptr<Hooked>> devices = hooked(run.stages, share > 0);
    model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(pool, kUbatch);
    Ended out;
    {
        server::Scheduler sched(*model, tok, kSeqs, 64, run.passes, false, 0, nullptr, 0, true, {}, server::Pacing::fixed(share));
        std::vector<std::shared_ptr<server::Request>> h;
        for (const Req& r : reqs) h.push_back(sched.submit(r.prompt, params_of(r)));
        size_t submits = 0;
        devices.back()->hook = [&] { hook(h, ++submits); };
        std::thread runner([&] { sched.run(); });
        try {
            for (auto& r : h) {
                out.replies.push_back(collect(*r));
                out.finish.push_back(r->finish());
                out.error.push_back(r->error());
            }
            ledger(sched.stats(), *model, name(run));
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    for (auto& d : devices) d->hook = nullptr;
    return out;
}

// Four capped requests, one sampled, two with prompts past the ubatch.
std::vector<Req> four(uint32_t vocab) {
    return {{prompt_of(11, 20, vocab), 200}, {prompt_of(12, 7, vocab), 200, {}, 0.9f, 5}, {prompt_of(13, 30, vocab), 200}, {prompt_of(14, 3, vocab), 200}};
}

// A request cancelled from inside the last stage's twelfth and thirteenth submissions, one of which records a pass it is in: it ends cancelled with its ids alone so far, the others with their replies whole.
// With a `share` the stages are no CPU's and fed in pieces, so a submission is a piece's or a pass apart's between two pieces.
void cancelled(const gguf::GGUFModel& weights, const Make& one, const bpe::Tokenizer& tok, uint32_t vocab, double share = 0) {
    const size_t pool = 16 * kBlock;
    const std::vector<Req> reqs = four(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs())
        for (size_t at : {12, 13}) {
            if (share > 0 && run.passes != run.stages) continue;
            const std::string what = "a request cancelled at the last stage's submission " + std::to_string(at) + " on " + name(run) + (share > 0 ? " with a share" : "");
            std::unique_ptr<infer::Model> model;
            const Ended e = hooked_run(weights, tok, run, pool, [at](const std::vector<std::shared_ptr<server::Request>>& h, size_t n) {
                if (n == at) h[0]->cancel();
            }, reqs, model, share);
            // A cancelled request's rows are dropped unread, so its tokens read after the cancel carry no values: its ids must be a prefix of its ids alone.
            require(e.finish[0] == "cancel" && e.replies[0].size() < ref[0].size(), what + ": it ended with " + e.finish[0] + " after " +
                    std::to_string(e.replies[0].size()) + " tokens");
            for (size_t t = 0; t < e.replies[0].size(); ++t)
                require(e.replies[0][t].id == ref[0][t].id, what + ": its token " + std::to_string(t) + " differs from alone");
            for (size_t i = 1; i < reqs.size(); ++i) {
                require(e.finish[i] == "length", what + ": request " + std::to_string(i) + " ended with " + e.finish[i]);
                same(ref[i], e.replies[i], what + ", request " + std::to_string(i));
            }
            whole_pool_free(*model, pool, vocab, what);
        }
}

// The last stage's twentieth submission throws, failing the pass it records: that pass's requests end with the error, and with passes in flight beside it the others run to their ends as alone.
// With a `share` the failing submission is a piece's or a pass apart's between two pieces, and fails that pass alone.
void failed(const gguf::GGUFModel& weights, const Make& one, const bpe::Tokenizer& tok, uint32_t vocab, double share = 0) {
    const size_t pool = 16 * kBlock;
    const std::vector<Req> reqs = four(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs()) {
        if (share > 0 && run.passes != run.stages) continue;
        const std::string what = "a stage failing on " + name(run) + (share > 0 ? " with a share" : "");
        std::unique_ptr<infer::Model> model;
        const Ended e = hooked_run(weights, tok, run, pool, [](const std::vector<std::shared_ptr<server::Request>>&, size_t n) {
            if (n == 20) throw std::runtime_error("injected");
        }, reqs, model, share);
        size_t errors = 0;
        for (size_t i = 0; i < reqs.size(); ++i) {
            if (e.finish[i] == "error") {
                require(e.error[i] == "injected", what + ": request " + std::to_string(i) + " ended with the error " + e.error[i]);
                same(ref[i], e.replies[i], what + ", request " + std::to_string(i) + " before the failure", true);
                ++errors;
                continue;
            }
            require(e.finish[i] == "length", what + ": request " + std::to_string(i) + " ended with " + e.finish[i]);
            same(ref[i], e.replies[i], what + ", request " + std::to_string(i));
        }
        // Four decoding requests share the passes in flight evenly once the passes fill the stages, so a failed pass takes some and leaves the rest.
        const bool shared = share > 0 || (run.passes >= run.stages && run.passes > 1);
        require(errors > 0 && (!shared || errors < reqs.size()),
                what + ": " + std::to_string(errors) + " of " + std::to_string(reqs.size()) + " requests ended with the error");
        whole_pool_free(*model, pool, vocab, what);
    }
}

// A stop from inside the first stage's twelfth submission: every request ends cancelled, and the ledger and the pool hold nothing.
void stopped(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab, double share = 0) {
    const size_t pool = 16 * kBlock;
    for (const Run& run : runs()) {
        if (share > 0 && run.passes != run.stages) continue;
        const std::string what = "a stop from inside a stage on " + name(run) + (share > 0 ? " with a share" : "");
        const std::vector<std::shared_ptr<Hooked>> devices = hooked(run.stages, share > 0);
        auto model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(pool, kUbatch);
        {
            server::Scheduler sched(*model, tok, kSeqs, 64, run.passes, false, 0, nullptr, 0, true, {}, server::Pacing::fixed(share));
            std::vector<std::shared_ptr<server::Request>> h;
            for (const Req& r : four(vocab)) h.push_back(sched.submit(r.prompt, params_of(r)));
            size_t submits = 0;
            devices.front()->hook = [&] { if (++submits == 12) sched.stop(); };
            std::thread runner([&] { sched.run(); });
            runner.join();
            devices.front()->hook = nullptr;
            for (auto& r : h) {
                collect(*r);
                require(r->finish() == "cancel", what + ": a request ended with " + r->finish());
            }
            const auto s = sched.stats();
            require(s.active == 0 && s.queued == 0 && s.paused == 0 && s.donors == 0 && s.in_flight == 0, what + ": requests, donors or passes were left once it stopped");
            for (size_t p = 0; p < s.reserved.size(); ++p) require(s.reserved[p] == 0, what + ": the ledger holds blocks once it stopped");
        }
        whole_pool_free(*model, pool, vocab, what);
    }
}

// Every case over `weights`, served one pass at a time on one CPU and with passes in flight on splits.
void cases(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const Make one = on(weights, [] { return cpus(1); });
    const auto split = [&weights](size_t stages) { return on(weights, [stages] { return cpus(stages); }); };
    // Passes in flight need a pipelined split, and one CPU runs one at a time.
    {
        auto model = one(kBlock, kUbatch);
        bool refused = false;
        try {
            server::Scheduler sched(*model, tok, kSeqs, 64, 2);
        } catch (const std::runtime_error& e) {
            refused = std::string(e.what()).find("passes in flight") != std::string::npos;
        }
        require(refused, "two passes in flight on one CPU were not refused");
    }
    paused(one, split, tok, vocab);
    replayed(one, split, tok, vocab);
    slices(one, split, tok, vocab);
    in_place(one, split, tok, vocab);
    cancelled(weights, one, tok, vocab);
    failed(weights, one, tok, vocab);
    stopped(weights, tok, vocab);
}

// The loads over tensor groups of two CPUs (docs/TENSOR-SPLIT.md, step 2): every reply on 1, 2 and 3 stages of width 2 at each P its reply alone on one group with one pass in flight, since a group gives its own bits, not one CPU's.
void grouped(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const Make one = on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2);
    const auto split = [&weights](size_t stages) { return on(weights, [stages] { return cpus(2 * stages); }, 8, 0, 0, 0, false, 2); };
    paused(one, split, tok, vocab);
    replayed(one, split, tok, vocab);
}

// Two stages of CPU groups over a routed model at P = 2 and 4: each stage's thread records routed layers, which rebuild the run list of their rows, while the other stage's thread records its own; every reply its reply alone on one group.
void routed_groups(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const std::vector<Reply> ref = alone(on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2), tok, pool, reqs);
    const Make split = on(weights, [] { return cpus(4); }, 8, 0, 0, 0, false, 2);
    for (const size_t passes : {size_t(2), size_t(4)}) {
        auto model = split(pool, kUbatch);
        server::Scheduler::Stats s;
        const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, &s, passes);
        require(s.passes == passes, "routed layers over two stages of groups: the scheduler kept " + std::to_string(s.passes) + " passes in flight");
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], "routed layers over two stages of groups at P = " + std::to_string(passes) + ", request " + std::to_string(i));
    }
}

// A layer split whose stages are no CPU's, as a split over devices is: the paused and the held load at each P, every reply its reply alone.
// Then the rule by the threads that submit: a request alone by the scheduler's thread only, the steady load by more than one (recorded_apart).
struct Submitters {
    std::mutex m;
    std::set<std::thread::id> threads;
    size_t count() {
        std::lock_guard<std::mutex> lk(m);
        return threads.size();
    }
    void clear() {
        std::lock_guard<std::mutex> lk(m);
        threads.clear();
    }
};
struct NoCpu : backend::CpuBackend {
    Submitters* seen;
    // The CPU's collective takes CPU members, which these are for as long as one is made.
    static inline thread_local bool joining = false;
    explicit NoCpu(Submitters* s) : seen(s) { set_threads(1); }
    bool is_cpu() const override { return joining; }
    // It cannot tell its free memory, so a fit does not wait for it to settle.
    std::optional<size_t> memory_available() const override { return std::nullopt; }
    std::unique_ptr<backend::Collective> join(const std::vector<backend::Backend*>& members, size_t rows, size_t width) override {
        joining = true;
        std::unique_ptr<backend::Collective> c = CpuBackend::join(members, rows, width);
        joining = false;
        return c;
    }
    backend::Ticket submit() override {
        {
            std::lock_guard<std::mutex> lk(seen->m);
            seen->threads.insert(std::this_thread::get_id());
        }
        return CpuBackend::submit();
    }
};
void recorded_stages(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const Make one = on(weights, [] { return cpus(1); });
    Submitters seen;
    const auto split = [&weights, &seen](size_t stages) {
        return on(weights, [stages, &seen] {
            std::vector<backend::BackendPtr> v;
            for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<NoCpu>(&seen));
            return v;
        });
    };
    paused(one, split, tok, vocab);
    for (const double share : {0.5, 0.25}) {
        const uint64_t between = paused(one, split, tok, vocab, share);
        require(between > 0, "stages that are no CPU's at a share of " + std::to_string(share) + ": no pass apart was recorded between a prompt pass's pieces");
        std::cout << "server-passes-cpu: at a share of " << share << " the stages recorded " << between << " passes apart between a prompt pass's pieces\n";
    }
    // The same over stages of tensor groups of two such backends, where a piece ends at a sum.
    const Make one_group = on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2);
    const auto groups = [&weights, &seen](size_t stages) {
        return on(weights, [stages, &seen] {
            std::vector<backend::BackendPtr> v;
            for (size_t i = 0; i < 2 * stages; ++i) v.push_back(std::make_shared<NoCpu>(&seen));
            return v;
        }, 8, 0, 0, 0, false, 2);
    };
    {
        const uint64_t between = paused(one_group, groups, tok, vocab, 0.5);
        require(between > 0, "stages of groups that are no CPU's at a share of one half: no pass apart was recorded between a prompt pass's pieces");
        std::cout << "server-passes-cpu: over stages of groups at a share of 0.5 the stages recorded " << between << " passes apart between a prompt pass's pieces\n";
    }
    // A request cancelled, a submission failing and a stop, each from inside a piece or a pass apart between two.
    cancelled(weights, one, tok, vocab, 0.5);
    failed(weights, one, tok, vocab, 0.5);
    stopped(weights, tok, vocab, 0.5);
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs()) {
        if (run.stages < 2 || run.passes < run.stages) continue;
        const std::string what = "stages that are no CPU's on " + name(run);
        auto model = split(run.stages)(pool, kUbatch);
        seen.clear();
        same(ref[0], serve(*model, tok, kSeqs, {{reqs[0]}}, nullptr, run.passes)[0], what + ", a request alone");
        require(seen.count() == 1, what + ": " + std::to_string(seen.count()) + " threads submitted a lone request's stages, against the scheduler's alone");
        model = split(run.stages)(pool, kUbatch);
        seen.clear();
        const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, nullptr, run.passes);
        require(seen.count() > 1, what + ": no stage of several requests' passes was recorded on a thread of its own");
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], what + ", request " + std::to_string(i));
    }
}

// A collective that counts the waits of a stage fed in pieces: for everything its last sum submitted, or with one sum left ahead.
// A sum takes `nap`, as a device's work up to it does, so a group's pass and its pieces have a time whether fed whole or in pieces.
struct WatchedSum : backend::Collective {
    std::unique_ptr<backend::Collective> sum;
    std::atomic<size_t>&idle, &ahead;
    std::chrono::milliseconds nap;
    WatchedSum(std::unique_ptr<backend::Collective> s, std::atomic<size_t>& i, std::atomic<size_t>& a, std::chrono::milliseconds n) : sum(std::move(s)), idle(i), ahead(a), nap(n) {}
    backend::Slice partial(size_t member) override { return sum->partial(member); }
    void sum_into(const std::vector<backend::Slice>& residual, size_t rows, size_t width) override {
        std::this_thread::sleep_for(nap);
        sum->sum_into(residual, rows, width);
    }
    void wait(size_t back) noexcept override {
        ++(back ? ahead : idle);
        sum->wait(back);
    }
};

// A backend that is no CPU's and takes `nap` a submission: with a nap above the automatic rule's target gap every prompt's pass outlasts the gap, so generated rows never ride one.
// It counts the submissions a thread other than its first submitter's, the scheduler's, made without having waited for its own last one: a stage's thread that holds no piece ahead makes one only after a pass of generated rows.
struct Slow : NoCpu {
    std::function<void()> piece;   // run in every submission but the first submitter's, so on a stage's thread
    static inline std::atomic<size_t> idle_sums{0}, ahead_sums{0};
    std::unique_ptr<backend::Collective> join(const std::vector<backend::Backend*>& members, size_t rows, size_t width) override {
        // In a group the time is the sums', and the members' submissions take none; every collective joined keeps the delay, the model's probe included.
        for (backend::Backend* m : members) static_cast<Slow*>(m)->grouped = true;
        return std::make_unique<WatchedSum>(NoCpu::join(members, rows, width), idle_sums, ahead_sums, nap);
    }
    std::chrono::milliseconds nap{0};
    bool grouped = false;
    std::mutex wm;
    std::map<std::thread::id, std::pair<backend::Ticket, bool>> last;   // per thread, its last ticket and whether it waited for it
    std::thread::id first;
    size_t behind = 0, threaded = 0;
    Slow(Submitters* s, int ms) : NoCpu(s), nap(ms) {}
    bool on_stage() {
        std::lock_guard<std::mutex> lk(wm);
        return !last.empty() && std::this_thread::get_id() != first;
    }
    backend::Ticket submit() override {
        if (!grouped) std::this_thread::sleep_for(nap);
        if (piece && on_stage()) piece();
        const backend::Ticket t = NoCpu::submit();
        std::lock_guard<std::mutex> lk(wm);
        const std::thread::id me = std::this_thread::get_id();
        if (last.empty()) first = me;
        const auto it = last.find(me);
        if (me != first) ++threaded;
        if (me != first && it != last.end() && !it->second.second) ++behind;
        last[me] = {t, false};
        return t;
    }
    void wait(backend::Ticket t) noexcept override {
        {
            std::lock_guard<std::mutex> lk(wm);
            const auto it = last.find(std::this_thread::get_id());
            if (it != last.end() && it->second.first == t) it->second.second = true;
        }
        NoCpu::wait(t);
    }
};

// The automatic rule (server::Pacing::automatic) on one device and on one tensor group of two, each no CPU's.
// Where replies and prompts never wait together the scheduler's thread submits every stage, as without pacing; where they do, passes of generated rows are recorded between a slow prompt pass's pieces.
void automatic(const gguf::GGUFModel& weights, uint32_t vocab, size_t only = 0) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 16 * kBlock;
    Submitters seen;
    for (const size_t width : {size_t(1), size_t(2)}) {
        if (only && width != only) continue;
        const std::string where = width == 1 ? "one device" : "a tensor group of two";
        // Each request alone on CPUs of the same width, whose sums a group's replies follow.
        const Make one = on(weights, [width] { return cpus(width); }, 8, 0, 0, 0, false, width);
        std::shared_ptr<Slow> device;
        const auto made = [&](int nap) {
            return on(weights, [&seen, &device, width, nap] {
                std::vector<backend::BackendPtr> v;
                for (size_t i = 0; i < width; ++i) v.push_back(std::make_shared<Slow>(&seen, nap));
                device = std::static_pointer_cast<Slow>(v[0]);
                return v;
            }, 8, 0, 0, 0, false, width);
        };
        // Requests one after another, each ended before the next is submitted: no reply ever waits beside a prompt.
        const std::vector<Req> together{{prompt_of(31, 5, vocab), 20}, {prompt_of(32, 4, vocab), 24, {}, 0.8f, 7}, {prompt_of(33, 6, vocab), 16}};
        std::vector<std::vector<Req>> in_turn;
        for (const Req& r : together) in_turn.push_back({r});
        const std::vector<Reply> ref = alone(one, tok, pool, together);
        for (const bool pacing : {true, false}) {
            const std::string what = std::string(pacing ? "the automatic rule" : "no pacing") + " on " + where + ", nothing overlapping";
            auto model = made(0)(pool, kUbatch);
            seen.clear();
            server::Scheduler::Stats s;
            const std::vector<Reply> got = serve(*model, tok, kSeqs, in_turn, &s, 0, 0, nullptr, 0, false, pacing ? server::Pacing::automatic() : server::Pacing{});
            for (size_t i = 0; i < together.size(); ++i) same(ref[i], got[i], what + ", request " + std::to_string(i));
            require(s.overlaps == 0, what + ": " + std::to_string(s.overlaps) + " overlaps were counted");
            require(seen.count() == 1, what + ": " + std::to_string(seen.count()) + " threads submitted the stages, against the scheduler's alone");
            require(s.automatic == pacing && s.apart == 0 && s.between == 0 && s.gaps == 0,
                    what + ": " + std::to_string(s.apart) + " passes apart, " + std::to_string(s.between) + " between pieces, " + std::to_string(s.gaps) + " gaps counted");
        }
        // A reply beside a prompt of six passes, each submission taking longer than the target gap: the reply's rows go apart and are taken between the prompt pass's pieces.
        const std::vector<Req> beside{{prompt_of(41, 4, vocab), 8}, {prompt_of(42, 96, vocab), 3}};
        const std::vector<Reply> own = alone(one, tok, pool, beside);
        for (const bool pacing : {true, false}) {
            const std::string what = std::string(pacing ? "the automatic rule" : "no pacing") + " on " + where + ", a reply beside a prompt";
            auto model = made((int)server::kServiceGapMs + 20)(pool, kUbatch);
            Slow::idle_sums = Slow::ahead_sums = 0;
            server::Scheduler::Stats s;
            const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &s, 0, 0, nullptr, 0, false, pacing ? server::Pacing::automatic() : server::Pacing{});
            for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], what + ", request " + std::to_string(i));
            if (!pacing) {
                require(s.apart == 0 && s.between == 0 && s.gaps == 0, what + ": a pass went apart or a gap was counted");
                continue;
            }
            require(s.between > 0, what + ": no pass of generated rows was recorded between the prompt pass's pieces");
            // One reply beside one prompt is one overlap, whose end no pass in flight and no round without a ready pass may fake.
            require(s.overlaps == 1, what + ": " + std::to_string(s.overlaps) + " overlaps were counted for one reply beside one prompt");
            require(s.gaps > 0 && s.gaps_late <= s.gaps && s.longest_piece_ms > 0,
                    what + ": " + std::to_string(s.gaps) + " gaps, " + std::to_string(s.gaps_late) + " above the target, longest piece " + std::to_string(s.longest_piece_ms) + " ms");
            // On one device a piece ends at a submission: the stage's thread waits for each piece before the boundary, so a pass taken there finds the device idle.
            if (width == 1) std::cout << "server-passes-cpu: the stage's thread made " << device->threaded << " submissions, " << device->behind << " behind one it had not waited for" << std::endl;
            // On a group a piece ends at a sum: every wait there is for the whole sum, none with a sum left ahead.
            require(width == 1 || (Slow::idle_sums > 0 && Slow::ahead_sums == 0), what + ": " + std::to_string(Slow::ahead_sums.load()) + " waits left a sum ahead, " +
                                                                                  std::to_string(Slow::idle_sums.load()) + " waited for the whole sum");
            require(width > 1 || 2 * device->behind < device->threaded, what + ": the stage's thread made " + std::to_string(device->behind) + " of its " + std::to_string(device->threaded) +
                                                                            " submissions behind one it had not waited for");
            std::cout << "server-passes-cpu: the automatic rule on " << where << " recorded " << s.between << " passes between pieces, " << s.gaps << " gaps, "
                      << s.gaps_late << " above the target, " << s.floor_deferred << " deferred by the floor\n";
        }
    }
    // A prompt read one token a pass, whose passes never give a row's time: fast passes are ridden throughout, past the fixed share's ninth-pass probe.
    // Passes longer than the gap, once one fed whole has shown it, are fed in pieces with the reply's rows between them.
    if (!only)
        for (const int nap : {0, (int)server::kServiceGapMs + 20}) {
            const std::string what = std::string("the automatic rule, a prompt read one token a pass, ") + (nap ? "slow passes" : "fast passes");
            const Make cpu = on(weights, [] { return cpus(1); });
            const std::vector<Req> reqs{{prompt_of(61, 4, vocab), 30}, {prompt_of(62, nap ? 6 : 40, vocab), 3}};
            std::vector<Reply> ref;
            for (const Req& r : reqs) {
                auto single = cpu(pool, 1);
                ref.push_back(serve(*single, tok, kSeqs, {{r}}, nullptr, 1)[0]);
            }
            auto model = on(weights, [&seen, nap] { return std::vector<backend::BackendPtr>{std::make_shared<Slow>(&seen, nap)}; })(pool, 1);
            server::Scheduler::Stats s;
            const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, &s, 0, 0, nullptr, 0, false, server::Pacing::automatic());
            for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], what + ", request " + std::to_string(i));
            if (nap) require(s.rode <= 2 && s.between > 0, what + ": " + std::to_string(s.rode) + " passes ridden and " + std::to_string(s.between) + " passes between pieces");
            else require(s.rode > server::RideRun::kProbe + 1 && s.apart == 0, what + ": " + std::to_string(s.rode) + " passes ridden and " + std::to_string(s.apart) + " apart");
        }
    // Requests paused after their replies began and resumed beside other replies, reading their histories again as prompt rows: each reply its reply alone.
    if (!only) {
        const Make cpu = on(weights, [] { return cpus(1); });
        const Make device = on(weights, [&seen] { return std::vector<backend::BackendPtr>{std::make_shared<NoCpu>(&seen)}; });
        const std::vector<Req> load = paused_load(vocab);
        const std::vector<Reply> ref = alone(cpu, tok, 8 * kBlock, load);
        auto model = device(8 * kBlock, kUbatch);
        server::Scheduler::Stats s;
        const std::vector<Reply> got = serve(*model, tok, kSeqs, {load}, &s, 0, 0, nullptr, 0, false, server::Pacing::automatic());
        for (size_t i = 0; i < load.size(); ++i) same(ref[i], got[i], "the automatic rule, requests paused and resumed, request " + std::to_string(i));
        require(s.pauses > 0 && s.recomputed > 0, "the paused load under the automatic rule: " + std::to_string(s.pauses) + " pauses, " + std::to_string(s.recomputed) + " rows recomputed");
    }
    // Where no stage is fed in pieces the rule is refused as the scheduler is made: on the CPU and over a layer split.
    for (const size_t stages : {size_t(1), size_t(2)}) {
        auto model = stages == 1 ? on(weights, [] { return cpus(1); })(pool, kUbatch) : on(weights, [&seen] {
            return std::vector<backend::BackendPtr>{std::make_shared<NoCpu>(&seen), std::make_shared<NoCpu>(&seen)};
        })(pool, kUbatch);
        bool refused = false;
        try {
            server::Scheduler sched(*model, tok, kSeqs, 64, 0, false, 0, nullptr, 0, true, {}, server::Pacing::automatic());
        } catch (const std::logic_error&) {
            refused = true;
        }
        require(refused, std::string("the automatic rule was taken ") + (stages == 1 ? "on the CPU" : "over a layer split"));
    }
}

// A slow device whose decode kernels hold 16 columns, so a reply's pass has columns for its drafts.
struct SlowWide : Slow {
    using Slow::Slow;
    size_t decode_columns() const override { return 16; }
};

// A proposer that drafts a known reply's own next tokens, so every draft is kept, or with `wrong` a token that is not the reply's at every third chain, which a verify refuses and a retract undoes.
// Its chain takes `work` and runs a hook as it begins.
struct Known : infer::spec::Proposer {
    std::vector<uint32_t> prompt, reply;
    std::function<void()> hook;
    std::chrono::milliseconds work{0};
    bool wrong = false;
    size_t calls = 0;
    void draft(infer::Sequence*, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out) override {
        out.clear();
        if (h.size() < prompt.size() || !std::equal(prompt.begin(), prompt.end(), h.begin())) return;
        for (size_t at = h.size() - prompt.size(); at < reply.size() && out.size() < k; ++at) out.push_back(reply[at]);
        if (wrong && calls % 3 == 0 && !out.empty()) out.back() = out.back() ? out.back() - 1 : 1;
    }
    void draft_all(Ask* asks, size_t n) override {
        ++calls;
        if (hook) hook();
        std::this_thread::sleep_for(work);
        Proposer::draft_all(asks, n);
    }
};

// A service of generated rows in an overlap is its drafts' chain and its pass, admitted once by the stage at a piece boundary with nothing owed (docs/SERVER.md, A generating request beside a prompt).
// A chain of 300 ms and a pass of 120 ms owe the floor more than a piece repays, so the next service is always wanted while something is owed: no chain may begin then, nor before the service before it was charged and repaid.
void drafted_service(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 16 * kBlock;
    const int nap = (int)server::kServiceGapMs + 20, work = 300;
    Submitters seen;
    const Make one = on(weights, [] { return cpus(1); });
    const std::vector<Req> reqs{{prompt_of(51, 4, vocab), 40}, {prompt_of(52, 128, vocab), 3}};
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    // Besides: nothing; every third chain drafting a wrong token, whose retract is a hold outside any grant; a draft that throws at the fourth chain.
    // And the reply's client leaving there, or while a service is wanted and not yet granted.
    enum class At { none, wrong, fail, cancel, unused };
    for (const At at : {At::none, At::wrong, At::fail, At::cancel, At::unused}) {
        const std::string what = std::string("a drafted service under the automatic rule") +
                                 (at == At::wrong ? ", drafts refused" : at == At::fail ? ", a draft failing" : at == At::cancel ? ", the reply cancelled" : at == At::unused ? ", a grant left unused" : "");
        std::shared_ptr<SlowWide> device;
        auto model = on(weights, [&] {
            device = std::make_shared<SlowWide>(&seen, nap);
            return std::vector<backend::BackendPtr>{device};
        }, 8, 0, 4, 4)(pool, kUbatch);
        Known known;
        known.prompt = reqs[0].prompt;
        for (const auto& t : ref[0]) known.reply.push_back(t.id);
        known.work = std::chrono::milliseconds(work);
        known.wrong = at == At::wrong;
        server::Scheduler sched(*model, tok, kSeqs, 64, 0, false, 0, &known, 3, false, {}, server::Pacing::automatic());
        std::vector<std::shared_ptr<server::Request>> h;
        std::atomic<size_t> chains{0}, ungranted{0}, owing{0}, early{0}, left{0};
        std::atomic<bool> hooked{false};
        std::chrono::steady_clock::time_point last{};
        known.hook = [&] {
            const server::Scheduler::Stats s = sched.stats();
            if (!s.overlap) return;
            const size_t n = ++chains;
            if (!s.service_granted) ++ungranted;
            if (s.owed_ms > 0) ++owing;
            // The service before took its chain and its pass, and half of that was owed and repaid by pieces at half their time: two services' time at least between two chains.
            const auto now = std::chrono::steady_clock::now();
            // The chain that threw did no work, so the one after it is not held to that time.
            if (n > 1 && !(at == At::fail && n == 5) && now - last < std::chrono::milliseconds(2 * (work + nap) * 9 / 10)) ++early;
            last = now;
            if (n == 4 && at == At::cancel) {
                h[0]->cancel();
                hooked = true;
            }
            if (n == 4 && at == At::fail) {
                hooked = true;
                throw std::runtime_error("injected draft failure");
            }
        };
        // The client leaves from the stage's thread, during the piece that repays what is owed, with a service wanted after the second chain.
        // The grant at that piece's end then finds nothing to form, unless the prompt's pass ended there and the wish was withdrawn first.
        device->piece = [&] {
            if (at != At::unused || hooked || chains < 2) return;
            const server::Scheduler::Stats s = sched.stats();
            if (!s.service_wanted || s.service_granted || s.owed_ms > (1 - server::kPromptFloor) * nap) return;
            h[0]->cancel();
            left = chains.load();
            hooked = true;
        };
        for (const Req& r : reqs) h.push_back(sched.submit(r.prompt, params_of(r)));
        std::thread runner([&] { sched.run(); });
        const bool leaves = at == At::cancel || at == At::unused;
        std::vector<Reply> got;
        for (auto& r : h) got.push_back(leaves && r == h[0] ? collect(*r) : drain(*r));
        const server::Scheduler::Stats s = sched.stats();
        sched.stop();
        runner.join();
        device->piece = nullptr;
        same(ref[0], got[0], what + ", the reply", leaves);
        same(ref[1], got[1], what + ", the prompt's request");
        require(chains > 1 && ungranted == 0 && owing == 0 && early == 0, what + ": of " + std::to_string(chains.load()) + " chains in the overlap " + std::to_string(ungranted.load()) +
                                                                            " began without the stage's grant, " + std::to_string(owing.load()) + " with something owed and " +
                                                                            std::to_string(early.load()) + " before the service before it was repaid");
        // Each granted pass and its chain were charged whole, the pass before every grant among them, and a wish was deferred while they were owed.
        require(s.services > 0 && s.services <= s.grants && s.uncharged_at_grant == 0 && s.charged_ms >= (double)s.services * (work + nap) * 0.9 && s.floor_deferred > 0,
                what + ": " + std::to_string(s.services) + " granted passes of " + std::to_string(s.grants) + " grants, " + std::to_string(s.charged_ms) + " ms charged, " +
                    std::to_string(s.uncharged_at_grant) + " grants before a charge, " + std::to_string(s.floor_deferred) + " wishes deferred");
        size_t kept = 0;
        for (const size_t k : s.kept) kept += k;
        require(kept > 0, what + ": no draft was kept");
        require(at == At::none || at == At::wrong || hooked, what + ": the hook never fired");
        require(!leaves || h[0]->finish() == "cancel", what + ": the reply ended with " + h[0]->finish());
        require(at != At::fail || s.draft_failures == 1, what + ": " + std::to_string(s.draft_failures) + " draft failures counted");
        // A grant given for a request that left forms nothing: at most one more grant than passes, and no chain after the client left.
        if (at == At::unused) std::cout << "server-passes-cpu: the client left with a service wanted: " << s.grants - s.services << " grant left unused" << std::endl;
        require(at != At::unused || (s.grants <= s.services + 1 && chains == left), what + ": " + std::to_string(s.grants) + " grants for " + std::to_string(s.services) + " passes, " +
                                                                                    std::to_string(chains.load()) + " chains against " + std::to_string(left.load()) + " as the client left");
        if (at == At::none)
            std::cout << "server-passes-cpu: a drafted service: " << chains << " chains in the overlap, each on a grant with nothing owed, " << kept << " drafts kept, " << s.services << " of " << s.grants
                      << " grants recorded, " << s.floor_deferred << " wishes deferred" << std::endl;
    }
}

// A scheduler whose stages have threads, made and destroyed at once, many times: a stage's thread that first runs as the scheduler is destroyed must still find its own record.
// It needs no assertion: the fault it guards is a crash, which the C++ library of the macOS build shows and those of the Linux and Windows builds hide (docs/STATUS.md, a stage's thread is given its record).
void made_and_destroyed(const gguf::GGUFModel& weights) {
    const bpe::Tokenizer tok(weights);
    Submitters seen;
    auto model = on(weights, [&seen] { return std::vector<backend::BackendPtr>{std::make_shared<NoCpu>(&seen)}; })(16 * kBlock, kUbatch);
    for (size_t i = 0; i < 400; ++i) server::Scheduler sched(*model, tok, kSeqs, 64, 0, false, 0, nullptr, 0, true, {}, server::Pacing::automatic());
    ++checks;
}

// A device that reports `room` bytes free and counts what is allocated on it.
struct Metered : NoCpu {
    size_t room = 0;
    std::atomic<size_t> taken{0};
    using NoCpu::NoCpu;
    std::optional<size_t> memory_available() const override { return room; }
    size_t decode_columns() const override { return 16; }
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        taken += bytes;
        return NoCpu::alloc(bytes, where);
    }
};

// A model fitted to a device's free memory as the command line asks for a server (`open_model`), then a scheduler made over it: the bytes both allocated on the device, and the bytes of one KV block.
// Without `arena` the request leaves the draft rows out of a pass's rows, and without `logits` out of its logits rows, each as before the fix.
std::pair<size_t, size_t> fitted_server(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, size_t room, bool drafting, bool arena = true, bool logits = true) {
    Submitters seen;
    auto device = std::make_shared<Metered>(&seen);
    device->room = room;
    const size_t seqs = 16, slots = 1;
    infer::PlacementRequest request;
    request.names = {"device"};
    request.fit_kv = true;
    request.slots = slots;
    request.decode_rows = seqs;
    request.draft_rows = arena ? server::Scheduler::draft_rows(drafting) : 0;
    request.logit_rows = server::logit_rows(slots, logits ? server::Scheduler::logits_a_pass(seqs, drafting) : seqs).size;
    request.apart_rows = server::Scheduler::apart_rows(seqs, drafting);
    infer::ModelOptions options;
    options.kv_tokens = size_t(1) << 20;
    options.mark_slots = drafting ? 4 : 0;
    options.mark_rows = drafting ? 4 : 0;
    auto model = infer::place_model(infer::gguf_weights(weights), {device}, request, options).model;
    infer::spec::Lookup lookup;
    server::Scheduler sched(*model, tok, seqs, 64, 0, false, 0, drafting ? &lookup : nullptr, drafting ? 3 : 0, false, {}, server::Pacing::automatic());
    return {device->taken.load(), model->kv_allocated_bytes() / (model->kv_tokens_total() / kBlock)};
}
// A drafter's rows come out of the fitted KV budget and not on top of it: fitted to the same free memory, a drafting server allocates within the room and no more than one without a drafter.
// Fitted without the draft rows in a pass's rows, in its logits rows, or in both as before the fix, it allocates more, so each half of the fix is seen alone.
void fitted_reservation(const gguf::GGUFModel& weights) {
    const bpe::Tokenizer tok(weights);
    // A room just below a KV block's edge: the logits rows alone are a quarter of a block on this model, and are seen only where the room leaves less than that.
    const size_t room = (size_t(24) << 20) + (size_t(36) << 10);
    const auto plain = fitted_server(weights, tok, room, false), drafting = fitted_server(weights, tok, room, true), before = fitted_server(weights, tok, room, true, false, false);
    const auto no_arena = fitted_server(weights, tok, room, true, false, true), no_logits = fitted_server(weights, tok, room, true, true, false);
    const std::string what = "a server fitted to " + std::to_string(room) + " bytes: " + std::to_string(plain.first) + " allocated without a drafter, " + std::to_string(drafting.first) +
                             " with one, " + std::to_string(before.first) + " with the draft rows left out of the fit, " + std::to_string(no_arena.first) + " out of a pass's rows alone, " +
                             std::to_string(no_logits.first) + " out of its logits rows alone, a KV block " + std::to_string(plain.second);
    // The drafts' rows, whole blocks together on this model, leave the fitted server at the bytes of one without a drafter.
    require(plain.first <= room && drafting.first <= room && drafting.first <= plain.first, what);
    require(before.first > plain.first + plain.second && no_arena.first > plain.first + plain.second && no_logits.first > plain.first, what + "; the comparison does not tell the fits apart");
}

// A CPU backend whose decode kernels hold 16 columns, so a pass of several decoding requests has columns to spare for their drafts.
// It counts its submissions in a plain member, as a device keeps its open command buffer, so a submission beside a recording is a data race the thread sanitizer reports.
struct Wide : backend::CpuBackend {
    size_t submissions = 0;
    Wide() { set_threads(1); }
    size_t decode_columns() const override { return 16; }
    backend::Ticket submit() override {
        ++submissions;
        return CpuBackend::submit();
    }
};

// Drafting over two stages of tensor groups at two passes, each stage recorded on its own thread (docs/TENSOR-SPLIT.md, step 5): a hybrid model's lookup-draft verifies give the replies they give alone without drafts on one group.
// Each member saves a marked entry's recurrent inputs on its stage's thread and reruns its state on a retract, called with that thread idle.
void drafted_groups(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    std::vector<Req> reqs;
    for (uint32_t r = 0; r < 3; ++r) {
        std::vector<uint32_t> prompt = prompt_of(10 + r, 60, vocab);
        prompt.insert(prompt.end(), {prompt[0], prompt[1], prompt[2]});
        reqs.push_back({prompt, 48});
    }
    const std::vector<Reply> ref = alone(on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2), tok, pool, reqs);
    const Make make = on(weights, [] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 4; ++i) v.push_back(std::make_shared<Wide>());
        return v;
    }, 8, 0, 4, 4, false, 2);
    auto model = make(pool, kUbatch);
    infer::spec::Lookup lookup;
    server::Scheduler::Stats stats;
    const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, &stats, 2, 0, &lookup, 3);
    for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], "drafts over two stages of hybrid groups, request " + std::to_string(i));
    size_t drafted = 0, kept = 0;
    for (size_t n : stats.drafted) drafted += n;
    for (size_t n : stats.kept) kept += n;
    require(stats.passes == 2 && drafted > 0 && kept > 0,
            "drafts over two stages of hybrid groups: " + std::to_string(stats.passes) + " passes in flight, " + std::to_string(kept) + " of " + std::to_string(drafted) + " drafts kept");
}

// The same placement drafting with the file's embedded drafter, whose chain is work on the head's group that the scheduler's thread does while the last stage's thread may be recording there: every reply its reply alone without a drafter.
void chained_groups(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    std::vector<Req> reqs;
    for (uint32_t r = 0; r < 3; ++r) reqs.push_back({prompt_of(10 + r, 60, vocab), 48});
    const std::vector<Reply> ref = alone(on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2), tok, pool, reqs);
    const Make make = on(weights, [] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 4; ++i) v.push_back(std::make_shared<Wide>());
        return v;
    }, 8, 0, 4, 4, true, 2);
    auto model = make(pool, kUbatch);
    infer::spec::Embedded drafter(*model);
    server::Scheduler::Stats stats;
    const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, &stats, 2, 0, &drafter, 3);
    for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], "the embedded drafter over two stages of hybrid groups, request " + std::to_string(i));
    size_t drafted = 0;
    for (size_t n : stats.drafted) drafted += n;
    require(stats.passes == 2 && drafted > 0, "the embedded drafter over two stages of hybrid groups: " + std::to_string(stats.passes) + " passes in flight, " + std::to_string(drafted) + " drafts");
}

// A CPU backend that reports a device time, as a timed device does, and notes whether that reading ever met a submission of its own: a device takes its timing read from the thread that records on it and from no other while it records.
struct Timed : backend::CpuBackend {
    std::atomic<bool> submitting{false}, reading{false};
    std::atomic<size_t>* met = nullptr;
    backend::Ticket submit() override {
        submitting = true;
        if (reading) ++*met;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        const backend::Ticket t = CpuBackend::submit();
        submitting = false;
        return t;
    }
    double device_ms() override {
        reading = true;
        if (submitting) ++*met;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        reading = false;
        return 0.0;
    }
};

// A timed scheduler (serve --timing) over two stages of tensor groups with two passes in flight: no reading of a stage's device time may meet a recording on its devices, and every reply stays its reply alone.
void timed_groups(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const std::vector<Reply> ref = alone(on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2), tok, pool, reqs);
    std::atomic<size_t> met{0};
    const Make make = on(weights, [&met] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 4; ++i) {
            auto c = std::make_shared<Timed>();
            c->set_threads(1);
            c->met = &met;
            v.push_back(c);
        }
        return v;
    }, 8, 0, 0, 0, false, 2);
    auto model = make(pool, kUbatch);
    std::vector<std::shared_ptr<server::Request>> handles;
    std::vector<Reply> got;
    {
        server::Scheduler sched(*model, tok, kSeqs, 64, 2, true);
        for (const Req& r : reqs) handles.push_back(sched.submit(r.prompt, params_of(r)));
        std::thread runner([&] { sched.run(); });
        try {
            for (auto& h : handles) got.push_back(drain(*h));
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], "a timed scheduler over two stages of groups, request " + std::to_string(i));
    require(met == 0, "a timed scheduler read a stage's device time " + std::to_string(met.load()) + " times while that stage was being recorded");
    ++checks;
}

// A timed scheduler (serve --timing) on the CPU alone and split over two CPUs: stages on the host keep a time from the first pass on, every reply is its reply alone and the timing counts its rounds.
void timed_cpu(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const Make one = on(weights, [] { return cpus(1); });
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const size_t stages : {size_t(1), size_t(2)}) {
        const std::string what = "a timed scheduler on " + std::to_string(stages) + " CPU stage" + (stages > 1 ? "s" : "");
        auto model = on(weights, [stages] { return cpus(stages); })(pool, kUbatch);
        std::vector<std::shared_ptr<server::Request>> handles;
        std::vector<Reply> got;
        server::Scheduler::Stats stats;
        {
            server::Scheduler sched(*model, tok, kSeqs, 64, stages, true);
            for (const Req& r : reqs) handles.push_back(sched.submit(r.prompt, params_of(r)));
            std::thread runner([&] { sched.run(); });
            try {
                for (auto& h : handles) got.push_back(drain(*h));
                stats = sched.stats();
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
        }
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], what + ", request " + std::to_string(i));
        require(stats.timed && stats.timing.rounds > 0 && stats.timing.stage_ms.size() == stages, what + ": the timing holds " + std::to_string(stats.timing.rounds) + " rounds and " + std::to_string(stats.timing.stage_ms.size()) + " stages");
        for (double ms : stats.timing.stage_ms) require(std::isfinite(ms) && ms >= 0, what + ": a stage's time is " + std::to_string(ms));
        ++checks;
    }
}

} // namespace

// The room rule at its owner: the scheduler refuses, without a route before it, a prompt that leaves its reply less than the room, and admits one at the edge.
void reply_room(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    auto model = on(weights, [] { return cpus(1); })(8 * kBlock, kUbatch);
    server::Scheduler sched(*model, tok, kSeqs, 64);
    const size_t limit = sched.token_limit(), room = limit - limit / server::Scheduler::kReplyRoom;
    server::SampleParams open, capped;
    open.until_limit = true;
    capped.max_tokens = 100;
    require(sched.prompt_room(open) == room && sched.prompt_room(capped) == limit - 100, "the rows a prompt may take beside its reply: " + std::to_string(sched.prompt_room(open)) + " of " + std::to_string(limit));
    const auto refused = [&](size_t n, const server::SampleParams& p) {
        try {
            sched.submit(prompt_of(71, n, vocab), p);
        } catch (const server::TooLong& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    const std::string why = refused(room + 1, open);
    require(!why.empty() && why.find(std::to_string(room + 1) + " tokens") != std::string::npos && why.find("leaves " + std::to_string(limit - room - 1) + " for its reply") != std::string::npos &&
                why.find(std::to_string(limit - room)) != std::string::npos,
            "an uncapped prompt a row past its room was admitted, or refused without its numbers: " + why);
    require(refused(room, open).empty() && refused(limit - 100, capped).empty() && !refused(limit - 99, capped).empty(), "the room's edge, uncapped and capped, was not held at submit");
}

int main(int argc, char** argv) {
    try {
        // What a caller asks the fit for is what the scheduler reserves: a row a request and, where requests draft, a verify's draft rows, in each logits ring.
        require(server::Scheduler::logits_a_pass(16, false) == 16 && server::Scheduler::logits_a_pass(16, true) == 16 + server::Scheduler::draft_rows(true) &&
                    server::Scheduler::draft_rows(false) == 0 && server::Scheduler::draft_rows(true) == 64 && server::Scheduler::apart_rows(16, true) == server::Scheduler::logits_a_pass(16, true),
                "the logits rows of a pass with and without drafts");
        reply_room(served(kSplit), (uint32_t)kSplit.vocab);
        // `automatic-group` runs the automatic rule's cases on the tensor group alone.
        if (argc > 1 && std::string(argv[1]) == "automatic-group") {
            automatic(served(kSplit), (uint32_t)kSplit.vocab, 2);
            return 0;
        }
        grouped(served(kSplit), (uint32_t)kSplit.vocab);
        routed_groups(served_routed(kSplit), (uint32_t)kSplit.vocab);
        grouped(served_hybrid(kHybridEven), (uint32_t)kHybridEven.vocab);
        drafted_groups(served_hybrid(kHybridEven), (uint32_t)kHybridEven.vocab);
        chained_groups(served_hybrid(kHybridEven, true), (uint32_t)kHybridEven.vocab);
        timed_groups(served(kSplit), (uint32_t)kSplit.vocab);
        timed_cpu(served(kSplit), (uint32_t)kSplit.vocab);
        recorded_stages(served(kSplit), (uint32_t)kSplit.vocab);
        made_and_destroyed(served(kSplit));
        automatic(served(kSplit), (uint32_t)kSplit.vocab);
        drafted_service(served(kSplit), (uint32_t)kSplit.vocab);
        fitted_reservation(served(kSplit));
        cases(served(kSplit), (uint32_t)kSplit.vocab);
        // A hybrid model, whose linear-attention layers keep a recurrent state, over the same cases: without checkpoint slots it keeps no donor, and a stage may hold only states.
        cases(served_hybrid(kHybrid), (uint32_t)kHybrid.vocab);
        std::cout << "server-passes-cpu: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-passes-cpu: " << e.what() << '\n';
        return 1;
    }
}
