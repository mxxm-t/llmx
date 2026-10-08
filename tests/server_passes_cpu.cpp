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

// The paused load and the held load at every run: each reply its reply alone, the paused load pausing a request on every run and the held load's older request waiting on the younger in flight at P = S.
void paused(const Make& one, const std::function<Make(size_t)>& split, const bpe::Tokenizer& tok, uint32_t vocab) {
    const std::vector<Req> reqs = paused_load(vocab), held = held_load(vocab);
    const std::vector<Reply> ref = alone(one, tok, 8 * kBlock, reqs), held_ref = alone(one, tok, 9 * kBlock, held);
    for (const Run& run : runs()) {
        auto model = split(run.stages)(8 * kBlock, kUbatch);
        server::Scheduler::Stats s;
        const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, &s, run.passes);
        require(s.passes == run.passes, name(run) + ": the scheduler kept " + std::to_string(s.passes) + " passes in flight");
        require(s.pauses > 0, "the paused load on " + name(run) + ": nothing paused");
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], "the paused load on " + name(run) + ", request " + std::to_string(i));
        model = split(run.stages)(9 * kBlock, kUbatch);
        const std::vector<Reply> kept = serve(*model, tok, kSeqs, {held}, &s, run.passes);
        require(s.pauses == 1 && (run.passes != run.stages || run.stages == 1 || s.waits > 0),
                "the held load on " + name(run) + ": " + std::to_string(s.pauses) + " pauses and " + std::to_string(s.waits) +
                " plans waiting on a request in flight, against 1 and some at P = S");
        for (size_t i = 0; i < held.size(); ++i) same(held_ref[i], kept[i], "the held load on " + name(run) + ", request " + std::to_string(i));
    }
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
                 const std::vector<Req>& reqs, std::unique_ptr<infer::Model>& model) {
    const std::vector<std::shared_ptr<Hooked>> devices = hooked(run.stages);
    model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(pool, kUbatch);
    Ended out;
    {
        server::Scheduler sched(*model, tok, kSeqs, 64, run.passes);
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
void cancelled(const gguf::GGUFModel& weights, const Make& one, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    const std::vector<Req> reqs = four(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs())
        for (size_t at : {12, 13}) {
            const std::string what = "a request cancelled at the last stage's submission " + std::to_string(at) + " on " + name(run);
            std::unique_ptr<infer::Model> model;
            const Ended e = hooked_run(weights, tok, run, pool, [at](const std::vector<std::shared_ptr<server::Request>>& h, size_t n) {
                if (n == at) h[0]->cancel();
            }, reqs, model);
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
void failed(const gguf::GGUFModel& weights, const Make& one, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    const std::vector<Req> reqs = four(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs()) {
        const std::string what = "a stage failing on " + name(run);
        std::unique_ptr<infer::Model> model;
        const Ended e = hooked_run(weights, tok, run, pool, [](const std::vector<std::shared_ptr<server::Request>>&, size_t n) {
            if (n == 20) throw std::runtime_error("injected");
        }, reqs, model);
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
        const bool shared = run.passes >= run.stages && run.passes > 1;
        require(errors > 0 && (!shared || errors < reqs.size()),
                what + ": " + std::to_string(errors) + " of " + std::to_string(reqs.size()) + " requests ended with the error");
        whole_pool_free(*model, pool, vocab, what);
    }
}

// A stop from inside the first stage's twelfth submission: every request ends cancelled, and the ledger and the pool hold nothing.
void stopped(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    for (const Run& run : runs()) {
        const std::string what = "a stop from inside a stage on " + name(run);
        const std::vector<std::shared_ptr<Hooked>> devices = hooked(run.stages);
        auto model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(pool, kUbatch);
        {
            server::Scheduler sched(*model, tok, kSeqs, 64, run.passes);
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

// A layer split whose stages are no CPU's, as a split over devices is: with several passes in flight a stage is recorded on a thread of its own (Model::stage_waits) while another pass is in flight, and on the scheduler's thread while its pass is alone.
// The paused and the held load on 1, 2 and 3 such stages at each P, every reply its reply alone on one CPU, whose bits a layer split gives.
// Then the rule itself, by the threads that submit to the backends: one request alone is submitted by one thread, the scheduler's, and the steady load at P = S and above by more than one, every reply its reply alone.
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
    explicit NoCpu(Submitters* s) : seen(s) { set_threads(1); }
    bool is_cpu() const override { return false; }
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

// A CPU backend whose decode kernels hold 16 columns, so a pass of several decoding requests has columns to spare for their drafts.
// It counts its submissions in a plain member, as a device keeps its open command buffer: a submission by one thread while another records on it, which the recorder rule forbids (backends/backend.hpp, wait), is then a data race the thread sanitizer reports, where the CPU's own eager work shares nothing.
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

int main() {
    try {
        grouped(served(kSplit), (uint32_t)kSplit.vocab);
        routed_groups(served_routed(kSplit), (uint32_t)kSplit.vocab);
        grouped(served_hybrid(kHybridEven), (uint32_t)kHybridEven.vocab);
        drafted_groups(served_hybrid(kHybridEven), (uint32_t)kHybridEven.vocab);
        chained_groups(served_hybrid(kHybridEven, true), (uint32_t)kHybridEven.vocab);
        timed_groups(served(kSplit), (uint32_t)kSplit.vocab);
        timed_cpu(served(kSplit), (uint32_t)kSplit.vocab);
        recorded_stages(served(kSplit), (uint32_t)kSplit.vocab);
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
