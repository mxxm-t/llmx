// The scheduler with passes in flight (docs/SERVER.md, the round) over the synthetic Q8_0 model and a hybrid one, on one CPU and split over two and three, at P = 1, S, S + 1 and 2S, each request equal to its run alone.
// Cancels, stops and failures from inside a stage, and the replay of a never-pausing load through Model::forward, are checked too, each leaving every block free (AGENTS.md, Tests).
#include <thread>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <condition_variable>
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

// The paused load and the held load at every run, each reply its reply alone.
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

// A burst of four 512-token prompts on two CPU stages with a ubatch of 256 takes whole ubatches, every reply its reply alone.
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
        for (uint32_t i = 0; i < 4; ++i) burst.push_back({prompt_of(40 + i, 512, vocab), 1});
        std::vector<server::Scheduler::Retired> passes;
        std::vector<std::shared_ptr<server::Request>> handles;
        const std::vector<Reply> got = run(burst, nullptr, passes, handles);
        // The last of the burst may be read alone once the others have ended, and is then cut as a lone prompt is.
        for (const auto& t : passes)
            for (size_t e = 0; e < t.requests.size(); ++e)
                require(t.requests[e] == handles.back().get() || t.rows[e] == 1 || t.rows[e] == (size_t)kWide, "a burst of four prompts took a cut slice of " + std::to_string(t.rows[e]) + " rows");
        for (size_t i = 0; i < burst.size(); ++i) same(alone_on_one(burst[i]), got[i], "the burst, request " + std::to_string(i));
    }
    // A lone prompt: its first slice a quarter of it so both stages read while it is the only work, every slice after at least 128 rows but its last, at most a ubatch.
    {
        const std::vector<Req> lone{{prompt_of(44, 768, vocab), 1}};
        std::vector<server::Scheduler::Retired> passes;
        std::vector<std::shared_ptr<server::Request>> handles;
        const std::vector<Reply> got = run(lone, nullptr, passes, handles);
        std::vector<size_t> cut;
        for (const auto& t : passes)
            for (size_t e = 0; e < t.requests.size(); ++e) cut.push_back(t.rows[e]);
        require(!cut.empty() && cut[0] == 192, "a lone 768-token prompt's first slice took " + std::to_string(cut.empty() ? 0 : cut[0]) + " rows, against 192");
        for (size_t i = 0; i + 1 < cut.size(); ++i) require(cut[i] >= 128 && cut[i] <= (size_t)kWide, "a lone prompt's slice " + std::to_string(i) + " took " + std::to_string(cut[i]) + " rows");
        same(alone_on_one(lone[0]), got[0], "the lone prompt");
    }
}

// Short prompts beside a long one (docs/SERVER.md): a 60-token prompt arriving behind a 2000-token one, on one CPU and over two and three at P = S and 2S, is read before the long prompt is,
// and both replies are their replies alone.
void short_beside_long(const std::function<Make(size_t)>& split, const Make& one, const bpe::Tokenizer& tok, uint32_t vocab) {
    constexpr int kWide = 256;
    const size_t pool = 32 * kBlock;
    const Req long_one{prompt_of(60, 2000, vocab), 4}, short_one{prompt_of(61, 60, vocab), 4};
    const auto alone_on_one = [&](const Req& r) {
        auto model = one(pool, kWide);
        return serve(*model, tok, kSeqs, {{r}}, nullptr, 1)[0];
    };
    const Reply own_long = alone_on_one(long_one), own_short = alone_on_one(short_one);
    for (const size_t stages : {size_t(1), size_t(2), size_t(3)})
        for (const size_t passes : {stages, 2 * stages}) {
            if (stages == 1 && passes > 1) continue;
            const std::string where = "a short prompt behind a long one on " + std::to_string(stages) + " CPU" + (stages == 1 ? "" : "s") + " at P = " + std::to_string(passes);
            auto model = split(stages)(pool, kWide);
            server::Scheduler sched(*model, tok, kSeqs, 64, passes);
            std::vector<server::Scheduler::Retired> retired;
            std::shared_ptr<server::Request> late;
            std::atomic<bool> sent{false};
            // The short prompt is submitted on the scheduler thread as the long prompt's first pass retires, so the long one is mid-prompt.
            sched.on_retire = [&](const server::Scheduler::Retired& t) {
                retired.push_back(t);
                if (!sent.load()) {
                    late = sched.submit(short_one.prompt, params_of(short_one));
                    sent.store(true);
                }
            };
            auto first = sched.submit(long_one.prompt, params_of(long_one));
            std::thread runner([&] { sched.run(); });
            Reply got_long, got_short;
            try {
                got_long = drain(*first);
                while (!sent.load()) std::this_thread::yield();
                got_short = drain(*late);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
            // The pass that ends each prompt: the short one's must retire before the long one's.
            size_t long_end = retired.size(), short_end = retired.size();
            for (size_t p = 0; p < retired.size(); ++p)
                for (size_t e = 0; e < retired[p].requests.size(); ++e) {
                    if (retired[p].from[e] + retired[p].rows[e] != retired[p].requests[e]->prompt_tokens()) continue;
                    if (retired[p].requests[e] == first.get()) long_end = std::min(long_end, p);
                    if (retired[p].requests[e] == late.get()) short_end = std::min(short_end, p);
                }
            require(short_end < long_end && long_end < retired.size(), where + ": the short prompt ended in pass " + std::to_string(short_end) + ", not before the long one's " + std::to_string(long_end));
            same(own_long, got_long, where + ", the long prompt");
            same(own_short, got_short, where + ", the short prompt");
            ++checks;
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
    uint64_t continued = 0;   // the scheduler's passes that continued a prompt (Stats::continued)
};
Ended hooked_run(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, const Run& run, size_t pool,
                 const std::function<void(const std::vector<std::shared_ptr<server::Request>>&, size_t)>& hook,
                 const std::vector<Req>& reqs, std::unique_ptr<infer::Model>& model) {
    const std::vector<std::shared_ptr<Hooked>> devices = hooked(run.stages, false);
    model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(pool, kUbatch);
    Ended out;
    {
        server::Scheduler sched(*model, tok, kSeqs, 64, run.passes, false, 0, nullptr, 0, true, {});
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
            const auto work = sched.stats();
            out.continued = work.continued;
            ledger(work, *model, name(run));
            uint64_t published = 0;
            for (const Reply& reply : out.replies) published += reply.size();
            require(work.published_tokens == published, "cancelled or failed passes changed published output accounting");
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

// A lone prompt whose slice fails at the last stage while the slice after it is in flight at an earlier one: the request ends with the error and the pool comes back whole.
// The last stage takes 2 ms a submission, so a slice waits there and the next continues behind it.
void failed_continued(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    const std::vector<Req> lone{{prompt_of(7, 300, vocab), 3}};
    for (const Run& run : runs()) {
        if (run.stages < 2 || run.passes < run.stages) continue;
        for (size_t at : {size_t(2), size_t(3)}) {
            const std::string what = "a continued prompt's slice failing at the last stage's submission " + std::to_string(at) + " on " + name(run);
            std::unique_ptr<infer::Model> model;
            const Ended e = hooked_run(weights, tok, run, pool, [at](const std::vector<std::shared_ptr<server::Request>>&, size_t n) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                if (n == at) throw std::runtime_error("injected");
            }, lone, model);
            require(e.finish[0] == "error" && e.error[0] == "injected", what + ": it ended with " + e.finish[0]);
            require(e.continued > 0, what + ": no slice continued another, so none was in flight beside the failing one");
            whole_pool_free(*model, pool, vocab, what);
        }
    }
}

// A stop from inside the first stage's twelfth submission: every request ends cancelled, and the ledger and the pool hold nothing.
void stopped(const gguf::GGUFModel& weights, const bpe::Tokenizer& tok, uint32_t vocab) {
    const size_t pool = 16 * kBlock;
    for (const Run& run : runs()) {
        const std::string what = "a stop from inside a stage on " + name(run);
        const std::vector<std::shared_ptr<Hooked>> devices = hooked(run.stages, false);
        auto model = on(weights, [&devices] { return std::vector<backend::BackendPtr>(devices.begin(), devices.end()); })(pool, kUbatch);
        {
            server::Scheduler sched(*model, tok, kSeqs, 64, run.passes, false, 0, nullptr, 0, true, {});
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

// Work advances while requests are running, cached rows are skipped, and a split counts each row and published token once.
void live_work(const Make& one, const std::function<Make(size_t)>& split, const bpe::Tokenizer& tok, uint32_t vocab) {
    size_t callback_checks = 0;
    std::string callback_error;
    const auto callback_require = [&](bool ok, const std::string& what) {
        ++callback_checks;
        if (!ok && callback_error.empty()) callback_error = what;
        return ok;
    };
    {
        auto model = one(32 * kBlock, kUbatch);
        server::Scheduler sched(*model, tok, 1, 64);
        const Req request{prompt_of(3, 5, vocab), 1};
        const auto h = sched.submit(request.prompt, params_of(request));
        size_t retired = 0;
        sched.on_retire = [&](const server::Scheduler::Retired&) {
            ++retired;
            const auto s = sched.stats();
            callback_require(!s.live.total && !s.live.count && !s.live.sample_elapsed_ms &&
                             !s.live.prefilled_tokens && !s.live.published_tokens, "short request was sampled before demand");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        };
        std::thread runner([&] { sched.run(); });
        server::Scheduler::LiveRequests final, repeated;
        try {
            drain(*h);
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            do {
                final = sched.stats().live;
                if (final.published_tokens == 1) break;
                require(std::chrono::steady_clock::now() < until, "short completed request never reached its empty live snapshot");
                std::this_thread::yield();
            } while (true);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            repeated = sched.stats().live;
        } catch (...) {
            sched.stop(); runner.join(); throw;
        }
        sched.stop(); runner.join();
        require(retired == 1 && !final.total && !final.count && final.sample_elapsed_ms > 0 &&
                final.prefilled_tokens == request.prompt.size() && final.published_tokens == 1,
                "short request's final snapshot lost completed work or its sample time");
        require(final.sample_elapsed_ms == repeated.sample_elapsed_ms && final.prefilled_tokens == repeated.prefilled_tokens &&
                final.published_tokens == repeated.published_tokens, "idle polls changed the short request's final sample");
    }
    for (const Run run : {Run{1, 1}, Run{2, 2}}) {
        auto model = run.stages == 1 ? one(32 * kBlock, kUbatch) : split(run.stages)(32 * kBlock, kUbatch);
        server::Scheduler sched(*model, tok, kSeqs, 64, run.passes);
        const auto empty = sched.stats();
        require(empty.prefilled_tokens == 0 && empty.published_tokens == 0 &&
                empty.live.prefilled_tokens == 0 && empty.live.published_tokens == 0, "new scheduler has nonzero work");
        bool prompt_running = false, output_running = false, monotonic = true;
        bool live_prompt = false, live_output = false, live_reuse = false;
        std::map<uint64_t, server::Scheduler::RequestView> observed;
        uint64_t sampled = 0;
        uint64_t last_prompt = 0, last_output = 0;
        sched.on_retire = [&](const server::Scheduler::Retired&) {
            const auto s = sched.stats();
            monotonic = monotonic && s.prefilled_tokens >= last_prompt && s.published_tokens >= last_output;
            last_prompt = s.prefilled_tokens;
            last_output = s.published_tokens;
            prompt_running = prompt_running || (s.finished == 0 && s.prefilled_tokens > 0);
            output_running = output_running || (s.finished == 0 && s.published_tokens > 0);
            if (!callback_require(s.live.count <= s.live.limit && s.live.count <= s.live.total, "live request snapshot exceeds its bound")) return;
            callback_require(s.live.prefilled_tokens <= s.prefilled_tokens && s.live.published_tokens <= s.published_tokens,
                    "snapshot work is ahead of current cumulative work");
            const auto repeated = sched.stats().live;
            callback_require(repeated.sample_elapsed_ms == s.live.sample_elapsed_ms && repeated.prefilled_tokens == s.live.prefilled_tokens &&
                    repeated.published_tokens == s.live.published_tokens, "polls changed work before a scheduler boundary");
            callback_require(s.live.sample_elapsed_ms >= sampled, "live request sample time decreased");
            sampled = s.live.sample_elapsed_ms;
            std::set<uint64_t> ids;
            for (size_t i = 0; i < s.live.count; ++i) {
                const auto& r = s.live.items[i];
                callback_require(r.id && ids.insert(r.id).second, "live requests have invalid or repeated identities");
                const auto before = observed.find(r.id);
                if (before != observed.end()) {
                    const auto& old = before->second;
                    callback_require(r.prefilled_tokens >= old.prefilled_tokens && r.published_tokens >= old.published_tokens &&
                            r.elapsed_ms >= old.elapsed_ms && r.prompt_tokens == old.prompt_tokens, "live request counters decreased or identity changed");
                }
                observed[r.id] = r;
                live_prompt = live_prompt || (std::string(r.phase) == "prefill" && r.prefilled_tokens > 0);
                live_output = live_output || (std::string(r.phase) == "generate" && r.published_tokens > 0);
                live_reuse = live_reuse || r.reused_tokens > 0;
                callback_require(r.reused_tokens <= r.prompt_tokens && r.prefilled_tokens <= r.prompt_tokens - r.reused_tokens,
                        "live prompt work includes reused or generated rows on an unpaused request");
                callback_require(r.prefilled_tokens <= s.live.prefilled_tokens && r.published_tokens <= s.live.published_tokens,
                        "live request work exceeds its snapshot's aggregate work");
            }
        };
        const Req first{prompt_of(5, 513, vocab), 8};
        std::vector<std::shared_ptr<server::Request>> handles;
        uint64_t expected_prompt = 0, expected_output = 0;
        std::atomic<bool> polling{true}, poll_ok{true};
        std::thread poller([&] {
            uint64_t last_sample = 0;
            while (polling.load()) {
                const auto s = sched.stats();
                if (s.live.count > s.live.limit || s.live.count > s.live.total || s.live.sample_elapsed_ms < last_sample) poll_ok.store(false);
                last_sample = s.live.sample_elapsed_ms;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        std::thread runner([&] { sched.run(); });
        try {
            handles.push_back(sched.submit(first.prompt, params_of(first)));
            expected_output += drain(*handles.back()).size();
            // The repeated prompt can reuse the first request's donor while another prompt starts beside it.
            handles.push_back(sched.submit(first.prompt, params_of(first)));
            const Req other{prompt_of(7, 257, vocab), 8};
            handles.push_back(sched.submit(other.prompt, params_of(other)));
            expected_output += drain(*handles[1]).size();
            expected_output += drain(*handles[2]).size();
        } catch (...) {
            sched.stop();
            runner.join();
            polling.store(false);
            poller.join();
            throw;
        }
        const auto idle_until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (sched.stats().live.total && std::chrono::steady_clock::now() < idle_until) std::this_thread::yield();
        const auto idle = sched.stats().live;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const auto idle_again = sched.stats().live;
        const bool idle_ok = !idle.total && !idle.count && !idle_again.total && idle.sample_elapsed_ms == idle_again.sample_elapsed_ms &&
                             idle.prefilled_tokens == idle_again.prefilled_tokens && idle.published_tokens == idle_again.published_tokens;
        sched.stop();
        runner.join();
        polling.store(false);
        poller.join();
        for (const auto& r : handles) expected_prompt += r->prompt_tokens() - r->reused();
        const auto done = sched.stats();
        require(monotonic, "work counters decreased");
        require(prompt_running && output_running, "work did not advance before the first request finished");
        require(poll_ok.load() && live_prompt && live_output && observed.size() == 3, "live requests did not advance safely through their phases");
        require(done.live.count == 0 && done.live.total == 0, "ended requests remain in the live snapshot");
        require(idle_ok, "idle polling advanced the cached snapshot or retained ended requests");
        require(done.prefilled_tokens == expected_prompt, "retired prompt work includes cached, generated or duplicate stage rows");
        require(done.published_tokens == expected_output, "published work differs from the channels' tokens");
        require(done.live.prefilled_tokens == expected_prompt && done.live.published_tokens == expected_output,
                "final empty snapshot lost cumulative work of completed requests");
        if (!model->keeps_state()) require(handles[1]->reused() > 0 && live_reuse, "the work check did not exercise a cached prompt");
    }
    auto model = one(32 * kBlock, kUbatch);
    server::Scheduler bounded(*model, tok, 1, 70);
    const Req request{prompt_of(9, 513, vocab), 2};
    std::vector<std::shared_ptr<server::Request>> waiting;
    for (size_t i = 0; i < 70; ++i) waiting.push_back(bounded.submit(request.prompt, params_of(request)));
    bool saw_bound = false, first_poll = true;
    bounded.on_retire = [&](const server::Scheduler::Retired&) {
        const auto s = bounded.stats();
        if (first_poll) {
            callback_require(!s.live.total && !s.live.count && !s.live.sample_elapsed_ms &&
                             !s.live.prefilled_tokens && !s.live.published_tokens, "an unrequested snapshot was collected");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const auto repeated = bounded.stats();
            callback_require(repeated.live.sample_elapsed_ms == s.live.sample_elapsed_ms && repeated.live.total == s.live.total &&
                    repeated.live.prefilled_tokens == s.live.prefilled_tokens && repeated.live.published_tokens == s.live.published_tokens,
                    "polls changed a snapshot before a scheduler boundary");
            first_poll = false;
        }
        if (saw_bound || !s.live.count) return;
        callback_require(s.live.total == 70 && s.live.count == 64 && s.live.total - s.live.count == 6, "live request truncation lost its total or cap");
        callback_require(s.live.items[0].id == 1 && std::string(s.live.items[0].phase) == "prefill", "queued requests hid the active request");
        saw_bound = true;
        for (size_t i = 1; i < waiting.size(); ++i) waiting[i]->cancel();
    };
    std::thread runner([&] { bounded.run(); });
    try {
        drain(*waiting[0]);
        for (size_t i = 1; i < waiting.size(); ++i) {
            require(collect(*waiting[i]).empty() && waiting[i]->finish() == "cancel", "a queued cancelled request published tokens");
        }
    } catch (...) {
        bounded.stop(); runner.join(); throw;
    }
    const auto cancel_until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (bounded.stats().live.total && std::chrono::steady_clock::now() < cancel_until) std::this_thread::yield();
    const bool cleared = !bounded.stats().live.total;
    bounded.stop();
    runner.join();
    require(saw_bound && cleared && bounded.stats().live.count == 0, "bounded snapshot or cancellation cleanup was not observed");
    checks += callback_checks;
    require(callback_error.empty(), callback_error);
}

// Every case over `weights`, served one pass at a time on one CPU and with passes in flight on splits.
void cases(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const Make one = on(weights, [] { return cpus(1); });
    const auto split = [&weights](size_t stages) { return on(weights, [stages] { return cpus(stages); }); };
    live_work(one, split, tok, vocab);
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
    short_beside_long(split, one, tok, vocab);
    in_place(one, split, tok, vocab);
    cancelled(weights, one, tok, vocab);
    failed(weights, one, tok, vocab);
    failed_continued(weights, tok, vocab);
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
    // The same over stages of tensor groups of two such backends.
    const Make one_group = on(weights, [] { return cpus(2); }, 8, 0, 0, 0, false, 2);
    const auto groups = [&weights, &seen](size_t stages) {
        return on(weights, [stages, &seen] {
            std::vector<backend::BackendPtr> v;
            for (size_t i = 0; i < 2 * stages; ++i) v.push_back(std::make_shared<NoCpu>(&seen));
            return v;
        }, 8, 0, 0, 0, false, 2);
    };
    paused(one_group, groups, tok, vocab);
    const size_t pool = 32 * kBlock;
    const std::vector<Req> reqs = steady_load(vocab);
    const std::vector<Reply> ref = alone(one, tok, pool, reqs);
    for (const Run& run : runs()) {
        if (run.stages < 2 || run.passes < run.stages) continue;
        const std::string what = "stages that are no CPU's on " + name(run);
        auto model = split(run.stages)(pool, kUbatch);
        seen.clear();
        // A request alone whose prompt is one slice has one pass in flight at a time; a longer prompt's slices continue each other (continued) and are recorded on the stages' threads.
        same(ref[4], serve(*model, tok, kSeqs, {{reqs[4]}}, nullptr, run.passes)[0], what + ", a request alone");
        require(seen.count() == 1, what + ": " + std::to_string(seen.count()) + " threads submitted the stages of a lone request of one slice, against the scheduler's alone");
        model = split(run.stages)(pool, kUbatch);
        seen.clear();
        const std::vector<Reply> got = serve(*model, tok, kSeqs, {reqs}, nullptr, run.passes);
        require(seen.count() > 1, what + ": no stage of several requests' passes was recorded on a thread of its own");
        for (size_t i = 0; i < reqs.size(); ++i) same(ref[i], got[i], what + ", request " + std::to_string(i));
    }
}

// A backend that is no CPU's and takes `nap` a submission, so a split's stages run long enough for passes to wait between them.
struct Slow : NoCpu {
    std::chrono::milliseconds nap{0};
    Slow(Submitters* s, int ms) : NoCpu(s), nap(ms) {}
    backend::Ticket submit() override {
        std::this_thread::sleep_for(nap);
        return NoCpu::submit();
    }
};

// A scheduler whose stages have threads, made and destroyed at once, many times: a stage's thread that first runs as the scheduler is destroyed must still find its own record.
// It needs no assertion: the fault it guards is a crash, which the C++ library of the macOS build shows and those of the Linux and Windows builds hide (docs/STATUS.md, a stage's thread is given its record).
void made_and_destroyed(const gguf::GGUFModel& weights) {
    const bpe::Tokenizer tok(weights);
    Submitters seen;
    auto model = on(weights, [&seen] { return std::vector<backend::BackendPtr>{std::make_shared<NoCpu>(&seen)}; })(16 * kBlock, kUbatch);
    for (size_t i = 0; i < 400; ++i) server::Scheduler sched(*model, tok, kSeqs, 64, 0, false, 0, nullptr, 0, true, {});
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
    // Four drafts a request for 16 requests give the 64 draft rows a pass reserves at most.
    const size_t draft_max = drafting ? 4 : 0;
    request.draft_rows = arena ? server::Scheduler::draft_rows(seqs, draft_max) : 0;
    request.logit_rows = server::logit_rows(slots, logits ? server::Scheduler::logits_a_pass(seqs, draft_max) : seqs).size;
    infer::ModelOptions options;
    options.kv_tokens = size_t(1) << 20;
    options.mark_slots = drafting ? 4 : 0;
    options.mark_rows = drafting ? draft_max + 1 : 0;
    auto model = infer::place_model(infer::gguf_weights(weights), {device}, request, options).model;
    infer::spec::Lookup lookup;
    server::Scheduler sched(*model, tok, seqs, 64, 0, false, 0, drafting ? &lookup : nullptr, draft_max, false, {});
    return {device->taken.load(), model->kv_allocated_bytes() / (model->kv_tokens_total() / kBlock)};
}
// A drafter's rows come out of the fitted KV budget and not on top of it: fitted to the same free memory, a drafting server allocates within the room and no more than one without a drafter.
// Fitted without the draft rows in a pass's rows, in its logits rows, or in both as before the fix, it allocates more, so each half of the fix is seen alone.
void fitted_reservation(const gguf::GGUFModel& weights) {
    const bpe::Tokenizer tok(weights);
    // A room just below a KV block's edge: the logits rows alone are a quarter of a block on this model, and are seen only where the room leaves less than that.
    const size_t room = (size_t(24) << 20) + (size_t(52) << 10);
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

// A backend whose queue a submission holds for its whole nap, as a Vulkan member's is between prepare() and queued(): a copy, a write or a submission on it from another thread meanwhile is counted.
// While `watching`, a stage's submission holds the queue for kSlow and a call of the scheduler's thread first waits up to kWindow for one, which only a call outside the stages' hold can meet.
struct Owned : NoCpu {
    static constexpr std::chrono::milliseconds kWindow{500}, kSlow{200};
    std::chrono::milliseconds nap;
    std::atomic<std::thread::id> lent{}, scheduler{}, stage{};
    std::atomic<size_t> crossed{0}, waited{0};
    std::atomic<bool> watching{false};
    std::mutex m;
    std::condition_variable lending;
    Owned(Submitters* s, int ms) : NoCpu(s), nap(ms) {}
    void check() {
        const std::thread::id me = std::this_thread::get_id();
        if (watching.load() && me == scheduler.load()) {
            std::unique_lock<std::mutex> lk(m);
            ++waited;
            lending.wait_for(lk, kWindow, [&] { const std::thread::id h = lent.load(); return h != std::thread::id() && h != me; });
        }
        const std::thread::id holder = lent.load();
        if (holder != std::thread::id() && holder != me) ++crossed;
    }
    backend::Ticket submit() override {
        check();
        const std::thread::id me = std::this_thread::get_id();
        if (me != scheduler.load()) stage.store(me);
        {
            std::lock_guard<std::mutex> lk(m);
            lent.store(me);
        }
        lending.notify_all();
        std::this_thread::sleep_for(watching.load() && me != scheduler.load() ? kSlow : nap);
        const backend::Ticket t = NoCpu::submit();
        lent.store(std::thread::id());
        return t;
    }
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        check();
        NoCpu::copy(dst, dst_off, src, src_off, bytes);
    }
    void write(backend::Buffer& dst, size_t off, const void* src, size_t bytes) override {
        check();
        NoCpu::write(dst, off, src, bytes);
    }
};

// A boundary's state forked from host memory while a prompt is read beside a reply: the copies run on the stage's thread, and no call of the scheduler's thread goes into a held queue.
// A hybrid model on a layer split of two non-CPU devices at its default passes, so the last stage's device is recorded on its thread while another pass is in flight.
void boundary_fork_held(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    Submitters seen;
    std::shared_ptr<Owned> device;
    const Make owned = on(weights, [&seen, &device] {
        device = std::make_shared<Owned>(&seen, 15);
        return std::vector<backend::BackendPtr>{std::make_shared<Slow>(&seen, 15), device};
    }, 4, 3);
    auto model = owned(4096, 0);
    device->crossed = 0;
    std::vector<uint32_t> edit;
    Reply got;
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 4, 64, 0, false, (size_t)1 << 30, nullptr, 0, false, {});
        std::thread runner([&] { sched.run(); });
        device->scheduler = runner.get_id();
        try {
            // Two turns, each read again: the second's donor supersedes the first's, so an edit of the second turn forks the first turn's boundary with its state from host memory.
            std::vector<uint32_t> prompt = prompt_of(5, 300, vocab), first_next;
            for (size_t turn = 1; turn <= 2; ++turn) {
                const auto h = sched.submit(prompt, params_of(Req{prompt, 100}));
                std::vector<uint32_t> next = prompt;
                for (const auto& t : drain(*h)) next.push_back(t.id);
                next.push_back(1);
                next.push_back(2);
                sched.follow(h, next, true);
                const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                while (sched.stats().reprefills < turn || sched.stats().boundaries < turn) {
                    require(std::chrono::steady_clock::now() < until, "a boundary fork beside a prompt read beside a reply: a turn was not read again with its boundary kept in 60 seconds");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if (turn == 1) first_next = next;
                prompt = next;
                const std::vector<uint32_t> tail = prompt_of(10 + (int)turn, 50, vocab);
                prompt.insert(prompt.end(), tail.begin(), tail.end());
            }
            // A long request evicts the donors to host memory, so the edit's history is a host copy and its state the boundary's.
            const std::vector<uint32_t> evicting = prompt_of(9, 3600, vocab);
            drain(*sched.submit(evicting, params_of(Req{evicting, 8})));
            const std::vector<uint32_t> other = prompt_of(30, 50, vocab), reply_prompt = prompt_of(40, 50, vocab), long_prompt = prompt_of(41, 2000, vocab);
            edit = first_next;
            edit.insert(edit.end(), other.begin(), other.end());
            const auto r = sched.submit(reply_prompt, params_of(Req{reply_prompt, 300}));
            server::Request::Token first;
            require(r->next(first, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, "a boundary fork beside a prompt read beside a reply: the reply gave no token");
            // The long prompt is recorded on the stage's thread; once that thread has submitted, the edit comes, watched until its first token.
            device->stage = std::thread::id();
            const auto l = sched.submit(long_prompt, params_of(Req{long_prompt, 8}));
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60); device->stage.load() == std::thread::id();) {
                require(std::chrono::steady_clock::now() < until, "a boundary fork beside a prompt read beside a reply: no stage's thread submitted in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            device->watching = true;
            const auto e = sched.submit(edit, params_of(Req{edit, 32}));
            server::Request::Token edited;
            const bool came = e->next(edited, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id;
            device->watching = false;
            require(came, "a boundary fork beside a prompt read beside a reply: the edit gave no token");
            got.push_back(edited);
            for (const auto& t : drain(*e)) got.push_back(t);
            drain(*l);
            collect(*r);
            stats = sched.stats();
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    require(stats.boundary_hits >= 1, "a boundary fork beside a prompt read beside a reply: " + std::to_string(stats.boundary_hits) + " requests forked a boundary, against one at least");
    std::cout << "server-passes-cpu: a boundary fork beside a prompt read beside a reply: " << device->waited.load() << " calls of the scheduler's thread watched, " << device->crossed.load() << " met a held queue\n";
    require(device->crossed == 0, "a boundary fork beside a prompt read beside a reply: " + std::to_string(device->crossed.load()) + " calls met a queue a stage's submission held");
    const Make plain = on(weights, [] { return cpus(1); }, 3, 3);
    auto fresh = plain(4096, 0);
    same(serve(*fresh, tok, 3, {{Req{edit, 32}}})[0], got, "a boundary fork beside a prompt read beside a reply, the edit's reply");
    ++checks;
}

// Drafts shared among replies: four replies on a hybrid model whose two marks let two of them draft at once, each prompt ending as it began so lookup finds drafts for all, on one CPU and on two at two passes in flight.
// Every reply verifies drafts in at least half as many passes as the one that verifies most, a guard for DDEV's uneven replies (28, 28, 14 and 14 tokens a second on Flash-Next 2x2), which this does not reproduce.
void shared_drafts(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    std::vector<Req> reqs;
    for (uint32_t r = 0; r < 4; ++r) {
        std::vector<uint32_t> prompt = prompt_of(20 + r, 40, vocab);
        prompt.insert(prompt.end(), {prompt[0], prompt[1], prompt[2]});
        reqs.push_back({prompt, 64});
    }
    // A slow device that is no CPU's, so a pass stays in flight while the next is formed, with columns for drafts.
    struct WideSlow : Slow {
        using Slow::Slow;
        size_t decode_columns() const override { return 16; }
    };
    Submitters seen;
    for (const size_t stages : {size_t(1), size_t(2)}) {
        const Make make = on(weights, [stages, &seen] {
            std::vector<backend::BackendPtr> v;
            for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<WideSlow>(&seen, 2));
            return v;
        }, 8, 0, 2, 4);
        auto model = make(pool, kUbatch);
        infer::spec::Lookup lookup;
        server::Scheduler sched(*model, tok, kSeqs, 64, stages, false, 0, &lookup, 3, false, {});
        std::map<const server::Request*, size_t> verifies;
        std::vector<std::shared_ptr<server::Request>> handles;
        bool ended = false;
        sched.on_retire = [&](const server::Scheduler::Retired& t) {
            for (size_t e = 0; e < t.requests.size(); ++e) {
                ended = ended || t.from[e] + t.rows[e] + 4 >= t.requests[e]->prompt_tokens() + 64;
                if (!ended && t.every[e] && t.rows[e] > 1) ++verifies[t.requests[e]];
            }
        };
        for (const Req& r : reqs) handles.push_back(sched.submit(r.prompt, params_of(r)));
        std::thread runner([&] { sched.run(); });
        try {
            for (auto& h : handles) drain(*h);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        size_t lo = SIZE_MAX, hi = 0;
        std::string counts;
        for (const auto& h : handles) {
            lo = std::min(lo, verifies[h.get()]), hi = std::max(hi, verifies[h.get()]);
            counts += " " + std::to_string(verifies[h.get()]);
        }
        const std::string where = "drafts shared among four replies over two marks on " + std::to_string(stages) + " CPU" + (stages == 1 ? "" : "s");
        std::cout << "server-passes-cpu: " << where << ": verifies per reply" << counts << std::endl;
        require(hi > 0 && 2 * lo >= hi, where + ": verifies per reply" + counts);
    }
}

// A newcomer under overload (docs/SERVER.md, The service rule): two slow replies leave the objective unmet, beside a long prompt waiting for the bound and alone.
// An 85-token prompt sent then is quick and rides the replies' pass, held by counts of passes rather than their times: a slower runner adds no pass between its rides, since the objective only falls further short there.
void short_newcomer(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    // Devices whose decode kernels hold every reply in one pass, as a GPU's do, so the replies pack into one pass and a slot formed while it flies has no prompt due.
    struct Wide : Slow {
        using Slow::Slow;
        size_t decode_columns() const override { return 64; }
    };
    const Make split = on(weights, [&seen] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 2; ++i) v.push_back(std::make_shared<Slow>(&seen, 70));
        return v;
    });
    const Make wide = on(weights, [&seen] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 2; ++i) v.push_back(std::make_shared<Wide>(&seen, 70));
        return v;
    });
    const Req a{prompt_of(47, 4, vocab), 120}, b{prompt_of(48, 4, vocab), 120}, slow{prompt_of(49, 1500, vocab), 2}, late{prompt_of(50, 85, vocab), 2};
    // Beside the long prompt and alone, where a slot no prompt is due for goes to the replies and must still take the quick prompt.
    for (const bool beside : {true, false}) {
        const std::string where = beside ? " beside a long prompt" : " alone";
        auto model = (beside ? split : wide)(pool, 512);
        server::Scheduler sched(*model, tok, kSeqs, 64, 0);
        // The scheduler thread counts retired passes, and once the newcomer is sent, the replies' passes without its rows before its first rows and between its rides, and its rides; the newcomer is the only prompt of its length.
        std::atomic<size_t> retired{0};
        std::atomic<bool> sent{false};
        size_t misses = 0, before = 0, between = 0, rides = 0;
        bool rode = false, read = false;
        // Between two steps of a reply, the most other passes that retired and the most prompt rows one of them carried, counted on the scheduler thread and read once it has joined.
        // Counts rather than times: the 500 ms cap is an objective the prices aim at, and a runner slower at every pass widens the gap without the rule letting any more work in.
        struct Since {
            size_t passes = 0, rows = 0;
        };
        std::map<const server::Request*, Since> stepped;
        size_t most_passes = 0, most_rows = 0;
        sched.on_retire = [&](const server::Scheduler::Retired& t) {
            ++retired;
            size_t prompt_rows = 0;
            for (size_t e = 0; e < t.requests.size(); ++e)
                if (t.requests[e]->prompt_tokens() != 4) prompt_rows = std::max(prompt_rows, t.rows[e]);
            for (auto& entry : stepped) {
                ++entry.second.passes;
                entry.second.rows = std::max(entry.second.rows, prompt_rows);
            }
            for (size_t e = 0; e < t.requests.size(); ++e)
                if (t.requests[e]->prompt_tokens() == 4 && t.from[e] >= 4) {
                    const auto it = stepped.find(t.requests[e]);
                    if (it != stepped.end()) {
                        most_passes = std::max(most_passes, it->second.passes - 1);
                        most_rows = std::max(most_rows, it->second.rows);
                    }
                    stepped[t.requests[e]] = {};
                }
            if (!sent.load() || read) return;
            bool replies = false, carried = false;
            for (size_t e = 0; e < t.requests.size(); ++e) {
                replies = replies || (t.requests[e]->prompt_tokens() == 4 && t.from[e] >= 4);
                if (t.requests[e]->prompt_tokens() != late.prompt.size()) continue;
                carried = true;
                read = t.from[e] + t.rows[e] == late.prompt.size();
            }
            if (carried) {
                (rode ? between : before) = std::max(rode ? between : before, misses);
                misses = 0;
                rode = true;
                ++rides;
            } else if (replies) {
                ++misses;
            }
        };
        std::vector<std::shared_ptr<server::Request>> handles{sched.submit(a.prompt, params_of(a)), sched.submit(b.prompt, params_of(b))};
        if (beside) handles.push_back(sched.submit(slow.prompt, params_of(slow)));
        std::thread runner([&] { sched.run(); });
        try {
            // Once the objective has gone unmet beside the long prompt, or alone once the replies have taken a dozen passes and their pass is priced, the newcomer arrives.
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60); beside ? sched.stats().service.objective.unmet == 0 : retired.load() < 12;) {
                require(std::chrono::steady_clock::now() < until, "the reply objective never went unmet on two devices of 70 ms a pass");
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            sent = true;
            handles.push_back(sched.submit(late.prompt, params_of(late)));
            for (auto& h : handles) drain(*h);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        // Before its first rows, the passes in flight when it arrived (S + 1 = 3 slots), one formed before its admission and one replies' pass whose turn it was not; between its rides, the passes in flight beside its ride's and that one.
        // Each ride reads at least the 16-row learning step, so 85 rows take six at most.
        const size_t slots = 3;
        require(read, "a short newcomer under an unmet objective" + where + " never read its prompt");
        require(before <= slots + 2, "a short newcomer under an unmet objective" + where + " waited " + std::to_string(before) + " replies' passes for its first rows, against the passes in flight and two more");
        require(between <= slots, "a short newcomer under an unmet objective" + where + " missed " + std::to_string(between) + " replies' passes in a row between its rides, against the passes in flight");
        require(rides <= 6, "a short newcomer under an unmet objective" + where + " took " + std::to_string(rides) + " passes to read 85 rows, against six learning steps");
        std::cout << "server-passes-cpu: a quick newcomer" << where << ": at most " << most_passes << " other passes and " << most_rows << " prompt rows a pass between two steps of a reply" << std::endl;
        // Between two steps of a reply, the passes in flight beside its own, none holding more prompt rows than the newcomer's whole prompt.
        require(most_passes <= slots - 1, "a reply saw " + std::to_string(most_passes) + " other passes retire between two steps beside a quick newcomer" + where + ", against the passes in flight beside its own");
        require(most_rows <= late.prompt.size(), "a pass between two steps of a reply held " + std::to_string(most_rows) + " prompt rows beside a quick newcomer" + where + ", against the newcomer's whole prompt");
    }
}

// The cold rule (docs/SERVER.md, The service rule): a reply and a long prompt arriving together on a fresh server, ubatch 256, on one CPU and on layer splits of two and three slow devices at P = S + 1.
// Once the reply decodes, the prompt's slices start small, one step while the replies' pass has no price; on one pass in flight the replies' pass goes alone first, so it is measured.
void cold_start(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    constexpr int kWide = 256;
    Submitters seen;
    const Req reply{prompt_of(61, 4, vocab), 40}, prompt{prompt_of(62, 1800, vocab), 4};
    for (const size_t stages : {size_t(1), size_t(2), size_t(3)}) {
        const size_t passes = stages == 1 ? 1 : stages + 1;
        const std::string where = "a reply and a prompt on a fresh server on " + std::to_string(stages) + " stage" + (stages == 1 ? "" : "s");
        const Make split = on(weights, [&seen, stages] {
            std::vector<backend::BackendPtr> v;
            for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
            return v;
        });
        auto model = split(pool, kWide);
        server::Scheduler sched(*model, tok, kSeqs, 64, passes);
        std::vector<server::Scheduler::Retired> retired;
        sched.on_retire = [&](const server::Scheduler::Retired& t) { retired.push_back(t); };
        auto a = sched.submit(reply.prompt, params_of(reply));
        auto b = sched.submit(prompt.prompt, params_of(prompt));
        std::thread runner([&] { sched.run(); });
        try {
            drain(*a);
            drain(*b);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        // The pass that gave the reply its first token, then the first slice of the prompt formed after it: a pass retiring more than P passes later was formed after it.
        size_t first = retired.size();
        for (size_t i = 0; i < retired.size() && first == retired.size(); ++i)
            for (size_t e = 0; e < retired[i].requests.size(); ++e)
                if (retired[i].requests[e] == a.get() && retired[i].from[e] + retired[i].rows[e] == reply.prompt.size()) first = i;
        require(first < retired.size(), where + ": the reply's prompt never retired");
        size_t next = 0;
        bool alone = true;
        for (size_t i = first + 1; i < retired.size() && !next; ++i)
            for (size_t e = 0; e < retired[i].requests.size(); ++e) {
                if (retired[i].requests[e] != b.get() || retired[i].from[e] + retired[i].rows[e] == prompt.prompt.size()) continue;
                if (i == first + 1) alone = false;
                if (i > first + passes) next = retired[i].rows[e];
            }
        // One step while unpriced, or two once that step and the replies' pass are priced, against a whole ubatch before the rule.
        require(next > 0 && next <= 2 * server::kPieceStep, where + ": the prompt's first slice after the reply began decoding took " + std::to_string(next) + " rows, against one step of " +
                                                                std::to_string(server::kPieceStep) + " while the replies' pass had no price");
        require(stages > 1 || alone, where + ": on one pass in flight the reply's first decode pass carried the prompt's rows before it was priced");
        ++checks;
    }
}

// The passes in flight follow the queue (docs/SERVER.md, The service rule), on two slow devices at the default S + 1 slots.
// Four replies alone keep at most one pass a stage in flight, and a reply beside a long prompt uses the one more.
void passes_choice(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    const Make split = on(weights, [&seen] {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < 2; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
        return v;
    });
    // The most passes in flight the scheduler reports while `reqs` run, sampled every millisecond.
    const auto most = [&](const std::vector<Req>& reqs) {
        auto model = split(pool, 64);
        server::Scheduler sched(*model, tok, kSeqs, 64, 0);
        std::vector<std::shared_ptr<server::Request>> handles;
        for (const Req& r : reqs) handles.push_back(sched.submit(r.prompt, params_of(r)));
        std::atomic<bool> done{false};
        size_t seen_most = 0;
        std::thread runner([&] { sched.run(); });
        std::thread sampler([&] {
            while (!done.load()) {
                seen_most = std::max(seen_most, sched.stats().in_flight);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        try {
            for (auto& h : handles) drain(*h);
        } catch (...) {
            done = true;
            sampler.join();
            sched.stop();
            runner.join();
            throw;
        }
        done = true;
        sampler.join();
        require(sched.stats().passes == 3, "the scheduler reserved " + std::to_string(sched.stats().passes) + " slots on two stages, against 3");
        sched.stop();
        runner.join();
        return seen_most;
    };
    std::vector<Req> replies;
    for (uint32_t i = 0; i < 4; ++i) replies.push_back({prompt_of(41 + i, 4, vocab), 60});
    const size_t alone = most(replies);
    require(alone <= 2, "four replies alone on two stages kept " + std::to_string(alone) + " passes in flight, against one a stage");
    const size_t beside = most({{prompt_of(45, 4, vocab), 80}, {prompt_of(46, 1500, vocab), 2}});
    require(beside == 3, "a reply beside a long prompt on two stages kept at most " + std::to_string(beside) + " passes in flight, against 3");
}

// Four replies decoding together share the passes evenly: on one CPU and on layer splits of two and three slow devices at P = S, S + 1 and 2S, after every retired pass the steps each has taken differ by one at most.
// DDEV measured four replies at 28, 28, 14 and 14 tokens a second on Flash-Next 2x2; a share that gave its leftover to the same requests every round would show here.
void equal_replies(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 16 * kBlock;
    constexpr int kCap = 60;
    Submitters seen;
    std::vector<Req> replies;
    for (uint32_t i = 0; i < 4; ++i) replies.push_back({prompt_of(51 + i, 4, vocab), kCap});
    for (const size_t stages : {size_t(1), size_t(2), size_t(3)})
        for (const size_t passes : {stages, stages + 1, 2 * stages}) {
            if (stages == 1 && passes > 1) continue;
            const std::string where = "four replies on " + std::to_string(stages) + " stage" + (stages == 1 ? "" : "s") + " at P = " + std::to_string(passes);
            const Make split = on(weights, [&seen, stages] {
                std::vector<backend::BackendPtr> v;
                for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
                return v;
            });
            auto model = split(pool, kUbatch);
            server::Scheduler sched(*model, tok, kSeqs, 64, passes);
            std::vector<std::shared_ptr<server::Request>> handles;
            std::map<const server::Request*, size_t> steps;
            size_t worst = 0;
            bool ended = false;
            sched.on_retire = [&](const server::Scheduler::Retired& t) {
                for (size_t e = 0; e < t.requests.size(); ++e)
                    if (t.from[e] >= 4) {
                        ++steps[t.requests[e]];
                        ended = ended || t.from[e] + 2 >= 4 + (size_t)kCap;
                    }
                if (ended || steps.size() < replies.size()) return;
                size_t lo = SIZE_MAX, hi = 0;
                for (const auto& kv : steps) lo = std::min(lo, kv.second), hi = std::max(hi, kv.second);
                worst = std::max(worst, hi - lo);
            };
            for (const Req& r : replies) handles.push_back(sched.submit(r.prompt, params_of(r)));
            std::thread runner([&] { sched.run(); });
            try {
                for (auto& h : handles) drain(*h);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
            require(worst <= 1, where + ": the replies' steps differed by " + std::to_string(worst) + " after a retired pass");
            ++checks;
        }
}

// A prompt that lands while four replies decode on a layer split of two, at P = S and S + 1: it reads its prompt and gives its first token before any reply ends, every reply its reply alone.
// Production's case on Qwen3-8B (four uncapped replies, a 16k prompt that read no row for 900 s): the replies' passes took every free slot, since four decoders over the passes fill each one.
void prompt_beside_replies(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 16 * kBlock;
    Submitters seen;
    const Make one = on(weights, [] { return cpus(1); });
    const std::vector<Req> replies{{prompt_of(31, 4, vocab), 150}, {prompt_of(32, 4, vocab), 150}, {prompt_of(33, 4, vocab), 150}, {prompt_of(34, 4, vocab), 150}};
    const Req late{prompt_of(35, 200, vocab), 4};
    const std::vector<Reply> own = alone(one, tok, pool, replies), own_late = alone(one, tok, pool, {late});
    for (const size_t passes : {size_t(2), size_t(3)}) {
            const std::string where = std::string("a prompt beside four replies on two devices at P = ") + std::to_string(passes);
            const Make split = on(weights, [&seen] {
                std::vector<backend::BackendPtr> v;
                for (size_t i = 0; i < 2; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
                return v;
            });
            auto model = split(pool, kUbatch);
            server::Scheduler sched(*model, tok, kSeqs, 64, passes);
            std::vector<std::shared_ptr<server::Request>> handles;
            std::shared_ptr<server::Request> prompt;
            std::atomic<bool> sent{false};
            size_t pass = 0, first_token = 0, first_end = 0;
            // The prompt is submitted on the scheduler thread once every reply decodes, as the tenth pass retires; the passes that end the prompt and the first reply are counted.
            sched.on_retire = [&](const server::Scheduler::Retired& t) {
                ++pass;
                for (size_t e = 0; e < t.requests.size(); ++e) {
                    const server::Request* r = t.requests[e];
                    if (r == prompt.get() && t.want[e] && t.from[e] + t.rows[e] == r->prompt_tokens() && !first_token) first_token = pass;
                    if (r != prompt.get() && t.from[e] + 2 == r->prompt_tokens() + (size_t)replies[0].cap && !first_end) first_end = pass;
                }
                if (pass == 10 && !sent.load()) {
                    prompt = sched.submit(late.prompt, params_of(late));
                    sent.store(true);
                }
            };
            for (const Req& r : replies) handles.push_back(sched.submit(r.prompt, params_of(r)));
            std::thread runner([&] { sched.run(); });
            std::vector<Reply> got;
            Reply got_late;
            try {
                for (auto& h : handles) got.push_back(drain(*h));
                while (!sent.load()) std::this_thread::yield();
                got_late = drain(*prompt);
            } catch (...) {
                sched.stop();
                runner.join();
                throw;
            }
            sched.stop();
            runner.join();
            for (size_t i = 0; i < replies.size(); ++i) same(own[i], got[i], where + ", reply " + std::to_string(i));
            same(own_late[0], got_late, where + ", the prompt");
            require(first_token && first_end && first_token < first_end,
                    where + ": the prompt's first token came in pass " + std::to_string(first_token) + ", not before the first reply ended in pass " + std::to_string(first_end));
            ++checks;
        }
}

// A prompt in consecutive passes (docs/MULTI-DEVICE.md, phase 4) on layer splits of two and three slow devices, none a CPU, at P = S and S + 1.
// A lone prompt of 300 tokens continues while its slice before is in flight (Stats::continued); beside a reply the service rule never puts both in one pass (docs/SERVER.md, The service rule).
void continued(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 16 * kBlock;
    Submitters seen;
    const Make one = on(weights, [] { return cpus(1); });
    const std::vector<Req> lone{{prompt_of(61, 300, vocab), 3}};
    const std::vector<Req> beside{{prompt_of(62, 4, vocab), 40}, {prompt_of(63, 300, vocab), 3}};
    const std::vector<Reply> own_lone = alone(one, tok, pool, lone), own_beside = alone(one, tok, pool, beside);
    for (const size_t stages : {size_t(2), size_t(3)})
        for (const size_t passes : {stages, stages + 1})
            {
                const std::string where = "a continued prompt on " + std::to_string(stages) + " devices at P = " + std::to_string(passes);
                const Make split = on(weights, [&seen, stages] {
                    std::vector<backend::BackendPtr> v;
                    for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
                    return v;
                });
                {
                    auto model = split(pool, kUbatch);
                    server::Scheduler::Stats st;
                    const std::vector<Reply> got = serve(*model, tok, kSeqs, {lone}, &st, passes);
                    same(own_lone[0], got[0], where + ", alone");
                    require(st.continued > 0, where + ": a lone prompt never continued while its slice before was in flight");
                    ledger(st, *model, where + ", alone");
                }
                auto model = split(pool, kUbatch);
                server::Scheduler::Stats st;
                const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &st, passes);
                for (size_t i = 0; i < beside.size(); ++i) same(own_beside[i], got[i], where + ", beside a reply, request " + std::to_string(i));
                require(st.rule_replies > 0 && st.rule_chunks > 0 && st.rule_mixed == 0,
                        where + ": beside a reply " + std::to_string(st.rule_replies) + " passes of replies, " + std::to_string(st.rule_chunks) + " chunks and " + std::to_string(st.rule_mixed) + " passes of both");
                ledger(st, *model, where + ", beside a reply");
            }
}

// A message boundary on a split whose prompt continues in consecutive passes (Stats::continued), on a hybrid model with checkpoint slots and a host tier, at P = 2 and 3 on two slow devices.
// The boundary is kept once the pass whose slice reaches it retires, not when the pass of an earlier slice still in flight retires first, and an edit of the last message forks it with its reply alone;
// a boundary at the first slice's end is kept as before.
void boundary_continued(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    Submitters seen;
    const size_t pool = 16 * kBlock;
    // The host's free memory as a fixed ample figure, so a boundary's copy is admitted whatever the machine has free.
    const auto host_memory = []() -> std::optional<size_t> { return (size_t)64 << 30; };
    const Make one = on(weights, [] { return cpus(1); }, 3, 3, 0, 0, false, 1, host_memory);
    for (const size_t start : {size_t(600), size_t(200)})
    for (const size_t passes : {size_t(2), size_t(3)}) {
        const std::vector<uint32_t> before = prompt_of(11, start, vocab), message = prompt_of(12, 100, vocab), other = prompt_of(13, 50, vocab);
        std::vector<uint32_t> first = before, edit = before;
        first.insert(first.end(), message.begin(), message.end());
        edit.insert(edit.end(), other.begin(), other.end());
        const Reply own = serve(*one(pool, kUbatch), tok, 3, {{Req{edit, 3}}})[0];
        const std::string where = "a message boundary at " + std::to_string(start / kBlock * kBlock) + " on two slow devices at P = " + std::to_string(passes);
        const Make split = on(weights, [&seen] {
            return std::vector<backend::BackendPtr>{std::make_shared<Slow>(&seen, 2), std::make_shared<Slow>(&seen, 2)};
        }, 3, 3, 0, 0, false, 1, host_memory);
        auto model = split(pool, kUbatch);
        server::Scheduler sched(*model, tok, 3, 64, passes, false, (size_t)1 << 30);
        std::thread runner([&] { sched.run(); });
        Reply reply;
        size_t reused = 0;
        server::Scheduler::Stats after_first, st;
        try {
            drain(*sched.submit(first, params_of(Req{first, 3}), first.size(), before.size()));
            after_first = sched.stats();
            const auto e = sched.submit(edit, params_of(Req{edit, 3}), edit.size(), before.size());
            reply = drain(*e);
            reused = e->reused();
            st = sched.stats();
            ledger(st, *model, where);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
        require(start < 2 * kBlock || after_first.continued > 0, where + ": the prompt never continued while its slice before was in flight");
        require(after_first.boundaries == 1, where + ": " + std::to_string(after_first.boundaries) + " boundaries kept where the last message starts, against 1");
        const size_t want = before.size() / kBlock * kBlock;
        require(reused == want && st.boundary_hits == 1,
                where + ": the edit reused " + std::to_string(reused) + " tokens through " + std::to_string(st.boundary_hits) + " boundaries, against " + std::to_string(want) + " through 1");
        same(own, reply, where + ", the edit");
        ++checks;
    }
}

int main() {
    try {
        // What a caller asks the fit for is what the scheduler reserves: a row a request and, where requests draft, a pass's draft rows, at most 64 and at most --draft-max a request, in each logits ring.
        using S = server::Scheduler;
        require(S::logits_a_pass(16, 0) == 16 && S::draft_rows(16, 0) == 0 && S::draft_rows(1, 3) == 3 && S::draft_rows(16, 3) == 48 && S::logits_a_pass(16, 3) == 64 &&
                    S::draft_rows(16, 31) == 64 && S::draft_rows(2, 31) == 62 && S::draft_rows(3, 31) == 64 && S::draft_rows(1, 63) == 63,
                "the logits rows of a pass with and without drafts");
        reply_room(served(kSplit), (uint32_t)kSplit.vocab);
        grouped(served(kSplit), (uint32_t)kSplit.vocab);
        routed_groups(served_routed(kSplit), (uint32_t)kSplit.vocab);
        grouped(served_hybrid(kHybridEven), (uint32_t)kHybridEven.vocab);
        drafted_groups(served_hybrid(kHybridEven), (uint32_t)kHybridEven.vocab);
        chained_groups(served_hybrid(kHybridEven, true), (uint32_t)kHybridEven.vocab);
        timed_groups(served(kSplit), (uint32_t)kSplit.vocab);
        timed_cpu(served(kSplit), (uint32_t)kSplit.vocab);
        recorded_stages(served(kSplit), (uint32_t)kSplit.vocab);
        made_and_destroyed(served(kSplit));
        equal_replies(served(kSplit), (uint32_t)kSplit.vocab);
        short_newcomer(served(kSplit), (uint32_t)kSplit.vocab);
        passes_choice(served(kSplit), (uint32_t)kSplit.vocab);
        shared_drafts(served_hybrid(kHybrid), (uint32_t)kHybrid.vocab);
        cold_start(served(kSplit), (uint32_t)kSplit.vocab);
        prompt_beside_replies(served(kSplit), (uint32_t)kSplit.vocab);
        continued(served(kSplit), (uint32_t)kSplit.vocab);
        boundary_continued(served_hybrid(kHybrid), (uint32_t)kHybrid.vocab);
        fitted_reservation(served(kSplit));
        cases(served(kSplit), (uint32_t)kSplit.vocab);
        // A hybrid model, whose linear-attention layers keep a recurrent state, over the same cases: without checkpoint slots it keeps no donor, and a stage may hold only states.
        cases(served_hybrid(kHybrid), (uint32_t)kHybrid.vocab);
        boundary_fork_held(served_hybrid(kHybrid), (uint32_t)kHybrid.vocab);
        std::cout << "server-passes-cpu: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server-passes-cpu: " << e.what() << '\n';
        return 1;
    }
}
