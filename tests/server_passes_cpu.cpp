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
    // The arrival mid-prompt, past the share of a short prompt (server::short_prompts): the lone prompt's first slice a quarter of its 768 rows, cut slices while it is alone, then whole slices, the second request's only after it.
    // Slices continue each other (docs/MULTI-DEVICE.md, phase 4), so the cut slices before the arrival follow the stages' timing, and the slices are held to that rule, not to a count.
    {
        const Req lone{prompt_of(50, 768, vocab), 4}, late{prompt_of(51, 100, vocab), 4};
        std::vector<server::Scheduler::Retired> passes;
        std::vector<std::shared_ptr<server::Request>> handles;
        const std::vector<Reply> got = run({lone}, &late, passes, handles);
        std::vector<size_t> rows;
        size_t read = 0;
        for (const auto& t : passes)
            for (size_t e = 0; e < t.requests.size(); ++e)
                if (t.requests[e] == handles[0].get() && t.rows[e] > 1) {
                    rows.push_back(t.rows[e]);
                    read += t.rows[e];
                } else if (t.requests[e] == handles[1].get() && t.rows[e] > 1) {
                    require(read % kWide == 0 || read == 768, "the request that arrived took a prompt slice before the lone prompt was back on a whole ubatch");
                }
        require(!rows.empty() && rows[0] == 192 && read == 768, "the lone prompt's first slice was not a quarter of its rows, or its slices did not add up");
        bool whole = false;
        for (size_t i = 0, sum = 0; i < rows.size(); ++i) {
            if (whole) require(rows[i] == (size_t)kWide || sum + rows[i] == 768, "the lone prompt took a cut slice after it was back on whole slices");
            sum += rows[i];
            whole = whole || sum % kWide == 0;
        }
        require(whole || rows.size() == 1, "the lone prompt was not brought back to whole slices after a request arrived");
        same(alone_on_one(lone), got[0], "the lone prompt");
        same(alone_on_one(late), got[1], "the request that arrived mid-prompt");
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
            require(sched.stats().short_passes > 0, where + ": no pass was counted as reading a short prompt ahead");
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

// A device whose attention takes time in proportion to its query rows times the history they read, as attention at depth does: 0.4 us a row and a token read.
// It keeps the time of each prompt's attention over a history of at least `deep` tokens.
struct Deep : NoCpu {
    using NoCpu::NoCpu;
    size_t deep = 0;
    std::mutex m;
    double deepest_ms = 0;
    size_t deep_calls = 0;
    double reply_ms = 0;   // a decode call's own cost, a reply pass at depth, where it is set
    double fixed_ms = 0;   // a prompt call's cost that its rows do not change, as attention at depth has, where it is set
    std::vector<std::pair<size_t, double>> stairs;   // where set, a prompt call's whole cost by its rows: the first step whose rows reach them, as tiled attention at depth takes another row of tiles
    // Every prompt view's attention in order: its query rows, the history it reads and the views beside it in the call, a reply's row among them where generated rows ride.
    struct Call {
        size_t rows, length, views;
    };
    std::vector<Call> calls;
    // The modeled time runs on the queue: a submission takes its calls' time after the one before it and a wait ends with it, so a pass recorded between pieces is timed apart from the next piece.
    using Clock = std::chrono::steady_clock;
    double pending_ms = 0;
    Clock::time_point free_at{};
    std::map<backend::Ticket, Clock::time_point> done_at;
    backend::Ticket submit() override {
        const backend::Ticket t = NoCpu::submit();
        std::lock_guard<std::mutex> lk(m);
        free_at = std::max(Clock::now(), free_at) + std::chrono::microseconds((long long)(pending_ms * 1000));
        pending_ms = 0;
        done_at[t] = free_at;
        return t;
    }
    void wait(backend::Ticket t) noexcept override {
        Clock::time_point until{};
        {
            std::lock_guard<std::mutex> lk(m);
            const auto it = done_at.upper_bound(t);
            if (it != done_at.begin()) until = std::prev(it)->second;
        }
        std::this_thread::sleep_until(until);
        NoCpu::wait(t);
    }
    void sync() noexcept override {
        Clock::time_point until{};
        {
            std::lock_guard<std::mutex> lk(m);
            until = std::max(Clock::now(), free_at) + std::chrono::microseconds((long long)(pending_ms * 1000));
        }
        std::this_thread::sleep_until(until);
        NoCpu::sync();
    }
    void attention(backend::CSlice q, size_t layer, const backend::KVView* views, size_t n_views, backend::Slice out, int n_head, int n_head_kv, int head_dim) override {
        double work = 0;
        bool prompt_deep = false;
        for (size_t e = 0; e < n_views; ++e) {
            work += (double)views[e].nq * (double)views[e].length;
            prompt_deep = prompt_deep || (views[e].nq > 1 && views[e].length >= deep);
        }
        bool decode = n_views > 0;
        for (size_t e = 0; e < n_views; ++e) decode = decode && views[e].nq == 1;
        double ms = work * 0.0004 + (decode ? reply_ms : 0) + (n_views && !decode ? fixed_ms : 0);
        if (!stairs.empty() && n_views && !decode) {
            size_t rows = 0;
            for (size_t e = 0; e < n_views; ++e) rows += views[e].nq > 1 ? views[e].nq : 0;
            ms = stairs.back().second;
            for (auto it = stairs.rbegin(); it != stairs.rend(); ++it)
                if (rows <= it->first) ms = it->second;
        }
        {
            std::lock_guard<std::mutex> lk(m);
            pending_ms += ms;
            for (size_t e = 0; e < n_views; ++e)
                if (views[e].nq > 1) calls.push_back({views[e].nq, views[e].length, n_views});
            if (prompt_deep) {
                deepest_ms = std::max(deepest_ms, ms);
                ++deep_calls;
            }
        }
        NoCpu::attention(q, layer, views, n_views, out, n_head, n_head_kv, head_dim);
    }
};

// A prompt read beside a reply on a device whose attention grows with depth (docs/CACHE-PLAN.md, section 8 item 3): its slices are cut by the last piece's time, starting at a probe of one step.
// Whole ubatch slices take about 300 ms a piece at its end; every row stays in its class whatever the slices, so each reply is its reply alone, which a ubatch of 16 gave.
void pieces_by_time(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    const std::vector<Req> beside{{prompt_of(51, 4, vocab), 100}, {prompt_of(52, 1800, vocab), 3}};
    const std::vector<Reply> own = alone(on(weights, [] { return cpus(1); }), tok, pool, beside);
    std::shared_ptr<Deep> device;
    auto model = on(weights, [&seen, &device] {
        device = std::make_shared<Deep>(&seen);
        return std::vector<backend::BackendPtr>{device};
    })(pool, 512);
    device->deep = 1400;
    server::Scheduler::Stats s;
    const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &s, 0, 0, nullptr, 0, false, server::Pacing::automatic());
    for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], "a prompt cut by its pieces' time, request " + std::to_string(i));
    const std::string seen_text = std::to_string(s.between) + " passes between pieces, the longest piece " + std::to_string(s.longest_piece_ms) + " ms, the longest attention past 1400 tokens " +
                                  std::to_string(device->deepest_ms) + " ms of " + std::to_string(device->deep_calls) + ", " + std::to_string(s.apart) + " passes apart";
    std::cout << "server-passes-cpu: a prompt cut by its pieces' time: " << seen_text << std::endl;
    // How far the slices grow follows the search's target, which takes the reply's pass and the probe's piece as this runner measures them (PieceSearch::target), so no time bounds the attention here.
    // Whether slices grow past the probe, so the reply goes apart between pieces, is the runner's speed (a hosted macOS runner's pieces took 151 ms, none apart), which the 27B gate measures.
    require(device->deep_calls > 0, "a prompt cut by its pieces' time: " + seen_text + ", against some");
    // The record the health gives (SliceRecord): the stage's last slice beside the reply, its pieces two a layer and the tail after the last, its longest one of them, and the prompts' passes counted.
    const server::SliceRecord& last = s.slices.empty() ? server::SliceRecord{} : s.slices[0];
    const std::string part = last.longest_part;
    require(s.slices.size() == 1 && last.rows > 0 && last.pieces == 2 * (size_t)kSplit.layers + 1 && last.longest_ms <= last.pieces_ms && last.longest_layer >= -1 &&
                last.longest_layer < kSplit.layers && (part == "mixer" || part == "ffn" || part == "tail") && (part == "tail") == (last.longest_layer == -1) && s.prompt_passes >= 4,
            "a prompt cut by its pieces' time: the last slice's record holds " + std::to_string(last.rows) + " rows, " + std::to_string(last.pieces) + " pieces, the longest " + part + " of layer " +
                std::to_string(last.longest_layer) + ", and " + std::to_string(s.prompt_passes) + " prompt passes, against some rows, " + std::to_string(2 * kSplit.layers + 1) + " pieces and 4 passes at least");
    // The cut starts at a probe of one step in a pass of its own.
    require(std::any_of(device->calls.begin(), device->calls.end(), [](const Deep::Call& c) { return c.rows == server::kPieceStep && c.views == 1; }), "a prompt cut by its pieces' time: no probe of one step alone");
}

// A prompt starting deep beside a reply (docs/CACHE-PLAN.md, section 8 item 3): a follow-up of a 1600-token conversation arrives with a shallow prompt's pieces the last measured.
// Its first slice is not cut by their ratio but starts at one step and grows by its own, and every reply is its reply alone.
void pieces_at_new_depth(const gguf::GGUFModel& weights, uint32_t vocab) {
    const std::string what = "a prompt at a new depth";
    const bpe::Tokenizer tok(weights);
    const size_t pool = 64 * kBlock;
    Submitters seen;
    std::shared_ptr<Deep> device;
    auto model = on(weights, [&seen, &device] {
        device = std::make_shared<Deep>(&seen);
        return std::vector<backend::BackendPtr>{device};
    })(pool, 512);
    device->deep = 1400;
    const Req conv{prompt_of(53, 1600, vocab), 3}, reply{prompt_of(51, 4, vocab), 1900}, shallow{prompt_of(52, 150, vocab), 3};
    Req follow{conv.prompt, 3};
    std::vector<Reply> got(3);
    size_t reused = 0;
    {
        server::Scheduler sched(*model, tok, kSeqs, 64, 0, false, 0, nullptr, 0, false, {}, server::Pacing::automatic());
        std::thread runner([&] { sched.run(); });
        try {
            for (const auto& t : drain(*sched.submit(conv.prompt, params_of(conv)))) follow.prompt.push_back(t.id);
            const std::vector<uint32_t> more = prompt_of(54, 300, vocab);
            follow.prompt.insert(follow.prompt.end(), more.begin(), more.end());
            {
                std::lock_guard<std::mutex> lk(device->m);
                device->deepest_ms = 0;
                device->deep_calls = 0;
                device->calls.clear();
            }
            const auto r = sched.submit(reply.prompt, params_of(reply));
            const auto p1 = sched.submit(shallow.prompt, params_of(shallow));
            got[1] = drain(*p1);
            const auto p2 = sched.submit(follow.prompt, params_of(follow));
            got[2] = drain(*p2);
            reused = p2->reused();
            got[0] = drain(*r);
        } catch (...) {
            sched.stop();
            runner.join();
            throw;
        }
        sched.stop();
        runner.join();
    }
    const std::vector<Req> reqs{reply, shallow, follow};
    const std::vector<Reply> own = alone(on(weights, [] { return cpus(1); }), tok, pool, reqs);
    for (size_t i = 0; i < reqs.size(); ++i) same(own[i], got[i], what + ", request " + std::to_string(i));
    const std::string seen_text = "the follow-up reused " + std::to_string(reused) + " tokens, the longest attention past 1400 tokens " + std::to_string(device->deepest_ms) + " ms of " +
                                  std::to_string(device->deep_calls);
    std::cout << "server-passes-cpu: " << what << ": " << seen_text << std::endl;
    require(reused >= 1536 && device->deep_calls > 0, what + ": " + seen_text + ", against at least 1536 reused and some");
    // Its first slice past 1400 tokens is the probe, in a pass the reply does not ride; growth past it is the runner's speed, which this does not hold.
    const auto first = std::find_if(device->calls.begin(), device->calls.end(), [](const Deep::Call& c) { return c.length >= 1400; });
    require(first != device->calls.end() && first->rows == server::kPieceStep && first->views == 1,
            what + ": the first slice past 1400 tokens took " + (first == device->calls.end() ? std::string("none") : std::to_string(first->rows) + " rows in a call of " + std::to_string(first->views) + " views") +
                ", against a probe of 16 alone");
}

// A prompt read beside a reply whose own pass is slower than the piece budget, as a reply's at 211k tokens is (docs/CACHE-PLAN.md, section 8 item 3, F2DEV 2026-10-19 02:40): each decode call costs 150 ms on the device.
// How far its pieces grow follows the passes' measured time, which a slow runner changes, so this holds only that passes go apart and reach past 800 tokens, each reply its reply alone; the longest is printed.
void pieces_beside_slow_reply(const gguf::GGUFModel& weights, uint32_t vocab) {
    const std::string what = "a prompt beside a slow reply";
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    const std::vector<Req> beside{{prompt_of(51, 4, vocab), 40}, {prompt_of(52, 1800, vocab), 3}};
    const std::vector<Reply> own = alone(on(weights, [] { return cpus(1); }), tok, pool, beside);
    std::shared_ptr<Deep> device;
    auto model = on(weights, [&seen, &device] {
        device = std::make_shared<Deep>(&seen);
        return std::vector<backend::BackendPtr>{device};
    })(pool, 512);
    device->deep = 800;
    device->reply_ms = 150;
    server::Scheduler::Stats s;
    const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &s, 0, 0, nullptr, 0, false, server::Pacing::automatic());
    for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], what + ", request " + std::to_string(i));
    const std::string seen_text = std::to_string(s.apart) + " passes apart, the longest attention past 800 tokens " + std::to_string(device->deepest_ms) + " ms of " + std::to_string(device->deep_calls);
    std::cout << "server-passes-cpu: " << what << ": " << seen_text << std::endl;
    require(s.apart > 0 && device->deep_calls > 0, what + ": " + seen_text + ", against some passes apart and a slice past 800 tokens");
}

// After the probe doubles, measured host work can reopen the bracket at rounded-half of the current cut or grow it up to double.
// Only the final prompt tail may be shorter or unaligned.
bool adaptive_cuts(const std::vector<size_t>& cut) {
    bool adaptive = cut.size() >= 2 && cut[0] == server::kPieceStep && cut[1] == 2 * server::kPieceStep;
    for (size_t i = 1; adaptive && i < cut.size(); ++i) {
        const size_t lower = std::max(server::kPieceStep, cut[i - 1] / 2 / server::kPieceStep * server::kPieceStep);
        adaptive = cut[i] > 0 && cut[i] <= 2 * cut[i - 1] &&
                   (i + 1 == cut.size() || (cut[i] % server::kPieceStep == 0 && cut[i] >= lower));
    }
    return adaptive;
}

// A prompt beside a reply whose prompt attention has a fixed 60 ms a call, past half the target, where a linear cut kept every slice at one step (F2DEV, 2026-10-19 05:57).
// Held: one step, then two, each reply its reply alone; later cuts follow a runner's measured host time, so their growth is held by the policy tests.
void pieces_fixed_cost(const gguf::GGUFModel& weights, uint32_t vocab) {
    const std::string what = "a prompt beside a reply whose attention has a fixed part";
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    const std::vector<Req> beside{{prompt_of(51, 4, vocab), 200}, {prompt_of(52, 1800, vocab), 3}};
    const std::vector<Reply> own = alone(on(weights, [] { return cpus(1); }), tok, pool, beside);
    std::shared_ptr<Deep> device;
    auto model = on(weights, [&seen, &device] {
        device = std::make_shared<Deep>(&seen);
        return std::vector<backend::BackendPtr>{device};
    })(pool, 512);
    device->deep = 1u << 30;
    device->fixed_ms = 60;
    server::Scheduler::Stats s;
    const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &s, 0, 0, nullptr, 0, false, server::Pacing::automatic());
    for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], what + ", request " + std::to_string(i));
    // The slices cut alone in their passes from the first probe on, in order: a pass's layers each give a call of the same rows and history.
    std::vector<size_t> cut;
    {
        std::lock_guard<std::mutex> lk(device->m);
        size_t length = 0;
        for (const Deep::Call& c : device->calls) {
            if (c.views != 1 || (cut.empty() && c.rows != server::kPieceStep) || (!cut.empty() && c.rows == cut.back() && c.length == length)) continue;
            cut.push_back(c.rows);
            length = c.length;
        }
    }
    std::string seen_text;
    for (size_t r : cut) seen_text += (seen_text.empty() ? "" : ", ") + std::to_string(r);
    std::cout << "server-passes-cpu: " << what << ": slices of " << seen_text << " rows after the probe" << std::endl;
    require(cut.size() >= 2 && cut[0] == server::kPieceStep && cut[1] == 2 * server::kPieceStep,
            what + ": slices of " + seen_text + " rows after the probe, against one step and then two");
    require(s.automatic && s.overlaps > 0 && s.between > 0, what + ": automatic service between measured prompt pieces did not run");
}

// A prompt beside a reply on staircase attention: 100 ms through 48 rows, 175 through 96, and 300 past.
// Actual CPU work and host wakeup delay also enter the measured cost, so later slow pieces can reopen an earlier bracket; deterministic policy tests check exact timing responses.
void pieces_staircase(const gguf::GGUFModel& weights, uint32_t vocab) {
    const std::string what = "a prompt beside a reply whose attention is a staircase";
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    const std::vector<Req> beside{{prompt_of(51, 4, vocab), 400}, {prompt_of(53, 1000, vocab), 3}};
    const std::vector<Reply> own = alone(on(weights, [] { return cpus(1); }), tok, pool, beside);
    std::shared_ptr<Deep> device;
    auto model = on(weights, [&seen, &device] {
        device = std::make_shared<Deep>(&seen);
        return std::vector<backend::BackendPtr>{device};
    })(pool, 512);
    device->deep = 1u << 30;
    device->stairs = {{48, 100}, {96, 175}, {std::numeric_limits<size_t>::max(), 300}};
    server::Scheduler::Stats s;
    const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &s, 0, 0, nullptr, 0, false, server::Pacing::automatic());
    for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], what + ", request " + std::to_string(i));
    // The slices cut alone in their passes from the first probe on, in order: a pass's layers each give a call of the same rows and history.
    std::vector<size_t> cut;
    {
        std::lock_guard<std::mutex> lk(device->m);
        size_t length = 0;
        for (const Deep::Call& c : device->calls) {
            if (c.views != 1 || (cut.empty() && c.rows != server::kPieceStep) || (!cut.empty() && c.rows == cut.back() && c.length == length)) continue;
            cut.push_back(c.rows);
            length = c.length;
        }
    }
    std::string seen_text;
    for (size_t r : cut) seen_text += (seen_text.empty() ? "" : ", ") + std::to_string(r);
    std::cout << "server-passes-cpu: " << what << ": slices of " << seen_text << " rows after the probe" << std::endl;
    require(adaptive_cuts(cut), what + ": slices of " + seen_text + " rows after the probe, against a doubled probe then step-aligned adaptive cuts, with only the final tail allowed short");
    require(s.automatic && s.overlaps > 0 && s.between > 0 && s.longest_piece_ms > 0,
            what + ": automatic service between measured prompt pieces did not run");
}

// A short prompt arriving while a long one is read beside a reply under the automatic rule (docs/SERVER.md, Short prompts beside a long one), on one device whose attention grows with depth.
// It is read in a pass of its own, counted as short and as a prompt pass, and the long prompt's slices keep their search across it (adaptive_cuts).
void short_beside_automatic(const gguf::GGUFModel& weights, uint32_t vocab) {
    const std::string what = "a short prompt behind a long one beside a reply under the automatic rule";
    const bpe::Tokenizer tok(weights);
    const size_t pool = 32 * kBlock;
    Submitters seen;
    const Req reply{prompt_of(54, 4, vocab), 1000}, long_one{prompt_of(55, 1800, vocab), 3}, short_one{prompt_of(59, 8, vocab), 3};
    const std::vector<Reply> own = alone(on(weights, [] { return cpus(1); }), tok, pool, {reply, long_one, short_one});
    std::shared_ptr<Deep> device;
    auto model = on(weights, [&seen, &device] {
        device = std::make_shared<Deep>(&seen);
        return std::vector<backend::BackendPtr>{device};
    })(pool, 512);
    device->deep = 1u << 30;
    server::Scheduler sched(*model, tok, kSeqs, 64, 0, false, 0, nullptr, 0, false, {}, server::Pacing::automatic());
    std::shared_ptr<server::Request> first, late;
    std::atomic<bool> sent{false};
    size_t arrival_cut = 0, pass = 0, long_end = 0, short_end = 0;
    uint64_t prompt_passes = 0;
    // The short prompt is submitted on the scheduler thread once a slice of the long one past its first has grown to four steps, or the long one is half read.
    sched.on_retire = [&](const server::Scheduler::Retired& t) {
        bool prompt_rows = false;
        for (size_t e = 0; e < t.requests.size(); ++e) {
            prompt_rows = prompt_rows || t.from[e] < t.requests[e]->prompt_tokens();
            if (t.from[e] + t.rows[e] == t.requests[e]->prompt_tokens()) {
                if (t.requests[e] == first.get() && !long_end) long_end = pass + 1;
                if (late && t.requests[e] == late.get() && !short_end) short_end = pass + 1;
            }
            if (sent.load() || t.requests[e] != first.get() || !t.from[e]) continue;
            if (t.rows[e] >= 4 * server::kPieceStep || 2 * (t.from[e] + t.rows[e]) >= long_one.prompt.size()) {
                arrival_cut = t.rows[e];
                late = sched.submit(short_one.prompt, params_of(short_one));
                sent.store(true);
            }
        }
        prompt_passes += prompt_rows;
        ++pass;
    };
    const auto talking = sched.submit(reply.prompt, params_of(reply));
    first = sched.submit(long_one.prompt, params_of(long_one));
    std::thread runner([&] { sched.run(); });
    std::vector<Reply> got;
    server::Scheduler::Stats s;
    try {
        got.push_back(drain(*talking));
        got.push_back(drain(*first));
        require(sent.load(), what + ": the short prompt was never sent");
        got.push_back(drain(*late));
        s = sched.stats();
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    for (size_t i = 0; i < own.size(); ++i) same(own[i], got[i], what + ", request " + std::to_string(i));
    // The long prompt's slices cut alone in their passes from the first probe on, in order, and the slice before the short prompt's pass; every prompt at depth 0 is another's or its first pass.
    std::vector<size_t> cut;
    size_t before_short = 0;
    {
        std::lock_guard<std::mutex> lk(device->m);
        size_t length = 0;
        for (const Deep::Call& c : device->calls) {
            if (c.views != 1) continue;
            if (!c.length) {
                if (!cut.empty() && c.rows == short_one.prompt.size() && !before_short) before_short = cut.back();
                continue;
            }
            if ((cut.empty() && c.rows != server::kPieceStep) || (!cut.empty() && c.rows == cut.back() && c.length == length)) continue;
            cut.push_back(c.rows);
            length = c.length;
        }
    }
    std::string seen_text;
    for (size_t r : cut) seen_text += (seen_text.empty() ? "" : ", ") + std::to_string(r);
    std::cout << "server-passes-cpu: " << what << ": slices of " << seen_text << " rows after the probe, the short prompt sent after a slice of " << arrival_cut
              << " rows and read after one of " << before_short << ", " << s.short_passes << " short passes of " << s.prompt_passes << " prompt passes" << std::endl;
    require(s.automatic && s.overlaps > 0 && s.between > 0, what + ": automatic service between measured prompt pieces did not run");
    require(s.short_passes > 0, what + ": no pass was counted as reading a short prompt ahead");
    require(adaptive_cuts(cut), what + ": slices of " + seen_text + " rows after the probe, against a doubled probe then step-aligned adaptive cuts, with only the final tail allowed short");
    require(before_short > 0, what + ": the short prompt was not read in a pass of its own between the long prompt's slices");
    require(short_end && long_end && short_end < long_end, what + ": the short prompt ended in pass " + std::to_string(short_end) + ", not before the long one's " + std::to_string(long_end));
    // A short pass is a prompt pass too (Stats::prompt_passes counts every pass that carried prompt rows).
    require(s.prompt_passes == prompt_passes, what + ": " + std::to_string(s.prompt_passes) + " prompt passes counted against " + std::to_string(prompt_passes) + " retired with prompt rows");
    ++checks;
}

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
    void draft(infer::Sequence*, const std::vector<uint32_t>& h, size_t k, std::vector<uint32_t>& out, const infer::spec::ImageRows& = {}) override {
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
    // Four drafts a request for 16 requests give the 64 draft rows a pass reserves at most.
    const size_t draft_max = drafting ? 4 : 0;
    request.draft_rows = arena ? server::Scheduler::draft_rows(seqs, draft_max) : 0;
    request.logit_rows = server::logit_rows(slots, logits ? server::Scheduler::logits_a_pass(seqs, draft_max) : seqs).size;
    request.apart_rows = server::Scheduler::apart_rows(seqs, draft_max);
    infer::ModelOptions options;
    options.kv_tokens = size_t(1) << 20;
    options.mark_slots = drafting ? 4 : 0;
    options.mark_rows = drafting ? draft_max + 1 : 0;
    auto model = infer::place_model(infer::gguf_weights(weights), {device}, request, options).model;
    infer::spec::Lookup lookup;
    server::Scheduler sched(*model, tok, seqs, 64, 0, false, 0, drafting ? &lookup : nullptr, draft_max, false, {}, server::Pacing::automatic());
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

// A boundary's state forked from host memory while a prompt is fed in pieces beside a reply (production, 2026-10-17 11:15:54): its copies go through the scheduler's hold, never into a held queue.
// A hybrid model, a fixed share that feeds a pass beside another request in pieces whatever the runner's times, and one device that is no CPU's.
void boundary_fork_held(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    Submitters seen;
    std::shared_ptr<Owned> device;
    const Make owned = on(weights, [&seen, &device] {
        device = std::make_shared<Owned>(&seen, 15);
        return std::vector<backend::BackendPtr>{device};
    }, 4, 3);
    auto model = owned(4096, 0);
    device->crossed = 0;
    std::vector<uint32_t> edit;
    Reply got;
    server::Scheduler::Stats stats;
    {
        server::Scheduler sched(*model, tok, 4, 64, 0, false, (size_t)1 << 30, nullptr, 0, false, {}, server::Pacing::fixed(0.5));
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
                    require(std::chrono::steady_clock::now() < until, "a boundary fork beside a paced prompt: a turn was not read again with its boundary kept in 60 seconds");
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
            require(r->next(first, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id, "a boundary fork beside a paced prompt: the reply gave no token");
            // The long prompt is read in pieces on the stage's thread; once that thread has submitted, the edit comes, watched until its first token.
            device->stage = std::thread::id();
            const auto l = sched.submit(long_prompt, params_of(Req{long_prompt, 8}));
            for (const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(60); device->stage.load() == std::thread::id();) {
                require(std::chrono::steady_clock::now() < until, "a boundary fork beside a paced prompt: no stage's thread submitted in 60 seconds");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            device->watching = true;
            const auto e = sched.submit(edit, params_of(Req{edit, 32}));
            server::Request::Token edited;
            const bool came = e->next(edited, server::Request::Clock::now() + std::chrono::seconds(120)) == server::Request::Next::id;
            device->watching = false;
            require(came, "a boundary fork beside a paced prompt: the edit gave no token");
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
    require(stats.boundary_hits >= 1, "a boundary fork beside a paced prompt: " + std::to_string(stats.boundary_hits) + " requests forked a boundary, against one at least");
    std::cout << "server-passes-cpu: a boundary fork beside a paced prompt: " << device->waited.load() << " calls of the scheduler's thread watched, " << device->crossed.load() << " met a held queue\n";
    require(device->waited > 0, "a boundary fork beside a paced prompt: no call of the scheduler's thread was watched");
    require(device->crossed == 0, "a boundary fork beside a paced prompt: " + std::to_string(device->crossed.load()) + " calls met a queue a stage's submission held");
    const Make plain = on(weights, [] { return cpus(1); }, 3, 3);
    auto fresh = plain(4096, 0);
    same(serve(*fresh, tok, 3, {{Req{edit, 32}}})[0], got, "a boundary fork beside a paced prompt, the edit's reply");
    ++checks;
}

// The interleaved rule (server::Pacing::interleaved) on layer splits of two and three devices, none a CPU, at P = S and S + 1, every reply its reply alone.
// Requests in turn form no interleaved pass, beside a reply each prompt slice gets a pass of its own, and once a prompt beside a reply is sampled its request decodes in the normal round.
void interleaved(const gguf::GGUFModel& weights, uint32_t vocab) {
    const bpe::Tokenizer tok(weights);
    const size_t pool = 16 * kBlock;
    Submitters seen;
    const Make one = on(weights, [] { return cpus(1); });
    const std::vector<Req> beside{{prompt_of(51, 4, vocab), 24}, {prompt_of(52, 96, vocab), 3}, {prompt_of(53, 80, vocab), 2, {}, 0.8f, 5}};
    const std::vector<Req> pair{{prompt_of(54, 4, vocab), 60}, {prompt_of(55, 96, vocab), 24}};
    const std::vector<Reply> own = alone(one, tok, pool, beside), own_pair = alone(one, tok, pool, pair);
    std::vector<std::vector<Req>> in_turn;
    for (const Req& r : beside) in_turn.push_back({r});
    for (const size_t stages : {size_t(2), size_t(3)})
        for (const size_t passes : {stages, stages + 1}) {
            const std::string where = "the interleaved rule on " + std::to_string(stages) + " devices at P = " + std::to_string(passes);
            const Make split = on(weights, [&seen, stages] {
                std::vector<backend::BackendPtr> v;
                for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
                return v;
            });
            {
                auto model = split(pool, kUbatch);
                server::Scheduler::Stats s;
                const std::vector<Reply> got = serve(*model, tok, kSeqs, in_turn, &s, passes, 0, nullptr, 0, false, server::Pacing::interleaved());
                for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], where + ", requests in turn, request " + std::to_string(i));
                require(s.interleaved && s.interleaved_replies == 0 && s.interleaved_slices == 0 && s.interleaved_mixed == 0, where + ": requests in turn formed passes apart from each other");
            }
            {
                auto model = split(pool, kUbatch);
                server::Scheduler::Stats s;
                const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &s, passes, 0, nullptr, 0, false, server::Pacing::interleaved());
                for (size_t i = 0; i < beside.size(); ++i) same(own[i], got[i], where + ", a reply beside two prompts, request " + std::to_string(i));
                require(s.interleaved_replies > 0 && s.interleaved_slices > 0,
                        where + ": " + std::to_string(s.interleaved_replies) + " passes of generated rows and " + std::to_string(s.interleaved_slices) + " prompt slices formed apart beside a reply");
                require(s.interleaved_mixed == 0, where + ": " + std::to_string(s.interleaved_mixed) + " passes in an overlap held a reply's rows beside a prompt's or two prompts' slices");
                // Each stage's last slice and its service record (docs/SERVER.md, The price table): every pass timed at every stage, the later ones predicted, and the time split.
                for (size_t st = 0; st < stages; ++st) {
                    // A CPU stage has retired when its timing starts, so its time may read 0 where the clock does not advance between two reads.
                    require(s.slices[st].rows > 0 && s.slices[st].pieces == 1 && s.slices[st].longest_ms >= 0,
                            where + ": stage " + std::to_string(st) + "'s last slice reads empty: " + std::to_string(s.slices[st].rows) + " rows, " + std::to_string(s.slices[st].pieces) + " pieces, the longest " +
                                std::to_string(s.slices[st].longest_ms) + " ms");
                    const server::StageRecord& r = s.service.stages[st];
                    require(r.passes > 0 && r.predicted > 0 && r.measured_ms > 0 && r.recording_ms > 0 && !r.prices.decode.empty() && !r.prices.chunks.empty(),
                            where + ": stage " + std::to_string(st) + "'s service record holds " + std::to_string(r.passes) + " passes, " + std::to_string(r.predicted) + " predicted");
                }
                require(s.service.rows.decode > 0 && s.service.rows.prompt > 0 && s.service.first_token.count == beside.size() && s.service.inter_token.count > 0, where + ": the service record's rows or request times are empty");
                ledger(s, *model, where);
            }
            auto model = split(pool, kUbatch);
            server::Scheduler::Stats s;
            std::vector<std::vector<size_t>> retired;   // each retired pass's requests, by their place in `pair`
            size_t last = 0;                            // the pass that sampled the prompt's first token
            const auto observe = [&](const server::Scheduler::Retired& t) {
                std::vector<size_t> in;
                for (size_t e = 0; e < t.requests.size(); ++e) {
                    const size_t i = t.requests[e]->prompt_tokens() == pair[1].prompt.size();
                    in.push_back(i);
                    if (i && t.want[e] && t.from[e] + t.rows[e] == t.requests[e]->prompt_tokens()) last = retired.size();
                }
                retired.push_back(in);
            };
            const std::vector<Reply> got = serve(*model, tok, kSeqs, {pair}, &s, passes, 0, nullptr, 0, false, server::Pacing::interleaved(), observe);
            for (size_t i = 0; i < pair.size(); ++i) same(own_pair[i], got[i], where + ", a reply beside a prompt, request " + std::to_string(i));
            // The first request's last pass ends the window where both decode.
            size_t ends = retired.size();
            for (size_t i = 0; i < pair.size(); ++i) {
                size_t at = 0;
                for (size_t p = 0; p < retired.size(); ++p)
                    if (std::count(retired[p].begin(), retired[p].end(), i)) at = p;
                ends = std::min(ends, at);
            }
            require(last > 0 && ends > last + passes, where + ": the prompt was not sampled a lap before a request ended");
            require(s.interleaved_replies + s.interleaved_slices <= last + passes,
                    where + ": " + std::to_string(s.interleaved_replies + s.interleaved_slices) + " passes formed under it, past the " + std::to_string(last + 1) +
                        " retired when the prompt was sampled and the " + std::to_string(passes - 1) + " in flight then");
            bool next = false;
            for (size_t p = last + 1; p <= last + passes; ++p) next = next || std::count(retired[p].begin(), retired[p].end(), size_t(1));
            require(next, where + ": the prompt's request was in none of the " + std::to_string(passes) + " passes after its first token");
            for (size_t p = last + 1; p < ends; ++p)
                require(retired[p].size() == 1, where + ": pass " + std::to_string(p) + " after the prompt was sampled held " + std::to_string(retired[p].size()) + " of the two decoders");
            ledger(s, *model, where + ", a reply beside a prompt");
            // A short prompt behind a long one beside a reply (docs/SERVER.md, Short prompts beside a long one): read in a pass of its own before the long prompt ends, every reply its reply alone.
            {
                const std::vector<Req> three{{prompt_of(56, 4, vocab), 40}, {prompt_of(57, 300, vocab), 3}, {prompt_of(58, 3, vocab), 3}};
                const std::vector<Reply> own_three = alone(one, tok, pool, three);
                auto m3 = split(pool, kUbatch);
                server::Scheduler::Stats s3;
                size_t pass = 0, long_end = 0, short_end = 0;
                const auto ends = [&](const server::Scheduler::Retired& t) {
                    for (size_t e = 0; e < t.requests.size(); ++e)
                        if (t.from[e] + t.rows[e] == t.requests[e]->prompt_tokens()) {
                            if (t.requests[e]->prompt_tokens() == three[1].prompt.size() && !long_end) long_end = pass + 1;
                            if (t.requests[e]->prompt_tokens() == three[2].prompt.size() && !short_end) short_end = pass + 1;
                        }
                    ++pass;
                };
                const std::vector<Reply> got3 = serve(*m3, tok, kSeqs, {three}, &s3, passes, 0, nullptr, 0, false, server::Pacing::interleaved(), ends);
                for (size_t i = 0; i < three.size(); ++i) same(own_three[i], got3[i], where + ", a short prompt behind a long one beside a reply, request " + std::to_string(i));
                require(short_end && long_end && short_end < long_end, where + ": the short prompt ended in pass " + std::to_string(short_end) + ", not before the long one's " + std::to_string(long_end));
                require(s3.short_passes > 0, where + ": no pass beside the reply was counted as reading a short prompt ahead");
                ledger(s3, *m3, where + ", a short prompt behind a long one beside a reply");
            }
        }
}

// A prompt in consecutive passes (docs/MULTI-DEVICE.md, phase 4) on layer splits of two and three slow devices, none a CPU, at P = S and S + 1, without pacing and under the interleaved rule.
// A lone prompt of 300 tokens continues while its slice before is in flight (Stats::continued), and beside a reply only with a slot left for it; every reply is its reply alone and the ledger adds up.
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
            for (const bool rule : {false, true}) {
                const std::string where = "a continued prompt on " + std::to_string(stages) + " devices at P = " + std::to_string(passes) + (rule ? " under the interleaved rule" : "");
                const Make split = on(weights, [&seen, stages] {
                    std::vector<backend::BackendPtr> v;
                    for (size_t i = 0; i < stages; ++i) v.push_back(std::make_shared<Slow>(&seen, 2));
                    return v;
                });
                const server::Pacing pacing = rule ? server::Pacing::interleaved() : server::Pacing{};
                {
                    auto model = split(pool, kUbatch);
                    server::Scheduler::Stats st;
                    const std::vector<Reply> got = serve(*model, tok, kSeqs, {lone}, &st, passes, 0, nullptr, 0, false, pacing);
                    same(own_lone[0], got[0], where + ", alone");
                    require(st.continued > 0, where + ": a lone prompt never continued while its slice before was in flight");
                    ledger(st, *model, where + ", alone");
                }
                auto model = split(pool, kUbatch);
                server::Scheduler::Stats st;
                const std::vector<Reply> got = serve(*model, tok, kSeqs, {beside}, &st, passes, 0, nullptr, 0, false, pacing);
                for (size_t i = 0; i < beside.size(); ++i) same(own_beside[i], got[i], where + ", beside a reply, request " + std::to_string(i));
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

int main(int argc, char** argv) {
    try {
        // What a caller asks the fit for is what the scheduler reserves: a row a request and, where requests draft, a pass's draft rows, at most 64 and at most --draft-max a request, in each logits ring.
        using S = server::Scheduler;
        require(S::logits_a_pass(16, 0) == 16 && S::draft_rows(16, 0) == 0 && S::draft_rows(1, 3) == 3 && S::draft_rows(16, 3) == 48 && S::logits_a_pass(16, 3) == 64 &&
                    S::draft_rows(16, 31) == 64 && S::draft_rows(2, 31) == 62 && S::draft_rows(3, 31) == 64 && S::draft_rows(1, 63) == 63 && S::apart_rows(16, 31) == S::logits_a_pass(16, 31),
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
        pieces_by_time(served(kSplit), (uint32_t)kSplit.vocab);
        pieces_at_new_depth(served(kSplit), (uint32_t)kSplit.vocab);
        pieces_beside_slow_reply(served(kSplit), (uint32_t)kSplit.vocab);
        pieces_fixed_cost(served(kSplit), (uint32_t)kSplit.vocab);
        pieces_staircase(served(kSplit), (uint32_t)kSplit.vocab);
        short_beside_automatic(served(kSplit), (uint32_t)kSplit.vocab);
        drafted_service(served(kSplit), (uint32_t)kSplit.vocab);
        interleaved(served(kSplit), (uint32_t)kSplit.vocab);
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
