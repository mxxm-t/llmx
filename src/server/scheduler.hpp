#pragma once
// The scheduler of docs/SERVER.md: one thread drives the model, one Model::forward per iteration carrying every decoding request's next token and a slice of a prefilling prompt, sampling each request's logits into its channel.
// Admission is by the KV pool's budget, in queue order. A capped request reserves its whole reach up front and is never paused; an uncapped one reserves its prompt and grows as it generates. Room goes by first admission (make_room): a request that cannot grow takes donors, then pauses uncapped requests admitted after it, and otherwise sits out passes until room frees; paused requests wait apart from the queue and resume oldest first, where they stopped. Finished requests stay as prefix donors, whose full blocks a repeating prompt forks.
// A request records how each stretch of its history was computed (RowClass), and a resume recomputes what its cache lacks the same way, so a paused request gives the logits it gives when never paused.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "inference/logprobs.hpp"
#include "inference/sampler.hpp"
#include "model/arch_qwen.hpp"
#include "tokenizer/tokenizer.hpp"

namespace server {

// A request's sampling settings, with the defaults and ranges of infer::Sampling, and what only a request has: several stop texts and no cap.
struct SampleParams : infer::Sampling {
    std::vector<std::string> stop;
    // No cap from the client: submit sets max_tokens to what the request may hold, and its blocks are reserved as it grows.
    bool until_limit = false;
    // Each sampled token's log-probability, and the `top_logprobs` most likely tokens at its position, on its channel (Request::Token); computed only when asked.
    // They come from the logits row the sampler reads, before the penalty, the temperature, top-k and top-p: the model's own distribution.
    bool logprobs = false;
    size_t top_logprobs = 0;
};

// A stretch of a history computed one way: the rows before `end`, from the stretch before it on, took this extent and fresh count (infer::BatchEntry).
// A request's prompt takes its extent and the tokens its first admission prefilled, a generated token extent 1, and a prefix forked at the first admission keeps the stretches its donor recorded.
struct RowClass {
    size_t end, extent, fresh;
};

// A running request as make_room sees it: its first admission, whether it may be paused (an uncapped one), the blocks it has reserved, and those its history would keep as a donor once paused, per pool.
struct Holder {
    uint64_t admission;
    bool uncapped;
    std::vector<size_t> need, kept;
};

// What make_room takes, in the order it takes it: donors by index, then running requests to pause by index, each with whether the donor its history becomes goes too.
struct Taken {
    bool enough = false;
    std::vector<size_t> donors;
    std::vector<std::pair<size_t, bool>> paused;
};

// Who gives up blocks for whom: the one owner of that rule (docs/SERVER.md, room by first admission).
// A request needing `need` more blocks in each pool, beside `reserved` of `pool`, takes donors first, oldest first, the donor it forks (`keep`) last, or first when it shares every full block of it (`keep_first`).
// Only a request that grows (`preempt`) then pauses uncapped requests admitted after its own first admission (`after`), latest first, taking what each frees and then, if that is short, the donor its history becomes.
// Nothing is taken unless what may be taken is enough, so a request refused waits with nothing evicted for it: to be admitted later, or to grow at a later pass.
inline Taken make_room(const std::vector<size_t>& pool, const std::vector<size_t>& reserved, const std::vector<std::vector<size_t>>& donors,
                       size_t keep, bool keep_first, const std::vector<Holder>& active, uint64_t after, bool preempt,
                       const std::vector<size_t>& need) {
    std::vector<size_t> free(pool.size());
    for (size_t s = 0; s < pool.size(); ++s) free[s] = pool[s] - reserved[s];
    const auto fits = [&] {
        for (size_t s = 0; s < need.size(); ++s)
            if (need[s] > free[s]) return false;
        return true;
    };
    const auto gain = [&](const std::vector<size_t>& b, const std::vector<size_t>& less) {
        for (size_t s = 0; s < free.size(); ++s) {
            const size_t l = s < less.size() ? less[s] : 0, v = s < b.size() ? b[s] : 0;
            free[s] += v > l ? v - l : 0;
        }
    };
    Taken t;
    t.enough = fits();
    std::vector<size_t> order;
    if (keep < donors.size() && keep_first) order.push_back(keep);
    for (size_t d = 0; d < donors.size(); ++d)
        if (d != keep) order.push_back(d);
    if (keep < donors.size() && !keep_first) order.push_back(keep);
    for (size_t i = 0; i < order.size() && !t.enough; ++i) {
        gain(donors[order[i]], {});
        t.donors.push_back(order[i]);
        t.enough = fits();
    }
    if (preempt && !t.enough) {
        std::vector<size_t> later;
        for (size_t i = 0; i < active.size(); ++i)
            if (active[i].uncapped && active[i].admission > after) later.push_back(i);
        std::sort(later.begin(), later.end(), [&](size_t a, size_t b) { return active[a].admission > active[b].admission; });
        for (size_t i = 0; i < later.size() && !t.enough; ++i) {
            const Holder& h = active[later[i]];
            gain(h.need, h.kept);
            t.paused.push_back({later[i], false});
            if ((t.enough = fits()) || std::all_of(h.kept.begin(), h.kept.end(), [](size_t b) { return b == 0; })) continue;
            gain(h.kept, {});
            t.paused.back().second = true;
            t.enough = fits();
        }
    }
    if (!t.enough) return Taken{};
    return t;
}

// One request from submission to completion.
// The connection thread reads the channel: `next` waits for a token or the end until a deadline, so the thread can look at its client between tokens.
// Everything below the channel belongs to the scheduler thread.
class Request {
public:
    using Clock = std::chrono::steady_clock;

    Request(std::vector<uint32_t> prompt, SampleParams params)
        : prompt_(std::move(prompt)), params_(std::move(params)), submitted_(Clock::now()) {}

    // A sampled token as the channel delivers it; with logprobs asked, its log-probability and the most likely tokens at its position, most likely first.
    struct Token {
        uint32_t id = 0;
        float logprob = 0.0f;
        std::vector<infer::TokenLogprob> top;
        // With logprobs asked, the logits row the id was sampled from, which the channel carries to next and next turns into the values.
        std::vector<float> row;
    };
    // The most tokens of a request that wait on its channel with their rows.
    // A reader this far behind gets the next tokens' values computed by the scheduler instead, so however slowly a client reads, a request holds this many rows and two more.
    static constexpr size_t kRowsWaiting = 8;
    // What a wait on the channel found: a sampled token, the end of the request with `finish` set ("eos", "stop", "length", "cancel" or "error"), or neither by the deadline.
    // A token that comes with its row gets its log-probabilities here, in the thread that reads the channel, so no pass waits for their walk over the vocabulary.
    // The row then goes back to the request for a later pass to fill, and a cancelled request's rows are dropped unread.
    enum class Next { id, end, timeout };
    Next next(Token& t, Clock::time_point until) {
        {
            std::unique_lock<std::mutex> lk(m_);
            if (!cv_.wait_until(lk, until, [&] { return !out_.empty() || done_; })) return Next::timeout;
            if (out_.empty()) return Next::end;
            t = std::move(out_.front());
            out_.pop_front();
            if (!t.row.empty()) --rows_;
        }
        if (!t.row.empty()) {
            if (cancel_.load()) {
                t.row = std::vector<float>();
                return Next::id;
            }
            fill(t, t.row, params_.top_logprobs);
            std::lock_guard<std::mutex> lk(m_);
            spare_.push_back(std::move(t.row));
            t.row = std::vector<float>();
        }
        return Next::id;
    }
    // Tokens on the channel still carrying their rows, at most kRowsWaiting.
    size_t rows_waiting() const {
        std::lock_guard<std::mutex> lk(m_);
        return rows_;
    }
    std::string finish() const {
        std::lock_guard<std::mutex> lk(m_);
        return finish_;
    }
    std::string error() const {
        std::lock_guard<std::mutex> lk(m_);
        return error_;
    }
    // Set by the connection thread when the client goes away; at its next iteration the scheduler drops the request from the queue or from the batch.
    void cancel() { cancel_.store(true); }
    // The client's prompt tokens.
    size_t prompt_tokens() const { return prompt_.size(); }
    // Prompt tokens taken from a donor's cache rather than prefilled.
    size_t reused() const { return reused_.load(); }
    // Milliseconds from admission to the first token, and from the first token to the end; read once the request has ended.
    struct Timings {
        double queued_ms = 0, prompt_ms = 0, predicted_ms = 0;
        // The tokens predicted_ms covers: those after the first, which came with the prompt's pass.
        static size_t predicted(size_t tokens) { return tokens > 1 ? tokens - 1 : 0; }
        static double per_second(size_t n, double ms) { return ms > 0 ? 1000.0 * (double)n / ms : 0.0; }
        double predicted_per_second(size_t tokens) const { return per_second(predicted(tokens), predicted_ms); }
    };
    Timings timings() const {
        std::lock_guard<std::mutex> lk(m_);
        auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        Timings t;
        if (admitted_ != Clock::time_point{}) t.queued_ms = ms(submitted_, admitted_);
        if (first_ != Clock::time_point{}) t.prompt_ms = ms(admitted_, first_), t.predicted_ms = ms(first_, ended_);
        return t;
    }

private:
    friend class Scheduler;
    // A token's log-probability and the `top` most likely tokens at its position, from the row it was sampled from.
    static void fill(Token& t, const std::vector<float>& row, size_t top) {
        const double lse = infer::log_sum_exp(row.data(), row.size());
        t.logprob = infer::logprob(row.data(), lse, t.id);
        t.top = infer::top_logprobs(row.data(), row.size(), lse, top);
    }
    // A token onto the channel; when `spare` is given, a row next has finished with comes back through it.
    void push(Token t, std::vector<float>* spare = nullptr) {
        std::lock_guard<std::mutex> lk(m_);
        if (first_ == Clock::time_point{}) first_ = Clock::now();
        if (!t.row.empty()) ++rows_;
        if (spare && !spare_.empty()) {
            *spare = std::move(spare_.back());
            spare_.pop_back();
        }
        out_.push_back(std::move(t));
        cv_.notify_all();
    }
    void end(const std::string& why, const std::string& err = "") {
        std::lock_guard<std::mutex> lk(m_);
        ended_ = Clock::now();
        finish_ = why;
        error_ = err;
        done_ = true;
        cv_.notify_all();
    }

    const std::vector<uint32_t> prompt_;
    SampleParams params_;   // never changed once made, since next reads it in the connection thread
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Token> out_;
    size_t rows_ = 0;                          // tokens in out_ carrying their rows
    std::vector<std::vector<float>> spare_;    // rows next has finished with, for push to hand back
    bool done_ = false;
    std::string finish_, error_;
    std::atomic<bool> cancel_{false};
    std::atomic<size_t> reused_{0};
    Clock::time_point submitted_, admitted_, first_, ended_;   // admitted_ is set once, under m_

    // Scheduler state.
    // The history is the prompt then the generated tokens, neither ever rewritten; the cache holds its first seq_.length(), which is all the progress there is.
    infer::Sequence seq_;
    std::string finish_pending_;   // set by a sampled end, acted on after the pass
    std::vector<size_t> need_;     // blocks reserved for it, per cache pool
    std::vector<RowClass> classes_;   // how its history was computed, stretch by stretch; empty until the first admission
    uint32_t last_id_ = 0;
    std::vector<uint32_t> gen_;
    std::string decoded_;
    infer::RNG rng_;
    std::vector<float> logits_;
    uint64_t admission_ = 0;       // order of first admission, by which room goes; set once
    uint64_t donor_ = 0;           // the donor its last pause left, which it takes back whole on resuming unless something evicted it
    bool stalled_ = false;         // it could not grow before this pass and sits it out
    size_t stalls_ = 0;            // passes it sat out
    size_t pauses_ = 0;            // times it was paused
    size_t taken_back_ = 0;        // resumes that took its donor back
    size_t reached_ = 0;           // the longest history its cache has held, past which nothing is recomputed
    size_t recomputed_ = 0;        // rows its resumes computed again
};

// The queue is full: the request is refused now rather than waiting.
struct QueueFull : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// A request whose prompt and max_tokens pass what one request may hold (Scheduler::token_limit), or an uncapped one whose prompt leaves no room.
struct TooLong : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Scheduler {
public:
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t max_queue)
        : model_(model), tok_(tok), max_seqs_(max_seqs), ubatch_(model.prefill_batch()),
          max_queue_(max_queue), reserved_(model.kv_pools(), 0) {}

    // Tokens one request may hold, prompt and reply together: the model context or the KV pool, whichever is smaller.
    size_t token_limit() const {
        return std::min((size_t)model_.config().context_length, model_.kv_tokens_total());
    }

    // Queue a request; the handle's channel delivers its tokens.
    // A prompt the limit cannot hold is refused here, before it waits, and so is a request arriving at a full queue; an uncapped request's max_tokens is the room its prompt leaves.
    std::shared_ptr<Request> submit(std::vector<uint32_t> prompt, SampleParams params) {
        if (prompt.empty()) throw std::runtime_error("server: empty prompt");
        if (params.until_limit) {
            if (prompt.size() >= token_limit())
                throw TooLong("the prompt fills the " + std::to_string(token_limit()) + " tokens a request may hold");
            params.max_tokens = (int)(token_limit() - prompt.size());
        }
        if (params.max_tokens <= 0) throw std::runtime_error("server: max_tokens must be positive");
        if (prompt.size() + (size_t)params.max_tokens > token_limit())
            throw TooLong("prompt plus max_tokens exceeds the " + std::to_string(token_limit()) + " tokens a request may hold");
        auto r = std::make_shared<Request>(std::move(prompt), std::move(params));
        {
            std::lock_guard<std::mutex> lk(m_);
            if (queue_.size() >= max_queue_)
                throw QueueFull("server: the queue holds " + std::to_string(max_queue_) + " requests; try again later");
            queue_.push_back(r);
        }
        cv_.notify_all();
        return r;
    }

    struct Stats {
        size_t active = 0, queued = 0, donors = 0, prefix_hits = 0, prefix_tokens = 0, pauses = 0;
        size_t paused = 0;       // requests paused now, waiting to resume
        size_t stalls = 0;       // passes a request sat out, unable to grow
        size_t recomputed = 0;   // rows resumes computed again
        size_t taken_back = 0;   // resumes that took their own donor back whole
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        return Stats{active_count_.load(), queue_.size(), donors_.size(), prefix_hits_, prefix_tokens_, (size_t)pauses_,
                     paused_count_.load(), stalls_, recomputed_, taken_back_};
    }

    // The loop, in the caller's thread, until stop().
    // The only caller of Model::forward for this model.
    void run() {
        std::vector<std::shared_ptr<Request>> active;   // in order of first admission
        std::vector<infer::BatchEntry> entries;
        std::vector<std::shared_ptr<Request>> wanting;
        std::vector<size_t> lengths;   // per active request, its history before the pass
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stopping_ || !queue_.empty() || !active.empty() || !paused_.empty(); });
                if (stopping_) break;
                // A waiting request whose client left ends wherever it waits, queued or paused, not only once admission reaches it, which may be after every active request has finished.
                for (auto* waiting : {&queue_, &paused_})
                    for (auto it = waiting->begin(); it != waiting->end();) {
                        if ((*it)->cancel_.load()) { (*it)->end("cancel"); it = waiting->erase(it); }
                        else ++it;
                    }
                // Room goes by first admission: nothing resumes or is admitted while a request that could not grow sits out the passes, paused requests resume oldest first and stop at the first that does not fit, and new requests come in queue order only once none is paused.
                const bool stalled = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; });
                while (!stalled && !paused_.empty() && active.size() < max_seqs_ && enter(paused_.front(), active)) paused_.pop_front();
                while (!stalled && paused_.empty() && !queue_.empty() && active.size() < max_seqs_) {
                    // A client can leave after the sweep, while the requests before its own are admitted; its request ends here, before any donor gives blocks up for it or it forks one.
                    if (queue_.front()->cancel_.load()) { queue_.front()->end("cancel"); queue_.pop_front(); continue; }
                    if (!enter(queue_.front(), active)) break;
                    queue_.pop_front();
                }
                active_count_.store(active.size());
                paused_count_.store(paused_.size());
            }
            // Cancelled requests leave before the pass.
            for (size_t i = 0; i < active.size();) {
                if (active[i]->cancel_.load()) finish(active, i, "cancel");
                else ++i;
            }
            if (active.empty()) continue;
            grow(active);
            if (active.empty()) continue;

            // Decode entries first, but for a request that could not grow, then what the other requests' caches lack, one stretch's slice each, up to ubatch tokens.
            entries.clear();
            wanting.clear();
            lengths.clear();
            size_t budget = ubatch_;
            for (auto& r : active) {
                lengths.push_back(r->seq_.length());
                if (!decoding(*r) || r->stalled_) continue;
                entries.push_back(infer::BatchEntry{&r->seq_, &r->last_id_, 1, true});
                wanting.push_back(r);
            }
            for (auto& r : active) {
                if (decoding(*r) || !budget) continue;
                const size_t at = r->seq_.length(), end = history_tokens(*r);
                const RowClass& c = class_at(r->classes_, at);
                size_t n = std::min(c.end, end) - at;
                if (c.extent == 1 && at < r->reached_) {
                    // Rows of extent 1 a resume computes again, generated tokens or a forked reply's: a pass takes at most kReplayRows of them, each costing ubatch / kReplayRows of the budget, which is at most the whole budget, so a request gets one while the budget is untouched.
                    // A one-token prompt read for the first time costs its row as any prompt does.
                    const size_t cost = std::max<size_t>(1, ubatch_ / kReplayRows);
                    n = std::min({n, kReplayRows, budget / cost});
                    if (!n) continue;
                    budget -= std::min(budget, n * cost);
                } else {
                    n = std::min(n, budget);
                    budget -= n;
                }
                // The entry that ends the history wants the logits the next token is sampled from.
                const bool last = at + n == end;
                entries.push_back(infer::BatchEntry{&r->seq_, token_ptr(*r, at), n, last});
                // The stretch's extent and fresh count, so its rows take the kernels, and a streamed layer the path, that first computed them.
                entries.back().extent = c.extent;
                entries.back().fresh = c.fresh;
                if (last) wanting.push_back(r);
            }
            try {
                model_.forward(ctx_, entries.data(), entries.size());
                // Rows a resume computed again are those below the longest history the cache has held.
                size_t again = 0;
                for (size_t i = 0; i < active.size(); ++i) {
                    Request& r = *active[i];
                    const size_t len = r.seq_.length(), n = std::min(len, r.reached_) - std::min(lengths[i], r.reached_);
                    r.recomputed_ += n;
                    again += n;
                    r.reached_ = std::max(r.reached_, len);
                }
                if (again) {
                    std::lock_guard<std::mutex> lk(m_);
                    recomputed_ += again;
                }
                for (size_t w = 0; w < wanting.size(); ++w) {
                    auto& r = wanting[w];
                    const float* row = ctx_.logits(w);
                    r->logits_.assign(row, row + ctx_.width);
                    step(*r);
                }
            } catch (const std::exception& e) {
                // A failed pass leaves every history as it was; the requests in it end with the error rather than the loop.
                for (size_t i = 0; i < active.size();) finish(active, i, "error", e.what());
                continue;
            }
            for (size_t i = 0; i < active.size();) {
                if (!active[i]->finish_pending_.empty()) finish(active, i, active[i]->finish_pending_);
                else ++i;
            }
        }
        for (auto& r : active) { release(*r); r->end("cancel"); }
        std::lock_guard<std::mutex> lk(m_);
        for (auto* waiting : {&queue_, &paused_}) {
            for (auto& r : *waiting) r->end("cancel");
            waiting->clear();
        }
        while (!donors_.empty()) drop_donor();
        active_count_.store(0);
        paused_count_.store(0);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stopping_ = true;
        }
        cv_.notify_all();
    }

private:
    // Tokens an uncapped request reserves beyond what it holds, at admission and each time it grows.
    static constexpr size_t kGrowTokens = 256;
    // The most generated tokens a pass recomputes for one resume, and the budget each takes, ubatch / kReplayRows, since a generated token takes the decode kernels: on one MI50 with Qwen3-8B Q8_0, 64 of them cost a pass of 294 ms (batched decode 218 tok/s at 16 and at 64 sequences) against 559 ms for a 512-token prompt slice (916 tok/s).
    static constexpr size_t kReplayRows = 64;

    // The tokens of r's history, prompt then generated: what a resume holds its cache to.
    static size_t history_tokens(const Request& r) { return r.prompt_.size() + r.gen_.size(); }
    // Whether r's cache lacks only its last generated token, which a decode entry reads.
    static bool decoding(const Request& r) { return !r.gen_.empty() && r.seq_.length() + 1 == history_tokens(r); }
    // Where history token `at` of r lies, prompt or generated; a stretch never spans the two, since the prompt's own ends with it.
    static const uint32_t* token_ptr(const Request& r, size_t at) {
        return at < r.prompt_.size() ? r.prompt_.data() + at : r.gen_.data() + (at - r.prompt_.size());
    }
    // The stretch holding position `at`.
    static const RowClass& class_at(const std::vector<RowClass>& rows, size_t at) {
        size_t i = 0;
        while (rows[i].end <= at) ++i;
        return rows[i];
    }
    // The stretches of the first `n` positions.
    static std::vector<RowClass> clip(const std::vector<RowClass>& rows, size_t n) {
        std::vector<RowClass> out;
        for (size_t i = 0; i < rows.size() && (out.empty() || out.back().end < n); ++i) {
            out.push_back(rows[i]);
            out.back().end = std::min(out.back().end, n);
        }
        return out;
    }
    // How many of the first `n` positions two records computed alike.
    static size_t alike(const std::vector<RowClass>& a, const std::vector<RowClass>& b, size_t n) {
        size_t at = 0;
        for (size_t i = 0, j = 0; at < n && i < a.size() && j < b.size();) {
            if (a[i].extent != b[j].extent || a[i].fresh != b[j].fresh) break;
            at = std::min({a[i].end, b[j].end, n});
            if (a[i].end <= at) ++i;
            if (b[j].end <= at) ++j;
        }
        return std::min(at, n);
    }

    // Before a pass, every uncapped decoding request whose next token would pass its reservation takes another step, the earliest admitted first, with what make_room gives it.
    // A request make_room cannot give it to sits the pass out with its cache as it is (a stall) and asks again before the next; it is never paused for its own growth.
    void grow(std::vector<std::shared_ptr<Request>>& active) {
        for (size_t i = 0; i < active.size(); ++i) {
            Request& r = *active[i];
            r.stalled_ = false;
            if (!r.params_.until_limit || !decoding(r) || !beyond(blocks_for(r.seq_.length() + 1), r.need_)) continue;
            std::vector<size_t> step = blocks_for(r.seq_.length() + 1 + kGrowTokens);
            for (size_t s = 0; s < step.size(); ++s) step[s] = step[s] > r.need_[s] ? step[s] - r.need_[s] : 0;
            const Taken t = make_room(pools(), reserved_, donor_blocks(), npos, false, holders(active), r.admission_, true, step);
            if (!t.enough) {
                r.stalled_ = true;
                ++r.stalls_;
                std::lock_guard<std::mutex> lk(m_);
                ++stalls_;
                continue;
            }
            take(t, active);
            std::lock_guard<std::mutex> lk(m_);
            add(reserved_, step);
            add(r.need_, step);
        }
    }

    // Admits r, a queued request or a paused one, if make_room finds it room without pausing anyone: it reserves its history and max_tokens, or an uncapped one its history and a growth step, and takes back its own donor, forks the donor best_donor found or starts a fresh sequence.
    // The donor it forks goes last, or first when the request shares every full block of it, a follow-up turn or its own history, since that donor then holds nothing the request does not keep but a partial last block; a donor it takes is consumed, the blocks they share counted once.
    // Its own donor, taken back whole, goes first in the same way, and leaves the donors whether make_room needed its blocks or not.
    // Under the lock.
    bool enter(const std::shared_ptr<Request>& r, std::vector<std::shared_ptr<Request>>& active) {
        const size_t tokens = history_tokens(*r) + (r->params_.until_limit ? kGrowTokens : (size_t)r->params_.max_tokens - r->gen_.size());
        std::vector<size_t> need = blocks_for(tokens);
        size_t shared = 0;
        size_t d = own_donor(*r);
        const bool take = d < donors_.size();
        if (!take) d = best_donor(*r, shared);
        const bool keep = take || shared;
        const bool keep_first = take || (shared && donors_[d].tokens.size() - shared < model_.kv_block_tokens());
        const Taken t = make_room(pools(), reserved_, donor_blocks(), keep ? d : npos, keep_first, {}, 0, false, need);
        if (!t.enough) return false;
        std::vector<size_t> gone = t.donors;
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        bool consume = false;
        for (size_t i : gone) {
            if (keep && i == d) { consume = true; continue; }
            drop_donor(i);
            if (i < d) --d;
        }
        admit(*r, d, shared, take);
        if (consume && !take) drop_donor(d);
        add(reserved_, need);
        r->need_ = std::move(need);
        if (!r->admission_) r->admission_ = ++admissions_;
        active.insert(std::upper_bound(active.begin(), active.end(), r, by_admission), r);
        return true;
    }

    // Takes what make_room chose for a growing request: the donors, then the requests it pauses, latest admitted first, and the donor a paused request's history became where make_room took that too.
    void take(const Taken& t, std::vector<std::shared_ptr<Request>>& active) {
        {
            std::vector<size_t> gone = t.donors;
            std::sort(gone.begin(), gone.end(), std::greater<size_t>());
            std::lock_guard<std::mutex> lk(m_);
            for (size_t i : gone) drop_donor(i);
        }
        std::vector<std::pair<std::shared_ptr<Request>, bool>> victims;
        for (const auto& p : t.paused) victims.push_back({active[p.first], p.second});
        for (const auto& v : victims) {
            const size_t at = (size_t)(std::find(active.begin(), active.end(), v.first) - active.begin());
            if (pause(active, at) && v.second) {
                std::lock_guard<std::mutex> lk(m_);
                drop_donor(donors_.size() - 1);
            }
        }
    }

    // A request's history, its prompt then what it generated, which a donor keeps as tokens.
    static std::vector<uint32_t> history(const Request& r) {
        std::vector<uint32_t> h = r.prompt_;
        h.insert(h.end(), r.gen_.begin(), r.gen_.end());
        return h;
    }

    // A paused request's history and its record go to the donors and the request, unchanged, to the paused requests, which are kept in order of first admission and hold nothing else.
    // The request remembers that donor, which it takes back whole when it resumes unless something evicted it meanwhile; its history goes there even short of a full block, which only it can take.
    // Whether it left a donor.
    bool pause(std::vector<std::shared_ptr<Request>>& active, size_t i) {
        auto r = active[i];
        r->donor_ = park(active, i, history(*r), 1);
        ++r->pauses_;
        r->stalled_ = false;
        paused_.insert(std::upper_bound(paused_.begin(), paused_.end(), r, by_admission), r);
        paused_count_.store(paused_.size());
        std::lock_guard<std::mutex> lk(m_);
        ++pauses_;
        return r->donor_ != 0;
    }

    static bool by_admission(const std::shared_ptr<Request>& a, const std::shared_ptr<Request>& b) { return a->admission_ < b->admission_; }
    static constexpr size_t npos = std::numeric_limits<size_t>::max();
    // The ledger as make_room reads it: each pool's blocks, each donor's, and each running request's.
    std::vector<size_t> pools() const {
        std::vector<size_t> p(model_.kv_pools());
        for (size_t s = 0; s < p.size(); ++s) p[s] = model_.kv_pool_blocks(s);
        return p;
    }
    std::vector<std::vector<size_t>> donor_blocks() const {
        std::vector<std::vector<size_t>> b;
        for (const Donor& d : donors_) b.push_back(d.blocks);
        return b;
    }
    std::vector<Holder> holders(const std::vector<std::shared_ptr<Request>>& active) const {
        std::vector<Holder> h;
        for (const auto& r : active) {
            const size_t len = r->seq_.length();
            h.push_back(Holder{r->admission_, r->params_.until_limit, r->need_, len ? blocks_for(len) : std::vector<size_t>(model_.kv_pools(), 0)});
        }
        return h;
    }

    // A finished or paused request kept for its cache: the tokens its history holds, how they were computed, the sequence holding them and the blocks it has reserved.
    // Shared full blocks are immutable, so a fork of it is safe while it lives; its partial last block, which no fork shares, goes on only in the request that paused, taking it back whole.
    struct Donor {
        uint64_t id = 0;   // which paused request may take it back (Request::donor_)
        std::vector<uint32_t> tokens;
        std::vector<RowClass> classes;
        infer::Sequence seq;
        std::vector<size_t> blocks;   // per cache pool
    };

    // A resumed request's own donor, the one its pause left, when nothing has evicted it; donors_.size() when there is none.
    size_t own_donor(const Request& r) const {
        for (size_t d = 0; r.donor_ && d < donors_.size(); ++d)
            if (donors_[d].id == r.donor_) return d;
        return donors_.size();
    }

    // The donor sharing the longest run of full blocks with r's history, and the token count of that run; zero when no donor shares a block.
    // Tokens are compared, not hashed.
    // A request admitted before, which has a record of its rows, shares only rows its donor computed the way its own were: its own donor's, or an identical request's.
    // Only whole blocks are shared since a fork appends only into fresh blocks, and the last history token is always computed in a pass so the request has logits to sample from.
    size_t best_donor(const Request& r, size_t& tokens) const {
        const size_t bt = model_.kv_block_tokens(), h = history_tokens(r);
        size_t best = donors_.size();
        tokens = 0;
        for (size_t d = 0; d < donors_.size(); ++d) {
            const auto& t = donors_[d].tokens;
            size_t n = 0;
            const size_t limit = std::min(t.size(), h - 1);
            while (n < limit && t[n] == *token_ptr(r, n)) ++n;
            if (!r.classes_.empty()) n = alike(r.classes_, donors_[d].classes, n);
            n = n / bt * bt;
            if (n > tokens) { tokens = n; best = d; }
        }
        return best;
    }

    // A history for an admitted request: its own donor `d` taken back whole (`take`), which leaves the donors, or a fork of donor `d` holding the `shared` tokens best_donor found, or a fresh sequence when it found none.
    // The donor taken back holds exactly the rows the request computed before its pause, so it continues as if never paused.
    // A first admission records its rows: a forked prefix as its donor recorded it, the rest of the prompt at the prompt's extent with the tokens it prefills, then the generated tokens at extent 1.
    // Under the lock.
    void admit(Request& r, size_t d, size_t shared, bool take = false) {
        {
            std::lock_guard<std::mutex> lk(r.m_);
            if (r.admitted_ == Request::Clock::time_point{}) r.admitted_ = Request::Clock::now();
        }
        const bool first = r.classes_.empty();
        r.donor_ = 0;
        if (take) {
            r.seq_ = std::move(donors_[d].seq);
            sub(reserved_, donors_[d].blocks);
            donors_.erase(donors_.begin() + (std::ptrdiff_t)d);
            ++r.taken_back_;
            ++taken_back_;
        } else {
            r.seq_ = shared ? model_.fork(donors_[d].seq, shared) : model_.make_sequence();
        }
        if (!first) return;
        if (shared) {
            r.reused_.store(shared);
            ++prefix_hits_;
            prefix_tokens_ += shared;
            r.classes_ = clip(donors_[d].classes, shared);
        }
        const size_t p = r.prompt_.size();
        r.classes_.push_back(RowClass{p, p, p - shared});
        r.classes_.push_back(RowClass{std::numeric_limits<size_t>::max(), 1, 1});
        r.reached_ = shared;
        r.rng_.seed(r.params_.seed);
    }

    // A donor's blocks back to the pool, the oldest donor's unless another is named. Under the lock.
    void drop_donor(size_t i = 0) {
        Donor& d = donors_[i];
        try { model_.reset(d.seq); } catch (const std::exception&) {}
        sub(reserved_, d.blocks);
        donors_.erase(donors_.begin() + (std::ptrdiff_t)i);
    }

    // One sampled token for a request whose logits are in: pushed to its channel unless it ends the request; a request that ends is finished after the pass, once every entry's logits have been read.
    void step(Request& r) {
        const uint32_t id = infer::sample(r.logits_, r.params_, tok_.eos_id, r.gen_, r.rng_);
        if (tok_.is_eos(id)) { r.finish_pending_ = "eos"; return; }
        r.gen_.push_back(id);
        r.last_id_ = id;
        Request::Token t;
        t.id = id;
        if (!r.params_.logprobs) {
            r.push(std::move(t));
        } else if (r.rows_waiting() < Request::kRowsWaiting) {
            // The row the id was sampled from goes with it, and a row the reader has finished with comes back for the next pass to fill.
            t.row = std::move(r.logits_);
            r.logits_.clear();
            r.push(std::move(t), &r.logits_);
        } else {
            // The reader has fallen behind: the values go in the row's place, the same values the reader would compute, and the row stays for the next pass.
            Request::fill(t, r.logits_, r.params_.top_logprobs);
            r.push(std::move(t));
        }
        if (!r.params_.stop.empty()) {
            r.decoded_ += tok_.decode({id});
            for (const auto& s : r.params_.stop)
                if (!s.empty() && r.decoded_.find(s) != std::string::npos) { r.finish_pending_ = "stop"; return; }
        }
        if ((int)r.gen_.size() >= r.params_.max_tokens) r.finish_pending_ = "length";
    }

    // A finished request becomes a donor through park, and one line on stderr records it.
    void finish(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::string& why,
                const std::string& err = "") {
        auto r = active[i];
        if (why == "error") {
            active.erase(active.begin() + (std::ptrdiff_t)i);
            release(*r);
            r->seq_ = infer::Sequence{};
            active_count_.store(active.size());
        } else {
            park(active, i, history(*r));
        }
        r->end(why, err);
        const Request::Timings t = r->timings();
        char paused[128] = "", stalled[48] = "";
        if (r->pauses_)
            std::snprintf(paused, sizeof paused, ", paused %zu time%s, %zu taken back, %zu tokens recomputed", r->pauses_, r->pauses_ == 1 ? "" : "s",
                          r->taken_back_, r->recomputed_);
        if (r->stalls_) std::snprintf(stalled, sizeof stalled, ", sat out %zu pass%s", r->stalls_, r->stalls_ == 1 ? "" : "es");
        std::fprintf(stderr, "request: %zu prompt tokens (%zu reused), %zu generated, %.0f ms queued, %.0f ms to first token, %.1f tok/s, %s%s%s\n",
                     r->prompt_tokens(), r->reused(), r->gen_.size(), t.queued_ms, t.prompt_ms,
                     t.predicted_per_second(r->gen_.size()), why.c_str(), paused, stalled);
    }

    // A request leaves the active set, its history kept as a donor when it holds at least `least` tokens (a full block unless a paused request's own asks for less) and its blocks returned otherwise; the donor's id, 0 when none is kept.
    // A donor keeps only the blocks it holds reserved, and there are at most max_seqs donors, the oldest going when a newcomer needs the room.
    uint64_t park(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::vector<uint32_t>& h, size_t least = 0) {
        auto r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        const size_t held = r->seq_.length();
        uint64_t id = 0;
        if (held && held >= (least ? least : model_.kv_block_tokens())) {
            std::lock_guard<std::mutex> lk(m_);
            while (donors_.size() >= max_seqs_) drop_donor();
            Donor d;
            d.id = id = ++donor_ids_;
            d.tokens.assign(h.begin(), h.begin() + (std::ptrdiff_t)std::min(h.size(), held));
            d.classes = clip(r->classes_, held);
            d.seq = std::move(r->seq_);
            d.blocks = blocks_for(held);
            sub(reserved_, r->need_);
            add(reserved_, d.blocks);
            r->need_.clear();
            donors_.push_back(std::move(d));
        } else {
            release(*r);
        }
        r->seq_ = infer::Sequence{};
        active_count_.store(active.size());
        return id;
    }
    // reset waits for the last pass that touched the sequence, so its blocks return to the pool only once the device is done with them.
    void release(Request& r) {
        try { model_.reset(r.seq_); } catch (const std::exception&) {}
        sub(reserved_, r.need_);
        r.need_.clear();
    }

    // What `tokens` positions take in each cache pool, in that pool's own blocks, never more than the pool holds.
    std::vector<size_t> blocks_for(size_t tokens) const {
        std::vector<size_t> b(model_.kv_pools());
        for (size_t s = 0; s < b.size(); ++s)
            b[s] = std::min(backend::blocks_for(tokens, model_.kv_pool_block_tokens(s)), model_.kv_pool_blocks(s));
        return b;
    }
    // Whether every pool can take `more` beside what is reserved, once the `freed` blocks of a reservation that ends are returned.
    bool room_for(const std::vector<size_t>& more, const std::vector<size_t>& freed = {}) const {
        for (size_t s = 0; s < more.size(); ++s)
            if (reserved_[s] - (s < freed.size() ? freed[s] : 0) + more[s] > model_.kv_pool_blocks(s)) return false;
        return true;
    }
    // Whether `want` needs more than `held` in any pool; an empty `held` holds nothing.
    static bool beyond(const std::vector<size_t>& want, const std::vector<size_t>& held) {
        for (size_t s = 0; s < want.size(); ++s)
            if (want[s] > (s < held.size() ? held[s] : 0)) return true;
        return false;
    }
    static void add(std::vector<size_t>& to, const std::vector<size_t>& b) {
        if (to.size() < b.size()) to.resize(b.size(), 0);
        for (size_t s = 0; s < b.size(); ++s) to[s] += b[s];
    }
    static void sub(std::vector<size_t>& from, const std::vector<size_t>& b) {
        for (size_t s = 0; s < b.size(); ++s) from[s] -= b[s];
    }

    infer::Model& model_;
    const bpe::Tokenizer& tok_;
    size_t max_seqs_, ubatch_, max_queue_;
    infer::ExecContext ctx_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Request>> queue_;
    // Paused requests in order of first admission, the scheduler thread's; they hold nothing but a donor, and do not count against max_queue.
    std::deque<std::shared_ptr<Request>> paused_;
    std::deque<Donor> donors_;
    std::atomic<size_t> active_count_{0}, paused_count_{0};
    size_t prefix_hits_ = 0, prefix_tokens_ = 0;   // under the lock
    uint64_t admissions_ = 0;   // the scheduler thread's
    uint64_t pauses_ = 0;       // under the lock
    uint64_t donor_ids_ = 0;    // under the lock
    size_t stalls_ = 0;         // under the lock
    size_t recomputed_ = 0;     // under the lock
    size_t taken_back_ = 0;     // under the lock
    std::vector<size_t> reserved_;   // per cache pool, blocks promised to admitted requests and held by donors
    bool stopping_ = false;
};

} // namespace server
