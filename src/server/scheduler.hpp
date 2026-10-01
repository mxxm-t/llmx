#pragma once
// The scheduler of docs/SERVER.md: one thread drives the model in rounds over its pass API, each pass carrying a share of ready decoding requests and slices of what other requests' caches lack, and samples each request's logits into its channel.
// Room in the KV pool goes by first admission (make_room, with the rest of the policy core in server/policy.hpp), and a request records how each stretch of its history was computed (RowClass), so a paused request resumes to the logits it gives when never paused.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/cpus.hpp"
#include "inference/logprobs.hpp"
#include "inference/sampler.hpp"
#include "model/runtime.hpp"
#include "server/policy.hpp"
#include "server/sampling_pool.hpp"
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

// A stretch of a history computed one way: the rows before `end`, from the stretch before it on, took this extent (infer::BatchEntry), whose class (Model::row_class) is what chooses a device's kernels and whether a streamed layer runs on the device.
// A request's prompt takes its whole length, a generated token extent 1, and a prefix forked at the first admission keeps the stretches its donor recorded.
struct RowClass {
    size_t end, extent;
};

// One request from submission to completion: the connection thread reads its channel, and everything below the channel belongs to the scheduler thread.
class Request {
public:
    using Clock = std::chrono::steady_clock;

    // `stable` is how much of the prompt a follow-up turn would begin with, the whole prompt for a text (docs/SPECULATIVE.md, section 2).
    Request(std::vector<uint32_t> prompt, SampleParams params, size_t stable = std::numeric_limits<size_t>::max())
        : prompt_(std::move(prompt)), params_(std::move(params)), stable_(std::min(stable, prompt_.size())), submitted_(Clock::now()) {}

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
    // What a wait on the channel until `until` found: a sampled token, the end of the request with `finish` set ("eos", "stop", "length", "cancel" or "error"), or neither.
    // A token's row becomes its log-probabilities here, in the reader's thread, so no pass waits for a walk over the vocabulary, and goes back for a later pass to fill.
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
            fill(t, t.row.data(), t.row.size(), params_.top_logprobs);
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
    // Set by the connection thread when the client goes away; at its next round the scheduler drops the request from the queue or from the batch, once no pass in flight holds it.
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
    // A token's log-probability and the `top` most likely tokens at its position, from the row of n logits it was sampled from.
    static void fill(Token& t, const float* row, size_t n, size_t top) {
        const double lse = infer::log_sum_exp(row, n);
        t.logprob = infer::logprob(row, lse, t.id);
        t.top = infer::top_logprobs(row, n, lse, top);
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

    std::vector<uint32_t> prompt_;   // never changed but for a job's, which grows as the reply it follows is written
    SampleParams params_;   // never changed once made, since next reads it in the connection thread
    const size_t stable_;
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
    std::vector<float> logits_;    // with logprobs asked, the copy of the row its next token goes to the channel with
    uint64_t admission_ = 0;       // order of first admission, by which room goes; set once
    uint64_t donor_ = 0;           // the donor its last pause left, which it takes back whole on resuming unless something evicted it
    uint64_t landed_ = 0;          // the formation order of the last pass it left flight from
    bool stalled_ = false;         // it could not grow before this pass and sits it out
    size_t stalls_ = 0;            // passes it sat out
    size_t pauses_ = 0;            // times it was paused
    size_t taken_back_ = 0;        // resumes that took its donor back
    size_t reached_ = 0;           // the longest history its cache has held, past which nothing is recomputed
    size_t recomputed_ = 0;        // rows its resumes computed again
    size_t keep_at_ = 0;           // on a model that keeps a state, where the slice that reaches it keeps the state as a checkpoint; 0 once kept or skipped
    bool finished_ = false;        // it has left the active set for good
    // A job (docs/SPECULATIVE.md, section 2, Idle re-prefill): an internal request whose prompt, whole blocks, is what the conversation's next turn begins with after request `of_`'s reply, read as prompt rows of one class and kept as a donor that replaces the ones it supersedes; `whole_` once those ids follow the whole reply rather than the part written so far.
    bool job_ = false, whole_ = false;
    bool writing_ = false;         // a job begun while its reply was written, which keeps taking a busy pass's leftover budget once the reply has ended
    std::shared_ptr<Request> of_;
    uint64_t source_ = 0;          // the donor a job forked
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
    // The model's context is reserved here for `passes` passes in flight, each sized for up to one row per decoding request plus a ubatch of prompt or replay rows, each request wanting a logits row at most.
    // No `passes` takes the stage count on a pipelined layer split and one elsewhere, which cannot keep more; passes that do not fit the devices' memory run fewer, and stderr says so.
    // A `timed` scheduler times its rounds and reads each stage's device time (Timing), over backends made to time their work.
    // Up to kSamplers threads beside the scheduler thread sample a pass's rows, fewer where the process may use fewer CPUs.
    // A model whose layers keep a recurrent state must hold a state slot for each of the `max_seqs` requests it runs at once, so admission never waits on one.
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t max_queue, size_t passes = 0, bool timed = false)
        : model_(model), tok_(tok), max_seqs_(max_seqs), ubatch_(model.prefill_batch()), max_queue_(max_queue), timed_(timed),
          samplers_(std::min<size_t>(kSamplers, (size_t)core::automatic_threads() - 1)), reserved_(model.kv_pools(), 0) {
        if (model_.keeps_state() && max_seqs_ > model_.state_slots())
            throw std::logic_error("server: " + std::to_string(max_seqs_) + " requests at once need as many recurrent state slots, and the model holds " +
                                   std::to_string(model_.state_slots()));
        for (size_t s = 0; s < model_.kv_pools(); ++s) {
            pools_.blocks.push_back(model_.kv_pool_blocks(s));
            pools_.block_tokens.push_back(model_.kv_pool_block_tokens(s));
        }
        size_t p = passes ? passes : model_.pipelined() ? model_.stage_count() : 1;
        if (p > 1 && !model_.pipelined())
            throw std::runtime_error("server: " + std::to_string(p) + " passes in flight need a layer split over several devices, each running its layers whole; this placement runs one at a time");
        for (;; --p) {
            logit_rows_ = logit_rows(p, max_seqs_);
            try {
                model_.reserve_passes(ctx_, p, ubatch_ + max_seqs_, logit_rows_.size);
                break;
            } catch (const std::logic_error&) {
                throw;
            } catch (const std::exception& e) {
                if (p == 1) throw;
                std::fprintf(stderr, "server: %zu passes in flight do not fit (%s); running %zu\n", p, e.what(), p - 1);
            }
        }
        slots_.resize(p);
        // A job reads the next turn's ids at one extent for every extent that turn's prompt can have, so only where rows are one class from there up to the limit.
        const size_t limit = token_limit();
        steady_from_ = limit;
        while (steady_from_ > 2 && model_.row_class(steady_from_ - 1) == model_.row_class(limit)) --steady_from_;
    }

    // Tokens one request may hold, prompt and reply together: the model context or the KV pool, whichever is smaller.
    size_t token_limit() const {
        return std::min((size_t)model_.context_length(), model_.kv_tokens_total());
    }

    // Queue a request; the handle's channel delivers its tokens.
    // A prompt the limit cannot hold is refused here, before it waits, and so is a request arriving at a full queue; an uncapped request's max_tokens is the room its prompt leaves.
    std::shared_ptr<Request> submit(std::vector<uint32_t> prompt, SampleParams params, size_t stable = std::numeric_limits<size_t>::max()) {
        if (prompt.empty()) throw std::runtime_error("server: empty prompt");
        if (params.until_limit) {
            if (prompt.size() >= token_limit())
                throw TooLong("the prompt fills the " + std::to_string(token_limit()) + " tokens a request may hold");
            params.max_tokens = (int)(token_limit() - prompt.size());
        }
        if (params.max_tokens <= 0) throw std::runtime_error("server: max_tokens must be positive");
        if (prompt.size() + (size_t)params.max_tokens > token_limit())
            throw TooLong("prompt plus max_tokens exceeds the " + std::to_string(token_limit()) + " tokens a request may hold");
        auto r = std::make_shared<Request>(std::move(prompt), std::move(params), stable);
        {
            std::lock_guard<std::mutex> lk(m_);
            if (queue_.size() >= max_queue_)
                throw QueueFull("server: the queue holds " + std::to_string(max_queue_) + " requests; try again later");
            queue_.push_back(r);
        }
        cv_.notify_all();
        return r;
    }

    // Whether follow reads anything: rows of one class from some extent up to the limit, and on a model that keeps a state, checkpoint slots to keep it in.
    bool follows() const { return steady_from_ < token_limit() && (!model_.keeps_state() || model_.checkpoint_slots()); }

    // The ids the conversation's next turn begins with after request r's reply (docs/SPECULATIVE.md, section 2, Idle re-prefill): while the reply is written, as far as they are known, and once it has ended `whole`, or none where there is no next turn to prepare.
    // A job reads them on a fork of r's history as prompt rows of the class every longer prompt takes, to their last whole block, and keeps them as a donor, so a follow-up turn forks past the reply; ids shorter than where that class begins are not read.
    void follow(const std::shared_ptr<Request>& r, std::vector<uint32_t> ids, bool whole) {
        const size_t bt = model_.kv_block_tokens();
        ids.resize(std::min(ids.size(), token_limit() - 1) / bt * bt);
        if (!follows() || ids.size() < steady_from_) ids.clear();
        if (ids.empty() && !whole) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            follows_.push_back(Follow{r, std::move(ids), whole});
        }
        cv_.notify_all();
    }

    // What a timed scheduler measured (docs/SERVER.md, health), totals in milliseconds: its rounds, the thread's time in them by what it did and where it was held, and each stage's device time over the span its readings cover, with the rows the passes retired in that span carried.
    // A stage's device time comes from timestamps on a device and from the thread's own time on the host, whose stages compute as they are recorded.
    struct Timing {
        uint64_t rounds = 0;
        double round_ms = 0, recording_ms = 0, relaying_ms = 0, sampling_ms = 0, assembly_ms = 0;
        double receive_wait_ms = 0, staging_wait_ms = 0, open_wait_ms = 0, logits_wait_ms = 0;
        std::vector<double> stage_ms;
        double span_ms = 0;
        size_t rows = 0;
    };

    struct Stats {
        size_t active = 0, queued = 0, donors = 0, prefix_hits = 0, prefix_tokens = 0, pauses = 0;
        size_t paused = 0;       // requests paused now, waiting to resume
        size_t stalls = 0;       // passes a request sat out, unable to grow
        size_t waits = 0;        // of those, the ones whose room waited on a request in flight
        size_t recomputed = 0;   // rows resumes computed again
        size_t taken_back = 0;   // resumes that took their own donor back whole
        size_t checkpoints = 0;  // states kept as checkpoints now, on a model whose layers keep one
        size_t passes = 0, in_flight = 0;   // the passes the context keeps in flight at most, and those in flight now
        size_t samplers = 0;                // the threads that sample beside the scheduler thread
        std::vector<size_t> reserved, donor_blocks;   // per cache pool, the blocks the ledger holds reserved and those the donors hold
        bool timed = false;
        Timing timing;   // a timed scheduler's, as of its last round
        size_t reprefills = 0;         // jobs that kept a reply's next turn as a donor
        size_t reprefill_rows = 0;     // rows jobs read
        size_t reprefill_cancels = 0;  // jobs that gave way to a request at a pass boundary
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        Stats s{active_count_.load(), queue_.size(), donors_.size(), prefix_hits_, prefix_tokens_, (size_t)pauses_,
                paused_count_.load(), stalls_, waits_, recomputed_, taken_back_, checkpoints_.load(), slots_.size(), in_flight_.load(), samplers_.threads(), reserved_,
                std::vector<size_t>(reserved_.size(), 0), timed_, timing_};
        for (const Donor& d : donors_) add(s.donor_blocks, d.blocks);
        s.reprefills = reprefills_;
        s.reprefill_rows = reprefill_rows_;
        s.reprefill_cancels = reprefill_cancels_;
        return s;
    }

    // The loop, in the caller's thread, until stop(): rounds over the model's pass API with up to slots_.size() passes in flight (docs/SERVER.md, the round).
    // The only thread that calls the model.
    void run() {
        std::vector<std::shared_ptr<Request>> active;   // in order of first admission
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] {
                    return stopping_ || flying() || !queue_.empty() || !paused_.empty() || !follows_.empty() || (!jobs_.empty() && active.size() < max_seqs_) ||
                           std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return working(*r); });
                });
                if (stopping_) break;
                // A waiting request whose client left ends wherever it waits, queued or paused, not only once admission reaches it, which may be after every active request has finished.
                for (auto* waiting : {&queue_, &paused_})
                    for (auto it = waiting->begin(); it != waiting->end();) it = (*it)->cancel_.load() ? leave(*waiting, it) : it + 1;
            }
            const Clock::time_point round_start = timed_ ? Clock::now() : Clock::time_point{};
            // Stages on the host compute as they are recorded, so they wait until every device stage of the round is on its way.
            std::vector<std::pair<size_t, size_t>> host;
            const Steps steps = round_steps(flights(), model_.stage_count());
            for (const auto& a : steps.advance) {
                if (model_.stage_on_host(a.second)) host.push_back(a);
                else advance(active, a.first, a.second);
            }
            for (const size_t k : steps.retire) retire(active, k);
            for (size_t i = 0; i < active.size();) {
                if (!active[i]->finish_pending_.empty()) finish(active, i, active[i]->finish_pending_);
                else ++i;
            }
            complete_jobs(active);
            // Cancelled requests leave before the next pass, and a request in flight once its pass has retired.
            for (size_t i = 0; i < active.size();) {
                if (active[i]->cancel_.load() && !active[i]->seq_.in_flight()) finish(active, i, "cancel");
                else ++i;
            }
            // Room is made and passes formed only in a free slot; a request in flight is never paused, parked or given room.
            if (free_slot() < slots_.size()) {
                const Clock::time_point room_start = timed_ ? Clock::now() : Clock::time_point{};
                formed_stages_ms_ = 0;
                // A job gives way at this pass boundary to a waiting request that needs its seat or room, and after growth to one that could not grow; it runs again once nothing waits.
                bool waits;
                {
                    std::lock_guard<std::mutex> lk(m_);
                    waits = (!paused_.empty() && !fits_free(*paused_.front(), active.size())) || (!queue_.empty() && !fits_free(*queue_.front(), active.size()));
                }
                if (waits) yield_jobs(active);
                // Growth steps that fall due take their room before anything is admitted, so a request admitted now never holds what an older request's step needs in this pass.
                grow(active);
                if (std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; })) yield_jobs(active);
                {
                    std::lock_guard<std::mutex> lk(m_);
                    // Room goes by first admission: nothing resumes or is admitted while a request that could not grow sits out the pass, paused requests resume oldest first and stop at the first that does not fit, and new requests come in queue order only once none is paused.
                    // A client can leave after the sweep, while the requests before its own are admitted; its request ends here, before any donor gives blocks up for it or it forks one.
                    const bool stalled = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; });
                    while (!stalled && !paused_.empty() && active.size() < max_seqs_) {
                        if (paused_.front()->cancel_.load()) { leave(paused_, paused_.begin()); continue; }
                        if (!enter(paused_.front(), active)) break;
                        paused_.pop_front();
                    }
                    while (!stalled && paused_.empty() && !queue_.empty() && active.size() < max_seqs_) {
                        if (queue_.front()->cancel_.load()) { leave(queue_, queue_.begin()); continue; }
                        if (!enter(queue_.front(), active)) break;
                        queue_.pop_front();
                    }
                    active_count_.store(requests(active));
                    paused_count_.store(paused_.size());
                }
                follow_up(active);
                complete_jobs(active);
                try {
                    for (size_t k = free_slot(); k < slots_.size() && form(k, active, host); k = free_slot()) {}
                } catch (const std::exception& e) {
                    fail_all(active, e.what());
                    host.clear();
                }
                checkpoints_.store(model_.checkpoint_slots() - model_.checkpoints_free());
                if (timed_) round_.assembly_ms += ms_since(room_start) - formed_stages_ms_;
            }
            for (const auto& a : host)
                if (slots_[a.first].live && slots_[a.first].ran == a.second) advance(active, a.first, a.second);
            in_flight_.store(flights_live());
            if (timed_) end_round(round_start);
        }
        abort_all();
        for (auto& r : active) { release(*r); r->end("cancel"); }
        std::lock_guard<std::mutex> lk(m_);
        for (auto* waiting : {&queue_, &paused_}) {
            for (auto& r : *waiting) r->end("cancel");
            waiting->clear();
        }
        jobs_.clear();
        follows_.clear();
        while (!donors_.empty()) drop_donor();
        active_count_.store(0);
        paused_count_.store(0);
        in_flight_.store(0);
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stopping_ = true;
        }
        cv_.notify_all();
    }

    // A pass as it retires, for a test that replays the passes in their order through Model::forward: each entry's request, the history it found, its rows, its extent and whether it wanted logits, and a copy of each wanting row in entry order.
    struct Retired {
        std::vector<const Request*> requests;
        std::vector<size_t> from, rows, extent;
        std::vector<char> want;
        std::vector<std::vector<float>> logits;
    };
    // Called on the scheduler thread with each pass that retires, before its rows are sampled; set before run.
    std::function<void(const Retired&)> on_retire;

private:
    using Clock = std::chrono::steady_clock;
    // How requests reserve room, at admission and as uncapped ones grow.
    static constexpr Growth kGrowth{};
    // The most generated tokens a pass recomputes for one resume, each taking ubatch / kReplayRows of the budget since it takes the decode kernels; docs/STATUS.md (Exact resume) records the timing that sets it.
    static constexpr size_t kReplayRows = 64;
    // The most sampling threads beside the scheduler thread; docs/STATUS.md (layer split phase 3, step 4) records why four.
    static constexpr size_t kSamplers = 4;
    // The most rows of a job a pass carries beside requests' rows, while the reply it follows is written; docs/STATUS.md (step 2c) records the latency it costs.
    static constexpr size_t kJobChunk = 64;

    // One wanting row of a retiring pass as the sampling pool draws it: the request, its mapped logits row, read in place, and the token drawn.
    // With logprobs asked the row is copied for the channel, or, once the reader has fallen behind (Request::kRowsWaiting), the token takes its values instead.
    struct Draw {
        Request* r;
        const float* row;
        bool keep_row;
        Request::Token t;
    };

    // A slot of the context reserved for passes: the round's view of it (Flight), its pass's requests by entry with the history each had and the rows each adds, those it samples in logits order, its logits rows and its decode entries.
    struct Slot : Flight {
        std::vector<std::shared_ptr<Request>> members, wanting;
        std::vector<size_t> from, rows;
        size_t base = 0, want = 0, decoders = 0;
    };

    bool flying() const {
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live; });
    }
    size_t flights_live() const {
        return (size_t)std::count_if(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live; });
    }
    size_t free_slot() const {
        return (size_t)(std::find_if(slots_.begin(), slots_.end(), [](const Slot& k) { return !k.live; }) - slots_.begin());
    }
    std::vector<Flight> flights() const { return std::vector<Flight>(slots_.begin(), slots_.end()); }

    // A new pass in slot k: decode entries first, up to an even share of the decoding requests over the pass slots, but for a request that could not grow or is in flight, then what the other requests' caches lack, one stretch's slice each, up to ubatch tokens; then its logits rows, begin_pass and its first stage, or, on the host, that stage's place in `host`.
    // False when no request has rows to add, which with nothing in flight breaks the round's rules, since the oldest request sits a pass out only while a capped request holds room: that throws, rather than leaving the loop to spin.
    bool form(size_t k, std::vector<std::shared_ptr<Request>>& active, std::vector<std::pair<size_t, size_t>>& host) {
        Slot& f = slots_[k];
        entries_.clear();
        f.members.clear();
        f.wanting.clear();
        f.from.clear();
        f.rows.clear();
        f.decoders = 0;
        const auto add_entry = [&](const std::shared_ptr<Request>& r, const infer::BatchEntry& e) {
            entries_.push_back(e);
            f.members.push_back(r);
            f.from.push_back(r->seq_.length());
            f.rows.push_back(e.n);
            if (e.want_logits) f.wanting.push_back(r);
        };
        const auto ready = [](const Request& r) { return !r.seq_.in_flight() && !r.stalled_; };
        // The share goes to the decoders that left flight earliest, so one held back takes the next pass; they join in order of first admission.
        std::vector<const Request*> waiting;
        size_t decoders = 0;
        for (const Slot& s : slots_) decoders += s.live ? s.decoders : 0;
        for (const auto& r : active)
            if (ready(*r) && decoding(*r)) waiting.push_back(r.get());
        decoders += waiting.size();
        const size_t share = std::min(decode_share(decoders, slots_.size(), model_.stage_count()), waiting.size());
        std::stable_sort(waiting.begin(), waiting.end(), [](const Request* a, const Request* b) { return a->landed_ < b->landed_; });
        waiting.resize(share);
        for (auto& r : active)
            if (std::find(waiting.begin(), waiting.end(), r.get()) != waiting.end()) {
                add_entry(r, infer::BatchEntry{&r->seq_, &r->last_id_, 1, true});
                ++f.decoders;
            }
        size_t budget = ubatch_, keeps = 0;
        // What r's cache lacks, one stretch's slice of at most `most` rows within the budget.
        const auto slice = [&](const std::shared_ptr<Request>& r, size_t most) {
            const size_t at = r->seq_.length(), end = history_tokens(*r);
            if (at >= end) return;
            const RowClass& c = class_at(r->classes_, at);
            size_t n = std::min({c.end, end, at + most}) - at;
            // A job's slice ends on a whole block where it can, so what it has read can be forked there.
            const size_t bt = model_.kv_block_tokens();
            if (r->job_ && (at + n) / bt * bt > at) n = (at + n) / bt * bt - at;
            // A slice that would pass the request's checkpoint ends there, so its state can be kept there.
            if (r->keep_at_ > at && r->keep_at_ < at + n) n = r->keep_at_ - at;
            if (c.extent == 1 && at < r->reached_) {
                // Rows of extent 1 a resume computes again, generated tokens or a forked reply's: a pass takes at most kReplayRows of them, each costing ubatch / kReplayRows of the budget, which is at most the whole budget, so a request gets one while the budget is untouched.
                // A one-token prompt read for the first time costs its row as any prompt does.
                const size_t cost = std::max<size_t>(1, ubatch_ / kReplayRows);
                n = std::min({n, kReplayRows, budget / cost});
                if (!n) return;
                budget -= std::min(budget, n * cost);
            } else {
                n = std::min(n, budget);
                budget -= n;
            }
            // The entry that ends a request's history wants the logits the next token is sampled from, a job's none, and every entry carries its stretch's extent, so its rows take the kernels and the streamed path that first computed them.
            infer::BatchEntry e{&r->seq_, token_ptr(*r, at), n, at + n == end && !r->job_};
            e.extent = c.extent;
            if (r->keep_at_ && r->keep_at_ == at + n) {
                e.keep = checkpoint_room(keeps, r->source_);
                keeps += e.keep;
                r->keep_at_ = 0;
            }
            add_entry(r, e);
        };
        for (auto& r : active)
            if (!r->job_ && !decoding(*r) && !r->seq_.in_flight() && budget) slice(r, budget);
        // Then a job: in a pass no request has rows for, or, once it has read some of a reply while that reply was written, in the budget a pass leaves, at most kJobChunk rows of it.
        const bool idle = entries_.empty();
        for (auto& r : active)
            if (r->job_ && !r->seq_.in_flight() && budget && (idle || r->writing_)) slice(r, idle ? budget : kJobChunk);
        if (entries_.empty()) {
            if (!flying() && std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_; }))
                throw std::logic_error("server: a round with active requests formed an empty pass");
            return false;
        }
        f.want = f.wanting.size();
        f.base = take_rows(logit_rows_, f.want);
        if (f.base == logit_rows_.size) throw std::logic_error("server: no logits rows for a new pass");
        try {
            model_.begin_pass(ctx_, k, entries_.data(), entries_.size(), f.base);
        } catch (...) {
            give_rows(logit_rows_, f.base, f.want);
            throw;
        }
        f.live = true;
        f.formed = ++formed_;
        f.ran = 0;
        if (model_.stage_on_host(0)) host.push_back({k, 0});
        else advance(active, k, 0);
        return true;
    }

    // Stage s of the pass in slot k; a failure has abandoned the pass in the model, and it fails that pass alone.
    void advance(std::vector<std::shared_ptr<Request>>& active, size_t k, size_t s) {
        const Clock::time_point start = timed_ ? Clock::now() : Clock::time_point{};
        const backend::Backend::HostTimes before = timed_ ? host_times() : backend::Backend::HostTimes{};
        try {
            model_.run_pass_stage(ctx_, k, s);
        } catch (const std::exception& e) {
            fail_pass(active, k, e.what());
            return;
        }
        ++slots_[k].ran;
        if (!timed_) return;
        // The stage's time on the thread, apart from where it was held and its uploads; on the host that is the stage's own time.
        const backend::Backend::HostTimes after = host_times();
        const double took = ms_since(start), ticket = after.ticket_ms - before.ticket_ms, slot = after.slot_ms - before.slot_ms,
                     staging = after.staging_ms - before.staging_ms, write = after.write_ms - before.write_ms, own = took - ticket - slot - staging;
        round_.receive_wait_ms += ticket;
        round_.open_wait_ms += slot;
        round_.staging_wait_ms += staging;
        round_.relaying_ms += write;
        round_.recording_ms += own - write;
        if (model_.stage_on_host(s)) host_stage_ms_[s] += own;
        if (s == 0) formed_stages_ms_ += took;
    }

    // The host times of every stage's backend, summed.
    backend::Backend::HostTimes host_times() {
        backend::Backend::HostTimes t;
        for (size_t s = 0; s < model_.stage_count(); ++s) {
            const backend::Backend::HostTimes b = model_.stage_backend(s).host_times();
            t.ticket_ms += b.ticket_ms;
            t.slot_ms += b.slot_ms;
            t.staging_ms += b.staging_ms;
            t.write_ms += b.write_ms;
        }
        return t;
    }
    static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

    // A timed round's end: its time, and every kReadRounds rounds each stage's device time since the last reading, which on a device waits for its queue, so the span a reading covers starts after it.
    void end_round(Clock::time_point start) {
        static constexpr uint64_t kReadRounds = 32;
        round_.round_ms += ms_since(start);
        ++round_.rounds;
        const size_t S = model_.stage_count();
        if (round_.stage_ms.size() != S) {
            round_.stage_ms.assign(S, 0.0);
            host_stage_ms_.assign(S, 0.0);
            for (size_t s = 0; s < S; ++s) model_.stage_backend(s).device_ms();
            span_start_ = Clock::now();
            span_rows_ = 0;
        } else if (round_.rounds % kReadRounds == 0) {
            round_.span_ms += ms_since(span_start_);
            for (size_t s = 0; s < S; ++s) {
                const double ms = model_.stage_backend(s).device_ms();
                round_.stage_ms[s] += model_.stage_on_host(s) ? host_stage_ms_[s] : std::max(ms, 0.0);
                host_stage_ms_[s] = 0;
            }
            round_.rows += span_rows_;
            span_rows_ = 0;
            span_start_ = Clock::now();
        }
        std::lock_guard<std::mutex> lk(m_);
        timing_ = round_;
    }

    // The pass in slot k after its last stage: each wanting row drawn with its request's own state on the sampling pool, the tokens pushed in entry order, then the pass ended and the slot given back.
    // A request cancelled in flight is not sampled: its history stays what the pass computed, which its donor keeps.
    void retire(std::vector<std::shared_ptr<Request>>& active, size_t k) {
        Slot& f = slots_[k];
        try {
            if (on_retire) {
                Retired t;
                const std::vector<infer::BatchEntry>& entries = ctx_.passes[k].entries;
                for (size_t e = 0; e < f.members.size(); ++e) {
                    t.requests.push_back(f.members[e].get());
                    t.from.push_back(f.from[e]);
                    t.rows.push_back(f.rows[e]);
                    t.extent.push_back(entries[e].extent ? entries[e].extent : entries[e].n);
                    t.want.push_back(entries[e].want_logits);
                }
                for (size_t w = 0; w < f.wanting.size(); ++w) {
                    const float* row = model_.pass_logits(ctx_, k, w);
                    t.logits.emplace_back(row, row + ctx_.width);
                }
                on_retire(t);
            }
            // The scheduler thread waits for the logits and reads each request's state; the pool draws the rows, and every read of the pass's logits ends before end_pass gives them back.
            const Clock::time_point start = timed_ ? Clock::now() : Clock::time_point{};
            double waited = 0;
            draws_.clear();
            for (size_t w = 0; w < f.wanting.size(); ++w) {
                Request& r = *f.wanting[w];
                if (r.cancel_.load()) continue;
                const Clock::time_point read = timed_ ? Clock::now() : Clock::time_point{};
                const float* row = model_.pass_logits(ctx_, k, w);
                if (timed_) waited += ms_since(read);
                draws_.push_back(Draw{&r, row, r.params_.logprobs && r.rows_waiting() < Request::kRowsWaiting, {}});
            }
            samplers_.run(draws_.size(), [this](size_t i) { draw(draws_[i]); });
            for (Draw& d : draws_) step(d);
            if (timed_) {
                round_.logits_wait_ms += waited;
                round_.sampling_ms += ms_since(start) - waited;
                for (size_t n : f.rows) span_rows_ += n;
            }
        } catch (const std::exception& e) {
            model_.abort_pass(ctx_, k);
            fail_pass(active, k, e.what());
            return;
        }
        model_.end_pass(ctx_, k);
        // Rows a resume computed again are those below the longest history the cache has held.
        size_t again = 0, job_rows = 0;
        for (size_t e = 0; e < f.members.size(); ++e) {
            Request& r = *f.members[e];
            const size_t to = f.from[e] + f.rows[e], n = std::min(to, r.reached_) - std::min(f.from[e], r.reached_);
            r.recomputed_ += n;
            again += n;
            r.reached_ = std::max(r.reached_, to);
            r.landed_ = f.formed;
            if (r.job_) job_rows += f.rows[e];
        }
        if (again || job_rows) {
            std::lock_guard<std::mutex> lk(m_);
            recomputed_ += again;
            reprefill_rows_ += job_rows;
        }
        vacate(k);
    }

    // Slot k is free again, and the logits rows its pass held come back.
    void vacate(size_t k) {
        Slot& f = slots_[k];
        give_rows(logit_rows_, f.base, f.want);
        f.live = false;
        f.members.clear();
        f.wanting.clear();
        f.decoders = 0;
    }

    // A failed pass, which the model has abandoned, returned its requests' histories to where it found them: they end with the error and give their blocks back, while the other passes in flight go on, their rows in their own storages, handoff buffers and logits rows.
    void fail_pass(std::vector<std::shared_ptr<Request>>& active, size_t k, const std::string& what) {
        const std::vector<std::shared_ptr<Request>> members = slots_[k].members;
        vacate(k);
        for (const auto& m : members) {
            const auto it = std::find(active.begin(), active.end(), m);
            if (it != active.end()) finish(active, (size_t)(it - active.begin()), "error", what);
        }
    }

    // A round that the scheduler's own rules cannot go on with abandons every pass in flight, and every active request ends with the error rather than the loop.
    void fail_all(std::vector<std::shared_ptr<Request>>& active, const std::string& what) {
        abort_all();
        for (size_t i = 0; i < active.size();) finish(active, i, "error", what);
    }

    // Every pass in flight abandoned, each request's history back where its pass found it.
    void abort_all() {
        for (size_t k = 0; k < slots_.size(); ++k)
            if (slots_[k].live) {
                model_.abort_pass(ctx_, k);
                vacate(k);
            }
    }

    // The tokens of r's history, prompt then generated: what a resume holds its cache to.
    static size_t history_tokens(const Request& r) { return r.prompt_.size() + r.gen_.size(); }
    // Whether r's cache lacks only its last generated token, which a decode entry reads.
    static bool decoding(const Request& r) { return !r.gen_.empty() && r.seq_.length() + 1 == history_tokens(r); }
    // Whether r has work for a pass: a request always, a job while its cache lacks some of its ids.
    static bool working(const Request& r) { return !r.job_ || r.seq_.length() < r.prompt_.size(); }
    // The active requests a client waits on, jobs left out.
    static size_t requests(const std::vector<std::shared_ptr<Request>>& active) {
        return (size_t)std::count_if(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_; });
    }
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
    // How many of the first `n` positions two records computed alike, their extents of one class.
    size_t alike(const std::vector<RowClass>& a, const std::vector<RowClass>& b, size_t n) const {
        size_t at = 0;
        for (size_t i = 0, j = 0; at < n && i < a.size() && j < b.size();) {
            if (a[i].extent != b[j].extent && model_.row_class(a[i].extent) != model_.row_class(b[j].extent)) break;
            at = std::min({a[i].end, b[j].end, n});
            if (a[i].end <= at) ++i;
            if (b[j].end <= at) ++j;
        }
        return std::min(at, n);
    }

    // Before a pass, every uncapped decoding request not in flight whose next token would pass its reservation takes another step, the earliest admitted first, with what make_room gives it.
    // A request make_room cannot give it to, or whose plan would pause a request in flight, sits the pass out with its cache as it is (a stall) and asks again before the next; it is never paused for its own growth.
    void grow(std::vector<std::shared_ptr<Request>>& active) {
        for (size_t i = 0; i < active.size(); ++i) {
            Request& r = *active[i];
            r.stalled_ = false;
            if (r.seq_.in_flight()) continue;
            const std::vector<size_t> step = kGrowth.step(pools_, r.params_.until_limit, decoding(r), r.seq_.length(), r.need_);
            if (step.empty()) continue;
            const Taken t = make_room(pools_.blocks, reserved_, donor_blocks(), npos, false, holders(active), r.admission_, true, step);
            if (!t.enough || t.wait) {
                r.stalled_ = true;
                ++r.stalls_;
                std::lock_guard<std::mutex> lk(m_);
                ++stalls_;
                waits_ += t.wait;
                continue;
            }
            take(t, active);
            std::lock_guard<std::mutex> lk(m_);
            add(reserved_, step);
            add(r.need_, step);
        }
    }

    // Admits a queued or paused request if make_room finds room without pausing anyone, reserving its history and max_tokens, or uncapped a growth step: its own donor taken back whole, a fork of the donor best_donor found, or a fresh sequence.
    // A donor it takes back or shares every full block of goes first, one it forks otherwise last, and a forked donor make_room takes is consumed, the shared blocks counted once.
    // Under the lock.
    bool enter(const std::shared_ptr<Request>& r, std::vector<std::shared_ptr<Request>>& active) {
        std::vector<size_t> need = pools_.blocks_for(kGrowth.entry(history_tokens(*r), r->params_.until_limit, (size_t)r->params_.max_tokens, r->gen_.size()));
        size_t shared = 0;
        size_t d = own_donor(*r);
        const bool take = d < donors_.size();
        if (!take) d = best_donor(*r, shared);
        // A running job, as far as it has read, is a source too, and a job's own request while it runs, where no pass holds them.
        const Request* from = nullptr;
        for (const auto& a : active) {
            const bool source = a->job_ ? a != r : r->job_ && a == r->of_;
            if (take || !source || a->seq_.in_flight() || a->classes_.empty()) continue;
            const size_t n = shareable(*r, a->job_ ? a->prompt_ : history(*a), a->classes_, a->seq_);
            if (n > shared) {
                shared = n;
                d = donors_.size();
                from = a.get();
            }
        }
        const bool keep = !from && (take || shared);
        const bool keep_first = take || (keep && donors_[d].tokens.size() - shared < model_.kv_block_tokens());
        const Taken t = make_room(pools_.blocks, reserved_, donor_blocks(), keep ? d : npos, keep_first, {}, 0, false, need);
        if (!t.enough) return false;
        std::vector<size_t> gone = t.donors;
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        bool consume = false;
        for (size_t i : gone) {
            if (keep && i == d) { consume = true; continue; }
            drop_donor(i);
            if (i < d) --d;
        }
        admit(*r, d, shared, take, from);
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

    // A paused request's history goes to a donor, even short of a full block, which it takes back whole on resuming unless something evicted it, and the request to the paused requests in order of first admission; whether it left a donor.
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
    // The ledger as make_room reads it beside the pools: each donor's blocks, and each running request's.
    std::vector<std::vector<size_t>> donor_blocks() const {
        std::vector<std::vector<size_t>> b;
        for (const Donor& d : donors_) b.push_back(d.blocks);
        return b;
    }
    // A request in flight is counted by the history its pass found, since its storages disagree until the pass ends.
    std::vector<Holder> holders(const std::vector<std::shared_ptr<Request>>& active) const {
        std::vector<Holder> h;
        for (const auto& r : active) {
            size_t len = r->seq_.length();
            for (const Slot& s : slots_)
                for (size_t e = 0; s.live && e < s.members.size(); ++e)
                    if (s.members[e] == r) len = s.from[e];
            h.push_back(Holder{r->admission_, r->params_.until_limit, r->need_, len ? pools_.blocks_for(len) : std::vector<size_t>(pools_.blocks.size(), 0), r->seq_.in_flight()});
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

    // A waiting request whose client left ends where it waits, and a paused one's own donor goes with it when it holds less than a block, which no fork can share; the position after it.
    // Under the lock.
    std::deque<std::shared_ptr<Request>>::iterator leave(std::deque<std::shared_ptr<Request>>& waiting, std::deque<std::shared_ptr<Request>>::iterator it) {
        const size_t d = own_donor(**it);
        if (d < donors_.size() && donors_[d].tokens.size() < model_.kv_block_tokens()) drop_donor(d);
        (*it)->end("cancel");
        return waiting.erase(it);
    }

    // A free checkpoint slot for a keep beside `keeps` others the pass takes, the oldest donors that hold one giving theirs where none is free (make_room, with checkpoint slots as one more pool), but for the donor `spare` a job forked, whose slot its fork may still read; a checkpoint is never forced, so false leaves it out.
    bool checkpoint_room(size_t keeps, uint64_t spare = 0) {
        if (model_.checkpoints_free() > keeps) return true;
        const size_t slots = model_.checkpoint_slots();
        std::vector<std::vector<size_t>> held;
        std::vector<size_t> at;
        for (size_t d = 0; d < donors_.size(); ++d)
            if (!spare || donors_[d].id != spare) {
                held.push_back({model_.checkpoint(donors_[d].seq) ? size_t(1) : size_t(0)});
                at.push_back(d);
            }
        const Taken t = make_room({slots}, {slots - model_.checkpoints_free() + keeps}, held, npos, false, {}, 0, false, {1});
        if (!t.enough) return false;
        std::vector<size_t> gone;
        for (size_t i : t.donors) gone.push_back(at[i]);
        std::sort(gone.begin(), gone.end(), std::greater<size_t>());
        std::lock_guard<std::mutex> lk(m_);
        for (size_t i : gone) drop_donor(i);
        // A slot a fork still reads stays held after its donor goes.
        return model_.checkpoints_free() > keeps;
    }

    // The donor sharing the longest run of whole blocks with r's history by tokens, over rows computed as r's were or, at its first admission, as it would compute them: its prompt at the prompt's extent; the run's length goes to `tokens`, zero when none shares a block.
    // On a model that keeps a state the run reaches only as far as the donor's checkpoint, where a fork can read the state.
    // The last history token is never shared, since a pass must compute it to give logits.
    size_t best_donor(const Request& r, size_t& tokens) const {
        size_t best = donors_.size();
        tokens = 0;
        for (size_t d = 0; d < donors_.size(); ++d) {
            const size_t n = shareable(r, donors_[d].tokens, donors_[d].classes, donors_[d].seq);
            if (n > tokens) { tokens = n; best = d; }
        }
        return best;
    }
    // How much of r's history a fork of `seq`, holding the history `t` computed as `classes` record, can give it: whole blocks of the same tokens over rows computed as r's were or, at its first admission, as it would compute them (a job's at the class every longer prompt takes), within what `seq` holds; on a model that keeps a state, only as far as its checkpoint.
    size_t shareable(const Request& r, const std::vector<uint32_t>& t, const std::vector<RowClass>& classes, const infer::Sequence& seq) const {
        const size_t bt = model_.kv_block_tokens(), h = history_tokens(r);
        const std::vector<RowClass> first{RowClass{r.prompt_.size(), r.prompt_.size()}};
        const std::vector<RowClass>& own = r.classes_.empty() ? first : r.classes_;
        size_t n = 0;
        const size_t limit = std::min({t.size(), h - 1, seq.length()});
        while (n < limit && t[n] == *token_ptr(r, n)) ++n;
        n = alike(own, classes, n);
        n = n / bt * bt;
        if (model_.keeps_state()) {
            const std::optional<size_t> kept = model_.checkpoint(seq);
            n = kept && *kept <= n && *kept % bt == 0 ? *kept : 0;
        }
        return n;
    }

    // A history for an admitted request: its own donor `d` taken back whole (`take`), a fork at `shared` tokens of donor `d` or of the running request `from`, or a fresh sequence.
    // A first admission records its rows: a forked prefix as its source recorded it, the rest of the prompt at the prompt's extent, then the generated tokens at extent 1; a job's rows past the fork all at its prompt's extent, of the class every longer prompt takes.
    // Under the lock.
    void admit(Request& r, size_t d, size_t shared, bool take = false, const Request* from = nullptr) {
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
            r.seq_ = shared ? model_.fork(from ? from->seq_ : donors_[d].seq, shared) : model_.make_sequence();
        }
        if (!first) return;
        if (shared) {
            r.reused_.store(shared);
            r.classes_ = clip(from ? from->classes_ : donors_[d].classes, shared);
        }
        const size_t p = r.prompt_.size();
        if (r.job_) {
            r.source_ = shared && !from ? donors_[d].id : 0;
            r.classes_.push_back(RowClass{std::numeric_limits<size_t>::max(), p});
            r.reached_ = shared;
            return;
        }
        if (shared) {
            ++prefix_hits_;
            prefix_tokens_ += shared;
        }
        r.classes_.push_back(RowClass{p, p});
        r.classes_.push_back(RowClass{std::numeric_limits<size_t>::max(), 1});
        r.reached_ = shared;
        r.rng_.seed(r.params_.seed);
        // One checkpoint where a follow-up turn would fork it: whole blocks within the stable prefix, never the last prompt token, past what the request forked.
        const size_t bt = model_.kv_block_tokens(), at = std::min(r.stable_, p - 1) / bt * bt;
        if (model_.checkpoint_slots() && at > shared) r.keep_at_ = at;
    }

    // A donor's blocks back to the pool, the oldest donor's unless another is named.
    // Under the lock.
    void drop_donor(size_t i = 0) {
        Donor& d = donors_[i];
        try { model_.reset(d.seq); } catch (const std::exception&) {}
        sub(reserved_, d.blocks);
        donors_.erase(donors_.begin() + (std::ptrdiff_t)i);
    }

    // On a sampling thread: the token drawn from d's row with its request's own settings, history and generator, and with logprobs asked the row copied or its values computed.
    // It touches only d and its request's generator and row copy, which no other draw of the pass shares, since a request wants one row a pass.
    void draw(Draw& d) {
        Request& r = *d.r;
        d.t.id = infer::sample(d.row, ctx_.width, r.params_, tok_.eos_id, r.gen_, r.rng_);
        if (tok_.is_eos(d.t.id) || !r.params_.logprobs) return;
        if (d.keep_row) r.logits_.assign(d.row, d.row + ctx_.width);
        else Request::fill(d.t, d.row, ctx_.width, r.params_.top_logprobs);
    }

    // A drawn token on the scheduler thread: pushed to its channel unless it ends the request; a request that ends is finished after the pass, once every entry's logits have been read.
    void step(Draw& d) {
        Request& r = *d.r;
        const uint32_t id = d.t.id;
        if (tok_.is_eos(id)) { r.finish_pending_ = "eos"; return; }
        r.gen_.push_back(id);
        r.last_id_ = id;
        if (d.keep_row) {
            // The row the id was sampled from goes with it, and a row the reader has finished with comes back for the next pass to fill.
            d.t.row = std::move(r.logits_);
            r.logits_.clear();
            r.push(std::move(d.t), &r.logits_);
        } else {
            // Without logprobs the token is its id; a reader that has fallen behind gets the values the draw computed, the same the reader would.
            r.push(std::move(d.t));
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
        r->finished_ = true;
        if (r->job_) {
            // A failed job leaves the donor it forked and frees what it held.
            active.erase(active.begin() + (std::ptrdiff_t)i);
            release(*r);
            r->seq_ = infer::Sequence{};
            return;
        }
        if (why == "error") {
            active.erase(active.begin() + (std::ptrdiff_t)i);
            release(*r);
            r->seq_ = infer::Sequence{};
            active_count_.store(requests(active));
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
    // On a model whose layers keep a recurrent state, which exists only at the end of what it read, a donor ends at its checkpoint: a paused request's whole history kept as one where a checkpoint slot is free, else, as a finished request's, the checkpoint at its prompt; with none it keeps no donor and recomputes from its start.
    uint64_t park(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::vector<uint32_t>& h, size_t least = 0) {
        auto r = active[i];
        active.erase(active.begin() + (std::ptrdiff_t)i);
        size_t held = r->seq_.length();
        if (model_.keeps_state() && held && !(least && model_.keep(r->seq_))) {
            const std::optional<size_t> kept = model_.checkpoint(r->seq_);
            held = kept ? model_.retract(r->seq_, *kept) : 0;
        }
        uint64_t id = 0;
        if (held && held >= (least ? least : model_.kv_block_tokens())) {
            std::lock_guard<std::mutex> lk(m_);
            while (donors_.size() >= max_seqs_) drop_donor();
            Donor d;
            d.id = id = ++donor_ids_;
            d.tokens.assign(h.begin(), h.begin() + (std::ptrdiff_t)std::min(h.size(), held));
            d.classes = clip(r->classes_, held);
            d.seq = std::move(r->seq_);
            d.blocks = pools_.blocks_for(held);
            sub(reserved_, r->need_);
            add(reserved_, d.blocks);
            r->need_.clear();
            donors_.push_back(std::move(d));
        } else {
            release(*r);
        }
        r->seq_ = infer::Sequence{};
        active_count_.store(requests(active));
        return id;
    }
    // The ids a conversation's next turn begins with, given for request `of` (follow).
    struct Follow {
        std::shared_ptr<Request> of;
        std::vector<uint32_t> ids;
        bool whole;
    };

    // The job of request `of`, active or waiting, as an index into `active` or else into jobs_; npos for none.
    std::pair<size_t, size_t> job_of(const std::vector<std::shared_ptr<Request>>& active, const Request* of) const {
        for (size_t i = 0; i < active.size(); ++i)
            if (active[i]->job_ && active[i]->of_.get() == of) return {i, npos};
        for (size_t i = 0; i < jobs_.size(); ++i)
            if (jobs_[i]->of_.get() == of) return {npos, i};
        return {npos, npos};
    }

    // Whether r could be admitted now from a free seat and free blocks alone, as enter reserves for it, no donor evicted.
    // Under the lock.
    bool fits_free(const Request& r, size_t seats) const {
        if (seats >= max_seqs_) return false;
        const std::vector<size_t> need = pools_.blocks_for(kGrowth.entry(history_tokens(r), r.params_.until_limit, (size_t)r.params_.max_tokens, r.gen_.size()));
        for (size_t s = 0; s < need.size(); ++s)
            if (need[s] + reserved_[s] > pools_.blocks[s]) return false;
        return true;
    }

    // Each job not in flight gives way: it frees what it held and waits to start again from its source once nothing else waits.
    void yield_jobs(std::vector<std::shared_ptr<Request>>& active) {
        for (size_t i = 0; i < active.size();) {
            auto j = active[i];
            if (!j->job_ || j->seq_.in_flight()) {
                ++i;
                continue;
            }
            active.erase(active.begin() + (std::ptrdiff_t)i);
            release(*j);
            j->seq_ = infer::Sequence{};
            j->classes_.clear();
            j->source_ = 0;
            std::lock_guard<std::mutex> lk(m_);
            jobs_.push_front(j);
            ++reprefill_cancels_;
        }
    }

    // The ids given since the last round, each to its request's job: a new job, a job's ids grown as the reply is written or made whole, or a job that has read ids the reply's next turn no longer begins with started again; none drops the job.
    // Then, while no request waits and a seat is free, the oldest waiting job is admitted as a request is, through enter, so it forks the history that shares most with it and takes room as a first admission does.
    void follow_up(std::vector<std::shared_ptr<Request>>& active) {
        std::vector<Follow> given;
        {
            std::lock_guard<std::mutex> lk(m_);
            given.swap(follows_);
        }
        std::vector<Follow> later;
        for (Follow& f : given) {
            const std::pair<size_t, size_t> at = job_of(active, f.of.get());
            std::shared_ptr<Request> j = at.first != npos ? active[at.first] : at.second != npos ? jobs_[at.second] : nullptr;
            if (j && j->seq_.in_flight()) {
                later.push_back(std::move(f));
                continue;
            }
            // Ids given once the reply ended stay; any given before them is older.
            if (j && j->whole_ && !f.whole) continue;
            // What the job's cache holds must still begin the ids; otherwise it starts again.
            const size_t read = j ? j->seq_.length() : 0;
            const bool holds = j && !f.ids.empty() && f.ids.size() >= read && std::equal(j->prompt_.begin(), j->prompt_.begin() + (std::ptrdiff_t)read, f.ids.begin());
            // A running job keeps going only where the blocks its longer ids take past its reservation are free, reserved before it reads them; otherwise it gives its room back and waits to start again, as a first admission would.
            bool keeps = holds && at.first != npos;
            std::vector<size_t> more;
            if (keeps) {
                const std::vector<size_t> need = pools_.blocks_for(f.ids.size());
                more.resize(need.size());
                for (size_t p = 0; p < need.size(); ++p) {
                    more[p] = need[p] > j->need_[p] ? need[p] - j->need_[p] : 0;
                    keeps = keeps && reserved_[p] + more[p] <= pools_.blocks[p];
                }
            }
            if (j && at.first != npos && !keeps) {
                active.erase(active.begin() + (std::ptrdiff_t)at.first);
                release(*j);
                j->seq_ = infer::Sequence{};
            }
            std::lock_guard<std::mutex> lk(m_);
            if (j && (at.second != npos || !keeps)) {
                const auto w = std::find(jobs_.begin(), jobs_.end(), j);
                if (w != jobs_.end()) jobs_.erase(w);
            }
            if (f.ids.empty()) continue;
            if (keeps) {
                add(reserved_, more);
                add(j->need_, more);
                j->prompt_ = std::move(f.ids);
                j->whole_ = f.whole;
                continue;
            }
            server::SampleParams params;
            params.max_tokens = 0;
            auto n = std::make_shared<Request>(std::move(f.ids), params);
            n->job_ = true;
            n->whole_ = f.whole;
            n->writing_ = (j && j->writing_) || (!f.whole && !f.of->finished_);
            n->of_ = f.of;
            // A job per conversation, at most max_seqs waiting, the oldest dropped.
            if (jobs_.size() >= max_seqs_) jobs_.pop_front();
            jobs_.push_back(n);
        }
        std::lock_guard<std::mutex> lk(m_);
        follows_.insert(follows_.begin(), std::make_move_iterator(later.begin()), std::make_move_iterator(later.end()));
        const bool stalled = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; });
        while (!stalled && queue_.empty() && paused_.empty() && active.size() < max_seqs_ && !jobs_.empty()) {
            const std::shared_ptr<Request> j = jobs_.front();
            if (!enter(j, active)) {
                // With no request running, nothing will make room it does not find now.
                if (!requests(active)) jobs_.pop_front();
                break;
            }
            jobs_.pop_front();
        }
    }

    // On a model that keeps a state, each job between passes keeps its state where it has read to a whole block (Model::keep, its live slot becoming the checkpoint's, in the slot of the one it replaces or else one checkpoint_room finds), so a follow-up turn that arrives before it completes forks what it has read.
    // Each job that has read the whole of its ids, kept where a model that keeps a state needs it, becomes a donor beside the one it forked, which a regenerated reply still forks at its earlier checkpoint; both go by donor age (make_room) as room is needed.
    void complete_jobs(std::vector<std::shared_ptr<Request>>& active) {
        for (size_t i = 0; i < active.size();) {
            auto j = active[i];
            const size_t len = j->seq_.length();
            if (j->job_ && model_.keeps_state() && !j->seq_.in_flight() && len && len % model_.kv_block_tokens() == 0 &&
                model_.checkpoint(j->seq_) != std::optional<size_t>(len) && !model_.keep(j->seq_) && checkpoint_room(0, j->source_))
                model_.keep(j->seq_);
            if (!j->job_ || !j->whole_ || j->seq_.in_flight() || len < j->prompt_.size()) {
                ++i;
                continue;
            }
            // On a model that keeps a state its live state becomes its checkpoint, in a slot an older donor gives up or, where none can, the donor it forked.
            if (model_.keeps_state() && !model_.keep(j->seq_) && !(checkpoint_room(0, j->source_) && model_.keep(j->seq_))) {
                {
                    std::lock_guard<std::mutex> lk(m_);
                    for (size_t d = 0; d < donors_.size(); ++d)
                        if (j->source_ && donors_[d].id == j->source_) { drop_donor(d); break; }
                }
                if (!model_.keep(j->seq_)) {
                    finish(active, i, "error");
                    continue;
                }
            }
            const uint64_t id = park(active, i, j->prompt_);
            j->finished_ = true;
            std::lock_guard<std::mutex> lk(m_);
            if (id) ++reprefills_;
        }
    }

    // reset waits for the last pass that touched the sequence, so its blocks return to the pool only once the device is done with them.
    void release(Request& r) {
        try { model_.reset(r.seq_); } catch (const std::exception&) {}
        std::lock_guard<std::mutex> lk(m_);
        sub(reserved_, r.need_);
        r.need_.clear();
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
    const bool timed_;
    Timing round_;                        // the scheduler thread's, published to timing_ each round
    Timing timing_;                       // under the lock
    std::vector<double> host_stage_ms_;   // per stage on the host, its time since the last reading
    Clock::time_point span_start_;
    size_t span_rows_ = 0;                // rows the passes retired since the last reading carried
    double formed_stages_ms_ = 0;         // the round's first stages of new passes, which forming them does not count
    infer::ExecContext ctx_;
    std::vector<Slot> slots_;
    LogitRows logit_rows_;
    Pools pools_;                              // the model's cache pools, which never change size
    uint64_t formed_ = 0;                      // passes formed, which orders them
    std::vector<infer::BatchEntry> entries_;   // the pass being formed
    SamplingPool samplers_;
    std::vector<Draw> draws_;                  // the retiring pass's rows the pool draws
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Request>> queue_;
    // Paused requests in order of first admission, the scheduler thread's; they hold nothing but a donor, and do not count against max_queue.
    std::deque<std::shared_ptr<Request>> paused_;
    std::deque<Donor> donors_;
    std::vector<Follow> follows_;                 // under the lock, the ids given since the last round
    std::deque<std::shared_ptr<Request>> jobs_;   // under the lock, jobs waiting for a seat, oldest first
    size_t steady_from_ = 0;                      // the least extent from which rows are one class up to the limit
    size_t reprefills_ = 0, reprefill_rows_ = 0, reprefill_cancels_ = 0;   // under the lock
    std::atomic<size_t> active_count_{0}, paused_count_{0}, in_flight_{0}, checkpoints_{0};
    size_t prefix_hits_ = 0, prefix_tokens_ = 0;   // under the lock
    uint64_t admissions_ = 0;   // the scheduler thread's
    uint64_t pauses_ = 0;       // under the lock
    uint64_t donor_ids_ = 0;    // under the lock
    size_t stalls_ = 0;         // under the lock
    size_t waits_ = 0;          // under the lock
    size_t recomputed_ = 0;     // under the lock
    size_t taken_back_ = 0;     // under the lock
    std::vector<size_t> reserved_;   // per cache pool, blocks promised to admitted requests and held by donors
    bool stopping_ = false;
};

} // namespace server
