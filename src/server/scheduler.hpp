#pragma once
// The scheduler of docs/SERVER.md: one thread drives the model in rounds over its pass API, each pass carrying ready decoding requests and slices of what other caches lack, and samples each request's logits into its channel.
// Room in the KV pool goes by first admission (make_room, policy core in server/policy.hpp), and a request records how each stretch of its history was computed (RowClass), so a paused request resumes to the logits it gives when never paused.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <thread>
#include <tuple>
#include <deque>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/cpus.hpp"
#include "core/host_memory.hpp"
#include "core/job_threads.hpp"
#include "inference/logprobs.hpp"
#include "inference/sampler.hpp"
#include "inference/spec.hpp"
#include "model/runtime.hpp"
#include "model/stage_lanes.hpp"
#include "server/disk_tier.hpp"
#include "server/policy.hpp"
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

// One request from submission to completion: the connection thread reads its channel, and everything below the channel belongs to the scheduler thread.
// An image's encode as the scheduler runs it: `step` runs a piece a call, null until the last of its `pieces` has run, then the image's rows (docs/VISION.md, section 10).
struct ImageSteps {
    std::function<std::shared_ptr<const std::vector<float>>()> step;
    size_t pieces = 0;
};

class Request {
public:
    using Clock = std::chrono::steady_clock;

    // `stable` is how much of the prompt a follow-up turn would begin with, the whole prompt for a text, and `message` where its last user message starts, 0 where that is not known (docs/SPECULATIVE.md, section 2).
    // `received` is when its connection read it, before the prompt was rendered, tokenized and cut.
    Request(std::vector<uint32_t> prompt, SampleParams params, size_t stable = std::numeric_limits<size_t>::max(), size_t message = 0, Clock::time_point received = Clock::now())
        : prompt_(std::move(prompt)), params_(std::move(params)), stable_(std::min(stable, prompt_.size())), message_(std::min(message, prompt_.size())),
          received_(std::min(received, Clock::now())), submitted_(Clock::now()) {}

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
    // The rows of its history a read from disk it waited for covered, where the wait reached its bound and it computed them instead; 0 otherwise.
    size_t restore_late() const { return restore_late_.load(); }
    // Milliseconds from its reading to its submission (the prompt rendered, tokenized and cut), from submission to admission, from admission to the first token, and from the first token to the end; read once the request has ended.
    struct Timings {
        double prepared_ms = 0, queued_ms = 0, prompt_ms = 0, predicted_ms = 0;
        // What the client waits for its first token, from the request's reading.
        double first_token_ms() const { return prepared_ms + queued_ms + prompt_ms; }
        // The tokens predicted_ms covers: those after the first, which came with the prompt's pass.
        static size_t predicted(size_t tokens) { return tokens > 1 ? tokens - 1 : 0; }
        static double per_second(size_t n, double ms) { return ms > 0 ? 1000.0 * (double)n / ms : 0.0; }
        double predicted_per_second(size_t tokens) const { return per_second(predicted(tokens), predicted_ms); }
    };
    // What its restores from host memory or disk cost, read once the request has ended: restores, waits and each part (docs/SERVER.md, What a restore cost).
    // wait_ms, hold_ms and enqueue_ms fall in queued_ms and first_pass_ms in prompt_ms; a part that ends after the first token (a resume's) is after_first_ms; read_ms, slabs_ms and copies_ms overlap them, so they are not addends.
    struct Restore {
        const char* kind = "none";
        size_t restores = 0, tokens = 0, waits = 0, timeouts = 0, failed = 0, copies = 0, first_passes = 0, after_first = 0;
        uint64_t bytes = 0;
        double wait_ms = 0, read_ms = 0, slabs_ms = 0, hold_ms = 0, enqueue_ms = 0, copies_ms = 0, first_pass_ms = 0, after_first_ms = 0;
        // A part of `ms` ending now goes to `span` before the first token and to after_first_ms after it; under the request's lock, which first_ is set under.
        void part(double& span, double ms, Clock::time_point first) {
            if (first == Clock::time_point{}) {
                span += ms;
                return;
            }
            ++after_first;
            after_first_ms += ms;
        }
    };
    Restore restore() const {
        std::lock_guard<std::mutex> lk(m_);
        return restore_;
    }
    Timings timings() const {
        std::lock_guard<std::mutex> lk(m_);
        auto ms = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        Timings t;
        t.prepared_ms = ms(received_, submitted_);
        if (admitted_ != Clock::time_point{}) t.queued_ms = ms(submitted_, admitted_);
        if (first_ != Clock::time_point{}) t.prompt_ms = ms(admitted_, first_), t.predicted_ms = ms(first_, ended_);
        return t;
    }

private:
    friend class Scheduler;
    friend class Cache;
    friend size_t history_tokens(const Request& r);
    friend const uint32_t* token_ptr(const Request& r, size_t at);
    friend std::vector<uint32_t> history(const Request& r);
    friend size_t requests(const std::vector<std::shared_ptr<Request>>& active);
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
    const size_t message_;
    std::vector<uint32_t> cut_;    // a cut prompt's uncut rows through the first one the cut dropped (Api::fit), whose holders admission supersedes
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Token> out_;
    size_t rows_ = 0;                          // tokens in out_ carrying their rows
    std::vector<std::vector<float>> spare_;    // rows next has finished with, for push to hand back
    bool done_ = false;
    std::string finish_, error_;
    std::atomic<bool> cancel_{false};
    std::atomic<size_t> reused_{0};
    std::atomic<size_t> restore_late_{0};
    size_t disk_rows_ = 0;         // the rows of its history the read or proof it waits for covers
    Clock::time_point received_, submitted_, admitted_, first_, ended_;   // admitted_ is set once, under m_
    Restore restore_;   // under m_, written by the scheduler's thread at each part (Scheduler::promote, finish_read, end_wait)

    // Scheduler state.
    uint64_t id_ = 0, prefilled_ = 0;
    // The history is the prompt then the generated tokens, neither ever rewritten; the cache holds its first seq_.length(), which is all the progress there is.
    infer::Sequence seq_;
    std::string finish_pending_;   // set by a sampled end, acted on after the pass
    std::string error_pending_;    // with finish_pending_ "error", what failed
    std::vector<size_t> need_;     // blocks reserved for it, per cache pool
    std::vector<RowClass> classes_;   // how its history was computed, stretch by stretch; empty until the first admission
    // Its prompt's images in prompt order with their digests, any rows given held until a sequence takes them (docs/VISION.md, section 5).
    // `encode_` begins image i's encode on the scheduler's thread (ImageSteps); the sequence holds the images before `media_next_`, and its slices stop at the next one's first row until it has its rows.
    std::vector<infer::MediaRun> media_;
    std::function<ImageSteps(size_t)> encode_;
    size_t media_next_ = 0;
    uint32_t last_id_ = 0;
    // Drafting (docs/SPECULATIVE.md, section 3): its record, the draft its own history gave ahead of the proposer, and the verify its pass in flight carries, its last pick then the drafts, empty for a decode entry.
    infer::spec::Record record_;
    std::vector<uint32_t> own_draft_;
    std::vector<uint32_t> verify_;
    std::vector<uint32_t> gen_;
    std::string decoded_;
    infer::RNG rng_;
    std::vector<float> logits_;    // with logprobs asked, the copy of the row its next token goes to the channel with
    uint64_t admission_ = 0;       // order of first admission, by which room goes; set once
    uint64_t paused_as_ = 0;       // donor_ as its last pause set it, kept through the resume: the id of the copy that pause left in host memory
    uint64_t line_ = 0;            // its place in the line of what waits (admit_waiting): the number it took at its submission, 0 once paused for another's growth, and a new number when it gives the pool's edge up; under the lock
    size_t edge_steps_ = 0;        // growth steps it took, its admission one of them, while an uncapped request was paused, and when it took the last
    Clock::time_point stepped_{};
    uint64_t donor_ = 0;           // the donor its last pause left, which it takes back whole on resuming unless something evicted it
    uint64_t landed_ = 0;          // the formation order of the last pass it left flight from
    bool stalled_ = false;         // it could not grow before this pass and sits it out
    size_t stalls_ = 0;            // passes it sat out
    size_t pauses_ = 0;            // times it was paused
    size_t taken_back_ = 0;        // resumes that took its donor back
    size_t reached_ = 0;           // the longest history its cache has held, past which nothing is recomputed
    size_t recomputed_ = 0;        // rows its resumes computed again
    bool first_pass_ = false;      // a promotion it took waits for its next pass to retire, whose time is Restore::first_pass_ms
    Clock::time_point first_before_{};   // first_ as that promotion found it: unset where the pass gives the first token, so the pass is in prompt_ms
    size_t boundary_at_ = 0;       // on a model that keeps a state, where the slice that reaches it keeps the state as a message boundary; 0 once kept or skipped
    bool boundary_kept_ = false;   // its pass in flight keeps that state, which goes to host memory once the pass retires
    size_t keep_at_ = 0;           // on a model that keeps a state, where the slice that reaches it keeps the state as a checkpoint; 0 once kept or skipped
    size_t read_alone_ = 0;        // prompt rows it read while it was alone, which prompt_slice brings back to a whole ubatch once company comes; 0 after that
    backend::Ticket slice_ticket_ = 0;   // under the interleaved rule, its last slice's submission at the first stage, before whose retirement no next slice is formed
    bool finished_ = false;        // it has left the active set for good
    uint64_t parked_ = 0;          // the donor it left as it finished
    // The disk entry whose read it waits for before admission, which requests behind it that fit may pass (docs/DISK-TIER.md, Restore).
    // Also when the wait began, the most it lasts (the shared tokens recomputed at the measured prompt rate), and whether its match on disk was looked for.
    uint64_t disk_wait_ = 0;
    std::chrono::steady_clock::time_point disk_since_, disk_until_;
    bool disk_checked_ = false;
    bool looked_ = false;   // a job prefetch has looked at since it joined the line, which reading waits for
    // The proof of another build's files on its history's path (Scheduler::prove): its id, where the slice that reaches it ends for the next comparison and whether that one takes the state there.
    // `unproved_` says its cache holds rows restored from those files, which leave with a proof that does not end equal.
    uint64_t proof_ = 0;
    size_t prove_at_ = 0;
    bool prove_state_ = false, unproved_ = false;
    // A job (docs/SPECULATIVE.md, section 2, Idle re-prefill): an internal request whose prompt is what the conversation's next turn begins with after request `of_`'s reply, read as prompt rows of one class and kept as a donor.
    // `whole_` is set once those ids follow the whole reply rather than the part written so far.
    bool job_ = false, whole_ = false;
    bool no_room_ = false;         // a job that has said it waits for spare room (enter), said once
    bool room_wait_ = false;       // waiting for a donor's write, or for room for a restore, before its admission (waits_for_room), since room_since_
    bool room_timed_out_ = false;  // that wait reached its bound, counted once
    std::chrono::steady_clock::time_point room_until_{};   // the wait's bound: the scheduler's constant, or a write's remaining time at its measured rate (write_wait_ms), never lowered
    std::chrono::steady_clock::time_point room_since_{};
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


// The moves of the cache between its places that the scheduler says and counts (Scheduler::moved; docs/SERVER.md, the cache's moves).
enum class Move { parked, consumed, forked, promoted, promote_failed, to_host, not_to_host, host_dropped, host_lost, boundary_kept, boundary_not_kept, boundary_dropped, boundary_lost,
                  job_started, job_done, job_gave_way, job_lost, job_handed, job_not_handed, read_started, read_done, read_refused, read_failed, write_started, write_whole, write_failed, disk_removed, consumed_lost, count };
// A move's name in /v1/health, `reuse.moves.since_start`.
inline const char* move_name(Move m) {
    static const char* const names[] = {"parked", "consumed", "forked", "promoted", "promote_failed", "to_host", "not_to_host", "host_dropped", "host_lost", "boundary_kept", "boundary_not_kept", "boundary_dropped", "boundary_lost",
                                        "job_started", "job_done", "job_gave_way", "job_lost", "job_handed", "job_not_handed", "read_started", "read_done", "read_refused", "read_failed", "write_started", "write_whole", "write_failed", "disk_removed", "consumed_lost"};
    return names[(size_t)m];
}

} // namespace server

#include "server/cache.hpp"

namespace server {

class Scheduler {
public:
    using Moved = Cache::Moved;
    // The model's context is reserved here for `passes` passes in flight (the stage count on a pipelined layer split and one elsewhere when none is given; passes that do not fit run fewer, and stderr says so).
    // With `pacing` on, where a stage is fed in pieces (Model::paces), generating requests are served between a prompt pass's pieces (docs/SERVER.md, A generating request beside a prompt).
    Scheduler(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs, size_t max_queue, size_t passes = 0, bool timed = false,
              size_t host_bytes = 0, infer::spec::Proposer* proposer = nullptr, size_t draft_max = 0, bool priced = true, const DiskOptions& disk = {}, Pacing pacing = {},
              size_t ctx_size = 0)
        : model_(model), tok_(tok), max_seqs_(max_seqs), ubatch_(model.prefill_batch()), max_queue_(max_queue), limit_(request_limit(model, ctx_size)), timed_(timed), host_cap_(host_bytes), share_(pacing.mode == Pacing::Mode::fixed ? pacing.share : 0), auto_(pacing.mode == Pacing::Mode::automatic), interleave_(pacing.mode == Pacing::Mode::interleaved),
          wait_bound_(std::chrono::duration_cast<Clock::duration>(disk.wait ? *disk.wait : std::chrono::milliseconds(kWaitBound))), read_wait_(disk.read_wait), recompute_row_ms_(disk.recompute_row_ms),
          proposer_(draft_max ? proposer : nullptr), draft_max_(proposer ? draft_max : 0), draft_rows_(draft_rows(max_seqs, draft_max_)), priced_(priced), pools_(pools_of(model)),
          samplers_(std::min<size_t>(kSamplers, (size_t)core::automatic_threads() - 1)) {
        if (model_.keeps_state() && max_seqs_ > model_.state_slots())
            throw std::logic_error("server: " + std::to_string(max_seqs_) + " requests at once need as many recurrent state slots, and the model holds " +
                                   std::to_string(model_.state_slots()));
        size_t p = passes ? passes : model_.pipelined() ? model_.stage_count() : 1;
        if (p > 1 && !model_.pipelined())
            throw std::runtime_error("server: " + std::to_string(p) + " passes in flight need a layer split over several devices, each running its layers whole; this placement runs one at a time");
        // A pass carries at most draft_rows draft rows beside a row a decoding request and a ubatch of prompt rows, each draft row wanting its logits.
        const size_t drafts = draft_rows_;
        // A pass apart at each stage, with logits rows of their own, since they retire past the prompt passes.
        if (share_ < 0 || share_ >= 1) throw std::runtime_error("server: the share of generating requests is at least 0 and below 1");
        if (auto_ && !(model_.paces() && model_.stage_count() == 1)) throw std::logic_error("server: automatic pacing needs one stage fed in pieces");
        if (interleave_ && !(model_.pipelined() && model_.stage_count() > 1)) throw std::logic_error("server: interleaved pacing needs a pipelined split");
        const size_t a = (share_ > 0 || auto_) && model_.paces() ? model_.stage_count() : 0;
        apart_rows_ = logit_rows(a, logits_a_pass(max_seqs_, draft_max_));
        for (;; --p) {
            logit_rows_ = logit_rows(p, logits_a_pass(max_seqs_, draft_max_));
            try {
                model_.reserve_passes(ctx_, p + a, ubatch_ + max_seqs_ + drafts, logit_rows_.size + apart_rows_.size, a ? apart_rows(max_seqs_, draft_max_) : 0);
                break;
            } catch (const std::logic_error&) {
                throw;
            } catch (const std::exception& e) {
                if (p == 1) throw;
                std::fprintf(stderr, "server: %zu passes in flight do not fit (%s); running %zu\n", p, e.what(), p - 1);
            }
        }
        normal_ = p;
        slots_.resize(p + a);
        for (size_t k = p; k < slots_.size(); ++k) slots_[k].apart = true;
        // A stage whose recording waits on its devices gets a thread of its own, so the round goes on to the other stages while it records (docs/SERVER.md, the round).
        recorders_.resize(model_.stage_count());
        for (size_t s = 0; s < recorders_.size(); ++s) pace_.push_back(std::make_unique<StagePace>());
        plain_.resize(recorders_.size());
        for (size_t s = 0; s < recorders_.size(); ++s) {
            // Under the automatic rule a pass that is neither fed in pieces nor apart is recorded as without pacing, so only the stages that had a thread then take one.
            plain_[s] = (p > 1 && model_.stage_waits(s)) || (a && !auto_) || (!model_.stage_on_host(s) && model_.stage_reads(s));
            // Under the interleaved rule every device stage has one, which times a prompt's slices there (Recorder::Timing).
            plain_[s] = plain_[s] || (interleave_ && !model_.stage_on_host(s));
            // And where passes go apart every stage has one, the first among them, which is then fed in pieces.
            if (plain_[s] || a) {
                recorders_[s] = std::make_unique<Recorder>(*pace_[s], lanes_.add(s, pace_[s]->m));
                recorders_[s]->share.share = share_;
                // The lane is given its record, not the place it is kept: a scheduler destroyed before the thread first runs is emptying that place.
                Recorder* rec = recorders_[s].get();
                infer::StageLanes::Policy policy;
                policy.ready = [rec] { return rec->pass.posted || rec->apart.posted; };
                policy.step = [this, rec, s](std::unique_lock<std::mutex>& lk) { record(*rec, s, lk); };
                // Slices timed on this stage are settled as they retire, looked at every millisecond while any is out, and the round is woken for the next slice.
                policy.idle = [this, rec, s](std::unique_lock<std::mutex>& lk) -> std::optional<std::chrono::milliseconds> {
                    if (settle_timing(*rec, s)) {
                        lk.unlock();
                        wake();
                        lk.lock();
                    }
                    if (rec->timing.empty()) return std::nullopt;
                    return std::chrono::milliseconds(1);
                };
                lanes_.start(s, std::move(policy));
            }
        }
        // A job reads the next turn's ids at one extent for every extent that turn's prompt can have, so only where rows are one class from there up to the limit.
        const size_t limit = token_limit();
        steady_from_ = limit;
        while (steady_from_ > 2 && model_.row_class(steady_from_ - 1) == model_.row_class(limit)) --steady_from_;
        cache_.open_disk(disk, [this] { cv_.notify_all(); });
    }

    // Tokens one request may hold, prompt and reply together, the one figure admission, the cut, jobs and the host tier read: the smallest of what the caller asks (`asked`, 0 for none), the model's context and the KV pool.
    static size_t request_limit(const infer::Model& model, size_t asked) {
        const size_t limit = std::min((size_t)model.context_length(), model.kv_tokens_total());
        return asked ? std::min(asked, limit) : limit;
    }
    size_t token_limit() const { return limit_; }

    // The part of what a request may hold that a reply without a cap is given room for: an eighth.
    // With cuts of half the limit (kCutTo) a conversation then runs from about three eighths of the limit to seven between two cuts, so a cut reads three eighths again to buy half; a fraction, so it is the same at any context.
    static constexpr size_t kReplyRoom = 8;
    // The rows a prompt may take beside its reply: what a request may hold less the tokens it asks for, or less a reply's room where it names none.
    // The one rule of what fits: a route cuts a prompt to it (Api::fit) and submit refuses a prompt past it.
    size_t prompt_room(const SampleParams& params) const {
        const size_t limit = token_limit(), reply = params.until_limit ? std::max<size_t>(1, limit / kReplyRoom) : (size_t)std::max(params.max_tokens, 0);
        return limit > reply ? limit - reply : 0;
    }
    // Why a prompt of `n` tokens does not fit beside its reply, with its numbers.
    std::string too_long(size_t n, const SampleParams& params) const {
        const size_t limit = token_limit();
        if (!params.until_limit)
            return "the prompt is " + std::to_string(n) + " tokens and asks for " + std::to_string(params.max_tokens) + " more, and a request may hold " + std::to_string(limit);
        return "the prompt is " + std::to_string(n) + " tokens and leaves " + std::to_string(limit > n ? limit - n : 0) + " for its reply, fewer than the " + std::to_string(limit - prompt_room(params)) +
               " a request without max_tokens is given, of the " + std::to_string(limit) + " a request may hold";
    }

    // A prompt's images (docs/VISION.md, section 5): its runs in prompt order with their digests and, where the caller has them, their rows, and what gives an image's rows where a sequence reads them, which the scheduler's thread calls.
    struct Images {
        std::vector<infer::MediaRun> runs;
        std::function<ImageSteps(size_t)> encode;
    };

    // Queue a request; the handle's channel delivers its tokens.
    // A prompt that leaves its reply no room is refused here, before it waits (prompt_room), and so is a request arriving at a full queue; an uncapped request's max_tokens is the room its prompt leaves.
    std::shared_ptr<Request> submit(std::vector<uint32_t> prompt, SampleParams params, size_t stable = std::numeric_limits<size_t>::max(), size_t message = 0,
                                    Request::Clock::time_point received = Request::Clock::now(), std::vector<uint32_t> cut = {}, Images images = {}) {
        if (prompt.empty()) throw std::runtime_error("server: empty prompt");
        if (!images.runs.empty() && (!model_.takes_images() || !images.encode)) throw std::runtime_error("server: images for a model that takes none, or without their encoder");
        for (size_t i = 0; i < images.runs.size(); ++i) {
            const infer::MediaRun& m = images.runs[i];
            if (!m.rows || m.first + m.rows > prompt.size() || (i && m.first < images.runs[i - 1].first + images.runs[i - 1].rows))
                throw std::runtime_error("server: an image outside its prompt or out of order");
        }
        if (!params.until_limit && params.max_tokens <= 0) throw std::runtime_error("server: max_tokens must be positive");
        if (prompt.size() > prompt_room(params)) throw TooLong(too_long(prompt.size(), params));
        if (params.until_limit) params.max_tokens = (int)(token_limit() - prompt.size());
        auto r = std::make_shared<Request>(std::move(prompt), std::move(params), stable, message, received);
        r->cut_ = std::move(cut);
        r->media_ = std::move(images.runs);
        r->encode_ = std::move(images.encode);
        {
            std::lock_guard<std::mutex> lk(m_);
            if (queue_.size() >= max_queue_)
                throw QueueFull("server: the queue holds " + std::to_string(max_queue_) + " requests; try again later");
            r->line_ = ++lines_;
            r->id_ = ++request_ids_;
            queue_.push_back(r);
            ++arrivals_;
        }
        cv_.notify_all();
        return r;
    }

    // Whether follow reads anything: rows of one class from some extent up to the limit, and on a model that keeps a state, checkpoint slots to keep it in.
    bool follows() const { return steady_from_ < token_limit() && (!model_.keeps_state() || model_.checkpoint_slots()); }

    // The ids the conversation's next turn begins with after request r's reply (docs/SPECULATIVE.md, section 2, Idle re-prefill): those known while the reply is written, `whole` once it has ended, none where there is no next turn.
    // A job reads them on a fork of r's history as prompt rows of the class every longer prompt takes, to their last whole block, and keeps them as a donor; ids shorter than where that class begins are not read.
    void follow(const std::shared_ptr<Request>& r, std::vector<uint32_t> ids, bool whole) {
        const size_t bt = model_.kv_block_tokens();
        ids.resize(std::min(ids.size(), token_limit() - 1) / bt * bt);
        if (!follows() || ids.size() < steady_from_) ids.clear();
        if (ids.empty() && !whole) return;
        {
            std::lock_guard<std::mutex> lk(m_);
            cache_.follow(r, std::move(ids), whole);
            ++arrivals_;
        }
        cv_.notify_all();
    }

    // What a timed scheduler measured (docs/SERVER.md, health), totals in milliseconds: its rounds, the thread's time in them by what it did and where it was held, and each stage's device time over the span its readings cover.
    // A stage's device time comes from timestamps on a device and from the thread's own time on the host, whose stages compute as they are recorded.
    struct Timing {
        uint64_t rounds = 0;
        double round_ms = 0, recording_ms = 0, relaying_ms = 0, sampling_ms = 0, assembly_ms = 0;
        double drafting_ms = 0, settling_ms = 0;   // the drafters' chains as completed work, which the thread waits for, and the verifies' retracts as it records them
        double receive_wait_ms = 0, staging_wait_ms = 0, open_wait_ms = 0, logits_wait_ms = 0;
        std::vector<double> stage_ms;
        double span_ms = 0;
        size_t rows = 0;
    };

    struct RequestView {
        uint64_t id = 0, prefilled_tokens = 0, published_tokens = 0, elapsed_ms = 0;
        size_t prompt_tokens = 0, reused_tokens = 0;
        const char* phase = "queued";
        // In phase "encode": its image's place among the request's (from 1) and rows, the pieces run of the encode's and the time since it began, 0 of 0 in 0 ms while another image holds the encoder.
        size_t image_index = 0, image_count = 0, image_rows = 0, pieces_done = 0, pieces_total = 0;
        uint64_t image_elapsed_ms = 0;
    };
    struct LiveRequests {
        static constexpr size_t limit = 64;
        uint64_t sample_elapsed_ms = 0;
        uint64_t prefilled_tokens = 0, published_tokens = 0;
        size_t total = 0, count = 0;
        std::array<RequestView, limit> items{};
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
        std::vector<size_t> reserved, donor_blocks;   // per cache pool, the blocks the ledger holds reserved and those the donors hold, those being streamed in among them
        bool timed = false;
        Timing timing;   // a timed scheduler's, as of its last round
        size_t reprefills = 0;         // jobs that kept a reply's next turn as a donor
        size_t jobs = 0, job_rows_left = 0;   // jobs running or waiting to start now, and the ids they have yet to read
        size_t room_waits = 0, room_wait_timeouts = 0;   // admissions that waited for a donor's write to disk or for room for a restore, and those that reached their bound
        double room_wait_ms = 0, room_wait_longest_ms = 0;
        size_t reprefill_rows = 0;     // rows jobs read
        size_t reprefill_cancels = 0;  // jobs that gave way to a request at a pass boundary
        size_t reprefill_handed = 0;   // jobs whose rows a waiting request took over, the job's history made a donor for it
        size_t host_donors = 0, host_bytes = 0;   // donors held in host memory now, and their bytes
        size_t host_hits = 0;                     // donors promoted from host memory for a request
        size_t host_allocated = 0, host_yielded = 0;   // bytes of host slabs alive, held or idle, and bytes of idle slabs given back to a host short of memory
        size_t host_bytes_moved = 0;              // bytes copied between the devices and host memory, both ways
        size_t boundaries = 0, boundary_hits = 0; // message boundaries' states held in host memory now, and the requests that forked one
        size_t disk_entries = 0, disk_errors = 0; // files on disk now, and writes and reads that failed
        size_t disk_states = 0;                   // those of them that are states, the rest being segments
        size_t disk_hits = 0, disk_waits = 0;     // entries read back for a request, and requests that waited for a read
        size_t disk_wait_timeouts = 0;            // of those waits, the ones that reached their bound, the request computing what the read would have given
        size_t disk_unproved = 0;                 // another build's files on disk now, not yet proved this build's (--disk-cache-adopt proved)
        size_t disk_proved = 0, disk_disproved = 0;   // such files proofs made this build's, and those deleted with a build one of whose entries failed its proof
        size_t disk_refused = 0;                  // reads of a history on disk not started for want of host room or memory, the request reading its prompt
        size_t disk_refused_dirs = 0;             // kept directories of another identity the start did not adopt, kept until their age limit
        std::array<size_t, (size_t)Move::count> moves{};   // the cache's moves since start, by kind (moved)
        size_t reprefill_released = 0;            // rows jobs had read and released as they gave way
        size_t uncopyable = 0;                    // donors holding a block whose checkpoint no copy can take (copy_length 0), counted as they are made
        size_t lost_before_written = 0, disk_capped = 0;   // beside a disk tier, histories and boundaries' states released while no file held them (lost), and entries deleted for the disk's cap
        uint64_t disk_bytes_read = 0;             // their bytes
        double disk_wait_ms = 0;                  // the time requests waited for reads
        // Each read and promotion once, where Request::Restore attributes them to requests: counts, and each part's sum and largest (docs/SERVER.md, What a restore cost).
        using RestoreTotals = server::RestoreTotals;
        RestoreTotals restore{};
        size_t disk_in_flight = 0;                // the write and the reads the disk tier has in flight now
        bool disk_ready = false;                  // the disk tier's store made, once the model file's digest is known
        uint64_t disk_bytes = 0, disk_bytes_written = 0;   // their files' bytes, and every finished write's
        bool disk_writing = false;                // a disk tier that has not stopped writing
        std::vector<size_t> drafted{}, kept{};    // by draft position, the drafts verifies fed and those they kept
        std::vector<size_t> own_drafted{}, own_kept{};   // those of them requests' own histories gave ahead of the proposer (spec::lookup_first)
        size_t draft_failures = 0;                // drafts that threw, each costing its pass the drafts and no request
        size_t yields = 0, kv_tokens = 0, kv_donor_tokens = 0;   // pauses that gave the pool's edge to a paused request (Growth::yields), and the pool's reserved blocks as tokens, each the largest over the pools: the running requests' part, and the donors'
        size_t turns_without_memory = 0;          // turns at the pool's edge kept because the host had no memory for the copy the history would leave (host_room)
        uint64_t finished = 0, prompt_tokens = 0, generated_tokens = 0;   // since start, the clients' requests that were admitted and ended, and their prompt and generated tokens
        uint64_t prefilled_tokens = 0, published_tokens = 0;   // client prompt rows retired and output tokens published, including requests still running
        uint64_t host_limit = 0, disk_limit = 0;  // the host tier's and the disk tier's byte caps, 0 for a tier that is off
        uint64_t rode = 0;                        // and the passes in which generated rows rode a prompt's rows though passes go apart
        uint64_t apart = 0, between = 0;          // since start, the passes of generated rows formed apart, and those of them a stage recorded between a prompt pass's pieces
        uint64_t prompt_passes = 0;               // since start, the passes that carried a request's prompt rows
        uint64_t continued = 0;                   // since start, those formed while the same prompt's pass before them was in flight (Scheduler::continues)
        uint64_t short_passes = 0;                // since start, those that read a short prompt ahead of a long one (short_prompts)
        std::vector<SliceRecord> slices{};        // per stage, its last pass fed in pieces
        double share = 0;                         // the fixed share, 0 for none
        bool automatic = false;                   // generated rows are served by the automatic rule, whose clock the next four read
        bool interleaved = false;                 // or by the interleaved rule on a pipelined split
        uint64_t interleaved_replies = 0, interleaved_slices = 0;   // since start, its passes of generated rows and of a prompt's slice formed while replies and prompts both waited
        uint64_t interleaved_mixed = 0;           // and passes formed then that held both kinds or two prompts, which the rule never forms, for the tests
        uint64_t overlaps = 0;                    // the times replies and prompts began to wait together, for the tests
        bool overlap = false, service_wanted = false, service_granted = false;   // now: replies and prompts both wait, the round wants a service, and a stage stands parked on a grant
        double owed_ms = 0, charged_ms = 0;       // what the generated rows owe the floor now, and their time charged to it since start
        uint64_t grants = 0, services = 0, uncharged_at_grant = 0;   // since start, the services a stage admitted, the admitted passes it recorded, and the grants given before the pass before was charged
        uint64_t gaps = 0, gaps_late = 0;         // since start, the gaps between services of generated rows that ended while replies and prompts both waited, and those above the target
        uint64_t floor_deferred = 0;              // the passes apart that were due and waited for the prompts' floor
        double longest_piece_ms = 0;              // the longest piece of a prompt's pass fed in pieces
        LiveRequests live{};
    };
    Stats stats() const {
        std::lock_guard<std::mutex> lk(m_);
        Stats s{active_count_.load(), queue_.size(), 0, 0, 0, (size_t)pauses_, paused_count_.load(), stalls_, waits_, recomputed_, 0, checkpoints_.load(), normal_, in_flight_.load(), samplers_.threads(), {}, {},
                timed_, timing_};
        cache_.fill(s);
        for (size_t p = 0; p < s.reserved.size(); ++p) {
            const size_t held = p < s.donor_blocks.size() ? s.donor_blocks[p] : 0;
            s.kv_tokens = std::max(s.kv_tokens, (s.reserved[p] - held) * pools_.block_tokens[p]);
            s.kv_donor_tokens = std::max(s.kv_donor_tokens, held * pools_.block_tokens[p]);
        }
        s.finished = finished_.load();
        s.apart = apart_formed_.load();
        s.between = between_.load();
        s.prompt_passes = prompt_passes_.load();
        s.continued = continued_.load();
        s.short_passes = short_passes_.load();
        for (const auto& pace : pace_) {
            std::lock_guard<std::mutex> pl(pace->m);
            s.slices.push_back(pace->slice);
        }
        s.rode = rode_;
        s.share = share_;
        s.automatic = auto_;
        s.interleaved = interleave_;
        s.interleaved_replies = interleaved_replies_;
        s.interleaved_slices = interleaved_slices_;
        s.interleaved_mixed = interleaved_mixed_;
        if (auto_) {
            std::lock_guard<std::mutex> rl(recorders_[0]->m);
            const ServiceClock& c = recorders_[0]->clock;
            s.overlaps = c.overlaps;
            s.overlap = overlap_.load();
            s.service_wanted = recorders_[0]->want;
            s.service_granted = recorders_[0]->granted;
            s.owed_ms = recorders_[0]->floor.owed_ms;
            s.charged_ms = recorders_[0]->charged_ms;
            s.grants = recorders_[0]->grants;
            s.services = recorders_[0]->services;
            s.uncharged_at_grant = recorders_[0]->uncharged_at_grant;
            s.gaps = c.intervals;
            s.gaps_late = c.late;
            s.floor_deferred = c.deferred;
            s.longest_piece_ms = c.longest_piece_ms;
        }
        s.prompt_tokens = prompt_total_.load();
        s.generated_tokens = generated_total_.load();
        s.prefilled_tokens = prefilled_total_.load();
        s.published_tokens = published_total_.load();
        s.draft_failures = draft_failures_.load();
        s.yields = yields_.load();
        s.turns_without_memory = turns_without_memory_.load();
        s.host_limit = host_cap_;
        // Every source's by position, and the requests' own histories' apart.
        tally_.tally(infer::spec::Source::lookup, s.own_drafted, s.own_kept);
        tally_.tally(infer::spec::Source::proposer, s.drafted, s.kept);
        if (s.drafted.size() < s.own_drafted.size()) s.drafted.resize(s.own_drafted.size()), s.kept.resize(s.own_drafted.size());
        for (size_t i = 0; i < s.own_drafted.size(); ++i) s.drafted[i] += s.own_drafted[i], s.kept[i] += s.own_kept[i];
        s.live = live_;
        live_requested_ = true;
        return s;
    }

    // The loop, in the caller's thread, until stop(): rounds over the model's pass API with up to slots_.size() passes in flight (docs/SERVER.md, the round).
    // The only thread that calls the model.
    void run() {
        std::vector<std::shared_ptr<Request>> active;   // in order of first admission
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m_);
                cache_.note_room(active);
                if (live_requested_ || (live_.total && !active_count_.load() && queue_.empty() && paused_.empty())) snapshot_requests(active);
                // A pass being recorded gives the round nothing to do until its recorder is done, which wakes it.
                const auto due = [&] {
                    return stopping_ || ended() || acting() || Cache::ready(queue_) || Cache::ready(paused_) || cache_.due(active.size() < max_seqs_) ||
                           std::any_of(active.begin(), active.end(), [&](const std::shared_ptr<Request>& r) { return working(*r) && (!r->seq_.in_flight() || continues(*r, replying(active))); });
                };
                // Under --disk-cache-keep a server with nothing to do for kDiskIdle writes ahead (disk_round), and the loop wakes for it.
                cache_.idle_moment(!active.empty() || flying() || !queue_.empty() || !paused_.empty());
                const auto bound = cache_.next_disk_bound(lines());
                // With nothing to do the host tier's slabs are taken ahead, a step a round with no sleep between, so a copy made later allocates none (grow_host).
                const bool ahead = !stopping_ && active.empty() && !flying() && queue_.empty() && paused_.empty() && cache_.grow_ahead();
                if (!ahead && bound) cv_.wait_until(lk, *bound, due);
                else if (!ahead) cv_.wait(lk, due);
                if (stopping_) break;
                seen_ = arrivals_;
                cache_.host_pressure();
                cache_.write_turn(write_turn(active));
                cache_.disk_round(active, lines());
                // A segment that took background work's turn begins the count of gaps the next piece waits for, a job's or a write's.
                write_waits_ = cache_.write_held();
                if (cache_.took_write_turn()) {
                    reply_gaps_ = 0;
                    background_last_ = Background::write;
                }
                // A waiting request whose client left ends wherever it waits, queued or paused, not only once admission reaches it, which may be after every active request has finished.
                for (auto* waiting : {&queue_, &paused_})
                    for (auto it = waiting->begin(); it != waiting->end();) it = (*it)->cancel_.load() ? leave(*waiting, it) : it + 1;
            }
            const Clock::time_point round_start = timed_ ? Clock::now() : Clock::time_point{};
            acted_ = false;
            encoded_ = false;
            collect(active);
            cache_.copies_retired();
            // Stages on the host compute as they are recorded, so they wait until every device stage of the round is on its way.
            std::vector<std::pair<size_t, size_t>> host;
            formed_before_ = formed_;
            short_ahead_ = false;
            const Steps steps = round_steps(flights(), model_.stage_count());
            for (const auto& a : steps.advance) {
                // A failure abandons a continued prompt's other slices, which this round's steps may still name.
                if (!slots_[a.first].live || slots_[a.first].ran != a.second) continue;
                if (model_.stage_on_host(a.second)) host.push_back(a);
                else advance(active, a.first, a.second);
            }
            for (const size_t k : steps.retire)
                if (slots_[k].live) retire(active, k);
            for (size_t i = 0; i < active.size();) {
                if (!active[i]->finish_pending_.empty()) finish(active, i, active[i]->finish_pending_, active[i]->error_pending_);
                else ++i;
            }
            cache_.complete_jobs(active, lines());
            // Cancelled requests leave before the next pass, and a request in flight once its pass has retired.
            for (size_t i = 0; i < active.size();) {
                if (active[i]->cancel_.load() && !active[i]->seq_.in_flight()) finish(active, i, "cancel");
                else ++i;
            }
            // Room is made and passes formed only in a free slot; a request in flight is never paused, parked or given room.
            if (free_slot() < slots_.size() || free_apart() < slots_.size()) {
                const Clock::time_point room_start = timed_ ? Clock::now() : Clock::time_point{};
                formed_stages_ms_ = 0;
                formed_chains_ms_ = 0;
                // A job gives way at this pass boundary to a waiting request that needs its seat or room, and after growth to one that could not grow; it runs again once nothing waits.
                std::shared_ptr<Request> waiter;
                {
                    std::lock_guard<std::mutex> lk(m_);
                    if (!paused_.empty() && !cache_.admits(*paused_.front(), active.size(), active)) waiter = paused_.front();
                    else if (!queue_.empty() && !cache_.admits(*queue_.front(), active.size(), active)) waiter = queue_.front();
                }
                if (waiter) {
                    for (const auto& j : active)
                        if (j->job_) recall(*j);
                    cache_.hand_jobs(active, lines());
                    cache_.yield_jobs(active, lines(), waiter.get());
                }
                // Growth steps that fall due take their room before anything is admitted, so a request admitted now never holds what an older request's step needs in this pass.
                grow(active);
                // A request that could not grow makes the jobs give way and takes its step from what they held, in this round: with no other request ready, a round later nothing would be in flight.
                if (std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; }) && cache_.yield_jobs(active, lines())) grow(active, true);
                stalled_now_ = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; });
                jobs_yield_ = waiter || stalled_now_;
                {
                    std::lock_guard<std::mutex> lk(m_);
                    // Room goes by first admission: nothing resumes or is admitted while a request that could not grow sits out the pass, paused requests resume oldest first, and new requests come in queue order once none is paused.
                    // A client can leave after the sweep, while the requests before its own are admitted; its request ends here, before any donor gives blocks up for it or it forks one.
                    const bool stalled = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->stalled_; });
                    // A request whose history is being read from disk keeps its place, and those behind it that fit pass it (docs/DISK-TIER.md, Restore); one submitted since the round began is looked for first.
                    cache_.prefetch(active, lines());
                    using Waiting = std::deque<std::shared_ptr<Request>>;
                    if (!stalled)
                        admit_waiting(paused_, queue_, [&] { return active.size() < max_seqs_; },
                                      [&](Waiting& q, size_t i) {
                                          if (!q[i]->cancel_.load()) return false;
                                          leave(q, q.begin() + (std::ptrdiff_t)i);
                                          return true;
                                      },
                                      [&](const std::shared_ptr<Request>& r) { return cache_.reading(lines(), *r); },
                                      [&](const std::shared_ptr<Request>& r) { return cache_.enter(r, active); }, [](const std::shared_ptr<Request>& r) { return r->line_; });
                    active_count_.store(requests(active));
                    paused_count_.store(paused_.size());
                }
                for (size_t i = 0; i < active.size(); ++i)
                    if (active[i]->proof_ && !active[i]->seq_.in_flight()) cache_.prove(lines(), *active[i]);
                cache_.follow_up(active, lines());
                cache_.complete_jobs(active, lines());
                cache_.count_jobs(active);
                try {
                    // A pass is formed when its first stage can be recorded, so it takes the requests ready then and not those of a stage's time before.
                    // Generated rows ride the passes that read prompts, as without a share, or go in passes apart, which the stages take between a prompt pass's pieces (rides).
                    if (auto_ || interleave_) demand(active);
                    // A fixed share's ride ends after a run of passes to measure its price again (RideRun); the automatic rule rides by the prompt pass's own time unless the prompt is cut (cut_now).
                    // Under the interleaved rule generated rows ride only outside an overlap; in one they go in passes of their own, formed first.
                    const bool ride = interleave_ ? !overlap_.load() : normal_ == slots_.size() || (auto_ ? !overlap_ || (riding(active) && !cut_now(active)) : ride_run_.allow(riding(active)));
                    // In an overlap the prompt's pass goes to its stage first, so the service of generated rows is admitted between its pieces (serve_generated).
                    const bool admit = auto_ && overlap_ && !ride;
                    if (auto_ && !admit) want_service(false);
                    for (size_t k = free_apart(); !admit && !ride && k < slots_.size() && !first_stage_held(true) && form(k, active, host, Rows::generated); k = free_apart()) {}
                    for (size_t k = free_slot(); interleave_ && !ride && k < slots_.size() && !first_stage_held(false) && form(k, active, host, Rows::generated); k = free_slot()) {}
                    for (size_t k = free_slot(); k < slots_.size() && !first_stage_held(false) && form(k, active, host, ride ? Rows::all : Rows::prompt); k = free_slot()) {}
                    if (admit) serve_generated(active, host);
                    const auto left = std::find_if(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_ && Cache::image_stop(*r) > r->seq_.length(); });
                    if (!flying() && !encoded_ && left != active.end()) throw std::logic_error("server: a round with active requests formed an empty pass: " + left_out(active, **left, requests(active)));
                } catch (const std::exception& e) {
                    fail_all(active, e.what());
                    host.clear();
                }
                checkpoints_.store(model_.checkpoint_slots() - model_.checkpoints_free());
                if (timed_) round_.assembly_ms += ms_since(room_start) - formed_stages_ms_ - formed_chains_ms_;
            }
            // A failure may have emptied a deferred stage's slot and a new pass taken it, which the stage is not for.
            for (const auto& a : host)
                if (slots_[a.first].live && slots_[a.first].ran == a.second && (!a.second || slots_[a.first].formed <= formed_before_)) advance(active, a.first, a.second);
            in_flight_.store(flights_live());
            if (timed_) end_round(round_start);
            // A round that could do nothing while a stage records waits for what could give it something: a recorder done, a request or a reply to follow arriving, the disk tier, or the disk tier's next bound.
            // A cancellation does not wake it: a request whose client left ends at the next round, within a stage's time, as when a stage held this thread.
            if (!acted_ && recording()) {
                std::unique_lock<std::mutex> lk(m_);
                const auto woken = [&] { return ended() || granted() || stopping_ || cache_.woken() || arrivals_ != seen_; };
                const auto bound = cache_.next_disk_bound(lines());
                if (bound) cv_.wait_until(lk, *bound, woken);
                else cv_.wait(lk, woken);
            }
        }
        abort_all();
        // Under --disk-cache-keep every active job and request is parked at its checkpoint before anything is released: one that took its conversation's donor over is the only holder of those rows, which the flush then writes.
        // A request keeps its earlier turns' rows so, a hybrid one its prompt's whole blocks; its partial reply is not kept, and it ends cancelled.
        const bool keeps = cache_.keeps();
        while (keeps && !active.empty()) {
            const auto r = active.back();
            const size_t rows = r->seq_.length();
            const char* why = "";
            const uint64_t id = cache_.park(active, lines(), active.size() - 1, r->job_ ? r->prompt_ : history(*r), 0, &why);
            active_count_.store(requests(active));
            r->finished_ = true;
            if (id) std::fprintf(stderr, "server: cache: a %s holding %zu rows parked at the stop for the next server\n", r->job_ ? "job" : "request", rows);
            else std::fprintf(stderr, "server: cache: a %s holding %zu rows released at the stop, %s\n", r->job_ ? "job" : "request", rows, why);
            if (!r->job_) r->end("cancel");
        }
        for (auto& r : active) {
            if (r->job_) std::fprintf(stderr, "server: cache: a job holding %zu rows released at the stop\n", r->seq_.length());
            cache_.release(*r, active);
            r->end("cancel");
        }
        cache_.stop(active, lines(), cv_);
        std::lock_guard<std::mutex> lk(m_);
        for (auto* waiting : {&queue_, &paused_}) {
            for (auto& r : *waiting) r->end("cancel");
            waiting->clear();
        }
        live_.count = live_.total = 0;
        active_count_.store(0);
        paused_count_.store(0);
        in_flight_.store(0);
    }

    // The model's cache pools, the scheduler's and the cache's alike.
    static Pools pools_of(const infer::Model& model) {
        Pools p;
        for (size_t s = 0; s < model.kv_pools(); ++s) {
            p.blocks.push_back(model.kv_pool_blocks(s));
            p.block_tokens.push_back(model.kv_pool_block_tokens(s));
        }
        return p;
    }
    // The draft rows a pass carries at most, kDraftRows or `draft_max` a request where that is less, and the rows of a pass that want logits: a row a request and those drafts.
    // A caller asks the fit for both, so what the scheduler reserves is what was fitted.
    static size_t draft_rows(size_t max_seqs, size_t draft_max) { return std::min(kDraftRows, max_seqs * draft_max); }
    static size_t logits_a_pass(size_t max_seqs, size_t draft_max) { return max_seqs + draft_rows(max_seqs, draft_max); }
    // The rows of a pass apart at most: every row of it wants its logits.
    static size_t apart_rows(size_t max_seqs, size_t draft_max) { return logits_a_pass(max_seqs, draft_max); }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stopping_ = true;
            cache_.stopping();
        }
        cv_.notify_all();
    }

    // A pass as it retires, for a test that replays the passes in their order through Model::forward: each entry's request, the history it found, its rows, its extent and whether it wanted logits, and a copy of each wanting row in entry order.
    struct Retired {
        std::vector<const Request*> requests;
        std::vector<size_t> from, rows, extent;
        std::vector<char> want, every;
        std::vector<std::vector<float>> logits;
    };
    // Called on the scheduler thread with each pass that retires, before its rows are sampled; set before run.
    std::function<void(const Retired&)> on_retire;
    size_t turn_steps = 0;   // for tests: the growth steps of a turn at the pool's edge, in place of the measured ones; set before run

private:
    using Clock = std::chrono::steady_clock;
    // How requests reserve room, at admission and as uncapped ones grow.
    static constexpr Growth kGrowth{};
    // The most generated tokens a pass recomputes for one resume, each taking ubatch / kReplayRows of the budget since it takes the decode kernels; docs/STATUS-2026-09.md (Exact resume of a paused request) records the timing that sets it.
    static constexpr size_t kReplayRows = 64;
    // The most sampling threads beside the scheduler thread; docs/STATUS-2026-09.md (layer split phase 3, step 4) records why four.
    static constexpr size_t kSamplers = 4;
    // The most rows of a job a pass carries beside requests' rows, while the reply it follows is written; docs/STATUS.md (a reply read again as prompt rows) records the latency it costs.

    // The most draft rows a pass carries (docs/SPECULATIVE.md, section 3).
    static constexpr size_t kDraftRows = 64;

    // One wanting entry of a retiring pass as the sampling pool draws it: the request, its mapped logits rows read in place, the rows the channel may still take copies of, the tokens drawn and the rows sampled.
    // With logprobs asked a row is copied for the channel, or, once the reader has fallen behind (Request::kRowsWaiting), the token takes its values instead.
    struct Draw {
        Request* r;
        const float* row;
        size_t copies;
        std::vector<Request::Token> picks;
        size_t sampled = 0;
    };

    // A slot of the context reserved for passes: the round's view of it (Flight), its pass's requests by entry with the history each had and the rows each adds, those it samples in logits order, its logits rows and its decode entries.
    struct Slot : Flight {
        std::vector<std::shared_ptr<Request>> members, wanting;
        std::vector<size_t> from, rows, first_row;   // first_row: per wanting entry, its first logits row in the pass
        size_t base = 0, want = 0, decoders = 0;
        size_t generated = 0;        // its rows where every one is a generated token's, which the pass cost is measured on, else 0
        bool paced = false;          // its stages are fed in pieces (infer::Pace): a pass with prompt rows beside other requests, where passes go apart
        bool continues = false;      // formed while its request's pass before it was in flight (Scheduler::continues)
        bool timed = false;          // under the interleaved rule, a prompt's slice beside replies, whose stage time each stage's thread measures for the next slice (Recorder::Timing)
        bool short_pass = false;     // a pass that reads a short prompt ahead of a long one (short_prompts), whose time no stage's search takes
        uint64_t job = 0;            // its hand-over to a recorder, which the recorder's result names
        Clock::time_point begun;
        uint64_t prompt_of = 0;  // the admission of the first request whose prompt it slices, and that history's ends before and after it, which its pieces' record names (StagePace::piece_of)
        size_t prompt_from = 0, prompt_end = 0;
    };

    bool flying() const {
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live; });
    }
    size_t flights_live() const {
        return (size_t)std::count_if(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live; });
    }
    // A free slot for a pass, and one for a pass apart; slots_.size() when none.
    size_t free_slot() const {
        const size_t k = (size_t)(std::find_if(slots_.begin(), slots_.begin() + (std::ptrdiff_t)normal_, [](const Slot& s) { return !s.live; }) - slots_.begin());
        return k < normal_ ? k : slots_.size();
    }
    size_t free_apart() const {
        return (size_t)(std::find_if(slots_.begin() + (std::ptrdiff_t)normal_, slots_.end(), [](const Slot& s) { return !s.live; }) - slots_.begin());
    }
    // What a pass is formed of: every ready request's rows, the generating requests' alone, or the others' alone.
    enum class Rows { all, generated, prompt };
    // The rows of r's next prompt slice beside generated rows: cut from each stage's search where the slice continues the pass it measured (PieceSearch), else one step, a probe at a new depth.
    size_t piece_cut(const Request& r) {
        size_t cut = ubatch_;
        for (auto& pace : pace_) {
            std::lock_guard<std::mutex> lk(pace->m);
            // Under the interleaved rule a slice is timed after the next is formed, so any earlier slice of the same prompt continues it.
            const bool continues = pace->piece_of == r.admission_ && (interleave_ ? pace->piece_end <= r.seq_.length() : pace->piece_end == r.seq_.length());
            cut = std::min(cut, continues ? pace->search.next(ubatch_) : std::min(kPieceStep, ubatch_));
        }
        return cut;
    }
    // Whether a prompt beside generated rows is cut this round, a probe at a new depth among them: its pass is one the generated rows do not ride, so its stages measure its piece and the next slice grows from it (F2DEV, 2026-10-18 19:00).
    bool cut_now(const std::vector<std::shared_ptr<Request>>& active) {
        for (const auto& r : active)
            if (!r->job_ && !decoding(*r) && !r->seq_.in_flight() && piece_cut(*r) < std::min(history_tokens(*r) - std::min(history_tokens(*r), r->seq_.length()), ubatch_)) return true;
        return false;
    }
    // Whether the generating requests' rows ride the passes that read prompts this round: always where no pass goes apart, never while no prompt rows wait, and else by what each pass costs (rides).
    bool riding(const std::vector<std::shared_ptr<Request>>& active) const {
        if (normal_ == slots_.size()) return true;
        size_t prompt = 0, decoders = 0;
        for (const auto& r : active) {
            if (r->job_) continue;
            if (decoding(*r)) ++decoders;
            else prompt += history_tokens(*r) - std::min(history_tokens(*r), r->seq_.length());
        }
        if (!prompt) return false;
        if (!decoders) return true;
        // By the time a row where passes of 64 rows or more measured it; slices smaller than that, which never give a row's time, ride until a pass of theirs fed whole has taken longer than the gap.
        if (auto_) {
            const size_t rows = std::min(prompt, ubatch_);
            if (prompt_ms_row_ > 0) return rides_within_gap(prompt_ms_row_ * (double)rows);
            return rows < kMeasuredRows && (prompt_pass_ms_ <= 0 || rides_within_gap(prompt_pass_ms_));
        }
        // What a pass of generated rows takes of the stages itself, as their threads measured it between pieces (StageShare::price); before any, the passes' own price, measured while none waited for a piece.
        double own = 0;
        for (const auto& rec : recorders_) own += rec ? rec->price_ms.load() : 0.0;
        if (own <= 0) own = times_.pass_ms((double)decoders);
        return rides(prompt_ms_row_ * (double)std::min(prompt, ubatch_), own, share_);
    }
    // Whether replies and prompts both wait, told to the stage's clock where it changed; as the overlap ends, what the generated rows owed is dropped (PromptFloor).
    // Each request is a prompt's or a reply's by reads_prompt, which a pass in flight does not change, so it ends no overlap; a read-ahead job with rows left is the overlap's prompt where no other prompt waits, which the stage's recorder is told.
    void demand(const std::vector<std::shared_ptr<Request>>& active) {
        bool reply = false, prompt = false, job = false;
        for (const auto& r : active) {
            if (r->job_) job = job || working(*r);
            else (reads_prompt(r->gen_.size(), r->seq_.length(), history_tokens(*r)) ? prompt : reply) = true;
        }
        // A first stage on the host has no thread, and under the interleaved rule its clock serves nothing: only the overlap is kept.
        if (!recorders_[0]) {
            overlap_ = reply && (prompt || job);
            return;
        }
        Recorder& rec = *recorders_[0];
        const bool background = reply && job && !prompt;
        bool told = false;
        {
            std::lock_guard<std::mutex> lk(rec.m);
            told = rec.background != background;
            rec.background = background;
        }
        if (told) rec.cv.notify_all();
        if ((reply && (prompt || job)) == overlap_) return;
        overlap_ = reply && (prompt || job);
        std::lock_guard<std::mutex> lk(rec.m);
        if (rec.clock.demand(overlap_, now_ms())) rec.demand_ended = true;
    }

    double now_ms() const { return std::chrono::duration<double, std::milli>(Clock::now() - epoch_).count(); }
    // In an overlap a service of generated rows, a pass with its drafts' chain before it, is formed whole once the stage has admitted it (Recorder::granted).
    // While a prompt's pass is fed in pieces the round only asks (want) and the stage's thread grants at a piece boundary; with none being fed the pass is formed as without pacing.
    void serve_generated(std::vector<std::shared_ptr<Request>>& active, std::vector<std::pair<size_t, size_t>>& host) {
        const size_t k = free_apart();
        if (k == slots_.size() || first_stage_held(true)) return;
        const bool ready = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) {
            return !r->job_ && decoding(*r) && !r->seq_.in_flight() && !r->cancel_.load();
        });
        Recorder& rec = *recorders_[0];
        bool asked = false, waits = false;
        {
            std::lock_guard<std::mutex> lk(rec.m);
            if (!rec.granted) {
                const bool fed = rec.pass.posted && rec.pass.paced;
                rec.want = fed && ready;
                asked = rec.want;
                // A prompt's pass recorded and not yet retired has no boundary left to admit at: the service waits for the next pass's.
                const bool leaving = std::any_of(slots_.begin(), slots_.end(), [](const Slot& o) { return o.live && !o.apart; });
                waits = fed || leaving || !ready;
            }
        }
        // A recorder serving a job's gaps waits for the wish rather than meeting it at a boundary.
        if (asked) rec.cv.notify_all();
        if (waits) return;
        // The answer to a grant, whatever forming does: the stage stands parked until it comes.
        struct Answer {
            Scheduler& s;
            ~Answer() { s.want_service(false); }
        } answer{*this};
        // The service begins here, as the round takes the grant up, not when the stage offered it: what the round did in between was not the generated rows' work.
        const double began = now_ms();
        if (!form(k, active, host, Rows::generated)) return;
        std::lock_guard<std::mutex> lk(rec.m);
        if (rec.granted && rec.apart.posted) rec.clock.began(began);
    }
    // The round's wish for a service, withdrawn with any grant given for it, which lets a stage standing parked on one go on.
    void want_service(bool want) {
        Recorder& rec = *recorders_[0];
        {
            std::lock_guard<std::mutex> lk(rec.m);
            if (rec.want == want && !rec.granted) return;
            rec.want = want;
            rec.granted = false;
        }
        rec.cv.notify_all();
    }
    // Whether a stage has granted a service the round has not answered; under the lock.
    bool granted() const {
        if (!auto_) return false;
        std::lock_guard<std::mutex> lk(recorders_[0]->m);
        return recorders_[0]->granted;
    }
    std::vector<Flight> flights() const { return std::vector<Flight>(slots_.begin(), slots_.end()); }

    // A new pass in slot k: decode entries first, up to an even share of the decoding requests over the pass slots (not one that could not grow or is in flight), then one slice of what each other request's cache lacks, up to ubatch tokens.
    // Then its logits rows, begin_pass and its first stage, or on the host that stage's place in `host`; false when no request has rows to add, which with nothing in flight throws rather than leaving the loop to spin.
    bool form(size_t k, std::vector<std::shared_ptr<Request>>& active, std::vector<std::pair<size_t, size_t>>& host, Rows rows) {
        Slot& f = slots_[k];
        entries_.clear();
        f.members.clear();
        f.wanting.clear();
        f.from.clear();
        f.rows.clear();
        f.first_row.clear();
        f.decoders = 0;
        f.want = 0;
        f.short_pass = false;
        const auto add_entry = [&](const std::shared_ptr<Request>& r, const infer::BatchEntry& e) {
            entries_.push_back(e);
            f.members.push_back(r);
            f.from.push_back(r->seq_.length());
            f.rows.push_back(e.n);
            if (!e.want_logits) return;
            f.wanting.push_back(r);
            f.first_row.push_back(f.want);
            f.want += e.every_logits ? e.n : 1;
        };
        const auto ready = [](const Request& r) { return !r.seq_.in_flight() && !r.stalled_; };
        // The share goes to the decoders that left flight earliest, so one held back takes the next pass; they join in order of first admission.
        std::vector<const Request*> waiting;
        size_t decoders = 0;
        for (const Slot& s : slots_) decoders += s.live ? s.decoders : 0;
        for (const auto& r : active)
            if (rows != Rows::prompt && ready(*r) && decoding(*r)) waiting.push_back(r.get());
        decoders += waiting.size();
        const size_t share = std::min(decode_share(decoders, f.apart ? slots_.size() - normal_ : normal_, model_.stage_count()), waiting.size());
        std::stable_sort(waiting.begin(), waiting.end(), [](const Request* a, const Request* b) { return a->landed_ < b->landed_; });
        waiting.resize(share);
        // A decoding request with drafts takes a verify entry, its last pick then the drafts, every row's logits, extent 1, in the decode columns the pass's decoders leave.
        std::vector<std::shared_ptr<Request>> decoding_now;
        for (auto& r : active)
            if (std::find(waiting.begin(), waiting.end(), r.get()) != waiting.end()) decoding_now.push_back(r);
        propose(decoding_now, draft_columns(share, active.size()));
        for (auto& r : decoding_now) {
            if (r->verify_.size() > 1) {
                infer::BatchEntry e{&r->seq_, r->verify_.data(), r->verify_.size(), true, true};
                e.extent = 1;
                add_entry(r, e);
            } else {
                add_entry(r, infer::BatchEntry{&r->seq_, &r->last_id_, 1, true});
            }
            ++f.decoders;
        }
        size_t budget = rows == Rows::generated ? 0 : ubatch_, keeps = 0;
        // What r's cache lacks, one stretch's slice of at most `most` rows within the budget.
        const auto slice = [&](const std::shared_ptr<Request>& r, size_t most) {
            const size_t at = r->seq_.length(), end = history_tokens(*r), stop = Cache::image_stop(*r);
            if (at >= end || at >= stop) return;
            const RowClass& c = class_at(r->classes_, at);
            size_t n = std::min({c.end, end, stop, at + most}) - at;
            // A job's slice ends on a whole block where it can, so what it has read can be forked there.
            const size_t bt = model_.kv_block_tokens();
            if (r->job_ && (at + n) / bt * bt > at) n = (at + n) / bt * bt - at;
            // A slice that would pass the request's checkpoint or its message boundary ends there, so its state can be kept there.
            if (r->keep_at_ > at && r->keep_at_ < at + n) n = r->keep_at_ - at;
            if (r->boundary_at_ > at && r->boundary_at_ < at + n) n = r->boundary_at_ - at;
            // And at the next position its proof compares at (prove).
            if (r->prove_at_ > at && r->prove_at_ < at + n) n = r->prove_at_ - at;
            if (c.extent == 1 && at < r->reached_) {
                // Rows of extent 1 a resume computes again, generated tokens or a forked reply's: a pass takes at most kReplayRows of them, each costing ubatch / kReplayRows of the budget, so a request gets one while the budget is untouched.
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
                e.keep = cache_.checkpoint_room(keeps, r->source_);
                keeps += e.keep;
                r->keep_at_ = 0;
            }
            // The boundary's checkpoint goes to host memory once its pass retires (keep_boundary), and the request's own at keep_at_ then takes its place, a sequence holding one; where the two are one position, one checkpoint is both.
            if (r->boundary_at_ && r->boundary_at_ == at + n) {
                if (!e.keep) {
                    e.keep = cache_.checkpoint_room(keeps, r->source_);
                    keeps += e.keep;
                }
                if (!e.keep) cache_.moved(Move::boundary_not_kept, "server: cache: the message boundary asked at %zu tokens was not kept (no checkpoint slot for it)\n", at + n);
                r->boundary_at_ = 0;
                r->boundary_kept_ = e.keep;
            }
            // The slice that ends a proof's path keeps its state there, which the proof compares.
            if (r->prove_state_ && r->prove_at_ == at + n && !e.keep) {
                e.keep = cache_.checkpoint_room(keeps, r->source_);
                keeps += e.keep;
            }
            add_entry(r, e);
        };
        // A lone request's prompt slices shrink so the stages of a split read it together (prompt_slice); alone means nothing else active, queued or paused after this round's admissions, and company that arrives mid-prompt brings whole slices back.
        bool company;
        {
            std::lock_guard<std::mutex> lk(m_);
            company = !queue_.empty() || !paused_.empty();
        }
        const bool alone = active.size() == 1 && !active[0]->job_ && !company;
        // Under the automatic rule, beside generated rows, each stage's search over its passes fed in pieces cuts the next slice of the prompt it measured (PieceSearch).
        // A slice that does not continue that pass, another prompt or another depth, starts at one step and the search starts again from it.
        const bool cutting = (auto_ || interleave_) && overlap_.load();
        // Under the interleaved rule in an overlap a pass holds one prompt's slice, formed only once that prompt's last slice has left the first stage, so a pass of generated rows formed meanwhile goes before it there.
        const bool one_slice = interleave_ && cutting && rows == Rows::prompt;
        const bool generating = replying(active);
        f.prompt_of = 0;
        f.prompt_from = 0;
        f.prompt_end = 0;
        // Short prompts beside a long one (docs/SERVER.md): the later prompts whose remaining rows fit a share of the ubatch are read whole ahead of the oldest long prompt, in arrival order (short_prompts).
        // Beside generated rows one is read in a pass of its own, which no stage's search times; there, and ahead of a long prompt in flight, which would continue, only once a round.
        std::vector<Request*> shorts;
        const auto reads = [&](const Request& q) { return !q.job_ && !decoding(q) && q.seq_.length() < history_tokens(q); };
        const auto lead = std::find_if(active.begin(), active.end(), [&](const std::shared_ptr<Request>& q) { return reads(*q); });
        const size_t base = ubatch_;
        // Only a long prompt, one past the share, has short ones read ahead of it, so short prompts keep their order among themselves; and not while its slice takes it back to a whole ubatch in a pass of its own (prompt_slice).
        const size_t lead_left = lead == active.end() ? 0 : history_tokens(**lead) - std::min(history_tokens(**lead), (*lead)->seq_.length());
        if (rows != Rows::generated && budget && lead_left > base / kShortShare && !prompt_slice(lead_left, model_.stage_count(), ubatch_, alone, (*lead)->read_alone_).closes) {
            const bool ahead = cutting || (*lead)->seq_.in_flight();
            if (!ahead || !short_ahead_) {
                std::vector<size_t> waiting;
                std::vector<Request*> later;
                for (auto it = lead + 1; it != active.end(); ++it) {
                    if (!reads(**it)) continue;
                    later.push_back(it->get());
                    waiting.push_back(ready(**it) && !(*it)->cancel_.load() ? history_tokens(**it) - (*it)->seq_.length() : 0);
                }
                for (size_t i : short_prompts(waiting, base)) shorts.push_back(later[i]);
                if (cutting && shorts.size() > 1) shorts.resize(1);
                short_ahead_ = short_ahead_ || (ahead && !shorts.empty());
            }
        }
        const auto is_short = [&](const Request* q) { return std::find(shorts.begin(), shorts.end(), q) != shorts.end(); };
        std::vector<std::shared_ptr<Request>> order;
        for (auto& r : active)
            if (is_short(r.get())) order.push_back(r);
        for (auto& r : active)
            if (!is_short(r.get())) order.push_back(r);
        for (auto& r : order) {
            if (r->job_ || decoding(*r) || !budget) continue;
            if (one_slice && (f.prompt_of || (r->slice_ticket_ && !model_.stage_backend(0).done(r->slice_ticket_)))) continue;
            const size_t waiting_rows = history_tokens(*r) - std::min(history_tokens(*r), r->seq_.length());
            const PromptSlice s = prompt_slice(waiting_rows, model_.stage_count(), ubatch_, alone, r->read_alone_);
            // A short prompt is read whole, within the share its selection kept.
            const size_t most = is_short(r.get()) ? waiting_rows : std::min(s.most, cutting ? piece_cut(*r) : ubatch_);
            // A prompt whose slice in flight has committed the first stage continues in a pass of its own (continues).
            const bool continued = r->seq_.in_flight();
            if (continued && !(entries_.empty() && continues(*r, generating))) {
                if (s.closes) break;
                continue;
            }
            // A prompt at an image without rows spends its turn on one piece of that image's encode, one a round, every piece while nothing else could use the stages, and the pass takes no more prompt rows (docs/VISION.md, section 10).
            if (Cache::image_stop(*r) <= r->seq_.length()) {
                if (encoded_) continue;
                // A health poll ends the run of pieces too, so the round boundary it waits for comes.
                const auto more = [&] {
                    if (!alone || flying() || !entries_.empty() || r->cancel_.load()) return false;
                    std::lock_guard<std::mutex> lk(m_);
                    return arrivals_ == seen_ && !stopping_ && !live_requested_;
                };
                if (!continued && cache_.encode_piece(*r, more)) {
                    acted_ = encoded_ = true;
                    budget = 0;
                    break;
                }
                continue;
            }
            const size_t before = entries_.size();
            slice(r, std::min(budget, most));
            f.short_pass = f.short_pass || (is_short(r.get()) && entries_.size() > before);
            if (!f.prompt_of && entries_.size() > before) {
                f.prompt_of = r->admission_;
                f.prompt_from = r->seq_.length();
                f.prompt_end = r->seq_.length() + entries_.back().n;
            }
            // What it read alone counts until company comes or its prompt has been sliced whole.
            if (!alone || entries_.size() == before || r->seq_.length() + entries_.back().n >= history_tokens(*r)) r->read_alone_ = 0;
            else r->read_alone_ += entries_.back().n;
            // Beside generated rows a short prompt's pass holds no other prompt, so the long prompt's slices keep continuing its stages' search.
            if (cutting && f.short_pass) break;
            if (continued && entries_.size() > before) {
                budget = 0;
                break;
            }
            if (s.closes) break;
        }
        // Then a read-ahead job, a prompt of no client at the lowest priority: only where no other request reads its prompt, and whole while nothing generates either.
        // Beside replies it takes a piece after kBackgroundGaps passes of generated rows alone, as the overlap's prompt where a stage is fed in pieces, whose boundaries then serve as many (serve_gaps); a waiting turn-end write takes every other turn.
        const bool prompts_wait = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_ && reads_prompt(r->gen_.size(), r->seq_.length(), history_tokens(*r)); });
        const bool replies = generating;
        if (budget && !f.prompt_of && !prompts_wait && (!replies || (reply_gaps_ >= kBackgroundGaps && !(write_waits_ && background_last_ == Background::job)))) {
            for (auto& r : active) {
                if (!r->job_ || !budget) continue;
                // A job whose slice in flight has committed the first stage continues in a pass of its own, as a prompt does (continues).
                const bool continued = r->seq_.in_flight();
                if (continued && !(entries_.empty() && continues(*r, replies))) continue;
                if (Cache::image_stop(*r) <= r->seq_.length()) {
                    if (encoded_ || continued || !cache_.encode_piece(*r, [] { return false; })) continue;
                    acted_ = encoded_ = true;
                    budget = 0;
                    if (replies) {
                        reply_gaps_ = 0;
                        background_last_ = Background::job;
                    }
                    break;
                }
                const size_t before = entries_.size();
                slice(r, replies ? std::min(budget, piece_cut(*r)) : budget);
                if (entries_.size() == before) continue;
                if (!f.prompt_of) {
                    f.prompt_of = r->admission_;
                    f.prompt_from = r->seq_.length();
                    f.prompt_end = r->seq_.length() + entries_.back().n;
                }
                if (continued) budget = 0;
                if (!replies) {
                    if (continued) break;
                    continue;
                }
                reply_gaps_ = 0;
                background_last_ = Background::job;
                break;
            }
        }
        if (entries_.empty()) return false;
        const bool continuing = f.members.front()->seq_.in_flight();
        // A pass apart takes its logits rows from the rows after the other passes'.
        LogitRows& ring = f.apart ? apart_rows_ : logit_rows_;
        f.base = take_rows(ring, f.want);
        if (f.base == ring.size) throw std::logic_error("server: no logits rows for a new pass");
        try {
            model_.begin_pass(ctx_, k, entries_.data(), entries_.size(), f.base + (f.apart ? logit_rows_.size : 0), f.apart);
        } catch (...) {
            give_rows(ring, f.base, f.want);
            throw;
        }
        // Where passes go apart, a pass with prompt rows beside other requests is fed in pieces, so a pass apart waits a piece at a stage and not the stage.
        apart_formed_ += f.apart;
        prompt_passes_ += f.prompt_of != 0;
        continued_ += continuing;
        short_passes_ += f.short_pass;
        f.continues = continuing;
        if (f.apart) ride_run_.apart();
        if (normal_ < slots_.size() && !f.apart && f.decoders && f.members.size() > f.decoders) {
            ++rode_;
            ride_run_.rode();
        }
        // Under the automatic rule only a prompt's pass that generated rows do not ride is fed in pieces: a ridden pass has nothing to take between them.
        f.paced = normal_ < slots_.size() && !f.apart && (auto_ ? overlap_ && rows == Rows::prompt : requests(active) > 1);
        f.timed = one_slice && f.prompt_of != 0 && !f.short_pass;
        if (interleave_ && overlap_.load()) {
            size_t prompts = 0;
            for (size_t e = 0; e < f.members.size(); ++e) prompts += !decoding(*f.members[e]) && !f.members[e]->job_;
            interleaved_replies_ += rows == Rows::generated;
            interleaved_slices_ += f.timed;
            interleaved_mixed_ += (f.decoders && prompts) || prompts > 1;
        }
        f.live = true;
        f.formed = ++formed_;
        f.ran = 0;
        f.generated = 0;
        if (std::all_of(entries_.begin(), entries_.end(), [](const infer::BatchEntry& e) { return e.extent == 1 || e.n == 1; }))
            for (const infer::BatchEntry& e : entries_) f.generated += e.n;
        f.begun = Clock::now();
        if (model_.stage_on_host(0)) host.push_back({k, 0});
        else advance(active, k, 0);
        return true;
    }

    // Stage s of the pass in slot k; a failure has abandoned the pass in the model, and it fails that pass alone.
    void advance(std::vector<std::shared_ptr<Request>>& active, size_t k, size_t s) {
        // A pass fed in pieces goes to the stage's thread, and so does one that would overlap another: where the stage has a thread without pacing, and a pass apart or beside one.
        const bool threaded = plain_[s] || slots_[k].apart || std::any_of(slots_.begin(), slots_.end(), [](const Slot& o) { return o.live && o.apart; });
        // A pass that carries generated rows is a service of theirs, which the stage's clock is told as it begins; a pass apart tells it where it is recorded.
        const bool serves = auto_ && !slots_[k].apart && slots_[k].decoders;
        // Without the automatic rule nothing else tells a stage's clock of a service, which the background budget reads on every placement.
        if (!auto_ && slots_[k].decoders) {
            std::lock_guard<std::mutex> lk(pace_[s]->m);
            pace_[s]->clock.began(now_ms());
        }
        if (Recorder* r = slots_[k].paced || (slots_[k].timed && recorders_[s]) || (threaded && (model_.stage_reads(s) || recorded_apart(model_.stage_grouped(s), flights_live()))) ? recorders_[s].get() : nullptr) {
            // A stage records one pass of a kind at a time: a pass that finds its recorder taken waits for a later round, which gives the stage its oldest waiting pass.
            if (recording(s, slots_[k].apart)) return;
            acted_ = true;
            // The stage's blocks are taken here, on the thread that owns the pools; a pass that cannot have them fails alone, as a stage recorded inline does.
            try {
                model_.prepare_pass_stage(ctx_, k, s);
            } catch (const std::exception& e) {
                const std::vector<size_t> taken = abandon(k);
                fail_pass(active, k, e.what(), taken);
                return;
            }
            slots_[k].recording = true;
            {
                std::lock_guard<std::mutex> lk(r->m);
                Recorder::Post& post = slots_[k].apart ? r->apart : r->pass;
                post.slot = k;
                post.job = slots_[k].job = ++jobs_posted_;
                post.paced = slots_[k].paced;
                post.timed = slots_[k].timed;
                post.prompt_of = slots_[k].prompt_of;
                post.prompt_end = slots_[k].prompt_end;
                // A pass of generated rows alone gives no prompt piece's time (Slot::generated).
                post.rows = 0;
                for (size_t n : slots_[k].rows) post.rows += slots_[k].generated ? 0 : n;
                post.deferred = false;
                post.serves = serves || slots_[k].apart;
                post.posted = true;
            }
            r->cv.notify_all();
            return;
        }
        acted_ = true;
        if (serves) {
            std::lock_guard<std::mutex> lk(recorders_[s]->m);
            recorders_[s]->clock.began(now_ms());
        }
        const Clock::time_point start = timed_ ? Clock::now() : Clock::time_point{};
        const backend::Backend::HostTimes before = timed_ ? host_times() : backend::Backend::HostTimes{};
        try {
            // Undoing the pass drains every device, which a stage being recorded on its thread must not see, so the recorders go quiet first.
            model_.run_pass_stage(ctx_, k, s, [this] { quiet(); });
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

    // A timed round's end: its time, and every kReadRounds rounds each stage's device time since the last reading, which on a device waits for its queue, so the span a reading covers starts after it.
    // A reading is a call on the stage's devices, so it is taken with every recorder idle (quieted).
    void end_round(Clock::time_point start) {
        static constexpr uint64_t kReadRounds = 32;
        round_.round_ms += ms_since(start);
        ++round_.rounds;
        const size_t S = model_.stage_count();
        if (round_.stage_ms.size() != S) {
            round_.stage_ms.assign(S, 0.0);
            std::fill(host_stage_ms_.begin(), host_stage_ms_.end(), 0.0);
            for (size_t s = 0; s < S; ++s) quieted()->stage_backend(s).device_ms();
            span_start_ = Clock::now();
            span_rows_ = 0;
        } else if (round_.rounds % kReadRounds == 0) {
            round_.span_ms += ms_since(span_start_);
            for (size_t s = 0; s < S; ++s) {
                const double ms = quieted()->stage_backend(s).device_ms();
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
        if (turn_to_ && std::any_of(f.members.begin(), f.members.end(), [&](const std::shared_ptr<Request>& m) { return m->admission_ == turn_to_; })) {
            const double ms = ms_since(turned_);
            swap_ms_ = kGrowth.change(changed_ms_, ms);
            changed_ms_ = ms;
            turn_to_ = 0;
        }
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
                    t.every.push_back(entries[e].every_logits);
                }
                for (size_t w = 0; w < f.want; ++w) {
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
                const float* row = model_.pass_logits(ctx_, k, f.first_row[w]);
                if (timed_) waited += ms_since(read);
                const size_t waiting = r.params_.logprobs ? r.rows_waiting() : Request::kRowsWaiting;
                draws_.push_back(Draw{&r, row, waiting < Request::kRowsWaiting ? Request::kRowsWaiting - waiting : 0, {}, 0});
            }
            samplers_.run(draws_.size(), [this](size_t i) { draw(draws_[i]); });
            for (Draw& d : draws_) step(d);
            if (timed_) {
                round_.logits_wait_ms += waited;
                round_.sampling_ms += ms_since(start) - waited;
                for (size_t n : f.rows) span_rows_ += n;
            }
        } catch (const std::exception& e) {
            const std::vector<size_t> taken = abandon(k);
            fail_pass(active, k, e.what(), taken);
            return;
        }
        model_.end_pass(ctx_, k);
        uint64_t prefilled = 0;
        for (size_t e = f.decoders; e < f.members.size(); ++e) {
            Request& r = *f.members[e];
            if (!r.job_) {
                const size_t n = std::min(f.rows[e], r.prompt_.size() - std::min(f.from[e], r.prompt_.size()));
                r.prefilled_ += n;
                prefilled += n;
            }
        }
        if (prefilled) prefilled_total_ += prefilled;
        settle_verifies(k);
        // A pass's cost is the time the server gave it since the last pass of its own kind retired, less the chains drafted meanwhile, which the price counts apart; passes apart retire past the others.
        Retirement& last = retirement_[f.apart];
        const double since_retired = last.any ? ms_since(last.at) - last.chain_ms : std::numeric_limits<double>::infinity();
        const double took = std::min(ms_since(f.begun), since_retired);
        // A pass with prompt rows that was fed whole gives the time such a pass takes, whatever its rows are classed as below (riding).
        if (!f.apart && !f.paced && f.members.size() > f.decoders) prompt_pass_ms_ = took;
        if (f.generated) times_.pass(f.generated, took);
        else measure_prompt(f, since_retired);
        last = Retirement{Clock::now(), true, 0};
        // Rows a resume computed again are those below the longest history the cache has held, a verify's those its retract kept.
        size_t again = 0, job_rows = 0;
        for (size_t e = 0; e < f.members.size(); ++e) {
            Request& r = *f.members[e];
            const size_t to = r.seq_.length(), n = std::min(to, r.reached_) - std::min(f.from[e], r.reached_);
            r.recomputed_ += n;
            again += n;
            r.reached_ = std::max(r.reached_, to);
            r.landed_ = f.formed;
            if (r.first_pass_) {
                r.first_pass_ = false;
                std::lock_guard<std::mutex> lk(r.m_);
                ++r.restore_.first_passes;
                r.restore_.part(r.restore_.first_pass_ms, std::chrono::duration<double, std::milli>(Clock::now() - f.begun).count(), r.first_before_);
            }
            if (r.job_) job_rows += f.rows[e];
            if (r.boundary_kept_) {
                // A state computed over rows restored from unproved files is kept by no one: prove keeps the one at its path's end once the proof has ended equal.
                r.boundary_kept_ = r.unproved_ && to == r.prove_at_;
                std::lock_guard<std::mutex> lk(m_);
                if (!r.cancel_.load() && !r.unproved_) cache_.asked_boundary(r);
            }
        }
        // A job's pass begins the count of passes of generated rows alone its next piece beside replies waits for.
        if (job_rows) reply_gaps_ = 0;
        else if (f.decoders && f.decoders == f.members.size()) ++reply_gaps_;
        if (again || job_rows) {
            std::lock_guard<std::mutex> lk(m_);
            recomputed_ += again;
            cache_.job_rows_read(job_rows);
        }
        vacate(k);
    }

    // Slot k is free again, and the logits rows its pass held come back.
    void vacate(size_t k) {
        Slot& f = slots_[k];
        give_rows(f.apart ? apart_rows_ : logit_rows_, f.base, f.want);
        f.live = false;
        f.recording = false;
        f.members.clear();
        f.wanting.clear();
        f.decoders = 0;
        f.want = 0;
    }

    // The decode columns a pass with `decoders` decode entries leaves for drafts (docs/SPECULATIVE.md, section 3): those its kernels read each weight once for, past one a decoder, less the drafts passes in flight carry, at most draft_rows_.
    // None while every seat is taken (`seated`) and a request is queued for one, where drafts cost output and first-token p99 (docs/STATUS.md); a request paused or queued for room changes nothing.
    size_t draft_columns(size_t decoders, size_t seated) {
        if (!proposer_) return 0;
        if (seated >= max_seqs_) {
            std::lock_guard<std::mutex> lk(m_);
            if (!queue_.empty()) return 0;
        }
        const size_t columns = model_.decode_columns();
        size_t free = columns > decoders ? columns - decoders : 0;
        if (decoders == 1) free = std::max(free, draft_max_);
        size_t flying = 0;
        for (const Slot& s : slots_)
            if (s.live)
                for (const auto& m : s.members) flying += m->verify_.empty() ? 0 : m->verify_.size() - 1;
        return std::min(free, draft_rows_ - std::min(draft_rows_, flying));
    }

    // The verifies of the decoding requests `rs` in the pass being formed, each in r.verify_ on a mark of its sequence, its last pick then its drafts, or empty for a decode entry.
    // Each drafts within the `columns` left, its reservation and tokens left (spec::draft_length) and the price's gain (spec::draft_depths): from its own history where that convinces (spec::lookup_first), else from the proposer.
    void propose(const std::vector<std::shared_ptr<Request>>& rs, size_t columns) {
        using infer::spec::Source;
        for (const auto& r : rs) r->verify_.clear();
        if (!proposer_) return;
        keeps_.clear();
        askers_.clear();
        chained_.clear();
        const bool first = proposer_->after_lookup();
        for (const auto& r : rs) {
            // A request resumed by a fork of its whole history decodes before any pass has fed its sequence, and drafts from the pass after (Model::can_draft).
            if (r->job_ || !model_.can_draft(r->seq_)) continue;
            const size_t at = r->seq_.length(), room = std::min(reserved_tokens(*r), token_limit());
            const size_t left = (size_t)r->params_.max_tokens - r->gen_.size(), positions = room > at ? room - at : 0;
            infer::spec::Record& record = r->record_;
            record.drafted = Source::proposer;
            size_t k = 0;
            // Its history goes into the ask it would be this pass's next asker of, where the proposer's ask finds it again.
            if (first) {
                if (asks_.size() <= askers_.size()) asks_.resize(askers_.size() + 1);
                std::vector<uint32_t>& history = asks_[askers_.size()].history;
                history.assign(r->prompt_.begin(), r->prompt_.end());
                history.insert(history.end(), r->gen_.begin(), r->gen_.end());
                asks_[askers_.size()].images = model_.image_rows(r->seq_);
                k = infer::spec::lookup_first(record, draft_max_, left, positions, columns, history, r->own_draft_, asks_[askers_.size()].images);
            }
            if (!k) k = infer::spec::draft_length(draft_max_, left, positions, columns, record.proposer);
            if (!k) continue;
            columns -= k;
            keeps_.emplace_back(k);
            // A draft is priced by its source's acceptance, a history draft's by its match bucket's, each weighed with the server's of the same.
            for (size_t j = 0; j < k; ++j) keeps_.back()[j] = record.drafting().keeps(j, tally_.at(record.drafted, record.bucket));
            chained_.push_back(record.drafted == Source::proposer);
            askers_.push_back(r.get());
        }
        // A chain holds the scheduler thread and queues behind the head device's work, so every pass in flight waits on it as the one being formed does.
        infer::spec::PassCost cost = priced_ ? times_.cost() : infer::spec::PassCost{};
        const double stall = 1.0 + (double)flights_live();
        cost.step_ms *= stall;
        cost.step_row_ms *= stall;
        infer::spec::draft_depths(cost, rs.size(), keeps_, takes_, &chained_);
        // A model that keeps a state holds as many marks as its fit gave, some held by verifies in flight, and a request without one drafts nothing, so no more draft than are free.
        size_t held = 0;
        for (const Slot& s : slots_)
            if (s.live)
                for (const auto& m : s.members) held += !m->verify_.empty();
        const size_t marks = model_.mark_slots() > held ? model_.mark_slots() - held : 0;
        // The requests that draft, within the marks: `asked_[i]` is request i's place among the proposer's asks, or none for a draft of its own history.
        size_t n = 0, chains = 0, steps = 0;
        asked_.assign(askers_.size(), SIZE_MAX);
        for (size_t i = 0; i < askers_.size(); ++i) {
            if (!takes_[i] || n == marks) {
                takes_[i] = 0;
                continue;
            }
            ++n;
            if (!chained_[i]) continue;
            Request& r = *askers_[i];
            if (asks_.size() <= chains) asks_.resize(chains + 1);
            infer::spec::Proposer::Ask& a = asks_[chains];
            a.seq = &r.seq_;
            if (first) {
                if (chains != i) {
                    std::swap(a.history, asks_[i].history);
                    std::swap(a.images, asks_[i].images);
                }
            } else {
                a.history.assign(r.prompt_.begin(), r.prompt_.end());
                a.history.insert(a.history.end(), r.gen_.begin(), r.gen_.end());
                a.images = model_.image_rows(r.seq_);
            }
            a.k = takes_[i];
            asked_[i] = chains++;
            steps = std::max(steps, a.k);
        }
        if (!n) return;
        // A draft is an optimisation: one that fails costs its pass the proposer's drafts and no request.
        bool failed = false;
        if (chains) {
            const Clock::time_point start = Clock::now();
            try {
                // The chain is work on the head's device, which a stage's thread may be recording on.
                const Held still(*this, true);
                proposer_->draft_all(asks_.data(), chains);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "server: a draft failed and its pass runs without the proposer's drafts (%s)\n", e.what());
                ++draft_failures_;
                failed = true;
            }
            const double chain = ms_since(start);
            if (!failed) times_.chain(steps, chains, chain);
            round_.drafting_ms += chain;
            formed_chains_ms_ += chain;
            for (Retirement& kind : retirement_) kind.chain_ms += chain;
        }
        for (size_t i = 0; i < askers_.size(); ++i) {
            if (!takes_[i] || (chained_[i] && failed)) continue;
            Request& r = *askers_[i];
            const std::vector<uint32_t>& d = chained_[i] ? asks_[asked_[i]].out : r.own_draft_;
            // A draft past the vocabulary, and those after it, are not fed: a verify needs nothing a decode does not.
            const auto past = std::find_if(d.begin(), d.end(), [&](uint32_t id) { return id >= model_.n_vocab(); });
            const size_t k = std::min(takes_[i], (size_t)(past - d.begin()));
            if (!k || !quieted(true)->mark(r.seq_)) continue;
            r.verify_.assign(1, r.last_id_);
            r.verify_.insert(r.verify_.end(), d.begin(), d.begin() + (std::ptrdiff_t)k);
        }
    }

    // The positions request r's reserved blocks hold in every pool.
    size_t reserved_tokens(const Request& r) const {
        if (r.need_.empty()) return 0;
        size_t n = SIZE_MAX;
        for (size_t s = 0; s < r.need_.size(); ++s) n = std::min(n, r.need_[s] * pools_.block_tokens[s]);
        return n;
    }

    // After slot k's pass has ended, each verify's history goes back to what the run without drafts holds, the last pick and the drafts its picks kept, before anything parks, pauses or forks it (docs/SPECULATIVE.md, section 3).
    // A request cancelled in flight keeps its last pick alone, and a retract whose rerun fails ends the request with the error.
    void settle_verifies(size_t k) {
        Slot& f = slots_[k];
        const Clock::time_point start = timed_ ? Clock::now() : Clock::time_point{};
        for (size_t e = 0; e < f.members.size(); ++e) {
            Request& r = *f.members[e];
            if (r.verify_.empty()) continue;
            size_t keep = f.from[e] + 1;
            for (const Draw& d : draws_)
                if (d.r == &r) keep = f.from[e] + d.sampled;
            const size_t k_drafts = r.verify_.size() - 1;
            r.verify_.clear();
            try {
                quieted(true)->retract(r.seq_, keep);
            } catch (const std::exception& ex) {
                r.finish_pending_ = "error";
                r.error_pending_ = ex.what();
                continue;
            }
            std::lock_guard<std::mutex> lk(m_);
            tally_.at(r.record_.drafted, r.record_.bucket).verified(keep - f.from[e] - 1, k_drafts);
        }
        if (timed_) round_.settling_ms += ms_since(start);
    }

    // A stage's pacing state, which the scheduler keeps for every stage whether or not a thread records it: when generated rows were last served, what they owe prompts, the holds' time and the background time given.
    struct StagePace {
        std::mutex m;
        ServiceClock clock;
        PromptFloor floor;
        HeldTime held;
        PieceSearch search;      // its passes fed in pieces of one prompt, continuing each other, by which the next slice is cut
        uint64_t piece_of = 0;   // the last such pass's prompt (Request::admission_) and its history's end after it: a slice continues it only there
        size_t piece_end = 0;
        double apart_ms = 0;     // its last pass apart's time, which a piece beside replies may take (PieceSearch)
        SliceRecord slice;       // its last pass fed in pieces, for the health
    };
    // A stage's thread: it records the pass the round hands it (advance) and decides nothing, and the round takes what it did (collect); a failure is the scheduler's thread's to undo.
    // Beside its pass it holds one pass apart, recorded between the first's pieces once its rows have arrived and the generated rows are due (StageShare), or alone.
    struct Recorder {
        Recorder(StagePace& pace, infer::StageLanes::Lane& lane)
            : m(pace.m), cv(lane.cv), held(pace.held), clock(pace.clock), floor(pace.floor), apart_ms(pace.apart_ms), quit(lane.stop), stage(pace) {}
        // A slice timed on this stage: from when the stage's work before it has retired to when its own has, so its stage time as a pass behind it waits for it, which the stage's search takes as the slice's piece (PieceSearch).
        struct Timing {
            backend::Ticket before = 0, own = 0;
            size_t rows = 0;
            uint64_t prompt_of = 0;
            size_t prompt_end = 0;
            bool started = false;
            Clock::time_point from;
        };
        struct Post {
            bool posted = false;
            size_t slot = 0;
            uint64_t job = 0;   // the hand-over, which the recording's end names
            bool paced = false;
            bool deferred = false;   // a pass apart that was due and waited for the floor (ServiceClock::deferred)
            bool serves = false;     // it carries generated rows, so its recording's start is a service's (ServiceClock::began)
            size_t rows = 0;         // the pass's rows, where any is a prompt's
            bool timed = false;      // a prompt's slice under the interleaved rule, timed on this stage (Timing), of the prompt and to the history's end named here
            uint64_t prompt_of = 0;
            size_t prompt_end = 0;
        };
        struct End {
            size_t slot = 0;
            uint64_t job = 0;
            bool failed = false;
            std::string error;
            size_t rows = 0;                      // a pass fed in pieces: its rows, its longest piece and the time of all its pieces, the passes apart between them left out
            double longest_ms = 0;
            SliceRecord slice{};                  // and its pieces as the health gives them
        };
        std::mutex& m;           // its stage's (StagePace), as are held, clock and floor
        std::condition_variable& cv;   // its lane's wake (infer::StageLanes)
        Post pass, apart;        // under m
        std::vector<End> ends;   // under m, the recordings ended since the round last collected
        size_t holds = 0;        // under m, the scheduler's thread wants the stage's devices (hold)
        HeldTime& held;          // under m, what of those holds was the generated rows' own work
        bool background = false; // under m, the overlap's prompt is a read-ahead job alone, whose every piece waits for kBackgroundGaps services (serve_gaps)
        bool want = false;       // under m, the round has a service of generated rows to form once it is admitted (serve_generated)
        bool granted = false;    // under m, admitted: the thread stands parked at a piece boundary until the round has formed it or withdrawn
        bool want_deferred = false;   // under m, that wish was due and waited for the floor, counted once
        uint64_t grants = 0, services = 0;   // under m, the grants given and the granted passes recorded
        uint64_t recorded = 0, charged = 0, uncharged_at_grant = 0;   // under m, the passes recorded between pieces, those whose time was charged, and the grants given with one still uncharged
        double charged_ms = 0;        // under m, the generated rows' time charged to the floor
        // A service's time, a pass's or held work's, charged to the floor; under m.
        void charge(double ms) {
            floor.service(ms);
            charged_ms += ms;
        }
        ServiceClock& clock;     // under m
        bool demand_ended = false;     // under m, the overlap ended since the thread last looked
        PromptFloor& floor;      // under m
        double& apart_ms;        // under m, the stage's last pass apart's time (StagePace::apart_ms)
        bool parked = false;     // under m, it stands still at a piece boundary for that
        const bool& quit;        // under m, its lane's stop
        StageShare share;        // the recorder thread's
        std::atomic<double> price_ms{0};   // its share.price(), for the round's choice of riding
        Clock::time_point piece;
        StagePace& stage;               // its search, under m
        std::deque<Timing> timing;      // the thread's, slices recorded and not yet retired on the stage
        backend::Ticket last = 0;       // the thread's, the stage's submission of the last pass it recorded
    };

    // The timed slices whose work has retired on stage s, each given to the stage's search as its slice's piece; under r.m, and true where one was.
    bool settle_timing(Recorder& r, size_t s) {
        if (r.timing.empty()) return false;
        backend::Backend& b = model_.stage_backend(s);
        bool settled = false;
        while (!r.timing.empty()) {
            Recorder::Timing& t = r.timing.front();
            if (!t.started && (!t.before || b.done(t.before))) {
                t.started = true;
                t.from = Clock::now();
            }
            if (!t.started || !b.done(t.own)) break;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t.from).count();
            StagePace& p = r.stage;
            if (p.piece_of != t.prompt_of) p.search = PieceSearch{};
            p.search.add(t.rows, ms, p.apart_ms);
            p.piece_of = t.prompt_of;
            p.piece_end = t.prompt_end;
            r.timing.pop_front();
            settled = true;
        }
        return settled;
    }

    // One recording of the pass, or the pass apart, posted to stage s, on its lane's thread with the stage's lock held (infer::StageLanes).
    void record(Recorder& r, size_t s, std::unique_lock<std::mutex>& lk) {
        {
            // A pass with prompt rows first: the pass apart is taken between its pieces.
            Recorder::Post& post = r.pass.posted ? r.pass : r.apart;
            const Recorder::Post taken = post;
            // A pass apart recorded alone is a service the clock counts and the floor does not: no prompt's pass holds the stage.
            if (auto_ && taken.serves) r.clock.began(now_ms());
            lk.unlock();
            Recorder::End end{taken.slot, taken.job, false, {}};
            try {
                if (taken.paced) {
                    infer::Pace pace;
                    // Under the automatic rule no piece is held ahead, so a pass taken at a boundary runs on an idle device.
                    pace.ahead = !auto_;
                    // Each piece's time, the passes apart taken between pieces left out (passed restarts the piece), for the next prompt slice's rows.
                    const auto piece_done = [&](int layer, const char* part) {
                        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - r.piece).count();
                        end.longest_ms = std::max(end.longest_ms, ms);
                        end.slice.piece(ms, layer, part);
                    };
                    // Under a fixed share a piece is held ahead, so its time is the one before it on the device, which cuts nothing (XDEV, 2026-10-18 12:02).
                    end.rows = auto_ ? taken.rows : 0;
                    pace.between = [&] {
                        piece_done(pace.layer, pace.ffn ? "ffn" : "mixer");
                        auto_ ? between_auto(r, s, pace) : between(r, s, pace);
                    };
                    // A pass apart recorded with the device idle has retired: its time, and the piece's from here.
                    pace.passed = [&] {
                        const Clock::time_point now = Clock::now();
                        const double ms = std::chrono::duration<double, std::milli>(now - r.piece).count();
                        if (auto_) {
                            std::lock_guard<std::mutex> charge(r.m);
                            r.charge(ms);
                            r.apart_ms = ms;
                            ++r.charged;
                        } else {
                            r.share.generated(ms);
                            r.price_ms.store(r.share.price());
                        }
                        r.piece = now;
                    };
                    r.piece = Clock::now();
                    model_.record_pass_stage(ctx_, taken.slot, s, &pace);
                    // What follows the last layer, recorded and submitted but not waited for, so its device time falls in the next pass's first piece.
                    piece_done(-1, "tail");
                    end.slice.rows = end.rows;
                } else {
                    const backend::Ticket before = r.last;
                    model_.record_pass_stage(ctx_, taken.slot, s);
                    if (taken.timed) r.timing.push_back({before, model_.stage_ticket(ctx_, taken.slot), taken.rows, taken.prompt_of, taken.prompt_end, false, {}});
                }
                r.last = model_.stage_ticket(ctx_, taken.slot);
            } catch (const std::exception& e) {
                end.failed = true;
                end.error = e.what();
            }
            lk.lock();
            post.posted = false;
            r.ends.push_back(std::move(end));
            r.cv.notify_all();
            lk.unlock();
            wake();
            lk.lock();
        }
    }

    // A piece boundary of the pass a recorder holds, on its thread: the stage's time is counted, and a pass apart that waits there is recorded if its rows have arrived, asked without waiting, and the generated rows are due the stage.
    // First it stands still while the scheduler's thread works on the stage's devices (hold).
    void between(Recorder& r, size_t s, infer::Pace& pace) {
        const double piece_ms = std::chrono::duration<double, std::milli>(Clock::now() - r.piece).count();
        r.share.piece(piece_ms);
        const bool idle = !pace.ahead;
        Recorder::Post taken;
        bool held = false;
        const Clock::time_point parked = Clock::now();
        {
            std::unique_lock<std::mutex> lk(r.m);
            if (r.holds) {
                held = true;
                r.parked = true;
                r.cv.notify_all();
                r.cv.wait(lk, [&] { return !r.holds; });
                r.parked = false;
            }
            taken = r.apart;
        }
        r.piece = Clock::now();
        if (held) r.share.held(std::chrono::duration<double, std::milli>(r.piece - parked).count());
        // The device holds a piece ahead only while no pass apart waits, so one that waits for its share runs at once and is measured.
        pace.ahead = r.share.ahead() && !taken.posted;
        if (!taken.posted || !r.share.due() || !model_.stage_ready(ctx_, taken.slot, s)) return;
        Recorder::End end{taken.slot, taken.job, false, {}};
        try {
            model_.record_pass_within(ctx_, taken.slot, s);
        } catch (const std::exception& e) {
            end.failed = true;
            end.error = e.what();
        }
        // With the device idle the pass runs at once and is timed by its own wait; behind a piece ahead it is not seen.
        if (idle && !end.failed) pace.within = model_.stage_ticket(ctx_, taken.slot);
        else r.share.unmeasured(piece_ms);
        ++between_;
        {
            std::lock_guard<std::mutex> lk(r.m);
            r.apart.posted = false;
            r.ends.push_back(std::move(end));
        }
        r.cv.notify_all();
        wake();
    }

    // A piece boundary under the automatic rule: the piece pays the floor, and a pass apart that waits there is recorded once a service is due by the clock, nothing is owed and its rows have arrived, asked without waiting.
    // No piece is held ahead (Pace::ahead), so every such pass runs on an idle device and is timed whole by its own wait.
    void between_auto(Recorder& r, size_t s, infer::Pace& pace) {
        const double piece_ms = std::chrono::duration<double, std::milli>(Clock::now() - r.piece).count();
        Recorder::Post taken;
        bool admitted = false;
        {
            // One critical section from the holds to the grant, so a hold that arrives is drained and charged before the service is decided, and never finds the thread parked without being told.
            std::unique_lock<std::mutex> lk(r.m);
            r.floor.piece(piece_ms);
            r.clock.longest_piece_ms = std::max(r.clock.longest_piece_ms, piece_ms);
            if (r.demand_ended) r.floor.demand_ended();
            r.demand_ended = false;
            const auto stand = [&](bool granted) {
                r.parked = true;
                r.held.parked(now_ms());
                r.cv.notify_all();
                if (granted) {
                    // The round is told with the recorder's lock released, and answers by forming the service or withdrawing.
                    lk.unlock();
                    wake();
                    lk.lock();
                }
                r.cv.wait(lk, [&] { return !r.granted && !r.holds; });
                r.parked = false;
                r.held.resumed();
                // The scheduler's work for the generated rows under those cache_.holds(a chain, a mark, a retract) is a service of theirs, charged once; its other work is neither side's time.
                r.charge(r.held.take());
            };
            if (r.holds) stand(false);
            if (r.background) {
                serve_gaps(r, s, pace, lk, stand);
                r.piece = Clock::now();
                return;
            }
            taken = r.apart;
            // A service the round wants is admitted here once, from the floor as every earlier charge left it; the thread stands parked while the round runs the chain and forms the pass, then records it with no second test.
            if (!taken.posted && r.want) {
                const double now = now_ms();
                if (r.clock.due(now, piece_ms)) {
                    if (!r.floor.clear()) {
                        r.clock.defer(r.want_deferred);
                    } else {
                        r.want_deferred = false;
                        admitted = grant_service(r, stand);
                        taken = r.apart;
                    }
                }
            } else if (taken.posted && model_.stage_ready(ctx_, taken.slot, s)) {
                // A pass formed with no prompt's pass fed, which this pass then met: gated by the clock and the floor as it waits here.
                const double now = now_ms();
                if (r.clock.due(now, piece_ms)) {
                    if (!r.floor.clear()) r.clock.defer(r.apart.deferred);
                    else {
                        admitted = true;
                        r.clock.began(now);
                    }
                }
            }
        }
        r.piece = Clock::now();
        if (admitted) record_service(r, s, pace, taken);
    }

    // A service at a piece boundary granted: the round forms it while the thread stands parked (stand), and stamped its start as it took the grant up (serve_generated); whether a pass was formed, a grant that formed nothing beginning nothing.
    // The one owner of a boundary's grant, for between_auto and serve_gaps; under the recorder's lock.
    template <class Stand>
    bool grant_service(Recorder& r, Stand& stand) {
        r.granted = true;
        ++r.grants;
        if (r.charged != r.recorded) ++r.uncharged_at_grant;
        stand(true);
        if (!r.apart.posted) return false;
        ++r.services;
        return true;
    }
    // A pass of generated rows recorded within a piece boundary, its ticket the next piece's wait, and its end handed to the round: the one owner of a boundary's recording, for between_auto and serve_gaps; outside the recorder's lock.
    void record_service(Recorder& r, size_t s, infer::Pace& pace, const Recorder::Post& taken) {
        Recorder::End end{taken.slot, taken.job, false, {}};
        try {
            model_.record_pass_within(ctx_, taken.slot, s);
        } catch (const std::exception& e) {
            end.failed = true;
            end.error = e.what();
        }
        if (!end.failed) pace.within = model_.stage_ticket(ctx_, taken.slot);
        ++between_;
        {
            std::lock_guard<std::mutex> lk(r.m);
            if (!end.failed) ++r.recorded;
            r.apart.posted = false;
            r.ends.push_back(std::move(end));
        }
        r.cv.notify_all();
        wake();
    }

    // Whether a request of a client is generating, its prompt read.
    static bool replying(const std::vector<std::shared_ptr<Request>>& active) {
        return std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return !r->job_ && !reads_prompt(r->gen_.size(), r->seq_.length(), history_tokens(*r)); });
    }
    // How a turn-end write beyond the host tier may go this round (Cache::flush_one): whole while nothing runs or waits, a piece's time at a time beside prompts alone, and beside replies a piece only at background work's turn.
    // That turn comes after kBackgroundGaps reply-only gaps, and where a read-ahead job also waits the kind that went last yields, so the two take turns.
    Cache::WriteTurn write_turn(const std::vector<std::shared_ptr<Request>>& active) const {
        if (!requests(active) && queue_.empty() && paused_.empty()) return Cache::WriteTurn::whole;
        if (!replying(active)) return Cache::WriteTurn::piece;
        const bool jobs_wait = std::any_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& r) { return r->job_ && working(*r); });
        return reply_gaps_ >= kBackgroundGaps && !(jobs_wait && background_last_ == Background::write) ? Cache::WriteTurn::due : Cache::WriteTurn::none;
    }
    // At a read-ahead job's piece boundary beside replies, on its stage's thread under its lock: kBackgroundGaps services before the job's next piece, each granted and recorded as between_auto's, with no clock or floor (F2DEV, 2026-10-20 08:29).
    // The thread stands parked while it waits, so a hold or the end of the replies finds it; the last service's ticket is the next piece's wait.
    template <class Stand>
    void serve_gaps(Recorder& r, size_t s, infer::Pace& pace, std::unique_lock<std::mutex>& lk, Stand& stand) {
        for (size_t gaps = 0; r.background && !r.quit && gaps < kBackgroundGaps;) {
            if (r.holds) {
                stand(false);
                continue;
            }
            if (!r.apart.posted && !r.want) {
                r.parked = true;
                r.held.parked(now_ms());
                r.cv.notify_all();
                r.cv.wait(lk, [&] { return r.quit || r.holds || !r.background || r.want || r.apart.posted; });
                r.parked = false;
                r.held.resumed();
                r.charge(r.held.take());
                continue;
            }
            if (!r.apart.posted && !grant_service(r, stand)) continue;
            const Recorder::Post taken = r.apart;
            if (!model_.stage_ready(ctx_, taken.slot, s)) break;
            lk.unlock();
            record_service(r, s, pace, taken);
            lk.lock();
            ++gaps;
        }
    }

    // A recording's end told to the round without waiting for the scheduler's lock, which the scheduler may hold while it waits for this very recorder (quiet).
    // The round's waits read the ends themselves; the lock is passed through, by a spin that yields, so an end is not put between a wait's look and its sleep.
    void wake() {
        while (!quieting_.load()) {
            if (m_.try_lock()) {
                m_.unlock();
                break;
            }
            std::this_thread::yield();
        }
        cv_.notify_all();
    }

    // Whether a recorder has ended a recording the round has not collected; under the lock.
    bool ended() const {
        for (const auto& rec : recorders_) {
            if (!rec) continue;
            std::lock_guard<std::mutex> lk(rec->m);
            if (!rec->ends.empty()) return true;
        }
        return false;
    }

    // Whether a pass of that kind is being recorded at stage s, or any pass at any stage.
    bool recording(size_t s, bool apart) const {
        return std::any_of(slots_.begin(), slots_.end(), [&](const Slot& k) { return k.live && k.recording && k.ran == s && k.apart == apart; });
    }
    // Whether a pass of that kind waits for the first stage's recorder or is being recorded there, so that no other is formed behind it.
    bool first_stage_held(bool apart) const {
        return recorders_[0] && std::any_of(slots_.begin(), slots_.end(), [&](const Slot& k) { return k.live && k.ran == 0 && k.apart == apart; });
    }
    bool recording() const {
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live && k.recording; });
    }
    // Whether the round has a pass to advance or retire: one in flight that no recorder holds.
    bool acting() const {
        return std::any_of(slots_.begin(), slots_.end(), [](const Slot& k) { return k.live && !k.recording; });
    }

    // What the recorders finished since the last round: a recorded stage counts, and a failed one is undone here and fails its pass alone.
    void collect(std::vector<std::shared_ptr<Request>>& active) {
        for (size_t s = 0; s < recorders_.size(); ++s) {
            auto& rec = recorders_[s];
            if (!rec) continue;
            std::vector<Recorder::End> ends;
            {
                std::lock_guard<std::mutex> lk(rec->m);
                ends.swap(rec->ends);
            }
            for (const Recorder::End& e : ends) {
                // The end of a recording whose pass was abandoned meanwhile (abort_all) is nobody's.
                if (!slots_[e.slot].live || !slots_[e.slot].recording || slots_[e.slot].job != e.job) continue;
                acted_ = true;
                slots_[e.slot].recording = false;
                if (!e.failed) {
                    model_.commit_pass_stage(ctx_, e.slot);
                    // A timed slice's submission at the first stage, before whose retirement its prompt's next slice is not formed.
                    if (s == 0 && slots_[e.slot].timed)
                        for (const auto& m : slots_[e.slot].members)
                            if (m->admission_ == slots_[e.slot].prompt_of) m->slice_ticket_ = model_.stage_ticket(ctx_, e.slot);
                    ++slots_[e.slot].ran;
                    if (e.rows) {
                        std::lock_guard<std::mutex> lk(pace_[s]->m);
                        const Slot& f = slots_[e.slot];
                        if (take_piece(pace_[s]->search, pace_[s]->piece_of, pace_[s]->piece_end, f.short_pass, f.prompt_of, f.prompt_from, f.prompt_end, e.rows, e.longest_ms, pace_[s]->apart_ms))
                            pace_[s]->slice = e.slice;
                    }
                    continue;
                }
                const std::vector<size_t> taken = abandon(e.slot);
                fail_pass(active, e.slot, e.error, taken);
            }
        }
    }

    // Every recorder idle: before passes are abandoned, which drains every device and returns histories.
    void quiet() {
        ++quieting_;
        for (auto& rec : recorders_) {
            if (!rec) continue;
            std::unique_lock<std::mutex> lk(rec->m);
            // A stage standing parked on a grant is let go first: nothing is formed for it any more.
            rec->want = rec->granted = false;
            rec->cv.notify_all();
            rec->cv.wait(lk, [&] { return !rec->pass.posted && !rec->apart.posted; });
        }
        --quieting_;
    }
    // Every recorder idle or standing still at a piece boundary of the pass it feeds, until release(): the scheduler's thread then works on the stages' devices between two calls of that pass, as a pass apart is recorded there.
    // `generated` says the work is the generated rows' own, which the automatic rule charges them (PromptFloor).
    void hold(bool generated) {
        ++quieting_;
        for (auto& rec : recorders_) {
            if (!rec) continue;
            std::unique_lock<std::mutex> lk(rec->m);
            ++rec->holds;
            rec->held.begin(generated, now_ms());
            rec->cv.wait(lk, [&] { return rec->parked || (!rec->pass.posted && !rec->apart.posted); });
        }
    }
    void release() {
        for (auto& rec : recorders_) {
            if (!rec) continue;
            {
                std::lock_guard<std::mutex> lk(rec->m);
                --rec->holds;
                rec->held.end(now_ms());
            }
            rec->cv.notify_all();
        }
        --quieting_;
    }
    // The model for work on a stage's devices that is not a pass's stage (a history copied to or from host memory, a mark, a retract, a kept state, a drafter's chain): held for as long as the value lives, the expression it is called in.
    struct Held {
        Scheduler& s;
        explicit Held(Scheduler& sched, bool generated = false) : s(sched) { s.hold(generated); }
        Held(const Held&) = delete;
        Held& operator=(const Held&) = delete;
        ~Held() { s.release(); }
        infer::Model* operator->() const { return &s.model_; }
    };
    Held quieted(bool generated = false) { return Held(*this, generated); }

    // A failed pass, which the model has abandoned, returned its requests' histories to where it found them: they end with the error and give their blocks back, while the other passes in flight go on.
    // A continued prompt's later passes go with it: those the model abandoned with it (`taken`, abort_pass's), and those it could not, a stage having failed, abandoned here.
    void fail_pass(std::vector<std::shared_ptr<Request>>& active, size_t k, const std::string& what, const std::vector<size_t>& taken = {}) {
        const std::vector<std::shared_ptr<Request>> members = slots_[k].members;
        vacate(k);
        std::vector<size_t> later;
        for (size_t j = 0; j < slots_.size(); ++j)
            if (slots_[j].live && std::any_of(slots_[j].members.begin(), slots_[j].members.end(), [&](const std::shared_ptr<Request>& m) {
                    return std::find(members.begin(), members.end(), m) != members.end();
                }))
                later.push_back(j);
        std::sort(later.begin(), later.end(), [&](size_t a, size_t b) { return slots_[a].formed < slots_[b].formed; });
        if (!later.empty() && std::find(taken.begin(), taken.end(), later.front()) == taken.end()) {
            const std::vector<size_t> rest = abandon(later.front());
            for (size_t j : later)
                if (j != later.front() && std::find(rest.begin(), rest.end(), j) == rest.end()) throw std::logic_error("server: a continued prompt's later pass not abandoned with the one before it");
        }
        for (size_t j : later) {
            slots_[j].recording = false;
            vacate(j);
        }
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

    // The pass in slot k abandoned in the model, with every recorder idle first: abandoning a pass drains every device, and a device a recorder is recording on takes no call from this thread but a wait on a returned ticket.
    // So a failed pass waits for the stages being recorded, and leaves their passes as they are.
    std::vector<size_t> abandon(size_t k) {
        quiet();
        return model_.abort_pass(ctx_, k);
    }

    // A job whose slices follow each other in flight is never out of flight at a boundary, so to give way its passes, each holding it alone, are abandoned and it leaves flight now.
    void recall(const Request& j) {
        std::vector<size_t> held;
        for (size_t k = 0; k < slots_.size(); ++k)
            if (slots_[k].live && std::any_of(slots_[k].members.begin(), slots_[k].members.end(), [&](const std::shared_ptr<Request>& m) { return m.get() == &j; })) held.push_back(k);
        if (std::none_of(held.begin(), held.end(), [&](size_t k) { return slots_[k].continues; }) || std::any_of(held.begin(), held.end(), [&](size_t k) { return slots_[k].members.size() != 1; })) return;
        std::sort(held.begin(), held.end(), [&](size_t a, size_t b) { return slots_[a].formed < slots_[b].formed; });
        abandon(held.front());
        for (size_t k : held) vacate(k);
    }

    // Every pass in flight abandoned, each request's history back where its pass found it.
    // A continued prompt's later passes go with its earliest, so the passes are taken in formation order.
    void abort_all() {
        quiet();
        std::vector<size_t> order;
        for (size_t k = 0; k < slots_.size(); ++k)
            if (slots_[k].live) order.push_back(k);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return slots_[a].formed < slots_[b].formed; });
        std::vector<size_t> taken;
        for (size_t k : order) {
            if (std::find(taken.begin(), taken.end(), k) == taken.end()) {
                const std::vector<size_t> later = abandon(k);
                taken.insert(taken.end(), later.begin(), later.end());
            }
            slots_[k].recording = false;
            vacate(k);
        }
    }

    // Whether r, in flight, may continue its prompt or its job's ids in another pass of its own, as Model::begin_pass allows, after a slice formed in an earlier round (docs/MULTI-DEVICE.md, phase 4).
    // Not while a request could not grow, nor a cancelled request, nor a job while a request waits for its seat or room; beside generated rows a continuation leaves a slot free.
    bool continues(const Request& r, bool generating) const {
        if (!r.seq_.in_flight() || stalled_now_ || r.cancel_.load() || decoding(r) || (r.job_ && jobs_yield_)) return false;
        size_t free = 0;
        for (size_t k = 0; k < normal_; ++k) {
            const Slot& s = slots_[k];
            free += !s.live;
            if (!s.live) continue;
            const auto is = [&](const std::shared_ptr<Request>& m) { return m.get() == &r; };
            if (std::none_of(s.members.begin(), s.members.end(), is)) continue;
            if (s.members.size() != 1 || !s.ran || !s.wanting.empty() || s.formed > formed_before_) return false;
        }
        return free > (generating ? 1u : 0u);
    }

    // Whether r's cache lacks only its last generated token, which a decode entry reads.
    static bool decoding(const Request& r) { return !r.gen_.empty() && r.seq_.length() + 1 == history_tokens(r); }
    // Whether r has work for a pass: a request always, a job while its cache lacks some of its ids.
    static bool working(const Request& r) { return !r.job_ || r.seq_.length() < r.prompt_.size(); }

    // Before a pass, every uncapped decoding request not in flight whose next token would pass its reservation takes another step, the earliest admitted first, with what make_room gives it.
    // A request make_room cannot give it to, or whose plan would pause a request in flight, sits the pass out with its cache as it is (a stall) and asks again before the next; it is never paused for its own growth.
    void grow(std::vector<std::shared_ptr<Request>>& active, bool again = false) {
        // `again` is a round's second call, once the jobs gave way: only the stalled ask, and a stall is counted after the last call that could lift it.
        const bool last = again || std::none_of(active.begin(), active.end(), [](const std::shared_ptr<Request>& j) { return j->job_ && !j->seq_.in_flight(); });
        for (size_t i = 0; i < active.size(); ++i) {
            Request& r = *active[i];
            if (again && !r.stalled_) continue;
            r.stalled_ = false;
            if (r.seq_.in_flight()) continue;
            const std::vector<size_t> step = kGrowth.step(pools_, r.params_.until_limit, decoding(r), r.seq_.length(), r.need_);
            if (step.empty()) continue;
            if (yields(active, r)) {
                turn_to_ = waiting()->admission_;
                turned_ = Clock::now();
                pause(active, i--, true);
                continue;
            }
            const Taken t = make_room(cache_.room(active), cache_.ranked(), npos, false, holders(active), r.admission_, true, step, cache_.gain_of(active));
            if (!t.enough || t.wait) {
                r.stalled_ = true;
                if (!last) continue;
                ++r.stalls_;
                std::lock_guard<std::mutex> lk(m_);
                ++stalls_;
                waits_ += t.wait;
                continue;
            }
            take(t, active);
            r.edge_steps_ = waiting() ? r.edge_steps_ + 1 : 0;
            r.stepped_ = Clock::now();
            std::lock_guard<std::mutex> lk(m_);
            add(r.need_, step);
        }
    }

    // What the empty-pass error says of the request left without a pass: its history, whether it could not grow, how many requests are active, and each pool's free blocks.
    std::string left_out(const std::vector<std::shared_ptr<Request>>& active, const Request& r, size_t requests) const {
        std::string s = "a request of " + std::to_string(r.seq_.length()) + " tokens" + (r.stalled_ ? " could not grow" : " was not ready") + ", " + std::to_string(requests) + " active, blocks free";
        const std::vector<size_t> free = cache_.room(active);
        for (size_t p = 0; p < free.size(); ++p) s += " " + std::to_string(free[p]) + " of " + std::to_string(pools_.blocks[p]);
        return s;
    }

    // The uncapped request paused longest, which a turn at the pool's edge goes to.
    const Request* waiting() const {
        const auto p = std::find_if(paused_.begin(), paused_.end(), [](const std::shared_ptr<Request>& q) { return q->params_.until_limit && !q->cancel_.load(); });
        return p == paused_.end() ? nullptr : p->get();
    }
    // Whether `r`, whose growth step is due, gives the pool's edge up (Growth::yields): its turn is over, its history can be kept whole, and the one waiting fits once `r` holds nothing.
    // Kept whole means the copy its history would leave is one write_back keeps (host_room), and on a model that keeps a state a free checkpoint slot; a turn kept for want of host memory is counted and said.
    bool yields(const std::vector<std::shared_ptr<Request>>& active, const Request& r) {
        const Request* w = waiting();
        if (!w || !host_cap_) {
            end_memory_run();
            return false;
        }
        if (!kGrowth.yields(r.edge_steps_, turn_steps ? turn_steps : kGrowth.turn(swap_ms_, ms_since(r.stepped_)), true, !model_.keeps_state() || model_.checkpoints_free())) return false;
        std::vector<size_t> free = cache_.room(active);
        add(free, cache_.promised(r));
        add(free, model_.freed({&r.seq_}).blocks);
        if (!make_room(free, cache_.ranked(), npos, false, {}, 0, false, pools_.blocks_for(kGrowth.entry(history_tokens(*w), true, 0, w->gen_.size())), cache_.gain_of({}, npos, 0, false, {&r.seq_})).enough)
            return false;
        const size_t bt = model_.kv_block_tokens(), n = r.seq_.length() / bt * bt;
        const Cache::HostRoom::Why why = cache_.host_room(n, Cache::Rank{2, std::filesystem::file_time_type::clock::now()}).why;
        if (why == Cache::HostRoom::Why::memory) {
            ++turns_without_memory_;
            if (!memory_run_++) {
                memory_since_ = Clock::now();
                std::fprintf(stderr, "server: a turn at the pool's edge kept: the host has no memory for the copy of %zu tokens it would leave\n", n);
            }
            return false;
        }
        if (why != Cache::HostRoom::Why::kept) return false;
        end_memory_run();
        return true;
    }
    // A run of turns kept for want of host memory ends at a turn given up or where none waits: its count and length are said, and the next run's first is said again.
    void end_memory_run() {
        if (memory_run_) std::fprintf(stderr, "server: %zu turns at the pool's edge kept for want of host memory over %.1f s\n", memory_run_, ms_since(memory_since_) / 1000.0);
        memory_run_ = 0;
    }


    // Takes what make_room chose for a growing request: the donors, then the requests it pauses, latest admitted first, and the donor a paused request's history became where make_room took that too.
    void take(const Taken& t, std::vector<std::shared_ptr<Request>>& active) {
        cache_.take(t);
        std::vector<std::pair<std::shared_ptr<Request>, bool>> victims;
        for (const auto& p : t.paused) victims.push_back({active[p.first], p.second});
        for (const auto& v : victims) {
            const size_t at = (size_t)(std::find(active.begin(), active.end(), v.first) - active.begin());
            const uint64_t donor = pause(active, at);
            if (donor && v.second) cache_.evict(donor);
        }
    }


    // A paused request's history goes to a donor, even short of a full block, which it takes back whole on resuming unless something evicted it, and the request to the paused requests in order of first admission; the donor it left, or 0.
    uint64_t pause(std::vector<std::shared_ptr<Request>>& active, size_t i, bool yielded = false) {
        auto r = active[i];
        r->paused_as_ = r->donor_ = cache_.park(active, lines(), i, history(*r), 1);
        active_count_.store(requests(active));
        ++r->pauses_;
        r->stalled_ = false;
        yields_ += yielded;
        std::lock_guard<std::mutex> lk(m_);
        ++pauses_;
        r->line_ = yielded ? ++lines_ : 0;
        paused_.insert(std::upper_bound(paused_.begin(), paused_.end(), r, [](const auto& a, const auto& b) { return std::make_pair(a->line_, a->admission_) < std::make_pair(b->line_, b->admission_); }), r);
        paused_count_.store(paused_.size());
        return r->donor_;
    }

    static constexpr size_t npos = std::numeric_limits<size_t>::max();
    // A request in flight is counted by the history its earliest pass found, since its storages disagree until its passes end.
    std::vector<Holder> holders(const std::vector<std::shared_ptr<Request>>& active) const {
        std::vector<Holder> h;
        for (const auto& r : active) {
            size_t len = r->seq_.length();
            for (const Slot& s : slots_)
                for (size_t e = 0; s.live && e < s.members.size(); ++e)
                    if (s.members[e] == r) len = std::min(len, s.from[e]);
            h.push_back(Holder{r->admission_, r->params_.until_limit, len ? pools_.blocks_for(len) : std::vector<size_t>(pools_.blocks.size(), 0), r->seq_.in_flight()});
        }
        return h;
    }


    // A waiting request whose client left ends where it waits, and a paused one's own donor goes with it when it holds less than a block, which no fork can share; the position after it.
    // Under the lock.
    std::deque<std::shared_ptr<Request>>::iterator leave(std::deque<std::shared_ptr<Request>>& waiting, std::deque<std::shared_ptr<Request>>::iterator it) {
        cache_.leave(lines(), **it);
        (*it)->proof_ = 0;
        (*it)->end("cancel");
        return waiting.erase(it);
    }


    // A pass of prompt rows measures the prompt rate a read's bound takes, a moving mean over passes of 64 rows or more.
    void measure_prompt(const Slot& f, double since_retired) {
        size_t rows = 0;
        for (size_t n : f.rows) rows += n;
        if (rows < kMeasuredRows) return;
        const double per = std::min(ms_since(f.begun), since_retired) / (double)rows;
        prompt_ms_row_ = forget(prompt_ms_row_, per);
        cache_.prompt_measured(prompt_ms_row_);
    }


    // On a sampling thread: the tokens drawn from d's rows with its request's own settings, history and generator, in order, as infer::accept takes a verify's rows, a decode entry's one row being a verify of no drafts.
    // Each token goes into the history the next row is drawn after, an end token, a stop text or the length ends the request, and with logprobs asked each token's row is copied or its values computed; it touches only d and its request's state.
    void draw(Draw& d) {
        Request& r = *d.r;
        const size_t w = ctx_.width, k = r.verify_.empty() ? 0 : r.verify_.size() - 1;
        size_t i = 0;
        const auto pick = [&](uint32_t id) {
            const float* row = d.row + i++ * w;
            if (tok_.is_eos(id)) { r.finish_pending_ = "eos"; return false; }
            Request::Token t;
            t.id = id;
            if (r.params_.logprobs) {
                if (d.copies) {
                    // A row the reader has finished with is filled first.
                    --d.copies;
                    r.logits_.assign(row, row + w);
                    t.row = std::move(r.logits_);
                    r.logits_ = std::vector<float>();
                } else {
                    // A reader that has fallen behind gets the values the draw computes, the same the reader would.
                    Request::fill(t, row, w, r.params_.top_logprobs);
                }
            }
            r.gen_.push_back(id);
            d.picks.push_back(std::move(t));
            if (!r.params_.stop.empty()) {
                r.decoded_ += tok_.decode({id});
                for (const auto& s : r.params_.stop)
                    if (!s.empty() && r.decoded_.find(s) != std::string::npos) { r.finish_pending_ = "stop"; return false; }
            }
            if ((int)r.gen_.size() >= r.params_.max_tokens) { r.finish_pending_ = "length"; return false; }
            return true;
        };
        d.sampled = infer::accept(d.row, w, k ? r.verify_.data() + 1 : nullptr, k, r.params_, tok_.eos_id, r.gen_, r.rng_, pick).rows;
    }

    // A draw's tokens on the scheduler thread, pushed to the channel in order, the last its next decode feeds; a request that ends is finished after the pass, once every entry's logits have been read.
    void step(Draw& d) {
        Request& r = *d.r;
        for (Request::Token& t : d.picks) {
            r.last_id_ = t.id;
            // A row a token goes with comes back, once the reader has finished with another, for a later pass to fill.
            const bool row = !t.row.empty();
            r.push(std::move(t), row ? &r.logits_ : nullptr);
            ++published_total_;
        }
        if (!proposer_) return;
        if (r.verify_.empty()) r.record_.stepped();
        else r.record_.verified(d.sampled - 1, r.verify_.size() - 1);
    }

    // A finished request becomes a donor through park, and one line on stderr records it.
    void finish(std::vector<std::shared_ptr<Request>>& active, size_t i, const std::string& why,
                const std::string& err = "") {
        auto r = active[i];
        r->finished_ = true;
        cache_.drop_encode(*r);
        if (r->job_) {
            // A failed job leaves the donor it forked and frees what it held.
            active.erase(active.begin() + (std::ptrdiff_t)i);
            cache_.release(*r, active);
            r->seq_ = infer::Sequence{};
            return;
        }
        if (why == "error") {
            active.erase(active.begin() + (std::ptrdiff_t)i);
            cache_.release(*r, active);
            r->seq_ = infer::Sequence{};
            active_count_.store(requests(active));
        } else {
            r->parked_ = cache_.park(active, lines(), i, history(*r));
            active_count_.store(requests(active));
        }
        r->end(why, err);
        ++finished_;
        prompt_total_ += r->prompt_tokens();
        generated_total_ += r->gen_.size();
        const Request::Timings t = r->timings();
        char paused[128] = "", stalled[48] = "", late[64] = "";
        if (r->pauses_)
            std::snprintf(paused, sizeof paused, ", paused %zu time%s, %zu taken back, %zu tokens recomputed", r->pauses_, r->pauses_ == 1 ? "" : "s",
                          r->taken_back_, r->recomputed_);
        if (r->stalls_) std::snprintf(stalled, sizeof stalled, ", sat out %zu pass%s", r->stalls_, r->stalls_ == 1 ? "" : "es");
        if (r->restore_late()) std::snprintf(late, sizeof late, ", a read of %zu tokens past its bound", r->restore_late());
        std::fprintf(stderr, "request: %zu prompt tokens (%zu reused), %zu generated, %.0f ms preparing, %.0f ms queued, %.0f ms to first token, %.1f tok/s, %s%s%s%s%s\n",
                     r->prompt_tokens(), r->reused(), r->gen_.size(), t.prepared_ms, t.queued_ms, t.first_token_ms(),
                     t.predicted_per_second(r->gen_.size()), (err.empty() ? why : why + " (" + err + ")").c_str(), paused, stalled, late, restore_line(r->restore(), t).c_str());
    }

    // What a request's restores cost, for its line: the spans as their parts and remainders, adding to its time to first token, and the figures that overlap them; empty where it neither waited for a read nor restored.
    static std::string restore_line(const Request::Restore& x, const Request::Timings& t) {
        if (!x.restores && !x.waits) return {};
        char s[512];
        std::snprintf(s, sizeof s, ", restored %zu tokens from %s (%.1f MiB, %zu restore%s, %zu wait%s, %zu at the bound, %zu failed): queued %.0f ms of waiting %.0f, holding %.0f, enqueuing %.0f, rest %.0f;"
                      " prompt %.0f ms of first pass %.0f, rest %.0f; beside them read %.0f ms, slabs %.0f ms, copies %.0f ms; after the first token %.0f ms",
                      x.tokens, x.kind, (double)x.bytes / (1 << 20), x.restores, x.restores == 1 ? "" : "s", x.waits, x.waits == 1 ? "" : "s", x.timeouts, x.failed, t.queued_ms, x.wait_ms, x.hold_ms,
                      x.enqueue_ms, t.queued_ms - x.wait_ms - x.hold_ms - x.enqueue_ms, t.prompt_ms, x.first_pass_ms, t.prompt_ms - x.first_pass_ms, x.read_ms, x.slabs_ms, x.copies_ms, x.after_first_ms);
        return s;
    }


    // At a round boundary with m_ held: sampling has joined, publication has finished, and failed requests have left active.
    void snapshot_requests(const std::vector<std::shared_ptr<Request>>& active) {
        live_requested_ = false;
        const size_t total = active_count_.load() + paused_.size() + queue_.size();
        const uint64_t prefilled = prefilled_total_.load(), published = published_total_.load();
        if (!total && !live_.total && live_.prefilled_tokens == prefilled && live_.published_tokens == published) return;
        const auto now = Clock::now();
        live_.count = 0;
        live_.total = total;
        live_.sample_elapsed_ms = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(now - live_started_).count();
        live_.prefilled_tokens = prefilled;
        live_.published_tokens = published;
        const auto append = [&](const Request& r, const char* phase) {
            if (live_.count == LiveRequests::limit) return;
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - r.received_).count();
            live_.items[live_.count++] = {r.id_, r.prefilled_, r.gen_.size(), (uint64_t)std::max<int64_t>(0, elapsed),
                                        r.prompt_.size(), r.reused_.load(), phase};
        };
        for (const auto& r : active) {
            if (live_.count == LiveRequests::limit) break;
            if (r->job_) continue;
            if (Cache::image_stop(*r) > r->seq_.length()) {
                append(*r, r->gen_.empty() ? "prefill" : "generate");
                continue;
            }
            append(*r, "encode");
            RequestView& v = live_.items[live_.count - 1];
            v.image_index = r->media_next_ + 1;
            v.image_count = r->media_.size();
            v.image_rows = r->media_[r->media_next_].rows;
            const Cache::EncodeProgress e = cache_.encode_progress(*r);
            v.pieces_done = e.done;
            v.pieces_total = e.total;
            if (e.total) v.image_elapsed_ms = (uint64_t)std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::milliseconds>(now - e.began).count());
        }
        for (const auto* waiting : {&paused_, &queue_})
            for (const auto& r : *waiting) {
                if (live_.count == LiveRequests::limit) break;
                append(*r, waiting == &paused_ ? "paused" : "queued");
            }
    }


    infer::Model& model_;
    const bpe::Tokenizer& tok_;
    size_t max_seqs_, ubatch_, max_queue_;
    const size_t limit_;                  // the tokens a request may hold (request_limit)
    const bool timed_;
    const size_t host_cap_;               // the host tier's bytes at most
    const double share_;                  // of each device's time, what generating requests keep while prompts are read; 0 for none
    const bool auto_;                     // generated rows are served by the automatic rule (Pacing)
    const bool interleave_;               // or, on a pipelined split, by the interleaved rule (Pacing::interleaved)
    uint64_t interleaved_replies_ = 0, interleaved_slices_ = 0, interleaved_mixed_ = 0;   // the scheduler thread's: Stats::interleaved_*
    const Clock::duration wait_bound_;    // the bound of a wait for a restore or a save while nothing better is measured: kWaitBound, or DiskOptions::wait in tests
    const std::optional<std::chrono::milliseconds> read_wait_;   // DiskOptions::read_wait
    const std::optional<double> recompute_row_ms_;   // DiskOptions::recompute_row_ms
    bool jobs_yield_ = false;             // this round a request waits for a job's seat or room, or could not grow, so no job continues (continues)
    bool stalled_now_ = false;            // this round a request could not grow, so nothing continues and the request its plan waits on leaves flight
    uint64_t formed_before_ = 0;          // the passes formed before this round's steps: a deferred host stage is for one of them, and a continuation follows one of them
    bool short_ahead_ = false;            // a short prompt went ahead of a long one in a pass of its own this round (short_prompts), which happens once a round
    std::atomic<bool> overlap_{false};    // written by the scheduler thread: replies and prompts both wait (demand)
    const Clock::time_point epoch_ = Clock::now();   // the stages' clocks count from here
    std::vector<char> plain_;             // per stage, whether it has a thread for passes that are neither fed in pieces nor apart
    infer::spec::Proposer* const proposer_;   // none drafts nothing
    const size_t draft_max_;
    const size_t draft_rows_;             // the draft rows a pass reserves (draft_rows)
    const bool priced_;
    std::vector<infer::spec::Proposer::Ask> asks_;   // the drafts being proposed, the first of them in use
    std::vector<Request*> askers_;        // the requests that may draft in the pass being formed
    std::vector<uint8_t> chained_;        // per such request, whether the proposer drafts for it, not its own history
    std::vector<size_t> asked_;           // per such request, its place among the proposer's asks
    std::vector<std::vector<double>> keeps_;   // per request that may draft, the chance it keeps each draft
    std::vector<size_t> takes_;           // per request that may draft, the drafts the price gives it
    infer::spec::PassTimes times_;        // the passes' measured cost, the scheduler thread's
    struct Retirement {
        Clock::time_point at;             // when the last pass of a kind retired, if one has
        bool any = false;
        double chain_ms = 0;              // the chains drafted since then
    };
    Retirement retirement_[2];            // of the passes, and of the passes apart
    std::atomic<size_t> draft_failures_{0};   // Stats::draft_failures
    infer::spec::Record tally_;           // under the lock, every verify's drafts by position, a source apart
    Timing round_;                        // the scheduler thread's, published to timing_ each round
    Timing timing_;                       // under the lock
    std::vector<double> host_stage_ms_ = std::vector<double>(model_.stage_count(), 0.0);   // per stage on the host, its time since the last reading; a stage's first pass adds to it before any round has ended
    Clock::time_point span_start_;
    size_t span_rows_ = 0;                // rows the passes retired since the last reading carried
    double formed_stages_ms_ = 0;         // the round's first stages of new passes, which forming them does not count
    double formed_chains_ms_ = 0;         // and its drafters' chains, counted as drafting
    infer::ExecContext ctx_;
    std::vector<Slot> slots_;
    LogitRows logit_rows_;
    const Pools pools_;                        // the model's cache pools, which never change size
    uint64_t formed_ = 0;                      // passes formed, which orders them
    std::vector<infer::BatchEntry> entries_;   // the pass being formed
    core::JobThreads samplers_;
    std::vector<Draw> draws_;                  // the retiring pass's rows the pool draws
    mutable std::mutex m_;
    const Clock::time_point live_started_ = Clock::now();
    uint64_t request_ids_ = 0;
    LiveRequests live_;   // last requested boundary snapshot, under m_
    mutable bool live_requested_ = false;   // polls coalesce; analytics never wakes the scheduler
    std::condition_variable cv_;
    std::deque<std::shared_ptr<Request>> queue_;
    // Paused requests in order of first admission, the scheduler thread's; they hold nothing but a donor, and do not count against max_queue.
    std::deque<std::shared_ptr<Request>> paused_;
    size_t steady_from_ = 0;                      // the least extent from which rows are one class up to the limit
    std::atomic<size_t> active_count_{0}, paused_count_{0}, in_flight_{0}, checkpoints_{0};
    std::atomic<uint64_t> finished_{0}, prompt_total_{0}, generated_total_{0};   // what Stats gives as the totals since start
    std::atomic<uint64_t> prefilled_total_{0}, published_total_{0};
    std::atomic<uint64_t> rode_{0};   // Stats::rode
    RideRun ride_run_;    // the prompt passes the generated rows have ridden in a row
    std::atomic<uint64_t> apart_formed_{0}, between_{0};                         // Stats::apart and Stats::between, the second counted by the stages' threads
    std::atomic<uint64_t> prompt_passes_{0};                                     // Stats::prompt_passes
    std::atomic<uint64_t> continued_{0};                                         // Stats::continued
    std::atomic<uint64_t> short_passes_{0};                                      // Stats::short_passes
    uint64_t lines_ = 0;        // the numbers of Request::line_, under the lock
    // What a change of turn at the pool's edge takes (Growth::change over the last two timed, changed_ms_ the last), each from the yield (turned_) to the first pass retired that carries the request the turn went to.
    // That request is held by its first admission's number (turn_to_), given once and to no later request: met, it is set to 0; never met, its client gone before its pass, it is left and overwritten by the next yield.
    double swap_ms_ = 0, changed_ms_ = 0;
    Clock::time_point turned_{};
    uint64_t turn_to_ = 0;
    std::atomic<size_t> yields_{0};   // pauses that gave the pool's edge to a paused request
    std::atomic<size_t> turns_without_memory_{0};   // Stats::turns_without_memory
    size_t memory_run_ = 0;           // the scheduler thread's: turns kept for want of host memory since the last one given up, said as the run starts and ends
    Clock::time_point memory_since_{};
    uint64_t pauses_ = 0;       // under the lock
    size_t stalls_ = 0;         // under the lock
    size_t waits_ = 0;          // under the lock
    size_t recomputed_ = 0;     // under the lock
    bool stopping_ = false;
    size_t reply_gaps_ = 0;            // the scheduler thread's: passes of generated rows alone retired since background work's last piece beside replies, which its next waits for (kBackgroundGaps)
    enum class Background { job, write };
    Background background_last_ = Background::job;   // the scheduler thread's: the kind of background work that took the last turn, which yields the next where the other waits
    bool write_waits_ = false;         // the scheduler thread's: a turn-end write beyond the host tier waited for its turn this round (Cache::write_held)
    double prompt_ms_row_ = 0;         // the scheduler thread's, a pass's milliseconds a prompt row, measured; 0 until a pass has
    double prompt_pass_ms_ = 0;        // the scheduler thread's, the last pass with prompt rows that was fed whole, its milliseconds whatever its size
    static constexpr size_t kMeasuredRows = 64;   // the prompt rows a pass needs for its time to give a row's
    Cache cache_{model_, m_, pools_, max_seqs_, host_cap_, limit_, wait_bound_, read_wait_, recompute_row_ms_, epoch_, [this](bool generated) { hold(generated); }, [this] { release(); }};
    // The waiting lines, as the cache's calls that search them take them.
    Cache::Lines lines() { return {queue_, paused_}; }
public:
    // The cache's handler of each loss (Cache::on_move), which tests set before run.
    std::function<void(const Moved&)>& on_move = cache_.on_move;
private:
    std::atomic<int> quieting_{0};        // the scheduler's thread waits for the recorders or holds them, possibly under the lock
    size_t normal_ = 0;                // the slots before the slots of passes apart
    LogitRows apart_rows_;             // the logits rows of the passes apart, after the others'
    bool acted_ = false;               // the scheduler thread's, whether this round advanced, collected, retired or formed anything
    bool encoded_ = false;             // and whether it ran a piece of an image's encode, which a round forming no pass may have done
    uint64_t jobs_posted_ = 0;           // the scheduler thread's, the hand-overs to recorders so far
    uint64_t arrivals_ = 0, seen_ = 0;   // under the lock, the requests and follows submitted, and how many the round had seen as it began
    std::vector<std::unique_ptr<StagePace>> pace_;   // per stage, its pacing state; before the recorders, which refer to it
    std::vector<std::unique_ptr<Recorder>> recorders_;   // per stage, one where Model::stage_waits; before the lanes, whose threads use them
    infer::StageLanes lanes_;   // the recorders' threads (Recorder::cv, quit); last, so they end before anything they record on
};

} // namespace server


